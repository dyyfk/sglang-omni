# SPDX-License-Identifier: Apache-2.0
"""Streaming Code2Wav scheduler for MiniCPM-o.

Streaming clients get incremental audio: the talker streams their codec
tokens here (one ``stream`` message per decode step) and each 25-token chunk
is vocoded as it arrives, overlapping talker generation. Non-streaming
requests keep the one-shot whole-utterance vocode — chunked flow inference
costs roughly twice the GPU time of a single pass, so it is only paid where
it buys first-audio latency.

Chunking follows the checkpoint's reference streaming loop: the token buffer
is seeded with 3 silence tokens, each flow call consumes 25 tokens plus 3
lookahead tokens, and the buffer advances by 25. The stream-done flush feeds
the remainder with ``last_chunk=True``.

Each live stream holds a cloned Token2wav flow/hift cache on the GPU, so
concurrent streams are capped: requests that cannot get a slot keep buffering
tokens (cheap) and fall back to the one-shot vocode at stream-done, delivered
as a single audio chunk. Same fallback if no codes were streamed at all — the
audio never degrades, only its latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase

from .code2wav import (
    OUTPUT_SAMPLE_RATE,
    STREAM_CHUNK_SIZE,
    STREAM_PRE_LOOKAHEAD,
    STREAM_SILENCE_PREFIX,
    STREAM_SILENCE_TOKEN,
    MiniCPMOCode2Wav,
)

logger = logging.getLogger(__name__)

_STREAM_WINDOW = STREAM_CHUNK_SIZE + STREAM_PRE_LOOKAHEAD

# Each live stream pins a cloned flow/hift cache (hundreds of MiB) in a
# process that shares the GPU with the thinker and talker engines; six
# concurrent streams OOMed an H100 (run minicpmo-stream-ab-20260907-040720).
MAX_ACTIVE_STREAMS = 2


@dataclass
class MiniCPMOCode2WavStreamState:
    buffer: list[int] = field(
        default_factory=lambda: [STREAM_SILENCE_TOKEN] * STREAM_SILENCE_PREFIX
    )
    vocoder: dict | None = None
    total_codes: int = 0
    emitted_samples: int = 0
    first_audio_emitted: bool = False


class MiniCPMOCode2WavScheduler(
    StreamingVocoderBase[MiniCPMOCode2WavStreamState, None]
):
    """Chunked stepaudio2 vocoding with per-request flow/hift caches."""

    def __init__(
        self, model: MiniCPMOCode2Wav, *, max_active_streams: int = MAX_ACTIVE_STREAMS
    ) -> None:
        self._model = model
        self._max_active_streams = int(max_active_streams)
        self._active_streams = 0
        super().__init__(
            self._compute_one_shot,
            sample_rate=OUTPUT_SAMPLE_RATE,
            stream_source_hint="MiniCPM-o",
        )

    def _compute_one_shot(self, payload: StagePayload) -> StagePayload:
        from sglang_omni.models.minicpm_o.stages import _run_code2wav_payload

        return _run_code2wav_payload(payload, model=self._model)

    def create_stream_state(self, request_id: str) -> MiniCPMOCode2WavStreamState:
        del request_id
        return MiniCPMOCode2WavStreamState()

    def validate_chunk(
        self, request_id: str, state: MiniCPMOCode2WavStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        return codes.reshape(-1)

    def ingest(
        self, request_id: str, state: MiniCPMOCode2WavStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        tokens = codes.tolist()
        state.buffer.extend(int(token) for token in tokens)
        state.total_codes += len(tokens)

    def should_decode(
        self, state: MiniCPMOCode2WavStreamState, *, is_final: bool
    ) -> bool:
        del is_final
        if len(state.buffer) < _STREAM_WINDOW:
            return False
        return (
            state.vocoder is not None or self._active_streams < self._max_active_streams
        )

    def decode_delta(
        self,
        request_id: str,
        state: MiniCPMOCode2WavStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        if state.vocoder is None:
            # No live decode session. Acquiring one at stream-done would just
            # be a costlier one-shot (all audio lands at once either way), so
            # short or slot-starved streams take the one-shot fallback.
            if is_final or not self._acquire_stream_slot(request_id, state):
                return None
        pieces: list[np.ndarray] = []
        while len(state.buffer) >= _STREAM_WINDOW:
            pieces.append(
                self._stream_step(
                    request_id, state, state.buffer[:_STREAM_WINDOW], last_chunk=False
                )
            )
            del state.buffer[:STREAM_CHUNK_SIZE]
        if is_final and state.total_codes > 0 and state.buffer:
            pieces.append(
                self._stream_step(
                    request_id, state, list(state.buffer), last_chunk=True
                )
            )
            state.buffer.clear()
        if not pieces:
            return None
        waveform = np.concatenate(pieces)
        state.emitted_samples += int(waveform.shape[0])
        if is_final:
            self._emit_decode_end(request_id, state, status="ok")
        if not state.first_audio_emitted:
            state.first_audio_emitted = True
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(waveform.shape[0])},
            )
        return torch.from_numpy(waveform)

    def _acquire_stream_slot(
        self, request_id: str, state: MiniCPMOCode2WavStreamState
    ) -> bool:
        if self._active_streams >= self._max_active_streams:
            return False
        state.vocoder = self._model.new_stream_state()
        self._active_streams += 1
        _emit_event(
            request_id=request_id,
            stage=None,
            event_name="code2wav_decode_start",
            metadata={"streaming": True},
        )
        return True

    def _stream_step(
        self,
        request_id: str,
        state: MiniCPMOCode2WavStreamState,
        tokens: list[int],
        *,
        last_chunk: bool,
    ) -> np.ndarray:
        del request_id
        return self._model.stream_step(state.vocoder, tokens, last_chunk=last_chunk)

    def _emit_decode_end(
        self, request_id: str, state: MiniCPMOCode2WavStreamState, *, status: str
    ) -> None:
        _emit_event(
            request_id=request_id,
            stage=None,
            event_name="code2wav_decode_end",
            metadata={
                "codec_tokens": state.total_codes,
                "status": status,
                "audio_samples": state.emitted_samples,
                "audio_seconds": state.emitted_samples / OUTPUT_SAMPLE_RATE,
                "streaming": True,
            },
        )

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: MiniCPMOCode2WavStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        waveform = self._one_shot_waveform(payload)
        if waveform.shape[0] == 0:
            return None
        return torch.from_numpy(waveform)

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: MiniCPMOCode2WavStreamState,
    ) -> dict[str, Any]:
        if state.vocoder is not None and state.total_codes > 0:
            expected = self._payload_codec_count(payload)
            if state.total_codes != expected:
                logger.warning(
                    "MiniCPM-o code2wav streamed %d codes for %s but the "
                    "talker payload carries %d",
                    state.total_codes,
                    request_id,
                    expected,
                )
        return {"modality": "audio", "sample_rate": self._sample_rate}

    def release_stream_resources(
        self, request_id: str, state: MiniCPMOCode2WavStreamState
    ) -> None:
        del request_id
        if state.vocoder is not None:
            state.vocoder = None
            self._active_streams -= 1
        state.buffer = []

    @staticmethod
    def _payload_codec_count(payload: StagePayload) -> int:
        from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
        from sglang_omni.models.minicpm_o.request_builders import TALKER_STAGE

        state = MiniCPMOPipelineState.from_dict(payload.data)
        talker_out = state.engine_outputs.get(TALKER_STAGE) or {}
        codec_tokens = talker_out.get("codec_tokens")
        return int(codec_tokens.numel()) if codec_tokens is not None else 0

    def _one_shot_waveform(self, payload: StagePayload) -> np.ndarray:
        """Whole-utterance vocode with the one-shot profiler events."""
        result_payload = self._compute_one_shot(payload)
        data = result_payload.data
        # copy: frombuffer views are read-only, and the fallback path wraps
        # this array with torch.from_numpy.
        return np.frombuffer(data["audio_waveform"], dtype=np.float32).copy()


def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str | None = None,
) -> MiniCPMOCode2WavScheduler:
    from sglang_omni.utils.device import resolve_device_spec

    model = MiniCPMOCode2Wav(model_path, device=resolve_device_spec(device))
    return MiniCPMOCode2WavScheduler(model)
