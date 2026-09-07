# SPDX-License-Identifier: Apache-2.0
"""Streaming Code2Wav scheduler for MiniCPM-o.

The talker streams every generated codec token here (one ``stream`` message
per decode step, ``metadata={"stream": <client streaming?>}``), so vocoding
overlaps talker generation for every speech request. The chunk metadata's
``stream`` flag only controls delivery shape:

- streaming clients get incremental audio chunks plus a slim terminal result
- non-streaming clients get one terminal result carrying the full waveform

Chunking follows the checkpoint's reference streaming loop: the token buffer
is seeded with 3 silence tokens, each flow call consumes 25 tokens plus 3
lookahead tokens, and the buffer advances by 25. The stream-done flush feeds
the remainder with ``last_chunk=True``.

If a request reaches stream-done without any streamed codes (or the streamed
count disagrees with the talker's terminal payload), the scheduler falls back
to the one-shot whole-utterance vocode so the terminal result never degrades.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

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


@dataclass
class MiniCPMOCode2WavStreamState:
    buffer: list[int] = field(
        default_factory=lambda: [STREAM_SILENCE_TOKEN] * STREAM_SILENCE_PREFIX
    )
    vocoder: dict | None = None
    audio_parts: list[np.ndarray] = field(default_factory=list)
    stream_enabled: bool | None = None
    total_codes: int = 0
    decode_started: bool = False
    first_audio_emitted: bool = False


class MiniCPMOCode2WavScheduler(
    StreamingVocoderBase[MiniCPMOCode2WavStreamState, None]
):
    """Chunked stepaudio2 vocoding with per-request flow/hift caches."""

    def __init__(self, model: MiniCPMOCode2Wav) -> None:
        self._model = model
        super().__init__(
            None,
            sample_rate=OUTPUT_SAMPLE_RATE,
            stream_source_hint="MiniCPM-o",
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        # Every request rides the streaming machinery; the chunk metadata's
        # ``stream`` flag decides the delivery shape.
        del payload
        return True

    def create_stream_state(self, request_id: str) -> MiniCPMOCode2WavStreamState:
        del request_id
        return MiniCPMOCode2WavStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: MiniCPMOCode2WavStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        if origin != "stream metadata":
            return
        if state.stream_enabled is None:
            state.stream_enabled = bool(source["stream"])

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
        return len(state.buffer) >= _STREAM_WINDOW

    def decode_delta(
        self,
        request_id: str,
        state: MiniCPMOCode2WavStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
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
        if is_final and state.decode_started:
            self._emit_decode_end(request_id, state, status="ok")
        if not pieces:
            return None
        waveform = np.concatenate(pieces)
        state.audio_parts.append(waveform)
        if not state.stream_enabled:
            # Non-streaming client: accumulate only; the terminal result
            # carries the full waveform.
            return None
        if not state.first_audio_emitted:
            state.first_audio_emitted = True
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(waveform.shape[0])},
            )
        return torch.from_numpy(waveform)

    def _stream_step(
        self,
        request_id: str,
        state: MiniCPMOCode2WavStreamState,
        tokens: list[int],
        *,
        last_chunk: bool,
    ) -> np.ndarray:
        if not state.decode_started:
            state.decode_started = True
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_start",
                metadata={"streaming": True},
            )
        if state.vocoder is None:
            state.vocoder = self._model.new_stream_state()
        return self._model.stream_step(state.vocoder, tokens, last_chunk=last_chunk)

    def _emit_decode_end(
        self, request_id: str, state: MiniCPMOCode2WavStreamState, *, status: str
    ) -> None:
        samples = int(sum(part.shape[0] for part in state.audio_parts))
        _emit_event(
            request_id=request_id,
            stage=None,
            event_name="code2wav_decode_end",
            metadata={
                "codec_tokens": state.total_codes,
                "status": status,
                "audio_samples": samples,
                "audio_seconds": samples / OUTPUT_SAMPLE_RATE,
                "streaming": True,
            },
        )

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: MiniCPMOCode2WavStreamState,
    ) -> torch.Tensor | None:
        # Nothing was streamed. Streaming clients still need audio on the
        # wire, so vocode the terminal payload's codes as one chunk;
        # non-streaming clients get theirs from final_result_data.
        if not state.stream_enabled:
            return None
        waveform = self._one_shot_vocode(payload)
        if waveform.shape[0] == 0:
            return None
        state.audio_parts.append(waveform)
        return torch.from_numpy(waveform)

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: MiniCPMOCode2WavStreamState,
    ) -> dict[str, Any]:
        if state.stream_enabled:
            return {"modality": "audio", "sample_rate": self._sample_rate}
        expected = self._payload_codec_count(payload)
        if state.audio_parts and state.total_codes != expected:
            logger.warning(
                "MiniCPM-o code2wav streamed %d codes for %s but the talker "
                "payload carries %d; re-vocoding the full utterance",
                state.total_codes,
                request_id,
                expected,
            )
            state.audio_parts = []
        if state.audio_parts:
            waveform = np.concatenate(state.audio_parts)
        else:
            waveform = self._one_shot_vocode(payload)
        if waveform.shape[0]:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(waveform.shape[0])},
            )
        return dict(
            audio_waveform_payload(
                waveform,
                sample_rate=self._sample_rate,
                modality="audio",
                source_hint="MiniCPM-o",
            )
        )

    def release_stream_resources(
        self, request_id: str, state: MiniCPMOCode2WavStreamState
    ) -> None:
        del request_id
        state.vocoder = None
        state.audio_parts = []
        state.buffer = []

    def _payload_codec_count(self, payload: StagePayload) -> int:
        codec_tokens = self._payload_codec_tokens(payload)
        return int(codec_tokens.numel()) if codec_tokens is not None else 0

    @staticmethod
    def _payload_codec_tokens(payload: StagePayload) -> torch.Tensor | None:
        from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
        from sglang_omni.models.minicpm_o.request_builders import TALKER_STAGE

        state = MiniCPMOPipelineState.from_dict(payload.data)
        talker_out = state.engine_outputs.get(TALKER_STAGE) or {}
        return talker_out.get("codec_tokens")

    def _one_shot_vocode(self, payload: StagePayload) -> np.ndarray:
        """Whole-utterance vocode with the one-shot profiler events."""
        from sglang_omni.models.minicpm_o.stages import _run_code2wav_payload

        result_payload = _run_code2wav_payload(payload, model=self._model)
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
