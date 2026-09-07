# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o streaming code2wav: chunk windows, delivery shapes, fallbacks.

GPU-free: a fake vocoder records every ``stream_step`` window so the tests
can check the reference chunking contract (3-token silence prefix, 25-token
chunks with 3 lookahead tokens re-fed to the next window, ``last_chunk``
flush) without stepaudio2.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sglang_omni.models.minicpm_o.components.code2wav import (
    STREAM_CHUNK_SIZE,
    STREAM_PRE_LOOKAHEAD,
    STREAM_SILENCE_PREFIX,
    STREAM_SILENCE_TOKEN,
)
from sglang_omni.models.minicpm_o.components.code2wav_scheduler import (
    MiniCPMOCode2WavScheduler,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import (
    make_talker_stream_output_builder,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto import OmniRequest, StagePayload

SAMPLES_PER_TOKEN = 480


class _FakeVocoder:
    """Mimics MiniCPMOCode2Wav: 480 samples per fed token, windows recorded."""

    def __init__(self):
        self.stream_windows: list[tuple[list[int], bool]] = []
        self.one_shot_calls: list[int] = []

    def __call__(self, *, codec_tokens, **_):
        n = int(codec_tokens.numel())
        self.one_shot_calls.append(n)
        return {
            "waveform": np.zeros(n * SAMPLES_PER_TOKEN, dtype=np.float32),
            "sample_rate": 24000,
        }

    def new_stream_state(self) -> dict:
        return {"windows": 0}

    def stream_step(self, stream_state, tokens, *, last_chunk):
        stream_state["windows"] += 1
        self.stream_windows.append((list(tokens), bool(last_chunk)))
        return np.zeros(len(tokens) * SAMPLES_PER_TOKEN, dtype=np.float32)


def _payload(codec_count: int, *, stream: bool, request_id: str = "req-c2w"):
    state = MiniCPMOPipelineState(
        engine_outputs={
            "talker": {"codec_tokens": torch.arange(codec_count, dtype=torch.long)}
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={"stream": stream}, metadata={}),
        data=state.to_dict(),
    )


def _chunk(tokens: list[int], *, stream: bool, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=torch.tensor(tokens, dtype=torch.long),
        from_stage="talker",
        metadata={"stream": stream, "modality": "audio_codes"},
    )


def _drain_outbox(scheduler):
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def _drive(scheduler, request_id, payload, token_chunks, *, stream: bool):
    scheduler._on_streaming_new_request(request_id, payload)
    for idx, tokens in enumerate(token_chunks):
        scheduler._on_chunk(request_id, _chunk(tokens, stream=stream, chunk_id=idx))
    scheduler._on_done(request_id)
    return _drain_outbox(scheduler)


def test_stream_windows_follow_reference_chunking():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    tokens = list(range(60))
    messages = _drive(
        scheduler,
        "req-windows",
        _payload(60, stream=True, request_id="req-windows"),
        [tokens[:7], tokens[7:30], tokens[30:59], tokens[59:]],
        stream=True,
    )

    windows = vocoder.stream_windows
    assert windows, "no stream_step calls recorded"
    # First window opens with the silence prefix.
    assert windows[0][0][:STREAM_SILENCE_PREFIX] == (
        [STREAM_SILENCE_TOKEN] * STREAM_SILENCE_PREFIX
    )
    # Steady windows are chunk+lookahead wide and not last; only the flush is.
    for window, last in windows[:-1]:
        assert len(window) == STREAM_CHUNK_SIZE + STREAM_PRE_LOOKAHEAD
        assert last is False
    assert windows[-1][1] is True
    # Advancing by 25 re-feeds the 3 lookahead tokens: dropping each steady
    # window's lookahead tail and appending the flush reconstructs the full
    # token stream exactly once.
    consumed: list[int] = []
    for window, _ in windows[:-1]:
        consumed.extend(window[:STREAM_CHUNK_SIZE])
    consumed.extend(windows[-1][0])
    assert consumed == [STREAM_SILENCE_TOKEN] * STREAM_SILENCE_PREFIX + tokens

    stream_messages = [m for m in messages if m.type == "stream"]
    assert stream_messages, "streaming client got no audio chunks"
    for message in stream_messages:
        assert message.metadata == {"modality": "audio"}
        assert message.data["sample_rate"] == 24000
        assert message.data["audio_waveform_dtype"] == "float32"

    (result,) = [m for m in messages if m.type == "result"]
    assert result.data.data == {"modality": "audio", "sample_rate": 24000}
    assert not vocoder.one_shot_calls


def test_short_utterance_single_flush():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    messages = _drive(
        scheduler,
        "req-short",
        _payload(10, stream=True, request_id="req-short"),
        [list(range(10))],
        stream=True,
    )
    assert vocoder.stream_windows == [
        ([STREAM_SILENCE_TOKEN] * STREAM_SILENCE_PREFIX + list(range(10)), True)
    ]
    assert sum(1 for m in messages if m.type == "stream") == 1


def test_non_streaming_client_gets_full_waveform_result():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    messages = _drive(
        scheduler,
        "req-full",
        _payload(60, stream=False, request_id="req-full"),
        [list(range(30)), list(range(30, 60))],
        stream=False,
    )
    # Vocoding still ran chunked (overlaps talker generation)...
    assert len(vocoder.stream_windows) > 1
    assert not vocoder.one_shot_calls
    # ...but nothing streamed to the client; the result carries all audio.
    assert [m.type for m in messages] == ["result"]
    data = messages[0].data.data
    fed = sum(len(window) for window, _ in vocoder.stream_windows)
    assert data["audio_waveform_shape"] == [fed * SAMPLES_PER_TOKEN]
    assert data["sample_rate"] == 24000


def test_count_mismatch_revocodes_the_full_utterance():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    messages = _drive(
        scheduler,
        "req-mismatch",
        _payload(60, stream=False, request_id="req-mismatch"),
        [list(range(10))],  # 10 streamed vs 60 in the terminal payload
        stream=False,
    )
    assert vocoder.one_shot_calls == [60]
    data = messages[-1].data.data
    assert data["audio_waveform_shape"] == [60 * SAMPLES_PER_TOKEN]


def test_no_chunks_falls_back_to_one_shot():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    messages = _drive(
        scheduler,
        "req-oneshot",
        _payload(5, stream=False, request_id="req-oneshot"),
        [],
        stream=False,
    )
    assert vocoder.one_shot_calls == [5]
    assert not vocoder.stream_windows
    data = messages[-1].data.data
    assert data["audio_waveform_shape"] == [5 * SAMPLES_PER_TOKEN]


def test_empty_codec_stream_yields_empty_waveform_result():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    messages = _drive(
        scheduler,
        "req-empty",
        _payload(0, stream=False, request_id="req-empty"),
        [],
        stream=False,
    )
    data = messages[-1].data.data
    assert data["audio_waveform_shape"] == [0]
    assert data["sample_rate"] == 24000
    assert not vocoder.stream_windows


def test_chunks_and_done_before_payload():
    # Real arrival order: the talker streams while running, stream_done fires
    # at its completion, and the projected payload lands last.
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    tokens = list(range(30))
    scheduler._on_chunk("req-early", _chunk(tokens, stream=True))
    scheduler._on_done("req-early")
    scheduler._on_streaming_new_request(
        "req-early", _payload(30, stream=True, request_id="req-early")
    )
    messages = _drain_outbox(scheduler)
    assert vocoder.stream_windows[-1][1] is True
    assert [m.type for m in messages][-1] == "result"
    assert messages[-1].data.data == {"modality": "audio", "sample_rate": 24000}


def test_late_chunks_after_abort_are_dropped():
    vocoder = _FakeVocoder()
    scheduler = MiniCPMOCode2WavScheduler(vocoder)
    scheduler._on_streaming_new_request(
        "req-abort", _payload(30, stream=True, request_id="req-abort")
    )
    scheduler._on_chunk("req-abort", _chunk(list(range(30)), stream=True))
    scheduler.abort("req-abort")
    _drain_outbox(scheduler)
    windows_after_abort = len(vocoder.stream_windows)
    scheduler._on_chunk("req-abort", _chunk(list(range(30, 60)), stream=True))
    assert len(vocoder.stream_windows) == windows_after_abort
    assert not _drain_outbox(scheduler)


def test_streaming_events(tmp_path):
    recorder = get_recorder()
    assert not recorder.is_active()
    recorder.start("test-run", str(tmp_path), "speech")
    try:
        vocoder = _FakeVocoder()
        scheduler = MiniCPMOCode2WavScheduler(vocoder)
        _drive(
            scheduler,
            "req-ev",
            _payload(30, stream=True, request_id="req-ev"),
            [list(range(30))],
            stream=True,
        )
    finally:
        recorder.stop()
    events = []
    for path in sorted(tmp_path.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fp:
            events.extend(json.loads(line) for line in fp)
    by_name = {e["event_name"]: e for e in events}
    assert by_name["code2wav_decode_start"]["metadata"] == {"streaming": True}
    end = by_name["code2wav_decode_end"]["metadata"]
    assert end["codec_tokens"] == 30
    assert end["status"] == "ok"
    assert end["audio_samples"] > 0
    assert by_name["code2wav_first_audio"]["metadata"]["samples"] > 0


class _Req:
    def __init__(self, params):
        self.params = params
        self.metadata = {}


class _StagePayloadStub:
    def __init__(self, params):
        self.request = _Req(params)


class _ReqData:
    def __init__(self, *, params, empty_span=False):
        self.talker_model_inputs = {"empty_span": empty_span}
        self.stage_payload = _StagePayloadStub(params)


class _ReqOutput:
    def __init__(self, data):
        self.data = data


@pytest.mark.parametrize("client_streams", [True, False])
def test_talker_stream_builder_emits_every_code(client_streams):
    build = make_talker_stream_output_builder(codec_eos_id=6561)
    messages = build(
        "req-t", _ReqData(params={"stream": client_streams}), _ReqOutput(42)
    )
    (message,) = messages
    assert message.type == "stream"
    assert message.target == "code2wav"
    assert message.metadata == {"stream": client_streams, "modality": "audio_codes"}
    assert message.data.tolist() == [42]


def test_talker_stream_builder_filters_eos_empty_span_and_null():
    build = make_talker_stream_output_builder(codec_eos_id=6561)
    req_data = _ReqData(params={"stream": True})
    assert build("req-t", req_data, _ReqOutput(6561)) == []
    assert build("req-t", req_data, _ReqOutput(None)) == []
    assert (
        build(
            "req-t", _ReqData(params={"stream": True}, empty_span=True), _ReqOutput(7)
        )
        == []
    )
