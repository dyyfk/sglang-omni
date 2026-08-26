# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from benchmarks.eval.benchmark_code2wav_batching import (
    build_model_context,
    parse_args,
    run_equivalence,
    run_single,
)

_FAST = [
    "--fake",
    "--streams",
    "4",
    "--windows",
    "4",
    "--stream-chunk-size",
    "2",
    "--left-context-size",
    "1",
    "--timeout-s",
    "30",
]


def _run(arm: str, mode: str, *, wait_ms=None, floor=None):
    args = parse_args(_FAST + ["--arms", arm, "--modes", mode])
    ctx = build_model_context(args)
    if args.frame_interval_ms is None:
        args.frame_interval_ms = 1.0
    return run_single(
        ctx,
        args,
        arm=arm,
        streams=4,
        mode=mode,
        wait_ms=wait_ms,
        floor=floor,
        repeat=0,
    )


def test_serial_eager_aligned_run_completes() -> None:
    record = _run("serial-eager", "aligned")
    assert "invalid" not in record, record
    assert record["forward_calls"] > 0
    assert record["single_request_forward_share"] == 1.0
    assert record["ttfa_p50_s"] is not None
    assert record["xrt"] is not None
    assert record["chunks_per_request_mean"] > 0
    assert record["audio_s_total"] > 0


def test_batched_staggered_run_completes() -> None:
    record = _run("batched", "staggered", wait_ms=5, floor=2)
    assert "invalid" not in record, record
    assert record["forward_calls"] > 0
    assert record["single_request_forward_share"] is not None
    assert sum(record["fire_reasons"].values()) > 0
    assert record["errors"] == {}


def test_equivalence_lockstep_batches_and_matches_on_fake() -> None:
    args = parse_args(_FAST + ["--arms", "batched", "--modes", "aligned"])
    ctx = build_model_context(args)
    report = run_equivalence(ctx, args, streams=4)
    assert report["max_attained_batch"] >= 2
    assert report["failures"] == {}
    for entry in report["per_request"].values():
        assert entry["snr_db"] == math.inf
        assert not entry["length_mismatch"]
