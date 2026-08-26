# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.metrics.waveform_tolerance import compare_waveforms, tolerance_failures


def _check(comparison, **overrides):
    thresholds = {
        "min_snr_db": 40.0,
        "max_peak_diff": 0.2,
        "max_exceed_fraction": 0.01,
    }
    thresholds.update(overrides)
    return tolerance_failures(comparison, **thresholds)


def test_identical_waveforms_pass() -> None:
    wave = np.linspace(-1.0, 1.0, 1000)
    comparison = compare_waveforms(wave, wave.copy())
    assert comparison.snr_db == math.inf
    assert comparison.max_abs_diff == 0.0
    assert comparison.exceed_fraction == 0.0
    assert not comparison.length_mismatch
    assert _check(comparison) == []


def test_uniform_offset_has_exact_snr() -> None:
    reference = np.ones(1000)
    comparison = compare_waveforms(reference, reference + 0.01, diff_threshold=0.02)
    assert comparison.snr_db == pytest.approx(40.0, rel=1e-9)
    assert comparison.max_abs_diff == pytest.approx(0.01, rel=1e-9)
    assert comparison.exceed_fraction == 0.0


def test_exceed_fraction_counts_only_over_threshold() -> None:
    reference = np.ones(1000)
    candidate = reference.copy()
    candidate[:3] += 0.5
    comparison = compare_waveforms(reference, candidate, diff_threshold=0.1)
    assert comparison.exceed_fraction == pytest.approx(0.003, rel=1e-9)
    assert comparison.max_abs_diff == pytest.approx(0.5, rel=1e-9)


def test_length_mismatch_fails_but_reports_prefix_metrics() -> None:
    reference = np.ones(1000)
    comparison = compare_waveforms(reference, np.ones(999))
    assert comparison.length_mismatch
    assert comparison.num_samples == 999
    assert comparison.snr_db == math.inf
    assert _check(comparison) == ["waveform lengths differ"]


def test_silent_reference_with_noise_fails() -> None:
    comparison = compare_waveforms(np.zeros(100), np.full(100, 1e-3))
    assert comparison.snr_db == -math.inf
    assert any("snr" in reason for reason in _check(comparison))


def test_empty_waveforms_pass() -> None:
    comparison = compare_waveforms(np.array([]), np.array([]))
    assert comparison.num_samples == 0
    assert _check(comparison) == []


def test_thresholds_produce_expected_reasons() -> None:
    reference = np.ones(1000)
    candidate = reference.copy()
    candidate[:500] += 0.3
    comparison = compare_waveforms(reference, candidate)
    reasons = _check(comparison)
    assert len(reasons) == 3
    assert any("snr" in reason for reason in reasons)
    assert any("peak diff" in reason for reason in reasons)
    assert any("of samples over" in reason for reason in reasons)
