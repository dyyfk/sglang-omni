# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"
TTS_WORKFLOW = REPO_ROOT / ".github/workflows/test-tts-ci.yaml"
SELECTION_SCRIPT = REPO_ROOT / ".github/scripts/tts_ci_selection.py"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def _load_selection_module():
    assert SELECTION_SCRIPT.is_file(), "CPU TTS selection helper is missing"
    spec = importlib.util.spec_from_file_location("tts_ci_selection", SELECTION_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_keeps_ordinary_stage_and_adds_isolated_mps_validation() -> None:
    workflow = _workflow(TTS_WORKFLOW)
    jobs = workflow["jobs"]
    ordinary = jobs["stage-1-non-streaming"]
    mps = jobs["stage-5-mps"]
    assert mps["name"] == "stage 5 - MPS"

    ordinary_run = _step(ordinary, "Run TTS non-streaming benchmark stage")
    assert "test_tts_ci.py" in ordinary_run["run"]
    assert "TTS_MPS_CONFIG" not in ordinary_run.get("env", {})
    assert ordinary["container"]["options"].find("--cap-add SYS_NICE") == -1
    assert (
        _step(ordinary, "Upload non-streaming speed artifact")["with"]["name"]
        == "tts-stage-nonstream-speed-results"
    )

    assert mps["needs"] == "stage-4-serving"
    assert "always()" in mps["if"]
    assert (
        "test_tts_mps_ci.py"
        in _step(mps, "Run TTS MPS non-streaming validation")["run"]
    )
    assert "--cap-add SYS_NICE" in mps["container"]["options"]
    assert (
        _step(mps, "Upload TTS MPS evidence")["with"]["name"]
        == "tts-mps-${{ inputs.tts_ci_model }}-attempt-${{ github.run_attempt }}"
    )
    assert "${{ env.OMNI_CI_HOME }}" not in mps["env"]["TTS_MPS_OUTPUT_ROOT"]
    assert "${{ env.OMNI_CI_HOME }}" not in mps["env"]["TTS_MPS_STATE_ROOT"]
    assert mps["env"]["TTS_MPS_OUTPUT_ROOT"].startswith("${{ inputs.omni_ci_home")
    assert mps["env"]["TTS_MPS_STATE_ROOT"].startswith("${{ inputs.omni_ci_home")
    assert "run-${{ github.run_id }}" not in mps["env"]["TTS_MPS_STATE_ROOT"]
    assert "TTS_MPS_BASE_PORT" not in mps["env"]

    assert jobs["stage-2-streaming"]["needs"] == "stage-1-non-streaming"
    assert jobs["stage-3-consistency"]["needs"] == "stage-2-streaming"
    assert jobs["stage-4-serving"]["needs"] == "stage-3-consistency"


def test_mps_artifacts_cannot_be_consumed_by_canonical_consistency() -> None:
    workflow = _workflow(TTS_WORKFLOW)
    jobs = workflow["jobs"]
    mps = jobs["stage-5-mps"]
    mps_env = mps["env"]
    assert "/mps-nonstream/" in mps_env["TTS_MPS_OUTPUT_ROOT"]
    assert "/nonstream/" not in mps_env["TTS_MPS_OUTPUT_ROOT"]

    canonical_jobs = [
        jobs["stage-2-streaming"],
        jobs["stage-3-consistency"],
        jobs["stage-4-serving"],
    ]
    canonical_text = yaml.safe_dump(canonical_jobs)
    assert "tts-mps-" not in canonical_text
    assert "mps-nonstream" not in canonical_text


def test_cpu_selection_is_rerun_stable_and_conflicts_fail_before_h100() -> None:
    module = _load_selection_module()
    run_id = "30317997846"
    expected_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    expected_model = ("higgs", "moss")[int(expected_digest, 16) % 2]

    selected = module.select_tts_model(
        run_id=run_id,
        labels=[],
        override="",
        run_attempt="1",
    )
    rerun = module.select_tts_model(
        run_id=run_id,
        labels=[],
        override="",
        run_attempt="9",
    )
    assert selected.model == expected_model
    assert selected.selection_digest == expected_digest
    assert rerun == selected

    with pytest.raises(ValueError, match="both run-higgs and run-moss"):
        module.select_tts_model(
            run_id=run_id,
            labels=["run-higgs", "run-moss"],
            override="",
            run_attempt="1",
        )

    omni = _workflow(OMNI_WORKFLOW)
    assert "tts_stage1_topology" not in omni["on"]["workflow_dispatch"]["inputs"]
    assert "pick-tts-model" not in omni["jobs"]
    assert "selected_model" in omni["jobs"]["preflight"]["outputs"]


def test_mps_config_resolution_covers_both_selected_models() -> None:
    module = _load_selection_module()
    assert (
        module.resolve_mps_config("higgs")
        .as_posix()
        .endswith("examples/mps_dp/configs/higgs_h100_dp2.yaml")
    )
    assert (
        module.resolve_mps_config("moss")
        .as_posix()
        .endswith("examples/mps_dp/configs/moss_local_h100_dp2.yaml")
    )


def test_cpu_preflight_uses_only_standard_library_dependencies(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SELECTION_SCRIPT),
            "--run-id",
            "123",
            "--run-attempt",
            "1",
            "--labels-json",
            "[]",
            "--override",
            "higgs",
            "--exact-sha",
            "a" * 40,
            "--workflow-url",
            "https://example.invalid/run/123",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
