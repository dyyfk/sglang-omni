# SPDX-License-Identifier: Apache-2.0
"""Select one TTS CI model before any H100 job is scheduled."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIGS = {
    "higgs": (
        "HiggsTtsPipelineConfig",
        Path("examples/mps_dp/configs/higgs_h100_dp2.yaml"),
    ),
    "moss": (
        "MossTTSLocalPipelineConfig",
        Path("examples/mps_dp/configs/moss_local_h100_dp2.yaml"),
    ),
}


@dataclass(frozen=True)
class Selection:
    model: str
    selection_digest: str
    selection_reason: str


def select_tts_model(
    *,
    run_id: str,
    labels: list[str],
    override: str,
    run_attempt: str,
) -> Selection:
    del run_attempt
    normalized_override = override.strip().lower()
    known_labels = set(labels)
    if {"run-higgs", "run-moss"} <= known_labels:
        raise ValueError("both run-higgs and run-moss labels are present")
    if normalized_override:
        if normalized_override not in MODEL_CONFIGS:
            raise ValueError(
                f"unsupported tts_ci_model={override!r}; expected higgs or moss"
            )
        return Selection(normalized_override, "", "workflow_dispatch_override")
    if "run-higgs" in known_labels:
        return Selection("higgs", "", "run-higgs_label")
    if "run-moss" in known_labels:
        return Selection("moss", "", "run-moss_label")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    model = ("higgs", "moss")[int(digest, 16) % 2]
    return Selection(model, digest, "stable_github_run_id_digest")


def resolve_mps_config(model: str) -> Path:
    try:
        expected_config_cls, relative = MODEL_CONFIGS[model]
    except KeyError as exc:
        raise ValueError(f"unsupported TTS model {model!r}") from exc
    path = REPO_ROOT / relative
    if not path.is_file():
        raise ValueError(f"MPS config is missing for {model}: {relative}")
    config_cls = next(
        (
            line.partition(":")[2].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("config_cls:")
        ),
        None,
    )
    if config_cls != expected_config_cls:
        raise ValueError(
            f"{relative} config_cls={config_cls!r}; "
            f"expected {expected_config_cls!r}"
        )
    return relative


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--override", default="")
    parser.add_argument("--exact-sha", required=True)
    parser.add_argument("--workflow-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    labels = json.loads(args.labels_json)
    if not isinstance(labels, list) or not all(
        isinstance(item, str) for item in labels
    ):
        parser.error("--labels-json must contain a JSON string array")
    try:
        selection = select_tts_model(
            run_id=args.run_id,
            labels=labels,
            override=args.override,
            run_attempt=args.run_attempt,
        )
        config = resolve_mps_config(selection.model)
    except ValueError as exc:
        parser.error(str(exc))

    payload = {
        "schema_version": 1,
        "exact_sha": args.exact_sha,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "workflow_url": args.workflow_url,
        "selected_model": selection.model,
        "selection_digest": selection.selection_digest,
        "selection_reason": selection.selection_reason,
        "resolved_mps_config": config.as_posix(),
    }
    _atomic_write(args.output, payload)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"selected_model={selection.model}\n")
            stream.write(f"selection_digest={selection.selection_digest}\n")
            stream.write(f"resolved_mps_config={config.as_posix()}\n")
    print(json.dumps(asdict(selection), sort_keys=True))


if __name__ == "__main__":
    main()
