#!/usr/bin/env python3
"""Publish the aggregate-only projection of a private hybrid MNSol-10 score."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._secure_artifacts import (  # noqa: E402
    CapturedFile,
    SecureArtifactError,
    StabilityGuard,
    capture_clean_repository,
    capture_file,
    capture_repo_files,
    load_json_bytes,
    publish_json_noreplace,
)
from tools.route2_release._hybrid_mnsol10_chain import (
    capture_prediction_chain,
)  # noqa: E402

RUN_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/runs"
SEAL_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/seals"
INPUT_BUNDLE = REPO_ROOT / ".omx/route2/hybrid-mnsol10/label-free-input-v1.json"
OUTPUT = REPO_ROOT / (
    "docs/route2/evidence/" "mace-mdp-polar-hybrid-mnsol10-fullsolv-accuracy-v1.json"
)


def _load_captured_module(capture: CapturedFile, *, module_name: str) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(capture.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(capture.data, str(capture.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def project(
    score_path: str | Path,
) -> tuple[dict[str, object], StabilityGuard]:
    repository = capture_clean_repository(REPO_ROOT)
    score = capture_file(score_path, role="private accuracy score")
    private = load_json_bytes(score.data, role="private accuracy score")
    prediction_binding = private.get("prediction")
    if not isinstance(prediction_binding, dict):
        raise SecureArtifactError("private score has no prediction binding")
    execution_id = prediction_binding.get("execution_id")
    if not isinstance(execution_id, str) or len(execution_id) != 64:
        raise SecureArtifactError("private score execution identity is invalid")
    preliminary_seal = capture_file(
        SEAL_DIRECTORY / f"{execution_id}.json",
        role="computation seal preflight",
    )
    seal_payload = load_json_bytes(
        preliminary_seal.data, role="computation seal preflight"
    )
    source_hashes = seal_payload.get("source_files_sha256")
    if not isinstance(source_hashes, dict):
        raise SecureArtifactError("seal source ledger is invalid")
    sources = capture_repo_files(repository, source_hashes.keys())
    accuracy = _load_captured_module(
        sources["maple/solvation/release/accuracy_admission.py"],
        module_name="_maple_route2_captured_accuracy_publisher",
    )
    expected = (RUN_DIRECTORY / execution_id / "private-score.json").resolve()
    if score.path != expected:
        raise SecureArtifactError("private score path is not canonical")
    prediction_path = RUN_DIRECTORY / execution_id / "prediction-terminal.json"
    chain = capture_prediction_chain(
        repo_root=REPO_ROOT,
        prediction_path=prediction_path,
        seal_directory=SEAL_DIRECTORY,
        run_directory=RUN_DIRECTORY,
        input_bundle_path=INPUT_BUNDLE,
        accuracy=accuracy,
    )
    accuracy.validate_scored_bundle(
        private,
        expected_record_count=10,
        expected_partition_counts={"confirmation": 8, "development": 2},
    )
    if prediction_binding != {
        "attempt_slot_id": chain.prediction["attempt_slot_id"],
        "execution_id": execution_id,
        "file_sha256": chain.prediction_capture.sha256,
        "content_sha256": chain.prediction["content_sha256"],
        "measurement_sha256": chain.prediction["measurement_sha256"],
    }:
        raise SecureArtifactError("private score prediction chain drifted")
    if {
        "artifact_id": private.get("artifact_id"),
        "profile_id": private.get("profile_id"),
        "scalar_id": private.get("scalar_id"),
        "claim_boundary": private.get("claim_boundary"),
    } != {
        "artifact_id": "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-score-v1",
        "profile_id": chain.prediction.get("profile_id"),
        "scalar_id": chain.prediction.get("scalar_id"),
        "claim_boundary": chain.prediction.get("claim_boundary"),
    }:
        raise SecureArtifactError("private score relabeled the prediction identity")
    scorer_source = chain.source_captures[
        "tools/route2_release/score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py"
    ]
    if private.get("scorer_source_sha256") != scorer_source.sha256:
        raise SecureArtifactError("private score does not bind the tracked scorer")
    stability = StabilityGuard(
        chain.repository,
        (*chain.stability.captures, ("private accuracy score", score)),
    )
    projection = accuracy.public_accuracy_projection(private)
    stability.assert_stable()
    return projection, stability


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        projection, stability = project(args.score)
        publish_json_noreplace(
            OUTPUT,
            projection,
            root=REPO_ROOT,
            stability=stability.assert_stable,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": projection["status"],
                "aggregate_metrics": projection["aggregate_metrics"],
                "gates": projection["gates"],
                "output": os.fspath(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if projection["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
