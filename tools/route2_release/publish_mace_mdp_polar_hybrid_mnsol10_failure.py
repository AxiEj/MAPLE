#!/usr/bin/env python3
"""Publish aggregate-only evidence for a failed/aborted MNSol-10 attempt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._hybrid_mnsol10_chain import (
    capture_prediction_chain,
)  # noqa: E402
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

SEAL_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/seals"
RUN_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/runs"
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
    terminal_path: str | Path,
) -> tuple[dict[str, object], StabilityGuard]:
    repository = capture_clean_repository(REPO_ROOT)
    preliminary = capture_file(terminal_path, role="prediction failure preflight")
    terminal = load_json_bytes(preliminary.data, role="prediction failure preflight")
    execution_id = terminal.get("execution_id")
    if not isinstance(execution_id, str) or len(execution_id) != 64:
        raise SecureArtifactError("prediction failure execution identity is invalid")
    seal_capture = capture_file(
        SEAL_DIRECTORY / f"{execution_id}.json", role="computation seal preflight"
    )
    seal = load_json_bytes(seal_capture.data, role="computation seal preflight")
    source_hashes = seal.get("source_files_sha256")
    if not isinstance(source_hashes, dict):
        raise SecureArtifactError("seal source ledger is invalid")
    sources = capture_repo_files(repository, source_hashes.keys())
    accuracy = _load_captured_module(
        sources["maple/solvation/release/accuracy_admission.py"],
        module_name="_maple_route2_captured_accuracy_failure_publisher",
    )
    chain = capture_prediction_chain(
        repo_root=REPO_ROOT,
        prediction_path=terminal_path,
        seal_directory=SEAL_DIRECTORY,
        run_directory=RUN_DIRECTORY,
        input_bundle_path=INPUT_BUNDLE,
        accuracy=accuracy,
    )
    if chain.prediction.get("status") != "failure":
        raise SecureArtifactError("success terminal requires the score publisher")
    projection = accuracy.prediction_failure_public_projection(
        chain.prediction,
        terminal_file_sha256=chain.prediction_capture.sha256,
    )
    chain.stability.assert_stable()
    return projection, chain.stability


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        projection, stability = project(args.terminal)
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
        f"published failed attempt_slot_id={projection['attempt_slot_id']} "
        f"terminal_type={projection['terminal_type']} to {os.fspath(OUTPUT)}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
