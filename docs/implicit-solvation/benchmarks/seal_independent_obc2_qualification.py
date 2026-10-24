#!/usr/bin/env python3
"""Seal validated candidate evidence; never run this before the frozen gates pass.

Run from the committed candidate checkout. Existing receipts (even identical
ones) are refused. This operator does not generate evidence, activate a wheel,
or change any scientific gate. Receipt publication precedes pointer publication.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for directory in (ROOT, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import validate_independent_obc2_default as validation  # noqa: E402
from maple.function.calculator import set_calculator as runtime  # noqa: E402
from maple.function.calculator.extra_correction.implicit.obc2_profile import (  # noqa: E402
    OBC2_CANONICAL_PAYLOAD_SHA256,
    OBC2_FORMULA_MANIFEST_ID,
    OPENMM_CUSTOMGBFORCES_SHA256,
)

RECEIPT_REL = (
    "maple/function/calculator/data/independent_obc2_analytic_qualification.json"
)
POINTER_REL = "maple/function/calculator/_independent_obc2_qualified_cell.py"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_evidence(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    artifact = json.loads(raw)
    if not isinstance(artifact, dict):
        raise ValueError(f"Evidence must be a JSON object: {path}")
    binding = {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": artifact.get("content_sha256"),
    }
    return artifact, binding


def _unqualified_pointer(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("Qualification pointer cannot be a symlink")
    raw = path.read_bytes()
    tree = ast.parse(raw)
    assignments = {}
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id in assignments
        ):
            raise ValueError("Qualification pointer must contain only two data assignments")
        assignments[statement.targets[0].id] = ast.literal_eval(statement.value)
    if assignments != {
        "QUALIFICATION_RECEIPT_FILENAME": Path(RECEIPT_REL).name,
        "QUALIFICATION_RECEIPT_SHA256": None,
    }:
        raise ValueError("Existing qualification pointer is not unqualified; refusing overwrite")
    return raw


def _qualified_cells() -> list[dict]:
    return [
        {
            "gas": {
                "model": "ani2x",
                "checkpoint_sha256": runtime._ANI2X_QUALIFIED_CHECKPOINT_SHA256,
                "device": "cpu",
                "dtype": "float64",
                "d4": False,
            },
            "solvent": {
                "provider": "torch",
                "runtime_provider": "torch-obc2",
                "model": "obc2",
                "profile": "obc2-mbondi2",
                "nonpolar": nonpolar,
                "formula_manifest_id": OBC2_FORMULA_MANIFEST_ID,
                "formula_source_sha256": OPENMM_CUSTOMGBFORCES_SHA256,
                "canonical_payload_sha256": OBC2_CANONICAL_PAYLOAD_SHA256,
                "parameter_profile": "obc2-zero-salt-v1",
            },
        }
        for nonpolar in ("ace", "none")
    ]


def build_receipt(validation_path: Path, candidate_path: Path) -> dict:
    """Validate raw inputs and current source before constructing any output."""
    artifact, artifact_binding = _read_evidence(validation_path)
    candidate, candidate_binding = _read_evidence(candidate_path)
    validation.validate_artifact(artifact)
    if artifact.get("passed") is not True:
        raise ValueError("Full candidate validation did not pass")
    frozen = artifact["source_snapshot"]
    validation.validate_candidate_cell_evidence(
        candidate, expected_source_snapshot=frozen
    )
    if candidate != artifact.get("candidate_exact_cell"):
        raise ValueError("External and embedded candidate exact-cell evidence differ")
    validation.validate_current_source_binding(frozen)

    gates = artifact["qualification_gates"]
    if set(gates) != set(runtime._QUALIFICATION_GATE_COUNTS):
        raise ValueError("Qualification gate set does not match runtime contract")
    for name, (expected, passed) in runtime._QUALIFICATION_GATE_COUNTS.items():
        gate = gates[name]
        if (
            type(gate.get("expected_count")) is not int
            or type(gate.get("passed_count")) is not int
            or gate["expected_count"] != expected
            or gate["passed_count"] != passed
            or gate.get("passed") is not True
        ):
            raise ValueError(f"Qualification gate denominator changed: {name}")

    installed = runtime._installed_runtime_source_inventory()
    sources = frozen["files_sha256"]
    if not installed or any(sources.get(path) != sha for path, sha in installed.items()):
        raise ValueError("Installed runtime inventory is not covered by frozen sources")
    receipt = {
        "schema_version": 1,
        "artifact_type": "maple-independent-obc2-analytic-qualification",
        "status": "qualified",
        "candidate_git_sha": candidate["candidate_git_sha"],
        "qualification_source_aggregate_sha256": frozen["files_aggregate_sha256"],
        "installed_runtime_sources": installed,
        "qualification_sources": sources,
        "qualified_cells": _qualified_cells(),
        "qualification_gates": gates,
        "evidence": {
            "candidate_validation": artifact_binding,
            "candidate_exact_cell": candidate_binding,
            "candidate_harness_sha256": candidate["harness_sha256"],
        },
    }
    receipt["content_sha256"] = _sha256(runtime._canonical_receipt_payload(receipt))
    return receipt


def _publish(path: Path, raw: bytes, *, replace: bool) -> None:
    """Publish complete bytes atomically; a new receipt is never overwritten."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # An atomic exclusive install also closes the concurrent-sealer race.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seal_qualification(validation_path: Path, candidate_path: Path) -> dict:
    receipt_path = ROOT / RECEIPT_REL
    pointer_path = ROOT / POINTER_REL
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"Existing sealed receipt is immutable: {receipt_path}")
    original_pointer = _unqualified_pointer(pointer_path)
    receipt = build_receipt(validation_path, candidate_path)
    receipt_raw = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    pointer_raw = (
        f'QUALIFICATION_RECEIPT_FILENAME = "{receipt_path.name}"\n'
        f'QUALIFICATION_RECEIPT_SHA256 = "{_sha256(receipt_raw)}"\n'
    ).encode("utf-8")
    # Check again immediately before publication; no staging file enters inventory.
    validation.validate_current_source_binding(
        {"git_head": receipt["candidate_git_sha"],
         "files_sha256": receipt["qualification_sources"],
         "files_aggregate_sha256": receipt["qualification_source_aggregate_sha256"]}
    )
    if pointer_path.read_bytes() != original_pointer:
        raise ValueError("Qualification pointer changed during validation")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _publish(receipt_path, receipt_raw, replace=False)
    if receipt_path.read_bytes() != receipt_raw:
        raise ValueError("Published receipt bytes changed; pointer remains unqualified")
    if pointer_path.read_bytes() != original_pointer:
        raise ValueError("Qualification pointer changed before publication")
    _publish(pointer_path, pointer_raw, replace=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation", type=Path, default=ROOT / validation.OUTPUT_REL
    )
    parser.add_argument(
        "--candidate", type=Path, default=ROOT / validation.CANDIDATE_CELL_EVIDENCE_REL
    )
    args = parser.parse_args()
    receipt = seal_qualification(args.validation, args.candidate)
    print(json.dumps({
        "receipt": RECEIPT_REL, "pointer": POINTER_REL,
        "content_sha256": receipt["content_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
