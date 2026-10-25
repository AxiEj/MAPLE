#!/usr/bin/env python3
"""Seal validated candidate evidence; never run this before the frozen gates pass.

Run from the committed candidate checkout. Existing receipts (even identical
ones) are refused. This operator does not generate evidence, activate a wheel,
or change any scientific gate. Receipt publication precedes pointer publication.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
import validate_aimnet2025_analytic as aimnet_validation  # noqa: E402
from maple.function.calculator import set_calculator as runtime  # noqa: E402


RECEIPT_REL = (
    "maple/function/calculator/data/independent_obc2_analytic_qualification_v2.json"
)
POINTER_REL = "maple/function/calculator/_independent_obc2_qualified_cell.py"
AIMNET2025_EVIDENCE_REL = ".omx/benchmarks/route1-aimnet2025-20261024/candidate-aimnet2025-v3.json"


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


def build_receipt(validation_path: Path, candidate_path: Path,
                  aimnet2025_path: Path | None = None) -> dict:
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
    if not runtime._same_json(candidate["candidate_source_snapshot"], frozen):
        raise ValueError("ANI and OBC evidence must use the same complete source freeze")

    aimnet, aimnet_binding = _read_evidence(
        ROOT / AIMNET2025_EVIDENCE_REL if aimnet2025_path is None else aimnet2025_path
    )
    # This entry point replays raw sections, full failure denominators, numerical
    # gates, runtime/checkpoint/input/source before/after binding. Assess-only is
    # deliberately insufficient: it also accepts completed failed attempts.
    aimnet_gates = aimnet_validation.validate_artifact(
        aimnet, expected_source_snapshot=frozen
    )
    for field in ("source_snapshot_before", "source_snapshot_after"):
        if not runtime._same_json(aimnet.get(field), frozen):
            raise ValueError("AIMNet2025 and OBC evidence must share the same source freeze")
    if aimnet.get("candidate_git_sha") != candidate["candidate_git_sha"]:
        raise ValueError("AIMNet2025 and ANI candidate Git identities differ")
    # The v2 snapshot binds positive inputs AND original negative-control
    # sources/control file. Compare/store it whole; never project to positives.
    if (not isinstance(aimnet.get("inputs_before"), dict) or not aimnet["inputs_before"]
            or not runtime._same_json(aimnet["inputs_before"], aimnet.get("inputs_after"))):
        raise ValueError("AIMNet2025 frozen inputs changed during qualification")
    expected_aimnet_gates = {
        name for name in runtime._QUALIFICATION_GATE_COUNTS
        if name.startswith("aimnet2025_")
    }
    if not isinstance(aimnet_gates, dict) or set(aimnet_gates) != expected_aimnet_gates:
        raise ValueError("Recomputed AIMNet2025 qualification gate set mismatch")
    profiles = aimnet.get("model_identities_before")
    if not runtime._same_json(profiles, aimnet.get("model_identities_after")):
        raise ValueError("AIMNet2025 model identities changed during qualification")
    cells = runtime._qualified_independent_obc2_cells(profiles)
    gates = deepcopy(artifact["qualification_gates"])
    if set(gates) & set(aimnet_gates):
        raise ValueError("AIMNet2025 gates cannot replace original OBC/ANI gates")
    gates.update(deepcopy(aimnet_gates))
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

    if gates["candidate_exact_cell_public_composition"].get(
        "evidence_content_sha256"
    ) != candidate["content_sha256"]:
        raise ValueError("Qualification gate candidate evidence hash changed")

    installed = runtime._installed_runtime_source_inventory()
    sources = frozen["files_sha256"]
    if not installed or any(sources.get(path) != sha for path, sha in installed.items()):
        raise ValueError("Installed runtime inventory is not covered by frozen sources")
    receipt = {
        "schema_version": 2,
        "artifact_type": "maple-independent-obc2-analytic-qualification",
        "status": "qualified",
        "candidate_git_sha": candidate["candidate_git_sha"],
        "qualification_source_aggregate_sha256": frozen["files_aggregate_sha256"],
        "installed_runtime_sources": installed,
        "qualification_sources": sources,
        "qualified_cells": cells,
        "aimnet2025_runtime_profiles": deepcopy(profiles),
        "qualification_gates": gates,
        "evidence": {
            "candidate_validation": artifact_binding,
            "candidate_exact_cell": candidate_binding,
            "aimnet2025_candidate": aimnet_binding,
            "aimnet2025_inputs": deepcopy(aimnet["inputs_before"]),
            "aimnet2025_harness_sha256": aimnet["harness_sha256"],
            "aimnet2025_validator_sha256": aimnet["validator_sha256"],
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seal_qualification(validation_path: Path, candidate_path: Path,
                       aimnet2025_path: Path | None = None) -> dict:
    receipt_path = ROOT / RECEIPT_REL
    pointer_path = ROOT / POINTER_REL
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"Existing sealed receipt is immutable: {receipt_path}")
    original_pointer = _unqualified_pointer(pointer_path)
    receipt = build_receipt(validation_path, candidate_path, aimnet2025_path)
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
    try:
        _publish(pointer_path, pointer_raw, replace=True)
    except Exception:
        # A failure after replace (e.g. directory fsync) must not leave an
        # apparently successful admission. Retain immutable receipt evidence;
        # restore only this operator's own pointer, never another writer's.
        if pointer_path.read_bytes() == pointer_raw:
            _publish(pointer_path, original_pointer, replace=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation", type=Path, default=ROOT / validation.OUTPUT_REL
    )
    parser.add_argument(
        "--candidate", type=Path, default=ROOT / validation.CANDIDATE_CELL_EVIDENCE_REL
    )
    parser.add_argument(
        "--aimnet2025", type=Path, default=ROOT / AIMNET2025_EVIDENCE_REL,
        help="Complete source-bound AIMNet2025 raw qualification evidence",
    )
    args = parser.parse_args()
    receipt = seal_qualification(args.validation, args.candidate, args.aimnet2025)
    print(json.dumps({
        "receipt": RECEIPT_REL, "pointer": POINTER_REL,
        "content_sha256": receipt["content_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
