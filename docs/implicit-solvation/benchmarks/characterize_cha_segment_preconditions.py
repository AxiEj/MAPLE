#!/usr/bin/env python3
"""Preregister, then characterize, exact coordinate-only CHA segment facts.

This diagnostic never evaluates continuum energies, R6 fluxes, Born radii,
weighted signs, or forces.  The two CLI phases make the complete 283-case
ordering and all source/input hashes immutable before any characterization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, NamedTuple

SCRIPT = Path(__file__).resolve()
SOURCE_ROOT = SCRIPT.parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_segment_preconditions import (
    SegmentPreconditionBudget,
    SegmentPreconditionResourceError,
    SegmentPreconditionValidationError,
    characterize_cha_segment_preconditions,
)

SCHEMA = "route1-cha-segment-preconditions-v1"
EVIDENCE_ROOT = SOURCE_ROOT.parents[2]
PROTECTED_MANIFEST = (
    EVIDENCE_ROOT
    / ".omx/benchmarks/route1-cha-analytic-v1-20260930/candidate-source-manifest-v2.json"
)
SOURCE_PATHS = (
    SCRIPT,
    SOURCE_ROOT
    / "maple/function/calculator/extra_correction/implicit/torch_continuum_chagb_segment_preconditions.py",
    SOURCE_ROOT
    / "maple/function/calculator/extra_correction/implicit/continuum_chagb_inputs.py",
    SOURCE_ROOT / "maple/function/calculator/extra_correction/implicit/torch_chagb.py",
)
EXPECTED_FOUNDATION_MANIFEST_SHA256 = (
    "221d99870e4d096157b38c461efccf84a1f41262b2610b30f7395b9838444493"
)
APPROVED_PLAN = EVIDENCE_ROOT / ".omx/plans/route1-cha-segment-safety-v1-20261001.md"
APPROVED_PLAN_SHA256 = (
    "bb0e3081e4ebe1c56b57f7a2db1987fbccd379a2e05349c06c9175d297b54bb6"
)
ACCEPTED_HANDOFF = (
    EVIDENCE_ROOT / ".omx/plans/route1-cha-segment-preconditions-v1-handoff.json"
)
ACCEPTED_HANDOFF_SHA256 = (
    "2f0c76f8911a52ecde4b76fa77df29459e32b3ddda9a0cdc9a08348afc34f7aa"
)


class ValidatedProtocolInputs(NamedTuple):
    source: dict[str, Any]
    topology: dict[str, Any]
    source_file_sha256: str
    topology_file_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_hashed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value, digest


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
    ).hexdigest()


def _constant_record(
    name: str, expression: str, units: str, value: float, mathematical_use: str
):
    numerator, denominator = value.as_integer_ratio()
    return {
        "name": name,
        "source_expression": expression,
        "units": units,
        "float64_hex": struct.pack(">d", value).hex(),
        "exact_ratio": [str(numerator), str(denominator)],
        "mathematical_use": mathematical_use,
    }


def write_json_new_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _verify_approved_plan_and_handoff() -> None:
    if sha256_file(APPROVED_PLAN) != APPROVED_PLAN_SHA256:
        raise ValueError("approved plan SHA256 changed")
    if sha256_file(ACCEPTED_HANDOFF) != ACCEPTED_HANDOFF_SHA256:
        raise ValueError("accepted handoff SHA256 changed")


def _source_snapshot() -> dict[str, str]:
    snapshot = {
        str(path.relative_to(SOURCE_ROOT)): sha256_file(path) for path in SOURCE_PATHS
    }
    manifest_bytes = PROTECTED_MANIFEST.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != EXPECTED_FOUNDATION_MANIFEST_SHA256:
        raise ValueError("protected 19-file foundation manifest SHA256 changed")
    snapshot[f"external:{PROTECTED_MANIFEST}"] = manifest_hash
    manifest = json.loads(manifest_bytes)
    protected_files = manifest.get("files")
    if not isinstance(protected_files, dict) or len(protected_files) != 19:
        raise ValueError("protected foundation manifest does not enumerate 19 files")
    for relative, expected in sorted(protected_files.items()):
        path = SOURCE_ROOT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"protected foundation source changed: {relative}")
        snapshot[f"protected:{relative}"] = observed
    return snapshot


def _trial(center, atom_index: int, axis: int, displacement: float):
    # Preserve the original full-v4 runner's float64 update order.
    result = [list(row) for row in center]
    result[atom_index][axis] += displacement
    return result


def build_case_inventory(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("source study must contain exactly five retained rows")
    cases: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        center = row["positions_angstrom"]
        cases.append(
            {
                "case_id": f"center-{row_index:02d}",
                "kind": "singleton",
                "start_positions_angstrom": center,
                "trial_positions_angstrom": center,
                "source_row_index": row_index,
                "source_scale": row["scale"],
                "source_row_status": row["row_status"],
                "source_external_event_annotation": row["row_status"] == "EVENT",
            }
        )
    for row_index, row in enumerate(rows):
        center = row["positions_angstrom"]
        denominators = row["stencil"]["denominators"]
        if len(denominators) != 9:
            raise ValueError(
                "every retained row must contain nine stencil denominators"
            )
        for denominator_index, denominator in enumerate(denominators):
            samples = denominator["samples"]
            if len(samples) != 6:
                raise ValueError("every stencil denominator must retain six samples")
            for sample_index, sample in enumerate(samples):
                cases.append(
                    {
                        "case_id": (
                            f"stencil-{row_index:02d}-{denominator_index:02d}-{sample_index:02d}"
                        ),
                        "kind": "stencil",
                        "start_positions_angstrom": center,
                        "trial_positions_angstrom": _trial(
                            center,
                            denominator["atom_index"],
                            denominator["axis"],
                            sample["displacement_angstrom"],
                        ),
                        "source_row_index": row_index,
                        "source_scale": row["scale"],
                        "source_row_status": row["row_status"],
                        "source_denominator_classification": denominator[
                            "classification"
                        ],
                        "source_denominator_event_kinds": denominator["event_kinds"],
                        "source_sample_weighted_signs_relative_to_center": sample[
                            "weighted_signs_relative_to_center"
                        ],
                        "atom_index": denominator["atom_index"],
                        "axis": denominator["axis"],
                        "displacement_angstrom": sample["displacement_angstrom"],
                    }
                )
    for first in range(4):
        for direction, start_index, trial_index in (
            ("forward", first, first + 1),
            ("reverse", first + 1, first),
        ):
            cases.append(
                {
                    "case_id": f"adjacent-{first:02d}-{direction}",
                    "kind": "adjacent-center",
                    "start_positions_angstrom": rows[start_index]["positions_angstrom"],
                    "trial_positions_angstrom": rows[trial_index]["positions_angstrom"],
                    "source_row_index": start_index,
                    "source_trial_row_index": trial_index,
                    "source_scale": rows[start_index]["scale"],
                    "source_trial_scale": rows[trial_index]["scale"],
                    "source_row_status": rows[start_index]["row_status"],
                    "source_trial_row_status": rows[trial_index]["row_status"],
                    "source_external_event_annotation": any(
                        rows[index]["row_status"] == "EVENT"
                        for index in (start_index, trial_index)
                    ),
                }
            )
    if len(cases) != 283:
        raise AssertionError("frozen case inventory must contain 283 paths")
    return cases


def preregister(source_path: Path, topology_path: Path, output: Path) -> dict[str, Any]:
    _verify_approved_plan_and_handoff()
    source_bytes = source_path.read_bytes()
    topology_bytes = topology_path.read_bytes()
    source = json.loads(source_bytes)
    topology = json.loads(topology_bytes)
    cases = build_case_inventory(source)
    budget = SegmentPreconditionBudget()
    radii = [float(value) for value in topology["cha_radii_angstrom"]]
    probe = 1.4 - 0.52
    constants = [
        *(
            _constant_record(
                f"cha_radius_{index}",
                "topology.cha_radii_angstrom",
                "angstrom",
                radius,
                "radius-cubed weight and exact radius powers",
            )
            for index, radius in enumerate(radii)
        ),
        *(
            _constant_record(
                f"expanded_radius_{index}",
                "actual float64(cha_radius + float64(1.4 - 0.52))",
                "angstrom",
                float(radius + probe),
                "expanded-pair relation after the point-code rounded addition",
            )
            for index, radius in enumerate(radii)
        ),
        _constant_record(
            "effective_probe",
            "float64(1.4 - 0.52)",
            "angstrom",
            probe,
            "expanded-radius addition operand",
        ),
        _constant_record(
            "point_domain_margin",
            "1.0e-8",
            "angstrom",
            1.0e-8,
            "distinct-center and expanded-pair distance guards",
        ),
        _constant_record(
            "size_switch", "10.0", "angstrom", 10.0, "electrostatic-size branch"
        ),
        _constant_record(
            "size_guard",
            "1.0e-8",
            "angstrom",
            1.0e-8,
            "electrostatic-size branch exclusion band",
        ),
        _constant_record(
            "sphere_moment_factor",
            "float64(2.0 / 5.0)",
            "dimensionless",
            2.0 / 5.0,
            "positive sphere inertia moment",
        ),
        _constant_record(
            "size_prefactor",
            "float64(2.5)",
            "dimensionless",
            2.5,
            "exact (2.5 / mass)^3 prefactor for size^6",
        ),
    ]
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "phase": "preregistered-before-characterization",
        "source_result_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "topology_file_sha256": hashlib.sha256(topology_bytes).hexdigest(),
        "topology_content_sha256": topology["content_sha256"],
        "foundation_manifest_expected_sha256": EXPECTED_FOUNDATION_MANIFEST_SHA256,
        "approved_plan_sha256": APPROVED_PLAN_SHA256,
        "accepted_handoff_sha256": ACCEPTED_HANDOFF_SHA256,
        "source_snapshot": _source_snapshot(),
        "budget": asdict(budget),
        "constant_ledger": constants,
        "case_count": len(cases),
        "case_order_sha256": hashlib.sha256(
            canonical_bytes([case["case_id"] for case in cases])
        ).hexdigest(),
        "cases": cases,
        "status_rules": {
            "strongest_terminal": "SEGMENT_PRECONDITIONS_CHARACTERIZED",
            "dynamic_born_sign_status": "UNRESOLVED",
            "computed_quadrature_smoothness": "UNRESOLVED",
            "optimizer_eligible": False,
            "public_admission": False,
            "segment_certificate": False,
        },
        "dependency_ledger": {
            "omitted_flux": "R6 contact+torus flux J_i [angstrom^-3]",
            "omitted_unshifted_inverse_born": "J_i^(1/3) [angstrom^-1]",
            "omitted_born_radius": "B_i=[J_i^(1/3)+shift(size)]^-1 [angstrom]",
            "omitted_weighted_sign": "S_i uses dynamic B_i*B_j denominators",
        },
    }
    protocol["content_sha256"] = content_sha256(protocol)
    write_json_new_atomic(output, protocol)
    return protocol


def verify_protocol_inputs(
    protocol: dict[str, Any], source_path: Path, topology_path: Path
) -> ValidatedProtocolInputs:
    _verify_approved_plan_and_handoff()
    source, source_file_sha256 = _read_hashed_json(source_path, "source result")
    topology, topology_file_sha256 = _read_hashed_json(topology_path, "topology")
    if protocol.get("schema") != SCHEMA or protocol.get(
        "content_sha256"
    ) != content_sha256(protocol):
        raise ValueError("protocol identity or internal content SHA256 changed")
    if (
        protocol.get("approved_plan_sha256") != APPROVED_PLAN_SHA256
        or protocol.get("accepted_handoff_sha256") != ACCEPTED_HANDOFF_SHA256
    ):
        raise ValueError("protocol approved-plan or accepted-handoff identity changed")
    if source_file_sha256 != protocol["source_result_sha256"]:
        raise ValueError("source SHA256 differs from preregistration")
    if topology_file_sha256 != protocol["topology_file_sha256"]:
        raise ValueError("topology SHA256 differs from preregistration")
    if _source_snapshot() != protocol["source_snapshot"]:
        raise ValueError("live code source SHA256 differs from preregistration")
    if protocol["budget"] != asdict(SegmentPreconditionBudget()):
        raise ValueError("frozen segment budget differs from preregistration")
    if protocol["case_count"] != 283 or len(protocol["cases"]) != 283:
        raise ValueError("preregistered case inventory is incomplete")
    reconstructed = build_case_inventory(source)
    if canonical_bytes(reconstructed) != canonical_bytes(protocol["cases"]):
        raise ValueError(
            "preregistered cases differ from reconstructed source inventory"
        )
    observed_order = hashlib.sha256(
        canonical_bytes([case["case_id"] for case in reconstructed])
    ).hexdigest()
    if observed_order != protocol["case_order_sha256"]:
        raise ValueError("preregistered case ordering hash changed")
    return ValidatedProtocolInputs(
        source=source,
        topology=topology,
        source_file_sha256=source_file_sha256,
        topology_file_sha256=topology_file_sha256,
    )


def load_externally_pinned_protocol(
    protocol_path: Path, expected_protocol_sha256: str
) -> dict[str, Any]:
    if (
        not isinstance(expected_protocol_sha256, str)
        or len(expected_protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_protocol_sha256
        )
    ):
        raise ValueError("expected protocol SHA256 must be 64 lowercase hex characters")
    payload = protocol_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_protocol_sha256:
        raise ValueError("external protocol file SHA256 mismatch")
    protocol = json.loads(payload)
    if not isinstance(protocol, dict):
        raise ValueError("protocol JSON must contain an object")
    return protocol


def recheck_run_input_hashes(
    protocol_path: Path,
    source_path: Path,
    topology_path: Path,
    before: dict[str, str],
) -> dict[str, str]:
    after = {
        "protocol": sha256_file(protocol_path),
        "source_result": sha256_file(source_path),
        "topology": sha256_file(topology_path),
    }
    changed = [name for name, digest in after.items() if digest != before[name]]
    if changed:
        raise ValueError(
            "run input file changed during characterization: " + ", ".join(changed)
        )
    return after


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def run(
    protocol_path: Path,
    source_path: Path,
    topology_path: Path,
    output: Path,
    *,
    expected_protocol_sha256: str,
):
    import torch

    protocol = load_externally_pinned_protocol(protocol_path, expected_protocol_sha256)
    validated = verify_protocol_inputs(protocol, source_path, topology_path)
    before = _source_snapshot()
    if before != protocol["source_snapshot"]:
        raise ValueError(
            "preregistered source identity changed before characterization"
        )
    topology = ContinuumChaTopology.from_mapping(
        validated.topology,
        expected_content_sha256=protocol["topology_content_sha256"],
    )
    budget = SegmentPreconditionBudget(**protocol["budget"])
    results = []
    for case in protocol["cases"]:
        record: dict[str, Any] = {
            "case_id": case["case_id"],
            "source_annotations": {
                key: value for key, value in case.items() if key.startswith("source_")
            },
        }
        try:
            report = characterize_cha_segment_preconditions(
                torch.tensor(case["start_positions_angstrom"], dtype=torch.float64),
                torch.tensor(case["trial_positions_angstrom"], dtype=torch.float64),
                topology,
                expected_topology_sha256=protocol["topology_content_sha256"],
                budget=budget,
            )
        except (
            SegmentPreconditionValidationError,
            SegmentPreconditionResourceError,
        ) as error:
            record["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "dynamic_born_sign_status": "UNRESOLVED",
                "computed_quadrature_smoothness": "UNRESOLVED",
                "optimizer_eligible": False,
                "public_admission": False,
                "segment_certificate": False,
            }
        else:
            record["report"] = _jsonable(report)
        results.append(record)
    after = _source_snapshot()
    if after != before:
        raise ValueError("live source changed during characterization")
    input_hashes_before = {
        "protocol": expected_protocol_sha256,
        "source_result": validated.source_file_sha256,
        "topology": validated.topology_file_sha256,
    }
    input_hashes_after = recheck_run_input_hashes(
        protocol_path,
        source_path,
        topology_path,
        input_hashes_before,
    )
    failure_count = sum("failure" in record for record in results)
    unresolved_count = sum(
        record.get("report", {}).get("status") == "PRECONDITION_BOUNDS_UNRESOLVED"
        for record in results
    )
    terminal_status = (
        "PRECONDITION_TYPED_FAILURES_RETAINED"
        if failure_count
        else (
            "PRECONDITION_BOUNDS_UNRESOLVED"
            if unresolved_count
            else "SEGMENT_PRECONDITIONS_CHARACTERIZED"
        )
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_type": "coordinate-only-precondition-characterization",
        "protocol_sha256": protocol["content_sha256"],
        "protocol_file_sha256": expected_protocol_sha256,
        "input_file_hashes_before": input_hashes_before,
        "input_file_hashes_after": input_hashes_after,
        "input_files_unchanged": True,
        "case_count": len(results),
        "status": terminal_status,
        "typed_failure_count": failure_count,
        "unresolved_bound_count": unresolved_count,
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "source_snapshot_unchanged": True,
        "dynamic_born_sign_status": "UNRESOLVED",
        "computed_quadrature_smoothness": "UNRESOLVED",
        "optimizer_eligible": False,
        "public_admission": False,
        "segment_certificate": False,
        "results": results,
    }
    result["content_sha256"] = content_sha256(result)
    write_json_new_atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preregister", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--source-result", type=Path, required=True)
        command.add_argument("--topology", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--protocol", type=Path, required=True)
            command.add_argument("--expected-protocol-sha256", required=True)
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.source_result, args.topology, args.output)
    else:
        run(
            args.protocol,
            args.source_result,
            args.topology,
            args.output,
            expected_protocol_sha256=args.expected_protocol_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
