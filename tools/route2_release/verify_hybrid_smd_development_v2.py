"""Independent integrity audit for the frozen hybrid MNSol development run.

The preregistered aggregator is byte-locked and remains the decision artifact.
This verifier does not change that decision rule or permit method selection. It
adds orthogonal checks that every record is bound to the same preregistration,
runner, checkpoints, source snapshot, and deterministic record identity before
recomputing the declared metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


EXPECTED_ARTIFACT = "route2-hybrid-smd-development-record-v2"
EXPECTED_PREREGISTRATION = "route2-hybrid-smd-development-prereg-v2"
EXPECTED_RECORD_COUNT = 505


class HybridDevelopmentVerificationError(RuntimeError):
    """Raised when frozen hybrid evidence is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HybridDevelopmentVerificationError(
            f"{name} must be a lowercase SHA256 digest."
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise HybridDevelopmentVerificationError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HybridDevelopmentVerificationError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise HybridDevelopmentVerificationError(f"{name} must be finite.")
    return result


def _record_identity(
    *,
    preregistration_sha256: str,
    runner_sha256: str,
    mdp_checkpoint_sha256: str,
    polar_checkpoint_sha256: str,
    selection_index: int,
) -> str:
    return _canonical_sha256(
        {
            "contract": "route2-hybrid-smd-development-record-v2",
            "preregistration_sha256": preregistration_sha256,
            "runner_sha256": runner_sha256,
            "mdp_checkpoint_sha256": mdp_checkpoint_sha256,
            "polar_checkpoint_sha256": polar_checkpoint_sha256,
            "selection_index": selection_index,
        }
    )


def _metrics(records: list[Mapping[str, Any]]) -> dict[str, object]:
    signed = [
        _finite(row["predicted_delta_g_kcal_mol"], name="predicted energy")
        - _finite(row["experimental_delta_g_kcal_mol"], name="experimental energy")
        for row in records
    ]
    absolute = [abs(value) for value in signed]
    return {
        "record_count": len(records),
        "mean_signed_error_kcal_mol": fmean(signed),
        "mean_absolute_error_kcal_mol": fmean(absolute),
        "root_mean_square_error_kcal_mol": math.sqrt(
            fmean(value * value for value in signed)
        ),
        "maximum_absolute_error_kcal_mol": max(absolute),
        "ge_1_0_count": sum(value >= 1.0 for value in absolute),
        "ge_1_5_count": sum(value >= 1.5 for value in absolute),
    }


def _validate_preregistration(
    preregistration: Mapping[str, Any],
    *,
    runner_path: Path,
    aggregator_path: Path,
    source_root: Path,
    mdp_checkpoint_path: Path,
    polar_checkpoint_path: Path,
    expected_count: int,
) -> tuple[str, str, str, str]:
    if preregistration.get("artifact_id") != EXPECTED_PREREGISTRATION:
        raise HybridDevelopmentVerificationError("Unknown preregistration identity.")
    if preregistration.get("status") != "locked-before-first-v2-hybrid-evaluation":
        raise HybridDevelopmentVerificationError("Preregistration is not locked.")
    if preregistration.get("partition") != "development":
        raise HybridDevelopmentVerificationError("Partition is not development.")
    if preregistration.get("confirmation_partition_opened") is not False:
        raise HybridDevelopmentVerificationError("Confirmation partition was opened.")
    if preregistration.get("fitting_or_calibration_permitted") is not False:
        raise HybridDevelopmentVerificationError("Preregistration permits fitting.")
    if preregistration.get("record_count") != expected_count:
        raise HybridDevelopmentVerificationError("Preregistered record count drifted.")
    target = preregistration.get("hard_accuracy_target")
    if target != {
        "comparison": "<=",
        "metric": "mean_absolute_error_kcal_mol",
        "threshold_kcal_mol": 1.5,
    }:
        raise HybridDevelopmentVerificationError("Accuracy target drifted.")

    runner_sha256 = _sha256(runner_path)
    aggregator_sha256 = _sha256(aggregator_path)
    mdp_sha256 = _sha256(mdp_checkpoint_path)
    polar_sha256 = _sha256(polar_checkpoint_path)
    for key, actual in (
        ("runner_sha256", runner_sha256),
        ("aggregator_sha256", aggregator_sha256),
        ("mdp_checkpoint_sha256", mdp_sha256),
        ("polar_checkpoint_sha256", polar_sha256),
    ):
        if preregistration.get(key) != actual:
            raise HybridDevelopmentVerificationError(f"{key} drifted.")

    source_hashes = preregistration.get("source_files_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise HybridDevelopmentVerificationError("Source hash manifest is missing.")
    for relative, expected in source_hashes.items():
        if not isinstance(relative, str):
            raise HybridDevelopmentVerificationError("Source path is not text.")
        _digest(expected, name=f"source_files_sha256[{relative!r}]")
        path = source_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise HybridDevelopmentVerificationError(
                f"Frozen source file drifted: {relative}."
            )
    return runner_sha256, aggregator_sha256, mdp_sha256, polar_sha256


def _validate_record(
    row: Mapping[str, Any],
    *,
    index: int,
    preregistration_sha256: str,
    runner_sha256: str,
    mdp_checkpoint_sha256: str,
    polar_checkpoint_sha256: str,
) -> None:
    prefix = f"record {index:03d}"
    if row.get("artifact") != EXPECTED_ARTIFACT:
        raise HybridDevelopmentVerificationError(f"{prefix} artifact drifted.")
    if row.get("selection_index") != index:
        raise HybridDevelopmentVerificationError(f"{prefix} index drifted.")
    if row.get("partition") != "development":
        raise HybridDevelopmentVerificationError(f"{prefix} partition drifted.")
    if row.get("confirmation_partition_opened") is not False:
        raise HybridDevelopmentVerificationError(f"{prefix} opened confirmation.")
    if row.get("do_not_commit") is not True:
        raise HybridDevelopmentVerificationError(f"{prefix} commit guard drifted.")
    for key, expected in (
        ("preregistration_sha256", preregistration_sha256),
        ("runner_sha256", runner_sha256),
        ("mdp_checkpoint_sha256", mdp_checkpoint_sha256),
        ("polar_checkpoint_sha256", polar_checkpoint_sha256),
    ):
        if row.get(key) != expected:
            raise HybridDevelopmentVerificationError(f"{prefix} {key} drifted.")
    expected_identity = _record_identity(
        preregistration_sha256=preregistration_sha256,
        runner_sha256=runner_sha256,
        mdp_checkpoint_sha256=mdp_checkpoint_sha256,
        polar_checkpoint_sha256=polar_checkpoint_sha256,
        selection_index=index,
    )
    if row.get("record_identity_sha256") != expected_identity:
        raise HybridDevelopmentVerificationError(f"{prefix} identity drifted.")
    for key in ("opaque_record_id", "geometry_sha256", "hybrid_configuration_sha256"):
        _digest(row.get(key), name=f"{prefix} {key}")

    status = row.get("status")
    if status == "provider-failure":
        if not isinstance(row.get("failure_type"), str):
            raise HybridDevelopmentVerificationError(
                f"{prefix} provider failure lacks a type."
            )
        return
    if status != "pass":
        raise HybridDevelopmentVerificationError(f"{prefix} has unknown status.")
    for key in (
        "pes_configuration_sha256",
        "state_sha256",
        "electrostatic_root_sha256",
        "continuum_state_sha256",
    ):
        _digest(row.get(key), name=f"{prefix} {key}")
    predicted = _finite(
        row.get("predicted_delta_g_kcal_mol"), name=f"{prefix} predicted energy"
    )
    experimental = _finite(
        row.get("experimental_delta_g_kcal_mol"),
        name=f"{prefix} experimental energy",
    )
    signed = _finite(
        row.get("signed_error_kcal_mol"), name=f"{prefix} signed error"
    )
    absolute = _finite(
        row.get("absolute_error_kcal_mol"), name=f"{prefix} absolute error"
    )
    if signed != predicted - experimental or absolute != abs(signed):
        raise HybridDevelopmentVerificationError(f"{prefix} error ledger drifted.")
    residual = _finite(row.get("root_residual_eV"), name=f"{prefix} root residual")
    if residual < 0.0:
        raise HybridDevelopmentVerificationError(f"{prefix} residual is negative.")
    for key in (
        "vacuum_energy_eV",
        "continuum_polarization_kcal_mol",
        "smd_cds_kcal_mol",
        "solve_wall_seconds",
    ):
        _finite(row.get(key), name=f"{prefix} {key}")


def audit_hybrid_development(
    *,
    input_dir: Path,
    preregistration_path: Path,
    runner_path: Path,
    aggregator_path: Path,
    source_root: Path,
    mdp_checkpoint_path: Path,
    polar_checkpoint_path: Path,
    expected_count: int = EXPECTED_RECORD_COUNT,
    allow_incomplete: bool = False,
) -> dict[str, object]:
    preregistration = json.loads(preregistration_path.read_text())
    preregistration_sha256 = _sha256(preregistration_path)
    runner_sha256, aggregator_sha256, mdp_sha256, polar_sha256 = (
        _validate_preregistration(
            preregistration,
            runner_path=runner_path,
            aggregator_path=aggregator_path,
            source_root=source_root,
            mdp_checkpoint_path=mdp_checkpoint_path,
            polar_checkpoint_path=polar_checkpoint_path,
            expected_count=expected_count,
        )
    )

    expected_names = {f"index-{index:03d}.json" for index in range(expected_count)}
    present_paths = {path.name: path for path in input_dir.glob("index-*.json")}
    unexpected = sorted(set(present_paths) - expected_names)
    if unexpected:
        raise HybridDevelopmentVerificationError(
            f"Unexpected record files: {unexpected}."
        )
    missing = sorted(expected_names - set(present_paths))
    if missing and not allow_incomplete:
        raise HybridDevelopmentVerificationError(
            f"Development panel is incomplete: {len(missing)} records missing."
        )

    records: list[dict[str, Any]] = []
    record_file_hashes: list[tuple[int, str]] = []
    for index in range(expected_count):
        path = present_paths.get(f"index-{index:03d}.json")
        if path is None:
            continue
        row = json.loads(path.read_text())
        if not isinstance(row, dict):
            raise HybridDevelopmentVerificationError(
                f"record {index:03d} is not a JSON object."
            )
        _validate_record(
            row,
            index=index,
            preregistration_sha256=preregistration_sha256,
            runner_sha256=runner_sha256,
            mdp_checkpoint_sha256=mdp_sha256,
            polar_checkpoint_sha256=polar_sha256,
        )
        records.append(row)
        record_file_hashes.append((index, _sha256(path)))

    successes = [row for row in records if row["status"] == "pass"]
    failures = [row for row in records if row["status"] == "provider-failure"]
    aggregate = _metrics(successes) if successes else None
    by_solvent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in successes:
        solvent = row.get("canonical_solvent")
        if not isinstance(solvent, str) or not solvent:
            raise HybridDevelopmentVerificationError("Solvent identity is invalid.")
        by_solvent[solvent].append(row)
    per_solvent = {
        solvent: _metrics(rows) for solvent, rows in sorted(by_solvent.items())
    }
    complete = not missing and len(records) == expected_count
    threshold = 1.5
    target_passed: bool | None = None
    if complete and not failures and aggregate is not None:
        target_passed = aggregate["mean_absolute_error_kcal_mol"] <= threshold
    if not complete:
        status = "incomplete"
    elif failures:
        status = "provider-failure"
    elif target_passed:
        status = "pass"
    else:
        status = "accuracy-failure"

    hybrid_configurations = sorted(
        {row["hybrid_configuration_sha256"] for row in records}
    )
    if len(hybrid_configurations) > 1:
        raise HybridDevelopmentVerificationError(
            "Records contain multiple hybrid model configurations."
        )
    payload: dict[str, object] = {
        "artifact": "route2-hybrid-smd-development-integrity-audit-v1",
        "status": status,
        "do_not_commit": True,
        "partition": "development",
        "confirmation_partition_opened": False,
        "expected_record_count": expected_count,
        "record_count": len(records),
        "success_count": len(successes),
        "failure_count": len(failures),
        "missing_selection_indices": [
            int(name.removeprefix("index-").removesuffix(".json"))
            for name in missing
        ],
        "failed_selection_indices": [row["selection_index"] for row in failures],
        "preregistration_sha256": preregistration_sha256,
        "runner_sha256": runner_sha256,
        "locked_aggregator_sha256": aggregator_sha256,
        "mdp_checkpoint_sha256": mdp_sha256,
        "polar_checkpoint_sha256": polar_sha256,
        "hybrid_configuration_sha256": (
            hybrid_configurations[0] if hybrid_configurations else None
        ),
        "record_files_manifest_sha256": _canonical_sha256(record_file_hashes),
        "hard_accuracy_target": {
            "metric": "mean_absolute_error_kcal_mol",
            "comparison": "<=",
            "threshold_kcal_mol": threshold,
            "passed": target_passed,
        },
        "aggregate_metrics": aggregate,
        "per_solvent_metrics": per_solvent,
        "claim_boundary": (
            "Independent integrity audit of frozen development-only evidence; "
            "it does not replace the preregistered decision aggregator, open the "
            "confirmation partition, permit fitting, or establish derivative or "
            "release accuracy."
        ),
    }
    payload["verification_sha256"] = _canonical_sha256(payload)
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--aggregator", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, required=True)
    parser.add_argument("--polar-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    payload = audit_hybrid_development(
        input_dir=args.input_dir,
        preregistration_path=args.preregistration,
        runner_path=args.runner,
        aggregator_path=args.aggregator,
        source_root=args.source_root,
        mdp_checkpoint_path=args.mdp_checkpoint,
        polar_checkpoint_path=args.polar_checkpoint,
        allow_incomplete=args.allow_incomplete,
    )
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return {
        "pass": 0,
        "accuracy-failure": 2,
        "provider-failure": 3,
        "incomplete": 4,
    }[str(payload["status"])]


if __name__ == "__main__":
    raise SystemExit(main())
