#!/usr/bin/env python3
"""Open targets only after closing 505 zero-training v3 predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

EXPECTED_RECORD_COUNT = 505
EXPECTED_RECORD_ARTIFACT = "route2-mace-mdp-polar-zero-adt-ddx-development-record-v1"
EXPECTED_PREREGISTRATION = "route2-mace-mdp-polar-zero-adt-ddx-development-prereg-v1"
EXPECTED_PROFILE_ID = (
    "route2-research-macepolar-zero-point-mdp-alpha-polar-residual-"
    "role-separated-adt-ddx-operational-v3"
)
INPUT_FILES = {
    "dataset_zip_sha256": "MNSolDatabase_v2012.zip",
    "development_selection_sha256": (
        "route2-mnsol-development-selection-v1.private.json"
    ),
    "protocol_sha256": "route2-mnsol-protocol-v1.json",
    "pilot_selection_sha256": "route2-mnsol-pilot-selection-v1.json",
}
REPO_ROOT = Path(
    os.environ.get("MAPLE_ROUTE2_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
BENCHMARK_ROOT = REPO_ROOT / "docs/implicit-solvation/benchmarks"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metrics(predicted: np.ndarray, experimental: np.ndarray) -> dict[str, Any]:
    signed = predicted - experimental
    absolute = np.abs(signed)
    return {
        "record_count": int(predicted.size),
        "mean_predicted_delta_g_kcal_mol": float(np.mean(predicted)),
        "mean_experimental_delta_g_kcal_mol": float(np.mean(experimental)),
        "mean_signed_error_kcal_mol": float(np.mean(signed)),
        "mean_absolute_error_kcal_mol": float(np.mean(absolute)),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(np.square(signed)))),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "q95_absolute_error_kcal_mol": float(np.quantile(absolute, 0.95)),
        "ge_1_0_count": int(np.count_nonzero(absolute >= 1.0)),
        "ge_1_0_fraction": float(np.mean(absolute >= 1.0)),
        "ge_1_5_count": int(np.count_nonzero(absolute >= 1.5)),
        "ge_1_5_fraction": float(np.mean(absolute >= 1.5)),
        "ge_3_0_count": int(np.count_nonzero(absolute >= 3.0)),
        "ge_5_0_count": int(np.count_nonzero(absolute >= 5.0)),
    }


def _load_preregistration(path: Path, runner: Path, input_root: Path) -> dict[str, Any]:
    prereg = json.loads(path.read_text())
    digest = prereg.pop("preregistration_sha256", None)
    _require(digest == _canonical_sha256(prereg), "Preregistration hash drifted.")
    prereg["preregistration_sha256"] = digest
    _require(
        prereg.get("artifact_id") == EXPECTED_PREREGISTRATION,
        "Unknown preregistration identity.",
    )
    _require(
        prereg.get("status") == "locked-after-stage-b-pass-before-zero-adt-505",
        "Preregistration is not locked.",
    )
    _require(
        prereg.get("record_count") == EXPECTED_RECORD_COUNT
        and prereg.get("partition") == "development"
        and prereg.get("confirmation_partition_opened") is False,
        "Partition contract drifted.",
    )
    _require(
        prereg.get("prediction_runner_uses_experimental_targets") is False
        and prereg.get("prediction_runner_emits_experimental_targets") is False,
        "Prediction target-use contract drifted.",
    )
    _require(prereg.get("runner_sha256") == _sha256(runner), "Runner drifted.")
    _require(
        prereg.get("aggregator_sha256") == _sha256(Path(__file__)),
        "Aggregator drifted.",
    )
    for key, relative in INPUT_FILES.items():
        _require(
            prereg.get(key) == _sha256(input_root / relative),
            f"Frozen input drifted: {key}.",
        )
    return prereg


def _load_predictions(
    *,
    input_dir: Path,
    preregistration_path: Path,
    prereg: Mapping[str, Any],
    runner: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prereg_file_sha = _sha256(preregistration_path)
    runner_sha = _sha256(runner)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_names = {
        f"index-{index:03d}.json" for index in range(EXPECTED_RECORD_COUNT)
    }
    observed_names = {path.name for path in input_dir.glob("index-*.json")}
    _require(observed_names == expected_names, "Exact 505 prediction closure failed.")
    forbidden_keys = {
        "experimental_delta_g_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
        "target",
        "targets",
    }

    def keys_in(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys_in(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys_in(item) for item in value))
        return set()

    for index in range(EXPECTED_RECORD_COUNT):
        path = input_dir / f"index-{index:03d}.json"
        row = json.loads(path.read_text())
        digest = row.pop("record_sha256", None)
        _require(digest == _canonical_sha256(row), f"Record hash drift in {path}.")
        row["record_sha256"] = digest
        _require(
            row.get("artifact") == EXPECTED_RECORD_ARTIFACT
            and row.get("selection_index") == index
            and row.get("preregistration_sha256") == prereg_file_sha
            and row.get("runner_sha256") == runner_sha
            and row.get("partition") == "development"
            and row.get("confirmation_partition_opened") is False
            and row.get("do_not_commit") is True
            and row.get("profile_id") == EXPECTED_PROFILE_ID,
            f"Prediction identity drift in {path}.",
        )
        _require(
            row.get("experimental_targets_parsed_by_selection_loader") is True
            and row.get("experimental_targets_used_by_prediction") is False
            and row.get("experimental_targets_emitted") is False,
            f"Prediction target-use boundary drift in {path}.",
        )
        _require(
            keys_in(row).isdisjoint(forbidden_keys),
            f"Prediction record contains target/error data: {path}.",
        )
        if row.get("status") == "pass":
            for ledger in (
                "m0_electrostatic_only",
                "m1_electrostatic_plus_stock_smd_cds",
            ):
                values = row.get(ledger)
                _require(
                    isinstance(values, dict)
                    and set(values) == {"predicted_delta_g_kcal_mol"}
                    and np.isfinite(values["predicted_delta_g_kcal_mol"]),
                    f"Invalid target-unused {ledger} prediction in {path}.",
                )
            records.append(row)
        else:
            failures.append(row)
    return records, failures


def _load_reference_rows(input_root: Path) -> list[Any]:
    sys.path.insert(0, str(BENCHMARK_ROOT))
    from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
    from mnsol_partition import validate_frozen_mnsol_partition_selection

    protocol = load_mnsol_protocol(input_root / INPUT_FILES["protocol_sha256"])
    dataset = load_mnsol_v2012(input_root / INPUT_FILES["dataset_zip_sha256"], protocol)
    selection = json.loads(
        (input_root / INPUT_FILES["development_selection_sha256"]).read_text()
    )
    pilot = json.loads((input_root / INPUT_FILES["pilot_selection_sha256"]).read_text())
    rows = validate_frozen_mnsol_partition_selection(
        selection, dataset, protocol, pilot
    )
    _require(len(rows) == EXPECTED_RECORD_COUNT, "Reference selection is not 505 rows.")
    return list(rows)


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_dir.expanduser().resolve(strict=True)
    input_root = args.input_root.expanduser().resolve(strict=True)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    runner = args.runner.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    _require(not output.exists(), "Aggregate output already exists.")
    prereg = _load_preregistration(prereg_path, runner, input_root)
    predictions, failures = _load_predictions(
        input_dir=input_dir,
        preregistration_path=prereg_path,
        prereg=prereg,
        runner=runner,
    )
    _require(
        len(predictions) + len(failures) == EXPECTED_RECORD_COUNT,
        "Prediction count changed before target opening.",
    )

    evaluated: list[dict[str, Any]] = []
    if not failures:
        references = _load_reference_rows(input_root)
        for index, (prediction, selected) in enumerate(zip(predictions, references)):
            geometry = selected.eligible_record.geometry
            _require(
                prediction["selection_index"] == index
                and prediction["opaque_record_id"] == selected.opaque_record_id
                and prediction["geometry_sha256"] == geometry.sha256
                and prediction["dataset_row_sha256"]
                == selected.eligible_record.record.raw_row_sha256
                and prediction["selection_score_sha256"]
                == selected.selection_score_sha256
                and prediction["canonical_solvent"] == selected.canonical_solvent,
                f"Reference identity mismatch at selection index {index}.",
            )
            evaluated.append(
                {
                    "solvent": selected.canonical_solvent,
                    "experimental": float(
                        selected.eligible_record.record.delta_g_kcal_mol
                    ),
                    "m0": float(
                        prediction["m0_electrostatic_only"][
                            "predicted_delta_g_kcal_mol"
                        ]
                    ),
                    "m1": float(
                        prediction["m1_electrostatic_plus_stock_smd_cds"][
                            "predicted_delta_g_kcal_mol"
                        ]
                    ),
                }
            )

    def metric(rows: list[dict[str, Any]], ledger: str) -> dict[str, Any] | None:
        if not rows:
            return None
        return _metrics(
            np.asarray([row[ledger] for row in rows], dtype=float),
            np.asarray([row["experimental"] for row in rows], dtype=float),
        )

    m0 = metric(evaluated, "m0")
    m1 = metric(evaluated, "m1")
    by_solvent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        by_solvent[row["solvent"]].append(row)
    per_solvent = {
        solvent: {
            "m0_electrostatic_only": metric(rows, "m0"),
            "m1_electrostatic_plus_stock_smd_cds": metric(rows, "m1"),
        }
        for solvent, rows in sorted(by_solvent.items())
    }
    target_passed = bool(
        not failures
        and m1 is not None
        and m1["record_count"] == EXPECTED_RECORD_COUNT
        and m1["mean_absolute_error_kcal_mol"] <= 1.5
    )
    stretch_passed = bool(
        target_passed and m1 is not None and m1["mean_absolute_error_kcal_mol"] <= 1.0
    )
    helped = sum(
        abs(row["m1"] - row["experimental"]) < abs(row["m0"] - row["experimental"])
        for row in evaluated
    )
    worsened = sum(
        abs(row["m1"] - row["experimental"]) > abs(row["m0"] - row["experimental"])
        for row in evaluated
    )
    ordered_records = sorted(
        predictions + failures, key=lambda row: int(row["selection_index"])
    )
    payload: dict[str, object] = {
        "artifact": "route2-mace-mdp-polar-zero-adt-ddx-development-full-v1",
        "status": "pass" if target_passed else "fail",
        "do_not_commit": True,
        "partition": "development",
        "confirmation_partition_opened": False,
        "record_count": EXPECTED_RECORD_COUNT,
        "success_count": len(predictions),
        "failure_count": len(failures),
        "failed_selection_indices": [row["selection_index"] for row in failures],
        "preregistration_file_sha256": _sha256(prereg_path),
        "preregistration_artifact_sha256": prereg["preregistration_sha256"],
        "runner_sha256": _sha256(runner),
        "aggregator_sha256": _sha256(Path(__file__)),
        "prediction_record_sha256s": [row["record_sha256"] for row in ordered_records],
        "target_opening_stage": (
            "after-exact-505-prediction-record-closure"
            if not failures
            else "not-opened-because-provider-failure"
        ),
        "hard_accuracy_target": {
            "metric": "mean_absolute_error_kcal_mol",
            "comparison": "<=",
            "threshold_kcal_mol": 1.5,
            "passed": target_passed,
        },
        "m0_electrostatic_only_metrics": m0,
        "m1_electrostatic_plus_stock_smd_cds_metrics": m1,
        "paired_stock_cds_effect": {
            "helped_count": helped,
            "worsened_count": worsened,
            "tied_count": len(evaluated) - helped - worsened,
        },
        "stretch_accuracy_target": {
            "metric": "mean_absolute_error_kcal_mol",
            "comparison": "<=",
            "threshold_kcal_mol": 1.0,
            "passed": stretch_passed,
        },
        "per_solvent_metrics": per_solvent,
        "claim_boundary": (
            "Complete frozen MNSol development-only zero-training v3 M0/M1 "
            "energy decision. Targets were opened only after exact 505 prediction "
            "record closure. Development targets were historically known from older "
            "profiles; the 148-row confirmation partition remains sealed. No fitting, "
            "force, virial, Hessian, workflow, or production admission."
        ),
    }
    payload["verification_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
