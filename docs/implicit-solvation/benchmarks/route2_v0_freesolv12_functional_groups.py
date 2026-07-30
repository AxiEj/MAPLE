#!/usr/bin/env python3
"""Validate the immutable Route-2 V0 FreeSolv functional-group gate.

The prior historical FreeSolv-10 regression remains intact, including its
7.041442082076966 kcal/mol ethyl-acetate failure.  It contains two nonpolar
scaffold controls, however, so it cannot by itself satisfy the stricter
requirement for ten *actual* distinct functional groups.  This twelve-record
superset preserves all historical records and adds an amide and a carboxylic
acid.  It consequently has ten distinct source-labelled functional groups and
two required nonfunctional controls.

No V0 prediction is stored here.  This module only evaluates future candidate
predictions against frozen experimental labels; a small panel, MAE, RMSE, or a
post-hoc subset can never stand in for this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from benchmark_core import (  # pyright: ignore[reportImplicitRelativeImport]
    canonical_json_bytes,
    write_json_atomic,
)
from route2_v0_historical_freesolv10 import (  # pyright: ignore[reportImplicitRelativeImport]
    HISTORICAL_FREE_SOLV10_IDS,
    HISTORICAL_WORST_ABSOLUTE_ERROR_KCAL_MOL,
    HISTORICAL_WORST_RECORD_ID,
    load_historical_freesolv10_manifest,
)

FUNCTIONAL_GROUP_PANEL_ID = "route2-v0-freesolv12-functional-groups-v1"
FUNCTIONAL_GROUP_PANEL_IDS = (
    "mobley_9055303",
    "mobley_3053621",
    "mobley_1636752",
    "mobley_7015518",
    "mobley_3867265",
    "mobley_6973347",
    "mobley_7532833",
    "mobley_4883284",
    "mobley_2198613",
    "mobley_8578590",
    "mobley_8048190",
    "mobley_3034976",
)
FUNCTIONAL_GROUP_PANEL_LABELS: tuple[str | None, ...] = (
    None,
    None,
    "primary-alcohol",
    "dialkyl-ether",
    "ketone",
    "carboxylic-acid-ester",
    "carbonitrile",
    "primary-aromatic-amine",
    "alkyl-chloride",
    "sulfoxide",
    "primary-carboxylic-acid-amide",
    "carboxylic-acid",
)
FUNCTIONAL_GROUP_PANEL_NONFUNCTIONAL_CONTROL_IDS = (
    "mobley_9055303",
    "mobley_3053621",
)
MINIMUM_DISTINCT_FUNCTIONAL_GROUPS = 10
STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL = 1.5
FREESOLV_DATABASE_TXT_SHA256 = (
    "2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260"
)
FREESOLV_DATABASE_JSON_SHA256 = (
    "9133b2438af6081d4cf6ac040cb201c2e3ec953988a45248586f20f2f159c78a"
)
FREESOLV_MOL2_ARCHIVE_SHA256 = (
    "15ece6114442a8eb5e720c0ad25bb48e4a6a35392da36f72b80a6f2f415fd133"
)
LOCKED_RECORDS_SHA256 = (
    "065a3ff911d2a88dd62bdc38321ee2c359fce207eb4d9cdf4ec9f8b248b0aac3"
)
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "route2-v0-freesolv12-functional-groups-v1.json"
)


@dataclass(frozen=True)
class FreeSolvFunctionalGroupRecordEvaluation:
    """One frozen FreeSolv record and its recomputed candidate error."""

    compound_id: str
    name: str
    functional_group: str | None
    is_nonfunctional_control: bool
    experimental_kcal_mol: float
    predicted_kcal_mol: float
    signed_error_kcal_mol: float
    absolute_error_kcal_mol: float
    passes_strict_threshold: bool


@dataclass(frozen=True)
class FreeSolvFunctionalGroupPanelEvaluation:
    """All-record acceptance decision for the frozen twelve-record panel."""

    panel_id: str
    manifest_sha256: str
    record_count: int
    distinct_functional_group_count: int
    nonfunctional_control_count: int
    historical_freesolv10_regression_embedded: bool
    threshold_kcal_mol: float
    maximum_absolute_error_kcal_mol: float
    worst_compound_id: str
    strict_all_records_under_threshold: bool
    records: tuple[FreeSolvFunctionalGroupRecordEvaluation, ...]

    @property
    def status(self) -> str:
        return "pass" if self.strict_all_records_under_threshold else "fail"

    def as_dict(self) -> dict[str, Any]:
        """Return a stable full report; aggregate metrics cannot override it."""

        return {
            "schema_version": 1,
            "panel_id": self.panel_id,
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
            "acceptance": {
                "record_count": self.record_count,
                "minimum_distinct_functional_groups": (
                    MINIMUM_DISTINCT_FUNCTIONAL_GROUPS
                ),
                "distinct_functional_group_count": self.distinct_functional_group_count,
                "nonfunctional_control_count": self.nonfunctional_control_count,
                "historical_freesolv10_regression_embedded": (
                    self.historical_freesolv10_regression_embedded
                ),
                "threshold_kcal_mol": self.threshold_kcal_mol,
                "comparison": "strictly_less_than",
                "maximum_absolute_error_kcal_mol": self.maximum_absolute_error_kcal_mol,
                "worst_compound_id": self.worst_compound_id,
                "strict_all_records_under_threshold": (
                    self.strict_all_records_under_threshold
                ),
                "mae_or_rmse_alone_is_not_a_pass_condition": True,
            },
            "records": [asdict(record) for record in self.records],
        }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"FreeSolv functional-group manifest not found: {path}."
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"FreeSolv functional-group manifest is invalid JSON: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise TypeError("FreeSolv functional-group manifest must be a JSON object.")
    return value


def _require_historical_embedding(records: list[object]) -> None:
    """Ensure the full old 7 kcal/mol regression survives in this superset."""

    historical = load_historical_freesolv10_manifest()
    historical_records = historical["locked_records"]
    embedded = records[: len(HISTORICAL_FREE_SOLV10_IDS)]
    if not all(isinstance(record, dict) for record in embedded):
        raise ValueError("Functional-group panel historical records are malformed.")
    embedded_records = [cast(dict[str, Any], record) for record in embedded]
    if (
        tuple(record["compound_id"] for record in embedded_records)
        != HISTORICAL_FREE_SOLV10_IDS
    ):
        raise ValueError(
            "Functional-group panel no longer embeds historical FreeSolv-10."
        )

    required_fields = (
        "compound_id",
        "smiles",
        "name",
        "chemical_class",
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "experimental_reference",
        "mol2_archive_member",
        "mol2_sha256",
        "natoms",
    )
    for old_record, embedded_record in zip(
        historical_records, embedded_records, strict=True
    ):
        if any(
            embedded_record.get(field) != old_record.get(field)
            for field in required_fields
        ):
            raise ValueError(
                "Functional-group panel historical FreeSolv-10 record drifted: "
                f"{old_record['compound_id']}."
            )


def _validate_dataset_identity(manifest: Mapping[str, object]) -> None:
    identity = manifest.get("dataset_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("FreeSolv functional-group dataset_identity must be an object.")
    expected = {
        "name": "FreeSolv",
        "version": "0.52",
        "repository_commit": "6c7d19b4b565537365ffd22006aa2cd4643200c6",
        "database_txt_sha256": FREESOLV_DATABASE_TXT_SHA256,
        "database_json_sha256": FREESOLV_DATABASE_JSON_SHA256,
        "mol2_archive_sha256": FREESOLV_MOL2_ARCHIVE_SHA256,
        "solvent": "water",
        "standard_state": "the original FreeSolv experimental record convention",
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("FreeSolv functional-group dataset identity changed.")


def load_freesolv12_functional_group_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load a hash-locked panel with ten real functional groups, fail closed."""

    path = Path(manifest_path)
    manifest = _load_json_object(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "FreeSolv functional-group manifest requires schema_version=1."
        )
    if manifest.get("panel_id") != FUNCTIONAL_GROUP_PANEL_ID:
        raise ValueError("FreeSolv functional-group panel has an unexpected panel_id.")
    _validate_dataset_identity(manifest)

    records = manifest.get("locked_records")
    if not isinstance(records, list):
        raise TypeError("FreeSolv functional-group locked_records must be a list.")
    records_sha256 = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    if (
        manifest.get("locked_records_canonical_sha256") != LOCKED_RECORDS_SHA256
        or records_sha256 != LOCKED_RECORDS_SHA256
    ):
        raise ValueError("FreeSolv functional-group record identity changed.")

    observed_ids = tuple(
        record.get("compound_id") if isinstance(record, dict) else None
        for record in records
    )
    if observed_ids != FUNCTIONAL_GROUP_PANEL_IDS:
        raise ValueError("FreeSolv functional-group membership or order changed.")
    observed_labels = tuple(
        record.get("functional_group") if isinstance(record, dict) else None
        for record in records
    )
    if observed_labels != FUNCTIONAL_GROUP_PANEL_LABELS:
        raise ValueError("FreeSolv functional-group labels changed.")
    observed_controls = tuple(
        record["compound_id"]
        for record in records
        if isinstance(record, dict) and record.get("functional_group") is None
    )
    if observed_controls != FUNCTIONAL_GROUP_PANEL_NONFUNCTIONAL_CONTROL_IDS:
        raise ValueError("FreeSolv nonfunctional control identity changed.")
    if len({label for label in observed_labels if label is not None}) != (
        MINIMUM_DISTINCT_FUNCTIONAL_GROUPS
    ):
        raise ValueError("FreeSolv functional-group diversity changed.")
    _require_historical_embedding(records)

    historical = manifest.get("historical_regression_embedding")
    if not isinstance(historical, Mapping) or (
        historical.get("historical_panel_id")
        != "route2-v0-historical-freesolv10-regression-v1"
        or historical.get("historical_outlier_compound_id")
        != HISTORICAL_WORST_RECORD_ID
        or _finite_real(
            historical.get("historical_outlier_absolute_error_kcal_mol"),
            name="historical outlier absolute error",
        )
        != HISTORICAL_WORST_ABSOLUTE_ERROR_KCAL_MOL
    ):
        raise ValueError(
            "FreeSolv functional-group historical regression binding changed."
        )

    gate = manifest.get("acceptance_gate")
    if not isinstance(gate, Mapping):
        raise TypeError("FreeSolv functional-group acceptance_gate must be an object.")
    expected_gate = {
        "expected_record_count": len(FUNCTIONAL_GROUP_PANEL_IDS),
        "functional_group_field": "functional_group",
        "minimum_distinct_functional_groups": MINIMUM_DISTINCT_FUNCTIONAL_GROUPS,
        "functional_group_diversity_is_mandatory": True,
        "nonfunctional_controls_do_not_count_toward_functional_group_minimum": True,
        "historical_freesolv10_regression_is_embedded": True,
        "all_locked_records_required": True,
        "extra_records_fail": True,
        "maximum_absolute_error_threshold_kcal_mol": (
            STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL
        ),
        "comparison": "strictly_less_than",
        "all_records_must_individually_satisfy_threshold": True,
        "mae_or_rmse_alone_is_never_sufficient": True,
        "missing_record_is_failure": True,
        "historical_maximum_error_record_must_remain_included": (
            HISTORICAL_WORST_RECORD_ID
        ),
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise ValueError("FreeSolv functional-group acceptance gate changed.")
    return manifest


def _coerce_predictions(
    predictions: Iterable[Mapping[str, object]],
    *,
    expected_ids: tuple[str, ...],
) -> dict[str, float]:
    if isinstance(predictions, (str, bytes)):
        raise TypeError(
            "FreeSolv functional-group predictions must be record mappings."
        )

    observed: dict[str, float] = {}
    for index, record in enumerate(predictions):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"FreeSolv functional-group prediction record {index} must be a mapping."
            )
        compound_id = record.get("compound_id")
        if not isinstance(compound_id, str) or not compound_id:
            raise ValueError(
                f"FreeSolv functional-group prediction record {index} lacks compound_id."
            )
        if compound_id in observed:
            raise ValueError(
                f"Duplicate FreeSolv functional-group prediction: {compound_id}."
            )
        observed[compound_id] = _finite_real(
            record.get("predicted_kcal_mol"),
            name=f"predicted_kcal_mol for {compound_id}",
        )

    expected = set(expected_ids)
    observed_ids = set(observed)
    missing = sorted(expected - observed_ids)
    extra = sorted(observed_ids - expected)
    if missing or extra:
        raise ValueError(
            "FreeSolv functional-group prediction membership mismatch: "
            f"missing={missing}, extra={extra}."
        )
    return observed


def evaluate_freesolv12_functional_group_predictions(
    predictions: Iterable[Mapping[str, object]],
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> FreeSolvFunctionalGroupPanelEvaluation:
    """Recompute every error; only all twelve under 1.5 kcal/mol can pass."""

    path = Path(manifest_path)
    manifest = load_freesolv12_functional_group_manifest(path)
    records = manifest["locked_records"]
    expected_ids = tuple(record["compound_id"] for record in records)
    predicted_by_id = _coerce_predictions(predictions, expected_ids=expected_ids)
    threshold = _finite_real(
        manifest["acceptance_gate"]["maximum_absolute_error_threshold_kcal_mol"],
        name="FreeSolv functional-group threshold",
    )

    evaluated: list[FreeSolvFunctionalGroupRecordEvaluation] = []
    for record in records:
        compound_id = str(record["compound_id"])
        experimental = _finite_real(
            record["experimental_kcal_mol"],
            name=f"experimental_kcal_mol for {compound_id}",
        )
        predicted = predicted_by_id[compound_id]
        signed_error = predicted - experimental
        absolute_error = abs(signed_error)
        functional_group = record["functional_group"]
        evaluated.append(
            FreeSolvFunctionalGroupRecordEvaluation(
                compound_id=compound_id,
                name=str(record["name"]),
                functional_group=(
                    str(functional_group) if functional_group is not None else None
                ),
                is_nonfunctional_control=functional_group is None,
                experimental_kcal_mol=experimental,
                predicted_kcal_mol=predicted,
                signed_error_kcal_mol=signed_error,
                absolute_error_kcal_mol=absolute_error,
                passes_strict_threshold=absolute_error < threshold,
            )
        )

    worst = max(evaluated, key=lambda item: item.absolute_error_kcal_mol)
    return FreeSolvFunctionalGroupPanelEvaluation(
        panel_id=str(manifest["panel_id"]),
        manifest_sha256=_sha256_file(path),
        record_count=len(evaluated),
        distinct_functional_group_count=len(
            {record.functional_group for record in evaluated if record.functional_group}
        ),
        nonfunctional_control_count=sum(
            record.is_nonfunctional_control for record in evaluated
        ),
        historical_freesolv10_regression_embedded=True,
        threshold_kcal_mol=threshold,
        maximum_absolute_error_kcal_mol=worst.absolute_error_kcal_mol,
        worst_compound_id=worst.compound_id,
        strict_all_records_under_threshold=all(
            record.passes_strict_threshold for record in evaluated
        ),
        records=tuple(evaluated),
    )


def _read_prediction_payload(path: Path) -> list[Mapping[str, object]]:
    payload = _load_json_object(path)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise TypeError("Prediction input must contain a predictions list.")
    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSON object containing compound_id/predicted_kcal_mol prediction records.",
    )
    parser.add_argument(
        "--output", required=True, help="Path for the full per-record report."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Immutable twelve-record functional-group manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = evaluate_freesolv12_functional_group_predictions(
        _read_prediction_payload(Path(args.predictions)),
        manifest_path=Path(args.manifest),
    )
    write_json_atomic(args.output, evaluation.as_dict())
    print(
        f"FreeSolv functional-group-12 {evaluation.status}: "
        f"max_abs_error={evaluation.maximum_absolute_error_kcal_mol:.12g} kcal/mol, "
        f"worst={evaluation.worst_compound_id}."
    )
    return 0 if evaluation.strict_all_records_under_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
