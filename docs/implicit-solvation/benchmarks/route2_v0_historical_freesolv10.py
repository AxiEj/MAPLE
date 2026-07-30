#!/usr/bin/env python3
"""Validate the immutable historical FreeSolv-10 V0 regression panel.

The old fixed-charge GTO/QEq screen observed a 7.041442082076966 kcal/mol
absolute error on ethyl acetate.  That old method is not a V0 candidate, but
its exact ten FreeSolv record identities and fixed MOL2 conformers remain a
mandatory regression and chemistry-diversity panel.  Its ten records cover
ten distinct chemical-function or scaffold classes.  A V0 result may pass
this panel only when every one of the ten recomputed errors is *strictly*
below 1.5 kcal/mol.

This module deliberately recomputes errors from frozen experimental labels. It
never trusts a candidate-provided error, MAE, RMSE, or subset declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# This module is also executed directly from this non-package benchmark directory.
from benchmark_core import (  # pyright: ignore[reportImplicitRelativeImport]
    canonical_json_bytes,
    write_json_atomic,
)

HISTORICAL_PANEL_ID = "route2-v0-historical-freesolv10-regression-v1"
HISTORICAL_SOURCE_COMMIT = "c77eb6a295cbb39a1b6e66d21190a192cc8ef050"
HISTORICAL_SOURCE_BLOB_OID = "8a7c5f2ff8b52a4ccbb03f6385efe6db1d2ee2e4"
HISTORICAL_SOURCE_CONTENT_SHA256 = (
    "893ad62e476e6375602d82231def6b0ad22cab60128c8068894daaf4a6ee8c4c"
)
HISTORICAL_LOCKED_RECORDS_SHA256 = (
    "6bedf0b04ee32a31eb7a3acec2b33e03c640a128b328cc7871f2e9afb86e80b6"
)
HISTORICAL_FREE_SOLV10_IDS = (
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
)
# The locked panel deliberately has one representative for each listed
# functional-group or hydrocarbon/scaffold class.  A hydrocarbon has no
# functional group, so the invariant explicitly includes scaffolds rather than
# pretending methane or benzene is a functional group.
HISTORICAL_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES = (
    "alkane",
    "aromatic hydrocarbon",
    "alcohol",
    "ether",
    "ketone",
    "ester",
    "nitrile",
    "aromatic amine",
    "haloalkane",
    "sulfoxide",
)
MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES = 10
HISTORICAL_WORST_RECORD_ID = "mobley_6973347"
HISTORICAL_WORST_ABSOLUTE_ERROR_KCAL_MOL = 7.041442082076966
STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL = 1.5
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "route2-v0-historical-freesolv10-regression-v1.json"
)


@dataclass(frozen=True)
class HistoricalFreeSolv10RecordEvaluation:
    """One source-bound prediction and error on the historical panel."""

    compound_id: str
    name: str
    chemical_class: str
    experimental_kcal_mol: float
    predicted_kcal_mol: float
    signed_error_kcal_mol: float
    absolute_error_kcal_mol: float
    passes_strict_threshold: bool


@dataclass(frozen=True)
class HistoricalFreeSolv10PanelEvaluation:
    """Complete all-record acceptance decision for the locked panel."""

    panel_id: str
    manifest_sha256: str
    record_count: int
    threshold_kcal_mol: float
    distinct_functional_group_or_scaffold_class_count: int
    maximum_absolute_error_kcal_mol: float
    worst_compound_id: str
    strict_all_records_under_threshold: bool
    records: tuple[HistoricalFreeSolv10RecordEvaluation, ...]

    @property
    def status(self) -> str:
        return "pass" if self.strict_all_records_under_threshold else "fail"

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, fully per-record report for an external artifact."""

        return {
            "schema_version": 1,
            "panel_id": self.panel_id,
            "manifest_sha256": self.manifest_sha256,
            "status": self.status,
            "acceptance": {
                "record_count": self.record_count,
                "minimum_distinct_functional_group_or_scaffold_classes": (
                    MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES
                ),
                "distinct_functional_group_or_scaffold_class_count": (
                    self.distinct_functional_group_or_scaffold_class_count
                ),
                "threshold_kcal_mol": self.threshold_kcal_mol,
                "comparison": "strictly_less_than",
                "maximum_absolute_error_kcal_mol": self.maximum_absolute_error_kcal_mol,
                "worst_compound_id": self.worst_compound_id,
                "strict_all_records_under_threshold": self.strict_all_records_under_threshold,
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
            f"Historical FreeSolv-10 manifest not found: {path}."
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Historical FreeSolv-10 manifest is invalid JSON: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise TypeError("Historical FreeSolv-10 manifest must be a JSON object.")
    return value


def load_historical_freesolv10_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load and fail closed on any mutation of the historical panel identity."""

    path = Path(manifest_path)
    manifest = _load_json_object(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Historical FreeSolv-10 manifest requires schema_version=1.")
    if manifest.get("panel_id") != HISTORICAL_PANEL_ID:
        raise ValueError("Historical FreeSolv-10 manifest has an unexpected panel_id.")

    source = manifest.get("historical_source")
    if not isinstance(source, dict) or (
        source.get("git_commit") != HISTORICAL_SOURCE_COMMIT
        or source.get("git_blob_oid") != HISTORICAL_SOURCE_BLOB_OID
        or source.get("content_sha256") != HISTORICAL_SOURCE_CONTENT_SHA256
    ):
        raise ValueError("Historical FreeSolv-10 source artifact identity changed.")

    outlier = manifest.get("historical_observed_maximum_error")
    if not isinstance(outlier, dict) or (
        outlier.get("compound_id") != HISTORICAL_WORST_RECORD_ID
        or _finite_real(
            outlier.get("absolute_error_kcal_mol"),
            name="historical FreeSolv-10 maximum error",
        )
        != HISTORICAL_WORST_ABSOLUTE_ERROR_KCAL_MOL
    ):
        raise ValueError("Historical FreeSolv-10 outlier identity changed.")

    records = manifest.get("locked_records")
    if not isinstance(records, list):
        raise TypeError("Historical FreeSolv-10 locked_records must be a list.")
    actual_records_sha256 = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    if (
        manifest.get("locked_records_canonical_sha256")
        != HISTORICAL_LOCKED_RECORDS_SHA256
        or actual_records_sha256 != HISTORICAL_LOCKED_RECORDS_SHA256
    ):
        raise ValueError("Historical FreeSolv-10 locked record identity changed.")

    observed_ids = tuple(
        record.get("compound_id") if isinstance(record, dict) else None
        for record in records
    )
    if observed_ids != HISTORICAL_FREE_SOLV10_IDS:
        raise ValueError("Historical FreeSolv-10 record membership or order changed.")

    observed_classes = tuple(
        record.get("chemical_class") if isinstance(record, dict) else None
        for record in records
    )
    if observed_classes != HISTORICAL_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES:
        raise ValueError(
            "Historical FreeSolv-10 functional-group/scaffold diversity changed."
        )
    if (
        len(set(observed_classes))
        != MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES
    ):
        raise ValueError(
            "Historical FreeSolv-10 must retain ten distinct functional-group "
            "or scaffold classes."
        )

    gate = manifest.get("acceptance_gate")
    if not isinstance(gate, dict):
        raise TypeError("Historical FreeSolv-10 acceptance_gate must be an object.")
    if gate.get("expected_record_count") != len(HISTORICAL_FREE_SOLV10_IDS):
        raise ValueError("Historical FreeSolv-10 expected record count changed.")
    if (
        gate.get("functional_group_or_scaffold_field") != "chemical_class"
        or gate.get("minimum_distinct_functional_group_or_scaffold_classes")
        != MINIMUM_DISTINCT_FUNCTIONAL_GROUP_OR_SCAFFOLD_CLASSES
        or gate.get("functional_group_or_scaffold_diversity_is_mandatory") is not True
    ):
        raise ValueError(
            "Historical FreeSolv-10 functional-group/scaffold gate changed."
        )
    if (
        gate.get("maximum_absolute_error_threshold_kcal_mol")
        != STRICT_ABSOLUTE_ERROR_THRESHOLD_KCAL_MOL
    ):
        raise ValueError("Historical FreeSolv-10 strict threshold changed.")
    if gate.get("comparison") != "strictly_less_than":
        raise ValueError("Historical FreeSolv-10 must use a strict maximum-error gate.")
    if not all(
        gate.get(key) is True
        for key in (
            "all_locked_records_required",
            "extra_records_fail",
            "all_records_must_individually_satisfy_threshold",
            "mae_or_rmse_alone_is_never_sufficient",
            "missing_record_is_failure",
        )
    ):
        raise ValueError("Historical FreeSolv-10 all-record acceptance policy changed.")
    if (
        gate.get("historical_maximum_error_record_must_remain_included")
        != HISTORICAL_WORST_RECORD_ID
    ):
        raise ValueError(
            "Historical FreeSolv-10 outlier inclusion requirement changed."
        )

    return manifest


def _coerce_predictions(
    predictions: Iterable[Mapping[str, object]],
    *,
    expected_ids: tuple[str, ...],
) -> dict[str, float]:
    if isinstance(predictions, (str, bytes)):
        raise TypeError("Historical FreeSolv-10 predictions must be record mappings.")

    observed: dict[str, float] = {}
    for index, record in enumerate(predictions):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"Historical FreeSolv-10 prediction record {index} must be a mapping."
            )
        compound_id = record.get("compound_id")
        if not isinstance(compound_id, str) or not compound_id:
            raise ValueError(
                f"Historical FreeSolv-10 prediction record {index} lacks compound_id."
            )
        if compound_id in observed:
            raise ValueError(
                f"Duplicate historical FreeSolv-10 prediction: {compound_id}."
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
            "Historical FreeSolv-10 prediction membership mismatch: "
            f"missing={missing}, extra={extra}."
        )
    return observed


def evaluate_historical_freesolv10_predictions(
    predictions: Iterable[Mapping[str, object]],
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> HistoricalFreeSolv10PanelEvaluation:
    """Evaluate all and only the frozen records against the strict 1.5 gate."""

    path = Path(manifest_path)
    manifest = load_historical_freesolv10_manifest(path)
    records = manifest["locked_records"]
    expected_ids = tuple(record["compound_id"] for record in records)
    predicted_by_id = _coerce_predictions(predictions, expected_ids=expected_ids)
    threshold = _finite_real(
        manifest["acceptance_gate"]["maximum_absolute_error_threshold_kcal_mol"],
        name="historical FreeSolv-10 threshold",
    )

    evaluated: list[HistoricalFreeSolv10RecordEvaluation] = []
    for record in records:
        compound_id = str(record["compound_id"])
        experimental = _finite_real(
            record["experimental_kcal_mol"],
            name=f"experimental_kcal_mol for {compound_id}",
        )
        predicted = predicted_by_id[compound_id]
        signed_error = predicted - experimental
        absolute_error = abs(signed_error)
        evaluated.append(
            HistoricalFreeSolv10RecordEvaluation(
                compound_id=compound_id,
                name=str(record["name"]),
                chemical_class=str(record["chemical_class"]),
                experimental_kcal_mol=experimental,
                predicted_kcal_mol=predicted,
                signed_error_kcal_mol=signed_error,
                absolute_error_kcal_mol=absolute_error,
                passes_strict_threshold=absolute_error < threshold,
            )
        )

    worst = max(evaluated, key=lambda item: item.absolute_error_kcal_mol)
    return HistoricalFreeSolv10PanelEvaluation(
        panel_id=str(manifest["panel_id"]),
        manifest_sha256=_sha256_file(path),
        record_count=len(evaluated),
        threshold_kcal_mol=threshold,
        distinct_functional_group_or_scaffold_class_count=len(
            {record.chemical_class for record in evaluated}
        ),
        maximum_absolute_error_kcal_mol=worst.absolute_error_kcal_mol,
        worst_compound_id=worst.compound_id,
        strict_all_records_under_threshold=all(
            item.passes_strict_threshold for item in evaluated
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
        help="JSON object containing a predictions list of compound_id/predicted_kcal_mol records.",
    )
    parser.add_argument(
        "--output", required=True, help="Path for the full per-record report."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Immutable historical panel manifest (defaults to the tracked manifest).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = evaluate_historical_freesolv10_predictions(
        _read_prediction_payload(Path(args.predictions)),
        manifest_path=Path(args.manifest),
    )
    write_json_atomic(args.output, evaluation.as_dict())
    print(
        f"Historical FreeSolv-10 {evaluation.status}: "
        f"max_abs_error={evaluation.maximum_absolute_error_kcal_mol:.12g} kcal/mol, "
        f"worst={evaluation.worst_compound_id}."
    )
    return 0 if evaluation.strict_all_records_under_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
