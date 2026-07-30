#!/usr/bin/env python3
"""Audit the supplied G-NequIP SMD-water result table without model inference.

The official Zenodo archive contains precomputed gas/SMD NNP energies and an
experimental FreeSolv comparison.  This script verifies that immutable archive
and recomputes the published central statistics.  It deliberately does *not*
deserialize a checkpoint, construct a NequIP calculator, or treat an SMD-minus-
gas single-point difference as a MAPLE absolute-solvation calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ARCHIVE_SIZE_BYTES = 474332
ARCHIVE_SHA256 = "76522218fcadc921706ba489ab0b7ff1c5e28b335029f07ac7c5ce5877dae055"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/20690503/files/"
    "solvation_free_energies.zip/content"
)
SOURCE_REVISION = "c6d43de9630c98486ff7ed904c4c54cdeb7e14ad"
GNEQUIP_SMDW_SHA256 = "5615c0002a24ee3485368b59721198b2ca2e0432c1c6f7b73b5a31f6fe662484"
EV_TO_KCAL_MOL = 23.060548

ROOT = "solvation_free_energies"
EXPERIMENTAL_MEMBER = f"{ROOT}/experimental/experimental_sfe.csv"
GAS_MEMBER = f"{ROOT}/NNP_energies/gas_geom_gas_NNP.csv"
SMD_MEMBER = f"{ROOT}/NNP_energies/SMD_geom_SMD_NNP.csv"
SUMMARY_MEMBER = f"{ROOT}/results/bootstrap_metrics_summary.csv"
README_MEMBER = f"{ROOT}/README.md"
REQUIRED_MEMBERS = (
    EXPERIMENTAL_MEMBER,
    GAS_MEMBER,
    SMD_MEMBER,
    SUMMARY_MEMBER,
    README_MEMBER,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(archive: zipfile.ZipFile, member: str) -> str:
    try:
        return archive.read(member).decode("utf-8")
    except KeyError as exc:
        raise ValueError(
            f"Required upstream archive member is missing: {member}"
        ) from exc


def _read_experimental(text: str) -> dict[str, float]:
    rows = csv.reader(
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    values: dict[str, float] = {}
    for row in rows:
        if len(row) != 2:
            raise ValueError("Experimental SFE rows must contain exactly two columns.")
        record_id = row[0].strip()
        if not record_id or record_id in values:
            raise ValueError(
                f"Invalid or duplicate experimental record ID: {record_id!r}"
            )
        try:
            value = float(row[1])
        except ValueError as exc:
            raise ValueError(f"Invalid experimental energy for {record_id!r}.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite experimental energy for {record_id!r}.")
        values[record_id] = value
    return values


def _read_nnp_energy(text: str, *, member: str) -> dict[str, float]:
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames != ["Name", "NNP_Energy"]:
        raise ValueError(
            f"Unexpected NNP-energy columns in {member}: {reader.fieldnames!r}"
        )
    values: dict[str, float] = {}
    for row in reader:
        record_id = str(row["Name"]).strip()
        if not record_id or record_id in values:
            raise ValueError(
                f"Invalid or duplicate NNP-energy record ID: {record_id!r}"
            )
        try:
            value = float(str(row["NNP_Energy"]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid NNP energy for {record_id!r} in {member}."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite NNP energy for {record_id!r} in {member}.")
        values[record_id] = value
    return values


def _read_reported_panel_b(text: str) -> dict[str, float]:
    reader = csv.DictReader(text.splitlines())
    required = {"panel", "metric", "value", "ci_2.5", "ci_97.5"}
    if set(reader.fieldnames or ()) != required:
        raise ValueError("Unexpected upstream bootstrap-summary columns.")
    values: dict[str, float] = {}
    for row in reader:
        if row["panel"] == "b)":
            values[str(row["metric"])] = float(str(row["value"]))
    if set(values) != {"R", "RMSE", "MAE"}:
        raise ValueError(
            "Upstream bootstrap summary lacks a complete experimental panel b."
        )
    return values


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        raise ValueError("At least two records are required for Pearson correlation.")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    if denominator == 0.0:
        raise ValueError("Cannot calculate Pearson correlation for a constant panel.")
    return numerator / denominator


def _metrics(records: list[dict[str, float | str]]) -> dict[str, float]:
    experimental = [float(record["experimental_kcal_mol"]) for record in records]
    predicted = [float(record["predicted_kcal_mol"]) for record in records]
    errors = [
        prediction - reference for reference, prediction in zip(experimental, predicted)
    ]
    absolute = [abs(error) for error in errors]
    return {
        "mae_kcal_mol": sum(absolute) / len(absolute),
        "rmse_kcal_mol": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "maximum_absolute_error_kcal_mol": max(absolute),
        "mean_signed_error_kcal_mol": sum(errors) / len(errors),
        "pearson_r": _pearson(experimental, predicted),
    }


def audit_supplied_results(archive_path: Path) -> dict[str, Any]:
    """Validate one exact upstream archive and recompute its published panel b."""

    if archive_path.stat().st_size != ARCHIVE_SIZE_BYTES:
        raise ValueError(
            "G-NequIP supplied-results archive size does not match the pinned Zenodo file."
        )
    actual_sha256 = _sha256(archive_path)
    if actual_sha256 != ARCHIVE_SHA256:
        raise ValueError(
            "G-NequIP supplied-results archive SHA256 does not match the pinned Zenodo file."
        )

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("G-NequIP supplied-results archive has a corrupt member.")
        texts = {member: _read_text(archive, member) for member in REQUIRED_MEMBERS}

    if "ΔG_solv is estimated as the difference" not in texts[README_MEMBER]:
        raise ValueError(
            "Upstream README no longer declares the supplied difference protocol."
        )

    experimental = _read_experimental(texts[EXPERIMENTAL_MEMBER])
    gas = _read_nnp_energy(texts[GAS_MEMBER], member=GAS_MEMBER)
    smd = _read_nnp_energy(texts[SMD_MEMBER], member=SMD_MEMBER)
    record_ids = sorted(set(experimental).intersection(gas).intersection(smd))
    if len(record_ids) != 388:
        raise ValueError(
            "Pinned upstream intersection changed; expected exactly 388 evaluable records."
        )

    records: list[dict[str, float | str]] = []
    for record_id in record_ids:
        prediction = (smd[record_id] - gas[record_id]) * EV_TO_KCAL_MOL
        records.append(
            {
                "record_id": record_id,
                "experimental_kcal_mol": experimental[record_id],
                "predicted_kcal_mol": prediction,
                "gas_nnp_energy_ev": gas[record_id],
                "smd_nnp_energy_ev": smd[record_id],
            }
        )

    metrics = _metrics(records)
    reported = _read_reported_panel_b(texts[SUMMARY_MEMBER])
    reported_matches = {
        "pearson_r": math.isclose(metrics["pearson_r"], reported["R"], abs_tol=1e-8),
        "rmse_kcal_mol": math.isclose(
            metrics["rmse_kcal_mol"], reported["RMSE"], abs_tol=1e-8
        ),
        "mae_kcal_mol": math.isclose(
            metrics["mae_kcal_mol"], reported["MAE"], abs_tol=1e-8
        ),
    }
    if not all(reported_matches.values()):
        raise ValueError(
            "Recomputed supplied G-NequIP statistics do not match panel b."
        )

    return {
        "schema_version": 1,
        "model_id": "g-nequip-smdw-water",
        "evaluation_type": "static_upstream_energy_table_audit_no_checkpoint_inference",
        "acceptance_eligible": False,
        "benchmark_status": (
            "upstream-supplied FreeSolv water result reproduction only; not a "
            "MAPLE runtime benchmark, strict holdout, absolute-solvation calculation, "
            "or acceptance result"
        ),
        "provenance": {
            "source_repository": "https://github.com/otayfuroglu/deepPotential",
            "source_revision": SOURCE_REVISION,
            "smdw_checkpoint_sha256": GNEQUIP_SMDW_SHA256,
            "archive_url": ARCHIVE_URL,
            "archive_size_bytes": ARCHIVE_SIZE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_members": list(REQUIRED_MEMBERS),
        },
        "quantity_boundary": {
            "upstream_formula": "(E_SMD_NNP - E_gas_NNP) * 23.060548 kcal/mol",
            "upstream_geometry_policy": "separate supplied gas/SMD geometry tables",
            "temperature_kelvin": None,
            "standard_state": None,
            "energy_gauge": "not established across independently trained gas and SMD models",
            "sampling_estimator": "none; supplied single-point energy difference",
        },
        "validation_panel": {
            "dataset": "FreeSolv selected molecules from supplied Zenodo archive",
            "record_count": len(records),
            "selection": "intersection of supplied experimental, gas-NNP, and SMD-NNP tables",
            "training_overlap_status": "overlap_unknown",
            "record_accounting_complete": False,
        },
        "source_panel_b_reproduction": {
            "reported_metrics": {
                "pearson_r": reported["R"],
                "rmse_kcal_mol": reported["RMSE"],
                "mae_kcal_mol": reported["MAE"],
            },
            "recomputed_metrics_match": reported_matches,
        },
        "summary": {"coverage": 1.0, "metrics": metrics},
        "records": records,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", required=True, type=Path, help="Pinned Zenodo result archive."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output JSON audit artifact."
    )
    args = parser.parse_args()
    payload = audit_supplied_results(args.archive.expanduser().resolve())
    _write_json_atomic(args.output.expanduser().resolve(), payload)
    print(
        "GNEQUIP_SMDW_SUPPLIED_RESULT_AUDIT=PASS "
        f"records={payload['validation_panel']['record_count']} "
        f"mae={payload['summary']['metrics']['mae_kcal_mol']:.12f}"
    )


if __name__ == "__main__":
    main()
