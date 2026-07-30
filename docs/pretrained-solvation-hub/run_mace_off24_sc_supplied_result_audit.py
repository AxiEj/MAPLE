#!/usr/bin/env python3
"""Audit the official MACE-OFF24-SC supporting tables without model inference.

The published checkpoint is not publicly pinned, so this script verifies two
exact official supporting-information files and recomputes only their rounded
table statistics.  It does not deserialize a model, rerun the alchemical
protocol, or turn supplied results into a MAPLE runtime benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PAPER_DOI = "10.1021/jacs.5c10940"
PROTOCOL_SOURCE_REVISION = "c3056287622ba18f9b905e9df45affdff46cb147"

HYDRATION_CSV_SIZE_BYTES = 2650
HYDRATION_CSV_SHA256 = (
    "0b44cb14b4d1c97413c7f6fad45655ddf2b15b44370fa0f89b049eb9af19ff6a"
)
HYDRATION_CSV_MD5 = "d614cd9c6385195484eb6ed84c385a67"
HYDRATION_FIGSHARE_ARTICLE_ID = 31158975
HYDRATION_FIGSHARE_FILE_ID = 61402002
HYDRATION_FIGSHARE_DOI = "10.1021/jacs.5c10940.s002"
HYDRATION_CSV_URL = "https://ndownloader.figshare.com/files/61402002"

SUPPORTING_PDF_SIZE_BYTES = 1704085
SUPPORTING_PDF_SHA256 = (
    "7c675c6657a3b5b92a317ea8a3c17241f5738cc44113440caefe882227b535b1"
)
SUPPORTING_PDF_MD5 = "493ed524b036ebe1055bb144bcbc9335"
SUPPORTING_FIGSHARE_ARTICLE_ID = 31158972
SUPPORTING_FIGSHARE_FILE_ID = 61401999
SUPPORTING_FIGSHARE_DOI = "10.1021/jacs.5c10940.s001"
SUPPORTING_PDF_URL = "https://ndownloader.figshare.com/files/61401999"

FIGSHARE_LICENSE = "CC BY-NC 4.0"
MAXIMUM_ERROR_TARGET_KCAL_MOL = 1.5

# Each row error is the difference of two nominally two-decimal values.  Nearest
# rounding can move that error by at most 0.01 kcal/mol; even truncating both
# values moves it by less than 0.02.  MAE and RMSE are 1-Lipschitz in this
# per-record perturbation.  The two-decimal Table 1 aggregate adds at most
# 0.005 under nearest rounding or less than 0.01 under truncation.
ROW_VALUE_NEAREST_ROUNDING_BOUND_KCAL_MOL = 0.01
ROW_VALUE_CONSERVATIVE_TRUNCATION_BOUND_KCAL_MOL = 0.02
REPORTED_AGGREGATE_NEAREST_ROUNDING_BOUND_KCAL_MOL = 0.005
REPORTED_AGGREGATE_CONSERVATIVE_TRUNCATION_BOUND_KCAL_MOL = 0.01
COMBINED_CONSERVATIVE_DISPLAY_BOUND_KCAL_MOL = (
    ROW_VALUE_CONSERVATIVE_TRUNCATION_BOUND_KCAL_MOL
    + REPORTED_AGGREGATE_CONSERVATIVE_TRUNCATION_BOUND_KCAL_MOL
)
ARTICLE_TABLE_1_HYDRATION_METRICS = {
    "mae_kcal_mol": 0.69,
    "rmse_kcal_mol": 0.80,
}

HYDRATION_HEADER = [
    "Compound Name",
    "SMILES",
    "Exp",
    "MACE",
    "GAFF",
    "OpenFF",
    "Exp Error",
    "MACE Error",
    "GAFF Error",
    "OpenFF Error",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path, *, label: str, expected_size: int, expected_sha256: str
) -> None:
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label} size does not match the pinned official file.")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{label} SHA256 does not match the pinned official file.")


def _finite_float(value: str, *, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value in {context}: {value!r}")
    return parsed


def _read_hydration_csv(text: str) -> list[dict[str, float | str]]:
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("The official hydration CSV is empty.") from exc
    if header != HYDRATION_HEADER:
        raise ValueError(f"Unexpected hydration CSV header: {header!r}")

    records: list[dict[str, float | str]] = []
    identities: set[tuple[str, str]] = set()
    for line_number, fields in enumerate(reader, start=2):
        if not fields or not any(field.strip() for field in fields):
            continue
        if len(fields) < len(HYDRATION_HEADER):
            raise ValueError(
                f"Hydration CSV line {line_number} has fewer than ten fields."
            )

        # Two compound names contain unquoted commas in the official CSV.
        # Parsing from the right preserves the nine unambiguous trailing fields.
        name = ",".join(fields[:-9]).strip()
        smiles = fields[-9].strip()
        numeric = [
            _finite_float(value.strip(), context=f"hydration line {line_number}")
            for value in fields[-8:]
        ]
        identity = (name, smiles)
        if not name or not smiles or identity in identities:
            raise ValueError(
                f"Invalid or duplicate hydration identity on line {line_number}."
            )
        identities.add(identity)
        records.append(
            {
                "name": name,
                "smiles": smiles,
                "experimental_kcal_mol": numeric[0],
                "predicted_kcal_mol": numeric[1],
            }
        )

    if len(records) != 36:
        raise ValueError(
            "Pinned official hydration panel changed; expected exactly 36 records."
        )
    return records


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The audit requires the system `pdftotext` executable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or "unknown pdftotext failure"
        raise RuntimeError(f"Could not extract the supporting PDF: {detail}") from exc
    return completed.stdout


def _read_octanol_table(text: str) -> list[dict[str, float | str]]:
    start_marker = "Table S2: Solvation free energies in kcal/mol in octanol"
    end_marker = "S3        FreeSolv hydration free energy results"
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise ValueError("Could not locate the exact Table S2 octanol section.")

    records: list[dict[str, float | str]] = []
    identities: set[tuple[str, str]] = set()
    for line in text[start:end].splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            numeric = [float(value) for value in fields[-5:]]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Table S2 contains a non-finite numeric value.")

        name = " ".join(fields[:-6]).strip()
        smiles = fields[-6].strip()
        identity = (name, smiles)
        if not name or not smiles or identity in identities:
            raise ValueError("Table S2 contains an invalid or duplicate identity.")
        identities.add(identity)
        records.append(
            {
                "name": name,
                "smiles": smiles,
                "experimental_kcal_mol": numeric[0],
                "predicted_kcal_mol": numeric[2],
            }
        )

    if len(records) != 10:
        raise ValueError(
            "Pinned official octanol panel changed; expected exactly 10 records."
        )
    return records


def _pearson(left: list[float], right: list[float]) -> float:
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


def _metrics(
    records: list[dict[str, float | str]],
) -> tuple[dict[str, float], dict[str, float | str]]:
    experimental = [float(record["experimental_kcal_mol"]) for record in records]
    predicted = [float(record["predicted_kcal_mol"]) for record in records]
    errors = [
        prediction - reference for reference, prediction in zip(experimental, predicted)
    ]
    absolute = [abs(error) for error in errors]
    maximum_index = max(range(len(records)), key=absolute.__getitem__)
    maximum_record = records[maximum_index]
    return (
        {
            "mae_kcal_mol": sum(absolute) / len(absolute),
            "rmse_kcal_mol": math.sqrt(
                sum(error * error for error in errors) / len(errors)
            ),
            "maximum_absolute_error_kcal_mol": absolute[maximum_index],
            "mean_signed_error_kcal_mol": sum(errors) / len(errors),
            "pearson_r": _pearson(experimental, predicted),
        },
        {
            "record_name": str(maximum_record["name"]),
            "absolute_error_kcal_mol": absolute[maximum_index],
        },
    )


def _record_set_sha256(records: list[dict[str, float | str]]) -> str:
    canonical = "\n".join(f"{record['name']}\t{record['smiles']}" for record in records)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _panel_summary(
    records: list[dict[str, float | str]],
    *,
    dataset: str,
    solvent: str,
    evidence_identity: dict[str, Any],
) -> dict[str, Any]:
    metrics, maximum_record = _metrics(records)
    return {
        "dataset": dataset,
        "solvent": solvent,
        "record_count": len(records),
        "selection": "records selected and supplied by the article authors",
        "record_set_sha256": _record_set_sha256(records),
        "source_value_precision": "rounded supporting-information table values",
        "training_overlap_status": "overlap_unknown",
        "record_accounting_complete": False,
        "strict_independent_extrapolation": False,
        "evidence_identity_complete": False,
        "evidence_identity": evidence_identity,
        "metrics": metrics,
        "maximum_error_record": maximum_record,
        "passes_maximum_error_target": (
            metrics["maximum_absolute_error_kcal_mol"] < MAXIMUM_ERROR_TARGET_KCAL_MOL
        ),
    }


def _article_table_1_hydration_comparison(
    recomputed_metrics: dict[str, float],
) -> dict[str, Any]:
    metric_gaps = {
        metric: recomputed_metrics[metric] - reported_value
        for metric, reported_value in ARTICLE_TABLE_1_HYDRATION_METRICS.items()
    }
    reproduction = {
        metric: math.isclose(recomputed_metrics[metric], reported_value, abs_tol=5e-3)
        for metric, reported_value in ARTICLE_TABLE_1_HYDRATION_METRICS.items()
    }
    rounding_alone_can_explain = all(
        abs(gap) <= COMBINED_CONSERVATIVE_DISPLAY_BOUND_KCAL_MOL
        for gap in metric_gaps.values()
    )
    return {
        "reported_metrics": ARTICLE_TABLE_1_HYDRATION_METRICS,
        "rounded_csv_recomputed_metrics_match": reproduction,
        "metric_gaps_recomputed_minus_reported_kcal_mol": metric_gaps,
        "rounding_bounds_kcal_mol": {
            "row_values_nearest_rounding_aggregate_shift_upper_bound": (
                ROW_VALUE_NEAREST_ROUNDING_BOUND_KCAL_MOL
            ),
            "row_values_conservative_truncation_aggregate_shift_upper_bound": (
                ROW_VALUE_CONSERVATIVE_TRUNCATION_BOUND_KCAL_MOL
            ),
            "reported_aggregate_nearest_rounding_upper_bound": (
                REPORTED_AGGREGATE_NEAREST_ROUNDING_BOUND_KCAL_MOL
            ),
            "reported_aggregate_conservative_truncation_upper_bound": (
                REPORTED_AGGREGATE_CONSERVATIVE_TRUNCATION_BOUND_KCAL_MOL
            ),
            "combined_conservative_display_upper_bound": (
                COMBINED_CONSERVATIVE_DISPLAY_BOUND_KCAL_MOL
            ),
        },
        "rounding_alone_can_explain": rounding_alone_can_explain,
        "interpretation": (
            "both absolute metric gaps exceed the conservative 0.03 kcal/mol "
            "combined displayed-value bound, so rounded-table precision alone "
            "cannot explain the discrepancy; the underlying source-analysis "
            "difference remains unknown"
        ),
    }


def audit_supplied_results(csv_path: Path, pdf_path: Path) -> dict[str, Any]:
    """Verify the official SI files and recompute their rounded table metrics."""

    _verify_file(
        csv_path,
        label="MACE-OFF24-SC hydration CSV",
        expected_size=HYDRATION_CSV_SIZE_BYTES,
        expected_sha256=HYDRATION_CSV_SHA256,
    )
    _verify_file(
        pdf_path,
        label="MACE-OFF24-SC supporting PDF",
        expected_size=SUPPORTING_PDF_SIZE_BYTES,
        expected_sha256=SUPPORTING_PDF_SHA256,
    )

    hydration = _panel_summary(
        _read_hydration_csv(csv_path.read_text(encoding="utf-8-sig")),
        dataset="FreeSolv selected 36-record article panel",
        solvent="water",
        evidence_identity={
            "quantity": "absolute hydration free energy",
            "solvent_state": "explicit pure water",
            "temperature_kelvin": None,
            "temperature_status": (
                "not explicitly identified in the audited supplied-result table"
            ),
            "standard_state": None,
            "standard_state_status": (
                "not explicitly identified in the audited supplied-result table"
            ),
            "protonation_policy": (
                "article selection required unambiguous neutrality at pH 7 "
                "according to Schrodinger Epik"
            ),
            "tautomer_policy": "unknown from the supplied-result table",
            "conformer_policy": "unknown from the supplied-result table",
            "potential_identity": (
                "article MACE-OFF24-SC checkpoint; exact public checkpoint "
                "identity unavailable"
            ),
            "cavity_or_pmf": (
                "explicit-solvent soft-core lambda Hamiltonian; no continuum cavity"
            ),
            "sampling_protocol": (
                "16-replica Hamiltonian replica-exchange molecular dynamics over "
                "lambda windows; not rerun by this audit"
            ),
            "estimator": "MBAR; not rerun by this audit",
            "experimental_provenance": {
                "dataset": "FreeSolv",
                "exact_release": None,
                "status": (
                    "dataset named by the article; exact release and record "
                    "identifiers are not stated in the supplied-result table"
                ),
            },
        },
    )
    octanol = _panel_summary(
        _read_octanol_table(_extract_pdf_text(pdf_path)),
        dataset="MNSol selected 10-record article panel",
        solvent="1-octanol",
        evidence_identity={
            "quantity": "absolute solvation free energy in 1-octanol",
            "solvent_state": "explicit pure 1-octanol",
            "temperature_kelvin": None,
            "temperature_status": (
                "not explicitly identified in the audited supplied-result table"
            ),
            "standard_state": None,
            "standard_state_status": (
                "not explicitly identified in the audited supplied-result table"
            ),
            "protonation_policy": "unknown from the supplied-result table",
            "tautomer_policy": "unknown from the supplied-result table",
            "conformer_policy": "unknown from the supplied-result table",
            "potential_identity": (
                "article MACE-OFF24-SC checkpoint; exact public checkpoint "
                "identity unavailable"
            ),
            "cavity_or_pmf": (
                "explicit-solvent soft-core lambda Hamiltonian; no continuum cavity"
            ),
            "sampling_protocol": (
                "16-replica Hamiltonian replica-exchange molecular dynamics over "
                "lambda windows; not rerun by this audit"
            ),
            "estimator": "MBAR; not rerun by this audit",
            "experimental_provenance": {
                "dataset": "MNSol",
                "exact_release": None,
                "status": (
                    "dataset named by the article; exact release and record "
                    "identifiers are not stated in the supplied-result table"
                ),
            },
        },
    )

    reasons = [
        "the public MACE-OFF24-SC checkpoint is not pinned or released for "
        "reproduction",
        "the audit recomputes rounded supplied tables rather than model inference",
        "training overlap and record-complete accounting are unavailable",
        "neither panel is a never-used independent experimental extrapolation set",
    ]
    if not hydration["passes_maximum_error_target"]:
        reasons.append(
            "the hydration panel maximum absolute error exceeds the strict "
            "1.5 kcal/mol Route 4 target"
        )

    hydration_comparison = _article_table_1_hydration_comparison(hydration["metrics"])
    if not all(hydration_comparison["rounded_csv_recomputed_metrics_match"].values()):
        reasons.append(
            "the exact rounded official hydration CSV does not reproduce both "
            "article Table 1 aggregate metrics"
        )
    if not hydration_comparison["rounding_alone_can_explain"]:
        reasons.append(
            "the hydration aggregate gaps exceed the conservative rounding "
            "bound, so rounded-table precision alone cannot explain them"
        )

    return {
        "schema_version": 1,
        "model_id": "mace-off24-sc",
        "evaluation_type": (
            "static_official_supporting_information_audit_no_checkpoint_inference"
        ),
        "acceptance_eligible": False,
        "benchmark_status": (
            "official rounded supporting-table audit only; not a MAPLE runtime "
            "benchmark, checkpoint reproduction, strict holdout, or admission result"
        ),
        "provenance": {
            "paper_doi": PAPER_DOI,
            "paper_url": f"https://doi.org/{PAPER_DOI}",
            "protocol_source_repository": "https://github.com/jharrymoore/mace-md",
            "protocol_source_revision": PROTOCOL_SOURCE_REVISION,
            "artifacts": {
                "hydration_csv": {
                    "file_name": "ja5c10940_si_002.csv",
                    "figshare_article_id": HYDRATION_FIGSHARE_ARTICLE_ID,
                    "figshare_file_id": HYDRATION_FIGSHARE_FILE_ID,
                    "doi": HYDRATION_FIGSHARE_DOI,
                    "download_url": HYDRATION_CSV_URL,
                    "size_bytes": HYDRATION_CSV_SIZE_BYTES,
                    "sha256": HYDRATION_CSV_SHA256,
                    "figshare_md5": HYDRATION_CSV_MD5,
                    "license": FIGSHARE_LICENSE,
                },
                "supporting_information_pdf": {
                    "file_name": "ja5c10940_si_001.pdf",
                    "figshare_article_id": SUPPORTING_FIGSHARE_ARTICLE_ID,
                    "figshare_file_id": SUPPORTING_FIGSHARE_FILE_ID,
                    "doi": SUPPORTING_FIGSHARE_DOI,
                    "download_url": SUPPORTING_PDF_URL,
                    "size_bytes": SUPPORTING_PDF_SIZE_BYTES,
                    "sha256": SUPPORTING_PDF_SHA256,
                    "figshare_md5": SUPPORTING_PDF_MD5,
                    "license": FIGSHARE_LICENSE,
                },
            },
        },
        "checkpoint_identity": {
            "status": "unreleased_or_unpinned_for_public_reproduction",
            "public_protocol_repository_checkpoint": (
                "MACE-OFF23-SC only; it must not be substituted for MACE-OFF24-SC"
            ),
        },
        "quantity_boundary": {
            "source": "rounded official supporting-information tables",
            "checkpoint_inference_performed": False,
            "alchemical_protocol_rerun": False,
            "reported_uncertainties_recomputed": False,
            "full_precision_author_values_available": False,
        },
        "validation_panels": {
            "hydration": hydration,
            "octanol": octanol,
        },
        "article_table_1_hydration_comparison": hydration_comparison,
        "acceptance_gate": {
            "maximum_absolute_error_target_kcal_mol": (MAXIMUM_ERROR_TARGET_KCAL_MOL),
            "overall_passes": False,
            "reasons": reasons,
        },
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
        "--csv", required=True, type=Path, help="Pinned official hydration CSV."
    )
    parser.add_argument(
        "--pdf", required=True, type=Path, help="Pinned official supporting PDF."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output aggregate JSON artifact."
    )
    args = parser.parse_args()
    payload = audit_supplied_results(
        args.csv.expanduser().resolve(), args.pdf.expanduser().resolve()
    )
    _write_json_atomic(args.output.expanduser().resolve(), payload)
    hydration = payload["validation_panels"]["hydration"]
    octanol = payload["validation_panels"]["octanol"]
    print(
        "MACE_OFF24_SC_SUPPLIED_RESULT_AUDIT=PASS "
        f"hydration_records={hydration['record_count']} "
        f"hydration_mae={hydration['metrics']['mae_kcal_mol']:.12f} "
        f"octanol_records={octanol['record_count']} "
        f"octanol_mae={octanol['metrics']['mae_kcal_mol']:.12f}"
    )


if __name__ == "__main__":
    main()
