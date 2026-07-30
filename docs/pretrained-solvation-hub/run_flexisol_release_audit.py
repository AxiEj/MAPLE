#!/usr/bin/env python3
"""Audit the pinned FlexiSol release without consuming it as a model score.

The audit reads exact Git blobs for the public experimental-reference tables
and method registry, verifies their identity, and emits only coverage and
provenance summaries.  It never reads model predictions, computes model
errors, selects a model, or turns FlexiSol into MAPLE's final blind holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

AUDITED_ON = "2026-07-30"
PAPER_DOI = "10.1039/D5SC06406F"
SOURCE_REPOSITORY = "https://github.com/grimme-lab/flexisol"
SOURCE_REVISION = "7b44798f26c888ef541faa6143a813136921483f"
SOURCE_TREE = "8cf4879bc1435ca1061acfffd0daf10784445984"
SOURCE_COMMIT_DATE = "2026-04-02T07:31:28+02:00"
MINIMUM_REQUIRED_PURE_SOLVENTS = 10

DGSOLV_PATH = "data/references/dgsolv-references.csv"
LOGKAB_PATH = "data/references/logkab-references.csv"
REGISTRY_PATH = "flexisol_cli/registry.json"
README_PATH = "README.md"
LICENSE_PATH = "LICENSE.txt"

DGSOLV_HEADER = [
    "FlexiSol Name",
    "IUPAC Name",
    "Solvent",
    "SMILES",
    r"Value (\kcalpmole)",
    "Ref.",
]
LOGKAB_HEADER = [
    "FlexiSol Name",
    "IUPAC Name",
    "Solvents",
    "SMILES",
    "Value (log units)",
    "Ref.",
]

PINNED_BLOBS = {
    DGSOLV_PATH: {
        "size_bytes": 85_925,
        "sha256": "73a6f25ea9a9fae31bcb7e7c0fb04e452b0df2d24308e25419b98f01daa6ed5f",
    },
    LOGKAB_PATH: {
        "size_bytes": 63_281,
        "sha256": "52c0e0a70b5c57a8d46838d8ab5316a3f754783f6a0a6b1d585d923fc19de626",
    },
    REGISTRY_PATH: {
        "size_bytes": 1_194,
        "sha256": "dd71210cf545612bc4eab8141bdb2ddd146ff1171ee041b76f67ae33258e36ea",
    },
    README_PATH: {
        "size_bytes": 9_474,
        "sha256": "64f64dfbcae4c767e5b66247ae7275453f01d1c84b701678d7896f2518181bfb",
    },
    LICENSE_PATH: {
        "size_bytes": 1_093,
        "sha256": "d682927491968a96bff8adc618af6adb3d92eae54b7a9c3ac80bb685bf684c0f",
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The FlexiSol audit requires the system `git` executable."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git inspection failed for {root}: {detail}") from exc
    return completed.stdout


def _verify_source_identity(root: Path) -> dict[str, str]:
    revision = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git_bytes(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"FlexiSol revision {revision!r} does not match {SOURCE_REVISION!r}."
        )
    if tree != SOURCE_TREE:
        raise ValueError(f"FlexiSol tree {tree!r} does not match {SOURCE_TREE!r}.")
    return {"revision": revision, "tree": tree}


def _read_verified_blob(root: Path, path: str) -> bytes:
    specification = PINNED_BLOBS[path]
    data = _git_bytes(root, "show", f"{SOURCE_REVISION}:{path}")
    if len(data) != specification["size_bytes"]:
        raise ValueError(f"{path} size does not match the pinned Git blob.")
    if _sha256_bytes(data) != specification["sha256"]:
        raise ValueError(f"{path} SHA256 does not match the pinned Git blob.")
    return data


def _read_reference_csv(
    data: bytes,
    *,
    label: str,
    expected_header: list[str],
    solvent_column: str,
    value_column: str,
    expected_rows: int,
) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != expected_header:
        raise ValueError(
            f"Unexpected {label} header: {reader.fieldnames!r}; "
            f"expected {expected_header!r}."
        )

    rows: list[dict[str, str]] = []
    for line_number, raw in enumerate(reader, start=2):
        if None in raw:
            raise ValueError(f"{label} line {line_number} has extra CSV fields.")
        row = {key: (value or "").strip() for key, value in raw.items()}
        missing = [key for key, value in row.items() if not value]
        if missing:
            raise ValueError(
                f"{label} line {line_number} has empty fields: {missing!r}."
            )
        try:
            numeric = float(row[value_column])
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric reference in {label} line {line_number}."
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(
                f"Non-finite numeric reference in {label} line {line_number}."
            )
        if not row[solvent_column]:
            raise ValueError(f"{label} line {line_number} has no solvent identity.")
        rows.append(row)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Pinned {label} changed; expected {expected_rows} rows, "
            f"found {len(rows)}."
        )
    return rows


def _reference_summary(
    rows: list[dict[str, str]], *, solvent_column: str
) -> dict[str, Any]:
    solvent_counts = Counter(row[solvent_column] for row in rows)
    source_dois = sorted({row["Ref."].strip() for row in rows})
    identity_lines = [
        "\t".join(
            (
                row["FlexiSol Name"],
                row["IUPAC Name"],
                row[solvent_column],
                row["SMILES"],
                row["Ref."].strip(),
            )
        )
        for row in rows
    ]
    return {
        "record_count": len(rows),
        "unique_flexisol_name_count": len({row["FlexiSol Name"] for row in rows}),
        "unique_smiles_count": len({row["SMILES"] for row in rows}),
        "unique_record_identity_count": len(
            {
                (
                    row["FlexiSol Name"],
                    row[solvent_column],
                    row["SMILES"],
                    row["Ref."].strip(),
                )
                for row in rows
            }
        ),
        "unique_solvent_or_pair_count": len(solvent_counts),
        "solvent_or_pair_counts": dict(sorted(solvent_counts.items())),
        "source_doi_count": len(source_dois),
        "source_dois": source_dois,
        "ordered_identity_sha256": _sha256_bytes(
            "\n".join(identity_lines).encode("utf-8")
        ),
        "records_embedded": False,
    }


def _registry_summary(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("FlexiSol method registry is not valid JSON.") from exc
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("FlexiSol method registry has no method mapping.")

    grouped: dict[str, list[str]] = {"el": [], "solv": []}
    for name, record in methods.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("FlexiSol method registry contains an invalid entry.")
        method_type = record.get("type")
        method_name = record.get("method")
        if method_type not in grouped or not isinstance(method_name, str):
            raise ValueError(f"FlexiSol method {name!r} has an invalid contract.")
        grouped[method_type].append(name)

    solvation = sorted(grouped["solv"])
    electronic = sorted(grouped["el"])
    already_registered = sorted(
        name for name in ("cigin", "directml") if name in solvation
    )
    return {
        "registered_solvation_methods": solvation,
        "registered_electronic_methods": electronic,
        "route4_models_already_registered": already_registered,
        "model_specific_independence_required": True,
    }


def _verified_blob_record(path: str) -> dict[str, Any]:
    specification = PINNED_BLOBS[path]
    return {
        "size_bytes": specification["size_bytes"],
        "sha256": specification["sha256"],
        "verified": True,
    }


def audit_release(source_root: Path) -> dict[str, Any]:
    """Return a deterministic, static-only FlexiSol eligibility audit."""

    identity = _verify_source_identity(source_root)
    blobs = {path: _read_verified_blob(source_root, path) for path in PINNED_BLOBS}
    readme = blobs[README_PATH].decode("utf-8", errors="strict")
    license_text = blobs[LICENSE_PATH].decode("utf-8", errors="strict")
    for token in ("Correction", "hexadecane", PAPER_DOI):
        if token not in readme:
            raise ValueError(f"FlexiSol README is missing required token {token!r}.")
    if "MIT License" not in license_text:
        raise ValueError("FlexiSol license blob is not the pinned MIT license.")

    dgsolv_rows = _read_reference_csv(
        blobs[DGSOLV_PATH],
        label="DeltaGsolv reference table",
        expected_header=DGSOLV_HEADER,
        solvent_column="Solvent",
        value_column=r"Value (\kcalpmole)",
        expected_rows=530,
    )
    logkab_rows = _read_reference_csv(
        blobs[LOGKAB_PATH],
        label="logKab reference table",
        expected_header=LOGKAB_HEADER,
        solvent_column="Solvents",
        value_column="Value (log units)",
        expected_rows=294,
    )
    dgsolv = _reference_summary(dgsolv_rows, solvent_column="Solvent")
    logkab = _reference_summary(logkab_rows, solvent_column="Solvents")
    registry = _registry_summary(blobs[REGISTRY_PATH])

    pure_solvent_count = dgsolv["unique_solvent_or_pair_count"]
    coverage_passes = pure_solvent_count >= MINIMUM_REQUIRED_PURE_SOLVENTS
    return {
        "schema_version": 1,
        "audited_on": AUDITED_ON,
        "dataset": "FlexiSol",
        "paper": {
            "doi": PAPER_DOI,
            "title": (
                "A diverse and chemically relevant solvation model benchmark "
                "set with flexible molecules and conformer ensembles"
            ),
            "publication_year": 2025,
        },
        "identity": {
            "repository": SOURCE_REPOSITORY,
            "revision": identity["revision"],
            "tree": identity["tree"],
            "commit_date": SOURCE_COMMIT_DATE,
            "source_isolation": "exact_pinned_revision_Git_blobs",
            "license": "MIT",
            "files": {
                path: _verified_blob_record(path) for path in sorted(PINNED_BLOBS)
            },
        },
        "release_boundary": {
            "corrected_hexadecane_references_pinned": True,
            "readme_correction_notice_verified": True,
            "reason_for_pinning_head_instead_of_tag": (
                "the pinned HEAD contains the official post-release "
                "hexadecane reference correction"
            ),
        },
        "panels": {
            "dgsolv": dgsolv,
            "logkab": logkab,
        },
        "registry": registry,
        "coverage_gate": {
            "scope": "pure-solvent DeltaGsolv references",
            "observed_pure_solvents": pure_solvent_count,
            "minimum_required_pure_solvents": MINIMUM_REQUIRED_PURE_SOLVENTS,
            "passes": coverage_passes,
        },
        "independence_gate": {
            "training_overlap_status": "unknown_for_each_candidate_model",
            "published_benchmark_already_registers_route4_models": (
                registry["route4_models_already_registered"]
            ),
            "strict_model_specific_intersection_ledger_present": False,
            "never_used_independent_final_panel_demonstrated": False,
            "passes": False,
        },
        "validation_scope": {
            "static_release_identity_only": True,
            "experimental_reference_values_verified_as_finite": True,
            "model_predictions_read": False,
            "models_scored": [],
            "model_selection_performed": False,
            "maximum_error_computed": False,
            "records_redistributed": False,
        },
        "candidate_external_confirmation": True,
        "strict_final_holdout": False,
        "acceptance_eligible": False,
        "gpu_acceleration": {
            "relevant_to_static_audit": False,
            "gpu_execution_performed": False,
            "no_precision_loss_claim": False,
        },
        "route4_decision": {
            "status": "metadata_only_not_consumed_for_scoring",
            "allowed_future_use": (
                "external confirmation only after exact candidate-specific "
                "training/test overlap accounting"
            ),
            "forbidden_uses": [
                "training",
                "fine_tuning",
                "calibration",
                "model_selection",
                "claiming_a_never_used_final_holdout",
                "claiming_the_ten_solvent_coverage_gate_passes",
            ],
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MAPLE_FLEXISOL_AUDIT_ROOT",
                "/home/axie/.cache/maple-benchmarks/flexisol",
            )
        ),
        help="Exact pinned FlexiSol Git checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "benchmarks"
            / "flexisol-release-audit-2026-07-30.json"
        ),
        help="Destination for the deterministic summary artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = audit_release(args.source_root)
    _write_json_atomic(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
