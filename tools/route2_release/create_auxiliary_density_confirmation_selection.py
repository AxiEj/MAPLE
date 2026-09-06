#!/usr/bin/env python3
"""Freeze a target-free confirmation identity for the auxiliary-basis ladder."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = (
    "tools/route2_release/create_auxiliary_density_confirmation_selection.py"
)
ARTIFACT = "route2-auxiliary-density-confirmation-selection-v1"
EXPECTED_DATABASE_SHA256 = (
    "2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260"
)
DEVELOPMENT_PANEL = SOURCE_ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-freesolv12-zero-field-mace-static-surface-mep-prereg-v1.json"
)
SELECTION = (
    ("mobley_1036761", "saturated-carbocycle-primary-amine"),
    ("mobley_1659169", "saturated-heterocycle-amine-ether"),
    ("mobley_1723043", "perfluorinated-small-ring"),
    ("mobley_1107178", "iodinated-small-molecule"),
    ("mobley_1235151", "aliphatic-nitro"),
    ("mobley_1323538", "phosphate-ester"),
    ("mobley_1708457", "sulfone"),
    ("mobley_1952272", "small-nitro-symmetry-control"),
    ("mobley_1967551", "aldehyde"),
    ("mobley_1244778", "medium-carbocycle-alcohol"),
    ("mobley_1261349", "branched-hydrocarbon-control"),
    ("mobley_1929982", "sulfur-hydride-low-signal-control"),
)


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
        ).encode()
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def _metadata_rows(path: Path) -> dict[str, tuple[str, str]]:
    """Read only ID, SMILES, and name fields from the semicolon table."""

    rows: dict[str, tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 3:
            raise ValueError("FreeSolv metadata row is malformed.")
        rows[fields[0]] = (fields[1], fields[2])
    return rows


def create(args: argparse.Namespace) -> dict[str, Any]:
    from maple.function.read.filereader.mol2_reader import MOL2Reader
    from maple.solvation.coupling.state_equation import geometry_sha256

    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    database = (dataset_root / "database.txt").resolve(strict=True)
    mol2_root = (dataset_root / "mol2files_gaff").resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if _sha256(database) != EXPECTED_DATABASE_SHA256:
        raise ValueError("FreeSolv metadata file identity drifted.")
    development = _load_object(DEVELOPMENT_PANEL.resolve(strict=True))
    development_ids = {
        str(record["compound_id"]) for record in development["locked_records"]
    }
    selected_ids = {compound_id for compound_id, _role in SELECTION}
    if development_ids & selected_ids:
        raise ValueError("confirmation selection overlaps the development panel.")
    rows = _metadata_rows(database)
    records = []
    for ordinal, (compound_id, coverage_role) in enumerate(SELECTION, start=1):
        smiles, name = rows[compound_id]
        mol2 = (mol2_root / f"{compound_id}.mol2").resolve(strict=True)
        atoms = MOL2Reader(str(mol2), charge=0, mult=1)
        if abs(float(np.sum(atoms.get_initial_charges()))) > 1.0e-3:
            raise ValueError(f"{compound_id} input charges do not screen neutral.")
        records.append(
            {
                "ordinal": ordinal,
                "compound_id": compound_id,
                "name": name,
                "smiles": smiles,
                "coverage_role": coverage_role,
                "atom_count": len(atoms),
                "elements": sorted(set(atoms.get_chemical_symbols())),
                "geometry_sha256": geometry_sha256(atoms),
                "mol2_path": str(mol2),
                "mol2_sha256": _sha256(mol2),
            }
        )
    if len({record["smiles"] for record in records}) != len(records):
        raise ValueError("confirmation selection contains duplicate molecular graphs.")
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-any-standard-basis-ladder-result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "development_panel_path": str(DEVELOPMENT_PANEL),
            "development_panel_sha256": _sha256(DEVELOPMENT_PANEL),
            "database_path": str(database),
            "database_sha256": _sha256(database),
        },
        "selection_policy": {
            "record_count": len(records),
            "development_compound_overlap": False,
            "exact_smiles_duplicates": False,
            "coverage": (
                "new rings, nitro, aldehyde, phosphate, sulfone, F/P/S/I, "
                "branched and low-signal controls"
            ),
            "candidate_basis_or_mep_result_used": False,
            "experimental_solvation_field_indexed_or_emitted": False,
        },
        "records": records,
        "one_shot_contract": {
            "apply_only_to_first_development_passer": True,
            "identical_numerical_and_reaction_metric_gates": True,
            "confirmation_failure_closes_entire_frozen_ladder": True,
            "later_candidates_forbidden_after_confirmation_failure": True,
        },
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "model_output_read": False,
            "fit_or_training_performed": False,
            "capability_admitted": False,
        },
    }
    payload["selection_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    output.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
