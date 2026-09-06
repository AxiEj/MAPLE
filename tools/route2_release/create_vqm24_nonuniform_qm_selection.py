#!/usr/bin/env python3
"""Freeze a target-free VQM24 geometry split for external-field QM data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/create_vqm24_nonuniform_qm_selection.py"
ARTIFACT = "route2-vqm24-nonuniform-qm-geometry-selection-v1"
DATASET_ID = "colabfit/Vector-QM24_DFT_all"
DATASET_REVISION = "baf79fff54725bb9a349f6234f2e23f0de22a147"
SELECTION_SEED = "maple-route2-observable-scalar-vqm24-v1"
ALLOWED_ELEMENTS = frozenset(("H", "C", "N", "O", "F", "P", "S", "Cl", "Br"))
STRATUM_QUOTAS = {
    "base-hcno": 20,
    "fluorine": 8,
    "phosphorus": 8,
    "sulfur": 8,
    "chlorine": 8,
    "bromine": 8,
}
SPLIT_COUNTS = {
    "base-hcno": {"train": 12, "validation": 4, "blind": 4},
    "fluorine": {"train": 4, "validation": 2, "blind": 2},
    "phosphorus": {"train": 4, "validation": 2, "blind": 2},
    "sulfur": {"train": 4, "validation": 2, "blind": 2},
    "chlorine": {"train": 4, "validation": 2, "blind": 2},
    "bromine": {"train": 4, "validation": 2, "blind": 2},
}


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


def _array_sha256(values: object, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def _score(*parts: str) -> str:
    return hashlib.sha256("\0".join((SELECTION_SEED, *parts)).encode()).hexdigest()


def _stratum(elements: frozenset[str]) -> str | None:
    if "Br" in elements:
        return "bromine"
    if "Cl" in elements:
        return "chlorine"
    if "P" in elements:
        return "phosphorus"
    if "S" in elements:
        return "sulfur"
    if "F" in elements:
        return "fluorine"
    if elements <= frozenset(("H", "C", "N", "O")):
        return "base-hcno"
    return None


def _record(index: int, row: dict[str, Any], stratum: str) -> dict[str, Any]:
    positions = np.asarray(row["positions"], dtype=np.float64)
    numbers = np.asarray(row["atomic_numbers"], dtype=np.int64)
    if positions.shape != (len(numbers), 3) or not np.all(np.isfinite(positions)):
        raise ValueError("VQM24 geometry is invalid.")
    formula = str(row["chemical_formula_hill"])
    configuration_hash = str(row["configuration_hash"])
    structure_hash = str(row["structure_hash"])
    names = tuple(str(value) for value in row.get("names") or ())
    result: dict[str, Any] = {
        "dataset_index": index,
        "configuration_hash": configuration_hash,
        "structure_hash": structure_hash,
        "chemical_formula_hill": formula,
        "molecule_group_sha256": hashlib.sha256(formula.encode()).hexdigest(),
        "stratum": stratum,
        "names": list(names),
        "atomic_numbers": numbers.tolist(),
        "positions_angstrom": positions.tolist(),
        "atomic_numbers_sha256": _array_sha256(numbers, dtype="<i8"),
        "positions_angstrom_sha256": _array_sha256(positions, dtype="<f8"),
        "selection_score_sha256": _score(stratum, formula, configuration_hash),
    }
    result["record_sha256"] = _canonical_sha256(result)
    return result


def create(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION)
    if info.sha != DATASET_REVISION:
        raise RuntimeError("VQM24 Hugging Face revision drifted.")
    stream = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
    )
    best_by_formula: dict[str, dict[str, Any]] = {}
    scanned = 0
    eligible = 0
    for index, raw in enumerate(stream):
        scanned += 1
        if int(raw["multiplicity"]) != 1 or int(raw["nperiodic_dimensions"]) != 0:
            continue
        elements = frozenset(str(value) for value in raw["elements"])
        if not elements or not elements <= ALLOWED_ELEMENTS:
            continue
        if int(raw["nsites"]) > 24:
            continue
        stratum = _stratum(elements)
        if stratum is None:
            continue
        eligible += 1
        candidate = _record(index, raw, stratum)
        formula = str(candidate["chemical_formula_hill"])
        previous = best_by_formula.get(formula)
        if previous is None or candidate["selection_score_sha256"] < previous[
            "selection_score_sha256"
        ]:
            best_by_formula[formula] = candidate

    selected = []
    for stratum, quota in STRATUM_QUOTAS.items():
        candidates = sorted(
            (
                record
                for record in best_by_formula.values()
                if record["stratum"] == stratum
            ),
            key=lambda record: record["selection_score_sha256"],
        )
        if len(candidates) < quota:
            raise RuntimeError(f"VQM24 stratum {stratum!r} is undersubscribed.")
        chosen = candidates[:quota]
        counts = SPLIT_COUNTS[stratum]
        cursor = 0
        for split in ("train", "validation", "blind"):
            stop = cursor + counts[split]
            for record in chosen[cursor:stop]:
                selected.append(record | {"split": split})
            cursor = stop
        if cursor != quota:
            raise RuntimeError("VQM24 split counts do not close their quota.")
    selected.sort(
        key=lambda record: (
            ("train", "validation", "blind").index(record["split"]),
            record["stratum"],
            record["selection_score_sha256"],
        )
    )
    split_counts = {
        split: sum(record["split"] == split for record in selected)
        for split in ("train", "validation", "blind")
    }
    groups_by_split = {
        split: {
            record["molecule_group_sha256"]
            for record in selected
            if record["split"] == split
        }
        for split in split_counts
    }
    if any(
        groups_by_split[left] & groups_by_split[right]
        for left, right in (
            ("train", "validation"),
            ("train", "blind"),
            ("validation", "blind"),
        )
    ):
        raise RuntimeError("VQM24 molecular formula groups leaked across splits.")
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-any-maple-external-field-qm-calculation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "creator_path": SELF_REPO_PATH,
            "creator_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_doi": "10.60732/b49d5db4",
            "dataset_license": "CC-BY-4.0",
            "publication_doi": "10.1038/s41597-025-05428-4",
        },
        "selection_policy": {
            "seed": SELECTION_SEED,
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "neutral_closed_shell_source_dataset": True,
            "maximum_atom_count": 24,
            "one_configuration_per_formula": True,
            "stratum_priority": [
                "bromine",
                "chlorine",
                "phosphorus",
                "sulfur",
                "fluorine",
                "base-hcno",
            ],
            "stratum_quotas": STRATUM_QUOTAS,
            "split_counts_by_stratum": SPLIT_COUNTS,
            "energy_or_model_result_used_for_selection": False,
            "solvation_dataset_or_target_used": False,
        },
        "scan": {
            "row_count": scanned,
            "eligible_row_count": eligible,
            "eligible_unique_formula_count": len(best_by_formula),
        },
        "split_counts": split_counts,
        "records": selected,
        "claim_boundary": {
            "vqm24_energy_or_atomization_target_used_or_emitted": False,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    value = create(args)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
