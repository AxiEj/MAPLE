from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / (
    "docs/route2/preregistrations/"
    "vqm24-nonuniform-qm-geometry-selection-v1.json"
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


def _array_sha256(values: object, *, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def test_vqm24_selection_is_source_bound_and_frozen_before_qm() -> None:
    value = json.loads(SELECTION.read_text())
    creator = ROOT / value["source"]["creator_path"]
    unsigned = dict(value)
    observed = unsigned.pop("selection_sha256")

    assert value["artifact"] == "route2-vqm24-nonuniform-qm-geometry-selection-v1"
    assert value["status"] == "locked-before-any-maple-external-field-qm-calculation"
    assert value["source"]["creator_sha256"] == _sha256(creator)
    assert value["source"]["dataset_revision"] == (
        "baf79fff54725bb9a349f6234f2e23f0de22a147"
    )
    assert observed == _canonical_sha256(unsigned)
    assert value["claim_boundary"]["experimental_solvation_target_read"] is False
    assert value["claim_boundary"][
        "vqm24_energy_or_atomization_target_used_or_emitted"
    ] is False


def test_vqm24_splits_are_formula_disjoint_stratified_and_hash_bound() -> None:
    value = json.loads(SELECTION.read_text())
    records = value["records"]
    allowed = set(value["selection_policy"]["allowed_elements"])
    groups = {split: set() for split in ("train", "validation", "blind")}

    assert len(records) == 60
    assert value["split_counts"] == {"train": 32, "validation": 14, "blind": 14}
    for record in records:
        numbers = np.asarray(record["atomic_numbers"], dtype=np.int64)
        positions = np.asarray(record["positions_angstrom"], dtype=np.float64)
        unsigned = dict(record)
        observed = unsigned.pop("record_sha256")
        unsigned.pop("split")

        assert positions.shape == (len(numbers), 3)
        assert len(numbers) <= 24
        assert isinstance(record["chemical_formula_hill"], str)
        assert record["chemical_formula_hill"]
        assert record["stratum"] in value["selection_policy"]["stratum_quotas"]
        assert record["split"] in groups
        assert observed == _canonical_sha256(unsigned)
        assert record["atomic_numbers_sha256"] == _array_sha256(
            numbers,
            dtype="<i8",
        )
        assert record["positions_angstrom_sha256"] == _array_sha256(
            positions,
            dtype="<f8",
        )
        symbols = {
            1: "H",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            15: "P",
            16: "S",
            17: "Cl",
            35: "Br",
        }
        elements = {symbols[int(number)] for number in numbers}
        assert elements <= allowed
        groups[record["split"]].add(record["molecule_group_sha256"])
        forbidden = {
            "energy",
            "atomization_energy",
            "experimental_solvation_energy",
        }
        assert forbidden.isdisjoint(record)
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["blind"])
    assert not (groups["validation"] & groups["blind"])
