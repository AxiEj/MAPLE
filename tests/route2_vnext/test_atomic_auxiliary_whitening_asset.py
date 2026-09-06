from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / (
    "docs/implicit-solvation/benchmarks/route2-etb2-atomic-whitening-v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_whitening_asset_is_source_bound_and_complete() -> None:
    asset = json.loads((ASSET_ROOT / "asset.json").read_text())

    assert asset["status"] == "frozen-before-any-molecular-local-whitening-audit"
    assert tuple(asset["elements"]) == (
        "H",
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    )
    for relative, expected in asset["source_files_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    assert len(asset["records"]) == len(asset["elements"])
    assert asset["claim_boundary"]["capability_admitted"] is False


def test_atomic_whitening_files_replay_their_declared_hashes_and_shapes() -> None:
    asset = json.loads((ASSET_ROOT / "asset.json").read_text())

    for record in asset["records"]:
        basis = ASSET_ROOT / record["basis_file"]
        transform_path = ASSET_ROOT / record["transform_file"]
        transform = np.load(transform_path, allow_pickle=False)
        dimension = record["auxiliary_dimension"]

        assert _sha256(basis) == record["basis_file_sha256"]
        assert _sha256(transform_path) == record["transform_file_sha256"]
        assert transform.shape == (dimension, dimension)
        assert np.all(np.isfinite(transform))
        assert record["full_identity_max_absolute_error"] < 1.0e-8
        assert record["forbidden_angular_coupling_max_absolute_value"] < 1.0e-10
