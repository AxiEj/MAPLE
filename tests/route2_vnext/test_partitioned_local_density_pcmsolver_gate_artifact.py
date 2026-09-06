from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "partitioned-local-density-pcmsolver-gate-20260824"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text())
    assert isinstance(value, dict)
    return value


def test_partitioned_gate_replays_bound_sources_and_preregistration() -> None:
    preregistration = _load("preregistration.json")
    result = _load("result.json")

    for relative, expected in preregistration["source_files_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    assert result["preregistration_file_sha256"] == _sha256(
        EVIDENCE / "preregistration.json"
    )
    assert result["preregistration_sha256"] == preregistration[
        "preregistration_sha256"
    ]


def test_partitioned_labels_are_stable_but_fail_physical_and_grid_gates() -> None:
    result = _load("result.json")
    aggregate = result["aggregate"]

    assert result["status"] == "fail"
    assert result["gates"] == {
        "all_cases": False,
        "constraints": True,
        "grid_electron_count": False,
        "grid_first_moment": False,
        "local_condition": True,
        "maximum_bound": False,
        "mean_bound": False,
        "reciprocity": True,
    }
    assert aggregate["mean_energy_error_upper_bound_kcal_per_mol"] == (
        pytest.approx(0.7072101734976285, abs=0.0)
    )
    assert aggregate["maximum_energy_error_upper_bound_kcal_per_mol"] == (
        pytest.approx(1.0107770256291955, abs=0.0)
    )
    assert aggregate["maximum_local_metric_condition_number"] < 1.0e5
    assert aggregate["maximum_constraint_absolute_error"] < 1.0e-10
    assert result["claim_boundary"]["capability_admitted"] is False
