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
    / "auxiliary-density-standard-basis-ladder-20260824"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ladder_result_replays_its_source_and_preregistration_identity() -> None:
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


def test_all_three_standard_bases_fail_only_the_frozen_condition_lane() -> None:
    result = _load("result.json")
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in result["evaluated_candidates"]
    }

    assert result["status"] == "fail"
    assert result["selected_candidate_id"] is None
    assert tuple(result["candidate_order"]) == (
        "etb-beta-2p0",
        "autoaux",
        "etb-beta-1p5",
    )
    assert set(candidates) == set(result["candidate_order"])
    for candidate in candidates.values():
        assert candidate["status"] == "fail"
        assert candidate["gates"] == {
            "all_cases": False,
            "conditioning": False,
            "constraints": True,
            "maximum_bound": True,
            "mean_bound": True,
            "reciprocity": True,
        }


def test_etb2_accuracy_capacity_is_not_misreported_as_basis_admission() -> None:
    result = _load("result.json")
    etb2 = next(
        candidate
        for candidate in result["evaluated_candidates"]
        if candidate["candidate_id"] == "etb-beta-2p0"
    )
    aggregate = etb2["aggregate"]

    assert aggregate["mean_energy_error_upper_bound_kcal_per_mol"] == (
        pytest.approx(0.032561896166927715, abs=0.0)
    )
    assert aggregate["maximum_energy_error_upper_bound_kcal_per_mol"] == (
        pytest.approx(0.05095175718024519, abs=0.0)
    )
    assert aggregate["mean_actual_energy_error_kcal_per_mol"] == pytest.approx(
        0.00611268230357622,
        abs=0.0,
    )
    assert aggregate["maximum_metric_condition_number"] == pytest.approx(
        65856946995.98133,
        abs=0.0,
    )
    assert result["claim_boundary"]["capability_admitted"] is False
