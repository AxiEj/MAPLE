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
    / "auxiliary-density-pcmsolver-energy-gate-20260824"
)


def _load(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE / name).read_text())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pcm_energy_gate_is_bound_to_its_runner_and_preregistration() -> None:
    preregistration = _load("preregistration.json")
    result = _load("result.json")
    runner = ROOT / (
        "tools/route2_release/run_auxiliary_density_pcmsolver_energy_gate.py"
    )

    assert preregistration["source_files_sha256"][
        "tools/route2_release/run_auxiliary_density_pcmsolver_energy_gate.py"
    ] == _sha256(runner)
    assert result["preregistration_file_sha256"] == _sha256(
        EVIDENCE / "preregistration.json"
    )
    assert result["preregistration_sha256"] == preregistration[
        "preregistration_sha256"
    ]
    assert result["record_count"] == 12


def test_pcm_energy_gate_fails_only_its_locked_mean_bound() -> None:
    result = _load("result.json")
    aggregate = result["aggregate"]

    assert result["status"] == "fail"
    assert result["gates"] == {
        "all_cases": True,
        "constraints": True,
        "maximum_bound": True,
        "mean_bound": False,
        "reciprocity": True,
    }
    assert aggregate["mean_energy_error_upper_bound_kcal_per_mol"] == (
        pytest.approx(0.3300610463956254, abs=0.0)
    )
    assert aggregate["maximum_energy_error_upper_bound_kcal_per_mol"] == (
        pytest.approx(0.48021401418474585, abs=0.0)
    )
    assert aggregate["mean_actual_energy_error_kcal_per_mol"] == pytest.approx(
        0.1276112700278051,
        abs=0.0,
    )
    assert aggregate["maximum_constraint_absolute_error"] < 1.0e-8
    assert aggregate["maximum_reciprocity_relative_defect"] < 2.0e-10


def test_pcm_energy_gate_keeps_methane_and_does_not_read_targets() -> None:
    preregistration = _load("preregistration.json")
    result = _load("result.json")
    methane = next(
        record
        for record in result["records"]
        if record["compound_id"] == "mobley_9055303"
    )

    assert preregistration["case_policy"] == {
        "all_twelve_retained": True,
        "methane_or_symmetry_control_excluded": False,
        "v1_result_reinterpreted": False,
    }
    assert methane["energy_error_upper_bound_kcal_per_mol"] == pytest.approx(
        0.09949594718113523,
        abs=0.0,
    )
    assert result["claim_boundary"] == {
        "allowed_decision": (
            "reject or retain make_auxbasis as a representation oracle"
        ),
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
    }
