from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "vqm24-nonuniform-qm-gate0-20260824"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vqm24_qm_gate0_is_preregistered_and_file_bound() -> None:
    preregistration = json.loads((EVIDENCE / "preregistration.json").read_text())
    result = json.loads((EVIDENCE / "result.json").read_text())
    sealer = ROOT / result["source"]["sealer_path"]

    assert preregistration["status"] == "locked-before-first-vqm24-qm-gate0-execution"
    assert result["status"] == "pass-numerical-gate0"
    assert result["source"]["sealer_sha256"] == _sha256(sealer)
    assert result["source"]["preregistration_file_sha256"] == _sha256(
        EVIDENCE / "preregistration.json"
    )
    assert result["source"]["preregistration_sha256"] == preregistration[
        "preregistration_sha256"
    ]
    local_names = {
        "gas_json": "gas.json",
        "gas_checkpoint": "gas.chk",
        "response_json": "qm-response.json",
        "response_npz": "qm-response.npz",
    }
    for key, name in local_names.items():
        assert result["files"][key]["sha256"] == _sha256(EVIDENCE / name)


def test_vqm24_qm_gate0_response_is_converged_and_step_consistent() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    response = result["response"]

    assert result["gates"] == {
        "all_energy_curvatures_negative": True,
        "all_scf_converged": True,
        "dipole_step_consistency": True,
        "electron_count": True,
        "mep_step_consistency": True,
    }
    assert result["gas"]["density_binding_residual_inf"] < 1.0e-12
    assert response["mep_step_relative_difference"] == pytest.approx(
        0.0005971613974200663,
        abs=0.0,
    )
    assert response["dipole_step_relative_difference"] == pytest.approx(
        0.0005570226068365585,
        abs=0.0,
    )
    assert response["maximum_electron_count_error_e"] < 1.0e-12
    assert response["minimum_scf_cycles"] == 6
    assert response["maximum_scf_cycles"] == 8
    assert all(
        item["central_curvature_hartree_per_e2"] < 0.0
        for mode in response["energy_diagnostics"]
        for item in mode["by_step"]
    )

    with np.load(EVIDENCE / "qm-response.npz", allow_pickle=False) as state:
        assert state["induced_surface_mep_hartree_per_e_per_source_e"].shape == (
            2,
            4,
            55,
        )
        assert state["induced_dipole_e_bohr_per_source_e"].shape == (2, 4, 3)


def test_vqm24_qm_gate0_contains_no_solvation_or_pcm_training_target() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())

    assert result["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "independent_qm_training_datum_generated": True,
        "pcm_or_cavity_used": False,
    }
