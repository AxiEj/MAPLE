from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIRECTORY = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "vqm24-nonuniform-qm-gate-a-20260824"
)
STRATA = ("bromine", "chlorine", "fluorine", "phosphorus", "sulfur")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_directory(stratum: str) -> Path:
    return (
        ROOT
        / "docs"
        / "route2"
        / "evidence"
        / f"vqm24-nonuniform-qm-gate-a-{stratum}-20260824"
    )


def test_vqm24_qm_gate_a_is_complete_and_content_bound() -> None:
    summary = json.loads((SUMMARY_DIRECTORY / "summary.json").read_text())
    aggregator = ROOT / summary["source"]["aggregator_path"]

    assert summary["status"] == "pass-numerical-gate-a"
    assert summary["protocol"]["strata"] == list(STRATA)
    assert summary["source"]["aggregator_sha256"] == _sha256(aggregator)
    assert len(summary["records"]) == 5
    assert [record["stratum"] for record in summary["records"]] == list(STRATA)

    for record in summary["records"]:
        result_path = ROOT / record["result_path"]
        result = json.loads(result_path.read_text())
        sealer = ROOT / result["source"]["sealer_path"]
        assert record["result_file_sha256"] == _sha256(result_path)
        assert record["result_sha256"] == result["result_sha256"]
        assert result["source"]["sealer_sha256"] == _sha256(sealer)
        assert result["status"] == "pass-numerical-gate-a-record"
        for file_record in result["files"].values():
            assert _sha256(ROOT / file_record["path"]) == file_record["sha256"]


def test_vqm24_qm_gate_a_uses_one_level3_protocol_and_passes_response_gates() -> None:
    summary = json.loads((SUMMARY_DIRECTORY / "summary.json").read_text())

    assert summary["aggregate"] == {
        "all_records_pass": True,
        "maximum_dipole_step_relative_difference": pytest.approx(
            0.00027870660461010155, abs=0.0
        ),
        "maximum_electron_count_error_e": pytest.approx(
            1.4210854715202004e-13, abs=0.0
        ),
        "maximum_mep_step_relative_difference": pytest.approx(
            0.0005723545899987653, abs=0.0
        ),
    }
    for stratum in STRATA:
        directory = _record_directory(stratum)
        result = json.loads((directory / "result.json").read_text())
        gas = json.loads((directory / "gas.json").read_text())
        response = json.loads((directory / "qm-response.json").read_text())

        assert gas["numerics"]["nonlocal_grid_profile"] == "level"
        assert gas["numerics"]["nonlocal_grid_level"] == 3
        assert response["method"]["nonlocal_grid_profile"] == "PySCF-level-3"
        assert all(result["gates"].values())
        assert result["gas"]["density_binding_residual_inf"] < 1.0e-12
        assert result["response"]["mep_step_relative_difference"] < 0.02
        assert result["response"]["dipole_step_relative_difference"] < 0.02
        assert all(
            item["central_curvature_hartree_per_e2"] < 0.0
            for mode in result["response"]["energy_diagnostics"]
            for item in mode["by_step"]
        )
        with np.load(directory / "qm-response.npz", allow_pickle=False) as state:
            assert state[
                "induced_surface_mep_hartree_per_e_per_source_e"
            ].shape[:2] == (2, 4)
            assert state["induced_dipole_e_bohr_per_source_e"].shape == (2, 4, 3)


def test_vqm24_qm_gate_a_does_not_read_model_or_solvation_targets() -> None:
    summary = json.loads((SUMMARY_DIRECTORY / "summary.json").read_text())

    assert summary["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "independent_qm_training_data_generated": True,
        "model_accuracy_measured": False,
        "pcm_or_cavity_used": False,
        "rare_element_numerical_protocol_validated": True,
        "vqm24_energy_target_used": False,
    }
    for stratum in STRATA:
        result = json.loads((_record_directory(stratum) / "result.json").read_text())
        assert result["claim_boundary"] == {
            "capability_admitted": False,
            "experimental_solvation_target_read": False,
            "fit_or_training_performed": False,
            "independent_qm_training_datum_generated": True,
            "pcm_or_cavity_used": False,
            "vqm24_energy_target_used": False,
        }
