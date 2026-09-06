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
    / "vqm24-static-cpks-gate-b-20260824"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vqm24_static_cpks_gate_b_preserves_the_preregistered_failure() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    source = result["source"]

    assert result["status"] == "fail"
    assert source["sealer_sha256"] == _sha256(ROOT / source["sealer_path"])
    assert source["cpks_runner_sha256"] == _sha256(
        ROOT / "tools/route2_release/run_vqm24_static_cpks_response.py"
    )
    assert source["preregistration_file_sha256"] == _sha256(
        ROOT / source["preregistration_path"]
    )
    assert [(record["stratum"], record["status"]) for record in result["records"]] == [
        ("bromine", "pass"),
        ("fluorine", "pass"),
        ("phosphorus", "fail"),
        ("sulfur", "pass"),
    ]
    phosphorus = next(
        record for record in result["records"] if record["stratum"] == "phosphorus"
    )
    assert [name for name, passed in phosphorus["gates"].items() if not passed] == [
        "curvature_absolute"
    ]
    assert phosphorus["metrics"][
        "curvature_absolute_error_hartree_per_e2"
    ] == pytest.approx(0.0003609249898719191, abs=0.0)
    assert phosphorus["metrics"][
        "curvature_absolute_budget_hartree_per_e2"
    ] == pytest.approx(0.0003261043780690212, abs=0.0)


def test_vqm24_static_cpks_gate_b_observables_pass_but_do_not_override_failure() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())

    assert result["aggregate"] == {
        "all_records_pass": False,
        "maximum_cpks_residual_relative_frobenius": pytest.approx(
            4.312863068819442e-06, abs=0.0
        ),
        "maximum_dipole_symmetric_relative": pytest.approx(
            5.09966743392108e-05, abs=0.0
        ),
        "maximum_mep_symmetric_relative": pytest.approx(
            6.194549829016144e-05, abs=0.0
        ),
        "median_wall_time_ratio_cpks_over_finite": pytest.approx(
            0.7780798960405264, abs=0.0
        ),
        "record_count": 4,
    }
    assert all(
        record["gates"]["mep_global_relative"]
        and record["gates"]["dipole_global_relative"]
        and record["gates"]["cpks_residual_frobenius"]
        and record["gates"]["reciprocity"]
        and record["gates"]["passivity"]
        for record in result["records"]
    )
    for record in result["records"]:
        directory = EVIDENCE / record["stratum"]
        assert record["source"]["cpks_json_sha256"] == _sha256(
            directory / "cpks.json"
        )
        assert record["source"]["cpks_npz_sha256"] == _sha256(
            directory / "cpks.npz"
        )


def test_vqm24_static_cpks_gate_b_claim_boundary_is_fail_closed() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())

    assert result["claim_boundary"] == {
        "cpks_batch_backend_admitted_for_same_q_to_zero_qm_target": False,
        "experimental_solvation_target_read": False,
        "independent_qm_training_data_generated": True,
        "maple_capability_admitted": False,
        "model_accuracy_measured": False,
        "model_fit_or_training_performed": False,
        "pcm_or_cavity_used": False,
        "vqm24_energy_target_used": False,
    }
