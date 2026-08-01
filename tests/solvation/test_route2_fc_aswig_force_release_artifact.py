from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.function.calculator.extra_correction.implicit.route2_force_admission import (
    FC_ASWIG_JGP94_D2_DIRECT_PCM_FORCE_PES_VALIDATION_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_RELATIVE = (
    "docs/implicit-solvation/benchmarks/"
    "route2-fc-aswig-force-v3-release-evidence-v1.json"
)
ARTIFACT = ROOT / ARTIFACT_RELATIVE


def _assert_execution_source_binding(execution: dict[str, object]) -> None:
    assert_source_files_match_execution_commit(
        ROOT,
        {
            "execution_git_head": execution["execution_git_head"],
            "source_files_sha256": execution["source_files_sha256"],
        },
    )


def test_force_v3_release_evidence_is_source_bound_and_complete():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == "route2-fc-aswig-force-v3-release-evidence-v1"
    assert artifact["status"] == (
        "bounded-public-force-profile-admitted-with-open-domain-gates"
    )
    assert artifact["scientific_identity"][
        "common_stationary_electronic_functional_established"
    ] is False
    assert artifact["summary"]["all_registered_gates_passed"] is True

    executions = artifact["executions"]
    full_suite = executions["full_pes_gate_suite"]
    current = executions["current_public_and_kinematics_replay"]
    _assert_execution_source_binding(full_suite)
    _assert_execution_source_binding(current)

    payloads = full_suite["gate_payloads"]
    assert set(payloads) == {
        "acetone_kinematics",
        "acetone_cartesian_path",
        "acetone_closed_loop",
        "acetone_short_nve",
        "flexible_torsion",
        "flexible_closed_loop",
    }
    for payload in payloads.values():
        assert payload["git_worktree_status"] == []
        gate = next(value for key, value in payload.items() if key.endswith("_gate"))
        assert gate["passed"] is True

    assert current["git_worktree_status"] == []
    assert current["public_force_smoke"]["forces_evaluated"] is True
    assert current["public_force_smoke"]["force_release_admitted"] is True
    assert current["public_force_smoke"]["source_receiver_contract"][
        "public_capability"
    ] == "bounded-experimental-energy-and-conservative-forces"
    assert current["kinematics_payload"]["kinematic_gate"]["passed"] is True


def test_force_v3_release_metrics_preserve_the_registered_numerical_gates():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    summary = artifact["summary"]

    acetone = summary["acetone"]
    assert acetone["maximum_component_fd_error_ev_per_angstrom"] == pytest.approx(
        5.420641025864159e-06
    )
    assert acetone["cartesian_path_maximum_centered_fd_error_ev_per_angstrom"] == (
        pytest.approx(1.3103270510803168e-05)
    )
    assert abs(acetone["closed_loop_work_ev"]) < 1.0e-6
    assert acetone["surface_cardinality"] == 860
    assert acetone["nve_maximum_absolute_energy_drifts_ev"] == pytest.approx(
        [
            8.991255601169612e-07,
            2.244474771236367e-07,
            5.58862315502552e-08,
        ]
    )
    assert acetone["nve_fine_over_coarse_drift_ratios"] == pytest.approx(
        [0.24962862483237583, 0.24899469696186569]
    )

    flexible = summary["2_acetoxyethyl_acetate"]
    assert flexible["fine_over_coarse_error_ratios"] == pytest.approx(
        [0.2502357029260745, 0.25129871756734357]
    )
    assert abs(flexible["closed_loop_work_ev"]) < 1.0e-6
    assert flexible["surface_cardinality"] == 1720

    replay = summary["current_head_replay"]
    assert replay["kinematics_passed"] is True
    assert replay["maximum_component_fd_error_ev_per_angstrom"] == pytest.approx(
        5.420641025864159e-06
    )


def test_force_admission_contract_points_to_the_tracked_evidence():
    contract = FC_ASWIG_JGP94_D2_DIRECT_PCM_FORCE_PES_VALIDATION_CONTRACT

    assert contract.evidence_artifact_path == ARTIFACT_RELATIVE
    assert ARTIFACT.is_file()
