from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from ase.units import Hartree


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    REPO_ROOT
    / "docs/implicit-solvation/benchmarks/route2-mace-p1-water-operational-audit-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_p1_water_artifact_is_bound_to_its_exact_implementation_sources():
    artifact = _artifact()

    assert artifact["artifact"] == "route2-mace-p1-water-operational-audit-v1"
    assert artifact["schema_version"] == 1
    assert artifact["execution"]["source_binding"] == (
        "content-hash-bound-working-tree"
    )
    for relative, expected in artifact[
        "repository_source_files_sha256"
    ].items():
        assert _sha256(REPO_ROOT / relative) == expected
    assert artifact["p0_certificate"]["field_convention"] == (
        "potential-gradient"
    )
    assert set(artifact["p0_certificate"]["blocked_routes"]) == {
        "operational-force",
        "variational",
    }


def test_p1_protocol_is_neutral_water_fixed_cavity_electrostatics_only():
    protocol = _artifact()["protocol"]
    continuum = protocol["continuum"]

    assert protocol["molecule"] == "water"
    assert protocol["total_charge_e"] == 0.0
    assert protocol["atomic_numbers"] == [8, 1, 1]
    assert protocol["electrostatic_energy_ledger"] == (
        "pcm-half-coupling-only-v1"
    )
    assert protocol["standard_state_energy_hartree"] == 0.0
    assert protocol["scientific_route"] == "response-only"
    assert protocol["energy_source_conjugacy_direction"] == (
        "scale-converged-local-reaction-field-v1"
    )
    assert protocol["force_requested"] is False
    assert continuum["identity"] == (
        "route2-pminus1-water-ddpcm-fixed-cavity-v1"
    )
    assert continuum["continuum_equation"] == "ddPCM"
    assert continuum["dielectric"] == 78.39
    assert continuum["radii_angstrom_by_atomic_number"] == {
        "1": 1.2,
        "8": 1.52,
    }
    assert continuum["electrostatics_only"] is True
    assert continuum["include_cds"] is False
    assert len(protocol["cavity_sha256"]) == 64


def test_p1_outer_root_and_scalar_ledger_close_but_inner_residual_does_not():
    artifact = _artifact()
    outer = artifact["outer_scf"]
    audit = artifact["operational_audit"]
    final = outer["history"][-1]

    assert outer["convergence"]["reason"] == "nominal-density-and-energy-v1"
    assert outer["convergence"]["nominal_residual_gate_passed"] is True
    assert len(outer["history"]) == 7
    assert final["accepted"] is True
    assert final["next_density_update"] == "converged"
    assert final["monopole_residual_e"] <= 2.0e-12
    assert final["monopole_residual_rms_e"] <= 2.0e-12
    assert final["dipole_residual_e_angstrom"] <= 2.0e-12
    assert final["dipole_residual_rms_e_angstrom"] <= 2.0e-12
    assert abs(final["raw_response_charge_delta_e"]) <= 2.0e-12
    assert final["projected_total_charge_residual_e"] <= 2.0e-12
    assert final["molecular_dipole_residual_l2_e_angstrom"] <= 2.0e-12
    assert final["energy_residual_ev"] <= 2.0e-12

    assert audit["outer_scf"]["passed"] is True
    assert audit["source_field_pairing_ev"] * 0.5 == pytest.approx(
        audit["pcm_half_coupling_ev"]
    )
    assert audit["pcm_half_coupling_ev"] == pytest.approx(
        audit["pcm_provider_energy_ev"],
        abs=1.0e-10,
    )
    assert audit["pairing_identity_error_ev"] <= 1.0e-10
    assert audit["cds_energy_ev"] == 0.0
    assert audit["standard_state_energy_ev"] == 0.0
    assert audit["final_scalar_ev"] == pytest.approx(
        audit["energy_ledger_hartree"]["delta_g_solv"] * Hartree
    )
    assert audit["continuum_solve"]["status"] == "tolerance-only"
    assert audit["continuum_solve"]["requested_tolerance"] == 1.0e-12
    assert audit["continuum_solve"]["actual_residual"] is None
    assert audit["continuum_solve"]["residual_gate_passed"] is False
    assert audit["p1_complete"] is False
    assert audit["blockers"] == [
        "inner-continuum-algebraic-residual-not-certified"
    ]


def test_p1_conjugacy_failure_is_step_stable_and_blocks_no_claim_silently():
    artifact = _artifact()
    probes = artifact["energy_source_conjugacy_step_stability"]
    decision = artifact["decision"]

    assert [probe["finite_difference_step"] for probe in probes] == [
        2.0e-3,
        1.0e-3,
        5.0e-4,
    ]
    assert all(probe["status"] == "failed" for probe in probes)
    derivatives = np.asarray(
        [probe["finite_difference_derivative_ev"] for probe in probes]
    )
    assert float(np.ptp(derivatives)) <= 1.0e-6
    assert all(probe["relative_error"] > 1.0 for probe in probes)
    assert decision == {
        "blockers": ["inner-continuum-algebraic-residual-not-certified"],
        "energy_interpretation": "response-conditioned-operational-prediction",
        "inner_continuum_residual_status": "tolerance-only",
        "operational_force_status": "blocked-by-p0-and-profile",
        "p1_complete": False,
        "p2_p3_status": "blocked-by-p1",
        "variational_status": "blocked-by-p0-and-missing-functional",
    }
