from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-qeq-monopole-acetone-v1.json"
)
PREREGISTRATION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-qeq-monopole-acetone-prereg-v1.json"
)


def test_v0_qeq_monopole_acetone_canary_is_source_bound_and_claim_bounded():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert artifact["artifact"] == "route2-v0-qeq-monopole-acetone-v1"
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "pass"
    assert artifact["execution_git_head"] == (
        "701c5e57762bb5c35fa8d6cab1e0ba6d69036d1d"
    )
    assert_source_files_match_execution_commit(ROOT, artifact)

    preregistration_sha256 = hashlib.sha256(
        PREREGISTRATION.read_bytes()
    ).hexdigest()
    assert artifact["preregistration"] == {
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-qeq-monopole-acetone-prereg-v1.json"
        ),
        "sha256": preregistration_sha256,
        "protocol_id": "route2-v0-qeq-monopole-acetone-prereg-v1",
    }
    assert artifact["hard_constraints"] == preregistration["hard_constraints"]
    assert artifact["hard_constraints"]["post_training"] is False
    assert artifact["hard_constraints"]["fine_tuning"] is False
    assert artifact["hard_constraints"]["experimental_solvation_fit"] is False
    assert artifact["hard_constraints"]["map_or_uq_calibration"] is False
    assert "experimental solvation accuracy" in artifact["claim_boundary"]
    assert "total solvation free energy" in artifact["claim_boundary"]
    assert "force, PES" in artifact["claim_boundary"]

    system = artifact["system"]
    assert system["compound_id"] == "mobley_3867265"
    assert system["name"] == "acetone"
    assert system["atom_symbols"] == ["C", "C", "O", "C", "H", "H", "H", "H", "H", "H"]
    assert system["frozen_density_shape"] == [10, 1, 4]
    assert system["cavity_size"] == 516

    frozen = artifact["frozen_inputs"]
    assert frozen["archived_exact_gto_result"]["checkpoint"]["sha256"] == (
        "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
    )
    assert frozen["archived_exact_gto_state"]["sha256"] == (
        "68b7f77c66d51f7c48cde12be9117622848493ef9606d645b8bffc01ae106c30"
    )
    assert frozen["parsed_pcmsolver_input"]["sha256"] == (
        "4f4c56677bb99967855a4821937e74be94dafd9cc2ab59e1de70fb0d9eb8c069"
    )

    construction = artifact["construction"]
    assert construction["name"] == (
        "route2-v0-rappe-goddard-hardness-same-basis-monopole-v1"
    )
    assert construction["parameter_table_sha256"] == (
        "5d2b405b78dd59b95da89b69fb10b409bcb3f0f85921fc8cc4a67dd2aba8a618"
    )
    assert "not full QEq" in construction["electronic_response"]
    assert "exact KKT constraint" in construction["response_subspace"]
    continuum = construction["continuum"]
    assert continuum["source_receiver_basis_identical"] is True
    assert continuum["surface_response"] == "energy-conjugate"
    assert continuum["sigmas_angstrom"] == [1.5]
    assert continuum["production_profile"] is False

    gates = artifact["structural_gates"]
    lower_bound_gate_names = {
        "electronic_minimum_curvature_hartree",
        "joint_minimum_curvature_hartree",
    }
    expected_gate_names = {
        name.removesuffix("_min").removesuffix("_max")
        for name in preregistration["structural_gates"]
    }
    assert set(gates) == expected_gate_names
    for name, record in gates.items():
        value = record["value"]
        threshold = record["threshold"]
        assert np.isfinite(value)
        assert np.isfinite(threshold)
        if name in lower_bound_gate_names:
            assert threshold == preregistration["structural_gates"][f"{name}_min"]
            assert value > threshold
        else:
            assert threshold == preregistration["structural_gates"][f"{name}_max"]
            assert value <= threshold

    np.testing.assert_allclose(
        [
            gates["electronic_minimum_curvature_hartree"]["value"],
            gates["joint_minimum_curvature_hartree"]["value"],
            gates["operator_maximum_eigenvalue_hartree"]["value"],
            gates["response_maximum_eigenvalue_hartree"]["value"],
        ],
        [
            0.17823541796958414,
            0.17500216390832452,
            -8.783569157023157e-09,
            1.9748870847961372e-15,
        ],
        rtol=1.0e-12,
        atol=1.0e-18,
    )

    ledger = artifact["energy_ledger_hartree"]
    np.testing.assert_allclose(
        ledger["electronic_induction"] + ledger["continuum_polarization"],
        ledger["solute_continuum"],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        ledger["stationary_total"],
        ledger["solute_continuum"],
        rtol=0.0,
        atol=1.0e-15,
    )
