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
    "route2-v0-qeq-acetone-qm-field-v1.json"
)
PREREGISTRATION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-qeq-acetone-qm-field-prereg-v1.json"
)


def test_v0_qeq_acetone_qm_field_result_rejects_the_frozen_curvature():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert artifact["artifact"] == "route2-v0-qeq-acetone-qm-field-v1"
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "pass"
    assert artifact["execution_git_head"] == (
        "8e50e36c42a51b5b9acca0d64ddbcbd0edfd4dd0"
    )
    assert_source_files_match_execution_commit(ROOT, artifact)
    assert artifact["preregistration"] == {
        "path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-qeq-acetone-qm-field-prereg-v1.json"
        ),
        "sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "protocol_id": "route2-v0-qeq-acetone-qm-field-prereg-v1",
    }
    assert artifact["hard_constraints"] == preregistration["hard_constraints"]
    assert artifact["hard_constraints"]["post_training"] is False
    assert artifact["hard_constraints"]["fine_tuning"] is False
    assert artifact["hard_constraints"]["experimental_solvation_fit"] is False
    assert artifact["hard_constraints"]["qeq_response_rescaling"] is False

    qm = artifact["qm_reference"]
    selected = qm["polarizability_by_step"]["3.0e-04"]
    numerical = preregistration["numerical_gates"]
    assert selected["antisymmetry_relative_frobenius"] <= (
        numerical["qm_antisymmetry_relative_frobenius_max"]
    )
    assert selected["energy_dipole_diagonal_relative_max"] <= (
        numerical["energy_dipole_diagonal_relative_max"]
    )
    assert qm["field_step_consistency_relative_frobenius"] <= (
        numerical["field_step_consistency_relative_frobenius_max"]
    )
    assert min(qm["selected_eigenvalues_bohr3"]) >= (
        numerical["qm_minimum_eigenvalue_bohr3_min"]
    )

    qeq = artifact["qeq_control"]
    assert qeq["charge_constraint_residual_e_max"] <= (
        numerical["qeq_charge_constraint_residual_e_max"]
    )
    assert qeq["stationarity_residual_hartree_per_e_max"] <= (
        numerical["qeq_stationarity_residual_hartree_per_e_max"]
    )

    falsification = artifact["scientific_falsification"]
    assert falsification["passes_all_registered_checks"] is False
    assert falsification["verdict"] == "reject-fixed-qeq-monopole-curvature"
    checks = falsification["checks"]
    assert all(check["passes"] is False for check in checks.values())
    np.testing.assert_allclose(
        [
            checks["relative_frobenius_mismatch"]["value"],
            checks["trace_ratio"]["value"],
            checks["principal_value_relative_max"]["value"],
        ],
        [1.4295333806787403, 1.9767767410014712, 2.224402712926121],
        rtol=1.0e-12,
        atol=0.0,
    )

    runtime = artifact["runtime"]["pyscf_helper"]
    frozen_runtime = preregistration["execution_contract"]["runtime"]
    assert runtime["pyscf"] == frozen_runtime["pyscf_version"]
    assert runtime["python_resolved_sha256"] == (
        frozen_runtime["python_resolved_sha256"]
    )
    assert runtime["pyscf_init_sha256"] == (
        frozen_runtime["pyscf_init_sha256"]
    )
