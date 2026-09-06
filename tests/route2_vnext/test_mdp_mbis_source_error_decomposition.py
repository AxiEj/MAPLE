from __future__ import annotations

import numpy as np
import pytest

from tools.route2_release.analyze_mdp_mbis_source_error_decomposition import (
    _architecture_decision,
    _project_qp_to_charge_and_dipole,
)


def test_metric_projection_closes_charge_and_molecular_dipole() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [1.2, -0.4, 0.7]])
    charges = np.asarray([0.15, -0.10])
    dipoles = np.asarray([[0.02, 0.01, -0.03], [-0.04, 0.05, 0.01]])
    target = np.asarray([0.31, -0.27, 0.19])

    projected_q, projected_p, condition = _project_qp_to_charge_and_dipole(
        positions_angstrom=positions,
        charges_e=charges,
        dipoles_eangstrom=dipoles,
        target_charge_e=0.0,
        target_dipole_eangstrom=target,
        charge_sigma_e=0.03,
        dipole_sigma_eangstrom=0.007,
    )

    assert np.isfinite(condition) and condition > 0.0
    assert float(np.sum(projected_q)) == pytest.approx(0.0, abs=2.0e-12)
    reconstructed = np.sum(projected_q[:, None] * positions + projected_p, axis=0)
    np.testing.assert_allclose(reconstructed, target, rtol=0.0, atol=2.0e-12)


def test_architecture_decision_applies_frozen_thresholds() -> None:
    passing = _architecture_decision(
        qp_to_qpqo_rmse={"1.00": 0.020, "1.25": 0.010, "1.50": 0.006},
        qpq_to_qpqo_rmse={"1.00": 0.009, "1.25": 0.004, "1.50": 0.002},
        anchor_relative_rmse={"1.00": 0.03, "1.25": 0.04, "1.50": 0.05},
        anchor_energy_mae_kcal_mol=1.0,
    )
    assert passing["quadrupole_head_allowed"] is True
    assert passing["hard_anchor_allowed"] is True

    failing = _architecture_decision(
        qp_to_qpqo_rmse={"1.00": 0.020, "1.25": 0.010, "1.50": 0.006},
        qpq_to_qpqo_rmse={"1.00": 0.011, "1.25": 0.004, "1.50": 0.002},
        anchor_relative_rmse={"1.00": 0.03, "1.25": 0.051, "1.50": 0.04},
        anchor_energy_mae_kcal_mol=1.001,
    )
    assert failing["quadrupole_head_allowed"] is False
    assert failing["hard_anchor_mep_gate"] is False
    assert failing["hard_anchor_energy_gate"] is False
    assert failing["hard_anchor_allowed"] is False
