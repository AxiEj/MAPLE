from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    SurfaceChargeState,
)
from maple.function.calculator.extra_correction.implicit.source import (
    FrozenDensityMEPSource,
    solve_frozen_density_mep_continuum,
)


class _ReciprocalSurfaceResponse:
    def __init__(self, surface_points_bohr: np.ndarray) -> None:
        self.surface_points_bohr = np.asarray(surface_points_bohr, dtype=float)
        self._response = np.asarray(
            [
                [-0.31, 0.04, -0.02],
                [0.04, -0.27, 0.03],
                [-0.02, 0.03, -0.22],
            ]
        )

    def solve(self, surface_potential_hartree_per_e: np.ndarray):
        potential = np.asarray(surface_potential_hartree_per_e, dtype=float)
        charge = self._response @ potential
        return SurfaceChargeState(
            surface_potential_hartree_per_e=potential,
            direct_surface_charge_e=charge,
            adjoint_surface_charge_e=charge,
            energy_conjugate_surface_charge_e=charge,
            polarization_energy_hartree=0.5 * float(np.dot(potential, charge)),
        )


def test_frozen_density_mep_source_is_surface_bound_and_immutable():
    points = np.asarray(
        [
            [-3.0, 0.0, 0.2],
            [2.7, 0.4, -0.1],
            [0.1, -2.8, 0.7],
        ]
    )
    potential = np.asarray([0.4, -0.2, 0.1])

    source = FrozenDensityMEPSource(
        surface_points_bohr=points,
        surface_potential_hartree_per_e=potential,
        declared_total_charge_e=0.0,
        observed_total_charge_e=2.0e-9,
        source_model="salted-example-v1",
        density_representation="salted-ri-gto",
    )

    assert source.surface_points_bohr.flags.writeable is False
    assert source.surface_potential_hartree_per_e.flags.writeable is False
    assert source.provenance == {
        "solute_source": "frozen-density-surface-mep",
        "source_model": "salted-example-v1",
        "density_representation": "salted-ri-gto",
        "polarization_response": "fixed",
        "declared_total_charge_e": 0.0,
        "observed_total_charge_e": 2.0e-9,
        "absolute_charge_residual_e": 2.0e-9,
    }

    state = solve_frozen_density_mep_continuum(
        _ReciprocalSurfaceResponse(points),
        source,
    )

    assert state.source is source
    np.testing.assert_array_equal(
        state.response_state.surface_potential_hartree_per_e,
        potential,
    )
    assert state.response_state.polarization_energy_hartree < 0.0


def test_frozen_density_mep_source_rejects_charge_or_surface_drift():
    points = np.asarray(
        [
            [-3.0, 0.0, 0.2],
            [2.7, 0.4, -0.1],
            [0.1, -2.8, 0.7],
        ]
    )
    with pytest.raises(ValueError, match="declared molecular charge"):
        FrozenDensityMEPSource(
            surface_points_bohr=points,
            surface_potential_hartree_per_e=np.asarray([0.4, -0.2, 0.1]),
            declared_total_charge_e=0.0,
            observed_total_charge_e=2.0e-3,
            source_model="salted-example-v1",
            density_representation="salted-ri-gto",
        )

    source = FrozenDensityMEPSource(
        surface_points_bohr=points,
        surface_potential_hartree_per_e=np.asarray([0.4, -0.2, 0.1]),
        declared_total_charge_e=0.0,
        observed_total_charge_e=0.0,
        source_model="salted-example-v1",
        density_representation="salted-ri-gto",
    )
    shifted = points.copy()
    shifted[0, 0] += 1.0e-12
    with pytest.raises(ValueError, match="exact cavity surface"):
        solve_frozen_density_mep_continuum(
            _ReciprocalSurfaceResponse(shifted),
            source,
        )
