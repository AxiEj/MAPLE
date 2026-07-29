from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    SurfaceChargeState,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_frozen_density import (
    Route2V0FrozenDensityState,
    V0_FROZEN_DENSITY_CONSTRUCTION,
    evaluate_route2_v0_frozen_density,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (
    FixedCavityPCMReactionFieldLinearMap,
)


class _QuadraticContinuum:
    """Small reciprocal continuum oracle for the fixed-density scalar tests."""

    atom_count = 2
    energy_response_is_reciprocal = True
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)

    @property
    def surface_points_bohr(self) -> np.ndarray:
        return np.array(
            [[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            dtype=float,
        )[: self.matrix.shape[0]]

    @property
    def reference_positions_bohr(self) -> np.ndarray:
        return _positions() / Bohr

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.array([6, 8], dtype=float)

    def solve(self, potential: np.ndarray) -> SurfaceChargeState:
        values = np.asarray(potential, dtype=float)
        charge = self.matrix @ values
        return SurfaceChargeState(
            surface_potential_hartree_per_e=values,
            direct_surface_charge_e=charge,
            adjoint_surface_charge_e=charge,
            energy_conjugate_surface_charge_e=charge,
            polarization_energy_hartree=0.5 * float(values @ charge),
        )


def _density() -> np.ndarray:
    return np.array(
        [[0.17, 0.11, -0.02, 0.03], [-0.17, -0.04, 0.08, -0.01]],
        dtype=float,
    )


def _positions() -> np.ndarray:
    return np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=float)


def test_frozen_density_uses_the_single_energy_conjugate_continuum_scalar():
    continuum = _QuadraticContinuum(
        np.array(
            [[-0.4, 0.1, 0.0], [0.1, -0.3, 0.05], [0.0, 0.05, -0.2]],
            dtype=float,
        )
    )
    state = evaluate_route2_v0_frozen_density(
        density_coefficients=_density(),
        atom_positions_angstrom=_positions(),
        continuum=continuum,
    )

    potential = state.surface_potential_hartree_per_e
    np.testing.assert_allclose(
        potential,
        point_multipole_potential(
            continuum.surface_points_bohr,
            _positions(),
            _density(),
        ),
        rtol=0.0,
        atol=1.0e-15,
    )
    expected_charge = continuum.matrix @ potential
    expected_energy = 0.5 * float(potential @ expected_charge)
    assert state.construction == V0_FROZEN_DENSITY_CONSTRUCTION
    assert state.total_charge_e == pytest.approx(0.0, abs=1.0e-15)
    np.testing.assert_allclose(
        state.surface_charge_state.energy_conjugate_surface_charge_e,
        expected_charge,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert state.electrostatic_energy_hartree == pytest.approx(expected_energy)


def test_frozen_density_scalar_is_quadratic_under_source_scaling():
    continuum = _QuadraticContinuum(-np.eye(3, dtype=float))
    density = _density()
    positions = _positions()
    scale = 1.7

    base = evaluate_route2_v0_frozen_density(
        density_coefficients=density,
        atom_positions_angstrom=positions,
        continuum=continuum,
    )
    scaled = evaluate_route2_v0_frozen_density(
        density_coefficients=scale * density,
        atom_positions_angstrom=positions,
        continuum=continuum,
        target_total_charge_e=0.0,
    )

    np.testing.assert_allclose(
        scaled.surface_charge_state.energy_conjugate_surface_charge_e,
        scale * base.surface_charge_state.energy_conjugate_surface_charge_e,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert scaled.electrostatic_energy_hartree == pytest.approx(
        scale**2 * base.electrostatic_energy_hartree
    )


def test_frozen_density_matches_the_legacy_frozen_electrostatic_term():
    continuum = _QuadraticContinuum(-np.eye(3, dtype=float))
    density = _density()
    positions = _positions()

    v0_fd = evaluate_route2_v0_frozen_density(
        density_coefficients=density,
        atom_positions_angstrom=positions,
        continuum=continuum,
    )
    legacy_snapshot = FixedCavityPCMReactionFieldLinearMap(
        continuum,
        positions,
    ).scf_snapshot(density)

    assert v0_fd.electrostatic_energy_hartree == pytest.approx(
        legacy_snapshot.polarization_energy_hartree,
        abs=1.0e-15,
    )
    np.testing.assert_allclose(
        v0_fd.surface_potential_hartree_per_e,
        legacy_snapshot.mep_hartree_per_e,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_frozen_density_rejects_charge_drift_instead_of_projecting_it():
    density = _density()
    density[0, 0] += 1.0e-6

    with pytest.raises(ValueError, match="violates its total-charge constraint"):
        evaluate_route2_v0_frozen_density(
            density_coefficients=density,
            atom_positions_angstrom=_positions(),
            continuum=_QuadraticContinuum(-np.eye(3, dtype=float)),
        )


def test_frozen_density_rejects_a_non_energy_conjugate_continuum_response():
    continuum = _QuadraticContinuum(-np.eye(3, dtype=float))
    continuum.energy_response_is_reciprocal = False

    with pytest.raises(ValueError, match="energy-conjugate reciprocal"):
        evaluate_route2_v0_frozen_density(
            density_coefficients=_density(),
            atom_positions_angstrom=_positions(),
            continuum=continuum,
        )


def test_frozen_density_rejects_source_geometry_that_differs_from_the_cavity():
    with pytest.raises(ValueError, match="source geometry does not match"):
        evaluate_route2_v0_frozen_density(
            density_coefficients=_density(),
            atom_positions_angstrom=_positions() + 1.0e-4,
            continuum=_QuadraticContinuum(-np.eye(3, dtype=float)),
        )


def test_frozen_density_rejects_an_unknown_continuum_contract():
    continuum = _QuadraticContinuum(-np.eye(3, dtype=float))
    continuum.contract_version = 999

    with pytest.raises(ValueError, match="Unsupported external-MEP"):
        evaluate_route2_v0_frozen_density(
            density_coefficients=_density(),
            atom_positions_angstrom=_positions(),
            continuum=continuum,
        )


def test_frozen_density_state_rejects_a_lie_about_its_density_charge():
    continuum = _QuadraticContinuum(-np.eye(3, dtype=float))
    potential = point_multipole_potential(
        continuum.surface_points_bohr,
        _positions(),
        _density(),
    )
    surface_charge_state = continuum.solve(potential)

    with pytest.raises(ValueError, match="must equal the supplied density"):
        Route2V0FrozenDensityState(
            density_coefficients=_density(),
            atom_positions_angstrom=_positions(),
            surface_potential_hartree_per_e=potential,
            surface_charge_state=surface_charge_state,
            target_total_charge_e=0.0,
            total_charge_e=0.3,
            electrostatic_energy_hartree=surface_charge_state.polarization_energy_hartree,
        )
