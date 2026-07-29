from __future__ import annotations

import math

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_qeq_monopole import (
    RAPPE_GODDARD_QEQ_PARAMETER_SHA256,
    RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION,
    build_same_basis_qeq_hardness_curvature,
    embed_mace_polar_l1_density_in_single_radial_gto,
    evaluate_route2_v0_rappe_goddard_monopole,
)


class _ReciprocalSurfaceResponse:
    contract_version = EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    energy_response_is_reciprocal = True

    def __init__(
        self,
        positions_angstrom: np.ndarray,
        surface_points_bohr: np.ndarray,
        response_matrix: np.ndarray,
    ) -> None:
        self.atom_count = len(positions_angstrom)
        self._positions = np.asarray(positions_angstrom, dtype=float)
        self._points = np.asarray(surface_points_bohr, dtype=float)
        self._response = np.asarray(response_matrix, dtype=float)

    @property
    def atomic_numbers(self):
        return np.ones(self.atom_count)

    @property
    def reference_positions_bohr(self):
        return self._positions / Bohr

    @property
    def cavity_radii_angstrom(self):
        return np.full(self.atom_count, 1.5)

    @property
    def surface_points_bohr(self):
        return self._points.copy()

    @property
    def surface_areas_bohr2(self):
        return np.ones(len(self._points))

    def apply_energy_conjugate(self, potential):
        return self._response @ np.asarray(potential, dtype=float)


def _operator(
    basis: AtomCenteredL1GTOBasis | None = None,
) -> FixedCavityGTOGalerkinOperator:
    positions = np.asarray([[-0.7, 0.1, 0.0], [0.8, -0.2, 0.3]], dtype=float)
    rng = np.random.default_rng(20260729)
    points = rng.normal(size=(18, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= rng.uniform(4.0, 6.0, size=(18, 1))
    factor = rng.normal(size=(18, 18))
    response = -(factor.T @ factor) / 160.0
    return FixedCavityGTOGalerkinOperator(
        _ReciprocalSurfaceResponse(positions, points, response),
        positions,
        basis or AtomCenteredL1GTOBasis((MACE_POLAR_DENSITY_SIGMA_ANGSTROM,)),
    )


def test_published_qeq_hardness_table_is_pinned_and_rejects_unpublished_rows():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    curvature = build_same_basis_qeq_hardness_curvature(
        ("C", "O"),
        positions,
    )

    assert curvature.parameter_file_sha256 == RAPPE_GODDARD_QEQ_PARAMETER_SHA256
    assert curvature.symbols == ("C", "O")
    assert curvature.construction == RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION
    assert curvature.hardness_matrix_hartree_per_e2.flags.writeable is False

    with pytest.raises(ValueError, match="published Rappé-Goddard"):
        build_same_basis_qeq_hardness_curvature(
            ("C", "Xe"),
            positions,
        )


def test_qeq_hardness_uses_a_same_mace_gto_coulomb_metric():
    positions = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    curvature = build_same_basis_qeq_hardness_curvature(
        ("C", "O"),
        positions,
    )
    matrix = curvature.hardness_matrix_hartree_per_e2
    sigma_bohr = MACE_POLAR_DENSITY_SIGMA_ANGSTROM / Bohr
    distance_bohr = 1.4 / Bohr
    expected_pair = math.erf(distance_bohr / (2.0 * sigma_bohr)) / distance_bohr

    assert matrix[0, 0] == pytest.approx(10.126 / Hartree, abs=1.0e-15)
    assert matrix[1, 1] == pytest.approx(13.364 / Hartree, abs=1.0e-15)
    assert matrix[0, 1] == pytest.approx(expected_pair, abs=1.0e-15)
    assert matrix[1, 0] == pytest.approx(expected_pair, abs=1.0e-15)


def test_qeq_monopole_tangent_keeps_unmodeled_dipoles_frozen_exactly():
    operator = _operator()
    native_density = np.asarray(
        [[0.25, 0.12, -0.03, 0.07], [-0.25, -0.06, 0.08, -0.04]],
        dtype=float,
    )
    state = evaluate_route2_v0_rappe_goddard_monopole(
        symbols=("H", "C"),
        frozen_density_coefficients=embed_mace_polar_l1_density_in_single_radial_gto(
            native_density,
            operator.basis,
        ),
        operator=operator,
    )

    assert state.construction == RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION
    assert state.response_state.additional_constraint_residual_inf < 1.0e-12
    assert state.response_state.charge_constraint_residual_e < 1.0e-12
    np.testing.assert_allclose(
        state.response_state.induced_density_coefficients[:, :, 1:],
        np.zeros_like(state.response_state.induced_density_coefficients[:, :, 1:]),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.monopole_hardness_matrix_hartree_per_e2,
        state.qeq_curvature.hardness_matrix_hartree_per_e2,
        rtol=0.0,
        atol=0.0,
    )


def test_qeq_monopole_tangent_rejects_a_non_native_radial_density_basis():
    operator = _operator(AtomCenteredL1GTOBasis((1.5, 2.0)))
    frozen = np.zeros(operator.basis.coefficient_shape(operator.atom_count))

    with pytest.raises(ValueError, match="single-radial"):
        evaluate_route2_v0_rappe_goddard_monopole(
            symbols=("H", "C"),
            frozen_density_coefficients=frozen,
            operator=operator,
        )
