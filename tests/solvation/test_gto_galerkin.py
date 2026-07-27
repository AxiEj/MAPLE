from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    asc_reaction_potential_gradient,
    external_field_to_density_order,
    gaussian_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
)
from maple.function.calculator.extra_correction.implicit.pcm_energy_projection import (
    project_surface_potential_in_pcm_energy_norm,
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


def _geometry():
    positions = np.asarray(
        [[-0.8, 0.1, 0.2], [0.9, -0.3, 0.4]],
        dtype=float,
    )
    rng = np.random.default_rng(20260728)
    points = rng.normal(size=(18, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points *= rng.uniform(4.0, 6.0, size=(18, 1))
    factor = rng.normal(size=(18, 18))
    response = -(factor.T @ factor) / 50.0
    return positions, points, response


def test_gto_surface_operator_matches_analytic_gaussian_source():
    positions, points, _ = _geometry()
    basis = AtomCenteredL1GTOBasis((1.5, 3.0))
    rng = np.random.default_rng(9)
    coefficients = rng.normal(size=basis.coefficient_shape(len(positions)))
    matrix_value = basis.surface_operator(points, positions) @ coefficients.reshape(-1)
    analytic = sum(
        gaussian_multipole_potential(
            points,
            positions,
            coefficients[:, radial, :],
            sigma_angstrom=sigma,
        )
        for radial, sigma in enumerate(basis.sigmas_angstrom)
    )

    np.testing.assert_allclose(matrix_value, analytic, rtol=0.0, atol=2.0e-15)


def test_molecular_moment_constraints_match_hand_calculated_raw_order():
    positions = np.asarray(
        [[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]],
        dtype=float,
    )
    basis = AtomCenteredL1GTOBasis((1.5, 3.0))
    coefficients = np.zeros(basis.coefficient_shape(2))
    coefficients[0, 0] = [0.4, 0.1, 0.2, 0.3]
    coefficients[0, 1] = [-0.1, -0.2, 0.4, -0.5]
    coefficients[1, 0] = [0.2, 0.6, -0.3, 0.7]
    coefficients[1, 1] = [-0.5, -0.4, 0.8, -0.9]
    result = basis.molecular_charge_dipole_constraints(positions) @ (
        coefficients.reshape(-1)
    )
    charges = coefficients[:, :, 0]
    # Raw l=1 columns [1,2,3] represent Cartesian [y,z,x].
    intrinsic_dipoles = coefficients[:, :, 1:4][:, :, [2, 0, 1]]
    expected_charge = float(np.sum(charges))
    expected_dipole = np.einsum("ir,ix->x", charges, positions) + np.sum(
        intrinsic_dipoles,
        axis=(0, 1),
    )

    np.testing.assert_allclose(
        result,
        np.concatenate(([expected_charge], expected_dipole)),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_gto_surface_transpose_matches_analytic_receiver_fields():
    positions, points, _ = _geometry()
    basis = AtomCenteredL1GTOBasis((1.5, 3.0))
    rng = np.random.default_rng(17)
    surface_charge = rng.normal(size=len(points))
    matrix_dual = (
        basis.surface_operator(points, positions).T @ surface_charge
    ).reshape(basis.coefficient_shape(len(positions)))

    expected = np.empty_like(matrix_dual)
    for radial, sigma in enumerate(basis.sigmas_angstrom):
        potential, gradient_per_bohr = asc_reaction_potential_gradient(
            positions,
            points,
            surface_charge,
            sigma_angstrom=sigma,
        )
        expected[:, radial, :] = external_field_to_density_order(
            np.concatenate(
                (
                    potential[:, None],
                    gradient_per_bohr / Bohr,
                ),
                axis=1,
            )
        )

    np.testing.assert_allclose(matrix_dual, expected, rtol=0.0, atol=2.0e-14)


def test_same_basis_galerkin_operator_closes_reciprocity_and_energy():
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5, 3.0))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)
    rng = np.random.default_rng(31)
    left = rng.normal(size=basis.coefficient_shape(len(positions)))
    right = rng.normal(size=basis.coefficient_shape(len(positions)))

    assert np.vdot(left, operator.apply(right)) == pytest.approx(
        np.vdot(operator.apply(left), right),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    snapshot = operator.snapshot(left)
    assert snapshot.half_coupling_identity_error_hartree < 1.0e-12
    assert snapshot.polarization_energy_hartree == pytest.approx(
        0.5 * np.vdot(left, operator.apply(left)),
        abs=2.0e-13,
    )
    assert np.max(operator.eigenvalues_hartree) <= 1.0e-12
    assert operator.provenance["source_receiver_basis_identical"] is True
    assert operator.provenance["production_profile"] is False


def test_pcm_energy_projection_uses_affine_shifted_pythagorean_identity():
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)
    rng = np.random.default_rng(73)
    target_potential = rng.normal(size=len(points))
    projection = project_surface_potential_in_pcm_energy_norm(
        operator,
        target_potential,
        total_charge_e=0.0,
        molecular_dipole_e_angstrom=np.asarray([0.3, -0.2, 0.1]),
    )

    np.testing.assert_allclose(
        projection.achieved_constraints,
        projection.target_constraints,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert projection.constraint_residual_inf < 2.0e-12
    assert projection.retained_subspace_optimality_inf < 2.0e-11
    assert projection.shifted_pythagorean_error_hartree < 2.0e-11
    assert projection.residual_energy_norm_squared_hartree >= 0.0
    assert projection.target_polarization_energy_hartree < 0.0
    assert projection.fitted_polarization_energy_hartree < 0.0


def test_homogeneous_pcm_projection_recovers_absolute_energy_identity():
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)
    target = np.random.default_rng(101).normal(size=len(points))
    projection = project_surface_potential_in_pcm_energy_norm(
        operator,
        target,
        total_charge_e=0.0,
        molecular_dipole_e_angstrom=np.zeros(3),
    )

    assert projection.polarization_energy_error_hartree == pytest.approx(
        0.5 * projection.residual_energy_norm_squared_hartree,
        rel=2.0e-10,
        abs=2.0e-11,
    )


def test_pcm_projection_reports_truncated_spectral_conditioning():
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5, 3.0))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)
    target = np.random.default_rng(151).normal(size=len(points))
    common = {
        "total_charge_e": 0.0,
        "molecular_dipole_e_angstrom": np.asarray([0.2, -0.1, 0.3]),
    }
    tight = project_surface_potential_in_pcm_energy_norm(
        operator,
        target,
        relative_spectral_cutoff=1.0e-12,
        **common,
    )
    coarse = project_surface_potential_in_pcm_energy_norm(
        operator,
        target,
        relative_spectral_cutoff=1.0e-2,
        **common,
    )

    assert tight.reduced_dimension == coarse.reduced_dimension
    assert coarse.effective_rank < tight.effective_rank
    assert coarse.discarded_mode_count > tight.discarded_mode_count
    assert coarse.coefficient_l2_norm < tight.coefficient_l2_norm
    assert coarse.coefficient_max_abs < tight.coefficient_max_abs
    assert coarse.retained_condition_number < tight.retained_condition_number
    assert coarse.shifted_pythagorean_error_hartree < 1.0e-8
    assert coarse.retained_subspace_optimality_inf < 1.0e-9
    assert coarse.full_tangent_gradient_inf >= coarse.retained_subspace_optimality_inf


def test_pcm_energy_projection_rejects_a_positive_energy_response():
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        -response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)

    with pytest.raises(
        RuntimeError,
        match="positive-semidefinite PCM energy norm",
    ):
        project_surface_potential_in_pcm_energy_norm(
            operator,
            np.random.default_rng(211).normal(size=len(points)),
            total_charge_e=0.0,
            molecular_dipole_e_angstrom=np.zeros(3),
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_galerkin_operator_rejects_nonfinite_tolerances(invalid):
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))

    with pytest.raises(ValueError, match="Geometry tolerance"):
        FixedCavityGTOGalerkinOperator(
            response,
            positions,
            basis,
            geometry_tolerance_angstrom=invalid,
        )
    with pytest.raises(ValueError, match="Reciprocity tolerance"):
        FixedCavityGTOGalerkinOperator(
            response,
            positions,
            basis,
            reciprocity_tolerance_hartree=invalid,
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_galerkin_snapshot_rejects_nonfinite_identity_tolerance(invalid):
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)
    coefficients = np.zeros(basis.coefficient_shape(len(positions)))

    with pytest.raises(ValueError, match="Energy identity tolerance"):
        operator.snapshot(
            coefficients,
            identity_tolerance_hartree=invalid,
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "keyword",
    [
        "constraint_tolerance",
        "optimality_tolerance",
        "semidefinite_tolerance_hartree",
        "pythagorean_tolerance_hartree",
    ],
)
def test_pcm_projection_rejects_nonfinite_tolerances(keyword, invalid):
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)

    with pytest.raises(
        ValueError,
        match="Projection tolerances must be finite and positive",
    ):
        project_surface_potential_in_pcm_energy_norm(
            operator,
            np.random.default_rng(307).normal(size=len(points)),
            total_charge_e=0.0,
            molecular_dipole_e_angstrom=np.zeros(3),
            **{keyword: invalid},
        )


@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf, 0.0, -1.0, 1.0001],
)
def test_pcm_projection_rejects_invalid_spectral_cutoff(invalid):
    positions, points, response_matrix = _geometry()
    response = _ReciprocalSurfaceResponse(
        positions,
        points,
        response_matrix,
    )
    basis = AtomCenteredL1GTOBasis((1.5,))
    operator = FixedCavityGTOGalerkinOperator(response, positions, basis)

    with pytest.raises(ValueError, match="relative spectral cutoff"):
        project_surface_potential_in_pcm_energy_norm(
            operator,
            np.random.default_rng(401).normal(size=len(points)),
            total_charge_e=0.0,
            molecular_dipole_e_angstrom=np.zeros(3),
            relative_spectral_cutoff=invalid,
        )
