from __future__ import annotations

import numpy as np
import pytest
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
    FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION,
    assemble_total_solvation_coordinate_gradient,
    continuum_coupled_solvation_coordinate_gradient,
    fixed_cavity_energy_density_gradient,
    fixed_cavity_model_feature_energy_density_gradient,
    fixed_surface_solvation_coordinate_gradient,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    NeutralDensityCoordinates,
    UnmixedDensityResidualLinearization,
    project_neutral_density_tangent,
    solve_adjoint,
)


class _MatrixReactionField:
    def __init__(
        self,
        matrix: np.ndarray,
        atom_count: int,
        *,
        reciprocal: bool = True,
    ):
        self.matrix = np.asarray(matrix, dtype=float)
        self.atom_count = atom_count
        self.reciprocal_energy_pairing = reciprocal

    def apply(self, density: np.ndarray) -> np.ndarray:
        return (self.matrix @ np.asarray(density).reshape(-1)).reshape(
            self.atom_count,
            4,
        )

    def adjoint(self, field_cotangent: np.ndarray) -> np.ndarray:
        return (
            self.matrix.T @ np.asarray(field_cotangent).reshape(-1)
        ).reshape(self.atom_count, 4)


class _CoordinateDependentMatrixReactionField(_MatrixReactionField):
    full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )

    def __init__(
        self,
        matrix: np.ndarray,
        coordinate_derivative: np.ndarray,
        atom_count: int,
        *,
        fixed_surface_coordinate_derivative: np.ndarray | None = None,
    ):
        super().__init__(matrix, atom_count)
        self.coordinate_derivative = np.asarray(
            coordinate_derivative,
            dtype=float,
        )
        if fixed_surface_coordinate_derivative is None:
            fixed_surface_coordinate_derivative = coordinate_derivative
        self.fixed_surface_coordinate_derivative = np.asarray(
            fixed_surface_coordinate_derivative,
            dtype=float,
        )
        self.fixed_surface_position_vjp_calls = 0
        self.full_position_vjp_calls = 0

    def position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.fixed_surface_position_vjp_calls += 1
        result = np.zeros((self.atom_count, 3), dtype=float)
        result[0, 0] = np.vdot(
            np.asarray(field_cotangent).reshape(-1),
            self.fixed_surface_coordinate_derivative
            @ np.asarray(density).reshape(-1),
        )
        return result

    def full_position_vjp(
        self,
        density: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.full_position_vjp_calls += 1
        result = np.zeros((self.atom_count, 3), dtype=float)
        result[0, 0] = np.vdot(
            np.asarray(field_cotangent).reshape(-1),
            self.coordinate_derivative @ np.asarray(density).reshape(-1),
        )
        return result


class _MatrixDensityResponse:
    def __init__(self, matrix: np.ndarray, atom_count: int):
        self.matrix = np.asarray(matrix, dtype=float)
        self.atom_count = atom_count

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        return (
            self.matrix @ np.asarray(field_direction).reshape(-1)
        ).reshape(self.atom_count, 4)

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        return (
            self.matrix.T @ np.asarray(density_cotangent).reshape(-1)
        ).reshape(self.atom_count, 4)


class _MatrixModelFeatureReactionField:
    def __init__(
        self,
        feature_matrix: np.ndarray,
        atom_count: int,
        feature_count: int,
        *,
        reciprocal: bool = True,
    ):
        self.feature_matrix = np.asarray(feature_matrix, dtype=float)
        self.atom_count = atom_count
        self.model_feature_count = feature_count
        self.reciprocal_energy_pairing = reciprocal

    def model_feature_vjp(
        self,
        feature_cotangent: np.ndarray,
    ) -> np.ndarray:
        return (
            self.feature_matrix.T
            @ np.asarray(feature_cotangent).reshape(-1)
        ).reshape(self.atom_count, 4)


def _external_to_density_order_matrix(atom_count: int) -> np.ndarray:
    block = np.eye(4)[[0, 2, 3, 1]]
    return np.kron(np.eye(atom_count), block)


def test_external_field_to_density_order_matches_mace_l1_convention():
    field = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    converted = external_field_to_density_order(field)

    np.testing.assert_array_equal(
        converted,
        [
            [1.0, 3.0, 4.0, 2.0],
            [5.0, 7.0, 8.0, 6.0],
        ],
    )
    np.testing.assert_array_equal(field[0], [1.0, 2.0, 3.0, 4.0])


def test_density_to_external_field_order_is_the_inverse_transpose_permutation():
    density = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    converted = density_to_external_field_order(density)

    np.testing.assert_array_equal(
        converted,
        [
            [1.0, 4.0, 2.0, 3.0],
            [5.0, 8.0, 6.0, 7.0],
        ],
    )
    np.testing.assert_array_equal(
        external_field_to_density_order(converted),
        density,
    )
    np.testing.assert_array_equal(
        density_to_external_field_order(
            external_field_to_density_order(density)
        ),
        density,
    )


def test_fixed_cavity_energy_density_gradient_matches_neutral_finite_difference():
    atom_count = 3
    dimension = 4 * atom_count
    rng = np.random.default_rng(20260724)
    order = _external_to_density_order_matrix(atom_count)

    symmetric_kernel_seed = rng.normal(scale=0.05, size=(dimension, dimension))
    symmetric_kernel = symmetric_kernel_seed + symmetric_kernel_seed.T
    reaction_matrix = order.T @ symmetric_kernel
    reaction_field = _MatrixReactionField(reaction_matrix, atom_count)

    intrinsic_hessian_seed = rng.normal(
        scale=0.03,
        size=(dimension, dimension),
    )
    intrinsic_hessian = (
        intrinsic_hessian_seed + intrinsic_hessian_seed.T
    )
    intrinsic_linear = rng.normal(scale=0.2, size=dimension)
    density = rng.normal(scale=0.1, size=(atom_count, 4))
    field = reaction_field.apply(density)
    flat_field = field.reshape(-1)
    intrinsic_gradient = (
        intrinsic_hessian @ flat_field + intrinsic_linear
    ).reshape(atom_count, 4)

    analytic = fixed_cavity_energy_density_gradient(
        reaction_field,
        reaction_field_values=field,
        intrinsic_energy_field_gradient=intrinsic_gradient,
    )
    expected = project_neutral_density_tangent(
        (
            reaction_matrix.T @ intrinsic_gradient.reshape(-1)
            + order @ flat_field
        ).reshape(atom_count, 4)
    )
    np.testing.assert_allclose(analytic, expected, rtol=1.0e-13, atol=1.0e-13)

    direction = project_neutral_density_tangent(
        rng.normal(size=(atom_count, 4))
    )

    def energy(coefficients: np.ndarray) -> float:
        local_field = reaction_field.apply(coefficients).reshape(-1)
        intrinsic = (
            0.5 * local_field @ intrinsic_hessian @ local_field
            + intrinsic_linear @ local_field
        )
        polarization = (
            0.5
            * coefficients.reshape(-1)
            @ order
            @ local_field
        )
        return float(intrinsic + polarization)

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            energy(density + step * direction)
            - energy(density - step * direction)
        ) / (2.0 * step)
        assert finite_difference == pytest.approx(
            np.vdot(analytic, direction),
            rel=2.0e-10,
            abs=2.0e-11,
        )


@pytest.mark.parametrize(
    ("field", "gradient"),
    [
        (np.zeros((2, 4)), np.zeros((3, 4))),
        (np.full((3, 4), np.nan), np.zeros((3, 4))),
        (np.zeros((3, 4)), np.full((3, 4), np.inf)),
    ],
)
def test_fixed_cavity_energy_density_gradient_rejects_invalid_blocks(
    field,
    gradient,
):
    reaction_field = _MatrixReactionField(np.eye(12), 3)

    with pytest.raises(ValueError, match="finite with shape"):
        fixed_cavity_energy_density_gradient(
            reaction_field,
            reaction_field_values=field,
            intrinsic_energy_field_gradient=gradient,
        )


def test_fixed_cavity_energy_density_gradient_requires_reciprocity():
    reaction_field = _MatrixReactionField(
        np.eye(12),
        3,
        reciprocal=False,
    )

    with pytest.raises(ValueError, match="MATRIXSYMM=TRUE"):
        fixed_cavity_energy_density_gradient(
            reaction_field,
            reaction_field_values=np.zeros((3, 4)),
            intrinsic_energy_field_gradient=np.zeros((3, 4)),
        )


def test_model_feature_energy_density_gradient_matches_neutral_fd():
    atom_count = 3
    feature_count = 8
    density_dimension = 4 * atom_count
    feature_dimension = feature_count * atom_count
    rng = np.random.default_rng(20260727)
    order = _external_to_density_order_matrix(atom_count)

    symmetric_kernel_seed = rng.normal(
        scale=0.05,
        size=(density_dimension, density_dimension),
    )
    symmetric_kernel = symmetric_kernel_seed + symmetric_kernel_seed.T
    energy_dual_reaction = order.T @ symmetric_kernel
    feature_matrix = rng.normal(
        scale=0.04,
        size=(feature_dimension, density_dimension),
    )
    reaction_field = _MatrixModelFeatureReactionField(
        feature_matrix,
        atom_count,
        feature_count,
    )
    intrinsic_hessian_seed = rng.normal(
        scale=0.03,
        size=(feature_dimension, feature_dimension),
    )
    intrinsic_hessian = intrinsic_hessian_seed + intrinsic_hessian_seed.T
    intrinsic_linear = rng.normal(scale=0.2, size=feature_dimension)
    density = rng.normal(scale=0.1, size=(atom_count, 4))
    flat_density = density.reshape(-1)
    field = (energy_dual_reaction @ flat_density).reshape(atom_count, 4)
    features = feature_matrix @ flat_density
    intrinsic_gradient = (
        intrinsic_hessian @ features + intrinsic_linear
    ).reshape(atom_count, feature_count)

    analytic = fixed_cavity_model_feature_energy_density_gradient(
        reaction_field,
        reaction_field_values=field,
        intrinsic_energy_feature_gradient=intrinsic_gradient,
    )
    expected = project_neutral_density_tangent(
        (
            feature_matrix.T @ intrinsic_gradient.reshape(-1)
            + order @ field.reshape(-1)
        ).reshape(atom_count, 4)
    )
    np.testing.assert_allclose(
        analytic,
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    direction = project_neutral_density_tangent(
        rng.normal(size=(atom_count, 4))
    )

    def energy(coefficients: np.ndarray) -> float:
        flat = coefficients.reshape(-1)
        model_features = feature_matrix @ flat
        intrinsic = (
            0.5 * model_features @ intrinsic_hessian @ model_features
            + intrinsic_linear @ model_features
        )
        dual_field = energy_dual_reaction @ flat
        polarization = 0.5 * flat @ order @ dual_field
        return float(intrinsic + polarization)

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            energy(density + step * direction)
            - energy(density - step * direction)
        ) / (2.0 * step)
        assert finite_difference == pytest.approx(
            np.vdot(analytic, direction),
            rel=2.0e-10,
            abs=2.0e-11,
        )


def test_model_feature_energy_density_gradient_fails_closed():
    atom_count = 2
    feature_count = 3
    reaction_field = _MatrixModelFeatureReactionField(
        np.zeros((atom_count * feature_count, atom_count * 4)),
        atom_count,
        feature_count,
        reciprocal=False,
    )
    field = np.zeros((atom_count, 4))
    feature_gradient = np.zeros((atom_count, feature_count))

    with pytest.raises(ValueError, match="reciprocal"):
        fixed_cavity_model_feature_energy_density_gradient(
            reaction_field,
            reaction_field_values=field,
            intrinsic_energy_feature_gradient=feature_gradient,
        )

    reaction_field.reciprocal_energy_pairing = True
    reaction_field.model_feature_count = 0
    with pytest.raises(ValueError, match="count must be positive"):
        fixed_cavity_model_feature_energy_density_gradient(
            reaction_field,
            reaction_field_values=field,
            intrinsic_energy_feature_gradient=np.zeros((atom_count, 0)),
        )


def test_fixed_surface_coupled_coordinate_gradient_matches_resolved_root_fd():
    atom_count = 2
    dimension = 4 * atom_count
    rng = np.random.default_rng(20260725)
    order = _external_to_density_order_matrix(atom_count)
    coordinates = NeutralDensityCoordinates(atom_count)
    neutral_basis = np.column_stack(
        [
            coordinates.expand(np.eye(coordinates.dimension)[index]).reshape(-1)
            for index in range(coordinates.dimension)
        ]
    )

    kernel0_seed = rng.normal(scale=0.025, size=(dimension, dimension))
    kernel1_seed = rng.normal(scale=0.008, size=(dimension, dimension))
    kernel0 = kernel0_seed + kernel0_seed.T
    kernel1 = kernel1_seed + kernel1_seed.T
    response0 = order.T @ kernel0
    response1 = order.T @ kernel1

    density_response0 = neutral_basis @ rng.normal(
        scale=0.018,
        size=(coordinates.dimension, dimension),
    )
    density_response1 = neutral_basis @ rng.normal(
        scale=0.006,
        size=(coordinates.dimension, dimension),
    )
    base0 = neutral_basis @ rng.normal(
        scale=0.08,
        size=coordinates.dimension,
    )
    base1 = neutral_basis @ rng.normal(
        scale=0.025,
        size=coordinates.dimension,
    )

    intrinsic_hessian_seed = rng.normal(
        scale=0.02,
        size=(dimension, dimension),
    )
    intrinsic_hessian = intrinsic_hessian_seed + intrinsic_hessian_seed.T
    intrinsic_linear = rng.normal(scale=0.12, size=dimension)
    intrinsic_coordinate_field = rng.normal(scale=0.04, size=dimension)
    intrinsic_coordinate_quadratic = 0.35
    gas_linear = -0.17
    gas_quadratic = 0.21
    coordinate_value = 0.23

    def state_at(value: float):
        response_matrix = response0 + value * response1
        density_response_matrix = (
            density_response0 + value * density_response1
        )
        base = base0 + value * base1
        reduced_matrix = (
            np.eye(coordinates.dimension)
            - neutral_basis.T
            @ density_response_matrix
            @ response_matrix
            @ neutral_basis
        )
        reduced_rhs = neutral_basis.T @ base
        density = neutral_basis @ np.linalg.solve(reduced_matrix, reduced_rhs)
        field = response_matrix @ density
        residual = density - (
            base + density_response_matrix @ field
        )
        np.testing.assert_allclose(
            residual,
            np.zeros_like(residual),
            rtol=0.0,
            atol=2.0e-14,
        )
        return (
            density,
            field,
            response_matrix,
            density_response_matrix,
        )

    def energy_at(value: float) -> float:
        density, field, _, _ = state_at(value)
        intrinsic = (
            0.5 * field @ intrinsic_hessian @ field
            + (intrinsic_linear + value * intrinsic_coordinate_field) @ field
            + 0.5 * intrinsic_coordinate_quadratic * value**2
        )
        gas = gas_linear * value + 0.5 * gas_quadratic * value**2
        polarization = 0.5 * density @ order @ field
        return float(intrinsic - gas + polarization)

    (
        density,
        field,
        response_matrix,
        density_response_matrix,
    ) = state_at(coordinate_value)
    reaction_field = _CoordinateDependentMatrixReactionField(
        response_matrix,
        response1,
        atom_count,
    )
    density_response = _MatrixDensityResponse(
        density_response_matrix,
        atom_count,
    )
    intrinsic_field_gradient = (
        intrinsic_hessian @ field
        + intrinsic_linear
        + coordinate_value * intrinsic_coordinate_field
    ).reshape(atom_count, 4)
    energy_density_gradient = fixed_cavity_energy_density_gradient(
        reaction_field,
        reaction_field_values=field.reshape(atom_count, 4),
        intrinsic_energy_field_gradient=intrinsic_field_gradient,
    )
    linearization = UnmixedDensityResidualLinearization(
        atom_count=atom_count,
        reaction_field=reaction_field,
        density_response=density_response,
    )
    adjoint = solve_adjoint(
        linearization,
        energy_density_gradient,
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-13,
    ).solution

    direct_density_coordinate_derivative = (
        base1 + density_response1 @ field
    )
    adjoint_density_position_vjp = np.zeros((atom_count, 3))
    adjoint_density_position_vjp[0, 0] = np.vdot(
        adjoint.reshape(-1),
        direct_density_coordinate_derivative,
    )
    intrinsic_fixed_field_coordinate_gradient = (
        intrinsic_coordinate_field @ field
        + intrinsic_coordinate_quadratic * coordinate_value
    )
    gas_coordinate_gradient = (
        gas_linear + gas_quadratic * coordinate_value
    )
    solvent_fixed_field_forces = np.zeros((atom_count, 3))
    solvent_fixed_field_forces[0, 0] = (
        -intrinsic_fixed_field_coordinate_gradient
    )
    gas_forces = np.zeros((atom_count, 3))
    gas_forces[0, 0] = -gas_coordinate_gradient

    analytic = fixed_surface_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        density_coefficients=density.reshape(atom_count, 4),
        intrinsic_energy_field_gradient=intrinsic_field_gradient,
        adjoint_solution=adjoint,
        adjoint_density_position_vjp=adjoint_density_position_vjp,
        solvent_fixed_field_forces_ev_per_angstrom=(
            solvent_fixed_field_forces
        ),
        gas_forces_ev_per_angstrom=gas_forces,
    )
    continuum_analytic = continuum_coupled_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        density_coefficients=density.reshape(atom_count, 4),
        intrinsic_energy_field_gradient=intrinsic_field_gradient,
        adjoint_solution=adjoint,
        adjoint_density_position_vjp=adjoint_density_position_vjp,
        solvent_fixed_field_forces_ev_per_angstrom=(
            solvent_fixed_field_forces
        ),
        gas_forces_ev_per_angstrom=gas_forces,
    )
    np.testing.assert_allclose(
        continuum_analytic,
        analytic,
        rtol=0.0,
        atol=0.0,
    )

    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        finite_difference = (
            energy_at(coordinate_value + step)
            - energy_at(coordinate_value - step)
        ) / (2.0 * step)
        assert analytic[0, 0] == pytest.approx(
            finite_difference,
            rel=2.0e-9,
            abs=2.0e-10,
        )
    np.testing.assert_allclose(
        analytic[1:, :],
        np.zeros((atom_count - 1, 3)),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_continuum_coupled_gradient_uses_full_reaction_field_vjp():
    atom_count = 2
    dimension = atom_count * 4
    rng = np.random.default_rng(20260725)
    response = rng.normal(scale=0.04, size=(dimension, dimension))
    full_coordinate_derivative = rng.normal(
        scale=0.02,
        size=(dimension, dimension),
    )
    reaction_field = _CoordinateDependentMatrixReactionField(
        response,
        full_coordinate_derivative,
        atom_count,
        fixed_surface_coordinate_derivative=np.zeros_like(
            full_coordinate_derivative
        ),
    )
    density_response = _MatrixDensityResponse(
        np.zeros((dimension, dimension)),
        atom_count,
    )
    density = project_neutral_density_tangent(
        rng.normal(scale=0.1, size=(atom_count, 4))
    )
    field_gradient = rng.normal(scale=0.2, size=(atom_count, 4))
    adjoint = np.zeros((atom_count, 4))
    zero_coordinates = np.zeros((atom_count, 3))

    fixed_surface = fixed_surface_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        density_coefficients=density,
        intrinsic_energy_field_gradient=field_gradient,
        adjoint_solution=adjoint,
        adjoint_density_position_vjp=zero_coordinates,
        solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
        gas_forces_ev_per_angstrom=zero_coordinates,
    )
    continuum = continuum_coupled_solvation_coordinate_gradient(
        reaction_field,
        density_response,
        density_coefficients=density,
        intrinsic_energy_field_gradient=field_gradient,
        adjoint_solution=adjoint,
        adjoint_density_position_vjp=zero_coordinates,
        solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
        gas_forces_ev_per_angstrom=zero_coordinates,
    )

    combined_field_cotangent = (
        field_gradient + 0.5 * density_to_external_field_order(density)
    )
    expected = np.zeros((atom_count, 3))
    expected[0, 0] = np.vdot(
        combined_field_cotangent.reshape(-1),
        full_coordinate_derivative @ density.reshape(-1),
    )
    np.testing.assert_allclose(fixed_surface, zero_coordinates)
    np.testing.assert_allclose(continuum, expected)
    assert reaction_field.fixed_surface_position_vjp_calls == 1
    assert reaction_field.full_position_vjp_calls == 1


def test_continuum_coupled_gradient_fails_closed_without_full_derivative():
    atom_count = 2
    dimension = atom_count * 4
    reaction_field = _MatrixReactionField(np.eye(dimension), atom_count)
    density_response = _MatrixDensityResponse(np.eye(dimension), atom_count)
    zero_density = np.zeros((atom_count, 4))
    zero_coordinates = np.zeros((atom_count, 3))

    with pytest.raises(
        NotImplementedError,
        match="full reaction-field coordinate derivative",
    ):
        continuum_coupled_solvation_coordinate_gradient(
            reaction_field,
            density_response,
            density_coefficients=zero_density,
            intrinsic_energy_field_gradient=zero_density,
            adjoint_solution=zero_density,
            adjoint_density_position_vjp=zero_coordinates,
            solvent_fixed_field_forces_ev_per_angstrom=zero_coordinates,
            gas_forces_ev_per_angstrom=zero_coordinates,
        )


def test_continuum_coupled_gradient_rejects_bad_contract_or_output():
    atom_count = 2
    dimension = atom_count * 4
    reaction_field = _CoordinateDependentMatrixReactionField(
        np.eye(dimension),
        np.zeros((dimension, dimension)),
        atom_count,
    )
    density_response = _MatrixDensityResponse(np.eye(dimension), atom_count)
    zero_density = np.zeros((atom_count, 4))
    zero_coordinates = np.zeros((atom_count, 3))
    arguments = {
        "density_coefficients": zero_density,
        "intrinsic_energy_field_gradient": zero_density,
        "adjoint_solution": zero_density,
        "adjoint_density_position_vjp": zero_coordinates,
        "solvent_fixed_field_forces_ev_per_angstrom": zero_coordinates,
        "gas_forces_ev_per_angstrom": zero_coordinates,
    }

    reaction_field.full_position_derivative_contract_version = 99
    with pytest.raises(
        ValueError,
        match="full reaction-field coordinate-derivative contract version",
    ):
        continuum_coupled_solvation_coordinate_gradient(
            reaction_field,
            density_response,
            **arguments,
        )

    reaction_field.full_position_derivative_contract_version = (
        FULL_REACTION_FIELD_POSITION_DERIVATIVE_CONTRACT_VERSION
    )
    reaction_field.full_position_vjp = lambda density, cotangent: np.full(
        (atom_count, 3),
        np.nan,
    )
    with pytest.raises(
        ValueError,
        match="full reaction-field position VJP",
    ):
        continuum_coupled_solvation_coordinate_gradient(
            reaction_field,
            density_response,
            **arguments,
        )


def test_total_solvation_gradient_assembly_converts_units_and_force_sign():
    continuum_gradient_ev_per_angstrom = Hartree * np.asarray(
        [
            [1.0, -2.0, 0.5],
            [-0.25, 0.75, 1.5],
        ]
    )
    cds_gradient_hartree_per_angstrom = np.asarray(
        [
            [0.1, 0.2, -0.3],
            [0.4, -0.5, 0.6],
        ]
    )

    result = assemble_total_solvation_coordinate_gradient(
        continuum_gradient_ev_per_angstrom,
        cds_gradient_hartree_per_angstrom,
    )

    continuum_hartree = continuum_gradient_ev_per_angstrom / Hartree
    total_gradient = continuum_hartree + cds_gradient_hartree_per_angstrom
    np.testing.assert_allclose(
        result.continuum_position_gradient_hartree_per_angstrom,
        continuum_hartree,
    )
    np.testing.assert_allclose(
        result.cds_position_gradient_hartree_per_angstrom,
        cds_gradient_hartree_per_angstrom,
    )
    np.testing.assert_allclose(
        result.total_position_gradient_hartree_per_angstrom,
        total_gradient,
    )
    np.testing.assert_allclose(
        result.solvent_correction_forces_hartree_per_angstrom,
        -total_gradient,
    )
    with pytest.raises(ValueError, match="read-only"):
        result.total_position_gradient_hartree_per_angstrom[0, 0] = 0.0

    gas_forces = np.full((2, 3), 7.0)
    solution_forces = (
        gas_forces
        + result.solvent_correction_forces_hartree_per_angstrom
    )
    np.testing.assert_allclose(
        solution_forces,
        gas_forces - continuum_hartree - cds_gradient_hartree_per_angstrom,
    )


def test_total_solvation_gradient_assembly_matches_component_energy_fd():
    continuum_slope_ev_per_angstrom = np.asarray([[2.5, -1.5, 0.75]])
    cds_slope_hartree_per_angstrom = np.asarray([[0.03, 0.04, -0.02]])
    result = assemble_total_solvation_coordinate_gradient(
        continuum_slope_ev_per_angstrom,
        cds_slope_hartree_per_angstrom,
    )

    def energy_hartree(coordinates: np.ndarray) -> float:
        continuum = float(
            np.vdot(
                continuum_slope_ev_per_angstrom,
                coordinates,
            )
        ) / Hartree
        cds = float(
            np.vdot(
                cds_slope_hartree_per_angstrom,
                coordinates,
            )
        )
        return continuum + cds

    coordinates = np.asarray([[0.2, -0.4, 0.6]])
    finite_difference = np.empty_like(coordinates)
    step = 1.0e-5
    for atom_index in range(coordinates.shape[0]):
        for axis in range(3):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[atom_index, axis] += step
            minus[atom_index, axis] -= step
            finite_difference[atom_index, axis] = (
                energy_hartree(plus) - energy_hartree(minus)
            ) / (2.0 * step)

    np.testing.assert_allclose(
        result.total_position_gradient_hartree_per_angstrom,
        finite_difference,
        rtol=2.0e-11,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    ("continuum", "cds", "message"),
    [
        (
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            "continuum_position_gradient_ev_per_angstrom",
        ),
        (
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            "continuum_position_gradient_ev_per_angstrom",
        ),
        (
            np.zeros((2, 3)),
            np.zeros((3, 3)),
            "cds_position_gradient_hartree_per_angstrom",
        ),
        (
            np.zeros((2, 3)),
            np.full((2, 3), np.nan),
            "cds_position_gradient_hartree_per_angstrom",
        ),
    ],
)
def test_total_solvation_gradient_assembly_rejects_invalid_components(
    continuum,
    cds,
    message,
):
    with pytest.raises(ValueError, match=message):
        assemble_total_solvation_coordinate_gradient(continuum, cds)
