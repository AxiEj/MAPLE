from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_derivative import (
    fixed_cavity_energy_density_gradient,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    project_neutral_density_tangent,
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
