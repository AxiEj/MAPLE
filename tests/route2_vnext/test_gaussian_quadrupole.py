from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.coupling.gaussian_quadrupole import (
    gaussian_traceless_quadrupole_potential_torch,
    gaussian_traceless_quadrupole_surface_operator,
)
from maple.solvation.coupling.point_quadrupole import (
    point_traceless_quadrupole_surface_operator,
    quadrupole_rotation_matrix,
)


def _rotation() -> np.ndarray:
    angle = 0.57
    return np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def test_gaussian_quadrupole_has_the_point_limit_and_rotation_covariance() -> None:
    generator = np.random.default_rng(20260827)
    centers = generator.normal(size=(3, 3))
    points = generator.normal(size=(19, 3)) * 2.0 + np.asarray([8.0, -6.0, 7.0])
    coefficients = generator.normal(size=(3, 5))
    point = point_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=centers,
    ) @ coefficients.reshape(-1)
    narrow = gaussian_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=centers,
        sigma_angstrom=1.0e-3,
    ) @ coefficients.reshape(-1)
    np.testing.assert_allclose(narrow, point, rtol=2.0e-14, atol=2.0e-14)

    rotation = _rotation()
    representation = quadrupole_rotation_matrix(rotation)
    reference = gaussian_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=centers,
        sigma_angstrom=1.5,
    ) @ coefficients.reshape(-1)
    rotated = gaussian_traceless_quadrupole_surface_operator(
        points_bohr=points @ rotation.T,
        centers_angstrom=centers @ rotation.T,
        sigma_angstrom=1.5,
    ) @ (coefficients @ representation.T).reshape(-1)
    np.testing.assert_allclose(rotated, reference, rtol=3.0e-14, atol=3.0e-14)


def test_gaussian_quadrupole_torch_matches_numpy_and_has_coordinate_gradient() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    points_np = np.asarray([[4.2, -1.3, 2.1], [5.0, 0.7, -3.4], [-2.2, 4.1, 3.7]])
    centers_np = np.asarray([[0.1, -0.2, 0.3], [1.2, 0.4, -0.5]])
    coefficients_np = np.asarray(
        [[0.2, -0.1, 0.05, 0.04, -0.03], [-0.07, 0.11, 0.02, -0.06, 0.08]]
    )
    expected = gaussian_traceless_quadrupole_surface_operator(
        points_bohr=points_np,
        centers_angstrom=centers_np,
        sigma_angstrom=1.5,
    ) @ coefficients_np.reshape(-1)
    points = torch.tensor(points_np, dtype=dtype)
    centers = torch.tensor(centers_np, dtype=dtype, requires_grad=True)
    coefficients = torch.tensor(coefficients_np, dtype=dtype)
    values = gaussian_traceless_quadrupole_potential_torch(
        points_bohr=points,
        centers_angstrom=centers,
        coefficients_eangstrom2=coefficients,
        sigma_angstrom=1.5,
    )
    torch.testing.assert_close(
        values,
        torch.tensor(expected, dtype=dtype),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    direction = torch.tensor(
        [[0.03, -0.02, 0.01], [-0.01, 0.04, 0.02]], dtype=dtype
    )
    gradient = torch.autograd.grad(values.sum(), centers)[0]
    step = 2.0e-6
    plus = gaussian_traceless_quadrupole_potential_torch(
        points_bohr=points,
        centers_angstrom=centers.detach() + step * direction,
        coefficients_eangstrom2=coefficients,
        sigma_angstrom=1.5,
    ).sum()
    minus = gaussian_traceless_quadrupole_potential_torch(
        points_bohr=points,
        centers_angstrom=centers.detach() - step * direction,
        coefficients_eangstrom2=coefficients,
        sigma_angstrom=1.5,
    ).sum()
    finite_difference = (plus - minus) / (2.0 * step)
    assert torch.dot(gradient.reshape(-1), direction.reshape(-1)) == pytest.approx(
        float(finite_difference), rel=2.0e-8, abs=2.0e-10
    )
