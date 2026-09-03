"""Independent numerical controls for the private Torch smooth-PCM tests.

This module intentionally does not import :mod:`torch_smooth_pcm`.  In
particular, the source kernels, raw/Cartesian permutation, linear solves and
finite-difference formulas below are NumPy equations rather than wrappers
around implementation-private helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping

import numpy as np
from ase.units import Bohr, Hartree
from scipy.special import lpmv


COULOMB_EV_ANGSTROM_PER_E2 = Hartree * Bohr
RAW_TO_CARTESIAN = np.asarray((0, 3, 1, 2), dtype=int)
FIELD_TO_RAW = np.asarray((0, 2, 3, 1), dtype=int)
Q = np.asarray(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0, 0.0]],
    dtype=float,
)


@dataclass(frozen=True)
class PCMFixture:
    name: str
    positions: np.ndarray
    atomic_numbers: tuple[int, ...]
    radii: tuple[float, ...]
    source: np.ndarray
    transition_width: float = 0.08
    surface_lmax: int = 3
    partition_lmax: int = 6
    partition_order: int = 96
    source_order: int = 128
    double_layer_order: int = 128
    dielectric: float = 80.0

    def moved(self, positions: np.ndarray) -> "PCMFixture":
        return replace(self, positions=np.asarray(positions, dtype=float))


S = PCMFixture(
    "S",
    np.asarray([[0.0, 0.0, 0.0]]),
    (1,),
    (2.0,),
    np.asarray([[0.4, 0.2, -0.1, 0.3]]),
    surface_lmax=1,
    partition_lmax=2,
    dielectric=7.0,
)
P = PCMFixture(
    "P",
    np.asarray([[0.0, 0.0, 0.0], [2.83, -0.41, 0.29]]),
    (6, 8),
    (1.23, 1.09),
    np.asarray([[0.18, 0.030, -0.020, 0.010],
                [-0.18, -0.015, 0.025, -0.020]]),
)
W = PCMFixture(
    "W",
    np.asarray([[0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2399872, 0.927297, 0.0]]),
    (8, 1, 1),
    (2.294, 1.2, 1.2),
    np.asarray([[-0.70, 0.04, -0.02, 0.03],
                [0.35, 0.00, 0.01, -0.02],
                [0.35, -0.01, 0.00, 0.02]]),
)


def tangency_fixture(separation: float) -> PCMFixture:
    if separation not in (0.2, 1.8):
        raise ValueError("the frozen tangency separations are 0.2 and 1.8 Angstrom")
    return PCMFixture(
        f"T-{separation}",
        np.asarray([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]),
        (1, 1),
        (1.0, 0.8),
        np.asarray([[0.4, 0.05, -0.02, 0.03],
                    [-0.4, -0.01, 0.04, -0.02]]),
        surface_lmax=2,
        partition_lmax=4,
        dielectric=20.0,
    )


def block_q(atom_count: int) -> np.ndarray:
    return np.kron(np.eye(atom_count), Q)


def raw_source_to_cartesian(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    return source[:, RAW_TO_CARTESIAN].copy()


def rotate_raw_source(source: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(source, dtype=float, copy=True)
    cart = raw_source_to_cartesian(source)
    result[:, 1:4] = (cart[:, 1:4] @ np.asarray(rotation).T)[:, (1, 2, 0)]
    return result


def rotate_cartesian_field(field: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(field, dtype=float, copy=True)
    result[:, 1:4] = result[:, 1:4] @ np.asarray(rotation).T
    return result


def point_multipole_potential_ev(
    points_angstrom: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    source_raw: np.ndarray,
) -> np.ndarray:
    """Direct Coulomb potential in eV/e, with raw ``[q,py,pz,px]`` input."""

    points = np.asarray(points_angstrom, dtype=float)
    centers = np.asarray(atom_positions_angstrom, dtype=float)
    cart = raw_source_to_cartesian(source_raw)
    result = np.zeros(len(points), dtype=float)
    for center, coefficients in zip(centers, cart, strict=True):
        displacement = points - center
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance <= 1.0e-14):
            raise ValueError("point source is singular at its centre")
        result += coefficients[0] / distance
        result += np.einsum("si,i->s", displacement, coefficients[1:]) / distance**3
    return COULOMB_EV_ANGSTROM_PER_E2 * result


def point_multipole_field_ev(
    points_angstrom: np.ndarray,
    atom_positions_angstrom: np.ndarray,
    source_raw: np.ndarray,
) -> np.ndarray:
    """Return direct ``[V,dV/dx,dV/dy,dV/dz]`` in eV/e and eV/(e Angstrom)."""

    points = np.asarray(points_angstrom, dtype=float)
    centers = np.asarray(atom_positions_angstrom, dtype=float)
    cart = raw_source_to_cartesian(source_raw)
    potential = np.zeros(len(points), dtype=float)
    gradient = np.zeros((len(points), 3), dtype=float)
    for center, coefficients in zip(centers, cart, strict=True):
        displacement = points - center
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance <= 1.0e-14):
            raise ValueError("point source is singular at its centre")
        projection = displacement @ coefficients[1:]
        potential += coefficients[0] / distance + projection / distance**3
        gradient += (
            -coefficients[0] * displacement / distance[:, None] ** 3
            + coefficients[1:][None, :] / distance[:, None] ** 3
            - 3.0 * projection[:, None] * displacement / distance[:, None] ** 5
        )
    return COULOMB_EV_ANGSTROM_PER_E2 * np.column_stack((potential, gradient))


def real_harmonic_design(directions: np.ndarray, lmax: int) -> np.ndarray:
    """Orthonormal real Condon--Shortley harmonics in ``l,m`` order."""

    directions = np.asarray(directions, dtype=float)
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    columns = []
    for ell in range(lmax + 1):
        for signed_order in range(-ell, ell + 1):
            order = abs(signed_order)
            normalization = math.sqrt(
                (2 * ell + 1)
                * math.exp(math.lgamma(ell - order + 1) - math.lgamma(ell + order + 1))
                / (4 * math.pi)
            )
            complex_value = normalization * lpmv(order, ell, np.cos(theta)) * np.exp(1j * order * phi)
            if signed_order < 0:
                column = math.sqrt(2.0) * ((-1) ** order) * complex_value.imag
            elif signed_order == 0:
                column = complex_value.real
            else:
                column = math.sqrt(2.0) * ((-1) ** order) * complex_value.real
            columns.append(column)
    return np.column_stack(columns)


def direct_sphere_projection(
    *, target_center: np.ndarray, target_radius: float, source_center: np.ndarray,
    source_raw: np.ndarray, lmax: int, polar_order: int = 192, azimuthal_order: int = 384,
) -> np.ndarray:
    """Project a direct point MEP onto real harmonics by product quadrature."""

    cosine, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    phi = 2 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    sin_theta = np.sqrt(1 - cosine**2)
    directions = np.column_stack(
        (
            (sin_theta[:, None] * np.cos(phi)[None, :]).ravel(),
            (sin_theta[:, None] * np.sin(phi)[None, :]).ravel(),
            np.broadcast_to(cosine[:, None], (polar_order, azimuthal_order)).ravel(),
        )
    )
    weights = np.repeat(polar_weights, azimuthal_order) * (2 * np.pi / azimuthal_order)
    points = np.asarray(target_center) + target_radius * directions
    potential = point_multipole_potential_ev(points, np.asarray(source_center)[None, :], np.asarray(source_raw)[None, :])
    return real_harmonic_design(directions, lmax).T @ (weights * potential)


def isolated_sphere_energy(fixture: PCMFixture = S) -> float:
    q = float(fixture.source[0, 0])
    p2 = float(fixture.source[0, 1:] @ fixture.source[0, 1:])
    radius = fixture.radii[0]
    eps = fixture.dielectric
    return (
        -0.5 * COULOMB_EV_ANGSTROM_PER_E2 * (eps - 1.0) / eps * q * q / radius
        - COULOMB_EV_ANGSTROM_PER_E2 * (eps - 1.0) / (2.0 * eps + 1.0)
        * p2 / radius**3
    )


def solve_primal(matrices: Mapping[str, np.ndarray], source: np.ndarray) -> dict[str, np.ndarray | float]:
    """Evaluate the frozen PCM equations from detached matrices."""

    c = np.asarray(source, dtype=float).reshape(-1)
    B = np.asarray(matrices["B"], dtype=float)
    A_eps = np.asarray(matrices["A_eps"], dtype=float)
    A_inf = np.asarray(matrices["A_inf"], dtype=float)
    L = np.asarray(matrices["L"], dtype=float)
    C_raw = np.asarray(matrices["C_raw"], dtype=float)
    F = -(B @ c)
    G = np.linalg.solve(A_eps, A_inf @ F)
    X = np.linalg.solve(L, G)
    energy = 0.5 * float(c @ C_raw @ X)
    return {"F": F, "G": G, "X": X, "energy": energy}


def solve_transpose_kkt(
    matrices: Mapping[str, np.ndarray], source: np.ndarray
) -> dict[str, np.ndarray]:
    c = np.asarray(source, dtype=float).reshape(-1)
    L = np.asarray(matrices["L"], dtype=float)
    A_eps = np.asarray(matrices["A_eps"], dtype=float)
    A_inf = np.asarray(matrices["A_inf"], dtype=float)
    C_raw = np.asarray(matrices["C_raw"], dtype=float)
    lambda_x = np.linalg.solve(L.T, -0.5 * C_raw.T @ c)
    lambda_g = np.linalg.solve(A_eps.T, lambda_x)
    lambda_f = A_inf.T @ lambda_g
    return {"lambda_F": lambda_f, "lambda_G": lambda_g, "lambda_X": lambda_x}


def kkt_source_gradient(
    matrices: Mapping[str, np.ndarray], source: np.ndarray
) -> np.ndarray:
    primal = solve_primal(matrices, source)
    adjoint = solve_transpose_kkt(matrices, source)
    c = np.asarray(source, dtype=float).reshape(-1)
    C_raw = np.asarray(matrices["C_raw"], dtype=float)
    B = np.asarray(matrices["B"], dtype=float)
    return 0.5 * C_raw @ np.asarray(primal["X"]) + B.T @ adjoint["lambda_F"]


def lagrangian(
    matrices: Mapping[str, np.ndarray], source: np.ndarray,
    primal: Mapping[str, np.ndarray | float], adjoint: Mapping[str, np.ndarray],
) -> float:
    c = np.asarray(source, dtype=float).reshape(-1)
    F, G, X = (np.asarray(primal[name]) for name in ("F", "G", "X"))
    lf, lg, lx = (np.asarray(adjoint[name]) for name in ("lambda_F", "lambda_G", "lambda_X"))
    return float(
        0.5 * c @ np.asarray(matrices["C_raw"]) @ X
        + lf @ (F + np.asarray(matrices["B"]) @ c)
        + lg @ (np.asarray(matrices["A_eps"]) @ G - np.asarray(matrices["A_inf"]) @ F)
        + lx @ (np.asarray(matrices["L"]) @ X - G)
    )


def residual_metrics(operator: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    operator = np.asarray(operator, dtype=float)
    solution = np.asarray(solution, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    residual = operator @ solution - rhs
    absolute = float(np.linalg.norm(residual))
    denominator = float(np.linalg.norm(operator, 2) * np.linalg.norm(solution) + np.linalg.norm(rhs))
    eta = absolute / denominator if denominator else absolute
    singular = np.linalg.svd(operator, compute_uv=False)
    kappa = float(singular[0] / singular[-1])
    return {
        "absolute_residual": absolute,
        "rhs_relative_residual": absolute / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny),
        "backward_error": eta,
        "kappa": kappa,
        "amplification_indicator": kappa * eta,
        "sigma_min": float(singular[-1]),
        "relative_sigma_min": float(singular[-1] / singular[0]),
    }


def central_difference(function: Callable[[np.ndarray], float], value: np.ndarray, direction: np.ndarray, step: float) -> float:
    value = np.asarray(value, dtype=float)
    direction = np.asarray(direction, dtype=float)
    return float((function(value + step * direction) - function(value - step * direction)) / (2.0 * step))


def central_jacobian(function: Callable[[np.ndarray], np.ndarray], value: np.ndarray, step: float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    baseline = np.asarray(function(value), dtype=float)
    jacobian = np.empty((baseline.size, value.size), dtype=float)
    for index in range(value.size):
        delta = np.zeros(value.size, dtype=float)
        delta[index] = step
        plus = np.asarray(function((value.reshape(-1) + delta).reshape(value.shape)), dtype=float)
        minus = np.asarray(function((value.reshape(-1) - delta).reshape(value.shape)), dtype=float)
        jacobian[:, index] = ((plus - minus) / (2.0 * step)).reshape(-1)
    return jacobian


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.asarray([[0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0]])
    return math.cos(angle) * np.eye(3) + (1.0 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * cross


ROTATIONS = (
    axis_angle_rotation(np.asarray([1.0, 2.0, -1.0]), 0.37),
    axis_angle_rotation(np.asarray([-0.4, 0.3, 0.8]), 1.11),
    axis_angle_rotation(np.asarray([0.2, -0.7, 0.4]), 0.83),
)


def implementation_point_potential_control(
    points_bohr: np.ndarray, positions_angstrom: np.ndarray, source_raw: np.ndarray
) -> np.ndarray:
    """Current MAPLE control, converted only at the test boundary to eV/e."""

    from maple.function.calculator.extra_correction.implicit.gto_density import (
        point_multipole_potential,
    )

    return Hartree * point_multipole_potential(points_bohr, positions_angstrom, source_raw)
