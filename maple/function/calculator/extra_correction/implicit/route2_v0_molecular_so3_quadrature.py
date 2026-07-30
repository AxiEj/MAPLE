"""Deterministic Haar quadratures for Route-2 V0 rigid molecular orientations.

The V0 molecular liquid is written in the unnormalised Haar measure of
``SO(3)`` whose total mass is ``8*pi**2``.  A production liquid calculation
therefore cannot replace the orientation integral by one preferred solvent
pose or an error-selected random sample.

This module uses the Euler product rule described for rigid-molecular classical
DFT by Sundararaman and Arias, *Comput. Phys. Commun.* **185**, 818 (2014),
doi:10.1016/j.cpc.2013.11.013.  With

``R = R_z(alpha) R_y(arccos(x)) R_z(gamma)``,

Gauss--Legendre nodes in ``x in [-1, 1]`` and ``2*n`` equally spaced nodes in
each periodic angle define a deterministic full-``SO(3)`` rule of
``4*n**3`` orientations.  Its weights sum exactly (up to floating point) to
``8*pi**2`` and the rule has the documented Wigner-rank bandlimit
``2*n - 1``.  No point-group quotient is applied automatically: such a
reduction needs a separately source-bound molecular symmetry proof.

The Cartesian product helper creates the exact finite configuration measure
``d**3 X dOmega`` required by the existing molecular ideal/HNC scalar.  It is
a reference discretisation and a refinement input, not a claim that any
particular order or Cartesian grid is converged, physical, or accurate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from .route2_v0_molecular_external_potential import Route2V0MolecularConfigurations
from .route2_v0_molecular_ideal_gas import (
    RIGID_MOLECULAR_ORIENTATION_MEASURE,
    Route2V0MolecularConfigurationQuadrature,
)
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_MOLECULAR_SO3_EULER_QUADRATURE_CONSTRUCTION = (
    "route2-v0-molecular-so3-euler-product-quadrature-v1"
)
V0_MOLECULAR_CARTESIAN_EULER_QUADRATURE_CONSTRUCTION = (
    "route2-v0-molecular-cartesian-euler-product-quadrature-v1"
)


def _positive_integer(value: object, *, name: str) -> int:
    """Return one finite strictly positive integer without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer.")
        result = int(numeric)
    else:
        raise TypeError(f"{name} must be a positive integer.")
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _immutable_array(
    values: np.ndarray, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Validate and freeze one finite real array of the exact declared shape."""

    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _euler_product_rotations_and_weights(
    polar_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical ``R_z(alpha) R_y(beta) R_z(gamma)`` product rule."""

    nodes, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    periodic_order = 2 * polar_order
    angles = 2.0 * math.pi * np.arange(periodic_order, dtype=float) / periodic_order
    cos_beta, alpha, gamma = np.meshgrid(nodes, angles, angles, indexing="ij")
    sin_beta = np.sqrt(np.maximum(0.0, 1.0 - cos_beta * cos_beta))
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    rotation_shape = (4 * polar_order**3, 3, 3)
    rotations = np.empty(rotation_shape, dtype=float)
    rotations[:, 0, 0] = (
        cos_alpha * cos_beta * cos_gamma - sin_alpha * sin_gamma
    ).ravel()
    rotations[:, 0, 1] = (
        -cos_alpha * cos_beta * sin_gamma - sin_alpha * cos_gamma
    ).ravel()
    rotations[:, 0, 2] = (cos_alpha * sin_beta).ravel()
    rotations[:, 1, 0] = (
        sin_alpha * cos_beta * cos_gamma + cos_alpha * sin_gamma
    ).ravel()
    rotations[:, 1, 1] = (
        -sin_alpha * cos_beta * sin_gamma + cos_alpha * cos_gamma
    ).ravel()
    rotations[:, 1, 2] = (sin_alpha * sin_beta).ravel()
    rotations[:, 2, 0] = (-sin_beta * cos_gamma).ravel()
    rotations[:, 2, 1] = (sin_beta * sin_gamma).ravel()
    rotations[:, 2, 2] = cos_beta.ravel()

    angular_weight = (2.0 * math.pi / periodic_order) ** 2
    weights = np.broadcast_to(
        polar_weights[:, None, None] * angular_weight,
        cos_beta.shape,
    ).reshape((-1,))
    return rotations, weights


@dataclass(frozen=True)
class Route2V0EulerSO3Quadrature:
    """One canonical deterministic full-``SO(3)`` Euler product quadrature.

    ``polar_order = n`` uses ``n`` Gauss--Legendre nodes in ``cos(beta)`` and
    ``2*n`` uniform nodes in each periodic Euler angle.  The integral is in
    the unnormalised ``8*pi**2`` Haar convention used by the V0 scalar.
    """

    polar_order: int
    construction: str = V0_MOLECULAR_SO3_EULER_QUADRATURE_CONSTRUCTION
    rotations: np.ndarray = field(init=False, repr=False, compare=False)
    orientation_weights: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.construction != V0_MOLECULAR_SO3_EULER_QUADRATURE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 SO(3) quadrature construction.")
        polar_order = _positive_integer(
            self.polar_order,
            name="SO(3) Euler polar order",
        )
        rotations, weights = _euler_product_rotations_and_weights(polar_order)
        orientation_count = 4 * polar_order**3
        immutable_rotations = _immutable_array(
            rotations,
            name="SO(3) Euler rotations",
            shape=(orientation_count, 3, 3),
        )
        immutable_weights = _immutable_array(
            weights,
            name="SO(3) Euler orientation weights",
            shape=(orientation_count,),
        )
        if np.any(immutable_weights <= 0.0):
            raise RuntimeError("SO(3) Euler orientation weights must be positive.")
        if not math.isclose(
            float(np.sum(immutable_weights)),
            RIGID_MOLECULAR_ORIENTATION_MEASURE,
            rel_tol=0.0,
            abs_tol=1.0e-12 * RIGID_MOLECULAR_ORIENTATION_MEASURE,
        ):
            raise RuntimeError("SO(3) Euler orientation weights violate Haar measure.")
        identities = np.einsum("nij,nkj->nik", immutable_rotations, immutable_rotations)
        determinants = np.linalg.det(immutable_rotations)
        if not np.allclose(
            identities, np.eye(3), rtol=0.0, atol=1.0e-12
        ) or not np.allclose(
            determinants,
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("SO(3) Euler construction produced improper rotations.")
        object.__setattr__(self, "polar_order", polar_order)
        object.__setattr__(self, "rotations", immutable_rotations)
        object.__setattr__(self, "orientation_weights", immutable_weights)

    @property
    def orientation_count(self) -> int:
        """Return the deterministic number of full-``SO(3)`` orientations."""

        return int(self.orientation_weights.size)

    @property
    def periodic_order(self) -> int:
        """Return the uniform node count in each periodic Euler angle."""

        return 2 * self.polar_order

    @property
    def wigner_rank_bandlimit(self) -> int:
        """Return the documented product-rule Wigner rank ``2*n - 1``."""

        return 2 * self.polar_order - 1

    @property
    def orientation_measure(self) -> float:
        """Return the unnormalised Haar mass represented by the weights."""

        return float(np.sum(self.orientation_weights))


@dataclass(frozen=True)
class Route2V0CartesianEulerProductQuadrature:
    """Cartesian-grid times full-``SO(3)`` product measure for one V0 liquid.

    The representation deliberately materializes every translation/orientation
    pair because the current MACE external-potential contract consumes one
    explicit rigid configuration per energy evaluation.  It is appropriate for
    controlled refinement executions; a future production-scale path may only
    replace this storage with a matrix-free representation that preserves the
    exact same measure and adjoint identities.
    """

    grid: RegularCartesianGrid
    orientation_quadrature: Route2V0EulerSO3Quadrature
    construction: str = V0_MOLECULAR_CARTESIAN_EULER_QUADRATURE_CONSTRUCTION
    configurations: Route2V0MolecularConfigurations = field(
        init=False,
        repr=False,
        compare=False,
    )
    quadrature: Route2V0MolecularConfigurationQuadrature = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError(
                "Cartesian Euler quadrature requires a regular Cartesian grid."
            )
        if not isinstance(self.orientation_quadrature, Route2V0EulerSO3Quadrature):
            raise TypeError("Cartesian Euler quadrature requires an SO(3) quadrature.")
        if self.construction != V0_MOLECULAR_CARTESIAN_EULER_QUADRATURE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 Cartesian Euler construction.")
        translations = np.repeat(
            self.grid.points_bohr(),
            self.orientation_quadrature.orientation_count,
            axis=0,
        )
        rotations = np.tile(
            self.orientation_quadrature.rotations,
            (self.grid.point_count, 1, 1),
        )
        configurations = Route2V0MolecularConfigurations(
            translations_bohr=translations,
            rotations=rotations,
        )
        phase_space_weights = np.tile(
            self.grid.volume_element_bohr3
            * self.orientation_quadrature.orientation_weights,
            self.grid.point_count,
        )
        quadrature = Route2V0MolecularConfigurationQuadrature(
            configurations=configurations,
            phase_space_weights_bohr3=phase_space_weights,
        )
        expected_measure = (
            self.grid.point_count
            * self.grid.volume_element_bohr3
            * RIGID_MOLECULAR_ORIENTATION_MEASURE
        )
        if not math.isclose(
            quadrature.total_phase_space_measure_bohr3,
            expected_measure,
            rel_tol=0.0,
            abs_tol=1.0e-12 * max(1.0, expected_measure),
        ):
            raise RuntimeError(
                "Cartesian Euler product quadrature violates its declared phase-space "
                "measure."
            )
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "quadrature", quadrature)


def build_route2_v0_cartesian_euler_product_quadrature(
    *,
    grid: RegularCartesianGrid,
    polar_order: int,
) -> Route2V0CartesianEulerProductQuadrature:
    """Build one explicit Cartesian times deterministic full-``SO(3)`` rule."""

    return Route2V0CartesianEulerProductQuadrature(
        grid=grid,
        orientation_quadrature=Route2V0EulerSO3Quadrature(polar_order=polar_order),
    )


__all__ = [
    "V0_MOLECULAR_CARTESIAN_EULER_QUADRATURE_CONSTRUCTION",
    "V0_MOLECULAR_SO3_EULER_QUADRATURE_CONSTRUCTION",
    "Route2V0CartesianEulerProductQuadrature",
    "Route2V0EulerSO3Quadrature",
    "build_route2_v0_cartesian_euler_product_quadrature",
]
