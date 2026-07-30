"""Finite-dimensional Hessian stability diagnostics for Route-2 V0 liquids.

The molecular HNC and weighted-density bridge functionals already expose an
exact Hessian-vector action in molecular configuration-density variables. This
module turns that action into a small, controlled-grid certificate,

A = W^(1/2) H W^(-1/2),

where W = diag(w_i) is the positive molecular quadrature measure. Because H
maps a density direction to a dimensionless gradient, A and its spectrum carry
Bohr^3 units. The second variation is kBT * x.T @ A @ x for x = W^(1/2) d.
Thus a reciprocal action and a strictly positive minimum eigenvalue of A are
the local-minimum criterion on that exact finite discretisation.

The current V0 liquid scalar is grand-canonical, so this diagnostic uses the
full configuration-density space. A canonical fixed-number constraint requires
a separately declared constrained certificate; this module never silently
projects a mode away.

This is intentionally a dense diagnostic with a hard dimension cap. It does
not substitute a production matrix-free eigensolver, grid/orientation
refinement, stationarity verification, a physical solvent asset, or any
chemistry/accuracy claim. It refuses material nonreciprocity; it never
symmetrises a Hessian action, clips an eigenvalue, or adjusts a liquid
parameter to obtain a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Protocol

import numpy as np

V0_MOLECULAR_HESSIAN_STABILITY_CONSTRUCTION = (
    "route2-v0-molecular-hessian-stability-certificate-v1"
)

_MACHINE_EPSILON = float(np.finfo(float).eps)
_NUMERICAL_FACTOR = 128.0


class _MolecularQuadrature(Protocol):
    """The positive configuration-space measure of one molecular functional."""

    @property
    def phase_space_weights_bohr3(self) -> np.ndarray:
        """Return the declared positive configuration-space weights."""

        ...


class _MolecularProjection(Protocol):
    """The projection handle required to recover that exact measure."""

    @property
    def quadrature(self) -> _MolecularQuadrature:
        """Return the configuration-space quadrature."""

        ...


class _MolecularHessianFunctional(Protocol):
    """Minimal molecular-liquid Hessian interface used by this diagnostic."""

    @property
    def projection(self) -> _MolecularProjection:
        """Return the projection defining the Hessian pairing."""

        ...

    def dimensionless_hessian_matvec(
        self,
        configuration_density_bohr3: np.ndarray,
        direction_bohr3: np.ndarray,
    ) -> np.ndarray:
        """Return the dimensionless Hessian action in the quadrature pairing."""

        ...


def _positive_integer(value: object, *, name: str) -> int:
    """Return one finite strictly positive integer without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    if isinstance(value, Integral):
        integer = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer.")
        integer = int(numeric)
    else:
        raise TypeError(f"{name} must be a positive integer.")
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _immutable_vector(
    values: np.ndarray,
    *,
    name: str,
    dimension: int,
    positive: bool = False,
) -> np.ndarray:
    """Validate and freeze one finite real vector."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (dimension,)
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        requirement = f"finite with shape ({dimension},)"
        if positive:
            requirement += " and strictly positive"
        raise ValueError(f"{name} must be {requirement}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _quadrature_weights(functional: _MolecularHessianFunctional) -> np.ndarray:
    """Extract the one declared positive molecular quadrature measure."""

    try:
        raw_weights = functional.projection.quadrature.phase_space_weights_bohr3
    except AttributeError as exc:
        raise TypeError(
            "Molecular Hessian stability requires "
            + "projection.quadrature.phase_space_weights_bohr3."
        ) from exc
    if np.iscomplexobj(raw_weights):
        raise ValueError("Molecular quadrature weights must be real-valued.")
    weights = np.asarray(raw_weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(
            "Molecular quadrature weights must be a nonempty one-dimensional array."
        )
    return _immutable_vector(
        weights,
        name="Molecular quadrature weights",
        dimension=int(weights.size),
        positive=True,
    )


def _hessian_action(
    functional: _MolecularHessianFunctional,
    density: np.ndarray,
    direction: np.ndarray,
    *,
    dimension: int,
) -> np.ndarray:
    """Evaluate and validate one declared Hessian-vector action."""

    action = functional.dimensionless_hessian_matvec(density, direction)
    return _immutable_vector(
        action,
        name="Molecular Hessian-vector action",
        dimension=dimension,
    )


@dataclass(frozen=True)
class Route2V0MolecularHessianStabilityCertificate:
    """A fail-closed stability result on one declared finite discretisation.

    weighted_hessian_matrix is the raw weighted representation
    W^(1/2) H W^(-1/2), assembled from the supplied Hessian action. It is
    retained exactly as observed: the diagnostic rejects material
    nonreciprocity rather than symmetrising it before diagonalisation.

    A positive-definite classification is only a local-minimum result on this
    one finite configuration grid. Callers still need a stationary residual,
    source-bound physical liquid asset, and grid/orientation refinement before
    promoting it to a physical-liquid admission gate.
    """

    configuration_density_bohr3: np.ndarray
    quadrature_weights_bohr3: np.ndarray
    weighted_hessian_matrix: np.ndarray
    eigenvalues_bohr3: np.ndarray
    dimension: int
    maximum_dimension: int
    operator_two_norm_bohr3: float
    reciprocity_absolute_frobenius_residual_bohr3: float
    reciprocity_relative_frobenius_residual: float
    reciprocity_relative_tolerance: float
    numerical_eigenvalue_tolerance_bohr3: float
    minimum_eigenvalue_bohr3: float
    maximum_eigenvalue_bohr3: float
    classification: Literal[
        "positive-definite",
        "negative-mode",
        "numerically-singular",
    ]
    construction: str = V0_MOLECULAR_HESSIAN_STABILITY_CONSTRUCTION

    def __post_init__(self) -> None:
        """Freeze evidence arrays and reject internally inconsistent results."""

        if self.construction != V0_MOLECULAR_HESSIAN_STABILITY_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 molecular Hessian construction.")
        dimension = _positive_integer(
            self.dimension,
            name="Molecular Hessian dimension",
        )
        maximum_dimension = _positive_integer(
            self.maximum_dimension,
            name="Molecular Hessian maximum dimension",
        )
        if dimension > maximum_dimension:
            raise ValueError(
                "Molecular Hessian dimension must not exceed its declared dense "
                + "diagnostic limit."
            )
        density = _immutable_vector(
            self.configuration_density_bohr3,
            name="Molecular configuration density",
            dimension=dimension,
            positive=True,
        )
        weights = _immutable_vector(
            self.quadrature_weights_bohr3,
            name="Molecular quadrature weights",
            dimension=dimension,
            positive=True,
        )
        if np.iscomplexobj(self.weighted_hessian_matrix):
            raise ValueError("Weighted molecular Hessian matrix must be real-valued.")
        matrix = np.asarray(self.weighted_hessian_matrix, dtype=float)
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Weighted molecular Hessian matrix must be finite with shape "
                + f"({dimension}, {dimension})."
            )
        immutable_matrix = np.array(matrix, dtype=float, copy=True)
        immutable_matrix.setflags(write=False)
        eigenvalues = _immutable_vector(
            self.eigenvalues_bohr3,
            name="Molecular Hessian eigenvalues",
            dimension=dimension,
        )
        values = {
            "operator_two_norm_bohr3": self.operator_two_norm_bohr3,
            "reciprocity_absolute_frobenius_residual_bohr3": (
                self.reciprocity_absolute_frobenius_residual_bohr3
            ),
            "reciprocity_relative_frobenius_residual": (
                self.reciprocity_relative_frobenius_residual
            ),
            "reciprocity_relative_tolerance": self.reciprocity_relative_tolerance,
            "numerical_eigenvalue_tolerance_bohr3": self.numerical_eigenvalue_tolerance_bohr3,
            "minimum_eigenvalue_bohr3": self.minimum_eigenvalue_bohr3,
            "maximum_eigenvalue_bohr3": self.maximum_eigenvalue_bohr3,
        }
        for name, raw_value in values.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)

        antisymmetric = matrix - matrix.T
        antisymmetric_norm = float(np.linalg.norm(antisymmetric, ord="fro"))
        matrix_frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
        reciprocity_residual = antisymmetric_norm / max(1.0, matrix_frobenius_norm)
        operator_two_norm_bohr3 = float(np.linalg.norm(matrix, ord=2))
        reciprocity_tolerance = _NUMERICAL_FACTOR * _MACHINE_EPSILON * dimension
        eigenvalue_tolerance = (
            _NUMERICAL_FACTOR
            * _MACHINE_EPSILON
            * dimension
            * max(1.0, operator_two_norm_bohr3)
        )
        comparison_tolerance = 64.0 * _MACHINE_EPSILON

        def _matches(observed: float, expected: float) -> bool:
            return math.isclose(
                observed,
                expected,
                rel_tol=comparison_tolerance,
                abs_tol=comparison_tolerance * max(1.0, abs(expected)),
            )

        if not _matches(self.operator_two_norm_bohr3, operator_two_norm_bohr3):
            raise ValueError(
                "operator_two_norm_bohr3 must match the raw weighted Hessian matrix."
            )
        if not _matches(
            self.reciprocity_absolute_frobenius_residual_bohr3,
            antisymmetric_norm,
        ):
            raise ValueError(
                "reciprocity_absolute_frobenius_residual_bohr3 must match the raw "
                + "weighted Hessian matrix."
            )
        if not _matches(
            self.reciprocity_relative_frobenius_residual,
            reciprocity_residual,
        ):
            raise ValueError(
                "reciprocity_relative_frobenius_residual must match the raw "
                + "weighted Hessian matrix."
            )
        if not _matches(
            self.reciprocity_relative_tolerance,
            reciprocity_tolerance,
        ):
            raise ValueError(
                "reciprocity_relative_tolerance must use the declared "
                + "machine-precision rule."
            )
        if not _matches(
            self.numerical_eigenvalue_tolerance_bohr3,
            eigenvalue_tolerance,
        ):
            raise ValueError(
                "numerical_eigenvalue_tolerance_bohr3 must use the declared "
                + "machine-precision rule."
            )
        if reciprocity_residual > reciprocity_tolerance:
            raise ValueError(
                "Weighted molecular Hessian must be reciprocal before a "
                + "stability certificate can be constructed."
            )

        spectrum = np.linalg.eigvalsh(matrix)
        if not np.allclose(
            eigenvalues,
            spectrum,
            rtol=comparison_tolerance,
            atol=comparison_tolerance * max(1.0, operator_two_norm_bohr3),
        ):
            raise ValueError(
                "Molecular Hessian spectrum must match the raw weighted matrix."
            )
        minimum = float(np.min(spectrum))
        maximum = float(np.max(spectrum))
        if not _matches(self.minimum_eigenvalue_bohr3, minimum):
            raise ValueError(
                "Minimum molecular Hessian eigenvalue must match the spectrum."
            )
        if not _matches(self.maximum_eigenvalue_bohr3, maximum):
            raise ValueError(
                "Maximum molecular Hessian eigenvalue must match the spectrum."
            )
        expected_classification: Literal[
            "positive-definite",
            "negative-mode",
            "numerically-singular",
        ]
        if minimum > self.numerical_eigenvalue_tolerance_bohr3:
            expected_classification = "positive-definite"
        elif minimum < -self.numerical_eigenvalue_tolerance_bohr3:
            expected_classification = "negative-mode"
        else:
            expected_classification = "numerically-singular"
        if self.classification != expected_classification:
            raise ValueError(
                "Molecular Hessian classification must match the declared "
                + "numerical eigenvalue tolerance."
            )
        object.__setattr__(self, "configuration_density_bohr3", density)
        object.__setattr__(self, "quadrature_weights_bohr3", weights)
        object.__setattr__(self, "weighted_hessian_matrix", immutable_matrix)
        object.__setattr__(self, "eigenvalues_bohr3", eigenvalues)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "maximum_dimension", maximum_dimension)

    @property
    def is_positive_definite(self) -> bool:
        """Return the finite-grid local-minimum result without an asset claim."""

        return self.classification == "positive-definite"


def certify_route2_v0_molecular_hessian_stability(
    functional: _MolecularHessianFunctional,
    configuration_density_bohr3: np.ndarray,
    *,
    maximum_dimension: int = 256,
) -> Route2V0MolecularHessianStabilityCertificate:
    """Certify reciprocity and local curvature on a small fixed grid.

    The input configuration density must already be a declared stationary
    candidate when a caller wants to use this result as a local stability gate.
    This function does not solve the liquid equation, inspect a Picard mixing
    factor, or select any model parameter.
    """

    if not hasattr(functional, "dimensionless_hessian_matvec"):
        raise TypeError(
            "Molecular Hessian stability requires a dimensionless_hessian_matvec "
            + "method."
        )
    weights = _quadrature_weights(functional)
    dimension = int(weights.size)
    dense_limit = _positive_integer(
        maximum_dimension,
        name="Molecular Hessian maximum dimension",
    )
    if dimension > dense_limit:
        raise ValueError(
            "Molecular Hessian dimension exceeds the declared dense diagnostic "
            + "limit; use a separately validated matrix-free production "
            + "certificate."
        )
    density = _immutable_vector(
        configuration_density_bohr3,
        name="Molecular configuration density",
        dimension=dimension,
        positive=True,
    )
    sqrt_weights = np.sqrt(weights)
    matrix = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        direction = np.zeros(dimension, dtype=float)
        direction[column] = 1.0 / sqrt_weights[column]
        action = _hessian_action(
            functional,
            density,
            direction,
            dimension=dimension,
        )
        matrix[:, column] = sqrt_weights * action

    antisymmetric = matrix - matrix.T
    antisymmetric_norm = float(np.linalg.norm(antisymmetric, ord="fro"))
    matrix_frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    reciprocity_residual = antisymmetric_norm / max(1.0, matrix_frobenius_norm)
    reciprocity_tolerance = _NUMERICAL_FACTOR * _MACHINE_EPSILON * dimension
    if reciprocity_residual > reciprocity_tolerance:
        raise RuntimeError(
            "Molecular Hessian action is not reciprocal in the declared "
            + "quadrature pairing; refusing to symmetrise a material "
            + "nonreciprocity."
        )

    operator_two_norm_bohr3 = float(np.linalg.norm(matrix, ord=2))
    eigenvalue_tolerance = (
        _NUMERICAL_FACTOR
        * _MACHINE_EPSILON
        * dimension
        * max(1.0, operator_two_norm_bohr3)
    )
    # This is intentionally the raw accepted matrix, not a symmetrised repair.
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(np.min(eigenvalues))
    maximum = float(np.max(eigenvalues))
    classification: Literal[
        "positive-definite",
        "negative-mode",
        "numerically-singular",
    ]
    if minimum > eigenvalue_tolerance:
        classification = "positive-definite"
    elif minimum < -eigenvalue_tolerance:
        classification = "negative-mode"
    else:
        classification = "numerically-singular"
    return Route2V0MolecularHessianStabilityCertificate(
        configuration_density_bohr3=density,
        quadrature_weights_bohr3=weights,
        weighted_hessian_matrix=matrix,
        eigenvalues_bohr3=eigenvalues,
        dimension=dimension,
        maximum_dimension=dense_limit,
        operator_two_norm_bohr3=operator_two_norm_bohr3,
        reciprocity_absolute_frobenius_residual_bohr3=antisymmetric_norm,
        reciprocity_relative_frobenius_residual=reciprocity_residual,
        reciprocity_relative_tolerance=reciprocity_tolerance,
        numerical_eigenvalue_tolerance_bohr3=eigenvalue_tolerance,
        minimum_eigenvalue_bohr3=minimum,
        maximum_eigenvalue_bohr3=maximum,
        classification=classification,
    )


__all__ = [
    "V0_MOLECULAR_HESSIAN_STABILITY_CONSTRUCTION",
    "Route2V0MolecularHessianStabilityCertificate",
    "certify_route2_v0_molecular_hessian_stability",
]
