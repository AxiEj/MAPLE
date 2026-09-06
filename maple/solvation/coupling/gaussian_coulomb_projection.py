"""Target-independent charge/dipole projection in a Gaussian Coulomb metric.

The basis is the normalized atom-centred density

``rho_A(r) = q_A g_A(r) + p_A . (r-R_A) g_A(r) / sigma**2``

with ``g_A`` a three-dimensional Gaussian of Cartesian variance ``sigma**2``.
The projection changes an existing atomwise ``(q, p)`` source by the smallest
possible Coulomb self-energy while enforcing total charge and molecular dipole.
No solvation target, fitted covariance, element scale, or cavity parameter
enters this construction.

This module is a research primitive.  It defines the source coefficients and
their algebraic projection only; it does not claim a Route-2 energy, force, or
public capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf

_SQRT_PI = math.sqrt(math.pi)


def _finite_array(
    values: object,
    *,
    shape: tuple[int, ...] | None = None,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        expected = "" if shape is None else f" with shape {shape}"
        raise ValueError(f"{name} must be finite{expected}.")
    return result


def _gaussian_coulomb_kernel_jet(
    displacement_angstrom: object,
    *,
    sigma_angstrom: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``K``, ``grad_D K`` and ``hess_D K`` for equal-width Gaussians.

    ``K(D) = erf(|D| / (2 sigma)) / |D|``.  The small-distance branch is the
    analytic Taylor series, not a finite-difference or clipped denominator.
    """

    displacement = _finite_array(
        displacement_angstrom, shape=(3,), name="Gaussian displacement"
    )
    sigma = float(sigma_angstrom)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_angstrom must be finite and positive.")
    radius2 = float(np.dot(displacement, displacement))
    radius = math.sqrt(radius2)
    if radius <= 1.0e-4 * sigma:
        a0 = 1.0 / (_SQRT_PI * sigma)
        a2 = -1.0 / (12.0 * _SQRT_PI * sigma**3)
        a4 = 1.0 / (160.0 * _SQRT_PI * sigma**5)
        a6 = -1.0 / (2688.0 * _SQRT_PI * sigma**7)
        value = a0 + a2 * radius2 + a4 * radius2**2 + a6 * radius2**3
        radial_gradient = 2.0 * a2 + 4.0 * a4 * radius2 + 6.0 * a6 * radius2**2
        gradient = radial_gradient * displacement
        hessian = radial_gradient * np.eye(3) + (
            8.0 * a4 + 24.0 * a6 * radius2
        ) * np.outer(displacement, displacement)
        return float(value), gradient, hessian

    scaled = radius / (2.0 * sigma)
    exponential = math.exp(-(scaled**2))
    error_function = float(erf(scaled))
    value = error_function / radius
    first = exponential / (sigma * _SQRT_PI * radius) - error_function / radius**2
    second = (
        -exponential / (2.0 * _SQRT_PI * sigma**3)
        - 2.0 * exponential / (_SQRT_PI * sigma * radius**2)
        + 2.0 * error_function / radius**3
    )
    direction = displacement / radius
    gradient = first * direction
    hessian = (second - first / radius) * np.outer(direction, direction)
    hessian += (first / radius) * np.eye(3)
    return float(value), gradient, hessian


def gaussian_charge_dipole_coulomb_gram(
    positions_angstrom: object,
    *,
    sigma_angstrom: float,
) -> np.ndarray:
    """Return the Coulomb Gram matrix in per-atom ``[q, px, py, pz]`` order."""

    positions = _finite_array(positions_angstrom, name="positions_angstrom")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
        raise ValueError("positions_angstrom must have shape (N,3), N >= 1.")
    sigma = float(sigma_angstrom)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_angstrom must be finite and positive.")
    atom_count = len(positions)
    result = np.zeros((4 * atom_count, 4 * atom_count), dtype=np.float64)
    self_charge = 1.0 / (_SQRT_PI * sigma)
    self_dipole = 1.0 / (6.0 * _SQRT_PI * sigma**3)
    for atom in range(atom_count):
        block = slice(4 * atom, 4 * atom + 4)
        result[block, block] = np.diag(
            np.asarray((self_charge, self_dipole, self_dipole, self_dipole))
        )
        for other in range(atom + 1, atom_count):
            value, gradient, hessian = _gaussian_coulomb_kernel_jet(
                positions[atom] - positions[other], sigma_angstrom=sigma
            )
            pair = np.empty((4, 4), dtype=np.float64)
            pair[0, 0] = value
            pair[0, 1:] = -gradient
            pair[1:, 0] = gradient
            pair[1:, 1:] = -hessian
            other_block = slice(4 * other, 4 * other + 4)
            result[block, other_block] = pair
            result[other_block, block] = pair.T
    if not np.array_equal(result, result.T) or not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Gaussian Coulomb Gram assembly lost symmetry/finite values."
        )
    return result


def charge_dipole_constraint_matrix(positions_angstrom: object) -> np.ndarray:
    """Return the map from per-atom ``[q,px,py,pz]`` to ``[Q,mux,muy,muz]``."""

    positions = _finite_array(positions_angstrom, name="positions_angstrom")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
        raise ValueError("positions_angstrom must have shape (N,3), N >= 1.")
    result = np.zeros((4, 4 * len(positions)), dtype=np.float64)
    for atom, position in enumerate(positions):
        offset = 4 * atom
        result[0, offset] = 1.0
        result[1:, offset] = position
        result[1:, offset + 1 : offset + 4] = np.eye(3)
    return result


@dataclass(frozen=True, slots=True)
class GaussianCoulombProjection:
    """One immutable projected source plus numerical well-posedness evidence."""

    charges_e: np.ndarray
    dipoles_eangstrom: np.ndarray
    gram_minimum_eigenvalue: float
    gram_condition_number: float
    constraint_condition_number: float
    constraint_max_abs_error: float
    coulomb_correction_norm: float

    def __post_init__(self) -> None:
        charges = _finite_array(self.charges_e, name="charges_e")
        if charges.ndim != 1 or len(charges) < 1:
            raise ValueError("charges_e must have shape (N,), N >= 1.")
        dipoles = _finite_array(
            self.dipoles_eangstrom,
            shape=(len(charges), 3),
            name="dipoles_eangstrom",
        )
        for name in (
            "gram_minimum_eigenvalue",
            "gram_condition_number",
            "constraint_condition_number",
            "constraint_max_abs_error",
            "coulomb_correction_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        charges = np.frombuffer(charges.tobytes(), dtype=np.float64)
        dipoles = np.frombuffer(dipoles.tobytes(), dtype=np.float64).reshape(
            len(charges), 3
        )
        object.__setattr__(self, "charges_e", charges)
        object.__setattr__(self, "dipoles_eangstrom", dipoles)


def project_charge_dipoles_in_gaussian_coulomb_metric(
    *,
    positions_angstrom: object,
    charges_e: object,
    dipoles_eangstrom: object,
    target_total_charge_e: float,
    target_molecular_dipole_eangstrom: object,
    sigma_angstrom: float,
    maximum_gram_condition_number: float = 1.0e12,
    maximum_constraint_condition_number: float = 1.0e12,
) -> GaussianCoulombProjection:
    """Project a source to exact ``Q``/molecular-dipole closure.

    The correction solves

    ``min_delta 1/2 delta.T G delta  subject to C(c+delta)=target``.

    Cholesky and the four-dimensional Schur complement are used directly;
    there is no pseudoinverse, eigenvalue clipping, or hidden regularization.
    Ill-conditioned geometries fail closed.
    """

    positions = _finite_array(positions_angstrom, name="positions_angstrom")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
        raise ValueError("positions_angstrom must have shape (N,3), N >= 1.")
    charges = _finite_array(charges_e, shape=(len(positions),), name="charges_e")
    dipoles = _finite_array(
        dipoles_eangstrom,
        shape=(len(positions), 3),
        name="dipoles_eangstrom",
    )
    target_charge = float(target_total_charge_e)
    target_dipole = _finite_array(
        target_molecular_dipole_eangstrom,
        shape=(3,),
        name="target_molecular_dipole_eangstrom",
    )
    if not math.isfinite(target_charge):
        raise ValueError("target_total_charge_e must be finite.")
    for value, name in (
        (maximum_gram_condition_number, "maximum_gram_condition_number"),
        (
            maximum_constraint_condition_number,
            "maximum_constraint_condition_number",
        ),
    ):
        if not math.isfinite(float(value)) or float(value) < 1.0:
            raise ValueError(f"{name} must be finite and at least one.")

    gram = gaussian_charge_dipole_coulomb_gram(positions, sigma_angstrom=sigma_angstrom)
    eigenvalues = np.linalg.eigvalsh(gram)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    condition = math.inf if minimum <= 0.0 else maximum / minimum
    if minimum <= 0.0 or condition > float(maximum_gram_condition_number):
        raise np.linalg.LinAlgError(
            "Gaussian Coulomb metric is not numerically positive definite."
        )
    factor = cho_factor(gram, lower=True, check_finite=False)
    constraints = charge_dipole_constraint_matrix(positions)
    inverse_constraint_dual = cho_solve(factor, constraints.T, check_finite=False)
    schur = constraints @ inverse_constraint_dual
    schur_eigenvalues = np.linalg.eigvalsh(schur)
    schur_minimum = float(schur_eigenvalues[0])
    schur_maximum = float(schur_eigenvalues[-1])
    schur_condition = (
        math.inf if schur_minimum <= 0.0 else schur_maximum / schur_minimum
    )
    if schur_minimum <= 0.0 or schur_condition > float(
        maximum_constraint_condition_number
    ):
        raise np.linalg.LinAlgError(
            "Charge/dipole constraint Schur complement is ill-conditioned."
        )

    source = np.concatenate((charges[:, None], dipoles), axis=1).reshape(-1)
    target = np.concatenate(([target_charge], target_dipole))
    residual = target - constraints @ source
    multipliers = np.linalg.solve(schur, residual)
    correction = inverse_constraint_dual @ multipliers
    projected = (source + correction).reshape(len(positions), 4)
    closure_error = float(np.max(np.abs(constraints @ projected.reshape(-1) - target)))
    tolerance = 2.0e-11 * max(1.0, float(np.max(np.abs(target))))
    if closure_error > tolerance or not np.all(np.isfinite(projected)):
        raise RuntimeError("Gaussian Coulomb projection did not close constraints.")
    correction_norm2 = float(correction @ gram @ correction)
    if correction_norm2 < -1.0e-12:
        raise RuntimeError("Gaussian Coulomb correction norm became negative.")
    return GaussianCoulombProjection(
        charges_e=projected[:, 0],
        dipoles_eangstrom=projected[:, 1:],
        gram_minimum_eigenvalue=minimum,
        gram_condition_number=condition,
        constraint_condition_number=schur_condition,
        constraint_max_abs_error=closure_error,
        coulomb_correction_norm=math.sqrt(max(0.0, correction_norm2)),
    )


__all__ = [
    "GaussianCoulombProjection",
    "charge_dipole_constraint_matrix",
    "gaussian_charge_dipole_coulomb_gram",
    "project_charge_dipoles_in_gaussian_coulomb_metric",
]
