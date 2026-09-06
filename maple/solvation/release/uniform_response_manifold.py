"""Target-free dipole closure on a model's own uniform-field response manifold.

This module does not fit parameters.  It asks whether an atom-centred
field-responsive source can be moved, using only a spatially uniform affine
potential, until its molecular dipole matches an independently supplied
physical observable.  The source redistribution is therefore restricted to
the response directions already encoded by the frozen model.

The native MACE-POLAR receiver has two radial blocks.  A normalized spherical
convolution leaves an affine potential unchanged, so both radial blocks receive
the same potential and gradient.  Components retain the checkpoint's
``[potential_1, potential_2, gy_1, gz_1, gx_1, gy_2, gz_2, gx_2]`` order.

This is an evaluation primitive only.  A successful closure is not an accuracy,
energy, force, or variational admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
)


def _finite_positions(values: object) -> np.ndarray:
    positions = np.asarray(values, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("positions_angstrom must be finite with shape (N,3).")
    return np.array(positions, copy=True)


def _finite_vector(values: object, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape (3,).")
    return np.array(vector, copy=True)


def molecular_dipole_eangstrom(
    positions_angstrom: object,
    source4_raw_l1: object,
) -> np.ndarray:
    """Return ``sum_A(q_A R_A + p_A)`` in Cartesian ``e Angstrom``."""

    positions = _finite_positions(positions_angstrom)
    source = np.asarray(source4_raw_l1, dtype=float)
    if source.shape != (len(positions), 4) or not np.all(np.isfinite(source)):
        raise ValueError("source4_raw_l1 must be finite with shape (N,4).")
    charges, dipoles = cartesian_multipoles(source)
    return np.sum(charges[:, None] * positions + dipoles, axis=0)


def affine_uniform_native_field(
    positions_angstrom: object,
    potential_gradient_ev_per_e_angstrom: object,
) -> np.ndarray:
    """Construct the two-width native field for one affine potential.

    The gauge origin is the arithmetic centroid.  This makes the construction
    exactly translation invariant while preserving the full affine gradient.
    The returned vector is a potential-gradient chart, not a physical electric
    field (``E = -grad(phi)``).
    """

    positions = _finite_positions(positions_angstrom)
    gradient = _finite_vector(
        potential_gradient_ev_per_e_angstrom,
        name="potential_gradient_ev_per_e_angstrom",
    )
    centered = positions - np.mean(positions, axis=0)
    potential = centered @ gradient
    result = np.empty((len(positions), 8), dtype=float)
    result[:, 0] = potential
    result[:, 1] = potential
    result[:, 2] = gradient[1]
    result[:, 3] = gradient[2]
    result[:, 4] = gradient[0]
    result[:, 5] = gradient[1]
    result[:, 6] = gradient[2]
    result[:, 7] = gradient[0]
    return result


@dataclass(frozen=True, slots=True)
class UniformResponseDipoleClosure:
    """Immutable result of the fail-closed three-dimensional Newton solve."""

    source4_raw_l1: np.ndarray
    potential_gradient_ev_per_e_angstrom: np.ndarray
    molecular_dipole_eangstrom: np.ndarray
    target_molecular_dipole_eangstrom: np.ndarray
    residual_norm_eangstrom: float
    total_charge_change_e: float
    iterations: int
    maximum_jacobian_condition_number: float
    minimum_jacobian_singular_value: float
    converged: bool

    def __post_init__(self) -> None:
        source = np.asarray(self.source4_raw_l1, dtype=float)
        if source.ndim != 2 or source.shape[1] != 4 or not np.all(np.isfinite(source)):
            raise ValueError("source4_raw_l1 must be finite with shape (N,4).")
        source = np.frombuffer(
            np.ascontiguousarray(source, dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(source.shape)
        object.__setattr__(self, "source4_raw_l1", source)
        for name in (
            "potential_gradient_ev_per_e_angstrom",
            "molecular_dipole_eangstrom",
            "target_molecular_dipole_eangstrom",
        ):
            vector = _finite_vector(getattr(self, name), name=name)
            vector = np.frombuffer(vector.tobytes(), dtype=np.float64)
            object.__setattr__(self, name, vector)
        for name in (
            "residual_norm_eangstrom",
            "total_charge_change_e",
            "maximum_jacobian_condition_number",
            "minimum_jacobian_singular_value",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if isinstance(self.iterations, bool) or self.iterations < 0:
            raise ValueError("iterations must be a nonnegative integer.")
        if type(self.converged) is not bool:
            raise TypeError("converged must be exactly bool.")


def solve_uniform_response_dipole_closure(
    *,
    positions_angstrom: object,
    target_molecular_dipole_eangstrom: object,
    evaluate_source: Callable[[np.ndarray], np.ndarray],
    source_jvp: Callable[[np.ndarray, np.ndarray], np.ndarray],
    dipole_tolerance_eangstrom: float = 1.0e-10,
    charge_tolerance_e: float = 1.0e-10,
    maximum_iterations: int = 12,
    maximum_gradient_norm_ev_per_e_angstrom: float = 1.0,
    maximum_jacobian_condition_number: float = 1.0e8,
    minimum_jacobian_singular_value: float = 1.0e-8,
    minimum_line_search_fraction: float = 2.0**-12,
) -> UniformResponseDipoleClosure:
    """Close a target molecular dipole using only frozen-model response.

    ``evaluate_source(field)`` and ``source_jvp(field, direction)`` must consume
    the native two-width field defined by :func:`affine_uniform_native_field`.
    Any singular/ill-conditioned response, charge drift, excessive latent
    field, non-decreasing Newton step, or incomplete closure fails closed.
    """

    positions = _finite_positions(positions_angstrom)
    target = _finite_vector(
        target_molecular_dipole_eangstrom,
        name="target_molecular_dipole_eangstrom",
    )
    tolerance = float(dipole_tolerance_eangstrom)
    charge_tolerance = float(charge_tolerance_e)
    max_field = float(maximum_gradient_norm_ev_per_e_angstrom)
    max_condition = float(maximum_jacobian_condition_number)
    min_singular = float(minimum_jacobian_singular_value)
    min_fraction = float(minimum_line_search_fraction)
    if (
        tolerance <= 0.0
        or charge_tolerance <= 0.0
        or max_field <= 0.0
        or max_condition < 1.0
        or min_singular <= 0.0
        or not 0.0 < min_fraction <= 1.0
        or not all(
            np.isfinite(value)
            for value in (
                tolerance,
                charge_tolerance,
                max_field,
                max_condition,
                min_singular,
                min_fraction,
            )
        )
    ):
        raise ValueError("uniform-response closure thresholds are invalid.")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer.")

    zero_gradient = np.zeros(3, dtype=float)
    zero_field = affine_uniform_native_field(positions, zero_gradient)
    zero_source = np.asarray(evaluate_source(zero_field), dtype=float)
    if zero_source.shape != (len(positions), 4) or not np.all(np.isfinite(zero_source)):
        raise ValueError("evaluate_source returned an invalid zero-field source.")
    zero_charge = float(np.sum(zero_source[:, 0]))

    basis_fields = tuple(
        affine_uniform_native_field(positions, np.eye(3)[axis]) for axis in range(3)
    )
    gradient = zero_gradient.copy()
    maximum_condition = 0.0
    minimum_seen_singular = np.inf
    source = zero_source
    molecular = molecular_dipole_eangstrom(positions, source)

    for iteration in range(maximum_iterations + 1):
        residual = molecular - target
        residual_norm = float(np.linalg.norm(residual))
        charge_change = float(np.sum(source[:, 0]) - zero_charge)
        if abs(charge_change) > charge_tolerance:
            raise RuntimeError("uniform response changed the checkpoint total charge.")
        if residual_norm <= tolerance:
            return UniformResponseDipoleClosure(
                source4_raw_l1=source,
                potential_gradient_ev_per_e_angstrom=gradient,
                molecular_dipole_eangstrom=molecular,
                target_molecular_dipole_eangstrom=target,
                residual_norm_eangstrom=residual_norm,
                total_charge_change_e=charge_change,
                iterations=iteration,
                maximum_jacobian_condition_number=maximum_condition,
                minimum_jacobian_singular_value=(
                    0.0 if np.isinf(minimum_seen_singular) else minimum_seen_singular
                ),
                converged=True,
            )
        if iteration == maximum_iterations:
            break

        field = affine_uniform_native_field(positions, gradient)
        jacobian = np.empty((3, 3), dtype=float)
        for axis, direction in enumerate(basis_fields):
            tangent = np.asarray(source_jvp(field, direction), dtype=float)
            if tangent.shape != source.shape or not np.all(np.isfinite(tangent)):
                raise RuntimeError("source_jvp returned an invalid tangent.")
            jacobian[:, axis] = molecular_dipole_eangstrom(positions, tangent)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        smallest = float(singular_values[-1])
        condition = float(singular_values[0] / max(smallest, np.finfo(float).tiny))
        maximum_condition = max(maximum_condition, condition)
        minimum_seen_singular = min(minimum_seen_singular, smallest)
        if smallest < min_singular or condition > max_condition:
            raise RuntimeError(
                "uniform dipole response is singular or ill-conditioned."
            )
        step = np.linalg.solve(jacobian, -residual)

        accepted = False
        fraction = 1.0
        while fraction >= min_fraction:
            candidate_gradient = gradient + fraction * step
            if float(np.linalg.norm(candidate_gradient)) > max_field:
                fraction *= 0.5
                continue
            candidate_field = affine_uniform_native_field(positions, candidate_gradient)
            candidate_source = np.asarray(evaluate_source(candidate_field), dtype=float)
            if candidate_source.shape != source.shape or not np.all(
                np.isfinite(candidate_source)
            ):
                raise RuntimeError("evaluate_source returned an invalid source.")
            candidate_molecular = molecular_dipole_eangstrom(
                positions, candidate_source
            )
            if float(np.linalg.norm(candidate_molecular - target)) < residual_norm:
                gradient = candidate_gradient
                source = candidate_source
                molecular = candidate_molecular
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            raise RuntimeError("uniform-response Newton line search did not decrease.")

    raise RuntimeError("uniform-response dipole closure did not converge.")


__all__ = [
    "UniformResponseDipoleClosure",
    "affine_uniform_native_field",
    "molecular_dipole_eangstrom",
    "solve_uniform_response_dipole_closure",
]
