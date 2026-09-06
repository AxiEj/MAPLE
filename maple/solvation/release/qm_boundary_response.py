"""Metric-safe comparison of cavity-boundary potential responses.

This module contains only the small, model-independent algebra used by the
matched-QM boundary-response audit.  It deliberately knows nothing about
MACE, PySCF, PCMSolver, solvation labels, or a particular cavity generator.
The continuum response is supplied as a callable so the same frozen operator
is used for the reference, candidate, and uncertainty comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SurfaceResponse = Callable[[np.ndarray], np.ndarray]


def _vector(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector.")
    return np.ascontiguousarray(result)


def central_difference(plus: object, minus: object, *, step: float) -> np.ndarray:
    """Return ``(plus-minus)/(2*step)`` with strict finite-shape checks."""

    positive = np.asarray(plus, dtype=np.float64)
    negative = np.asarray(minus, dtype=np.float64)
    if positive.size < 1 or positive.ndim < 1 or not np.all(np.isfinite(positive)):
        raise ValueError("plus must be a nonempty finite array.")
    if negative.size < 1 or negative.ndim < 1 or not np.all(np.isfinite(negative)):
        raise ValueError("minus must be a nonempty finite array.")
    if positive.shape != negative.shape:
        raise ValueError("plus and minus must have the same shape.")
    step_value = float(step)
    if not np.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step must be finite and positive.")
    result = (positive - negative) / (2.0 * step_value)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("central difference is non-finite.")
    result.setflags(write=False)
    return result


def response_bilinear(
    left: object,
    right: object,
    *,
    apply_response: SurfaceResponse,
) -> float:
    """Return the passive-continuum bilinear form ``-left.T R right``."""

    left_value = _vector(left, name="left")
    right_value = _vector(right, name="right")
    if left_value.shape != right_value.shape:
        raise ValueError("left and right must have the same shape.")
    response = _vector(apply_response(right_value), name="surface response")
    if response.shape != right_value.shape:
        raise ValueError("surface response changed the vector shape.")
    value = -float(np.vdot(left_value, response))
    if not np.isfinite(value):
        raise RuntimeError("surface response bilinear form is non-finite.")
    return value


def response_norm(
    value: object,
    *,
    apply_response: SurfaceResponse,
    negative_tolerance: float = 2.0e-12,
) -> float:
    """Return ``sqrt(-v.T R v)`` and reject a non-passive response.

    PCMSolver's electrostatic response maps a boundary potential to apparent
    surface charge with a negative-semidefinite quadratic form under the
    convention used by Route 2.  A small absolute tolerance is retained only
    for floating-point roundoff at a true null direction.
    """

    vector = _vector(value, name="value")
    tolerance = float(negative_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("negative_tolerance must be finite and nonnegative.")
    squared = response_bilinear(
        vector,
        vector,
        apply_response=apply_response,
    )
    scale = max(1.0, float(np.vdot(vector, vector)))
    if squared < -tolerance * scale:
        raise RuntimeError("surface response is not passive in this direction.")
    return float(np.sqrt(max(0.0, squared)))


def area_weighted_relative_error(
    candidate: object,
    reference: object,
    *,
    areas: object,
) -> float:
    """Return the area-weighted relative L2 error on one frozen surface."""

    candidate_value = _vector(candidate, name="candidate")
    reference_value = _vector(reference, name="reference")
    area_value = _vector(areas, name="areas")
    if not (candidate_value.shape == reference_value.shape == area_value.shape):
        raise ValueError("candidate, reference, and areas must have one shape.")
    if np.any(area_value <= 0.0):
        raise ValueError("surface areas must be strictly positive.")
    error = candidate_value - reference_value
    numerator = float(np.vdot(area_value * error, error))
    denominator = float(np.vdot(area_value * reference_value, reference_value))
    if denominator <= np.finfo(np.float64).tiny:
        raise ValueError("reference has zero area-weighted norm.")
    return float(np.sqrt(max(0.0, numerator / denominator)))


@dataclass(frozen=True, slots=True)
class BoundaryResponseComparison:
    """One candidate/reference comparison in physical surface metrics."""

    reference_response_norm: float
    candidate_response_norm: float
    error_response_norm: float
    relative_response_error: float
    area_weighted_relative_error: float
    correlation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reference_response_norm": self.reference_response_norm,
            "candidate_response_norm": self.candidate_response_norm,
            "error_response_norm": self.error_response_norm,
            "relative_response_error": self.relative_response_error,
            "area_weighted_relative_error": self.area_weighted_relative_error,
            "correlation": self.correlation,
        }


def compare_boundary_response(
    candidate: object,
    reference: object,
    *,
    areas: object,
    apply_response: SurfaceResponse,
) -> BoundaryResponseComparison:
    """Compare two boundary-potential directions using one response operator."""

    candidate_value = _vector(candidate, name="candidate")
    reference_value = _vector(reference, name="reference")
    if candidate_value.shape != reference_value.shape:
        raise ValueError("candidate and reference must have the same shape.")
    error = candidate_value - reference_value
    reference_norm = response_norm(
        reference_value,
        apply_response=apply_response,
    )
    candidate_norm = response_norm(
        candidate_value,
        apply_response=apply_response,
    )
    error_norm = response_norm(error, apply_response=apply_response)
    denominator = max(reference_norm, np.finfo(np.float64).tiny)
    correlation_denominator = max(
        reference_norm * candidate_norm,
        np.finfo(np.float64).tiny,
    )
    correlation = (
        response_bilinear(
            candidate_value,
            reference_value,
            apply_response=apply_response,
        )
        / correlation_denominator
    )
    return BoundaryResponseComparison(
        reference_response_norm=reference_norm,
        candidate_response_norm=candidate_norm,
        error_response_norm=error_norm,
        relative_response_error=error_norm / denominator,
        area_weighted_relative_error=area_weighted_relative_error(
            candidate_value,
            reference_value,
            areas=areas,
        ),
        correlation=float(correlation),
    )


@dataclass(frozen=True, slots=True)
class ReferenceUncertainty:
    """Finite-field and basis components of one reference uncertainty."""

    finite_field_response_norm: float
    basis_response_norm: float
    envelope_response_norm: float
    envelope_relative_to_reference: float

    def as_dict(self) -> dict[str, float]:
        return {
            "finite_field_response_norm": self.finite_field_response_norm,
            "basis_response_norm": self.basis_response_norm,
            "envelope_response_norm": self.envelope_response_norm,
            "envelope_relative_to_reference": (self.envelope_relative_to_reference),
        }


def reference_uncertainty(
    *,
    primary_fine: object,
    primary_coarse: object,
    control_fine: object,
    apply_response: SurfaceResponse,
) -> ReferenceUncertainty:
    """Freeze the larger of finite-field and one-basis control differences."""

    fine = _vector(primary_fine, name="primary_fine")
    coarse = _vector(primary_coarse, name="primary_coarse")
    control = _vector(control_fine, name="control_fine")
    if not (fine.shape == coarse.shape == control.shape):
        raise ValueError("all reference responses must have the same shape.")
    finite_field = response_norm(
        fine - coarse,
        apply_response=apply_response,
    )
    basis = response_norm(fine - control, apply_response=apply_response)
    envelope = max(finite_field, basis)
    reference = response_norm(fine, apply_response=apply_response)
    return ReferenceUncertainty(
        finite_field_response_norm=finite_field,
        basis_response_norm=basis,
        envelope_response_norm=envelope,
        envelope_relative_to_reference=envelope
        / max(reference, np.finfo(np.float64).tiny),
    )


__all__ = [
    "BoundaryResponseComparison",
    "ReferenceUncertainty",
    "area_weighted_relative_error",
    "central_difference",
    "compare_boundary_response",
    "reference_uncertainty",
    "response_bilinear",
    "response_norm",
]
