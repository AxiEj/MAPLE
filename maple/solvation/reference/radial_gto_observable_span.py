"""Observable-only feasibility of the two-width MACE-POLAR radial source span.

The fitted coefficients are intentionally ephemeral and are never returned.
This diagnostic asks only whether a fixed geometry's full molecular MEP and
four directional response MEPs lie in the declared 8N radial-GTO source span,
with exact total-charge constraints and an independent dense audit partition.
It is a representation gate, not a coefficient-label generator or a trained
model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase.units import Hartree

from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.spaces import MACE_POLAR_RADIAL_GTO_SOURCE_SPACE


def _finite_vector(values: object, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({size},).")
    return result


def _weighted_relative(
    prediction: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> float:
    error = prediction - reference
    numerator = float(np.sum(weights * error * error))
    denominator = float(np.sum(weights * reference * reference))
    if denominator <= 0.0:
        raise ValueError("reference MEP has zero weighted norm.")
    return math.sqrt(numerator / denominator)


@dataclass(frozen=True, slots=True)
class RadialGTOObservableSpanResult:
    atom_count: int
    source_dimension: int
    reduced_dimension: int
    fit_point_count: int
    audit_point_count: int
    weighted_fit_operator_rank: int
    weighted_fit_operator_condition_number: float
    zero_mep_fit_relative: float
    zero_mep_audit_relative: float
    zero_mep_audit_max_absolute_hartree_per_e: float
    response_mep_fit_relative: float
    response_mep_audit_relative: float
    response_mep_audit_max_absolute_hartree_per_e_per_source_e: float
    maximum_charge_constraint_residual_e: float
    source_response_relative_error: float
    source_response_reciprocity_relative: float
    source_response_symmetric_eigenvalues_hartree_per_e2: tuple[float, ...]
    response_direction_rank: int
    positive_hessian_extension_minimum_gram_eigenvalue: float
    coefficient_label_emitted: bool = False
    model_fit_performed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "atom_count",
            "source_dimension",
            "reduced_dimension",
            "fit_point_count",
            "audit_point_count",
            "weighted_fit_operator_rank",
            "response_direction_rank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "weighted_fit_operator_condition_number",
            "zero_mep_fit_relative",
            "zero_mep_audit_relative",
            "zero_mep_audit_max_absolute_hartree_per_e",
            "response_mep_fit_relative",
            "response_mep_audit_relative",
            "response_mep_audit_max_absolute_hartree_per_e_per_source_e",
            "maximum_charge_constraint_residual_e",
            "source_response_relative_error",
            "source_response_reciprocity_relative",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        extension_eigenvalue = float(
            self.positive_hessian_extension_minimum_gram_eigenvalue
        )
        if not math.isfinite(extension_eigenvalue):
            raise ValueError("Hessian-extension Gram eigenvalue must be finite.")
        object.__setattr__(
            self,
            "positive_hessian_extension_minimum_gram_eigenvalue",
            extension_eigenvalue,
        )
        eigenvalues = tuple(
            float(value)
            for value in self.source_response_symmetric_eigenvalues_hartree_per_e2
        )
        if not eigenvalues or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("source-response eigenvalues must be finite and nonempty.")
        object.__setattr__(
            self,
            "source_response_symmetric_eigenvalues_hartree_per_e2",
            eigenvalues,
        )
        if self.coefficient_label_emitted or self.model_fit_performed:
            raise ValueError("observable span diagnostics cannot emit labels or fit a model.")


def evaluate_radial_gto_observable_span(
    *,
    atom_positions_angstrom: object,
    surface_points_bohr: object,
    quadrature_weights: object,
    partition_indices: object,
    source_surface_indices: object,
    zero_total_surface_mep_hartree_per_e: object,
    induced_surface_mep_hartree_per_e_per_source_e: object,
    total_charge_e: float,
    relative_svd_cutoff: float = 1.0e-12,
) -> RadialGTOObservableSpanResult:
    """Fit only the dense fit partition and report independent audit errors."""

    from scipy import linalg

    positions = np.asarray(atom_positions_angstrom, dtype=np.float64)
    points = np.asarray(surface_points_bohr, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("atom_positions_angstrom must have finite shape (N,3).")
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("surface_points_bohr must have finite shape (M,3).")
    weights = _finite_vector(
        quadrature_weights,
        size=len(points),
        name="quadrature_weights",
    )
    if np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be positive.")
    partitions = np.asarray(partition_indices)
    if (
        partitions.shape != (len(points),)
        or not np.issubdtype(partitions.dtype, np.integer)
        or set(partitions.tolist()) != {0, 1}
    ):
        raise ValueError("partition_indices must contain fit=0 and audit=1.")
    source_indices = np.asarray(source_surface_indices)
    if (
        source_indices.shape != (4,)
        or not np.issubdtype(source_indices.dtype, np.integer)
        or np.any(source_indices < 0)
        or np.any(source_indices >= len(points))
        or np.any(partitions[source_indices] != 0)
    ):
        raise ValueError("four source indices must lie in the fit partition.")
    zero_mep = _finite_vector(
        zero_total_surface_mep_hartree_per_e,
        size=len(points),
        name="zero_total_surface_mep_hartree_per_e",
    )
    response_mep = np.asarray(
        induced_surface_mep_hartree_per_e_per_source_e,
        dtype=np.float64,
    )
    if response_mep.shape != (4, len(points)) or not np.all(
        np.isfinite(response_mep)
    ):
        raise ValueError("induced surface MEP must have finite shape (4,M).")
    total_charge = float(total_charge_e)
    if not math.isfinite(total_charge):
        raise ValueError("total_charge_e must be finite.")
    cutoff = float(relative_svd_cutoff)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("relative_svd_cutoff must lie strictly in (0,1).")

    geometry = FixedSurfaceGeometry(positions, points)
    operator = MACEPolarRadialGTOCoupling().surface_operator(geometry) / Hartree
    atom_count = len(positions)
    source_dimension = operator.shape[1]
    charge_covector = np.tile(
        np.asarray(
            MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.effective_charge_weights,
            dtype=np.float64,
        ),
        atom_count,
    )
    nullspace = linalg.null_space(charge_covector[None, :])
    reduced_dimension = int(nullspace.shape[1])
    reference_source = (
        total_charge
        * charge_covector
        / float(np.dot(charge_covector, charge_covector))
    )
    fit = partitions == 0
    audit = partitions == 1
    fit_scale = np.sqrt(weights[fit] / float(np.mean(weights[fit])))
    weighted_operator = fit_scale[:, None] * (operator[fit] @ nullspace)
    singular_values = linalg.svdvals(weighted_operator)
    retained = singular_values > singular_values[0] * cutoff
    rank = int(np.count_nonzero(retained))
    if rank < reduced_dimension:
        raise RuntimeError("dense fit operator does not span the charge tangent.")
    condition = float(singular_values[0] / singular_values[-1])

    def solve_fit(target: np.ndarray, *, affine: bool) -> np.ndarray:
        offset = operator @ reference_source if affine else np.zeros(len(points))
        reduced = linalg.lstsq(
            weighted_operator,
            fit_scale * (target[fit] - offset[fit]),
            cond=cutoff,
            lapack_driver="gelsd",
        )[0]
        return (reference_source if affine else 0.0) + nullspace @ reduced

    zero_source = solve_fit(zero_mep, affine=True)
    response_sources = np.column_stack(
        [solve_fit(response_mep[mode], affine=False) for mode in range(4)]
    )
    predicted_zero = operator @ zero_source
    predicted_response = (operator @ response_sources).T
    source_response_reference = response_mep[:, source_indices].T
    source_response_prediction = operator[source_indices] @ response_sources
    source_response_scale = max(
        float(np.linalg.norm(source_response_reference)),
        float(np.linalg.norm(source_response_prediction)),
        np.finfo(float).tiny,
    )
    symmetric_response = 0.5 * (
        source_response_prediction + source_response_prediction.T
    )
    positive_gram = -0.5 * (
        response_sources.T @ operator[source_indices].T
        + operator[source_indices] @ response_sources
    )
    positive_gram = 0.5 * (positive_gram + positive_gram.T)
    return RadialGTOObservableSpanResult(
        atom_count=atom_count,
        source_dimension=source_dimension,
        reduced_dimension=reduced_dimension,
        fit_point_count=int(np.count_nonzero(fit)),
        audit_point_count=int(np.count_nonzero(audit)),
        weighted_fit_operator_rank=rank,
        weighted_fit_operator_condition_number=condition,
        zero_mep_fit_relative=_weighted_relative(
            predicted_zero[fit], zero_mep[fit], weights[fit]
        ),
        zero_mep_audit_relative=_weighted_relative(
            predicted_zero[audit], zero_mep[audit], weights[audit]
        ),
        zero_mep_audit_max_absolute_hartree_per_e=float(
            np.max(np.abs(predicted_zero[audit] - zero_mep[audit]))
        ),
        response_mep_fit_relative=_weighted_relative(
            predicted_response[:, fit],
            response_mep[:, fit],
            np.broadcast_to(weights[fit], predicted_response[:, fit].shape),
        ),
        response_mep_audit_relative=_weighted_relative(
            predicted_response[:, audit],
            response_mep[:, audit],
            np.broadcast_to(weights[audit], predicted_response[:, audit].shape),
        ),
        response_mep_audit_max_absolute_hartree_per_e_per_source_e=float(
            np.max(
                np.abs(predicted_response[:, audit] - response_mep[:, audit])
            )
        ),
        maximum_charge_constraint_residual_e=max(
            abs(float(np.dot(charge_covector, zero_source)) - total_charge),
            float(np.max(np.abs(charge_covector @ response_sources))),
        ),
        source_response_relative_error=float(
            np.linalg.norm(
                source_response_prediction - source_response_reference
            )
            / source_response_scale
        ),
        source_response_reciprocity_relative=float(
            np.linalg.norm(
                source_response_prediction - source_response_prediction.T
            )
            / max(
                float(np.linalg.norm(source_response_prediction)),
                np.finfo(float).tiny,
            )
        ),
        source_response_symmetric_eigenvalues_hartree_per_e2=tuple(
            float(value) for value in np.linalg.eigvalsh(symmetric_response)
        ),
        response_direction_rank=int(np.linalg.matrix_rank(response_sources)),
        positive_hessian_extension_minimum_gram_eigenvalue=float(
            np.linalg.eigvalsh(positive_gram)[0]
        ),
    )


__all__ = [
    "RadialGTOObservableSpanResult",
    "evaluate_radial_gto_observable_span",
]
