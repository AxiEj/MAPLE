"""Observable feasibility of radial8 plus static point-l=2 permanent sources."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase.units import Hartree

from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.point_quadrupole import (
    point_traceless_quadrupole_surface_operator,
)
from maple.solvation.coupling.spaces import MACE_POLAR_RADIAL_GTO_SOURCE_SPACE


@dataclass(frozen=True, slots=True)
class P13PermanentSpanResult:
    atom_count: int
    source_dimension: int
    reduced_dimension: int
    fit_point_count: int
    audit_point_count: int
    weighted_fit_operator_rank: int
    weighted_fit_operator_condition_number: float
    fit_relative_mep_error: float
    audit_relative_mep_error: float
    audit_max_absolute_hartree_per_e: float
    audit_minus_fit_relative: float
    audit_to_fit_ratio: float
    charge_constraint_residual_e: float
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
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "weighted_fit_operator_condition_number",
            "fit_relative_mep_error",
            "audit_relative_mep_error",
            "audit_max_absolute_hartree_per_e",
            "audit_minus_fit_relative",
            "audit_to_fit_ratio",
            "charge_constraint_residual_e",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        if self.coefficient_label_emitted or self.model_fit_performed:
            raise ValueError("P13 span diagnostics cannot emit labels or fit a model.")


def evaluate_p13_permanent_span(
    *,
    atom_positions_angstrom: object,
    surface_points_bohr: object,
    quadrature_weights: object,
    partition_indices: object,
    total_surface_mep_hartree_per_e: object,
    total_charge_e: float,
    relative_svd_cutoff: float = 1.0e-12,
) -> P13PermanentSpanResult:
    """Fit P13 only on partition zero and evaluate partition one."""

    from scipy import linalg

    positions = np.asarray(atom_positions_angstrom, dtype=np.float64)
    points = np.asarray(surface_points_bohr, dtype=np.float64)
    weights = np.asarray(quadrature_weights, dtype=np.float64)
    partitions = np.asarray(partition_indices)
    target = np.asarray(total_surface_mep_hartree_per_e, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("atom positions must have finite shape (N,3).")
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("surface points must have finite shape (M,3).")
    if (
        weights.shape != (len(points),)
        or target.shape != (len(points),)
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(target))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("weights and target MEP must be finite matched vectors.")
    if (
        partitions.shape != (len(points),)
        or not np.issubdtype(partitions.dtype, np.integer)
        or set(partitions.tolist()) != {0, 1}
    ):
        raise ValueError("partition indices must contain fit=0 and audit=1.")
    charge = float(total_charge_e)
    if not math.isfinite(charge):
        raise ValueError("total_charge_e must be finite.")
    cutoff = float(relative_svd_cutoff)
    if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
        raise ValueError("relative_svd_cutoff must lie in (0,1).")

    radial = MACEPolarRadialGTOCoupling().surface_operator(
        FixedSurfaceGeometry(positions, points)
    ) / Hartree
    quadrupole = point_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=positions,
    )
    operator = np.concatenate((radial, quadrupole), axis=1)
    atom_count = len(positions)
    charge_covector = np.concatenate(
        (
            np.tile(
                MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.effective_charge_weights,
                atom_count,
            ),
            np.zeros(5 * atom_count),
        )
    )
    nullspace = linalg.null_space(charge_covector[None, :])
    reduced_dimension = nullspace.shape[1]
    reference_source = (
        charge
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
        raise RuntimeError("P13 fit operator is not full rank at the frozen cutoff.")
    condition = float(singular_values[0] / singular_values[-1])
    offset = operator @ reference_source
    reduced = linalg.lstsq(
        weighted_operator,
        fit_scale * (target[fit] - offset[fit]),
        cond=cutoff,
        lapack_driver="gelsd",
    )[0]
    source = reference_source + nullspace @ reduced
    prediction = operator @ source

    def relative(mask: np.ndarray) -> float:
        difference = prediction[mask] - target[mask]
        return float(
            np.sqrt(
                np.sum(weights[mask] * difference * difference)
                / np.sum(weights[mask] * target[mask] * target[mask])
            )
        )

    fit_error = relative(fit)
    audit_error = relative(audit)
    return P13PermanentSpanResult(
        atom_count=atom_count,
        source_dimension=operator.shape[1],
        reduced_dimension=reduced_dimension,
        fit_point_count=int(np.count_nonzero(fit)),
        audit_point_count=int(np.count_nonzero(audit)),
        weighted_fit_operator_rank=rank,
        weighted_fit_operator_condition_number=condition,
        fit_relative_mep_error=fit_error,
        audit_relative_mep_error=audit_error,
        audit_max_absolute_hartree_per_e=float(
            np.max(np.abs(prediction[audit] - target[audit]))
        ),
        audit_minus_fit_relative=max(0.0, audit_error - fit_error),
        audit_to_fit_ratio=audit_error / max(fit_error, np.finfo(float).tiny),
        charge_constraint_residual_e=abs(
            float(np.dot(charge_covector, source)) - charge
        ),
    )


__all__ = ["P13PermanentSpanResult", "evaluate_p13_permanent_span"]
