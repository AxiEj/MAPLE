"""Target-independent sector-balanced nonuniform response-mode selection.

The selector operates only on analytic field rows induced by exterior point
charges.  It preserves an existing frozen mode prefix, removes the constant-
potential gauge direction, balances the radial and quadrupole sectors by
geometry-only RMS norms, and greedily maximizes the residual row volume.

No QM response, model prediction, PCM quantity, or solvation label enters the
selection.  Orthogonal changes within irreducible source/field blocks therefore
leave the selected exterior point identities unchanged up to numerical ties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _finite_matrix(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or 0 in result.shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty matrix.")
    return result


@dataclass(frozen=True, slots=True)
class SectorBalancedResponseModes:
    source_surface_indices: tuple[int, ...]
    frozen_prefix_count: int
    radial_gauge_reduced_rms: float
    l2_rms: float
    balanced_singular_values: tuple[float, ...]
    balanced_rank: int
    balanced_condition_number: float
    minimum_greedy_residual_norm: float
    radial_fraction_of_selected_balanced_norm: float
    target_used: bool = False

    def __post_init__(self) -> None:
        indices = tuple(int(value) for value in self.source_surface_indices)
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("source_surface_indices must be unique and nonnegative.")
        prefix = int(self.frozen_prefix_count)
        if not 0 <= prefix <= len(indices):
            raise ValueError("frozen_prefix_count is outside the selected modes.")
        positive = (
            "radial_gauge_reduced_rms",
            "l2_rms",
            "balanced_condition_number",
            "minimum_greedy_residual_norm",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        singular_values = tuple(float(value) for value in self.balanced_singular_values)
        if (
            len(singular_values) != len(indices)
            or not np.all(np.isfinite(singular_values))
            or min(singular_values) <= 0.0
        ):
            raise ValueError("balanced singular values must be positive and complete.")
        rank = int(self.balanced_rank)
        if rank != len(indices):
            raise ValueError("selected balanced modes must have full row rank.")
        fraction = float(self.radial_fraction_of_selected_balanced_norm)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("selected radial norm fraction must lie in (0,1).")
        if self.target_used:
            raise ValueError("response-mode selection cannot use a target.")
        object.__setattr__(self, "source_surface_indices", indices)
        object.__setattr__(self, "frozen_prefix_count", prefix)
        object.__setattr__(self, "balanced_singular_values", singular_values)
        object.__setattr__(self, "balanced_rank", rank)
        object.__setattr__(self, "radial_fraction_of_selected_balanced_norm", fraction)


def select_sector_balanced_response_modes(
    *,
    radial_field_operator: object,
    l2_field_operator: object,
    fit_surface_indices: object,
    frozen_prefix_indices: object,
    mode_count: int,
    relative_rank_tolerance: float = 1.0e-11,
) -> SectorBalancedResponseModes:
    """Select exterior modes by balanced gauge-reduced row-volume growth."""

    radial = _finite_matrix(radial_field_operator, name="radial_field_operator")
    l2 = _finite_matrix(l2_field_operator, name="l2_field_operator")
    if radial.shape[0] != l2.shape[0]:
        raise ValueError("radial and l2 operators must share exterior points.")
    if radial.shape[1] % 8 != 0 or l2.shape[1] % 5 != 0:
        raise ValueError("radial/l2 dimensions must be 8N and 5N.")
    atom_count = radial.shape[1] // 8
    if atom_count < 1 or l2.shape[1] != 5 * atom_count:
        raise ValueError("radial/l2 operators describe different atom counts.")
    fit = np.asarray(fit_surface_indices)
    prefix = np.asarray(frozen_prefix_indices)
    if (
        fit.ndim != 1
        or not np.issubdtype(fit.dtype, np.integer)
        or len(fit) == 0
        or len(set(fit.tolist())) != len(fit)
        or np.any(fit < 0)
        or np.any(fit >= radial.shape[0])
    ):
        raise ValueError("fit_surface_indices must be unique valid integers.")
    if (
        prefix.ndim != 1
        or not np.issubdtype(prefix.dtype, np.integer)
        or len(set(prefix.tolist())) != len(prefix)
        or not set(prefix.tolist()).issubset(set(fit.tolist()))
    ):
        raise ValueError("frozen_prefix_indices must be a unique subset of fit points.")
    if isinstance(mode_count, bool) or not isinstance(mode_count, int):
        raise TypeError("mode_count must be an integer.")
    if not len(prefix) <= mode_count <= len(fit):
        raise ValueError("mode_count must include the prefix and fit within candidates.")
    tolerance = float(relative_rank_tolerance)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0,1).")

    gauge = np.zeros(radial.shape[1], dtype=np.float64)
    gauge.reshape(atom_count, 8)[:, :2] = 1.0
    gauge /= np.linalg.norm(gauge)
    radial_reduced = radial - np.outer(radial @ gauge, gauge)
    radial_rms = math.sqrt(float(np.mean(np.sum(radial_reduced[fit] ** 2, axis=1))))
    l2_rms = math.sqrt(float(np.mean(np.sum(l2[fit] ** 2, axis=1))))
    if radial_rms <= 0.0 or l2_rms <= 0.0:
        raise RuntimeError("one response sector has zero geometry-only RMS norm.")
    balanced = np.concatenate(
        (radial_reduced / radial_rms, l2 / l2_rms), axis=1
    )

    selected = [int(value) for value in prefix]
    residual_norms: list[float] = []
    candidate_set = set(int(value) for value in fit)
    while len(selected) < mode_count:
        if selected:
            rows = balanced[selected]
            _, singular_values, right = np.linalg.svd(rows, full_matrices=False)
            threshold = singular_values[0] * tolerance
            retained = singular_values > threshold
            if int(np.count_nonzero(retained)) != len(selected):
                raise RuntimeError("frozen/selected response modes lost row rank.")
            basis = right[retained].T
        else:
            basis = np.empty((balanced.shape[1], 0), dtype=np.float64)
        best_index = None
        best_norm = -1.0
        for index in sorted(candidate_set.difference(selected)):
            row = balanced[index]
            residual = row - basis @ (basis.T @ row)
            norm = float(np.linalg.norm(residual))
            tie = 64.0 * np.finfo(float).eps * max(1.0, abs(best_norm), abs(norm))
            if norm > best_norm + tie or (
                abs(norm - best_norm) <= tie
                and (best_index is None or index < best_index)
            ):
                best_index = index
                best_norm = norm
        if best_index is None or best_norm <= 0.0:
            raise RuntimeError("candidate field rows cannot extend the selected rank.")
        selected.append(best_index)
        residual_norms.append(best_norm)

    selected_rows = balanced[selected]
    singular_values = np.linalg.svd(selected_rows, compute_uv=False)
    threshold = singular_values[0] * tolerance
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank != mode_count:
        raise RuntimeError("selected response modes are numerically rank deficient.")
    radial_selected = radial_reduced[selected] / radial_rms
    l2_selected = l2[selected] / l2_rms
    radial_norm2 = float(np.sum(radial_selected * radial_selected))
    l2_norm2 = float(np.sum(l2_selected * l2_selected))
    minimum_residual = (
        min(residual_norms) if residual_norms else float(singular_values[-1])
    )
    return SectorBalancedResponseModes(
        source_surface_indices=tuple(selected),
        frozen_prefix_count=len(prefix),
        radial_gauge_reduced_rms=radial_rms,
        l2_rms=l2_rms,
        balanced_singular_values=tuple(float(value) for value in singular_values),
        balanced_rank=rank,
        balanced_condition_number=float(singular_values[0] / singular_values[-1]),
        minimum_greedy_residual_norm=minimum_residual,
        radial_fraction_of_selected_balanced_norm=radial_norm2
        / (radial_norm2 + l2_norm2),
        target_used=False,
    )


__all__ = [
    "SectorBalancedResponseModes",
    "select_sector_balanced_response_modes",
]
