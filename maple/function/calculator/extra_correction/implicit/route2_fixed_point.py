"""Numerical fixed-point updates for the Route-2 ML--continuum root.

These updates only choose the next density at which the physical map is
evaluated.  Convergence is always judged with the unmixed residual
``M(P(c)) - c``; neither Picard damping nor Anderson acceleration is part of
the physical residual or energy functional.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

DAMPED_PICARD_SOLVER = "damped-picard-v1"
SAFEGUARDED_ANDERSON_SOLVER = "safeguarded-anderson-v1"
SUPPORTED_FIXED_POINT_SOLVERS = frozenset(
    {
        DAMPED_PICARD_SOLVER,
        SAFEGUARDED_ANDERSON_SOLVER,
    }
)


def _density_block(values: np.ndarray, *, name: str) -> np.ndarray:
    block = np.asarray(values, dtype=float)
    if block.ndim != 2 or block.shape[1] != 4 or not np.all(np.isfinite(block)):
        raise ValueError(f"{name} must be a finite (n_atoms, 4) block.")
    return block


@dataclass(frozen=True)
class FixedPointSample:
    """One evaluated density and its unmixed physical residual."""

    density: np.ndarray
    residual: np.ndarray

    def __post_init__(self) -> None:
        density = np.array(
            _density_block(self.density, name="Fixed-point density"),
            copy=True,
        )
        residual = np.array(
            _density_block(self.residual, name="Fixed-point residual"),
            copy=True,
        )
        if density.shape != residual.shape:
            raise ValueError("Fixed-point density and residual shapes must match.")
        density.setflags(write=False)
        residual.setflags(write=False)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "residual", residual)


@dataclass(frozen=True)
class FixedPointStep:
    """One proposed numerical update and its safeguard provenance."""

    density: np.ndarray
    method: str
    history_size: int
    predicted_residual_l2: float | None = None
    coefficient_l1: float | None = None
    step_ratio_to_picard: float | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        density = np.array(
            _density_block(self.density, name="Next fixed-point density"),
            copy=True,
        )
        density.setflags(write=False)
        object.__setattr__(self, "density", density)
        if self.method not in SUPPORTED_FIXED_POINT_SOLVERS:
            raise ValueError(f"Unsupported fixed-point step: {self.method}.")
        if self.history_size <= 0:
            raise ValueError("Fixed-point history size must be positive.")
        for name in (
            "predicted_residual_l2",
            "coefficient_l1",
            "step_ratio_to_picard",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"Fixed-point diagnostic {name} must be nonnegative.")


def _picard_step(
    sample: FixedPointSample,
    *,
    mixing: float,
    history_size: int,
    fallback_reason: str | None = None,
) -> FixedPointStep:
    return FixedPointStep(
        density=sample.density + mixing * sample.residual,
        method=DAMPED_PICARD_SOLVER,
        history_size=history_size,
        fallback_reason=fallback_reason,
    )


def next_fixed_point_density(
    samples: Sequence[FixedPointSample],
    *,
    solver: str,
    mixing: float,
    anderson_depth: int,
    anderson_regularization: float,
    anderson_coefficient_l1_limit: float,
    anderson_step_ratio_limit: float,
) -> FixedPointStep:
    """Return the next numerical density without changing the root equation.

    The Anderson branch uses the type-II update

    ``x + beta*g - (DeltaX + beta*DeltaG) gamma``

    with a small Tikhonov regularizer.  It falls back to damped Picard unless
    the least-squares prediction, coefficient norm, step size, finite-value,
    and affine total-charge checks all pass.
    """

    if not samples:
        raise ValueError("At least one fixed-point sample is required.")
    if solver not in SUPPORTED_FIXED_POINT_SOLVERS:
        raise ValueError(f"Unsupported fixed-point solver: {solver}.")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("Fixed-point mixing must lie in (0, 1].")
    if anderson_depth <= 0:
        raise ValueError("Anderson depth must be positive.")
    positive = {
        "anderson_regularization": anderson_regularization,
        "anderson_coefficient_l1_limit": (anderson_coefficient_l1_limit),
        "anderson_step_ratio_limit": anderson_step_ratio_limit,
    }
    invalid = [name for name, value in positive.items() if value <= 0.0]
    if invalid:
        raise ValueError(
            "Anderson safeguards must be positive: " + ", ".join(invalid) + "."
        )

    recent = tuple(samples[-(anderson_depth + 1) :])
    current = recent[-1]
    expected_shape = current.density.shape
    if any(
        sample.density.shape != expected_shape
        or sample.residual.shape != expected_shape
        for sample in recent
    ):
        raise ValueError("All fixed-point sample shapes must match.")
    if solver == DAMPED_PICARD_SOLVER or len(recent) < 2:
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
        )

    densities = [sample.density.reshape(-1) for sample in recent]
    residuals = [sample.residual.reshape(-1) for sample in recent]
    delta_density = np.column_stack(
        [densities[index + 1] - densities[index] for index in range(len(recent) - 1)]
    )
    delta_residual = np.column_stack(
        [residuals[index + 1] - residuals[index] for index in range(len(recent) - 1)]
    )
    residual = residuals[-1]
    residual_l2 = float(np.linalg.norm(residual))
    gram_matrix = delta_residual.T @ delta_residual
    gram_scale = float(np.trace(gram_matrix)) / float(
        gram_matrix.shape[0]
    )
    if not math.isfinite(gram_scale) or gram_scale <= np.finfo(float).tiny:
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
            fallback_reason="anderson-degenerate-history",
        )
    normal_matrix = (
        gram_matrix
        + anderson_regularization
        * gram_scale
        * np.eye(delta_residual.shape[1])
    )
    right_hand_side = delta_residual.T @ residual
    try:
        coefficients = np.linalg.solve(
            normal_matrix,
            right_hand_side,
        )
    except np.linalg.LinAlgError:
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
            fallback_reason="anderson-linear-solve",
        )
    coefficient_l1 = float(np.linalg.norm(coefficients, ord=1))
    if (
        not np.all(np.isfinite(coefficients))
        or coefficient_l1 > anderson_coefficient_l1_limit
    ):
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
            fallback_reason="anderson-coefficient-limit",
        )

    predicted_residual = residual - delta_residual @ coefficients
    predicted_residual_l2 = float(np.linalg.norm(predicted_residual))
    if not math.isfinite(
        predicted_residual_l2
    ) or predicted_residual_l2 > residual_l2 * (1.0 + 64.0 * np.finfo(float).eps):
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
            fallback_reason="anderson-predicted-residual",
        )

    picard_delta = mixing * residual
    accelerated_delta = (
        picard_delta - (delta_density + mixing * delta_residual) @ coefficients
    )
    picard_l2 = float(np.linalg.norm(picard_delta))
    accelerated_l2 = float(np.linalg.norm(accelerated_delta))
    step_ratio = accelerated_l2 / max(
        picard_l2,
        np.finfo(float).tiny,
    )
    candidate = current.density + accelerated_delta.reshape(expected_shape)
    if not np.all(np.isfinite(candidate)) or step_ratio > anderson_step_ratio_limit:
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
            fallback_reason="anderson-step-limit",
        )

    current_charge = float(np.sum(current.density[:, 0]))
    candidate_charge = float(np.sum(candidate[:, 0]))
    charge_tolerance = (
        128.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(current_charge),
            float(np.sum(np.abs(current.density[:, 0]))),
        )
    )
    if abs(candidate_charge - current_charge) > charge_tolerance:
        return _picard_step(
            current,
            mixing=mixing,
            history_size=len(recent),
            fallback_reason="anderson-charge-drift",
        )

    return FixedPointStep(
        density=candidate,
        method=SAFEGUARDED_ANDERSON_SOLVER,
        history_size=len(recent),
        predicted_residual_l2=predicted_residual_l2,
        coefficient_l1=coefficient_l1,
        step_ratio_to_picard=step_ratio,
    )


__all__ = [
    "DAMPED_PICARD_SOLVER",
    "SAFEGUARDED_ANDERSON_SOLVER",
    "SUPPORTED_FIXED_POINT_SOLVERS",
    "FixedPointSample",
    "FixedPointStep",
    "next_fixed_point_density",
]
