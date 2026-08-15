"""Stationary-point classification from molecular vibrational curvatures.

The mathematical classification is deliberately separate from display
tolerances. A minimum has only positive vibrational curvatures. A first-order
saddle has exactly one negative vibrational curvature; the configurable
wavenumber threshold is an admission guard against mistaking a shallow
numerical artifact for that reaction mode, not a new transition-state
definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

MINIMUM = "minimum"
TRANSITION_STATE = "transition_state"
STATIONARY_POINT_TARGETS = (MINIMUM, TRANSITION_STATE)


def normalize_stationary_point_target(target: object) -> str:
    """Return a declared stationary-point target or fail closed."""

    if not isinstance(target, str):
        raise ValueError("stationary_point must be 'minimum' or 'transition_state'.")
    normalized = target.strip().lower()
    if normalized not in STATIONARY_POINT_TARGETS:
        raise ValueError("stationary_point must be 'minimum' or 'transition_state'.")
    return normalized


def _positive_threshold(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("imaginary_threshold_cm1 must be a finite positive number.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "imaginary_threshold_cm1 must be a finite positive number."
        ) from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("imaginary_threshold_cm1 must be a finite positive number.")
    return threshold


def _mode_list(frequencies: np.ndarray, indices: tuple[int, ...]) -> str:
    return ", ".join(f"{frequencies[index]:.6g}" for index in indices)


@dataclass(frozen=True)
class StationaryPointAssessment:
    """Validated curvature signature for one stationary-point class."""

    target: str
    vibrational_frequencies_cm1: tuple[float, ...]
    imaginary_threshold_cm1: float
    robust_imaginary_mode_indices: tuple[int, ...]
    ambiguous_nonpositive_mode_indices: tuple[int, ...]
    reinterpreted_negative_frequencies_cm1: tuple[float, ...] = ()
    reinterpretation_threshold_cm1: float | None = None

    @property
    def label(self) -> str:
        if self.target == TRANSITION_STATE:
            return "first-order transition state"
        return "minimum"

    @property
    def thermochemistry_admitted(self) -> bool:
        return self.target == MINIMUM

    @property
    def imaginary_frequency_cm1(self) -> float | None:
        if len(self.robust_imaginary_mode_indices) != 1:
            return None
        return self.vibrational_frequencies_cm1[self.robust_imaginary_mode_indices[0]]


def assess_stationary_point(
    vibrational_frequencies_cm1: object,
    *,
    target: object,
    imaginary_threshold_cm1: object,
    reinterpreted_negative_frequencies_cm1: object = (),
    reinterpretation_threshold_cm1: object = None,
) -> StationaryPointAssessment:
    """Validate a minimum or first-order-saddle vibrational signature.

    Parameters are vibrational modes only; rigid translations and rotations
    must already have been projected by the shared mass-metric normal-mode
    kernel.
    """

    normalized_target = normalize_stationary_point_target(target)
    threshold = _positive_threshold(imaginary_threshold_cm1)
    frequencies = np.asarray(vibrational_frequencies_cm1, dtype=float)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)):
        raise ValueError(
            "vibrational_frequencies_cm1 must be a finite one-dimensional array."
        )

    reinterpreted = np.asarray(
        reinterpreted_negative_frequencies_cm1,
        dtype=float,
    )
    if (
        reinterpreted.ndim != 1
        or not np.all(np.isfinite(reinterpreted))
        or np.any(reinterpreted >= 0.0)
    ):
        raise ValueError(
            "reinterpreted_negative_frequencies_cm1 must contain only finite "
            "negative values."
        )
    if reinterpretation_threshold_cm1 is None:
        reinterpretation_threshold = None
        if reinterpreted.size:
            raise ValueError(
                "reinterpretation_threshold_cm1 is required when negative modes "
                "were reinterpreted."
            )
    else:
        reinterpretation_threshold = _positive_threshold(reinterpretation_threshold_cm1)
        if normalized_target != MINIMUM:
            raise ValueError(
                "negative-mode reinterpretation is admitted only for a minimum."
            )

    robust = tuple(
        int(index) for index in np.flatnonzero(frequencies <= -threshold).tolist()
    )
    ambiguous = tuple(
        int(index)
        for index in np.flatnonzero(
            (frequencies > -threshold) & (frequencies <= 0.0)
        ).tolist()
    )

    if normalized_target == MINIMUM:
        nonpositive = tuple(sorted(robust + ambiguous))
        if nonpositive:
            rendered = _mode_list(frequencies, nonpositive)
            raise ValueError(
                "A minimum requires all vibrational frequencies to be strictly "
                f"positive; found non-positive mode(s) at {rendered} cm^-1."
            )
    else:
        if ambiguous:
            rendered = _mode_list(frequencies, ambiguous)
            raise ValueError(
                "A first-order transition state has ambiguous non-positive "
                f"vibrational mode(s) at {rendered} cm^-1; each negative mode "
                f"must either pass the robust <= -{threshold:g} cm^-1 admission "
                "guard or be resolved by tighter geometry/Hessian settings."
            )
        if len(robust) != 1:
            raise ValueError(
                "A first-order transition state requires exactly one robust "
                f"imaginary vibrational mode at or below -{threshold:g} cm^-1; "
                f"found {len(robust)}."
            )

    return StationaryPointAssessment(
        target=normalized_target,
        vibrational_frequencies_cm1=tuple(float(value) for value in frequencies),
        imaginary_threshold_cm1=threshold,
        robust_imaginary_mode_indices=robust,
        ambiguous_nonpositive_mode_indices=ambiguous,
        reinterpreted_negative_frequencies_cm1=tuple(
            float(value) for value in reinterpreted
        ),
        reinterpretation_threshold_cm1=reinterpretation_threshold,
    )


__all__ = [
    "MINIMUM",
    "STATIONARY_POINT_TARGETS",
    "TRANSITION_STATE",
    "StationaryPointAssessment",
    "assess_stationary_point",
    "normalize_stationary_point_target",
]
