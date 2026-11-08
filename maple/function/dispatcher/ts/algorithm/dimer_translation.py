"""Fixed-scale Pure Torch V4 Dimer translation primitives.

The V4 translation scale ``beta`` has units of Angstrom squared per Hartree.
Multiplying it by the effective Dimer force (Hartree per Angstrom) gives a
Cartesian displacement in Angstrom.  The geometric cap is applied afterward
with component infinity-norm clipping; it does not change ``beta``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

TRANSLATION_POLICY = "v4_fixed_force_scale_evaluated_translation_v1"
BETA_UNIT = "A2/Ha"


def validate_translation_controls(
    *, beta: float, step_max: float, trust_radius: float, max_iter: int
) -> None:
    """Validate V4 dimensional controls before any physical evaluation."""

    for name, value in (
        ("step0", beta),
        ("step_max", step_max),
        ("trust_radius", trust_radius),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a finite positive scalar.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite positive scalar.") from exc
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} must be a finite positive scalar.")
    if type(max_iter) is not int:
        raise TypeError("max_iter must be a non-negative integer.")
    if max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer.")


def effective_translation_force(
    force: np.ndarray,
    direction: np.ndarray,
    *,
    curvature: float,
    kappa_to_flip: float,
) -> np.ndarray:
    """Return the Dimer force, flipping the mode component at negative curvature."""

    parallel = float(np.dot(force, direction)) * direction
    perpendicular = force - parallel
    if curvature < kappa_to_flip:
        return perpendicular - parallel
    return perpendicular


@dataclass(frozen=True, slots=True)
class TranslationProposal:
    """One finite fixed-beta Cartesian proposal before physical evaluation."""

    raw_step_angstrom: np.ndarray
    applied_step_angstrom: np.ndarray
    proposed_positions_angstrom: np.ndarray
    cap_angstrom: float
    clipped: bool


def propose_translation(
    current_positions_angstrom: np.ndarray,
    translation_force_Ha_per_A: np.ndarray,
    *,
    beta_A2_per_Ha: float,
    cap_angstrom: float,
) -> TranslationProposal:
    """Build one proposal using fixed beta and component infinity-norm clipping."""

    with np.errstate(over="ignore", invalid="ignore"):
        raw = beta_A2_per_Ha * np.asarray(
            translation_force_Ha_per_A, dtype=np.float64
        ).reshape(-1)
    if not np.isfinite(raw).all():
        raise FloatingPointError("Dimer translation raw step is nonfinite.")
    infinity_norm = float(np.max(np.abs(raw), initial=0.0))
    clipped = infinity_norm > cap_angstrom
    applied = np.array(raw, copy=True)
    if clipped:
        applied *= cap_angstrom / infinity_norm
    current = np.asarray(current_positions_angstrom, dtype=np.float64).reshape(-1)
    proposed = current + applied
    if not np.isfinite(applied).all() or not np.isfinite(proposed).all():
        raise FloatingPointError("Dimer translation proposal is nonfinite.")
    return TranslationProposal(
        raw_step_angstrom=_readonly(raw),
        applied_step_angstrom=_readonly(applied),
        proposed_positions_angstrom=_readonly(proposed),
        cap_angstrom=float(cap_angstrom),
        clipped=bool(clipped),
    )


def full_force_norm(force_Ha_per_A: np.ndarray) -> float:
    """Return the Euclidean norm of the complete Cartesian force vector."""

    return float(np.linalg.norm(np.asarray(force_Ha_per_A, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class DimerTranslationAttempt:
    """Immutable receipt for exactly one V4 translation proposal."""

    ordinal: int
    stage: str
    current_rotation: Mapping[str, Any]
    current_positions_angstrom: np.ndarray
    proposed_positions_angstrom: np.ndarray | None
    translation_force_Ha_per_A: np.ndarray | None
    beta_A2_per_Ha: float
    raw_step_angstrom: np.ndarray | None
    applied_step_angstrom: np.ndarray | None
    evaluated_displacement_angstrom: np.ndarray | None
    cap_angstrom: float
    clipped: bool | None
    current_force_norm_Ha_per_A: float
    trial_force_norm_Ha_per_A: float | None
    trial_rotation: Mapping[str, Any] | None
    accepted: bool
    reason: str
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "current_positions_angstrom",
            "proposed_positions_angstrom",
            "translation_force_Ha_per_A",
            "raw_step_angstrom",
            "applied_step_angstrom",
            "evaluated_displacement_angstrom",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "current_rotation", _freeze(self.current_rotation))
        if self.trial_rotation is not None:
            object.__setattr__(self, "trial_rotation", _freeze(self.trial_rotation))

    def to_record(self) -> dict[str, Any]:
        """Return a finite JSON-compatible record; unavailable arrays remain null."""

        return {
            "ordinal": self.ordinal,
            "stage": self.stage,
            "current_rotation": _thaw(self.current_rotation),
            "current_positions_angstrom": _optional_list(
                self.current_positions_angstrom
            ),
            "proposed_positions_angstrom": _optional_list(
                self.proposed_positions_angstrom
            ),
            "translation_force_Ha_per_A": _optional_list(
                self.translation_force_Ha_per_A
            ),
            "beta_A2_per_Ha": self.beta_A2_per_Ha,
            "beta_unit": BETA_UNIT,
            "raw_step_angstrom": _optional_list(self.raw_step_angstrom),
            "applied_step_angstrom": _optional_list(self.applied_step_angstrom),
            "evaluated_displacement_angstrom": _optional_list(
                self.evaluated_displacement_angstrom
            ),
            "cap_angstrom": self.cap_angstrom,
            "clipped": self.clipped,
            "current_force_norm_Ha_per_A": self.current_force_norm_Ha_per_A,
            "trial_force_norm_Ha_per_A": self.trial_force_norm_Ha_per_A,
            "trial_rotation": (
                None if self.trial_rotation is None else _thaw(self.trial_rotation)
            ),
            "accepted": self.accepted,
            "reason": self.reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True).reshape(-1)
    result.setflags(write=False)
    return result


def _optional_list(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        return None
    return array.tolist()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
