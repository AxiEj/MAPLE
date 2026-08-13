"""Isolated variable-size fixed-box evaluator for vNext diagnostics.

The historical calculator evaluator remains byte-identical and continues to
own the released 40-A experiment.  This subclass reuses that implementation
without widening its accepted profile set, while giving every additional box
length a distinct immutable identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from maple.function.calculator.mace._macepol_long_range import (
    MACEPolarLongRangeEvaluator,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS,
)

_DIAGNOSTIC_BOX_BY_PROFILE = {
    profile: float(box_length)
    for box_length, profile in MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS.items()
    if box_length != 40
}


@dataclass(frozen=True)
class MACEPolarFixedBoxDiagnosticEvaluator(MACEPolarLongRangeEvaluator):
    """One preregistered non-40-A reciprocal diagnostic operator."""

    def __post_init__(self) -> None:
        try:
            expected_box_length = _DIAGNOSTIC_BOX_BY_PROFILE[self.profile]
        except KeyError as exc:
            raise ValueError(
                "Unsupported vNext fixed-box diagnostic evaluator profile: "
                f"{self.profile}."
            ) from exc
        if self.use_pbc_evaluator is not True or (
            self.box_length_angstrom != expected_box_length
        ):
            raise ValueError(
                "The vNext fixed-box diagnostic evaluator fields are "
                f"inconsistent with profile={self.profile}."
            )

    @classmethod
    def from_profile(cls, profile: str) -> "MACEPolarFixedBoxDiagnosticEvaluator":
        normalized = str(profile).strip().lower()
        try:
            box_length = _DIAGNOSTIC_BOX_BY_PROFILE[normalized]
        except KeyError as exc:
            raise ValueError(
                "Unsupported vNext fixed-box diagnostic evaluator profile: "
                f"{profile}."
            ) from exc
        return cls(normalized, True, box_length)

    @property
    def provenance(self) -> dict[str, object]:
        result = super().provenance
        result["implementation_scope"] = (
            "vNext disabled box-convergence diagnostic; historical 40-A "
            "evaluator contract remains unchanged"
        )
        return result

    def prepare_batch(self, batch, *, r_max: float):
        try:
            return super().prepare_batch(batch, r_max=r_max)
        except RuntimeError as exc:
            if "validated 40 A fixed box" not in str(exc):
                raise
            raise RuntimeError(
                "The molecule plus the MACE cutoff does not fit inside the "
                f"preregistered {self.box_length_angstrom:g} A fixed box."
            ) from exc


__all__ = ["MACEPolarFixedBoxDiagnosticEvaluator"]
