"""Preregistered far-field and molecular-multipole source diagnostics.

This gate answers one narrow question: whether a failed cavity-near-field
source still preserves the total charge and low-order far-field information
needed to justify one separately named fixed radial-embedding experiment.  A
pass is necessary, not sufficient, for that experiment.  It never admits a
PCM source, an energy ledger, a force, or any public Route-2 capability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .source_mep import SourceElectrostaticObservables

FAR_FIELD_SOURCE_CONTRACT_VERSION = "route2-original-source-far-field-gate-v1"


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True, slots=True)
class FarFieldSourceThresholds:
    total_charge_absolute_tolerance_e: float
    dipole_absolute_tolerance_e_angstrom: float
    dipole_relative_tolerance: float
    quadrupole_absolute_tolerance_e_angstrom2: float
    quadrupole_relative_tolerance: float
    far_field_absolute_rms_tolerance_hartree_per_e: float
    far_field_relative_rms_tolerance: float
    contract_version: str = FAR_FIELD_SOURCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != FAR_FIELD_SOURCE_CONTRACT_VERSION:
            raise ValueError("far-field threshold contract version is unsupported.")
        for name in (
            "total_charge_absolute_tolerance_e",
            "dipole_absolute_tolerance_e_angstrom",
            "dipole_relative_tolerance",
            "quadrupole_absolute_tolerance_e_angstrom2",
            "quadrupole_relative_tolerance",
            "far_field_absolute_rms_tolerance_hartree_per_e",
            "far_field_relative_rms_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FarFieldShellComparison:
    radius_angstrom: float
    point_count: int
    reference_rms_hartree_per_e: float
    error_rms_hartree_per_e: float
    relative_l2_error: float
    allowed_error_rms_hartree_per_e: float
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "radius_angstrom",
            _finite_positive(self.radius_angstrom, name="radius_angstrom"),
        )
        if isinstance(self.point_count, bool) or int(self.point_count) < 1:
            raise ValueError("point_count must be a positive integer.")
        object.__setattr__(self, "point_count", int(self.point_count))
        for name in (
            "reference_rms_hartree_per_e",
            "error_rms_hartree_per_e",
            "relative_l2_error",
            "allowed_error_rms_hartree_per_e",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if type(self.passed) is not bool:
            raise TypeError("passed must be exactly bool.")
        expected = self.error_rms_hartree_per_e <= (
            self.allowed_error_rms_hartree_per_e
        )
        if self.passed is not expected:
            raise ValueError("shell pass flag is inconsistent with its tolerance.")

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FarFieldSourceComparison:
    total_charge_absolute_error_e: float
    total_charge_allowed_error_e: float
    dipole_reference_l2_e_angstrom: float
    dipole_error_l2_e_angstrom: float
    dipole_allowed_error_l2_e_angstrom: float
    dipole_relative_l2_error: float
    quadrupole_reference_frobenius_e_angstrom2: float
    quadrupole_error_frobenius_e_angstrom2: float
    quadrupole_allowed_error_frobenius_e_angstrom2: float
    quadrupole_relative_frobenius_error: float
    shell_comparisons: tuple[FarFieldShellComparison, ...]
    charge_gate_passed: bool
    dipole_gate_passed: bool
    quadrupole_gate_passed: bool
    far_field_mep_gate_passed: bool
    case_passed: bool
    contract_version: str = FAR_FIELD_SOURCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != FAR_FIELD_SOURCE_CONTRACT_VERSION:
            raise ValueError("far-field comparison contract version is unsupported.")
        for name in (
            "total_charge_absolute_error_e",
            "total_charge_allowed_error_e",
            "dipole_reference_l2_e_angstrom",
            "dipole_error_l2_e_angstrom",
            "dipole_allowed_error_l2_e_angstrom",
            "dipole_relative_l2_error",
            "quadrupole_reference_frobenius_e_angstrom2",
            "quadrupole_error_frobenius_e_angstrom2",
            "quadrupole_allowed_error_frobenius_e_angstrom2",
            "quadrupole_relative_frobenius_error",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        shells = tuple(self.shell_comparisons)
        if not shells or not all(
            isinstance(shell, FarFieldShellComparison) for shell in shells
        ):
            raise ValueError("shell_comparisons must contain typed shell results.")
        object.__setattr__(self, "shell_comparisons", shells)
        for name in (
            "charge_gate_passed",
            "dipole_gate_passed",
            "quadrupole_gate_passed",
            "far_field_mep_gate_passed",
            "case_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be exactly bool.")
        if self.charge_gate_passed is not (
            self.total_charge_absolute_error_e <= self.total_charge_allowed_error_e
        ):
            raise ValueError("charge gate flag is inconsistent with its tolerance.")
        if self.dipole_gate_passed is not (
            self.dipole_error_l2_e_angstrom <= self.dipole_allowed_error_l2_e_angstrom
        ):
            raise ValueError("dipole gate flag is inconsistent with its tolerance.")
        if self.quadrupole_gate_passed is not (
            self.quadrupole_error_frobenius_e_angstrom2
            <= self.quadrupole_allowed_error_frobenius_e_angstrom2
        ):
            raise ValueError("quadrupole gate flag is inconsistent with its tolerance.")
        if self.far_field_mep_gate_passed is not all(shell.passed for shell in shells):
            raise ValueError("far-field MEP gate must require every shell.")
        expected_case = bool(
            self.charge_gate_passed
            and self.dipole_gate_passed
            and self.quadrupole_gate_passed
            and self.far_field_mep_gate_passed
        )
        if self.case_passed is not expected_case:
            raise ValueError("case pass must require every far-field sub-gate.")

    def as_dict(self) -> dict[str, object]:
        return {
            field: (
                [shell.as_dict() for shell in self.shell_comparisons]
                if field == "shell_comparisons"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        } | {
            "capability_admitted": False,
            "claim_boundary": (
                "A pass is only a necessary precondition for one separately "
                "named fixed radial-embedding research experiment; it does not "
                "prove that the cavity-near-field defect is radial-only."
            ),
        }


def _relative_error(error_norm: float, reference_norm: float) -> float:
    return float(error_norm / max(reference_norm, np.finfo(float).tiny))


def compare_far_field_source_to_reference(
    predicted: SourceElectrostaticObservables,
    reference: SourceElectrostaticObservables,
    *,
    shell_radii_angstrom: object,
    shell_point_counts: object,
    angular_weights: object,
    thresholds: FarFieldSourceThresholds,
) -> FarFieldSourceComparison:
    """Compare matched low multipoles and deterministic far-field shells."""

    if not isinstance(predicted, SourceElectrostaticObservables) or not isinstance(
        reference, SourceElectrostaticObservables
    ):
        raise TypeError("predicted and reference must be source observables.")
    if not isinstance(thresholds, FarFieldSourceThresholds):
        raise TypeError("thresholds must be FarFieldSourceThresholds.")
    if predicted.multipole_origin_angstrom != reference.multipole_origin_angstrom:
        raise ValueError("predicted and reference multipoles must use one origin.")

    predicted_mep = np.asarray(predicted.sampled_mep_hartree_per_e, dtype=float)
    reference_mep = np.asarray(reference.sampled_mep_hartree_per_e, dtype=float)
    counts = np.asarray(shell_point_counts, dtype=int)
    radii = np.asarray(shell_radii_angstrom, dtype=float)
    weights = np.asarray(angular_weights, dtype=float)
    if (
        predicted_mep.ndim != 1
        or reference_mep.shape != predicted_mep.shape
        or counts.ndim != 1
        or counts.size < 1
        or radii.shape != counts.shape
        or np.any(counts < 1)
        or int(np.sum(counts)) != predicted_mep.size
        or weights.shape != predicted_mep.shape
        or not np.all(np.isfinite(predicted_mep))
        or not np.all(np.isfinite(reference_mep))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("far-field shells, values, and weights are inconsistent.")

    dipole_prediction = np.asarray(predicted.molecular_dipole_e_angstrom)
    dipole_reference = np.asarray(reference.molecular_dipole_e_angstrom)
    dipole_reference_norm = float(np.linalg.norm(dipole_reference))
    dipole_error_norm = float(np.linalg.norm(dipole_prediction - dipole_reference))
    dipole_limit = float(
        thresholds.dipole_absolute_tolerance_e_angstrom
        + thresholds.dipole_relative_tolerance * dipole_reference_norm
    )

    quadrupole_prediction = np.asarray(predicted.traceless_quadrupole_e_angstrom2)
    quadrupole_reference = np.asarray(reference.traceless_quadrupole_e_angstrom2)
    quadrupole_reference_norm = float(np.linalg.norm(quadrupole_reference))
    quadrupole_error_norm = float(
        np.linalg.norm(quadrupole_prediction - quadrupole_reference)
    )
    quadrupole_limit = float(
        thresholds.quadrupole_absolute_tolerance_e_angstrom2
        + thresholds.quadrupole_relative_tolerance * quadrupole_reference_norm
    )

    shell_results: list[FarFieldShellComparison] = []
    start = 0
    for radius, count in zip(radii, counts, strict=True):
        stop = start + int(count)
        shell_weights = weights[start:stop]
        difference = predicted_mep[start:stop] - reference_mep[start:stop]
        reference_values = reference_mep[start:stop]
        normalization = float(np.sum(shell_weights))
        reference_rms = float(
            np.sqrt(
                np.dot(shell_weights, reference_values * reference_values)
                / normalization
            )
        )
        error_rms = float(
            np.sqrt(np.dot(shell_weights, difference * difference) / normalization)
        )
        allowed = float(
            thresholds.far_field_absolute_rms_tolerance_hartree_per_e
            + thresholds.far_field_relative_rms_tolerance * reference_rms
        )
        shell_results.append(
            FarFieldShellComparison(
                radius_angstrom=float(radius),
                point_count=int(count),
                reference_rms_hartree_per_e=reference_rms,
                error_rms_hartree_per_e=error_rms,
                relative_l2_error=_relative_error(error_rms, reference_rms),
                allowed_error_rms_hartree_per_e=allowed,
                passed=bool(error_rms <= allowed),
            )
        )
        start = stop

    charge_error = abs(predicted.total_charge_e - reference.total_charge_e)
    charge_passed = bool(charge_error <= thresholds.total_charge_absolute_tolerance_e)
    dipole_passed = bool(dipole_error_norm <= dipole_limit)
    quadrupole_passed = bool(quadrupole_error_norm <= quadrupole_limit)
    far_field_passed = bool(all(shell.passed for shell in shell_results))
    return FarFieldSourceComparison(
        total_charge_absolute_error_e=charge_error,
        total_charge_allowed_error_e=(thresholds.total_charge_absolute_tolerance_e),
        dipole_reference_l2_e_angstrom=dipole_reference_norm,
        dipole_error_l2_e_angstrom=dipole_error_norm,
        dipole_allowed_error_l2_e_angstrom=dipole_limit,
        dipole_relative_l2_error=_relative_error(
            dipole_error_norm, dipole_reference_norm
        ),
        quadrupole_reference_frobenius_e_angstrom2=quadrupole_reference_norm,
        quadrupole_error_frobenius_e_angstrom2=quadrupole_error_norm,
        quadrupole_allowed_error_frobenius_e_angstrom2=quadrupole_limit,
        quadrupole_relative_frobenius_error=_relative_error(
            quadrupole_error_norm, quadrupole_reference_norm
        ),
        shell_comparisons=tuple(shell_results),
        charge_gate_passed=charge_passed,
        dipole_gate_passed=dipole_passed,
        quadrupole_gate_passed=quadrupole_passed,
        far_field_mep_gate_passed=far_field_passed,
        case_passed=bool(
            charge_passed and dipole_passed and quadrupole_passed and far_field_passed
        ),
    )


__all__ = [
    "FAR_FIELD_SOURCE_CONTRACT_VERSION",
    "FarFieldShellComparison",
    "FarFieldSourceComparison",
    "FarFieldSourceThresholds",
    "compare_far_field_source_to_reference",
]
