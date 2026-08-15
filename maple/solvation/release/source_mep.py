"""Reference-bound physical diagnostics for an original four-channel source.

These diagnostics separate source physics from continuum and energy-ledger
errors.  They never infer a QM reference and never admit a capability when a
reference observable is absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
    gaussian_multipole_potential,
)

SOURCE_MEP_DIAGNOSTIC_CONTRACT_VERSION = "route2-source-mep-physical-gate-v1"


def _positions(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] != 3
        or not np.all(np.isfinite(result))
    ):
        raise ValueError("positions must be finite with shape (N,3).")
    return np.array(result, copy=True)


def _source(values: object, atom_count: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (atom_count, 4) or not np.all(np.isfinite(result)):
        raise ValueError("source must be finite with shape (N,4).")
    return np.array(result, copy=True)


def _relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(predicted - reference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


@dataclass(frozen=True, slots=True)
class SourceElectrostaticObservables:
    total_charge_e: float
    molecular_dipole_e_angstrom: tuple[float, float, float]
    traceless_quadrupole_e_angstrom2: tuple[tuple[float, float, float], ...]
    sampled_mep_hartree_per_e: tuple[float, ...]
    contract_version: str = SOURCE_MEP_DIAGNOSTIC_CONTRACT_VERSION

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.total_charge_e,
                *self.molecular_dipole_e_angstrom,
                *np.asarray(self.traceless_quadrupole_e_angstrom2).reshape(-1),
                *self.sampled_mep_hartree_per_e,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("source electrostatic observables must be finite.")

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "total_charge_e": self.total_charge_e,
            "molecular_dipole_e_angstrom": list(self.molecular_dipole_e_angstrom),
            "traceless_quadrupole_e_angstrom2": [
                list(row) for row in self.traceless_quadrupole_e_angstrom2
            ],
            "sampled_mep_hartree_per_e": list(self.sampled_mep_hartree_per_e),
        }


@dataclass(frozen=True, slots=True)
class SourceMEPComparison:
    total_charge_absolute_error_e: float
    dipole_relative_l2_error: float
    quadrupole_relative_frobenius_error: float
    sampled_mep_relative_l2_error: float
    fixed_source_pcm_energy_absolute_error_eV: float
    predicted_fixed_source_pcm_energy_eV: float
    reference_fixed_source_pcm_energy_eV: float
    reference_identity: str
    contract_version: str = SOURCE_MEP_DIAGNOSTIC_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference_identity, str)
            or not self.reference_identity.strip()
        ):
            raise ValueError("reference_identity must be non-empty.")
        values = np.asarray(
            (
                self.total_charge_absolute_error_e,
                self.dipole_relative_l2_error,
                self.quadrupole_relative_frobenius_error,
                self.sampled_mep_relative_l2_error,
                self.fixed_source_pcm_energy_absolute_error_eV,
                self.predicted_fixed_source_pcm_energy_eV,
                self.reference_fixed_source_pcm_energy_eV,
            )
        )
        if not np.all(np.isfinite(values)) or np.any(values[:5] < 0.0):
            raise ValueError(
                "source/MEP comparison values must be finite and non-negative."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "reference_identity": self.reference_identity,
            "total_charge_absolute_error_e": self.total_charge_absolute_error_e,
            "dipole_relative_l2_error": self.dipole_relative_l2_error,
            "quadrupole_relative_frobenius_error": (
                self.quadrupole_relative_frobenius_error
            ),
            "sampled_mep_relative_l2_error": self.sampled_mep_relative_l2_error,
            "fixed_source_pcm_energy_absolute_error_eV": (
                self.fixed_source_pcm_energy_absolute_error_eV
            ),
            "predicted_fixed_source_pcm_energy_eV": (
                self.predicted_fixed_source_pcm_energy_eV
            ),
            "reference_fixed_source_pcm_energy_eV": (
                self.reference_fixed_source_pcm_energy_eV
            ),
            "capability_admitted": False,
        }


def source_electrostatic_observables(
    *,
    positions_angstrom: object,
    source4: object,
    evaluation_points_bohr: object,
    sigma_angstrom: float = 1.5,
) -> SourceElectrostaticObservables:
    """Compute charge, molecular multipoles, and a sampled Gaussian MEP."""

    positions = _positions(positions_angstrom)
    source = _source(source4, len(positions))
    points = np.asarray(evaluation_points_bohr, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] < 1
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("evaluation_points_bohr must be finite with shape (M,3).")
    if not np.isfinite(sigma_angstrom) or sigma_angstrom <= 0.0:
        raise ValueError("sigma_angstrom must be finite and positive.")
    charges, atomic_dipoles = cartesian_multipoles(source)
    molecular_dipole = np.sum(charges[:, None] * positions + atomic_dipoles, axis=0)
    identity = np.eye(3)
    quadrupole = np.zeros((3, 3))
    for position, charge, dipole in zip(
        positions, charges, atomic_dipoles, strict=True
    ):
        radius2 = float(np.vdot(position, position))
        quadrupole += charge * (3.0 * np.outer(position, position) - radius2 * identity)
        quadrupole += (
            3.0 * (np.outer(position, dipole) + np.outer(dipole, position))
            - 2.0 * float(np.vdot(position, dipole)) * identity
        )
    mep = gaussian_multipole_potential(
        points,
        positions,
        source,
        sigma_angstrom=float(sigma_angstrom),
    )
    return SourceElectrostaticObservables(
        total_charge_e=float(np.sum(charges)),
        molecular_dipole_e_angstrom=tuple(float(value) for value in molecular_dipole),
        traceless_quadrupole_e_angstrom2=tuple(
            tuple(float(value) for value in row) for row in quadrupole
        ),
        sampled_mep_hartree_per_e=tuple(float(value) for value in mep),
    )


def compare_source_mep_to_reference(
    predicted: SourceElectrostaticObservables,
    reference: SourceElectrostaticObservables,
    *,
    predicted_fixed_source_pcm_energy_eV: float,
    reference_fixed_source_pcm_energy_eV: float,
    reference_identity: str,
) -> SourceMEPComparison:
    """Compare matched observables without mixing cavities or energy components."""

    if not isinstance(predicted, SourceElectrostaticObservables) or not isinstance(
        reference, SourceElectrostaticObservables
    ):
        raise TypeError("predicted and reference must be source observables.")
    if len(predicted.sampled_mep_hartree_per_e) != len(
        reference.sampled_mep_hartree_per_e
    ):
        raise ValueError(
            "predicted and reference MEP samples must use identical points."
        )
    predicted_energy = float(predicted_fixed_source_pcm_energy_eV)
    reference_energy = float(reference_fixed_source_pcm_energy_eV)
    if not np.isfinite(predicted_energy) or not np.isfinite(reference_energy):
        raise ValueError("matched fixed-source PCM energies must be finite.")
    predicted_dipole = np.asarray(predicted.molecular_dipole_e_angstrom)
    reference_dipole = np.asarray(reference.molecular_dipole_e_angstrom)
    predicted_quadrupole = np.asarray(predicted.traceless_quadrupole_e_angstrom2)
    reference_quadrupole = np.asarray(reference.traceless_quadrupole_e_angstrom2)
    predicted_mep = np.asarray(predicted.sampled_mep_hartree_per_e)
    reference_mep = np.asarray(reference.sampled_mep_hartree_per_e)
    return SourceMEPComparison(
        total_charge_absolute_error_e=abs(
            predicted.total_charge_e - reference.total_charge_e
        ),
        dipole_relative_l2_error=_relative_l2(predicted_dipole, reference_dipole),
        quadrupole_relative_frobenius_error=_relative_l2(
            predicted_quadrupole, reference_quadrupole
        ),
        sampled_mep_relative_l2_error=_relative_l2(predicted_mep, reference_mep),
        fixed_source_pcm_energy_absolute_error_eV=abs(
            predicted_energy - reference_energy
        ),
        predicted_fixed_source_pcm_energy_eV=predicted_energy,
        reference_fixed_source_pcm_energy_eV=reference_energy,
        reference_identity=reference_identity,
    )


__all__ = [
    "SOURCE_MEP_DIAGNOSTIC_CONTRACT_VERSION",
    "SourceElectrostaticObservables",
    "SourceMEPComparison",
    "compare_source_mep_to_reference",
    "source_electrostatic_observables",
]
