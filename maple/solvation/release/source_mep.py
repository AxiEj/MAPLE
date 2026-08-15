"""Reference-bound physical diagnostics for an original four-channel source.

These diagnostics separate source physics from continuum and energy-ledger
errors.  They never infer a QM reference and never admit a capability when a
reference observable is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from maple.function.calculator.extra_correction.implicit.gto_density import (
    cartesian_multipoles,
    gaussian_multipole_potential,
)

SOURCE_MEP_DIAGNOSTIC_CONTRACT_VERSION = "route2-source-mep-physical-gate-v1"
CONTINUUM_ACTIVE_SOURCE_CONTRACT_VERSION = (
    "route2-continuum-active-source-a-inverse-gate-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return result


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


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


@dataclass(frozen=True, slots=True)
class ContinuumActiveSourceComparison:
    """One matched QM/source case in the continuum energy metric.

    The reference is a boundary right-hand side projected directly from the
    matched QM electrostatic potential.  It is not reconstructed from another
    atom-charge fit.  A case pass is only a diagnostic datum; profile or public
    capability admission remains a separate aggregate decision.
    """

    reference_identity: str
    reference_artifact_sha256: str
    geometry_sha256: str
    continuum_configuration_sha256: str
    continuum_provenance_sha256: str
    topology_sha256: str
    cavity_profile_id: str
    predicted_boundary_rhs_sha256: str
    reference_boundary_rhs_sha256: str
    boundary_rhs_relative_l2_error: float
    reference_a_inverse_norm_sqrt_eV: float
    a_inverse_rhs_distance_sqrt_eV: float
    maximum_allowed_a_inverse_distance_sqrt_eV: float
    fixed_source_energy_error_budget_eV: float
    predicted_fixed_source_energy_eV: float
    reference_fixed_source_energy_eV: float
    fixed_source_energy_absolute_error_eV: float
    fixed_source_energy_error_upper_bound_eV: float
    case_passed: bool
    contract_version: str = CONTINUUM_ACTIVE_SOURCE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("reference_identity", "cavity_profile_id", "contract_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "reference_artifact_sha256",
            "geometry_sha256",
            "continuum_configuration_sha256",
            "continuum_provenance_sha256",
            "topology_sha256",
            "predicted_boundary_rhs_sha256",
            "reference_boundary_rhs_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        nonnegative = (
            "boundary_rhs_relative_l2_error",
            "reference_a_inverse_norm_sqrt_eV",
            "a_inverse_rhs_distance_sqrt_eV",
            "maximum_allowed_a_inverse_distance_sqrt_eV",
            "fixed_source_energy_error_budget_eV",
            "fixed_source_energy_absolute_error_eV",
            "fixed_source_energy_error_upper_bound_eV",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in (
            "predicted_fixed_source_energy_eV",
            "reference_fixed_source_energy_eV",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.fixed_source_energy_error_budget_eV <= 0.0:
            raise ValueError("fixed-source energy error budget must be positive.")
        if type(self.case_passed) is not bool:
            raise TypeError("case_passed must be exactly bool.")
        if (
            self.fixed_source_energy_absolute_error_eV
            > self.fixed_source_energy_error_upper_bound_eV + 2.0e-12
        ):
            raise ValueError(
                "reported A-inverse energy bound does not bound the error."
            )

    def as_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__} | {
            "capability_admitted": False,
            "claim_boundary": (
                "A case pass is matched electrostatic source evidence only; it "
                "does not select an operational ledger or admit E/F/H/V/M."
            ),
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


def compare_continuum_active_source_to_reference(
    *,
    continuum: object,
    predicted_source: object,
    reference_boundary_rhs: object,
    reference_identity: str,
    reference_artifact_sha256: str,
    fixed_source_energy_error_budget_eV: float,
) -> ContinuumActiveSourceComparison:
    """Compare ``B c`` with a directly projected QM boundary RHS.

    For the registered separated harmonic continuum, ``A`` is symmetric
    positive definite and ``G(b)=-1/2 b.T A^-1 b``.  Consequently, with
    ``e=b_pred-b_ref``, ``x=||b_ref||_(A^-1)``, and ``d=||e||_(A^-1)``,

    ``|G_pred-G_ref| <= x d + 1/2 d^2``.

    The accepted distance is derived from the caller's preregistered energy
    budget rather than from an arbitrary coefficient-space RMSE threshold.
    """

    source_rhs = getattr(continuum, "source_rhs", None)
    if not callable(source_rhs):
        raise TypeError("continuum must expose source_rhs(predicted_source).")
    predicted_rhs = np.asarray(source_rhs(predicted_source), dtype=float)
    reference_rhs = np.asarray(reference_boundary_rhs, dtype=float)
    if (
        predicted_rhs.ndim != 1
        or reference_rhs.shape != predicted_rhs.shape
        or predicted_rhs.size < 1
        or not np.all(np.isfinite(predicted_rhs))
        or not np.all(np.isfinite(reference_rhs))
    ):
        raise ValueError(
            "predicted and reference boundary RHS values must be finite equal vectors."
        )
    surface = np.asarray(getattr(continuum, "surface_operator", None), dtype=float)
    if surface.shape != (predicted_rhs.size, predicted_rhs.size) or not np.all(
        np.isfinite(surface)
    ):
        raise ValueError("continuum surface operator must be finite and RHS-sized.")
    scale = max(1.0, float(np.linalg.norm(surface, ord="fro")))
    if not np.allclose(surface, surface.T, rtol=0.0, atol=2.0e-12 * scale):
        raise ValueError("A-inverse source gate requires a symmetric surface operator.")
    eigenvalues = np.linalg.eigvalsh(0.5 * (surface + surface.T))
    if eigenvalues[0] <= 1.0e-12 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError(
            "A-inverse source gate requires a strictly positive-definite operator."
        )
    budget = float(fixed_source_energy_error_budget_eV)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError(
            "fixed-source energy error budget must be positive and finite."
        )

    reference_solution = np.linalg.solve(surface, reference_rhs)
    predicted_solution = np.linalg.solve(surface, predicted_rhs)
    error = predicted_rhs - reference_rhs
    error_solution = np.linalg.solve(surface, error)
    reference_norm = float(
        np.sqrt(max(0.0, np.vdot(reference_rhs, reference_solution)))
    )
    distance = float(np.sqrt(max(0.0, np.vdot(error, error_solution))))
    upper_bound = reference_norm * distance + 0.5 * distance * distance
    # Stable evaluation of sqrt(x^2 + 2 eps) - x for eps << x^2.
    maximum_distance = (2.0 * budget) / (
        np.sqrt(reference_norm * reference_norm + 2.0 * budget) + reference_norm
    )
    predicted_energy = -0.5 * float(np.vdot(predicted_rhs, predicted_solution))
    reference_energy = -0.5 * float(np.vdot(reference_rhs, reference_solution))
    energy_error = abs(predicted_energy - reference_energy)
    relative_rhs = _relative_l2(predicted_rhs, reference_rhs)
    case_passed = bool(
        distance <= maximum_distance + 2.0e-15 and energy_error <= budget + 2.0e-12
    )

    return ContinuumActiveSourceComparison(
        reference_identity=reference_identity,
        reference_artifact_sha256=reference_artifact_sha256,
        geometry_sha256=_digest(
            getattr(continuum, "geometry_sha256", None),
            name="continuum.geometry_sha256",
        ),
        continuum_configuration_sha256=_digest(
            getattr(continuum, "continuum_configuration_sha256", None),
            name="continuum.continuum_configuration_sha256",
        ),
        continuum_provenance_sha256=_digest(
            getattr(continuum, "provenance_sha256", None),
            name="continuum.provenance_sha256",
        ),
        topology_sha256=_digest(
            getattr(continuum, "topology_sha256", None),
            name="continuum.topology_sha256",
        ),
        cavity_profile_id=_text(
            getattr(continuum, "cavity_profile_id", None),
            name="continuum.cavity_profile_id",
        ),
        predicted_boundary_rhs_sha256=_array_sha256(predicted_rhs),
        reference_boundary_rhs_sha256=_array_sha256(reference_rhs),
        boundary_rhs_relative_l2_error=relative_rhs,
        reference_a_inverse_norm_sqrt_eV=reference_norm,
        a_inverse_rhs_distance_sqrt_eV=distance,
        maximum_allowed_a_inverse_distance_sqrt_eV=float(maximum_distance),
        fixed_source_energy_error_budget_eV=budget,
        predicted_fixed_source_energy_eV=predicted_energy,
        reference_fixed_source_energy_eV=reference_energy,
        fixed_source_energy_absolute_error_eV=energy_error,
        fixed_source_energy_error_upper_bound_eV=upper_bound,
        case_passed=case_passed,
    )


__all__ = [
    "CONTINUUM_ACTIVE_SOURCE_CONTRACT_VERSION",
    "SOURCE_MEP_DIAGNOSTIC_CONTRACT_VERSION",
    "ContinuumActiveSourceComparison",
    "SourceElectrostaticObservables",
    "SourceMEPComparison",
    "compare_continuum_active_source_to_reference",
    "compare_source_mep_to_reference",
    "source_electrostatic_observables",
]
