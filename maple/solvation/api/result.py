"""Immutable structured result for one registered Route-2 scalar evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from .capabilities import CapabilityStatus
from .profiles import get_solvation_profile
from .provenance import ProvenanceBundle
from .units import ASE_PUBLIC_UNITS, UnitContract


# Versioned and intentionally not constructor-configurable.  Legacy adapters may
# use this bound when checking externally assembled totals before creating leaves.
_RESULT_CLOSURE_TOLERANCE_EV_V1 = 1.0e-10
if not 0.0 < _RESULT_CLOSURE_TOLERANCE_EV_V1 <= 1.0e-8:  # pragma: no cover
    raise RuntimeError("Route-2 result closure tolerance is outside its audited bound.")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real finite number.")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real finite number.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite.")
    return converted


def _vector3_rows(values: Iterable[Iterable[float]], label: str) -> tuple[tuple[float, float, float], ...]:
    rows = []
    for index, row in enumerate(values):
        converted = tuple(_finite(value, f"{label}[{index}]") for value in row)
        if len(converted) != 3:
            raise ValueError(f"{label}[{index}] must have shape (3,), got ({len(converted)},).")
        rows.append(converted)
    return tuple(rows)  # type: ignore[return-value]


def _artifact_ids(values: Iterable[str]) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError("evidence_artifact_ids must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError("evidence_artifact_ids entries must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class EnergyComponent:
    name: str
    value_eV: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Energy component name must be non-empty.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "value_eV", _finite(self.value_eV, self.name))


@dataclass(frozen=True, slots=True)
class ForceComponent:
    name: str
    values_eV_per_A: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Force component name must be non-empty.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "values_eV_per_A", _vector3_rows(self.values_eV_per_A, self.name))


@dataclass(frozen=True, slots=True)
class Route2Result:
    """One result whose identity and capabilities come only from PROFILE_REGISTRY."""

    atom_count: int
    profile_id: str
    energy_components: tuple[EnergyComponent, ...]
    provenance: ProvenanceBundle
    primal_residual: float
    adjoint_residual: float | None
    root_identity: str
    root_sha256: str
    force_components: tuple[ForceComponent, ...] = ()
    force_error_estimate_eV_per_A: float | None = None
    evidence_artifact_ids: tuple[str, ...] = ()
    units: UnitContract = ASE_PUBLIC_UNITS
    admitted_domain: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    fail_closed: bool = True
    scalar_id: str = field(init=False)
    state_equation_id: str = field(init=False)
    capabilities: CapabilityStatus = field(init=False)

    def __post_init__(self) -> None:
        if type(self.atom_count) is not int or self.atom_count <= 0:
            raise ValueError("atom_count must be a positive integer.")
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string.")
        profile = get_solvation_profile(self.profile_id.strip())
        object.__setattr__(self, "profile_id", profile.profile_id)
        object.__setattr__(self, "scalar_id", profile.scalar_id)
        object.__setattr__(self, "state_equation_id", profile.state_equation_id)
        object.__setattr__(self, "capabilities", profile.capabilities)

        if not isinstance(self.provenance, ProvenanceBundle):
            raise TypeError("provenance must be a ProvenanceBundle.")
        if self.units != ASE_PUBLIC_UNITS:
            raise ValueError("Public Route-2 results must use the declared ASE unit contract.")
        if type(self.fail_closed) is not bool:
            raise TypeError("fail_closed must be a bool.")
        if not profile.enabled and not self.fail_closed:
            raise ValueError("A disabled profile may only create fail-closed internal evidence.")
        if not self.fail_closed and not profile.capabilities.energy:
            raise ValueError("A public result requires admitted energy capability.")

        evidence = _artifact_ids(self.evidence_artifact_ids)
        if evidence != profile.evidence_artifact_ids:
            raise ValueError("Result evidence_artifact_ids must exactly match the registered profile admission.")
        object.__setattr__(self, "evidence_artifact_ids", evidence)

        primal_residual = _finite(self.primal_residual, "primal_residual")
        if primal_residual < 0:
            raise ValueError("primal_residual must be non-negative.")
        object.__setattr__(self, "primal_residual", primal_residual)
        if self.adjoint_residual is not None:
            adjoint_residual = _finite(
                self.adjoint_residual, "adjoint_residual"
            )
            if adjoint_residual < 0:
                raise ValueError("adjoint_residual must be non-negative.")
            object.__setattr__(self, "adjoint_residual", adjoint_residual)
        if not isinstance(self.root_identity, str) or not self.root_identity.strip():
            raise ValueError("root_identity must be a non-empty string.")
        object.__setattr__(self, "root_identity", self.root_identity.strip())

        components = tuple(self.energy_components)
        if not components or any(not isinstance(item, EnergyComponent) for item in components):
            raise ValueError("energy_components must contain EnergyComponent leaves.")
        if len({item.name for item in components}) != len(components):
            raise ValueError("Energy component names must be unique.")
        object.__setattr__(self, "energy_components", components)

        if not isinstance(self.root_sha256, str):
            raise TypeError("root_sha256 must be a string.")
        digest = self.root_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("root_sha256 must contain exactly 64 hexadecimal digits.")
        object.__setattr__(self, "root_sha256", digest)

        force_components = tuple(self.force_components)
        if any(not isinstance(item, ForceComponent) for item in force_components):
            raise TypeError("force_components must contain ForceComponent leaves.")
        if len({item.name for item in force_components}) != len(force_components):
            raise ValueError("Force component names must be unique.")
        if any(len(item.values_eV_per_A) != self.atom_count for item in force_components):
            raise ValueError("Every force component must have shape (atom_count, 3).")
        if force_components and not profile.capabilities.conservative_force:
            raise ValueError("Force leaves require registered conservative-force admission.")
        force_error = self.force_error_estimate_eV_per_A
        if force_error is not None:
            force_error = _finite(
                force_error, "force_error_estimate_eV_per_A"
            )
            if force_error < 0.0:
                raise ValueError(
                    "force_error_estimate_eV_per_A must be non-negative."
                )
            object.__setattr__(
                self, "force_error_estimate_eV_per_A", force_error
            )
        if force_components and self.adjoint_residual is None and force_error is None:
            raise ValueError(
                "Force leaves require an adjoint residual or a numerical force "
                "error estimate."
            )
        if not force_components and force_error is not None:
            raise ValueError(
                "A force error estimate is invalid when no force leaves are present."
            )
        object.__setattr__(self, "force_components", force_components)

        domain = tuple(sorted(tuple(item) for item in self.admitted_domain))
        if any(len(item) != 2 or not all(isinstance(value, str) and value.strip() for value in item) for item in domain):
            raise ValueError("admitted_domain must contain non-empty string key/value pairs.")
        if len({key for key, _ in domain}) != len(domain):
            raise ValueError("admitted_domain keys must be unique.")
        if domain and not profile.enabled:
            raise ValueError("A disabled profile cannot declare an admitted public domain.")
        object.__setattr__(self, "admitted_domain", domain)
        warnings = tuple(self.warnings)
        if any(not isinstance(item, str) or not item.strip() for item in warnings):
            raise ValueError("warnings must contain non-empty strings.")
        object.__setattr__(self, "warnings", warnings)

    @property
    def total_energy_eV(self) -> float:
        """Total scalar energy computed exclusively from immutable leaves."""

        return math.fsum(component.value_eV for component in self.energy_components)

    @property
    def total_forces_eV_per_A(self) -> tuple[tuple[float, float, float], ...] | None:
        """Total conservative force computed exclusively from immutable leaves."""

        if not self.force_components:
            return None
        return tuple(
            tuple(
                math.fsum(component.values_eV_per_A[atom][axis] for component in self.force_components)
                for axis in range(3)
            )
            for atom in range(self.atom_count)
        )  # type: ignore[return-value]


__all__ = ["EnergyComponent", "ForceComponent", "Route2Result"]
