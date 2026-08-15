"""Two explicit, disabled operational energy ledgers for the separated root.

The fixed point selects a state; it does not uniquely select an energy.  These
classes therefore keep the current vacuum-plus-continuum ledger (Phi0) and the
field-conditioned intrinsic-energy ledger (Phi1) as different scalar IDs.
Neither class asserts that the original checkpoint source is an energy
gradient, and neither enables a public capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    get_scalar_definition,
)
from maple.solvation.api.state_registry import (
    SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
)

from .separated_state import SeparatedOperationalStateEquation
from .state_equation import geometry_sha256, provider_behavior_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class SeparatedVacuumScalarProvider(Protocol):
    provider_id: str
    model_profile_id: str
    provenance_sha256: str

    def configuration_sha256(self) -> str: ...

    def evaluate_energy(self, geometry: Any) -> float: ...


@dataclass(frozen=True, slots=True)
class OperationalLedgerEvaluation:
    scalar_id: str
    state_equation_id: str
    geometry_sha256: str
    equation_sha256: str
    ledger_sha256: str
    reduced_coordinates_sha256: str
    root_residual_norm: float
    root_tolerance: float
    components_eV: tuple[tuple[str, float], ...]
    total_energy_eV: float

    def __post_init__(self) -> None:
        for name in (
            "scalar_id",
            "state_equation_id",
            "geometry_sha256",
            "equation_sha256",
            "ledger_sha256",
            "reduced_coordinates_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "geometry_sha256",
            "equation_sha256",
            "ledger_sha256",
            "reduced_coordinates_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        if not self.components_eV:
            raise ValueError("components_eV must be non-empty.")
        if len({name for name, _ in self.components_eV}) != len(self.components_eV):
            raise ValueError("ledger component names must be unique.")
        if any(
            not isinstance(name, str) or not name.strip() or not np.isfinite(value)
            for name, value in self.components_eV
        ):
            raise ValueError("ledger components must have names and finite values.")
        if not np.isfinite(self.total_energy_eV):
            raise ValueError("total_energy_eV must be finite.")
        if not np.isclose(
            self.total_energy_eV,
            sum(value for _, value in self.components_eV),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("ledger total does not close its immutable components.")
        if (
            not np.isfinite(self.root_residual_norm)
            or self.root_residual_norm < 0.0
            or not np.isfinite(self.root_tolerance)
            or self.root_tolerance <= 0.0
            or self.root_residual_norm > self.root_tolerance
        ):
            raise ValueError("ledger evaluation requires a converged root residual.")

    def as_dict(self) -> dict[str, object]:
        return {
            "scalar_id": self.scalar_id,
            "state_equation_id": self.state_equation_id,
            "geometry_sha256": self.geometry_sha256,
            "equation_sha256": self.equation_sha256,
            "ledger_sha256": self.ledger_sha256,
            "reduced_coordinates_sha256": self.reduced_coordinates_sha256,
            "root_residual_norm": self.root_residual_norm,
            "root_tolerance": self.root_tolerance,
            "components_eV": dict(self.components_eV),
            "total_energy_eV": self.total_energy_eV,
            "capabilities": "none",
        }


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


class _SeparatedLedgerBase:
    __slots__ = (
        "_configuration_sha256",
        "_sealed",
        "equation",
        "scalar_id",
        "vacuum",
    )

    implementation_entry_point = ""

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
        vacuum: SeparatedVacuumScalarProvider | None,
        scalar_id: str,
    ) -> None:
        if not isinstance(equation, SeparatedOperationalStateEquation):
            raise TypeError("equation must be SeparatedOperationalStateEquation.")
        definition = get_scalar_definition(scalar_id)
        if definition.state_equation_id != SEPARATED_OPERATIONAL_STATE_EQUATION_ID:
            raise ValueError("scalar is not bound to the separated state equation.")
        if definition.implementation_entry_point != self.implementation_entry_point:
            raise ValueError("scalar registry entry point does not match the ledger.")
        if equation.continuum.scalar_id != scalar_id:
            raise ValueError("continuum snapshot is bound to a different scalar ID.")
        if definition.enabled or definition.admitted_capabilities.enabled_tiers:
            raise ValueError("separated operational ledgers must remain disabled.")
        if vacuum is not None:
            for name in ("provider_id", "model_profile_id", "provenance_sha256"):
                value = getattr(vacuum, name, None)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"vacuum.{name} must be a non-empty identity.")
            configuration = getattr(vacuum, "configuration_sha256", None)
            if not callable(configuration) or not callable(
                getattr(vacuum, "evaluate_energy", None)
            ):
                raise TypeError(
                    "vacuum must expose configuration_sha256 and evaluate_energy."
                )
            if vacuum.model_profile_id != equation.electronic.model_profile_id:
                raise ValueError(
                    "vacuum and response must bind the same model profile."
                )
            configuration()
        equation.fingerprint_sha256()
        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "vacuum", vacuum)
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("operational ledger is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        electronic_energy_behavior = "not-consumed-by-this-ledger"
        if self.scalar_id == (
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ):
            electronic_energy_behavior = provider_behavior_sha256(
                self.equation.electronic,
                ("configuration_sha256", "intrinsic_energy_ev"),
                label="separated_intrinsic_energy",
            )
        vacuum_record: dict[str, str] | None = None
        if self.vacuum is not None:
            vacuum_record = {
                "provider_id": self.vacuum.provider_id,
                "model_profile_id": self.vacuum.model_profile_id,
                "provenance_sha256": self.vacuum.provenance_sha256,
                "configuration_sha256": self.vacuum.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.vacuum,
                    ("configuration_sha256", "evaluate_energy"),
                    label="separated_vacuum",
                ),
            }
        payload = {
            "scalar_id": self.scalar_id,
            "registry_formula": get_scalar_definition(self.scalar_id).exact_formula,
            "implementation_entry_point": self.implementation_entry_point,
            "equation_sha256": self.equation.fingerprint_sha256(),
            "electronic_energy_behavior_sha256": electronic_energy_behavior,
            "vacuum": vacuum_record,
            "capabilities": "none",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("separated operational ledger configuration drifted.")
        return current

    def _components(
        self,
        geometry: object,
        source: np.ndarray,
        boundary_state: np.ndarray,
        native_field: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        raise NotImplementedError

    def evaluate_root(
        self,
        geometry: object,
        reduced_coordinates: object,
        *,
        root_tolerance: float,
    ) -> OperationalLedgerEvaluation:
        self.configuration_sha256()
        if not np.isfinite(root_tolerance) or root_tolerance <= 0.0:
            raise ValueError("root_tolerance must be finite and positive.")
        if geometry_sha256(geometry) != self.equation.continuum.geometry_sha256:
            raise ValueError("geometry does not match the ledger continuum snapshot.")
        y = np.asarray(reduced_coordinates, dtype=float)
        if y.shape != (self.equation.reduced_dimension,) or not np.all(np.isfinite(y)):
            raise ValueError("reduced_coordinates have an invalid shape.")
        residual = self.equation.residual(geometry, y)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > root_tolerance:
            raise ValueError(
                "operational ledger refuses a state above the root tolerance."
            )
        source = self.equation.source(y)
        boundary = self.equation.boundary_state(y)
        field = self.equation.continuum.native_field_from_boundary(boundary)
        components = self._components(geometry, source, boundary, field)
        total = float(sum(value for _, value in components))
        return OperationalLedgerEvaluation(
            scalar_id=self.scalar_id,
            state_equation_id=self.equation.state_equation_id,
            geometry_sha256=geometry_sha256(geometry),
            equation_sha256=self.equation.fingerprint_sha256(),
            ledger_sha256=self.configuration_sha256(),
            reduced_coordinates_sha256=_array_sha256(y),
            root_residual_norm=residual_norm,
            root_tolerance=float(root_tolerance),
            components_eV=components,
            total_energy_eV=total,
        )


class FrozenVacuumContinuumLedger(_SeparatedLedgerBase):
    """Phi0: frozen vacuum energy plus the signed continuum stationary energy."""

    implementation_entry_point = (
        "maple.solvation.coupling.separated_ledgers:FrozenVacuumContinuumLedger"
    )

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
        vacuum: SeparatedVacuumScalarProvider,
    ) -> None:
        super().__init__(
            equation=equation,
            vacuum=vacuum,
            scalar_id=(
                OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
            ),
        )

    def _components(
        self,
        geometry: object,
        source: np.ndarray,
        boundary_state: np.ndarray,
        native_field: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        del native_field
        if self.vacuum is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Phi0 vacuum provider is absent.")
        vacuum = float(self.vacuum.evaluate_energy(geometry))
        rhs = self.equation.continuum.source_rhs(source)
        continuum = -0.5 * float(np.vdot(rhs, boundary_state))
        if not np.isfinite(vacuum) or not np.isfinite(continuum):
            raise RuntimeError("Phi0 ledger produced a non-finite component.")
        return (
            ("macepolar_vacuum_energy", vacuum),
            ("smooth_harmonic_continuum_stationary_energy", continuum),
        )


class ExternalEnthalpyOperationalLedger(_SeparatedLedgerBase):
    """Phi1: checkpoint intrinsic field energy plus continuum self energy."""

    implementation_entry_point = (
        "maple.solvation.coupling.separated_ledgers:ExternalEnthalpyOperationalLedger"
    )

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
    ) -> None:
        if not callable(getattr(equation.electronic, "intrinsic_energy_ev", None)):
            raise TypeError("Phi1 electronic provider must expose intrinsic_energy_ev.")
        super().__init__(
            equation=equation,
            vacuum=None,
            scalar_id=(
                OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
            ),
        )

    def _components(
        self,
        geometry: object,
        source: np.ndarray,
        boundary_state: np.ndarray,
        native_field: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        del source
        intrinsic = float(
            self.equation.electronic.intrinsic_energy_ev(geometry, native_field)
        )
        self_energy = 0.5 * float(
            np.vdot(
                boundary_state,
                self.equation.continuum.surface_operator @ boundary_state,
            )
        )
        if not np.isfinite(intrinsic) or not np.isfinite(self_energy):
            raise RuntimeError("Phi1 ledger produced a non-finite component.")
        return (
            ("macepolar_intrinsic_field_conditioned_energy", intrinsic),
            ("continuum_polarization_self_energy", self_energy),
        )


__all__ = [
    "ExternalEnthalpyOperationalLedger",
    "FrozenVacuumContinuumLedger",
    "OperationalLedgerEvaluation",
    "SeparatedVacuumScalarProvider",
]
