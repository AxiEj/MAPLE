"""Capability layers, provenance, profiles, and admission gates for Route 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ....route2_model_contracts import (
    ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL,
    ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY,
)
from .route2_field_state import LocalReactionField
from .route2_plugin_errors import PluginContractError
from .route2_plugin_spaces import (
    ContinuumCoupling,
    CouplingAdjointEvidence,
    ElectronicSourceSpace,
    FieldDualSpace,
)

PluginExecutionMode = Literal["frozen", "one-shot", "scf"]
PluginAdmissionRoute = Literal["response-only", "operational-force", "variational"]
FieldConvention = Literal[
    "potential-gradient",
    "physical-electric-field",
    "unattested",
]
OptimizerCoverageStatus = Literal[
    "passed",
    "unattestable",
    "not-applicable",
]


def _nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _readonly_finite(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _frozen_mapping(value: Mapping[str, object], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    copied: dict[str, object] = {}
    for raw_key, item in value.items():
        key = _nonempty(raw_key, name=f"{name} key")
        copied[key] = _freeze_value(item, name=f"{name}.{key}")
    return MappingProxyType(copied)


def _freeze_value(value: object, *, name: str) -> object:
    if isinstance(value, Mapping):
        return _frozen_mapping(value, name=name)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, np.generic):
        return _freeze_value(value.item(), name=name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must not contain NaN or infinity.")
        return value
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}.")


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PluginProvenance:
    """Required model identity; unknown values are recorded as ``unattestable``."""

    checkpoint_sha256: str
    training_code_sha: str
    inference_code_sha: str
    representation: str
    supported_atomic_numbers: frozenset[int] | None
    total_charge_domain: tuple[float, float] | None
    field_convention: FieldConvention
    potential_unit: str
    gradient_unit: str
    energy_semantics: str
    optimizer_coverage: OptimizerCoverageStatus
    model_field_evaluator: str = "unattested"
    coordinate_frame_policy: str = "unattested"
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "training_code_sha",
            "inference_code_sha",
            "representation",
            "potential_unit",
            "gradient_unit",
            "energy_semantics",
            "model_field_evaluator",
            "coordinate_frame_policy",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        if self.field_convention not in {
            "potential-gradient",
            "physical-electric-field",
            "unattested",
        }:
            raise ValueError("Unsupported plug-in field convention.")
        if self.optimizer_coverage not in {"passed", "unattestable", "not-applicable"}:
            raise ValueError("Unsupported optimizer coverage status.")
        numbers = self.supported_atomic_numbers
        if numbers is not None:
            normalized = frozenset(numbers)
            if not normalized or any(
                isinstance(number, bool) or not isinstance(number, int) or number < 1
                for number in normalized
            ):
                raise ValueError(
                    "supported_atomic_numbers must contain positive atomic numbers."
                )
            object.__setattr__(self, "supported_atomic_numbers", normalized)
        domain = self.total_charge_domain
        if domain is not None:
            if len(domain) != 2 or not all(
                np.isfinite(float(value)) for value in domain
            ):
                raise ValueError(
                    "total_charge_domain must be two finite values or None."
                )
            lower, upper = map(float, domain)
            if lower > upper:
                raise ValueError("total_charge_domain lower bound exceeds upper bound.")
            object.__setattr__(self, "total_charge_domain", (lower, upper))
        object.__setattr__(
            self, "extra", _frozen_mapping(self.extra, name="plugin provenance")
        )

    def as_provenance(self) -> dict[str, object]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_code_sha": self.training_code_sha,
            "inference_code_sha": self.inference_code_sha,
            "representation": self.representation,
            "supported_atomic_numbers": (
                None
                if self.supported_atomic_numbers is None
                else sorted(self.supported_atomic_numbers)
            ),
            "total_charge_domain": self.total_charge_domain,
            "field_convention": self.field_convention,
            "potential_unit": self.potential_unit,
            "gradient_unit": self.gradient_unit,
            "energy_semantics": self.energy_semantics,
            "optimizer_coverage": self.optimizer_coverage,
            "model_field_evaluator": self.model_field_evaluator,
            "coordinate_frame_policy": self.coordinate_frame_policy,
            "extra": _thaw_value(self.extra),
        }


@dataclass(frozen=True)
class PluginElectronicState:
    """Model state normalized at the plug-in boundary."""

    energy_ev: float
    source: np.ndarray
    dipole_e_angstrom: np.ndarray
    fixed_field_forces_ev_per_angstrom: np.ndarray | None = None

    def __post_init__(self) -> None:
        energy = float(self.energy_ev)
        if not np.isfinite(energy):
            raise ValueError("plug-in energy must be finite.")
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(
            self, "source", _readonly_finite(self.source, name="plug-in source")
        )
        dipole = _readonly_finite(self.dipole_e_angstrom, name="plug-in dipole")
        if dipole.shape != (3,):
            raise ValueError("plug-in dipole must have shape (3,).")
        object.__setattr__(self, "dipole_e_angstrom", dipole)
        if self.fixed_field_forces_ev_per_angstrom is not None:
            forces = _readonly_finite(
                self.fixed_field_forces_ev_per_angstrom,
                name="plug-in fixed-field forces",
            )
            if forces.ndim != 2 or forces.shape[1] != 3:
                raise ValueError(
                    "plug-in fixed-field forces must have shape (n_atoms, 3)."
                )
            object.__setattr__(self, "fixed_field_forces_ev_per_angstrom", forces)


@runtime_checkable
class ElectronicSourceProvider(Protocol):
    """M0 minimum: auditable zero- and fixed-field electronic source states."""

    plugin_id: str
    source_space: ElectronicSourceSpace
    field_dual_space: FieldDualSpace
    provenance: PluginProvenance

    def cached_source_state(
        self, atoms: Any, *, require_forces: bool = False
    ) -> PluginElectronicState | None: ...

    def evaluate_source_state(
        self,
        atoms: Any,
        field: LocalReactionField | None,
        *,
        compute_forces: bool = False,
    ) -> tuple[PluginElectronicState, Mapping[str, object]]: ...


@runtime_checkable
class FieldResponsiveModel(ElectronicSourceProvider, Protocol):
    """Adds fixed-geometry source JVP/VJP under a nodewise local field."""

    def linearize_source_response(
        self, atoms: Any, field: LocalReactionField
    ) -> Any: ...


@runtime_checkable
class DifferentiableElectronicModel(FieldResponsiveModel, Protocol):
    """Adds the derivative interfaces needed by the operational-force route."""

    def source_position_vjp(
        self,
        atoms: Any,
        field: LocalReactionField,
        *,
        source_cotangent: np.ndarray,
    ) -> np.ndarray: ...

    def feature_vjp(
        self,
        atoms: Any,
        field: LocalReactionField,
        *,
        feature_cotangent: np.ndarray,
    ) -> np.ndarray: ...

    def fixed_field_forces(
        self,
        atoms: Any,
        field: LocalReactionField,
    ) -> np.ndarray: ...


@runtime_checkable
class VariationalElectronicModel(DifferentiableElectronicModel, Protocol):
    """Adds the common-functional derivatives required by a KKT solve."""

    def electronic_functional(self, atoms: Any, source: np.ndarray) -> float: ...

    def electronic_source_gradient(
        self, atoms: Any, source: np.ndarray
    ) -> np.ndarray: ...

    def electronic_source_hvp(
        self,
        atoms: Any,
        source: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PluginCapabilityDeclaration:
    """Declared derivative availability; calls remain fail-closed if absent."""

    field_response: bool = False
    fixed_field_forces: bool = False
    source_position_vjp: bool = False
    feature_vjp: bool = False
    variational_functional: bool = False

    def __post_init__(self) -> None:
        for name in (
            "field_response",
            "fixed_field_forces",
            "source_position_vjp",
            "feature_vjp",
            "variational_functional",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"plug-in capability {name} must be bool.")
        if self.variational_functional and not all(
            (
                self.field_response,
                self.fixed_field_forces,
                self.source_position_vjp,
                self.feature_vjp,
            )
        ):
            raise ValueError(
                "a variational plug-in must declare the differentiable capability layer."
            )


@dataclass(frozen=True)
class PluginProfile:
    """A narrow admission profile independent of legacy public parser profiles."""

    name: str
    solvent: str = "water"
    electrostatics_only: bool = True
    include_cds: bool = False
    primary_continuum_backend: str = "pyddx-ddpcm"
    independent_audit_backend: str = "pcmsolver-iefpcm"
    cavity_policy: str = "frozen-pminus1-water-v1"
    default_execution_mode: PluginExecutionMode = "scf"
    default_admission_route: PluginAdmissionRoute = "response-only"
    allowed_routes: frozenset[PluginAdmissionRoute] = frozenset({"response-only"})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _nonempty(self.name, name="plug-in profile name")
        )
        if self.solvent != "water" or not self.electrostatics_only:
            raise ValueError(
                "P-1 plug-in profiles are water-only and electrostatics-only."
            )
        if self.include_cds:
            raise ValueError("P-1 plug-in profiles freeze CDS outside the active path.")
        for name in (
            "primary_continuum_backend",
            "independent_audit_backend",
            "cavity_policy",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=f"plug-in profile {name}"),
            )
        if self.default_execution_mode not in {"frozen", "one-shot", "scf"}:
            raise ValueError("unsupported default plug-in execution mode.")
        if self.default_admission_route not in {
            "response-only",
            "operational-force",
            "variational",
        }:
            raise ValueError("unsupported default plug-in admission route.")
        routes = frozenset(self.allowed_routes)
        if not routes or not routes <= {
            "response-only",
            "operational-force",
            "variational",
        }:
            raise ValueError("plug-in profile allowed_routes are invalid.")
        if self.default_admission_route not in routes:
            raise ValueError("default plug-in route must be explicitly allowed.")
        object.__setattr__(self, "allowed_routes", routes)


P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE = PluginProfile(
    name="route2-pminus1-water-electrostatics-response-only-v1"
)


@dataclass(frozen=True)
class PluginAdmission:
    profile: PluginProfile
    execution_mode: PluginExecutionMode
    route: PluginAdmissionRoute
    plugin_id: str
    coupling_evidence: CouplingAdjointEvidence | None
    diagnostics_only: bool


def validate_plugin_source_provider(plugin: object) -> ElectronicSourceProvider:
    required = (
        "plugin_id",
        "source_space",
        "field_dual_space",
        "provenance",
        "cached_source_state",
        "evaluate_source_state",
    )
    missing = [
        name
        for name in required
        if not callable(getattr(plugin, name, None))
        and name not in {"plugin_id", "source_space", "field_dual_space", "provenance"}
    ]
    missing.extend(
        name
        for name in ("plugin_id", "source_space", "field_dual_space", "provenance")
        if not hasattr(plugin, name)
    )
    if missing:
        raise PluginContractError(
            "plug-in is not an ElectronicSourceProvider; missing: "
            + ", ".join(sorted(set(missing)))
        )
    if not isinstance(plugin.source_space, ElectronicSourceSpace):
        raise PluginContractError(
            "plug-in source_space is not an ElectronicSourceSpace."
        )
    if not isinstance(plugin.field_dual_space, FieldDualSpace):
        raise PluginContractError("plug-in field_dual_space is not a FieldDualSpace.")
    if not isinstance(plugin.provenance, PluginProvenance):
        raise PluginContractError("plug-in provenance is not a PluginProvenance.")
    _nonempty(plugin.plugin_id, name="plug-in id")
    return plugin  # type: ignore[return-value]


def _require_callable(plugin: object, name: str, *, route: str) -> None:
    if not callable(getattr(plugin, name, None)):
        raise PluginContractError(
            f"{route} declares a capability but does not implement {name}()."
        )


def _validate_plugin_identity(provider: ElectronicSourceProvider) -> None:
    if provider.provenance.representation != provider.source_space.representation:
        raise PluginContractError(
            "plug-in provenance/source-space representation mismatch."
        )
    if provider.provenance.potential_unit != provider.field_dual_space.potential_unit:
        raise PluginContractError(
            "plug-in provenance/field-dual potential-unit mismatch."
        )
    if provider.provenance.gradient_unit != provider.field_dual_space.gradient_unit:
        raise PluginContractError(
            "plug-in provenance/field-dual gradient-unit mismatch."
        )


def _validate_plugin_domain(
    provider: ElectronicSourceProvider,
    *,
    atomic_numbers: object | None,
    total_charge_e: float | None,
) -> None:
    if atomic_numbers is not None:
        values = np.asarray(atomic_numbers)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError("atomic_numbers must be a non-empty integer vector.")
        supported = provider.provenance.supported_atomic_numbers
        if supported is None:
            raise PluginContractError("plug-in element domain is unattestable.")
        outside = sorted(set(map(int, values)) - supported)
        if outside:
            raise PluginContractError(
                "plug-in element domain excludes atomic numbers: "
                + ", ".join(map(str, outside))
                + "."
            )
    if total_charge_e is not None:
        charge = float(total_charge_e)
        if not np.isfinite(charge):
            raise ValueError("total_charge_e must be finite.")
        domain = provider.provenance.total_charge_domain
        if domain is None:
            raise PluginContractError("plug-in charge domain is unattestable.")
        if not domain[0] <= charge <= domain[1]:
            raise PluginContractError(
                f"plug-in charge domain {domain} excludes total charge {charge}."
            )


def _declared_capabilities(plugin: object) -> PluginCapabilityDeclaration:
    value = getattr(plugin, "capabilities", None)
    if not isinstance(value, PluginCapabilityDeclaration):
        raise PluginContractError("plug-in must declare PluginCapabilityDeclaration.")
    return value


def admit_plugin(
    plugin: object,
    *,
    profile: PluginProfile = P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE,
    execution_mode: PluginExecutionMode | None = None,
    route: PluginAdmissionRoute | None = None,
    coupling: ContinuumCoupling | None = None,
    adjoint_probe: tuple[object, object, int] | None = None,
    atomic_numbers: object | None = None,
    total_charge_e: float | None = None,
) -> PluginAdmission:
    """Validate an explicit route without promoting response data to a functional.

    A response-only admission is deliberately diagnostic when the coupling
    cannot establish an adjoint pairing.  Operational-force and variational
    routes require increasingly strong declared and callable derivatives.
    """

    provider = validate_plugin_source_provider(plugin)
    _validate_plugin_identity(provider)
    _validate_plugin_domain(
        provider,
        atomic_numbers=atomic_numbers,
        total_charge_e=total_charge_e,
    )
    capabilities = _declared_capabilities(provider)
    selected_mode = (
        profile.default_execution_mode if execution_mode is None else execution_mode
    )
    selected_route = profile.default_admission_route if route is None else route
    if selected_mode not in {"frozen", "one-shot", "scf"}:
        raise ValueError("unsupported plug-in execution mode.")
    if selected_route not in {"response-only", "operational-force", "variational"}:
        raise ValueError("unsupported plug-in admission route.")
    if selected_route not in profile.allowed_routes:
        raise PluginContractError(
            f"profile {profile.name!r} does not admit route {selected_route!r}."
        )
    required_energy_semantics = (
        ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL
        if selected_route == "variational"
        else ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY
    )
    if provider.provenance.energy_semantics != required_energy_semantics:
        raise PluginContractError(
            f"{selected_route} requires energy semantics "
            f"{required_energy_semantics!r}; received "
            f"{provider.provenance.energy_semantics!r}."
        )
    if selected_mode != "frozen" and not capabilities.field_response:
        raise PluginContractError(
            "field-conditioned execution requires declared source response."
        )
    if (
        selected_mode != "frozen"
        and provider.provenance.field_convention == "unattested"
    ):
        raise PluginContractError(
            "field-conditioned execution is blocked until the plug-in field convention is audited."
        )
    if selected_mode != "frozen":
        _require_callable(
            provider,
            "linearize_source_response",
            route="field-conditioned execution",
        )

    evidence: CouplingAdjointEvidence | None = None
    if coupling is not None:
        if (
            coupling.source_space.name != provider.source_space.name
            or coupling.field_dual_space.name != provider.field_dual_space.name
        ):
            raise PluginContractError(
                "plug-in and continuum coupling spaces do not match."
            )
        if adjoint_probe is not None:
            source, surface, atom_count = adjoint_probe
            evidence = coupling.verify_adjoint(source, surface, atom_count=atom_count)

    if selected_route == "response-only":
        return PluginAdmission(
            profile=profile,
            execution_mode=selected_mode,
            route=selected_route,
            plugin_id=provider.plugin_id,
            coupling_evidence=evidence,
            diagnostics_only=evidence is not None and not evidence.passed,
        )

    if selected_route == "operational-force":
        missing: list[str] = []
        if selected_mode != "scf":
            missing.append("scf execution mode")
        if not capabilities.fixed_field_forces:
            missing.append("fixed-field forces")
        if not capabilities.source_position_vjp:
            missing.append("source-position VJP")
        if not capabilities.feature_vjp:
            missing.append("feature VJP")
        if missing:
            raise PluginContractError(
                "operational-force route requires " + ", ".join(missing) + "."
            )
        for method in (
            "fixed_field_forces",
            "source_position_vjp",
            "feature_vjp",
        ):
            _require_callable(provider, method, route="operational-force route")
        return PluginAdmission(
            profile, selected_mode, selected_route, provider.plugin_id, evidence, False
        )

    missing = []
    if selected_mode != "scf":
        missing.append("scf execution mode")
    if not capabilities.variational_functional:
        missing.append("variational functional/source gradient/HVP")
    if not all(
        (
            capabilities.fixed_field_forces,
            capabilities.source_position_vjp,
            capabilities.feature_vjp,
        )
    ):
        missing.append("differentiable electronic-model capability layer")
    if coupling is None:
        missing.append("explicit continuum coupling")
    if evidence is None or not evidence.passed:
        missing.append("verified B/B* adjoint pairing")
    if provider.source_space.total_charge is None:
        missing.append("source total-charge functional")
    if missing:
        raise PluginContractError(
            "variational route requires " + ", ".join(missing) + "."
        )
    for method in (
        "fixed_field_forces",
        "source_position_vjp",
        "feature_vjp",
        "electronic_functional",
        "electronic_source_gradient",
        "electronic_source_hvp",
    ):
        _require_callable(provider, method, route="variational route")
    return PluginAdmission(
        profile, selected_mode, selected_route, provider.plugin_id, evidence, False
    )


__all__ = [
    "DifferentiableElectronicModel",
    "ElectronicSourceProvider",
    "FieldResponsiveModel",
    "P_MINUS_1_WATER_RESPONSE_ONLY_PROFILE",
    "PluginAdmission",
    "PluginAdmissionRoute",
    "PluginCapabilityDeclaration",
    "PluginElectronicState",
    "PluginExecutionMode",
    "PluginProfile",
    "PluginProvenance",
    "VariationalElectronicModel",
    "admit_plugin",
    "validate_plugin_source_provider",
]
