"""Electronic-model adapter contract for Route-2 continuum coupling.

The continuum engine must depend on physical capabilities and a declared
source/dual space, not on a concrete MLIP class name.  This module defines the
current atom-centred ``l<=1`` interoperability boundary and a small adapter
protocol.  Native model representations remain private to their adapters.

Existing released Route-2 profiles are still model-identity-bound.  The
adapter layer enables a new MLIP to reuse the SCF/PCM/COSMO/adjoint machinery,
but it does not allow that MLIP to impersonate an already certified profile.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ....route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
    validate_route2_electrostatic_energy_ledger,
)
from ....route2_model_contracts import (
    ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL,
    ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    ROUTE2_ATOMIC_L1_SOURCE_SPACE,
    ROUTE2_ELECTRONIC_MODEL_ADAPTER_VERSION,
)
from .electrostatic_pairing import ElectrostaticPairing, MACE_POLAR_L1_PAIRING
from .gto_field_projection import ExactGTOFieldProjector
from .route2_field_state import ReactionFieldDrive

FIELD_CONDITIONED_OPERATIONAL_ENERGY = ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY
COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL = (
    ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL
)
_ENERGY_SEMANTICS = frozenset(
    {
        FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL,
    }
)


def _nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _freeze_provenance(value: object, *, path: str) -> object:
    """Copy one JSON-compatible audit value into immutable containers."""

    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise TypeError(f"{path} keys must be non-empty strings.")
            frozen[raw_key] = _freeze_provenance(
                item,
                path=f"{path}.{raw_key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_provenance(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, np.generic):
        return _freeze_provenance(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    raise TypeError(
        f"{path} contains non-JSON provenance value {type(value).__name__}."
    )


def _thaw_provenance(value: object) -> object:
    """Return a detached JSON-ready copy of immutable audit provenance."""

    if isinstance(value, Mapping):
        return {key: _thaw_provenance(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_provenance(item) for item in value]
    return value


@dataclass(frozen=True)
class AtomicL1SourceSpace:
    """Canonical atom-centred monopole/dipole source and dual pairing.

    Adapters convert their native output to ``(n_atoms, 4)`` arrays in this
    space.  The scalar component is the net atomic monopole and component zero
    is therefore the fixed-total-charge coordinate.  The remaining raw l=1
    order is defined exclusively by ``pairing``.
    """

    name: str = ROUTE2_ATOMIC_L1_SOURCE_SPACE
    pairing: ElectrostaticPairing = MACE_POLAR_L1_PAIRING
    component_count: int = 4
    charge_component: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, name="source-space name"))
        if self.component_count != 4 or self.charge_component != 0:
            raise ValueError(
                "The Route-2 atomic-l1 source space requires four components "
                "with the monopole at component zero."
            )
        if not isinstance(self.pairing, ElectrostaticPairing):
            raise TypeError("Atomic-l1 source space requires an ElectrostaticPairing.")

    def validate_source(
        self,
        values: np.ndarray,
        *,
        atom_count: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        expected = (atom_count, self.component_count)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must be finite in source space {self.name!r} with "
                f"shape {expected}; received {array.shape}."
            )
        return np.array(array, dtype=float, copy=True)

    def validate_field(
        self,
        values: np.ndarray,
        *,
        atom_count: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        expected = (atom_count, self.component_count)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must be finite in the dual of {self.name!r} with "
                f"shape {expected}; received {array.shape}."
            )
        return np.array(array, dtype=float, copy=True)

    def total_charge(self, source: np.ndarray) -> float:
        array = np.asarray(source, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.component_count:
            raise ValueError("Atomic-l1 source must have shape (n_atoms, 4).")
        return float(np.sum(array[:, self.charge_component]))

    def pair(self, source: np.ndarray, field: np.ndarray) -> float:
        return self.pairing.pair(source, field)


ATOMIC_L1_SOURCE_SPACE = AtomicL1SourceSpace()


@dataclass(frozen=True)
class Route2ElectronicModelCapabilities:
    """Capabilities advertised by one adapter; absence always fails closed."""

    state_projectors: frozenset[str]
    gas_forces: bool = False
    response_projectors: frozenset[str] = frozenset()
    position_vjp_projectors: frozenset[str] = frozenset()
    fixed_field_force_projectors: frozenset[str] = frozenset()
    energy_gradient_projectors: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.gas_forces, bool):
            raise TypeError("Electronic-model gas_forces capability must be bool.")
        known_projectors = frozenset({"local-jet", "exact-gto-v1"})
        capability_names = (
            "state_projectors",
            "response_projectors",
            "position_vjp_projectors",
            "fixed_field_force_projectors",
            "energy_gradient_projectors",
        )
        for name in capability_names:
            raw = getattr(self, name)
            if isinstance(raw, str) or not isinstance(
                raw,
                (set, frozenset, tuple, list),
            ):
                raise TypeError(f"Electronic-model capability {name} must be a set.")
            values = frozenset(str(value).strip() for value in raw)
            if (name == "state_projectors" and not values) or "" in values:
                raise ValueError(
                    "Electronic-model state_projectors must be non-empty, "
                    "and projector identities must never be blank."
                )
            unknown = values - known_projectors
            if unknown:
                raise ValueError(
                    f"Unsupported Route-2 field projectors in {name}: "
                    + ", ".join(sorted(unknown))
                    + "."
                )
            object.__setattr__(self, name, values)
        for name in capability_names[1:]:
            if not getattr(self, name) <= self.state_projectors:
                raise ValueError(
                    f"Electronic-model {name} must be a subset of " "state_projectors."
                )

    def as_provenance(self) -> dict[str, object]:
        """Return deterministic, JSON-ready capability evidence."""

        return {
            "state_projectors": sorted(self.state_projectors),
            "gas_forces": self.gas_forces,
            "response_projectors": sorted(self.response_projectors),
            "position_vjp_projectors": sorted(self.position_vjp_projectors),
            "fixed_field_force_projectors": sorted(self.fixed_field_force_projectors),
            "energy_gradient_projectors": sorted(self.energy_gradient_projectors),
        }


@dataclass(frozen=True)
class Route2ElectronicModelDescriptor:
    """Immutable scientific identity and capability declaration for an MLIP."""

    adapter_name: str
    model_family: str
    field_evaluator: str
    source_space: AtomicL1SourceSpace
    capabilities: Route2ElectronicModelCapabilities
    energy_semantics: str
    profile_binding: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    adapter_version: int = ROUTE2_ELECTRONIC_MODEL_ADAPTER_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adapter_name",
            _nonempty(self.adapter_name, name="adapter_name"),
        )
        object.__setattr__(
            self,
            "model_family",
            _nonempty(self.model_family, name="model_family"),
        )
        object.__setattr__(
            self,
            "field_evaluator",
            _nonempty(self.field_evaluator, name="field_evaluator"),
        )
        if self.adapter_version != ROUTE2_ELECTRONIC_MODEL_ADAPTER_VERSION:
            raise ValueError("Unsupported Route-2 electronic-model adapter version.")
        if not isinstance(self.source_space, AtomicL1SourceSpace):
            raise TypeError("Route-2 descriptor requires an AtomicL1SourceSpace.")
        if not isinstance(self.capabilities, Route2ElectronicModelCapabilities):
            raise TypeError("Route-2 descriptor capabilities are malformed.")
        profile_binding = self.profile_binding
        if profile_binding is not None:
            profile_binding = _nonempty(profile_binding, name="profile_binding")
        object.__setattr__(self, "profile_binding", profile_binding)
        semantics = _nonempty(self.energy_semantics, name="energy_semantics")
        if semantics not in _ENERGY_SEMANTICS:
            raise ValueError(f"Unsupported Route-2 energy semantics: {semantics!r}.")
        object.__setattr__(self, "energy_semantics", semantics)
        if not isinstance(self.provenance, Mapping):
            raise TypeError("Electronic-model provenance must be a mapping.")
        object.__setattr__(
            self,
            "provenance",
            _freeze_provenance(self.provenance, path="electronic-model provenance"),
        )

    def as_provenance(self) -> dict[str, object]:
        """Return the complete immutable model/adapter identity for audits."""

        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "model_family": self.model_family,
            "source_space": self.source_space.name,
            "field_evaluator": self.field_evaluator,
            "energy_semantics": self.energy_semantics,
            "profile_binding": self.profile_binding,
            "capabilities": self.capabilities.as_provenance(),
            "provenance": _thaw_provenance(self.provenance),
        }


@runtime_checkable
class Route2ElectronicState(Protocol):
    """Standard state returned by a Route-2 electronic-model adapter."""

    energy_ev: float
    density_coefficients: np.ndarray
    dipole_e_angstrom: np.ndarray
    fixed_field_forces_ev_per_angstrom: np.ndarray | None


@runtime_checkable
class Route2ElectronicModel(Protocol):
    """Model-neutral interface consumed by the Route-2 continuum engine."""

    descriptor: Route2ElectronicModelDescriptor
    cache_identity: int

    def cached_state(
        self,
        atoms: Any,
        *,
        require_forces: bool = False,
    ) -> Route2ElectronicState | None: ...

    def evaluate_state(
        self,
        atoms: Any,
        drive: ReactionFieldDrive | None,
        *,
        compute_forces: bool = False,
    ) -> tuple[Route2ElectronicState, Mapping[str, object]]: ...

    def linearize_source_response(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
    ) -> Any: ...

    def preprojected_field_projector(self) -> Any:
        """Return the adapter-owned projector into native field features."""
        ...

    def field_conditioned_energy_field_gradient(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
    ) -> np.ndarray: ...

    def source_position_vjp(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
        *,
        source_cotangent: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class AtomicL1CalculatorAdapter:
    """Adapt the established calculator method surface to the model protocol.

    This is the only compatibility layer that knows the legacy method names.
    A different MLIP may either expose the same methods with a descriptor or
    return its own adapter from ``route2_electronic_model_adapter()``.
    """

    calculator: Any
    descriptor: Route2ElectronicModelDescriptor

    @property
    def cache_identity(self) -> int:
        return id(self.calculator)

    def cached_state(
        self,
        atoms: Any,
        *,
        require_forces: bool = False,
    ) -> Route2ElectronicState | None:
        cached = getattr(self.calculator, "cached_polar_state", None)
        if callable(cached):
            return cached(atoms, require_forces=require_forces)
        # A raw ``_last_polar_state`` carries no geometry-identity contract.
        # Re-evaluation is safer than accepting a potentially stale source.
        return None

    def evaluate_state(
        self,
        atoms: Any,
        drive: ReactionFieldDrive | None,
        *,
        compute_forces: bool = False,
    ) -> tuple[Route2ElectronicState, Mapping[str, object]]:
        evaluate = getattr(self.calculator, "polar_state", None)
        if not callable(evaluate):
            raise TypeError("Electronic-model adapter requires polar_state().")
        force_kwargs = {"compute_forces": True} if compute_forces else {}
        if drive is None:
            return evaluate(atoms, **force_kwargs)
        if drive.model_field_features is not None:
            return evaluate(
                atoms,
                model_field_features=drive.model_field_features,
                **force_kwargs,
            )
        field = (
            drive.density_dual_field_ev
            if drive.model_local_field_ev is None
            else drive.model_local_field_ev
        )
        return evaluate(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            **force_kwargs,
        )

    def linearize_source_response(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
    ) -> Any:
        if drive.model_field_features is not None:
            linearize = getattr(
                self.calculator,
                "linearize_density_response_features",
                None,
            )
            if not callable(linearize):
                raise NotImplementedError(
                    "Electronic model has no preprojected-feature response linearization."
                )
            return linearize(atoms, model_field_features=drive.model_field_features)
        linearize = getattr(self.calculator, "linearize_density_response", None)
        if not callable(linearize):
            raise NotImplementedError("Electronic model has no density JVP/VJP.")
        field = (
            drive.density_dual_field_ev
            if drive.model_local_field_ev is None
            else drive.model_local_field_ev
        )
        return linearize(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )

    def preprojected_field_projector(self) -> ExactGTOFieldProjector:
        projection_spec = getattr(
            self.calculator,
            "route2_gto_field_projection_spec",
            None,
        )
        if not callable(projection_spec):
            raise NotImplementedError(
                "Electronic model has no preprojected field-feature interface."
            )
        return ExactGTOFieldProjector(projection_spec())

    def field_conditioned_energy_field_gradient(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
    ) -> np.ndarray:
        if drive.model_field_features is not None:
            raise NotImplementedError(
                "Preprojected-feature energy derivatives require a distinct adapter."
            )
        derivative = getattr(
            self.calculator,
            "intrinsic_energy_field_gradient",
            None,
        )
        if not callable(derivative):
            raise NotImplementedError(
                "Electronic model has no intrinsic energy/field derivative."
            )
        field = (
            drive.density_dual_field_ev
            if drive.model_local_field_ev is None
            else drive.model_local_field_ev
        )
        return derivative(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
        )

    def source_position_vjp(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
        *,
        source_cotangent: np.ndarray,
    ) -> np.ndarray:
        if drive.model_field_features is not None:
            raise NotImplementedError(
                "Preprojected-feature coordinate VJP requires a distinct adapter."
            )
        derivative = getattr(self.calculator, "density_position_vjp", None)
        if not callable(derivative):
            raise NotImplementedError("Electronic model has no density position VJP.")
        field = (
            drive.density_dual_field_ev
            if drive.model_local_field_ev is None
            else drive.model_local_field_ev
        )
        return derivative(
            atoms,
            node_potential_ev=field[:, 0],
            node_gradient_ev_per_angstrom=field[:, 1:],
            density_cotangent=source_cotangent,
        )


def _validate_adapter(value: object) -> Route2ElectronicModel:
    descriptor = getattr(value, "descriptor", None)
    if not isinstance(descriptor, Route2ElectronicModelDescriptor):
        raise TypeError(
            "Route-2 electronic adapter must expose a validated descriptor."
        )
    cache_identity = getattr(value, "cache_identity", None)
    if isinstance(cache_identity, bool) or not isinstance(cache_identity, int):
        raise TypeError("Route-2 electronic adapter cache_identity must be an int.")
    required = (
        "cached_state",
        "evaluate_state",
        "linearize_source_response",
        "preprojected_field_projector",
        "field_conditioned_energy_field_gradient",
        "source_position_vjp",
    )
    missing = [name for name in required if not callable(getattr(value, name, None))]
    if missing:
        raise TypeError(
            "Route-2 electronic adapter is incomplete; missing: "
            + ", ".join(missing)
            + "."
        )
    return value  # type: ignore[return-value]


def resolve_route2_electronic_model(candidate: object) -> Route2ElectronicModel:
    """Resolve an explicit adapter or an adapter-producing calculator.

    There is deliberately no arbitrary ``polar_state`` duck-typing fallback.
    New calculators opt in by exposing a validated descriptor or a factory;
    this prevents an unrelated calculator with coincidentally named methods
    from entering a scientific Route-2 profile.
    """

    factory = getattr(candidate, "route2_electronic_model_adapter", None)
    if callable(factory):
        return _validate_adapter(factory())
    descriptor = getattr(candidate, "route2_electronic_model_descriptor", None)
    if isinstance(descriptor, Route2ElectronicModelDescriptor):
        return AtomicL1CalculatorAdapter(
            calculator=candidate,
            descriptor=descriptor,
        )
    return _validate_adapter(candidate)


def validate_route2_electronic_model_capabilities(
    model: Route2ElectronicModel,
    *,
    expected_model_family: str,
    expected_source_space: str,
    expected_profile_binding: str | None,
    expected_field_evaluator: str,
    expected_energy_semantics: str,
    reaction_field_projector: str,
    electrostatic_energy_ledger: str,
    need_forces: bool,
    response_mode: str = "scf",
) -> None:
    """Fail closed unless one adapter can execute the selected profile."""

    descriptor = model.descriptor
    expected_family = _nonempty(
        expected_model_family,
        name="expected_model_family",
    )
    expected_space = _nonempty(
        expected_source_space,
        name="expected_source_space",
    )
    if descriptor.model_family != expected_family:
        raise TypeError(
            "Route-2 profile/model mismatch: expected electronic family "
            f"{expected_family!r}, received {descriptor.model_family!r}."
        )
    if descriptor.source_space.name != expected_space:
        raise TypeError(
            "Route-2 profile/source-space mismatch: expected "
            f"{expected_space!r}, received {descriptor.source_space.name!r}."
        )
    if descriptor.profile_binding != expected_profile_binding:
        raise TypeError(
            "Route-2 profile binding mismatch: expected "
            f"{expected_profile_binding!r}, received "
            f"{descriptor.profile_binding!r}."
        )
    expected_evaluator = _nonempty(
        expected_field_evaluator,
        name="expected_field_evaluator",
    )
    if descriptor.field_evaluator != expected_evaluator:
        raise TypeError(
            "Route-2 model-field/long-range evaluator mismatch: expected "
            f"{expected_evaluator!r}, received "
            f"{descriptor.field_evaluator!r}."
        )
    expected_semantics = _nonempty(
        expected_energy_semantics,
        name="expected_energy_semantics",
    )
    if descriptor.energy_semantics != expected_semantics:
        raise TypeError(
            "Route-2 profile/model energy-semantics mismatch: expected "
            f"{expected_semantics!r}, received "
            f"{descriptor.energy_semantics!r}."
        )
    capabilities = descriptor.capabilities
    mode = _nonempty(response_mode, name="response_mode").lower()
    if mode not in {"frozen", "one-shot", "scf"}:
        raise ValueError(f"Unsupported Route-2 response mode: {mode!r}.")
    projector = _nonempty(
        reaction_field_projector,
        name="reaction_field_projector",
    )
    if projector not in {"local-jet", "exact-gto-v1"}:
        raise ValueError(
            f"Unsupported Route-2 reaction-field projector: {projector!r}."
        )
    if mode != "frozen" and projector not in capabilities.state_projectors:
        raise TypeError(
            "Electronic model cannot consume reaction-field projector "
            f"{projector!r}."
        )

    ledger = validate_route2_electrostatic_energy_ledger(electrostatic_energy_ledger)
    if not need_forces:
        return
    if mode != "scf":
        raise TypeError("Route-2 force derivatives require response_mode='scf'.")
    missing: list[str] = []
    if not capabilities.gas_forces:
        missing.append("gas-forces")
    if projector not in capabilities.response_projectors:
        missing.append("source-response-jvp-vjp")
    if projector not in capabilities.position_vjp_projectors:
        missing.append("source-position-vjp")
    if ledger == LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1:
        if projector not in capabilities.fixed_field_force_projectors:
            missing.append("fixed-field-forces")
        if projector not in capabilities.energy_gradient_projectors:
            missing.append("field-conditioned-energy-field-gradient")
    if ledger not in {
        LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
        PCM_HALF_COUPLING_ONLY_V1,
    }:
        raise AssertionError("Validated Route-2 energy ledger is unknown.")
    if missing:
        raise TypeError(
            "Electronic model lacks capabilities required for Route-2 forces: "
            + ", ".join(missing)
            + "."
        )


__all__ = [
    "ATOMIC_L1_SOURCE_SPACE",
    "AtomicL1CalculatorAdapter",
    "AtomicL1SourceSpace",
    "COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL",
    "FIELD_CONDITIONED_OPERATIONAL_ENERGY",
    "Route2ElectronicModel",
    "Route2ElectronicModelCapabilities",
    "Route2ElectronicModelDescriptor",
    "Route2ElectronicState",
    "resolve_route2_electronic_model",
    "validate_route2_electronic_model_capabilities",
]
