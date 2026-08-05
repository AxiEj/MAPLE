"""Explicit atomic-l1 bridge from M0 plug-ins to the established Route-2 engine.

The bridge is intentionally limited to the legacy local-jet source/field space.
It translates state names and field containers only; SCF, PCM, energy-ledger,
and force algorithms remain owned by the existing Route-2 engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .route2_electronic_model import (
    ATOMIC_L1_SOURCE_SPACE,
    Route2ElectronicModelCapabilities,
    Route2ElectronicModelDescriptor,
)
from .route2_field_state import ReactionFieldDrive
from .route2_plugin_contracts import (
    ElectronicSourceProvider,
    PluginCapabilityDeclaration,
    PluginElectronicState,
    validate_plugin_source_provider,
)
from .route2_plugin_errors import PluginContractError
from .route2_plugin_spaces import (
    ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
    ATOMIC_L1_PLUGIN_SOURCE_SPACE,
)


def _nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class LegacyAtomicL1BridgeSpec:
    """Explicit scientific identity used by an internal compatibility run."""

    model_family: str
    field_evaluator: str
    profile_binding: str

    def __post_init__(self) -> None:
        for name in ("model_family", "field_evaluator", "profile_binding"):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=f"bridge {name}"),
            )


@dataclass(frozen=True)
class _LegacyElectronicStateView:
    """Name-only state conversion expected by the existing continuum engine."""

    energy_ev: float
    density_coefficients: np.ndarray
    dipole_e_angstrom: np.ndarray
    fixed_field_forces_ev_per_angstrom: np.ndarray | None


@dataclass(frozen=True)
class AtomicL1PluginEngineAdapter:
    """Adapt one explicitly registered M0 plug-in to the old local-jet engine."""

    plugin: ElectronicSourceProvider
    descriptor: Route2ElectronicModelDescriptor
    capabilities: PluginCapabilityDeclaration

    @classmethod
    def from_plugin(
        cls,
        plugin: object,
        *,
        spec: LegacyAtomicL1BridgeSpec,
    ) -> "AtomicL1PluginEngineAdapter":
        provider = validate_plugin_source_provider(plugin)
        capabilities = getattr(provider, "capabilities", None)
        if not isinstance(capabilities, PluginCapabilityDeclaration):
            raise PluginContractError(
                "atomic-l1 engine bridge requires PluginCapabilityDeclaration."
            )
        if provider.source_space.name != ATOMIC_L1_PLUGIN_SOURCE_SPACE.name:
            raise PluginContractError(
                "legacy engine bridge accepts only the canonical atomic-l1 source space."
            )
        if provider.field_dual_space.name != ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.name:
            raise PluginContractError(
                "legacy engine bridge accepts only the canonical local-jet field dual."
            )
        local_jet = frozenset({"local-jet"})
        descriptor = Route2ElectronicModelDescriptor(
            adapter_name=f"{provider.plugin_id}-atomic-l1-engine-bridge-v1",
            model_family=spec.model_family,
            field_evaluator=spec.field_evaluator,
            source_space=ATOMIC_L1_SOURCE_SPACE,
            capabilities=Route2ElectronicModelCapabilities(
                state_projectors=local_jet,
                gas_forces=capabilities.fixed_field_forces,
                response_projectors=(
                    local_jet if capabilities.field_response else frozenset()
                ),
                position_vjp_projectors=(
                    local_jet if capabilities.source_position_vjp else frozenset()
                ),
                fixed_field_force_projectors=(
                    local_jet if capabilities.fixed_field_forces else frozenset()
                ),
                # The M0 derivative layer exposes feature VJP, not the legacy
                # field-conditioned energy gradient.  The legacy ledger
                # therefore remains correctly unavailable through this bridge.
                energy_gradient_projectors=frozenset(),
            ),
            energy_semantics=provider.provenance.energy_semantics,
            profile_binding=spec.profile_binding,
            provenance={"m0_plugin": provider.provenance.as_provenance()},
        )
        return cls(provider, descriptor, capabilities)

    @property
    def cache_identity(self) -> int:
        return id(self.plugin)

    @staticmethod
    def _legacy_state(state: PluginElectronicState) -> _LegacyElectronicStateView:
        if not isinstance(state, PluginElectronicState):
            raise PluginContractError(
                "M0 plug-in must return PluginElectronicState at the legacy bridge."
            )
        return _LegacyElectronicStateView(
            energy_ev=state.energy_ev,
            density_coefficients=np.array(state.source, copy=True),
            dipole_e_angstrom=np.array(state.dipole_e_angstrom, copy=True),
            fixed_field_forces_ev_per_angstrom=(
                None
                if state.fixed_field_forces_ev_per_angstrom is None
                else np.array(state.fixed_field_forces_ev_per_angstrom, copy=True)
            ),
        )

    def cached_state(self, atoms: Any, *, require_forces: bool = False):
        state = self.plugin.cached_source_state(atoms, require_forces=require_forces)
        return None if state is None else self._legacy_state(state)

    def evaluate_state(
        self,
        atoms: Any,
        drive: ReactionFieldDrive | None,
        *,
        compute_forces: bool = False,
    ) -> tuple[_LegacyElectronicStateView, Mapping[str, object]]:
        field = None if drive is None else drive.local_reaction_field()
        state, provenance = self.plugin.evaluate_source_state(
            atoms,
            field,
            compute_forces=compute_forces,
        )
        return self._legacy_state(state), provenance

    def linearize_source_response(self, atoms: Any, drive: ReactionFieldDrive) -> Any:
        if not self.capabilities.field_response:
            raise PluginContractError("plug-in has no admitted field-response layer.")
        method = getattr(self.plugin, "linearize_source_response", None)
        if not callable(method):
            raise PluginContractError(
                "plug-in declares field response but has no linearize_source_response()."
            )
        return method(atoms, drive.local_reaction_field())

    def preprojected_field_projector(self) -> Any:
        raise NotImplementedError(
            "The generic M0 bridge does not coerce exact-GTO features into a local jet."
        )

    def field_conditioned_energy_field_gradient(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
    ) -> np.ndarray:
        del atoms, drive
        raise NotImplementedError(
            "The M0 contract does not infer a legacy field-conditioned energy gradient."
        )

    def source_position_vjp(
        self,
        atoms: Any,
        drive: ReactionFieldDrive,
        *,
        source_cotangent: np.ndarray,
    ) -> np.ndarray:
        if not self.capabilities.source_position_vjp:
            raise PluginContractError("plug-in has no admitted source-position VJP.")
        method = getattr(self.plugin, "source_position_vjp", None)
        if not callable(method):
            raise PluginContractError(
                "plug-in declares source-position VJP but does not implement it."
            )
        return np.asarray(
            method(
                atoms,
                drive.local_reaction_field(),
                source_cotangent=source_cotangent,
            ),
            dtype=float,
        )


def adapt_atomic_l1_plugin_to_legacy_engine(
    plugin: object,
    *,
    spec: LegacyAtomicL1BridgeSpec,
) -> AtomicL1PluginEngineAdapter:
    """Create the sole compatibility bridge into existing SCF/PCM machinery."""

    return AtomicL1PluginEngineAdapter.from_plugin(plugin, spec=spec)


__all__ = [
    "AtomicL1PluginEngineAdapter",
    "LegacyAtomicL1BridgeSpec",
    "adapt_atomic_l1_plugin_to_legacy_engine",
]
