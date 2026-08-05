"""MACE-POLAR compatibility plug-in for the Route-2 M0 contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .route2_field_state import LocalReactionField, ReactionFieldDrive
from .route2_mace_p0 import MACEP0AuditCertificate
from .route2_plugin_contracts import (
    PluginCapabilityDeclaration,
    PluginElectronicState,
    PluginProvenance,
)
from .route2_plugin_errors import PluginContractError
from .route2_plugin_spaces import (
    ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
    ATOMIC_L1_PLUGIN_SOURCE_SPACE,
    ElectronicSourceSpace,
    FieldDualSpace,
)


@dataclass(frozen=True)
class MACEPolarPlugin:
    """MACE-POLAR implementation built on the established AtomicL1 adapter.

    The field convention and optimizer coverage deliberately default to
    ``unattested`` until P0 evidence is recorded.  The class is therefore a
    compatible source provider, not proof that MACE-POLAR may run an SCF,
    force, or variational route under the new contract.
    """

    legacy_adapter: Any
    provenance: PluginProvenance
    capabilities: PluginCapabilityDeclaration
    plugin_id: str = "mace-polar-1"
    source_space: ElectronicSourceSpace = ATOMIC_L1_PLUGIN_SOURCE_SPACE
    field_dual_space: FieldDualSpace = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE

    @classmethod
    def from_calculator(
        cls,
        calculator: object,
        *,
        p0_certificate: MACEP0AuditCertificate | None = None,
    ) -> "MACEPolarPlugin":
        # Local import prevents the new contract from becoming an import cycle
        # for the legacy calculator module.
        from .route2_electronic_model import (
            AtomicL1CalculatorAdapter,
            Route2ElectronicModelDescriptor,
        )

        descriptor = getattr(calculator, "route2_electronic_model_descriptor", None)
        if not isinstance(descriptor, Route2ElectronicModelDescriptor):
            raise PluginContractError(
                "MACEPolarPlugin requires the explicit legacy Route-2 descriptor."
            )
        if descriptor.model_family != "mace-polar-1":
            raise PluginContractError(
                "MACEPolarPlugin only accepts the MACE-POLAR family."
            )
        checkpoint_sha = "unattestable"
        raw_provenance = descriptor.as_provenance().get("provenance", {})
        if isinstance(raw_provenance, Mapping):
            checkpoint = raw_provenance.get("checkpoint")
            if isinstance(checkpoint, Mapping) and isinstance(
                checkpoint.get("sha256"), str
            ):
                checkpoint_sha = checkpoint["sha256"]
        old_capabilities = descriptor.capabilities
        atomic_numbers = getattr(calculator, "atomic_numbers", None)
        supported_atomic_numbers = None
        if atomic_numbers is not None:
            supported_atomic_numbers = frozenset(map(int, atomic_numbers))
        field_convention = "unattested"
        inference_code_sha = "unattestable"
        total_charge_domain = None
        certificate_provenance: Mapping[str, object] | None = None
        if p0_certificate is not None:
            if not isinstance(p0_certificate, MACEP0AuditCertificate):
                raise PluginContractError(
                    "MACE-POLAR P0 evidence must be a validated audit certificate."
                )
            p0_certificate.verify_calculator(calculator)
            field_convention = p0_certificate.field_convention
            inference_code_sha = p0_certificate.inference_code_sha256
            supported_atomic_numbers = p0_certificate.audited_atomic_numbers
            total_charge_domain = p0_certificate.total_charge_domain_e
            certificate_provenance = {
                "artifact_id": p0_certificate.artifact_id,
                "artifact_path": p0_certificate.artifact_path,
                "artifact_sha256": p0_certificate.artifact_sha256,
                "blocked_routes": p0_certificate.blocked_routes,
                "coordinate_frame_policy": (p0_certificate.coordinate_frame_policy),
                "model_field_evaluator": (p0_certificate.model_field_evaluator),
            }
        return cls(
            legacy_adapter=AtomicL1CalculatorAdapter(
                calculator=calculator, descriptor=descriptor
            ),
            provenance=PluginProvenance(
                checkpoint_sha256=checkpoint_sha,
                training_code_sha="unattestable",
                inference_code_sha=inference_code_sha,
                representation=ATOMIC_L1_PLUGIN_SOURCE_SPACE.representation,
                supported_atomic_numbers=supported_atomic_numbers,
                total_charge_domain=total_charge_domain,
                field_convention=field_convention,
                potential_unit=ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.potential_unit,
                gradient_unit=ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.gradient_unit,
                energy_semantics=descriptor.energy_semantics,
                optimizer_coverage="unattestable",
                model_field_evaluator=str(
                    getattr(calculator, "long_range_evaluator_profile", "unattested")
                ),
                coordinate_frame_policy=str(
                    getattr(
                        calculator,
                        "route2_mace_geometry_frame_policy",
                        "unattested",
                    )
                ),
                extra={
                    "legacy_descriptor": descriptor.as_provenance(),
                    "p0_certificate": certificate_provenance,
                },
            ),
            capabilities=PluginCapabilityDeclaration(
                field_response=bool(old_capabilities.response_projectors),
                fixed_field_forces=bool(old_capabilities.fixed_field_force_projectors),
                source_position_vjp=bool(old_capabilities.position_vjp_projectors),
                feature_vjp=False,
                variational_functional=False,
            ),
        )

    @staticmethod
    def _legacy_drive(field: LocalReactionField) -> ReactionFieldDrive:
        if field.gauge != "continuum-zero-at-infinity":
            raise PluginContractError(
                "The legacy MACE-POLAR adapter only supports the continuum-zero-at-infinity gauge."
            )
        return ReactionFieldDrive.local_jet(field.as_nodewise_jet())

    def _state(self, value: object, *, atom_count: int) -> PluginElectronicState:
        source = self.source_space.validate_source(
            getattr(value, "density_coefficients"), atom_count=atom_count
        )
        return PluginElectronicState(
            energy_ev=float(getattr(value, "energy_ev")),
            source=source,
            dipole_e_angstrom=np.asarray(
                getattr(value, "dipole_e_angstrom"), dtype=float
            ),
            fixed_field_forces_ev_per_angstrom=getattr(
                value, "fixed_field_forces_ev_per_angstrom", None
            ),
        )

    def cached_source_state(
        self, atoms: Any, *, require_forces: bool = False
    ) -> PluginElectronicState | None:
        state = self.legacy_adapter.cached_state(atoms, require_forces=require_forces)
        return None if state is None else self._state(state, atom_count=len(atoms))

    def evaluate_source_state(
        self,
        atoms: Any,
        field: LocalReactionField | None,
        *,
        compute_forces: bool = False,
    ) -> tuple[PluginElectronicState, Mapping[str, object]]:
        state, provenance = self.legacy_adapter.evaluate_state(
            atoms,
            None if field is None else self._legacy_drive(field),
            compute_forces=compute_forces,
        )
        return self._state(state, atom_count=len(atoms)), provenance

    def linearize_source_response(self, atoms: Any, field: LocalReactionField) -> Any:
        if not self.capabilities.field_response:
            raise PluginContractError(
                "MACE-POLAR plug-in did not declare source-response derivatives."
            )
        return self.legacy_adapter.linearize_source_response(
            atoms, self._legacy_drive(field)
        )

    def source_position_vjp(
        self,
        atoms: Any,
        field: LocalReactionField,
        *,
        source_cotangent: np.ndarray,
    ) -> np.ndarray:
        if not self.capabilities.source_position_vjp:
            raise PluginContractError(
                "MACE-POLAR plug-in did not declare source-position VJP."
            )
        return self.legacy_adapter.source_position_vjp(
            atoms,
            self._legacy_drive(field),
            source_cotangent=source_cotangent,
        )

    def feature_vjp(
        self, atoms: Any, field: LocalReactionField, *, feature_cotangent: np.ndarray
    ) -> np.ndarray:
        del atoms, field, feature_cotangent
        raise PluginContractError("MACE-POLAR has no admitted feature VJP in P-1.")

    def fixed_field_forces(self, atoms: Any, field: LocalReactionField) -> np.ndarray:
        state, _ = self.evaluate_source_state(atoms, field, compute_forces=True)
        if state.fixed_field_forces_ev_per_angstrom is None:
            raise PluginContractError("MACE-POLAR did not provide fixed-field forces.")
        return state.fixed_field_forces_ev_per_angstrom

    def route2_electronic_model_adapter(self) -> Any:
        """Reuse the old engine through the existing explicit adapter boundary."""
        return self.legacy_adapter


__all__ = ["MACEPolarPlugin"]
