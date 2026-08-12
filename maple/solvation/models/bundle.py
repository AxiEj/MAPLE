"""Validated composition of vacuum and optional response model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .base import (
    ElectronicSourceModel,
    ModelProvenance,
    VacuumEnergyModel,
    require_model_methods,
    validate_model_identity,
    validate_source_model_identity,
)

_VERSIONED_PROFILE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*$")


@dataclass(frozen=True, slots=True)
class ModelCapabilityDeclaration:
    """Structural provider declaration; release capabilities remain separate."""

    mutual_polarization: bool = False
    response_linearization: bool = False
    variational_functional: bool = False

    def __post_init__(self) -> None:
        for name in (
            "mutual_polarization",
            "response_linearization",
            "variational_functional",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool.")
        if self.response_linearization and not self.mutual_polarization:
            raise ValueError("response linearization requires mutual polarization.")
        if self.variational_functional:
            raise ValueError(
                "variational_functional is fail-closed until release evidence registry admission."
            )


@dataclass(frozen=True, slots=True)
class SoluteModelBundle:
    """One explicit Route-2 solute composition; never guesses compatibility."""

    vacuum: VacuumEnergyModel
    response: ElectronicSourceModel | None
    profile_id: str
    capabilities: ModelCapabilityDeclaration = field(
        default_factory=ModelCapabilityDeclaration
    )
    composite_validation_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, ModelCapabilityDeclaration):
            raise TypeError("capabilities must be ModelCapabilityDeclaration.")
        require_model_methods(self.vacuum, "evaluate_vacuum")
        if not isinstance(self.profile_id, str) or not _VERSIONED_PROFILE.fullmatch(
            self.profile_id
        ):
            raise ValueError(
                "profile_id must be an explicit lowercase versioned ID ending in -vN."
            )
        vacuum_provenance = validate_model_identity(self.vacuum)
        if vacuum_provenance.model_profile_id != self.profile_id:
            raise ValueError("vacuum model_profile_id must match bundle profile_id.")
        if self.response is None:
            if self.capabilities.mutual_polarization:
                raise ValueError("mutual polarization requires a response provider.")
            if self.composite_validation_evidence_sha256 is not None:
                raise ValueError(
                    "a vacuum-only bundle cannot declare composite validation evidence."
                )
            return

        require_model_methods(self.response, "evaluate_source")
        response_provenance = validate_source_model_identity(self.response)
        if response_provenance.model_profile_id != self.profile_id:
            raise ValueError("response model_profile_id must match bundle profile_id.")
        if self.capabilities.response_linearization:
            require_model_methods(
                self.response, "source_jvp", "source_vjp", "source_position_vjp"
            )
        compatibility_fields = (
            "model_profile_id",
            "model_family",
            "checkpoint_sha256",
            "upstream_version",
            "upstream_commit",
            "inference_code_sha256",
            "dtype",
            "device",
            "domain",
            "field_convention",
            "coordinate_frame_policy",
            "optimizer_parameter_groups_audited",
            "optimizer_audit_evidence_sha256",
        )
        mismatches = tuple(
            name
            for name in compatibility_fields
            if getattr(vacuum_provenance, name) != getattr(response_provenance, name)
        )
        if mismatches:
            raise ValueError(
                "mixed or incompatible model provenance is fail-closed until a "
                "registered composite profile and release evidence registry exist; "
                f"mismatched fields: {', '.join(mismatches)}."
            )
        if self.composite_validation_evidence_sha256 is not None:
            raise ValueError(
                "unregistered composite evidence digests cannot authorize a model bundle."
            )

    @property
    def vacuum_provenance(self) -> ModelProvenance:
        return self.vacuum.provenance

    @property
    def response_provenance(self) -> ModelProvenance | None:
        return None if self.response is None else self.response.provenance

    @property
    def mutual_polarization_capable(self) -> bool:
        return self.capabilities.mutual_polarization

    @property
    def field_response_linearization_capable(self) -> bool:
        return self.capabilities.response_linearization

    @property
    def variational_functional_admitted(self) -> bool:
        """Release Tier V cannot be opened by a provider declaration."""

        return False


__all__ = ["ModelCapabilityDeclaration", "SoluteModelBundle"]
