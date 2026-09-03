"""Explicit source/receiver semantics for public Route-2 profiles.

This contract separates a continuum pairing fact from a claim about the
learned electronic response.  In particular, a point-multipole/local-jet
continuum pairing does not make the MACE-POLAR fixed point a stationary
electronic free energy, while the public exact-GTO receiver is known not to be
the transpose of its point-multipole source.  Capability is profile-specific:
most profiles remain energy-only, while the separately versioned fixed-topology
profile may expose a bounded, per-geometry-certified operational-scalar force.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from maple.function.route2_smd_profiles import route2_smd_profile_spec

ROUTE2_SOURCE_RECEIVER_CONTRACT_VERSION = "route2-source-receiver-contract-v1"

ContinuumPairingStatus = Literal[
    "point-multipole-local-jet-dual",
    "known-nonconjugate-point-source-gto-receiver",
    "energy-conjugate-local-jet-known-nonpassive",
]

Route2PublicCapability = Literal[
    "experimental-energy-only",
    "bounded-experimental-energy-and-conservative-forces",
    "known-invalid-diagnostic-energy-only",
    "known-invalid-diagnostic-energy-force-numerical-hessian",
]


@dataclass(frozen=True)
class Route2SourceReceiverContract:
    """The physical and public-capability scope of one Route-2 profile."""

    profile: str
    solute_source: str
    reaction_field_receiver: str
    continuum_pairing_status: ContinuumPairingStatus
    continuum_pairing_established: bool
    common_stationary_electronic_functional_established: bool
    public_capability: Route2PublicCapability
    next_required_physical_gate: str
    prohibited_claims: tuple[str, ...]
    contract_version: str = ROUTE2_SOURCE_RECEIVER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.profile,
                self.solute_source,
                self.reaction_field_receiver,
                self.public_capability,
                self.next_required_physical_gate,
                self.contract_version,
            )
        ):
            raise ValueError("Source/receiver contract requires nonempty strings.")
        if self.continuum_pairing_status not in (
            "point-multipole-local-jet-dual",
            "known-nonconjugate-point-source-gto-receiver",
            "energy-conjugate-local-jet-known-nonpassive",
        ):
            raise ValueError("Unsupported Route-2 continuum pairing status.")
        if self.continuum_pairing_established != (
            self.continuum_pairing_status
            != "known-nonconjugate-point-source-gto-receiver"
        ):
            raise ValueError("Continuum pairing status and established flag disagree.")
        if self.common_stationary_electronic_functional_established:
            raise ValueError(
                "Legacy Route-2 profiles do not establish a common stationary "
                "electronic functional."
            )
        if self.public_capability not in (
            "experimental-energy-only",
            "bounded-experimental-energy-and-conservative-forces",
            "known-invalid-diagnostic-energy-only",
            "known-invalid-diagnostic-energy-force-numerical-hessian",
        ):
            raise ValueError("Unsupported Route-2 public capability.")
        if not self.prohibited_claims or not all(
            isinstance(claim, str) and claim.strip() for claim in self.prohibited_claims
        ):
            raise ValueError("Source/receiver contract requires prohibited claims.")
        if self.contract_version != ROUTE2_SOURCE_RECEIVER_CONTRACT_VERSION:
            raise ValueError("Unsupported Route-2 source/receiver contract version.")

    def as_provenance(self) -> dict[str, object]:
        """Return only JSON-native fields for manifests and result records."""

        return {
            "contract_version": self.contract_version,
            "profile": self.profile,
            "solute_source": self.solute_source,
            "reaction_field_receiver": self.reaction_field_receiver,
            "continuum_pairing_status": self.continuum_pairing_status,
            "continuum_pairing_established": self.continuum_pairing_established,
            "common_stationary_electronic_functional_established": (
                self.common_stationary_electronic_functional_established
            ),
            "public_capability": self.public_capability,
            "next_required_physical_gate": self.next_required_physical_gate,
            "prohibited_claims": list(self.prohibited_claims),
        }


def route2_source_receiver_contract(
    profile: str,
) -> Route2SourceReceiverContract:
    """Return the fail-closed semantic contract selected by a profile."""

    spec = route2_smd_profile_spec(profile)
    common_prohibited_claims = (
        "variational SCRF",
        "common-energy stationary electronic state",
    )
    energy_only_prohibited_claims = common_prohibited_claims + (
        "solution-phase PES",
        "analytic solution-phase forces",
    )
    if spec.known_nonpassive_diagnostic:
        derivative_capability = spec.diagnostic_derivative_eligible
        return Route2SourceReceiverContract(
            profile=spec.name,
            solute_source=spec.solute_source,
            reaction_field_receiver=spec.reaction_field_projector,
            continuum_pairing_status=("energy-conjugate-local-jet-known-nonpassive"),
            continuum_pairing_established=True,
            common_stationary_electronic_functional_established=False,
            public_capability=(
                "known-invalid-diagnostic-energy-force-numerical-hessian"
                if derivative_capability
                else "known-invalid-diagnostic-energy-only"
            ),
            next_required_physical_gate=(
                "replace-checkpoint-and-pass-electronic-passivity"
            ),
            prohibited_claims=(
                (
                    "physically valid implicit-solvent prediction",
                    "variational SCRF",
                    "stable electronic polarization",
                    "chemical accuracy",
                    "physically valid solution-phase PES",
                    "physically predictive solution-phase forces",
                    "physically predictive geometry optimization",
                    "molecular dynamics",
                )
                if derivative_capability
                else (
                    "physically valid implicit-solvent prediction",
                    "variational SCRF",
                    "stable electronic polarization",
                    "chemical accuracy",
                    "solution-phase PES",
                    "analytic solution-phase forces",
                    "geometry optimization",
                    "molecular dynamics",
                )
            ),
        )
    if spec.reaction_field_projector == "exact-gto-v1":
        return Route2SourceReceiverContract(
            profile=spec.name,
            solute_source=spec.solute_source,
            reaction_field_receiver=spec.reaction_field_projector,
            continuum_pairing_status=("known-nonconjugate-point-source-gto-receiver"),
            continuum_pairing_established=False,
            common_stationary_electronic_functional_established=False,
            public_capability="experimental-energy-only",
            next_required_physical_gate=(
                "same-basis-gto-galerkin-source-receiver-and-common-scalar"
            ),
            prohibited_claims=energy_only_prohibited_claims,
        )
    if spec.reaction_field_projector == "local-jet":
        if spec.force_release_eligible:
            return Route2SourceReceiverContract(
                profile=spec.name,
                solute_source=spec.solute_source,
                reaction_field_receiver=spec.reaction_field_projector,
                continuum_pairing_status="point-multipole-local-jet-dual",
                continuum_pairing_established=True,
                common_stationary_electronic_functional_established=False,
                public_capability=(
                    "bounded-experimental-energy-and-conservative-forces"
                ),
                next_required_physical_gate=(
                    "common-electronic-scalar-and-broader-force-domain-validation"
                ),
                prohibited_claims=common_prohibited_claims
                + (
                    "unbounded solution-phase PES",
                    "universal analytic solution-phase forces",
                ),
            )
        return Route2SourceReceiverContract(
            profile=spec.name,
            solute_source=spec.solute_source,
            reaction_field_receiver=spec.reaction_field_projector,
            continuum_pairing_status="point-multipole-local-jet-dual",
            continuum_pairing_established=True,
            common_stationary_electronic_functional_established=False,
            public_capability="experimental-energy-only",
            next_required_physical_gate=(
                "common-electronic-scalar-energy-density-conjugacy-and-kkt"
            ),
            prohibited_claims=energy_only_prohibited_claims,
        )
    raise ValueError(
        "Route-2 profile uses an unrecognized reaction-field receiver: "
        f"{spec.reaction_field_projector}."
    )


__all__ = [
    "ROUTE2_SOURCE_RECEIVER_CONTRACT_VERSION",
    "Route2PublicCapability",
    "Route2SourceReceiverContract",
    "route2_source_receiver_contract",
]
