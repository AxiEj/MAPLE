"""Thin profile builders for same-scalar harmonic-ddPCM operational PESs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2,
    OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2,
)
from maple.solvation.continuum.harmonic_ddpcm_gaussian import (
    build_mace_polar_gaussian_harmonic_ddpcm_snapshot,
)
from maple.solvation.continuum.harmonic_ddpcm_hybrid import (
    build_harmonic_ddpcm_hybrid_snapshot,
)
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.operational_pes import (
    OperationalEquationLedgerBundle,
)
from maple.solvation.coupling.permanent_induced_ledgers import (
    HybridHarmonicDDPCMPhi0Ledger,
)
from maple.solvation.coupling.permanent_induced_state import (
    PermanentInducedOperationalStateEquation,
    PermanentInducedResponseModel,
)
from maple.solvation.coupling.separated_ledgers import (
    HarmonicDDPCMFrozenVacuumLedger,
)
from maple.solvation.coupling.separated_state import (
    SeparatedElectronicResponseProvider,
    SeparatedOperationalStateEquation,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.coupling.state_equation import provider_behavior_sha256
from maple.solvation.models.base import atom_count, model_charge_and_multiplicity
from maple.solvation.release.field_semantics import FieldSemanticsManifest


PURE_GAUSSIAN_BUILDER_PROVIDER_ID = (
    "maple.route2.builder.macepolar-gto1p5-harmonic-ddpcm-phi0.impl.v2"
)
HYBRID_GENERAL_SOURCE_BUILDER_PROVIDER_ID = (
    "maple.route2.builder.macedp-polar-general-source-harmonic-ddpcm-phi0.impl.v2"
)
_MODULE_PATH = Path(__file__).resolve()


def _implementation_sha256() -> str:
    return hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()


class MACEPolarGaussianHarmonicDDPCMBuilder:
    """Build pure original-source MACE-POLAR + Gaussian-source ddPCM Phi0."""

    __slots__ = (
        "_configuration_sha256",
        "_continuum",
        "_electronic",
        "_field_semantics",
        "_receiver_order",
        "_sealed",
    )

    provider_id = PURE_GAUSSIAN_BUILDER_PROVIDER_ID
    scalar_id = (
        OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2
    )

    def __init__(
        self,
        *,
        electronic: SeparatedElectronicResponseProvider,
        continuum: object,
        field_semantics_manifest: FieldSemanticsManifest,
        receiver_radial_quadrature_order: int = 128,
    ) -> None:
        if not isinstance(electronic, SeparatedElectronicResponseProvider):
            raise TypeError(
                "electronic must satisfy SeparatedElectronicResponseProvider."
            )
        for name in (
            "configuration_sha256",
            "debug_geometry_matrices",
        ):
            if not callable(getattr(continuum, name, None)):
                raise TypeError(f"continuum requires callable {name}().")
        if not isinstance(field_semantics_manifest, FieldSemanticsManifest):
            raise TypeError("field_semantics_manifest has the wrong type.")
        if type(receiver_radial_quadrature_order) is not int or not (
            1 <= receiver_radial_quadrature_order <= 4096
        ):
            raise ValueError("receiver radial quadrature order is invalid.")
        if (
            field_semantics_manifest.model_provider_id != electronic.provider_id
            or field_semantics_manifest.model_profile_id
            != electronic.model_profile_id
        ):
            raise ValueError("field semantics do not bind the electronic provider.")
        electronic.configuration_sha256()
        continuum.configuration_sha256()
        object.__setattr__(self, "_electronic", electronic)
        object.__setattr__(self, "_continuum", continuum)
        object.__setattr__(self, "_field_semantics", field_semantics_manifest)
        object.__setattr__(self, "_receiver_order", receiver_radial_quadrature_order)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("pure Gaussian operational builder is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "pure-macepolar-gaussian-harmonic-ddpcm-builder-v2",
                "provider_id": self.provider_id,
                "scalar_id": self.scalar_id,
                "electronic_configuration_sha256": (
                    self._electronic.configuration_sha256()
                ),
                "electronic_provenance_sha256": self._electronic.provenance_sha256,
                "electronic_behavior_sha256": provider_behavior_sha256(
                    self._electronic,
                    (
                        "configuration_sha256",
                        "evaluate_source",
                        "field_jvp",
                        "field_vjp",
                        "coordinate_vjp",
                        "evaluate_energy",
                        "coordinate_gradient",
                    ),
                    label="pure_macepolar_operational_builder_electronic",
                ),
                "continuum_configuration_sha256": (
                    self._continuum.configuration_sha256()
                ),
                "continuum_provenance_sha256": self._continuum.provenance_sha256,
                "field_semantics_sha256": (
                    self._field_semantics.configuration_sha256()
                ),
                "receiver_radial_quadrature_order": self._receiver_order,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("pure Gaussian operational builder drifted.")
        return current

    def build(self, geometry: object) -> OperationalEquationLedgerBundle:
        self.configuration_sha256()
        count = atom_count(geometry)
        charge, _multiplicity = model_charge_and_multiplicity(geometry)
        coordinates = AffineChargeCoordinates(
            atom_count=count,
            total_charge=float(charge),
            source_space=ATOMIC_L1_SOURCE_SPACE,
        )
        continuum = build_mace_polar_gaussian_harmonic_ddpcm_snapshot(
            self._continuum,
            geometry,
            receiver_radial_quadrature_order=self._receiver_order,
        )
        equation = SeparatedOperationalStateEquation(
            coordinates, self._electronic, continuum
        )
        ledger = HarmonicDDPCMFrozenVacuumLedger(
            equation=equation,
            vacuum=self._electronic,
            field_semantics_manifest=self._field_semantics,
        )
        return OperationalEquationLedgerBundle(
            equation=equation,
            ledger=ledger,
            scalar_id=self.scalar_id,
            topology_id=continuum.topology_sha256,
        )


class MACE_MDPPolarGeneralSourceHarmonicDDPCMBuilder:
    """Build point-permanent/Gaussian-induced MDP+POLAR ddPCM Phi0."""

    __slots__ = (
        "_configuration_sha256",
        "_continuum",
        "_hybrid",
        "_receiver_order",
        "_sealed",
    )

    provider_id = HYBRID_GENERAL_SOURCE_BUILDER_PROVIDER_ID
    scalar_id = OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2

    def __init__(
        self,
        *,
        hybrid: PermanentInducedResponseModel,
        continuum: object,
        receiver_radial_quadrature_order: int = 128,
    ) -> None:
        if not isinstance(hybrid, PermanentInducedResponseModel):
            raise TypeError("hybrid must satisfy PermanentInducedResponseModel.")
        if not callable(getattr(hybrid, "prepare", None)):
            raise TypeError("hybrid requires callable prepare().")
        for name in (
            "configuration_sha256",
            "debug_geometry_matrices",
        ):
            if not callable(getattr(continuum, name, None)):
                raise TypeError(f"continuum requires callable {name}().")
        if type(receiver_radial_quadrature_order) is not int or not (
            1 <= receiver_radial_quadrature_order <= 4096
        ):
            raise ValueError("receiver radial quadrature order is invalid.")
        hybrid.configuration_sha256()
        continuum.configuration_sha256()
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_continuum", continuum)
        object.__setattr__(self, "_receiver_order", receiver_radial_quadrature_order)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid operational builder is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "mdp-polar-general-source-harmonic-ddpcm-builder-v2",
                "provider_id": self.provider_id,
                "scalar_id": self.scalar_id,
                "hybrid_configuration_sha256": self._hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": self._hybrid.provenance_sha256,
                "hybrid_behavior_sha256": provider_behavior_sha256(
                    self._hybrid,
                    (
                        "configuration_sha256",
                        "prepare",
                        "induced_source",
                        "field_jvp",
                        "field_vjp",
                        "permanent_source_position_vjp",
                        "induced_source_position_vjp",
                        "vacuum_energy_ev",
                        "vacuum_forces_ev_per_angstrom",
                    ),
                    label="hybrid_operational_builder_model",
                ),
                "continuum_configuration_sha256": (
                    self._continuum.configuration_sha256()
                ),
                "continuum_provenance_sha256": self._continuum.provenance_sha256,
                "receiver_radial_quadrature_order": self._receiver_order,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid operational builder drifted.")
        return current

    def build(self, geometry: object) -> OperationalEquationLedgerBundle:
        self.configuration_sha256()
        count = atom_count(geometry)
        coordinates = AffineChargeCoordinates(
            atom_count=count,
            total_charge=0.0,
            source_space=ATOMIC_L1_SOURCE_SPACE,
        )
        anchor = self._hybrid.prepare(geometry)
        continuum = build_harmonic_ddpcm_hybrid_snapshot(
            self._continuum,
            geometry,
            receiver_radial_quadrature_order=self._receiver_order,
        )
        equation = PermanentInducedOperationalStateEquation(
            coordinates,
            self._hybrid,
            anchor,
            continuum,
        )
        ledger = HybridHarmonicDDPCMPhi0Ledger(equation)
        return OperationalEquationLedgerBundle(
            equation=equation,
            ledger=ledger,
            scalar_id=self.scalar_id,
            topology_id=continuum.topology_sha256,
        )


__all__ = [
    "HYBRID_GENERAL_SOURCE_BUILDER_PROVIDER_ID",
    "MACEPolarGaussianHarmonicDDPCMBuilder",
    "MACE_MDPPolarGeneralSourceHarmonicDDPCMBuilder",
    "PURE_GAUSSIAN_BUILDER_PROVIDER_ID",
]
