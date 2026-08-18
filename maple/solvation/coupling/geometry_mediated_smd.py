"""SMD-CDS companion for the fixed-charge AIMNet2/ddPCM baseline.

The electronic/continuum part remains the explicit one-shot composition

``R -> (E_AIMNet2(R), q_NQE(R)) -> G_ddPCM(R, q_NQE(R))``.

This module adds the official PySCF SMD cavity--dispersion--solvent-structure
energy and the analytic coordinate gradient returned with that same energy.
It never supplies a field to AIMNet2 and introduces no electronic fixed point.

The resulting profile is deliberately disabled.  It closes the missing
same-ledger nonpolar *implementation* for force diagnostics.  AIMNet2's
published charges are gas-phase Hirshfeld-like monopoles, not a continuous
QM density or a solvent-polarized response; consequently the composite is not
strict original SMD and cannot inherit SMD accuracy.  It also does not establish
a globally smooth PES, chemical accuracy, Hessians, dynamics, or a public
capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.pyscf_runtime import (
    TESTED_PYSCF_VERSION,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.route2_solvents import route2_solvent_spec
from maple.solvation.api.profiles import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_PROFILE_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.api.scalar_registry import (
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1,
    get_scalar_definition,
)
from maple.solvation.models.base import atom_count

from .geometry_mediated import (
    GeometryMediatedElectrostaticScalar,
    GeometryMediatedEnergyEvaluation,
    GeometryMediatedEvaluation,
)

PYSCF_WATER_SMD_CDS_NONPOLAR_PROFILE_ID = (
    "pyscf-2.13.1-water-smd-cds-analytic-gradient-v1"
)
PYSCF_SMD_CDS_NONPOLAR_PROVIDER_ID = (
    "maple.route2.nonpolar.pyscf-smd-cds-energy-gradient.impl.v1"
)
PYSCF_MULTISOLVENT_SMD_CDS_NONPOLAR_PROFILE_ID = (
    "pyscf-2.13.1-multisolvent-smd-cds-analytic-gradient-v1"
)
PYSCF_MULTISOLVENT_SMD_CDS_NONPOLAR_PROVIDER_ID = (
    "maple.route2.nonpolar.pyscf-multisolvent-smd-cds-energy-gradient.impl.v1"
)


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _readonly_gradient(values: object, *, count: int, name: str) -> np.ndarray:
    gradient = np.asarray(values, dtype=float)
    if gradient.shape != (count, 3) or not np.all(np.isfinite(gradient)):
        raise ValueError(f"{name} must be finite with shape {(count, 3)}.")
    result = np.array(gradient, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GeometryNonpolarEvaluation:
    """One nonpolar energy and the coordinate gradient of that exact energy."""

    provider_id: str
    nonpolar_profile_id: str
    configuration_sha256: str
    energy_eV: float
    gradient_eV_per_A: np.ndarray
    runtime_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("provider_id", "nonpolar_profile_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        configuration = str(self.configuration_sha256).lower()
        if len(configuration) != 64 or any(
            character not in "0123456789abcdef" for character in configuration
        ):
            raise ValueError("configuration_sha256 must be a SHA256 string.")
        energy = float(self.energy_eV)
        if not np.isfinite(energy):
            raise ValueError("nonpolar energy must be finite.")
        gradient = np.asarray(self.gradient_eV_per_A, dtype=float)
        if (
            gradient.ndim != 2
            or gradient.shape[0] < 1
            or gradient.shape[1] != 3
            or not np.all(np.isfinite(gradient))
        ):
            raise ValueError("nonpolar gradient must be finite with shape (N,3).")
        frozen_gradient = np.array(gradient, copy=True)
        frozen_gradient.setflags(write=False)
        provenance = dict(self.runtime_provenance)
        if any(not isinstance(key, str) or not key for key in provenance):
            raise ValueError("nonpolar runtime provenance keys must be non-empty.")
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "gradient_eV_per_A", frozen_gradient)
        object.__setattr__(self, "runtime_provenance", MappingProxyType(provenance))


@dataclass(frozen=True, slots=True)
class PySCFSMDCDSNonpolarFunctional:
    """Pinned official PySCF SMD-CDS energy/gradient provider."""

    solvent: str = "water"

    provider_id = PYSCF_SMD_CDS_NONPOLAR_PROVIDER_ID
    nonpolar_profile_id = PYSCF_WATER_SMD_CDS_NONPOLAR_PROFILE_ID
    derivatives_generated_from_same_scalar = True
    coordinate_hessian_available = False

    def __post_init__(self) -> None:
        spec = route2_solvent_spec(self.solvent)
        if spec.name != "water":
            raise ValueError(
                "The registered AIMNet2 harmonic-ddPCM SMD-CDS profile is "
                "water-only."
            )
        object.__setattr__(self, "solvent", spec.name)

    def configuration_sha256(self) -> str:
        spec = route2_solvent_spec(self.solvent)
        return _hash(
            {
                "provider_id": self.provider_id,
                "nonpolar_profile_id": self.nonpolar_profile_id,
                "pyscf_version": TESTED_PYSCF_VERSION,
                "upstream_entrypoint": "pyscf.solvent.smd.get_cds_legacy",
                "solvent": spec.name,
                "pyscf_smd_solvent": spec.pyscf_smd_name,
                "solvent_descriptors": spec.descriptors.as_pyscf_tuple(),
                "energy_unit": "eV",
                "gradient_unit": "eV/angstrom",
                "coordinate_hessian_available": False,
            }
        )

    def evaluate(self, geometry: Any) -> GeometryNonpolarEvaluation:
        count = atom_count(geometry)
        symbols_getter = getattr(geometry, "get_chemical_symbols", None)
        positions_getter = getattr(geometry, "get_positions", None)
        if not callable(symbols_getter) or not callable(positions_getter):
            raise TypeError(
                "PySCF SMD-CDS requires geometry chemical symbols and positions."
            )
        symbols = tuple(str(value) for value in symbols_getter())
        positions = np.asarray(positions_getter(), dtype=float)
        if len(symbols) != count or positions.shape != (count, 3):
            raise ValueError("geometry symbols and positions disagree on atom count.")
        upstream = pyscf_smd_cds(symbols, positions, solvent=self.solvent)
        return GeometryNonpolarEvaluation(
            provider_id=self.provider_id,
            nonpolar_profile_id=self.nonpolar_profile_id,
            configuration_sha256=self.configuration_sha256(),
            energy_eV=float(upstream.energy_hartree) * Hartree,
            gradient_eV_per_A=(
                np.asarray(upstream.position_gradient_hartree_per_angstrom) * Hartree
            ),
            runtime_provenance=dict(upstream.runtime_provenance),
        )


@dataclass(frozen=True, slots=True)
class PySCFMultisolventSMDCDSNonpolarFunctional(PySCFSMDCDSNonpolarFunctional):
    """Pinned PySCF SMD-CDS energy/gradient for a registered Route-2 solvent."""

    provider_id = PYSCF_MULTISOLVENT_SMD_CDS_NONPOLAR_PROVIDER_ID
    nonpolar_profile_id = PYSCF_MULTISOLVENT_SMD_CDS_NONPOLAR_PROFILE_ID

    def __post_init__(self) -> None:
        spec = route2_solvent_spec(self.solvent)
        object.__setattr__(self, "solvent", spec.name)


@dataclass(frozen=True, slots=True)
class GeometryMediatedSMDTotalEnergyEvaluation:
    """Closed aqueous vacuum + ddPCM electrostatic + SMD-CDS ledger."""

    scalar_id: str
    profile_id: str
    vacuum_energy_eV: float
    continuum_energy_eV: float
    nonpolar_energy_eV: float
    total_energy_eV: float

    def __post_init__(self) -> None:
        for name in ("scalar_id", "profile_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty.")
        leaves = np.asarray(
            (
                self.vacuum_energy_eV,
                self.continuum_energy_eV,
                self.nonpolar_energy_eV,
                self.total_energy_eV,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(leaves)):
            raise ValueError("total SMD energy leaves must be finite.")
        if not np.isclose(float(np.sum(leaves[:3])), leaves[3], rtol=0.0, atol=1.0e-10):
            raise ValueError("total SMD energy ledger does not close.")


@dataclass(frozen=True, slots=True)
class GeometryMediatedSMDTotalEvaluation:
    """First derivative of the complete registered aqueous candidate scalar."""

    energy: GeometryMediatedSMDTotalEnergyEvaluation
    electrostatic: GeometryMediatedEvaluation
    nonpolar: GeometryNonpolarEvaluation
    nonpolar_gradient_eV_per_A: np.ndarray
    total_gradient_eV_per_A: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.energy, GeometryMediatedSMDTotalEnergyEvaluation):
            raise TypeError("energy must be a total SMD energy evaluation.")
        if not isinstance(self.electrostatic, GeometryMediatedEvaluation):
            raise TypeError("electrostatic must be a geometry-mediated evaluation.")
        if not isinstance(self.nonpolar, GeometryNonpolarEvaluation):
            raise TypeError("nonpolar must be a geometry nonpolar evaluation.")
        count = self.electrostatic.source.shape[0]
        nonpolar_gradient = _readonly_gradient(
            self.nonpolar_gradient_eV_per_A,
            count=count,
            name="nonpolar_gradient_eV_per_A",
        )
        total = _readonly_gradient(
            self.total_gradient_eV_per_A,
            count=count,
            name="total_gradient_eV_per_A",
        )
        expected = self.electrostatic.total_gradient_eV_per_A + nonpolar_gradient
        if not np.allclose(expected, total, rtol=0.0, atol=2.0e-10):
            raise ValueError("total SMD gradient ledger does not close.")
        if not np.array_equal(nonpolar_gradient, self.nonpolar.gradient_eV_per_A):
            raise ValueError("nonpolar result and total gradient ledger disagree.")
        object.__setattr__(self, "nonpolar_gradient_eV_per_A", nonpolar_gradient)
        object.__setattr__(self, "total_gradient_eV_per_A", total)

    @property
    def source(self) -> np.ndarray:
        return self.electrostatic.source

    @property
    def reaction_field(self) -> np.ndarray:
        return self.electrostatic.reaction_field

    @property
    def intrinsic_gradient_eV_per_A(self) -> np.ndarray:
        return self.electrostatic.intrinsic_gradient_eV_per_A

    @property
    def continuum_fixed_source_gradient_eV_per_A(self) -> np.ndarray:
        return self.electrostatic.continuum_fixed_source_gradient_eV_per_A

    @property
    def source_response_gradient_eV_per_A(self) -> np.ndarray:
        return self.electrostatic.source_response_gradient_eV_per_A

    @property
    def electrostatic_total_gradient_eV_per_A(self) -> np.ndarray:
        return self.electrostatic.total_gradient_eV_per_A

    @property
    def reciprocity_gradient_error_eV_per_source_unit(self) -> float:
        return self.electrostatic.reciprocity_gradient_error_eV_per_source_unit

    @property
    def reciprocity_audit(self):
        return self.electrostatic.reciprocity_audit

    @property
    def forces_eV_per_A(self) -> np.ndarray:
        result = -np.asarray(self.total_gradient_eV_per_A)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class GeometryMediatedSMDTotalScalar:
    """Disabled total aqueous scalar with one-shot AIMNet2 charges."""

    electrostatic: GeometryMediatedElectrostaticScalar
    nonpolar: PySCFSMDCDSNonpolarFunctional
    scalar_id: str = (
        CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_V1
    )
    profile_id: str = (
        CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
    )
    _construction_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.electrostatic, GeometryMediatedElectrostaticScalar):
            raise TypeError(
                "total SMD scalar requires GeometryMediatedElectrostaticScalar."
            )
        if not isinstance(self.nonpolar, PySCFSMDCDSNonpolarFunctional):
            raise TypeError("total SMD scalar requires PySCFSMDCDSNonpolarFunctional.")
        if (
            self.electrostatic.scalar_id
            != CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
            or self.electrostatic.profile_id
            != CANDIDATE_AIMNET2_FROZEN_CHARGE_WATER_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
        ):
            raise ValueError(
                "total SMD scalar requires the exact water-bound frozen-charge "
                "harmonic-ddPCM electrostatic child."
            )
        definition = get_scalar_definition(self.scalar_id)
        profile = get_solvation_profile(self.profile_id)
        child_definition = get_scalar_definition(self.electrostatic.scalar_id)
        child_profile = get_solvation_profile(self.electrostatic.profile_id)
        if definition.implementation_entry_point != (
            "maple.solvation.coupling.geometry_mediated_smd:"
            "GeometryMediatedSMDTotalScalar"
        ):
            raise ValueError("registered total SMD implementation entry drifted.")
        if (
            profile.scalar_id != self.scalar_id
            or profile.state_equation_id != definition.state_equation_id
            or profile.model_profile != child_profile.model_profile
            or profile.continuum_profile != child_definition.continuum_profile
            or profile.cavity_profile != child_definition.cavity_profile
            or profile.nonpolar_profile != self.nonpolar.nonpolar_profile_id
            or definition.nonpolar_profile != self.nonpolar.nonpolar_profile_id
        ):
            raise ValueError("total SMD scalar/profile/component binding drifted.")
        if definition.enabled or profile.enabled:
            raise ValueError("total SMD candidate must remain disabled.")
        fingerprint = self._current_fingerprint_sha256()
        object.__setattr__(self, "_construction_fingerprint", fingerprint)

    @property
    def model(self):
        return self.electrostatic.model

    @property
    def continuum(self):
        return self.electrostatic.continuum

    def _current_fingerprint_sha256(self) -> str:
        definition = get_scalar_definition(self.scalar_id)
        profile = get_solvation_profile(self.profile_id)
        return _hash(
            {
                "scalar": definition.as_dict(),
                "profile": profile.as_dict(),
                "electrostatic_fingerprint_sha256": (
                    self.electrostatic.fingerprint_sha256()
                ),
                "nonpolar_provider_id": self.nonpolar.provider_id,
                "nonpolar_configuration_sha256": (self.nonpolar.configuration_sha256()),
                "aimnet2_source_evaluation": "one-shot-per-geometry",
                "continuum_field_supplied_to_aimnet2": False,
                "electronic_scf_iteration": False,
            }
        )

    def fingerprint_sha256(self) -> str:
        current = self._current_fingerprint_sha256()
        if self._construction_fingerprint and current != self._construction_fingerprint:
            raise ValueError("total SMD scalar/component configuration drifted.")
        return current

    def _energy_ledger(
        self,
        electrostatic: GeometryMediatedEnergyEvaluation,
        nonpolar: GeometryNonpolarEvaluation,
    ) -> GeometryMediatedSMDTotalEnergyEvaluation:
        return GeometryMediatedSMDTotalEnergyEvaluation(
            scalar_id=self.scalar_id,
            profile_id=self.profile_id,
            vacuum_energy_eV=electrostatic.vacuum_energy_eV,
            continuum_energy_eV=electrostatic.continuum_energy_eV,
            nonpolar_energy_eV=nonpolar.energy_eV,
            total_energy_eV=(electrostatic.total_energy_eV + nonpolar.energy_eV),
        )

    def evaluate_energy_components(
        self, geometry: Any
    ) -> GeometryMediatedSMDTotalEnergyEvaluation:
        self.fingerprint_sha256()
        electrostatic = self.electrostatic.evaluate_energy_components(geometry)
        nonpolar = self.nonpolar.evaluate(geometry)
        return self._energy_ledger(electrostatic, nonpolar)

    def evaluate_energy(self, geometry: Any) -> float:
        return self.evaluate_energy_components(geometry).total_energy_eV

    def evaluate(self, geometry: Any) -> GeometryMediatedSMDTotalEvaluation:
        self.fingerprint_sha256()
        electrostatic = self.electrostatic.evaluate(geometry)
        nonpolar = self.nonpolar.evaluate(geometry)
        nonpolar_gradient = _readonly_gradient(
            nonpolar.gradient_eV_per_A,
            count=electrostatic.source.shape[0],
            name="nonpolar gradient",
        )
        return GeometryMediatedSMDTotalEvaluation(
            energy=self._energy_ledger(electrostatic.energy, nonpolar),
            electrostatic=electrostatic,
            nonpolar=nonpolar,
            nonpolar_gradient_eV_per_A=nonpolar_gradient,
            total_gradient_eV_per_A=(
                electrostatic.total_gradient_eV_per_A + nonpolar_gradient
            ),
        )

    def hessian_vector_product(self, geometry: Any, coordinate_direction: object):
        del geometry, coordinate_direction
        raise NotImplementedError(
            "The PySCF SMD-CDS total profile has no admitted same-scalar HVP; "
            "PySCF's CDS Hessian is semi-analytical and must pass a separately "
            "registered convergence and symmetry gate before Tier H/FREQ use."
        )


@dataclass(frozen=True, slots=True)
class GeometryMediatedMultisolventSMDTotalScalar(GeometryMediatedSMDTotalScalar):
    """Disabled multi-solvent total scalar for one-shot AIMNet2 NQE charges."""

    nonpolar: PySCFMultisolventSMDCDSNonpolarFunctional
    scalar_id: str = (
        CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_V1
    )
    profile_id: str = (
        CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_SMDCDS_PROFILE_V1
    )
    _construction_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.electrostatic, GeometryMediatedElectrostaticScalar):
            raise TypeError(
                "multi-solvent total SMD scalar requires "
                "GeometryMediatedElectrostaticScalar."
            )
        if not isinstance(self.nonpolar, PySCFMultisolventSMDCDSNonpolarFunctional):
            raise TypeError(
                "multi-solvent total SMD scalar requires the exact registered "
                "PySCF multi-solvent SMD-CDS functional."
            )
        if (
            self.electrostatic.scalar_id
            != CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_ELECTROSTATIC_V1
            or self.electrostatic.profile_id
            != CANDIDATE_AIMNET2_FROZEN_CHARGE_MULTISOLVENT_SMOOTH_PARTITION_HARMONIC_DDPCM_PROFILE_V1
        ):
            raise ValueError(
                "multi-solvent total SMD scalar requires the exact AIMNet2 "
                "smooth-partition ddPCM electrostatic child."
            )
        continuum_solvent = getattr(self.electrostatic.continuum, "solvent", None)
        if continuum_solvent != self.nonpolar.solvent:
            raise ValueError(
                "electrostatic and SMD-CDS components must bind the same solvent."
            )
        definition = get_scalar_definition(self.scalar_id)
        profile = get_solvation_profile(self.profile_id)
        child_definition = get_scalar_definition(self.electrostatic.scalar_id)
        child_profile = get_solvation_profile(self.electrostatic.profile_id)
        if definition.implementation_entry_point != (
            "maple.solvation.coupling.geometry_mediated_smd:"
            "GeometryMediatedMultisolventSMDTotalScalar"
        ):
            raise ValueError(
                "registered multi-solvent total SMD implementation entry drifted."
            )
        if (
            profile.scalar_id != self.scalar_id
            or profile.state_equation_id != definition.state_equation_id
            or profile.model_profile != child_profile.model_profile
            or profile.continuum_profile != child_definition.continuum_profile
            or profile.cavity_profile != child_definition.cavity_profile
            or profile.nonpolar_profile != self.nonpolar.nonpolar_profile_id
            or definition.nonpolar_profile != self.nonpolar.nonpolar_profile_id
        ):
            raise ValueError(
                "multi-solvent total SMD scalar/profile/component binding drifted."
            )
        if definition.enabled or profile.enabled:
            raise ValueError("multi-solvent total SMD candidate must remain disabled.")
        object.__setattr__(
            self, "_construction_fingerprint", self._current_fingerprint_sha256()
        )


__all__ = [
    "GeometryMediatedMultisolventSMDTotalScalar",
    "GeometryMediatedSMDTotalEnergyEvaluation",
    "GeometryMediatedSMDTotalEvaluation",
    "GeometryMediatedSMDTotalScalar",
    "GeometryNonpolarEvaluation",
    "PYSCF_SMD_CDS_NONPOLAR_PROVIDER_ID",
    "PYSCF_MULTISOLVENT_SMD_CDS_NONPOLAR_PROFILE_ID",
    "PYSCF_MULTISOLVENT_SMD_CDS_NONPOLAR_PROVIDER_ID",
    "PYSCF_WATER_SMD_CDS_NONPOLAR_PROFILE_ID",
    "PySCFSMDCDSNonpolarFunctional",
    "PySCFMultisolventSMDCDSNonpolarFunctional",
]
