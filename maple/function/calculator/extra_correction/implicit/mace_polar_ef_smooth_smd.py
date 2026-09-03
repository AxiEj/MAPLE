"""Known-nonpassive MACE-POLAR-EF/smooth-PCM single-point diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from ....route2_energy_ledger import (
    MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1,
    route2_energy_composition_description,
)
from ....route2_model_contracts import (
    ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
    ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING,
)
from ....route2_smd_profiles import (
    MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE,
    route2_smd_profile_spec,
)
from ....route2_solvents import normalize_route2_solvent_name
from ...calculator_base import EV2HARTREE
from .mace_polar_ef import MACEPolarEFEnergyModel
from .mace_polar_ef_smooth_pcm import (
    MACEPolarEFSmoothPCMConfig,
    MACEPolarEFSmoothPCMCoupling,
)
from .mace_polar_ef_stationary import (
    MACEPolarEFSCFSettings,
    MACEPolarEFStationaryResult,
)
from .pyscf_smd_cds import PySCFSMDCDSResult, pyscf_smd_cds
from .result import SolvationResult
from .route2_domain import validate_route2_domain
from .smd_cds import route2_coulomb_radii
from .torch_smooth_pcm import TorchSmoothPCM

SMOOTH_PCM_TRANSITION_WIDTH_ANGSTROM2 = 0.08
SMOOTH_PCM_SURFACE_LMAX = 3
SMOOTH_PCM_PARTITION_LMAX = 6
SMOOTH_PCM_PARTITION_RADIAL_ORDER = 96
SMOOTH_PCM_SOURCE_RADIAL_ORDER = 128
SMOOTH_PCM_DOUBLE_LAYER_RADIAL_ORDER = 128
SMOOTH_PCM_SOURCE_SHELL_CLEARANCE_ANGSTROM = 0.05
SMOOTH_PCM_SOURCE_RESIDUAL_TOLERANCE = 5.0e-5
SMOOTH_PCM_MAXIMUM_ITERATIONS = 100
SMOOTH_PCM_SCF_MIXING = 0.5


@dataclass
class MACEPolarEFSmoothPCMKnownNonpassiveDiagnostic:
    """Explicitly acknowledged energy-only diagnostic; never a prediction API."""

    atoms: Any
    solvation_options: dict[str, Any]
    audit_dir: Path | None = None

    supported_properties = frozenset({"energy"})

    def __post_init__(self) -> None:
        self.solvation_options = dict(self.solvation_options)
        self.profile = str(self.solvation_options.get("profile", "")).lower()
        self.provider = str(self.solvation_options.get("provider", "")).lower()
        self.response = str(self.solvation_options.get("response", "scf")).lower()
        self.standard_state = str(
            self.solvation_options.get("standard_state", "1m")
        ).lower()
        self.solvent = normalize_route2_solvent_name(
            self.solvation_options.get("implicit", "")
        )
        self.solvation_options["implicit"] = self.solvent
        self.profile_spec = route2_smd_profile_spec(self.profile)
        self._validate_options()
        validate_route2_domain(self.atoms)
        self._reference_numbers = np.asarray(self.atoms.numbers, dtype=int).copy()
        radii = route2_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            solvent=self.solvent,
            profile=self.profile,
        )
        self.continuum = TorchSmoothPCM(
            atomic_numbers=tuple(int(value) for value in self._reference_numbers),
            radii_angstrom=tuple(float(value) for value in radii),
            transition_width_angstrom2=(SMOOTH_PCM_TRANSITION_WIDTH_ANGSTROM2),
            surface_lmax=SMOOTH_PCM_SURFACE_LMAX,
            partition_lmax=SMOOTH_PCM_PARTITION_LMAX,
            partition_radial_quadrature_order=(SMOOTH_PCM_PARTITION_RADIAL_ORDER),
            source_radial_quadrature_order=SMOOTH_PCM_SOURCE_RADIAL_ORDER,
            double_layer_radial_quadrature_order=(SMOOTH_PCM_DOUBLE_LAYER_RADIAL_ORDER),
            dielectric=78.355,
            source_shell_clearance_angstrom=(
                SMOOTH_PCM_SOURCE_SHELL_CLEARANCE_ANGSTROM
            ),
        )
        self._coupling: MACEPolarEFSmoothPCMCoupling | None = None
        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = {
            "provider": self.provider,
            "method": "smd",
            "profile": self.profile,
            "solvent": self.solvent,
            "response": self.response,
            "standard_state": "1M(gas)->1M(solution)",
            "known_nonpassive_diagnostic": True,
            "known_nonpassive_acknowledged": True,
            "electronic_model_family": ROUTE2_MACE_POLAR_EF_V2_MODEL_FAMILY,
            "electronic_profile_binding": (ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING),
            "scientifically_valid": False,
            "intended_use": "implementation-diagnostic-only-not-prediction",
            "accuracy_certified": False,
            "solution_phase_pes": False,
            "forces_available": False,
            "default_eligible": False,
            "release_admitted": False,
            "electrostatic_energy_ledger": (
                MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1
            ),
            "energy_composition": route2_energy_composition_description(
                MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1,
                continuum_symbol="smooth-ddPCM",
            ),
            "continuum": dict(self.continuum.execution_provenance()),
            "continuum_configuration_sha256": (self.continuum.configuration_sha256()),
        }

    def _validate_options(self) -> None:
        if self.solvation_options.get("experimental") is not True:
            raise ValueError("The diagnostic requires experimental=true.")
        if self.solvation_options.get("acknowledge_known_nonpassive") is not True:
            raise ValueError(
                "The diagnostic requires acknowledge_known_nonpassive=true."
            )
        if str(self.solvation_options.get("method", "")).lower() != "smd":
            raise ValueError("The diagnostic requires method=smd.")
        if self.provider != "torch-smooth-pcm":
            raise ValueError("The diagnostic requires provider=torch-smooth-pcm.")
        if self.profile != MACE_POLAR_EF_SMOOTH_PCM_DIAGNOSTIC_PROFILE:
            raise ValueError("Unsupported MACE-POLAR-EF smooth-PCM profile.")
        if not self.profile_spec.known_nonpassive_diagnostic:
            raise ValueError("Profile is not classified as known nonpassive.")
        if self.solvent != "water":
            raise ValueError("The diagnostic is water-only.")
        if self.response != "scf":
            raise ValueError("The diagnostic requires response=scf.")
        if self.standard_state != "1m":
            raise ValueError("The diagnostic requires standard_state=1m.")
        if "cavity_policy" in self.solvation_options:
            raise ValueError("The smooth PCM profile owns its cavity policy.")

    def _validate_atoms(self, atoms) -> None:
        validate_route2_domain(atoms)
        if not np.array_equal(
            np.asarray(atoms.numbers, dtype=int),
            self._reference_numbers,
        ):
            raise ValueError(
                "The diagnostic does not permit atom identity/order changes."
            )

    def _coupling_for_calculator(
        self,
        calculator,
        atoms,
    ) -> MACEPolarEFSmoothPCMCoupling:
        if calculator is None:
            raise TypeError("The diagnostic requires MACEPolarEFCalculator.")
        if (
            getattr(calculator, "route2_smd_profile", None)
            != ROUTE2_MACE_POLAR_EF_V2_PROFILE_BINDING
        ):
            raise TypeError("MACE-POLAR-EF profile binding mismatch.")
        evaluator_factory = getattr(calculator, "mace_polar_ef_evaluator", None)
        if not callable(evaluator_factory):
            raise TypeError("Calculator has no MACE-POLAR-EF evaluator boundary.")
        evaluator = cast(MACEPolarEFEnergyModel, evaluator_factory(atoms))
        if not isinstance(evaluator, MACEPolarEFEnergyModel):
            raise TypeError("Calculator returned an invalid EF evaluator.")
        if self._coupling is None:
            config = MACEPolarEFSmoothPCMConfig(
                electronic=evaluator.config,
                continuum=self.continuum,
                scf=MACEPolarEFSCFSettings(
                    source_residual_tolerance=(SMOOTH_PCM_SOURCE_RESIDUAL_TOLERANCE),
                    maximum_iterations=SMOOTH_PCM_MAXIMUM_ITERATIONS,
                    mixing=SMOOTH_PCM_SCF_MIXING,
                    electronic_passivity_policy=("record-known-failure-diagnostic"),
                ),
            )
            self._coupling = MACEPolarEFSmoothPCMCoupling(
                config,
                electronic_model=evaluator,
            )
        elif self._coupling.electronic is not evaluator:
            raise RuntimeError("MACE-POLAR-EF evaluator identity changed.")
        return self._coupling

    def _write_audit(
        self,
        atoms,
        *,
        coupled: MACEPolarEFStationaryResult,
        gas_energy_ev: float,
        cds: PySCFSMDCDSResult,
        components: dict[str, float],
    ) -> None:
        if self.audit_dir is None:
            return
        state_path = self.audit_dir / "mace-polar-ef-smooth-pcm-state.npz"
        np.savez_compressed(
            state_path,
            positions_angstrom=np.asarray(atoms.get_positions(), dtype=float),
            source_raw=coupled.source_raw,
            field_cartesian=coupled.field_cartesian,
            source_residual=coupled.source_residual,
            coordinate_gradient_ev_per_angstrom=(
                coupled.coordinate_gradient_ev_per_angstrom
            ),
        )
        payload = {
            "schema_version": 1,
            "known_nonpassive_diagnostic": True,
            "known_nonpassive_acknowledged": True,
            "scientifically_valid": False,
            "profile": self.profile,
            "gas_energy_ev": gas_energy_ev,
            "coupled_common_scalar_energy_ev": coupled.total_energy_ev,
            "continuum_energy_ev": coupled.continuum_energy_ev,
            "source_field_pairing_ev": coupled.source_field_pairing_ev,
            "components_hartree": components,
            "iterations": coupled.iterations,
            "maximum_source_residual": coupled.maximum_source_residual,
            "maximum_field_replay_difference": (
                coupled.maximum_field_replay_difference
            ),
            "electronic_passivity": dict(
                cast(
                    Mapping[str, object],
                    coupled.provenance["electronic_passivity"],
                )
            ),
            "cds_provider": dict(cds.runtime_provenance),
            "state_archive": state_path.name,
            "configuration_sha256": coupled.configuration_sha256,
        }
        (self.audit_dir / "mace-polar-ef-smooth-pcm-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def evaluate(
        self,
        atoms,
        *,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if need_forces:
            raise NotImplementedError("The known-nonpassive diagnostic is energy-only.")
        self._validate_atoms(atoms)
        coupling = self._coupling_for_calculator(calculator, atoms)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        coupled = coupling.evaluate(positions)
        gas_energy_method = getattr(
            calculator,
            "mace_polar_ef_gas_energy_ev",
            None,
        )
        if not callable(gas_energy_method):
            raise TypeError("Calculator has no gas-energy cache boundary.")
        gas_energy_ev = float(cast(float, gas_energy_method(atoms)))
        cds = pyscf_smd_cds(
            atoms.get_chemical_symbols(),
            positions,
            solvent=self.solvent,
        )
        components = {
            "solute_polarization": (
                coupled.electronic_energy_ev
                - gas_energy_ev
                - coupled.source_field_pairing_ev
            )
            * EV2HARTREE,
            "pcm_polarization": coupled.continuum_energy_ev * EV2HARTREE,
            "cds": cds.energy_hartree,
            "standard_state": 0.0,
        }
        components["electrostatic"] = (
            components["solute_polarization"] + components["pcm_polarization"]
        )
        components["delta_g_solv"] = components["electrostatic"] + components["cds"]
        self._write_audit(
            atoms,
            coupled=coupled,
            gas_energy_ev=gas_energy_ev,
            cds=cds,
            components=components,
        )
        runtime_provenance = {
            **self.provenance,
            "converged": True,
            "iterations": coupled.iterations,
            "configuration_sha256": coupled.configuration_sha256,
            "electronic_passivity": dict(
                cast(
                    Mapping[str, object],
                    coupled.provenance["electronic_passivity"],
                )
            ),
            "maximum_source_residual": coupled.maximum_source_residual,
            "maximum_field_replay_difference": (
                coupled.maximum_field_replay_difference
            ),
            "cds_provider": dict(cds.runtime_provenance),
            "audit_directory": (
                None if self.audit_dir is None else str(self.audit_dir)
            ),
        }
        return SolvationResult(
            energy_hartree=components["delta_g_solv"],
            components_hartree=components,
            provenance=runtime_provenance,
        )


__all__ = [
    "MACEPolarEFSmoothPCMKnownNonpassiveDiagnostic",
    "SMOOTH_PCM_DOUBLE_LAYER_RADIAL_ORDER",
    "SMOOTH_PCM_PARTITION_LMAX",
    "SMOOTH_PCM_PARTITION_RADIAL_ORDER",
    "SMOOTH_PCM_SOURCE_RADIAL_ORDER",
    "SMOOTH_PCM_SURFACE_LMAX",
    "SMOOTH_PCM_TRANSITION_WIDTH_ANGSTROM2",
]
