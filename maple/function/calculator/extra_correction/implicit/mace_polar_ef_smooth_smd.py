"""Known-nonpassive MACE-POLAR-EF/smooth-PCM diagnostic provider."""

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
from ....route2_smd_profiles import (
    route2_smd_profile_spec,
)
from ....route2_solvents import (
    normalize_route2_solvent_name,
    route2_solvent_spec,
)
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
from .torch_smooth_cosmo import TorchSmoothCOSMO
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
    """Explicitly acknowledged energy/derivative diagnostic provider."""

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
        self.derivatives_enabled = bool(
            self.profile_spec.diagnostic_derivative_eligible
        )
        self.supported_properties = frozenset(
            {"energy", "forces", "numerical_hessian"}
            if self.derivatives_enabled
            else {"energy"}
        )
        self.solvent_spec = route2_solvent_spec(self.solvent)
        validate_route2_domain(self.atoms)
        self._reference_numbers = np.asarray(self.atoms.numbers, dtype=int).copy()
        radii = route2_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            solvent=self.solvent,
            profile=self.profile,
        )
        continuum_kwargs = dict(
            atomic_numbers=tuple(int(value) for value in self._reference_numbers),
            radii_angstrom=tuple(float(value) for value in radii),
            transition_width_angstrom2=(SMOOTH_PCM_TRANSITION_WIDTH_ANGSTROM2),
            surface_lmax=SMOOTH_PCM_SURFACE_LMAX,
            partition_lmax=SMOOTH_PCM_PARTITION_LMAX,
            partition_radial_quadrature_order=(SMOOTH_PCM_PARTITION_RADIAL_ORDER),
            source_radial_quadrature_order=SMOOTH_PCM_SOURCE_RADIAL_ORDER,
            double_layer_radial_quadrature_order=(SMOOTH_PCM_DOUBLE_LAYER_RADIAL_ORDER),
            source_shell_clearance_angstrom=(
                SMOOTH_PCM_SOURCE_SHELL_CLEARANCE_ANGSTROM
            ),
        )
        if self.profile_spec.electrostatics_model == "smooth-cosmo":
            self.continuum = TorchSmoothCOSMO(**continuum_kwargs)
            continuum_symbol = "smooth-COSMO"
        else:
            self.continuum = TorchSmoothPCM(
                **continuum_kwargs,
                dielectric=self.solvent_spec.descriptors.dielectric,
            )
            continuum_symbol = "smooth-ddPCM"
        self._coupling: MACEPolarEFSmoothPCMCoupling | None = None
        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = {
            "provider": self.provider,
            "method": str(self.solvation_options.get("method", "")),
            "profile": self.profile,
            "solvent": self.solvent,
            "response": self.response,
            "electrostatics_model": self.profile_spec.electrostatics_model,
            "standard_state": "1M(gas)->1M(solution)",
            "known_nonpassive_diagnostic": True,
            "known_nonpassive_acknowledged": True,
            "diagnostic_derivative_eligible": self.derivatives_enabled,
            "unvalidated_derivatives_acknowledged": self.derivatives_enabled,
            "electronic_model_family": (
                self.profile_spec.electronic_model_family
            ),
            "electronic_profile_binding": (
                self.profile_spec.electronic_profile_binding
            ),
            "scientifically_valid": False,
            "intended_use": (
                "energy-force-fd-hessian-workflow-diagnostic-only-not-prediction"
                if self.derivatives_enabled
                else "implementation-diagnostic-only-not-prediction"
            ),
            "accuracy_certified": False,
            "solution_phase_pes": False,
            "diagnostic_solution_phase_scalar_available": (
                self.derivatives_enabled
            ),
            "forces_available": self.derivatives_enabled,
            "numerical_hessian_available": self.derivatives_enabled,
            "default_eligible": False,
            "release_admitted": False,
            "electrostatic_energy_ledger": (
                MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1
            ),
            "energy_composition": route2_energy_composition_description(
                MACE_EF_KNOWN_NONPASSIVE_COMMON_SCALAR_DIAGNOSTIC_V1,
                continuum_symbol=continuum_symbol,
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
        method = str(self.solvation_options.get("method", "")).lower()
        expected_method = (
            "cosmo"
            if self.profile_spec.electrostatics_model == "smooth-cosmo"
            else "smd"
        )
        if method != expected_method:
            raise ValueError(
                f"The selected profile requires method={expected_method}."
            )
        if self.provider != self.profile_spec.provider:
            raise ValueError(
                "The diagnostic provider does not match its profile."
            )
        if not self.profile_spec.known_nonpassive_diagnostic:
            raise ValueError("Profile is not classified as known nonpassive.")
        if not self.profile_spec.supports_solvent(self.solvent):
            raise ValueError(
                f"Profile={self.profile} does not support solvent={self.solvent}."
            )
        derivative_acknowledgement = self.solvation_options.get(
            "acknowledge_unvalidated_derivatives"
        )
        if (
            self.profile_spec.diagnostic_derivative_eligible
            and derivative_acknowledgement is not True
        ):
            raise ValueError(
                "The diagnostic derivative profile requires "
                "acknowledge_unvalidated_derivatives=true."
            )
        if (
            not self.profile_spec.diagnostic_derivative_eligible
            and "acknowledge_unvalidated_derivatives" in self.solvation_options
        ):
            raise ValueError(
                "acknowledge_unvalidated_derivatives is valid only for the "
                "diagnostic derivative profile."
            )
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

    def _evaluate_nonpolar(
        self,
        atoms,
        positions: np.ndarray,
    ) -> PySCFSMDCDSResult:
        if self.profile_spec.nonpolar_model == "none":
            return PySCFSMDCDSResult(
                energy_hartree=0.0,
                energy_kcal_mol=0.0,
                position_gradient_hartree_per_angstrom=np.zeros_like(positions),
                runtime_provenance={
                    "provider": "none",
                    "pyscf_version": "not-used",
                    "solvent": self.solvent,
                    "pyscf_smd_solvent": "not-used",
                    "upstream_entrypoint": "not-used",
                },
            )
        return pyscf_smd_cds(
            atoms.get_chemical_symbols(),
            positions,
            solvent=self.solvent,
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
            != self.profile_spec.electronic_profile_binding
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
        correction_forces_hartree_per_angstrom: np.ndarray | None,
        diagnostic_derivative_evidence: dict[str, object] | None,
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
            correction_forces_hartree_per_angstrom=(
                np.empty((0, 3), dtype=float)
                if correction_forces_hartree_per_angstrom is None
                else correction_forces_hartree_per_angstrom
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
            "diagnostic_derivative_evidence": diagnostic_derivative_evidence,
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
        if need_forces and not self.derivatives_enabled:
            raise NotImplementedError(
                "The selected known-nonpassive diagnostic is energy-only."
            )
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
        cds = self._evaluate_nonpolar(atoms, positions)
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
        correction_forces = None
        derivative_evidence = None
        if need_forces:
            gas_gradient_method = getattr(
                calculator,
                "mace_polar_ef_gas_gradient_ev_per_angstrom",
                None,
            )
            if not callable(gas_gradient_method):
                raise TypeError("Calculator has no gas-gradient cache boundary.")
            gas_gradient = np.asarray(
                gas_gradient_method(atoms),
                dtype=float,
            )
            if gas_gradient.shape != positions.shape or not np.all(
                np.isfinite(gas_gradient)
            ):
                raise ValueError("MACE-POLAR-EF gas gradient is malformed.")
            combined_gradient = (
                coupled.coordinate_gradient_ev_per_angstrom * EV2HARTREE
                + cds.position_gradient_hartree_per_angstrom
            )
            correction_gradient = (
                combined_gradient - gas_gradient * EV2HARTREE
            )
            correction_forces = -correction_gradient
            derivative_evidence = {
                "scope": "known-nonpassive-diagnostic-derivative-v1",
                "scalar": (
                    "E_MACE-EF(R,f)+U_smooth_PCM(R,c)-<c,f>+G_CDS"
                ),
                "first_derivative": "stationary-envelope-analytic",
                "hessian": "central-finite-difference-of-analytic-forces",
                "maximum_combined_net_force_hartree_per_angstrom": float(
                    np.max(np.abs(np.sum(-combined_gradient, axis=0)))
                ),
                "maximum_source_residual": coupled.maximum_source_residual,
                "maximum_field_replay_difference": (
                    coupled.maximum_field_replay_difference
                ),
                "scientifically_valid": False,
                "release_admitted": False,
            }
        self._write_audit(
            atoms,
            coupled=coupled,
            gas_energy_ev=gas_energy_ev,
            cds=cds,
            components=components,
            correction_forces_hartree_per_angstrom=correction_forces,
            diagnostic_derivative_evidence=derivative_evidence,
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
            "diagnostic_derivative_evidence": derivative_evidence,
            "audit_directory": (
                None if self.audit_dir is None else str(self.audit_dir)
            ),
        }
        return SolvationResult(
            energy_hartree=components["delta_g_solv"],
            forces_hartree_per_angstrom=correction_forces,
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
