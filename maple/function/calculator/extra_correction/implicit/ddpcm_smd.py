"""Self-consistent MACE-POLAR/pyddx/SMD research force candidates.

This provider is deliberately separate from the public PCMSolver/GePol energy
proof of concept.  One profile-selected pyddx ddPCM or scaled-ddCOSMO object
owns the scalar polarization energy, reaction-field forward/adjoint maps, and
complete coordinate derivative.  The official PySCF SMD CDS entrypoint
supplies its scalar energy and matching analytic gradient.  Components from
different continuum equations are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from ....route2_smd_profiles import (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_PROFILE,
    DDPCM_MULTISOLVENT_SMD_PROFILE,
    DDPCM_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_PROFILE,
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    SUPPORTED_PYDDX_SMD_PROFILES,
    route2_smd_profile_spec,
)
from ....route2_solvents import (
    normalize_route2_solvent_name,
    route2_solvent_spec,
)
from ...calculator_base import ROUTE2_SMD_CALCULATOR_PROFILE
from .pyddx_pcm_response import (
    PyDDXCOSMOReactionFieldLinearMap,
    PyDDXPCMReactionFieldLinearMap,
)
from .pyscf_smd_cds import pyscf_smd_cds
from .result import SolvationResult
from .route2_domain import validate_route2_domain
from .route2_engine import (
    Route2ContinuumEngine,
    Route2CoupledState,
    Route2EngineSettings,
)
from .smd_cds import route2_coulomb_radii


WATER_STATIC_DIELECTRIC = 78.39
DDPCM_LMAX = 15
DDPCM_N_LEBEDEV = 1202
DDPCM_SOLVER_TOLERANCE = 1.0e-12
DDPCM_ETA = 0.1
SCF_MIXING = 1.0
SCF_DENSITY_TOLERANCE = 2.0e-12
SCF_ENERGY_TOLERANCE_EV = 1.0e-10
SCF_MAX_ITERATIONS = 100
ADJOINT_RELATIVE_TOLERANCE = 1.0e-10
ADJOINT_ABSOLUTE_TOLERANCE = 1.0e-13
ADJOINT_MAX_ITERATIONS = 100
ENERGY_IDENTITY_TOLERANCE_EV = 2.0e-10
FORCE_STATE_ENERGY_TOLERANCE_EV = 1.0e-9
NEUTRAL_DENSITY_TOLERANCE = 1.0e-8

_DDPCM_ENGINE_SETTINGS = Route2EngineSettings(
    continuum_label="ddPCM",
    scf_mixing=SCF_MIXING,
    scf_density_tolerance=SCF_DENSITY_TOLERANCE,
    scf_energy_tolerance_ev=SCF_ENERGY_TOLERANCE_EV,
    scf_max_iterations=SCF_MAX_ITERATIONS,
    adjoint_relative_tolerance=ADJOINT_RELATIVE_TOLERANCE,
    adjoint_absolute_tolerance=ADJOINT_ABSOLUTE_TOLERANCE,
    adjoint_max_iterations=ADJOINT_MAX_ITERATIONS,
    energy_identity_tolerance_ev=ENERGY_IDENTITY_TOLERANCE_EV,
    force_state_energy_tolerance_ev=FORCE_STATE_ENERGY_TOLERANCE_EV,
    neutral_density_tolerance=NEUTRAL_DENSITY_TOLERANCE,
)


_CONTINUUM_LABELS = {
    "ddpcm": "ddPCM",
    "ddcosmo": "ddCOSMO",
}


def _engine_settings(electrostatics_model: str) -> Route2EngineSettings:
    try:
        label = _CONTINUUM_LABELS[electrostatics_model]
    except KeyError as exc:
        raise ValueError(
            "The pyddx Route-2 provider requires electrostatics_model="
            "ddpcm or ddcosmo."
        ) from exc
    if label == _DDPCM_ENGINE_SETTINGS.continuum_label:
        return _DDPCM_ENGINE_SETTINGS
    return replace(
        _DDPCM_ENGINE_SETTINGS,
        continuum_label=label,
    )


def _normalized_mol2_atom_types(atoms) -> tuple[str, ...] | None:
    mol2 = atoms.info.get("mol2")
    if not isinstance(mol2, dict) or mol2.get("atom_types") is None:
        return None
    return tuple(
        str(atom_type).strip().lower()
        for atom_type in mol2["atom_types"]
    )


@dataclass
class PyDDXSMDImplicitSolvation:
    """SMD-CDS correction with one profile-selected pyddx equation."""

    atoms: Any
    solvation_options: dict[str, Any]
    audit_dir: Path | None = None

    supported_properties = frozenset({"energy", "forces"})

    def __post_init__(self) -> None:
        self.solvation_options = dict(self.solvation_options)
        if "profile" not in self.solvation_options:
            raise ValueError(
                "Route 2 provider=pyddx requires an explicit "
                "versioned profile."
            )
        self.provider = str(
            self.solvation_options.get("provider", "")
        ).lower()
        self.profile = str(
            self.solvation_options["profile"]
        ).lower()
        self.response = str(
            self.solvation_options.get("response", "scf")
        ).lower()
        self.standard_state = str(
            self.solvation_options.get("standard_state", "1m")
        ).lower()
        self.profile_spec = route2_smd_profile_spec(self.profile)
        self.solvent = normalize_route2_solvent_name(
            self.solvation_options.get("implicit", "")
        )
        self.solvation_options["implicit"] = self.solvent
        self.solvent_spec = route2_solvent_spec(self.solvent)
        self._validate_options()
        self.electrostatics_model = self.profile_spec.electrostatics_model
        self.continuum_label = _CONTINUUM_LABELS[self.electrostatics_model]
        self.continuum_dielectric = (
            WATER_STATIC_DIELECTRIC
            if self.profile_spec.dielectric_policy
            == "legacy-water-78.39"
            else self.solvent_spec.descriptors.dielectric
        )
        validate_route2_domain(self.atoms)
        self._reference_numbers = np.asarray(
            self.atoms.numbers,
            dtype=int,
        ).copy()

        self._reference_mol2_atom_types = _normalized_mol2_atom_types(
            self.atoms
        )
        self.coulomb_radii_angstrom = route2_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            solvent=self.solvent,
            atom_types=self._reference_mol2_atom_types,
            profile=self.profile,
        )
        self._engine = Route2ContinuumEngine(
            reaction_field_factory=self._build_reaction_field,
            cds_evaluator=self._evaluate_cds,
            settings=_engine_settings(self.electrostatics_model),
        )
        self._cached_state: Route2CoupledState | None = None

        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        numerics = {
            "dielectric": self.continuum_dielectric,
            "dielectric_policy": self.profile_spec.dielectric_policy,
            "coulomb_radii_policy": (
                self.profile_spec.coulomb_radii_policy
            ),
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "pyddx_n_proc": self.profile_spec.ddpcm_n_proc,
            "pyddx_solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "pyddx_eta": DDPCM_ETA,
            "scf_mixing": SCF_MIXING,
            "scf_density_tolerance_e": SCF_DENSITY_TOLERANCE,
            "scf_energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
            "scf_maximum_iterations": SCF_MAX_ITERATIONS,
            "adjoint_relative_tolerance": ADJOINT_RELATIVE_TOLERANCE,
            "adjoint_absolute_tolerance": ADJOINT_ABSOLUTE_TOLERANCE,
            "adjoint_maximum_iterations": ADJOINT_MAX_ITERATIONS,
        }
        if self.electrostatics_model == "ddpcm":
            numerics.update(
                {
                    "ddpcm_n_proc": self.profile_spec.ddpcm_n_proc,
                    "ddpcm_solver_tolerance": DDPCM_SOLVER_TOLERANCE,
                    "ddpcm_eta": DDPCM_ETA,
                }
            )
        self.provenance = {
            "provider": "pyddx",
            "method": "smd",
            "profile": self.profile,
            "scientific_identity": (
                "MACE-POLAR/(l<=1)-point-multipole + "
                f"{self.continuum_label} + SMD-CDS"
            ),
            "solvent": self.solvent,
            "pyscf_smd_solvent": self.solvent_spec.pyscf_smd_name,
            "response": "scf",
            "standard_state": "1M(gas)->1M(solution)",
            "standard_state_correction_hartree": 0.0,
            "density_source": (
                "official MACE-POLAR-1-M l<=1 residual charge density"
            ),
            "density_interpretation": (
                "coarse-grained net charge density, not a QM electron density"
            ),
            "electrostatics": self.continuum_label,
            "electrostatics_model": (
                self.profile_spec.electrostatics_model
            ),
            "solute_source": self.profile_spec.solute_source,
            "reaction_field_projector": (
                self.profile_spec.reaction_field_projector
            ),
            "nonpolar_model": self.profile_spec.nonpolar_model,
            "strict_original_smd_equivalence": (
                self.profile_spec.strict_original_smd_equivalence
            ),
            "pcm_projection": "atom-centred l<=1 real spherical multipoles",
            "cavity_radii": (
                (
                    "SMD Coulomb radii with GAFF/GAFF2 carbonyl oxygen "
                    "(atom type o) overridden to 1.70 A"
                )
                if self.profile_spec.uses_gaff2_carbonyl_oxygen
                else (
                    "PySCF 2.13.1 SMD solvent-acidity-dependent Coulomb "
                    "radii with revised Br=2.60 A and I=2.74 A"
                    if self.profile_spec.coulomb_radii_policy
                    == "pyscf-smd-2.13.1"
                    else "frozen legacy Route-2 water-v1 Coulomb radii"
                )
            ),
            "mace_long_range_evaluator": (
                self.profile_spec.mace_long_range_evaluator
            ),
            "mace_long_range_evaluator_status": (
                "experimental fixed-box operator variant; not equivalent "
                "to the default molecular real-space evaluator"
                if self.profile_spec.mace_long_range_evaluator
                == MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
                else "official default molecular real-space evaluator"
            ),
            "cds": (
                "official PySCF SMD libsolvent energy and gradient for "
                f"{self.solvent_spec.pyscf_smd_name}"
            ),
            "solvent_descriptors": {
                "source": self.solvent_spec.descriptor_source,
                "values": self.solvent_spec.descriptors.as_pyscf_tuple(),
                "mnsol_name": self.solvent_spec.mnsol_name,
                "mnsol_doi": self.solvent_spec.experimental_dataset_doi,
            },
            "route_role": "research-innovation",
            "scientific_status": "single-point-force-candidate",
            "solution_phase_pes": False,
            "forces_available": True,
            "accuracy_certified": False,
            "default_eligible": False,
            "energy_composition": (
                "delta_G_solv = (E_MACE_intrinsic[V_reac]-E_MACE_gas) "
                f"+ E_{self.continuum_label} + G_CDS"
            ),
            "force_composition": (
                "F_solution = F_MACE_gas "
                "- d(delta_G_solv)/dR, with the converged-density "
                "response eliminated by one adjoint solve"
            ),
            "numerics": numerics,
        }
        self._write_manifest()

    def _validate_options(self) -> None:
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "The pyddx Route-2 force candidate requires "
                "experimental=true explicitly."
            )
        if str(self.solvation_options.get("method", "")).lower() != "smd":
            raise ValueError("PyDDXSMDImplicitSolvation requires method=smd.")
        if self.provider != "pyddx":
            raise ValueError(
                "PyDDXSMDImplicitSolvation requires provider=pyddx."
            )
        if self.profile not in SUPPORTED_PYDDX_SMD_PROFILES:
            supported = ", ".join(sorted(SUPPORTED_PYDDX_SMD_PROFILES))
            raise ValueError(
                "The pyddx Route-2 profile must be one of: "
                f"{supported}."
            )
        if not self.profile_spec.supports_solvent(self.solvent):
            raise ValueError(
                f"Route 2 profile={self.profile} does not support "
                f"solvent={self.solvent}."
            )
        if self.response != "scf":
            raise ValueError(
                "The pyddx Route-2 force candidate requires response=scf."
            )
        if self.standard_state != "1m":
            raise ValueError(
                "Route 2 uses the 1 M gas -> 1 M solution convention only; "
                "standard_state must be 1m."
            )
        if "cavity_policy" in self.solvation_options:
            raise ValueError(
                "cavity_policy is specific to the PCMSolver/GePol provider "
                "and is not valid for provider=pyddx."
            )

    def _write_manifest(self) -> None:
        if self.audit_dir is None:
            return
        manifest = {
            "schema_version": 1,
            "solvation_options": self.solvation_options,
            "provenance": self.provenance,
            "elements": self.atoms.get_chemical_symbols(),
            "positions_angstrom": np.asarray(
                self.atoms.get_positions(),
                dtype=float,
            ).tolist(),
        }
        (self.audit_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _validate_atoms(self, atoms) -> None:
        validate_route2_domain(atoms)
        numbers = np.asarray(atoms.numbers, dtype=int)
        if not np.array_equal(numbers, self._reference_numbers):
            raise ValueError(
                "Route 2 does not permit atom identity/order changes."
            )
        atom_types = _normalized_mol2_atom_types(atoms)
        if atom_types != self._reference_mol2_atom_types:
            raise ValueError(
                "Route 2 does not permit MOL2 atom type/order changes after "
                "the cavity radii are initialized."
            )

    def _validate_calculator(self, calculator, *, need_forces: bool) -> None:
        if calculator is None or not callable(
            getattr(calculator, "polar_state", None)
        ):
            raise TypeError(
                "The pyddx Route-2 provider requires the "
                "MACEPolCalculator polar_state() API."
            )
        if (
            getattr(calculator, "route2_smd_profile", None)
            != ROUTE2_SMD_CALCULATOR_PROFILE
        ):
            raise TypeError(
                "Route 2 requires the official MACE-POLAR-1-M float64 "
                "local-field calculator profile."
            )
        expected_evaluator = self.profile_spec.mace_long_range_evaluator
        actual_evaluator = getattr(
            calculator,
            "long_range_evaluator_profile",
            None,
        )
        if actual_evaluator != expected_evaluator:
            raise TypeError(
                "The selected Route-2 profile requires MACE-POLAR "
                f"long-range evaluator {expected_evaluator!r}; received "
                f"{actual_evaluator!r}."
            )
        if not need_forces:
            return
        required = (
            "intrinsic_energy_field_gradient",
            "linearize_density_response",
            "density_position_vjp",
        )
        missing = [
            name
            for name in required
            if not callable(getattr(calculator, name, None))
        ]
        if missing:
            raise TypeError(
                "The pyddx Route-2 force candidate requires the complete "
                "MACE response API; missing: "
                + ", ".join(missing)
                + "."
            )

    def _validate_density(
        self,
        values: np.ndarray,
        atom_count: int,
        *,
        name: str,
    ) -> np.ndarray:
        return self._engine.validate_density(
            values,
            atom_count,
            name=name,
        )

    def _gas_state(self, calculator, atoms, *, need_forces: bool) -> Any:
        return self._engine.gas_state(
            calculator,
            atoms,
            need_forces=need_forces,
        )

    def _build_reaction_field(self, atoms):
        reaction_field_type = {
            "ddpcm": PyDDXPCMReactionFieldLinearMap,
            "ddcosmo": PyDDXCOSMOReactionFieldLinearMap,
        }[self.electrostatics_model]
        return reaction_field_type(
            np.asarray(atoms.get_positions(), dtype=float),
            self.coulomb_radii_angstrom,
            dielectric=self.continuum_dielectric,
            lmax=DDPCM_LMAX,
            n_lebedev=DDPCM_N_LEBEDEV,
            n_proc=self.profile_spec.ddpcm_n_proc,
            solver_tolerance=DDPCM_SOLVER_TOLERANCE,
            eta=DDPCM_ETA,
        )

    def _evaluate_cds(self, atoms):
        return pyscf_smd_cds(
            atoms.get_chemical_symbols(),
            np.asarray(atoms.get_positions(), dtype=float),
            solvent=self.solvent,
        )

    def _solve_coupled_state(
        self,
        atoms,
        calculator,
        gas_state,
    ) -> Route2CoupledState:
        return self._engine.solve_coupled_state(
            atoms,
            calculator,
            gas_state,
            provider_cache_signature=(
                self.solvent,
                self.electrostatics_model,
                self._reference_mol2_atom_types,
            ),
        )

    def _coupled_state(
        self,
        atoms,
        calculator,
        gas_state,
    ) -> Route2CoupledState:
        cached = self._cached_state
        if cached is not None and cached.matches(
            calculator,
            atoms,
            provider_cache_signature=(
                self.solvent,
                self.electrostatics_model,
                self._reference_mol2_atom_types,
            ),
        ):
            return cached
        state = self._solve_coupled_state(
            atoms,
            calculator,
            gas_state,
        )
        self._cached_state = state
        return state

    def _solvent_correction_force(
        self,
        atoms,
        calculator,
        gas_state,
        coupled: Route2CoupledState,
    ):
        return self._engine.solvent_correction_force(
            atoms,
            calculator,
            gas_state,
            coupled,
        )

    def _write_result_audit(
        self,
        *,
        atoms,
        gas_state,
        coupled: Route2CoupledState,
        components: dict[str, float],
        derivative: dict[str, Any] | None,
    ) -> None:
        if self.audit_dir is None:
            return
        arrays: dict[str, np.ndarray] = {
            "positions_angstrom": np.asarray(
                atoms.get_positions(),
                dtype=float,
            ),
            "cavity_radii_angstrom": np.asarray(
                self.coulomb_radii_angstrom,
                dtype=float,
            ),
            "density_coefficients": coupled.density_coefficients,
            "reaction_field_values_ev": (
                coupled.reaction_field_values_ev
            ),
        }
        if derivative is not None:
            arrays.update(
                {
                    key: np.asarray(value, dtype=float)
                    for key, value in derivative.items()
                    if key != "adjoint"
                }
            )
        audit_stem = f"route2-{self.electrostatics_model}"
        state_path = self.audit_dir / f"{audit_stem}-state.npz"
        archive_arrays: dict[str, Any] = dict(arrays)
        np.savez_compressed(state_path, **archive_arrays)

        payload = {
            "schema_version": 1,
            "converged": True,
            "forces_evaluated": derivative is not None,
            "profile": self.profile,
            "solvent": self.solvent,
            "energies_hartree": components,
            "gas_mace_energy_ev": float(gas_state.energy_ev),
            "solvent_intrinsic_mace_energy_ev": float(
                coupled.solvent_state.energy_ev
            ),
            "polarization_energy_identity_error_ev": (
                coupled.energy_identity_error_ev
            ),
            "scf": {
                "mixing": SCF_MIXING,
                "density_tolerance_e": SCF_DENSITY_TOLERANCE,
                "energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
                "maximum_iterations": SCF_MAX_ITERATIONS,
                "iterations": len(coupled.history),
                "history": list(coupled.history),
            },
            "providers": {
                "continuum": dict(
                    coupled.reaction_field.runtime_provenance
                ),
                "cds": dict(coupled.cds_result.runtime_provenance),
            },
            "adjoint": (
                None if derivative is None else derivative["adjoint"]
            ),
            "array_archive": str(state_path),
        }
        (self.audit_dir / f"{audit_stem}-result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        self._validate_atoms(atoms)
        self._validate_calculator(
            calculator,
            need_forces=need_forces,
        )
        gas_state = self._gas_state(
            calculator,
            atoms,
            need_forces=need_forces,
        )
        self._validate_density(
            gas_state.density_coefficients,
            len(atoms),
            name="Gas MACE-POLAR density",
        )
        coupled = self._coupled_state(
            atoms,
            calculator,
            gas_state,
        )

        components = self._engine.energy_components(
            gas_state,
            coupled,
        )
        total_energy = components["delta_g_solv"]

        correction_forces = None
        derivative = None
        if need_forces:
            correction_forces, derivative = (
                self._solvent_correction_force(
                    atoms,
                    calculator,
                    gas_state,
                    coupled,
                )
            )
        self._write_result_audit(
            atoms=atoms,
            gas_state=gas_state,
            coupled=coupled,
            components=components,
            derivative=derivative,
        )

        runtime_provenance = {
            **self.provenance,
            "converged": True,
            "iterations": len(coupled.history),
            "continuum_provider": dict(
                coupled.reaction_field.runtime_provenance
            ),
            "cds_provider": dict(
                coupled.cds_result.runtime_provenance
            ),
            "calculator_profile": getattr(
                calculator,
                "route2_smd_profile",
                None,
            ),
            "mace_torch_version": getattr(
                calculator,
                "mace_torch_version",
                None,
            ),
            "mace_dtype": str(getattr(calculator, "dtype", None)),
            "mace_long_range_evaluator": dict(
                getattr(
                    calculator,
                    "long_range_evaluator_provenance",
                    {},
                )
            ),
            "audit_directory": (
                None
                if self.audit_dir is None
                else str(self.audit_dir)
            ),
        }
        return SolvationResult(
            energy_hartree=total_energy,
            forces_hartree_per_angstrom=correction_forces,
            components_hartree=components,
            provenance=runtime_provenance,
        )


# Compatibility alias for imports written before the ddCOSMO equation profile
# existed.  Provider dispatch uses the equation-neutral class name.
DDPCMSMDImplicitSolvation = PyDDXSMDImplicitSolvation


__all__ = [
    "ADJOINT_ABSOLUTE_TOLERANCE",
    "ADJOINT_MAX_ITERATIONS",
    "ADJOINT_RELATIVE_TOLERANCE",
    "DDPCM_ETA",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_PROFILE",
    "DDPCM_LMAX",
    "DDPCM_N_LEBEDEV",
    "DDPCM_SMD_PROFILE",
    "DDPCM_SOLVER_TOLERANCE",
    "DDPCMSMDImplicitSolvation",
    "FORCE_STATE_ENERGY_TOLERANCE_EV",
    "PyDDXSMDImplicitSolvation",
    "SCF_DENSITY_TOLERANCE",
    "SCF_ENERGY_TOLERANCE_EV",
    "SCF_MAX_ITERATIONS",
    "SCF_MIXING",
    "WATER_STATIC_DIELECTRIC",
]
