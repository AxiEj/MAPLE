"""Self-consistent MACE-POLAR/pyddx/SMD research energy provider.

This provider is deliberately separate from the public PCMSolver/GePol energy
proof of concept.  One profile-selected pyddx ddPCM or scaled-ddCOSMO object
owns the scalar polarization energy, reaction-field forward/adjoint maps, and
research-only coordinate derivative evidence.  The official PySCF SMD CDS entrypoint
supplies its scalar energy and matching analytic gradient.  Components from
different continuum equations are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ase.units import Hartree

from ....route2_energy_ledger import (
    PCM_HALF_COUPLING_ONLY_V1,
    route2_energy_composition_description,
)
from ....route2_smd_profiles import (
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
    DDPCM_GAFF2_CARBONYL_O_PROFILE,
    DDPCM_MULTISOLVENT_SMD_PROFILE,
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
    DDPCM_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_PROFILE,
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
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
    TESTED_PYDDX_VERSION,
    PyDDXCOSMOReactionFieldLinearMap,
    PyDDXPCMReactionFieldLinearMap,
)
from .pyscf_smd_cds import pyscf_smd_cds
from .result import SinglePointDerivativeEvidence, SolvationResult
from .route2_domain import validate_route2_domain
from .route2_engine import (
    Route2ContinuumEngine,
    Route2CoupledState,
    Route2EngineSettings,
    Route2FiniteResolutionPolicy,
    SCF_ACCEPTED_RESIDUAL_SOURCE,
    SCF_ACTUAL_RESIDUAL_OBJECTIVE_FORMULA,
    SCF_FINITE_RESOLUTION_HISTORY_SOURCE,
    SCF_REJECTED_GROWTH_ACTION,
    Route2SCFConvergenceError,
    Route2SCFHistoryRecord,
)
from .route2_force_admission import (
    PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT,
)
from .route2_fixed_point import SAFEGUARDED_ANDERSON_SOLVER
from .smd_cds import route2_coulomb_radii

WATER_STATIC_DIELECTRIC = 78.39
DDPCM_LMAX = 15
DDPCM_N_LEBEDEV = 1202
# ddX stops on a relative hnorm iterate change; 1e-14 is the empirical inner
# resolution selected by the archived index248 diagnostic.
DDPCM_SOLVER_TOLERANCE = 1.0e-14
DDPCM_ETA = 0.1
SCF_MIXING = 1.0
SCF_DENSITY_TOLERANCE = 2.0e-12
SCF_DIPOLE_TOLERANCE_E_ANGSTROM = 2.0e-12
SCF_ENERGY_TOLERANCE_EV = 1.0e-10
SCF_MAX_ITERATIONS = 100
SCF_SOLVER = SAFEGUARDED_ANDERSON_SOLVER
SCF_ANDERSON_DEPTH = 6
SCF_ANDERSON_REGULARIZATION = 1.0e-12
SCF_ANDERSON_COEFFICIENT_L1_LIMIT = 100.0
SCF_ANDERSON_STEP_RATIO_LIMIT = 100.0
SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT = 2.0
SCF_TOTAL_CHARGE_E = 0.0
SCF_FINITE_RESOLUTION_POLICY_VERSION = "finite-resolution-stagnation-v2"
SCF_FINITE_RESOLUTION_HISTORY_LENGTH = 7
SCF_FINITE_RESOLUTION_MAP_REPLAY_COUNT = 3
SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E = 1.0e-10
SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM = 1.0e-10
SCF_FINITE_RESOLUTION_POTENTIAL_SPAN_TOLERANCE_EV = 1.0e-10
SCF_FINITE_RESOLUTION_GRADIENT_SPAN_TOLERANCE_EV_PER_ANGSTROM = 1.0e-10
SCF_FINITE_RESOLUTION_LEDGER_SPAN_TOLERANCE_EV = 1.0e-10
ADJOINT_RELATIVE_TOLERANCE = 1.0e-10
ADJOINT_ABSOLUTE_TOLERANCE = 1.0e-13
ADJOINT_MAX_ITERATIONS = 100
ENERGY_IDENTITY_TOLERANCE_EV = 2.0e-10
FORCE_STATE_ENERGY_TOLERANCE_EV = 1.0e-9
NEUTRAL_DENSITY_TOLERANCE = 1.0e-8

PYDDX_DERIVATIVE_EVIDENCE_ONLY_ERROR = (
    "Route 2 pyddx does not expose forces through its production result API; "
    "use evaluate_single_point_derivative_evidence() only for explicit "
    "research validation."
)

_MACE_POLAR_IDENTIFIER = "polar-1-m"
_MACE_POLAR_RELEASE_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_polar_1/MACE-POLAR-1-M.model"
)
_MACE_POLAR_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
_MACE_POLAR_CHECKPOINT_SIZE_BYTES = 68_133_235
_MACE_TORCH_VERSION = "0.3.16"
_GRAPH_LONGRANGE_VERSION = "0.4.0"
_TORCH_VERSION = "2.12.0+cu130"

_MULTISOLVENT_DDPCM_FINITE_RESOLUTION_POLICY = (
    Route2FiniteResolutionPolicy(
        version=SCF_FINITE_RESOLUTION_POLICY_VERSION,
        history_length=SCF_FINITE_RESOLUTION_HISTORY_LENGTH,
        map_replay_count=SCF_FINITE_RESOLUTION_MAP_REPLAY_COUNT,
        monopole_residual_ceiling_e=(
            SCF_FINITE_RESOLUTION_MONOPOLE_CEILING_E
        ),
        dipole_residual_ceiling_e_angstrom=(
            SCF_FINITE_RESOLUTION_DIPOLE_CEILING_E_ANGSTROM
        ),
        potential_span_tolerance_ev=(
            SCF_FINITE_RESOLUTION_POTENTIAL_SPAN_TOLERANCE_EV
        ),
        gradient_span_tolerance_ev_per_angstrom=(
            SCF_FINITE_RESOLUTION_GRADIENT_SPAN_TOLERANCE_EV_PER_ANGSTROM
        ),
        ledger_span_tolerance_ev=(
            SCF_FINITE_RESOLUTION_LEDGER_SPAN_TOLERANCE_EV
        ),
    )
)

_DDPCM_ENGINE_SETTINGS = Route2EngineSettings(
    continuum_label="ddPCM",
    scf_mixing=SCF_MIXING,
    scf_density_tolerance=SCF_DENSITY_TOLERANCE,
    scf_dipole_tolerance_e_angstrom=(
        SCF_DIPOLE_TOLERANCE_E_ANGSTROM
    ),
    scf_energy_tolerance_ev=SCF_ENERGY_TOLERANCE_EV,
    scf_max_iterations=SCF_MAX_ITERATIONS,
    adjoint_relative_tolerance=ADJOINT_RELATIVE_TOLERANCE,
    adjoint_absolute_tolerance=ADJOINT_ABSOLUTE_TOLERANCE,
    adjoint_max_iterations=ADJOINT_MAX_ITERATIONS,
    energy_identity_tolerance_ev=ENERGY_IDENTITY_TOLERANCE_EV,
    force_state_energy_tolerance_ev=FORCE_STATE_ENERGY_TOLERANCE_EV,
    neutral_density_tolerance=NEUTRAL_DENSITY_TOLERANCE,
    scf_solver=SCF_SOLVER,
    scf_anderson_depth=SCF_ANDERSON_DEPTH,
    scf_anderson_regularization=SCF_ANDERSON_REGULARIZATION,
    scf_anderson_coefficient_l1_limit=(SCF_ANDERSON_COEFFICIENT_L1_LIMIT),
    scf_anderson_step_ratio_limit=SCF_ANDERSON_STEP_RATIO_LIMIT,
    scf_anderson_residual_growth_limit=(SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT),
    scf_total_charge_e=SCF_TOTAL_CHARGE_E,
)


_CONTINUUM_LABELS = {
    "ddpcm": "ddPCM",
    "ddcosmo": "ddCOSMO",
}


def _engine_settings(
    electrostatics_model: str,
    profile: str,
) -> Route2EngineSettings:
    try:
        label = _CONTINUUM_LABELS[electrostatics_model]
    except KeyError as exc:
        raise ValueError(
            "The pyddx Route-2 provider requires electrostatics_model="
            "ddpcm or ddcosmo."
        ) from exc
    policy = (
        _MULTISOLVENT_DDPCM_FINITE_RESOLUTION_POLICY
        if (
            label == _DDPCM_ENGINE_SETTINGS.continuum_label
            and profile
            in {
                DDPCM_MULTISOLVENT_SMD_PROFILE,
                DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
            }
        )
        else None
    )
    return replace(
        _DDPCM_ENGINE_SETTINGS,
        continuum_label=label,
        scf_finite_resolution_policy=policy,
    )


def _torch_thread_count() -> int:
    import torch

    return int(torch.get_num_threads())


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _array_sha256(values: np.ndarray, *, dtype: str) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _normalized_mol2_atom_types(atoms) -> tuple[str, ...] | None:
    mol2 = atoms.info.get("mol2")
    if not isinstance(mol2, dict) or mol2.get("atom_types") is None:
        return None
    return tuple(str(atom_type).strip().lower() for atom_type in mol2["atom_types"])


@dataclass
class PyDDXSMDImplicitSolvation:
    """SMD-CDS correction with one profile-selected pyddx equation."""

    atoms: Any
    solvation_options: dict[str, Any]
    audit_dir: Path | None = None

    supported_properties = frozenset({"energy"})

    def __post_init__(self) -> None:
        self.solvation_options = dict(self.solvation_options)
        if "profile" not in self.solvation_options:
            raise ValueError(
                "Route 2 provider=pyddx requires an explicit " "versioned profile."
            )
        self.provider = str(self.solvation_options.get("provider", "")).lower()
        self.profile = str(self.solvation_options["profile"]).lower()
        self.response = str(self.solvation_options.get("response", "scf")).lower()
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
            if self.profile_spec.dielectric_policy == "legacy-water-78.39"
            else self.solvent_spec.descriptors.dielectric
        )
        validate_route2_domain(self.atoms)
        self._reference_numbers = np.asarray(
            self.atoms.numbers,
            dtype=int,
        ).copy()

        self._reference_mol2_atom_types = _normalized_mol2_atom_types(self.atoms)
        self.coulomb_radii_angstrom = route2_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            solvent=self.solvent,
            atom_types=self._reference_mol2_atom_types,
            profile=self.profile,
        )
        self._engine = Route2ContinuumEngine(
            reaction_field_factory=self._build_reaction_field,
            cds_evaluator=self._evaluate_cds,
            settings=_engine_settings(
                self.electrostatics_model,
                self.profile,
            ),
        )
        self._cached_state: Route2CoupledState | None = None

        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)

        numerics = {
            "dielectric": self.continuum_dielectric,
            "dielectric_policy": self.profile_spec.dielectric_policy,
            "coulomb_radii_policy": (self.profile_spec.coulomb_radii_policy),
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "pyddx_n_proc": self.profile_spec.ddpcm_n_proc,
            "pyddx_solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "pyddx_eta": DDPCM_ETA,
            "scf_mixing": SCF_MIXING,
            "scf_density_tolerance_e": SCF_DENSITY_TOLERANCE,
            "scf_dipole_tolerance_e_angstrom": (
                SCF_DIPOLE_TOLERANCE_E_ANGSTROM
            ),
            "scf_energy_tolerance_ev": SCF_ENERGY_TOLERANCE_EV,
            "scf_maximum_iterations": SCF_MAX_ITERATIONS,
            "scf_solver": SCF_SOLVER,
            "scf_anderson_depth": SCF_ANDERSON_DEPTH,
            "scf_anderson_regularization": SCF_ANDERSON_REGULARIZATION,
            "scf_anderson_coefficient_l1_limit": (SCF_ANDERSON_COEFFICIENT_L1_LIMIT),
            "scf_anderson_step_ratio_limit": (SCF_ANDERSON_STEP_RATIO_LIMIT),
            "scf_anderson_residual_growth_limit": (SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT),
            "scf_total_charge_e": SCF_TOTAL_CHARGE_E,
            "scf_finite_resolution_policy": (
                None
                if self._engine.settings.scf_finite_resolution_policy is None
                else self._engine.settings.scf_finite_resolution_policy.as_dict()
            ),
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
        if self.profile_spec.uses_gaff2_carbonyl_oxygen:
            cavity_radii_provenance = (
                "SMD Coulomb radii with GAFF/GAFF2 carbonyl oxygen "
                "(atom type o) overridden to 1.70 A"
            )
        elif self.profile_spec.coulomb_radii_policy == "pyscf-smd-2.13.1":
            cavity_radii_provenance = (
                "PySCF 2.13.1 SMD solvent-acidity-dependent Coulomb "
                "radii with revised Br=2.60 A and I=2.74 A"
            )
        elif self.profile_spec.coulomb_radii_policy == "smd-water-reference-smd18-v1":
            cavity_radii_provenance = (
                "SMD/SMD18 atomic-number-indexed water Coulomb "
                "radii (P=2.12 A, S=2.49 A, Cl=2.38 A)"
            )
        else:
            raise RuntimeError(
                "Unsupported Route-2 Coulomb-radii provenance policy: "
                f"{self.profile_spec.coulomb_radii_policy!r}."
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
            "density_source": ("official MACE-POLAR-1-M l<=1 residual charge density"),
            "density_interpretation": (
                "coarse-grained net charge density, not a QM electron density"
            ),
            "electrostatics": self.continuum_label,
            "electrostatics_model": (self.profile_spec.electrostatics_model),
            "solute_source": self.profile_spec.solute_source,
            "reaction_field_projector": (self.profile_spec.reaction_field_projector),
            "nonpolar_model": self.profile_spec.nonpolar_model,
            "strict_original_smd_equivalence": (
                self.profile_spec.strict_original_smd_equivalence
            ),
            "pcm_projection": "atom-centred l<=1 real spherical multipoles",
            "cavity_radii": cavity_radii_provenance,
            "mace_long_range_evaluator": (self.profile_spec.mace_long_range_evaluator),
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
            "scientific_status": "single-point-energy-research",
            "solution_phase_pes": False,
            "forces_available": False,
            # Both frozen ledgers now have their own outer-adjoint derivative
            # specification.  This advertises only an explicit single-point
            # research check: pyddx's variable active surface topology still
            # prevents a public PES/ASE force contract.
            "research_derivative_evidence_available": True,
            "research_derivative_evidence_scope": (
                "single-point validation only; not an ASE force or "
                "solution-phase PES capability"
            ),
            "accuracy_certified": False,
            "default_eligible": False,
            "electrostatic_energy_ledger": (
                self.profile_spec.electrostatic_energy_ledger
            ),
            "energy_composition": route2_energy_composition_description(
                self.profile_spec.electrostatic_energy_ledger,
                continuum_symbol=self.continuum_label,
            ),
            "research_derivative_evidence_composition": (
                "-d[0.5*<c,P_R c> + G_CDS]/dR evaluated with the "
                "converged-density response eliminated by the direct-PCM "
                "ledger-specific adjoint"
                if self.profile_spec.electrostatic_energy_ledger
                == PCM_HALF_COUPLING_ONLY_V1
                else "-d(delta_G_solv)/dR evaluated with the "
                "converged-density response eliminated by one adjoint solve"
            ),
            "numerics": numerics,
        }
        self._write_manifest()

    def _validate_options(self) -> None:
        if self.solvation_options.get("experimental") is not True:
            raise ValueError(
                "The pyddx Route-2 research provider requires "
                "experimental=true explicitly."
            )
        if str(self.solvation_options.get("method", "")).lower() != "smd":
            raise ValueError("PyDDXSMDImplicitSolvation requires method=smd.")
        if self.provider != "pyddx":
            raise ValueError("PyDDXSMDImplicitSolvation requires provider=pyddx.")
        if self.profile not in SUPPORTED_PYDDX_SMD_PROFILES:
            supported = ", ".join(sorted(SUPPORTED_PYDDX_SMD_PROFILES))
            raise ValueError(
                "The pyddx Route-2 profile must be one of: " f"{supported}."
            )
        if not self.profile_spec.supports_solvent(self.solvent):
            raise ValueError(
                f"Route 2 profile={self.profile} does not support "
                f"solvent={self.solvent}."
            )
        if self.response != "scf":
            raise ValueError(
                "The pyddx Route-2 research provider requires response=scf."
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
            raise ValueError("Route 2 does not permit atom identity/order changes.")
        atom_types = _normalized_mol2_atom_types(atoms)
        if atom_types != self._reference_mol2_atom_types:
            raise ValueError(
                "Route 2 does not permit MOL2 atom type/order changes after "
                "the cavity radii are initialized."
            )

    def _validate_calculator(self, calculator, *, need_forces: bool) -> None:
        if calculator is None or not callable(getattr(calculator, "polar_state", None)):
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
            name for name in required if not callable(getattr(calculator, name, None))
        ]
        if missing:
            raise TypeError(
                "The pyddx Route-2 force candidate requires the complete "
                "MACE response API; missing: " + ", ".join(missing) + "."
            )

    def _finite_resolution_runtime_identity(
        self,
        calculator,
        atoms,
    ) -> dict[str, Any] | None:
        """Return the exact runtime lock that arms the profile-local fallback."""

        if self._engine.settings.scf_finite_resolution_policy is None:
            return None
        checkpoint = getattr(
            calculator,
            "mace_polar_checkpoint_provenance",
            None,
        )
        if not isinstance(checkpoint, Mapping):
            return None
        exact_checkpoint = {
            "identifier": _MACE_POLAR_IDENTIFIER,
            "release_url": _MACE_POLAR_RELEASE_URL,
            "sha256": _MACE_POLAR_CHECKPOINT_SHA256,
            "size_bytes": _MACE_POLAR_CHECKPOINT_SIZE_BYTES,
        }
        if any(checkpoint.get(key) != value for key, value in exact_checkpoint.items()):
            return None
        runtime = {
            "mace_torch_version": str(
                getattr(calculator, "mace_torch_version", "")
            ),
            "graph_longrange_version": str(
                getattr(calculator, "graph_longrange_version", "")
            ),
            "mace_long_range_evaluator_profile": str(
                getattr(
                    calculator,
                    "long_range_evaluator_profile",
                    "",
                )
            ),
            "mace_dtype": str(getattr(calculator, "dtype", "")),
            "device": str(getattr(calculator, "device", "")),
            "torch_threads": _torch_thread_count(),
            "torch_version": _torch_version(),
        }
        expected_runtime = {
            "mace_torch_version": _MACE_TORCH_VERSION,
            "graph_longrange_version": _GRAPH_LONGRANGE_VERSION,
            "mace_long_range_evaluator_profile": (
                self.profile_spec.mace_long_range_evaluator
            ),
            "mace_dtype": "torch.float64",
            "device": "cpu",
            "torch_threads": 1,
            "torch_version": _TORCH_VERSION,
        }
        if runtime != expected_runtime:
            return None
        positions = np.asarray(atoms.get_positions(), dtype=float)
        numbers = np.asarray(atoms.numbers, dtype=np.int64)
        radii = np.asarray(self.coulomb_radii_angstrom, dtype=float)
        return {
            "profile": self.profile,
            "solvent": self.solvent,
            "continuum_equation": self.electrostatics_model,
            "mace_checkpoint_identifier": exact_checkpoint["identifier"],
            "mace_checkpoint_release_url": exact_checkpoint["release_url"],
            "mace_checkpoint_sha256": exact_checkpoint["sha256"],
            "mace_checkpoint_size_bytes": exact_checkpoint["size_bytes"],
            **runtime,
            "pyddx_version": TESTED_PYDDX_VERSION,
            "pyddx_n_proc": self.profile_spec.ddpcm_n_proc,
            "pyddx_solver_tolerance": DDPCM_SOLVER_TOLERANCE,
            "continuum_dielectric": self.continuum_dielectric,
            "lmax": DDPCM_LMAX,
            "n_lebedev": DDPCM_N_LEBEDEV,
            "eta": DDPCM_ETA,
            "atomic_numbers_sha256": _array_sha256(
                numbers,
                dtype="<i8",
            ),
            "positions_angstrom_sha256": _array_sha256(
                positions,
                dtype="<f8",
            ),
            "cavity_radii_angstrom_sha256": _array_sha256(
                radii,
                dtype="<f8",
            ),
        }

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
        try:
            return self._engine.solve_coupled_state(
                atoms,
                calculator,
                gas_state,
                provider_cache_signature=(
                    self.solvent,
                    self.electrostatics_model,
                    self._reference_mol2_atom_types,
                ),
                finite_resolution_runtime_identity=(
                    self._finite_resolution_runtime_identity(
                        calculator,
                        atoms,
                    )
                ),
                electrostatic_energy_ledger=(
                    self.profile_spec.electrostatic_energy_ledger
                ),
            )
        except Route2SCFConvergenceError as exc:
            self._write_scf_failure_audit(
                atoms=atoms,
                gas_state=gas_state,
                error=exc,
            )
            raise

    def _scf_audit_payload(
        self,
        history: tuple[Route2SCFHistoryRecord, ...],
        *,
        convergence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = self._engine.settings
        policy = settings.scf_finite_resolution_policy
        return {
            "solver": settings.scf_solver,
            "mixing": settings.scf_mixing,
            "density_tolerance_e": settings.scf_density_tolerance,
            "dipole_density_tolerance_e_angstrom": (
                settings.scf_dipole_tolerance_e_angstrom
            ),
            "energy_tolerance_ev": settings.scf_energy_tolerance_ev,
            "maximum_iterations": settings.scf_max_iterations,
            "anderson_depth": settings.scf_anderson_depth,
            "anderson_regularization": settings.scf_anderson_regularization,
            "anderson_coefficient_l1_limit": (
                settings.scf_anderson_coefficient_l1_limit
            ),
            "anderson_step_ratio_limit": (
                settings.scf_anderson_step_ratio_limit
            ),
            "anderson_residual_growth_limit": (
                settings.scf_anderson_residual_growth_limit
            ),
            "anderson_residual_growth_objective": (
                SCF_ACTUAL_RESIDUAL_OBJECTIVE_FORMULA
            ),
            "anderson_residual_growth_rejection_inequality": (
                "Phi_trial > residual_growth_limit * Phi_anchor"
            ),
            "anderson_accepted_residual_source": SCF_ACCEPTED_RESIDUAL_SOURCE,
            "anderson_actual_growth_action": SCF_REJECTED_GROWTH_ACTION,
            "rejected_trials_count_toward_maximum_iterations": True,
            "rejected_trials_in_anderson_history": False,
            "rejected_trials_eligible_for_best_state": False,
            "finite_resolution_history_source": (
                SCF_FINITE_RESOLUTION_HISTORY_SOURCE
            ),
            "total_charge_e": settings.scf_total_charge_e,
            "residual_definition": (
                "unmixed neutral-tangent Pi0[M(P(c))-c]"
            ),
            "coefficient_units": {
                "monopole": "e",
                "dipole": "e angstrom",
            },
            "field_units": {
                "potential": "eV/e",
                "gradient": "eV/(e angstrom)",
            },
            "finite_resolution_policy": (
                None if policy is None else policy.as_dict()
            ),
            "convergence": (
                None if convergence is None else dict(convergence)
            ),
            "iterations": len(history),
            "history": list(history),
        }

    def _write_scf_failure_audit(
        self,
        *,
        atoms,
        gas_state,
        error: Route2SCFConvergenceError,
    ) -> None:
        if self.audit_dir is None:
            return
        audit_stem = f"route2-{self.electrostatics_model}"
        best_state_payload: dict[str, Any] | None = None
        if error.best_state is not None:
            best_state_path = self.audit_dir / f"{audit_stem}-failure-best-state.npz"
            arrays: dict[str, Any] = {
                "density_coefficients": error.best_state.density_coefficients,
                "response_density_coefficients": (
                    error.best_state.response_density_coefficients
                ),
                "reaction_field_values_ev": (
                    error.best_state.reaction_field_values_ev
                ),
            }
            if error.best_state.model_local_field_values_ev is not None:
                arrays["model_local_field_values_ev"] = (
                    error.best_state.model_local_field_values_ev
                )
            if error.best_state.model_field_features is not None:
                arrays["model_field_features"] = (
                    error.best_state.model_field_features
                )
            np.savez_compressed(best_state_path, **arrays)
            best_state_payload = {
                "iteration": error.best_state.iteration,
                "density_residual_e": error.best_state.density_residual_e,
                "intrinsic_energy_ev": error.best_state.intrinsic_energy_ev,
                "array_artifact": best_state_path.name,
                "array_keys": sorted(arrays),
            }
        payload = {
            "schema_version": 3,
            "converged": False,
            "error": str(error),
            "profile": self.profile,
            "solvent": self.solvent,
            "gas_mace_energy_ev": float(gas_state.energy_ev),
            "positions_angstrom": np.asarray(
                atoms.get_positions(),
                dtype=float,
            ).tolist(),
            "cavity_radii_angstrom": np.asarray(
                self.coulomb_radii_angstrom,
                dtype=float,
            ).tolist(),
            "scf": self._scf_audit_payload(error.history),
            "best_iteration_state": best_state_payload,
        }
        (self.audit_dir / f"{audit_stem}-failure.json").write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
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
            force_admission_continuum=(
                PYDDX_HARD_ACTIVE_SET_SMOOTHNESS_CONTRACT
            ),
            electrostatic_energy_ledger=(
                self.profile_spec.electrostatic_energy_ledger
            ),
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
            "response_density_coefficients": (
                coupled.response_density_coefficients
            ),
            "reaction_field_values_ev": (coupled.reaction_field_values_ev),
        }
        if derivative is not None:
            arrays.update(
                {
                    key: np.asarray(value, dtype=float)
                    for key, value in derivative.items()
                    if key not in {"adjoint", "force_admission"}
                }
            )
        audit_stem = f"route2-{self.electrostatics_model}"
        state_path = self.audit_dir / f"{audit_stem}-state.npz"
        archive_arrays: dict[str, Any] = dict(arrays)
        np.savez_compressed(state_path, **archive_arrays)

        payload = {
            "schema_version": 3,
            "converged": True,
            "forces_evaluated": derivative is not None,
            "profile": self.profile,
            "solvent": self.solvent,
            "energies_hartree": components,
            "electrostatic_energy_ledger": (
                self.profile_spec.electrostatic_energy_ledger
            ),
            "gas_mace_energy_ev": float(gas_state.energy_ev),
            "solvent_intrinsic_mace_energy_ev": float(coupled.solvent_state.energy_ev),
            "field_conditioned_mace_energy_change_hartree": (
                (
                    float(coupled.solvent_state.energy_ev)
                    - float(gas_state.energy_ev)
                )
                / Hartree
            ),
            "polarization_energy_identity_error_ev": (coupled.energy_identity_error_ev),
            "scf": self._scf_audit_payload(
                coupled.history,
                convergence=coupled.scf_convergence,
            ),
            "providers": {
                "continuum": dict(coupled.reaction_field.runtime_provenance),
                "cds": dict(coupled.cds_result.runtime_provenance),
            },
            "adjoint": (None if derivative is None else derivative["adjoint"]),
            "force_admission": (
                None if derivative is None else derivative["force_admission"]
            ),
            "array_archive": str(state_path),
        }
        (self.audit_dir / f"{audit_stem}-result.json").write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if need_forces:
            raise NotImplementedError(PYDDX_DERIVATIVE_EVIDENCE_ONLY_ERROR)
        return self._evaluate(
            atoms,
            need_forces=False,
            calculator=calculator,
        )

    def evaluate_single_point_derivative_evidence(
        self,
        atoms,
        *,
        calculator=None,
    ) -> SinglePointDerivativeEvidence:
        """Return explicitly labelled derivative evidence outside ASE/PES APIs."""

        result = self._evaluate(
            atoms,
            need_forces=True,
            calculator=calculator,
        )
        if result.forces_hartree_per_angstrom is None:
            raise RuntimeError("pyddx derivative evidence did not produce forces.")
        return SinglePointDerivativeEvidence(
            energy_hartree=result.energy_hartree,
            forces_hartree_per_angstrom=result.forces_hartree_per_angstrom,
            components_hartree=result.components_hartree,
            provenance={
                **result.provenance,
                "forces_available": False,
                "research_derivative_evidence": True,
                "research_derivative_evidence_scope": (
                    "single-point validation only; not an ASE force or "
                    "solution-phase PES capability"
                ),
            },
        )

    def _evaluate(
        self,
        atoms,
        *,
        need_forces: bool,
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
            electrostatic_energy_ledger=(
                self.profile_spec.electrostatic_energy_ledger
            ),
        )
        total_energy = components["delta_g_solv"]

        correction_forces = None
        derivative = None
        if need_forces:
            correction_forces, derivative = self._solvent_correction_force(
                atoms,
                calculator,
                gas_state,
                coupled,
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
            "scf_convergence": dict(coupled.scf_convergence),
            "continuum_provider": dict(coupled.reaction_field.runtime_provenance),
            "cds_provider": dict(coupled.cds_result.runtime_provenance),
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
                None if self.audit_dir is None else str(self.audit_dir)
            ),
        }
        if derivative is not None:
            runtime_provenance["force_admission"] = derivative[
                "force_admission"
            ]
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
    "DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE",
    "DDPCM_LMAX",
    "DDPCM_N_LEBEDEV",
    "DDPCM_SMD_PROFILE",
    "DDPCM_SOLVER_TOLERANCE",
    "DDPCMSMDImplicitSolvation",
    "FORCE_STATE_ENERGY_TOLERANCE_EV",
    "PYDDX_DERIVATIVE_EVIDENCE_ONLY_ERROR",
    "PyDDXSMDImplicitSolvation",
    "SCF_DENSITY_TOLERANCE",
    "SCF_ENERGY_TOLERANCE_EV",
    "SCF_MAX_ITERATIONS",
    "SCF_MIXING",
    "SCF_SOLVER",
    "SCF_ANDERSON_DEPTH",
    "SCF_ANDERSON_REGULARIZATION",
    "SCF_ANDERSON_COEFFICIENT_L1_LIMIT",
    "SCF_ANDERSON_STEP_RATIO_LIMIT",
    "SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT",
    "SCF_TOTAL_CHARGE_E",
    "WATER_STATIC_DIELECTRIC",
]
