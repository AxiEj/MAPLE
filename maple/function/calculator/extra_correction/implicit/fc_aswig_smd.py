"""Aqueous fixed-topology C-PCM/SMD Route-2 research provider.

This provider is intentionally separate from the legacy PCMSolver and pyddx
providers.  It is the narrow composition needed to assess a conservative
Route-2 force profile without changing their validated energy-only identities:

``MACE-POLAR local jet -> JGP94 fixed-topology amplitude C-PCM
                   + JGP94 fixed-topology aqueous SMD-CDS``.

The class exposes ordinary energy evaluation and a deliberately labelled
single-point derivative-evidence API.  It remains force-closed until the
profile-level finite-difference, symmetry, path, closed-work, and NVE gates
are supplied to the admission certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.units import Hartree

from ....route2_energy_ledger import PCM_HALF_COUPLING_ONLY_V1
from ....route2_smd_profiles import (
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
    route2_smd_profile_spec,
)
from ....route2_solvents import normalize_route2_solvent_name
from .result import SinglePointDerivativeEvidence, SolvationResult
from .route2_domain import validate_route2_domain
from .route2_engine import (
    Route2ContinuumEngine,
    Route2CoupledState,
    Route2EngineSettings,
)
from .route2_electronic_model import (
    Route2ElectronicModel,
    resolve_route2_electronic_model,
    validate_route2_electronic_model_capabilities,
)
from .route2_fc_aswig_smd_cds import FixedTopologyAqueousSMDCDSResult
from .route2_force_admission import (
    FC_ASWIG_JGP94_D2_DIRECT_PCM_FORCE_PES_VALIDATION_CONTRACT,
    JGP94_FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT,
    UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT,
)
from .route2_fixed_point import SAFEGUARDED_ANDERSON_SOLVER
from .route2_jgp94_fc_aswig import (
    JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP,
    JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse,
    JGP94BodyFrameFixedTopologyAqueousSMDCDS,
)
from .route2_root_study import (
    Route2MultiStartRootAgreement,
    evaluate_route2_multi_start_root_agreement,
)
from .smd_cds import route2_coulomb_radii


FC_ASWIG_STATIC_DIELECTRIC = 78.39
FC_ASWIG_LEBEDEV_ORDER = 15
FC_ASWIG_SCF_DENSITY_TOLERANCE_E = 2.0e-12
FC_ASWIG_SCF_DIPOLE_TOLERANCE_E_ANGSTROM = 2.0e-12
FC_ASWIG_SCF_ENERGY_TOLERANCE_EV = 1.0e-10
FC_ASWIG_SCF_MAX_ITERATIONS = 100
FC_ASWIG_SCF_SOLVER = SAFEGUARDED_ANDERSON_SOLVER
FC_ASWIG_SCF_ANDERSON_DEPTH = 6
FC_ASWIG_SCF_ANDERSON_REGULARIZATION = 1.0e-12
FC_ASWIG_SCF_ANDERSON_COEFFICIENT_L1_LIMIT = 100.0
FC_ASWIG_SCF_ANDERSON_STEP_RATIO_LIMIT = 100.0
FC_ASWIG_SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT = 2.0
FC_ASWIG_ADJOINT_RELATIVE_TOLERANCE = 1.0e-10
FC_ASWIG_ADJOINT_ABSOLUTE_TOLERANCE = 1.0e-13
FC_ASWIG_ADJOINT_MAX_ITERATIONS = 100
FC_ASWIG_ENERGY_IDENTITY_TOLERANCE_EV = 2.0e-10
FC_ASWIG_FORCE_STATE_ENERGY_TOLERANCE_EV = 1.0e-9
FC_ASWIG_NEUTRAL_DENSITY_TOLERANCE = 1.0e-8

FC_ASWIG_DERIVATIVE_EVIDENCE_ONLY_ERROR = (
    "The fixed-topology Route-2 profile exposes derivative evidence only until "
    "its complete solution-phase PES admission record is frozen and passed."
)

_SUPPORTED_FC_ASWIG_AQUEOUS_DIRECT_PCM_PROFILES = frozenset(
    {
        FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
        FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
    }
)


_FC_ASWIG_ENGINE_SETTINGS = Route2EngineSettings(
    continuum_label="JGP94-FC-aSWIG-CPCM",
    scf_mixing=1.0,
    scf_density_tolerance=FC_ASWIG_SCF_DENSITY_TOLERANCE_E,
    scf_dipole_tolerance_e_angstrom=(
        FC_ASWIG_SCF_DIPOLE_TOLERANCE_E_ANGSTROM
    ),
    scf_energy_tolerance_ev=FC_ASWIG_SCF_ENERGY_TOLERANCE_EV,
    scf_max_iterations=FC_ASWIG_SCF_MAX_ITERATIONS,
    # This changes only the numerical root finder: all acceptance checks use
    # the unmixed physical residual in Route2ContinuumEngine.  Picard stalled
    # close to the strict 2e-12 e/A residual on otherwise smooth, displaced
    # geometries, whereas the existing safeguarded-Anderson solver preserves
    # the same root equation and has rollback on actual-residual growth.
    scf_solver=FC_ASWIG_SCF_SOLVER,
    scf_anderson_depth=FC_ASWIG_SCF_ANDERSON_DEPTH,
    scf_anderson_regularization=FC_ASWIG_SCF_ANDERSON_REGULARIZATION,
    scf_anderson_coefficient_l1_limit=(
        FC_ASWIG_SCF_ANDERSON_COEFFICIENT_L1_LIMIT
    ),
    scf_anderson_step_ratio_limit=FC_ASWIG_SCF_ANDERSON_STEP_RATIO_LIMIT,
    scf_anderson_residual_growth_limit=(
        FC_ASWIG_SCF_ANDERSON_RESIDUAL_GROWTH_LIMIT
    ),
    adjoint_relative_tolerance=FC_ASWIG_ADJOINT_RELATIVE_TOLERANCE,
    adjoint_absolute_tolerance=FC_ASWIG_ADJOINT_ABSOLUTE_TOLERANCE,
    adjoint_max_iterations=FC_ASWIG_ADJOINT_MAX_ITERATIONS,
    energy_identity_tolerance_ev=FC_ASWIG_ENERGY_IDENTITY_TOLERANCE_EV,
    force_state_energy_tolerance_ev=FC_ASWIG_FORCE_STATE_ENERGY_TOLERANCE_EV,
    neutral_density_tolerance=FC_ASWIG_NEUTRAL_DENSITY_TOLERANCE,
    scf_total_charge_e=0.0,
)


@dataclass(frozen=True)
class _FCAqueousGeometryBundle:
    """One shared frame and its two same-geometry continuum components."""

    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    response: JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse
    cds: JGP94BodyFrameFixedTopologyAqueousSMDCDS

    def matches(self, atoms) -> bool:
        return np.array_equal(
            self.atomic_numbers,
            np.asarray(atoms.numbers, dtype=int),
        ) and np.array_equal(
            self.positions_angstrom,
            np.asarray(atoms.get_positions(), dtype=float),
        )


@dataclass
class FixedTopologyASWIGAqueousSMDImplicitSolvation:
    """A water-only direct-PCM Route-2 provider with smooth fixed topology."""

    atoms: Any
    solvation_options: dict[str, Any]
    audit_dir: Path | None = None

    supported_properties = frozenset({"energy"})

    def __post_init__(self) -> None:
        self.solvation_options = dict(self.solvation_options)
        self.provider = str(self.solvation_options.get("provider", "")).lower()
        self.profile = str(self.solvation_options.get("profile", "")).lower()
        self.solvent = normalize_route2_solvent_name(
            self.solvation_options.get("implicit", "")
        )
        self.response = str(self.solvation_options.get("response", "scf")).lower()
        self.standard_state = str(
            self.solvation_options.get("standard_state", "1m")
        ).lower()
        self._validate_options()
        self._public_force_profile = route2_smd_profile_spec(
            self.profile
        ).force_release_eligible
        self.supported_properties = frozenset(
            {"energy", "forces"}
            if self._public_force_profile
            else {"energy"}
        )
        validate_route2_domain(self.atoms)
        self._reference_atomic_numbers = np.asarray(
            self.atoms.numbers,
            dtype=int,
        ).copy()
        self.coulomb_radii_angstrom = route2_coulomb_radii(
            self.atoms.get_chemical_symbols(),
            solvent=self.solvent,
            profile=self.profile,
        )
        self._engine = Route2ContinuumEngine(
            reaction_field_factory=self._build_reaction_field,
            cds_evaluator=self._evaluate_cds,
            settings=_FC_ASWIG_ENGINE_SETTINGS,
        )
        self._bundle: _FCAqueousGeometryBundle | None = None
        self._cached_state: Route2CoupledState | None = None
        if self.audit_dir is not None:
            self.audit_dir = Path(self.audit_dir).resolve()
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = {
            "provider": "fc-aswig",
            "method": "smd",
            "profile": self.profile,
            "scientific_identity": (
                "MACE-POLAR/(l<=1)-point-multipole + JGP94 fixed-topology "
                "amplitude-CPCM + fixed-topology aqueous SMD-CDS"
            ),
            "solvent": self.solvent,
            "response": "scf",
            "standard_state": "1M(gas)->1M(solution)",
            "standard_state_correction_hartree": 0.0,
            "density_source": "official MACE-POLAR-1-M l<=1 residual charge density",
            "density_interpretation": "coarse-grained net charge density, not a QM electron density",
            "electrostatics": "C-PCM",
            "electrostatics_model": "cpcm",
            "solute_source": "point-multipole-l1",
            "reaction_field_projector": "local-jet",
            "nonpolar_model": "fixed-topology-aqueous-smd-cds",
            "strict_original_smd_equivalence": False,
            "electrostatic_energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
            "laboratory_grid_rotation_remedy": "jgp94-nuclear-charge-principal-frame-v1",
            "minimum_relative_eigengap_guard": (
                JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP
            ),
            "mace_geometry_frame_policy": route2_smd_profile_spec(
                self.profile
            ).mace_geometry_frame_policy,
            "lebedev_order": FC_ASWIG_LEBEDEV_ORDER,
            "static_dielectric": FC_ASWIG_STATIC_DIELECTRIC,
            "scf_solver": FC_ASWIG_SCF_SOLVER,
            "scf_mixing": _FC_ASWIG_ENGINE_SETTINGS.scf_mixing,
            "scf_density_tolerance_e": FC_ASWIG_SCF_DENSITY_TOLERANCE_E,
            "scf_dipole_tolerance_e_angstrom": (
                FC_ASWIG_SCF_DIPOLE_TOLERANCE_E_ANGSTROM
            ),
            "scf_energy_tolerance_ev": FC_ASWIG_SCF_ENERGY_TOLERANCE_EV,
            "forces_available": self._public_force_profile,
            "research_derivative_evidence_available": (
                not self._public_force_profile
            ),
            "research_derivative_evidence_scope": (
                "same-scalar single-point evidence only; no public force, "
                "optimizer, scan, or MD capability before PES admission"
                if not self._public_force_profile
                else "superseded by the bounded force-v3 admission contract"
            ),
            "solution_phase_pes": self._public_force_profile,
            "solution_phase_pes_scope": (
                "neutral, closed-shell, non-periodic, 16--500 Da connected "
                "molecules in water; local-jet only; nondegenerate JGP94 frame; "
                "per-geometry root, conditioning, and force-admission gates"
                if self._public_force_profile
                else "not publicly available"
            ),
            "accuracy_certified": False,
            "default_eligible": False,
        }

    def _validate_options(self) -> None:
        if self.provider != "fc-aswig":
            raise ValueError(
                "Fixed-topology Route 2 requires provider='fc-aswig'."
            )
        if self.profile not in _SUPPORTED_FC_ASWIG_AQUEOUS_DIRECT_PCM_PROFILES:
            raise ValueError(
                "Fixed-topology Route 2 requires profile to be one of: "
                + ", ".join(
                    repr(name)
                    for name in sorted(_SUPPORTED_FC_ASWIG_AQUEOUS_DIRECT_PCM_PROFILES)
                )
                + "."
            )
        spec = route2_smd_profile_spec(self.profile)
        if (
            spec.provider != self.provider
            or spec.electrostatic_energy_ledger != PCM_HALF_COUPLING_ONLY_V1
            or spec.electrostatics_model != "cpcm"
        ):
            raise RuntimeError("The fixed-topology Route-2 profile contract is invalid.")
        if self.solvent != "water":
            raise ValueError("The fixed-topology aqueous SMD-CDS profile supports water only.")
        if self.response != "scf":
            raise ValueError("Fixed-topology Route 2 requires response='scf'.")
        if self.standard_state != "1m":
            raise ValueError("Fixed-topology Route 2 requires standard_state='1m'.")

    def _validate_atoms(self, atoms) -> None:
        validate_route2_domain(atoms)
        numbers = np.asarray(atoms.numbers, dtype=int)
        if not np.array_equal(numbers, self._reference_atomic_numbers):
            raise ValueError(
                "Fixed-topology Route 2 cannot change atomic identities after "
                "provider construction."
            )

    def _validate_public_force_calculator(self, calculator) -> None:
        """Reject a public force request unless its MACE transform is exact."""

        if not self._public_force_profile:
            return
        spec = route2_smd_profile_spec(self.profile)
        actual_policy = str(
            getattr(calculator, "route2_mace_geometry_frame_policy", "")
        ).strip().lower()
        actual_evaluator = str(
            getattr(calculator, "long_range_evaluator_profile", "")
        ).strip().lower()
        if actual_policy != spec.mace_geometry_frame_policy:
            raise TypeError(
                "The Route-2 force-v3 profile requires the registered "
                "JGP94-D2 MACE geometry transform."
            )
        if actual_evaluator != spec.mace_long_range_evaluator:
            raise TypeError(
                "The Route-2 force-v3 profile requires its registered "
                "MACE long-range evaluator."
            )

    def _validate_electronic_model(
        self,
        calculator,
        *,
        need_forces: bool,
    ) -> Route2ElectronicModel:
        if calculator is None:
            raise TypeError(
                "The fixed-topology Route-2 provider requires an electronic model."
            )
        model = resolve_route2_electronic_model(calculator)
        spec = route2_smd_profile_spec(self.profile)
        validate_route2_electronic_model_capabilities(
            model,
            expected_model_family=spec.electronic_model_family,
            expected_source_space=spec.electronic_source_space,
            expected_profile_binding=spec.electronic_profile_binding,
            expected_field_evaluator=spec.model_field_evaluator,
            expected_energy_semantics=spec.electronic_energy_semantics,
            reaction_field_projector=spec.reaction_field_projector,
            electrostatic_energy_ledger=spec.electrostatic_energy_ledger,
            need_forces=need_forces,
        )
        return model

    def _force_pes_validation_contract(self):
        if self._public_force_profile:
            return FC_ASWIG_JGP94_D2_DIRECT_PCM_FORCE_PES_VALIDATION_CONTRACT
        return UNSPECIFIED_FORCE_PES_VALIDATION_CONTRACT

    def _geometry_bundle(self, atoms) -> _FCAqueousGeometryBundle:
        cached = self._bundle
        if cached is not None and cached.matches(atoms):
            return cached
        positions = np.asarray(atoms.get_positions(), dtype=float)
        response = JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse(
            atoms.get_chemical_symbols(),
            positions,
            self.coulomb_radii_angstrom,
            dielectric=FC_ASWIG_STATIC_DIELECTRIC,
            nuclear_charges=np.asarray(atoms.numbers, dtype=float),
            minimum_relative_eigengap=(
                JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP
            ),
            lebedev_order=FC_ASWIG_LEBEDEV_ORDER,
        )
        cds = JGP94BodyFrameFixedTopologyAqueousSMDCDS(
            atoms.get_chemical_symbols(),
            response.frame,
            np.asarray(atoms.numbers, dtype=float),
            lebedev_order=FC_ASWIG_LEBEDEV_ORDER,
        )
        bundle = _FCAqueousGeometryBundle(
            atomic_numbers=np.asarray(atoms.numbers, dtype=int).copy(),
            positions_angstrom=np.array(positions, dtype=float, copy=True),
            response=response,
            cds=cds,
        )
        self._bundle = bundle
        return bundle

    def _build_reaction_field(self, atoms):
        bundle = self._geometry_bundle(atoms)
        return bundle.response.reaction_field_linear_map(
            np.asarray(atoms.get_positions(), dtype=float)
        )

    def _evaluate_cds(self, atoms) -> FixedTopologyAqueousSMDCDSResult:
        return self._geometry_bundle(atoms).cds.result()

    @property
    def _provider_cache_signature(self) -> tuple[object, ...]:
        return (
            self.profile,
            self.solvent,
            route2_smd_profile_spec(self.profile).mace_geometry_frame_policy,
            FC_ASWIG_STATIC_DIELECTRIC,
            FC_ASWIG_LEBEDEV_ORDER,
            JGP94_DEFAULT_MINIMUM_RELATIVE_EIGENGAP,
        )

    def _solve_coupled_state(
        self,
        atoms,
        electronic_model: Route2ElectronicModel,
        gas_state,
        *,
        initial_density_coefficients: np.ndarray | None = None,
        initial_density_label: str | None = None,
    ) -> Route2CoupledState:
        return self._engine.solve_coupled_state(
            atoms,
            electronic_model,
            gas_state,
            provider_cache_signature=self._provider_cache_signature,
            initial_density_coefficients=initial_density_coefficients,
            initial_density_label=initial_density_label,
            electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        )

    def _coupled_state(
        self,
        atoms,
        electronic_model: Route2ElectronicModel,
        gas_state,
    ) -> Route2CoupledState:
        cached = self._cached_state
        if cached is not None and cached.matches(
            electronic_model,
            atoms,
            provider_cache_signature=self._provider_cache_signature,
        ):
            return cached
        state = self._solve_coupled_state(atoms, electronic_model, gas_state)
        self._cached_state = state
        return state

    def _multi_start_root_agreement(
        self,
        atoms,
        electronic_model: Route2ElectronicModel,
        gas_state,
        primary: Route2CoupledState,
    ) -> Route2MultiStartRootAgreement:
        gas_density = np.asarray(gas_state.density_coefficients, dtype=float)
        starts = {
            "gas": (primary, None, None),
            "zero": (
                None,
                np.zeros_like(gas_density),
                "zero-density",
            ),
            "negated-gas": (
                None,
                -np.array(gas_density, copy=True),
                "negated-gas-density",
            ),
        }
        states: dict[str, Route2CoupledState] = {}
        energies_ev: dict[str, float] = {}
        for key, (existing, seed, label) in starts.items():
            state = (
                existing
                if existing is not None
                else self._solve_coupled_state(
                    atoms,
                    electronic_model,
                    gas_state,
                    initial_density_coefficients=seed,
                    initial_density_label=label,
                )
            )
            states[key] = state
            energies_ev[key] = (
                self._engine.energy_components(
                    gas_state,
                    state,
                    electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
                )["delta_g_solv"]
                * Hartree
            )
        return evaluate_route2_multi_start_root_agreement(
            states,
            declared_scalar_energies_ev=energies_ev,
        )

    def _evaluate(
        self,
        atoms,
        *,
        calculator=None,
        need_forces: bool,
    ) -> SolvationResult:
        self._validate_atoms(atoms)
        electronic_model = self._validate_electronic_model(
            calculator,
            need_forces=need_forces,
        )
        self._validate_public_force_calculator(calculator)
        gas_state = self._engine.gas_state(
            electronic_model,
            atoms,
            need_forces=need_forces,
        )
        self._engine.validate_density(
            gas_state.density_coefficients,
            len(atoms),
            name="Gas electronic source",
        )
        coupled = self._coupled_state(atoms, electronic_model, gas_state)
        components = self._engine.energy_components(
            gas_state,
            coupled,
            electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        )
        forces = None
        derivative: dict[str, Any] | None = None
        root_agreement: Route2MultiStartRootAgreement | None = None
        if need_forces:
            root_agreement = self._multi_start_root_agreement(
                atoms,
                electronic_model,
                gas_state,
                coupled,
            )
            forces, derivative = self._engine.solvent_correction_force(
                atoms,
                electronic_model,
                gas_state,
                coupled,
                force_admission_continuum=(
                    JGP94_FIXED_TOPOLOGY_ASWIG_CPCM_AQUEOUS_SMD_SMOOTHNESS_CONTRACT
                ),
                multi_start_root_agreement=root_agreement.agreed,
                force_admission_pes_validation=(
                    self._force_pes_validation_contract()
                ),
                electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
            )
        provenance = {
            **self.provenance,
            "converged": True,
            "iterations": len(coupled.history),
            "scf_convergence": dict(coupled.scf_convergence),
            "continuum_provider": dict(coupled.reaction_field.runtime_provenance),
            # The result is deliberately just a scalar/gradient record.  The
            # numerical-method provenance belongs to the retained CDS object
            # in the shared geometry bundle.
            "cds_provider": dict(
                self._geometry_bundle(atoms).cds.runtime_provenance
            ),
            "calculator_profile": getattr(calculator, "route2_smd_profile", None),
            "electronic_model": electronic_model.descriptor.as_provenance(),
            "mace_geometry_frame": dict(
                getattr(
                    calculator,
                    "route2_mace_geometry_frame_provenance",
                    {},
                )
            ),
        }
        if root_agreement is not None:
            provenance["multi_start_root_agreement"] = root_agreement.as_dict()
        if derivative is not None:
            provenance["force_admission"] = derivative["force_admission"]
        return SolvationResult(
            energy_hartree=components["delta_g_solv"],
            forces_hartree_per_angstrom=forces,
            components_hartree=components,
            provenance=provenance,
        )

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if need_forces and not self._public_force_profile:
            raise NotImplementedError(FC_ASWIG_DERIVATIVE_EVIDENCE_ONLY_ERROR)
        result = self._evaluate(
            atoms,
            calculator=calculator,
            need_forces=need_forces,
        )
        if need_forces:
            admission = result.provenance.get("force_admission", {})
            if admission.get("release_admitted") is not True:
                raise RuntimeError(
                    "The Route-2 force-v3 per-geometry admission certificate "
                    "did not pass; refusing to return a solvent force."
                )
        return result

    def evaluate_single_point_derivative_evidence(
        self,
        atoms,
        *,
        calculator=None,
    ) -> SinglePointDerivativeEvidence:
        """Return a labelled non-public derivative calculation for one geometry."""

        result = self._evaluate(atoms, calculator=calculator, need_forces=True)
        if result.forces_hartree_per_angstrom is None:
            raise RuntimeError("Fixed-topology derivative evidence omitted forces.")
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


__all__ = [
    "FC_ASWIG_ADJOINT_ABSOLUTE_TOLERANCE",
    "FC_ASWIG_ADJOINT_MAX_ITERATIONS",
    "FC_ASWIG_ADJOINT_RELATIVE_TOLERANCE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE",
    "FC_ASWIG_DERIVATIVE_EVIDENCE_ONLY_ERROR",
    "FC_ASWIG_LEBEDEV_ORDER",
    "FC_ASWIG_STATIC_DIELECTRIC",
    "FixedTopologyASWIGAqueousSMDImplicitSolvation",
]
