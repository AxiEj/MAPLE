"""Versioned Route-2 SMD profile registry.

This lightweight module is the shared contract between input validation,
electronic-model dispatch, and continuum providers.  A public profile selects
one complete, reproducible combination of electronic-model family, source
space, field evaluator, continuum backend, cavity, energy ledger, and
nonpolar functional; users cannot compose unvalidated mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, cast

from maple.solvation.api.profiles import (
    MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS,
    MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX40_EVALUATOR_ID,
    MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CPU_V2,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V4,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4,
)

from .route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
    Route2ElectrostaticEnergyLedger,
    validate_route2_electrostatic_energy_ledger,
)
from .route2_solvents import (
    SUPPORTED_ROUTE2_SMD_SOLVENTS,
    SUPPORTED_SMD_SOLVENTS,
    normalize_route2_solvent_name,
)
from .route2_model_contracts import (
    ROUTE2_ATOMIC_L1_SOURCE_SPACE,
    ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    ROUTE2_MACE_POLAR_MODEL_FAMILY,
    ROUTE2_MACE_POLAR_PROFILE_BINDING,
)

CANONICAL_SMD_PROFILE = "smd-iefpcm"
GAFF2_CARBONYL_O_PROFILE = "smd-iefpcm-gaff2-o"
PCMSOLVER_EXACT_GTO_FIELD_PROFILE = (
    "smd-iefpcm-point-l1-exact-gto-v1"
)
PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE = (
    "smd-iefpcm-point-l1-local-jet-atomic-mean-v1"
)
PCMSOLVER_INTRINSIC_CAVITY_PROFILE = (
    "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-v1"
)
PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE = (
    "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-v1"
)
PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE = (
    "macepolar-mlpcm-smdcds-iefpcm-intrinsic-cavity-exact-gto-"
    "pcm-half-coupling-v1"
)
DDPCM_SMD_PROFILE = "smd-ddpcm-l15-n1202-v1"
DDPCM_SMD_DIRECT_PCM_PROFILE = "smd-ddpcm-l15-n1202-pcm-half-coupling-v1"
DDPCM_MULTISOLVENT_SMD_PROFILE = "smd-ddpcm-l15-n1202-multisolv-v1"
DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE = (
    "smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v1"
)
DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE = (
    "smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v2"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-workflow-v1"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cpu-v2"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cuda-v2"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v3"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cuda-v3"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v4"
)
PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE = (
    "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cuda-v4"
)
PURE_TORCH_FP32_V4_WORKFLOW_POLICY = "pure-torch-fp32-v4-workflows-v1"

_PURE_MACEPOLAR_EXPECTED_DEVICE = {
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE: "cpu",
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE: "cpu",
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE: "cuda",
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE: "cpu",
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE: "cuda",
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE: "cpu",
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE: "cuda",
}
_PURE_MACEPOLAR_CHARGED_PROFILES = frozenset(
    {
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    }
)
DDCOSMO_MULTISOLVENT_SMD_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-v1"
)
DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v1"
)
DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE = (
    "smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v2"
)
FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE = (
    "smd-cpcm-fc-aswig-jgp94-aqueous-pcm-half-coupling-v1"
)
FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE = (
    "smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-v2"
)
FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE = (
    "smd-cpcm-fc-aswig-jgp94-d2-mace-aqueous-pcm-half-coupling-force-v3"
)
DDPCM_GAFF2_CARBONYL_O_PROFILE = "smd-ddpcm-l15-n1202-gaff2-o-v1"
DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE = (
    "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-v1"
)
DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE = (
    "smd-ddpcm-l15-n1202-gaff2-o-mace-kspace40-omp4-v1"
)

MACEPOL_MOLECULAR_REALSPACE_PROFILE = MACE_POLAR_MOLECULAR_REALSPACE_EVALUATOR_ID
MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE = (
    MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX40_EVALUATOR_ID
)
MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES = (
    MACE_POLAR_FORCED_RECIPROCAL_FIXED_BOX_EVALUATOR_IDS
)


@dataclass(frozen=True)
class Route2SMDProfileSpec:
    """One frozen audited combination, not an open run-time plug-in schema."""

    name: str
    provider: Literal["pcmsolver", "pyddx", "fc-aswig", "torch"]
    cavity: Literal[
        "canonical-smd",
        "gaff2-carbonyl-o",
        "fixed-topology-smd",
    ]
    mace_long_range_evaluator: str
    electrostatics_model: Literal["iefpcm", "ddpcm", "ddcosmo", "cpcm"]
    solute_source: Literal["point-multipole-l1"]
    reaction_field_projector: Literal[
        "local-jet",
        "exact-gto-v1",
    ]
    model_field_gauge: Literal[
        "continuum-zero-at-infinity",
        "atomic-center-mean-zero-v1",
    ]
    nonpolar_model: Literal[
        "native-water-smd-cds",
        "pyscf-smd-cds",
        "fixed-topology-aqueous-smd-cds",
        "torch-legacy-smd-cds",
    ]
    dielectric_policy: Literal[
        "pcmsolver-water-keyword",
        "explicit-smd-water-78.355-v1",
        "legacy-water-78.39",
        "pyscf-smd-2.13.1",
    ]
    coulomb_radii_policy: Literal[
        "smd-water-reference-smd18-v1",
        "pyscf-smd-2.13.1",
    ]
    supported_solvents: frozenset[str]
    electronic_model_family: str = ROUTE2_MACE_POLAR_MODEL_FAMILY
    electronic_source_space: str = ROUTE2_ATOMIC_L1_SOURCE_SPACE
    electronic_profile_binding: str = ROUTE2_MACE_POLAR_PROFILE_BINDING
    electronic_energy_semantics: str = (
        ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY
    )
    mace_geometry_frame_policy: Literal[
        "laboratory-v1",
        "jgp94-d2-canonical-v1",
    ] = "laboratory-v1"
    electrostatic_energy_ledger: Route2ElectrostaticEnergyLedger = (
        LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
    )
    strict_original_smd_equivalence: bool = False
    ddpcm_n_proc: int = 1
    default_eligible: bool = False
    force_release_eligible: bool = False
    execution_route: Literal[
        "legacy-additive-correction",
        "pure-frozen-total-pes",
        "pure-torch-analytic-total-pes",
    ] = "legacy-additive-correction"
    scalar_contract_id: str | None = None
    allowed_response_modes: tuple[Literal["frozen", "scf"], ...] = (
        "frozen",
        "scf",
    )
    experimental_task_allowlist: tuple[str, ...] = ("sp",)
    workflow_policy_id: str | None = None
    pcmsolver_cavity_generation: Literal[
        "legacy-builtin-solvent-probe-v1",
        "intrinsic-probe0-noaddsph-v1",
    ] | None = None

    @property
    def uses_gaff2_carbonyl_oxygen(self) -> bool:
        return self.cavity == "gaff2-carbonyl-o"

    @property
    def model_field_evaluator(self) -> str:
        """Model-neutral alias consumed by the electronic adapter boundary."""

        return self.mace_long_range_evaluator

    def __post_init__(self) -> None:
        validate_route2_electrostatic_energy_ledger(
            self.electrostatic_energy_ledger
        )
        if not str(self.electronic_model_family).strip():
            raise ValueError("Route-2 electronic-model family must be non-empty.")
        if not str(self.electronic_source_space).strip():
            raise ValueError("Route-2 electronic source space must be non-empty.")
        if not str(self.electronic_profile_binding).strip():
            raise ValueError("Route-2 electronic profile binding must be non-empty.")
        if not str(self.model_field_evaluator).strip():
            raise ValueError("Route-2 model-field evaluator must be non-empty.")
        if not str(self.electronic_energy_semantics).strip():
            raise ValueError("Route-2 electronic energy semantics must be non-empty.")
        if self.execution_route not in {
            "legacy-additive-correction",
            "pure-frozen-total-pes",
            "pure-torch-analytic-total-pes",
        }:
            raise ValueError("Unsupported Route-2 execution route.")
        if not self.allowed_response_modes or any(
            mode not in {"frozen", "scf"} for mode in self.allowed_response_modes
        ):
            raise ValueError("Route-2 allowed response modes are invalid.")
        if not self.experimental_task_allowlist or any(
            task not in {"sp", "opt", "freq", "ts", "scan", "irc", "md"}
            for task in self.experimental_task_allowlist
        ):
            raise ValueError("Route-2 experimental task allowlist is invalid.")
        if self.execution_route == "pure-frozen-total-pes":
            expected_scalar = {
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CPU_V2,
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2,
            }.get(self.name)
            if (
                self.provider != "pyddx"
                or self.electrostatics_model != "ddpcm"
                or self.nonpolar_model != "pyscf-smd-cds"
                or expected_scalar is None
                or self.scalar_contract_id != expected_scalar
                or self.allowed_response_modes != ("frozen",)
            ):
                raise ValueError(
                    "The pure frozen total-PES route requires a registered "
                    "device-specific point-l1 ddPCM/PySCF-SMD scalar and "
                    "response=frozen."
                )
        if self.execution_route == "pure-torch-analytic-total-pes":
            expected_scalar = {
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V4,
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE:
                    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4,
            }.get(self.name)
            if (
                self.provider != "torch"
                or self.electrostatics_model != "ddpcm"
                or self.nonpolar_model != "torch-legacy-smd-cds"
                or expected_scalar is None
                or self.scalar_contract_id != expected_scalar
                or self.allowed_response_modes != ("frozen",)
            ):
                raise ValueError(
                    "The pure Torch analytic route requires its registered "
                    "device-specific point-l1 ddPCM/legacy-SMD scalar and "
                    "response=frozen."
                )
        if self.workflow_policy_id is not None:
            if self.workflow_policy_id != PURE_TORCH_FP32_V4_WORKFLOW_POLICY:
                raise ValueError("Unsupported Route-2 workflow policy.")
            if self.name not in {
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
                PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
            }:
                raise ValueError("The FP32 V4 workflow policy requires a V4 profile.")
        if self.mace_geometry_frame_policy not in {
            "laboratory-v1",
            "jgp94-d2-canonical-v1",
        }:
            raise ValueError("Unsupported Route-2 MACE geometry-frame policy.")
        if self.force_release_eligible and (
            self.provider != "fc-aswig"
            or self.cavity != "fixed-topology-smd"
            or self.electrostatics_model != "cpcm"
            or self.electronic_model_family != ROUTE2_MACE_POLAR_MODEL_FAMILY
            or self.electronic_profile_binding
            != ROUTE2_MACE_POLAR_PROFILE_BINDING
            or self.reaction_field_projector != "local-jet"
            or self.mace_geometry_frame_policy != "jgp94-d2-canonical-v1"
            or self.electrostatic_energy_ledger != PCM_HALF_COUPLING_ONLY_V1
            or self.supported_solvents != _WATER_ONLY
        ):
            raise ValueError(
                "A Route-2 public force profile must be the bounded "
                "water-only JGP94-D2 fixed-topology direct-CPCM profile."
            )

    def supports_solvent(self, solvent: str) -> bool:
        try:
            canonical = normalize_route2_solvent_name(solvent)
        except ValueError:
            return False
        return canonical in self.supported_solvents

    @property
    def uses_intrinsic_pcmsolver_cavity(self) -> bool:
        return (
            self.pcmsolver_cavity_generation
            == "intrinsic-probe0-noaddsph-v1"
        )


@dataclass(frozen=True, slots=True)
class Route2WorkflowCapability:
    """One task admission row; it does not implement the task itself."""

    execution_class: Literal["bounded_execution", "candidate_only", "closed"]
    image_policy: Literal["single", "multi"]
    outcome_policy: Literal[
        "task_success", "path_success", "candidate_validation", "fixed_steps",
        "numerical_stationary", "numerical_irc_auto", "exploratory_irc",
        "single_point_frames", "frequency_quantities",
    ]
    constraints_allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FrequencyMethodRequest:
    """One raw frequency method and its profile-specific execution identity."""

    raw_method: str
    canonical_method: str
    deprecated_alias: str | None


def normalize_frequency_method_request(
    method: object | None, *, pure_torch_v4: bool
) -> FrequencyMethodRequest:
    """Normalize the V4 ``nonmw`` alias without widening other profiles."""

    raw = str(method or "mw").strip().lower()
    if raw == "cartesian-curvature" and not pure_torch_v4:
        raise ValueError("method=cartesian-curvature is V4-only.")
    if raw == "nonmw" and pure_torch_v4:
        return FrequencyMethodRequest(raw, "cartesian-curvature", "nonmw")
    return FrequencyMethodRequest(raw, raw, None)


_OPEN_SINGLE = Route2WorkflowCapability(
    "bounded_execution", "single", "task_success", True, "supported"
)
_OPEN_PATH = Route2WorkflowCapability(
    "bounded_execution", "multi", "path_success", False, "supported"
)
_NUMERICAL_STATIONARY_SINGLE = Route2WorkflowCapability(
    "bounded_execution", "single", "numerical_stationary", False,
    "computed_stationary_point_required",
)
_NUMERICAL_STATIONARY_PATH = Route2WorkflowCapability(
    "bounded_execution", "multi", "numerical_stationary", False,
    "computed_stationary_point_required",
)
_NUMERICAL_IRC_AUTO = Route2WorkflowCapability(
    "bounded_execution", "single", "numerical_irc_auto", False,
    "computed_index_one_seed_and_both_branches_required",
)
_EXPLORATORY_IRC = Route2WorkflowCapability(
    "candidate_only", "single", "exploratory_irc", False,
    "explicit_raw_mode_is_exploratory",
)
_SP_TRAJECTORY = Route2WorkflowCapability(
    "bounded_execution", "multi", "single_point_frames", True,
    "same_identity_trajectory",
)

_PURE_TORCH_FP32_V4_CAPABILITIES: Mapping[
    tuple[str, str, str | None], Route2WorkflowCapability
] = MappingProxyType({
    ("sp", "sp", None): _OPEN_SINGLE,
    ("sp", "trajectory", None): _SP_TRAJECTORY,
    **{("opt", method, None): _OPEN_SINGLE for method in ("lbfgs", "sd", "cg", "sdcg")},
    ("opt", "rfo", None): Route2WorkflowCapability(
        "bounded_execution", "single", "task_success", False,
        "constrained_hessian_projection_unavailable",
    ),
    ("scan", "rigid", None): _OPEN_SINGLE,
    **{("scan", method, None): _OPEN_SINGLE for method in ("lbfgs", "sd", "cg", "sdcg")},
    ("scan", "rfo", None): Route2WorkflowCapability(
        "closed", "single", "task_success", False,
        "constrained_hessian_projection_unavailable",
    ),
    ("freq", "mw", None): Route2WorkflowCapability(
        "bounded_execution", "single", "task_success", False, "supported"
    ),
    ("freq", "cartesian-curvature", None): Route2WorkflowCapability(
        "bounded_execution", "single", "frequency_quantities", False,
        "cartesian_curvature_diagnostic",
    ),
    ("freq", "both", None): Route2WorkflowCapability(
        "bounded_execution", "single", "frequency_quantities", False,
        "physical_and_cartesian_quantities",
    ),
    ("ts", "neb", None): _OPEN_PATH,
    ("ts", "neb", "cineb"): _OPEN_PATH,
    ("ts", "string", None): _OPEN_PATH,
    ("ts", "string", "cistring"): _OPEN_PATH,
    ("ts", "autoneb", None): _OPEN_PATH,
    ("ts", "prfo", None): _NUMERICAL_STATIONARY_SINGLE,
    ("ts", "dimer", None): _NUMERICAL_STATIONARY_SINGLE,
    ("ts", "neb", "nebts"): _NUMERICAL_STATIONARY_PATH,
    ("ts", "string", "stringts"): _NUMERICAL_STATIONARY_PATH,
    **{("irc", method, None): _NUMERICAL_IRC_AUTO for method in ("gs", "hpc", "eulerpc", "lqa")},
    ("md", "nve", None): Route2WorkflowCapability(
        "bounded_execution", "single", "fixed_steps", False, "supported"
    ),
    ("md", "nvt", None): Route2WorkflowCapability(
        "bounded_execution", "single", "fixed_steps", False, "supported"
    ),
    ("md", "npt", None): Route2WorkflowCapability(
        "closed", "single", "fixed_steps", False,
        "periodic_stress_contract_unavailable",
    ),
})


def route2_workflow_capability(
    policy_id: str,
    *,
    task: object,
    method: object | None,
    refine: object | None = None,
) -> Route2WorkflowCapability:
    """Resolve one normalized task row or fail closed for an unknown method."""

    if str(policy_id).strip().lower() != PURE_TORCH_FP32_V4_WORKFLOW_POLICY:
        raise ValueError(f"Unsupported Route-2 workflow policy: {policy_id}.")
    normalized_task = str(task).strip().lower()
    normalized_method = str(method or normalized_task).strip().lower()
    normalized_refine = None if refine in (None, "") else str(refine).strip().lower()
    try:
        return _PURE_TORCH_FP32_V4_CAPABILITIES[
            (normalized_task, normalized_method, normalized_refine)
        ]
    except KeyError as exc:
        raise ValueError(
            f"Workflow policy={policy_id} has no capability for "
            f"task={normalized_task}, method={normalized_method}, "
            f"refine={normalized_refine}."
        ) from exc


def route2_workflow_capability_for_params(
    profile: Route2SMDProfileSpec, task: str, params: Mapping[str, object]
) -> Route2WorkflowCapability | None:
    if profile.workflow_policy_id is None:
        return None
    if task == "sp":
        mode = str(params.get("mode", "single")).strip().lower()
        if mode not in {"single", "trajectory"}:
            raise ValueError(
                "V4 SP mode must be 'single' or 'trajectory'; heterogeneous "
                "collections remain closed."
            )
        method = "trajectory" if mode == "trajectory" else "sp"
    elif task == "md":
        method = params.get("ensemble", "nve")
    elif task == "scan" and str(params.get("mode", "relaxed")).lower() == "rigid":
        method = "rigid"
    elif task == "freq":
        method = normalize_frequency_method_request(
            params.get("method"), pure_torch_v4=True
        ).canonical_method
    else:
        method = params.get("method") or {
            "opt": "lbfgs", "freq": "mw", "irc": "gs"
        }.get(task, task)
    capability = route2_workflow_capability(
        profile.workflow_policy_id,
        task=task,
        method=method,
        refine=params.get("refine"),
    )
    if task == "irc" and params.get("mode_index") is not None:
        return _EXPLORATORY_IRC
    return capability

_WATER_ONLY = frozenset({"water"})
_TORCH_V3_SOLVENTS = frozenset({"water", "hexane"})
_TORCH_V4_SOLVENTS = SUPPORTED_SMD_SOLVENTS


_PROFILE_SPECS = {
    CANONICAL_SMD_PROFILE: Route2SMDProfileSpec(
        name=CANONICAL_SMD_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    GAFF2_CARBONYL_O_PROFILE: Route2SMDProfileSpec(
        name=GAFF2_CARBONYL_O_PROFILE,
        provider="pcmsolver",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    PCMSOLVER_EXACT_GTO_FIELD_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_EXACT_GTO_FIELD_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="exact-gto-v1",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="pcmsolver-water-keyword",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="legacy-builtin-solvent-probe-v1",
    ),
    PCMSOLVER_INTRINSIC_CAVITY_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_INTRINSIC_CAVITY_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="explicit-smd-water-78.355-v1",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="intrinsic-probe0-noaddsph-v1",
    ),
    PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="exact-gto-v1",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="explicit-smd-water-78.355-v1",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        pcmsolver_cavity_generation="intrinsic-probe0-noaddsph-v1",
    ),
    PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE,
        provider="pcmsolver",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="iefpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="exact-gto-v1",
        model_field_gauge="atomic-center-mean-zero-v1",
        nonpolar_model="native-water-smd-cds",
        dielectric_policy="explicit-smd-water-78.355-v1",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        pcmsolver_cavity_generation="intrinsic-probe0-noaddsph-v1",
    ),
    DDPCM_SMD_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_SMD_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
    ),
    DDPCM_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_SMD_DIRECT_PCM_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    DDPCM_MULTISOLVENT_SMD_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_MULTISOLVENT_SMD_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
    ),
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    # v2 is a numerical-contract revision of the paired direct-PCM study.
    # The physical equation, SMD descriptors, source, and ledger are exactly
    # the v1 values.  It differs only by allowing both paired continuum
    # equations to use the pre-registered, replay-certified finite-resolution
    # energy gate.  Keeping a new profile name prevents v1 evidence from being
    # reinterpreted under a later acceptance policy.
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        execution_route="pure-frozen-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq", "ts"),
        electronic_profile_binding=(
            "official-mace-polar-1-m/float64/cpu/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "zero-field-vacuum-energy-plus-frozen-source-ddpcm-plus-"
            "pyscf-smd-cds-v1"
        ),
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        execution_route="pure-frozen-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CPU_V2
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq", "ts", "scan", "irc"),
        electronic_profile_binding=(
            "official-mace-polar-1-m/float64/cpu/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "zero-field-vacuum-energy-plus-frozen-source-ddpcm-plus-"
            "pyscf-smd-cds-v1"
        ),
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        execution_route="pure-frozen-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq", "ts", "scan", "irc"),
        electronic_profile_binding=(
            "official-mace-polar-1-m/float64/cuda/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "zero-field-vacuum-energy-plus-frozen-source-ddpcm-plus-"
            "pyscf-smd-cds-v1"
        ),
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE,
        provider="torch",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="torch-legacy-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=_TORCH_V3_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        execution_route="pure-torch-analytic-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq"),
        electronic_profile_binding=(
            "official-mace-polar-1-m/float64/cpu/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "single-graph-torch-float64-zero-field-vacuum-plus-frozen-source-"
            "ddpcm-plus-legacy-smd-cds-v3"
        ),
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE,
        provider="torch",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="torch-legacy-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=_TORCH_V3_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        execution_route="pure-torch-analytic-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq"),
        electronic_profile_binding=(
            "official-mace-polar-1-m/float64/cuda/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "single-graph-torch-float64-zero-field-vacuum-plus-frozen-source-"
            "ddpcm-plus-legacy-smd-cds-v3"
        ),
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
        provider="torch",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="torch-legacy-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=_TORCH_V4_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        default_eligible=True,
        execution_route="pure-torch-analytic-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V4
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq", "ts", "scan", "irc", "md"),
        workflow_policy_id=PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
        electronic_profile_binding=(
            "official-mace-polar-1-m/float32/cpu/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "single-graph-torch-float32-zero-field-vacuum-plus-frozen-source-"
            "ddpcm-l5-n194-plus-legacy-smd-cds-v4"
        ),
    ),
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE: Route2SMDProfileSpec(
        name=PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
        provider="torch",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="torch-legacy-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=_TORCH_V4_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        default_eligible=True,
        execution_route="pure-torch-analytic-total-pes",
        scalar_contract_id=(
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4
        ),
        allowed_response_modes=("frozen",),
        experimental_task_allowlist=("sp", "opt", "freq", "ts", "scan", "irc", "md"),
        workflow_policy_id=PURE_TORCH_FP32_V4_WORKFLOW_POLICY,
        electronic_profile_binding=(
            "official-mace-polar-1-m/float32/cuda/zero-field-radial-gto-source/v1"
        ),
        electronic_energy_semantics=(
            "single-graph-torch-float32-zero-field-vacuum-plus-frozen-source-"
            "ddpcm-l5-n194-plus-legacy-smd-cds-v4"
        ),
    ),
    DDCOSMO_MULTISOLVENT_SMD_PROFILE: Route2SMDProfileSpec(
        name=DDCOSMO_MULTISOLVENT_SMD_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddcosmo",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
    ),
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddcosmo",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE: Route2SMDProfileSpec(
        name=DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
        provider="pyddx",
        cavity="canonical-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddcosmo",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="pyscf-smd-2.13.1",
        coulomb_radii_policy="pyscf-smd-2.13.1",
        supported_solvents=SUPPORTED_ROUTE2_SMD_SOLVENTS,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE: Route2SMDProfileSpec(
        name=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE,
        provider="fc-aswig",
        cavity="fixed-topology-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="cpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="fixed-topology-aqueous-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    # v2 is deliberately a new evaluation operator rather than a silent
    # mutation of v1.  It applies a four-branch proper-D2 Reynolds average in
    # a nondegenerate JGP94 molecular frame *at the MACE boundary*, removing
    # the fixed laboratory-axis finite-difference artefact of the official
    # real-space MACE-POLAR evaluator.  Weights, continuum parameters, SMD
    # descriptors, cavity, and the direct PCM ledger are unchanged.
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE: Route2SMDProfileSpec(
        name=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE,
        provider="fc-aswig",
        cavity="fixed-topology-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="cpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="fixed-topology-aqueous-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        mace_geometry_frame_policy="jgp94-d2-canonical-v1",
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    ),
    # v3 changes only the public capability boundary.  It reuses the frozen
    # v2 scalar/operator and admits forces only through the per-geometry
    # force certificate; v1/v2 remain energy-only identities.
    FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE: Route2SMDProfileSpec(
        name=FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE,
        provider="fc-aswig",
        cavity="fixed-topology-smd",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="cpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="fixed-topology-aqueous-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        mace_geometry_frame_policy="jgp94-d2-canonical-v1",
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        force_release_eligible=True,
    ),
    DDPCM_GAFF2_CARBONYL_O_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_GAFF2_CARBONYL_O_PROFILE,
        provider="pyddx",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=MACEPOL_MOLECULAR_REALSPACE_PROFILE,
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
    ),
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE,
        provider="pyddx",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
        ),
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
    ),
    DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE: Route2SMDProfileSpec(
        name=DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE,
        provider="pyddx",
        cavity="gaff2-carbonyl-o",
        mace_long_range_evaluator=(
            MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE
        ),
        electrostatics_model="ddpcm",
        solute_source="point-multipole-l1",
        reaction_field_projector="local-jet",
        model_field_gauge="continuum-zero-at-infinity",
        nonpolar_model="pyscf-smd-cds",
        dielectric_policy="legacy-water-78.39",
        coulomb_radii_policy="smd-water-reference-smd18-v1",
        supported_solvents=_WATER_ONLY,
        ddpcm_n_proc=4,
    ),
}

SUPPORTED_PCMSOLVER_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pcmsolver"
)
SUPPORTED_PYDDX_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pyddx"
)
SUPPORTED_FC_ASWIG_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "fc-aswig"
)
SUPPORTED_TORCH_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "torch"
)
SUPPORTED_DDPCM_SMD_PROFILES = frozenset(
    name
    for name, spec in _PROFILE_SPECS.items()
    if spec.provider == "pyddx" and spec.electrostatics_model == "ddpcm"
)
SUPPORTED_ROUTE2_SMD_PROFILES = frozenset(_PROFILE_SPECS)
Route2SMDResponseMode = Literal["frozen", "scf"]
SUPPORTED_ROUTE2_SMD_RESPONSE_MODES = frozenset({"frozen", "scf"})


def route2_smd_profiles_for_provider(provider: str) -> frozenset[str]:
    normalized = str(provider).strip().lower()
    if normalized == "pcmsolver":
        return SUPPORTED_PCMSOLVER_SMD_PROFILES
    if normalized == "pyddx":
        return SUPPORTED_PYDDX_SMD_PROFILES
    if normalized == "fc-aswig":
        return SUPPORTED_FC_ASWIG_SMD_PROFILES
    if normalized == "torch":
        return SUPPORTED_TORCH_SMD_PROFILES
    raise ValueError(f"Unsupported Route 2 SMD provider: {provider}.")


def resolve_route2_smd_profile(
    provider: str,
    profile: object | None,
    *,
    device: object,
) -> str:
    """Resolve only the pure-Torch default; preserve explicit profile identity."""

    normalized_provider = str(provider).strip().lower()
    if profile is not None and str(profile).strip():
        return str(profile).strip().lower()
    if normalized_provider != "torch":
        raise ValueError("Route 2 SMD requires an explicit versioned profile.")
    normalized_device = str(device).strip().lower()
    if normalized_device == "cpu":
        return PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE
    if normalized_device == "cuda" or (
        normalized_device.startswith("cuda:")
        and normalized_device[5:].isdigit()
    ):
        return PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE
    raise ValueError(
        "The default pure-Torch Route 2 profile requires device=cpu, cuda, or cuda:N."
    )


def route2_smd_profile_spec(profile: str) -> Route2SMDProfileSpec:
    normalized = str(profile).strip().lower()
    try:
        return _PROFILE_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Route 2 SMD profile: {profile}.") from exc


def pure_macepolar_expected_device(
    profile: str | Route2SMDProfileSpec,
) -> str:
    """Return the immutable device family for a canonical pure profile."""

    name = profile.name if isinstance(profile, Route2SMDProfileSpec) else str(profile)
    try:
        return _PURE_MACEPOLAR_EXPECTED_DEVICE[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Profile {name!r} is not a pure MACE-POLAR profile.") from exc


def pure_macepolar_allows_charged(
    profile: str | Route2SMDProfileSpec,
) -> bool:
    """Return whether one exact registered pure profile admits charged solutes."""

    name = profile.name if isinstance(profile, Route2SMDProfileSpec) else str(profile)
    return name.strip().lower() in _PURE_MACEPOLAR_CHARGED_PROFILES


def validate_route2_smd_profile(
    provider: str,
    profile: str,
    *,
    solvent: str | None = None,
) -> Route2SMDProfileSpec:
    normalized_provider = str(provider).strip().lower()
    spec = route2_smd_profile_spec(profile)
    if spec.provider != normalized_provider:
        raise ValueError(
            f"Route 2 provider={normalized_provider} does not support "
            f"profile={spec.name}."
        )
    if solvent is not None and not spec.supports_solvent(solvent):
        raise ValueError(
            f"Route 2 profile={spec.name} does not support solvent={solvent}."
        )
    return spec


def validate_route2_smd_response_mode(
    profile_spec: Route2SMDProfileSpec,
    response: str,
) -> Route2SMDResponseMode:
    """Validate the response strategy bound to one scientific profile.

    This is the single response-admission boundary shared by input parsing,
    calculator construction, and continuum providers.  Keeping the rule next
    to the profile registry prevents those layers from drifting apart.
    """

    if not isinstance(profile_spec, Route2SMDProfileSpec):
        raise TypeError("Route-2 response validation requires a profile spec.")
    normalized = str(response).strip().lower()
    if normalized not in SUPPORTED_ROUTE2_SMD_RESPONSE_MODES:
        raise ValueError("SMD response must be frozen or scf.")
    if normalized not in profile_spec.allowed_response_modes:
        allowed = ", ".join(profile_spec.allowed_response_modes)
        raise ValueError(
            f"Route 2 profile={profile_spec.name} requires response in: {allowed}."
        )
    if (
        profile_spec.provider == "pyddx"
        and normalized == "frozen"
        and profile_spec.electrostatic_energy_ledger
        != PCM_HALF_COUPLING_ONLY_V1
    ):
        raise ValueError(
            "Route 2 provider=pyddx response=frozen requires a direct PCM "
            "half-coupling profile."
        )
    if profile_spec.provider == "fc-aswig" and normalized != "scf":
        raise ValueError("Fixed-topology Route 2 requires response='scf'.")
    if (
        profile_spec.provider == "pcmsolver"
        and normalized != "scf"
        and (
            profile_spec.reaction_field_projector != "local-jet"
            or profile_spec.model_field_gauge
            != "continuum-zero-at-infinity"
        )
    ):
        raise ValueError(
            "A non-default reaction-field projector or model-field gauge "
            "requires response=scf; a frozen response would configure but "
            "never apply that model drive."
        )
    return cast(Route2SMDResponseMode, normalized)


__all__ = [
    "CANONICAL_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE",
    "DDCOSMO_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_OMP4_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_MACE_KSPACE40_PROFILE",
    "DDPCM_GAFF2_CARBONYL_O_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_PROFILE",
    "DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE",
    "DDPCM_SMD_DIRECT_PCM_PROFILE",
    "DDPCM_SMD_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_CANONICAL_MACE_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_FORCE_PROFILE",
    "FC_ASWIG_AQUEOUS_SMD_DIRECT_PCM_PROFILE",
    "GAFF2_CARBONYL_O_PROFILE",
    "MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE",
    "MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES",
    "MACEPOL_MOLECULAR_REALSPACE_PROFILE",
    "PCMSOLVER_CENTERED_LOCAL_JET_FIELD_PROFILE",
    "PCMSOLVER_EXACT_GTO_FIELD_PROFILE",
    "PCMSOLVER_INTRINSIC_CAVITY_PROFILE",
    "PCMSOLVER_INTRINSIC_EXACT_GTO_DIRECT_PCM_PROFILE",
    "PCMSOLVER_INTRINSIC_EXACT_GTO_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_WORKFLOW_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CPU_V2_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_NONMD_CUDA_V2_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V3_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE",
    "PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE",
    "PURE_TORCH_FP32_V4_WORKFLOW_POLICY",
    "FrequencyMethodRequest",
    "normalize_frequency_method_request",
    "SUPPORTED_DDPCM_SMD_PROFILES",
    "SUPPORTED_FC_ASWIG_SMD_PROFILES",
    "SUPPORTED_PCMSOLVER_SMD_PROFILES",
    "SUPPORTED_PYDDX_SMD_PROFILES",
    "SUPPORTED_ROUTE2_SMD_PROFILES",
    "SUPPORTED_TORCH_SMD_PROFILES",
    "SUPPORTED_ROUTE2_SMD_RESPONSE_MODES",
    "Route2SMDProfileSpec",
    "Route2SMDResponseMode",
    "Route2WorkflowCapability",
    "route2_workflow_capability",
    "route2_workflow_capability_for_params",
    "route2_smd_profile_spec",
    "resolve_route2_smd_profile",
    "pure_macepolar_allows_charged",
    "pure_macepolar_expected_device",
    "route2_smd_profiles_for_provider",
    "validate_route2_smd_profile",
    "validate_route2_smd_response_mode",
]
