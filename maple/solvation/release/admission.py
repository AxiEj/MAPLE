"""Purpose-specific, fail-closed Route-2 E/F revalidation contracts.

This module validates JSON-decoded identities only.  It performs no filesystem
I/O, registry mutation, or capability activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import numpy as np
from pathlib import PurePosixPath
import re
from typing import Mapping, Protocol, TypeAlias, TypeVar
from uuid import UUID

from maple.solvation.api.capabilities import CapabilityStatus

from .evidence import canonical_json_sha256

COMPUTATION_SEAL_V2_SCHEMA = "maple-route2-computation-seal-v2"
REPLICATE_ADMISSION_RECORD_V2_SCHEMA = "maple-route2-replicate-admission-record-v2"
EXECUTION_FAILURE_V2_SCHEMA = "maple-route2-execution-failure-v2"
AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA = "maple-route2-aggregate-admission-inputs-v2"
ADMISSION_OVERLAY_V2_SCHEMA = "maple-route2-admission-overlay-v2"
CLAIM_BOUNDARY_ID = "maple-route2-harmonic-electrostatic-ef-only-v2"
EXPOSURE_AWARE_PROTOCOL_LABEL = "route2-h1-exposure-aware-ef-revalidation-v2"
H1_V2_PROFILE_ID = (
    "route2-profile-experimental-macemdppoint-macepolarinduced-"
    "smoothharmonicgalerkin-local-electrostatic-ef-v2"
)
H1_V2_SCALAR_ID = (
    "route2-experimental-macemdppoint-macepolarinduced-"
    "smoothharmonicgalerkin-local-electrostatic-ef-v2"
)
H1_V2_STATE_ID = (
    "route2-mace-mdp-permanent-macepolar-induced-harmonic-local-field-root-v2"
)

REQUIRED_RUNTIME_GUARDS = (
    "cublas-workspace-config-fixed",
    "cuda-device-required",
    "deterministic-algorithms-required",
    "deterministic-debug-mode-required",
    "float64-required",
    "independent-cold-processes-required",
    "pythonhashseed-explicit",
)
REQUIRED_NON_ADMISSIONS = (
    "chemical_accuracy",
    "complete_solvation",
    "named_solvent_thermodynamics",
    "analytic_force",
    "broad_distorted_pes",
    "hessian_hvp_workflows",
    "opt_freq_ts_irc",
    "md_nve",
    "tier_v",
)
REQUIRED_DOMAIN_GUARDS = (
    "local-force-panel-domain",
    "neutral-singlet",
    "per-call-root-replay-charge-topology-richardson",
    "supported-elements-intersection",
)
REQUIRED_ASSET_KEYS = (
    "benzene_mol2",
    "benzene_projection_result",
    "parent_panel",
    "v1_preregistration",
    "verified_pro_audit",
    "verified_pro_prompt",
    "verified_pro_response",
    "verified_pro_mode_verification",
)
REQUIRED_CHECKPOINT_KEYS = (
    "mace_mdp_checkpoint",
    "mace_polar_checkpoint",
)
H1_V2_CONTINUUM_PROFILE_ID = "smooth-weighted-harmonic-galerkin-cpcm-candidate-v1"
H1_V2_CAVITY_PROFILE_ID = "smooth-weighted-overlap-harmonic-cavity-candidate-v1"
H1_V2_RADII_PROVIDER_ID = "smd-water-coulomb-radii-v1"
H1_V2_PERMANENT_SOURCE_KERNEL = (
    "invariant exterior point multipoles projected to complete harmonic coefficients"
)
H1_V2_INDUCED_SOURCE_KERNEL = (
    "normalized sigma=1.5-A Gaussian multipoles projected to complete harmonic "
    "coefficients"
)
H1_V2_RECEIVER_KERNEL = (
    "minus exact transpose of the normalized sigma=1.5/3.0-A Gaussian harmonic "
    "source operator"
)
H1_V2_SOURCE_COEFFICIENT_BASIS_ID = "maple.route2.atomic-l1-source-space.v1"
H1_V2_SOURCE_COEFFICIENT_ORDER = (
    "net_monopole",
    "real_l1_m0",
    "real_l1_m1",
    "real_l1_m_minus1",
)
H1_V2_SOURCE_SPACE_CONTRACT_SHA256 = (
    "e1a5668e2fe296b92885c0d42cb8298dcd40af37cfa907ac99a40f3c638cdfcd"
)
H1_V2_SOURCE_COEFFICIENT_UNITS = (
    "e",
    "e*angstrom",
    "e*angstrom",
    "e*angstrom",
)
H1_V2_RECEIVER_SPACE_ID = "maple.route2.mace-polar-native-radial-field-space.v1"
H1_V2_RECEIVER_SPACE_CONTRACT_SHA256 = (
    "050844a5cbcacfda5ad8b49b70541b5ea73337954631633ae83e63d61f9bbba0"
)
H1_V2_RECEIVER_COMPONENT_ORDER = (
    "potential_sigma_1p5",
    "potential_sigma_3p0",
    "potential_gradient_y_sigma_1p5",
    "potential_gradient_z_sigma_1p5",
    "potential_gradient_x_sigma_1p5",
    "potential_gradient_y_sigma_3p0",
    "potential_gradient_z_sigma_3p0",
    "potential_gradient_x_sigma_3p0",
)
H1_V2_RECEIVER_UNITS = (
    "eV/e",
    "eV/e",
    "eV/(e*angstrom)",
    "eV/(e*angstrom)",
    "eV/(e*angstrom)",
    "eV/(e*angstrom)",
    "eV/(e*angstrom)",
    "eV/(e*angstrom)",
)
def h1_v2_audit_coefficient_sum_contract() -> dict[str, object]:
    """Return the non-operational audit projection contract as a fresh JSON value."""

    return {
        "semantic_role": "algebraic_audit_projection",
        "physical_kernel": None,
        "operator_dispatch": "forbidden",
        "derived_from": ["permanent_point_source4", "induced_gto_source4"],
    }


def verified_pro_schema_amendment_contract() -> dict[str, object]:
    """Return the reviewed H0/archive and rich-v2 protocol amendment."""

    return {
        "amendment_id": (
            "route2-rich-v2-source-space-and-h0-archive-amendment-20260823"
        ),
        "reason": (
            "Historical H0 rich leaves are non-identifiable; verified Pro approved "
            "an archive-only v1 compatibility gate and required machine-checkable "
            "rich-v2 representation contracts before H1 execution."
        ),
        "verified_pro_audit": {
            "relative_path": (
                "docs/route2/evidence/rich-v2-source-space-pro-q3-audit.json"
            ),
            "raw_file_sha256": (
                "024e8ab3617f34f48e35fc1b88e2257ee741fac49ba0b84d9cc62f99f298d436"
            ),
            "audit_sha256": (
                "d9e1023e5c098456107a74cb665cfac893c6314787d197ba9bb994e80015f229"
            ),
            "verdict": "APPROVE",
            "terminal_marker": "MAPLE RICH V2 SOURCE SPACE AUDIT COMPLETE",
        },
        "required_contract_refinements": [
            "bind-source-and-receiver-space-convention-digests",
            "type-audit-coefficient-sum-as-nonoperational-and-forbid-dispatch",
            "kernel-tag-permanent-point-and-induced-gto-branches-separately",
            "retain-polar-final-and-zero-reference-arrays-and-derive-induced",
            "limit-replay-claim-to-endpoint-defined-gates",
            "bind-state-specific-vacuum-energy-and-add-exactly-once",
            "separate-state-content-from-solve-occurrence-identity",
            "use-acyclic-hash-dag-and-recompute-prepared-inputs-from-captured-objects",
        ],
        "historical_h0_archive_policy": {
            "real_h0_v1_bytes_and_digest": (
                "pin-exactly-as-observed-archive-evidence"
            ),
            "historical_rich_preimage": "non-identifiable-do-not-fabricate",
            "synthetic_rich_golden": "adapter-compatibility-only",
            "archive_gate": "independent-of-h1-admission",
            "admission_transfer_to_h1": False,
        },
        "h1_admission_requirement": (
            "two-independent-clean-sealed-replays-plus-mechanical-aggregation-and-"
            "data-only-overlay"
        ),
        "capability_effect_before_h1_overlay": "none",
    }
H1_V2_ENDPOINT_REPLAY_SCOPE = (
    "endpoint-defined-gates-only; solver-trajectory-reconstruction-not-claimed"
)
H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT = (
    "maple.route2.canonical-float64-little-endian-c-array-bytes.v1"
)
H1_V2_LONG_RANGE_EVALUATOR_ID = (
    "graph-longrange-analytic-gaussian-multipole-realspace-v1"
)
H1_V2_EXACT_SCALAR = (
    "E_exp(R)=E_vac^MACEPOLAR(R)-0.5*b(u*)^T*A_harm(R)^-1*b(u*); "
    "b=B_point_harm*c_MDP+B_GTO1p5_harm*(M_POLAR(R,u*)-M_POLAR(R,0)); "
    "u*=-S8_harm(R)^T*A_harm(R)^-1*b(u*)"
)
H1_V2_INCLUDED_COMPONENTS = (
    "macepolar_zero_field_vacuum_energy",
    "hybrid_smooth_harmonic_half_coupling_electrostatic",
)
H1_V2_EXCLUDED_COMPONENTS = (
    "macepolar_field_conditioned_raw_energy_difference",
    "mace_mdp_polarizability_response",
    "nonpolar_smd_cds",
    "thermal_and_standard_state_terms",
    "complete_solvation_free_energy_claim",
    "analytic_coordinate_derivative",
    "strict_common_functional_claim",
)
H1_V2_ROOT_METHOD = "undamped Picard"
H1_V2_SECOND_START = "twice the permanent-source reaction field"
H1_V2_FORCE_DERIVATIVE = (
    "fourth-order central Richardson derivative of the complete re-solved scalar"
)
H1_V2_TOPOLOGY_POLICY = (
    "identical fixed coefficient topology hash at center and every stencil point"
)
H1_V2_DISPLACEMENT_POLICY = (
    "rebuild E,K,point/Gaussian source maps,A,b and both-start root at every point"
)
REPLICATE_GATE_NAMES = (
    "clean_content_addressed_execution",
    "benzene_predecessor_failure_coordinate_passes_h_h2_h4",
    "water_full_cartesian_h_h2_force_is_finite",
    "all_local_richardson_error_estimates_below_2e-4_ev_per_angstrom",
    "independent_h4_directional_derivative_below_5e-4_ev_per_angstrom",
    "translation_energy_and_net_force_gates_pass",
    "rotation_energy_and_force_covariance_gates_pass",
    "closed_loop_work_plus_numerical_error_bound_gate_passes",
    "all_roots_converge_from_both_starts",
    "all_final_residuals_and_total_charge_errors_pass",
)
AGGREGATE_GATE_NAMES = ("two_independent_processes_same_measurement_sha256",)
V1_ADMISSION_GATE_NAMES = (
    "clean_content_addressed_execution",
    "two_independent_processes_same_measurement_sha256",
    *REPLICATE_GATE_NAMES[1:],
)
RICH_MEASUREMENT_SCHEMA_ID = "maple-route2-harmonic-ef-rich-leaves-v2"
LEGACY_MEASUREMENT_SCHEMA_ID = "maple-route2-harmonic-ef-measurement-v2"
MEASUREMENT_SCHEMA_ID = RICH_MEASUREMENT_SCHEMA_ID
STATE_LEAF_V2_SCHEMA = "maple-route2-harmonic-ef-state-leaf-v2"
SOLVE_EVENT_V2_SCHEMA = "maple-route2-harmonic-ef-solve-event-v2"

_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_PROVIDER_CONFIGURATION_KEYS = frozenset(
    {
        "hybrid_configuration",
        "hybrid_provenance",
        "continuum_configuration",
        "continuum_provenance",
        "cavity_configuration",
        "prepared_input_manifest",
        "force_panel_contract",
        "benzene_pes_configuration",
        "water_pes_configuration",
    }
)
_SCIENTIFIC_KEYS = frozenset(
    {
        "continuum",
        "cavity",
        "source_receiver",
        "energy_ledger",
        "root_algorithm",
        "force_stencil",
    }
)
_CONTINUUM_KEYS = frozenset(
    {
        "profile_id",
        "transition_width_angstrom2",
        "surface_lmax",
        "exposure_lmax",
        "exposure_radial_quadrature_order",
        "source_radial_quadrature_order",
        "green_radial_quadrature_order",
    }
)
_CAVITY_KEYS = frozenset({"profile_id", "radii_provider_id"})
_SOURCE_RECEIVER_KEYS = frozenset(
    {
        "permanent_source_kernel",
        "induced_source_kernel",
        "receiver_kernel",
        "long_range_evaluator_id",
        "source_space_id",
        "source_space_contract_sha256",
        "receiver_space_id",
        "receiver_space_contract_sha256",
    }
)
_ENERGY_LEDGER_KEYS = frozenset(
    {"exact_scalar", "included_components", "excluded_components"}
)
_ROOT_ALGORITHM_KEYS = frozenset(
    {
        "method",
        "tolerance_ev",
        "maximum_iterations",
        "second_start",
        "total_charge_tolerance_e",
        "multi_start_field_tolerance_ev",
        "multi_start_energy_tolerance_ev",
        "evidence_scope",
    }
)
_FORCE_STENCIL_KEYS = frozenset(
    {
        "derivative",
        "coarse_step_angstrom",
        "fine_step_angstrom",
        "independent_step_angstrom",
        "maximum_local_error_ev_per_angstrom",
        "topology_policy",
        "displacement_policy",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "python",
        "implementation",
        "platform",
        "machine",
        "cpu_model",
        "packages",
        "numpy",
        "torch",
        "environment",
        "execution_device",
        "execution_dtype",
    }
)
_PACKAGE_KEYS = frozenset(
    {
        "maple",
        "ase",
        "numpy",
        "scipy",
        "torch",
        "mace-torch",
        "graph-longrange",
        "pyscf",
        "pyddx",
    }
)
_NUMPY_KEYS = frozenset({"version", "show_config"})
_TORCH_KEYS = frozenset(
    {
        "version",
        "cuda_version",
        "cudnn_version",
        "cuda_available",
        "devices",
        "default_dtype",
        "threads",
        "interop_threads",
        "driver_version",
        "device_uuids",
        "device_capabilities",
        "device_multiprocessor_counts",
        "deterministic_algorithms_enabled",
        "deterministic_debug_mode",
        "cudnn_benchmark",
        "cudnn_deterministic",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_LAUNCH_BLOCKING",
        "PYTHONHASHSEED",
    }
)
_CAPABILITY_KEYS = frozenset({"E", "F", "H", "V", "M"})
_EF_CAPABILITIES = CapabilityStatus(energy=True, conservative_force=True)

JsonScalar: TypeAlias = None | bool | int | float | str


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    items: tuple["FrozenJson", ...]


@dataclass(frozen=True, slots=True)
class _FrozenObject:
    items: tuple[tuple[str, "FrozenJson"], ...]


FrozenJson: TypeAlias = JsonScalar | _FrozenArray | _FrozenObject
_Contract = TypeVar("_Contract")


class _KeyValidator(Protocol):
    def __call__(self, value: object, *, name: str) -> str: ...


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = frozenset(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}.")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}.")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string.")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    return None if value is None else _text(value, name=name)


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool.")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a finite positive number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be a full lowercase SHA256 digest.")
    return text


def _git_object(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _GIT_OBJECT.fullmatch(text) is None:
        raise ValueError(f"{name} must be a full lowercase 40-hex Git object ID.")
    return text


def _exact_tuple(
    value: object, expected: tuple[str, ...], *, name: str
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    actual = tuple(_text(item, name=f"{name} item") for item in value)
    if actual != expected:
        raise ValueError(f"{name} must equal the frozen {name} contract.")
    return actual


def _exact_set(
    value: object, expected: tuple[str, ...], *, name: str
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    actual = tuple(_text(item, name=f"{name} item") for item in value)
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"{name} must equal the frozen {name} set.")
    return expected


def _guards(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    guards = tuple(_text(item, name=f"{name} item") for item in value)
    if not guards:
        raise ValueError(f"{name} must not be empty.")
    if len(set(guards)) != len(guards):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(sorted(guards))


def _components(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    result = tuple(_text(item, name=f"{name} item") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and duplicate-free.")
    return result


def _repo_relative_path(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if "\\" in text:
        raise ValueError(f"{name} must use canonical POSIX separators.")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a non-traversing repository-relative path.")
    if path.as_posix() != text:
        raise ValueError(f"{name} must be a canonical repository-relative path.")
    return text


def _stable_label(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _STABLE_LABEL.fullmatch(text) is None or text in {".", ".."}:
        raise ValueError(f"{name} must be a lowercase stable non-path label.")
    return text


def _sha_ledger(
    value: object,
    *,
    name: str,
    key_validator: _KeyValidator,
    exact_keys: frozenset[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    ledger = _object(value, name=name)
    if exact_keys is not None:
        _exact_fields(ledger, exact_keys, name=name)
    elif not ledger:
        raise ValueError(f"{name} must not be empty.")
    return tuple(
        sorted(
            (
                key_validator(key, name=f"{name} key"),
                _sha256(digest, name=f"{name}[{key!r}]"),
            )
            for key, digest in ledger.items()
        )
    )


def _freeze_json(value: object, *, name: str) -> FrozenJson:
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number.")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _FrozenArray(
            tuple(
                _freeze_json(item, name=f"{name}[{index}]")
                for index, item in enumerate(value)
            )
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} object keys must be strings.")
        return _FrozenObject(
            tuple(
                (key, _freeze_json(item, name=f"{name}.{key}"))
                for key, item in sorted(value.items())
            )
        )
    raise TypeError(f"{name} must contain JSON-native values only.")


def _freeze_object(value: Mapping[str, object], *, name: str) -> _FrozenObject:
    frozen = _freeze_json(value, name=name)
    if not isinstance(frozen, _FrozenObject):  # pragma: no cover
        raise TypeError(f"{name} must be a JSON object.")
    return frozen


def _thaw_json(value: FrozenJson) -> object:
    if isinstance(value, _FrozenObject):
        return {key: _thaw_json(item) for key, item in value.items}
    if isinstance(value, _FrozenArray):
        return [_thaw_json(item) for item in value.items]
    return value


def _finite_number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{name} must be a finite JSON number, not bool or text.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = " nonnegative" if nonnegative else ""
        raise ValueError(f"{name} must be a finite{qualifier} JSON number.")
    return result


def _finite_matrix(
    value: object, *, rows: int | None, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a nonempty JSON matrix.")
    if rows is not None and len(value) != rows:
        raise ValueError(f"{name} must have exactly {rows} rows.")
    result = tuple(
        (
            tuple(
                _finite_number(item, name=f"{name}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            )
            if isinstance(row, list) and len(row) == columns
            else ()
        )
        for row_index, row in enumerate(value)
    )
    if any(len(row) != columns for row in result):
        raise ValueError(f"{name} must have exactly {columns} columns.")
    return result


def canonical_numeric_array_sha256_v2(
    value: object,
    *,
    semantic_type: str,
    geometry_sha256: str,
    channel_space_contract_sha256: str,
    units: tuple[str, ...],
) -> str:
    """Hash one finite matrix with an explicit, domain-separated byte contract."""

    array = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("canonical numeric arrays must be nonempty finite matrices.")
    if array.shape[1] != len(units):
        raise ValueError("canonical numeric array units do not match its columns.")
    contiguous = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    payload = {
        "encoding_contract": H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT,
        "semantic_type": _text(semantic_type, name="semantic_type"),
        "shape": list(contiguous.shape),
        "dtype": "float64",
        "byte_order": "little",
        "memory_order": "C",
        "units": list(units),
        "atom_order_geometry_sha256": _sha256(
            geometry_sha256, name="atom_order_geometry_sha256"
        ),
        "channel_space_contract_sha256": _sha256(
            channel_space_contract_sha256,
            name="channel_space_contract_sha256",
        ),
        "nan_and_infinity_policy": "forbidden",
        "signed_zero_policy": "preserve-ieee754-sign-bit",
        "data_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }
    return canonical_json_sha256(payload)


def canonical_source_monopole_sum_v2(value: object) -> float:
    """Return the deterministic high-accuracy row-ordered monopole audit sum."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 4:
        raise ValueError("source4 must be a nonempty matrix with four columns.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("source4 must contain only finite values.")
    return float(math.fsum(float(value) for value in matrix[:, 0]))


_START_LEAF_FIELDS = frozenset(
    {
        "initial_state_sha256",
        "converged",
        "iterations",
        "final_native_field_eV",
        "final_residual_eV",
        "final_polarization_energy_eV",
        "final_total_energy_eV",
    }
)
_STATE_LEAF_FIELDS = frozenset(
    {
        "schema_id",
        "state_leaf_sha256",
        "legacy_root_sha256",
        "prepared_pes_configuration_sha256",
        "geometry_sha256",
        "provider_configuration_sha256",
        "topology_id",
        "target_charge_e",
        "audit_coefficient_sum_charge_e",
        "source_coefficient_basis_id",
        "source_space_contract_sha256",
        "source_coefficient_order",
        "receiver_space_id",
        "receiver_space_contract_sha256",
        "receiver_component_order",
        "audit_projection_contract",
        "numeric_array_encoding_contract",
        "endpoint_replay_scope",
        "permanent_point_source4",
        "polar_zero_reference_source4",
        "polar_final_source4",
        "induced_gto_source4",
        "audit_coefficient_sum4",
        "array_content_sha256s",
        "vacuum_energy_eV",
        "polarization_energy_eV",
        "total_energy_eV",
        "cold_start",
        "wide_start",
    }
)


def _start_leaf(value: object, *, atom_count: int, name: str) -> dict[str, object]:
    raw = _object(value, name=name)
    _exact_fields(raw, _START_LEAF_FIELDS, name=name)
    return {
        "initial_state_sha256": _sha256(
            raw["initial_state_sha256"], name=f"{name}.initial_state_sha256"
        ),
        "converged": _strict_bool(raw["converged"], name=f"{name}.converged"),
        "iterations": _positive_int(raw["iterations"], name=f"{name}.iterations"),
        "final_native_field_eV": [
            list(row)
            for row in _finite_matrix(
                raw["final_native_field_eV"],
                rows=atom_count,
                columns=8,
                name=f"{name}.final_native_field_eV",
            )
        ],
        "final_residual_eV": [
            list(row)
            for row in _finite_matrix(
                raw["final_residual_eV"],
                rows=atom_count,
                columns=8,
                name=f"{name}.final_residual_eV",
            )
        ],
        "final_polarization_energy_eV": _finite_number(
            raw["final_polarization_energy_eV"],
            name=f"{name}.final_polarization_energy_eV",
        ),
        "final_total_energy_eV": _finite_number(
            raw["final_total_energy_eV"], name=f"{name}.final_total_energy_eV"
        ),
    }


@dataclass(frozen=True, slots=True, init=False)
class StateLeafV2:
    """Canonical deduplicated rich-v2 state leaf; no gate is trusted here."""

    schema_id: str
    state_leaf_sha256: str
    legacy_root_sha256: str
    prepared_pes_configuration_sha256: str
    geometry_sha256: str
    provider_configuration_sha256: str
    topology_id: str
    target_charge_e: float
    audit_coefficient_sum_charge_e: float
    source_coefficient_basis_id: str
    source_space_contract_sha256: str
    source_coefficient_order: _FrozenArray
    receiver_space_id: str
    receiver_space_contract_sha256: str
    receiver_component_order: _FrozenArray
    audit_projection_contract: _FrozenObject
    numeric_array_encoding_contract: str
    endpoint_replay_scope: str
    permanent_point_source4: _FrozenArray
    polar_zero_reference_source4: _FrozenArray
    polar_final_source4: _FrozenArray
    induced_gto_source4: _FrozenArray
    audit_coefficient_sum4: _FrozenArray
    array_content_sha256s: _FrozenObject
    vacuum_energy_eV: float
    polarization_energy_eV: float
    total_energy_eV: float
    cold_start: _FrozenObject
    wide_start: _FrozenObject
    cross_start_energy_abs_difference_eV: float
    cross_start_field_max_abs_difference_eV: float

    def __init__(self) -> None:
        raise TypeError("StateLeafV2 must be loaded with from_mapping().")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StateLeafV2":
        raw = _object(value, name="state leaf")
        _exact_fields(raw, _STATE_LEAF_FIELDS, name="state leaf")
        if raw["schema_id"] != STATE_LEAF_V2_SCHEMA:
            raise ValueError("state leaf schema_id is unsupported.")
        source_names = (
            "permanent_point_source4",
            "polar_zero_reference_source4",
            "polar_final_source4",
            "induced_gto_source4",
            "audit_coefficient_sum4",
        )
        sources = {
            name: _finite_matrix(raw[name], rows=None, columns=4, name=name)
            for name in source_names
        }
        atom_count = len(sources["audit_coefficient_sum4"])
        if any(len(source) != atom_count for source in sources.values()):
            raise ValueError("all state source arrays must have the same atom count.")
        for row_index in range(atom_count):
            for column_index in range(4):
                expected = (
                    sources["polar_final_source4"][row_index][column_index]
                    - sources["polar_zero_reference_source4"][row_index][column_index]
                )
                if sources["induced_gto_source4"][row_index][column_index] != expected:
                    raise ValueError(
                        "induced_gto_source4 must equal the frozen subtraction "
                        "polar_final_source4 - polar_zero_reference_source4."
                    )
                audit_expected = (
                    sources["permanent_point_source4"][row_index][column_index]
                    + sources["induced_gto_source4"][row_index][column_index]
                )
                if (
                    sources["audit_coefficient_sum4"][row_index][column_index]
                    != audit_expected
                ):
                    raise ValueError(
                        "audit_coefficient_sum4 must equal permanent plus induced."
                    )
        audit_charge = _finite_number(
            raw["audit_coefficient_sum_charge_e"],
            name="audit_coefficient_sum_charge_e",
        )
        if audit_charge != canonical_source_monopole_sum_v2(
            sources["audit_coefficient_sum4"]
        ):
            raise ValueError(
                "audit_coefficient_sum_charge_e does not equal audit monopoles."
            )
        source_basis = _text(
            raw["source_coefficient_basis_id"], name="source_coefficient_basis_id"
        )
        if source_basis != H1_V2_SOURCE_COEFFICIENT_BASIS_ID:
            raise ValueError("state source coefficient basis is unsupported.")
        source_space_sha256 = _sha256(
            raw["source_space_contract_sha256"],
            name="source_space_contract_sha256",
        )
        if source_space_sha256 != H1_V2_SOURCE_SPACE_CONTRACT_SHA256:
            raise ValueError("state source-space contract digest is unsupported.")
        raw_source_order = raw["source_coefficient_order"]
        if (
            not isinstance(raw_source_order, list)
            or tuple(raw_source_order) != H1_V2_SOURCE_COEFFICIENT_ORDER
        ):
            raise ValueError("state source coefficient order is unsupported.")
        receiver_space_id = _text(raw["receiver_space_id"], name="receiver_space_id")
        if receiver_space_id != H1_V2_RECEIVER_SPACE_ID:
            raise ValueError("state receiver-space identity is unsupported.")
        receiver_space_sha256 = _sha256(
            raw["receiver_space_contract_sha256"],
            name="receiver_space_contract_sha256",
        )
        if receiver_space_sha256 != H1_V2_RECEIVER_SPACE_CONTRACT_SHA256:
            raise ValueError("state receiver-space contract digest is unsupported.")
        raw_receiver_order = raw["receiver_component_order"]
        if (
            not isinstance(raw_receiver_order, list)
            or tuple(raw_receiver_order) != H1_V2_RECEIVER_COMPONENT_ORDER
        ):
            raise ValueError("state receiver component order is unsupported.")
        audit_contract = _object(
            raw["audit_projection_contract"], name="audit_projection_contract"
        )
        if audit_contract != h1_v2_audit_coefficient_sum_contract():
            raise ValueError("state audit projection contract is unsupported.")
        if (
            raw["numeric_array_encoding_contract"]
            != H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT
        ):
            raise ValueError("state numeric array encoding contract is unsupported.")
        if raw["endpoint_replay_scope"] != H1_V2_ENDPOINT_REPLAY_SCOPE:
            raise ValueError("state replay evidence scope is unsupported.")
        cold = _start_leaf(raw["cold_start"], atom_count=atom_count, name="cold_start")
        wide = _start_leaf(raw["wide_start"], atom_count=atom_count, name="wide_start")
        total_energy = _finite_number(raw["total_energy_eV"], name="total_energy_eV")
        polarization = _finite_number(
            raw["polarization_energy_eV"], name="polarization_energy_eV"
        )
        vacuum = _finite_number(raw["vacuum_energy_eV"], name="vacuum_energy_eV")
        if total_energy != vacuum + polarization:
            raise ValueError("total energy must equal vacuum plus polarization energy.")
        if cold["final_polarization_energy_eV"] != polarization:
            raise ValueError(
                "cold_start polarization energy must bind the selected state energy."
            )
        for start_name, start in (("cold_start", cold), ("wide_start", wide)):
            if (
                start["final_total_energy_eV"]
                != vacuum + start["final_polarization_energy_eV"]
            ):
                raise ValueError(
                    f"{start_name} total energy must equal this state's vacuum plus "
                    "its polarization energy."
                )
        if cold["final_total_energy_eV"] != total_energy:
            raise ValueError("cold_start total energy must bind the selected state energy.")
        cross_start_energy_difference = abs(
            cold["final_total_energy_eV"] - wide["final_total_energy_eV"]
        )
        cross_start_field_difference = max(
            abs(cold_value - wide_value)
            for cold_row, wide_row in zip(
                cold["final_native_field_eV"],
                wide["final_native_field_eV"],
                strict=True,
            )
            for cold_value, wide_value in zip(cold_row, wide_row, strict=True)
        )
        array_inputs = {
            "permanent_point_source4": (
                sources["permanent_point_source4"],
                "permanent-point-source4",
                source_space_sha256,
                H1_V2_SOURCE_COEFFICIENT_UNITS,
            ),
            "polar_zero_reference_source4": (
                sources["polar_zero_reference_source4"],
                "polar-zero-reference-source4-nonoperational",
                source_space_sha256,
                H1_V2_SOURCE_COEFFICIENT_UNITS,
            ),
            "polar_final_source4": (
                sources["polar_final_source4"],
                "polar-final-source4",
                source_space_sha256,
                H1_V2_SOURCE_COEFFICIENT_UNITS,
            ),
            "induced_gto_source4": (
                sources["induced_gto_source4"],
                "induced-gto1p5-source4",
                source_space_sha256,
                H1_V2_SOURCE_COEFFICIENT_UNITS,
            ),
            "audit_coefficient_sum4": (
                sources["audit_coefficient_sum4"],
                "audit-coefficient-sum4-nonoperational",
                source_space_sha256,
                H1_V2_SOURCE_COEFFICIENT_UNITS,
            ),
            "cold_start.final_native_field_eV": (
                cold["final_native_field_eV"],
                "cold-final-native-field8",
                receiver_space_sha256,
                H1_V2_RECEIVER_UNITS,
            ),
            "cold_start.final_residual_eV": (
                cold["final_residual_eV"],
                "cold-final-actual-residual-field8",
                receiver_space_sha256,
                H1_V2_RECEIVER_UNITS,
            ),
            "wide_start.final_native_field_eV": (
                wide["final_native_field_eV"],
                "wide-final-native-field8",
                receiver_space_sha256,
                H1_V2_RECEIVER_UNITS,
            ),
            "wide_start.final_residual_eV": (
                wide["final_residual_eV"],
                "wide-final-actual-residual-field8",
                receiver_space_sha256,
                H1_V2_RECEIVER_UNITS,
            ),
        }
        expected_array_sha256s = {
            name: canonical_numeric_array_sha256_v2(
                values,
                semantic_type=semantic_type,
                geometry_sha256=_sha256(
                    raw["geometry_sha256"], name="geometry_sha256"
                ),
                channel_space_contract_sha256=space_sha256,
                units=units,
            )
            for name, (values, semantic_type, space_sha256, units) in array_inputs.items()
        }
        raw_array_sha256s = _object(
            raw["array_content_sha256s"], name="array_content_sha256s"
        )
        _exact_fields(
            raw_array_sha256s,
            frozenset(expected_array_sha256s),
            name="array_content_sha256s",
        )
        normalized_array_sha256s = {
            name: _sha256(raw_array_sha256s[name], name=f"array_content_sha256s.{name}")
            for name in expected_array_sha256s
        }
        if normalized_array_sha256s != expected_array_sha256s:
            raise ValueError("state array content digests do not match canonical bytes.")

        payload: dict[str, object] = {
            "schema_id": STATE_LEAF_V2_SCHEMA,
            "legacy_root_sha256": _sha256(
                raw["legacy_root_sha256"], name="legacy_root_sha256"
            ),
            "prepared_pes_configuration_sha256": _sha256(
                raw["prepared_pes_configuration_sha256"],
                name="prepared_pes_configuration_sha256",
            ),
            "geometry_sha256": _sha256(raw["geometry_sha256"], name="geometry_sha256"),
            "provider_configuration_sha256": _sha256(
                raw["provider_configuration_sha256"],
                name="provider_configuration_sha256",
            ),
            "topology_id": _sha256(raw["topology_id"], name="topology_id"),
            "target_charge_e": _finite_number(
                raw["target_charge_e"], name="target_charge_e"
            ),
            "audit_coefficient_sum_charge_e": audit_charge,
            "source_coefficient_basis_id": source_basis,
            "source_space_contract_sha256": source_space_sha256,
            "source_coefficient_order": list(H1_V2_SOURCE_COEFFICIENT_ORDER),
            "receiver_space_id": receiver_space_id,
            "receiver_space_contract_sha256": receiver_space_sha256,
            "receiver_component_order": list(H1_V2_RECEIVER_COMPONENT_ORDER),
            "audit_projection_contract": h1_v2_audit_coefficient_sum_contract(),
            "numeric_array_encoding_contract": H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT,
            "endpoint_replay_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
            **{name: [list(row) for row in sources[name]] for name in source_names},
            "array_content_sha256s": normalized_array_sha256s,
            "vacuum_energy_eV": vacuum,
            "polarization_energy_eV": polarization,
            "total_energy_eV": total_energy,
            "cold_start": cold,
            "wide_start": wide,
        }
        digest = _sha256(raw["state_leaf_sha256"], name="state_leaf_sha256")
        if digest != canonical_json_sha256(payload):
            raise ValueError(
                "state_leaf_sha256 does not match normalized state content."
            )
        frozen = _freeze_object(payload, name="state leaf")
        values = {key: value for key, value in frozen.items}
        values["state_leaf_sha256"] = digest
        values["cross_start_energy_abs_difference_eV"] = cross_start_energy_difference
        values["cross_start_field_max_abs_difference_eV"] = cross_start_field_difference
        return _sealed_instance(cls, values)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, object]:
        payload = {
            field: _thaw_json(getattr(self, field))
            for field in _STATE_LEAF_FIELDS
            if field != "state_leaf_sha256"
        }
        return {**payload, "state_leaf_sha256": self.state_leaf_sha256}


_SOLVE_EVENT_FIELDS = frozenset(
    {
        "schema_id",
        "solve_event_sha256",
        "event_index",
        "state_leaf_sha256",
        "geometry_sha256",
        "provider_configuration_sha256",
        "topology_id",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class SolveEventV2:
    schema_id: str
    solve_event_sha256: str
    event_index: int
    state_leaf_sha256: str
    geometry_sha256: str
    provider_configuration_sha256: str
    topology_id: str

    def __init__(self) -> None:
        raise TypeError("SolveEventV2 must be loaded with from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        state_leaves: Mapping[str, StateLeafV2],
        expected_index: int,
    ) -> "SolveEventV2":
        raw = _object(value, name="solve event")
        _exact_fields(raw, _SOLVE_EVENT_FIELDS, name="solve event")
        if raw["schema_id"] != SOLVE_EVENT_V2_SCHEMA:
            raise ValueError("solve event schema_id is unsupported.")
        index = _nonnegative_int(raw["event_index"], name="event_index")
        if index != expected_index:
            raise ValueError("solve events must be contiguous and in execution order.")
        leaf_digest = _sha256(raw["state_leaf_sha256"], name="state_leaf_sha256")
        if leaf_digest not in state_leaves:
            raise ValueError("solve event references an unknown state leaf.")
        leaf = state_leaves[leaf_digest]
        payload = {
            "schema_id": SOLVE_EVENT_V2_SCHEMA,
            "event_index": index,
            "state_leaf_sha256": leaf_digest,
            "geometry_sha256": _sha256(raw["geometry_sha256"], name="geometry_sha256"),
            "provider_configuration_sha256": _sha256(
                raw["provider_configuration_sha256"],
                name="provider_configuration_sha256",
            ),
            "topology_id": _sha256(raw["topology_id"], name="topology_id"),
        }
        for field in (
            "geometry_sha256",
            "provider_configuration_sha256",
            "topology_id",
        ):
            if payload[field] != getattr(leaf, field):
                raise ValueError(f"solve event {field} does not bind its state leaf.")
        digest = _sha256(raw["solve_event_sha256"], name="solve_event_sha256")
        if digest != canonical_json_sha256(payload):
            raise ValueError(
                "solve_event_sha256 does not match normalized event content."
            )
        return _sealed_instance(cls, {**payload, "solve_event_sha256": digest})  # type: ignore[return-value]


_COMPONENT_STENCIL_FIELDS = frozenset(
    {
        "atom_index",
        "axis_index",
        "coarse_step_angstrom",
        "fine_step_angstrom",
        "center_event_index",
        "plus_h_event_index",
        "minus_h_event_index",
        "plus_h2_event_index",
        "minus_h2_event_index",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class ComponentStencilV2:
    atom_index: int
    axis_index: int
    coarse_step_angstrom: float
    fine_step_angstrom: float
    event_indices: tuple[int, int, int, int, int]
    richardson_force_eV_per_angstrom: float
    local_error_eV_per_angstrom: float

    def __init__(self) -> None:
        raise TypeError("ComponentStencilV2 must be loaded with from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
    ) -> "ComponentStencilV2":
        raw = _object(value, name="component stencil")
        _exact_fields(raw, _COMPONENT_STENCIL_FIELDS, name="component stencil")
        atom = _nonnegative_int(raw["atom_index"], name="atom_index")
        axis = _nonnegative_int(raw["axis_index"], name="axis_index")
        if axis >= 3:
            raise ValueError("axis_index must be smaller than three.")
        coarse = _positive_float(raw["coarse_step_angstrom"], name="coarse_step")
        fine = _positive_float(raw["fine_step_angstrom"], name="fine_step")
        if fine != coarse / 2.0:
            raise ValueError("component stencil steps must be exact nested halvings.")
        names = (
            "center_event_index",
            "plus_h_event_index",
            "minus_h_event_index",
            "plus_h2_event_index",
            "minus_h2_event_index",
        )
        indices = tuple(_nonnegative_int(raw[name], name=name) for name in names)
        if len(set(indices)) != len(indices) or any(
            i >= len(solve_events) for i in indices
        ):
            raise ValueError(
                "component stencil event references must be distinct and in range."
            )
        events = tuple(solve_events[index] for index in indices)
        leaves = tuple(state_leaves[event.state_leaf_sha256] for event in events)
        if len({leaf.topology_id for leaf in leaves}) != 1:
            raise ValueError("component stencil states must share one topology.")
        center, plus_h, minus_h, plus_h2, minus_h2 = leaves
        derivative_h = (plus_h.total_energy_eV - minus_h.total_energy_eV) / (
            2.0 * coarse
        )
        derivative_h2 = (plus_h2.total_energy_eV - minus_h2.total_energy_eV) / (
            2.0 * fine
        )
        richardson = (4.0 * derivative_h2 - derivative_h) / 3.0
        return _sealed_instance(
            cls,
            {
                "atom_index": atom,
                "axis_index": axis,
                "coarse_step_angstrom": coarse,
                "fine_step_angstrom": fine,
                "event_indices": indices,
                "richardson_force_eV_per_angstrom": -richardson,
                "local_error_eV_per_angstrom": abs(richardson - derivative_h2),
            },
        )  # type: ignore[return-value]


_BENZENE_STENCIL_FIELDS = _COMPONENT_STENCIL_FIELDS | frozenset(
    {
        "independent_step_angstrom",
        "plus_h4_event_index",
        "minus_h4_event_index",
        "audit_only",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class BenzenePredecessorStencilV2:
    component: ComponentStencilV2
    independent_step_angstrom: float
    h4_event_indices: tuple[int, int]
    independent_h4_force_difference_eV_per_angstrom: float
    audit_only: bool

    def __init__(self) -> None:
        raise TypeError("BenzenePredecessorStencilV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
    ) -> "BenzenePredecessorStencilV2":
        raw = _object(value, name="benzene predecessor stencil")
        _exact_fields(raw, _BENZENE_STENCIL_FIELDS, name="benzene predecessor stencil")
        component = ComponentStencilV2.from_mapping(
            {key: raw[key] for key in _COMPONENT_STENCIL_FIELDS},
            solve_events=solve_events,
            state_leaves=state_leaves,
        )
        independent = _positive_float(
            raw["independent_step_angstrom"], name="independent_step_angstrom"
        )
        if independent != component.fine_step_angstrom / 2.0:
            raise ValueError("benzene independent step must be the exact h/4 step.")
        indices = tuple(
            _nonnegative_int(raw[name], name=name)
            for name in ("plus_h4_event_index", "minus_h4_event_index")
        )
        if len(set((*component.event_indices, *indices))) != 7 or any(
            index >= len(solve_events) for index in indices
        ):
            raise ValueError(
                "benzene h/4 event references must be distinct and in range."
            )
        leaves = tuple(
            state_leaves[solve_events[index].state_leaf_sha256] for index in indices
        )
        center_topology = state_leaves[
            solve_events[component.event_indices[0]].state_leaf_sha256
        ].topology_id
        if any(leaf.topology_id != center_topology for leaf in leaves):
            raise ValueError("benzene h/4 states must share the predecessor topology.")
        derivative_h4 = (leaves[0].total_energy_eV - leaves[1].total_energy_eV) / (
            2.0 * independent
        )
        audit_only = _strict_bool(raw["audit_only"], name="audit_only")
        if audit_only is not True:
            raise ValueError(
                "benzene h/4 comparison must remain explicitly audit_only."
            )
        return _sealed_instance(
            cls,
            {
                "component": component,
                "independent_step_angstrom": independent,
                "h4_event_indices": indices,
                "independent_h4_force_difference_eV_per_angstrom": abs(
                    component.richardson_force_eV_per_angstrom - (-derivative_h4)
                ),
                "audit_only": True,
            },
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class CartesianPanelV2:
    label: str
    components: tuple[ComponentStencilV2, ...]

    def __init__(self) -> None:
        raise TypeError("CartesianPanelV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
    ) -> "CartesianPanelV2":
        raw = _object(value, name="Cartesian panel")
        _exact_fields(raw, frozenset({"label", "components"}), name="Cartesian panel")
        label = _text(raw["label"], name="Cartesian panel label")
        if label not in {"base", "translated", "rotated"}:
            raise ValueError("Cartesian panel label is unsupported.")
        raw_components = raw["components"]
        if not isinstance(raw_components, list) or len(raw_components) != 9:
            raise ValueError("Cartesian panel must contain exactly nine components.")
        components = tuple(
            ComponentStencilV2.from_mapping(
                item, solve_events=solve_events, state_leaves=state_leaves
            )
            for item in raw_components
        )
        for index, component in enumerate(components):
            if (component.atom_index, component.axis_index) != divmod(index, 3):
                raise ValueError(
                    "Cartesian components must use index=3*atom+axis ordering."
                )
        if len({component.event_indices[0] for component in components}) != 1:
            raise ValueError(
                "Cartesian components must reference one shared center event."
            )
        return _sealed_instance(cls, {"label": label, "components": components})  # type: ignore[return-value]


def _ase_geometry_sha256(
    atomic_numbers: tuple[int, ...],
    positions_angstrom: tuple[tuple[float, float, float], ...],
) -> str:
    return canonical_json_sha256(
        {
            "kind": "ase-geometry-v1",
            "atomic_numbers": list(atomic_numbers),
            "positions_A": [list(row) for row in positions_angstrom],
            "cell_A": [[0.0, 0.0, 0.0]] * 3,
            "pbc": [False, False, False],
        }
    )


def _event_leaf(
    index: int,
    *,
    solve_events: tuple[SolveEventV2, ...],
    state_leaves: Mapping[str, StateLeafV2],
    name: str,
) -> StateLeafV2:
    event_index = _nonnegative_int(index, name=name)
    if event_index >= len(solve_events):
        raise ValueError(f"{name} is out of range.")
    return state_leaves[solve_events[event_index].state_leaf_sha256]


def _panel_center_leaf(
    panel: CartesianPanelV2,
    *,
    solve_events: tuple[SolveEventV2, ...],
    state_leaves: Mapping[str, StateLeafV2],
) -> StateLeafV2:
    return _event_leaf(
        panel.components[0].event_indices[0],
        solve_events=solve_events,
        state_leaves=state_leaves,
        name="Cartesian panel center_event_index",
    )


def _validate_component_geometry(
    component: ComponentStencilV2,
    center_positions: np.ndarray,
    *,
    solve_events: tuple[SolveEventV2, ...],
    state_leaves: Mapping[str, StateLeafV2],
) -> None:
    displacements = (
        0.0,
        component.coarse_step_angstrom,
        -component.coarse_step_angstrom,
        component.fine_step_angstrom,
        -component.fine_step_angstrom,
    )
    for event_index, displacement in zip(
        component.event_indices, displacements, strict=True
    ):
        positions = np.array(center_positions, copy=True)
        positions[component.atom_index, component.axis_index] += displacement
        leaf = _event_leaf(
            event_index,
            solve_events=solve_events,
            state_leaves=state_leaves,
            name="component geometry event",
        )
        expected = _ase_geometry_sha256(
            (8, 1, 1),
            tuple(tuple(float(item) for item in row) for row in positions),
        )
        if leaf.geometry_sha256 != expected:
            raise ValueError("component stencil geometry does not match its preimage.")


@dataclass(frozen=True, slots=True, init=False)
class DirectionalCheckV2:
    seed: int
    normalized_direction: _FrozenArray
    step_angstrom: float
    event_indices: tuple[int, int]
    energy_derivative_eV_per_angstrom: float
    negative_force_projection_eV_per_angstrom: float
    absolute_error_eV_per_angstrom: float

    def __init__(self) -> None:
        raise TypeError("DirectionalCheckV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
        base_panel: CartesianPanelV2,
        prepared_water_positions: tuple[tuple[float, float, float], ...],
    ) -> "DirectionalCheckV2":
        raw = _object(value, name="directional check")
        _exact_fields(
            raw,
            frozenset(
                {
                    "seed",
                    "normalized_direction",
                    "step_angstrom",
                    "plus_event_index",
                    "minus_event_index",
                }
            ),
            name="directional check",
        )
        seed = _nonnegative_int(raw["seed"], name="directional seed")
        if seed != 20260816:
            raise ValueError("directional seed must equal 20260816.")
        direction = _finite_matrix(
            raw["normalized_direction"],
            rows=3,
            columns=3,
            name="normalized_direction",
        )
        expected = np.random.default_rng(seed).normal(size=(3, 3))
        expected /= np.linalg.norm(expected)
        if direction != tuple(tuple(float(item) for item in row) for row in expected):
            raise ValueError("normalized_direction differs from the frozen NumPy draw.")
        step = _positive_float(raw["step_angstrom"], name="directional step")
        if step != 1.25e-4:
            raise ValueError("directional step must equal the frozen h/4 step.")
        indices = tuple(
            _nonnegative_int(raw[name], name=name)
            for name in ("plus_event_index", "minus_event_index")
        )
        if indices[0] == indices[1]:
            raise ValueError("directional event references must be distinct.")
        leaves = tuple(
            _event_leaf(
                index,
                solve_events=solve_events,
                state_leaves=state_leaves,
                name="directional event index",
            )
            for index in indices
        )
        if len({leaf.topology_id for leaf in leaves}) != 1:
            raise ValueError("directional states must share one topology.")
        positions = np.asarray(prepared_water_positions, dtype=float)
        if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("prepared water positions must be finite (3,3).")
        for leaf, sign in zip(leaves, (1.0, -1.0), strict=True):
            displaced = positions + sign * step * expected
            geometry = _ase_geometry_sha256(
                (8, 1, 1),
                tuple(tuple(float(item) for item in row) for row in displaced),
            )
            if leaf.geometry_sha256 != geometry:
                raise ValueError(
                    "directional event geometry does not match its preimage."
                )
        if base_panel.label != "base":
            raise ValueError("directional force projection requires the base panel.")
        base_center = _panel_center_leaf(
            base_panel, solve_events=solve_events, state_leaves=state_leaves
        )
        if base_center.geometry_sha256 != _ase_geometry_sha256(
            (8, 1, 1),
            tuple(tuple(float(item) for item in row) for row in positions),
        ):
            raise ValueError("base panel center does not bind prepared water geometry.")
        for component in base_panel.components:
            _validate_component_geometry(
                component,
                positions,
                solve_events=solve_events,
                state_leaves=state_leaves,
            )
        if any(leaf.topology_id != base_center.topology_id for leaf in leaves):
            raise ValueError("directional and base states must share one topology.")
        energy_derivative = (leaves[0].total_energy_eV - leaves[1].total_energy_eV) / (
            2.0 * step
        )
        force_projection = -sum(
            component.richardson_force_eV_per_angstrom
            * direction[component.atom_index][component.axis_index]
            for component in base_panel.components
        )
        return _sealed_instance(
            cls,
            {
                "seed": seed,
                "normalized_direction": _freeze_json(
                    [list(row) for row in direction], name="normalized_direction"
                ),
                "step_angstrom": step,
                "event_indices": indices,
                "energy_derivative_eV_per_angstrom": energy_derivative,
                "negative_force_projection_eV_per_angstrom": force_projection,
                "absolute_error_eV_per_angstrom": abs(
                    energy_derivative - force_projection
                ),
            },
        )  # type: ignore[return-value]


def _transformed_panel(
    raw_panel: object,
    *,
    expected_label: str,
    expected_positions: np.ndarray,
    solve_events: tuple[SolveEventV2, ...],
    state_leaves: Mapping[str, StateLeafV2],
) -> CartesianPanelV2:
    panel = CartesianPanelV2.from_mapping(
        _object(raw_panel, name=f"{expected_label} panel"),
        solve_events=solve_events,
        state_leaves=state_leaves,
    )
    if panel.label != expected_label:
        raise ValueError(f"transformed panel label must be {expected_label}.")
    expected_geometry = _ase_geometry_sha256(
        (8, 1, 1),
        tuple(tuple(float(item) for item in row) for row in expected_positions),
    )
    if (
        _panel_center_leaf(
            panel, solve_events=solve_events, state_leaves=state_leaves
        ).geometry_sha256
        != expected_geometry
    ):
        raise ValueError(f"{expected_label} panel center geometry is invalid.")
    for component in panel.components:
        _validate_component_geometry(
            component,
            expected_positions,
            solve_events=solve_events,
            state_leaves=state_leaves,
        )
    return panel


@dataclass(frozen=True, slots=True, init=False)
class RigidTranslationV2:
    translation_angstrom: tuple[float, float, float]
    panel: CartesianPanelV2

    def __init__(self) -> None:
        raise TypeError("RigidTranslationV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
        prepared_water_positions: tuple[tuple[float, float, float], ...],
    ) -> "RigidTranslationV2":
        raw = _object(value, name="rigid translation")
        _exact_fields(
            raw, frozenset({"translation_angstrom", "panel"}), name="rigid translation"
        )
        vector = tuple(
            _finite_number(item, name="translation component")
            for item in _strict_json_array(
                raw["translation_angstrom"], length=3, name="translation_angstrom"
            )
        )
        if vector != (4.2, -3.1, 1.7):
            raise ValueError("translation_angstrom differs from the frozen vector.")
        positions = np.asarray(prepared_water_positions, dtype=float)
        panel = _transformed_panel(
            raw["panel"],
            expected_label="translated",
            expected_positions=positions + np.asarray(vector),
            solve_events=solve_events,
            state_leaves=state_leaves,
        )
        return _sealed_instance(
            cls, {"translation_angstrom": vector, "panel": panel}
        )  # type: ignore[return-value]


def _strict_json_array(value: object, *, length: int, name: str) -> list[object]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a JSON array of length {length}.")
    return value


def _frozen_rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


@dataclass(frozen=True, slots=True, init=False)
class RigidRotationV2:
    seed: int
    rotation_matrix: _FrozenArray
    panel: CartesianPanelV2

    def __init__(self) -> None:
        raise TypeError("RigidRotationV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
        prepared_water_positions: tuple[tuple[float, float, float], ...],
    ) -> "RigidRotationV2":
        raw = _object(value, name="rigid rotation")
        _exact_fields(
            raw,
            frozenset({"seed", "rotation_matrix", "panel"}),
            name="rigid rotation",
        )
        seed = _nonnegative_int(raw["seed"], name="rotation seed")
        if seed != 20260817:
            raise ValueError("rotation seed must equal 20260817.")
        matrix = _finite_matrix(
            raw["rotation_matrix"], rows=3, columns=3, name="rotation_matrix"
        )
        values = np.asarray(matrix)
        expected = _frozen_rotation(seed)
        if matrix != tuple(tuple(float(item) for item in row) for row in expected):
            raise ValueError("rotation_matrix differs from the frozen QR rotation.")
        if not np.allclose(values @ values.T, np.eye(3), atol=2.0e-15, rtol=0.0):
            raise ValueError("rotation_matrix is not orthogonal.")
        if not math.isclose(
            float(np.linalg.det(values)), 1.0, rel_tol=0.0, abs_tol=2.0e-15
        ):
            raise ValueError("rotation_matrix is not a proper rotation.")
        positions = np.asarray(prepared_water_positions, dtype=float)
        centroid = np.mean(positions, axis=0)
        rotated = (positions - centroid) @ values.T + centroid
        panel = _transformed_panel(
            raw["panel"],
            expected_label="rotated",
            expected_positions=rotated,
            solve_events=solve_events,
            state_leaves=state_leaves,
        )
        return _sealed_instance(
            cls,
            {
                "seed": seed,
                "rotation_matrix": _freeze_json(
                    [list(row) for row in matrix], name="rotation_matrix"
                ),
                "panel": panel,
            },
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class ClosedLoopV2:
    cartesian_dofs: tuple[tuple[int, int], tuple[int, int]]
    half_width_angstrom: float
    orientation: str
    components: tuple[ComponentStencilV2, ...]
    closed_loop_work_eV: float
    numerical_work_error_bound_eV: float
    guarded_absolute_work_eV: float

    def __init__(self) -> None:
        raise TypeError("ClosedLoopV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
        prepared_water_positions: tuple[tuple[float, float, float], ...],
    ) -> "ClosedLoopV2":
        raw = _object(value, name="closed loop")
        _exact_fields(
            raw,
            frozenset(
                {"cartesian_dofs", "half_width_angstrom", "orientation", "edges"}
            ),
            name="closed loop",
        )
        dofs_raw = _strict_json_array(
            raw["cartesian_dofs"], length=2, name="cartesian_dofs"
        )
        dofs = tuple(
            tuple(
                _nonnegative_int(item, name="cartesian DOF index")
                for item in _strict_json_array(dof, length=2, name="cartesian DOF")
            )
            for dof in dofs_raw
        )
        if dofs != ((0, 0), (1, 1)):
            raise ValueError("closed-loop cartesian_dofs differ from the frozen DOFs.")
        half_width = _positive_float(raw["half_width_angstrom"], name="half width")
        if half_width != 1.0e-3:
            raise ValueError("closed-loop half_width_angstrom must equal 0.001.")
        orientation = _text(raw["orientation"], name="orientation")
        if orientation != "counterclockwise":
            raise ValueError("closed-loop orientation must be counterclockwise.")
        edge_contract = (
            ("bottom", (0.0, -half_width), 0, 0, 2.0 * half_width),
            ("right", (half_width, 0.0), 1, 1, 2.0 * half_width),
            ("top", (0.0, half_width), 0, 0, -2.0 * half_width),
            ("left", (-half_width, 0.0), 1, 1, -2.0 * half_width),
        )
        edges = _strict_json_array(raw["edges"], length=4, name="closed-loop edges")
        positions = np.asarray(prepared_water_positions, dtype=float)
        components: list[ComponentStencilV2] = []
        center_topologies: list[str] = []
        work = 0.0
        error_bound = 0.0
        for index, (edge_raw, contract) in enumerate(
            zip(edges, edge_contract, strict=True)
        ):
            edge = _object(edge_raw, name=f"closed-loop edge {index}")
            _exact_fields(
                edge,
                frozenset(
                    {
                        "label",
                        "midpoint_offsets_angstrom",
                        "displacement_angstrom",
                        "component",
                    }
                ),
                name=f"closed-loop edge {index}",
            )
            label, offsets, atom, axis, displacement = contract
            if edge["label"] != label:
                raise ValueError("closed-loop edge labels/order drifted.")
            actual_offsets = tuple(
                _finite_number(item, name="midpoint offset")
                for item in _strict_json_array(
                    edge["midpoint_offsets_angstrom"],
                    length=2,
                    name="midpoint offsets",
                )
            )
            if actual_offsets != offsets:
                raise ValueError("closed-loop midpoint offsets drifted.")
            actual_displacement = _finite_number(
                edge["displacement_angstrom"], name="edge displacement"
            )
            if actual_displacement != displacement:
                raise ValueError("closed-loop edge displacement drifted.")
            component = ComponentStencilV2.from_mapping(
                _object(edge["component"], name="edge component"),
                solve_events=solve_events,
                state_leaves=state_leaves,
            )
            if (component.atom_index, component.axis_index) != (atom, axis):
                raise ValueError("closed-loop edge component DOF drifted.")
            midpoint = np.array(positions, copy=True)
            midpoint[0, 0] += offsets[0]
            midpoint[1, 1] += offsets[1]
            center_leaf = _event_leaf(
                component.event_indices[0],
                solve_events=solve_events,
                state_leaves=state_leaves,
                name="closed-loop center event",
            )
            if center_leaf.geometry_sha256 != _ase_geometry_sha256(
                (8, 1, 1),
                tuple(tuple(float(item) for item in row) for row in midpoint),
            ):
                raise ValueError("closed-loop midpoint geometry is invalid.")
            _validate_component_geometry(
                component,
                midpoint,
                solve_events=solve_events,
                state_leaves=state_leaves,
            )
            center_topologies.append(center_leaf.topology_id)
            contribution = component.richardson_force_eV_per_angstrom * displacement
            work += contribution
            error_bound += component.local_error_eV_per_angstrom * abs(displacement)
            components.append(component)
        if len(set(center_topologies)) != 1:
            raise ValueError("closed-loop midpoint states must share one topology.")
        return _sealed_instance(
            cls,
            {
                "cartesian_dofs": dofs,
                "half_width_angstrom": half_width,
                "orientation": orientation,
                "components": tuple(components),
                "closed_loop_work_eV": work,
                "numerical_work_error_bound_eV": error_bound,
                "guarded_absolute_work_eV": abs(work) + error_bound,
            },
        )  # type: ignore[return-value]


def _event_range(value: object, *, length: int, name: str) -> tuple[int, int]:
    items = _strict_json_array(value, length=2, name=name)
    start, end = tuple(_nonnegative_int(item, name=f"{name} index") for item in items)
    if end - start != length:
        raise ValueError(f"{name} must be half-open with exact length {length}.")
    return start, end


def _panel_reference_indices(panel: CartesianPanelV2) -> tuple[int, ...]:
    return tuple(
        index for component in panel.components for index in component.event_indices
    )


def _require_panel_schedule(panel: CartesianPanelV2, *, start: int) -> None:
    for component_index, component in enumerate(panel.components):
        displaced = start + 1 + 4 * component_index
        expected = (start, displaced, displaced + 1, displaced + 2, displaced + 3)
        if component.event_indices != expected:
            raise ValueError("Cartesian panel event occurrence schedule drifted.")


def _range_leaves(
    event_range: tuple[int, int],
    *,
    solve_events: tuple[SolveEventV2, ...],
    state_leaves: Mapping[str, StateLeafV2],
) -> tuple[StateLeafV2, ...]:
    start, end = event_range
    if end > len(solve_events):
        raise ValueError("system event_range exceeds the solve-event ledger.")
    return tuple(
        state_leaves[solve_events[index].state_leaf_sha256]
        for index in range(start, end)
    )


def _root_metrics(leaves: tuple[StateLeafV2, ...]) -> dict[str, object]:
    starts = tuple(
        (_thaw_json(leaf.cold_start), _thaw_json(leaf.wide_start)) for leaf in leaves
    )
    return {
        "maximum_final_residual_norm_eV": max(
            math.sqrt(
                sum(
                    float(value) ** 2
                    for row in start["final_residual_eV"]
                    for value in row
                )
            )
            for pair in starts
            for start in pair
        ),
        "maximum_total_charge_error_e": max(
            abs(leaf.audit_coefficient_sum_charge_e - leaf.target_charge_e)
            for leaf in leaves
        ),
        "maximum_cross_start_energy_abs_difference_eV": max(
            leaf.cross_start_energy_abs_difference_eV for leaf in leaves
        ),
        "maximum_cross_start_field_max_abs_difference_eV": max(
            leaf.cross_start_field_max_abs_difference_eV for leaf in leaves
        ),
        "maximum_cold_iterations": max(pair[0]["iterations"] for pair in starts),
        "maximum_wide_iterations": max(pair[1]["iterations"] for pair in starts),
        "all_starts_converged": all(
            start["converged"] for pair in starts for start in pair
        ),
    }


@dataclass(frozen=True, slots=True, init=False)
class BenzeneSystemV2:
    event_range: tuple[int, int]
    predecessor_stencil: BenzenePredecessorStencilV2
    reported_center_event_index: int
    topology_id: str
    maximum_final_residual_norm_eV: float
    maximum_total_charge_error_e: float
    maximum_cross_start_energy_abs_difference_eV: float
    maximum_cross_start_field_max_abs_difference_eV: float
    maximum_cold_iterations: int
    maximum_wide_iterations: int
    all_starts_converged: bool

    def __init__(self) -> None:
        raise TypeError("BenzeneSystemV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
        prepared_pes_configuration_sha256: str,
    ) -> "BenzeneSystemV2":
        raw = _object(value, name="benzene system")
        _exact_fields(
            raw,
            frozenset(
                {"event_range", "predecessor_stencil", "reported_center_event_index"}
            ),
            name="benzene system",
        )
        occurrence_range = _event_range(
            raw["event_range"], length=8, name="benzene event_range"
        )
        if occurrence_range != (0, 8):
            raise ValueError("benzene event_range must equal [0, 8].")
        start, end = occurrence_range
        stencil = BenzenePredecessorStencilV2.from_mapping(
            _object(raw["predecessor_stencil"], name="predecessor_stencil"),
            solve_events=solve_events,
            state_leaves=state_leaves,
        )
        expected_references = tuple(range(start, start + 7))
        actual_references = (
            *stencil.component.event_indices,
            *stencil.h4_event_indices,
        )
        if actual_references != expected_references:
            raise ValueError("benzene predecessor event occurrence schedule drifted.")
        reported_center = _nonnegative_int(
            raw["reported_center_event_index"], name="reported_center_event_index"
        )
        if reported_center != start + 7:
            raise ValueError("benzene reported center must be the eighth occurrence.")
        leaves = _range_leaves(
            occurrence_range, solve_events=solve_events, state_leaves=state_leaves
        )
        prepared_pes = _sha256(
            prepared_pes_configuration_sha256,
            name="benzene prepared PES configuration",
        )
        if any(
            leaf.prepared_pes_configuration_sha256 != prepared_pes for leaf in leaves
        ):
            raise ValueError(
                "benzene leaves do not bind the sealed prepared PES configuration."
            )
        topologies = {leaf.topology_id for leaf in leaves}
        if len(topologies) != 1:
            raise ValueError("benzene system occurrences must share one topology.")
        if (
            solve_events[start].state_leaf_sha256
            != solve_events[reported_center].state_leaf_sha256
        ):
            raise ValueError(
                "benzene reported center must be a distinct occurrence of the exact "
                "predecessor center StateLeaf."
            )
        if set((*actual_references, reported_center)) != set(range(start, end)):
            raise ValueError(
                "benzene event_range contains unused or extra occurrences."
            )
        return _sealed_instance(
            cls,
            {
                "event_range": occurrence_range,
                "predecessor_stencil": stencil,
                "reported_center_event_index": reported_center,
                "topology_id": next(iter(topologies)),
                **_root_metrics(leaves),
            },
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, init=False)
class WaterSystemV2:
    event_range: tuple[int, int]
    base_panel: CartesianPanelV2
    directional: DirectionalCheckV2
    translation: RigidTranslationV2
    rotation: RigidRotationV2
    closed_loop: ClosedLoopV2
    topology_id: str
    base_forces_eV_per_angstrom: _FrozenArray
    translated_forces_eV_per_angstrom: _FrozenArray
    rotated_forces_eV_per_angstrom: _FrozenArray
    base_center_energy_eV: float
    translated_center_energy_eV: float
    rotated_center_energy_eV: float
    net_force_norm_eV_per_angstrom: float
    translation_energy_abs_difference_eV: float
    translation_force_relative_difference: float
    translation_force_max_abs_difference_eV_per_angstrom: float
    rotation_energy_abs_difference_eV: float
    rotation_force_covariance_relative_difference: float
    rotation_force_covariance_max_abs_difference_eV_per_angstrom: float
    directional_absolute_error_eV_per_angstrom: float
    maximum_local_error_eV_per_angstrom: float
    closed_loop_work_eV: float
    closed_loop_numerical_error_bound_eV: float
    closed_loop_guarded_absolute_work_eV: float
    maximum_final_residual_norm_eV: float
    maximum_total_charge_error_e: float
    maximum_cross_start_energy_abs_difference_eV: float
    maximum_cross_start_field_max_abs_difference_eV: float
    maximum_cold_iterations: int
    maximum_wide_iterations: int
    all_starts_converged: bool

    def __init__(self) -> None:
        raise TypeError("WaterSystemV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        solve_events: tuple[SolveEventV2, ...],
        state_leaves: Mapping[str, StateLeafV2],
        prepared_water_positions: tuple[tuple[float, float, float], ...],
        prepared_pes_configuration_sha256: str,
    ) -> "WaterSystemV2":
        raw = _object(value, name="water system")
        _exact_fields(
            raw,
            frozenset(
                {
                    "event_range",
                    "base_panel",
                    "directional",
                    "translation",
                    "rotation",
                    "closed_loop",
                }
            ),
            name="water system",
        )
        occurrence_range = _event_range(
            raw["event_range"], length=133, name="water event_range"
        )
        if occurrence_range != (8, 141):
            raise ValueError("water event_range must equal [8, 141].")
        start, end = occurrence_range
        base = CartesianPanelV2.from_mapping(
            _object(raw["base_panel"], name="base_panel"),
            solve_events=solve_events,
            state_leaves=state_leaves,
        )
        _require_panel_schedule(base, start=start)
        directional = DirectionalCheckV2.from_mapping(
            _object(raw["directional"], name="directional"),
            solve_events=solve_events,
            state_leaves=state_leaves,
            base_panel=base,
            prepared_water_positions=prepared_water_positions,
        )
        if directional.event_indices != (start + 37, start + 38):
            raise ValueError("water directional occurrence schedule drifted.")
        translation = RigidTranslationV2.from_mapping(
            _object(raw["translation"], name="translation"),
            solve_events=solve_events,
            state_leaves=state_leaves,
            prepared_water_positions=prepared_water_positions,
        )
        _require_panel_schedule(translation.panel, start=start + 39)
        rotation = RigidRotationV2.from_mapping(
            _object(raw["rotation"], name="rotation"),
            solve_events=solve_events,
            state_leaves=state_leaves,
            prepared_water_positions=prepared_water_positions,
        )
        _require_panel_schedule(rotation.panel, start=start + 76)
        closed_loop = ClosedLoopV2.from_mapping(
            _object(raw["closed_loop"], name="closed_loop"),
            solve_events=solve_events,
            state_leaves=state_leaves,
            prepared_water_positions=prepared_water_positions,
        )
        expected_loop_indices = tuple(range(start + 113, end))
        actual_loop_indices = tuple(
            index
            for component in closed_loop.components
            for index in component.event_indices
        )
        if actual_loop_indices != expected_loop_indices:
            raise ValueError("water closed-loop occurrence schedule drifted.")

        references = (
            *_panel_reference_indices(base),
            *directional.event_indices,
            *_panel_reference_indices(translation.panel),
            *_panel_reference_indices(rotation.panel),
            *actual_loop_indices,
        )
        counts = {index: references.count(index) for index in set(references)}
        expected_repeated_centers = {start, start + 39, start + 76}
        if set(counts) != set(range(start, end)) or any(
            count != (9 if index in expected_repeated_centers else 1)
            for index, count in counts.items()
        ):
            raise ValueError(
                "water reference union has duplicate-equivalent, unused, or extra events."
            )
        leaves = _range_leaves(
            occurrence_range, solve_events=solve_events, state_leaves=state_leaves
        )
        prepared_pes = _sha256(
            prepared_pes_configuration_sha256,
            name="water prepared PES configuration",
        )
        if any(
            leaf.prepared_pes_configuration_sha256 != prepared_pes for leaf in leaves
        ):
            raise ValueError(
                "water leaves do not bind the sealed prepared PES configuration."
            )
        topologies = {leaf.topology_id for leaf in leaves}
        if len(topologies) != 1:
            raise ValueError("all water occurrences must share one topology.")

        panels = (base, translation.panel, rotation.panel)
        forces = tuple(
            np.asarray(
                [
                    component.richardson_force_eV_per_angstrom
                    for component in panel.components
                ],
                dtype=float,
            ).reshape(3, 3)
            for panel in panels
        )
        centers = tuple(
            _panel_center_leaf(
                panel, solve_events=solve_events, state_leaves=state_leaves
            ).total_energy_eV
            for panel in panels
        )
        base_forces, translated_forces, rotated_forces = forces
        translation_delta = translated_forces - base_forces
        rotation_matrix = np.asarray(_thaw_json(rotation.rotation_matrix), dtype=float)
        expected_rotated = base_forces @ rotation_matrix.T
        rotation_delta = rotated_forces - expected_rotated
        tiny = np.finfo(float).tiny
        local_errors = [
            component.local_error_eV_per_angstrom
            for panel in panels
            for component in panel.components
        ] + [
            component.local_error_eV_per_angstrom
            for component in closed_loop.components
        ]
        payload = {
            "event_range": occurrence_range,
            "base_panel": base,
            "directional": directional,
            "translation": translation,
            "rotation": rotation,
            "closed_loop": closed_loop,
            "topology_id": next(iter(topologies)),
            "base_forces_eV_per_angstrom": _freeze_json(
                base_forces.tolist(), name="base forces"
            ),
            "translated_forces_eV_per_angstrom": _freeze_json(
                translated_forces.tolist(), name="translated forces"
            ),
            "rotated_forces_eV_per_angstrom": _freeze_json(
                rotated_forces.tolist(), name="rotated forces"
            ),
            "base_center_energy_eV": centers[0],
            "translated_center_energy_eV": centers[1],
            "rotated_center_energy_eV": centers[2],
            "net_force_norm_eV_per_angstrom": float(
                np.linalg.norm(np.sum(base_forces, axis=0))
            ),
            "translation_energy_abs_difference_eV": abs(centers[1] - centers[0]),
            "translation_force_relative_difference": float(
                np.linalg.norm(translation_delta)
                / max(float(np.linalg.norm(base_forces)), tiny)
            ),
            "translation_force_max_abs_difference_eV_per_angstrom": float(
                np.max(np.abs(translation_delta))
            ),
            "rotation_energy_abs_difference_eV": abs(centers[2] - centers[0]),
            "rotation_force_covariance_relative_difference": float(
                np.linalg.norm(rotation_delta)
                / max(float(np.linalg.norm(expected_rotated)), tiny)
            ),
            "rotation_force_covariance_max_abs_difference_eV_per_angstrom": float(
                np.max(np.abs(rotation_delta))
            ),
            "directional_absolute_error_eV_per_angstrom": (
                directional.absolute_error_eV_per_angstrom
            ),
            "maximum_local_error_eV_per_angstrom": max(local_errors),
            "closed_loop_work_eV": closed_loop.closed_loop_work_eV,
            "closed_loop_numerical_error_bound_eV": (
                closed_loop.numerical_work_error_bound_eV
            ),
            "closed_loop_guarded_absolute_work_eV": (
                closed_loop.guarded_absolute_work_eV
            ),
            **_root_metrics(leaves),
        }
        return _sealed_instance(cls, payload)  # type: ignore[return-value]


def _reported_metrics(
    benzene: BenzeneSystemV2, water: WaterSystemV2
) -> dict[str, object]:
    stencil = benzene.predecessor_stencil
    metrics: dict[str, object] = {
        "benzene_local_error_eV_per_angstrom": (
            stencil.component.local_error_eV_per_angstrom
        ),
        "benzene_h4_difference_eV_per_angstrom": (
            stencil.independent_h4_force_difference_eV_per_angstrom
        ),
        "benzene_center_legacy_root_sha256": None,
        "benzene_displaced_legacy_root_sha256s": [],
        "water_maximum_local_error_eV_per_angstrom": (
            water.maximum_local_error_eV_per_angstrom
        ),
        "water_directional_error_eV_per_angstrom": (
            water.directional_absolute_error_eV_per_angstrom
        ),
        "water_net_force_norm_eV_per_angstrom": (water.net_force_norm_eV_per_angstrom),
        "water_translation_energy_error_eV": (
            water.translation_energy_abs_difference_eV
        ),
        "water_translation_force_relative_error": (
            water.translation_force_relative_difference
        ),
        "water_translation_force_max_abs_error_eV_per_angstrom": (
            water.translation_force_max_abs_difference_eV_per_angstrom
        ),
        "water_rotation_energy_error_eV": water.rotation_energy_abs_difference_eV,
        "water_rotation_force_relative_error": (
            water.rotation_force_covariance_relative_difference
        ),
        "water_rotation_force_max_abs_error_eV_per_angstrom": (
            water.rotation_force_covariance_max_abs_difference_eV_per_angstrom
        ),
        "water_closed_loop_work_eV": water.closed_loop_work_eV,
        "water_closed_loop_numerical_error_bound_eV": (
            water.closed_loop_numerical_error_bound_eV
        ),
        "water_closed_loop_guarded_absolute_work_eV": (
            water.closed_loop_guarded_absolute_work_eV
        ),
    }
    for prefix, system in (("benzene", benzene), ("water", water)):
        metrics.update(
            {
                f"{prefix}_maximum_final_residual_norm_eV": (
                    system.maximum_final_residual_norm_eV
                ),
                f"{prefix}_maximum_total_charge_error_e": (
                    system.maximum_total_charge_error_e
                ),
                f"{prefix}_maximum_cross_start_energy_abs_difference_eV": (
                    system.maximum_cross_start_energy_abs_difference_eV
                ),
                f"{prefix}_maximum_cross_start_field_max_abs_difference_eV": (
                    system.maximum_cross_start_field_max_abs_difference_eV
                ),
                f"{prefix}_maximum_cold_iterations": system.maximum_cold_iterations,
                f"{prefix}_maximum_wide_iterations": system.maximum_wide_iterations,
                f"{prefix}_all_starts_converged": system.all_starts_converged,
            }
        )
    return metrics


def _replicate_gates(
    metrics: Mapping[str, object],
    *,
    seal: ComputationSealV2,
    force_panel: Mapping[str, object],
) -> dict[str, bool]:
    from .harmonic_ef_measurement import HARMONIC_EF_REPLICATE_GATE_NAMES

    if tuple(HARMONIC_EF_REPLICATE_GATE_NAMES) != REPLICATE_GATE_NAMES:
        raise ValueError("admission and engine replicate gate names drifted.")
    science = seal.as_dict()["scientific_settings"]
    root = science["root_algorithm"]
    stencil = science["force_stencil"]
    local_threshold = stencil["maximum_local_error_ev_per_angstrom"]
    maximum_iterations = root["maximum_iterations"]
    systems = ("benzene", "water")
    roots_converged = all(
        metrics[f"{prefix}_all_starts_converged"] is True
        and 1 <= metrics[f"{prefix}_maximum_cold_iterations"] <= maximum_iterations
        and 1 <= metrics[f"{prefix}_maximum_wide_iterations"] <= maximum_iterations
        for prefix in systems
    )
    root_metrics_pass = all(
        metrics[f"{prefix}_maximum_final_residual_norm_eV"] < root["tolerance_ev"]
        and metrics[f"{prefix}_maximum_total_charge_error_e"]
        <= root["total_charge_tolerance_e"]
        and metrics[f"{prefix}_maximum_cross_start_field_max_abs_difference_eV"]
        <= root["multi_start_field_tolerance_ev"]
        and metrics[f"{prefix}_maximum_cross_start_energy_abs_difference_eV"]
        <= root["multi_start_energy_tolerance_ev"]
        for prefix in systems
    )
    water_forces_finite = all(
        math.isfinite(float(item))
        for key in (
            "water_net_force_norm_eV_per_angstrom",
            "water_translation_force_relative_error",
            "water_translation_force_max_abs_error_eV_per_angstrom",
            "water_rotation_force_relative_error",
            "water_rotation_force_max_abs_error_eV_per_angstrom",
        )
        for item in (metrics[key],)
    )
    return {
        "clean_content_addressed_execution": seal.git_clean,
        "benzene_predecessor_failure_coordinate_passes_h_h2_h4": bool(
            metrics["benzene_local_error_eV_per_angstrom"] <= local_threshold
            and metrics["benzene_h4_difference_eV_per_angstrom"]
            <= force_panel["maximum_benzene_h4_difference_ev_per_angstrom"]
        ),
        "water_full_cartesian_h_h2_force_is_finite": water_forces_finite,
        "all_local_richardson_error_estimates_below_2e-4_ev_per_angstrom": bool(
            max(
                metrics["benzene_local_error_eV_per_angstrom"],
                metrics["water_maximum_local_error_eV_per_angstrom"],
            )
            <= local_threshold
        ),
        "independent_h4_directional_derivative_below_5e-4_ev_per_angstrom": bool(
            metrics["water_directional_error_eV_per_angstrom"]
            <= force_panel["maximum_independent_directional_error_ev_per_angstrom"]
        ),
        "translation_energy_and_net_force_gates_pass": bool(
            metrics["water_translation_energy_error_eV"]
            <= force_panel["maximum_translation_energy_error_ev"]
            and metrics["water_translation_force_relative_error"]
            <= force_panel["maximum_translation_force_relative_error"]
            and metrics["water_translation_force_max_abs_error_eV_per_angstrom"]
            <= force_panel["maximum_translation_force_absolute_error_ev_per_angstrom"]
            and metrics["water_net_force_norm_eV_per_angstrom"]
            <= force_panel["maximum_net_force_norm_ev_per_angstrom"]
        ),
        "rotation_energy_and_force_covariance_gates_pass": bool(
            metrics["water_rotation_energy_error_eV"]
            <= force_panel["maximum_rotation_energy_error_ev"]
            and metrics["water_rotation_force_relative_error"]
            <= force_panel["maximum_rotation_force_relative_error"]
            and metrics["water_rotation_force_max_abs_error_eV_per_angstrom"]
            <= force_panel["maximum_rotation_force_absolute_error_ev_per_angstrom"]
        ),
        "closed_loop_work_plus_numerical_error_bound_gate_passes": bool(
            metrics["water_closed_loop_guarded_absolute_work_eV"]
            <= force_panel["maximum_closed_loop_work_abs_ev"]
        ),
        "all_roots_converge_from_both_starts": roots_converged,
        "all_final_residuals_and_total_charge_errors_pass": root_metrics_pass,
    }


def _scientific_settings(value: object) -> _FrozenObject:
    raw = _object(value, name="scientific_settings")
    _exact_fields(raw, _SCIENTIFIC_KEYS, name="scientific_settings")

    continuum = _object(raw["continuum"], name="scientific_settings.continuum")
    _exact_fields(continuum, _CONTINUUM_KEYS, name="scientific_settings.continuum")
    continuum_payload = {
        "profile_id": _text(continuum["profile_id"], name="continuum.profile_id"),
        "transition_width_angstrom2": _positive_float(
            continuum["transition_width_angstrom2"],
            name="continuum.transition_width_angstrom2",
        ),
        "surface_lmax": _nonnegative_int(
            continuum["surface_lmax"], name="continuum.surface_lmax"
        ),
        "exposure_lmax": _nonnegative_int(
            continuum["exposure_lmax"], name="continuum.exposure_lmax"
        ),
        "exposure_radial_quadrature_order": _positive_int(
            continuum["exposure_radial_quadrature_order"],
            name="continuum.exposure_radial_quadrature_order",
        ),
        "source_radial_quadrature_order": _positive_int(
            continuum["source_radial_quadrature_order"],
            name="continuum.source_radial_quadrature_order",
        ),
        "green_radial_quadrature_order": _positive_int(
            continuum["green_radial_quadrature_order"],
            name="continuum.green_radial_quadrature_order",
        ),
    }

    cavity = _object(raw["cavity"], name="scientific_settings.cavity")
    _exact_fields(cavity, _CAVITY_KEYS, name="scientific_settings.cavity")
    cavity_payload = {
        key: _text(cavity[key], name=f"cavity.{key}") for key in sorted(_CAVITY_KEYS)
    }

    source = _object(raw["source_receiver"], name="scientific_settings.source_receiver")
    _exact_fields(
        source, _SOURCE_RECEIVER_KEYS, name="scientific_settings.source_receiver"
    )
    source_payload = {
        key: _text(source[key], name=f"source_receiver.{key}")
        for key in sorted(_SOURCE_RECEIVER_KEYS)
    }

    ledger = _object(raw["energy_ledger"], name="scientific_settings.energy_ledger")
    _exact_fields(ledger, _ENERGY_LEDGER_KEYS, name="scientific_settings.energy_ledger")
    included = _components(ledger["included_components"], name="included_components")
    excluded = _components(ledger["excluded_components"], name="excluded_components")
    if set(included) & set(excluded):
        raise ValueError("included/excluded energy components must be disjoint.")
    ledger_payload = {
        "exact_scalar": _text(
            ledger["exact_scalar"], name="energy_ledger.exact_scalar"
        ),
        "included_components": list(included),
        "excluded_components": list(excluded),
    }

    root = _object(raw["root_algorithm"], name="scientific_settings.root_algorithm")
    _exact_fields(root, _ROOT_ALGORITHM_KEYS, name="scientific_settings.root_algorithm")
    root_payload = {
        "method": _text(root["method"], name="root_algorithm.method"),
        "tolerance_ev": _positive_float(
            root["tolerance_ev"], name="root_algorithm.tolerance_ev"
        ),
        "maximum_iterations": _positive_int(
            root["maximum_iterations"], name="root_algorithm.maximum_iterations"
        ),
        "second_start": _text(root["second_start"], name="root_algorithm.second_start"),
        "total_charge_tolerance_e": _positive_float(
            root["total_charge_tolerance_e"],
            name="root_algorithm.total_charge_tolerance_e",
        ),
        "multi_start_field_tolerance_ev": _positive_float(
            root["multi_start_field_tolerance_ev"],
            name="root_algorithm.multi_start_field_tolerance_ev",
        ),
        "multi_start_energy_tolerance_ev": _positive_float(
            root["multi_start_energy_tolerance_ev"],
            name="root_algorithm.multi_start_energy_tolerance_ev",
        ),
        "evidence_scope": _text(
            root["evidence_scope"], name="root_algorithm.evidence_scope"
        ),
    }

    stencil = _object(raw["force_stencil"], name="scientific_settings.force_stencil")
    _exact_fields(
        stencil, _FORCE_STENCIL_KEYS, name="scientific_settings.force_stencil"
    )
    stencil_payload = {
        "derivative": _text(stencil["derivative"], name="force_stencil.derivative"),
        "coarse_step_angstrom": _positive_float(
            stencil["coarse_step_angstrom"], name="force_stencil.coarse_step_angstrom"
        ),
        "fine_step_angstrom": _positive_float(
            stencil["fine_step_angstrom"], name="force_stencil.fine_step_angstrom"
        ),
        "independent_step_angstrom": _positive_float(
            stencil["independent_step_angstrom"],
            name="force_stencil.independent_step_angstrom",
        ),
        "maximum_local_error_ev_per_angstrom": _positive_float(
            stencil["maximum_local_error_ev_per_angstrom"],
            name="force_stencil.maximum_local_error_ev_per_angstrom",
        ),
        "topology_policy": _text(
            stencil["topology_policy"], name="force_stencil.topology_policy"
        ),
        "displacement_policy": _text(
            stencil["displacement_policy"], name="force_stencil.displacement_policy"
        ),
    }
    expected = {
        "continuum": {
            "profile_id": H1_V2_CONTINUUM_PROFILE_ID,
            "transition_width_angstrom2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "source_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
        },
        "cavity": {
            "profile_id": H1_V2_CAVITY_PROFILE_ID,
            "radii_provider_id": H1_V2_RADII_PROVIDER_ID,
        },
        "source_receiver": {
            "permanent_source_kernel": H1_V2_PERMANENT_SOURCE_KERNEL,
            "induced_source_kernel": H1_V2_INDUCED_SOURCE_KERNEL,
            "receiver_kernel": H1_V2_RECEIVER_KERNEL,
            "long_range_evaluator_id": H1_V2_LONG_RANGE_EVALUATOR_ID,
            "source_space_id": H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
            "source_space_contract_sha256": H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            "receiver_space_id": H1_V2_RECEIVER_SPACE_ID,
            "receiver_space_contract_sha256": H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
        },
        "energy_ledger": {
            "exact_scalar": H1_V2_EXACT_SCALAR,
            "included_components": list(H1_V2_INCLUDED_COMPONENTS),
            "excluded_components": list(H1_V2_EXCLUDED_COMPONENTS),
        },
        "root_algorithm": {
            "method": H1_V2_ROOT_METHOD,
            "tolerance_ev": 1.0e-10,
            "maximum_iterations": 40,
            "second_start": H1_V2_SECOND_START,
            "total_charge_tolerance_e": 1.0e-8,
            "multi_start_field_tolerance_ev": 2.0e-9,
            "multi_start_energy_tolerance_ev": 1.0e-10,
            "evidence_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
        },
        "force_stencil": {
            "derivative": H1_V2_FORCE_DERIVATIVE,
            "coarse_step_angstrom": 5.0e-4,
            "fine_step_angstrom": 2.5e-4,
            "independent_step_angstrom": 1.25e-4,
            "maximum_local_error_ev_per_angstrom": 2.0e-4,
            "topology_policy": H1_V2_TOPOLOGY_POLICY,
            "displacement_policy": H1_V2_DISPLACEMENT_POLICY,
        },
    }
    normalized = {
        "continuum": continuum_payload,
        "cavity": cavity_payload,
        "source_receiver": source_payload,
        "energy_ledger": ledger_payload,
        "root_algorithm": root_payload,
        "force_stencil": stencil_payload,
    }
    coarse = stencil_payload["coarse_step_angstrom"]
    fine = stencil_payload["fine_step_angstrom"]
    independent = stencil_payload["independent_step_angstrom"]
    if not (
        coarse > fine > independent > 0.0
        and fine == coarse / 2.0
        and independent == fine / 2.0
    ):
        raise ValueError("force stencil steps must be exact nested halvings.")
    if normalized != expected:
        raise ValueError("scientific_settings differ from the exact H1 v2 contract.")
    return _freeze_object(
        normalized,
        name="scientific_settings",
    )


def _runtime_fingerprint(value: object) -> _FrozenObject:
    raw = _object(value, name="runtime_fingerprint")
    _exact_fields(raw, _RUNTIME_KEYS, name="runtime_fingerprint")
    packages = _object(raw["packages"], name="runtime_fingerprint.packages")
    _exact_fields(packages, _PACKAGE_KEYS, name="runtime_fingerprint.packages")
    package_payload = {
        key: _optional_text(packages[key], name=f"packages.{key}")
        for key in sorted(_PACKAGE_KEYS)
    }
    numpy = _object(raw["numpy"], name="runtime_fingerprint.numpy")
    _exact_fields(numpy, _NUMPY_KEYS, name="runtime_fingerprint.numpy")
    numpy_payload = {
        "version": _text(numpy["version"], name="numpy.version"),
        "show_config": _text(numpy["show_config"], name="numpy.show_config"),
    }
    torch = _object(raw["torch"], name="runtime_fingerprint.torch")
    _exact_fields(torch, _TORCH_KEYS, name="runtime_fingerprint.torch")
    devices = torch["devices"]
    if not isinstance(devices, list):
        raise TypeError("torch.devices must be a JSON array.")
    device_names = tuple(_text(item, name="torch.devices item") for item in devices)
    if not device_names or len(set(device_names)) != len(device_names):
        raise ValueError("torch.devices must be nonempty and duplicate-free.")
    raw_uuids = torch["device_uuids"]
    raw_capabilities = torch["device_capabilities"]
    raw_sm_counts = torch["device_multiprocessor_counts"]
    for field, items in (
        ("device_uuids", raw_uuids),
        ("device_capabilities", raw_capabilities),
        ("device_multiprocessor_counts", raw_sm_counts),
    ):
        if not isinstance(items, list):
            raise TypeError(f"torch.{field} must be a JSON array.")
    device_uuids = tuple(
        _device_uuid(item, name="torch.device_uuids item") for item in raw_uuids
    )
    if not device_uuids or len(set(device_uuids)) != len(device_uuids):
        raise ValueError("torch.device_uuids must be nonempty and duplicate-free.")
    device_capabilities = tuple(
        _text(item, name="torch.device_capabilities item") for item in raw_capabilities
    )
    if any(
        re.fullmatch(r"[0-9]+\.[0-9]+", item) is None for item in device_capabilities
    ):
        raise ValueError("torch.device_capabilities must use '<major>.<minor>'.")
    device_sm_counts = tuple(
        _positive_int(item, name="torch.device_multiprocessor_counts item")
        for item in raw_sm_counts
    )
    device_count = len(device_names)
    if not (
        len(device_uuids)
        == len(device_capabilities)
        == len(device_sm_counts)
        == device_count
    ):
        raise ValueError("all torch device identity lists must have the same length.")
    if torch["cuda_available"] is not True:
        raise ValueError("runtime requires cuda_available=true.")
    if torch["deterministic_algorithms_enabled"] is not True:
        raise ValueError("runtime requires deterministic algorithms.")
    debug_mode = torch["deterministic_debug_mode"]
    if debug_mode != 2 or type(debug_mode) is not int:
        raise ValueError("runtime deterministic_debug_mode must be 2 (error mode).")
    if torch["cudnn_benchmark"] is not False:
        raise ValueError("runtime requires cudnn_benchmark=false.")
    if torch["cudnn_deterministic"] is not True:
        raise ValueError("runtime requires cudnn_deterministic=true.")
    if torch["default_dtype"] != "torch.float64":
        raise ValueError("runtime requires torch.float64 default_dtype.")
    torch_payload = {
        "version": _text(torch["version"], name="torch.version"),
        "cuda_version": _text(torch["cuda_version"], name="torch.cuda_version"),
        "cudnn_version": _positive_int(
            torch["cudnn_version"], name="torch.cudnn_version"
        ),
        "cuda_available": True,
        "devices": list(device_names),
        "default_dtype": "torch.float64",
        "threads": _positive_int(torch["threads"], name="torch.threads"),
        "interop_threads": _positive_int(
            torch["interop_threads"], name="torch.interop_threads"
        ),
        "driver_version": _text(torch["driver_version"], name="torch.driver_version"),
        "device_uuids": list(device_uuids),
        "device_capabilities": list(device_capabilities),
        "device_multiprocessor_counts": list(device_sm_counts),
        "deterministic_algorithms_enabled": True,
        "deterministic_debug_mode": debug_mode,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    environment = _object(raw["environment"], name="runtime_fingerprint.environment")
    _exact_fields(
        environment, _ENVIRONMENT_KEYS, name="runtime_fingerprint.environment"
    )
    environment_payload = {
        key: _optional_text(environment[key], name=f"environment.{key}")
        for key in sorted(_ENVIRONMENT_KEYS)
    }
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if environment_payload[key] != "1":
            raise ValueError(f"environment.{key} must equal '1'.")
    if torch_payload["threads"] != 1:
        raise ValueError("runtime torch.threads must equal 1.")
    if torch_payload["interop_threads"] != 1:
        raise ValueError("runtime torch.interop_threads must equal 1.")
    if environment_payload["CUDA_VISIBLE_DEVICES"] != "0":
        raise ValueError("environment.CUDA_VISIBLE_DEVICES must equal '0'.")
    seed = environment_payload["PYTHONHASHSEED"]
    if seed is None or re.fullmatch(r"[0-9]+", seed) is None:
        raise ValueError("environment.PYTHONHASHSEED must be a decimal integer string.")
    if environment_payload["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise ValueError("environment.CUBLAS_WORKSPACE_CONFIG is not deterministic.")
    if raw["execution_device"] != "cuda":
        raise ValueError("runtime execution_device must be cuda.")
    if raw["execution_dtype"] != "float64":
        raise ValueError("runtime execution_dtype must be float64.")
    if package_payload["numpy"] != numpy_payload["version"]:
        raise ValueError("packages.numpy must equal numpy.version.")
    if package_payload["torch"] != torch_payload["version"]:
        raise ValueError("packages.torch must equal torch.version.")
    return _freeze_object(
        {
            "python": _text(raw["python"], name="runtime.python"),
            "implementation": _text(
                raw["implementation"], name="runtime.implementation"
            ),
            "platform": _text(raw["platform"], name="runtime.platform"),
            "machine": _text(raw["machine"], name="runtime.machine"),
            "cpu_model": _text(raw["cpu_model"], name="runtime.cpu_model"),
            "packages": package_payload,
            "numpy": numpy_payload,
            "torch": torch_payload,
            "environment": environment_payload,
            "execution_device": "cuda",
            "execution_dtype": "float64",
        },
        name="runtime_fingerprint",
    )


def _capabilities(value: object) -> CapabilityStatus:
    raw = _object(value, name="capabilities")
    _exact_fields(raw, _CAPABILITY_KEYS, name="capabilities")
    for tier in sorted(_CAPABILITY_KEYS):
        _strict_bool(raw[tier], name=f"capabilities.{tier}")
    status = CapabilityStatus(
        energy=raw["E"],
        conservative_force=raw["F"],
        hessian=raw["H"],
        variational_functional=raw["V"],
        molecular_dynamics=raw["M"],
    )
    if status != _EF_CAPABILITIES:
        raise ValueError("capabilities must admit exactly E/F and deny H/V/M.")
    return status


def _capability_payload(status: CapabilityStatus) -> dict[str, bool]:
    return {
        "E": status.energy,
        "F": status.conservative_force,
        "H": status.hessian,
        "V": status.variational_functional,
        "M": status.molecular_dynamics,
    }


def _admission_claim_values(raw: Mapping[str, object]) -> dict[str, object]:
    return {
        "profile_id": _text(raw["profile_id"], name="profile_id"),
        "scalar_id": _text(raw["scalar_id"], name="scalar_id"),
        "state_id": _text(raw["state_id"], name="state_id"),
        "capabilities": _capabilities(raw["capabilities"]),
        "runtime_guards": _exact_set(
            raw["runtime_guards"], REQUIRED_RUNTIME_GUARDS, name="runtime_guards"
        ),
        "domain_guards": _guards(raw["domain_guards"], name="domain_guards"),
        "claim_boundary_id": _text(raw["claim_boundary_id"], name="claim_boundary_id"),
        "non_admissions": _exact_tuple(
            raw["non_admissions"], REQUIRED_NON_ADMISSIONS, name="non_admissions"
        ),
    }


def _validate_cross_bindings(
    contract: object, expected: Mapping[str, object], *, name: str
) -> None:
    mismatches = tuple(
        field
        for field, expected_value in expected.items()
        if getattr(contract, field) != expected_value
    )
    if mismatches:
        raise ValueError(
            f"{name} cross-binding mismatch: " + ", ".join(mismatches) + "."
        )


def _sealed_instance(cls: type[_Contract], values: Mapping[str, object]) -> _Contract:
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _uuid(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    try:
        parsed = UUID(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UUID.") from error
    if str(parsed) != text:
        raise ValueError(f"{name} must be a canonical lowercase UUID.")
    return text


def _device_uuid(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    prefix = "GPU-"
    suffix = text[len(prefix) :] if text.startswith(prefix) else text
    try:
        parsed = UUID(suffix)
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical GPU or bare UUID.") from error
    canonical = prefix + str(parsed) if text.startswith(prefix) else str(parsed)
    if text != canonical:
        raise ValueError(f"{name} must be a canonical GPU or bare UUID.")
    return text


def _utc_text(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be valid ISO UTC text.") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise ValueError(f"{name} must include a UTC offset.")
    return text


_SEAL_FIELDS = frozenset(
    {
        "schema_id",
        "artifact_id",
        "profile_id",
        "scalar_id",
        "state_id",
        "git_head",
        "git_tree",
        "git_clean",
        "preregistration_id",
        "preregistration_sha256",
        "runner_id",
        "runner_sha256",
        "aggregator_id",
        "aggregator_sha256",
        "computation_source_sha256s",
        "asset_sha256s",
        "checkpoint_sha256s",
        "provider_configuration_sha256s",
        "scientific_settings",
        "runtime_fingerprint",
        "runtime_guards",
        "domain_guards",
        "exposure_aware_protocol_label",
        "claim_boundary_id",
        "non_admissions",
        "content_sha256",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class ComputationSealV2:
    schema_id: str
    artifact_id: str
    profile_id: str
    scalar_id: str
    state_id: str
    git_head: str
    git_tree: str
    git_clean: bool
    preregistration_id: str
    preregistration_sha256: str
    runner_id: str
    runner_sha256: str
    aggregator_id: str
    aggregator_sha256: str
    computation_source_sha256s: tuple[tuple[str, str], ...]
    asset_sha256s: tuple[tuple[str, str], ...]
    checkpoint_sha256s: tuple[tuple[str, str], ...]
    provider_configuration_sha256s: tuple[tuple[str, str], ...]
    scientific_settings: _FrozenObject
    runtime_fingerprint: _FrozenObject
    runtime_guards: tuple[str, ...]
    domain_guards: tuple[str, ...]
    exposure_aware_protocol_label: str
    claim_boundary_id: str
    non_admissions: tuple[str, ...]
    content_sha256: str

    def __init__(self) -> None:
        raise TypeError("ComputationSealV2 must be loaded with from_mapping().")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ComputationSealV2":
        raw = _object(value, name="computation seal")
        _exact_fields(raw, _SEAL_FIELDS, name="computation seal")
        if raw["schema_id"] != COMPUTATION_SEAL_V2_SCHEMA:
            raise ValueError("computation seal schema_id is unsupported.")
        if raw["git_clean"] is not True:
            raise ValueError("computation seal requires an explicitly clean Git tree.")
        if raw["claim_boundary_id"] != CLAIM_BOUNDARY_ID:
            raise ValueError("computation seal claim_boundary_id is unsupported.")
        if raw["exposure_aware_protocol_label"] != EXPOSURE_AWARE_PROTOCOL_LABEL:
            raise ValueError(
                "computation seal exposure-aware protocol label is invalid."
            )
        for field, expected in (
            ("profile_id", H1_V2_PROFILE_ID),
            ("scalar_id", H1_V2_SCALAR_ID),
            ("state_id", H1_V2_STATE_ID),
        ):
            if raw[field] != expected:
                raise ValueError(f"computation seal {field} is not the H1 v2 ID.")
        values: dict[str, object] = {
            "schema_id": COMPUTATION_SEAL_V2_SCHEMA,
            "artifact_id": _text(raw["artifact_id"], name="artifact_id"),
            "profile_id": H1_V2_PROFILE_ID,
            "scalar_id": H1_V2_SCALAR_ID,
            "state_id": H1_V2_STATE_ID,
            "git_head": _git_object(raw["git_head"], name="git_head"),
            "git_tree": _git_object(raw["git_tree"], name="git_tree"),
            "git_clean": True,
            "preregistration_id": _text(
                raw["preregistration_id"], name="preregistration_id"
            ),
            "preregistration_sha256": _sha256(
                raw["preregistration_sha256"], name="preregistration_sha256"
            ),
            "runner_id": _text(raw["runner_id"], name="runner_id"),
            "runner_sha256": _sha256(raw["runner_sha256"], name="runner_sha256"),
            "aggregator_id": _text(raw["aggregator_id"], name="aggregator_id"),
            "aggregator_sha256": _sha256(
                raw["aggregator_sha256"], name="aggregator_sha256"
            ),
            "computation_source_sha256s": _sha_ledger(
                raw["computation_source_sha256s"],
                name="computation_source_sha256s",
                key_validator=_repo_relative_path,
            ),
            "asset_sha256s": _sha_ledger(
                raw["asset_sha256s"],
                name="asset_sha256s",
                key_validator=_stable_label,
                exact_keys=frozenset(REQUIRED_ASSET_KEYS),
            ),
            "checkpoint_sha256s": _sha_ledger(
                raw["checkpoint_sha256s"],
                name="checkpoint_sha256s",
                key_validator=_stable_label,
                exact_keys=frozenset(REQUIRED_CHECKPOINT_KEYS),
            ),
            "provider_configuration_sha256s": _sha_ledger(
                raw["provider_configuration_sha256s"],
                name="provider_configuration_sha256s",
                key_validator=_stable_label,
                exact_keys=_PROVIDER_CONFIGURATION_KEYS,
            ),
            "scientific_settings": _scientific_settings(raw["scientific_settings"]),
            "runtime_fingerprint": _runtime_fingerprint(raw["runtime_fingerprint"]),
            "runtime_guards": _exact_set(
                raw["runtime_guards"], REQUIRED_RUNTIME_GUARDS, name="runtime_guards"
            ),
            "domain_guards": _exact_set(
                raw["domain_guards"],
                REQUIRED_DOMAIN_GUARDS,
                name="domain_guards",
            ),
            "exposure_aware_protocol_label": EXPOSURE_AWARE_PROTOCOL_LABEL,
            "claim_boundary_id": CLAIM_BOUNDARY_ID,
            "non_admissions": _exact_tuple(
                raw["non_admissions"], REQUIRED_NON_ADMISSIONS, name="non_admissions"
            ),
            "content_sha256": _sha256(raw["content_sha256"], name="content_sha256"),
        }
        seal = _sealed_instance(cls, values)
        if seal.content_sha256 != canonical_json_sha256(seal._content_payload()):
            raise ValueError("computation seal content_sha256 does not match content.")
        return seal

    @property
    def source_ledger_sha256(self) -> str:
        return canonical_json_sha256(dict(self.computation_source_sha256s))

    @property
    def asset_ledger_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "assets": dict(self.asset_sha256s),
                "checkpoints": dict(self.checkpoint_sha256s),
            }
        )

    @property
    def runtime_fingerprint_sha256(self) -> str:
        return canonical_json_sha256(_thaw_json(self.runtime_fingerprint))

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "artifact_id": self.artifact_id,
            "profile_id": self.profile_id,
            "scalar_id": self.scalar_id,
            "state_id": self.state_id,
            "git_head": self.git_head,
            "git_tree": self.git_tree,
            "git_clean": self.git_clean,
            "preregistration_id": self.preregistration_id,
            "preregistration_sha256": self.preregistration_sha256,
            "runner_id": self.runner_id,
            "runner_sha256": self.runner_sha256,
            "aggregator_id": self.aggregator_id,
            "aggregator_sha256": self.aggregator_sha256,
            "computation_source_sha256s": dict(self.computation_source_sha256s),
            "asset_sha256s": dict(self.asset_sha256s),
            "checkpoint_sha256s": dict(self.checkpoint_sha256s),
            "provider_configuration_sha256s": dict(self.provider_configuration_sha256s),
            "scientific_settings": _thaw_json(self.scientific_settings),
            "runtime_fingerprint": _thaw_json(self.runtime_fingerprint),
            "runtime_guards": list(self.runtime_guards),
            "domain_guards": list(self.domain_guards),
            "exposure_aware_protocol_label": self.exposure_aware_protocol_label,
            "claim_boundary_id": self.claim_boundary_id,
            "non_admissions": list(self.non_admissions),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_sha256": self.content_sha256}


_REPLICATE_FIELDS = frozenset(
    {
        "schema_id",
        "artifact_id",
        "artifact_sha256",
        "label",
        "process_uuid",
        "process_started_at_utc",
        "seal_id",
        "seal_sha256",
        "label",
        "git_head",
        "git_tree",
        "source_ledger_sha256",
        "asset_ledger_sha256",
        "runtime_fingerprint_sha256",
        "measurement_schema_id",
        "measurement_sha256",
        "measurement",
        "gate_results",
    }
)
_RICH_MEASUREMENT_FIELDS = frozenset(
    {
        "schema_id",
        "seal_id",
        "seal_sha256",
        "profile_id",
        "scalar_id",
        "state_id",
        "protocol",
        "panel_contract_sha256",
        "provider_ledger_sha256",
        "prepared_inputs",
        "prepared_providers",
        "state_leaves",
        "solve_events",
        "systems",
        "reported_metrics",
        "reported_local_gate_results",
        "local_decision",
    }
)
_MEASUREMENT_PROTOCOL_FIELDS = frozenset(
    {
        "preregistration_id",
        "preregistration_sha256",
        "asset_ledger_sha256",
        "scientific_settings_sha256",
        "fit_calibration_or_case_selection",
    }
)
_MEASUREMENT_DECISION_FIELDS = frozenset(
    {
        "candidate_energy_force_gate_passed",
        "awaiting_independent_replay",
        "public_capability_admitted",
        "chemical_accuracy_admitted",
        "complete_solvation_free_energy_admitted",
        "analytic_force_admitted",
        "hessian_frequency_md_admitted",
        "tier_v_admitted",
    }
)
_LOCAL_REPLICATE_DECISION: dict[str, bool] = {
    "candidate_energy_force_gate_passed": True,
    "awaiting_independent_replay": True,
    "public_capability_admitted": False,
    "chemical_accuracy_admitted": False,
    "complete_solvation_free_energy_admitted": False,
    "analytic_force_admitted": False,
    "hessian_frequency_md_admitted": False,
    "tier_v_admitted": False,
}
_PREPARED_INPUT_FIELDS = frozenset(
    {
        "benzene",
        "water",
        "v1_preregistration_sha256",
        "parent_panel_sha256",
        "force_panel_contract",
        "force_panel_contract_sha256",
    }
)
_PREPARED_BENZENE_FIELDS = frozenset(
    {
        "compound_id",
        "name",
        "atomic_numbers",
        "positions_angstrom",
        "charge",
        "multiplicity",
        "cavity_radii_angstrom",
        "mol2_sha256",
        "projection_result_sha256",
        "predecessor_dof",
    }
)
_PREPARED_WATER_FIELDS = frozenset(
    {
        "atomic_numbers",
        "positions_angstrom",
        "charge",
        "multiplicity",
        "cavity_radii_angstrom",
    }
)
_PREPARED_PROVIDER_FIELDS = frozenset(
    {
        "profile_id",
        "scalar_id",
        "state_id",
        "checkpoint_sha256s",
        "provider_configuration_sha256s",
    }
)
_FROZEN_WATER_GEOMETRY_ANGSTROM = (
    (0.0, 0.0, 0.0),
    (0.9572, 0.0, 0.0),
    (-0.239, 0.9266, 0.0),
)
_FORCE_PANEL_FIELDS = frozenset(
    {
        "gepol_regression_compound_id",
        "gepol_regression_cartesian_dof",
        "water_geometry_angstrom",
        "water_full_cartesian_force",
        "independent_direction_seed",
        "maximum_independent_directional_error_ev_per_angstrom",
        "maximum_benzene_h4_difference_ev_per_angstrom",
        "translation_angstrom",
        "maximum_translation_energy_error_ev",
        "maximum_translation_force_relative_error",
        "maximum_translation_force_absolute_error_ev_per_angstrom",
        "maximum_net_force_norm_ev_per_angstrom",
        "rotation_seed",
        "maximum_rotation_energy_error_ev",
        "maximum_rotation_force_relative_error",
        "maximum_rotation_force_absolute_error_ev_per_angstrom",
        "closed_loop_cartesian_dofs",
        "closed_loop_half_width_angstrom",
        "maximum_closed_loop_work_abs_ev",
    }
)
_FORCE_PANEL_POSITIVE_VALUES = {
    "maximum_independent_directional_error_ev_per_angstrom": 5.0e-4,
    "maximum_benzene_h4_difference_ev_per_angstrom": 2.0e-4,
    "maximum_translation_energy_error_ev": 1.0e-7,
    "maximum_translation_force_relative_error": 2.0e-5,
    "maximum_translation_force_absolute_error_ev_per_angstrom": 5.0e-5,
    "maximum_net_force_norm_ev_per_angstrom": 1.0e-3,
    "maximum_rotation_energy_error_ev": 1.0e-7,
    "maximum_rotation_force_relative_error": 2.0e-5,
    "maximum_rotation_force_absolute_error_ev_per_angstrom": 5.0e-5,
    "closed_loop_half_width_angstrom": 1.0e-3,
    "maximum_closed_loop_work_abs_ev": 2.0e-7,
}


def _strict_integer_array(
    value: object, *, length: int, name: str, positive: bool = False
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a JSON array of length {length}.")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or (positive and item <= 0):
            qualifier = " positive" if positive else ""
            raise TypeError(f"{name}[{index}] must be a{qualifier} JSON integer.")
        result.append(item)
    return tuple(result)


def _positive_number_array(
    value: object, *, length: int, name: str
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a JSON array of length {length}.")
    return tuple(
        _positive_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _force_panel_contract(value: object) -> dict[str, object]:
    raw = _object(value, name="measurement.prepared_inputs.force_panel_contract")
    _exact_fields(
        raw,
        _FORCE_PANEL_FIELDS,
        name="measurement.prepared_inputs.force_panel_contract",
    )
    compound = _text(
        raw["gepol_regression_compound_id"],
        name="force_panel.gepol_regression_compound_id",
    )
    predecessor = _strict_integer_array(
        raw["gepol_regression_cartesian_dof"],
        length=2,
        name="force_panel.gepol_regression_cartesian_dof",
    )
    water_geometry = _finite_matrix(
        raw["water_geometry_angstrom"],
        rows=3,
        columns=3,
        name="force_panel.water_geometry_angstrom",
    )
    full_cartesian = _strict_bool(
        raw["water_full_cartesian_force"],
        name="force_panel.water_full_cartesian_force",
    )
    direction_seed = _nonnegative_int(
        raw["independent_direction_seed"], name="force_panel.independent_direction_seed"
    )
    rotation_seed = _nonnegative_int(
        raw["rotation_seed"], name="force_panel.rotation_seed"
    )
    translation = tuple(
        _finite_number(item, name="force_panel.translation_angstrom item")
        for item in _strict_json_array(
            raw["translation_angstrom"], length=3, name="translation_angstrom"
        )
    )
    raw_loop_dofs = _strict_json_array(
        raw["closed_loop_cartesian_dofs"],
        length=2,
        name="closed_loop_cartesian_dofs",
    )
    loop_dofs = tuple(
        _strict_integer_array(
            item, length=2, name="force_panel.closed_loop_cartesian_dof"
        )
        for item in raw_loop_dofs
    )
    positive_values = {
        field: _positive_float(raw[field], name=f"force_panel.{field}")
        for field in _FORCE_PANEL_POSITIVE_VALUES
    }
    normalized = {
        "gepol_regression_compound_id": compound,
        "gepol_regression_cartesian_dof": list(predecessor),
        "water_geometry_angstrom": [list(row) for row in water_geometry],
        "water_full_cartesian_force": full_cartesian,
        "independent_direction_seed": direction_seed,
        **{
            field: positive_values[field]
            for field in (
                "maximum_independent_directional_error_ev_per_angstrom",
                "maximum_benzene_h4_difference_ev_per_angstrom",
            )
        },
        "translation_angstrom": list(translation),
        **{
            field: positive_values[field]
            for field in (
                "maximum_translation_energy_error_ev",
                "maximum_translation_force_relative_error",
                "maximum_translation_force_absolute_error_ev_per_angstrom",
                "maximum_net_force_norm_ev_per_angstrom",
            )
        },
        "rotation_seed": rotation_seed,
        **{
            field: positive_values[field]
            for field in (
                "maximum_rotation_energy_error_ev",
                "maximum_rotation_force_relative_error",
                "maximum_rotation_force_absolute_error_ev_per_angstrom",
            )
        },
        "closed_loop_cartesian_dofs": [list(item) for item in loop_dofs],
        "closed_loop_half_width_angstrom": positive_values[
            "closed_loop_half_width_angstrom"
        ],
        "maximum_closed_loop_work_abs_ev": positive_values[
            "maximum_closed_loop_work_abs_ev"
        ],
    }
    expected = {
        "gepol_regression_compound_id": "mobley_3053621",
        "gepol_regression_cartesian_dof": [0, 0],
        "water_geometry_angstrom": [
            list(row) for row in _FROZEN_WATER_GEOMETRY_ANGSTROM
        ],
        "water_full_cartesian_force": True,
        "independent_direction_seed": 20260816,
        "maximum_independent_directional_error_ev_per_angstrom": 5.0e-4,
        "maximum_benzene_h4_difference_ev_per_angstrom": 2.0e-4,
        "translation_angstrom": [4.2, -3.1, 1.7],
        "maximum_translation_energy_error_ev": 1.0e-7,
        "maximum_translation_force_relative_error": 2.0e-5,
        "maximum_translation_force_absolute_error_ev_per_angstrom": 5.0e-5,
        "maximum_net_force_norm_ev_per_angstrom": 1.0e-3,
        "rotation_seed": 20260817,
        "maximum_rotation_energy_error_ev": 1.0e-7,
        "maximum_rotation_force_relative_error": 2.0e-5,
        "maximum_rotation_force_absolute_error_ev_per_angstrom": 5.0e-5,
        "closed_loop_cartesian_dofs": [[0, 0], [1, 1]],
        "closed_loop_half_width_angstrom": 1.0e-3,
        "maximum_closed_loop_work_abs_ev": 2.0e-7,
    }
    if normalized != expected:
        raise ValueError(
            "force_panel_contract differs from the frozen preregistration."
        )
    return normalized


def _prepared_inputs(
    value: object,
    *,
    seal: ComputationSealV2,
    panel_contract_sha256: str,
) -> _FrozenObject:
    raw = _object(value, name="measurement.prepared_inputs")
    _exact_fields(raw, _PREPARED_INPUT_FIELDS, name="measurement.prepared_inputs")
    benzene = _object(raw["benzene"], name="measurement.prepared_inputs.benzene")
    water = _object(raw["water"], name="measurement.prepared_inputs.water")
    _exact_fields(
        benzene,
        _PREPARED_BENZENE_FIELDS,
        name="measurement.prepared_inputs.benzene",
    )
    _exact_fields(
        water,
        _PREPARED_WATER_FIELDS,
        name="measurement.prepared_inputs.water",
    )

    compound_id = _text(
        benzene["compound_id"],
        name="measurement.prepared_inputs.benzene.compound_id",
    )
    benzene_name = _text(
        benzene["name"], name="measurement.prepared_inputs.benzene.name"
    )
    if compound_id != "mobley_3053621" or benzene_name != "benzene":
        raise ValueError("prepared benzene compound identity is not frozen benzene.")
    benzene_numbers = _strict_integer_array(
        benzene["atomic_numbers"],
        length=12,
        name="measurement.prepared_inputs.benzene.atomic_numbers",
        positive=True,
    )
    benzene_positions = _finite_matrix(
        benzene["positions_angstrom"],
        rows=12,
        columns=3,
        name="measurement.prepared_inputs.benzene.positions_angstrom",
    )
    if type(benzene["charge"]) is not int or benzene["charge"] != 0:
        raise ValueError("prepared benzene charge must be the JSON integer 0.")
    if type(benzene["multiplicity"]) is not int or benzene["multiplicity"] != 1:
        raise ValueError("prepared benzene multiplicity must be the JSON integer 1.")
    benzene_radii = _positive_number_array(
        benzene["cavity_radii_angstrom"],
        length=12,
        name="measurement.prepared_inputs.benzene.cavity_radii_angstrom",
    )
    mol2_sha256 = _sha256(
        benzene["mol2_sha256"],
        name="measurement.prepared_inputs.benzene.mol2_sha256",
    )
    projection_sha256 = _sha256(
        benzene["projection_result_sha256"],
        name="measurement.prepared_inputs.benzene.projection_result_sha256",
    )
    predecessor_dof = _strict_integer_array(
        benzene["predecessor_dof"],
        length=2,
        name="measurement.prepared_inputs.benzene.predecessor_dof",
    )
    if predecessor_dof != (0, 0):
        raise ValueError("prepared benzene predecessor_dof must equal [0, 0].")

    water_numbers = _strict_integer_array(
        water["atomic_numbers"],
        length=3,
        name="measurement.prepared_inputs.water.atomic_numbers",
        positive=True,
    )
    if water_numbers != (8, 1, 1):
        raise ValueError("prepared water atomic_numbers must equal [8, 1, 1].")
    water_positions = _finite_matrix(
        water["positions_angstrom"],
        rows=3,
        columns=3,
        name="measurement.prepared_inputs.water.positions_angstrom",
    )
    if water_positions != _FROZEN_WATER_GEOMETRY_ANGSTROM:
        raise ValueError("prepared water geometry drifted from the frozen panel.")
    if type(water["charge"]) is not int or water["charge"] != 0:
        raise ValueError("prepared water charge must be the JSON integer 0.")
    if type(water["multiplicity"]) is not int or water["multiplicity"] != 1:
        raise ValueError("prepared water multiplicity must be the JSON integer 1.")
    water_radii = _positive_number_array(
        water["cavity_radii_angstrom"],
        length=3,
        name="measurement.prepared_inputs.water.cavity_radii_angstrom",
    )

    v1_sha256 = _sha256(
        raw["v1_preregistration_sha256"],
        name="measurement.prepared_inputs.v1_preregistration_sha256",
    )
    parent_sha256 = _sha256(
        raw["parent_panel_sha256"],
        name="measurement.prepared_inputs.parent_panel_sha256",
    )
    force_panel_sha256 = _sha256(
        raw["force_panel_contract_sha256"],
        name="measurement.prepared_inputs.force_panel_contract_sha256",
    )
    force_panel = _force_panel_contract(raw["force_panel_contract"])
    normalized_force_panel_sha256 = canonical_json_sha256(force_panel)
    if force_panel_sha256 != normalized_force_panel_sha256:
        raise ValueError(
            "prepared_inputs force_panel_contract_sha256 does not match the exact "
            "force_panel_contract object."
        )
    assets = dict(seal.asset_sha256s)
    expected_hashes = {
        "v1_preregistration_sha256": assets["v1_preregistration"],
        "parent_panel_sha256": assets["parent_panel"],
        "mol2_sha256": assets["benzene_mol2"],
        "projection_result_sha256": assets["benzene_projection_result"],
        "force_panel_contract_sha256": panel_contract_sha256,
    }
    actual_hashes = {
        "v1_preregistration_sha256": v1_sha256,
        "parent_panel_sha256": parent_sha256,
        "mol2_sha256": mol2_sha256,
        "projection_result_sha256": projection_sha256,
        "force_panel_contract_sha256": force_panel_sha256,
    }
    mismatches = sorted(
        field
        for field, expected in expected_hashes.items()
        if actual_hashes[field] != expected
    )
    if mismatches:
        raise ValueError(
            "prepared_inputs sealed hash mismatch: " + ", ".join(mismatches) + "."
        )

    normalized = {
        "benzene": {
            "compound_id": compound_id,
            "name": benzene_name,
            "atomic_numbers": list(benzene_numbers),
            "positions_angstrom": [list(row) for row in benzene_positions],
            "charge": 0,
            "multiplicity": 1,
            "cavity_radii_angstrom": list(benzene_radii),
            "mol2_sha256": mol2_sha256,
            "projection_result_sha256": projection_sha256,
            "predecessor_dof": [0, 0],
        },
        "water": {
            "atomic_numbers": [8, 1, 1],
            "positions_angstrom": [list(row) for row in water_positions],
            "charge": 0,
            "multiplicity": 1,
            "cavity_radii_angstrom": list(water_radii),
        },
        "v1_preregistration_sha256": v1_sha256,
        "parent_panel_sha256": parent_sha256,
        "force_panel_contract": force_panel,
        "force_panel_contract_sha256": force_panel_sha256,
    }
    provider_hashes = dict(seal.provider_configuration_sha256s)
    if force_panel_sha256 != provider_hashes["force_panel_contract"]:
        raise ValueError(
            "prepared_inputs force_panel_contract_sha256 differs from the sealed "
            "force-panel contract."
        )
    if canonical_json_sha256(normalized) != provider_hashes["prepared_input_manifest"]:
        raise ValueError(
            "prepared_inputs normalized content differs from the sealed manifest."
        )
    return _freeze_object(normalized, name="measurement.prepared_inputs")


def _prepared_providers(value: object, *, seal: ComputationSealV2) -> _FrozenObject:
    raw = _object(value, name="measurement.prepared_providers")
    _exact_fields(raw, _PREPARED_PROVIDER_FIELDS, name="measurement.prepared_providers")
    checkpoint_ledger = dict(
        _sha_ledger(
            raw["checkpoint_sha256s"],
            name="measurement.prepared_providers.checkpoint_sha256s",
            key_validator=_stable_label,
            exact_keys=frozenset(REQUIRED_CHECKPOINT_KEYS),
        )
    )
    provider_ledger = dict(
        _sha_ledger(
            raw["provider_configuration_sha256s"],
            name="measurement.prepared_providers.provider_configuration_sha256s",
            key_validator=_stable_label,
            exact_keys=_PROVIDER_CONFIGURATION_KEYS,
        )
    )
    for name, ledger in (
        ("checkpoint_sha256s", checkpoint_ledger),
        ("provider_configuration_sha256s", provider_ledger),
    ):
        if len(set(ledger.values())) != len(ledger):
            raise ValueError(
                f"measurement.prepared_providers.{name} contains digest aliases."
            )
    combined_digests = (*checkpoint_ledger.values(), *provider_ledger.values())
    if len(set(combined_digests)) != len(combined_digests):
        raise ValueError(
            "measurement.prepared_providers ledgers contain cross-ledger digest "
            "aliases."
        )
    normalized = {
        "profile_id": _text(
            raw["profile_id"], name="measurement.prepared_providers.profile_id"
        ),
        "scalar_id": _text(
            raw["scalar_id"], name="measurement.prepared_providers.scalar_id"
        ),
        "state_id": _text(
            raw["state_id"], name="measurement.prepared_providers.state_id"
        ),
        "checkpoint_sha256s": checkpoint_ledger,
        "provider_configuration_sha256s": provider_ledger,
    }
    expected = {
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
        "checkpoint_sha256s": dict(seal.checkpoint_sha256s),
        "provider_configuration_sha256s": dict(seal.provider_configuration_sha256s),
    }
    if normalized != expected:
        raise ValueError(
            "measurement.prepared_providers does not equal the computation seal."
        )
    return _freeze_object(normalized, name="measurement.prepared_providers")


def _replicate_envelope(
    value: Mapping[str, object], *, seal: ComputationSealV2
) -> tuple[
    Mapping[str, object],
    dict[str, object],
    tuple[tuple[str, bool], ...],
    str,
]:
    """Validate provenance and digest fields before the scientific tree."""

    raw = _object(value, name="replicate admission record")
    _exact_fields(raw, _REPLICATE_FIELDS, name="replicate admission record")
    if raw["schema_id"] != REPLICATE_ADMISSION_RECORD_V2_SCHEMA:
        raise ValueError("replicate schema_id is unsupported.")
    if raw["measurement_schema_id"] == LEGACY_MEASUREMENT_SCHEMA_ID:
        raise ValueError(
            "historical v1 measurement scope is not a rich-v2 admission preimage."
        )
    if raw["measurement_schema_id"] != RICH_MEASUREMENT_SCHEMA_ID:
        raise ValueError("replicate measurement_schema_id is unsupported.")
    label = raw["label"]
    if label not in {"a", "b"}:
        raise ValueError("replicate label must be a or b.")
    gates = _object(raw["gate_results"], name="gate_results")
    _exact_fields(gates, frozenset(REPLICATE_GATE_NAMES), name="gate_results")
    normalized_gates = tuple(
        sorted(
            (name, _strict_bool(gates[name], name=f"gate_results.{name}"))
            for name in REPLICATE_GATE_NAMES
        )
    )
    if not all(passed for _, passed in normalized_gates):
        raise ValueError("every replicate admission gate must pass.")
    measurement_raw = _object(raw["measurement"], name="measurement")
    measurement_sha256 = _sha256(raw["measurement_sha256"], name="measurement_sha256")
    if measurement_sha256 != canonical_json_sha256(measurement_raw):
        raise ValueError("measurement_sha256 does not match the measurement.")
    values: dict[str, object] = {
        "schema_id": REPLICATE_ADMISSION_RECORD_V2_SCHEMA,
        "artifact_id": _text(raw["artifact_id"], name="artifact_id"),
        "artifact_sha256": _sha256(raw["artifact_sha256"], name="artifact_sha256"),
        "label": label,
        "process_uuid": _uuid(raw["process_uuid"], name="process_uuid"),
        "process_started_at_utc": _utc_text(
            raw["process_started_at_utc"], name="process_started_at_utc"
        ),
        "seal_id": _text(raw["seal_id"], name="seal_id"),
        "seal_sha256": _sha256(raw["seal_sha256"], name="seal_sha256"),
        "git_head": _git_object(raw["git_head"], name="git_head"),
        "git_tree": _git_object(raw["git_tree"], name="git_tree"),
        "source_ledger_sha256": _sha256(
            raw["source_ledger_sha256"], name="source_ledger_sha256"
        ),
        "asset_ledger_sha256": _sha256(
            raw["asset_ledger_sha256"], name="asset_ledger_sha256"
        ),
        "runtime_fingerprint_sha256": _sha256(
            raw["runtime_fingerprint_sha256"], name="runtime_fingerprint_sha256"
        ),
        "measurement_schema_id": RICH_MEASUREMENT_SCHEMA_ID,
        "measurement_sha256": measurement_sha256,
        "gate_results": normalized_gates,
    }
    expected = {
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "git_head": seal.git_head,
        "git_tree": seal.git_tree,
        "source_ledger_sha256": seal.source_ledger_sha256,
        "asset_ledger_sha256": seal.asset_ledger_sha256,
        "runtime_fingerprint_sha256": seal.runtime_fingerprint_sha256,
    }
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(f"replicate {field} does not match its computation seal.")
    return raw, values, normalized_gates, measurement_sha256


def _candidate_from_normalized_inputs(
    *,
    seal: ComputationSealV2,
    prepared_inputs: _FrozenObject,
    prepared_providers: _FrozenObject,
    recording: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    _exact_fields(
        recording,
        frozenset({"state_leaves", "solve_events", "systems"}),
        name="rich measurement recording",
    )
    raw_leaves = _object(recording["state_leaves"], name="measurement.state_leaves")
    if not raw_leaves:
        raise ValueError("measurement.state_leaves must not be empty.")
    state_leaves: dict[str, StateLeafV2] = {}
    for digest, leaf_raw in raw_leaves.items():
        key = _sha256(digest, name="measurement.state_leaves key")
        leaf = StateLeafV2.from_mapping(_object(leaf_raw, name=f"state_leaves[{key}]"))
        if leaf.state_leaf_sha256 != key:
            raise ValueError("state leaf storage key must equal its content digest.")
        state_leaves[key] = leaf
    raw_events = recording["solve_events"]
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("measurement.solve_events must be a nonempty JSON array.")
    solve_events = tuple(
        SolveEventV2.from_mapping(
            _object(event, name=f"solve_events[{index}]"),
            state_leaves=state_leaves,
            expected_index=index,
        )
        for index, event in enumerate(raw_events)
    )
    if set(state_leaves) != {event.state_leaf_sha256 for event in solve_events}:
        raise ValueError(
            "measurement state_leaves must equal the solve-event referenced leaf set."
        )
    if len(solve_events) != 141:
        raise ValueError("rich-v2 measurement must contain exactly 141 solve events.")
    systems = _object(recording["systems"], name="measurement.systems")
    _exact_fields(systems, frozenset({"benzene", "water"}), name="measurement.systems")
    provider_payload = _thaw_json(prepared_providers)
    benzene = BenzeneSystemV2.from_mapping(
        _object(systems["benzene"], name="measurement.systems.benzene"),
        solve_events=solve_events,
        state_leaves=state_leaves,
        prepared_pes_configuration_sha256=provider_payload[
            "provider_configuration_sha256s"
        ]["benzene_pes_configuration"],
    )
    prepared_inputs_payload = _thaw_json(prepared_inputs)
    water_positions = tuple(
        tuple(float(value) for value in row)
        for row in prepared_inputs_payload["water"]["positions_angstrom"]
    )
    water = WaterSystemV2.from_mapping(
        _object(systems["water"], name="measurement.systems.water"),
        solve_events=solve_events,
        state_leaves=state_leaves,
        prepared_water_positions=water_positions,
        prepared_pes_configuration_sha256=provider_payload[
            "provider_configuration_sha256s"
        ]["water_pes_configuration"],
    )
    metrics = _reported_metrics(benzene, water)
    predecessor_indices = (
        *benzene.predecessor_stencil.component.event_indices,
        *benzene.predecessor_stencil.h4_event_indices,
    )
    metrics["benzene_center_legacy_root_sha256"] = state_leaves[
        solve_events[benzene.reported_center_event_index].state_leaf_sha256
    ].legacy_root_sha256
    metrics["benzene_displaced_legacy_root_sha256s"] = [
        state_leaves[solve_events[index].state_leaf_sha256].legacy_root_sha256
        for index in predecessor_indices[1:]
    ]
    force_panel = prepared_inputs_payload["force_panel_contract"]
    derived_gates = _replicate_gates(metrics, seal=seal, force_panel=force_panel)
    normalized_events = [
        {
            "schema_id": event.schema_id,
            "solve_event_sha256": event.solve_event_sha256,
            "event_index": event.event_index,
            "state_leaf_sha256": event.state_leaf_sha256,
            "geometry_sha256": event.geometry_sha256,
            "provider_configuration_sha256": event.provider_configuration_sha256,
            "topology_id": event.topology_id,
        }
        for event in solve_events
    ]
    candidate = {
        "schema_id": RICH_MEASUREMENT_SCHEMA_ID,
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
        "protocol": {
            "preregistration_id": seal.preregistration_id,
            "preregistration_sha256": seal.preregistration_sha256,
            "asset_ledger_sha256": seal.asset_ledger_sha256,
            "scientific_settings_sha256": canonical_json_sha256(
                _thaw_json(seal.scientific_settings)
            ),
            "fit_calibration_or_case_selection": False,
        },
        "panel_contract_sha256": prepared_inputs_payload["force_panel_contract_sha256"],
        "provider_ledger_sha256": canonical_json_sha256(provider_payload),
        "prepared_inputs": prepared_inputs_payload,
        "prepared_providers": provider_payload,
        "state_leaves": {
            digest: leaf.as_dict() for digest, leaf in sorted(state_leaves.items())
        },
        "solve_events": normalized_events,
        "systems": {
            "benzene": dict(systems["benzene"]),
            "water": dict(systems["water"]),
        },
        "reported_metrics": metrics,
        "reported_local_gate_results": derived_gates,
        "local_decision": dict(_LOCAL_REPLICATE_DECISION),
    }
    return candidate, derived_gates


def build_rich_harmonic_ef_measurement_candidate_v2(
    *,
    seal: ComputationSealV2,
    prepared_inputs: Mapping[str, object],
    prepared_providers: Mapping[str, object],
    recording: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    """Build one validated data-only rich candidate without admitting capability."""

    if not isinstance(seal, ComputationSealV2):
        raise TypeError("seal must be a ComputationSealV2.")
    panel_sha = _sha256(
        prepared_inputs.get("force_panel_contract_sha256"),
        name="prepared_inputs.force_panel_contract_sha256",
    )
    normalized_inputs = _prepared_inputs(
        prepared_inputs, seal=seal, panel_contract_sha256=panel_sha
    )
    normalized_providers = _prepared_providers(prepared_providers, seal=seal)
    candidate, gates = _candidate_from_normalized_inputs(
        seal=seal,
        prepared_inputs=normalized_inputs,
        prepared_providers=normalized_providers,
        recording=recording,
    )
    return (
        json.loads(json.dumps(candidate, sort_keys=True, allow_nan=False)),
        dict(gates),
    )


def _measurement(
    value: object, *, seal: ComputationSealV2
) -> tuple[_FrozenObject, dict[str, bool]]:
    raw = _object(value, name="measurement")
    schema_id = raw.get("schema_id")
    if schema_id == LEGACY_MEASUREMENT_SCHEMA_ID:
        raise ValueError(
            "historical v1 measurement scope is not a rich-v2 admission preimage."
        )
    if schema_id != RICH_MEASUREMENT_SCHEMA_ID:
        raise ValueError("measurement schema_id is unsupported.")
    _exact_fields(raw, _RICH_MEASUREMENT_FIELDS, name="measurement")
    expected_ids = {
        "schema_id": RICH_MEASUREMENT_SCHEMA_ID,
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
    }
    for field, expected in expected_ids.items():
        if raw[field] != expected:
            raise ValueError(f"measurement {field} does not match its seal.")

    protocol = _object(raw["protocol"], name="measurement.protocol")
    _exact_fields(protocol, _MEASUREMENT_PROTOCOL_FIELDS, name="measurement.protocol")
    normalized_protocol = {
        "preregistration_id": _text(
            protocol["preregistration_id"],
            name="measurement.protocol.preregistration_id",
        ),
        "preregistration_sha256": _sha256(
            protocol["preregistration_sha256"],
            name="measurement.protocol.preregistration_sha256",
        ),
        "asset_ledger_sha256": _sha256(
            protocol["asset_ledger_sha256"],
            name="measurement.protocol.asset_ledger_sha256",
        ),
        "scientific_settings_sha256": _sha256(
            protocol["scientific_settings_sha256"],
            name="measurement.protocol.scientific_settings_sha256",
        ),
        "fit_calibration_or_case_selection": _strict_bool(
            protocol["fit_calibration_or_case_selection"],
            name="measurement.protocol.fit_calibration_or_case_selection",
        ),
    }
    expected_protocol = {
        "preregistration_id": seal.preregistration_id,
        "preregistration_sha256": seal.preregistration_sha256,
        "asset_ledger_sha256": seal.asset_ledger_sha256,
        "scientific_settings_sha256": canonical_json_sha256(
            _thaw_json(seal.scientific_settings)
        ),
        "fit_calibration_or_case_selection": False,
    }
    if normalized_protocol != expected_protocol:
        raise ValueError("measurement.protocol does not match its sealed contract.")

    decision = _object(raw["local_decision"], name="measurement.local_decision")
    _exact_fields(
        decision, _MEASUREMENT_DECISION_FIELDS, name="measurement.local_decision"
    )
    normalized_decision = {
        field: _strict_bool(decision[field], name=f"measurement.local_decision.{field}")
        for field in _MEASUREMENT_DECISION_FIELDS
    }
    if normalized_decision != _LOCAL_REPLICATE_DECISION:
        raise ValueError("local_decision is not the fail-closed local decision.")

    panel_contract_sha256 = _sha256(
        raw["panel_contract_sha256"], name="measurement.panel_contract_sha256"
    )
    provider_ledger_sha256 = _sha256(
        raw["provider_ledger_sha256"],
        name="measurement.provider_ledger_sha256",
    )
    prepared_inputs = _prepared_inputs(
        raw["prepared_inputs"],
        seal=seal,
        panel_contract_sha256=panel_contract_sha256,
    )
    prepared_providers = _prepared_providers(raw["prepared_providers"], seal=seal)
    if provider_ledger_sha256 != canonical_json_sha256(_thaw_json(prepared_providers)):
        raise ValueError(
            "measurement.provider_ledger_sha256 does not match normalized "
            "prepared_providers."
        )
    candidate, derived_gates = _candidate_from_normalized_inputs(
        seal=seal,
        prepared_inputs=prepared_inputs,
        prepared_providers=prepared_providers,
        recording={
            "state_leaves": raw["state_leaves"],
            "solve_events": raw["solve_events"],
            "systems": raw["systems"],
        },
    )
    if raw["reported_metrics"] != candidate["reported_metrics"]:
        raise ValueError("reported_metrics do not equal raw-derived system metrics.")
    if raw["reported_local_gate_results"] != candidate["reported_local_gate_results"]:
        raise ValueError(
            "reported_local_gate_results do not equal raw-derived replicate gates."
        )
    if not all(derived_gates.values()):
        failed = sorted(name for name, passed in derived_gates.items() if not passed)
        raise ValueError("rich-v2 local admission gates failed: " + ", ".join(failed))
    if dict(raw) != candidate:
        raise ValueError(
            "measurement does not equal its canonical data-only candidate."
        )
    return _freeze_object(candidate, name="measurement"), derived_gates


_EXECUTION_FAILURE_FIELDS = frozenset(
    {
        "schema_id",
        "artifact_id",
        "artifact_sha256",
        "seal_id",
        "seal_sha256",
        "label",
        "process_uuid",
        "process_started_at_utc",
        "stage",
        "exception_type",
        "exception_message",
        "exception_message_sha256",
        "exception_message_truncated",
        "available_partial_evidence",
        "capabilities",
        "claim_boundary_id",
        "non_admissions",
    }
)

_EXECUTION_FAILURE_STAGES = frozenset(
    {"science-execution", "candidate-validation", "rich-v2-terminal"}
)
_EXECUTION_FAILURE_CAPTURE_KEYS = frozenset(
    {
        "seal",
        "preregistration",
        "v1_preregistration",
        "parent_panel",
        "benzene_mol2",
        "benzene_projection",
        "mace_mdp_checkpoint",
        "mace_polar_checkpoint",
        "runner",
    }
)
_EXECUTION_FAILURE_PARTIAL_FIELDS = frozenset(
    {
        "captured_sha256s",
        "runtime_fingerprint_sha256",
        "recording_sha256",
        "state_leaf_count",
        "solve_event_count",
        "system_keys",
        "system_event_ranges",
        "candidate_measurement_sha256",
        "derived_gate_results",
    }
)
_EXECUTION_FAILURE_MESSAGE_LIMIT = 4096
_EXECUTION_FAILURE_MAX_CANONICAL_BYTES = 32768


def _failure_partial_evidence(value: object) -> _FrozenObject:
    raw = _object(value, name="available_partial_evidence")
    _exact_fields(
        raw,
        _EXECUTION_FAILURE_PARTIAL_FIELDS,
        name="available_partial_evidence",
    )
    captures = dict(
        _sha_ledger(
            raw["captured_sha256s"],
            name="available_partial_evidence.captured_sha256s",
            key_validator=_stable_label,
            exact_keys=_EXECUTION_FAILURE_CAPTURE_KEYS,
        )
    )
    runtime_sha = _sha256(
        raw["runtime_fingerprint_sha256"],
        name="available_partial_evidence.runtime_fingerprint_sha256",
    )
    recording_sha = (
        None
        if raw["recording_sha256"] is None
        else _sha256(raw["recording_sha256"], name="recording_sha256")
    )
    state_count = (
        None
        if raw["state_leaf_count"] is None
        else _nonnegative_int(raw["state_leaf_count"], name="state_leaf_count")
    )
    event_count = (
        None
        if raw["solve_event_count"] is None
        else _nonnegative_int(raw["solve_event_count"], name="solve_event_count")
    )
    if recording_sha is not None and (state_count is None or event_count is None):
        raise ValueError("recording digest requires state/event counts.")
    if event_count not in {None, 141}:
        raise ValueError("solve_event_count must be null or 141.")
    system_keys = raw["system_keys"]
    if system_keys is not None and system_keys != ["benzene", "water"]:
        raise ValueError("system_keys must be null or ['benzene', 'water'].")
    ranges = raw["system_event_ranges"]
    if ranges is not None:
        range_object = _object(ranges, name="system_event_ranges")
        _exact_fields(
            range_object,
            frozenset({"benzene", "water"}),
            name="system_event_ranges",
        )
        if range_object != {"benzene": [0, 8], "water": [8, 141]}:
            raise ValueError("system_event_ranges differ from the frozen graph.")
    candidate_sha = (
        None
        if raw["candidate_measurement_sha256"] is None
        else _sha256(
            raw["candidate_measurement_sha256"],
            name="candidate_measurement_sha256",
        )
    )
    raw_gates = raw["derived_gate_results"]
    if raw_gates is None:
        gates = None
    else:
        gate_object = _object(raw_gates, name="derived_gate_results")
        _exact_fields(
            gate_object,
            frozenset(REPLICATE_GATE_NAMES),
            name="derived_gate_results",
        )
        gates = {
            name: _strict_bool(gate_object[name], name=f"derived_gate_results.{name}")
            for name in REPLICATE_GATE_NAMES
        }
    return _freeze_object(
        {
            "captured_sha256s": captures,
            "runtime_fingerprint_sha256": runtime_sha,
            "recording_sha256": recording_sha,
            "state_leaf_count": state_count,
            "solve_event_count": event_count,
            "system_keys": system_keys,
            "system_event_ranges": ranges,
            "candidate_measurement_sha256": candidate_sha,
            "derived_gate_results": gates,
        },
        name="available_partial_evidence",
    )


@dataclass(frozen=True, slots=True, init=False)
class ExecutionFailureV2:
    schema_id: str
    artifact_id: str
    artifact_sha256: str
    seal_id: str
    seal_sha256: str
    label: str
    process_uuid: str
    process_started_at_utc: str
    stage: str
    exception_type: str
    exception_message: str
    exception_message_sha256: str
    exception_message_truncated: bool
    available_partial_evidence: _FrozenObject
    capabilities: CapabilityStatus
    claim_boundary_id: str
    non_admissions: tuple[str, ...]

    def __init__(self) -> None:
        raise TypeError("ExecutionFailureV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, seal: ComputationSealV2
    ) -> "ExecutionFailureV2":
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        raw = _object(value, name="execution failure")
        _exact_fields(raw, _EXECUTION_FAILURE_FIELDS, name="execution failure")
        if raw["schema_id"] != EXECUTION_FAILURE_V2_SCHEMA:
            raise ValueError("execution failure schema_id is unsupported.")
        raw_capabilities = _object(raw["capabilities"], name="capabilities")
        _exact_fields(raw_capabilities, _CAPABILITY_KEYS, name="capabilities")
        for tier in sorted(_CAPABILITY_KEYS):
            _strict_bool(raw_capabilities[tier], name=f"capabilities.{tier}")
        capabilities = CapabilityStatus(
            energy=raw_capabilities["E"],
            conservative_force=raw_capabilities["F"],
            hessian=raw_capabilities["H"],
            variational_functional=raw_capabilities["V"],
            molecular_dynamics=raw_capabilities["M"],
        )
        if capabilities != CapabilityStatus():
            raise ValueError("execution failure capabilities must all be false.")
        label = raw["label"]
        if label not in {"a", "b"}:
            raise ValueError("execution failure label must be a or b.")
        stage = _text(raw["stage"], name="stage")
        if stage not in _EXECUTION_FAILURE_STAGES:
            raise ValueError("execution failure stage is unsupported.")
        exception_message = _text(raw["exception_message"], name="exception_message")
        if len(exception_message) > _EXECUTION_FAILURE_MESSAGE_LIMIT:
            raise ValueError("execution failure exception_message is too long.")
        exception_message_sha256 = _sha256(
            raw["exception_message_sha256"], name="exception_message_sha256"
        )
        exception_message_truncated = _strict_bool(
            raw["exception_message_truncated"],
            name="exception_message_truncated",
        )
        if (
            not exception_message_truncated
            and exception_message_sha256 != canonical_json_sha256(exception_message)
        ):
            raise ValueError("untruncated exception message digest does not match.")
        values: dict[str, object] = {
            "schema_id": EXECUTION_FAILURE_V2_SCHEMA,
            "artifact_id": _text(raw["artifact_id"], name="artifact_id"),
            "artifact_sha256": _sha256(raw["artifact_sha256"], name="artifact_sha256"),
            "seal_id": _text(raw["seal_id"], name="seal_id"),
            "seal_sha256": _sha256(raw["seal_sha256"], name="seal_sha256"),
            "label": label,
            "process_uuid": _uuid(raw["process_uuid"], name="process_uuid"),
            "process_started_at_utc": _utc_text(
                raw["process_started_at_utc"], name="process_started_at_utc"
            ),
            "stage": stage,
            "exception_type": _text(raw["exception_type"], name="exception_type"),
            "exception_message": exception_message,
            "exception_message_sha256": exception_message_sha256,
            "exception_message_truncated": exception_message_truncated,
            "available_partial_evidence": _failure_partial_evidence(
                raw["available_partial_evidence"]
            ),
            "capabilities": capabilities,
            "claim_boundary_id": _text(
                raw["claim_boundary_id"], name="claim_boundary_id"
            ),
            "non_admissions": _exact_tuple(
                raw["non_admissions"],
                REQUIRED_NON_ADMISSIONS,
                name="non_admissions",
            ),
        }
        failure = _sealed_instance(cls, values)
        expected = {
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "claim_boundary_id": seal.claim_boundary_id,
            "non_admissions": seal.non_admissions,
        }
        _validate_cross_bindings(failure, expected, name="execution failure")
        if failure.artifact_sha256 != canonical_json_sha256(failure._content_payload()):
            raise ValueError(
                "artifact_sha256 does not match execution failure content."
            )
        encoded = json.dumps(
            failure.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > _EXECUTION_FAILURE_MAX_CANONICAL_BYTES:
            raise ValueError("execution failure artifact exceeds the size limit.")
        return failure

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "artifact_id": self.artifact_id,
            "seal_id": self.seal_id,
            "seal_sha256": self.seal_sha256,
            "label": self.label,
            "process_uuid": self.process_uuid,
            "process_started_at_utc": self.process_started_at_utc,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "exception_message_sha256": self.exception_message_sha256,
            "exception_message_truncated": self.exception_message_truncated,
            "available_partial_evidence": _thaw_json(self.available_partial_evidence),
            "capabilities": _capability_payload(self.capabilities),
            "claim_boundary_id": self.claim_boundary_id,
            "non_admissions": list(self.non_admissions),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True, init=False)
class ReplicateAdmissionRecordV2:
    schema_id: str
    artifact_id: str
    artifact_sha256: str
    label: str
    process_uuid: str
    process_started_at_utc: str
    seal_id: str
    seal_sha256: str
    git_head: str
    git_tree: str
    source_ledger_sha256: str
    asset_ledger_sha256: str
    runtime_fingerprint_sha256: str
    measurement_schema_id: str
    measurement_sha256: str
    measurement: _FrozenObject
    gate_results: tuple[tuple[str, bool], ...]

    def __init__(self) -> None:
        raise TypeError("ReplicateAdmissionRecordV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, seal: ComputationSealV2
    ) -> "ReplicateAdmissionRecordV2":
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        raw, values, normalized_gates, measurement_sha256 = _replicate_envelope(
            value, seal=seal
        )
        measurement, derived_gates = _measurement(raw["measurement"], seal=seal)
        if dict(normalized_gates) != derived_gates:
            raise ValueError(
                "reported replicate gate_results do not equal engine-derived gates."
            )
        if not all(passed for _, passed in normalized_gates):
            raise ValueError("every replicate admission gate must pass.")
        values["measurement"] = measurement
        record = _sealed_instance(cls, values)
        if record.artifact_sha256 != canonical_json_sha256(record._content_payload()):
            raise ValueError("replicate artifact_sha256 does not match content.")
        record.validate_bindings(seal=seal)
        return record

    def validate_bindings(self, *, seal: ComputationSealV2) -> None:
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        expected = {
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "git_head": seal.git_head,
            "git_tree": seal.git_tree,
            "source_ledger_sha256": seal.source_ledger_sha256,
            "asset_ledger_sha256": seal.asset_ledger_sha256,
            "runtime_fingerprint_sha256": seal.runtime_fingerprint_sha256,
        }
        _validate_cross_bindings(self, expected, name="replicate")

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "artifact_id": self.artifact_id,
            "label": self.label,
            "process_uuid": self.process_uuid,
            "process_started_at_utc": self.process_started_at_utc,
            "seal_id": self.seal_id,
            "seal_sha256": self.seal_sha256,
            "git_head": self.git_head,
            "git_tree": self.git_tree,
            "source_ledger_sha256": self.source_ledger_sha256,
            "asset_ledger_sha256": self.asset_ledger_sha256,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "measurement_schema_id": self.measurement_schema_id,
            "measurement_sha256": self.measurement_sha256,
            "measurement": _thaw_json(self.measurement),
            "gate_results": dict(self.gate_results),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "artifact_sha256": self.artifact_sha256}


_AGGREGATE_FIELDS = frozenset(
    {
        "schema_id",
        "aggregate_id",
        "aggregate_sha256",
        "seal_id",
        "seal_sha256",
        "profile_id",
        "scalar_id",
        "state_id",
        "capabilities",
        "runtime_guards",
        "domain_guards",
        "claim_boundary_id",
        "non_admissions",
        "replicates",
        "aggregate_gate_results",
    }
)


@dataclass(frozen=True, slots=True, init=False)
class AggregateAdmissionInputsV2:
    schema_id: str
    aggregate_id: str
    aggregate_sha256: str
    seal_id: str
    seal_sha256: str
    profile_id: str
    scalar_id: str
    state_id: str
    capabilities: CapabilityStatus
    runtime_guards: tuple[str, ...]
    domain_guards: tuple[str, ...]
    claim_boundary_id: str
    non_admissions: tuple[str, ...]
    replicates: tuple[ReplicateAdmissionRecordV2, ReplicateAdmissionRecordV2]
    aggregate_gate_results: tuple[tuple[str, bool], ...]

    def __init__(self) -> None:
        raise TypeError("AggregateAdmissionInputsV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, seal: ComputationSealV2
    ) -> "AggregateAdmissionInputsV2":
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        raw = _object(value, name="aggregate admission inputs")
        _exact_fields(raw, _AGGREGATE_FIELDS, name="aggregate admission inputs")
        if raw["schema_id"] != AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA:
            raise ValueError("aggregate schema_id is unsupported.")
        outer_values: dict[str, object] = {
            "seal_id": _text(raw["seal_id"], name="seal_id"),
            "seal_sha256": _sha256(raw["seal_sha256"], name="seal_sha256"),
            **_admission_claim_values(raw),
        }
        expected_outer = {
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "profile_id": seal.profile_id,
            "scalar_id": seal.scalar_id,
            "state_id": seal.state_id,
            "runtime_guards": seal.runtime_guards,
            "domain_guards": seal.domain_guards,
            "claim_boundary_id": seal.claim_boundary_id,
            "non_admissions": seal.non_admissions,
            "capabilities": _EF_CAPABILITIES,
        }
        for field, expected_value in expected_outer.items():
            if outer_values[field] != expected_value:
                raise ValueError(f"aggregate {field} does not match its seal.")
        raw_aggregate_gates = _object(
            raw["aggregate_gate_results"], name="aggregate_gate_results"
        )
        _exact_fields(
            raw_aggregate_gates,
            frozenset(AGGREGATE_GATE_NAMES),
            name="aggregate_gate_results",
        )
        aggregate_gate_results = tuple(
            (
                name,
                _strict_bool(
                    raw_aggregate_gates[name],
                    name=f"aggregate_gate_results.{name}",
                ),
            )
            for name in AGGREGATE_GATE_NAMES
        )
        if not all(passed for _, passed in aggregate_gate_results):
            raise ValueError("every aggregate admission gate must pass.")
        raw_replicates = raw["replicates"]
        if not isinstance(raw_replicates, list) or len(raw_replicates) != 2:
            raise ValueError("aggregate requires exactly two replicate records.")
        envelopes = tuple(
            sorted(
                (
                    _replicate_envelope(
                        _object(item, name=f"replicates[{index}]"), seal=seal
                    )
                    for index, item in enumerate(raw_replicates)
                ),
                key=lambda item: str(item[1]["label"]),
            )
        )
        envelope_values = tuple(item[1] for item in envelopes)
        if tuple(item["label"] for item in envelope_values) != ("a", "b"):
            raise ValueError("aggregate requires replicate labels a and b.")
        distinct_fields = (
            "artifact_id",
            "artifact_sha256",
            "process_uuid",
            "process_started_at_utc",
        )
        for field in distinct_fields:
            if envelope_values[0][field] == envelope_values[1][field]:
                raise ValueError(f"replicate {field} values must be distinct.")
        shared_fields = (
            "seal_id",
            "seal_sha256",
            "git_head",
            "git_tree",
            "source_ledger_sha256",
            "asset_ledger_sha256",
            "runtime_fingerprint_sha256",
            "measurement_schema_id",
            "measurement_sha256",
            "gate_results",
        )
        for field in shared_fields:
            if envelope_values[0][field] != envelope_values[1][field]:
                raise ValueError(f"replicate {field} values must match exactly.")
        if envelopes[0][0]["measurement"] != envelopes[1][0]["measurement"]:
            raise ValueError("replicate measurement objects must match exactly.")
        replicates = tuple(
            sorted(
                (
                    ReplicateAdmissionRecordV2.from_mapping(item, seal=seal)
                    for item in raw_replicates
                ),
                key=lambda item: item.label,
            )
        )
        if tuple(item.label for item in replicates) != ("a", "b"):
            raise ValueError("aggregate requires replicate labels a and b.")
        values: dict[str, object] = {
            "schema_id": AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA,
            "aggregate_id": _text(raw["aggregate_id"], name="aggregate_id"),
            "aggregate_sha256": _sha256(
                raw["aggregate_sha256"], name="aggregate_sha256"
            ),
            **outer_values,
            "replicates": replicates,
            "aggregate_gate_results": aggregate_gate_results,
        }
        aggregate = _sealed_instance(cls, values)
        if aggregate.aggregate_sha256 != canonical_json_sha256(
            aggregate._content_payload()
        ):
            raise ValueError("aggregate_sha256 does not match aggregate content.")
        aggregate.validate_bindings(seal=seal)
        return aggregate

    @property
    def all_gates_passed(self) -> bool:
        local_passed = all(
            all(passed for _, passed in record.gate_results)
            for record in self.replicates
        )
        return local_passed and all(passed for _, passed in self.aggregate_gate_results)

    @property
    def measurement_sha256(self) -> str:
        return self.replicates[0].measurement_sha256

    def validate_bindings(self, *, seal: ComputationSealV2) -> None:
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        expected = {
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "profile_id": seal.profile_id,
            "scalar_id": seal.scalar_id,
            "state_id": seal.state_id,
            "runtime_guards": seal.runtime_guards,
            "domain_guards": seal.domain_guards,
            "claim_boundary_id": seal.claim_boundary_id,
            "non_admissions": seal.non_admissions,
            "capabilities": _EF_CAPABILITIES,
        }
        _validate_cross_bindings(self, expected, name="aggregate")
        if not self.all_gates_passed:
            raise ValueError("aggregate requires every replicated gate to pass.")

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "aggregate_id": self.aggregate_id,
            "seal_id": self.seal_id,
            "seal_sha256": self.seal_sha256,
            "profile_id": self.profile_id,
            "scalar_id": self.scalar_id,
            "state_id": self.state_id,
            "capabilities": _capability_payload(self.capabilities),
            "runtime_guards": list(self.runtime_guards),
            "domain_guards": list(self.domain_guards),
            "claim_boundary_id": self.claim_boundary_id,
            "non_admissions": list(self.non_admissions),
            "replicates": [item.as_dict() for item in self.replicates],
            "aggregate_gate_results": dict(self.aggregate_gate_results),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "aggregate_sha256": self.aggregate_sha256}


_OVERLAY_FIELDS = frozenset(
    {
        "schema_id",
        "overlay_id",
        "seal_id",
        "seal_sha256",
        "aggregate_id",
        "aggregate_sha256",
        "profile_id",
        "scalar_id",
        "state_id",
        "capabilities",
        "runtime_guards",
        "domain_guards",
        "claim_boundary_id",
        "non_admissions",
        "content_sha256",
    }
)


def _overlay_values(
    value: Mapping[str, object],
    *,
    seal: ComputationSealV2,
    aggregate_values: Mapping[str, object],
) -> dict[str, object]:
    """Normalize and cross-bind overlay data independently of construction."""

    raw = _object(value, name="admission overlay")
    _exact_fields(raw, _OVERLAY_FIELDS, name="admission overlay")
    if raw["schema_id"] != ADMISSION_OVERLAY_V2_SCHEMA:
        raise ValueError("overlay schema_id is unsupported.")
    values: dict[str, object] = {
        "schema_id": ADMISSION_OVERLAY_V2_SCHEMA,
        "overlay_id": _text(raw["overlay_id"], name="overlay_id"),
        "seal_id": _text(raw["seal_id"], name="seal_id"),
        "seal_sha256": _sha256(raw["seal_sha256"], name="seal_sha256"),
        "aggregate_id": _text(raw["aggregate_id"], name="aggregate_id"),
        "aggregate_sha256": _sha256(raw["aggregate_sha256"], name="aggregate_sha256"),
        **_admission_claim_values(raw),
        "content_sha256": _sha256(raw["content_sha256"], name="content_sha256"),
    }
    expected = {
        "seal_id": seal.artifact_id,
        "seal_sha256": seal.content_sha256,
        **{
            field: aggregate_values[field]
            for field in (
                "aggregate_id",
                "aggregate_sha256",
                "profile_id",
                "scalar_id",
                "state_id",
                "capabilities",
                "runtime_guards",
                "domain_guards",
                "claim_boundary_id",
                "non_admissions",
            )
        },
    }
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(f"overlay {field} does not match its bound evidence.")
    return values


@dataclass(frozen=True, slots=True, init=False)
class AdmissionOverlayV2:
    schema_id: str
    overlay_id: str
    seal_id: str
    seal_sha256: str
    aggregate_id: str
    aggregate_sha256: str
    profile_id: str
    scalar_id: str
    state_id: str
    capabilities: CapabilityStatus
    runtime_guards: tuple[str, ...]
    domain_guards: tuple[str, ...]
    claim_boundary_id: str
    non_admissions: tuple[str, ...]
    content_sha256: str

    def __init__(self) -> None:
        raise TypeError("AdmissionOverlayV2 requires from_mapping().")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        seal: ComputationSealV2,
        aggregate: AggregateAdmissionInputsV2,
    ) -> "AdmissionOverlayV2":
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        if not isinstance(aggregate, AggregateAdmissionInputsV2):
            raise TypeError("aggregate must be an AggregateAdmissionInputsV2.")
        aggregate.validate_bindings(seal=seal)
        aggregate_values = {
            field: getattr(aggregate, field)
            for field in (
                "aggregate_id",
                "aggregate_sha256",
                "profile_id",
                "scalar_id",
                "state_id",
                "capabilities",
                "runtime_guards",
                "domain_guards",
                "claim_boundary_id",
                "non_admissions",
            )
        }
        values = _overlay_values(value, seal=seal, aggregate_values=aggregate_values)
        overlay = _sealed_instance(cls, values)
        if overlay.content_sha256 != canonical_json_sha256(overlay._content_payload()):
            raise ValueError("overlay content_sha256 does not match content.")
        overlay.validate_bindings(seal=seal, aggregate=aggregate)
        return overlay

    def validate_bindings(
        self, *, seal: ComputationSealV2, aggregate: AggregateAdmissionInputsV2
    ) -> None:
        if not isinstance(seal, ComputationSealV2):
            raise TypeError("seal must be a ComputationSealV2.")
        if not isinstance(aggregate, AggregateAdmissionInputsV2):
            raise TypeError("aggregate must be an AggregateAdmissionInputsV2.")
        aggregate.validate_bindings(seal=seal)
        expected = {
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "aggregate_id": aggregate.aggregate_id,
            "aggregate_sha256": aggregate.aggregate_sha256,
            "profile_id": aggregate.profile_id,
            "scalar_id": aggregate.scalar_id,
            "state_id": aggregate.state_id,
            "capabilities": aggregate.capabilities,
            "runtime_guards": aggregate.runtime_guards,
            "domain_guards": aggregate.domain_guards,
            "claim_boundary_id": aggregate.claim_boundary_id,
            "non_admissions": aggregate.non_admissions,
        }
        _validate_cross_bindings(self, expected, name="overlay")

    def _content_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "overlay_id": self.overlay_id,
            "seal_id": self.seal_id,
            "seal_sha256": self.seal_sha256,
            "aggregate_id": self.aggregate_id,
            "aggregate_sha256": self.aggregate_sha256,
            "profile_id": self.profile_id,
            "scalar_id": self.scalar_id,
            "state_id": self.state_id,
            "capabilities": _capability_payload(self.capabilities),
            "runtime_guards": list(self.runtime_guards),
            "domain_guards": list(self.domain_guards),
            "claim_boundary_id": self.claim_boundary_id,
            "non_admissions": list(self.non_admissions),
        }

    def as_dict(self) -> dict[str, object]:
        return {**self._content_payload(), "content_sha256": self.content_sha256}


__all__ = [
    "ADMISSION_OVERLAY_V2_SCHEMA",
    "AGGREGATE_GATE_NAMES",
    "AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA",
    "CLAIM_BOUNDARY_ID",
    "COMPUTATION_SEAL_V2_SCHEMA",
    "EXPOSURE_AWARE_PROTOCOL_LABEL",
    "EXECUTION_FAILURE_V2_SCHEMA",
    "H1_V2_CAVITY_PROFILE_ID",
    "H1_V2_CONTINUUM_PROFILE_ID",
    "H1_V2_DISPLACEMENT_POLICY",
    "H1_V2_EXACT_SCALAR",
    "H1_V2_EXCLUDED_COMPONENTS",
    "H1_V2_FORCE_DERIVATIVE",
    "H1_V2_INCLUDED_COMPONENTS",
    "H1_V2_INDUCED_SOURCE_KERNEL",
    "H1_V2_LONG_RANGE_EVALUATOR_ID",
    "H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT",
    "H1_V2_PERMANENT_SOURCE_KERNEL",
    "H1_V2_PROFILE_ID",
    "H1_V2_RADII_PROVIDER_ID",
    "H1_V2_RECEIVER_KERNEL",
    "H1_V2_RECEIVER_COMPONENT_ORDER",
    "H1_V2_RECEIVER_SPACE_CONTRACT_SHA256",
    "H1_V2_RECEIVER_SPACE_ID",
    "H1_V2_RECEIVER_UNITS",
    "H1_V2_ROOT_METHOD",
    "H1_V2_SCALAR_ID",
    "H1_V2_SECOND_START",
    "H1_V2_SOURCE_COEFFICIENT_BASIS_ID",
    "H1_V2_SOURCE_COEFFICIENT_ORDER",
    "H1_V2_SOURCE_COEFFICIENT_UNITS",
    "H1_V2_SOURCE_SPACE_CONTRACT_SHA256",
    "H1_V2_ENDPOINT_REPLAY_SCOPE",
    "H1_V2_STATE_ID",
    "H1_V2_TOPOLOGY_POLICY",
    "LEGACY_MEASUREMENT_SCHEMA_ID",
    "MEASUREMENT_SCHEMA_ID",
    "RICH_MEASUREMENT_SCHEMA_ID",
    "REPLICATE_GATE_NAMES",
    "REQUIRED_ASSET_KEYS",
    "REQUIRED_CHECKPOINT_KEYS",
    "REQUIRED_DOMAIN_GUARDS",
    "REQUIRED_NON_ADMISSIONS",
    "REQUIRED_RUNTIME_GUARDS",
    "REPLICATE_ADMISSION_RECORD_V2_SCHEMA",
    "SOLVE_EVENT_V2_SCHEMA",
    "STATE_LEAF_V2_SCHEMA",
    "V1_ADMISSION_GATE_NAMES",
    "AdmissionOverlayV2",
    "AggregateAdmissionInputsV2",
    "BenzenePredecessorStencilV2",
    "BenzeneSystemV2",
    "CartesianPanelV2",
    "ComputationSealV2",
    "ComponentStencilV2",
    "ClosedLoopV2",
    "DirectionalCheckV2",
    "ExecutionFailureV2",
    "ReplicateAdmissionRecordV2",
    "RigidRotationV2",
    "RigidTranslationV2",
    "SolveEventV2",
    "StateLeafV2",
    "WaterSystemV2",
    "build_rich_harmonic_ef_measurement_candidate_v2",
    "canonical_numeric_array_sha256_v2",
    "canonical_source_monopole_sum_v2",
    "h1_v2_audit_coefficient_sum_contract",
    "verified_pro_schema_amendment_contract",
]
