from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import numpy as np

import pytest

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.release import admission as admission_contracts
from maple.solvation.release.admission import (
    BenzenePredecessorStencilV2,
    BenzeneSystemV2,
    CartesianPanelV2,
    ADMISSION_OVERLAY_V2_SCHEMA,
    AGGREGATE_GATE_NAMES,
    AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA,
    CLAIM_BOUNDARY_ID,
    COMPUTATION_SEAL_V2_SCHEMA,
    ComponentStencilV2,
    ClosedLoopV2,
    DirectionalCheckV2,
    ExecutionFailureV2,
    EXECUTION_FAILURE_V2_SCHEMA,
    EXPOSURE_AWARE_PROTOCOL_LABEL,
    H1_V2_CAVITY_PROFILE_ID,
    H1_V2_CONTINUUM_PROFILE_ID,
    H1_V2_DISPLACEMENT_POLICY,
    H1_V2_EXACT_SCALAR,
    H1_V2_EXCLUDED_COMPONENTS,
    H1_V2_FORCE_DERIVATIVE,
    H1_V2_INCLUDED_COMPONENTS,
    H1_V2_INDUCED_SOURCE_KERNEL,
    H1_V2_LONG_RANGE_EVALUATOR_ID,
    H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT,
    H1_V2_PERMANENT_SOURCE_KERNEL,
    H1_V2_PROFILE_ID,
    H1_V2_RADII_PROVIDER_ID,
    H1_V2_RECEIVER_KERNEL,
    H1_V2_RECEIVER_COMPONENT_ORDER,
    H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
    H1_V2_RECEIVER_SPACE_ID,
    H1_V2_RECEIVER_UNITS,
    H1_V2_ROOT_METHOD,
    H1_V2_SCALAR_ID,
    H1_V2_SECOND_START,
    H1_V2_STATE_ID,
    H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
    H1_V2_SOURCE_COEFFICIENT_ORDER,
    H1_V2_SOURCE_COEFFICIENT_UNITS,
    H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
    H1_V2_ENDPOINT_REPLAY_SCOPE,
    H1_V2_TOPOLOGY_POLICY,
    MEASUREMENT_SCHEMA_ID,
    LEGACY_MEASUREMENT_SCHEMA_ID,
    RICH_MEASUREMENT_SCHEMA_ID,
    REQUIRED_ASSET_KEYS,
    REQUIRED_CHECKPOINT_KEYS,
    REQUIRED_DOMAIN_GUARDS,
    REQUIRED_NON_ADMISSIONS,
    REQUIRED_RUNTIME_GUARDS,
    REPLICATE_ADMISSION_RECORD_V2_SCHEMA,
    REPLICATE_GATE_NAMES,
    V1_ADMISSION_GATE_NAMES,
    AdmissionOverlayV2,
    AggregateAdmissionInputsV2,
    ComputationSealV2,
    ReplicateAdmissionRecordV2,
    RigidRotationV2,
    RigidTranslationV2,
    SOLVE_EVENT_V2_SCHEMA,
    STATE_LEAF_V2_SCHEMA,
    SolveEventV2,
    StateLeafV2,
    WaterSystemV2,
    build_rich_harmonic_ef_measurement_candidate_v2,
    canonical_numeric_array_sha256_v2,
    canonical_source_monopole_sum_v2,
    h1_v2_audit_coefficient_sum_contract,
)
from maple.solvation.release.evidence import canonical_json_sha256

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def _bind(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = canonical_json_sha256(payload)
    return result


def _rehash(payload: dict[str, object], field: str) -> None:
    payload[field] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _rehash_replicate_measurement(payload: dict[str, object]) -> None:
    payload["measurement_sha256"] = canonical_json_sha256(payload["measurement"])
    _rehash(payload, "artifact_sha256")


def _science() -> dict[str, object]:
    return {
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


def _runtime() -> dict[str, object]:
    return {
        "python": "3.11.13",
        "implementation": "CPython",
        "platform": "Linux-6.6-x86_64",
        "machine": "x86_64",
        "cpu_model": "Example CPU",
        "packages": {
            "maple": None,
            "ase": "3.26.0",
            "numpy": "2.1.0",
            "scipy": "1.15.0",
            "torch": "2.7.0",
            "mace-torch": "0.3.14",
            "graph-longrange": "0.1.0",
            "pyscf": None,
            "pyddx": "0.6.0",
        },
        "numpy": {"version": "2.1.0", "show_config": "blas=openblas"},
        "torch": {
            "version": "2.7.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "cuda_available": True,
            "devices": ["NVIDIA Example GPU"],
            "default_dtype": "torch.float64",
            "threads": 1,
            "interop_threads": 1,
            "driver_version": "555.42.02",
            "device_uuids": ["GPU-33333333-3333-4333-8333-333333333333"],
            "device_capabilities": ["8.9"],
            "device_multiprocessor_counts": [128],
            "deterministic_algorithms_enabled": True,
            "deterministic_debug_mode": 2,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_LAUNCH_BLOCKING": None,
            "PYTHONHASHSEED": "0",
        },
        "execution_device": "cuda",
        "execution_dtype": "float64",
    }


def _force_panel_payload() -> dict[str, object]:
    return {
        "gepol_regression_compound_id": "mobley_3053621",
        "gepol_regression_cartesian_dof": [0, 0],
        "water_geometry_angstrom": [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239, 0.9266, 0.0],
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


FORCE_PANEL_SHA = canonical_json_sha256(_force_panel_payload())


def _prepared_inputs_from_assets(assets: dict[str, str]) -> dict[str, object]:
    return {
        "benzene": {
            "compound_id": "mobley_3053621",
            "name": "benzene",
            "atomic_numbers": [6] * 12,
            "positions_angstrom": [[float(index), 0.0, 0.0] for index in range(12)],
            "charge": 0,
            "multiplicity": 1,
            "cavity_radii_angstrom": [1.7] * 12,
            "mol2_sha256": assets["benzene_mol2"],
            "projection_result_sha256": assets["benzene_projection_result"],
            "predecessor_dof": [0, 0],
        },
        "water": {
            "atomic_numbers": [8, 1, 1],
            "positions_angstrom": [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.239, 0.9266, 0.0],
            ],
            "charge": 0,
            "multiplicity": 1,
            "cavity_radii_angstrom": [1.5, 1.2, 1.2],
        },
        "v1_preregistration_sha256": assets["v1_preregistration"],
        "parent_panel_sha256": assets["parent_panel"],
        "force_panel_contract": _force_panel_payload(),
        "force_panel_contract_sha256": FORCE_PANEL_SHA,
    }


def _seal_payload() -> dict[str, object]:
    asset_values = (SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, SHA_F, "0" * 64, "8" * 64)
    assets = {
        key: asset_values[index]
        for index, key in enumerate(REQUIRED_ASSET_KEYS)
    }
    return _bind(
        {
            "schema_id": COMPUTATION_SEAL_V2_SCHEMA,
            "artifact_id": "route2-v2-computation-seal",
            "profile_id": H1_V2_PROFILE_ID,
            "scalar_id": H1_V2_SCALAR_ID,
            "state_id": H1_V2_STATE_ID,
            "git_head": "1" * 40,
            "git_tree": "2" * 40,
            "git_clean": True,
            "preregistration_id": "route2-v2-preregistration",
            "preregistration_sha256": SHA_A,
            "runner_id": "route2-v2-runner",
            "runner_sha256": SHA_B,
            "aggregator_id": "route2-v2-aggregator-program",
            "aggregator_sha256": SHA_C,
            "computation_source_sha256s": {
                "maple/solvation/release/pes_validation.py": SHA_D,
                "tools/route2_release/run_validation.py": SHA_E,
            },
            "asset_sha256s": assets,
            "checkpoint_sha256s": {
                key: (SHA_A, SHA_B)[index]
                for index, key in enumerate(REQUIRED_CHECKPOINT_KEYS)
            },
            "provider_configuration_sha256s": {
                "hybrid_configuration": "1" * 64,
                "hybrid_provenance": "2" * 64,
                "continuum_configuration": "3" * 64,
                "continuum_provenance": "4" * 64,
                "cavity_configuration": "5" * 64,
                "benzene_pes_configuration": "6" * 64,
                "water_pes_configuration": "7" * 64,
                "prepared_input_manifest": canonical_json_sha256(
                    _prepared_inputs_from_assets(assets)
                ),
                "force_panel_contract": FORCE_PANEL_SHA,
            },
            "scientific_settings": _science(),
            "runtime_fingerprint": _runtime(),
            "runtime_guards": list(REQUIRED_RUNTIME_GUARDS),
            "domain_guards": list(REQUIRED_DOMAIN_GUARDS),
            "exposure_aware_protocol_label": EXPOSURE_AWARE_PROTOCOL_LABEL,
            "claim_boundary_id": CLAIM_BOUNDARY_ID,
            "non_admissions": list(REQUIRED_NON_ADMISSIONS),
        },
        "content_sha256",
    )


def _seal() -> ComputationSealV2:
    return ComputationSealV2.from_mapping(_seal_payload())


def _start_leaf(atom_count: int, *, initial: str = SHA_A) -> dict[str, object]:
    return {
        "initial_state_sha256": initial,
        "converged": True,
        "iterations": 3,
        "final_native_field_eV": [[0.0] * 8 for _ in range(atom_count)],
        "final_residual_eV": [[0.0] * 8 for _ in range(atom_count)],
        "final_polarization_energy_eV": -0.25,
        "final_total_energy_eV": -1.25,
    }


def _state_leaf_payload(
    atom_count: int = 3,
    *,
    total_energy: float = -1.25,
    geometry: str = SHA_A,
    prepared_pes_configuration: str = "7" * 64,
) -> dict[str, object]:
    permanent = [[0.0] * 4 for _ in range(atom_count)]
    induced = [[0.0] * 4 for _ in range(atom_count)]
    permanent[0][0] = 0.25
    induced[0][0] = -0.25
    zero = [[0.0] * 4 for _ in range(atom_count)]
    polar_final = [
        [left + right for left, right in zip(z, i, strict=True)]
        for z, i in zip(zero, induced, strict=True)
    ]
    audit_sum = [
        [left + right for left, right in zip(p, i, strict=True)]
        for p, i in zip(permanent, induced, strict=True)
    ]
    payload = {
        "schema_id": STATE_LEAF_V2_SCHEMA,
        "legacy_root_sha256": "9" * 64,
        "prepared_pes_configuration_sha256": prepared_pes_configuration,
        "geometry_sha256": geometry,
        "provider_configuration_sha256": SHA_B,
        "topology_id": SHA_C,
        "target_charge_e": 0.0,
        "audit_coefficient_sum_charge_e": 0.0,
        "source_coefficient_basis_id": H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
        "source_space_contract_sha256": H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
        "source_coefficient_order": list(H1_V2_SOURCE_COEFFICIENT_ORDER),
        "receiver_space_id": H1_V2_RECEIVER_SPACE_ID,
        "receiver_space_contract_sha256": H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
        "receiver_component_order": list(H1_V2_RECEIVER_COMPONENT_ORDER),
        "audit_projection_contract": h1_v2_audit_coefficient_sum_contract(),
        "numeric_array_encoding_contract": H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT,
        "endpoint_replay_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
        "permanent_point_source4": permanent,
        "polar_zero_reference_source4": zero,
        "polar_final_source4": polar_final,
        "induced_gto_source4": induced,
        "audit_coefficient_sum4": audit_sum,
        "vacuum_energy_eV": -1.0,
        "polarization_energy_eV": -0.25,
        "total_energy_eV": total_energy,
        "cold_start": _start_leaf(atom_count),
        "wide_start": _start_leaf(atom_count, initial=SHA_B),
    }
    payload["vacuum_energy_eV"] = total_energy + 0.25
    payload["cold_start"]["final_total_energy_eV"] = total_energy
    payload["wide_start"]["final_total_energy_eV"] = total_energy
    array_inputs = {
        "permanent_point_source4": (
            permanent,
            "permanent-point-source4",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "polar_zero_reference_source4": (
            zero,
            "polar-zero-reference-source4-nonoperational",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "polar_final_source4": (
            polar_final,
            "polar-final-source4",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "induced_gto_source4": (
            induced,
            "induced-gto1p5-source4",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "audit_coefficient_sum4": (
            audit_sum,
            "audit-coefficient-sum4-nonoperational",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
    }
    for start_name in ("cold_start", "wide_start"):
        array_inputs[f"{start_name}.final_native_field_eV"] = (
            payload[start_name]["final_native_field_eV"],
            f"{start_name.removesuffix('_start')}-final-native-field8",
            H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
            H1_V2_RECEIVER_UNITS,
        )
        array_inputs[f"{start_name}.final_residual_eV"] = (
            payload[start_name]["final_residual_eV"],
            f"{start_name.removesuffix('_start')}-final-actual-residual-field8",
            H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
            H1_V2_RECEIVER_UNITS,
        )
    payload["array_content_sha256s"] = {
        name: canonical_numeric_array_sha256_v2(
            values,
            semantic_type=semantic_type,
            geometry_sha256=geometry,
            channel_space_contract_sha256=space_sha256,
            units=units,
        )
        for name, (values, semantic_type, space_sha256, units) in array_inputs.items()
    }
    return _bind(payload, "state_leaf_sha256")


def _refresh_state_leaf_array_content_sha256s(payload: dict[str, object]) -> None:
    source_arrays = {
        "permanent_point_source4": "permanent-point-source4",
        "polar_zero_reference_source4": (
            "polar-zero-reference-source4-nonoperational"
        ),
        "polar_final_source4": "polar-final-source4",
        "induced_gto_source4": "induced-gto1p5-source4",
        "audit_coefficient_sum4": "audit-coefficient-sum4-nonoperational",
    }
    digests = {
        name: canonical_numeric_array_sha256_v2(
            payload[name],
            semantic_type=semantic_type,
            geometry_sha256=payload["geometry_sha256"],
            channel_space_contract_sha256=H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            units=H1_V2_SOURCE_COEFFICIENT_UNITS,
        )
        for name, semantic_type in source_arrays.items()
    }
    for start_name in ("cold_start", "wide_start"):
        prefix = start_name.removesuffix("_start")
        start = payload[start_name]
        for field, semantic_suffix in (
            ("final_native_field_eV", "final-native-field8"),
            ("final_residual_eV", "final-actual-residual-field8"),
        ):
            key = f"{start_name}.{field}"
            digests[key] = canonical_numeric_array_sha256_v2(
                start[field],
                semantic_type=f"{prefix}-{semantic_suffix}",
                geometry_sha256=payload["geometry_sha256"],
                channel_space_contract_sha256=(
                    H1_V2_RECEIVER_SPACE_CONTRACT_SHA256
                ),
                units=H1_V2_RECEIVER_UNITS,
            )
    payload["array_content_sha256s"] = digests


def _solve_event_payload(leaf: StateLeafV2, index: int = 0) -> dict[str, object]:
    return _bind(
        {
            "schema_id": SOLVE_EVENT_V2_SCHEMA,
            "event_index": index,
            "state_leaf_sha256": leaf.state_leaf_sha256,
            "geometry_sha256": leaf.geometry_sha256,
            "provider_configuration_sha256": leaf.provider_configuration_sha256,
            "topology_id": leaf.topology_id,
        },
        "solve_event_sha256",
    )


def _prepared_inputs_payload(seal: ComputationSealV2) -> dict[str, object]:
    return _prepared_inputs_from_assets(dict(seal.asset_sha256s))


def _prepared_providers_payload(seal: ComputationSealV2) -> dict[str, object]:
    return {
        "profile_id": seal.profile_id,
        "scalar_id": seal.scalar_id,
        "state_id": seal.state_id,
        "checkpoint_sha256s": dict(seal.checkpoint_sha256s),
        "provider_configuration_sha256s": dict(seal.provider_configuration_sha256s),
    }


def _rich_measurement_payload(seal: ComputationSealV2) -> dict[str, object]:
    prepared_providers = _prepared_providers_payload(seal)
    water_raw, benzene_raw, solve_events, state_leaves = _water_system_fixture()
    benzene = BenzeneSystemV2.from_mapping(
        benzene_raw,
        solve_events=solve_events,
        state_leaves=state_leaves,
        prepared_pes_configuration_sha256="6" * 64,
    )
    water = WaterSystemV2.from_mapping(
        water_raw,
        solve_events=solve_events,
        state_leaves=state_leaves,
        prepared_water_positions=WATER_POSITIONS,
        prepared_pes_configuration_sha256="7" * 64,
    )
    metrics = admission_contracts._reported_metrics(benzene, water)
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
    gates = admission_contracts._replicate_gates(
        metrics,
        seal=seal,
        force_panel=_prepared_inputs_payload(seal)["force_panel_contract"],
    )
    return {
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
                seal.as_dict()["scientific_settings"]
            ),
            "fit_calibration_or_case_selection": False,
        },
        "panel_contract_sha256": FORCE_PANEL_SHA,
        "provider_ledger_sha256": canonical_json_sha256(prepared_providers),
        "prepared_inputs": _prepared_inputs_payload(seal),
        "prepared_providers": prepared_providers,
        "state_leaves": {
            digest: leaf.as_dict() for digest, leaf in state_leaves.items()
        },
        "solve_events": [
            {
                "schema_id": event.schema_id,
                "solve_event_sha256": event.solve_event_sha256,
                "event_index": event.event_index,
                "state_leaf_sha256": event.state_leaf_sha256,
                "geometry_sha256": event.geometry_sha256,
                "provider_configuration_sha256": (event.provider_configuration_sha256),
                "topology_id": event.topology_id,
            }
            for event in solve_events
        ],
        "systems": {"benzene": benzene_raw, "water": water_raw},
        "reported_metrics": metrics,
        "reported_local_gate_results": gates,
        "local_decision": {
            "candidate_energy_force_gate_passed": True,
            "awaiting_independent_replay": True,
            "public_capability_admitted": False,
            "chemical_accuracy_admitted": False,
            "complete_solvation_free_energy_admitted": False,
            "analytic_force_admitted": False,
            "hessian_frequency_md_admitted": False,
            "tier_v_admitted": False,
        },
    }


def _rich_scaffold_replicate_payload(
    seal: ComputationSealV2,
) -> dict[str, object]:
    return _replicate_payload(seal, "a")


def _replicate_payload(seal: ComputationSealV2, label: str) -> dict[str, object]:
    suffix = "a" if label == "a" else "b"
    measurement = _rich_measurement_payload(seal)
    return _bind(
        {
            "schema_id": REPLICATE_ADMISSION_RECORD_V2_SCHEMA,
            "artifact_id": f"replicate-{suffix}",
            "label": label,
            "process_uuid": (
                "11111111-1111-4111-8111-111111111111"
                if label == "a"
                else "22222222-2222-4222-8222-222222222222"
            ),
            "process_started_at_utc": (
                "2026-08-23T01:00:00+00:00"
                if label == "a"
                else "2026-08-23T02:00:00+00:00"
            ),
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "git_head": seal.git_head,
            "git_tree": seal.git_tree,
            "source_ledger_sha256": seal.source_ledger_sha256,
            "asset_ledger_sha256": seal.asset_ledger_sha256,
            "runtime_fingerprint_sha256": seal.runtime_fingerprint_sha256,
            "measurement_schema_id": MEASUREMENT_SCHEMA_ID,
            "measurement_sha256": canonical_json_sha256(measurement),
            "measurement": measurement,
            "gate_results": {name: True for name in REPLICATE_GATE_NAMES},
        },
        "artifact_sha256",
    )


def _aggregate_payload(seal: ComputationSealV2) -> dict[str, object]:
    return _bind(
        {
            "schema_id": AGGREGATE_ADMISSION_INPUTS_V2_SCHEMA,
            "aggregate_id": "route2-v2-aggregate",
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "profile_id": seal.profile_id,
            "scalar_id": seal.scalar_id,
            "state_id": seal.state_id,
            "capabilities": {
                "E": True,
                "F": True,
                "H": False,
                "V": False,
                "M": False,
            },
            "runtime_guards": list(seal.runtime_guards),
            "domain_guards": list(seal.domain_guards),
            "claim_boundary_id": seal.claim_boundary_id,
            "non_admissions": list(seal.non_admissions),
            "replicates": [
                _replicate_payload(seal, "a"),
                _replicate_payload(seal, "b"),
            ],
            "aggregate_gate_results": {name: True for name in AGGREGATE_GATE_NAMES},
        },
        "aggregate_sha256",
    )


def _aggregate(seal: ComputationSealV2) -> AggregateAdmissionInputsV2:
    return AggregateAdmissionInputsV2.from_mapping(_aggregate_payload(seal), seal=seal)


def _overlay_payload(
    seal: ComputationSealV2, aggregate: AggregateAdmissionInputsV2
) -> dict[str, object]:
    return _overlay_payload_from_values(
        seal,
        {
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
        },
    )


def _overlay_payload_from_values(
    seal: ComputationSealV2, aggregate: dict[str, object]
) -> dict[str, object]:
    capabilities = aggregate["capabilities"]
    if isinstance(capabilities, CapabilityStatus):
        capability_payload = {
            "E": capabilities.energy,
            "F": capabilities.conservative_force,
            "H": capabilities.hessian,
            "V": capabilities.variational_functional,
            "M": capabilities.molecular_dynamics,
        }
    else:
        capability_payload = capabilities
    return _bind(
        {
            "schema_id": ADMISSION_OVERLAY_V2_SCHEMA,
            "overlay_id": "route2-v2-ef-overlay",
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "aggregate_id": aggregate["aggregate_id"],
            "aggregate_sha256": aggregate["aggregate_sha256"],
            "profile_id": aggregate["profile_id"],
            "scalar_id": aggregate["scalar_id"],
            "state_id": aggregate["state_id"],
            "capabilities": capability_payload,
            "runtime_guards": list(aggregate["runtime_guards"]),
            "domain_guards": list(aggregate["domain_guards"]),
            "claim_boundary_id": aggregate["claim_boundary_id"],
            "non_admissions": list(aggregate["non_admissions"]),
        },
        "content_sha256",
    )


def test_state_leaf_and_solve_event_exact_round_trip() -> None:
    payload = _state_leaf_payload()
    leaf = StateLeafV2.from_mapping(json.loads(json.dumps(payload)))
    assert leaf.as_dict() == payload
    event_payload = _solve_event_payload(leaf)
    event = SolveEventV2.from_mapping(
        event_payload,
        state_leaves={leaf.state_leaf_sha256: leaf},
        expected_index=0,
    )
    assert event.solve_event_sha256 == event_payload["solve_event_sha256"]


def test_state_leaf_accepts_finite_wide_replay_difference_and_derives_audits() -> None:
    payload = _state_leaf_payload()
    payload["wide_start"]["final_total_energy_eV"] = (
        payload["total_energy_eV"] + 1.0e-12
    )
    payload["wide_start"]["final_polarization_energy_eV"] += 1.0e-12
    payload["wide_start"]["final_native_field_eV"][1][3] = 2.0e-12
    _refresh_state_leaf_array_content_sha256s(payload)
    _rehash(payload, "state_leaf_sha256")

    leaf = StateLeafV2.from_mapping(payload)

    assert leaf.cross_start_energy_abs_difference_eV == abs(
        payload["cold_start"]["final_total_energy_eV"]
        - payload["wide_start"]["final_total_energy_eV"]
    )
    assert leaf.cross_start_field_max_abs_difference_eV == 2.0e-12
    assert "cross_start_energy_abs_difference_eV" not in leaf.as_dict()
    assert "cross_start_field_max_abs_difference_eV" not in leaf.as_dict()


def test_state_leaf_rejects_cold_start_energy_mismatch() -> None:
    payload = _state_leaf_payload()
    payload["cold_start"]["final_total_energy_eV"] = (
        payload["total_energy_eV"] + 1.0e-12
    )
    _rehash(payload, "state_leaf_sha256")
    with pytest.raises(ValueError, match="cold_start total energy"):
        StateLeafV2.from_mapping(payload)


def test_state_leaf_uses_native_l1_source_basis_and_split_kernel_audit_sum() -> None:
    payload = _state_leaf_payload()
    payload["polar_zero_reference_source4"][0][0] = 0.75
    payload["polar_final_source4"][0][0] = 0.5
    _refresh_state_leaf_array_content_sha256s(payload)
    _rehash(payload, "state_leaf_sha256")

    leaf = StateLeafV2.from_mapping(payload)

    assert leaf.source_coefficient_basis_id == H1_V2_SOURCE_COEFFICIENT_BASIS_ID
    assert leaf.source_coefficient_order.items == H1_V2_SOURCE_COEFFICIENT_ORDER
    assert leaf.audit_projection_contract == admission_contracts._freeze_object(
        h1_v2_audit_coefficient_sum_contract(), name="audit projection"
    )
    assert len(leaf.permanent_point_source4.items[0].items) == 4
    assert leaf.audit_coefficient_sum4.items[0].items[0] == (
        leaf.permanent_point_source4.items[0].items[0]
        + leaf.induced_gto_source4.items[0].items[0]
    )
    assert leaf.polar_zero_reference_source4.items[0].items[0] == 0.75


def test_state_leaf_requires_frozen_final_minus_zero_subtraction_direction() -> None:
    payload = _state_leaf_payload()
    zero = 1.799707382720902
    asserted_induced = 1.1441658720372287
    rounded_final = zero + asserted_induced
    payload["polar_zero_reference_source4"][0][1] = zero
    payload["polar_final_source4"][0][1] = rounded_final
    payload["induced_gto_source4"][0][1] = asserted_induced
    payload["audit_coefficient_sum4"][0][1] = (
        payload["permanent_point_source4"][0][1] + asserted_induced
    )
    assert rounded_final - zero != asserted_induced
    _refresh_state_leaf_array_content_sha256s(payload)
    _rehash(payload, "state_leaf_sha256")
    with pytest.raises(ValueError, match="frozen subtraction"):
        StateLeafV2.from_mapping(payload)


def test_audit_projection_contract_factory_is_nested_mutation_safe() -> None:
    first = h1_v2_audit_coefficient_sum_contract()
    first["derived_from"][0] = "tampered"
    assert h1_v2_audit_coefficient_sum_contract() == {
        "semantic_role": "algebraic_audit_projection",
        "physical_kernel": None,
        "operator_dispatch": "forbidden",
        "derived_from": ["permanent_point_source4", "induced_gto_source4"],
    }


def test_canonical_monopole_sum_avoids_backend_reduction_rounding_drift() -> None:
    source = [[1.0e16, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [-1.0e16, 0.0, 0.0, 0.0]]
    assert float(np.sum(np.asarray(source)[:, 0])) == 0.0
    assert canonical_source_monopole_sum_v2(source) == 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_coefficient_basis_id", "undocumented-cartesian-q-px-py-pz"),
        (
            "source_coefficient_order",
            ["net_monopole", "cartesian_px", "cartesian_py", "cartesian_pz"],
        ),
        ("source_space_contract_sha256", "f" * 64),
        ("receiver_space_contract_sha256", "e" * 64),
        (
            "audit_projection_contract",
            {
                **h1_v2_audit_coefficient_sum_contract(),
                "operator_dispatch": "allowed",
            },
        ),
    ),
)
def test_state_leaf_rejects_source_category_reinterpretation(
    field: str, value: object
) -> None:
    payload = _state_leaf_payload()
    payload[field] = value
    _rehash(payload, "state_leaf_sha256")
    with pytest.raises(ValueError):
        StateLeafV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("cold_start", "final_native_field_eV"), [[0.0] * 7] * 3),
        (("wide_start", "final_residual_eV"), [[0.0] * 8] * 2),
        (("cold_start", "iterations"), 0),
        (("cold_start", "converged"), 1),
        (("audit_coefficient_sum_charge_e",), 1.0),
        (("total_energy_eV",), float("nan")),
        (
            ("audit_coefficient_sum4",),
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] + [[0.0] * 9] * 2,
        ),
    ),
)
def test_state_leaf_shape_type_and_derived_binding_tamper_fails(
    path: tuple[str, ...], value: object
) -> None:
    payload = _state_leaf_payload()
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    if path != ("total_energy_eV",):
        _rehash(payload, "state_leaf_sha256")
    with pytest.raises((TypeError, ValueError)):
        StateLeafV2.from_mapping(payload)


@pytest.mark.parametrize("operation", ("missing", "unknown", "digest"))
def test_state_leaf_exact_fields_and_non_circular_digest(operation: str) -> None:
    payload = _state_leaf_payload()
    if operation == "missing":
        payload.pop("topology_id")
    elif operation == "unknown":
        payload["reported_residual_norm"] = 0.0
    else:
        payload["state_leaf_sha256"] = SHA_F
    with pytest.raises(ValueError):
        StateLeafV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("event_index", 1, "execution order"),
        ("state_leaf_sha256", SHA_F, "unknown state leaf"),
        ("geometry_sha256", SHA_F, "geometry_sha256"),
    ),
)
def test_solve_event_order_reference_and_leaf_bindings_fail_closed(
    field: str, value: object, message: str
) -> None:
    leaf = StateLeafV2.from_mapping(_state_leaf_payload())
    payload = _solve_event_payload(leaf)
    payload[field] = value
    _rehash(payload, "solve_event_sha256")
    with pytest.raises(ValueError, match=message):
        SolveEventV2.from_mapping(
            payload,
            state_leaves={leaf.state_leaf_sha256: leaf},
            expected_index=0,
        )


def test_legacy_measurement_is_explicitly_not_a_v2_admission_preimage() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement_schema_id"] = LEGACY_MEASUREMENT_SCHEMA_ID
    payload["measurement"]["schema_id"] = LEGACY_MEASUREMENT_SCHEMA_ID
    payload["measurement_sha256"] = canonical_json_sha256(payload["measurement"])
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="historical v1 measurement scope"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_rich_schema_id_is_public_h1_measurement_identity() -> None:
    assert MEASUREMENT_SCHEMA_ID == RICH_MEASUREMENT_SCHEMA_ID


def test_prepared_inputs_are_exact_cross_bound_and_deeply_frozen() -> None:
    seal = _seal()
    frozen = admission_contracts._prepared_inputs(
        _prepared_inputs_payload(seal),
        seal=seal,
        panel_contract_sha256=FORCE_PANEL_SHA,
    )
    normalized = admission_contracts._thaw_json(frozen)
    assert normalized["benzene"]["compound_id"] == "mobley_3053621"
    assert normalized["benzene"]["name"] == "benzene"
    assert len(normalized["benzene"]["positions_angstrom"]) == 12
    assert normalized["water"]["atomic_numbers"] == [8, 1, 1]
    assert normalized["water"]["positions_angstrom"] == [
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.239, 0.9266, 0.0],
    ]
    assert normalized["force_panel_contract"] == _force_panel_payload()
    assert normalized["force_panel_contract_sha256"] == FORCE_PANEL_SHA
    with pytest.raises(FrozenInstanceError):
        frozen.items = ()


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("gepol_regression_compound_id",), "other", "frozen preregistration"),
        (("gepol_regression_cartesian_dof",), [0, 1], "frozen preregistration"),
        (("water_geometry_angstrom", 1, 0), 0.9573, "frozen preregistration"),
        (("water_full_cartesian_force",), 1, "bool"),
        (("independent_direction_seed",), True, "nonnegative integer"),
        (
            ("maximum_independent_directional_error_ev_per_angstrom",),
            "0.0005",
            "positive number",
        ),
        (
            ("maximum_translation_energy_error_ev",),
            2.0e-7,
            "frozen preregistration",
        ),
        (("translation_angstrom",), [4.2, -3.1, 1.8], "frozen preregistration"),
        (("rotation_seed",), 20260818, "frozen preregistration"),
        (("closed_loop_cartesian_dofs",), [[1, 1], [0, 0]], "frozen preregistration"),
    ),
)
def test_force_panel_contract_rejects_type_shape_order_and_policy_drift(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    payload = _force_panel_payload()
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        admission_contracts._force_panel_contract(payload)


def test_force_panel_contract_rejects_unknown_and_missing_fields() -> None:
    unknown = _force_panel_payload()
    unknown["caller_threshold"] = 1.0
    with pytest.raises(ValueError, match="unknown fields"):
        admission_contracts._force_panel_contract(unknown)
    missing = _force_panel_payload()
    del missing["maximum_closed_loop_work_abs_ev"]
    with pytest.raises(ValueError, match="missing fields"):
        admission_contracts._force_panel_contract(missing)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("benzene", "name"), "Benzene", "compound identity"),
        (("benzene", "atomic_numbers", 0), True, "JSON integer"),
        (("benzene", "atomic_numbers", 0), 7, "sealed manifest"),
        (("benzene", "positions_angstrom", 0, 0), True, "JSON number"),
        (("benzene", "positions_angstrom", 0, 0), "0.0", "JSON number"),
        (("benzene", "positions_angstrom", 0, 0), float("nan"), "finite"),
        (("benzene", "positions_angstrom", 0, 0), 99.0, "sealed manifest"),
        (("benzene", "cavity_radii_angstrom", 0), 0.0, "positive"),
        (("benzene", "cavity_radii_angstrom", 0), 99.0, "sealed manifest"),
        (("benzene", "predecessor_dof"), [0, 1], "predecessor_dof"),
        (("benzene", "mol2_sha256"), SHA_F, "mol2_sha256"),
        (("benzene", "projection_result_sha256"), SHA_F, "projection_result"),
        (("water", "atomic_numbers"), [1, 8, 1], "atomic_numbers"),
        (("water", "positions_angstrom", 1, 0), 0.9573, "geometry drifted"),
        (("water", "charge"), False, "charge"),
        (("water", "cavity_radii_angstrom", 0), "1.5", "positive"),
        (("water", "cavity_radii_angstrom", 0), 99.0, "sealed manifest"),
        (("v1_preregistration_sha256",), SHA_F, "v1_preregistration"),
        (("parent_panel_sha256",), SHA_F, "parent_panel"),
        (("force_panel_contract_sha256",), SHA_F, "force_panel_contract"),
    ),
)
def test_prepared_inputs_reject_type_shape_order_and_hash_drift(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    seal = _seal()
    payload = _prepared_inputs_payload(seal)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        admission_contracts._prepared_inputs(
            payload,
            seal=seal,
            panel_contract_sha256=FORCE_PANEL_SHA,
        )


def test_prepared_inputs_reject_unknown_missing_and_wrong_shapes() -> None:
    seal = _seal()
    unknown = _prepared_inputs_payload(seal)
    unknown["extra"] = None
    with pytest.raises(ValueError, match="unknown fields"):
        admission_contracts._prepared_inputs(
            unknown, seal=seal, panel_contract_sha256=FORCE_PANEL_SHA
        )

    missing = _prepared_inputs_payload(seal)
    del missing["water"]["multiplicity"]
    with pytest.raises(ValueError, match="missing fields"):
        admission_contracts._prepared_inputs(
            missing, seal=seal, panel_contract_sha256=FORCE_PANEL_SHA
        )

    wrong_shape = _prepared_inputs_payload(seal)
    wrong_shape["benzene"]["positions_angstrom"] = [[0.0, 0.0, 0.0]] * 11
    with pytest.raises(ValueError, match="12 rows"):
        admission_contracts._prepared_inputs(
            wrong_shape, seal=seal, panel_contract_sha256=FORCE_PANEL_SHA
        )


@pytest.mark.parametrize(
    ("provider_key", "message"),
    (
        ("prepared_input_manifest", "sealed manifest"),
        ("force_panel_contract", "sealed force-panel contract"),
    ),
)
def test_prepared_inputs_reject_sealed_provider_digest_tamper(
    provider_key: str, message: str
) -> None:
    seal_payload = _seal_payload()
    seal_payload["provider_configuration_sha256s"][provider_key] = SHA_F
    _rehash(seal_payload, "content_sha256")
    seal = ComputationSealV2.from_mapping(seal_payload)
    with pytest.raises(ValueError, match=message):
        admission_contracts._prepared_inputs(
            _prepared_inputs_payload(seal),
            seal=seal,
            panel_contract_sha256=FORCE_PANEL_SHA,
        )


def test_prepared_and_measurement_force_panel_hashes_must_match() -> None:
    seal = _seal()
    with pytest.raises(ValueError, match="force_panel_contract_sha256"):
        admission_contracts._prepared_inputs(
            _prepared_inputs_payload(seal),
            seal=seal,
            panel_contract_sha256=SHA_F,
        )


def test_prepared_providers_are_exact_cross_bound_and_deeply_frozen() -> None:
    seal = _seal()
    frozen = admission_contracts._prepared_providers(
        _prepared_providers_payload(seal), seal=seal
    )
    normalized = admission_contracts._thaw_json(frozen)
    assert normalized == _prepared_providers_payload(seal)
    assert canonical_json_sha256(normalized) == canonical_json_sha256(
        _prepared_providers_payload(seal)
    )
    with pytest.raises(FrozenInstanceError):
        frozen.items = ()


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("profile_id",), "wrong-profile", "does not equal"),
        (("checkpoint_sha256s", "mace_mdp_checkpoint"), SHA_F, "does not equal"),
        (("checkpoint_sha256s", "mace_mdp_checkpoint"), 7, "trimmed string"),
        (
            ("provider_configuration_sha256s", "hybrid_configuration"),
            SHA_F,
            "does not equal",
        ),
        (
            ("provider_configuration_sha256s", "hybrid_configuration"),
            7.0,
            "trimmed string",
        ),
    ),
)
def test_prepared_providers_reject_identity_ledger_and_type_tamper(
    path: tuple[str, ...], replacement: object, message: str
) -> None:
    seal = _seal()
    payload = _prepared_providers_payload(seal)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        admission_contracts._prepared_providers(payload, seal=seal)


@pytest.mark.parametrize(
    "ledger_name", ("checkpoint_sha256s", "provider_configuration_sha256s")
)
def test_prepared_providers_reject_digest_aliases(ledger_name: str) -> None:
    seal = _seal()
    payload = _prepared_providers_payload(seal)
    ledger = payload[ledger_name]
    first, second = list(ledger)[:2]
    ledger[second] = ledger[first]
    with pytest.raises(ValueError, match="digest aliases"):
        admission_contracts._prepared_providers(payload, seal=seal)


def test_prepared_providers_reject_cross_ledger_digest_alias() -> None:
    seal = _seal()
    payload = _prepared_providers_payload(seal)
    payload["provider_configuration_sha256s"]["hybrid_configuration"] = payload[
        "checkpoint_sha256s"
    ]["mace_mdp_checkpoint"]
    with pytest.raises(ValueError, match="cross-ledger digest aliases"):
        admission_contracts._prepared_providers(payload, seal=seal)


def test_prepared_providers_reject_unknown_and_missing_fields() -> None:
    seal = _seal()
    unknown = _prepared_providers_payload(seal)
    unknown["alias"] = unknown["profile_id"]
    with pytest.raises(ValueError, match="unknown fields"):
        admission_contracts._prepared_providers(unknown, seal=seal)
    missing = _prepared_providers_payload(seal)
    del missing["checkpoint_sha256s"]["mace_polar_checkpoint"]
    with pytest.raises(ValueError, match="missing fields"):
        admission_contracts._prepared_providers(missing, seal=seal)


def test_measurement_provider_ledger_digest_is_derived_not_caller_chosen() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"]["provider_ledger_sha256"] = SHA_F
    payload["measurement_sha256"] = canonical_json_sha256(payload["measurement"])
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="provider_ledger_sha256"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_complete_rich_tree_builds_typed_replicate_after_protocol_amendment() -> None:
    seal = _seal()
    payload = _rich_scaffold_replicate_payload(seal)
    record = ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)
    assert record.as_dict() == payload


def test_data_only_candidate_builder_is_deterministic_copy_safe_and_preterminal_exact() -> (
    None
):
    seal = _seal()
    expected = _rich_measurement_payload(seal)
    recording = {
        key: json.loads(json.dumps(expected[key]))
        for key in ("state_leaves", "solve_events", "systems")
    }
    candidate, gates = build_rich_harmonic_ef_measurement_candidate_v2(
        seal=seal,
        prepared_inputs=expected["prepared_inputs"],
        prepared_providers=expected["prepared_providers"],
        recording=recording,
    )
    second, second_gates = build_rich_harmonic_ef_measurement_candidate_v2(
        seal=seal,
        prepared_inputs=expected["prepared_inputs"],
        prepared_providers=expected["prepared_providers"],
        recording=recording,
    )
    assert candidate == expected
    assert second == candidate
    assert gates == second_gates == expected["reported_local_gate_results"]
    recording["systems"]["water"]["event_range"] = [0, 1]
    candidate["reported_metrics"]["water_closed_loop_work_eV"] = 99.0
    assert second == expected
    normalized, normalized_gates = admission_contracts._measurement(expected, seal=seal)
    assert admission_contracts._thaw_json(normalized) == expected
    assert normalized_gates == expected["reported_local_gate_results"]


def _execution_failure_payload(seal: ComputationSealV2) -> dict[str, object]:
    return _bind(
        {
            "schema_id": EXECUTION_FAILURE_V2_SCHEMA,
            "artifact_id": "route2-h1-replicate-a-failure",
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "label": "a",
            "process_uuid": "11111111-1111-4111-8111-111111111111",
            "process_started_at_utc": "2026-08-23T01:00:00+00:00",
            "stage": "rich-v2-terminal",
            "exception_type": "RuntimeError",
            "exception_message": "synthetic failure",
            "exception_message_sha256": canonical_json_sha256("synthetic failure"),
            "exception_message_truncated": False,
            "available_partial_evidence": {
                "captured_sha256s": {
                    role: SHA_A
                    for role in (
                        "seal",
                        "preregistration",
                        "v1_preregistration",
                        "parent_panel",
                        "benzene_mol2",
                        "benzene_projection",
                        "mace_mdp_checkpoint",
                        "mace_polar_checkpoint",
                        "runner",
                    )
                },
                "runtime_fingerprint_sha256": seal.runtime_fingerprint_sha256,
                "recording_sha256": SHA_B,
                "state_leaf_count": 12,
                "solve_event_count": 141,
                "system_keys": ["benzene", "water"],
                "system_event_ranges": {"benzene": [0, 8], "water": [8, 141]},
                "candidate_measurement_sha256": SHA_C,
                "derived_gate_results": {name: True for name in REPLICATE_GATE_NAMES},
            },
            "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
            "claim_boundary_id": seal.claim_boundary_id,
            "non_admissions": list(seal.non_admissions),
        },
        "artifact_sha256",
    )


def test_execution_failure_exact_round_trip_and_deep_copy() -> None:
    seal = _seal()
    payload = _execution_failure_payload(seal)
    failure = ExecutionFailureV2.from_mapping(payload, seal=seal)
    assert failure.as_dict() == payload
    payload["available_partial_evidence"]["state_leaf_count"] = 999
    assert failure.as_dict()["available_partial_evidence"]["state_leaf_count"] == 12
    assert failure.capabilities == CapabilityStatus()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("label", "c", "label"),
        ("seal_sha256", SHA_F, "cross-binding"),
        ("stage", "water-rotation", "unsupported"),
        ("exception_message_sha256", SHA_F, "digest"),
        ("artifact_sha256", SHA_F, "artifact_sha256"),
        (
            "capabilities",
            {"E": True, "F": False, "H": False, "V": False, "M": False},
            "all be false",
        ),
    ),
)
def test_execution_failure_rejects_tamper_and_capability_claims(
    field: str, replacement: object, message: str
) -> None:
    seal = _seal()
    payload = _execution_failure_payload(seal)
    payload[field] = replacement
    if field != "artifact_sha256":
        _rehash(payload, "artifact_sha256")
    with pytest.raises((TypeError, ValueError), match=message):
        ExecutionFailureV2.from_mapping(payload, seal=seal)


def test_execution_failure_rejects_unknown_fields_and_aggregate_use() -> None:
    seal = _seal()
    payload = _execution_failure_payload(seal)
    unknown = dict(payload)
    unknown["aggregate_admitted"] = False
    with pytest.raises(ValueError, match="unknown fields"):
        ExecutionFailureV2.from_mapping(unknown, seal=seal)
    failure = ExecutionFailureV2.from_mapping(payload, seal=seal)
    with pytest.raises(ValueError):
        ReplicateAdmissionRecordV2.from_mapping(failure.as_dict(), seal=seal)
    aggregate = _aggregate_payload(seal)
    aggregate["replicates"][0] = failure.as_dict()
    _rehash(aggregate, "aggregate_sha256")
    with pytest.raises(ValueError):
        AggregateAdmissionInputsV2.from_mapping(aggregate, seal=seal)


def test_execution_failure_rejects_unbounded_or_raw_partial_evidence() -> None:
    seal = _seal()
    payload = _execution_failure_payload(seal)
    payload["available_partial_evidence"]["raw_state_leaves"] = {}
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="unknown fields"):
        ExecutionFailureV2.from_mapping(payload, seal=seal)

    payload = _execution_failure_payload(seal)
    payload["exception_message"] = "x" * 4097
    payload["exception_message_sha256"] = canonical_json_sha256("x" * 4097)
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="too long"):
        ExecutionFailureV2.from_mapping(payload, seal=seal)

    payload = _execution_failure_payload(seal)
    payload["exception_type"] = "X" * 40000
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="size limit"):
        ExecutionFailureV2.from_mapping(payload, seal=seal)


def _component_fixture():
    energies = (-1.25, -1.249, -1.251, -1.2495, -1.2505)
    geometries = (SHA_A, SHA_B, SHA_C, SHA_D, SHA_E)
    leaves = tuple(
        StateLeafV2.from_mapping(
            _state_leaf_payload(total_energy=energy, geometry=geometry)
        )
        for energy, geometry in zip(energies, geometries, strict=True)
    )
    leaf_map = {leaf.state_leaf_sha256: leaf for leaf in leaves}
    events = tuple(
        SolveEventV2.from_mapping(
            _solve_event_payload(leaf, index),
            state_leaves=leaf_map,
            expected_index=index,
        )
        for index, leaf in enumerate(leaves)
    )
    raw = {
        "atom_index": 0,
        "axis_index": 0,
        "coarse_step_angstrom": 1.0e-3,
        "fine_step_angstrom": 5.0e-4,
        "center_event_index": 0,
        "plus_h_event_index": 1,
        "minus_h_event_index": 2,
        "plus_h2_event_index": 3,
        "minus_h2_event_index": 4,
    }
    return raw, events, leaf_map


def test_component_stencil_derives_richardson_force_from_raw_state_energies() -> None:
    raw, events, leaves = _component_fixture()
    stencil = ComponentStencilV2.from_mapping(
        raw, solve_events=events, state_leaves=leaves
    )
    assert stencil.event_indices == (0, 1, 2, 3, 4)
    assert stencil.richardson_force_eV_per_angstrom == pytest.approx(-1.0)
    assert stencil.local_error_eV_per_angstrom < 2.0e-13


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fine_step_angstrom", 4.0e-4),
        ("axis_index", 3),
        ("minus_h2_event_index", 3),
        ("plus_h_event_index", 99),
    ),
)
def test_component_stencil_order_step_and_range_tamper_fails(
    field: str, value: object
) -> None:
    raw, events, leaves = _component_fixture()
    raw[field] = value
    with pytest.raises(ValueError):
        ComponentStencilV2.from_mapping(raw, solve_events=events, state_leaves=leaves)


def test_benzene_h4_stencil_is_exact_and_audit_only() -> None:
    raw, events, leaves = _component_fixture()
    event_list, leaf_map = list(events), dict(leaves)
    for energy in (-1.24975, -1.25025):
        index = len(event_list)
        leaf = StateLeafV2.from_mapping(
            _state_leaf_payload(total_energy=energy, geometry=f"{index + 20:064x}")
        )
        leaf_map[leaf.state_leaf_sha256] = leaf
        event_list.append(
            SolveEventV2.from_mapping(
                _solve_event_payload(leaf, index),
                state_leaves=leaf_map,
                expected_index=index,
            )
        )
    rich = {
        **raw,
        "independent_step_angstrom": 2.5e-4,
        "plus_h4_event_index": 5,
        "minus_h4_event_index": 6,
        "audit_only": True,
    }
    stencil = BenzenePredecessorStencilV2.from_mapping(
        rich, solve_events=tuple(event_list), state_leaves=leaf_map
    )
    assert stencil.audit_only is True
    assert stencil.independent_h4_force_difference_eV_per_angstrom < 1.0e-12
    for field, value in (
        ("audit_only", False),
        ("independent_step_angstrom", 3.0e-4),
        ("minus_h4_event_index", 5),
    ):
        with pytest.raises(ValueError):
            BenzenePredecessorStencilV2.from_mapping(
                {**rich, field: value},
                solve_events=tuple(event_list),
                state_leaves=leaf_map,
            )


def test_cartesian_panel_requires_exact_order_label_and_shared_center() -> None:
    raw, events, leaves = _component_fixture()
    components = []
    for index in range(9):
        atom, axis = divmod(index, 3)
        components.append({**raw, "atom_index": atom, "axis_index": axis})
    panel_raw = {"label": "base", "components": components}
    panel = CartesianPanelV2.from_mapping(
        panel_raw, solve_events=events, state_leaves=leaves
    )
    assert len(panel.components) == 9
    for bad in ("label", "count", "order"):
        candidate = json.loads(json.dumps(panel_raw))
        if bad == "label":
            candidate["label"] = "translation"
        elif bad == "count":
            candidate["components"].pop()
        else:
            candidate["components"][0], candidate["components"][1] = (
                candidate["components"][1],
                candidate["components"][0],
            )
        with pytest.raises(ValueError):
            CartesianPanelV2.from_mapping(
                candidate, solve_events=events, state_leaves=leaves
            )


WATER_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.9572, 0.0, 0.0),
    (-0.239, 0.9266, 0.0),
)


def _panel_fixture(label: str, center_positions: np.ndarray):
    center_geometry = admission_contracts._ase_geometry_sha256(
        (8, 1, 1),
        tuple(tuple(float(item) for item in row) for row in center_positions),
    )
    leaf_map: dict[str, StateLeafV2] = {}
    events: list[SolveEventV2] = []
    center = _append_event(events, leaf_map, energy=-1.25, geometry=center_geometry)
    components = []
    coarse = 1.0e-3
    fine = 5.0e-4
    for atom in range(3):
        for axis in range(3):
            displaced_indices = []
            for displacement, energy in (
                (coarse, -1.249),
                (-coarse, -1.251),
                (fine, -1.2495),
                (-fine, -1.2505),
            ):
                positions = np.array(center_positions, copy=True)
                positions[atom, axis] += displacement
                displaced_indices.append(
                    _append_event(
                        events,
                        leaf_map,
                        energy=energy,
                        geometry=admission_contracts._ase_geometry_sha256(
                            (8, 1, 1),
                            tuple(
                                tuple(float(item) for item in row) for row in positions
                            ),
                        ),
                    )
                )
            components.append(
                {
                    "atom_index": atom,
                    "axis_index": axis,
                    "coarse_step_angstrom": coarse,
                    "fine_step_angstrom": fine,
                    "center_event_index": center,
                    "plus_h_event_index": displaced_indices[0],
                    "minus_h_event_index": displaced_indices[1],
                    "plus_h2_event_index": displaced_indices[2],
                    "minus_h2_event_index": displaced_indices[3],
                }
            )
    panel_raw = {
        "label": label,
        "components": components,
    }
    panel = CartesianPanelV2.from_mapping(
        panel_raw, solve_events=tuple(events), state_leaves=leaf_map
    )
    return panel_raw, panel, events, leaf_map


def _append_event(
    events: list[SolveEventV2],
    leaves: dict[str, StateLeafV2],
    *,
    energy: float,
    geometry: str,
    prepared_pes_configuration: str = "7" * 64,
) -> int:
    index = len(events)
    leaf = StateLeafV2.from_mapping(
        _state_leaf_payload(
            total_energy=energy,
            geometry=geometry,
            prepared_pes_configuration=prepared_pes_configuration,
        )
    )
    leaves[leaf.state_leaf_sha256] = leaf
    events.append(
        SolveEventV2.from_mapping(
            _solve_event_payload(leaf, index),
            state_leaves=leaves,
            expected_index=index,
        )
    )
    return index


def test_directional_check_regenerates_seeded_direction_and_derives_error() -> None:
    positions = np.asarray(WATER_POSITIONS)
    _, panel, events, leaves = _panel_fixture("base", positions)
    direction = np.random.default_rng(20260816).normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    step = 1.25e-4
    indices = []
    for sign, energy in ((1.0, -1.250125), (-1.0, -1.249875)):
        displaced = positions + sign * step * direction
        indices.append(
            _append_event(
                events,
                leaves,
                energy=energy,
                geometry=admission_contracts._ase_geometry_sha256(
                    (8, 1, 1),
                    tuple(tuple(float(item) for item in row) for row in displaced),
                ),
            )
        )
    raw = {
        "seed": 20260816,
        "normalized_direction": direction.tolist(),
        "step_angstrom": step,
        "plus_event_index": indices[0],
        "minus_event_index": indices[1],
    }
    check = DirectionalCheckV2.from_mapping(
        raw,
        solve_events=tuple(events),
        state_leaves=leaves,
        base_panel=panel,
        prepared_water_positions=WATER_POSITIONS,
    )
    assert check.energy_derivative_eV_per_angstrom == pytest.approx(-1.0)
    assert check.absolute_error_eV_per_angstrom == pytest.approx(
        abs(
            check.energy_derivative_eV_per_angstrom
            - check.negative_force_projection_eV_per_angstrom
        )
    )
    for field, replacement in (
        ("seed", 1),
        ("normalized_direction", np.flip(direction, axis=0).tolist()),
        ("plus_event_index", indices[1]),
        ("plus_event_index", 1),
    ):
        with pytest.raises(ValueError):
            DirectionalCheckV2.from_mapping(
                {**raw, field: replacement},
                solve_events=tuple(events),
                state_leaves=leaves,
                base_panel=panel,
                prepared_water_positions=WATER_POSITIONS,
            )


def test_rigid_translation_and_rotation_bind_transformed_geometry() -> None:
    positions = np.asarray(WATER_POSITIONS)
    translation = np.asarray([4.2, -3.1, 1.7])
    translated_raw, _, translated_events, translated_leaves = _panel_fixture(
        "translated", positions + translation
    )
    translated = RigidTranslationV2.from_mapping(
        {"translation_angstrom": translation.tolist(), "panel": translated_raw},
        solve_events=tuple(translated_events),
        state_leaves=translated_leaves,
        prepared_water_positions=WATER_POSITIONS,
    )
    assert translated.translation_angstrom == (4.2, -3.1, 1.7)

    rotation = admission_contracts._frozen_rotation(20260817)
    centroid = np.mean(positions, axis=0)
    rotated_positions = (positions - centroid) @ rotation.T + centroid
    rotated_raw, _, rotated_events, rotated_leaves = _panel_fixture(
        "rotated", rotated_positions
    )
    rotated = RigidRotationV2.from_mapping(
        {"seed": 20260817, "rotation_matrix": rotation.tolist(), "panel": rotated_raw},
        solve_events=tuple(rotated_events),
        state_leaves=rotated_leaves,
        prepared_water_positions=WATER_POSITIONS,
    )
    assert rotated.seed == 20260817
    with pytest.raises(ValueError, match="frozen vector"):
        RigidTranslationV2.from_mapping(
            {"translation_angstrom": [4.2, -3.1, 1.8], "panel": translated_raw},
            solve_events=tuple(translated_events),
            state_leaves=translated_leaves,
            prepared_water_positions=WATER_POSITIONS,
        )
    shifted_water = tuple((x + 0.01, y, z) for x, y, z in WATER_POSITIONS)
    with pytest.raises(ValueError, match="center geometry"):
        RigidTranslationV2.from_mapping(
            {"translation_angstrom": translation.tolist(), "panel": translated_raw},
            solve_events=tuple(translated_events),
            state_leaves=translated_leaves,
            prepared_water_positions=shifted_water,
        )
    bad_rotation = rotation.tolist()
    bad_rotation[0][0] += 1.0e-6
    with pytest.raises(ValueError, match="frozen QR"):
        RigidRotationV2.from_mapping(
            {"seed": 20260817, "rotation_matrix": bad_rotation, "panel": rotated_raw},
            solve_events=tuple(rotated_events),
            state_leaves=rotated_leaves,
            prepared_water_positions=WATER_POSITIONS,
        )


def _closed_loop_fixture():
    positions = np.asarray(WATER_POSITIONS)
    half_width = 1.0e-3
    contracts = (
        ("bottom", (0.0, -half_width), 0, 0, 2.0 * half_width),
        ("right", (half_width, 0.0), 1, 1, 2.0 * half_width),
        ("top", (0.0, half_width), 0, 0, -2.0 * half_width),
        ("left", (-half_width, 0.0), 1, 1, -2.0 * half_width),
    )
    events: list[SolveEventV2] = []
    leaves: dict[str, StateLeafV2] = {}
    edges = []
    for label, offsets, atom, axis, displacement in contracts:
        midpoint = np.array(positions, copy=True)
        midpoint[0, 0] += offsets[0]
        midpoint[1, 1] += offsets[1]
        center = _append_event(
            events,
            leaves,
            energy=-1.25,
            geometry=admission_contracts._ase_geometry_sha256(
                (8, 1, 1),
                tuple(tuple(float(item) for item in row) for row in midpoint),
            ),
        )
        displaced = []
        for component_displacement, energy in (
            (1.0e-3, -1.249),
            (-1.0e-3, -1.251),
            (5.0e-4, -1.2495),
            (-5.0e-4, -1.2505),
        ):
            displaced_positions = np.array(midpoint, copy=True)
            displaced_positions[atom, axis] += component_displacement
            displaced.append(
                _append_event(
                    events,
                    leaves,
                    energy=energy,
                    geometry=admission_contracts._ase_geometry_sha256(
                        (8, 1, 1),
                        tuple(
                            tuple(float(item) for item in row)
                            for row in displaced_positions
                        ),
                    ),
                )
            )
        component = {
            "atom_index": atom,
            "axis_index": axis,
            "coarse_step_angstrom": 1.0e-3,
            "fine_step_angstrom": 5.0e-4,
            "center_event_index": center,
            "plus_h_event_index": displaced[0],
            "minus_h_event_index": displaced[1],
            "plus_h2_event_index": displaced[2],
            "minus_h2_event_index": displaced[3],
        }
        edges.append(
            {
                "label": label,
                "midpoint_offsets_angstrom": list(offsets),
                "displacement_angstrom": displacement,
                "component": component,
            }
        )
    return (
        {
            "cartesian_dofs": [[0, 0], [1, 1]],
            "half_width_angstrom": half_width,
            "orientation": "counterclockwise",
            "edges": edges,
        },
        tuple(events),
        leaves,
    )


def test_closed_loop_derives_work_error_and_rejects_edge_contract_drift() -> None:
    raw, events, leaves = _closed_loop_fixture()
    loop = ClosedLoopV2.from_mapping(
        raw,
        solve_events=events,
        state_leaves=leaves,
        prepared_water_positions=WATER_POSITIONS,
    )
    assert loop.guarded_absolute_work_eV == pytest.approx(
        abs(loop.closed_loop_work_eV) + loop.numerical_work_error_bound_eV
    )
    for path, replacement in (
        (("orientation",), "clockwise"),
        (("cartesian_dofs",), [[1, 1], [0, 0]]),
        (("edges", 0, "label"), "top"),
        (("edges", 1, "midpoint_offsets_angstrom"), [0.0, 0.0]),
        (("edges", 2, "displacement_angstrom"), 2.0e-3),
    ):
        candidate = json.loads(json.dumps(raw))
        cursor = candidate
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = replacement
        with pytest.raises(ValueError):
            ClosedLoopV2.from_mapping(
                candidate,
                solve_events=events,
                state_leaves=leaves,
                prepared_water_positions=WATER_POSITIONS,
            )
    shifted_water = tuple((x, y + 0.01, z) for x, y, z in WATER_POSITIONS)
    with pytest.raises(ValueError, match="geometry"):
        ClosedLoopV2.from_mapping(
            raw,
            solve_events=events,
            state_leaves=leaves,
            prepared_water_positions=shifted_water,
        )


def _benzene_system_fixture():
    events: list[SolveEventV2] = []
    leaves: dict[str, StateLeafV2] = {}
    geometries = [SHA_A, SHA_B, SHA_C, SHA_D, SHA_E, SHA_F, "7" * 64, SHA_A]
    energies = [-1.25, -1.249, -1.251, -1.2495, -1.2505, -1.24975, -1.25025, -1.25]
    for geometry, energy in zip(geometries, energies, strict=True):
        _append_event(
            events,
            leaves,
            energy=energy,
            geometry=geometry,
            prepared_pes_configuration="6" * 64,
        )
    raw = {
        "event_range": [0, 8],
        "predecessor_stencil": {
            "atom_index": 0,
            "axis_index": 0,
            "coarse_step_angstrom": 1.0e-3,
            "fine_step_angstrom": 5.0e-4,
            "center_event_index": 0,
            "plus_h_event_index": 1,
            "minus_h_event_index": 2,
            "plus_h2_event_index": 3,
            "minus_h2_event_index": 4,
            "independent_step_angstrom": 2.5e-4,
            "plus_h4_event_index": 5,
            "minus_h4_event_index": 6,
            "audit_only": True,
        },
        "reported_center_event_index": 7,
    }
    return raw, events, leaves


def _append_panel_to_graph(
    events: list[SolveEventV2],
    leaves: dict[str, StateLeafV2],
    *,
    label: str,
    positions: np.ndarray,
    force_matrix: np.ndarray | None = None,
) -> dict[str, object]:
    center = _append_event(
        events,
        leaves,
        energy=-1.25,
        geometry=admission_contracts._ase_geometry_sha256(
            (8, 1, 1),
            tuple(tuple(float(item) for item in row) for row in positions),
        ),
    )
    components = []
    forces = (
        np.asarray(force_matrix, dtype=float)
        if force_matrix is not None
        else -np.ones((3, 3), dtype=float)
    )
    for atom in range(3):
        for axis in range(3):
            indices = []
            derivative = -float(forces[atom, axis])
            for displacement, energy in (
                (1.0e-3, -1.25 + 1.0e-3 * derivative),
                (-1.0e-3, -1.25 - 1.0e-3 * derivative),
                (5.0e-4, -1.25 + 5.0e-4 * derivative),
                (-5.0e-4, -1.25 - 5.0e-4 * derivative),
            ):
                displaced = np.array(positions, copy=True)
                displaced[atom, axis] += displacement
                indices.append(
                    _append_event(
                        events,
                        leaves,
                        energy=energy,
                        geometry=admission_contracts._ase_geometry_sha256(
                            (8, 1, 1),
                            tuple(
                                tuple(float(item) for item in row) for row in displaced
                            ),
                        ),
                    )
                )
            components.append(
                {
                    "atom_index": atom,
                    "axis_index": axis,
                    "coarse_step_angstrom": 1.0e-3,
                    "fine_step_angstrom": 5.0e-4,
                    "center_event_index": center,
                    "plus_h_event_index": indices[0],
                    "minus_h_event_index": indices[1],
                    "plus_h2_event_index": indices[2],
                    "minus_h2_event_index": indices[3],
                }
            )
    return {"label": label, "components": components}


def _append_loop_to_graph(
    events: list[SolveEventV2], leaves: dict[str, StateLeafV2], positions: np.ndarray
) -> dict[str, object]:
    half_width = 1.0e-3
    contracts = (
        ("bottom", (0.0, -half_width), 0, 0, 2.0 * half_width),
        ("right", (half_width, 0.0), 1, 1, 2.0 * half_width),
        ("top", (0.0, half_width), 0, 0, -2.0 * half_width),
        ("left", (-half_width, 0.0), 1, 1, -2.0 * half_width),
    )
    edges = []
    for label, offsets, atom, axis, edge_displacement in contracts:
        midpoint = np.array(positions, copy=True)
        midpoint[0, 0] += offsets[0]
        midpoint[1, 1] += offsets[1]
        center = _append_event(
            events,
            leaves,
            energy=-1.25,
            geometry=admission_contracts._ase_geometry_sha256(
                (8, 1, 1),
                tuple(tuple(float(item) for item in row) for row in midpoint),
            ),
        )
        indices = []
        for displacement, energy in (
            (1.0e-3, -1.249),
            (-1.0e-3, -1.251),
            (5.0e-4, -1.2495),
            (-5.0e-4, -1.2505),
        ):
            displaced = np.array(midpoint, copy=True)
            displaced[atom, axis] += displacement
            indices.append(
                _append_event(
                    events,
                    leaves,
                    energy=energy,
                    geometry=admission_contracts._ase_geometry_sha256(
                        (8, 1, 1),
                        tuple(tuple(float(item) for item in row) for row in displaced),
                    ),
                )
            )
        edges.append(
            {
                "label": label,
                "midpoint_offsets_angstrom": list(offsets),
                "displacement_angstrom": edge_displacement,
                "component": {
                    "atom_index": atom,
                    "axis_index": axis,
                    "coarse_step_angstrom": 1.0e-3,
                    "fine_step_angstrom": 5.0e-4,
                    "center_event_index": center,
                    "plus_h_event_index": indices[0],
                    "minus_h_event_index": indices[1],
                    "plus_h2_event_index": indices[2],
                    "minus_h2_event_index": indices[3],
                },
            }
        )
    return {
        "cartesian_dofs": [[0, 0], [1, 1]],
        "half_width_angstrom": half_width,
        "orientation": "counterclockwise",
        "edges": edges,
    }


def _water_system_fixture():
    benzene_raw, events, leaves = _benzene_system_fixture()
    positions = np.asarray(WATER_POSITIONS)
    base_forces = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    base = _append_panel_to_graph(
        events,
        leaves,
        label="base",
        positions=positions,
        force_matrix=base_forces,
    )
    direction = np.random.default_rng(20260816).normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    step = 1.25e-4
    directional_indices = []
    force_projection = -float(np.vdot(base_forces, direction))
    for sign in (1.0, -1.0):
        displaced = positions + sign * step * direction
        directional_indices.append(
            _append_event(
                events,
                leaves,
                energy=-1.25 + sign * step * force_projection,
                geometry=admission_contracts._ase_geometry_sha256(
                    (8, 1, 1),
                    tuple(tuple(float(item) for item in row) for row in displaced),
                ),
            )
        )
    translation_vector = np.asarray([4.2, -3.1, 1.7])
    translated = _append_panel_to_graph(
        events,
        leaves,
        label="translated",
        positions=positions + translation_vector,
        force_matrix=base_forces,
    )
    rotation_matrix = admission_contracts._frozen_rotation(20260817)
    centroid = np.mean(positions, axis=0)
    rotated_positions = (positions - centroid) @ rotation_matrix.T + centroid
    rotated = _append_panel_to_graph(
        events,
        leaves,
        label="rotated",
        positions=rotated_positions,
        force_matrix=base_forces @ rotation_matrix.T,
    )
    loop = _append_loop_to_graph(events, leaves, positions)
    assert len(events) == 141
    return (
        {
            "event_range": [8, 141],
            "base_panel": base,
            "directional": {
                "seed": 20260816,
                "normalized_direction": direction.tolist(),
                "step_angstrom": step,
                "plus_event_index": directional_indices[0],
                "minus_event_index": directional_indices[1],
            },
            "translation": {
                "translation_angstrom": translation_vector.tolist(),
                "panel": translated,
            },
            "rotation": {
                "seed": 20260817,
                "rotation_matrix": rotation_matrix.tolist(),
                "panel": rotated,
            },
            "closed_loop": loop,
        },
        benzene_raw,
        tuple(events),
        leaves,
    )


def test_benzene_system_enforces_exact_occurrence_schedule_and_topology() -> None:
    raw, events, leaves = _benzene_system_fixture()
    system = BenzeneSystemV2.from_mapping(
        raw,
        solve_events=tuple(events),
        state_leaves=leaves,
        prepared_pes_configuration_sha256="6" * 64,
    )
    assert system.event_range == (0, 8)
    assert system.all_starts_converged is True
    assert system.maximum_final_residual_norm_eV == 0.0
    for path, replacement in (
        (("event_range",), [0, 9]),
        (("reported_center_event_index",), 6),
        (("predecessor_stencil", "plus_h4_event_index"), 6),
    ):
        candidate = json.loads(json.dumps(raw))
        cursor = candidate
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = replacement
        with pytest.raises(ValueError):
            BenzeneSystemV2.from_mapping(
                candidate,
                solve_events=tuple(events),
                state_leaves=leaves,
                prepared_pes_configuration_sha256="6" * 64,
            )
    with pytest.raises(ValueError, match="sealed prepared PES"):
        BenzeneSystemV2.from_mapping(
            raw,
            solve_events=tuple(events),
            state_leaves=leaves,
            prepared_pes_configuration_sha256="8" * 64,
        )


def _replace_event_leaf(
    events: tuple[SolveEventV2, ...] | list[SolveEventV2],
    leaves: dict[str, StateLeafV2],
    index: int,
    mutate,
):
    event_list = list(events)
    original = leaves[event_list[index].state_leaf_sha256]
    payload = original.as_dict()
    mutate(payload)
    _refresh_state_leaf_array_content_sha256s(payload)
    _rehash(payload, "state_leaf_sha256")
    replacement = StateLeafV2.from_mapping(payload)
    leaf_map = dict(leaves)
    leaf_map[replacement.state_leaf_sha256] = replacement
    event_payload = {
        "schema_id": SOLVE_EVENT_V2_SCHEMA,
        "event_index": index,
        "state_leaf_sha256": replacement.state_leaf_sha256,
        "geometry_sha256": replacement.geometry_sha256,
        "provider_configuration_sha256": replacement.provider_configuration_sha256,
        "topology_id": replacement.topology_id,
    }
    event_payload["solve_event_sha256"] = canonical_json_sha256(event_payload)
    event_list[index] = SolveEventV2.from_mapping(
        event_payload, state_leaves=leaf_map, expected_index=index
    )
    return tuple(event_list), leaf_map


def test_benzene_reported_center_requires_exact_same_state_leaf_digest() -> None:
    raw, events, leaves = _benzene_system_fixture()

    def change_energy(payload: dict[str, object]) -> None:
        payload["vacuum_energy_eV"] = -0.999999
        payload["total_energy_eV"] = (
            payload["vacuum_energy_eV"] + payload["polarization_energy_eV"]
        )
        payload["cold_start"]["final_total_energy_eV"] = payload["total_energy_eV"]
        payload["wide_start"]["final_total_energy_eV"] = payload["total_energy_eV"]

    events, leaves = _replace_event_leaf(
        events,
        leaves,
        7,
        change_energy,
    )
    with pytest.raises(ValueError, match="exact predecessor center StateLeaf"):
        BenzeneSystemV2.from_mapping(
            raw,
            solve_events=tuple(events),
            state_leaves=leaves,
            prepared_pes_configuration_sha256="6" * 64,
        )


def test_water_system_consumes_exact_occurrences_and_derives_raw_metrics() -> None:
    raw, _, events, leaves = _water_system_fixture()
    system = WaterSystemV2.from_mapping(
        raw,
        solve_events=events,
        state_leaves=leaves,
        prepared_water_positions=WATER_POSITIONS,
        prepared_pes_configuration_sha256="7" * 64,
    )
    assert system.event_range == (8, 141)
    assert system.base_center_energy_eV == -1.25
    assert system.translation_energy_abs_difference_eV == 0.0
    assert system.maximum_local_error_eV_per_angstrom < 2.0e-12
    base_forces = np.asarray(
        admission_contracts._thaw_json(system.base_forces_eV_per_angstrom)
    )
    rotated_forces = np.asarray(
        admission_contracts._thaw_json(system.rotated_forces_eV_per_angstrom)
    )
    rotation = np.asarray(
        admission_contracts._thaw_json(system.rotation.rotation_matrix)
    )
    expected_delta = rotated_forces - base_forces @ rotation.T
    assert system.rotation_force_covariance_max_abs_difference_eV_per_angstrom == (
        pytest.approx(float(np.max(np.abs(expected_delta))))
    )
    assert system.maximum_final_residual_norm_eV == 0.0
    assert system.maximum_total_charge_error_e == 0.0
    assert system.all_starts_converged is True
    with pytest.raises(ValueError, match="sealed prepared PES"):
        WaterSystemV2.from_mapping(
            raw,
            solve_events=events,
            state_leaves=leaves,
            prepared_water_positions=WATER_POSITIONS,
            prepared_pes_configuration_sha256="8" * 64,
        )


def _valid_reported_metrics():
    raw, benzene_raw, events, leaves = _water_system_fixture()
    benzene = BenzeneSystemV2.from_mapping(
        benzene_raw,
        solve_events=events,
        state_leaves=leaves,
        prepared_pes_configuration_sha256="6" * 64,
    )
    water = WaterSystemV2.from_mapping(
        raw,
        solve_events=events,
        state_leaves=leaves,
        prepared_water_positions=WATER_POSITIONS,
        prepared_pes_configuration_sha256="7" * 64,
    )
    return admission_contracts._reported_metrics(benzene, water)


def test_replicate_gate_threshold_boundaries_are_exact() -> None:
    seal = _seal()
    force_panel = _force_panel_payload()
    metrics = _valid_reported_metrics()
    directional_threshold = force_panel[
        "maximum_independent_directional_error_ev_per_angstrom"
    ]
    metrics["water_directional_error_eV_per_angstrom"] = directional_threshold
    assert (
        admission_contracts._replicate_gates(
            metrics, seal=seal, force_panel=force_panel
        )["independent_h4_directional_derivative_below_5e-4_ev_per_angstrom"]
        is True
    )
    metrics["water_directional_error_eV_per_angstrom"] = np.nextafter(
        directional_threshold, math.inf
    )
    assert (
        admission_contracts._replicate_gates(
            metrics, seal=seal, force_panel=force_panel
        )["independent_h4_directional_derivative_below_5e-4_ev_per_angstrom"]
        is False
    )

    metrics = _valid_reported_metrics()
    root_tolerance = seal.as_dict()["scientific_settings"]["root_algorithm"][
        "tolerance_ev"
    ]
    metrics["water_maximum_final_residual_norm_eV"] = root_tolerance
    assert (
        admission_contracts._replicate_gates(
            metrics, seal=seal, force_panel=force_panel
        )["all_final_residuals_and_total_charge_errors_pass"]
        is False
    )

    metrics = _valid_reported_metrics()
    rotation_threshold = force_panel["maximum_rotation_force_relative_error"]
    metrics["water_rotation_force_relative_error"] = rotation_threshold
    assert (
        admission_contracts._replicate_gates(
            metrics, seal=seal, force_panel=force_panel
        )["rotation_energy_and_force_covariance_gates_pass"]
        is True
    )
    metrics["water_rotation_force_relative_error"] = np.nextafter(
        rotation_threshold, math.inf
    )
    assert (
        admission_contracts._replicate_gates(
            metrics, seal=seal, force_panel=force_panel
        )["rotation_energy_and_force_covariance_gates_pass"]
        is False
    )

    metrics = _valid_reported_metrics()
    charge_threshold = seal.as_dict()["scientific_settings"]["root_algorithm"][
        "total_charge_tolerance_e"
    ]
    metrics["benzene_maximum_total_charge_error_e"] = charge_threshold
    assert (
        admission_contracts._replicate_gates(
            metrics, seal=seal, force_panel=force_panel
        )["all_final_residuals_and_total_charge_errors_pass"]
        is True
    )


def test_water_root_metrics_use_vector_norm_and_both_start_convergence_flags() -> None:
    raw, _, events, leaves = _water_system_fixture()

    def inject_residual_and_failure(payload: dict[str, object]) -> None:
        payload["cold_start"]["final_residual_eV"][0] = [0.75] * 8
        payload["cold_start"]["final_residual_eV"][1] = [0.75] * 8
        payload["wide_start"]["converged"] = False

    events, leaves = _replace_event_leaf(
        events, leaves, 140, inject_residual_and_failure
    )
    system = WaterSystemV2.from_mapping(
        raw,
        solve_events=events,
        state_leaves=leaves,
        prepared_water_positions=WATER_POSITIONS,
        prepared_pes_configuration_sha256="7" * 64,
    )
    assert system.maximum_final_residual_norm_eV == pytest.approx(
        math.sqrt(16.0 * 0.75**2)
    )
    assert system.maximum_final_residual_norm_eV > math.sqrt(8.0 * 0.75**2)
    assert system.all_starts_converged is False


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("event_range",), [8, 140], "event_range"),
        (("directional", "plus_event_index"), 46, "directional"),
        (
            ("base_panel", "components", 1, "plus_h_event_index"),
            9,
            "occurrence schedule",
        ),
        (
            ("translation", "panel", "components", 0, "center_event_index"),
            8,
            "geometry|schedule|shared center",
        ),
        (
            ("closed_loop", "edges", 0, "component", "center_event_index"),
            122,
            "schedule|geometry|distinct",
        ),
    ),
)
def test_water_system_rejects_coverage_order_and_duplicate_substitutions(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    raw, _, events, leaves = _water_system_fixture()
    candidate = json.loads(json.dumps(raw))
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError, match=message):
        WaterSystemV2.from_mapping(
            candidate,
            solve_events=events,
            state_leaves=leaves,
            prepared_water_positions=WATER_POSITIONS,
            prepared_pes_configuration_sha256="7" * 64,
        )


def test_water_system_rejects_cross_topology_occurrence() -> None:
    raw, _, events, leaves = _water_system_fixture()
    event_index = 140
    original_event = events[event_index]
    original_leaf = leaves[original_event.state_leaf_sha256]
    leaf_payload = original_leaf.as_dict()
    leaf_payload["topology_id"] = SHA_F
    _rehash(leaf_payload, "state_leaf_sha256")
    replacement_leaf = StateLeafV2.from_mapping(leaf_payload)
    leaves = dict(leaves)
    leaves[replacement_leaf.state_leaf_sha256] = replacement_leaf
    event_payload = {
        "schema_id": SOLVE_EVENT_V2_SCHEMA,
        "event_index": event_index,
        "state_leaf_sha256": replacement_leaf.state_leaf_sha256,
        "geometry_sha256": replacement_leaf.geometry_sha256,
        "provider_configuration_sha256": (
            replacement_leaf.provider_configuration_sha256
        ),
        "topology_id": replacement_leaf.topology_id,
    }
    event_payload["solve_event_sha256"] = canonical_json_sha256(event_payload)
    events = list(events)
    events[event_index] = SolveEventV2.from_mapping(
        event_payload,
        state_leaves=leaves,
        expected_index=event_index,
    )
    with pytest.raises(ValueError, match="topology"):
        WaterSystemV2.from_mapping(
            raw,
            solve_events=tuple(events),
            state_leaves=leaves,
            prepared_water_positions=WATER_POSITIONS,
            prepared_pes_configuration_sha256="7" * 64,
        )


def test_exact_round_trip_and_derived_gate_result() -> None:
    seal_payload = _seal_payload()
    seal = ComputationSealV2.from_mapping(
        json.loads(json.dumps(seal_payload, sort_keys=True))
    )
    aggregate_payload = _aggregate_payload(seal)
    aggregate = AggregateAdmissionInputsV2.from_mapping(
        json.loads(json.dumps(aggregate_payload, sort_keys=True)), seal=seal
    )
    overlay_payload = _overlay_payload(seal, aggregate)
    overlay = AdmissionOverlayV2.from_mapping(
        json.loads(json.dumps(overlay_payload, sort_keys=True)),
        seal=seal,
        aggregate=aggregate,
    )
    assert seal.as_dict() == seal_payload
    assert aggregate.as_dict() == aggregate_payload
    assert overlay.as_dict() == overlay_payload
    assert aggregate.all_gates_passed is True
    assert aggregate.measurement_sha256 == aggregate.replicates[0].measurement_sha256
    assert overlay.capabilities == CapabilityStatus(
        energy=True, conservative_force=True
    )


def test_deep_immutability_and_no_constructor_bypass() -> None:
    payload = _seal_payload()
    seal = ComputationSealV2.from_mapping(payload)
    payload["scientific_settings"]["continuum"]["surface_lmax"] = 99
    payload["runtime_fingerprint"]["torch"]["devices"].append("other")
    assert seal.as_dict()["scientific_settings"] == _science()
    assert seal.as_dict()["runtime_fingerprint"] == _runtime()
    with pytest.raises(FrozenInstanceError):
        seal.git_head = "3" * 40  # type: ignore[misc]
    for contract in (
        ComputationSealV2,
        ReplicateAdmissionRecordV2,
        AggregateAdmissionInputsV2,
        AdmissionOverlayV2,
    ):
        with pytest.raises(TypeError, match="from_mapping"):
            contract()
        with pytest.raises(TypeError):
            contract(schema_id="unsafe")  # type: ignore[call-arg]


def test_replicate_measurement_round_trip_and_deep_immutability() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    record = ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)
    assert record.as_dict() == payload
    payload["measurement"]["local_decision"]["public_capability_admitted"] = True
    assert (
        record.as_dict()["measurement"]["local_decision"]["public_capability_admitted"]
        is False
    )
    with pytest.raises(FrozenInstanceError):
        record.measurement_sha256 = SHA_A  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_id", "wrong-measurement-schema"),
        ("seal_id", "wrong-seal"),
        ("profile_id", "wrong-profile"),
        ("local_decision", {}),
    ],
)
def test_measurement_schema_binding_and_nonempty_objects(
    field: str, value: object
) -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"][field] = value
    payload["measurement_sha256"] = canonical_json_sha256(payload["measurement"])
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_measurement_tamper_rejects_same_claimed_digest() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    original_digest = payload["measurement_sha256"]
    payload["measurement"]["reported_metrics"]["tampered_metric"] = 1.0e-12
    assert payload["measurement_sha256"] == original_digest
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="measurement_sha256"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    "section",
    ("benzene_gepol_regression", "water_force_symmetry_loop", "aggregate"),
)
def test_opaque_or_minimal_science_measurement_is_rejected(section: str) -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"][section] = {"passed": True}
    _rehash_replicate_measurement(payload)
    with pytest.raises(ValueError, match="unknown fields"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_reported_metric_tamper_without_raw_leaf_change_is_rejected() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"]["reported_metrics"][
        "water_maximum_local_error_eV_per_angstrom"
    ] = 1.0
    _rehash_replicate_measurement(payload)
    with pytest.raises(ValueError, match="raw-derived system metrics"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_reported_local_gate_tamper_without_raw_change_is_rejected() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"]["reported_local_gate_results"][
        "translation_energy_and_net_force_gates_pass"
    ] = False
    _rehash_replicate_measurement(payload)
    with pytest.raises(ValueError, match="raw-derived replicate gates"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_unreferenced_state_leaf_is_rejected_before_admission() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    extra = StateLeafV2.from_mapping(
        _state_leaf_payload(geometry="8" * 64, total_energy=-1.3)
    )
    payload["measurement"]["state_leaves"][extra.state_leaf_sha256] = extra.as_dict()
    _rehash_replicate_measurement(payload)
    with pytest.raises(ValueError, match="referenced leaf set"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_reported_false_gate_fails_envelope_preflight() -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["gate_results"][REPLICATE_GATE_NAMES[0]] = False
    _rehash(payload, "artifact_sha256")
    with pytest.raises(ValueError, match="every replicate admission gate"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize("section", ("protocol", "local_decision"))
@pytest.mark.parametrize("operation", ("missing", "unknown"))
def test_protocol_and_decision_fields_are_exact(section: str, operation: str) -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    record = payload["measurement"][section]
    if operation == "missing":
        record.pop(next(iter(record)))
    else:
        record["unexpected"] = False
    _rehash_replicate_measurement(payload)
    with pytest.raises(ValueError, match="fields"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("preregistration_id", "different-preregistration"),
        ("preregistration_sha256", SHA_B),
        ("fit_calibration_or_case_selection", True),
        ("fit_calibration_or_case_selection", 0),
    ),
)
def test_protocol_cannot_widen_or_escape_seal(field: str, value: object) -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"]["protocol"][field] = value
    _rehash_replicate_measurement(payload)
    with pytest.raises((TypeError, ValueError)):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    "field",
    (
        "public_capability_admitted",
        "chemical_accuracy_admitted",
        "complete_solvation_free_energy_admitted",
        "analytic_force_admitted",
        "hessian_frequency_md_admitted",
        "tier_v_admitted",
    ),
)
def test_local_replicate_cannot_set_any_admission_decision_true(field: str) -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"]["local_decision"][field] = True
    _rehash_replicate_measurement(payload)
    with pytest.raises(ValueError, match="fail-closed local decision"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_energy_force_gate_passed", False),
        ("awaiting_independent_replay", False),
        ("candidate_energy_force_gate_passed", 1),
    ),
)
def test_local_candidate_decision_is_exact(field: str, value: object) -> None:
    seal = _seal()
    payload = _replicate_payload(seal, "a")
    payload["measurement"]["local_decision"][field] = value
    _rehash_replicate_measurement(payload)
    with pytest.raises((TypeError, ValueError)):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


def test_engine_and_admission_gate_name_drift_fails_closed(monkeypatch) -> None:
    from maple.solvation.release import harmonic_ef_measurement

    seal = _seal()
    payload = _replicate_payload(seal, "a")
    monkeypatch.setattr(
        harmonic_ef_measurement,
        "HARMONIC_EF_REPLICATE_GATE_NAMES",
        (*REPLICATE_GATE_NAMES[:-1], "drifted_gate"),
    )
    with pytest.raises(ValueError, match="gate names drifted"):
        ReplicateAdmissionRecordV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("continuum", "profile_id"),
        ("continuum", "transition_width_angstrom2"),
        ("continuum", "surface_lmax"),
        ("continuum", "exposure_lmax"),
        ("continuum", "exposure_radial_quadrature_order"),
        ("continuum", "source_radial_quadrature_order"),
        ("continuum", "green_radial_quadrature_order"),
        ("cavity", "profile_id"),
        ("cavity", "radii_provider_id"),
        ("source_receiver", "permanent_source_kernel"),
        ("source_receiver", "induced_source_kernel"),
        ("source_receiver", "receiver_kernel"),
        ("source_receiver", "long_range_evaluator_id"),
        ("energy_ledger", "exact_scalar"),
        ("energy_ledger", "included_components"),
        ("energy_ledger", "excluded_components"),
        ("root_algorithm", "method"),
        ("root_algorithm", "tolerance_ev"),
        ("root_algorithm", "maximum_iterations"),
        ("root_algorithm", "second_start"),
        ("root_algorithm", "total_charge_tolerance_e"),
        ("root_algorithm", "multi_start_field_tolerance_ev"),
        ("root_algorithm", "multi_start_energy_tolerance_ev"),
        ("force_stencil", "derivative"),
        ("force_stencil", "coarse_step_angstrom"),
        ("force_stencil", "fine_step_angstrom"),
        ("force_stencil", "independent_step_angstrom"),
        ("force_stencil", "maximum_local_error_ev_per_angstrom"),
        ("force_stencil", "topology_policy"),
        ("force_stencil", "displacement_policy"),
    ],
)
def test_every_scientific_field_is_required(section: str, field: str) -> None:
    payload = _seal_payload()
    payload["scientific_settings"][section].pop(field)
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="missing fields"):
        ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("continuum", "unknown", 1),
        ("cavity", "unknown", 1),
        ("source_receiver", "unknown", 1),
        ("energy_ledger", "unknown", 1),
        ("root_algorithm", "unknown", 1),
        ("force_stencil", "unknown", 1),
        ("continuum", "transition_width_angstrom2", 0.0),
        ("continuum", "surface_lmax", -1),
        ("continuum", "exposure_radial_quadrature_order", 0),
        ("root_algorithm", "tolerance_ev", float("inf")),
        ("root_algorithm", "maximum_iterations", 0),
        ("force_stencil", "fine_step_angstrom", -1.0),
        ("energy_ledger", "included_components", []),
        ("energy_ledger", "excluded_components", ["cds", "cds"]),
    ],
)
def test_arbitrary_or_invalid_scientific_settings_fail_closed(
    section: str, field: str, value: object
) -> None:
    payload = _seal_payload()
    payload["scientific_settings"][section][field] = value
    if value != float("inf"):
        _rehash(payload, "content_sha256")
    with pytest.raises((TypeError, ValueError)):
        ComputationSealV2.from_mapping(payload)


def test_energy_components_must_be_disjoint() -> None:
    payload = _seal_payload()
    payload["scientific_settings"]["energy_ledger"]["excluded_components"] = [
        H1_V2_INCLUDED_COMPONENTS[0]
    ]
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="disjoint"):
        ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("torch", "cuda_available"), False),
        (("torch", "devices"), []),
        (("torch", "deterministic_algorithms_enabled"), False),
        (("torch", "deterministic_debug_mode"), -1),
        (("torch", "deterministic_debug_mode"), 0),
        (("torch", "deterministic_debug_mode"), 1),
        (("torch", "cudnn_benchmark"), True),
        (("torch", "cudnn_deterministic"), False),
        (("torch", "default_dtype"), "torch.float32"),
        (("torch", "threads"), 2),
        (("torch", "interop_threads"), 2),
        (("torch", "driver_version"), ""),
        (("torch", "device_uuids"), []),
        (("torch", "device_uuids"), ["not-a-uuid"]),
        (("torch", "device_capabilities"), ["sm_89"]),
        (("torch", "device_multiprocessor_counts"), [0]),
        (("environment", "OMP_NUM_THREADS"), None),
        (("environment", "OMP_NUM_THREADS"), "2"),
        (("environment", "MKL_NUM_THREADS"), None),
        (("environment", "OPENBLAS_NUM_THREADS"), "4"),
        (("environment", "CUBLAS_WORKSPACE_CONFIG"), None),
        (("environment", "CUBLAS_WORKSPACE_CONFIG"), ":bad"),
        (("environment", "PYTHONHASHSEED"), None),
        (("environment", "PYTHONHASHSEED"), "random"),
        (("environment", "PYTHONHASHSEED"), "-1"),
        (("environment", "CUDA_VISIBLE_DEVICES"), ""),
        (("environment", "CUDA_VISIBLE_DEVICES"), None),
        (("environment", "CUDA_VISIBLE_DEVICES"), "-1"),
        (("environment", "CUDA_VISIBLE_DEVICES"), "1"),
        (("execution_device",), "cpu"),
        (("execution_dtype",), "float32"),
    ],
)
def test_nondeterministic_or_broad_runtime_is_rejected(
    path: tuple[str, ...], value: object
) -> None:
    payload = _seal_payload()
    runtime = payload["runtime_fingerprint"]
    if len(path) == 1:
        runtime[path[0]] = value
    else:
        runtime[path[0]][path[1]] = value
    _rehash(payload, "content_sha256")
    with pytest.raises((TypeError, ValueError)):
        ComputationSealV2.from_mapping(payload)


def test_device_identity_lists_reject_length_mismatch_and_duplicate_uuid() -> None:
    payload = _seal_payload()
    payload["runtime_fingerprint"]["torch"]["device_capabilities"] = []
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="same length"):
        ComputationSealV2.from_mapping(payload)

    payload = _seal_payload()
    torch = payload["runtime_fingerprint"]["torch"]
    torch["devices"] = ["GPU A", "GPU B"]
    torch["device_uuids"] = [
        "GPU-33333333-3333-4333-8333-333333333333",
        "GPU-33333333-3333-4333-8333-333333333333",
    ]
    torch["device_capabilities"] = ["8.9", "8.9"]
    torch["device_multiprocessor_counts"] = [128, 128]
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="duplicate-free"):
        ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("runtime", "implementation"),
        ("torch", "threads"),
        ("torch", "interop_threads"),
        ("torch", "driver_version"),
        ("torch", "device_uuids"),
        ("torch", "device_capabilities"),
        ("torch", "device_multiprocessor_counts"),
        ("packages", "numpy"),
        ("numpy", "show_config"),
        ("environment", "OMP_NUM_THREADS"),
    ],
)
def test_runtime_missing_fields_are_rejected(section: str, field: str) -> None:
    payload = _seal_payload()
    runtime = payload["runtime_fingerprint"]
    target = runtime if section == "runtime" else runtime[section]
    target.pop(field)
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="missing fields"):
        ComputationSealV2.from_mapping(payload)


def test_unknown_runtime_fields_are_rejected_at_fixed_levels() -> None:
    for section in (None, "torch", "packages", "numpy", "environment"):
        payload = _seal_payload()
        runtime = payload["runtime_fingerprint"]
        target = runtime if section is None else runtime[section]
        target["unknown"] = 1
        _rehash(payload, "content_sha256")
        with pytest.raises(ValueError, match="unknown fields"):
            ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("package", "runtime_section", "version"),
    [
        ("numpy", "numpy", "9.9.9"),
        ("torch", "torch", "9.9.9"),
    ],
)
def test_package_versions_must_match_runtime_sections(
    package: str, runtime_section: str, version: str
) -> None:
    payload = _seal_payload()
    payload["runtime_fingerprint"]["packages"][package] = version
    assert payload["runtime_fingerprint"][runtime_section]["version"] != version
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match=package):
        ComputationSealV2.from_mapping(payload)


def test_any_runtime_broad_claim_and_missing_nonadmissions_are_rejected() -> None:
    for field, value in (
        ("runtime_guards", ["any-runtime"]),
        ("claim_boundary_id", "broad-production-claim"),
        ("non_admissions", list(REQUIRED_NON_ADMISSIONS[:-1])),
    ):
        payload = _seal_payload()
        payload[field] = value
        _rehash(payload, "content_sha256")
        with pytest.raises(ValueError):
            ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    "label", ["blind-revalidation", "any-protocol", "arbitrary", ""]
)
def test_exposure_aware_protocol_label_is_exact(label: str) -> None:
    payload = _seal_payload()
    payload["exposure_aware_protocol_label"] = label
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="protocol label"):
        ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_id", "arbitrary-profile"),
        ("scalar_id", "arbitrary-scalar"),
        ("state_id", "arbitrary-state"),
        ("domain_guards", ["any-domain"]),
    ],
)
def test_h1_ids_and_domain_are_exact(field: str, value: object) -> None:
    payload = _seal_payload()
    payload[field] = value
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError):
        ComputationSealV2.from_mapping(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("continuum", "profile_id", "arbitrary-continuum"),
        ("continuum", "transition_width_angstrom2", 0.19),
        ("continuum", "surface_lmax", 2),
        ("source_receiver", "permanent_source_kernel", "arbitrary-kernel"),
        ("source_receiver", "receiver_kernel", "arbitrary-receiver"),
        ("energy_ledger", "exact_scalar", "arbitrary-scalar"),
        (
            "energy_ledger",
            "included_components",
            list(reversed(H1_V2_INCLUDED_COMPONENTS)),
        ),
        ("root_algorithm", "method", "arbitrary-root"),
        ("force_stencil", "derivative", "arbitrary-derivative"),
        ("force_stencil", "coarse_step_angstrom", 1.0e-4),
        ("force_stencil", "fine_step_angstrom", 3.0e-4),
        ("force_stencil", "independent_step_angstrom", 1.2e-4),
    ],
)
def test_any_h1_scientific_semantic_drift_is_rejected(
    section: str, field: str, value: object
) -> None:
    payload = _seal_payload()
    payload["scientific_settings"][section][field] = value
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError):
        ComputationSealV2.from_mapping(payload)


def test_asset_and_checkpoint_keys_are_exact() -> None:
    seal = _seal()
    assert set(dict(seal.asset_sha256s)).isdisjoint(REQUIRED_CHECKPOINT_KEYS)
    assert set(dict(seal.checkpoint_sha256s)) == set(REQUIRED_CHECKPOINT_KEYS)

    for field in ("asset_sha256s", "checkpoint_sha256s"):
        payload = _seal_payload()
        ledger = payload[field]
        removed = next(iter(ledger))
        ledger["arbitrary_asset"] = ledger.pop(removed)
        _rehash(payload, "content_sha256")
        with pytest.raises(ValueError, match="fields"):
            ComputationSealV2.from_mapping(payload)


def test_checkpoint_cannot_be_duplicated_conflictingly_in_asset_ledger() -> None:
    payload = _seal_payload()
    asset_key = next(iter(payload["asset_sha256s"]))
    payload["asset_sha256s"].pop(asset_key)
    payload["asset_sha256s"][REQUIRED_CHECKPOINT_KEYS[0]] = SHA_F
    assert (
        payload["asset_sha256s"][REQUIRED_CHECKPOINT_KEYS[0]]
        != payload["checkpoint_sha256s"][REQUIRED_CHECKPOINT_KEYS[0]]
    )
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError, match="fields"):
        ComputationSealV2.from_mapping(payload)


def _mutate_replicate(
    aggregate_payload: dict[str, object], index: int, field: str, value: object
) -> None:
    replicate = aggregate_payload["replicates"][index]
    replicate[field] = value
    _rehash(replicate, "artifact_sha256")
    _rehash(aggregate_payload, "aggregate_sha256")


def test_aggregate_requires_exactly_two_typed_replicates() -> None:
    seal = _seal()
    for replicates in ([], [_replicate_payload(seal, "a")]):
        payload = _aggregate_payload(seal)
        payload["replicates"] = replicates
        _rehash(payload, "aggregate_sha256")
        with pytest.raises(ValueError, match="exactly two"):
            AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


def test_replicates_must_be_independent_processes() -> None:
    seal = _seal()
    for field in (
        "artifact_id",
        "artifact_sha256",
        "process_uuid",
        "process_started_at_utc",
    ):
        payload = _aggregate_payload(seal)
        first = payload["replicates"][0]
        if field == "artifact_sha256":
            second = dict(first)
            second["label"] = "b"
            second["artifact_id"] = "replicate-b"
            second["process_uuid"] = "22222222-2222-4222-8222-222222222222"
            second["process_started_at_utc"] = "2026-08-23T02:00:00+00:00"
            second["artifact_sha256"] = first["artifact_sha256"]
            payload["replicates"][1] = second
        else:
            _mutate_replicate(payload, 1, field, first[field])
        _rehash(payload, "aggregate_sha256")
        with pytest.raises(ValueError, match=field):
            AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


def test_replicate_uuid_and_start_time_are_typed() -> None:
    seal = _seal()
    for field, value in (
        ("process_uuid", "not-a-uuid"),
        ("process_started_at_utc", "2026-08-23T01:00:00"),
    ):
        payload = _aggregate_payload(seal)
        _mutate_replicate(payload, 0, field, value)
        with pytest.raises(ValueError):
            AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


def test_one_false_missing_or_extra_gate_is_rejected() -> None:
    seal = _seal()
    for mode in ("false", "missing", "extra"):
        payload = _aggregate_payload(seal)
        gates = payload["replicates"][0]["gate_results"]
        if mode == "false":
            gates[V1_ADMISSION_GATE_NAMES[0]] = False
            payload["replicates"][1]["gate_results"][V1_ADMISSION_GATE_NAMES[0]] = False
            _rehash(payload["replicates"][1], "artifact_sha256")
        elif mode == "missing":
            gates.pop(V1_ADMISSION_GATE_NAMES[0])
        else:
            gates["invented_gate"] = True
        _rehash(payload["replicates"][0], "artifact_sha256")
        _rehash(payload, "aggregate_sha256")
        expected = "every replicate admission gate" if mode == "false" else "fields"
        with pytest.raises(ValueError, match=expected):
            AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)

    payload = _aggregate_payload(seal)
    replicate = payload["replicates"][0]
    replicate["gate_results"][AGGREGATE_GATE_NAMES[0]] = True
    _rehash(replicate, "artifact_sha256")
    _rehash(payload, "aggregate_sha256")
    with pytest.raises(ValueError, match="unknown fields"):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize("mode", ["false", "missing", "extra"])
def test_aggregate_gate_is_required_only_at_aggregate_level(mode: str) -> None:
    seal = _seal()
    payload = _aggregate_payload(seal)
    gates = payload["aggregate_gate_results"]
    if mode == "false":
        gates[AGGREGATE_GATE_NAMES[0]] = False
    elif mode == "missing":
        gates.pop(AGGREGATE_GATE_NAMES[0])
    else:
        gates["invented_cross_run_gate"] = True
    _rehash(payload, "aggregate_sha256")
    expected = "every aggregate admission gate" if mode == "false" else "fields"
    with pytest.raises(ValueError, match=expected):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


def test_aggregate_rejects_same_digest_with_different_measurement_object() -> None:
    seal = _seal()
    payload = _aggregate_payload(seal)
    second = payload["replicates"][1]
    first_digest = payload["replicates"][0]["measurement_sha256"]
    second["measurement"]["reported_metrics"]["tampered_metric"] = 1.0e-12
    second["measurement_sha256"] = first_digest
    _rehash(second, "artifact_sha256")
    _rehash(payload, "aggregate_sha256")
    with pytest.raises(ValueError, match="measurement_sha256"):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    "field",
    [
        "seal_id",
        "seal_sha256",
        "git_head",
        "git_tree",
        "source_ledger_sha256",
        "asset_ledger_sha256",
        "runtime_fingerprint_sha256",
    ],
)
def test_replicate_bindings_must_equal_seal(field: str) -> None:
    seal = _seal()
    payload = _aggregate_payload(seal)
    value = SHA_C if field.endswith("sha256") else "wrong"
    if field in {"git_head", "git_tree"}:
        value = "3" * 40
    _mutate_replicate(payload, 0, field, value)
    with pytest.raises(ValueError, match=field):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    "field",
    [
        "measurement_sha256",
        "source_ledger_sha256",
        "asset_ledger_sha256",
        "runtime_fingerprint_sha256",
        "git_head",
    ],
)
def test_replicate_proofs_must_match_each_other(field: str) -> None:
    seal = _seal()
    payload = _aggregate_payload(seal)
    value = SHA_B if field.endswith("sha256") else "3" * 40
    _mutate_replicate(payload, 1, field, value)
    with pytest.raises(ValueError, match=field):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


def test_aggregate_has_no_caller_set_all_gates_field() -> None:
    seal = _seal()
    payload = _aggregate_payload(seal)
    payload["all_gates_passed"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_guards", ["any-runtime"]),
        ("domain_guards", ["any-domain"]),
        ("claim_boundary_id", "broad-claim"),
        ("non_admissions", list(REQUIRED_NON_ADMISSIONS[:-1])),
        (
            "capabilities",
            {"E": True, "F": True, "H": True, "V": False, "M": False},
        ),
    ],
)
def test_aggregate_cannot_broaden_seal(field: str, value: object) -> None:
    seal = _seal()
    payload = _aggregate_payload(seal)
    payload[field] = value
    _rehash(payload, "aggregate_sha256")
    with pytest.raises(ValueError):
        AggregateAdmissionInputsV2.from_mapping(payload, seal=seal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_guards", ["any-runtime"]),
        ("domain_guards", ["any-domain"]),
        ("claim_boundary_id", "broad-claim"),
        ("non_admissions", list(REQUIRED_NON_ADMISSIONS[:-1])),
        ("aggregate_id", "wrong-aggregate"),
        ("seal_id", "wrong-seal"),
    ],
)
def test_overlay_exactly_cross_binds_normalized_aggregate_evidence(
    field: str, value: object
) -> None:
    seal = _seal()
    aggregate_payload = _aggregate_payload(seal)
    aggregate_values = {
        "aggregate_id": aggregate_payload["aggregate_id"],
        "aggregate_sha256": aggregate_payload["aggregate_sha256"],
        "profile_id": aggregate_payload["profile_id"],
        "scalar_id": aggregate_payload["scalar_id"],
        "state_id": aggregate_payload["state_id"],
        "capabilities": CapabilityStatus(energy=True, conservative_force=True),
        "runtime_guards": tuple(aggregate_payload["runtime_guards"]),
        "domain_guards": tuple(aggregate_payload["domain_guards"]),
        "claim_boundary_id": aggregate_payload["claim_boundary_id"],
        "non_admissions": tuple(aggregate_payload["non_admissions"]),
    }
    payload = _overlay_payload_from_values(seal, aggregate_values)
    payload[field] = value
    _rehash(payload, "content_sha256")
    with pytest.raises(ValueError):
        admission_contracts._overlay_values(
            payload, seal=seal, aggregate_values=aggregate_values
        )
