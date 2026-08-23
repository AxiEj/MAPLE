#!/usr/bin/env python3
"""Seal the exact label-free hybrid MNSol-10 prediction execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import sys
from typing import Mapping, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release._secure_artifacts import (  # noqa: E402
    CapturedFile,
    SecureArtifactError,
    StabilityGuard,
    canonical_sha256,
    capture_clean_repository,
    capture_file,
    capture_repo_files,
    load_json_bytes,
    host_user_custody_contract,
    publish_json_noreplace,
    secure_publication_contract,
    source_sha256s,
)
from tools.route2_release._hybrid_mnsol10_chain import (  # noqa: E402
    build_attempt_slot_identity,
)

BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
PREREGISTRATION = BENCHMARK_DIR / (
    "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json"
)
PROTOCOL = BENCHMARK_DIR / "route2-mnsol-protocol-v1.json"
SELECTION = BENCHMARK_DIR / "route2-mnsol-pilot-selection-v1.json"
INPUT_BUNDLE = REPO_ROOT / ".omx/route2/hybrid-mnsol10/label-free-input-v1.json"
DEFAULT_MDP_CHECKPOINT = Path.home() / ".cache/mace/MACE-MDP.model"
DEFAULT_POLAR_CHECKPOINT = Path.home() / ".cache/mace/MACE-POLAR-1-M.model"
SEAL_DIRECTORY = REPO_ROOT / ".omx/route2/hybrid-mnsol10/seals"
SEAL_ARTIFACT_ID = "route2-mace-mdp-polar-hybrid-mnsol10-computation-seal-v1"
REQUIRED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}
SOURCE_FILES = (
    "GOAL.md",
    "docs/route2/HYBRID_MNSOL_ACCURACY.md",
    "docs/route2/evidence/mace-mdp-polar-hybrid-mnsol10-pro-audit-v1.json",
    "maple/function/calculator/extra_correction/implicit/pyscf_smd_cds.py",
    "maple/function/calculator/extra_correction/implicit/smd_cds.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/route2_solvents.py",
    "maple/solvation/api/identities.py",
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/continuum/harmonic_ddpcm_functional.py",
    "maple/solvation/continuum/harmonic_ddpcm_hybrid.py",
    "maple/solvation/continuum/harmonic_ddpcm_primitives.py",
    "maple/solvation/continuum/harmonic_schwarz_primitives.py",
    "maple/solvation/coupling/additive_solvent_ledgers.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/metrics.py",
    "maple/solvation/coupling/permanent_induced_ledgers.py",
    "maple/solvation/coupling/permanent_induced_state.py",
    "maple/solvation/coupling/separated_fixed_point.py",
    "maple/solvation/coupling/separated_ledgers.py",
    "maple/solvation/coupling/separated_operators.py",
    "maple/solvation/coupling/spaces.py",
    "maple/solvation/models/checkpoint_bytes.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/accuracy_admission.py",
    "maple/solvation/release/admission.py",
    "maple/solvation/solvent_terms.py",
    "docs/implicit-solvation/benchmarks/benchmark_core.py",
    "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
    "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
    "docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json",
    "docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json",
    "docs/implicit-solvation/benchmarks/route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json",
    "docs/route2/evidence/mace-polar-radial-gto-mnsol10-accuracy-v1.json",
    "docs/route2/evidence/mace-polar-point-l1-mnsol10-accuracy-v1.json",
    "docs/route2/evidence/mace-polar-point-l1-harmonic-ddpcm-mnsol10-accuracy-v1.json",
    "docs/route2/evidence/mace-polar-point-l1-harmonic-ddcosmo-mnsol10-accuracy-v1.json",
    "docs/route2/evidence/mace-mdp-polar-harmonic-ddpcm-general-source-four-4cf8db40.json",
    "docs/route2/evidence/mace-mdp-polar-harmonic-ddpcm-general-source-water-force-4cf8db40.json",
    "docs/route2/evidence/operational-harmonic-ddpcm-water-hvp-canary-4cf8db40.json",
    "tools/route2_release/_secure_artifacts.py",
    "tools/route2_release/_mnsol_label_free_inputs.py",
    "tools/route2_release/_hybrid_mnsol10_chain.py",
    "tools/route2_release/prepare_mace_mdp_polar_hybrid_mnsol10_inputs.py",
    "tools/route2_release/seal_mace_mdp_polar_hybrid_mnsol10_fullsolv.py",
    "tools/route2_release/run_mace_mdp_polar_hybrid_mnsol10_fullsolv.py",
    "tools/route2_release/recover_mace_mdp_polar_hybrid_mnsol10_attempt.py",
    "tools/route2_release/score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py",
    "tools/route2_release/publish_mace_mdp_polar_hybrid_mnsol10_public.py",
    "tools/route2_release/publish_mace_mdp_polar_hybrid_mnsol10_failure.py",
)


def _environment() -> dict[str, str]:
    actual = {name: os.environ.get(name) for name in REQUIRED_ENVIRONMENT}
    if actual != REQUIRED_ENVIRONMENT:
        raise SecureArtifactError(
            f"deterministic environment must be exact: {actual!r}"
        )
    return dict(REQUIRED_ENVIRONMENT)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SecureArtifactError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SecureArtifactError(
            f"{name} keys changed; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _sequence(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise SecureArtifactError(f"{name} must be a list")
    return cast(list[object], value)


def _validate_preregistration(
    payload: Mapping[str, object],
    *,
    preregistration: CapturedFile,
    protocol: CapturedFile,
    selection: CapturedFile,
    mdp_checkpoint: CapturedFile,
    polar_checkpoint: CapturedFile,
) -> None:
    _exact_keys(
        payload,
        {
            "artifact_id",
            "schema_version",
            "created_at_utc",
            "status",
            "protocol_classification",
            "target_identity",
            "dataset_contract",
            "model_contract",
            "continuum_contract",
            "root_contract",
            "nonpolar_contract",
            "decision_rule",
            "post_execution_prohibitions",
            "claim_boundary",
            "execution_contract",
            "verified_pro_audit",
        },
        name="preregistration",
    )
    if payload.get("artifact_id") != (
        "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1"
    ):
        raise SecureArtifactError("unsupported hybrid MNSol-10 preregistration")
    if payload.get("status") != (
        "prepared-exposure-aware-known-panel-regression-no-hybrid-output-observed"
    ):
        raise SecureArtifactError("preregistration is no longer pre-result")
    classification = _mapping(
        payload.get("protocol_classification"), name="protocol_classification"
    )
    _exact_keys(
        classification,
        {
            "is_blind_preregistration",
            "is_exposure_aware_revalidation",
            "unobserved_target",
            "known_prior_evidence",
            "non_transfer_rule",
            "prior_observation_ledger",
        },
        name="protocol_classification",
    )
    if (
        classification.get("is_blind_preregistration") is not False
        or classification.get("is_exposure_aware_revalidation") is not True
    ):
        raise SecureArtifactError("exposure classification drifted")
    target = _mapping(payload.get("target_identity"), name="target_identity")
    _exact_keys(
        target,
        {
            "profile_id",
            "scalar_id",
            "state_equation_id",
            "exact_solvation_scalar",
            "gas_energy_cancellation",
            "excluded_energy_terms",
        },
        name="target_identity",
    )
    dataset = _mapping(payload.get("dataset_contract"), name="dataset_contract")
    model = _mapping(payload.get("model_contract"), name="model_contract")
    _exact_keys(
        dataset,
        {
            "protocol_path",
            "protocol_sha256",
            "selection_path",
            "selection_sha256",
            "selection_fingerprint",
            "record_count",
            "solvent_count",
            "partition_counts",
            "geometry_policy",
            "temperature_k",
            "standard_state",
            "row_level_output",
        },
        name="dataset_contract",
    )
    if {
        "protocol_path": dataset.get("protocol_path"),
        "protocol_sha256": dataset.get("protocol_sha256"),
        "selection_path": dataset.get("selection_path"),
        "selection_sha256": dataset.get("selection_sha256"),
        "selection_fingerprint": dataset.get("selection_fingerprint"),
        "record_count": dataset.get("record_count"),
        "solvent_count": dataset.get("solvent_count"),
        "partition_counts": dataset.get("partition_counts"),
        "geometry_policy": dataset.get("geometry_policy"),
        "temperature_k": dataset.get("temperature_k"),
        "standard_state": dataset.get("standard_state"),
        "row_level_output": dataset.get("row_level_output"),
    } != {
        "protocol_path": PROTOCOL.relative_to(REPO_ROOT).as_posix(),
        "protocol_sha256": protocol.sha256,
        "selection_path": SELECTION.relative_to(REPO_ROOT).as_posix(),
        "selection_sha256": selection.sha256,
        "selection_fingerprint": (
            "967979795b3bd1483db93f7b463f9a65f3f2c0859f69228100d2adc107138fba"
        ),
        "record_count": 10,
        "solvent_count": 10,
        "partition_counts": {"confirmation": 8, "development": 2},
        "geometry_policy": ("one-fixed-MNSol-M06-2X-MG3S-gas-geometry-no-optimization"),
        "temperature_k": 298.0,
        "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        "row_level_output": "private-below-.omx-no-redistribution",
    }:
        raise SecureArtifactError("preregistered dataset contract drifted")
    _exact_keys(
        model,
        {
            "mace_mdp_checkpoint_sha256",
            "mace_mdp_device",
            "mace_polar_checkpoint_sha256",
            "mace_polar_device",
            "dtype",
            "long_range_evaluator_id",
            "permanent_source_kernel",
            "induced_source_kernel",
            "receiver_space",
        },
        name="model_contract",
    )
    if {
        "mace_mdp_checkpoint_sha256": model.get("mace_mdp_checkpoint_sha256"),
        "mace_mdp_device": model.get("mace_mdp_device"),
        "mace_polar_checkpoint_sha256": model.get("mace_polar_checkpoint_sha256"),
        "mace_polar_device": model.get("mace_polar_device"),
        "dtype": model.get("dtype"),
        "long_range_evaluator_id": model.get("long_range_evaluator_id"),
        "permanent_source_kernel": model.get("permanent_source_kernel"),
        "induced_source_kernel": model.get("induced_source_kernel"),
        "receiver_space": model.get("receiver_space"),
    } != {
        "mace_mdp_checkpoint_sha256": mdp_checkpoint.sha256,
        "mace_mdp_device": "cpu",
        "mace_polar_checkpoint_sha256": polar_checkpoint.sha256,
        "mace_polar_device": "cuda",
        "dtype": "float64",
        "long_range_evaluator_id": (
            "graph-longrange-analytic-gaussian-multipole-realspace-v1"
        ),
        "permanent_source_kernel": ("MACE-MDP exterior point monopoles and dipoles"),
        "induced_source_kernel": (
            "MACE-POLAR field-induced increment in the normalized sigma=1.5-A "
            "Gaussian multipole basis"
        ),
        "receiver_space": (
            "checkpoint-native sigma=1.5/3.0-A eight-channel potential/gradient field"
        ),
    }:
        raise SecureArtifactError("preregistered input/checkpoint digest drifted")
    if preregistration.path != PREREGISTRATION.resolve(strict=True):
        raise SecureArtifactError("preregistration path override is forbidden")
    decision = _mapping(payload.get("decision_rule"), name="decision_rule")
    _exact_keys(
        decision,
        {
            "primary_metric",
            "maximum_mae_kcal_mol",
            "required_gates",
            "required_reports",
            "failure",
            "success",
        },
        name="decision_rule",
    )
    if decision.get("maximum_mae_kcal_mol") != 1.5 or decision.get(
        "required_gates"
    ) != [
        "complete_exact_frozen_panel",
        "ten_distinct_solvents",
        "all_cold_wide_roots_pass",
        "all_actual_root_residuals_pass",
        "all_permanent_and_induced_charge_checks_pass",
        "mae_at_most_1_5_kcal_mol",
    ]:
        raise SecureArtifactError("hard MAE target drifted")
    continuum = _mapping(payload.get("continuum_contract"), name="continuum_contract")
    expected_continuum = {
        "model": "finite-dielectric-smooth-partition-harmonic-ddpcm-general-source",
        "radii": "PySCF-2.13.1 SMD solvent-dependent Coulomb radii",
        "dielectric": "route2_solvent_spec canonical solvent dielectric",
        "transition_width_angstrom2": 0.08,
        "surface_lmax": 3,
        "partition_lmax": 6,
        "partition_radial_quadrature_order": 96,
        "source_radial_quadrature_order": 128,
        "double_layer_radial_quadrature_order": 128,
        "receiver_radial_quadrature_order": 128,
        "device": "cpu",
    }
    _exact_keys(
        continuum,
        set(expected_continuum) | {"numerics_provenance"},
        name="continuum_contract",
    )
    if any(continuum.get(key) != value for key, value in expected_continuum.items()):
        raise SecureArtifactError("continuum contract drifted")
    root = _mapping(payload.get("root_contract"), name="root_contract")
    expected_root = {
        "method": "anderson",
        "tolerance": 1.0e-10,
        "max_iterations": 100,
        "damping": 0.5,
        "history": 6,
        "starts": ["cold-zero", "deterministic-wide-0.05"],
        "wide_seed_base": 20260816,
        "wide_initial_amplitude": 0.05,
        "maximum_actual_residual_norm": 1.0e-10,
        "maximum_induced_source_absolute_difference_e": 2.0e-9,
        "maximum_native_field_absolute_difference_eV_per_e": 2.0e-9,
        "maximum_continuum_energy_absolute_difference_eV": 1.0e-9,
        "maximum_permanent_charge_error_e": 1.0e-8,
        "maximum_induced_charge_error_e": 1.0e-8,
        "maximum_combined_charge_error_e": 1.0e-8,
    }
    _exact_keys(
        root,
        set(expected_root) | {"root_semantics_provenance"},
        name="root_contract",
    )
    if any(root.get(key) != value for key, value in expected_root.items()):
        raise SecureArtifactError("root contract drifted")
    nonpolar = _mapping(payload.get("nonpolar_contract"), name="nonpolar_contract")
    if dict(nonpolar) != {
        "provider_id": "maple.route2.solvent-term.pyscf-smd-cds.impl.v1",
        "model": "PySCF-2.13.1 SMD-CDS",
        "coefficient_scale": 1.0,
        "fit_or_calibration": False,
        "combination_rule": "strict additive G_hybrid_ddPCM + G_SMD_CDS",
    }:
        raise SecureArtifactError("nonpolar contract drifted")
    prohibitions = _mapping(
        payload.get("post_execution_prohibitions"),
        name="post_execution_prohibitions",
    )
    if set(prohibitions) != {
        "fit",
        "calibration",
        "case_selection",
        "threshold_change",
        "continuum_setting_change",
        "root_setting_change",
        "source_kernel_change",
        "cds_scale_change",
        "ledger_change",
    } or set(prohibitions.values()) != {False}:
        raise SecureArtifactError("post-execution prohibitions drifted")
    execution = _mapping(payload.get("execution_contract"), name="execution_contract")
    _exact_keys(
        execution,
        {
            "label_boundary",
            "seal",
            "one_time_claim",
            "prediction_terminal",
            "scoring_boundary",
            "public_projection",
            "publication_durability",
            "orphan_recovery",
            "evidence_class",
            "custody_contract",
            "required_environment",
        },
        name="execution_contract",
    )
    required_environment = _mapping(
        execution.get("required_environment"), name="required_environment"
    )
    if dict(required_environment) != REQUIRED_ENVIRONMENT:
        raise SecureArtifactError("execution environment contract drifted")
    if execution.get("evidence_class") != ("exposure-aware-known-panel-regression"):
        raise SecureArtifactError("execution evidence class drifted")
    if execution.get("custody_contract") != host_user_custody_contract():
        raise SecureArtifactError("execution custody contract drifted")
    pro_audit = _mapping(payload.get("verified_pro_audit"), name="verified_pro_audit")
    _exact_keys(
        pro_audit,
        {
            "path",
            "file_sha256",
            "artifact_id",
            "final_verdict",
            "final_prompt_sha256",
            "final_response_sha256",
            "terminal_marker",
            "scientific_output_observed_before_final_approval",
        },
        name="verified_pro_audit",
    )
    if {
        "path": pro_audit.get("path"),
        "artifact_id": pro_audit.get("artifact_id"),
        "final_verdict": pro_audit.get("final_verdict"),
        "terminal_marker": pro_audit.get("terminal_marker"),
        "scientific_output_observed_before_final_approval": pro_audit.get(
            "scientific_output_observed_before_final_approval"
        ),
    } != {
        "path": (
            "docs/route2/evidence/" "mace-mdp-polar-hybrid-mnsol10-pro-audit-v1.json"
        ),
        "artifact_id": (
            "route2-mace-mdp-polar-hybrid-mnsol10-pro-architecture-audit-v1"
        ),
        "final_verdict": "APPROVE",
        "terminal_marker": "MAPLE HYBRID ACCURACY FINAL REPAIR COMPLETE",
        "scientific_output_observed_before_final_approval": False,
    }:
        raise SecureArtifactError("verified Pro audit contract drifted")
    ledger = _sequence(
        classification.get("prior_observation_ledger"),
        name="prior_observation_ledger",
    )
    if len(ledger) != 11:
        raise SecureArtifactError("prior-observation ledger is incomplete")


def _capture_prior_observations(
    payload: Mapping[str, object],
) -> dict[str, CapturedFile]:
    classification = _mapping(
        payload.get("protocol_classification"), name="protocol_classification"
    )
    ledger = _sequence(
        classification.get("prior_observation_ledger"),
        name="prior_observation_ledger",
    )
    captures: dict[str, CapturedFile] = {}
    for index, raw in enumerate(ledger):
        item = _mapping(raw, name=f"prior_observation_ledger[{index}]")
        _exact_keys(
            item,
            {"role", "artifact_id", "path", "file_sha256"},
            name=f"prior_observation_ledger[{index}]",
        )
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise SecureArtifactError("prior-observation path is invalid")
        candidate = (REPO_ROOT / relative).resolve(strict=True)
        try:
            normalized = candidate.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise SecureArtifactError(
                "prior-observation path escapes repository"
            ) from exc
        if normalized != relative:
            raise SecureArtifactError("prior-observation path is not canonical")
        capture = capture_file(candidate, role=f"prior observation {index}")
        if capture.sha256 != item.get("file_sha256"):
            raise SecureArtifactError(
                f"prior-observation digest drifted at index {index}"
            )
        captures[f"prior_observation_{index}"] = capture
    return captures


def _capture_verified_pro_artifacts(
    payload: Mapping[str, object],
) -> dict[str, CapturedFile]:
    contract = _mapping(payload.get("verified_pro_audit"), name="verified_pro_audit")
    audit_relative = contract.get("path")
    if not isinstance(audit_relative, str):
        raise SecureArtifactError("verified Pro audit path is invalid")
    audit_capture = capture_file(
        REPO_ROOT / audit_relative, role="verified Pro tracked audit"
    )
    if audit_capture.sha256 != contract.get("file_sha256"):
        raise SecureArtifactError("verified Pro tracked audit digest drifted")
    audit = load_json_bytes(audit_capture.data, role="verified Pro tracked audit")
    if {
        "artifact_id": audit.get("artifact_id"),
        "status": audit.get("status"),
        "final_verdict": audit.get("final_verdict"),
        "scientific_output_observed_before_final_approval": audit.get(
            "scientific_output_observed_before_final_approval"
        ),
    } != {
        "artifact_id": contract.get("artifact_id"),
        "status": "verified-pro-final-approve-before-scientific-execution",
        "final_verdict": "APPROVE",
        "scientific_output_observed_before_final_approval": False,
    }:
        raise SecureArtifactError("verified Pro tracked audit contents drifted")
    rounds = _sequence(audit.get("rounds"), name="verified Pro audit rounds")
    if len(rounds) != 3:
        raise SecureArtifactError("verified Pro audit must contain three rounds")
    captures = {"verified_pro_tracked_audit": audit_capture}
    for index, raw_round in enumerate(rounds):
        round_contract = _mapping(raw_round, name=f"verified Pro round {index}")
        for role in ("prompt", "mode", "submission", "response", "uia"):
            relative = round_contract.get(f"{role}_path")
            expected = round_contract.get(f"{role}_sha256")
            if not isinstance(relative, str) or not relative.startswith(
                ".omx/artifacts/pro-hybrid-accuracy-chain-"
            ):
                raise SecureArtifactError(f"verified Pro {role} path is not canonical")
            capture = capture_file(
                REPO_ROOT / relative,
                role=f"verified Pro round {index} {role}",
            )
            if capture.sha256 != expected:
                raise SecureArtifactError(
                    f"verified Pro round {index} {role} digest drifted"
                )
            captures[f"verified_pro_{index}_{role}"] = capture
    final_round = _mapping(rounds[-1], name="final verified Pro round")
    if (
        final_round.get("verdict") != "APPROVE"
        or final_round.get("prompt_sha256") != contract.get("final_prompt_sha256")
        or final_round.get("response_sha256") != contract.get("final_response_sha256")
        or final_round.get("end_marker") != contract.get("terminal_marker")
    ):
        raise SecureArtifactError("final verified Pro approval binding drifted")
    return captures


def build_seal(
    *,
    source_path: str | Path,
    mdp_checkpoint_path: str | Path,
    polar_checkpoint_path: str | Path,
) -> tuple[dict[str, object], StabilityGuard]:
    environment = _environment()
    repository = capture_clean_repository(REPO_ROOT)
    tracked = capture_repo_files(repository, SOURCE_FILES)
    captures = {
        "mnsol_source": capture_file(source_path, role="MNSol distribution"),
        "preregistration": capture_file(PREREGISTRATION, role="preregistration"),
        "protocol": capture_file(PROTOCOL, role="MNSol protocol"),
        "selection": capture_file(SELECTION, role="MNSol selection"),
        "input_bundle": capture_file(INPUT_BUNDLE, role="label-free input bundle"),
        "mace_mdp_checkpoint": capture_file(
            mdp_checkpoint_path, role="MACE-MDP checkpoint"
        ),
        "mace_polar_checkpoint": capture_file(
            polar_checkpoint_path, role="MACE-POLAR checkpoint"
        ),
    }
    preregistration = load_json_bytes(
        captures["preregistration"].data, role="preregistration"
    )
    prior_observations = _capture_prior_observations(preregistration)
    verified_pro_artifacts = _capture_verified_pro_artifacts(preregistration)
    stability = StabilityGuard(
        repository,
        tuple(
            [(f"tracked source {name}", capture) for name, capture in tracked.items()]
            + list(captures.items())
            + list(prior_observations.items())
            + list(verified_pro_artifacts.items())
        ),
    )
    input_bundle = load_json_bytes(
        captures["input_bundle"].data, role="label-free input bundle"
    )
    selection_payload = load_json_bytes(
        captures["selection"].data, role="MNSol selection"
    )
    _validate_preregistration(
        preregistration,
        preregistration=captures["preregistration"],
        protocol=captures["protocol"],
        selection=captures["selection"],
        mdp_checkpoint=captures["mace_mdp_checkpoint"],
        polar_checkpoint=captures["mace_polar_checkpoint"],
    )
    accuracy = importlib.import_module("maple.solvation.release.accuracy_admission")
    accuracy.validate_label_free_input_bundle(input_bundle, expected_record_count=10)
    label_free = importlib.import_module(
        "tools.route2_release._mnsol_label_free_inputs"
    )
    expected_dataset, expected_records = label_free.derive_mnsol10_label_free_inputs(
        source_bytes=captures["mnsol_source"].data,
        protocol_path=captures["protocol"].path,
        selection_payload=selection_payload,
        benchmark_directory=BENCHMARK_DIR,
    )
    if input_bundle.get("artifact_id") != (
        "route2-mace-mdp-polar-hybrid-mnsol10-label-free-input-v1"
    ):
        raise SecureArtifactError("label-free input artifact identity drifted")
    if {
        "preregistration_sha256": input_bundle.get("preregistration_sha256"),
        "protocol_sha256": input_bundle.get("protocol_sha256"),
        "selection_sha256": input_bundle.get("selection_sha256"),
        "selection_fingerprint": input_bundle.get("selection_fingerprint"),
        "dataset": input_bundle.get("dataset"),
        "records": input_bundle.get("records"),
    } != {
        "preregistration_sha256": captures["preregistration"].sha256,
        "protocol_sha256": captures["protocol"].sha256,
        "selection_sha256": captures["selection"].sha256,
        "selection_fingerprint": selection_payload.get("selection_fingerprint"),
        "dataset": expected_dataset,
        "records": expected_records,
    }:
        raise SecureArtifactError(
            "label-free input bundle differs from source-backed frozen selection"
        )
    input_git = _mapping(input_bundle.get("git"), name="input bundle Git identity")
    if (
        input_git.get("head") != repository.head
        or input_git.get("tree") != repository.tree
    ):
        raise SecureArtifactError("label-free inputs were prepared on another tree")
    api = importlib.import_module("maple.solvation.api")
    state_api = importlib.import_module("maple.solvation.api.state_registry")
    target = _mapping(preregistration.get("target_identity"), name="target_identity")
    if target.get("profile_id") != (
        api.OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
    ) or target.get("scalar_id") != (
        api.OPERATIONAL_MACE_MDP_POLAR_HYBRID_SMD_TOTAL_SMOOTH_HARMONIC_DDPCM_V1
    ):
        raise SecureArtifactError("registered total-SMD target identity drifted")
    profile = api.get_solvation_profile(str(target["profile_id"]))
    scalar = api.get_scalar_definition(str(target["scalar_id"]))
    if (
        target.get("state_equation_id")
        != state_api.PERMANENT_INDUCED_SEPARATED_OPERATIONAL_STATE_EQUATION_ID
        or profile.state_equation_id != target.get("state_equation_id")
        or scalar.state_equation_id != target.get("state_equation_id")
        or profile.coupling_id != api.MACE_MDP_POLAR_HYBRID_HARMONIC_DDPCM_COUPLING_ID
        or profile.nonpolar_profile != "pyscf-smd-cds-v1"
        or scalar.implementation_entry_point
        != (
            "maple.solvation.coupling.additive_solvent_ledgers:"
            "AdditiveSolventOperationalLedger"
        )
    ):
        raise SecureArtifactError("registered total-SMD runtime contract drifted")
    source_hashes = source_sha256s(tracked)
    source_validation = {
        "mnsol_source_file_sha256": captures["mnsol_source"].sha256,
        "dataset": expected_dataset,
        "derived_records_sha256": canonical_sha256(expected_records),
    }
    attempt_slot_identity = build_attempt_slot_identity(
        preregistration=preregistration,
        source_validation=source_validation,
        protocol_sha256=captures["protocol"].sha256,
        selection_sha256=captures["selection"].sha256,
        mace_mdp_checkpoint_sha256=captures["mace_mdp_checkpoint"].sha256,
        mace_polar_checkpoint_sha256=captures["mace_polar_checkpoint"].sha256,
    )
    attempt_slot_id = canonical_sha256(attempt_slot_identity)
    publication_contract = secure_publication_contract()
    custody_contract = host_user_custody_contract()
    core = {
        "contract": "route2-hybrid-mnsol10-execution-identity-v1",
        "attempt_slot_id": attempt_slot_id,
        "git_head": repository.head,
        "git_tree": repository.tree,
        "profile_id": target["profile_id"],
        "scalar_id": target["scalar_id"],
        "preregistration_sha256": captures["preregistration"].sha256,
        "protocol_sha256": captures["protocol"].sha256,
        "selection_sha256": captures["selection"].sha256,
        "input_bundle_sha256": captures["input_bundle"].sha256,
        "input_content_sha256": input_bundle["content_sha256"],
        "mace_mdp_checkpoint_sha256": captures["mace_mdp_checkpoint"].sha256,
        "mace_polar_checkpoint_sha256": captures["mace_polar_checkpoint"].sha256,
        "source_files_sha256": source_hashes,
        "source_validation": source_validation,
        "verified_pro_audit_sha256": verified_pro_artifacts[
            "verified_pro_tracked_audit"
        ].sha256,
        "required_environment": environment,
        "publication_contract": publication_contract,
        "custody_contract": custody_contract,
    }
    execution_id = canonical_sha256(core)
    seal: dict[str, object] = {
        "schema_id": "maple-route2-accuracy-computation-seal-v1",
        "artifact_id": SEAL_ARTIFACT_ID,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_id": execution_id,
        "attempt_slot_id": attempt_slot_id,
        "attempt_slot_identity": attempt_slot_identity,
        "execution_identity": core,
        "git": repository.as_dict(),
        "profile_id": target["profile_id"],
        "scalar_id": target["scalar_id"],
        "preregistration": {
            "path": PREREGISTRATION.relative_to(REPO_ROOT).as_posix(),
            "sha256": captures["preregistration"].sha256,
        },
        "protocol_sha256": captures["protocol"].sha256,
        "selection_sha256": captures["selection"].sha256,
        "input_bundle": {
            "path": os.fspath(INPUT_BUNDLE.relative_to(REPO_ROOT)),
            "file_sha256": captures["input_bundle"].sha256,
            "content_sha256": input_bundle["content_sha256"],
        },
        "checkpoints": {
            "mace_mdp_sha256": captures["mace_mdp_checkpoint"].sha256,
            "mace_polar_sha256": captures["mace_polar_checkpoint"].sha256,
        },
        "source_files_sha256": source_hashes,
        "source_validation": source_validation,
        "verified_pro_audit_sha256": verified_pro_artifacts[
            "verified_pro_tracked_audit"
        ].sha256,
        "required_environment": environment,
        "publication_contract": publication_contract,
        "custody_contract": custody_contract,
        "record_count": 10,
        "claim_boundary": preregistration["claim_boundary"],
    }
    seal["content_sha256"] = canonical_sha256(seal)
    stability.assert_stable()
    return seal, stability


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mdp-checkpoint", type=Path, default=DEFAULT_MDP_CHECKPOINT)
    parser.add_argument(
        "--polar-checkpoint", type=Path, default=DEFAULT_POLAR_CHECKPOINT
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        seal, stability = build_seal(
            source_path=args.source,
            mdp_checkpoint_path=args.mdp_checkpoint,
            polar_checkpoint_path=args.polar_checkpoint,
        )
        output = SEAL_DIRECTORY / f"{seal['execution_id']}.json"
        publish_json_noreplace(
            output,
            seal,
            root=REPO_ROOT,
            stability=stability.assert_stable,
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
