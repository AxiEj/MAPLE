#!/usr/bin/env python3
"""Run the frozen smooth-harmonic hybrid E/F admission panel.

This runner evaluates exactly one disabled operational scalar:

``E = E_vac^MACE-POLAR - 1/2 b(u*)^T A_harm(R)^-1 b(u*)``.

The MACE-MDP permanent point multipoles and the MACE-POLAR induced Gaussian
multipoles share one fixed-dimensional smooth harmonic-Galerkin continuum.  A
complete re-solve is performed at every finite-difference point.  Passing this
panel can support only experimental electrostatic energy and its numerical
scalar-gradient force.  It cannot admit chemical accuracy, complete solvation
free energy, analytic derivatives, Hessians, frequencies, MD, or Tier V.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1,
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
)
from maple.solvation.experimental.mace_mdp_polar_harmonic import (
    NUMERICAL_FORCE_COARSE_STEP_ANGSTROM,
    NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
)
from maple.solvation.models import (
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_mace_mdp_anchored_mace_polar_hybrid,
    build_mace_mdp_moment_adapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release.harmonic_ef_measurement import (
    run_harmonic_ef_measurement,
)
from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-mdp-polar-hybrid-harmonic-force-admission-v1"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-force-admission-v1.json"
)
PARENT_PREREGISTRATION_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
EXPECTED_MACE_POLAR_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/continuum/harmonic_point_source.py",
    "maple/solvation/continuum/harmonic_torch_functional.py",
    "maple/solvation/continuum/harmonic_torch_primitives.py",
    "maple/solvation/derivatives/scalar_finite_difference.py",
    "maple/solvation/experimental/mace_mdp_polar_harmonic.py",
    "maple/solvation/models/mace_mdp.py",
    "maple/solvation/models/mace_mdp_polar_hybrid.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/models/runtime/analytic_gaussian_multipole.py",
    "maple/solvation/release/evidence.py",
    "maple/solvation/release/harmonic_ef_measurement.py",
    "tools/route2_release/run_mace_mdp_polar_hybrid_harmonic_force_admission.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--mace-mdp-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-MDP.model",
    )
    parser.add_argument(
        "--mace-polar-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACEPOLAR1Mmodel",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--replicate-label", choices=("a", "b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(20260816)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260816)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _checkpoint_with_role(path: Path, role: str) -> dict[str, object]:
    record = checkpoint_record(path)
    record["role"] = role
    return record


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mace_mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.mace_polar_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(mdp_checkpoint) != MACE_MDP_EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-MDP checkpoint does not match the frozen model.")
    if sha256_file(polar_checkpoint) != EXPECTED_MACE_POLAR_CHECKPOINT_SHA256:
        raise RuntimeError("MACE-POLAR checkpoint does not match the frozen model.")

    preregistration_path = repository.root / PREREGISTRATION_RELATIVE_PATH
    preregistration = _load_json(
        preregistration_path, name="smooth-harmonic hybrid force preregistration"
    )
    if (
        preregistration.get("target_profile_id")
        != EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
        or preregistration.get("target_scalar_id")
        != EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
        or preregistration.get("status") != "frozen-before-full-admission-execution"
    ):
        raise RuntimeError("smooth-harmonic force preregistration identity is invalid.")
    if preregistration.get("target_capabilities") != {
        "E": True,
        "F": True,
        "H": False,
        "V": False,
        "M": False,
    }:
        raise RuntimeError("smooth-harmonic target capabilities drifted.")
    runtime_contract = preregistration.get("frozen_runtime_contract")
    force_panel = preregistration.get("force_panel")
    parent_binding = preregistration.get("parent_panel")
    if not all(
        isinstance(value, dict)
        for value in (runtime_contract, force_panel, parent_binding)
    ):
        raise RuntimeError("smooth-harmonic preregistration is incomplete.")
    assert isinstance(runtime_contract, dict)
    assert isinstance(force_panel, dict)
    assert isinstance(parent_binding, dict)
    if (
        float(runtime_contract["force_coarse_step_angstrom"])
        != NUMERICAL_FORCE_COARSE_STEP_ANGSTROM
        or float(runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"])
        != NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
    ):
        raise RuntimeError("force implementation drifted from preregistration.")
    if (
        runtime_contract.get("mace_mdp_checkpoint_sha256")
        != MACE_MDP_EXPECTED_CHECKPOINT_SHA256
        or runtime_contract.get("mace_polar_checkpoint_sha256")
        != EXPECTED_MACE_POLAR_CHECKPOINT_SHA256
        or runtime_contract.get("mace_polar_long_range_evaluator")
        != MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
    ):
        raise RuntimeError("checkpoint/evaluator preregistration drifted.")

    parent_path = repository.root / PARENT_PREREGISTRATION_RELATIVE_PATH
    if parent_binding.get("relative_path") != PARENT_PREREGISTRATION_RELATIVE_PATH:
        raise RuntimeError("parent preregistration path drifted.")
    _validated_sha(
        parent_path,
        parent_binding.get("sha256"),
        name="parent PCM preregistration",
    )
    parent = _load_json(parent_path, name="parent PCM preregistration")

    import torch

    _configure_determinism(torch)
    mdp = build_mace_mdp_moment_adapter(checkpoint_path=mdp_checkpoint, device="cpu")
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    hybrid = build_mace_mdp_anchored_mace_polar_hybrid(
        permanent=mdp,
        response=MACEPolarOriginalSourceNativeFieldAdapter(radial),
    )
    science = run_harmonic_ef_measurement(
        asset_root=asset_root,
        parent=parent,
        hybrid=hybrid,
        radial=radial,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    execution_gate = bool(science["aggregate"]["all_execution_gates_passed"])
    measurement = {
        "protocol": {
            "profile_id": (
                EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
            ),
            "scalar_id": (
                EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
            ),
            "preregistration_artifact_id": preregistration["artifact_id"],
            "preregistration_sha256": sha256_file(preregistration_path),
            "parent_preregistration_artifact_id": parent["artifact_id"],
            "parent_preregistration_sha256": sha256_file(parent_path),
            "frozen_runtime_contract": runtime_contract,
            "force_panel": force_panel,
            "fit_calibration_or_case_selection": False,
        },
        **science,
        "decision": {
            "candidate_energy_force_gate_passed": execution_gate,
            "awaiting_independent_replay": execution_gate,
            "public_capability_admitted": False,
            "chemical_accuracy_admitted": False,
            "complete_solvation_free_energy_admitted": False,
            "analytic_force_admitted": False,
            "hessian_frequency_md_admitted": False,
            "tier_v_admitted": False,
        },
    }
    measurement_sha256 = canonical_json_sha256(measurement)
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    payload = {
        "artifact_id": (
            "route2-mace-mdp-polar-hybrid-harmonic-force-admission-"
            f"replicate-{args.replicate_label}-v1"
        ),
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "preregistered-experimental-energy-force-admission-run",
        "status": (
            "candidate-energy-force-gate-passed-awaiting-independent-replay"
            if execution_gate
            else "candidate-energy-force-gate-failed"
        ),
        "claim_boundary": preregistration["claim_boundary"],
        "capabilities": NO_CAPABILITIES,
        "replicate_label": args.replicate_label,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
                "artifact_id": preregistration["artifact_id"],
            },
            "parent_preregistration": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
                "artifact_id": parent["artifact_id"],
            },
            "mace_mdp_checkpoint": _checkpoint_with_role(
                mdp_checkpoint, "mace-mdp-official-checkpoint"
            ),
            "mace_polar_checkpoint": _checkpoint_with_role(
                polar_checkpoint, "mace-polar-1-m-official-checkpoint"
            ),
        },
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": str(radial.dtype).replace("torch.", ""),
        **measurement,
        "measurement_sha256": measurement_sha256,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_MDP_POLAR_HARMONIC_FORCE="
        + json.dumps(
            {
                "artifact": artifact,
                "aggregate": measurement["aggregate"],
                "decision": measurement["decision"],
                "measurement_sha256": measurement_sha256,
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
