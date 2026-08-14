#!/usr/bin/env python3
"""Run the disabled real-checkpoint common-stationarity water canary.

This runner evaluates the *changed-source* MACE-POLAR candidate whose complete
eight-channel source is generated from the anchored field-energy scalar.  It
couples that model scalar to the fixed-topology reciprocal C-PCM scalar, solves
their common stationarity equation, and checks one stationary-envelope
coordinate derivative at three step sizes.

The conventional laboratory-fixed Lebedev assembly is not a structural SO(3)
method.  A passing result here is therefore implementation evidence for one
disabled scalar at one geometry, not Tier E/F/H/V/M admission.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from ase import Atoms

# Set before Torch creates a CUDA context.  The runtime manifest records it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.api.profiles import (
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1,
)
from maple.solvation.api.scalar_registry import (
    VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
)
from maple.solvation.continuum import (
    FixedReciprocalCPCMFunctional,
    build_water_radial_gto_cpcm_backend,
)
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
)
from maple.solvation.coupling.variational_state import (
    build_variational_common_functional,
)
from maple.solvation.models import (
    MACEPolarVariationalFieldEnergy,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-variational-common-water-real-checkpoint-canary-v1"
ARTIFACT_KIND = "disabled-real-checkpoint-changed-source-common-scalar-canary"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
RANDOM_SEED = 20260814
FD_STEPS_ANGSTROM = (5.0e-4, 2.0e-4, 1.0e-4)
ROOT_CONTEXT_ID = "route2-variational-common-water-canary-v1/equilibrium"
ROOT_TOLERANCE = 2.0e-10
ROOT_OPTIONS = FixedPointOptions(
    method="anderson",
    tolerance=ROOT_TOLERANCE,
    max_iterations=100,
    damping=0.5,
    history=6,
)
SOURCE_REPLAY_RELATIVE_TOLERANCE = 1.0e-8
ENERGY_REPLAY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
DIRECTIONAL_RELATIVE_TOLERANCE = 2.0e-3
DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A = 5.0e-4
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/continuum/conjugate_fixed_topology_cpcm.py",
    "maple/solvation/continuum/fixed_reciprocal_functional.py",
    "maple/solvation/continuum/functional.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/coupling/variational_adapters.py",
    "maple/solvation/coupling/variational_state.py",
    "maple/solvation/models/field_energy.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_variational.py",
    "maple/solvation/release/evidence.py",
    "maple/function/calculator/mace/_macepol_calculator.py",
    "maple/function/calculator/mace/_macepol_long_range.py",
    "tools/route2_release/run_variational_common_water_canary.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a source-bound, disabled real-checkpoint common-scalar water "
            "canary; this does not admit Route-2 E/F/H/V/M."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]],
            dtype=float,
        ),
        info={"charge": 0, "mult": 1},
    )


def _direction(shape: tuple[int, int]) -> np.ndarray:
    values = np.random.default_rng(RANDOM_SEED).normal(size=shape)
    values -= np.mean(values, axis=0, keepdims=True)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise RuntimeError("The deterministic coordinate direction is singular.")
    return values / norm


def _directional_error(analytic: float, finite_difference: float) -> dict[str, object]:
    absolute = abs(analytic - finite_difference)
    relative = absolute / max(abs(analytic), abs(finite_difference), 1.0e-12)
    return {
        "analytic_eV_per_A": analytic,
        "finite_difference_eV_per_A": finite_difference,
        "absolute_error_eV_per_A": absolute,
        "relative_error": relative,
        "gate_passed": (
            absolute <= DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A
            and relative <= DIRECTIONAL_RELATIVE_TOLERANCE
        ),
    }


def _iteration_record(state) -> list[dict[str, object]]:
    return [
        {
            "iteration": item.iteration,
            "actual_unmixed_residual_norm": item.actual_unmixed_residual_norm,
            "step_norm": item.step_norm,
            "history_depth": item.history_depth,
        }
        for item in state.iterations
    ]


def _state_record(common, atoms: Atoms, state) -> dict[str, object]:
    scalar = common.evaluate_state(atoms, state)
    return {
        "initialization": state.initialization,
        "converged": state.converged,
        "iterations": len(state.iterations) - 1,
        "iteration_history": _iteration_record(state),
        "primal_tolerance": state.primal_tolerance,
        "actual_unmixed_residual_norm": state.actual_unmixed_residual_norm,
        "root_context_id": state.root_context_id,
        "root_hash": state.root_hash,
        "y": list(state.y),
        "source": state.source_array().tolist(),
        "field": state.field_array().tolist(),
        "model_energy_eV": scalar.model_energy_eV,
        "continuum_energy_eV": scalar.continuum_energy_eV,
        "coupling_energy_eV": scalar.coupling_energy_eV,
        "gauge_potential_eV_per_e": scalar.gauge_potential_eV_per_e,
        "total_energy_eV": scalar.total_energy_eV,
    }


def _configure_determinism(torch) -> None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _build_common(atoms: Atoms, checkpoint: Path, device: str):
    base = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=device,
    )
    model = MACEPolarVariationalFieldEnergy(base)
    backend = build_water_radial_gto_cpcm_backend(atoms.get_chemical_symbols())
    continuum = FixedReciprocalCPCMFunctional(
        backend,
        dtype=base._calculator.dtype,
        device=base._calculator.device,
    )
    common = build_variational_common_functional(
        model,
        continuum,
        atoms,
        scalar_id=VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_V1,
        profile_id=(VARIATIONAL_MACEPOLAR_ENERGYGRADIENT_FIXEDCAVITY_CPCM_PROFILE_V1),
    )
    return base, model, continuum, common


def _solve_center(common, atoms: Atoms):
    cold = common.solve_state(
        atoms,
        root_context_id=ROOT_CONTEXT_ID,
        options=ROOT_OPTIONS,
    )
    warm = common.solve_state(
        atoms,
        root_context_id=ROOT_CONTEXT_ID,
        initial_y=cold.y,
        options=ROOT_OPTIONS,
    )
    cold_record = _state_record(common, atoms, cold)
    warm_record = _state_record(common, atoms, warm)
    source_difference = float(np.linalg.norm(cold.source_array() - warm.source_array()))
    source_relative = source_difference / max(
        float(np.linalg.norm(cold.source_array())),
        float(np.linalg.norm(warm.source_array())),
        1.0e-15,
    )
    energy_difference = abs(
        float(cold_record["total_energy_eV"]) - float(warm_record["total_energy_eV"])
    )
    replay = {
        "contract": "route2-root-equivalence-v1",
        "numerically_equivalent": roots_numerically_equivalent(cold, warm),
        "source_l2_difference": source_difference,
        "source_relative_difference": source_relative,
        "energy_abs_difference_eV": energy_difference,
        "gate_passed": (
            roots_numerically_equivalent(cold, warm)
            and source_relative <= SOURCE_REPLAY_RELATIVE_TOLERANCE
            and energy_difference <= ENERGY_REPLAY_ABSOLUTE_TOLERANCE_EV
        ),
        "cold": cold_record,
        "warm": warm_record,
    }
    return cold, replay


def _envelope_record(common, atoms: Atoms, center_state):
    gradient = common.envelope_coordinate_gradient(atoms, center_state)
    values = np.asarray(gradient.total_coordinate_gradient_eV_per_A, dtype=float)
    direction = _direction(values.shape)
    analytic = float(np.vdot(values, direction))
    steps: list[dict[str, object]] = []
    timings: list[dict[str, float]] = []
    for step in FD_STEPS_ANGSTROM:
        energies: dict[int, float] = {}
        residuals: dict[int, float] = {}
        root_hashes: dict[int, str] = {}
        for sign in (1, -1):
            displaced = atoms.copy()
            displaced.positions += sign * step * direction
            started = time.perf_counter()
            state = common.solve_state(
                displaced,
                root_context_id=(f"{ROOT_CONTEXT_ID}/step={step:.1e}/sign={sign:+d}"),
                initial_y=center_state.y,
                options=ROOT_OPTIONS,
            )
            timings.append(
                {
                    "step_A": step,
                    "sign": float(sign),
                    "solve_seconds": time.perf_counter() - started,
                }
            )
            energies[sign] = common.evaluate_state(displaced, state).total_energy_eV
            residuals[sign] = state.actual_unmixed_residual_norm
            root_hashes[sign] = state.root_hash
        finite_difference = (energies[1] - energies[-1]) / (2.0 * step)
        steps.append(
            {
                "step_A": step,
                "plus_total_energy_eV": energies[1],
                "minus_total_energy_eV": energies[-1],
                "plus_primal_residual": residuals[1],
                "minus_primal_residual": residuals[-1],
                "plus_root_hash": root_hashes[1],
                "minus_root_hash": root_hashes[-1],
                **_directional_error(analytic, finite_difference),
            }
        )
    return (
        {
            "admitted": gradient.admitted,
            "coordinate_direction": direction.tolist(),
            "model_coordinate_gradient_eV_per_A": list(
                map(list, gradient.model_coordinate_gradient_eV_per_A)
            ),
            "continuum_coordinate_gradient_eV_per_A": list(
                map(list, gradient.continuum_coordinate_gradient_eV_per_A)
            ),
            "total_coordinate_gradient_eV_per_A": values.tolist(),
            "directional_steps": steps,
            "all_steps_passed": all(bool(item["gate_passed"]) for item in steps),
        },
        timings,
    )


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    _configure_determinism(torch)
    atoms = _water()
    base, model, continuum, common = _build_common(atoms, checkpoint, args.device)
    center_state, replay = _solve_center(common, atoms)
    envelope, solve_timings = _envelope_record(common, atoms, center_state)
    identity = {
        "scalar_id": common.scalar_id,
        "profile_id": common.profile_id,
        "state_equation_id": common.equation.state_equation_id,
        "common_functional_sha256": common.fingerprint_sha256(),
        "state_equation_sha256": common.equation.fingerprint_sha256(),
        "model_provider_id": model.provider_id,
        "model_profile_id": model.model_profile_id,
        "model_configuration_sha256": model.configuration_sha256(),
        "model_provenance_sha256": model.provenance_sha256,
        "field_graph_configuration_sha256": (model._field_graph.configuration_sha256()),
        "continuum_provider_id": continuum.provider_id,
        "continuum_profile_id": continuum.continuum_profile_id,
        "cavity_profile_id": continuum.cavity_profile_id,
        "continuum_configuration_contract_id": (continuum.configuration_contract_id),
        "continuum_configuration_sha256": continuum.configuration_sha256(),
        "continuum_provenance_sha256": continuum.provenance_sha256,
        "backend_configuration_sha256": continuum.backend.configuration_sha256(),
        "source_space_sha256": model.source_space.metadata_hash(),
        "field_space_sha256": model.field_space.metadata_hash(),
        "pairing_sha256": continuum.pairing.metadata_hash(),
        "duality_map_sha256": model.duality_map.configuration_sha256(),
        "conjugacy_sign": common.conjugacy_sign,
        "total_charge": common.total_charge,
    }
    geometry = {
        "formula": atoms.get_chemical_formula(),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "charge": 0,
        "multiplicity": 1,
    }
    protocol = {
        "random_seed": RANDOM_SEED,
        "root_context_id": ROOT_CONTEXT_ID,
        "root_options": {
            "method": ROOT_OPTIONS.method,
            "tolerance": ROOT_OPTIONS.tolerance,
            "max_iterations": ROOT_OPTIONS.max_iterations,
            "damping": ROOT_OPTIONS.damping,
            "history": ROOT_OPTIONS.history,
        },
        "fd_steps_A": list(FD_STEPS_ANGSTROM),
        "root_replay_thresholds": {
            "source_relative": SOURCE_REPLAY_RELATIVE_TOLERANCE,
            "energy_absolute_eV": ENERGY_REPLAY_ABSOLUTE_TOLERANCE_EV,
        },
        "directional_thresholds": {
            "relative": DIRECTIONAL_RELATIVE_TOLERANCE,
            "absolute_eV_per_A": DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A,
            "both_required": True,
        },
        "continuum_representation": (
            "fixed-topology amplitude-SWIG on a laboratory-fixed 194-point "
            "Lebedev rule; not a structural SO(3) representation"
        ),
    }
    decision = {
        "stationary_root_converged": bool(center_state.converged),
        "cold_warm_root_gate_passed": bool(replay["gate_passed"]),
        "three_step_envelope_directional_gate_passed": bool(
            envelope["all_steps_passed"]
        ),
        "same_scalar_real_checkpoint_canary_passed": bool(
            center_state.converged
            and replay["gate_passed"]
            and envelope["all_steps_passed"]
        ),
        "source_model_identity": (
            "changed-complete-eight-channel-field-energy-gradient-effective-source"
        ),
        "original_density_head_role": "zero-field-anchor-and-diagnostic-only",
        "global_so3_guarantee": False,
        "tier_v_admitted": False,
        "public_force_admitted": False,
    }
    measured = {
        "protocol": protocol,
        "geometry": geometry,
        "identity": identity,
        "center_root_replay": replay,
        "envelope_directional_derivative": envelope,
        "decision": decision,
    }
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": (
            "same-scalar-canary-passed-not-admitted"
            if decision["same_scalar_real_checkpoint_canary_passed"]
            else "same-scalar-canary-failed-not-admitted"
        ),
        "claim_boundary": (
            "This one-water result checks a changed-source scalar-first model, "
            "one fixed-topology reciprocal C-PCM scalar, one stationary root, "
            "and one coordinate direction. It does not prove global passivity, "
            "root uniqueness, SO(3) covariance, a moving-cavity derivative, "
            "chemical accuracy, or Route-2 E/F/H/V/M admission."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": model.dtype,
        **measured,
        "measurement_sha256": canonical_json_sha256(measured),
        "solve_timings_seconds": solve_timings,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_VARIATIONAL_COMMON_WATER_CANARY="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "decision": decision,
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
