#!/usr/bin/env python3
"""Run the disabled original-source operational harmonic water canary.

The unique scalar is ``E_vac + G_harm`` along the constrained root
``c=M_orig(grad_Q G_harm)``.  The MACE evaluator uses the separately identified
analytic isotropic Gaussian multipole implementation, while the electronic
source remains the original four-channel density head embedded in the first
radial block.  This runner never admits Route-2 E/F/H/V/M.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.continuum import (
    radial_gto_source_rotation_matrix,
)
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
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

from operational_analytic_harmonic_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    EXPOSURE_LMAX,
    EXPOSURE_RADIAL_QUADRATURE_ORDER,
    GREEN_RADIAL_QUADRATURE_ORDER,
    NO_CAPABILITIES,
    SOURCE_RADIAL_QUADRATURE_ORDER,
    SURFACE_LMAX,
    TRANSITION_WIDTH_ANGSTROM2,
    build_system,
)

SCHEMA_VERSION = (
    "route2-operational-analytic-original-source-harmonic-water-"
    "real-checkpoint-canary-v1"
)
ARTIFACT_KIND = (
    "disabled-real-checkpoint-operational-analytic-original-source-"
    "harmonic-scalar-canary"
)
RANDOM_SEED = 20260818
ROTATION_SEED = 20260819
ROOT_CONTEXT_ID = "route2-operational-analytic-harmonic-water-canary-v1/equilibrium"
ROOT_OPTIONS = FixedPointOptions(
    method="anderson",
    tolerance=2.0e-10,
    max_iterations=120,
    damping=0.5,
    history=6,
)
ADJOINT_OPTIONS = AdjointOptions(
    relative_tolerance=1.0e-11,
    absolute_tolerance=1.0e-13,
    max_iterations=600,
)
FD_STEPS_ANGSTROM = (5.0e-4, 2.0e-4, 1.0e-4)
SOURCE_REPLAY_RELATIVE_TOLERANCE = 1.0e-8
ENERGY_REPLAY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
DIRECTIONAL_RELATIVE_TOLERANCE = 2.0e-3
DIRECTIONAL_ABSOLUTE_TOLERANCE_EV_PER_A = 5.0e-4
CONTINUUM_SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV = 1.0e-10
MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE = 1.0e-8
ROTATION_ENERGY_ABSOLUTE_TOLERANCE_EV = 1.0e-8
ROTATION_SOURCE_RELATIVE_TOLERANCE = 1.0e-7
ROTATION_FIELD_RELATIVE_TOLERANCE = 1.0e-7
ROTATION_FORCE_RELATIVE_TOLERANCE = 2.0e-6
ROTATION_FORCE_MAX_ABSOLUTE_TOLERANCE_EV_PER_A = 5.0e-7
CONTINUUM_ROTATION_ENERGY_ABSOLUTE_TOLERANCE_EV = 1.0e-10
CONTINUUM_ROTATION_GRADIENT_ABSOLUTE_TOLERANCE_EV_PER_A = 1.0e-9
NET_FORCE_TOLERANCE_EV_PER_A = 5.0e-7
TORQUE_TOLERANCE_EV = 5.0e-7
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "tools/route2_release/run_operational_analytic_harmonic_water_canary.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the disabled operational analytic/original-source harmonic "
            "water canary; this does not admit Route-2 E/F/H/V/M."
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


def _rotation() -> np.ndarray:
    matrix = np.random.default_rng(ROTATION_SEED).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15, rtol=0.0):
        raise RuntimeError("The deterministic rotation lost orthogonality.")
    return rotation


def _relative_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = float(np.linalg.norm(actual - expected))
    return difference / max(
        float(np.linalg.norm(actual)), float(np.linalg.norm(expected)), 1.0e-15
    )


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


def _measurement_record(**records: object) -> dict[str, object]:
    return dict(records)


def _configure_determinism(torch) -> None:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _build_scalar(atoms: Atoms, checkpoint: Path, device: str):
    _, model, continuum, _, scalar = build_system(atoms, checkpoint, device)
    return model, continuum, scalar


def _solve(scalar, atoms: Atoms, *, context: str, initial_y=None):
    return solve_fixed_point(
        scalar.equation,
        atoms,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id=context,
        initial_y=initial_y,
        options=ROOT_OPTIONS,
    )


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


def _state_record(scalar, atoms: Atoms, state) -> dict[str, object]:
    energy = scalar.evaluate_energy_components(atoms, state.y)
    source = state.source_array()
    field = state.field_array()
    coupling = float(scalar.metric.pair(source, field))
    missing = source[:, (1, 5, 6, 7)]
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
        "source": source.tolist(),
        "field": field.tolist(),
        "vacuum_energy_eV": energy.vacuum_energy,
        "continuum_energy_eV": energy.continuum_energy,
        "coupling_energy_eV": coupling,
        "total_energy_eV": energy.total_energy,
        "missing_radial_block_l2": float(np.linalg.norm(missing)),
        "missing_radial_block_max_abs": float(np.max(np.abs(missing))),
    }


def _center_replay(scalar, atoms: Atoms):
    started = time.perf_counter()
    cold = _solve(scalar, atoms, context=ROOT_CONTEXT_ID)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm = _solve(scalar, atoms, context=ROOT_CONTEXT_ID, initial_y=cold.y)
    warm_seconds = time.perf_counter() - started
    cold_record = _state_record(scalar, atoms, cold)
    warm_record = _state_record(scalar, atoms, warm)
    source_relative = _relative_difference(cold.source_array(), warm.source_array())
    energy_absolute = abs(
        float(cold_record["total_energy_eV"]) - float(warm_record["total_energy_eV"])
    )
    equivalent = roots_numerically_equivalent(cold, warm)
    replay = {
        "contract": "route2-root-equivalence-v1",
        "numerically_equivalent": equivalent,
        "source_relative_difference": source_relative,
        "energy_abs_difference_eV": energy_absolute,
        "gate_passed": (
            equivalent
            and source_relative <= SOURCE_REPLAY_RELATIVE_TOLERANCE
            and energy_absolute <= ENERGY_REPLAY_ABSOLUTE_TOLERANCE_EV
        ),
        "cold": cold_record,
        "warm": warm_record,
    }
    return cold, replay, {"cold": cold_seconds, "warm": warm_seconds}


def _scalar_identity(scalar, atoms: Atoms, state) -> dict[str, object]:
    source = state.source_array()
    field = state.field_array()
    half_coupling = 0.5 * float(scalar.metric.pair(source, field))
    functional_energy = scalar.equation.continuum.functional.energy_eV(atoms, source)
    error = abs(functional_energy - half_coupling)
    missing = source[:, (1, 5, 6, 7)]
    missing_max = float(np.max(np.abs(missing)))
    return {
        "functional_energy_eV": functional_energy,
        "half_coupling_energy_eV": half_coupling,
        "absolute_error_eV": error,
        "missing_radial_block_max_abs": missing_max,
        "thresholds": {
            "scalar_identity_absolute_eV": (
                CONTINUUM_SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV
            ),
            "missing_radial_block_max_abs": (MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE),
        },
        "gate_passed": (
            error <= CONTINUUM_SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV
            and missing_max <= MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE
        ),
    }


def _force_directional(scalar, atoms: Atoms, center_state):
    started = time.perf_counter()
    gradient = scalar.implicit_gradient(
        atoms, center_state, adjoint_options=ADJOINT_OPTIONS
    )
    gradient_seconds = time.perf_counter() - started
    values = np.asarray(gradient.total_coordinate_gradient).reshape(len(atoms), 3)
    forces = -values
    direction = _direction(values.shape)
    analytic = float(np.vdot(values, direction))
    records: list[dict[str, object]] = []
    solve_timings: list[dict[str, float]] = []
    for step in FD_STEPS_ANGSTROM:
        energies: dict[int, float] = {}
        residuals: dict[int, float] = {}
        for sign in (1, -1):
            displaced = atoms.copy()
            displaced.positions += sign * step * direction
            started = time.perf_counter()
            state = _solve(
                scalar,
                displaced,
                context=f"{ROOT_CONTEXT_ID}/step={step:.1e}/sign={sign:+d}",
                initial_y=center_state.y,
            )
            solve_timings.append(
                {
                    "step_A": step,
                    "sign": float(sign),
                    "solve_seconds": time.perf_counter() - started,
                }
            )
            energies[sign] = scalar.evaluate_energy(displaced, state.y)
            residuals[sign] = state.actual_unmixed_residual_norm
        finite_difference = (energies[1] - energies[-1]) / (2.0 * step)
        records.append(
            {
                "step_A": step,
                "plus_total_energy_eV": energies[1],
                "minus_total_energy_eV": energies[-1],
                "plus_primal_residual": residuals[1],
                "minus_primal_residual": residuals[-1],
                **_directional_error(analytic, finite_difference),
            }
        )
    centered = atoms.positions - np.mean(atoms.positions, axis=0, keepdims=True)
    net_force = np.sum(forces, axis=0)
    torque = np.sum(np.cross(centered, forces), axis=0)
    record = {
        "admitted": False,
        "coordinate_direction": direction.tolist(),
        "total_coordinate_gradient_eV_per_A": values.tolist(),
        "forces_eV_per_A": forces.tolist(),
        "adjoint": {
            "converged": gradient.adjoint.converged,
            "iterations": gradient.adjoint.iterations,
            "true_residual_norm": gradient.adjoint.true_residual_norm,
            "acceptance_tolerance": gradient.adjoint.acceptance_tolerance,
        },
        "directional_steps": records,
        "all_steps_passed": all(bool(item["gate_passed"]) for item in records),
        "net_force_norm_eV_per_A": float(np.linalg.norm(net_force)),
        "torque_norm_eV": float(np.linalg.norm(torque)),
        "net_force_gate_passed": (
            float(np.linalg.norm(net_force)) <= NET_FORCE_TOLERANCE_EV_PER_A
        ),
        "torque_gate_passed": float(np.linalg.norm(torque)) <= TORQUE_TOLERANCE_EV,
    }
    return gradient, record, gradient_seconds, solve_timings


def _rotation_record(scalar, atoms: Atoms, center_state, center_gradient):
    rotation = _rotation()
    representation = radial_gto_source_rotation_matrix(rotation, atom_count=len(atoms))
    expected_source = (
        representation @ center_state.source_array().reshape(-1)
    ).reshape(center_state.source_array().shape)
    expected_field = (representation @ center_state.field_array().reshape(-1)).reshape(
        center_state.field_array().shape
    )
    center_forces = -np.asarray(center_gradient.total_coordinate_gradient).reshape(
        len(atoms), 3
    )
    expected_forces = center_forces @ rotation.T
    center_energy = center_gradient.scalar.total_energy

    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    initial_y = scalar.equation.coordinates.reduce(expected_source)
    started = time.perf_counter()
    rotated_state = _solve(
        scalar,
        rotated,
        context=f"{ROOT_CONTEXT_ID}/rigid-rotation-seed={ROTATION_SEED}",
        initial_y=initial_y,
    )
    solve_seconds = time.perf_counter() - started
    started = time.perf_counter()
    rotated_gradient = scalar.implicit_gradient(
        rotated, rotated_state, adjoint_options=ADJOINT_OPTIONS
    )
    gradient_seconds = time.perf_counter() - started
    rotated_forces = -np.asarray(rotated_gradient.total_coordinate_gradient).reshape(
        len(atoms), 3
    )
    source_relative = _relative_difference(
        rotated_state.source_array(), expected_source
    )
    field_relative = _relative_difference(rotated_state.field_array(), expected_field)
    force_relative = _relative_difference(rotated_forces, expected_forces)
    force_maximum = float(np.max(np.abs(rotated_forces - expected_forces)))
    energy_absolute = abs(rotated_gradient.scalar.total_energy - center_energy)

    functional = scalar.equation.continuum.functional
    center_continuum_energy = functional.energy_eV(atoms, center_state.source_array())
    rotated_continuum_energy = functional.energy_eV(rotated, expected_source)
    center_continuum_gradient = functional.coordinate_partial(
        atoms, center_state.source_array()
    )
    rotated_continuum_gradient = functional.coordinate_partial(rotated, expected_source)
    expected_continuum_gradient = center_continuum_gradient @ rotation.T
    continuum_energy_error = abs(rotated_continuum_energy - center_continuum_energy)
    continuum_gradient_error = float(
        np.max(np.abs(rotated_continuum_gradient - expected_continuum_gradient))
    )
    continuum_gate = (
        continuum_energy_error <= CONTINUUM_ROTATION_ENERGY_ABSOLUTE_TOLERANCE_EV
        and continuum_gradient_error
        <= CONTINUUM_ROTATION_GRADIENT_ABSOLUTE_TOLERANCE_EV_PER_A
    )
    operational_gate = (
        rotated_state.converged
        and source_relative <= ROTATION_SOURCE_RELATIVE_TOLERANCE
        and field_relative <= ROTATION_FIELD_RELATIVE_TOLERANCE
        and energy_absolute <= ROTATION_ENERGY_ABSOLUTE_TOLERANCE_EV
        and force_relative <= ROTATION_FORCE_RELATIVE_TOLERANCE
        and force_maximum <= ROTATION_FORCE_MAX_ABSOLUTE_TOLERANCE_EV_PER_A
    )
    return (
        {
            "rotation_seed": ROTATION_SEED,
            "rotation_matrix": rotation.tolist(),
            "continuum_structural_check": {
                "energy_absolute_error_eV": continuum_energy_error,
                "gradient_max_absolute_error_eV_per_A": continuum_gradient_error,
                "gate_passed": continuum_gate,
            },
            "operational_scalar_check": {
                "converged": rotated_state.converged,
                "actual_unmixed_residual_norm": (
                    rotated_state.actual_unmixed_residual_norm
                ),
                "source_relative_error": source_relative,
                "field_relative_error": field_relative,
                "energy_absolute_error_eV": energy_absolute,
                "force_relative_error": force_relative,
                "force_max_absolute_error_eV_per_A": force_maximum,
                "gate_passed": operational_gate,
            },
            "gate_passed": continuum_gate and operational_gate,
        },
        {"solve": solve_seconds, "gradient": gradient_seconds},
    )


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    _configure_determinism(torch)
    atoms = _water()
    model, continuum, scalar = _build_scalar(atoms, checkpoint, args.device)
    center_state, replay, replay_timings = _center_replay(scalar, atoms)
    scalar_identity = _scalar_identity(scalar, atoms, center_state)
    center_gradient, force, gradient_seconds, displacement_timings = _force_directional(
        scalar, atoms, center_state
    )
    rotation, rotation_timings = _rotation_record(
        scalar, atoms, center_state, center_gradient
    )

    identity = {
        "scalar_id": scalar.scalar_id,
        "profile_id": scalar.profile_id,
        "state_equation_id": scalar.equation.state_equation_id,
        "operational_scalar_sha256": scalar.fingerprint_sha256(),
        "state_equation_sha256": scalar.equation.fingerprint_sha256(),
        "model_provider_id": model.provider_id,
        "model_profile_id": model.model_profile_id,
        "model_configuration_sha256": model.configuration_sha256(),
        "model_provenance_sha256": model.provenance_sha256,
        "continuum_provider_id": continuum.provider_id,
        "continuum_profile_id": continuum.continuum_profile_id,
        "cavity_profile_id": continuum.cavity_profile_id,
        "continuum_configuration_contract_id": continuum.configuration_contract_id,
        "continuum_configuration_sha256": continuum.configuration_sha256(),
        "continuum_provenance_sha256": continuum.provenance_sha256,
        "source_space_sha256": model.source_space.metadata_hash(),
        "field_space_sha256": model.field_space.metadata_hash(),
        "pairing_sha256": continuum.pairing.metadata_hash(),
        "long_range_evaluator_profile": (
            model.release_contract.long_range_evaluator_profile
        ),
        "long_range_symmetry_contract_id": (
            model.release_contract.long_range_symmetry_contract_id
        ),
        "long_range_structural_so3_equivariance_admitted": (
            model.release_contract.structural_so3_equivariance_admitted
        ),
    }
    geometry = {
        "formula": atoms.get_chemical_formula(),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "charge": 0,
        "multiplicity": 1,
        "cavity_radii_A": list(continuum.radii_angstrom),
    }
    protocol = {
        "random_seed": RANDOM_SEED,
        "rotation_seed": ROTATION_SEED,
        "root_context_id": ROOT_CONTEXT_ID,
        "root_options": {
            "method": ROOT_OPTIONS.method,
            "tolerance": ROOT_OPTIONS.tolerance,
            "max_iterations": ROOT_OPTIONS.max_iterations,
            "damping": ROOT_OPTIONS.damping,
            "history": ROOT_OPTIONS.history,
        },
        "fd_steps_A": list(FD_STEPS_ANGSTROM),
        "harmonic_configuration": {
            "surface_lmax": SURFACE_LMAX,
            "exposure_lmax": EXPOSURE_LMAX,
            "transition_width_A2": TRANSITION_WIDTH_ANGSTROM2,
            "exposure_radial_quadrature_order": (EXPOSURE_RADIAL_QUADRATURE_ORDER),
            "source_radial_quadrature_order": SOURCE_RADIAL_QUADRATURE_ORDER,
            "green_radial_quadrature_order": GREEN_RADIAL_QUADRATURE_ORDER,
            "cavity_identity": "regularized-smooth-weighted-overlap-not-sharp-union",
        },
        "source_model_identity": (
            "original-four-channel-density-head-embedded-first-radial-block"
        ),
        "continuum_representation": (
            "fixed-dimensional complete real-harmonic irreps; invariant 1D "
            "geometry quadrature and finite-band-exact contractions; no "
            "laboratory-fixed cavity grid"
        ),
    }
    decision = {
        "stationary_root_converged": bool(center_state.converged),
        "cold_warm_root_gate_passed": bool(replay["gate_passed"]),
        "continuum_scalar_identity_gate_passed": bool(scalar_identity["gate_passed"]),
        "three_step_force_directional_gate_passed": bool(force["all_steps_passed"]),
        "net_force_gate_passed": bool(force["net_force_gate_passed"]),
        "torque_gate_passed": bool(force["torque_gate_passed"]),
        "rigid_rotation_gate_passed": bool(rotation["gate_passed"]),
        "source_model_identity": (
            "original-four-channel-density-head-embedded-first-radial-block"
        ),
        "field_conditioned_model_energy_difference_included": False,
        "continuum_coefficient_architecture_is_so3_equivariant": True,
        "operational_canary_passed": bool(
            center_state.converged
            and replay["gate_passed"]
            and scalar_identity["gate_passed"]
            and force["all_steps_passed"]
            and force["net_force_gate_passed"]
            and force["torque_gate_passed"]
            and rotation["gate_passed"]
        ),
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "tier_v_admitted": False,
    }
    measured = _measurement_record(
        protocol=protocol,
        geometry=geometry,
        identity=identity,
        center_root_replay=replay,
        scalar_identity=scalar_identity,
        force_directional_derivative=force,
        rigid_rotation=rotation,
        decision=decision,
    )

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": (
            "operational-canary-passed-not-admitted"
            if decision["operational_canary_passed"]
            else "operational-canary-failed-not-admitted"
        ),
        "claim_boundary": (
            "This one-water result checks one separately registered operational "
            "scalar, the analytic Gaussian multipole evaluator, the original "
            "four-channel density response, one smooth harmonic continuum, one "
            "coordinate direction, and one rigid rotation. It does not prove "
            "chemical accuracy, conformer/domain coverage, global root uniqueness, "
            "closed-loop conservativity, Hessian/FREQ/MD readiness, sharp-union "
            "equivalence, or Route-2 E/F/H/V/M admission."
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
        "timings_seconds": {
            "root_replay": replay_timings,
            "center_gradient": gradient_seconds,
            "displaced_roots": displacement_timings,
            "rotation": rotation_timings,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_OPERATIONAL_ANALYTIC_HARMONIC_WATER_CANARY="
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
