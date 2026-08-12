#!/usr/bin/env python3
"""Capture real-stack PES diagnostics for the disabled fixed-box/CPCM590 path.

This command evaluates exactly one registered operational scalar but never
publishes a Route-2 result and never changes E/F/H/V/M admission.  It requires
a clean checkout and writes the JSON artifact outside that checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Callable, Sequence

from ase import Atoms
import numpy as np

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
)
from maple.solvation.api.profiles import (
    DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.continuum import build_water_radial_gto_cpcm_590_candidate
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.spaces import (
    LinearChargeCoordinates,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import (
    ReducedStateEquation,
    geometry_sha256,
)
from maple.solvation.models import (
    ElectronicResponseEquationAdapter,
    VacuumScalarEquationAdapter,
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


SCHEMA_VERSION = "route2-fixedbox40-cpcm590-water-pes-diagnostic-v1"
DEFAULT_CHECKPOINT = Path.home() / ".cache" / "mace" / "MACEPOLAR1Mmodel"
DEFAULT_CARTESIAN_STEPS_A = (4.0e-4, 2.0e-4, 1.0e-4)
DEFAULT_ORIENTATION_COUNT = 6
RANDOM_SEED = 20260813
REQUIRED_SOURCE_PATHS = (
    "maple/solvation/api/profiles.py",
    "maple/solvation/api/scalar_registry.py",
    "maple/solvation/api/state_registry.py",
    "maple/solvation/continuum/conjugate_fixed_topology_cpcm.py",
    "maple/solvation/coupling/adjoint.py",
    "maple/solvation/coupling/energy.py",
    "maple/solvation/coupling/fixed_point.py",
    "maple/solvation/coupling/linearization.py",
    "maple/solvation/coupling/state_equation.py",
    "maple/solvation/models/equation_adapter.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_feature_vjp.py",
    "maple/solvation/release/evidence.py",
    "tools/route2_release/run_fixedbox590_water_pes_diagnostic.py",
)


def _parse_steps(values: Sequence[str]) -> tuple[float, ...]:
    steps = tuple(float(value) for value in values)
    if not steps or any(not math.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("Cartesian steps must be finite positive numbers.")
    if len(set(steps)) != len(steps):
        raise ValueError("Cartesian steps must be unique.")
    return steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("orientation", "cartesian", "full"),
        default="full",
        help="full runs cold/warm, symmetry orientations, and all Cartesian FDs",
    )
    parser.add_argument(
        "--cartesian-step",
        action="append",
        default=[],
        metavar="ANGSTROM",
    )
    parser.add_argument(
        "--orientation-count", type=int, default=DEFAULT_ORIENTATION_COUNT
    )
    parser.add_argument("--primal-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--max-iterations", type=int, default=100)
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


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    orthogonal, triangular = np.linalg.qr(matrix)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    orthogonal = orthogonal @ np.diag(signs)
    if np.linalg.det(orthogonal) < 0.0:
        orthogonal[:, 0] *= -1.0
    return orthogonal


def _rotate_radial(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    for raw_indices in ((2, 3, 4), (5, 6, 7)):
        cartesian = result[:, raw_indices][:, (2, 0, 1)]
        result[:, raw_indices] = (cartesian @ rotation.T)[:, (1, 2, 0)]
    return result


def _root_context(geometry: Atoms, label: str) -> str:
    return f"fixedbox40-cpcm590-water-pes-v1/{label}/{geometry_sha256(geometry)}"


def _build_system(atoms: Atoms, checkpoint: Path, device: str):
    profile = get_solvation_profile(
        DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1
    )
    if profile.enabled or profile.capabilities.enabled_tiers:
        raise RuntimeError("The diagnostic profile must remain completely disabled.")
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=device,
        long_range_evaluator_profile=MACEPOL_FORCED_RECIPROCAL_FIXED_BOX40_PROFILE,
    )
    continuum = build_water_radial_gto_cpcm_590_candidate(
        atoms.get_chemical_symbols()
    )
    coordinates = LinearChargeCoordinates(
        len(atoms),
        total_charge=0.0,
        source_space=MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
        component_scales=(1.0,) * 8,
    )
    equation = ReducedStateEquation(
        coordinates,
        ElectronicResponseEquationAdapter(model, MACE_POLAR_RADIAL_GTO_COUPLING_ID),
        continuum,
    )
    scalar = OperationalElectrostaticScalar(
        equation,
        VacuumScalarEquationAdapter(model),
        profile_id=DIAGNOSTIC_FIXED_BOX40_CPCM_590_RADIAL_GTO_PROFILE_V1,
    )
    return profile, model, continuum, equation, scalar


def _state_record(state, scalar, geometry: Atoms) -> dict[str, object]:
    evaluated = scalar.evaluate(geometry, state.y)
    return {
        "initialization": state.initialization,
        "iterations": len(state.iterations) - 1,
        "root_context_id": state.root_context_id,
        "root_hash": state.root_hash,
        "primal_tolerance": state.primal_tolerance,
        "actual_unmixed_residual_norm": state.actual_unmixed_residual_norm,
        "source": state.source_array().tolist(),
        "field": state.field_array().tolist(),
        "vacuum_energy_eV": evaluated.vacuum_energy,
        "continuum_energy_eV": evaluated.continuum_energy,
        "total_energy_eV": evaluated.total_energy,
    }


def _cold_warm_record(cold, warm, scalar, atoms: Atoms) -> dict[str, object]:
    cold_scalar = scalar.evaluate(atoms, cold.y)
    warm_scalar = scalar.evaluate(atoms, warm.y)
    source_difference = float(
        np.linalg.norm(cold.source_array() - warm.source_array())
    )
    field_difference = float(np.linalg.norm(cold.field_array() - warm.field_array()))
    energy_difference = abs(cold_scalar.total_energy - warm_scalar.total_energy)
    return {
        "contract": "route2-root-equivalence-v1",
        "numerically_equivalent": roots_numerically_equivalent(cold, warm),
        "source_l2_difference": source_difference,
        "field_l2_difference": field_difference,
        "energy_abs_difference_eV": energy_difference,
        "gates": {
            "source_le_1e-8": source_difference <= 1.0e-8,
            "energy_le_1e-8_eV": energy_difference <= 1.0e-8,
        },
        "cold": _state_record(cold, scalar, atoms),
        "warm": _state_record(warm, scalar, atoms),
    }


def _orientation_record(
    atoms: Atoms,
    base_state,
    base_gradient,
    solve: Callable[..., object],
    scalar,
    continuum,
    count: int,
) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    base_forces = np.asarray(base_gradient.forces).reshape(len(atoms), 3)
    base_energy = base_gradient.scalar.total_energy
    base_source = base_state.source_array()
    base_surface = continuum.build_state(atoms, base_state.source)
    hashes = {base_surface.surface.topology_hash}
    records: list[dict[str, object]] = []
    for index in range(count):
        rotation = _random_rotation(rng)
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        state = solve(rotated, label=f"rotation-{index}")
        gradient = scalar.implicit_gradient(rotated, state)
        forces = np.asarray(gradient.forces).reshape(len(atoms), 3)
        topology = continuum.build_state(rotated, state.source).surface.topology_hash
        hashes.add(topology)
        records.append(
            {
                "index": index,
                "rotation_matrix": rotation.tolist(),
                "rotation_sha256": hashlib.sha256(rotation.tobytes()).hexdigest(),
                "determinant": float(np.linalg.det(rotation)),
                "energy_abs_eV": abs(gradient.scalar.total_energy - base_energy),
                "force_covariance_relative": float(
                    np.linalg.norm(forces - base_forces @ rotation.T)
                    / max(np.linalg.norm(base_forces), 1.0)
                ),
                "source_covariance_relative": float(
                    np.linalg.norm(
                        state.source_array() - _rotate_radial(base_source, rotation)
                    )
                    / max(np.linalg.norm(base_source), 1.0)
                ),
                "primal_residual": state.actual_unmixed_residual_norm,
                "adjoint_residual": gradient.adjoint.true_residual_norm,
                "topology_hash": topology,
            }
        )
    centred = atoms.positions - np.mean(atoms.positions, axis=0)
    torque = float(np.linalg.norm(np.sum(np.cross(centred, base_forces), axis=0)))
    net_force = float(np.linalg.norm(np.sum(base_forces, axis=0)))
    maxima = {
        "energy_abs_eV": max(item["energy_abs_eV"] for item in records),
        "force_covariance_relative": max(
            item["force_covariance_relative"] for item in records
        ),
        "source_covariance_relative": max(
            item["source_covariance_relative"] for item in records
        ),
    }
    return {
        "seed": RANDOM_SEED,
        "orientation_count": count,
        "base_net_force_norm_eV_per_A": net_force,
        "base_torque_norm_eV": torque,
        "maximum": maxima,
        "topology_hashes": sorted(hashes),
        "gates": {
            "rotation_energy_le_1e-6_eV": maxima["energy_abs_eV"] <= 1.0e-6,
            "rotation_force_relative_le_1e-4": (
                maxima["force_covariance_relative"] <= 1.0e-4
            ),
            "torque_le_1e-4_eV": torque <= 1.0e-4,
            "net_force_le_1e-5_eV_per_A": net_force <= 1.0e-5,
            "fixed_topology": len(hashes) == 1,
        },
        "records": records,
    }


def _cartesian_record(
    atoms: Atoms,
    base_state,
    base_gradient,
    solve: Callable[..., object],
    scalar,
    continuum,
    steps: tuple[float, ...],
) -> dict[str, object]:
    analytic = np.asarray(base_gradient.total_coordinate_gradient).reshape(
        len(atoms), 3
    )
    maximum_residual = base_state.actual_unmixed_residual_norm
    topology_hashes = {
        continuum.build_state(atoms, base_state.source).surface.topology_hash
    }
    records: list[dict[str, object]] = []
    for step in steps:
        finite_difference = np.zeros_like(analytic)
        for atom in range(len(atoms)):
            for axis in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                plus.positions[atom, axis] += step
                minus.positions[atom, axis] -= step
                plus_state = solve(
                    plus,
                    label=f"cartesian/{step}/atom-{atom}/axis-{axis}/plus",
                    initial_y=base_state.y_array(),
                )
                minus_state = solve(
                    minus,
                    label=f"cartesian/{step}/atom-{atom}/axis-{axis}/minus",
                    initial_y=base_state.y_array(),
                )
                maximum_residual = max(
                    maximum_residual,
                    plus_state.actual_unmixed_residual_norm,
                    minus_state.actual_unmixed_residual_norm,
                )
                plus_continuum = continuum.build_state(plus, plus_state.source)
                minus_continuum = continuum.build_state(minus, minus_state.source)
                topology_hashes.update(
                    (
                        plus_continuum.surface.topology_hash,
                        minus_continuum.surface.topology_hash,
                    )
                )
                finite_difference[atom, axis] = (
                    scalar.evaluate(plus, plus_state.y).total_energy
                    - scalar.evaluate(minus, minus_state.y).total_energy
                ) / (2.0 * step)
            print(
                f"completed Cartesian step={step:.8g} atom={atom}",
                file=sys.stderr,
                flush=True,
            )
        error = analytic - finite_difference
        rms = float(np.sqrt(np.mean(error**2)))
        maximum = float(np.max(np.abs(error)))
        records.append(
            {
                "step_A": step,
                "analytic_gradient_eV_per_A": analytic.tolist(),
                "finite_difference_gradient_eV_per_A": finite_difference.tolist(),
                "error_eV_per_A": error.tolist(),
                "rms_error_eV_per_A": rms,
                "maximum_error_eV_per_A": maximum,
                "relative_frobenius_error": float(
                    np.linalg.norm(error)
                    / max(np.linalg.norm(finite_difference), 1.0e-15)
                ),
                "gates": {
                    "rms_le_5e-4_eV_per_A": rms <= 5.0e-4,
                    "maximum_le_2e-3_eV_per_A": maximum <= 2.0e-3,
                },
            }
        )
    convergence_ratios = [
        records[index]["rms_error_eV_per_A"]
        / records[index + 1]["rms_error_eV_per_A"]
        for index in range(len(records) - 1)
    ]
    return {
        "records": records,
        "rms_error_ratios_in_given_step_order": convergence_ratios,
        "maximum_displaced_primal_residual": maximum_residual,
        "topology_hashes": sorted(topology_hashes),
        "fixed_topology": len(topology_hashes) == 1,
    }


def _translation_record(
    atoms: Atoms,
    cold,
    base_gradient,
    solve: Callable[..., object],
    scalar,
    continuum,
    adjoint_options: AdjointOptions,
    base_topology_hash: str,
) -> dict[str, object]:
    translation = np.asarray([1.7, -0.8, 0.5])
    translated = atoms.copy()
    translated.positions += translation
    state = solve(translated, label="translated", initial_y=cold.y_array())
    gradient = scalar.implicit_gradient(
        translated, state, adjoint_options=adjoint_options
    )
    base_forces = np.asarray(base_gradient.forces).reshape(len(atoms), 3)
    forces = np.asarray(gradient.forces).reshape(len(atoms), 3)
    energy_error = abs(
        gradient.scalar.total_energy - base_gradient.scalar.total_energy
    )
    force_error = float(
        np.linalg.norm(forces - base_forces)
        / max(np.linalg.norm(base_forces), 1.0)
    )
    net_force = float(np.linalg.norm(np.sum(forces, axis=0)))
    topology_hash = continuum.build_state(
        translated, state.source
    ).surface.topology_hash
    return {
        "translation_A": translation.tolist(),
        "energy_abs_eV": energy_error,
        "force_relative": force_error,
        "net_force_norm_eV_per_A": net_force,
        "primal_residual": state.actual_unmixed_residual_norm,
        "adjoint_residual": gradient.adjoint.true_residual_norm,
        "topology_hash": topology_hash,
        "gates": {
            "energy_le_1e-6_eV": energy_error <= 1.0e-6,
            "force_relative_le_1e-4": force_error <= 1.0e-4,
            "net_force_le_1e-5_eV_per_A": net_force <= 1.0e-5,
            "fixed_topology": topology_hash == base_topology_hash,
        },
    }


def _identity_record(model, continuum, equation, scalar) -> dict[str, object]:
    return {
        "profile_id": scalar.profile_id,
        "scalar_id": scalar.scalar_id,
        "state_equation_id": equation.state_equation_id,
        "model_provider_id": model.provider_id,
        "model_profile_id": model.model_profile_id,
        "model_provenance_sha256": model.provenance_sha256,
        "model_provenance": model.provenance.metadata(),
        "model_configuration_sha256": model.configuration_sha256(),
        "long_range_evaluator_provenance": (
            model._calculator.long_range_evaluator_provenance
        ),
        "continuum_provider_id": continuum.provider_id,
        "continuum_profile_id": continuum.continuum_profile_id,
        "cavity_profile_id": continuum.cavity_profile_id,
        "coupling_id": continuum.coupling_id,
        "continuum_configuration_contract_id": (
            continuum.configuration_contract_id
        ),
        "continuum_configuration_sha256": continuum.configuration_sha256(),
        "continuum_provenance_sha256": continuum.provenance_sha256,
        "continuum_runtime_provenance": dict(continuum.runtime_provenance),
        "equation_sha256": equation.fingerprint_sha256(),
        "scalar_sha256": scalar.fingerprint_sha256(),
    }


def _base_state_record(
    atoms: Atoms, cold, base_gradient, continuum_state, continuum
) -> dict[str, object]:
    return {
        "primal_residual": cold.actual_unmixed_residual_norm,
        "adjoint_residual": base_gradient.adjoint.true_residual_norm,
        "adjoint_acceptance_tolerance": base_gradient.adjoint.acceptance_tolerance,
        "root_hash": cold.root_hash,
        "surface_candidate_count": continuum_state.surface.candidate_count,
        "surface_topology_hash": continuum_state.surface.topology_hash,
        "surface_state_hash": continuum_state.surface.state_hash,
        "half_coupling_error_eV": abs(
            continuum_state.polarization_energy_ev
            - 0.5
            * continuum.pairing.pair(
                cold.source_array(), continuum_state.reaction_field
            )
        ),
        "vacuum_energy_eV": base_gradient.scalar.vacuum_energy,
        "continuum_energy_eV": base_gradient.scalar.continuum_energy,
        "total_energy_eV": base_gradient.scalar.total_energy,
        "forces_eV_per_A": np.asarray(base_gradient.forces)
        .reshape(len(atoms), 3)
        .tolist(),
    }


def main() -> None:
    args = _parse_args()
    if type(args.orientation_count) is not int or args.orientation_count < 1:
        raise ValueError("--orientation-count must be a positive integer.")
    steps = (
        _parse_steps(args.cartesian_step)
        if args.cartesian_step
        else DEFAULT_CARTESIAN_STEPS_A
    )
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    atoms = _water()
    profile, model, continuum, equation, scalar = _build_system(
        atoms, checkpoint, args.device
    )
    primal_options = FixedPointOptions(
        tolerance=args.primal_tolerance,
        max_iterations=args.max_iterations,
        damping=0.7,
        history=6,
    )
    adjoint_options = AdjointOptions(
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
        max_iterations=500,
    )

    def solve(geometry: Atoms, *, label: str, initial_y=None):
        return solve_fixed_point(
            equation,
            geometry,
            scalar_id=scalar.scalar_id,
            profile_id=scalar.profile_id,
            scalar_binding=scalar,
            root_context_id=_root_context(geometry, label),
            initial_y=initial_y,
            options=primal_options,
        )

    cold = solve(atoms, label="base")
    warm = solve(
        atoms,
        label="base",
        initial_y=cold.y_array() + 1.0e-6,
    )
    base_gradient = scalar.implicit_gradient(
        atoms, cold, adjoint_options=adjoint_options
    )
    continuum_state = continuum.build_state(atoms, cold.source)
    sections: dict[str, object] = {
        "cold_warm": _cold_warm_record(cold, warm, scalar, atoms),
    }
    if args.mode in ("orientation", "full"):
        sections["orientations"] = _orientation_record(
            atoms,
            cold,
            base_gradient,
            solve,
            scalar,
            continuum,
            args.orientation_count,
        )
        sections["translation"] = _translation_record(
            atoms,
            cold,
            base_gradient,
            solve,
            scalar,
            continuum,
            adjoint_options,
            continuum_state.surface.topology_hash,
        )
    if args.mode in ("cartesian", "full"):
        sections["cartesian_force_fd"] = _cartesian_record(
            atoms,
            cold,
            base_gradient,
            solve,
            scalar,
            continuum,
            steps,
        )

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=REQUIRED_SOURCE_PATHS,
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    identities = _identity_record(model, continuum, equation, scalar)
    base_state = _base_state_record(
        atoms, cold, base_gradient, continuum_state, continuum
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-pes-diagnostic",
        "status": "diagnostic-success",
        "claim_boundary": (
            "One-water evidence for a disabled fixed-box40/CPCM590 candidate; "
            "not a public result, not Tier E/F/H/V/M, not a complete solvation "
            "free energy, and not chemical-accuracy evidence."
        ),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "checkpoint": checkpoint_record(checkpoint),
        "runtime": runtime_record(),
        "seed": RANDOM_SEED,
        "device": args.device,
        "dtype": model.dtype,
        "geometry": {
            "name": "water-equilibrium-diagnostic",
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "geometry_sha256": geometry_sha256(atoms),
        },
        "identities": identities,
        "base_state": base_state,
        "measurements": sections,
        "validation_scope": {
            "single_molecule": True,
            "single_geometry": True,
            "multi_molecule_pes_panel": False,
            "box_convergence": False,
            "closed_loop_work": False,
            "complete_nonpolar_free_energy": False,
            "chemical_accuracy": False,
        },
        "profile_registry_disabled": (
            not profile.enabled and not profile.capabilities.enabled_tiers
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "identities": payload["identities"],
            "base_state": payload["base_state"],
            "measurements": payload["measurements"],
        }
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_WATER_PES_DIAGNOSTIC="
        + json.dumps(
            {
                "artifact": file_record,
                "measurement_sha256": payload["measurement_sha256"],
                "profile_id": scalar.profile_id,
                "capabilities": payload["capabilities"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
