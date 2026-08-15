#!/usr/bin/env python3
"""Capture the disabled stationary-water dense Hessian/FREQ canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from scipy.optimize import root

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.geometry_mediated import (
    geometry_mediated_trial_step_guard,
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
from maple.solvation.release.geometry_mediated_frequency import (
    AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A,
    AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_CONTRACT_VERSION,
    summarize_aimnet2_geometry_mediated_frequency_water,
)
from maple.solvation.release.geometry_mediated_path import (
    aimnet2_geometry_mediated_water_loop_atoms,
)
from maple.solvation.release.geometry_mediated_stationary import (
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD,
    AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL,
    aimnet2_geometry_mediated_stationary_water_geometry,
    aimnet2_geometry_mediated_stationary_water_initial_coordinates,
)
from aimnet2_geometry_mediated_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    CONTINUUM_REQUIRED_SOURCE_PATHS,
    MODEL_RUNTIME_REQUIRED_SOURCE_PATHS,
    NO_CAPABILITIES,
    build_geometry_mediated_stack,
    geometry_mediated_topologies,
    verify_route2_checkpoint,
)

SCHEMA_VERSION = "route2-aimnet2-geometry-mediated-frequency-water-artifact-v1"
RUNTIME_KIND = "reconstructed-python-float64"
CONTINUUM_KIND = "harmonic-point"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/function/dispatcher/frequency/normal_modes.py",
    "maple/solvation/release/geometry_mediated_frequency.py",
    "maple/solvation/release/geometry_mediated_frequency_records.py",
    "maple/solvation/release/geometry_mediated_hessian.py",
    "maple/solvation/release/geometry_mediated_stationary.py",
    "tools/route2_release/run_aimnet2_geometry_mediated_frequency.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the disabled source-bound AIMNet2/smooth-harmonic "
            "stationary-water dense Hessian and vibrational-subspace canary."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _search_record(model, continuum, scalar, reference):
    initial = aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    baseline_model_topology, baseline_continuum_topology = geometry_mediated_topologies(
        model, continuum, reference
    )
    evaluations: list[dict[str, object]] = []
    cache: dict[str, object] = {}

    def evaluate_internal(coordinates):
        values = np.asarray(coordinates, dtype=float)
        key = values.tobytes()
        if cache.get("key") == key:
            return cache["state"]
        geometry = aimnet2_geometry_mediated_stationary_water_geometry(values)
        atoms = reference.copy()
        atoms.positions = geometry.positions_A
        model_topology, continuum_topology = geometry_mediated_topologies(
            model, continuum, atoms
        )
        if evaluations:
            previous = cache["state"]
            center_positions = previous["atoms"].positions
            center_model_topology = previous["record"]["model_topology"]
            center_continuum_topology = previous["record"]["continuum_topology"]
        else:
            center_positions = reference.positions
            center_model_topology = baseline_model_topology
            center_continuum_topology = baseline_continuum_topology
        guard = geometry_mediated_trial_step_guard(
            center_positions_A=center_positions,
            trial_positions_A=atoms.positions,
            center_model_topology=center_model_topology,
            trial_model_topology=model_topology,
            center_continuum_topology=center_continuum_topology,
            trial_continuum_topology=continuum_topology,
        )
        if not guard["gate_passed"]:
            raise RuntimeError("stationary root trial left the guarded smooth stratum.")
        result = scalar.evaluate(atoms)
        gradient = np.asarray(result.total_gradient_eV_per_A, dtype=float)
        internal_gradient = np.einsum("ick,ic->k", geometry.jacobian, gradient)
        record = {
            "internal_coordinates": values.tolist(),
            "positions_A": atoms.positions.tolist(),
            "geometry_sha256": geometry_sha256(atoms),
            "energy_eV": result.energy.total_energy_eV,
            "total_gradient_eV_per_A": gradient.tolist(),
            "internal_gradient": internal_gradient.tolist(),
            "model_topology": model_topology,
            "continuum_topology": continuum_topology,
        }
        state = {
            "geometry": geometry,
            "atoms": atoms,
            "result": result,
            "internal_gradient": internal_gradient,
            "record": record,
        }
        cache.clear()
        cache.update(key=key, state=state)
        evaluations.append(record)
        print(
            f"stationary root evaluation {len(evaluations)}: "
            f"|g_internal|={np.linalg.norm(internal_gradient):.6e}",
            file=sys.stderr,
            flush=True,
        )
        return state

    def function(coordinates):
        return evaluate_internal(coordinates)["internal_gradient"]

    def jacobian(coordinates):
        state = evaluate_internal(coordinates)
        geometry = state["geometry"]
        gradient = np.asarray(state["result"].total_gradient_eV_per_A, dtype=float)
        hessian_times_jacobian = np.stack(
            [
                scalar.hessian_vector_product(
                    state["atoms"], geometry.jacobian[:, :, coordinate]
                ).total_hvp_eV_per_A2
                for coordinate in range(3)
            ],
            axis=-1,
        )
        internal_hessian = np.einsum(
            "ica,icb->ab", geometry.jacobian, hessian_times_jacobian
        ) + np.einsum("ic,icab->ab", gradient, geometry.second_derivatives)
        if np.max(np.abs(internal_hessian - internal_hessian.T)) > 1.0e-7:
            raise RuntimeError("stationary-search internal Hessian lost symmetry.")
        return internal_hessian

    solved = root(
        function,
        initial,
        jac=jacobian,
        method=AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD,
        options={
            "factor": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR,
            "xtol": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL,
            "maxfev": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV,
        },
    )
    final_state = evaluate_internal(solved.x)
    if not solved.success:
        raise RuntimeError(f"stationary water root solve failed: {solved.message}")
    if len(evaluations) > int(solved.nfev):
        raise RuntimeError(
            "stationary root solution was not present in its solver trace."
        )
    search = {
        "protocol": {
            "solver": "scipy.optimize.root",
            "method": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_METHOD,
            "factor": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_FACTOR,
            "xtol": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_XTOL,
            "maxfev": AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_ROOT_MAXFEV,
            "internal_coordinate_names": list(
                AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_COORDINATE_NAMES
            ),
            "internal_bounds": [
                list(bounds)
                for bounds in AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS
            ],
            "public_optimizer": False,
        },
        "result": {
            "success": bool(solved.success),
            "status": int(solved.status),
            "message": str(solved.message),
            "nfev": int(solved.nfev),
            "njev": int(solved.njev),
            "solution": np.asarray(solved.x, dtype=float).tolist(),
        },
        "evaluations": evaluations,
    }
    return search, final_state["atoms"]


def _center_measurement(model, continuum, scalar, atoms):
    result = scalar.evaluate(atoms)
    replay = scalar.evaluate(atoms)
    model_topology, continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    return result, {
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "masses_amu": atoms.get_masses().tolist(),
        "positions_A": atoms.positions.tolist(),
        "energy": {
            "vacuum_energy_eV": result.energy.vacuum_energy_eV,
            "continuum_energy_eV": result.energy.continuum_energy_eV,
            "total_energy_eV": result.energy.total_energy_eV,
        },
        "source": result.source.tolist(),
        "reaction_field": result.reaction_field.tolist(),
        "intrinsic_gradient_eV_per_A": result.intrinsic_gradient_eV_per_A.tolist(),
        "continuum_fixed_source_gradient_eV_per_A": (
            result.continuum_fixed_source_gradient_eV_per_A.tolist()
        ),
        "source_response_gradient_eV_per_A": (
            result.source_response_gradient_eV_per_A.tolist()
        ),
        "total_gradient_eV_per_A": result.total_gradient_eV_per_A.tolist(),
        "model_topology": model_topology,
        "continuum_topology": continuum_topology,
        "stationarity": continuum.stationarity_audit(atoms, result.source),
        "reciprocity": result.reciprocity_audit.as_dict(),
        "replay": {
            "total_energy_eV": replay.energy.total_energy_eV,
            "source": replay.source.tolist(),
            "total_gradient_eV_per_A": replay.total_gradient_eV_per_A.tolist(),
        },
    }


def _hvp_measurement(coordinate_index, result):
    return {
        "coordinate_index": coordinate_index,
        "scalar_id": result.scalar_id,
        "profile_id": result.profile_id,
        "scalar_fingerprint_sha256": result.scalar_fingerprint_sha256,
        "model_second_order_behavior_sha256": (
            result.model_second_order_behavior_sha256
        ),
        "continuum_second_order_behavior_sha256": (
            result.continuum_second_order_behavior_sha256
        ),
        "coordinate_direction": result.coordinate_direction.tolist(),
        "source": result.source.tolist(),
        "source_gradient_cotangent": result.source_gradient_cotangent.tolist(),
        "source_position_jvp": result.source_position_jvp.tolist(),
        "intrinsic_energy_hvp_eV_per_A2": (
            result.intrinsic_energy_hvp_eV_per_A2.tolist()
        ),
        "continuum_joint_position_hvp_eV_per_A2": (
            result.continuum_joint_position_hvp_eV_per_A2.tolist()
        ),
        "continuum_source_response_pullback_eV_per_A2": (
            result.continuum_source_response_pullback_eV_per_A2.tolist()
        ),
        "contracted_source_hessian_eV_per_A2": (
            result.contracted_source_hessian_eV_per_A2.tolist()
        ),
        "total_hvp_eV_per_A2": result.total_hvp_eV_per_A2.tolist(),
        "model_standard_decomposed_energy_absolute_error_eV": (
            result.model_standard_decomposed_energy_absolute_error_eV
        ),
        "model_standard_decomposed_charge_max_absolute_error_e": (
            result.model_standard_decomposed_charge_max_absolute_error_e
        ),
        "model_standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A": (
            result.model_standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A
        ),
        "model_standard_decomposed_charge_vjp_max_absolute_error_eV_per_A": (
            result.model_standard_decomposed_charge_vjp_max_absolute_error_eV_per_A
        ),
        "model_charge_tangent_residual_e_per_A": (
            result.model_charge_tangent_residual_e_per_A
        ),
        "diagnostic_only": result.diagnostic_only,
        "tier_h_admitted": result.tier_h_admitted,
    }


def _gradient_endpoint(model, continuum, scalar, atoms):
    result = scalar.evaluate(atoms)
    model_topology, continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    return {
        "geometry_sha256": geometry_sha256(atoms),
        "positions_A": atoms.positions.tolist(),
        "total_gradient_eV_per_A": result.total_gradient_eV_per_A.tolist(),
        "model_topology": model_topology,
        "continuum_topology": continuum_topology,
    }


def _finite_difference_measurements(model, continuum, scalar, atoms):
    records = []
    for step_index, step in enumerate(AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A):
        axes = []
        for coordinate_index in range(9):
            direction = np.zeros((3, 3), dtype=float)
            direction.reshape(-1)[coordinate_index] = 1.0
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions += step * direction
            minus.positions -= step * direction
            axes.append(
                {
                    "coordinate_index": coordinate_index,
                    "plus": _gradient_endpoint(model, continuum, scalar, plus),
                    "minus": _gradient_endpoint(model, continuum, scalar, minus),
                }
            )
        records.append({"step_A": step, "axes": axes})
        print(
            f"completed dense Hessian central stencil {step_index + 1}/"
            f"{len(AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A)}",
            file=sys.stderr,
            flush=True,
        )
    return records


def _capability_record(model, continuum, model_runtime):
    capabilities = continuum.capabilities
    return {
        "model_variational_functional_admitted": (
            model.variational_functional_admitted
        ),
        "continuum_capabilities": {
            "E": capabilities.energy,
            "F": capabilities.conservative_force,
            "H": capabilities.hessian,
            "V": capabilities.variational_functional,
            "M": capabilities.molecular_dynamics,
        },
        "runtime_public_hessian": model_runtime["public_hessian"],
        "runtime_public_hvp": model_runtime["public_hvp"],
        "runtime_public_ase_calculator": model_runtime["public_ase_calculator"],
        "route2_public_ase_admitted": model_runtime["route2_public_ase_admitted"],
        "opt_admitted": False,
        "freq_admitted": False,
        "ts_admitted": False,
        "irc_admitted": False,
        "md_admitted": False,
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(REPOSITORY_ROOT)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    verify_route2_checkpoint(checkpoint)
    reference = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    started = time.perf_counter()
    model, continuum, scalar, continuum_protocol, model_runtime = (
        build_geometry_mediated_stack(
            reference,
            checkpoint,
            args.device,
            CONTINUUM_KIND,
            RUNTIME_KIND,
        )
    )
    search_record, stationary_atoms = _search_record(
        model, continuum, scalar, reference
    )
    _, center_record = _center_measurement(model, continuum, scalar, stationary_atoms)

    hessian_vector_records = []
    for coordinate_index in range(9):
        direction = np.zeros((3, 3), dtype=float)
        direction.reshape(-1)[coordinate_index] = 1.0
        result = scalar.hessian_vector_product(stationary_atoms, direction)
        hessian_vector_records.append(_hvp_measurement(coordinate_index, result))
        print(
            f"completed dense Cartesian HVP {coordinate_index + 1}/9",
            file=sys.stderr,
            flush=True,
        )
    finite_difference_records = _finite_difference_measurements(
        model, continuum, scalar, stationary_atoms
    )
    summary = summarize_aimnet2_geometry_mediated_frequency_water(
        search_record=search_record,
        center_record=center_record,
        hessian_vector_records=hessian_vector_records,
        finite_difference_records=finite_difference_records,
    )
    admission_boundary = _capability_record(model, continuum, model_runtime)
    if admission_boundary["continuum_capabilities"] != NO_CAPABILITIES or any(
        value is not False
        for key, value in admission_boundary.items()
        if key != "continuum_capabilities"
    ):
        raise RuntimeError("frequency evidence stack attempted to open a capability.")

    measured = {
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_CONTRACT_VERSION,
        "protocol": {
            "aimnet_runtime": RUNTIME_KIND,
            "continuum_kind": CONTINUUM_KIND,
            "continuum": continuum_protocol,
            "stationary_search": "three-coordinate scipy.optimize.root-hybr",
            "dense_hessian": "nine complete weak-scalar Cartesian HVP columns",
            "finite_difference_target": "total-scalar-gradient",
            "mass_weighting_order": (
                "H_mw=M^-1/2 H_cart M^-1/2 before rigid-mode projection"
            ),
            "hessian_input_unit": "eV/angstrom^2",
            "outer_charge_fixed_point": False,
            "fixed_geometry_mutual_polarization": False,
        },
        "identity": {
            "scalar_id": scalar.scalar_id,
            "profile_id": scalar.profile_id,
            "scalar_fingerprint_sha256": scalar.fingerprint_sha256(),
            "model_provider_id": model.provider_id,
            "model_configuration_sha256": model.configuration_sha256(),
            "model_provenance_sha256": model.provenance_sha256,
            "continuum_provider_id": continuum.provider_id,
            "continuum_configuration_sha256": continuum.configuration_sha256(),
            "continuum_provenance_sha256": continuum.provenance_sha256,
            "source_space_sha256": model.source_space.metadata_hash(),
            "field_space_sha256": model.field_space.metadata_hash(),
            "pairing_sha256": continuum.pairing.metadata_hash(),
            "model_runtime": model_runtime,
        },
        "admission_boundary": admission_boundary,
        "search_record": search_record,
        "center_record": center_record,
        "hessian_vector_records": hessian_vector_records,
        "finite_difference_records": finite_difference_records,
        "summary": summary,
    }

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=(
            REQUIRED_SOURCE_PATHS
            + CONTINUUM_REQUIRED_SOURCE_PATHS[CONTINUUM_KIND]
            + MODEL_RUNTIME_REQUIRED_SOURCE_PATHS[RUNTIME_KIND]
        ),
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": (
            "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
            "smooth-harmonic-stationary-water-dense-hessian-frequency"
        ),
        "status": (
            "diagnostic-frequency-gates-passed-not-admitted"
            if summary["diagnostic_gates_passed"]
            else "diagnostic-frequency-gates-failed-not-admitted"
        ),
        "claim_boundary": (
            "This one-water artifact validates a guarded internal-coordinate "
            "stationary solve, the complete dense weak-scalar Cartesian Hessian, "
            "all-column total-gradient finite differences, Hessian symmetry, "
            "three translational and three stationary rotational zero modes, and "
            "a correctly mass-weighted three-mode vibrational subspace on one "
            "fixed graph/cavity stratum. Frequencies are implementation evidence "
            "for a conductor-reference point-charge research scalar, not physical "
            "solvent predictions, chemical accuracy, global C2 regularity, or "
            "Tier H, FREQ/TS/IRC, OPT, MD, public ASE, or E/F/H/V/M admission."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": source_hashes,
        "checkpoint": checkpoint_record(
            checkpoint,
            role="aimnet2-wb97m-d3-local-sha256-bound-checkpoint",
        ),
        "runtime": runtime_record(),
        "device": args.device,
        "aimnet_runtime": RUNTIME_KIND,
        "continuum_kind": CONTINUUM_KIND,
        "dtype": model.dtype,
        **measured,
        "measurement_sha256": canonical_json_sha256(measured),
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_AIMNET2_GEOMETRY_MEDIATED_FREQUENCY="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "status": payload["status"],
                "diagnostic_gates_passed": summary["diagnostic_gates_passed"],
                "frequencies_cm1": summary["vibrational_analysis"]["frequencies_cm1"],
                "smallest_step_FD_frobenius_error_eV_per_A2": summary[
                    "finite_difference_error_norms"
                ]["frobenius_eV_per_A2"][-1],
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
