#!/usr/bin/env python3
"""Capture the disabled source-bound AIMNet2/harmonic water HVP canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES,
    AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A,
    AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_CONTRACT_VERSION,
    RepositorySnapshot,
    aimnet2_geometry_mediated_hvp_directions,
    aimnet2_geometry_mediated_water_loop_atoms,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    summarize_aimnet2_geometry_mediated_hvp_water,
    write_external_json_artifact,
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

SCHEMA_VERSION = "route2-aimnet2-geometry-mediated-hvp-water-artifact-v1"
RUNTIME_KIND = "reconstructed-python-float64"
CONTINUUM_KIND = "harmonic-point"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/geometry_mediated_hessian.py",
    "maple/solvation/release/geometry_mediated_panel.py",
    "maple/solvation/release/geometry_mediated_path.py",
    "maple/solvation/release/pes_panel.py",
    "tools/route2_release/run_aimnet2_geometry_mediated_hvp.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the disabled source-bound float64 AIMNet2/smooth-harmonic "
            "complete weak-scalar water HVP canary."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _center_measurement(model, continuum, scalar, atoms):
    result = scalar.evaluate(atoms)
    model_topology, continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    return result, {
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "energy": {
            "vacuum_energy_eV": result.energy.vacuum_energy_eV,
            "continuum_energy_eV": result.energy.continuum_energy_eV,
            "total_energy_eV": result.energy.total_energy_eV,
        },
        "source": result.source.tolist(),
        "reaction_field": result.reaction_field.tolist(),
        "intrinsic_gradient_eV_per_A": (result.intrinsic_gradient_eV_per_A.tolist()),
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
    }


def _hvp_measurement(label, result):
    return {
        "label": label,
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
        "continuum_joint_source_hvp": result.continuum_joint_source_hvp.tolist(),
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


def _endpoint_measurement(
    model,
    continuum,
    scalar,
    atoms,
    *,
    source_cotangent,
):
    result = scalar.evaluate(atoms)
    model_topology, continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    zero_field = np.zeros(model.field_space.shape(len(atoms)), dtype=float)
    fixed_vjp = model.source_position_vjp(
        atoms,
        zero_field,
        source_cotangent,
    )
    return {
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "source": result.source.tolist(),
        "fixed_source_cotangent_vjp_eV_per_A": np.asarray(
            fixed_vjp, dtype=float
        ).tolist(),
        "total_gradient_eV_per_A": result.total_gradient_eV_per_A.tolist(),
        "model_topology": model_topology,
        "continuum_topology": continuum_topology,
    }


def _finite_difference_measurements(
    model,
    continuum,
    scalar,
    atoms,
    *,
    direction,
    source_cotangent,
):
    records = []
    for index, step in enumerate(AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A):
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        records.append(
            {
                "step_A": step,
                "plus": _endpoint_measurement(
                    model,
                    continuum,
                    scalar,
                    plus,
                    source_cotangent=source_cotangent,
                ),
                "minus": _endpoint_measurement(
                    model,
                    continuum,
                    scalar,
                    minus,
                    source_cotangent=source_cotangent,
                ),
            }
        )
        print(
            f"completed HVP central stencil {index + 1}/"
            f"{len(AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A)}",
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
    atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    started = time.perf_counter()
    model, continuum, scalar, continuum_protocol, model_runtime = (
        build_geometry_mediated_stack(
            atoms,
            checkpoint,
            args.device,
            CONTINUUM_KIND,
            RUNTIME_KIND,
        )
    )
    _, center_record = _center_measurement(model, continuum, scalar, atoms)
    directions = aimnet2_geometry_mediated_hvp_directions()
    direction_records = []
    hvp_results = []
    for index, (label, direction) in enumerate(
        zip(AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES, directions, strict=True)
    ):
        result = scalar.hessian_vector_product(atoms, direction)
        hvp_results.append(result)
        direction_records.append(_hvp_measurement(label, result))
        print(
            f"completed HVP direction {index + 1}/{len(directions)}: {label}",
            file=sys.stderr,
            flush=True,
        )
    finite_difference_records = _finite_difference_measurements(
        model,
        continuum,
        scalar,
        atoms,
        direction=directions[0],
        source_cotangent=hvp_results[0].source_gradient_cotangent,
    )
    summary = summarize_aimnet2_geometry_mediated_hvp_water(
        center_record=center_record,
        direction_records=direction_records,
        finite_difference_records=finite_difference_records,
    )
    admission_boundary = _capability_record(model, continuum, model_runtime)
    if admission_boundary["continuum_capabilities"] != NO_CAPABILITIES or any(
        value is not False
        for key, value in admission_boundary.items()
        if key != "continuum_capabilities"
    ):
        raise RuntimeError("HVP evidence stack attempted to open a public capability.")

    measured = {
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_CONTRACT_VERSION,
        "protocol": {
            "aimnet_runtime": RUNTIME_KIND,
            "continuum_kind": CONTINUUM_KIND,
            "continuum": continuum_protocol,
            "hvp_formula": (
                "H_E h + (G_RR h + G_Rc J_c h) + "
                "J_c^T(G_cR h + G_cc J_c h) + D_R[J_c^T v][h]"
            ),
            "finite_difference_target": "total-scalar-gradient",
            "fixed_source_cotangent_in_contracted_charge_hessian_fd": True,
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
        "center_record": center_record,
        "direction_records": direction_records,
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
            "smooth-harmonic-water-complete-hvp"
        ),
        "status": (
            "diagnostic-hvp-gates-passed-not-admitted"
            if summary["diagnostic_gates_passed"]
            else "diagnostic-hvp-gates-failed-not-admitted"
        ),
        "claim_boundary": (
            "This one-water artifact validates a complete weak-scalar HVP ledger, "
            "central finite differences of the charge JVP, fixed-cotangent charge "
            "Hessian, and total-gradient HVP, bilinear symmetry, all three rigid "
            "translations, stationarity, reciprocity, and local event guards on "
            "one fixed graph/cavity stratum. It is not a global C2 proof, a full "
            "H/C/N/O domain panel, finite-dielectric or nonpolar validation, a "
            "stationary-point frequency calculation, fixed-R mutual polarization, "
            "chemical-accuracy evidence, or Tier H, FREQ/TS/IRC, MD, public ASE, "
            "or any E/F/H/V/M admission."
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
        "ROUTE2_AIMNET2_GEOMETRY_MEDIATED_HVP="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "status": payload["status"],
                "diagnostic_gates_passed": summary["diagnostic_gates_passed"],
                "smallest_step_total_hvp_error_eV_per_A2": summary[
                    "finite_difference_error_norms"
                ]["total_hvp_eV_per_A2"][-1],
                "bilinear_symmetry_error_eV_per_A2": summary["bilinear_symmetry"][
                    "absolute_error_eV_per_A2"
                ],
                "translation_hvp_norms_eV_per_A2": summary["translation_zero_modes"][
                    "total_hvp_norms_eV_per_A2"
                ],
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
