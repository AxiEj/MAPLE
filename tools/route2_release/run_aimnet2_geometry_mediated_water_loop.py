#!/usr/bin/env python3
"""Capture the disabled source-bound AIMNet2/harmonic water loop.

Both traversals evaluate every point independently.  No outer charge state,
warm start, graph feedback, or latent response is introduced.  The artifact
is diagnostic-only and cannot admit a public capability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION,
    RepositorySnapshot,
    aimnet2_geometry_mediated_water_loop_atoms,
    aimnet2_geometry_mediated_water_loop_coefficients,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    summarize_aimnet2_geometry_mediated_water_loop,
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

SCHEMA_VERSION = "route2-aimnet2-geometry-mediated-water-loop-artifact-v1"
RUNTIME_KIND = "reconstructed-python-float64"
CONTINUUM_KIND = "harmonic-point"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/geometry_mediated_panel.py",
    "maple/solvation/release/geometry_mediated_path.py",
    "maple/solvation/release/pes_panel.py",
    "maple/solvation/release/pes_validation.py",
    "tools/route2_release/run_aimnet2_geometry_mediated_water_loop.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the disabled source-bound float64 AIMNet2/smooth-harmonic "
            "bidirectional water closed loop."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _point_measurement(model, continuum, scalar, coefficient):
    atoms = aimnet2_geometry_mediated_water_loop_atoms(coefficient)
    result = scalar.evaluate(atoms)
    model_topology, continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    return {
        "coefficient": list(coefficient),
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "energy": {
            "vacuum_energy_eV": result.energy.vacuum_energy_eV,
            "continuum_energy_eV": result.energy.continuum_energy_eV,
            "total_energy_eV": result.energy.total_energy_eV,
        },
        "forces_eV_per_A": result.forces_eV_per_A.tolist(),
        "total_gradient_eV_per_A": result.total_gradient_eV_per_A.tolist(),
        "source": result.source.tolist(),
        "reaction_field": result.reaction_field.tolist(),
        "model_topology": model_topology,
        "continuum_topology": continuum_topology,
        "stationarity": continuum.stationarity_audit(atoms, result.source),
        "reciprocity": result.reciprocity_audit.as_dict(),
    }


def _traversal(model, continuum, scalar, *, reverse: bool):
    label = "reverse" if reverse else "forward"
    records = []
    coefficients = aimnet2_geometry_mediated_water_loop_coefficients(reverse=reverse)
    for index, coefficient in enumerate(coefficients):
        records.append(_point_measurement(model, continuum, scalar, coefficient))
        print(
            f"completed {label} loop point {index + 1}/{len(coefficients)}",
            file=sys.stderr,
            flush=True,
        )
    return records


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(REPOSITORY_ROOT)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    verify_route2_checkpoint(checkpoint)
    base_atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    started = time.perf_counter()
    model, continuum, scalar, continuum_protocol, model_runtime = (
        build_geometry_mediated_stack(
            base_atoms,
            checkpoint,
            args.device,
            CONTINUUM_KIND,
            RUNTIME_KIND,
        )
    )
    forward_records = _traversal(model, continuum, scalar, reverse=False)
    reverse_records = _traversal(model, continuum, scalar, reverse=True)
    summary = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=forward_records,
        reverse_records=reverse_records,
    )
    measured = {
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP_CONTRACT_VERSION,
        "protocol": {
            "aimnet_runtime": RUNTIME_KIND,
            "continuum_kind": CONTINUUM_KIND,
            "continuum": continuum_protocol,
            "independent_forward_reverse_evaluations": True,
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
        "forward_records": forward_records,
        "reverse_records": reverse_records,
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
            "smooth-harmonic-water-bidirectional-loop"
        ),
        "status": (
            "diagnostic-gates-passed-not-admitted"
            if summary["diagnostic_gates_passed"]
            else "diagnostic-gates-failed-not-admitted"
        ),
        "claim_boundary": (
            "This water-only artifact checks bidirectional closed-loop force work, "
            "same-coordinate replay, stationary harmonic solves, reciprocity, and "
            "conservative straight-segment topology/event certificates for the "
            "explicit R->q_AIMNet2(R) conductor-reference scalar. It is not the "
            "complete H/C/N/O panel, a global C1 proof, finite-dielectric solvent "
            "validation, fixed-R mutual polarization, chemical-accuracy evidence, "
            "or E/F/H/V/M, OPT, FREQ/TS/IRC, or MD admission."
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
        "ROUTE2_AIMNET2_GEOMETRY_MEDIATED_WATER_LOOP="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "status": payload["status"],
                "diagnostic_gates_passed": summary["diagnostic_gates_passed"],
                "forward_work_eV": summary["forward"]["work"]["simpson_work_eV"],
                "reverse_work_eV": summary["reverse"]["work"]["simpson_work_eV"],
                "forward_reverse_work_sum_eV": summary["forward_reverse_work_sum_eV"],
                "minimum_neighbor_cutoff_margin_A": summary[
                    "minimum_neighbor_cutoff_margin_A"
                ],
                "minimum_point_source_shell_margin_A": summary[
                    "minimum_point_source_shell_margin_A"
                ],
                "minimum_sphere_tangency_margin_A": summary[
                    "minimum_sphere_tangency_margin_A"
                ],
                "maximum_reciprocity_absolute_error_eV": summary[
                    "maximum_reciprocity_absolute_error_eV"
                ],
                "maximum_charge_fd_absolute_error_eV_per_e": summary[
                    "maximum_charge_fd_absolute_error_eV_per_e"
                ],
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
