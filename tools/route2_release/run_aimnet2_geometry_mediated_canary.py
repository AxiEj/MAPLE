#!/usr/bin/env python3
"""Capture a disabled AIMNet2 geometry-mediated continuum diagnostic.

The runner supports the finite-grid pyddx ddPCM branch, the structurally
SO(3)-controlled smooth harmonic point-charge conductor branch, and its
finite-dielectric double-layer ddPCM sibling.  It also keeps
the historical float32 TorchScript runtime as a negative-control arm beside a
source-bound float64 reconstruction of the same checkpoint.  It records
positive and negative gate evidence without admitting E/F/H/V/M, OPT,
FREQ/TS/IRC, or MD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
from ase import Atoms

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    write_external_json_artifact,
)
from maple.solvation.release.geometry_mediated import (
    GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION,
    GEOMETRY_MEDIATED_COORDINATE_STEPS_A,
    geometry_mediated_admission_decision,
    geometry_mediated_coordinate_direction,
    geometry_mediated_rotations,
    summarize_geometry_mediated_cartesian_audit,
    summarize_geometry_mediated_directional_audit,
    summarize_geometry_mediated_rotation_audit,
)
from maple.solvation.release.geometry_mediated_adaptive import (
    capture_geometry_mediated_adaptive_directional_samples,
    summarize_geometry_mediated_adaptive_directional_audit,
)
from aimnet2_geometry_mediated_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    CONTINUUM_REQUIRED_SOURCE_PATHS,
    MODEL_RUNTIME_REQUIRED_SOURCE_PATHS,
    NO_CAPABILITIES,
    build_geometry_mediated_stack,
    deterministic_replay,
    geometry_mediated_topologies,
    verify_route2_checkpoint,
)

SCHEMA_VERSION = "route2-aimnet2-geometry-mediated-real-stack-canary-v6"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/geometry_mediated_adaptive.py",
    "tools/route2_release/run_aimnet2_geometry_mediated_canary.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a disabled AIMNet2 geometry-mediated continuum canary; "
            "this never admits Route-2 capabilities."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--aimnet-runtime",
        choices=tuple(MODEL_RUNTIME_REQUIRED_SOURCE_PATHS),
        default="legacy-jit-float32",
        help=(
            "Keep the historical hard-coded-float32 TorchScript runtime as a "
            "negative control, or use the source-bound official-Python float64 "
            "reconstruction (CPU only)."
        ),
    )
    parser.add_argument(
        "--continuum",
        choices=("ddpcm", "harmonic-point", "harmonic-ddpcm"),
        default="ddpcm",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2399872, 0.927297, 0.0],
            ]
        ),
        info={"charge": 0, "mult": 1},
    )


def _cartesian_samples(model, continuum, scalar, atoms: Atoms):
    """Measure every Cartesian central-difference component on frozen steps."""

    samples: list[dict[str, object]] = []
    for step in GEOMETRY_MEDIATED_COORDINATE_STEPS_A:
        components: list[dict[str, object]] = []
        for atom in range(len(atoms)):
            for axis in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                plus.positions[atom, axis] += step
                minus.positions[atom, axis] -= step
                plus_model_topology, plus_continuum_topology = (
                    geometry_mediated_topologies(model, continuum, plus)
                )
                minus_model_topology, minus_continuum_topology = (
                    geometry_mediated_topologies(model, continuum, minus)
                )
                components.append(
                    {
                        "atom": atom,
                        "axis": axis,
                        "plus_energy_eV": scalar.evaluate_energy(plus),
                        "minus_energy_eV": scalar.evaluate_energy(minus),
                        "plus_model_topology": plus_model_topology,
                        "minus_model_topology": minus_model_topology,
                        "plus_continuum_topology": plus_continuum_topology,
                        "minus_continuum_topology": minus_continuum_topology,
                    }
                )
        samples.append({"step_A": step, "components": components})
    return samples


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(REPOSITORY_ROOT)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    verify_route2_checkpoint(checkpoint)
    started = time.perf_counter()
    atoms = _water()
    model, continuum, scalar, continuum_protocol, model_runtime = (
        build_geometry_mediated_stack(
            atoms,
            checkpoint,
            args.device,
            args.continuum,
            args.aimnet_runtime,
        )
    )

    first = scalar.evaluate(atoms)
    second = scalar.evaluate(atoms)
    replay = deterministic_replay(first, second)
    center_model_topology, center_continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    direction = geometry_mediated_coordinate_direction(len(atoms))

    def adaptive_sample(displacement_A: float) -> dict[str, object]:
        displaced = atoms.copy()
        displaced.positions = atoms.positions + displacement_A * direction
        model_topology, continuum_topology = geometry_mediated_topologies(
            model, continuum, displaced
        )
        return {
            "energy_eV": scalar.evaluate_energy(displaced),
            "positions_A": displaced.positions.tolist(),
            "model_topology": model_topology,
            "continuum_topology": continuum_topology,
        }

    adaptive_samples = capture_geometry_mediated_adaptive_directional_samples(
        adaptive_sample
    )
    adaptive_directional = summarize_geometry_mediated_adaptive_directional_audit(
        analytic_gradient_eV_per_A=first.total_gradient_eV_per_A,
        direction=direction,
        center_positions_A=atoms.positions,
        center_model_topology=center_model_topology,
        center_continuum_topology=center_continuum_topology,
        adaptive_record=adaptive_samples,
        reciprocity_audit=first.reciprocity_audit.as_dict(),
    )
    directional_samples: list[dict[str, object]] = []
    for step in GEOMETRY_MEDIATED_COORDINATE_STEPS_A:
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions += step * direction
        minus.positions -= step * direction
        plus_model_topology, plus_continuum_topology = geometry_mediated_topologies(
            model, continuum, plus
        )
        minus_model_topology, minus_continuum_topology = geometry_mediated_topologies(
            model, continuum, minus
        )
        directional_samples.append(
            {
                "step_A": step,
                "plus_energy_eV": scalar.evaluate_energy(plus),
                "minus_energy_eV": scalar.evaluate_energy(minus),
                "plus_model_topology": plus_model_topology,
                "minus_model_topology": minus_model_topology,
                "plus_continuum_topology": plus_continuum_topology,
                "minus_continuum_topology": minus_continuum_topology,
            }
        )
    directional = summarize_geometry_mediated_directional_audit(
        analytic_gradient_eV_per_A=first.total_gradient_eV_per_A,
        direction=direction,
        center_model_topology=center_model_topology,
        center_continuum_topology=center_continuum_topology,
        samples=directional_samples,
        reciprocity_audit=first.reciprocity_audit.as_dict(),
    )
    cartesian = summarize_geometry_mediated_cartesian_audit(
        analytic_gradient_eV_per_A=first.total_gradient_eV_per_A,
        center_model_topology=center_model_topology,
        center_continuum_topology=center_continuum_topology,
        samples=_cartesian_samples(model, continuum, scalar, atoms),
        reciprocity_audit=first.reciprocity_audit.as_dict(),
    )

    rotation_records: list[dict[str, object]] = []
    for rotation in geometry_mediated_rotations():
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        result = scalar.evaluate(rotated)
        model_topology, continuum_topology = geometry_mediated_topologies(
            model, continuum, rotated
        )
        rotation_records.append(
            {
                "rotation_matrix": rotation.tolist(),
                "energy_eV": result.energy.total_energy_eV,
                "forces_eV_per_A": result.forces_eV_per_A.tolist(),
                "source": result.source.tolist(),
                "model_topology": model_topology,
                "continuum_topology": continuum_topology,
            }
        )
    rotation = summarize_geometry_mediated_rotation_audit(
        positions_A=atoms.positions,
        base_energy_eV=first.energy.total_energy_eV,
        base_forces_eV_per_A=first.forces_eV_per_A,
        base_source=first.source,
        base_model_topology=center_model_topology,
        base_continuum_topology=center_continuum_topology,
        rotation_records=rotation_records,
    )
    stationarity = (
        continuum.stationarity_audit(atoms, first.source)
        if args.continuum in {"harmonic-point", "harmonic-ddpcm"}
        else None
    )
    post_solve_residual_available = bool(
        stationarity is not None and stationarity["gate_passed"] is True
    )
    decision = geometry_mediated_admission_decision(
        deterministic_replay_passed=bool(replay["gate_passed"]),
        directional_audit=directional,
        cartesian_audit=cartesian,
        rotation_audit=rotation,
        # pyddx 0.8.0 reports only the requested tolerance.  The harmonic
        # branch records its actual dense stationary residual.
        post_solve_residual_available=post_solve_residual_available,
    )
    decision["adaptive_directional_gate_passed"] = bool(
        adaptive_directional["gate_passed"]
    )
    decision["local_diagnostic_gates_passed"] = bool(
        decision["local_diagnostic_gates_passed"]
        and decision["adaptive_directional_gate_passed"]
    )

    measured = {
        "protocol": {
            "audit_schema_version": GEOMETRY_MEDIATED_AUDIT_SCHEMA_VERSION,
            "coordinate_steps_A": list(GEOMETRY_MEDIATED_COORDINATE_STEPS_A),
            "coordinate_direction": direction.tolist(),
            "cartesian_component_count": int(first.total_gradient_eV_per_A.size),
            "rotation_matrices": [
                rotation_matrix.tolist()
                for rotation_matrix in geometry_mediated_rotations()
            ],
            "continuum": continuum_protocol,
        },
        "geometry": {
            "formula": atoms.get_chemical_formula(),
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "charge": 0,
            "multiplicity": 1,
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
            "center_model_topology": center_model_topology,
            "center_continuum_topology": center_continuum_topology,
            "model_runtime": model_runtime,
        },
        "center": {
            "vacuum_energy_eV": first.energy.vacuum_energy_eV,
            "continuum_energy_eV": first.energy.continuum_energy_eV,
            "total_energy_eV": first.energy.total_energy_eV,
            "source": first.source.tolist(),
            "reaction_field": first.reaction_field.tolist(),
            "forces_eV_per_A": first.forces_eV_per_A.tolist(),
            "gradient_components_eV_per_A": {
                "intrinsic": first.intrinsic_gradient_eV_per_A.tolist(),
                "continuum_fixed_source": (
                    first.continuum_fixed_source_gradient_eV_per_A.tolist()
                ),
                "source_response": first.source_response_gradient_eV_per_A.tolist(),
                "total": first.total_gradient_eV_per_A.tolist(),
            },
        },
        "deterministic_replay": replay,
        "reciprocity_metric_charge_gauge": first.reciprocity_audit.as_dict(),
        "stationarity": stationarity,
        "coordinate_directional": {
            "raw": directional_samples,
            "summary": directional,
        },
        "coordinate_directional_adaptive": {
            "raw": adaptive_samples,
            "summary": adaptive_directional,
        },
        "coordinate_cartesian": cartesian,
        "rigid_rotation": {
            "raw": rotation_records,
            "summary": rotation,
        },
        "decision": decision,
    }

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root,
        required_paths=(
            REQUIRED_SOURCE_PATHS
            + CONTINUUM_REQUIRED_SOURCE_PATHS[args.continuum]
            + MODEL_RUNTIME_REQUIRED_SOURCE_PATHS[args.aimnet_runtime]
        ),
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": (
            "disabled-aimnet2-"
            f"{args.aimnet_runtime}-geometry-mediated-{args.continuum}-"
            "real-stack-canary"
        ),
        "status": (
            "diagnostic-gates-passed-not-admitted"
            if decision["local_diagnostic_gates_passed"]
            else "diagnostic-gates-failed-not-admitted"
        ),
        "claim_boundary": (
            "This one-water artifact audits the explicit geometry map "
            "R->q_AIMNet2(R), the selected continuum half-coupling under the "
            "registered metric, "
            "charge-gauge response, one pinned adaptive directional trace, one "
            "three-step coordinate direction, a three-step full Cartesian "
            "panel, hard neighbor/continuum strata and event clearances, and "
            "three rigid rotations. The harmonic-point arm is a conductor "
            "reference; harmonic-ddpcm adds the finite-dielectric double-layer "
            "PCM equation rather than a uniform COSMO scale. Neither is fixed-R "
            "mutual polarization, chemical-accuracy evidence, a global C1 proof, "
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
        "aimnet_runtime": args.aimnet_runtime,
        "continuum_kind": args.continuum,
        "dtype": model.dtype,
        **measured,
        "measurement_sha256": canonical_json_sha256(measured),
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_AIMNET2_GEOMETRY_MEDIATED_CANARY="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "status": payload["status"],
                "decision": decision,
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
