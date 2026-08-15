#!/usr/bin/env python3
"""Capture one disabled AIMNet2 geometry-mediated H/C/N/O PES shard.

The runner is intentionally restricted to the source-bound reconstructed
float64 AIMNet2 runtime and smooth harmonic point-charge conductor reference.
Each invocation evaluates exactly one frozen molecule across three geometry
variants, three internal directions, and three central-difference steps.  It
writes evidence outside the checkout and cannot admit any public capability.
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
    AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS,
    AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION,
    AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    RepositorySnapshot,
    aimnet2_geometry_mediated_pes_molecule,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    panel_directions,
    panel_geometries,
    runtime_record,
    summarize_aimnet2_geometry_mediated_pes_shard,
    write_external_json_artifact,
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

RUNTIME_KIND = "reconstructed-python-float64"
CONTINUUM_KIND = "harmonic-point"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/geometry_mediated_panel.py",
    "maple/solvation/release/pes_panel.py",
    "tools/route2_release/run_aimnet2_geometry_mediated_pes_panel.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one disabled source-bound float64 AIMNet2/smooth-harmonic "
            "PES-panel shard."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--molecule-index",
        type=int,
        required=True,
        choices=range(len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _center_record(result) -> dict[str, object]:
    return {
        "vacuum_energy_eV": result.energy.vacuum_energy_eV,
        "continuum_energy_eV": result.energy.continuum_energy_eV,
        "total_energy_eV": result.energy.total_energy_eV,
        "source": result.source.tolist(),
        "reaction_field": result.reaction_field.tolist(),
        "forces_eV_per_A": result.forces_eV_per_A.tolist(),
        "total_gradient_eV_per_A": result.total_gradient_eV_per_A.tolist(),
        "gradient_components_eV_per_A": {
            "intrinsic": result.intrinsic_gradient_eV_per_A.tolist(),
            "continuum_fixed_source": (
                result.continuum_fixed_source_gradient_eV_per_A.tolist()
            ),
            "source_response": result.source_response_gradient_eV_per_A.tolist(),
        },
    }


def _geometry_measurement(model, continuum, scalar, molecule, variant, atoms):
    first = scalar.evaluate(atoms)
    second = scalar.evaluate(atoms)
    model_topology, continuum_topology = geometry_mediated_topologies(
        model, continuum, atoms
    )
    directions: dict[str, object] = {}
    for direction_name, direction in panel_directions(
        atoms, molecule.molecule_id
    ).items():
        samples: list[dict[str, object]] = []
        for step in PES_PANEL_DIRECTIONAL_STEPS_A:
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions += step * direction
            minus.positions -= step * direction
            plus_model_topology, plus_continuum_topology = geometry_mediated_topologies(
                model, continuum, plus
            )
            minus_model_topology, minus_continuum_topology = (
                geometry_mediated_topologies(model, continuum, minus)
            )
            samples.append(
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
        directions[direction_name] = {
            "direction": direction.tolist(),
            "samples": samples,
        }
    return {
        "molecule_id": molecule.molecule_id,
        "source_record_id": molecule.source_record_id,
        "chemical_formula": molecule.chemical_formula,
        "scope_tags": list(molecule.scope_tags),
        "variant": variant,
        "geometry_sha256": geometry_sha256(atoms),
        "atomic_numbers": atoms.numbers.tolist(),
        "positions_A": atoms.positions.tolist(),
        "charge": 0,
        "multiplicity": 1,
        "center": _center_record(first),
        "center_model_topology": model_topology,
        "center_continuum_topology": continuum_topology,
        "deterministic_replay": deterministic_replay(first, second),
        "reciprocity_metric_charge_gauge": first.reciprocity_audit.as_dict(),
        "stationarity": continuum.stationarity_audit(atoms, first.source),
        "directions": directions,
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(REPOSITORY_ROOT)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    verify_route2_checkpoint(checkpoint)
    molecule = aimnet2_geometry_mediated_pes_molecule(args.molecule_index)
    geometries = panel_geometries(molecule)
    started = time.perf_counter()
    model, continuum, scalar, continuum_protocol, model_runtime = (
        build_geometry_mediated_stack(
            geometries["reference"],
            checkpoint,
            args.device,
            CONTINUUM_KIND,
            RUNTIME_KIND,
        )
    )
    records = [
        _geometry_measurement(
            model,
            continuum,
            scalar,
            molecule,
            variant,
            atoms,
        )
        for variant, atoms in geometries.items()
    ]
    summary = summarize_aimnet2_geometry_mediated_pes_shard(
        molecule_index=args.molecule_index,
        records=records,
    )
    measured = {
        "contract_version": AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION,
        "panel_asset_sha256": PES_PANEL_ASSET_SHA256,
        "protocol": {
            "aimnet_runtime": RUNTIME_KIND,
            "continuum_kind": CONTINUUM_KIND,
            "directional_steps_A": list(PES_PANEL_DIRECTIONAL_STEPS_A),
            "continuum": continuum_protocol,
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
        "records": records,
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
        "schema_version": AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": (
            "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
            "smooth-harmonic-pes-shard"
        ),
        "status": (
            "diagnostic-gates-passed-not-admitted"
            if summary["diagnostic_gates_passed"]
            else "diagnostic-gates-failed-not-admitted"
        ),
        "claim_boundary": (
            "This single-molecule H/C/N/O shard recomputes 3 geometry variants "
            "x 3 internal directions x 3 central-difference steps for the "
            "explicit R->q_AIMNet2(R) smooth-harmonic conductor-reference scalar. "
            "It is not the complete 17-molecule panel, a finite-dielectric solvent "
            "model, fixed-R mutual polarization, chemical-accuracy evidence, a "
            "global C1 proof, or E/F/H/V/M, OPT, FREQ/TS/IRC, or MD admission."
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
        "ROUTE2_AIMNET2_GEOMETRY_MEDIATED_PES_SHARD="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "status": payload["status"],
                "molecule_index": args.molecule_index,
                "molecule_id": molecule.molecule_id,
                "diagnostic_gates_passed": summary["diagnostic_gates_passed"],
                "gates": summary["gates"],
                "maximum_directional_absolute_error_eV_per_A": summary[
                    "maximum_directional_absolute_error_eV_per_A"
                ],
                "minimum_neighbor_cutoff_margin_A": summary[
                    "minimum_neighbor_cutoff_margin_A"
                ],
                "minimum_continuum_event_margin_A": summary[
                    "minimum_continuum_event_margin_A"
                ],
                "minimum_sphere_tangency_margin_A": summary[
                    "minimum_sphere_tangency_margin_A"
                ],
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
