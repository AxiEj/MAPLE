#!/usr/bin/env python3
"""Capture disabled fixed-box-size convergence evidence on one water geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
import warnings

import numpy as np

from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.release import (
    BOX_LENGTHS_A,
    MAXIMUM_ADJOINT_RESIDUAL,
    MAXIMUM_PRIMAL_RESIDUAL,
    RepositorySnapshot,
    TAIL_CONTINUUM_ENERGY_TOLERANCE_EV,
    TAIL_FORCE_MAX_TOLERANCE_EV_PER_A,
    TAIL_FORCE_RMS_TOLERANCE_EV_PER_A,
    TAIL_SOURCE_RELATIVE_TOLERANCE,
    TAIL_TOTAL_ENERGY_TOLERANCE_EV,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    summarize_box_convergence,
    write_external_json_artifact,
)

from fixedbox590_water_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    build_system,
    identity_record,
    root_context,
    water,
)

SCHEMA_VERSION = "route2-fixedbox590-water-box-convergence-diagnostic-v1"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/box_convergence.py",
    "tools/route2_release/run_fixedbox590_water_box_convergence.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--primal-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def _warning_records(captured: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {
            "category": item.category.__name__,
            "message": str(item.message),
            "filename": item.filename,
            "lineno": str(item.lineno),
        }
        for item in captured
    ]


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    geometry = water()
    started = time.perf_counter()
    records: list[dict[str, object]] = []

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        for box_length in BOX_LENGTHS_A:
            profile, model, continuum, equation, scalar = build_system(
                geometry, checkpoint, args.device, box_length=box_length
            )
            primal_options = FixedPointOptions(
                tolerance=args.primal_tolerance,
                max_iterations=args.max_iterations,
                damping=0.7,
                history=6,
            )
            state = solve_fixed_point(
                equation,
                geometry,
                scalar_id=scalar.scalar_id,
                profile_id=scalar.profile_id,
                scalar_binding=scalar,
                root_context_id=root_context(
                    geometry, "box-convergence", box_length=box_length
                ),
                options=primal_options,
            )
            gradient = scalar.implicit_gradient(
                geometry,
                state,
                adjoint_options=AdjointOptions(
                    relative_tolerance=1.0e-11,
                    absolute_tolerance=1.0e-13,
                    max_iterations=500,
                ),
            )
            continuum_state = continuum.build_state(geometry, state.source)
            identities = identity_record(model, continuum, equation, scalar)
            if profile.enabled or profile.capabilities.enabled_tiers:
                raise RuntimeError("Box-convergence profile must remain disabled.")
            records.append(
                {
                    "box_length_A": box_length,
                    "profile_id": scalar.profile_id,
                    "scalar_id": scalar.scalar_id,
                    "state_equation_id": equation.state_equation_id,
                    "model_profile_id": model.model_profile_id,
                    "model_provider_id": model.provider_id,
                    "model_configuration_sha256": model.configuration_sha256(),
                    "model_provenance_sha256": model.provenance_sha256,
                    "equation_sha256": equation.fingerprint_sha256(),
                    "scalar_sha256": scalar.fingerprint_sha256(),
                    "identities": identities,
                    "root_hash": state.root_hash,
                    "primal_residual": state.actual_unmixed_residual_norm,
                    "adjoint_residual": gradient.adjoint.true_residual_norm,
                    "iterations": len(state.iterations) - 1,
                    "vacuum_energy_eV": gradient.scalar.vacuum_energy,
                    "continuum_energy_eV": gradient.scalar.continuum_energy,
                    "total_energy_eV": gradient.scalar.total_energy,
                    "forces_eV_per_A": np.asarray(gradient.forces)
                    .reshape(len(geometry), 3)
                    .tolist(),
                    "source": state.source_array().tolist(),
                    "field": state.field_array().tolist(),
                    "surface_topology_hash": (continuum_state.surface.topology_hash),
                    "surface_state_hash": continuum_state.surface.state_hash,
                    "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
                }
            )
            print(
                f"completed fixed-box convergence size={box_length} A",
                file=sys.stderr,
                flush=True,
            )
    summary = summarize_box_convergence(records)
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-fixed-box-convergence-diagnostic",
        "status": "diagnostic-success",
        "claim_boundary": (
            "One equilibrium water fixed-box operator convergence diagnostic; "
            "not Tier E/F/H/V/M, not a PES panel, not complete solvation free "
            "energy, and not chemical-accuracy evidence."
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
        "device": args.device,
        "dtype": records[0]["identities"]["model_provenance"]["dtype"],
        "box_convergence_contract": {
            "box_lengths_A": list(BOX_LENGTHS_A),
            "reference_box_length_A": BOX_LENGTHS_A[-1],
            "tail_pair_A": list(BOX_LENGTHS_A[-2:]),
            "thresholds": {
                "total_energy_abs_eV": TAIL_TOTAL_ENERGY_TOLERANCE_EV,
                "continuum_energy_abs_eV": TAIL_CONTINUUM_ENERGY_TOLERANCE_EV,
                "force_rms_eV_per_A": TAIL_FORCE_RMS_TOLERANCE_EV_PER_A,
                "force_max_eV_per_A": TAIL_FORCE_MAX_TOLERANCE_EV_PER_A,
                "source_relative": TAIL_SOURCE_RELATIVE_TOLERANCE,
                "primal_residual": MAXIMUM_PRIMAL_RESIDUAL,
                "adjoint_residual": MAXIMUM_ADJOINT_RESIDUAL,
            },
        },
        "measurements": summary,
        "warnings": _warning_records(captured_warnings),
        "validation_scope": {
            "single_molecule": True,
            "single_geometry": True,
            "box_operator_convergence": True,
            "multi_geometry_box_convergence": False,
            "multi_molecule_pes_panel": False,
            "complete_nonpolar_free_energy": False,
            "chemical_accuracy": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "contract": payload["box_convergence_contract"],
            "measurements": payload["measurements"],
        }
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_WATER_BOX_CONVERGENCE="
        + json.dumps(
            {
                "artifact": file_record,
                "measurement_sha256": payload["measurement_sha256"],
                "capabilities": payload["capabilities"],
                "gates": summary["gates"],
                "all_gates_passed": summary["all_gates_passed"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
