#!/usr/bin/env python3
"""Capture multi-geometry and closed-loop evidence for fixed-box590 water.

The selected profile remains disabled.  This command writes raw evidence only;
it cannot publish a Route-2 result or alter E/F/H/V/M admission.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys
import time
import warnings

from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.release import (
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    closed_rectangular_loop,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    water_vibrational_directions,
    write_external_json_artifact,
)

from fixedbox590_water_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    build_system,
    identity_record,
    water,
)
from fixedbox590_water_path import WaterPathDiagnostic

SCHEMA_VERSION = "route2-fixedbox40-cpcm590-water-path-diagnostic-v1"
DEFAULT_DIRECTIONAL_STEPS_A = (4.0e-4, 2.0e-4, 1.0e-4)
DEFAULT_LOOP_SUBDIVISIONS = 4
PANEL_COEFFICIENTS_A = {
    "equilibrium": {},
    "symmetric-compressed": {"symmetric_stretch": -0.08},
    "symmetric-stretched": {"symmetric_stretch": 0.08},
    "asymmetric-distorted": {"asymmetric_stretch": 0.08},
    "bend-open": {"bend": -0.08},
    "bend-closed": {"bend": 0.08},
    "combined-distorted": {
        "symmetric_stretch": 0.05,
        "asymmetric_stretch": -0.04,
        "bend": 0.05,
    },
}
LOOP_AMPLITUDES_A = (0.05, 0.05)
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "tools/route2_release/fixedbox590_water_path.py",
    "tools/route2_release/run_fixedbox590_water_path_diagnostic.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--directional-step", action="append", type=float, default=[])
    parser.add_argument(
        "--loop-subdivisions", type=int, default=DEFAULT_LOOP_SUBDIVISIONS
    )
    parser.add_argument("--primal-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def _steps(values: list[float]) -> tuple[float, ...]:
    result = tuple(values) if values else DEFAULT_DIRECTIONAL_STEPS_A
    if (
        not result
        or len(set(result)) != len(result)
        or any(not math.isfinite(value) or value <= 0.0 for value in result)
    ):
        raise ValueError("Directional steps must be unique finite positive values.")
    return result


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
    steps = _steps(args.directional_step)
    coefficients = closed_rectangular_loop(subdivisions_per_edge=args.loop_subdivisions)
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        base_atoms = water()
        directions = water_vibrational_directions(base_atoms.positions)
        profile, model, continuum, equation, scalar = build_system(
            base_atoms, checkpoint, args.device
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

        diagnostic = WaterPathDiagnostic(
            base_atoms=base_atoms,
            directions=directions,
            continuum=continuum,
            equation=equation,
            scalar=scalar,
            primal_options=primal_options,
            adjoint_options=adjoint_options,
        )
        panel, directional_all_pass, panel_roots_pass, equilibrium = (
            diagnostic.panel_record(PANEL_COEFFICIENTS_A, steps)
        )
        loop_records, loop_all_pass = diagnostic.loop_record(
            coefficients,
            LOOP_AMPLITUDES_A,
            args.loop_subdivisions,
            equilibrium.state.y_array(),
        )
        cold_warm_all_pass = panel_roots_pass and bool(
            loop_records["gates"]["all_loop_cold_warm_roots"]
        )
        tracker = diagnostic.tracker

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(repository, source_paths)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-water-path-diagnostic",
        "status": "diagnostic-success",
        "claim_boundary": (
            "Multi-geometry water path evidence for a disabled fixed-box40/CPCM590 "
            "candidate; not Tier E/F/H/V/M, not a multi-molecule PES panel, not "
            "complete solvation free energy, and not chemical-accuracy evidence."
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
        "dtype": model.dtype,
        "identities": identity_record(model, continuum, equation, scalar),
        "panel_contract": {
            "geometry_coefficients_A": PANEL_COEFFICIENTS_A,
            "directional_steps_A": steps,
            "direction_names": list(directions),
            "loop_amplitudes_A": list(LOOP_AMPLITUDES_A),
            "loop_subdivisions_per_edge": args.loop_subdivisions,
        },
        "measurements": {
            "panel": panel,
            "closed_loop": loop_records,
            "topology_hashes": sorted(tracker.topology_hashes),
            "maximum_primal_residual": tracker.maximum_primal_residual,
            "maximum_adjoint_residual": tracker.maximum_adjoint_residual,
        },
        "gates": {
            "all_panel_directional_force_fd": directional_all_pass,
            "all_panel_cold_warm_roots": cold_warm_all_pass,
            "fixed_topology": len(tracker.topology_hashes) == 1,
            **loop_records["gates"],
            "loop_aggregate": loop_all_pass,
        },
        "warnings": _warning_records(captured_warnings),
        "validation_scope": {
            "single_molecule": True,
            "multiple_geometries": True,
            "distorted_geometries": True,
            "bidirectional_cold_warm_closed_loop": True,
            "multi_molecule_pes_panel": False,
            "box_convergence": False,
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
            "panel_contract": payload["panel_contract"],
            "measurements": payload["measurements"],
            "gates": payload["gates"],
        }
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_WATER_PATH_DIAGNOSTIC="
        + json.dumps(
            {
                "artifact": file_record,
                "measurement_sha256": payload["measurement_sha256"],
                "profile_id": scalar.profile_id,
                "capabilities": payload["capabilities"],
                "gates": payload["gates"],
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
