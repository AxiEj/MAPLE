#!/usr/bin/env python3
"""Run one preregistered shard of the disabled fixed-box590 PES panel.

Each shard writes raw evidence outside the checkout.  Aggregation and any
admission decision are separate; this runner cannot enable E/F/H/V/M.
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

import numpy as np

from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    PES_PANEL_ADDITIONAL_PATHS,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_CONTRACT_VERSION,
    PES_PANEL_DIRECTION_NAMES,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    PES_PANEL_MOLECULE_COUNT,
    PES_PANEL_VARIANT_NAMES,
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_directions,
    panel_geometries,
    panel_paths,
    runtime_record,
    stretch_tangent,
    torsion_tangent,
    write_external_json_artifact,
)
from maple.solvation.release.pes_validation import summarize_directional_derivatives
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter
from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)

from fixedbox590_water_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    build_system_with_model,
    cold_warm_record,
    identity_record,
    root_context,
    state_record,
)

SCHEMA_VERSION = "route2-fixedbox590-pes-panel-shard-v1"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/pes_panel.py",
    "tools/route2_release/run_fixedbox590_pes_panel.py",
)
PANEL_ASSET_PATH = "tools/route2_release/data/fixedbox590_pes_panel_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--molecule-start", type=int, default=0)
    parser.add_argument("--molecule-stop", type=int, default=PES_PANEL_MOLECULE_COUNT)
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


class _ShardRunner:
    def __init__(
        self,
        molecule,
        model,
        primal_options: FixedPointOptions,
        adjoint_options: AdjointOptions,
    ) -> None:
        self.molecule = molecule
        self.profile, self.model, self.continuum, self.equation, self.scalar = (
            build_system_with_model(molecule.atoms, model)
        )
        self.primal_options = primal_options
        self.adjoint_options = adjoint_options

    def solve(self, atoms, label: str, initial_y=None):
        return solve_fixed_point(
            self.equation,
            atoms,
            scalar_id=self.scalar.scalar_id,
            profile_id=self.scalar.profile_id,
            scalar_binding=self.scalar,
            root_context_id=root_context(
                atoms,
                label,
                system_id=self.molecule.molecule_id,
            ),
            initial_y=initial_y,
            options=self.primal_options,
        )

    def direction_record(self, atoms, label, state, analytic_gradient, name, direction):
        analytic_gradient = np.asarray(analytic_gradient, dtype=float).reshape(
            len(atoms), 3
        )
        analytic = float(np.vdot(analytic_gradient, direction))
        samples: list[tuple[float, float, float]] = []
        displaced: list[dict[str, object]] = []
        maximum_primal = state.actual_unmixed_residual_norm
        topology_hashes = {
            self.continuum.surface_provider.build_state(atoms).topology_hash
        }
        for step in PES_PANEL_DIRECTIONAL_STEPS_A:
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions += step * direction
            minus.positions -= step * direction
            plus_state = self.solve(
                plus,
                f"{label}/{name}/{step}/plus",
                state.y_array(),
            )
            minus_state = self.solve(
                minus,
                f"{label}/{name}/{step}/minus",
                state.y_array(),
            )
            maximum_primal = max(
                maximum_primal,
                plus_state.actual_unmixed_residual_norm,
                minus_state.actual_unmixed_residual_norm,
            )
            plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)
            minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)
            # Fixed-topology identity is a surface property.  Reading it from
            # the surface provider avoids a second dense continuum build at
            # every displaced energy point.
            plus_topology = self.continuum.surface_provider.build_state(
                plus
            ).topology_hash
            minus_topology = self.continuum.surface_provider.build_state(
                minus
            ).topology_hash
            topology_hashes.update((plus_topology, minus_topology))
            samples.append((step, plus_energy, minus_energy))
            displaced.append(
                {
                    "step_A": step,
                    "plus_geometry_sha256": geometry_sha256(plus),
                    "minus_geometry_sha256": geometry_sha256(minus),
                    "plus_energy_eV": plus_energy,
                    "minus_energy_eV": minus_energy,
                    "plus_primal_residual": (plus_state.actual_unmixed_residual_norm),
                    "minus_primal_residual": (minus_state.actual_unmixed_residual_norm),
                    "plus_topology_hash": plus_topology,
                    "minus_topology_hash": minus_topology,
                }
            )
        result = summarize_directional_derivatives(analytic, samples)
        result.update(
            {
                "displaced_states": displaced,
                "maximum_primal_residual": maximum_primal,
                "topology_hashes": sorted(topology_hashes),
                "fixed_topology": len(topology_hashes) == 1,
            }
        )
        return result

    def run(self) -> list[dict[str, object]]:
        from fixedbox590_water_path import GradientEvaluation

        records: list[dict[str, object]] = []
        for variant, atoms in panel_geometries(self.molecule).items():
            cold = self.solve(atoms, f"panel/{variant}")
            warm_seed = np.full(self.equation.reduced_dimension, 1.0e-3)
            warm = self.solve(atoms, f"panel/{variant}", warm_seed)
            gradient = self.scalar.implicit_gradient(
                atoms, cold, adjoint_options=self.adjoint_options
            )
            topology = self.continuum.build_state(
                atoms, cold.source
            ).surface.topology_hash
            evaluation = GradientEvaluation(atoms.copy(), cold, gradient, topology)
            warm_energy = self.scalar.evaluate_energy_components(atoms, warm.y)
            root = cold_warm_record(
                cold,
                warm,
                self.scalar,
                atoms,
                cold_evaluation=gradient.scalar,
                warm_evaluation=warm_energy,
            )
            directions = panel_directions(atoms, self.molecule.molecule_id)
            directional = {
                name: self.direction_record(
                    atoms,
                    f"panel/{variant}",
                    evaluation.state,
                    evaluation.gradient.total_coordinate_gradient,
                    name,
                    directions[name],
                )
                for name in PES_PANEL_DIRECTION_NAMES
            }
            maximum_primal = max(
                cold.actual_unmixed_residual_norm,
                warm.actual_unmixed_residual_norm,
                *(record["maximum_primal_residual"] for record in directional.values()),
            )
            records.append(
                {
                    "molecule_id": self.molecule.molecule_id,
                    "source_record_id": self.molecule.source_record_id,
                    "chemical_formula": self.molecule.chemical_formula,
                    "scope_tags": list(self.molecule.scope_tags),
                    "variant": variant,
                    "atomic_numbers": atoms.numbers.tolist(),
                    "positions_A": atoms.positions.tolist(),
                    "geometry_sha256": geometry_sha256(atoms),
                    "cold_state": state_record(
                        cold, self.scalar, atoms, evaluation=gradient.scalar
                    ),
                    "cold_warm": root,
                    "adjoint_residual": gradient.adjoint.true_residual_norm,
                    "maximum_primal_residual": maximum_primal,
                    "forces_eV_per_A": np.asarray(gradient.forces)
                    .reshape(len(atoms), 3)
                    .tolist(),
                    "topology_hash": topology,
                    "directional_force_fd": directional,
                }
            )
            print(
                f"completed molecule={self.molecule.molecule_id} variant={variant}",
                file=sys.stderr,
                flush=True,
            )
        return records

    def path_point_record(self, point, previous_y=None):
        atoms = point.atoms
        cold = self.solve(atoms, f"path/{point.path_name}/{point.point_label}")
        warm_seed = (
            np.full(self.equation.reduced_dimension, 1.0e-3)
            if previous_y is None
            else previous_y
        )
        warm = self.solve(
            atoms, f"path/{point.path_name}/{point.point_label}", warm_seed
        )
        gradient = self.scalar.implicit_gradient(
            atoms, cold, adjoint_options=self.adjoint_options
        )
        tangent = (
            torsion_tangent(atoms)
            if point.path_name == PES_PANEL_ADDITIONAL_PATHS[0]
            else stretch_tangent(atoms)
        )
        analytic_derivative = float(
            np.vdot(
                np.asarray(gradient.total_coordinate_gradient).reshape(len(atoms), 3),
                tangent,
            )
        )
        topology = self.continuum.surface_provider.build_state(atoms).topology_hash
        local_force = self.direction_record(
            atoms,
            f"path/{point.path_name}/{point.point_label}",
            cold,
            gradient.total_coordinate_gradient,
            "coordinate-tangent",
            tangent,
        )
        warm_energy = self.scalar.evaluate_energy_components(atoms, warm.y)
        return (
            {
                "path_name": point.path_name,
                "molecule_id": point.molecule_id,
                "point_label": point.point_label,
                "coordinate_name": point.coordinate_name,
                "coordinate_value": point.coordinate_value,
                "coordinate_unit": point.coordinate_unit,
                "scope_tags": list(point.scope_tags),
                "atomic_numbers": atoms.numbers.tolist(),
                "positions_A": atoms.positions.tolist(),
                "geometry_sha256": geometry_sha256(atoms),
                "cold_state": state_record(
                    cold, self.scalar, atoms, evaluation=gradient.scalar
                ),
                "cold_warm": cold_warm_record(
                    cold,
                    warm,
                    self.scalar,
                    atoms,
                    cold_evaluation=gradient.scalar,
                    warm_evaluation=warm_energy,
                ),
                "energy_eV": gradient.scalar.total_energy,
                "forces_eV_per_A": np.asarray(gradient.forces)
                .reshape(len(atoms), 3)
                .tolist(),
                "coordinate_tangent_A_per_coordinate_unit": tangent.tolist(),
                "analytic_coordinate_derivative_eV_per_coordinate_unit": (
                    analytic_derivative
                ),
                "local_tangent_force_fd": local_force,
                "maximum_primal_residual": max(
                    cold.actual_unmixed_residual_norm,
                    warm.actual_unmixed_residual_norm,
                ),
                "adjoint_residual": gradient.adjoint.true_residual_norm,
                "topology_hash": topology,
            },
            cold.y_array(),
        )


def main() -> None:
    args = _parse_args()
    if (
        type(args.molecule_start) is not int
        or type(args.molecule_stop) is not int
        or not 0 <= args.molecule_start < args.molecule_stop <= PES_PANEL_MOLECULE_COUNT
    ):
        raise ValueError("Molecule shard must satisfy 0 <= start < stop <= 20.")
    if not math.isfinite(args.primal_tolerance) or args.primal_tolerance <= 0.0:
        raise ValueError("--primal-tolerance must be finite and positive.")
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    molecules = load_pes_panel()[args.molecule_start : args.molecule_stop]
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    path_records: list[dict[str, object]] = []
    identities: dict[str, object] | None = None
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
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        # A shard uses one exact checkpoint/model adapter. Per-molecule state
        # equations and continuum surfaces remain independently sized and
        # hashed, while checkpoint loading is not repeated 20 times.
        shared_model = build_official_mace_polar_1_m_radial_gto_adapter(
            checkpoint_path=checkpoint,
            device=args.device,
            long_range_evaluator_profile=(
                MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[40]
            ),
        )
        for molecule in molecules:
            runner = _ShardRunner(
                molecule,
                shared_model,
                primal_options,
                adjoint_options,
            )
            current_identities = identity_record(
                runner.model,
                runner.continuum,
                runner.equation,
                runner.scalar,
            )
            if identities is None:
                identities = current_identities
            elif current_identities["profile_id"] != identities["profile_id"]:
                raise RuntimeError("PES-panel profile identity changed within a shard.")
            if runner.profile.enabled or runner.profile.capabilities.enabled_tiers:
                raise RuntimeError("PES-panel diagnostic profile must remain disabled.")
            records.extend(runner.run())
            for path_name, points in panel_paths().items():
                if points[0].molecule_id != molecule.molecule_id:
                    continue
                previous_y = None
                for point in points:
                    path_record, previous_y = runner.path_point_record(
                        point, previous_y
                    )
                    path_records.append(path_record)
                    print(
                        f"completed path={path_name} point={point.point_label}",
                        file=sys.stderr,
                        flush=True,
                    )

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    # collect_loaded_repository_sources intentionally accepts only Python
    # sources. Bind the frozen JSON geometry asset separately as an exact Git
    # blob, then store both in the one source-files ledger.
    source_hashes = committed_source_hashes(
        repository, (*source_paths, PANEL_ASSET_PATH)
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-fixedbox590-pes-panel-shard",
        "status": "diagnostic-shard-success",
        "claim_boundary": (
            "One preregistered shard of the disabled 20-molecule same-scalar PES "
            "panel; not aggregate admission, not Tier E/F/H/V/M, not complete "
            "solvation free energy, and not chemical-accuracy evidence."
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
        "dtype": "float64",
        "identities": identities,
        "panel_contract": {
            "contract_version": PES_PANEL_CONTRACT_VERSION,
            "asset_sha256": PES_PANEL_ASSET_SHA256,
            "molecule_count": PES_PANEL_MOLECULE_COUNT,
            "variant_names": list(PES_PANEL_VARIANT_NAMES),
            "additional_path_contracts": list(PES_PANEL_ADDITIONAL_PATHS),
            "direction_names": list(PES_PANEL_DIRECTION_NAMES),
            "directional_steps_A": list(PES_PANEL_DIRECTIONAL_STEPS_A),
            "shard_start": args.molecule_start,
            "shard_stop": args.molecule_stop,
            "shard_molecule_ids": [item.molecule_id for item in molecules],
        },
        "measurements": records,
        "path_measurements": path_records,
        "warnings": _warning_records(captured_warnings),
        "validation_scope": {
            "aggregate_multi_molecule_pes_panel": False,
            "shard_geometry_count": len(records),
            "shard_path_geometry_count": len(path_records),
            "charged_systems_supported": False,
            "complete_nonpolar_free_energy": False,
            "chemical_accuracy": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {
            "contract": payload["panel_contract"],
            "measurements": payload["measurements"],
            "path_measurements": payload["path_measurements"],
        }
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_PES_PANEL_SHARD="
        + json.dumps(
            {
                "artifact": file_record,
                "measurement_sha256": payload["measurement_sha256"],
                "capabilities": payload["capabilities"],
                "shard_start": args.molecule_start,
                "shard_stop": args.molecule_stop,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
