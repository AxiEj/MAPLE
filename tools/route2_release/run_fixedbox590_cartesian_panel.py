#!/usr/bin/env python3
"""Run one disabled shard of the preregistered full-Cartesian PES panel."""

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

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter
from maple.solvation.release import (
    PES_CARTESIAN_PANEL_CONTRACT_VERSION,
    PES_CARTESIAN_PANEL_STEPS_A,
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_MOLECULE_COUNT,
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_geometries,
    runtime_record,
    summarize_cartesian_force_differences,
    write_external_json_artifact,
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
from run_fixedbox590_pes_panel import (
    _configure_numerical_determinism,
    _warning_records,
)

SCHEMA_VERSION = "route2-fixedbox590-cartesian-panel-shard-v1"
PANEL_ASSET_PATH = "tools/route2_release/data/fixedbox590_pes_panel_v1.json"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/pes_panel.py",
    "tools/route2_release/run_fixedbox590_pes_panel.py",
    "tools/route2_release/run_fixedbox590_cartesian_panel.py",
)


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


class _CartesianShardRunner:
    def __init__(self, molecule, model, primal_options, adjoint_options) -> None:
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
                atoms, label, system_id=self.molecule.molecule_id
            ),
            initial_y=initial_y,
            options=self.primal_options,
        )

    def run(self) -> dict[str, object]:
        atoms = panel_geometries(self.molecule)[PES_CARTESIAN_PANEL_VARIANT]
        cold = self.solve(atoms, "cartesian-panel/reference")
        warm = self.solve(
            atoms,
            "cartesian-panel/reference",
            np.full(self.equation.reduced_dimension, 1.0e-3),
        )
        gradient = self.scalar.implicit_gradient(
            atoms, cold, adjoint_options=self.adjoint_options
        )
        analytic = np.asarray(gradient.total_coordinate_gradient, dtype=float).reshape(
            len(atoms), 3
        )
        base_topology = self.continuum.surface_provider.build_state(
            atoms
        ).topology_hash
        topology_hashes = {base_topology}
        maximum_primal = max(
            cold.actual_unmixed_residual_norm,
            warm.actual_unmixed_residual_norm,
        )
        raw_steps: list[dict[str, object]] = []
        summary_samples: list[tuple[float, np.ndarray]] = []
        for step in PES_CARTESIAN_PANEL_STEPS_A:
            finite_difference = np.zeros_like(analytic)
            components: list[dict[str, object]] = []
            for atom in range(len(atoms)):
                for axis in range(3):
                    plus = atoms.copy()
                    minus = atoms.copy()
                    plus.positions[atom, axis] += step
                    minus.positions[atom, axis] -= step
                    plus_state = self.solve(
                        plus,
                        f"cartesian/{step}/atom-{atom}/axis-{axis}/plus",
                        cold.y_array(),
                    )
                    plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)
                    plus_topology = self.continuum.surface_provider.build_state(
                        plus
                    ).topology_hash
                    minus_state = self.solve(
                        minus,
                        f"cartesian/{step}/atom-{atom}/axis-{axis}/minus",
                        cold.y_array(),
                    )
                    minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)
                    minus_topology = self.continuum.surface_provider.build_state(
                        minus
                    ).topology_hash
                    finite_difference[atom, axis] = (
                        plus_energy - minus_energy
                    ) / (2.0 * step)
                    maximum_primal = max(
                        maximum_primal,
                        plus_state.actual_unmixed_residual_norm,
                        minus_state.actual_unmixed_residual_norm,
                    )
                    topology_hashes.update((plus_topology, minus_topology))
                    components.append(
                        {
                            "atom": atom,
                            "axis": axis,
                            "plus_geometry_sha256": geometry_sha256(plus),
                            "minus_geometry_sha256": geometry_sha256(minus),
                            "plus_energy_eV": plus_energy,
                            "minus_energy_eV": minus_energy,
                            "plus_primal_residual": (
                                plus_state.actual_unmixed_residual_norm
                            ),
                            "minus_primal_residual": (
                                minus_state.actual_unmixed_residual_norm
                            ),
                            "plus_topology_hash": plus_topology,
                            "minus_topology_hash": minus_topology,
                        }
                    )
                print(
                    f"completed Cartesian molecule={self.molecule.molecule_id} "
                    f"step={step:.8g} atom={atom}",
                    file=sys.stderr,
                    flush=True,
                )
            raw_steps.append({"step_A": step, "components": components})
            summary_samples.append((step, finite_difference))
        cartesian = summarize_cartesian_force_differences(analytic, summary_samples)
        cartesian.update(
            {
                "raw_displaced_components": raw_steps,
                "maximum_displaced_primal_residual": maximum_primal,
                "topology_hashes": sorted(topology_hashes),
                "fixed_topology": len(topology_hashes) == 1,
            }
        )
        warm_energy = self.scalar.evaluate_energy_components(atoms, warm.y)
        root = cold_warm_record(
            cold,
            warm,
            self.scalar,
            atoms,
            cold_evaluation=gradient.scalar,
            warm_evaluation=warm_energy,
        )
        print(
            f"completed Cartesian molecule={self.molecule.molecule_id}",
            file=sys.stderr,
            flush=True,
        )
        return {
            "molecule_id": self.molecule.molecule_id,
            "source_record_id": self.molecule.source_record_id,
            "chemical_formula": self.molecule.chemical_formula,
            "scope_tags": list(self.molecule.scope_tags),
            "variant": PES_CARTESIAN_PANEL_VARIANT,
            "component_count": 3 * len(atoms),
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
            "topology_hash": base_topology,
            "cartesian_force_fd": cartesian,
        }


def main() -> None:
    args = _parse_args()
    if (
        type(args.molecule_start) is not int
        or type(args.molecule_stop) is not int
        or not 0
        <= args.molecule_start
        < args.molecule_stop
        <= PES_PANEL_MOLECULE_COUNT
    ):
        raise ValueError("Molecule shard must satisfy 0 <= start < stop <= 20.")
    if not math.isfinite(args.primal_tolerance) or args.primal_tolerance <= 0.0:
        raise ValueError("--primal-tolerance must be finite and positive.")
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    determinism = _configure_numerical_determinism(args.device)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    molecules = load_pes_panel()[args.molecule_start : args.molecule_stop]
    started = time.perf_counter()
    records: list[dict[str, object]] = []
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
        shared_model = build_official_mace_polar_1_m_radial_gto_adapter(
            checkpoint_path=checkpoint,
            device=args.device,
            long_range_evaluator_profile=(
                MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[40]
            ),
        )
        for molecule in molecules:
            runner = _CartesianShardRunner(
                molecule, shared_model, primal_options, adjoint_options
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
                raise RuntimeError("Cartesian-panel profile changed within a shard.")
            if runner.profile.enabled or runner.profile.capabilities.enabled_tiers:
                raise RuntimeError("Cartesian diagnostic profile must remain disabled.")
            records.append(runner.run())

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(
        repository, (*source_paths, PANEL_ASSET_PATH)
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-fixedbox590-cartesian-panel-shard",
        "status": "diagnostic-shard-success",
        "claim_boundary": (
            "One preregistered shard of the disabled 20-molecule full-Cartesian "
            "same-scalar panel; not aggregate admission, not Tier E/F/H/V/M, not "
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
        "numerical_determinism": determinism,
        "device": args.device,
        "dtype": "float64",
        "identities": identities,
        "panel_contract": {
            "contract_version": PES_CARTESIAN_PANEL_CONTRACT_VERSION,
            "asset_sha256": PES_PANEL_ASSET_SHA256,
            "molecule_count": PES_PANEL_MOLECULE_COUNT,
            "variant": PES_CARTESIAN_PANEL_VARIANT,
            "cartesian_steps_A": list(PES_CARTESIAN_PANEL_STEPS_A),
            "convergence_contract": (
                "central-order-or-ten-percent-error-plateau-v1"
            ),
            "shard_start": args.molecule_start,
            "shard_stop": args.molecule_stop,
            "shard_molecule_ids": [item.molecule_id for item in molecules],
        },
        "measurements": records,
        "warnings": _warning_records(captured_warnings),
        "validation_scope": {
            "aggregate_cartesian_pes_panel": False,
            "shard_geometry_count": len(records),
            "shard_component_count": sum(
                int(record["component_count"]) for record in records
            ),
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
        }
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_CARTESIAN_PANEL_SHARD="
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
