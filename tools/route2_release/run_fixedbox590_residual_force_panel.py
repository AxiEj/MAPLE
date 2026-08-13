#!/usr/bin/env python3
"""Run one disabled shard of the residual-refinement force-error panel."""

from __future__ import annotations

import argparse
import json
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
    ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES,
    ADJOINT_REFINEMENT_RELATIVE_TOLERANCES,
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_MOLECULE_COUNT,
    PRIMAL_REFINEMENT_TOLERANCES,
    RESIDUAL_FORCE_CONTRACT_VERSION,
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_geometries,
    runtime_record,
    summarize_residual_force_refinement,
    write_external_json_artifact,
)

from fixedbox590_water_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    build_system_with_model,
    identity_record,
    root_context,
)
from run_fixedbox590_pes_panel import (
    _configure_numerical_determinism,
    _warning_records,
)

SCHEMA_VERSION = "route2-fixedbox590-residual-force-panel-shard-v1"
PANEL_ASSET_PATH = "tools/route2_release/data/fixedbox590_pes_panel_v1.json"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/pes_panel.py",
    "maple/solvation/release/residual_force.py",
    "tools/route2_release/aggregate_fixedbox590_residual_force_panel.py",
    "tools/route2_release/run_fixedbox590_residual_force_panel.py",
)
CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--molecule-start", type=int, default=0)
    parser.add_argument("--molecule-stop", type=int, default=PES_PANEL_MOLECULE_COUNT)
    parser.add_argument("--max-iterations", type=int, default=160)
    return parser.parse_args()


class _ResidualForceRunner:
    def __init__(self, molecule, model, max_iterations: int) -> None:
        self.molecule = molecule
        self.profile, self.model, self.continuum, self.equation, self.scalar = (
            build_system_with_model(molecule.atoms, model)
        )
        self.max_iterations = max_iterations

    def _solve(self, atoms, tolerance: float, initial_y=None):
        return solve_fixed_point(
            self.equation,
            atoms,
            scalar_id=self.scalar.scalar_id,
            profile_id=self.scalar.profile_id,
            scalar_binding=self.scalar,
            root_context_id=root_context(
                atoms,
                f"residual-force/primal-{tolerance:.0e}",
                system_id=self.molecule.molecule_id,
            ),
            initial_y=initial_y,
            options=FixedPointOptions(
                tolerance=tolerance,
                max_iterations=self.max_iterations,
                damping=0.7,
                history=6,
            ),
        )

    def _force_record(self, atoms, state, relative: float, absolute: float):
        gradient = self.scalar.implicit_gradient(
            atoms,
            state,
            adjoint_options=AdjointOptions(
                relative_tolerance=relative,
                absolute_tolerance=absolute,
                max_iterations=800,
            ),
        )
        return {
            "primal_tolerance": state.primal_tolerance,
            "adjoint_relative_tolerance": relative,
            "adjoint_absolute_tolerance": absolute,
            "actual_primal_residual": state.actual_unmixed_residual_norm,
            "actual_adjoint_residual": gradient.adjoint.true_residual_norm,
            "root_hash": state.root_hash,
            "forces_eV_per_A": np.asarray(gradient.forces, dtype=float)
            .reshape(len(atoms), 3)
            .tolist(),
        }

    def run(self) -> dict[str, object]:
        atoms = panel_geometries(self.molecule)[PES_CARTESIAN_PANEL_VARIANT]
        states = []
        initial_y = None
        for tolerance in PRIMAL_REFINEMENT_TOLERANCES:
            state = self._solve(atoms, tolerance, initial_y=initial_y)
            states.append(state)
            initial_y = state.y_array()

        adjoint_levels = [
            self._force_record(atoms, states[0], relative, absolute)
            for relative, absolute in zip(
                ADJOINT_REFINEMENT_RELATIVE_TOLERANCES,
                ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES,
            )
        ]
        primal_levels = [
            self._force_record(
                atoms,
                state,
                ADJOINT_REFINEMENT_RELATIVE_TOLERANCES[-1],
                ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES[-1],
            )
            for state in states
        ]
        # Reuse the exact already-computed bridge record, avoiding even a
        # redundant deterministic solve in the evidence payload.
        primal_levels[0] = dict(adjoint_levels[-1])
        summary = summarize_residual_force_refinement(primal_levels, adjoint_levels)
        topology_hash = self.continuum.surface_provider.build_state(atoms).topology_hash
        print(
            f"completed residual-force molecule={self.molecule.molecule_id}",
            file=sys.stderr,
            flush=True,
        )
        return {
            "molecule_id": self.molecule.molecule_id,
            "source_record_id": self.molecule.source_record_id,
            "chemical_formula": self.molecule.chemical_formula,
            "scope_tags": list(self.molecule.scope_tags),
            "variant": PES_CARTESIAN_PANEL_VARIANT,
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "geometry_sha256": geometry_sha256(atoms),
            "topology_hash": topology_hash,
            **summary,
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
    if type(args.max_iterations) is not int or args.max_iterations < 1:
        raise ValueError("--max-iterations must be a positive integer.")
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    determinism = _configure_numerical_determinism(args.device)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    molecules = load_pes_panel()[args.molecule_start : args.molecule_stop]
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    identities: dict[str, object] | None = None
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
            runner = _ResidualForceRunner(molecule, shared_model, args.max_iterations)
            current_identities = identity_record(
                runner.model,
                runner.continuum,
                runner.equation,
                runner.scalar,
            )
            if identities is None:
                identities = current_identities
            elif current_identities["profile_id"] != identities["profile_id"]:
                raise RuntimeError("Residual-force profile changed within a shard.")
            if runner.profile.enabled or runner.profile.capabilities.enabled_tiers:
                raise RuntimeError("Residual-force diagnostic profile must be disabled.")
            records.append(runner.run())

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(
        repository, (*source_paths, PANEL_ASSET_PATH)
    )
    contract = {
        "contract_version": RESIDUAL_FORCE_CONTRACT_VERSION,
        "asset_sha256": PES_PANEL_ASSET_SHA256,
        "molecule_count": PES_PANEL_MOLECULE_COUNT,
        "variant": PES_CARTESIAN_PANEL_VARIANT,
        "primal_tolerances": list(PRIMAL_REFINEMENT_TOLERANCES),
        "adjoint_relative_tolerances": list(
            ADJOINT_REFINEMENT_RELATIVE_TOLERANCES
        ),
        "adjoint_absolute_tolerances": list(
            ADJOINT_REFINEMENT_ABSOLUTE_TOLERANCES
        ),
        "shard_start": args.molecule_start,
        "shard_stop": args.molecule_stop,
        "shard_molecule_ids": [item.molecule_id for item in molecules],
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-residual-force-panel-shard",
        "status": "diagnostic-shard-success",
        "claim_boundary": (
            "One source-bound residual-refinement shard; empirical a-posteriori "
            "force-error estimate, not a rigorous upper bound, Tier admission, "
            "complete solvation free energy, or chemical-accuracy evidence."
        ),
        "capabilities": CAPABILITIES,
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
        "panel_contract": contract,
        "measurements": records,
        "warnings": _warning_records(captured_warnings),
        "runtime_seconds": time.perf_counter() - started,
    }
    payload["measurement_sha256"] = canonical_json_sha256(
        {"contract": contract, "measurements": records}
    )
    repository.assert_unchanged()
    file_record = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_FIXEDBOX590_RESIDUAL_FORCE_SHARD="
        + json.dumps(
            {
                "artifact": file_record,
                "measurement_sha256": payload["measurement_sha256"],
                "capabilities": CAPABILITIES,
                "shard_start": args.molecule_start,
                "shard_stop": args.molecule_stop,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
