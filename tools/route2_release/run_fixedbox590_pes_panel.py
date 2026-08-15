#!/usr/bin/env python3
"""Run one preregistered shard of the disabled fixed-box590 PES panel.

Each shard writes raw evidence outside the checkout.  Aggregation and any
admission decision are separate; this runner cannot enable E/F/H/V/M.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from panel_continuum_identity import continuum_topology_hash

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


def _configure_numerical_determinism(device: str) -> dict[str, object]:
    """Fail closed unless an advertised CUDA shard is replay-deterministic."""

    normalized = str(device).strip().lower()
    if not normalized:
        raise ValueError("--device must be a non-empty device identifier.")
    if not normalized.startswith("cuda"):
        return {
            "mode": "cpu-runtime-default",
            "device": normalized,
            "torch_deterministic_algorithms": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUDA PES evidence requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8."
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PES evidence was requested but CUDA is unavailable.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms did not remain enabled.")
    return {
        "mode": "pytorch-deterministic-cuda-v1",
        "device": normalized,
        "torch_deterministic_algorithms": True,
        "torch_deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": workspace,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }


class _ShardRunner:
    def __init__(
        self,
        molecule,
        model,
        primal_options: FixedPointOptions,
        adjoint_options: AdjointOptions,
        *,
        system_builder=build_system_with_model,
        root_context_builder=root_context,
        state_record_builder=state_record,
        cold_warm_record_builder=cold_warm_record,
        topology_hash_builder=continuum_topology_hash,
        scalar_identity_builder=None,
        domain_record_builder=None,
        alternate_root_seed_values=(1.0e-3,),
        record_root_multistart: bool = False,
        box_length: int = 40,
    ) -> None:
        self.molecule = molecule
        self.profile, self.model, self.continuum, self.equation, self.scalar = (
            system_builder(molecule.atoms, model)
        )
        self.primal_options = primal_options
        self.adjoint_options = adjoint_options
        self.root_context_builder = root_context_builder
        self.state_record_builder = state_record_builder
        self.cold_warm_record_builder = cold_warm_record_builder
        self.topology_hash_builder = topology_hash_builder
        self.scalar_identity_builder = scalar_identity_builder
        self.domain_record_builder = domain_record_builder
        self.alternate_root_seed_values = tuple(
            float(value) for value in alternate_root_seed_values
        )
        self.record_root_multistart = bool(record_root_multistart)
        if not self.alternate_root_seed_values or any(
            not math.isfinite(value) for value in self.alternate_root_seed_values
        ):
            raise ValueError("alternate_root_seed_values must be finite and non-empty.")
        self.box_length = box_length

    def solve(self, atoms, label: str, initial_y=None):
        return solve_fixed_point(
            self.equation,
            atoms,
            scalar_id=self.scalar.scalar_id,
            profile_id=self.scalar.profile_id,
            scalar_binding=self.scalar,
            root_context_id=self.root_context_builder(
                atoms,
                label,
                box_length=self.box_length,
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
        topology_hashes = {self.topology_hash_builder(self.continuum, atoms)}
        plus_seed = state.y_array()
        minus_seed = state.y_array()
        for step in PES_PANEL_DIRECTIONAL_STEPS_A:
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions += step * direction
            minus.positions -= step * direction
            plus_state = self.solve(
                plus,
                f"{label}/{name}/{step}/plus",
                plus_seed,
            )
            # Consume the plus geometry while its exact single-entry dense
            # continuum factorization is still cached. Solving minus first and
            # returning to plus would rebuild the same geometry unnecessarily.
            plus_energy = self.scalar.evaluate_energy(plus, plus_state.y)
            plus_topology = self.topology_hash_builder(self.continuum, plus)
            minus_state = self.solve(
                minus,
                f"{label}/{name}/{step}/minus",
                minus_seed,
            )
            minus_energy = self.scalar.evaluate_energy(minus, minus_state.y)
            minus_topology = self.topology_hash_builder(self.continuum, minus)
            maximum_primal = max(
                maximum_primal,
                plus_state.actual_unmixed_residual_norm,
                minus_state.actual_unmixed_residual_norm,
            )
            plus_seed = plus_state.y_array()
            minus_seed = minus_state.y_array()
            # Fixed-topology identity is a surface property.  Reading it from
            # the surface provider avoids a second dense continuum build at
            # every displaced energy point.
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
                    "plus_iterations": len(plus_state.iterations) - 1,
                    "minus_iterations": len(minus_state.iterations) - 1,
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
        records: list[dict[str, object]] = []
        for variant, atoms in panel_geometries(self.molecule).items():
            cold = self.solve(atoms, f"panel/{variant}")
            gradient = self.scalar.implicit_gradient(
                atoms, cold, adjoint_options=self.adjoint_options
            )
            topology = self.topology_hash_builder(self.continuum, atoms)
            root_replays = []
            warm_states = []
            for seed_value in self.alternate_root_seed_values:
                warm_seed = np.full(self.equation.reduced_dimension, seed_value)
                warm = self.solve(atoms, f"panel/{variant}", warm_seed)
                warm_states.append(warm)
                warm_energy = self.scalar.evaluate_energy_components(atoms, warm.y)
                root_replays.append(
                    self.cold_warm_record_builder(
                        cold,
                        warm,
                        self.scalar,
                        atoms,
                        cold_evaluation=gradient.scalar,
                        warm_evaluation=warm_energy,
                    )
                )
            root = root_replays[0]
            directions = panel_directions(atoms, self.molecule.molecule_id)
            directional = {
                name: self.direction_record(
                    atoms,
                    f"panel/{variant}",
                    cold,
                    gradient.total_coordinate_gradient,
                    name,
                    directions[name],
                )
                for name in PES_PANEL_DIRECTION_NAMES
            }
            maximum_primal = max(
                cold.actual_unmixed_residual_norm,
                *(warm.actual_unmixed_residual_norm for warm in warm_states),
                *(record["maximum_primal_residual"] for record in directional.values()),
            )
            record = {
                "molecule_id": self.molecule.molecule_id,
                "source_record_id": self.molecule.source_record_id,
                "chemical_formula": self.molecule.chemical_formula,
                "scope_tags": list(self.molecule.scope_tags),
                "variant": variant,
                "atomic_numbers": atoms.numbers.tolist(),
                "positions_A": atoms.positions.tolist(),
                "geometry_sha256": geometry_sha256(atoms),
                "cold_state": self.state_record_builder(
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
            if self.record_root_multistart:
                record["root_multistart"] = root_replays
            if self.scalar_identity_builder is not None:
                record["scalar_identity"] = self.scalar_identity_builder(
                    self.continuum, self.scalar, atoms, cold
                )
            if self.domain_record_builder is not None:
                record["domain"] = self.domain_record_builder(self.continuum, atoms)
            records.append(record)
            print(
                f"completed molecule={self.molecule.molecule_id} variant={variant}",
                file=sys.stderr,
                flush=True,
            )
        return records

    def path_point_record(self, point, previous_y=None):
        atoms = point.atoms
        cold = self.solve(atoms, f"path/{point.path_name}/{point.point_label}")
        warm_seeds = [
            np.full(self.equation.reduced_dimension, value)
            for value in self.alternate_root_seed_values
        ]
        if previous_y is not None:
            warm_seeds[0] = np.asarray(previous_y, dtype=float)
        warm_states = [
            self.solve(
                atoms,
                f"path/{point.path_name}/{point.point_label}",
                warm_seed,
            )
            for warm_seed in warm_seeds
        ]
        gradient = self.scalar.implicit_gradient(
            atoms, cold, adjoint_options=self.adjoint_options
        )
        tangent = (
            torsion_tangent(atoms)
            if point.path_name == PES_PANEL_ADDITIONAL_PATHS[0]
            else stretch_tangent(atoms)
        )
        tangent_norm = float(np.linalg.norm(tangent))
        if not math.isfinite(tangent_norm) or tangent_norm <= 1.0e-15:
            raise RuntimeError("Path coordinate tangent is singular.")
        unit_direction = tangent / tangent_norm
        analytic_derivative = float(
            np.vdot(
                np.asarray(gradient.total_coordinate_gradient).reshape(len(atoms), 3),
                tangent,
            )
        )
        topology = self.topology_hash_builder(self.continuum, atoms)
        local_force = self.direction_record(
            atoms,
            f"path/{point.path_name}/{point.point_label}",
            cold,
            gradient.total_coordinate_gradient,
            "coordinate-tangent",
            unit_direction,
        )
        root_replays = [
            self.cold_warm_record_builder(
                cold,
                warm,
                self.scalar,
                atoms,
                cold_evaluation=gradient.scalar,
                warm_evaluation=self.scalar.evaluate_energy_components(atoms, warm.y),
            )
            for warm in warm_states
        ]
        record = {
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
            "cold_state": self.state_record_builder(
                cold, self.scalar, atoms, evaluation=gradient.scalar
            ),
            "cold_warm": root_replays[0],
            "energy_eV": gradient.scalar.total_energy,
            "forces_eV_per_A": np.asarray(gradient.forces)
            .reshape(len(atoms), 3)
            .tolist(),
            "coordinate_tangent_A_per_coordinate_unit": tangent.tolist(),
            "coordinate_tangent_norm_A_per_coordinate_unit": tangent_norm,
            "analytic_coordinate_derivative_eV_per_coordinate_unit": (
                analytic_derivative
            ),
            "normalized_cartesian_direction": unit_direction.tolist(),
            "local_tangent_force_fd": local_force,
            "maximum_primal_residual": max(
                cold.actual_unmixed_residual_norm,
                *(warm.actual_unmixed_residual_norm for warm in warm_states),
            ),
            "adjoint_residual": gradient.adjoint.true_residual_norm,
            "topology_hash": topology,
        }
        if self.record_root_multistart:
            record["root_multistart"] = root_replays
        if self.scalar_identity_builder is not None:
            record["scalar_identity"] = self.scalar_identity_builder(
                self.continuum, self.scalar, atoms, cold
            )
        if self.domain_record_builder is not None:
            record["domain"] = self.domain_record_builder(self.continuum, atoms)
        return record, cold.y_array()


def run_pes_panel(
    args: argparse.Namespace,
    *,
    schema_version: str = SCHEMA_VERSION,
    contract_version: str = PES_PANEL_CONTRACT_VERSION,
    required_source_paths=REQUIRED_SOURCE_PATHS,
    model_evaluator_profile=MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES[40],
    system_builder=build_system_with_model,
    root_context_builder=root_context,
    state_record_builder=state_record,
    cold_warm_record_builder=cold_warm_record,
    identity_record_builder=identity_record,
    topology_hash_builder=continuum_topology_hash,
    scalar_identity_builder=None,
    domain_record_builder=None,
    alternate_root_seed_values=(1.0e-3,),
    record_root_multistart: bool = False,
    contract_metadata=None,
    box_length: int = 40,
    output_marker: str = "ROUTE2_FIXEDBOX590_PES_PANEL_SHARD",
    artifact_kind: str = "disabled-real-stack-fixedbox590-pes-panel-shard",
) -> None:
    if (
        type(args.molecule_start) is not int
        or type(args.molecule_stop) is not int
        or not 0 <= args.molecule_start < args.molecule_stop <= PES_PANEL_MOLECULE_COUNT
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
            long_range_evaluator_profile=model_evaluator_profile,
        )
        for molecule in molecules:
            runner = _ShardRunner(
                molecule,
                shared_model,
                primal_options,
                adjoint_options,
                system_builder=system_builder,
                root_context_builder=root_context_builder,
                state_record_builder=state_record_builder,
                cold_warm_record_builder=cold_warm_record_builder,
                topology_hash_builder=topology_hash_builder,
                scalar_identity_builder=scalar_identity_builder,
                domain_record_builder=domain_record_builder,
                alternate_root_seed_values=alternate_root_seed_values,
                record_root_multistart=record_root_multistart,
                box_length=box_length,
            )
            current_identities = identity_record_builder(
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
        repository.root, required_paths=required_source_paths
    )
    # collect_loaded_repository_sources intentionally accepts only Python
    # sources. Bind the frozen JSON geometry asset separately as an exact Git
    # blob, then store both in the one source-files ledger.
    source_hashes = committed_source_hashes(
        repository, (*source_paths, PANEL_ASSET_PATH)
    )
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "artifact_kind": artifact_kind,
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
        "numerical_determinism": determinism,
        "device": args.device,
        "dtype": "float64",
        "identities": identities,
        "panel_contract": {
            "contract_version": contract_version,
            "asset_sha256": PES_PANEL_ASSET_SHA256,
            "molecule_count": PES_PANEL_MOLECULE_COUNT,
            "variant_names": list(PES_PANEL_VARIANT_NAMES),
            "additional_path_contracts": list(PES_PANEL_ADDITIONAL_PATHS),
            "direction_names": list(PES_PANEL_DIRECTION_NAMES),
            "directional_steps_A": list(PES_PANEL_DIRECTIONAL_STEPS_A),
            "shard_start": args.molecule_start,
            "shard_stop": args.molecule_stop,
            "shard_molecule_ids": [item.molecule_id for item in molecules],
            **({} if contract_metadata is None else dict(contract_metadata)),
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
        output_marker
        + "="
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


def main() -> None:
    run_pes_panel(_parse_args())


if __name__ == "__main__":
    main()
