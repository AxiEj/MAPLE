#!/usr/bin/env python3
"""Run one disabled shard of the 20-molecule symmetry/loop panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
import warnings

from ase import Atoms
import numpy as np

from maple.function.route2_smd_profiles import (
    MACEPOL_FORCED_RECIPROCAL_FIXED_BOX_PROFILES,
)
from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter
from maple.solvation.release import (
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_MOLECULE_COUNT,
    SYMMETRY_PANEL_CONTRACT_VERSION,
    SYMMETRY_PANEL_LOOP_AMPLITUDES_A,
    SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
    SYMMETRY_PANEL_ROTATION_COUNT,
    SYMMETRY_PANEL_TRANSLATION_A,
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    closed_loop_work,
    closed_rectangular_loop,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_directions,
    panel_geometries,
    reverse_closed_path,
    runtime_record,
    summarize_bidirectional_loop_record,
    summarize_rigid_symmetry,
    symmetry_loop_point_label,
    symmetry_panel_permutation,
    symmetry_panel_rotations,
    write_external_json_artifact,
)

from fixedbox590_water_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    build_system_with_model,
    cold_warm_record,
    identity_record,
    root_context,
)
from run_fixedbox590_pes_panel import (
    _configure_numerical_determinism,
    _warning_records,
)

SCHEMA_VERSION = "route2-fixedbox590-symmetry-panel-shard-v1"
PANEL_ASSET_PATH = "tools/route2_release/data/fixedbox590_pes_panel_v1.json"
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/pes_panel.py",
    "maple/solvation/release/pes_validation.py",
    "maple/solvation/release/symmetry_panel.py",
    "tools/route2_release/aggregate_fixedbox590_symmetry_panel.py",
    "tools/route2_release/run_fixedbox590_symmetry_panel.py",
)
CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--molecule-start", type=int, default=0)
    parser.add_argument("--molecule-stop", type=int, default=PES_PANEL_MOLECULE_COUNT)
    parser.add_argument("--max-iterations", type=int, default=120)
    return parser.parse_args()


class _SymmetryRunner:
    def __init__(self, molecule, model, max_iterations: int) -> None:
        self.molecule = molecule
        self.profile, self.model, self.continuum, self.equation, self.scalar = (
            build_system_with_model(molecule.atoms, model)
        )
        self.primal_options = FixedPointOptions(
            tolerance=1.0e-12,
            max_iterations=max_iterations,
            damping=0.7,
            history=6,
        )
        self.adjoint_options = AdjointOptions(
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            max_iterations=500,
        )

    def solve(self, atoms: Atoms, label: str, initial_y=None):
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

    def gradient(self, atoms: Atoms, label: str, initial_y=None):
        state = self.solve(atoms, label, initial_y)
        gradient = self.scalar.implicit_gradient(
            atoms, state, adjoint_options=self.adjoint_options
        )
        topology = self.continuum.surface_provider.build_state(atoms).topology_hash
        return state, gradient, topology

    def _rigid(self, atoms, base_state, base_gradient, base_topology):
        base_forces = np.asarray(base_gradient.forces, dtype=float).reshape(
            len(atoms), 3
        )
        base_source = base_state.source_array()
        base_energy = float(base_gradient.scalar.total_energy)
        translated = atoms.copy()
        translated.positions += np.asarray(SYMMETRY_PANEL_TRANSLATION_A)
        translated_state, translated_gradient, translated_topology = self.gradient(
            translated, "symmetry/translation", base_state.y_array()
        )
        translation_record = {
            "translation_A": list(SYMMETRY_PANEL_TRANSLATION_A),
            "energy_eV": translated_gradient.scalar.total_energy,
            "forces_eV_per_A": np.asarray(translated_gradient.forces)
            .reshape(len(atoms), 3)
            .tolist(),
            "source": translated_state.source_array().tolist(),
            "primal_residual": translated_state.actual_unmixed_residual_norm,
            "adjoint_residual": translated_gradient.adjoint.true_residual_norm,
            "topology_hash": translated_topology,
        }
        rotation_records = []
        for index, rotation in enumerate(
            symmetry_panel_rotations(self.molecule.molecule_id)
        ):
            rotated = atoms.copy()
            rotated.positions = atoms.positions @ rotation.T
            state, gradient, topology = self.gradient(
                rotated, f"symmetry/rotation-{index}"
            )
            rotation_records.append(
                {
                    "index": index,
                    "rotation_matrix": rotation.tolist(),
                    "expected_rotation_matrix": rotation.tolist(),
                    "energy_eV": gradient.scalar.total_energy,
                    "forces_eV_per_A": np.asarray(gradient.forces)
                    .reshape(len(atoms), 3)
                    .tolist(),
                    "source": state.source_array().tolist(),
                    "primal_residual": state.actual_unmixed_residual_norm,
                    "adjoint_residual": gradient.adjoint.true_residual_norm,
                    "topology_hash": topology,
                }
            )
        permutation = symmetry_panel_permutation(atoms.numbers)
        permuted = Atoms(
            numbers=atoms.numbers[permutation],
            positions=atoms.positions[permutation],
            info=dict(atoms.info),
        )
        permuted_state, permuted_gradient, permuted_topology = self.gradient(
            permuted, "symmetry/permutation"
        )
        permutation_record = {
            "permutation": permutation.tolist(),
            "energy_eV": permuted_gradient.scalar.total_energy,
            "forces_eV_per_A": np.asarray(permuted_gradient.forces)
            .reshape(len(atoms), 3)
            .tolist(),
            "source": permuted_state.source_array().tolist(),
            "primal_residual": permuted_state.actual_unmixed_residual_norm,
            "adjoint_residual": permuted_gradient.adjoint.true_residual_norm,
            "topology_hash": permuted_topology,
        }
        summary = summarize_rigid_symmetry(
            positions_A=atoms.positions,
            base_energy_eV=base_energy,
            base_forces_eV_per_A=base_forces,
            base_source=base_source,
            base_topology_hash=base_topology,
            translation_record=translation_record,
            permutation_record=permutation_record,
            rotation_records=rotation_records,
        )
        return summary, {
            "translation": translation_record,
            "permutation": permutation_record,
            "rotations": rotation_records,
        }

    @staticmethod
    def _loop_geometry(atoms, first, second, coefficient):
        result = atoms.copy()
        result.positions += (
            SYMMETRY_PANEL_LOOP_AMPLITUDES_A[0] * coefficient[0] * first
            + SYMMETRY_PANEL_LOOP_AMPLITUDES_A[1] * coefficient[1] * second
        )
        return result

    def _loop(self, atoms, base_state):
        directions = panel_directions(atoms, self.molecule.molecule_id)
        first = directions["seeded-internal"]
        raw_second = directions["radial-internal"]
        second = raw_second - float(np.vdot(first, raw_second)) * first
        second /= float(np.linalg.norm(second))
        forward = closed_rectangular_loop(
            subdivisions_per_edge=SYMMETRY_PANEL_LOOP_SUBDIVISIONS
        )
        reverse = reverse_closed_path(forward)

        def evaluate(coefficients, *, warm):
            values = []
            previous = base_state.y_array()
            for coefficient in coefficients:
                geometry = self._loop_geometry(atoms, first, second, coefficient)
                state, gradient, topology = self.gradient(
                    geometry,
                    symmetry_loop_point_label(coefficient),
                    previous if warm else None,
                )
                values.append((geometry, state, gradient, topology))
                if warm:
                    previous = state.y_array()
            return values

        cold = evaluate(forward, warm=False)
        warm_forward = evaluate(forward, warm=True)
        warm_reverse = evaluate(reverse, warm=True)

        def work(values):
            return closed_loop_work(
                [item[0].positions for item in values],
                [
                    np.asarray(item[2].forces, dtype=float).reshape(len(atoms), 3)
                    for item in values
                ],
                subdivisions_per_edge=SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
            )

        cold_forward = work(cold)
        cold_reverse = work(tuple(reversed(cold)))
        warm_forward_work = work(warm_forward)
        warm_reverse_work = work(warm_reverse)
        forward_roots = [
            cold_warm_record(cold_item[1], warm_item[1], self.scalar, cold_item[0])
            for cold_item, warm_item in zip(cold, warm_forward, strict=True)
        ]
        reverse_roots = [
            cold_warm_record(cold_item[1], warm_item[1], self.scalar, cold_item[0])
            for cold_item, warm_item in zip(
                tuple(reversed(cold)), warm_reverse, strict=True
            )
        ]
        all_roots = all(
            bool(item["numerically_equivalent"])
            and all(bool(value) for value in item["gates"].values())
            for item in (*forward_roots, *reverse_roots)
        )
        repeat_records = []
        repeat_pass = True
        for forward_item, reverse_item in zip(
            warm_forward, tuple(reversed(warm_reverse)), strict=True
        ):
            source_difference = float(
                np.linalg.norm(
                    forward_item[1].source_array() - reverse_item[1].source_array()
                )
            )
            source_relative = source_difference / max(
                float(np.linalg.norm(forward_item[1].source_array())),
                float(np.linalg.norm(reverse_item[1].source_array())),
                1.0e-15,
            )
            energy_difference = abs(
                forward_item[2].scalar.total_energy
                - reverse_item[2].scalar.total_energy
            )
            passed = source_relative <= 1.0e-8 and energy_difference <= 1.0e-8
            repeat_pass &= passed
            repeat_records.append(
                {
                    "geometry_sha256": geometry_sha256(forward_item[0]),
                    "source_relative_difference": source_relative,
                    "energy_abs_difference_eV": energy_difference,
                    "gate_passed": passed,
                }
            )
        all_evaluations = (*cold, *warm_forward, *warm_reverse)
        raw = {
            "coordinate_names": ["seeded-internal", "orthogonal-radial-internal"],
            "coordinate_directions": [first.tolist(), second.tolist()],
            "amplitudes_A": list(SYMMETRY_PANEL_LOOP_AMPLITUDES_A),
            "subdivisions_per_edge": SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
            "forward_coefficients": forward,
            "reverse_coefficients": reverse,
            "cold_forward": cold_forward,
            "cold_reverse": cold_reverse,
            "warm_forward": warm_forward_work,
            "warm_reverse": warm_reverse_work,
            "all_cold_warm_roots": all_roots,
            "warm_forward_reverse_repeat": repeat_pass,
            "cold_warm_roots": {
                "forward": forward_roots,
                "reverse": reverse_roots,
            },
            "warm_repeat_records": repeat_records,
            "maximum_primal_residual": max(
                item[1].actual_unmixed_residual_norm for item in all_evaluations
            ),
            "maximum_adjoint_residual": max(
                item[2].adjoint.true_residual_norm for item in all_evaluations
            ),
            "topology_hashes": sorted({item[3] for item in all_evaluations}),
        }
        summary = summarize_bidirectional_loop_record(raw)
        return {**raw, **summary}

    def run(self):
        atoms = panel_geometries(self.molecule)[PES_CARTESIAN_PANEL_VARIANT]
        base_state, base_gradient, base_topology = self.gradient(atoms, "base")
        rigid, raw_rigid = self._rigid(
            atoms, base_state, base_gradient, base_topology
        )
        loop = self._loop(atoms, base_state)
        print(
            f"completed symmetry molecule={self.molecule.molecule_id}",
            file=sys.stderr,
            flush=True,
        )
        return {
            "molecule_id": self.molecule.molecule_id,
            "source_record_id": self.molecule.source_record_id,
            "chemical_formula": self.molecule.chemical_formula,
            "scope_tags": list(self.molecule.scope_tags),
            "variant": PES_CARTESIAN_PANEL_VARIANT,
            "contract_version": SYMMETRY_PANEL_CONTRACT_VERSION,
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "geometry_sha256": geometry_sha256(atoms),
            "base": {
                "energy_eV": base_gradient.scalar.total_energy,
                "forces_eV_per_A": np.asarray(base_gradient.forces)
                .reshape(len(atoms), 3)
                .tolist(),
                "source": base_state.source_array().tolist(),
                "primal_residual": base_state.actual_unmixed_residual_norm,
                "adjoint_residual": base_gradient.adjoint.true_residual_norm,
                "topology_hash": base_topology,
            },
            "rigid_symmetry": rigid,
            "_raw_rigid_records": raw_rigid,
            "closed_loop": loop,
            "all_gates_passed": bool(rigid["all_gates_passed"])
            and bool(loop["all_gates_passed"]),
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
    records = []
    identities = None
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
            runner = _SymmetryRunner(molecule, shared_model, args.max_iterations)
            current = identity_record(
                runner.model, runner.continuum, runner.equation, runner.scalar
            )
            if identities is None:
                identities = current
            elif current["profile_id"] != identities["profile_id"]:
                raise RuntimeError("Symmetry-panel profile changed within a shard.")
            if runner.profile.enabled or runner.profile.capabilities.enabled_tiers:
                raise RuntimeError("Symmetry diagnostic profile must remain disabled.")
            records.append(runner.run())

    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(
        repository, (*source_paths, PANEL_ASSET_PATH)
    )
    contract = {
        "contract_version": SYMMETRY_PANEL_CONTRACT_VERSION,
        "asset_sha256": PES_PANEL_ASSET_SHA256,
        "molecule_count": PES_PANEL_MOLECULE_COUNT,
        "variant": PES_CARTESIAN_PANEL_VARIANT,
        "rotation_count": SYMMETRY_PANEL_ROTATION_COUNT,
        "translation_A": list(SYMMETRY_PANEL_TRANSLATION_A),
        "loop_amplitudes_A": list(SYMMETRY_PANEL_LOOP_AMPLITUDES_A),
        "loop_subdivisions_per_edge": SYMMETRY_PANEL_LOOP_SUBDIVISIONS,
        "shard_start": args.molecule_start,
        "shard_stop": args.molecule_stop,
        "shard_molecule_ids": [item.molecule_id for item in molecules],
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "disabled-real-stack-symmetry-loop-panel-shard",
        "status": "diagnostic-shard-success",
        "claim_boundary": (
            "One source-bound rigid-symmetry and closed-loop shard for a disabled "
            "electrostatic profile; not Tier admission, complete solvation free "
            "energy, chemical accuracy, Hessian, FREQ, TS, HVP, NVE, or MD."
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
        "ROUTE2_FIXEDBOX590_SYMMETRY_PANEL_SHARD="
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
