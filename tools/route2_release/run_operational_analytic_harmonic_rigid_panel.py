#!/usr/bin/env python3
"""Run a disabled multi-molecule rigid-symmetry shard for the harmonic PES."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
import warnings

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.solvation.coupling.adjoint import AdjointOptions
from maple.solvation.coupling.fixed_point import (
    FixedPointOptions,
    roots_numerically_equivalent,
    solve_fixed_point,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release import (
    PES_CARTESIAN_PANEL_VARIANT,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_MOLECULE_COUNT,
    SYMMETRY_PANEL_ROTATION_COUNT,
    SYMMETRY_PANEL_TRANSLATION_A,
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    load_pes_panel,
    panel_geometries,
    rotate_radial_gto_blocks,
    runtime_record,
    summarize_rigid_symmetry,
    symmetry_panel_permutation,
    symmetry_panel_rotations,
    write_external_json_artifact,
)

from operational_analytic_harmonic_common import (
    COMMON_REQUIRED_SOURCE_PATHS,
    DEFAULT_CHECKPOINT,
    EXPOSURE_LMAX,
    EXPOSURE_RADIAL_QUADRATURE_ORDER,
    GREEN_RADIAL_QUADRATURE_ORDER,
    NO_CAPABILITIES,
    PROFILE_ID,
    SCALAR_ID,
    SOURCE_RADIAL_QUADRATURE_ORDER,
    SURFACE_LMAX,
    TRANSITION_WIDTH_ANGSTROM2,
    build_model,
    build_system_with_model,
    identity_record,
    root_context,
)

SCHEMA_VERSION = "route2-operational-analytic-harmonic-rigid-panel-shard-v1"
CONTRACT_VERSION = "route2-operational-analytic-harmonic-rigid-panel-v1"
ARTIFACT_KIND = "disabled-operational-analytic-harmonic-rigid-panel-shard"
PANEL_ASSET_PATH = "tools/route2_release/data/fixedbox590_pes_panel_v1.json"
FIELD_COVARIANCE_RELATIVE_TOLERANCE = 1.0e-4
SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV = 1.0e-10
MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE = 1.0e-8
PRIMAL_TOLERANCE = 1.0e-12
ROOT_DAMPING = 0.7
ROOT_HISTORY = 6
ADJOINT_OPTIONS = AdjointOptions(
    relative_tolerance=1.0e-11,
    absolute_tolerance=1.0e-13,
    max_iterations=500,
)
REQUIRED_SOURCE_PATHS = COMMON_REQUIRED_SOURCE_PATHS + (
    "maple/solvation/release/pes_panel.py",
    "maple/solvation/release/symmetry_panel.py",
    "tools/route2_release/run_operational_analytic_harmonic_rigid_panel.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one disabled operational harmonic rigid-symmetry shard; "
            "this cannot admit Route-2 E/F/H/V/M."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--molecule-start", type=int, default=0)
    parser.add_argument("--molecule-stop", type=int, default=PES_PANEL_MOLECULE_COUNT)
    parser.add_argument("--max-iterations", type=int, default=160)
    return parser.parse_args()


def _configure_determinism(torch, device: str) -> dict[str, object]:
    torch.manual_seed(20260820)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260820)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA panel requested but CUDA is unavailable.")
    return {
        "seed": 20260820,
        "device": device,
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _warning_records(captured) -> list[dict[str, object]]:
    return [
        {
            "category": item.category.__name__,
            "message": str(item.message),
            "filename": item.filename,
            "lineno": item.lineno,
        }
        for item in captured
    ]


def _relative(actual: object, expected: object) -> float:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    return float(np.linalg.norm(left - right)) / max(
        float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0e-15
    )


class _RigidRunner:
    def __init__(self, molecule, model, max_iterations: int) -> None:
        self.molecule = molecule
        (
            self.profile,
            self.model,
            self.continuum,
            self.equation,
            self.scalar,
        ) = build_system_with_model(molecule.atoms, model)
        self.root_options = FixedPointOptions(
            method="anderson",
            tolerance=PRIMAL_TOLERANCE,
            max_iterations=max_iterations,
            damping=ROOT_DAMPING,
            history=ROOT_HISTORY,
        )

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
            options=self.root_options,
        )

    def gradient(self, atoms, label: str, initial_y=None):
        state = self.solve(atoms, label, initial_y)
        gradient = self.scalar.implicit_gradient(
            atoms, state, adjoint_options=ADJOINT_OPTIONS
        )
        return state, gradient

    @staticmethod
    def _state_record(state, gradient) -> dict[str, object]:
        atom_count = state.source_array().shape[0]
        return {
            "converged": state.converged,
            "initialization": state.initialization,
            "iterations": len(state.iterations) - 1,
            "root_context_id": state.root_context_id,
            "root_hash": state.root_hash,
            "primal_residual": state.actual_unmixed_residual_norm,
            "adjoint_residual": gradient.adjoint.true_residual_norm,
            "energy_eV": gradient.scalar.total_energy,
            "forces_eV_per_A": np.asarray(gradient.forces)
            .reshape(atom_count, 3)
            .tolist(),
            "source": state.source_array().tolist(),
            "field": state.field_array().tolist(),
        }

    def _cold_warm(self, atoms, cold, cold_gradient):
        warm = self.solve(atoms, "reference", cold.y)
        warm_energy = self.scalar.evaluate_energy(atoms, warm.y)
        source_relative = _relative(cold.source_array(), warm.source_array())
        field_relative = _relative(cold.field_array(), warm.field_array())
        energy_absolute = abs(cold_gradient.scalar.total_energy - warm_energy)
        record = {
            "contract": "route2-root-equivalence-v1",
            "numerically_equivalent": roots_numerically_equivalent(cold, warm),
            "source_relative_difference": source_relative,
            "field_relative_difference": field_relative,
            "energy_absolute_difference_eV": energy_absolute,
            "warm_iterations": len(warm.iterations) - 1,
            "warm_primal_residual": warm.actual_unmixed_residual_norm,
        }
        record["gate_passed"] = bool(
            record["numerically_equivalent"]
            and source_relative <= 1.0e-8
            and field_relative <= 1.0e-8
            and energy_absolute <= 1.0e-8
        )
        return record

    def _scalar_identity(self, atoms, state) -> dict[str, object]:
        source = state.source_array()
        field = state.field_array()
        functional = self.continuum.energy_eV(atoms, source)
        half_coupling = 0.5 * float(self.scalar.metric.pair(source, field))
        error = abs(functional - half_coupling)
        missing_max = float(np.max(np.abs(source[:, (1, 5, 6, 7)])))
        return {
            "functional_energy_eV": functional,
            "half_coupling_energy_eV": half_coupling,
            "absolute_error_eV": error,
            "missing_radial_block_max_abs": missing_max,
            "gate_passed": (
                error <= SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV
                and missing_max <= MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE
            ),
        }

    def run(self) -> dict[str, object]:
        atoms = panel_geometries(self.molecule)[PES_CARTESIAN_PANEL_VARIANT]
        base_state, base_gradient = self.gradient(atoms, "reference")
        topology = self.continuum.topology_sha256()
        base = self._state_record(base_state, base_gradient)
        base["topology_hash"] = topology
        cold_warm = self._cold_warm(atoms, base_state, base_gradient)
        scalar_identity = self._scalar_identity(atoms, base_state)

        translated = atoms.copy()
        translated.positions += np.asarray(SYMMETRY_PANEL_TRANSLATION_A)
        translated_state, translated_gradient = self.gradient(
            translated, "translation", base_state.y
        )
        translation = self._state_record(translated_state, translated_gradient)
        translation.update(
            {
                "translation_A": list(SYMMETRY_PANEL_TRANSLATION_A),
                "topology_hash": self.continuum.topology_sha256(),
            }
        )

        permutation = symmetry_panel_permutation(atoms.numbers)
        permuted = atoms[permutation]
        permuted.info.update(atoms.info)
        expected_permuted_source = base_state.source_array()[permutation]
        permuted_state, permuted_gradient = self.gradient(
            permuted,
            "permutation",
            self.equation.coordinates.reduce(expected_permuted_source),
        )
        permutation_record = self._state_record(permuted_state, permuted_gradient)
        permutation_record.update(
            {
                "permutation": permutation.tolist(),
                "topology_hash": self.continuum.topology_sha256(),
            }
        )

        rotations = []
        for index, rotation in enumerate(
            symmetry_panel_rotations(self.molecule.molecule_id)
        ):
            rotated = atoms.copy()
            rotated.positions = atoms.positions @ rotation.T
            expected_source = rotate_radial_gto_blocks(
                base_state.source_array(), rotation
            )
            state, gradient = self.gradient(
                rotated,
                f"rotation-{index}",
                self.equation.coordinates.reduce(expected_source),
            )
            record = self._state_record(state, gradient)
            record.update(
                {
                    "index": index,
                    "rotation_matrix": rotation.tolist(),
                    "expected_rotation_matrix": rotation.tolist(),
                    "topology_hash": self.continuum.topology_sha256(),
                }
            )
            rotations.append(record)

        rigid = summarize_rigid_symmetry(
            positions_A=atoms.positions,
            base_energy_eV=base["energy_eV"],
            base_forces_eV_per_A=base["forces_eV_per_A"],
            base_source=base["source"],
            base_topology_hash=topology,
            translation_record=translation,
            permutation_record=permutation_record,
            rotation_records=rotations,
        )
        field_errors = {
            "translation_relative": _relative(
                translation["field"], base_state.field_array()
            ),
            "permutation_relative": _relative(
                permutation_record["field"], base_state.field_array()[permutation]
            ),
            "rotation_relative": [
                _relative(
                    record["field"],
                    rotate_radial_gto_blocks(base_state.field_array(), rotation),
                )
                for record, rotation in zip(
                    rotations,
                    symmetry_panel_rotations(self.molecule.molecule_id),
                    strict=True,
                )
            ],
        }
        field_errors["maximum_relative"] = max(
            field_errors["translation_relative"],
            field_errors["permutation_relative"],
            *field_errors["rotation_relative"],
        )
        field_errors["gate_passed"] = bool(
            field_errors["maximum_relative"] <= FIELD_COVARIANCE_RELATIVE_TOLERANCE
        )
        all_gates = bool(
            base_state.converged
            and rigid["all_gates_passed"]
            and field_errors["gate_passed"]
            and cold_warm["gate_passed"]
            and scalar_identity["gate_passed"]
        )
        return {
            "molecule_id": self.molecule.molecule_id,
            "source_record_id": self.molecule.source_record_id,
            "chemical_formula": self.molecule.chemical_formula,
            "scope_tags": list(self.molecule.scope_tags),
            "variant": PES_CARTESIAN_PANEL_VARIANT,
            "geometry_sha256": geometry_sha256(atoms),
            "atomic_numbers": atoms.numbers.tolist(),
            "positions_A": atoms.positions.tolist(),
            "identity": identity_record(
                self.model,
                self.continuum,
                self.equation,
                self.scalar,
            ),
            "base": base,
            "cold_warm": cold_warm,
            "scalar_identity": scalar_identity,
            "rigid_symmetry": rigid,
            "field_covariance": field_errors,
            "_raw_rigid_records": {
                "translation": translation,
                "permutation": permutation_record,
                "rotations": rotations,
            },
            "all_gates_passed": all_gates,
        }


def main() -> None:
    args = _parse_args()
    if (
        type(args.molecule_start) is not int
        or type(args.molecule_stop) is not int
        or not 0 <= args.molecule_start < args.molecule_stop <= PES_PANEL_MOLECULE_COUNT
    ):
        raise ValueError("Molecule shard must satisfy 0 <= start < stop <= 20.")
    if type(args.max_iterations) is not int or args.max_iterations < 1:
        raise ValueError("--max-iterations must be a positive integer.")
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    started = time.perf_counter()
    import torch

    determinism = _configure_determinism(torch, args.device)
    molecules = load_pes_panel()[args.molecule_start : args.molecule_stop]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = build_model(checkpoint, args.device)
        records = []
        for molecule in molecules:
            records.append(_RigidRunner(molecule, model, args.max_iterations).run())
            print(
                f"completed operational harmonic rigid molecule={molecule.molecule_id}",
                file=sys.stderr,
                flush=True,
            )
    contract = {
        "contract_version": CONTRACT_VERSION,
        "asset_sha256": PES_PANEL_ASSET_SHA256,
        "asset_path": PANEL_ASSET_PATH,
        "profile_id": PROFILE_ID,
        "scalar_id": SCALAR_ID,
        "variant": PES_CARTESIAN_PANEL_VARIANT,
        "molecule_count": PES_PANEL_MOLECULE_COUNT,
        "shard_start": args.molecule_start,
        "shard_stop": args.molecule_stop,
        "shard_molecule_ids": [item.molecule_id for item in molecules],
        "rotation_count": SYMMETRY_PANEL_ROTATION_COUNT,
        "translation_A": list(SYMMETRY_PANEL_TRANSLATION_A),
        "root": {
            "method": "anderson",
            "tolerance": PRIMAL_TOLERANCE,
            "max_iterations": args.max_iterations,
            "damping": ROOT_DAMPING,
            "history": ROOT_HISTORY,
        },
        "field_covariance_relative_tolerance": (FIELD_COVARIANCE_RELATIVE_TOLERANCE),
        "scalar_identity_absolute_tolerance_eV": (
            SCALAR_IDENTITY_ABSOLUTE_TOLERANCE_EV
        ),
        "missing_radial_block_absolute_tolerance": (
            MISSING_RADIAL_BLOCK_ABSOLUTE_TOLERANCE
        ),
        "harmonic_configuration": {
            "surface_lmax": SURFACE_LMAX,
            "exposure_lmax": EXPOSURE_LMAX,
            "transition_width_A2": TRANSITION_WIDTH_ANGSTROM2,
            "exposure_radial_quadrature_order": (EXPOSURE_RADIAL_QUADRATURE_ORDER),
            "source_radial_quadrature_order": SOURCE_RADIAL_QUADRATURE_ORDER,
            "green_radial_quadrature_order": GREEN_RADIAL_QUADRATURE_ORDER,
            "laboratory_fixed_surface_grid": False,
        },
    }
    decision = {
        "all_shard_gates_passed": all(item["all_gates_passed"] for item in records),
        "legacy_laboratory_grid_rehabilitated": False,
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "tier_v_admitted": False,
    }
    measurements = {
        "contract": contract,
        "molecules": records,
        "decision": decision,
    }
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    source_hashes = committed_source_hashes(
        repository, (*source_paths, PANEL_ASSET_PATH)
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": (
            "rigid-panel-shard-passed-not-admitted"
            if decision["all_shard_gates_passed"]
            else "rigid-panel-shard-failed-not-admitted"
        ),
        "claim_boundary": (
            "A source-bound rigid translation/permutation/rotation shard for one "
            "disabled operational scalar. It is not a closed-loop, Cartesian-FD, "
            "solvation-accuracy, PES/domain, Hessian/FREQ/TS/HVP/NVE/MD, or "
            "capability-admission result."
        ),
        "capabilities": NO_CAPABILITIES,
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
        "dtype": model.dtype,
        **measurements,
        "measurement_sha256": canonical_json_sha256(measurements),
        "warnings": _warning_records(captured),
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_OPERATIONAL_ANALYTIC_HARMONIC_RIGID_PANEL_SHARD="
        + json.dumps(
            {
                "artifact": artifact,
                "measurement_sha256": payload["measurement_sha256"],
                "decision": decision,
                "capabilities": NO_CAPABILITIES,
                "runtime_seconds": payload["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    if not decision["all_shard_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
