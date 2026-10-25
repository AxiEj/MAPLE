#!/usr/bin/env python3
"""Validate the independent OBC-II provider against sealed Stage-0 oracles.

This runner is label-blind.  It replays the exact sealed AM1-BCC-526 ACE/NONE
energies, the frozen 12-case/48-cell E/F and radius matrix, the preregistered
solvent Hessian/HVP matrix, and a guarded no-OpenMM subprocess.  OpenMM is used
only as a pinned Reference force-FD oracle for Hessian validation; it is never a
runtime fallback for the independent provider.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any, Mapping
from unittest.mock import patch
from uuid import uuid4

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_core as core  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

BASE_HEAD = "e4044219"
RECORDED_DATE = "2026-10-25"
RUNNER_REL = "docs/implicit-solvation/benchmarks/validate_independent_obc2_default.py"
TEST_REL = "tests/solvation/test_independent_obc2_default_validation.py"
OUTPUT_REL = (
    ".omx/benchmarks/route1-aimnet2025-performance-20261025/obc-ani/"
    "candidate-validation.json"
)
CANDIDATE_CELL_EVIDENCE_REL = (
    ".omx/benchmarks/route1-aimnet2025-performance-20261025/obc-ani/"
    "candidate-exact-cell.json"
)
CANDIDATE_CELL_HARNESS_REL = "tests/solvation/run_independent_obc2_candidate_cell.py"
QUALIFICATION_SOURCE_EXCLUDES = frozenset(
    {
        "maple/function/calculator/_independent_obc2_qualified_cell.py",
        "maple/function/calculator/data/independent_obc2_analytic_qualification.json",
        "maple/function/calculator/data/independent_obc2_analytic_qualification_v2.json",
        "maple/function/calculator/data/independent_obc2_analytic_qualification_v3.json",
    }
)
ANI2X_CHECKPOINT_SHA256 = (
    "c8c41e89d5ccbe7b5c9f46ce4a262b913ad0c428a3a3742cc6e83ac10d4d43aa"
)
HISTORICAL_REVALIDATION_REL = (
    ".omx/benchmarks/route1-independent-recheck-20260930T145212Z/accuracy-source/"
    "docs/implicit-solvation/benchmarks/route1-independent-obc2-validation-2026-09-30.json"
)
HISTORICAL_REVALIDATION_FILE_SHA256 = (
    "ed1c103517ca8ff71084866414db95cceed7018dc3927695d8a4e11accfb44fe"
)
HISTORICAL_REVALIDATION_CONTENT_SHA256 = (
    "871b4fbd58525bd6ff2a30478e41d30d7a41034d39fbc6a858a9093a6753b43d"
)
STAGE0_GENERATOR_REL = (
    "docs/implicit-solvation/benchmarks/" "freeze_obc2_am1bcc_526_reference.py"
)
STAGE0_MAIN_REL = (
    "docs/implicit-solvation/benchmarks/"
    "route1-obc2-am1bcc-526-reference-2026-09-27.json"
)
STAGE0_PORTABLE_REL = "tests/solvation/data/obc2_openmm_8_5_2_reference_v1.json"
CORPUS_ROOT_REL = ".omx/benchmarks/neutral-water-freesolv-route1-20260723"

STAGE0_GENERATOR_SHA256 = (
    "252eeea7b3220a322b99ccf7c2d0ef17577d29923ff01f4a3e31e44cfffddd8e"
)
STAGE0_MAIN_FILE_SHA256 = (
    "4ced4b9dc665b2016a979428ca95576063765bbab3e32e7a55c4ecf05e39396b"
)
STAGE0_MAIN_CONTENT_SHA256 = (
    "9e3a5faf9518576e7db50e18f1e756b6ec1c97f881bd321ff5919c748d85524c"
)
STAGE0_PORTABLE_FILE_SHA256 = (
    "2bb109734adf1b8a21a371cbfcc15dbc4b718f6d6cc00b1072f390733a087275"
)
STAGE0_PORTABLE_CONTENT_SHA256 = (
    "37a7e7608f22e6bb174d37223b84338bae50331fb8a433282ea071c7a7a02eb4"
)
STAGE0_SOURCE_FREEZE_SHA256 = (
    "35ff044b4bc1a44d81fd27768a021c8e675281234c3e8674b90a4da262829641"
)
OPENMM_VERSION = "8.5.2"
OPENMM_SOURCE_SHA256 = (
    "f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a"
)

PROFILE_ORDER = ("ace", "none")
GEOMETRY_ORDER = ("base", "displaced")
PORTABLE_CASES = (
    "benzene",
    "methanol",
    "bromobenzene",
    "methyl-hexanoate",
    "iodobenzene",
    "glucose",
    "4-nitroaniline",
    "dimethyl-sulfide",
    "fluorobenzene",
    "nitralin",
    "aniline",
    "chlorobenzene",
)

# Frozen before independent-provider calculations: nitralin is the unique maximum
# atom-count member (42) of the sealed 12-case corpus.
FULL_H_CASES = ("water", "methanol", "methyl-hexanoate", "nitralin")
FULL_H_SELECTION_RULE = (
    "water+methanol+methyl-hexanoate+max-atom-count-then-lexicographic"
)
WATER_DISPLACEMENT_SEED = 20260920
DISPLACEMENT_SCALE_ANGSTROM = 2.0e-4
HVP_BASE_SEED = 2026092701
HVP_DIRECTION_COUNT = 3

# Approved D64 force-FD ladder, ordered coarse -> fine.  The coarse point is
# always logged, but admission is fixed to both predeclared finest points plus
# their pairwise-matrix consistency.  There is no best-step selection and no
# post-result monotonicity gate.
SOLVENT_FD_STEPS_ANGSTROM = (0.001, 0.0005, 0.00025)
SOLVENT_FD_FINEST_STEPS_ANGSTROM = (0.0005, 0.00025)

GATES = {
    "energy_abs_hartree": 2.0e-10,
    "energy_relative": 1.0e-9,
    "component_abs_hartree": 2.0e-10,
    "force_max_hartree_per_angstrom": 2.0e-7,
    "force_rms_hartree_per_angstrom": 5.0e-8,
    "radius_parameter_abs": 1.0e-15,
    "hessian_asymmetry_hartree_per_angstrom2": 1.0e-10,
    "hvp_atol_hartree_per_angstrom2": 1.0e-9,
    "hvp_rtol": 1.0e-9,
    "translation_absolute_hartree_per_angstrom2": 1.0e-8,
    "translation_relative_frobenius": 1.0e-8,
    "hessian_fd_rms_hartree_per_angstrom2": 5.0e-6,
    "hessian_fd_max_hartree_per_angstrom2": 2.0e-5,
    "bundle_energy_abs_hartree": 1.0e-12,
    "bundle_force_max_hartree_per_angstrom": 1.0e-12,
}

KCAL_PER_HARTREE = 2625.4996394799 / 4.184

WATER_MOL2 = """@<TRIPOS>MOLECULE
WATER
 3 2 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 O1          0.000000    0.000000    0.000000 O.3       1 WAT       -0.834000
      2 H1          0.957200    0.000000    0.000000 H         1 WAT        0.417000
      3 H2         -0.239987    0.927297    0.000000 H         1 WAT        0.417000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
@<TRIPOS>SUBSTRUCTURE
     1 WAT         1 TEMP              0 ****  ****    0 ROOT
"""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _canonical_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    descriptor = {
        "dtype": "float64",
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return core.sha256_bytes(core.canonical_json_bytes(descriptor))


def _load_self_hashed(path: Path, *, file_sha: str, content_sha: str) -> dict[str, Any]:
    if core.sha256_file(path) != file_sha:
        raise ValueError(f"Sealed file SHA changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != content_sha:
        raise ValueError(f"Sealed content SHA field changed: {path}")
    if core.artifact_content_sha256(value) != content_sha:
        raise ValueError(f"Sealed content hash does not replay: {path}")
    return value


def load_stage0() -> tuple[dict[str, Any], dict[str, Any]]:
    if core.sha256_file(ROOT / STAGE0_GENERATOR_REL) != STAGE0_GENERATOR_SHA256:
        raise ValueError("Stage-0 generator bytes changed")
    main = _load_self_hashed(
        ROOT / STAGE0_MAIN_REL,
        file_sha=STAGE0_MAIN_FILE_SHA256,
        content_sha=STAGE0_MAIN_CONTENT_SHA256,
    )
    portable = _load_self_hashed(
        ROOT / STAGE0_PORTABLE_REL,
        file_sha=STAGE0_PORTABLE_FILE_SHA256,
        content_sha=STAGE0_PORTABLE_CONTENT_SHA256,
    )
    if main.get("record_count") != 1052 or main.get("case_count") != 526:
        raise ValueError("Stage-0 526 oracle matrix is incomplete")
    if portable.get("comparison_count") != 48 or portable.get("case_count") != 12:
        raise ValueError("Stage-0 portable E/F matrix is incomplete")
    if main.get("source_freeze", {}).get("source_freeze_sha256") != (
        STAGE0_SOURCE_FREEZE_SHA256
    ):
        raise ValueError("Stage-0 source-freeze identity changed")
    if main.get("semantic_decoder_guard", {}).get("runtime_violation_count") != 0:
        raise ValueError("Stage-0 oracle reports a semantic decoder violation")
    return main, portable


def select_full_h_cases(portable: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in portable["records"]:
        counts.setdefault(str(record["case"]), int(record["atom_count"]))
        if counts[str(record["case"])] != int(record["atom_count"]):
            raise ValueError("Portable case atom count changes across records")
    maximum = max(counts.values())
    selected_crowded = sorted(
        case for case, count in counts.items() if count == maximum
    )[0]
    selected = ("water", "methanol", "methyl-hexanoate", selected_crowded)
    if selected != FULL_H_CASES or selected_crowded != "nitralin" or maximum != 42:
        raise ValueError("Preregistered largest/crowded full-H selection changed")
    return {
        "rule": FULL_H_SELECTION_RULE,
        "portable_atom_counts": dict(sorted(counts.items())),
        "maximum_atom_count": maximum,
        "selected_largest_case": selected_crowded,
        "selected_cases": list(selected),
    }


def normalized_direction(
    *, case_index: int, geometry_index: int, direction_index: int, size: int
) -> tuple[int, np.ndarray]:
    seed = HVP_BASE_SEED + 100 * case_index + 10 * geometry_index + direction_index
    vector = np.random.default_rng(seed).normal(size=size)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Frozen HVP direction has invalid norm")
    vector = np.asarray(vector / norm, dtype=np.float64)
    return seed, vector


def _source_paths() -> list[Path]:
    output = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=ROOT,
    )
    relatives = sorted(
        relative.decode("utf-8")
        for relative in output.split(b"\0")
        if relative
    )
    paths = []
    for relative in relatives:
        if relative in QUALIFICATION_SOURCE_EXCLUDES:
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Qualification source is not a file: {relative}")
        paths.append(path)
    return paths


def source_snapshot(*, require_clean: bool = True) -> dict[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if require_clean and status:
        raise RuntimeError(
            "Formal qualification requires a clean committed tree; dirty paths: "
            + ", ".join(status)
        )
    files = {
        path.relative_to(ROOT).as_posix(): core.sha256_file(path)
        for path in _source_paths()
    }
    files_aggregate = core.sha256_bytes(core.canonical_json_bytes(files))
    payload = {
        "git_head": _git("rev-parse", "HEAD"),
        "files_sha256": files,
        "files_aggregate_sha256": files_aggregate,
        "inventory_policy": "git-ls-files-cached-plus-nonignored-untracked-v1",
        "excluded_data_only_paths": sorted(QUALIFICATION_SOURCE_EXCLUDES),
        "worktree_status": status,
    }
    payload["aggregate_sha256"] = core.sha256_bytes(core.canonical_json_bytes(payload))
    return payload


def validate_current_source_binding(frozen: Mapping[str, Any]) -> None:
    """Permit committed sealing data, not source drift or unrelated lineage."""
    current = source_snapshot(require_clean=True)
    if (
        current["files_sha256"] != frozen.get("files_sha256")
        or current["files_aggregate_sha256"] != frozen.get("files_aggregate_sha256")
    ):
        raise ValueError("Current source inventory does not match frozen qualification source")
    candidate_head = frozen.get("git_head")
    if (
        not isinstance(candidate_head, str) or len(candidate_head) != 40
        or any(character not in "0123456789abcdef" for character in candidate_head)
    ):
        raise ValueError("Frozen qualification candidate Git SHA is malformed")
    try:
        _git("merge-base", "--is-ancestor", candidate_head, current["git_head"])
    except subprocess.CalledProcessError as exc:
        raise ValueError("Qualification candidate is not an ancestor of current HEAD") from exc
    changed_paths = {
        path for path in _git(
            "log", "--format=", "--name-only", "--diff-merges=first-parent",
            f"{candidate_head}..{current['git_head']}",
        ).splitlines() if path
    }
    if changed_paths - QUALIFICATION_SOURCE_EXCLUDES:
        raise ValueError("Post-qualification commits must contain only excluded data-only paths")


def _provider_class():
    from maple.function.calculator.extra_correction.implicit.independent_obc2 import (
        IndependentOBC2,
    )

    return IndependentOBC2


def _atoms_from_portable(record: Mapping[str, Any]):
    path = ROOT / str(record["source_mol2_relative_path"])
    if core.sha256_file(path) != record["source_mol2_sha256"]:
        raise ValueError(f"Portable MOL2 changed: {path}")
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    atoms.set_positions(np.asarray(record["coordinates_angstrom"], dtype=np.float64))
    charges = np.asarray(record["ordered_charges_e"], dtype=np.float64)
    atoms.set_initial_charges(charges)
    return atoms, charges


def _atoms_from_526(record: Mapping[str, Any]):
    path = ROOT / CORPUS_ROOT_REL / str(record["source_mol2_relative_path"])
    if core.sha256_file(path) != record["source_mol2_sha256"]:
        raise ValueError(f"526 MOL2 changed: {path}")
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    atoms.set_positions(np.asarray(record["coordinates_angstrom"], dtype=np.float64))
    charges = np.asarray(record["ordered_am1bcc_charges_e"], dtype=np.float64)
    atoms.set_initial_charges(charges)
    return atoms, charges


@contextmanager
def _water_geometries():
    with tempfile.TemporaryDirectory(prefix="maple-obc2-validation-") as temporary:
        path = Path(temporary) / "water.mol2"
        path.write_text(WATER_MOL2, encoding="utf-8")
        base = MOL2Reader(str(path), charge=0, mult=1)
        displaced = base.copy()
        displaced.positions += np.random.default_rng(WATER_DISPLACEMENT_SEED).normal(
            scale=DISPLACEMENT_SCALE_ANGSTROM, size=(len(base), 3)
        )
        yield {
            "base": (base, np.asarray(base.get_initial_charges(), dtype=np.float64)),
            "displaced": (
                displaced,
                np.asarray(base.get_initial_charges(), dtype=np.float64),
            ),
            "source_sha256": _sha256_text(WATER_MOL2),
        }


def _portable_index(portable: Mapping[str, Any]) -> dict[tuple[str, str, str], dict]:
    result = {}
    for row in portable["records"]:
        key = (str(row["case"]), str(row["nonpolar"]), str(row["geometry"]))
        if key in result:
            raise ValueError(f"Duplicate portable record: {key}")
        result[key] = row
    return result


def _parameter_deltas(provider, golden: Mapping[str, Any]) -> dict[str, float]:
    candidate = {
        "provider_parameters_radius_nm_screen": np.asarray(
            provider.radius_result.provider_parameters, dtype=np.float64
        ),
        "intrinsic_radii_angstrom": np.asarray(
            provider.radius_result.radii_angstrom, dtype=np.float64
        ),
        "screen_factors": np.asarray(
            provider.radius_result.provider_parameters[:, 1], dtype=np.float64
        ),
        "offset_radii_nm": np.asarray(
            provider.parameters.offset_radii_nm, dtype=np.float64
        ),
        "scaled_offset_radii_nm": np.asarray(
            provider.parameters.scaled_offset_radii_nm, dtype=np.float64
        ),
        "finalized_particle_charge_offset_scaled": np.column_stack(
            (
                np.asarray(provider.parameters.charges, dtype=np.float64),
                np.asarray(provider.parameters.offset_radii_nm, dtype=np.float64),
                np.asarray(
                    provider.parameters.scaled_offset_radii_nm, dtype=np.float64
                ),
            )
        ),
    }
    deltas = {}
    for name, values in candidate.items():
        expected = np.asarray(golden[name], dtype=np.float64)
        if values.shape != expected.shape:
            raise ValueError(f"Radius/parameter shape changed for {name}")
        deltas[name] = float(np.max(np.abs(values - expected), initial=0.0))
    return deltas


def _energy_metrics(candidate, golden: Mapping[str, Any]) -> dict[str, Any]:
    deltas = {
        "polar_abs_hartree": abs(
            float(candidate.components_hartree["polar"])
            - float(golden["polar_hartree"])
        ),
        "nonpolar_abs_hartree": abs(
            float(candidate.components_hartree["nonpolar"])
            - float(golden["nonpolar_hartree"])
        ),
        "total_abs_hartree": abs(
            float(candidate.energy_hartree) - float(golden["total_hartree"])
        ),
    }
    energy_limit = GATES["energy_abs_hartree"] + GATES["energy_relative"] * abs(
        float(golden["total_hartree"])
    )
    passed = (
        deltas["total_abs_hartree"] <= energy_limit
        and deltas["polar_abs_hartree"] <= GATES["component_abs_hartree"]
        and deltas["nonpolar_abs_hartree"] <= GATES["component_abs_hartree"]
    )
    return {**deltas, "total_limit_hartree": energy_limit, "passed": bool(passed)}


def replay_526(stage0: Mapping[str, Any]) -> dict[str, Any]:
    Provider = _provider_class()
    rows = []
    total_drifts = []
    maximum_parameter_delta = 0.0
    for index, golden in enumerate(stage0["records"], start=1):
        atoms, charges = _atoms_from_526(golden)
        provider = Provider(
            atoms,
            charges,
            nonpolar=str(golden["nonpolar"]),
            device="cpu",
        )
        candidate = provider.evaluate(atoms, need_forces=False)
        energy = _energy_metrics(candidate, golden)
        parameter_deltas = _parameter_deltas(provider, golden["parameters"])
        parameter_max = max(parameter_deltas.values())
        maximum_parameter_delta = max(maximum_parameter_delta, parameter_max)
        topology_match = (
            provider.topology.identity_sha256
            == golden["topology"]["topology_sha256"]
        )
        charge_match = (
            provider.provenance.get("ordered_charge_sha256")
            == golden["charge_vector_sha256"]
        )
        passed = (
            energy["passed"]
            and parameter_max <= GATES["radius_parameter_abs"]
            and topology_match
            and charge_match
        )
        total_drifts.append(energy["total_abs_hartree"])
        rows.append(
            {
                "compound_id": golden["compound_id"],
                "nonpolar": golden["nonpolar"],
                "input_identity_sha256": golden["input_identity_sha256"],
                "topology_sha256": provider.topology.identity_sha256,
                "topology_match": topology_match,
                "charge_identity_match": charge_match,
                "golden_record_content_sha256": golden["record_content_sha256"],
                "energy": energy,
                "parameter_max_abs": parameter_max,
                "parameter_deltas": parameter_deltas,
                "provider_provenance": provider.provenance,
                "passed": bool(passed),
            }
        )
        if index == 1 or index % 50 == 0 or index == len(stage0["records"]):
            print(
                f"[validation] 526 replay {index}/{len(stage0['records'])}", flush=True
            )
    drift = np.asarray(total_drifts, dtype=np.float64)
    mean_abs = float(np.mean(drift))
    section = {
        "case_count": 526,
        "profile_count": 2,
        "record_count": len(rows),
        "maximum_absolute_prediction_drift_hartree": float(np.max(drift)),
        "mean_absolute_prediction_drift_hartree": mean_abs,
        # Triangle inequality bound using prediction drift only; no labels read.
        "mae_drift_upper_bound_hartree": mean_abs,
        "mae_drift_upper_bound_kcal_mol": mean_abs * KCAL_PER_HARTREE,
        "maximum_parameter_abs": maximum_parameter_delta,
        "records": rows,
    }
    section["passed"] = bool(all(row["passed"] for row in rows))
    return section


def replay_portable_48(portable: Mapping[str, Any]) -> dict[str, Any]:
    Provider = _provider_class()
    rows = []
    for golden in portable["records"]:
        atoms, charges = _atoms_from_portable(golden)
        provider = Provider(
            atoms,
            charges,
            nonpolar=str(golden["nonpolar"]),
            device="cpu",
        )
        candidate = provider.evaluate(atoms, need_forces=True)
        energy = _energy_metrics(candidate, golden)
        difference = np.asarray(
            candidate.forces_hartree_per_angstrom, dtype=np.float64
        ) - np.asarray(golden["forces_hartree_per_angstrom"], dtype=np.float64)
        force_max = float(np.max(np.abs(difference), initial=0.0))
        force_rms = float(np.sqrt(np.mean(difference * difference)))
        parameter_deltas = _parameter_deltas(provider, golden["parameters"])
        parameter_max = max(parameter_deltas.values())
        topology_match = (
            provider.topology.identity_sha256
            == golden["topology"]["topology_sha256"]
        )
        charge_match = (
            provider.provenance.get("ordered_charge_sha256")
            == golden["charge_vector_sha256"]
        )
        passed = (
            energy["passed"]
            and force_max <= GATES["force_max_hartree_per_angstrom"]
            and force_rms <= GATES["force_rms_hartree_per_angstrom"]
            and parameter_max <= GATES["radius_parameter_abs"]
            and topology_match
            and charge_match
        )
        rows.append(
            {
                "case": golden["case"],
                "geometry": golden["geometry"],
                "nonpolar": golden["nonpolar"],
                "golden_record_content_sha256": golden["record_content_sha256"],
                "topology_sha256": provider.topology.identity_sha256,
                "topology_match": topology_match,
                "charge_identity_match": charge_match,
                "energy": energy,
                "force_max_abs_hartree_per_angstrom": force_max,
                "force_rms_hartree_per_angstrom": force_rms,
                "parameter_max_abs": parameter_max,
                "parameter_deltas": parameter_deltas,
                "provider_provenance": provider.provenance,
                "passed": bool(passed),
            }
        )
    return {
        "case_count": 12,
        "comparison_count": len(rows),
        "maximum_total_energy_abs_hartree": max(
            row["energy"]["total_abs_hartree"] for row in rows
        ),
        "maximum_force_abs_hartree_per_angstrom": max(
            row["force_max_abs_hartree_per_angstrom"] for row in rows
        ),
        "maximum_force_rms_hartree_per_angstrom": max(
            row["force_rms_hartree_per_angstrom"] for row in rows
        ),
        "maximum_parameter_abs": max(row["parameter_max_abs"] for row in rows),
        "records": rows,
        "passed": bool(all(row["passed"] for row in rows)),
    }


def _full_h_inputs(portable: Mapping[str, Any]):
    index = _portable_index(portable)
    with _water_geometries() as water:
        for case in FULL_H_CASES:
            for profile in PROFILE_ORDER:
                for geometry in GEOMETRY_ORDER:
                    if case == "water":
                        atoms, charges = water[geometry]
                        source_identity = water["source_sha256"]
                    else:
                        golden = index[(case, profile, geometry)]
                        atoms, charges = _atoms_from_portable(golden)
                        source_identity = golden["record_content_sha256"]
                    yield case, profile, geometry, atoms, charges, source_identity


def _openmm_force_fd(provider, atoms, step: float) -> np.ndarray:
    size = 3 * len(atoms)
    matrix = np.empty((size, size), dtype=np.float64)
    for coordinate in range(size):
        atom, axis = divmod(coordinate, 3)
        plus = atoms.copy()
        minus = atoms.copy()
        plus.positions[atom, axis] += step
        minus.positions[atom, axis] -= step
        plus_force = provider.evaluate(
            plus, need_forces=True
        ).forces_hartree_per_angstrom.reshape(-1)
        minus_force = provider.evaluate(
            minus, need_forces=True
        ).forces_hartree_per_angstrom.reshape(-1)
        matrix[:, coordinate] = -(plus_force - minus_force) / (2.0 * step)
    return matrix


def live_openmm_oracle_identity() -> dict[str, Any]:
    from maple.function.calculator.extra_correction.implicit.openmm_compat import (
        require_verified_obc2_expression_source,
    )
    from openmm import Platform

    compatibility = require_verified_obc2_expression_source()
    if compatibility["source_sha256"] != OPENMM_SOURCE_SHA256:
        raise ImportError("Live OpenMM force-FD source identity changed")
    available = [
        Platform.getPlatform(index).getName()
        for index in range(Platform.getNumPlatforms())
    ]
    if "Reference" not in available:
        raise RuntimeError("Live OpenMM Reference platform is unavailable")
    return {
        "version": OPENMM_VERSION,
        "module": "openmm.app.internal.customgbforces",
        "source_sha256": compatibility["source_sha256"],
        "expression_profile": compatibility["expression_profile"],
        "platform": "Reference",
        "available_platforms": available,
    }


def assess_fd_ladder(
    analytic: np.ndarray, matrices: Mapping[float, np.ndarray]
) -> dict[str, Any]:
    records = []
    for step in SOLVENT_FD_STEPS_ANGSTROM:
        difference = np.asarray(analytic) - np.asarray(matrices[step])
        rms = float(np.sqrt(np.mean(difference * difference)))
        maximum = float(np.max(np.abs(difference), initial=0.0))
        records.append(
            {
                "step_angstrom": step,
                "rms_hartree_per_angstrom2": rms,
                "max_hartree_per_angstrom2": maximum,
                "matrix_sha256": _canonical_array_sha256(matrices[step]),
                "passed": bool(
                    rms <= GATES["hessian_fd_rms_hartree_per_angstrom2"]
                    and maximum <= GATES["hessian_fd_max_hartree_per_angstrom2"]
                ),
            }
        )
    by_step = {row["step_angstrom"]: row for row in records}
    finest_pass = all(
        by_step[step]["passed"] for step in SOLVENT_FD_FINEST_STEPS_ANGSTROM
    )
    all_pass = all(row["passed"] for row in records)
    fine_difference = matrices[0.00025] - matrices[0.0005]
    fine_rms = float(np.sqrt(np.mean(fine_difference * fine_difference)))
    fine_max = float(np.max(np.abs(fine_difference), initial=0.0))
    fine_pair_pass = (
        fine_rms <= 2.0 * GATES["hessian_fd_rms_hartree_per_angstrom2"]
        and fine_max <= 2.0 * GATES["hessian_fd_max_hartree_per_angstrom2"]
    )
    return {
        "step_order": list(SOLVENT_FD_STEPS_ANGSTROM),
        "records": records,
        "coarse_pass_observed": bool(by_step[0.001]["passed"]),
        "all_three_pass_observed": bool(all_pass),
        "both_finest_pass": bool(finest_pass),
        "finest_pair_difference_rms_hartree_per_angstrom2": fine_rms,
        "finest_pair_difference_max_hartree_per_angstrom2": fine_max,
        "finest_pair_pass": bool(fine_pair_pass),
        "passed": bool(finest_pass and fine_pair_pass),
    }


def full_hessian_matrix(portable: Mapping[str, Any]):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    if importlib.metadata.version("openmm") != OPENMM_VERSION:
        raise ImportError("Live force-FD oracle requires pinned OpenMM 8.5.2")
    oracle_identity = live_openmm_oracle_identity()
    Provider = _provider_class()
    records = []
    matrices: dict[tuple[str, str, str], np.ndarray] = {}
    for case, profile, geometry, atoms, charges, source_identity in _full_h_inputs(
        portable
    ):
        independent = Provider(atoms, charges, nonpolar=profile, device="cpu")
        raw_hessian = np.asarray(independent.hessian(atoms), dtype=np.float64)
        expected_shape = (3 * len(atoms),) * 2
        if raw_hessian.shape != expected_shape or not np.isfinite(raw_hessian).all():
            raise ValueError(
                f"Invalid independent Hessian for {case}/{profile}/{geometry}"
            )
        matrices[(case, profile, geometry)] = raw_hessian.copy()
        asymmetry = float(np.max(np.abs(raw_hessian - raw_hessian.T), initial=0.0))
        hessian_norm = float(np.linalg.norm(raw_hessian))
        translation_norms = []
        for axis in range(3):
            translation = np.zeros(3 * len(atoms), dtype=np.float64)
            translation[axis::3] = 1.0
            translation_norms.append(float(np.linalg.norm(raw_hessian @ translation)))
        translation_limit = max(
            GATES["translation_absolute_hartree_per_angstrom2"],
            GATES["translation_relative_frobenius"] * hessian_norm,
        )

        oracle = OpenMMGB(
            atoms,
            charges,
            model="obc2",
            nonpolar=profile,
            platform="Reference",
        )
        if oracle.platform != "Reference":
            raise RuntimeError("Live Hessian FD oracle did not use Reference")
        fd_matrices = {
            step: _openmm_force_fd(oracle, atoms, step)
            for step in SOLVENT_FD_STEPS_ANGSTROM
        }
        ladder = assess_fd_ladder(raw_hessian, fd_matrices)
        passed = (
            asymmetry <= GATES["hessian_asymmetry_hartree_per_angstrom2"]
            and max(translation_norms) <= translation_limit
            and ladder["passed"]
        )
        records.append(
            {
                "case": case,
                "nonpolar": profile,
                "geometry": geometry,
                "atom_count": len(atoms),
                "source_identity_sha256": source_identity,
                "coordinate_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(
                        np.asarray(atoms.get_positions(), dtype=np.float64).tolist()
                    )
                ),
                "topology_sha256": independent.topology.identity_sha256,
                "raw_hessian_shape": list(raw_hessian.shape),
                "raw_hessian_sha256": _canonical_array_sha256(raw_hessian),
                "raw_hessian_max_abs": float(np.max(np.abs(raw_hessian), initial=0.0)),
                "raw_hessian_frobenius": hessian_norm,
                "raw_hessian_asymmetry_max": asymmetry,
                "translation_norms": translation_norms,
                "translation_limit": translation_limit,
                "fd_ladder": ladder,
                "provider_provenance": independent.provenance,
                "passed": bool(passed),
            }
        )
        print(f"[validation] full H {case}/{profile}/{geometry}", flush=True)
    section = {
        "selection": select_full_h_cases(portable),
        "live_force_fd_oracle": oracle_identity,
        "water_displacement_seed": WATER_DISPLACEMENT_SEED,
        "displacement_scale_angstrom": DISPLACEMENT_SCALE_ANGSTROM,
        "record_count": len(records),
        "records": records,
        "passed": bool(len(records) == 16 and all(row["passed"] for row in records)),
    }
    return section, matrices


def _forbidden_dense_hessian(*_args, **_kwargs):
    raise RuntimeError("direct HVP attempted to materialize a dense Hessian")


def hvp_matrix(
    portable: Mapping[str, Any],
    full_hessians: Mapping[tuple[str, str, str], np.ndarray],
) -> dict[str, Any]:
    Provider = _provider_class()
    index = _portable_index(portable)
    records = []
    for case_index, case in enumerate(PORTABLE_CASES):
        for geometry_index, geometry in enumerate(GEOMETRY_ORDER):
            for profile in PROFILE_ORDER:
                golden = index[(case, profile, geometry)]
                atoms, charges = _atoms_from_portable(golden)
                provider = Provider(atoms, charges, nonpolar=profile, device="cpu")
                ordinary = provider.evaluate(atoms, need_forces=True)
                for direction_index in range(HVP_DIRECTION_COUNT):
                    seed, direction = normalized_direction(
                        case_index=case_index,
                        geometry_index=geometry_index,
                        direction_index=direction_index,
                        size=3 * len(atoms),
                    )
                    with (
                        patch.object(
                            provider.kernel,
                            "hessian",
                            side_effect=_forbidden_dense_hessian,
                        ),
                        patch.object(
                            provider, "hessian", side_effect=_forbidden_dense_hessian
                        ),
                    ):
                        directional = provider.directional_derivatives(atoms, direction)
                    hvp = np.asarray(
                        directional.hvp_hartree_per_angstrom2, dtype=np.float64
                    )
                    if hvp.shape != direction.shape or not np.isfinite(hvp).all():
                        raise ValueError(f"Invalid HVP for {case}/{profile}/{geometry}")
                    energy_delta = abs(
                        float(directional.energy_hartree)
                        - float(ordinary.energy_hartree)
                    )
                    force_delta = float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    directional.forces_hartree_per_angstrom,
                                    dtype=np.float64,
                                )
                                - np.asarray(
                                    ordinary.forces_hartree_per_angstrom,
                                    dtype=np.float64,
                                )
                            ),
                            initial=0.0,
                        )
                    )
                    hessian_key = (case, profile, geometry)
                    h_product = None
                    hvp_max = None
                    hvp_scaled = None
                    hvp_pass = True
                    if hessian_key in full_hessians:
                        h_product = np.asarray(full_hessians[hessian_key]) @ direction
                        difference = hvp - h_product
                        hvp_max = float(np.max(np.abs(difference), initial=0.0))
                        scale = np.maximum(
                            np.abs(h_product), GATES["hvp_atol_hartree_per_angstrom2"]
                        )
                        hvp_scaled = float(
                            np.max(np.abs(difference) / scale, initial=0.0)
                        )
                        hvp_pass = bool(
                            np.allclose(
                                hvp,
                                h_product,
                                rtol=GATES["hvp_rtol"],
                                atol=GATES["hvp_atol_hartree_per_angstrom2"],
                            )
                        )
                    passed = (
                        energy_delta <= GATES["bundle_energy_abs_hartree"]
                        and force_delta
                        <= GATES["bundle_force_max_hartree_per_angstrom"]
                        and hvp_pass
                    )
                    records.append(
                        {
                            "case": case,
                            "nonpolar": profile,
                            "geometry": geometry,
                            "direction_index": direction_index,
                            "seed": seed,
                            "direction": direction.tolist(),
                            "direction_sha256": _canonical_array_sha256(direction),
                            "direction_l2_norm": float(np.linalg.norm(direction)),
                            "hvp": hvp.tolist(),
                            "hvp_sha256": _canonical_array_sha256(hvp),
                            "same_scalar_energy_abs_hartree": energy_delta,
                            "same_scalar_force_max_hartree_per_angstrom": force_delta,
                            "hvp_vs_raw_hessian_max": hvp_max,
                            "hvp_vs_raw_hessian_scaled_max": hvp_scaled,
                            "raw_hessian_product": (
                                None if h_product is None else h_product.tolist()
                            ),
                            "raw_hessian_product_sha256": (
                                None
                                if h_product is None
                                else _canonical_array_sha256(h_product)
                            ),
                            "derivative_provenance": directional.provenance,
                            "dense_hessian_forbidden_during_call": True,
                            "passed": bool(passed),
                        }
                    )
        print(f"[validation] HVP {case_index + 1}/{len(PORTABLE_CASES)}", flush=True)
    return {
        "base_seed": HVP_BASE_SEED,
        "direction_count_per_case_geometry_profile": HVP_DIRECTION_COUNT,
        "record_count": len(records),
        "records": records,
        "passed": bool(len(records) == 144 and all(row["passed"] for row in records)),
    }


def _no_openmm_script() -> str:
    return r"""
import hashlib
import importlib.abc
import importlib.metadata
import json
import socket
import sys

import numpy as np

ROOT, MOL2 = sys.argv[1:3]
sys.path.insert(0, ROOT)

class BlockOpenMM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openmm" or fullname.startswith("openmm."):
            raise AssertionError(f"forbidden OpenMM import: {fullname}")
        return None

original_version = importlib.metadata.version
original_distribution = importlib.metadata.distribution

def guarded_version(name):
    if str(name).lower() == "openmm":
        raise AssertionError("forbidden OpenMM version lookup")
    return original_version(name)

def guarded_distribution(name):
    if str(name).lower() == "openmm":
        raise AssertionError("forbidden OpenMM distribution lookup")
    return original_distribution(name)

def denied_network(*args, **kwargs):
    raise AssertionError("network access is forbidden")

importlib.metadata.version = guarded_version
importlib.metadata.distribution = guarded_distribution
sys.meta_path.insert(0, BlockOpenMM())
socket.socket.connect = denied_network
socket.create_connection = denied_network

from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.calculator.extra_correction.implicit.independent_obc2 import IndependentOBC2

atoms = MOL2Reader(MOL2, charge=0, mult=1)
provider = IndependentOBC2(atoms, atoms.get_initial_charges(), nonpolar="ace", device="cpu")
result = provider.evaluate(atoms, need_forces=True)
hessian = np.asarray(provider.hessian(atoms), dtype=np.float64)
direction = np.arange(1, 3*len(atoms)+1, dtype=np.float64)
direction /= np.linalg.norm(direction)
directional = provider.directional_derivatives(atoms, direction)
hvp = np.asarray(directional.hvp_hartree_per_angstrom2, dtype=np.float64)
assert np.allclose(hvp, hessian @ direction, rtol=1e-9, atol=1e-9)
assert not any(name == "openmm" or name.startswith("openmm.") for name in sys.modules)

def digest(array):
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()

print(json.dumps({
    "provider": provider.provenance["provider"],
    "energy_hartree": float(result.energy_hartree),
    "force_sha256": digest(result.forces_hartree_per_angstrom),
    "hessian_sha256": digest(hessian),
    "hvp_sha256": digest(hvp),
    "openmm_loaded": False,
    "ok": True,
}, sort_keys=True))
""".strip()


def guarded_no_openmm_evaluation() -> dict[str, Any]:
    script = _no_openmm_script()
    with tempfile.TemporaryDirectory(prefix="maple-obc2-no-openmm-") as temporary:
        mol2 = Path(temporary) / "water.mol2"
        mol2.write_text(WATER_MOL2, encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            MAPLE_OFFLINE="1",
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, str(ROOT), str(mol2)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "Guarded no-OpenMM subprocess failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result.get("ok") is not True or result.get("openmm_loaded") is not False:
        raise RuntimeError("Guarded no-OpenMM subprocess did not prove independence")
    return {
        "script_sha256": _sha256_text(script),
        "python_executable": sys.executable,
        "network_disabled": True,
        "import_blocker": True,
        "metadata_blocker": True,
        "result": result,
        "passed": True,
    }


@contextmanager
def _network_denied():
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def denied(*_args, **_kwargs):
        raise RuntimeError("Validation network access is forbidden")

    socket.socket.connect = denied  # type: ignore[method-assign]
    socket.create_connection = denied  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create  # type: ignore[assignment]


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("ase", "numpy", "openmm", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "device": "cpu",
        "torch_threads": 1,
    }


def _build_artifact() -> dict[str, Any]:
    stage0, portable = load_stage0()
    selection = select_full_h_cases(portable)
    before = source_snapshot(require_clean=True)
    candidate_cell = json.loads(
        (ROOT / CANDIDATE_CELL_EVIDENCE_REL).read_text(encoding="utf-8")
    )
    validate_candidate_cell_evidence(
        candidate_cell,
        expected_source_snapshot=before,
    )
    if candidate_cell["candidate_git_sha"] != before["git_head"]:
        raise ValueError("Candidate exact-cell evidence Git SHA changed")
    import torch

    torch.set_num_threads(1)
    with _network_denied():
        independence = guarded_no_openmm_evaluation()
        replay = replay_526(stage0)
        parity = replay_portable_48(portable)
        full_h, matrices = full_hessian_matrix(portable)
        hvp = hvp_matrix(portable, matrices)
    after = source_snapshot(require_clean=True)
    if before != after:
        raise RuntimeError("Validation source tree changed during execution")
    passed = all(
        section["passed"] for section in (independence, replay, parity, full_h, hvp)
    ) and candidate_cell["passed"]
    qualification_gates = {
        "sealed_oracle_1052_energy": _count_gate(replay, 1052),
        "portable_oracle_48_energy_force": _count_gate(parity, 48),
        "analytic_hessian_16_fd": {
            **_count_gate(full_h, 16),
            "steps_angstrom": list(SOLVENT_FD_STEPS_ANGSTROM),
        },
        "direct_hvp_144": {
            **_count_gate(hvp, 144),
            "dense_hessian_forbidden": all(
                row["dense_hessian_forbidden_during_call"] for row in hvp["records"]
            ),
        },
        "no_openmm_candidate_runtime": {
            "expected_count": 1,
            "passed_count": int(independence["passed"]),
            "passed": independence["passed"],
        },
        "candidate_exact_cell_public_composition": {
            "expected_count": 2,
            "passed_count": candidate_cell["passed_count"],
            "nonpolar": candidate_cell["nonpolar"],
            "passed": candidate_cell["passed"],
            "evidence_content_sha256": candidate_cell["content_sha256"],
        },
        "source_unchanged": {
            "expected_count": 1,
            "passed_count": int(before == after),
            "passed": before == after,
        },
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-independent-obc2-validation-v1",
        "recorded_date": RECORDED_DATE,
        "claim_boundary": (
            "Equation, derivative, and runtime-independence validation only; no "
            "experimental labels, fitting, hydration ranking, gas-MLIP admission, "
            "or public scientific promotion."
        ),
        "stage0_binding": {
            "generator": STAGE0_GENERATOR_REL,
            "generator_sha256": STAGE0_GENERATOR_SHA256,
            "oracle": STAGE0_MAIN_REL,
            "oracle_file_sha256": STAGE0_MAIN_FILE_SHA256,
            "oracle_content_sha256": STAGE0_MAIN_CONTENT_SHA256,
            "portable_fixture": STAGE0_PORTABLE_REL,
            "portable_file_sha256": STAGE0_PORTABLE_FILE_SHA256,
            "portable_content_sha256": STAGE0_PORTABLE_CONTENT_SHA256,
            "source_freeze_sha256": STAGE0_SOURCE_FREEZE_SHA256,
        },
        "supersession": {
            "historical_artifact": HISTORICAL_REVALIDATION_REL,
            "historical_file_sha256": HISTORICAL_REVALIDATION_FILE_SHA256,
            "historical_content_sha256": HISTORICAL_REVALIDATION_CONTENT_SHA256,
            "supersedes_core_claims": [
                "OBC-II ACE-or-none solvent scalar/component/force parity",
                "OBC-II solvent raw Hessian and direct-HVP core parity",
            ],
            "does_not_supersede_claims": [
                "ANI2x gas-plus-solvent analytic FREQ workflow qualification",
                "ANI2x gas-plus-solvent P-RFO and Dimer workflow qualification",
                "gas-MLIP checkpoint/device/dtype/D4 cell qualification",
                "historical performance observations",
            ],
        },
        "source_snapshot": before,
        "source_unchanged": True,
        "environment": _environment(),
        "selection": selection,
        "gates": GATES,
        "fd_steps_angstrom": list(SOLVENT_FD_STEPS_ANGSTROM),
        "guarded_no_openmm": independence,
        "replay_526": replay,
        "portable_48_ef_radius": parity,
        "full_hessian_matrix": full_h,
        "direct_hvp_matrix": hvp,
        "candidate_exact_cell": candidate_cell,
        "qualification_gates": qualification_gates,
        "mae_drift_bound_definition": (
            "mean absolute new-minus-sealed prediction drift only; no label or old "
            "score was read or recomputed"
        ),
        "passed": bool(passed),
    }
    return core.seal_artifact(artifact)


def _count_gate(section: Mapping[str, Any], expected: int) -> dict[str, Any]:
    records = section.get("records", [])
    passed_count = sum(bool(row.get("passed")) for row in records)
    return {
        "expected_count": expected,
        "passed_count": passed_count,
        "passed": bool(
            section.get("passed") and len(records) == expected and passed_count == expected
        ),
    }


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_candidate_record(record: Mapping[str, Any], nonpolar: str) -> None:
    if record.get("nonpolar") != nonpolar or record.get("passed") is not True:
        raise ValueError(f"Candidate record for {nonpolar} did not pass")
    bounds = {
        "same_scalar_energy_abs_hartree": 1.0e-12,
        "same_scalar_force_max_hartree_per_angstrom": 1.0e-12,
        "hessian_asymmetry_max_hartree_per_angstrom2": 1.0e-10,
        "hessian_composition_max_hartree_per_angstrom2": 1.0e-12,
        "direct_hvp_dense_max_hartree_per_angstrom2": 1.0e-9,
    }
    for field, limit in bounds.items():
        value = record.get(field)
        if _finite_metric(value, field) > limit:
            raise ValueError(f"Candidate record {nonpolar} failed {field}")
    if record.get("direct_hvp_dense_hessian_forbidden") is not True:
        raise ValueError(f"Candidate record {nonpolar} did not forbid dense Hessian")
    ladder = record.get("hessian_force_fd_ladder")
    if not isinstance(ladder, list) or len(ladder) != 3:
        raise ValueError(f"Candidate record {nonpolar} has incomplete force-FD ladder")
    if [row.get("step_angstrom") for row in ladder] != [1.0e-6, 2.0e-6, 5.0e-6]:
        raise ValueError(f"Candidate record {nonpolar} changed force-FD steps")
    for row in ladder:
        rms = row.get("rms_hartree_per_angstrom2")
        maximum = row.get("max_hartree_per_angstrom2")
        if (
            row.get("passed") is not True
            or _finite_metric(rms, "candidate FD RMS") > 5.0e-6
            or _finite_metric(maximum, "candidate FD maximum") > 2.0e-5
        ):
            raise ValueError(f"Candidate record {nonpolar} failed force-FD gate")
    identity = record.get("gate_identity")
    provider = identity.get("provider") if isinstance(identity, Mapping) else None
    expected_identity = {
        "model": "ani2x",
        "device": "cpu",
        "d4": False,
        "checkpoint_sha256": ANI2X_CHECKPOINT_SHA256,
        "dtype": "float64",
    }
    if not isinstance(identity, Mapping) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError(f"Candidate record {nonpolar} gate identity changed")
    expected_provider = {
        "runtime_provider": "torch-obc2",
        "model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar": nonpolar,
    }
    if not isinstance(provider, Mapping) or any(
        provider.get(key) != value for key, value in expected_provider.items()
    ):
        raise ValueError(f"Candidate record {nonpolar} provider identity changed")


def validate_candidate_cell_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_source_snapshot: Mapping[str, Any] | None = None,
) -> None:
    if core.artifact_content_sha256(evidence) != evidence.get("content_sha256"):
        raise ValueError("Candidate exact-cell evidence self-hash mismatch")
    if evidence.get("artifact_type") != "independent-obc2-candidate-exact-cell-v1":
        raise ValueError("Unexpected candidate exact-cell evidence type")
    if evidence.get("qualification_stage") != (
        "candidate qualification, not installed qualification"
    ):
        raise ValueError("Candidate evidence overstates installed qualification")
    if evidence.get("nonpolar") != ["ace", "none"]:
        raise ValueError("Candidate evidence must contain exactly ACE and NONE")
    if evidence.get("expected_count") != 2 or evidence.get("passed_count") != 2:
        raise ValueError("Candidate exact-cell evidence denominator is incomplete")
    if evidence.get("passed") is not True:
        raise ValueError("Candidate exact-cell evidence did not pass")
    expected_cell = {
        "checkpoint_sha256": ANI2X_CHECKPOINT_SHA256,
        "model": "ani2x",
        "device": "cpu",
        "dtype": "float64",
        "d4": False,
    }
    if any(evidence.get(key) != value for key, value in expected_cell.items()):
        raise ValueError("Candidate exact-cell top-level identity changed")
    candidate_git_sha = evidence.get("candidate_git_sha")
    if (
        not isinstance(candidate_git_sha, str)
        or len(candidate_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in candidate_git_sha)
    ):
        raise ValueError("Candidate exact-cell Git SHA is malformed")
    harness_sha = evidence.get("harness_sha256")
    if not _valid_sha256(harness_sha):
        raise ValueError("Candidate exact-cell harness SHA is malformed")
    current_harness_sha = core.sha256_file(ROOT / CANDIDATE_CELL_HARNESS_REL)
    if harness_sha != current_harness_sha:
        raise ValueError("Candidate exact-cell harness SHA does not match current harness")
    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("Candidate exact-cell records must contain exactly ACE and NONE")
    if [record.get("nonpolar") for record in records if isinstance(record, Mapping)] != [
        "ace",
        "none",
    ]:
        raise ValueError("Candidate exact-cell records changed ACE/NONE identity")
    for record, nonpolar in zip(records, ("ace", "none"), strict=True):
        if not isinstance(record, Mapping):
            raise ValueError("Candidate exact-cell records must be mappings")
        _validate_candidate_record(record, nonpolar)
    snapshot = evidence.get("candidate_source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Candidate source snapshot is missing")
    files = snapshot.get("files_sha256")
    if not isinstance(files, Mapping) or not files or not all(
        isinstance(path, str) and _valid_sha256(digest) for path, digest in files.items()
    ):
        raise ValueError("Candidate source snapshot file inventory is malformed")
    observed_files_aggregate = core.sha256_bytes(core.canonical_json_bytes(files))
    if snapshot.get("files_aggregate_sha256") != observed_files_aggregate:
        raise ValueError("Candidate source snapshot aggregate does not replay")
    if snapshot.get("git_head") != candidate_git_sha:
        raise ValueError("Candidate source snapshot Git SHA does not match candidate")
    if snapshot.get("worktree_status") != []:
        raise ValueError("Candidate source snapshot was not generated from a clean tree")
    if expected_source_snapshot is not None:
        if expected_source_snapshot.get("worktree_status") != []:
            raise ValueError("Current qualification source snapshot is dirty")
        if snapshot.get("files_sha256") != expected_source_snapshot.get("files_sha256"):
            raise ValueError("Candidate source inventory does not match current qualification source")
        if snapshot.get("files_aggregate_sha256") != expected_source_snapshot.get(
            "files_aggregate_sha256"
        ):
            raise ValueError("Candidate source aggregate does not match current qualification source")


def _finite_metric(value: object, label: str, *, nonnegative: bool = True) -> float:
    if (
        type(value) not in (int, float) or not math.isfinite(value)
        or (nonnegative and value < 0)
    ):
        raise ValueError(f"Invalid finite numeric metric: {label}")
    return float(value)


def _record_index(section: Mapping[str, Any], fields: tuple[str, ...], expected: set) -> dict:
    rows = section.get("records")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f"Full-matrix records incomplete: expected {len(expected)}")
    indexed = {}
    expected_types = [
        {type(identity[index]) for identity in expected}
        for index in range(len(fields))
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Full-matrix records must be mappings")
        key = tuple(row.get(field) for field in fields)
        if any(type(value) not in types for value, types in zip(key, expected_types, strict=True)):
            raise ValueError("Malformed full-matrix record identity types")
        try:
            if key not in expected or key in indexed:
                raise ValueError(f"Unexpected or duplicate full-matrix record identity: {key}")
            indexed[key] = row
        except TypeError as exc:
            raise ValueError("Malformed full-matrix record identity") from exc
    return indexed


def _require_equal(observed: object, expected: object, label: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise ValueError(f"Full-matrix {label} does not replay")


def _row_verdict(row: Mapping[str, Any], passed: bool, label: str) -> None:
    _require_equal(row.get("passed"), bool(passed), f"{label} verdict")
    if not passed:
        raise ValueError(f"Full-matrix {label} failed original thresholds")


def _validate_energy_rows(section, golden_rows, *, portable: bool) -> None:
    fields = ("case", "nonpolar", "geometry") if portable else ("compound_id", "nonpolar")
    golden_index = {tuple(row[field] for field in fields): row for row in golden_rows}
    rows = _record_index(section, fields, set(golden_index))
    parameter_names = {
        "provider_parameters_radius_nm_screen", "intrinsic_radii_angstrom",
        "screen_factors", "offset_radii_nm", "scaled_offset_radii_nm",
        "finalized_particle_charge_offset_scaled",
    }
    drifts, parameters, force_maxima, force_rms_values = [], [], [], []
    for key, golden in golden_index.items():
        row = rows[key]
        _require_equal(row.get("golden_record_content_sha256"), golden["record_content_sha256"], "golden record hash")
        _require_equal(row.get("topology_sha256"), golden["topology"]["topology_sha256"], "topology hash")
        _require_equal(row.get("topology_match"), True, "topology identity")
        _require_equal(row.get("charge_identity_match"), True, "charge identity")
        if not portable:
            _require_equal(row.get("input_identity_sha256"), golden["input_identity_sha256"], "input identity")
        provenance = row.get("provider_provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("Missing full-matrix provider provenance")
        for field, expected in {
            "runtime_provider": "torch-obc2", "runtime_openmm_required": False,
            "model": "obc2", "profile": "obc2-mbondi2", "nonpolar": golden["nonpolar"],
            "ordered_charge_sha256": golden["charge_vector_sha256"],
            "topology_sha256": golden["topology"]["topology_sha256"],
        }.items():
            _require_equal(provenance.get(field), expected, f"provider {field}")
        energy = row.get("energy")
        if not isinstance(energy, Mapping):
            raise ValueError("Missing full-matrix energy metrics")
        energy_limit = GATES["energy_abs_hartree"] + GATES["energy_relative"] * abs(float(golden["total_hartree"]))
        _require_equal(energy.get("total_limit_hartree"), energy_limit, "energy threshold")
        total = _finite_metric(energy.get("total_abs_hartree"), "total energy")
        polar = _finite_metric(energy.get("polar_abs_hartree"), "polar energy")
        nonpolar = _finite_metric(energy.get("nonpolar_abs_hartree"), "nonpolar energy")
        energy_pass = (
            total <= energy_limit and polar <= GATES["component_abs_hartree"]
            and nonpolar <= GATES["component_abs_hartree"]
        )
        _row_verdict(energy, energy_pass, "energy")
        deltas = row.get("parameter_deltas")
        if not isinstance(deltas, Mapping) or set(deltas) != parameter_names:
            raise ValueError("Full-matrix parameter metrics incomplete")
        parameter_max = max(_finite_metric(value, name) for name, value in deltas.items())
        _require_equal(row.get("parameter_max_abs"), parameter_max, "parameter maximum")
        passed = energy_pass and parameter_max <= GATES["radius_parameter_abs"]
        if portable:
            force_max = _finite_metric(row.get("force_max_abs_hartree_per_angstrom"), "force maximum")
            force_rms = _finite_metric(row.get("force_rms_hartree_per_angstrom"), "force RMS")
            passed = passed and force_max <= GATES["force_max_hartree_per_angstrom"] and force_rms <= GATES["force_rms_hartree_per_angstrom"]
            force_maxima.append(force_max)
            force_rms_values.append(force_rms)
        _row_verdict(row, passed, "portable E/F" if portable else "sealed E")
        drifts.append(total)
        parameters.append(parameter_max)
    _require_equal(section.get("passed"), True, "energy section verdict")
    _require_equal(section.get("maximum_parameter_abs"), max(parameters), "parameter aggregate")
    if portable:
        _require_equal(section.get("case_count"), 12, "portable case count")
        _require_equal(section.get("comparison_count"), 48, "portable denominator")
        for field, value in (
            ("maximum_total_energy_abs_hartree", max(drifts)),
            ("maximum_force_abs_hartree_per_angstrom", max(force_maxima)),
            ("maximum_force_rms_hartree_per_angstrom", max(force_rms_values)),
        ):
            _require_equal(section.get(field), value, field)
    else:
        mean = float(np.mean(drifts))
        for field, value in (
            ("case_count", 526), ("profile_count", 2), ("record_count", 1052),
            ("maximum_absolute_prediction_drift_hartree", max(drifts)),
            ("mean_absolute_prediction_drift_hartree", mean),
            ("mae_drift_upper_bound_hartree", mean),
            ("mae_drift_upper_bound_kcal_mol", mean * KCAL_PER_HARTREE),
        ):
            _require_equal(section.get(field), value, field)


def _validate_fd_metrics(ladder: Mapping[str, Any]) -> None:
    if not isinstance(ladder, Mapping):
        raise ValueError("Missing Hessian FD ladder")
    _require_equal(ladder.get("step_order"), list(SOLVENT_FD_STEPS_ANGSTROM), "FD steps")
    rows = ladder.get("records")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Full-matrix FD records incomplete")
    verdicts = []
    for row, step in zip(rows, SOLVENT_FD_STEPS_ANGSTROM, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("Malformed FD record")
        _require_equal(row.get("step_angstrom"), step, "FD step")
        if not _valid_sha256(row.get("matrix_sha256")):
            raise ValueError("Missing FD matrix hash")
        rms = _finite_metric(row.get("rms_hartree_per_angstrom2"), "FD RMS")
        maximum = _finite_metric(row.get("max_hartree_per_angstrom2"), "FD maximum")
        passed = rms <= GATES["hessian_fd_rms_hartree_per_angstrom2"] and maximum <= GATES["hessian_fd_max_hartree_per_angstrom2"]
        # Coarse failure is logged, not promoted into an admission gate.
        _require_equal(row.get("passed"), passed, "FD row verdict")
        verdicts.append(passed)
    pair_rms = _finite_metric(ladder.get("finest_pair_difference_rms_hartree_per_angstrom2"), "FD pair RMS")
    pair_max = _finite_metric(ladder.get("finest_pair_difference_max_hartree_per_angstrom2"), "FD pair maximum")
    pair_pass = pair_rms <= 2 * GATES["hessian_fd_rms_hartree_per_angstrom2"] and pair_max <= 2 * GATES["hessian_fd_max_hartree_per_angstrom2"]
    for field, value in (
        ("coarse_pass_observed", verdicts[0]), ("all_three_pass_observed", all(verdicts)),
        ("both_finest_pass", all(verdicts[1:])), ("finest_pair_pass", pair_pass),
    ):
        _require_equal(ladder.get(field), value, field)
    _row_verdict(ladder, all(verdicts[1:]) and pair_pass, "Hessian FD ladder")


def _validate_hessian_rows(section, portable) -> set:
    from maple.function.calculator.extra_correction.implicit.obc2_topology import OBC2Topology

    inputs = {
        (case, profile, geometry): (atoms, source)
        for case, profile, geometry, atoms, _charges, source in _full_h_inputs(portable)
    }
    rows = _record_index(section, ("case", "nonpolar", "geometry"), set(inputs))
    _require_equal(section.get("selection"), select_full_h_cases(portable), "Hessian selection")
    _require_equal(section.get("water_displacement_seed"), WATER_DISPLACEMENT_SEED, "water seed")
    _require_equal(section.get("displacement_scale_angstrom"), DISPLACEMENT_SCALE_ANGSTROM, "water displacement")
    oracle = section.get("live_force_fd_oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError("Missing live force-FD oracle identity")
    for field, value in (
        ("version", OPENMM_VERSION), ("module", "openmm.app.internal.customgbforces"),
        ("source_sha256", OPENMM_SOURCE_SHA256), ("platform", "Reference"),
    ):
        _require_equal(oracle.get(field), value, f"live oracle {field}")
    for key, row in rows.items():
        atoms, source = inputs[key]
        for field, value in (
            ("atom_count", len(atoms)), ("source_identity_sha256", source),
            ("coordinate_sha256", core.sha256_bytes(core.canonical_json_bytes(np.asarray(atoms.positions, dtype=np.float64).tolist()))),
            ("topology_sha256", OBC2Topology.from_atoms(atoms).identity_sha256),
            ("raw_hessian_shape", [3 * len(atoms)] * 2),
        ):
            _require_equal(row.get(field), value, f"Hessian {field}")
        provenance = row.get("provider_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Missing Hessian provider provenance")
        for field, value in (
            ("runtime_provider", "torch-obc2"), ("runtime_openmm_required", False),
            ("model", "obc2"), ("profile", "obc2-mbondi2"), ("nonpolar", key[1]),
            ("topology_sha256", row["topology_sha256"]),
        ):
            _require_equal(provenance.get(field), value, f"Hessian provider {field}")
        if not _valid_sha256(row.get("raw_hessian_sha256")):
            raise ValueError("Missing raw Hessian hash")
        _finite_metric(row.get("raw_hessian_max_abs"), "Hessian maximum")
        norm = _finite_metric(row.get("raw_hessian_frobenius"), "Hessian norm")
        asymmetry = _finite_metric(row.get("raw_hessian_asymmetry_max"), "Hessian asymmetry")
        translations = row.get("translation_norms")
        if not isinstance(translations, list) or len(translations) != 3:
            raise ValueError("Hessian translation norms incomplete")
        translations = [_finite_metric(value, "translation norm") for value in translations]
        limit = max(GATES["translation_absolute_hartree_per_angstrom2"], GATES["translation_relative_frobenius"] * norm)
        _require_equal(row.get("translation_limit"), limit, "translation limit")
        _validate_fd_metrics(row.get("fd_ladder"))
        _row_verdict(row, asymmetry <= GATES["hessian_asymmetry_hartree_per_angstrom2"] and max(translations) <= limit, "Hessian")
    _require_equal(section.get("record_count"), 16, "Hessian denominator")
    _require_equal(section.get("passed"), True, "Hessian section verdict")
    return set(rows)


def _finite_vector(value: object, size: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Invalid full-matrix vector shape: {label}")
    for item in value:
        _finite_metric(item, label, nonnegative=False)
    return np.asarray(value, dtype=np.float64)


def _validate_hvp_rows(section, portable, hessian_keys) -> None:
    golden_index = _portable_index(portable)
    expected = {
        (case, profile, geometry, direction)
        for case in PORTABLE_CASES for profile in PROFILE_ORDER
        for geometry in GEOMETRY_ORDER for direction in range(HVP_DIRECTION_COUNT)
    }
    rows = _record_index(section, ("case", "nonpolar", "geometry", "direction_index"), expected)
    compared = 0
    for key, row in rows.items():
        case, profile, geometry, direction_index = key
        golden = golden_index[(case, profile, geometry)]
        size = 3 * golden["atom_count"]
        seed, expected_direction = normalized_direction(
            case_index=PORTABLE_CASES.index(case), geometry_index=GEOMETRY_ORDER.index(geometry),
            direction_index=direction_index, size=size,
        )
        provenance = row.get("derivative_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Missing HVP derivative provenance")
        for field, value in (
            ("runtime_provider", "torch-obc2"), ("runtime_openmm_required", False),
            ("model", "obc2"), ("profile", "obc2-mbondi2"), ("nonpolar", profile),
            ("topology_sha256", golden["topology"]["topology_sha256"]),
            ("ordered_charge_sha256", golden["charge_vector_sha256"]),
        ):
            _require_equal(provenance.get(field), value, f"HVP provider {field}")
        _require_equal(row.get("seed"), seed, "HVP seed")
        direction = _finite_vector(row.get("direction"), size, "HVP direction")
        hvp = _finite_vector(row.get("hvp"), size, "HVP")
        if direction.shape != (size,) or not np.array_equal(direction, expected_direction):
            raise ValueError("HVP seeded direction changed")
        if hvp.shape != (size,) or not np.isfinite(hvp).all():
            raise ValueError("HVP vector is incomplete or non-finite")
        _require_equal(row.get("direction_sha256"), _canonical_array_sha256(direction), "direction hash")
        _require_equal(row.get("direction_l2_norm"), float(np.linalg.norm(direction)), "direction norm")
        _require_equal(row.get("hvp_sha256"), _canonical_array_sha256(hvp), "HVP hash")
        _require_equal(row.get("dense_hessian_forbidden_during_call"), True, "direct-HVP prohibition")
        energy = _finite_metric(row.get("same_scalar_energy_abs_hartree"), "HVP bundle energy")
        force = _finite_metric(row.get("same_scalar_force_max_hartree_per_angstrom"), "HVP bundle force")
        hvp_pass = True
        if key[:3] in hessian_keys:
            compared += 1
            product = _finite_vector(row.get("raw_hessian_product"), size, "raw Hessian product")
            if product.shape != (size,) or not np.isfinite(product).all():
                raise ValueError("Compared HVP raw Hessian product is missing or non-finite")
            _require_equal(row.get("raw_hessian_product_sha256"), _canonical_array_sha256(product), "raw Hessian product hash")
            difference = hvp - product
            maximum = float(np.max(np.abs(difference), initial=0.0))
            scaled = float(np.max(np.abs(difference) / np.maximum(np.abs(product), GATES["hvp_atol_hartree_per_angstrom2"]), initial=0.0))
            _require_equal(row.get("hvp_vs_raw_hessian_max"), maximum, "HVP maximum")
            _require_equal(row.get("hvp_vs_raw_hessian_scaled_max"), scaled, "HVP scaled maximum")
            hvp_pass = bool(np.allclose(hvp, product, rtol=GATES["hvp_rtol"], atol=GATES["hvp_atol_hartree_per_angstrom2"]))
        else:
            for field in ("raw_hessian_product", "raw_hessian_product_sha256", "hvp_vs_raw_hessian_max", "hvp_vs_raw_hessian_scaled_max"):
                if row.get(field) is not None:
                    raise ValueError("Unexpected dense comparison outside frozen HVP selection")
        _row_verdict(row, energy <= GATES["bundle_energy_abs_hartree"] and force <= GATES["bundle_force_max_hartree_per_angstrom"] and hvp_pass, "direct HVP")
    _require_equal(compared, 36, "dense HVP comparison denominator")
    for field, value in (
        ("base_seed", HVP_BASE_SEED), ("direction_count_per_case_geometry_profile", HVP_DIRECTION_COUNT),
        ("record_count", 144), ("passed", True),
    ):
        _require_equal(section.get(field), value, f"HVP {field}")


def _validate_full_matrix_records(artifact: Mapping[str, Any]) -> None:
    # Fail on absent arrays before consulting any caller-controlled summary.
    for name, count in (
        ("replay_526", 1052), ("portable_48_ef_radius", 48),
        ("full_hessian_matrix", 16), ("direct_hvp_matrix", 144),
    ):
        section = artifact.get(name)
        if not isinstance(section, Mapping) or not isinstance(section.get("records"), list) or len(section["records"]) != count:
            raise ValueError(f"{name} records incomplete: expected {count}")
    stage0, portable = load_stage0()
    expected_binding = {
        "generator": STAGE0_GENERATOR_REL,
        "generator_sha256": STAGE0_GENERATOR_SHA256,
        "oracle": STAGE0_MAIN_REL,
        "oracle_file_sha256": STAGE0_MAIN_FILE_SHA256,
        "oracle_content_sha256": STAGE0_MAIN_CONTENT_SHA256,
        "portable_fixture": STAGE0_PORTABLE_REL,
        "portable_file_sha256": STAGE0_PORTABLE_FILE_SHA256,
        "portable_content_sha256": STAGE0_PORTABLE_CONTENT_SHA256,
        "source_freeze_sha256": STAGE0_SOURCE_FREEZE_SHA256,
    }
    _require_equal(artifact.get("schema_version"), 1, "schema")
    _require_equal(artifact.get("stage0_binding"), expected_binding, "sealed reference binding")
    _require_equal(artifact.get("gates"), GATES, "frozen thresholds")
    _require_equal(artifact.get("fd_steps_angstrom"), list(SOLVENT_FD_STEPS_ANGSTROM), "frozen FD ladder")
    _require_equal(artifact.get("selection"), select_full_h_cases(portable), "frozen selection")
    _validate_energy_rows(artifact["replay_526"], stage0["records"], portable=False)
    _validate_energy_rows(artifact["portable_48_ef_radius"], portable["records"], portable=True)
    hessian_keys = _validate_hessian_rows(artifact["full_hessian_matrix"], portable)
    _validate_hvp_rows(artifact["direct_hvp_matrix"], portable, hessian_keys)
    independence = artifact.get("guarded_no_openmm")
    if not isinstance(independence, Mapping):
        raise ValueError("Missing no-OpenMM proof")
    for field in ("network_disabled", "import_blocker", "metadata_blocker", "passed"):
        _require_equal(independence.get(field), True, f"no-OpenMM {field}")
    _require_equal(independence.get("script_sha256"), _sha256_text(_no_openmm_script()), "no-OpenMM script")
    result = independence.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("Missing no-OpenMM proof result")
    for field, value in (("ok", True), ("openmm_loaded", False), ("provider", "torch-obc2")):
        _require_equal(result.get(field), value, f"no-OpenMM result {field}")
    _finite_metric(result.get("energy_hartree"), "no-OpenMM energy", nonnegative=False)
    if not all(_valid_sha256(result.get(field)) for field in ("force_sha256", "hessian_sha256", "hvp_sha256")):
        raise ValueError("No-OpenMM result hashes incomplete")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Independent OBC validation artifact self-hash mismatch")
    if artifact.get("artifact_type") != "route1-independent-obc2-validation-v1":
        raise ValueError("Unexpected independent OBC validation artifact type")
    if artifact.get("recorded_date") != RECORDED_DATE:
        raise ValueError("Independent OBC validation artifact date changed")
    if artifact.get("source_unchanged") is not True:
        raise ValueError("Validation artifact does not prove source stability")
    source = artifact.get("source_snapshot")
    if not isinstance(source, Mapping):
        raise ValueError("Validation qualification source snapshot is missing")
    candidate_cell = artifact.get("candidate_exact_cell")
    if not isinstance(candidate_cell, Mapping):
        raise ValueError("Validation candidate exact-cell evidence is missing")
    validate_candidate_cell_evidence(
        candidate_cell,
        expected_source_snapshot=source,
    )
    if candidate_cell.get("candidate_git_sha") != source.get("git_head"):
        raise ValueError("Validation candidate/source Git identity changed")
    validate_current_source_binding(source)
    _validate_full_matrix_records(artifact)
    if artifact.get("selection", {}).get("selected_cases") != list(FULL_H_CASES):
        raise ValueError("Full-H preregistered selection changed")
    if artifact.get("replay_526", {}).get("record_count") != 1052:
        raise ValueError("526 replay is incomplete")
    if artifact.get("portable_48_ef_radius", {}).get("comparison_count") != 48:
        raise ValueError("Portable 48-cell replay is incomplete")
    if artifact.get("full_hessian_matrix", {}).get("record_count") != 16:
        raise ValueError("Full-H matrix is incomplete")
    if artifact.get("direct_hvp_matrix", {}).get("record_count") != 144:
        raise ValueError("Direct-HVP matrix is incomplete")
    expected_gates = {
        "sealed_oracle_1052_energy",
        "portable_oracle_48_energy_force",
        "analytic_hessian_16_fd",
        "direct_hvp_144",
        "no_openmm_candidate_runtime",
        "candidate_exact_cell_public_composition",
        "source_unchanged",
    }
    gates = artifact.get("qualification_gates", {})
    if set(gates) != expected_gates:
        raise ValueError("Qualification gate set changed")
    gate_counts = {
        "sealed_oracle_1052_energy": 1052,
        "portable_oracle_48_energy_force": 48,
        "analytic_hessian_16_fd": 16,
        "direct_hvp_144": 144,
        "no_openmm_candidate_runtime": 1,
        "candidate_exact_cell_public_composition": 2,
        "source_unchanged": 1,
    }
    for name, count in gate_counts.items():
        gate = gates[name]
        if not isinstance(gate, Mapping):
            raise ValueError("Qualification gate is not a mapping")
        _require_equal(gate.get("expected_count"), count, f"{name} expected denominator")
        _require_equal(gate.get("passed_count"), count, f"{name} passed denominator")
        _require_equal(gate.get("passed"), True, f"{name} verdict")
    _require_equal(
        gates["candidate_exact_cell_public_composition"].get("evidence_content_sha256"),
        candidate_cell["content_sha256"], "candidate gate evidence hash",
    )
    if gates["analytic_hessian_16_fd"].get("steps_angstrom") != list(
        SOLVENT_FD_STEPS_ANGSTROM
    ):
        raise ValueError("Hessian FD ladder changed")
    if gates["direct_hvp_144"].get("dense_hessian_forbidden") is not True:
        raise ValueError("Direct-HVP dense-Hessian prohibition was not proven")
    if gates["candidate_exact_cell_public_composition"].get("nonpolar") != [
        "ace",
        "none",
    ]:
        raise ValueError("Candidate exact-cell nonpolar denominator changed")
    sections = (
        "guarded_no_openmm",
        "replay_526",
        "portable_48_ef_radius",
        "full_hessian_matrix",
        "direct_hvp_matrix",
    )
    if artifact.get("passed") is not all(artifact[name]["passed"] for name in sections):
        raise ValueError("Validation artifact aggregate status is inconsistent")


def _write_artifact(artifact: Mapping[str, Any]) -> None:
    destination = ROOT / OUTPUT_REL
    if destination.exists():
        raise FileExistsError(f"Validation artifacts are immutable: {destination}")
    core.write_json_atomic(destination, artifact)


def _write_rejected_attempt(artifact: Mapping[str, Any], error: Exception) -> Path:
    """Retain a completed rejection separately from passing/sealable evidence."""
    destination = (ROOT / OUTPUT_REL).parent / f"completed-rejected-{uuid4().hex}.json"
    attempt = core.seal_artifact({
        "schema_version": 1,
        "artifact_type": "independent-obc2-completed-rejected-attempt-v1",
        "status": "rejected",
        "source_snapshot": artifact.get("source_snapshot"),
        "stage0_binding": artifact.get("stage0_binding"),
        "validation_exception": {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        },
        "raw_artifact": artifact,
    })
    raw = (json.dumps(attempt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic exclusive publication, including in the unlikely event of an ID collision.
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _self_test() -> None:
    _main, portable = load_stage0()
    selection = select_full_h_cases(portable)
    assert selection["selected_largest_case"] == "nitralin"
    assert selection["maximum_atom_count"] == 42
    seed, first = normalized_direction(
        case_index=0, geometry_index=0, direction_index=0, size=36
    )
    seed_again, second = normalized_direction(
        case_index=0, geometry_index=0, direction_index=0, size=36
    )
    assert seed == seed_again == HVP_BASE_SEED
    np.testing.assert_array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0, rtol=0.0, atol=1.0e-15)
    analytic = np.zeros((2, 2), dtype=np.float64)
    matrices = {
        step: np.zeros((2, 2), dtype=np.float64) for step in SOLVENT_FD_STEPS_ANGSTROM
    }
    ladder = assess_fd_ladder(analytic, matrices)
    assert ladder["all_three_pass_observed"] is True
    assert ladder["both_finest_pass"] is True
    assert ladder["passed"] is True
    print("independent-obc2-validation-self-test=passed")


def _core_canary() -> dict[str, Any]:
    _stage0, portable = load_stage0()
    Provider = _provider_class()
    row = _portable_index(portable)[("methanol", "ace", "base")]
    atoms, charges = _atoms_from_portable(row)
    provider = Provider(atoms, charges, nonpolar="ace", device="cpu")
    result = provider.evaluate(atoms, need_forces=True)
    metrics = _energy_metrics(result, row)
    force_difference = np.asarray(result.forces_hartree_per_angstrom) - np.asarray(
        row["forces_hartree_per_angstrom"]
    )
    metrics.update(
        force_max=float(np.max(np.abs(force_difference), initial=0.0)),
        parameter_max=max(_parameter_deltas(provider, row["parameters"]).values()),
    )
    seed, direction = normalized_direction(
        case_index=1, geometry_index=0, direction_index=0, size=3 * len(atoms)
    )
    with (
        patch.object(provider.kernel, "hessian", side_effect=_forbidden_dense_hessian),
        patch.object(provider, "hessian", side_effect=_forbidden_dense_hessian),
    ):
        directional = provider.directional_derivatives(atoms, direction)
    metrics["hvp_seed"] = seed
    metrics["hvp_sha256"] = _canonical_array_sha256(
        directional.hvp_hartree_per_angstrom2
    )
    metrics["passed"] = bool(
        metrics["passed"]
        and metrics["force_max"] <= GATES["force_max_hartree_per_angstrom"]
        and metrics["parameter_max"] <= GATES["radius_parameter_abs"]
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--core-canary", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.validate_only:
        artifact = json.loads((ROOT / OUTPUT_REL).read_text(encoding="utf-8"))
        validate_artifact(artifact)
        print(
            json.dumps(
                {
                    "path": OUTPUT_REL,
                    "file_sha256": core.sha256_file(ROOT / OUTPUT_REL),
                    "content_sha256": artifact["content_sha256"],
                    "passed": artifact["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.core_canary:
        print(json.dumps(_core_canary(), indent=2, sort_keys=True))
        return
    stage0, portable = load_stage0()
    preflight = {
        "status": (
            "READY"
            if (
                ROOT
                / "maple/function/calculator/extra_correction/implicit/independent_obc2.py"
            ).is_file()
            else "WAITING_FOR_PROVIDER"
        ),
        "git_head": _git("rev-parse", "HEAD"),
        "source_snapshot": source_snapshot(),
        "stage0_main_content_sha256": stage0["content_sha256"],
        "stage0_portable_content_sha256": portable["content_sha256"],
        "selection": select_full_h_cases(portable),
        "hvp_base_seed": HVP_BASE_SEED,
        "fd_steps_angstrom": list(SOLVENT_FD_STEPS_ANGSTROM),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    if preflight["status"] != "READY":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        raise RuntimeError("IndependentOBC2 core is not yet available")
    if (ROOT / OUTPUT_REL).exists():
        raise FileExistsError(
            f"Validation artifact already exists: {ROOT / OUTPUT_REL}"
        )
    artifact = _build_artifact()
    try:
        validate_artifact(artifact)
    except Exception as error:
        try:
            path = _write_rejected_attempt(artifact, error)
            print(f"Completed rejected attempt retained: {path}", file=sys.stderr)
        except Exception as retention_error:
            # A storage failure must not replace the original validation rejection.
            print(
                f"Failed to retain completed rejected attempt: {retention_error}",
                file=sys.stderr,
            )
        raise
    _write_artifact(artifact)
    print(
        json.dumps(
            {
                "path": OUTPUT_REL,
                "file_sha256": core.sha256_file(ROOT / OUTPUT_REL),
                "content_sha256": artifact["content_sha256"],
                "passed": artifact["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
