#!/usr/bin/env python3
"""Run the preregistered Route 1 ammonia inversion stationary-point checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stationary_point_checks import analyze_stationary_point

from maple.function.calculator.extra_correction.implicit.amber_chagb import (
    render_typed_mol2,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.frequency.frequency import MWFrequency
from maple.function.dispatcher.optimization.optimization import (
    Optimization,
)
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.read.filereader.mol2_reader import MOL2Reader

PROTOCOL_ID = "route1-foundation-stationary-20260913-v1"
HESSIAN_STEPS_ANGSTROM = (0.0005, 0.00025)
NEGATIVE_CUTOFF_CM1 = 30.0
MODE_DISPLACEMENT_ANGSTROM = 0.05
FORCE_MAX_STOP = 1.0e-4
FORCE_RMS_STOP = 7.5e-5
STEP_MAX_STOP = 3.0e-4
STEP_RMS_STOP = 2.0e-4
HESSIAN_ABSOLUTE_BUDGET = 5.0e-4
HESSIAN_RELATIVE_BUDGET_FACTOR = 1.0e-3
MODE_OVERLAP_MINIMUM = 0.80
EQUIVALENT_NH_BOND_TOLERANCE_ANGSTROM = 1.0e-3
EQUIVALENT_HNH_ANGLE_TOLERANCE_DEGREES = 0.1
EQUIVALENT_ENERGY_TOLERANCE_HARTREE = 1.0e-6


@dataclass(frozen=True)
class AttemptConfig:
    attempt_id: str = "post-rigid-subspace-repair-original-planar-v3"
    initialization: str = (
        "project N onto the plane of the original labeled H atoms; preserve "
        "the original H coordinates, topology, atom labels, charges, and radii"
    )
    prfo_max_iterations: int = 80
    prfo_recalc: int = 1
    prfo_trust_radius: float = 0.05
    prfo_trust_min: float = 0.001
    prfo_trust_max: float = 0.2
    prfo_hessian_update: str = "bofill"
    prfo_project_rigid_modes: bool = True
    downhill_method: str = "lbfgs"
    downhill_max_iterations: int = 100
    downhill_max_step_angstrom: float = 0.05
    downhill_curvature: float = 5.0
    force_max_stop_hartree_per_angstrom: float = FORCE_MAX_STOP
    force_rms_stop_hartree_per_angstrom: float = FORCE_RMS_STOP
    step_max_stop_angstrom: float = STEP_MAX_STOP
    step_rms_stop_angstrom: float = STEP_RMS_STOP
    mode_displacement_angstrom: float = MODE_DISPLACEMENT_ANGSTROM
    negative_cutoff_cm1: float = NEGATIVE_CUTOFF_CM1
    minimum_mass_metric_mode_overlap: float = MODE_OVERLAP_MINIMUM
    equivalent_nh_bond_tolerance_angstrom: float = (
        EQUIVALENT_NH_BOND_TOLERANCE_ANGSTROM
    )
    equivalent_hnh_angle_tolerance_degrees: float = (
        EQUIVALENT_HNH_ANGLE_TOLERANCE_DEGREES
    )
    equivalent_energy_tolerance_hartree: float = EQUIVALENT_ENERGY_TOLERANCE_HARTREE
    hessian_step_large_angstrom: float = HESSIAN_STEPS_ANGSTROM[0]
    hessian_step_small_angstrom: float = HESSIAN_STEPS_ANGSTROM[1]
    hessian_absolute_budget_hartree_per_angstrom2: float = HESSIAN_ABSOLUTE_BUDGET
    hessian_relative_budget_factor: float = HESSIAN_RELATIVE_BUDGET_FACTOR


ATTEMPT_CONFIG = AttemptConfig()

EXECUTION_SOURCE_PATHS = (
    "docs/implicit-solvation/benchmarks/run_route1_stationary_validation.py",
    "docs/implicit-solvation/benchmarks/stationary_point_checks.py",
    "maple/function/calculator/calculator_base.py",
    "maple/function/calculator/set_calculator.py",
    "maple/function/calculator/ani/_ani_calculator.py",
    "maple/function/calculator/extra_correction/implicit/charges.py",
    "maple/function/calculator/extra_correction/implicit/correction.py",
    "maple/function/calculator/extra_correction/implicit/ddx_lpb.py",
    "maple/function/calculator/extra_correction/implicit/openmm_gb.py",
    "maple/function/dispatcher/frequency/frequency.py",
    "maple/function/dispatcher/optimization/algorithm/LBFGS.py",
    "maple/function/dispatcher/optimization/algorithm/_common.py",
    "maple/function/dispatcher/optimization/optimization.py",
    "maple/function/dispatcher/ts/algorithm/PRFO.py",
    "maple/function/read/filereader/mol2_reader.py",
)

ENDPOINTS = {
    "ani2x-openmm-obc2-ace": {
        "implicit": "gb",
        "solvation_options": {
            "implicit": "water",
            "method": "gb",
            "provider": "openmm",
            "model": "obc2",
            "profile": "obc2-mbondi2",
            "nonpolar": "ace",
            "platform": "CPU",
            "experimental": True,
        },
    },
    "ani2x-ddlpb-kappa-0.1-polar-only": {
        "implicit": "pb",
        "solvation_options": {
            "implicit": "water",
            "method": "pb",
            "provider": "ddx",
            "model": "lpb",
            "profile": "ddlpb-union-mbondi2-v1",
            "nonpolar": "none",
            "solvent_kappa_inverse_angstrom": 0.1,
            "experimental": True,
        },
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError("refusing to serialize a non-finite numerical array")
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("refusing to serialize a non-finite numerical value")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _planar_ammonia(atoms: Atoms) -> Atoms:
    symbols = atoms.get_chemical_symbols()
    nitrogen = [index for index, symbol in enumerate(symbols) if symbol == "N"]
    hydrogens = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if len(nitrogen) != 1 or len(hydrogens) != 3 or len(atoms) != 4:
        raise ValueError("primary stationary anchor must be NH3 with one N and three H")
    result = atoms.copy()
    positions = result.get_positions()
    hydrogen_positions = positions[hydrogens]
    normal = np.cross(
        hydrogen_positions[1] - hydrogen_positions[0],
        hydrogen_positions[2] - hydrogen_positions[0],
    )
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1.0e-10:
        raise ValueError("NH3 hydrogen plane is degenerate")
    normal /= norm
    n_index = nitrogen[0]
    signed_height = float(
        np.dot(positions[n_index] - hydrogen_positions[0], normal)
    )
    positions[n_index] -= signed_height * normal
    result.set_positions(positions)
    return result


def _nitrogen_height(atoms: Atoms) -> float:
    symbols = atoms.get_chemical_symbols()
    nitrogen = symbols.index("N")
    hydrogens = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    positions = atoms.get_positions()
    h0, h1, h2 = positions[hydrogens]
    normal = np.cross(h1 - h0, h2 - h0)
    normal /= np.linalg.norm(normal)
    return float(np.dot(positions[nitrogen] - h0, normal))


def _umbrella_reference_mode(atoms: Atoms) -> np.ndarray:
    """Return the rigid-projected, mass-normalized N-height displacement."""
    positions = atoms.get_positions().copy()
    gradient = np.empty(3 * len(atoms), dtype=np.float64)
    delta = 1.0e-5
    for coordinate in range(gradient.size):
        plus = positions.reshape(-1).copy()
        minus = positions.reshape(-1).copy()
        plus[coordinate] += delta
        minus[coordinate] -= delta
        plus_atoms = atoms.copy()
        minus_atoms = atoms.copy()
        plus_atoms.set_positions(plus.reshape((-1, 3)))
        minus_atoms.set_positions(minus.reshape((-1, 3)))
        gradient[coordinate] = (
            _nitrogen_height(plus_atoms) - _nitrogen_height(minus_atoms)
        ) / (2.0 * delta)

    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    root_masses = np.repeat(np.sqrt(masses), 3)
    mass_weighted_direction = gradient / root_masses
    frequency = MWFrequency("/dev/null", atoms)
    rigid = frequency._build_translation_rotation_basis(masses, positions)
    left, singular_values, _ = np.linalg.svd(rigid, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(rigid.shape) * singular_values[0]
    basis = left[:, singular_values > tolerance]
    mass_weighted_direction -= basis @ (basis.T @ mass_weighted_direction)
    mass_weighted_direction /= np.linalg.norm(mass_weighted_direction)
    return (mass_weighted_direction / root_masses).reshape((len(atoms), 3))


def _mass_metric_overlap(atoms: Atoms, first: np.ndarray, second: np.ndarray) -> float:
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)[:, None]
    numerator = float(abs(np.sum(masses * first * second)))
    norm_first = float(np.sqrt(np.sum(masses * np.square(first))))
    norm_second = float(np.sqrt(np.sum(masses * np.square(second))))
    if not np.isfinite([numerator, norm_first, norm_second]).all():
        raise ValueError("mode overlap inputs and norms must be finite")
    if norm_first <= 0.0 or norm_second <= 0.0:
        raise ValueError("mode overlap requires nonzero mass-metric norms")
    return numerator / (norm_first * norm_second)


def _nh3_internal_coordinates(atoms: Atoms) -> dict[str, Any]:
    symbols = atoms.get_chemical_symbols()
    nitrogen = [index for index, symbol in enumerate(symbols) if symbol == "N"]
    hydrogens = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    if len(nitrogen) != 1 or len(hydrogens) != 3 or len(atoms) != 4:
        return {"intact": False, "reason": "composition is not N+3H"}
    positions = atoms.get_positions()
    n_position = positions[nitrogen[0]]
    vectors = positions[hydrogens] - n_position
    bonds = np.linalg.norm(vectors, axis=1)
    angles = []
    for first in range(3):
        for second in range(first):
            cosine = np.dot(vectors[first], vectors[second]) / (
                bonds[first] * bonds[second]
            )
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return {
        "intact": bool(np.all((bonds > 0.7) & (bonds < 1.5))),
        "sorted_nh_bond_lengths_angstrom": np.sort(bonds),
        "sorted_hnh_angles_degrees": np.sort(angles),
    }


def _iteration_count(output: Path) -> int:
    if not output.is_file():
        return 0
    return len(
        re.findall(
            r"^\s*Iteration:\s+\d+\s*$",
            output.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def _partial_attempt_evidence(attempt_dir: Path) -> dict[str, Any]:
    files = [path for path in attempt_dir.rglob("*") if path.is_file()]
    downhill_outputs = sorted(attempt_dir.glob("downhill-*.out"))
    return {
        "actual_prfo_iterations": _iteration_count(attempt_dir / "prfo.out"),
        "actual_downhill_iterations": {
            path.name: _iteration_count(path) for path in downhill_outputs
        },
        "partial_output_hashes": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path)
            for path in sorted(files)
            if path.name != "result.json"
        },
    }


def _set_stop_thresholds(atoms: Any) -> None:
    atoms.f_max_th = ATTEMPT_CONFIG.force_max_stop_hartree_per_angstrom
    atoms.f_rms_th = ATTEMPT_CONFIG.force_rms_stop_hartree_per_angstrom
    atoms.dp_max_th = ATTEMPT_CONFIG.step_max_stop_angstrom
    atoms.dp_rms_th = ATTEMPT_CONFIG.step_rms_stop_angstrom


def _calculator(atoms: Atoms, endpoint: dict[str, Any], output: Path):
    import torch

    return SetCalculator(
        torch.device("cpu"),
        "ani2x",
        str(output),
        atoms=atoms,
        implicit=endpoint["implicit"],
        solvent="water",
        model_options={"hessian": "numerical", "dtype": "float64"},
        solvation_options=endpoint["solvation_options"],
        charge_options={
            "source": "mol2",
            "mode": "fixed",
            "geometry": "keep",
            "label": "am1bcc-frozen-route1-foundation-20260913",
        },
    ).set_calculator()


def _raw_force_hessian(atoms: Atoms, step: float) -> tuple[np.ndarray, np.ndarray]:
    calculator = atoms.calc
    if calculator is None:
        raise RuntimeError("stationary point has no calculator")
    positions = atoms.get_positions().copy()
    dimension = 3 * len(atoms)
    raw = np.empty((dimension, dimension), dtype=np.float64)
    try:
        for coordinate in range(dimension):
            plus = positions.reshape(-1).copy()
            minus = positions.reshape(-1).copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            atoms.set_positions(plus.reshape((-1, 3)))
            force_plus = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)
            atoms.set_positions(minus.reshape((-1, 3)))
            force_minus = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(-1)
            raw[:, coordinate] = -(force_plus - force_minus) / (2.0 * step)
    finally:
        atoms.set_positions(positions)
    return raw, 0.5 * (raw + raw.T)


def _optimizer_metrics(atoms: Atoms) -> dict[str, Any]:
    return {
        "maximum_force_hartree_per_angstrom": float(
            np.max(np.abs(atoms.get_forces()))
        ),
        "rms_force_hartree_per_angstrom": float(
            np.sqrt(np.mean(np.square(atoms.get_forces())))
        ),
        "maximum_last_step_angstrom": float(getattr(atoms, "max_dp", np.inf)),
        "rms_last_step_angstrom": float(getattr(atoms, "rms_dp", np.inf)),
    }


def _optimize_downhill(
    saddle: Atoms,
    mode: np.ndarray,
    sign: int,
    attempt_dir: Path,
) -> tuple[Atoms, dict[str, Any]]:
    branch = saddle.copy()
    displacement = np.asarray(mode, dtype=np.float64).reshape((len(branch), 3))
    scale = ATTEMPT_CONFIG.mode_displacement_angstrom / float(
        np.max(np.linalg.norm(displacement, axis=1))
    )
    branch.positions += sign * scale * displacement
    branch.calc = saddle.calc
    _set_stop_thresholds(branch)
    initial_height = _nitrogen_height(branch)
    initial_energy = float(branch.get_potential_energy(force_consistent=True))
    output = attempt_dir / f"downhill-{sign:+d}.out"
    optimized = Optimization(
        {
            "method": ATTEMPT_CONFIG.downhill_method,
            "max_iter": ATTEMPT_CONFIG.downhill_max_iterations,
            "max_step": ATTEMPT_CONFIG.downhill_max_step_angstrom,
            "curvature": ATTEMPT_CONFIG.downhill_curvature,
            "verbose": 1,
        },
        str(output),
        branch,
    ).run()
    final_energy = float(optimized.get_potential_energy(force_consistent=True))
    metrics = _optimizer_metrics(optimized)
    converged = bool(
        metrics["maximum_force_hartree_per_angstrom"]
        <= ATTEMPT_CONFIG.force_max_stop_hartree_per_angstrom
        and metrics["rms_force_hartree_per_angstrom"]
        <= ATTEMPT_CONFIG.force_rms_stop_hartree_per_angstrom
        and metrics["maximum_last_step_angstrom"]
        <= ATTEMPT_CONFIG.step_max_stop_angstrom
        and metrics["rms_last_step_angstrom"]
        <= ATTEMPT_CONFIG.step_rms_stop_angstrom
    )
    return optimized, {
        "sign": sign,
        "displacement_scale": scale,
        "maximum_atom_displacement_angstrom": (
            ATTEMPT_CONFIG.mode_displacement_angstrom
        ),
        "initial_energy_hartree": initial_energy,
        "final_energy_hartree": final_energy,
        "initial_nitrogen_height_angstrom": initial_height,
        "final_nitrogen_height_angstrom": _nitrogen_height(optimized),
        "optimizer_metrics": metrics,
        "optimizer_converged": converged,
        "actual_iterations": _iteration_count(output),
        "internal_coordinates": _nh3_internal_coordinates(optimized),
    }


def _endpoint_attempt(
    source_path: Path,
    source_atoms: Atoms,
    endpoint_name: str,
    endpoint: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    attempt_dir = work_dir / f"ts-{endpoint_name}-{ATTEMPT_CONFIG.attempt_id}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    source_freeze = attempt_dir / "source-freeze"
    source_freeze.mkdir()
    frozen_source_hashes: dict[str, str] = {}
    for relative_source in EXECUTION_SOURCE_PATHS:
        source = REPOSITORY_ROOT / relative_source
        destination = source_freeze / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        frozen_source_hashes[relative_source] = _sha256_file(destination)
    _write_json(
        attempt_dir / "attempt-config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "attempt_configuration": asdict(ATTEMPT_CONFIG),
            "endpoint": endpoint_name,
            "endpoint_configuration": endpoint,
            "source_mol2": str(source_path.relative_to(REPOSITORY_ROOT)),
            "source_mol2_sha256": _sha256_file(source_path),
            "frozen_source_hashes": frozen_source_hashes,
        },
    )
    planar = _planar_ammonia(source_atoms)
    charges = np.asarray(source_atoms.get_initial_charges(), dtype=np.float64)
    planar_mol2 = attempt_dir / "planar-initial.mol2"
    planar_mol2.write_text(
        render_typed_mol2(
            source_path.read_text(encoding="utf-8"),
            planar.get_positions(),
            charges,
        ),
        encoding="utf-8",
    )
    planar = MOL2Reader(str(planar_mol2), charge=0, mult=1, validate_charge=True)
    calculator = _calculator(planar, endpoint, attempt_dir / "calculator.out")
    planar.calc = calculator
    _set_stop_thresholds(planar)

    provider = calculator.solvent_correction.provider
    provider_charges = np.asarray(provider.charges, dtype=np.float64)
    radius_values = getattr(provider, "radii", None)
    if radius_values is None:
        radius_values = provider.radius_result.radii_angstrom
    provider_radii = np.asarray(radius_values, dtype=np.float64)
    identity_before = {
        "charges_sha256": _sha256_array(provider_charges),
        "radii_sha256": _sha256_array(provider_radii),
    }

    initial_energy = float(planar.get_potential_energy(force_consistent=True))
    initial_forces = np.asarray(planar.get_forces(), dtype=np.float64)
    prfo_output = attempt_dir / "prfo.out"
    prfo = PRFO(
        str(prfo_output),
        planar,
        paras={
            "max_iter": ATTEMPT_CONFIG.prfo_max_iterations,
            "recalc": ATTEMPT_CONFIG.prfo_recalc,
            "trust_radius": ATTEMPT_CONFIG.prfo_trust_radius,
            "trust_min": ATTEMPT_CONFIG.prfo_trust_min,
            "trust_max": ATTEMPT_CONFIG.prfo_trust_max,
            "hessian_update": ATTEMPT_CONFIG.prfo_hessian_update,
            "project_rigid_modes": ATTEMPT_CONFIG.prfo_project_rigid_modes,
        },
    )
    if getattr(prfo.params, "project_rigid_modes", None) is not True:
        raise RuntimeError("PRFO rigid-mode projection repair is not available")
    optimized = prfo.run()
    final_energy = float(optimized.get_potential_energy(force_consistent=True))
    final_forces = np.asarray(optimized.get_forces(), dtype=np.float64)
    optimizer_metrics = _optimizer_metrics(optimized)
    optimizer_converged = bool(
        optimizer_metrics["maximum_force_hartree_per_angstrom"]
        <= ATTEMPT_CONFIG.force_max_stop_hartree_per_angstrom
        and optimizer_metrics["rms_force_hartree_per_angstrom"]
        <= ATTEMPT_CONFIG.force_rms_stop_hartree_per_angstrom
        and optimizer_metrics["maximum_last_step_angstrom"]
        <= ATTEMPT_CONFIG.step_max_stop_angstrom
        and optimizer_metrics["rms_last_step_angstrom"]
        <= ATTEMPT_CONFIG.step_rms_stop_angstrom
    )
    prfo_iterations = _iteration_count(prfo_output)

    hessian_records: list[dict[str, Any]] = []
    raw_hessians: list[np.ndarray] = []
    symmetric_hessians: list[np.ndarray] = []
    analyses: list[dict[str, Any]] = []
    for step in (
        ATTEMPT_CONFIG.hessian_step_large_angstrom,
        ATTEMPT_CONFIG.hessian_step_small_angstrom,
    ):
        raw, symmetric = _raw_force_hessian(optimized, step)
        label = f"{step:.8f}".rstrip("0")
        raw_path = attempt_dir / f"hessian-raw-step-{label}-angstrom.npy"
        symmetric_path = attempt_dir / f"hessian-symmetric-step-{label}-angstrom.npy"
        np.save(raw_path, raw)
        np.save(symmetric_path, symmetric)
        analysis = analyze_stationary_point(
            optimized,
            raw,
            final_forces,
            negative_cutoff_cm1=ATTEMPT_CONFIG.negative_cutoff_cm1,
        )
        analyses.append(analysis)
        raw_hessians.append(raw)
        symmetric_hessians.append(symmetric)
        hessian_records.append(
            {
                "step_angstrom": step,
                "raw_matrix": str(raw_path.relative_to(REPOSITORY_ROOT)),
                "raw_matrix_sha256": _sha256_file(raw_path),
                "symmetric_matrix": str(symmetric_path.relative_to(REPOSITORY_ROOT)),
                "symmetric_matrix_sha256": _sha256_file(symmetric_path),
                "maximum_absolute_raw_hessian_hartree_per_angstrom2": float(
                    np.max(np.abs(raw))
                ),
                "relative_max_raw_hessian_asymmetry": (
                    float(np.max(np.abs(raw - raw.T)) / np.max(np.abs(raw)))
                    if np.max(np.abs(raw)) > 0.0
                    else 0.0
                ),
                "analysis": analysis,
            }
        )

    raw_step_change = float(np.max(np.abs(raw_hessians[0] - raw_hessians[1])))
    symmetric_step_change = float(
        np.max(np.abs(symmetric_hessians[0] - symmetric_hessians[1]))
    )
    hessian_scale = max(
        record["maximum_absolute_raw_hessian_hartree_per_angstrom2"]
        for record in hessian_records
    )
    hessian_budget = (
        ATTEMPT_CONFIG.hessian_absolute_budget_hartree_per_angstrom2
        + ATTEMPT_CONFIG.hessian_relative_budget_factor * hessian_scale
    )
    raw_asymmetry_pass = all(
        record["analysis"][
            "maximum_raw_hessian_asymmetry_hartree_per_angstrom2"
        ]
        <= ATTEMPT_CONFIG.hessian_absolute_budget_hartree_per_angstrom2
        + ATTEMPT_CONFIG.hessian_relative_budget_factor
        * record["maximum_absolute_raw_hessian_hartree_per_angstrom2"]
        for record in hessian_records
    )
    hessian_step_pass = bool(raw_step_change <= hessian_budget)

    mode0 = analyses[0]["negative_mode_cartesian"]
    mode1 = analyses[1]["negative_mode_cartesian"]
    mode_overlap = None
    umbrella_overlaps: list[float | None] = []
    umbrella_reference = _umbrella_reference_mode(optimized)
    for analysis in analyses:
        negative_mode = analysis["negative_mode_cartesian"]
        umbrella_overlaps.append(
            _mass_metric_overlap(optimized, negative_mode, umbrella_reference)
            if negative_mode is not None
            else None
        )
    if mode0 is not None and mode1 is not None:
        mode_overlap = _mass_metric_overlap(optimized, mode0, mode1)
    stable_index_one = bool(
        all(analysis["is_first_order_saddle"] for analysis in analyses)
        and hessian_step_pass
        and raw_asymmetry_pass
        and mode_overlap is not None
        and mode_overlap >= ATTEMPT_CONFIG.minimum_mass_metric_mode_overlap
        and all(
            overlap is not None
            and overlap >= ATTEMPT_CONFIG.minimum_mass_metric_mode_overlap
            for overlap in umbrella_overlaps
        )
    )

    downhill_records: list[dict[str, Any]] = []
    downhill_atoms: list[Atoms] = []
    if analyses[1]["negative_mode_cartesian"] is not None:
        for sign in (-1, 1):
            optimized_downhill, downhill = _optimize_downhill(
                optimized,
                analyses[1]["negative_mode_cartesian"],
                sign,
                attempt_dir,
            )
            downhill_atoms.append(optimized_downhill)
            downhill_records.append(downhill)
    mode_matches_umbrella = bool(
        len(umbrella_overlaps) == 2
        and all(
            overlap is not None
            and overlap >= ATTEMPT_CONFIG.minimum_mass_metric_mode_overlap
            for overlap in umbrella_overlaps
        )
    )
    equivalent_downhill_endpoints = False
    endpoint_comparison: dict[str, Any] = {}
    if len(downhill_atoms) == 2:
        first_internal = _nh3_internal_coordinates(downhill_atoms[0])
        second_internal = _nh3_internal_coordinates(downhill_atoms[1])
        bond_difference = float(
            np.max(
                np.abs(
                    first_internal["sorted_nh_bond_lengths_angstrom"]
                    - second_internal["sorted_nh_bond_lengths_angstrom"]
                )
            )
        )
        angle_difference = float(
            np.max(
                np.abs(
                    first_internal["sorted_hnh_angles_degrees"]
                    - second_internal["sorted_hnh_angles_degrees"]
                )
            )
        )
        energy_difference = abs(
            downhill_records[0]["final_energy_hartree"]
            - downhill_records[1]["final_energy_hartree"]
        )
        endpoint_comparison = {
            "maximum_sorted_nh_bond_difference_angstrom": bond_difference,
            "maximum_sorted_hnh_angle_difference_degrees": angle_difference,
            "absolute_energy_difference_hartree": energy_difference,
        }
        equivalent_downhill_endpoints = bool(
            first_internal["intact"]
            and second_internal["intact"]
            and bond_difference
            <= ATTEMPT_CONFIG.equivalent_nh_bond_tolerance_angstrom
            and angle_difference
            <= ATTEMPT_CONFIG.equivalent_hnh_angle_tolerance_degrees
            and energy_difference
            <= ATTEMPT_CONFIG.equivalent_energy_tolerance_hartree
        )
    downhill_pass = bool(
        mode_matches_umbrella
        and all(record["optimizer_converged"] for record in downhill_records)
        and all(record["final_energy_hartree"] < final_energy for record in downhill_records)
        and downhill_records[0]["final_nitrogen_height_angstrom"]
        * downhill_records[1]["final_nitrogen_height_angstrom"]
        < 0.0
        and equivalent_downhill_endpoints
    )

    identity_after = {
        "charges_sha256": _sha256_array(np.asarray(provider.charges)),
        "radii_sha256": _sha256_array(
            np.asarray(
                getattr(provider, "radii", provider.radius_result.radii_angstrom)
            )
        ),
    }
    precision = dict(calculator.inference_precision_provenance)
    result = {
        "protocol_id": PROTOCOL_ID,
        "attempt_id": ATTEMPT_CONFIG.attempt_id,
        "status": "completed",
        "acceptance_eligible": True,
        "attempt_configuration": asdict(ATTEMPT_CONFIG),
        "endpoint": endpoint_name,
        "endpoint_configuration": endpoint,
        "source_mol2": str(source_path.relative_to(REPOSITORY_ROOT)),
        "source_mol2_sha256": _sha256_file(source_path),
        "planar_mol2": str(planar_mol2.relative_to(REPOSITORY_ROOT)),
        "planar_mol2_sha256": _sha256_file(planar_mol2),
        "initial_energy_hartree": initial_energy,
        "initial_forces_hartree_per_angstrom": initial_forces,
        "final_energy_hartree": final_energy,
        "final_forces_hartree_per_angstrom": final_forces,
        "optimizer_metrics": optimizer_metrics,
        "optimizer_converged": optimizer_converged,
        "actual_prfo_iterations": prfo_iterations,
        "optimizer_limits": {
            "maximum_iterations": 80,
            "force_max_hartree_per_angstrom": (
                ATTEMPT_CONFIG.force_max_stop_hartree_per_angstrom
            ),
            "force_rms_hartree_per_angstrom": (
                ATTEMPT_CONFIG.force_rms_stop_hartree_per_angstrom
            ),
            "step_max_angstrom": ATTEMPT_CONFIG.step_max_stop_angstrom,
            "step_rms_angstrom": ATTEMPT_CONFIG.step_rms_stop_angstrom,
        },
        "hessian_records": hessian_records,
        "maximum_raw_hessian_step_change_hartree_per_angstrom2": raw_step_change,
        "relative_raw_hessian_step_change": raw_step_change / hessian_scale,
        "maximum_symmetric_hessian_step_change_hartree_per_angstrom2": (
            symmetric_step_change
        ),
        "hessian_change_budget_hartree_per_angstrom2": hessian_budget,
        "hessian_step_pass": hessian_step_pass,
        "raw_asymmetry_pass": raw_asymmetry_pass,
        "negative_mode_cartesian_overlap_between_steps": mode_overlap,
        "negative_mode_umbrella_mass_metric_overlaps": umbrella_overlaps,
        "minimum_required_mass_metric_mode_overlap": (
            ATTEMPT_CONFIG.minimum_mass_metric_mode_overlap
        ),
        "stable_index_one": stable_index_one,
        "mode_matches_ammonia_umbrella": mode_matches_umbrella,
        "downhill_optimizations": downhill_records,
        "downhill_endpoint_comparison": endpoint_comparison,
        "equivalent_intact_downhill_endpoints": equivalent_downhill_endpoints,
        "downhill_connection_pass": downhill_pass,
        "input_identity_before": identity_before,
        "input_identity_after": identity_after,
        "input_identity_unchanged": identity_before == identity_after,
        "inference_precision": precision,
        "provider_provenance": provider.provenance,
        "source_hashes": frozen_source_hashes,
    }
    live_source_hashes_after = {
        path: _sha256_file(REPOSITORY_ROOT / path)
        for path in EXECUTION_SOURCE_PATHS
    }
    result["live_source_hashes_after"] = live_source_hashes_after
    result["source_unchanged_during_attempt"] = (
        live_source_hashes_after == frozen_source_hashes
    )
    result["accepted"] = bool(
        optimizer_converged
        and stable_index_one
        and downhill_pass
        and result["input_identity_unchanged"]
        and precision.get("effective_dtype") == "float64"
        and result["source_unchanged_during_attempt"]
    )
    result["output_hashes"] = {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path)
        for path in sorted(attempt_dir.rglob("*"))
        if path.is_file() and path.name != "result.json"
    }
    _write_json(attempt_dir / "result.json", result)
    return result


def run(
    source_path: Path,
    work_dir: Path,
    endpoint_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    source_atoms = MOL2Reader(
        str(source_path), charge=0, mult=1, validate_charge=True
    )
    selected_endpoints = ENDPOINTS
    if endpoint_names is not None:
        selected_endpoints = {name: ENDPOINTS[name] for name in endpoint_names}
    summary: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "attempt_id": ATTEMPT_CONFIG.attempt_id,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "source_mol2": str(source_path.relative_to(REPOSITORY_ROOT)),
        "source_mol2_sha256": _sha256_file(source_path),
        "frozen_denominator": {
            "attempt_configuration": asdict(ATTEMPT_CONFIG),
            "source_mol2_sha256": _sha256_file(source_path),
            "execution_source_hashes": {
                path: _sha256_file(REPOSITORY_ROOT / path)
                for path in EXECUTION_SOURCE_PATHS
            },
        },
        "required_endpoints": list(ENDPOINTS),
        "selected_endpoints": list(selected_endpoints),
        "attempts": [],
    }
    for endpoint_name, endpoint in selected_endpoints.items():
        try:
            summary["attempts"].append(
                _endpoint_attempt(
                    source_path,
                    source_atoms,
                    endpoint_name,
                    endpoint,
                    work_dir,
                )
            )
        except FileExistsError:
            raise
        except Exception as exc:  # noqa: BLE001 - preserve every real attempt.
            failure_dir = (
                work_dir / f"ts-{endpoint_name}-{ATTEMPT_CONFIG.attempt_id}"
            )
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "protocol_id": PROTOCOL_ID,
                "attempt_id": ATTEMPT_CONFIG.attempt_id,
                "status": "failed",
                "acceptance_eligible": True,
                "attempt_configuration": asdict(ATTEMPT_CONFIG),
                "endpoint": endpoint_name,
                "source_mol2": str(source_path.relative_to(REPOSITORY_ROOT)),
                "source_mol2_sha256": _sha256_file(source_path),
                "source_hashes": summary["frozen_denominator"],
                "accepted": False,
                "failure": {
                    "exception_class": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "partial_evidence": _partial_attempt_evidence(failure_dir),
            }
            summary["attempts"].append(failure)
            _write_json(failure_dir / "result.json", failure)
    summary["coverage_complete"] = set(selected_endpoints) == set(ENDPOINTS)
    summary["selected_all_accepted"] = bool(
        len(summary["attempts"]) == len(selected_endpoints)
        and all(attempt.get("accepted") is True for attempt in summary["attempts"])
    )
    completed_attempts = [
        attempt for attempt in summary["attempts"] if attempt.get("status") == "completed"
    ]
    checkpoint_hashes = {
        attempt.get("inference_precision", {}).get("original_checkpoint_sha256")
        for attempt in completed_attempts
    }
    charge_hashes = {
        attempt.get("input_identity_before", {}).get("charges_sha256")
        for attempt in completed_attempts
    }
    radii_hashes = {
        attempt.get("input_identity_before", {}).get("radii_sha256")
        for attempt in completed_attempts
    }
    precision_pass = all(
        attempt.get("inference_precision", {}).get("requested_dtype") == "float64"
        and attempt.get("inference_precision", {}).get("effective_dtype") == "float64"
        and attempt.get("inference_precision", {}).get("numerical_curvature_prepared")
        is True
        for attempt in completed_attempts
    )
    summary["cross_endpoint_identity"] = {
        "checkpoint_sha256_values": sorted(str(value) for value in checkpoint_hashes),
        "charges_sha256_values": sorted(str(value) for value in charge_hashes),
        "radii_sha256_values": sorted(str(value) for value in radii_hashes),
        "precision_pass": precision_pass,
    }
    summary["cross_endpoint_identity_pass"] = bool(
        len(completed_attempts) == len(ENDPOINTS)
        and len(checkpoint_hashes) == 1
        and None not in checkpoint_hashes
        and len(charge_hashes) == 1
        and None not in charge_hashes
        and len(radii_hashes) == 1
        and None not in radii_hashes
        and precision_pass
    )
    summary["all_required_endpoints_accepted"] = bool(
        summary["coverage_complete"]
        and summary["selected_all_accepted"]
        and summary["cross_endpoint_identity_pass"]
    )
    selected_label = "--".join(selected_endpoints)
    _write_json(
        work_dir
        / f"ts-summary-{ATTEMPT_CONFIG.attempt_id}--{selected_label}.json",
        summary,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--endpoint", action="append", choices=tuple(ENDPOINTS))
    args = parser.parse_args()
    summary = run(
        args.input,
        args.work_dir,
        tuple(args.endpoint) if args.endpoint else None,
    )
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True))
    return 0 if summary["all_required_endpoints_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
