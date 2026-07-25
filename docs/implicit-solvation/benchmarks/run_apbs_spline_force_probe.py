#!/usr/bin/env python3
"""Probe APBS SPL4 polar forces without changing the Route-1 product provider."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.apbs_pb import (  # noqa: E402
    APBSLPB,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

DEFAULT_MOL2 = SCRIPT_DIR / "route1-multi-mlip-obc2-ti-raw/inputs/mobley_1017962.mol2"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-apbs-spline-force-probe-methyl-hexanoate-2026-07-25.json"
)
DEFAULT_WORK_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/apbs-spline-force-probe-route1-20260725"
)
DEFAULT_APBS = "/tmp/APBS-3.4.1.Linux-rmODq5/APBS-3.4.1.Linux/bin/apbs"
DEFAULT_RELEASE_ARCHIVE = "/tmp/APBS-3.4.1.Linux.zip"
APBS_RELEASE_URL = (
    "https://github.com/Electrostatics/apbs/releases/download/v3.4.1/"
    "APBS-3.4.1.Linux.zip"
)
EXPECTED_RELEASE_ARCHIVE_SHA256 = (
    "750f6a2df7b5a82b69be5c4cb192115c02fa261d0ff52cf1d9632ed1eda8b4f0"
)
EXPECTED_APBS_EXECUTABLE_SHA256 = (
    "6ebdacce26e31aa01cd221a534147218088de9d55ff111a1c8bbea56a60a32bd"
)
EXPECTED_MOL2_SHA256 = (
    "783df578c85819b79b59537197a0850e074426d2e9eef5bfef020c9a4849ce23"
)
GRIDS = (
    {"name": "spl4-129x0.25", "points": 129, "spacing_angstrom": 0.25},
    {"name": "spl4-161x0.20", "points": 161, "spacing_angstrom": 0.20},
)
FD_STEPS_ANGSTROM = (0.003, 0.010)
AXES = ("x", "y", "z")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def parse_apbs_version(text: str) -> str:
    match = re.search(
        r"\bAPBS(?:\s+version)?\s+v?([0-9]+(?:\.[0-9]+)+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(r"(?m)^\s*([0-9]+(?:\.[0-9]+)+)\s*$", text)
    if match:
        return match.group(1)
    raise ValueError("Could not parse the APBS version.")


def parse_polar_print_section(
    output: str, *, expected_atoms: int, require_forces: bool
) -> dict[str, Any]:
    try:
        section = output.split("PRINT STATEMENTS", 1)[1]
    except IndexError as exc:
        raise ValueError("APBS output has no PRINT STATEMENTS section.") from exc
    energy_match = re.search(
        rf"Global\s+net\s+ELEC\s+energy\s*=\s*({FLOAT})\s*kJ/mol",
        section,
        re.IGNORECASE,
    )
    if energy_match is None:
        raise ValueError("APBS output has no printed polar energy difference.")
    parsed: dict[str, Any] = {"polar_energy_kj_mol": float(energy_match.group(1))}
    if not require_forces:
        return parsed
    if not re.search(
        r"print\s+force\s+1\s+\(solv\)\s+-\s+2\s+\(ref\)\s+end",
        section,
        re.IGNORECASE,
    ):
        raise ValueError("APBS output has no printed solvated-minus-reference force.")
    rows = re.findall(
        rf"^\s*tot\s+(\d+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$",
        section,
        re.MULTILINE,
    )
    indices = [int(row[0]) for row in rows]
    if indices != list(range(expected_atoms)):
        raise ValueError(
            "APBS printed force atom indices are incomplete or out of order: "
            f"{indices!r}."
        )
    parsed["polar_force_kj_mol_angstrom"] = [
        [float(value) for value in row[1:]] for row in rows
    ]
    return parsed


def render_polar_input(
    pqr_name: str,
    *,
    points: int,
    spacing_angstrom: float,
    require_forces: bool,
    surface: str = "spl4",
) -> str:
    if surface not in {"mol", "spl4"}:
        raise ValueError(f"Unsupported APBS probe surface: {surface!r}.")
    force_mode = "comps" if require_forces else "no"
    shared = f"""\
    mg-manual
    dime {points} {points} {points}
    nlev 4
    grid {spacing_angstrom:.6f} {spacing_angstrom:.6f} {spacing_angstrom:.6f}
    gcent mol 1
    mol 1
    lpbe
    bcfl mdh
    pdie 1.0
    chgm spl2
    srfm {surface}
    srad 1.400000
    swin 0.3
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce {force_mode}"""
    force_print = "print elecForce solv - ref end\n" if require_forces else ""
    return f"""\
read
    mol pqr {pqr_name}
end
elec name solv
{shared}
    sdie 78.5
end
elec name ref
{shared}
    sdie 1.0
end
print elecEnergy solv - ref end
{force_print}quit
"""


def _resolve_executable(value: str) -> Path:
    resolved = shutil.which(value)
    path = Path(resolved if resolved is not None else value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"APBS executable was not found: {value}")
    return path


def inspect_apbs(executable: Path, *, allow_other_binary: bool) -> dict[str, str]:
    executable_hash = sha256_file(executable)
    completed = subprocess.run(
        [str(executable), "--version"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    version_text = completed.stdout + "\n" + completed.stderr
    version = parse_apbs_version(version_text)
    if not allow_other_binary and (
        version != "3.4.1" or executable_hash != EXPECTED_APBS_EXECUTABLE_SHA256
    ):
        raise ValueError(
            "This frozen probe requires the official APBS 3.4.1 Linux binary. "
            "Use --allow-other-binary only for a separately labelled comparison."
        )
    return {
        "path": str(executable),
        "version": version,
        "sha256": executable_hash,
    }


def _positions_sha256(positions: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(positions, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _job_signature(
    *,
    positions: np.ndarray,
    grid: dict[str, Any],
    require_forces: bool,
    executable_sha256: str,
    molecule_sha256: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "positions_sha256": _positions_sha256(positions),
                "grid": grid,
                "surface": "spl4",
                "charge_mapping": "spl2",
                "require_forces": require_forces,
                "executable_sha256": executable_sha256,
                "molecule_sha256": molecule_sha256,
            }
        )
    )


def run_job(
    *,
    atoms,
    charges: np.ndarray,
    positions: np.ndarray,
    grid: dict[str, Any],
    require_forces: bool,
    executable: Path,
    executable_sha256: str,
    molecule_sha256: str,
    job_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    signature = _job_signature(
        positions=positions,
        grid=grid,
        require_forces=require_forces,
        executable_sha256=executable_sha256,
        molecule_sha256=molecule_sha256,
    )
    result_path = job_dir / "result.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("job_signature") == signature:
            return result
        raise ValueError(f"Incompatible cached APBS job: {result_path}")

    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=False)
    job_atoms = atoms.copy()
    job_atoms.positions[:] = positions
    provider = APBSLPB(
        job_atoms,
        charges,
        executable=str(executable),
        grid_spacing=float(grid["spacing_angstrom"]),
        grid_points=int(grid["points"]),
        timeout=timeout,
    )
    pqr = job_dir / "molecule.pqr"
    inp = job_dir / "apbs.in"
    stdout_path = job_dir / "apbs.stdout.log"
    stderr_path = job_dir / "apbs.stderr.log"
    provider.write_pqr(pqr, job_atoms)
    inp.write_text(
        render_polar_input(
            pqr.name,
            points=int(grid["points"]),
            spacing_angstrom=float(grid["spacing_angstrom"]),
            require_forces=require_forces,
        ),
        encoding="utf-8",
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable), inp.name],
        cwd=job_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-1200:].strip()
        raise RuntimeError(f"APBS failed with code {completed.returncode}: {detail}")
    parsed = parse_polar_print_section(
        completed.stdout,
        expected_atoms=len(job_atoms),
        require_forces=require_forces,
    )
    result = {
        "schema_version": 1,
        "job_signature": signature,
        "positions_sha256": _positions_sha256(positions),
        "grid": grid,
        "surface": "spl4",
        "charge_mapping": "spl2",
        "require_forces": require_forces,
        "elapsed_seconds": elapsed,
        "input_sha256": sha256_file(inp),
        "pqr_sha256": sha256_file(pqr),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        **parsed,
    }
    write_json_atomic(result_path, result)
    return result


def run_molecular_surface_force_rejection(
    *,
    atoms,
    charges: np.ndarray,
    executable: Path,
    executable_sha256: str,
    molecule_sha256: str,
    work_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    job_dir = work_dir / "molecular-surface-force-rejection"
    signature = sha256_bytes(
        canonical_json_bytes(
            {
                "surface": "mol",
                "require_forces": True,
                "grid": GRIDS[0],
                "executable_sha256": executable_sha256,
                "molecule_sha256": molecule_sha256,
            }
        )
    )
    result_path = job_dir / "result.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("job_signature") == signature:
            return result
        raise ValueError(
            f"Incompatible cached APBS molecular-surface job: {result_path}"
        )
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    provider = APBSLPB(
        atoms,
        charges,
        executable=str(executable),
        grid_spacing=float(GRIDS[0]["spacing_angstrom"]),
        grid_points=int(GRIDS[0]["points"]),
        timeout=timeout,
    )
    pqr = job_dir / "molecule.pqr"
    inp = job_dir / "apbs.in"
    stdout_path = job_dir / "apbs.stdout.log"
    stderr_path = job_dir / "apbs.stderr.log"
    provider.write_pqr(pqr, atoms)
    inp.write_text(
        render_polar_input(
            pqr.name,
            points=int(GRIDS[0]["points"]),
            spacing_angstrom=float(GRIDS[0]["spacing_angstrom"]),
            require_forces=True,
            surface="mol",
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(executable), inp.name],
        cwd=job_dir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    combined = completed.stdout + "\n" + completed.stderr
    rejection_message = "Forces *must* be calculated with spline-based surfaces!"
    rejected = completed.returncode != 0 and rejection_message in combined
    if not rejected:
        raise RuntimeError(
            "The pinned APBS binary did not reproduce the expected molecular-"
            "surface force rejection."
        )
    result = {
        "schema_version": 1,
        "job_signature": signature,
        "surface": "mol",
        "require_forces": True,
        "returncode": completed.returncode,
        "rejection_message": rejection_message,
        "rejected_by_apbs": True,
        "input_sha256": sha256_file(inp),
        "pqr_sha256": sha256_file(pqr),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    write_json_atomic(result_path, result)
    return result


def _fd_job_name(
    grid_name: str, step: float, atom_index: int, axis_index: int, sign: int
) -> str:
    step_text = f"{step:.3f}".replace(".", "p")
    direction = "plus" if sign > 0 else "minus"
    return (
        f"{grid_name}/fd-{step_text}/"
        f"atom-{atom_index:03d}-{AXES[axis_index]}-{direction}"
    )


def _error_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray(
        [row["signed_error_kj_mol_angstrom"] for row in rows],
        dtype=np.float64,
    )
    absolute = np.abs(errors)
    largest = int(np.argmax(absolute))
    return {
        "component_count": len(rows),
        "mean_signed_error_kj_mol_angstrom": float(np.mean(errors)),
        "mean_absolute_error_kj_mol_angstrom": float(np.mean(absolute)),
        "rmse_kj_mol_angstrom": float(np.sqrt(np.mean(errors * errors))),
        "maximum_absolute_error_kj_mol_angstrom": float(absolute[largest]),
        "largest_error_component": {
            key: rows[largest][key]
            for key in (
                "atom_index_zero_based",
                "element",
                "axis",
                "analytical_force_kj_mol_angstrom",
                "finite_difference_force_kj_mol_angstrom",
                "signed_error_kj_mol_angstrom",
            )
        },
    }


def _cross_grid_metrics(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    coarse_force = np.asarray(coarse["polar_force_kj_mol_angstrom"], dtype=np.float64)
    fine_force = np.asarray(fine["polar_force_kj_mol_angstrom"], dtype=np.float64)
    difference = fine_force - coarse_force
    absolute = np.abs(difference)
    largest = np.unravel_index(int(np.argmax(absolute)), difference.shape)
    return {
        "fine_minus_coarse_energy_kj_mol": (
            float(fine["polar_energy_kj_mol"]) - float(coarse["polar_energy_kj_mol"])
        ),
        "force_mean_absolute_difference_kj_mol_angstrom": float(np.mean(absolute)),
        "force_rmse_difference_kj_mol_angstrom": float(
            np.sqrt(np.mean(difference * difference))
        ),
        "force_maximum_absolute_difference_kj_mol_angstrom": float(absolute[largest]),
        "largest_force_difference_component": {
            "atom_index_zero_based": int(largest[0]),
            "axis": AXES[int(largest[1])],
            "coarse_force_kj_mol_angstrom": float(coarse_force[largest]),
            "fine_force_kj_mol_angstrom": float(fine_force[largest]),
            "fine_minus_coarse_kj_mol_angstrom": float(difference[largest]),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mol2 = Path(args.mol2).resolve()
    if not mol2.is_file():
        raise FileNotFoundError(mol2)
    molecule_hash = sha256_file(mol2)
    if molecule_hash != EXPECTED_MOL2_SHA256:
        raise ValueError("The frozen methyl-hexanoate MOL2 hash does not match.")

    executable = _resolve_executable(args.apbs)
    apbs = inspect_apbs(executable, allow_other_binary=bool(args.allow_other_binary))
    archive_record: dict[str, Any] | None = None
    if args.release_archive:
        archive = Path(args.release_archive).resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archive_hash = sha256_file(archive)
        if (
            not args.allow_other_binary
            and archive_hash != EXPECTED_RELEASE_ARCHIVE_SHA256
        ):
            raise ValueError("The APBS 3.4.1 release archive hash does not match.")
        archive_record = {
            "path": str(archive),
            "sha256": archive_hash,
            "source_url": APBS_RELEASE_URL,
        }

    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
        raise ValueError("The MOL2 must contain one finite AM1-BCC charge per atom.")
    if not math.isclose(float(np.sum(charges)), 0.0, abs_tol=1.0e-6):
        raise ValueError("The frozen neutral molecule has a nonzero charge sum.")
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    molecular_surface_rejection = run_molecular_surface_force_rejection(
        atoms=atoms,
        charges=charges,
        executable=executable,
        executable_sha256=apbs["sha256"],
        molecule_sha256=molecule_hash,
        work_dir=work_dir,
        timeout=args.timeout,
    )

    jobs: list[dict[str, Any]] = []
    for grid in GRIDS:
        jobs.append(
            {
                "name": f"{grid['name']}/base-force",
                "grid": grid,
                "positions": positions,
                "require_forces": True,
            }
        )
        for step in FD_STEPS_ANGSTROM:
            for atom_index in range(len(atoms)):
                for axis_index in range(3):
                    for sign in (-1, 1):
                        displaced = positions.copy()
                        displaced[atom_index, axis_index] += sign * step
                        jobs.append(
                            {
                                "name": _fd_job_name(
                                    grid["name"],
                                    step,
                                    atom_index,
                                    axis_index,
                                    sign,
                                ),
                                "grid": grid,
                                "positions": displaced,
                                "require_forces": False,
                            }
                        )

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job,
                atoms=atoms,
                charges=charges,
                positions=job["positions"],
                grid=job["grid"],
                require_forces=job["require_forces"],
                executable=executable,
                executable_sha256=apbs["sha256"],
                molecule_sha256=molecule_hash,
                job_dir=work_dir / job["name"],
                timeout=args.timeout,
            ): job["name"]
            for job in jobs
        }
        completed_count = 0
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            completed_count += 1
            if completed_count % 25 == 0 or completed_count == len(jobs):
                print(
                    f"APBS SPL4 probe: {completed_count}/{len(jobs)} jobs ready.",
                    flush=True,
                )

    symbols = atoms.get_chemical_symbols()
    grid_results: list[dict[str, Any]] = []
    for grid in GRIDS:
        base_name = f"{grid['name']}/base-force"
        base = results[base_name]
        analytical = np.asarray(base["polar_force_kj_mol_angstrom"], dtype=np.float64)
        fd_results: dict[str, Any] = {}
        for step in FD_STEPS_ANGSTROM:
            component_rows: list[dict[str, Any]] = []
            for atom_index in range(len(atoms)):
                for axis_index, axis in enumerate(AXES):
                    minus_name = _fd_job_name(
                        grid["name"], step, atom_index, axis_index, -1
                    )
                    plus_name = _fd_job_name(
                        grid["name"], step, atom_index, axis_index, 1
                    )
                    minus_energy = float(results[minus_name]["polar_energy_kj_mol"])
                    plus_energy = float(results[plus_name]["polar_energy_kj_mol"])
                    finite_difference = -(plus_energy - minus_energy) / (2.0 * step)
                    value = float(analytical[atom_index, axis_index])
                    component_rows.append(
                        {
                            "atom_index_zero_based": atom_index,
                            "element": symbols[atom_index],
                            "axis": axis,
                            "step_angstrom": step,
                            "minus_energy_kj_mol": minus_energy,
                            "plus_energy_kj_mol": plus_energy,
                            "analytical_force_kj_mol_angstrom": value,
                            "finite_difference_force_kj_mol_angstrom": (
                                finite_difference
                            ),
                            "signed_error_kj_mol_angstrom": (value - finite_difference),
                        }
                    )
            fd_results[f"{step:.3f}"] = {
                **_error_metrics(component_rows),
                "components": component_rows,
            }
        net_force = np.sum(analytical, axis=0)
        grid_results.append(
            {
                "grid": grid,
                "polar_energy_kj_mol": base["polar_energy_kj_mol"],
                "polar_force_kj_mol_angstrom": (base["polar_force_kj_mol_angstrom"]),
                "net_force_kj_mol_angstrom": net_force.tolist(),
                "net_force_norm_kj_mol_angstrom": float(np.linalg.norm(net_force)),
                "base_force_elapsed_seconds": base["elapsed_seconds"],
                "finite_difference": fd_results,
            }
        )

    cross_grid = _cross_grid_metrics(
        results[f"{GRIDS[0]['name']}/base-force"],
        results[f"{GRIDS[1]['name']}/base-force"],
    )
    thresholds = {
        "finite_difference_rmse_kj_mol_angstrom": 0.05,
        "finite_difference_maximum_kj_mol_angstrom": 0.25,
        "cross_grid_force_rmse_kj_mol_angstrom": 0.10,
        "cross_grid_force_maximum_kj_mol_angstrom": 0.50,
        "net_force_norm_kj_mol_angstrom": 0.10,
        "cross_grid_energy_difference_kj_mol": 0.20,
    }
    checks: dict[str, bool] = {}
    for grid_result in grid_results:
        for step, fd in grid_result["finite_difference"].items():
            prefix = f"{grid_result['grid']['name']}-fd-{step}"
            checks[f"{prefix}-rmse"] = (
                fd["rmse_kj_mol_angstrom"]
                <= thresholds["finite_difference_rmse_kj_mol_angstrom"]
            )
            checks[f"{prefix}-maximum"] = (
                fd["maximum_absolute_error_kj_mol_angstrom"]
                <= thresholds["finite_difference_maximum_kj_mol_angstrom"]
            )
        checks[f"{grid_result['grid']['name']}-net-force"] = (
            grid_result["net_force_norm_kj_mol_angstrom"]
            <= thresholds["net_force_norm_kj_mol_angstrom"]
        )
    checks["cross-grid-force-rmse"] = (
        cross_grid["force_rmse_difference_kj_mol_angstrom"]
        <= thresholds["cross_grid_force_rmse_kj_mol_angstrom"]
    )
    checks["cross-grid-force-maximum"] = (
        cross_grid["force_maximum_absolute_difference_kj_mol_angstrom"]
        <= thresholds["cross_grid_force_maximum_kj_mol_angstrom"]
    )
    checks["cross-grid-energy"] = (
        abs(cross_grid["fine_minus_coarse_energy_kj_mol"])
        <= thresholds["cross_grid_energy_difference_kj_mol"]
    )
    numerical_gate_passed = all(checks.values())

    job_hashes = {
        name: sha256_file(work_dir / name / "result.json") for name in sorted(results)
    }
    elapsed_values = np.asarray(
        [float(result["elapsed_seconds"]) for result in results.values()],
        dtype=np.float64,
    )
    payload = {
        "schema_version": 1,
        "recorded_date": "2026-07-25",
        "artifact_type": "route1-apbs-spline-polar-force-probe",
        "claim_scope": (
            "One-molecule, polar-only numerical capability probe of APBS SPL4 "
            "forces. It does not use hydration labels, does not benchmark "
            "solvation accuracy, and does not promote a production provider."
        ),
        "route1_contract": {
            "name": "Additive fixed-charge PB/GB implicit solvation",
            "role": "Baseline/Product Route",
            "formula": (
                "E_solution(R) = E_MLIP,gas(R) + " "G_polar(R,q_fixed) + G_nonpolar(R)"
            ),
            "gas_phase_mm_energy": False,
            "mlip_retraining": False,
            "hydration_label_fit_or_residual": False,
            "fixed_charge_source": "AM1-BCC charges frozen in the MOL2",
        },
        "method": {
            "equation": "linearized Poisson-Boltzmann",
            "polar_surface": "spl4",
            "charge_mapping": "spl2",
            "radius_profile": "OpenMM mbondi2",
            "spline_radius_reparameterization_validated": False,
            "solute_dielectric": 1.0,
            "solvent_dielectric": 78.5,
            "ionic_strength": 0.0,
            "nonpolar_included": False,
            "grids": list(GRIDS),
            "finite_difference_steps_angstrom": list(FD_STEPS_ANGSTROM),
            "finite_difference_formula": "-(E(R+h)-E(R-h))/(2h)",
            "components_checked": "all 3N Cartesian components",
            "grid_center_policy": "gcent mol 1 on every APBS call",
        },
        "molecule": {
            "compound_id": "mobley_1017962",
            "name": "methyl hexanoate",
            "atom_count": len(atoms),
            "mol2": str(mol2.relative_to(REPOSITORY_ROOT)),
            "mol2_sha256": molecule_hash,
            "charge_sum_e": float(np.sum(charges)),
            "charge_vector_sha256": sha256_bytes(
                canonical_json_bytes(charges.tolist())
            ),
        },
        "software": {
            "apbs": apbs,
            "official_release_archive": archive_record,
            "expected_release_archive_sha256": (EXPECTED_RELEASE_ARCHIVE_SHA256),
            "existing_apbs_provider_sha256": sha256_file(
                REPOSITORY_ROOT
                / "maple/function/calculator/extra_correction/implicit/apbs_pb.py"
            ),
            "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "grid_results": grid_results,
        "cross_grid": cross_grid,
        "molecular_surface_force_request": molecular_surface_rejection,
        "numerical_gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": numerical_gate_passed,
            "scope": (
                "Numerical derivative and grid-behavior gate only; passing "
                "would not establish FreeSolv accuracy or useful throughput."
            ),
        },
        "execution": {
            "work_dir": str(work_dir),
            "worker_count": args.workers,
            "job_count": len(results),
            "expected_rejection_job_count": 1,
            "minimum_job_elapsed_seconds": float(np.min(elapsed_values)),
            "median_job_elapsed_seconds": float(np.median(elapsed_values)),
            "maximum_job_elapsed_seconds": float(np.max(elapsed_values)),
            "job_result_sha256": job_hashes,
        },
        "decision": {
            "production_provider_changed": False,
            "existing_molecular_surface_apbs_supported_tasks": ["sp"],
            "spline_force_candidate_promoted": False,
            "classification": (
                "numerically-promising-not-promoted"
                if numerical_gate_passed
                else "rejected-as-current-product-force-endpoint"
            ),
            "reason": (
                "The molecular-surface endpoint is rejected by APBS itself. "
                "The SPL4 substitute fails the frozen numerical checks, and "
                "its mbondi2 radii have not been reparameterized for a spline "
                "surface. Independent accuracy, speed, and multi-geometry "
                "validation would still be required after those failures."
            ),
            "nonpolar_force_status": (
                "not assessed here; APBS APOLAR is audited separately"
            ),
        },
        "primary_sources": {
            "apbs_force_documentation": (
                "https://apbs.readthedocs.io/en/nathan-docs/using/input/"
                "generic/calcforce.html"
            ),
            "apbs_print_documentation": (
                "https://apbs.readthedocs.io/en/stable/using/input/old/print.html"
            ),
            "apbs_surface_documentation": (
                "https://apbs.readthedocs.io/en/latest/using/input/old/elec/"
                "srfm.html"
            ),
            "apbs_release": APBS_RELEASE_URL,
        },
    }
    seal_artifact(payload)
    write_json_atomic(args.output, payload)
    Path(args.output).chmod(0o644)
    if artifact_content_sha256(payload) != payload["content_sha256"]:
        raise RuntimeError("The written APBS SPL4 artifact self-hash is invalid.")
    print(f"Wrote APBS SPL4 force probe to {Path(args.output).resolve()}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", default=str(DEFAULT_MOL2))
    parser.add_argument("--apbs", default=DEFAULT_APBS)
    parser.add_argument("--release-archive", default=DEFAULT_RELEASE_ARCHIVE)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--allow-other-binary", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
