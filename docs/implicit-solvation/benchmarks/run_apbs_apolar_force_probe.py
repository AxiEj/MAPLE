#!/usr/bin/env python3
"""Audit APBS APOLAR force output against its reported nonpolar energy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
from run_apbs_spline_force_probe import (  # noqa: E402
    APBS_RELEASE_URL,
    AXES,
    DEFAULT_APBS,
    DEFAULT_MOL2,
    DEFAULT_RELEASE_ARCHIVE,
    EXPECTED_MOL2_SHA256,
    EXPECTED_RELEASE_ARCHIVE_SHA256,
    _resolve_executable,
    inspect_apbs,
)

DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-apbs-apolar-force-probe-methyl-hexanoate-2026-07-25.json"
)
DEFAULT_WORK_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/apbs-apolar-force-probe-route1-20260725"
)
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
GAMMA_KJ_MOL_ANGSTROM2 = 0.105
SURFACE_DENSITY = 10.0
APBS_INTERNAL_DPOS_ANGSTROM = 0.05
EXTERNAL_FD_STEPS_ANGSTROM = (0.010, 0.050)


def render_apolar_input(
    pqr_name: str,
    *,
    require_forces: bool,
    dpos_angstrom: float = APBS_INTERNAL_DPOS_ANGSTROM,
) -> str:
    force_mode = "comps" if require_forces else "no"
    force_print = "print apolForce nonpolar end\n" if require_forces else ""
    return f"""\
read
    mol pqr {pqr_name}
end
apolar name nonpolar
    mol 1
    srfm sacc
    srad 1.400000
    swin 0.3
    sdens {SURFACE_DENSITY:.6f}
    gamma {GAMMA_KJ_MOL_ANGSTROM2:.8f}
    press 0.00000000
    bconc 0.0
    dpos {dpos_angstrom:.6f}
    grid 0.5 0.5 0.5
    temp 298.15
    calcenergy total
    calcforce {force_mode}
end
print apolEnergy nonpolar end
{force_print}quit
"""


def _parse_force_rows(section: str, *, expected_atoms: int) -> list[list[float]]:
    rows = re.findall(
        rf"^\s*tot\s+(\d+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$",
        section,
        re.MULTILINE,
    )
    indices = [int(row[0]) for row in rows]
    if indices != list(range(expected_atoms)):
        raise ValueError(
            "APBS APOLAR force atom indices are incomplete or out of order: "
            f"{indices!r}."
        )
    return [[float(value) for value in row[1:]] for row in rows]


def parse_apolar_output(
    output: str, *, expected_atoms: int, require_forces: bool
) -> dict[str, Any]:
    energy_match = re.search(
        rf"Global\s+net\s+APOL\s+energy\s*=\s*({FLOAT})\s*kJ/mol",
        output,
        re.IGNORECASE,
    )
    if energy_match is None:
        raise ValueError("APBS output has no printed APOLAR energy.")
    parsed: dict[str, Any] = {"apolar_energy_kj_mol": float(energy_match.group(1))}
    if not require_forces:
        return parsed

    calculation_match = re.search(
        r"CALCULATION\s+#\d+\s+\(nonpolar\):\s+APOLAR(.*?)"
        r"(?:Solvent Accessible Surface Area|PRINT STATEMENTS)",
        output,
        re.IGNORECASE | re.DOTALL,
    )
    if calculation_match is None:
        raise ValueError("APBS output has no APOLAR calculation force block.")
    print_match = re.search(
        r"print\s+APOL\s+force\s+\d+\s+\(nonpolar\)\s+end(.*?)(?:-{10,}|$)",
        output,
        re.IGNORECASE | re.DOTALL,
    )
    if print_match is None:
        raise ValueError("APBS output has no printed APOLAR force block.")
    parsed["calculation_total_force_kj_mol_angstrom"] = _parse_force_rows(
        calculation_match.group(1), expected_atoms=expected_atoms
    )
    parsed["printed_total_component_native"] = _parse_force_rows(
        print_match.group(1), expected_atoms=expected_atoms
    )
    return parsed


def _positions_sha256(positions: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(positions, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _signature(
    *,
    positions: np.ndarray,
    require_forces: bool,
    dpos_angstrom: float,
    executable_sha256: str,
    molecule_sha256: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "positions_sha256": _positions_sha256(positions),
                "require_forces": require_forces,
                "dpos_angstrom": dpos_angstrom,
                "surface": "sacc",
                "surface_density": SURFACE_DENSITY,
                "gamma_kj_mol_angstrom2": GAMMA_KJ_MOL_ANGSTROM2,
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
    require_forces: bool,
    dpos_angstrom: float,
    executable: Path,
    executable_sha256: str,
    molecule_sha256: str,
    job_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    signature = _signature(
        positions=positions,
        require_forces=require_forces,
        dpos_angstrom=dpos_angstrom,
        executable_sha256=executable_sha256,
        molecule_sha256=molecule_sha256,
    )
    result_path = job_dir / "result.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("job_signature") == signature:
            return result
        raise ValueError(f"Incompatible cached APBS APOLAR job: {result_path}")

    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=False)
    job_atoms = atoms.copy()
    job_atoms.positions[:] = positions
    provider = APBSLPB(
        job_atoms,
        charges,
        executable=str(executable),
        grid_spacing=0.25,
        grid_points=129,
        timeout=timeout,
    )
    pqr = job_dir / "molecule.pqr"
    inp = job_dir / "apbs.in"
    stdout_path = job_dir / "apbs.stdout.log"
    stderr_path = job_dir / "apbs.stderr.log"
    provider.write_pqr(pqr, job_atoms)
    inp.write_text(
        render_apolar_input(
            pqr.name,
            require_forces=require_forces,
            dpos_angstrom=dpos_angstrom,
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
        raise RuntimeError(
            f"APBS APOLAR failed with code {completed.returncode}: {detail}"
        )
    result = {
        "schema_version": 1,
        "job_signature": signature,
        "positions_sha256": _positions_sha256(positions),
        "require_forces": require_forces,
        "dpos_angstrom": dpos_angstrom,
        "elapsed_seconds": elapsed,
        "input_sha256": sha256_file(inp),
        "pqr_sha256": sha256_file(pqr),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        **parse_apolar_output(
            completed.stdout,
            expected_atoms=len(job_atoms),
            require_forces=require_forces,
        ),
    }
    write_json_atomic(result_path, result)
    return result


def _fd_job_name(step: float, atom_index: int, axis_index: int, sign: int) -> str:
    step_text = f"{step:.3f}".replace(".", "p")
    direction = "plus" if sign > 0 else "minus"
    return (
        f"external-fd-{step_text}/"
        f"atom-{atom_index:03d}-{AXES[axis_index]}-{direction}"
    )


def _metrics(
    reference: np.ndarray,
    finite_difference: np.ndarray,
    symbols: list[str],
) -> dict[str, Any]:
    error = reference - finite_difference
    absolute = np.abs(error)
    largest = np.unravel_index(int(np.argmax(absolute)), error.shape)
    rows = []
    for atom_index in range(reference.shape[0]):
        for axis_index, axis in enumerate(AXES):
            rows.append(
                {
                    "atom_index_zero_based": atom_index,
                    "element": symbols[atom_index],
                    "axis": axis,
                    "reported_force_kj_mol_angstrom": float(
                        reference[atom_index, axis_index]
                    ),
                    "finite_difference_force_kj_mol_angstrom": float(
                        finite_difference[atom_index, axis_index]
                    ),
                    "signed_error_kj_mol_angstrom": float(
                        error[atom_index, axis_index]
                    ),
                }
            )
    return {
        "component_count": int(error.size),
        "mean_absolute_error_kj_mol_angstrom": float(np.mean(absolute)),
        "rmse_kj_mol_angstrom": float(np.sqrt(np.mean(error * error))),
        "maximum_absolute_error_kj_mol_angstrom": float(absolute[largest]),
        "largest_error_component": {
            "atom_index_zero_based": int(largest[0]),
            "element": symbols[int(largest[0])],
            "axis": AXES[int(largest[1])],
            "reported_force_kj_mol_angstrom": float(reference[largest]),
            "finite_difference_force_kj_mol_angstrom": float(
                finite_difference[largest]
            ),
            "signed_error_kj_mol_angstrom": float(error[largest]),
        },
        "components": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mol2 = Path(args.mol2).resolve()
    molecule_hash = sha256_file(mol2)
    if molecule_hash != EXPECTED_MOL2_SHA256:
        raise ValueError("The frozen methyl-hexanoate MOL2 hash does not match.")
    executable = _resolve_executable(args.apbs)
    apbs = inspect_apbs(executable, allow_other_binary=bool(args.allow_other_binary))
    archive_record: dict[str, Any] | None = None
    if args.release_archive:
        archive = Path(args.release_archive).resolve()
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
    if not math.isclose(float(np.sum(charges)), 0.0, abs_tol=1.0e-6):
        raise ValueError("The frozen neutral molecule has a nonzero charge sum.")
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = [
        {
            "name": "base-force-dpos-0p050",
            "positions": positions,
            "require_forces": True,
            "dpos_angstrom": APBS_INTERNAL_DPOS_ANGSTROM,
        }
    ]
    for step in EXTERNAL_FD_STEPS_ANGSTROM:
        for atom_index in range(len(atoms)):
            for axis_index in range(3):
                for sign in (-1, 1):
                    displaced = positions.copy()
                    displaced[atom_index, axis_index] += sign * step
                    jobs.append(
                        {
                            "name": _fd_job_name(step, atom_index, axis_index, sign),
                            "positions": displaced,
                            "require_forces": False,
                            "dpos_angstrom": APBS_INTERNAL_DPOS_ANGSTROM,
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
                require_forces=job["require_forces"],
                dpos_angstrom=job["dpos_angstrom"],
                executable=executable,
                executable_sha256=apbs["sha256"],
                molecule_sha256=molecule_hash,
                job_dir=work_dir / job["name"],
                timeout=args.timeout,
            ): job["name"]
            for job in jobs
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    base = results["base-force-dpos-0p050"]
    calculation_force = np.asarray(
        base["calculation_total_force_kj_mol_angstrom"], dtype=np.float64
    )
    printed_component = np.asarray(
        base["printed_total_component_native"], dtype=np.float64
    )
    scaled_print = -GAMMA_KJ_MOL_ANGSTROM2 * printed_component
    scaling_residual = calculation_force - scaled_print
    symbols = atoms.get_chemical_symbols()
    finite_differences: dict[str, Any] = {}
    fd_arrays: dict[float, np.ndarray] = {}
    for step in EXTERNAL_FD_STEPS_ANGSTROM:
        fd = np.zeros((len(atoms), 3), dtype=np.float64)
        for atom_index in range(len(atoms)):
            for axis_index in range(3):
                minus = results[_fd_job_name(step, atom_index, axis_index, -1)][
                    "apolar_energy_kj_mol"
                ]
                plus = results[_fd_job_name(step, atom_index, axis_index, 1)][
                    "apolar_energy_kj_mol"
                ]
                fd[atom_index, axis_index] = -(float(plus) - float(minus)) / (
                    2.0 * step
                )
        fd_arrays[step] = fd
        finite_differences[f"{step:.3f}"] = {
            "calculation_total_force": _metrics(calculation_force, fd, symbols),
            "printed_total_component": _metrics(printed_component, fd, symbols),
        }

    fd_step_difference = (
        fd_arrays[EXTERNAL_FD_STEPS_ANGSTROM[1]]
        - fd_arrays[EXTERNAL_FD_STEPS_ANGSTROM[0]]
    )
    fd_step_absolute = np.abs(fd_step_difference)
    thresholds = {
        "calculation_force_rmse_at_matching_step_kj_mol_angstrom": 0.10,
        "calculation_force_maximum_at_matching_step_kj_mol_angstrom": 0.50,
        "net_force_norm_kj_mol_angstrom": 0.10,
        "external_fd_step_rmse_kj_mol_angstrom": 0.10,
        "external_fd_step_maximum_kj_mol_angstrom": 0.50,
    }
    matching = finite_differences[f"{APBS_INTERNAL_DPOS_ANGSTROM:.3f}"][
        "calculation_total_force"
    ]
    net_force = np.sum(calculation_force, axis=0)
    checks = {
        "matching-step-rmse": (
            matching["rmse_kj_mol_angstrom"]
            <= thresholds["calculation_force_rmse_at_matching_step_kj_mol_angstrom"]
        ),
        "matching-step-maximum": (
            matching["maximum_absolute_error_kj_mol_angstrom"]
            <= thresholds["calculation_force_maximum_at_matching_step_kj_mol_angstrom"]
        ),
        "net-force": (
            float(np.linalg.norm(net_force))
            <= thresholds["net_force_norm_kj_mol_angstrom"]
        ),
        "external-fd-step-rmse": (
            float(np.sqrt(np.mean(fd_step_difference * fd_step_difference)))
            <= thresholds["external_fd_step_rmse_kj_mol_angstrom"]
        ),
        "external-fd-step-maximum": (
            float(np.max(fd_step_absolute))
            <= thresholds["external_fd_step_maximum_kj_mol_angstrom"]
        ),
    }
    elapsed = np.asarray(
        [float(result["elapsed_seconds"]) for result in results.values()]
    )
    payload = {
        "schema_version": 1,
        "recorded_date": "2026-07-25",
        "artifact_type": "route1-apbs-apolar-force-probe",
        "claim_scope": (
            "One-molecule numerical audit of the existing APBS APOLAR "
            "SASA-energy force output. It is not an accuracy benchmark."
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
        },
        "method": {
            "component": "nonpolar only",
            "surface": "sacc",
            "surface_density": SURFACE_DENSITY,
            "gamma_kj_mol_angstrom2": GAMMA_KJ_MOL_ANGSTROM2,
            "pressure_kj_mol_angstrom3": 0.0,
            "bulk_solvent_density_angstrom3": 0.0,
            "apbs_internal_dpos_angstrom": APBS_INTERNAL_DPOS_ANGSTROM,
            "external_fd_steps_angstrom": list(EXTERNAL_FD_STEPS_ANGSTROM),
            "components_checked": "all 3N Cartesian components",
        },
        "molecule": {
            "compound_id": "mobley_1017962",
            "name": "methyl hexanoate",
            "atom_count": len(atoms),
            "mol2": str(mol2.relative_to(REPOSITORY_ROOT)),
            "mol2_sha256": molecule_hash,
        },
        "software": {
            "apbs": apbs,
            "official_release_archive": archive_record,
            "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "base": {
            "apolar_energy_kj_mol": base["apolar_energy_kj_mol"],
            "calculation_total_force_kj_mol_angstrom": (
                base["calculation_total_force_kj_mol_angstrom"]
            ),
            "printed_total_component_native": (base["printed_total_component_native"]),
            "net_calculation_force_kj_mol_angstrom": net_force.tolist(),
            "net_calculation_force_norm_kj_mol_angstrom": float(
                np.linalg.norm(net_force)
            ),
            "print_to_calculation_relation": {
                "formula": (
                    "calculation_total_force ~= " "-gamma * printed_total_component"
                ),
                "maximum_absolute_residual_kj_mol_angstrom": float(
                    np.max(np.abs(scaling_residual))
                ),
                "note": (
                    "The PRINT APOL component is not the scaled total force "
                    "that differentiates gamma*SASA."
                ),
            },
        },
        "finite_difference": finite_differences,
        "external_fd_step_sensitivity": {
            "rmse_kj_mol_angstrom": float(
                np.sqrt(np.mean(fd_step_difference * fd_step_difference))
            ),
            "maximum_absolute_difference_kj_mol_angstrom": float(
                np.max(fd_step_absolute)
            ),
        },
        "numerical_gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": all(checks.values()),
        },
        "execution": {
            "work_dir": str(work_dir),
            "worker_count": args.workers,
            "job_count": len(results),
            "minimum_job_elapsed_seconds": float(np.min(elapsed)),
            "median_job_elapsed_seconds": float(np.median(elapsed)),
            "maximum_job_elapsed_seconds": float(np.max(elapsed)),
            "job_result_sha256": {
                name: sha256_file(work_dir / name / "result.json")
                for name in sorted(results)
            },
        },
        "decision": {
            "production_provider_changed": False,
            "apbs_apolar_force_promoted": False,
            "classification": (
                "numerically-promising-not-promoted"
                if all(checks.values())
                else "rejected-for-force-capable-product-use"
            ),
            "policy": (
                "Keep APBS APOLAR energy-only. If a PB polar force candidate "
                "ever passes all gates, pair it only with a separately verified "
                "analytic nonpolar provider such as the existing OpenMM ACE path."
            ),
        },
        "primary_sources": {
            "apbs_nonpolar_documentation": (
                "https://apbs.readthedocs.io/en/stable/using/input/new/"
                "calculate/nonpolar.html"
            ),
            "apbs_print_documentation": (
                "https://apbs.readthedocs.io/en/stable/using/input/old/print.html"
            ),
            "apbs_release": APBS_RELEASE_URL,
        },
    }
    seal_artifact(payload)
    write_json_atomic(args.output, payload)
    Path(args.output).chmod(0o644)
    if artifact_content_sha256(payload) != payload["content_sha256"]:
        raise RuntimeError("The written APBS APOLAR artifact self-hash is invalid.")
    print(f"Wrote APBS APOLAR force probe to {Path(args.output).resolve()}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", default=str(DEFAULT_MOL2))
    parser.add_argument("--apbs", default=DEFAULT_APBS)
    parser.add_argument("--release-archive", default=DEFAULT_RELEASE_ARCHIVE)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--allow-other-binary", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
