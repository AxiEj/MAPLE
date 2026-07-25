#!/usr/bin/env python3
"""Audit pyddx/ddPCM coordinate derivatives and numerical convergence.

The probe is external benchmark code.  It does not register pyddx as a MAPLE
runtime provider and does not change the Route 1 production default.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
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

import benchmark_core as core  # noqa: E402
from maple.function.calculator.extra_correction.implicit.common import (  # noqa: E402
    build_openmm_topology,
)
from maple.function.calculator.extra_correction.implicit.radii import (  # noqa: E402
    OpenMMMbondi2RadiusProvider,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402
import run_ddx_pcm_screen as screen  # noqa: E402

DEFAULT_PROTOCOL = SCRIPT_DIR / "ddx_pcm_protocol.json"
DEFAULT_MANIFEST = SCRIPT_DIR / "apbs_ace_source_manifest.json"
DEFAULT_SOURCE_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_MOL2 = SCRIPT_DIR / "route1-multi-mlip-obc2-ti-raw/inputs/mobley_1017962.mol2"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json"
)

HARTREE_TO_KJ_MOL = 2625.4996394799
AXES = ("x", "y", "z")


def ddpcm_energy_force_hartree_per_angstrom(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    radii_angstrom: np.ndarray,
    *,
    lmax: int,
    n_lebedev: int,
    solvent_epsilon: float,
    solver_tolerance: float,
    expected_version: str,
) -> tuple[float, np.ndarray]:
    """Return ddPCM energy and conventional force in Hartree/angstrom."""
    pyddx = screen._require_pyddx(expected_version)
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64)
    radii = np.asarray(radii_angstrom, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("ddPCM positions must have shape (N, 3).")
    if charges.shape != (positions.shape[0],):
        raise ValueError("ddPCM charge count does not match the atom count.")
    if radii.shape != charges.shape or np.any(radii <= 0.0):
        raise ValueError("ddPCM requires one positive radius per atom.")
    if not (
        np.isfinite(positions).all()
        and np.isfinite(charges).all()
        and np.isfinite(radii).all()
    ):
        raise ValueError("ddPCM inputs must be finite.")

    model = pyddx.Model(
        "pcm",
        np.asfortranarray(positions.T * screen.BOHR_PER_ANGSTROM),
        radii * screen.BOHR_PER_ANGSTROM,
        solvent_epsilon=float(solvent_epsilon),
        lmax=int(lmax),
        n_lebedev=int(n_lebedev),
        shift=0.0,
        enable_fmm=False,
        enable_force=True,
        n_proc=1,
    )
    multipoles = np.asfortranarray(charges.reshape(1, -1) / np.sqrt(4.0 * np.pi))
    electrostatics = model.multipole_electrostatics(multipoles)
    state = pyddx.State(
        model,
        model.multipole_psi(multipoles),
        electrostatics["phi"],
    )
    energy, solvation_terms = state.ddrun(electrostatics, tol=float(solver_tolerance))
    native_positive_gradient = np.asarray(
        solvation_terms, dtype=np.float64
    ) + np.asarray(state.multipole_force_terms(multipoles), dtype=np.float64)
    force = -native_positive_gradient.T * screen.BOHR_PER_ANGSTROM
    if force.shape != positions.shape or not np.isfinite(force).all():
        raise ValueError("ddPCM returned an invalid force array.")
    return float(energy), force


def _fd_worker(
    job: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        float,
        float,
        str,
        int,
        int,
        float,
        int,
    ],
) -> tuple[int, int, int, float]:
    (
        positions,
        charges,
        radii,
        lmax,
        n_lebedev,
        solvent_epsilon,
        solver_tolerance,
        expected_version,
        atom_index,
        axis_index,
        step_angstrom,
        sign,
    ) = job
    displaced = np.asarray(positions, dtype=np.float64).copy()
    displaced[atom_index, axis_index] += sign * step_angstrom
    energy = screen.ddpcm_energy_hartree(
        displaced,
        charges,
        radii,
        lmax=lmax,
        n_lebedev=n_lebedev,
        solvent_epsilon=solvent_epsilon,
        solver_tolerance=solver_tolerance,
        expected_version=expected_version,
    )
    return atom_index, axis_index, sign, energy


def _finite_difference_force(
    positions: np.ndarray,
    charges: np.ndarray,
    radii: np.ndarray,
    *,
    lmax: int,
    n_lebedev: int,
    step_angstrom: float,
    protocol: dict[str, Any],
    workers: int,
) -> tuple[np.ndarray, float]:
    physical = protocol["physical_model"]
    jobs = [
        (
            positions,
            charges,
            radii,
            lmax,
            n_lebedev,
            float(physical["solvent_dielectric"]),
            float(physical["solver_tolerance"]),
            protocol["external_provider"]["version"],
            atom_index,
            axis_index,
            step_angstrom,
            sign,
        )
        for atom_index in range(positions.shape[0])
        for axis_index in range(3)
        for sign in (-1, 1)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(_fd_worker, jobs))
    elapsed = time.perf_counter() - started
    by_coordinate = {
        (atom_index, axis_index, sign): energy
        for atom_index, axis_index, sign, energy in values
    }
    force = np.empty_like(positions)
    for atom_index in range(positions.shape[0]):
        for axis_index in range(3):
            plus = by_coordinate[(atom_index, axis_index, 1)]
            minus = by_coordinate[(atom_index, axis_index, -1)]
            force[atom_index, axis_index] = -(plus - minus) / (2.0 * step_angstrom)
    return force, elapsed


def _component_records(
    analytic_hartree_per_angstrom: np.ndarray,
    finite_difference_hartree_per_angstrom: np.ndarray,
) -> list[dict[str, float | int | str]]:
    records = []
    for atom_index in range(analytic_hartree_per_angstrom.shape[0]):
        for axis_index, axis in enumerate(AXES):
            analytic = (
                analytic_hartree_per_angstrom[atom_index, axis_index]
                * HARTREE_TO_KJ_MOL
            )
            numerical = (
                finite_difference_hartree_per_angstrom[atom_index, axis_index]
                * HARTREE_TO_KJ_MOL
            )
            records.append(
                {
                    "atom_index": atom_index,
                    "axis": axis,
                    "analytic_kj_mol_angstrom": float(analytic),
                    "finite_difference_kj_mol_angstrom": float(numerical),
                    "error_kj_mol_angstrom": float(analytic - numerical),
                }
            )
    return records


def _force_profile(
    name: str,
    profile: dict[str, Any],
    positions: np.ndarray,
    charges: np.ndarray,
    base_radii: np.ndarray,
    *,
    protocol: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    radii = base_radii + float(profile["radius_offset_angstrom"])
    physical = protocol["physical_model"]
    candidate_started = time.perf_counter()
    energy, force = ddpcm_energy_force_hartree_per_angstrom(
        positions,
        charges,
        radii,
        lmax=int(profile["lmax"]),
        n_lebedev=int(profile["n_lebedev"]),
        solvent_epsilon=float(physical["solvent_dielectric"]),
        solver_tolerance=float(physical["solver_tolerance"]),
        expected_version=protocol["external_provider"]["version"],
    )
    candidate_seconds = time.perf_counter() - candidate_started

    reference_settings = profile["force_reference"]
    reference_started = time.perf_counter()
    reference_energy, reference_force = ddpcm_energy_force_hartree_per_angstrom(
        positions,
        charges,
        radii,
        lmax=int(reference_settings["lmax"]),
        n_lebedev=int(reference_settings["n_lebedev"]),
        solvent_epsilon=float(physical["solvent_dielectric"]),
        solver_tolerance=float(physical["solver_tolerance"]),
        expected_version=protocol["external_provider"]["version"],
    )
    reference_seconds = time.perf_counter() - reference_started
    reference_delta = (force - reference_force) * HARTREE_TO_KJ_MOL

    finite_difference: dict[str, Any] = {}
    for step in protocol["force_gate"]["finite_difference_steps_angstrom"]:
        numerical_force, elapsed = _finite_difference_force(
            positions,
            charges,
            radii,
            lmax=int(profile["lmax"]),
            n_lebedev=int(profile["n_lebedev"]),
            step_angstrom=float(step),
            protocol=protocol,
            workers=workers,
        )
        components = _component_records(force, numerical_force)
        errors = np.asarray(
            [record["error_kj_mol_angstrom"] for record in components],
            dtype=np.float64,
        )
        finite_difference[f"{float(step):.3f}"] = {
            "step_angstrom": float(step),
            "component_count": len(components),
            "wall_seconds": elapsed,
            "rmse_kj_mol_angstrom": float(np.sqrt(np.mean(errors * errors))),
            "maximum_absolute_error_kj_mol_angstrom": float(np.max(np.abs(errors))),
            "components": components,
        }

    return {
        "name": name,
        "radius_offset_angstrom": float(profile["radius_offset_angstrom"]),
        "candidate": {
            "lmax": int(profile["lmax"]),
            "n_lebedev": int(profile["n_lebedev"]),
            "energy_hartree": energy,
            "energy_kj_mol": energy * HARTREE_TO_KJ_MOL,
            "analytic_force_seconds": candidate_seconds,
            "force_kj_mol_angstrom": (force * HARTREE_TO_KJ_MOL).tolist(),
        },
        "reference": {
            "lmax": int(reference_settings["lmax"]),
            "n_lebedev": int(reference_settings["n_lebedev"]),
            "energy_hartree": reference_energy,
            "energy_delta_kj_mol": (energy - reference_energy) * HARTREE_TO_KJ_MOL,
            "analytic_force_seconds": reference_seconds,
            "force_rmse_kj_mol_angstrom": float(
                np.sqrt(np.mean(reference_delta * reference_delta))
            ),
            "force_maximum_difference_kj_mol_angstrom": float(
                np.max(np.abs(reference_delta))
            ),
        },
        "net_force_norm_kj_mol_angstrom": float(
            np.linalg.norm(force.sum(axis=0)) * HARTREE_TO_KJ_MOL
        ),
        "finite_difference": finite_difference,
    }


def _convergence_worker(
    job: tuple[str, dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    source_dir_text, row, protocol = job
    source_dir = Path(source_dir_text)
    mol2_path = source_dir / row["source_mol2_relative_path"]
    if core.sha256_file(mol2_path) != row["source_mol2_sha256"]:
        raise ValueError(f"Frozen MOL2 hash mismatch: {row['compound_id']}.")
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    charges = np.asarray(row["am1bcc_charges_e"], dtype=np.float64)
    radius_result = OpenMMMbondi2RadiusProvider().assign(build_openmm_topology(atoms))
    radii = np.asarray(radius_result.radii_angstrom, dtype=np.float64)
    numerical = protocol["numerical_convergence"]
    physical = protocol["physical_model"]
    settings = {
        "candidate": numerical["candidate"],
        "reference": numerical["reference"],
    }
    energies: dict[str, float] = {}
    timings: dict[str, float] = {}
    for key, setting in settings.items():
        started = time.perf_counter()
        energies[key] = screen.ddpcm_energy_hartree(
            positions,
            charges,
            radii,
            lmax=int(setting["lmax"]),
            n_lebedev=int(setting["n_lebedev"]),
            solvent_epsilon=float(physical["solvent_dielectric"]),
            solver_tolerance=float(physical["solver_tolerance"]),
            expected_version=protocol["external_provider"]["version"],
        )
        timings[key] = time.perf_counter() - started
    delta = (energies["candidate"] - energies["reference"]) * screen.HARTREE_TO_KCAL_MOL
    return {
        "compound_id": row["compound_id"],
        "atom_count": len(atoms),
        "source_mol2_sha256": row["source_mol2_sha256"],
        "candidate_energy_hartree": energies["candidate"],
        "reference_energy_hartree": energies["reference"],
        "candidate_minus_reference_kcal_mol": delta,
        "candidate_seconds": timings["candidate"],
        "reference_seconds": timings["reference"],
    }


def _numerical_convergence(
    protocol: dict[str, Any],
    manifest_path: Path,
    source_dir: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    if (
        core.sha256_file(manifest_path)
        != protocol["source_evidence"]["source_manifest_sha256"]
    ):
        raise ValueError("Convergence source-manifest hash mismatch.")
    manifest = core.load_json(manifest_path)
    selected_ids = set(manifest["grid_sensitivity_selection"]["compound_ids"])
    rows = [row for row in manifest["records"] if row["compound_id"] in selected_ids]
    if len(rows) != protocol["numerical_convergence"]["case_count"]:
        raise ValueError("Unexpected numerical-convergence selection size.")
    started = time.perf_counter()
    jobs = ((str(source_dir), row, protocol) for row in rows)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_convergence_worker, jobs))
    records.sort(key=lambda record: record["compound_id"])
    deltas = np.asarray(
        [record["candidate_minus_reference_kcal_mol"] for record in records],
        dtype=np.float64,
    )
    absolute = np.abs(deltas)
    summary = {
        "mean_signed_kcal_mol": float(deltas.mean()),
        "mae_kcal_mol": float(absolute.mean()),
        "p90_absolute_kcal_mol": float(np.quantile(absolute, 0.9)),
        "maximum_absolute_kcal_mol": float(absolute.max()),
    }
    gate = protocol["numerical_convergence"]
    passed = summary["p90_absolute_kcal_mol"] <= float(
        gate["p90_absolute_difference_kcal_mol"]
    ) and summary["maximum_absolute_kcal_mol"] <= float(
        gate["maximum_absolute_difference_kcal_mol"]
    )
    return {
        "selection_policy": manifest["grid_sensitivity_selection"]["policy"],
        "experimental_labels_read": False,
        "case_count": len(records),
        "candidate": gate["candidate"],
        "reference": gate["reference"],
        "thresholds": {
            "p90_absolute_kcal_mol": gate["p90_absolute_difference_kcal_mol"],
            "maximum_absolute_kcal_mol": gate["maximum_absolute_difference_kcal_mol"],
        },
        "summary": summary,
        "passed": passed,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }


def _high_order_methyl_convergence(
    positions: np.ndarray,
    charges: np.ndarray,
    radii: np.ndarray,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    physical = protocol["physical_model"]
    values = []
    for lmax, n_lebedev in ((7, 194), (9, 302), (11, 434), (13, 590)):
        started = time.perf_counter()
        energy, force = ddpcm_energy_force_hartree_per_angstrom(
            positions,
            charges,
            radii,
            lmax=lmax,
            n_lebedev=n_lebedev,
            solvent_epsilon=float(physical["solvent_dielectric"]),
            solver_tolerance=float(physical["solver_tolerance"]),
            expected_version=protocol["external_provider"]["version"],
        )
        values.append(
            {
                "lmax": lmax,
                "n_lebedev": n_lebedev,
                "energy_hartree": energy,
                "force_hartree_angstrom": force,
                "seconds": time.perf_counter() - started,
            }
        )
    reference = values[-1]
    rows = []
    for value in values:
        force_delta = (
            value["force_hartree_angstrom"] - reference["force_hartree_angstrom"]
        ) * HARTREE_TO_KJ_MOL
        rows.append(
            {
                "lmax": value["lmax"],
                "n_lebedev": value["n_lebedev"],
                "seconds": value["seconds"],
                "energy_kj_mol": value["energy_hartree"] * HARTREE_TO_KJ_MOL,
                "energy_delta_vs_l13_n590_kj_mol": (
                    value["energy_hartree"] - reference["energy_hartree"]
                )
                * HARTREE_TO_KJ_MOL,
                "force_rmse_vs_l13_n590_kj_mol_angstrom": float(
                    np.sqrt(np.mean(force_delta * force_delta))
                ),
                "force_maximum_vs_l13_n590_kj_mol_angstrom": float(
                    np.max(np.abs(force_delta))
                ),
            }
        )
    return rows


def _performance_comparison(
    protocol: dict[str, Any], vdw_profile: dict[str, Any]
) -> dict[str, Any]:
    evidence = protocol["source_evidence"]
    gpu_trace_path = SCRIPT_DIR / evidence["product_performance_trace"]
    cpu_trace_path = SCRIPT_DIR / evidence["product_performance_cpu_trace"]
    if (
        core.sha256_file(gpu_trace_path) != evidence["product_performance_trace_sha256"]
        or core.sha256_file(cpu_trace_path)
        != evidence["product_performance_cpu_trace_sha256"]
    ):
        raise ValueError("Product performance trace hash mismatch.")
    gpu_trace = core.load_json(gpu_trace_path)
    cpu_trace = core.load_json(cpu_trace_path)
    openmm_median_ms = float(
        gpu_trace["warm_energy_force"]["openmm_obc2_ace_correction_reference"][
            "median_ms"
        ]
    )
    mace_gpu_median_ms = float(
        gpu_trace["warm_energy_force"]["maceoff23m_gas_paired_reference_cuda"][
            "median_ms"
        ]
    )
    mace_cpu_median_ms = float(
        cpu_trace["warm_energy_force"]["maceoff23m_gas_paired_reference_cpu"][
            "median_ms"
        ]
    )
    ddpcm_ms = float(vdw_profile["candidate"]["analytic_force_seconds"]) * 1000.0
    return {
        "claim_scope": (
            "Same local 23-atom molecule; ddPCM is one audit call while "
            "OpenMM and MACE values are warm medians from frozen traces. "
            "Ratios are diagnostic, not universal throughput claims."
        ),
        "ddpcm_vdw_analytic_force_ms": ddpcm_ms,
        "openmm_obc2_ace_correction_median_ms": openmm_median_ms,
        "maceoff23m_gas_gpu_median_ms": mace_gpu_median_ms,
        "maceoff23m_gas_cpu_one_thread_median_ms": mace_cpu_median_ms,
        "ddpcm_vdw_force_vs_openmm_obc2_ace_correction_ratio": (
            ddpcm_ms / openmm_median_ms
        ),
        "ddpcm_vdw_force_vs_maceoff23m_gas_gpu_ratio": (ddpcm_ms / mace_gpu_median_ms),
        "ddpcm_vdw_force_vs_maceoff23m_gas_cpu_ratio": (ddpcm_ms / mace_cpu_median_ms),
        "faster_than_bare_mm_claim_allowed": False,
    }


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    manifest_path = Path(args.manifest).resolve()
    source_dir = Path(args.source_dir).resolve()
    mol2_path = Path(args.mol2).resolve()
    output = Path(args.output).resolve()
    protocol = screen.load_protocol(protocol_path)
    screen._verify_source_evidence(protocol, manifest_path)
    screen._require_pyddx(protocol["external_provider"]["version"])
    if core.sha256_file(mol2_path) != protocol["force_gate"]["mol2_sha256"]:
        raise ValueError("Force-probe MOL2 hash differs from the protocol.")

    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1, validate_charge=True)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    radius_result = OpenMMMbondi2RadiusProvider().assign(build_openmm_topology(atoms))
    base_radii = np.asarray(radius_result.radii_angstrom, dtype=np.float64)

    profiles = [
        _force_profile(
            name,
            profile,
            positions,
            charges,
            base_radii,
            protocol=protocol,
            workers=int(args.workers),
        )
        for name, profile in protocol["profiles"].items()
    ]
    profile_by_name = {profile["name"]: profile for profile in profiles}
    primary_step = f"{protocol['force_gate']['primary_step_angstrom']:.3f}"
    primary_results = [
        profile["finite_difference"][primary_step] for profile in profiles
    ]
    thresholds = protocol["force_gate"]
    force_passed = all(
        result["component_count"] == thresholds["component_count"]
        and result["rmse_kj_mol_angstrom"] <= thresholds["maximum_rmse_kj_mol_angstrom"]
        and result["maximum_absolute_error_kj_mol_angstrom"]
        <= thresholds["maximum_component_error_kj_mol_angstrom"]
        for result in primary_results
    ) and all(
        profile["net_force_norm_kj_mol_angstrom"]
        <= thresholds["maximum_net_force_norm_kj_mol_angstrom"]
        for profile in profiles
    )

    convergence = _numerical_convergence(
        protocol,
        manifest_path,
        source_dir,
        workers=int(args.workers),
    )
    high_order = _high_order_methyl_convergence(
        positions, charges, base_radii, protocol
    )
    primary_name = "ddpcm_vdw_mbondi2_l7_n194"
    performance = _performance_comparison(protocol, profile_by_name[primary_name])

    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-ddx-ddpcm-force-and-convergence-probe",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": core.sha256_file(protocol_path),
            "claim_scope": (
                "External-provider derivative, convergence, and local timing "
                "audit only. It does not promote a MAPLE runtime provider."
            ),
            "route1_contract": protocol["route1_boundary"],
            "experimental_labels_read": False,
            "external_provider": protocol["external_provider"],
            "physical_model": protocol["physical_model"],
            "molecule": {
                "compound_id": protocol["force_gate"]["compound_id"],
                "name": protocol["force_gate"]["molecule"],
                "mol2": mol2_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "mol2_sha256": core.sha256_file(mol2_path),
                "atom_count": len(atoms),
                "charge_sum_e": float(charges.sum()),
                "charge_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(charges.tolist())
                ),
                "radius_profile": radius_result.profile,
                "radius_vector_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(base_radii.tolist())
                ),
            },
            "profiles": profiles,
            "force_gate": {
                "primary_step_angstrom": float(thresholds["primary_step_angstrom"]),
                "component_count_per_profile": int(thresholds["component_count"]),
                "thresholds": {
                    "maximum_rmse_kj_mol_angstrom": thresholds[
                        "maximum_rmse_kj_mol_angstrom"
                    ],
                    "maximum_component_error_kj_mol_angstrom": thresholds[
                        "maximum_component_error_kj_mol_angstrom"
                    ],
                    "maximum_net_force_norm_kj_mol_angstrom": thresholds[
                        "maximum_net_force_norm_kj_mol_angstrom"
                    ],
                },
                "passed": force_passed,
            },
            "numerical_convergence_gate": convergence,
            "methyl_hexanoate_high_order_convergence": high_order,
            "performance_comparison": performance,
            "decision": {
                "analytic_polar_derivative_validated": force_passed,
                "label_blind_numerical_gate_passed": convergence["passed"],
                "runtime_provider_added": False,
                "project_dependency_added": False,
                "production_default_changed": False,
                "reason": (
                    "Derivative validity is necessary but not sufficient; "
                    "accuracy and product-appropriate performance are scored "
                    "separately before any provider admission."
                ),
            },
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "protocol": protocol_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "source_dir": Path(args.source_dir).as_posix(),
                    "mol2": mol2_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "workers": int(args.workers),
                    "output": output.relative_to(REPOSITORY_ROOT).as_posix(),
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
    )
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "force_gate": artifact["force_gate"],
                "numerical_convergence": convergence["summary"],
                "performance_comparison": performance,
                "decision": artifact["decision"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--mol2", default=DEFAULT_MOL2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    run(args)


if __name__ == "__main__":
    main()
