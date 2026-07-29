#!/usr/bin/env python3
"""Execute a pinned LSNN-v1/FreeSolv compatibility MBAR pilot.

This script intentionally lives outside the production calculator registry.
The public upstream CLI is not runnable at its pinned revision, so this
research probe loads the public state dict into ``lsnn_model.py`` and samples
the resulting conservative potential with OpenMM-Torch.  It uses FreeSolv's
archived GAFF/AM1-BCC vacuum Hamiltonian, not upstream's OpenFF Hamiltonian.
"""

# OpenMM's dynamic unit/Quantity API is not fully represented by its type stubs.
# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import openmm
import pymbar
import torch
from lsnn_model import load_lsnn_v1
from openmm import NonbondedForce, Platform, app, unit
from openmm.app.internal.customgbforces import GBSAGBn2Force
from openmmtorch import TorchForce
from pymbar import timeseries

KJ_PER_KCAL = 4.184
R_KJ_PER_MOL_K = 0.00831446261815324
TORCH_FORCE_GROUP = 31
TORCH_FORCE_GROUP_MASK = 1 << TORCH_FORCE_GROUP
MIN_DECORRELATED_FRAMES = 20
MIN_ADJACENT_OVERLAP = 0.03
MAX_REPLICATE_SD_KCAL_MOL = 0.5
MAX_FORCE_RELATIVE_L2_ERROR = 0.01
MIN_INDEPENDENT_REPLICATES = 3
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260729
MAX_OPENMM_RANDOM_SEED = 2_147_483_647
FREESOLV_COMPOUND_ID = re.compile(r"mobley_[0-9]+")
PINNED_UPSTREAM_REVISION = "1768d068dcb1ea65e8585af3f4a0cbf4047d9125"
PINNED_STATE_DICT_SHA256 = (
    "5b7f9ec224f9264c0220e072ed917201e84e702bab62b331bade37c1ede3a83b"
)
PINNED_FREESOLV_REVISION = "6c7d19b4b565537365ffd22006aa2cd4643200c6"
PINNED_AMBER_TAR_SHA256 = (
    "f1a72d5e2328b3a28dfee3c3590667b756a3ac9c0527e8ddf945044e664389a2"
)
PINNED_FREESOLV_DATABASE_SHA256 = (
    "2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_finite(name: str, values: Any) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains non-finite values")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # pyright: ignore[reportArgumentType]
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_sources(
    upstream_repo: Path, freesolv_repo: Path
) -> tuple[Path, Path, Path]:
    upstream_revision = _git_revision(upstream_repo)
    if upstream_revision != PINNED_UPSTREAM_REVISION:
        raise RuntimeError(
            f"LSNN revision mismatch: {upstream_revision} != "
            f"{PINNED_UPSTREAM_REVISION}"
        )
    state_dict = upstream_repo / "Best_Trained_Models" / "280KDATASET2Kv3model.dict"
    if _sha256(state_dict) != PINNED_STATE_DICT_SHA256:
        raise RuntimeError("LSNN state-dict SHA256 mismatch")

    freesolv_revision = _git_revision(freesolv_repo)
    if freesolv_revision != PINNED_FREESOLV_REVISION:
        raise RuntimeError(
            f"FreeSolv revision mismatch: {freesolv_revision} != "
            f"{PINNED_FREESOLV_REVISION}"
        )
    amber_tar = freesolv_repo / "amber.tar.gz"
    if _sha256(amber_tar) != PINNED_AMBER_TAR_SHA256:
        raise RuntimeError("FreeSolv amber.tar.gz SHA256 mismatch")
    database = freesolv_repo / "database.txt"
    if _sha256(database) != PINNED_FREESOLV_DATABASE_SHA256:
        raise RuntimeError("FreeSolv database.txt SHA256 mismatch")
    return state_dict, amber_tar, database


def _load_freesolv_labels(
    database: Path, compounds: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Load labels only after prediction and verify panel identity."""

    requested = {compound["compound_id"]: compound for compound in compounds}
    labels: dict[str, dict[str, Any]] = {}
    for raw_line in database.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = [field.strip() for field in raw_line.split(";")]
        compound_id = fields[0]
        if compound_id not in requested:
            continue
        panel_entry = requested[compound_id]
        if fields[1] != panel_entry["smiles"]:
            raise RuntimeError(
                f"FreeSolv SMILES mismatch for {compound_id}: "
                f"{fields[1]} != {panel_entry['smiles']}"
            )
        labels[compound_id] = {
            "experimental_kcal_mol": float(fields[3]),
            "experimental_uncertainty_kcal_mol": float(fields[4]),
            "experimental_reference": fields[7],
        }
    missing = set(requested).difference(labels)
    if missing:
        raise RuntimeError(f"FreeSolv labels missing for {sorted(missing)}")
    return labels


def _extract_amber_case(
    amber_tar: Path, compound_id: str, destination: Path
) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    with tarfile.open(amber_tar, "r:gz") as archive:
        for suffix in ("prmtop", "inpcrd"):
            member_name = f"amber/{compound_id}.{suffix}"
            member = archive.getmember(member_name)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read {member_name}")
            output = destination / f"{compound_id}.{suffix}"
            output.write_bytes(source.read())
            output_paths.append(output)
    return output_paths[0], output_paths[1]


def _gbn2_features(topology: app.Topology, system: openmm.System) -> np.ndarray:
    nonbonded = next(
        force for force in system.getForces() if isinstance(force, NonbondedForce)
    )
    charges = np.asarray(
        [
            nonbonded.getParticleParameters(index)[0].value_in_unit(
                unit.elementary_charge
            )
            for index in range(system.getNumParticles())
        ]
    )
    gbn2 = GBSAGBn2Force(
        cutoff=None,
        SA="ACE",
        soluteDielectric=1,
        solventDielectric=78.5,
    )
    standard = np.asarray(gbn2.getStandardParameters(topology))
    gbn2.addParticles(np.concatenate((charges[:, None], standard), axis=1))
    gbn2.finalize()
    features = np.asarray(
        [gbn2.getParticleParameters(index) for index in range(gbn2.getNumParticles())],
        dtype=float,
    )
    _require_finite("GBn2 features", features)
    return features


def _lambda_schedule(values: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    states = [(value, 0.0) for value in values]
    states.extend((1.0, value) for value in values[1:])
    return tuple(states)


def _force_fd_diagnostic(
    simulation: app.Simulation,
    *,
    step_nm: float = 1.0e-4,
    max_coordinates: int = 12,
) -> dict[str, Any]:
    state = simulation.context.getState(
        getPositions=True,
        getForces=True,
        groups=TORCH_FORCE_GROUP_MASK,
    )
    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=float,
    )
    analytic = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=float,
    ).reshape(-1)
    _require_finite("force diagnostic positions", positions)
    _require_finite("analytic LSNN forces", analytic)
    selected = np.linspace(
        0,
        analytic.size - 1,
        num=min(max_coordinates, analytic.size),
        dtype=int,
    )
    finite_difference = np.full(analytic.shape, np.nan, dtype=float)
    for coordinate in selected:
        plus = positions.copy().reshape(-1)
        minus = positions.copy().reshape(-1)
        plus[coordinate] += step_nm
        minus[coordinate] -= step_nm
        simulation.context.setPositions(plus.reshape(positions.shape) * unit.nanometer)
        plus_energy = simulation.context.getState(
            getEnergy=True, groups=TORCH_FORCE_GROUP_MASK
        ).getPotentialEnergy()
        simulation.context.setPositions(minus.reshape(positions.shape) * unit.nanometer)
        minus_energy = simulation.context.getState(
            getEnergy=True, groups=TORCH_FORCE_GROUP_MASK
        ).getPotentialEnergy()
        plus_kj_mol = plus_energy.value_in_unit(unit.kilojoule_per_mole)
        minus_kj_mol = minus_energy.value_in_unit(unit.kilojoule_per_mole)
        _require_finite("finite-difference energies", (plus_kj_mol, minus_kj_mol))
        finite_difference[coordinate] = -(plus_kj_mol - minus_kj_mol) / (2 * step_nm)
    simulation.context.setPositions(positions * unit.nanometer)
    difference = finite_difference[selected] - analytic[selected]
    reference_norm = np.linalg.norm(analytic[selected])
    diagnostic = {
        "force_group": TORCH_FORCE_GROUP,
        "step_nm": step_nm,
        "coordinates_checked": selected.tolist(),
        "maximum_absolute_error_kj_mol_nm": float(np.max(np.abs(difference))),
        "rms_error_kj_mol_nm": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(reference_norm, 1.0)
        ),
    }
    _require_finite(
        "force finite-difference diagnostic",
        (
            diagnostic["maximum_absolute_error_kj_mol_nm"],
            diagnostic["rms_error_kj_mol_nm"],
            diagnostic["relative_l2_error"],
        ),
    )
    return diagnostic


def _build_simulation(
    *,
    prmtop_path: Path,
    inpcrd_path: Path,
    state_dict_path: Path,
    scripted_model_path: Path,
    lambda_state: tuple[float, float],
    temperature_kelvin: float,
    timestep_femtoseconds: float,
    seed: int,
    platform_name: str,
    validate_force: bool,
) -> tuple[app.Simulation, tuple[float, float], dict[str, Any]]:
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    coordinates = app.AmberInpcrdFile(str(inpcrd_path))
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    features = _gbn2_features(prmtop.topology, system)
    model = load_lsnn_v1(
        str(state_dict_path), torch.tensor(features, dtype=torch.float32)
    )
    scripted = torch.jit.script(model)
    scripted.save(str(scripted_model_path))

    torch_force = TorchForce(str(scripted_model_path))
    torch_force.addGlobalParameter("lambda_sterics", lambda_state[0])
    torch_force.addGlobalParameter("lambda_electrostatics", lambda_state[1])
    torch_force.setForceGroup(TORCH_FORCE_GROUP)
    system.addForce(torch_force)

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_kelvin * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_femtoseconds * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)
    selected_platform = Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(prmtop.topology, system, integrator, selected_platform)
    simulation.context.setPositions(coordinates.positions)
    simulation.context.setParameter("lambda_sterics", lambda_state[0])
    simulation.context.setParameter("lambda_electrostatics", lambda_state[1])
    simulation.minimizeEnergy(maxIterations=500)
    simulation.context.setVelocitiesToTemperature(
        temperature_kelvin * unit.kelvin, seed
    )

    static_zero = float(
        model(
            torch.tensor(
                coordinates.positions.value_in_unit(unit.nanometer),
                dtype=torch.float32,
            ),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ).item()
    )
    _require_finite("LSNN decoupled endpoint energy", static_zero)
    metadata = {
        "atom_count": system.getNumParticles(),
        "gbn2_feature_shape": list(features.shape),
        "scripted_model_sha256": _sha256(scripted_model_path),
        "lambda_zero_energy_kj_mol": static_zero,
    }
    if validate_force:
        metadata["force_finite_difference"] = _force_fd_diagnostic(simulation)
    return simulation, lambda_state, metadata


def _sample_state(
    simulation: app.Simulation,
    current_state: tuple[float, float],
    all_states: tuple[tuple[float, float], ...],
    *,
    equilibration_steps: int,
    production_steps: int,
    sample_interval: int,
) -> np.ndarray:
    simulation.step(equilibration_steps)
    samples: list[list[float]] = []
    for _ in range(production_steps // sample_interval):
        simulation.step(sample_interval)
        energies = []
        for lambda_s, lambda_e in all_states:
            simulation.context.setParameter("lambda_sterics", lambda_s)
            simulation.context.setParameter("lambda_electrostatics", lambda_e)
            energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            energy_kj_mol = energy.value_in_unit(unit.kilojoule_per_mole)
            _require_finite("sampled potential energy", energy_kj_mol)
            energies.append(energy_kj_mol)
        samples.append(energies)
        simulation.context.setParameter("lambda_sterics", current_state[0])
        simulation.context.setParameter("lambda_electrostatics", current_state[1])
    return np.asarray(samples, dtype=float).T


def _decorrelate(
    energies: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    if energies.ndim != 2 or energies.shape[1] == 0:
        raise RuntimeError("state energy matrix must be non-empty and two-dimensional")
    _require_finite("state energy matrix", energies)
    observable = energies[-1] - energies[0]
    try:
        equilibration, statistical_inefficiency, effective = (
            timeseries.detect_equilibration(observable)
        )
        _require_finite(
            "decorrelation diagnostics", (statistical_inefficiency, effective)
        )
        if statistical_inefficiency <= 0:
            raise RuntimeError("statistical inefficiency must be positive")
        relative = timeseries.subsample_correlated_data(
            observable[equilibration:], g=statistical_inefficiency
        )
        indices = np.asarray(relative, dtype=int) + int(equilibration)
    except Exception as exc:  # noqa: BLE001 - failure becomes an explicit quality gate
        indices = np.arange(energies.shape[1], dtype=int)
        return (
            energies,
            {
                "fallback": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "raw_frames": energies.shape[1],
                "selected_frames": energies.shape[1],
            },
            indices,
        )
    if indices.size < 2:
        indices = np.arange(energies.shape[1], dtype=int)
        fallback = True
    else:
        fallback = False
    return (
        energies[:, indices],
        {
            "fallback": fallback,
            "equilibration_frame": int(equilibration),
            "statistical_inefficiency": float(statistical_inefficiency),
            "estimated_effective_samples": float(effective),
            "raw_frames": energies.shape[1],
            "selected_frames": int(indices.size),
        },
        indices,
    )


def _mbar(
    state_energies: list[np.ndarray],
    *,
    temperature_kelvin: float,
) -> dict[str, Any]:
    if not state_energies:
        raise RuntimeError("MBAR requires at least one lambda state")
    expected_states = len(state_energies)
    for energies in state_energies:
        if energies.ndim != 2 or energies.shape[0] != expected_states:
            raise RuntimeError(
                "each sampled-state matrix must evaluate every lambda state"
            )
        _require_finite("MBAR state energies", energies)
    selected: list[np.ndarray] = []
    selected_indices: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for energies in state_energies:
        decorrelated, diagnostic, indices = _decorrelate(energies)
        selected.append(decorrelated)
        selected_indices.append(indices)
        diagnostics.append(diagnostic)
    counts = np.asarray([item.shape[1] for item in selected], dtype=int)
    if np.any(counts <= 0):
        raise RuntimeError("MBAR state counts must be positive")
    beta = 1.0 / (R_KJ_PER_MOL_K * temperature_kelvin)
    reduced = beta * np.concatenate(selected, axis=1)
    _require_finite("MBAR reduced potentials", reduced)
    estimator = pymbar.MBAR(reduced, counts, verbose=False, relative_tolerance=1e-10)
    free_energy = estimator.compute_free_energy_differences()
    delta_kj_mol = float(free_energy["Delta_f"][0, -1] / beta)
    uncertainty_kj_mol = float(free_energy["dDelta_f"][0, -1] / beta)
    overlap = estimator.compute_overlap()
    overlap_matrix = np.asarray(overlap["matrix"], dtype=float)
    _require_finite(
        "MBAR outputs",
        (
            delta_kj_mol,
            uncertainty_kj_mol,
            *overlap_matrix.reshape(-1).tolist(),
        ),
    )
    if uncertainty_kj_mol < 0:
        raise RuntimeError("MBAR uncertainty must be non-negative")
    if np.any(overlap_matrix < 0) or np.any(overlap_matrix > 1):
        raise RuntimeError("MBAR overlap matrix must lie within [0, 1]")
    adjacent = [
        min(overlap_matrix[index, index + 1], overlap_matrix[index + 1, index])
        for index in range(overlap_matrix.shape[0] - 1)
    ]
    minimum_overlap = min(adjacent) if adjacent else 1.0
    _require_finite("minimum adjacent MBAR overlap", minimum_overlap)
    quality_failures = []
    if any(item["fallback"] for item in diagnostics):
        quality_failures.append("decorrelation_fallback")
    if int(np.min(counts)) < MIN_DECORRELATED_FRAMES:
        quality_failures.append(f"decorrelated_frames_below_{MIN_DECORRELATED_FRAMES}")
    if minimum_overlap < MIN_ADJACENT_OVERLAP:
        quality_failures.append(f"adjacent_overlap_below_{MIN_ADJACENT_OVERLAP}")
    return {
        "converged": not quality_failures,
        "quality_failures": quality_failures,
        "delta_g_kcal_mol": delta_kj_mol / KJ_PER_KCAL,
        "mbar_uncertainty_kcal_mol": uncertainty_kj_mol / KJ_PER_KCAL,
        "n_k": counts.tolist(),
        "decorrelation": diagnostics,
        "overlap_matrix": overlap_matrix.tolist(),
        "minimum_adjacent_overlap": minimum_overlap,
        "_selected_frame_indices": selected_indices,
    }


def _run_compound(
    *,
    compound: dict[str, Any],
    state_dict: Path,
    amber_tar: Path,
    work_dir: Path,
    lambda_states: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    temperature_kelvin: float,
    timestep_femtoseconds: float,
    equilibration_steps: int,
    production_steps: int,
    sample_interval: int,
    platform_name: str,
    sensitivity_only: bool,
) -> dict[str, Any]:
    compound_id = compound["compound_id"]
    cases_dir = (work_dir / "cases").resolve()
    case_dir = (cases_dir / compound_id).resolve()
    if not case_dir.is_relative_to(cases_dir):
        raise RuntimeError(f"compound path escapes work directory: {compound_id}")
    prmtop, inpcrd = _extract_amber_case(amber_tar, compound_id, case_dir)
    seed_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    build_metadata: list[dict[str, Any]] = []
    for seed in seeds:
        per_state: list[np.ndarray] = []
        for state_index, lambda_state in enumerate(lambda_states):
            scripted_path = (
                case_dir / f"lsnn-{seed}-{state_index}-"
                f"{lambda_state[0]:.3f}-{lambda_state[1]:.3f}.pt"
            )
            simulation, current_state, metadata = _build_simulation(
                prmtop_path=prmtop,
                inpcrd_path=inpcrd,
                state_dict_path=state_dict,
                scripted_model_path=scripted_path,
                lambda_state=lambda_state,
                temperature_kelvin=temperature_kelvin,
                timestep_femtoseconds=timestep_femtoseconds,
                seed=seed * 100 + state_index,
                platform_name=platform_name,
                validate_force=(seed == seeds[0] and lambda_state == lambda_states[-1]),
            )
            if seed == seeds[0]:
                build_metadata.append(metadata)
            zero_energy = metadata["lambda_zero_energy_kj_mol"]
            _require_finite("LSNN decoupled endpoint energy", zero_energy)
            if abs(zero_energy) > 1e-7:
                raise RuntimeError("LSNN decoupled endpoint is not zero")
            per_state.append(
                _sample_state(
                    simulation,
                    current_state,
                    lambda_states,
                    equilibration_steps=equilibration_steps,
                    production_steps=production_steps,
                    sample_interval=sample_interval,
                )
            )
        seed_result = _mbar(per_state, temperature_kelvin=temperature_kelvin)
        selected_indices = seed_result.pop("_selected_frame_indices")
        raw_energy_path = case_dir / f"raw-u-kln-seed-{seed}.npz"
        raw_energies = np.stack(per_state, axis=0)
        _require_finite("raw u_kln energies", raw_energies)
        beta = 1.0 / (R_KJ_PER_MOL_K * temperature_kelvin)
        raw_arrays = {
            "potential_energies_kj_mol": raw_energies,
            "reduced_potentials": beta * raw_energies,
            "lambda_states": np.asarray(lambda_states, dtype=float),
            "sampled_state_indices": np.arange(len(lambda_states), dtype=int),
            "beta_mol_per_kj": np.asarray(beta, dtype=float),
        }
        raw_arrays.update(
            {
                f"selected_frame_indices_state_{index}": indices
                for index, indices in enumerate(selected_indices)
            }
        )
        _atomic_npz(raw_energy_path, **raw_arrays)
        seed_result["seed"] = seed
        seed_result["raw_energy_artifact"] = {
            "path": str(raw_energy_path.relative_to(work_dir)),
            "sha256": _sha256(raw_energy_path),
            "array_layout": (
                "[sampled_lambda_state, evaluated_lambda_state, raw_frame]"
            ),
            "shape": list(raw_energies.shape),
            "energy_unit": "kJ/mol",
            "selected_index_arrays": [
                f"selected_frame_indices_state_{index}"
                for index in range(len(selected_indices))
            ],
        }
        seed_results.append(seed_result)

    predictions = [item["delta_g_kcal_mol"] for item in seed_results]
    _require_finite("replicate predictions", predictions)
    prediction = statistics.fmean(predictions)
    replicate_sd = statistics.stdev(predictions) if len(predictions) > 1 else 0.0
    replicate_sem = replicate_sd / math.sqrt(len(predictions)) if predictions else None
    mbar_uncertainties = [item["mbar_uncertainty_kcal_mol"] for item in seed_results]
    _require_finite("MBAR uncertainties", mbar_uncertainties)
    within_seed_variance_of_mean = (
        sum(value**2 for value in mbar_uncertainties) / len(seed_results) ** 2
    )
    combined_uncertainty = math.sqrt(
        within_seed_variance_of_mean + (replicate_sem or 0.0) ** 2
    )
    _require_finite(
        "replicate statistics",
        (prediction, replicate_sd, replicate_sem, combined_uncertainty),
    )
    quality_failures = [
        f"seed_{item['seed']}:{failure}"
        for item in seed_results
        for failure in item["quality_failures"]
    ]
    if replicate_sd > MAX_REPLICATE_SD_KCAL_MOL:
        quality_failures.append(
            f"replicate_sd_above_{MAX_REPLICATE_SD_KCAL_MOL}_kcal_mol"
        )
    quality_notes = []
    if len(seed_results) < MIN_INDEPENDENT_REPLICATES:
        quality_failures.append(f"replicate_count_below_{MIN_INDEPENDENT_REPLICATES}")
    if sensitivity_only:
        quality_notes.append("sensitivity_only_excluded_from_accuracy_metrics")
    for metadata in build_metadata:
        diagnostic = metadata.get("force_finite_difference")
        if diagnostic:
            relative_error = diagnostic["relative_l2_error"]
            _require_finite("force finite-difference relative error", relative_error)
            if relative_error > MAX_FORCE_RELATIVE_L2_ERROR:
                quality_failures.append(
                    f"force_fd_relative_l2_above_{MAX_FORCE_RELATIVE_L2_ERROR}"
                )
    status = "ok" if not quality_failures else "unconverged"
    run_mode = "sensitivity_only" if sensitivity_only else "formal"
    record = {
        "status": status,
        "run_mode": run_mode,
        "quality_failures": quality_failures,
        "quality_notes": quality_notes,
        "compound_id": compound_id,
        "name": compound["name"],
        "smiles": compound["smiles"],
        "prediction_kcal_mol": prediction,
        "replicate_sd_kcal_mol": replicate_sd,
        "replicate_count": len(seed_results),
        "replicate_sem_kcal_mol": replicate_sem,
        "combined_uncertainty_kcal_mol": combined_uncertainty,
        "combined_uncertainty_method": (
            "quadrature of the mean within-seed MBAR variance and between-seed "
            "SEM; sampling-only heuristic, not a calibrated model-error interval"
        ),
        "seed_results": seed_results,
        "build": build_metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    # Seal label-free output before labels are accessed by the summary stage.
    _atomic_json(case_dir / "prediction-label-free.json", record)
    return record


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["included_in_metrics"]]
    errors = np.asarray(
        [record["signed_error_kcal_mol"] for record in successes], dtype=float
    )
    _require_finite("accuracy errors", errors)
    if errors.size:
        random = np.random.default_rng(BOOTSTRAP_SEED)
        bootstrap_indices = random.integers(
            0, errors.size, size=(BOOTSTRAP_RESAMPLES, errors.size)
        )
        bootstrap_mae = np.mean(np.abs(errors[bootstrap_indices]), axis=1)
        mae_confidence_interval = np.quantile(bootstrap_mae, (0.025, 0.975)).tolist()
    else:
        mae_confidence_interval = [None, None]
    return {
        "n_requested": len(records),
        "n_success": len(successes),
        "coverage": len(successes) / len(records) if records else 0.0,
        "mae_kcal_mol": float(np.mean(np.abs(errors))) if errors.size else None,
        "rmse_kcal_mol": (
            float(np.sqrt(np.mean(np.square(errors)))) if errors.size else None
        ),
        "median_absolute_error_kcal_mol": (
            float(np.median(np.abs(errors))) if errors.size else None
        ),
        "maximum_absolute_error_kcal_mol": (
            float(np.max(np.abs(errors))) if errors.size else None
        ),
        "mean_signed_error_kcal_mol": (float(np.mean(errors)) if errors.size else None),
        "mae_bootstrap_95_ci_kcal_mol": mae_confidence_interval,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _parse_values(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(item) for item in raw.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lambda values must be numeric") from exc
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError(
            "lambda values must be finite and within [0, 1]"
        )
    if len(values) < 2 or values[0] != 0.0 or values[-1] != 1.0:
        raise argparse.ArgumentTypeError("lambda values must contain endpoints 0 and 1")
    return values


def _parse_seeds(raw: str) -> tuple[int, ...]:
    fields = [item.strip() for item in raw.split(",")]
    if not fields or any(not item for item in fields):
        raise argparse.ArgumentTypeError(
            "seeds must be a non-empty comma-separated list"
        )
    try:
        seeds = tuple(int(item) for item in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be integers") from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    if any(seed <= 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be positive integers")
    return seeds


def main() -> int:
    started_at = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(__file__).with_name("panel.json"),
    )
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--freesolv-repo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-compounds", type=int)
    parser.add_argument(
        "--compound-id",
        action="append",
        help="Run only this panel compound ID; repeat for multiple compounds.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=(20260729, 20260730, 20260731),
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help=(
            "Allow fewer than three seeds, mark every prediction sensitivity-only, "
            "and exclude it from accuracy metrics."
        ),
    )
    parser.add_argument("--lambda-values", type=_parse_values, default=(0.0, 0.5, 1.0))
    parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--equilibration-steps", type=int, default=200)
    parser.add_argument("--production-steps", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--platform", default="CPU")
    args = parser.parse_args()

    if not math.isfinite(args.temperature_kelvin) or args.temperature_kelvin <= 0:
        parser.error("temperature-kelvin must be finite and positive")
    if not math.isfinite(args.timestep_fs) or args.timestep_fs <= 0:
        parser.error("timestep-fs must be finite and positive")
    if args.equilibration_steps <= 0:
        parser.error("equilibration-steps must be positive")
    if args.production_steps <= 0:
        parser.error("production-steps must be positive")
    if args.sample_interval <= 0:
        parser.error("sample-interval must be positive")
    if args.production_steps % args.sample_interval:
        parser.error("production steps must be divisible by sample interval")
    if args.max_compounds is not None and args.max_compounds <= 0:
        parser.error("max-compounds must be positive")
    if len(args.seeds) < MIN_INDEPENDENT_REPLICATES and not args.sensitivity_only:
        parser.error(
            f"formal accuracy runs require at least {MIN_INDEPENDENT_REPLICATES} "
            "unique seeds; pass --sensitivity-only for a diagnostic run"
        )
    if args.work_dir.exists():
        if not args.work_dir.is_dir():
            parser.error("work-dir must be a directory")
        if any(args.work_dir.iterdir()):
            parser.error("work-dir must be empty to prevent mixed-run artifacts")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if not isinstance(panel, dict) or not isinstance(panel.get("molecules"), list):
        parser.error("panel must contain a molecules list")
    compounds = list(panel["molecules"])
    if not compounds:
        parser.error("panel must contain at least one molecule")
    panel_compound_ids = [compound["compound_id"] for compound in compounds]
    if len(panel_compound_ids) != len(set(panel_compound_ids)):
        parser.error("panel contains duplicate compound IDs")
    invalid_compound_ids = [
        compound_id
        for compound_id in panel_compound_ids
        if FREESOLV_COMPOUND_ID.fullmatch(compound_id) is None
    ]
    if invalid_compound_ids:
        parser.error(
            f"panel contains invalid FreeSolv compound IDs: {invalid_compound_ids}"
        )
    if args.compound_id:
        if len(args.compound_id) != len(set(args.compound_id)):
            parser.error("compound-id selections must be unique")
        requested = set(args.compound_id)
        compounds = [
            compound for compound in compounds if compound["compound_id"] in requested
        ]
        found = {compound["compound_id"] for compound in compounds}
        if found != requested:
            parser.error(f"compound IDs absent from panel: {sorted(requested - found)}")
    if args.max_compounds is not None:
        compounds = compounds[: args.max_compounds]
    if not compounds:
        parser.error("compound selection must contain at least one molecule")
    seeds = args.seeds
    lambda_states = _lambda_schedule(args.lambda_values)
    maximum_seed = (MAX_OPENMM_RANDOM_SEED - len(lambda_states) + 1) // 100
    if any(seed > maximum_seed for seed in seeds):
        parser.error(f"seeds must be <= {maximum_seed} for derived OpenMM random seeds")
    state_dict, amber_tar, database = _validate_sources(
        args.upstream_repo, args.freesolv_repo
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    model_path = runner_path.with_name("lsnn_model.py")
    repository_root = runner_path.parents[4]
    source_provenance = {
        "maple_revision": _git_revision(repository_root),
        "runner_sha256": _sha256(runner_path),
        "model_sha256": _sha256(model_path),
        "panel_sha256": _sha256(args.panel),
    }

    predictions: list[dict[str, Any]] = []
    for compound in compounds:
        try:
            predictions.append(
                _run_compound(
                    compound=compound,
                    state_dict=state_dict,
                    amber_tar=amber_tar,
                    work_dir=args.work_dir,
                    lambda_states=lambda_states,
                    seeds=seeds,
                    temperature_kelvin=args.temperature_kelvin,
                    timestep_femtoseconds=args.timestep_fs,
                    equilibration_steps=args.equilibration_steps,
                    production_steps=args.production_steps,
                    sample_interval=args.sample_interval,
                    platform_name=args.platform,
                    sensitivity_only=args.sensitivity_only,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-compound failure records
            predictions.append(
                {
                    "status": "failed",
                    "compound_id": compound["compound_id"],
                    "name": compound["name"],
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )

    # Experimental labels are joined only after every label-free prediction has
    # either been sealed or failed.
    labels = _load_freesolv_labels(database, compounds)

    external_paths_at_completion = _validate_sources(
        args.upstream_repo, args.freesolv_repo
    )
    if tuple(path.resolve() for path in external_paths_at_completion) != tuple(
        path.resolve() for path in (state_dict, amber_tar, database)
    ):
        raise RuntimeError("benchmark external asset paths changed during execution")
    source_provenance_at_completion = {
        "maple_revision": _git_revision(repository_root),
        "runner_sha256": _sha256(runner_path),
        "model_sha256": _sha256(model_path),
        "panel_sha256": _sha256(args.panel),
    }
    if source_provenance_at_completion != source_provenance:
        raise RuntimeError("benchmark source changed during execution")

    records: list[dict[str, Any]] = []
    for prediction in predictions:
        label = labels[prediction["compound_id"]]
        record = dict(prediction)
        record.update(label)
        record["leakage_label"] = (
            "FreeSolv-aware test-split policy; exact public membership unavailable"
        )
        record["included_in_metrics"] = (
            record["status"] == "ok" and not args.sensitivity_only
        )
        if record["included_in_metrics"]:
            record["metric_exclusion_reasons"] = []
        elif record["status"] == "failed":
            record["metric_exclusion_reasons"] = [record["exception"]]
        else:
            record["metric_exclusion_reasons"] = list(
                record.get("quality_failures", ())
            ) + list(record.get("quality_notes", ()))
        if "prediction_kcal_mol" in record:
            record["signed_error_kcal_mol"] = (
                record["prediction_kcal_mol"] - record["experimental_kcal_mol"]
            )
            record["absolute_error_kcal_mol"] = abs(record["signed_error_kcal_mol"])
        records.append(record)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "openmm": openmm.__version__,
        "openmmtorch": importlib.metadata.version("openmmtorch"),
        "pymbar": pymbar.__version__,
        "numpy": np.__version__,
        "openmm_cpu_threads": os.environ.get("OPENMM_CPU_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "openmm_platforms": [
            Platform.getPlatform(index).getName()
            for index in range(Platform.getNumPlatforms())
        ],
    }
    completed_at = datetime.now(timezone.utc)
    summary = {
        "artifact_type": "lsnn-v1-conservative-gaff-freesolv-mbar-pilot",
        "created_at": completed_at.isoformat(),
        "execution": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": (completed_at - started_at).total_seconds(),
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            **source_provenance,
            "source_rechecked_at_completion": True,
            "external_assets_rechecked_at_completion": True,
        },
        "scientific_status": "compatibility-pilot-not-upstream-reproduction",
        "model": {
            "id": "lsnn-v1-conservative-energy-reconstruction",
            "upstream_revision": PINNED_UPSTREAM_REVISION,
            "state_dict_sha256": PINNED_STATE_DICT_SHA256,
            "runtime": (
                "MAPLE torch-only conservative-energy reconstruction of the "
                "public MIT graph"
            ),
        },
        "dataset": {
            "id": "FreeSolv-0.52",
            "revision": PINNED_FREESOLV_REVISION,
            "amber_tar_sha256": PINNED_AMBER_TAR_SHA256,
            "database_txt_sha256": PINNED_FREESOLV_DATABASE_SHA256,
        },
        "protocol": {
            "estimator": "PyMBAR 4",
            "vacuum_hamiltonian": "FreeSolv archived GAFF/AM1-BCC",
            "upstream_reference_hamiltonian": "OpenFF SMIRNOFFTemplateGenerator",
            "temperature_kelvin": args.temperature_kelvin,
            "timestep_femtoseconds": args.timestep_fs,
            "equilibration_steps_per_window": args.equilibration_steps,
            "production_steps_per_window": args.production_steps,
            "sample_interval_steps": args.sample_interval,
            "lambda_states": [list(state) for state in lambda_states],
            "seeds": list(seeds),
            "run_mode": "sensitivity_only" if args.sensitivity_only else "formal",
            "openmm_platform": args.platform,
            "fitting_or_calibration_against_labels": False,
            "quality_thresholds": {
                "minimum_decorrelated_frames_per_window": (MIN_DECORRELATED_FRAMES),
                "minimum_adjacent_overlap": MIN_ADJACENT_OVERLAP,
                "maximum_replicate_sd_kcal_mol": MAX_REPLICATE_SD_KCAL_MOL,
                "maximum_force_fd_relative_l2_error": (MAX_FORCE_RELATIVE_L2_ERROR),
                "minimum_independent_replicates": MIN_INDEPENDENT_REPLICATES,
            },
        },
        "claim_boundary": {
            "actual_public_weights_executed": True,
            "actual_md_sampling_executed": True,
            "actual_mbar_executed": True,
            "strict_independent_holdout": False,
            "reason": (
                "The LSNN split policy was constructed with FreeSolv similarity; "
                "the public repository does not provide an exact membership manifest."
            ),
            "upstream_cli_reproduced_verbatim": False,
            "upstream_cli_reason": (
                "Pinned upstream CLI and environment file are incomplete; the "
                "public state dict and equations are executed by a recorded "
                "research reconstruction instead."
            ),
            "upstream_hamiltonian_reproduced": False,
            "hamiltonian_reason": (
                "This compatibility pilot uses FreeSolv's archived GAFF/"
                "AM1-BCC topology; upstream constructs an OpenFF system."
            ),
            "upstream_force_convention_reproduced": False,
            "force_reason": (
                "The public upstream graph detaches the electrostatic energy "
                "before returning explicit forces. This pilot instead uses "
                "the conservative negative gradient of its full energy."
            ),
        },
        "metrics": _metrics(records),
        "records": records,
        "environment": environment,
    }
    _atomic_json(args.work_dir / "summary.json", summary)
    _atomic_json(args.work_dir / "environment.lock.json", environment)
    fields = [
        "compound_id",
        "name",
        "status",
        "included_in_metrics",
        "metric_exclusion_reasons",
        "prediction_kcal_mol",
        "replicate_count",
        "replicate_sd_kcal_mol",
        "combined_uncertainty_kcal_mol",
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
        "elapsed_seconds",
        "leakage_label",
        "quality_failures",
        "quality_notes",
        "exception",
    ]
    with (args.work_dir / "records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    return 0 if all(record["status"] == "ok" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
