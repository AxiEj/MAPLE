#!/usr/bin/env python3
"""Relax stratified Route 1 conformers on MACE gas and MACE+GB surfaces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from ase.units import Hartree

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_conformer_sensitivity import _read_xyz_ensemble  # noqa: E402
from run_mlip_conformer_weighting import (  # noqa: E402
    KCAL_PER_HARTREE,
    _aligned_rmsd,
    _load_calculator,
    load_weighting_protocol,
    reference_geometry_union,
)
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and set(text) <= HEX


def load_relaxation_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unsupported MLIP conformer-relaxation protocol schema.")
    if protocol.get("source_partition") != "development":
        raise ValueError("MLIP conformer relaxation must remain development-only.")
    for key in (
        "source_evidence",
        "model",
        "solvation",
        "start_policy",
        "optimizer",
        "failure_policy",
    ):
        if not isinstance(protocol.get(key), dict):
            raise ValueError(f"MLIP conformer-relaxation protocol lacks {key}.")
    for key in (
        "weighting_protocol_sha256",
        "weighting_protocol_fingerprint",
        "weighting_summary_sha256",
        "prepared_sha256",
    ):
        if not _is_sha256(protocol["source_evidence"].get(key)):
            raise ValueError(f"source_evidence.{key} must be a SHA256.")
    if not _is_sha256(protocol["model"].get("checkpoint_sha256")):
        raise ValueError("model.checkpoint_sha256 must be a SHA256.")
    if protocol["model"].get("maple_calculator_output_unit") != "hartree":
        raise ValueError("The optimizer unit bridge requires Hartree MAPLE outputs.")
    if protocol["start_policy"].get("ordered_roles") != [
        "reference",
        "gas-dominant",
        "solution-dominant",
    ]:
        raise ValueError("The frozen relaxation start-role order changed.")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("MLIP conformer-relaxation protocol has no cases.")
    identifiers = [case.get("compound_id") for case in cases]
    if any(not identifier for identifier in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("MLIP conformer-relaxation case identifiers must be unique.")
    optimizer = protocol["optimizer"]
    for key in ("fmax_eV_per_angstrom", "maxstep_angstrom"):
        value = optimizer.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"optimizer.{key} must be positive and finite.")
    max_steps = optimizer.get("max_steps")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("optimizer.max_steps must be a positive integer.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


class HartreeToEVCalculator(Calculator):
    """Present MAPLE Hartree outputs to ASE optimizers in native eV units."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, source_calculator):
        super().__init__()
        self.source_calculator = source_calculator

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        requested = set(properties or ["energy"])
        source_properties = ["energy"]
        if "forces" in requested:
            source_properties.append("forces")
        self.source_calculator.calculate(
            atoms,
            properties=source_properties,
            system_changes=system_changes,
        )
        source = self.source_calculator.results
        energy_eV = float(source["energy"]) * Hartree
        self.results = {
            "energy": energy_eV,
            "free_energy": float(source.get("free_energy", source["energy"])) * Hartree,
        }
        if "forces" in source:
            self.results["forces"] = (
                np.asarray(source["forces"], dtype=np.float64) * Hartree
            )


def select_start_states(weighting_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Select and deduplicate the frozen reference/gas/solution dominant states."""

    role_indices = [
        (
            "reference",
            weighting_record["reference_geometry_union"]["reference_index"],
        ),
        (
            "gas-dominant",
            weighting_record["result"]["gas_dominant_conformer_index"],
        ),
        (
            "solution-dominant",
            weighting_record["result"]["solution_dominant_conformer_index"],
        ),
    ]
    selected: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}
    for role, raw_index in role_indices:
        if (
            not isinstance(raw_index, int)
            or isinstance(raw_index, bool)
            or raw_index < 0
        ):
            raise ValueError(f"Invalid {role} partition-state index: {raw_index!r}")
        state = by_index.get(raw_index)
        if state is None:
            state = {"index": raw_index, "roles": []}
            selected.append(state)
            by_index[raw_index] = state
        state["roles"].append(role)
    return selected


def relaxed_transfer_decomposition(
    *,
    gas_min_energy_hartree: float,
    solution_min_gas_energy_hartree: float,
    solution_min_solvent_energy_hartree: float,
) -> dict[str, float]:
    """Decompose the separately relaxed 0 K transfer potential energy."""

    values = np.asarray(
        [
            gas_min_energy_hartree,
            solution_min_gas_energy_hartree,
            solution_min_solvent_energy_hartree,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Relaxed-transfer decomposition inputs must be finite.")
    reorganization = (
        float(solution_min_gas_energy_hartree) - float(gas_min_energy_hartree)
    ) * KCAL_PER_HARTREE
    solvent = float(solution_min_solvent_energy_hartree) * KCAL_PER_HARTREE
    transfer = (
        float(solution_min_gas_energy_hartree)
        + float(solution_min_solvent_energy_hartree)
        - float(gas_min_energy_hartree)
    ) * KCAL_PER_HARTREE
    closure = transfer - (reorganization + solvent)
    if abs(closure) < 1.0e-9:
        closure = 0.0
    return {
        "gas_reorganization_cost_kcal_mol": reorganization,
        "solvent_correction_at_solution_min_kcal_mol": solvent,
        "relaxed_transfer_energy_kcal_mol": transfer,
        "decomposition_closure_kcal_mol": closure,
    }


def _write_xyz_atomic(path: Path, symbols: list[str], positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write("MAPLE MLIP conformer relaxation final geometry\n")
        for symbol, (x, y, z) in zip(symbols, positions, strict=True):
            handle.write(f"{symbol:<2s} {x: .12f} {y: .12f} {z: .12f}\n")
    temporary.replace(path)


def _maximum_force(forces_eV_per_angstrom: np.ndarray) -> float:
    forces = np.asarray(forces_eV_per_angstrom, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
        raise ValueError("Optimizer forces must be a finite (N, 3) array.")
    return float(np.linalg.norm(forces, axis=1).max())


def _source_energy_result(source_calculator) -> dict[str, Any]:
    results = source_calculator.results
    total = float(results["energy"])
    if not math.isfinite(total):
        raise ValueError("MAPLE returned a non-finite energy.")
    structured = results.get("solvation")
    if structured is None:
        return {
            "total_energy_hartree": total,
            "gas_energy_hartree": total,
            "solvent_energy_hartree": 0.0,
            "solvent_components_hartree": {},
        }
    gas = float(structured["gas_energy_hartree"])
    solvent = float(structured["energy_hartree"])
    components = {
        str(key): float(value)
        for key, value in structured["components_hartree"].items()
    }
    if abs(total - (gas + solvent)) > 1.0e-10:
        raise ValueError("Combined MACE+solvent energy does not close.")
    if abs(solvent - sum(components.values())) > 1.0e-10:
        raise ValueError("Implicit-solvent component energies do not close.")
    return {
        "total_energy_hartree": total,
        "gas_energy_hartree": gas,
        "solvent_energy_hartree": solvent,
        "solvent_components_hartree": components,
    }


def _run_optimization(
    *,
    initial_atoms,
    source_calculator,
    optimizer_options: dict[str, Any],
    phase: str,
    branch_dir: Path,
) -> dict[str, Any]:
    atoms = initial_atoms.copy()
    atoms.calc = HartreeToEVCalculator(source_calculator)
    branch_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        initial_forces = atoms.get_forces()
        initial_energy_eV = float(atoms.get_potential_energy())
        initial_source = _source_energy_result(source_calculator)
        initial_positions = atoms.get_positions().copy()
        optimizer = LBFGS(
            atoms,
            logfile=str(branch_dir / f"{phase}.log"),
            trajectory=str(branch_dir / f"{phase}.traj"),
            maxstep=float(optimizer_options["maxstep_angstrom"]),
        )
        optimizer_reported_converged = bool(
            optimizer.run(
                fmax=float(optimizer_options["fmax_eV_per_angstrom"]),
                steps=int(optimizer_options["max_steps"]),
            )
        )
        final_forces = atoms.get_forces()
        final_energy_eV = float(atoms.get_potential_energy())
        final_source = _source_energy_result(source_calculator)
        final_positions = atoms.get_positions().copy()
        initial_fmax = _maximum_force(initial_forces)
        final_fmax = _maximum_force(final_forces)
        converged = bool(
            optimizer_reported_converged
            and final_fmax <= float(optimizer_options["fmax_eV_per_angstrom"]) + 1.0e-12
        )
        if not np.isfinite(
            np.asarray(
                [
                    initial_energy_eV,
                    final_energy_eV,
                    initial_fmax,
                    final_fmax,
                ]
            )
        ).all():
            raise ValueError("Optimizer returned a non-finite energy or force.")
        geometry_path = branch_dir / f"{phase}-final.xyz"
        symbols = list(atoms.get_chemical_symbols())
        _write_xyz_atomic(geometry_path, symbols, final_positions)
        heavy = np.asarray(
            [index for index, symbol in enumerate(symbols) if symbol != "H"],
            dtype=int,
        )
        if not len(heavy):
            heavy = np.arange(len(symbols), dtype=int)
        return {
            "status": "success",
            "converged": converged,
            "optimizer_reported_converged": optimizer_reported_converged,
            "steps": int(optimizer.nsteps),
            "elapsed_seconds": time.perf_counter() - start,
            "initial_energy_eV": initial_energy_eV,
            "final_energy_eV": final_energy_eV,
            "energy_decrease_eV": initial_energy_eV - final_energy_eV,
            "initial_max_force_eV_per_angstrom": initial_fmax,
            "final_max_force_eV_per_angstrom": final_fmax,
            "initial_energy": initial_source,
            "final_energy": final_source,
            "initial_to_final_aligned_heavy_atom_rmsd_angstrom": _aligned_rmsd(
                initial_positions,
                final_positions,
                heavy,
            ),
            "final_positions_angstrom": final_positions.tolist(),
            "final_geometry_sha256": sha256_file(geometry_path),
        }
    except Exception as exc:
        return {
            "status": "failure",
            "converged": False,
            "elapsed_seconds": time.perf_counter() - start,
            "failure": {
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            },
        }


def _validate_sources(
    protocol: dict[str, Any],
    protocol_path: Path,
    weighting_output_dir: Path,
    conformer_output_dir: Path,
    base_work_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    source = protocol["source_evidence"]
    weighting_protocol_path = protocol_path.parent / source["weighting_protocol"]
    weighting_summary_path = protocol_path.parent / source["weighting_summary"]
    if sha256_file(weighting_protocol_path) != source["weighting_protocol_sha256"]:
        raise ValueError("Frozen MLIP weighting protocol hash changed.")
    _weighting_protocol, weighting_fingerprint = load_weighting_protocol(
        weighting_protocol_path
    )
    if weighting_fingerprint != source["weighting_protocol_fingerprint"]:
        raise ValueError("Frozen MLIP weighting protocol fingerprint changed.")
    if sha256_file(weighting_summary_path) != source["weighting_summary_sha256"]:
        raise ValueError("Frozen MLIP weighting summary hash changed.")
    weighting_summary = load_json(weighting_summary_path)
    if weighting_summary.get("protocol_fingerprint") != weighting_fingerprint:
        raise ValueError("Frozen MLIP weighting summary targets another protocol.")
    prepared_path = base_work_dir / "prepared.json"
    if sha256_file(prepared_path) != source["prepared_sha256"]:
        raise ValueError("Frozen prepared FreeSolv candidate artifact changed.")
    prepared = load_json(prepared_path)
    candidates = {
        candidate["compound_id"]: candidate for candidate in prepared["candidates"]
    }
    weighting_records: dict[str, dict[str, Any]] = {}
    conformer_records: dict[str, dict[str, Any]] = {}
    for case in protocol["cases"]:
        compound_id = case["compound_id"]
        candidate = candidates.get(compound_id)
        if candidate is None or candidate.get("partition") != "development":
            raise ValueError(f"Relaxation case is not development-only: {compound_id}")
        weighting_path = weighting_output_dir / "records" / f"{compound_id}.json"
        if (
            sha256_file(weighting_path)
            != weighting_summary["record_sha256"][compound_id]
        ):
            raise ValueError(f"Frozen MLIP weighting record changed: {compound_id}")
        weighting_record = load_json(weighting_path)
        if weighting_record.get("status") != "success":
            raise ValueError(f"Frozen MLIP weighting record failed: {compound_id}")
        conformer_path = conformer_output_dir / "records" / f"{compound_id}.json"
        if sha256_file(conformer_path) != weighting_record["source_record_sha256"]:
            raise ValueError(f"Frozen conformer source record changed: {compound_id}")
        weighting_records[compound_id] = weighting_record
        conformer_records[compound_id] = load_json(conformer_path)
    return weighting_summary, candidates, weighting_records, conformer_records


def _load_case_atoms_and_frames(
    *,
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    weighting_record: dict[str, Any],
    conformer_record: dict[str, Any],
    base_work_dir: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    mol2_path = base_work_dir / candidate["mol2_relative_path"]
    if sha256_file(mol2_path) != candidate["mol2_sha256"]:
        raise ValueError("Frozen case MOL2 hash changed.")
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    charge_path = (
        base_work_dir
        / "charges"
        / candidate["compound_id"]
        / f"{protocol['solvation']['charge_method']}.json"
    )
    charge_record = load_json(charge_path)
    if (
        charge_record.get("charge_method") != protocol["solvation"]["charge_method"]
        or charge_record.get("mol2_sha256") != candidate["mol2_sha256"]
    ):
        raise ValueError("Frozen ABCG2 charge record does not match the case.")
    charges = np.asarray(charge_record["charges_e"], dtype=np.float64)
    if charges.shape != (len(atoms),) or not np.isfinite(charges).all():
        raise ValueError("Frozen ABCG2 charges are invalid.")
    atoms.set_initial_charges(charges)
    ensemble_path = Path(conformer_record["audit_dir"]) / "crest_conformers.xyz"
    if sha256_file(ensemble_path) != weighting_record["ensemble_sha256"]:
        raise ValueError("Frozen CREST ensemble hash changed.")
    frames = _read_xyz_ensemble(
        ensemble_path,
        list(atoms.get_chemical_symbols()),
    )
    frames, reference_union = reference_geometry_union(
        frames,
        atoms.get_positions(),
        list(atoms.get_chemical_symbols()),
        threshold_angstrom=float(
            load_json(SCRIPT_DIR / protocol["source_evidence"]["weighting_protocol"])[
                "evaluation"
            ]["reference_geometry_rmsd_dedup_angstrom"]
        ),
    )
    if canonical_json_bytes(reference_union) != canonical_json_bytes(
        weighting_record["reference_geometry_union"]
    ):
        raise ValueError("Reconstructed reference-geometry union changed.")
    if len(frames) != weighting_record["partition_state_count"]:
        raise ValueError("Reconstructed partition-state count changed.")
    return (
        atoms,
        frames,
        reference_union,
        {
            "path": str(charge_path),
            "sha256": sha256_file(charge_path),
            "sum_e": float(charges.sum()),
            "protocol_fingerprint": charge_record.get("protocol_fingerprint"),
            "provenance": charge_record.get("provenance", {}),
        },
    )


def _load_solution_calculator(
    *,
    protocol: dict[str, Any],
    atoms,
    device: str,
    output_path: Path,
):
    import torch

    solvent = protocol["solvation"]
    calculator = SetCalculator(
        torch.device(device),
        protocol["model"]["name"],
        str(output_path),
        atoms=atoms,
        implicit=solvent["method"],
        solvent=solvent["solvent"],
        solvation_options={
            "experimental": True,
            "method": solvent["method"],
            "model": solvent["model"],
            "provider": solvent["provider"],
            "profile": solvent["profile"],
            "nonpolar": solvent["nonpolar"],
            "platform": solvent["platform"],
        },
        charge_options={
            "source": "mol2",
            "mode": solvent["charge_mode"],
            "geometry": solvent["charge_geometry"],
            "label": f"{solvent['charge_method']}-frozen",
        },
    ).set_calculator()
    checkpoint = (
        REPOSITORY_ROOT
        / "maple/function/calculator/model"
        / protocol["model"]["checkpoint_filename"]
    )
    if sha256_file(checkpoint) != protocol["model"]["checkpoint_sha256"]:
        raise ValueError("MLIP checkpoint hash does not match the relaxation protocol.")
    return calculator


def _run_case(
    *,
    case: dict[str, Any],
    protocol: dict[str, Any],
    fingerprint: str,
    candidate: dict[str, Any],
    weighting_record: dict[str, Any],
    conformer_record: dict[str, Any],
    weighting_record_sha256: str,
    gas_calculator,
    environment: dict[str, Any],
    device: str,
    output_dir: Path,
    base_work_dir: Path,
) -> dict[str, Any]:
    compound_id = case["compound_id"]
    case_dir = output_dir / "audit" / compound_id
    atoms, frames, reference_union, charge_evidence = _load_case_atoms_and_frames(
        protocol=protocol,
        candidate=candidate,
        weighting_record=weighting_record,
        conformer_record=conformer_record,
        base_work_dir=base_work_dir,
    )
    solution_calculator = _load_solution_calculator(
        protocol=protocol,
        atoms=atoms,
        device=device,
        output_path=case_dir / "solution-calculator.out",
    )
    starts = select_start_states(weighting_record)
    branches: list[dict[str, Any]] = []
    for start_state in starts:
        state_index = start_state["index"]
        initial_atoms = atoms.copy()
        initial_atoms.set_positions(frames[state_index]["positions_angstrom"])
        branch_dir = case_dir / f"state-{state_index:04d}"
        gas = _run_optimization(
            initial_atoms=initial_atoms,
            source_calculator=gas_calculator,
            optimizer_options=protocol["optimizer"],
            phase="gas",
            branch_dir=branch_dir,
        )
        solution = _run_optimization(
            initial_atoms=initial_atoms,
            source_calculator=solution_calculator,
            optimizer_options=protocol["optimizer"],
            phase="solution",
            branch_dir=branch_dir,
        )
        branches.append(
            {
                "partition_state_index": state_index,
                "roles": start_state["roles"],
                "source": frames[state_index].get("source", "crest-conformer"),
                "gas": gas,
                "solution": solution,
            }
        )
    converged_gas = [
        branch
        for branch in branches
        if branch["gas"].get("status") == "success"
        and branch["gas"].get("converged") is True
    ]
    converged_solution = [
        branch
        for branch in branches
        if branch["solution"].get("status") == "success"
        and branch["solution"].get("converged") is True
    ]
    if not converged_gas or not converged_solution:
        raise ValueError(
            "Case lacks a converged minimum for one or both phases: "
            f"gas={len(converged_gas)}, solution={len(converged_solution)}."
        )
    gas_minimum = min(
        converged_gas,
        key=lambda branch: branch["gas"]["final_energy"]["total_energy_hartree"],
    )
    solution_minimum = min(
        converged_solution,
        key=lambda branch: branch["solution"]["final_energy"]["total_energy_hartree"],
    )
    gas_energy = gas_minimum["gas"]["final_energy"]["total_energy_hartree"]
    solution_energy = solution_minimum["solution"]["final_energy"]
    decomposition = relaxed_transfer_decomposition(
        gas_min_energy_hartree=gas_energy,
        solution_min_gas_energy_hartree=solution_energy["gas_energy_hartree"],
        solution_min_solvent_energy_hartree=solution_energy["solvent_energy_hartree"],
    )
    symbols = list(atoms.get_chemical_symbols())
    heavy = np.asarray(
        [index for index, symbol in enumerate(symbols) if symbol != "H"],
        dtype=int,
    )
    if not len(heavy):
        heavy = np.arange(len(symbols), dtype=int)
    gas_positions = np.asarray(
        gas_minimum["gas"]["final_positions_angstrom"], dtype=np.float64
    )
    solution_positions = np.asarray(
        solution_minimum["solution"]["final_positions_angstrom"], dtype=np.float64
    )
    gas_energies = np.asarray(
        [
            branch["gas"]["final_energy"]["total_energy_hartree"]
            for branch in converged_gas
        ],
        dtype=np.float64,
    )
    solution_energies = np.asarray(
        [
            branch["solution"]["final_energy"]["total_energy_hartree"]
            for branch in converged_solution
        ],
        dtype=np.float64,
    )
    first_stage = weighting_record["result"]
    experimental = float(candidate["experimental_kcal_mol"])
    transfer = decomposition["relaxed_transfer_energy_kcal_mol"]
    result = {
        **decomposition,
        "gas_minimum_partition_state_index": gas_minimum["partition_state_index"],
        "gas_minimum_start_roles": gas_minimum["roles"],
        "solution_minimum_partition_state_index": solution_minimum[
            "partition_state_index"
        ],
        "solution_minimum_start_roles": solution_minimum["roles"],
        "gas_solution_minima_aligned_heavy_atom_rmsd_angstrom": _aligned_rmsd(
            gas_positions,
            solution_positions,
            heavy,
        ),
        "gas_converged_minimum_span_kcal_mol": float(
            (gas_energies.max() - gas_energies.min()) * KCAL_PER_HARTREE
        ),
        "solution_converged_minimum_span_kcal_mol": float(
            (solution_energies.max() - solution_energies.min()) * KCAL_PER_HARTREE
        ),
        "reference_geometry_kcal_mol": float(
            first_stage["reference_geometry_kcal_mol"]
        ),
        "first_stage_mlip_weighted_kcal_mol": float(first_stage["ensemble_kcal_mol"]),
        "experimental_kcal_mol": experimental,
        "relaxed_transfer_error_kcal_mol": transfer - experimental,
        "change_from_reference_geometry_kcal_mol": transfer
        - float(first_stage["reference_geometry_kcal_mol"]),
        "change_from_first_stage_weighting_kcal_mol": transfer
        - float(first_stage["ensemble_kcal_mol"]),
    }
    return {
        "schema_version": 1,
        "artifact_type": "mlip-conformer-relaxation-case",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "compound_id": compound_id,
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "selection_role": case["selection_role"],
        "experimental_kcal_mol": experimental,
        "model": protocol["model"],
        "solvation": protocol["solvation"],
        "environment": environment,
        "source_weighting_record_sha256": weighting_record_sha256,
        "source_ensemble_sha256": weighting_record["ensemble_sha256"],
        "reference_geometry_union": reference_union,
        "charge_evidence": charge_evidence,
        "selected_starts": starts,
        "branches": branches,
        "converged_branch_count": {
            "gas": len(converged_gas),
            "solution": len(converged_solution),
        },
        "result": result,
        "status": "success",
        "interpretation": protocol["claim_scope"],
    }


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_relaxation_protocol(protocol_path)
    weighting_output_dir = Path(args.weighting_output_dir).resolve()
    conformer_output_dir = Path(args.conformer_output_dir).resolve()
    base_work_dir = Path(args.base_work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    (
        weighting_summary,
        candidates,
        weighting_records,
        conformer_records,
    ) = _validate_sources(
        protocol,
        protocol_path,
        weighting_output_dir,
        conformer_output_dir,
        base_work_dir,
    )
    first_case = protocol["cases"][0]
    first_candidate = candidates[first_case["compound_id"]]
    first_atoms = MOL2Reader(
        str(base_work_dir / first_candidate["mol2_relative_path"]),
        charge=0,
        mult=1,
    )
    gas_calculator, environment = _load_calculator(
        protocol,
        args.device,
        first_atoms,
    )
    output_dir.joinpath("records").mkdir(parents=True, exist_ok=True)
    requested = set(args.case or [])
    unknown = requested.difference(case["compound_id"] for case in protocol["cases"])
    if unknown:
        raise ValueError(f"Unknown requested relaxation cases: {sorted(unknown)}")
    completed = 0
    skipped = 0
    filtered = 0
    for case in protocol["cases"]:
        compound_id = case["compound_id"]
        if requested and compound_id not in requested:
            filtered += 1
            continue
        destination = output_dir / "records" / f"{compound_id}.json"
        if destination.is_file():
            skipped += 1
            continue
        try:
            record = _run_case(
                case=case,
                protocol=protocol,
                fingerprint=fingerprint,
                candidate=candidates[compound_id],
                weighting_record=weighting_records[compound_id],
                conformer_record=conformer_records[compound_id],
                weighting_record_sha256=weighting_summary["record_sha256"][compound_id],
                gas_calculator=gas_calculator,
                environment=environment,
                device=args.device,
                output_dir=output_dir,
                base_work_dir=base_work_dir,
            )
        except Exception as exc:
            record = {
                "schema_version": 1,
                "artifact_type": "mlip-conformer-relaxation-case",
                "protocol_id": protocol["protocol_id"],
                "protocol_fingerprint": fingerprint,
                "source_partition": "development",
                "compound_id": compound_id,
                "name": case["name"],
                "flexibility_bin": case["flexibility_bin"],
                "status": "failure",
                "failure": {
                    "exception_class": type(exc).__name__,
                    "reason": str(exc),
                },
            }
        write_json_atomic(destination, record)
        completed += 1
        print(f"{compound_id}: {record['status']}", flush=True)
    print(
        "Wrote "
        f"{completed} MLIP relaxation records; resumed/skipped {skipped}; "
        f"filtered {filtered}.",
        flush=True,
    )


def _metric_block(
    predicted: np.ndarray,
    experimental: np.ndarray,
) -> dict[str, float | int]:
    errors = np.asarray(predicted) - np.asarray(experimental)
    return {
        "n": int(len(errors)),
        "mse_kcal_mol": float(errors.mean()),
        "mae_kcal_mol": float(np.abs(errors).mean()),
        "rmse_kcal_mol": float(np.sqrt(np.square(errors).mean())),
        "maximum_absolute_error_kcal_mol": float(np.abs(errors).max()),
    }


def _summary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    experimental = np.asarray(
        [record["experimental_kcal_mol"] for record in records],
        dtype=np.float64,
    )
    reference = np.asarray(
        [record["result"]["reference_geometry_kcal_mol"] for record in records],
        dtype=np.float64,
    )
    weighted = np.asarray(
        [record["result"]["first_stage_mlip_weighted_kcal_mol"] for record in records],
        dtype=np.float64,
    )
    relaxed = np.asarray(
        [record["result"]["relaxed_transfer_energy_kcal_mol"] for record in records],
        dtype=np.float64,
    )
    metrics = {
        "reference_geometry": _metric_block(reference, experimental),
        "first_stage_mlip_weighted": _metric_block(weighted, experimental),
        "relaxed_transfer_diagnostic": _metric_block(relaxed, experimental),
    }

    def outcomes(baseline: np.ndarray) -> dict[str, int]:
        counts = {"improved": 0, "unchanged": 0, "worsened": 0}
        for baseline_error, relaxed_error in zip(
            np.abs(baseline - experimental),
            np.abs(relaxed - experimental),
            strict=True,
        ):
            if relaxed_error < baseline_error - 1.0e-12:
                counts["improved"] += 1
            elif relaxed_error > baseline_error + 1.0e-12:
                counts["worsened"] += 1
            else:
                counts["unchanged"] += 1
        return counts

    metrics["case_outcomes_vs_reference_geometry"] = outcomes(reference)
    metrics["case_outcomes_vs_first_stage_mlip_weighted"] = outcomes(weighted)
    reference_gain = float(
        metrics["reference_geometry"]["mae_kcal_mol"]
        - metrics["relaxed_transfer_diagnostic"]["mae_kcal_mol"]
    )
    weighting_gain = float(
        metrics["first_stage_mlip_weighted"]["mae_kcal_mol"]
        - metrics["relaxed_transfer_diagnostic"]["mae_kcal_mol"]
    )
    without_alachlor = [
        index
        for index, record in enumerate(records)
        if record["compound_id"] != "mobley_8124669"
    ]
    if without_alachlor:
        retained_reference = reference[without_alachlor]
        retained_weighted = weighted[without_alachlor]
        retained_relaxed = relaxed[without_alachlor]
        retained_experimental = experimental[without_alachlor]
        reference_without_gain = float(
            np.abs(retained_reference - retained_experimental).mean()
            - np.abs(retained_relaxed - retained_experimental).mean()
        )
        weighting_without_gain = float(
            np.abs(retained_weighted - retained_experimental).mean()
            - np.abs(retained_relaxed - retained_experimental).mean()
        )
    else:
        reference_without_gain = float("nan")
        weighting_without_gain = float("nan")
    metrics["influence_analysis"] = {
        "reference_to_relaxed_mae_improvement_kcal_mol": reference_gain,
        "reference_to_relaxed_mae_improvement_without_alachlor_kcal_mol": reference_without_gain,
        "first_stage_weighting_to_relaxed_mae_improvement_kcal_mol": weighting_gain,
        "first_stage_weighting_to_relaxed_mae_improvement_without_alachlor_kcal_mol": weighting_without_gain,
    }
    return metrics


def summarize(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_relaxation_protocol(protocol_path)
    output_dir = Path(args.output_dir).resolve()
    expected = [case["compound_id"] for case in protocol["cases"]]
    record_dir = output_dir / "records"
    missing = [
        case_id
        for case_id in expected
        if not (record_dir / f"{case_id}.json").is_file()
    ]
    extra = sorted(
        path.stem
        for path in record_dir.glob("*.json")
        if path.stem not in set(expected)
    )
    if missing or extra:
        raise ValueError(
            "MLIP relaxation record reconciliation failed: "
            f"missing={len(missing)}, extra={len(extra)}."
        )
    records = [load_json(record_dir / f"{case_id}.json") for case_id in expected]
    invalid = [
        record["compound_id"]
        for record in records
        if record.get("protocol_fingerprint") != fingerprint
        or record.get("status") != "success"
    ]
    if invalid:
        raise ValueError(f"MLIP relaxation has invalid/failed records: {invalid}")
    environments = {canonical_json_bytes(record["environment"]) for record in records}
    if len(environments) != 1:
        raise ValueError("MLIP relaxation records used inconsistent environments.")
    phase_counts = {
        phase: {
            "attempted": sum(len(record["branches"]) for record in records),
            "successful": sum(
                branch[phase].get("status") == "success"
                for record in records
                for branch in record["branches"]
            ),
            "converged": sum(
                branch[phase].get("converged") is True
                for record in records
                for branch in record["branches"]
            ),
            "elapsed_seconds": sum(
                float(branch[phase]["elapsed_seconds"])
                for record in records
                for branch in record["branches"]
            ),
        }
        for phase in ("gas", "solution")
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "mlip-conformer-relaxation-summary",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "case_count": len(records),
        "model": protocol["model"],
        "solvation": protocol["solvation"],
        "optimizer": protocol["optimizer"],
        "environment": json.loads(next(iter(environments))),
        "optimization": phase_counts,
        "metrics": _summary_metrics(records),
        "cases": {
            record["compound_id"]: {
                "name": record["name"],
                "flexibility_bin": record["flexibility_bin"],
                "selected_start_count": len(record["selected_starts"]),
                "converged_branch_count": record["converged_branch_count"],
                **record["result"],
            }
            for record in records
        },
        "record_sha256": {
            case_id: sha256_file(record_dir / f"{case_id}.json") for case_id in expected
        },
        "interpretation": protocol["claim_scope"],
        "limitations": protocol["limitations"],
        "next_gate": protocol["next_gate"],
    }
    write_json_atomic(args.output, summary)
    print(f"Wrote MLIP conformer-relaxation summary to {Path(args.output).resolve()}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="relax frozen starts on gas and implicit-solution surfaces",
    )
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--weighting-output-dir", required=True)
    run_parser.add_argument("--conformer-output-dir", required=True)
    run_parser.add_argument("--base-work-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument(
        "--case",
        action="append",
        help="run only one frozen compound id; repeat to select more",
    )
    run_parser.set_defaults(handler=run)

    summary_parser = subparsers.add_parser(
        "summarize",
        help="summarize MLIP conformer-relaxation records",
    )
    summary_parser.add_argument("--protocol", required=True)
    summary_parser.add_argument("--output-dir", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(handler=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
