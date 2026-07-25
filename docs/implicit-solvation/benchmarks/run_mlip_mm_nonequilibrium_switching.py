#!/usr/bin/env python3
"""Run label-free MM/GB-to-Route-1-MLIP/GB nonequilibrium switching."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from ase.calculators.calculator import all_changes

BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[2]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.free_energy import (  # noqa: E402
    LinearHamiltonianCalculator,
    analyze_nonequilibrium_switching,
    run_linear_nonequilibrium_switch,
)
from run_mlip_mm_reference_reweighting import (  # noqa: E402
    ReferencePotential,
    _charge_rows,
    _direct_mbar_rows,
    _load_case_atoms,
    _load_reference_frames,
    _load_target_calculator,
    _load_target_record,
    _source_record_rows,
    _target_endpoint_frames,
    _validate_reference_inputs,
)

PROTOCOL_ID = "maple-route1-mmgb-to-mlipgb-nonequilibrium-switching-v1"
KCAL_PER_HARTREE = 627.5094740631


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _hash_matches(path: Path, expected: str, *, description: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen {description} hash changed: {path}.")


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only nonequilibrium-switching schema version 1 is supported.")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected nonequilibrium-switching protocol ID.")

    route = protocol.get("route_contract", {})
    required_true = (
        "fixed_charge_across_states",
        "no_hydration_residual_model",
        "no_mlip_retraining",
    )
    if not all(route.get(name) is True for name in required_true):
        raise ValueError("The switching protocol violates the Route 1 boundary.")
    if route.get("reference_mm_energy_in_target") is not False:
        raise ValueError("The Route 1 target may not contain an MM energy.")
    if route.get("reference_mm_role") != "endpoint sampling and switching bridge only":
        raise ValueError("The MM Hamiltonian may only sample or bridge endpoints.")
    expected_target = (
        "U_target,phase(R)=U_MLIP,gas(R)+I_phase*" "[G_polar(R,q_fixed)+G_nonpolar(R)]"
    )
    if route.get("target_formula") != expected_target:
        raise ValueError("The exact Route 1 target formula changed.")

    alignment = protocol["thermodynamic_cycle"]["energy_alignment"]
    if (
        alignment.get("same_constant_for_gas_and_solution") is not True
        or alignment.get("forces_are_unshifted") is not True
        or alignment.get("phase_specific_alignment_forbidden") is not True
    ):
        raise ValueError("Energy alignment must use one force-neutral cycle constant.")

    boundary = protocol.get("execution_boundary", {})
    for name in (
        "sampling_or_switching_reads_experimental_labels",
        "endpoint_equilibrium_claim",
        "independent_work_values_claim",
        "promotion_allowed",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"The development boundary requires {name}=false.")
    if boundary.get("target_force_evaluations_required") is not True:
        raise ValueError("Nonequilibrium switching requires target forces.")

    source = protocol["source_evidence"]
    for file_key, hash_key in (
        (
            "reference_reweighting_artifact",
            "reference_reweighting_artifact_file_sha256",
        ),
        ("direct_target_mbar_artifact", "direct_target_mbar_artifact_file_sha256"),
        ("target_ti_protocol", "target_ti_protocol_sha256"),
        ("target_record_manifest", "target_record_manifest_sha256"),
        ("charge_manifest", "charge_manifest_sha256"),
    ):
        _hash_matches(
            protocol_path.parent / source[file_key],
            source[hash_key],
            description=file_key,
        )

    switching = protocol["switching"]
    timestep = float(switching["timestep_fs"])
    lengths = [float(value) for value in switching["switch_lengths_fs"]]
    if len(lengths) != 2 or not 0.0 < lengths[0] < lengths[1]:
        raise ValueError("Exactly two increasing switch lengths are required.")
    computed_steps = {
        f"{length:.1f}": int(round(length / timestep)) for length in lengths
    }
    if computed_steps != switching["steps_by_length"]:
        raise ValueError("Switch lengths and frozen step counts disagree.")
    work_count = protocol["endpoint_sources"]["forward_reference"][
        "work_values_per_direction"
    ]
    if (
        work_count
        != protocol["endpoint_sources"]["reverse_target"]["work_values_per_direction"]
        or work_count < 2
    ):
        raise ValueError("Forward and reverse work counts must match.")
    expected_per_case = int(
        sum(
            (steps + 1) * work_count * 2 * 2
            for steps in switching["steps_by_length"].values()
        )
    )
    if (
        expected_per_case
        != switching["expected_target_energy_force_evaluations_per_model_case"]
    ):
        raise ValueError("Frozen switching operation count is inconsistent.")
    if expected_per_case * len(protocol["models"]) * len(protocol["cases"]) != (
        switching["expected_target_energy_force_evaluations_full_matrix"]
    ):
        raise ValueError("Frozen full-matrix operation count is inconsistent.")

    case_ids = [case["compound_id"] for case in protocol["cases"]]
    model_names = [model["name"] for model in protocol["models"]]
    if len(case_ids) != len(set(case_ids)) or len(model_names) != len(set(model_names)):
        raise ValueError("Case and model identities must be unique.")
    if "experimental_kcal_mol" in json.dumps(protocol, sort_keys=True).lower():
        raise ValueError("The label-free protocol contains an experimental value.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def aligned_indirect_cycle(
    reference_solvation_kcal_mol: float,
    aligned_solution_correction_kcal_mol: float,
    aligned_gas_correction_kcal_mol: float,
) -> float:
    values = np.asarray(
        (
            reference_solvation_kcal_mol,
            aligned_solution_correction_kcal_mol,
            aligned_gas_correction_kcal_mol,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Indirect-cycle terms must be finite.")
    return float(values[0] + values[1] - values[2])


def physical_endpoint_correction(
    aligned_correction_kcal_mol: float,
    common_target_offset_kcal_mol: float,
) -> float:
    values = np.asarray(
        (aligned_correction_kcal_mol, common_target_offset_kcal_mol),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Endpoint correction and energy offset must be finite.")
    return float(values.sum())


def switching_seed(
    *,
    case_seed: int,
    model_index: int,
    phase_index: int,
    direction_index: int,
    length_index: int,
    replicate_index: int,
    frame_index: int,
) -> int:
    values = (
        case_seed,
        model_index,
        phase_index,
        direction_index,
        length_index,
        replicate_index,
        frame_index,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("Switching seed inputs must be nonnegative integers.")
    return int(
        case_seed
        + model_index * 10_000_000
        + phase_index * 1_000_000
        + direction_index * 100_000
        + length_index * 10_000
        + replicate_index * 1_000
        + frame_index
    )


class OpenMMReferenceEvaluator:
    """Expose one frozen ReferencePotential phase through the MAPLE endpoint API."""

    def __init__(self, reference: ReferencePotential, phase: str):
        if phase not in {"gas", "solution"}:
            raise ValueError("Reference phase must be gas or solution.")
        self.reference = reference
        self.phase = phase

    def evaluate(self, atoms, *, need_forces: bool) -> dict[str, Any]:
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if positions.shape != (len(self.reference.symbols), 3):
            raise ValueError("Reference evaluator atom count changed.")
        context = self.reference._energy_contexts[self.phase]
        unit = self.reference._unit
        context.setPositions(positions * unit.angstrom)
        state = context.getState(getEnergy=True, getForces=need_forces)
        energy_hartree = (
            state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            / KCAL_PER_HARTREE
        )
        forces = None
        if need_forces:
            forces = (
                np.asarray(
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilocalorie_per_mole / unit.angstrom
                    ),
                    dtype=np.float64,
                )
                / KCAL_PER_HARTREE
            )
        return {
            "energy_hartree": float(energy_hartree),
            "forces_hartree_per_angstrom": forces,
        }


class Route1TargetEvaluator:
    """Expose the exact MLIP gas or MLIP+fixed-charge-GB target endpoint."""

    def __init__(self, gas_calculator, correction, phase: str):
        if phase not in {"gas", "solution"}:
            raise ValueError("Target phase must be gas or solution.")
        self.gas_calculator = gas_calculator
        self.correction = correction
        self.phase = phase

    def evaluate(self, atoms, *, need_forces: bool) -> dict[str, Any]:
        properties = ["energy", "forces"] if need_forces else ["energy"]
        self.gas_calculator.calculate(
            atoms,
            properties=properties,
            system_changes=all_changes,
        )
        energy = float(self.gas_calculator.results["energy"])
        forces = (
            np.asarray(self.gas_calculator.results["forces"], dtype=np.float64)
            if need_forces
            else None
        )
        if self.phase == "solution":
            solvent = self.correction.evaluate(atoms, need_forces=need_forces)
            energy += float(solvent.energy_hartree)
            if need_forces:
                solvent_forces = solvent.forces_hartree_per_angstrom
                if solvent_forces is None:
                    raise NotImplementedError(
                        "The selected implicit-solvent endpoint has no forces."
                    )
                forces = forces + np.asarray(solvent_forces, dtype=np.float64)
        return {
            "energy_hartree": energy,
            "forces_hartree_per_angstrom": forces,
        }


def _load_sealed_artifact(
    path: Path,
    *,
    expected_content_sha256: str,
) -> dict[str, Any]:
    artifact = load_json(path)
    if (
        artifact.get("content_sha256") != expected_content_sha256
        or artifact_content_sha256(artifact) != expected_content_sha256
    ):
        raise ValueError(f"Sealed artifact content hash changed: {path}.")
    if "experimental_kcal_mol" in json.dumps(artifact, sort_keys=True).lower():
        raise ValueError(f"Label-free source artifact contains a label: {path}.")
    return artifact


def _reference_record_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source = protocol["source_evidence"]
    artifact_path = protocol_path.parent / source["reference_reweighting_artifact"]
    artifact = _load_sealed_artifact(
        artifact_path,
        expected_content_sha256=source["reference_reweighting_artifact_content_sha256"],
    )
    if (
        artifact.get("protocol_fingerprint")
        != source["reference_reweighting_protocol_fingerprint"]
    ):
        raise ValueError("Reference-reweighting protocol fingerprint changed.")
    manifest_path = _resolve_path(artifact["raw_manifest"]["path"])
    _hash_matches(
        manifest_path,
        artifact["raw_manifest"]["file_sha256"],
        description="reference raw manifest",
    )
    manifest = _load_sealed_artifact(
        manifest_path,
        expected_content_sha256=artifact["raw_manifest"]["content_sha256"],
    )
    rows = {}
    for row in manifest["reference_records"]:
        compound_id = row["compound_id"]
        if compound_id in rows:
            raise ValueError(f"Duplicate reference record: {compound_id}.")
        rows[compound_id] = row
    return rows


def _load_reference_record(row: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_path(row["relative_path"])
    _hash_matches(path, row["file_sha256"], description="reference record")
    return _load_sealed_artifact(
        path,
        expected_content_sha256=row["content_sha256"],
    )


def _compact_reference_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    source = protocol["source_evidence"]
    artifact = _load_sealed_artifact(
        protocol_path.parent / source["reference_reweighting_artifact"],
        expected_content_sha256=source["reference_reweighting_artifact_content_sha256"],
    )
    return {(row["model"], row["compound_id"]): row for row in artifact["records"]}


def _selected_reference_endpoints(
    reference_record: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    endpoints = _load_reference_frames(reference_record)
    source = protocol["endpoint_sources"]["forward_reference"]
    required_ids = source["required_replicates"]
    indices = source["frame_indices_per_replicate"]
    selected: dict[str, list[dict[str, Any]]] = {"gas": [], "solution": []}
    for phase in selected:
        by_id = {record["replicate_id"]: record for record in endpoints[phase]}
        if set(by_id) != set(required_ids):
            raise ValueError("Reference endpoint replicate identities changed.")
        for replicate_id in required_ids:
            record = by_id[replicate_id]
            if any(index < 0 or index >= len(record["positions"]) for index in indices):
                raise ValueError("Reference endpoint frame index is out of range.")
            selected[phase].append(
                {
                    "replicate_id": replicate_id,
                    "trajectory": record["trajectory"],
                    "trajectory_sha256": record["trajectory_sha256"],
                    "frame_indices": list(indices),
                    "positions": [record["positions"][index] for index in indices],
                }
            )
    return selected


def _selected_target_endpoints(
    target_record: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    source = protocol["endpoint_sources"]["reverse_target"]
    selected: dict[str, list[dict[str, Any]]] = {"gas": [], "solution": []}
    for phase, coupling in (
        ("gas", float(source["gas_lambda"])),
        ("solution", float(source["solution_lambda"])),
    ):
        records = _target_endpoint_frames(
            target_record,
            coupling=coupling,
            expected_chain_count=len(source["required_chains"]),
            expected_sample_count=40,
        )
        by_id = {record["chain_id"]: record for record in records}
        if set(by_id) != set(source["required_chains"]):
            raise ValueError("Target endpoint chain identities changed.")
        indices = source["production_frame_indices_per_chain"]
        for chain_id in source["required_chains"]:
            record = by_id[chain_id]
            if any(index < 0 or index >= len(record["positions"]) for index in indices):
                raise ValueError("Target endpoint frame index is out of range.")
            selected[phase].append(
                {
                    "replicate_id": chain_id,
                    "trajectory": record["trajectory"],
                    "trajectory_sha256": record["trajectory_sha256"],
                    "frame_indices": list(indices),
                    "positions": [record["positions"][index] for index in indices],
                }
            )
    return selected


def _switch_replicates(
    *,
    atoms_template,
    reference_evaluator,
    target_evaluator,
    target_offset_hartree: float,
    source_records: list[dict[str, Any]],
    direction: str,
    steps: int,
    length_fs: float,
    length_index: int,
    phase_index: int,
    model_index: int,
    case_seed: int,
    switching: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, int], float]:
    if direction == "forward":
        schedule = np.linspace(0.0, 1.0, steps + 1)
        direction_index = 0
    elif direction == "reverse":
        schedule = np.linspace(1.0, 0.0, steps + 1)
        direction_index = 1
    else:
        raise ValueError("Switch direction must be forward or reverse.")

    work_replicates: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    counts = {
        "reference_energy_force_evaluations": 0,
        "target_energy_force_evaluations": 0,
    }
    elapsed = 0.0
    for replicate_index, source in enumerate(source_records):
        switches = []
        work = []
        for frame_index, positions in zip(
            source["frame_indices"],
            source["positions"],
            strict=True,
        ):
            atoms = atoms_template.copy()
            atoms.set_positions(np.asarray(positions, dtype=np.float64))
            calculator = LinearHamiltonianCalculator(
                reference_evaluator,
                target_evaluator,
                coupling=float(schedule[0]),
                target_energy_offset_hartree=target_offset_hartree,
            )
            atoms.calc = calculator
            seed = switching_seed(
                case_seed=case_seed,
                model_index=model_index,
                phase_index=phase_index,
                direction_index=direction_index,
                length_index=length_index,
                replicate_index=replicate_index,
                frame_index=frame_index,
            )
            started = time.perf_counter()
            result = run_linear_nonequilibrium_switch(
                atoms,
                calculator,
                lambda_schedule=schedule,
                temperature_kelvin=switching["temperature_kelvin"],
                timestep_fs=switching["timestep_fs"],
                friction_per_fs=switching["friction_per_fs"],
                seed=seed,
            )
            result["wall_seconds"] = time.perf_counter() - started
            result["work_kcal_mol"] = result["work_hartree"] * KCAL_PER_HARTREE
            result["switch_length_fs"] = length_fs
            result["source_frame_index"] = frame_index
            work.append(result["work_kcal_mol"])
            elapsed += result["wall_seconds"]
            for name, value in result["operation_counts"].items():
                counts[name] += value
            switches.append(result)
        work_replicates.append(np.asarray(work, dtype=np.float64))
        records.append(
            {
                "replicate_id": source["replicate_id"],
                "source_trajectory": source["trajectory"],
                "source_trajectory_sha256": source["trajectory_sha256"],
                "switches": switches,
            }
        )
    return work_replicates, records, counts, elapsed


def _analysis_kwargs(protocol: dict[str, Any]) -> dict[str, Any]:
    analysis = protocol["analysis"]
    boundary = protocol["execution_boundary"]
    return {
        "work_unit": analysis["work_unit"],
        "temperature_kelvin": protocol["switching"]["temperature_kelvin"],
        "standard_state": analysis["standard_state"],
        "endpoint_equilibrium_claim": boundary["endpoint_equilibrium_claim"],
        "independent_work_values_claim": boundary["independent_work_values_claim"],
        "minimum_work_values_per_direction": analysis[
            "minimum_work_values_per_direction"
        ],
        "minimum_independent_replicates_per_direction": analysis[
            "minimum_independent_replicates_per_direction"
        ],
        "minimum_bar_overlap": analysis["minimum_bar_overlap"],
        "maximum_forward_reverse_disagreement_kcal_mol": analysis[
            "maximum_forward_reverse_disagreement_kcal_mol"
        ],
        "maximum_bar_uncertainty_kcal_mol": analysis[
            "maximum_bar_uncertainty_kcal_mol"
        ],
        "maximum_replicate_exp_range_kcal_mol": analysis[
            "maximum_replicate_exp_range_kcal_mol"
        ],
        "maximum_leave_one_replicate_out_deviation_kcal_mol": analysis[
            "maximum_leave_one_replicate_out_deviation_kcal_mol"
        ],
        "maximum_negative_dissipation_kcal_mol": analysis[
            "maximum_negative_dissipation_kcal_mol"
        ],
        "maximum_iterations": analysis["maximum_iterations"],
        "relative_tolerance": analysis["relative_tolerance"],
    }


def _compact_analysis(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "bar_delta_g_kcal_mol": result["bar"]["delta_g_kcal_mol"],
        "bar_uncertainty_kcal_mol": result["bar"]["uncertainty_kcal_mol"],
        "bar_overlap": result["bar"]["overlap"],
        "bar_solver_diagnostics": result["bar"]["solver_diagnostics"],
        "forward_exp_delta_g_kcal_mol": result["directional"]["forward_exp"][
            "delta_g_kcal_mol"
        ],
        "reverse_exp_delta_g_kcal_mol": result["directional"]["reverse_exp"][
            "delta_g_kcal_mol"
        ],
        "forward_reverse_disagreement_kcal_mol": result["directional"][
            "absolute_forward_reverse_disagreement_kcal_mol"
        ],
        "forward_replicate_range_kcal_mol": result["directional"][
            "forward_replicate_range_kcal_mol"
        ],
        "reverse_replicate_range_kcal_mol": result["directional"][
            "reverse_replicate_range_kcal_mol"
        ],
        "leave_one_out_maximum_deviation_kcal_mol": result["leave_one_replicate_out"][
            "maximum_absolute_deviation_from_pooled_kcal_mol"
        ],
        "forward_mean_dissipation_kcal_mol": result["dissipation"][
            "forward_mean_kcal_mol"
        ],
        "reverse_mean_dissipation_kcal_mol": result["dissipation"][
            "reverse_mean_kcal_mol"
        ],
        "numerical_gates_passed": result["gates"]["numerical_gates_passed"],
        "scientific_gates_passed": result["gates"]["scientific_gates_passed"],
        "checks": result["gates"]["checks"],
    }


def _run_model_case(
    *,
    model: dict[str, Any],
    model_index: int,
    case: dict[str, Any],
    atoms,
    gas_calculator,
    correction,
    reference: ReferencePotential,
    reference_record: dict[str, Any],
    target_record: dict[str, Any],
    prior_row: dict[str, Any],
    direct_row: dict[str, Any],
    protocol: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    reference_endpoints = _selected_reference_endpoints(reference_record, protocol)
    target_endpoints = _selected_target_endpoints(target_record, protocol)
    expected_symbols = atoms.get_chemical_symbols()
    for endpoints in (reference_endpoints, target_endpoints):
        for records in endpoints.values():
            for record in records:
                if any(
                    len(frame) != len(expected_symbols) for frame in record["positions"]
                ):
                    raise ValueError("Endpoint frame atom count changed.")

    reference_evaluators = {
        phase: OpenMMReferenceEvaluator(reference, phase)
        for phase in ("gas", "solution")
    }
    target_evaluators = {
        phase: Route1TargetEvaluator(gas_calculator, correction, phase)
        for phase in ("gas", "solution")
    }

    anchor = reference_endpoints["gas"][0]["positions"][0]
    anchor_atoms = atoms.copy()
    anchor_atoms.set_positions(anchor)
    reference_gas = reference_evaluators["gas"].evaluate(
        anchor_atoms,
        need_forces=False,
    )["energy_hartree"]
    target_gas = target_evaluators["gas"].evaluate(
        anchor_atoms,
        need_forces=False,
    )["energy_hartree"]
    reference_solution = reference_evaluators["solution"].evaluate(
        anchor_atoms,
        need_forces=False,
    )["energy_hartree"]
    target_solution = target_evaluators["solution"].evaluate(
        anchor_atoms,
        need_forces=False,
    )["energy_hartree"]
    target_offset_hartree = target_gas - reference_gas
    provider_difference_kcal_mol = (
        abs((target_solution - target_gas) - (reference_solution - reference_gas))
        * KCAL_PER_HARTREE
    )

    switching = protocol["switching"]
    switch_records: dict[str, Any] = {}
    compact_analyses: dict[str, Any] = {}
    phase_analyses: dict[str, dict[str, Any]] = {}
    total_counts = {
        "reference_energy_force_evaluations": 0,
        "target_energy_force_evaluations": 0,
    }
    total_seconds = 0.0
    lengths = [float(value) for value in switching["switch_lengths_fs"]]
    for length_index, length in enumerate(lengths):
        length_key = f"{length:.1f}"
        steps = switching["steps_by_length"][length_key]
        switch_records[length_key] = {}
        compact_analyses[length_key] = {}
        phase_analyses[length_key] = {}
        for phase_index, phase in enumerate(("gas", "solution")):
            forward, forward_records, forward_counts, forward_seconds = (
                _switch_replicates(
                    atoms_template=atoms,
                    reference_evaluator=reference_evaluators[phase],
                    target_evaluator=target_evaluators[phase],
                    target_offset_hartree=target_offset_hartree,
                    source_records=reference_endpoints[phase],
                    direction="forward",
                    steps=steps,
                    length_fs=length,
                    length_index=length_index,
                    phase_index=phase_index,
                    model_index=model_index,
                    case_seed=case["seed"],
                    switching=switching,
                )
            )
            reverse, reverse_records, reverse_counts, reverse_seconds = (
                _switch_replicates(
                    atoms_template=atoms,
                    reference_evaluator=reference_evaluators[phase],
                    target_evaluator=target_evaluators[phase],
                    target_offset_hartree=target_offset_hartree,
                    source_records=target_endpoints[phase],
                    direction="reverse",
                    steps=steps,
                    length_fs=length,
                    length_index=length_index,
                    phase_index=phase_index,
                    model_index=model_index,
                    case_seed=case["seed"],
                    switching=switching,
                )
            )
            analysis = analyze_nonequilibrium_switching(
                forward,
                reverse,
                **_analysis_kwargs(protocol),
            )
            switch_records[length_key][phase] = {
                "forward": forward_records,
                "reverse": reverse_records,
            }
            phase_analyses[length_key][phase] = analysis
            compact_analyses[length_key][phase] = _compact_analysis(analysis)
            for name in total_counts:
                total_counts[name] += forward_counts[name] + reverse_counts[name]
            total_seconds += forward_seconds + reverse_seconds

    expected_calls = switching[
        "expected_target_energy_force_evaluations_per_model_case"
    ]
    if total_counts["target_energy_force_evaluations"] != expected_calls:
        raise ValueError(
            "Observed target force-call count does not match the frozen protocol."
        )

    reference_solvation = float(prior_row["reference_solvation_mbar_kcal_mol"])
    direct_target = float(direct_row["mbar"]["endpoint_delta_g_kcal_mol"])
    cycles = {}
    for length in lengths:
        key = f"{length:.1f}"
        gas_delta = phase_analyses[key]["gas"]["bar"]["delta_g_kcal_mol"]
        solution_delta = phase_analyses[key]["solution"]["bar"]["delta_g_kcal_mol"]
        indirect = aligned_indirect_cycle(
            reference_solvation,
            solution_delta,
            gas_delta,
        )
        cycles[key] = {
            "reference_solvation_mbar_kcal_mol": reference_solvation,
            "aligned_gas_reference_to_target_bar_kcal_mol": gas_delta,
            "aligned_solution_reference_to_target_bar_kcal_mol": solution_delta,
            "physical_gas_reference_to_target_bar_kcal_mol": (
                physical_endpoint_correction(
                    gas_delta,
                    target_offset_hartree * KCAL_PER_HARTREE,
                )
            ),
            "physical_solution_reference_to_target_bar_kcal_mol": (
                physical_endpoint_correction(
                    solution_delta,
                    target_offset_hartree * KCAL_PER_HARTREE,
                )
            ),
            "net_reference_to_target_correction_kcal_mol": (solution_delta - gas_delta),
            "indirect_target_solvation_kcal_mol": indirect,
            "direct_target_multistate_mbar_kcal_mol": direct_target,
            "indirect_minus_direct_mbar_kcal_mol": indirect - direct_target,
        }

    short_key = f"{lengths[0]:.1f}"
    long_key = f"{lengths[1]:.1f}"
    convergence = {
        "gas_short_long_bar_difference_kcal_mol": abs(
            phase_analyses[short_key]["gas"]["bar"]["delta_g_kcal_mol"]
            - phase_analyses[long_key]["gas"]["bar"]["delta_g_kcal_mol"]
        ),
        "solution_short_long_bar_difference_kcal_mol": abs(
            phase_analyses[short_key]["solution"]["bar"]["delta_g_kcal_mol"]
            - phase_analyses[long_key]["solution"]["bar"]["delta_g_kcal_mol"]
        ),
        "cycle_short_long_difference_kcal_mol": abs(
            cycles[short_key]["indirect_target_solvation_kcal_mol"]
            - cycles[long_key]["indirect_target_solvation_kcal_mol"]
        ),
    }
    analysis_threshold = protocol["analysis"]
    checks = {
        "provider_parity": provider_difference_kcal_mol
        <= analysis_threshold[
            "maximum_reference_target_solvent_provider_difference_kcal_mol"
        ],
        "expected_target_force_calls": total_counts["target_energy_force_evaluations"]
        == expected_calls,
        "short_gas_numerical_gates": phase_analyses[short_key]["gas"]["gates"][
            "numerical_gates_passed"
        ],
        "short_solution_numerical_gates": phase_analyses[short_key]["solution"][
            "gates"
        ]["numerical_gates_passed"],
        "long_gas_numerical_gates": phase_analyses[long_key]["gas"]["gates"][
            "numerical_gates_passed"
        ],
        "long_solution_numerical_gates": phase_analyses[long_key]["solution"]["gates"][
            "numerical_gates_passed"
        ],
        "gas_switch_length_convergence": convergence[
            "gas_short_long_bar_difference_kcal_mol"
        ]
        <= analysis_threshold["maximum_short_long_bar_difference_kcal_mol"],
        "solution_switch_length_convergence": convergence[
            "solution_short_long_bar_difference_kcal_mol"
        ]
        <= analysis_threshold["maximum_short_long_bar_difference_kcal_mol"],
        "cycle_switch_length_convergence": convergence[
            "cycle_short_long_difference_kcal_mol"
        ]
        <= analysis_threshold["maximum_short_long_bar_difference_kcal_mol"],
        "long_cycle_direct_mbar_agreement": abs(
            cycles[long_key]["indirect_minus_direct_mbar_kcal_mol"]
        )
        <= analysis_threshold[
            "maximum_long_cycle_vs_direct_target_mbar_difference_kcal_mol"
        ],
    }
    record = {
        "schema_version": 1,
        "artifact_type": "route1-mmgb-to-mlipgb-nonequilibrium-switching-case",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "model": model["name"],
        "compound_id": case["compound_id"],
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "route_contract": {
            "reference_mm_energy_in_target": False,
            "target_formula": protocol["route_contract"]["target_formula"],
            "no_hydration_residual_model": True,
            "no_mlip_retraining": True,
        },
        "energy_alignment": {
            "anchor_source": {
                "replicate_id": reference_endpoints["gas"][0]["replicate_id"],
                "frame_index": reference_endpoints["gas"][0]["frame_indices"][0],
                "trajectory": reference_endpoints["gas"][0]["trajectory"],
                "trajectory_sha256": reference_endpoints["gas"][0]["trajectory_sha256"],
            },
            "target_offset_hartree": target_offset_hartree,
            "target_offset_kcal_mol": target_offset_hartree * KCAL_PER_HARTREE,
            "same_constant_for_gas_and_solution": True,
            "forces_shifted": False,
            "phase_specific_alignment_used": False,
        },
        "provider_parity": {
            "anchor_reference_solvent_kcal_mol": (reference_solution - reference_gas)
            * KCAL_PER_HARTREE,
            "anchor_target_solvent_kcal_mol": (target_solution - target_gas)
            * KCAL_PER_HARTREE,
            "absolute_difference_kcal_mol": provider_difference_kcal_mol,
            "prior_reference_reweighting_maximum_difference_kcal_mol": prior_row[
                "maximum_provider_difference_kcal_mol"
            ],
        },
        "switching_records": switch_records,
        "analyses": compact_analyses,
        "cycles": cycles,
        "switch_length_convergence": convergence,
        "numerical_checks": checks,
        "all_numerical_checks_pass": all(checks.values()),
        "operation_counts": total_counts,
        "runtime_observations": {
            "switching_wall_seconds": total_seconds,
        },
        "label_boundary": {
            "experimental_labels_read": False,
            "scoring_performed": False,
        },
        "claim_boundary": {
            "endpoint_equilibrium_proven": False,
            "independent_work_values_proven": False,
            "chemical_accuracy_established": False,
            "acceleration_established": False,
            "product_promotion_allowed": False,
        },
    }
    return seal_artifact(record)


def _load_reusable_record(
    path: Path,
    *,
    protocol_fingerprint: str,
    identity: tuple[str, str],
) -> dict[str, Any]:
    record = load_json(path)
    if (
        record.get("content_sha256") != artifact_content_sha256(record)
        or record.get("artifact_type")
        != "route1-mmgb-to-mlipgb-nonequilibrium-switching-case"
        or record.get("protocol_fingerprint") != protocol_fingerprint
        or (record.get("model"), record.get("compound_id")) != identity
    ):
        raise ValueError(f"Cannot resume unverified switching record: {path}.")
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = load_protocol(protocol_path)
    selected_cases = [
        case
        for case in protocol["cases"]
        if not args.case or case["compound_id"] in set(args.case)
    ]
    selected_models = [
        model
        for model in protocol["models"]
        if not args.model or model["name"] in set(args.model)
    ]
    if not selected_cases or not selected_models:
        raise ValueError("At least one frozen case and model must be selected.")

    reference_input_dir = Path(args.reference_input_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = _reference_record_rows(protocol_path, protocol)
    prior_rows = _compact_reference_rows(protocol_path, protocol)
    target_rows = _source_record_rows(protocol_path, protocol)
    direct_rows = _direct_mbar_rows(protocol_path, protocol)
    charge_rows = _charge_rows(protocol_path, protocol)

    device_text = str(args.device).strip().lower()
    if device_text.startswith("gpu"):
        device_text = f"cuda:{device_text[3:] or '0'}"
    device = torch.device(device_text)
    records: list[dict[str, Any]] = []
    raw_paths: dict[tuple[str, str], Path] = {}
    checkpoint_records: dict[str, dict[str, str]] = {}
    for model in selected_models:
        model_index = next(
            index
            for index, candidate in enumerate(protocol["models"])
            if candidate["name"] == model["name"]
        )
        first_case_dir = reference_input_dir / selected_cases[0]["compound_id"]
        first_atoms = _load_case_atoms(
            first_case_dir,
            charge_rows[selected_cases[0]["compound_id"]],
        )
        gas_calculator, checkpoint = _load_target_calculator(
            model,
            device,
            first_atoms,
            raw_dir / f"{model['name']}-calculator.log",
        )
        checkpoint_records[model["name"]] = {
            "path": _display_path(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
        for case in selected_cases:
            identity = (model["name"], case["compound_id"])
            record_path = raw_dir / model["name"] / f"{case['compound_id']}.json"
            if args.resume_existing and record_path.is_file():
                record = _load_reusable_record(
                    record_path,
                    protocol_fingerprint=fingerprint,
                    identity=identity,
                )
            else:
                case_dir = reference_input_dir / case["compound_id"]
                _validate_reference_inputs(case_dir, case)
                atoms = _load_case_atoms(case_dir, charge_rows[case["compound_id"]])
                correction = ImplicitSolvationCorrection(
                    atoms,
                    {
                        "source": "mol2",
                        "mode": "fixed",
                        "geometry": "keep",
                        "label": "am1bcc-frozen-manifest",
                    },
                    {
                        "implicit": "water",
                        "method": "gb",
                        "model": "obc2",
                        "nonpolar": "ace",
                        "platform": "Reference",
                        "experimental": True,
                    },
                    output=(
                        raw_dir
                        / model["name"]
                        / f"{case['compound_id']}-correction.out"
                    ),
                )
                reference = ReferencePotential(case_dir, protocol)
                record = _run_model_case(
                    model=model,
                    model_index=model_index,
                    case=case,
                    atoms=atoms,
                    gas_calculator=gas_calculator,
                    correction=correction,
                    reference=reference,
                    reference_record=_load_reference_record(
                        reference_rows[case["compound_id"]]
                    ),
                    target_record=_load_target_record(target_rows[identity]),
                    prior_row=prior_rows[identity],
                    direct_row=direct_rows[identity],
                    protocol=protocol,
                    fingerprint=fingerprint,
                )
                record_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(record_path, record)
            records.append(record)
            raw_paths[identity] = record_path
        del gas_calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = seal_artifact(
        {
            "schema_version": 1,
            "manifest_id": "maple-route1-mmgb-to-mlipgb-nonequilibrium-raw-v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": fingerprint,
            "records": [
                {
                    "model": model,
                    "compound_id": compound_id,
                    "relative_path": _display_path(path),
                    "file_sha256": sha256_file(path),
                    "content_sha256": next(
                        record["content_sha256"]
                        for record in records
                        if record["model"] == model
                        and record["compound_id"] == compound_id
                    ),
                }
                for (model, compound_id), path in sorted(raw_paths.items())
            ],
            "label_boundary": {
                "experimental_labels_read": False,
            },
        }
    )
    manifest_path = raw_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    lengths = [
        f"{float(value):.1f}" for value in protocol["switching"]["switch_lengths_fs"]
    ]
    long_key = lengths[-1]
    compact_records = [
        {
            "model": record["model"],
            "compound_id": record["compound_id"],
            "name": record["name"],
            "flexibility_bin": record["flexibility_bin"],
            "gas_bar_overlap": record["analyses"][long_key]["gas"]["bar_overlap"],
            "solution_bar_overlap": record["analyses"][long_key]["solution"][
                "bar_overlap"
            ],
            "gas_forward_reverse_disagreement_kcal_mol": record["analyses"][long_key][
                "gas"
            ]["forward_reverse_disagreement_kcal_mol"],
            "solution_forward_reverse_disagreement_kcal_mol": record["analyses"][
                long_key
            ]["solution"]["forward_reverse_disagreement_kcal_mol"],
            "short_long_cycle_difference_kcal_mol": record["switch_length_convergence"][
                "cycle_short_long_difference_kcal_mol"
            ],
            "indirect_target_solvation_kcal_mol": record["cycles"][long_key][
                "indirect_target_solvation_kcal_mol"
            ],
            "direct_target_multistate_mbar_kcal_mol": record["cycles"][long_key][
                "direct_target_multistate_mbar_kcal_mol"
            ],
            "indirect_minus_direct_mbar_kcal_mol": record["cycles"][long_key][
                "indirect_minus_direct_mbar_kcal_mol"
            ],
            "all_numerical_checks_pass": record["all_numerical_checks_pass"],
            "numerical_checks": record["numerical_checks"],
            "target_energy_force_evaluations": record["operation_counts"][
                "target_energy_force_evaluations"
            ],
            "switching_wall_seconds": record["runtime_observations"][
                "switching_wall_seconds"
            ],
            "raw_record": _display_path(
                raw_paths[(record["model"], record["compound_id"])]
            ),
            "raw_record_sha256": sha256_file(
                raw_paths[(record["model"], record["compound_id"])]
            ),
        }
        for record in records
    ]
    matrix_complete = len(selected_models) == len(protocol["models"]) and len(
        selected_cases
    ) == len(protocol["cases"])
    output = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-mmgb-to-mlipgb-nonequilibrium-switching-summary",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": fingerprint,
            "record_count": len(compact_records),
            "model_count": len(selected_models),
            "case_count": len(selected_cases),
            "matrix_complete": matrix_complete,
            "records": compact_records,
            "aggregate_diagnostics": {
                "numerical_gate_pass_count": sum(
                    row["all_numerical_checks_pass"] for row in compact_records
                ),
                "minimum_gas_bar_overlap": min(
                    row["gas_bar_overlap"] for row in compact_records
                ),
                "minimum_solution_bar_overlap": min(
                    row["solution_bar_overlap"] for row in compact_records
                ),
                "maximum_short_long_cycle_difference_kcal_mol": max(
                    row["short_long_cycle_difference_kcal_mol"]
                    for row in compact_records
                ),
                "maximum_absolute_indirect_minus_direct_mbar_kcal_mol": max(
                    abs(row["indirect_minus_direct_mbar_kcal_mol"])
                    for row in compact_records
                ),
                "total_target_energy_force_evaluations": sum(
                    row["target_energy_force_evaluations"] for row in compact_records
                ),
                "total_switching_wall_seconds": sum(
                    row["switching_wall_seconds"] for row in compact_records
                ),
            },
            "checkpoint_records": checkpoint_records,
            "raw_manifest": {
                "path": _display_path(manifest_path),
                "file_sha256": sha256_file(manifest_path),
                "content_sha256": manifest["content_sha256"],
            },
            "implementation_provenance": {
                "runner": {
                    "path": _display_path(Path(__file__)),
                    "sha256": sha256_file(Path(__file__)),
                },
                "switching_engine": {
                    "path": "maple/function/free_energy/switching.py",
                    "sha256": sha256_file(
                        REPOSITORY_ROOT / "maple/function/free_energy/switching.py"
                    ),
                },
                "nonequilibrium_analysis": {
                    "path": "maple/function/free_energy/nonequilibrium.py",
                    "sha256": sha256_file(
                        REPOSITORY_ROOT / "maple/function/free_energy/nonequilibrium.py"
                    ),
                },
                "protocol": {
                    "path": _display_path(protocol_path),
                    "sha256": sha256_file(protocol_path),
                },
            },
            "label_boundary": {
                "experimental_labels_read": False,
                "scoring_performed": False,
            },
            "decision": {
                "promotion_allowed": False,
                "candidate_validated": False,
                "reason": (
                    "This prospectively frozen run is a short, label-free "
                    "nonequilibrium diagnostic with reused non-equilibrium endpoints."
                ),
            },
            "claim_boundary": {
                "endpoint_equilibrium_proven": False,
                "independent_work_values_proven": False,
                "chemical_accuracy_established": False,
                "acceleration_established": False,
                "product_promotion_allowed": False,
            },
        }
    )
    write_json_atomic(Path(args.output).resolve(), output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(BENCHMARK_DIR / "mlip_mm_nonequilibrium_switching_protocol.json"),
    )
    parser.add_argument(
        "--reference-input-dir",
        default=str(BENCHMARK_DIR / "route1-mm-reference-reweighting-raw" / "inputs"),
    )
    parser.add_argument(
        "--raw-dir",
        default=str(BENCHMARK_DIR / "route1-mm-mlip-nonequilibrium-switching-raw"),
    )
    parser.add_argument(
        "--output",
        default=str(
            BENCHMARK_DIR / "route1-mm-mlip-nonequilibrium-switching-2026-07-25.json"
        ),
    )
    parser.add_argument("--device", default="gpu0")
    parser.add_argument("--model", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--resume-existing", action="store_true")
    return parser


if __name__ == "__main__":
    artifact = run(build_parser().parse_args())
    print(json.dumps(artifact["aggregate_diagnostics"], indent=2))
