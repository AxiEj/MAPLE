#!/usr/bin/env python3
"""Audit MM/GB reference sampling with sparse Route 1 MLIP reweighting."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

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
    analyze_bidirectional_reweighting,
    analyze_mbar,
)
from run_route1_task_matrix import read_xyz_frames  # noqa: E402

PROTOCOL_ID = "maple-route1-mmgb-reference-mlip-reweighting-v1"
KCAL_PER_HARTREE = 627.5094740631
R_KCAL_PER_MOL_K = 0.00198720425864083


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_record_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPOSITORY_ROOT / candidate


def _finite_array(values: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only reference-reweighting schema version 1 is supported.")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected reference-reweighting protocol ID.")

    route = protocol.get("route_contract", {})
    required_true = (
        "fixed_charge_across_states",
        "no_hydration_residual_model",
        "no_mlip_retraining",
    )
    if not all(route.get(key) is True for key in required_true):
        raise ValueError("The protocol violates the fixed-charge Route 1 boundary.")
    if route.get("reference_mm_role") != "sampling_only":
        raise ValueError("The MM potential may only be a reference sampler.")
    if route.get("reference_mm_energy_in_target") is not False:
        raise ValueError("The Route 1 target may not contain an MM energy.")
    expected_formula = (
        "U_target,lambda(R)=U_MLIP,gas(R)+lambda*" "[G_polar(R,q_fixed)+G_nonpolar(R)]"
    )
    if route.get("target_formula") != expected_formula:
        raise ValueError("The exact Route 1 target formula changed.")

    boundary = protocol.get("execution_boundary", {})
    if boundary.get("sampling_or_energy_phase_reads_experimental_labels") is not False:
        raise ValueError("Reference reweighting must remain label-free.")
    if boundary.get("equilibrium_claim") is not False:
        raise ValueError("The short diagnostic cannot claim equilibrium.")
    if boundary.get("promotion_allowed") is not False:
        raise ValueError("The short diagnostic cannot promote a product method.")
    if (
        boundary.get("target_samples_used_only_for_bidirectional_validation")
        is not True
    ):
        raise ValueError("Target samples must remain validation-only.")

    source = protocol["source_evidence"]
    for file_key, hash_key in (
        ("target_ti_protocol", "target_ti_protocol_sha256"),
        ("target_record_manifest", "target_record_manifest_sha256"),
        ("direct_target_mbar_artifact", "direct_target_mbar_artifact_sha256"),
        ("charge_manifest", "charge_manifest_sha256"),
    ):
        source_path = protocol_path.parent / source[file_key]
        if sha256_file(source_path) != source[hash_key]:
            raise ValueError(f"Frozen source hash changed: {source_path.name}.")

    sampling = protocol["reference_sampling"]
    expected_samples = (
        sampling["production_steps_per_endpoint"] // sampling["sample_stride_steps"]
    )
    integer_fields = (
        "equilibration_steps_per_endpoint",
        "production_steps_per_endpoint",
        "sample_stride_steps",
        "expected_samples_per_endpoint_replicate",
    )
    if any(
        isinstance(sampling[name], bool)
        or not isinstance(sampling[name], int)
        or sampling[name] < 1
        for name in integer_fields
    ):
        raise ValueError("Reference sampling counts must be positive integers.")
    if expected_samples != sampling["expected_samples_per_endpoint_replicate"]:
        raise ValueError("Reference sampling count does not match the frozen protocol.")
    if len(sampling.get("replicates", [])) < 2:
        raise ValueError("At least two independent reference replicates are required.")

    case_ids = [case["compound_id"] for case in protocol["cases"]]
    model_names = [model["name"] for model in protocol["models"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Reference-reweighting case IDs must be unique.")
    if len(model_names) != len(set(model_names)):
        raise ValueError("Reference-reweighting model names must be unique.")

    lowered = json.dumps(protocol, sort_keys=True).lower()
    if "experimental_kcal_mol" in lowered:
        raise ValueError("The label-free protocol contains an experimental value.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _source_record_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    source = protocol["source_evidence"]
    manifest = load_json(protocol_path.parent / source["target_record_manifest"])
    records = manifest.get("records", [])
    if manifest.get("schema_version") != 1 or len(records) != 9:
        raise ValueError("Unexpected target-record source manifest.")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        identity = (row["model"], row["compound_id"])
        if identity in rows:
            raise ValueError(f"Duplicate target source identity: {identity}.")
        rows[identity] = row
    if "experimental_kcal_mol" in json.dumps(manifest, sort_keys=True).lower():
        raise ValueError("The target source manifest contains an experimental label.")
    return rows


def _load_target_record(row: dict[str, Any]) -> dict[str, Any]:
    path = _resolve_record_path(row["relative_path"])
    if sha256_file(path) != row["file_sha256"]:
        raise ValueError(f"Target source file hash changed: {path}.")
    record = load_json(path)
    if (
        record.get("content_sha256") != row["content_sha256"]
        or artifact_content_sha256(record) != row["content_sha256"]
    ):
        raise ValueError(f"Target source content hash changed: {path}.")
    if (record.get("model"), record.get("compound_id")) != (
        row["model"],
        row["compound_id"],
    ):
        raise ValueError(f"Target source identity changed: {path}.")
    if "experimental_kcal_mol" in json.dumps(record, sort_keys=True).lower():
        raise ValueError(
            f"Target source record contains an experimental label: {path}."
        )
    return record


def _direct_mbar_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    source = protocol["source_evidence"]
    path = protocol_path.parent / source["direct_target_mbar_artifact"]
    artifact = load_json(path)
    expected = artifact.get("content_sha256")
    if expected != artifact_content_sha256(artifact):
        raise ValueError("The direct-target MBAR artifact has an invalid self-hash.")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in artifact.get("records", []):
        identity = (row["model"], row["compound_id"])
        if identity in rows:
            raise ValueError(f"Duplicate direct MBAR identity: {identity}.")
        rows[identity] = row
    return rows


def _charge_rows(
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source = protocol["source_evidence"]
    manifest = load_json(protocol_path.parent / source["charge_manifest"])
    return {row["compound_id"]: row for row in manifest["records"]}


def _validate_reference_inputs(case_dir: Path, case: dict[str, Any]) -> None:
    for name, expected in case["reference_input_sha256"].items():
        path = case_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen reference input changed: {path}.")


def _copy_reference_inputs(
    source_case_dir: Path,
    raw_dir: Path,
    case: dict[str, Any],
) -> dict[str, dict[str, str]]:
    destination = raw_dir / "inputs" / case["compound_id"]
    destination.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, str]] = {}
    for name, expected in case["reference_input_sha256"].items():
        source = source_case_dir / name
        target = destination / name
        shutil.copyfile(source, target)
        observed = sha256_file(target)
        if observed != expected:
            raise ValueError(f"Published reference input hash mismatch: {target}.")
        records[name] = {
            "path": _display_path(target),
            "sha256": observed,
        }
    return records


def _load_case_atoms(
    case_dir: Path,
    charge_row: dict[str, Any],
):
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    atoms = MOL2Reader(
        str(case_dir / "m.mol2"),
        charge=0,
        mult=1,
        validate_charge=True,
    )
    expected = np.asarray(charge_row["am1bcc_charges_e"], dtype=np.float64)
    observed = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    if (
        observed.shape != expected.shape
        or np.max(np.abs(observed - expected)) > 1.0e-10
    ):
        raise ValueError(
            "Reference topology does not contain the frozen AM1-BCC vector."
        )
    return atoms


def _write_xyz(
    path: Path,
    symbols: list[str],
    frames: list[np.ndarray],
    *,
    comment_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, positions in enumerate(frames):
        array = np.asarray(positions, dtype=np.float64)
        if array.shape != (len(symbols), 3) or not np.all(np.isfinite(array)):
            raise ValueError("Cannot write a non-finite or shape-mismatched XYZ frame.")
        lines.extend((str(len(symbols)), f"{comment_prefix} sample={index}"))
        lines.extend(
            f"{symbol} {xyz[0]:.15f} {xyz[1]:.15f} {xyz[2]:.15f}"
            for symbol, xyz in zip(symbols, array, strict=True)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ReferencePotential:
    """Frozen GAFF2 gas and GAFF2/OBC-II/ACE reference Hamiltonians."""

    def __init__(self, case_dir: Path, protocol: dict[str, Any]):
        from openmm import Context, Platform, VerletIntegrator, unit
        from openmm.app import (
            AmberInpcrdFile,
            AmberPrmtopFile,
            NoCutoff,
            OBC2,
        )

        specification = protocol["reference_potential"]
        self._unit = unit
        self._context_type = Context
        self._platform = Platform.getPlatformByName(specification["openmm_platform"])
        self._platform_properties = dict(specification["openmm_platform_properties"])
        self._prmtop = AmberPrmtopFile(str(case_dir / "p"))
        self._inpcrd = AmberInpcrdFile(str(case_dir / "initial.rst7"))
        common = {
            "nonbondedMethod": NoCutoff,
            "constraints": None,
            "soluteDielectric": specification["solute_dielectric"],
            "solventDielectric": specification["solvent_dielectric"],
            "removeCMMotion": True,
        }
        self._systems = {
            "gas": self._prmtop.createSystem(
                implicitSolvent=None,
                **common,
            ),
            "solution": self._prmtop.createSystem(
                implicitSolvent=OBC2,
                implicitSolventSaltConc=(
                    specification["salt_concentration_molar"] * unit.mole / unit.liter
                ),
                sasaMethod=specification["sasa_method"],
                **common,
            ),
        }
        self.symbols = [atom.element.symbol for atom in self._prmtop.topology.atoms()]
        self._energy_contexts = {}
        self._energy_integrators = {}
        for phase, system in self._systems.items():
            integrator = VerletIntegrator(0.5 * unit.femtoseconds)
            context = Context(
                system,
                integrator,
                self._platform,
                self._platform_properties,
            )
            self._energy_integrators[phase] = integrator
            self._energy_contexts[phase] = context

    def energies(
        self,
        frames_angstrom: Iterable[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        gas: list[float] = []
        solution: list[float] = []
        for frame in frames_angstrom:
            positions = np.asarray(frame, dtype=np.float64)
            if positions.shape != (len(self.symbols), 3):
                raise ValueError("Reference energy frame has the wrong atom count.")
            for phase, values in (("gas", gas), ("solution", solution)):
                context = self._energy_contexts[phase]
                context.setPositions(positions * self._unit.angstrom)
                state = context.getState(getEnergy=True)
                values.append(
                    state.getPotentialEnergy().value_in_unit(
                        self._unit.kilocalorie_per_mole
                    )
                )
        return (
            _finite_array(gas, name="reference gas energies"),
            _finite_array(solution, name="reference solution energies"),
        )

    def sample(
        self,
        *,
        phase: str,
        seed: int,
        protocol: dict[str, Any],
    ) -> tuple[list[np.ndarray], float]:
        from openmm import LangevinMiddleIntegrator

        sampling = protocol["reference_sampling"]
        integrator = LangevinMiddleIntegrator(
            sampling["temperature_kelvin"] * self._unit.kelvin,
            sampling["friction_per_ps"] / self._unit.picosecond,
            sampling["timestep_fs"] * self._unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(int(seed))
        context = self._context_type(
            self._systems[phase],
            integrator,
            self._platform,
            self._platform_properties,
        )
        context.setPositions(self._inpcrd.positions)
        context.setVelocitiesToTemperature(
            sampling["temperature_kelvin"] * self._unit.kelvin,
            int(seed) + 1,
        )
        started = time.perf_counter()
        integrator.step(sampling["equilibration_steps_per_endpoint"])
        frames: list[np.ndarray] = []
        stride = sampling["sample_stride_steps"]
        for _ in range(sampling["expected_samples_per_endpoint_replicate"]):
            integrator.step(stride)
            state = context.getState(getPositions=True)
            frames.append(
                np.asarray(
                    state.getPositions(asNumpy=True).value_in_unit(self._unit.angstrom),
                    dtype=np.float64,
                )
            )
        elapsed = time.perf_counter() - started
        del context
        del integrator
        return frames, elapsed


def _reference_record(
    *,
    case: dict[str, Any],
    case_dir: Path,
    raw_dir: Path,
    protocol: dict[str, Any],
    protocol_fingerprint: str,
    published_inputs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    potential = ReferencePotential(case_dir, protocol)
    endpoints: dict[str, list[dict[str, Any]]] = {"gas": [], "solution": []}
    sampling = protocol["reference_sampling"]
    for phase, phase_offset in (
        ("gas", sampling["gas_seed_offset"]),
        ("solution", sampling["solution_seed_offset"]),
    ):
        for repeat in sampling["replicates"]:
            seed = case["seed"] + phase_offset + repeat["seed_offset"]
            frames, elapsed = potential.sample(
                phase=phase,
                seed=seed,
                protocol=protocol,
            )
            gas, solution = potential.energies(frames)
            trajectory = (
                raw_dir
                / "trajectories"
                / case["compound_id"]
                / phase
                / f"{repeat['replicate_id']}.xyz"
            )
            _write_xyz(
                trajectory,
                potential.symbols,
                frames,
                comment_prefix=(
                    f"{PROTOCOL_ID} phase={phase} "
                    f"replicate={repeat['replicate_id']}"
                ),
            )
            endpoints[phase].append(
                {
                    "replicate_id": repeat["replicate_id"],
                    "seed": seed,
                    "trajectory": _display_path(trajectory),
                    "trajectory_sha256": sha256_file(trajectory),
                    "sample_count": len(frames),
                    "reference_gas_kcal_mol": gas.tolist(),
                    "reference_solution_kcal_mol": solution.tolist(),
                    "sampling_wall_seconds": elapsed,
                }
            )
    record = {
        "schema_version": 1,
        "artifact_type": "route1-mmgb-reference-samples",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": protocol_fingerprint,
        "compound_id": case["compound_id"],
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "published_reference_inputs": published_inputs,
        "openmm_version": importlib.metadata.version("openmm"),
        "endpoints": endpoints,
        "label_boundary": {
            "experimental_labels_read": False,
        },
        "claim_boundary": {
            "equilibrium_proven": False,
            "conformer_mixing_proven": False,
            "product_promotion_allowed": False,
        },
    }
    return seal_artifact(record)


def _load_reference_frames(
    reference_record: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    endpoints: dict[str, list[dict[str, Any]]] = {"gas": [], "solution": []}
    for phase in endpoints:
        for repeat in reference_record["endpoints"][phase]:
            trajectory = _resolve_record_path(repeat["trajectory"])
            if sha256_file(trajectory) != repeat["trajectory_sha256"]:
                raise ValueError(f"Reference trajectory hash changed: {trajectory}.")
            parsed = read_xyz_frames(trajectory)
            if len(parsed) != repeat["sample_count"]:
                raise ValueError("Reference trajectory sample count changed.")
            endpoints[phase].append(
                {
                    **repeat,
                    "positions": [
                        np.asarray(frame["positions_angstrom"], dtype=np.float64)
                        for frame in parsed
                    ],
                    "symbols": [frame["symbols"] for frame in parsed],
                    "reference_gas": _finite_array(
                        repeat["reference_gas_kcal_mol"],
                        name="reference gas samples",
                    ),
                    "reference_solution": _finite_array(
                        repeat["reference_solution_kcal_mol"],
                        name="reference solution samples",
                    ),
                }
            )
    return endpoints


def _target_endpoint_frames(
    record: dict[str, Any],
    *,
    coupling: float,
    expected_chain_count: int,
    expected_sample_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chain in record.get("chains", []):
        matches = [
            window
            for window in chain["windows"]
            if np.isclose(float(window["lambda"]), coupling)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Target chain {chain['chain_id']} lacks lambda={coupling}."
            )
        window = matches[0]
        trajectory = _resolve_record_path(window["trajectory"])
        if sha256_file(trajectory) != window["trajectory_sha256"]:
            raise ValueError(f"Target trajectory hash changed: {trajectory}.")
        frames = read_xyz_frames(trajectory)
        discarded = int(window["discarded_frame_count"])
        production = frames[discarded:]
        samples = window["samples"]
        if (
            len(production) != expected_sample_count
            or len(samples) != expected_sample_count
            or window["production_sample_count"] != expected_sample_count
        ):
            raise ValueError("Target production-frame selection changed.")
        gas = (
            np.asarray(
                [sample["gas_energy_hartree"] for sample in samples],
                dtype=np.float64,
            )
            * KCAL_PER_HARTREE
        )
        solution = (
            gas
            + np.asarray(
                [sample["solvent_energy_hartree"] for sample in samples],
                dtype=np.float64,
            )
            * KCAL_PER_HARTREE
        )
        if not np.all(np.isfinite(gas)) or not np.all(np.isfinite(solution)):
            raise ValueError("Target endpoint energies are not finite.")
        output.append(
            {
                "chain_id": chain["chain_id"],
                "trajectory": _display_path(trajectory),
                "trajectory_sha256": window["trajectory_sha256"],
                "discarded_frame_count": discarded,
                "sample_count": len(samples),
                "positions": [
                    np.asarray(frame["positions_angstrom"], dtype=np.float64)
                    for frame in production
                ],
                "symbols": [frame["symbols"] for frame in production],
                "target_gas": gas,
                "target_solution": solution,
            }
        )
    if len(output) != expected_chain_count:
        raise ValueError("Target endpoint chain count changed.")
    return output


def _load_target_calculator(
    model: dict[str, Any],
    device,
    first_atoms,
    output: Path,
):
    from maple.function.calculator.set_calculator import SetCalculator

    checkpoint = (
        REPOSITORY_ROOT / "maple/function/calculator/model" / f"{model['name']}.pt"
    )
    if sha256_file(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash changed for {model['name']}.")
    calculator = SetCalculator(
        device,
        model["name"],
        str(output),
        atoms=first_atoms,
    ).set_calculator()
    return calculator, checkpoint


def _target_energies(
    atoms_template,
    calculator,
    correction,
    frames: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    gas: list[float] = []
    solution: list[float] = []
    started = time.perf_counter()
    for positions in frames:
        atoms = atoms_template.copy()
        atoms.set_positions(np.asarray(positions, dtype=np.float64))
        calculator.calculate(
            atoms,
            properties=["energy"],
            system_changes=all_changes,
        )
        gas_energy = float(calculator.results["energy"]) * KCAL_PER_HARTREE
        solvent = correction.evaluate(atoms, need_forces=False)
        solution_energy = gas_energy + float(solvent.energy_hartree) * KCAL_PER_HARTREE
        gas.append(gas_energy)
        solution.append(solution_energy)
    elapsed = time.perf_counter() - started
    return (
        _finite_array(gas, name="target gas energies"),
        _finite_array(solution, name="target solution energies"),
        elapsed,
    )


def _matrix(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    if reference.shape != target.shape:
        raise ValueError("Reduced-potential columns must have matching shapes.")
    matrix = np.column_stack((reference, target)) * beta
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Reduced-potential matrix is not finite.")
    return matrix


def _analysis_kwargs(protocol: dict[str, Any]) -> dict[str, Any]:
    analysis = protocol["analysis"]
    return {
        "temperature_kelvin": protocol["reference_sampling"]["temperature_kelvin"],
        "standard_state": analysis["standard_state"],
        "equilibrium_claim": protocol["execution_boundary"]["equilibrium_claim"],
        "detect_equilibration": analysis["detect_equilibration"],
        "equilibration_nskip": analysis["equilibration_nskip"],
        "minimum_uncorrelated_samples_per_state": (
            analysis["minimum_uncorrelated_samples_per_state"]
        ),
        "minimum_effective_samples_per_state": (
            analysis["minimum_effective_samples_per_state"]
        ),
        "minimum_bar_overlap": analysis["minimum_bar_overlap"],
        "minimum_directional_effective_fraction": (
            analysis["minimum_directional_effective_fraction"]
        ),
        "maximum_directional_disagreement_kcal_mol": (
            analysis["maximum_directional_disagreement_kcal_mol"]
        ),
        "maximum_bar_uncertainty_kcal_mol": (
            analysis["maximum_bar_uncertainty_kcal_mol"]
        ),
        "maximum_bar_mbar_disagreement_kcal_mol": (
            analysis["maximum_bar_mbar_disagreement_kcal_mol"]
        ),
        "maximum_iterations": analysis["maximum_iterations"],
        "relative_tolerance": analysis["relative_tolerance"],
    }


def indirect_cycle(
    reference_solvation_kcal_mol: float,
    solution_correction_kcal_mol: float,
    gas_correction_kcal_mol: float,
) -> float:
    """Return the exact reference-potential thermodynamic-cycle identity."""

    values = np.asarray(
        (
            reference_solvation_kcal_mol,
            solution_correction_kcal_mol,
            gas_correction_kcal_mol,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Indirect-cycle terms must be finite.")
    return float(values[0] + values[1] - values[2])


def _compact_reweighting(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "mbar_delta_g_kcal_mol": result["mbar"]["endpoint_delta_g_kcal_mol"],
        "mbar_uncertainty_kcal_mol": result["mbar"]["endpoint_uncertainty_kcal_mol"],
        "mbar_solver_diagnostics": result["mbar"]["solver_diagnostics"],
        "bar_delta_g_kcal_mol": result["bar"]["delta_g_kcal_mol"],
        "bar_uncertainty_kcal_mol": result["bar"]["uncertainty_kcal_mol"],
        "bar_solver_diagnostics": result["bar"]["solver_diagnostics"],
        "bar_overlap": result["bar"]["overlap"],
        "bar_mbar_difference_kcal_mol": result["bar"][
            "absolute_difference_from_mbar_kcal_mol"
        ],
        "forward_exp_delta_g_kcal_mol": result["directional"][
            "reference_to_target_exp"
        ]["delta_g_kcal_mol"],
        "forward_exp_uncertainty_kcal_mol": result["directional"][
            "reference_to_target_exp"
        ]["uncertainty_kcal_mol"],
        "forward_exp_effective_sample_fraction": result["directional"][
            "reference_to_target_exp"
        ]["effective_sample_fraction"],
        "reverse_exp_delta_g_kcal_mol": result["directional"][
            "target_to_reference_exp"
        ]["delta_g_kcal_mol"],
        "reverse_exp_effective_sample_fraction": result["directional"][
            "target_to_reference_exp"
        ]["effective_sample_fraction"],
        "forward_reverse_disagreement_kcal_mol": result["directional"][
            "absolute_forward_reverse_disagreement_kcal_mol"
        ],
        "selected_sample_counts": result["mbar"]["sample_counts"],
        "effective_sample_count_by_state": result["mbar"][
            "effective_sample_count_by_state"
        ],
        "minimum_directional_mbar_overlap": result["mbar"]["overlap"][
            "minimum_directional_adjacent_overlap"
        ],
        "checks": result["gates"]["checks"],
    }


def _analyze_model_case(
    *,
    model: dict[str, Any],
    case: dict[str, Any],
    atoms,
    calculator,
    correction,
    reference: ReferencePotential,
    reference_record: dict[str, Any],
    target_record: dict[str, Any],
    direct_mbar_row: dict[str, Any],
    protocol: dict[str, Any],
    protocol_fingerprint: str,
) -> dict[str, Any]:
    target_source = protocol["target_sample_source"]
    target_endpoints = {
        "gas": _target_endpoint_frames(
            target_record,
            coupling=0.0,
            expected_chain_count=target_source["required_independent_chains"],
            expected_sample_count=target_source[
                "expected_production_samples_per_endpoint_chain"
            ],
        ),
        "solution": _target_endpoint_frames(
            target_record,
            coupling=1.0,
            expected_chain_count=target_source["required_independent_chains"],
            expected_sample_count=target_source[
                "expected_production_samples_per_endpoint_chain"
            ],
        ),
    }
    reference_endpoints = _load_reference_frames(reference_record)
    if any(
        symbols != atoms.get_chemical_symbols()
        for endpoint in reference_endpoints.values()
        for repeat in endpoint
        for symbols in repeat["symbols"]
    ):
        raise ValueError("Reference trajectory atom order differs from the target.")
    if any(
        symbols != atoms.get_chemical_symbols()
        for endpoint in target_endpoints.values()
        for repeat in endpoint
        for symbols in repeat["symbols"]
    ):
        raise ValueError("Target trajectory atom order differs from the target MOL2.")

    target_on_reference: dict[str, list[dict[str, Any]]] = {
        "gas": [],
        "solution": [],
    }
    target_energy_seconds = 0.0
    target_energy_calls = 0
    for phase in ("gas", "solution"):
        for repeat in reference_endpoints[phase]:
            gas, solution, elapsed = _target_energies(
                atoms,
                calculator,
                correction,
                repeat["positions"],
            )
            target_energy_seconds += elapsed
            target_energy_calls += len(gas)
            target_on_reference[phase].append(
                {
                    "replicate_id": repeat["replicate_id"],
                    "target_gas": gas,
                    "target_solution": solution,
                }
            )

    reference_on_target: dict[str, list[dict[str, Any]]] = {
        "gas": [],
        "solution": [],
    }
    reference_energy_started = time.perf_counter()
    for phase in ("gas", "solution"):
        for repeat in target_endpoints[phase]:
            gas, solution = reference.energies(repeat["positions"])
            reference_on_target[phase].append(
                {
                    "chain_id": repeat["chain_id"],
                    "reference_gas": gas,
                    "reference_solution": solution,
                }
            )
    reference_target_frame_seconds = time.perf_counter() - reference_energy_started

    temperature = protocol["reference_sampling"]["temperature_kelvin"]
    beta = 1.0 / (R_KCAL_PER_MOL_K * temperature)
    ref_gas_matrices = [
        _matrix(
            repeat["reference_gas"],
            repeat["reference_solution"],
            beta=beta,
        )
        for repeat in reference_endpoints["gas"]
    ]
    ref_solution_matrices = [
        _matrix(
            repeat["reference_gas"],
            repeat["reference_solution"],
            beta=beta,
        )
        for repeat in reference_endpoints["solution"]
    ]
    gas_correction_reference = [
        _matrix(
            reference_repeat["reference_gas"],
            target_repeat["target_gas"],
            beta=beta,
        )
        for reference_repeat, target_repeat in zip(
            reference_endpoints["gas"],
            target_on_reference["gas"],
            strict=True,
        )
    ]
    solution_correction_reference = [
        _matrix(
            reference_repeat["reference_solution"],
            target_repeat["target_solution"],
            beta=beta,
        )
        for reference_repeat, target_repeat in zip(
            reference_endpoints["solution"],
            target_on_reference["solution"],
            strict=True,
        )
    ]
    gas_correction_target = [
        _matrix(
            reference_repeat["reference_gas"],
            target_repeat["target_gas"],
            beta=beta,
        )
        for reference_repeat, target_repeat in zip(
            reference_on_target["gas"],
            target_endpoints["gas"],
            strict=True,
        )
    ]
    solution_correction_target = [
        _matrix(
            reference_repeat["reference_solution"],
            target_repeat["target_solution"],
            beta=beta,
        )
        for reference_repeat, target_repeat in zip(
            reference_on_target["solution"],
            target_endpoints["solution"],
            strict=True,
        )
    ]
    direct_target_gas = [
        _matrix(
            repeat["target_gas"],
            repeat["target_solution"],
            beta=beta,
        )
        for repeat in target_endpoints["gas"]
    ]
    direct_target_solution = [
        _matrix(
            repeat["target_gas"],
            repeat["target_solution"],
            beta=beta,
        )
        for repeat in target_endpoints["solution"]
    ]

    kwargs = _analysis_kwargs(protocol)
    reference_solvation = analyze_bidirectional_reweighting(
        [ref_gas_matrices, ref_solution_matrices],
        **kwargs,
    )
    gas_correction = analyze_bidirectional_reweighting(
        [gas_correction_reference, gas_correction_target],
        **kwargs,
    )
    solution_correction = analyze_bidirectional_reweighting(
        [solution_correction_reference, solution_correction_target],
        **kwargs,
    )
    direct_endpoint = analyze_bidirectional_reweighting(
        [direct_target_gas, direct_target_solution],
        **kwargs,
    )

    bidirectional_cycle = indirect_cycle(
        reference_solvation["mbar"]["endpoint_delta_g_kcal_mol"],
        solution_correction["mbar"]["endpoint_delta_g_kcal_mol"],
        gas_correction["mbar"]["endpoint_delta_g_kcal_mol"],
    )
    accelerated_cycle = indirect_cycle(
        reference_solvation["mbar"]["endpoint_delta_g_kcal_mol"],
        solution_correction["directional"]["reference_to_target_exp"][
            "delta_g_kcal_mol"
        ],
        gas_correction["directional"]["reference_to_target_exp"]["delta_g_kcal_mol"],
    )
    direct_mbar = float(direct_mbar_row["mbar"]["endpoint_delta_g_kcal_mol"])

    provider_differences: list[np.ndarray] = []
    for phase in ("gas", "solution"):
        for reference_repeat, target_repeat in zip(
            reference_endpoints[phase],
            target_on_reference[phase],
            strict=True,
        ):
            provider_differences.append(
                (target_repeat["target_solution"] - target_repeat["target_gas"])
                - (
                    reference_repeat["reference_solution"]
                    - reference_repeat["reference_gas"]
                )
            )
        for reference_repeat, target_repeat in zip(
            reference_on_target[phase],
            target_endpoints[phase],
            strict=True,
        ):
            provider_differences.append(
                (target_repeat["target_solution"] - target_repeat["target_gas"])
                - (
                    reference_repeat["reference_solution"]
                    - reference_repeat["reference_gas"]
                )
            )
    provider_difference = np.concatenate(provider_differences)
    maximum_provider_difference = float(np.max(np.abs(provider_difference)))

    analysis = protocol["analysis"]
    accelerated_difference = abs(accelerated_cycle - bidirectional_cycle)
    direct_difference = abs(bidirectional_cycle - direct_mbar)
    numerical_checks = {
        "finite_cycle_outputs": bool(
            np.all(
                np.isfinite(
                    (
                        bidirectional_cycle,
                        accelerated_cycle,
                        direct_mbar,
                    )
                )
            )
        ),
        "reference_solvation_uncorrelated_samples": reference_solvation["gates"][
            "checks"
        ]["minimum_uncorrelated_samples_per_state"],
        "gas_correction_uncorrelated_samples": gas_correction["gates"]["checks"][
            "minimum_uncorrelated_samples_per_state"
        ],
        "solution_correction_uncorrelated_samples": solution_correction["gates"][
            "checks"
        ]["minimum_uncorrelated_samples_per_state"],
        "direct_endpoint_uncorrelated_samples": direct_endpoint["gates"]["checks"][
            "minimum_uncorrelated_samples_per_state"
        ],
        "reference_solvation_effective_samples": reference_solvation["gates"]["checks"][
            "minimum_effective_samples_per_state"
        ],
        "gas_correction_effective_samples": gas_correction["gates"]["checks"][
            "minimum_effective_samples_per_state"
        ],
        "solution_correction_effective_samples": solution_correction["gates"]["checks"][
            "minimum_effective_samples_per_state"
        ],
        "direct_endpoint_effective_samples": direct_endpoint["gates"]["checks"][
            "minimum_effective_samples_per_state"
        ],
        "reference_solvation_solver_convergence": reference_solvation["gates"][
            "checks"
        ]["mbar_solver_convergence"],
        "gas_correction_solver_convergence": gas_correction["gates"]["checks"][
            "mbar_solver_convergence"
        ],
        "solution_correction_solver_convergence": solution_correction["gates"][
            "checks"
        ]["mbar_solver_convergence"],
        "direct_endpoint_solver_convergence": direct_endpoint["gates"]["checks"][
            "mbar_solver_convergence"
        ],
        "reference_solvation_bar_solver_convergence": reference_solvation["gates"][
            "checks"
        ]["bar_solver_convergence"],
        "gas_correction_bar_solver_convergence": gas_correction["gates"]["checks"][
            "bar_solver_convergence"
        ],
        "solution_correction_bar_solver_convergence": solution_correction["gates"][
            "checks"
        ]["bar_solver_convergence"],
        "direct_endpoint_bar_solver_convergence": direct_endpoint["gates"]["checks"][
            "bar_solver_convergence"
        ],
        "reference_solvation_mbar_directional_overlap": reference_solvation["gates"][
            "checks"
        ]["minimum_mbar_directional_overlap"],
        "gas_correction_mbar_directional_overlap": gas_correction["gates"]["checks"][
            "minimum_mbar_directional_overlap"
        ],
        "solution_correction_mbar_directional_overlap": solution_correction["gates"][
            "checks"
        ]["minimum_mbar_directional_overlap"],
        "direct_endpoint_mbar_directional_overlap": direct_endpoint["gates"]["checks"][
            "minimum_mbar_directional_overlap"
        ],
        "reference_solvation_overlap": reference_solvation["gates"]["checks"][
            "minimum_bar_overlap"
        ],
        "gas_correction_overlap": gas_correction["gates"]["checks"][
            "minimum_bar_overlap"
        ],
        "solution_correction_overlap": solution_correction["gates"]["checks"][
            "minimum_bar_overlap"
        ],
        "direct_endpoint_overlap": direct_endpoint["gates"]["checks"][
            "minimum_bar_overlap"
        ],
        "reference_solvation_bar_uncertainty": reference_solvation["gates"]["checks"][
            "maximum_bar_uncertainty"
        ],
        "gas_correction_bar_uncertainty": gas_correction["gates"]["checks"][
            "maximum_bar_uncertainty"
        ],
        "solution_correction_bar_uncertainty": solution_correction["gates"]["checks"][
            "maximum_bar_uncertainty"
        ],
        "direct_endpoint_bar_uncertainty": direct_endpoint["gates"]["checks"][
            "maximum_bar_uncertainty"
        ],
        "gas_directional_effective_fraction": gas_correction["gates"]["checks"][
            "minimum_directional_effective_fraction"
        ],
        "solution_directional_effective_fraction": solution_correction["gates"][
            "checks"
        ]["minimum_directional_effective_fraction"],
        "gas_directional_agreement": gas_correction["gates"]["checks"][
            "maximum_directional_disagreement"
        ],
        "solution_directional_agreement": solution_correction["gates"]["checks"][
            "maximum_directional_disagreement"
        ],
        "reference_solvation_bar_mbar_agreement": reference_solvation["gates"][
            "checks"
        ]["maximum_bar_mbar_disagreement"],
        "gas_correction_bar_mbar_agreement": gas_correction["gates"]["checks"][
            "maximum_bar_mbar_disagreement"
        ],
        "solution_correction_bar_mbar_agreement": solution_correction["gates"][
            "checks"
        ]["maximum_bar_mbar_disagreement"],
        "direct_endpoint_bar_mbar_agreement": direct_endpoint["gates"]["checks"][
            "maximum_bar_mbar_disagreement"
        ],
        "accelerated_vs_bidirectional_cycle": (
            accelerated_difference
            <= analysis[
                "maximum_accelerated_vs_bidirectional_cycle_difference_kcal_mol"
            ]
        ),
        "bidirectional_cycle_vs_direct_mbar": (
            direct_difference
            <= analysis[
                "maximum_bidirectional_cycle_vs_direct_mbar_difference_kcal_mol"
            ]
        ),
        "reference_target_solvent_provider_parity": (
            maximum_provider_difference
            <= analysis["maximum_reference_target_solvent_provider_difference_kcal_mol"]
        ),
    }
    endpoint_corrections = (
        gas_correction["mbar"]["endpoint_delta_g_kcal_mol"],
        solution_correction["mbar"]["endpoint_delta_g_kcal_mol"],
    )
    record = {
        "schema_version": 1,
        "artifact_type": "route1-mmgb-reference-mlip-reweighting-case",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": protocol_fingerprint,
        "model": model["name"],
        "compound_id": case["compound_id"],
        "name": case["name"],
        "flexibility_bin": case["flexibility_bin"],
        "thermodynamic_cycle": {
            "reference_solvation_mbar_kcal_mol": reference_solvation["mbar"][
                "endpoint_delta_g_kcal_mol"
            ],
            "gas_reference_to_target_mbar_kcal_mol": endpoint_corrections[0],
            "solution_reference_to_target_mbar_kcal_mol": endpoint_corrections[1],
            "net_reference_to_target_correction_kcal_mol": (
                endpoint_corrections[1] - endpoint_corrections[0]
            ),
            "maximum_absolute_endpoint_reference_to_target_correction_kcal_mol": (
                max(abs(value) for value in endpoint_corrections)
            ),
            "bidirectional_mbar_delta_g_kcal_mol": bidirectional_cycle,
            "reference_only_accelerated_delta_g_kcal_mol": accelerated_cycle,
            "accelerated_minus_bidirectional_kcal_mol": (
                accelerated_cycle - bidirectional_cycle
            ),
            "direct_target_endpoint_mbar_kcal_mol": direct_endpoint["mbar"][
                "endpoint_delta_g_kcal_mol"
            ],
            "direct_target_multistate_mbar_kcal_mol": direct_mbar,
            "bidirectional_minus_direct_mbar_kcal_mol": (
                bidirectional_cycle - direct_mbar
            ),
        },
        "reweighting_diagnostics": {
            "reference_solvation": _compact_reweighting(reference_solvation),
            "gas_reference_to_target": _compact_reweighting(gas_correction),
            "solution_reference_to_target": _compact_reweighting(solution_correction),
            "direct_target_endpoint": _compact_reweighting(direct_endpoint),
        },
        "provider_parity": {
            "maximum_absolute_reference_minus_target_solvent_term_kcal_mol": (
                maximum_provider_difference
            ),
            "root_mean_square_difference_kcal_mol": float(
                np.sqrt(np.mean(np.square(provider_difference)))
            ),
            "sample_count": int(len(provider_difference)),
        },
        "operation_counts": {
            "reference_force_sampling_steps": (
                len(protocol["reference_sampling"]["replicates"])
                * 2
                * (
                    protocol["reference_sampling"]["equilibration_steps_per_endpoint"]
                    + protocol["reference_sampling"]["production_steps_per_endpoint"]
                )
            ),
            "target_energy_only_evaluations_for_accelerated_candidate": (
                target_energy_calls
            ),
            "target_force_evaluations_for_accelerated_candidate": 0,
            "target_validation_frames_reused_without_new_target_calls": sum(
                len(repeat["target_gas"])
                for endpoint in target_endpoints.values()
                for repeat in endpoint
            ),
        },
        "runtime_observations": {
            "target_energy_only_seconds": target_energy_seconds,
            "reference_energy_on_target_frames_seconds": (
                reference_target_frame_seconds
            ),
            "hardware_specific_and_not_a_universal_speed_claim": True,
        },
        "numerical_checks": numerical_checks,
        "all_numerical_checks_pass": all(numerical_checks.values()),
        "label_boundary": {
            "experimental_labels_read": False,
            "experimental_scoring_performed": False,
        },
        "claim_boundary": {
            "equilibrium_proven": False,
            "conformer_mixing_proven": False,
            "reference_only_production_validated": False,
            "chemical_accuracy_established": False,
            "product_promotion_allowed": False,
        },
        "energy_samples": {
            "reference_ensemble": {
                phase: [
                    {
                        "replicate_id": reference_repeat["replicate_id"],
                        "reference_gas_kcal_mol": (
                            reference_repeat["reference_gas"].tolist()
                        ),
                        "reference_solution_kcal_mol": (
                            reference_repeat["reference_solution"].tolist()
                        ),
                        "target_gas_kcal_mol": target_repeat["target_gas"].tolist(),
                        "target_solution_kcal_mol": target_repeat[
                            "target_solution"
                        ].tolist(),
                    }
                    for reference_repeat, target_repeat in zip(
                        reference_endpoints[phase],
                        target_on_reference[phase],
                        strict=True,
                    )
                ]
                for phase in ("gas", "solution")
            },
            "target_validation_ensemble": {
                phase: [
                    {
                        "chain_id": target_repeat["chain_id"],
                        "reference_gas_kcal_mol": reference_repeat[
                            "reference_gas"
                        ].tolist(),
                        "reference_solution_kcal_mol": reference_repeat[
                            "reference_solution"
                        ].tolist(),
                        "target_gas_kcal_mol": target_repeat["target_gas"].tolist(),
                        "target_solution_kcal_mol": target_repeat[
                            "target_solution"
                        ].tolist(),
                    }
                    for reference_repeat, target_repeat in zip(
                        reference_on_target[phase],
                        target_endpoints[phase],
                        strict=True,
                    )
                ]
                for phase in ("gas", "solution")
            },
        },
    }
    return seal_artifact(record)


def _load_reusable_record(
    path: Path,
    *,
    artifact_type: str,
    protocol_fingerprint: str,
    identity: tuple[str, ...],
) -> dict[str, Any]:
    record = load_json(path)
    expected = record.get("content_sha256")
    if expected != artifact_content_sha256(record):
        raise ValueError(f"Reusable record has an invalid self-hash: {path}.")
    observed_identity = tuple(
        str(record[key])
        for key in (
            ("model", "compound_id")
            if artifact_type.endswith("-case")
            else ("compound_id",)
        )
    )
    if (
        record.get("artifact_type") != artifact_type
        or record.get("protocol_fingerprint") != protocol_fingerprint
        or observed_identity != identity
    ):
        raise ValueError(f"Reusable record identity changed: {path}.")
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
    source_rows = _source_record_rows(protocol_path, protocol)
    direct_rows = _direct_mbar_rows(protocol_path, protocol)
    charge_rows = _charge_rows(protocol_path, protocol)

    reference_records: dict[str, dict[str, Any]] = {}
    reference_record_paths: dict[str, Path] = {}
    for case in selected_cases:
        case_dir = reference_input_dir / case["compound_id"]
        _validate_reference_inputs(case_dir, case)
        published_inputs = _copy_reference_inputs(case_dir, raw_dir, case)
        record_path = raw_dir / "reference" / f"{case['compound_id']}.json"
        if args.resume_existing and record_path.is_file():
            record = _load_reusable_record(
                record_path,
                artifact_type="route1-mmgb-reference-samples",
                protocol_fingerprint=fingerprint,
                identity=(case["compound_id"],),
            )
        else:
            record = _reference_record(
                case=case,
                case_dir=case_dir,
                raw_dir=raw_dir,
                protocol=protocol,
                protocol_fingerprint=fingerprint,
                published_inputs=published_inputs,
            )
            write_json_atomic(record_path, record)
        reference_records[case["compound_id"]] = record
        reference_record_paths[case["compound_id"]] = record_path

    device_text = str(args.device).strip().lower()
    if device_text.startswith("gpu"):
        device_text = f"cuda:{device_text[3:] or '0'}"
    device = torch.device(device_text)
    model_records: list[dict[str, Any]] = []
    model_record_paths: dict[tuple[str, str], Path] = {}
    checkpoint_records: dict[str, dict[str, str]] = {}
    for model in selected_models:
        first_case = selected_cases[0]
        first_case_dir = reference_input_dir / first_case["compound_id"]
        first_atoms = _load_case_atoms(
            first_case_dir,
            charge_rows[first_case["compound_id"]],
        )
        calculator, checkpoint = _load_target_calculator(
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
            record_path = (
                raw_dir / "models" / model["name"] / f"{case['compound_id']}.json"
            )
            if args.resume_existing and record_path.is_file():
                record = _load_reusable_record(
                    record_path,
                    artifact_type="route1-mmgb-reference-mlip-reweighting-case",
                    protocol_fingerprint=fingerprint,
                    identity=(model["name"], case["compound_id"]),
                )
            else:
                case_dir = reference_input_dir / case["compound_id"]
                atoms = _load_case_atoms(
                    case_dir,
                    charge_rows[case["compound_id"]],
                )
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
                        / "models"
                        / model["name"]
                        / f"{case['compound_id']}-correction.out"
                    ),
                )
                reference = ReferencePotential(case_dir, protocol)
                source_identity = (model["name"], case["compound_id"])
                record = _analyze_model_case(
                    model=model,
                    case=case,
                    atoms=atoms,
                    calculator=calculator,
                    correction=correction,
                    reference=reference,
                    reference_record=reference_records[case["compound_id"]],
                    target_record=_load_target_record(source_rows[source_identity]),
                    direct_mbar_row=direct_rows[source_identity],
                    protocol=protocol,
                    protocol_fingerprint=fingerprint,
                )
                record_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(record_path, record)
            model_records.append(record)
            model_record_paths[(model["name"], case["compound_id"])] = record_path
        del calculator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = seal_artifact(
        {
            "schema_version": 1,
            "manifest_id": "maple-route1-mmgb-reference-mlip-reweighting-raw-v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_fingerprint": fingerprint,
            "label_boundary": {
                "experimental_labels_read": False,
            },
            "reference_records": [
                {
                    "compound_id": compound_id,
                    "relative_path": _display_path(path),
                    "file_sha256": sha256_file(path),
                    "content_sha256": reference_records[compound_id]["content_sha256"],
                }
                for compound_id, path in sorted(reference_record_paths.items())
            ],
            "model_records": [
                {
                    "model": model,
                    "compound_id": compound_id,
                    "relative_path": _display_path(path),
                    "file_sha256": sha256_file(path),
                    "content_sha256": next(
                        record["content_sha256"]
                        for record in model_records
                        if record["model"] == model
                        and record["compound_id"] == compound_id
                    ),
                }
                for (model, compound_id), path in sorted(model_record_paths.items())
            ],
        }
    )
    manifest_path = raw_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    compact_records = [
        {
            "model": record["model"],
            "compound_id": record["compound_id"],
            "name": record["name"],
            "flexibility_bin": record["flexibility_bin"],
            "reference_solvation_mbar_kcal_mol": record["thermodynamic_cycle"][
                "reference_solvation_mbar_kcal_mol"
            ],
            "bidirectional_mbar_delta_g_kcal_mol": record["thermodynamic_cycle"][
                "bidirectional_mbar_delta_g_kcal_mol"
            ],
            "reference_only_accelerated_delta_g_kcal_mol": record[
                "thermodynamic_cycle"
            ]["reference_only_accelerated_delta_g_kcal_mol"],
            "accelerated_minus_bidirectional_kcal_mol": record["thermodynamic_cycle"][
                "accelerated_minus_bidirectional_kcal_mol"
            ],
            "direct_target_multistate_mbar_kcal_mol": record["thermodynamic_cycle"][
                "direct_target_multistate_mbar_kcal_mol"
            ],
            "bidirectional_minus_direct_mbar_kcal_mol": record["thermodynamic_cycle"][
                "bidirectional_minus_direct_mbar_kcal_mol"
            ],
            "gas_bar_overlap": record["reweighting_diagnostics"][
                "gas_reference_to_target"
            ]["bar_overlap"],
            "solution_bar_overlap": record["reweighting_diagnostics"][
                "solution_reference_to_target"
            ]["bar_overlap"],
            "gas_mbar_directional_overlap": record["reweighting_diagnostics"][
                "gas_reference_to_target"
            ]["minimum_directional_mbar_overlap"],
            "solution_mbar_directional_overlap": record["reweighting_diagnostics"][
                "solution_reference_to_target"
            ]["minimum_directional_mbar_overlap"],
            "solver_convergence_checks_pass": all(
                record["numerical_checks"][name]
                for name in (
                    "reference_solvation_solver_convergence",
                    "gas_correction_solver_convergence",
                    "solution_correction_solver_convergence",
                    "direct_endpoint_solver_convergence",
                    "reference_solvation_bar_solver_convergence",
                    "gas_correction_bar_solver_convergence",
                    "solution_correction_bar_solver_convergence",
                    "direct_endpoint_bar_solver_convergence",
                )
            ),
            "gas_forward_ess_fraction": record["reweighting_diagnostics"][
                "gas_reference_to_target"
            ]["forward_exp_effective_sample_fraction"],
            "solution_forward_ess_fraction": record["reweighting_diagnostics"][
                "solution_reference_to_target"
            ]["forward_exp_effective_sample_fraction"],
            "maximum_provider_difference_kcal_mol": record["provider_parity"][
                "maximum_absolute_reference_minus_target_solvent_term_kcal_mol"
            ],
            "target_energy_only_evaluations": record["operation_counts"][
                "target_energy_only_evaluations_for_accelerated_candidate"
            ],
            "target_energy_only_seconds": record["runtime_observations"][
                "target_energy_only_seconds"
            ],
            "numerical_checks": record["numerical_checks"],
            "all_numerical_checks_pass": record["all_numerical_checks_pass"],
            "raw_record": _display_path(
                model_record_paths[(record["model"], record["compound_id"])]
            ),
            "raw_record_sha256": sha256_file(
                model_record_paths[(record["model"], record["compound_id"])]
            ),
        }
        for record in model_records
    ]
    accelerated_differences = [
        abs(row["accelerated_minus_bidirectional_kcal_mol"]) for row in compact_records
    ]
    direct_differences = [
        abs(row["bidirectional_minus_direct_mbar_kcal_mol"]) for row in compact_records
    ]
    output = {
        "schema_version": 1,
        "artifact_type": "route1-mmgb-reference-mlip-reweighting-summary",
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "record_count": len(compact_records),
        "case_count": len(selected_cases),
        "model_count": len(selected_models),
        "records": compact_records,
        "aggregate_diagnostics": {
            "numerical_pass_count": sum(
                row["all_numerical_checks_pass"] for row in compact_records
            ),
            "minimum_gas_bar_overlap": float(
                min(row["gas_bar_overlap"] for row in compact_records)
            ),
            "minimum_solution_bar_overlap": float(
                min(row["solution_bar_overlap"] for row in compact_records)
            ),
            "minimum_gas_mbar_directional_overlap": float(
                min(row["gas_mbar_directional_overlap"] for row in compact_records)
            ),
            "minimum_solution_mbar_directional_overlap": float(
                min(row["solution_mbar_directional_overlap"] for row in compact_records)
            ),
            "solver_convergence_pass_count": sum(
                row["solver_convergence_checks_pass"] for row in compact_records
            ),
            "minimum_gas_forward_ess_fraction": float(
                min(row["gas_forward_ess_fraction"] for row in compact_records)
            ),
            "minimum_solution_forward_ess_fraction": float(
                min(row["solution_forward_ess_fraction"] for row in compact_records)
            ),
            "maximum_accelerated_vs_bidirectional_difference_kcal_mol": float(
                max(accelerated_differences)
            ),
            "maximum_bidirectional_vs_direct_mbar_difference_kcal_mol": float(
                max(direct_differences)
            ),
            "maximum_provider_parity_difference_kcal_mol": float(
                max(
                    row["maximum_provider_difference_kcal_mol"]
                    for row in compact_records
                )
            ),
            "total_sparse_target_energy_evaluations": sum(
                row["target_energy_only_evaluations"] for row in compact_records
            ),
            "total_target_energy_only_seconds": float(
                sum(row["target_energy_only_seconds"] for row in compact_records)
            ),
        },
        "raw_manifest": {
            "path": _display_path(manifest_path),
            "file_sha256": sha256_file(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "checkpoint_records": checkpoint_records,
        "decision": {
            "result": "diagnostic_only_not_promotable",
            "promotion_allowed": False,
            "reference_only_production_validated": False,
            "reason": protocol["decision_policy"]["reason"],
        },
        "label_boundary": {
            "experimental_labels_read": False,
            "experimental_scoring_performed": False,
        },
        "performance_claim_boundary": protocol["performance_claim_boundary"],
        "claim_boundary": {
            "target_formula_preserved": True,
            "reference_mm_energy_in_target": False,
            "target_force_free_accelerated_candidate_exercised": True,
            "equilibrium_proven": False,
            "chemical_accuracy_established": False,
            "product_promotion_allowed": False,
        },
        "implementation_provenance": {
            "runner": _display_path(Path(__file__)),
            "runner_sha256": sha256_file(__file__),
            "analysis_module": _display_path(
                Path(sys.modules[analyze_bidirectional_reweighting.__module__].__file__)
            ),
            "analysis_module_sha256": sha256_file(
                sys.modules[analyze_bidirectional_reweighting.__module__].__file__
            ),
            "mbar_analysis_module": _display_path(
                Path(sys.modules[analyze_mbar.__module__].__file__)
            ),
            "mbar_analysis_module_sha256": sha256_file(
                sys.modules[analyze_mbar.__module__].__file__
            ),
            "pymbar_version": importlib.metadata.version("pymbar"),
            "openmm_version": importlib.metadata.version("openmm"),
        },
    }
    return seal_artifact(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(BENCHMARK_DIR / "mlip_mm_reference_reweighting_protocol.json"),
    )
    parser.add_argument(
        "--reference-input-dir",
        default=str(
            REPOSITORY_ROOT
            / ".omx/benchmarks/mlip-chagb-metropolis-route1-20260724/cases"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=str(BENCHMARK_DIR / "route1-mm-reference-reweighting-raw"),
    )
    parser.add_argument(
        "--output",
        default=str(BENCHMARK_DIR / "route1-mm-reference-reweighting-2026-07-25.json"),
    )
    parser.add_argument("--device", default="gpu0")
    parser.add_argument("--model", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    write_json_atomic(args.output, result)
    print(json.dumps(result["aggregate_diagnostics"], sort_keys=True))
    print(result["decision"]["result"])


if __name__ == "__main__":
    main()
