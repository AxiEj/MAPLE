#!/usr/bin/env python3
"""Qualify one global optimizer policy for the Route 1 qRRHO successor.

This engineering experiment is label blind. It reuses the frozen v8 source
states, phase-specific seed rule, MLIP checkpoints, AM1-BCC charges, and
OBC-II/ACE solution force. It compares the preregistered optimizer candidates
on the same flexible-case branches and chooses at most one policy by a frozen
model-neutral rule. It never evaluates hydration accuracy.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any

from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.optimize import BFGSLineSearch, FIRE2, LBFGSLineSearch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
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
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402

DEFAULT_PROTOCOL = SCRIPT_DIR / "multi_mlip_optimizer_qualification_protocol.json"
DEFAULT_QRRHO_RUNNER = SCRIPT_DIR / "run_multi_mlip_phase_specific_qrrho.py"
FORBIDDEN_LABEL_KEYS = {
    "experimental_kcal_mol",
    "experimental_reference",
    "experimental_uncertainty_kcal_mol",
}


def _load_qrrho_module():
    specification = importlib.util.spec_from_file_location(
        "route1_qrrho_runner",
        DEFAULT_QRRHO_RUNNER,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the frozen Route 1 qRRHO runner.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


QRRHO = _load_qrrho_module()


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_LABEL_KEYS or _contains_forbidden_label_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Evidence path escapes the repository: {path}.") from exc


def _resolve_repository_file(path: str, *, name: str) -> Path:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError(f"{name} must be a non-empty POSIX repository path.")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be repository relative.")
    resolved = (REPOSITORY_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} escapes through a symlink.") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} is missing: {path}.")
    return resolved


def _validate_self_hash(value: dict[str, Any], *, name: str) -> None:
    if value.get("content_sha256") != artifact_content_sha256(value):
        raise ValueError(f"{name} has an invalid content SHA256.")


def load_protocol(
    path: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Validate the optimizer protocol and every frozen upstream dependency."""

    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if (
        not isinstance(protocol, dict)
        or protocol.get("schema_version") != 1
        or protocol.get("protocol_id")
        != "maple-route1-multi-mlip-optimizer-qualification-v2"
        or protocol.get("status") != "preregistered-before-qualification-execution"
    ):
        raise ValueError("Unsupported optimizer-qualification protocol.")
    if _contains_forbidden_label_key(protocol):
        raise ValueError("Optimizer qualification crosses the label boundary.")

    label_boundary = protocol.get("label_boundary", {})
    if (
        label_boundary.get("experimental_labels_read") is not False
        or label_boundary.get("hydration_accuracy_evaluated") is not False
        or label_boundary.get("selection_uses_only_optimizer_observables") is not True
    ):
        raise ValueError("Optimizer qualification is not label blind.")
    route = protocol.get("route_boundary", {})
    if (
        route.get("gas_phase_mm_energy") is not False
        or route.get("hydration_label_residual") is not False
        or route.get("mlip_retraining") is not False
        or route.get("charge_refitting") is not False
    ):
        raise ValueError("Optimizer qualification violates Route 1.")

    upstream = protocol["frozen_upstream"]
    qrrho_protocol_path = _resolve_repository_file(
        upstream["qrrho_protocol"]["path"],
        name="Frozen qRRHO protocol",
    )
    if sha256_file(qrrho_protocol_path) != upstream["qrrho_protocol"]["file_sha256"]:
        raise ValueError("Frozen qRRHO protocol bytes changed.")
    qrrho_protocol, qrrho_fingerprint, source_manifest = QRRHO.load_qrrho_protocol(
        qrrho_protocol_path
    )
    if (
        qrrho_fingerprint != upstream["qrrho_protocol"]["protocol_fingerprint"]
        or qrrho_protocol["protocol_id"]
        != upstream["qrrho_protocol"]["protocol_id"]
    ):
        raise ValueError("Frozen qRRHO protocol identity changed.")

    failure_path = _resolve_repository_file(
        upstream["v8_failure_audit"]["path"],
        name="v8 failure audit",
    )
    if sha256_file(failure_path) != upstream["v8_failure_audit"]["file_sha256"]:
        raise ValueError("v8 failure-audit bytes changed.")
    failure = load_json(failure_path)
    _validate_self_hash(failure, name="v8 failure audit")
    if (
        failure.get("content_sha256")
        != upstream["v8_failure_audit"]["content_sha256"]
        or failure.get("status") != "failed-closed-before-seal-or-label-scoring"
        or failure.get("label_boundary", {}).get("experimental_labels_read")
        is not False
    ):
        raise ValueError("v8 failure audit is not the frozen label-blind evidence.")

    interruption_path = _resolve_repository_file(
        upstream["v1_interruption_audit"]["path"],
        name="v1 interruption audit",
    )
    if (
        sha256_file(interruption_path)
        != upstream["v1_interruption_audit"]["file_sha256"]
    ):
        raise ValueError("v1 interruption-audit bytes changed.")
    interruption = load_json(interruption_path)
    _validate_self_hash(interruption, name="v1 interruption audit")
    if (
        interruption.get("content_sha256")
        != upstream["v1_interruption_audit"]["content_sha256"]
        or interruption.get("status")
        != "interrupted-before-complete-matrix-or-selection"
        or interruption.get("decision", {}).get("v1_resume_allowed") is not False
        or interruption.get("amendment_basis", {}).get("experimental_labels_read")
        is not False
    ):
        raise ValueError("v1 interruption audit is not the frozen amendment evidence.")

    implementation = protocol["implementation_freeze"]
    if platform.python_version() != implementation["required_python"]:
        raise ValueError("Required Python version changed.")
    for distribution, expected_version in implementation["required_versions"].items():
        if importlib.metadata.version(distribution) != expected_version:
            raise ValueError(f"Required {distribution} version changed.")
    for source in implementation["source_files"]:
        source_path = _resolve_repository_file(
            source["path"],
            name="Frozen qualification source",
        )
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Frozen source changed: {source['path']}.")

    candidates = protocol.get("optimizer_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("At least two optimizer candidates are required.")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if (
        len(set(candidate_ids)) != len(candidate_ids)
        or any(not isinstance(value, str) or not value for value in candidate_ids)
    ):
        raise ValueError("Optimizer candidate identifiers are invalid.")
    supported = {"ASE-BFGSLineSearch", "ASE-LBFGSLineSearch", "ASE-FIRE2-ABC"}
    if any(candidate.get("algorithm") not in supported for candidate in candidates):
        raise ValueError("The optimizer matrix contains an unsupported algorithm.")

    scope = protocol["qualification_scope"]
    case_ids = scope["case_ids"]
    model_ids = scope["model_ids"]
    if case_ids != ["mobley_4463913"]:
        raise ValueError("The frozen flexible qualification case changed.")
    if model_ids != [model["name"] for model in qrrho_protocol["models"]]:
        raise ValueError("The frozen multi-MLIP qualification roster changed.")
    if scope["phases"] != ["gas", "solution"]:
        raise ValueError("Both Route 1 phases must be qualified.")
    expected_branch_count = (
        len(model_ids)
        * len(scope["phases"])
        * int(qrrho_protocol["phase_specific_sampling"][
            "maximum_seed_count_per_phase_model_case"
        ])
    )
    if int(scope["expected_branch_count_per_candidate"]) != expected_branch_count:
        raise ValueError("Expected optimizer branch count is inconsistent.")

    fingerprint = sha256_bytes(canonical_json_bytes(protocol))
    return protocol, fingerprint, qrrho_protocol, source_manifest


class CountingHartreeToEVCalculator(QRRHO.HartreeToEVCalculator):
    """Count ASE-level calculator evaluations without changing energies or forces."""

    def __init__(self, source_calculator, *, maximum_calculate_calls: int):
        super().__init__(source_calculator)
        self.calculate_calls = 0
        self.maximum_calculate_calls = int(maximum_calculate_calls)
        if self.maximum_calculate_calls <= 0:
            raise ValueError("Calculator-evaluation budget must be positive.")

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if self.calculate_calls >= self.maximum_calculate_calls:
            raise RuntimeError(
                "Frozen per-branch calculator-evaluation budget exhausted: "
                f"{self.maximum_calculate_calls}."
            )
        self.calculate_calls += 1
        return super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )


def _make_optimizer(
    candidate: dict[str, Any],
    *,
    atoms: Atoms,
    logfile: Path,
    trajectory: Path,
):
    common = {
        "atoms": atoms,
        "logfile": str(logfile),
        "trajectory": str(trajectory),
        "maxstep": float(candidate["maxstep_angstrom"]),
    }
    algorithm = candidate["algorithm"]
    if algorithm == "ASE-BFGSLineSearch":
        return BFGSLineSearch(
            **common,
            alpha=float(candidate["alpha"]),
            c1=float(candidate["c1"]),
            c2=float(candidate["c2"]),
            stpmax=float(candidate["stpmax"]),
        )
    if algorithm == "ASE-LBFGSLineSearch":
        return LBFGSLineSearch(
            **common,
            memory=int(candidate["memory"]),
            damping=float(candidate["damping"]),
            alpha=float(candidate["alpha"]),
        )
    if algorithm == "ASE-FIRE2-ABC":
        return FIRE2(
            **common,
            dt=float(candidate["dt"]),
            dtmax=float(candidate["dtmax"]),
            dtmin=float(candidate["dtmin"]),
            Nmin=int(candidate["Nmin"]),
            finc=float(candidate["finc"]),
            fdec=float(candidate["fdec"]),
            astart=float(candidate["astart"]),
            fa=float(candidate["fa"]),
            use_abc=True,
        )
    raise ValueError(f"Unsupported optimizer algorithm: {algorithm}.")


def _run_branch(
    *,
    template_atoms: Atoms,
    initial_positions: np.ndarray,
    source_state_index: int,
    source_calculator,
    candidate: dict[str, Any],
    gates: dict[str, Any],
    phase: str,
    branch_dir: Path,
) -> dict[str, Any]:
    """Run one frozen branch and retain the observable-only convergence trace."""

    atoms = template_atoms.copy()
    atoms.set_positions(initial_positions)
    adapter = CountingHartreeToEVCalculator(
        source_calculator,
        maximum_calculate_calls=int(
            gates["maximum_calculator_evaluations_per_branch"]
        ),
    )
    atoms.calc = adapter
    branch_dir.mkdir(parents=True, exist_ok=True)
    logfile = branch_dir / f"{phase}.log"
    trajectory = branch_dir / f"{phase}.traj"
    trace: list[dict[str, float | int]] = []

    def observe() -> None:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        trace.append(
            {
                "observation_index": len(trace),
                "energy_eV": energy,
                "maximum_force_eV_per_angstrom": QRRHO._maximum_force(forces),
            }
        )

    start = time.perf_counter()
    try:
        observe()
        initial_source = QRRHO._source_energy_result(source_calculator)
        optimizer = _make_optimizer(
            candidate,
            atoms=atoms,
            logfile=logfile,
            trajectory=trajectory,
        )
        optimizer.attach(observe, interval=1)
        optimizer_reported_converged = bool(
            optimizer.run(
                fmax=float(gates["fmax_eV_per_angstrom"]),
                steps=int(gates["maximum_steps"]),
            )
        )
        observe()
        final_source = QRRHO._source_energy_result(source_calculator)
        final_force = float(trace[-1]["maximum_force_eV_per_angstrom"])
        finite_trace = all(
            math.isfinite(float(row["energy_eV"]))
            and math.isfinite(float(row["maximum_force_eV_per_angstrom"]))
            for row in trace
        )
        final_energy_not_increased = (
            float(trace[-1]["energy_eV"]) - float(trace[0]["energy_eV"])
            <= float(gates["maximum_final_energy_increase_eV"])
        )
        force_converged = (
            final_force
            <= float(gates["fmax_eV_per_angstrom"])
            + float(gates["force_comparison_tolerance_eV_per_angstrom"])
        )
        passed = bool(
            optimizer_reported_converged
            and force_converged
            and finite_trace
            and final_energy_not_increased
        )
        energy_differences = np.diff(
            np.asarray([row["energy_eV"] for row in trace], dtype=np.float64)
        )
        return {
            "status": "success",
            "phase": phase,
            "source_state_index": int(source_state_index),
            "passed": passed,
            "converged": bool(optimizer_reported_converged and force_converged),
            "optimizer_reported_converged": optimizer_reported_converged,
            "force_converged": force_converged,
            "finite_trace": finite_trace,
            "final_energy_not_increased": final_energy_not_increased,
            "steps": int(optimizer.nsteps),
            "calculator_evaluations": int(adapter.calculate_calls),
            "elapsed_seconds": time.perf_counter() - start,
            "initial_max_force_eV_per_angstrom": float(
                trace[0]["maximum_force_eV_per_angstrom"]
            ),
            "final_max_force_eV_per_angstrom": final_force,
            "initial_energy": initial_source,
            "final_energy": final_source,
            "final_minus_initial_energy_eV": (
                float(trace[-1]["energy_eV"]) - float(trace[0]["energy_eV"])
            ),
            "maximum_observed_uphill_step_eV": (
                max(0.0, float(energy_differences.max()))
                if len(energy_differences)
                else 0.0
            ),
            "trace": trace,
            "trace_sha256": sha256_bytes(canonical_json_bytes(trace)),
            "final_positions_angstrom": atoms.get_positions().tolist(),
            "logfile": _relative_to_repository(logfile),
            "trajectory": _relative_to_repository(trajectory),
        }
    except Exception as exc:
        finite_trace = all(
            math.isfinite(float(row["energy_eV"]))
            and math.isfinite(float(row["maximum_force_eV_per_angstrom"]))
            for row in trace
        )
        return {
            "status": "failure",
            "phase": phase,
            "source_state_index": int(source_state_index),
            "passed": False,
            "converged": False,
            "calculator_evaluations": int(adapter.calculate_calls),
            "elapsed_seconds": time.perf_counter() - start,
            "finite_trace": finite_trace,
            "trace": trace,
            "trace_sha256": sha256_bytes(canonical_json_bytes(trace)),
            "logfile": _relative_to_repository(logfile),
            "trajectory": (
                _relative_to_repository(trajectory)
                if trajectory.is_file()
                else None
            ),
            "calculator_evaluation_budget_exhausted": (
                isinstance(exc, RuntimeError)
                and str(exc).startswith(
                    "Frozen per-branch calculator-evaluation budget exhausted:"
                )
            ),
            "failure": {
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            },
        }


def summarize_candidate_records(
    records: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive global pass/fail and efficiency metrics for every candidate."""

    expected = int(
        protocol["qualification_scope"]["expected_branch_count_per_candidate"]
    )
    summaries = []
    for candidate in protocol["optimizer_candidates"]:
        candidate_id = candidate["candidate_id"]
        branches = [
            branch
            for record in records
            if record["candidate_id"] == candidate_id
            for phase in protocol["qualification_scope"]["phases"]
            for branch in record["branches"][phase]
        ]
        passed = len(branches) == expected and all(
            branch.get("status") == "success" and branch.get("passed") is True
            for branch in branches
        )
        evaluations = sum(int(branch["calculator_evaluations"]) for branch in branches)
        steps = sum(int(branch.get("steps", 0)) for branch in branches)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "candidate_order": int(candidate["candidate_order"]),
                "expected_branch_count": expected,
                "observed_branch_count": len(branches),
                "passed_branch_count": sum(
                    branch.get("status") == "success"
                    and branch.get("passed") is True
                    for branch in branches
                ),
                "all_branches_passed": passed,
                "total_calculator_evaluations": evaluations,
                "total_optimizer_steps": steps,
                "maximum_final_force_eV_per_angstrom": (
                    max(
                        float(branch["final_max_force_eV_per_angstrom"])
                        for branch in branches
                        if branch.get("status") == "success"
                    )
                    if any(branch.get("status") == "success" for branch in branches)
                    else None
                ),
                "maximum_final_energy_increase_eV": (
                    max(
                        float(branch["final_minus_initial_energy_eV"])
                        for branch in branches
                        if branch.get("status") == "success"
                    )
                    if any(branch.get("status") == "success" for branch in branches)
                    else None
                ),
                "wall_seconds": sum(float(branch["elapsed_seconds"]) for branch in branches),
            }
        )
    return summaries


def select_global_candidate(
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Apply the frozen label-blind, no-per-model-cherry-picking rule."""

    eligible = [summary for summary in summaries if summary["all_branches_passed"]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda summary: (
            int(summary["total_calculator_evaluations"]),
            int(summary["total_optimizer_steps"]),
            int(summary["candidate_order"]),
        ),
    )


def _model_case_record(
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    qrrho_protocol: dict[str, Any],
    source_manifest: dict[str, Any],
    model: dict[str, Any],
    candidate: dict[str, Any],
    calculator,
    work_dir: Path,
) -> dict[str, Any]:
    case_id = protocol["qualification_scope"]["case_ids"][0]
    protocol_case = next(
        case for case in qrrho_protocol["cases"] if case["compound_id"] == case_id
    )
    source_case = next(
        case for case in source_manifest["cases"] if case["compound_id"] == case_id
    )
    atoms, positions, solvent_kcal, charge_evidence = QRRHO._load_case_sources(
        protocol_case=protocol_case,
        source_case=source_case,
    )
    calculator.solvent_correction = None
    gas_energy, repeat_evidence = QRRHO._evaluate_state_energies(
        calculator=calculator,
        template_atoms=atoms,
        positions=positions,
        batch_size=int(qrrho_protocol["execution"]["batch_size"]),
        repeat_count=int(qrrho_protocol["execution"]["source_energy_repeat_count"]),
        maximum_relative_difference_kcal_mol=float(
            qrrho_protocol["execution"][
                "maximum_repeat_relative_energy_difference_kcal_mol"
            ]
        ),
    )
    solution_potential = gas_energy + solvent_kcal / QRRHO.KCAL_PER_HARTREE
    symbols = list(atoms.get_chemical_symbols())
    maximum_seed_count = int(
        qrrho_protocol["phase_specific_sampling"][
            "maximum_seed_count_per_phase_model_case"
        ]
    )
    rmsd_threshold = float(
        qrrho_protocol["optimized_minimum_deduplication"][
            "heavy_atom_rmsd_threshold_angstrom"
        ]
    )
    seeds = {
        "gas": QRRHO.select_diverse_seeds(
            positions_angstrom=positions,
            potential_hartree=gas_energy,
            symbols=symbols,
            maximum_count=maximum_seed_count,
            threshold_angstrom=rmsd_threshold,
        ),
        "solution": QRRHO.select_diverse_seeds(
            positions_angstrom=positions,
            potential_hartree=solution_potential,
            symbols=symbols,
            maximum_count=maximum_seed_count,
            threshold_angstrom=rmsd_threshold,
        ),
    }
    case_dir = (
        work_dir
        / "audit"
        / candidate["candidate_id"]
        / model["name"]
        / case_id
    )
    gates = protocol["qualification_gates"]
    branches: dict[str, list[dict[str, Any]]] = {}
    for phase in protocol["qualification_scope"]["phases"]:
        if phase == "gas":
            calculator.solvent_correction = None
        else:
            QRRHO._attach_solution_correction(
                calculator,
                atoms=atoms,
                charge_evidence=charge_evidence,
                protocol=qrrho_protocol,
            )
        branches[phase] = [
            _run_branch(
                template_atoms=atoms,
                initial_positions=positions[seed["source_state_index"]],
                source_state_index=int(seed["source_state_index"]),
                source_calculator=calculator,
                candidate=candidate,
                gates=gates,
                phase=phase,
                branch_dir=case_dir / f"state-{seed['source_state_index']:04d}",
            )
            for seed in seeds[phase]
        ]
    calculator.solvent_correction = None
    record = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-optimizer-qualification-model-candidate",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "status": "complete",
        "source_partition": "development",
        "model": model,
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "case": {
            "compound_id": case_id,
            "name": protocol_case["name"],
            "flexibility_bin": protocol_case["flexibility_bin"],
        },
        "source_state_evidence": {
            "state_file": protocol_case["state_file"],
            "state_file_sha256": protocol_case["state_file_sha256"],
            "gas_energy_sha256": QRRHO._float_list_sha256(gas_energy),
            "solution_potential_sha256": QRRHO._float_list_sha256(solution_potential),
            "repeat_evidence": repeat_evidence,
        },
        "selected_phase_seeds": seeds,
        "branches": branches,
        "label_boundary": protocol["label_boundary"],
    }
    if _contains_forbidden_label_key(record):
        raise ValueError("Qualification record contains a forbidden label.")
    return seal_artifact(record)


def validate_record(
    record: dict[str, Any],
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    model: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    _validate_self_hash(record, name="optimizer qualification record")
    if (
        record.get("protocol_id") != protocol["protocol_id"]
        or record.get("protocol_fingerprint") != fingerprint
        or record.get("status") != "complete"
        or record.get("model") != model
        or record.get("candidate_id") != candidate["candidate_id"]
        or record.get("candidate") != candidate
        or _contains_forbidden_label_key(record)
    ):
        raise ValueError("Optimizer qualification record identity changed.")
    phases = protocol["qualification_scope"]["phases"]
    if set(record.get("branches", {})) != set(phases):
        raise ValueError("Optimizer qualification phase roster changed.")
    expected_per_phase = (
        int(protocol["qualification_scope"]["expected_branch_count_per_candidate"])
        // len(protocol["qualification_scope"]["model_ids"])
        // len(phases)
    )
    for phase in phases:
        branches = record["branches"][phase]
        if len(branches) != expected_per_phase:
            raise ValueError("Optimizer qualification branch count changed.")
        expected_indices = [
            int(seed["source_state_index"])
            for seed in record["selected_phase_seeds"][phase]
        ]
        if [int(branch["source_state_index"]) for branch in branches] != expected_indices:
            raise ValueError("Optimizer qualification branch order changed.")


def validate_command(args: argparse.Namespace) -> None:
    protocol, fingerprint, _, _ = load_protocol(args.protocol)
    print(
        f"Validated {protocol['protocol_id']} fingerprint={fingerprint} "
        f"for {len(protocol['optimizer_candidates'])} candidates x "
        f"{protocol['qualification_scope']['expected_branch_count_per_candidate']} "
        "branches."
    )


def run_command(args: argparse.Namespace) -> None:
    import torch

    protocol, fingerprint, qrrho_protocol, source_manifest = load_protocol(
        args.protocol
    )
    device = torch.device(protocol["execution"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The preregistered cuda:0 device is unavailable.")
    work_dir = Path(args.work_dir).resolve()
    _relative_to_repository(work_dir)
    records_dir = work_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    case_id = protocol["qualification_scope"]["case_ids"][0]
    first_case = next(
        case for case in qrrho_protocol["cases"] if case["compound_id"] == case_id
    )
    source_case = next(
        case for case in source_manifest["cases"] if case["compound_id"] == case_id
    )
    first_atoms, _, _, _ = QRRHO._load_case_sources(
        protocol_case=first_case,
        source_case=source_case,
    )
    models = [
        model
        for model in qrrho_protocol["models"]
        if model["name"] in protocol["qualification_scope"]["model_ids"]
    ]
    for model in models:
        calculator = SetCalculator(
            device,
            model["name"],
            str(work_dir / f"{model['name']}.log"),
            atoms=first_atoms,
            model_options={"hessian": "numerical"},
        ).set_calculator()
        for candidate in protocol["optimizer_candidates"]:
            destination = (
                records_dir / f"{candidate['candidate_id']}--{model['name']}.json"
            )
            if destination.exists():
                existing = load_json(destination)
                validate_record(
                    existing,
                    protocol=protocol,
                    fingerprint=fingerprint,
                    model=model,
                    candidate=candidate,
                )
                print(
                    f"skip {candidate['candidate_id']}/{model['name']}",
                    flush=True,
                )
                continue
            print(f"run {candidate['candidate_id']}/{model['name']}", flush=True)
            record = _model_case_record(
                protocol=protocol,
                fingerprint=fingerprint,
                qrrho_protocol=qrrho_protocol,
                source_manifest=source_manifest,
                model=model,
                candidate=candidate,
                calculator=calculator,
                work_dir=work_dir,
            )
            validate_record(
                record,
                protocol=protocol,
                fingerprint=fingerprint,
                model=model,
                candidate=candidate,
            )
            write_json_atomic(destination, record)
            passed = all(
                branch["passed"]
                for phase in protocol["qualification_scope"]["phases"]
                for branch in record["branches"][phase]
            )
            print(
                f"{candidate['candidate_id']}/{model['name']}: "
                f"{'pass' if passed else 'fail'}",
                flush=True,
            )


def _load_complete_records(
    *,
    record_dir: Path,
    protocol: dict[str, Any],
    fingerprint: str,
    qrrho_protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    models = {model["name"]: model for model in qrrho_protocol["models"]}
    records = []
    for candidate in protocol["optimizer_candidates"]:
        for model_name in protocol["qualification_scope"]["model_ids"]:
            path = record_dir / f"{candidate['candidate_id']}--{model_name}.json"
            if not path.is_file():
                raise ValueError(f"Missing optimizer qualification record: {path}.")
            record = load_json(path)
            validate_record(
                record,
                protocol=protocol,
                fingerprint=fingerprint,
                model=models[model_name],
                candidate=candidate,
            )
            records.append(record)
    return records


def _copy_tree_atomic(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"Refusing to replace durable evidence: {destination}.")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            target = temporary / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def seal_command(args: argparse.Namespace) -> None:
    protocol, fingerprint, qrrho_protocol, _ = load_protocol(args.protocol)
    record_dir = Path(args.record_dir).resolve()
    if not record_dir.is_dir():
        raise ValueError("Optimizer qualification record directory is missing.")
    _relative_to_repository(record_dir)
    output = Path(args.output).resolve()
    if output.parent != SCRIPT_DIR:
        raise ValueError("Durable qualification output must be in the benchmark dir.")
    _relative_to_repository(output)
    if output.exists():
        raise ValueError("Refusing to replace a durable qualification artifact.")

    records = _load_complete_records(
        record_dir=record_dir,
        protocol=protocol,
        fingerprint=fingerprint,
        qrrho_protocol=qrrho_protocol,
    )
    summaries = summarize_candidate_records(records, protocol=protocol)
    selected = select_global_candidate(summaries)
    work_dir = record_dir.parent
    durable_raw = output.parent / f"{output.stem}-raw"
    _copy_tree_atomic(work_dir, durable_raw)
    raw_evidence = []
    for path in sorted(durable_raw.rglob("*")):
        if path.is_file():
            raw_evidence.append(
                {
                    "path": _relative_to_repository(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-optimizer-qualification",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "status": (
            "qualified-one-global-policy"
            if selected is not None
            else "failed-closed-no-global-policy"
        ),
        "qualification_scope": protocol["qualification_scope"],
        "qualification_gates": protocol["qualification_gates"],
        "candidate_summaries": summaries,
        "selected_global_policy": (
            next(
                candidate
                for candidate in protocol["optimizer_candidates"]
                if candidate["candidate_id"] == selected["candidate_id"]
            )
            if selected is not None
            else None
        ),
        "selection_evidence": selected,
        "selection_rule": protocol["selection_rule"],
        "claim_boundary": protocol["claim_boundary"],
        "label_boundary": protocol["label_boundary"],
        "route_boundary": protocol["route_boundary"],
        "records": [
            {
                "candidate_id": record["candidate_id"],
                "model": record["model"]["name"],
                "content_sha256": record["content_sha256"],
            }
            for record in records
        ],
        "raw_evidence": raw_evidence,
    }
    write_json_atomic(output, seal_artifact(artifact))
    print(
        f"Sealed {output} status={artifact['status']} "
        f"selected={selected['candidate_id'] if selected else None}."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.set_defaults(func=validate_command)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--work-dir", required=True)
    run_parser.set_defaults(func=run_command)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--record-dir", required=True)
    seal_parser.add_argument("--output", required=True)
    seal_parser.set_defaults(func=seal_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
