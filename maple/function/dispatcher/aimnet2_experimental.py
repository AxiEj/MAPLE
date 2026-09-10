"""Experimental AIMNet2 task execution, separate from scientific admission.

Only this opt-in lane tightens legacy optimizer outcomes.  The raw calculator
always speaks ASE units; the old optimizers receive the existing Hartree view.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from ase import Atoms

from ..calculator.calculator_base import EV2HARTREE
from .legacy_units import legacy_hartree_job_calculators

WORKFLOW_ID = "aimnet2-smooth-ddpcm-experimental-workflows-v2"
OPTIMIZATION_FORCE_TARGET_EV_PER_A = 1.0e-5
STATIONARITY_CEILING_EV_PER_A = 1.0e-3
_METRICS = ("max_f", "rms_f", "max_dp", "rms_dp")


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (str, int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite and positive.")
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _json_safe(value):
    """Preserve nonfinite failure evidence without emitting invalid JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_status(output: str, record: dict) -> None:
    destination = Path(str(output) + ".experimental.json")
    destination.write_text(
        json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n"
    )
    with open(output, "a") as handle:
        handle.write(
            f"\n[EXPERIMENTAL] {record['task']} run_status={record['run_status']}; "
            "formal scientific admission=false\n"
            f"[EXPERIMENTAL] Task evidence: {destination}\n"
        )


def _validated_results(atoms: Atoms, calculator, record: dict) -> float:
    energy = float(calculator.results["energy"])
    forces = np.asarray(calculator.results["forces"], dtype=float)
    record["energy_eV"] = energy
    record["forces_eV_per_A"] = forces.tolist()
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError("Final total forces must be finite and shape (N, 3).")
    if not np.isfinite(energy):
        raise ValueError("Final total energy must be finite.")
    maximum = float(np.max(np.abs(forces)))
    record["maximum_force_eV_per_A"] = maximum
    record["rms_force_eV_per_A"] = (
        maximum * float(np.sqrt(np.mean((forces / maximum) ** 2)))
        if maximum > 0.0
        else 0.0
    )
    guard = getattr(calculator, "last_domain_guard", None)
    record["domain_guard"] = guard
    if not isinstance(guard, dict) or guard.get("gate_passed") is not True:
        raise RuntimeError("Final total evaluation has no passing domain guard.")
    record["calculator_provenance"] = getattr(calculator, "provenance", None)
    return maximum


def _fresh_final(atoms: Atoms, calculator, record: dict) -> float:
    # Do not reset(): that would erase accepted-geometry topology history.
    before = calculator.evaluation_count
    calculator.calculate(atoms, properties=("energy", "forces"))
    record["postflight_evaluation_count"] = calculator.evaluation_count - before
    return _validated_results(atoms, calculator, record)


def _work_budget(task, params, count, *, hessian_evaluations_per_atom=18):
    hessian_cost = (
        _positive_integer(hessian_evaluations_per_atom, "hessian_evaluations_per_atom")
        * count
        + 2
    )
    if task == "sp":
        return 1, 0
    if task == "freq":
        return 1 + hessian_cost, 1
    iterations = _positive_integer(
        params.get("max_iter", 64 if task == "opt" else 24), "max_iter"
    )
    if task == "opt":
        return iterations + 2, 0
    recalc = _positive_integer(params.get("recalc", 6), "recalc")
    calls = (iterations + recalc - 1) // recalc + 1
    # Existing PRFO: <=8 trial attempts/iteration, plus an iteration center;
    # one possible final center on exhaustion and one forced final replay.
    return 9 * iterations + 2 + calls * hessian_cost, calls


def _run_optimizer(dispatcher, control, task, atoms, output, record):
    from .optimization.algorithm.LBFGS import LBFGS
    from .optimization.algorithm._common import is_converged
    from .ts.algorithm.PRFO import PRFO

    calculator: Any = atoms.calc
    if calculator is None:
        raise ValueError("Experimental workflow requires an attached calculator.")
    params = dict(control.params)
    maximum_iterations = _positive_integer(
        params.get("max_iter", 64 if task == "opt" else 24), "max_iter"
    )
    params["max_iter"] = maximum_iterations
    requested_thresholds = {
        name: _positive(params[name], name)
        for name in ("f_max_th", "f_rms_th", "dp_max_th", "dp_rms_th")
        if name in params
    }
    dispatcher.commandcontrol = control
    dispatcher.set_throshould(atoms)
    for name in ("f_max_th", "f_rms_th"):
        threshold = min(
            getattr(atoms, name),
            OPTIMIZATION_FORCE_TARGET_EV_PER_A * EV2HARTREE,
            requested_thresholds.get(name, np.inf),
        )
        setattr(atoms, name, threshold)
        params[name] = threshold
    for name in ("dp_max_th", "dp_rms_th"):
        threshold = min(getattr(atoms, name), requested_thresholds.get(name, np.inf))
        setattr(atoms, name, threshold)
        params[name] = threshold
    for name in _METRICS:
        if hasattr(atoms, name):
            delattr(atoms, name)
    if task == "opt":
        params["max_step"] = min(
            _positive(params.get("max_step", 0.02), "max_step"), 0.02
        )
    else:
        params["recalc"] = _positive_integer(params.get("recalc", 6), "recalc")
        for name in ("trust_radius", "trust_max"):
            params[name] = min(_positive(params.get(name, 0.02), name), 0.02)
        if (
            "trust_min" in params
            and _positive(params["trust_min"], "trust_min") > params["trust_max"]
        ):
            raise ValueError("trust_min must not exceed the experimental trust_max.")

    start_count = calculator.evaluation_count
    initial_positions = np.array(atoms.positions, copy=True)
    record["optimizer_parameters"] = params
    with legacy_hartree_job_calculators(atoms):
        if task == "opt":
            optimizer = LBFGS(atoms=atoms, output=output, paras=params)
            convergence_check = is_converged
        else:
            prfo = PRFO(atoms=atoms, output=output, paras=params)
            optimizer = prfo
            convergence_check = prfo.check_convergence
        optimizer.run()
        metrics = {name: float(getattr(atoms, name, np.nan)) for name in _METRICS}
        record["optimizer_metrics_hartree_per_A_and_A"] = metrics
        finite = all(np.isfinite(value) and value >= 0 for value in metrics.values())
        converged = finite and convergence_check(atoms)
    phase_count = calculator.evaluation_count - start_count
    record["optimizer_phase_evaluation_count"] = phase_count
    if task == "opt":
        record["optimizer_initial_evaluation_count"] = min(1, phase_count)
        record["optimizer_iteration_evaluation_count"] = max(0, phase_count - 1)
        record["evaluation_budget"] = maximum_iterations + 2
    final_maximum = _fresh_final(atoms, calculator, record)
    if (
        task == "opt"
        and calculator.evaluation_count - start_count > maximum_iterations + 2
    ):
        raise RuntimeError(
            "Experimental OPT exceeded its declared E/F evaluation budget."
        )
    displacement = float(np.max(np.abs(atoms.positions - initial_positions)))
    record["maximum_total_displacement_A"] = displacement
    if not np.isfinite(displacement) or displacement == 0.0:
        raise RuntimeError(
            "Experimental optimizer made no finite geometry displacement; not convergence."
        )
    if phase_count < 2:
        raise RuntimeError(
            "Experimental optimizer made no evaluated iteration; not convergence."
        )
    if not converged:
        raise RuntimeError(
            "Experimental optimizer did not converge; final structure retained."
        )
    force_limit = float(atoms.f_max_th) / EV2HARTREE
    if final_maximum > force_limit:
        raise RuntimeError("Fresh final forces do not satisfy optimizer convergence.")
    rms_limit = float(atoms.f_rms_th) / EV2HARTREE
    if record["rms_force_eV_per_A"] > rms_limit:
        raise RuntimeError(
            "Fresh final RMS forces do not satisfy optimizer convergence."
        )
    record["optimizer_converged"] = True


def _run_frequency(atoms, output, parameters, record, *, ts_postflight=False):
    from .frequency import Frequency

    params = dict(parameters)
    if not ts_postflight and params.get("stationary_point", "minimum") != "minimum":
        raise ValueError(
            "Experimental FREQ targets a minimum; TS performs its own saddle postflight."
        )
    for key, default, ceiling in (
        ("stationarity_tolerance_ev_per_a", 1.0e-3, 1.0e-3),
        ("rigid_mode_tolerance_cm1", 5.0, 5.0),
    ):
        if _positive(params.get(key, default), key) > ceiling:
            raise ValueError(f"Experimental FREQ {key} cannot exceed {ceiling}.")
    if params.get("treat_imag_as_real", False):
        raise ValueError("Experimental FREQ does not reinterpret negative modes.")
    if (
        _positive(
            params.get("transition_state_imaginary_threshold_cm1", 50.0),
            "transition_state_imaginary_threshold_cm1",
        )
        < 50.0
    ):
        raise ValueError(
            "The robust imaginary-mode threshold cannot be relaxed below 50 cm-1."
        )
    calculator: Any = atoms.calc
    hessians_before = calculator.hessian_call_count
    frequency = Frequency(output=output, atoms=atoms, paras=params)
    record["frequency_parameters"] = asdict(frequency.params)
    record["frequency_output"] = output
    record["stationary_point"] = frequency.params.stationary_point
    try:
        frequency.run()
    finally:
        # A rejected stationary-point index must retain the actual spectrum.
        # Reuse the same MW kernel on the returned matrix, not text parsing or
        # another expensive Hessian evaluation.  Do not reuse an older H if
        # stationarity rejected this request before a new Hessian was called.
        diagnostic = getattr(calculator, "last_hessian_diagnostics", None)
        if calculator.hessian_call_count == hessians_before + 1 and isinstance(
            diagnostic, dict
        ):
            record["frequency_hessian_diagnostic"] = diagnostic.get("sidecar_path")
            if diagnostic.get("status") == "passed":
                from .frequency.normal_modes import (
                    analyze_cartesian_hessian,
                    rigid_body_hessian_residual_cm1,
                )

                analysis = analyze_cartesian_hessian(
                    diagnostic["returned_hessian_eV_per_A2"],
                    atoms.get_masses(),
                    atoms.positions,
                )
                record["vibrational_frequencies_cm1"] = (
                    analysis.frequencies_cm1.tolist()
                )
                record["vibrational_eigenvalues_eV_per_A2_amu"] = (
                    analysis.eigenvalues_eV_per_A2_amu.tolist()
                )
                record["rigid_residual_cm1"] = rigid_body_hessian_residual_cm1(analysis)
    if calculator.hessian_call_count - hessians_before != 1:
        raise RuntimeError(
            "Frequency analysis must evaluate exactly one fresh Hessian."
        )


def run_experimental_workflow(dispatcher, control, task, atoms, output, extra=None):
    """Run only the opted-in lane, writing an honest result even on failure."""
    from ..aimnet2_experimental import validate_aimnet2_experimental_atoms

    if not isinstance(atoms, Atoms):
        raise ValueError("Experimental AIMNet2 workflows require one molecule.")
    calculator: Any = atoms.calc
    if calculator is None:
        raise ValueError("Experimental workflow requires an attached calculator.")
    record = {
        "workflow_id": getattr(calculator, "experimental_workflow_id", WORKFLOW_ID),
        "task": task,
        "run_status": "failed",
        "scientific_admission": False,
        "claim_boundary": "Experimental workflow evidence, not chemical accuracy or E/F/H/V/M admission.",
        "parameters": dict(control.params),
        "initial_positions_A": atoms.positions.tolist(),
        "dispatcher_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    started = time.monotonic()
    start_count = calculator.evaluation_count
    start_hessians = getattr(calculator, "hessian_call_count", 0)
    previous_budget = getattr(calculator, "evaluation_budget", None)
    previous_hessian_budget = getattr(calculator, "hessian_call_budget", None)
    try:
        validate_aimnet2_experimental_atoms(atoms)
        expected_methods = {
            "sp": {None, ""},
            "opt": {None, "", "lbfgs"},
            "freq": {None, "mw"},
            "ts": {"prfo"},
        }
        if (
            task not in expected_methods
            or control.params.get("method") not in expected_methods[task]
        ):
            raise ValueError(
                "Experimental AIMNet2 supports SP, OPT/LBFGS, FREQ/MW and TS/PRFO only."
            )
        budget, hessian_budget = _work_budget(
            task,
            control.params,
            len(atoms),
            hessian_evaluations_per_atom=getattr(
                calculator, "hessian_evaluations_per_atom", 18
            ),
        )
        calculator.evaluation_budget = start_count + budget
        calculator.hessian_call_budget = start_hessians + hessian_budget
        record["evaluation_budget"] = budget
        record["hessian_call_budget"] = hessian_budget
        record.update(
            scalar_id=calculator.scalar.scalar_id,
            profile_id=calculator.scalar.profile_id,
            scalar_fingerprint_sha256=calculator.scalar.fingerprint_sha256(),
            checkpoint_sha256=hashlib.sha256(
                calculator.checkpoint_path.read_bytes()
            ).hexdigest(),
        )
        with open(output, "a") as handle:
            handle.write(
                "\n[EXPERIMENTAL] Frozen AIMNet2 + smooth ddPCM + SMD-CDS; no electronic SCF.\n"
            )
        dispatcher.output = output
        dispatcher.commandcontrol = control
        if task == "sp":
            with legacy_hartree_job_calculators(atoms):
                dispatcher._dispatch(control, task, atoms, output, extra)
            atoms.get_potential_energy()
            atoms.get_forces()
            _validated_results(atoms, calculator, record)
        elif task in {"opt", "ts"}:
            _run_optimizer(dispatcher, control, task, atoms, output, record)
            if task == "ts":
                _run_frequency(
                    atoms,
                    str(output) + ".ts-validation.out",
                    {
                        "method": "mw",
                        "stationary_point": "transition_state",
                        "verbosity": 10,
                    },
                    record,
                    ts_postflight=True,
                )
        else:
            _run_frequency(atoms, output, control.params, record)
        if calculator.evaluation_count - start_count > budget:
            raise RuntimeError(
                "Experimental task exceeded its declared E/F evaluation budget."
            )
        if (
            getattr(calculator, "hessian_call_count", 0) - start_hessians
            > hessian_budget
        ):
            raise RuntimeError(
                "Experimental task exceeded its declared Hessian-call budget."
            )
        if (
            task in {"freq", "ts"}
            and getattr(calculator, "hessian_call_count", 0) <= start_hessians
        ):
            raise RuntimeError(
                "Experimental frequency postflight did not evaluate a fresh Hessian."
            )
        record["domain_guard"] = getattr(calculator, "last_domain_guard", None)
        record["calculator_provenance"] = getattr(calculator, "provenance", None)
        record["run_status"] = "converged" if task in {"opt", "ts"} else "completed"
        return record
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if record["run_status"] == "failed":
            record["last_attempted_evaluation"] = getattr(
                calculator, "last_attempted_evaluation", None
            )
        record["final_positions_A"] = atoms.positions.tolist()
        record["elapsed_seconds"] = time.monotonic() - started
        record["total_evaluation_count"] = calculator.evaluation_count - start_count
        record["hessian_call_count"] = (
            getattr(calculator, "hessian_call_count", 0) - start_hessians
        )
        diagnostic = getattr(calculator, "last_hessian_diagnostics", None)
        if (
            isinstance(diagnostic, dict)
            and getattr(calculator, "hessian_call_count", 0) > start_hessians
        ):
            record["last_hessian_diagnostic"] = {
                "sidecar_path": diagnostic.get("sidecar_path"),
                "status": diagnostic.get("status"),
                "schema_version": diagnostic.get("schema_version"),
                "refinement": diagnostic.get("refinement"),
                "failure": diagnostic.get("failure"),
            }
        calculator.evaluation_budget = previous_budget
        calculator.hessian_call_budget = previous_hessian_budget
        _write_status(output, record)
