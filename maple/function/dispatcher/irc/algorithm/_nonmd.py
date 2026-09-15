"""Outcome preservation for the explicitly selected pure non-MD IRC path."""
from __future__ import annotations

import numpy as np


class IRCIterationLimit(RuntimeError):
    """A bounded numerical subproblem stopped before a valid next step."""


def is_nonmd(algorithm):
    return getattr(algorithm, "preselected_mode_mw", None) is not None


def remember_initial(algorithm, energy, max_force, rms_force):
    if is_nonmd(algorithm):
        algorithm._active_direction_log["initial_evaluation"] = {
            "E": float(energy), "maxG": float(max_force), "rmsG": float(rms_force),
            "x": algorithm.atoms.get_positions().copy(),
        }


def finish_direction(algorithm, title, records, energy_ts, reason, converged):
    result = {"title": title, "records": records, "E_ts": energy_ts}
    if is_nonmd(algorithm):
        last = records[-1] if records else None
        result.update(
            converged=bool(converged), termination_reason=reason,
            termination_class="converged" if converged else "bounded_nonconvergence",
            iterations=max(0, len(records) - 1), endpoint_minimum_verified=False,
            final_metrics={} if last is None else {
                "energy_hartree": float(last["E"]),
                "max_force_hartree_per_angstrom": float(last["maxG"]),
                "rms_force_hartree_per_angstrom": float(last["rmsG"]),
            },
        )
        algorithm._active_direction_log = result
    return result


def run_directions(algorithm, positions_ts, mode_mw, energy_ts):
    """Use one mode in both directions and retain each outcome, including errors."""
    mode = np.asarray(mode_mw, dtype=float)
    if mode.shape != (3 * len(algorithm.atoms),) or not np.isfinite(mode).all() or not np.linalg.norm(mode):
        raise ValueError("preselected IRC mode must be a finite nonzero 3N vector")
    result = {}
    algorithm.completed_directions = result
    for name, sign in (("forward", 1.0), ("backward", -1.0)):
        algorithm.atoms.set_positions(np.asarray(positions_ts).reshape(-1, 3))
        algorithm._active_direction_log = {
            "title": name, "E_ts": energy_ts, "records": [],
            "termination_reason": "direction_execution_failed",
        }
        try:
            side = algorithm._one_side(
                forward=name == "forward", sign=sign, q_ts_cart=positions_ts,
                v_neg_mw=mode.copy(), E_ts=energy_ts,
            )
        except Exception as exc:
            side = dict(algorithm._active_direction_log)
            if not side["records"] and "initial_evaluation" in side:
                side["records"] = [side["initial_evaluation"]]
            numerical_limit = isinstance(exc, IRCIterationLimit)
            side.update(
                converged=False,
                termination_reason=str(exc) if numerical_limit else side["termination_reason"],
                termination_class="bounded_nonconvergence" if numerical_limit else "execution_failure",
                iterations=max(0, len(side["records"]) - 1),
                endpoint_minimum_verified=False, error_type=type(exc).__name__, error=str(exc),
            )
        result[name] = side
    forward, backward = result["forward"], result["backward"]
    result["summary"] = {}
    if forward["records"] and backward["records"]:
        result["summary"] = algorithm._merge_and_mark_ts(forward, backward)
        if algorithm.p.write_traj:
            algorithm._write_trajs(forward, backward)
    result["completed_directions"] = ["forward", "backward"]
    return result
