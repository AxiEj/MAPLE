"""Typed actual-evaluation ownership for V4 IRC terminal localization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("IRC terminal values must be finite.")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class IRCCorrectorResult:
    kind: str
    coordinate_mw: np.ndarray
    bracket_left_mw: np.ndarray | None = None
    bracket_right_mw: np.ndarray | None = None
    richardson_level: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "coordinate_mw", _readonly(self.coordinate_mw))
        for name in ("bracket_left_mw", "bracket_right_mw"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly(value))


@dataclass(frozen=True, slots=True)
class IRCEvaluation:
    coordinate_mw: np.ndarray
    positions_angstrom: np.ndarray
    energy_hartree: float
    forces_hartree_per_angstrom: np.ndarray
    gradient_mw: np.ndarray
    max_force: float
    rms_force: float

    def __post_init__(self):
        for name in (
            "coordinate_mw",
            "positions_angstrom",
            "forces_hartree_per_angstrom",
            "gradient_mw",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        for name in ("energy_hartree", "max_force", "rms_force"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"IRC evaluation {name} must be finite.")


@dataclass(frozen=True, slots=True)
class IRCMicroStepResult:
    kind: str
    evaluation: IRCEvaluation
    displacement_mw: np.ndarray
    termination_reason: str
    candidate_count: int
    new_evaluation_count: int

    def __post_init__(self):
        object.__setattr__(self, "displacement_mw", _readonly(self.displacement_mw))


def is_v4_algorithm(algorithm) -> bool:
    from ...pure_nonmd_status import is_pure_torch_v4

    return is_pure_torch_v4(algorithm.atoms)


def evaluate_coordinate(algorithm, coordinate_mw) -> IRCEvaluation:
    coordinate = _readonly(coordinate_mw)
    energy, forces = algorithm._energy_forces_from_mw(coordinate)
    force = _readonly(forces)
    gradient = _readonly(algorithm._gradient_mw_from_forces(force))
    return IRCEvaluation(
        coordinate_mw=coordinate,
        positions_angstrom=_readonly(algorithm.atoms.get_positions()),
        energy_hartree=float(energy),
        forces_hartree_per_angstrom=force,
        gradient_mw=gradient,
        max_force=float(np.max(np.abs(force))),
        rms_force=float(np.sqrt(np.mean(np.square(force)))),
    )


def evaluation_from_values(algorithm, coordinate_mw, energy, forces) -> IRCEvaluation:
    coordinate = _readonly(coordinate_mw)
    force = _readonly(forces)
    return IRCEvaluation(
        coordinate_mw=coordinate,
        positions_angstrom=_readonly(algorithm.atoms.get_positions()),
        energy_hartree=float(energy),
        forces_hartree_per_angstrom=force,
        gradient_mw=_readonly(algorithm._gradient_mw_from_forces(force)),
        max_force=float(np.max(np.abs(force))),
        rms_force=float(np.sqrt(np.mean(np.square(force)))),
    )


def restore_evaluation(algorithm, evaluation: IRCEvaluation) -> None:
    algorithm.mw_coords = np.array(evaluation.coordinate_mw, copy=True)
    algorithm.atoms.set_positions(evaluation.positions_angstrom)


def actual_force_passes(evaluation: IRCEvaluation, *, f_max, f_rms) -> bool:
    return bool(evaluation.max_force <= f_max and evaluation.rms_force <= f_rms)


def commit_micro_step(
    algorithm,
    *,
    kind,
    evaluation,
    macro_start_mw,
    termination_reason,
    candidate_count=1,
    new_evaluation_count=1,
) -> IRCMicroStepResult:
    """Commit one evaluated V4 micro-step outcome before returning it."""
    result = IRCMicroStepResult(
        kind=kind,
        evaluation=evaluation,
        displacement_mw=_readonly(
            evaluation.coordinate_mw - np.asarray(macro_start_mw)
        ),
        termination_reason=termination_reason,
        candidate_count=candidate_count,
        new_evaluation_count=new_evaluation_count,
    )
    restore_evaluation(algorithm, evaluation)
    algorithm.micro_counter += 1
    return result


def _coordinate_key(value) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float64).tobytes()


def localize_terminal_bracket(
    algorithm,
    corrector: IRCCorrectorResult,
    predictor: IRCEvaluation,
    *,
    macro_start_mw,
    f_max: float,
    f_rms: float,
    candidate_cap: int = 12,
) -> IRCMicroStepResult:
    if corrector.kind != "bracket":
        raise ValueError("terminal localization requires a typed bracket.")
    evaluations = {_coordinate_key(predictor.coordinate_mw): predictor}
    new_calls = 0

    def evaluate(coordinate):
        nonlocal new_calls
        key = _coordinate_key(coordinate)
        if key not in evaluations:
            evaluations[key] = evaluate_coordinate(algorithm, coordinate)
            new_calls += 1
        return evaluations[key]

    def finish(kind, evaluation, reason):
        return commit_micro_step(
            algorithm,
            kind=kind,
            evaluation=evaluation,
            macro_start_mw=macro_start_mw,
            termination_reason=reason,
            candidate_count=len(evaluations),
            new_evaluation_count=new_calls,
        )

    left_coord = np.asarray(corrector.bracket_left_mw, dtype=np.float64)
    right_coord = np.asarray(corrector.bracket_right_mw, dtype=np.float64)
    segment = right_coord - left_coord
    length = float(np.linalg.norm(segment))
    if not np.isfinite(length) or length <= np.finfo(np.float64).tiny:
        return finish("bounded_failure", predictor, "degenerate_terminal_bracket")
    tangent = segment / length

    def score(evaluation):
        force_score = max(evaluation.max_force / f_max, evaluation.rms_force / f_rms)
        segment_coordinate = float(
            np.dot(evaluation.coordinate_mw - left_coord, tangent) / length
        )
        return force_score, segment_coordinate

    def passing(evaluation):
        return actual_force_passes(evaluation, f_max=f_max, f_rms=f_rms)

    def fail(reason):
        return finish("bounded_failure", min(evaluations.values(), key=score), reason)

    left, right = evaluate(left_coord), evaluate(right_coord)
    passing_endpoints = [
        evaluation for evaluation in (left, right) if passing(evaluation)
    ]
    if passing_endpoints:
        return finish(
            "actual_endpoint",
            min(passing_endpoints, key=score),
            "endpoint_force_criteria_satisfied",
        )
    phi_left = float(np.dot(left.gradient_mw, tangent))
    phi_right = float(np.dot(right.gradient_mw, tangent))
    if phi_left == 0.0 or phi_right == 0.0:
        return fail("endpoint_localization_force_not_converged")
    if phi_left * phi_right > 0.0:
        return fail("surrogate_stationary_unconfirmed")

    while len(evaluations) < candidate_cap:
        middle_coord = 0.5 * (left.coordinate_mw + right.coordinate_mw)
        if _coordinate_key(middle_coord) in {
            _coordinate_key(left.coordinate_mw),
            _coordinate_key(right.coordinate_mw),
        }:
            return fail("endpoint_localization_no_coordinate_progress")
        # A cached interior point still reduces the bracket; only endpoint
        # equality means floating-point bisection cannot advance.
        middle = evaluate(middle_coord)
        if passing(middle):
            return finish(
                "actual_endpoint", middle, "endpoint_force_criteria_satisfied"
            )
        phi_middle = float(np.dot(middle.gradient_mw, tangent))
        if phi_middle == 0.0:
            return fail("endpoint_localization_force_not_converged")
        if phi_left * phi_middle < 0.0:
            right, phi_right = middle, phi_middle
        else:
            left, phi_left = middle, phi_middle
    return fail("endpoint_localization_force_not_converged")
