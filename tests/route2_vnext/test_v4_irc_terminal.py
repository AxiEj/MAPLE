from __future__ import annotations

import numpy as np
from ase import Atoms

from maple.function.dispatcher.irc.algorithm._terminal import (
    IRCCorrectorResult,
    IRCEvaluation,
    localize_terminal_bracket,
)


class _Algorithm:
    def __init__(self, *, transverse=0.0, root=0.5):
        self.atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        self.mw_coords = np.zeros(2)
        self.micro_counter = 0
        self.transverse = transverse
        self.root = root

    def _energy_forces_from_mw(self, coordinate):
        q = np.asarray(coordinate, dtype=float)
        self.atoms.positions[0, 0] = q[0]
        return float(np.dot(q, q)), np.array([q[0] - self.root, self.transverse])

    @staticmethod
    def _gradient_mw_from_forces(forces):
        return np.asarray(forces, dtype=float)


def _predictor(algorithm, coordinate=(0.2, 0.0)):
    energy, forces = algorithm._energy_forces_from_mw(coordinate)
    return IRCEvaluation(
        coordinate_mw=np.asarray(coordinate, dtype=float),
        positions_angstrom=algorithm.atoms.positions.copy(),
        energy_hartree=energy,
        forces_hartree_per_angstrom=np.asarray(forces, dtype=float),
        gradient_mw=np.asarray(forces, dtype=float),
        max_force=float(np.max(np.abs(forces))),
        rms_force=float(np.sqrt(np.mean(np.square(forces)))),
    )


def _bracket():
    return IRCCorrectorResult(
        kind="bracket",
        coordinate_mw=np.array([0.0, 0.0]),
        bracket_left_mw=np.array([0.0, 0.0]),
        bracket_right_mw=np.array([1.0, 0.0]),
        richardson_level=1,
    )


def test_actual_bracket_midpoint_passes_both_force_gates_and_restores_state():
    algorithm = _Algorithm()
    result = localize_terminal_bracket(
        algorithm,
        _bracket(),
        _predictor(algorithm),
        macro_start_mw=np.zeros(2),
        f_max=1e-4,
        f_rms=1e-4,
    )
    assert result.kind == "actual_endpoint"
    assert result.candidate_count == 4
    assert result.new_evaluation_count == 3
    np.testing.assert_allclose(result.evaluation.coordinate_mw, [0.5, 0.0])
    np.testing.assert_allclose(algorithm.mw_coords, result.evaluation.coordinate_mw)
    np.testing.assert_allclose(
        algorithm.atoms.positions, result.evaluation.positions_angstrom
    )


def test_zero_directional_derivative_with_transverse_force_is_not_success():
    algorithm = _Algorithm(transverse=1.0)
    result = localize_terminal_bracket(
        algorithm,
        _bracket(),
        _predictor(algorithm),
        macro_start_mw=np.zeros(2),
        f_max=1e-4,
        f_rms=1e-4,
    )
    assert result.kind == "bounded_failure"
    assert result.termination_reason == "endpoint_localization_force_not_converged"


def test_localization_cap_counts_exact_distinct_candidates():
    algorithm = _Algorithm(transverse=1.0, root=0.3)
    result = localize_terminal_bracket(
        algorithm,
        _bracket(),
        _predictor(algorithm),
        macro_start_mw=np.zeros(2),
        f_max=1e-4,
        f_rms=1e-4,
        candidate_cap=12,
    )
    assert result.kind == "bounded_failure"
    assert result.candidate_count == 12
    assert result.new_evaluation_count == 11
    assert result.termination_reason == "endpoint_localization_force_not_converged"


def test_unconfirmed_surrogate_is_not_passed_to_localizer():
    algorithm = _Algorithm()
    corrector = IRCCorrectorResult(
        kind="unconfirmed",
        coordinate_mw=np.zeros(2),
        bracket_left_mw=np.zeros(2),
        bracket_right_mw=np.ones(2),
    )
    try:
        localize_terminal_bracket(
            algorithm,
            corrector,
            _predictor(algorithm),
            macro_start_mw=np.zeros(2),
            f_max=1e-4,
            f_rms=1e-4,
        )
    except ValueError as exc:
        assert "typed bracket" in str(exc)
    else:
        raise AssertionError("unconfirmed surrogate must fail closed")


def test_degenerate_bracket_returns_typed_bounded_failure_not_exception():
    algorithm = _Algorithm()
    predictor = _predictor(algorithm)
    point = np.array([0.3, 0.0])
    result = localize_terminal_bracket(
        algorithm,
        IRCCorrectorResult(
            kind="bracket",
            coordinate_mw=point,
            bracket_left_mw=point,
            bracket_right_mw=point,
        ),
        predictor,
        macro_start_mw=np.array([0.0, 0.0]),
        f_max=1e-4,
        f_rms=1e-4,
    )
    assert result.kind == "bounded_failure"
    assert result.termination_reason == "degenerate_terminal_bracket"


def test_localized_displacement_is_measured_from_macro_start():
    algorithm = _Algorithm()
    result = localize_terminal_bracket(
        algorithm,
        _bracket(),
        _predictor(algorithm, coordinate=(0.2, 0.0)),
        macro_start_mw=np.array([0.1, 0.0]),
        f_max=1e-4,
        f_rms=1e-4,
    )
    np.testing.assert_allclose(result.displacement_mw, [0.4, 0.0])


def test_adjacent_float_midpoint_stops_and_tie_prefers_lower_segment_coordinate():
    left = 1.0
    right = np.nextafter(left, 2.0)

    class Discrete(_Algorithm):
        def _energy_forces_from_mw(self, coordinate):
            q = np.asarray(coordinate, dtype=float)
            self.atoms.positions[0, 0] = q[0]
            sign = -1.0 if q[0] == left else 1.0
            return float(q[0]), np.array([sign, 0.0])

    algorithm = Discrete()
    predictor = _predictor(algorithm, coordinate=(2.0, 0.0))
    result = localize_terminal_bracket(
        algorithm,
        IRCCorrectorResult(
            kind="bracket",
            coordinate_mw=np.array([left, 0.0]),
            bracket_left_mw=np.array([left, 0.0]),
            bracket_right_mw=np.array([right, 0.0]),
        ),
        predictor,
        macro_start_mw=np.zeros(2),
        f_max=0.1,
        f_rms=0.1,
    )
    assert result.kind == "bounded_failure"
    assert result.termination_reason == "endpoint_localization_no_coordinate_progress"
    assert result.candidate_count == 3
    np.testing.assert_array_equal(result.evaluation.coordinate_mw, [left, 0.0])
