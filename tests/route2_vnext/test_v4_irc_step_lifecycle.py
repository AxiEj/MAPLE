from __future__ import annotations

import importlib
from types import MethodType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.irc.algorithm._terminal import (
    IRCCorrectorResult,
    IRCMicroStepResult,
    evaluate_coordinate,
    localize_terminal_bracket,
)
from maple.function.dispatcher.irc.algorithm.eulerpc import DWI as EulerDWI
from maple.function.dispatcher.irc.algorithm.eulerpc import EulerPC, EulerPCParams
from maple.function.dispatcher.irc.algorithm.hpc import DWI as HPCDWI
from maple.function.dispatcher.irc.algorithm.hpc import HPC, HPCParams

IMPLEMENTATIONS = (
    (HPC, HPCParams, "maple.function.dispatcher.irc.algorithm.hpc"),
    (EulerPC, EulerPCParams, "maple.function.dispatcher.irc.algorithm.eulerpc"),
)


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, root=0.0):
        super().__init__()
        self.root = root

    def check_state(self, atoms, tol=1e-15):
        return all_changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        positions = np.asarray(self.atoms.get_positions(), dtype=np.longdouble)
        displacement = positions.copy()
        displacement[0, 0] -= self.root
        energy = float(0.5 * np.sum(displacement * displacement))
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": np.asarray(-displacement, dtype=np.float64),
        }


def make_algorithm(monkeypatch, implementation, *, root, threshold=1e-12):
    algorithm_type, params_type, module_name = implementation
    monkeypatch.setattr(
        importlib.import_module(module_name), "is_v4_algorithm", lambda _: True
    )
    atoms = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    atoms.calc = QuadraticCalculator(root=root)
    params = params_type(f_max_th=threshold, f_rms_th=threshold, hessian_update="bfgs")
    if algorithm_type is HPC:
        params.euler_n = 8
    else:
        params.max_pred_steps = 20
    algorithm = algorithm_type(atoms, output="", params=params)
    algorithm._D = np.ones(3)
    algorithm._step_len_mw = 0.1
    algorithm._step_len_umw = 0.1
    algorithm.mw_coords = np.array([1.0, 0.0, 0.0])
    algorithm.mw_hessian = np.eye(3)
    if isinstance(algorithm, HPC):
        algorithm._dwi = HPCDWI()
    else:
        algorithm._dwi = EulerDWI()
    return algorithm


def assert_committed(algorithm, result, *, counter):
    coordinate = result.evaluation.coordinate_mw
    displacement = coordinate.astype(np.longdouble)
    displacement[0] -= algorithm.atoms.calc.root
    np.testing.assert_allclose(algorithm.mw_coords, coordinate)
    np.testing.assert_allclose(
        algorithm.atoms.get_positions(), coordinate.reshape(1, 3)
    )
    np.testing.assert_allclose(
        result.evaluation.positions_angstrom, coordinate.reshape(1, 3)
    )
    assert result.evaluation.energy_hartree == pytest.approx(
        float(0.5 * np.dot(displacement, displacement))
    )
    np.testing.assert_allclose(
        result.evaluation.forces_hartree_per_angstrom,
        -np.asarray(displacement, dtype=np.float64),
    )
    assert algorithm.micro_counter == counter


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_passing_predictor_commits_once_without_calling_corrector(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=0.0, threshold=2.0)

    def forbidden(*args):
        raise AssertionError("passing predictor must bypass the corrector")

    algorithm._corrector_step = MethodType(forbidden, algorithm)
    result = algorithm._micro_step()

    assert isinstance(result, IRCMicroStepResult)
    assert result.kind == "actual_endpoint"
    assert result.termination_reason == "endpoint_force_criteria_satisfied"
    assert_committed(algorithm, result, counter=1)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_stationary_macro_start_returns_a_committed_zero_endpoint(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=1.0)
    result = algorithm._micro_step()

    assert isinstance(result, IRCMicroStepResult)
    assert result.kind == "actual_endpoint"
    np.testing.assert_array_equal(result.displacement_mw, np.zeros(3))
    assert_committed(algorithm, result, counter=1)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_invalid_predictor_result_does_not_advance_committed_state(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=0.0, threshold=2.0)
    assert algorithm.mw_coords is not None
    start = algorithm.mw_coords.copy()
    terminal = importlib.import_module(
        "maple.function.dispatcher.irc.algorithm._terminal"
    )

    def invalid_result(**kwargs):
        raise ValueError("result validation failed")

    monkeypatch.setattr(terminal, "IRCMicroStepResult", invalid_result)
    with pytest.raises(ValueError, match="result validation failed"):
        algorithm._micro_step()

    np.testing.assert_array_equal(algorithm.mw_coords, start)
    assert algorithm.micro_counter == 0


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_regular_corrector_result_becomes_the_committed_step(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=-10.0)
    target = np.array([0.75, 0.0, 0.0])
    algorithm._corrector_step = lambda init_mw, step_length, dwi: IRCCorrectorResult(
        kind="regular", coordinate_mw=target
    )

    result = algorithm._micro_step()

    assert isinstance(result, IRCMicroStepResult)
    assert result.kind == "regular"
    np.testing.assert_allclose(result.evaluation.coordinate_mw, target)
    assert_committed(algorithm, result, counter=1)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_unconfirmed_corrector_commits_the_actual_predictor_as_bounded_failure(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=-10.0)
    algorithm._corrector_step = lambda init_mw, step_length, dwi: IRCCorrectorResult(
        kind="unconfirmed", coordinate_mw=np.zeros(3)
    )

    result = algorithm._micro_step()

    assert isinstance(result, IRCMicroStepResult)
    assert result.kind == "bounded_failure"
    assert result.termination_reason == "surrogate_stationary_unconfirmed"
    assert_committed(algorithm, result, counter=1)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_two_regular_steps_advance_one_counter_and_one_geometry_each(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=-10.0)
    targets = iter((np.array([0.75, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])))
    algorithm._corrector_step = lambda init_mw, step_length, dwi: IRCCorrectorResult(
        kind="regular", coordinate_mw=next(targets)
    )

    first = algorithm._micro_step()
    assert isinstance(first, IRCMicroStepResult)
    assert_committed(algorithm, first, counter=1)
    second = algorithm._micro_step()

    assert isinstance(second, IRCMicroStepResult)
    np.testing.assert_allclose(first.evaluation.coordinate_mw, [0.75, 0.0, 0.0])
    np.testing.assert_allclose(second.evaluation.coordinate_mw, [0.5, 0.0, 0.0])
    assert_committed(algorithm, second, counter=2)


def localization_algorithm(*, root):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = QuadraticCalculator(root=root)
    algorithm = HPC(atoms, output="")
    algorithm._D = np.ones(3)
    algorithm.mw_coords = np.zeros(3)
    return algorithm


def test_cached_interior_predictor_is_reused_before_localizing_the_root():
    algorithm = localization_algorithm(root=0.75)
    predictor = evaluate_coordinate(algorithm, np.array([0.5, 0.0, 0.0]))
    result = localize_terminal_bracket(
        algorithm,
        IRCCorrectorResult(
            kind="bracket",
            coordinate_mw=np.array([0.5, 0.0, 0.0]),
            bracket_left_mw=np.array([0.0, 0.0, 0.0]),
            bracket_right_mw=np.array([1.0, 0.0, 0.0]),
        ),
        predictor,
        macro_start_mw=np.zeros(3),
        f_max=1e-12,
        f_rms=1e-12,
    )

    assert result.kind == "actual_endpoint"
    np.testing.assert_allclose(result.evaluation.coordinate_mw, [0.75, 0.0, 0.0])
    assert result.candidate_count == 4
    assert result.new_evaluation_count == 3


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_native_corrector_brackets_consecutive_states_across_a_minimum(
    monkeypatch, implementation
):
    algorithm = make_algorithm(monkeypatch, implementation, root=1.0)
    assert algorithm._dwi is not None
    for coordinate in (0.9, 1.1):
        point = np.array([coordinate, 0.0, 0.0])
        gradient = point - np.array([1.0, 0.0, 0.0])
        algorithm._dwi.update(
            point, float(0.5 * gradient @ gradient), gradient, np.eye(3)
        )
    start = np.array([0.999, 0.0, 0.0])

    bracket = algorithm._corrector_step(start, 0.1, algorithm._dwi)

    assert isinstance(bracket, IRCCorrectorResult)
    assert bracket.kind == "bracket"
    assert bracket.bracket_left_mw is not None
    assert bracket.bracket_right_mw is not None
    assert bracket.bracket_left_mw[0] < 1.0 < bracket.bracket_right_mw[0]

    predictor = evaluate_coordinate(algorithm, np.array([1.01, 0.0, 0.0]))
    result = localize_terminal_bracket(
        algorithm,
        bracket,
        predictor,
        macro_start_mw=start,
        f_max=0.002,
        f_rms=0.0005,
    )
    assert result.kind == "actual_endpoint"
    assert result.evaluation.max_force <= 0.002
    assert result.evaluation.rms_force <= 0.0005
    assert_committed(algorithm, result, counter=1)


def test_adjacent_float_endpoints_stop_as_bounded_nonprogress():
    left = np.float64(1.0)
    right = np.nextafter(left, np.float64(2.0))
    root = (np.longdouble(left) + np.longdouble(right)) / 2
    algorithm = localization_algorithm(root=root)
    predictor = evaluate_coordinate(algorithm, np.array([2.0, 0.0, 0.0]))
    result = localize_terminal_bracket(
        algorithm,
        IRCCorrectorResult(
            kind="bracket",
            coordinate_mw=np.array([left, 0.0, 0.0]),
            bracket_left_mw=np.array([left, 0.0, 0.0]),
            bracket_right_mw=np.array([right, 0.0, 0.0]),
        ),
        predictor,
        macro_start_mw=np.zeros(3),
        f_max=1e-30,
        f_rms=1e-30,
    )

    assert result.kind == "bounded_failure"
    assert result.termination_reason == "endpoint_localization_no_coordinate_progress"
    np.testing.assert_array_equal(result.evaluation.coordinate_mw, [left, 0.0, 0.0])
