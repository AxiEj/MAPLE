from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from maple.function.dispatcher.irc.algorithm.gs import GS
from maple.function.dispatcher.irc.algorithm.hpc import HPC
from maple.function.dispatcher.irc.algorithm.eulerpc import EulerPC
from maple.function.dispatcher.irc.algorithm.lqa import LQA
from maple.function.dispatcher.irc.algorithm._nonmd import preselected_mode_message
from maple.function.dispatcher.irc.algorithm._nonmd import run_directions
from maple.function.dispatcher.irc import irc as irc_module


def _atoms():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms, energy=0.0, free_energy=0.0, forces=np.zeros((2, 3), dtype=np.float64)
    )
    return atoms


def _side(reason="endpoint_force_criteria_satisfied", converged=True):
    return {
        "title": "side",
        "records": [
            {
                "E": -1.0,
                "maxG": 0.0 if converged else 1.0,
                "rmsG": 0.0 if converged else 1.0,
                "x": np.zeros((2, 3), dtype=np.float64),
            }
        ],
        "E_ts": 0.0,
        "converged": converged,
        "termination_reason": reason,
        "termination_class": "converged" if converged else "bounded_nonconvergence",
        "iterations": 0,
        "final_metrics": {},
        "endpoint_minimum_verified": False,
    }


@pytest.mark.parametrize("algorithm_type", [GS, HPC, EulerPC, LQA])
def test_gs_uses_one_preselected_mode_for_opposite_directions(
    monkeypatch, tmp_path, algorithm_type
):
    atoms = _atoms()
    gs = algorithm_type(
        atoms, output=str(tmp_path / "irc.out"), paras={"write_traj": False}
    )
    mode = np.arange(1.0, 7.0)
    gs.preselected_mode_mw = mode
    calls = []

    monkeypatch.setattr(
        gs, "_get_hessian_cart", lambda: pytest.fail("raw TS Hessian reselected")
    )

    def one_side(**kwargs):
        calls.append((kwargs["sign"], kwargs["v_neg_mw"].copy()))
        return _side()

    monkeypatch.setattr(gs, "_one_side", one_side)
    monkeypatch.setattr(gs, "_merge_and_mark_ts", lambda forward, backward: {})

    result = gs.run()

    assert [sign for sign, _ in calls] == [+1.0, -1.0]
    assert np.array_equal(calls[0][1], mode)
    assert np.array_equal(calls[1][1], mode)
    assert result["completed_directions"] == ["forward", "backward"]


@pytest.mark.parametrize("algorithm_type", [GS, HPC, EulerPC, LQA])
def test_gs_preserves_forward_when_backward_raises(
    monkeypatch, tmp_path, algorithm_type
):
    atoms = _atoms()
    gs = algorithm_type(
        atoms, output=str(tmp_path / "irc.out"), paras={"write_traj": False}
    )
    gs.preselected_mode_mw = np.ones(6)

    def one_side(*, forward, **kwargs):
        if forward:
            return _side()
        gs._active_direction_log = _side("corrector_failure", False)
        raise RuntimeError("mBS Richardson extrapolation did not converge")

    monkeypatch.setattr(gs, "_one_side", one_side)
    result = gs.run()

    assert result["completed_directions"] == ["forward", "backward"]
    assert result["forward"]["converged"] is True
    assert result["backward"]["converged"] is False
    assert result["backward"]["termination_reason"] == "corrector_failure"
    assert result["backward"]["termination_class"] == "execution_failure"
    assert result["backward"]["records"]


def test_v4_direction_exception_restores_last_authoritative_record(monkeypatch):
    atoms = _atoms()
    authoritative = np.array([[0.1, 0.0, 0.0], [0.8, 0.0, 0.0]])

    class Algorithm:
        def __init__(self):
            self.atoms = atoms
            self.preselected_mode_mw = np.ones(6)
            self.p = SimpleNamespace(write_traj=False)
            self._active_direction_log = {}
            self.mw_coords = np.zeros(3 * len(atoms))

        def _mw_from_cart(self, positions):
            return np.asarray(positions, dtype=float)

        def _merge_and_mark_ts(self, forward, backward):
            return {}

        def _write_trajs(self, forward, backward):
            return None

        def _one_side(self, *, forward, **kwargs):
            if forward:
                return _side()
            self.atoms.positions[:] = 9.0
            self._active_direction_log["records"] = [
                {
                    "E": -2.0,
                    "maxG": 0.1,
                    "rmsG": 0.05,
                    "x": authoritative.copy(),
                }
            ]
            raise RuntimeError("corrector failed")

    algorithm = Algorithm()
    monkeypatch.setattr(
        "maple.function.dispatcher.irc.algorithm._nonmd.is_v4_algorithm",
        lambda algorithm: True,
    )
    result = run_directions(algorithm, atoms.positions.reshape(-1), np.ones(6), 0.0)
    np.testing.assert_allclose(atoms.positions, authoritative)
    np.testing.assert_allclose(algorithm.mw_coords, authoritative.reshape(-1))
    assert result["backward"]["termination_class"] == "execution_failure"


def test_wrapper_preserves_algorithm_side_statuses(monkeypatch, tmp_path):
    atoms = _atoms()
    mode = np.ones(6) / np.sqrt(6.0)
    diagnostic = {"resolved_negative_count": 1}
    monkeypatch.setattr(
        irc_module,
        "_preselect_pure_nonmd_internal_mode",
        lambda atoms: (mode, diagnostic),
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2",
        lambda atoms: True,
    )

    class FakeGS:
        def __init__(self, *args, **kwargs):
            self.p = SimpleNamespace(f_max_th=0.1, f_rms_th=0.1)

        def run(self):
            return {
                "forward": _side("maximum_steps_reached", False),
                "backward": _side("microcycle_limit_reached", False),
                "completed_directions": ["forward", "backward"],
            }

    monkeypatch.setattr("maple.function.dispatcher.irc.algorithm.GS", FakeGS)
    status = irc_module.IRC({}, str(tmp_path / "irc.out"), atoms, "gs").run()

    assert status["termination_class"] == "bounded_nonconvergence"
    assert (
        status["directions"]["forward"]["termination_reason"] == "maximum_steps_reached"
    )
    assert (
        status["directions"]["backward"]["termination_reason"]
        == "microcycle_limit_reached"
    )
    assert status["directions"]["backward"]["endpoint_minimum_verified"] is False
    assert status["trace"][1]["records"][0]["positions_angstrom"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def test_automatic_irc_status_separates_branch_success_endpoint_and_connectivity(
    monkeypatch,
):
    atoms = _atoms()
    diagnostic = {
        "selection_kind": "computed_index_one_automatic",
        "computed_stationary_assessment": {"computed_index_one": True},
    }
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        irc_module,
        "_endpoint_computed_assessment",
        lambda atoms, positions: {"computed_minimum": True},
    )

    status = irc_module._irc_status(
        "gs",
        atoms,
        {"forward": _side(), "backward": _side()},
        diagnostic,
        0.002,
        0.0005,
    )

    assert status["converged"] is True
    assert status["final_metrics"]["numerical_stationary_point_validated"] is True
    assert status["final_metrics"]["endpoint_computed_minima"] == {
        "forward": True,
        "backward": True,
    }
    assert status["final_metrics"]["reaction_connectivity_verified"] is None


@pytest.mark.parametrize("endpoint_result", [False, None, RuntimeError("H failed")])
def test_automatic_irc_endpoint_failure_cannot_be_task_success(
    monkeypatch, endpoint_result
):
    atoms = _atoms()
    diagnostic = {
        "selection_kind": "computed_index_one_automatic",
        "computed_stationary_assessment": {"computed_index_one": True},
    }
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )

    def endpoint(*args):
        if isinstance(endpoint_result, Exception):
            raise endpoint_result
        return {"computed_minimum": endpoint_result}

    monkeypatch.setattr(irc_module, "_endpoint_computed_assessment", endpoint)
    status = irc_module._irc_status(
        "gs",
        atoms,
        {"forward": _side(), "backward": _side()},
        diagnostic,
        0.002,
        0.0005,
    )

    assert status["converged"] is False
    assert status["termination_class"] == "validation_failure"
    assert status["termination_reason"] == "endpoint_computed_minimum_validation_failed"
    assert status["directions"]["forward"]["converged"] is True
    assert status["directions"]["backward"]["converged"] is True
    assert status["final_metrics"]["numerical_stationary_point_validated"] is False


def test_mode_validation_failure_is_not_bounded_nonconvergence(monkeypatch, tmp_path):
    atoms = _atoms()
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        irc_module,
        "_preselect_pure_nonmd_internal_mode",
        lambda atoms: (_ for _ in ()).throw(
            irc_module.InvalidStartingHessian(
                "IRC requires exactly one uncertainty-resolved negative internal mode."
            )
        ),
    )

    status = irc_module.IRC({}, str(tmp_path / "irc.out"), atoms, "gs").run()

    assert status["converged"] is False
    assert status["termination_class"] == "validation_failure"
    assert status["termination_reason"] == "invalid_starting_hessian_index"


def test_v4_explicit_internal_mode_is_candidate_not_resolved(monkeypatch):
    atoms = _atoms()
    evaluation = SimpleNamespace(evaluation_sha256="e" * 64)
    identified = SimpleNamespace(evaluation=evaluation)
    atoms.calc = SimpleNamespace(
        get_hessian_evaluation=lambda atoms: evaluation,
        get_identified_hessian_evaluation=lambda atoms: identified,
    )
    analysis = SimpleNamespace(
        is_resolved_index_one=False,
        internal_dimension=2,
        statuses=("uncertain", "uncertain"),
        eigenvalues_eV_per_A2_amu=np.array([-0.2, 0.4]),
        modes_cartesian=np.array(
            [[1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, -1.0, 0.0]]
        ),
        resolved_negative_count=0,
        resolved_positive_count=0,
        uncertain_count=2,
        uncertainty_eV_per_A2_amu=None,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2",
        lambda atoms: False,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.molecular_modes.analyze_hessian_evaluation",
        lambda atoms, evaluation: analysis,
    )

    mode, diagnostic = irc_module._preselect_pure_nonmd_internal_mode(
        atoms, mode_index=0
    )
    assert np.isclose(np.linalg.norm(mode), 1.0)
    assert diagnostic["selection_kind"] == "explicit_raw_negative_candidate"
    assert diagnostic["selected_raw_eigenvalue_eV_per_A2_amu"] == -0.2
    assert diagnostic["uncertainty_eV_per_A2_amu"] is None


def test_v4_automatic_mode_requires_computed_index_one(monkeypatch):
    atoms = _atoms()
    evaluation = SimpleNamespace(evaluation_sha256="e" * 64)
    atoms.calc = SimpleNamespace(
        get_hessian_evaluation=lambda atoms: evaluation,
        get_identified_hessian_evaluation=lambda atoms: SimpleNamespace(
            evaluation=evaluation
        ),
    )
    analysis = SimpleNamespace(
        is_resolved_index_one=False,
        internal_dimension=2,
        statuses=("uncertain", "uncertain"),
        eigenvalues_eV_per_A2_amu=np.array([-0.2, 0.4]),
        modes_cartesian=np.array(
            [[1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, -1.0, 0.0]]
        ),
        resolved_negative_count=0,
        resolved_positive_count=0,
        uncertain_count=2,
        uncertainty_eV_per_A2_amu=None,
    )
    assessment = SimpleNamespace(
        computed_index_one=True,
        computed_minimum=False,
        computed_operator_quality_passed=True,
        computed_negative_count=1,
        computed_positive_count=1,
        computed_unresolved_count=0,
        computed_eigenvalues_eV_per_A2_amu=np.array([-0.2, 0.4]),
        computed_resolution_radius=1e-12,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2",
        lambda atoms: False,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.molecular_modes.analyze_hessian_evaluation",
        lambda atoms, evaluation: analysis,
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.computed_stationary.analyze_identified_hessian",
        lambda atoms, context: SimpleNamespace(
            mode_analysis=analysis,
            stationary_assessment=assessment,
        ),
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.computed_stationary.computed_assessment_record",
        lambda assessment: {"computed_index_one": True},
    )

    mode, diagnostic = irc_module._preselect_pure_nonmd_internal_mode(atoms)
    assert np.isclose(np.linalg.norm(mode), 1.0)
    assert diagnostic["selection_kind"] == "computed_index_one_automatic"
    assert diagnostic["computed_stationary_assessment"]["computed_index_one"] is True
    assert diagnostic["independent_hessian_uncertainty"] is None


def test_v4_explicit_internal_mode_rejects_nonnegative_eigenvalue(monkeypatch):
    atoms = _atoms()
    atoms.calc = SimpleNamespace(
        get_hessian_evaluation=lambda atoms: SimpleNamespace(evaluation_sha256="e" * 64)
    )
    analysis = SimpleNamespace(
        is_resolved_index_one=False,
        internal_dimension=1,
        statuses=("uncertain",),
        eigenvalues_eV_per_A2_amu=np.array([0.0]),
        modes_cartesian=np.ones((1, 6)),
        resolved_negative_count=0,
        resolved_positive_count=0,
        uncertain_count=1,
        uncertainty_eV_per_A2_amu=None,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2",
        lambda atoms: False,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.molecular_modes.analyze_hessian_evaluation",
        lambda atoms, evaluation: analysis,
    )
    with pytest.raises(irc_module.InvalidStartingHessian, match="raw-negative"):
        irc_module._preselect_pure_nonmd_internal_mode(atoms, mode_index=0)


def test_preselected_mode_log_keeps_v2_wording_and_labels_v4_candidate():
    assert preselected_mode_message("GS-IRC", {}) == (
        "\n[INFO] GS-IRC: Using dispatcher-preselected "
        "uncertainty-resolved internal negative mode.\n"
    )
    assert preselected_mode_message(
        "GS-IRC", {"selection_kind": "explicit_raw_negative_candidate"}
    ) == (
        "\n[INFO] GS-IRC: Using dispatcher-preselected "
        "explicit raw-negative candidate mode.\n"
    )
    assert preselected_mode_message(
        "GS-IRC", {"selection_kind": "computed_index_one_automatic"}
    ) == (
        "\n[INFO] GS-IRC: Using dispatcher-preselected "
        "automatic computed-index-one internal mode.\n"
    )


def test_v2_mode_diagnostic_does_not_gain_v4_candidate_fields(monkeypatch):
    atoms = _atoms()
    evaluation = SimpleNamespace(evaluation_sha256="e" * 64)
    atoms.calc = SimpleNamespace(get_hessian_evaluation=lambda atoms: evaluation)
    analysis = SimpleNamespace(
        is_resolved_index_one=True,
        internal_dimension=1,
        statuses=("negative",),
        eigenvalues_eV_per_A2_amu=np.array([-0.2]),
        modes_cartesian=np.ones((1, 6)),
        resolved_negative_count=1,
        resolved_positive_count=0,
        uncertain_count=0,
        uncertainty_eV_per_A2_amu=1e-6,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_nonmd_v2",
        lambda atoms: True,
    )
    monkeypatch.setattr(
        "maple.function.dispatcher.pure_nonmd_status.is_pure_torch_v4",
        lambda atoms: False,
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.molecular_modes.analyze_hessian_evaluation",
        lambda atoms, evaluation: analysis,
    )
    _, diagnostic = irc_module._preselect_pure_nonmd_internal_mode(atoms)
    assert diagnostic == {
        "mode_index": 0,
        "mode_statuses": ["negative"],
        "resolved_negative_count": 1,
        "resolved_positive_count": 0,
        "uncertain_count": 0,
        "uncertainty_eV_per_A2_amu": 1e-6,
        "hessian_evaluation_sha256": "e" * 64,
    }


def test_gs_microcycle_cap_stops_without_accepting_a_false_endpoint(
    monkeypatch, tmp_path
):
    atoms = _atoms()
    runner = GS(
        atoms,
        output=str(tmp_path / "micro.out"),
        paras={
            "write_traj": False,
            "max_steps": 2,
            "max_micro_cycles": 0,
        },
    )
    runner.preselected_mode_mw = np.ones(6)
    monkeypatch.setattr(runner, "_get_hessian_cart", lambda: np.eye(6))
    monkeypatch.setattr(
        runner, "_energy_forces_from_mw", lambda q: (-1.0, np.ones((2, 3)))
    )
    result = runner.run()
    for name in ("forward", "backward"):
        assert result[name]["converged"] is False
        assert result[name]["termination_reason"] == "microcycle_limit_reached"
        assert len(result[name]["records"]) == 1
