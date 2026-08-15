from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from maple.function.dispatcher.irc.path import (
    DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM,
    DEFAULT_IRC_MAX_STEPS,
    DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM,
    IRCPathParams,
    assemble_irc_path,
    enforce_irc_path_admission,
    finalize_irc_branch,
    make_irc_record,
    render_irc_path_summary,
    validate_irc_path_params,
    write_irc_trajectories,
)
from maple.function.dispatcher.dispatcher import Dispatcher


def _record(
    coordinate: float,
    energy: float,
    *,
    maximum_force: float = 1.0e-3,
    rms_force: float = 2.0e-4,
) -> dict:
    return {
        "E": energy,
        "maxG": maximum_force,
        "rmsG": rms_force,
        "x": np.array([[coordinate, 0.0, 0.0]]),
    }


def _branch(
    direction: str,
    records: list[dict],
    *,
    termination_reason: str = "force_converged",
) -> dict:
    return finalize_irc_branch(
        title=f"{direction.upper()} IRC",
        direction=direction,
        records=records,
        transition_state_energy_hartree=-1.0,
        termination_reason=termination_reason,
        iterations_attempted=max(len(records) - 1, 0),
        maximum_force_threshold_hartree_per_A=2.0e-3,
        rms_force_threshold_hartree_per_A=5.0e-4,
    )


def _converged_path() -> tuple[dict, dict, dict, dict]:
    forward = _branch(
        "forward",
        [_record(1.0, -1.2), _record(2.0, -2.0)],
    )
    backward = _branch(
        "backward",
        [_record(-1.0, -1.1), _record(-2.0, -1.5)],
    )
    transition_state = make_irc_record(
        energy_hartree=-1.0,
        forces_hartree_per_A=np.zeros((1, 3)),
        positions_angstrom=np.zeros((1, 3)),
        point_kind="transition_state",
    )
    summary = assemble_irc_path(
        method_label="TEST",
        forward=forward,
        backward=backward,
        transition_state_record=transition_state,
    )
    return forward, backward, transition_state, summary


def test_branch_convergence_requires_both_force_thresholds():
    converged = _branch("forward", [_record(1.0, -1.2)])
    assert converged["status"]["converged"] is True
    assert converged["status"]["termination_reason"] == "force_converged"

    not_converged = _branch(
        "forward",
        [_record(1.0, -1.2, maximum_force=3.0e-3)],
        termination_reason="maximum_steps",
    )
    assert not_converged["status"]["converged"] is False
    assert not_converged["status"]["force_criteria_satisfied"] is False
    assert not_converged["status"]["termination_reason"] == "maximum_steps"
    assert not_converged["status"]["final_maximum_force_hartree_per_A"] == (
        pytest.approx(3.0e-3)
    )


def test_non_force_stop_reason_is_preserved_even_if_final_forces_pass():
    stopped = _branch(
        "forward",
        [_record(1.0, -1.2)],
        termination_reason="step_too_small",
    )

    assert stopped["status"]["force_criteria_satisfied"] is True
    assert stopped["status"]["converged"] is False
    assert stopped["status"]["termination_reason"] == "step_too_small"


def test_path_contains_the_exact_transition_state_once_in_endpoint_order():
    forward, backward, transition_state, summary = _converged_path()

    assert summary["converged"] is True
    assert summary["first_direction"] == "forward"
    assert summary["ts_index"] == 3
    assert [record["x"][0, 0] for record in summary["records"]] == [
        2.0,
        1.0,
        0.0,
        -1.0,
        -2.0,
    ]
    ts_record = summary["records"][summary["ts_index"] - 1]
    assert ts_record is summary["transition_state_record"]
    assert ts_record["E"] == pytest.approx(transition_state["E"])
    assert ts_record["x"] == pytest.approx(transition_state["x"])
    assert ts_record["forces_hartree_per_A"] == pytest.approx(np.zeros((1, 3)))
    assert ts_record["forces_hartree_per_A"].flags.writeable is False
    assert (
        sum(record["point_kind"] == "transition_state" for record in summary["records"])
        == 1
    )
    assert summary["endpoint_minima_verified"] is False
    assert summary["chemical_connectivity_verified"] is False
    assert summary["step_size_convergence_verified"] is False
    assert summary["maximum_path_energy_excess_hartree"] < 0.0
    assert forward["status"]["converged"]
    assert backward["status"]["converged"]


def test_path_rejects_a_point_above_the_validated_transition_state():
    forward = _branch("forward", [_record(1.0, -0.9)])
    backward = _branch("backward", [_record(-1.0, -1.1)])
    transition_state = make_irc_record(
        energy_hartree=-1.0,
        forces_hartree_per_A=np.zeros((1, 3)),
        positions_angstrom=np.zeros((1, 3)),
        point_kind="transition_state",
    )

    with pytest.raises(ValueError, match="above the validated transition state"):
        assemble_irc_path(
            method_label="TEST",
            forward=forward,
            backward=backward,
            transition_state_record=transition_state,
        )


def test_path_energy_tolerance_must_be_explicit_for_float32_quantization():
    transition_state_energy = float(np.float32(-100.0))
    one_float32_ulp_up = float(
        np.nextafter(
            np.float32(transition_state_energy),
            np.float32(np.inf),
        )
    )
    forward = _branch(
        "forward",
        [_record(1.0, one_float32_ulp_up)],
    )
    backward = _branch(
        "backward",
        [_record(-1.0, transition_state_energy - 0.1)],
    )
    forward["E_ts"] = transition_state_energy
    backward["E_ts"] = transition_state_energy
    transition_state = make_irc_record(
        energy_hartree=transition_state_energy,
        forces_hartree_per_A=np.zeros((1, 3)),
        positions_angstrom=np.zeros((1, 3)),
        point_kind="transition_state",
    )

    with pytest.raises(ValueError, match="above the validated transition state"):
        assemble_irc_path(
            method_label="TEST",
            forward=forward,
            backward=backward,
            transition_state_record=transition_state,
        )

    explicit_tolerance = 2.0 * (one_float32_ulp_up - transition_state_energy)
    summary = assemble_irc_path(
        method_label="TEST",
        forward=forward,
        backward=backward,
        transition_state_record=transition_state,
        path_energy_tolerance_hartree=explicit_tolerance,
    )

    assert summary["maximum_path_energy_excess_hartree"] > 0.0
    assert summary["transition_state_is_strict_path_maximum"] is False
    assert summary["transition_state_is_path_maximum_within_tolerance"] is True
    assert summary["path_energy_tolerance_hartree"] == pytest.approx(
        explicit_tolerance
    )


def test_energy_degenerate_displaced_image_is_not_a_strict_ts_maximum():
    forward = _branch("forward", [_record(1.0, -1.0)])
    backward = _branch("backward", [_record(-1.0, -1.1)])
    transition_state = make_irc_record(
        energy_hartree=-1.0,
        forces_hartree_per_A=np.zeros((1, 3)),
        positions_angstrom=np.zeros((1, 3)),
        point_kind="transition_state",
    )

    summary = assemble_irc_path(
        method_label="TEST",
        forward=forward,
        backward=backward,
        transition_state_record=transition_state,
    )

    assert summary["maximum_path_energy_excess_hartree"] == pytest.approx(0.0)
    assert summary["transition_state_is_strict_path_maximum"] is False
    assert summary["transition_state_is_path_maximum_within_tolerance"] is True


def test_rendered_summary_persists_path_energy_admission_provenance():
    _, _, _, summary = _converged_path()

    rendered = "".join(render_irc_path_summary(summary))

    assert "Path-energy admission" in rendered
    assert "tolerance=1e-07 Eh" in rendered
    assert "maximum displaced-image excess=" in rendered
    assert "strict TS maximum=True" in rendered
    assert "TS maximum within tolerance=True" in rendered


def test_path_order_does_not_flip_when_backward_endpoint_is_lower():
    forward = _branch(
        "forward",
        [_record(1.0, -1.2), _record(2.0, -1.5)],
    )
    backward = _branch(
        "backward",
        [_record(-1.0, -1.3), _record(-2.0, -2.0)],
    )
    transition_state = make_irc_record(
        energy_hartree=-1.0,
        forces_hartree_per_A=np.zeros((1, 3)),
        positions_angstrom=np.zeros((1, 3)),
        point_kind="transition_state",
    )

    summary = assemble_irc_path(
        method_label="TEST",
        forward=forward,
        backward=backward,
        transition_state_record=transition_state,
    )

    assert summary["first_direction"] == "forward"
    assert [record["x"][0, 0] for record in summary["records"]] == [
        2.0,
        1.0,
        0.0,
        -1.0,
        -2.0,
    ]
    assert summary["E_ref"] == pytest.approx(-2.0)


def test_path_rejects_a_branch_bound_to_a_different_transition_state():
    forward, backward, transition_state, _ = _converged_path()
    forward["E_ts"] = -0.9

    with pytest.raises(ValueError, match="does not share"):
        assemble_irc_path(
            method_label="TEST",
            forward=forward,
            backward=backward,
            transition_state_record=transition_state,
        )


def test_shared_defaults_are_explicit_cartesian_legacy_units():
    params = IRCPathParams()

    assert params.max_steps == DEFAULT_IRC_MAX_STEPS == 50
    assert params.f_max_th == pytest.approx(
        DEFAULT_IRC_MAX_FORCE_HARTREE_PER_ANGSTROM
    )
    assert params.f_rms_th == pytest.approx(
        DEFAULT_IRC_RMS_FORCE_HARTREE_PER_ANGSTROM
    )


def test_nonconverged_endpoint_is_fail_closed_unless_explicitly_retained():
    forward, backward, _, summary = _converged_path()
    forward = _branch(
        "forward",
        [
            _record(
                1.0,
                -1.2,
                maximum_force=3.0e-3,
                rms_force=7.0e-4,
            )
        ],
        termination_reason="maximum_steps",
    )
    summary = assemble_irc_path(
        method_label="TEST",
        forward=forward,
        backward=backward,
        transition_state_record=summary["transition_state_record"],
    )

    assert summary["converged"] is False
    enforce_irc_path_admission(summary, require_converged_endpoints=False)
    with pytest.raises(RuntimeError, match="forward endpoint did not converge"):
        enforce_irc_path_admission(summary, require_converged_endpoints=True)


def test_trajectory_writer_emits_ts_to_endpoint_sides_and_endpoint_to_endpoint_full(
    tmp_path,
):
    forward, backward, _, summary = _converged_path()
    output = tmp_path / "run.out"
    paths = write_irc_trajectories(
        template_atoms=Atoms("H", positions=[[9.0, 0.0, 0.0]]),
        output=str(output),
        forward=forward,
        backward=backward,
        summary=summary,
    )

    expected = {
        "forward": [0.0, 1.0, 2.0],
        "backward": [0.0, -1.0, -2.0],
        "full": [2.0, 1.0, 0.0, -1.0, -2.0],
    }
    for name, coordinates in expected.items():
        path = Path(paths[name])
        frames = read(path, ":")
        assert [frame.positions[0, 0] for frame in frames] == pytest.approx(coordinates)
        assert [int(frame.info["image"]) for frame in frames] == list(
            range(len(frames))
        )
        assert all("energy_hartree" in frame.info for frame in frames)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (IRCPathParams(max_steps=0), "max_steps"),
        (IRCPathParams(step_length_bohr=0.0), "step_length_bohr"),
        (IRCPathParams(f_max_th=0.0), "f_max_th"),
        (IRCPathParams(f_rms_th=-1.0), "f_rms_th"),
        (
            IRCPathParams(path_energy_tolerance_hartree=0.0),
            "path_energy_tolerance_hartree",
        ),
        (
            IRCPathParams(require_converged_endpoints=1),
            "require_converged_endpoints",
        ),
    ],
)
def test_shared_path_parameters_fail_early(params, message):
    with pytest.raises(ValueError, match=message):
        validate_irc_path_params(params)


def test_dispatcher_returns_irc_result_without_overwriting_path_thresholds(
    tmp_path,
    monkeypatch,
):
    import maple.function.dispatcher.irc as irc_package

    expected = {"summary": {"converged": True}}
    observed = {}

    class _FakeIRC:
        def __init__(self, *, output, atoms, method, params):
            observed["params"] = dict(params)
            observed["method"] = method

        def run(self):
            return expected

    monkeypatch.setattr(irc_package, "IRC", _FakeIRC)
    command = SimpleNamespace(
        params={
            "method": "gs",
            "f_max_th": 2.0e-3,
            "f_rms_th": 5.0e-4,
        }
    )

    result = Dispatcher()._dispatch(
        command,
        "irc",
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        str(tmp_path / "irc.out"),
    )

    assert result is expected
    assert observed["method"] == "gs"
    assert observed["params"]["f_max_th"] == pytest.approx(2.0e-3)
    assert observed["params"]["f_rms_th"] == pytest.approx(5.0e-4)


def test_path_rejects_a_branch_whose_first_record_duplicates_the_ts():
    forward, backward, transition_state, _ = _converged_path()
    forward["records"][0]["x"] = np.array(transition_state["x"], copy=True)

    with pytest.raises(ValueError, match="not numerically distinct"):
        assemble_irc_path(
            method_label="TEST",
            forward=forward,
            backward=backward,
            transition_state_record=transition_state,
        )


def test_path_rejects_a_later_record_that_duplicates_the_ts():
    forward, backward, transition_state, _ = _converged_path()
    forward["records"].append(
        {
            "E": -1.1,
            "maxG": 1.0e-3,
            "rmsG": 2.0e-4,
            "x": np.array(transition_state["x"], copy=True),
            "point_kind": "path",
        }
    )

    with pytest.raises(ValueError, match="path point 2"):
        assemble_irc_path(
            method_label="TEST",
            forward=forward,
            backward=backward,
            transition_state_record=transition_state,
        )
