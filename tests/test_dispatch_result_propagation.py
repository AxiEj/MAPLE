from types import SimpleNamespace

import pytest
from ase import Atoms

from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.ts.ts import TransitionState
from maple.function.engine import engine


def test_dispatcher_returns_transition_state_result(monkeypatch, tmp_path):
    marker = object()
    monkeypatch.setattr(TransitionState, "run", lambda self: marker)
    control = SimpleNamespace(params={"method": "prfo"})
    result = Dispatcher()(control, "ts", Atoms("H"), str(tmp_path / "job.out"))
    assert result is marker


def test_engine_exposes_dispatch_result_as_last_result(monkeypatch, tmp_path):
    instance = engine()
    marker = object()
    atoms = Atoms("H")
    output = str(tmp_path / "job.out")

    def read_input(_input, _output):
        instance.atoms = atoms
        instance.output = output
        instance.device = "cpu"
        instance.model = "fake"
        instance.jobtype = "ts"
        instance.commandcontrol = {"method": "prfo"}
        instance.extra = {}

    monkeypatch.setattr(instance, "_input_reader", read_input)
    monkeypatch.setattr(instance, "_mlp_initiator", lambda *_args: setattr(instance, "calulator", object()))
    monkeypatch.setattr(instance, "_jobtype_dispatcher", lambda *_args, **_kwargs: marker)
    result = instance("unused.inp", output)
    assert result is marker
    assert instance.last_result is marker


def test_engine_clears_last_result_before_a_failed_new_attempt(monkeypatch, tmp_path):
    instance = engine()
    prior = object()
    instance.last_result = prior

    def fail_input(_input, _output):
        raise ValueError("new input failed")

    monkeypatch.setattr(instance, "_input_reader", fail_input)
    with pytest.raises(ValueError, match="new input failed"):
        instance("failed.inp", str(tmp_path / "failed.out"))
    assert instance.last_result is None


def test_transition_state_retains_autoneb_result(monkeypatch, tmp_path):
    from maple.function.dispatcher.ts import algorithm

    marker = object()

    class FakeAutoNEB:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return marker

    monkeypatch.setattr(algorithm, "AutoNEB", FakeAutoNEB)
    transition = TransitionState(
        str(tmp_path / "autoneb.out"),
        [Atoms("H"), Atoms("H", positions=[[1.0, 0.0, 0.0]])],
        {"method": "autoneb", "n_max": 3, "climb": False},
        method="autoneb",
    )

    assert transition.run() is marker
    assert transition.last_result is marker
