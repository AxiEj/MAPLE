from types import SimpleNamespace

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
