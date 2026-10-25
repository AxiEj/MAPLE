"""Static exported programs isolate signatures without Dynamo cache bypasses."""

import gc
import weakref

import pytest
import torch

from maple.function.calculator.consolv._compiled import ConSolvCompiledGraph


class _Potential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([1.0, 2.0, 3.0]))

    def _energy_graph(self, species, positions, solvent_index, sender, receiver):
        return (positions.square() * self.weight).sum() + solvent_index * 0.01


def _args(n=2, solvent=19, grad=False):
    positions = (
        torch.arange(n * 3, dtype=torch.float32).reshape(n, 3).requires_grad_(grad)
    )
    return (
        torch.zeros(n, dtype=torch.long),
        positions,
        solvent,
        torch.arange(n, dtype=torch.long),
        torch.arange(n, dtype=torch.long),
    )


@pytest.fixture
def fx_backend(monkeypatch):
    from torch._dynamo.backends import registry

    calls = []

    def lookup(name):
        if callable(name):
            return name
        assert name == "inductor"

        def backend(graph, inputs, **options):
            calls.append(options)
            return graph

        return backend

    monkeypatch.setattr(registry, "lookup_backend", lookup)
    return calls


def test_two_owned_ad_variants_preserve_grad_and_parent_context(fx_backend):
    potential = _Potential()
    owner = ConSolvCompiledGraph(potential)
    plain = _args()
    owner(*plain)
    args = _args(grad=True)
    with torch.no_grad():
        result = owner(*args)
    assert torch.equal(
        torch.autograd.grad(result, args[1])[0], 2 * args[1] * potential.weight
    )
    owner(*_args(grad=True))
    assert set(owner._variants) == {False, True}
    assert len(fx_backend) == 2
    assert all(state["compilations"] == 1 for _, state in owner._variants.values())


def test_owner_rejects_changed_signature_before_export(fx_backend):
    potential = _Potential()
    owner = ConSolvCompiledGraph(potential)
    owner(*_args())
    for args in (_args(3), _args(solvent=45)):
        with pytest.raises(ValueError, match="construct a new calculator"):
            owner(*args)
    assert len(fx_backend) == 1


def test_index_dtype_and_layout_are_explicit():
    args = list(_args())
    args[0] = args[0].float()
    with pytest.raises(ValueError, match="int64"):
        ConSolvCompiledGraph._input_signature(*args)
    args = list(_args())
    args[1] = torch.ones((3, 2)).T
    with pytest.raises(ValueError, match="contiguous"):
        ConSolvCompiledGraph._input_signature(*args)


def test_exported_owner_does_not_retain_original_potential(fx_backend):
    potential = _Potential()
    reference = weakref.ref(potential)
    buffer = weakref.ref(potential.weight)
    owner = ConSolvCompiledGraph(potential)
    owner(*_args())
    del potential
    gc.collect()
    assert reference() is None
    del owner
    gc.collect()
    assert buffer() is None


def test_export_does_not_change_source_buffers(fx_backend):
    potential = _Potential()
    before = {
        name: (value.clone(), value._version)
        for name, value in potential.named_buffers()
    }
    owner = ConSolvCompiledGraph(potential)
    owner(*_args())
    for name, value in potential.named_buffers():
        assert torch.equal(value, before[name][0])
        assert value._version == before[name][1]
    assert owner.export_records[0]["owned_state_tensors"] == 0
    assert owner.export_records[0]["tensor_constants"] == 0


def test_first_input_must_match_potential_dtype(fx_backend):
    potential = _Potential()
    owner = ConSolvCompiledGraph(potential)
    args = list(_args())
    args[1] = args[1].double()
    with pytest.raises(ValueError, match="potential device and dtype"):
        owner(*args)
    assert not fx_backend


def test_six_shapes_two_solvents_do_not_change_global_cache_limits(fx_backend):
    import torch._dynamo.config as config

    limits = (config.recompile_limit, config.accumulated_recompile_limit)
    for n in range(1, 7):
        for solvent in (19, 45):
            potential = _Potential()
            owner = ConSolvCompiledGraph(potential)
            owner(*_args(n, solvent))
            args = _args(n, solvent, grad=True)
            result = owner(*args)
            assert torch.isfinite(torch.autograd.grad(result, args[1])[0]).all()
            assert len(owner._variants) == 2
            del owner, potential, result, args
            gc.collect()
    assert (config.recompile_limit, config.accumulated_recompile_limit) == limits
    assert len(fx_backend) == 24
