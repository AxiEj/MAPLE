"""Dedicated 2025 boundary tests; native execution is explicitly opt-in."""
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

torch = pytest.importorskip("torch")
from maple.function.calculator import calculator_base as base
from maple.function.calculator.analytic_protocol import CheckpointFileIdentity
from maple.function.calculator.aimnet import _aimnet2025_calculator as adapter
from maple.function.calculator.aimnet import _aimnet2025_identity as identity


def _caller_precision():
    return {"cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "grad_enabled": bool(torch.is_grad_enabled())}


def _packed_graph():
    nb = np.array([[1, 2, 3, 3], [0, 2, 3, 3], [0, 1, 3, 3], [3, 3, 3, 3]], dtype=np.int32)
    return {"nbmat": nb, "nbmat_lr": nb.copy(),
            "numbers": np.array([1, 6, 8, 0], dtype=np.int32),
            "mol_idx": np.array([0, 0, 0, 1], dtype=np.int32)}


def test_canonical_graph_preserves_row_multisets_and_raw_packing_separately():
    from maple.function.calculator.analytic_protocol import named_array_sha256
    graph = _packed_graph()
    original = deepcopy(graph)
    permuted = {key: value[:, [2, 0, 3, 1]].copy() if key.startswith("nbmat") else value.copy()
                for key, value in graph.items()}
    assert named_array_sha256(graph) != named_array_sha256(permuted)
    for value in graph.values(): value.setflags(write=False)
    assert identity.canonical_neighbor_graph_sha256(graph) == identity.canonical_neighbor_graph_sha256(permuted)
    assert identity.NEIGHBOR_GRAPH_SCHEME == "row-multiset-with-padding-v1"
    for key in graph:
        np.testing.assert_array_equal(graph[key], original[key])
        assert graph[key].dtype == original[key].dtype and graph[key].shape == original[key].shape


@pytest.mark.parametrize("mutation", ["edge", "missing_key", "missing_column", "duplicate", "sentinel",
                                      "dummy_row", "shape", "dtype", "numbers_order", "mol_idx_order", "row_order"])
def test_canonical_graph_does_not_hide_connectivity_or_layout_changes(mutation):
    graph = _packed_graph()
    changed = deepcopy(graph)
    if mutation == "edge": changed["nbmat"][0, 0] = 0
    elif mutation == "missing_key": changed.pop("nbmat_lr")
    elif mutation == "missing_column": changed["nbmat"] = changed["nbmat"][:, :-1]
    elif mutation == "duplicate": changed["nbmat"][0, 1] = 1
    elif mutation == "sentinel": changed["nbmat"][0, -1] = -1
    elif mutation == "dummy_row": changed["nbmat"] = changed["nbmat"][:-1]
    elif mutation == "shape": changed["nbmat"] = changed["nbmat"].reshape(2, 8)
    elif mutation == "dtype": changed["nbmat"] = changed["nbmat"].astype(np.int64)
    elif mutation == "numbers_order": changed["numbers"] = changed["numbers"][::-1].copy()
    elif mutation == "mol_idx_order": changed["mol_idx"] = changed["mol_idx"][::-1].copy()
    elif mutation == "row_order": changed["nbmat"] = changed["nbmat"][[1, 0, 2, 3]]
    assert identity.canonical_neighbor_graph_sha256(graph) != identity.canonical_neighbor_graph_sha256(changed)


@pytest.mark.parametrize("bad", [np.zeros((2, 2), dtype=float), np.zeros((2, 2), dtype=bool),
                                np.zeros(4, dtype=np.int32), np.zeros((1, 2, 2), dtype=np.int32)])
def test_canonical_graph_rejects_unsupported_neighbor_layouts(bad):
    with pytest.raises((TypeError, ValueError), match="integer|2D|layout"):
        identity.canonical_neighbor_graph_sha256({"nbmat": bad})


@pytest.mark.parametrize("bad", [{}, {"numbers": np.array([1])}])
def test_canonical_graph_requires_neighbor_matrices(bad):
    with pytest.raises(ValueError): identity.canonical_neighbor_graph_sha256(bad)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.outputs = torch.nn.ModuleDict({
            "atomic_shift": torch.nn.ModuleDict({"shifts": torch.nn.Embedding(2, 1, dtype=torch.float64)}),
            "srcoulomb": torch.nn.Module(),
        })
        self.outputs["atomic_shift"]["shifts"].weight.requires_grad_(False)
        sr = self.outputs["srcoulomb"]
        sr.register_buffer("rc", torch.tensor(4.599999904632568))
        sr.envelope, sr.key_in, sr.key_out = "exp", "charges", "energy"
        sr._factor = 7.199822675975274
        self._metadata = deepcopy(identity.MODEL_METADATA)
        self._required_keys = ["coord", "numbers", "charge"]
        self._required_keys_dtype = [torch.float32, torch.int64, torch.float32]
        self.eval()

    def forward(self, data):
        # A complete scalar surrogate with FP32 coordinate response + FP64 sum.
        x = data["coord"]
        data["a"] = x * self.weight
        data["charges"] = x[:, 0] * 0.0
        data["energy"] = ((x * x).sum() * self.weight / 2).double() + 3.1234567890123
        return data


class FakeBackend:
    keys_in = {"coord": torch.float32, "numbers": torch.int32, "charge": torch.float32}
    def __init__(self, model, *, device, train, compile_model):
        assert train is False and compile_model is False
        self.device = device
        self.model = FakeModel().to(device)
        self.external_coulomb = torch.nn.Module()
        c = self.external_coulomb
        c.method, c.subtract_sr, c.envelope = "simple", False, "exp"
        c.key_in, c.key_out, c._factor = "charges", "energy", 7.199822675975274
        c.register_buffer("rc", torch.tensor(4.599999904632568, device=device))
        self.external_dftd3 = torch.nn.Module()
        for key, value in dict(s6=1., s8=1.5, a1=.37, a2=4.1,
                               smoothing_on=12., smoothing_off=15., key_out="energy").items():
            setattr(self.external_dftd3, key, value)
        self._train, self._was_compiled, self.lr = False, False, True
        self.cutoff, self.cutoff_lr = 5., float("inf")
        self._coulomb_method, self._coulomb_cutoff, self._dftd3_cutoff = "simple", float("inf"), 15.
        self.nb_threshold = 120
        self._nblist = SimpleNamespace(cutoff=5.)
        self._nblist_lr = SimpleNamespace(cutoff=15.)
        self._nblist_dftd3 = self._nblist_coulomb = None
        self.calls = []

    @property
    def metadata(self):
        return self.model._metadata

    def __call__(self, data, *, forces, hessian):
        assert data["coord"].is_leaf and data["coord"].requires_grad
        assert data["coord"].dtype is torch.float32
        assert torch.is_grad_enabled() and forces is False and hessian is False
        self.calls.append((forces, hessian))
        data["nbmat"] = torch.tensor([[1], [0]], dtype=torch.int32)
        data["mol_idx"] = torch.zeros(len(data["coord"]), dtype=torch.int32)
        return self.model(data)


@pytest.fixture
def calc(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"only fake checkpoint")
    monkeypatch.setattr(adapter, "_resolve_checkpoint", lambda _: CheckpointFileIdentity.bind(path))
    monkeypatch.setattr(adapter, "_bind_external_runtime", lambda: ({"packages": {}, "source_sha256": {}}, ()))
    monkeypatch.setattr(adapter, "_require_external_sources", lambda files: None, raising=False)
    monkeypatch.setattr(adapter, "_import_native_backend", lambda: FakeBackend)
    monkeypatch.setattr(adapter, "_verify_checkpoint_state", lambda *args: None)
    return adapter.AIMNet2025Calculator("cpu")


@pytest.fixture
def atoms():
    return Atoms("H2", positions=[[.1, -.2, .3], [.85, .1, -.1]])


def forbid(*args, **kwargs):
    raise AssertionError("forbidden derivative path")


def test_aliases_and_factory_options_are_new_only():
    cls = adapter.AIMNet2025Calculator
    assert cls.MODEL_NAMES == ("aimnet2025", "aimnet2-2025", "aimnet2_2025")
    assert cls.CHECKPOINT_FILENAME is None and cls.REQUIRES_LOCAL_MODEL_FILE is False
    assert cls.ANALYTIC_IMPLICIT_HESSIAN_DEVICES == ("cpu", "cuda")
    for alias in cls.MODEL_NAMES:
        assert base.get_registered_calculator(alias) is cls
    assert "aimnet2" not in cls.MODEL_NAMES
    assert cls.build_kwargs_from_options("aimnet2025", {"dtype": "float32", "d4": "false", "model_path": "exact.pt"}) == {
        "dtype": "float32", "d4": False, "hessian": "analytic", "model_path": "exact.pt"}


def test_energy_is_lazy_and_units_and_force_sign_are_exact(calc, atoms, monkeypatch):
    monkeypatch.setattr(calc, "get_hessian", forbid)
    original_grad = torch.autograd.grad
    monkeypatch.setattr(torch.autograd, "grad", forbid)
    calc.calculate(atoms, ["energy"])
    expected_e = ((torch.tensor(atoms.positions, dtype=torch.float32)**2).sum().double() + 3.1234567890123).item()
    assert calc.results["energy"] == expected_e * base.EV2HARTREE
    assert "forces" not in calc.results and "hessian" not in calc.results
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    calc.calculate(atoms, ["forces"])
    expected_f = -2 * np.asarray(atoms.positions, dtype=np.float32).astype(np.float64) * base.EV2HARTREE
    np.testing.assert_array_equal(calc.results["forces"], expected_f)
    calc.calculate(atoms, ["energy"])
    assert "forces" not in calc.results
    assert all(call == (False, False) for call in calc.backend.calls)


def test_raw_gas_hessian_and_direct_hvp_do_not_use_dense_h_or_fd(calc, atoms, monkeypatch):
    h = calc.get_hessian(atoms)
    np.testing.assert_allclose(h, np.eye(6) * 2 * base.EV2HARTREE, atol=1e-8)
    monkeypatch.setattr(calc, "get_hessian", forbid)
    monkeypatch.setattr(calc, "_analytic_hessian", forbid)
    monkeypatch.setattr(base, "hessian_via_double_autograd", forbid)
    monkeypatch.setattr(base, "numerical_hessian_from_atoms", forbid)
    monkeypatch.setattr(torch.autograd.functional, "hessian", forbid)
    v = np.arange(6, dtype=float) / 7
    hv, f, e = calc.get_hvp(atoms, v)
    assert hv.shape == f.shape == (6,) and e.ndim == 0
    assert hv.dtype is f.dtype is e.dtype is torch.float64
    np.testing.assert_allclose(hv.numpy(), h @ v, atol=1e-8)
    calc.calculate(atoms, ["forces"])
    assert e.item() == calc.results["energy"]
    np.testing.assert_array_equal(f.numpy(), calc.results["forces"].reshape(-1))


class SyntheticOBC64:
    dtype = torch.float64
    supported_properties = {"energy", "forces", "hessian"}
    last_derivative_provenance = {"provider": "synthetic-obc64"}
    def __init__(self):
        self.tiny = 1.234567891234e-10
        self.calls = []
    def evaluate(self, atoms, *, need_forces, calculator):
        self.calls.append(("ef", need_forces))
        return SimpleNamespace(energy_hartree=self.tiny,
            forces_hartree_per_angstrom=np.full((len(atoms), 3), self.tiny) if need_forces else None,
            components_hartree={}, provenance={})
    def get_hessian(self, atoms):
        self.calls.append(("h", True))
        return np.eye(3 * len(atoms)) * self.tiny
    def get_directional_derivatives(self, atoms, n):
        self.calls.append(("hv", True))
        return SimpleNamespace(energy_hartree=self.tiny,
            forces_hartree_per_angstrom=np.full((len(atoms), 3), self.tiny),
            hvp_hartree_per_angstrom2=np.asarray(n) * self.tiny, provenance={"dtype": "float64"})


def test_composed_h_remains_blocked_and_preparation_never_admits(calc, atoms):
    calc.solvent_correction = SyntheticOBC64()
    calc.prepare_analytic_derivatives()
    assert calc.analytic_implicit_derivatives_admitted is False
    with pytest.raises(NotImplementedError):
        calc.get_hessian(atoms)
    with pytest.raises(NotImplementedError):
        calc.get_hvp(atoms, np.ones(6))


def test_composition_promotes_gas_before_obc64_addition(calc, atoms, monkeypatch):
    # Isolated candidate simulation, not a shipped caller bypass.
    gas_h = calc.get_hessian(atoms)
    gas_hv, gas_f, gas_e = calc.get_hvp(atoms, np.ones(6))
    correction = SyntheticOBC64()
    calc.solvent_correction = correction
    monkeypatch.setattr(calc, "analytic_implicit_derivatives_admitted", True)
    calc.calculate(atoms, ["energy", "forces", "hessian"])
    assert correction.calls == [("h", True), ("ef", True)]
    np.testing.assert_array_equal(calc.results["hessian"], gas_h + np.eye(6) * correction.tiny)
    np.testing.assert_array_equal(calc.results["forces"].reshape(-1), gas_f.numpy() + correction.tiny)
    hv, f, e = calc.get_hvp(atoms, np.ones(6))
    np.testing.assert_array_equal(hv.numpy(), gas_hv.numpy() + correction.tiny)
    np.testing.assert_array_equal(f.numpy(), gas_f.numpy() + correction.tiny)
    assert e.item() == gas_e.item() + correction.tiny
    assert calc.last_hvp_provenance["gas_ad_class"] == "N32"
    assert calc.last_hvp_provenance["solvent_dtype"] == "float64"


@pytest.mark.parametrize("mutation", ["dtype", "train", "d3", "coulomb", "cutoff", "weights", "checkpoint",
                                      "backend_device", "device", "sr", "state_replace", "train_flag", "metadata"])
def test_drift_fails_closed_even_for_ase_cached_energy(calc, atoms, mutation):
    atoms.calc = calc
    atoms.get_potential_energy()
    if mutation == "dtype": calc.model.double()
    elif mutation == "train": calc.model.train()
    elif mutation == "d3": calc.backend.external_dftd3.s8 = 0.
    elif mutation == "coulomb": calc.backend.external_coulomb.method = "dsf"
    elif mutation == "cutoff": calc.backend._nblist.cutoff = 4.
    elif mutation == "weights":
        with torch.no_grad(): calc.model.weight.add_(.1)
    elif mutation == "checkpoint":
        with open(calc._checkpoint.path, "ab") as handle: handle.write(b"drift")
    elif mutation == "backend_device": calc.backend.device = "cuda"
    elif mutation == "device": calc.device = torch.device("cuda:0")
    elif mutation == "sr": calc.model.outputs["srcoulomb"].envelope = "cosine"
    elif mutation == "state_replace": calc.model.weight = torch.nn.Parameter(torch.tensor(2.))
    elif mutation == "train_flag": calc.backend._train = True
    elif mutation == "metadata": calc.model._metadata["d3_params"]["s8"] = 7.
    with pytest.raises((RuntimeError, ValueError), match="identity|changed|drift|profile"):
        atoms.get_potential_energy()
    assert calc.results == {}


def test_charge_mult_cache_and_point_quantization_are_honest(calc, atoms):
    atoms.calc = calc
    atoms.get_potential_energy()
    first = calc.last_analytic_point_identity
    cell = calc.analytic_cell_identity
    cell["model"] = "spoof"
    assert calc.analytic_cell_identity["model"] == "aimnet2025"
    atoms.positions[0, 0] += 1e-12
    atoms.get_potential_energy()
    second = calc.last_analytic_point_identity
    assert second["response_settings"]["autocast_enabled"] is False
    assert second["response_settings"]["grad_enabled"] is True
    assert second["response_settings"]["device_type"] == "cpu"
    assert first["geometry_sha256"] != second["geometry_sha256"]
    assert first["native_geometry_sha256"] == second["native_geometry_sha256"]
    atoms.info["charge"] = 1
    atoms.get_potential_energy()
    assert calc.last_analytic_point_identity["charge"] == 1
    atoms.info["mult"] = 3
    with pytest.raises(ValueError, match="multiplicity"):
        atoms.get_potential_energy()


def test_domain_and_context_guards(calc, atoms):
    for bad in [Atoms("He", positions=[[0,0,0]]), Atoms("H", positions=[[np.nan,0,0]])]:
        with pytest.raises(ValueError): calc.calculate(bad)
    atoms.pbc = True
    with pytest.raises(NotImplementedError, match="PBC"): calc.calculate(atoms)
    atoms.pbc = False
    with torch.no_grad():
        calc.calculate(atoms)
        assert torch.is_grad_enabled() is False
    with torch.inference_mode(), pytest.raises((RuntimeError, NotImplementedError), match="inference|differentiable"):
        calc.calculate(atoms)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(NotImplementedError, match="AMP|autocast"):
        calc.calculate(atoms)
    with pytest.raises(ValueError, match="direction"):
        calc.get_hvp(atoms, np.ones(5))


def test_scoped_precision_applies_ieee_and_restores_caller_flags_and_grad(calc, atoms, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with torch.no_grad():
        before = _caller_precision()
        calc.calculate(atoms, ["forces"])
        assert _caller_precision() == before
        point = calc.last_analytic_point_identity["response_settings"]
        assert point["cuda_matmul_allow_tf32"] is point["cudnn_allow_tf32"] is False
        assert point["grad_enabled"] is True
    record = calc.last_precision_execution_provenance
    assert record["caller_before"] == record["caller_after"] == before
    assert record["effective"] == dict(before, cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, grad_enabled=True)
    assert record["status"] == "success"


def test_precision_context_is_reentrant_and_restores_after_exceptions(calc, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with torch.no_grad():
        before = _caller_precision()
        with pytest.raises(RuntimeError, match="unit failure"):
            with adapter._native_precision_context(calc, "outer-test"):
                assert _caller_precision() == dict(before, cuda_matmul_allow_tf32=False, cudnn_allow_tf32=False, grad_enabled=True)
                with adapter._native_precision_context(calc, "inner-test"):
                    assert torch.is_grad_enabled()
                assert not torch.backends.cudnn.allow_tf32
                raise RuntimeError("unit failure")
        assert _caller_precision() == before
    assert calc.last_precision_execution_provenance["status"] == "error"
    record = calc.last_precision_execution_provenance
    assert record["caller_backend_selectors_before"] == record["caller_backend_selectors_after"]


def test_internal_tf32_toggle_is_not_hidden_by_nested_operation(calc, atoms, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    before = _caller_precision()
    original = calc.backend.__class__.__call__
    def corrupt_backend(backend, *args, **kwargs):
        result = original(backend, *args, **kwargs)
        torch.backends.cuda.matmul.allow_tf32 = True
        # A reentrant property must reject, not silently reset the corruption.
        calc.analytic_cell_identity
        return result
    monkeypatch.setattr(calc.backend.__class__, "__call__", corrupt_backend)
    with pytest.raises(NotImplementedError, match="TF32|precision"):
        calc.calculate(atoms, ["forces"])
    assert calc.results == {} and _caller_precision() == before


@pytest.mark.parametrize("flag", ["matmul", "cudnn"])
def test_internal_tf32_toggle_is_rejected_before_result_publication(calc, atoms, monkeypatch, flag):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    before = _caller_precision()
    original = calc.backend.__class__.__call__
    def corrupt_backend(backend, *args, **kwargs):
        result = original(backend, *args, **kwargs)
        target = torch.backends.cuda.matmul if flag == "matmul" else torch.backends.cudnn
        target.allow_tf32 = True
        return result
    monkeypatch.setattr(calc.backend.__class__, "__call__", corrupt_backend)
    with pytest.raises(NotImplementedError, match="TF32"):
        calc.calculate(atoms, ["forces"])
    assert calc.results == {} and _caller_precision() == before


@pytest.mark.parametrize("mode", ["amp", "inference"])
def test_unsupported_contexts_are_rejected_without_bypass_and_restore_flags(calc, atoms, monkeypatch, mode):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(calc.model, "forward", forbid)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if mode == "amp" else torch.inference_mode()
    with context:
        before = _caller_precision()
        with pytest.raises(NotImplementedError, match="AMP|autocast|inference"):
            calc.calculate(atoms)
        assert _caller_precision() == before


def test_constructor_error_restores_caller_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with torch.no_grad():
        before = _caller_precision()
        with pytest.raises(ValueError, match="float64"):
            adapter.AIMNet2025Calculator("cpu", dtype="float64")
        assert _caller_precision() == before


def test_scoped_precision_restores_public_backend_selectors_exactly(calc, atoms, monkeypatch):
    # Torch 2.12 legacy allow_tf32 setters lose inherited 'none' selectors,
    # even when their effective boolean values appear restored.
    backends = {"matmul": torch.backends.cuda.matmul, "cudnn": torch.backends.cudnn,
                "conv": torch.backends.cudnn.conv, "rnn": torch.backends.cudnn.rnn}
    for key, value in {"matmul": "none", "cudnn": "none", "conv": "tf32", "rnn": "tf32"}.items():
        monkeypatch.setattr(backends[key], "fp32_precision", value)
    before = {key: backend.fp32_precision for key, backend in backends.items()}
    calc.calculate(atoms, ["forces"])
    after = {key: backend.fp32_precision for key, backend in backends.items()}
    assert after == before
    record = calc.last_precision_execution_provenance
    assert record["caller_backend_selectors_before"] == record["caller_backend_selectors_after"] == before


def test_point_keeps_canonical_and_raw_hashes_without_changing_forward_graph(calc, atoms):
    from maple.function.calculator.analytic_protocol import named_array_sha256
    captures = []
    def capture(module, args):
        captures.append({key: value.detach().cpu().numpy().copy() for key, value in args[0].items()
                         if isinstance(value, torch.Tensor) and key.startswith(("nbmat", "mol_idx"))})
    hook = calc.model.register_forward_pre_hook(capture)
    try:
        calc.calculate(atoms, ["energy"])
    finally:
        hook.remove()
    point = calc.last_analytic_point_identity
    assert point["neighbor_graph_sha256"] == identity.canonical_neighbor_graph_sha256(captures[0])
    assert point["neighbor_packing_sha256"] == named_array_sha256(captures[0])
    assert point["neighbor_graph_scheme"] == identity.NEIGHBOR_GRAPH_SCHEME


def test_cell_identity_depends_on_effective_profile_not_caller_flags(calc, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    first = calc.analytic_cell_identity
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with torch.no_grad(): second = calc.runtime_profile_identity
    assert first == second
    assert first["precision_execution_policy"] == "model-scoped-ieee-no-tf32-v1"
    assert first["neighbor_graph_scheme"] == "row-multiset-with-padding-v1"


def test_cutoff_guard_uses_native_and_original_geometry(calc):
    atoms = Atoms("H2", positions=[[0,0,0], [4.999,0,0]])
    calc.calculate(atoms, ["energy"])
    assert calc.last_analytic_point_identity["minimum_cutoff_gap_angstrom"] < .003
    with pytest.raises(NotImplementedError, match="cutoff"):
        calc.get_hvp(atoms, np.ones(6))


def test_point_cutoff_refusals_do_not_revoke_an_unchanged_candidate_cell(calc, monkeypatch):
    # Isolated negative-control simulation, not a caller admission flag.
    calc.solvent_correction = SyntheticOBC64()
    monkeypatch.setattr(calc, "analytic_implicit_derivatives_admitted", True)
    atoms = Atoms("H2", positions=[[0, 0, 0], [4.999, 0, 0]])
    before = _caller_precision()
    for operation in (lambda: calc._analytic_hessian(atoms), lambda: calc.get_hessian(atoms),
                      lambda: calc.get_hvp(atoms, np.ones(6))):
        with pytest.raises(NotImplementedError, match="within native/original cutoff guard"):
            operation()
        assert calc.analytic_implicit_derivatives_admitted is True
        assert _caller_precision() == before


@pytest.mark.parametrize("kwargs,match", [({"dtype":"float64"}, "float64"), ({"d4":True}, "D4"),
                                           ({"model":"aimnet2"}, "model"), ({"dtype":"half"}, "dtype")])
def test_unsupported_options_fail_before_native_import(monkeypatch, kwargs, match):
    monkeypatch.setattr(adapter, "_import_native_backend", forbid)
    with pytest.raises(ValueError, match=match): adapter.AIMNet2025Calculator("cpu", **kwargs)


def test_python_missing_dependency_and_offline_wrong_hash_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter.sys, "version_info", (3, 10, 16))
    with pytest.raises(RuntimeError, match="3.11"):
        adapter.AIMNet2025Calculator("cpu")
    monkeypatch.undo()
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MAPLE_OFFLINE", "true")
    with pytest.raises(FileNotFoundError, match="MAPLE_OFFLINE"):
        adapter._resolve_checkpoint(None)
    path = tmp_path / identity.CHECKPOINT_FILENAME
    path.write_bytes(b"wrong member")
    with pytest.raises(ValueError, match="SHA256"):
        adapter._resolve_checkpoint(None)
    monkeypatch.setattr(adapter, "_import_native_backend", forbid)
    with pytest.raises(ValueError, match="SHA256"):
        adapter.AIMNet2025Calculator("cpu", model_path=path)


def test_external_identity_is_source_and_table_bound():
    assert identity.SOURCE_SHA256["dftd3_data.pt"] == "83d1d0ff69de90cb3ded49c80c8151786106f2aea7eb3425abe625e4b1909ce2"
    assert "models/base.py" in identity.SOURCE_SHA256
    assert all(not key.startswith("/") for key in identity.SOURCE_SHA256)


def test_release_source_inventory_covers_complete_python_dependency_closure():
    assert {"ops.py", "nbops.py", "modules/core.py", "config.py",
            "__init__.py", "calculators/model_registry.py"} <= identity.SOURCE_SHA256.keys()
    assert len(identity.SOURCE_SHA256) == 40
    assert sum(name.endswith(".py") for name in identity.SOURCE_SHA256) == 34


@pytest.fixture
def copied_source_package(monkeypatch, tmp_path):
    """Only fake package copies are changed; installed/cache sources are untouched."""
    import hashlib
    root = tmp_path / "aimnet"
    pins = {}
    for relative in identity.SOURCE_SHA256:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = (f"# fake official release fixture {relative}\n").encode()
        if relative == "ops.py":
            source += b"def cosine_cutoff(x):\n    return x * x\n"
        path.write_bytes(source)
        pins[relative] = hashlib.sha256(source).hexdigest()
    # The pre-fix map omits precisely this executing helper. Its source still
    # exists in the installation, so an incomplete allowlist must not bless it.
    ops = root / "ops.py"
    if not ops.exists():
        ops.write_text("def cosine_cutoff(x):\n    return x * x\n")
    monkeypatch.setattr(identity, "SOURCE_SHA256", pins)
    monkeypatch.setattr(identity.metadata, "version", lambda name: identity.PACKAGES[name])
    monkeypatch.setattr(identity.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda p: tmp_path / p))
    monkeypatch.setattr(identity, "_executing_aimnet_modules", lambda: (), raising=False)
    return root


def test_same_version_executing_math_helper_drift_is_rejected(copied_source_package):
    # Real original defect: unchanged package version/old pins, changed delegated
    # ops.cosine_cutoff implementation. Never alter the installed helper.
    (copied_source_package / "ops.py").write_text("def cosine_cutoff(x):\n    return 0.0\n")
    with pytest.raises((RuntimeError, ValueError), match="source.*identity|inventory"):
        identity.bind_external_runtime()


@pytest.mark.parametrize("mutation", ["added", "missing"])
def test_complete_source_inventory_rejects_added_or_missing_helpers(copied_source_package, mutation):
    if mutation == "added":
        (copied_source_package / "new_math.py").write_text("def wrong(x): return x\n")
    else:
        (copied_source_package / "ops.py").unlink()
    with pytest.raises((RuntimeError, ValueError), match="inventory"):
        identity.bind_external_runtime()


@pytest.mark.parametrize("mutation", ["drift", "added", "missing"])
def test_external_helper_changes_block_ase_cached_return(calc, atoms, copied_source_package, monkeypatch, mutation):
    _, calc._external_files = identity.bind_external_runtime()
    monkeypatch.setattr(adapter, "_require_external_sources", lambda files: identity.require_external_sources(files), raising=False)
    atoms.calc = calc
    atoms.get_potential_energy()
    monkeypatch.setattr(calc, "_state_digest", forbid)
    if mutation == "drift":
        (copied_source_package / "ops.py").write_text("def cosine_cutoff(x):\n    return 0.0\n")
    elif mutation == "added":
        nested = copied_source_package / "new_dispatch"
        nested.mkdir()
        (nested / "math.py").write_text("def wrong(x): return x\n")
    else:
        (copied_source_package / "ops.py").unlink()
    with pytest.raises((RuntimeError, ValueError), match="identity|inventory|readable|changed"):
        atoms.get_potential_energy()
    assert calc.results == {} and calc.analytic_implicit_derivatives_admitted is False


@pytest.mark.parametrize("mutation", ["foreign_origin", "wrong_module", "search_path", "no_origin"])
def test_executing_math_modules_must_resolve_to_bound_distribution(copied_source_package, monkeypatch, tmp_path, mutation):
    root = copied_source_package
    module = SimpleNamespace(__file__=str(root / "ops.py"),
                             __spec__=SimpleNamespace(origin=str(root / "ops.py")))
    if mutation == "foreign_origin":
        module.__file__ = module.__spec__.origin = str(tmp_path / "foreign" / "ops.py")
    elif mutation == "wrong_module":
        module.__file__ = module.__spec__.origin = str(root / "nbops.py")
    elif mutation == "search_path":
        module.__path__ = [str(root), str(tmp_path / "foreign")]
    else:
        module.__spec__ = None
    monkeypatch.setattr(identity, "_executing_aimnet_modules", lambda: (("aimnet.ops", module),))
    with pytest.raises(ValueError, match="executing.*identity"):
        identity.bind_external_runtime()


def test_new_symlink_dispatch_directory_cannot_escape_inventory(copied_source_package, tmp_path):
    dispatch = tmp_path / "foreign_dispatch"
    dispatch.mkdir()
    (dispatch / "math.py").write_text("def wrong(x): return x\n")
    (copied_source_package / "dispatch").symlink_to(dispatch, target_is_directory=True)
    with pytest.raises(ValueError, match="inventory"):
        identity.bind_external_runtime()


def test_live_source_stat_guards_do_not_hash_bytes(calc, atoms, copied_source_package, monkeypatch):
    _, files = identity.bind_external_runtime()
    calc._external_files = files
    monkeypatch.setattr(adapter, "_require_external_sources", identity.require_external_sources)
    monkeypatch.setattr(CheckpointFileIdentity, "bind", forbid)
    monkeypatch.setattr(calc, "_state_digest", forbid)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_potential_energy()
    atoms.get_forces()


def test_python310_core_import_is_lazy_and_syntax_is_supported():
    import ast
    from pathlib import Path
    for module in (adapter, identity):
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
    code = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','aimnet','warp','nvalchemiops'}:
            raise ImportError('optional import forbidden during core module import')
sys.meta_path.insert(0, BlockOptional())
from maple.function.calculator.aimnet._aimnet2025_calculator import AIMNet2025Calculator
assert 'torch' not in sys.modules and 'aimnet' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_missing_native_dependency_has_install_hint(monkeypatch):
    def missing(*args):
        raise importlib.metadata.PackageNotFoundError("aimnet")
    monkeypatch.setattr(identity.metadata, "version", missing)
    with pytest.raises(ImportError, match="aimnet==0.2.0"):
        identity.bind_external_runtime()


def test_wrong_package_version_fails_even_without_source_import(monkeypatch):
    monkeypatch.setattr(identity.metadata, "version", lambda name: "0.2.1")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        identity.bind_external_runtime()


@pytest.mark.parametrize("relative", ["models/base.py", "dftd3_data.pt"])
def test_same_version_wrong_source_or_table_fails(monkeypatch, tmp_path, relative):
    import hashlib
    path = tmp_path / "aimnet" / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong installed bytes")
    monkeypatch.setattr(identity, "SOURCE_SHA256", {relative: hashlib.sha256(b"right bytes").hexdigest()})
    monkeypatch.setattr(identity.metadata, "version", lambda name: identity.PACKAGES[name])
    monkeypatch.setattr(identity.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda p: tmp_path / p))
    with pytest.raises(RuntimeError, match="source identity mismatch"):
        identity.bind_external_runtime()


def test_preparation_fingerprints_state_but_force_calls_do_not(calc, atoms, monkeypatch):
    original = calc._state_digest
    calls = []
    def digest():
        calls.append(True)
        return original()
    monkeypatch.setattr(calc, "_state_digest", digest)
    calc.prepare_analytic_derivatives()
    assert calls == [True] and calc.analytic_implicit_derivatives_admitted is False
    calc.calculate(atoms, ["forces"])
    calc.calculate(atoms, ["forces"])
    assert calls == [True]
    # Raw .data writes evade Torch's mutation counter: preparation still hashes
    # all weights, rather than falsely treating a stat/version check as bytes.
    calc.model.weight.data.add_(1.)
    with pytest.raises(ValueError, match="state identity changed"):
        calc.prepare_analytic_derivatives()


def test_native_hessian_is_only_computed_when_requested(calc, atoms, monkeypatch):
    calls = []
    original = calc.get_hessian
    def hessian(atoms):
        calls.append(True)
        return original(atoms)
    monkeypatch.setattr(calc, "get_hessian", hessian)
    calc.calculate(atoms, ["forces"])
    assert not calls
    calc.calculate(atoms, ["hessian"])
    assert calls == [True] and "hessian" in calc.results and "forces" not in calc.results


def test_offline_false_still_never_downloads_or_searches(monkeypatch, tmp_path):
    monkeypatch.setenv("AIMNET_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MAPLE_OFFLINE", "false")
    with pytest.raises(FileNotFoundError, match="No automatic download"):
        adapter._resolve_checkpoint(None)
    assert list(tmp_path.iterdir()) == []


def test_fixed_solvent_charges_are_never_replaced_by_predicted_gas_charges(calc, atoms):
    atoms.set_initial_charges([-.25, .25])
    original = atoms.get_initial_charges().copy()
    calc.solvent_correction = SyntheticOBC64()
    calc.calculate(atoms, ["energy", "forces"])
    np.testing.assert_array_equal(atoms.get_initial_charges(), original)
    assert "charges" not in calc.results


def test_ase_cache_invalidates_when_shared_solvent_is_attached_or_removed(calc, atoms):
    atoms.calc = calc
    gas_energy = atoms.get_potential_energy()
    correction = SyntheticOBC64()
    calc.solvent_correction = correction
    assert atoms.get_potential_energy() == gas_energy + correction.tiny
    assert correction.calls == [("ef", False)]
    calc.solvent_correction = None
    assert atoms.get_potential_energy() == gas_energy


NATIVE_MODEL = os.environ.get("AIMNET_TEST_MODEL_PATH")
@pytest.mark.skipif(not NATIVE_MODEL, reason="set AIMNET_TEST_MODEL_PATH for default-caller native execution")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_default_caller_constructor_and_gas_operations(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    # Exercise permissive caller settings, including ordinary cuDNN=True.
    # The adapter, not a test's manual false setting, must establish IEEE.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    with torch.no_grad():
        before = _caller_precision()
        calc = adapter.AIMNet2025Calculator(device, model_path=NATIVE_MODEL)
        assert _caller_precision() == before
        atoms = molecule("H2O")
        calc.calculate(atoms, ["energy", "forces"])
        assert np.isfinite(calc.results["energy"]) and np.isfinite(calc.results["forces"]).all()
        assert _caller_precision() == before
        hv, force, energy = calc.get_hvp(atoms, np.ones(9))
        assert all(torch.isfinite(value).all() for value in (hv, force, energy))
        assert _caller_precision() == before
        settings = calc.last_analytic_point_identity["response_settings"]
        assert settings["cuda_matmul_allow_tf32"] is settings["cudnn_allow_tf32"] is False
        assert settings["grad_enabled"] is True


@pytest.mark.skipif(not NATIVE_MODEL, reason="set AIMNET_TEST_MODEL_PATH to opt into exact official native tests")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_complete_scalar_reference_hessian_and_direct_hvp(device, monkeypatch, request):
    import json
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    before = _caller_precision()
    calc = adapter.AIMNet2025Calculator(device, model_path=NATIVE_MODEL)
    assert _caller_precision() == before
    calc.prepare_analytic_derivatives()
    atoms = molecule("H2O")
    calc.calculate(atoms, ["energy", "forces"])
    # Independent upstream reference also runs explicitly in the same IEEE
    # profile; ambient caller defaults are never manually changed by the test.
    with adapter._native_precision_context(calc, "native-reference"):
        official = calc.backend({"coord": atoms.positions, "numbers": atoms.numbers, "charge":0.}, forces=True)
    assert _caller_precision() == before
    assert abs(calc.results["energy"] - official["energy"].item()*base.EV2HARTREE) < 2e-5
    np.testing.assert_allclose(calc.results["forces"], official["forces"].cpu().numpy()*base.EV2HARTREE, atol=2e-6, rtol=0)
    h = calc.get_hessian(atoms)
    np.testing.assert_allclose(h, h.T, atol=2e-6 * max(1, np.abs(h).max()), rtol=0)
    # Fixed diagnostic, not formal qualification or a tuned threshold.
    fd, diagnostic = base.numerical_hessian_from_atoms(calc, atoms, .001, return_diagnostics=True)
    assert np.isfinite(diagnostic["maximum_raw_asymmetry_hartree_per_angstrom2"])
    delta = fd - h
    assert np.isfinite(delta).all()
    request.node.user_properties.append(("gas_fd_diagnostic_not_qualification", json.dumps({
        "step_angstrom": .001, "max_hartree_per_angstrom2": float(np.max(np.abs(delta))),
        "rms_hartree_per_angstrom2": float(np.sqrt(np.mean(delta**2))),
        "raw_fd_diagnostics": diagnostic})))
    v = np.cos(np.arange(9)+.25); v /= np.linalg.norm(v)
    monkeypatch.setattr(calc, "get_hessian", forbid)
    monkeypatch.setattr(calc, "_analytic_hessian", forbid)
    monkeypatch.setattr(calc.backend, "calculate_hessian", forbid)
    monkeypatch.setattr(base, "numerical_hessian_from_atoms", forbid)
    hv, f, e = calc.get_hvp(atoms, v)
    np.testing.assert_allclose(hv.cpu().numpy(), h@v, atol=5e-6, rtol=5e-5)
    assert calc.inference_precision_provenance["effective_ad_class"] == "N32"
    assert calc.last_analytic_point_identity["intermediate_dtypes"]["energy"] == "torch.float64"
    assert calc.analytic_implicit_derivatives_admitted is False
    assert _caller_precision() == before
    record = calc.last_precision_execution_provenance
    assert record["caller_backend_selectors_before"] == record["caller_backend_selectors_after"]


@pytest.mark.skipif(not NATIVE_MODEL, reason="set AIMNET_TEST_MODEL_PATH for native complete-term closure")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_public_scalar_retains_all_terms_and_original_state(device, monkeypatch):
    import json
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    before = _caller_precision()
    calc = adapter.AIMNet2025Calculator(device, model_path=NATIVE_MODEL)
    atoms = molecule("H2O")
    state = calc._state_digest()
    modes = calc._live_runtime_identity()
    stages, d3_kwargs = {}, []
    handles = []
    def capture(name):
        def hook(module, args, output):
            stages[name] = output["energy"].sum()
        return hook
    for name, module in calc._modules():
        handles.append(module.register_forward_hook(capture(name)))
    handles.append(calc.backend.external_dftd3.register_forward_pre_hook(
        lambda module, args, kwargs: d3_kwargs.append(dict(kwargs)), with_kwargs=True))
    q = torch.tensor(atoms.positions, device=device, dtype=torch.float32, requires_grad=True)
    try:
        with adapter._native_precision_context(calc, "native-term-reference"):
            result = calc.backend({"coord":q, "numbers":atoms.numbers, "charge":0.}, forces=False, hessian=False)
            energy = result["energy"].sum()
            gradient = torch.autograd.grad(energy, q, retain_graph=True)[0]
            term_gradients = [torch.autograd.grad(term, q, retain_graph=True)[0] for term in
                             (stages["model"], stages["coulomb"]-stages["model"], stages["d3"]-stages["coulomb"])]
    finally:
        for handle in handles: handle.remove()
    assert d3_kwargs == [{"hessian": True}]
    terms = [stages["model"], stages["coulomb"]-stages["model"], stages["d3"]-stages["coulomb"]]
    assert all(abs(term.item()) > 1e-8 for term in terms)
    assert sum(terms).item() == energy.item()
    np.testing.assert_allclose(sum(term_gradients).detach().cpu().numpy(), gradient.detach().cpu().numpy(), atol=1e-6, rtol=1e-5)
    calc.calculate(atoms, ["energy", "forces"])
    assert calc.results["energy"] == energy.item() * base.EV2HARTREE
    np.testing.assert_array_equal(calc.results["forces"], -gradient.detach().cpu().numpy().astype(np.float64)*base.EV2HARTREE)
    assert calc._state_digest() == state and calc._live_runtime_identity() == modes
    json.dumps(calc.runtime_profile_identity, allow_nan=False)
    json.dumps(calc.last_analytic_point_identity, allow_nan=False)
    assert _caller_precision() == before


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_native_growing_molecule_preserves_fresh_complete_potential(device):
    """A small prior molecule must not truncate the next molecule's neighbors."""
    import os
    path=os.environ.get('AIMNET_TEST_MODEL_PATH')
    if not path:
        pytest.skip('set AIMNET_TEST_MODEL_PATH for the native runtime')
    from ase.build import molecule
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('native CUDA unavailable')
    ring=molecule('C6H6');second=ring.copy();second.translate([0,0,3.3]);large=ring+second
    water=molecule('H2O')
    for item in (water,large):item.info.update(charge=0,mult=1)
    reused=adapter.AIMNet2025Calculator(device,model_path=path)
    reused.calculate(water,properties=['energy','forces'])
    reused.calculate(large,properties=['energy','forces'])
    fresh=adapter.AIMNet2025Calculator(device,model_path=path)
    fresh.calculate(large,properties=['energy','forces'])
    assert abs(reused.results['energy']-fresh.results['energy'])<=2e-5
    np.testing.assert_allclose(reused.results['forces'],fresh.results['forces'],atol=2e-6,rtol=0)
    assert reused.last_analytic_point_identity['neighbor_graph_sha256']==fresh.last_analytic_point_identity['neighbor_graph_sha256']
