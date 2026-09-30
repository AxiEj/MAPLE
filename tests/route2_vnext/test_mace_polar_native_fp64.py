"""Strict process-FP64 boundary tests for the native MACE-POLAR adapter."""

from __future__ import annotations

import importlib
import sys

import pytest

torch = pytest.importorskip("torch")


class _Domain:
    def validate_atoms(self, atoms):
        del atoms


class _Release:
    checkpoint_sha256 = "a" * 64
    long_range_evaluator_profile = "molecular-realspace"


class _Base:
    device = "cpu"
    dtype = torch.float64
    provider_id = "old-provider"
    model_profile_id = "old-profile"
    provenance = {}
    provenance_sha256 = "b" * 64
    domain = _Domain()
    release_contract = _Release()

    def configuration_sha256(self):
        return "c" * 64


class _Graph:
    device = "cpu"
    dtype = torch.float64
    domain = _Domain()
    release_contract = _Release()

    def __init__(self):
        self.calls = 0

    def configuration_sha256(self):
        return "d" * 64

    def metadata(self):
        return {"base_configuration_sha256": "c" * 64}

    def topology_diagnostics(self, atoms):
        del atoms
        return {"locally_fixed_neighbor_mask": True}

    def energy_source_torch(self, atoms, positions):
        del atoms
        self.calls += 1
        return positions.square().sum(), positions.new_zeros((len(positions), 4))


def _set_default(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    return old


def test_import_has_no_default_dtype_side_effect():
    name = "maple.solvation.models.mace_polar_native_fp64"
    old = _set_default(torch.float32)
    try:
        sys.modules.pop(name, None)
        importlib.import_module(name)
        assert torch.get_default_dtype() is torch.float32
    finally:
        torch.set_default_dtype(old)


def test_factory_rejects_float32_before_official_construction(monkeypatch):
    import maple.solvation.models.mace_polar_native_fp64 as module

    called = False

    def build(**kwargs):
        nonlocal called
        called = True
        return _Base()

    monkeypatch.setattr(module, "_build_official_base", build)
    monkeypatch.setattr(module, "_verify_graph_longrange_runtime", lambda: {})
    old = _set_default(torch.float32)
    try:
        with pytest.raises(RuntimeError, match="before official construction"):
            module.build_official_mace_polar_native_fp64_adapter("cpu")
        assert called is False
    finally:
        torch.set_default_dtype(old)


def test_factory_catches_dtype_drift_during_construction(monkeypatch):
    import maple.solvation.models.mace_polar_native_fp64 as module

    def build(**kwargs):
        torch.set_default_dtype(torch.float32)
        return _Base()

    monkeypatch.setattr(module, "_build_official_base", build)
    monkeypatch.setattr(module, "_verify_graph_longrange_runtime", lambda: {})
    old = _set_default(torch.float64)
    try:
        with pytest.raises(RuntimeError, match="after official construction"):
            module.build_official_mace_polar_native_fp64_adapter("cpu")
    finally:
        torch.set_default_dtype(old)


def test_factory_is_the_only_path_to_a_production_adapter(monkeypatch):
    import maple.solvation.models.mace_polar_native_fp64 as module
    import maple.solvation.models.mace_polar_torch as graph_module

    runtime = {
        "graph_longrange_version": "0.4.0",
        "graph_longrange_realspace_sha256": module._REALSPACE_ELECTROSTATICS_SHA256,
        "graph_longrange_realspace_path": "/installed/realspace_electrostatics.py",
    }
    graph = _Graph()
    monkeypatch.setattr(module, "_build_official_base", lambda **kwargs: _Base())
    monkeypatch.setattr(module, "_verify_graph_longrange_runtime", lambda: runtime)
    monkeypatch.setattr(graph_module, "MACEPolarTorchGraphAdapter", lambda base: graph)
    old = _set_default(torch.float64)
    try:
        adapter = module.build_official_mace_polar_native_fp64_adapter("cpu")
        assert adapter.is_production_adapter is True
        assert adapter.provider_id == module.NATIVE_FP64_PROVIDER_ID
        assert adapter.configuration_sha256() == adapter.configuration_sha256()
    finally:
        torch.set_default_dtype(old)


def test_forward_guards_before_and_after_and_checks_outputs(monkeypatch):
    import maple.solvation.models.mace_polar_native_fp64 as module

    graph = _Graph()
    adapter = module.MACEPolarNativeFP64Adapter._from_graph_for_testing(graph)
    positions = torch.zeros((2, 3), dtype=torch.float64)
    old = _set_default(torch.float64)
    try:
        energy, source = adapter.energy_source_torch(object(), positions)
        assert energy.dtype is source.dtype is torch.float64
        assert graph.calls == 1

        torch.set_default_dtype(torch.float32)
        with pytest.raises(RuntimeError, match="before model forward"):
            adapter.energy_source_torch(object(), positions)
        assert graph.calls == 1

        torch.set_default_dtype(torch.float64)

        def drifting(atoms, value):
            del atoms
            graph.calls += 1
            torch.set_default_dtype(torch.float32)
            return value.sum(), value.new_zeros((len(value), 4))

        monkeypatch.setattr(graph, "energy_source_torch", drifting)
        with pytest.raises(RuntimeError, match="after model forward"):
            adapter.energy_source_torch(object(), positions)
    finally:
        torch.set_default_dtype(old)


def test_failed_forward_still_checks_drift_and_keeps_original_context(monkeypatch):
    import maple.solvation.models.mace_polar_native_fp64 as module

    graph = _Graph()
    adapter = module.MACEPolarNativeFP64Adapter._from_graph_for_testing(graph)
    positions = torch.zeros((2, 3), dtype=torch.float64)

    def fail_and_drift(atoms, value):
        del atoms, value
        torch.set_default_dtype(torch.float32)
        raise ValueError("original forward failure")

    monkeypatch.setattr(graph, "energy_source_torch", fail_and_drift)
    old = _set_default(torch.float64)
    try:
        with pytest.raises(RuntimeError, match="after model forward") as captured:
            adapter.energy_source_torch(object(), positions)
        assert isinstance(captured.value.__context__, ValueError)
        assert str(captured.value.__context__) == "original forward failure"
    finally:
        torch.set_default_dtype(old)


def test_adapter_identity_policy_and_metadata_are_distinct_and_immutable():
    import maple.solvation.models.mace_polar_native_fp64 as module

    adapter = module.MACEPolarNativeFP64Adapter._from_graph_for_testing(_Graph())
    assert "native-fp64" in adapter.provider_id
    assert "v2" in adapter.model_profile_id
    metadata = adapter.metadata()
    assert metadata["fp64_policy_id"] == module.NATIVE_FP64_POLICY_ID
    assert metadata["model_dtype"] == "torch.float64"
    assert metadata["input_dtype"] == "torch.float64"
    assert metadata["output_dtypes"] == ("torch.float64", "torch.float64")
    assert len(adapter.configuration_sha256()) == 64
    with pytest.raises(AttributeError):
        adapter.device = "cuda:0"


def test_public_constructor_cannot_mint_production_identity_from_arbitrary_graph():
    import maple.solvation.models.mace_polar_native_fp64 as module

    with pytest.raises(ValueError, match="factory"):
        module.MACEPolarNativeFP64Adapter(_Graph(), runtime={})
    adapter = module.MACEPolarNativeFP64Adapter._from_graph_for_testing(_Graph())
    assert adapter.is_production_adapter is False
    assert "engineering-test" in adapter.provider_id


def test_adapter_accepts_official_string_float64_metadata():
    import maple.solvation.models.mace_polar_native_fp64 as module

    graph = _Graph()
    graph.dtype = "float64"
    adapter = module.MACEPolarNativeFP64Adapter._from_graph_for_testing(graph)
    assert adapter.dtype is torch.float64


@pytest.mark.parametrize("invalid_dtype", ["float32", torch.float32, "torch.float16"])
def test_adapter_rejects_non_fp64_model_metadata(invalid_dtype):
    import maple.solvation.models.mace_polar_native_fp64 as module

    graph = _Graph()
    graph.dtype = invalid_dtype
    with pytest.raises(TypeError, match="float64 model graph"):
        module.MACEPolarNativeFP64Adapter._from_graph_for_testing(graph)


def test_runtime_dependency_drift_is_rechecked(monkeypatch):
    import maple.solvation.models.mace_polar_native_fp64 as module

    graph = _Graph()
    runtime = {
        "graph_longrange_version": "0.4.0",
        "graph_longrange_realspace_sha256": module._REALSPACE_ELECTROSTATICS_SHA256,
        "graph_longrange_realspace_path": "/installed/realspace_electrostatics.py",
    }
    adapter = module.MACEPolarNativeFP64Adapter(
        graph, runtime=runtime, _factory_token=module._FACTORY_TOKEN
    )
    monkeypatch.setattr(
        module,
        "_verify_graph_longrange_runtime",
        lambda: {**runtime, "graph_longrange_version": "0.4.1"},
    )
    with pytest.raises(RuntimeError, match="runtime dependency drifted"):
        adapter.configuration_sha256()


def test_metadata_returns_a_deep_copy():
    import maple.solvation.models.mace_polar_native_fp64 as module

    adapter = module.MACEPolarNativeFP64Adapter._from_graph_for_testing(_Graph())
    first = adapter.metadata()
    first["base_graph_metadata"]["base_configuration_sha256"] = "mutated"
    assert (
        adapter.metadata()["base_graph_metadata"]["base_configuration_sha256"]
        == "c" * 64
    )
