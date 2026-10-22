"""Release and precision contracts for the official MACE-POLAR FP32 adapter."""

from __future__ import annotations

import importlib
import sys

import pytest
from ase import Atoms

torch = pytest.importorskip("torch")


class _Domain:
    def validate_atoms(self, atoms):
        del atoms


class _Release:
    checkpoint_sha256 = "a" * 64


class _Graph:
    device = "cpu"
    dtype = torch.float32
    domain = _Domain()
    release_contract = _Release()

    def __init__(self):
        self.calls = 0
        self.digest = "b" * 64

    def configuration_sha256(self):
        return self.digest

    def metadata(self):
        return {"base_configuration_sha256": "c" * 64}

    def topology_diagnostics(self, atoms):
        del atoms
        return {"locally_fixed_neighbor_mask": True}

    def energy_source_torch(self, atoms, positions):
        self.calls += 1
        return positions.square().sum(), positions.new_zeros((len(atoms), 4))


def _set_default(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    return old


def test_import_has_no_default_dtype_side_effect():
    name = "maple.solvation.models.mace_polar_fp32"
    old = _set_default(torch.float64)
    try:
        sys.modules.pop(name, None)
        importlib.import_module(name)
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(old)


def test_factory_rejects_non_fp32_before_runtime_or_checkpoint(monkeypatch):
    import maple.solvation.models.mace_polar_fp32 as module

    called = []
    monkeypatch.setattr(module, "_verify_runtime", lambda: called.append("runtime"))
    old = _set_default(torch.float64)
    try:
        with pytest.raises(RuntimeError, match="before official construction"):
            module.build_official_mace_polar_fp32_adapter()
        assert called == []
    finally:
        torch.set_default_dtype(old)


def test_factory_normalizes_cuda_and_is_only_production_path(monkeypatch):
    import maple.solvation.models.mace_polar_fp32 as module

    graph = _Graph()
    events = []
    runtime = {
        "mace_torch_version": "0.3.16",
        "graph_longrange_version": "0.4.0",
        "graph_longrange_realspace_path": (
            "graph_longrange/realspace_electrostatics.py"
        ),
        "graph_longrange_realspace_sha256": module._GRAPH_LONGRANGE_REALSPACE_SHA256,
    }
    monkeypatch.setattr(
        module,
        "_prepare_cuda_device",
        lambda device: events.append(("prepare", device)),
    )
    monkeypatch.setattr(
        module,
        "_verify_runtime",
        lambda: events.append(("runtime",)) or runtime,
    )
    monkeypatch.setattr(
        module,
        "_resolve_checkpoint",
        lambda value: events.append(("checkpoint", value)) or "checkpoint",
    )
    monkeypatch.setattr(
        module,
        "_build_graph",
        lambda **kwargs: events.append(("build", kwargs)) or graph,
    )
    old = _set_default(torch.float32)
    try:
        adapter = module.build_official_mace_polar_fp32_adapter("cuda")
        assert adapter.is_production_adapter is True
        assert events == [
            ("prepare", "cuda:0"),
            ("runtime",),
            ("checkpoint", None),
            ("build", {"device": "cuda:0", "checkpoint": "checkpoint"}),
        ]
        assert adapter.configuration_sha256() == adapter.configuration_sha256()
    finally:
        torch.set_default_dtype(old)

    with pytest.raises(ValueError, match="official factory"):
        module.MACEPolarFP32Adapter(graph, runtime=runtime)


def test_cuda_preparation_precedes_availability_check(monkeypatch):
    import maple.solvation.experimental.cuda_execution as cuda_execution
    import maple.solvation.models.mace_polar_fp32 as module

    events = []
    monkeypatch.setattr(
        cuda_execution,
        "prepare_cuda_execution",
        lambda: events.append("prepare"),
    )

    def unavailable():
        assert events == ["prepare"]
        events.append("availability")
        return False

    monkeypatch.setattr(torch.cuda, "is_available", unavailable)
    with pytest.raises(RuntimeError, match="CUDA device is unavailable"):
        module._prepare_cuda_device("cuda:0")
    assert events == ["prepare", "availability"]


def test_cuda_device_index_is_validated_after_preparation(monkeypatch):
    import maple.solvation.experimental.cuda_execution as cuda_execution
    import maple.solvation.models.mace_polar_fp32 as module

    events = []
    monkeypatch.setattr(
        cuda_execution,
        "prepare_cuda_execution",
        lambda: events.append("prepare"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="only 1 CUDA device"):
        module._prepare_cuda_device("cuda:1")
    assert events == ["prepare"]


def test_adapter_metadata_is_fp32_without_claiming_all_op_fp32():
    import maple.solvation.models.mace_polar_fp32 as module

    adapter = module.MACEPolarFP32Adapter._from_graph_for_testing(_Graph())
    metadata = adapter.metadata()
    assert adapter.dtype is torch.float32
    assert metadata["model_dtype"] == "torch.float32"
    assert metadata["input_dtype"] == "torch.float32"
    assert metadata["output_dtypes"] == ("torch.float32", "torch.float32")
    assert metadata["all_operations_forced_to_fp32"] is False
    assert metadata["upstream_explicit_fp64_reductions_preserved"] is True
    assert adapter.topology_diagnostics(object())["locally_fixed_neighbor_mask"]
    with pytest.raises(AttributeError):
        adapter.device = "cuda:0"


@pytest.mark.parametrize("invalid_dtype", [torch.float64, "float64", "float16"])
def test_adapter_rejects_non_fp32_graph(invalid_dtype):
    import maple.solvation.models.mace_polar_fp32 as module

    graph = _Graph()
    graph.dtype = invalid_dtype
    with pytest.raises(TypeError, match="float32 model graph"):
        module.MACEPolarFP32Adapter._from_graph_for_testing(graph)


def test_forward_requires_exact_input_and_output_contract(monkeypatch):
    import maple.solvation.models.mace_polar_fp32 as module

    graph = _Graph()
    adapter = module.MACEPolarFP32Adapter._from_graph_for_testing(graph)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    positions = torch.tensor(atoms.positions, dtype=torch.float32)
    old = _set_default(torch.float32)
    try:
        energy, source = adapter.energy_source_torch(atoms, positions)
        assert energy.shape == () and energy.dtype is torch.float32
        assert source.shape == (2, 4) and source.dtype is torch.float32
        assert graph.calls == 1

        with pytest.raises(TypeError, match="positions must be float32"):
            adapter.energy_source_torch(atoms, positions.double())
        assert graph.calls == 1

        monkeypatch.setattr(
            graph,
            "energy_source_torch",
            lambda atoms, value: (
                value.sum().double(),
                value.new_zeros((len(atoms), 4)),
            ),
        )
        with pytest.raises(RuntimeError, match="scalar float32 energy"):
            adapter.energy_source_torch(atoms, positions)
    finally:
        torch.set_default_dtype(old)


def test_forward_checks_ambient_dtype_before_and_after(monkeypatch):
    import maple.solvation.models.mace_polar_fp32 as module

    graph = _Graph()
    adapter = module.MACEPolarFP32Adapter._from_graph_for_testing(graph)
    atoms = Atoms("H", positions=[[0, 0, 0]])
    positions = torch.zeros((1, 3), dtype=torch.float32)
    old = _set_default(torch.float64)
    try:
        with pytest.raises(RuntimeError, match="before model forward"):
            adapter.energy_source_torch(atoms, positions)
        assert graph.calls == 0

        torch.set_default_dtype(torch.float32)

        def drift(atoms, value):
            torch.set_default_dtype(torch.float64)
            return value.sum(), value.new_zeros((len(atoms), 4))

        monkeypatch.setattr(graph, "energy_source_torch", drift)
        with pytest.raises(RuntimeError, match="after model forward"):
            adapter.energy_source_torch(atoms, positions)
    finally:
        torch.set_default_dtype(old)


def test_configuration_rechecks_graph_and_runtime(monkeypatch):
    import maple.solvation.models.mace_polar_fp32 as module

    graph = _Graph()
    adapter = module.MACEPolarFP32Adapter._from_graph_for_testing(graph)
    graph.digest = "d" * 64
    with pytest.raises(RuntimeError, match="configuration drifted"):
        adapter.configuration_sha256()

    runtime = {
        "mace_torch_version": "0.3.16",
        "graph_longrange_version": "0.4.0",
        "graph_longrange_realspace_path": (
            "graph_longrange/realspace_electrostatics.py"
        ),
        "graph_longrange_realspace_sha256": module._GRAPH_LONGRANGE_REALSPACE_SHA256,
    }
    production = module.MACEPolarFP32Adapter(
        _Graph(), runtime=runtime, _factory_token=module._FACTORY_TOKEN
    )
    monkeypatch.setattr(
        module,
        "_verify_runtime",
        lambda: {**runtime, "graph_longrange_version": "0.4.1"},
    )
    with pytest.raises(RuntimeError, match="runtime dependency drifted"):
        production.configuration_sha256()


def test_checkpoint_rejects_wrong_bytes(tmp_path):
    import maple.solvation.models.mace_polar_fp32 as module

    checkpoint = tmp_path / "MACE-POLAR-1-M.model"
    checkpoint.write_bytes(b"not-the-official-checkpoint")
    with pytest.raises(RuntimeError, match="size/SHA-256"):
        module._resolve_checkpoint(checkpoint)


def test_metadata_returns_deep_copy():
    import maple.solvation.models.mace_polar_fp32 as module

    adapter = module.MACEPolarFP32Adapter._from_graph_for_testing(_Graph())
    metadata = adapter.metadata()
    metadata["base_graph_metadata"]["base_configuration_sha256"] = "changed"
    assert (
        adapter.metadata()["base_graph_metadata"]["base_configuration_sha256"]
        == "c" * 64
    )
