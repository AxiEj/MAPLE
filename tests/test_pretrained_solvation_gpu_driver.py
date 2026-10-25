"""CPU-only driver contracts: no weights, CUDA workloads, or compiler execution."""

from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from maple.function.benchmarking import pretrained_solvation_gpu as driver
from maple.function.benchmarking.solvation_gpu import (
    load_frozen_panel,
    FROZEN_PANEL_SHA256,
)

PANEL = Path(".omx/validation/gpu-fp32-20261025/frozen-panel-v2.json")


class Execution:
    device = "cpu"
    dtype = "float64"
    inference = "reference"
    torch_dtype = torch.float64
    provenance = {"device": "cpu", "compute_dtype": "float64"}

    def __init__(self):
        self.depth = 0
        self.fingerprints = 0

    @contextmanager
    def precision_context(self):
        self.depth += 1
        try:
            yield self
        finally:
            self.depth -= 1

    def check_precision(self, *, inputs=(), model=None):
        assert self.depth

        def check(item):
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                assert item.dtype == self.torch_dtype
            elif isinstance(item, dict):
                for value in item.values():
                    check(value)
            elif isinstance(item, (tuple, list)):
                for value in item:
                    check(value)

        check(inputs)
        if model is not None:
            check(list(model.buffers()))
        return {"compute_dtype": self.dtype}

    def runtime_context(self):
        assert self.depth
        self.fingerprints += 1
        return {"test": "owned_context"}


@pytest.fixture
def panel():
    return load_frozen_panel(PANEL, FROZEN_PANEL_SHA256)


def test_cast_floating_data_preserves_indices():
    index = torch.tensor([1, 2], dtype=torch.int32)
    data = {"a": [torch.ones(2), (torch.ones(1), index)]}
    result = driver.cast_floating_data(data, torch.float64)
    assert result["a"][0].dtype == torch.float64
    assert result["a"][1][0].dtype == torch.float64
    assert result["a"][1][1] is index


def test_any_checkpoint_mismatch_precedes_load(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"bad")
    calls = []
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        driver.AnySolvDiagnostic(
            Execution(),
            tmp_path,
            checkpoint=checkpoint,
            runtime_factory=lambda **kw: calls.append(kw),
            validate_source=lambda root: {},
            checkpoint_sha256="0" * 64,
        )
    assert not calls


@pytest.mark.parametrize("mode, expected", [("reference", 1), ("optimized", 3)])
def test_any_runtime_bounded_explicit_controls(tmp_path, mode, expected):
    execution = Execution()
    execution.inference = mode
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    instances = []

    class Runtime:
        def __init__(self, **kwargs):
            assert execution.depth
            assert kwargs["dtype"] == "float64"
            assert kwargs["inference"] == mode
            instances.append(kwargs)

        def evaluate(self, numbers, positions, **kwargs):
            assert kwargs == dict(
                charge=0, multiplicity=1, solvent=current[0], scalar_only=True
            )
            return 2.0, None

    adapter = driver.AnySolvDiagnostic(
        execution,
        tmp_path,
        runtime_factory=Runtime,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(b"weights").hexdigest(),
        validate_source=lambda root: {"revision": "fake"},
    )
    current = ["water"]
    for solvent, numbers in [
        ("water", [1, 8, 1]),
        ("water", [8, 1, 1]),
        ("ethanol", [1, 8, 1]),
        ("water", [1, 8, 1]),
    ]:
        current[0] = solvent
        assert (
            adapter(np.array(numbers), np.zeros((3, 3)), solvent, False)["energy_ev"]
            == 2.0
        )
    assert len(instances) == expected
    assert adapter.diagnostics()["resident_runtime_count"] == 1
    assert len(adapter.events) == expected
    assert execution.fingerprints == 0


def test_any_initialization_failure_not_retried(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"x")
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise ValueError("broken owner")

    adapter = driver.AnySolvDiagnostic(
        Execution(),
        tmp_path,
        runtime_factory=fail,
        checkpoint=path,
        checkpoint_sha256=hashlib.sha256(b"x").hexdigest(),
        validate_source=lambda root: {},
    )
    for solvent in ["water", "ethanol"]:
        with pytest.raises((ValueError, RuntimeError)):
            adapter(np.array([1]), np.zeros((1, 3)), solvent, False)
    assert len(calls) == 1
    assert "initialization_error" in adapter.events[0]


def test_gas_six_exact_geometries_actual_dtype_and_combination(monkeypatch, panel):
    execution = Execution()
    calls = []
    monkeypatch.setattr(
        driver,
        "verify_checkpoint",
        lambda *a: {"path": "mock", "sha256": driver.AIMNET2_SHA256},
    )

    class Calculator:
        def __init__(self, **kwargs):
            assert kwargs["coulomb_method"] == "simple"
            assert execution.depth
            self.model = torch.nn.Linear(1, 1)

        def _build_data(self, positions, atoms):
            assert execution.depth
            assert positions.dtype == torch.float64
            assert atoms.info == {"charge": 0, "mult": 1}
            calls.append((atoms.numbers.tolist(), positions.clone()))
            return {
                "positions": positions,
                "charge": torch.zeros(1),
                "nested": [torch.ones(1)],
                "indices": torch.ones(1, dtype=torch.int64),
            }

        def _forward_energy(self, data):
            assert execution.depth
            assert next(self.model.parameters()).dtype == torch.float64
            assert data["charge"].dtype == data["nested"][0].dtype == torch.float64
            assert data["indices"].dtype == torch.int64
            return data["positions"].square().sum()

    cases = [c for c in panel["scalar_cases"] if c["route"] == "anysolv"]
    corrections = [
        {"case_id": c["case_id"], "status": "success", "energy_ev": 1.0} for c in cases
    ]
    result = driver.collect_gas_base(
        execution, panel, corrections, calculator_factory=Calculator
    )
    assert len(calls) == len(result["gas_records"]) == 6
    assert len(result["combined_records"]) == 18
    assert result["gas_reference_compute_dtype"] == "float64"
    for row, case in zip(result["combined_records"], cases):
        assert row["geometry_sha256"] == case["geometry_sha256"]
        assert row["combined_ev"] == pytest.approx(
            np.square(case["positions_angstrom"]).sum() + 1.0
        )
    unique = {c["geometry_sha256"]: c for c in cases}
    for record, (numbers, positions) in zip(result["gas_records"], calls):
        original = unique[record["geometry_sha256"]]
        assert numbers == original["numbers"]
        assert torch.equal(
            positions, torch.tensor(original["positions_angstrom"], dtype=torch.float64)
        )


def test_panel_initialization_failure_keeps_denominators(monkeypatch, panel):
    monkeypatch.setattr(driver, "source_files", lambda *args: [])

    def fail(*args):
        raise ValueError("weights unavailable")

    report = driver.run_diagnostic("consolv", Execution(), panel, adapter_factory=fail)
    assert report["counts"]["scalar"] == 12
    assert report["counts"]["vacuum"] == 6
    assert report["counts"]["finite_difference_components"] == 234
    assert report["counts"]["failed_records"] == 272
    assert report["initialization_error"]["error_type"] == "ValueError"
    assert report["force_reference"] == "cpu64_central_fd"
    assert report["no_loss"] is False


def test_source_drift_and_gas_after_correction_timing(tmp_path, monkeypatch, panel):
    source = tmp_path / "source.py"
    source.write_text("before")
    monkeypatch.setattr(driver, "source_files", lambda *args: [source])
    order = []

    class Adapter:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            order.append("correction")
            return {"energy_ev": 0.0}

        def diagnostics(self):
            return {}

    def gas(*args):
        assert order and all(item == "correction" for item in order)
        order.append("gas")
        source.write_text("after")
        return {}

    monkeypatch.setattr(driver, "collect_gas_base", gas)
    execution = Execution()
    report = driver.run_diagnostic(
        "anysolv", execution, panel, withgas=True, adapter_factory=Adapter
    )
    assert report["source_drift"] is True
    assert report["timing"]["fresh_inference"]["count"] == 10
    assert order[-1] == "gas"
    assert execution.fingerprints == 1


def test_cli_requires_diagnostic_and_never_overwrites(tmp_path):
    with pytest.raises(SystemExit):
        driver.main(
            [
                "--route",
                "anysolv",
                "--panel",
                str(PANEL),
                "--output",
                str(tmp_path / "new"),
            ]
        )
    path = tmp_path / "existing"
    path.write_text("retain")
    with pytest.raises(FileExistsError):
        driver.main(
            [
                "--diagnostic",
                "--route",
                "anysolv",
                "--panel",
                str(PANEL),
                "--output",
                str(path),
            ]
        )
    assert path.read_text() == "retain"


def test_con_reuses_weights_scopes_callbacks_and_bounds_owners(monkeypatch, tmp_path):
    execution = Execution()
    execution.inference = "optimized"
    loaded, owners = [], []
    monkeypatch.setattr(
        driver,
        "verify_checkpoint",
        lambda path, sha: {"path": str(path), "sha256": sha},
    )

    class Owner:
        def __init__(self, model):
            self.export_records = [{"tensor_constants": 0}]
            self._variants = {False: (None, {"compilations": 1, "executions": 1})}
            owners.append(self)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weights", torch.ones(1, dtype=torch.float64))
            self._compiled_energy = None

        def _valid_edges(self, r):
            return torch.arange(len(r)), None, None

        def energy_ev(self, n, r, index):
            assert execution.depth
            assert r.dtype == torch.float64 and n.dtype == torch.long
            return r.square().sum()

        def energy_and_forces_ev(self, n, r, index):
            return self.energy_ev(n, r, index), -2 * r

    def load(path, **kwargs):
        assert execution.depth
        assert kwargs == dict(
            device="cpu",
            dtype=torch.float64,
            expected_state="solution" if not loaded else "vacuum",
            inference="optimized",
        )
        model = Model()
        loaded.append(model)
        return model

    adapter = driver.ConSolvDiagnostic(
        execution,
        loader=load,
        compiled_factory=Owner,
        model_path=tmp_path / "solution.pt",
    )
    for n, solvent in [
        (3, "ethanol"),
        (3, "ethanol"),
        (4, "ethanol"),
        (4, "toluene"),
        (3, "vacuum"),
    ]:
        result = adapter(np.ones(n), np.ones((n, 3)), solvent, True)
        assert result["energy_ev"] == 3 * n
    metadata = adapter.diagnostics()
    assert len(loaded) == 2
    assert len(owners) == 4
    assert metadata["current_compiled_owners"] == 2
    assert metadata["source_state_unchanged"]
    assert metadata["owners"][2]["export_records"] == [{"tensor_constants": 0}]
    assert execution.fingerprints == 0


def test_cli_records_execution_initialization_failure(tmp_path):
    import json

    path = tmp_path / "failure.json"
    status = driver.main(
        [
            "--diagnostic",
            "--route",
            "anysolv",
            "--device",
            "invalid",
            "--panel",
            str(PANEL),
            "--output",
            str(path),
        ]
    )
    assert status == 1
    receipt = json.loads(path.read_text())
    assert receipt["diagnostic"] is True
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "ValueError"


def test_gas_checkpoint_mismatch_precedes_load(tmp_path, panel):
    checkpoint = tmp_path / "gas.pt"
    checkpoint.write_bytes(b"bad")
    loaded = []
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        driver.collect_gas_base(
            Execution(),
            panel,
            [],
            checkpoint=checkpoint,
            calculator_factory=lambda **kw: loaded.append(kw),
        )
    assert not loaded


def test_peak_statistics_initialize_cuda_before_reset(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("initialize"))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", device)),
    )
    execution = Execution()
    execution.device = "cuda:0"
    driver._reset_peak(execution)
    assert calls == ["initialize", ("reset", "cuda:0")]
