"""Resident identity/cache ownership tests; synthetic unless explicitly enabled."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import anysolv


@pytest.fixture
def official_api(tmp_path, monkeypatch):
    @dataclass
    class Settings:
        execution_mode: str = "general"
        tf32: bool = False
        compile: bool = False
        merge_mole: bool = False

    class Model:
        norm = {"energy": {"rmsd": 2.0, "mean": 0.5}}
        eps_transform = "log"

        def __call__(self, data):
            if getattr(self, "_prepared", False):
                assert torch._dynamo.config.fail_on_recompile_limit_hit is True
            if getattr(self, "_prepared", False):
                return self._run(data)
            return self._raw_forward(data)

        def _raw_forward(self, data):
            return {
                "energy": {
                    "energy": torch.tensor(
                        [-0.25 if not data["solvent"].any() else data["spin"]],
                        dtype=data["dtype"],
                    )
                },
                "forces": {
                    "forces": torch.ones((len(data["numbers"]), 3), dtype=data["dtype"])
                },
            }

    source = tmp_path / "anysolv"
    source.mkdir()
    package = ModuleType("anysolv")
    package.__file__ = str(source / "__init__.py")
    package.InferenceSettings = Settings
    loads, data_calls = [], []

    def load(checkpoint, **controls):
        loads.append((checkpoint, controls))
        return Model()

    def forbidden_predict(*args, **kwargs):
        pytest.fail("unowned upstream pathname cache/predictor reached")

    package.load_model = load
    package.predict_solvation_energy = forbidden_predict
    data = ModuleType("anysolv.data")

    def build(inputs, **kwargs):
        data_calls.append(kwargs)
        kwargs = dict(kwargs, solvent=kwargs["solvent"].to(dtype=kwargs["dtype"]))
        return {"numbers": inputs[0], **kwargs}

    data.build_atomic_data = build
    solvent = ModuleType("anysolv.solvent")
    solvent.get_solvent_vector = lambda name, **kwargs: torch.tensor(
        [[0.0 if name is None else {"water": 1.0, "toluene": 2.0}[name]]]
    )
    predict = ModuleType("anysolv.predict")
    predict._MODEL_CACHE = {}
    for name, module in (
        ("anysolv", package),
        ("anysolv.data", data),
        ("anysolv.solvent", solvent),
        ("anysolv.predict", predict),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        source=source,
        checkpoint=tmp_path / "model.pt",
        loads=loads,
        data_calls=data_calls,
        settings=Settings,
        predict=predict,
    )


def _runtime(api, digest="verified-digest", checkpoint=None):
    return anysolv._OfficialAnySolvRuntime(
        dtype="float64",
        source_root=api.source,
        checkpoint=checkpoint or api.checkpoint,
        checkpoint_sha256=digest,
        device="cpu",
    )


def test_hostile_upstream_cache_is_neither_trusted_nor_modified(
    official_api, monkeypatch
):
    api = official_api
    key = (str(api.checkpoint), "cpu", str(torch.float64), repr(api.settings()))
    hostile = object()
    api.predict._MODEL_CACHE[key] = hostile
    runtime = _runtime(api)
    assert len(api.loads) == 1
    assert runtime._resident.model is not hostile
    assert api.predict._MODEL_CACHE[key] is hostile
    monkeypatch.setattr(
        anysolv, "_sha256", lambda _: pytest.fail("per-evaluation rehash")
    )
    energy, forces = runtime.evaluate(
        np.array([1]), np.zeros((1, 3)), charge=0, multiplicity=2, solvent="water"
    )
    assert energy == 4.5
    assert np.array_equal(forces, np.full((1, 3), 2.0))
    assert api.predict._MODEL_CACHE[key] is hostile


def test_owned_model_shared_by_digest_across_paths_and_solvents(official_api):
    api = official_api
    water = _runtime(api)
    toluene = _runtime(api, checkpoint=api.checkpoint.with_name("same-bytes.pt"))
    assert water._resident is toluene._resident
    for runtime, solvent in ((water, "water"), (toluene, "toluene")):
        runtime.evaluate(
            np.array([1]), np.zeros((1, 3)), charge=0, multiplicity=1, solvent=solvent
        )
    assert len(api.loads) == 1
    controls = api.loads[0][1]
    assert controls["device"] == "cpu"
    assert controls["dtype"] is torch.float64
    assert controls["inference_settings"] == api.settings()
    assert [call["solvent"].item() for call in api.data_calls] == [1.0, 2.0]
    different = _runtime(api, digest="different-checkpoint-digest")
    assert different._resident is not water._resident
    assert len(api.loads) == 2


def test_private_cache_releases_resident_when_last_runtime_is_gone(official_api):
    runtime = _runtime(official_api)
    identity = runtime._resident.identity
    resident_ref = weakref.ref(runtime._resident)
    del runtime
    gc.collect()
    assert resident_ref() is None
    assert identity not in anysolv._RESIDENTS


@pytest.mark.skipif(
    os.environ.get("MAPLE_TEST_ANYSOLV_MOE") != "1",
    reason="explicit real AnySolv-MoE parity opt-in required",
)
def test_real_owned_model_matches_original_predictor_on_two_examples(monkeypatch):
    source = Path(
        os.environ.get(
            "MAPLE_ANYSOLV_TEST_SOURCE_ROOT", "/home/axie/.cache/maple/anysolv/anysolv"
        )
    )
    anysolv._validate_source(source)
    examples = [
        {
            "numbers": [8, 1, 1],
            "positions": [[0, 0, 0], [0.95, 0, 0], [-0.24, 0.92, 0]],
            "solvent": "water",
        },
        {
            "numbers": [6, 1, 1, 1, 1],
            "positions": [
                [0, 0, 0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
            "solvent": "toluene",
        },
    ]
    # The original API runs in a separate process, so parity requires only one
    # 2.3 GB resident per process and never adopts its pathname-cache entries.
    code = """
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('anysolv',root/'__init__.py',submodule_search_locations=[str(root)])
module=importlib.util.module_from_spec(spec)
sys.modules['anysolv']=module
spec.loader.exec_module(module)
records=[]
for case in json.loads(sys.argv[3]):
    energy,forces=module.predict_solvation_energy(
        (np.array(case['numbers']),np.array(case['positions'])),charge=0,spin=1,
        solvent=case['solvent'],checkpoint=sys.argv[2],device='cpu',dtype=torch.float64,
        inference_settings='default')
    records.append({'energy_ev':energy,'forces_ev_per_angstrom':forces.tolist()})
print(json.dumps(records))
"""
    environment = {
        **os.environ,
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(source),
            str(anysolv.ANYSOLV_MOE_DEFAULT_PATH),
            json.dumps(examples),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    references = json.loads(completed.stdout.splitlines()[-1])
    # Occupy the real official pathname cache with an unusable, unowned object.
    # MAPLE must neither consume it nor replace another caller's cache entry.
    from importlib import import_module, util

    if "anysolv" not in sys.modules:
        spec = util.spec_from_file_location(
            "anysolv", source / "__init__.py", submodule_search_locations=[str(source)]
        )
        assert spec is not None and spec.loader is not None
        module = util.module_from_spec(spec)
        sys.modules["anysolv"] = module
        spec.loader.exec_module(module)

    official_predict = import_module("anysolv.predict")
    official_settings = import_module("anysolv._backbone._compat.inference")
    cache_key = (
        str(anysolv.ANYSOLV_MOE_DEFAULT_PATH.resolve()),
        "cpu",
        str(torch.float64),
        repr(official_settings.guess_inference_settings("default")),
    )
    hostile = object()
    monkeypatch.setitem(official_predict._MODEL_CACHE, cache_key, hostile)
    backends = [
        anysolv.AnySolvBackend(
            device="cpu",
            dtype="float64",
            solvent=case["solvent"],
            source_root=str(source),
        )
        for case in examples
    ]
    assert backends[0]._runtime._resident is backends[1]._runtime._resident
    assert official_predict._MODEL_CACHE[cache_key] is hostile
    monkeypatch.setattr(anysolv, "_sha256", lambda _: pytest.fail("per-record rehash"))
    records = []
    for case, backend, reference in zip(examples, backends, references):
        energy, forces = backend._runtime.evaluate(
            np.array(case["numbers"]),
            np.array(case["positions"]),
            charge=0,
            multiplicity=1,
            solvent=case["solvent"],
        )
        energy_error = abs(energy - reference["energy_ev"])
        force_error = float(
            np.max(np.abs(forces - reference["forces_ev_per_angstrom"]))
        )
        assert energy_error <= 1e-12
        assert force_error <= 1e-12
        result = backend.evaluate(
            Atoms(
                numbers=case["numbers"],
                positions=case["positions"],
                info={"charge": 0, "mult": 1},
            )
        )
        assert result.energy_hartree == pytest.approx(energy / anysolv.EV_PER_HARTREE)
        assert result.forces_hartree_per_angstrom is None
        records.append(
            {
                "input": case,
                "reference": reference,
                "owned_energy_ev": energy,
                "energy_abs_error_ev": energy_error,
                "forces_max_abs_error_ev_per_angstrom": force_error,
            }
        )
    artifact = os.environ.get("MAPLE_ANYSOLV_REPAIR_PARITY_ARTIFACT")
    if artifact:
        payload = {
            "scope": "official predictor versus privately owned reference inference; no force admission",
            "source_revision": anysolv.ANYSOLV_SOURCE_REVISION,
            "checkpoint_sha256": anysolv.ANYSOLV_MOE_CHECKPOINT_SHA256,
            "resident_identity": backends[0]._runtime._resident.identity,
            "solvents_share_one_resident": True,
            "hostile_official_path_cache_ignored_and_preserved": True,
            "absolute_tolerance": 1e-12,
            "records": records,
        }
        Path(artifact).write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture
def mocked_cuda(official_api, monkeypatch):
    """Exercise execution policy without creating any CUDA tensors/workloads."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(anysolv.SolvationExecution, "runtime_context", lambda self: {})
    api = official_api
    original_load = sys.modules["anysolv"].load_model
    api.preparations = []
    api.fallback = False
    api.prepare_failure = False

    class UMASFastGPUBackend:
        pass

    def load(checkpoint, **controls):
        model = original_load(checkpoint, **controls)
        model._settings = controls["inference_settings"]
        model._prepared = False
        model._run = None
        model.backbone = SimpleNamespace(
            backend=UMASFastGPUBackend(),
            _merged_composition=None,
            on_predict_check=lambda data: None,
        )

        def prepare(data):
            api.preparations.append(data)
            assert model._settings.compile is False
            if api.prepare_failure:
                raise RuntimeError("mock preparation failure")
            model._run = model._raw_forward
            model.backbone._merged_composition = {"composition": "frozen"}
            model._prepared = True

        model._prepare = prepare
        return model

    monkeypatch.setattr(sys.modules["anysolv"], "load_model", load)
    api.backend_calls = []
    api.compile_calls = []
    api.real_compile = torch.compile

    def inductor(graph, inputs, *, config_patches):
        assert config_patches == anysolv.STRICT_INDUCTOR_OPTIONS
        api.backend_calls.append(config_patches)
        return graph

    def compile_mock(raw, *, backend, dynamic):
        assert dynamic is True
        api.compile_calls.append(raw)
        executable = None

        def compiled(data):
            nonlocal executable
            if api.fallback:
                return raw(data)
            if executable is None:
                executable = backend(raw, [data])
            return executable(data)

        compiled._torchdynamo_orig_callable = raw
        return compiled

    from torch._dynamo.backends import registry

    monkeypatch.setattr(registry, "lookup_backend", lambda name: inductor)
    monkeypatch.setattr(torch, "compile", compile_mock)
    return api


def _cuda_runtime(api, *, device="cuda:0", inference="optimized", dtype="float32"):
    return anysolv._OfficialAnySolvRuntime(
        source_root=api.source,
        checkpoint=api.checkpoint,
        checkpoint_sha256="verified-digest",
        device=device,
        dtype=dtype,
        inference=inference,
    )


def _evaluate(
    runtime,
    *,
    numbers=(8, 1, 1),
    charge=0,
    multiplicity=1,
    solvent="water",
    displacement=0.0,
    scalar_only=True,
):
    return runtime.evaluate(
        np.array(numbers),
        np.zeros((len(numbers), 3)) + displacement,
        charge=charge,
        multiplicity=multiplicity,
        solvent=solvent,
        scalar_only=scalar_only,
    )


def test_runtime_defaults_fp32_and_cache_separates_dtype(official_api):
    api = official_api
    fp64 = _runtime(api)
    fp32 = anysolv._OfficialAnySolvRuntime(
        source_root=api.source,
        checkpoint=api.checkpoint,
        checkpoint_sha256="verified-digest",
        device="cpu",
    )
    assert fp32._resident is not fp64._resident
    assert api.loads[-1][1]["dtype"] is torch.float32
    _evaluate(fp32)
    assert api.data_calls[-1]["dtype"] is torch.float32
    assert api.data_calls[-1]["device"] == "cpu"


def test_device_mode_and_owner_separation_with_exact_settings(mocked_cuda):
    api = mocked_cuda
    reference = _cuda_runtime(api, inference="reference")
    shared = _cuda_runtime(api, inference="reference")
    indexed = _cuda_runtime(api, device="cuda:1", inference="reference")
    optimized = _cuda_runtime(api)
    second_owner = _cuda_runtime(api)
    assert reference._resident is shared._resident
    assert indexed._resident is not reference._resident
    assert optimized._resident is not reference._resident
    assert optimized._resident is not second_owner._resident
    controls = api.loads[-1][1]
    assert controls == {
        "device": "cuda:0",
        "dtype": torch.float32,
        "inference_settings": api.settings(
            execution_mode="umas_fast_gpu", tf32=False, compile=False, merge_mole=True
        ),
    }
    assert api.preparations == []
    _evaluate(optimized)
    assert api.data_calls[-1]["device"] == "cuda:0"
    assert api.data_calls[-1]["dtype"] is torch.float32
    assert optimized._resident.identity not in anysolv._RESIDENTS


@pytest.mark.parametrize(
    "change",
    [
        {"numbers": (8, 1)},
        {"numbers": (6, 1, 1)},
        {"charge": 1},
        {"multiplicity": 2},
        {"solvent": "toluene"},
    ],
)
def test_optimized_signature_change_fails_before_build_or_remerge(mocked_cuda, change):
    api = mocked_cuda
    runtime = _cuda_runtime(api)
    _evaluate(runtime)
    count = len(api.data_calls)
    with pytest.raises(ValueError, match="construct a new calculator"):
        _evaluate(runtime, **change)
    assert len(api.data_calls) == count
    assert len(api.preparations) == 1


def test_optimized_allows_geometry_change_and_equal_composition_permutation(
    mocked_cuda,
):
    api = mocked_cuda
    runtime = _cuda_runtime(api)
    first = _evaluate(runtime)
    assert _evaluate(runtime, displacement=0.05, numbers=(1, 8, 1)) == first
    assert len(api.preparations) == 1


@pytest.mark.parametrize("failure", [None, "fallback", "prepare_failure"])
def test_optimized_restores_hostile_global_and_rejects_fallback(
    mocked_cuda, monkeypatch, failure
):
    api = mocked_cuda
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 7)
    if failure:
        setattr(api, failure, True)
    runtime = _cuda_runtime(api)
    if failure:
        with pytest.raises(
            RuntimeError, match="fallback is forbidden|preparation failure"
        ):
            _evaluate(runtime)
    else:
        _evaluate(runtime)
    assert torch._dynamo.config.recompile_limit == 7
    assert torch._dynamo.config.fail_on_recompile_limit_hit is False


def test_optimized_rejects_backend_fallback_even_with_compiled_wrapper(mocked_cuda):
    runtime = _cuda_runtime(mocked_cuda)
    _evaluate(runtime)
    runtime._resident.model.backbone.backend = object()
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        _evaluate(runtime)


def test_scalar_only_avoids_diagnostic_force_transfer(official_api, monkeypatch):
    runtime = _runtime(official_api)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self: pytest.fail("force transfer"))
    energy, forces = _evaluate(runtime)
    assert energy == 2.5
    assert forces is None


def test_optimized_owner_releases_without_cache_retention(mocked_cuda):
    runtime = _cuda_runtime(mocked_cuda)
    resident = weakref.ref(runtime._resident)
    del runtime
    gc.collect()
    assert resident() is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reference_vacuum_keeps_exact_zero_and_requested_dtype(official_api, dtype):
    runtime = anysolv._OfficialAnySolvRuntime(
        source_root=official_api.source,
        checkpoint=official_api.checkpoint,
        checkpoint_sha256="verified-digest",
        device="cpu",
        dtype=dtype,
    )
    energy, forces = _evaluate(runtime, solvent=None)
    assert energy == 0.0
    assert forces is None
    assert official_api.data_calls[-1]["solvent"].count_nonzero() == 0
    assert official_api.data_calls[-1]["dtype"] == getattr(torch, dtype)


def test_optimized_vacuum_is_zero_and_cannot_be_reused_for_solution(mocked_cuda):
    runtime = _cuda_runtime(mocked_cuda)
    assert _evaluate(runtime, solvent=None) == (0.0, None)
    with pytest.raises(ValueError, match="construct a new calculator"):
        _evaluate(runtime, solvent="water")


def test_reference_cache_binds_source_digest_and_root(official_api, monkeypatch):
    first = _runtime(official_api)
    monkeypatch.setattr(anysolv, "_SOURCE_DIGEST", "different-source-digest")
    second = _runtime(official_api)
    assert first._resident is not second._resident
    assert second._resident.identity[1] == "different-source-digest"


def test_optimized_rejects_global_compiler_error_suppression(mocked_cuda, monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    runtime = _cuda_runtime(mocked_cuda)
    with pytest.raises(ValueError, match="suppress_errors=False"):
        _evaluate(runtime)
    assert mocked_cuda.preparations == []
    assert torch._dynamo.config.suppress_errors is True


@pytest.mark.parametrize("inference", ["reference", "optimized"])
def test_every_fresh_call_checks_live_precision_without_fingerprinting(
    mocked_cuda, monkeypatch, inference
):
    from maple.function.calculator import solvation_execution

    runtime = _cuda_runtime(mocked_cuda, inference=inference)
    monkeypatch.setattr(
        solvation_execution.metadata, "version", lambda _: pytest.fail("version lookup")
    )
    monkeypatch.setattr(
        runtime._execution.__class__,
        "runtime_context",
        lambda _: pytest.fail("full fingerprint"),
    )
    checks = []
    original = runtime._execution.__class__.check_precision

    def checked(self, **kwargs):
        state = original(self, **kwargs)
        checks.append(state)
        assert not state["cudnn_tf32"] and not state["matmul_tf32"]
        return state

    monkeypatch.setattr(runtime._execution.__class__, "check_precision", checked)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    _evaluate(runtime)
    first = len(checks)
    _evaluate(runtime, displacement=0.01)
    assert first >= 2 and len(checks) >= 2 * first
    assert torch.backends.cudnn.allow_tf32 is True
    if inference == "optimized":
        assert runtime.compilation_active
        assert runtime._compiled_backend_executions == 2
        assert len(mocked_cuda.backend_calls) == 1


def test_compiled_marker_without_backend_execution_is_rejected_on_fresh_call(
    mocked_cuda,
):
    runtime = _cuda_runtime(mocked_cuda)
    _evaluate(runtime)
    assert runtime.compilation_active
    assert callable(runtime._resident.model._run._torchdynamo_orig_callable)
    mocked_cuda.fallback = True
    with pytest.raises(RuntimeError, match="compiled backend did not execute"):
        _evaluate(runtime)
    assert not runtime.compilation_active


def test_optimized_checks_actual_merge_signature_on_every_call(mocked_cuda):
    runtime = _cuda_runtime(mocked_cuda)
    _evaluate(runtime)

    def bad_merge(data):
        raise ValueError("actual merged signature differs")

    runtime._resident.model.backbone.on_predict_check = bad_merge
    with pytest.raises(ValueError, match="actual merged signature differs"):
        _evaluate(runtime)


@pytest.mark.parametrize("failure", ["fallback", "prepare_failure"])
def test_optimized_model_errors_restore_precision_and_compiler_state(
    mocked_cuda, monkeypatch, failure
):
    import torch._inductor.config as config

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    runtime = _cuda_runtime(mocked_cuda)
    setattr(mocked_cuda, failure, True)
    with config.patch({"use_fast_math": True}):
        with pytest.raises(RuntimeError):
            _evaluate(runtime)
        assert config.use_fast_math is True
    assert (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    ) == previous


def test_real_dynamo_executes_owned_backend_sentinel_with_cpu_mock_graphs(
    mocked_cuda, monkeypatch
):
    """Real Torch Dynamo, synthetic CUDA policy, CPU graph backend; no GPU work."""
    from torch._dynamo.backends import registry

    graphs = []

    def cpu_graph_backend(graph, inputs, *, config_patches):
        assert config_patches == anysolv.STRICT_INDUCTOR_OPTIONS
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(torch, "compile", mocked_cuda.real_compile)
    monkeypatch.setattr(
        registry,
        "lookup_backend",
        lambda name: cpu_graph_backend if name == "inductor" else name,
    )
    runtime = _cuda_runtime(mocked_cuda)
    # Keep real Dynamo RNG preservation entirely CPU-side after policy setup.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialization")
    )
    assert _evaluate(runtime)[0] == 2.5
    assert runtime.compilation_active and graphs
    executed = runtime._compiled_backend_executions
    assert _evaluate(runtime, displacement=0.01)[0] == 2.5
    assert runtime._compiled_backend_executions > executed


def test_compiled_activation_callback_does_not_retain_runtime(monkeypatch):
    from torch._dynamo.backends import registry

    runtime = object.__new__(anysolv._OfficialAnySolvRuntime)
    runtime._compiled_backend_executions = 0
    runtime._resident = SimpleNamespace(
        model=SimpleNamespace(_raw_forward=lambda x: x + 1)
    )
    monkeypatch.setattr(
        registry,
        "lookup_backend",
        lambda name: lambda graph, inputs, **kwargs: (lambda x: x + 1),
    )
    # Simulate Dynamo's global cache retaining the backend executable, not the
    # transient source bound method. It must not pin the 1.165 GB model owner.
    cached = []

    def compile(fn, *, backend, dynamic):
        executable = backend(None, [])
        cached.append(executable)
        return executable

    monkeypatch.setattr(torch, "compile", compile)
    runtime._attach_owned_compile()
    token = anysolv._ACTIVE_COMPILED_OWNER.set(weakref.ref(runtime))
    try:
        assert runtime._compiled_run(torch.tensor(1)).item() == 2
    finally:
        anysolv._ACTIVE_COMPILED_OWNER.reset(token)
    assert runtime._compiled_backend_executions == 1
    owner = weakref.ref(runtime)
    del runtime
    gc.collect()
    assert owner() is None
    with pytest.raises(RuntimeError, match="owner has been released"):
        cached[0](torch.tensor(1))
