from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError

import pytest
import torch

from maple.function.calculator import solvation_execution as module
from maple.function.calculator.solvation_execution import (
    SolvationExecution,
    is_pretrained_solvent_route,
    normalize_execution_options,
)


def _enable_synthetic_cuda(monkeypatch, *, count=2):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Synthetic GPU")


def test_normalize_options_does_not_import_or_query_torch(monkeypatch):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        imported.append(name)
        if name == "torch":
            raise AssertionError("header normalization imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert normalize_execution_options({}) == {
        "dtype": "float32",
        "inference": "reference",
    }
    assert normalize_execution_options(
        {"dtype": "torch.float64", "inference": "REFERENCE"}
    ) == {"dtype": "float64", "inference": "reference"}
    assert "torch" not in imported


@pytest.mark.parametrize(
    "model,solvation_options,expected",
    [
        ("consolv", None, True),
        ("aimnet2", {"method": "anysolv"}, True),
        ("aimnet2", {"method": "gb"}, False),
        ("aimnet2", None, False),
    ],
)
def test_route_detection_is_limited_to_the_two_pretrained_solvent_routes(
    model, solvation_options, expected
):
    assert is_pretrained_solvent_route(model, solvation_options) is expected


@pytest.mark.parametrize(
    "spelling,canonical",
    [
        ("cuda", "cuda:0"),
        ("cuda0", "cuda:0"),
        ("cuda:1", "cuda:1"),
        ("gpu", "cuda:0"),
        ("gpu0", "cuda:0"),
        ("gpu:1", "cuda:1"),
    ],
)
def test_cuda_aliases_normalize_without_running_a_gpu_kernel(
    monkeypatch, spelling, canonical
):
    _enable_synthetic_cuda(monkeypatch)

    execution = SolvationExecution(device=spelling)

    assert execution.device == canonical
    assert execution.dtype == "float32"
    assert execution.inference == "reference"
    assert execution.torch_dtype is torch.float32


@pytest.mark.parametrize("dtype", ["float32", "float64", "torch.float64"])
def test_explicit_cpu_preserves_requested_supported_precision(monkeypatch, dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    execution = SolvationExecution(device="cpu", dtype=dtype)

    assert execution.device == "cpu"
    assert execution.dtype == dtype.removeprefix("torch.")
    assert execution.runtime_context() is None


def test_execution_configuration_is_constructor_immutable(monkeypatch):
    _enable_synthetic_cuda(monkeypatch)
    execution = SolvationExecution()

    with pytest.raises(FrozenInstanceError):
        execution.device = "cpu"


def test_default_execution_is_cuda_float32_reference(monkeypatch):
    _enable_synthetic_cuda(monkeypatch)

    execution = SolvationExecution()

    assert (execution.device, execution.dtype, execution.inference) == (
        "cuda:0",
        "float32",
        "reference",
    )


@pytest.mark.parametrize("device", ["mps", "xpu", "rocm", "cuda:-1"])
def test_unknown_accelerators_are_rejected(monkeypatch, device):
    _enable_synthetic_cuda(monkeypatch)

    with pytest.raises(ValueError, match="explicit CPU or CUDA"):
        SolvationExecution(device=device)


def test_unavailable_cuda_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="unavailable.*fallback is disabled"):
        SolvationExecution(device="cuda:0")


def test_out_of_range_cuda_index_is_rejected(monkeypatch):
    _enable_synthetic_cuda(monkeypatch, count=1)

    with pytest.raises(ValueError, match="CUDA device 1 is unavailable"):
        SolvationExecution(device="cuda:1")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float128", "none"])
def test_unsupported_scalar_precision_is_rejected(dtype):
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        SolvationExecution(device="cpu", dtype=dtype)


def test_unknown_inference_mode_is_rejected():
    with pytest.raises(ValueError, match="inference must be reference or optimized"):
        SolvationExecution(device="cpu", inference="turbo")


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", "float32"), ("cpu", "float64"), ("cuda", "float64")],
)
def test_optimized_mode_requires_cuda_float32(monkeypatch, device, dtype):
    _enable_synthetic_cuda(monkeypatch)

    with pytest.raises(ValueError, match="requires CUDA and float32"):
        SolvationExecution(device=device, dtype=dtype, inference="optimized")


def test_optimized_cuda_float32_is_an_explicit_configuration(monkeypatch):
    _enable_synthetic_cuda(monkeypatch)

    execution = SolvationExecution(
        device="cuda:1", dtype="float32", inference="optimized"
    )

    assert execution.provenance == {
        "device": "cuda:1",
        "compute_dtype": "float32",
        "inference_mode": "optimized",
        "matmul_precision": "highest",
        "tf32": False,
        "autocast": False,
    }


def test_runtime_context_records_live_hardware_versions_and_precision(
    monkeypatch,
):
    _enable_synthetic_cuda(monkeypatch)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: False)
    monkeypatch.setattr(module.metadata, "version", lambda name: f"{name}-version")
    execution = SolvationExecution(device="cuda:1")

    context = execution.runtime_context()

    assert context["accelerator_backend"] == "cuda"
    assert context["hardware"] == "Synthetic GPU"
    assert context["software_versions"]["torch"] == torch.__version__
    assert context["software_versions"]["e3nn"] == "e3nn-version"
    assert context["precision_controls"]["compute_dtype"] == "float32"
    assert context["precision_controls"]["matmul_precision"] == "highest"
    assert all(
        value is False
        for name, value in context["precision_controls"].items()
        if name not in {"compute_dtype", "matmul_precision"}
    )


@pytest.mark.parametrize(
    "matmul_precision,tf32,autocast",
    [
        ("medium", False, False),
        ("highest", True, False),
        ("highest", False, True),
    ],
)
def test_runtime_context_rejects_reduced_or_implicit_precision(
    monkeypatch, matmul_precision, tf32, autocast
):
    _enable_synthetic_cuda(monkeypatch)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: matmul_precision)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: autocast)
    execution = SolvationExecution()

    with pytest.raises(ValueError, match="highest.*TF32 off.*autocast off"):
        execution.runtime_context()


def test_validate_card_forwards_live_context_without_changing_controls(monkeypatch):
    _enable_synthetic_cuda(monkeypatch)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: False)
    monkeypatch.setattr(module.metadata, "version", lambda name: "1")
    received = {}

    class Card:
        def validate_device(self, device, **kwargs):
            received.update(device=device, **kwargs)

    execution = SolvationExecution(device="gpu0", inference="reference")
    execution.validate_card(Card(), task="sp")

    assert received["device"] == "cuda:0"
    assert received["task"] == "sp"
    assert received["inference_mode"] == "reference"
    assert received["runtime_context"]["hardware"] == "Synthetic GPU"


@pytest.mark.parametrize("device", [None, "none", "NONE", ""])
def test_missing_device_is_rejected_instead_of_becoming_cpu(device):
    with pytest.raises(ValueError, match="explicit CPU or CUDA"):
        SolvationExecution(device=device)


def _caller_flags():
    return (torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_owned_context_restores_hostile_precision_and_autocast(monkeypatch, fail, device):
    _enable_synthetic_cuda(monkeypatch)
    execution = SolvationExecution(device=device)
    original = _caller_flags()
    try:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        hostile = _caller_flags()
        # Real thread-local autocast nesting; no CUDA tensors/kernels.
        device_type = device.split(":")[0]
        autocast_dtype = torch.bfloat16 if device_type == "cpu" else torch.float16
        with torch.autocast(device_type, dtype=autocast_dtype):
            try:
                with execution.precision_context():
                    state = execution.check_precision()
                    assert state["matmul_precision"] == "highest"
                    assert not state["matmul_tf32"] and not state["cudnn_tf32"]
                    assert not state["autocast"]
                    if device == "cpu":
                        assert not torch.is_autocast_enabled("cpu")
                    # RLock permits nested card/report contexts on the owner.
                    with execution.precision_context():
                        execution.check_precision()
                    if fail:
                        raise RuntimeError("model failure")
            except RuntimeError as exc:
                assert fail and str(exc) == "model failure"
            assert torch.is_autocast_enabled(device_type)
        assert _caller_flags() == hostile
    finally:
        torch.set_float32_matmul_precision(original[0])
        torch.backends.cuda.matmul.allow_tf32 = original[1]
        torch.backends.cudnn.allow_tf32 = original[2]


def test_owned_context_serializes_threads_and_restores_caller():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    execution = SolvationExecution(device="cpu")
    entered, attempted, second_entered, release = (Event() for _ in range(4))
    original = _caller_flags()

    def first():
        with execution.precision_context():
            entered.set()
            assert release.wait(5)
            execution.check_precision()

    def second():
        assert entered.wait(5)
        attempted.set()
        with execution.precision_context():
            second_entered.set()
            execution.check_precision()

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(first)
        contender = pool.submit(second)
        try:
            assert attempted.wait(5)
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        owner.result(timeout=5)
        contender.result(timeout=5)
    assert second_entered.is_set()
    assert _caller_flags() == original


def test_live_cudnn_tf32_is_rejected_and_reported(monkeypatch):
    _enable_synthetic_cuda(monkeypatch)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    execution = SolvationExecution()
    assert execution.precision_state()["cudnn_tf32"] is True
    with pytest.raises(ValueError, match="cuDNN TF32 off"):
        execution.runtime_context()


def test_fast_precision_check_never_looks_up_package_versions(monkeypatch):
    monkeypatch.setattr(module.metadata, "version", lambda _: pytest.fail("version lookup"))
    execution = SolvationExecution(device="cpu")
    with execution.precision_context():
        execution.check_precision(inputs={"r": torch.zeros(2, 3)})
        execution.check_precision()


@pytest.mark.parametrize("location", ["input", "parameter", "buffer"])
def test_check_precision_rejects_mixed_floating_dtype(location):
    execution = SolvationExecution(device="cpu")
    model = torch.nn.Module()
    inputs = ()
    if location == "input":
        inputs = {"r": torch.ones(1, dtype=torch.float16)}
    elif location == "parameter":
        model.register_parameter("weight", torch.nn.Parameter(torch.ones(1, dtype=torch.float64)))
    else:
        model.register_buffer("weight", torch.ones(1, dtype=torch.bfloat16))
    with execution.precision_context(), pytest.raises(ValueError, match="selected dtype"):
        execution.check_precision(inputs=inputs, model=model)


def test_optimized_compiler_flags_are_explicit_scoped_and_reported(monkeypatch):
    import torch._inductor.config as config

    _enable_synthetic_cuda(monkeypatch)
    monkeypatch.setattr(module.metadata, "version", lambda _: "test")
    execution = SolvationExecution(inference="optimized")
    hostile = {key: not value for key, value in module.STRICT_INDUCTOR_OPTIONS.items()}
    with config.patch(hostile):
        with execution.precision_context():
            state = execution.check_precision()
            assert state["compiler_options"] == module.STRICT_INDUCTOR_OPTIONS
            context = execution.runtime_context()
            assert context["precision_controls"]["fast_math"] is False
            assert context["precision_protocol"]["relaxed_convergence"] == "inapplicable_direct_feed_forward_sp"
            assert context["precision_protocol"]["shortened_sampling"] == "inapplicable_direct_feed_forward_sp"
            assert context["precision_state"]["available_reduced_precision_flags"] == state["available_reduced_precision_flags"]
        assert {key: getattr(config, key) for key in hostile} == hostile


def test_card_validation_owns_strict_context_and_restores_after_failure(monkeypatch):
    _enable_synthetic_cuda(monkeypatch)
    monkeypatch.setattr(module.metadata, "version", lambda _: "test")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    execution = SolvationExecution()
    before = _caller_flags()

    class Card:
        def validate_device(self, device, **kwargs):
            execution.check_precision()
            assert kwargs["runtime_context"]["precision_state"]["cudnn_tf32"] is False
            raise ValueError("card rejected")

    with pytest.raises(ValueError, match="card rejected"):
        execution.validate_card(Card())
    assert _caller_flags() == before


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_restore_exact_per_operator_overrides_and_legacy_precision(precision):
    execution = SolvationExecution(device="cpu")
    scopes = (torch.backends.cuda.matmul, torch.backends.mkldnn.matmul,
              torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
    original = tuple(scope.fp32_precision for scope in scopes)
    old_matmul, _, old_cudnn = _caller_flags()
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = True
        overrides = tuple(scope.fp32_precision for scope in scopes)
        before = _caller_flags()
        with execution.precision_context():
            execution.check_precision()
        assert tuple(scope.fp32_precision for scope in scopes) == overrides
        assert _caller_flags() == before
    finally:
        torch.set_float32_matmul_precision(old_matmul)
        torch.backends.cudnn.allow_tf32 = old_cudnn
        for scope, value in zip(scopes, original):
            scope.fp32_precision = value
