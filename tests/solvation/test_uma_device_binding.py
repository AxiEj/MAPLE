from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from maple.function.calculator.uma import _uma_calculator as uma_module
from maple.function.calculator.uma._uma_calculator import UMACalculator

FAIRCHEM_DEVICE_ENV = "CURRRENT_DEVICE_TYPE"


def _select_local_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "uma.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        UMACalculator,
        "_prepare_compat_checkpoint",
        staticmethod(lambda _name, path: str(path)),
    )
    return checkpoint


def test_cuda_ordinal_is_bound_while_fairchem_receives_literal_cuda(
    monkeypatch, tmp_path
):
    checkpoint = _select_local_checkpoint(monkeypatch, tmp_path)
    events = []

    @contextmanager
    def fake_cuda_device(index):
        events.append(("enter", index))
        try:
            yield
        finally:
            events.append(("exit", index))

    def fake_load(path, **kwargs):
        assert path == str(checkpoint)
        assert kwargs["device"] == "cuda"
        assert uma_module.os.environ[FAIRCHEM_DEVICE_ENV] == "cuda"
        events.append(("load", kwargs["device"]))
        return SimpleNamespace(device="cuda:2")

    monkeypatch.setattr(
        UMACalculator, "_normalize_device", staticmethod(lambda _device: "cuda:2")
    )
    monkeypatch.setattr(uma_module.torch.cuda, "device", fake_cuda_device)
    monkeypatch.setattr(uma_module, "load_predict_unit", fake_load)
    monkeypatch.delenv(FAIRCHEM_DEVICE_ENV, raising=False)

    predictor = UMACalculator._build_predictor(
        "local", None, "cuda:2", checkpoint_path=str(checkpoint)
    )

    assert predictor.device == "cuda:2"
    assert events == [("enter", 2), ("load", "cuda"), ("exit", 2)]
    assert FAIRCHEM_DEVICE_ENV not in uma_module.os.environ


@pytest.mark.parametrize("initial", [None, "cpu", "cuda"])
def test_fairchem_device_environment_is_restored(monkeypatch, tmp_path, initial):
    checkpoint = _select_local_checkpoint(monkeypatch, tmp_path)

    @contextmanager
    def fake_cuda_device(_index):
        yield

    monkeypatch.setattr(
        UMACalculator, "_normalize_device", staticmethod(lambda _device: "cuda:1")
    )
    monkeypatch.setattr(uma_module.torch.cuda, "device", fake_cuda_device)
    monkeypatch.setattr(
        uma_module,
        "load_predict_unit",
        lambda *_args, **_kwargs: SimpleNamespace(device="cuda:1"),
    )
    if initial is None:
        monkeypatch.delenv(FAIRCHEM_DEVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(FAIRCHEM_DEVICE_ENV, initial)

    UMACalculator._build_predictor(
        "local", None, "cuda:1", checkpoint_path=str(checkpoint)
    )

    assert uma_module.os.environ.get(FAIRCHEM_DEVICE_ENV) == initial


def test_fairchem_device_environment_is_restored_when_loading_raises(
    monkeypatch, tmp_path
):
    checkpoint = _select_local_checkpoint(monkeypatch, tmp_path)

    @contextmanager
    def fake_cuda_device(_index):
        yield

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        UMACalculator, "_normalize_device", staticmethod(lambda _device: "cuda:3")
    )
    monkeypatch.setattr(uma_module.torch.cuda, "device", fake_cuda_device)
    monkeypatch.setattr(uma_module, "load_predict_unit", fail_load)
    monkeypatch.setenv(FAIRCHEM_DEVICE_ENV, "cpu")

    with pytest.raises(RuntimeError, match="load failed"):
        UMACalculator._build_predictor(
            "local", None, "cuda:3", checkpoint_path=str(checkpoint)
        )

    assert uma_module.os.environ[FAIRCHEM_DEVICE_ENV] == "cpu"


def test_predictor_device_mismatch_is_rejected_and_environment_restored(
    monkeypatch, tmp_path
):
    checkpoint = _select_local_checkpoint(monkeypatch, tmp_path)

    @contextmanager
    def fake_cuda_device(_index):
        yield

    monkeypatch.setattr(
        UMACalculator, "_normalize_device", staticmethod(lambda _device: "cuda:2")
    )
    monkeypatch.setattr(uma_module.torch.cuda, "device", fake_cuda_device)
    monkeypatch.setattr(
        uma_module,
        "load_predict_unit",
        lambda *_args, **_kwargs: SimpleNamespace(device="cuda:0"),
    )
    monkeypatch.setenv(FAIRCHEM_DEVICE_ENV, "cpu")

    with pytest.raises(RuntimeError, match="cuda:2.*cuda:0"):
        UMACalculator._build_predictor(
            "local", None, "cuda:2", checkpoint_path=str(checkpoint)
        )

    assert uma_module.os.environ[FAIRCHEM_DEVICE_ENV] == "cpu"


def test_cpu_path_keeps_literal_device_and_does_not_enter_cuda_context(
    monkeypatch, tmp_path
):
    checkpoint = _select_local_checkpoint(monkeypatch, tmp_path)

    def fail_cuda_context(_index):
        raise AssertionError("CPU predictor construction must not enter a CUDA context")

    def fake_load(_path, **kwargs):
        assert kwargs["device"] == "cpu"
        return SimpleNamespace(device="cpu")

    monkeypatch.setattr(
        UMACalculator, "_normalize_device", staticmethod(lambda _device: "cpu")
    )
    monkeypatch.setattr(uma_module.torch.cuda, "device", fail_cuda_context)
    monkeypatch.setattr(uma_module, "load_predict_unit", fake_load)
    monkeypatch.setenv(FAIRCHEM_DEVICE_ENV, "cuda")

    predictor = UMACalculator._build_predictor(
        "local", None, "cpu", checkpoint_path=str(checkpoint)
    )

    assert predictor.device == "cpu"
    assert uma_module.os.environ[FAIRCHEM_DEVICE_ENV] == "cuda"
