"""Public fail-closed contract for the private dual-device Torch-FP32 CHA route."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from maple.function.calculator.extra_correction.implicit import private_torch_chagb
from maple.function.calculator.extra_correction.implicit.correction import (
    _validate_direct_contract,
    _validate_provider_identity,
)
from maple.function.read.command_control import CommandControl


def _solvation_line() -> str:
    return (
        "#solv(implicit=water,method=gb,model=chagb,dtype=float32,"
        "prepared_topology=prepared.json,"
        f"prepared_topology_sha256={'a' * 64},experimental=true)"
    )


def _parse(*extra: str) -> dict:
    lines = ["#model=ani2x", *extra, "#sp", "#charge(source=prepared)", _solvation_line()]
    return CommandControl.from_settings(lines).as_dict()


def _direct_options() -> dict:
    return {
        "implicit": "water",
        "method": "gb",
        "model": "chagb",
        "profile": "chagb-bondi-pbsa-inp2",
        "nonpolar": "cavity-dispersion",
        "dtype": "float32",
        "prepared_topology": "prepared.json",
        "prepared_topology_sha256": "a" * 64,
        "experimental": True,
    }


def test_omitted_device_preserves_maple_cpu_default():
    assert _parse().get("device", "cpu") == "cpu"


def test_explicit_cpu_command_is_preserved():
    assert _parse("#device=cpu")["device"] == "cpu"


@pytest.mark.parametrize("option", ["platform", "precision", "device_index", "opencl_platform_index"])
def test_private_openmm_option_error_describes_shared_torch_device(option):
    options = {**_direct_options(), option: "unused"}
    with pytest.raises(ValueError, match="Torch device.*rejects OpenMM execution options"):
        _validate_provider_identity(options, "gb", "private-torch-cha")


def test_explicit_cuda_device_is_preserved():
    assert _parse("#device=cuda:1")["device"] == "cuda:1"


@pytest.mark.parametrize("device", ["gpu", "gpu0", "gpu:0", "auto"])
def test_shared_maple_device_syntax_is_not_overridden(device):
    assert _parse(f"#device={device}")["device"] == device


@pytest.mark.parametrize("task", ["opt", "freq", "ts", "md"])
def test_private_torch_contract_rejects_derivative_and_md_tasks(task):
    with pytest.raises(ValueError, match="single-point energy-only"):
        _validate_direct_contract(_direct_options(), task)


def test_unified_release_identity_is_fail_closed_until_sealed():
    assert private_torch_chagb._RELEASE_DATA_PATH.name == (
        "private_cha_torch_fp32_release_v1.json"
    )
    pin = private_torch_chagb._PINNED_RELEASE_FILE_SHA256
    if pin is None:
        with pytest.raises(RuntimeError, match="not sealed"):
            private_torch_chagb._load_release_contract()
    else:
        assert len(pin) == 64
        private_torch_chagb._load_release_contract()
    assert Path(private_torch_chagb._RELEASE_DATA_PATH).is_file()


def test_canonical_torch_device_is_not_resolved_again(monkeypatch):
    canonical = torch.device("cpu")
    monkeypatch.setattr(
        private_torch_chagb,
        "resolve_torch_device",
        lambda value: pytest.fail(f"canonical device was re-resolved: {value!r}"),
    )
    assert private_torch_chagb._resolve_private_torch_device(canonical) is canonical


@pytest.mark.parametrize("raw", [None, "cpu", "auto", "gpu", "gpu:0"])
def test_direct_raw_device_is_resolved_exactly_once(raw, monkeypatch):
    selected = torch.device("cpu")
    calls = []

    def resolve(value):
        calls.append(value)
        return selected

    monkeypatch.setattr(private_torch_chagb, "resolve_torch_device", resolve)
    assert private_torch_chagb._resolve_private_torch_device(raw) is selected
    assert calls == [raw]


@pytest.mark.parametrize("device", [torch.device("mps"), torch.device("xpu:0")])
def test_resolved_unsupported_device_fails_without_fallback(device, monkeypatch):
    monkeypatch.setattr(
        private_torch_chagb,
        "resolve_torch_device",
        lambda value: pytest.fail("canonical device was re-resolved"),
    )
    with pytest.raises(ValueError, match="CPU and CUDA"):
        private_torch_chagb._resolve_private_torch_device(device)


def test_direct_default_cpu_does_not_query_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("CPU default queried CUDA availability"),
    )
    assert private_torch_chagb._resolve_private_torch_device(None) == torch.device("cpu")


def test_unavailable_explicit_cuda_fails_before_private_release_io(monkeypatch):
    monkeypatch.setattr(
        private_torch_chagb,
        "resolve_torch_device",
        lambda value: (_ for _ in ()).throw(ValueError("CUDA is not available")),
    )
    monkeypatch.setattr(
        private_torch_chagb,
        "_load_release_contract",
        lambda: pytest.fail("private release was read after device failure"),
    )
    with pytest.raises(ValueError, match="CUDA is not available"):
        private_torch_chagb.PrivateTorchChaGB._initialize_loaded(
            object.__new__(private_torch_chagb.PrivateTorchChaGB),
            prepared_topology="unused.json",
            prepared_topology_sha256="a" * 64,
            device="cuda:0",
        )
