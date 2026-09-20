from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms

from maple.function.read.command_control import CommandControl
from maple.function.read.input_reader import InputReader


def test_cpu_frequency_import_and_eigh_without_torch_module(tmp_path):
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules['torch'] = None
        import numpy as np
        from ase import Atoms
        from maple.function.dispatcher.frequency.frequency import MWFrequency

        atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        job = MWFrequency({str(tmp_path / 'torchless-frequency.out')!r}, atoms, device='cpu')
        frequencies, modes = job.compute_frequencies(np.eye(6))
        assert np.all(np.isfinite(frequencies))
        assert modes.shape == (6, 6)
        try:
            MWFrequency({str(tmp_path / 'torchless-cuda.out')!r}, atoms, device='cuda:0')
        except ValueError as exc:
            assert 'PyTorch is not installed' in str(exc)
        else:
            raise AssertionError('explicit CUDA unexpectedly fell back without PyTorch')
        print('torchless_cpu_frequency_ok')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "torchless_cpu_frequency_ok" in completed.stdout


def _device_from(*lines: str) -> str:
    return CommandControl.from_settings(list(lines)).as_dict()["device"]


@pytest.mark.parametrize(
    "lines",
    [
        ("#model=ani2x(hessian=numerical)", "#device=cuda", "#freq"),
        ("#model=ani2x(hessian=numerical)", "#freq", "#device=cuda"),
    ],
)
def test_global_device_is_not_overwritten_by_task_defaults(lines):
    assert _device_from(*lines) == "cuda"


@pytest.mark.parametrize(
    "lines",
    [
        ("#model=ani2x(hessian=numerical)", "#device=cuda", "#freq(device=cpu)"),
        ("#model=ani2x(hessian=numerical)", "#freq(device=cpu)", "#device=cuda"),
    ],
)
def test_explicit_task_device_overrides_global_device_independent_of_order(lines):
    assert _device_from(*lines) == "cpu"


def test_unspecified_device_remains_cpu():
    reader = InputReader()
    assert reader._resolve_device(None) == torch.device("cpu")


def test_auto_preserves_optional_accelerator_semantics(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert InputReader._resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert InputReader._resolve_device("auto") == torch.device("cuda:0")


@pytest.mark.parametrize("name", ["gpufoo", "cuda:abc", "cuda:-1", "cuda:0:1"])
def test_malformed_accelerator_device_is_rejected(name):
    with pytest.raises(ValueError, match="Invalid device"):
        InputReader._resolve_device(name)


def test_explicit_cuda_request_fails_if_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA.*not available"):
        InputReader._resolve_device("cuda")


def test_explicit_cuda_index_must_exist(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="index 2.*2 device"):
        InputReader._resolve_device("cuda:2")


def test_cuda_aliases_resolve_to_canonical_torch_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert InputReader._resolve_device("cuda") == torch.device("cuda:0")
    assert InputReader._resolve_device("gpu2") == torch.device("cuda:2")


def test_explicit_mps_request_fails_if_mps_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
    )
    with pytest.raises(ValueError, match="MPS.*not available"):
        InputReader._resolve_device("mps")


def test_explicit_xpu_request_fails_if_xpu_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    with pytest.raises(ValueError, match="XPU.*not available"):
        InputReader._resolve_device("xpu")


def test_xpu_device_index_is_validated_and_preserved(monkeypatch):
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
    )
    assert InputReader._resolve_device("xpu:1") == torch.device("xpu:1")
    with pytest.raises(ValueError, match="XPU device index 2.*2 device"):
        InputReader._resolve_device("xpu:2")


def _openmm_gb_params(*options: str):
    return CommandControl.from_settings(
        [
            "#model=ani2x",
            "#sp",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=gb,provider=openmm,experimental=true,"
            + ",".join(options)
            + ")",
        ]
    ).as_dict()["solv"]


def test_openmm_execution_options_are_normalized_and_preserved():
    options = _openmm_gb_params(
        "platform=OpenCL",
        "precision=DOUBLE",
        'device_index="0, 1"',
        "opencl_platform_index=0",
    )
    assert options["platform"] == "opencl"
    assert options["precision"] == "double"
    assert options["device_index"] == "0,1"
    assert options["opencl_platform_index"] == 0


def test_openmm_multi_gpu_device_index_accepts_unquoted_input_syntax():
    options = _openmm_gb_params("platform=CUDA", "device_index=0, 2")
    assert options["device_index"] == "0,2"


@pytest.mark.parametrize("platform", ["auto", "CPU", "Reference", "CUDA", "HIP", "OpenCL"])
def test_openmm_platform_allowlist(platform):
    assert _openmm_gb_params(f"platform={platform}")["platform"] == platform.lower()


@pytest.mark.parametrize(
    "option, message",
    [
        ("platform=metal", "platform"),
        ("precision=half", "precision"),
        ("device_index=-1", "device_index"),
        ("device_index=gpu0", "device_index"),
        ("opencl_platform_index=-1", "opencl_platform_index"),
    ],
)
def test_invalid_openmm_execution_options_are_rejected(option, message):
    with pytest.raises(ValueError, match=message):
        _openmm_gb_params(option)


@pytest.mark.parametrize(
    "solv_line",
    [
        "#solv(implicit=water,method=gb,provider=ambertools,precision=double,experimental=true)",
        "#solv(implicit=water,method=pb,provider=apbs,device_index=0,experimental=true)",
        (
            "#solv(implicit=water,method=pb,provider=ddx,opencl_platform_index=0,"
            "solvent_kappa_inverse_angstrom=0.1,experimental=true)"
        ),
    ],
)
def test_openmm_execution_options_are_rejected_by_other_providers(solv_line):
    with pytest.raises(ValueError, match="only.*provider=openmm"):
        CommandControl.from_settings(
            ["#model=ani2x", "#sp", "#charge(source=mol2)", solv_line]
        )


def test_uma_preserves_explicit_cuda_ordinal(monkeypatch):
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert UMACalculator._normalize_device("cuda") == "cuda:0"
    assert UMACalculator._normalize_device("cuda:2") == "cuda:2"


def test_uma_rejects_cuda_when_cuda_is_unavailable(monkeypatch):
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA.*not available"):
        UMACalculator._normalize_device("cuda:2")


def test_uma_validates_cuda_index(monkeypatch):
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="index 2.*2 device"):
        UMACalculator._normalize_device("cuda:2")


@pytest.mark.parametrize("device", ["mps", "xpu", "gpu0", "cuda:abc", None])
def test_uma_rejects_unsupported_or_malformed_devices(device):
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    with pytest.raises(ValueError, match="UMA device"):
        UMACalculator._normalize_device(device)


def test_frequency_diagonalization_device_is_separate_from_model_device():
    params = CommandControl.from_settings(
        [
            "#model=ani2x(hessian=numerical)",
            "#device=cuda",
            "#freq(diagonalization_device=cpu)",
        ]
    ).as_dict()
    assert params["device"] == "cuda"
    assert params["diagonalization_device"] == "cpu"


@pytest.mark.parametrize(
    "paras, expected",
    [
        ({"method": "mw", "device": "cuda:1"}, "cuda:1"),
        (
            {
                "method": "mw",
                "device": "cuda:1",
                "diagonalization_device": "cpu",
            },
            "cpu",
        ),
    ],
)
def test_frequency_driver_forwards_resolved_diagonalization_selection(
    monkeypatch, tmp_path, paras, expected
):
    import maple.function.dispatcher.frequency.frequency as frequency_module

    captured = {}

    class FakeFrequencyJob:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(frequency_module, "MWFrequency", FakeFrequencyJob)
    driver = frequency_module.Frequency(
        str(tmp_path / "driver.out"),
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        paras=paras,
    )
    driver.run()
    assert captured["device"] == expected


def _mw_frequency(tmp_path, device):
    from maple.function.dispatcher.frequency.frequency import MWFrequency

    return MWFrequency(
        str(tmp_path / "frequency-device.out"),
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        device=device,
    )


def test_frequency_rejects_unavailable_explicit_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA.*not available"):
        _mw_frequency(tmp_path, "cuda:0")


def test_frequency_rejects_malformed_device(tmp_path):
    with pytest.raises(ValueError, match="Invalid device"):
        _mw_frequency(tmp_path, "cuda:abc")


def test_frequency_accepts_available_xpu_ordinal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
    )
    assert _mw_frequency(tmp_path, "xpu:1").device == "xpu:1"


def test_frequency_rejects_mps_for_float64_eigensolve(monkeypatch, tmp_path):
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
    )
    with pytest.raises(ValueError, match="MPS.*float64"):
        _mw_frequency(tmp_path, "mps")


def test_frequency_xpu_eigh_uses_the_resolved_ordinal(monkeypatch, tmp_path):
    import numpy as np

    import maple.function.dispatcher.frequency.frequency as frequency_module

    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
    )
    job = _mw_frequency(tmp_path, "xpu:1")
    calls = {}

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    def fake_tensor(value, *, dtype, device):
        calls["device"] = device
        calls["dtype"] = dtype
        return FakeTensor(value)

    monkeypatch.setattr(frequency_module.torch, "tensor", fake_tensor)
    monkeypatch.setattr(
        frequency_module.torch.linalg,
        "eigh",
        lambda tensor: (FakeTensor([1.0, 2.0]), FakeTensor(np.eye(2))),
    )

    values, vectors = job._eigh(np.diag([1.0, 2.0]))
    assert calls == {"device": "xpu:1", "dtype": torch.float64}
    assert values.tolist() == [1.0, 2.0]
    assert vectors.tolist() == np.eye(2).tolist()
