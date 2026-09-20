from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from maple.function.calculator.calculator_base import CalcABC, calculator_execution
from maple.function.calculator.extra_correction.implicit.openmm_execution import (
    resolve_openmm_platform,
)


class Platform:
    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def getPropertyNames(self):
        if self.name == "CPU":
            return ("Threads", "DeterministicForces")
        if self.name in {"CUDA", "HIP"}:
            return ("Precision", "DeviceIndex", "DeterministicForces")
        if self.name == "OpenCL":
            return ("Precision", "DeviceIndex", "OpenCLPlatformIndex")
        return ()


@pytest.fixture
def platforms(monkeypatch):
    import openmm

    names = ["Reference", "CPU", "CUDA", "HIP", "OpenCL"]
    registry = SimpleNamespace(
        getNumPlatforms=lambda: len(names),
        getPlatform=lambda index: Platform(names[index]),
        getPlatformByName=lambda name: Platform(name),
    )
    monkeypatch.setattr(openmm, "Platform", registry)
    return names


def test_explicit_cpu_preserves_existing_platform_controls(platforms):
    selected, properties, record = resolve_openmm_platform("CPU", model_device="cuda:2")
    assert selected.getName() == "CPU"
    assert properties == {"Threads": "1", "DeterministicForces": "true"}
    assert record["selection_reason"] == "explicit_platform"


def test_auto_cuda_follows_model_ordinal_with_double_precision(platforms, monkeypatch):
    monkeypatch.setattr(sys.modules["torch"].version, "hip", None)
    selected, properties, record = resolve_openmm_platform(
        "auto", model_device="cuda:2"
    )
    assert selected.getName() == "CUDA"
    assert properties == {
        "Precision": "double",
        "DeviceIndex": "2",
        "DeterministicForces": "true",
    }
    assert record["gpu_requested_but_unavailable"] is False


def test_auto_rocm_uses_hip_not_cuda(platforms, monkeypatch):
    monkeypatch.setattr(sys.modules["torch"].version, "hip", "7.0")
    selected, properties, _ = resolve_openmm_platform("auto", model_device="cuda:1")
    assert selected.getName() == "HIP"
    assert properties["DeviceIndex"] == "1"


@pytest.mark.parametrize("hip,selected", [(None, "HIP"), ("7.0", "CUDA")])
def test_cross_backend_device_namespaces_are_not_inferred(
    platforms, monkeypatch, hip, selected
):
    monkeypatch.setattr(sys.modules["torch"].version, "hip", hip)
    _, properties, record = resolve_openmm_platform(selected, model_device="cuda:2")
    assert "DeviceIndex" not in properties
    assert record["device_index_source"] == "native_default"


@pytest.mark.parametrize("device", ["mps", "xpu:1"])
def test_auto_does_not_guess_opencl_device_from_another_runtime(platforms, device):
    selected, _, record = resolve_openmm_platform("auto", model_device=device)
    assert selected.getName() == "CPU"
    assert record["gpu_requested_but_unavailable"] is True


@pytest.mark.parametrize(
    "reported,parameter", [("cpu", "cpu"), ("cuda:1", "cpu"), ("cuda:1", "cuda:0")]
)
def test_factory_rejects_known_device_substitution(tmp_path, reported, parameter):
    from maple.function.calculator.set_calculator import SetCalculator

    backend = SimpleNamespace(
        device=reported,
        model=SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=parameter)])
        ),
    )
    factory = SetCalculator("cuda:1", "test", str(tmp_path / "test.out"))
    with pytest.raises(ValueError, match="device mismatch"):
        factory._validate_execution_placement(backend)


def test_explicit_opencl_records_its_own_platform_and_device_indices(platforms):
    selected, properties, _ = resolve_openmm_platform(
        "OpenCL",
        model_device="cuda:2",
        precision="double",
        device_index="0,1",
        opencl_platform_index=1,
    )
    assert selected.getName() == "OpenCL"
    assert properties == {
        "Precision": "double",
        "DeviceIndex": "0,1",
        "OpenCLPlatformIndex": "1",
    }


def test_explicit_unavailable_gpu_never_falls_back(platforms):
    platforms.remove("CUDA")
    with pytest.raises(ValueError, match="available"):
        resolve_openmm_platform("CUDA")


def test_auto_unavailable_gpu_is_an_explicitly_recorded_hybrid(platforms, monkeypatch):
    monkeypatch.setattr(sys.modules["torch"].version, "hip", None)
    platforms.remove("CUDA")
    selected, _, record = resolve_openmm_platform("auto", model_device="cuda:0")
    assert selected.getName() == "CPU"
    assert record["gpu_requested_but_unavailable"] is True
    assert record["selection_reason"] == "requested_accelerator_unavailable"


@pytest.mark.parametrize("value", [-1, True, 1.2, "0,0", "-1", "x", "0,"])
def test_device_indices_are_not_silently_reinterpreted(platforms, value):
    with pytest.raises((TypeError, ValueError), match="device_index"):
        resolve_openmm_platform("CUDA", device_index=value)


@pytest.mark.parametrize(
    "platform,options",
    [
        ("CPU", {"precision": "double"}),
        ("CPU", {"device_index": 0}),
        ("CUDA", {"opencl_platform_index": 0}),
        ("Reference", {"precision": "single"}),
    ],
)
def test_platform_specific_options_cannot_be_silently_ignored(
    platforms, platform, options
):
    with pytest.raises(ValueError):
        resolve_openmm_platform(platform, **options)


def test_reference_fixed_double_is_not_passed_as_unknown_native_property(platforms):
    selected, properties, _ = resolve_openmm_platform("Reference", precision="double")
    assert selected.getName() == "Reference"
    assert properties == {}


def test_execution_provenance_distinguishes_gpu_model_from_cpu_solver_and_is_live():
    parameter = SimpleNamespace(device="cuda:1", dtype="torch.float32")
    calc = SimpleNamespace(
        device="cuda:1",
        requested_device="cuda:1",
        hessian="numerical",
        model=SimpleNamespace(parameters=lambda: iter([parameter])),
        solvent_correction=SimpleNamespace(
            provider=SimpleNamespace(provenance={"provider": "ddx"})
        ),
    )
    before = calculator_execution(calc)
    assert before["model"]["reported_device"] == "cuda:1"
    assert before["model"]["first_parameter_device"] == "cuda:1"
    assert before["solvent"]["platform"] == "CPU"
    parameter.dtype = "torch.float64"
    assert calculator_execution(calc)["model"]["effective_dtype"] == "torch.float64"
    assert before["model"]["effective_dtype"] == "torch.float32"


def test_execution_metadata_does_not_pollute_ase_standard_properties():
    import numpy as np
    from ase import Atoms

    calc = CalcABC()
    calc.device = "cpu"
    calc._finalize_results(Atoms("H"), energy=0.0, forces=np.zeros((1, 3)))
    assert calc.export_properties()["energy"] == 0.0
    assert "execution" not in calc.results
    assert calc.execution_provenance["model"]["reported_device"] == "cpu"


@pytest.mark.parametrize(
    "provider,method", [("ddx", "pb"), ("apbs", "pb"), ("ambertools", "gb")]
)
def test_direct_correction_rejects_gpu_options_for_cpu_only_solvers(
    water_mol2, tmp_path, provider, method
):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match="OpenMM execution options"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2", "label": "test"},
            {
                "method": method,
                "provider": provider,
                "experimental": True,
                "platform": "CUDA",
            },
            output=str(tmp_path / "unsupported.out"),
        )


@pytest.mark.parametrize("requested_platform", [None, "auto"])
def test_correction_requires_explicit_auto_to_follow_model_device(
    water_mol2, tmp_path, requested_platform
):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    options = {"method": "gb", "provider": "openmm", "experimental": True}
    if requested_platform is not None:
        options["platform"] = requested_platform
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "label": "test"},
        options,
        model_device="cuda:0",
        output=str(tmp_path / "execution.out"),
    )
    execution = correction.provider.provenance["execution"]
    assert execution["requested_platform"] == (requested_platform or "CPU")
    if requested_platform is None:
        assert correction.provider.platform == "CPU"
