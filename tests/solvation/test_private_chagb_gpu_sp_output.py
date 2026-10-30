from __future__ import annotations
from types import SimpleNamespace
import pytest
from ase import Atoms
from maple.function.dispatcher.sp.sp import SinglePoint

def _atoms_with_solvation(*, provenance: dict) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SimpleNamespace(results={"solvation": {"energy_hartree": -0.02, "gas_energy_hartree": -1.0, "delta_g_solv_hartree": -0.02, "combined_energy_hartree": -1.02, "components_hartree": {"polar": -0.03, "cavity": 0.011, "dispersion": -0.001}, "provenance": provenance, "ase_free_energy_is_thermochemical_gibbs": False}})
    return atoms

def _current(device: str) -> dict:
    cpu = device == "cpu"
    return {"runtime_provider": "private-torch-cha", "engine": {"api_version": 4}, "platform": "CPU" if cpu else "CUDA", "device": device, "arithmetic_profile": "torch-fp32-sp-v1", "supports_forces": False, "runtime": {"device": device, "dtype": "torch.float32", "workspace_bytes": 3_221_225_472, "peak_allocated_bytes": None if cpu else 734_003_200, "peak_reserved_bytes": None if cpu else 805_306_368}}

@pytest.mark.parametrize(("device", "platform"), [("cpu", "CPU"), ("cuda:2", "CUDA")])
def test_private_chagb_sp_output_reports_dual_device_runtime_and_components(device, platform):
    rendered = "".join(SinglePoint._solvation_lines(_atoms_with_solvation(provenance=_current(device))))
    assert f"Private CHA Torch runtime: device={device}, platform={platform}" in rendered
    assert "dtype=torch.float32, profile=torch-fp32-sp-v1" in rendered
    assert "Private CHA polar component: -0.0300000000 Hartree" in rendered
    assert "Private CHA cavity component: 0.0110000000 Hartree" in rendered
    assert "Private CHA dispersion component: -0.0010000000 Hartree" in rendered
    assert "Private CHA workspace: 3221225472 bytes" in rendered
    assert ("Private CHA CUDA memory: not applicable on CPU" in rendered) if device == "cpu" else ("peak allocated=734003200 bytes, peak reserved=805306368 bytes" in rendered)
    assert "supports_forces=false (energy-only)" in rendered

@pytest.mark.parametrize("provenance", [
    {"runtime_provider": "private-torch-cha", "engine": {"api_version": 2}, "arithmetic_profile": "cpu-fp32-exact-sp-v1"},
    {"runtime_provider": "private-torch-cha", "engine": {"api_version": 3}, "arithmetic_profile": "cuda-fp32-sp-v1"},
    {"runtime_provider": "other-provider", "engine": {"api_version": 4}},
])
def test_legacy_and_other_provider_records_keep_basic_output(provenance):
    rendered = "".join(SinglePoint._solvation_lines(_atoms_with_solvation(provenance=provenance)))
    assert "Private CHA" not in rendered
    assert "Gas-phase MLIP energy" in rendered

@pytest.mark.parametrize("mutate", [
    lambda p: p.update(arithmetic_profile="cuda-fp32-sp-v1"), lambda p: p.update(platform="CUDA"), lambda p: p.update(supports_forces=True), lambda p: p["runtime"].update(device="cuda:0"), lambda p: p["runtime"].update(dtype="torch.float64"), lambda p: p["runtime"].update(peak_allocated_bytes=0),
])
def test_malformed_current_api4_evidence_raises(mutate):
    provenance = _current("cpu"); mutate(provenance)
    with pytest.raises(ValueError, match="Malformed private CHA Torch SP provenance"):
        SinglePoint._solvation_lines(_atoms_with_solvation(provenance=provenance))

def test_current_api4_requires_all_component_and_runtime_fields():
    atoms = _atoms_with_solvation(provenance=_current("cuda:0")); del atoms.calc.results["solvation"]["components_hartree"]["cavity"]
    with pytest.raises(ValueError, match="Malformed private CHA Torch SP provenance"):
        SinglePoint._solvation_lines(atoms)

@pytest.mark.parametrize("engine", [None, {"api_version": 3}])
def test_current_profile_requires_exact_api4_engine_identity(engine):
    provenance = _current("cpu")
    if engine is None:
        del provenance["engine"]
    else:
        provenance["engine"] = engine
    with pytest.raises(
        ValueError,
        match="torch-fp32-sp-v1 requires an API 4 engine identity",
    ):
        SinglePoint._solvation_lines(_atoms_with_solvation(provenance=provenance))
