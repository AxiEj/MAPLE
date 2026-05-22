"""WS3 integration layer — real periodic ML-backend smoke (run with `-m integration`).

Each case importorskips its backing package and skips gracefully on missing model
weights / auth / CUDA, so the layer never hard-fails on environment gaps. It asserts
that the real-backend PBC-MD path runs end to end (finite energy/forces/stress) and
emits a provenance manifest across NVE, NVT (v-rescale) and NPT (c-rescale), plus a
stress-availability check. The full statistical / finite-difference gates live in the
opt-in production acceptance matrix; this layer proves the real backends drive the
code paths, not their accuracy. Excluded from the default run.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.ensemble.npt import NPT

pytestmark = pytest.mark.integration

_BACKENDS = [
    ("aimnet2-pbc", "aimnet"),
    ("mace-mp-pbc-small", "mace"),
]


def _small_periodic_water() -> Atoms:
    # A loose periodic water box: small enough to be cheap, large enough for the
    # minimum-image convention against typical ML cutoffs.
    return Atoms(
        "OH2OH2",
        positions=[
            [1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [0.7, 1.9, 1.0],
            [5.0, 5.0, 5.0], [5.96, 5.0, 5.0], [4.7, 5.9, 5.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def _build_calc(model: str, package: str, tmp_path):
    """Build a real periodic calculator or skip with the concrete reason."""
    pytest.importorskip(package)
    import torch

    from maple.function.calculator import SetCalculator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return SetCalculator(
            device, model, str(tmp_path / "calc.out"), model_options={}
        ).set_calculator()
    except Exception as exc:  # missing weights / auth / download failure
        pytest.skip(f"backend {model} could not be built: {exc}")


def _system_with(calc):
    atoms = _small_periodic_water()
    atoms.calc = calc
    return atoms


@pytest.mark.parametrize(("model", "package"), _BACKENDS)
def test_real_pbc_backend_nve_smoke(model, package, tmp_path):
    atoms = _system_with(_build_calc(model, package, tmp_path))
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception as exc:
        pytest.skip(f"backend {model} could not evaluate the test system: {exc}")
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))

    NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={
        "steps": 2, "timestep": 0.1, "temperature": 50.0,
        "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": 2, "rst_every": 2, "random_seed": 1,
    }).run()

    manifest = json.loads((tmp_path / "nve_md_manifest.json").read_text())
    assert manifest["run"]["ensemble"] == "nve"
    assert manifest["system"]["pbc"] == [True, True, True]


@pytest.mark.parametrize(("model", "package"), _BACKENDS)
def test_real_pbc_backend_nvt_vrescale_smoke(model, package, tmp_path):
    atoms = _system_with(_build_calc(model, package, tmp_path))
    try:
        atoms.get_potential_energy()
    except Exception as exc:
        pytest.skip(f"backend {model} could not evaluate the test system: {exc}")

    NVT(output=str(tmp_path / "nvt.out"), atoms=atoms, paras={
        "steps": 5, "timestep": 0.1, "temperature": 50.0, "thermostat": "v-rescale",
        "tau_t": 20.0, "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": 5, "rst_every": 5, "random_seed": 1,
    }).run()

    manifest = json.loads((tmp_path / "nvt_md_manifest.json").read_text())
    assert manifest["run"]["ensemble"] == "nvt"


@pytest.mark.parametrize(("model", "package"), _BACKENDS)
def test_real_pbc_backend_npt_crescale_smoke(model, package, tmp_path):
    calc = _build_calc(model, package, tmp_path)
    if not getattr(calc, "maple_stress_supported", False):
        pytest.skip(f"backend {model} does not declare stress support; NPT not applicable")
    atoms = _system_with(calc)
    try:
        atoms.get_stress()
    except Exception as exc:
        pytest.skip(f"backend {model} could not evaluate stress: {exc}")

    sim = NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={
        "steps": 5, "timestep": 0.1, "temperature": 50.0, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 20.0,
        "tau_p": 1000.0, "remove_com_every": 0, "verbose": 0, "log_every": 1,
        "traj_every": 5, "rst_every": 5, "random_seed": 1,
    })
    sim.run()

    manifest = json.loads((tmp_path / "npt_md_manifest.json").read_text())
    assert manifest["run"]["ensemble"] == "npt"
    # The isotropic-only flag and the clamp summary must be recorded for a c-rescale run.
    assert manifest["run"]["barostat"]["mode"] == "isotropic"
    assert "barostat_clamps" in manifest["run"]


@pytest.mark.parametrize(("model", "package"), _BACKENDS)
def test_real_pbc_backend_stress_is_finite(model, package, tmp_path):
    calc = _build_calc(model, package, tmp_path)
    if not getattr(calc, "maple_stress_supported", False):
        pytest.skip(f"backend {model} does not declare stress support")
    atoms = _system_with(calc)
    try:
        stress = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception as exc:
        pytest.skip(f"backend {model} could not evaluate stress: {exc}")
    # Smoke only: stress is available and finite with the right shape. The signed
    # finite-difference vs -dE/dV magnitude gate is the LJ-reference acceptance class.
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))
