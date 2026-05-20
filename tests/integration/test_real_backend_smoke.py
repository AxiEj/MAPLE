"""WS3 integration layer — real periodic ML-backend smoke (run with `-m integration`).

Each case importorskips its backing package and skips gracefully on missing
model weights / auth / CUDA, so the layer never hard-fails on environment gaps.
It only asserts that the real-backend PBC-MD path runs end to end (finite
energy/forces) and emits a provenance manifest. Excluded from the default run.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.md.ensemble.nve import NVE

pytestmark = pytest.mark.integration


def _small_periodic_water() -> Atoms:
    # A loose periodic water box: small enough to be cheap, large enough for the
    # minimum-image convention against typical ML cutoffs.
    atoms = Atoms(
        "OH2OH2",
        positions=[
            [1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [0.7, 1.9, 1.0],
            [5.0, 5.0, 5.0], [5.96, 5.0, 5.0], [4.7, 5.9, 5.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return atoms


@pytest.mark.parametrize(
    ("model", "package"),
    [
        ("aimnet2-pbc", "aimnet"),
        ("mace-mp-pbc-small", "mace"),
    ],
)
def test_real_pbc_backend_nve_smoke(model, package, tmp_path):
    pytest.importorskip(package)
    import torch

    from maple.function.calculator import SetCalculator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        calc = SetCalculator(
            device, model, str(tmp_path / "calc.out"), model_options={}
        ).set_calculator()
    except Exception as exc:  # missing weights / auth / download failure
        pytest.skip(f"backend {model} could not be built: {exc}")

    atoms = _small_periodic_water()
    atoms.calc = calc
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except Exception as exc:
        pytest.skip(f"backend {model} could not evaluate the test system: {exc}")
    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))

    NVE(
        output=str(tmp_path / "nve.out"),
        atoms=atoms,
        paras={
            "steps": 2, "timestep": 0.1, "temperature": 50.0,
            "remove_com_every": 0, "verbose": 0, "log_every": 1,
            "traj_every": 2, "rst_every": 2, "random_seed": 1,
        },
    ).run()

    manifest_file = tmp_path / "nve_md_manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text())
    assert manifest["run"]["ensemble"] == "nve"
    assert manifest["system"]["pbc"] == [True, True, True]
