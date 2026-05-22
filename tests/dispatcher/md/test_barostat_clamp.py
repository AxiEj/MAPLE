"""WS-C — c-rescale stability clamp is counted, recorded, and fatal in production.

A fired per-step volume clamp truncates the stochastic-cell-rescaling ensemble, so the
NPT driver aborts immediately (no manifest, no "completed" banner) unless the run opts
into ``allow_barostat_clamp``. Either way the clamp count is recorded in the manifest.
"""

import json

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.md.barostat.crescale import CRescaleBarostat, MDBarostatClampError
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.validation import MapleLJReferenceCalculator


def _ar():
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.6, 0.0, 0.0]],
                  cell=[14.0, 14.0, 14.0], pbc=True)
    atoms.calc = MapleLJReferenceCalculator()
    return atoms


# Extreme but range-valid coupling that drives a pathological one-step volume move so
# the [0.125, 8.0] clamp is forced to clip (huge pressure target, tiny tau_p).
_CLAMP_PARAS = {
    "pressure": 1.0e6, "tau_p": 1.0, "compressibility": 0.1, "timestep": 10.0,
    "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 50.0,
    "temperature": 80.0, "remove_com_every": 0, "verbose": 0,
    "log_every": 1, "traj_every": 1, "rst_every": 0, "random_seed": 5,
}


# ── barostat unit: clamp counter ────────────────────────────────────────────

def test_crescale_counts_and_records_clamp():
    atoms = _ar()
    baro = CRescaleBarostat(atoms, pressure=1.0e6, temperature=80.0, tau_p=1.0,
                            timestep=10.0, compressibility=0.1,
                            rng=np.random.default_rng(0))
    assert baro.clamp_count == 0 and baro.last_clamped is False
    baro.apply(np.zeros((len(atoms), 3)))
    assert baro.last_clamped is True
    assert baro.clamp_count == 1
    assert baro.max_abs_log_excursion > 0.0


def test_crescale_no_clamp_near_equilibrium():
    atoms = _ar()
    baro = CRescaleBarostat(atoms, pressure=1.0, temperature=80.0, tau_p=1000.0,
                            timestep=0.5, compressibility=4.5e-5,
                            rng=np.random.default_rng(0))
    for _ in range(20):
        baro.apply(np.zeros((len(atoms), 3)))
    assert baro.clamp_count == 0
    assert baro.last_clamped is False


# ── NPT end-to-end: immediate-fatal vs explicit override ────────────────────

def test_npt_clamp_is_fatal_by_default(tmp_path):
    out = tmp_path / "clamp.out"
    sim = NPT(output=str(out), atoms=_ar(), paras={**_CLAMP_PARAS, "steps": 5})
    with pytest.raises(MDBarostatClampError, match="stability clamp fired"):
        sim.run()
    # No success banner, and no manifest written for an aborted run.
    text = out.read_text()
    assert "completed successfully" not in text
    assert "ABORTED" in text
    assert not (tmp_path / "clamp_md_manifest.json").exists()


def test_npt_clamp_allowed_with_override_records_in_manifest(tmp_path):
    out = tmp_path / "clamp_ok.out"
    sim = NPT(output=str(out), atoms=_ar(),
              paras={**_CLAMP_PARAS, "steps": 3, "allow_barostat_clamp": True})
    sim.run()  # must not raise
    assert sim.barostat.clamp_count > 0
    manifest = json.loads((tmp_path / "clamp_ok_md_manifest.json").read_text())
    clamps = manifest["run"]["barostat_clamps"]
    assert clamps["allowed"] is True
    assert clamps["count"] == sim.barostat.clamp_count
    assert clamps["count"] > 0
