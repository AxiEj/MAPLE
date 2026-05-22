"""WS-B — single restart velocity-representation normalization gate (P0-2).

A Langevin LF-Middle checkpoint loaded into another ensemble must be converted back
to a standard velocity, never consumed as a full-step velocity. Every ensemble routes
its loaded velocities through ``normalize_velocities_to_standard`` with the source
timestep taken from the RST; a missing source timestep or an unrecognized
representation is rejected, not guessed.
"""

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.rst_io import read_rst, write_rst
from maple.function.dispatcher.md.utils import (
    AMU_TO_AU,
    FS_TO_AU,
    VELOCITY_REPR_LFMIDDLE_CARRIED,
    VELOCITY_REPR_STANDARD,
    forces_au,
    normalize_velocities_to_standard,
)
from maple.function.dispatcher.md.validation import MapleLJReferenceCalculator


def _ar():
    # Ar pair in a box whose min-image radius (7.0 Å) exceeds the LJ rc (6.5 Å), with a
    # non-zero pair force so the carried<->standard conversion is observable.
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.6, 0.0, 0.0]],
                  cell=[14.0, 14.0, 14.0], pbc=True)
    atoms.calc = MapleLJReferenceCalculator()
    return atoms


# ── helper: conversion math + reject contract (condition 5) ─────────────────

def test_normalize_carried_to_standard_uses_source_dt():
    atoms = _ar()
    rng = np.random.default_rng(0)
    v = rng.standard_normal((2, 3)) * 1e-3
    f = rng.standard_normal((2, 3)) * 1e-3
    dt_au = 0.5 * FS_TO_AU
    out, rep = normalize_velocities_to_standard(
        atoms, v, VELOCITY_REPR_LFMIDDLE_CARRIED, f, dt_au
    )
    masses = atoms.get_masses() * AMU_TO_AU
    assert rep == VELOCITY_REPR_STANDARD
    assert np.allclose(out, v + 0.5 * dt_au * f / masses[:, None])


def test_normalize_standard_is_identity():
    atoms = _ar()
    v = np.full((2, 3), 1e-3)
    out, rep = normalize_velocities_to_standard(atoms, v, VELOCITY_REPR_STANDARD, None, None)
    assert rep == VELOCITY_REPR_STANDARD
    assert np.allclose(out, v)


def test_normalize_rejects_unknown_representation():
    atoms = _ar()
    with pytest.raises(ValueError, match="Unrecognized velocity_representation"):
        normalize_velocities_to_standard(atoms, np.zeros((2, 3)), "bogus", np.zeros((2, 3)), 1.0)


def test_normalize_rejects_carried_without_source_timestep():
    atoms = _ar()
    with pytest.raises(ValueError, match="without the source timestep"):
        normalize_velocities_to_standard(
            atoms, np.zeros((2, 3)), VELOCITY_REPR_LFMIDDLE_CARRIED, np.zeros((2, 3)), None
        )


def test_read_rst_rejects_unknown_velocity_representation(tmp_path):
    rst = tmp_path / "bad.rst"
    write_rst(rst, _ar(), np.zeros((2, 3)), step=0, timestep=0.5, ensemble="nvt",
              energy=0.0, velocity_representation="bogus")
    with pytest.raises(ValueError, match="Unrecognized velocity_representation"):
        read_rst(rst)


# ── end-to-end: a carried checkpoint is normalized on load, not consumed raw ─

def _run_load(cls, paras, rep, v_raw, dt, ensemble, tmp_path):
    p = tmp_path / f"{ensemble}_{rep}.rst"
    write_rst(p, _ar(), v_raw, step=0, timestep=dt, ensemble=ensemble,
              energy=0.0, velocity_representation=rep)
    atoms = _ar()
    base = {
        "steps": 2, "verbose": 0, "load_state": True, "rst_file": str(p),
        "traj_every": 1, "log_every": 1, "remove_com": False, "remove_com_every": 0,
        "random_seed": 1234,
    }
    base.update(paras)
    sim = cls(output=str(tmp_path / f"{ensemble}_{rep}.out"), atoms=atoms, paras=base)
    sim.run()
    return np.asarray(sim.atoms.arrays["velocities"], dtype=float)


@pytest.mark.parametrize("ensemble,cls,paras", [
    ("nve", NVE, {}),
    ("nvt", NVT, {"thermostat": "v-rescale"}),
    ("npt", NPT, {"thermostat": "v-rescale", "barostat": "c-rescale", "pressure": 1.0}),
])
def test_carried_checkpoint_normalized_not_consumed_raw(ensemble, cls, paras, tmp_path):
    # Same stored numbers, loaded once as a standard checkpoint and once as a Langevin
    # LF-Middle carried checkpoint, with the same RNG seed so only the initial velocity
    # differs.  The carried one is converted (+0.5·dt·F/m) before integrating; with a
    # non-zero pair force the two trajectories must therefore differ.  The NVE hazard
    # was that a carried checkpoint was consumed as-is, giving identical results.
    dt = 0.5
    v_raw = np.random.default_rng(7).standard_normal((2, 3)) * 1e-3
    v_from_standard = _run_load(cls, paras, "standard", v_raw, dt, ensemble, tmp_path)
    v_from_carried = _run_load(
        cls, paras, VELOCITY_REPR_LFMIDDLE_CARRIED, v_raw, dt, ensemble, tmp_path
    )
    assert not np.allclose(v_from_standard, v_from_carried, atol=1e-10)
