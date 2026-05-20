"""WS4a — Berendsen NPT is hard-rejected for production unless opted in.

Berendsen suppresses volume fluctuations and does not sample the NPT ensemble,
so it must raise by default; ``allow_equilibration_only_barostat=true`` is the
explicit experimental escape hatch (mirroring the allow_partial_pbc idiom).
"""

import numpy as np
import pytest
from ase.build import bulk

from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.validation import MapleLJReferenceCalculator


def _lj_crystal():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)
    atoms.calc = MapleLJReferenceCalculator(rc=4.0)  # PBC + stress capable, min-image clean
    return atoms


def test_npt_berendsen_rejected_by_default(tmp_path):
    atoms = _lj_crystal()
    with pytest.raises(ValueError, match="equilibration-only"):
        NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={
            "steps": 0, "barostat": "berendsen", "verbose": 0, "remove_com_every": 0,
        })


def test_npt_berendsen_runs_with_escape_hatch(tmp_path):
    atoms = _lj_crystal()
    out = str(tmp_path / "npt.out")
    NPT(output=out, atoms=atoms, paras={
        "steps": 2, "timestep": 0.5, "barostat": "berendsen",
        "allow_equilibration_only_barostat": True, "verbose": 0,
        "remove_com_every": 0, "log_every": 2, "traj_every": 2, "rst_every": 0,
        "random_seed": 1,
    }).run()
    # The equilibration-only warning is still emitted on the opted-in path.
    assert "equilibration-only" in (tmp_path / "npt.out").read_text()


def test_npt_crescale_is_unaffected(tmp_path):
    atoms = _lj_crystal()
    # The production default barostat constructs without the escape hatch.
    sim = NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={
        "steps": 0, "barostat": "c-rescale", "verbose": 0, "remove_com_every": 0,
    })
    assert sim.params.barostat == "c-rescale"
