from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.solvent_terms import PySCFSMDCDSTerm, SolventEnergyState


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.927, 0.0]]
        ),
        info={"charge": 0, "multiplicity": 1},
    )


def test_pyscf_smd_cds_term_normalizes_solvent_and_converts_energy_gradient(
    monkeypatch,
):
    gradient_hartree_per_angstrom = np.arange(9, dtype=float).reshape(3, 3) / 1000

    def fake_cds(symbols, positions, *, solvent):
        assert tuple(symbols) == ("O", "H", "H")
        assert np.asarray(positions).shape == (3, 3)
        assert solvent == "dichloromethane"
        return SimpleNamespace(
            energy_hartree=-0.0125,
            position_gradient_hartree_per_angstrom=(gradient_hartree_per_angstrom),
        )

    monkeypatch.setattr("maple.solvation.solvent_terms.pyscf_smd_cds", fake_cds)
    term = PySCFSMDCDSTerm(("O", "H", "H"), "DCM")
    state = term.evaluate(_water(), need_gradient=True)
    assert state.energy_eV == pytest.approx(-0.0125 * HARTREE_TO_EV, abs=0.0)
    np.testing.assert_allclose(
        state.gradient_eV_per_A,
        gradient_hartree_per_angstrom * HARTREE_TO_EV,
        rtol=0.0,
        atol=0.0,
    )
    assert term.solvent == "dichloromethane"
    assert len(term.configuration_sha256()) == 64
    with pytest.raises(ValueError):
        state.gradient_eV_per_A.setflags(write=True)

    displaced = _water()
    displaced.positions[0, 0] += 1.0e-4
    displaced_state = term.evaluate(displaced, need_gradient=False)
    assert displaced_state.topology_id == state.topology_id
    assert displaced_state.geometry_sha256 != state.geometry_sha256
    assert displaced_state.state_sha256 != state.state_sha256
    assert state.topology_observation_coverage == "unobservable"
    assert state.unobservable_topology_components == (
        "pyscf-smd-libsolvent-internal-surface",
    )


def test_pyscf_smd_cds_term_omits_unrequested_gradient_and_rejects_wrong_domain(
    monkeypatch,
):
    monkeypatch.setattr(
        "maple.solvation.solvent_terms.pyscf_smd_cds",
        lambda *_args, **_kwargs: SimpleNamespace(
            energy_hartree=0.001,
            position_gradient_hartree_per_angstrom=np.zeros((3, 3)),
        ),
    )
    term = PySCFSMDCDSTerm(("O", "H", "H"), "water")
    state = term.evaluate(_water(), need_gradient=False)
    assert isinstance(state, SolventEnergyState)
    assert state.gradient_eV_per_A is None

    wrong_symbols = Atoms(
        "NH2",
        positions=_water().positions,
        info={"charge": 0, "multiplicity": 1},
    )
    with pytest.raises(ValueError, match="symbols"):
        term.evaluate(wrong_symbols, need_gradient=False)
    charged = _water()
    charged.info["charge"] = 1
    with pytest.raises(ValueError, match="neutral singlets"):
        term.evaluate(charged, need_gradient=False)


def test_solvent_term_import_does_not_import_optional_pyscf_runtime():
    code = r"""
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pyscf" or name.startswith("pyscf."):
        raise RuntimeError("pyscf import forbidden")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import maple.solvation.solvent_terms
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
