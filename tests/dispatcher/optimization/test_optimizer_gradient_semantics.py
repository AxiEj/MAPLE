import importlib

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from maple.function.dispatcher.optimization.algorithm.LBFGS import LBFGS
from maple.function.dispatcher.optimization.algorithm.SDCG import SDCG
from maple.function.dispatcher.optimization.algorithm import _common
from maple.function.dispatcher.optimization.algorithm._common import is_converged


sdcg_module = importlib.import_module(
    "maple.function.dispatcher.optimization.algorithm.SDCG"
)


def _set_convergence_thresholds(atoms):
    atoms.f_max_th = 0.02
    atoms.f_rms_th = 0.015
    atoms.dp_max_th = 0.005
    atoms.dp_rms_th = 0.003


def _cu_dimer(distance=2.3):
    atoms = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    atoms.calc = EMT()
    _set_convergence_thresholds(atoms)
    return atoms


class TracedLBFGS(LBFGS):
    def __init__(self, *args, **kwargs):
        self.descent_dots = []
        super().__init__(*args, **kwargs)

    def _two_loop(self, grad_flat):
        direction = super()._two_loop(grad_flat)
        self.descent_dots.append(float(np.dot(direction, grad_flat)))
        return direction


def test_lbfgs_prunes_bad_curvature_and_falls_back_to_sd(tmp_path):
    optimizer = LBFGS(
        _cu_dimer(),
        output=str(tmp_path / "bad_history.out"),
        paras={"verbose": 0},
    )
    s_vec = np.array([1.0])
    y_vec = np.array([-1.0])

    optimizer._update_history(s_vec, y_vec)

    assert optimizer.S == []
    assert optimizer.Y == []
    assert optimizer.rhos == []

    optimizer.S = [s_vec]
    optimizer.Y = [y_vec]
    optimizer.rhos = [-1.0]

    grad = np.array([1.0])
    direction = optimizer._two_loop(grad)

    assert np.allclose(direction, -grad / optimizer.params.curvature)
    assert np.dot(direction, grad) < 0.0


def test_lbfgs_emt_cu_converges_with_descent_directions(tmp_path):
    atoms = _cu_dimer()
    optimizer = TracedLBFGS(
        atoms,
        output=str(tmp_path / "cu_lbfgs.out"),
        paras={
            "max_iter": 100,
            "max_step": 0.05,
            "verbose": 0,
            "log_final_paths": False,
        },
    )

    optimizer.run()

    assert optimizer.descent_dots
    assert max(optimizer.descent_dots) < 0.0
    assert is_converged(atoms)


@pytest.mark.parametrize("method", ["sd"])
def test_sdcg_uses_common_writer_append_mode(tmp_path, monkeypatch, method):
    assert sdcg_module.write_xyz is _common.write_xyz
    calls = []

    def spy_write_xyz(filename, atoms_list, energies=None, mode="w", start_index=0):
        calls.append({"mode": mode, "start_index": start_index})
        return _common.write_xyz(
            filename,
            atoms_list,
            energies=energies,
            mode=mode,
            start_index=start_index,
        )

    monkeypatch.setattr(sdcg_module, "write_xyz", spy_write_xyz)

    optimizer = SDCG(
        _cu_dimer(),
        output=str(tmp_path / "sdcg_append.out"),
        paras={
            "method": method,
            "max_iter": 1,
            "diis_enabled": False,
            "verbose": 0,
            "log_final_paths": False,
        },
    )

    optimizer.run()

    assert {"mode": "a", "start_index": 1} in calls
