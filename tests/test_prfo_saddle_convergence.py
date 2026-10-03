"""A small force alone cannot establish a first-order saddle."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.dispatcher.frequency.frequency import HARTREE_AMU_ANGSTROM2_TO_CM1
from maple.function.dispatcher.ts.algorithm.PRFO import (
    PRFO,
    _mass_weighted_optimization_basis,
)


@pytest.mark.parametrize(
    "frequencies,accepted",
    [
        ([100.0, 200.0, 300.0], False),
        ([-5.0, 200.0, 300.0], False),
        ([-100.0, 200.0, 300.0], True),
        ([-100.0, -200.0, 300.0], False),
    ],
)
def test_prfo_requires_exactly_one_material_negative_mode(
    tmp_path, frequencies, accepted
):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    basis, rank = _mass_weighted_optimization_basis(atoms, project_rigid_modes=True)
    assert rank == 6
    freq = np.asarray(frequencies)
    eigenvalues = np.sign(freq) * (freq / HARTREE_AMU_ANGSTROM2_TO_CM1) ** 2
    mass = np.repeat(np.sqrt(atoms.get_masses()), 3)
    hessian = (mass[:, None] * (basis @ np.diag(eigenvalues) @ basis.T)) * mass[None, :]
    calls = []

    def fresh_hessian(current):
        calls.append(current.get_positions().copy())
        return hessian

    atoms.calc = SimpleNamespace(get_hessian=fresh_hessian)
    optimizer = PRFO(
        str(tmp_path / "index.out"), atoms, paras={"project_rigid_modes": True}
    )
    assert optimizer.check_saddle_curvature(atoms) is accepted
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], atoms.get_positions())
    assert optimizer.last_saddle_curvature is not None
    assert optimizer.last_saddle_curvature["negative_mode_count"] == np.count_nonzero(
        freq < -10
    )


def test_prfo_nonfinite_final_hessian_cannot_certify_convergence(tmp_path):
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.calc = SimpleNamespace(get_hessian=lambda current: np.full((3, 3), np.nan))
    optimizer = PRFO(str(tmp_path / "bad.out"), atoms)
    with pytest.raises(ValueError, match="finite"):
        optimizer.check_saddle_curvature(atoms)
