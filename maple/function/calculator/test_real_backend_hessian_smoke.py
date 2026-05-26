"""Opt-in smoke tests for real calculator numerical Hessian refactors.

These tests compare the shared ``FDHessianEvaluator`` against the old-style
per-DOF force loop on one tiny molecule. They are disabled by default because
they load model weights and optional backend packages.

Run explicitly with:

    MAPLE_REAL_BACKEND_SMOKE=1 python -m pytest -q \
        maple/function/calculator/test_real_backend_hessian_smoke.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

from maple.function.calculator._batch_eval import FDHessianEvaluator


pytestmark = pytest.mark.skipif(
    os.environ.get("MAPLE_REAL_BACKEND_SMOKE") != "1",
    reason="set MAPLE_REAL_BACKEND_SMOKE=1 to load real model backends",
)


MODEL_DIR = Path(__file__).resolve().parent / "model"


def _legacy_fd_hessian(calc, atoms: Atoms, delta: float) -> np.ndarray:
    pos0 = atoms.get_positions().copy()
    n_atoms = len(atoms)
    H = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=np.float64)
    try:
        for atom_idx in range(n_atoms):
            for axis in range(3):
                row = 3 * atom_idx + axis

                pos_p = pos0.copy()
                pos_p[atom_idx, axis] += delta
                atoms.set_positions(pos_p)
                calc.calculate(atoms, properties=["forces"], system_changes=all_changes)
                F_plus = np.asarray(calc.results["forces"], dtype=np.float64)

                pos_m = pos0.copy()
                pos_m[atom_idx, axis] -= delta
                atoms.set_positions(pos_m)
                calc.calculate(atoms, properties=["forces"], system_changes=all_changes)
                F_minus = np.asarray(calc.results["forces"], dtype=np.float64)

                H[row, :] = (-(F_plus - F_minus) / (2.0 * delta)).reshape(-1)
    finally:
        atoms.set_positions(pos0)
    return H


def _assert_fd_matches_legacy(make_calc: Callable[[], object], *, delta: float = 1e-3) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    calc = make_calc()
    atoms.calc = calc

    legacy = _legacy_fd_hessian(calc, atoms, delta=delta)
    shared = FDHessianEvaluator(calc).hessian(atoms, delta=delta)

    np.testing.assert_allclose(shared, legacy, rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(atoms.get_positions(), np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))


def test_ani_numerical_hessian_matches_legacy_fd_loop():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    if not (MODEL_DIR / "ani1x.pt").exists():
        pytest.skip("ani1x.pt missing")

    def make_calc():
        calc = ANICalculator(torch.device("cpu"), model="ani1x", implicit="none")
        calc.hessian = "numerical"
        return calc

    _assert_fd_matches_legacy(make_calc)


def test_aimnet2_numerical_hessian_matches_legacy_fd_loop():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

    if not (MODEL_DIR / "aimnet2.pt").exists():
        pytest.skip("aimnet2.pt missing")

    def make_calc():
        calc = AIMNet2Calculator(torch.device("cpu"), implicit="none")
        calc.hessian = "numerical"
        return calc

    _assert_fd_matches_legacy(make_calc)


def test_mace_numerical_hessian_matches_legacy_fd_loop():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import MACECalculator

    model_path = MODEL_DIR / "maceoff23m.pt"
    if not model_path.exists():
        pytest.skip("maceoff23m.pt missing")

    def make_calc():
        calc = MACECalculator(
            torch.device("cpu"),
            model="maceoff23m",
            model_path=str(model_path),
            implicit="none",
        )
        calc.hessian = "numerical"
        return calc

    _assert_fd_matches_legacy(make_calc)


def test_macepol_numerical_hessian_matches_legacy_fd_loop():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

    model_path = MODEL_DIR / "macepols.pt"
    if not model_path.exists():
        pytest.skip("macepols.pt missing")

    def make_calc():
        calc = MACEPolCalculator(
            torch.device("cpu"),
            model="macepols",
            model_path=str(model_path),
            implicit="none",
        )
        calc.hessian = "numerical"
        return calc

    _assert_fd_matches_legacy(make_calc)


def test_uma_calculate_many_contract_and_hessian_match_legacy_fd_loop():
    torch = pytest.importorskip("torch")
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    model_path = MODEL_DIR / "uma-s-1p1.pt"
    if not model_path.exists():
        pytest.skip("uma-s-1p1.pt missing")

    def make_calc():
        return UMACalculator(
            torch.device("cpu"),
            model="uma-s-1p1",
            checkpoint_path=str(model_path),
            implicit="none",
            task="omol",
        )

    calc = make_calc()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    result = calc.calculate_many([atoms], properties=("energy", "forces"))
    assert result.energies is not None and result.energies.shape == (1,)
    assert result.forces is not None and result.forces[0].shape == (2, 3)

    atoms.calc = calc
    legacy = _legacy_fd_hessian(calc, atoms, delta=1e-3)
    shared = FDHessianEvaluator(calc).hessian(atoms, delta=1e-3)
    np.testing.assert_allclose(shared, legacy, rtol=1e-5, atol=5e-5)
