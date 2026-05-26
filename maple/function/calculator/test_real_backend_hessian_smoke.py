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


def _assert_fd_matches_legacy(
    make_calc: Callable[[], object],
    *,
    delta: float = 1e-3,
    rtol: float = 1e-5,
    atol: float = 5e-5,
) -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    calc = make_calc()
    atoms.calc = calc

    legacy = _legacy_fd_hessian(calc, atoms, delta=delta)
    shared = FDHessianEvaluator(calc).hessian(atoms, delta=delta)

    np.testing.assert_allclose(shared, legacy, rtol=rtol, atol=atol)
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


def test_ani_analytic_hessian_matches_reference_autograd_loop():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    if not (MODEL_DIR / "ani1x.pt").exists():
        pytest.skip("ani1x.pt missing")

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    calc = ANICalculator(torch.device("cpu"), model="ani1x", implicit="none")

    coordinates = torch.tensor(
        atoms.get_positions(),
        dtype=calc.dtype,
        device=calc.device,
        requires_grad=True,
    ).unsqueeze(0)
    energy = calc.get_energy(atoms, coordinates)
    grad = torch.autograd.grad(energy, coordinates, create_graph=True)[0].view(-1)
    expected = torch.stack([
        torch.autograd.grad(component, coordinates, retain_graph=True)[0].view(-1)
        for component in grad.unbind()
    ]).detach().cpu().numpy()

    actual = calc._get_hessian_analytic(atoms).detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-5)


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


def test_aimnet2_analytic_hessian_batched_vjp_matches_loop():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator, EV2HARTREE

    if not (MODEL_DIR / "aimnet2.pt").exists():
        pytest.skip("aimnet2.pt missing")

    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ],
    )
    calc = AIMNet2Calculator(torch.device("cpu"), implicit="none")

    data_loop, n_atoms = calc._build_single_molecule_hessian_data(atoms)
    energy_loop = calc.get_energy(data_loop) * EV2HARTREE
    expected = calc._hessian_from_energy_loop(
        data_loop["coord"],
        energy_loop,
        n_atoms=n_atoms,
    ).detach().cpu().numpy()

    actual = calc._get_hessian_analytic(atoms)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=5e-5)


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
    settings = calc._predictor_unit.inference_settings
    assert not getattr(settings, "merge_mole", False)
    assert not getattr(settings, "compile", False)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    result = calc.calculate_many([atoms], properties=("energy", "forces"))
    assert result.energies is not None and result.energies.shape == (1,)
    assert result.forces is not None and result.forces[0].shape == (2, 3)

    atoms.calc = calc
    legacy = _legacy_fd_hessian(calc, atoms, delta=1e-3)
    shared = FDHessianEvaluator(calc).hessian(atoms, delta=1e-3)
    np.testing.assert_allclose(shared, legacy, rtol=1e-5, atol=5e-5)


def test_uma_calculate_many_uses_one_fairchem_batch_predict(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("fairchem")
    from maple.function.calculator.uma._uma_calculator import UMACalculator

    model_path = MODEL_DIR / "uma-s-1p1.pt"
    if not model_path.exists():
        pytest.skip("uma-s-1p1.pt missing")

    atoms_list = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.72 + 0.01 * i, 0.0, 0.0]])
        for i in range(3)
    ]

    batch_calc = UMACalculator(
        torch.device("cpu"),
        model="uma-s-1p1",
        checkpoint_path=str(model_path),
        implicit="none",
        task="omol",
    )
    seq_calc = UMACalculator(
        torch.device("cpu"),
        model="uma-s-1p1",
        checkpoint_path=str(model_path),
        implicit="none",
        task="omol",
    )

    predict_calls = []
    original_predict = batch_calc._predictor_unit.predict

    def counted_predict(data, *args, **kwargs):
        predict_calls.append(int(data.num_graphs))
        return original_predict(data, *args, **kwargs)

    monkeypatch.setattr(batch_calc._predictor_unit, "predict", counted_predict)

    batched = batch_calc.calculate_many(atoms_list, properties=("energy", "forces"))
    assert predict_calls == [len(atoms_list)]

    seq_energies = []
    seq_forces = []
    for at in atoms_list:
        seq_calc.calculate(at, properties=["energy", "forces"], system_changes=all_changes)
        seq_energies.append(seq_calc.results["energy"])
        seq_forces.append(np.asarray(seq_calc.results["forces"], dtype=np.float64))

    np.testing.assert_allclose(batched.energies, np.asarray(seq_energies), rtol=1e-6, atol=1e-6)
    assert batched.forces is not None
    for got, expected in zip(batched.forces, seq_forces):
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_ani_calculate_many_uses_one_torch_batch():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    if not (MODEL_DIR / "ani1x.pt").exists():
        pytest.skip("ani1x.pt missing")

    atoms_list = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.72 + 0.01 * i, 0.0, 0.0]])
        for i in range(3)
    ]
    batch_calc = ANICalculator(torch.device("cpu"), model="ani1x", implicit="none")
    seq_calc = ANICalculator(torch.device("cpu"), model="ani1x", implicit="none")

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def __call__(self, species, coordinates):
            self.calls.append(tuple(coordinates.shape))
            return self.model(species, coordinates)

    counter = CountingModel(batch_calc.model)
    batch_calc.model = counter

    batched = batch_calc.calculate_many(atoms_list, properties=("energy", "forces"))
    assert counter.calls == [(len(atoms_list), 2, 3)]

    seq_energies, seq_forces = [], []
    for at in atoms_list:
        seq_calc.calculate(at, properties=["energy", "forces"], system_changes=all_changes)
        seq_energies.append(seq_calc.results["energy"])
        seq_forces.append(np.asarray(seq_calc.results["forces"], dtype=np.float64))

    np.testing.assert_allclose(batched.energies, np.asarray(seq_energies), rtol=1e-5, atol=5e-5)
    assert batched.forces is not None
    for got, expected in zip(batched.forces, seq_forces):
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=5e-5)


def test_aimnet2_calculate_many_uses_one_mol_idx_batch():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator

    if not (MODEL_DIR / "aimnet2.pt").exists():
        pytest.skip("aimnet2.pt missing")

    atoms_list = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.72 + 0.01 * i, 0.0, 0.0]])
        for i in range(3)
    ]
    batch_calc = AIMNet2Calculator(torch.device("cpu"), implicit="none")
    seq_calc = AIMNet2Calculator(torch.device("cpu"), implicit="none")

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def __call__(self, data):
            self.calls.append(int(data["mol_idx"].max().item()))
            return self.model(data)

    counter = CountingModel(batch_calc.model)
    batch_calc.model = counter

    batched = batch_calc.calculate_many(atoms_list, properties=("energy", "forces"))
    assert counter.calls == [len(atoms_list)]

    seq_energies, seq_forces = [], []
    for at in atoms_list:
        seq_calc.calculate(at, properties=["energy", "forces"], system_changes=all_changes)
        seq_energies.append(seq_calc.results["energy"])
        seq_forces.append(np.asarray(seq_calc.results["forces"], dtype=np.float64))

    np.testing.assert_allclose(batched.energies, np.asarray(seq_energies), rtol=1e-5, atol=5e-5)
    assert batched.forces is not None
    for got, expected in zip(batched.forces, seq_forces):
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=5e-5)


def test_mace_calculate_many_uses_one_disconnected_graph_batch():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import MACECalculator

    model_path = MODEL_DIR / "maceoff23m.pt"
    if not model_path.exists():
        pytest.skip("maceoff23m.pt missing")

    atoms_list = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.72 + 0.01 * i, 0.0, 0.0]])
        for i in range(3)
    ]
    batch_calc = MACECalculator(torch.device("cpu"), model="maceoff23m", model_path=str(model_path), implicit="none")
    seq_calc = MACECalculator(torch.device("cpu"), model="maceoff23m", model_path=str(model_path), implicit="none")

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def forward(self, data, local_or_ghost, compute_virials=False):
            self.calls.append(int(data["ptr"].numel() - 1))
            return self.model.forward(data=data, local_or_ghost=local_or_ghost, compute_virials=compute_virials)

    counter = CountingModel(batch_calc.model)
    batch_calc.model = counter

    batched = batch_calc.calculate_many(atoms_list, properties=("energy", "forces"))
    assert counter.calls == [len(atoms_list)]

    seq_energies, seq_forces = [], []
    for at in atoms_list:
        seq_calc.calculate(at, properties=["energy", "forces"], system_changes=all_changes)
        seq_energies.append(seq_calc.results["energy"])
        seq_forces.append(np.asarray(seq_calc.results["forces"], dtype=np.float64))

    np.testing.assert_allclose(batched.energies, np.asarray(seq_energies), rtol=1e-5, atol=5e-5)
    assert batched.forces is not None
    for got, expected in zip(batched.forces, seq_forces):
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=5e-5)


def test_mace_energy_forces_reuses_one_forward():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import MACECalculator

    model_path = MODEL_DIR / "maceoff23m.pt"
    if not model_path.exists():
        pytest.skip("maceoff23m.pt missing")

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    calc = MACECalculator(torch.device("cpu"), model="maceoff23m", model_path=str(model_path), implicit="none")

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def forward(self, data, local_or_ghost, compute_virials=False):
            self.calls += 1
            return self.model.forward(data=data, local_or_ghost=local_or_ghost, compute_virials=compute_virials)

    counter = CountingModel(calc.model)
    calc.model = counter
    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)

    assert counter.calls == 1
    assert "energy" in calc.results
    assert np.asarray(calc.results["forces"]).shape == (2, 3)


def test_maceoff_small_raw_upstream_model_contract():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace")
    from mace.calculators import mace_off
    from maple.function.calculator.mace._mace_calculator import EV2HARTREE, MACECalculator

    cache_path = Path.home() / ".cache" / "mace" / "MACE-OFF23_small.model"
    if not cache_path.exists():
        pytest.skip("MACE-OFF23 small cache missing; avoid network download in smoke test")

    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ],
    )
    calc = MACECalculator(torch.device("cpu"), model="maceoff23s", model_path=None, implicit="none")
    assert calc._raw_mace_model
    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)

    atoms_upstream = atoms.copy()
    atoms_upstream.calc = mace_off(model=str(cache_path), device="cpu", default_dtype="float64")
    expected_energy = atoms_upstream.get_potential_energy() * EV2HARTREE
    expected_forces = atoms_upstream.get_forces() * EV2HARTREE

    assert calc.results["energy"] == pytest.approx(expected_energy, rel=0.0, abs=1e-10)
    np.testing.assert_allclose(calc.results["forces"], expected_forces, rtol=0.0, atol=1e-10)

    hessian = calc.get_hessian(atoms)
    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()


def test_maceomol_raw_upstream_model_contract():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace")
    from mace.calculators import mace_omol
    from maple.function.calculator.mace._mace_calculator import EV2HARTREE, MACECalculator

    cache_path = Path.home() / ".cache" / "mace" / "MACE-omol-0-extra-large-1024.model"
    if not cache_path.exists():
        pytest.skip("MACE-OMOL cache missing; avoid large network download in smoke test")

    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ],
    )
    calc = MACECalculator(torch.device("cpu"), model="maceomol", model_path=str(cache_path), implicit="none")
    assert calc._raw_mace_model
    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)

    atoms_upstream = atoms.copy()
    atoms_upstream.calc = mace_omol(model=str(cache_path), device="cpu", default_dtype="float64")
    expected_energy = atoms_upstream.get_potential_energy() * EV2HARTREE
    expected_forces = atoms_upstream.get_forces() * EV2HARTREE

    assert calc.results["energy"] == pytest.approx(expected_energy, rel=0.0, abs=1e-10)
    np.testing.assert_allclose(calc.results["forces"], expected_forces, rtol=0.0, atol=1e-10)

    atoms_list = [
        atoms.copy(),
        Atoms(
            "H2O",
            positions=[
                [0.0, 0.0, 0.0],
                [0.9672, 0.0, 0.0],
                [-0.2390, 0.9170, 0.0],
            ],
        ),
    ]
    batched = calc.calculate_many(atoms_list, properties=("energy", "forces"))
    seq_energies, seq_forces = [], []
    for at in atoms_list:
        calc.calculate(at, properties=["energy", "forces"], system_changes=all_changes)
        seq_energies.append(calc.results["energy"])
        seq_forces.append(np.asarray(calc.results["forces"], dtype=np.float64))

    np.testing.assert_allclose(batched.energies, np.asarray(seq_energies), rtol=0.0, atol=1e-10)
    assert batched.forces is not None
    for got, expected in zip(batched.forces, seq_forces):
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-10)


def test_egret_mace_wrapper_uses_checkpoint_dtype():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._mace_calculator import MACECalculator

    model_path = MODEL_DIR / "egret1s.pt"
    if not model_path.exists():
        pytest.skip("egret1s.pt missing")

    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ],
    )
    calc = MACECalculator(
        torch.device("cpu"),
        model="egret",
        model_path=str(model_path),
        implicit="none",
    )

    assert calc.dtype == torch.float32
    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
    assert "energy" in calc.results
    assert np.asarray(calc.results["forces"]).shape == (3, 3)


def test_macepol_energy_forces_reuses_one_forward():
    torch = pytest.importorskip("torch")
    from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

    model_path = MODEL_DIR / "macepols.pt"
    if not model_path.exists():
        pytest.skip("macepols.pt missing")

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    calc = MACEPolCalculator(torch.device("cpu"), model="macepols", model_path=str(model_path), implicit="none")

    class CountingModel:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.model(*args, **kwargs)

    counter = CountingModel(calc.model)
    calc.model = counter
    calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)

    assert counter.calls == 1
    assert "energy" in calc.results
    assert np.asarray(calc.results["forces"]).shape == (2, 3)
