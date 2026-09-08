"""Optional parity checks against native backends using the same weights."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule


def _asset(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to a trusted local checkpoint")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{variable} is not a file: {path}")
    return path


def _assert_mace_weights_match(eager, traced) -> None:
    eager_state = eager.state_dict()
    traced_state = traced.state_dict()
    missing = []
    for key, expected in eager_state.items():
        traced_key = f"model.{key}"
        if traced_key not in traced_state:
            missing.append(key)
            continue
        assert expected.shape == traced_state[traced_key].shape
        assert expected.detach().cpu().equal(traced_state[traced_key].detach().cpu())
    assert not missing


@pytest.fixture(scope="module")
def matched_macepolar_paths():
    eager_path = _asset("MAPLE_TEST_MACE_NATIVE_MODEL")
    trace_path = _asset("MAPLE_TEST_MACE_TRACE_MODEL")
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace.calculators")
    # The eager file uses Python pickle. The environment variable is therefore
    # deliberately restricted to a trusted local checkpoint.
    eager_model = torch.load(eager_path, map_location="cpu", weights_only=False)
    traced_model = torch.jit.load(str(trace_path), map_location="cpu")
    _assert_mace_weights_match(eager_model, traced_model)
    return eager_path, trace_path


@pytest.mark.backend
@pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2), (0, 3)])
def test_macepolar_trace_matches_native_mace_with_identical_weights(
    matched_macepolar_paths, charge: int, multiplicity: int
) -> None:
    import torch
    from mace.calculators import MACECalculator

    from maple.function.calculator.calculator_base import EV2HARTREE
    from maple.function.calculator.mace._macepol_calculator import MACEPolCalculator

    eager_path, trace_path = matched_macepolar_paths
    native = MACECalculator(
        model_paths=str(eager_path),
        device="cpu",
        default_dtype="float32",
        model_type="PolarMACE",
    )
    maple = MACEPolCalculator(
        device=torch.device("cpu"),
        model="macepols",
        model_path=str(trace_path),
    )
    native_atoms = molecule("H2O")
    native_atoms.info.update(charge=charge, spin=multiplicity)
    maple_atoms = native_atoms.copy()
    maple_atoms.info.pop("spin")
    maple_atoms.info["mult"] = multiplicity
    native_atoms.calc = native
    maple_atoms.calc = maple

    native_energy = native_atoms.get_potential_energy() * EV2HARTREE
    native_forces = native_atoms.get_forces() * EV2HARTREE
    native_charges = np.asarray(native.results["charges"])
    maple.calculate(maple_atoms, ["energy", "forces", "charges"])

    assert maple.results["energy"] == pytest.approx(native_energy, abs=2.0e-10)
    np.testing.assert_allclose(maple.results["forces"], native_forces, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(maple.results["charges"], native_charges, rtol=0.0, atol=2.0e-7)


def _configure_native_aimnet_coulomb(base, method: str) -> None:
    modules = [
        module
        for name, module in base.model.named_modules()
        if name.rsplit(".", 1)[-1] == "lrcoulomb"
    ]
    assert modules
    for module in modules:
        module.method = method
        if method == "dsf":
            module.dsf_rc = 15.0
            module.dsf_alpha = 0.2
    base.cutoff_lr = float("inf") if method == "simple" else 15.0


@pytest.mark.backend
@pytest.mark.parametrize(
    "model_name,model_variable",
    [
        ("aimnet2", "MAPLE_TEST_AIMNET2_MODEL"),
        ("aimnet2nse", "MAPLE_TEST_AIMNET2NSE_MODEL"),
    ],
)
@pytest.mark.parametrize("method", ["simple", "dsf"])
def test_aimnet2_wrapper_matches_official_runtime_on_the_same_checkpoint(
    model_name: str, model_variable: str, method: str
) -> None:
    model_path = _asset(model_variable)
    torch = pytest.importorskip("torch")
    pytest.importorskip("aimnet.calculators")
    from aimnet.calculators import AIMNet2Calculator as NativeAIMNet2
    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    from maple.function.calculator.aimnet._aimnet2_calculator import AIMNet2Calculator
    from maple.function.calculator.calculator_base import EV2HARTREE

    native_base = NativeAIMNet2(str(model_path), device="cpu")
    _configure_native_aimnet_coulomb(native_base, method)
    native = AIMNet2ASE(native_base)
    maple = AIMNet2Calculator(
        device=torch.device("cpu"),
        model=model_name,
        model_path=str(model_path),
        coulomb_method=method,
    )
    native_atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    native_atoms.info.update(charge=1, mult=2)
    maple_atoms = native_atoms.copy()
    native_atoms.calc = native
    maple_atoms.calc = maple

    native_energy = native_atoms.get_potential_energy() * EV2HARTREE
    native_forces = native_atoms.get_forces() * EV2HARTREE
    native_charges = np.asarray(native.results["charges"])
    maple.calculate(maple_atoms, ["energy", "forces", "charges"])

    assert maple.results["energy"] == pytest.approx(native_energy, abs=2.0e-10)
    np.testing.assert_allclose(maple.results["forces"], native_forces, rtol=0.0, atol=2.0e-9)
    np.testing.assert_allclose(maple.results["charges"], native_charges, rtol=0.0, atol=2.0e-7)

    # Change the actual kernel on a warm calculator, not just its declaration.
    other_method = "dsf" if method == "simple" else "simple"
    _configure_native_aimnet_coulomb(native_base, other_method)
    native.reset()
    maple._set_lrcoulomb_method(other_method)
    new_reference = native_atoms.get_potential_energy() * EV2HARTREE
    assert abs(new_reference - native_energy) > 1e-5
    assert maple_atoms.get_potential_energy() == pytest.approx(new_reference, abs=2e-10)
    np.testing.assert_allclose(
        maple_atoms.get_forces(), native_atoms.get_forces() * EV2HARTREE,
        rtol=0.0, atol=2e-9,
    )


@pytest.mark.backend
def test_enabling_d4_recomputes_real_ani_energy_instead_of_returning_cache() -> None:
    model_path = _asset("MAPLE_TEST_ANI_MODEL")
    torch = pytest.importorskip("torch")
    pytest.importorskip("tad_dftd4")
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    atoms = molecule("H2O")
    atoms.calc = ANICalculator(
        device=torch.device("cpu"),
        model="ani1x",
        model_path=str(model_path),
        d4=False,
    )
    energy_without_d4 = atoms.get_potential_energy()

    atoms.calc.d4 = True
    energy_with_d4 = atoms.get_potential_energy()
    cold_reference = ANICalculator(
        device=torch.device("cpu"),
        model="ani1x",
        model_path=str(model_path),
        d4=True,
    )
    reference_atoms = atoms.copy()
    reference_atoms.calc = cold_reference

    assert energy_with_d4 == pytest.approx(
        reference_atoms.get_potential_energy(), abs=2.0e-10
    )
    assert abs(energy_with_d4 - energy_without_d4) > 1.0e-8
