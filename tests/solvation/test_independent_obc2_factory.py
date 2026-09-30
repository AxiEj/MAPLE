"""Real model inputs join the same solvent layer through one factory seam."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.filereader.mol2_reader import MOL2Reader

OPERATIONS = frozenset({"energy_forces", "hessian", "hvp"})


def candidate_resolver(gas_cell, solvent_cell):
    """Test-only, directly injected admission; never an installed receipt."""
    return AnalyticCompositionAdmission(
        profile_id="CANDIDATE:factory-contract",
        evidence_id="CANDIDATE:factory-contract",
        gas_admission=AnalyticDerivativeAdmission(
            "CANDIDATE:gas", "CANDIDATE:factory-contract", gas_cell, OPERATIONS
        ),
        solvent_admission=AnalyticDerivativeAdmission(
            "CANDIDATE:solvent", "CANDIDATE:factory-contract", solvent_cell, OPERATIONS
        ),
        operations=OPERATIONS,
    )


def factory(model, atoms, tmp_path, *, resolver=candidate_resolver):
    checkpoint = Path(__file__).resolve().parents[2] / (
        f"maple/function/calculator/model/{model}.pt"
    )
    assert (
        checkpoint.is_file()
    ), f"Required local validation checkpoint missing: {checkpoint}"
    return SetCalculator(
        device="cpu",
        model=model,
        atoms=atoms,
        output=str(tmp_path / f"{model}.out"),
        implicit="gb",
        solvent="water",
        model_options={"hessian": "analytic", "model_path": str(checkpoint)},
        charge_options={"source": "mol2", "mode": "fixed", "label": "factory-test"},
        solvation_options={
            "method": "gb",
            "provider": "torch",
            "model": "obc2",
            "profile": "obc2-mbondi2",
            "nonpolar": "ace",
            "experimental": True,
        },
        task_context={"task": "freq", "method": "mw"},
        _analytic_admission_resolver=resolver,
    )


@pytest.mark.parametrize("model", ["ani2x", "aimnet2", "maceoff23m"])
def test_candidate_factory_composes_each_real_backend_once(model, water_mol2, tmp_path):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    calculator = factory(model, atoms, tmp_path).set_calculator()
    assert calculator.analytic_composition_admission.profile_id.startswith("CANDIDATE:")
    calculator.calculate(atoms, properties=["energy", "forces"])
    gas = calculator._analytic_energy_forces_operation(atoms, need_forces=True)
    solvent = calculator.solvent_correction.evaluate(atoms, need_forces=True)
    assert calculator.results["energy"] == pytest.approx(
        gas.energy_hartree + solvent.energy_hartree, abs=1e-12
    )
    np.testing.assert_allclose(
        calculator.results["forces"],
        gas.forces_hartree_per_angstrom + solvent.forces_hartree_per_angstrom,
        rtol=0,
        atol=1e-12,
    )
    direction = np.ones(3 * len(atoms)) / np.sqrt(3 * len(atoms))
    hessian = calculator.get_hessian(atoms)
    hn, forces, energy = calculator.get_hvp(atoms, direction)
    np.testing.assert_allclose(hn, hessian @ direction, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(
        forces.reshape(-1, 3), calculator.results["forces"], atol=1e-7
    )
    assert energy == pytest.approx(calculator.results["energy"], abs=2e-5)


def test_missing_installed_receipt_fails_in_factory(water_mol2, tmp_path, monkeypatch):
    from maple.function.calculator import analytic_admission

    monkeypatch.setattr(analytic_admission, "INSTALLED_QUALIFICATIONS", ())
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(AnalyticCapabilityError, match="installed qualification"):
        factory("ani2x", atoms, tmp_path, resolver=None).set_calculator()


def test_private_resolver_cannot_bind_a_different_live_cell(water_mol2, tmp_path):
    def wrong_cell(gas, solvent):
        return candidate_resolver(replace(gas, checkpoint_sha256="0" * 64), solvent)

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(AnalyticIdentityError, match="live component cells"):
        factory("ani2x", atoms, tmp_path, resolver=wrong_cell).set_calculator()


def test_ani_preparation_is_not_qualification_and_rejects_domain_drift(water_mol2):
    from maple.function.calculator.ani._ani_calculator import ANICalculator

    calculator = ANICalculator("cpu", model="ani2x", implicit="none", d4=False)
    calculator.prepare_analytic_derivatives()
    assert calculator.analytic_derivative_admission is None
    assert calculator.analytic_composition_admission is None
    assert calculator.analytic_implicit_derivatives_admitted is False
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    atoms.info.update(charge=1, mult=2)
    with pytest.raises(AnalyticIdentityError, match="neutral singlet"):
        calculator._analytic_energy_forces_operation(atoms, need_forces=True)


@pytest.mark.parametrize("model", ["aimnet2", "maceoff23m"])
def test_single_atom_context_has_infinite_cutoff_gap(model, tmp_path):
    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]], info={"charge": 0, "mult": 1})
    checkpoint = Path(__file__).resolve().parents[2] / (
        f"maple/function/calculator/model/{model}.pt"
    )
    calculator = SetCalculator(
        "cpu", model, str(tmp_path / "single-atom.out"), atoms=atoms,
        model_options={"model_path": str(checkpoint)},
    ).set_calculator()
    context = calculator._build_analytic_context(atoms, enforce_cutoff_guard=False)
    assert context.identity.minimum_cutoff_gap_angstrom == float("inf")
