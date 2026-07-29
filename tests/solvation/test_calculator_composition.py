from __future__ import annotations

import numpy as np
from ase import Atoms
import pytest

from maple.function.calculator.calculator_base import (
    CalcABC,
    reject_implicit_solvent_derivatives,
)
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.dispatcher.sp.sp import SinglePoint


class DummyCorrection:
    supported_properties = {"energy", "forces"}

    def evaluate(self, atoms, need_forces=False):
        return SolvationResult(
            energy_hartree=0.25,
            forces_hartree_per_angstrom=np.full((len(atoms), 3), 0.5) if need_forces else None,
            components_hartree={"polar": 0.2, "nonpolar": 0.05},
            provenance={"provider": "dummy"},
        )


def test_shared_finalizer_adds_correction_energy_and_force_exactly_once():
    calc = CalcABC()
    calc.solvent_correction = DummyCorrection()
    atoms = Atoms("H", positions=[[0, 0, 0]])
    calc._finalize_results(
        atoms,
        energy=1.0,
        forces=np.asarray([[1.0, 2.0, 3.0]]),
        unit="hartree",
    )
    assert calc.results["energy"] == 1.25
    assert np.allclose(calc.results["forces"], [[1.5, 2.5, 3.5]])
    solvation = calc.results["solvation"]
    assert solvation["components_hartree"]["nonpolar"] == 0.05
    assert solvation["gas_energy_hartree"] == 1.0
    assert solvation["delta_g_solv_hartree"] == 0.25
    assert solvation["combined_energy_hartree"] == 1.25
    assert solvation["ase_free_energy_is_thermochemical_gibbs"] is False


def test_single_point_output_labels_gas_solvation_and_combined_energies():
    calc = CalcABC()
    calc.solvent_correction = DummyCorrection()
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.calc = calc
    calc._finalize_results(atoms, energy=1.0, forces=None, unit="hartree")

    text = "".join(SinglePoint._solvation_lines(atoms))

    assert "Gas-phase MLIP energy: 1.0000000000 Hartree" in text
    assert "Delta G_solv" in text
    assert "0.2500000000 Hartree" in text
    assert "Combined E_MLIP(gas)+Delta G_solv: 1.2500000000 Hartree" in text
    assert "not a thermochemical Gibbs free energy" in text


def test_shared_finalizer_does_not_leak_a_previous_solvation_result():
    calc = CalcABC()
    calc.solvent_correction = DummyCorrection()
    atoms = Atoms("H", positions=[[0, 0, 0]])
    calc._finalize_results(atoms, energy=1.0, forces=None, unit="hartree")
    assert "solvation" in calc.results

    calc.solvent_correction = None
    calc._finalize_results(atoms, energy=2.0, forces=None, unit="hartree")

    assert calc.results["energy"] == 2.0
    assert "solvation" not in calc.results
    assert not hasattr(calc, "solvation_result")


def test_shared_finalizer_records_frequency_type_when_present():
    class FakeCorrection(DummyCorrection):
        pass

    calc = CalcABC()
    calc.solvent_correction = FakeCorrection()
    calc.frequency_type = "effective_solution_pmf"
    atoms = Atoms("H", positions=[[0, 0, 0]])
    calc._finalize_results(
        atoms,
        energy=1.0,
        forces=np.asarray([[1.0, 2.0, 3.0]]),
        unit="hartree",
    )

    solvation = calc.results["solvation"]
    assert solvation["frequency_type"] == "effective_solution_pmf"


def test_numerical_hessian_blocked_with_implicit_correction_without_frequency_type(
    monkeypatch,
):
    called = {"ok": False}

    def _raise_if_called(*_args, **_kwargs):
        called["ok"] = True
        return np.zeros((3, 3))

    monkeypatch.setattr("maple.function.calculator.calculator_base.numerical_hessian_from_atoms", _raise_if_called)

    class FakeCorrection:
        supported_properties = {"energy", "forces"}

    calc = CalcABC()
    calc.solvent_correction = FakeCorrection()
    calc.hessian = "numerical"
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    with pytest.raises(NotImplementedError, match="implicit-solvent provider"):
        calc.get_hessian(atoms)

    assert called["ok"] is False


def test_numerical_hessian_with_frequency_type_uses_numerical_solver(
    monkeypatch,
):
    output = np.arange(36, dtype=float).reshape(6, 6)

    def _fake_numerical_hessian_from_atoms(_calc, _atoms, delta=0.002):
        return output

    monkeypatch.setattr("maple.function.calculator.calculator_base.numerical_hessian_from_atoms", _fake_numerical_hessian_from_atoms)

    class FakeCorrection:
        supported_properties = {"energy", "forces"}

    calc = CalcABC()
    calc.solvent_correction = FakeCorrection()
    calc.frequency_type = "effective_solution_pmf"
    calc.hessian = "numerical"
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    hessian = calc.get_hessian(atoms)

    assert hessian is output


def test_frequency_type_allows_combined_hessian_through_shared_finalizer():
    calc = CalcABC()
    calc.solvent_correction = DummyCorrection()
    calc.frequency_type = "effective_solution_pmf"
    atoms = Atoms("H", positions=[[0, 0, 0]])
    hessian = np.eye(3)

    calc._finalize_results(
        atoms,
        energy=1.0,
        hessian=hessian,
        unit="hartree",
    )

    np.testing.assert_allclose(calc.results["hessian"], hessian)
    assert calc.results["solvation"]["frequency_type"] == "effective_solution_pmf"


def test_frequency_type_allows_hessian_request_but_not_hvp():
    calc = CalcABC()
    calc.solvent_correction = DummyCorrection()
    calc.frequency_type = "effective_solution_pmf"

    assert reject_implicit_solvent_derivatives(calc, ["hessian"]) == ["hessian"]
    with pytest.raises(NotImplementedError, match="implicit-solvent provider"):
        calc.get_hvp(Atoms("H", positions=[[0, 0, 0]]), np.ones(3))
