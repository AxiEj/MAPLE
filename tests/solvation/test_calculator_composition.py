from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLength, FixCartesian

from maple.function.calculator.calculator_base import (
    IMPLICIT_SOLVENT_FORCE_ERROR,
    CalcABC,
    init_implicit_solvent,
    numerical_hessian_from_atoms,
)
from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.dispatcher.sp.sp import SinglePoint


class DummyCorrection:
    supported_properties = {"energy", "forces"}

    def evaluate(self, atoms, need_forces=False, calculator=None):
        return SolvationResult(
            energy_hartree=0.25,
            forces_hartree_per_angstrom=(
                np.full((len(atoms), 3), 0.5) if need_forces else None
            ),
            components_hartree={"polar": 0.2, "nonpolar": 0.05},
            provenance={"provider": "dummy"},
        )


class EnergyOnlyCorrection:
    supported_properties = {"energy"}


class InternallyBrokenCorrection:
    supported_properties = {"energy", "forces"}

    def evaluate(self, atoms, need_forces=False, calculator=None):
        if calculator is not None:
            raise TypeError("primary contract bug")
        return SolvationResult(
            energy_hartree=0.25,
            forces_hartree_per_angstrom=(
                np.full((len(atoms), 3), 0.5) if need_forces else None
            ),
            components_hartree={"polar": 0.2, "nonpolar": 0.05},
            provenance={"provider": "invalid-fallback"},
        )


class QuadraticCorrection:
    supported_properties = {"energy", "forces"}

    def __init__(self, force_constant_hartree_per_angstrom2):
        self.force_constant = float(force_constant_hartree_per_angstrom2)

    def evaluate(self, atoms, need_forces=False, **_kwargs):
        positions = atoms.get_positions()
        energy = 0.5 * self.force_constant * float(np.square(positions).sum())
        forces = -self.force_constant * positions if need_forces else None
        return SolvationResult(
            energy_hartree=energy,
            forces_hartree_per_angstrom=forces,
            components_hartree={"polar": energy, "nonpolar": 0.0},
            provenance={"provider": "quadratic-test"},
        )


class QuadraticGasCalculator(CalcABC):
    MODEL_ENERGY_UNIT = "hartree"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")

    def __init__(self, force_constant_hartree_per_angstrom2):
        super().__init__()
        self.force_constant = float(force_constant_hartree_per_angstrom2)
        self.hessian = "numerical"

    def calculate(self, atoms=None, properties=None, system_changes=None):
        atoms = super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        self._finalize_results(
            atoms,
            energy=0.5 * self.force_constant * float(np.square(positions).sum()),
            forces=-self.force_constant * positions,
            unit="hartree",
        )

    def _analytic_hessian(self, atoms):
        return np.eye(3 * len(atoms)) * self.force_constant


class PluginWithoutImplicitComposition:
    SUPPORTS_PBC = False
    SUPPORTS_CHARGE_MULT = True
    SUPPORTED_HESSIAN_MODES = ("numerical",)


class NonConservativeCalculator(CalcABC):
    MODEL_ENERGY_UNIT = "hartree"

    def calculate(self, atoms=None, properties=None, system_changes=None):
        atoms = super().calculate(atoms, properties, system_changes)
        x, y, z = atoms.get_positions()[0]
        self._finalize_results(
            atoms,
            energy=0.0,
            forces=np.asarray([[-x - 2.0 * y, -3.0 * x - y, -z]]),
            unit="hartree",
        )


def test_calcabc_implicit_solvation_capability_is_opt_in():
    class UnreviewedCalculator(CalcABC):
        pass

    assert CalcABC.SUPPORTS_IMPLICIT_SOLVATION is False
    assert UnreviewedCalculator.SUPPORTS_IMPLICIT_SOLVATION is False


def test_legacy_gbsa_qeq_constructor_path_is_removed():
    class Dummy:
        pass

    with pytest.raises(ValueError, match="legacy implicit='gbsa'"):
        init_implicit_solvent(Dummy(), "gbsa", "water", "cpu")


def test_uma_has_no_unstructured_legacy_solvent_fallback():
    source = (
        Path(__file__).resolve().parents[2]
        / "maple/function/calculator/uma/_uma_calculator.py"
    ).read_text(encoding="utf-8")

    assert "self.chargecalc(calc_atoms" not in source
    assert "self.solvent_correction.get_energy(calc_atoms)" not in source


def test_registered_plugin_without_route1_composition_capability_fails_closed(
    tmp_path,
):
    setter = SetCalculator(
        "cpu",
        "plugin-without-route1",
        str(tmp_path / "plugin.log"),
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        implicit="gb",
        solvent="water",
    )

    with pytest.raises(
        NotImplementedError,
        match="SUPPORTS_IMPLICIT_SOLVATION",
    ):
        setter._validate_against_class(PluginWithoutImplicitComposition)


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


def test_shared_finalizer_does_not_hide_provider_internal_type_errors():
    calc = CalcABC()
    calc.solvent_correction = InternallyBrokenCorrection()
    atoms = Atoms("H", positions=[[0, 0, 0]])

    with pytest.raises(TypeError, match="primary contract bug"):
        calc._finalize_results(
            atoms,
            energy=1.0,
            forces=np.asarray([[1.0, 2.0, 3.0]]),
            unit="hartree",
        )


def test_energy_only_provider_fails_closed_before_gas_only_force_can_escape():
    calc = CalcABC()
    calc.solvent_correction = EnergyOnlyCorrection()
    atoms = Atoms("H", positions=[[0, 0, 0]])

    with pytest.raises(NotImplementedError, match=IMPLICIT_SOLVENT_FORCE_ERROR):
        calc.calculate(atoms, properties=["forces"])


def test_numerical_hessian_differentiates_the_complete_composed_potential():
    atoms = Atoms("H2", positions=[[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]])
    calc = QuadraticGasCalculator(0.7)
    calc.solvent_correction = QuadraticCorrection(0.2)
    calc._finalize_results(
        atoms,
        energy=1.25,
        forces=np.zeros((2, 3)),
        unit="hartree",
    )
    previous_results = dict(calc.results)
    previous_solvation_result = calc.solvation_result

    hessian = calc.get_hessian(atoms, delta=1.0e-4)

    assert np.allclose(hessian, np.eye(6) * 0.9, atol=1.0e-10)
    assert calc.results == previous_results
    assert calc.solvation_result is previous_solvation_result


def test_numerical_hessian_can_report_presymmetrization_diagnostics():
    atoms = Atoms("H", positions=[[0.2, -0.1, 0.3]])
    calc = NonConservativeCalculator()

    hessian, diagnostics = numerical_hessian_from_atoms(
        calc,
        atoms,
        delta=1.0e-4,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(
        hessian,
        [[1.0, 2.5, 0.0], [2.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1.0e-12,
    )
    assert diagnostics == {
        "cartesian_displacement_angstrom": 1.0e-4,
        "maximum_absolute_raw_hessian_hartree_per_angstrom2": (
            pytest.approx(3.0)
        ),
        "maximum_raw_asymmetry_hartree_per_angstrom2": pytest.approx(1.0),
        "raw_hessian_frobenius_norm_hartree_per_angstrom2": (
            pytest.approx(4.0)
        ),
        "raw_asymmetry_frobenius_norm_hartree_per_angstrom2": (
            pytest.approx(np.sqrt(2.0))
        ),
        "relative_raw_asymmetry_frobenius": (
            pytest.approx(np.sqrt(2.0) / 4.0)
        ),
    }
    assert isinstance(
        numerical_hessian_from_atoms(calc, atoms, delta=1.0e-4),
        np.ndarray,
    )


@pytest.mark.parametrize(
    "constraint",
    [
        FixBondLength(0, 1),
        FixCartesian(0, mask=(True, False, False)),
    ],
    ids=["fixed-bond-length", "fixed-cartesian"],
)
def test_numerical_hessian_rejects_unimplemented_constraints(constraint):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    atoms.set_constraint(constraint)

    with pytest.raises(NotImplementedError, match="FixAtoms"):
        numerical_hessian_from_atoms(
            QuadraticGasCalculator(0.7),
            atoms,
            delta=1.0e-4,
        )


def test_numerical_hessian_fixatoms_phva_rows_and_columns_are_zero():
    atoms = Atoms("H2", positions=[[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]])
    atoms.set_constraint(FixAtoms(indices=[0]))

    hessian = numerical_hessian_from_atoms(
        QuadraticGasCalculator(0.7),
        atoms,
        delta=1.0e-4,
    )

    np.testing.assert_allclose(hessian[:3, :], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(hessian[:, :3], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(hessian[3:, 3:], np.eye(3) * 0.7, atol=1.0e-10)


def test_analytic_hessian_with_implicit_solvent_remains_fail_closed():
    atoms = Atoms("H", positions=[[0.2, -0.1, 0.3]])
    calc = QuadraticGasCalculator(0.7)
    calc.hessian = "analytic"
    calc.solvent_correction = QuadraticCorrection(0.2)

    with pytest.raises(NotImplementedError, match=IMPLICIT_SOLVENT_FORCE_ERROR):
        calc.get_hessian(atoms)


def test_numerical_hessian_with_energy_only_solvent_remains_fail_closed():
    atoms = Atoms("H", positions=[[0.2, -0.1, 0.3]])
    calc = QuadraticGasCalculator(0.7)
    calc.solvent_correction = EnergyOnlyCorrection()

    with pytest.raises(NotImplementedError, match=IMPLICIT_SOLVENT_FORCE_ERROR):
        calc.get_hessian(atoms)
