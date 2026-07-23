from __future__ import annotations

import numpy as np
from ase import Atoms

from maple.function.calculator.calculator_base import CalcABC
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
