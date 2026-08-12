from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.calculator_base import EV2HARTREE
from maple.function.dispatcher.sp.sp import SinglePoint


class _CountingEnergyForceCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def __init__(self) -> None:
        super().__init__()
        self.calculate_calls = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calculate_calls += 1
        self.results = {
            "energy": -1.25,
            "forces": np.array([[0.1, -0.2, 0.3]], dtype=float),
        }


def _atoms_with_counting_calculator() -> tuple[Atoms, _CountingEnergyForceCalculator]:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = _CountingEnergyForceCalculator()
    atoms.calc = calculator
    return atoms, calculator


def test_verbose_single_point_requests_energy_and_forces_once(tmp_path):
    atoms, calculator = _atoms_with_counting_calculator()
    output = tmp_path / "single.out"

    SinglePoint(str(output), atoms, {"sp": {"verbose": 1}}).run()

    assert calculator.calculate_calls == 1
    text = output.read_text(encoding="utf-8")
    assert f"Energy: {-1.25 * EV2HARTREE:.10f} Hartree" in text
    assert "Gradients (Hartree/Angstrom):" in text
    assert f"{-0.1 * EV2HARTREE:.8f}" in text
    assert f"{0.2 * EV2HARTREE:.8f}" in text
    assert f"{-0.3 * EV2HARTREE:.8f}" in text
    # The direct job boundary must not rewrite public ASE result units.
    assert calculator.results["energy"] == -1.25
    np.testing.assert_array_equal(calculator.results["forces"], [[0.1, -0.2, 0.3]])


def test_verbose_trajectory_evaluates_each_frame_once(tmp_path):
    first, first_calculator = _atoms_with_counting_calculator()
    second, second_calculator = _atoms_with_counting_calculator()
    second.positions[0, 0] = 0.1
    output = tmp_path / "trajectory.out"

    SinglePoint(
        str(output),
        [first, second],
        {"sp": {"verbose": 1}},
    ).run()

    assert first_calculator.calculate_calls == 1
    assert second_calculator.calculate_calls == 1
    text = output.read_text(encoding="utf-8")
    assert text.count(f"Energy: {-1.25 * EV2HARTREE:.10f} Hartree") == 2
    assert text.count("Gradients (Hartree/Angstrom):") == 2
