from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.result import SolvationResult
from maple.function.calculator.uma import _uma_calculator as uma_module
from maple.function.calculator.uma._uma_calculator import UMACalculator
from maple.function.dispatcher.sp.sp import SinglePoint


def _uma_without_weights(monkeypatch, *, include_free_energy: bool):
    calculator = object.__new__(UMACalculator)
    calculator._auto_task = False
    calculator._task_name = "omol"
    calculator.device = torch.device("cpu")
    calculator.hessian = "numerical"
    calculator.solvent_correction = SimpleNamespace(
        evaluate=lambda _atoms, need_forces=False: SolvationResult(
            energy_hartree=0.25,
            components_hartree={"polar": 0.2, "nonpolar": 0.05},
            provenance={"provider": "dummy", "standard_state": "1 mol/L"},
        )
    )

    def fake_fairchem_calculate(self, atoms, properties, system_changes):
        self.atoms = atoms
        self.results = {"energy": 27.211386245988}
        if include_free_energy:
            self.results["free_energy"] = 54.422772491976

    monkeypatch.setattr(
        uma_module.FAIRChemCalculator,
        "calculate",
        fake_fairchem_calculate,
    )
    return calculator


@pytest.mark.parametrize("include_free_energy", [False, True])
def test_uma_emits_complete_shared_solvation_record(
    monkeypatch, include_free_energy
):
    calculator = _uma_without_weights(
        monkeypatch,
        include_free_energy=include_free_energy,
    )
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = calculator

    results = calculator.calculate(atoms, properties=["energy"])

    assert results["energy"] == pytest.approx(1.25)
    if include_free_energy:
        assert results["free_energy"] == pytest.approx(2.25)
    else:
        assert "free_energy" not in results
    assert results["solvation"] == {
        "energy_hartree": pytest.approx(0.25),
        "gas_energy_hartree": pytest.approx(1.0),
        "combined_energy_hartree": pytest.approx(1.25),
        "components_hartree": {"polar": 0.2, "nonpolar": 0.05},
        "provenance": {"provider": "dummy", "standard_state": "1 mol/L"},
        "ase_free_energy_is_thermochemical_gibbs": False,
        "delta_g_solv_hartree": pytest.approx(0.25),
    }
    rendered = "".join(SinglePoint._solvation_lines(atoms))
    assert "Gas-phase MLIP energy: 1.0000000000 Hartree" in rendered
    assert "Delta G_solv, 1 mol/L): 0.2500000000 Hartree" in rendered
    assert "Combined E_MLIP(gas)+Delta G_solv: 1.2500000000 Hartree" in rendered


def test_uma_does_not_treat_free_energy_as_missing_gas_energy(monkeypatch):
    calculator = _uma_without_weights(monkeypatch, include_free_energy=True)

    def fake_fairchem_calculate(self, atoms, properties, system_changes):
        self.atoms = atoms
        self.results = {"free_energy": 54.422772491976}

    monkeypatch.setattr(
        uma_module.FAIRChemCalculator,
        "calculate",
        fake_fairchem_calculate,
    )

    with pytest.raises(
        ValueError,
        match="requires FAIR-Chem energy output",
    ):
        calculator.calculate(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            properties=["energy"],
        )
