from __future__ import annotations

from types import SimpleNamespace

import pytest
from ase import Atoms

from maple.function.dispatcher.sp.sp import SinglePoint


def _atoms_with_results(results: dict) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SimpleNamespace(results=results)
    return atoms


def test_sp_keeps_missing_solvation_record_silent():
    assert SinglePoint._solvation_lines(_atoms_with_results({})) == []


@pytest.mark.parametrize(
    "solvation",
    [
        None,
        {"energy_hartree": -0.02},
        {
            "energy_hartree": -0.02,
            "gas_energy_hartree": -1.0,
            "combined_energy_hartree": -1.02,
            "components_hartree": {},
            "provenance": {},
        },
    ],
)
def test_sp_rejects_a_present_malformed_solvation_record(solvation):
    with pytest.raises(ValueError, match="Malformed solvation result"):
        SinglePoint._solvation_lines(
            _atoms_with_results({"solvation": solvation})
        )
