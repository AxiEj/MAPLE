from __future__ import annotations

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import MOL2Reader


def test_mol2_reader_preserves_order_topology_and_partial_charges(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert np.allclose(atoms.get_initial_charges(), [-0.834, 0.417, 0.417])
    assert atoms.info["charge"] == 0
    assert atoms.info["mult"] == 1
    assert atoms.info["mol2"]["bonds"] == [[0, 1, "1"], [0, 2, "1"]]
    assert atoms.info["mol2"]["charge_type"] == "USER_CHARGES"


def test_mol2_reader_does_not_treat_lowercase_gaff_type_as_element(water_mol2):
    text = water_mol2.read_text().replace(" H         1 WAT", " ho        1 WAT")
    water_mol2.write_text(text)

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    assert atoms.get_chemical_symbols() == ["O", "H", "H"]


def test_mol2_reader_rejects_disconnected_molecule(water_mol2):
    text = water_mol2.read_text().replace(" 3 2 1 0 0", " 3 1 1 0 0").replace(
        "     2    1    3 1\n", ""
    )
    water_mol2.write_text(text)
    with pytest.raises(ValueError, match="connected molecule"):
        MOL2Reader(str(water_mol2), charge=0, mult=1)


def test_mol2_reader_does_not_renormalize_charge(water_mol2):
    text = water_mol2.read_text().replace("0.417000", "0.400000", 1)
    water_mol2.write_text(text)
    with pytest.raises(ValueError, match="sum.*declared"):
        MOL2Reader(str(water_mol2), charge=0, mult=1, validate_charge=True)
