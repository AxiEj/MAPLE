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
    text = (
        water_mol2.read_text()
        .replace(" 3 2 1 0 0", " 3 1 1 0 0")
        .replace("     2    1    3 1\n", "")
    )
    water_mol2.write_text(text)
    with pytest.raises(ValueError, match="connected molecule"):
        MOL2Reader(str(water_mol2), charge=0, mult=1)


def test_mol2_reader_does_not_renormalize_charge(water_mol2):
    text = water_mol2.read_text().replace("0.417000", "0.400000", 1)
    water_mol2.write_text(text)
    with pytest.raises(ValueError, match="sum.*declared"):
        MOL2Reader(str(water_mol2), charge=0, mult=1, validate_charge=True)


@pytest.mark.parametrize(
    "name,atom_type,element",
    [
        ("CA", "ca", "C"),
        ("NA", "na", "N"),
        ("HO", "ho", "H"),
        ("C", "Ca", "Ca"),
        ("N", "Na", "Na"),
        ("HO", "H", "H"),
    ],
)
def test_explicit_type_not_adversarial_atom_name(water_mol2, name, atom_type, element):
    text = (
        water_mol2.read_text()
        .replace("O1          ", f"{name:<12}")
        .replace("O.3", atom_type)
    )
    water_mol2.write_text(text)
    atoms = MOL2Reader(str(water_mol2))
    assert atoms.get_chemical_symbols()[0] == element
    assert atoms.info["mol2"]["atom_types"][0] == atom_type
    assert atoms.info["mol2"]["atom_names"][0] == name


@pytest.mark.parametrize("atom_type", ["madeup", "c999", "ca.unknown", "X"])
def test_unknown_type_cannot_be_rescued_by_atom_name(water_mol2, atom_type):
    water_mol2.write_text(water_mol2.read_text().replace("O.3", atom_type))
    with pytest.raises(ValueError, match="atom type"):
        MOL2Reader(str(water_mol2))


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "1e999"])
@pytest.mark.parametrize("field", ["coordinate", "charge"])
def test_nonfinite_data_rejected_without_charge_validation(water_mol2, token, field):
    original = "0.957200" if field == "coordinate" else "-0.834000"
    water_mol2.write_text(water_mol2.read_text().replace(original, token, 1))
    with pytest.raises(ValueError, match="finite"):
        MOL2Reader(str(water_mol2))
