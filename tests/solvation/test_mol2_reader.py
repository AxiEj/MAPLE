from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2Reader,
    _element_from_mol2,
)


def test_mol2_reader_preserves_order_topology_and_partial_charges(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert np.allclose(atoms.get_initial_charges(), [-0.834, 0.417, 0.417])
    assert atoms.info["charge"] == 0
    assert atoms.info["mult"] == 1
    assert atoms.info["mol2"]["atom_ids"] == [1, 2, 3]
    assert atoms.arrays[MOL2_ATOM_ID_ARRAY].tolist() == [1, 2, 3]
    assert atoms.info["mol2"]["source_sha256"] == hashlib.sha256(
        water_mol2.read_bytes()
    ).hexdigest()
    assert atoms.info["mol2"]["bonds"] == [[0, 1, "1"], [0, 2, "1"]]
    assert atoms.info["mol2"]["charge_type"] == "USER_CHARGES"


def test_mol2_reader_does_not_treat_lowercase_gaff_type_as_element(water_mol2):
    text = water_mol2.read_text().replace(" H         1 WAT", " ho        1 WAT")
    water_mol2.write_text(text)

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    assert atoms.get_chemical_symbols() == ["O", "H", "H"]


@pytest.mark.parametrize(
    ("atom_name", "atom_type", "expected"),
    [
        ("CA", "ca", "C"),
        ("NA", "na", "N"),
        ("HO", "ho", "H"),
        ("CA", "Ca", "Ca"),
        ("NA", "Na", "Na"),
        ("CL1", "cl", "Cl"),
        ("BR1", "br", "Br"),
        ("C1", "C.3", "C"),
        ("N1", "N.am", "N"),
    ],
)
def test_mol2_element_resolution_uses_explicit_tripos_or_gaff_types(
    atom_name, atom_type, expected
):
    assert _element_from_mol2(atom_name, atom_type) == expected


def test_mol2_element_resolution_never_guesses_from_ambiguous_atom_name():
    with pytest.raises(ValueError, match="Cannot determine an element"):
        _element_from_mol2("CA", "unknown")


def test_mol2_reader_rejects_duplicate_bond_pairs(water_mol2, tmp_path):
    text = water_mol2.read_text(encoding="utf-8")
    text = text.replace(" 3 2 1 0 0", " 3 3 1 0 0")
    text = text.replace(
        "     2    1    3 1\n",
        "     2    1    3 1\n     3    1    2 1\n",
    )
    path = tmp_path / "duplicate-bond.mol2"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate MOL2 bond between"):
        MOL2Reader(str(path), charge=0, mult=1)


def test_mol2_reader_rejects_self_referential_bonds(water_mol2, tmp_path):
    text = water_mol2.read_text(encoding="utf-8").replace(
        "     1    1    2 1",
        "     1    1    1 1",
    )
    path = tmp_path / "self-bond.mol2"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Self-referential MOL2 bond"):
        MOL2Reader(str(path), charge=0, mult=1)


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
