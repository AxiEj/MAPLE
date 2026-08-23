from __future__ import annotations

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import MOL2Reader, mol2_atoms_from_bytes


def test_mol2_reader_preserves_order_topology_and_partial_charges(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert np.allclose(atoms.get_initial_charges(), [-0.834, 0.417, 0.417])
    assert atoms.info["charge"] == 0
    assert atoms.info["mult"] == 1
    assert atoms.info["mol2"]["bonds"] == [[0, 1, "1"], [0, 2, "1"]]
    assert atoms.info["mol2"]["charge_type"] == "USER_CHARGES"


def test_captured_bytes_parser_never_reopens_mutated_path(water_mol2):
    captured = water_mol2.read_bytes()
    expected = MOL2Reader(str(water_mol2), charge=0, mult=1)
    water_mol2.write_text("not a molecule", encoding="utf-8")

    atoms = mol2_atoms_from_bytes(
        captured,
        source_id="captured-sha256:test",
        charge=0,
        mult=1,
    )

    assert atoms.get_chemical_symbols() == expected.get_chemical_symbols()
    assert np.array_equal(atoms.positions, expected.positions)
    assert np.array_equal(atoms.get_initial_charges(), expected.get_initial_charges())
    assert atoms.info["mol2"]["path"] == "captured-sha256:test"


def test_captured_bytes_parser_rejects_invalid_utf8_without_repair() -> None:
    with pytest.raises(ValueError, match="not valid UTF-8"):
        mol2_atoms_from_bytes(
            b"@<TRIPOS>MOLECULE\ninvalid-\xff\n",
            source_id="captured-sha256:invalid",
        )


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
    ("old", "new", "message"),
    (
        ("0.957200", "nan", "coordinates must be finite"),
        ("-0.834000", "inf", "partial charge must be finite"),
    ),
)
def test_mol2_reader_rejects_nonfinite_numeric_leaves(
    water_mol2, old: str, new: str, message: str
) -> None:
    water_mol2.write_text(
        water_mol2.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        MOL2Reader(str(water_mol2), charge=0, mult=1)
