"""MOL2 element-resolution regressions exposed by Route-2 topology input."""

from __future__ import annotations

from maple.function.read.filereader.mol2_reader import _element_from_mol2


def test_uppercase_atom_name_does_not_turn_gaff_aromatic_carbon_into_calcium():
    # `ca` is the lowercase GAFF/GAFF2 aromatic-carbon atom type.  An atom name
    # such as `CA` must not be title-cased to the element Ca.
    assert _element_from_mol2("CA", "ca") == "C"


def test_lowercase_two_letter_halogen_types_still_resolve_correctly():
    assert _element_from_mol2("CL", "cl") == "Cl"
    assert _element_from_mol2("BR", "br") == "Br"
