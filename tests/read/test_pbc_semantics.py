import pytest

from maple.function.read.command_control import CommandControl


def test_two_value_pbc_is_rejected_instead_of_fake_3d_slab():
    with pytest.raises(ValueError, match="2-value slab syntax is not supported"):
        CommandControl.from_settings([
            "#pbc(20,20)",
            "#sp",
        ])


def test_three_value_pbc_remains_full_3d_orthorhombic_cell():
    control = CommandControl.from_settings([
        "#pbc(20,21,22)",
        "#sp",
    ])

    assert control.params["pbc"] == [20.0, 21.0, 22.0, 90.0, 90.0, 90.0]
