import pytest

from maple.function.read.command_control import CommandControl


def test_macepol_pbc_model_name_keeps_hyphens():
    control = CommandControl.from_settings([
        "#model=macepol-pbc-small",
        "#sp",
    ])

    assert control.params["model"] == "macepol-pbc-small"


def test_macepol_pbc_default_dtype_round_trips_to_validator():
    control = CommandControl.from_settings([
        "#model=macepol-pbc-small(default_dtype=float64)",
        "#sp",
    ])

    assert control.params["model_options"]["default_dtype"] == "float64"

    with pytest.raises(ValueError, match="bfloat16.*Supported values"):
        CommandControl.from_settings([
            "#model=macepol-pbc-small(default_dtype=bfloat16)",
            "#sp",
        ])
