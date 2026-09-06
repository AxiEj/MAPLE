"""The existing provider must not round high-precision upstream energy output.

The numerical repair belongs at the Fortran/C output producer, not in a
post-parser correction or a replacement force model. The isolated producer
writes its full-precision record after the unchanged human-readable block.
"""

import pytest

from maple.function.calculator.extra_correction.implicit.amber_chagb import (
    parse_gbnsr6_components,
    parse_pbsa_components,
)


@pytest.mark.parametrize(
    "parser,legacy,marker,expected",
    [
        (
            parse_gbnsr6_components,
            "EGB = -6.2823 ESURF = 0.0012",
            (
                "MAPLE_GBNSR6_ENERGY_V1 EGB= -6.2823123456789012E+000 "
                "ESURF= 1.2000123456789012E-003"
            ),
            {"polar": -6.2823123456789012, "surface_tension": 0.0012000123456789012},
        ),
        (
            parse_pbsa_components,
            "ECAVITY = 20.9485 EDISPER = -18.7755",
            (
                "MAPLE_PBSA_ENERGY_V1 ECAVITY= 2.0948450000000000E+001 "
                "EDISPER= -1.8775526364632878E+001"
            ),
            {"cavity": 20.94845, "dispersion": -18.775526364632878},
        ),
    ],
)
def test_machine_components_preserve_precision(parser, legacy, marker, expected):
    assert parser(legacy + "\n" + marker + "\n") == expected


def test_last_complete_energy_block_wins():
    text = (
        "MAPLE_GBNSR6_ENERGY_V1 EGB= -1.123456789 ESURF= 2.123456789\n"
        "EGB = -6.2823 ESURF = 1.2000\n"
        "MAPLE_GBNSR6_ENERGY_V1 EGB= -6.2823123456789 ESURF= 1.2000123456789\n"
    )
    assert parse_gbnsr6_components(text)["polar"] == -6.2823123456789


def test_legacy_outputs_remain_supported():
    assert parse_gbnsr6_components("EGB = -6.2823 ESURF = 1.2") == {
        "polar": -6.2823,
        "surface_tension": 1.2,
    }
    assert parse_pbsa_components("ECAVITY = 20.9485 EDISPER = -18.7755") == {
        "cavity": 20.9485,
        "dispersion": -18.7755,
    }


def test_nonfinite_energy_is_still_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        parse_gbnsr6_components("MAPLE_GBNSR6_ENERGY_V1 EGB= 1e999 ESURF= 1")
    with pytest.raises(ValueError, match="non-finite"):
        parse_pbsa_components("MAPLE_PBSA_ENERGY_V1 ECAVITY= 1 EDISPER= -1e999")
