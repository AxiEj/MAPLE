from __future__ import annotations

from pathlib import Path

import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_rism_thermodynamic_source import (
    V0_RISM_SELF_TEST_CONSTRUCTION,
    load_route2_v0_rism_self_test,
    parse_route2_v0_rism_self_test,
)

_WATER_SELF_TEST = """NET CHARGE NEUTRALITY [sqrt(kT A)]
  ZERO Input from MDL          0.0000000000000000E+00
  ZERO Sum of excess charges   4.0783888266560098E-08

EXCESS CHEMICAL POTENTIAL [kT]
  Equation                                  O                        H1
  Pettitt-Rossky (PR)   -1.1988603301683053E+01  6.3352252572422012E+00
  Singer-Chandler (SC)  -1.6725834591508658E+01  5.1443673849019698E+00
  Schmeer-Maurer (SM)   -1.5008644390925495E+01  6.0179833774532581E+00

TOTAL EXCESS FREE ENERGY PER UNIT VOLUME [kT/A^3]
  Free energy                           -3.8826435474256121E-01
  Sum of ExChem (SM) - Excess pressure  -3.8826435474376891E-01
Relative difference [kT/A^3]
  ZERO ExChem (SM) - ExP to Free Energy -3.1105111541542655E-12

Pressure [kT/A^3]
  Pressure (free energy)   1.2193949411141802E-01
  Pressure (virial)        5.2406569568823602E-01
Relative difference [kT/A^3]
  ZERO Pressure           -3.2977519261264852E+00
"""


def test_rism_self_test_parses_real_water_thermodynamic_identities():
    result = parse_route2_v0_rism_self_test(_WATER_SELF_TEST)

    assert result.construction == V0_RISM_SELF_TEST_CONSTRUCTION
    assert result.input_net_charge_sqrt_kt_angstrom == pytest.approx(0.0)
    assert result.excess_charge_sum_sqrt_kt_angstrom == pytest.approx(
        4.0783888266560098e-8
    )
    assert result.total_excess_free_energy_kt_per_angstrom3 == pytest.approx(
        -0.38826435474256121
    )
    assert result.sm_identity_relative_residual == pytest.approx(
        -3.1105111541542655e-12
    )
    assert result.pressure_free_energy_kt_per_angstrom3 == pytest.approx(
        0.12193949411141802
    )
    assert result.pressure_virial_kt_per_angstrom3 == pytest.approx(0.52406569568823602)


def test_rism_self_test_loader_reads_one_hash_bound_file(tmp_path: Path):
    path = tmp_path / "water.self.test"
    path.write_text(_WATER_SELF_TEST, encoding="utf-8")

    result = load_route2_v0_rism_self_test(path)

    assert result.pressure_virial_kt_per_angstrom3 != pytest.approx(
        result.pressure_free_energy_kt_per_angstrom3
    )


def test_rism_self_test_rejects_missing_and_nonfinite_fields():
    missing = _WATER_SELF_TEST.replace(
        "  Pressure (free energy)   1.2193949411141802E-01\n",
        "",
    )
    with pytest.raises(ValueError, match="missing required field"):
        parse_route2_v0_rism_self_test(missing)

    nonfinite = _WATER_SELF_TEST.replace("5.2406569568823602E-01", "NaN")
    with pytest.raises(ValueError, match="must be finite"):
        parse_route2_v0_rism_self_test(nonfinite)


def test_rism_self_test_rejects_duplicate_conflicting_field():
    duplicate = _WATER_SELF_TEST.replace(
        "  Pressure (virial)        5.2406569568823602E-01\n",
        "  Pressure (virial)        5.2406569568823602E-01\n"
        "  Pressure (virial)        6.2406569568823602E-01\n",
    )

    with pytest.raises(ValueError, match="duplicate field virial-route pressure"):
        parse_route2_v0_rism_self_test(duplicate)


@pytest.mark.parametrize(
    ("duplicate", "message"),
    [
        (
            "\n  Pressure (virial)        6.2406569568823602E-01\n",
            "duplicate field virial-route pressure",
        ),
        (
            "\n  ZERO Input from MDL          1.0000000000000000E-03\n",
            "duplicate field input net charge",
        ),
    ],
)
def test_rism_self_test_rejects_cross_section_or_trailing_duplicate(
    duplicate,
    message,
):
    with pytest.raises(ValueError, match=message):
        parse_route2_v0_rism_self_test(_WATER_SELF_TEST + duplicate)


def test_rism_self_test_rejects_duplicate_input_charge_inside_pressure_section():
    duplicate = _WATER_SELF_TEST.replace(
        "Pressure [kT/A^3]\n",
        "Pressure [kT/A^3]\n  ZERO Input from MDL          1.0000000000000000E-03\n",
    )

    with pytest.raises(ValueError, match="duplicate field input net charge"):
        parse_route2_v0_rism_self_test(duplicate)


def test_rism_self_test_rejects_field_in_wrong_section():
    misplaced = _WATER_SELF_TEST.replace(
        "  Pressure (virial)        5.2406569568823602E-01\n",
        "",
    ).replace(
        "  Free energy                           -3.8826435474256121E-01\n",
        "  Free energy                           -3.8826435474256121E-01\n"
        "  Pressure (virial)        5.2406569568823602E-01\n",
    )

    with pytest.raises(
        ValueError, match="missing required field virial-route pressure"
    ):
        parse_route2_v0_rism_self_test(misplaced)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("NET CHARGE NEUTRALITY [sqrt(kT A)]", "NET CHARGE NEUTRALITY [e]"),
        (
            "TOTAL EXCESS FREE ENERGY PER UNIT VOLUME [kT/A^3]",
            "TOTAL EXCESS FREE ENERGY PER UNIT VOLUME [kcal/mol/A^3]",
        ),
        ("Relative difference [kT/A^3]", "Relative difference [kcal/mol/A^3]"),
        ("Pressure [kT/A^3]", "Pressure [bar]"),
    ],
)
def test_rism_self_test_rejects_section_unit_drift(old, new):
    with pytest.raises(ValueError, match="missing required section"):
        parse_route2_v0_rism_self_test(_WATER_SELF_TEST.replace(old, new))


def test_rism_self_test_rejects_failed_zero_identities():
    charge_failure = _WATER_SELF_TEST.replace("4.0783888266560098E-08", "4.0E-03")
    with pytest.raises(ValueError, match="excess-charge neutrality"):
        parse_route2_v0_rism_self_test(charge_failure)

    energy_failure = _WATER_SELF_TEST.replace("-3.1105111541542655E-12", "2.0E-04")
    with pytest.raises(ValueError, match="SM free-energy identity"):
        parse_route2_v0_rism_self_test(energy_failure)
