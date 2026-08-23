from __future__ import annotations

from fractions import Fraction
import math
import random

import pytest

from maple.solvation.release.float64_ledger import (
    EV_TO_KCAL_MOL,
    Float64LedgerClosureError,
    binary64_bits,
    canonical_fsum,
    canonical_product,
    cross_ledger_ulp_envelope,
    exact_dyadic_product_reference,
    exact_dyadic_sum_reference,
    validate_additive_float64_closure,
)


def _valid_fields(vacuum: float, polarization: float, cds: float):
    delta = canonical_fsum((polarization, cds))
    phi = canonical_fsum((vacuum, polarization, cds))
    permanent_charge = 2.0e-12
    induced_charge = -1.0e-12
    return {
        "vacuum_energy_eV": vacuum,
        "polarization_energy_eV": polarization,
        "cds_energy_eV": cds,
        "solution_total_energy_eV": phi,
        "predicted_delta_g_eV": delta,
        "predicted_delta_g_kcal_mol": canonical_product(delta, EV_TO_KCAL_MOL),
        "polarization_kcal_mol": canonical_product(polarization, EV_TO_KCAL_MOL),
        "cds_kcal_mol": canonical_product(cds, EV_TO_KCAL_MOL),
        "permanent_charge_e": permanent_charge,
        "induced_charge_e": induced_charge,
        "combined_charge_e": canonical_fsum((permanent_charge, induced_charge)),
    }


def test_catastrophic_vacuum_cancellation_passes_only_redundant_ulp_envelope():
    vacuum = float.fromhex("-0x1.0000000000000p+52")
    polarization = float.fromhex("0x1.0000000000000p+0")
    cds = float.fromhex("0x1.0000000000000p-2")
    fields = _valid_fields(vacuum, polarization, cds)
    assert fields["predicted_delta_g_eV"] == 1.25
    assert fields["solution_total_energy_eV"] == -4503599627370495.0
    assert fields["solution_total_energy_eV"] - vacuum == 1.0
    closure = validate_additive_float64_closure(**fields)
    assert closure.energy_envelope.residual == Fraction(-1, 4)
    assert closure.energy_envelope.bound == Fraction(1, 4) + Fraction(1, 2**53)
    assert closure.energy_envelope.passes


def test_observed_scale_class_rejects_old_fixed_tolerance_but_passes_ulp_contract():
    fields = _valid_fields(-100000.123456789, -10.23456789, -3.456789)
    obsolete_residual = (
        fields["solution_total_energy_eV"]
        - fields["vacuum_energy_eV"]
        - fields["predicted_delta_g_eV"]
    )
    assert abs(obsolete_residual) > 1.0e-12
    closure = validate_additive_float64_closure(**fields)
    assert closure.energy_envelope.passes
    assert abs(closure.energy_envelope.residual) <= closure.energy_envelope.bound


@pytest.mark.parametrize("vacuum_sign", (-1.0, 1.0))
@pytest.mark.parametrize("polarization_sign", (-1.0, 1.0))
@pytest.mark.parametrize("cds_sign", (-1.0, 1.0))
def test_all_component_sign_combinations_match_exact_dyadic_shadow(
    vacuum_sign: float, polarization_sign: float, cds_sign: float
):
    fields = _valid_fields(
        vacuum_sign * 2.0**52,
        polarization_sign * 1.0,
        cds_sign * 0.25,
    )
    closure = validate_additive_float64_closure(**fields)
    assert closure.energy_envelope.passes


def test_nextafter_boundary_is_exact_and_never_forgives_stored_total_drift():
    vacuum = -(2.0**52)
    delta = 1.25
    canonical_phi = -4503599627370495.0
    inside = math.nextafter(canonical_phi, math.inf)
    outside = math.nextafter(inside, math.inf)
    assert cross_ledger_ulp_envelope(
        vacuum_energy_eV=vacuum,
        solution_total_energy_eV=inside,
        delta_eV=delta,
    ).passes
    assert not cross_ledger_ulp_envelope(
        vacuum_energy_eV=vacuum,
        solution_total_energy_eV=outside,
        delta_eV=delta,
    ).passes

    fields = _valid_fields(vacuum, 1.0, 0.25)
    for altered in (inside, outside):
        changed = dict(fields, solution_total_energy_eV=altered)
        with pytest.raises(
            Float64LedgerClosureError,
            match="solution_total_energy_eV differs from canonical binary64 bits",
        ):
            validate_additive_float64_closure(**changed)


def test_one_ulp_prediction_or_component_error_fails_before_envelope():
    fields = _valid_fields(-(2.0**52), 1.0, 0.25)
    changed = dict(
        fields,
        predicted_delta_g_eV=math.nextafter(fields["predicted_delta_g_eV"], math.inf),
    )
    with pytest.raises(
        Float64LedgerClosureError,
        match="predicted_delta_g_eV differs from canonical binary64 bits",
    ):
        validate_additive_float64_closure(**changed)

    wrong_ledgers = (
        (1.0, 0.0),
        (1.0, -0.25),
        (1.0, 0.5),
        (1.0, 0.25 * EV_TO_KCAL_MOL),
    )
    for wrong_polarization, wrong_cds in wrong_ledgers:
        changed = dict(
            fields,
            predicted_delta_g_eV=canonical_fsum((wrong_polarization, wrong_cds)),
        )
        with pytest.raises(Float64LedgerClosureError):
            validate_additive_float64_closure(**changed)


def test_exact_dyadic_shadow_matches_fsum_and_one_rounded_product():
    generator = random.Random(20260824)
    for _ in range(1000):
        values = tuple(
            math.ldexp(generator.uniform(-1.0, 1.0), generator.randint(-100, 100))
            for _ in range(3)
        )
        assert binary64_bits(canonical_fsum(values)) == binary64_bits(
            exact_dyadic_sum_reference(values)
        )
        assert binary64_bits(canonical_product(values[0], EV_TO_KCAL_MOL)) == (
            binary64_bits(exact_dyadic_product_reference(values[0], EV_TO_KCAL_MOL))
        )


def test_exceptional_and_subnormal_values_fail_closed_or_use_exact_fraction_bound():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(Float64LedgerClosureError):
            canonical_fsum((value, 1.0))
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    with pytest.raises(Float64LedgerClosureError):
        canonical_fsum((maximum, maximum))
    with pytest.raises(Float64LedgerClosureError):
        exact_dyadic_sum_reference((maximum, maximum))
    tiny = math.ulp(0.0)
    fields = _valid_fields(-tiny, tiny, -tiny)
    closure = validate_additive_float64_closure(**fields)
    assert closure.energy_envelope.bound >= 0
    assert closure.energy_envelope.passes
