"""Prospective binary64 closure contract for additive energy ledgers.

The scored difference is constructed directly from its physical components.
The large composite scalar is constructed independently.  Exact-dyadic shadow
sums validate both constructions bit-for-bit; an exact-rational local ULP
envelope is used only for the redundant real-arithmetic relation between them.

This module is I/O-free and admits no capability.  The closed MNSol-10 attempt
did not use this contract and must never be reinterpreted or rerun with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import struct
from typing import Iterable

FLOAT64_ADDITIVE_LEDGER_CONTRACT_ID = (
    "maple-route2-float64-additive-ledger-exact-dyadic-ulp-v1"
)
EV_TO_KCAL_MOL = float.fromhex("0x1.70f8010085c0dp+4")


class Float64LedgerClosureError(ValueError):
    """Raised when a stored binary64 ledger violates its frozen construction."""


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Float64LedgerClosureError(f"{name} must be a finite binary64 value")
    result = float(value)
    if not math.isfinite(result):
        raise Float64LedgerClosureError(f"{name} must be finite")
    return result


def binary64_bits(value: object) -> int:
    number = _finite(value, name="binary64 value")
    return struct.unpack(">Q", struct.pack(">d", number))[0]


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _power_of_two_exponent(value: int) -> int:
    if value < 1 or value & (value - 1):
        raise Float64LedgerClosureError(
            "binary64 exact-rational denominator must be a power of two"
        )
    return value.bit_length() - 1


def _round_right_ties_even(value: int, shift: int) -> int:
    if shift < 1:
        return value << (-shift)
    quotient = value >> shift
    remainder = value - (quotient << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return quotient


def _round_exact_dyadic_to_binary64(numerator: int, denominator_shift: int) -> float:
    """Round ``numerator * 2**-denominator_shift`` once, ties-to-even."""

    if denominator_shift < 0:
        numerator <<= -denominator_shift
        denominator_shift = 0
    if numerator == 0:
        return 0.0
    sign = -1.0 if numerator < 0 else 1.0
    magnitude = abs(numerator)
    bit_count = magnitude.bit_length()
    exponent = bit_count - 1 - denominator_shift

    if exponent >= -1022:
        significand = _round_right_ties_even(magnitude, bit_count - 53)
        if significand == 1 << 53:
            significand >>= 1
            exponent += 1
        if exponent > 1023:
            raise Float64LedgerClosureError("exact dyadic sum overflows binary64")
        try:
            result = math.ldexp(float(significand), exponent - 52)
        except OverflowError as exc:
            raise Float64LedgerClosureError(
                "exact dyadic sum overflows binary64"
            ) from exc
    else:
        subnormal_units = _round_right_ties_even(magnitude, denominator_shift - 1074)
        if subnormal_units == 0:
            return -0.0 if sign < 0.0 else 0.0
        if subnormal_units > 1 << 52:
            raise Float64LedgerClosureError(
                "subnormal rounding produced an invalid significand"
            )
        result = math.ldexp(float(subnormal_units), -1074)
    result = math.copysign(result, sign)
    if not math.isfinite(result):
        raise Float64LedgerClosureError("exact dyadic rounding is non-finite")
    return _canonical_zero(result)


def exact_dyadic_sum_reference(values: Iterable[object]) -> float:
    numbers = tuple(_finite(value, name="sum component") for value in values)
    if not numbers:
        raise Float64LedgerClosureError("exact-dyadic sum requires components")
    ratios = [number.as_integer_ratio() for number in numbers]
    shifts = [_power_of_two_exponent(denominator) for _numerator, denominator in ratios]
    common_shift = max(shifts)
    exact_integer = sum(
        component_numerator << (common_shift - component_shift)
        for (component_numerator, _denominator), component_shift in zip(
            ratios, shifts, strict=True
        )
    )
    return _round_exact_dyadic_to_binary64(exact_integer, common_shift)


def exact_dyadic_product_reference(first: object, second: object) -> float:
    left = _finite(first, name="product left operand")
    right = _finite(second, name="product right operand")
    left_numerator, left_denominator = left.as_integer_ratio()
    right_numerator, right_denominator = right.as_integer_ratio()
    return _round_exact_dyadic_to_binary64(
        left_numerator * right_numerator,
        _power_of_two_exponent(left_denominator)
        + _power_of_two_exponent(right_denominator),
    )


def canonical_fsum(values: Iterable[object]) -> float:
    numbers = tuple(_finite(value, name="sum component") for value in values)
    if not numbers:
        raise Float64LedgerClosureError("canonical fsum requires components")
    try:
        result = math.fsum(numbers)
    except (OverflowError, ValueError) as exc:
        raise Float64LedgerClosureError("canonical fsum failed") from exc
    if not math.isfinite(result):
        raise Float64LedgerClosureError("canonical fsum is non-finite")
    return _canonical_zero(result)


def canonical_product(first: object, second: object) -> float:
    left = _finite(first, name="product left operand")
    right = _finite(second, name="product right operand")
    result = left * right
    if not math.isfinite(result):
        raise Float64LedgerClosureError("canonical product is non-finite")
    return _canonical_zero(result)


def _require_bits(actual: object, expected: float, *, name: str) -> float:
    value = _finite(actual, name=name)
    if binary64_bits(value) != binary64_bits(expected):
        raise Float64LedgerClosureError(f"{name} differs from canonical binary64 bits")
    return value


def _canonical_sum_with_shadow(values: tuple[float, ...], *, name: str) -> float:
    canonical = canonical_fsum(values)
    shadow = exact_dyadic_sum_reference(values)
    if binary64_bits(canonical) != binary64_bits(shadow):
        raise Float64LedgerClosureError(
            f"{name} math.fsum differs from exact-dyadic rounded shadow"
        )
    return canonical


def _canonical_product_with_shadow(first: float, second: float, *, name: str) -> float:
    canonical = canonical_product(first, second)
    shadow = exact_dyadic_product_reference(first, second)
    if binary64_bits(canonical) != binary64_bits(shadow):
        raise Float64LedgerClosureError(
            f"{name} product differs from exact-dyadic rounded shadow"
        )
    return canonical


@dataclass(frozen=True, slots=True)
class ExactEnvelope:
    residual: Fraction
    bound: Fraction

    @property
    def passes(self) -> bool:
        return abs(self.residual) <= self.bound

    def as_dict(self) -> dict[str, object]:
        return {
            "residual_ratio": [self.residual.numerator, self.residual.denominator],
            "bound_ratio": [self.bound.numerator, self.bound.denominator],
            "passes": self.passes,
        }


def cross_ledger_ulp_envelope(
    *, vacuum_energy_eV: object, solution_total_energy_eV: object, delta_eV: object
) -> ExactEnvelope:
    vacuum = _finite(vacuum_energy_eV, name="vacuum_energy_eV")
    total = _finite(solution_total_energy_eV, name="solution_total_energy_eV")
    delta = _finite(delta_eV, name="delta_eV")
    residual = (
        Fraction.from_float(total)
        - Fraction.from_float(vacuum)
        - Fraction.from_float(delta)
    )
    bound = (
        Fraction.from_float(math.ulp(total)) + Fraction.from_float(math.ulp(delta))
    ) / 2
    return ExactEnvelope(residual=residual, bound=bound)


def kcal_decomposition_ulp_envelope(
    *,
    delta_eV: object,
    delta_kcal_mol: object,
    polarization_kcal_mol: object,
    cds_kcal_mol: object,
) -> ExactEnvelope:
    delta = _finite(delta_eV, name="delta_eV")
    delta_kcal = _finite(delta_kcal_mol, name="delta_kcal_mol")
    polarization = _finite(polarization_kcal_mol, name="polarization_kcal_mol")
    cds = _finite(cds_kcal_mol, name="cds_kcal_mol")
    residual = (
        Fraction.from_float(delta_kcal)
        - Fraction.from_float(polarization)
        - Fraction.from_float(cds)
    )
    bound = (
        abs(Fraction.from_float(EV_TO_KCAL_MOL)) * Fraction.from_float(math.ulp(delta))
        + Fraction.from_float(math.ulp(delta_kcal))
        + Fraction.from_float(math.ulp(polarization))
        + Fraction.from_float(math.ulp(cds))
    ) / 2
    return ExactEnvelope(residual=residual, bound=bound)


@dataclass(frozen=True, slots=True)
class AdditiveFloat64Closure:
    delta_eV: float
    phi_eV: float
    delta_kcal_mol: float
    polarization_kcal_mol: float
    cds_kcal_mol: float
    combined_charge_e: float
    energy_envelope: ExactEnvelope
    kcal_envelope: ExactEnvelope

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": FLOAT64_ADDITIVE_LEDGER_CONTRACT_ID,
            "bits": {
                "delta_eV": f"{binary64_bits(self.delta_eV):016x}",
                "phi_eV": f"{binary64_bits(self.phi_eV):016x}",
                "delta_kcal_mol": f"{binary64_bits(self.delta_kcal_mol):016x}",
                "polarization_kcal_mol": (
                    f"{binary64_bits(self.polarization_kcal_mol):016x}"
                ),
                "cds_kcal_mol": f"{binary64_bits(self.cds_kcal_mol):016x}",
                "combined_charge_e": f"{binary64_bits(self.combined_charge_e):016x}",
            },
            "energy_cross_ledger_envelope": self.energy_envelope.as_dict(),
            "kcal_decomposition_envelope": self.kcal_envelope.as_dict(),
        }


def validate_additive_float64_closure(
    *,
    vacuum_energy_eV: object,
    polarization_energy_eV: object,
    cds_energy_eV: object,
    solution_total_energy_eV: object,
    predicted_delta_g_eV: object,
    predicted_delta_g_kcal_mol: object,
    polarization_kcal_mol: object,
    cds_kcal_mol: object,
    permanent_charge_e: object,
    induced_charge_e: object,
    combined_charge_e: object,
) -> AdditiveFloat64Closure:
    vacuum = _finite(vacuum_energy_eV, name="vacuum_energy_eV")
    polarization = _finite(polarization_energy_eV, name="polarization_energy_eV")
    cds = _finite(cds_energy_eV, name="cds_energy_eV")
    permanent_charge = _finite(permanent_charge_e, name="permanent_charge_e")
    induced_charge = _finite(induced_charge_e, name="induced_charge_e")

    delta = _canonical_sum_with_shadow((polarization, cds), name="delta_eV")
    phi = _canonical_sum_with_shadow(
        (vacuum, polarization, cds), name="solution_total_energy_eV"
    )
    stored_delta = _require_bits(
        predicted_delta_g_eV, delta, name="predicted_delta_g_eV"
    )
    stored_phi = _require_bits(
        solution_total_energy_eV, phi, name="solution_total_energy_eV"
    )

    delta_kcal = _canonical_product_with_shadow(
        delta, EV_TO_KCAL_MOL, name="predicted_delta_g_kcal_mol"
    )
    polarization_kcal = _canonical_product_with_shadow(
        polarization, EV_TO_KCAL_MOL, name="polarization_kcal_mol"
    )
    cds_kcal = _canonical_product_with_shadow(cds, EV_TO_KCAL_MOL, name="cds_kcal_mol")
    stored_delta_kcal = _require_bits(
        predicted_delta_g_kcal_mol,
        delta_kcal,
        name="predicted_delta_g_kcal_mol",
    )
    stored_polarization_kcal = _require_bits(
        polarization_kcal_mol,
        polarization_kcal,
        name="polarization_kcal_mol",
    )
    stored_cds_kcal = _require_bits(cds_kcal_mol, cds_kcal, name="cds_kcal_mol")

    combined = _canonical_sum_with_shadow(
        (permanent_charge, induced_charge), name="combined_charge_e"
    )
    stored_combined = _require_bits(
        combined_charge_e, combined, name="combined_charge_e"
    )

    energy_envelope = cross_ledger_ulp_envelope(
        vacuum_energy_eV=vacuum,
        solution_total_energy_eV=stored_phi,
        delta_eV=stored_delta,
    )
    if not energy_envelope.passes:
        raise Float64LedgerClosureError("cross-ledger exact ULP envelope failed")
    kcal_envelope = kcal_decomposition_ulp_envelope(
        delta_eV=stored_delta,
        delta_kcal_mol=stored_delta_kcal,
        polarization_kcal_mol=stored_polarization_kcal,
        cds_kcal_mol=stored_cds_kcal,
    )
    if not kcal_envelope.passes:
        raise Float64LedgerClosureError("kcal exact ULP envelope failed")
    return AdditiveFloat64Closure(
        delta_eV=stored_delta,
        phi_eV=stored_phi,
        delta_kcal_mol=stored_delta_kcal,
        polarization_kcal_mol=stored_polarization_kcal,
        cds_kcal_mol=stored_cds_kcal,
        combined_charge_e=stored_combined,
        energy_envelope=energy_envelope,
        kcal_envelope=kcal_envelope,
    )


__all__ = [
    "AdditiveFloat64Closure",
    "EV_TO_KCAL_MOL",
    "ExactEnvelope",
    "FLOAT64_ADDITIVE_LEDGER_CONTRACT_ID",
    "Float64LedgerClosureError",
    "binary64_bits",
    "canonical_fsum",
    "canonical_product",
    "cross_ledger_ulp_envelope",
    "exact_dyadic_product_reference",
    "exact_dyadic_sum_reference",
    "kcal_decomposition_ulp_envelope",
    "validate_additive_float64_closure",
]
