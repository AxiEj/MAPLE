"""Semantic audit for AMBER ``rism1d`` ``*.self.test`` output.

The report records source-side thermodynamic identities and two distinct
pressure routes.  Route-2 V0 preserves both pressures and verifies the fields
labelled ``ZERO`` without selecting a route from target solvation accuracy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

V0_RISM_SELF_TEST_CONSTRUCTION = "route2-v0-rism-self-test-v1"
MAX_INPUT_NET_CHARGE_SQRT_KT_ANGSTROM = 1.0e-10
MAX_EXCESS_CHARGE_SUM_SQRT_KT_ANGSTROM = 1.0e-6
MAX_SM_IDENTITY_RELATIVE_RESIDUAL = 1.0e-8
_NET_CHARGE_HEADER = "NET CHARGE NEUTRALITY [sqrt(kT A)]"
_FREE_ENERGY_HEADER = "TOTAL EXCESS FREE ENERGY PER UNIT VOLUME [kT/A^3]"
_RELATIVE_DIFFERENCE_HEADER = "Relative difference [kT/A^3]"
_PRESSURE_HEADER = "Pressure [kT/A^3]"


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _required_line_value(text: str, *, label: str, name: str) -> float:
    matches = tuple(
        re.finditer(
            rf"^\s*{label}\s+(?P<value>\S+)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    if not matches:
        raise ValueError(f"RISM self-test is missing required field {name}.")
    if len(matches) != 1:
        raise ValueError(f"RISM self-test contains duplicate field {name}.")
    return _finite_float(matches[0].group("value"), name=name)


def _required_sections(text: str) -> dict[str, str]:
    """Return exact unit-bearing self-test sections in their required order."""

    expected_counts = {
        _NET_CHARGE_HEADER: 1,
        _FREE_ENERGY_HEADER: 1,
        _RELATIVE_DIFFERENCE_HEADER: 2,
        _PRESSURE_HEADER: 1,
    }
    matches_by_header: dict[str, tuple[re.Match[str], ...]] = {}
    for header, expected_count in expected_counts.items():
        matches = tuple(
            re.finditer(
                rf"^{re.escape(header)}\s*$",
                text,
                flags=re.MULTILINE,
            )
        )
        if len(matches) != expected_count:
            qualifier = "missing" if not matches else "wrong-count"
            raise ValueError(
                f"RISM self-test has {qualifier} required section {header!r}."
            )
        matches_by_header[header] = matches
    net = matches_by_header[_NET_CHARGE_HEADER][0]
    free_energy = matches_by_header[_FREE_ENERGY_HEADER][0]
    relative_energy, relative_pressure = matches_by_header[_RELATIVE_DIFFERENCE_HEADER]
    pressure = matches_by_header[_PRESSURE_HEADER][0]
    ordered = (net, free_energy, relative_energy, pressure, relative_pressure)
    if any(
        ordered[index].start() >= ordered[index + 1].start()
        for index in range(len(ordered) - 1)
    ):
        raise ValueError("RISM self-test required sections are out of order.")
    return {
        _NET_CHARGE_HEADER: text[net.end() : free_energy.start()],
        _FREE_ENERGY_HEADER: text[free_energy.end() : relative_energy.start()],
        _RELATIVE_DIFFERENCE_HEADER: text[relative_energy.end() : pressure.start()],
        _PRESSURE_HEADER: text[pressure.end() : relative_pressure.start()],
    }


@dataclass(frozen=True)
class Route2V0RismSelfTest:
    """Source thermodynamic identities from one ``rism1d`` self-test."""

    input_net_charge_sqrt_kt_angstrom: float
    excess_charge_sum_sqrt_kt_angstrom: float
    total_excess_free_energy_kt_per_angstrom3: float
    sm_identity_relative_residual: float
    pressure_free_energy_kt_per_angstrom3: float
    pressure_virial_kt_per_angstrom3: float
    construction: str = V0_RISM_SELF_TEST_CONSTRUCTION

    def __post_init__(self) -> None:
        values = {
            "input_net_charge_sqrt_kt_angstrom": _finite_float(
                self.input_net_charge_sqrt_kt_angstrom,
                name="input net charge",
            ),
            "excess_charge_sum_sqrt_kt_angstrom": _finite_float(
                self.excess_charge_sum_sqrt_kt_angstrom,
                name="excess charge sum",
            ),
            "total_excess_free_energy_kt_per_angstrom3": _finite_float(
                self.total_excess_free_energy_kt_per_angstrom3,
                name="total excess free-energy density",
            ),
            "sm_identity_relative_residual": _finite_float(
                self.sm_identity_relative_residual,
                name="SM thermodynamic-identity residual",
            ),
            "pressure_free_energy_kt_per_angstrom3": _finite_float(
                self.pressure_free_energy_kt_per_angstrom3,
                name="free-energy-route pressure",
            ),
            "pressure_virial_kt_per_angstrom3": _finite_float(
                self.pressure_virial_kt_per_angstrom3,
                name="virial-route pressure",
            ),
        }
        if self.construction != V0_RISM_SELF_TEST_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 RISM self-test construction.")
        if (
            abs(values["input_net_charge_sqrt_kt_angstrom"])
            > MAX_INPUT_NET_CHARGE_SQRT_KT_ANGSTROM
        ):
            raise ValueError("RISM self-test input charge-neutrality identity failed.")
        if (
            abs(values["excess_charge_sum_sqrt_kt_angstrom"])
            > MAX_EXCESS_CHARGE_SUM_SQRT_KT_ANGSTROM
        ):
            raise ValueError("RISM self-test excess-charge neutrality identity failed.")
        if (
            abs(values["sm_identity_relative_residual"])
            > MAX_SM_IDENTITY_RELATIVE_RESIDUAL
        ):
            raise ValueError("RISM self-test SM free-energy identity failed.")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def parse_route2_v0_rism_self_test(text: str) -> Route2V0RismSelfTest:
    """Parse and validate one AMBER ``rism1d`` self-test report."""

    if not isinstance(text, str):
        raise TypeError("RISM self-test content must be text.")
    sections = _required_sections(text)
    protected_fields = (
        (r"ZERO\s+Input\s+from\s+MDL", "input net charge"),
        (r"ZERO\s+Sum\s+of\s+excess\s+charges", "excess charge sum"),
        (r"Free\s+energy", "total excess free-energy density"),
        (
            r"ZERO\s+ExChem\s+\(SM\)\s+-\s+ExP\s+to\s+Free\s+Energy",
            "SM thermodynamic-identity residual",
        ),
        (r"Pressure\s+\(free\s+energy\)", "free-energy-route pressure"),
        (r"Pressure\s+\(virial\)", "virial-route pressure"),
    )
    for label, name in protected_fields:
        _ = _required_line_value(text, label=label, name=name)
    return Route2V0RismSelfTest(
        input_net_charge_sqrt_kt_angstrom=_required_line_value(
            sections[_NET_CHARGE_HEADER],
            label=r"ZERO\s+Input\s+from\s+MDL",
            name="input net charge",
        ),
        excess_charge_sum_sqrt_kt_angstrom=_required_line_value(
            sections[_NET_CHARGE_HEADER],
            label=r"ZERO\s+Sum\s+of\s+excess\s+charges",
            name="excess charge sum",
        ),
        total_excess_free_energy_kt_per_angstrom3=_required_line_value(
            sections[_FREE_ENERGY_HEADER],
            label=r"Free\s+energy",
            name="total excess free-energy density",
        ),
        sm_identity_relative_residual=_required_line_value(
            sections[_RELATIVE_DIFFERENCE_HEADER],
            label=r"ZERO\s+ExChem\s+\(SM\)\s+-\s+ExP\s+to\s+Free\s+Energy",
            name="SM thermodynamic-identity residual",
        ),
        pressure_free_energy_kt_per_angstrom3=_required_line_value(
            sections[_PRESSURE_HEADER],
            label=r"Pressure\s+\(free\s+energy\)",
            name="free-energy-route pressure",
        ),
        pressure_virial_kt_per_angstrom3=_required_line_value(
            sections[_PRESSURE_HEADER],
            label=r"Pressure\s+\(virial\)",
            name="virial-route pressure",
        ),
    )


def load_route2_v0_rism_self_test(path: str | Path) -> Route2V0RismSelfTest:
    """Load and validate one AMBER ``rism1d`` self-test report."""

    return parse_route2_v0_rism_self_test(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "MAX_EXCESS_CHARGE_SUM_SQRT_KT_ANGSTROM",
    "MAX_INPUT_NET_CHARGE_SQRT_KT_ANGSTROM",
    "MAX_SM_IDENTITY_RELATIVE_RESIDUAL",
    "V0_RISM_SELF_TEST_CONSTRUCTION",
    "Route2V0RismSelfTest",
    "load_route2_v0_rism_self_test",
    "parse_route2_v0_rism_self_test",
]
