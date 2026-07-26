"""Regression contract: Route 2 restrictions must not disable base MAPLE APIs."""

from __future__ import annotations

import pytest

from maple.function.read.command_control import CommandControl


def _parse(*lines: str) -> dict[str, object]:
    return CommandControl.from_settings(list(lines)).as_dict()


def test_gas_phase_charge_configuration_remains_available_outside_route2():
    """A Route-2-only rejection must not become a branch-global rejection."""

    params = _parse(
        "#model=aimnet2",
        "#sp",
        "#charge(source=mol2)",
    )

    assert params["charge"] == {"source": "mol2"}
    assert "solv" not in params


def test_route2_still_rejects_an_independent_charge_provider():
    with pytest.raises(ValueError, match=r"remove #charge"):
        _parse(
            "#model=macepol-m",
            "#sp",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=smd,experimental=true)",
        )
