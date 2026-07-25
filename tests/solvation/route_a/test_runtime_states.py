from __future__ import annotations

import pytest

from maple.function.dispatcher.solvfe.states import (
    CalculatorSpec,
    SolvFERequest,
)


SHA = "a" * 64


def test_calculator_spec_is_immutable_and_requires_provenance():
    spec = CalculatorSpec(
        role="target",
        name="maceomol",
        checkpoint="/tmp/model",
        sha256=SHA,
        license_acknowledged=True,
        capabilities=frozenset({"energy", "forces", "charge", "multiplicity"}),
        options={"device": "cpu"},
    )

    assert spec.content_hash
    with pytest.raises(Exception):
        spec.name = "changed"

    with pytest.raises(ValueError, match="license"):
        CalculatorSpec(
            role="target",
            name="maceomol",
            checkpoint="/tmp/model",
            sha256=SHA,
            license_acknowledged=False,
            capabilities=frozenset({"energy"}),
        )


def test_solvfe_request_is_narrow_and_typed():
    request = SolvFERequest.from_params(
        {
            "method": "qct",
            "solvent": "water",
            "temperature": 298.15,
            "pressure_bar": 1.0,
            "standard_state": "1m",
            "protocol": "docs/solvation/route-a/protocol-v1.json",
            "experimental": True,
            "dry_run": True,
        }
    )
    assert request.dry_run is True

    with pytest.raises(ValueError, match="1m"):
        SolvFERequest.from_params(
            {
                **request.as_dict(),
                "standard_state": "1atm",
            }
        )

