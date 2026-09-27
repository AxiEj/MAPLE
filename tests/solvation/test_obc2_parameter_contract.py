"""Constructor-level ownership and semantic checks for shared OBC-II inputs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
    OBC2Parameters,
    build_obc2_parameters,
)


def _parameters(**changes):
    provider = np.array([[0.15, 0.8], [0.17, 0.9]])
    offset = provider[:, 0] - 0.009
    values = {
        "charges": np.array([0.4, -0.4]),
        "provider_parameters": provider,
        "offset_radii_nm": offset,
        "scaled_offset_radii_nm": provider[:, 1] * offset,
        "nonpolar": "ace",
        "provenance": {"provider": {"versions": ["8.5.2"]}},
    }
    values.update(changes)
    return OBC2Parameters(**values)


def test_direct_constructor_owns_and_freezes_arrays_and_nested_provenance():
    charges = np.array([0.4, -0.4])
    provider = np.array([[0.15, 0.8], [0.17, 0.9]])
    offset = provider[:, 0] - 0.009
    scaled = provider[:, 1] * offset
    provenance = {"provider": {"versions": ["8.5.2"]}}

    parameters = _parameters(
        charges=charges,
        provider_parameters=provider,
        offset_radii_nm=offset,
        scaled_offset_radii_nm=scaled,
        provenance=provenance,
    )
    charges[0] = 9.0
    provider[0, 0] = 9.0
    offset[0] = 9.0
    scaled[0] = 9.0
    provenance["provider"]["versions"].append("changed")

    assert parameters.charges.tolist() == [0.4, -0.4]
    assert parameters.provider_parameters[0].tolist() == [0.15, 0.8]
    assert parameters.offset_radii_nm[0] == pytest.approx(0.141)
    assert parameters.scaled_offset_radii_nm[0] == pytest.approx(0.1128)
    assert parameters.provenance["provider"]["versions"] == ("8.5.2",)
    for values in (
        parameters.charges,
        parameters.provider_parameters,
        parameters.offset_radii_nm,
        parameters.scaled_offset_radii_nm,
    ):
        assert values.flags.writeable is False
    with pytest.raises(TypeError):
        parameters.provenance["new"] = "value"


@pytest.mark.parametrize(
    "change",
    [
        {"provider_parameters": [[0.15, 0.8]]},
        {"offset_radii_nm": [0.141]},
        {"scaled_offset_radii_nm": [0.1128]},
        {"provider_parameters": [[0.009, 0.8], [0.17, 0.9]]},
        {"provider_parameters": [[0.15, 0.0], [0.17, 0.9]]},
        {"offset_radii_nm": [0.14, 0.161]},
        {"scaled_offset_radii_nm": [0.1, 0.1449]},
        {"charges": [float("nan"), 0.0]},
        {"nonpolar": "lcpo"},
        {"nonpolar": None},
        {"solvent_dielectric": 0.0},
        {"solvent_dielectric": 80.0},
        {"solute_dielectric": 2.0},
        {"solute_dielectric": float("inf")},
        {"provenance": "not-a-mapping"},
    ],
)
def test_direct_constructor_rejects_invalid_finalized_parameters(change):
    with pytest.raises((TypeError, ValueError)):
        _parameters(**change)


def test_dataclass_replace_cannot_bypass_finalized_parameter_validation():
    parameters = _parameters()
    with pytest.raises(ValueError, match="offset"):
        replace(parameters, offset_radii_nm=np.array([0.14, 0.161]))


def test_builder_rejects_non_string_nonpolar_without_coercion():
    radius_result = SimpleNamespace(
        provider_parameters=np.array([[0.15, 0.8], [0.17, 0.9]]),
        profile="obc2-mbondi2",
        provenance={"provider": "test"},
    )
    with pytest.raises(TypeError, match="explicit string"):
        build_obc2_parameters(np.array([0.4, -0.4]), radius_result, nonpolar=None)


def test_builder_canonicalizes_nonpolar_field_and_provenance_together():
    radius_result = SimpleNamespace(
        provider_parameters=np.array([[0.15, 0.8], [0.17, 0.9]]),
        profile="obc2-mbondi2",
        provenance={"provider": "test"},
    )
    parameters = build_obc2_parameters(
        np.array([0.4, -0.4]), radius_result, nonpolar="ACE"
    )

    assert parameters.nonpolar == "ace"
    assert parameters.provenance["nonpolar"] == "ace"
