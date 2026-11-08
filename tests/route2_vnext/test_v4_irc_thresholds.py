from maple.function.dispatcher.dispatcher import (
    _apply_v4_irc_threshold_precedence,
)
import pytest


def test_explicit_v4_irc_thresholds_override_level_and_record_sources():
    params = {"f_max_th": 0.00085, "f_rms_th": 0.00055}
    receipt = _apply_v4_irc_threshold_precedence(
        params,
        explicit={"f_max_th": 0.002, "f_rms_th": 0.0005},
        requested_level="tight",
        method="hpc",
    )
    assert params["f_max_th"] == 0.002
    assert params["f_rms_th"] == 0.0005
    assert receipt == {
        "policy": "v4_explicit_over_level",
        "requested_level": "tight",
        "endpoint_localization_candidate_cap": 12,
        "f_max_hartree_per_angstrom": 0.002,
        "f_max_source": "explicit",
        "f_rms_hartree_per_angstrom": 0.0005,
        "f_rms_source": "explicit",
    }


def test_level_and_fallback_sources_remain_distinct():
    level = {"f_max_th": 0.00085, "f_rms_th": 0.00055}
    receipt = _apply_v4_irc_threshold_precedence(
        level, explicit={}, requested_level="tight", method="gs"
    )
    assert receipt["f_max_source"] == "level:tight"
    assert receipt["endpoint_localization_candidate_cap"] is None

    fallback = {"f_max_th": 0.00285, "f_rms_th": 0.0019}
    receipt = _apply_v4_irc_threshold_precedence(
        fallback, explicit={}, requested_level="unknown", method="eulerpc"
    )
    assert receipt["f_max_source"] == "fallback:medium"
    assert receipt["f_rms_source"] == "fallback:medium"
    assert receipt["endpoint_localization_candidate_cap"] == 12


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_explicit_thresholds_fail_before_mutating_params(value):
    params = {"f_max_th": 0.00085, "f_rms_th": 0.00055}
    before = dict(params)
    with pytest.raises((TypeError, ValueError)):
        _apply_v4_irc_threshold_precedence(
            params,
            explicit={"f_max_th": value},
            requested_level="tight",
            method="hpc",
        )
    assert params == before


def test_missing_level_is_reported_as_fallback_medium():
    params = {"f_max_th": 0.00285, "f_rms_th": 0.0019}
    receipt = _apply_v4_irc_threshold_precedence(
        params, explicit={}, requested_level=None, method="gs"
    )
    assert receipt["requested_level"] is None
    assert receipt["f_max_source"] == "fallback:medium"
    assert receipt["f_rms_source"] == "fallback:medium"
