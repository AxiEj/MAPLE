from __future__ import annotations

import numpy as np
import pytest

from maple.function.dispatcher.frequency.stationary_points import (
    assess_stationary_point,
)


def test_minimum_and_first_order_saddle_contracts_are_mutually_exclusive():
    minimum = np.array([120.0, 750.0, 1600.0])
    saddle = np.array([-120.0, 750.0, 1600.0])

    minimum_assessment = assess_stationary_point(
        minimum,
        target="minimum",
        imaginary_threshold_cm1=50.0,
    )
    saddle_assessment = assess_stationary_point(
        saddle,
        target="transition_state",
        imaginary_threshold_cm1=50.0,
    )

    assert minimum_assessment.target == "minimum"
    assert minimum_assessment.thermochemistry_admitted is True
    assert saddle_assessment.target == "transition_state"
    assert saddle_assessment.robust_imaginary_mode_indices == (0,)
    assert saddle_assessment.imaginary_frequency_cm1 == pytest.approx(-120.0)
    assert saddle_assessment.thermochemistry_admitted is False

    with pytest.raises(ValueError, match="minimum requires all vibrational"):
        assess_stationary_point(
            saddle,
            target="minimum",
            imaginary_threshold_cm1=50.0,
        )
    with pytest.raises(ValueError, match="exactly one robust imaginary"):
        assess_stationary_point(
            minimum,
            target="transition_state",
            imaginary_threshold_cm1=50.0,
        )


def test_first_order_saddle_rejects_higher_order_saddle():
    with pytest.raises(ValueError, match="exactly one robust imaginary"):
        assess_stationary_point(
            np.array([-220.0, -80.0, 1600.0]),
            target="transition_state",
            imaginary_threshold_cm1=50.0,
        )


def test_first_order_saddle_rejects_ambiguous_shallow_negative_mode():
    with pytest.raises(ValueError, match="ambiguous non-positive"):
        assess_stationary_point(
            np.array([-20.0, 750.0, 1600.0]),
            target="transition_state",
            imaginary_threshold_cm1=50.0,
        )


def test_minimum_assessment_records_explicit_negative_mode_reinterpretation():
    assessment = assess_stationary_point(
        np.array([5.0, 750.0, 1600.0]),
        target="minimum",
        imaginary_threshold_cm1=50.0,
        reinterpreted_negative_frequencies_cm1=(-5.0,),
        reinterpretation_threshold_cm1=10.0,
    )

    assert assessment.reinterpreted_negative_frequencies_cm1 == (-5.0,)
    assert assessment.reinterpretation_threshold_cm1 == pytest.approx(10.0)


@pytest.mark.parametrize("target", ["ts", "saddle", "minimum_or_ts", ""])
def test_stationary_point_assessor_rejects_undeclared_targets(target):
    with pytest.raises(ValueError, match="minimum.*transition_state"):
        assess_stationary_point(
            np.array([100.0]),
            target=target,
            imaginary_threshold_cm1=50.0,
        )
