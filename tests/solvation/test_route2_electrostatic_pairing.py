from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
    MACE_POLAR_MODEL_FEATURE_FIELD_INDICES,
)


def test_mace_polar_pairing_is_the_single_ordering_contract():
    density = np.asarray(
        [
            [0.20, 1.0, 2.0, 3.0],
            [-0.20, -4.0, 5.0, -6.0],
        ]
    )
    field = np.asarray(
        [
            [7.0, 11.0, 13.0, 17.0],
            [19.0, 23.0, 29.0, 31.0],
        ]
    )

    density_order = MACE_POLAR_L1_PAIRING.field_to_density_order(field)
    np.testing.assert_array_equal(
        density_order,
        field[:, [0, 2, 3, 1]],
    )
    np.testing.assert_array_equal(
        MACE_POLAR_L1_PAIRING.density_to_field_order(density_order),
        field,
    )
    assert MACE_POLAR_L1_PAIRING.pair(density, field) == pytest.approx(
        float(np.vdot(density, field[:, [0, 2, 3, 1]]))
    )
    assert MACE_POLAR_MODEL_FEATURE_FIELD_INDICES == (0, 3, 1, 2)


def test_neutral_density_pairing_is_invariant_to_constant_potential_gauge():
    density = np.asarray(
        [
            [0.35, 0.2, -0.1, 0.4],
            [-0.15, -0.3, 0.5, 0.1],
            [-0.20, 0.7, -0.2, -0.6],
        ]
    )
    field = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [-0.5, 0.6, -0.7, 0.8],
            [0.9, -1.0, 1.1, -1.2],
        ]
    )
    shifted = field.copy()
    shifted[:, 0] += 12.345

    assert np.sum(density[:, 0]) == pytest.approx(0.0)
    assert MACE_POLAR_L1_PAIRING.pair(
        density,
        shifted,
    ) == pytest.approx(
        MACE_POLAR_L1_PAIRING.pair(density, field),
        abs=1.0e-14,
    )


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 3)),
        np.zeros((2, 5)),
        np.asarray([[0.0, 0.0, np.nan, 0.0]]),
    ],
)
def test_pairing_rejects_noncanonical_or_nonfinite_blocks(values):
    with pytest.raises(ValueError, match=r"shape \(n_atoms, 4\)"):
        MACE_POLAR_L1_PAIRING.field_to_density_order(values)
