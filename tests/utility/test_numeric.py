import numpy as np
import pytest

from maple.function.dispatcher.optimization.algorithm import _common
from maple.function.utility import numeric


def test_to_numpy_f64_returns_float64_contiguous_array():
    arr = numeric.to_numpy_f64([[1, 2], [3, 4]])

    assert arr.dtype == np.float64
    assert arr.flags.c_contiguous


def test_vec1d_round_trips_1d_list():
    vec = numeric.vec1d([1, 2, 3], n_expected=3)

    np.testing.assert_array_equal(vec, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_vec1d_raises_on_expected_size_mismatch():
    with pytest.raises(ValueError, match="Expected size 4, got 3"):
        numeric.vec1d([1, 2, 3], n_expected=4)


def test_optimization_common_reexports_numeric_helpers():
    assert _common.to_numpy_f64 is numeric.to_numpy_f64
