from __future__ import annotations

import numpy as np
import pytest

from tools.route2_release.cds_lstsq import fit


def test_full_rank_exact_recovery_with_signed_coefficients():
    X = np.array([[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0], [4.0, 0.5]])
    expected = np.array([-2.5, 3.25])
    result = fit(X, X @ expected, np.array([7.0, -4.0]))
    np.testing.assert_allclose(result["theta"], expected, atol=1e-12)
    assert result["train_sse"] == pytest.approx(0.0, abs=1e-24)
    assert result["rank"] == 2
    assert result["diagnostics"]["unique_supported_solution"]


def test_noisy_fit_matches_direct_ordinary_least_squares():
    X = np.array([[1.0, 2.0], [2.0, -1.0], [4.0, 0.5], [-2.0, 3.0]])
    y = np.array([1.0, -2.0, 3.5, 0.25])
    prior = np.array([0.3, -0.2])
    result = fit(X, y, prior)
    expected = np.linalg.lstsq(X, y, rcond=max(X.shape) * np.finfo(float).eps)[0]
    np.testing.assert_allclose(result["theta"], expected, rtol=1e-12, atol=1e-12)
    expected_sse = np.sum(np.square(y - X @ expected))
    assert result["train_sse"] == pytest.approx(expected_sse)


def test_solver_does_not_add_an_intercept():
    X = np.array([[1.0], [2.0], [3.0]])
    y = 2.0 * X[:, 0] + 5.0
    result = fit(X, y, np.zeros(1))
    assert result["train_sse"] > 0.0
    assert not np.allclose(X @ result["theta"], y)


def test_zero_coverage_column_is_held_bitwise_at_prior():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    prior = np.array([0.0, np.float64(-7.125)])
    result = fit(X, 2.0 * X[:, 0], prior)
    assert result["column_coverage"].tolist() == [3, 0]
    assert result["supported_mask"].tolist() == [True, False]
    assert result["theta"][1].tobytes() == prior[1].tobytes()
    assert result["coefficient_correction"][1] == 0.0
    assert result["diagnostics"]["n_supported"] == 1


def test_collinear_design_reports_non_unique_minimum_scaled_change():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    prior = np.array([1.0, -1.0])
    y = np.array([4.0, 8.0, 12.0])
    result = fit(X, y, prior)
    diagnostics = result["diagnostics"]
    assert result["rank"] == 1
    assert diagnostics["nullity"] == 1
    assert not diagnostics["unique_supported_solution"]
    assert "not uniquely identified" in diagnostics["representative"]
    assert diagnostics["null_space_basis_scaled"].shape == (2, 1)
    assert diagnostics["null_space_basis_coefficient"].shape == (2, 1)
    null_direction = diagnostics["null_space_basis_coefficient"][:, 0]
    np.testing.assert_allclose(X @ null_direction, 0.0, atol=1e-12)
    np.testing.assert_allclose(X @ result["theta"], y, atol=1e-12)
    # The selected representative minimizes ||scales * (theta - prior)||_2.
    scaled_change = result["scales"] * (result["theta"] - prior)
    scaled_null = result["scales"] * null_direction
    assert scaled_change @ scaled_null == pytest.approx(0.0, abs=1e-12)
    assert np.isinf(diagnostics["condition_number"])
    assert diagnostics["rcond_cutoff"] > 0.0


def test_wide_design_reports_complete_null_space():
    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    result = fit(X, np.array([2.0, -3.0]), np.zeros(3))
    diagnostics = result["diagnostics"]
    assert result["rank"] == 2
    assert diagnostics["nullity"] == 1
    assert diagnostics["null_space_basis_scaled"].shape == (3, 1)
    assert diagnostics["null_space_basis_coefficient"].shape == (3, 1)
    np.testing.assert_allclose(
        X @ diagnostics["null_space_basis_coefficient"], 0.0, atol=1e-12
    )


def test_all_zero_design_keeps_prior_and_reports_no_supported_columns():
    X = np.zeros((3, 2))
    y = np.array([1.0, -2.0, 4.0])
    prior = np.array([3.25, -7.5])
    result = fit(X, y, prior)
    np.testing.assert_array_equal(result["theta"], prior)
    np.testing.assert_array_equal(result["column_coverage"], [0, 0])
    assert result["rank"] == 0
    assert result["singular_values"].size == 0
    assert result["diagnostics"]["n_supported"] == 0
    assert result["diagnostics"]["nullity"] == 0
    assert result["train_sse"] == pytest.approx(float(y @ y))


def test_overflowed_rms_scale_is_rejected():
    X = np.array([[np.finfo(np.float64).max], [np.finfo(np.float64).max]])
    with pytest.raises(ValueError, match="RMS scales"):
        fit(X, np.ones(2), np.zeros(1))


def test_inputs_are_not_mutated():
    X = np.array([[1.0, -2.0], [3.0, 4.0], [-1.0, 0.5]])
    y = np.array([2.0, -1.0, 3.0])
    prior = np.array([0.25, -0.75])
    originals = (X.copy(), y.copy(), prior.copy())
    fit(X, y, prior)
    for actual, expected in zip((X, y, prior), originals):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "X,y,prior",
    [
        (np.ones(3), np.ones(3), np.ones(1)),
        (np.ones((2, 1)), np.ones(3), np.ones(1)),
        (np.ones((2, 1)), np.ones(2), np.ones(2)),
        (np.empty((0, 1)), np.empty(0), np.ones(1)),
        (np.array([[np.nan]]), np.ones(1), np.ones(1)),
        (np.ones((1, 1)), np.array([np.inf]), np.ones(1)),
        (np.ones((1, 1)), np.ones(1), np.array([np.inf])),
    ],
)
def test_invalid_shapes_and_nonfinite_values_raise(X, y, prior):
    with pytest.raises(ValueError):
        fit(X, y, prior)
