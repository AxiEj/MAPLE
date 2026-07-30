"""Executable contracts for the frozen FeNNix HFE protocol and estimators.

These tests cover estimator mathematics and fail-closed metadata only. They do
not constitute HFE sampling, experimental accuracy, GPU admission, or timing.
"""

# pyright: reportArgumentType=false

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import json
import math

import numpy as np
import pytest

from maple.function.solvfe.fennix_hfe.estimators import (
    EV_TO_KCAL_PER_MOL,
    EstimatorProvenance,
    bidirectional_reweighting_diagnostics,
    evaluate_half_trajectory_gate,
    evaluate_interwalker_gate,
    first_second_half_diagnostic,
    fixed_window_ti_sum,
    integrate_native_lambda_leg,
    integrated_autocorrelation_time,
    interwalker_diagnostic,
    moving_block_bootstrap_mean,
    sha256_numeric_inputs,
    solve_adjacent_bar,
)
from maple.function.solvfe.fennix_hfe.protocol import (
    FeNNixHFEProtocol,
    FeNNixProtocolProvenance,
)
from maple.function.solvfe.fennix_hfe.types import (
    FENNIX_PACKAGE_TREE_FILE_COUNT,
    FENNIX_PACKAGE_TREE_SHA256,
    FeNNixKernelIdentity,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _identity(*, reverse_mapping_order: bool = False) -> FeNNixKernelIdentity:
    source_items = [("preprocessing.py", _SHA_A), ("fennix.py", _SHA_B)]
    package_items = [("jax", "0.10.2"), ("jaxlib", "0.10.2")]
    if reverse_mapping_order:
        source_items.reverse()
        package_items.reverse()
    return FeNNixKernelIdentity(
        checkpoint_sha256=_SHA_A,
        source_revision="d62b8740343b803a2b864140ec79e347f8ba034e",
        checkpoint_source_revision="83f299b81c1d62e2a15c892280559a7c0cc2fac3",
        runtime_distribution="FeNNol",
        runtime_version="2026.6.29",
        runtime_source_sha256=dict(source_items),
        runtime_package_versions=dict(package_items),
        fennol_package_tree_sha256=FENNIX_PACKAGE_TREE_SHA256,
        fennol_package_tree_file_count=FENNIX_PACKAGE_TREE_FILE_COUNT,
        original_parameter_tree_fingerprint=_SHA_A,
        derived_parameter_tree_fingerprint=_SHA_B,
        fixed_species_encoding_float64_sha256=_SHA_A,
        jax_enable_x64=True,
        matmul_precision="highest",
        tf32_enabled=False,
    )


@pytest.fixture
def provenance() -> EstimatorProvenance:
    return EstimatorProvenance(_SHA_A, _SHA_B, "synthetic analytic test fixture")


@pytest.fixture
def protocol() -> FeNNixHFEProtocol:
    return FeNNixHFEProtocol(_identity())


def test_protocol_defaults_define_21_unique_two_leg_native_states(protocol):
    states = protocol.native_lambda_states

    assert len(states) == 21
    assert len(set(states)) == 21
    assert states[0] == (0.0, 0.0)
    assert states[10] == protocol.shared_native_lambda_state == (0.0, 1.0)
    assert states[-1] == (1.0, 1.0)


def test_protocol_is_immutable(protocol):
    with pytest.raises(FrozenInstanceError):
        protocol.temperature_kelvin = 300.0  # type: ignore[misc]


def test_protocol_provenance_mapping_is_immutable():
    with pytest.raises(TypeError):
        FeNNixHFEProtocol.FIELD_PROVENANCE["temperature_kelvin"] = (  # type: ignore[index]
            FeNNixProtocolProvenance.PAPER_EXPLICIT
        )


def test_protocol_canonical_hash_is_independent_of_mapping_insertion_order():
    left = FeNNixHFEProtocol(_identity())
    right = FeNNixHFEProtocol(_identity(reverse_mapping_order=True))

    assert left.canonical_json() == right.canonical_json()
    assert left.canonical_sha256() == right.canonical_sha256()
    assert len(left.canonical_sha256()) == 64


def test_protocol_canonical_payload_records_provenance_for_every_field(protocol):
    payload = protocol.canonical_payload()
    recorded = payload["fields"]

    assert payload["schema"] == "maple.fennix_hfe_protocol.v1"
    assert set(recorded) == {item.name for item in fields(protocol)}
    assert all(
        item["provenance"] in {member.value for member in FeNNixProtocolProvenance}
        for item in recorded.values()
    )
    assert recorded["timestep_femtoseconds"]["provenance"] == "paper_explicit"
    assert recorded["temperature_kelvin"]["provenance"] == "maple_reconstruction"


def test_protocol_canonical_json_is_ascii_finite_and_round_trippable(protocol):
    encoded = protocol.canonical_json()

    assert encoded.isascii()
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert json.loads(encoded) == protocol.canonical_payload()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"temperature_kelvin": 300.0}, "temperature_kelvin.*frozen"),
        ({"walker_count": 2}, "walker_count.*frozen"),
        ({"repulsion_leg_lambda_v": (0.0, 1.0)}, "repulsion_leg_lambda_v.*frozen"),
        ({"hfe_admitted": True}, "hfe_admitted.*frozen"),
        ({"accuracy_admitted": True}, "accuracy_admitted.*frozen"),
        ({"gpu_admitted": True}, "gpu_admitted.*frozen"),
        ({"performance_admitted": True}, "performance_admitted.*frozen"),
    ],
)
def test_protocol_rejects_any_frozen_contract_drift(protocol, change, message):
    with pytest.raises(ValueError, match=message):
        replace(protocol, **change)


def test_protocol_rejects_non_kernel_identity():
    with pytest.raises(TypeError, match="FeNNixKernelIdentity"):
        FeNNixHFEProtocol(object())  # type: ignore[arg-type]


def test_protocol_keeps_all_admissions_closed_and_contains_no_result_fields(protocol):
    recorded = protocol.canonical_payload()["fields"]

    assert protocol.hfe_admitted is False
    assert protocol.accuracy_admitted is False
    assert protocol.gpu_admitted is False
    assert protocol.performance_admitted is False
    assert protocol.sensitivity_experimental_selection_forbidden is True
    assert not (
        {"experimental_values", "predictions", "errors", "timing", "speedup"}
        & set(recorded)
    )


def test_simpson_integrates_uniform_cubic_exactly(provenance):
    grid = np.linspace(0.0, 1.0, 5)
    result = integrate_native_lambda_leg(
        grid, grid**3, energy_unit="eV", provenance=provenance
    )

    assert result.simpson == pytest.approx(0.25, abs=1.0e-15)


def test_trapezoid_is_preserved_as_an_explicit_discretization_diagnostic(provenance):
    grid = np.linspace(0.0, 1.0, 5)
    result = integrate_native_lambda_leg(
        grid, grid**3, energy_unit="eV", provenance=provenance
    )

    assert result.trapezoid == pytest.approx(0.265625, abs=1.0e-15)
    assert result.absolute_discrepancy == pytest.approx(0.015625, abs=1.0e-15)


def test_simpson_integrates_nonuniform_quadratic_exactly(provenance):
    grid = np.array([0.0, 0.13, 0.51, 0.77, 1.0])
    derivative = 3.0 * grid**2 + 2.0 * grid + 4.0
    result = integrate_native_lambda_leg(
        grid, derivative, energy_unit="kJ/mol", provenance=provenance
    )

    assert result.simpson == pytest.approx(6.0, abs=1.0e-14)


def test_fixed_window_ti_applies_sign_and_ev_to_kcal_conversion(provenance):
    first = integrate_native_lambda_leg(
        [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], energy_unit="eV", provenance=provenance
    )
    second = integrate_native_lambda_leg(
        [0.0, 0.5, 1.0], [2.0, 2.0, 2.0], energy_unit="eV", provenance=provenance
    )

    result = fixed_window_ti_sum(
        [first, second], sign=-1, output_energy_unit="kcal/mol", provenance=provenance
    )

    assert result.sign == -1
    assert result.simpson == pytest.approx(-3.0 * EV_TO_KCAL_PER_MOL)
    assert result.energy_unit == "kcal/mol"


def test_bar_recovers_analytic_constant_work_free_energy(provenance):
    result = solve_adjacent_bar(
        np.full(16, 2.75), np.full(16, -2.75), provenance=provenance
    )

    assert result.converged is True
    assert result.delta_f == pytest.approx(2.75, abs=1.0e-11)
    assert abs(result.residual) <= 1.0e-10


def test_bar_is_antisymmetric_when_directions_are_swapped(provenance):
    forward = np.array([-1.5, -0.2, 0.4, 1.9])
    reverse = np.array([-1.1, 0.1, 0.8, 2.2])

    ab = solve_adjacent_bar(forward, reverse, provenance=provenance)
    ba = solve_adjacent_bar(reverse, forward, provenance=provenance)

    assert ab.delta_f == pytest.approx(-ba.delta_f, abs=1.0e-11)


def test_bar_remains_stable_for_large_finite_log_works(provenance):
    result = solve_adjacent_bar(
        np.full(8, 1000.0), np.full(8, -1000.0), provenance=provenance
    )

    assert result.converged is True
    assert result.delta_f == pytest.approx(1000.0, abs=1.0e-9)
    assert math.isfinite(result.residual)


def test_bar_rejects_disconnected_no_overlap_work_distributions(provenance):
    with pytest.raises(ValueError, match="overlap|disconnected"):
        solve_adjacent_bar(
            np.full(16, 1000.0), np.full(16, 1000.0), provenance=provenance
        )


def test_reweighting_diagnostics_report_perfect_overlap_and_full_ess(provenance):
    result = bidirectional_reweighting_diagnostics(
        np.zeros(10), np.zeros(10), 0.0, provenance=provenance
    )

    assert result.overlap == pytest.approx(1.0)
    assert result.forward_effective_sample_size == pytest.approx(10.0)
    assert result.reverse_effective_sample_size == pytest.approx(10.0)
    assert result.forward_ess_fraction == pytest.approx(1.0)
    assert result.reverse_ess_fraction == pytest.approx(1.0)


def test_reweighting_ess_detects_weight_collapse(provenance):
    result = bidirectional_reweighting_diagnostics(
        [0.0, 100.0, 100.0, 100.0],
        [0.0, 100.0, 100.0, 100.0],
        0.0,
        provenance=provenance,
    )

    assert result.forward_effective_sample_size == pytest.approx(1.0)
    assert result.reverse_effective_sample_size == pytest.approx(1.0)
    assert result.forward_ess_fraction == pytest.approx(0.25)


def test_autocorrelation_constant_trace_has_tau_one_and_full_ess(provenance):
    result = integrated_autocorrelation_time(np.ones(32), provenance=provenance)

    assert result.integrated_autocorrelation_time == 1.0
    assert result.effective_sample_size == 32.0
    assert result.maximum_lag_used == 0


def test_autocorrelation_detects_positive_serial_correlation(provenance):
    values = np.repeat(np.arange(8, dtype=float), 8)
    result = integrated_autocorrelation_time(values, provenance=provenance)

    assert result.integrated_autocorrelation_time > 1.0
    assert result.effective_sample_size < result.sample_count
    assert result.maximum_lag_used > 0


def test_moving_block_bootstrap_is_bitwise_deterministic_for_fixed_seed(provenance):
    kwargs = dict(
        block_length=3,
        resample_count=64,
        seed=2026073199,
        confidence_level=0.95,
        unit="eV",
        provenance=provenance,
    )

    first = moving_block_bootstrap_mean(np.arange(10.0), **kwargs)
    second = moving_block_bootstrap_mean(np.arange(10.0), **kwargs)

    assert first.bootstrap_means == second.bootstrap_means
    assert first.standard_error == second.standard_error
    assert first.confidence_interval == second.confidence_interval


def test_moving_block_bootstrap_changes_resamples_when_seed_changes(provenance):
    common = dict(
        block_length=3,
        resample_count=32,
        confidence_level=0.90,
        unit="eV",
        provenance=provenance,
    )

    first = moving_block_bootstrap_mean(np.arange(10.0), seed=7, **common)
    second = moving_block_bootstrap_mean(np.arange(10.0), seed=8, **common)

    assert first.bootstrap_means != second.bootstrap_means


def test_half_trajectory_diagnostic_exposes_shift_that_fails_protocol_gate(
    protocol, provenance
):
    result = first_second_half_diagnostic(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], unit="eV", provenance=provenance
    )
    threshold_ev = protocol.full_vs_final_half_ti_abs_kcal_per_mol / EV_TO_KCAL_PER_MOL
    gate = evaluate_half_trajectory_gate(
        result, maximum_absolute_mean_shift=threshold_ev
    )

    assert result.signed_mean_shift == 1.0
    assert gate.maximum_absolute_mean_shift == pytest.approx(threshold_ev)
    assert gate.passed is False


def test_interwalker_diagnostic_exposes_span_that_fails_protocol_gate(
    protocol, provenance
):
    result = interwalker_diagnostic(
        [[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]], unit="kcal/mol", provenance=provenance
    )
    gate = evaluate_interwalker_gate(
        result,
        maximum_walker_mean_standard_deviation=0.25,
        maximum_absolute_walker_mean_deviation=(
            protocol.walker_max_span_kcal_per_mol / 2.0
        ),
    )

    assert result.walker_means == pytest.approx((0.0, 0.1, 1.0))
    assert gate.passed is False


def test_half_trajectory_gate_accepts_shift_at_inclusive_threshold(provenance):
    diagnostic = first_second_half_diagnostic(
        [0.0, 0.0, 0.25, 0.25], unit="eV", provenance=provenance
    )

    gate = evaluate_half_trajectory_gate(diagnostic, maximum_absolute_mean_shift=0.25)

    assert gate.passed is True


def test_interwalker_gate_requires_both_spread_criteria(provenance):
    diagnostic = interwalker_diagnostic(
        [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]],
        unit="kcal/mol",
        provenance=provenance,
    )

    standard_deviation_failure = evaluate_interwalker_gate(
        diagnostic,
        maximum_walker_mean_standard_deviation=0.19,
        maximum_absolute_walker_mean_deviation=0.21,
    )
    maximum_deviation_failure = evaluate_interwalker_gate(
        diagnostic,
        maximum_walker_mean_standard_deviation=0.21,
        maximum_absolute_walker_mean_deviation=0.19,
    )
    passing = evaluate_interwalker_gate(
        diagnostic,
        maximum_walker_mean_standard_deviation=0.21,
        maximum_absolute_walker_mean_deviation=0.21,
    )

    assert standard_deviation_failure.passed is False
    assert maximum_deviation_failure.passed is False
    assert passing.passed is True


@pytest.mark.parametrize(
    ("function", "args", "kwargs", "message"),
    [
        (
            integrate_native_lambda_leg,
            ([[0.0, 0.5], [1.0, 1.5]], [1.0, 2.0]),
            {"energy_unit": "eV"},
            "one-dimensional",
        ),
        (
            integrate_native_lambda_leg,
            ([0.0, 0.5, 1.0], [1.0, 2.0]),
            {"energy_unit": "eV"},
            "same shape",
        ),
        (
            integrate_native_lambda_leg,
            ([0.0, 0.5, 0.5], [1.0, 2.0, 3.0]),
            {"energy_unit": "eV"},
            "strictly increasing",
        ),
        (
            integrate_native_lambda_leg,
            ([0.0, 1.0], [1.0, np.nan]),
            {"energy_unit": "eV"},
            "finite",
        ),
        (
            integrate_native_lambda_leg,
            ([0.0, 1.0], [1.0, 2.0]),
            {"energy_unit": "hartree"},
            "energy unit",
        ),
        (solve_adjacent_bar, ([0.0, np.inf], [0.0]), {}, "finite"),
        (bidirectional_reweighting_diagnostics, ([0.0], [0.0], np.nan), {}, "delta_f"),
        (integrated_autocorrelation_time, ([0.0],), {}, "at least 2"),
        (
            moving_block_bootstrap_mean,
            ([0.0, 1.0],),
            {
                "block_length": 3,
                "resample_count": 10,
                "seed": 1,
                "confidence_level": 0.95,
                "unit": "eV",
            },
            "block_length",
        ),
        (
            first_second_half_diagnostic,
            ([0.0, 1.0, 2.0],),
            {"unit": "eV"},
            "at least 4",
        ),
        (interwalker_diagnostic, ([0.0, 1.0, 2.0],), {"unit": "eV"}, "shape"),
    ],
)
def test_estimators_reject_malformed_nonfinite_shape_grid_and_unit_inputs(
    function, args, kwargs, message, provenance
):
    with pytest.raises(ValueError, match=message):
        function(*args, provenance=provenance, **kwargs)


def test_estimator_provenance_rejects_invalid_hash_and_empty_label():
    with pytest.raises(ValueError, match="SHA-256"):
        EstimatorProvenance("not-a-hash", _SHA_B, "source")
    with pytest.raises(ValueError, match="nonempty"):
        EstimatorProvenance(_SHA_A, _SHA_B, "  ")


def test_numeric_input_hash_is_dtype_normalized_and_shape_sensitive():
    float32_hash = sha256_numeric_inputs(np.array([1.0, 2.0], dtype=np.float32))
    float64_hash = sha256_numeric_inputs(np.array([1.0, 2.0], dtype=np.float64))
    split_hash = sha256_numeric_inputs([1.0], [2.0])

    assert float32_hash == float64_hash
    assert split_hash != float64_hash
