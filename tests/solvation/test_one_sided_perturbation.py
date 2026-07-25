from __future__ import annotations

import importlib

import numpy as np
import pytest


def _module():
    return importlib.import_module("maple.function.free_energy.perturbation")


def test_one_sided_perturbation_validates_before_loading_pymbar(monkeypatch):
    module = _module()

    def unexpected_dependency_load():
        raise AssertionError("invalid input must fail before loading PyMBAR")

    monkeypatch.setattr(module, "_load_pymbar", unexpected_dependency_load)
    with pytest.raises(ValueError, match="one-dimensional"):
        module.analyze_one_sided_perturbation(
            [np.zeros((4, 1))],
            temperature_kelvin=298.15,
            reference_equilibrium_claim=False,
        )
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        module.analyze_one_sided_perturbation(
            [np.zeros(4), np.zeros(4)],
            temperature_kelvin=298.15,
            reference_equilibrium_claim=False,
            minimum_effective_sample_fraction=1.1,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_one_sided_perturbation_recovers_a_nearly_constant_offset():
    module = _module()
    values = np.tile([0.99, 1.01], 100)
    result = module.analyze_one_sided_perturbation(
        [values, values.copy()],
        temperature_kelvin=298.15,
        reference_equilibrium_claim=True,
        detect_equilibration=False,
        minimum_uncorrelated_samples_per_replicate=100,
        minimum_effective_sample_count=100.0,
        minimum_effective_sample_fraction=0.90,
        maximum_normalized_weight=0.01,
        maximum_replicate_difference_kcal_mol=0.01,
    )

    beta = 1.0 / result["kbt_kcal_mol"]
    expected = -np.log(np.mean(np.exp(-beta * values))) / beta
    assert result["combined"]["delta_g_kcal_mol"] == pytest.approx(
        expected,
        abs=1.0e-12,
    )
    assert result["combined"]["effective_sample_fraction"] > 0.99
    assert result["replicate_estimate_range_kcal_mol"] == pytest.approx(0.0)
    assert result["gates"]["numerical_gates_passed"] is True
    assert result["gates"]["scientific_gates_passed"] is True
    assert result["estimator"]["handwritten_estimator"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_one_sided_perturbation_fails_closed_on_weight_collapse():
    module = _module()
    values = np.zeros(100)
    values[0] = -10.0
    result = module.analyze_one_sided_perturbation(
        [values, values.copy()],
        temperature_kelvin=298.15,
        reference_equilibrium_claim=True,
        detect_equilibration=False,
        minimum_uncorrelated_samples_per_replicate=20,
        minimum_effective_sample_count=20.0,
        minimum_effective_sample_fraction=0.10,
        maximum_normalized_weight=0.20,
        maximum_replicate_difference_kcal_mol=0.50,
    )

    checks = result["gates"]["checks"]
    assert checks["minimum_effective_sample_count"] is False
    assert checks["minimum_effective_sample_fraction"] is False
    assert checks["maximum_normalized_weight"] is False
    assert result["gates"]["numerical_gates_passed"] is False
    assert result["gates"]["scientific_gates_passed"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_one_sided_perturbation_separates_numerical_and_equilibrium_claims():
    module = _module()
    values = np.tile([0.19, 0.21], 100)
    result = module.analyze_one_sided_perturbation(
        [values, values.copy()],
        temperature_kelvin=298.15,
        reference_equilibrium_claim=False,
        detect_equilibration=False,
        minimum_uncorrelated_samples_per_replicate=100,
        minimum_effective_sample_count=100.0,
        minimum_effective_sample_fraction=0.90,
        maximum_normalized_weight=0.01,
        maximum_replicate_difference_kcal_mol=0.01,
    )

    assert result["gates"]["checks"]["reference_equilibrium_claim"] is False
    assert result["gates"]["numerical_gates_passed"] is True
    assert result["gates"]["scientific_gates_passed"] is False
