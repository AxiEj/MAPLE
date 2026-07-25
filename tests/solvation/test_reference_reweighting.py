from __future__ import annotations

import importlib

import numpy as np
import pytest


def _module():
    return importlib.import_module("maple.function.free_energy.reweighting")


def _reduced_potentials(
    coordinate: np.ndarray,
    *,
    displacement: float,
    target_offset_reduced: float,
) -> np.ndarray:
    reference = 0.5 * coordinate**2
    target = 0.5 * (coordinate - displacement) ** 2 + target_offset_reduced
    return np.column_stack((reference, target))


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_bidirectional_reweighting_recovers_target_offset():
    module = _module()
    rng = np.random.default_rng(20260725)
    sample_count = 1200
    displacement = 1.0
    target_offset_reduced = 0.7
    reference_coordinate = rng.normal(0.0, 1.0, sample_count)
    target_coordinate = rng.normal(displacement, 1.0, sample_count)

    result = module.analyze_bidirectional_reweighting(
        [
            [
                _reduced_potentials(
                    reference_coordinate,
                    displacement=displacement,
                    target_offset_reduced=target_offset_reduced,
                )
            ],
            [
                _reduced_potentials(
                    target_coordinate,
                    displacement=displacement,
                    target_offset_reduced=target_offset_reduced,
                )
            ],
        ],
        temperature_kelvin=298.15,
        standard_state="synthetic reference-to-target correction",
        equilibrium_claim=True,
        detect_equilibration=False,
        minimum_uncorrelated_samples_per_state=50,
        minimum_effective_samples_per_state=50,
        minimum_bar_overlap=0.03,
        minimum_directional_effective_fraction=0.10,
        maximum_directional_disagreement_kcal_mol=0.20,
        maximum_bar_uncertainty_kcal_mol=0.10,
    )

    expected = target_offset_reduced * result["kbt_kcal_mol"]
    assert result["bar"]["delta_g_kcal_mol"] == pytest.approx(expected, abs=0.05)
    assert result["mbar"]["endpoint_delta_g_kcal_mol"] == pytest.approx(
        result["bar"]["delta_g_kcal_mol"],
        abs=1.0e-10,
    )
    assert result["bar"]["absolute_difference_from_mbar_kcal_mol"] < 1.0e-10
    assert result["bar"]["overlap"] > 0.03
    assert (
        result["directional"]["reference_to_target_exp"]["effective_sample_fraction"]
        > 0.10
    )
    assert (
        result["directional"]["target_to_reference_exp"]["effective_sample_fraction"]
        > 0.10
    )
    assert result["gates"]["checks"]["minimum_uncorrelated_samples_per_state"] is True
    assert result["gates"]["checks"]["minimum_effective_samples_per_state"] is True
    assert result["gates"]["checks"]["minimum_mbar_directional_overlap"] is True
    assert result["gates"]["checks"]["maximum_bar_uncertainty"] is True
    assert result["gates"]["checks"]["mbar_solver_convergence"] is True
    assert result["gates"]["checks"]["bar_solver_convergence"] is True
    assert result["gates"]["statistical_gates_passed"] is True
    assert result["estimator"]["handwritten_estimator"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_bidirectional_reweighting_fails_closed_on_poor_overlap():
    module = _module()
    rng = np.random.default_rng(711)
    sample_count = 800
    displacement = 5.0
    reference_coordinate = rng.normal(0.0, 1.0, sample_count)
    target_coordinate = rng.normal(displacement, 1.0, sample_count)

    result = module.analyze_bidirectional_reweighting(
        [
            [
                _reduced_potentials(
                    reference_coordinate,
                    displacement=displacement,
                    target_offset_reduced=0.0,
                )
            ],
            [
                _reduced_potentials(
                    target_coordinate,
                    displacement=displacement,
                    target_offset_reduced=0.0,
                )
            ],
        ],
        temperature_kelvin=298.15,
        standard_state="synthetic poor-overlap correction",
        equilibrium_claim=True,
        detect_equilibration=False,
        minimum_uncorrelated_samples_per_state=50,
        minimum_effective_samples_per_state=50,
        minimum_bar_overlap=0.03,
        minimum_directional_effective_fraction=0.10,
        maximum_directional_disagreement_kcal_mol=0.20,
        maximum_bar_uncertainty_kcal_mol=0.10,
    )

    checks = result["gates"]["checks"]
    assert (
        checks["minimum_bar_overlap"] is False
        or checks["minimum_directional_effective_fraction"] is False
        or checks["maximum_directional_disagreement"] is False
    )
    assert result["gates"]["statistical_gates_passed"] is False
    assert result["claim_boundary"]["reference_only_production_validated"] is False


def test_bidirectional_reweighting_validates_thresholds_before_pymbar():
    module = _module()
    samples = np.zeros((4, 2))

    with pytest.raises(
        ValueError,
        match="minimum_directional_effective_fraction",
    ):
        module.analyze_bidirectional_reweighting(
            [[samples], [samples]],
            temperature_kelvin=298.15,
            standard_state="synthetic",
            equilibrium_claim=False,
            minimum_directional_effective_fraction=1.1,
        )

    with pytest.raises(ValueError, match="minimum_bar_overlap"):
        module.analyze_bidirectional_reweighting(
            [[samples], [samples]],
            temperature_kelvin=298.15,
            standard_state="synthetic",
            equilibrium_claim=False,
            minimum_bar_overlap=1.0,
        )

    with pytest.raises(
        ValueError,
        match="maximum_bar_mbar_disagreement_kcal_mol",
    ):
        module.analyze_bidirectional_reweighting(
            [[samples], [samples]],
            temperature_kelvin=298.15,
            standard_state="synthetic",
            equilibrium_claim=False,
            maximum_bar_mbar_disagreement_kcal_mol=0.0,
        )
