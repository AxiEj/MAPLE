from __future__ import annotations

import importlib

import numpy as np
import pytest


def _module():
    return importlib.import_module("maple.function.free_energy.nonequilibrium")


def _instantaneous_work_replicates(
    *,
    displacement: float,
    target_offset_reduced: float,
    sample_count_per_replicate: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(2026072507)
    forward = []
    reverse = []
    for _ in range(2):
        reference_coordinate = rng.normal(0.0, 1.0, sample_count_per_replicate)
        target_coordinate = rng.normal(
            displacement,
            1.0,
            sample_count_per_replicate,
        )
        reference_on_reference = 0.5 * reference_coordinate**2
        target_on_reference = (
            0.5 * (reference_coordinate - displacement) ** 2 + target_offset_reduced
        )
        reference_on_target = 0.5 * target_coordinate**2
        target_on_target = (
            0.5 * (target_coordinate - displacement) ** 2 + target_offset_reduced
        )
        forward.append(target_on_reference - reference_on_reference)
        reverse.append(reference_on_target - target_on_target)
    return forward, reverse


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_nonequilibrium_analysis_recovers_offset_and_reports_work_diagnostics():
    module = _module()
    forward, reverse = _instantaneous_work_replicates(
        displacement=1.0,
        target_offset_reduced=0.7,
        sample_count_per_replicate=800,
    )

    result = module.analyze_nonequilibrium_switching(
        forward,
        reverse,
        work_unit="reduced",
        temperature_kelvin=298.15,
        standard_state="synthetic reference-to-target correction",
        endpoint_equilibrium_claim=True,
        independent_work_values_claim=True,
        minimum_work_values_per_direction=100,
        minimum_independent_replicates_per_direction=2,
        minimum_bar_overlap=0.03,
        maximum_forward_reverse_disagreement_kcal_mol=0.20,
        maximum_bar_uncertainty_kcal_mol=0.10,
        maximum_replicate_exp_range_kcal_mol=0.20,
        maximum_leave_one_replicate_out_deviation_kcal_mol=0.10,
    )

    expected = 0.7 * result["kbt_kcal_mol"]
    assert result["bar"]["delta_g_kcal_mol"] == pytest.approx(expected, abs=0.05)
    assert result["bar"]["overlap"] > 0.03
    assert result["directional"]["forward_exp"]["delta_g_kcal_mol"] == pytest.approx(
        expected,
        abs=0.08,
    )
    assert result["directional"]["reverse_exp"]["delta_g_kcal_mol"] == pytest.approx(
        expected,
        abs=0.08,
    )
    assert result["dissipation"]["forward_mean_kcal_mol"] >= -0.10
    assert result["dissipation"]["reverse_mean_kcal_mol"] >= -0.10
    assert result["gates"]["numerical_gates_passed"] is True
    assert result["gates"]["scientific_gates_passed"] is True
    assert result["estimator"]["handwritten_estimator"] is False
    assert (
        result["standard_state_declaration"]
        == "synthetic reference-to-target correction"
    )
    assert result["standard_state_conversion"] == {
        "applied": False,
        "correction_kcal_mol": 0.0,
        "responsibility": "caller-supplied work values or downstream cycle",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_nonequilibrium_analysis_is_covariant_to_target_energy_alignment():
    module = _module()
    forward, reverse = _instantaneous_work_replicates(
        displacement=0.7,
        target_offset_reduced=0.2,
        sample_count_per_replicate=500,
    )
    common = {
        "work_unit": "reduced",
        "temperature_kelvin": 298.15,
        "standard_state": "synthetic aligned correction",
        "endpoint_equilibrium_claim": True,
        "independent_work_values_claim": True,
        "minimum_work_values_per_direction": 100,
        "minimum_independent_replicates_per_direction": 2,
        "minimum_bar_overlap": 0.03,
        "maximum_forward_reverse_disagreement_kcal_mol": 0.30,
        "maximum_bar_uncertainty_kcal_mol": 0.10,
        "maximum_replicate_exp_range_kcal_mol": 0.30,
        "maximum_leave_one_replicate_out_deviation_kcal_mol": 0.15,
    }
    baseline = module.analyze_nonequilibrium_switching(
        forward,
        reverse,
        **common,
    )

    reduced_shift = 12.5
    shifted = module.analyze_nonequilibrium_switching(
        [values + reduced_shift for values in forward],
        [values - reduced_shift for values in reverse],
        **common,
    )

    expected_shift = reduced_shift * baseline["kbt_kcal_mol"]
    assert shifted["bar"]["delta_g_kcal_mol"] == pytest.approx(
        baseline["bar"]["delta_g_kcal_mol"] + expected_shift,
        abs=1.0e-10,
    )
    assert shifted["bar"]["overlap"] == pytest.approx(
        baseline["bar"]["overlap"],
        abs=1.0e-12,
    )
    assert shifted["dissipation"]["forward_mean_kcal_mol"] == pytest.approx(
        baseline["dissipation"]["forward_mean_kcal_mol"],
        abs=1.0e-10,
    )
    assert shifted["dissipation"]["reverse_mean_kcal_mol"] == pytest.approx(
        baseline["dissipation"]["reverse_mean_kcal_mol"],
        abs=1.0e-10,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_nonequilibrium_analysis_fails_closed_on_poor_overlap():
    module = _module()
    forward, reverse = _instantaneous_work_replicates(
        displacement=5.0,
        target_offset_reduced=0.0,
        sample_count_per_replicate=500,
    )

    result = module.analyze_nonequilibrium_switching(
        forward,
        reverse,
        work_unit="reduced",
        temperature_kelvin=298.15,
        standard_state="synthetic poor-overlap correction",
        endpoint_equilibrium_claim=True,
        independent_work_values_claim=True,
        minimum_work_values_per_direction=100,
        minimum_independent_replicates_per_direction=2,
        minimum_bar_overlap=0.03,
        maximum_forward_reverse_disagreement_kcal_mol=0.20,
        maximum_bar_uncertainty_kcal_mol=0.10,
        maximum_replicate_exp_range_kcal_mol=0.20,
        maximum_leave_one_replicate_out_deviation_kcal_mol=0.10,
    )

    checks = result["gates"]["checks"]
    assert (
        checks["minimum_bar_overlap"] is False
        or checks["maximum_forward_reverse_disagreement"] is False
        or checks["maximum_bar_uncertainty"] is False
    )
    assert result["gates"]["numerical_gates_passed"] is False
    assert result["gates"]["scientific_gates_passed"] is False


def test_nonequilibrium_analysis_validates_contract_before_loading_pymbar():
    module = _module()
    work = [np.zeros(4), np.zeros(4)]

    with pytest.raises(ValueError, match="work_unit"):
        module.analyze_nonequilibrium_switching(
            work,
            work,
            work_unit="hartree",
            temperature_kelvin=298.15,
            standard_state="synthetic",
            endpoint_equilibrium_claim=False,
            independent_work_values_claim=False,
        )

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        module.analyze_nonequilibrium_switching(
            work,
            work,
            work_unit="kcal/mol",
            temperature_kelvin=298.15,
            standard_state="synthetic",
            endpoint_equilibrium_claim=False,
            independent_work_values_claim=False,
            minimum_bar_overlap=1.0,
        )

    with pytest.raises(ValueError, match="finite one-dimensional"):
        module.analyze_nonequilibrium_switching(
            [np.zeros((4, 1))],
            work,
            work_unit="kcal/mol",
            temperature_kelvin=298.15,
            standard_state="synthetic",
            endpoint_equilibrium_claim=False,
            independent_work_values_claim=False,
        )
