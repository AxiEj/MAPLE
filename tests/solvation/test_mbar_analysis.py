import importlib
import logging
import sys

import numpy as np
import pytest


def _module():
    return importlib.import_module("maple.function.free_energy.mbar")


def test_mbar_inputs_fail_closed_before_loading_optional_dependency():
    module = _module()
    state = np.zeros((4, 2))

    with pytest.raises(ValueError, match=r"span \[0, 1\]"):
        module.analyze_mbar(
            [[state], [state]],
            lambda_values=[0.0, 0.5],
            temperature_kelvin=298.15,
            standard_state="gas 1 M to solution 1 M",
            equilibrium_claim=False,
        )

    with pytest.raises(ValueError, match="reduced-potential matrix"):
        module.analyze_mbar(
            [[np.zeros((4, 3))], [state]],
            lambda_values=[0.0, 1.0],
            temperature_kelvin=298.15,
            standard_state="gas 1 M to solution 1 M",
            equilibrium_claim=False,
        )


def test_missing_pymbar_has_an_actionable_error(monkeypatch):
    module = _module()
    real_import = module.import_module

    def guarded_import(name):
        if name.startswith("pymbar"):
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(module, "import_module", guarded_import)
    state = np.zeros((4, 2))
    with pytest.raises(module.MBARDependencyError, match="implicit-free-energy"):
        module.analyze_mbar(
            [[state], [state]],
            lambda_values=[0.0, 1.0],
            temperature_kelvin=298.15,
            standard_state="gas 1 M to solution 1 M",
            equilibrium_claim=False,
        )


def test_solver_warning_capture_distinguishes_fallback_from_final_failure():
    module = _module()
    logger = logging.getLogger("pymbar.mbar_solvers")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL)

    try:
        with module._capture_solver_warnings() as messages:
            logger.warning(
                "Failed to reach a solution to within tolerance with hybr: "
                "trying next method"
            )
        assert logger.level == logging.CRITICAL
    finally:
        logger.setLevel(previous_level)
    fallback = module._solver_diagnostics(messages)
    assert fallback["warning_count"] == 1
    assert fallback["convergence_established"] is True
    assert (
        fallback["convergence_signal_source"]
        == "pymbar-4.0.3-warning-contract"
    )

    with module._capture_solver_warnings() as messages:
        logging.getLogger("pymbar.mbar_solvers").warning(
            "UPSTREAM: No Solution Found to within Tolerance."
        )
    failure = module._solver_diagnostics(messages)
    assert failure["convergence_established"] is False
    assert failure["final_nonconvergence_messages"] == [
        "UPSTREAM: No Solution Found to within Tolerance."
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("pymbar") is None,
    reason="optional PyMBAR dependency is not installed",
)
def test_pymbar_analysis_recovers_a_two_state_offset():
    module = _module()
    rng = np.random.default_rng(9147)
    sample_count = 500
    coordinate_state_0 = rng.normal(0.0, 1.0, sample_count)
    coordinate_state_1 = rng.normal(1.0, 1.0, sample_count)

    def reduced_potentials(coordinate):
        return np.column_stack(
            (
                0.5 * coordinate**2,
                0.5 * (coordinate - 1.0) ** 2,
            )
        )

    result = module.analyze_mbar(
        [
            [reduced_potentials(coordinate_state_0)],
            [reduced_potentials(coordinate_state_1)],
        ],
        lambda_values=[0.0, 1.0],
        temperature_kelvin=298.15,
        standard_state="synthetic equal-partition test",
        equilibrium_claim=True,
        detect_equilibration=False,
        minimum_uncorrelated_samples_per_state=20,
        minimum_effective_samples_per_state=20,
        minimum_adjacent_overlap=0.03,
    )

    assert result["endpoint_delta_g_kcal_mol"] == pytest.approx(0.0, abs=0.08)
    assert result["overlap"]["minimum_directional_adjacent_overlap"] > 0.03
    assert result["solver_diagnostics"]["convergence_established"] is True
    assert result["gates"]["checks"]["solver_convergence"] is True
    assert result["gates"]["statistical_gates_passed"] is True
    assert result["estimator"]["handwritten_estimator"] is False
    assert result["standard_state_declaration"] == "synthetic equal-partition test"
    assert result["standard_state_conversion"] == {
        "applied": False,
        "correction_kcal_mol": 0.0,
        "responsibility": "caller-supplied energies or downstream cycle",
    }
    assert "pymbar" in sys.modules
