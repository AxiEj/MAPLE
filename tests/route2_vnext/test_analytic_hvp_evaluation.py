from __future__ import annotations

import gc
from dataclasses import FrozenInstanceError, replace
import weakref

import numpy as np
import pytest

from maple.solvation.derivatives.analytic import (
    ANALYTIC_FORCE_DERIVATIVE_POLICY_SHA256,
    ANALYTIC_HVP_DERIVATIVE_POLICY,
    ANALYTIC_HVP_DERIVATIVE_POLICY_SHA256,
    AnalyticForceEvaluation,
    AnalyticHVPEvaluation,
)


def _force_evaluation(**overrides) -> AnalyticForceEvaluation:
    values = {
        "forces_eV_per_A": np.arange(6, dtype=np.float32).reshape(2, 3) / 10,
        "energy_eV": -1.25,
        "geometry_sha256": "a" * 64,
        "model_input_sha256": "b" * 64,
        "configuration_sha256": "c" * 64,
        "scalar_contract_id": "route2-test-force-scalar-v1",
        "device": "cuda:0",
        "component_energies_eV": {
            "vacuum": -1.0,
            "ddpcm": -0.2,
            "cds": -0.05,
        },
        "computation_dtype": "float32",
    }
    values.update(overrides)
    return AnalyticForceEvaluation(**values)


def _evaluation(**overrides) -> AnalyticHVPEvaluation:
    values = {
        "force_evaluation": _force_evaluation(),
        "direction": np.array(
            [[np.float32(1.0 / 3.0), 0.0, 0.25], [0.5, -0.75, 1.0]],
            dtype=np.float32,
        ),
        "hvp_eV_per_A2": np.arange(6, dtype=np.float32).reshape(2, 3) / 7,
    }
    values.update(overrides)
    return AnalyticHVPEvaluation(**values)


def test_hvp_evaluation_is_immutable_detached_and_hvp_specific():
    direction = np.arange(6, dtype=np.float32).reshape(2, 3)
    hvp = direction + 10.0
    evaluation = _evaluation(direction=direction, hvp_eV_per_A2=hvp)
    direction[0, 0] = 99.0
    hvp[0, 0] = 99.0

    assert evaluation.direction[0, 0] == 0.0
    assert evaluation.hvp_eV_per_A2[0, 0] == 10.0
    assert evaluation.direction.dtype == np.float64
    assert evaluation.hvp_eV_per_A2.dtype == np.float64
    assert not evaluation.direction.flags.writeable
    assert not evaluation.hvp_eV_per_A2.flags.writeable
    assert evaluation.derivative_policy_sha256 == ANALYTIC_HVP_DERIVATIVE_POLICY_SHA256
    assert (
        evaluation.derivative_policy_sha256 != ANALYTIC_FORCE_DERIVATIVE_POLICY_SHA256
    )
    assert evaluation.derivative_method == "torch-autograd"
    assert evaluation.dtype == evaluation.computation_dtype == "float32"
    assert evaluation.storage_dtype == "float64"
    assert ANALYTIC_HVP_DERIVATIVE_POLICY["coordinate_finite_differences"] is False
    assert ANALYTIC_HVP_DERIVATIVE_POLICY["dense_hessian_materialization"] is False
    assert ANALYTIC_HVP_DERIVATIVE_POLICY["hvp_evaluation"] == "direct"
    with pytest.raises(ValueError):
        evaluation.direction[0, 0] = 1.0
    with pytest.raises(ValueError):
        evaluation.hvp_eV_per_A2[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        evaluation.force_evaluation = _force_evaluation()


def test_hvp_evaluation_does_not_retain_a_torch_graph():
    torch = pytest.importorskip("torch")
    positions = torch.arange(6, dtype=torch.float32, requires_grad=True)
    computed_direction = (positions * 2.0).reshape(2, 3)
    graph_value = weakref.ref(computed_direction)

    evaluation = _evaluation(
        direction=computed_direction.detach().numpy(),
        hvp_eV_per_A2=(computed_direction + 1.0).detach().numpy(),
    )
    del computed_direction
    del positions
    gc.collect()

    assert graph_value() is None
    assert isinstance(evaluation.direction, np.ndarray)
    assert isinstance(evaluation.hvp_eV_per_A2, np.ndarray)


def test_hvp_digest_binds_nested_identity_direction_and_hvp():
    original = _evaluation()
    changed_force = _evaluation(
        force_evaluation=_force_evaluation(model_input_sha256="d" * 64)
    )
    changed_direction = _evaluation(direction=original.direction + 0.125)
    changed_hvp = _evaluation(hvp_eV_per_A2=original.hvp_eV_per_A2 + 0.125)

    assert (
        len(
            {
                original.evaluation_sha256,
                changed_force.evaluation_sha256,
                changed_direction.evaluation_sha256,
                changed_hvp.evaluation_sha256,
            }
        )
        == 4
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"direction": np.zeros((2, 2))}, "direction shape"),
        ({"direction": np.full((2, 3), np.nan)}, "direction.*finite"),
        ({"hvp_eV_per_A2": np.zeros((3, 3))}, "hvp_eV_per_A2 shape"),
        ({"hvp_eV_per_A2": np.full((2, 3), np.inf)}, "hvp_eV_per_A2.*finite"),
    ],
)
def test_hvp_evaluation_rejects_shape_and_nonfinite_content(change, match):
    with pytest.raises(ValueError, match=match):
        _evaluation(**change)


def test_hvp_evaluation_requires_exact_nested_force_contract_type():
    with pytest.raises(TypeError, match="force_evaluation.*AnalyticForceEvaluation"):
        _evaluation(force_evaluation=object())


def test_hvp_evaluation_preserves_actual_f32_probe_values_and_f64_values():
    requested = np.array(
        [[1.0 / 3.0, np.pi, np.e], [-1.0 / 7.0, 1.0e-20, 1.0e20]],
        dtype=np.float64,
    )
    actual_probe = requested.astype(np.float32)
    f32 = _evaluation(direction=actual_probe)
    f64 = _evaluation(
        force_evaluation=_force_evaluation(computation_dtype="float64"),
        direction=requested,
        hvp_eV_per_A2=requested * 2.0,
    )

    np.testing.assert_array_equal(f32.direction, actual_probe.astype(np.float64))
    assert np.any(f32.direction != requested)
    np.testing.assert_array_equal(f64.direction, requested)
    np.testing.assert_array_equal(f64.hvp_eV_per_A2, requested * 2.0)


def test_forged_hvp_digest_and_nested_force_identity_are_rejected():
    original = _evaluation()
    with pytest.raises(ValueError, match="does not match analytic HVP content"):
        replace(original, direction=original.direction + 1.0)

    forged_force = object.__new__(AnalyticForceEvaluation)
    with pytest.raises(TypeError, match="valid AnalyticForceEvaluation"):
        _evaluation(force_evaluation=forged_force)

    tampered_force = _force_evaluation()
    object.__setattr__(tampered_force, "evaluation_sha256", "f" * 64)
    with pytest.raises(TypeError, match="valid AnalyticForceEvaluation"):
        _evaluation(force_evaluation=tampered_force)
