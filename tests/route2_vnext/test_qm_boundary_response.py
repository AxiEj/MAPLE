from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from maple.solvation.release.qm_boundary_response import (
    area_weighted_relative_error,
    central_difference,
    compare_boundary_response,
    reference_uncertainty,
    response_norm,
)

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = SOURCE_ROOT / (
    "docs/route2/preregistrations/" "mace-mdp-polar-qm-boundary-response-v1.json"
)
RUNNER = SOURCE_ROOT / (
    "tools/route2_release/run_mace_mdp_polar_qm_boundary_response.py"
)
CREATOR = SOURCE_ROOT / (
    "tools/route2_release/"
    "create_mace_mdp_polar_qm_boundary_response_preregistration.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(matrix: np.ndarray):
    def apply(value: np.ndarray) -> np.ndarray:
        return matrix @ np.asarray(value, dtype=np.float64)

    return apply


def test_central_difference_is_exact_for_affine_samples() -> None:
    intercept = np.asarray([0.2, -0.4, 0.7])
    derivative = np.asarray([1.5, -2.0, 0.25])
    step = 2.0e-3
    result = central_difference(
        intercept + step * derivative,
        intercept - step * derivative,
        step=step,
    )
    np.testing.assert_allclose(result, derivative, rtol=0.0, atol=4.0e-14)
    assert not result.flags.writeable


def test_central_difference_preserves_density_matrix_shape() -> None:
    derivative = np.asarray([[1.0, 0.2], [0.2, -1.0]])
    origin = np.eye(2)
    step = 1.0e-4
    result = central_difference(
        origin + step * derivative,
        origin - step * derivative,
        step=step,
    )
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, derivative, rtol=0.0, atol=2.0e-13)


@pytest.mark.parametrize(
    ("plus", "minus", "step"),
    [
        ([1.0], [1.0, 2.0], 1.0),
        ([1.0], [1.0], 0.0),
        ([1.0], [1.0], float("nan")),
        ([float("nan")], [1.0], 1.0),
    ],
)
def test_central_difference_fails_closed(
    plus: object,
    minus: object,
    step: float,
) -> None:
    with pytest.raises(ValueError):
        central_difference(plus, minus, step=step)


def test_response_norm_uses_negative_pcmsolver_quadratic_form() -> None:
    response = _response(-np.diag([4.0, 9.0]))
    assert response_norm([1.0, 2.0], apply_response=response) == pytest.approx(
        np.sqrt(40.0)
    )


def test_response_norm_rejects_nonpassive_operator() -> None:
    with pytest.raises(RuntimeError, match="not passive"):
        response_norm([1.0], apply_response=_response(np.eye(1)))


def test_boundary_comparison_matches_dense_oracle() -> None:
    metric = np.diag([2.0, 3.0, 5.0])
    response = _response(-metric)
    reference = np.asarray([1.0, -2.0, 0.5])
    candidate = np.asarray([0.8, -1.6, 0.4])
    areas = np.asarray([1.0, 2.0, 4.0])
    result = compare_boundary_response(
        candidate,
        reference,
        areas=areas,
        apply_response=response,
    )
    reference_norm = np.sqrt(reference @ metric @ reference)
    candidate_norm = np.sqrt(candidate @ metric @ candidate)
    error = candidate - reference
    error_norm = np.sqrt(error @ metric @ error)
    assert result.reference_response_norm == pytest.approx(reference_norm)
    assert result.candidate_response_norm == pytest.approx(candidate_norm)
    assert result.error_response_norm == pytest.approx(error_norm)
    assert result.relative_response_error == pytest.approx(error_norm / reference_norm)
    assert result.correlation == pytest.approx(1.0)
    assert result.area_weighted_relative_error == pytest.approx(0.2)
    assert result.as_dict()["error_response_norm"] == pytest.approx(error_norm)


def test_reference_uncertainty_uses_larger_physical_component() -> None:
    response = _response(-np.eye(2))
    result = reference_uncertainty(
        primary_fine=[2.0, 0.0],
        primary_coarse=[1.9, 0.0],
        control_fine=[2.0, 0.3],
        apply_response=response,
    )
    assert result.finite_field_response_norm == pytest.approx(0.1)
    assert result.basis_response_norm == pytest.approx(0.3)
    assert result.envelope_response_norm == pytest.approx(0.3)
    assert result.envelope_relative_to_reference == pytest.approx(0.15)


def test_area_weighted_metric_rejects_invalid_areas_and_zero_reference() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        area_weighted_relative_error([1.0], [1.0], areas=[0.0])
    with pytest.raises(ValueError, match="zero"):
        area_weighted_relative_error([1.0], [0.0], areas=[1.0])


def test_preregistration_is_self_hashed_target_free_and_source_bound() -> None:
    creator = _load_module(CREATOR, "route2_qm_boundary_creator_test")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    unsigned = dict(prereg)
    expected = unsigned.pop("preregistration_sha256")
    assert expected == creator._canonical_sha256(unsigned)
    assert prereg["status"] == (
        "frozen-before-first-qm-induced-boundary-response-execution"
    )
    decision = prereg["decision_contract"]
    assert decision["not_a_fit"] is True
    assert decision["no_projector_selected"] is True
    assert decision["no_source_rescaling"] is True
    assert decision["no_solvation_target_read"] is True
    assert decision["no_capability_admission"] is True
    assert prereg["reference_protocol"]["field_gradient_steps_volt_per_angstrom"] == [
        1.0e-3,
        5.0e-4,
    ]
    for relative, expected_sha in prereg["source_files_sha256"].items():
        content = (SOURCE_ROOT / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_sha


def test_runner_ast_has_no_experimental_solvation_target_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    strings = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    forbidden = (
        "experimental_delta_g",
        "freesolv",
        "mnsol",
        "hydration target",
        "solvation residual",
    )
    assert not any(token in value for token in forbidden for value in strings)
