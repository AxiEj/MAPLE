from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.route2_release import (
    create_maple_cds_w1_positive_ph1_preregistration as CREATOR,
)
from tools.route2_release import fit_maple_cds_w1_positive_ph1 as FITTER

ROOT = Path(__file__).resolve().parents[2]
CREATOR_PATH = (
    ROOT / "tools/route2_release/create_maple_cds_w1_positive_ph1_preregistration.py"
)
FITTER_PATH = ROOT / "tools/route2_release/fit_maple_cds_w1_positive_ph1.py"


def _design(seed: int = 19) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(CREATOR.EXPECTED_WATER_COUNT, 18))
    values += np.linspace(0.05, 0.4, 18)[None, :]
    return values


def _partition() -> tuple[np.ndarray, np.ndarray]:
    validation = np.arange(0, CREATOR.EXPECTED_WATER_COUNT, 10, dtype=int)
    training = np.asarray(
        sorted(set(range(CREATOR.EXPECTED_WATER_COUNT)) - set(validation.tolist())),
        dtype=int,
    )
    return training, validation


def test_target_blind_subspace_has_canonical_metric_invariants() -> None:
    values = _design()
    training, validation = _partition()
    result = CREATOR.build_target_blind_subspace(
        values,
        training,
        validation,
        label="synthetic-fold",
    )
    frame = np.asarray(result["canonical_frame"], dtype=float)
    stock_unit = np.asarray(result["stock_unit_vector"], dtype=float)
    projector = np.asarray(result["rank_two_projector"], dtype=float)
    modes = np.asarray(result["physical_coefficient_modes"], dtype=float)
    column_norms = np.asarray(result["column_l2_norms"], dtype=float)
    np.testing.assert_allclose(frame.T @ frame, np.eye(2), atol=2.0e-14)
    np.testing.assert_allclose(frame.T @ stock_unit, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(projector @ frame, frame, atol=2.0e-14)
    np.testing.assert_allclose(modes * column_norms[:, None], frame, atol=0.0)
    assert result["target_values_read_or_used"] is False
    assert result["validation_geometry_used_to_construct_subspace_or_scaling"] is False


def test_canonical_frame_depends_only_on_projector() -> None:
    generator = np.random.default_rng(31)
    raw = generator.normal(size=(18, 2))
    basis, _ = np.linalg.qr(raw)
    projector = basis @ basis.T
    frame, pair, diagnostics = CREATOR._canonical_frame(projector)
    rotation = np.asarray(((0.6, -0.8), (0.8, 0.6)), dtype=float)
    rotated = basis @ rotation
    frame_rotated, pair_rotated, diagnostics_rotated = CREATOR._canonical_frame(
        rotated @ rotated.T
    )
    np.testing.assert_allclose(frame_rotated, frame, atol=3.0e-15)
    assert pair_rotated == pair
    assert diagnostics_rotated["selected_principal_minor"] == pytest.approx(
        diagnostics["selected_principal_minor"], abs=3.0e-16
    )


def test_subspace_rejects_inactive_column() -> None:
    values = _design()
    values[:, 4] = 0.0
    training, validation = _partition()
    with pytest.raises(CREATOR.PositivePH1PreregistrationError, match="inactive"):
        CREATOR.build_target_blind_subspace(
            values,
            training,
            validation,
            label="inactive",
        )


def test_pseudo_huber_gradient_and_hessian_match_finite_difference() -> None:
    generator = np.random.default_rng(47)
    design = generator.normal(size=(41, 3))
    target = generator.normal(size=41)
    point = generator.normal(size=3)
    _objective, gradient, hessian, _residuals = FITTER._evaluate_pseudo_huber(
        point,
        design,
        target,
    )
    step = 2.0e-6
    finite_gradient = np.empty(3)
    finite_hessian = np.empty((3, 3))
    for index in range(3):
        direction = np.zeros(3)
        direction[index] = step
        plus = FITTER._evaluate_pseudo_huber(point + direction, design, target)
        minus = FITTER._evaluate_pseudo_huber(point - direction, design, target)
        finite_gradient[index] = (plus[0] - minus[0]) / (2.0 * step)
        finite_hessian[:, index] = (plus[1] - minus[1]) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_gradient, rtol=3.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(hessian, finite_hessian, rtol=2.0e-9, atol=2.0e-9)


def test_certified_newton_two_start_finds_unique_minimizer() -> None:
    generator = np.random.default_rng(59)
    all_design = generator.normal(size=(CREATOR.EXPECTED_WATER_COUNT, 3))
    all_design /= np.linalg.norm(all_design[:275], axis=0)
    target = all_design @ np.asarray((1.2, -0.7, 0.3))
    target += 0.08 * generator.normal(size=CREATOR.EXPECTED_WATER_COUNT)
    target[::43] += 2.5
    result = FITTER.fit_certified_pseudo_huber(
        all_row_design=all_design,
        training_indices=np.arange(275, dtype=int),
        target_all_rows=target,
    )
    assert result["zero_start"]["certificate"]["certificate_passed"] is True
    assert (
        result["svd_least_squares_start"]["certificate"]["certificate_passed"] is True
    )
    assert (
        result["two_start_endpoint_l2_distance"] <= result["two_start_replay_allowance"]
    )
    endpoint = np.asarray(result["authoritative_scaled_endpoint"], dtype=float)
    _value, gradient, _hessian, _residual = FITTER._evaluate_pseudo_huber(
        endpoint,
        all_design[:275],
        target[:275],
    )
    assert np.linalg.norm(gradient) < 1.0e-10


def test_minimizer_certificate_rejects_nonstationary_point() -> None:
    generator = np.random.default_rng(61)
    all_design = generator.normal(size=(CREATOR.EXPECTED_WATER_COUNT, 3))
    target = generator.normal(size=CREATOR.EXPECTED_WATER_COUNT)
    passed, certificate = FITTER._minimizer_certificate(
        point=np.asarray((9.0, -7.0, 5.0)),
        training_design=all_design[:275],
        all_row_design=all_design,
        target=target[:275],
    )
    assert passed is False
    assert certificate["certificate_passed"] is False
    assert certificate["gradient_l2_norm"] >= certificate["strict_gradient_bound"]


def test_physical_and_optimizer_predictions_reconstruct() -> None:
    values = _design(71)
    training, validation = _partition()
    subspace = CREATOR.build_target_blind_subspace(
        values,
        training,
        validation,
        label="reconstruction",
    )
    optimizer = np.asarray(subspace["all_row_optimizer_design"], dtype=float)
    target = optimizer @ np.asarray((0.8, -0.4, 0.2))
    result = FITTER.fit_certified_pseudo_huber(
        all_row_design=optimizer,
        training_indices=training,
        target_all_rows=target,
    )
    theta, physical, error, point = FITTER._subspace_to_theta_and_prediction(
        subspace=subspace,
        scaled_endpoint=np.asarray(
            result["authoritative_scaled_endpoint"],
            dtype=float,
        ),
        design=values,
    )
    assert theta.shape == (18,)
    assert point.shape == (3,)
    assert error <= CREATOR.REPRESENTATION_TOLERANCE_KCAL_MOL
    np.testing.assert_allclose(physical, target, atol=1.0e-6)


def test_actual_positive_parent_matrix_passes_all_pretarget_subspace_gates() -> None:
    aggregate_path = (
        Path.home()
        / ".local/share/maple/route2/maple-cds-w1-positive-parent-features-v2/aggregate.json"
    )
    m3_path = (
        Path.home()
        / ".local/share/maple/route2/maple-cds-w1-m3-v1/preregistration.json"
    )
    if not aggregate_path.exists() or not m3_path.exists():
        pytest.skip("external target-blind positive-parent evidence is unavailable")
    aggregate = json.loads(aggregate_path.read_text())
    design = np.asarray(
        [record["design_row_angstrom2_div_1000"] for record in aggregate["records"]],
        dtype=float,
    )
    m3 = json.loads(m3_path.read_text())
    row_folds = np.asarray(m3["grouped_oof"]["row_fold_assignments"], dtype=int)
    all_rows = np.arange(CREATOR.EXPECTED_WATER_COUNT, dtype=int)
    conditions = []
    for fold in range(CREATOR.OOF_FOLD_COUNT):
        subspace = CREATOR.build_target_blind_subspace(
            design,
            all_rows[row_folds != fold],
            all_rows[row_folds == fold],
            label=f"actual-fold-{fold}",
        )
        conditions.append(subspace["training_optimizer_condition_number"])
    deployment = CREATOR.build_target_blind_subspace(
        design,
        all_rows,
        np.empty(0, dtype=int),
        label="actual-deployment",
    )
    assert max(conditions) < 5.0
    assert deployment["training_optimizer_condition_number"] < 5.0


def test_creator_source_does_not_read_target_or_hybrid_values() -> None:
    text = CREATOR_PATH.read_text()
    assert "experimental_delta_g_kcal_mol" not in text
    assert "continuum_polarization_kcal_mol" not in text
    assert "smd_cds_kcal_mol" not in text
    assert CREATOR.frozen_fit_contract()["intercept"] is False
    assert CREATOR.frozen_fit_contract()["regularizer"] is False
    assert CREATOR.frozen_pro_review()["q12_marker"] == "FREEZE PH1.0 CONTRACT"


@pytest.mark.parametrize("path", (CREATOR_PATH, FITTER_PATH))
def test_positive_ph1_cli_help_is_dependency_local(path: Path) -> None:
    result = subprocess.run(
        (sys.executable, str(path), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-root" in result.stdout
