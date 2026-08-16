from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.route2_release import create_maple_cds_w1_m4_preregistration as m4
from tools.route2_release import fit_maple_cds_w1_m4_pseudohuber as fitter


def test_pseudo_huber_gradient_and_hessian_are_exact_finite_differences():
    generator = np.random.default_rng(20260816)
    design = generator.normal(size=(41, 3))
    target = generator.normal(size=41)
    point = generator.normal(size=3)
    value, gradient, hessian = fitter._pseudo_huber_value_gradient_hessian(
        point, design, target
    )
    assert np.isfinite(value)
    assert hessian == pytest.approx(hessian.T, abs=1.0e-14)
    assert np.linalg.eigvalsh(hessian)[0] > 0.0

    step = 2.0e-6
    fd_gradient = np.empty(3)
    fd_hessian = np.empty((3, 3))
    for coordinate in range(3):
        direction = np.zeros(3)
        direction[coordinate] = step
        plus = fitter._pseudo_huber_value_gradient_hessian(
            point + direction, design, target
        )
        minus = fitter._pseudo_huber_value_gradient_hessian(
            point - direction, design, target
        )
        fd_gradient[coordinate] = (plus[0] - minus[0]) / (2.0 * step)
        fd_hessian[:, coordinate] = (plus[1] - minus[1]) / (2.0 * step)
    assert gradient == pytest.approx(fd_gradient, abs=2.0e-10)
    assert hessian == pytest.approx(fd_hessian, abs=2.0e-10)


def test_pseudo_huber_is_unique_reproducible_and_robust_to_vertical_outliers():
    generator = np.random.default_rng(7)
    design = generator.normal(size=(300, 3))
    expected = np.array([0.7, -1.2, 0.35])
    response = design @ expected + generator.normal(scale=0.2, size=300)
    response[:5] += 20.0
    validation = generator.normal(size=(27, 3))

    result = fitter.fit_unique_pseudo_huber(
        design, response, validation_design=validation
    )
    coefficients = np.asarray(result["reduced_coefficients"])
    ols = np.linalg.lstsq(design, response, rcond=None)[0]

    assert coefficients == pytest.approx(expected, abs=3.0e-2)
    assert np.linalg.norm(coefficients - expected) < np.linalg.norm(ols - expected)
    assert (
        result["independent_start_validation_prediction_max_abs_difference_kcal_mol"]
        < 1.0e-10
    )
    assert (
        result["zero_start_certificate"]["gradient_infinity_norm"]
        <= m4.STATIONARITY_INFINITY_TOLERANCE
    )


def test_pseudo_huber_rejects_an_ill_conditioned_full_rank_design():
    generator = np.random.default_rng(19)
    first = generator.normal(size=80)
    design = np.column_stack(
        (
            first,
            first + 1.0e-6 * generator.normal(size=80),
            generator.normal(size=80),
        )
    )
    response = generator.normal(size=80)

    with pytest.raises(fitter.M4FitError, match="ill-conditioned"):
        fitter.fit_unique_pseudo_huber(design, response)


def _write_read_only_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload, sort_keys=True))
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m4_parent_requires_the_exact_terminal_m3_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prereg: dict[str, object] = {
        "artifact": m4.M3_PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-m3-target-use-or-fit",
        "source_git_head": m4.M3_SOURCE_GIT_HEAD,
        "water_record_count": m4.EXPECTED_WATER_COUNT,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_opened": False,
        "fitting_or_calibration_performed": False,
        "grouped_oof": {
            "fold_count": m4.OOF_FOLD_COUNT,
            "family_count": 202,
            "maximum_family_size": 7,
            "fold_record_counts": [31, 31, 31, 31, 31, 31, 30, 30, 30, 30],
        },
    }
    prereg["self_sha256"] = m4.canonical_json_sha256(prereg)
    prereg_path = tmp_path / "m3-prereg.json"
    prereg_file_sha = _write_read_only_json(prereg_path, prereg)

    failure: dict[str, object] = {
        "artifact": m4.M3_FAILURE_ARTIFACT,
        "schema_version": 1,
        "status": "development-fail-numerically-nonunique-lad",
        "source_git_head": m4.M3_SOURCE_GIT_HEAD,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": prereg_file_sha,
        "preregistration_self_sha256": prereg["self_sha256"],
        "command_exit_code": 1,
        "terminal_error": "Primary LAD coefficient 0 is numerically non-unique.",
        "oof_prediction_completed": False,
        "oof_mae_computed": False,
        "threshold_or_uniqueness_rule_changed_after_failure": False,
        "confirmation_partition_opened": False,
    }
    failure["self_sha256"] = m4.canonical_json_sha256(failure)
    failure_path = tmp_path / "m3-failure.json"
    failure_file_sha = _write_read_only_json(failure_path, failure)
    monkeypatch.setattr(m4, "M3_PREREGISTRATION_FILE_SHA256", prereg_file_sha)
    monkeypatch.setattr(m4, "M3_PREREGISTRATION_SELF_SHA256", prereg["self_sha256"])
    monkeypatch.setattr(m4, "M3_FAILURE_FILE_SHA256", failure_file_sha)
    monkeypatch.setattr(m4, "M3_FAILURE_SELF_SHA256", failure["self_sha256"])

    observed_prereg, observed_failure = m4.validate_m3_terminal_parent(
        m3_preregistration_path=prereg_path,
        m3_failure_path=failure_path,
    )
    assert observed_prereg == prereg
    assert observed_failure == failure

    failure["oof_mae_computed"] = True
    failure["self_sha256"] = m4.canonical_json_sha256(
        {key: value for key, value in failure.items() if key != "self_sha256"}
    )
    failure_path.chmod(0o644)
    changed_file_sha = _write_read_only_json(failure_path, failure)
    monkeypatch.setattr(m4, "M3_FAILURE_FILE_SHA256", changed_file_sha)
    monkeypatch.setattr(m4, "M3_FAILURE_SELF_SHA256", failure["self_sha256"])
    with pytest.raises(m4.M4PreregistrationError, match="oof_mae_computed"):
        m4.validate_m3_terminal_parent(
            m3_preregistration_path=prereg_path,
            m3_failure_path=failure_path,
        )


def test_m4_fitter_rejects_a_self_rehashed_fit_contract_drift(tmp_path: Path):
    payload: dict[str, object] = {key: None for key in m4.PREREGISTRATION_KEYS}
    payload.update(
        {
            "artifact": m4.PREREGISTRATION_ARTIFACT,
            "schema_version": 1,
            "status": "locked-before-first-m4-target-use-or-fit",
            "candidate_profile_id": m4.CANDIDATE_PROFILE_ID,
            "partition": "development-water-only",
            "water_record_count": m4.EXPECTED_WATER_COUNT,
            "source_root": str(m4._SOURCE_ROOT),
            "experimental_targets_read_by_m4_preregistration": False,
            "hybrid_prediction_records_read_by_m4_preregistration": False,
            "confirmation_selection_manifest_opened": False,
            "confirmation_records_opened": False,
            "fitting_or_calibration_performed": False,
            "m3_oof_mae_computed": False,
            "m3_threshold_or_rule_changed_after_failure": False,
            "fit_contract": {**m4.frozen_fit_contract(), "delta_tuned": True},
            "development_decision_rule": m4.frozen_development_decision_rule(),
        }
    )
    payload["self_sha256"] = m4.canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "self_sha256"}
    )
    path = tmp_path / "m4-prereg.json"
    _write_read_only_json(path, payload)

    with pytest.raises(fitter.M4FitError, match="fit contract"):
        fitter._read_preregistration(path, m4._SOURCE_ROOT)
