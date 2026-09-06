"""CPU regression tests for the research-only MACE-EF V2 SCRF audit."""

from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_response import (
    FixedChargeCoordinates,
)

PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/implicit-solvation/benchmarks/audit_mace_ef_v2_scrf.py"
)


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("mace_ef_v2_scrf_audit", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _QuadraticContinuum:
    def drive_cartesian(self, source, *, warm_start):
        assert warm_start is False
        source = np.asarray(source, dtype=float)
        return -2.0 * MACE_POLAR_L1_PAIRING.density_to_field_order(source)

    @staticmethod
    def energy_ev(source):
        source = np.asarray(source, dtype=float)
        return -float(np.vdot(source, source))


class _QuadraticMonitor:
    def __init__(self, affine_source, *, maximum_iterations=40):
        self.affine_source = np.asarray(affine_source, dtype=float)
        self.coupling = SimpleNamespace(
            config=SimpleNamespace(
                scf=SimpleNamespace(
                    maximum_iterations=maximum_iterations,
                    source_residual_tolerance=1e-11,
                    mixing=0.7,
                    anderson_depth=4,
                    anderson_regularization=1e-10,
                    anderson_coefficient_l1_limit=100.0,
                    anderson_step_ratio_limit=10.0,
                )
            ),
            _validated_source=self._validated_source,
        )

    @staticmethod
    def _validated_source(source):
        source = np.asarray(source, dtype=float)
        if source.shape != (2, 4) or not np.isfinite(source).all():
            raise ValueError("source must be a finite two-atom block")
        return source.copy()

    def sample(self, field):
        field = np.asarray(field, dtype=float)
        field_in_density_order = MACE_POLAR_L1_PAIRING.field_to_density_order(field)
        energy = MACE_POLAR_L1_PAIRING.pair(self.affine_source, field) - 0.1 * float(
            np.vdot(field, field)
        )
        return energy, self.affine_source - 0.2 * field_in_density_order


@pytest.fixture
def quadratic_toy():
    affine_source = np.array([[0.3, 0.12, -0.08, 0.04], [-0.3, -0.06, 0.02, 0.10]])
    return _QuadraticMonitor(affine_source), _QuadraticContinuum(), affine_source


def test_solve_seed_converges_to_same_raw_quadratic_root_from_distinct_seeds(
    audit, quadratic_toy
):
    monitor, continuum, affine_source = quadratic_toy
    seeds = (
        np.zeros((2, 4)),
        np.array([[0.7, -0.2, 0.4, 0.1], [-0.7, 0.3, -0.1, -0.5]]),
    )
    results = [audit.solve_seed(monitor, continuum, seed) for seed in seeds]
    expected = affine_source / 0.6

    assert [result["status"] for result in results] == ["converged", "converged"]
    for result in results:
        np.testing.assert_allclose(result["source_raw"], expected, atol=2e-11, rtol=0.0)
        np.testing.assert_allclose(
            result["field_cartesian"],
            continuum.drive_cartesian(expected, warm_start=False),
            atol=4e-11,
            rtol=0.0,
        )
        assert result["trace"][-1]["maximum_residual"] <= 1e-11


def test_solve_seed_calls_stock_fixed_point_step(audit, quadratic_toy, monkeypatch):
    from maple.function.calculator.extra_correction.implicit import route2_fixed_point

    monitor, continuum, _ = quadratic_toy
    stock_step = route2_fixed_point.next_fixed_point_density
    calls = []

    def recording_stock_step(*args, **kwargs):
        calls.append(kwargs["solver"])
        return stock_step(*args, **kwargs)

    monkeypatch.setattr(
        route2_fixed_point, "next_fixed_point_density", recording_stock_step
    )
    result = audit.solve_seed(monitor, continuum, np.zeros((2, 4)))

    assert result["status"] == "converged"
    assert calls
    assert set(calls) == {route2_fixed_point.SAFEGUARDED_ANDERSON_SOLVER}
    assert all("step_method" in row for row in result["trace"][:-1])


def test_solve_seed_nonconvergence_does_not_report_a_fake_root(audit, quadratic_toy):
    _, continuum, affine_source = quadratic_toy
    monitor = _QuadraticMonitor(affine_source, maximum_iterations=1)
    result = audit.solve_seed(monitor, continuum, np.zeros((2, 4)))

    assert result["status"] == "not-converged"
    assert "source_raw" not in result
    assert "field_cartesian" not in result
    assert len(result["trace"]) == 1
    assert result["trace"][0]["maximum_residual"] > 1e-11
    assert not np.allclose(result["last_source_raw"], affine_source / 0.6)


def test_ledger_recomputes_quadratic_energy_identity_and_both_residuals(
    audit, quadratic_toy
):
    monitor, continuum, affine_source = quadratic_toy
    source = affine_source / 0.6
    field = continuum.drive_cartesian(source, warm_start=False)
    electronic = MACE_POLAR_L1_PAIRING.pair(affine_source, field) - 0.1 * float(
        np.vdot(field, field)
    )
    row = audit.ledger(monitor, continuum, source)

    assert row["electronic_energy_ev"] == pytest.approx(electronic)
    assert row["continuum_energy_ev"] == pytest.approx(-np.vdot(source, source))
    assert row["source_field_pairing_ev"] == pytest.approx(-2 * np.vdot(source, source))
    assert row["total_energy_ev"] == pytest.approx(electronic + np.vdot(source, source))
    assert row["half_identity_error_ev"] == pytest.approx(0.0, abs=1e-14)
    assert row["fresh_source_residual"] == pytest.approx(0.0, abs=1e-14)
    assert row["mapped_source_field_residual"] == pytest.approx(0.0, abs=1e-14)


def test_private_candidate_spec_is_distinct_metadata_without_global_registration(
    audit, monkeypatch, tmp_path
):
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef_specs import (
        MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
        registered_mace_polar_ef_checkpoint_specs,
    )

    before = dict(registered_mace_polar_ef_checkpoint_specs())
    monkeypatch.setattr(audit, "verified_bytes", lambda _path, _digest: b"metadata")
    candidate = audit.private_candidate_spec(tmp_path / "not-loaded.pt")
    after = dict(registered_mace_polar_ef_checkpoint_specs())

    assert after == before
    assert candidate is not MACE_POLAR_EF_V2_CHECKPOINT_SPEC
    assert candidate.name not in after
    assert candidate.model_id != MACE_POLAR_EF_V2_CHECKPOINT_SPEC.model_id
    assert candidate.model_family != MACE_POLAR_EF_V2_CHECKPOINT_SPEC.model_family
    assert candidate.checkpoint_size == len(b"metadata")


@pytest.mark.parametrize("atom_count", [1, 3])
def test_tangent_basis_is_cartesian_fixed_charge_isometry(audit, atom_count):
    basis = audit.tangent_basis(atom_count)
    coordinates = FixedChargeCoordinates(atom_count)

    expected_columns = []
    for index in range(coordinates.dimension):
        reduced = np.zeros(coordinates.dimension)
        reduced[index] = 1.0
        raw_density = coordinates.expand(reduced)
        cartesian_source = MACE_POLAR_L1_PAIRING.density_to_field_order(raw_density)
        expected_columns.append(cartesian_source.reshape(-1))
    expected = np.column_stack(expected_columns)

    assert basis.shape == (4 * atom_count, 4 * atom_count - 1)
    np.testing.assert_allclose(basis, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4 * atom_count - 1), atol=1e-14)
    np.testing.assert_allclose(basis[0::4].sum(axis=0), 0.0, atol=1e-14)


@pytest.mark.parametrize("atom_count", [0, -1, 1.5, np.nan])
def test_tangent_basis_rejects_invalid_atom_count(audit, atom_count):
    with pytest.raises((TypeError, ValueError)):
        audit.tangent_basis(atom_count)


def test_fd_jacobian_returns_rectangular_directional_derivatives(audit):
    linear = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0]])
    point = np.array([0.3, -0.2, 0.7])
    directions = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.5]])

    actual = audit.fd_jacobian(
        lambda value: linear @ value + np.array([0.2, -0.4]),
        point,
        directions,
        1e-5,
    )

    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, linear @ directions, atol=1e-10, rtol=0.0)


def test_fd_jacobian_preserves_raw_nonsymmetry(audit):
    raw = np.array([[0.1, 2.0], [-0.5, -0.3]])
    actual = audit.fd_jacobian(lambda value: raw @ value, np.zeros(2), np.eye(2), 1e-4)

    np.testing.assert_allclose(actual, raw, atol=1e-12, rtol=0.0)
    assert not np.allclose(actual, actual.T)


@pytest.mark.parametrize(
    "point,directions,step",
    [
        (np.zeros((2, 1)), np.eye(2), 1e-4),
        (np.zeros(2), np.ones((3, 1)), 1e-4),
        (np.array([0.0, np.nan]), np.eye(2), 1e-4),
        (np.zeros(2), np.array([[1.0], [np.inf]]), 1e-4),
        (np.zeros(2), np.eye(2), 0.0),
        (np.zeros(2), np.eye(2), -1e-4),
        (np.zeros(2), np.eye(2), np.nan),
    ],
)
def test_fd_jacobian_rejects_invalid_domain_inputs(audit, point, directions, step):
    with pytest.raises(ValueError):
        audit.fd_jacobian(lambda value: value, point, directions, step)


@pytest.mark.parametrize(
    "function",
    [
        lambda _value: np.array([[1.0]]),
        lambda _value: np.array([np.nan]),
    ],
)
def test_fd_jacobian_rejects_invalid_function_outputs(audit, function):
    with pytest.raises(ValueError):
        audit.fd_jacobian(function, np.zeros(2), np.eye(2), 1e-4)


def test_local_metrics_reports_known_stable_case(audit):
    identity = np.eye(3)
    metrics = audit.local_metrics(-2.0 * identity, -0.2 * identity, 0.4 * identity)

    assert metrics["continuum_support_rank"] == 3
    assert metrics["continuum_support_nullity"] == 0
    assert metrics["electronic_maximum_eigenvalue"] == pytest.approx(-0.2)
    assert metrics["electronic_passivity_passed"] is True
    assert metrics["electronic_antisymmetry_norm"] == pytest.approx(0.0)
    assert metrics["electronic_reciprocity_passed"] is True
    assert metrics["joint_minimum_eigenvalue"] == pytest.approx(0.6)
    assert metrics["joint_positive"] is True
    assert metrics["fixed_point_spectral_radius"] == pytest.approx(0.4)
    assert metrics["residual_sigma_min"] == pytest.approx(0.6)
    assert metrics["residual_condition_number"] == pytest.approx(1.0)
    assert metrics["jacobian_chain_relative_error"] == pytest.approx(0.0)


def test_joint_positive_does_not_override_electronic_passivity_failure(audit):
    identity = np.eye(2)
    metrics = audit.local_metrics(-0.1 * identity, 0.2 * identity, -0.02 * identity)

    assert metrics["joint_minimum_eigenvalue"] == pytest.approx(1.02)
    assert metrics["joint_positive"] is True
    assert metrics["electronic_maximum_eigenvalue"] == pytest.approx(0.2)
    assert metrics["electronic_passivity_passed"] is False


def test_local_metrics_reports_unstable_response_without_clipping(audit):
    identity = np.eye(2)
    metrics = audit.local_metrics(-2.0 * identity, -0.7 * identity, 1.4 * identity)

    assert metrics["fixed_point_spectral_radius"] == pytest.approx(1.4)
    assert metrics["residual_sigma_min"] == pytest.approx(0.4)
    assert metrics["jacobian_chain_relative_error"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "continuum",
    [
        np.array([[-1.0, 0.2], [0.0, -1.0]]),
        np.array([[-1.0, 0.0], [0.0, 0.1]]),
    ],
)
def test_local_metrics_rejects_nonreciprocal_or_wrong_sign_continuum(audit, continuum):
    with pytest.raises(ValueError):
        audit.local_metrics(continuum, -0.2 * np.eye(2), 0.2 * np.eye(2))


def test_rank_deficient_continuum_preserves_nullity_and_full_residual_sensitivity(
    audit,
):
    continuum = np.diag([-2.0, 0.0])
    uncoupled_hessian = np.diag([-0.2, 0.1])
    coupled_hessian = np.array([[-0.2, 0.0], [0.4, 0.1]])
    uncoupled = audit.local_metrics(
        continuum, uncoupled_hessian, uncoupled_hessian @ continuum
    )
    coupled = audit.local_metrics(
        continuum, coupled_hessian, coupled_hessian @ continuum
    )

    assert coupled["continuum_support_rank"] == 1
    assert coupled["continuum_support_nullity"] == 1
    assert coupled["joint_minimum_eigenvalue"] == pytest.approx(0.6)
    assert coupled["fixed_point_spectral_radius"] == pytest.approx(0.4)
    assert coupled["residual_sigma_min"] < uncoupled["residual_sigma_min"]
    assert coupled["residual_condition_number"] > uncoupled["residual_condition_number"]


def test_electronic_antisymmetry_is_reported_not_projected_away(audit):
    continuum = -0.1 * np.eye(2)
    hessian = np.array([[-0.2, 1.0], [-1.0, -0.2]])
    metrics = audit.local_metrics(continuum, hessian, hessian @ continuum)

    assert metrics["electronic_maximum_eigenvalue"] == pytest.approx(-0.2)
    assert metrics["electronic_passivity_passed"] is True
    assert metrics["electronic_antisymmetry_norm"] == pytest.approx(np.sqrt(2.0))
    assert metrics["electronic_reciprocity_passed"] is False


@pytest.mark.parametrize(
    "continuum,hessian,jacobian",
    [
        (np.eye(2), np.eye(3), np.eye(2)),
        (np.eye(2), np.eye(2), np.eye(3)),
        (np.eye(2), np.full((2, 2), np.nan), np.eye(2)),
    ],
)
def test_local_metrics_rejects_mismatched_or_nonfinite_matrices(
    audit, continuum, hessian, jacobian
):
    with pytest.raises(ValueError):
        audit.local_metrics(continuum, hessian, jacobian)


def _parent_receipt(audit):
    receipt = {
        "mode": "centered_v",
        "checkpoint_sha256": audit.CANDIDATE_SHA256,
        "state_dict_sha256": "state",
        "changed_members": [audit.SOURCE_MEMBER],
        "parent_checkpoint_sha256": audit.CHECKPOINT_SHA256,
    }
    parent = {
        "variants": [receipt],
        "state_dict_sha256": "state",
        "records": [
            {
                "mode": "centered_v",
                "gate_assessment": {"all_executed_checks_passed": True},
            }
        ],
    }
    protocol = {
        "candidate_sha256": audit.CANDIDATE_SHA256,
        "candidate_state_dict_sha256": "state",
    }
    return parent, protocol


def test_parent_receipt_binds_program_not_only_same_tensors(audit):
    parent, protocol = _parent_receipt(audit)
    audit.validate_parent_binding(parent, protocol)
    parent["variants"][0]["checkpoint_sha256"] = "different-program-same-tensors"
    with pytest.raises(ValueError, match="candidate program"):
        audit.validate_parent_binding(parent, protocol)


@pytest.mark.parametrize("failure", ["duplicate", "state", "members", "empty_records"])
def test_parent_receipt_rejects_incomplete_or_ambiguous_evidence(audit, failure):
    parent, protocol = _parent_receipt(audit)
    if failure == "duplicate":
        parent["variants"].append(copy.deepcopy(parent["variants"][0]))
    elif failure == "state":
        parent["state_dict_sha256"] = "another-state"
    elif failure == "members":
        parent["variants"][0]["changed_members"] = []
    else:
        parent["records"] = []
    with pytest.raises(ValueError):
        audit.validate_parent_binding(parent, protocol)


def test_local_interpretation_error_keeps_computed_raw_matrices(
    audit, quadratic_toy, monkeypatch
):
    monitor, continuum, affine_source = quadratic_toy
    protocol = json.loads((PATH.parent / "mace-ef-v2-scrf-prereg-v1.json").read_text())

    def reject(*args, **kwargs):
        raise ValueError("deliberate interpretation failure")

    monkeypatch.setattr(audit, "local_metrics", reject)
    result = audit.audit_local(monitor, continuum, affine_source / 0.6, protocol)
    assert result["all_local_gates_passed"] is False
    np.testing.assert_allclose(result["continuum_K"], -2 * np.eye(7), atol=1e-14)
    assert len(result["scan"]) == 3
    for row in result["scan"]:
        np.testing.assert_allclose(row["raw_H"], -0.2 * np.eye(7), atol=1e-12)
        np.testing.assert_allclose(row["raw_J_fd"], 0.4 * np.eye(7), atol=1e-12)
        assert row["interpretation_error"]["error_type"] == "ValueError"


def test_extremal_mode_confirms_small_active_response_without_threshold_clipping(
    audit, quadratic_toy
):
    _, continuum, affine_source = quadratic_toy

    class ActiveMonitor(_QuadraticMonitor):
        def __init__(self):
            super().__init__(affine_source)
            self.states = []

        def sample(self, field):
            field = np.asarray(field)
            energy = MACE_POLAR_L1_PAIRING.pair(
                self.affine_source, field
            ) + 0.004 * np.vdot(field, field)
            source = (
                self.affine_source
                + 0.008 * MACE_POLAR_L1_PAIRING.field_to_density_order(field)
            )
            self.states.append({"field": field.tolist(), "source": source.tolist()})
            return float(energy), source

    monitor = ActiveMonitor()
    protocol = json.loads((PATH.parent / "mace-ef-v2-scrf-prereg-v1.json").read_text())
    confirmation = json.loads(
        (PATH.parent / "mace-ef-v2-scrf-mode-confirmation-prereg-v1.json").read_text()
    )
    local = {
        "tangent_basis_cartesian": audit.tangent_basis(2).tolist(),
        "source_metric_diagonal": [1.0] * 8,
        "continuum_K": (-2 * np.eye(7)).tolist(),
        "scan": [{"raw_H": (0.008 * np.eye(7)).tolist(), "metrics": {}}],
    }
    result = audit.confirm_extremal_mode(
        monitor, continuum, affine_source / 1.016, local, protocol, confirmation
    )
    assert result["all_measured_curvatures_positive"] is True
    assert result["predicted_curvature"] == pytest.approx(0.008)
    assert result["realization_error"] < 1e-14
    for row in result["scan"]:
        assert row["curvature"] == pytest.approx(0.008, abs=1e-12)
    assert len(monitor.states) == 9


def test_frozen_scrf_evidence_recomputes_without_promoting_numeric_pass(audit):
    directory = PATH.parent
    result = json.loads(
        (directory / "mace-ef-v2-scrf-water-methanol-v1.json").read_text()
    )
    protocol_path = directory / "mace-ef-v2-scrf-prereg-v1.json"
    protocol = json.loads(protocol_path.read_text())
    assert (
        result["preregistration_sha256"]
        == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )
    key = "docs/implicit-solvation/benchmarks/audit_mace_ef_v2_scrf.py"
    assert (
        result["source_files_sha256"][key]
        == hashlib.sha256(PATH.read_bytes()).hexdigest()
    )
    assert result["release_admitted"] is False
    assert [r["molecule"]["name"] for r in result["records"]] == ["water", "methanol"]
    for record in result["records"]:
        assert record["root_repeatability_passed"] is True
        assert (
            record["candidate_state_sha256_after"]
            == result["candidate_state_dict_sha256"]
        )
        local = record["local_stability"]
        assert local["physical_concavity_certified"] is False
        for scan in local["scan"]:
            recomputed = audit.local_metrics(
                local["continuum_K"], scan["raw_H"], scan["raw_J_fd"]
            )
            for field in [
                "electronic_maximum_eigenvalue",
                "joint_minimum_eigenvalue",
                "fixed_point_spectral_radius",
                "residual_sigma_min",
            ]:
                assert recomputed[field] == pytest.approx(
                    scan["metrics"][field], abs=1e-12
                )
        confirmation = record["extremal_mode_confirmation"]
        for scan in confirmation["scan"]:
            plus, minus = scan["sides"]
            assert scan["curvature"] == pytest.approx(
                (plus["derivative"] - minus["derivative"]) / (2 * scan["step"]),
                abs=1e-12,
            )
            for side in scan["sides"]:
                raw = record["electronic_states"][side["electronic_state_index"]][
                    "source_raw"
                ]
                expected = (
                    MACE_POLAR_L1_PAIRING.pair(
                        raw, confirmation["field_direction_cartesian"]
                    )
                    / protocol["metric"]["energy_scale_ev"]
                )
                assert side["derivative"] == pytest.approx(expected, abs=1e-12)
    methanol = result["records"][1]
    assert methanol["local_stability"]["all_local_gates_passed"] is True
    assert (
        methanol["extremal_mode_confirmation"]["all_measured_curvatures_positive"]
        is True
    )
    assert (
        methanol["local_stability"]["scan"][-1]["metrics"][
            "symmetric_spectrum_nonpositive"
        ]
        is False
    )
