from __future__ import annotations

import ast
from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import validate_cha_r6_v2_local as validation  # pyright: ignore[reportMissingImports]

FIXTURE = (
    Path(__file__).resolve().parent / "data/cha_r6_v2/frozen_v1_terminal_fixture.json"
)


@pytest.fixture(scope="module")
def fixture():
    return validation.load_fixture(FIXTURE)


def _actual_envelope(fixture, evidence_root: Path):
    rows = deepcopy(fixture["rows"])
    receipts = {}
    for row in rows:
        row["provenance"]["source_identity_sha256"] = "b" * 64
        receipt = {
            "schema": "route1-cha-r6-v1-fresh-replay-row-v1",
            "row_id": row["row_id"],
            "numerical_profile_id": validation.V1_PROFILE_ID,
            "preregistration_sha256": "a" * 64,
            "normalized_row_sha256": validation.hashlib.sha256(
                validation._canonical(row)
            ).hexdigest(),
            "normalized_row": row,
            "source_identity_sha256": "b" * 64,
        }
        receipt["content_sha256"] = validation._content_sha256(receipt)
        relative = Path("rows") / f"{row['row_id']}.json"
        path = evidence_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(validation._canonical(receipt) + b"\n")
        receipts[row["row_id"]] = {
            "relative_path": str(relative),
            "file_sha256": validation._sha256_file(path),
            "content_sha256": receipt["content_sha256"],
        }
    value = {
        "schema": "route1-cha-r6-v1-fresh-replay-v1",
        "fresh_run": True,
        "producer": "root-v1-replay-from-auditor-v1",
        "producer_source_sha256": "d" * 64,
        "source_identity_sha256": "b" * 64,
        "numerical_profile_id": validation.V1_PROFILE_ID,
        "preregistration_sha256": "a" * 64,
        "source_before": {"production.py": "b" * 64},
        "source_after": {"production.py": "b" * 64},
        "input_before": {"topology": "c" * 64},
        "input_after": {"topology": "c" * 64},
        "fresh_receipts": receipts,
        "rows": rows,
    }
    value["content_sha256"] = validation._content_sha256(value)
    return value


def _reseal_actual(value, evidence_root: Path):
    by_id = {row["row_id"]: row for row in value["rows"]}
    for row_id, descriptor in value["fresh_receipts"].items():
        path = evidence_root / descriptor["relative_path"]
        receipt = validation.json.loads(path.read_text())
        receipt["normalized_row"] = by_id[row_id]
        receipt["normalized_row_sha256"] = validation.hashlib.sha256(
            validation._canonical(by_id[row_id])
        ).hexdigest()
        receipt["content_sha256"] = validation._content_sha256(receipt)
        path.write_bytes(validation._canonical(receipt) + b"\n")
        descriptor["file_sha256"] = validation._sha256_file(path)
        descriptor["content_sha256"] = receipt["content_sha256"]
    value["content_sha256"] = validation._content_sha256(value)
    return value


def test_fixture_is_label_free_and_has_exact_center_major_width_minor_roster(fixture):
    expected = [
        (center, width)
        for center in (0.96, 0.98, 1.0, 1.02, 1.04)
        for width in (0.001, 0.003, 0.01)
    ]

    assert fixture["label_free"] is True
    assert fixture["experimental_labels_read"] is False
    assert [
        (row["center_scale"], row["sigma_e"]) for row in fixture["rows"]
    ] == expected
    assert len({row["row_id"] for row in fixture["rows"]}) == 15


def test_fixture_binds_exact_old_campaign_lineage_and_frozen_oracle(fixture):
    lineage = fixture["lineage"]

    assert lineage["campaign_id"] == "campaign-v2-20261001T110415Z"
    assert len(lineage["terminal_row_file_sha256"]) == 15
    assert lineage["frozen_numpy_scipy_oracle_sha256"] == (
        "e69c73bd7d98c6d1c802ebe9231c847db73c35e3b6b6be1a93ac67cf8c3c5b75"
    )
    assert (
        fixture["lineage_sha256"]
        == validation.hashlib.sha256(validation._canonical(lineage)).hexdigest()
    )


def test_fixture_external_file_hash_is_checked_before_parsing_or_self_seal(
    fixture, tmp_path
):
    identical = tmp_path / "identical.json"
    identical.write_bytes(FIXTURE.read_bytes())
    assert (
        validation.load_fixture(identical)["content_sha256"]
        == fixture["content_sha256"]
    )

    changed = deepcopy(fixture)
    changed["gates"] = {key: 1e300 for key in changed["gates"]}
    changed["content_sha256"] = validation._content_sha256(changed)
    tampered = tmp_path / "tampered.json"
    tampered.write_text(validation.json.dumps(changed, sort_keys=True))
    with pytest.raises(ValueError, match="external SHA256"):
        validation.load_fixture(tampered)


def test_fixture_locks_original_four_two_two_terminal_outcome(fixture):
    old = fixture["frozen_v1_outcome"]

    assert old == {
        "row_count": 15,
        "solvent_independent_pass_count": 4,
        "combined_total_fd_pass_count": 2,
        "both_pass_count": 2,
        "status": "GAUSSIAN_OPT_INCOMPLETE",
    }


def test_v1_replay_exact_fixture_passes_comparator(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)

    result = validation.compare_v1_replay(actual, fixture, evidence_root=tmp_path)

    assert result["passed"] is True
    assert result["status"] == "COMPARATOR_PASSED_ONLY"
    assert result["freshness_verified_by_comparator"] is False
    assert result["freshness_requires_producer_postflight"] is True
    assert result["science_qualified"] is False
    assert result["direct_field_failure_count"] == 0
    assert result["raw_fd_bit_mismatch_count"] == 0


def test_v1_replay_allows_only_enumerated_provenance_drift(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)
    actual["rows"][0]["provenance"] = {
        **actual["rows"][0]["provenance"],
        "worktree": "/new/worktree",
        "timestamp": "new",
    }

    assert (
        validation.compare_v1_replay(
            _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
        )["passed"]
        is True
    )

    actual["rows"][0]["provenance"]["model_identity"] = "changed-model"
    assert (
        validation.compare_v1_replay(
            _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
        )["passed"]
        is False
    )

    actual = _actual_envelope(fixture, tmp_path)
    actual["rows"][0]["provenance"]["unapproved_scientific_flag"] = True
    assert (
        validation.compare_v1_replay(
            _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
        )["passed"]
        is False
    )


def test_v1_direct_native_field_uses_128_epsilon_scaled_tolerance(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)
    baseline = actual["rows"][0]["solvent"]["components_kcal_mol"]["polar"]
    tolerance = validation.direct_field_tolerance(baseline)
    actual["rows"][0]["solvent"]["components_kcal_mol"]["polar"] = (
        baseline + 2 * tolerance
    )

    result = validation.compare_v1_replay(
        _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
    )

    assert result["passed"] is False
    assert result["direct_field_failure_count"] == 1


def test_v1_allowed_direct_drift_may_change_metric_bits_without_changing_verdict(
    fixture, tmp_path
):
    actual = _actual_envelope(fixture, tmp_path)
    baseline = actual["rows"][0]["solvent"]["forces_kcal_mol_per_angstrom"][0][0]
    actual["rows"][0]["solvent"]["forces_kcal_mol_per_angstrom"][0][0] = (
        baseline + 0.5 * validation.direct_field_tolerance(baseline)
    )
    actual["rows"][0]["frozen_v1_result"] = validation.derive_frozen_v1_row(
        actual["rows"][0], fixture
    )

    result = validation.compare_v1_replay(
        _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
    )

    assert result["passed"] is True
    assert result["outcome_metric_deltas"][0]["solvent"] != {
        key: 0.0 for key in fixture["rows"][0]["frozen_v1_result"]["solvent_metrics"]
    }


def test_v1_raw_fd_energy_requires_float64_bit_identity(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)
    original = actual["rows"][0]["solvent"]["raw_fd"][0]["energies"][0]
    actual["rows"][0]["solvent"]["raw_fd"][0]["energies"][0] = float(
        np.nextafter(np.float64(original), np.float64(math.inf))
    )

    result = validation.compare_v1_replay(
        _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
    )

    assert result["passed"] is False
    assert result["raw_fd_bit_mismatch_count"] == 1


def test_v1_replay_requires_raw_derived_four_two_two_outcome(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)
    actual["rows"][0]["frozen_v1_result"]["combined_passed"] = True

    result = validation.compare_v1_replay(
        _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
    )

    assert result["passed"] is False
    assert result["outcome_failure_count"] == 1


def test_v1_replay_rejects_corrupt_raw_fd_metadata(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)
    raw = actual["rows"][0]["solvent"]["raw_fd"][0]
    raw.update(
        atom_index=2,
        axis=2,
        step_angstrom=123.0,
        multipliers=[99.0, 99.0, 99.0, 99.0],
    )

    result = validation.compare_v1_replay(
        _reseal_actual(actual, tmp_path), fixture, evidence_root=tmp_path
    )

    assert result["passed"] is False
    assert result["raw_fd_metadata_failure_count"] == 1


def test_v1_replay_rejects_bare_fixture_rows_without_fresh_envelope(fixture):
    with pytest.raises(ValueError, match="fresh replay envelope"):
        validation.compare_v1_replay(
            deepcopy(fixture["rows"]), fixture, evidence_root=Path(".")
        )


def test_v1_replay_rejects_envelope_reusing_frozen_receipt_hashes(fixture, tmp_path):
    actual = _actual_envelope(fixture, tmp_path)
    frozen_hashes = list(fixture["lineage"]["terminal_row_file_sha256"].values())
    for index, row in enumerate(fixture["rows"]):
        actual["fresh_receipts"][row["row_id"]]["file_sha256"] = frozen_hashes[index]
    actual["content_sha256"] = validation._content_sha256(actual)

    with pytest.raises(ValueError, match="fresh receipt"):
        validation.compare_v1_replay(actual, fixture, evidence_root=tmp_path)


def test_fixture_pins_units_dtype_domain_and_profiles(fixture):
    identity = fixture["scientific_identity"]

    assert identity["dtype"] == "float64"
    assert identity["device"] == "cpu"
    assert identity["energy_units"] == {
        "solvent": "kcal/mol",
        "combined": "hartree",
    }
    assert identity["force_units"] == {
        "solvent": "kcal/mol/angstrom",
        "combined": "hartree/angstrom",
    }
    assert identity["site_count"] == 3
    assert identity["electrostatic_size_strictly_below_angstrom"] == 9.5
    assert identity["v1_profile_id"] == "gaussian-cha-r6-v1"
    assert identity["v2_profile_id"] == (
        "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement"
    )
    assert fixture["gates"] == {
        "energy_max_abs_kcal_mol": 1e-6,
        "force_max_abs_kcal_mol_per_angstrom": 2e-5,
        "force_rms_kcal_mol_per_angstrom": 5e-6,
        "finest_step_disagreement_max_kcal_mol_per_angstrom": 1e-5,
    }
    assert fixture["combined_gates"] == {
        "force_max_abs_hartree_per_angstrom": 3.187202875281032e-8,
        "force_rms_hartree_per_angstrom": 7.96800718820258e-9,
        "finest_step_disagreement_max_hartree_per_angstrom": 1.593601437640516e-8,
    }


def test_local_provenance_rejects_dtype_or_profile_drift():
    provenance = {
        "numerical_profile_id": validation.V2_PROFILE_ID,
        "r6_backend_diagnostics": {
            "profile_id": validation.V2_PROFILE_ID,
            "backend_function": "torch_continuum_r6_derivative_v2._r6_inverse_born_v2",
            "order": 64,
        },
        "execution": {
            "device": "cpu",
            "dtype": "torch.float64",
            "programmatic_only": True,
            "public_provider_registered": False,
        },
        "model_identity": "chagb-r6-pbsa-gaussian-sign-v1",
        "provider": "gaussian-cha-continuum",
        "method": "gb",
        "model": "chagb-r6-pbsa-gaussian-sign-v1",
        "profile": "three-site-water-programmatic-experimental-v1",
        "topology_sha256": "t" * 64,
        "sigma_e": 0.01,
        "quadrature_order": 64,
        "native_force": True,
        "numerical_force": False,
        "derivatives": "torch automatic differentiation of complete scalar",
        "production_admitted": False,
        "accuracy_certified": False,
        "physical_accuracy_claim": False,
        "polar_diagnostics": {
            "electrostatic_size_angstrom": 2.0,
            "component_sum_residual_hartree": 0.0,
        },
        "point_domain": {
            "scope": "three-site-r6-geometric-point-only",
            "segment_certified": False,
        },
    }

    validation._validate_production_provenance(
        provenance,
        validation.V2_PROFILE_ID,
        64,
        topology_sha256="t" * 64,
        sigma_e=0.01,
    )
    provenance["native_force"] = False
    provenance["numerical_force"] = True
    provenance["derivatives"] = "finite differences"
    provenance["topology_sha256"] = "wrong"
    provenance["sigma_e"] = 999
    provenance["quadrature_order"] = 999
    provenance["polar_diagnostics"]["electrostatic_size_angstrom"] = 9.5
    provenance["execution"]["programmatic_only"] = False
    provenance["execution"]["public_provider_registered"] = True
    with pytest.raises(ValueError, match="profile/dtype/domain"):
        validation._validate_production_provenance(
            provenance,
            validation.V2_PROFILE_ID,
            64,
            topology_sha256="t" * 64,
            sigma_e=0.01,
        )


def test_production_component_closure_uses_existing_one_e_minus_nine_kcal_gate():
    class Result:
        components_hartree = {"polar": 0.1, "cavity": 0.2, "dispersion": 0.3}
        energy_hartree = 0.7
        forces_hartree_per_angstrom = np.zeros((3, 3))
        provenance = {}

    class Correction:
        @staticmethod
        def evaluate(atoms, need_forces=False):
            return Result()

    with pytest.raises(ValueError, match="component closure"):
        validation._production_result(Correction(), object(), True)


def test_reference_cache_reuses_only_exact_coordinate_parameter_order():
    calls = []

    class FakeOracle:
        @staticmethod
        def prepare_gaussian_reference_geometry(*args, **kwargs):
            calls.append((np.asarray(args[0]).copy(), kwargs["azimuth_orders"]))
            return object()

    topology = {
        "effective_charges_e": [-0.8, 0.4, 0.4],
        "cha_radii_angstrom": [1.8, 1.0, 1.0],
        "lj_rmin_angstrom": [1.8, 0.3, 0.3],
        "lj_epsilon_kcal_mol": [0.09, 0.004, 0.004],
    }
    cache = validation.ReferenceGeometryCache(FakeOracle(), topology, 1e-12, 1e-12)
    positions = np.zeros((3, 3))

    first = cache.get(positions, 64)
    same_other_width = cache.get(positions.copy(), 64)
    other_order = cache.get(positions, 96)
    moved = positions.copy()
    moved[0, 0] = 1e-12
    moved_result = cache.get(moved, 64)

    assert first is same_other_width
    assert other_order is not first
    assert moved_result is not first
    assert len(calls) == 3


@pytest.mark.parametrize("production_order", [64, 96])
def test_every_displaced_independent_fd_uses_fixed_order128(
    fixture, monkeypatch, production_order
):
    calls = []

    class Level:
        azimuth_order = 128
        inverse_cube_quad_error_estimate_per_angstrom3 = 0.0
        meridian_evaluations = 1

    class Geometry:
        coordinates_sha256 = "a" * 64
        parameters_sha256 = "b" * 64
        identity_sha256 = "c" * 64
        payload_sha256 = "d" * 64
        r6_levels = (Level(),)
        r6_diagnostics = {"nonrigorous": True}
        diagnostics = {"error_estimates_are_rigorous_bounds": False}

    class Cache:
        tracer = None

        @staticmethod
        def get(positions, order):
            calls.append((np.asarray(positions).copy(), order))
            return Geometry()

    class Reference:
        energies_kcal_mol = {
            "polar": 0.0,
            "cavity": 0.0,
            "dispersion": 0.0,
            "total": 0.0,
        }
        total_kcal_mol = 0.0

    monkeypatch.setattr(
        validation,
        "_production_result",
        lambda correction, atoms, need_forces: (
            {"polar": 0.0, "cavity": 0.0, "dispersion": 0.0, "total": 0.0},
            np.zeros((3, 3)) if need_forces else None,
            {},
        ),
    )
    monkeypatch.setattr(
        validation, "_validate_production_provenance", lambda *a, **k: None
    )
    monkeypatch.setattr(validation, "_oracle_energy", lambda *a, **k: Reference())

    result = validation._evaluate_local_order(
        fixture["rows"][0],
        fixture["topology"],
        production_order,
        validation.V2_PROFILE_ID,
        object(),
        type(
            "CorrectionModule",
            (),
            {"GaussianChaCorrection": lambda *a, **k: object()},
        ),
        object(),
        Cache(),
        fixture["gates"],
    )

    assert calls[0][1] == production_order
    assert {order for _, order in calls[1:]} == {validation.REFERENCE_FD_ORDER}
    assert result["production_order"] == production_order
    assert result["reference_fd_order"] == 128


def test_preregistration_binds_production_and_reference_fd_orders(fixture):
    preregistration = validation.build_local_preregistration(
        fixture, {"center-00-sigma-02"}, validation.V2_PROFILE_ID
    )

    assert [
        (cell["production_order"], cell["reference_fd_order"])
        for cell in preregistration["cells"]
    ] == [
        (64, 128),
        (96, 128),
        (128, 128),
    ]


def test_local_metrics_retain_three_force_comparisons_separately():
    analytic = np.arange(9, dtype=float).reshape(3, 3) * 1e-5
    own_fd = analytic + 1e-7
    oracle_fd = analytic - 2e-7

    result = validation.force_comparison_metrics(analytic, own_fd, oracle_fd)

    assert set(result) == {"ad_own_fd", "production_fd_oracle_fd", "ad_oracle_fd"}
    assert result["ad_own_fd"]["maximum_absolute_error"] == pytest.approx(1e-7)
    assert result["production_fd_oracle_fd"]["maximum_absolute_error"] == pytest.approx(
        3e-7
    )
    assert result["ad_oracle_fd"]["maximum_absolute_error"] == pytest.approx(2e-7)


def test_v2_summary_requires_every_row_and_each_order():
    rows = []
    for center in range(5):
        for width in range(3):
            rows.append(
                {
                    "row_id": f"center-{center:02d}-sigma-{width:02d}",
                    "orders": {str(order): {"passed": True} for order in (64, 96, 128)},
                }
            )

    passed = validation.summarize_v2_local(rows)
    rows[4]["orders"]["64"]["passed"] = False
    failed = validation.summarize_v2_local(rows)

    assert passed["status"] == "LOCAL_MATRIX_PASSED_ONLY"
    assert passed["full_campaign_performed"] is False
    assert passed["science_qualified"] is False
    assert failed["status"] == "GAUSSIAN_OPT_INCOMPLETE"
    assert failed["order_pass_counts"] == {"64": 14, "96": 15, "128": 15}


def test_bounded_summary_fails_when_any_selected_order_fails():
    rows = [
        {
            "row_id": "center-00-sigma-02",
            "orders": {
                "64": {"passed": True},
                "96": {"passed": False},
                "128": {"passed": True},
            },
        }
    ]

    summary = validation.summarize_bounded_local(
        rows, ["center-00-sigma-02"], sources_stable=True
    )

    assert summary["status"] == "GAUSSIAN_OPT_INCOMPLETE"
    assert summary["all_passed"] is False
    assert validation.local_run_succeeded(summary) is False
    assert summary["full_campaign_performed"] is False
    assert summary["science_qualified"] is False


def test_run_local_cli_returns_nonzero_for_bounded_failed_summary(
    fixture, tmp_path, monkeypatch
):
    monkeypatch.setattr(validation, "load_fixture", lambda path: fixture)
    monkeypatch.setattr(
        validation,
        "run_local_campaign",
        lambda *args, **kwargs: {
            "status": "GAUSSIAN_OPT_INCOMPLETE",
            "all_passed": False,
            "full_campaign_performed": False,
            "science_qualified": False,
        },
    )

    code = validation.main(
        [
            "run-local",
            "--profile",
            validation.V2_PROFILE_ID,
            "--output",
            str(tmp_path / "campaign"),
            "--expected-preregistration-sha256",
            "a" * 64,
        ]
    )

    assert code == 1


def test_local_row_failure_is_retained_for_every_selected_order(fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("retained failure")

    monkeypatch.setattr(validation, "_evaluate_local_order", fail)
    result = validation.run_v2_local(
        fixture,
        profile_id=validation.V2_PROFILE_ID,
        row_ids={"center-00-sigma-02"},
        runtime_context=validation.testing_runtime_context(),
    )

    orders = result["rows"][0]["orders"]
    assert set(orders) == {"64", "96", "128"}
    assert all(record["status"] == "FAILED" for record in orders.values())
    assert result["summary"]["status"] == "GAUSSIAN_OPT_INCOMPLETE"


def test_local_campaign_writes_three_atomic_failed_cell_receipts(
    fixture, tmp_path, monkeypatch
):
    output = tmp_path / "bounded"
    validation.preregister_local(
        fixture,
        output,
        {"center-00-sigma-02"},
        validation.V2_PROFILE_ID,
    )
    prereg_sha = validation._sha256_file(output / "preregistration.json")

    def fail(*args, **kwargs):
        raise RuntimeError("retained cell failure")

    monkeypatch.setattr(validation, "_evaluate_local_order", fail)
    summary = validation.run_local_campaign(
        fixture,
        output,
        prereg_sha,
        profile_id=validation.V2_PROFILE_ID,
    )

    receipts = sorted((output / "cells").glob("*.json"))
    assert len(receipts) == 3
    assert all(
        validation.json.loads(path.read_text())["result"]["status"] == "FAILED"
        for path in receipts
    )
    assert summary["status"] == "GAUSSIAN_OPT_INCOMPLETE"
    assert summary["content_sha256"] == validation._content_sha256(summary)


def test_local_preregistration_binds_45_cells_sources_and_oracle_closure(fixture):
    preregistration = validation.build_local_preregistration(fixture)

    assert len(preregistration["cells"]) == 45
    assert preregistration["fixture_file_sha256"] == validation.FROZEN_FIXTURE_SHA256
    assert preregistration["oracle_closure_sha256"] == {
        "docs/implicit-solvation/benchmarks/cha_gaussian_reference.py": (
            "e69c73bd7d98c6d1c802ebe9231c847db73c35e3b6b6be1a93ac67cf8c3c5b75"
        ),
        "docs/implicit-solvation/benchmarks/cha_continuum_reference.py": (
            "826091b814df9a68ac8d608be83a7381ed2ac7bc7fe838cb97667c3e097fee07"
        ),
        "maple/function/calculator/extra_correction/implicit/sphere_union_volume.py": (
            "f5631cf299df7d20ae7ec8e092d12b964213f9c6e0af093553d6e2cbc13267ed"
        ),
        "maple/function/calculator/extra_correction/implicit/sphere_union_dispersion.py": (
            "d2c1acbc98b515471d39d1fb720a940d6012ecba27c261d4064a7d09d7b62a3f"
        ),
    }
    assert preregistration["content_sha256"] == validation._content_sha256(
        preregistration
    )


def test_local_preregistration_rejects_empty_bounded_roster(fixture):
    with pytest.raises(ValueError, match="nonempty"):
        validation.build_local_preregistration(fixture, set(), validation.V2_PROFILE_ID)

    summary = validation.summarize_bounded_local([], [], sources_stable=True)
    assert summary["all_passed"] is False
    assert summary["status"] == "GAUSSIAN_OPT_INCOMPLETE"


def test_local_preregistration_external_pin_and_post_source_stability(
    fixture, tmp_path
):
    preregistration = validation.build_local_preregistration(fixture)
    path = tmp_path / "preregistration.json"
    path.write_bytes(validation._canonical(preregistration) + b"\n")
    expected = validation._sha256_file(path)

    loaded = validation.load_local_preregistration(path, expected)
    validation.assert_local_sources_stable(loaded, loaded["source_before"])
    changed = dict(loaded["source_before"])
    first = next(iter(changed))
    changed[first] = "0" * 64
    with pytest.raises(ValueError, match="source manifest changed"):
        validation.assert_local_sources_stable(loaded, changed)
    with pytest.raises(ValueError, match="external SHA256"):
        validation.load_local_preregistration(path, "0" * 64)

    empty = deepcopy(preregistration)
    empty["bounded"] = True
    empty["cells"] = []
    empty["content_sha256"] = validation._content_sha256(empty)
    empty_path = tmp_path / "empty.json"
    empty_path.write_bytes(validation._canonical(empty) + b"\n")
    with pytest.raises(ValueError, match="cell inventory"):
        validation.load_local_preregistration(
            empty_path, validation._sha256_file(empty_path)
        )


def test_oracle_call_tracer_rejects_v2_formula_frame():
    tracer = validation.OracleCallTracer()

    def forbidden():
        return 1

    forbidden.__code__ = forbidden.__code__.replace(
        co_filename="/tmp/torch_continuum_r6_contact_v2.py"
    )
    with pytest.raises(RuntimeError, match="forbidden production frame"):
        tracer.call(forbidden)
    assert tracer.report()["prohibited_call_count"] == 1


def test_oracle_call_tracer_resolves_each_unique_filename_once(monkeypatch):
    original = Path.resolve
    resolutions = 0

    def counting_resolve(self, *args, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)
    tracer = validation.OracleCallTracer()

    def inner():
        return 1

    def outer():
        return inner() + inner()

    assert tracer.call(outer) == 2
    first_calls = tracer.call_count
    first_resolutions = resolutions
    assert tracer.call(outer) == 2

    assert tracer.call_count == 2 * first_calls
    assert resolutions == first_resolutions
    assert tracer.report()["filename_resolution_count"] == first_resolutions
    assert tracer.report()["filename_cache_entries"] == first_resolutions


def test_frozen_oracle_source_has_no_v2_classifier_or_contact_import():
    oracle_path = BENCHMARK_DIR / "cha_gaussian_reference.py"
    assert validation._sha256_file(oracle_path) == validation.FROZEN_ORACLE_SHA256
    tree = ast.parse(oracle_path.read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert not any("contact_v2" in name for name in imports)
    assert not any("gaussian_cha_profiles" in name for name in imports)
    assert not any("torch_continuum_r6" in name for name in imports)
