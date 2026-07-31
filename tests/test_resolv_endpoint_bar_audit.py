from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/pretrained-solvation-hub/run_resolv_endpoint_bar_audit.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("resolv_endpoint_bar_audit", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(index: int, group: str, prediction: float, experiment: float) -> dict:
    return {
        "k_index": index,
        "trajectory_id": index + 1,
        "mobley_id": f"mobley_{index:07d}",
        "smiles": "C",
        "groups": [group, "organic compound"],
        "primary_functional_group": group,
        "prediction_kcal_mol": prediction,
        "experimental_kcal_mol": experiment,
        "experimental_uncertainty_kcal_mol": 0.1,
    }


def test_runner_source_never_generates_a_fresh_conformer() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = ("EmbedMolecule(", "EmbedMultipleConfs(", "ETKDG(", "ETKDGv3(")
    assert all(token not in source for token in forbidden)


def test_aggregate_results_emits_record_errors_and_complete_group_labels() -> None:
    runner = _load_runner_module()
    records = [
        _record(0, "primary alcohol", -3.0, -2.5),
        _record(1, "ketone", -1.0, -2.0),
    ]
    result = runner.aggregate_results(records, minimum_primary_groups=10)
    assert result["records"][0]["signed_error_kcal_mol"] == pytest.approx(-0.5)
    assert result["records"][0]["absolute_error_kcal_mol"] == pytest.approx(0.5)
    assert result["records"][0]["groups"] == [
        "primary alcohol",
        "organic compound",
    ]
    assert result["records"][1]["signed_error_kcal_mol"] == pytest.approx(1.0)


def test_aggregate_results_recomputes_pooled_and_per_group_metrics() -> None:
    runner = _load_runner_module()
    records = [
        _record(0, "primary alcohol", -3.0, -2.5),
        _record(1, "primary alcohol", -1.0, -2.0),
        _record(2, "ketone", 0.0, -2.0),
    ]
    result = runner.aggregate_results(records, minimum_primary_groups=2)
    pooled = result["metrics"]["pooled"]
    assert pooled["n"] == 3
    assert pooled["mae_kcal_mol"] == pytest.approx(3.5 / 3.0)
    assert pooled["rmse_kcal_mol"] == pytest.approx(math.sqrt(5.25 / 3.0))
    assert pooled["maxae_kcal_mol"] == pytest.approx(2.0)
    by_group = result["metrics"]["by_primary_functional_group"]
    assert by_group["primary alcohol"] == {
        "n": 2,
        "mae_kcal_mol": pytest.approx(0.75),
        "rmse_kcal_mol": pytest.approx(math.sqrt(0.625)),
        "maxae_kcal_mol": pytest.approx(1.0),
    }
    assert by_group["ketone"]["n"] == 1
    assert by_group["ketone"]["maxae_kcal_mol"] == pytest.approx(2.0)


def test_fewer_than_ten_primary_groups_is_not_accuracy_claim_eligible() -> None:
    runner = _load_runner_module()
    records = [_record(index, f"group-{index}", 0.0, 0.0) for index in range(9)]
    result = runner.aggregate_results(records, minimum_primary_groups=10)
    assert result["scientific_accuracy_eligible"] is False
    assert (
        "classified_primary_functional_groups_below_10"
        in result["eligibility_failures"]
    )


def test_ten_primary_groups_satisfies_only_the_group_count_gate() -> None:
    runner = _load_runner_module()
    records = [_record(index, f"group-{index}", 0.0, 0.0) for index in range(10)]
    result = runner.aggregate_results(records, minimum_primary_groups=10)
    assert result["classified_primary_functional_group_count"] == 10
    assert (
        "classified_primary_functional_groups_below_10"
        not in result["eligibility_failures"]
    )


def test_unclassified_records_do_not_count_as_functional_groups() -> None:
    runner = _load_runner_module()
    records = [
        _record(index, f"group-{index}", 0.0, 0.0)
        for index in range(26)
    ]
    records.extend(
        _record(100 + index, "unclassified", 0.0, 0.0)
        for index in range(9)
    )

    result = runner.aggregate_results(records, minimum_primary_groups=10)

    assert result["classified_primary_functional_group_count"] == 26
    assert result["primary_functional_group_count"] == 26
    assert result["primary_functional_group_label_count_including_unclassified"] == 27
    assert result["unclassified_record_count"] == 9
    assert runner.UNCLASSIFIED_NON_GROUP_BUCKET in (
        result["metrics"]["by_primary_functional_group"]
    )


def _bar_diagnostics(vacuum: list[float], water: list[float]) -> dict:
    return {
        "dimensionless_work": {
            "vacuum_ensemble": vacuum,
            "water_ensemble": water,
        }
    }


def test_equal_endpoint_bar_identical_endpoints_has_exact_root() -> None:
    runner = _load_runner_module()
    root = 2.5
    result = runner.diagnose_equal_endpoint_bar(
        _bar_diagnostics([root] * 40, [root] * 40),
        upstream_delta_g_kcal_mol=runner.KBT_KCAL_MOL * root,
    )

    assert result["bar_root_dimensionless"] == root
    assert result["bar_root_residual"] == 0.0
    assert result["bar_overlap_omega"] == 1.0
    assert result["bar_vacuum_direction_kish_ess"] == pytest.approx(40.0)
    assert result["bar_water_direction_kish_ess"] == pytest.approx(40.0)
    assert result["bar_conditional_iid_se_kcal_mol"] == 0.0
    assert result["statistical_support_eligible"] is True
    assert result["bar_uncertainty_inferential"] is False
    assert result["bar_confidence_interval_available"] is False
    assert result["bar_bootstrap_performed"] is False
    assert "dimensionless_work" not in result


def test_equal_endpoint_bar_extreme_values_are_overflow_stable() -> None:
    runner = _load_runner_module()
    result = runner.diagnose_equal_endpoint_bar(
        _bar_diagnostics([1000.0] * 40, [-1000.0] * 40),
        upstream_delta_g_kcal_mol=0.0,
    )
    assert result["bar_root_dimensionless"] == 0.0
    assert result["bar_root_residual"] == 0.0
    assert result["bar_overlap_omega"] == 0.0
    assert result["bar_conditional_iid_se_kcal_mol"] is None
    assert result["statistical_support_eligible"] is False


def test_low_overlap_fails_support_even_with_high_directional_kish_ess() -> None:
    runner = _load_runner_module()
    result = runner.diagnose_equal_endpoint_bar(
        _bar_diagnostics([100.0] * 40, [-100.0] * 40),
        upstream_delta_g_kcal_mol=0.0,
    )
    assert result["bar_min_directional_kish_ess"] == pytest.approx(40.0)
    assert result["bar_overlap_omega"] < 0.25
    assert result["bar_overlap_support_pass"] is False
    assert result["statistical_support_eligible"] is False


def test_aggregate_reports_low_overlap_support_failures() -> None:
    runner = _load_runner_module()
    supported = runner.diagnose_equal_endpoint_bar(
        _bar_diagnostics([0.0] * 40, [0.0] * 40),
        upstream_delta_g_kcal_mol=0.0,
    )
    unsupported = runner.diagnose_equal_endpoint_bar(
        _bar_diagnostics([100.0] * 40, [-100.0] * 40),
        upstream_delta_g_kcal_mol=0.0,
    )
    records = [
        {**_record(0, "alcohol", 0.0, 0.0), **supported},
        {**_record(1, "ketone", 0.0, 0.0), **unsupported},
    ]

    result = runner.aggregate_results(
        records,
        minimum_primary_groups=2,
        require_bar_diagnostics=True,
    )

    support = result["statistical_support"]
    assert result["statistical_support_eligible"] is False
    assert support["support_failure_count"] == 1
    assert support["minimum_overlap_omega"] < 0.25
    assert support["median_overlap_omega"] == pytest.approx(
        (
            supported["bar_overlap_omega"]
            + unsupported["bar_overlap_omega"]
        )
        / 2.0
    )
    assert support["minimum_directional_kish_ess"] == pytest.approx(40.0)
    assert support["maximum_conditional_iid_se_kcal_mol"] is not None
    assert support["inferential_uncertainty"] is False
    assert support["confidence_interval_available"] is False
    assert support["bootstrap_performed"] is False
    assert "bar_statistical_support_failed" in result["eligibility_failures"]


def test_merge_keeps_raw_bar_arrays_out_of_canonical_record() -> None:
    runner = _load_runner_module()
    source = {
        "k_index": 0,
        "trajectory_id": 1,
        "mobley_id": "mobley_0000001",
    }
    worker = {
        "k_index": 0,
        "delta_g_kcal_mol": 0.0,
        "bar_diagnostics": _bar_diagnostics([0.0] * 40, [0.0] * 40),
    }

    merged = runner._merge_prediction_records([source], [worker])

    assert len(merged) == 1
    assert "bar_diagnostics" not in merged[0]
    assert "dimensionless_work" not in merged[0]
    assert merged[0]["bar_overlap_omega"] == 1.0


@pytest.mark.parametrize(
    ("vacuum", "water"),
    [
        ([0.0] * 39, [0.0] * 40),
        ([math.nan] + [0.0] * 39, [0.0] * 40),
        ([math.inf] + [0.0] * 39, [0.0] * 40),
        ([-math.inf] + [0.0] * 39, [0.0] * 40),
    ],
)
def test_equal_endpoint_bar_rejects_bad_work_arrays(
    vacuum: list[float],
    water: list[float],
) -> None:
    runner = _load_runner_module()
    with pytest.raises(runner.ReSolvAuditError):
        runner.diagnose_equal_endpoint_bar(
            _bar_diagnostics(vacuum, water),
            upstream_delta_g_kcal_mol=0.0,
        )


def test_equal_endpoint_bar_rejects_upstream_root_mismatch() -> None:
    runner = _load_runner_module()
    with pytest.raises(runner.ReSolvAuditError, match="disagrees"):
        runner.diagnose_equal_endpoint_bar(
            _bar_diagnostics([0.0] * 40, [0.0] * 40),
            upstream_delta_g_kcal_mol=1.0e-6,
        )


def test_cpu_only_jax_runtime_reports_gpu_as_unavailable() -> None:
    runner = _load_runner_module()
    runtime = runner.detect_jax_runtime(
        device_records=[{"platform": "cpu", "device_kind": "cpu", "id": 0}],
        x64_enabled=True,
    )
    assert runtime["gpu_status"] == "unavailable"
    assert runtime["gpu_results_present"] is False
    assert runtime["x64_enabled"] is True
    assert runtime["gpu_unavailable_reason"]


def test_requested_platform_and_x64_are_fail_closed() -> None:
    runner = _load_runner_module()
    with pytest.raises(runner.ReSolvAuditError, match="x64"):
        runner._validate_requested_runtime(
            requested_platform="cpu",
            runtime_probe={
                "x64_enabled": False,
                "devices": [{"platform": "cpu"}],
            },
        )
    with pytest.raises(runner.ReSolvAuditError, match="does not match"):
        runner._validate_requested_runtime(
            requested_platform="gpu",
            runtime_probe={
                "x64_enabled": True,
                "devices": [{"platform": "cpu"}],
            },
        )
    with pytest.raises(runner.ReSolvAuditError, match="worker backend"):
        runner._validate_requested_runtime(
            requested_platform="gpu",
            runtime_probe={
                "x64_enabled": True,
                "devices": [{"platform": "gpu"}],
            },
            worker_response={
                "backend": "cpu",
                "dtype": {"jax_enable_x64": True},
            },
        )


def test_repeat_predictions_are_compared_by_immutable_identity_not_position() -> None:
    runner = _load_runner_module()
    manifest = [
        {
            "k_index": 3,
            "trajectory_id": 4,
            "mobley_id": "mobley_a",
        },
        {
            "k_index": 8,
            "trajectory_id": 9,
            "mobley_id": "mobley_b",
        },
    ]
    first = [
        {"k_index": 3, "delta_g_kcal_mol": 1.25},
        {"k_index": 8, "delta_g_kcal_mol": -2.5},
    ]
    shuffled = list(reversed(first))
    assert runner._predictions_by_identity(
        manifest,
        first,
    ) == runner._predictions_by_identity(manifest, shuffled)


def test_repeat_prediction_identity_rejects_duplicates() -> None:
    runner = _load_runner_module()
    manifest = [
        {
            "k_index": 3,
            "trajectory_id": 4,
            "mobley_id": "mobley_a",
        }
    ]
    with pytest.raises(runner.ReSolvAuditError, match="duplicate"):
        runner._predictions_by_identity(
            manifest,
            [
                {"k_index": 3, "delta_g_kcal_mol": 1.25},
                {"k_index": 3, "delta_g_kcal_mol": 1.25},
            ],
        )


def test_audit_binds_runner_worker_and_protocol_source_hashes() -> None:
    runner = _load_runner_module()
    assert runner.RUNNER_PATH == RUNNER
    assert runner.WORKER_PATH.is_file()
    assert runner.PROTOCOL_PATH.is_file()
    assert runner._code_sha256() == {
        "audit_runner": _sha256(runner.RUNNER_PATH),
        "worker": _sha256(runner.WORKER_PATH),
        "protocol": _sha256(runner.PROTOCOL_PATH),
    }


def test_worker_batches_execute_immutable_snapshot_not_live_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    live_worker = tmp_path / "live_resolv_worker.py"
    worker_snapshot = tmp_path / "snapshot_resolv_worker.py"
    live_worker.write_bytes(b"initial live worker")
    worker_snapshot.write_bytes(b"immutable admitted worker")
    monkeypatch.setattr(runner, "WORKER_PATH", live_worker)
    executed_paths: list[Path] = []

    def fake_run_worker(**kwargs):
        executed_paths.append(Path(kwargs["worker_path"]))
        live_worker.write_bytes(b"mutated between batches")
        return (
            {
                "source_revision": runner.RESOLV_UPSTREAM_REVISION,
                "backend": "cpu",
                "devices": ["TFRT_CPU_0"],
                "dtype": {"jax_enable_x64": True},
                "source_snapshot": {
                    "revision": runner.RESOLV_UPSTREAM_REVISION,
                    "archive_sha256": "a" * 64,
                    "exported_roots": ["chemtrain", "jax_sgmc", "util"],
                    "source": "git-archive",
                },
                "platform_attestation": {"default_backend": "cpu"},
                "runtime_environment": {"jax": "0.4.23"},
                "records": [
                    {
                        "k_index": kwargs["request"]["records"][0]["k_index"],
                        "delta_g_kcal_mol": 0.0,
                    }
                ],
                "runtime": {},
            },
            0.0,
        )

    monkeypatch.setattr(runner, "_run_worker", fake_run_worker)
    response, _ = runner._run_worker_batches(
        worker_path=worker_snapshot,
        python_executable=Path("/unused/python"),
        base_request={},
        batches=[
            [{"k_index": 1}],
            [{"k_index": 2}],
        ],
        platform="cpu",
        timeout_seconds=1.0,
    )

    assert executed_paths == [worker_snapshot, worker_snapshot]
    assert worker_snapshot.read_bytes() == b"immutable admitted worker"
    assert live_worker.read_bytes() == b"mutated between batches"
    assert [record["k_index"] for record in response["records"]] == [1, 2]
    assert response["source_snapshot"]["archive_sha256"] == "a" * 64


def test_runner_declares_dedicated_bar_semantics_not_calculator_or_continuum() -> None:
    runner = _load_runner_module()
    semantics = runner.PROTOCOL_SEMANTICS
    assert semantics["protocol_kind"] == "dedicated_hydration_free_energy"
    assert semantics["estimator"] == "BAR"
    assert semantics["ordinary_calculator"] is False
    assert semantics["additive_continuum"] is False
    assert semantics["fresh_conformer_generation"] is False


def test_shape_bounded_batches_limit_distinct_jax_shapes_per_worker() -> None:
    runner = _load_runner_module()
    records = [
        {"k_index": 0, "smiles": "C"},
        {"k_index": 1, "smiles": "CC"},
        {"k_index": 2, "smiles": "CCC"},
        {"k_index": 3, "smiles": "CCCC"},
        {"k_index": 4, "smiles": "CCCCC"},
    ]
    atom_counts = {0: 5, 1: 8, 2: 11, 3: 14, 4: 17}
    batches = runner._shape_bounded_batches(
        records,
        atom_counts=atom_counts,
        max_shapes_per_worker=2,
    )
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert all(
        len({atom_counts[int(record["k_index"])] for record in batch}) <= 2
        for batch in batches
    )


def test_portable_manifest_removes_host_specific_trajectory_roots() -> None:
    runner = _load_runner_module()
    record = {
        "k_index": 271,
        "vacuum_trajectory": "/host/cache/250_50ps_load_traj_mol_272_AC",
        "water_trajectory": "/host/cache/250_50ps_load_wat_traj_mol_272_AC",
    }
    portable = runner._portable_manifest([record])
    assert portable[0]["vacuum_trajectory"] == (
        "250_50ps_load_traj_mol_272_AC"
    )
    assert portable[0]["water_trajectory"] == (
        "250_50ps_load_wat_traj_mol_272_AC"
    )


def test_full_test_manifest_payload_fails_closed_on_identity_drift() -> None:
    runner = _load_runner_module()
    fake_records = [{"k_index": index} for index in range(162)]
    with pytest.raises(runner.ReSolvAuditError, match="manifest identity drifted"):
        runner._portable_manifest_payload(
            fake_records,
            require_frozen_full_test_identity=True,
        )


def test_repeat_comparison_uses_literal_binary64_bits() -> None:
    runner = _load_runner_module()
    identity = (1, 2, "mobley_0000001")
    positive_zero = runner._prediction_bits_by_identity({identity: 0.0})
    negative_zero = runner._prediction_bits_by_identity({identity: -0.0})
    assert positive_zero != negative_zero


def test_code_snapshot_rejects_source_changed_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner_module()
    original_read_bytes = Path.read_bytes

    def changed_runner(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path.resolve() == runner.RUNNER_PATH:
            return payload + b"\nPOST_IMPORT_RACE_SENTINEL = True\n"
        return payload

    monkeypatch.setattr(Path, "read_bytes", changed_runner)
    with pytest.raises(runner.ReSolvAuditError, match="runner source changed"):
        runner._read_code_snapshot()


def test_code_snapshot_accepts_the_unmodified_executing_sources() -> None:
    runner = _load_runner_module()
    source_hashes, worker_bytes = runner._read_code_snapshot()
    assert source_hashes == runner._code_sha256()
    assert hashlib.sha256(worker_bytes).hexdigest() == source_hashes["worker"]


def test_code_snapshot_accepts_the_pinned_python_runtime() -> None:
    python_executable = (
        Path.home() / "miniconda3/envs/maple-resolv/bin/python"
    )
    if not python_executable.is_file():
        pytest.skip("pinned ReSolv CPU environment is not installed")
    script = (
        "import importlib.util\n"
        "from pathlib import Path\n"
        f"path = Path({str(RUNNER)!r}).resolve()\n"
        "spec = importlib.util.spec_from_file_location('snapshot_probe', path)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module._read_code_snapshot()\n"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_jax_probe_runs_in_the_real_pinned_cpu_subprocess() -> None:
    runner = _load_runner_module()
    python_executable = (
        Path.home() / "miniconda3/envs/maple-resolv/bin/python"
    )
    if not python_executable.is_file():
        pytest.skip("pinned ReSolv CPU environment is not installed")
    receipt = runner._probe_jax(python_executable, platform="cpu")
    assert receipt["x64_enabled"] is True
    assert receipt["gpu_status"] == "unavailable"
    devices = receipt["devices"]
    assert isinstance(devices, list) and devices
    assert all(device["platform"] == "cpu" for device in devices)
    assert all(isinstance(device["id"], int) for device in devices)
