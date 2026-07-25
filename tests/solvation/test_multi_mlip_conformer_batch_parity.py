from __future__ import annotations

import json
from pathlib import Path
import sys

from ase import Atoms
import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_conformer_batch_parity_protocol.json"
SOURCE_PROTOCOL_PATH = BENCHMARK_DIR / "multi_mlip_discrete_conformer_protocol.json"
SOURCE_MANIFEST_PATH = (
    BENCHMARK_DIR / "multi_mlip_discrete_conformer_source_manifest.json"
)
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-multi-mlip-conformer-batch-parity-2026-07-25.json"
)
V2_REVIEW_ARTIFACT_PATH = (
    BENCHMARK_DIR
    / "route1-multi-mlip-conformer-batch-parity-v2-review-run-2026-07-25.json"
)
RUNNER_PATH = BENCHMARK_DIR / "run_multi_mlip_conformer_batch_parity.py"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_multi_mlip_conformer_batch_parity as runner


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_batch_parity_protocol_records_review_amendment():
    protocol, fingerprint = runner.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["protocol_id"].endswith("-v6")
    assert protocol["source_partition"] == "development"
    assert protocol["execution"]["batch_size"] == 16
    assert protocol["execution"]["repeat_count"] == 4
    assert protocol["execution"]["device"] == "cuda:0"
    assert protocol["execution"]["require_gpu_endpoint_snapshots"] is True
    assert protocol["execution"]["maximum_preflight_load_per_logical_cpu"] == 0.25
    assert protocol["gates"] == {
        "maximum_absolute_energy_difference_hartree": 1.5936014376405156e-06,
        "maximum_relative_energy_difference_kcal_mol": 0.001,
        "maximum_delta_g_difference_kcal_mol": 0.001,
        "minimum_paired_repeat_speedup": 1.25,
    }
    assert protocol["claim_boundary"]["faster_than_bare_mm_claim"] is False
    assert protocol["claim_boundary"]["solvent_accuracy_promotion"] is False
    assert protocol["route"]["gas_phase_mm_energy"] is False
    assert protocol["route"]["hydration_label_residual"] is False
    assert protocol["route"]["mlip_retraining"] is False
    assert (
        "review-amended confirmation" in protocol["threshold_basis"]["pilot_disclosure"]
    )
    assert "174 s" in protocol["threshold_basis"]["pilot_disclosure"]
    assert (
        "endpoint snapshots cannot prove whole-run GPU exclusivity"
        in protocol["threshold_basis"]["pilot_disclosure"]
    )


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("batch_vs_serial_same_mlip_only", False),
        ("hydration_free_energy_claim", True),
        ("public_solvfe_eligible", True),
    ],
)
def test_batch_parity_protocol_rejects_unsafe_claim_mutations(
    monkeypatch,
    field,
    unsafe_value,
):
    mutated = _load(PROTOCOL_PATH)
    mutated["claim_boundary"][field] = unsafe_value
    original_load_json = core.load_json

    def load_json(path):
        if Path(path).resolve() == PROTOCOL_PATH.resolve():
            return mutated
        return original_load_json(path)

    monkeypatch.setattr(runner, "load_json", load_json)
    with pytest.raises(ValueError, match="claim boundary"):
        runner.load_protocol(PROTOCOL_PATH)


def test_batch_parity_protocol_rejects_unbalanced_execution_order(monkeypatch):
    mutated = _load(PROTOCOL_PATH)
    mutated["execution"]["balanced_order"] = [
        ["serial", "batch"],
        ["serial", "batch"],
        ["serial", "batch"],
        ["serial", "batch"],
    ]
    original_load_json = core.load_json

    def load_json(path):
        if Path(path).resolve() == PROTOCOL_PATH.resolve():
            return mutated
        return original_load_json(path)

    monkeypatch.setattr(runner, "load_json", load_json)
    with pytest.raises(ValueError, match="equally often"):
        runner.load_protocol(PROTOCOL_PATH)


def test_gpu_process_parser_and_timing_preflight_fail_closed(monkeypatch):
    assert runner._parse_gpu_compute_processes("") == []
    assert runner._parse_gpu_compute_processes("1234, python\n5678, [Not Found]\n") == [
        {"pid": 1234, "process_name": "python"},
        {"pid": 5678, "process_name": "[Not Found]"},
    ]
    with pytest.raises(RuntimeError, match="process row"):
        runner._parse_gpu_compute_processes("invalid")
    with pytest.raises(RuntimeError, match="process PID"):
        runner._parse_gpu_compute_processes("not-a-pid, python")

    completed = type(
        "Completed",
        (),
        {"stdout": f"{runner.os.getpid()}, python\n999999, other-job\n"},
    )()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: completed,
    )
    with pytest.raises(RuntimeError, match="competing compute processes"):
        runner._timing_preflight(0.25)

    quiet_gpu = type("Completed", (), {"stdout": ""})()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: quiet_gpu,
    )
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (4.0, 3.0, 2.0))
    snapshot = runner._timing_preflight(0.25)
    assert snapshot["gpu_preflight"]["passed"] is True
    assert snapshot["host_preflight"]["load_per_logical_cpu"] == 0.2

    monkeypatch.setattr(runner.os, "getloadavg", lambda: (6.0, 3.0, 2.0))
    with pytest.raises(RuntimeError, match="excessive host load"):
        runner._timing_preflight(0.25)


def test_gpu_postflight_failure_prevents_artifact_write(monkeypatch, tmp_path):
    artifact = {"timing_isolation": {"endpoint_snapshots_passed": None}}
    write_called = False

    def fail_postflight(_stage):
        raise RuntimeError("postflight competition")

    def record_write(_path, _artifact):
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(runner, "_exclusive_gpu_snapshot", fail_postflight)
    monkeypatch.setattr(runner, "write_json_atomic", record_write)

    with pytest.raises(RuntimeError, match="postflight competition"):
        runner._seal_and_write_after_gpu_postflight(
            tmp_path / "must-not-exist.json",
            artifact,
        )

    assert write_called is False
    assert not (tmp_path / "must-not-exist.json").exists()


def test_warmup_selection_spans_cases_and_returns_exact_count():
    cases = [
        {
            "template": Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            "positions": np.asarray([[[0.1, 0.0, 0.0]]]),
        },
        {
            "template": Atoms("He", positions=[[0.0, 0.0, 0.0]]),
            "positions": np.asarray([[[0.2, 0.0, 0.0]]]),
        },
    ]

    warmup = runner._select_warmup_structures(cases, 2)

    assert [atoms.symbols[0] for atoms in warmup] == ["H", "He"]
    np.testing.assert_allclose(
        [atoms.positions[0, 0] for atoms in warmup],
        [0.1, 0.2],
    )
    with pytest.raises(RuntimeError, match="only found 2"):
        runner._select_warmup_structures(cases, 3)


def test_speed_claims_require_absolute_energy_parity():
    gates = {
        "maximum_absolute_energy_difference_hartree": 1.0e-6,
        "maximum_relative_energy_difference_kcal_mol": 0.001,
        "maximum_delta_g_difference_kcal_mol": 0.001,
        "minimum_paired_repeat_speedup": 1.25,
    }
    summaries = {
        "single_state_model": {
            "maximum_absolute_energy_difference_hartree": 2.0e-6,
            "maximum_relative_energy_difference_kcal_mol": 0.0,
            "maximum_delta_g_difference_kcal_mol": 0.0,
            "speedup": 2.0,
            "speedup_by_repeat": [2.0, 2.0],
        }
    }

    decisions = runner._engineering_decisions(
        summaries,
        gates,
        all_model_case_records_present=True,
        all_models_use_native_batch=True,
    )

    assert (
        decisions["engineering_gates"]["absolute_energy_parity_within_limit"] is False
    )
    assert decisions["batch_interface_admission_allowed"] is False
    assert decisions["universal_material_speedup_claim_allowed"] is False
    assert decisions["per_model_material_speedup"] == {"single_state_model": False}


def test_speed_claims_require_every_paired_repeat_to_pass():
    gates = {
        "maximum_absolute_energy_difference_hartree": 1.0e-6,
        "maximum_relative_energy_difference_kcal_mol": 0.001,
        "maximum_delta_g_difference_kcal_mol": 0.001,
        "minimum_paired_repeat_speedup": 1.25,
    }
    summaries = {
        "unstable_model": {
            "maximum_absolute_energy_difference_hartree": 0.0,
            "maximum_relative_energy_difference_kcal_mol": 0.0,
            "maximum_delta_g_difference_kcal_mol": 0.0,
            "speedup": 1.5,
            "speedup_by_repeat": [1.0, 2.0],
        }
    }

    decisions = runner._engineering_decisions(
        summaries,
        gates,
        all_model_case_records_present=True,
        all_models_use_native_batch=True,
    )

    assert (
        decisions["engineering_gates"]["every_paired_repeat_meets_speedup_floor"]
        is False
    )
    assert decisions["universal_material_speedup_claim_allowed"] is False
    assert decisions["per_model_material_speedup"] == {"unstable_model": False}


def test_v2_review_artifact_preserves_the_unstable_mace_speed_decision():
    artifact = _load(V2_REVIEW_ARTIFACT_PATH)
    mace = artifact["model_summaries"]["maceoff23m"]

    assert artifact["content_sha256"] == core.artifact_content_sha256(artifact)
    assert artifact["protocol_id"].endswith("-v2")
    assert mace["speedup"] >= 1.25
    assert min(mace["speedup_by_repeat"]) < 1.25
    assert max(mace["speedup_by_repeat"]) > 1.25


def test_batch_parity_artifact_separates_validity_from_speed_claims():
    artifact = _load(ARTIFACT_PATH)
    protocol, fingerprint = runner.load_protocol(PROTOCOL_PATH)

    assert artifact["content_sha256"] == core.artifact_content_sha256(artifact)
    assert artifact["protocol_fingerprint"] == fingerprint
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        RUNNER_PATH
    )
    assert artifact["source_protocol_file_sha256"] == core.sha256_file(
        SOURCE_PROTOCOL_PATH
    )
    assert artifact["source_manifest_file_sha256"] == core.sha256_file(
        SOURCE_MANIFEST_PATH
    )
    assert artifact["selection"]["case_count"] == 19
    assert artifact["selection"]["state_count"] == 1270
    timing = artifact["timing_isolation"]
    assert timing["required"] is True
    assert timing["evidence_scope"] == "preflight_and_postflight_snapshots_only"
    assert timing["endpoint_snapshots_passed"] is True
    assert timing["whole_run_gpu_exclusivity_proven"] is False
    assert timing["gpu_preflight"]["competing_compute_processes"] == []
    assert timing["gpu_postflight"]["competing_compute_processes"] == []
    assert timing["host_preflight"]["maximum_load_per_logical_cpu"] == 0.25
    assert timing["continuous_host_isolation_monitored"] is False
    assert timing["continuous_gpu_isolation_monitored"] is False
    assert set(artifact["model_summaries"]) == {
        "maceoff23m",
        "aimnet2",
        "ani2x",
    }
    assert artifact["engineering_gates"] == {
        "absolute_energy_parity_within_limit": True,
        "all_model_case_records_present": True,
        "all_models_use_native_batch": True,
        "discrete_delta_g_parity_within_limit": True,
        "every_paired_repeat_meets_speedup_floor": False,
        "relative_energy_parity_within_limit": True,
    }
    assert artifact["batch_interface_admission_allowed"] is True
    assert artifact["universal_material_speedup_claim_allowed"] is False
    assert artifact["decision_boundary"][
        "universal_material_speedup_claim_requires"
    ] == [
        "batch_interface_admission_allowed",
        "every_paired_repeat_meets_speedup_floor",
    ]
    assert artifact["per_model_material_speedup"] == {
        "maceoff23m": False,
        "aimnet2": True,
        "ani2x": True,
    }
    mace_sources = artifact["model_environments"]["maceoff23m"]["source_hashes"]
    assert mace_sources["mace_common"]["path"] == (
        "maple/function/calculator/mace/_common.py"
    )
    compatibility = artifact["hessian_diagnostics_compatibility_update"]
    for environment in artifact["model_environments"].values():
        assert {
            "benchmark_core",
            "calculator",
            "calculator_base",
            "batch_types",
            "batch_utils",
            "conformer_evaluation",
            "discrete_conformers",
            "set_calculator",
            "source_runner",
        }.issubset(environment["source_hashes"])
        assert environment["ase_version"]
        assert environment["calculator_dtype"]
        assert isinstance(environment["deterministic_algorithms_enabled"], bool)
        assert environment["float32_matmul_precision"] in {
            "highest",
            "high",
            "medium",
        }
        for source in environment["source_hashes"].values():
            observed_sha256 = core.sha256_file(REPOSITORY_ROOT / source["path"])
            if observed_sha256 != source["sha256"]:
                assert source["path"] == compatibility["source"]["path"]
                assert source["sha256"] == compatibility["source"][
                    "historical_execution_sha256"
                ]
                assert observed_sha256 == compatibility["source"][
                    "current_sha256"
                ]
                assert compatibility["historical_execution_hash_retained"] is True
                assert compatibility["batch_energy_force_path_changed"] is False
    for summary in artifact["model_summaries"].values():
        assert (
            summary["maximum_relative_energy_difference_kcal_mol"]
            <= protocol["gates"]["maximum_relative_energy_difference_kcal_mol"]
        )
        assert (
            summary["maximum_delta_g_difference_kcal_mol"]
            <= protocol["gates"]["maximum_delta_g_difference_kcal_mol"]
        )


def test_batch_parity_documentation_uses_current_artifact_timings():
    artifact = _load(ARTIFACT_PATH)
    documentation = "\n".join(
        (REPOSITORY_ROOT / "docs" / "implicit-solvation" / name).read_text(
            encoding="utf-8"
        )
        for name in (
            "FORMULAS_AND_REFERENCES.md",
            "ROUTE1_PRODUCT_SPEC.md",
            "VALIDATION_STATUS.md",
            "benchmarks/README.md",
        )
    )

    for model in ("maceoff23m", "aimnet2", "ani2x"):
        summary = artifact["model_summaries"][model]
        assert f"`{summary['speedup']:.2f}x`" in documentation
        assert f"`{min(summary['speedup_by_repeat']):.3f}x`" in documentation

    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    for model in ("maceoff23m", "aimnet2", "ani2x"):
        summary = artifact["model_summaries"][model]
        assert f"{summary['serial_states_per_second']:.2f}" in benchmark
        assert f"{summary['batch_states_per_second']:.2f}" in benchmark
