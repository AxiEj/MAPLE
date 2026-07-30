from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "run_anisolv_compact_gpu_no_loss_audit.py"
)
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "anisolv-compact-gpu-no-loss-audit-2026-07-30.json"
)
REFERENCE_ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "anisolv-compact-upstream-sample-audit-2026-07-30.json"
)
WATCHLIST = ROOT / "docs" / "pretrained-solvation-hub" / "research_watchlist.yaml"
MODEL_CARD = (
    ROOT / "maple" / "function" / "calculator" / "model_cards" / "anisolv-compact.yaml"
)
CONFIGURED_UPSTREAM = os.environ.get("MAPLE_ANISOLV_GPU_AUDIT_ROOT")


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "anisolv_compact_gpu_no_loss_audit",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _scientific_projection(payload):
    projection = deepcopy(payload)
    for lane in projection["lanes"].values():
        lane.pop("timing_diagnostic_not_an_admission_observable")
    projection.pop("runtime")
    return projection


def test_literal_energy_equality_is_required():
    panel = [
        {
            "atomic_numbers": [1],
            "g2_name": "H",
            "label": "synthetic",
            "positions_angstrom": [[0.0, 0.0, 0.0]],
        }
    ]
    lane, manifest = AUDIT._compare_lane(
        dtype_name="float64",
        panel=panel,
        reference_energies_ev=[0.0],
        accelerator_energies_ev=[1.0e-16],
        reference_timing_seconds=[1.0, 1.0, 1.0],
        accelerator_timing_seconds=[0.5, 0.5, 0.5],
    )

    assert lane["energy_parity"]["passed_exact_equality"] is False
    assert lane["no_loss_passed"] is False
    assert lane["verdict"] == "fail_no_loss"
    assert (
        lane["timing_diagnostic_not_an_admission_observable"][
            "useful_acceleration_observed"
        ]
        is True
    )
    assert len(manifest) == 1


def test_frozen_gpu_audit_is_physical_negative_evidence_not_admission():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["acceptance_eligible"] is False
    assert payload["artifact_kind"] == (
        "negative_physical_hardware_observation_not_gpu_admission_evidence"
    )
    assert payload["identity"]["revision"] == AUDIT.SOURCE_AUDIT.UPSTREAM_REVISION
    assert payload["identity"]["tree"] == AUDIT.SOURCE_AUDIT.UPSTREAM_TREE
    assert payload["identity"]["checkpoint"]["sha256"] == (
        AUDIT.SOURCE_AUDIT.CHECKPOINT["sha256"]
    )
    assert payload["panel"]["geometry_and_source_row_sha256"] == (
        AUDIT.SOURCE_AUDIT.PANEL_IDENTITY_SHA256
    )
    assert payload["panel"]["experimental_label_values_used_for_identity"] is False
    assert payload["panel"]["record_count_per_dtype"] == 6
    assert payload["verdict"] == "fail_no_loss"
    assert payload["gpu_admission"] == {
        "accuracy_panel_eligible": False,
        "allowed_inference_modes": [],
        "allowed_tasks": [],
        "card_evidence_must_remain_null": True,
        "no_loss_parity_verified": False,
        "smaller_panel_can_reject_but_never_unlock": True,
        "status": "blocked_nonzero_cpu_cuda_scalar_difference",
    }
    assert payload["panel"]["accuracy_metric_reporting_allowed"] is False
    assert payload["panel"]["functional_group_accuracy_gate"] == {
        "functional_group_taxonomy": None,
        "minimum_distinct_functional_groups": 10,
        "minimum_record_count": 10,
        "observed_distinct_functional_groups": 0,
        "observed_record_count": 6,
        "passes": False,
        "record_assignments_predeclared": False,
    }
    assert (
        payload["scientific_scope"]["performance_can_override_scientific_failure"]
        is False
    )
    assert payload["scientific_scope"]["strict_literal_zero_loss_required"] is True
    assert payload["scientific_scope"]["experimental_labels_used"] is False
    assert (
        "experimental_error_comparisons_are_diagnostic_only"
        not in payload["scientific_scope"]
    )
    assert payload["scientific_scope"]["no_training_or_fine_tuning_performed"] is True
    assert payload["scientific_scope"]["no_fitting_or_calibration_performed"] is True

    runtime = payload["runtime"]
    assert runtime["accelerator_backend"] == "cuda"
    assert runtime["cublas_workspace_config"] == ":4096:8"
    assert runtime["deterministic_algorithms"] is True
    assert runtime["gpu"]["name"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert runtime["gpu"]["compute_capability"] == "8.9"
    assert runtime["torch_version"] == "2.12.0+cu130"
    assert runtime["cuda_runtime_version"] == "13.0"

    precision_state = payload["precision_backend_state"]
    assert precision_state["float32_matmul_precision"] == "highest"
    for name, value in precision_state.items():
        if name != "float32_matmul_precision":
            assert value is False, name

    float64 = payload["lanes"]["float64"]
    assert float64["energy_parity"]["comparison_count"] == 6
    assert float64["energy_parity"]["nonzero_difference_count"] == 5
    assert float64["energy_parity"][
        "observed_maximum_abs_difference_ev"
    ] == pytest.approx(1.1102230246251565e-16)
    assert float64["no_loss_passed"] is False

    float32 = payload["lanes"]["float32"]
    assert float32["energy_parity"]["comparison_count"] == 6
    assert float32["energy_parity"]["nonzero_difference_count"] == 6
    assert float32["energy_parity"][
        "observed_maximum_abs_difference_ev"
    ] == pytest.approx(4.3986372960658215e-08)
    assert float32["no_loss_passed"] is False

    for lane in payload["lanes"].values():
        precision = lane["precision_controls"]
        assert precision["same_scalar_precision"] is True
        assert precision["reference_dtype"] == precision["accelerator_dtype"]
        assert precision["matmul_precision"] == "highest"
        for name in (
            "autocast",
            "bfloat16",
            "fast_math",
            "float16",
            "reduced_matmul",
            "relaxed_convergence",
            "shortened_sampling",
            "tf32",
        ):
            assert precision[name] is False
        timing = lane["timing_diagnostic_not_an_admission_observable"]
        assert timing["reference_median_seconds"] > 0.0
        assert timing["accelerator_median_seconds"] > 0.0
        assert timing["speedup_reference_over_accelerator"] > 0.0
        assert timing["useful_acceleration_observed"] is False


def test_frozen_gpu_audit_recomputes_raw_hashes_and_manifest():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE_ARTIFACT.read_text(encoding="utf-8"))
    reference_records = {
        record["g2_name"]: record for record in reference["panel"]["records"]
    }
    manifest_records = {}

    for dtype_name, lane in payload["lanes"].items():
        paired_outputs = lane["energy_parity"]["paired_outputs"]
        assert len(paired_outputs) == 6
        cpu_energies = []
        cuda_energies = []
        differences = []
        for comparison in paired_outputs:
            g2_name = comparison["record_id"].split(":", 1)[1]
            item = reference_records[g2_name]
            cpu_energy = comparison["reference_energy_ev"]
            cuda_energy = comparison["accelerator_energy_ev"]
            assert math.isfinite(cpu_energy)
            assert math.isfinite(cuda_energy)
            assert comparison["reference_sha256"] == AUDIT._output_sha256(
                comparison["record_id"],
                "reference",
                cpu_energy,
            )
            assert comparison["accelerator_sha256"] == AUDIT._output_sha256(
                comparison["record_id"],
                "accelerator",
                cuda_energy,
            )
            difference = abs(cuda_energy - cpu_energy)
            assert comparison["absolute_energy_difference_ev"] == difference
            assert comparison["energy_exactly_equal"] is (difference == 0.0)
            cpu_energies.append(cpu_energy)
            cuda_energies.append(cuda_energy)
            differences.append(difference)
            manifest_records[comparison["record_id"]] = {
                "configuration_sha256": AUDIT._configuration_sha256(
                    item,
                    dtype_name=dtype_name,
                ),
                "observable_output_sha256s": {
                    "energy": {
                        "accelerator": comparison["accelerator_sha256"],
                        "reference": comparison["reference_sha256"],
                    }
                },
                "panel_id": AUDIT.SOURCE_AUDIT.PANEL_IDENTITY_SHA256,
                "record_id": comparison["record_id"],
            }

        assert lane["energy_parity"]["observed_maximum_abs_difference_ev"] == max(
            differences
        )
        assert lane["energy_parity"]["nonzero_difference_count"] == sum(
            difference != 0.0 for difference in differences
        )
        assert lane["no_loss_passed"] is False

    manifest = payload["comparison_manifest"]
    assert {
        record["record_id"]: record for record in manifest["records"]
    } == manifest_records
    assert manifest["observable_counts"] == {"energy": 12}
    assert (
        payload["comparison_manifest_sha256"]
        == hashlib.sha256(AUDIT.SOURCE_AUDIT._canonical_bytes(manifest)).hexdigest()
    )

    float32_cpu = {
        comparison["record_id"].split(":", 1)[1]: comparison["reference_energy_ev"]
        for comparison in payload["lanes"]["float32"]["energy_parity"]["paired_outputs"]
    }
    assert float32_cpu == {
        name: record["correction_energy_ev"]
        for name, record in reference_records.items()
    }


def test_watchlist_binds_negative_gpu_audit_but_model_card_stays_closed():
    watchlist = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    candidate = next(
        model for model in watchlist["models"] if model["model_id"] == "anisolv-compact"
    )
    audit = candidate["gpu_no_loss_audit"]

    assert audit["artifact"] == ARTIFACT.relative_to(WATCHLIST.parent).as_posix()
    assert audit["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert "failed_literal_zero_loss" in audit["status"]
    assert "can_reject_but_never_unlock_gpu" in audit["status"]
    assert audit["no_loss_parity_verified"] is False
    assert audit["gpu_admitted"] is False

    card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
    gpu = card["gpu_acceleration"]
    assert gpu["no_loss_parity_verified"] is False
    assert gpu["evidence_artifact"] is None
    assert gpu["evidence_sha256"] is None
    assert gpu["allowed_tasks"] == []
    assert gpu["allowed_inference_modes"] == []


def test_fresh_physical_gpu_audit_matches_frozen_scientific_projection(tmp_path):
    if CONFIGURED_UPSTREAM is None:
        pytest.skip(
            "Set MAPLE_ANISOLV_GPU_AUDIT_ROOT to run the physical-GPU integration test."
        )
    upstream = Path(CONFIGURED_UPSTREAM)
    if not upstream.is_dir():
        pytest.fail(f"Configured AniSolv GPU audit checkout does not exist: {upstream}")

    output = tmp_path / "gpu-audit.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--upstream-root",
            str(upstream),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fresh = json.loads(output.read_text(encoding="utf-8"))
    frozen = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert _scientific_projection(fresh) == _scientific_projection(frozen)
