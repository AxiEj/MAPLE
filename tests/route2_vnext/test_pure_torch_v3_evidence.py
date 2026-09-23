"""The public v3 experimental claim must be source-bound and bounded."""

import hashlib
import json
from pathlib import Path

from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
    SCALAR_REGISTRY,
)
from tools.route2_release.run_pure_mace_polar_torch_canary import _source_manifest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ID = "route2-pure-macepolar-torch-analytic-v3-contract-tests"
EVIDENCE = ROOT / "docs/route2/evidence" / f"{EVIDENCE_ID}.json"
EXPECTED_CASES = {
    (case, device)
    for case in ("water-water", "methane-water", "methane-hexane")
    for device in ("cpu", "cuda:0")
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_evidence_is_frozen_to_the_current_implementation_and_has_no_admission():
    record = json.loads(EVIDENCE.read_text())
    assert record["schema_version"] == 1
    assert record["artifact_id"] == EVIDENCE_ID
    assert record["source_manifest_sha256"] == _digest(_source_manifest())
    assert record["source_file_count"] == len(_source_manifest())
    assert record["scientific_release_admitted"] is False
    assert record["physical_force_accuracy_admitted"] is False
    assert record["experimental_solvation_accuracy_admitted"] is False
    assert record["confirmation_partition_opened"] is False
    assert record["dataset_labels_used"] is False
    assert record["method"]["frozen_zero_field_source"] is True
    assert set(record["profiles"]) == {
        "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cpu-v3",
        "pure-macepolar-frozen-point-l1-ddpcm-smd-torch-cuda-v3",
    }
    for scalar in (
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CPU_V3,
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_TORCH_CUDA_V3,
    ):
        assert SCALAR_REGISTRY[scalar].experimental_evidence_artifact_ids == (
            EVIDENCE_ID,
        )
        assert SCALAR_REGISTRY[scalar].admitted_capabilities.enabled_tiers == ()


def test_all_six_cases_and_failures_are_accounted_for_without_selection():
    record = json.loads(EVIDENCE.read_text())
    cases = record["cases"]
    assert {(item["case"], item["device"]) for item in cases} == EXPECTED_CASES
    assert len(cases) == len(EXPECTED_CASES)
    assert record["requested_case_count"] == 6
    passed = sum(item["pass"] is True for item in cases)
    assert record["passed_case_count"] == passed
    assert record["status"] == (
        "bounded-implementation-parity-pass" if passed == 6 else "partial-or-failed"
    )
    for item in cases:
        assert item["source_unchanged"] is True
        assert len(item["result_sha256"]) == 64
        assert item["checkpoint_sha256"] == record["method"]["checkpoint_sha256"]
        if item["pass"]:
            assert item["energy_error_eV"] <= 1e-7
            assert item["force_error_eV_per_A"] <= 1e-6
            assert item["hvp_error_eV_per_A2"] <= 1e-4
            assert item["raw_hessian_antisymmetry_eV_per_A2"] <= 1e-4
            assert item["independent_fd_audit_pass"] is True
        else:
            assert item["failure"] is not None


def test_cross_device_e_f_h_parity_is_measured_not_assumed():
    record = json.loads(EVIDENCE.read_text())
    for item in record["cross_device_parity"]:
        assert item["case"] in {case for case, _ in EXPECTED_CASES}
        assert item["energy_difference_eV"] <= 1e-7
        assert item["force_difference_eV_per_A"] <= 1e-6
        assert item["hessian_difference_eV_per_A2"] <= 1e-4
        assert item["both_case_passed"] is True
    if record["status"] == "bounded-implementation-parity-pass":
        assert len(record["cross_device_parity"]) == 3
