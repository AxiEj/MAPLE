from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "run_fennix_bio1_native_kernel_smoke.py"
)
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "fennix-bio1-native-kernel-mechanics-2026-07-31.json"
)


def _load_module():
    specification = importlib.util.spec_from_file_location(
        "fennix_bio1_native_kernel_smoke",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


AUDIT = _load_module()


def _artifact():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_frozen_artifact_preserves_full_mechanics_and_closed_gates():
    payload = _artifact()

    assert payload["verdict"] == (
        "production_native_kernel_mechanics_available_admission_blocked"
    )
    assert payload["acceptance_eligible"] is False
    assert payload["formal_gates"] == AUDIT.FORMAL_GATES
    assert payload["identity"]["checkpoint_sha256"] == AUDIT.CHECKPOINT_SHA256
    assert payload["identity"]["alchemical_parameters"] == (
        AUDIT.EXPECTED_ALCHEMICAL_PARAMETERS
    )
    assert payload["identity"]["maple_kernel_source_sha256"] == (
        AUDIT.EXPECTED_MAPLE_KERNEL_SOURCE_SHA256
    )
    assert payload["identity"]["fennol_package_tree_sha256"] == (
        AUDIT.EXPECTED_FENNOL_PACKAGE_TREE_SHA256
    )
    assert payload["identity"]["fennol_package_tree_file_count"] == (
        AUDIT.EXPECTED_FENNOL_PACKAGE_TREE_FILE_COUNT
    )
    assert payload["identity"]["repulsion_nlh_coefficients_provenance"] == (
        AUDIT.EXPECTED_REPULSION_NLH_COEFFICIENTS_PROVENANCE
    )
    assert (
        payload["identity"]["runtime_source_sha256"][
            "fennol/models/physics/nlh_coeffs.dat"
        ]
        == AUDIT.EXPECTED_RUNTIME_SOURCE_SHA256["fennol/models/physics/nlh_coeffs.dat"]
    )
    boundary = payload["scientific_boundary"]
    assert boundary["full_energy_force_cell_gradient_virial_lambda_arrays_saved"]
    assert boundary["nontrivial_interacting_coupling_required"]
    assert boundary["protocol_reconstruction_not_paper_reproduction"]
    assert boundary["real_pinned_close_contact_softcore_canary_run"]
    assert boundary["close_contact_nontrivial_active_softcore_response_required"]
    assert boundary["close_contact_p_equals_one_evaluated"] is False
    assert boundary["sampling_run"] is False
    assert boundary["lambda_abf_run"] is False
    assert boundary["hfe_estimated"] is False
    assert boundary["experimental_records_used"] == 0
    assert boundary["distinct_primary_functional_groups_used"] == 0
    assert boundary["gpu_admitted"] is False
    assert boundary["matched_qm_timing_eligible"] is False
    assert boundary["performance_claim_allowed"] is False

    for platform in ("cpu", "gpu"):
        receipt = payload[f"{platform}_receipt"]
        assert len(receipt["rows"]) == 5
        assert receipt["identity"]["alchemical_parameters"] == (
            AUDIT.EXPECTED_ALCHEMICAL_PARAMETERS
        )
        for row in receipt["rows"]:
            assert len(row["forces_ev_per_angstrom"]) == 6
            assert all(len(vector) == 3 for vector in row["forces_ev_per_angstrom"])
            assert len(row["cell_gradient_ev_per_angstrom"]) == 3
            assert all(
                len(vector) == 3 for vector in row["cell_gradient_ev_per_angstrom"]
            )
            assert len(row["virial_ev"]) == 3
            assert all(len(vector) == 3 for vector in row["virial_ev"])
        canary = receipt["close_contact_canary"]
        assert canary["progress_states"] == [0.0, 0.25]
        assert canary["neighbor_overflow_observed"] is False
        assert canary["p_equals_one_evaluated"] is False
        assert canary["mechanics_only"] is True
        assert len(canary["rows"]) == 2
        for row in canary["rows"]:
            assert len(row["forces_ev_per_angstrom"]) == 6
            assert len(row["cell_gradient_ev_per_angstrom"]) == 3
            assert len(row["virial_ev"]) == 3


def test_frozen_artifact_recomputes_same_system_full_array_comparison():
    payload = _artifact()
    cpu = payload["cpu_receipt"]
    gpu = payload["gpu_receipt"]

    cpu_metrics = AUDIT.validate_receipt(cpu, expected_platform="cpu")
    gpu_metrics = AUDIT.validate_receipt(gpu, expected_platform="gpu")
    comparison = AUDIT.compare_receipts(cpu, gpu)

    assert comparison == payload["recomputed_same_system_comparison"]
    assert cpu_metrics["energy_span_ev"] > AUDIT.MINIMUM_ENERGY_SPAN_EV
    assert gpu_metrics["energy_span_ev"] > AUDIT.MINIMUM_ENERGY_SPAN_EV
    assert (
        cpu_metrics["maximum_absolute_active_derivative_ev"]
        > AUDIT.MINIMUM_ACTIVE_DERIVATIVE_EV
    )
    assert (
        gpu_metrics["maximum_absolute_active_derivative_ev"]
        > AUDIT.MINIMUM_ACTIVE_DERIVATIVE_EV
    )
    assert comparison["global_max_abs_diff"] < 1.0e-12
    assert comparison["same_alchemical_parameters"] is True
    assert comparison["same_maple_kernel_source_sha256"] is True
    assert comparison["same_fennol_package_tree_sha256"] is True
    assert comparison["same_repulsion_nlh_coefficients_provenance"] is True
    assert comparison["close_contact_canary"]["same_system"] is True
    assert comparison["close_contact_canary"]["neighbor_overflow_observed"] is False
    assert comparison["close_contact_canary"]["p_equals_one_evaluated"] is False
    assert comparison["close_contact_canary"]["all_full_arrays_finite"] is True
    assert comparison["close_contact_canary"]["global_max_abs_diff"] < 1.0e-12
    for platform in ("cpu", "gpu"):
        metrics = comparison["close_contact_canary"][
            f"{platform}_nontrivial_active_softcore_response"
        ]
        assert metrics["passed"] is True
        assert (
            metrics["maximum_energy_change_from_p0_ev"] > AUDIT.MINIMUM_ENERGY_SPAN_EV
        )
        assert (
            metrics["maximum_absolute_active_derivative_ev"]
            > AUDIT.MINIMUM_ACTIVE_DERIVATIVE_EV
        )


def test_validation_rejects_beyond_cutoff_trivial_and_tampered_receipts():
    cpu = _artifact()["cpu_receipt"]

    beyond_cutoff = deepcopy(cpu)
    beyond_cutoff["system"]["solute_solvent_oxygen_distance_angstrom"] = 8.0
    beyond_cutoff["system"]["inside_cutoff"] = False
    with pytest.raises(ValueError, match="outside the 7.5 angstrom cutoff"):
        AUDIT.validate_receipt(beyond_cutoff, expected_platform="cpu")

    trivial = deepcopy(cpu)
    energy = trivial["rows"][0]["energy_ev"]
    for row in trivial["rows"]:
        row["energy_ev"] = energy
        row["denergy_dprogress_ev"] = 0.0
    with pytest.raises(ValueError, match="coupling is trivial"):
        AUDIT.validate_receipt(trivial, expected_platform="cpu")

    checkpoint_tamper = deepcopy(cpu)
    checkpoint_tamper["identity"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint_sha256 identity mismatch"):
        AUDIT.validate_receipt(checkpoint_tamper, expected_platform="cpu")

    softcore_tamper = deepcopy(cpu)
    softcore_tamper["identity"]["alchemical_parameters"][
        "repulsion_softcore_angstrom"
    ] = 0.25
    with pytest.raises(ValueError, match="alchemical_parameters identity mismatch"):
        AUDIT.validate_receipt(softcore_tamper, expected_platform="cpu")

    source_tamper = deepcopy(cpu)
    source_tamper["maple_kernel_source_sha256"]["kernel.py"] = "f" * 64
    with pytest.raises(ValueError, match="MAPLE kernel source identity mismatch"):
        AUDIT.validate_receipt(source_tamper, expected_platform="cpu")

    tree_tamper = deepcopy(cpu)
    tree_tamper["identity"]["fennol_package_tree_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="fennol_package_tree_sha256"):
        AUDIT.validate_receipt(tree_tamper, expected_platform="cpu")

    nlh_tamper = deepcopy(cpu)
    nlh_tamper["identity"]["repulsion_nlh_coefficients_provenance"] = "forged"
    with pytest.raises(ValueError, match="repulsion_nlh_coefficients_provenance"):
        AUDIT.validate_receipt(nlh_tamper, expected_platform="cpu")

    canary_overflow = deepcopy(cpu)
    canary_overflow["close_contact_canary"]["neighbor_overflow_observed"] = True
    with pytest.raises(ValueError, match="observed neighbor overflow"):
        AUDIT.validate_receipt(canary_overflow, expected_platform="cpu")

    canary_nonfinite = deepcopy(cpu)
    canary_nonfinite["close_contact_canary"]["rows"][0]["energy_ev"] = float("nan")
    with pytest.raises(ValueError, match="energy_ev must be finite"):
        AUDIT.validate_receipt(canary_nonfinite, expected_platform="cpu")

    canary_trivial = deepcopy(cpu)
    canary_energy = canary_trivial["close_contact_canary"]["rows"][0]["energy_ev"]
    for row in canary_trivial["close_contact_canary"]["rows"]:
        row["energy_ev"] = canary_energy
        row["denergy_dprogress_ev"] = 0.0
    with pytest.raises(ValueError, match="nontrivial active-softcore response"):
        AUDIT.validate_receipt(canary_trivial, expected_platform="cpu")


def test_frozen_artifact_hashes_and_deterministic_builder_match():
    payload = _artifact()
    source_hashes = payload["source_sha256"]

    assert source_hashes["runner"] == AUDIT._sha256_file(SCRIPT)
    assert (
        source_hashes["cpu_receipt"]
        == "173f36762e74be8c85cc9ac12f4cc51b65ce749df5ceee80ba5b0380345d4773"
    )
    assert (
        source_hashes["gpu_receipt"]
        == "b2f9cdf1b85ba619d80833199fca0ce4ebdbb64dfc691f2a8cc9568973742cf3"
    )
    assert source_hashes["comparison_receipt"] == (
        "a5b70d241f7e8e776bfb4d9421602950bd6cef839eee0ad6ff2de9615dc55c48"
    )

    rebuilt = AUDIT.build_artifact(
        cpu=payload["cpu_receipt"],
        gpu=payload["gpu_receipt"],
        source_comparison=payload["source_comparison_receipt"],
        cpu_sha256=source_hashes["cpu_receipt"],
        gpu_sha256=source_hashes["gpu_receipt"],
        comparison_sha256=source_hashes["comparison_receipt"],
        runner_sha256=source_hashes["runner"],
    )
    assert rebuilt == payload
