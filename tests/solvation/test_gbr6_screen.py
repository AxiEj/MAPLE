from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_gbr6_screen as screen

PROTOCOL_PATH = BENCHMARK_DIR / "gbr6_screen_protocol.json"
ENERGY_PATH = BENCHMARK_DIR / "route1-gbr6-energy-screen-2026-07-24.json"
SUMMARY_PATH = BENCHMARK_DIR / "route1-gbr6-development-summary-2026-07-24.json"


def test_gbr6_parser_is_finite_and_fails_closed():
    assert screen.parse_gbr6_energy("Delta Ggbr6 = -12.3456\n") == pytest.approx(
        -12.3456
    )
    with pytest.raises(ValueError, match="does not contain"):
        screen.parse_gbr6_energy("Total energy = -12.3456\n")
    with pytest.raises(ValueError, match="non-finite"):
        screen.parse_gbr6_energy("Delta Ggbr6 = 1e309\n")


def test_gbr6_protocol_preserves_the_route1_and_derivative_boundaries():
    protocol = screen.load_protocol(PROTOCOL_PATH)

    assert protocol["route1_boundary"] == {
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    assert protocol["energy_endpoint"]["radius_profile"] == "gbn-bondi"
    assert (
        protocol["execution_boundary"]["energy_phase_reads_experimental_labels"]
        is False
    )
    assert (
        protocol["derivative_audit"]["released_force_or_gradient_interface_identified"]
        is False
    )


def test_gbr6_external_source_hash_mismatch_fails_closed(tmp_path: Path):
    protocol = screen.load_protocol(PROTOCOL_PATH)
    archive = tmp_path / "GBr6.tgz"
    fortran = tmp_path / "GBr6_v1.f"
    executable = tmp_path / "GBr6_v1.x"
    archive.write_bytes(b"not-the-pinned-archive")
    fortran.write_bytes(b"not-the-pinned-source")
    executable.write_bytes(b"not-a-real-executable")

    with pytest.raises(ValueError, match="archive SHA-256"):
        screen._verify_external_source(
            protocol,
            archive=archive,
            fortran_source=fortran,
            executable=executable,
        )


def test_frozen_gbr6_energy_artifact_is_complete_label_free_evidence():
    artifact = core.load_json(ENERGY_PATH)

    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["protocol_sha256"] == core.sha256_file(PROTOCOL_PATH)
    assert artifact["case_count"] == 526
    assert len(artifact["records"]) == 526
    assert len({record["compound_id"] for record in artifact["records"]}) == 526
    assert "experimental" not in json.dumps(artifact["records"], sort_keys=True).lower()
    assert all(
        len(record["charge_vector_sha256"]) == 64
        and len(record["radius_vector_sha256"]) == 64
        for record in artifact["records"]
    )


def test_frozen_gbr6_summary_rejects_the_candidate_without_a_residual():
    summary = core.load_json(SUMMARY_PATH)

    assert core.artifact_content_sha256(summary) == summary["content_sha256"]
    assert summary["energy_artifact_sha256"] == core.sha256_file(ENERGY_PATH)
    assert summary["case_count"] == 526
    assert summary["label_use_boundary"]["experimental_fit_or_residual"] is False
    assert summary["methods"]["obc2_ace_kcal_mol"]["mae_kcal_mol"] == pytest.approx(
        1.7603512076636891
    )
    assert summary["methods"]["chagb_cavity_dispersion_kcal_mol"][
        "mae_kcal_mol"
    ] == pytest.approx(1.321851711026616)
    assert summary["methods"]["gbr6_cavity_dispersion_kcal_mol"][
        "mae_kcal_mol"
    ] == pytest.approx(2.256742205323194)
    assert summary["paired_gain"]["obc2_to_gbr6"][
        "bootstrap_ci_95_kcal_mol"
    ] == pytest.approx([-0.6810422877129002, -0.3173464998868841])
    assert summary["paired_gain"]["chagb_to_gbr6"][
        "mean_mae_gain_kcal_mol"
    ] == pytest.approx(-0.9348904942965781)
    assert summary["decision"] == {
        "accuracy_gate_passed": False,
        "disposition": "reject-product-provider",
        "maintained_dependency_gate_passed": False,
        "product_default_changed": False,
        "reason": (
            "GBr6/PBSA cavity-dispersion is less accurate than both frozen "
            "comparators, and the released program exposes no force interface."
        ),
        "released_force_interface_identified": False,
    }


@pytest.mark.parametrize("artifact_path", [ENERGY_PATH, SUMMARY_PATH])
def test_gbr6_artifacts_pin_the_runner_that_created_them(artifact_path: Path):
    artifact = core.load_json(artifact_path)
    command = artifact["command_provenance"]
    script = REPOSITORY_ROOT / command["script"]

    assert command["script"] == (
        "docs/implicit-solvation/benchmarks/run_gbr6_screen.py"
    )
    assert command["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
