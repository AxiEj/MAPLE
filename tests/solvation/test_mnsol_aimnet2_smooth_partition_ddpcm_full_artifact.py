from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route2-mnsol-aimnet2-smooth-partition-ddpcm-full-v1.json"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark_core import canonical_json_bytes  # noqa: E402
import run_mnsol_aimnet2_smooth_partition_ddpcm as runner  # noqa: E402
import finalize_mnsol_aimnet2_smooth_partition_ddpcm as finalizer  # noqa: E402

MEASUREMENT_HEAD = "c578fda2b974dd44833b8f0813bd689751f47d4d"
AGGREGATION_HEAD = "0e84615a6d94f995e288e4e003f6d66e2c094036"
MEASUREMENT_SHA256 = "27cecc9043804d69ddbed1e7262ec362d5da60a5c24601d08555d8687b391605"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"


def _artifact() -> dict[str, object]:
    value = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _assert_sources_match_git_commit(
    commit: str, source_files_sha256: dict[str, str]
) -> None:
    assert source_files_sha256
    for relative, expected in source_files_sha256.items():
        blob = subprocess.run(
            ("git", "show", f"{commit}:{relative}"),
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == expected


def test_full_smooth_ddpcm_artifact_is_public_safe_commit_bound_and_fail_closed():
    artifact = _artifact()

    assert artifact["artifact"] == runner.ARTIFACT
    assert artifact["schema_version"] == 1
    assert artifact["complete_panel"] is True
    assert artifact["execution_git_head"] == MEASUREMENT_HEAD
    assert artifact["record_count"] == 653
    assert artifact["unique_geometry_count"] == 395
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["do_not_commit"] is False
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert artifact["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "M": False,
        "V": False,
    }
    assert artifact["decision"] == {
        "accuracy_admitted": False,
        "complete_653_record_accuracy_panel": True,
        "force_admitted": False,
        "hessian_admitted": False,
        "profile_enabled": False,
        "reason": (
            "The complete panel has 2.581704293058602 kcal/mol MAE, "
            "+2.542606484695728 kcal/mol signed bias, and 3.433153943927792 "
            "kcal/mol water MAE. Smooth-vs-high-resolution ddPCM parity is only "
            "0.0618835476444038 kcal/mol MAE, so numerical refinement of this "
            "smooth operator cannot remove the dominant frozen-source/energy-ledger "
            "error. Accuracy and E/F/H/V/M therefore remain closed; force, "
            "order-convergence, path, HVP, and workflow gates are additionally "
            "separate."
        ),
        "variational_electronic_scf_admitted": False,
    }
    assert artifact["scientific_identity"] == {
        "experimental_values_used_for_method_selection": False,
        "fit_or_calibration": False,
        "method_label": (
            "fixed-AIMNet2-Hirshfeld-like-point-charge smooth ddPCM plus a "
            "separately evaluated PySCF-2.13.1 SMD-CDS comparator"
        ),
        "smd_cds_role": (
            "separate cavity-dispersion-solvent-structure component; it does not "
            "turn the fixed-point-charge electrostatics into original density-based "
            "self-consistent SMD"
        ),
        "source": finalizer.SOURCE_SEMANTICS,
    }
    assert artifact["references"] == finalizer.PRIMARY_REFERENCES
    assert "not original SMD" in artifact["claim_boundary"]
    assert "approximately 1 kcal/mol" in artifact["claim_boundary"]
    runner._assert_public_safe(artifact)

    measurement = artifact["measurement_provenance"]
    aggregation = artifact["aggregation_provenance"]
    assert measurement["git_head"] == MEASUREMENT_HEAD
    assert aggregation["git_head"] == AGGREGATION_HEAD
    assert measurement["shards"] == {
        "count": 653,
        "provenance_sha256": (
            "5a3c574cc5c8c9f7591409d2affa93660f6396baf5bb2937e6af2d9695eba74c"
        ),
    }
    assert measurement["private_artifact_sha256"] == (
        "6844978ae5f0bcd9e05949903ea66e1976a32dca6eb5ce0e7f2b92d17bbf4ae6"
    )
    assert measurement["raw_public_artifact_sha256"] == (
        "be79e5710b404c5d8897e2e16be596995c695437a6a405dd026ee1102fb23c9b"
    )
    _assert_sources_match_git_commit(
        MEASUREMENT_HEAD, measurement["source_files_sha256"]
    )
    _assert_sources_match_git_commit(
        AGGREGATION_HEAD, aggregation["source_files_sha256"]
    )


def test_full_smooth_ddpcm_metrics_replay_the_public_measurement_digest():
    artifact = _artifact()
    aggregate = artifact["aggregate_metrics"]
    assert aggregate["record_count"] == 653
    assert aggregate["mean_absolute_error_kcal_mol"] == pytest.approx(
        2.581704293058602, rel=0.0, abs=0.0
    )
    assert aggregate["mean_signed_error_kcal_mol"] == pytest.approx(
        2.542606484695728, rel=0.0, abs=0.0
    )
    assert aggregate["root_mean_square_error_kcal_mol"] == pytest.approx(
        3.3525667444695353, rel=0.0, abs=0.0
    )
    assert aggregate["maximum_absolute_error_kcal_mol"] == pytest.approx(
        10.534260946189113, rel=0.0, abs=0.0
    )
    assert aggregate["reference_highres_mean_absolute_error_kcal_mol"] == (
        pytest.approx(2.5442693466168476, rel=0.0, abs=0.0)
    )
    assert aggregate["smooth_vs_reference_mean_absolute_kcal_mol"] == (
        pytest.approx(0.0618835476444038, rel=0.0, abs=0.0)
    )
    assert aggregate["smooth_vs_reference_maximum_absolute_kcal_mol"] == (
        pytest.approx(0.4710462188888806, rel=0.0, abs=0.0)
    )

    assert artifact["partition_metrics"]["development"]["record_count"] == 505
    assert artifact["partition_metrics"]["confirmation"]["record_count"] == 148
    assert artifact["partition_metrics"]["confirmation"][
        "mean_absolute_error_kcal_mol"
    ] == pytest.approx(2.411133789710193, rel=0.0, abs=0.0)
    assert artifact["solvent_metrics"]["water"]["record_count"] == 387
    assert artifact["solvent_metrics"]["water"][
        "mean_absolute_error_kcal_mol"
    ] == pytest.approx(3.433153943927792, rel=0.0, abs=0.0)
    assert artifact["atom_count_bin_metrics"]["21-plus"][
        "mean_absolute_error_kcal_mol"
    ] == pytest.approx(4.023831292476598, rel=0.0, abs=0.0)

    assert artifact["factor_degree_preflight"] == {
        "algebraic_degree_bound": 192,
        "failed_record_count": 0,
        "maximum_factor_count": 22,
        "maximum_required_algebraic_degree": 184,
    }
    assert artifact["source"]["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["source"]["continuum_field_supplied_to_aimnet2"] is False
    assert artifact["source"]["electronic_scf_iteration"] is False
    assert artifact["dataset"]["standard_state"] == (
        "1M-ideal-gas-to-1M-ideal-solution"
    )

    scientific_measurement = {
        "record_payload_sha256": artifact["measurement_provenance"][
            "record_payload_sha256"
        ],
        "aggregate_metrics": artifact["aggregate_metrics"],
        "partition_metrics": artifact["partition_metrics"],
        "solvent_metrics": artifact["solvent_metrics"],
        "atom_count_bin_metrics": artifact["atom_count_bin_metrics"],
        "factor_degree_preflight": artifact["factor_degree_preflight"],
        "method": artifact["method"],
        "dataset": artifact["dataset"],
    }
    assert hashlib.sha256(canonical_json_bytes(scientific_measurement)).hexdigest() == (
        MEASUREMENT_SHA256
    )
