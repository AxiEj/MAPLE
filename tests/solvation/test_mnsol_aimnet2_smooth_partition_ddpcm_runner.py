from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mnsol_aimnet2_smooth_partition_ddpcm as runner  # noqa: E402


def _record(error: float, parity: float, *, atoms: int = 3) -> dict[str, object]:
    return {
        "signed_error_kcal_mol": error,
        "smooth_minus_reference_kcal_mol": parity,
        "atom_count": atoms,
        "partition": "development",
        "canonical_solvent": "water",
    }


def test_factor_degree_preflight_counts_exact_pair_products():
    count, degree = runner._factor_degree_preflight(
        [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [-0.74, 0.0, 0.0]],
        [1.2, 1.2, 1.2],
    )

    assert count == 2
    assert degree == 3 * runner.AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX
    assert degree <= runner.HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE


def test_metrics_report_accuracy_tails_and_reference_parity():
    metrics = runner._metrics([_record(-1.0, 0.1), _record(3.0, -0.2)])

    assert metrics["record_count"] == 2
    assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(1.0)
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(2.0)
    assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(5.0**0.5)
    assert metrics["reference_highres_mean_absolute_error_kcal_mol"] == pytest.approx(
        2.15
    )
    assert metrics["smooth_vs_reference_mean_absolute_kcal_mol"] == pytest.approx(0.15)
    assert metrics["smooth_vs_reference_maximum_absolute_kcal_mol"] == pytest.approx(
        0.2
    )


def test_resume_shard_requires_exact_commit_and_task_fingerprint():
    task = {
        "selection_index": 7,
        "partition_selection_index": 3,
        "task_sha256": "a" * 64,
        "opaque_record_id": "private-record",
        "partition": "development",
        "geometry_handle": "private-geometry",
        "geometry_sha256": "d" * 64,
        "canonical_solvent": "water",
        "atom_count": 3,
        "factor_count": 2,
        "required_algebraic_degree": 24,
        "reference_ddpcm_energy_hartree": -0.01,
        "smd_cds_energy_kcal_mol": -0.5,
        "experimental_delta_g_kcal_mol": -4.0,
    }
    continuum_ev = -0.2
    continuum_kcal = continuum_ev / runner.HARTREE_TO_EV * runner.HARTREE_TO_KCAL_MOL
    reference_ev = task["reference_ddpcm_energy_hartree"] * runner.HARTREE_TO_EV
    parity_ev = continuum_ev - reference_ev
    parity_kcal = parity_ev / runner.HARTREE_TO_EV * runner.HARTREE_TO_KCAL_MOL
    predicted = continuum_kcal + task["smd_cds_energy_kcal_mol"]
    signed_error = predicted - task["experimental_delta_g_kcal_mol"]
    shard = {
        "artifact": runner.ARTIFACT,
        "schema_version": 1,
        "do_not_commit": True,
        "execution_git_head": "b" * 40,
        "selection_index": 7,
        "partition_selection_index": 3,
        "task_sha256": "a" * 64,
        "opaque_record_id": "private-record",
        "partition": "development",
        "geometry_handle": "private-geometry",
        "geometry_sha256": "d" * 64,
        "canonical_solvent": "water",
        "atom_count": 3,
        "factor_count": 2,
        "required_algebraic_degree": 24,
        "continuum_energy_ev": continuum_ev,
        "continuum_energy_kcal_mol": continuum_kcal,
        "reference_ddpcm_energy_ev": reference_ev,
        "smooth_minus_reference_ev": parity_ev,
        "smooth_minus_reference_kcal_mol": parity_kcal,
        "smd_cds_energy_kcal_mol": task["smd_cds_energy_kcal_mol"],
        "experimental_delta_g_kcal_mol": task["experimental_delta_g_kcal_mol"],
        "predicted_delta_g_kcal_mol": predicted,
        "signed_error_kcal_mol": signed_error,
        "absolute_error_kcal_mol": abs(signed_error),
        "configuration_sha256": "e" * 64,
        "provenance_sha256": "f" * 64,
        "timing_seconds": 1.0,
    }

    assert runner._valid_shard(shard, task=task, execution_git_head="b" * 40)
    shard["task_sha256"] = "c" * 64
    assert not runner._valid_shard(shard, task=task, execution_git_head="b" * 40)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("canonical_solvent", "hexane"),
        ("configuration_sha256", "not-a-sha"),
        ("predicted_delta_g_kcal_mol", 0.0),
        ("absolute_error_kcal_mol", float("nan")),
    ),
)
def test_resume_shard_rejects_identity_hash_and_ledger_corruption(
    field: str, replacement: object
):
    task = {
        "selection_index": 0,
        "partition_selection_index": 0,
        "task_sha256": "a" * 64,
        "opaque_record_id": "private-record",
        "partition": "confirmation",
        "geometry_handle": "private-geometry",
        "geometry_sha256": "b" * 64,
        "canonical_solvent": "water",
        "atom_count": 1,
        "factor_count": 0,
        "required_algebraic_degree": 8,
        "reference_ddpcm_energy_hartree": -0.005,
        "smd_cds_energy_kcal_mol": 0.25,
        "experimental_delta_g_kcal_mol": -1.5,
    }
    continuum_ev = -0.1
    continuum_kcal = continuum_ev / runner.HARTREE_TO_EV * runner.HARTREE_TO_KCAL_MOL
    reference_ev = task["reference_ddpcm_energy_hartree"] * runner.HARTREE_TO_EV
    parity_ev = continuum_ev - reference_ev
    parity_kcal = parity_ev / runner.HARTREE_TO_EV * runner.HARTREE_TO_KCAL_MOL
    predicted = continuum_kcal + task["smd_cds_energy_kcal_mol"]
    signed_error = predicted - task["experimental_delta_g_kcal_mol"]
    shard = {
        "artifact": runner.ARTIFACT,
        "schema_version": 1,
        "do_not_commit": True,
        "execution_git_head": "c" * 40,
        "selection_index": 0,
        "partition_selection_index": 0,
        "task_sha256": "a" * 64,
        "opaque_record_id": "private-record",
        "partition": "confirmation",
        "geometry_handle": "private-geometry",
        "geometry_sha256": "b" * 64,
        "canonical_solvent": "water",
        "atom_count": 1,
        "factor_count": 0,
        "required_algebraic_degree": 8,
        "continuum_energy_ev": continuum_ev,
        "continuum_energy_kcal_mol": continuum_kcal,
        "reference_ddpcm_energy_ev": reference_ev,
        "smooth_minus_reference_ev": parity_ev,
        "smooth_minus_reference_kcal_mol": parity_kcal,
        "smd_cds_energy_kcal_mol": task["smd_cds_energy_kcal_mol"],
        "experimental_delta_g_kcal_mol": task["experimental_delta_g_kcal_mol"],
        "predicted_delta_g_kcal_mol": predicted,
        "signed_error_kcal_mol": signed_error,
        "absolute_error_kcal_mol": abs(signed_error),
        "configuration_sha256": "d" * 64,
        "provenance_sha256": "e" * 64,
        "timing_seconds": 0.5,
    }
    shard[field] = replacement

    assert not runner._valid_shard(shard, task=task, execution_git_head="c" * 40)


def test_public_guard_rejects_restricted_row_fields():
    runner._assert_public_safe({"aggregate_metrics": {"record_count": 653}})
    with pytest.raises(ValueError, match="leaks private keys"):
        runner._assert_public_safe({"nested": {"geometry_handle": "private"}})
