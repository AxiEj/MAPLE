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
    task = {"selection_index": 7, "task_sha256": "a" * 64}
    shard = {
        "artifact": runner.ARTIFACT,
        "schema_version": 1,
        "do_not_commit": True,
        "execution_git_head": "b" * 40,
        "selection_index": 7,
        "task_sha256": "a" * 64,
        "continuum_energy_ev": -0.2,
    }

    assert runner._valid_shard(shard, task=task, execution_git_head="b" * 40)
    shard["task_sha256"] = "c" * 64
    assert not runner._valid_shard(shard, task=task, execution_git_head="b" * 40)


def test_public_guard_rejects_restricted_row_fields():
    runner._assert_public_safe({"aggregate_metrics": {"record_count": 653}})
    with pytest.raises(ValueError, match="leaks private keys"):
        runner._assert_public_safe({"nested": {"geometry_handle": "private"}})
