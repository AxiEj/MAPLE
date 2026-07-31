from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_route2_direct_pcm_freesolv12_paired as runner  # noqa: E402
from route2_v0_freesolv12_functional_groups import (  # noqa: E402
    load_freesolv12_functional_group_manifest,
)


def _row(compound_id: str, *, pcm: float, cosmo: float) -> dict[str, object]:
    return {
        "compound_id": compound_id,
        "methods": {
            "ddpcm": {
                "predicted_kcal_mol": pcm,
                "signed_error_kcal_mol": pcm,
                "absolute_error_kcal_mol": abs(pcm),
            },
            "ddcosmo": {
                "predicted_kcal_mol": cosmo,
                "signed_error_kcal_mol": cosmo,
                "absolute_error_kcal_mol": abs(cosmo),
            },
        },
    }


def test_runner_is_locked_to_two_direct_pcm_equation_profiles():
    assert runner.METHOD_PROFILES == (
        ("ddpcm", "smd-ddpcm-l15-n1202-multisolv-pcm-half-coupling-v2"),
        ("ddcosmo", "smd-ddcosmo-l15-n1202-multisolv-pcm-half-coupling-v2"),
    )
    assert runner.ARTIFACT == "route2-direct-pcm-freesolv12-ddpcm-ddcosmo-v2"
    assert runner.SCHEMA_VERSION == 2
    assert "0.5*<c_MACE-POLAR, f_reac_PCM>" in (
        runner.paired_benchmark.paired_energy_composition(
            runner.PCM_HALF_COUPLING_ONLY_V1
        )
    )


def test_locked_panel_really_has_twelve_records_and_ten_functional_groups():
    manifest = load_freesolv12_functional_group_manifest()
    records = manifest["locked_records"]

    assert len(records) == 12
    assert len({record["functional_group"] for record in records if record["functional_group"]}) == 10


def test_metrics_and_pair_comparison_keep_all_rows():
    rows = [
        _row("first", pcm=1.0, cosmo=0.5),
        _row("second", pcm=-2.0, cosmo=-3.0),
    ]

    pcm = runner._error_metrics(rows, "ddpcm")
    comparison = runner._paired_error_comparison(rows)

    assert pcm["record_count"] == 2
    assert pcm["mae_kcal_mol"] == pytest.approx(1.5)
    assert pcm["records_at_or_above_1_5_kcal_mol"] == 1
    assert comparison["record_count"] == 2
    assert comparison["ddcosmo_lower_absolute_error_count"] == 1
    assert comparison["ddpcm_lower_absolute_error_count"] == 1


def test_mol2_resolution_accepts_only_the_locked_member_and_hash(tmp_path):
    record = {
        "compound_id": "locked",
        "mol2_archive_member": "mol2files_gaff/locked.mol2",
        "mol2_sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    }
    member = tmp_path / "mol2files_gaff" / "locked.mol2"
    member.parent.mkdir()
    member.write_text("hello", encoding="utf-8")

    assert runner._resolve_mol2_path(tmp_path, record) == member
    record["mol2_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity drifted"):
        runner._resolve_mol2_path(tmp_path, record)


def test_failed_rows_require_explicit_retry(tmp_path):
    row = tmp_path / "row.json"
    assert runner._record_needs_run(row, retry_failures=False) is True
    row.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    assert runner._record_needs_run(row, retry_failures=True) is False
    row.write_text(json.dumps({"status": "failure"}), encoding="utf-8")
    assert runner._record_needs_run(row, retry_failures=False) is False
    assert runner._record_needs_run(row, retry_failures=True) is True
