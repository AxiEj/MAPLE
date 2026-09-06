from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core  # pyright: ignore[reportMissingImports]
import build_matched_series as series_builder  # pyright: ignore[reportMissingImports]
import ranking_metrics  # pyright: ignore[reportMissingImports]
import run_route1_rank_reserve as rank_reserve  # pyright: ignore[reportMissingImports]


PROTOCOL_PATH = BENCHMARK_DIR / "route1_rank_protocol_v1.json"
SOURCE_MANIFEST_PATH = BENCHMARK_DIR / "route1_freesolv_reserve_source_manifest.json"
SCORE_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-score-2026-07-25.json"
SERIES_PATH = (
    BENCHMARK_DIR / "route1-freesolv-reserve-rank-series-2026-07-29.json"
)
DIAGNOSTIC_PATH = (
    BENCHMARK_DIR / "route1-freesolv-reserve-rank-diagnostic-2026-07-29.json"
)


def _write_mol2(path: Path, name: str) -> None:
    path.write_text(
        f"""@<TRIPOS>MOLECULE
{name}
12 12 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
1 C1 1.0 0.0 0.0 ca 1 MOL 0.0
2 C2 0.5 0.9 0.0 ca 1 MOL 0.0
3 C3 -0.5 0.9 0.0 ca 1 MOL 0.0
4 C4 -1.0 0.0 0.0 ca 1 MOL 0.0
5 C5 -0.5 -0.9 0.0 ca 1 MOL 0.0
6 C6 0.5 -0.9 0.0 ca 1 MOL 0.0
7 H1 2.0 0.0 0.0 ha 1 MOL 0.0
8 H2 1.0 1.8 0.0 ha 1 MOL 0.0
9 H3 -1.0 1.8 0.0 ha 1 MOL 0.0
10 H4 -2.0 0.0 0.0 ha 1 MOL 0.0
11 H5 -1.0 -1.8 0.0 ha 1 MOL 0.0
12 H6 1.0 -1.8 0.0 ha 1 MOL 0.0
@<TRIPOS>BOND
1 1 2 ar
2 2 3 ar
3 3 4 ar
4 4 5 ar
5 5 6 ar
6 6 1 ar
7 1 7 1
8 2 8 1
9 3 9 1
10 4 10 1
11 5 11 1
12 6 12 1
""",
        encoding="utf-8",
    )


def test_rank_metrics_handle_ties_and_directional_pairs():
    assert ranking_metrics.kendall_tau_b([0.0, 0.0, 1.0], [0.0, 1.0, 1.0]) == pytest.approx(
        0.5
    )
    assert ranking_metrics.spearman_rho([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]) == pytest.approx(
        0.5
    )

    metrics = ranking_metrics.series_ranking_metrics(
        compound_ids=["a", "b", "c"],
        experimental=[-4.0, -3.0, -1.0],
        experimental_uncertainty=[0.1, 0.1, 0.1],
        predicted=[-5.0, -2.0, -3.0],
        fixed_delta_thresholds=[0.5, 1.0],
        uncertainty_z=1.96,
        top_ks=[1, 3],
        confidence_margin_thresholds=[0.0, 2.0],
    )

    assert metrics["kendall_tau_b"] == pytest.approx(1.0 / 3.0)
    assert metrics["spearman_rho"] == pytest.approx(0.5)
    assert metrics["pair_metrics"]["fixed_abs_delta_gte_1.0"]["pair_count"] == 3
    assert metrics["pair_metrics"]["fixed_abs_delta_gte_1.0"][
        "forced_sign_accuracy"
    ] == pytest.approx(2.0 / 3.0)
    assert metrics["top_k"]["1"]["overlap_count"] == 1
    assert metrics["top_k"]["1"]["best_of_predicted_top_k_regret"] == pytest.approx(
        0.0
    )
    assert metrics["coverage_risk_curve"]["2.0"]["coverage"] == pytest.approx(
        2.0 / 3.0
    )


def test_rank_metrics_exclude_experimentally_indistinguishable_pairs():
    metrics = ranking_metrics.series_ranking_metrics(
        compound_ids=["a", "b", "c"],
        experimental=[0.0, 0.3, 3.0],
        experimental_uncertainty=[0.2, 0.2, 0.2],
        predicted=[0.0, 1.0, 2.0],
        fixed_delta_thresholds=[0.5, 1.0],
        uncertainty_z=1.96,
        top_ks=[1],
        confidence_margin_thresholds=[0.0],
    )

    uncertainty_gate = metrics["pair_metrics"]["uncertainty_z_1.96"]
    assert uncertainty_gate["pair_count"] == 2
    assert uncertainty_gate["forced_sign_accuracy"] == pytest.approx(1.0)
    assert metrics["pair_metrics"]["fixed_abs_delta_gte_0.5"]["pair_count"] == 2


def test_murcko_series_builder_is_label_free_and_groups_structure_only(tmp_path):
    source_root = tmp_path / "source"
    molecule_dir = source_root / "mol2"
    molecule_dir.mkdir(parents=True)
    first = molecule_dir / "first.mol2"
    second = molecule_dir / "second.mol2"
    _write_mol2(first, "first")
    _write_mol2(second, "second")
    manifest = {
        "case_count": 2,
        "records": [
            {
                "compound_id": "first",
                "source_mol2_relative_path": "mol2/first.mol2",
                "source_mol2_sha256": core.sha256_file(first),
            },
            {
                "compound_id": "second",
                "source_mol2_relative_path": "mol2/second.mol2",
                "source_mol2_sha256": core.sha256_file(second),
            },
        ],
    }
    source_manifest = tmp_path / "source-manifest.json"
    core.write_json_atomic(source_manifest, manifest)

    artifact = series_builder.build_series_manifest(
        source_manifest=source_manifest,
        source_root=source_root,
        expected_source_manifest_sha256=core.sha256_file(source_manifest),
        min_series_size=2,
        protocol_id="test-route1-rank",
        protocol_sha256="0" * 64,
        protocol_fingerprint="1" * 64,
    )

    assert artifact["case_count"] == 2
    assert artifact["evaluable_series_count"] == 1
    assert artifact["evaluable_member_count"] == 2
    assert artifact["records"][0]["series_id"] == artifact["records"][1]["series_id"]
    assert "experimental" not in json.dumps(artifact).lower()

    manifest["records"][0]["experimental_kcal_mol"] = -5.0
    core.write_json_atomic(source_manifest, manifest)
    with pytest.raises(ValueError, match="label field"):
        series_builder.build_series_manifest(
            source_manifest=source_manifest,
            source_root=source_root,
            expected_source_manifest_sha256=core.sha256_file(source_manifest),
            min_series_size=2,
            protocol_id="test-route1-rank",
            protocol_sha256="0" * 64,
            protocol_fingerprint="1" * 64,
        )


def test_series_builder_cli_initializes_murcko_before_main(tmp_path):
    output = tmp_path / "series.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_DIR / "build_matched_series.py"),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = core.load_json(output)
    assert artifact["evaluable_series_count"] == 7
    assert artifact["evaluable_member_count"] == 48
    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]


def test_frozen_route1_rank_contract_and_diagnostic_preserve_boundaries():
    protocol, fingerprint = rank_reserve.load_protocol(PROTOCOL_PATH)
    series = core.load_json(SERIES_PATH)
    diagnostic = core.load_json(DIAGNOSTIC_PATH)

    assert protocol["evaluation_design"]["series_disjoint_split"] is False
    assert protocol["evaluation_design"]["certified_ranking_available"] is False
    assert core.sha256_file(SOURCE_MANIFEST_PATH) == protocol["source_evidence"][
        "source_manifest_sha256"
    ]
    assert core.sha256_file(SCORE_PATH) == protocol["source_evidence"][
        "score_artifact_sha256"
    ]
    assert core.artifact_content_sha256(series) == series["content_sha256"]
    assert core.artifact_content_sha256(diagnostic) == diagnostic["content_sha256"]
    assert series["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "build_matched_series.py"
    )
    assert diagnostic["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_route1_rank_reserve.py"
    )
    assert series["case_count"] == 116
    assert series["evaluable_series_count"] == 7
    assert series["evaluable_member_count"] == 48
    assert diagnostic["protocol_fingerprint"] == fingerprint
    assert diagnostic["case_count"] == 116
    assert diagnostic["evaluable_series_count"] == 7
    assert diagnostic["decision"]["certified_ranking_available"] is False
    assert diagnostic["decision"]["endpoint_selection_allowed"] is False
    assert diagnostic["decision"]["reserve_tuning_allowed"] is False
    assert "experimental_kcal_mol" not in json.dumps(diagnostic).lower()


def test_frozen_route1_rank_diagnostic_reports_macro_not_benzene_weighted_metrics():
    diagnostic = core.load_json(DIAGNOSTIC_PATH)
    baseline = diagnostic["methods"]["am1bcc_obc2_ace"]
    candidate = diagnostic["methods"]["am1bcc_chagb_pbsa_cavity_dispersion"]

    assert baseline["series_macro"]["kendall_tau_b"] == pytest.approx(
        0.09254725772772081
    )
    assert candidate["series_macro"]["kendall_tau_b"] == pytest.approx(
        0.37537845130095604
    )
    assert baseline["series_macro"]["spearman_rho"] == pytest.approx(
        0.12176440602614982
    )
    assert candidate["series_macro"]["spearman_rho"] == pytest.approx(
        0.433169671997267
    )
    paired = diagnostic["paired_series_macro_change"]
    assert paired["kendall_tau_b"]["candidate_minus_baseline"] == pytest.approx(
        0.28283119357323526
    )
    assert paired["spearman_rho"]["candidate_minus_baseline"] == pytest.approx(
        0.3114052659711172
    )
