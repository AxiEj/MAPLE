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
import run_two_level_final_sp as two_level

PROTOCOL_PATH = BENCHMARK_DIR / "two_level_final_sp_protocol.json"
ENERGY_PATH = BENCHMARK_DIR / "route1-two-level-final-sp-rerank-2026-07-24.json"
SUMMARY_PATH = (
    BENCHMARK_DIR / "route1-two-level-final-sp-rerank-summary-2026-07-24.json"
)


def test_two_level_protocol_exposes_both_potentials_and_label_status():
    protocol = two_level.load_protocol(PROTOCOL_PATH)

    assert protocol["route1_boundary"] == {
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    assert protocol["derivative_contract"] == {
        "optimization_potential_has_consistent_forces": True,
        "final_sp_potential_has_a_runtime_force_claim": False,
        "same_potential_for_optimization_and_final_sp": False,
        "both_potentials_must_be_reported": True,
    }
    assert protocol["label_boundary"][
        "source_selection_and_input_records_are_label_exposed"
    ]
    assert not protocol["label_boundary"][
        "experimental_labels_used_for_fit_or_residual"
    ]


def test_two_level_protocol_rejects_a_gas_phase_mm_term(tmp_path: Path):
    protocol = core.load_json(PROTOCOL_PATH)
    protocol["route1_boundary"]["gas_phase_mm_energy"] = True
    altered = tmp_path / "two-level-with-mm.json"
    altered.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen Route 1 boundary"):
        two_level.load_protocol(altered)


def test_two_level_mol2_rewrite_changes_only_coordinates_and_charges():
    source = """@<TRIPOS>MOLECULE
example
 2 1 1 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
1 C1 0.0 1.0 2.0 c3 1 MOL -0.1
2 O1 1.2 1.0 2.0 oh 1 MOL -0.5
@<TRIPOS>BOND
1 1 2 1
"""
    rewritten = two_level.mol2_with_positions_and_charges(
        source,
        [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
        [0.25, -0.25],
    )
    atom_lines = (
        rewritten.split("@<TRIPOS>ATOM\n", 1)[1]
        .split("@<TRIPOS>BOND", 1)[0]
        .splitlines()
    )

    assert atom_lines[0].split()[2:9] == [
        "3.0000000000",
        "4.0000000000",
        "5.0000000000",
        "c3",
        "1",
        "MOL",
        "0.250000000000",
    ]
    assert atom_lines[1].split()[2:9] == [
        "6.0000000000",
        "7.0000000000",
        "8.0000000000",
        "oh",
        "1",
        "MOL",
        "-0.250000000000",
    ]


def test_frozen_two_level_energy_artifact_preserves_the_energy_only_boundary():
    artifact = core.load_json(ENERGY_PATH)

    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["protocol_sha256"] == core.sha256_file(PROTOCOL_PATH)
    assert artifact["case_count"] == 6
    assert artifact["candidate_count"] == 12
    assert len(artifact["records"]) == 6
    assert "experimental_kcal_mol" not in json.dumps(
        artifact["records"], sort_keys=True
    )
    assert sum(record["selection_changed"] for record in artifact["records"]) == 1
    for record in artifact["records"]:
        for candidate in record["candidates"]:
            assert candidate["final_sp_solvation_kcal_mol"] == pytest.approx(
                sum(candidate["components_kcal_mol"].values())
            )
            assert candidate["two_level_relaxed_transfer_kcal_mol"] == pytest.approx(
                candidate["gas_reorganization_cost_kcal_mol"]
                + candidate["final_sp_solvation_kcal_mol"]
            )


def test_frozen_two_level_summary_does_not_promote_a_negative_probe():
    summary = core.load_json(SUMMARY_PATH)

    assert core.artifact_content_sha256(summary) == summary["content_sha256"]
    assert summary["energy_artifact_sha256"] == core.sha256_file(ENERGY_PATH)
    assert summary["case_count"] == 6
    assert summary["selection_changed_case_count"] == 1
    assert summary["metrics"]["low_level_relaxed"]["mae_kcal_mol"] == pytest.approx(
        2.8344577856260673
    )
    assert summary["metrics"]["fixed_geometry_high_level"][
        "mae_kcal_mol"
    ] == pytest.approx(1.7726166666666672)
    assert summary["metrics"]["two_level_low_selected"][
        "mae_kcal_mol"
    ] == pytest.approx(1.8705286727975927)
    assert summary["metrics"]["two_level_high_reranked"][
        "mae_kcal_mol"
    ] == pytest.approx(1.9010846077411223)
    comparison = summary["paired_comparisons"][
        "fixed_geometry_high_level_to_two_level_high_reranked"
    ]
    assert comparison["mean_absolute_error_gain_kcal_mol"] == pytest.approx(
        -0.12846794107445494
    )
    assert comparison["improved_case_count"] == 0
    assert comparison["worsened_case_count"] == 6
    assert summary["decision"]["disposition"] == "do-not-promote-two-level-default"
    assert summary["decision"]["product_default_changed"] is False
    assert summary["decision"]["force_claim_for_final_sp"] is False


@pytest.mark.parametrize("artifact_path", [ENERGY_PATH, SUMMARY_PATH])
def test_two_level_artifacts_pin_the_runner_that_created_them(
    artifact_path: Path,
):
    artifact = core.load_json(artifact_path)
    command = artifact["command_provenance"]
    script = REPOSITORY_ROOT / command["script"]

    assert command["script"] == (
        "docs/implicit-solvation/benchmarks/run_two_level_final_sp.py"
    )
    assert command["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
