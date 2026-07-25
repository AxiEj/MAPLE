from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_route1_freesolv_reserve as reserve

PROTOCOL_PATH = BENCHMARK_DIR / "route1_freesolv_reserve_protocol.json"
MANIFEST_PATH = BENCHMARK_DIR / "route1_freesolv_reserve_source_manifest.json"
ENERGY_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-energy-2026-07-25.json"
SCORE_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-score-2026-07-25.json"


def test_protocol_preserves_route1_and_pre_registers_no_tuning_rule():
    protocol, fingerprint = reserve.load_protocol(PROTOCOL_PATH)

    assert len(fingerprint) == 64
    assert protocol["route1_boundary"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    assert protocol["evaluation_design"]["historical_label_exposure"] is True
    assert protocol["evaluation_design"]["independent_blind_confirmation"] is False
    assert protocol["evaluation_design"]["energy_phase_reads_labels"] is False
    assert protocol["evaluation_design"]["no_post_score_tuning"] is True
    rule = protocol["pre_registered_decision_rule"]
    assert rule["required_case_count"] == 116
    assert rule["retain_chagb_sp_accuracy_profile_only_if"][
        "paired_mean_mae_gain_min_kcal_mol"
    ] == pytest.approx(0.15)
    assert rule["production_default_may_change"] is False
    assert rule["runtime_force_capability_may_be_inferred"] is False


def test_frozen_source_manifest_is_label_free_and_complete():
    protocol, _fingerprint = reserve.load_protocol(PROTOCOL_PATH)
    manifest = reserve.load_source_manifest(
        PROTOCOL_PATH, protocol, MANIFEST_PATH
    )

    assert manifest["case_count"] == 116
    assert len(manifest["records"]) == 116
    assert len({row["compound_id"] for row in manifest["records"]}) == 116
    assert "experimental" not in json.dumps(manifest).lower()
    assert core.sha256_file(MANIFEST_PATH) == protocol["source_evidence"][
        "source_manifest_sha256"
    ]


def test_sealed_energy_artifact_is_complete_and_label_free():
    protocol, fingerprint = reserve.load_protocol(PROTOCOL_PATH)
    artifact = reserve.load_energy_artifact(
        ENERGY_PATH,
        protocol_path=PROTOCOL_PATH,
        protocol=protocol,
        fingerprint=fingerprint,
        manifest_path=MANIFEST_PATH,
    )

    assert artifact["case_count"] == 116
    assert len(artifact["records"]) == 116
    assert "experimental" not in json.dumps(artifact).lower()
    assert core.sha256_file(ENERGY_PATH) == (
        "5439b65f047d0aafa14282dfa47df8f0a05d5d965cce9cc3a52caf5b5eec3c09"
    )
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_route1_freesolv_reserve.py"
    )
    for record in artifact["records"]:
        components = record["components_kcal_mol"]
        predictions = record["predictions_kcal_mol"]
        assert predictions["am1bcc_obc2_ace"] == pytest.approx(
            components["obc2_polar"] + components["ace_nonpolar"]
        )
        assert predictions["am1bcc_chagb_pbsa_cavity_dispersion"] == pytest.approx(
            components["chagb_polar"]
            + components["pbsa_cavity"]
            + components["pbsa_dispersion"]
        )


def test_score_artifact_reconciles_pre_registered_gates():
    protocol, fingerprint = reserve.load_protocol(PROTOCOL_PATH)
    score = core.load_json(SCORE_PATH)

    assert core.artifact_content_sha256(score) == score["content_sha256"]
    assert core.sha256_file(SCORE_PATH) == (
        "2503405c47bd87a237e65c686950907783e2d3848f30a9cab60bdf0ae31ebfef"
    )
    assert score["protocol_fingerprint"] == fingerprint
    assert score["case_count"] == 116
    assert score["sealed_energy_artifact_sha256"] == core.sha256_file(ENERGY_PATH)
    assert score["evaluation_design"]["independent_blind_confirmation"] is False
    for endpoint in reserve.ENDPOINTS:
        assert score["methods"][endpoint]["n"] == 116
        assert score["methods"][endpoint]["failure_count"] == 0
    gates = score["decision"]["candidate_gate_results"]
    assert score["decision"]["chagb_sp_accuracy_profile_retained"] is all(
        gates.values()
    )
    assert score["decision"]["production_default_changed"] is False
    assert score["decision"]["runtime_force_capability_established"] is False
    assert score["decision"]["post_score_tuning_allowed"] is False
    assert score["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_route1_freesolv_reserve.py"
    )


def test_frozen_reserve_metrics_and_paired_interval():
    score = core.load_json(SCORE_PATH)
    baseline = score["methods"]["am1bcc_obc2_ace"]
    candidate = score["methods"]["am1bcc_chagb_pbsa_cavity_dispersion"]
    paired = score["paired_absolute_error_gain"]

    assert baseline["mae"] == pytest.approx(1.7909349293140766)
    assert baseline["rmse"] == pytest.approx(2.464251419348348)
    assert baseline["max_absolute_error"] == pytest.approx(9.622027088379586)
    assert candidate["mae"] == pytest.approx(1.3008163793103444)
    assert candidate["rmse"] == pytest.approx(1.8456267151865633)
    assert candidate["max_absolute_error"] == pytest.approx(6.3763000000000005)
    assert paired["mean_mae_gain_kcal_mol"] == pytest.approx(0.4901185500037324)
    assert paired["bootstrap_ci"] == pytest.approx(
        [0.21460197835681527, 0.7821405195061892]
    )
    assert paired["case_outcomes"] == {
        "improved": 77,
        "unchanged": 0,
        "worsened": 39,
    }


def test_reserve_docs_preserve_scope_and_decision():
    paths = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
        BENCHMARK_DIR / "README.md",
    )
    normalized = " ".join(
        " ".join(path.read_text(encoding="utf-8").split()) for path in paths
    )

    assert "label-exposed, held-out-by-computation reserve" in normalized
    assert "1.791/2.464/9.622" in normalized
    assert "1.301/1.846/6.376" in normalized
    assert "0.490 kcal/mol" in normalized
    assert "SP-only fixed-geometry accuracy profile" in normalized
    assert "OBC-II/ACE remains the force-capable default" in normalized
    assert "independent confirmation gate remains open" in normalized
