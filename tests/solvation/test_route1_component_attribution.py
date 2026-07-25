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
import run_route1_component_attribution as attribution

PROTOCOL_PATH = BENCHMARK_DIR / "route1_component_attribution_protocol.json"
ARTIFACT_PATH = (
    BENCHMARK_DIR / "route1-chagb-component-attribution-2026-07-25.json"
)


def test_component_attribution_protocol_preserves_route1_boundary():
    protocol = attribution.load_protocol(PROTOCOL_PATH)

    assert protocol["route1_boundary"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    assert protocol["label_use_boundary"] == {
        "component_energies_were_frozen_before_this_analysis": True,
        "development_labels_are_read": True,
        "experimental_fit_or_residual": False,
        "confirmation_remains_closed": True,
    }
    assert protocol["decision_rule"]["runtime_provider_change_allowed"] is False


def test_component_attribution_protocol_fails_closed_on_residual(
    tmp_path: Path,
):
    protocol = core.load_json(PROTOCOL_PATH)
    protocol["route1_boundary"]["hydration_label_residual"] = True
    protocol["label_use_boundary"]["experimental_fit_or_residual"] = True
    path = tmp_path / "invalid-component-attribution.json"
    core.write_json_atomic(path, protocol)

    with pytest.raises(ValueError, match="Route 1 boundary|residual"):
        attribution.load_protocol(path)


def test_frozen_component_attribution_artifact_is_complete_and_reproducible():
    artifact = core.load_json(ARTIFACT_PATH)

    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["protocol_sha256"] == core.sha256_file(PROTOCOL_PATH)
    assert artifact["command_provenance"]["script"] == (
        "docs/implicit-solvation/benchmarks/run_route1_component_attribution.py"
    )
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_route1_component_attribution.py"
    )
    assert artifact["case_count"] == 526
    assert len(artifact["records"]) == 526
    assert len({record["compound_id"] for record in artifact["records"]}) == 526
    assert artifact["source_partition"] == "development"
    assert artifact["label_use_boundary"]["experimental_fit_or_residual"] is False
    for method in attribution.METHODS:
        assert artifact["methods"][method]["n"] == 526
        assert artifact["methods"][method]["failure_count"] == 0


@pytest.mark.parametrize(
    ("method", "mae", "rmse", "maximum"),
    [
        ("obc2_ace", 1.7603512076636894, 2.536883433943894, 13.550422693879428),
        (
            "obc2_pbsa_cavity_dispersion",
            2.1031247378245204,
            3.0106473949765307,
            15.784010745573193,
        ),
        ("chagb_ace", 1.8194582574545626, 2.4119951759962115, 10.03563337164332),
        (
            "chagb_surface_tension",
            1.4494134980988593,
            1.975307296818173,
            9.5998,
        ),
        (
            "chagb_pbsa_cavity_dispersion",
            1.321851711026616,
            1.8542079387884627,
            9.416399999999994,
        ),
        ("apbs_ace", 1.6288990168391229, 2.46221030127189, 14.1155090678913),
        (
            "apbs_pbsa_cavity_dispersion",
            2.1611029059850537,
            3.142554064811233,
            16.31029898826481,
        ),
    ],
)
def test_frozen_component_endpoint_metrics(method, mae, rmse, maximum):
    metrics = core.load_json(ARTIFACT_PATH)["methods"][method]

    assert metrics["mae"] == pytest.approx(mae)
    assert metrics["rmse"] == pytest.approx(rmse)
    assert metrics["max_absolute_error"] == pytest.approx(maximum)


def test_isolated_component_swaps_fail_but_joint_pairing_passes():
    artifact = core.load_json(ARTIFACT_PATH)
    paired = artifact["paired_absolute_error_gain_vs_obc2_ace"]

    polar_only = paired["chagb_ace"]
    assert polar_only["mean_mae_gain_kcal_mol"] == pytest.approx(
        -0.05910704979087306
    )
    assert polar_only["bootstrap_ci"][0] < 0.0 < polar_only["bootstrap_ci"][1]

    nonpolar_only = paired["obc2_pbsa_cavity_dispersion"]
    assert nonpolar_only["mean_mae_gain_kcal_mol"] == pytest.approx(
        -0.34277353016083095
    )
    assert nonpolar_only["bootstrap_ci"][1] < 0.0

    joint = paired["chagb_pbsa_cavity_dispersion"]
    assert joint["mean_mae_gain_kcal_mol"] == pytest.approx(
        0.43849949663707327
    )
    assert joint["bootstrap_ci"][0] > 0.0

    decision = artifact["decision"]
    assert decision["isolated_chagb_polar_swap_passed"] is False
    assert decision["isolated_pbsa_nonpolar_swap_passed"] is False
    assert decision["joint_chagb_pbsa_swap_passed_development_gate"] is True
    assert decision["co_calibrated_pairing_evidence"] is True
    assert decision["runtime_provider_added"] is False
    assert decision["production_default_changed"] is False
    assert decision["confirmation_remains_closed"] is True


def test_shapley_attribution_closes_without_claiming_microscopic_causality():
    result = core.load_json(ARTIFACT_PATH)["factorial_shapley_attribution"]

    assert result["total_joint_mae_gain_kcal_mol"] == pytest.approx(
        0.43849949663707344
    )
    assert result["polar_swap_shapley_gain_kcal_mol"] == pytest.approx(
        0.3610829885035156
    )
    assert result["nonpolar_swap_shapley_gain_kcal_mol"] == pytest.approx(
        0.07741650813355783
    )
    assert result["marginal_gain_interaction_kcal_mol"] == pytest.approx(
        0.8403800765887777
    )
    assert result["closure_residual_kcal_mol"] == pytest.approx(0.0, abs=1.0e-15)
    assert "do not prove microscopic causality" in result["interpretation"]


def test_route1_docs_freeze_the_joint_pairing_conclusion():
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

    assert "Route 1 component-attribution audit" in normalized
    assert "route1-chagb-component-attribution-2026-07-25.json" in normalized
    assert "2.103/3.011" in normalized
    assert "1.819/2.412" in normalized
    assert "1.322/1.854" in normalized
    assert "0.361/0.077 kcal/mol" in normalized
    assert "does not prove microscopic causality" in normalized
    assert "neither isolated component swap is promoted" in normalized


def test_component_attribution_artifact_file_has_stable_sha256_shape():
    assert len(hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()) == 64
    assert json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))["case_count"] == 526
