from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
RUNNER_PATH = BENCHMARK_DIR / "run_route1_explicit_component_diagnostic.py"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_explicit_component_diagnostic_protocol_v1.json"
COMPONENT_ARTIFACT_PATH = BENCHMARK_DIR / "route1-chagb-component-attribution-2026-07-25.json"
FREESOLV_DATABASE_PATH = (
    REPOSITORY_ROOT
    / ".omx/vendor-audits/freesolv-current-20260726-v1/database.json"
)
FROZEN_OUTPUT = BENCHMARK_DIR / "route1-freesolv-explicit-component-diagnostic-2026-07-29.json"

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "route1_explicit_component_diagnostic", RUNNER_PATH
)
assert RUNNER_SPEC is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def _run(
    tmp_path: Path,
    *,
    protocol: Path = PROTOCOL_PATH,
    component_artifact: Path = COMPONENT_ARTIFACT_PATH,
    freesolv_database: Path = FREESOLV_DATABASE_PATH,
    output_name: str = "route1-explicit-component-diagnostic-test.json",
):
    return runner.run(
        argparse.Namespace(
            protocol=protocol,
            component_artifact=component_artifact,
            freesolv_database=freesolv_database,
            output=tmp_path / output_name,
        )
    )


def _write_json(path: Path, value: object) -> Path:
    runner.core.write_json_atomic(path, value)
    return path


def _aligned_protocol(
    tmp_path: Path,
    *,
    component_artifact: Path = COMPONENT_ARTIFACT_PATH,
    freesolv_database: Path = FREESOLV_DATABASE_PATH,
) -> Path:
    protocol = copy.deepcopy(runner.core.load_json(PROTOCOL_PATH))
    component = runner.core.load_json(component_artifact)
    protocol["source_evidence"] = {
        **protocol["source_evidence"],
        "component_artifact_sha256": runner.core.sha256_file(component_artifact),
        "component_artifact_content_sha256": runner.core.artifact_content_sha256(component),
        "freesolv_database_sha256": runner.core.sha256_file(freesolv_database),
    }
    return _write_json(tmp_path / "aligned_protocol.json", protocol)


def test_explicit_component_diagnostic_reproduces_frozen_result(tmp_path: Path) -> None:
    artifact = _run(tmp_path)
    frozen = runner.core.load_json(FROZEN_OUTPUT)

    comparable = copy.deepcopy(artifact)
    comparable.pop("content_sha256")
    comparable.pop("command_provenance")
    frozen_comparable = copy.deepcopy(frozen)
    frozen_comparable.pop("content_sha256")
    frozen_comparable.pop("command_provenance")
    assert comparable == frozen_comparable
    assert artifact["case_count"] == frozen["case_count"] == 526
    assert artifact["component_comparison"] == frozen["component_comparison"]
    assert artifact["label_exposed_error_decomposition"] == frozen[
        "label_exposed_error_decomposition"
    ]
    assert artifact["component_dominance"] == frozen["component_dominance"]
    assert artifact["polar_model_comparison_against_published_charging"] == frozen[
        "polar_model_comparison_against_published_charging"
    ]
    assert artifact["descriptive_associations"] == frozen["descriptive_associations"]


def test_explicit_component_diagnostic_reports_paired_cha_not_radius_rollback_evidence(
    tmp_path: Path,
) -> None:
    artifact = _run(tmp_path)
    comparison = artifact["polar_model_comparison_against_published_charging"]
    all_cases = comparison["paired_absolute_mismatch_reduction_chagb_minus_obc2"]
    tail_cases = comparison[
        "route_error_tail_paired_absolute_mismatch_reduction_chagb_minus_obc2"
    ]

    assert artifact["component_comparison"][
        "obc2_polar_minus_published_charging"
    ]["mean_absolute_kcal_mol"] == pytest.approx(2.0695804333327033)
    assert all_cases == {
        "case_count": 526,
        "mean_absolute_mismatch_reduction_kcal_mol": pytest.approx(1.1903374675532357),
        "chagb_lower_absolute_mismatch_count": 458,
        "obc2_lower_absolute_mismatch_count": 68,
        "exact_tie_count": 0,
    }
    assert tail_cases == {
        "case_count": 170,
        "mean_absolute_mismatch_reduction_kcal_mol": pytest.approx(1.48648232795682),
        "chagb_lower_absolute_mismatch_count": 147,
        "obc2_lower_absolute_mismatch_count": 23,
        "exact_tie_count": 0,
    }


def test_explicit_component_diagnostic_is_self_sealed_and_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path, output_name="same.json")
    second = _run(tmp_path, output_name="same.json")

    assert first["content_sha256"] == second["content_sha256"]
    assert first["content_sha256"] == runner.core.artifact_content_sha256(first)
    assert first["component_comparison"]["component_identity_residual"][
        "maximum_absolute_kcal_mol"
    ] < 2e-12
    assert first["component_comparison"]["published_component_rounding_residual"][
        "maximum_absolute_kcal_mol"
    ] <= 0.0011


def test_explicit_component_diagnostic_keeps_records_free_of_experimental_values(
    tmp_path: Path,
) -> None:
    artifact = _run(tmp_path)
    assert len(artifact["records"]) == 526
    for record in artifact["records"]:
        assert "experimental_kcal_mol" not in record
        assert "route_error_kcal_mol" not in record
        assert "published_explicit_error_kcal_mol" not in record
        assert set(record) == {
            "compound_id",
            "source_record_sha256",
            "polar_mismatch_kcal_mol",
            "obc2_polar_mismatch_kcal_mol",
            "nonpolar_mismatch_kcal_mol",
            "route_minus_published_explicit_kcal_mol",
            "published_component_rounding_residual_kcal_mol",
            "component_identity_residual_kcal_mol",
            "route_absolute_error_exceeds_fixed_gate",
            "published_explicit_absolute_error_exceeds_fixed_gate",
            "continuum_reference_difference_exceeds_fixed_gate",
        }


def test_explicit_component_diagnostic_refuses_any_selection_or_correction(
    tmp_path: Path,
) -> None:
    artifact = _run(tmp_path)
    decision = artifact["decision"]
    assert decision["status"] == "diagnostic_bottleneck_decomposition_not_endpoint_evidence"
    assert decision["endpoint_selection_allowed"] is False
    assert decision["charge_method_selection_allowed"] is False
    assert decision["energy_correction_allowed"] is False
    assert decision["parameter_update_allowed"] is False
    assert decision["certified_ranking_allowed"] is False


def test_explicit_component_diagnostic_rejects_component_artifact_hash_tamper(
    tmp_path: Path,
) -> None:
    tampered = copy.deepcopy(runner.core.load_json(COMPONENT_ARTIFACT_PATH))
    tampered["records"][0]["components_kcal_mol"]["chagb_polar"] += 0.1
    tampered["content_sha256"] = runner.core.artifact_content_sha256(tampered)
    tampered_path = _write_json(tmp_path / "tampered_components.json", tampered)

    with pytest.raises(ValueError, match="Frozen component-artifact file hash mismatch"):
        _run(tmp_path, component_artifact=tampered_path)


def test_explicit_component_diagnostic_rejects_freesolv_hash_tamper(tmp_path: Path) -> None:
    tampered = copy.deepcopy(runner.core.load_json(FREESOLV_DATABASE_PATH))
    tampered["mobley_1017962"]["calc_charging"] += 0.1
    tampered_path = _write_json(tmp_path / "tampered_database.json", tampered)

    with pytest.raises(ValueError, match="Frozen FreeSolv database hash mismatch"):
        _run(tmp_path, freesolv_database=tampered_path)


def test_explicit_component_diagnostic_rejects_published_component_sum_beyond_rounding(
    tmp_path: Path,
) -> None:
    tampered = copy.deepcopy(runner.core.load_json(FREESOLV_DATABASE_PATH))
    tampered["mobley_1017962"]["calc_charging"] += 0.1
    tampered_path = _write_json(tmp_path / "inconsistent_database.json", tampered)
    protocol = _aligned_protocol(tmp_path, freesolv_database=tampered_path)

    with pytest.raises(ValueError, match="Published component sum exceeds rounding tolerance"):
        _run(
            tmp_path,
            protocol=protocol,
            freesolv_database=tampered_path,
        )


def test_explicit_component_diagnostic_protocol_rejects_parameter_update_permission(
    tmp_path: Path,
) -> None:
    protocol = copy.deepcopy(runner.core.load_json(PROTOCOL_PATH))
    protocol["pre_registered_decision_rule"]["parameter_update_allowed"] = True
    protocol_path = _write_json(tmp_path / "weak_protocol.json", protocol)

    with pytest.raises(ValueError, match="permits unsupported action"):
        runner.load_protocol(protocol_path)
