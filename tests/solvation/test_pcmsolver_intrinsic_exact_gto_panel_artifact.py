from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_intrinsic_exact_gto_panel_is_real_data_bound_and_reproducible():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == (
        "route2-pcmsolver-intrinsic-exact-gto-freesolv-ten-v1"
    )
    assert artifact["schema_version"] == 1
    assert artifact["selection"]["record_count"] == 10
    assert artifact["selection"]["partition"] == "development-only"
    assert artifact["selection"]["used_candidate_errors"] is False
    assert len(set(artifact["selection"]["functional_group_classes"])) == 10

    reference = artifact["experimental_reference"]
    assert reference == {
        "all_records_neutral": True,
        "dataset": "FreeSolv",
        "dataset_citations": [
            "10.1007/s10822-014-9747-x",
            "10.1021/acs.jced.7b00104",
        ],
        "database_json_sha256": (
            "9133b2438af6081d4cf6ac040cb201c2e3ec953988a45248586f20f2f159c78a"
        ),
        "database_txt_sha256": (
            "2d13f095713bc39b85f85dd7b4e5483fbb12fc694bf253bb1d92a4c4d484f260"
        ),
        "geometry_archive_sha256": (
            "15ece6114442a8eb5e720c0ad25bb48e4a6a35392da36f72b80a6f2f415fd133"
        ),
        "row_values_verified_against_pinned_database_txt": True,
        "solvent": "water",
        "upstream_commit": "6c7d19b4b565537365ffd22006aa2cd4643200c6",
        "version": "0.52",
    }

    records = artifact["records"]
    assert len(records) == 10
    assert all(math.isfinite(row["experimental_kcal_mol"]) for row in records)
    assert all(
        row["experimental_uncertainty_kcal_mol"] > 0.0 for row in records
    )
    assert all(row["experimental_reference"]["source"] for row in records)
    assert all(
        len(row["experimental_reference"]["database_record_sha256"]) == 64
        for row in records
    )

    execution = artifact["execution_source"]
    source_paths = {
        "profile_registry_sha256": ROOT
        / "maple/function/route2_smd_profiles.py",
        "intrinsic_cavity_adapter_sha256": ROOT
        / (
            "maple/function/calculator/extra_correction/implicit/"
            "route2_pcmsolver_cavity.py"
        ),
        "smd_provider_sha256": ROOT
        / "maple/function/calculator/extra_correction/implicit/smd.py",
    }
    for key, path in source_paths.items():
        assert execution[key] == _sha256(path)


def test_intrinsic_exact_gto_panel_metrics_and_non_default_decision_are_locked():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = artifact["records"]

    expected = {
        "mace_fixed_l1": {
            "mae": 1.0839509358424428,
            "rmse": 1.2938834602242526,
            "max": 2.6472294792602096,
        },
        "mace_scf_local_jet": {
            "mae": 1.7516453017511917,
            "rmse": 2.5159311232209682,
            "max": 5.743892024212777,
        },
        "mace_scf_exact_gto": {
            "mae": 0.9152863783301732,
            "rmse": 1.3054578142348061,
            "max": 3.2473279956943566,
        },
    }
    for method, expected_metrics in expected.items():
        errors = [
            row["methods"][method]["signed_error_kcal_mol"] for row in records
        ]
        absolute_errors = [abs(value) for value in errors]
        recomputed = {
            "mae": sum(absolute_errors) / len(absolute_errors),
            "rmse": math.sqrt(
                sum(value * value for value in errors) / len(errors)
            ),
            "max": max(absolute_errors),
        }
        metrics = artifact["aggregate_metrics"][method]
        assert metrics["record_count"] == 10
        for name, expected_value in expected_metrics.items():
            assert recomputed[name] == pytest.approx(expected_value, abs=1.0e-12)
        assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
            recomputed["mae"], abs=1.0e-12
        )
        assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
            recomputed["rmse"], abs=1.0e-12
        )
        assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
            recomputed["max"], abs=1.0e-12
        )

    paired = artifact["paired_exact_gto_vs_local_jet"]
    assert paired["exact_gto_win_count"] == 7
    assert paired["local_jet_win_count"] == 3

    by_name = {row["name"]: row for row in records}
    assert by_name["acetone"]["methods"]["mace_scf_exact_gto"][
        "absolute_error_kcal_mol"
    ] == pytest.approx(0.9332623313117052, abs=1.0e-12)
    assert by_name["acetic acid"]["methods"]["mace_scf_exact_gto"][
        "absolute_error_kcal_mol"
    ] == pytest.approx(3.2473279956943566, abs=1.0e-12)

    decision = artifact["preregistered_decision"]
    assert decision["gates"] == {
        "all_native_warning_counts_zero": True,
        "candidate_acetone_below_1": True,
        "candidate_mae_not_above_control": True,
        "candidate_max_error_below_2": False,
        "candidate_panel_mae_below_1": True,
    }
    assert decision["overall_pass"] is False
    assert "non-default experimental profile" in decision["disposition"]
    assert "do not replace the global default" in decision["disposition"]
