from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-aimnet2-multisolvent-pilot-v1.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_all_keys(item) for item in value.values()),
        )
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_mnsol_aimnet2_pilot_is_aggregate_only_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == ("route2-mnsol-aimnet2-multisolvent-pilot-v1")
    assert artifact["execution_git_head"] == (
        "7bc6164390b7aab61e2877ae9c845a62b41820e8"
    )
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["dataset"]["row_level_data_emitted"] is False
    assert artifact["dataset"]["standard_state"] == (
        "1M-ideal-gas-to-1M-ideal-solution"
    )
    assert artifact["selection_record_count"] == 10
    assert artifact["selection_solvent_count"] == 10
    assert artifact["selection_partition_counts"] == {
        "confirmation": 8,
        "development": 2,
    }
    assert "records" not in artifact
    keys = _all_keys(artifact)
    for forbidden in (
        "entry_number",
        "geometry_handle",
        "solute_name",
        "experimental_delta_g_kcal_mol",
        "charges_e",
    ):
        assert forbidden not in keys

    identity = artifact["scientific_identity"]
    assert identity["solute_source"] == "AIMNet2 NQE point-charge-l0"
    assert identity["polarization_response"] == "fixed"
    assert identity["continuum_equations"] == [
        "pyddx ddPCM",
        "pyddx ddCOSMO",
    ]
    assert identity["nonpolar_model"] == "PySCF 2.13.1 SMD-CDS"
    assert identity["strict_original_smd_equivalence"] is False
    assert identity["mutual_ml_continuum_polarization"] is False
    assert identity["cpcm_included"] is False
    assert identity["cosmo_rs_included"] is False
    assert (
        "One record per solvent cannot certify accuracy" in artifact["claim_boundary"]
    )

    assert artifact["checkpoint"]["sha256"] == (
        "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
    )
    assert artifact["selection_fingerprint"] == (
        "967979795b3bd1483db93f7b463f9a65f3f2c0859f69228100d2adc107138fba"
    )
    charge = artifact["charge_quality"]
    assert charge["maximum_absolute_raw_charge_residual_e"] <= 1.0e-6
    assert charge["maximum_absolute_projected_charge_sum_e"] <= 2.0e-16

    expected = {
        "ddpcm": {
            "mae": 1.143062508322529,
            "rmse": 1.3515629726549894,
            "mse": 1.0333127266316697,
            "max": 2.2322398232614793,
        },
        "ddcosmo": {
            "mae": 1.0120975743588836,
            "rmse": 1.2743163910385147,
            "mse": 0.8300034969173833,
            "max": 2.193276816293233,
        },
    }
    for method, values in expected.items():
        metrics = artifact["aggregate_metrics"][method]
        assert metrics["record_count"] == 10
        assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
            values["mae"],
            abs=1.0e-12,
        )
        assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
            values["rmse"],
            abs=1.0e-12,
        )
        assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(
            values["mse"],
            abs=1.0e-12,
        )
        assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
            values["max"],
            abs=1.0e-12,
        )
        assert metrics["maximum_half_coupling_identity_error_ev"] <= 4.0e-14

    paired = artifact["paired_method_comparison"]
    assert paired["ddcosmo_lower_absolute_error_count"] == 8
    assert paired["ddpcm_lower_absolute_error_count"] == 2
    assert paired["absolute_error_tie_count"] == 0
    assert paired["mean_ddcosmo_minus_ddpcm_kcal_mol"] == pytest.approx(
        -0.20330922971428644,
        abs=1.0e-12,
    )

    timing = artifact["timing_seconds"]
    assert timing["total_wall"] > 0.0
    assert timing["method_execution_order"] == ["ddpcm", "ddcosmo"]
    assert "not-a-speed-ranking" in timing["comparison_status"]
    assert artifact["runtime"]["pyddx"] == "0.8.0"
    assert artifact["runtime"]["pyscf"] == "2.13.1"

    assert_source_files_match_execution_commit(ROOT, artifact)
