from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
ARTIFACT_V1_PATH = BENCHMARKS / "mace-ef-cosmo-freesolv20-two-step-v1.json"
ARTIFACT_PATH = BENCHMARKS / "mace-ef-cosmo-freesolv20-two-step-v2.json"
PREREGISTRATION_PATH = BENCHMARKS / "mace-ef-cosmo-freesolv20-two-step-prereg-v1.json"
PRIMARY_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-diverse-v3.json"
BUNDLE_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-profile-bundle-v1.tar.gz"
BUNDLE_MANIFEST_PATH = BENCHMARKS / "mace-ef-cosmors-freesolv20-profile-bundle-v1.json"
EV_TO_KCAL_MOL = 627.5094740631 / 27.211386245988


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_two_step_artifact_is_complete_source_bound_and_partial_only():
    v1 = _load(ARTIFACT_V1_PATH)
    assert _sha256(ARTIFACT_V1_PATH) == (
        "7796395485a194f67997563b26123e41ae677be88c92f2e7124e233571540bff"
    )
    assert_source_files_match_execution_commit(ROOT, v1)
    artifact = _load(ARTIFACT_PATH)

    assert _sha256(ARTIFACT_PATH) == (
        "d7747841158fed7692129826fefa2b59fd18d3d0cbf9c49a9763705f2453e1eb"
    )
    assert artifact["artifact"] == "mace-ef-cosmo-freesolv20-two-step-v2"
    assert artifact["status"] == "complete"
    assert artifact["scientific_status"] == ("complete-partial-component-diagnostic")
    assert artifact["diagnostic_only"] is True
    assert artifact["admission_eligible"] is False
    assert artifact["failures"] == []
    assert len(artifact["records"]) == 20
    assert artifact["preregistration"]["sha256"] == _sha256(PREREGISTRATION_PATH)
    assert artifact["primary_artifact"]["sha256"] == _sha256(PRIMARY_PATH)
    assert artifact["geometry_bundle"]["sha256"] == _sha256(BUNDLE_PATH)
    assert artifact["geometry_bundle"]["manifest_sha256"] == _sha256(
        BUNDLE_MANIFEST_PATH
    )
    assert_source_files_match_execution_commit(ROOT, artifact)

    assert artifact["method"]["reported_energy"]["primary_component"] == (
        "U_segment_COSMO(c2)"
    )
    assert artifact["method"]["reported_energy"]["included_terms"] == [
        "Torch segment-COSMO conductor polarization"
    ]
    assert artifact["interpretation"]["automatic_accuracy_verdict"] is None
    assert artifact["interpretation"]["stationary_solution_claimed"] is False
    assert (
        artifact["interpretation"]["two_step_self_consistent_solution_claimed"] is False
    )
    assert artifact["interpretation"]["mathematical_contraction_claimed"] is False
    assert (
        artifact["interpretation"][
            "iteration_count_may_not_be_changed_using_this_result"
        ]
        is True
    )
    assert {
        "chemical_potential",
        "delta_g_solvation_kcal_mol",
        "log_activity",
        "model_result",
    }.isdisjoint(_all_keys(artifact))


def test_two_step_artifact_reconstructs_exactly_two_maps_and_cosmo_identity():
    artifact = _load(ARTIFACT_PATH)
    primary = _load(PRIMARY_PATH)
    primary_by_id = {record["compound_id"]: record for record in primary["records"]}

    for record in artifact["records"]:
        assert record["map_applications"] == 2
        assert record["electronic_evaluation_count"] == 3
        assert record["third_map_evaluated"] is False
        assert len(record["stages"]) == 3
        assert [stage["source_index"] for stage in record["stages"]] == [0, 1, 2]
        assert all(
            stage["not_total_hydration_free_energy"] is True
            for stage in record["stages"]
        )
        assert all(
            abs(float(stage["source_charge_e"])) <= 2.0e-5 for stage in record["stages"]
        )
        assert all(
            float(stage["half_coupling_identity_error_ev"]) <= 4.0e-16
            for stage in record["stages"]
        )
        for stage in record["stages"]:
            energy_ev = float(stage["cosmo_boundary_energy_ev"])
            pairing_ev = float(stage["source_field_pairing_ev"])
            assert energy_ev == pytest.approx(0.5 * pairing_ev, abs=4.0e-16)
            assert float(stage["cosmo_boundary_energy_kcal_mol"]) == pytest.approx(
                energy_ev * EV_TO_KCAL_MOL,
                abs=1.0e-12,
            )
        energies = [
            float(stage["cosmo_boundary_energy_kcal_mol"]) for stage in record["stages"]
        ]
        assert energies[2] < energies[1] < energies[0]
        continuum = record["continuum"]
        expected_config = primary_by_id[record["compound_id"]]["mace_ef_surface"][
            "continuum_configuration_sha256"
        ]
        assert continuum["configuration_sha256"] == expected_config
        assert continuum["matches_converged_comparator"] is True
        assert continuum["cavity_radii_identity"] == (
            "audited-openCOSMO-RS-24a-ORCA6-conductor-radii"
        )
        assert continuum["finite_dielectric_scaling"] is False
        assert len(continuum["radii_angstrom"]) == int(
            primary_by_id[record["compound_id"]]["atom_count"]
        )
        for stage in record["stages"]:
            expected = float(stage["cosmo_boundary_energy_kcal_mol"]) - float(
                record["experimental_total_hydration_label_kcal_mol"]
            )
            assert stage[
                "partial_minus_total_freesolv_label_kcal_mol"
            ] == pytest.approx(expected)


def test_two_step_summary_freezes_response_amplification_without_accuracy_claim():
    artifact = _load(ARTIFACT_PATH)
    summaries = artifact["summaries"]

    expected_partial_mean_absolute = {
        "c0": 1.5531019396694563,
        "c1": 3.5718428525545773,
        "c2": 5.149706025135921,
    }
    for index, (stage, expected) in enumerate(expected_partial_mean_absolute.items()):
        observed = summaries["stages"][stage][
            "partial_component_difference_from_total_freesolv_label"
        ]
        primitive = np.asarray(
            [
                float(
                    record["stages"][index][
                        "partial_minus_total_freesolv_label_kcal_mol"
                    ]
                )
                for record in artifact["records"]
            ]
        )
        assert observed["count"] == 20
        assert observed["mean_absolute"] == pytest.approx(expected)
        assert observed["mean"] == pytest.approx(np.mean(primitive))
        assert observed["mean_absolute"] == pytest.approx(np.mean(np.abs(primitive)))
        assert observed["median_absolute"] == pytest.approx(
            np.median(np.abs(primitive))
        )
        assert observed["maximum_absolute"] == pytest.approx(np.max(np.abs(primitive)))
        assert observed["rmse"] == pytest.approx(np.sqrt(np.mean(primitive**2)))
        assert observed["units"] == "kcal/mol"
    c2_converged = summaries["stages"]["c2"]["difference_from_converged_boundary"]
    c2_converged_primitive = np.asarray(
        [
            float(record["stages"][2]["difference_from_converged_boundary_kcal_mol"])
            for record in artifact["records"]
        ]
    )
    assert c2_converged["mean_absolute"] == pytest.approx(1.3564897508181877)
    assert c2_converged["maximum_absolute"] == pytest.approx(7.569310309114897)
    assert c2_converged["mean_absolute"] == pytest.approx(
        np.mean(np.abs(c2_converged_primitive))
    )

    updates = summaries["source_update_raw_component_max"]
    ratios = np.asarray(
        [
            float(record["source_updates"]["second_to_first_ratio"])
            for record in artifact["records"]
        ]
    )
    first_updates = np.asarray(
        [
            float(record["source_updates"]["first_maximum_absolute_component"])
            for record in artifact["records"]
        ]
    )
    second_updates = np.asarray(
        [
            float(record["source_updates"]["second_maximum_absolute_component"])
            for record in artifact["records"]
        ]
    )
    assert updates["second_update_smaller_record_count"] == 20
    assert updates["second_update_not_smaller_record_count"] == 0
    assert updates["coordinate_invariant_norm"] is False
    assert updates["mathematical_contraction_claimed"] is False
    assert updates["first_maximum_absolute_component"]["mean"] == pytest.approx(
        np.mean(first_updates)
    )
    assert updates["second_maximum_absolute_component"]["mean"] == pytest.approx(
        np.mean(second_updates)
    )
    assert updates["second_to_first_ratio"]["mean"] == pytest.approx(np.mean(ratios))
    assert updates["second_to_first_ratio"]["maximum_absolute"] == pytest.approx(
        np.max(np.abs(ratios))
    )
    assert summaries["electronic_passivity"] == {
        "audit_source": (
            "inherited primary same-geometry/checkpoint uniform-field audit"
        ),
        "c2_state_passivity_recomputed": False,
        "failed_count": 15,
        "passed_count": 5,
        "two_step_route_admission_eligible": False,
    }
