from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from maple.function.route2_smd_profiles import (
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
    route2_smd_profile_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIRECTORY = (
    REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
)
KCAL_PER_HARTREE = 627.5094740631


def _load(name: str) -> dict:
    return json.loads(
        (BENCHMARK_DIRECTORY / name).read_text(encoding="utf-8")
    )


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    int(value, 16)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_four_molecule_accuracy_preregistration_is_single_profile_and_locked():
    protocol = _load("route2-four-molecule-accuracy-prereg-v1.json")

    assert protocol["schema_version"] == 1
    assert protocol["status"] == "pre-registered"
    assert protocol["protocol_id"] == (
        "route2-four-molecule-same-geometry-accuracy-v1"
    )
    assert protocol["energy_definition"] == (
        "delta_g_solv = delta_e_solute + u_pol + g_cds"
    )
    profile = route2_smd_profile_spec(protocol["route2_profile"])
    assert profile.provider == "pyddx"
    assert profile.cavity == "gaff2-carbonyl-o"
    assert (
        profile.mace_long_range_evaluator
        == MACEPOL_MOLECULAR_REALSPACE_PROFILE
    )

    panel = protocol["fixed_panel"]
    assert [record["compound_id"] for record in panel] == [
        "mobley_1636752",
        "mobley_3867265",
        "mobley_3982371",
        "mobley_352111",
    ]
    assert len({record["mol2_sha256"] for record in panel}) == 4
    for record in panel:
        _assert_sha256(record["mol2_sha256"])
        assert record["training_membership_confirmed_excluded"] is False

    assert protocol["evidence_policy"][
        "molecule_specific_tuning_forbidden"
    ] is True
    route1 = protocol["route1_control"]
    assert route1["source_branch"] == "implicitsolv-route1"
    assert route1["method"] == "obc2_ace"
    assert route1["write_prohibited"] is True
    _assert_sha256(route1["source_sha256"])
    assert protocol["execution_budget"] == {
        "new_qm_same_geometry_single_points": 1,
        "new_route2_single_points": 4,
        "qm_target": "methyl acetate only",
        "route1_recalculation": False,
        "route1_source_is_read_only": True,
        "stop_after_budget": True,
    }
    assert protocol["primary_gates"] == {
        "all_four_records_complete": True,
        "maximum_route2_absolute_error_vs_experiment_kcal_mol": 1.5,
        "maximum_route2_absolute_error_vs_qm_kcal_mol": 1.5,
        "minimum_route2_per_molecule_wins_or_ties_vs_route1": 3,
        "route2_mae_vs_experiment_must_be_lower_than_route1": True,
        "route2_mae_vs_qm_kcal_mol": 1.0,
        "single_indivisible_route2_profile": True,
    }


def test_four_molecule_accuracy_preregistration_was_not_executed():
    disposition = _load(
        "route2-four-molecule-accuracy-prereg-v1-disposition.json"
    )
    preregistration_path = (
        REPOSITORY_ROOT / disposition["preregistration_path"]
    )

    assert disposition["schema_version"] == 1
    assert disposition["protocol_id"] == (
        "route2-four-molecule-same-geometry-accuracy-v1"
    )
    assert disposition["decision"] == "withdrawn-before-execution"
    assert disposition["preregistration_sha256"] == _sha256(
        preregistration_path
    )
    assert disposition["executed_new_qm_single_points"] == 0
    assert disposition["executed_new_route1_calculations"] == 0
    assert disposition["executed_new_route2_single_points"] == 0
    assert disposition[
        "thresholds_or_membership_changed_after_registration"
    ] is False


def test_iswig_discriminator_preregistration_has_exact_unretunable_budget():
    protocol = _load(
        "route2-pyscf-iswig-discriminator-prereg-v1.json"
    )

    assert protocol["schema_version"] == 1
    assert protocol["status"] == "pre-registered"
    assert protocol["protocol_id"] == (
        "route2-pyscf-iswig-fixed-density-rotation-v1"
    )
    assert protocol["control"]["surface_discretization_method"] == "SWIG"
    assert protocol["candidate"][
        "surface_discretization_method"
    ] == "ISWIG"
    budget = protocol["exact_evaluation_budget"]
    assert budget["continuum_scalar_solves"] == (
        budget["methods"]
        * budget["molecules"]
        * budget["orientations_per_method_and_molecule"]
    )
    assert budget["reruns_after_observing_results"] == 0
    assert protocol["rotation"]["angles_radians"] == [
        0.0,
        0.731,
        1.947,
    ]
    assert protocol["locked_gates"] == {
        "all_surface_solves_finite": True,
        "candidate_maximum_rotation_span_kcal_mol": 0.001,
        "candidate_parent_counts_constant_across_orientations": True,
        "candidate_rotation_span_not_greater_than_control": True,
        "candidate_rotation_span_ratio_to_control_maximum": 0.75,
        "maximum_half_coupling_identity_error_hartree": 1e-10,
    }
    for record in protocol["fixed_inputs"]:
        _assert_sha256(record["mol2_sha256"])
        _assert_sha256(record["density_npz_sha256"])


def test_iswig_discriminator_result_recomputes_the_locked_rejection():
    protocol = _load(
        "route2-pyscf-iswig-discriminator-prereg-v1.json"
    )
    result = _load("route2-pyscf-iswig-discriminator-v1.json")

    assert result["protocol_id"] == protocol["protocol_id"]
    assert result["status"] == "fail"
    assert result["run_exit_code"] == 1
    preregistration_path = (
        REPOSITORY_ROOT / result["preregistration_path"]
    )
    runner_path = REPOSITORY_ROOT / result["runner_path"]
    assert result["preregistration_sha256"] == _sha256(
        preregistration_path
    )
    assert result["runner_sha256"] == _sha256(runner_path)
    _assert_sha256(result["source_artifact_sha256"])
    assert result["radius_provenance"] == {
        "actual_radius_selector": "smd-iefpcm-gaff2-o",
        "equivalence_scope": (
            "At git head a55a0d9 both names select the strict "
            "GAFF/GAFF2 carbonyl-o cavity branch (o=1.70 angstrom, "
            "os=1.52 angstrom); only smd-iefpcm-gaff2-o was passed "
            "to route2_water_coulomb_radii by the executed runner."
        ),
        "preregistered_profile_context": (
            "smd-ddpcm-l15-n1202-gaff2-o-v1"
        ),
        "production_profile_changed": False,
    }

    records = {
        (
            record["compound_id"],
            record["surface_discretization_method"],
        ): record
        for record in result["records"]
    }
    actual_solves = sum(
        len(record["orientations"]) for record in records.values()
    )
    expected_solves = protocol["exact_evaluation_budget"][
        "continuum_scalar_solves"
    ]
    all_orientations = [
        orientation
        for record in records.values()
        for orientation in record["orientations"]
    ]
    thresholds = protocol["locked_gates"]
    candidate_records = [
        record
        for record in records.values()
        if record["surface_discretization_method"] == "ISWIG"
    ]

    for record in records.values():
        assert [
            orientation["angle_radians"]
            for orientation in record["orientations"]
        ] == protocol["rotation"]["angles_radians"]
        for orientation in record["orientations"]:
            assert math.isfinite(orientation["energy_hartree"])
            assert math.isfinite(
                orientation["half_coupling_identity_error_hartree"]
            )
            assert sum(orientation["parent_counts"]) == orientation[
                "surface_size"
            ]
            assert all(count > 0 for count in orientation["parent_counts"])
        energies = [
            orientation["energy_hartree"]
            for orientation in record["orientations"]
        ]
        span_hartree = max(energies) - min(energies)
        assert math.isclose(
            record["rotation_span_hartree"],
            span_hartree,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
        assert math.isclose(
            record["rotation_span_kcal_mol"],
            span_hartree * KCAL_PER_HARTREE,
            rel_tol=1e-15,
            abs_tol=1e-18,
        )

    rotation_ratios = {}
    not_worse = True
    ratio_gate = True
    for compound_id in ("mobley_1636752", "mobley_3867265"):
        control = records[(compound_id, "SWIG")]
        candidate = records[(compound_id, "ISWIG")]
        ratio = (
            candidate["rotation_span_kcal_mol"]
            / control["rotation_span_kcal_mol"]
        )
        assert math.isclose(
            ratio,
            result["rotation_span_ratios_iswig_over_swig"][
                compound_id
            ],
            rel_tol=1e-12,
        )
        rotation_ratios[compound_id] = ratio
        not_worse &= (
            candidate["rotation_span_kcal_mol"]
            <= control["rotation_span_kcal_mol"]
        )
        ratio_gate &= ratio <= thresholds[
            "candidate_rotation_span_ratio_to_control_maximum"
        ]
    assert result["rotation_span_ratios_iswig_over_swig"] == (
        rotation_ratios
    )

    gates = {
        "exact_evaluation_budget": actual_solves == expected_solves,
        "all_surface_solves_finite": all(
            math.isfinite(orientation["energy_hartree"])
            for orientation in all_orientations
        ),
        "maximum_half_coupling_identity_error": max(
            orientation["half_coupling_identity_error_hartree"]
            for orientation in all_orientations
        )
        <= thresholds["maximum_half_coupling_identity_error_hartree"],
        "candidate_maximum_rotation_span": max(
            record["rotation_span_kcal_mol"]
            for record in candidate_records
        )
        <= thresholds["candidate_maximum_rotation_span_kcal_mol"],
        "candidate_parent_counts_constant_across_orientations": all(
            len(
                {
                    tuple(orientation["parent_counts"])
                    for orientation in record["orientations"]
                }
            )
            == 1
            for record in candidate_records
        ),
        "candidate_rotation_span_not_greater_than_control": not_worse,
        "candidate_rotation_span_ratio_to_control": ratio_gate,
    }
    assert result["gates"] == gates
    gate_order = [
        "exact_evaluation_budget",
        "all_surface_solves_finite",
        "maximum_half_coupling_identity_error",
        "candidate_maximum_rotation_span",
        "candidate_parent_counts_constant_across_orientations",
        "candidate_rotation_span_not_greater_than_control",
        "candidate_rotation_span_ratio_to_control",
    ]
    assert result["failed_gates"] == [
        name for name in gate_order if not gates[name]
    ]
