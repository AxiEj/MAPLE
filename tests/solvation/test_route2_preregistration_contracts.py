from __future__ import annotations

import json
from pathlib import Path

from maple.function.route2_smd_profiles import (
    MACEPOL_MOLECULAR_REALSPACE_PROFILE,
    route2_smd_profile_spec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIRECTORY = (
    REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
)


def _load(name: str) -> dict:
    return json.loads(
        (BENCHMARK_DIRECTORY / name).read_text(encoding="utf-8")
    )


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    int(value, 16)


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
    assert protocol["primary_gates"][
        "route2_mae_vs_qm_kcal_mol"
    ] == 1.0
    assert protocol["primary_gates"][
        "minimum_route2_per_molecule_wins_or_ties_vs_route1"
    ] == 3


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
    assert protocol["locked_gates"][
        "candidate_maximum_rotation_span_kcal_mol"
    ] == 0.001
    assert protocol["locked_gates"][
        "candidate_rotation_span_ratio_to_control_maximum"
    ] == 0.75
    for record in protocol["fixed_inputs"]:
        _assert_sha256(record["mol2_sha256"])
        _assert_sha256(record["density_npz_sha256"])
