from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = (
    BENCHMARK_DIR / "route1_3drism_thermodynamic_consistency_protocol_v1.json"
)
ARTIFACT_PATH = (
    BENCHMARK_DIR
    / "route1-3drism-ethanol-thermodynamic-consistency-2026-07-29.json"
)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "route1_3drism_thermodynamic_consistency",
    BENCHMARK_DIR / "run_route1_3drism_thermodynamic_consistency.py",
)
assert RUNNER_SPEC is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def test_protocol_is_research_only_and_accuracy_ineligible() -> None:
    protocol, _ = runner._load_protocol(PROTOCOL_PATH)

    boundary = protocol["route_boundary"]
    reference = protocol["thermodynamic_reference"]
    assert boundary["product_route1_formula_unchanged"] is True
    assert boundary["candidate_route_identity"] == (
        "separate_3drism_research_comparator"
    )
    assert boundary["candidate_may_satisfy_product_route1_accuracy_gate"] is False
    assert boundary["experimental_solvation_labels_loaded"] is False
    assert reference["target_temperature_kelvin"] == 298.0
    assert reference["asset_temperature_kelvin"] == 298.15
    assert reference["accuracy_temperature_eligible"] is False
    assert reference["accuracy_ensemble_eligible"] is False
    assert (
        reference[
            "neutral_excess_chemical_potential_standard_state_shift_kcal_mol"
        ]
        == 0.0
    )


def test_protocol_binds_the_prior_label_free_numerical_pilot() -> None:
    protocol, _ = runner._load_protocol(PROTOCOL_PATH)
    pilot_protocol, artifact, _, _ = runner._load_bound_pilot(protocol)

    assert pilot_protocol["solver"]["pc_plus_is_primary"] is False
    assert pilot_protocol["solver"]["standard_state_conversion_applied"] is False
    assert artifact["admission"]["accuracy_claim"] == "none"


def test_rigid_coordinate_transforms_preserve_internal_distances() -> None:
    coordinates = [
        (0.0, 0.0, 0.0),
        (1.2, -0.5, 3.0),
        (-2.0, 0.75, 0.4),
    ]
    translated = runner._transform_coordinates(
        coordinates, translation=(37.0, -19.0, 11.0)
    )
    rotated = runner._transform_coordinates(
        coordinates, rotation=runner._rotation_matrix([37.0, 53.0, 71.0])
    )

    assert runner._maximum_pair_distance_change(coordinates, translated) < 1e-12
    assert runner._maximum_pair_distance_change(coordinates, rotated) < 1e-12


def _section(name: str, values: list[float]) -> str:
    return (
        f"%FLAG {name}\n"
        "%FORMAT(5E16.8)\n"
        + "".join(f"{value:16.8E}" for value in values)
        + "\n"
    )


def test_prmtop_section_evidence_distinguishes_operations(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.prmtop"
    modified = tmp_path / "modified.prmtop"
    original.write_text(
        _section("BOND_FORCE_CONSTANT", [1.0, 2.0])
        + _section("CHARGE", [0.5, -0.5])
        + _section("LENNARD_JONES_ACOEF", [3.0]),
        encoding="ascii",
    )
    modified.write_text(
        _section("BOND_FORCE_CONSTANT", [1.5, 3.0])
        + _section("CHARGE", [0.5, -0.5])
        + _section("LENNARD_JONES_ACOEF", [0.0]),
        encoding="ascii",
    )

    evidence = runner._section_evidence(
        original,
        modified,
        scaled_flags=["BOND_FORCE_CONSTANT"],
        scale_factor=1.5,
        unchanged_flags=["CHARGE"],
        zeroed_flags=["LENNARD_JONES_ACOEF"],
    )
    assert evidence["passed"] is True
    assert all(entry["passed"] for entry in evidence["flags"].values())


def test_negative_artifact_is_sealed_and_fails_closed() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["content_sha256"] == runner.core.artifact_content_sha256(
        artifact
    )
    assert artifact["conclusion"] == {
        "accuracy_admission": False,
        "next_gate": "return_to_fixed_charge_chagb_pbsa_ranking_mainline",
        "numerical_consistency_passed": False,
        "product_route1_admission": False,
        "status": "research_comparator_consistency_failed_accuracy_blocked",
    }
    assert artifact["containment"]["experimental_solvation_labels_loaded"] is False
    assert artifact["containment"]["fit_or_tuning_performed"] is False
    assert artifact["gates"]["translation_invariance"]["passed"] is True
    assert (
        artifact["gates"]["bonded_parameter_independence"]["passed"] is True
    )
    assert artifact["gates"]["proper_rotation_invariance"]["passed"] is False
    assert artifact["gates"]["zero_interaction_limit"]["passed"] is False


def test_negative_artifact_exposes_both_control_failures() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    reference = artifact["cases"]["reference"]["solver"]
    rotated = artifact["cases"]["rotated"]["solver"]
    ghost = artifact["cases"]["zero_interaction"]["solver"]

    assert reference["grid_points_xyz"] != rotated["grid_points_xyz"]
    assert (
        artifact["gates"]["proper_rotation_invariance"][
            "maximum_energy_difference_kcal_mol"
        ]
        > 0.01
    )
    assert abs(ghost["raw_excess_chemical_potential_kcal_mol"]) > 0.005
    assert abs(
        ghost["gaussian_fluctuation_excess_chemical_potential_kcal_mol"]
    ) > 0.005
    assert math.isfinite(ghost["pc_plus_excess_chemical_potential_kcal_mol"])
    assert math.isfinite(ghost["partial_molar_volume_angstrom3"])
