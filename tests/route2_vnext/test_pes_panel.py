from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.release.pes_panel import (
    PES_PANEL_ADDITIONAL_PATHS,
    PES_PANEL_ASSET_SHA256,
    PES_PANEL_BOND_DISPLACEMENT_A,
    PES_PANEL_DIRECTION_NAMES,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    PES_PANEL_MOLECULE_COUNT,
    PES_PANEL_STRETCH_CHANGES_A,
    PES_PANEL_TORSION_ANGLES_DEG,
    PES_PANEL_VARIANT_NAMES,
    PES_CARTESIAN_PANEL_CONTRACT_VERSION,
    PES_CARTESIAN_PANEL_STEPS_A,
    PES_CARTESIAN_PANEL_VARIANT,
    bond_direction,
    load_pes_panel,
    panel_directions,
    panel_geometries,
    panel_paths,
    select_bond,
    stretch_tangent,
    summarize_cartesian_pes_panel,
    summarize_pes_panel,
    summarize_pes_paths,
    torsion_tangent,
)

ROOT = Path(__file__).parents[2]
ASSET = ROOT / "tools/route2_release/data/fixedbox590_pes_panel_v1.json"


def test_panel_asset_is_frozen_diverse_and_within_declared_checkpoint_domain():
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == PES_PANEL_ASSET_SHA256
    panel = load_pes_panel()
    assert len(panel) == PES_PANEL_MOLECULE_COUNT == 20
    assert len({record.molecule_id for record in panel}) == 20
    required = {"water", "acetone", "benzene", "acetonitrile", "methanol"}
    assert required <= {record.molecule_id for record in panel}
    assert PES_PANEL_ADDITIONAL_PATHS == (
        "trans-butane-central-bond-torsion",
        "hydrogen-peroxide-oxygen-oxygen-stretch",
    )
    tags = {tag for record in panel for tag in record.scope_tags}
    assert {"flexible torsion", "sulfur heteroaromatic", "halogenated control"} <= tags
    for record in panel:
        assert record.atoms.info == {"charge": 0, "mult": 1}
        assert np.all(np.isin(record.atoms.numbers, np.arange(1, 84)))


def test_panel_variants_change_exactly_one_deterministic_bond_coordinate():
    for molecule in load_pes_panel():
        geometries = panel_geometries(molecule)
        assert tuple(geometries) == PES_PANEL_VARIANT_NAMES
        first, second = select_bond(molecule.atoms)
        reference = np.linalg.norm(
            molecule.atoms.positions[second] - molecule.atoms.positions[first]
        )
        compressed = np.linalg.norm(
            geometries["bond-compressed"].positions[second]
            - geometries["bond-compressed"].positions[first]
        )
        stretched = np.linalg.norm(
            geometries["bond-stretched"].positions[second]
            - geometries["bond-stretched"].positions[first]
        )
        expected_change = np.sqrt(2.0) * PES_PANEL_BOND_DISPLACEMENT_A
        assert compressed == pytest.approx(reference - expected_change, abs=2e-15)
        assert stretched == pytest.approx(reference + expected_change, abs=2e-15)


def test_panel_directions_are_deterministic_normalized_and_translation_free():
    for molecule in load_pes_panel():
        first = panel_directions(molecule.atoms, molecule.molecule_id)
        second = panel_directions(molecule.atoms, molecule.molecule_id)
        assert tuple(first) == PES_PANEL_DIRECTION_NAMES
        for name in PES_PANEL_DIRECTION_NAMES:
            np.testing.assert_array_equal(first[name], second[name])
            assert np.linalg.norm(first[name]) == pytest.approx(1.0, abs=2e-15)
            np.testing.assert_allclose(first[name].sum(axis=0), 0.0, atol=2e-15)
        np.testing.assert_array_equal(
            first["bond-stretch"], bond_direction(molecule.atoms)
        )


def test_preregistered_paths_are_real_geometries_not_metadata_only():
    paths = panel_paths()
    assert tuple(paths) == PES_PANEL_ADDITIONAL_PATHS
    torsion, stretch = paths.values()
    assert tuple(point.coordinate_value for point in torsion) == (
        PES_PANEL_TORSION_ANGLES_DEG
    )
    assert len(
        {
            hashlib.sha256(point.atoms.positions.tobytes()).hexdigest()
            for point in torsion
        }
    ) == len(torsion)
    assert "close contact" in torsion[-1].scope_tags
    np.testing.assert_allclose(
        [point.atoms.get_dihedral(0, 1, 2, 3) for point in torsion],
        PES_PANEL_TORSION_ANGLES_DEG,
        atol=2e-12,
    )

    reference_length = stretch[2].coordinate_value
    np.testing.assert_allclose(
        [point.coordinate_value - reference_length for point in stretch],
        PES_PANEL_STRETCH_CHANGES_A,
        atol=2e-12,
    )
    assert "TS-like geometry" in stretch[-1].scope_tags
    assert all(
        point.path_name in PES_PANEL_ADDITIONAL_PATHS for point in (*torsion, *stretch)
    )


def test_path_tangents_match_central_coordinate_differences():
    paths = panel_paths()
    torsion = paths["trans-butane-central-bond-torsion"][0].atoms
    step = 1.0e-6
    from maple.solvation.release import pes_panel as contract

    plus = contract._rotate_about_axis(
        torsion.positions,
        axis=contract.PES_PANEL_TORSION_AXIS,
        rotated_atoms=contract.PES_PANEL_TORSION_ROTATED_ATOMS,
        angle_degrees=np.degrees(step),
    )
    minus = contract._rotate_about_axis(
        torsion.positions,
        axis=contract.PES_PANEL_TORSION_AXIS,
        rotated_atoms=contract.PES_PANEL_TORSION_ROTATED_ATOMS,
        angle_degrees=-np.degrees(step),
    )
    np.testing.assert_allclose(
        torsion_tangent(torsion), (plus - minus) / (2.0 * step), atol=3e-10
    )

    stretch = paths["hydrogen-peroxide-oxygen-oxygen-stretch"][2].atoms
    plus = stretch.copy()
    minus = stretch.copy()
    plus.positions += step * stretch_tangent(stretch)
    minus.positions -= step * stretch_tangent(stretch)
    derivative = (
        np.linalg.norm(plus.positions[1] - plus.positions[0])
        - np.linalg.norm(minus.positions[1] - minus.positions[0])
    ) / (2.0 * step)
    assert derivative == pytest.approx(1.0, abs=2e-10)
    assert np.linalg.norm(torsion_tangent(torsion)) != pytest.approx(1.0)
    assert np.linalg.norm(stretch_tangent(stretch)) == pytest.approx(
        1.0 / np.sqrt(2.0), abs=2e-15
    )


def test_path_coverage_is_explicitly_torsional_close_contact_stretched_and_ts_like():
    paths = panel_paths()
    all_points = tuple(point for path in paths.values() for point in path)
    all_tags = {tag for point in all_points for tag in point.scope_tags}
    assert {
        "flexible torsion",
        "close contact",
        "stretched bond",
        "reaction-coordinate surrogate",
        "TS-like geometry",
    } <= all_tags
    assert len(paths["trans-butane-central-bond-torsion"]) == 5
    assert len(paths["hydrogen-peroxide-oxygen-oxygen-stretch"]) == 6


def _record(molecule_id, variant, *, passed=True, topology=None):
    direction = {
        name: {
            "all_gates_passed": passed,
            "fixed_topology": passed,
            "maximum_primal_residual": 1e-13,
        }
        for name in PES_PANEL_DIRECTION_NAMES
    }
    return {
        "molecule_id": molecule_id,
        "variant": variant,
        "directional_force_fd": direction,
        "cold_warm": {
            "numerically_equivalent": passed,
            "gates": {"source": passed, "energy": passed},
        },
        "topology_hash": topology or f"{molecule_id:0<64}"[:64],
        "maximum_primal_residual": 1e-13,
        "adjoint_residual": 1e-14,
    }


def test_panel_summary_requires_all_20_by_3_records_and_never_hides_failure():
    records = [
        _record(molecule.molecule_id, variant)
        for molecule in load_pes_panel()
        for variant in PES_PANEL_VARIANT_NAMES
    ]
    summary = summarize_pes_panel(records)
    assert summary["geometry_count"] == 60
    assert summary["directional_record_count"] == 180
    assert summary["directional_sample_count"] == 180 * len(
        PES_PANEL_DIRECTIONAL_STEPS_A
    )
    assert summary["all_gates_passed"] is True

    failed = deepcopy(records)
    failed[7]["directional_force_fd"]["bond-stretch"]["all_gates_passed"] = False
    summary = summarize_pes_panel(failed)
    assert summary["gates"]["all_directional_force_fd"] is False
    assert summary["all_gates_passed"] is False
    with pytest.raises(ValueError, match="exactly 60"):
        summarize_pes_panel(records[:-1])


def _cartesian_record(molecule, *, passed=True):
    return {
        "molecule_id": molecule.molecule_id,
        "variant": PES_CARTESIAN_PANEL_VARIANT,
        "component_count": 3 * len(molecule.atoms),
        "cartesian_force_fd": {
            "all_gates_passed": passed,
            "fixed_topology": passed,
            "topology_hashes": [f"{molecule.molecule_id:0<64}"[:64]],
            "maximum_displaced_primal_residual": 1.0e-13,
            "records": [
                {
                    "step_A": step,
                    "rms_error_eV_per_A": 1.0e-6,
                    "maximum_error_eV_per_A": 2.0e-6,
                }
                for step in PES_CARTESIAN_PANEL_STEPS_A
            ],
            "convergence": {"low_error_plateau": True},
        },
        "cold_warm": {
            "numerically_equivalent": passed,
            "gates": {"source": passed, "energy": passed},
        },
        "maximum_primal_residual": 1.0e-13,
        "adjoint_residual": 1.0e-14,
    }


def test_cartesian_panel_contract_is_reference_only_complete_and_fail_closed():
    assert PES_CARTESIAN_PANEL_CONTRACT_VERSION.endswith("-v1")
    assert PES_CARTESIAN_PANEL_VARIANT == "reference"
    assert PES_CARTESIAN_PANEL_STEPS_A == (4.0e-4, 2.0e-4, 1.0e-4)
    panel = load_pes_panel()
    records = [_cartesian_record(molecule) for molecule in panel]
    summary = summarize_cartesian_pes_panel(records)
    assert summary["molecule_count"] == 20
    assert summary["geometry_count"] == 20
    assert summary["component_count"] == sum(3 * len(item.atoms) for item in panel)
    assert summary["component_sample_count"] == summary["component_count"] * 3
    assert summary["step_record_count"] == 60
    assert summary["maximum_rms_error_eV_per_A"] == pytest.approx(1.0e-6)
    assert summary["maximum_component_error_eV_per_A"] == pytest.approx(2.0e-6)
    assert summary["low_error_plateau_count"] == 20
    assert summary["all_gates_passed"] is True

    failed = deepcopy(records)
    failed[3]["cartesian_force_fd"]["all_gates_passed"] = False
    assert summarize_cartesian_pes_panel(failed)["all_gates_passed"] is False
    with pytest.raises(ValueError, match="exactly 20"):
        summarize_cartesian_pes_panel(records[:-1])


def _path_record(point, *, passed=True):
    return {
        "path_name": point.path_name,
        "molecule_id": point.molecule_id,
        "point_label": point.point_label,
        "coordinate_name": point.coordinate_name,
        "coordinate_value": point.coordinate_value,
        "coordinate_unit": point.coordinate_unit,
        "geometry_sha256": geometry_sha256(point.atoms),
        "energy_eV": -1.0,
        "local_tangent_force_fd": {
            "all_gates_passed": passed,
            "fixed_topology": passed,
            "maximum_primal_residual": 1e-13,
        },
        "cold_warm": {
            "numerically_equivalent": passed,
            "gates": {"source": passed, "energy": passed},
        },
        "maximum_primal_residual": 1e-13,
        "adjoint_residual": 1e-14,
        "topology_hash": f"{point.path_name:0<64}"[:64],
    }


def test_path_summary_requires_every_real_point_and_preserves_failure():
    points = tuple(point for path in panel_paths().values() for point in path)
    records = [_path_record(point) for point in points]
    summary = summarize_pes_paths(records)
    assert summary["path_count"] == 2
    assert summary["path_geometry_count"] == 11
    assert summary["all_gates_passed"] is True

    failed = deepcopy(records)
    failed[-1]["local_tangent_force_fd"]["all_gates_passed"] = False
    assert summarize_pes_paths(failed)["all_gates_passed"] is False
    with pytest.raises(ValueError, match="exactly 11"):
        summarize_pes_paths(records[:-1])
