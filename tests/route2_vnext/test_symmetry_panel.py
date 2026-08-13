from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from maple.solvation.release.symmetry_panel import (
    FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION,
    SYMMETRY_PANEL_CONTRACT_VERSION,
    SYMMETRY_PANEL_TRANSLATION_A,
    rotate_radial_gto_blocks,
    summarize_bidirectional_loop_record,
    summarize_rigid_symmetry,
    summarize_symmetry_panel,
    symmetry_loop_point_label,
    symmetry_panel_permutation,
    symmetry_panel_rotations,
)


def test_loop_point_context_depends_only_on_physical_geometry():
    assert symmetry_loop_point_label((0.0, -1.0)) == "loop/point/+0.00000000/-1.00000000"
    assert symmetry_loop_point_label(np.asarray([0.0, -1.0])) == (
        "loop/point/+0.00000000/-1.00000000"
    )
    with pytest.raises(ValueError, match="finite two-vector"):
        symmetry_loop_point_label((0.0, np.nan))


def _rigid_record(molecule_id="water"):
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # A central equal/opposite pair has zero net force and torque.
    forces = np.asarray([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    source = np.arange(16, dtype=float).reshape(2, 8) / 100.0
    energy = -1.25
    topology = "a" * 64
    rotations = symmetry_panel_rotations(molecule_id)
    records = []
    for index, rotation in enumerate(rotations):
        records.append(
            {
                "index": index,
                "rotation_matrix": rotation,
                "expected_rotation_matrix": rotation,
                "energy_eV": energy,
                "forces_eV_per_A": forces @ rotation.T,
                "source": rotate_radial_gto_blocks(source, rotation),
                "primal_residual": 1.0e-13,
                "adjoint_residual": 1.0e-14,
                "topology_hash": topology,
            }
        )
    translation = {
        "translation_A": SYMMETRY_PANEL_TRANSLATION_A,
        "energy_eV": energy,
        "forces_eV_per_A": forces,
        "source": source,
        "primal_residual": 1.0e-13,
        "adjoint_residual": 1.0e-14,
        "topology_hash": topology,
    }
    permutation = symmetry_panel_permutation(np.asarray([1, 1]))
    permutation_record = {
        "permutation": permutation,
        "energy_eV": energy,
        "forces_eV_per_A": forces[permutation],
        "source": source[permutation],
        "primal_residual": 1.0e-13,
        "adjoint_residual": 1.0e-14,
        "topology_hash": topology,
    }
    return (
        positions,
        forces,
        source,
        energy,
        topology,
        translation,
        permutation_record,
        records,
    )


def test_rotations_are_deterministic_proper_and_molecule_specific():
    water = symmetry_panel_rotations("water")
    np.testing.assert_array_equal(water, symmetry_panel_rotations("water"))
    assert not np.array_equal(water, symmetry_panel_rotations("methanol"))
    for rotation in water:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2e-15)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=2e-15)


def test_rigid_symmetry_recomputes_exact_invariant_or_covariant_values():
    positions, forces, source, energy, topology, translation, permutation, rotations = (
        _rigid_record()
    )
    summary = summarize_rigid_symmetry(
        positions_A=positions,
        base_energy_eV=energy,
        base_forces_eV_per_A=forces,
        base_source=source,
        base_topology_hash=topology,
        translation_record=translation,
        permutation_record=permutation,
        rotation_records=rotations,
    )
    assert summary["all_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["base_net_force_norm_eV_per_A"] == pytest.approx(0.0)
    assert summary["base_torque_norm_eV"] == pytest.approx(0.0)


def test_rigid_symmetry_fails_force_covariance_and_rejects_rotation_spoof():
    positions, forces, source, energy, topology, translation, permutation, rotations = (
        _rigid_record()
    )
    broken = deepcopy(rotations)
    broken[1]["forces_eV_per_A"] = np.asarray(
        broken[1]["forces_eV_per_A"]
    ) + 1.0e-3
    summary = summarize_rigid_symmetry(
        positions_A=positions,
        base_energy_eV=energy,
        base_forces_eV_per_A=forces,
        base_source=source,
        base_topology_hash=topology,
        translation_record=translation,
        permutation_record=permutation,
        rotation_records=broken,
    )
    assert summary["gates"][
        "all_rotation_force_covariance_relative_le_1e-4"
    ] is False
    assert summary["all_gates_passed"] is False

    broken = deepcopy(rotations)
    broken[0]["expected_rotation_matrix"] = np.eye(3)
    with pytest.raises(ValueError, match="out of contract"):
        summarize_rigid_symmetry(
            positions_A=positions,
            base_energy_eV=energy,
            base_forces_eV_per_A=forces,
            base_source=source,
            base_topology_hash=topology,
            translation_record=translation,
            permutation_record=permutation,
            rotation_records=broken,
        )


def test_symmetry_panel_requires_all_20_records_and_preserves_loop_failure():
    from maple.solvation.release.pes_panel import load_pes_panel

    records = []
    for molecule in load_pes_panel():
        records.append(
            {
                "molecule_id": molecule.molecule_id,
                "contract_version": SYMMETRY_PANEL_CONTRACT_VERSION,
                "rigid_symmetry": {
                    "all_gates_passed": True,
                    "gates": {"rigid": True},
                    "translation": {
                        "energy_abs_eV": 0.0,
                        "net_force_norm_eV_per_A": 0.0,
                    },
                    "base_net_force_norm_eV_per_A": 0.0,
                    "base_torque_norm_eV": 0.0,
                    "maximum_rotation_energy_abs_eV": 0.0,
                    "maximum_rotation_force_covariance_relative": 0.0,
                        "maximum_rotation_source_covariance_relative": 0.0,
                        "permutation": {
                            "energy_abs_eV": 0.0,
                            "force_covariance_relative": 0.0,
                            "source_covariance_relative": 0.0,
                        },
                    "maximum_primal_residual": 1.0e-13,
                    "maximum_adjoint_residual": 1.0e-14,
                },
                "closed_loop": {
                    "all_gates_passed": True,
                    "gates": {"loop": True},
                    "fixed_topology": True,
                    "maximum_absolute_loop_work_eV": 1.0e-7,
                    "maximum_primal_residual": 1.0e-13,
                    "maximum_adjoint_residual": 1.0e-14,
                },
            }
        )
    summary = summarize_symmetry_panel(records)
    assert summary["molecule_count"] == 20
    assert summary["all_gates_passed"] is True
    failed = deepcopy(records)
    failed[-1]["closed_loop"]["gates"]["loop"] = False
    assert summarize_symmetry_panel(failed)["all_gates_passed"] is False
    with pytest.raises(ValueError, match="coverage"):
        summarize_symmetry_panel(records[:-1])

    high_order = deepcopy(records)
    for record in high_order:
        record["contract_version"] = FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION
    high_order_summary = summarize_symmetry_panel(
        high_order,
        contract_version=FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION,
    )
    assert high_order_summary["all_gates_passed"] is True
    assert high_order_summary["contract_version"] == (
        FIXEDBOX1202_SYMMETRY_PANEL_CONTRACT_VERSION
    )
    with pytest.raises(ValueError, match="Unknown symmetry-panel"):
        summarize_symmetry_panel(high_order, contract_version="forged-contract")


def test_bidirectional_loop_summary_recomputes_work_and_residual_gates():
    record = {
        name: {"simpson_work_eV": sign * 2.0e-6, "gate_threshold_eV": 1.0e-5}
        for name, sign in (
            ("cold_forward", 1.0),
            ("cold_reverse", -1.0),
            ("warm_forward", 1.0),
            ("warm_reverse", -1.0),
        )
    }
    record.update(
        {
            "maximum_primal_residual": 1.0e-13,
            "maximum_adjoint_residual": 1.0e-14,
            "topology_hashes": ["b" * 64],
            "all_cold_warm_roots": True,
            "warm_forward_reverse_repeat": True,
        }
    )
    summary = summarize_bidirectional_loop_record(record)
    assert summary["all_gates_passed"] is True
    failed = deepcopy(record)
    failed["warm_forward"]["simpson_work_eV"] = 2.0e-5
    assert summarize_bidirectional_loop_record(failed)["all_gates_passed"] is False
