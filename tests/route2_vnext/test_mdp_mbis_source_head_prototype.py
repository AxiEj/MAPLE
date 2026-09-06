from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_release/run_mdp_mbis_source_head_prototype.py"
)
SPEC = importlib.util.spec_from_file_location("mdp_mbis_source_head_prototype", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _rotation() -> np.ndarray:
    axis = np.asarray((0.3, -0.5, 0.8), dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def test_joint_projection_closes_and_is_rotation_covariant() -> None:
    positions = np.asarray(((0.1, -0.2, 0.4), (1.2, 0.5, -0.7)))
    charges = np.asarray((0.2, -0.1))
    dipoles = np.asarray(((0.03, -0.04, 0.02), (-0.01, 0.05, 0.06)))
    target_charge = -0.3
    target_dipole = np.asarray((0.7, -0.2, 0.4))
    projected_q, projected_p, condition = MODULE.project_charge_dipole_source(
        positions_angstrom=positions,
        raw_charges_e=charges,
        raw_dipoles_eangstrom=dipoles,
        total_charge_e=target_charge,
        molecular_dipole_eangstrom=target_dipole,
        charge_sigma_e=0.08,
        dipole_sigma_eangstrom=0.03,
    )
    assert np.isfinite(condition)
    assert np.allclose(
        MODULE.charge_dipole_constraints(positions, projected_q, projected_p),
        np.concatenate(([target_charge], target_dipole)),
        rtol=0.0,
        atol=2.0e-12,
    )

    rotation = _rotation()
    rotated_q, rotated_p, _ = MODULE.project_charge_dipole_source(
        positions_angstrom=positions @ rotation.T,
        raw_charges_e=charges,
        raw_dipoles_eangstrom=dipoles @ rotation.T,
        total_charge_e=target_charge,
        molecular_dipole_eangstrom=target_dipole @ rotation.T,
        charge_sigma_e=0.08,
        dipole_sigma_eangstrom=0.03,
    )
    assert np.allclose(rotated_q, projected_q, rtol=0.0, atol=2.0e-13)
    assert np.allclose(rotated_p, projected_p @ rotation.T, rtol=0.0, atol=2.0e-13)


def test_joint_projection_is_translation_and_permutation_covariant() -> None:
    positions = np.asarray(((0.0, 0.2, -0.4), (1.1, -0.3, 0.8), (-0.7, 0.6, 0.2)))
    charges = np.asarray((0.3, -0.2, 0.1))
    dipoles = np.asarray(((0.01, 0.02, 0.03), (0.04, -0.02, 0.0), (-0.01, 0.03, 0.02)))
    total_charge = 1.0
    target_dipole = np.asarray((0.5, -0.6, 0.2))
    projected_q, projected_p, _ = MODULE.project_charge_dipole_source(
        positions_angstrom=positions,
        raw_charges_e=charges,
        raw_dipoles_eangstrom=dipoles,
        total_charge_e=total_charge,
        molecular_dipole_eangstrom=target_dipole,
        charge_sigma_e=0.1,
        dipole_sigma_eangstrom=0.04,
    )
    translation = np.asarray((1.2, -0.8, 0.5))
    translated_q, translated_p, _ = MODULE.project_charge_dipole_source(
        positions_angstrom=positions + translation,
        raw_charges_e=charges,
        raw_dipoles_eangstrom=dipoles,
        total_charge_e=total_charge,
        molecular_dipole_eangstrom=target_dipole + total_charge * translation,
        charge_sigma_e=0.1,
        dipole_sigma_eangstrom=0.04,
    )
    assert np.allclose(translated_q, projected_q, rtol=0.0, atol=2.0e-13)
    assert np.allclose(translated_p, projected_p, rtol=0.0, atol=2.0e-13)

    permutation = np.asarray((2, 0, 1))
    permuted_q, permuted_p, _ = MODULE.project_charge_dipole_source(
        positions_angstrom=positions[permutation],
        raw_charges_e=charges[permutation],
        raw_dipoles_eangstrom=dipoles[permutation],
        total_charge_e=total_charge,
        molecular_dipole_eangstrom=target_dipole,
        charge_sigma_e=0.1,
        dipole_sigma_eangstrom=0.04,
    )
    assert np.allclose(permuted_q, projected_q[permutation], rtol=0.0, atol=2.0e-13)
    assert np.allclose(permuted_p, projected_p[permutation], rtol=0.0, atol=2.0e-13)


def test_joint_projection_remains_full_rank_for_one_atom() -> None:
    q, p, condition = MODULE.project_charge_dipole_source(
        positions_angstrom=np.asarray(((3.0, -2.0, 1.0),)),
        raw_charges_e=np.asarray((0.0,)),
        raw_dipoles_eangstrom=np.zeros((1, 3)),
        total_charge_e=1.0,
        molecular_dipole_eangstrom=np.asarray((3.2, -1.7, 0.6)),
        charge_sigma_e=0.2,
        dipole_sigma_eangstrom=0.1,
    )
    assert np.isfinite(condition)
    assert np.allclose(q, (1.0,), rtol=0.0, atol=2.0e-13)
    assert np.allclose(p, ((0.2, 0.3, -0.4),), rtol=0.0, atol=2.0e-13)


def test_ridge_and_molecule_split_are_deterministic() -> None:
    features = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    targets = np.asarray((2.0, -1.0, 1.0))
    first = MODULE.fit_ridge(features, targets, 1.0e-8)
    second = MODULE.fit_ridge(features, targets, 1.0e-8)
    assert np.array_equal(first, second)
    assert MODULE.molecule_split("molecule-A") == MODULE.molecule_split("molecule-A")
    assert MODULE.molecule_split("molecule-A") in {"train", "validation", "test"}


def test_help_does_not_require_heavy_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "MBIS source readout" in completed.stdout

