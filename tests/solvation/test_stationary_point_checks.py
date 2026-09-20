from __future__ import annotations

# pyright: reportMissingImports=false
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from stationary_point_checks import analyze_stationary_point


def _water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.7586, 0.0, 0.5043],
            [-0.7586, 0.0, 0.5043],
        ],
    )


def _hessian_with_vibrational_eigenvalues(
    atoms: Atoms,
    eigenvalues: list[float],
) -> np.ndarray:
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    center = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
    rigid = np.zeros((3 * len(atoms), 6), dtype=np.float64)
    for index, (mass, position) in enumerate(zip(masses, positions)):
        root_mass = np.sqrt(mass)
        offset = position - center
        rigid[3 * index : 3 * index + 3, :3] = root_mass * np.eye(3)
        rigid[3 * index : 3 * index + 3, 3:] = root_mass * np.asarray(
            [
                [0.0, -offset[2], offset[1]],
                [offset[2], 0.0, -offset[0]],
                [-offset[1], offset[0], 0.0],
            ]
        )
    left, singular_values, _ = np.linalg.svd(rigid, full_matrices=True)
    tolerance = np.finfo(np.float64).eps * max(rigid.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    vibrational_basis = left[:, rank:]
    assert vibrational_basis.shape[1] == len(eigenvalues)
    mass_weighted = vibrational_basis @ np.diag(eigenvalues) @ vibrational_basis.T
    root_masses = np.repeat(np.sqrt(masses), 3)
    return mass_weighted * root_masses[:, None] * root_masses[None, :]


def test_rejects_minimum_as_first_order_saddle():
    atoms = _water()
    hessian = _hessian_with_vibrational_eigenvalues(atoms, [0.1, 0.2, 0.3])

    result = analyze_stationary_point(atoms, hessian, np.zeros((3, 3)))

    assert result["is_minimum"] is True
    assert result["is_first_order_saddle"] is False
    assert result["significant_negative_count"] == 0


def test_rejects_higher_order_saddle_as_first_order_saddle():
    atoms = _water()
    hessian = _hessian_with_vibrational_eigenvalues(atoms, [-0.2, -0.1, 0.3])

    result = analyze_stationary_point(atoms, hessian, np.zeros((3, 3)))

    assert result["is_first_order_saddle"] is False
    assert result["significant_negative_count"] == 2


def test_rejects_nonstationary_geometry_even_with_one_negative_mode():
    atoms = _water()
    hessian = _hessian_with_vibrational_eigenvalues(atoms, [-0.2, 0.1, 0.3])
    forces = np.zeros((3, 3))
    forces[0, 0] = 3.0e-4

    result = analyze_stationary_point(atoms, hessian, forces)

    assert result["significant_negative_count"] == 1
    assert result["force_converged"] is False
    assert result["is_first_order_saddle"] is False


@pytest.mark.parametrize("field", ["hessian", "forces"])
def test_rejects_nonfinite_numerical_input(field):
    atoms = _water()
    hessian = _hessian_with_vibrational_eigenvalues(atoms, [-0.2, 0.1, 0.3])
    forces = np.zeros((3, 3))
    if field == "hessian":
        hessian[0, 0] = np.nan
    else:
        forces[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        analyze_stationary_point(atoms, hessian, forces)


def test_accepts_stationary_first_order_saddle_and_returns_negative_mode():
    atoms = _water()
    hessian = _hessian_with_vibrational_eigenvalues(atoms, [-0.2, 0.1, 0.3])

    result = analyze_stationary_point(atoms, hessian, np.zeros((3, 3)))

    assert result["force_converged"] is True
    assert result["is_first_order_saddle"] is True
    assert result["significant_negative_count"] == 1
    assert result["negative_mode_cartesian"].shape == (3, 3)
    assert np.max(np.linalg.norm(result["negative_mode_cartesian"], axis=1)) > 0.0
    assert len(result["raw_frequencies_cm1"]) == 9
    assert len(result["physical_frequencies_cm1"]) == 9


def test_retains_soft_imaginary_modes_below_classification_cutoff():
    atoms = _water()
    soft_eigenvalue = -(20.0 / 2720.22864939427) ** 2
    hessian = _hessian_with_vibrational_eigenvalues(
        atoms,
        [soft_eigenvalue, 0.1, 0.3],
    )

    result = analyze_stationary_point(atoms, hessian, np.zeros((3, 3)))

    assert result["significant_negative_count"] == 0
    assert np.any(result["physical_frequencies_cm1"] < -10.0)
    assert np.any(result["physical_frequencies_cm1"] > -30.0)


def test_rejects_badly_asymmetric_raw_hessian_before_classification():
    atoms = _water()
    hessian = _hessian_with_vibrational_eigenvalues(atoms, [0.1, 0.2, 0.3])
    hessian[0, 1] += 100.0
    hessian[1, 0] -= 100.0

    result = analyze_stationary_point(atoms, hessian, np.zeros((3, 3)))

    assert result["is_minimum_candidate"] is True
    assert result["hessian_quality_pass"] is False
    assert result["is_minimum"] is False
