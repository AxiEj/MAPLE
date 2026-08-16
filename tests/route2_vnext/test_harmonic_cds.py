from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from ase import Atoms
import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    aqueous_atomic_surface_tensions,
    smd_sasa_radii,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.harmonic_cds_area import (
    SmoothHarmonicExposureArea,
)
from maple.solvation.harmonic_cds import (
    SmoothHarmonicAqueousLinearCDSTerm,
    build_stock_smd_water_harmonic_cds,
)
from maple.solvation.solvent_terms import SolventEnergyTerm

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("O", "C", "N", "H", "H", "H", "H", "H")
POSITIONS = np.asarray(
    [
        [0.00, 0.00, 0.00],
        [1.31, 0.07, -0.03],
        [2.53, 0.19, 0.11],
        [-0.52, 0.86, 0.02],
        [-0.46, -0.89, -0.09],
        [1.39, 1.02, 0.17],
        [1.48, -0.91, -0.21],
        [3.04, 0.89, 0.03],
    ]
)
RADII = tuple(float(value) for value in smd_sasa_radii(SYMBOLS))


def _area(*, width: float = 0.20):
    torch = pytest.importorskip("torch")
    atoms = Atoms(SYMBOLS, positions=POSITIONS)
    return SmoothHarmonicExposureArea(
        atomic_numbers=tuple(int(value) for value in atoms.numbers),
        radii_angstrom=RADII,
        transition_width_angstrom2=width,
        exposure_lmax=4,
        radial_quadrature_order=48,
        dtype=torch.float64,
        device="cpu",
    )


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def test_import_is_dependency_light() -> None:
    script = r"""
import sys
sys.modules['torch'] = None
sys.modules['pyscf'] = None
from maple.solvation.harmonic_cds import SmoothHarmonicAqueousLinearCDSTerm
assert SmoothHarmonicAqueousLinearCDSTerm.provider_id.endswith('.impl.v1')
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_stock_term_is_one_linear_scalar_and_structurally_conforms() -> None:
    atoms = Atoms(SYMBOLS, positions=POSITIONS)
    term = build_stock_smd_water_harmonic_cds(symbols=SYMBOLS, area=_area())
    assert isinstance(term, SolventEnergyTerm)
    state = term.evaluate(atoms, need_gradient=True)
    design = term.design_row_kcal_mol(atoms)
    energy_kcal_mol = float(
        design @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    )
    assert state.energy_eV == pytest.approx(
        energy_kcal_mol * HARTREE_TO_EV / HARTREE_TO_KCAL_MOL,
        abs=2.0e-15,
    )
    areas = term.area.atom_areas_angstrom2(atoms)
    tensions = aqueous_atomic_surface_tensions(SYMBOLS, POSITIONS)
    assert energy_kcal_mol == pytest.approx(
        float(np.dot(areas, tensions) / 1000.0), abs=2.0e-14
    )
    assert state.gradient_eV_per_A is not None
    assert state.topology_observation_coverage == "complete"
    assert state.unobservable_topology_components == ()


def test_arbitrary_frozen_coefficients_and_complete_gradient_match_scalar_fd() -> None:
    atoms = Atoms(SYMBOLS, positions=POSITIONS)
    rng = np.random.default_rng(1907)
    coefficients = rng.normal(scale=35.0, size=18)
    term = SmoothHarmonicAqueousLinearCDSTerm(
        symbols=SYMBOLS,
        area=_area(),
        coefficients_cal_mol_angstrom2=coefficients,
    )
    state = term.evaluate(atoms, need_gradient=True)
    assert state.gradient_eV_per_A is not None
    direction = rng.normal(size=POSITIONS.shape)
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    analytic = float(np.vdot(state.gradient_eV_per_A, direction))

    def energy(displaced: np.ndarray) -> float:
        return term.evaluate(
            Atoms(SYMBOLS, positions=displaced), need_gradient=False
        ).energy_eV

    errors = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        coarse = (
            energy(POSITIONS + step * direction) - energy(POSITIONS - step * direction)
        ) / (2.0 * step)
        half = step / 2.0
        fine = (
            energy(POSITIONS + half * direction) - energy(POSITIONS - half * direction)
        ) / (2.0 * half)
        richardson = (4.0 * fine - coarse) / 3.0
        errors.append(abs(analytic - richardson))
    assert errors[-1] < 2.0e-9
    assert errors[-1] < errors[0] / 50.0
    np.testing.assert_allclose(
        np.sum(state.gradient_eV_per_A, axis=0), 0.0, atol=3.0e-13, rtol=0.0
    )


def test_scalar_and_gradient_are_rotation_translation_and_permutation_covariant() -> (
    None
):
    atoms = Atoms(SYMBOLS, positions=POSITIONS)
    term = build_stock_smd_water_harmonic_cds(symbols=SYMBOLS, area=_area())
    reference = term.evaluate(atoms, need_gradient=True)
    assert reference.gradient_eV_per_A is not None
    rotation = _rotation(877)
    shift = np.asarray([0.31, -0.62, 0.27])
    transformed_positions = POSITIONS @ rotation.T + shift
    transformed = term.evaluate(
        Atoms(SYMBOLS, positions=transformed_positions), need_gradient=True
    )
    assert transformed.energy_eV == pytest.approx(reference.energy_eV, abs=2.0e-14)
    np.testing.assert_allclose(
        transformed.gradient_eV_per_A,
        reference.gradient_eV_per_A @ rotation.T,
        atol=3.0e-12,
        rtol=0.0,
    )

    permutation = np.asarray([2, 0, 1, 7, 3, 4, 5, 6])
    permuted_symbols = tuple(SYMBOLS[index] for index in permutation)
    permuted_radii = tuple(RADII[index] for index in permutation)
    permuted_area = SmoothHarmonicExposureArea(
        atomic_numbers=tuple(int(value) for value in Atoms(permuted_symbols).numbers),
        radii_angstrom=permuted_radii,
        transition_width_angstrom2=0.20,
        exposure_lmax=4,
        radial_quadrature_order=48,
        dtype=pytest.importorskip("torch").float64,
        device="cpu",
    )
    permuted_term = build_stock_smd_water_harmonic_cds(
        symbols=permuted_symbols,
        area=permuted_area,
    )
    permuted = permuted_term.evaluate(
        Atoms(permuted_symbols, positions=POSITIONS[permutation]),
        need_gradient=True,
    )
    assert permuted.energy_eV == pytest.approx(reference.energy_eV, abs=3.0e-14)
    np.testing.assert_allclose(
        permuted.gradient_eV_per_A,
        reference.gradient_eV_per_A[permutation],
        atol=3.0e-12,
        rtol=0.0,
    )


def test_configuration_and_geometry_mismatch_fail_closed() -> None:
    area = _area()
    term = build_stock_smd_water_harmonic_cds(symbols=SYMBOLS, area=area)
    changed = build_stock_smd_water_harmonic_cds(
        symbols=SYMBOLS, area=_area(width=0.21)
    )
    assert changed.configuration_sha256() != term.configuration_sha256()
    with pytest.raises(ValueError, match="atomic numbers"):
        build_stock_smd_water_harmonic_cds(
            symbols=("C",) + SYMBOLS[1:],
            area=area,
        )
    wrong_radii = SmoothHarmonicExposureArea(
        atomic_numbers=tuple(int(value) for value in Atoms(SYMBOLS).numbers),
        radii_angstrom=tuple(value - 0.4 for value in RADII),
        transition_width_angstrom2=0.20,
        exposure_lmax=4,
        radial_quadrature_order=48,
        dtype=pytest.importorskip("torch").float64,
        device="cpu",
    )
    with pytest.raises(ValueError, match="published SMD SASA radii"):
        build_stock_smd_water_harmonic_cds(symbols=SYMBOLS, area=wrong_radii)
    with pytest.raises(ValueError, match="geometry symbols"):
        term.evaluate(Atoms("COCHHHHH", positions=POSITIONS), need_gradient=False)
    with pytest.raises(TypeError, match="need_gradient must be bool"):
        term.evaluate(Atoms(SYMBOLS, positions=POSITIONS), need_gradient=1)
    with pytest.raises(AttributeError, match="immutable"):
        term._profile_id = "changed"
