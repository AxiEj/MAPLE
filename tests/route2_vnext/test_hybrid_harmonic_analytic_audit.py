from __future__ import annotations

from ase import Atoms
from ase.units import Bohr
import numpy as np
import pytest
from scipy.special import erf
import torch

from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.harmonic_point_source import (
    point_harmonic_source_operator,
)
from maple.solvation.continuum.harmonic_torch_functional import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)

RADIUS_ANGSTROM = 2.3


def _screening_factor(dielectric: float | None) -> float:
    return 1.0 if dielectric is None else (dielectric - 1.0) / dielectric


def _matrices(dielectric: float | None) -> tuple[object, dict[str, np.ndarray]]:
    continuum = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=(1,),
        radii_angstrom=(RADIUS_ANGSTROM,),
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=48,
        source_radial_quadrature_order=64,
        green_radial_quadrature_order=64,
        dtype=torch.float64,
        device="cpu",
        dielectric=dielectric,
        scalar_id=(
            EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
        ),
    )
    return continuum, continuum.debug_geometry_matrices(
        Atoms("H", positions=[[0.0, 0.0, 0.0]])
    )


def _energy(
    surface_operator: np.ndarray,
    right_hand_side: np.ndarray,
    *,
    dielectric: float | None,
) -> float:
    state = np.linalg.solve(surface_operator, right_hand_side)
    return -0.5 * _screening_factor(dielectric) * float(np.vdot(right_hand_side, state))


@pytest.mark.parametrize("dielectric", (None, 2.0, 78.355))
def test_one_sphere_point_monopole_and_dipoles_match_exact_cpcm_modes(
    dielectric: float | None,
) -> None:
    continuum, matrices = _matrices(dielectric)
    raw_point = point_harmonic_source_operator(
        positions_angstrom=np.zeros((1, 3)),
        radii_angstrom=(RADIUS_ANGSTROM,),
        surface_lmax=continuum.physical_lmax,
        radial_quadrature_order=64,
    )
    point = matrices["weighted_basis"].T @ raw_point
    surface = matrices["surface_operator"]
    factor = _screening_factor(dielectric)
    point_charge_energy = -0.5 * factor * HARTREE_TO_EV * Bohr / RADIUS_ANGSTROM
    point_dipole_energy = -0.5 * factor * HARTREE_TO_EV * Bohr / RADIUS_ANGSTROM**3

    assert _energy(surface, point[:, 0], dielectric=dielectric) == pytest.approx(
        point_charge_energy, rel=2.0e-14, abs=2.0e-14
    )
    for column in (1, 2, 3):
        assert _energy(
            surface, point[:, column], dielectric=dielectric
        ) == pytest.approx(point_dipole_energy, rel=2.0e-14, abs=2.0e-14)


@pytest.mark.parametrize("dielectric", (None, 2.0, 78.355))
@pytest.mark.parametrize(
    "sigma_angstrom,charge_column,dipole_columns",
    ((1.5, 0, (2, 3, 4)), (3.0, 1, (5, 6, 7))),
)
def test_one_sphere_gaussian_monopole_and_dipoles_match_exact_cpcm_modes(
    dielectric: float | None,
    sigma_angstrom: float,
    charge_column: int,
    dipole_columns: tuple[int, int, int],
) -> None:
    _continuum, matrices = _matrices(dielectric)
    surface = matrices["surface_operator"]
    source = matrices["source_operator"]
    radius_bohr = RADIUS_ANGSTROM / Bohr
    sigma_bohr = sigma_angstrom / Bohr
    argument = radius_bohr / (np.sqrt(2.0) * sigma_bohr)
    kernel = erf(argument) / radius_bohr
    radial_derivative_per_angstrom = (
        np.sqrt(2.0 / np.pi) * np.exp(-(argument**2)) / (sigma_bohr * radius_bohr)
        - erf(argument) / radius_bohr**2
    ) / Bohr
    common = -0.5 * _screening_factor(dielectric) * HARTREE_TO_EV * radius_bohr
    charge_energy = common * kernel**2
    dipole_energy = common * radial_derivative_per_angstrom**2

    assert _energy(
        surface, source[:, charge_column], dielectric=dielectric
    ) == pytest.approx(charge_energy, rel=3.0e-14, abs=3.0e-14)
    for column in dipole_columns:
        assert _energy(
            surface, source[:, column], dielectric=dielectric
        ) == pytest.approx(dipole_energy, rel=3.0e-14, abs=3.0e-14)
