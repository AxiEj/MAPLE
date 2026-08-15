from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.gto_density import (
    gaussian_multipole_potential,
)
from maple.solvation.continuum import (
    build_harmonic_gaussian_source,
    build_smooth_harmonic_exposure,
)
from maple.solvation.release.harmonic_surface_mep import (
    HarmonicSurfaceMEPProjector,
)

POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [2.18, -0.37, 0.29],
        [-0.42, 2.06, -0.33],
    ]
)
RADII = (1.52, 1.21, 1.70)
SOURCE8 = np.asarray(
    [
        [0.21, -0.03, 0.04, -0.02, 0.01, 0.006, -0.004, 0.009],
        [-0.16, 0.02, -0.03, 0.01, 0.02, -0.008, 0.005, -0.006],
        [-0.05, 0.01, 0.02, 0.03, -0.01, 0.002, -0.007, 0.004],
    ]
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_snapshot():
    exposure = build_smooth_harmonic_exposure(
        atomic_numbers=(8, 1, 6),
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.20,
        surface_lmax=3,
        exposure_lmax=6,
        radial_quadrature_order=48,
    )
    return build_harmonic_gaussian_source(
        exposure,
        radial_quadrature_order=64,
    )


def _projector(polar_order: int = 16) -> tuple[object, HarmonicSurfaceMEPProjector]:
    source = _source_snapshot()
    projector = HarmonicSurfaceMEPProjector(
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        physical_lmax=source.physical_charge_lmax,
        surface_lmax=source.surface_lmax,
        polar_order=polar_order,
        weighted_basis_operator=source.weighted_basis_operator,
        geometry_sha256=source.exposure.geometry_sha256,
        continuum_configuration_sha256=_digest("continuum-config"),
        continuum_provenance_sha256=_digest("continuum-provenance"),
        topology_sha256=source.exposure.topology_sha256,
        cavity_profile_id="test.smooth-harmonic-cavity.v1",
    )
    return source, projector


def _gaussian_potential(projector: HarmonicSurfaceMEPProjector) -> np.ndarray:
    potential = np.zeros(projector.atom_count * projector.points_per_atom)
    for sigma, columns in (
        (1.5, (0, 2, 3, 4)),
        (3.0, (1, 5, 6, 7)),
    ):
        potential += gaussian_multipole_potential(
            projector.points_bohr,
            POSITIONS,
            SOURCE8[:, columns],
            sigma_angstrom=sigma,
        )
    return potential.reshape(projector.atom_count, projector.points_per_atom)


def test_direct_mep_projection_matches_existing_harmonic_B_operator():
    source, projector = _projector()
    projection = projector.project(_gaussian_potential(projector))
    expected_raw = (source.raw_source_operator @ SOURCE8.reshape(-1)).reshape(
        source.atom_count, -1
    )
    expected_rhs = source.source_operator @ SOURCE8.reshape(-1)
    np.testing.assert_allclose(
        projection.raw_harmonic_coefficients_ev_per_e,
        expected_raw,
        atol=1.5e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        projection.boundary_rhs,
        expected_rhs,
        atol=1.5e-12,
        rtol=0.0,
    )
    assert projection.projector_configuration_sha256 == (projector.configuration_sha256)
    assert projection.as_dict()["capability_admitted"] is False


def test_projector_is_content_addressed_immutable_and_exposes_reference_weights():
    _, first = _projector(16)
    _, replay = _projector(16)
    _, changed = _projector(18)
    assert first.configuration_sha256 == replay.configuration_sha256
    assert first.configuration_sha256 != changed.configuration_sha256
    assert first.points_bohr.flags.writeable is False
    assert first.weighted_basis_operator.flags.writeable is False
    weights = first.squared_exposure_angular_weights
    assert weights.shape == (first.atom_count, first.points_per_atom)
    assert np.all(weights >= 0.0)
    assert np.all(np.sum(weights, axis=1) > 0.0)
    assert first.as_dict()["quadrature_role"].startswith("reference-integration")


def test_projector_fails_closed_on_wrong_chart_or_potential():
    source, projector = _projector()
    with pytest.raises(ValueError, match="one value per projector point"):
        projector.project(np.zeros(projector.points_per_atom))
    bad_basis = source.weighted_basis_operator.copy()
    physical = source.physical_charge_space.single_atom_dimension
    surface = source.surface_space.single_atom_dimension
    bad_basis[:physical, surface] = 1.0e-9
    with pytest.raises(ValueError, match="atom-major ownership"):
        HarmonicSurfaceMEPProjector(
            positions_angstrom=POSITIONS,
            radii_angstrom=RADII,
            physical_lmax=source.physical_charge_lmax,
            surface_lmax=source.surface_lmax,
            polar_order=16,
            weighted_basis_operator=bad_basis,
            geometry_sha256=source.exposure.geometry_sha256,
            continuum_configuration_sha256=_digest("continuum-config"),
            continuum_provenance_sha256=_digest("continuum-provenance"),
            topology_sha256=source.exposure.topology_sha256,
            cavity_profile_id="test.smooth-harmonic-cavity.v1",
        )
