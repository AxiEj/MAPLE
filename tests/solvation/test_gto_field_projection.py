from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from ase.units import Bohr, Hartree
from numpy.polynomial.hermite import hermgauss

from maple.function.calculator.extra_correction.implicit.gto_field_projection import (
    ExactGTOFieldProjector,
    MACEPolarGTOFieldProjectionSpec,
)


_RECEIVER_SIGMAS_ANGSTROM = (1.5, 3.0)
_UPSTREAM_MATRIX = np.asarray(
    [
        [3.544907701811032, 0.0, 0.0, 0.0],
        [3.544907701811032, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.771474235728387],
        [0.0, 5.771474235728387, 0.0, 0.0],
        [0.0, 0.0, 5.771474235728387, 0.0],
        [0.0, 0.0, 0.0, 11.542948471456774],
        [0.0, 11.542948471456774, 0.0, 0.0],
        [0.0, 0.0, 11.542948471456774, 0.0],
    ]
)


def _projector() -> ExactGTOFieldProjector:
    return ExactGTOFieldProjector(
        MACEPolarGTOFieldProjectionSpec(
            receiver_sigmas_angstrom=_RECEIVER_SIGMAS_ANGSTROM,
            receiver_max_l=1,
            receiver_normalization="receiver",
            upstream_matrix=_UPSTREAM_MATRIX,
        )
    )


def _gauss_hermite_reaction_moments(
    center_bohr: np.ndarray,
    sigma_angstrom: float,
    surface_centers_bohr: np.ndarray,
    asc_e: np.ndarray,
    *,
    order: int = 18,
) -> tuple[float, np.ndarray]:
    """Independent 3-D quadrature of Gaussian-averaged V and grad(V)."""

    nodes, weights = hermgauss(order)
    sigma_bohr = sigma_angstrom / Bohr
    normalization = math.pi ** (-1.5)
    potential = 0.0
    x_potential = np.zeros(3)
    for i, j, k in itertools.product(range(order), repeat=3):
        displacement = (
            math.sqrt(2.0)
            * sigma_bohr
            * np.asarray([nodes[i], nodes[j], nodes[k]])
        )
        point = center_bohr + displacement
        radii = np.linalg.norm(surface_centers_bohr - point, axis=1)
        value = float(np.dot(asc_e, 1.0 / radii))
        weight = normalization * weights[i] * weights[j] * weights[k]
        potential += weight * value
        x_potential += weight * displacement * value
    gradient = x_potential / sigma_bohr**2
    return potential, gradient


def test_affine_potential_exact_projection_reduces_to_upstream_local_jet():
    projector = _projector()
    potential_ev = np.asarray([0.3, -0.2])
    gradient_ev_per_angstrom = np.asarray(
        [[0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]]
    )
    potentials = np.repeat(
        potential_ev[None, :],
        len(_RECEIVER_SIGMAS_ANGSTROM),
        axis=0,
    )
    gradients = np.repeat(
        gradient_ev_per_angstrom[None, :, :],
        len(_RECEIVER_SIGMAS_ANGSTROM),
        axis=0,
    )

    actual = projector.project_smoothed_fields(
        potentials,
        gradients,
    )
    local_values = np.concatenate(
        (potential_ev[:, None], gradient_ev_per_angstrom),
        axis=1,
    )
    expected = np.einsum(
        "pf,nf->np",
        _UPSTREAM_MATRIX,
        local_values[:, [0, 3, 1, 2]],
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)


def test_common_scalar_gauge_is_removed_from_every_receiver_width():
    projector = _projector()
    shift_ev = 0.25
    potentials = np.full(
        (len(_RECEIVER_SIGMAS_ANGSTROM), 2),
        shift_ev,
    )
    gradients = np.zeros((len(_RECEIVER_SIGMAS_ANGSTROM), 2, 3))

    features = projector.project_smoothed_fields(
        potentials,
        gradients,
        scalar_potential_gauge_reference_ev=shift_ev,
    )

    np.testing.assert_allclose(features, 0.0, rtol=0.0, atol=1.0e-15)


def test_asc_projection_uses_one_atomic_center_mean_gauge_for_all_sigmas():
    projector = _projector()
    positions_angstrom = np.asarray(
        [[-0.3, 0.1, 0.2], [0.7, -0.2, -0.1]]
    )
    surface_centers_bohr = np.asarray(
        [[7.0, 2.0, -1.0], [-6.0, 4.0, 3.0], [2.0, -8.0, 5.0]]
    )
    asc_e = np.asarray([0.3, -0.2, 0.15])

    features, gauge_reference_ev = projector.project_asc_with_gauge(
        positions_angstrom,
        surface_centers_bohr,
        asc_e,
    )
    potentials, gradients = projector.smoothed_reaction_fields_hartree(
        positions_angstrom,
        surface_centers_bohr,
        asc_e,
    )

    assert gauge_reference_ev == pytest.approx(
        projector.atomic_center_mean_gauge_reference_hartree(
            positions_angstrom,
            surface_centers_bohr,
            asc_e,
        )
        * Hartree
    )
    np.testing.assert_allclose(
        features,
        projector.project_smoothed_fields(
            potentials * Hartree,
            gradients * Hartree / Bohr,
            scalar_potential_gauge_reference_ev=gauge_reference_ev,
        ),
    )


def test_analytic_asc_projection_matches_independent_3d_quadrature():
    projector = _projector()
    positions_angstrom = np.asarray([[0.2, -0.1, 0.3]])
    surface_centers_bohr = np.asarray(
        [
            [34.0, -4.0, 1.0],
            [-32.0, 8.0, -5.0],
            [6.0, 33.0, 3.0],
        ]
    )
    asc_e = np.asarray([0.35, -0.22, 0.11])

    potentials, gradients = projector.smoothed_reaction_fields_hartree(
        positions_angstrom,
        surface_centers_bohr,
        asc_e,
    )

    for sigma_index, sigma in enumerate(_RECEIVER_SIGMAS_ANGSTROM):
        expected_potential, expected_gradient = (
            _gauss_hermite_reaction_moments(
                positions_angstrom[0] / Bohr,
                sigma,
                surface_centers_bohr,
                asc_e,
            )
        )
        assert potentials[sigma_index, 0] == pytest.approx(
            expected_potential,
            rel=2.0e-7,
            abs=2.0e-9,
        )
        np.testing.assert_allclose(
            gradients[sigma_index, 0],
            expected_gradient,
            rtol=3.0e-6,
            atol=3.0e-8,
        )


def test_exact_gto_projection_is_rotation_covariant_before_model_injection():
    projector = _projector()
    positions_angstrom = np.asarray(
        [[0.2, -0.1, 0.3], [-0.4, 0.7, -0.2]]
    )
    surface_centers_bohr = np.asarray(
        [[6.0, 2.0, -1.0], [-5.0, 4.0, 3.0], [2.0, -7.0, 4.0]]
    )
    asc_e = np.asarray([0.3, -0.2, 0.15])
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    potentials, gradients = projector.smoothed_reaction_fields_hartree(
        positions_angstrom,
        surface_centers_bohr,
        asc_e,
    )
    rotated_features = projector.project_asc(
        positions_angstrom @ rotation.T,
        surface_centers_bohr @ rotation.T,
        asc_e,
    )
    gauge_reference_ev = (
        projector.atomic_center_mean_gauge_reference_hartree(
            positions_angstrom,
            surface_centers_bohr,
            asc_e,
        )
        * Hartree
    )
    expected_rotated_features = projector.project_smoothed_fields(
        potentials * Hartree,
        (gradients @ rotation.T) * Hartree / Bohr,
        scalar_potential_gauge_reference_ev=gauge_reference_ev,
    )

    np.testing.assert_allclose(
        rotated_features,
        expected_rotated_features,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_exact_gto_projection_adjoint_matches_the_full_discrete_transpose():
    projector = _projector()
    positions_angstrom = np.asarray(
        [[0.2, -0.1, 0.3], [-0.4, 0.7, -0.2]]
    )
    surface_centers_bohr = np.asarray(
        [[6.0, 2.0, -1.0], [-5.0, 4.0, 3.0], [2.0, -7.0, 4.0]]
    )
    rng = np.random.default_rng(20260727)
    feature_cotangent = rng.normal(
        size=(len(positions_angstrom), projector.feature_count)
    )
    surface_dimension = len(surface_centers_bohr)
    forward_matrix = np.column_stack(
        [
            projector.project_asc(
                positions_angstrom,
                surface_centers_bohr,
                basis,
            ).reshape(-1)
            for basis in np.eye(surface_dimension)
        ]
    )

    actual = projector.project_asc_adjoint(
        positions_angstrom,
        surface_centers_bohr,
        feature_cotangent,
    )

    np.testing.assert_allclose(
        actual,
        forward_matrix.T @ feature_cotangent.reshape(-1),
        rtol=2.0e-13,
        atol=2.0e-12,
    )


def test_projection_spec_rejects_a_matrix_that_cannot_match_the_receiver_basis():
    with pytest.raises(ValueError, match="matrix shape"):
        MACEPolarGTOFieldProjectionSpec(
            receiver_sigmas_angstrom=_RECEIVER_SIGMAS_ANGSTROM,
            receiver_max_l=1,
            receiver_normalization="receiver",
            upstream_matrix=np.eye(4),
        )


def test_projection_spec_fingerprints_the_ordered_live_matrix():
    spec = _projector().spec

    assert spec.provenance == {
        "contract_version": 1,
        "graph_longrange_version": "0.4.0",
        "feature_layout": (
            "graph-longrange-0.4.0-l0-radial-first-"
            "l1-radial-cartesian-v1"
        ),
        "receiver_sigmas_angstrom": [1.5, 3.0],
        "receiver_max_l": 1,
        "receiver_normalization": "receiver",
        "upstream_matrix_shape": [8, 4],
        "upstream_matrix_sha256": (
            "9ab5ed2a60579d9ff0c8b0da7de251077cb932e3b4ab7f9a6"
            "c7ec956432df108"
        ),
    }


def test_projection_spec_rejects_unvalidated_layout_or_runtime():
    kwargs = {
        "receiver_sigmas_angstrom": _RECEIVER_SIGMAS_ANGSTROM,
        "receiver_max_l": 1,
        "receiver_normalization": "receiver",
        "upstream_matrix": _UPSTREAM_MATRIX,
    }
    with pytest.raises(ValueError, match="graph-longrange 0.4.0"):
        MACEPolarGTOFieldProjectionSpec(
            **kwargs,
            graph_longrange_version="0.4.1",
        )
    with pytest.raises(ValueError, match="receiver-feature layout"):
        MACEPolarGTOFieldProjectionSpec(
            **kwargs,
            feature_layout="unknown",
        )
