from __future__ import annotations

import math

import numpy as np
import pytest

from maple.solvation.api import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1,
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1,
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
)
from maple.solvation.continuum import (
    COULOMB_EV_ANGSTROM_PER_E2,
    SmoothPointChargeHarmonicDDPCMFunctionalCandidate,
    SmoothPointChargeHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_torch_primitives import (
    _assemble_double_layer,
    _assemble_single_layer,
)

POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.72, -0.31, 0.21], [-0.39, 1.65, -0.17]])
NUMBERS = (8, 1, 6)
RADII = (1.43, 1.18, 1.57)
SOURCE = np.asarray(
    [[0.20, 0.0, 0.0, 0.0], [-0.14, 0.0, 0.0, 0.0], [-0.06, 0.0, 0.0, 0.0]]
)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _functional(*, dielectric: float = 78.39):
    torch = pytest.importorskip("torch")
    return SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        dielectric=dielectric,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
    )


def test_double_layer_single_sphere_has_exact_principal_value_spectrum():
    torch = pytest.importorskip("torch")
    positions = torch.zeros((1, 3), dtype=torch.float64)
    actual = np.asarray(
        _assemble_double_layer(
            positions,
            radii=(1.37,),
            lmax=4,
            radial_order=24,
        ).detach(),
        dtype=float,
    )
    expected = np.diag(
        [-2.0 * np.pi / (2 * ell + 1) for ell in range(5) for _ in range(2 * ell + 1)]
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-15, rtol=0.0)


@pytest.mark.parametrize(
    ("displacement", "radii"),
    (
        ((2.83, -0.41, 0.29), (1.23, 1.09)),  # separated
        ((1.50, 0.20, -0.10), (1.23, 1.09)),  # intersecting
        ((0.80, 0.20, -0.10), (0.50, 2.00)),  # target nested in source
        ((0.80, 0.20, -0.10), (2.00, 0.50)),  # source nested in target
    ),
)
def test_double_layer_cross_block_is_source_normal_derivative_of_single_layer(
    displacement, radii
):
    torch = pytest.importorskip("torch")
    positions = torch.tensor([[0.0, 0.0, 0.0], displacement], dtype=torch.float64)
    lmax = 2
    dimension = (lmax + 1) ** 2
    actual = np.asarray(
        _assemble_double_layer(
            positions,
            radii=radii,
            lmax=lmax,
            radial_order=64,
        )[:dimension, dimension:].detach(),
        dtype=float,
    )
    step = 2.0e-6
    plus = _assemble_single_layer(
        positions,
        radii=(radii[0], radii[1] + step),
        lmax=lmax,
        radial_order=64,
    )[:dimension, dimension:]
    minus = _assemble_single_layer(
        positions,
        radii=(radii[0], radii[1] - step),
        lmax=lmax,
        radial_order=64,
    )[:dimension, dimension:]
    expected = (
        radii[1] ** 2
        * np.asarray(((plus - minus) / (2.0 * step)).detach(), dtype=float)
        / COULOMB_EV_ANGSTROM_PER_E2
    )
    np.testing.assert_allclose(actual, expected, atol=2.0e-9, rtol=3.0e-10)


def test_single_sphere_ddpcm_screening_is_mode_dependent_and_has_correct_limits():
    torch = pytest.importorskip("torch")
    dielectric = 7.0
    functional = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8,),
        radii_angstrom=(1.4,),
        dielectric=dielectric,
        transition_width_angstrom2=0.18,
        surface_lmax=3,
        exposure_lmax=6,
        exposure_radial_quadrature_order=20,
        green_radial_quadrature_order=20,
        dtype=torch.float64,
        device="cpu",
    )
    matrices = functional.debug_ddpcm_matrices(np.zeros((1, 3)))
    g = (dielectric + 1.0) / (dielectric - 1.0)
    np.testing.assert_allclose(
        matrices["dielectric_operator"] - matrices["conductor_limit_operator"],
        4.0 * np.pi / (dielectric - 1.0) * matrices["mass"],
        atol=5.0e-15,
        rtol=0.0,
    )
    expected_factors = np.asarray(
        [
            (1.0 + 1.0 / (2 * ell + 1)) / (g + 1.0 / (2 * ell + 1))
            for ell in range(4)
            for _ in range(2 * ell + 1)
        ]
    )
    response = np.linalg.solve(
        matrices["dielectric_operator"], matrices["conductor_limit_operator"]
    )
    np.testing.assert_allclose(response, np.diag(expected_factors), atol=3e-14)
    assert expected_factors[0] != pytest.approx(expected_factors[-1])

    conductor = SmoothPointChargeHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=(8,),
        radii_angstrom=(1.4,),
        transition_width_angstrom2=0.18,
        surface_lmax=3,
        exposure_lmax=6,
        exposure_radial_quadrature_order=20,
        green_radial_quadrature_order=20,
        dtype=torch.float64,
        device="cpu",
    )
    source = np.asarray([[0.7, 0.0, 0.0, 0.0]])
    infinite = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8,),
        radii_angstrom=(1.4,),
        dielectric=1.0e14,
        transition_width_angstrom2=0.18,
        surface_lmax=3,
        exposure_lmax=6,
        exposure_radial_quadrature_order=20,
        green_radial_quadrature_order=20,
        dtype=torch.float64,
        device="cpu",
    )
    assert infinite.energy_eV(np.zeros((1, 3)), source) == pytest.approx(
        conductor.energy_eV(np.zeros((1, 3)), source), rel=3.0e-14
    )
    near_vacuum = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(8,),
        radii_angstrom=(1.4,),
        dielectric=1.0 + 1.0e-8,
        transition_width_angstrom2=0.18,
        surface_lmax=3,
        exposure_lmax=6,
        exposure_radial_quadrature_order=20,
        green_radial_quadrature_order=20,
        dtype=torch.float64,
        device="cpu",
    )
    assert abs(near_vacuum.energy_eV(np.zeros((1, 3)), source)) < 1.0e-7


def test_ddpcm_stationarity_reciprocity_derivatives_and_so3_close():
    functional = _functional()
    assert functional.energy_eV(POSITIONS, 2.0 * SOURCE) == pytest.approx(
        4.0 * functional.energy_eV(POSITIONS, SOURCE), abs=3.0e-13
    )
    np.testing.assert_allclose(
        functional.drive(POSITIONS, 2.0 * SOURCE),
        2.0 * functional.drive(POSITIONS, SOURCE),
        atol=2.0e-12,
        rtol=0.0,
    )
    audit = functional.stationarity_audit(POSITIONS, SOURCE)
    assert audit["gate_passed"] is True
    assert audit["capability_admitted"] is False
    assert audit["finite_dielectric_parameterization"] is True
    for residual in audit["residuals"].values():
        assert residual["relative"] <= 1.0e-10
        assert residual["scaled"] <= 1.0e-10
        assert residual["right_hand_side_norm"] >= 0.0
    assert audit["residuals"]["single_layer_adjoint"]["unit"] == "eV/e"
    assert audit["residuals"]["dielectric_adjoint"]["unit"] == "e"
    assert audit["residuals"]["vacuum_projection_adjoint"]["unit"] == "e"
    response = audit["response_operator_audit"]
    assert response["pairing_metric_id"] == functional.pairing.scalar_id
    assert response["primal_response_used_as_provider_field"] is False
    assert response["primal_response_is_energy_cotangent"] is False
    assert response["primal_response_relative_asymmetry"] > 1.0e-5
    assert response["primal_charge_tangent_relative_asymmetry"] > 1.0e-5
    assert response["energy_cotangent_relative_asymmetry"] < 1.0e-12
    assert response["energy_cotangent_charge_tangent_relative_asymmetry"] < 1.0e-12
    assert response["energy_cotangent_vs_symmetric_primal_relative_error"] < 1.0e-12
    assert response["kkt_vs_autograd_relative_error"] < 1.0e-12
    assert response["primal_vs_energy_cotangent_relative_error"] > 1.0e-5
    assert response["half_coupling_absolute_error_eV"] < 1.0e-12
    assert response["gate_passed"] is True

    rng = np.random.default_rng(91)
    u = np.zeros_like(SOURCE)
    w = np.zeros_like(SOURCE)
    u[:, 0] = rng.normal(size=len(SOURCE))
    w[:, 0] = rng.normal(size=len(SOURCE))
    u[:, 0] -= np.mean(u[:, 0])
    w[:, 0] -= np.mean(w[:, 0])
    ju = functional.source_jvp(POSITIONS, SOURCE, u)
    jw = functional.source_jvp(POSITIONS, SOURCE, w)
    assert np.vdot(u, jw) == pytest.approx(np.vdot(w, ju), abs=2.0e-12)

    source_step = 2.0e-6
    numerical = (
        functional.energy_eV(POSITIONS, SOURCE + source_step * u)
        - functional.energy_eV(POSITIONS, SOURCE - source_step * u)
    ) / (2.0 * source_step)
    assert np.vdot(functional.drive(POSITIONS, SOURCE), u) == pytest.approx(
        numerical, abs=3.0e-10
    )

    direction = rng.normal(size=POSITIONS.shape)
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    analytic = np.vdot(functional.coordinate_partial(POSITIONS, SOURCE), direction)
    errors = []
    for step in (1.0e-3, 1.0e-4, 1.0e-5):
        finite = (
            functional.energy_eV(POSITIONS + step * direction, SOURCE)
            - functional.energy_eV(POSITIONS - step * direction, SOURCE)
        ) / (2.0 * step)
        errors.append(abs(analytic - finite))
    assert errors[-1] < 3.0e-9
    assert errors[-1] < errors[0] / 20.0

    rotation = _rotation(20260815)
    rotated_positions = POSITIONS @ rotation.T
    energy = functional.energy_eV(POSITIONS, SOURCE)
    gradient = functional.coordinate_partial(POSITIONS, SOURCE)
    assert functional.energy_eV(rotated_positions, SOURCE) == pytest.approx(
        energy, abs=3.0e-12
    )
    np.testing.assert_allclose(
        functional.coordinate_partial(rotated_positions, SOURCE),
        gradient @ rotation.T,
        atol=2.0e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0.0, atol=3.0e-12)
    centered = POSITIONS - np.mean(POSITIONS, axis=0)
    np.testing.assert_allclose(
        np.sum(np.cross(centered, -gradient), axis=0), 0.0, atol=2.0e-11
    )


def test_ddpcm_joint_hvp_is_symmetric_and_generated_from_the_scalar():
    functional = _functional()
    rng = np.random.default_rng(903)
    h_r = rng.normal(size=POSITIONS.shape)
    h_r -= np.mean(h_r, axis=0)
    h_c = np.zeros_like(SOURCE)
    h_c[:, 0] = rng.normal(size=len(SOURCE))
    k_r = rng.normal(size=POSITIONS.shape)
    k_r -= np.mean(k_r, axis=0)
    k_c = np.zeros_like(SOURCE)
    k_c[:, 0] = rng.normal(size=len(SOURCE))
    hh_r, hh_c = functional.joint_position_source_hvp(POSITIONS, SOURCE, h_r, h_c)
    hk_r, hk_c = functional.joint_position_source_hvp(POSITIONS, SOURCE, k_r, k_c)
    left = float(np.vdot(k_r, hh_r) + np.vdot(k_c, hh_c))
    right = float(np.vdot(h_r, hk_r) + np.vdot(h_c, hk_c))
    assert left == pytest.approx(right, abs=2.0e-9)

    step = 2.0e-5
    plus_positions = POSITIONS + step * h_r
    minus_positions = POSITIONS - step * h_r
    plus_source = SOURCE + step * h_c
    minus_source = SOURCE - step * h_c
    finite_r = (
        functional.coordinate_partial(plus_positions, plus_source)
        - functional.coordinate_partial(minus_positions, minus_source)
    ) / (2.0 * step)
    finite_c = (
        functional.pairing.field_to_source_dual(
            functional.drive(plus_positions, plus_source)
        )
        - functional.pairing.field_to_source_dual(
            functional.drive(minus_positions, minus_source)
        )
    ) / (2.0 * step)
    np.testing.assert_allclose(hh_r, finite_r, atol=4.0e-7, rtol=2.0e-6)
    np.testing.assert_allclose(hh_c, finite_c, atol=4.0e-8, rtol=2.0e-6)


def test_ddpcm_double_layer_radial_quadrature_has_a_refinement_plateau():
    torch = pytest.importorskip("torch")
    records = []
    for order in (16, 24, 32):
        functional = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
            atomic_numbers=NUMBERS,
            radii_angstrom=RADII,
            dielectric=78.39,
            transition_width_angstrom2=0.18,
            surface_lmax=1,
            exposure_lmax=2,
            exposure_radial_quadrature_order=64,
            green_radial_quadrature_order=order,
            dtype=torch.float64,
            device="cpu",
        )
        records.append(
            (
                functional.energy_eV(POSITIONS, SOURCE),
                functional.drive(POSITIONS, SOURCE),
                functional.coordinate_partial(POSITIONS, SOURCE),
            )
        )
    assert abs(records[1][0] - records[2][0]) < 1.0e-12
    assert abs(records[0][0] - records[2][0]) < 5.0e-10
    np.testing.assert_allclose(records[1][1], records[2][1], atol=2.0e-11, rtol=0.0)
    np.testing.assert_allclose(records[1][2], records[2][2], atol=2.0e-11, rtol=0.0)


@pytest.mark.parametrize("distance", (0.2, 1.8))
def test_ddpcm_scalar_and_derivatives_fail_closed_on_sphere_tangency(distance):
    torch = pytest.importorskip("torch")
    positions = np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    with pytest.raises(ValueError, match="tangency event surface"):
        _assemble_double_layer(
            torch.as_tensor(positions, dtype=torch.float64),
            radii=(1.0, 0.8),
            lmax=1,
            radial_order=24,
        )

    functional = SmoothPointChargeHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(1, 1),
        radii_angstrom=(1.0, 0.8),
        dielectric=78.39,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=2,
        exposure_radial_quadrature_order=24,
        green_radial_quadrature_order=24,
        dtype=torch.float64,
        device="cpu",
    )
    source = np.asarray([[0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]])
    operations = (
        lambda: functional.energy_eV(positions, source),
        lambda: functional.drive(positions, source),
        lambda: functional.coordinate_partial(positions, source),
        lambda: functional.stationarity_audit(positions, source),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="tangency event surface"):
            operation()


@pytest.mark.parametrize("dielectric", (1.0, 0.0, -2.0, math.inf, math.nan, True))
def test_invalid_dielectric_fails_closed(dielectric):
    with pytest.raises((TypeError, ValueError), match="dielectric"):
        _functional(dielectric=dielectric)


def test_ddpcm_has_distinct_immutable_registry_identity_and_no_admission():
    functional = _functional()
    other_dielectric = _functional(dielectric=12.0)
    profile = PROFILE_REGISTRY[
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_PROFILE_V1
    ]
    scalar = SCALAR_REGISTRY[
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_SMOOTH_HARMONIC_DDPCM_ELECTROSTATIC_V1
    ]
    assert functional.scalar_id == scalar.scalar_id == profile.scalar_id
    assert functional.continuum_profile_id == profile.continuum_profile
    assert functional.configuration_contract_id == (
        profile.continuum_configuration_contract_id
    )
    assert functional.configuration_contract_id.endswith(
        "ddpcm-parameterized-diagnostic.v1"
    )
    assert functional.dielectric == pytest.approx(78.39)
    assert functional.configuration_sha256() != other_dielectric.configuration_sha256()
    provenance = dict(functional.runtime_provenance())
    assert provenance["dielectric"] == pytest.approx(78.39)
    assert provenance["finite_dielectric_parameterization"] is True
    assert provenance["uniform_cosmo_dielectric_energy_scaling"] is False
    assert provenance["provider_field_semantics"].startswith(
        "energy-conjugate source derivative"
    )
    assert provenance["admission_identity"].startswith("parameterized-diagnostic-only")
    assert provenance["double_layer_radial_quadrature_order"] == 32
    assert "Gauss-Legendre double layer" in provenance["geometry_quadrature"]
    assert "finite_dielectric_ddpcm_double_layer_response" in scalar.included_components
    assert "uniform_cosmo_dielectric_energy_scaling" in scalar.excluded_components
    assert scalar.enabled is profile.enabled is False
    assert scalar.admitted_capabilities.enabled_tiers == ()
    assert profile.capabilities.enabled_tiers == ()
    assert functional.fixed_geometry_electronic_mutual_polarization is False
    with pytest.raises(AttributeError, match="immutable"):
        functional._dielectric = 80.0
