from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.continuum.harmonic_cds_area import (
    SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
    SmoothHarmonicExposureArea,
)
from maple.solvation.continuum.harmonic_coefficients import _real_harmonic_design
from maple.solvation.continuum.harmonic_exposure import (
    build_smooth_harmonic_exposure,
)
from maple.solvation.continuum.harmonic_torch_functional import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
)

ROOT = Path(__file__).resolve().parents[2]
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.72, -0.31, 0.21], [-0.39, 1.65, -0.17]])
NUMBERS = (8, 1, 6)
RADII = (1.43, 1.18, 1.57)


def _area(
    *,
    numbers: tuple[int, ...] = NUMBERS,
    radii: tuple[float, ...] = RADII,
):
    torch = pytest.importorskip("torch")
    return SmoothHarmonicExposureArea(
        atomic_numbers=numbers,
        radii_angstrom=radii,
        transition_width_angstrom2=0.18,
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


def _sphere_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    cosine, polar_weights = np.polynomial.legendre.leggauss(order)
    azimuthal_count = 2 * order + 1
    azimuth = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    directions = np.stack(
        (
            np.repeat(sine, azimuthal_count) * np.tile(np.cos(azimuth), order),
            np.repeat(sine, azimuthal_count) * np.tile(np.sin(azimuth), order),
            np.repeat(cosine, azimuthal_count),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_count), azimuthal_count
    )
    return directions, weights


def _sharp_two_sphere_exposed_area(
    *, radius_i: float, radius_j: float, distance: float
) -> float:
    full = 4.0 * np.pi * radius_i * radius_i
    if distance + radius_i <= radius_j:
        return 0.0
    if distance >= radius_i + radius_j or distance + radius_j <= radius_i:
        return full
    cap_cosine = (radius_i * radius_i + distance * distance - radius_j * radius_j) / (
        2.0 * radius_i * distance
    )
    buried_cap = 2.0 * np.pi * radius_i * radius_i * (1.0 - cap_cosine)
    return full - buried_cap


def _independent_flat_step(values: np.ndarray) -> np.ndarray:
    """Test-only copy of the declared compact smooth-step mathematics."""

    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    result[values <= -1.0] = 0.0
    result[values >= 1.0] = 1.0
    interior = np.abs(values) < 1.0
    magnitude = np.abs(values[interior])
    positive = 1.0 / (
        1.0 + np.exp(-(1.0 / (1.0 - magnitude) - 1.0 / (1.0 + magnitude)))
    )
    result[interior] = np.where(values[interior] < 0.0, 1.0 - positive, positive)
    return result


def _direct_smooth_exposure_area(
    *,
    positions: np.ndarray,
    radii: tuple[float, ...],
    atom_index: int,
    transition_width_angstrom2: float,
    sphere_order: int,
) -> float:
    """Direct unprojected product integral used only as a test oracle."""

    directions, weights = _sphere_rule(sphere_order)
    radius = radii[atom_index]
    surface_points = positions[atom_index] + radius * directions
    exposure = np.ones(directions.shape[0], dtype=float)
    for other_index, (center, other_radius) in enumerate(zip(positions, radii)):
        if other_index == atom_index:
            continue
        signed_overlap = (
            np.sum((surface_points - center) ** 2, axis=1) - other_radius**2
        )
        exposure *= _independent_flat_step(signed_overlap / transition_width_angstrom2)
    return float(radius * radius * np.dot(weights, exposure))


def test_import_is_dependency_light_and_capabilities_remain_closed() -> None:
    script = r"""
import sys
sys.modules['torch'] = None
sys.modules['pyscf'] = None
from maple.solvation.continuum.harmonic_cds_area import (
    SmoothHarmonicExposureArea,
)
assert SmoothHarmonicExposureArea.capabilities.enabled_tiers == ()
assert SmoothHarmonicExposureArea.laboratory_fixed_surface_grid is False
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_isolated_sphere_has_exact_full_area_and_zero_gradient() -> None:
    radius = 1.37
    area = _area(numbers=(8,), radii=(radius,))
    positions = np.asarray([[0.3, -0.2, 0.7]])
    np.testing.assert_allclose(
        area.atom_areas_angstrom2(positions),
        [4.0 * np.pi * radius * radius],
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        area.position_vjp(positions, [2.3]),
        0.0,
        atol=0.0,
        rtol=0.0,
    )


def test_linear_exposure_area_converges_to_analytic_two_sphere_area() -> None:
    torch = pytest.importorskip("torch")
    radii = (1.40, 1.20)
    distance = 1.50
    positions = np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    expected = _sharp_two_sphere_exposed_area(
        radius_i=radii[0], radius_j=radii[1], distance=distance
    )
    errors = []
    for order in (48, 96, 192, 384):
        area = SmoothHarmonicExposureArea(
            atomic_numbers=(6, 8),
            radii_angstrom=radii,
            transition_width_angstrom2=0.50,
            exposure_lmax=2,
            radial_quadrature_order=order,
            dtype=torch.float64,
            device="cpu",
        )
        observed = float(area.atom_areas_angstrom2(positions)[0])
        errors.append(abs(observed - expected))
    assert errors[-1] < 2.0e-7
    assert errors[-1] < errors[0] / 10_000.0


def test_containment_and_near_tangency_are_bounded_and_continuous() -> None:
    torch = pytest.importorskip("torch")
    containment = SmoothHarmonicExposureArea(
        atomic_numbers=(1, 6),
        radii_angstrom=(0.5, 2.0),
        transition_width_angstrom2=0.18,
        exposure_lmax=4,
        radial_quadrature_order=96,
        dtype=torch.float64,
        device="cpu",
    )
    positions = np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        containment.atom_areas_angstrom2(positions),
        [0.0, 4.0 * np.pi * 2.0**2],
        atol=2.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        containment.position_vjp(positions, [1.0, -0.7]),
        0.0,
        atol=0.0,
        rtol=0.0,
    )

    tangent_areas = []
    for distance in np.linspace(2.72, 2.90, 10):
        candidate = SmoothHarmonicExposureArea(
            atomic_numbers=(6, 6),
            radii_angstrom=(1.4, 1.4),
            transition_width_angstrom2=0.18,
            exposure_lmax=4,
            radial_quadrature_order=96,
            dtype=torch.float64,
            device="cpu",
        )
        tangent_areas.append(
            candidate.atom_areas_angstrom2(
                np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
            )[0]
        )
    assert np.all(np.diff(tangent_areas) >= -2.0e-12)
    assert tangent_areas[-1] <= 4.0 * np.pi * 1.4**2 + 2.0e-12


@pytest.mark.parametrize("transition_width", (0.08, 0.18, 0.35))
def test_triple_overlap_product_projection_converges_to_direct_smooth_area(
    transition_width: float,
) -> None:
    torch = pytest.importorskip("torch")
    positions = np.asarray([[0.0, 0.0, 0.0], [1.35, 0.10, 0.0], [0.25, 1.38, 0.15]])
    radii = (1.4, 1.2, 1.3)
    reference = _direct_smooth_exposure_area(
        positions=positions,
        radii=radii,
        atom_index=0,
        transition_width_angstrom2=transition_width,
        sphere_order=384,
    )
    errors = []
    for exposure_lmax in (2, 5, 12):
        candidate = SmoothHarmonicExposureArea(
            atomic_numbers=(6, 8, 7),
            radii_angstrom=radii,
            transition_width_angstrom2=transition_width,
            exposure_lmax=exposure_lmax,
            radial_quadrature_order=256,
            dtype=torch.float64,
            device="cpu",
        )
        observed = candidate.atom_areas_angstrom2(positions)[0]
        errors.append(abs(observed - reference))
    assert errors[-1] < 4.0e-3
    assert errors[-1] < 0.35 * errors[0]


def test_triple_overlap_radial_projection_converges() -> None:
    torch = pytest.importorskip("torch")
    positions = np.asarray([[0.0, 0.0, 0.0], [1.35, 0.10, 0.0], [0.25, 1.38, 0.15]])
    radii = (1.4, 1.2, 1.3)
    values = []
    for radial_order in (96, 192, 384, 512):
        candidate = SmoothHarmonicExposureArea(
            atomic_numbers=(6, 8, 7),
            radii_angstrom=radii,
            transition_width_angstrom2=0.18,
            exposure_lmax=8,
            radial_quadrature_order=radial_order,
            dtype=torch.float64,
            device="cpu",
        )
        values.append(candidate.atom_areas_angstrom2(positions)[0])
    terminal = values[-1]
    errors = np.abs(np.asarray(values[:-1]) - terminal)
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 4.0e-5


def test_coefficients_match_numpy_cavity_and_linear_exposure_area() -> None:
    area = _area()
    coefficients = area.exposure_coefficients(POSITIONS)
    reference = build_smooth_harmonic_exposure(
        atomic_numbers=NUMBERS,
        positions_angstrom=POSITIONS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=1,
        exposure_lmax=4,
        radial_quadrature_order=48,
    )
    np.testing.assert_allclose(
        coefficients,
        reference.exposure_coefficients,
        atol=3.0e-14,
        rtol=0.0,
    )

    expected = np.asarray(RADII) ** 2 * np.sqrt(4.0 * np.pi) * coefficients[:, 0]
    observed = area.atom_areas_angstrom2(POSITIONS)
    np.testing.assert_allclose(observed, expected, atol=5.0e-14, rtol=0.0)
    assert np.all(observed >= 0.0)

    directions, weights = _sphere_rule(16)
    design = _real_harmonic_design(directions, lmax=4)
    quadrature = np.asarray(RADII) ** 2 * np.asarray(
        [
            np.dot(weights, design @ atom_coefficients)
            for atom_coefficients in coefficients
        ]
    )
    np.testing.assert_allclose(observed, quadrature, atol=7.0e-14, rtol=0.0)

    quadratic_participation = np.asarray(RADII) ** 2 * np.sum(
        coefficients * coefficients, axis=1
    )
    assert np.any(observed > quadratic_participation + 1.0e-6)
    assert np.all(quadratic_participation <= observed + 1.0e-12)


def test_area_bounds_fail_closed_without_clipping(monkeypatch) -> None:
    area = _area()

    def invalid_areas(self, positions):
        del self
        full = 4.0 * np.pi * np.asarray(RADII) ** 2
        return positions.new_tensor([-1.0, full[1], full[2] + 1.0])

    monkeypatch.setattr(SmoothHarmonicExposureArea, "_areas_torch", invalid_areas)
    with pytest.raises(RuntimeError, match="positive-parent projection bounds"):
        area.atom_areas_angstrom2(POSITIONS)
    with pytest.raises(RuntimeError, match="positive-parent projection bounds"):
        area.position_vjp(POSITIONS, np.ones(3))


def test_area_position_vjp_rejects_a_disconnected_gradient_graph(monkeypatch) -> None:
    area = _area()

    def disconnected_areas(self, positions):
        del self
        full = 4.0 * np.pi * np.asarray(RADII) ** 2
        return positions.new_tensor(full, requires_grad=True)

    monkeypatch.setattr(SmoothHarmonicExposureArea, "_areas_torch", disconnected_areas)
    with pytest.raises(RuntimeError, match="gradient graph is disconnected"):
        area.position_vjp(POSITIONS, np.ones(3))


def test_area_position_vjp_matches_multistep_central_difference() -> None:
    area = _area()
    rng = np.random.default_rng(20260816)
    cotangent = rng.normal(size=3)
    direction = rng.normal(size=POSITIONS.shape)
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    analytic = float(np.vdot(area.position_vjp(POSITIONS, cotangent), direction))
    richardson_errors = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        coarse = (
            np.vdot(
                cotangent,
                area.atom_areas_angstrom2(POSITIONS + step * direction),
            )
            - np.vdot(
                cotangent,
                area.atom_areas_angstrom2(POSITIONS - step * direction),
            )
        ) / (2.0 * step)
        half_step = step / 2.0
        fine = (
            np.vdot(
                cotangent,
                area.atom_areas_angstrom2(POSITIONS + half_step * direction),
            )
            - np.vdot(
                cotangent,
                area.atom_areas_angstrom2(POSITIONS - half_step * direction),
            )
        ) / (2.0 * half_step)
        richardson = (4.0 * fine - coarse) / 3.0
        richardson_errors.append(abs(analytic - richardson))
    assert richardson_errors[-1] < 2.0e-9
    assert richardson_errors[-1] < richardson_errors[0] / 100.0
    np.testing.assert_allclose(
        np.sum(area.position_vjp(POSITIONS, cotangent), axis=0),
        0.0,
        atol=2.0e-13,
        rtol=0.0,
    )


def test_area_is_rotation_translation_and_label_permutation_invariant() -> None:
    area = _area()
    reference = area.atom_areas_angstrom2(POSITIONS)
    cotangent = np.asarray([0.7, -0.2, 1.1])
    gradient = area.position_vjp(POSITIONS, cotangent)
    rotation = _rotation(311)
    shift = np.asarray([0.71, -0.48, 0.29])
    transformed = POSITIONS @ rotation.T + shift
    np.testing.assert_allclose(
        area.atom_areas_angstrom2(transformed),
        reference,
        atol=8.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        area.position_vjp(transformed, cotangent),
        gradient @ rotation.T,
        atol=2.0e-12,
        rtol=0.0,
    )

    permutation = np.asarray([2, 0, 1])
    permuted = _area(
        numbers=tuple(NUMBERS[index] for index in permutation),
        radii=tuple(RADII[index] for index in permutation),
    )
    np.testing.assert_allclose(
        permuted.atom_areas_angstrom2(POSITIONS[permutation]),
        reference[permutation],
        atol=8.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        permuted.position_vjp(POSITIONS[permutation], cotangent[permutation]),
        gradient[permutation],
        atol=2.0e-12,
        rtol=0.0,
    )


def test_identity_is_content_bound_and_invalid_inputs_fail_closed() -> None:
    area = _area()
    atoms = Atoms(numbers=NUMBERS, positions=POSITIONS)
    identity = area.state_identity(atoms)
    assert identity["contract_id"] == (SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID)
    assert identity["area_measure"] == "a_i^2-integral-e_i-domega"
    assert identity["coordinate_derivative_available"] is True
    assert identity["laboratory_fixed_surface_grid"] is False
    assert len(identity["configuration_sha256"]) == 64
    assert len(identity["geometry_sha256"]) == 64

    changed = SmoothHarmonicExposureArea(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.19,
        exposure_lmax=4,
        radial_quadrature_order=48,
        dtype=pytest.importorskip("torch").float64,
        device="cpu",
    )
    assert changed.configuration_sha256() != area.configuration_sha256()

    with pytest.raises(ValueError, match="atomic numbers"):
        area.atom_areas_angstrom2(Atoms("CHH", positions=POSITIONS))
    with pytest.raises(ValueError, match="area_cotangent"):
        area.position_vjp(POSITIONS, [1.0, 2.0])
    with pytest.raises(ValueError, match="distinct"):
        area.atom_areas_angstrom2(np.zeros_like(POSITIONS))
    with pytest.raises(AttributeError, match="immutable"):
        area._transition_width_angstrom2 = 0.3


def test_area_validates_exact_cavity_equality_with_harmonic_continuum() -> None:
    torch = pytest.importorskip("torch")
    area = _area()
    continuum = SmoothWeightedHarmonicGalerkinFunctionalCandidate(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        surface_lmax=2,
        exposure_lmax=4,
        exposure_radial_quadrature_order=48,
        source_radial_quadrature_order=32,
        green_radial_quadrature_order=32,
        dtype=torch.float64,
        device="cpu",
        scalar_id=(
            EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
        ),
    )
    assert area.validate_same_cavity_as(continuum) == continuum.configuration_sha256()

    changed_area = SmoothHarmonicExposureArea(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.19,
        exposure_lmax=4,
        radial_quadrature_order=48,
        dtype=torch.float64,
        device="cpu",
    )
    with pytest.raises(ValueError, match="cavity descriptors differ"):
        changed_area.validate_same_cavity_as(continuum)

    float32_area = SmoothHarmonicExposureArea(
        atomic_numbers=NUMBERS,
        radii_angstrom=RADII,
        transition_width_angstrom2=0.18,
        exposure_lmax=4,
        radial_quadrature_order=48,
        dtype=torch.float32,
        device="cpu",
    )
    with pytest.raises(ValueError, match="cavity descriptors differ"):
        float32_area.validate_same_cavity_as(continuum)

    object.__setattr__(area, "_transition_width_angstrom2", 0.19)
    with pytest.raises(RuntimeError, match="configuration drifted"):
        area.validate_same_cavity_as(continuum)
