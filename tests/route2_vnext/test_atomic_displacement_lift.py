from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.atomic_displacement_lift import (
    CanonicalAtomicDisplacementLift,
    gaussian_enclosed_electrons_over_r3,
    gaussian_translation_dipole_self_work,
)


def _mixtures() -> dict[int, GaussianMixtureAtom]:
    return {
        1: GaussianMixtureAtom(
            electron_counts=np.asarray([0.3, 0.7]),
            gaussian_exponents_bohr2=np.asarray([0.4, 1.6]),
        ),
        2: GaussianMixtureAtom(
            electron_counts=np.asarray([0.8, 1.2]),
            gaussian_exponents_bohr2=np.asarray([0.25, 1.1]),
        ),
    }


def _lift(numbers=(1, 2, 1)) -> CanonicalAtomicDisplacementLift:
    return CanonicalAtomicDisplacementLift.from_mixtures(
        atomic_numbers=numbers,
        mixtures_by_atomic_number=_mixtures(),
        source_asset_sha256="a" * 64,
    )


def _rotation() -> np.ndarray:
    axis = np.asarray([0.3, -0.7, 0.5], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        math.cos(angle) * np.eye(3)
        + (1.0 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * skew
    )


def test_gaussian_self_work_matches_independent_fourier_quadrature() -> None:
    mixture = _mixtures()[1]
    counts = np.asarray(mixture.electron_counts)
    exponents = np.asarray(mixture.gaussian_exponents_bohr2)

    def density_fourier(wave_number: float) -> float:
        return float(
            np.sum(counts * np.exp(-(wave_number**2) / (4.0 * exponents)))
        )

    energy_coefficient, _error = quad(
        lambda wave_number: wave_number**2 * density_fourier(wave_number) ** 2,
        0.0,
        np.inf,
        epsabs=2.0e-13,
        epsrel=2.0e-13,
    )
    # E = p^2/(3*pi*Z^2) * integral; E = 0.5*kappa*p^2.
    independent = 2.0 * energy_coefficient / (
        3.0 * math.pi * mixture.electron_count**2
    )
    assert gaussian_translation_dipole_self_work(mixture) == pytest.approx(
        independent, rel=3.0e-14, abs=0.0
    )


def test_enclosed_electron_factor_has_correct_origin_and_far_field_limits() -> None:
    mixture = _mixtures()[1]
    radii = np.asarray([0.0, 1.0e-9, 1.0e-5, 0.4, 30.0])
    factor = gaussian_enclosed_electrons_over_r3(mixture, radii)
    expected_origin = sum(
        float(count)
        * (4.0 / (3.0 * math.sqrt(math.pi)))
        * float(exponent) ** 1.5
        for count, exponent in zip(
            mixture.electron_counts,
            mixture.gaussian_exponents_bohr2,
            strict=True,
        )
    )
    assert factor[0] == pytest.approx(expected_origin, rel=2.0e-15, abs=0.0)
    assert factor[1] == pytest.approx(expected_origin, rel=2.0e-15, abs=0.0)
    assert factor[-1] == pytest.approx(
        mixture.electron_count / radii[-1] ** 3, rel=2.0e-15, abs=0.0
    )
    assert np.all(factor >= 0.0)


def test_minimum_self_work_lift_is_right_inverse_and_unique_kkt_minimum() -> None:
    lift = _lift()
    dipole = np.asarray([0.4, -0.2, 0.7])
    atomic = lift.lift_molecular_dipole(dipole)
    np.testing.assert_allclose(np.sum(atomic, axis=0), dipole, atol=2.0e-14)
    np.testing.assert_allclose(
        lift.molecular_dipole_map @ lift.minimum_metric_right_inverse,
        np.eye(3),
        atol=2.0e-14,
    )
    rng = np.random.default_rng(8)
    perturbation = rng.normal(size=atomic.shape)
    perturbation -= np.mean(perturbation, axis=0, keepdims=True)
    hessian = lift.metric_matrix
    optimum = atomic.reshape(-1)
    alternative = (atomic + perturbation).reshape(-1)
    optimum_work = 0.5 * float(optimum @ hessian @ optimum)
    alternative_work = 0.5 * float(alternative @ hessian @ alternative)
    assert alternative_work > optimum_work
    kkt_gradient = hessian @ optimum
    per_atom = kkt_gradient.reshape(lift.atom_count, 3)
    np.testing.assert_allclose(
        per_atom,
        np.broadcast_to(per_atom[0], per_atom.shape),
        atol=3.0e-14,
    )


def test_tangent_is_first_order_limit_of_exact_translated_electron_density() -> None:
    lift = _lift(numbers=(1, 2))
    centers = np.asarray([[0.0, 0.0, 0.0], [1.1, -0.2, 0.3]])
    points = np.asarray([[2.3, 0.4, -0.6], [-0.7, 1.9, 0.8], [0.2, -1.4, 1.7]])
    direction = np.asarray([0.7, -0.3, 0.2])
    errors = []
    for scale in (2.0e-3, 1.0e-3, 5.0e-4):
        atomic = lift.lift_molecular_dipole(scale * direction)
        tangent = lift.tangent_potential(
            points_bohr=points,
            centers_bohr=centers,
            atomic_dipoles_ebohr=atomic,
        )
        exact = lift.exact_translated_potential_difference(
            points_bohr=points,
            centers_bohr=centers,
            atomic_dipoles_ebohr=atomic,
        )
        errors.append(float(np.linalg.norm(exact - tangent)))
    assert errors[1] < 0.27 * errors[0]
    assert errors[2] < 0.27 * errors[1]


def test_surface_operator_replays_source_and_has_exact_far_field_dipole() -> None:
    lift = _lift(numbers=(1, 2))
    centers = np.asarray([[0.0, 0.0, 0.0], [1.1, -0.2, 0.3]])
    points = np.asarray([[1.7, 0.4, -0.6], [-0.7, 1.9, 0.8]])
    molecular = np.asarray([0.3, -0.4, 0.2])
    atomic = lift.lift_molecular_dipole(molecular)
    operator = lift.atomic_surface_operator(points_bohr=points, centers_bohr=centers)
    direct = lift.tangent_potential(
        points_bohr=points,
        centers_bohr=centers,
        atomic_dipoles_ebohr=atomic,
    )
    np.testing.assert_allclose(operator @ atomic.reshape(-1), direct, atol=2.0e-15)

    far = np.asarray([[4.0e5, -2.0e5, 3.0e5]])
    potential = lift.tangent_potential(
        points_bohr=far,
        centers_bohr=centers,
        atomic_dipoles_ebohr=atomic,
    )[0]
    displacement = far[0]
    expected = float(molecular @ displacement) / np.linalg.norm(displacement) ** 3
    assert potential == pytest.approx(expected, rel=6.0e-6, abs=0.0)


def test_rotation_translation_and_permutation_covariance() -> None:
    numbers = np.asarray([1, 2, 1])
    lift = _lift(numbers=tuple(numbers))
    centers = np.asarray(
        [[0.1, -0.2, 0.3], [1.2, 0.4, -0.5], [-0.8, 1.1, 0.6]]
    )
    points = np.asarray([[2.0, 0.3, -0.7], [-1.3, 1.8, 0.4]])
    molecular = np.asarray([0.3, -0.5, 0.2])
    atomic = lift.lift_molecular_dipole(molecular)
    value = lift.tangent_potential(
        points_bohr=points,
        centers_bohr=centers,
        atomic_dipoles_ebohr=atomic,
    )

    rotation = _rotation()
    shift = np.asarray([0.7, -1.1, 0.4])
    rotated = lift.tangent_potential(
        points_bohr=points @ rotation.T + shift,
        centers_bohr=centers @ rotation.T + shift,
        atomic_dipoles_ebohr=atomic @ rotation.T,
    )
    np.testing.assert_allclose(rotated, value, atol=4.0e-16, rtol=2.0e-14)

    permutation = np.asarray([2, 0, 1])
    permuted_lift = _lift(numbers=tuple(numbers[permutation]))
    permuted_atomic = permuted_lift.lift_molecular_dipole(molecular)
    np.testing.assert_allclose(permuted_atomic, atomic[permutation], atol=2.0e-15)
    permuted = permuted_lift.tangent_potential(
        points_bohr=points,
        centers_bohr=centers[permutation],
        atomic_dipoles_ebohr=permuted_atomic,
    )
    np.testing.assert_allclose(permuted, value, atol=4.0e-16, rtol=2.0e-14)


def test_constructor_detaches_inputs_and_rejects_nonphysical_assets() -> None:
    numbers = np.asarray([1, 2])
    mixtures = _mixtures()
    lift = CanonicalAtomicDisplacementLift.from_mixtures(
        atomic_numbers=numbers,
        mixtures_by_atomic_number=mixtures,
        source_asset_sha256="b" * 64,
    )
    digest = lift.configuration_sha256
    numbers[:] = 1
    mixtures[1].electron_counts.setflags(write=True)
    mixtures[1].electron_counts[:] = 9.0
    assert lift.configuration_sha256 == digest
    assert tuple(lift.atomic_numbers) == (1, 2)

    with pytest.raises(ValueError, match="normalized"):
        CanonicalAtomicDisplacementLift.from_mixtures(
            atomic_numbers=[2],
            mixtures_by_atomic_number={
                2: GaussianMixtureAtom(np.asarray([1.0]), np.asarray([0.8]))
            },
            source_asset_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="SHA256"):
        CanonicalAtomicDisplacementLift.from_mixtures(
            atomic_numbers=[1],
            mixtures_by_atomic_number=_mixtures(),
            source_asset_sha256="not-a-digest",
        )
