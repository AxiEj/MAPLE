from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.neutral_atom_penetration import (
    NEUTRAL_ATOM_PENETRATION_SUPPORTED_ATOMIC_NUMBERS,
    load_neutral_atom_penetration_mixtures,
    neutral_atom_penetration_potential,
    neutral_atom_penetration_potential_and_gradient,
)

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_TABLE = ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-neutral-atom-penetration-gaussian-mixture-v1.npz"
)
OFFICIAL_MANIFEST = OFFICIAL_TABLE.with_suffix(".json")


def _mixtures():
    return {
        1: GaussianMixtureAtom(
            electron_counts=np.asarray([0.3, 0.7]),
            gaussian_exponents_bohr2=np.asarray([0.4, 1.6]),
        ),
        2: GaussianMixtureAtom(
            electron_counts=np.asarray([2.0]),
            gaussian_exponents_bohr2=np.asarray([0.9]),
        ),
    }


def _central_gradient(function, point, step=1.0e-6):
    result = np.empty(3, dtype=float)
    for coordinate in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        result[coordinate] = (function(plus) - function(minus)) / (2.0 * step)
    return result


def test_official_asset_loads_with_exact_element_coverage_and_normalization():
    mixtures = load_neutral_atom_penetration_mixtures(
        table_path=OFFICIAL_TABLE,
        manifest_path=OFFICIAL_MANIFEST,
    )
    assert tuple(mixtures) == NEUTRAL_ATOM_PENETRATION_SUPPORTED_ATOMIC_NUMBERS
    for atomic_number, mixture in mixtures.items():
        assert mixture.electron_count == pytest.approx(
            float(atomic_number), rel=0.0, abs=5.0e-12
        )


def test_official_asset_rejects_tampering(tmp_path):
    table = tmp_path / OFFICIAL_TABLE.name
    table.write_bytes(OFFICIAL_TABLE.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_neutral_atom_penetration_mixtures(
            table_path=table,
            manifest_path=OFFICIAL_MANIFEST,
        )

    manifest = tmp_path / OFFICIAL_MANIFEST.name
    manifest.write_bytes(OFFICIAL_MANIFEST.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_neutral_atom_penetration_mixtures(
            table_path=OFFICIAL_TABLE,
            manifest_path=manifest,
        )


def test_neutral_atom_potential_is_nucleus_minus_gaussian_electron_cloud():
    point = np.asarray([[0.8, -0.4, 0.7]])
    mixture = _mixtures()[1]
    radius = float(np.linalg.norm(point[0]))
    expected = 1.0 / radius - sum(
        float(count) * erf(math.sqrt(float(exponent)) * radius) / radius
        for count, exponent in zip(
            mixture.electron_counts,
            mixture.gaussian_exponents_bohr2,
            strict=True,
        )
    )
    actual = neutral_atom_penetration_potential(
        points_bohr=point,
        centers_bohr=np.zeros((1, 3)),
        atomic_numbers=[1],
        mixtures_by_atomic_number={1: mixture},
    )
    assert actual[0] == pytest.approx(expected, rel=2.0e-15, abs=2.0e-16)


def test_spatial_gradient_matches_central_difference():
    point = np.asarray([0.7, -0.9, 0.5])
    centers = np.asarray([[0.1, -0.2, 0.3], [1.4, 0.3, -0.7]])
    numbers = np.asarray([1, 2])
    potential, gradient = neutral_atom_penetration_potential_and_gradient(
        points_bohr=point[None, :],
        centers_bohr=centers,
        atomic_numbers=numbers,
        mixtures_by_atomic_number=_mixtures(),
    )

    def value(candidate):
        return neutral_atom_penetration_potential(
            points_bohr=candidate[None, :],
            centers_bohr=centers,
            atomic_numbers=numbers,
            mixtures_by_atomic_number=_mixtures(),
        )[0]

    np.testing.assert_allclose(
        gradient[0],
        _central_gradient(value, point),
        rtol=3.0e-9,
        atol=3.0e-10,
    )
    assert potential.flags.writeable is False
    assert gradient.flags.writeable is False


def test_translation_and_rotation_covariance():
    points = np.asarray([[1.2, -0.2, 0.4], [-0.3, 1.5, 0.8]])
    centers = np.asarray([[0.1, 0.2, -0.4], [0.8, -0.7, 0.3]])
    numbers = np.asarray([1, 2])
    angle = 0.61
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shift = np.asarray([0.4, -1.3, 0.6])
    value, gradient = neutral_atom_penetration_potential_and_gradient(
        points_bohr=points,
        centers_bohr=centers,
        atomic_numbers=numbers,
        mixtures_by_atomic_number=_mixtures(),
    )
    transformed_value, transformed_gradient = (
        neutral_atom_penetration_potential_and_gradient(
            points_bohr=points @ rotation.T + shift,
            centers_bohr=centers @ rotation.T + shift,
            atomic_numbers=numbers,
            mixtures_by_atomic_number=_mixtures(),
        )
    )
    np.testing.assert_allclose(transformed_value, value, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        transformed_gradient,
        gradient @ rotation.T,
        rtol=2.0e-14,
        atol=2.0e-15,
    )


def test_far_field_decays_without_monopole_or_dipole_tail():
    mixture = _mixtures()[1]
    near = neutral_atom_penetration_potential(
        points_bohr=[[4.0, 0.0, 0.0]],
        centers_bohr=[[0.0, 0.0, 0.0]],
        atomic_numbers=[1],
        mixtures_by_atomic_number={1: mixture},
    )[0]
    far = neutral_atom_penetration_potential(
        points_bohr=[[8.0, 0.0, 0.0]],
        centers_bohr=[[0.0, 0.0, 0.0]],
        atomic_numbers=[1],
        mixtures_by_atomic_number={1: mixture},
    )[0]
    assert near > 0.0
    assert far >= 0.0
    assert far < near * 1.0e-5


def test_missing_element_non_neutral_mixture_and_nuclear_coincidence_fail_closed():
    with pytest.raises(ValueError, match="Z=8"):
        neutral_atom_penetration_potential(
            points_bohr=[[1.0, 0.0, 0.0]],
            centers_bohr=[[0.0, 0.0, 0.0]],
            atomic_numbers=[8],
            mixtures_by_atomic_number=_mixtures(),
        )
    with pytest.raises(ValueError, match="not normalized"):
        neutral_atom_penetration_potential(
            points_bohr=[[1.0, 0.0, 0.0]],
            centers_bohr=[[0.0, 0.0, 0.0]],
            atomic_numbers=[2],
            mixtures_by_atomic_number={
                2: GaussianMixtureAtom(
                    electron_counts=np.asarray([1.0]),
                    gaussian_exponents_bohr2=np.asarray([0.9]),
                )
            },
        )
    with pytest.raises(ValueError, match="singular"):
        neutral_atom_penetration_potential(
            points_bohr=[[0.0, 0.0, 0.0]],
            centers_bohr=[[0.0, 0.0, 0.0]],
            atomic_numbers=[1],
            mixtures_by_atomic_number={1: _mixtures()[1]},
        )
