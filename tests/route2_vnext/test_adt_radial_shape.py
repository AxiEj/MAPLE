from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.adt_radial_shape import (
    ADT_TRANSLATION_SHAPE_ROLE,
    ADTRadialShape,
    ADTRadialShapeRegistry,
    load_repository_adt_radial_shape_registry,
)
from maple.solvation.coupling.atomic_displacement_lift import (
    CANONICAL_ADT_LIFT_CONTRACT,
    ROLE_SEPARATED_ADT_LIFT_CONTRACT,
    CanonicalAtomicDisplacementLift,
    gaussian_translation_dipole_self_work,
)
from maple.solvation.coupling.neutral_atom_penetration import (
    NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
    NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
    NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
    load_neutral_atom_penetration_mixtures,
    neutral_atom_penetration_potential,
)


def _registry() -> ADTRadialShapeRegistry:
    return load_repository_adt_radial_shape_registry(source_root=".")


def test_repository_registry_separates_iodine_valence_response_role() -> None:
    registry = _registry()
    iodine = registry.shape(53)
    assert iodine.role == ADT_TRANSLATION_SHAPE_ROLE
    assert iodine.effective_electron_count == pytest.approx(25.0, abs=0.0)
    assert iodine.ecp_core_electron_count == 28
    assert iodine.mixture.electron_count == pytest.approx(25.0, abs=5.0e-15)
    assert iodine.representation == "ecp_valence_pseudodensity"
    assert iodine.neutral_penetration_compatible is False
    assert registry.supported_atomic_numbers == (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)


def test_iodine_role_shape_cannot_masquerade_as_neutral_penetration_density() -> None:
    iodine = _registry().shape(53)
    with pytest.raises(ValueError, match="not normalized to Z"):
        neutral_atom_penetration_potential(
            points_bohr=np.asarray([[8.0, 0.0, 0.0]]),
            centers_bohr=np.zeros((1, 3)),
            atomic_numbers=np.asarray([53]),
            mixtures_by_atomic_number={53: iodine.mixture},
        )


def test_role_separated_lift_uses_effective_count_and_exact_far_field_dipole() -> None:
    lift = CanonicalAtomicDisplacementLift.from_radial_shapes(
        atomic_numbers=[53], registry=_registry()
    )
    assert lift.contract_id == ROLE_SEPARATED_ADT_LIFT_CONTRACT
    assert lift.effective_electron_counts_by_atomic_number[53] == 25.0
    dipole = np.asarray([[0.7, -0.4, 0.2]])
    far = np.asarray([[4.0e6, -2.0e6, 3.0e6]])
    potential = lift.tangent_potential(
        points_bohr=far,
        centers_bohr=np.zeros((1, 3)),
        atomic_dipoles_ebohr=dipole,
    )[0]
    expected = float(dipole[0] @ far[0]) / np.linalg.norm(far[0]) ** 3
    assert potential == pytest.approx(expected, rel=3.0e-12, abs=0.0)

    point = np.asarray([[8.0, 1.0, -0.3]])
    centers = np.zeros((1, 3))

    def linearization_error(scale: float) -> float:
        exact = lift.exact_translated_potential_difference(
            points_bohr=point,
            centers_bohr=centers,
            atomic_dipoles_ebohr=scale * dipole,
        )
        tangent = lift.tangent_potential(
            points_bohr=point,
            centers_bohr=centers,
            atomic_dipoles_ebohr=scale * dipole,
        )
        return float(np.linalg.norm(exact - tangent))

    # A first-order translation tangent has an O(scale**2) remainder.  Verify
    # that rate directly rather than demanding an impossible relative error at
    # a finite displacement.
    coarse_error = linearization_error(2.0e-4)
    fine_error = linearization_error(1.0e-4)
    assert coarse_error / fine_error == pytest.approx(4.0, rel=8.0e-3)


def test_uniform_density_rescaling_leaves_adt_tangent_and_self_work_exact() -> None:
    iodine = _registry().shape(53)
    scaled_mixture = GaussianMixtureAtom(
        iodine.mixture.electron_counts * (53.0 / 25.0),
        iodine.mixture.gaussian_exponents_bohr2,
    )
    scaled_shape = ADTRadialShape(
        atomic_number=53,
        effective_electron_count=53.0,
        ecp_core_electron_count=0,
        mixture=scaled_mixture,
        representation="uniformly_scaled_adt_shape_algebra_control",
        source_asset_sha256="7" * 64,
    )
    scaled_registry = ADTRadialShapeRegistry(
        shapes_by_atomic_number={53: scaled_shape},
        parent_asset_sha256s=("7" * 64,),
    )
    physical = CanonicalAtomicDisplacementLift.from_radial_shapes(
        atomic_numbers=[53], registry=_registry()
    )
    scaled = CanonicalAtomicDisplacementLift.from_radial_shapes(
        atomic_numbers=[53], registry=scaled_registry
    )
    assert gaussian_translation_dipole_self_work(iodine.mixture) == pytest.approx(
        gaussian_translation_dipole_self_work(scaled_mixture), rel=2.0e-15, abs=0.0
    )
    points = np.asarray([[2.1, -0.4, 0.8], [7.0, 1.5, -2.0]])
    centers = np.zeros((1, 3))
    dipole = np.asarray([[0.3, -0.2, 0.5]])
    np.testing.assert_allclose(
        physical.tangent_potential(
            points_bohr=points,
            centers_bohr=centers,
            atomic_dipoles_ebohr=dipole,
        ),
        scaled.tangent_potential(
            points_bohr=points,
            centers_bohr=centers,
            atomic_dipoles_ebohr=dipole,
        ),
        atol=2.0e-17,
        rtol=2.0e-15,
    )


def test_noniodine_v2_operator_is_bitwise_equal_to_legacy_v1() -> None:
    neutral = load_neutral_atom_penetration_mixtures(
        table_path=NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
        manifest_path=NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
    )
    numbers = np.asarray([6, 1, 8, 1], dtype=np.int64)
    legacy = CanonicalAtomicDisplacementLift.from_mixtures(
        atomic_numbers=numbers,
        mixtures_by_atomic_number=neutral,
        source_asset_sha256=NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
    )
    role_separated = CanonicalAtomicDisplacementLift.from_radial_shapes(
        atomic_numbers=numbers,
        registry=_registry(),
    )
    assert legacy.contract_id == CANONICAL_ADT_LIFT_CONTRACT
    assert legacy.configuration_sha256 == (
        "b8182ce5704c30c150030dcff0383820f2935910a411701fee34062bfef3840c"
    )
    np.testing.assert_array_equal(
        role_separated.self_work_hartree_per_ebohr2,
        legacy.self_work_hartree_per_ebohr2,
    )
    np.testing.assert_array_equal(
        role_separated.atomic_dipole_weights, legacy.atomic_dipole_weights
    )
    points = np.asarray([[3.1, -0.2, 0.7], [-1.4, 2.6, 0.5]])
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.2, -0.3], [-0.5, 0.7, 0.1], [1.8, -0.9, 0.4]]
    )
    np.testing.assert_array_equal(
        role_separated.atomic_surface_operator(
            points_bohr=points, centers_bohr=centers
        ),
        legacy.atomic_surface_operator(points_bohr=points, centers_bohr=centers),
    )


def test_registry_and_shapes_are_immutable_and_content_bound() -> None:
    registry = _registry()
    iodine = registry.shape(53)
    with pytest.raises(ValueError, match="configuration SHA256"):
        replace(iodine, representation="tampered-iodine-response-shape")
    with pytest.raises(TypeError):
        registry.shapes_by_atomic_number[53] = iodine
