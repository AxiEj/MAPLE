from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.release.uniform_response_manifold import (
    affine_uniform_native_field,
    molecular_dipole_eangstrom,
)
from maple.solvation.release.uniform_susceptibility_replacement import (
    prepare_uniform_susceptibility_field_transform,
)


def _rotation() -> np.ndarray:
    axis = np.array([0.3, -0.4, 0.5], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _rotate_native_field(field: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(field, copy=True)
    for channels in ((4, 2, 3), (7, 5, 6)):
        cartesian = field[:, channels]
        rotated = cartesian @ rotation.T
        result[:, channels] = rotated
    return result


def _rotate_source_jacobian(jacobian: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.empty_like(jacobian)
    for axis in range(3):
        old_axis_source = np.einsum("nsc,c->ns", jacobian, rotation.T[:, axis])
        result[:, 0, axis] = old_axis_source[:, 0]
        old_dipoles = old_axis_source[:, (3, 1, 2)]
        rotated_dipoles = old_dipoles @ rotation.T
        result[:, (3, 1, 2), axis] = rotated_dipoles
    return result


def _problem():
    rng = np.random.default_rng(20260817)
    atom_count = 3
    positions = rng.normal(size=(atom_count, 3))
    uniform_basis = np.stack(
        [affine_uniform_native_field(positions, np.eye(3)[axis]) for axis in range(3)],
        axis=-1,
    )

    polar_uniform_jacobian = rng.normal(scale=0.03, size=(atom_count, 4, 3))
    polar_uniform_jacobian[:, 0, :] -= np.mean(
        polar_uniform_jacobian[:, 0, :], axis=0, keepdims=True
    )
    polar_molecular = np.column_stack(
        [
            molecular_dipole_eangstrom(positions, polar_uniform_jacobian[:, :, axis])
            for axis in range(3)
        ]
    )
    # Make the synthetic response nonsingular and physically signed without
    # changing its atomwise response span.
    desired_polar_alpha = np.diag([0.8, 0.9, 1.0])
    polar_uniform_jacobian = np.einsum(
        "nsc,cd->nsd",
        polar_uniform_jacobian,
        np.linalg.solve(polar_molecular, -desired_polar_alpha),
    )
    mdp_alpha = np.array([[1.1, 0.05, 0.02], [0.05, 1.3, 0.03], [0.02, 0.03, 1.5]])
    prepared = prepare_uniform_susceptibility_field_transform(
        positions_angstrom=positions,
        uniform_native_basis=uniform_basis,
        polar_zero_uniform_source_jacobian=polar_uniform_jacobian,
        mdp_molecular_polarizability_eangstrom2_per_volt=mdp_alpha,
    )
    return rng, positions, prepared, uniform_basis, polar_uniform_jacobian, mdp_alpha


def test_closes_molecular_susceptibility_without_inserting_mdp_atomic_response() -> (
    None
):
    _, positions, prepared, uniform_basis, polar_jacobian, mdp_alpha = _problem()
    coefficients = np.array([0.12, -0.07, 0.03])
    field = np.einsum("nfc,c->nf", uniform_basis, coefficients)
    transformed_field = prepared.transform_field(field)
    expected_field = np.einsum(
        "nfc,c->nf",
        uniform_basis,
        prepared.uniform_coordinate_transform @ coefficients,
    )
    assert transformed_field == pytest.approx(expected_field, abs=2.0e-14)
    assert np.array_equal(
        prepared.transform_field(np.zeros_like(field)), np.zeros_like(field)
    )

    transformed_source_jacobian = np.einsum(
        "nsc,cd->nsd", polar_jacobian, prepared.uniform_coordinate_transform
    )
    molecular = np.column_stack(
        [
            molecular_dipole_eangstrom(
                positions, transformed_source_jacobian[:, :, axis]
            )
            for axis in range(3)
        ]
    )
    assert transformed_source_jacobian == pytest.approx(
        prepared.transformed_uniform_source_jacobian, abs=2.0e-14
    )
    assert molecular == pytest.approx(-mdp_alpha, abs=2.0e-13)
    assert np.max(np.abs(np.sum(transformed_source_jacobian[:, 0, :], axis=0))) < (
        2.0e-14
    )


def test_nonuniform_zero_mean_gradient_field_is_left_unchanged() -> None:
    rng, _, prepared, _, _, _ = _problem()
    direction = rng.normal(size=(3, 8))
    coordinates = prepared.uniform_coordinates(direction)
    assert direction.flags.writeable
    direction[:, (4, 7)] -= coordinates[0]
    direction[:, (2, 5)] -= coordinates[1]
    direction[:, (3, 6)] -= coordinates[2]

    assert prepared.uniform_coordinates(direction) == pytest.approx(
        np.zeros(3), abs=2.0e-14
    )
    assert prepared.transform_field(direction) == pytest.approx(direction, abs=2.0e-14)

    # Potential values and their constant gauge have different units from the
    # gradients and must never be interpreted as a uniform electric field.
    potential_only = np.zeros_like(direction)
    potential_only[:, :2] = rng.normal(size=(3, 2))
    assert np.array_equal(prepared.uniform_coordinates(potential_only), np.zeros(3))
    assert np.array_equal(prepared.transform_field(potential_only), potential_only)


def test_field_transform_jvp_and_vjp_are_exact_transposes() -> None:
    rng, _, prepared, _, _, _ = _problem()
    direction = rng.normal(size=(3, 8))
    cotangent = rng.normal(size=(3, 8))

    tangent = prepared.transform_field_direction(direction)
    pulled_back = prepared.pullback_field_cotangent(cotangent)
    assert float(np.vdot(cotangent, tangent)) == pytest.approx(
        float(np.vdot(pulled_back, direction)), abs=2.0e-13
    )


def test_additive_tangent_form_has_same_zero_field_uniform_susceptibility() -> None:
    _, positions, prepared, uniform_basis, polar_jacobian, mdp_alpha = _problem()
    corrected_jacobian = np.empty_like(polar_jacobian)
    zero_source = np.zeros((len(positions), 4))
    for axis in range(3):
        correction = prepared.tangent_source_direction(uniform_basis[:, :, axis])
        corrected_jacobian[:, :, axis] = polar_jacobian[:, :, axis] + correction
        assert prepared.tangent_corrected_source(
            zero_source, uniform_basis[:, :, axis]
        ) == pytest.approx(correction, abs=2.0e-14)

    molecular = np.column_stack(
        [
            molecular_dipole_eangstrom(positions, corrected_jacobian[:, :, axis])
            for axis in range(3)
        ]
    )
    assert corrected_jacobian == pytest.approx(
        prepared.transformed_uniform_source_jacobian, abs=2.0e-14
    )
    assert molecular == pytest.approx(-mdp_alpha, abs=2.0e-13)
    assert np.max(np.abs(np.sum(corrected_jacobian[:, 0, :], axis=0))) < 2.0e-14
    assert np.array_equal(
        prepared.tangent_source_correction(np.zeros((len(positions), 8))),
        np.zeros((len(positions), 4)),
    )


def test_additive_tangent_correction_leaves_nonuniform_chart_kernel_unchanged() -> None:
    rng, _, prepared, _, _, _ = _problem()
    field = rng.normal(size=(prepared.atom_count, 8))
    coordinates = prepared.uniform_coordinates(field)
    field[:, (4, 7)] -= coordinates[0]
    field[:, (2, 5)] -= coordinates[1]
    field[:, (3, 6)] -= coordinates[2]
    assert prepared.uniform_coordinates(field) == pytest.approx(
        np.zeros(3), abs=2.0e-14
    )
    assert prepared.tangent_source_correction(field) == pytest.approx(
        np.zeros((prepared.atom_count, 4)), abs=2.0e-14
    )


def test_additive_tangent_jvp_and_vjp_are_exact_transposes() -> None:
    rng, _, prepared, _, _, _ = _problem()
    direction = rng.normal(size=(prepared.atom_count, 8))
    source_cotangent = rng.normal(size=(prepared.atom_count, 4))
    tangent = prepared.tangent_source_direction(direction)
    pulled_back = prepared.pullback_tangent_source_cotangent(source_cotangent)
    assert float(np.vdot(source_cotangent, tangent)) == pytest.approx(
        float(np.vdot(pulled_back, direction)), abs=2.0e-13
    )


def test_transform_is_rotation_translation_and_permutation_covariant() -> None:
    rng, positions, prepared, _, polar_jacobian, mdp_alpha = _problem()
    rotation = _rotation()
    permutation = np.array([2, 0, 1])
    shift = np.array([1.7, -0.6, 0.9])
    transformed_positions = (positions @ rotation.T + shift)[permutation]
    transformed_basis = np.stack(
        [
            affine_uniform_native_field(transformed_positions, np.eye(3)[axis])
            for axis in range(3)
        ],
        axis=-1,
    )
    transformed_jacobian = _rotate_source_jacobian(polar_jacobian, rotation)[
        permutation
    ]
    transformed_alpha = rotation @ mdp_alpha @ rotation.T
    transformed = prepare_uniform_susceptibility_field_transform(
        positions_angstrom=transformed_positions,
        uniform_native_basis=transformed_basis,
        polar_zero_uniform_source_jacobian=transformed_jacobian,
        mdp_molecular_polarizability_eangstrom2_per_volt=transformed_alpha,
    )

    field = rng.normal(size=(3, 8))
    rotated_permuted_field = _rotate_native_field(field, rotation)[permutation]
    expected = _rotate_native_field(prepared.transform_field(field), rotation)[
        permutation
    ]
    assert transformed.transform_field(rotated_permuted_field) == pytest.approx(
        expected, abs=3.0e-13
    )
    assert transformed.uniform_coordinate_transform == pytest.approx(
        rotation @ prepared.uniform_coordinate_transform @ rotation.T,
        abs=3.0e-13,
    )


def test_fail_closed_for_nonphysical_or_ill_conditioned_inputs() -> None:
    _, positions, _, uniform_basis, polar_jacobian, mdp_alpha = _problem()
    with pytest.raises(ValueError, match="positive definite"):
        prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=uniform_basis,
            polar_zero_uniform_source_jacobian=polar_jacobian,
            mdp_molecular_polarizability_eangstrom2_per_volt=np.zeros((3, 3)),
        )
    bad_basis = uniform_basis.copy()
    bad_basis[..., 2] = bad_basis[..., 1]
    with pytest.raises(ValueError, match="rank"):
        prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=bad_basis,
            polar_zero_uniform_source_jacobian=polar_jacobian,
            mdp_molecular_polarizability_eangstrom2_per_volt=mdp_alpha,
        )
    bad_jacobian = polar_jacobian.copy()
    bad_jacobian[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="total charge"):
        prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=uniform_basis,
            polar_zero_uniform_source_jacobian=bad_jacobian,
            mdp_molecular_polarizability_eangstrom2_per_volt=mdp_alpha,
        )
    singular_response = polar_jacobian.copy()
    singular_response[..., 2] = singular_response[..., 1]
    with pytest.raises(ValueError, match="rank"):
        prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=uniform_basis,
            polar_zero_uniform_source_jacobian=singular_response,
            mdp_molecular_polarizability_eangstrom2_per_volt=mdp_alpha,
        )
