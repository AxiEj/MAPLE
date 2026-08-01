"""Algebraic gates for the JGP94 fixed-topology C-PCM wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_jgp94_fc_aswig import (
    JGP94BodyFrameFixedTopologyAqueousSMDCDS,
    JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse,
)


_SIX_POINT_SPHERE = np.asarray(
    [
        [1.0, 0.0, 0.0, 1.0 / 6.0],
        [-1.0, 0.0, 0.0, 1.0 / 6.0],
        [0.0, 1.0, 0.0, 1.0 / 6.0],
        [0.0, -1.0, 0.0, 1.0 / 6.0],
        [0.0, 0.0, 1.0, 1.0 / 6.0],
        [0.0, 0.0, -1.0, 1.0 / 6.0],
    ],
    dtype=float,
)
_SWITCHING_CONSTANT = 4.84566077868
_SYMBOLS = ("C", "O", "N", "H")
_NUCLEAR_CHARGES = np.asarray([6.0, 8.0, 7.0, 1.0])
_RADII = np.asarray([1.70, 1.52, 1.55, 1.20])
_POSITIONS = np.asarray(
    [
        [-1.4, -0.2, 0.3],
        [-0.1, 0.7, -0.4],
        [1.2, -0.5, 0.2],
        [0.4, 1.5, 0.8],
    ],
    dtype=float,
)
_DENSITY = np.asarray(
    [
        [0.30, 0.10, -0.20, 0.40],
        [-0.20, 0.30, 0.40, -0.10],
        [0.10, -0.50, 0.10, 0.20],
        [-0.20, 0.20, 0.10, -0.30],
    ],
    dtype=float,
)
_FIELD_COTANGENT = np.asarray(
    [
        [0.22, -0.11, 0.14, -0.07],
        [-0.13, 0.17, -0.05, 0.19],
        [0.09, 0.08, -0.21, 0.03],
        [-0.18, -0.04, 0.16, -0.12],
    ],
    dtype=float,
)


def _rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    theta = np.deg2rad(degrees)
    x, y, z = vector
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * skew @ skew


def _rotate_density(density: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    external = density_to_external_field_order(density)
    rotated = np.array(external, copy=True)
    rotated[:, 1:] = external[:, 1:] @ rotation.T
    return external_field_to_density_order(rotated)


def _rotate_field(field: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(field, copy=True)
    result[:, 1:] = field[:, 1:] @ rotation.T
    return result


def _response(positions: np.ndarray):
    return JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse(
        _SYMBOLS,
        positions,
        _RADII,
        dielectric=78.39,
        nuclear_charges=_NUCLEAR_CHARGES,
        minimum_relative_eigengap=0.01,
        _unit_sphere=_SIX_POINT_SPHERE,
        _switching_constant=_SWITCHING_CONSTANT,
    )


def _map(positions: np.ndarray):
    return _response(positions).reaction_field_linear_map(positions)


def _cds(positions: np.ndarray):
    response = _response(positions)
    return JGP94BodyFrameFixedTopologyAqueousSMDCDS(
        _SYMBOLS,
        response.frame,
        _NUCLEAR_CHARGES,
        _unit_sphere=_SIX_POINT_SPHERE,
        _switching_constant=_SWITCHING_CONSTANT,
    ).result()


def test_jgp94_fixed_topology_map_is_rotation_covariant_without_lab_grid_error():
    base = _map(_POSITIONS)
    rotation = _rotation((1.0, -2.0, 3.0), 71.0)
    rotated_positions = _POSITIONS @ rotation.T
    rotated_density = _rotate_density(_DENSITY, rotation)
    rotated_cotangent = _rotate_field(_FIELD_COTANGENT, rotation)
    rotated = _map(rotated_positions)

    base_energy = base.scf_polarization_energy_hartree(_DENSITY)
    rotated_energy = rotated.scf_polarization_energy_hartree(rotated_density)
    assert rotated_energy == pytest.approx(base_energy, abs=2.0e-14, rel=0.0)

    base_field = base.apply(_DENSITY)
    rotated_field = rotated.apply(rotated_density)
    np.testing.assert_allclose(
        rotated_field,
        _rotate_field(base_field, rotation),
        rtol=0.0,
        atol=2.0e-13,
    )

    base_vjp = base.full_position_vjp(_DENSITY, _FIELD_COTANGENT)
    rotated_vjp = rotated.full_position_vjp(rotated_density, rotated_cotangent)
    np.testing.assert_allclose(
        rotated_vjp,
        base_vjp @ rotation.T,
        rtol=0.0,
        atol=2.0e-11,
    )


def test_jgp94_fixed_topology_map_has_full_lab_coordinate_vjp():
    reaction = _map(_POSITIONS)
    analytic = reaction.full_position_vjp(_DENSITY, _FIELD_COTANGENT)
    finite_difference = np.empty_like(_POSITIONS)
    step = 1.0e-5
    for atom_index in range(len(_POSITIONS)):
        for axis_index in range(3):
            values = []
            for sign in (-1.0, 1.0):
                displaced = _POSITIONS.copy()
                displaced[atom_index, axis_index] += sign * step
                values.append(
                    float(
                        np.vdot(
                            _FIELD_COTANGENT,
                            _map(displaced).apply(_DENSITY),
                        )
                    )
                )
            finite_difference[atom_index, axis_index] = (
                values[1] - values[0]
            ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=0.0,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        np.sum(analytic, axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=3.0e-12,
    )


def test_jgp94_fixed_topology_map_is_translation_invariant_and_reciprocal():
    base = _map(_POSITIONS)
    translated = _map(_POSITIONS + np.asarray([2.1, -1.3, 0.7]))
    np.testing.assert_allclose(
        translated.apply(_DENSITY),
        base.apply(_DENSITY),
        rtol=0.0,
        atol=2.0e-13,
    )
    assert translated.scf_polarization_energy_hartree(_DENSITY) == pytest.approx(
        base.scf_polarization_energy_hartree(_DENSITY),
        abs=2.0e-14,
        rel=0.0,
    )

    left = np.roll(_DENSITY, 1, axis=0)
    assert float(np.vdot(_FIELD_COTANGENT, base.apply(left))) == pytest.approx(
        float(np.vdot(base.adjoint(_FIELD_COTANGENT), left)),
        abs=2.0e-12,
        rel=0.0,
    )
    assert (
        0.5 * MACE_POLAR_L1_PAIRING.pair(_DENSITY, base.apply_scf(_DENSITY))
    ) == pytest.approx(
        base.scf_polarization_energy_hartree(_DENSITY) * Hartree,
        abs=2.0e-12,
        rel=0.0,
    )


def test_jgp94_fixed_topology_profile_fails_closed_at_moment_degeneracy():
    with pytest.raises(RuntimeError, match="nondegenerate eigengap"):
        JGP94BodyFrameFixedTopologyAmplitudeSWIGCPCMResponse(
            ("H", "H"),
            np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([1.1, 1.1]),
            dielectric=78.39,
            minimum_relative_eigengap=0.01,
            _unit_sphere=_SIX_POINT_SPHERE,
            _switching_constant=_SWITCHING_CONSTANT,
        )


def test_jgp94_fixed_topology_aqueous_cds_has_rotation_covariance_and_fd_gradient():
    base = _cds(_POSITIONS)
    rotation = _rotation((-2.0, 1.0, 3.0), 53.0)
    rotated = _cds(_POSITIONS @ rotation.T)
    assert rotated.energy_hartree == pytest.approx(
        base.energy_hartree,
        abs=2.0e-14,
        rel=0.0,
    )
    np.testing.assert_allclose(
        rotated.position_gradient_hartree_per_angstrom,
        base.position_gradient_hartree_per_angstrom @ rotation.T,
        rtol=0.0,
        atol=3.0e-12,
    )

    step = 1.0e-5
    analytic = base.position_gradient_hartree_per_angstrom
    finite_difference = np.empty_like(analytic)
    for atom_index in range(len(_POSITIONS)):
        for axis_index in range(3):
            values = []
            for sign in (-1.0, 1.0):
                displaced = _POSITIONS.copy()
                displaced[atom_index, axis_index] += sign * step
                values.append(_cds(displaced).energy_hartree)
            finite_difference[atom_index, axis_index] = (
                values[1] - values[0]
            ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=0.0,
        atol=2.0e-8,
    )
