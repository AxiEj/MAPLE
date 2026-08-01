"""Algebraic gates for the JGP94-D2 MACE boundary canonicalisation."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.gto_density import (
    density_to_external_field_order,
    external_field_to_density_order,
)
from maple.function.calculator.extra_correction.implicit.route2_jgp94_mace_frame import (
    JGP94_D2_CANONICAL_MACE_FRAME_POLICY,
    JGP94D2CanonicalDensityResponse,
    JGP94D2CanonicalMACEContext,
)


_POSITIONS = np.asarray(
    [
        [-1.40, -0.20, 0.30],
        [-0.10, 0.70, -0.40],
        [1.20, -0.50, 0.20],
        [0.40, 1.50, 0.80],
    ],
    dtype=float,
)
_NUMBERS = np.asarray([6, 8, 7, 1], dtype=int)


def _atoms(positions: np.ndarray = _POSITIONS) -> Atoms:
    atoms = Atoms(numbers=_NUMBERS, positions=positions)
    atoms.info["charge"] = 0
    atoms.info["mult"] = 1
    return atoms


def _rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    theta = np.deg2rad(degrees)
    x, y, z = vector
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * skew @ skew


def _rotate_raw_density(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    external = density_to_external_field_order(values)
    rotated = np.array(external, copy=True)
    rotated[:, 1:] = external[:, 1:] @ rotation.T
    return external_field_to_density_order(rotated)


def _rotate_field(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result[:, 1:] = values[:, 1:] @ rotation.T
    return result


class _DenseNativeResponse:
    """A deliberately non-equivariant canonical-frame linear test operator."""

    def __init__(self, matrix: np.ndarray, atom_count: int):
        self._matrix = np.asarray(matrix, dtype=float)
        self._atom_count = atom_count

    def jvp(self, field_direction: np.ndarray) -> np.ndarray:
        return (self._matrix @ np.asarray(field_direction).reshape(-1)).reshape(
            self._atom_count,
            4,
        )

    def vjp(self, density_cotangent: np.ndarray) -> np.ndarray:
        return (self._matrix.T @ np.asarray(density_cotangent).reshape(-1)).reshape(
            self._atom_count,
            4,
        )


def _response(context: JGP94D2CanonicalMACEContext) -> JGP94D2CanonicalDensityResponse:
    rng = np.random.default_rng(20260801)
    size = 4 * context.atom_count
    matrix = rng.normal(size=(size, size))
    return JGP94D2CanonicalDensityResponse(
        context=context,
        branch_responses=tuple(
            _DenseNativeResponse(matrix, context.atom_count)
            for _ in context.branches
        ),
    )


def test_d2_canonical_context_eliminates_principal_axis_sign_gauge_under_rotation():
    base = JGP94D2CanonicalMACEContext.from_atoms(
        _atoms(),
        minimum_relative_eigengap=0.01,
    )
    rotation = _rotation((1.0, -2.0, 3.0), 71.0)
    rotated = JGP94D2CanonicalMACEContext.from_atoms(
        _atoms(_POSITIONS @ rotation.T),
        minimum_relative_eigengap=0.01,
    )

    # Individual eigenvector signs are intentionally not asserted.  Instead,
    # the unordered proper-D2 branch orbit must be exactly the same body
    # geometry, which is the canonicalisation's actual invariant.
    base_orbit = sorted(
        tuple(np.round(branch.frame.body_positions, 13).ravel())
        for branch in base.branches
    )
    rotated_orbit = sorted(
        tuple(np.round(branch.frame.body_positions, 13).ravel())
        for branch in rotated.branches
    )
    assert base_orbit == rotated_orbit
    assert base.policy == JGP94_D2_CANONICAL_MACE_FRAME_POLICY
    assert base.provenance["mace_geometry_frame_branch_count"] == 4


def test_d2_canonical_density_response_is_adjoint_consistent_and_rotation_covariant():
    base_context = JGP94D2CanonicalMACEContext.from_atoms(
        _atoms(),
        minimum_relative_eigengap=0.01,
    )
    base = _response(base_context)
    rng = np.random.default_rng(20260802)
    direction = rng.normal(size=(len(_NUMBERS), 4))
    cotangent = rng.normal(size=(len(_NUMBERS), 4))
    assert float(np.vdot(cotangent, base.jvp(direction))) == pytest.approx(
        float(np.vdot(base.vjp(cotangent), direction)),
        abs=3.0e-12,
        rel=0.0,
    )

    rotation = _rotation((-2.0, 1.0, 3.0), 53.0)
    rotated_context = JGP94D2CanonicalMACEContext.from_atoms(
        _atoms(_POSITIONS @ rotation.T),
        minimum_relative_eigengap=0.01,
    )
    rotated = _response(rotated_context)
    rotated_direction = _rotate_field(direction, rotation)
    np.testing.assert_allclose(
        rotated.jvp(rotated_direction),
        _rotate_raw_density(base.jvp(direction), rotation),
        rtol=0.0,
        atol=3.0e-12,
    )


def test_d2_canonical_context_fails_closed_at_jgp94_degeneracy():
    with pytest.raises(RuntimeError, match="nondegenerate eigengap"):
        JGP94D2CanonicalMACEContext.from_atoms(
            Atoms("HH", positions=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            minimum_relative_eigengap=0.01,
        )
