from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.route2_pcm_response import (
    FixedCavityPCMReactionFieldLinearMap,
)


class _FakeSymmetricPCMSolverSession:
    def __init__(
        self,
        atom_positions_angstrom: np.ndarray,
        cavity_centers_bohr: np.ndarray,
        response_matrix: np.ndarray,
        *,
        symmetric: bool = True,
    ):
        self.atomic_numbers = np.ones(len(atom_positions_angstrom))
        self.coordinates_bohr = np.asarray(atom_positions_angstrom) / Bohr
        self.cavity_centers_bohr = np.asarray(cavity_centers_bohr, dtype=float)
        self.response_operator_is_symmetric = symmetric
        self._response_matrix = np.asarray(response_matrix, dtype=float)

    def compute_asc(self, mep: np.ndarray) -> np.ndarray:
        return self._response_matrix @ np.asarray(mep, dtype=float)


def _operator(*, symmetric: bool = True):
    positions = np.asarray(
        [
            [-0.7, 0.1, 0.2],
            [0.8, -0.2, 0.3],
            [0.2, 0.9, -0.4],
        ]
    )
    centers = np.asarray(
        [
            [-3.0, 0.0, 0.5],
            [2.5, 1.0, -0.2],
            [0.3, -2.8, 1.2],
            [1.4, 2.2, 2.0],
            [-1.7, 1.8, -2.1],
        ]
    )
    rng = np.random.default_rng(20260724)
    factor = rng.normal(size=(centers.shape[0], centers.shape[0]))
    response = factor + factor.T
    session = _FakeSymmetricPCMSolverSession(
        positions,
        centers,
        response,
        symmetric=symmetric,
    )
    return (
        FixedCavityPCMReactionFieldLinearMap(session, positions),
        positions,
        session,
    )


def test_fixed_cavity_pcm_apply_and_adjoint_obey_discrete_pairing():
    operator, positions, _ = _operator()
    rng = np.random.default_rng(19)
    density_direction = rng.normal(size=(len(positions), 4))
    field_cotangent = rng.normal(size=(len(positions), 4))

    applied = operator.apply(density_direction)
    adjoint = operator.adjoint(field_cotangent)

    assert np.vdot(field_cotangent, applied) == pytest.approx(
        np.vdot(adjoint, density_direction),
        rel=2.0e-13,
        abs=2.0e-11,
    )


def test_fixed_cavity_pcm_adjoint_matches_the_full_discrete_transpose():
    operator, positions, _ = _operator()
    dimension = 4 * len(positions)
    basis = np.eye(dimension)
    forward_matrix = np.column_stack(
        [
            operator.apply(vector.reshape(len(positions), 4)).reshape(-1)
            for vector in basis
        ]
    )
    adjoint_matrix = np.column_stack(
        [
            operator.adjoint(vector.reshape(len(positions), 4)).reshape(-1)
            for vector in basis
        ]
    )

    np.testing.assert_allclose(
        adjoint_matrix,
        forward_matrix.T,
        rtol=2.0e-13,
        atol=2.0e-11,
    )


def test_fixed_cavity_pcm_map_is_linear():
    operator, positions, _ = _operator()
    rng = np.random.default_rng(23)
    first = rng.normal(size=(len(positions), 4))
    second = rng.normal(size=(len(positions), 4))

    np.testing.assert_allclose(
        operator.apply(0.7 * first - 1.3 * second),
        0.7 * operator.apply(first) - 1.3 * operator.apply(second),
        rtol=2.0e-13,
        atol=2.0e-11,
    )


def test_fixed_cavity_pcm_map_can_reuse_surface_at_displaced_solute_positions():
    operator, positions, session = _operator()
    density = np.asarray(
        [
            [0.2, -0.1, 0.3, -0.4],
            [-0.3, 0.5, -0.2, 0.1],
            [0.1, -0.4, 0.2, 0.3],
        ]
    )
    displaced_positions = positions.copy()
    displaced_positions[1, 2] += 0.015
    original_field = operator.apply(density)

    displaced = operator.at_solute_positions(displaced_positions)
    applied = displaced.apply(density)
    mep = point_multipole_potential(
        session.cavity_centers_bohr,
        displaced_positions,
        density,
    )
    asc = session.compute_asc(mep)
    potential, gradient = point_asc_reaction_potential_gradient(
        displaced_positions,
        session.cavity_centers_bohr,
        asc,
    )
    expected = np.concatenate(
        (
            (potential * Hartree)[:, None],
            gradient * Hartree / Bohr,
        ),
        axis=1,
    )

    np.testing.assert_allclose(applied, expected, rtol=2.0e-13, atol=2.0e-11)
    np.testing.assert_allclose(
        operator.apply(density),
        original_field,
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_cavity_pcm_map_rejects_surface_motion_beyond_readback_tolerance():
    operator, positions, session = _operator()
    original_centers = session.cavity_centers_bohr.copy()

    session.cavity_centers_bohr[0, 0] += 0.5e-12 / Bohr
    operator.at_solute_positions(positions)

    session.cavity_centers_bohr = original_centers.copy()
    session.cavity_centers_bohr[0, 0] += 1.0e-8 / Bohr
    with pytest.raises(RuntimeError, match="surface changed"):
        operator.at_solute_positions(positions)


def test_fixed_surface_pcm_position_vjp_matches_central_difference():
    operator, positions, session = _operator()
    rng = np.random.default_rng(29)
    density = rng.normal(size=(len(positions), 4))
    field_cotangent = rng.normal(size=(len(positions), 4))

    analytic = operator.position_vjp(density, field_cotangent)
    finite_difference = np.empty_like(positions)
    step_angstrom = 1.0e-6

    def scalar(position_values: np.ndarray) -> float:
        mep = point_multipole_potential(
            session.cavity_centers_bohr,
            position_values,
            density,
        )
        asc = session.compute_asc(mep)
        potential, gradient = point_asc_reaction_potential_gradient(
            position_values,
            session.cavity_centers_bohr,
            asc,
        )
        field = np.concatenate(
            (
                (potential * Hartree)[:, None],
                gradient * Hartree / Bohr,
            ),
            axis=1,
        )
        return float(np.vdot(field_cotangent, field))

    for atom_index in range(len(positions)):
        for coordinate in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, coordinate] += step_angstrom
            minus[atom_index, coordinate] -= step_angstrom
            finite_difference[atom_index, coordinate] = (
                scalar(plus) - scalar(minus)
            ) / (2.0 * step_angstrom)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-7,
    )


def test_fixed_cavity_pcm_map_requires_hermitivized_response():
    with pytest.raises(ValueError, match="MATRIXSYMM=TRUE"):
        _operator(symmetric=False)


def test_fixed_cavity_pcm_map_rejects_geometry_mismatch():
    _, positions, session = _operator()
    shifted = positions.copy()
    shifted[0, 0] += 1.0e-5

    with pytest.raises(ValueError, match="does not match"):
        FixedCavityPCMReactionFieldLinearMap(session, shifted)


@pytest.mark.parametrize(
    "bad_values",
    [
        np.zeros((2, 4)),
        np.full((3, 4), np.nan),
    ],
)
def test_fixed_cavity_pcm_map_rejects_invalid_blocks(bad_values):
    operator, _, _ = _operator()

    with pytest.raises(ValueError, match="finite with shape"):
        operator.apply(bad_values)
    with pytest.raises(ValueError, match="finite with shape"):
        operator.adjoint(bad_values)
    with pytest.raises(ValueError, match="finite with shape"):
        operator.position_vjp(bad_values, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="finite with shape"):
        operator.position_vjp(np.zeros((3, 4)), bad_values)
