from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.continuum import (
    SEPARATED_SOURCE_DDX_PROVIDER_ID,
    SeparatedSourceDDXBackend,
    SeparatedGeneralDDXSolution,
    SeparatedSourceDDXState,
    embed_atomic_l1_in_first_radial_channel,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING

POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.43, 0.10, -0.20], [0.20, 1.10, 0.30]])
PERMANENT = np.asarray(
    [
        [0.12, 0.02, 0.01, -0.01],
        [-0.08, -0.01, 0.03, 0.02],
        [-0.04, 0.004, -0.005, 0.006],
    ]
)
RADIAL = np.asarray(
    [
        [0.01, -0.003, 0.002, 0.001, -0.001, 0.0004, 0.0008, -0.0003],
        [-0.008, 0.001, -0.001, 0.003, 0.002, -0.0006, 0.0002, 0.0009],
        [-0.002, 0.0, 0.0004, -0.0005, 0.0006, 0.0001, -0.0003, 0.0002],
    ]
)


def _backend() -> SeparatedSourceDDXBackend:
    return SeparatedSourceDDXBackend(
        ("H", "H", "H"),
        np.asarray([1.5, 1.6, 1.2]),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )


def test_import_is_dependency_light_without_pyddx():
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pyddx' or name.startswith('pyddx.'):
        raise AssertionError('pyddx imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from maple.solvation.continuum.separated_source_ddx import SeparatedSourceDDXBackend
print(SeparatedSourceDDXBackend.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "SeparatedSourceDDXBackend"


def test_atomic_l1_embedding_keeps_only_the_first_radial_block():
    embedded = embed_atomic_l1_in_first_radial_channel(PERMANENT)
    np.testing.assert_array_equal(embedded[:, (0, 2, 3, 4)], PERMANENT)
    np.testing.assert_array_equal(embedded[:, (1, 5, 6, 7)], 0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_direct_sum_keeps_operational_field_and_energy_gradient_distinct():
    backend = _backend()
    assert backend.provider_id == SEPARATED_SOURCE_DDX_PROVIDER_ID
    assert isinstance(backend.capabilities, CapabilityStatus)
    assert backend.capabilities.enabled_tiers == ()
    prepared = backend.prepare(POSITIONS, PERMANENT)
    assert prepared.cavity_points_bohr.shape == (prepared.cavity_point_count, 3)
    assert not prepared.cavity_points_bohr.flags.writeable
    assert prepared.cavity_parent_indices.shape == (prepared.cavity_point_count,)
    np.testing.assert_allclose(
        prepared.permanent_point_mep(2.0 * PERMANENT),
        2.0 * prepared.permanent_point_mep(PERMANENT),
        rtol=0.0,
        atol=3.0e-13,
    )
    fixed_permanent_energy = prepared.fixed_permanent_source_energy_ev(PERMANENT)
    assert fixed_permanent_energy == pytest.approx(
        prepared.solve(np.zeros_like(RADIAL)).polarization_energy_ev,
        rel=0.0,
        abs=2.0e-12,
    )
    state = prepared.solve(RADIAL)
    assert isinstance(state, SeparatedSourceDDXState)
    assert not state.permanent_source.flags.writeable
    assert not state.radial_source.flags.writeable
    assert not state.model_field.flags.writeable
    assert not state.energy_source_gradient.flags.writeable
    assert state.state_sha256 == prepared.solve(RADIAL).state_sha256
    with pytest.raises(ValueError, match="state_sha256"):
        replace(state, polarization_energy_ev=state.polarization_energy_ev + 1.0)

    direction = np.random.default_rng(4).normal(size=RADIAL.shape) * 0.01
    step = 2.0e-6
    finite_difference = (
        prepared.solve(RADIAL + step * direction).polarization_energy_ev
        - prepared.solve(RADIAL - step * direction).polarization_energy_ev
    ) / (2.0 * step)
    expected = MACE_POLAR_RADIAL_GTO_PAIRING.pair(
        direction, state.energy_source_gradient
    )
    assert finite_difference == pytest.approx(expected, abs=3.0e-9)
    assert np.linalg.norm(state.model_field - state.energy_source_gradient) > 1.0e-7

    # The quadratic continuum scalar obeys the exact direct-sum Euler/half-work
    # identity when each source is contracted with its own energy cotangent.
    # The MLIP-driving external-MEP receiver is intentionally not that object.
    assert state.energy_dual_work_ev == pytest.approx(
        2.0 * state.polarization_energy_ev, abs=3.0e-12
    )
    assert state.half_work_identity_residual_ev == pytest.approx(0.0, abs=3.0e-12)
    naive_native_endpoint_work = MACE_POLAR_RADIAL_GTO_PAIRING.pair(
        embed_atomic_l1_in_first_radial_channel(state.permanent_source)
        + state.radial_source,
        state.model_field,
    )
    assert abs(naive_native_endpoint_work - state.energy_dual_work_ev) > 1.0e-5

    permanent_direction = np.random.default_rng(14).normal(size=PERMANENT.shape) * 0.01
    permanent_finite_difference = (
        backend.prepare(POSITIONS, PERMANENT + step * permanent_direction)
        .solve(RADIAL)
        .polarization_energy_ev
        - backend.prepare(POSITIONS, PERMANENT - step * permanent_direction)
        .solve(RADIAL)
        .polarization_energy_ev
    ) / (2.0 * step)
    assert permanent_finite_difference == pytest.approx(
        np.vdot(state.permanent_energy_gradient, permanent_direction), abs=3.0e-9
    )

    cotangent = np.random.default_rng(5).normal(size=RADIAL.shape) * 0.01
    jvp = prepared.radial_jvp(direction)
    vjp = prepared.radial_vjp(cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction), abs=8.0e-10
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_general_problem_data_boundary_replays_existing_source_paths():
    prepared = _backend().prepare(POSITIONS, PERMANENT)
    assert prepared.symbols == ("H", "H", "H")
    permanent_psi, permanent_phi = prepared.bound_permanent_problem_data()
    replayed_psi, replayed_phi = prepared.point_problem_data(PERMANENT)
    np.testing.assert_allclose(replayed_psi, permanent_psi, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(replayed_phi, permanent_phi, rtol=0.0, atol=2.0e-13)

    radial_psi, radial_phi = prepared.radial_problem_data(RADIAL)
    solution = prepared.solve_problem_data(
        permanent_psi + radial_psi,
        permanent_phi + radial_phi,
    )
    assert isinstance(solution, SeparatedGeneralDDXSolution)
    reference = prepared.solve(RADIAL)
    assert solution.polarization_energy_ev == pytest.approx(
        reference.polarization_energy_ev, abs=2.0e-12
    )
    np.testing.assert_allclose(
        solution.model_field, reference.model_field, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        solution.radial_energy_gradient,
        reference.energy_source_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )

    radial_psi_matrix, radial_phi_matrix = prepared.radial_problem_matrices()
    point_psi_matrix, point_phi_matrix = prepared.point_problem_matrices()
    assert not radial_psi_matrix.flags.writeable
    assert not radial_phi_matrix.flags.writeable
    assert not point_psi_matrix.flags.writeable
    assert not point_phi_matrix.flags.writeable
    np.testing.assert_allclose(
        (point_psi_matrix @ PERMANENT.reshape(-1)).reshape(permanent_psi.shape),
        permanent_psi,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        point_phi_matrix @ PERMANENT.reshape(-1),
        permanent_phi,
        rtol=0.0,
        atol=2.0e-13,
    )

    generic_gradient = prepared.problem_data_energy_gradient(
        solution,
        psi_matrix=radial_psi_matrix,
        phi_matrix=radial_phi_matrix,
        source_shape=RADIAL.shape,
        name="radial replay",
    )
    np.testing.assert_allclose(
        generic_gradient,
        reference.energy_source_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )

    direction = np.random.default_rng(101).normal(size=RADIAL.shape) * 0.01
    direction_psi, direction_phi = prepared.radial_problem_data(direction)
    np.testing.assert_allclose(
        prepared.problem_data_model_field_jvp(direction_psi, direction_phi),
        prepared.radial_jvp(direction),
        rtol=0.0,
        atol=2.0e-12,
    )
    cotangent = np.random.default_rng(102).normal(size=RADIAL.shape) * 0.01
    np.testing.assert_allclose(
        prepared.problem_data_model_field_vjp(
            cotangent,
            source_psi_matrix=radial_psi_matrix,
            source_shape=RADIAL.shape,
            name="radial replay",
        ),
        prepared.radial_vjp(cotangent),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_separated_fixed_source_and_model_field_coordinate_vjps():
    backend = _backend()
    prepared = backend.prepare(POSITIONS, PERMANENT)
    state, energy_gradient = prepared.solve_with_fixed_source_energy_derivatives(RADIAL)
    field_cotangent = np.random.default_rng(51).normal(size=RADIAL.shape) * 0.01
    field_gradient = prepared.model_field_coordinate_vjp(RADIAL, field_cotangent)
    direction = np.random.default_rng(52).normal(size=POSITIONS.shape)
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    step = 2.0e-5

    plus = POSITIONS + step * direction
    minus = POSITIONS - step * direction
    plus_state = backend.prepare(plus, PERMANENT).solve(RADIAL)
    minus_state = backend.prepare(minus, PERMANENT).solve(RADIAL)
    energy_finite_difference = (
        plus_state.polarization_energy_ev - minus_state.polarization_energy_ev
    ) / (2.0 * step)
    field_finite_difference = np.vdot(
        field_cotangent,
        (plus_state.model_field - minus_state.model_field) / (2.0 * step),
    )
    assert np.vdot(energy_gradient, direction) == pytest.approx(
        energy_finite_difference, abs=5.0e-9
    )
    assert np.vdot(field_gradient, direction) == pytest.approx(
        field_finite_difference, abs=5.0e-9
    )
    assert state.state_sha256 == prepared.solve(RADIAL).state_sha256


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_zero_permanent_recovers_radial_energy_and_ledger_gradient_exactly():
    backend = _backend()
    prepared = backend.prepare(POSITIONS, np.zeros_like(PERMANENT))
    separated = prepared.solve(RADIAL)
    radial = backend.radial_backend.build_state(POSITIONS, RADIAL)
    assert separated.polarization_energy_ev == pytest.approx(
        radial.polarization_energy_ev, abs=2.0e-12
    )
    np.testing.assert_allclose(
        separated.energy_source_gradient,
        radial.reaction_field,
        atol=2.0e-12,
        rtol=0.0,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_permanent_point_kernel_is_not_silently_replaced_by_gaussian_alias():
    backend = _backend()
    point_state = backend.prepare(POSITIONS, PERMANENT).solve(np.zeros_like(RADIAL))
    gaussian_state = backend.prepare(POSITIONS, np.zeros_like(PERMANENT)).solve(
        embed_atomic_l1_in_first_radial_channel(PERMANENT)
    )
    assert (
        abs(point_state.polarization_energy_ev - gaussian_state.polarization_energy_ev)
        > 1.0e-7
    )
    with pytest.raises(AttributeError, match="immutable"):
        backend._radial_backend = object()
