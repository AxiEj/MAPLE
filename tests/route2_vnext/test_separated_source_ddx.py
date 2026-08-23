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
