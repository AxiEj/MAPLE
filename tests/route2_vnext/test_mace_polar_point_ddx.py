from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from maple.solvation.continuum.mace_polar_point_ddx import (
    MACE_POLAR_POINT_DDX_PROFILE_ID,
    MACE_POLAR_POINT_DDX_PROVIDER_ID,
    MACE_POLAR_POINT_DDX_SCALAR_ID,
    MACEPolarPointEmbeddedDDXBackend,
)
from maple.solvation.continuum.separated_source_ddx import (
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING


POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [1.43, 0.10, -0.20], [0.20, 1.10, 0.30]]
)
POINT_SOURCE = np.asarray(
    [
        [0.12, 0.02, 0.01, -0.01],
        [-0.08, -0.01, 0.03, 0.02],
        [-0.04, 0.004, -0.005, 0.006],
    ]
)
SOURCE = embed_atomic_l1_in_first_radial_channel(POINT_SOURCE)


def _backend() -> MACEPolarPointEmbeddedDDXBackend:
    return MACEPolarPointEmbeddedDDXBackend(
        ("H", "H", "H"),
        np.asarray([1.5, 1.6, 1.2]),
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_point_embedding_is_exactly_the_separated_point_branch():
    backend = _backend()
    state = backend.build_state(POSITIONS, SOURCE)
    reference_backend = SeparatedSourceDDXBackend(
        ("H", "H", "H"),
        np.asarray([1.5, 1.6, 1.2]),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )
    reference = reference_backend.prepare(POSITIONS, POINT_SOURCE).solve(
        np.zeros_like(SOURCE)
    )
    assert backend.provider_id == MACE_POLAR_POINT_DDX_PROVIDER_ID
    assert backend.continuum_profile_id == MACE_POLAR_POINT_DDX_PROFILE_ID
    assert backend.scalar_id == MACE_POLAR_POINT_DDX_SCALAR_ID
    assert state.polarization_energy_ev == pytest.approx(
        reference.polarization_energy_ev, abs=2.0e-14
    )
    np.testing.assert_allclose(
        state.reaction_field,
        embed_atomic_l1_in_first_radial_channel(
            reference.permanent_energy_gradient
        ),
        atol=2.0e-13,
        rtol=0.0,
    )
    assert state.polarization_energy_ev == pytest.approx(
        0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(
            state.source, state.reaction_field
        ),
        abs=2.0e-12,
    )
    assert not state.source.flags.writeable
    assert not state.reaction_field.flags.writeable
    assert state.state_hash == backend.build_state(POSITIONS, SOURCE).state_hash


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_point_embedding_source_and_coordinate_derivatives_close_the_scalar():
    backend = _backend()
    rng = np.random.default_rng(20260816)
    point_direction = rng.normal(size=POINT_SOURCE.shape)
    direction = embed_atomic_l1_in_first_radial_channel(point_direction)
    field = backend.source_jvp(POSITIONS, SOURCE, direction)
    cotangent = embed_atomic_l1_in_first_radial_channel(
        rng.normal(size=POINT_SOURCE.shape)
    )
    vjp = backend.source_vjp(POSITIONS, SOURCE, cotangent)
    assert np.vdot(cotangent, field) == pytest.approx(
        np.vdot(vjp, direction), abs=8.0e-10
    )
    source_step = 2.0e-6
    source_fd = (
        backend.energy(POSITIONS, SOURCE + source_step * direction)
        - backend.energy(POSITIONS, SOURCE - source_step * direction)
    ) / (2.0 * source_step)
    assert source_fd == pytest.approx(
        MACE_POLAR_RADIAL_GTO_PAIRING.pair(
            direction, backend.evaluate_field(POSITIONS, SOURCE)
        ),
        abs=2.0e-9,
    )

    state, gradient = backend.build_state_with_fixed_source_coordinate_gradient(
        POSITIONS, SOURCE
    )
    assert state.state_hash == backend.build_state(POSITIONS, SOURCE).state_hash
    coordinate_direction = rng.normal(size=POSITIONS.shape)
    coordinate_direction -= np.mean(coordinate_direction, axis=0, keepdims=True)
    coordinate_direction /= np.linalg.norm(coordinate_direction)
    projected = float(np.vdot(gradient, coordinate_direction))
    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        fd = (
            backend.energy(POSITIONS + step * coordinate_direction, SOURCE)
            - backend.energy(POSITIONS - step * coordinate_direction, SOURCE)
        ) / (2.0 * step)
        errors.append(abs(fd - projected))
    assert errors[-1] < 2.0e-8
    assert errors[-1] < errors[0] / 6.0
    np.testing.assert_allclose(np.sum(gradient, axis=0), 0.0, atol=3.0e-11)


def test_point_embedding_rejects_a_nonzero_unused_radial_source_block():
    backend = _backend()
    invalid = SOURCE.copy()
    invalid[0, 1] = 1.0e-12
    with pytest.raises(ValueError, match="zero second radial source block"):
        backend.build_state(POSITIONS, invalid)

