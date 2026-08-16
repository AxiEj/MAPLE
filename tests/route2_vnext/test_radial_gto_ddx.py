from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1,
    PROFILE_REGISTRY,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1,
    DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1,
)
from maple.solvation.continuum import (
    DDX_CAVITY_PROFILE_ID,
    DDX_PCM_PROFILE_ID,
    DDX_RADIAL_PROVIDER_ID,
    DDX_WATER_194_CONFIGURATION_CONTRACT_ID,
    RadialGTODDXBackend,
    RadialGTODDXState,
    build_water_radial_gto_ddpcm_194_candidate,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.function.calculator.extra_correction.implicit.pyddx_pcm_response import (
    mace_polar_density_to_pyddx_multipoles,
)
from maple.solvation.release import rotate_radial_gto_blocks, symmetry_panel_rotations

POSITIONS = np.asarray([[0.0, 0.0, 0.0], [1.43, 0.10, -0.20], [0.20, 1.10, 0.30]])
SOURCE = np.asarray(
    [
        [0.12, -0.03, 0.02, 0.01, -0.01, 0.004, 0.008, -0.003],
        [-0.08, 0.01, -0.01, 0.03, 0.02, -0.006, 0.002, 0.009],
        [-0.02, 0.00, 0.004, -0.005, 0.006, 0.001, -0.003, 0.002],
    ]
)


def _backend() -> RadialGTODDXBackend:
    return RadialGTODDXBackend(
        ("H", "H", "H"),
        np.asarray([1.5, 1.6, 1.2]),
        continuum_model="pcm",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )


def _direction(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def test_import_is_dependency_light_without_pyddx():
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pyddx' or name.startswith('pyddx.'):
        raise AssertionError('pyddx imported eagerly')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import maple.solvation
from maple.solvation.continuum.radial_gto_ddx import RadialGTODDXBackend
print(RadialGTODDXBackend.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "RadialGTODDXBackend"


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_joint_psi_phi_map_closes_scalar_source_and_adjoint():
    backend = _backend()
    state = backend.build_state(POSITIONS, SOURCE)
    assert isinstance(state, RadialGTODDXState)
    assert backend.provider_id == DDX_RADIAL_PROVIDER_ID
    assert backend.scalar_id == DIAGNOSTIC_DDX_DDPCM_RADIAL_GTO_ELECTROSTATIC_V1
    assert backend.continuum_profile_id == DDX_PCM_PROFILE_ID
    assert backend.cavity_profile_id == DDX_CAVITY_PROFILE_ID
    assert isinstance(backend.capabilities, CapabilityStatus)
    assert backend.capabilities.enabled_tiers == ()
    assert state.capabilities.enabled_tiers == ()
    assert state.polarization_energy_ev == pytest.approx(
        0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(state.source, state.reaction_field),
        abs=2.0e-12,
    )
    assert not state.source.flags.writeable
    assert not state.reaction_field.flags.writeable
    assert len(state.cavity_topology_sha256) == 64
    assert state.state_hash == backend.build_state(POSITIONS, SOURCE).state_hash
    displaced_state = backend.build_state(
        POSITIONS + np.asarray([[0.0, 0.0, 0.0], [1.0e-5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        SOURCE,
    )
    assert displaced_state.geometry_sha256 != state.geometry_sha256
    assert displaced_state.state_hash != state.state_hash
    with pytest.raises(ValueError, match="state hash"):
        replace(state, provenance_sha256="0" * 64)
    with pytest.raises(ValueError, match="state hash"):
        replace(state, geometry_sha256="0" * 64)
    with pytest.raises(ValueError, match="continuum profile identity"):
        replace(state, continuum_model="cosmo")
    with pytest.raises(ValueError, match="capabilities remain closed"):
        replace(state, capabilities=CapabilityStatus(energy=True))
    periodic = Atoms("H3", positions=POSITIONS, pbc=True)
    with pytest.raises(ValueError, match="only nonperiodic"):
        backend.energy(periodic, SOURCE)

    direction = _direction(1, SOURCE.shape)
    cotangent = _direction(2, SOURCE.shape)
    jvp = backend.source_jvp(POSITIONS, SOURCE, direction)
    vjp = backend.source_vjp(POSITIONS, SOURCE, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(
        np.vdot(vjp, direction), abs=8.0e-10
    )
    step = 2.0e-6
    finite_difference = (
        backend.energy(POSITIONS, SOURCE + step * direction)
        - backend.energy(POSITIONS, SOURCE - step * direction)
    ) / (2.0 * step)
    assert finite_difference == pytest.approx(
        MACE_POLAR_RADIAL_GTO_PAIRING.pair(direction, state.reaction_field),
        abs=2.0e-9,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_fixed_source_coordinate_gradient_is_the_same_ddx_scalar_derivative():
    backend = _backend()
    state, analytic = backend.build_state_with_fixed_source_coordinate_gradient(
        POSITIONS, SOURCE
    )
    np.testing.assert_allclose(
        analytic,
        backend.fixed_source_coordinate_gradient(POSITIONS, SOURCE),
        atol=0.0,
        rtol=0.0,
    )
    direction = _direction(27, POSITIONS.shape)
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    projected = float(np.vdot(analytic, direction))
    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        finite_difference = (
            backend.energy(POSITIONS + step * direction, SOURCE)
            - backend.energy(POSITIONS - step * direction, SOURCE)
        ) / (2.0 * step)
        errors.append(abs(finite_difference - projected))
    assert errors[-1] < 2.0e-8
    assert errors[-1] < errors[0] / 6.0
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=3.0e-11)
    assert (
        state.cavity_topology_sha256
        == backend.build_state(
            POSITIONS + 1.0e-5 * direction, SOURCE
        ).cavity_topology_sha256
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_second_radial_block_is_not_collapsed_to_four_channel_point_source():
    backend = _backend()
    second_only = np.zeros_like(SOURCE)
    second_only[:, (1, 5, 6, 7)] = SOURCE[:, (1, 5, 6, 7)]
    first_width_alias = np.zeros_like(SOURCE)
    first_width_alias[:, (0, 2, 3, 4)] = SOURCE[:, (1, 5, 6, 7)]
    second_field = backend.evaluate_field(POSITIONS, second_only)
    aliased_field = backend.evaluate_field(POSITIONS, first_width_alias)
    assert np.linalg.norm(second_field) > 1.0e-8
    assert np.linalg.norm(second_field - aliased_field) > 1.0e-7


def test_ddx_local_multipoles_reuse_the_independently_tested_raw_l1_convention():
    coefficients = SOURCE[:, (0, 2, 3, 4)] + SOURCE[:, (1, 5, 6, 7)]
    np.testing.assert_allclose(
        RadialGTODDXBackend._ddx_total_multipoles(SOURCE),
        mace_polar_density_to_pyddx_multipoles(coefficients),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_coordinate_vjp_is_derivative_of_the_same_bilinear_map():
    backend = _backend()
    cotangent = _direction(3, SOURCE.shape) * 0.02
    analytic = backend.coordinate_vjp(POSITIONS, SOURCE, cotangent).reshape(
        POSITIONS.shape
    )
    direction = _direction(4, POSITIONS.shape)
    direction -= np.mean(direction, axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)
    projected = float(np.vdot(analytic, direction))
    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        plus = POSITIONS + step * direction
        minus = POSITIONS - step * direction
        finite_difference = (
            np.vdot(cotangent, backend.evaluate_field(plus, SOURCE))
            - np.vdot(cotangent, backend.evaluate_field(minus, SOURCE))
        ) / (2.0 * step)
        errors.append(abs(finite_difference - projected))
    assert errors[-1] < 2.0e-10
    assert errors[-1] < errors[0] / 8.0
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-12)


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_water_factory_is_frozen_and_fail_closed():
    backend = build_water_radial_gto_ddpcm_194_candidate(("O", "H", "H"))
    profile = PROFILE_REGISTRY[DIAGNOSTIC_DDX_DDPCM_194_RADIAL_GTO_PROFILE_V1]
    assert backend.configuration_contract_id == DDX_WATER_194_CONFIGURATION_CONTRACT_ID
    assert backend.continuum_profile_id == profile.continuum_profile
    assert backend.cavity_profile_id == profile.cavity_profile
    assert backend.scalar_id == profile.scalar_id
    assert backend.coupling_id == profile.coupling_id
    assert backend.configuration_contract_id == (
        profile.continuum_configuration_contract_id
    )
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    with pytest.raises(AttributeError, match="immutable"):
        backend._lmax = 9
    with pytest.raises(ValueError, match="exact frozen settings"):
        RadialGTODDXBackend(
            ("O", "H", "H"),
            backend.cavity_radii_angstrom,
            continuum_model="pcm",
            dielectric=78.39,
            lmax=9,
            n_lebedev=194,
            configuration_contract_id=DDX_WATER_194_CONFIGURATION_CONTRACT_ID,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_finite_grid_rotation_is_measured_and_not_misrepresented_as_structural():
    atoms = Atoms("H3", positions=POSITIONS)
    backend = _backend()
    base_energy = backend.energy(atoms, SOURCE)
    base_field = backend.evaluate_field(atoms, SOURCE)
    rotation = symmetry_panel_rotations("radial-ddx-h3")[0]
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated_source = rotate_radial_gto_blocks(SOURCE, rotation)
    rotated_energy = backend.energy(rotated, rotated_source)
    rotated_field = backend.evaluate_field(rotated, rotated_source)
    energy_drift = abs(rotated_energy - base_energy)
    field_drift = np.linalg.norm(
        rotated_field - rotate_radial_gto_blocks(base_field, rotation)
    )
    assert backend.structurally_rotation_equivariant is False
    assert backend.achieved_algebraic_residual_available is False
    assert energy_drift > 1.0e-10
    assert field_drift > 1.0e-9
    assert dict(backend.runtime_provenance)["rotation_status"] == (
        "finite-grid-not-structurally-equivariant"
    )
    assert dict(backend.runtime_provenance)["inner_solve_evidence"] == (
        "requested-tolerance-only"
    )
    assert len(dict(backend.runtime_provenance)["ddx_source_commit"]) == 40


@pytest.mark.skipif(
    importlib.util.find_spec("pyddx") is None,
    reason="optional pyddx==0.8.0 runtime is unavailable",
)
def test_ddcosmo_is_separate_and_same_scalar_but_not_a_registered_release_profile():
    pcm = _backend()
    cosmo = RadialGTODDXBackend(
        ("H", "H", "H"),
        np.asarray([1.5, 1.6, 1.2]),
        continuum_model="cosmo",
        dielectric=78.39,
        lmax=8,
        n_lebedev=194,
        solver_tolerance=1.0e-12,
    )
    assert cosmo.continuum_profile_id != pcm.continuum_profile_id
    assert cosmo.scalar_id == DIAGNOSTIC_DDX_DDCOSMO_RADIAL_GTO_ELECTROSTATIC_V1
    assert cosmo.scalar_id != pcm.scalar_id
    assert cosmo.configuration_sha256() != pcm.configuration_sha256()
    state = cosmo.build_state(POSITIONS, SOURCE)
    assert state.scalar_id == cosmo.scalar_id
    assert state.polarization_energy_ev == pytest.approx(
        0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(state.source, state.reaction_field),
        abs=2.0e-12,
    )
    assert all(
        profile.continuum_profile != cosmo.continuum_profile_id
        for profile in PROFILE_REGISTRY.values()
    )
