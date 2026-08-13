from __future__ import annotations

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.profiles import (
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
    WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
    WATER_CPCM_590_CONFIGURATION_CONTRACT_ID,
    WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum import (
    ConjugateRadialGTOFixedTopologyCPCMBackend,
    build_water_radial_gto_cpcm_590_candidate,
    build_water_radial_gto_cpcm_1202_candidate,
    build_water_radial_gto_cpcm_backend,
)
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

SIX_POINT_SPHERE = np.asarray(
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
SWITCHING_CONSTANT = 4.84566077868
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.2, -0.1]])
SOURCE = np.asarray(
    [
        [0.30, 0.02, 0.02, 0.01, -0.01, 0.005, -0.008, 0.012],
        [-0.32, 0.00, 0.01, -0.02, 0.03, -0.006, 0.009, -0.004],
    ]
)


def backend():
    return ConjugateRadialGTOFixedTopologyCPCMBackend(
        ("H", "H"),
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        unit_sphere=SIX_POINT_SPHERE,
        switching_constant=SWITCHING_CONSTANT,
        runtime_version="unit-test-grid-v1",
    )


def _rotation() -> np.ndarray:
    axis = np.asarray([1.0, -2.0, 3.0], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _rotate_radial_blocks(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    rotated = np.array(values, dtype=float, copy=True)
    for raw_indices in ((2, 3, 4), (5, 6, 7)):
        raw = rotated[:, raw_indices]
        cartesian = raw[:, (2, 0, 1)]
        rotated_cartesian = cartesian @ rotation.T
        rotated[:, raw_indices] = rotated_cartesian[:, (1, 2, 0)]
    return rotated


def test_radial_cpcm_is_one_stationary_half_coupling_with_closed_capabilities():
    provider = backend()
    state = provider.build_state(POSITIONS, SOURCE)
    assert provider.scalar_id == OPERATIONAL_CPCM_ELECTROSTATIC_V1
    assert provider.coupling_id == MACE_POLAR_RADIAL_GTO_COUPLING_ID
    assert provider.source_space is MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    assert provider.field_space is MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    assert (
        provider.configuration_contract_id
        == UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID
    )
    assert provider.capabilities.enabled_tiers == ()
    assert state.capabilities.enabled_tiers == ()
    assert state.polarization_energy_ev == pytest.approx(
        0.5 * MACE_POLAR_RADIAL_GTO_PAIRING.pair(state.source, state.reaction_field),
        abs=2e-13,
    )
    assert state.polarization_energy_ev == pytest.approx(
        0.5 * np.dot(state.surface_potential_ev, state.surface_charge),
        abs=2e-13,
    )
    assert state.polarization_energy_ev == pytest.approx(
        state.polarization_energy_hartree * HARTREE_TO_EV,
        abs=2e-13,
    )
    for array in (
        state.source,
        state.reaction_field,
        state.surface_potential_ev,
        state.surface_charge,
    ):
        assert not array.flags.writeable


def test_radial_cpcm_source_jvp_vjp_and_energy_derivative():
    provider = backend()
    rng = np.random.default_rng(20260813)
    direction = rng.normal(scale=0.1, size=SOURCE.shape)
    cotangent = rng.normal(scale=0.1, size=SOURCE.shape)
    jvp = provider.source_jvp(POSITIONS, SOURCE, direction)
    vjp = provider.source_vjp(POSITIONS, SOURCE, cotangent)
    assert np.vdot(cotangent, jvp) == pytest.approx(np.vdot(vjp, direction), abs=3e-12)
    step = 1e-6
    field_fd = (
        provider.evaluate_field(POSITIONS, SOURCE + step * direction)
        - provider.evaluate_field(POSITIONS, SOURCE - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(jvp, field_fd, rtol=2e-9, atol=2e-9)
    energy_fd = (
        provider.energy(POSITIONS, SOURCE + step * direction)
        - provider.energy(POSITIONS, SOURCE - step * direction)
    ) / (2 * step)
    analytic = np.vdot(provider.evaluate_field(POSITIONS, SOURCE), direction)
    assert analytic == pytest.approx(energy_fd, abs=2e-8)
    dense = provider.reaction_field_matrix(POSITIONS)
    np.testing.assert_allclose(
        (dense @ SOURCE.reshape(-1)).reshape(SOURCE.shape),
        provider.evaluate_field(POSITIONS, SOURCE),
        atol=3e-13,
        rtol=3e-13,
    )
    np.testing.assert_allclose(dense, dense.T, atol=2e-14, rtol=0.0)
    assert not dense.flags.writeable


def test_radial_cpcm_complete_bilinear_and_scalar_coordinate_vjps():
    provider = backend()
    cotangent = np.asarray(
        [
            [0.12, -0.03, 0.01, 0.02, -0.04, 0.03, -0.01, 0.02],
            [-0.20, 0.04, -0.02, 0.05, 0.01, -0.02, 0.03, -0.01],
        ]
    )
    analytic = provider.coordinate_vjp(POSITIONS, SOURCE, cotangent).reshape(2, 3)
    scalar_analytic = provider.coordinate_vjp(POSITIONS, SOURCE, 0.5 * SOURCE).reshape(
        2, 3
    )
    bilinear_fd = np.empty((2, 3))
    scalar_fd = np.empty((2, 3))
    step = 1e-5
    for atom in range(2):
        for axis in range(3):
            plus = POSITIONS.copy()
            minus = POSITIONS.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            bilinear_fd[atom, axis] = (
                np.vdot(cotangent, provider.evaluate_field(plus, SOURCE))
                - np.vdot(cotangent, provider.evaluate_field(minus, SOURCE))
            ) / (2 * step)
            scalar_fd[atom, axis] = (
                provider.energy(plus, SOURCE) - provider.energy(minus, SOURCE)
            ) / (2 * step)
    np.testing.assert_allclose(analytic, bilinear_fd, rtol=0.0, atol=1.2e-7)
    np.testing.assert_allclose(scalar_analytic, scalar_fd, rtol=0.0, atol=8e-8)
    np.testing.assert_allclose(analytic.sum(axis=0), np.zeros(3), atol=5e-11)
    np.testing.assert_allclose(scalar_analytic.sum(axis=0), np.zeros(3), atol=5e-11)


def test_radial_cpcm_geometry_cache_is_exact_single_entry_and_source_independent(
    monkeypatch,
):
    provider = backend()
    calls = 0
    original = provider._legacy_response

    def counted(positions):
        nonlocal calls
        calls += 1
        return original(positions)

    monkeypatch.setattr(
        type(provider),
        "_legacy_response",
        lambda self, positions: counted(positions),
    )
    first = provider.build_state(POSITIONS, SOURCE)
    second = provider.build_state(POSITIONS.copy(), 0.5 * SOURCE)
    assert calls == 1
    assert first.surface.state_hash == second.surface.state_hash

    displaced = POSITIONS.copy()
    displaced[0, 0] = np.nextafter(displaced[0, 0], np.inf)
    displaced_state = provider.build_state(displaced, SOURCE)
    assert calls == 2
    assert displaced_state.surface.state_hash != first.surface.state_hash

    # Single-entry eviction must rebuild the original geometry rather than
    # retaining an unbounded PES-path cache of dense surface matrices.
    replay = provider.build_state(POSITIONS, SOURCE)
    assert calls == 3
    assert replay.state_hash == first.state_hash


def test_radial_cpcm_cold_replay_and_configuration_are_content_bound():
    first = backend()
    second = backend()
    first_state = first.build_state(POSITIONS, SOURCE)
    second_state = second.build_state(POSITIONS.copy(), SOURCE.copy())
    assert first.configuration_sha256() == second.configuration_sha256()
    assert first.provenance_sha256 == second.provenance_sha256
    assert first_state.state_hash == second_state.state_hash
    np.testing.assert_array_equal(
        first_state.reaction_field, second_state.reaction_field
    )
    with pytest.raises(AttributeError, match="immutable"):
        first.dielectric = 2.0
    bad = SOURCE.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        first.build_state(POSITIONS, bad)


def test_radial_cpcm_translation_and_corotated_quadrature_covariance():
    provider = backend()
    cotangent = np.asarray(
        [
            [0.12, -0.03, 0.01, 0.02, -0.04, 0.03, -0.01, 0.02],
            [-0.20, 0.04, -0.02, 0.05, 0.01, -0.02, 0.03, -0.01],
        ]
    )
    base_energy = provider.energy(POSITIONS, SOURCE)
    base_field = provider.evaluate_field(POSITIONS, SOURCE)
    base_coordinate_vjp = provider.coordinate_vjp(POSITIONS, SOURCE, cotangent).reshape(
        2, 3
    )

    shift = np.asarray([1.7, -0.8, 0.5])
    shifted_positions = POSITIONS + shift
    assert provider.energy(shifted_positions, SOURCE) == pytest.approx(
        base_energy, abs=2e-13
    )
    np.testing.assert_allclose(
        provider.evaluate_field(shifted_positions, SOURCE),
        base_field,
        rtol=5e-13,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        provider.coordinate_vjp(shifted_positions, SOURCE, cotangent).reshape(2, 3),
        base_coordinate_vjp,
        rtol=8e-12,
        atol=8e-11,
    )

    rotation = _rotation()
    rotated_grid = SIX_POINT_SPHERE.copy()
    rotated_grid[:, :3] = rotated_grid[:, :3] @ rotation.T
    rotated_provider = ConjugateRadialGTOFixedTopologyCPCMBackend(
        ("H", "H"),
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        unit_sphere=rotated_grid,
        switching_constant=SWITCHING_CONSTANT,
        runtime_version="unit-test-grid-corotated-v1",
    )
    rotated_positions = POSITIONS @ rotation.T
    rotated_source = _rotate_radial_blocks(SOURCE, rotation)
    rotated_cotangent = _rotate_radial_blocks(cotangent, rotation)
    assert rotated_provider.energy(rotated_positions, rotated_source) == pytest.approx(
        base_energy, abs=3e-12
    )
    np.testing.assert_allclose(
        rotated_provider.evaluate_field(rotated_positions, rotated_source),
        _rotate_radial_blocks(base_field, rotation),
        rtol=2e-11,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        rotated_provider.coordinate_vjp(
            rotated_positions, rotated_source, rotated_cotangent
        ).reshape(2, 3),
        base_coordinate_vjp @ rotation.T,
        rtol=3e-10,
        atol=3e-9,
    )


def test_water_configuration_contract_rejects_parameter_forgery_before_runtime():
    symbols = ("O", "H", "H")
    radii = smd_water_coulomb_radii(symbols)
    common = dict(
        symbols=symbols,
        cavity_radii_angstrom=radii,
        dielectric=78.39,
        lebedev_order=23,
        configuration_contract_id=WATER_CPCM_194_CONFIGURATION_CONTRACT_ID,
    )
    with pytest.raises(ValueError, match="dielectric=78.39 exactly"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(**{**common, "dielectric": 80.0})
    with pytest.raises(ValueError, match="Lebedev order 23"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(**{**common, "lebedev_order": 17})
    with pytest.raises(ValueError, match="exact versioned SMD-water"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(
            **{**common, "cavity_radii_angstrom": radii + 1.0e-8}
        )
    with pytest.raises(ValueError, match="forbids injected"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(
            **common,
            unit_sphere=SIX_POINT_SPHERE,
            switching_constant=SWITCHING_CONSTANT,
        )
    with pytest.raises(ValueError, match="Unknown radial C-PCM"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(
            ("H",),
            np.asarray([1.2]),
            dielectric=78.39,
            unit_sphere=SIX_POINT_SPHERE,
            switching_constant=SWITCHING_CONSTANT,
            configuration_contract_id="forged-water-contract",
        )


def test_water_factory_binds_all_profile_parameters_when_pyscf_is_available():
    pytest.importorskip("pyscf")
    provider = build_water_radial_gto_cpcm_backend(("O", "H", "H"))
    assert (
        provider.configuration_contract_id == WATER_CPCM_194_CONFIGURATION_CONTRACT_ID
    )
    assert provider.dielectric == 78.39
    assert provider.surface_provider.lebedev_order == 23
    assert provider.surface_provider.unit_sphere.shape == (194, 4)
    np.testing.assert_array_equal(
        provider.cavity_radii_angstrom,
        smd_water_coulomb_radii(("O", "H", "H")),
    )


def test_high_order_water_factory_binds_590_point_candidate_and_rejects_forgery():
    pytest.importorskip("pyscf")
    symbols = ("O", "H", "H")
    provider = build_water_radial_gto_cpcm_590_candidate(symbols)
    assert (
        provider.configuration_contract_id == WATER_CPCM_590_CONFIGURATION_CONTRACT_ID
    )
    assert provider.dielectric == 78.39
    assert provider.surface_provider.lebedev_order == 41
    assert provider.surface_provider.unit_sphere.shape == (590, 4)
    with pytest.raises(ValueError, match="Lebedev order 41"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(
            symbols,
            smd_water_coulomb_radii(symbols),
            dielectric=78.39,
            lebedev_order=35,
            configuration_contract_id=WATER_CPCM_590_CONFIGURATION_CONTRACT_ID,
        )


def test_1202_water_factory_binds_distinct_candidate_and_rejects_forgery():
    pytest.importorskip("pyscf")
    symbols = ("O", "H", "H")
    provider = build_water_radial_gto_cpcm_1202_candidate(symbols)
    assert (
        provider.configuration_contract_id == WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID
    )
    assert provider.dielectric == 78.39
    assert provider.surface_provider.lebedev_order == 59
    assert provider.surface_provider.unit_sphere.shape == (1202, 4)
    with pytest.raises(ValueError, match="Lebedev order 59"):
        ConjugateRadialGTOFixedTopologyCPCMBackend(
            symbols,
            smd_water_coulomb_radii(symbols),
            dielectric=78.39,
            lebedev_order=41,
            configuration_contract_id=WATER_CPCM_1202_CONFIGURATION_CONTRACT_ID,
        )
