from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest
from ase.units import Hartree as ASE_HARTREE_TO_EV

from maple.function.calculator.extra_correction.implicit.route2_fc_aswig_cpcm import (
    FixedTopologyAmplitudeSWIGCPCMResponse,
)
from maple.solvation.continuum import FixedTopologyCPCMBackend
from maple.solvation.coupling import ATOMIC_L1_PAIRING
from maple.solvation.coupling.energy import reciprocal_linear_half_coupling
from maple.solvation.coupling.energy import OperationalElectrostaticScalar
from maple.solvation.coupling.fixed_point import FixedPointOptions, solve_fixed_point
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.coupling.state_equation import ReducedStateEquation
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
    get_scalar_definition,
)
from maple.solvation.api.profiles import (
    DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
    LOCAL_JET_DIAGNOSTIC_COUPLING_ID,
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
    PROFILE_REGISTRY,
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
AUTHORITATIVE_HARTREE_TO_EV = 27.211386245988
LEGACY_TO_AUTHORITATIVE_SCALE = AUTHORITATIVE_HARTREE_TO_EV / ASE_HARTREE_TO_EV
POSITIONS = np.asarray([[0.0, 0.0, 0.0], [2.5, 0.2, -0.1]])
SOURCE = np.asarray([[0.3, 0.02, 0.01, -0.01], [-0.3, 0.01, -0.02, 0.03]])


def backend() -> FixedTopologyCPCMBackend:
    return FixedTopologyCPCMBackend(
        ("H", "H"),
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        unit_sphere=SIX_POINT_SPHERE,
        switching_constant=SWITCHING_CONSTANT,
        runtime_version="unit-test-grid-v1",
    )


def legacy(positions=POSITIONS):
    return FixedTopologyAmplitudeSWIGCPCMResponse(
        ("H", "H"),
        positions,
        np.asarray([1.1, 1.1]),
        dielectric=78.39,
        _unit_sphere=SIX_POINT_SPHERE,
        _switching_constant=SWITCHING_CONSTANT,
        _runtime_version="unit-test-grid-v1",
    ).reaction_field_linear_map(positions)


def test_adapter_is_exactly_the_legacy_same_scalar_not_a_second_cpcm():
    provider = backend()
    state = provider.build_state(POSITIONS, SOURCE)
    legacy_map = legacy()

    np.testing.assert_allclose(
        state.reaction_field,
        legacy_map.apply_scf(SOURCE) * LEGACY_TO_AUTHORITATIVE_SCALE,
        rtol=2e-15,
        atol=0,
    )
    assert state.polarization_energy_hartree == pytest.approx(
        legacy_map.scf_polarization_energy_hartree(SOURCE), rel=0, abs=0
    )
    assert state.polarization_energy_ev == pytest.approx(
        0.5 * ATOMIC_L1_PAIRING.pair(SOURCE, state.reaction_field),
        rel=1e-13,
        abs=1e-13,
    )
    assert not state.source.flags.writeable
    assert not state.reaction_field.flags.writeable
    assert provider.configuration_sha256()
    assert provider.continuum_profile_id == "fixed-topology-linear-reciprocal-cpcm-v1"
    assert provider.cavity_profile_id == "fixed-topology-amplitude-swig-v1"
    assert provider.capabilities.enabled_tiers == ()
    runtime = dict(provider.runtime_provenance)
    assert runtime["continuum_oracle"] == "torch-parity-available-not-release-admitted"
    assert runtime["surface_primitive_oracle"] == "missing-not-admitted"
    assert provider.coupling_id == LOCAL_JET_DIAGNOSTIC_COUPLING_ID
    assert provider.scalar_id == DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1
    diagnostic_profile = PROFILE_REGISTRY[
        DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1
    ]
    assert diagnostic_profile.scalar_id == provider.scalar_id
    assert diagnostic_profile.coupling_id == provider.coupling_id
    assert (
        provider.coupling_id
        != PROFILE_REGISTRY[OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1].coupling_id
    )
    assert state.polarization_energy_ev == pytest.approx(
        state.polarization_energy_hartree * AUTHORITATIVE_HARTREE_TO_EV,
        rel=2e-15,
        abs=1e-15,
    )
    coupling = reciprocal_linear_half_coupling(provider, POSITIONS, SOURCE)
    assert coupling.energy == pytest.approx(
        state.polarization_energy_ev, rel=1e-13, abs=1e-13
    )
    np.testing.assert_allclose(
        coupling.total_source_gradient_array(),
        ATOMIC_L1_PAIRING.field_to_source_dual(state.reaction_field),
        rtol=2e-13,
        atol=2e-13,
    )

    # A production adapter is not admission evidence.  Gate 4 remains closed.
    scalar = get_scalar_definition(provider.scalar_id)
    assert scalar.enabled is False
    assert scalar.admitted_capabilities.enabled_tiers == ()


def test_local_jet_backend_has_a_separate_disabled_scalar_and_profile():
    class Electronic:
        provider_id = "test.local-jet-electronic.v1"
        model_profile_id = "mace-polar-route2-source-field-contract-v1"
        coupling_id = LOCAL_JET_DIAGNOSTIC_COUPLING_ID
        provenance_sha256 = "e" * 64
        source_space = ATOMIC_L1_SOURCE_SPACE
        field_space = ATOMIC_L1_FIELD_DUAL_SPACE

        def configuration_sha256(self):
            return hashlib.sha256(b"test-local-jet-electronic-v1").hexdigest()

        def evaluate_source(self, geometry, field):
            del geometry
            bias = np.asarray([[0.1, 0.02, 0.0, 0.0], [-0.1, -0.01, 0.0, 0.0]])
            return bias + 0.01 * np.asarray(field)

        def field_jvp(self, geometry, field, direction):
            del geometry, field
            return 0.01 * np.asarray(direction)

        def field_vjp(self, geometry, field, cotangent):
            del geometry, field
            return 0.01 * np.asarray(cotangent)

        def coordinate_vjp(self, geometry, field, cotangent):
            del field, cotangent
            return np.zeros(np.asarray(geometry).size)

    class Vacuum:
        provider_id = "test.local-jet-vacuum.v1"
        model_profile_id = "mace-polar-route2-source-field-contract-v1"
        provenance_sha256 = "a" * 64

        def configuration_sha256(self):
            return hashlib.sha256(b"test-local-jet-vacuum-v1").hexdigest()

        def evaluate_energy(self, geometry):
            del geometry
            return 0.0

        def coordinate_gradient(self, geometry):
            return np.zeros(np.asarray(geometry).size)

    provider = backend()
    equation = ReducedStateEquation(
        AffineChargeCoordinates(2, total_charge=0.0), Electronic(), provider
    )
    scalar = OperationalElectrostaticScalar(
        equation,
        Vacuum(),
        scalar_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_V1,
        profile_id=DIAGNOSTIC_LOCAL_JET_CPCM_ELECTROSTATIC_PROFILE_V1,
    )
    state = solve_fixed_point(
        equation,
        POSITIONS,
        scalar_id=scalar.scalar_id,
        profile_id=scalar.profile_id,
        scalar_binding=scalar,
        root_context_id="unit-test/local-jet-diagnostic",
        options=FixedPointOptions(tolerance=1e-12, max_iterations=30),
    )
    assert state.converged
    assert np.isfinite(provider.energy(POSITIONS, state.source_array()))
    profile = PROFILE_REGISTRY[scalar.profile_id]
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    with pytest.raises(ValueError, match="Continuum scalar ID"):
        OperationalElectrostaticScalar(
            equation,
            Vacuum(),
            profile_id=OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
        )


def test_surface_snapshot_retains_all_candidates_and_fixed_owner_order():
    provider = backend().surface_provider
    buried = provider.build_state(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    exposed = provider.build_state(np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]))

    assert buried.candidate_count == exposed.candidate_count == 12
    np.testing.assert_array_equal(buried.parent_atom_indices, np.repeat([0, 1], 6))
    np.testing.assert_array_equal(
        buried.parent_atom_indices, exposed.parent_atom_indices
    )
    assert buried.topology_hash == exposed.topology_hash
    assert buried.state_hash != exposed.state_hash
    assert np.count_nonzero(buried.exposure_amplitudes == 0.0) >= 2
    assert np.count_nonzero(buried.effective_areas_bohr2 == 0.0) >= 2
    for array in (
        buried.reference_positions_bohr,
        buried.parent_atom_indices,
        buried.surface_points_bohr,
        buried.exposure_amplitudes,
    ):
        assert not array.flags.writeable


def test_linear_jvp_vjp_reciprocity_and_source_energy_derivative():
    provider = backend()
    direction = np.asarray([[0.11, -0.04, 0.03, 0.02], [-0.07, 0.01, -0.05, 0.06]])
    cotangent = np.asarray([[-0.2, 0.03, 0.04, -0.01], [0.13, -0.02, 0.05, 0.07]])
    field = provider.field(POSITIONS, SOURCE)
    jvp = provider.source_jvp(POSITIONS, SOURCE, direction)
    vjp = provider.source_vjp(POSITIONS, SOURCE, cotangent)
    legacy_map = legacy()
    np.testing.assert_allclose(
        jvp,
        legacy_map.apply(direction) * LEGACY_TO_AUTHORITATIVE_SCALE,
        rtol=2e-15,
        atol=0,
    )
    np.testing.assert_allclose(
        vjp,
        legacy_map.adjoint(cotangent) * LEGACY_TO_AUTHORITATIVE_SCALE,
        rtol=2e-15,
        atol=0,
    )
    assert float(np.vdot(cotangent, jvp)) == pytest.approx(
        float(np.vdot(vjp, direction)), rel=2e-13, abs=2e-13
    )

    paired_source_cotangent = ATOMIC_L1_PAIRING.source_to_field_dual(SOURCE)
    response_gradient = provider.source_vjp(POSITIONS, SOURCE, paired_source_cotangent)
    np.testing.assert_allclose(
        response_gradient,
        ATOMIC_L1_PAIRING.field_to_source_dual(field),
        rtol=2e-13,
        atol=2e-13,
    )
    step = 2e-6
    fd = (
        provider.energy(POSITIONS, SOURCE + step * direction)
        - provider.energy(POSITIONS, SOURCE - step * direction)
    ) / (2 * step)
    analytic = float(np.vdot(ATOMIC_L1_PAIRING.field_to_source_dual(field), direction))
    assert analytic == pytest.approx(fd, rel=0, abs=2e-8)


def test_complete_coordinate_vjp_matches_rebuilt_geometry_finite_difference():
    provider = backend()
    cotangent = np.asarray([[0.12, 0.03, -0.05, 0.04], [-0.20, 0.02, 0.05, -0.03]])
    analytic = provider.coordinate_vjp(POSITIONS, SOURCE, cotangent).reshape(2, 3)
    np.testing.assert_allclose(
        analytic,
        legacy().full_position_vjp(SOURCE, cotangent) * LEGACY_TO_AUTHORITATIVE_SCALE,
        rtol=2e-15,
        atol=0,
    )
    step = 1e-5
    finite_difference = np.empty_like(analytic)
    for atom in range(2):
        for axis in range(3):
            plus = POSITIONS.copy()
            minus = POSITIONS.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            finite_difference[atom, axis] = (
                np.vdot(cotangent, provider.field(plus, SOURCE))
                - np.vdot(cotangent, provider.field(minus, SOURCE))
            ) / (2 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=0, atol=7e-8)
    np.testing.assert_allclose(
        np.sum(analytic, axis=0), np.zeros(3), rtol=0, atol=3e-12
    )

    # The canonical scalar uses 0.5*Q.T*c as its field cotangent.  Check that
    # exact same-scalar coordinate gradient, not only an arbitrary bilinear.
    coupling = reciprocal_linear_half_coupling(provider, POSITIONS, SOURCE)
    scalar_analytic = coupling.coordinate_gradient_array().reshape(2, 3)
    scalar_fd = np.empty_like(scalar_analytic)
    for atom in range(2):
        for axis in range(3):
            plus = POSITIONS.copy()
            minus = POSITIONS.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            scalar_fd[atom, axis] = (
                provider.energy(plus, SOURCE) - provider.energy(minus, SOURCE)
            ) / (2 * step)
    np.testing.assert_allclose(scalar_analytic, scalar_fd, rtol=0, atol=4e-8)


def test_determinism_and_stable_provenance_are_cold_replay_exact():
    first_provider = backend()
    second_provider = backend()
    first = first_provider.build_state(POSITIONS, SOURCE)
    second = second_provider.build_state(POSITIONS.copy(), SOURCE.copy())
    assert first_provider.provenance_sha256 == second_provider.provenance_sha256
    assert first_provider.runtime_provenance == second_provider.runtime_provenance
    assert first.surface.topology_hash == second.surface.topology_hash
    assert first.surface.state_hash == second.surface.state_hash
    assert first.state_hash == second.state_hash
    np.testing.assert_array_equal(first.reaction_field, second.reaction_field)


def test_configuration_is_immutable_and_every_call_checks_its_fingerprint():
    provider = backend()
    original_radii = provider.cavity_radii_angstrom
    original_radii.setflags(write=True)
    original_radii[0] = 99.0
    assert provider.cavity_radii_angstrom[0] == pytest.approx(1.1)
    with pytest.raises(AttributeError, match="immutable"):
        provider.dielectric = 2.0
    with pytest.raises(AttributeError, match="immutable"):
        provider.surface_provider._switching_constant = 1.0

    object.__setattr__(provider, "_configuration", provider._configuration[:-1])
    for operation in (
        lambda: provider.build_state(POSITIONS, SOURCE),
        lambda: provider.source_jvp(POSITIONS, SOURCE, SOURCE),
        lambda: provider.source_vjp(POSITIONS, SOURCE, SOURCE),
        lambda: provider.coordinate_vjp(POSITIONS, SOURCE, SOURCE),
    ):
        with pytest.raises(RuntimeError, match="configuration fingerprint"):
            operation()

    missing = backend()
    object.__delattr__(missing, "_configuration")
    with pytest.raises(RuntimeError, match="configuration is missing"):
        missing.build_state(POSITIONS, SOURCE)


def test_surface_and_cpcm_states_reject_replace_forgery():
    state = backend().build_state(POSITIONS, SOURCE)
    surface = state.surface
    with pytest.raises(ValueError, match="topology_hash"):
        replace(surface, topology_hash="0" * 64)
    changed_points = list(surface.surface_point_values)
    changed_points[0] += 0.1
    with pytest.raises(ValueError, match="parent/radius/direction"):
        replace(surface, surface_point_values=tuple(changed_points))
    changed_directions = list(surface.unit_direction_values)
    changed_directions[0] *= 2.0
    with pytest.raises(ValueError, match="unit norm"):
        replace(surface, unit_direction_values=tuple(changed_directions))
    with pytest.raises(ValueError, match="half coupling"):
        replace(state, polarization_energy_ev=state.polarization_energy_ev + 1e-4)
    with pytest.raises(ValueError, match="state_hash"):
        replace(state, state_hash="0" * 64)
    other_surface = FixedTopologyCPCMBackend(
        ("H", "H"),
        np.asarray([1.2, 1.2]),
        dielectric=78.39,
        unit_sphere=SIX_POINT_SPHERE,
        switching_constant=SWITCHING_CONSTANT,
    ).surface_provider.build_state(POSITIONS)
    with pytest.raises(ValueError, match="not bound to this surface"):
        replace(state, surface=other_surface)
    with pytest.raises(ValueError, match="backend_configuration"):
        replace(state, backend_configuration=state.backend_configuration[:-1])
    with pytest.raises(ValueError, match="provider_id"):
        replace(state, provider_id="forged-provider")
    with pytest.raises(ValueError, match="capabilities remain closed"):
        from maple.solvation.api.capabilities import CapabilityStatus

        replace(state, capabilities=CapabilityStatus(energy=True))


def test_nonfinite_singular_and_incomplete_derivative_paths_fail_closed():
    provider = backend()
    bad_source = SOURCE.copy()
    bad_source[0, 0] = np.nan
    with pytest.raises(ValueError, match="source must be finite"):
        provider.build_state(POSITIONS, bad_source)

    coincident_atoms = np.zeros((2, 3))
    with pytest.raises(ValueError, match="coincident candidate"):
        provider.build_state(coincident_atoms, SOURCE)

    class IncompleteBackend(FixedTopologyCPCMBackend):
        def _map_and_surface(self, geometry):
            surface, reaction_map, _ = super()._map_and_surface(geometry)
            return surface, reaction_map, False

    incomplete = IncompleteBackend(
        ("H", "H"),
        [1.1, 1.1],
        dielectric=78.39,
        unit_sphere=SIX_POINT_SPHERE,
        switching_constant=SWITCHING_CONSTANT,
    )
    with pytest.raises(
        NotImplementedError, match="complete coordinate derivative contract"
    ):
        incomplete.coordinate_vjp(POSITIONS, SOURCE, np.ones_like(SOURCE))


def test_constructor_rejects_partial_grid_injection_and_invalid_dielectric():
    with pytest.raises(ValueError, match="supplied together"):
        FixedTopologyCPCMBackend(
            ("H",), [1.1], dielectric=78.39, unit_sphere=SIX_POINT_SPHERE
        )
    with pytest.raises(ValueError, match="greater than one"):
        FixedTopologyCPCMBackend(
            ("H",),
            [1.1],
            dielectric=1.0,
            unit_sphere=SIX_POINT_SPHERE,
            switching_constant=SWITCHING_CONSTANT,
        )
