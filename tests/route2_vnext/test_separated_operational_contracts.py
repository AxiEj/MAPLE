from __future__ import annotations

import hashlib

import numpy as np
import pytest

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    SCALAR_REGISTRY,
)
from maple.solvation.api.state_registry import (
    SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
    STATE_REGISTRY,
)
from maple.solvation.coupling.separated_ledgers import (
    ExternalEnthalpyOperationalLedger,
    FrozenVacuumContinuumLedger,
)
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SeparatedContinuumSnapshot,
    build_mace_polar_harmonic_separated_snapshot,
)
from maple.solvation.coupling.separated_state import (
    SeparatedOperationalStateEquation,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_SOURCE_SPACE,
    AffineChargeCoordinates,
)
from maple.solvation.release.root_well_posedness import (
    certify_root_well_posedness,
    dense_state_map_jacobian,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _FakeHarmonicFunctional:
    provider_id = "test.harmonic.functional.v1"
    continuum_profile_id = "test.harmonic.continuum.v1"
    cavity_profile_id = "test.harmonic.cavity.v1"
    provenance_sha256 = _digest("continuum-provenance")
    scalar_id = OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1

    def __init__(self, surface, source):
        self._surface = np.asarray(surface, dtype=float)
        self._source = np.asarray(source, dtype=float)

    def configuration_sha256(self):
        return _digest("continuum-configuration")

    def topology_sha256(self):
        return _digest("continuum-topology")

    def debug_geometry_matrices(self, geometry):
        del geometry
        return {
            "surface_operator": self._surface.copy(),
            "source_operator": self._source.copy(),
        }


class _SeparatedElectronic:
    provider_id = "test.separated.electronic.v1"
    model_profile_id = "test.separated.model.v1"
    provenance_sha256 = _digest("electronic-provenance")
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE

    def __init__(self, coupling_id, jacobian):
        self.coupling_id = coupling_id
        self._jacobian = np.asarray(jacobian, dtype=float)

    def configuration_sha256(self):
        return _digest("electronic-configuration" + repr(self._jacobian.tolist()))

    def evaluate_source(self, geometry, field):
        del geometry
        result = self._jacobian @ np.asarray(field).reshape(-1)
        return result.reshape(1, 4)

    def field_jvp(self, geometry, field, field_direction):
        del geometry, field
        result = self._jacobian @ np.asarray(field_direction).reshape(-1)
        return result.reshape(1, 4)

    def field_vjp(self, geometry, field, source_cotangent):
        del geometry, field
        result = self._jacobian.T @ np.asarray(source_cotangent).reshape(-1)
        return result.reshape(1, 8)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        del geometry, field, source_cotangent
        return np.zeros((1, 3))

    def intrinsic_energy_ev(self, geometry, field):
        del geometry
        return 2.0 + 0.25 * float(np.vdot(field, field))


class _VacuumModel:
    provider_id = "test.vacuum.v1"
    model_profile_id = "test.separated.model.v1"
    provenance_sha256 = _digest("vacuum-provenance")

    def configuration_sha256(self):
        return _digest("vacuum-configuration")

    def evaluate_energy(self, geometry):
        del geometry
        return 1.5

    def coordinate_gradient(self, geometry):
        return np.zeros(np.asarray(geometry).size)


def _snapshot(
    geometry,
    scalar_id=(OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1),
):
    surface = np.asarray([[2.0, 0.2], [0.2, 1.5]])
    source = np.asarray(
        [
            [0.4, 0.0, 0.1, -0.2],
            [0.1, 0.2, -0.1, 0.3],
        ]
    )
    receiver = np.asarray(
        [
            [0.2, -0.1],
            [-0.3, 0.4],
            [0.1, 0.2],
            [0.0, 0.1],
            [0.2, 0.3],
            [-0.1, 0.2],
            [0.4, -0.2],
            [0.2, 0.1],
        ]
    )
    embedding = np.zeros((8, 4))
    embedding[(0, 2, 3, 4), (0, 1, 2, 3)] = 1.0
    from maple.solvation.coupling.state_equation import geometry_sha256

    return SeparatedContinuumSnapshot(
        atom_count=1,
        geometry_digest=geometry_sha256(geometry),
        continuum_provider_id="test.continuum.v1",
        continuum_profile_id="test.continuum.profile.v1",
        cavity_profile_id="test.cavity.profile.v1",
        scalar_id=scalar_id,
        continuum_configuration_sha256=_digest("continuum-configuration"),
        continuum_provenance_sha256=_digest("continuum-provenance"),
        topology_sha256=_digest("continuum-topology"),
        surface_operator=surface,
        source_to_boundary=source,
        boundary_to_native_field=receiver,
        source_embedding=embedding,
    )


def test_harmonic_builder_separates_source4_boundary_and_native_field8():
    geometry = np.asarray([[0.1, -0.2, 0.3]])
    surface = np.asarray([[2.0, 0.1], [0.1, 1.5]])
    source8 = np.arange(16.0).reshape(2, 8) / 11.0
    snapshot = build_mace_polar_harmonic_separated_snapshot(
        _FakeHarmonicFunctional(surface, source8), geometry
    )
    assert snapshot.source_to_boundary.shape == (2, 4)
    assert snapshot.boundary_to_native_field.shape == (8, 2)
    np.testing.assert_allclose(
        snapshot.source_to_boundary,
        source8 @ snapshot.source_embedding,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        snapshot.boundary_to_native_field, -source8.T, rtol=0.0, atol=0.0
    )
    source4 = np.asarray([[0.2, -0.1, 0.04, 0.03]])
    direction = np.asarray([[0.0, 0.02, -0.01, 0.03]])
    step = 1.0e-6
    finite_difference = (
        snapshot.continuum_energy_eV(source4 + step * direction)
        - snapshot.continuum_energy_eV(source4 - step * direction)
    ) / (2.0 * step)
    analytic = float(np.vdot(snapshot.continuum_source_gradient(source4), direction))
    assert analytic == pytest.approx(finite_difference, rel=2.0e-10, abs=2.0e-12)
    with pytest.raises(AttributeError, match="immutable"):
        snapshot.coupling_id = "forged"


def test_separated_state_jvp_vjp_ledgers_and_local_root_certificate():
    geometry = np.asarray([[0.1, -0.2, 0.3]])
    snapshot = _snapshot(geometry)
    jacobian = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.05, 0.03, 0.0, 0.04, -0.01, 0.02],
            [-0.03, 0.06, 0.08, -0.02, 0.01, 0.0, 0.02, -0.04],
            [0.02, 0.01, -0.04, 0.07, -0.02, 0.03, 0.0, 0.01],
        ]
    )
    electronic = _SeparatedElectronic(snapshot.coupling_id, jacobian)
    coordinates = AffineChargeCoordinates(
        atom_count=1, total_charge=0.0, source_space=ATOMIC_L1_SOURCE_SPACE
    )
    equation = SeparatedOperationalStateEquation(coordinates, electronic, snapshot)
    y = np.zeros(equation.reduced_dimension)
    rng = np.random.default_rng(20260815)
    direction = rng.normal(size=y.shape)
    cotangent = rng.normal(size=y.shape)
    jvp = equation.residual_jvp(geometry, y, direction)
    vjp = equation.residual_vjp(geometry, y, cotangent)
    assert float(np.vdot(jvp, cotangent)) == pytest.approx(
        float(np.vdot(direction, vjp)), rel=2.0e-13, abs=2.0e-14
    )
    map_jacobian = dense_state_map_jacobian(
        lambda value: equation.state_map_jvp(geometry, y, value),
        dimension=equation.reduced_dimension,
    )
    certificate = certify_root_well_posedness(map_jacobian)
    assert certificate.local_implicit_branch_certified is True
    assert certificate.global_unique_root_certified is False
    assert certificate.residual_error_bound(1.0e-12) is None

    vacuum = _VacuumModel()
    phi0 = FrozenVacuumContinuumLedger(equation=equation, vacuum=vacuum)
    with pytest.raises(ValueError, match="different scalar ID"):
        ExternalEnthalpyOperationalLedger(equation=equation)
    phi1_snapshot = _snapshot(
        geometry,
        OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    )
    phi1_equation = SeparatedOperationalStateEquation(
        coordinates, electronic, phi1_snapshot
    )
    phi1 = ExternalEnthalpyOperationalLedger(equation=phi1_equation)
    result0 = phi0.evaluate_root(geometry, y, root_tolerance=1.0e-14)
    result1 = phi1.evaluate_root(geometry, y, root_tolerance=1.0e-14)
    assert result0.total_energy_eV == pytest.approx(1.5)
    assert result1.total_energy_eV == pytest.approx(2.0)
    assert result0.scalar_id != result1.scalar_id
    assert result0.components_eV != result1.components_eV


def test_separated_registry_entries_are_distinct_disabled_ledgers():
    assert SEPARATED_OPERATIONAL_STATE_EQUATION_ID in STATE_REGISTRY
    phi0 = SCALAR_REGISTRY[
        OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
    ]
    phi1 = SCALAR_REGISTRY[
        OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
    ]
    assert phi0.state_equation_id == phi1.state_equation_id
    assert phi0.exact_formula != phi1.exact_formula
    assert phi0.enabled is phi1.enabled is False
    assert phi0.admitted_capabilities.enabled_tiers == ()
    assert phi1.admitted_capabilities.enabled_tiers == ()
    assert "field_conditioned_intrinsic_energy" in phi0.excluded_components
    assert "macepolar_intrinsic_field_conditioned_energy" in phi1.included_components
