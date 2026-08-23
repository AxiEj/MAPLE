from __future__ import annotations

import hashlib

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.continuum.harmonic_ddpcm_functional import (
    SmoothPartitionHarmonicDDPCMFunctionalCandidate,
)
from maple.solvation.coupling.fixed_point import FixedPointOptions
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.operational_pes import OperationalImplicitPES
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.derivatives import RichardsonScalarHessian
from maple.solvation.experimental.harmonic_ddpcm_operational import (
    MACEPolarGaussianHarmonicDDPCMBuilder,
)
from maple.solvation.release.field_semantics import FieldSemanticsManifest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _TwoAtomElectronic:
    provider_id = "test.operational-pes.two-atom-electronic.v1"
    model_profile_id = "test.operational-pes.two-atom-model.v1"
    coupling_id = SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID
    provenance_sha256 = _digest("one-atom-electronic-provenance")
    checkpoint_sha256 = _digest("one-atom-electronic-checkpoint")
    field_energy_pairing_sha256 = MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash()
    source_space = ATOMIC_L1_SOURCE_SPACE
    receiver_space = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE

    def __init__(self) -> None:
        generator = np.random.default_rng(20260902)
        self._jacobian = generator.normal(scale=3.0e-4, size=(8, 16))
        self._base = np.asarray(
            [[0.0, 0.02, -0.01, 0.03], [0.0, -0.01, 0.015, -0.02]]
        )

    def configuration_sha256(self) -> str:
        return _digest(repr((self._jacobian.tolist(), self._base.tolist())))

    def evaluate_source(self, geometry, field):
        del geometry
        return self._base + (
            self._jacobian @ np.asarray(field).reshape(-1)
        ).reshape(2, 4)

    def evaluate_energy(self, geometry):
        positions = np.asarray(geometry.positions, dtype=float)
        return 0.5 * 0.02 * float(np.vdot(positions, positions))

    def coordinate_gradient(self, geometry):
        return 0.02 * np.asarray(geometry.positions, dtype=float)

    def field_jvp(self, geometry, field, field_direction):
        del geometry, field
        return (
            self._jacobian @ np.asarray(field_direction).reshape(-1)
        ).reshape(2, 4)

    def field_vjp(self, geometry, field, source_cotangent):
        del geometry, field
        return (
            self._jacobian.T @ np.asarray(source_cotangent).reshape(-1)
        ).reshape(2, 8)

    def coordinate_vjp(self, geometry, field, source_cotangent):
        del field, source_cotangent
        return np.zeros((len(geometry), 3))


def _manifest(electronic: _TwoAtomElectronic) -> FieldSemanticsManifest:
    return FieldSemanticsManifest(
        checkpoint_sha256=electronic.checkpoint_sha256,
        adapter_configuration_sha256=electronic.configuration_sha256(),
        adapter_provenance_sha256=electronic.provenance_sha256,
        model_provider_id=electronic.provider_id,
        model_profile_id=electronic.model_profile_id,
        source_space_sha256=electronic.source_space.metadata_hash(),
        native_field_space_sha256=electronic.receiver_space.metadata_hash(),
        pairing_metric_sha256=MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
        field_channel_order=electronic.receiver_space.components,
        field_radial_widths_angstrom=(1.5, 3.0),
        real_ylm_convention="test raw real-l1 convention",
        cartesian_spherical_l1_transform="test l1 transform",
        field_units=electronic.receiver_space.units,
        energy_unit="eV",
        external_potential_sign=None,
        uniform_field_sign=None,
        spin_channel_factor=None,
        native_injection_explicit_work_included=None,
        upstream_uniform_explicit_work_included=None,
        origin_convention="test fixed origin",
        evidence_measurement_sha256s=(),
    )


def _pes() -> OperationalImplicitPES:
    torch = pytest.importorskip("torch")
    electronic = _TwoAtomElectronic()
    continuum = SmoothPartitionHarmonicDDPCMFunctionalCandidate(
        atomic_numbers=(1, 1),
        radii_angstrom=(1.2, 1.2),
        transition_width_angstrom2=0.08,
        surface_lmax=1,
        partition_lmax=2,
        partition_radial_quadrature_order=24,
        source_radial_quadrature_order=24,
        double_layer_radial_quadrature_order=24,
        dielectric=80.0,
        dtype=torch.float64,
        device="cpu",
    )
    builder = MACEPolarGaussianHarmonicDDPCMBuilder(
        electronic=electronic,
        continuum=continuum,
        field_semantics_manifest=_manifest(electronic),
        receiver_radial_quadrature_order=24,
    )
    return OperationalImplicitPES(
        builder,
        root_options=FixedPointOptions(
            method="anderson",
            tolerance=1.0e-12,
            max_iterations=60,
            damping=0.8,
            history=5,
        ),
        hessian_backend=RichardsonScalarHessian(
            coarse_step_angstrom=2.0e-3,
            maximum_error_eV_per_A2=2.0e-5,
            maximum_antisymmetry_eV_per_A2=2.0e-5,
        ),
    )


def test_operational_pes_reuses_one_scalar_for_energy_force_virial_hvp_hessian():
    geometry = Atoms(
        "H2",
        positions=[[0.3, -0.2, 0.1], [1.05, 0.12, -0.08]],
        info={"charge": 0, "multiplicity": 1},
    )
    pes = _pes()
    energy = pes.sample(geometry)
    force = pes.force_sample(geometry)
    assert force.energy_sample == energy
    assert np.all(np.isfinite(force.forces_eV_per_A))

    virial = pes.molecular_virial(geometry, central_force=force)
    np.testing.assert_allclose(
        virial.net_force_eV_per_A,
        np.sum(force.forces_eV_per_A, axis=0),
        atol=0.0,
    )

    first = np.asarray([[1.0, 2.0, -1.0], [0.2, -0.4, 0.8]])
    second = np.asarray([[-0.5, 1.0, 2.0], [0.7, 0.3, -0.9]])
    hvp_first = pes.hessian_vector_product(geometry, first)
    hvp_second = pes.hessian_vector_product(geometry, second)
    assert np.vdot(first, hvp_second.hvp_eV_per_A2) == pytest.approx(
        np.vdot(hvp_first.hvp_eV_per_A2, second), abs=2.0e-7
    )

    hessian = pes.evaluate_hessian(geometry)
    np.testing.assert_allclose(
        hessian.raw_hessian_eV_per_A2,
        hessian.raw_hessian_eV_per_A2.T,
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        hessian.hessian_eV_per_A2 @ first.reshape(-1),
        hvp_first.hvp_eV_per_A2.reshape(-1),
        rtol=0.0,
        atol=2.0e-7,
    )
    assert hessian.maximum_antisymmetry_eV_per_A2 < 2.0e-7
