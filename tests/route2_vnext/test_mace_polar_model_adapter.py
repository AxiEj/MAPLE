from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.profiles import LOCAL_JET_DIAGNOSTIC_COUPLING_ID
from maple.solvation.models import (
    ElectronicResponseEquationAdapter,
    MACEPolarLocalFieldModelAdapter,
    MACEPolarReleaseContract,
    VacuumScalarEquationAdapter,
    validate_response_linearization,
    validate_source_evaluation,
    validate_vacuum_evaluation,
)


@dataclass
class _State:
    energy_ev: float
    density_coefficients: np.ndarray
    fixed_field_forces_ev_per_angstrom: np.ndarray | None


class _Linearization:
    def __init__(self, jacobian: np.ndarray):
        self.jacobian = jacobian

    def jvp(self, direction):
        return np.asarray(direction) @ self.jacobian.T

    def vjp(self, cotangent):
        return np.asarray(cotangent) @ self.jacobian


class _FakeMACEPolarCalculator:
    dtype = "float64"
    device = "cpu"
    mace_torch_version = "test-mace-1"
    graph_longrange_version = "test-graph-1"
    route2_mace_geometry_frame_policy = "laboratory-v1"
    long_range_evaluator_profile = "test-molecular-realspace-v1"
    atomic_numbers = (1, 6, 8)

    def __init__(self, checkpoint_path, release):
        self.mace_polar_checkpoint_provenance = {
            "identifier": release.checkpoint_identifier,
            "release_url": release.checkpoint_release_url,
            "resolved_path": str(checkpoint_path),
            "size_bytes": release.checkpoint_size_bytes,
            "sha256": release.checkpoint_sha256,
        }
        source_basis = SimpleNamespace(sigmas=(1.5,), max_l=1, normalize="multipoles")
        self.model = SimpleNamespace(
            coulomb_energy=SimpleNamespace(density_basis=source_basis)
        )
        self.jacobian = np.asarray(
            [
                [0.12, -0.03, 0.02, 0.01],
                [0.04, 0.08, -0.02, 0.03],
                [-0.01, 0.05, 0.09, -0.04],
                [0.02, -0.01, 0.03, 0.07],
            ]
        )
        self.position_source_vector = np.asarray([0.2, -0.1, 0.04, 0.03])

    def route2_gto_field_projection_spec(self):
        return SimpleNamespace(
            receiver_sigmas_angstrom=(1.5, 3.0),
            receiver_max_l=1,
            receiver_normalization="receiver",
            upstream_matrix=np.arange(32.0).reshape(8, 4) / 31.0,
        )

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev=None,
        node_gradient_ev_per_angstrom=None,
        compute_forces=False,
    ):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        field = np.zeros((len(atoms), 4))
        if node_potential_ev is not None:
            field[:, 0] = node_potential_ev
            field[:, 1:] = node_gradient_ev_per_angstrom
        density = 0.05 + field @ self.jacobian.T
        density += 0.1 * positions[:, :1] * self.position_source_vector.reshape(1, 4)
        energy = float(0.5 * np.sum(positions**2) + 0.1 * np.sum(field**2))
        forces = -positions if compute_forces else None
        return _State(energy, density, forces), {"fake": True}

    def linearize_density_response(
        self, atoms, *, node_potential_ev, node_gradient_ev_per_angstrom
    ):
        del atoms, node_potential_ev, node_gradient_ev_per_angstrom
        return _Linearization(self.jacobian)

    def density_position_vjp(
        self,
        atoms,
        *,
        node_potential_ev,
        node_gradient_ev_per_angstrom,
        density_cotangent,
    ):
        del node_potential_ev, node_gradient_ev_per_angstrom
        result = np.zeros((len(atoms), 3))
        result[:, 0] = 0.1 * (
            np.asarray(density_cotangent) @ self.position_source_vector
        )
        return result


def _adapter(tmp_path):
    checkpoint = tmp_path / "test-polar.model"
    checkpoint.write_bytes(b"test MACE-POLAR checkpoint bytes\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    release = MACEPolarReleaseContract(
        provider_id="maple.route2.model.test-mace-polar.impl.v1",
        model_profile_id="route2-test-mace-polar-model-v1",
        checkpoint_identifier="test-polar",
        checkpoint_release_url="file://test-polar.model",
        checkpoint_sha256=digest,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        mace_torch_version="test-mace-1",
        graph_longrange_version="test-graph-1",
        upstream_commit="test-upstream-commit",
        release_status="test-only-unadmitted",
    )
    calculator = _FakeMACEPolarCalculator(checkpoint, release)
    return MACEPolarLocalFieldModelAdapter(calculator, release), calculator


def _atoms():
    return Atoms(
        "HC",
        positions=np.asarray([[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]]),
        info={"charge": 0, "mult": 1},
    )


def test_mace_polar_adapter_binds_checkpoint_runtime_and_negative_exact_gto_audit(
    tmp_path,
):
    adapter, _ = _adapter(tmp_path)
    assert adapter.coupling_id == LOCAL_JET_DIAGNOSTIC_COUPLING_ID
    assert (
        adapter.provenance.checkpoint_sha256
        == hashlib.sha256((tmp_path / "test-polar.model").read_bytes()).hexdigest()
    )
    assert adapter.provenance.optimizer_parameter_groups_audited is False
    assert adapter.variational_functional_admitted is False
    assert adapter.exact_gto_operational_available is False
    audit = adapter.exact_gto_audit
    assert audit.source_sigmas_angstrom == (1.5,)
    assert audit.receiver_sigmas_angstrom == (1.5, 3.0)
    assert audit.upstream_matrix_shape == (8, 4)
    assert audit.same_basis_conjugacy is False
    assert "cannot be the adjoint" in audit.reason
    assert len(adapter.configuration_sha256()) == 64


def test_mace_polar_model_states_linearization_and_equation_bridges(tmp_path):
    adapter, _ = _adapter(tmp_path)
    atoms = _atoms()
    field = np.arange(8.0).reshape(2, 4) / 20.0
    vacuum = validate_vacuum_evaluation(adapter, atoms, need_forces=True)
    source = validate_source_evaluation(
        adapter, atoms, field, need_fixed_field_forces=True
    )
    assert vacuum.forces_eV_per_A.shape == (2, 3)
    assert source.source.shape == (2, 4)
    direction = np.arange(8.0, 16.0).reshape(2, 4) / 17.0
    cotangent = np.arange(-4.0, 4.0).reshape(2, 4) / 11.0
    jvp, vjp, position_vjp = validate_response_linearization(
        adapter,
        atoms,
        field,
        field_direction=direction,
        source_cotangent=cotangent,
        transpose_atol=1e-14,
        transpose_rtol=1e-14,
    )
    assert float(np.vdot(jvp, cotangent)) == pytest.approx(
        float(np.vdot(direction, vjp)), abs=1e-14
    )
    assert position_vjp.shape == (2, 3)

    equation = ElectronicResponseEquationAdapter(
        adapter, LOCAL_JET_DIAGNOSTIC_COUPLING_ID
    )
    vacuum_leaf = VacuumScalarEquationAdapter(adapter)
    np.testing.assert_allclose(equation.evaluate_source(atoms, field), source.source)
    assert equation.coordinate_vjp(atoms, field, cotangent).shape == (6,)
    assert vacuum_leaf.evaluate_energy(atoms) == pytest.approx(vacuum.energy_eV)
    np.testing.assert_allclose(
        vacuum_leaf.coordinate_gradient(atoms),
        -vacuum.forces_eV_per_A.reshape(-1),
    )


def test_mace_polar_adapter_fails_closed_on_runtime_or_checkpoint_drift(tmp_path):
    adapter, calculator = _adapter(tmp_path)
    with pytest.raises(AttributeError, match="immutable"):
        adapter.device = "cuda"
    calculator.graph_longrange_version = "drifted"
    with pytest.raises(ValueError, match="configuration drifted"):
        adapter.configuration_sha256()

    adapter, calculator = _adapter(tmp_path)
    calculator.linearize_density_response = lambda *args, **kwargs: None
    with pytest.raises(TypeError, match="instance-level callable rebinding"):
        adapter.configuration_sha256()

    adapter, _ = _adapter(tmp_path)
    checkpoint = tmp_path / "test-polar.model"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint file identity changed"):
        adapter.configuration_sha256()
