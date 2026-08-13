from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.gto_field_projection import (
    MACEPolarGTOFieldProjectionSpec,
)
from maple.solvation.api.profiles import LOCAL_JET_DIAGNOSTIC_COUPLING_ID
from maple.solvation.coupling.exact_gto import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.models import (
    ElectronicResponseEquationAdapter,
    MACEPolarLocalFieldModelAdapter,
    MACEPolarRadialGTOModelAdapter,
    MACEPolarReleaseContract,
    MACE_POLAR_1_M_FIXED_BOX40_CONTRACT,
    MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID,
    MACE_POLAR_FIXED_BOX_RELEASE_CONTRACTS,
    VacuumScalarEquationAdapter,
    validate_response_linearization,
    validate_source_evaluation,
    validate_vacuum_evaluation,
)
from maple.solvation.models.mace_polar_variational import (
    MACEPolarDifferentiableFieldGraph,
    MACEPolarVariationalFieldEnergy,
    MACE_POLAR_VARIATIONAL_DUALITY_MAP,
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
    graph_longrange_version = "0.4.0"
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
        self.feature_jacobian = np.asarray(
            [
                [0.12, -0.03, 0.02, 0.01, 0.04, -0.02, 0.03, 0.05],
                [0.04, 0.08, -0.02, 0.03, -0.01, 0.06, 0.02, -0.04],
                [-0.01, 0.05, 0.09, -0.04, 0.03, 0.01, -0.05, 0.02],
                [0.02, -0.01, 0.03, 0.07, -0.02, 0.04, 0.06, 0.01],
            ]
        )

    def route2_gto_field_projection_spec(self):
        return MACEPolarGTOFieldProjectionSpec(
            receiver_sigmas_angstrom=(1.5, 3.0),
            receiver_max_l=1,
            receiver_normalization="receiver",
            upstream_matrix=np.asarray(
                [
                    [3.544907808303833, 0.0, 0.0, 0.0],
                    [3.544907808303833, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.771474361419678],
                    [0.0, 5.771474361419678, 0.0, 0.0],
                    [0.0, 0.0, 5.771474361419678, 0.0],
                    [0.0, 0.0, 0.0, 11.542948722839355],
                    [0.0, 11.542948722839355, 0.0, 0.0],
                    [0.0, 0.0, 11.542948722839355, 0.0],
                ]
            ),
        )

    def polar_state(
        self,
        atoms,
        *,
        node_potential_ev=None,
        node_gradient_ev_per_angstrom=None,
        model_field_features=None,
        compute_forces=False,
    ):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        field = np.zeros((len(atoms), 4))
        if node_potential_ev is not None:
            field[:, 0] = node_potential_ev
            field[:, 1:] = node_gradient_ev_per_angstrom
        if model_field_features is not None:
            features = np.asarray(model_field_features)
            density = 0.05 + features @ self.feature_jacobian.T
            field_energy = np.sum(features**2)
        else:
            density = 0.05 + field @ self.jacobian.T
            field_energy = np.sum(field**2)
        density += 0.1 * positions[:, :1] * self.position_source_vector.reshape(1, 4)
        energy = float(0.5 * np.sum(positions**2) + 0.1 * field_energy)
        forces = -positions if compute_forces else None
        return _State(energy, density, forces), {"fake": True}

    def linearize_density_response(
        self, atoms, *, node_potential_ev, node_gradient_ev_per_angstrom
    ):
        del atoms, node_potential_ev, node_gradient_ev_per_angstrom
        return _Linearization(self.jacobian)

    def linearize_density_response_features(self, atoms, *, model_field_features):
        del atoms, model_field_features
        return _Linearization(self.feature_jacobian)

    def intrinsic_energy_model_feature_gradient(self, atoms, *, model_field_features):
        del atoms
        return 0.2 * np.asarray(model_field_features, dtype=float)

    def polar_output_torch(
        self,
        atoms,
        *,
        model_field_features,
        positions_angstrom=None,
        **kwargs,
    ):
        del kwargs
        torch = pytest.importorskip("torch")
        if positions_angstrom is None:
            positions_angstrom = torch.as_tensor(atoms.positions, dtype=torch.float64)
        features = model_field_features
        feature_jacobian = torch.as_tensor(
            self.feature_jacobian, dtype=features.dtype, device=features.device
        )
        position_vector = torch.as_tensor(
            self.position_source_vector,
            dtype=features.dtype,
            device=features.device,
        )
        density = 0.05 + features @ feature_jacobian.T
        density = density + 0.1 * positions_angstrom[:, :1] * position_vector
        density = density.clone()
        density[:, 0] -= density[:, 0].mean()
        energy = 0.5 * torch.sum(positions_angstrom**2) + 0.1 * torch.sum(features**2)
        return {"energy": energy, "density_coefficients": density}

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

    def density_position_vjp_features(
        self, atoms, *, model_field_features, density_cotangent
    ):
        del model_field_features
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
        long_range_evaluator_profile="test-molecular-realspace-v1",
        checkpoint_identifier="test-polar",
        checkpoint_release_url="file://test-polar.model",
        checkpoint_sha256=digest,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        mace_torch_version="test-mace-1",
        graph_longrange_version="0.4.0",
        upstream_commit="test-upstream-commit",
        release_status="test-only-unadmitted",
    )
    calculator = _FakeMACEPolarCalculator(checkpoint, release)
    return MACEPolarLocalFieldModelAdapter(calculator, release), calculator


class _FakeDifferentiableFieldGraph:
    def __init__(self, calculator):
        self.calculator = calculator

    def configuration_sha256(self):
        return "a" * 64

    def __call__(
        self,
        atoms,
        *,
        model_field_features,
        positions_angstrom,
    ):
        return self.calculator.polar_output_torch(
            atoms,
            model_field_features=model_field_features,
            positions_angstrom=positions_angstrom,
        )


class _AlternateFakeDifferentiableFieldGraph(_FakeDifferentiableFieldGraph):
    def configuration_sha256(self):
        return "b" * 64


def _variational_adapter(base, calculator):
    return MACEPolarVariationalFieldEnergy(
        MACEPolarRadialGTOModelAdapter(base),
        field_graph=_FakeDifferentiableFieldGraph(calculator),
    )


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


def test_radial_adapter_exposes_intrinsic_scalar_gradient_and_dense_source_jacobian(
    tmp_path,
):
    base, calculator = _adapter(tmp_path)
    adapter = MACEPolarRadialGTOModelAdapter(base)
    atoms = _atoms()
    field = np.linspace(-0.008, 0.011, len(atoms) * 8).reshape(len(atoms), 8)

    features = adapter.field_transform.to_model_features(field)
    expected_energy = float(
        0.5 * np.sum(atoms.positions**2) + 0.1 * np.sum(features**2)
    )
    assert adapter.intrinsic_energy_ev(atoms, field) == pytest.approx(
        expected_energy, rel=0.0, abs=1.0e-14
    )
    expected_gradient = adapter.field_transform.vjp(0.2 * features)
    np.testing.assert_allclose(
        adapter.intrinsic_energy_field_gradient(atoms, field),
        expected_gradient,
        rtol=0.0,
        atol=1.0e-14,
    )
    jacobian = adapter.dense_source_jacobian(atoms, field)
    direction = np.linspace(0.003, -0.002, field.size).reshape(field.shape)
    np.testing.assert_allclose(
        (jacobian @ direction.reshape(-1)).reshape(field.shape),
        adapter.source_jvp(atoms, field, direction),
        rtol=0.0,
        atol=2.0e-14,
    )
    missing_rows = np.asarray([1, 5, 6, 7, 9, 13, 14, 15])
    np.testing.assert_array_equal(jacobian[missing_rows], 0.0)
    assert calculator is base._calculator


def test_radial_adapter_forward_mode_energy_derivative_uses_same_scalar_graph(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    base, calculator = _adapter(tmp_path)
    adapter = MACEPolarRadialGTOModelAdapter(base)
    atoms = _atoms()
    field = np.linspace(-0.008, 0.011, len(atoms) * 8).reshape(len(atoms), 8)
    direction = np.linspace(0.003, -0.002, field.size).reshape(field.shape)

    def polar_output_torch(atoms_arg, *, model_field_features, **kwargs):
        del atoms_arg, kwargs
        return {"energy": 0.1 * torch.sum(model_field_features**2)}

    monkeypatch.setattr(
        calculator, "polar_output_torch", polar_output_torch, raising=False
    )
    expected = float(
        np.vdot(adapter.intrinsic_energy_field_gradient(atoms, field), direction)
    )
    assert adapter.intrinsic_energy_field_directional_derivative(
        atoms, field, direction
    ) == pytest.approx(expected, abs=1.0e-14)


def test_variational_adapter_anchors_zero_field_source_and_fills_all_radial_channels(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    base, calculator = _adapter(tmp_path)
    operational = MACEPolarRadialGTOModelAdapter(base)
    variational = MACEPolarVariationalFieldEnergy(
        operational,
        field_graph=_FakeDifferentiableFieldGraph(calculator),
    )
    atoms = _atoms()
    count = len(atoms)
    total_charge = 0.0
    zero = np.zeros((count, 8))
    original_output = calculator.polar_output_torch(
        atoms,
        model_field_features=torch.zeros((count, 8), dtype=torch.float64),
    )
    original = np.zeros((count, 8))
    original[:, (0, 2, 3, 4)] = np.asarray(
        original_output["density_coefficients"].detach().cpu()
    )
    anchored = variational.evaluate_source(atoms, zero)
    np.testing.assert_allclose(anchored, original, atol=3e-14, rtol=0.0)

    field = np.linspace(-0.004, 0.006, count * 8).reshape(count, 8)
    source = variational.evaluate_source(atoms, field)
    assert np.linalg.norm(source[:, (1, 5, 6, 7)]) > 1.0e-6
    assert variational.source_space.total_charge(
        source, atom_count=count
    ) == pytest.approx(total_charge, abs=2e-13)
    assert variational.original_density_observable is operational
    assert variational.variational_functional_admitted is False
    assert variational.capabilities.enabled_tiers == ()
    assert variational.metadata()["capabilities"] == {tier: False for tier in "EFHVM"}

    coordinates = MACE_POLAR_VARIATIONAL_DUALITY_MAP.coordinates(
        atom_count=count, total_charge=total_charge
    )
    reduced = MACE_POLAR_VARIATIONAL_DUALITY_MAP.reduce_field(
        field, atom_count=count, total_charge=total_charge
    )
    direction = np.linspace(0.003, -0.002, coordinates.reduced_dimension)
    source_direction = variational.source_jvp(
        atoms, reduced, direction, total_charge=total_charge
    )
    source_cotangent = np.linspace(-0.2, 0.3, count * 8).reshape(count, 8)
    reduced_cotangent = variational.source_vjp(
        atoms,
        reduced,
        source_cotangent,
        total_charge=total_charge,
    )
    assert np.vdot(source_direction, source_cotangent) == pytest.approx(
        np.vdot(direction, reduced_cotangent), abs=2e-12
    )

    alternate = MACEPolarVariationalFieldEnergy(
        operational,
        field_graph=_AlternateFakeDifferentiableFieldGraph(calculator),
    )
    assert alternate.provenance_sha256 != variational.provenance_sha256
    assert alternate.configuration_sha256() != variational.configuration_sha256()


def test_variational_adapter_fails_closed_if_zero_field_density_breaks_charge(
    tmp_path,
):
    pytest.importorskip("torch")

    class BrokenChargeCalculator(_FakeMACEPolarCalculator):
        def polar_output_torch(self, *args, **kwargs):
            output = super().polar_output_torch(*args, **kwargs)
            density = output["density_coefficients"].clone()
            density[:, 0] += 0.1
            output["density_coefficients"] = density
            return output

    checkpoint = tmp_path / "test-polar.model"
    checkpoint.write_bytes(b"test MACE-POLAR checkpoint bytes\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    release = MACEPolarReleaseContract(
        provider_id="maple.route2.model.test-mace-polar.impl.v1",
        model_profile_id="route2-test-mace-polar-model-v1",
        long_range_evaluator_profile="test-molecular-realspace-v1",
        checkpoint_identifier="test-polar",
        checkpoint_release_url="file://test-polar.model",
        checkpoint_sha256=digest,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        mace_torch_version="test-mace-1",
        graph_longrange_version="0.4.0",
        upstream_commit="test-upstream-commit",
        release_status="test-only-unadmitted",
    )
    base = MACEPolarLocalFieldModelAdapter(
        BrokenChargeCalculator(checkpoint, release), release
    )
    variational = _variational_adapter(base, base._calculator)
    with pytest.raises(RuntimeError, match="zero-field density anchor"):
        variational.evaluate_source(_atoms(), np.zeros((2, 8)))


def test_candidate_field_graph_preserves_coordinates_without_touching_legacy_files(
    tmp_path,
):
    from contextlib import contextmanager

    torch = pytest.importorskip("torch")
    base, calculator = _adapter(tmp_path)
    atoms = _atoms()
    captured = {}

    def fake_batch(_atoms_arg):
        return {"positions": torch.zeros((len(atoms), 3), dtype=torch.float64)}

    def fake_forward(batch, **kwargs):
        del kwargs
        captured["positions"] = batch["positions"]
        return {
            "energy": torch.sum(batch["positions"] ** 2),
            "density_coefficients": torch.zeros((len(atoms), 4), dtype=torch.float64),
        }

    class _Projector:
        @contextmanager
        def use_model_field_features(self, values):
            del values
            yield

    calculator._batch_dict = fake_batch
    calculator._model_forward = fake_forward
    calculator._reaction_projector = _Projector()
    calculator._long_range_evaluator = SimpleNamespace(
        is_default=True,
        profile="test-molecular-realspace-v1",
    )
    graph = MACEPolarDifferentiableFieldGraph(calculator)
    positions = torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True)
    output = graph(
        atoms,
        model_field_features=torch.zeros((len(atoms), 8), dtype=torch.float64),
        positions_angstrom=positions,
    )
    (gradient,) = torch.autograd.grad(output["energy"], (positions,))
    torch.testing.assert_close(gradient, 2.0 * positions)
    assert captured["positions"] is positions
    assert len(graph.configuration_sha256()) == 64
    with pytest.raises(ValueError, match="supplied atoms geometry"):
        graph(
            atoms,
            model_field_features=torch.zeros((len(atoms), 8), dtype=torch.float64),
            positions_angstrom=positions + 0.1,
        )
    assert base._calculator is calculator


def test_fixed_box40_contract_has_a_distinct_fail_closed_model_identity():
    assert (
        MACE_POLAR_1_M_FIXED_BOX40_CONTRACT.model_profile_id
        == MACE_POLAR_FIXED_BOX40_MODEL_PROFILE_ID
    )
    assert (
        MACE_POLAR_1_M_FIXED_BOX40_CONTRACT.model_profile_id
        != "mace-polar-route2-source-field-contract-v1"
    )
    assert MACE_POLAR_1_M_FIXED_BOX40_CONTRACT.provider_id.endswith(
        "fixed-box40-local-field.impl.v1"
    )
    assert "E/F/H/V/M unadmitted" in (
        MACE_POLAR_1_M_FIXED_BOX40_CONTRACT.release_status
    )
    assert tuple(MACE_POLAR_FIXED_BOX_RELEASE_CONTRACTS) == (32, 40, 48, 56)
    for box_length, contract in MACE_POLAR_FIXED_BOX_RELEASE_CONTRACTS.items():
        assert contract.model_profile_id.endswith(f"fixed-box{box_length}-contract-v1")
        assert contract.provider_id.endswith(
            f"fixed-box{box_length}-local-field.impl.v1"
        )
        assert contract.long_range_evaluator_profile.endswith(
            f"fixed-box{box_length}-v1"
        )
        assert "E/F/H/V/M unadmitted" in contract.release_status


def test_fixed_box_configuration_hash_binds_runtime_evaluator_identity(tmp_path):
    adapter, calculator = _adapter(tmp_path)
    original = adapter.configuration_sha256()

    class _Evaluator:
        profile = "test-molecular-realspace-v1"
        provenance = {"operator": "changed-after-adapter-construction"}

    calculator._long_range_evaluator = _Evaluator()
    with pytest.raises(ValueError, match="configuration drifted"):
        adapter.configuration_sha256()
    assert original != ""


def test_release_contract_rejects_evaluator_identity_spoof(tmp_path):
    adapter, calculator = _adapter(tmp_path)
    wrong = replace(
        adapter._release_contract,
        long_range_evaluator_profile="different-evaluator-v1",
    )
    with pytest.raises(ValueError, match="long-range evaluator"):
        MACEPolarLocalFieldModelAdapter(calculator, wrong)


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


def test_radial_gto_model_adapter_closes_rectangular_response_and_position_vjp(
    tmp_path,
):
    local, calculator = _adapter(tmp_path)
    adapter = MACEPolarRadialGTOModelAdapter(local)
    atoms = _atoms()
    field = np.arange(16.0).reshape(2, 8) / 37.0
    state = validate_source_evaluation(
        adapter, atoms, field, need_fixed_field_forces=True
    )
    assert adapter.coupling_id == MACE_POLAR_RADIAL_GTO_COUPLING_ID
    assert adapter.exact_gto_coupling_available is True
    assert adapter.exact_gto_operational_available is False
    assert adapter.variational_functional_admitted is False
    assert state.source.shape == (2, 8)
    np.testing.assert_array_equal(state.source[:, (1, 5, 6, 7)], 0.0)

    rng = np.random.default_rng(20260813)
    direction = rng.normal(size=(2, 8))
    cotangent = rng.normal(size=(2, 8))
    jvp, vjp, position_vjp = validate_response_linearization(
        adapter,
        atoms,
        field,
        field_direction=direction,
        source_cotangent=cotangent,
        transpose_atol=2e-12,
        transpose_rtol=2e-12,
    )
    assert np.vdot(jvp, cotangent) == pytest.approx(np.vdot(direction, vjp), abs=2e-12)
    learned_cotangent = cotangent[:, (0, 2, 3, 4)]
    expected_position = calculator.density_position_vjp_features(
        atoms,
        model_field_features=adapter.field_transform.to_model_features(field),
        density_cotangent=learned_cotangent,
    )
    np.testing.assert_allclose(position_vjp, expected_position, atol=1e-14)

    equation = ElectronicResponseEquationAdapter(
        adapter, MACE_POLAR_RADIAL_GTO_COUPLING_ID
    )
    np.testing.assert_allclose(equation.evaluate_source(atoms, field), state.source)
    assert equation.coordinate_vjp(atoms, field, cotangent).shape == (6,)
    assert len(adapter.configuration_sha256()) == 64


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
