from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_PROFILE_V1,
    PROFILE_REGISTRY,
    SCALAR_REGISTRY,
)
from maple.solvation.api.profiles import AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID
from maple.solvation.api.profiles import (
    AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import (
    DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1,
)
from maple.solvation.continuum.atomic_l1_pyddx import (
    ATOMIC_L1_DDX_CAVITY_PROFILE_ID,
    ATOMIC_L1_DDX_PCM_PROFILE_ID,
    AtomicL1PyDDXPCMBackend,
)
from maple.solvation.continuum.functional import ContinuumEnergyFunctional
from maple.solvation.coupling.geometry_mediated import (
    GeometryMediatedElectrostaticScalar,
)
from maple.solvation.coupling.metrics import ATOMIC_L1_PAIRING
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)
from maple.solvation.models import (
    AIMNet2CheckpointContract,
    AIMNet2GeometryMediatedModelAdapter,
)

_ALPHA = 0.07
_BETA = 0.03


class _FakeAIMNet2:
    model_name = "aimnet2"
    _supports_atom_charges = True
    _coulomb_method = "simple"
    solvent_correction = None
    device = "cpu"
    cutoff = 5.0
    cutoff_lr = float("inf")

    def __init__(self, model_path):
        self.model_path = str(model_path)

    @staticmethod
    def _charges(atoms):
        x = np.asarray(atoms.positions, dtype=float)[:, 0]
        base = np.linspace(-0.4, 0.4, len(atoms))
        base -= float(np.mean(base))
        return base + _ALPHA * (x - float(np.mean(x)))

    def charge_state(self, atoms):
        charges = self._charges(atoms)
        return SimpleNamespace(
            energy_ev=0.5 * float(np.vdot(atoms.positions, atoms.positions)),
            charges_e=charges,
            requested_total_charge_e=0.0,
            model_name="aimnet2",
        )

    def charge_position_response(self, atoms, charge_cotangent_ev_per_e):
        cotangent = np.asarray(charge_cotangent_ev_per_e, dtype=float)
        charge_vjp = np.zeros_like(atoms.positions)
        charge_vjp[:, 0] = _ALPHA * (cotangent - float(np.mean(cotangent)))
        return SimpleNamespace(
            charge_state=self.charge_state(atoms),
            charge_cotangent_ev_per_e=cotangent.copy(),
            intrinsic_energy_gradient_ev_per_angstrom=np.array(
                atoms.positions, copy=True
            ),
            charge_position_vjp_ev_per_angstrom=charge_vjp,
        )

    def charge_position_second_order(
        self, atoms, charge_cotangent_ev_per_e, coordinate_direction
    ):
        response = self.charge_position_response(atoms, charge_cotangent_ev_per_e)
        direction = np.asarray(coordinate_direction, dtype=float)
        charge_jvp = _ALPHA * (direction[:, 0] - float(np.mean(direction[:, 0])))
        return SimpleNamespace(
            **vars(response),
            coordinate_direction=np.array(direction, copy=True),
            charge_position_jvp_e_per_angstrom=charge_jvp,
            intrinsic_energy_hvp_ev_per_angstrom2=np.array(direction, copy=True),
            contracted_charge_hessian_ev_per_angstrom2=np.zeros_like(direction),
            standard_decomposed_energy_absolute_error_ev=0.0,
            standard_decomposed_charge_max_absolute_error_e=0.0,
            standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom=0.0,
            standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom=0.0,
            charge_tangent_residual_e_per_angstrom=abs(float(np.sum(charge_jvp))),
        )


class _FakeReactionMap:
    cavity_topology_sha256 = "1" * 64
    cavity_active_node_pairs = ((0, 0), (1, 0))
    minimum_cavity_active_set_clearance_angstrom = 0.25

    def __init__(self, positions, radii, **kwargs):
        del radii, kwargs
        self.positions = np.asarray(positions, dtype=float)
        self.scale = 1.0 + _BETA * float(np.vdot(self.positions, self.positions))
        self.runtime_provenance = {"backend": "fake-pyddx"}

    def apply(self, source):
        return self.scale * ATOMIC_L1_PAIRING.source_to_field_dual(source)

    def adjoint(self, field_cotangent):
        return self.scale * ATOMIC_L1_PAIRING.field_to_source_dual(field_cotangent)

    def full_position_vjp(self, source, field_cotangent):
        contraction = float(
            np.vdot(ATOMIC_L1_PAIRING.field_to_source_dual(field_cotangent), source)
        )
        return 2.0 * _BETA * contraction * self.positions


class _ZeroChargeAIMNet2(_FakeAIMNet2):
    @staticmethod
    def _charges(atoms):
        return np.zeros(len(atoms))


class _BadGaugeAIMNet2(_FakeAIMNet2):
    def charge_position_response(self, atoms, charge_cotangent_ev_per_e):
        response = super().charge_position_response(atoms, charge_cotangent_ev_per_e)
        response.charge_position_vjp_ev_per_angstrom[:, 0] += float(
            np.mean(charge_cotangent_ev_per_e)
        )
        return response


class _SkewReactionMap(_FakeReactionMap):
    def __init__(self, positions, radii, **kwargs):
        super().__init__(positions, radii, **kwargs)
        count = len(self.positions)
        projection = np.eye(count) - np.ones((count, count)) / count
        seed = np.zeros((count, count))
        seed[0, 1] = 1.0
        seed[1, 0] = -1.0
        self.skew = 0.2 * projection @ seed @ projection

    def apply(self, source):
        result = super().apply(source)
        result[:, 0] += self.skew @ np.asarray(source)[:, 0]
        return result


class _QuadraticAtomicL1Functional(ContinuumEnergyFunctional):
    __slots__ = ("_configuration", "provenance_sha256")

    provider_id = "maple.route2.continuum.test-quadratic-atomic-l1.impl.v1"
    continuum_profile_id = ATOMIC_L1_DDX_PCM_PROFILE_ID
    cavity_profile_id = ATOMIC_L1_DDX_CAVITY_PROFILE_ID
    configuration_contract_id = UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID
    coupling_id = AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID
    scalar_id = DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_ELECTROSTATIC_V1
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE
    pairing = ATOMIC_L1_PAIRING
    linear_response = True
    reciprocal = True

    def __init__(self, *, dtype, device):
        super().__init__(
            source_space=self.source_space,
            field_space=self.field_space,
            pairing=self.pairing,
            dtype=dtype,
            device=device,
            expected_atomic_numbers=(6, 8),
        )
        configuration = hashlib.sha256(b"test-quadratic-atomic-l1-v1").hexdigest()
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(
            self,
            "provenance_sha256",
            hashlib.sha256(configuration.encode()).hexdigest(),
        )

    def configuration_sha256(self):
        return self._configuration

    def _energy_torch(self, positions, source):
        scale = 1.0 + _BETA * (positions * positions).sum()
        return 0.5 * scale * (source * source).sum()

    def evaluate_field(self, geometry, source):
        return self.drive(geometry, source)

    field = evaluate_field

    def coordinate_vjp(self, geometry, source, field_cotangent):
        return self.mixed_coordinate_source_vjp(
            geometry, source, field_cotangent
        ).reshape(-1)

    def adjoint(self, field_cotangent):
        result = super().adjoint(field_cotangent)
        result[:, 0] += self.skew.T @ np.asarray(field_cotangent)[:, 0]
        return result


def _atoms():
    return Atoms(
        "CO",
        positions=[[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]],
        info={"charge": 0, "mult": 1},
    )


def _three_atoms():
    return Atoms(
        "CON",
        positions=[
            [0.1, -0.2, 0.3],
            [1.2, 0.4, -0.1],
            [-0.4, 0.7, 0.5],
        ],
        info={"charge": 0, "mult": 1},
    )


def _scalar(
    tmp_path,
    *,
    atoms=None,
    calculator_type=_FakeAIMNet2,
    map_factory=_FakeReactionMap,
):
    atoms = _atoms() if atoms is None else atoms
    model = _model_adapter(tmp_path, calculator_type=calculator_type)
    continuum = AtomicL1PyDDXPCMBackend(
        atoms,
        np.full(len(atoms), 1.6),
        dielectric=20.0,
        lmax=7,
        n_lebedev=302,
        solver_tolerance=1.0e-12,
        _map_factory=map_factory,
    )
    return GeometryMediatedElectrostaticScalar(model, continuum)


def _model_adapter(tmp_path, *, calculator_type=_FakeAIMNet2):
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"test geometry-mediated checkpoint\n")
    contract = AIMNet2CheckpointContract(
        provider_id="maple.route2.model.test-aimnet2-geometry-mediated.impl.v1",
        model_profile_id=AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
        checkpoint_identifier="test",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        model_name="aimnet2",
        coulomb_method="simple",
        inference_dtype="float32",
        supported_atomic_numbers=(1, 6, 7, 8),
        upstream_repository="test",
        publication_doi="10.1039/D4SC08572H",
        upstream_version="test",
        upstream_commit="test",
        checkpoint_origin_status="test",
    )
    return AIMNet2GeometryMediatedModelAdapter(calculator_type(checkpoint), contract)


def _hvp_scalar(tmp_path):
    torch = pytest.importorskip("torch")
    return GeometryMediatedElectrostaticScalar(
        _model_adapter(tmp_path),
        _QuadraticAtomicL1Functional(dtype=torch.float64, device="cpu"),
    )


def test_geometry_mediated_scalar_gradient_matches_its_complete_finite_difference(
    tmp_path,
):
    scalar = _scalar(tmp_path)
    atoms = _atoms()
    result = scalar.evaluate(atoms)
    step = 1.0e-5
    numerical = np.zeros_like(atoms.positions)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, axis] += step
            minus.positions[atom_index, axis] -= step
            numerical[atom_index, axis] = (
                scalar.evaluate_energy(plus) - scalar.evaluate_energy(minus)
            ) / (2.0 * step)

    np.testing.assert_allclose(
        result.total_gradient_eV_per_A,
        numerical,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(result.intrinsic_gradient_eV_per_A, atoms.positions)
    assert np.linalg.norm(result.source_response_gradient_eV_per_A) > 0.0
    assert np.linalg.norm(result.continuum_fixed_source_gradient_eV_per_A) > 0.0
    assert result.reciprocity_gradient_error_eV_per_source_unit < 1.0e-14
    assert result.reciprocity_audit.gate_passed is True
    assert len(result.reciprocity_audit.bilinear_records) == 4
    assert len(result.reciprocity_audit.charge_directional_fd_records) == 12
    assert result.reciprocity_audit.charge_gauge_vjp_norm_eV_per_A == 0.0
    assert result.energy.total_energy_eV == pytest.approx(scalar.evaluate_energy(atoms))
    assert result.total_gradient_eV_per_A.flags.writeable is False
    assert result.forces_eV_per_A.flags.writeable is False


def test_geometry_mediated_profile_is_registered_but_has_no_public_capability(tmp_path):
    scalar = _scalar(tmp_path)
    profile = PROFILE_REGISTRY[
        DIAGNOSTIC_AIMNET2_GEOMETRY_MEDIATED_DDX_DDPCM_PROFILE_V1
    ]
    definition = SCALAR_REGISTRY[profile.scalar_id]
    assert profile.enabled is False
    assert profile.capabilities.enabled_tiers == ()
    assert definition.enabled is False
    assert definition.admitted_capabilities.enabled_tiers == ()
    assert "fixed_geometry_electronic_mutual_polarization" in (
        definition.excluded_components
    )
    assert "nonpolar_smd_cds" in definition.excluded_components
    assert scalar.continuum.fixed_topology is False
    assert len(scalar.fingerprint_sha256()) == 64


def test_geometry_mediated_scalar_rejects_skew_reaction_operator_even_at_zero_source(
    tmp_path,
):
    atoms = _three_atoms()
    scalar = _scalar(
        tmp_path,
        atoms=atoms,
        calculator_type=_ZeroChargeAIMNet2,
        map_factory=_SkewReactionMap,
    )
    assert scalar.evaluate_energy(atoms) == pytest.approx(
        0.5 * float(np.vdot(atoms.positions, atoms.positions))
    )
    with pytest.raises(ValueError, match="metric/reciprocity/charge-gauge"):
        scalar.evaluate(atoms)


def test_geometry_mediated_scalar_rejects_charge_projection_with_nonzero_gauge_vjp(
    tmp_path,
):
    atoms = _three_atoms()
    scalar = _scalar(tmp_path, atoms=atoms, calculator_type=_BadGaugeAIMNet2)
    with pytest.raises(ValueError, match="metric/reciprocity/charge-gauge"):
        scalar.evaluate(atoms)


def test_geometry_mediated_complete_hvp_ledger_matches_analytic_and_force_fd(
    tmp_path,
):
    scalar = _hvp_scalar(tmp_path)
    atoms = _atoms()
    direction = np.asarray([[0.3, -0.2, 0.4], [-0.1, 0.5, -0.6]])
    result = scalar.hessian_vector_product(atoms, direction)

    positions = np.asarray(atoms.positions, dtype=float)
    source = np.asarray(result.source, dtype=float)
    source_jvp = np.zeros_like(source)
    source_jvp[:, 0] = _ALPHA * (direction[:, 0] - float(np.mean(direction[:, 0])))
    scale = 1.0 + _BETA * float(np.vdot(positions, positions))
    source_norm2 = float(np.vdot(source, source))
    source_direction_pair = float(np.vdot(source, source_jvp))
    expected_continuum_position = (
        _BETA * source_norm2 * direction
        + 2.0 * _BETA * source_direction_pair * positions
    )
    expected_continuum_source = (
        scale * source_jvp + 2.0 * _BETA * float(np.vdot(positions, direction)) * source
    )
    expected_pullback = np.zeros_like(direction)
    expected_pullback[:, 0] = _ALPHA * (
        expected_continuum_source[:, 0]
        - float(np.mean(expected_continuum_source[:, 0]))
    )
    expected_total = direction + expected_continuum_position + expected_pullback

    np.testing.assert_allclose(result.source_position_jvp, source_jvp, atol=1e-15)
    np.testing.assert_allclose(
        result.intrinsic_energy_hvp_eV_per_A2, direction, atol=1e-15
    )
    np.testing.assert_allclose(
        result.continuum_joint_position_hvp_eV_per_A2,
        expected_continuum_position,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.continuum_joint_source_hvp,
        expected_continuum_source,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.continuum_source_response_pullback_eV_per_A2,
        expected_pullback,
        atol=2e-15,
    )
    np.testing.assert_array_equal(result.contracted_source_hessian_eV_per_A2, 0.0)
    np.testing.assert_allclose(result.total_hvp_eV_per_A2, expected_total, atol=3e-15)

    step = 1.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * direction
    minus.positions -= step * direction
    gradient_fd = (
        scalar.evaluate(plus).total_gradient_eV_per_A
        - scalar.evaluate(minus).total_gradient_eV_per_A
    ) / (2.0 * step)
    np.testing.assert_allclose(result.total_hvp_eV_per_A2, gradient_fd, atol=3e-10)
    assert result.diagnostic_only is True
    assert result.tier_h_admitted is False
    assert result.total_hvp_eV_per_A2.flags.writeable is False


def test_geometry_mediated_hvp_is_symmetric_but_does_not_admit_tier_h(tmp_path):
    scalar = _hvp_scalar(tmp_path)
    atoms = _atoms()
    first = np.asarray([[0.3, -0.2, 0.4], [-0.1, 0.5, -0.6]])
    second = np.asarray([[-0.4, 0.1, 0.2], [0.6, -0.3, 0.5]])
    h_first = scalar.hessian_vector_product(atoms, first).total_hvp_eV_per_A2
    h_second = scalar.hessian_vector_product(atoms, second).total_hvp_eV_per_A2
    assert np.vdot(first, h_second) == pytest.approx(
        np.vdot(second, h_first), abs=2e-14
    )
    profile = PROFILE_REGISTRY[scalar.profile_id]
    assert profile.capabilities.hessian is False


def test_geometry_mediated_hvp_rejects_nonsealed_continuum_backend(tmp_path):
    scalar = _scalar(tmp_path)
    with pytest.raises(NotImplementedError, match="sealed.*joint Hessian"):
        scalar.hessian_vector_product(_atoms(), np.ones((2, 3)))
