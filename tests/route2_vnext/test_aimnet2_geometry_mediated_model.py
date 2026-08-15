from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.api.profiles import (
    AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
    AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
)
from maple.solvation.models import (
    AIMNet2CheckpointContract,
    AIMNet2GeometryMediatedModelAdapter,
    validate_response_linearization,
    validate_source_evaluation,
    validate_vacuum_evaluation,
)

_ALPHA = 0.07


class _FakeAIMNet2Calculator:
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
        x = np.asarray(atoms.get_positions(), dtype=float)[:, 0]
        base = np.linspace(-0.4, 0.4, len(atoms))
        base -= np.mean(base)
        return base + _ALPHA * (x - float(np.mean(x)))

    def charge_state(self, atoms):
        charges = self._charges(atoms)
        return SimpleNamespace(
            energy_ev=0.5 * float(np.vdot(atoms.positions, atoms.positions)),
            charges_e=charges,
            requested_total_charge_e=0.0,
            model_name=self.model_name,
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


class _NoSecondOrderAIMNet2Calculator(_FakeAIMNet2Calculator):
    charge_position_second_order = None


def _atoms():
    return Atoms(
        "CO",
        positions=[[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]],
        info={"charge": 0, "mult": 1},
    )


def _adapter(tmp_path, calculator_type=_FakeAIMNet2Calculator):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"test AIMNet2 checkpoint\n")
    contract = AIMNet2CheckpointContract(
        provider_id="maple.route2.model.test-aimnet2-geometry-mediated.impl.v1",
        model_profile_id=AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
        checkpoint_identifier="test-aimnet2",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        model_name="aimnet2",
        coulomb_method="simple",
        inference_dtype="float32",
        supported_atomic_numbers=(1, 6, 7, 8),
        upstream_repository="https://example.invalid/aimnet2",
        publication_doi="10.1039/D4SC08572H",
        upstream_version="test",
        upstream_commit="test",
        checkpoint_origin_status="test-only",
    )
    calculator = calculator_type(checkpoint)
    return AIMNet2GeometryMediatedModelAdapter(calculator, contract), calculator


def test_aimnet2_adapter_embeds_geometry_charges_and_exposes_vacuum_gradient(
    tmp_path,
):
    adapter, calculator = _adapter(tmp_path)
    atoms = _atoms()
    field = np.linspace(-0.2, 0.3, 8).reshape(2, 4)

    source = validate_source_evaluation(
        adapter, atoms, field, need_fixed_field_forces=True
    )
    charges = calculator._charges(atoms)
    np.testing.assert_allclose(source.source[:, 0], charges)
    np.testing.assert_array_equal(source.source[:, 1:], 0.0)
    np.testing.assert_allclose(source.fixed_field_forces_eV_per_A, -atoms.positions)
    assert adapter.field_independent is True
    assert adapter.electronic_mutual_polarization is False
    assert adapter.coupling_id == AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID
    assert adapter.model_profile_id == AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID

    vacuum = validate_vacuum_evaluation(adapter, atoms, need_forces=True)
    assert vacuum.energy_eV == pytest.approx(
        0.5 * np.vdot(atoms.positions, atoms.positions)
    )
    np.testing.assert_allclose(vacuum.forces_eV_per_A, -atoms.positions)
    assert source.source.flags.writeable is False
    assert vacuum.forces_eV_per_A.flags.writeable is False


def test_aimnet2_adapter_has_zero_field_response_but_complete_charge_position_vjp(
    tmp_path,
):
    adapter, _ = _adapter(tmp_path)
    atoms = _atoms()
    field = np.linspace(-0.1, 0.2, 8).reshape(2, 4)
    direction = np.linspace(0.3, -0.2, 8).reshape(2, 4)
    cotangent = np.linspace(-0.4, 0.5, 8).reshape(2, 4)

    jvp, vjp, position_vjp = validate_response_linearization(
        adapter,
        atoms,
        field,
        field_direction=direction,
        source_cotangent=cotangent,
    )
    np.testing.assert_array_equal(jvp, 0.0)
    np.testing.assert_array_equal(vjp, 0.0)
    expected = np.zeros((2, 3))
    expected[:, 0] = _ALPHA * (cotangent[:, 0] - float(np.mean(cotangent[:, 0])))
    np.testing.assert_allclose(position_vjp, expected)

    changed_only_in_unused_l1 = cotangent.copy()
    changed_only_in_unused_l1[:, 0] = 0.0
    np.testing.assert_array_equal(
        adapter.source_position_vjp(atoms, field, changed_only_in_unused_l1), 0.0
    )


def test_aimnet2_adapter_exposes_complete_point_l0_second_order_ledger(tmp_path):
    adapter, _ = _adapter(tmp_path)
    atoms = _atoms()
    field = np.linspace(-0.1, 0.2, 8).reshape(2, 4)
    cotangent = np.linspace(-0.4, 0.5, 8).reshape(2, 4)
    direction = np.asarray([[0.3, -0.2, 0.4], [-0.1, 0.5, -0.6]])

    result = adapter.source_position_second_order(
        atoms,
        field,
        cotangent,
        direction,
    )
    expected_jvp = _ALPHA * (direction[:, 0] - float(np.mean(direction[:, 0])))
    expected_vjp = np.zeros((2, 3))
    expected_vjp[:, 0] = _ALPHA * (cotangent[:, 0] - float(np.mean(cotangent[:, 0])))
    np.testing.assert_allclose(result.source_position_jvp[:, 0], expected_jvp)
    np.testing.assert_array_equal(result.source_position_jvp[:, 1:], 0.0)
    np.testing.assert_allclose(result.source_position_vjp_eV_per_A, expected_vjp)
    np.testing.assert_allclose(result.intrinsic_energy_hvp_eV_per_A2, direction)
    np.testing.assert_array_equal(result.contracted_source_hessian_eV_per_A2, 0.0)
    np.testing.assert_array_equal(result.source_cotangent, cotangent)
    assert np.sum(result.source_position_jvp[:, 0]) == pytest.approx(0.0, abs=1e-15)
    assert result.standard_decomposed_energy_absolute_error_eV == 0.0
    assert result.source_position_jvp.flags.writeable is False


def test_aimnet2_adapter_second_order_path_fails_closed_when_runtime_lacks_it(
    tmp_path,
):
    adapter, _ = _adapter(tmp_path, _NoSecondOrderAIMNet2Calculator)
    atoms = _atoms()
    zeros = np.zeros((2, 4))
    with pytest.raises(NotImplementedError, match="no source-bound second-order"):
        adapter.source_position_second_order(atoms, zeros, zeros, atoms.positions)


def test_aimnet2_neighbor_topology_is_rigid_invariant_and_cutoff_event_sensitive(
    tmp_path,
):
    adapter, _ = _adapter(tmp_path)
    atoms = _atoms()
    base = adapter.neighbor_topology(atoms)
    translated = atoms.copy()
    translated.positions += np.asarray([1.2, -0.7, 0.4])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    separated = atoms.copy()
    separated.positions[1] += np.asarray([10.0, 0.0, 0.0])

    assert adapter.neighbor_topology(translated).topology_sha256 == (
        base.topology_sha256
    )
    assert adapter.neighbor_topology(rotated).topology_sha256 == base.topology_sha256
    changed = adapter.neighbor_topology(separated)
    assert changed.topology_sha256 != base.topology_sha256
    assert base.minimum_cutoff_margin_angstrom is not None
    assert base.minimum_cutoff_margin_angstrom > 0.0
    assert base.as_dict()["active_pairs_by_graph"] == {
        "short_range": [[0, 1]],
        "long_range_effective": [[0, 1]],
    }


def test_aimnet2_adapter_fails_closed_on_domain_solvent_and_checkpoint_drift(tmp_path):
    adapter, calculator = _adapter(tmp_path)
    atoms = _atoms()
    charged = atoms.copy()
    charged.info["charge"] = 1
    with pytest.raises(ValueError, match="outside model domain"):
        adapter.evaluate_vacuum(charged, need_forces=False)
    open_shell = atoms.copy()
    open_shell.info["mult"] = 3
    with pytest.raises(ValueError, match="spin multiplicity"):
        adapter.evaluate_vacuum(open_shell, need_forces=False)
    unsupported = Atoms(
        "NaH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], info=atoms.info
    )
    with pytest.raises(ValueError, match="atomic numbers outside model domain"):
        adapter.evaluate_vacuum(unsupported, need_forces=False)

    with pytest.raises(AttributeError, match="immutable"):
        adapter.device = "cuda"
    with open(calculator.model_path, "ab") as handle:
        handle.write(b"drift")
    with pytest.raises(RuntimeError, match="file identity changed"):
        adapter.configuration_sha256()

    clean_adapter, clean_calculator = _adapter(tmp_path / "second")
    clean_calculator.solvent_correction = object()
    with pytest.raises(ValueError, match="configuration drifted"):
        clean_adapter.configuration_sha256()

    cutoff_adapter, cutoff_calculator = _adapter(tmp_path / "third")
    cutoff_calculator.cutoff = 4.5
    with pytest.raises(ValueError, match="neighbor-cutoff.*drifted"):
        cutoff_adapter.configuration_sha256()


def test_aimnet2_adapter_rejects_internal_solvent_correction_at_construction(tmp_path):
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"test AIMNet2 checkpoint\n")
    contract = AIMNet2CheckpointContract(
        provider_id="test",
        model_profile_id=AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
        checkpoint_identifier="test",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        model_name="aimnet2",
        coulomb_method="simple",
        inference_dtype="float32",
        supported_atomic_numbers=(1, 6, 7, 8),
        upstream_repository="test",
        publication_doi="test",
        upstream_version="test",
        upstream_commit="test",
        checkpoint_origin_status="test",
    )
    calculator = _FakeAIMNet2Calculator(checkpoint)
    calculator.solvent_correction = object()
    with pytest.raises(ValueError, match="must not include"):
        AIMNet2GeometryMediatedModelAdapter(calculator, contract)


def test_aimnet2_adapter_binds_and_rechecks_inference_dtype(tmp_path):
    adapter, calculator = _adapter(tmp_path)
    calculator.inference_dtype = "float64"

    with pytest.raises(ValueError, match="configuration drifted"):
        adapter.configuration_sha256()

    with pytest.raises(ValueError, match="inference dtype"):
        AIMNet2GeometryMediatedModelAdapter(calculator, adapter.checkpoint_contract)


def test_aimnet2_adapter_binds_optional_runtime_provenance(tmp_path):
    baseline, original = _adapter(tmp_path)
    calculator = _FakeAIMNet2Calculator(original.model_path)
    runtime = {"runtime_kind": "test", "source_sha256": "1" * 64}
    calculator.runtime_provenance = lambda: dict(runtime)
    adapter = AIMNet2GeometryMediatedModelAdapter(
        calculator, baseline.checkpoint_contract
    )

    runtime["source_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="runtime provenance drifted"):
        adapter.configuration_sha256()
