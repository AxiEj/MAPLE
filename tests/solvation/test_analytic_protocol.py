from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    AnalyticOperationResult,
    CheckpointFileIdentity,
    CompositePointIdentity,
    ScalarCellIdentity,
    ScalarPointIdentity,
    compose_energy_forces,
    compose_hessian,
    compose_hvp,
    freeze_identity_options,
    geometry_sha256,
    named_array_sha256,
    require_no_amp_tf32,
    torch_model_runtime_identity,
    torch_response_settings,
)
from maple.function.calculator.extra_correction.implicit.result import (
    SolvationDirectionalResult,
    SolvationResult,
)


def _cell(component: str, *, checkpoint_sha256: str | None = None):
    return ScalarCellIdentity(
        component=component,
        implementation="quadratic-gas-v1" if component == "gas" else "obc2-test-v1",
        model="quadratic" if component == "gas" else "obc2-mbondi2",
        checkpoint_path="/models/quadratic.pt" if component == "gas" else None,
        checkpoint_sha256=checkpoint_sha256 if component == "gas" else None,
        device="cpu",
        effective_dtype="torch.float64",
        d4=False,
        source_energy_unit="hartree",
        cutoff_guard_policy=(
            "neighbor-cutoff-guard-v1" if component == "gas" else None
        ),
        cutoff_guard_margin_angstrom=0.003 if component == "gas" else None,
        options=freeze_identity_options(
            {"profile": "gas-test" if component == "gas" else "obc2-none"}
        ),
    )


def _point(
    atoms: Atoms,
    cell: ScalarCellIdentity,
    *,
    geometry: str | None = None,
    graph: str | None = None,
):
    return ScalarPointIdentity(
        cell=cell,
        geometry_sha256=geometry or geometry_sha256(atoms),
        atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers()),
        charge=float(atoms.info.get("charge", 0.0)),
        multiplicity=int(atoms.info.get("mult", 1)),
        atom_topology_sha256=None,
        neighbor_graph_sha256=graph,
        active_cutoffs_angstrom=(('radial', 5.1),) if cell.component == "gas" else (),
        minimum_cutoff_gap_angstrom=1.25 if cell.component == "gas" else None,
        response_settings=freeze_identity_options({"autocast": False, "tf32": False}),
    )


def _atoms() -> Atoms:
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.1, -0.2]],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


def test_identity_helpers_are_canonical_and_geometry_sensitive():
    first = freeze_identity_options(
        {"z": [2, 1], "a": {"enabled": True, "threshold": 0.003}}
    )
    second = freeze_identity_options(
        {"a": {"threshold": 0.003, "enabled": True}, "z": [2, 1]}
    )
    assert first == second

    arrays_a = {
        "edge_index": np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        "shifts": np.zeros((2, 3), dtype=np.float64),
    }
    arrays_b = {"shifts": arrays_a["shifts"], "edge_index": arrays_a["edge_index"]}
    assert named_array_sha256(arrays_a) == named_array_sha256(arrays_b)
    changed = dict(arrays_a)
    changed["edge_index"] = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    assert named_array_sha256(changed) != named_array_sha256(arrays_a)

    atoms = _atoms()
    same = atoms.copy()
    moved = atoms.copy()
    moved.positions[0, 0] += 1.0e-12
    reordered = atoms[[1, 0]]
    assert geometry_sha256(atoms) == geometry_sha256(same)
    assert geometry_sha256(atoms) != geometry_sha256(moved)
    assert geometry_sha256(atoms) != geometry_sha256(reordered)


def test_checkpoint_binding_detects_replacement_and_torch_flags_are_frozen(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"first checkpoint bytes")
    identity = CheckpointFileIdentity.bind(checkpoint)
    identity.require_unchanged()
    assert identity.sha256 == "a042907e1a9e895544e147308ff94617fa190bd7df84d86ae777120abc7c6078"

    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"replacement checkpoint")
    replacement.replace(checkpoint)
    with pytest.raises(AnalyticIdentityError, match="changed"):
        identity.require_unchanged()

    torch = pytest.importorskip("torch")
    settings = torch_response_settings(torch.device("cpu"))
    assert dict(settings)["device_type"] == '"cpu"'
    require_no_amp_tf32(torch.device("cpu"))

    model = torch.nn.Linear(2, 1)
    before = torch_model_runtime_identity(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    after = torch_model_runtime_identity(model)
    assert before != after


def test_identity_and_operation_values_are_immutable_and_raw_hessian_is_preserved():
    atoms = _atoms()
    cell = _cell("gas", checkpoint_sha256="a" * 64)
    point = _point(atoms, cell, graph="b" * 64)
    raw = np.zeros((6, 6), dtype=np.float64)
    raw[0, 1] = 2.0

    result = AnalyticOperationResult.hessian(
        identity=point,
        hessian_hartree_per_angstrom2=raw,
    )
    raw[0, 1] = 99.0

    assert result.hessian_hartree_per_angstrom2[0, 1] == 2.0
    assert result.hessian_hartree_per_angstrom2[1, 0] == 0.0
    with pytest.raises(ValueError):
        result.hessian_hartree_per_angstrom2[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        point.charge = 1.0


def test_operation_result_enforces_kind_specific_values_shapes_and_finiteness():
    atoms = _atoms()
    point = _point(atoms, _cell("gas", checkpoint_sha256="a" * 64))

    with pytest.raises(ValueError, match="energy_forces.*hessian"):
        AnalyticOperationResult(
            operation="energy_forces",
            identity=point,
            energy_hartree=0.0,
            hessian_hartree_per_angstrom2=np.eye(6),
        )
    with pytest.raises(ValueError, match=r"forces.*\(N, 3\)"):
        AnalyticOperationResult.energy_forces(
            identity=point,
            energy_hartree=0.0,
            forces_hartree_per_angstrom=np.zeros(6),
        )
    with pytest.raises(ValueError, match="finite"):
        AnalyticOperationResult.hessian(
            identity=point,
            hessian_hartree_per_angstrom2=np.full((6, 6), np.nan),
        )
    with pytest.raises(ValueError, match="direction.*3N"):
        AnalyticOperationResult.hvp(
            identity=point,
            direction=np.zeros(5),
            hvp_hartree_per_angstrom2=np.zeros(6),
            forces_hartree_per_angstrom=np.zeros((2, 3)),
            energy_hartree=0.0,
        )


def test_component_admission_requires_exact_cell_and_operation():
    atoms = _atoms()
    admitted_cell = _cell("gas", checkpoint_sha256="a" * 64)
    admission = AnalyticDerivativeAdmission(
        profile_id="test-gas-qualified-v1",
        evidence_id="test-evidence-v1",
        cell=admitted_cell,
        operations=frozenset({"energy_forces", "hessian"}),
    )
    admission.require(_point(atoms, admitted_cell), "hessian")

    drifted = _cell("gas", checkpoint_sha256="c" * 64)
    with pytest.raises(AnalyticIdentityError, match="cell identity"):
        admission.require(_point(atoms, drifted), "hessian")
    with pytest.raises(AnalyticCapabilityError, match="hvp"):
        admission.require(_point(atoms, admitted_cell), "hvp")


def test_exact_composition_closes_energy_force_hessian_and_direct_hvp_once():
    atoms = _atoms()
    gas_point = _point(
        atoms,
        _cell("gas", checkpoint_sha256="a" * 64),
        graph="b" * 64,
    )
    solvent_point = _point(atoms, _cell("solvent"))
    gas_ef = AnalyticOperationResult.energy_forces(
        identity=gas_point,
        energy_hartree=1.25,
        forces_hartree_per_angstrom=np.full((2, 3), 0.5),
    )
    solvent_ef = SolvationResult(
        energy_hartree=-0.25,
        forces_hartree_per_angstrom=np.full((2, 3), -0.125),
        components_hartree={"polar": -0.25, "nonpolar": 0.0},
        provenance={"provider": "test-obc2"},
    )
    composed_ef = compose_energy_forces(gas_ef, solvent_ef, solvent_point)
    assert isinstance(composed_ef.identity, CompositePointIdentity)
    assert composed_ef.energy_hartree == 1.0
    np.testing.assert_array_equal(
        composed_ef.forces_hartree_per_angstrom,
        np.full((2, 3), 0.375),
    )

    gas_h = AnalyticOperationResult.hessian(
        identity=gas_point,
        hessian_hartree_per_angstrom2=np.eye(6) * 2.0,
    )
    solvent_h = np.eye(6) * 0.25
    composed_h = compose_hessian(gas_h, solvent_h, solvent_point)
    np.testing.assert_array_equal(
        composed_h.hessian_hartree_per_angstrom2,
        np.eye(6) * 2.25,
    )

    direction = np.arange(1.0, 7.0)
    gas_hvp = AnalyticOperationResult.hvp(
        identity=gas_point,
        direction=direction,
        hvp_hartree_per_angstrom2=2.0 * direction,
        forces_hartree_per_angstrom=np.full((2, 3), 0.5),
        energy_hartree=1.25,
    )
    solvent_hvp = SolvationDirectionalResult(
        hvp_hartree_per_angstrom2=0.25 * direction,
        forces_hartree_per_angstrom=np.full((2, 3), -0.125),
        energy_hartree=-0.25,
        provenance={"provider": "test-obc2"},
    )
    composed_hvp = compose_hvp(gas_hvp, solvent_hvp, solvent_point)
    np.testing.assert_array_equal(
        composed_hvp.hvp_hartree_per_angstrom2,
        2.25 * direction,
    )
    np.testing.assert_array_equal(
        composed_hvp.forces_hartree_per_angstrom,
        np.full((2, 3), 0.375),
    )
    assert composed_hvp.energy_hartree == 1.0


def test_composition_rejects_point_drift_before_arithmetic():
    atoms = _atoms()
    gas_point = _point(atoms, _cell("gas", checkpoint_sha256="a" * 64))
    moved = atoms.copy()
    moved.positions[0, 0] += 0.01
    solvent_point = _point(moved, _cell("solvent"))
    gas_h = AnalyticOperationResult.hessian(
        identity=gas_point,
        hessian_hartree_per_angstrom2=np.eye(6),
    )

    with pytest.raises(AnalyticIdentityError, match="geometry|center"):
        compose_hessian(gas_h, np.eye(6), solvent_point)


def test_composition_rejects_broadcastable_or_nonreal_solvent_values():
    atoms = _atoms()
    gas_point = _point(atoms, _cell("gas", checkpoint_sha256="a" * 64))
    solvent_point = _point(atoms, _cell("solvent"))
    gas_ef = AnalyticOperationResult.energy_forces(
        identity=gas_point,
        energy_hartree=1.0,
        forces_hartree_per_angstrom=np.zeros((2, 3)),
    )
    with pytest.raises(ValueError, match=r"Solvent forces.*\(N, 3\)"):
        compose_energy_forces(
            gas_ef,
            SolvationResult(
                energy_hartree=0.0,
                forces_hartree_per_angstrom=np.ones(3),
            ),
            solvent_point,
        )
    with pytest.raises(TypeError, match="bool"):
        compose_energy_forces(
            gas_ef,
            SolvationResult(
                energy_hartree=True,
                forces_hartree_per_angstrom=np.zeros((2, 3)),
            ),
            solvent_point,
        )

    direction = np.ones(6)
    gas_hvp = AnalyticOperationResult.hvp(
        identity=gas_point,
        direction=direction,
        hvp_hartree_per_angstrom2=direction,
        forces_hartree_per_angstrom=np.zeros((2, 3)),
        energy_hartree=1.0,
    )
    with pytest.raises(ValueError, match="Solvent HVP.*3N"):
        compose_hvp(
            gas_hvp,
            SolvationDirectionalResult(
                hvp_hartree_per_angstrom2=np.ones(1),
                forces_hartree_per_angstrom=np.zeros((2, 3)),
                energy_hartree=0.0,
            ),
            solvent_point,
        )
    with pytest.raises(TypeError, match="real"):
        compose_hvp(
            gas_hvp,
            SolvationDirectionalResult(
                hvp_hartree_per_angstrom2=np.ones(6, dtype=np.complex128),
                forces_hartree_per_angstrom=np.zeros((2, 3)),
                energy_hartree=0.0,
            ),
            solvent_point,
        )


def test_composite_admission_binds_both_exact_component_admissions():
    atoms = _atoms()
    gas_cell = _cell("gas", checkpoint_sha256="a" * 64)
    solvent_cell = _cell("solvent")
    gas = AnalyticDerivativeAdmission(
        profile_id="gas-v1",
        evidence_id="gas-evidence-v1",
        cell=gas_cell,
        operations=frozenset({"energy_forces", "hessian", "hvp"}),
    )
    solvent = AnalyticDerivativeAdmission(
        profile_id="solvent-v1",
        evidence_id="solvent-evidence-v1",
        cell=solvent_cell,
        operations=frozenset({"energy_forces", "hessian", "hvp"}),
    )
    admission = AnalyticCompositionAdmission(
        profile_id="gas-plus-solvent-v1",
        evidence_id="composition-evidence-v1",
        gas_admission=gas,
        solvent_admission=solvent,
        operations=frozenset({"energy_forces", "hessian", "hvp"}),
    )

    admission.require(_point(atoms, gas_cell), _point(atoms, solvent_cell), "hvp")
    with pytest.raises(AnalyticIdentityError, match="gas.*cell identity"):
        admission.require(
            _point(atoms, _cell("gas", checkpoint_sha256="f" * 64)),
            _point(atoms, solvent_cell),
            "hvp",
        )
