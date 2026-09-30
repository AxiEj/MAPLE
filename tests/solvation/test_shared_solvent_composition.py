from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.analytic_protocol import (
    AnalyticCapabilityError,
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    AnalyticOperationResult,
    ScalarCellIdentity,
    ScalarPointIdentity,
    freeze_identity_options,
    geometry_sha256,
)
from maple.function.calculator.extra_correction.implicit.result import (
    SolvationDirectionalResult,
    SolvationResult,
)
from maple.function.calculator.solvent_composition import (
    compose_analytic_hessian,
    compose_direct_hvp,
    finalize_energy_forces,
)


def _cell(component: str) -> ScalarCellIdentity:
    return ScalarCellIdentity(
        component=component,
        implementation=f"fixture-{component}-v1",
        model="quadratic" if component == "gas" else "obc2-mbondi2",
        checkpoint_path="/fixture/model.pt" if component == "gas" else None,
        checkpoint_sha256="a" * 64 if component == "gas" else None,
        device="cpu",
        effective_dtype="torch.float64",
        d4=False,
        source_energy_unit="hartree",
        cutoff_guard_policy=None,
        cutoff_guard_margin_angstrom=None,
        options=freeze_identity_options({"fixture": True}),
    )


def _point(atoms, cell, *, displaced: bool = False) -> ScalarPointIdentity:
    geometry = geometry_sha256(atoms)
    if displaced:
        moved = atoms.copy()
        moved.positions[0, 0] += 0.01
        geometry = geometry_sha256(moved)
    return ScalarPointIdentity(
        cell=cell,
        geometry_sha256=geometry,
        atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers()),
        charge=float(atoms.info.get("charge", 0.0)),
        multiplicity=int(atoms.info.get("mult", 1)),
        atom_topology_sha256=None,
        neighbor_graph_sha256=None,
        active_cutoffs_angstrom=(),
        minimum_cutoff_gap_angstrom=None,
        response_settings=freeze_identity_options({"fixture": True}),
    )


def _atoms() -> Atoms:
    atoms = Atoms("H2", positions=[[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]])
    atoms.info.update(charge=0, mult=1)
    return atoms


class _Correction:
    supported_properties = frozenset({"energy", "forces", "hessian", "hvp"})

    def __init__(self, atoms, *, force_constant: float = 0.25):
        self.force_constant = force_constant
        self.cell = _cell("solvent")
        self.analytic_derivative_admission = AnalyticDerivativeAdmission(
            profile_id="fixture-solvent-v1",
            evidence_id="fixture-solvent-evidence-v1",
            cell=self.cell,
            operations=frozenset({"energy_forces", "hessian", "hvp"}),
        )
        self.evaluate_calls = 0
        self.hessian_calls = 0
        self.directional_calls = 0
        self.drift = False
        self.bound_atom_count = len(atoms)

    def analytic_scalar_identity(self, atoms):
        return _point(atoms, self.cell, displaced=self.drift)

    def evaluate(self, atoms, *, need_forces=False, calculator=None):
        assert calculator is None, "the shared layer must not receive a calculator owner"
        self.evaluate_calls += 1
        positions = np.asarray(atoms.get_positions())
        energy = 0.5 * self.force_constant * float(np.square(positions).sum())
        return SolvationResult(
            energy_hartree=energy,
            forces_hartree_per_angstrom=(
                -self.force_constant * positions if need_forces else None
            ),
            components_hartree={"polar": energy, "nonpolar": 0.0},
            provenance={"provider": "fixture-obc2"},
        )

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        return np.eye(3 * len(atoms)) * self.force_constant

    def get_directional_derivatives(self, atoms, direction):
        self.directional_calls += 1
        positions = np.asarray(atoms.get_positions())
        direction = np.asarray(direction).reshape(-1)
        return SolvationDirectionalResult(
            hvp_hartree_per_angstrom2=self.force_constant * direction,
            forces_hartree_per_angstrom=-self.force_constant * positions,
            energy_hartree=(
                0.5 * self.force_constant * float(np.square(positions).sum())
            ),
            provenance={"provider": "fixture-obc2"},
        )


def _admissions(atoms):
    gas_cell = _cell("gas")
    gas_admission = AnalyticDerivativeAdmission(
        profile_id="fixture-gas-v1",
        evidence_id="fixture-gas-evidence-v1",
        cell=gas_cell,
        operations=frozenset({"energy_forces", "hessian", "hvp"}),
    )
    correction = _Correction(atoms)
    composition = AnalyticCompositionAdmission(
        profile_id="fixture-composition-v1",
        evidence_id="fixture-composition-evidence-v1",
        gas_admission=gas_admission,
        solvent_admission=correction.analytic_derivative_admission,
        operations=frozenset({"energy_forces", "hessian", "hvp"}),
    )
    return gas_cell, gas_admission, correction, composition


def _gas_operations(atoms):
    gas_cell, gas_admission, correction, composition = _admissions(atoms)
    point = _point(atoms, gas_cell)
    positions = np.asarray(atoms.get_positions())
    energy = float(np.square(positions).sum())
    direction = np.arange(1.0, 3 * len(atoms) + 1.0)
    return (
        gas_admission,
        correction,
        composition,
        AnalyticOperationResult.energy_forces(
            identity=point,
            energy_hartree=energy,
            forces_hartree_per_angstrom=-2.0 * positions,
        ),
        AnalyticOperationResult.hessian(
            identity=point,
            hessian_hartree_per_angstrom2=np.eye(3 * len(atoms)) * 2.0,
        ),
        AnalyticOperationResult.hvp(
            identity=point,
            direction=direction,
            hvp_hartree_per_angstrom2=2.0 * direction,
            forces_hartree_per_angstrom=-2.0 * positions,
            energy_hartree=energy,
        ),
        direction,
    )


def test_raw_energy_force_finalization_preserves_schema_and_owner_independence():
    atoms = _atoms()
    correction = _Correction(atoms)

    payload, solvent = finalize_energy_forces(
        atoms,
        correction=correction,
        energy=27.211386245988,
        forces=np.ones((2, 3)) * 27.211386245988,
        source_unit="eV",
        execution_provenance={"model": {"backend": "duck"}, "solvent": None},
        precision_provenance={"effective_dtype": "float64"},
    )

    assert correction.evaluate_calls == 1
    assert correction.hessian_calls == correction.directional_calls == 0
    assert payload["energy"] == pytest.approx(1.0 + solvent.energy_hartree)
    assert payload["free_energy"] == payload["energy"]
    np.testing.assert_allclose(
        payload["forces"],
        np.ones((2, 3)) + solvent.forces_hartree_per_angstrom,
    )
    assert payload["solvation"]["gas_energy_hartree"] == pytest.approx(1.0)
    assert payload["solvation"]["delta_g_solv_hartree"] == pytest.approx(
        solvent.energy_hartree
    )
    assert payload["inference_precision"] == {"effective_dtype": "float64"}


def test_typed_composition_requests_each_derivative_exactly_once():
    atoms = _atoms()
    gas_admission, correction, composition, ef, hessian, hvp, direction = (
        _gas_operations(atoms)
    )

    payload, solvent = finalize_energy_forces(
        atoms,
        correction=correction,
        gas_operation=ef,
        composition_admission=composition,
        gas_admission=gas_admission,
        execution_provenance={},
    )
    complete_hessian, h_provenance = compose_analytic_hessian(
        atoms,
        gas_operation=hessian,
        correction=correction,
        composition_admission=composition,
        gas_admission=gas_admission,
        operation_provenance={"hessian_enabled": True},
    )
    (hn, forces, energy), hvp_provenance = compose_direct_hvp(
        atoms,
        direction,
        gas_operation=hvp,
        correction=correction,
        composition_admission=composition,
        gas_admission=gas_admission,
        operation_provenance={"hessian_enabled": False},
    )

    assert solvent is not None
    assert correction.evaluate_calls == 1
    assert correction.hessian_calls == 1
    assert correction.directional_calls == 1
    np.testing.assert_array_equal(complete_hessian, np.eye(6) * 2.25)
    np.testing.assert_array_equal(hn, 2.25 * direction)
    np.testing.assert_array_equal(forces, (-2.25 * atoms.positions).reshape(-1))
    assert energy == pytest.approx(payload["energy"])
    assert h_provenance["operation"] == {"hessian_enabled": True}
    assert hvp_provenance["operation"] == {"hessian_enabled": False}


@pytest.mark.parametrize("operation", ["energy_forces", "hessian", "hvp"])
def test_identity_drift_fails_before_solvent_derivative_side_effects(operation):
    atoms = _atoms()
    gas_admission, correction, composition, ef, hessian, hvp, direction = (
        _gas_operations(atoms)
    )
    correction.drift = True

    with pytest.raises(AnalyticIdentityError):
        if operation == "energy_forces":
            finalize_energy_forces(
                atoms,
                correction=correction,
                gas_operation=ef,
                composition_admission=composition,
                gas_admission=gas_admission,
            )
        elif operation == "hessian":
            compose_analytic_hessian(
                atoms,
                gas_operation=hessian,
                correction=correction,
                composition_admission=composition,
                gas_admission=gas_admission,
            )
        else:
            compose_direct_hvp(
                atoms,
                direction,
                gas_operation=hvp,
                correction=correction,
                composition_admission=composition,
                gas_admission=gas_admission,
            )

    assert correction.evaluate_calls == 0
    assert correction.hessian_calls == 0
    assert correction.directional_calls == 0


def test_unsupported_capability_fails_before_provider_evaluation():
    atoms = _atoms()
    gas_admission, correction, composition, ef, *_ = _gas_operations(atoms)
    correction.supported_properties = frozenset({"energy"})

    with pytest.raises(AnalyticCapabilityError, match="forces"):
        finalize_energy_forces(
            atoms,
            correction=correction,
            gas_operation=ef,
            composition_admission=composition,
            gas_admission=gas_admission,
        )

    assert correction.evaluate_calls == 0


def test_hvp_rejects_a_direction_mismatch_before_solvent_evaluation():
    atoms = _atoms()
    gas_admission, correction, composition, _, _, hvp, direction = _gas_operations(
        atoms
    )

    with pytest.raises(AnalyticIdentityError, match="direction"):
        compose_direct_hvp(
            atoms,
            -direction,
            gas_operation=hvp,
            correction=correction,
            composition_admission=composition,
            gas_admission=gas_admission,
        )

    assert correction.directional_calls == 0


def test_raw_hessian_is_gas_only_legacy_passthrough():
    atoms = _atoms()
    raw_hessian = np.eye(6)
    payload, solvent = finalize_energy_forces(
        atoms,
        energy=1.0,
        source_unit="hartree",
        raw_hessian=raw_hessian,
    )
    assert solvent is None
    assert payload["hessian"] is raw_hessian

    with pytest.raises(NotImplementedError, match="implicit-solvent"):
        finalize_energy_forces(
            atoms,
            correction=_Correction(atoms),
            energy=1.0,
            source_unit="hartree",
            raw_hessian=raw_hessian,
        )


def test_already_composed_typed_input_cannot_be_added_twice():
    atoms = _atoms()
    gas_admission, correction, composition, ef, *_ = _gas_operations(atoms)
    payload, _ = finalize_energy_forces(
        atoms,
        correction=correction,
        gas_operation=ef,
        composition_admission=composition,
        gas_admission=gas_admission,
    )
    assert "analytic_composition_identity" in payload

    solvent_point = correction.analytic_scalar_identity(atoms)
    from maple.function.calculator.analytic_protocol import compose_energy_forces

    composed = compose_energy_forces(
        ef,
        correction.evaluate(atoms, need_forces=True),
        solvent_point,
    )
    with pytest.raises(AnalyticIdentityError, match="already composed"):
        finalize_energy_forces(
            atoms,
            correction=correction,
            gas_operation=composed,
            composition_admission=composition,
            gas_admission=gas_admission,
        )
