from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

import maple.function.calculator.calculator_base as calculator_base
from maple.function.calculator.analytic_protocol import (
    AnalyticCompositionAdmission,
    AnalyticDerivativeAdmission,
    AnalyticIdentityError,
    AnalyticOperationResult,
    ScalarCellIdentity,
    ScalarPointIdentity,
    compose_hessian,
    freeze_identity_options,
    geometry_sha256,
)
from maple.function.calculator.calculator_base import CalcABC
from maple.function.calculator.extra_correction.implicit.result import (
    SolvationDirectionalResult,
    SolvationResult,
)


def _cell(component: str, *, checkpoint_sha256: str | None = None):
    return ScalarCellIdentity(
        component=component,
        implementation="typed-quadratic-v1" if component == "gas" else "typed-obc2-v1",
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
        options=freeze_identity_options({"fixture": "typed-quadratic"}),
    )


def _point(atoms, cell, *, geometry_override=None):
    return ScalarPointIdentity(
        cell=cell,
        geometry_sha256=geometry_override or geometry_sha256(atoms),
        atomic_numbers=tuple(int(value) for value in atoms.get_atomic_numbers()),
        charge=float(atoms.info.get("charge", 0.0)),
        multiplicity=int(atoms.info.get("mult", 1)),
        atom_topology_sha256=None,
        neighbor_graph_sha256=None,
        active_cutoffs_angstrom=(('radial', 5.1),) if cell.component == "gas" else (),
        minimum_cutoff_gap_angstrom=1.0 if cell.component == "gas" else None,
        response_settings=freeze_identity_options({"fixture": True}),
    )


class _TypedQuadraticCalculator(CalcABC):
    MODEL_ENERGY_UNIT = "hartree"
    SUPPORTED_HESSIAN_MODES = ("analytic", "numerical")
    SUPPORTS_IMPLICIT_SOLVATION = True

    def __init__(self, force_constant=2.0):
        super().__init__()
        self.hessian = "analytic"
        self.force_constant = float(force_constant)
        self.runtime_cell = _cell("gas", checkpoint_sha256="a" * 64)
        self.analytic_derivative_admission = AnalyticDerivativeAdmission(
            profile_id="typed-quadratic-gas-v1",
            evidence_id="typed-quadratic-gas-evidence-v1",
            cell=self.runtime_cell,
            operations=frozenset({"energy_forces", "hessian", "hvp"}),
        )
        self.energy_force_calls = 0
        self.hessian_calls = 0
        self.hvp_calls = 0

    def _operation_point(self, atoms):
        return _point(atoms, self.runtime_cell)

    def _analytic_energy_forces_operation(self, atoms, *, need_forces):
        self.energy_force_calls += 1
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        return AnalyticOperationResult.energy_forces(
            identity=self._operation_point(atoms),
            energy_hartree=(
                0.5 * self.force_constant * float(np.square(positions).sum())
            ),
            forces_hartree_per_angstrom=(
                -self.force_constant * positions if need_forces else None
            ),
        )

    def _analytic_hessian_operation(self, atoms):
        self.hessian_calls += 1
        return AnalyticOperationResult.hessian(
            identity=self._operation_point(atoms),
            hessian_hartree_per_angstrom2=(
                np.eye(3 * len(atoms)) * self.force_constant
            ),
        )

    def _analytic_hvp_operation(self, atoms, direction):
        self.hvp_calls += 1
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        return AnalyticOperationResult.hvp(
            identity=self._operation_point(atoms),
            direction=direction,
            hvp_hartree_per_angstrom2=self.force_constant * direction,
            forces_hartree_per_angstrom=-self.force_constant * positions,
            energy_hartree=(
                0.5 * self.force_constant * float(np.square(positions).sum())
            ),
        )

    def _analytic_hessian(self, atoms):
        return self._analytic_hessian_operation(
            atoms
        ).hessian_hartree_per_angstrom2

    def calculate(self, atoms=None, properties=None, system_changes=None):
        properties = self._normalize_properties(properties)
        atoms = super().calculate(atoms, properties, system_changes)
        energy_forces = self._analytic_energy_forces_operation(
            atoms,
            need_forces="forces" in properties,
        )
        gas_hessian = (
            self._analytic_hessian_operation(atoms)
            if "hessian" in properties
            else None
        )
        self._finalize_results(
            atoms,
            gas_energy_forces=energy_forces,
            gas_hessian=gas_hessian,
        )


class _TypedQuadraticCorrection:
    mode = "fixed"
    inner_mode = None
    supported_properties = frozenset({"energy", "forces", "hessian", "hvp"})

    def __init__(self, force_constant=0.25):
        self.force_constant = float(force_constant)
        self.runtime_cell = _cell("solvent")
        self.analytic_derivative_admission = AnalyticDerivativeAdmission(
            profile_id="typed-obc2-solvent-v1",
            evidence_id="typed-obc2-solvent-evidence-v1",
            cell=self.runtime_cell,
            operations=frozenset({"energy_forces", "hessian", "hvp"}),
        )
        self.evaluate_calls = 0
        self.hessian_calls = 0
        self.directional_calls = 0
        self.identity_calls = 0
        self.drift_geometry = False
        self.last_derivative_provenance = {"provider": "typed-obc2-v1"}

    def analytic_scalar_identity(self, atoms):
        self.identity_calls += 1
        geometry = None
        if self.drift_geometry:
            moved = atoms.copy()
            moved.positions[0, 0] += 0.01
            geometry = geometry_sha256(moved)
        return _point(atoms, self.runtime_cell, geometry_override=geometry)

    def evaluate(self, atoms, *, need_forces=False, calculator=None):
        self.evaluate_calls += 1
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        energy = 0.5 * self.force_constant * float(np.square(positions).sum())
        return SolvationResult(
            energy_hartree=energy,
            forces_hartree_per_angstrom=(
                -self.force_constant * positions if need_forces else None
            ),
            components_hartree={"polar": energy, "nonpolar": 0.0},
            provenance={"provider": "typed-obc2-v1"},
        )

    def get_hessian(self, atoms):
        self.hessian_calls += 1
        return np.eye(3 * len(atoms)) * self.force_constant

    def get_directional_derivatives(self, atoms, direction):
        self.directional_calls += 1
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        return SolvationDirectionalResult(
            hvp_hartree_per_angstrom2=self.force_constant * direction,
            forces_hartree_per_angstrom=-self.force_constant * positions,
            energy_hartree=(
                0.5 * self.force_constant * float(np.square(positions).sum())
            ),
            provenance={"provider": "typed-obc2-v1"},
        )


def _atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]],
    )
    atoms.info.update(charge=0, mult=1)
    return atoms


def _composed_calculator():
    calculator = _TypedQuadraticCalculator()
    correction = _TypedQuadraticCorrection()
    calculator.solvent_correction = correction
    calculator.admit_analytic_composition(
        AnalyticCompositionAdmission(
            profile_id="typed-quadratic-plus-obc2-v1",
            evidence_id="typed-quadratic-plus-obc2-evidence-v1",
            gas_admission=calculator.analytic_derivative_admission,
            solvent_admission=correction.analytic_derivative_admission,
            operations=frozenset({"energy_forces", "hessian", "hvp"}),
        )
    )
    return calculator, correction


def test_public_calculate_hessian_composes_each_component_exactly_once():
    atoms = _atoms()
    calculator, correction = _composed_calculator()

    calculator.calculate(atoms, properties=["hessian"])

    positions = atoms.get_positions()
    expected_energy = 0.5 * (2.0 + 0.25) * float(np.square(positions).sum())
    assert calculator.results["energy"] == pytest.approx(expected_energy)
    np.testing.assert_array_equal(calculator.results["hessian"], np.eye(6) * 2.25)
    assert "forces" not in calculator.results
    assert calculator.energy_force_calls == 1
    assert calculator.hessian_calls == 1
    assert calculator.hvp_calls == 0
    assert correction.evaluate_calls == 1
    assert correction.hessian_calls == 1
    assert correction.directional_calls == 0


def test_energy_force_request_never_evaluates_hessian_or_hvp_and_clears_old_hessian():
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    calculator.results = {"hessian": np.eye(6)}

    calculator.calculate(atoms, properties=["energy", "forces"])

    assert "hessian" not in calculator.results
    assert calculator.hessian_calls == 0
    assert calculator.hvp_calls == 0
    assert correction.hessian_calls == 0
    assert correction.directional_calls == 0
    np.testing.assert_allclose(
        calculator.results["forces"],
        -(2.0 + 0.25) * atoms.get_positions(),
        rtol=0.0,
        atol=0.0,
    )


def test_direct_hvp_is_lazy_and_preserves_dimer_tuple_without_dense_hessian(
    monkeypatch,
):
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    direction = np.arange(1.0, 7.0)

    def forbidden_hessian(*_args, **_kwargs):
        raise AssertionError("direct HVP must not materialize a Hessian")

    def forbidden_fd(*_args, **_kwargs):
        raise AssertionError("analytic HVP must not use finite differences")

    monkeypatch.setattr(calculator, "_analytic_hessian_operation", forbidden_hessian)
    monkeypatch.setattr(correction, "get_hessian", forbidden_hessian)
    monkeypatch.setattr(calculator_base, "numerical_hessian_from_atoms", forbidden_fd)

    hn, forces, energy = calculator.get_hvp(atoms, direction)

    np.testing.assert_array_equal(hn, 2.25 * direction)
    np.testing.assert_array_equal(forces, (-2.25 * atoms.get_positions()).reshape(-1))
    assert energy == pytest.approx(
        0.5 * 2.25 * float(np.square(atoms.get_positions()).sum())
    )
    assert calculator.energy_force_calls == 0
    assert calculator.hessian_calls == 0
    assert calculator.hvp_calls == 1
    assert correction.evaluate_calls == 0
    assert correction.hessian_calls == 0
    assert correction.directional_calls == 1


def test_solvent_point_drift_fails_before_provider_evaluation_or_hessian():
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    correction.drift_geometry = True

    with pytest.raises(AnalyticIdentityError, match="geometry|center"):
        calculator.calculate(atoms, properties=["hessian"])

    assert correction.evaluate_calls == 0
    assert correction.hessian_calls == 0
    assert correction.directional_calls == 0


def test_checkpoint_cell_drift_fails_before_solvent_provider_call():
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    calculator.runtime_cell = _cell("gas", checkpoint_sha256="f" * 64)

    with pytest.raises(AnalyticIdentityError, match="cell identity"):
        calculator.calculate(atoms, properties=["energy", "forces"])

    assert correction.evaluate_calls == 0
    assert correction.hessian_calls == 0
    assert correction.directional_calls == 0


def test_full_hessian_getter_and_public_property_share_composition_path():
    atoms = _atoms()
    calculator, correction = _composed_calculator()

    direct = calculator.get_hessian(atoms)
    assert calculator.hessian_calls == 1
    assert correction.hessian_calls == 1

    calculator.calculate(atoms, properties=["hessian"])
    np.testing.assert_array_equal(calculator.results["hessian"], direct)
    assert calculator.hessian_calls == 2
    assert correction.hessian_calls == 2


def test_finalizer_rejects_an_already_composed_hessian_as_gas_input():
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    gas_energy_forces = calculator._analytic_energy_forces_operation(
        atoms,
        need_forces=False,
    )
    gas_hessian = calculator._analytic_hessian_operation(atoms)
    already_composed = compose_hessian(
        gas_hessian,
        np.eye(6) * correction.force_constant,
        correction.analytic_scalar_identity(atoms),
    )

    with pytest.raises(AnalyticIdentityError, match="gas.*component|already composed"):
        calculator._finalize_results(
            atoms,
            gas_energy_forces=gas_energy_forces,
            gas_hessian=already_composed,
        )

    assert correction.evaluate_calls == 0
    assert correction.hessian_calls == 0
    assert correction.directional_calls == 0


def test_failed_composed_hessian_keeps_prior_public_state(monkeypatch):
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    old_results = {"energy": -7.0, "forces": np.ones((2, 3))}
    old_execution = {"model": {"backend": "prior"}}
    old_solvation = object()
    old_hessian_provenance = {"composition": "prior"}
    calculator.results = old_results
    calculator.execution_provenance = old_execution
    calculator.solvation_result = old_solvation
    calculator.last_analytic_hessian_provenance = old_hessian_provenance

    def fail_hessian(_atoms):
        raise RuntimeError("solvent Hessian failed")

    monkeypatch.setattr(correction, "get_hessian", fail_hessian)

    with pytest.raises(RuntimeError, match="solvent Hessian failed"):
        calculator.calculate(atoms, properties=["hessian"])

    assert calculator.results is old_results
    assert calculator.execution_provenance is old_execution
    assert calculator.solvation_result is old_solvation
    assert calculator.last_analytic_hessian_provenance is old_hessian_provenance
    assert correction.evaluate_calls == 1


def test_failed_hessian_and_hvp_getters_keep_prior_provenance(monkeypatch):
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    old_hessian_provenance = {"composition": "prior-hessian"}
    old_hvp_provenance = {"composition": "prior-hvp"}
    calculator.last_analytic_hessian_provenance = old_hessian_provenance
    calculator.last_hvp_provenance = old_hvp_provenance

    def fail_hessian(_atoms):
        raise RuntimeError("H failed")

    def fail_hvp(_atoms, _direction):
        raise RuntimeError("HVP failed")

    monkeypatch.setattr(correction, "get_hessian", fail_hessian)
    with pytest.raises(RuntimeError, match="H failed"):
        calculator.get_hessian(atoms)
    assert calculator.last_analytic_hessian_provenance is old_hessian_provenance

    monkeypatch.setattr(correction, "get_directional_derivatives", fail_hvp)
    with pytest.raises(RuntimeError, match="HVP failed"):
        calculator.get_hvp(atoms, np.arange(1.0, 7.0))
    assert calculator.last_hvp_provenance is old_hvp_provenance


def test_hessian_diagnostics_commit_only_after_successful_mode_switch(monkeypatch):
    atoms = _atoms()
    calculator, correction = _composed_calculator()
    previous_numeric = {"prior": "numerical"}
    previous_analytic = {"prior": "analytic"}
    calculator.last_numerical_hessian_diagnostics = previous_numeric
    calculator.last_analytic_hessian_provenance = previous_analytic
    original_hessian = correction.get_hessian

    def fail(_atoms):
        raise RuntimeError("derivative failed")

    monkeypatch.setattr(correction, "get_hessian", fail)
    with pytest.raises(RuntimeError, match="derivative failed"):
        calculator.get_hessian(atoms)
    assert calculator.last_numerical_hessian_diagnostics is previous_numeric
    assert calculator.last_analytic_hessian_provenance is previous_analytic
    monkeypatch.setattr(correction, "get_hessian", original_hessian)
    calculator.get_hessian(atoms)
    assert calculator.last_numerical_hessian_diagnostics is None
    assert calculator.last_analytic_hessian_provenance is not previous_analytic
    calculator.hessian = "numerical"
    calculator.get_hessian(atoms, delta=0.001)
    assert calculator.last_analytic_hessian_provenance is None
    assert calculator.last_numerical_hessian_diagnostics is not None
