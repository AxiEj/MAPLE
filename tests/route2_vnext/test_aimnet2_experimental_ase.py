from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
import pytest

from maple.solvation.coupling.aimnet2_experimental_ase import (
    AIMNet2ExperimentalBudgetError,
    AIMNet2ExperimentalCalculator,
    AIMNet2ExperimentalHessianError,
    EXPERIMENTAL_WORKFLOW_ID,
)
from maple.solvation.coupling.geometry_mediated_ase import (
    GeometryMediatedOperationalDomainError,
)


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


class _Domain:
    def validate_atoms(self, atoms):
        if any(int(number) not in {1, 6, 7, 8} for number in atoms.numbers):
            raise ValueError("unsupported element")


class _Model:
    def __init__(self, checkpoint, *, topology_threshold=None):
        payload = checkpoint.read_bytes()
        self._checkpoint_path = checkpoint.resolve()
        self.checkpoint_contract = SimpleNamespace(
            checkpoint_size_bytes=len(payload), checkpoint_sha256=_sha(payload)
        )
        self.domain = _Domain()
        self.topology_threshold = topology_threshold
        self.provenance_sha256 = _sha("model-provenance")
        self.configuration_sha256 = _sha("model-configuration")

    def neighbor_topology(self, atoms):
        changed = (
            self.topology_threshold is not None
            and np.max(np.abs(atoms.positions)) > self.topology_threshold
        )
        return SimpleNamespace(
            as_dict=lambda: {
                "configuration_sha256": self.configuration_sha256,
                "topology_sha256": _sha("changed" if changed else "center"),
                "minimum_cutoff_margin_angstrom": 1.0,
            }
        )


class _Continuum:
    def topology_state(self, atoms):
        return {
            "configuration_sha256": _sha("continuum-config"),
            "cavity_topology_sha256": _sha("continuum"),
            "coefficient_count": 4 * len(atoms),
            "minimum_point_source_shell_margin_angstrom": 1.0,
            "minimum_sphere_tangency_margin_angstrom": 1.0,
            "point_source_topology_sha256": _sha("point"),
            "sphere_pair_topology_sha256": _sha("sphere"),
        }


class _Scalar:
    scalar_id = "synthetic-total-scalar"
    profile_id = "synthetic-disabled-profile"

    def __init__(
        self,
        checkpoint,
        matrix,
        *,
        rough=0.0,
        topology_threshold=None,
        nonfinite_threshold=None,
        replay_energy_offset=0.0,
        cubic=0.0,
    ):
        self.model = _Model(checkpoint, topology_threshold=topology_threshold)
        self.continuum = _Continuum()
        self.matrix = np.asarray(matrix, dtype=float)
        self.rough = float(rough)
        self.nonfinite_threshold = nonfinite_threshold
        self.replay_energy_offset = float(replay_energy_offset)
        self.cubic = float(cubic)
        self.fingerprint = _sha("scalar")
        self.calls = 0

    def fingerprint_sha256(self):
        return self.fingerprint

    def evaluate(self, atoms):
        self.calls += 1
        positions = np.asarray(atoms.positions, dtype=float).reshape(-1)
        forces = -(self.matrix @ positions)
        if self.rough:
            forces -= self.rough * positions * np.abs(positions) ** 0.1
        if self.cubic:
            forces -= self.cubic * positions**3
        if self.nonfinite_threshold is not None and np.max(np.abs(positions)) > (
            self.nonfinite_threshold
        ):
            forces[0] = np.nan
        energy = 0.5 * float(positions @ self.matrix @ positions)
        if self.calls == 20:
            energy += self.replay_energy_offset
        return SimpleNamespace(
            energy=SimpleNamespace(total_energy_eV=energy),
            forces_eV_per_A=forces.reshape((-1, 3)),
        )

    def hessian_vector_product(self, *args, **kwargs):
        raise AssertionError("analytic HVP must not be called")


def _atoms():
    return Atoms("H", positions=np.zeros((1, 3)), info={"charge": 0, "mult": 1})


def _calculator(tmp_path, matrix=None, **scalar_options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"synthetic frozen checkpoint")
    matrix = np.diag([1.25, 2.5, 3.75]) if matrix is None else matrix
    scalar = _Scalar(checkpoint, matrix, **scalar_options)
    calculator = AIMNet2ExperimentalCalculator(
        scalar, checkpoint_path=checkpoint, output=tmp_path / "job.out"
    )
    return calculator, scalar, checkpoint


def test_harmonic_hessian_has_correct_sign_units_budget_and_raw_evidence(tmp_path):
    calculator, scalar, _ = _calculator(tmp_path)
    atoms = _atoms()
    atoms.calc = calculator

    hessian = calculator.get_hessian(atoms)

    np.testing.assert_allclose(hessian, np.diag([1.25, 2.5, 3.75]), atol=1e-12)
    assert calculator.experimental_workflow_id == EXPERIMENTAL_WORKFLOW_ID
    assert calculator.evaluation_count == 20 == scalar.calls
    diagnostic = calculator.last_hessian_diagnostics
    assert diagnostic["status"] == "passed"
    assert diagnostic["evaluation_attempts"] == 20
    assert len(diagnostic["endpoints"]) == 18
    assert len(diagnostic["raw_hessians_eV_per_A2"]) == 3
    assert diagnostic["refinement"]["plateau"] is True
    assert all(diagnostic["gates"].values())
    sidecar = tmp_path / "job.out.hessian-0001.json"
    assert json.loads(sidecar.read_text())["status"] == "passed"
    assert not (tmp_path / "job.out").exists()


def test_each_endpoint_is_guarded_from_center_not_from_previous_endpoint(tmp_path):
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"synthetic frozen checkpoint")
    scalar = _Scalar(checkpoint, np.eye(6))
    calculator = AIMNet2ExperimentalCalculator(scalar, checkpoint_path=checkpoint)
    atoms = Atoms("HH", positions=np.zeros((2, 3)), info={"charge": 0, "mult": 1})
    calculator.get_hessian(atoms)
    for endpoint in calculator.last_hessian_diagnostics["endpoints"]:
        segment = endpoint["guard"]["accepted_to_trial_segment"]
        assert segment["relative_displacement_bound_A"] == pytest.approx(
            endpoint["step_angstrom"]
        )


def test_nonconservative_terminal_matrix_fails_before_symmetrization(tmp_path):
    matrix = np.array([[1.0, 2.0e-3, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    calculator, _, _ = _calculator(tmp_path, matrix=matrix)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="antisymmetry"):
        calculator.get_hessian(_atoms())
    diagnostic = calculator.last_hessian_diagnostics
    assert diagnostic["status"] == "failed"
    assert diagnostic["gates"]["terminal_raw_antisymmetry"] is False
    raw = np.asarray(diagnostic["raw_hessians_eV_per_A2"][-1])
    assert not np.allclose(raw, raw.T)
    assert "returned_hessian_eV_per_A2" not in diagnostic


def test_nonconvergent_step_ladder_preserves_all_raw_failures(tmp_path):
    calculator, _, _ = _calculator(tmp_path, rough=0.01)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="refinement"):
        calculator.get_hessian(_atoms())
    diagnostic = calculator.last_hessian_diagnostics
    assert diagnostic["refinement"]["D2_over_D1"] > 0.55
    assert diagnostic["gates"]["refinement_ratio_or_plateau"] is False
    assert len(diagnostic["raw_hessians_eV_per_A2"]) == 3
    assert (
        json.loads((tmp_path / "job.out.hessian-0001.json").read_text())["status"]
        == "failed"
    )


def test_center_replay_drift_fails_after_preserving_complete_raw_ladder(tmp_path):
    calculator, _, _ = _calculator(tmp_path, replay_energy_offset=2.0e-10)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="center_replay"):
        calculator.get_hessian(_atoms())
    diagnostic = calculator.last_hessian_diagnostics
    assert diagnostic["gates"]["center_replay"] is False
    assert len(diagnostic["raw_hessians_eV_per_A2"]) == 3
    assert diagnostic["evaluation_attempts"] == 20


def test_endpoint_topology_and_nonfinite_values_fail_closed_with_raw_records(tmp_path):
    topology, _, _ = _calculator(tmp_path / "topology", topology_threshold=3.0e-4)
    with pytest.raises(GeometryMediatedOperationalDomainError, match="same_model"):
        topology.get_hessian(_atoms())
    assert topology.last_hessian_diagnostics["status"] == "failed"
    assert topology.last_hessian_diagnostics["evaluation_attempts"] < 20
    failed_endpoint = topology.last_hessian_diagnostics["endpoints"][-1]
    assert failed_endpoint["status"] == "failed"
    assert failed_endpoint["failure"]["type"] == (
        "GeometryMediatedOperationalDomainError"
    )
    assert failed_endpoint["guard"]["gate_passed"] is False
    assert (
        failed_endpoint["guard"]["accepted_to_trial_segment"]["same_model_topology"]
        is False
    )
    assert failed_endpoint["guard"]["accepted_to_trial_segment"]["neighbor_cutoff"][
        "trial_margin_A"
    ] == pytest.approx(1.0)

    nonfinite, _, _ = _calculator(tmp_path / "nonfinite", nonfinite_threshold=3.0e-4)
    with pytest.raises(ValueError, match="invalid energy/force ledger"):
        nonfinite.get_hessian(_atoms())
    assert nonfinite.last_hessian_diagnostics["status"] == "failed"
    assert nonfinite.last_hessian_diagnostics["failure"]["type"] == "ValueError"
    nonfinite_endpoint = nonfinite.last_hessian_diagnostics["endpoints"][-1]
    assert nonfinite_endpoint["status"] == "failed"
    assert nonfinite_endpoint["guard"]["gate_passed"] is True
    assert nonfinite_endpoint["raw_evaluation"]["forces_eV_per_A"][0][0] == "nan"
    assert nonfinite.last_attempted_evaluation == nonfinite_endpoint["raw_evaluation"]

    malformed, malformed_scalar, _ = _calculator(tmp_path / "malformed")
    ordinary_evaluate = malformed_scalar.evaluate

    def malformed_evaluate(atoms):
        evaluation = ordinary_evaluate(atoms)
        if np.max(np.abs(atoms.positions)) > 3.0e-4:
            evaluation.forces_eV_per_A = np.zeros(3)
        return evaluation

    malformed_scalar.evaluate = malformed_evaluate
    with pytest.raises(ValueError, match="invalid energy/force ledger"):
        malformed.get_hessian(_atoms())
    malformed_endpoint = malformed.last_hessian_diagnostics["endpoints"][-1]
    assert malformed_endpoint["raw_evaluation"]["forces_eV_per_A"] == [0.0, 0.0, 0.0]


def test_success_and_failure_restore_calculator_and_caller_state(tmp_path):
    calculator, _, _ = _calculator(tmp_path / "success")
    atoms = _atoms()
    atoms.calc = calculator
    calculator.get_property("energy", atoms)
    old_atoms = calculator.atoms.copy()
    old_results = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in calculator.results.items()
    }
    old_last = calculator.last_evaluation
    old_guard = calculator.last_domain_guard
    count = calculator.evaluation_count
    calculator.get_hessian(atoms)
    assert calculator.evaluation_count == count + 20
    np.testing.assert_array_equal(calculator.atoms.positions, old_atoms.positions)
    np.testing.assert_array_equal(calculator.results["forces"], old_results["forces"])
    assert calculator.results["energy"] == old_results["energy"]
    assert calculator.last_evaluation is old_last
    assert calculator.last_domain_guard == old_guard
    assert atoms.calc is calculator
    np.testing.assert_array_equal(atoms.positions, np.zeros((1, 3)))

    failed, _, _ = _calculator(tmp_path / "failure", topology_threshold=3.0e-4)
    failed_atoms = _atoms()
    failed_atoms.calc = failed
    failed.get_property("energy", failed_atoms)
    old_results = dict(failed.results)
    with pytest.raises(GeometryMediatedOperationalDomainError):
        failed.get_hessian(failed_atoms)
    assert failed.results["energy"] == old_results["energy"]
    np.testing.assert_array_equal(failed.results["forces"], old_results["forces"])
    assert failed_atoms.calc is failed
    np.testing.assert_array_equal(failed_atoms.positions, np.zeros((1, 3)))


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda atoms, calc, scalar, path: atoms.info.__setitem__("charge", 1),
            "charge=0",
        ),
        (lambda atoms, calc, scalar, path: atoms.info.__setitem__("mult", 3), "mult=1"),
        (
            lambda atoms, calc, scalar, path: setattr(
                scalar, "fingerprint", _sha("new")
            ),
            "fingerprint",
        ),
        (
            lambda atoms, calc, scalar, path: path.write_bytes(
                b"synthetic frozen checkpoinu"
            ),
            "bytes changed",
        ),
    ],
)
def test_cached_property_requests_revalidate_info_source_and_checkpoint(
    tmp_path, mutation, match
):
    calculator, scalar, checkpoint = _calculator(tmp_path)
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    assert "forces" in calculator.results
    before = calculator.evaluation_count
    mutation(atoms, calculator, scalar, checkpoint)
    with pytest.raises((ValueError, RuntimeError), match=match):
        atoms.get_forces()
    assert calculator.evaluation_count == before


def test_property_order_never_creates_a_hessian_cache_and_serials_are_unique(tmp_path):
    calculator, _, checkpoint = _calculator(tmp_path)
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_forces()
    atoms.get_potential_energy()
    assert calculator.evaluation_count == 1
    assert "hessian" not in calculator.implemented_properties
    calculator.get_hessian(atoms)
    calculator.get_hessian(atoms)
    assert (tmp_path / "job.out.hessian-0001.json").is_file()
    assert (tmp_path / "job.out.hessian-0002.json").is_file()
    assert calculator.checkpoint_path == checkpoint.resolve()


def test_permuted_cartesian_column_order_replays_the_same_hessian(tmp_path):
    forward, _, _ = _calculator(tmp_path / "forward")
    reverse, _, _ = _calculator(tmp_path / "reverse")
    reverse._coordinate_indices = lambda count: range(count - 1, -1, -1)
    np.testing.assert_allclose(
        reverse.get_hessian(_atoms()), forward.get_hessian(_atoms()), atol=0.0, rtol=0.0
    )
    assert [
        record["coordinate"]
        for record in forward.last_hessian_diagnostics["endpoints"][:6]
    ] == [0, 0, 1, 1, 2, 2]
    assert [
        record["coordinate"]
        for record in reverse.last_hessian_diagnostics["endpoints"][:6]
    ] == [2, 2, 1, 1, 0, 0]


def test_changed_geometry_recomputes_nonlinear_hessian_without_stale_cache(tmp_path):
    calculator, scalar, _ = _calculator(tmp_path, cubic=1.0)
    center = _atoms()
    first = calculator.get_hessian(center)
    calls_after_first = scalar.calls
    displaced = center.copy()
    displaced.positions[0, 0] = 1.0e-2
    second = calculator.get_hessian(displaced)
    assert scalar.calls == calls_after_first + 20
    assert calculator.hessian_call_count == 2
    assert second[0, 0] - first[0, 0] == pytest.approx(3.0e-4, abs=1.0e-10)


def test_constraints_periodicity_elements_and_replaced_sources_are_rejected(tmp_path):
    calculator, scalar, _ = _calculator(tmp_path)
    atoms = _atoms()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraints"):
        calculator.get_property("energy", atoms)
    atoms.set_constraint()
    atoms.pbc = True
    with pytest.raises(GeometryMediatedOperationalDomainError, match="non-periodic"):
        calculator.get_property("energy", atoms)
    atoms.pbc = False
    atoms.numbers[0] = 9
    with pytest.raises(ValueError, match="unsupported element"):
        calculator.get_property("energy", atoms)
    atoms.numbers[0] = 1
    scalar.continuum = _Continuum()
    with pytest.raises(ValueError, match="continuum source was replaced"):
        calculator.get_property("energy", atoms)


def test_absolute_hessian_call_budget_rejects_before_any_new_endpoint(tmp_path):
    calculator, scalar, _ = _calculator(tmp_path)
    calculator.hessian_call_budget = 1
    calculator.evaluation_budget = 20
    calculator.get_hessian(_atoms())
    assert calculator.hessian_call_count == 1
    assert calculator.evaluation_count == 20
    assert scalar.calls == 20

    with pytest.raises(AIMNet2ExperimentalBudgetError, match="Hessian call budget"):
        calculator.get_hessian(_atoms())
    assert calculator.hessian_call_count == 1
    assert calculator.evaluation_count == 20
    assert scalar.calls == 20


def test_absolute_evaluation_budget_fails_before_delegating_excess_attempt(tmp_path):
    calculator, scalar, _ = _calculator(tmp_path)
    calculator.evaluation_budget = 2
    with pytest.raises(AIMNet2ExperimentalBudgetError, match="E/F evaluation budget"):
        calculator.get_hessian(_atoms())
    assert calculator.hessian_call_count == 1
    assert calculator.evaluation_count == 2
    assert scalar.calls == 2
    diagnostic = calculator.last_hessian_diagnostics
    assert diagnostic["status"] == "failed"
    assert diagnostic["evaluation_attempts"] == 2
    assert diagnostic["failure"]["type"] == "AIMNet2ExperimentalBudgetError"


def test_evaluation_budget_is_absolute_across_cached_and_uncached_properties(tmp_path):
    calculator, scalar, _ = _calculator(tmp_path)
    calculator.evaluation_budget = 1
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    atoms.get_forces()
    assert calculator.evaluation_count == 1
    atoms.positions[0, 0] = 1.0e-3
    with pytest.raises(AIMNet2ExperimentalBudgetError, match="1/1"):
        atoms.get_potential_energy()
    assert calculator.evaluation_count == 1
    assert scalar.calls == 1


@pytest.mark.parametrize("name", ["evaluation_budget", "hessian_call_budget"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_budget_mutation_is_revalidated_fail_closed(tmp_path, name, value):
    calculator, _, _ = _calculator(tmp_path)
    setattr(calculator, name, value)
    method = (
        calculator.get_hessian
        if name == "hessian_call_budget"
        else (lambda atoms: calculator.calculate(atoms, ("energy",)))
    )
    with pytest.raises(ValueError, match=name):
        method(_atoms())
