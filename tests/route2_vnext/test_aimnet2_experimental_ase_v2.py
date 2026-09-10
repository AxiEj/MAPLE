from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest

from maple.solvation.coupling.aimnet2_experimental_ase import (
    AIMNet2ExperimentalBudgetError,
    AIMNet2ExperimentalHessianError,
)
from maple.solvation.coupling.aimnet2_experimental_ase_v2 import (
    AIMNet2ExperimentalCalculatorV2,
    EXPERIMENTAL_WORKFLOW_ID_V2,
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
        if tuple(atoms.numbers) != (1,):
            raise ValueError("test domain requires hydrogen")


class _Model:
    def __init__(self, checkpoint, topology_threshold=None):
        payload = checkpoint.read_bytes()
        self._checkpoint_path = checkpoint.resolve()
        self.checkpoint_contract = SimpleNamespace(
            checkpoint_size_bytes=len(payload), checkpoint_sha256=_sha(payload)
        )
        self.domain = _Domain()
        self.topology_threshold = topology_threshold

    def neighbor_topology(self, atoms):
        changed = (
            self.topology_threshold is not None
            and np.max(np.abs(atoms.positions)) > self.topology_threshold
        )
        return SimpleNamespace(
            as_dict=lambda: {
                "configuration_sha256": _sha("model-config"),
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
        *,
        matrix=None,
        quartic=0.0,
        sextic=0.0,
        rough=0.0,
        topology_threshold=None,
        nonfinite_threshold=None,
        replay_energy_offset=0.0,
    ):
        self.model = _Model(checkpoint, topology_threshold)
        self.continuum = _Continuum()
        self.matrix = np.diag([1.0, 2.0, 3.0]) if matrix is None else np.asarray(matrix)
        self.quartic = float(quartic)
        self.sextic = float(sextic)
        self.rough = float(rough)
        self.nonfinite_threshold = nonfinite_threshold
        self.replay_energy_offset = float(replay_energy_offset)
        self.calls = 0

    def fingerprint_sha256(self):
        return _sha("scalar-v2")

    def evaluate(self, atoms):
        self.calls += 1
        positions = np.asarray(atoms.positions, dtype=float).reshape(-1)
        forces = -(self.matrix @ positions)
        forces -= self.quartic * positions**3 + self.sextic * positions**5
        if self.rough:
            forces -= self.rough * positions * np.abs(positions) ** 0.1
        if self.nonfinite_threshold is not None and np.max(np.abs(positions)) > (
            self.nonfinite_threshold
        ):
            forces[0] = np.nan
        energy = 0.5 * float(positions @ self.matrix @ positions)
        energy += self.quartic * float(np.sum(positions**4)) / 4.0
        energy += self.sextic * float(np.sum(positions**6)) / 6.0
        if self.calls == 26:
            energy += self.replay_energy_offset
        return SimpleNamespace(
            energy=SimpleNamespace(total_energy_eV=energy),
            forces_eV_per_A=forces.reshape((-1, 3)),
        )

    def hessian_vector_product(self, *args, **kwargs):
        raise AssertionError("v2 must not call an analytic HVP")


def _atoms(x=0.0):
    return Atoms("H", positions=[[x, 0.0, 0.0]], info={"charge": 0, "mult": 1})


def _calculator(tmp_path, **options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "aimnet2.pt"
    checkpoint.write_bytes(b"synthetic v2 checkpoint")
    scalar = _Scalar(checkpoint, **options)
    calculator = AIMNet2ExperimentalCalculatorV2(
        scalar, checkpoint_path=checkpoint, output=tmp_path / "job.out"
    )
    return calculator, scalar


def test_v2_four_step_richardson_cancels_quartic_and_bounds_sextic_error(tmp_path):
    calculator, scalar = _calculator(tmp_path, quartic=1.0e4, sextic=1.0e10)
    hessian = calculator.get_hessian(_atoms())
    np.testing.assert_allclose(
        hessian, np.diag([1.0, 2.0, 3.0]) - np.eye(3) * 2.5e-7, atol=2.0e-11
    )
    diagnostic = calculator.last_hessian_diagnostics
    assert calculator.experimental_workflow_id == EXPERIMENTAL_WORKFLOW_ID_V2
    assert calculator.hessian_evaluations_per_atom == 24
    assert calculator.evaluation_count == scalar.calls == 26
    assert diagnostic["evaluation_attempts"] == 26
    assert len(diagnostic["endpoints"]) == 24
    assert len(diagnostic["raw_hessians_eV_per_A2"]) == 4
    assert len(diagnostic["raw_richardson_hessians_eV_per_A2"]) == 3
    assert diagnostic["richardson_refinement"]["E2_over_E1"] == pytest.approx(
        1.0 / 16.0, rel=5e-5
    )
    assert all(diagnostic["gates"].values())


def test_v2_noisy_and_dual_antisymmetry_gates_preserve_raw_matrices(tmp_path):
    noisy, _ = _calculator(tmp_path / "noise", rough=0.01)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="refinement"):
        noisy.get_hessian(_atoms())
    assert (
        noisy.last_hessian_diagnostics["gates"][
            "richardson_refinement_ratio_or_plateau"
        ]
        is False
    )
    assert len(noisy.last_hessian_diagnostics["raw_hessians_eV_per_A2"]) == 4

    matrix = np.array([[1.0, 2.0e-3, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    nonconservative, _ = _calculator(tmp_path / "antisymmetry", matrix=matrix)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="antisymmetry"):
        nonconservative.get_hessian(_atoms())
    gates = nonconservative.last_hessian_diagnostics["gates"]
    assert gates["raw_finest_antisymmetry"] is False
    assert gates["raw_K2_antisymmetry"] is False


def test_v2_absolute_richardson_error_gate_is_independent_of_ratio(tmp_path):
    calculator, _ = _calculator(tmp_path, sextic=3.0e12)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="richardson_error"):
        calculator.get_hessian(_atoms())
    diagnostic = calculator.last_hessian_diagnostics
    assert diagnostic["richardson_refinement"]["E2_over_E1"] == pytest.approx(
        1.0 / 16.0, rel=5e-5
    )
    assert diagnostic["gates"]["richardson_refinement_ratio_or_plateau"] is True
    assert diagnostic["gates"]["richardson_error"] is False
    assert diagnostic["gates"]["raw_finest_antisymmetry"] is True
    assert diagnostic["gates"]["raw_K2_antisymmetry"] is True


def test_v2_center_replay_and_endpoint_failures_keep_current_evidence(tmp_path):
    replay, _ = _calculator(tmp_path / "replay", replay_energy_offset=2.0e-10)
    with pytest.raises(AIMNet2ExperimentalHessianError, match="center_replay"):
        replay.get_hessian(_atoms())
    assert replay.last_hessian_diagnostics["gates"]["center_replay"] is False

    topology, _ = _calculator(tmp_path / "topology", topology_threshold=3.0e-4)
    with pytest.raises(GeometryMediatedOperationalDomainError, match="same_model"):
        topology.get_hessian(_atoms())
    endpoint = topology.last_hessian_diagnostics["endpoints"][-1]
    assert endpoint["status"] == "failed"
    assert endpoint["guard"]["gate_passed"] is False
    assert (
        endpoint["guard"]["accepted_to_trial_segment"]["same_model_topology"] is False
    )

    nonfinite, _ = _calculator(tmp_path / "nonfinite", nonfinite_threshold=3.0e-4)
    with pytest.raises(ValueError, match="invalid energy/force ledger"):
        nonfinite.get_hessian(_atoms())
    endpoint = nonfinite.last_hessian_diagnostics["endpoints"][-1]
    assert endpoint["guard"]["gate_passed"] is True
    assert endpoint["failure"]["type"] == "ValueError"
    assert endpoint["raw_evaluation"]["forces_eV_per_A"][0][0] == "nan"


def test_v2_absolute_budgets_use_24n_plus_2_and_do_not_roll_back(tmp_path):
    calculator, scalar = _calculator(tmp_path)
    calculator.evaluation_budget = 26
    calculator.hessian_call_budget = 1
    calculator.get_hessian(_atoms())
    assert calculator.evaluation_count == 26
    assert calculator.hessian_call_count == 1
    with pytest.raises(AIMNet2ExperimentalBudgetError, match="Hessian call budget"):
        calculator.get_hessian(_atoms())
    assert calculator.evaluation_count == scalar.calls == 26
    assert calculator.hessian_call_count == 1

    short, short_scalar = _calculator(tmp_path / "short")
    short.evaluation_budget = 25
    with pytest.raises(AIMNet2ExperimentalBudgetError, match="E/F evaluation budget"):
        short.get_hessian(_atoms())
    assert short.evaluation_count == short_scalar.calls == 25
    assert short.hessian_call_count == 1
    assert short.last_hessian_diagnostics["status"] == "failed"


def test_v2_changed_geometry_recomputes_all_rows_without_stale_reuse(tmp_path):
    calculator, scalar = _calculator(tmp_path, quartic=1.0)
    first = calculator.get_hessian(_atoms())
    second = calculator.get_hessian(_atoms(1.0e-2))
    assert scalar.calls == 52
    assert calculator.hessian_call_count == 2
    assert second[0, 0] - first[0, 0] == pytest.approx(3.0e-4, abs=1.0e-10)
    assert (tmp_path / "job.out.hessian-0001.json").is_file()
    assert (tmp_path / "job.out.hessian-0002.json").is_file()


def test_v2_success_and_endpoint_failure_restore_all_accepted_state(tmp_path):
    calculator, _ = _calculator(tmp_path / "success")
    atoms = _atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    prior_atoms = calculator.atoms.copy()
    prior_results = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in calculator.results.items()
    }
    prior_positions = np.array(calculator._accepted_positions, copy=True)
    prior_numbers = calculator._accepted_atomic_numbers
    prior_model_topology = dict(calculator._accepted_model_topology)
    prior_continuum_topology = dict(calculator._accepted_continuum_topology)
    prior_evaluation = calculator.last_evaluation
    calculator.get_hessian(atoms)
    np.testing.assert_array_equal(calculator.atoms.positions, prior_atoms.positions)
    assert calculator.results["energy"] == prior_results["energy"]
    np.testing.assert_array_equal(calculator.results["forces"], prior_results["forces"])
    np.testing.assert_array_equal(calculator._accepted_positions, prior_positions)
    assert calculator._accepted_atomic_numbers == prior_numbers
    assert calculator._accepted_model_topology == prior_model_topology
    assert calculator._accepted_continuum_topology == prior_continuum_topology
    assert calculator.last_evaluation is prior_evaluation
    assert atoms.calc is calculator
    np.testing.assert_array_equal(atoms.positions, np.zeros((1, 3)))

    failed, _ = _calculator(tmp_path / "failure", topology_threshold=3.0e-4)
    failed_atoms = _atoms()
    failed_atoms.calc = failed
    failed_atoms.get_forces()
    prior_results = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in failed.results.items()
    }
    prior_positions = np.array(failed._accepted_positions, copy=True)
    prior_evaluation = failed.last_evaluation
    with pytest.raises(GeometryMediatedOperationalDomainError):
        failed.get_hessian(failed_atoms)
    assert failed.results["energy"] == prior_results["energy"]
    np.testing.assert_array_equal(failed.results["forces"], prior_results["forces"])
    np.testing.assert_array_equal(failed._accepted_positions, prior_positions)
    assert failed.last_evaluation is prior_evaluation
    assert failed_atoms.calc is failed
    np.testing.assert_array_equal(failed_atoms.positions, np.zeros((1, 3)))
