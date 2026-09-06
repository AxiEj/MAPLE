from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from maple.function.dispatcher.dispatcher import Dispatcher
from maple.function.dispatcher.frequency.frequency import Frequency, FrequencyParams
from maple.function.dispatcher.ts.algorithm.PRFO import _pure_frozen_ts_postcondition
from maple.function.calculator.route2 import PureMACEPolarDDXCalculator
from maple.solvation.derivatives.molecular_modes import analyze_molecular_modes


def _rigid_basis(positions, masses):
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    centered = positions - np.average(positions, axis=0, weights=masses)
    root = np.sqrt(masses)
    columns = []
    for axis in np.eye(3):
        columns.append((root[:, None] * axis).reshape(-1))
    for axis in np.eye(3):
        columns.append((root[:, None] * np.cross(axis, centered)).reshape(-1))
    return np.column_stack(columns)


def _internal_basis(positions, masses):
    rigid = _rigid_basis(positions, masses)
    u, singular, _ = np.linalg.svd(rigid, full_matrices=True)
    tolerance = np.finfo(float).eps * max(rigid.shape) * singular[0]
    return u[:, np.sum(singular > tolerance) :]


def _cartesian_hessian_from_internal(positions, masses, eigenvalues):
    q = _internal_basis(positions, masses)
    root_mass = np.repeat(np.sqrt(masses), 3)
    mass_weighted = q @ np.diag(eigenvalues) @ q.T
    return root_mass[:, None] * mass_weighted * root_mass[None, :]


def test_mass_weights_before_internal_projection_for_heteroatomic_molecule():
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.22, 0.91, 0.0]])
    masses = np.array([15.999, 1.008, 2.014])
    hessian = _cartesian_hessian_from_internal(positions, masses, [0.2, 0.7, 1.3])

    result = analyze_molecular_modes(
        positions_angstrom=positions,
        masses_amu=masses,
        hessian_eV_per_A2=hessian,
        richardson_error_eV_per_A2=np.zeros((9, 9)),
    )

    assert result.rigid_rank == 6
    assert result.internal_dimension == 3
    assert result.eigenvalues_eV_per_A2_amu == pytest.approx([0.2, 0.7, 1.3])
    metric = np.repeat(masses, 3)
    assert result.modes_cartesian @ np.diag(
        metric
    ) @ result.modes_cartesian.T == pytest.approx(np.eye(3))


@pytest.mark.parametrize(
    ("atoms", "expected_rank"),
    [
        (Atoms("CO2", positions=[[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]]), 5),
        (Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]), 6),
    ],
)
def test_rigid_rank_distinguishes_linear_and_nonlinear(atoms, expected_rank):
    dimension = 3 * len(atoms)
    result = analyze_molecular_modes(
        positions_angstrom=atoms.positions,
        masses_amu=atoms.get_masses(),
        hessian_eV_per_A2=np.eye(dimension),
        richardson_error_eV_per_A2=np.zeros((dimension, dimension)),
    )
    assert result.rigid_rank == expected_rank
    assert result.internal_dimension == dimension - expected_rank


def test_weak_mode_is_uncertain_and_raw_sign_is_preserved():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    masses = atoms.get_masses()
    hessian = _cartesian_hessian_from_internal(
        atoms.positions, masses, [-1e-5, 0.4, 0.8]
    )
    result = analyze_molecular_modes(
        positions_angstrom=atoms.positions,
        masses_amu=masses,
        hessian_eV_per_A2=hessian,
        richardson_error_eV_per_A2=np.eye(9) * 1e-3,
    )
    assert result.eigenvalues_eV_per_A2_amu[0] < 0
    assert result.statuses[0] == "uncertain"
    assert result.uncertain_count >= 1


def test_global_error_bound_cannot_cancel_under_internal_projection():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    masses = atoms.get_masses()
    translation_x = np.zeros(9)
    translation_x[0::3] = np.sqrt(masses)
    error = np.outer(translation_x, translation_x)
    hessian = _cartesian_hessian_from_internal(atoms.positions, masses, [0.2, 0.4, 0.8])
    result = analyze_molecular_modes(
        positions_angstrom=atoms.positions,
        masses_amu=masses,
        hessian_eV_per_A2=hessian,
        richardson_error_eV_per_A2=error,
    )
    assert result.uncertainty_eV_per_A2_amu > 0.0
    np.testing.assert_array_equal(result.richardson_error_eV_per_A2, error)
    assert not result.richardson_error_eV_per_A2.flags.writeable


@pytest.mark.parametrize(
    ("eigenvalues", "minimum", "index_one"),
    [
        ([0.2, 0.4, 0.8], True, False),
        ([-0.2, 0.4, 0.8], False, True),
        ([-0.2, -0.1, 0.8], False, False),
    ],
)
def test_stationary_point_classification(eigenvalues, minimum, index_one):
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    hessian = _cartesian_hessian_from_internal(
        atoms.positions, atoms.get_masses(), eigenvalues
    )
    result = analyze_molecular_modes(
        positions_angstrom=atoms.positions,
        masses_amu=atoms.get_masses(),
        hessian_eV_per_A2=hessian,
        richardson_error_eV_per_A2=np.zeros((9, 9)),
    )
    assert result.is_resolved_minimum is minimum
    assert result.is_resolved_index_one is index_one


class _WorkflowCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, workflow_kind=None):
        super().__init__()
        self.workflow_kind = workflow_kind

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.zeros((len(atoms), 3), dtype=float),
        }


def _pure_workflow_calculator(atoms):
    atoms.info.update(charge=0, mult=1)

    class TestPES:
        scalar_contract_id = (
            "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-v1"
        )

        def configuration_sha256(self):
            return "d" * 64

        def evaluate_forces(self, geometry):
            return SimpleNamespace(
                central_state=SimpleNamespace(total_energy_eV=0.0),
                total_forces_eV_per_A=np.zeros((len(geometry), 3)),
            )

    return PureMACEPolarDDXCalculator._from_test_pes(
        atoms=atoms, solvent="water", pes=TestPES()
    )


def test_frequency_profile_forbids_sign_relabeling(tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = _pure_workflow_calculator(atoms)
    params = FrequencyParams(method="mw", treat_imag_as_real=True)
    with pytest.raises(ValueError, match="forbids treat_imag_as_real"):
        Frequency(str(tmp_path / "freq.out"), atoms, params=params).run()


def test_unrelated_frequency_calculator_keeps_legacy_mw_path(monkeypatch, tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = _WorkflowCalculator()
    called = []
    monkeypatch.setattr(
        "maple.function.dispatcher.frequency.frequency.MWFrequency.run",
        lambda self: called.append(type(self).__name__),
    )
    Frequency(str(tmp_path / "freq.out"), atoms).run()
    assert called == ["MWFrequency"]


def test_full_dispatch_pure_ts_iteration_cap_is_a_failure_after_artifacts(tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    calculator = _pure_workflow_calculator(atoms)
    atoms.calc = calculator
    output = tmp_path / "capped-ts.out"
    command = SimpleNamespace(params={"method": "prfo", "max_iter": 0})

    with pytest.raises(RuntimeError, match="maximum iteration limit"):
        Dispatcher()(command, "ts", atoms, str(output))

    assert calculator.last_ts_validation == {
        "optimizer_converged": False,
        "index_one_resolved": False,
        "workflow_success": False,
        "reason": "maximum_iterations_reached",
    }
    assert (tmp_path / "capped-ts_prfo_ts.xyz").is_file()
    text = output.read_text()
    assert "Maximum Iterations Reached" in text
    assert "Wrote final TS structure" in text
    assert "Normal Termination" not in text


def test_full_dispatch_legacy_ts_iteration_cap_keeps_legacy_return(tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = _WorkflowCalculator()
    command = SimpleNamespace(params={"method": "prfo", "max_iter": 0})

    assert Dispatcher()(command, "ts", atoms, str(tmp_path / "legacy.out")) is None


@pytest.mark.parametrize(
    ("statuses", "should_pass"),
    [
        (("positive", "positive", "positive"), False),
        (("negative", "positive", "positive"), True),
        (("negative", "uncertain", "positive"), False),
        (("negative", "negative", "positive"), False),
    ],
)
def test_ts_postcondition_requires_one_resolved_negative_and_rest_positive(
    monkeypatch, statuses, should_pass
):
    negative_count = statuses.count("negative")
    positive_count = statuses.count("positive")
    uncertain_count = statuses.count("uncertain")
    analysis = SimpleNamespace(
        internal_dimension=len(statuses),
        rigid_rank=6,
        eigenvalues_eV_per_A2_amu=np.arange(len(statuses), dtype=float),
        uncertainty_eV_per_A2_amu=0.1,
        resolved_negative_count=negative_count,
        resolved_positive_count=positive_count,
        uncertain_count=uncertain_count,
        is_resolved_index_one=(
            negative_count == 1
            and positive_count == len(statuses) - 1
            and uncertain_count == 0
        ),
        is_resolved_minimum=positive_count == len(statuses),
        statuses=statuses,
        eigenvalue_intervals_eV_per_A2_amu=tuple(
            (float(value - 0.1), float(value + 0.1))
            for value in np.arange(len(statuses), dtype=float)
        ),
    )

    monkeypatch.setattr(
        "maple.solvation.derivatives.molecular_modes.analyze_hessian_evaluation",
        lambda atoms, evaluation: analysis,
    )
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    calculator = _pure_workflow_calculator(atoms)
    calculator.get_hessian_evaluation = lambda atoms: SimpleNamespace(
        evaluation_sha256="a" * 64
    )
    monkeypatch.setattr(
        "maple.solvation.derivatives.molecular_modes.hessian_numerical_diagnostics",
        lambda evaluation: {"evaluation_sha256": evaluation.evaluation_sha256},
    )
    atoms.calc = calculator
    diagnostic, _ = _pure_frozen_ts_postcondition(atoms)
    assert diagnostic["optimizer_converged"] is True
    assert diagnostic["workflow_success"] is should_pass
    assert calculator.last_ts_validation == diagnostic


def test_a_marker_only_calculator_does_not_select_pure_frequency(monkeypatch, tmp_path):
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = _WorkflowCalculator("pure-frozen-total-pes")
    called = []
    monkeypatch.setattr(
        "maple.function.dispatcher.frequency.frequency.MWFrequency.run",
        lambda self: called.append(type(self).__name__),
    )
    Frequency(str(tmp_path / "marker.out"), atoms).run()
    assert called == ["MWFrequency"]
    with pytest.raises(RuntimeError, match="registered"):
        _pure_frozen_ts_postcondition(atoms)
