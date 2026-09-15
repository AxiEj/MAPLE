from __future__ import annotations

import json
from types import MethodType

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.pure_nonmd_status import make_status, write_status


def test_status_sidecar_retains_false_cap_and_geometry(tmp_path):
    atoms = Atoms("H", positions=[[1.0, 2.0, 3.0]])
    status = make_status(
        workflow="opt",
        method="lbfgs",
        converged=False,
        termination_reason="maximum_iterations_reached",
        iterations=4,
        final_metrics={"max_force_hartree_per_angstrom": 0.2},
        atoms=atoms,
        trace=[{"iteration": 4, "max_force_hartree_per_angstrom": 0.2}],
    )
    path = write_status(tmp_path / "job.out", status)
    loaded = json.loads(path.read_text())
    assert loaded["converged"] is False
    assert loaded["termination_reason"] == "maximum_iterations_reached"
    assert loaded["iterations"] == 4
    assert loaded["geometry"][0]["symbols"] == ["H"]
    np.testing.assert_allclose(
        loaded["geometry"][0]["positions_angstrom"], [[1.0, 2.0, 3.0]]
    )


def test_autoneb_cap_state_is_failed_not_converged():
    from maple.function.dispatcher.ts.algorithm.autoneb import _capped_path_status

    assert _capped_path_status() == "failed"


def test_string_refinement_routes_are_dependency_truthful(monkeypatch, tmp_path):
    from maple.function.dispatcher.ts.algorithm.string import GSM

    calls = []
    monkeypatch.setattr(GSM, "_run_cistring", lambda self, images, hei, base: calls.append("cistring"))
    monkeypatch.setattr(GSM, "restart_run", lambda self, images, hei, base: calls.append("prfo"))

    gsm = object.__new__(GSM)
    gsm.output = str(tmp_path / "string.out")
    gsm.params = type("P", (), {"refine": None})()
    assert gsm._dispatch_refinement([], 0, "x") is None
    assert calls == []

    gsm.params.refine = "cistring"
    gsm._dispatch_refinement([], 0, "x")
    assert calls == ["cistring"]

    calls.clear()
    gsm.params.refine = "stringts"
    gsm._dispatch_refinement([], 0, "x")
    assert calls == ["prfo"]


def test_irc_preselected_mode_skips_raw_hessian_reselection(monkeypatch, tmp_path):
    from maple.function.dispatcher.irc.algorithm.gs import GS, GSParams

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
    atoms.get_potential_energy = lambda force_consistent=True: 0.0
    runner = object.__new__(GS)
    runner.atoms = atoms
    runner.output = str(tmp_path / "irc.out")
    runner.p = GSParams(write_traj=False)
    runner.preselected_mode_mw = np.arange(6, dtype=float) + 1.0
    runner.preselected_mode_diagnostic = {"resolved_negative_count": 1}
    seen = []

    def one_side(self, *, forward, sign, q_ts_cart, v_neg_mw, E_ts):
        seen.append((forward, sign, v_neg_mw.copy()))
        return {"records": [{"E": 0.0, "maxG": 0.0, "rmsG": 0.0, "x": atoms.positions.copy()}]}

    runner._one_side = MethodType(one_side, runner)
    runner._merge_and_mark_ts = MethodType(lambda self, f, b: {}, runner)
    monkeypatch.setattr(np.linalg, "eigh", lambda *_: (_ for _ in ()).throw(AssertionError("raw reselection")))

    runner.run()
    assert seen[0][1] == 1.0 and seen[1][1] == -1.0
    np.testing.assert_allclose(seen[0][2], seen[1][2])


def test_dimer_fd_mode_does_not_call_calculator_hvp(tmp_path):
    from maple.function.dispatcher.ts.algorithm.dimer import Dimer

    class Quadratic(Calculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.get_positions()
            self.results["energy"] = 0.5 * float(np.sum(positions * positions))
            self.results["free_energy"] = self.results["energy"]
            self.results["forces"] = -positions

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.calc = Quadratic()
    result = Dimer(
        output=str(tmp_path / "dimer.out"), atoms_init=atoms,
        paras={"use_hvp": False, "remove_rigid": False, "max_iter": 0, "save_traj": False},
    ).run()
    assert result is atoms


def test_dispatcher_cap_retains_status_and_does_not_report_normal_success(monkeypatch, tmp_path):
    import pytest
    from types import SimpleNamespace
    from maple.function.dispatcher.dispatcher import Dispatcher
    atoms = _pure_test_atoms('cap-test-pes')
    status = make_status(workflow='opt', method='lbfgs', converged=False,
                         termination_reason='maximum_iterations_reached', iterations=2,
                         final_metrics={'max_force': 0.5}, atoms=atoms)
    dispatcher = Dispatcher()
    monkeypatch.setattr(dispatcher, '_dispatch_legacy', lambda *a, **k: status)
    output = tmp_path / 'cap.out'
    with pytest.raises(RuntimeError, match='maximum_iterations_reached'):
        dispatcher(SimpleNamespace(params={'method': 'lbfgs'}), 'opt', atoms, str(output))
    record = json.loads((tmp_path / 'cap_status.json').read_text())
    assert record['termination_class'] == 'bounded_nonconvergence'
    assert record['iterations'] == 2


def _pure_test_atoms(configuration):
    from types import SimpleNamespace
    from maple.function.calculator.route2 import PureMACEPolarDDXCalculator
    from maple.function.route2_smd_profiles import route2_smd_profile_spec
    profile = route2_smd_profile_spec('pure-macepolar-frozen-point-l1-ddpcm-smd-nonmd-cpu-v2')
    atoms = Atoms('OHH', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], info={'charge': 0, 'mult': 1})
    pes = SimpleNamespace(scalar_contract_id=profile.scalar_contract_id,
                          configuration_sha256=lambda: configuration)
    atoms.calc = PureMACEPolarDDXCalculator._from_test_pes(
        atoms=atoms, solvent='water', pes=pes, profile_spec=profile)
    return atoms


def test_path_rejects_distinct_pes_configuration_before_dispatch(monkeypatch, tmp_path):
    import pytest
    from types import SimpleNamespace
    from maple.function.dispatcher.dispatcher import Dispatcher
    atoms = [_pure_test_atoms('configuration-a'), _pure_test_atoms('configuration-b')]
    dispatcher = Dispatcher()
    monkeypatch.setattr(dispatcher, '_dispatch_legacy', lambda *a, **k: pytest.fail('different PES entered algorithm'))
    with pytest.raises(ValueError, match='configuration'):
        dispatcher(SimpleNamespace(params={'method': 'neb'}), 'ts', atoms, str(tmp_path / 'path.out'))


def test_path_rejects_wrong_atom_order_before_dispatch(monkeypatch, tmp_path):
    import pytest
    from types import SimpleNamespace
    from maple.function.dispatcher.dispatcher import Dispatcher
    first = _pure_test_atoms('same-pes')
    second = first[[1, 0, 2]]
    second.calc = first.calc
    dispatcher = Dispatcher()
    monkeypatch.setattr(dispatcher, '_dispatch_legacy', lambda *a, **k: pytest.fail('wrong atom order entered algorithm'))
    with pytest.raises(ValueError, match='symbols|atoms'):
        dispatcher(SimpleNamespace(params={'method': 'neb'}), 'ts', [first, second], str(tmp_path / 'path.out'))


def test_dimer_uses_declared_canary_seed(monkeypatch, tmp_path):
    from maple.function.dispatcher.ts.algorithm.dimer import Dimer
    from ase.calculators.singlepoint import SinglePointCalculator
    atoms = Atoms('H2', positions=[[0, 0, 0], [0.7, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((2, 3)))
    monkeypatch.setenv('MAPLE_NONMD_SEED', '1321')
    one = Dimer(str(tmp_path / 'one.out'), atoms, paras={'remove_rigid': False})
    two = Dimer(str(tmp_path / 'two.out'), atoms, paras={'remove_rigid': False})
    np.testing.assert_array_equal(one.n, two.n)


def test_scan_keeps_completed_points_when_later_point_raises(monkeypatch, tmp_path):
    from maple.function.dispatcher.scan.scan import Scan
    scan = object.__new__(Scan)
    scan.atoms = _pure_test_atoms('scan-pes')
    scan.output = str(tmp_path / 'scan.out')
    scan.mode, scan.method = 'relaxed', 'lbfgs'
    scan._last_atoms = scan.atoms
    scan._total_combinations = 2
    def fail_second():
        scan.point_statuses = [{'point_index': 1, 'converged': True, 'coordinates': [1.0],
                               'termination_reason': 'converged', 'iterations': 2,
                               'final_metrics': {'energy_hartree': -1.0}}]
        raise RuntimeError('second-point provider failure')
    monkeypatch.setattr(scan, 'run_scan', fail_second)
    status = scan.run()
    assert status['converged'] is False
    assert status['termination_class'] == 'execution_failure'
    assert status['points'][0]['point_index'] == 1
    assert 'second-point provider failure' in status['error']


def test_optimizer_failure_retains_iteration_count_and_recorded_metrics(monkeypatch, tmp_path):
    from maple.function.dispatcher.optimization.optimization import Optimization
    atoms = _pure_test_atoms('failure-pes')
    atoms.max_f, atoms.rms_f, atoms.max_dp, atoms.rms_dp = 0.1, 0.05, 0.01, 0.005
    class FailingRFO:
        def __init__(self, *args, **kwargs):
            self.last_iterations = 3
        def run(self):
            raise RuntimeError('Hessian antisymmetry guard')
    monkeypatch.setattr('maple.function.dispatcher.optimization.algorithm.RFO', FailingRFO)
    result = Optimization({'method': 'rfo'}, str(tmp_path / 'opt.out'), atoms).run()
    status = result._maple_nonmd_status
    assert status['iterations'] == 3
    assert status['termination_class'] == 'execution_failure'
    assert status['final_metrics']['last_recorded_metrics']['max_f'] == 0.1
    assert 'antisymmetry' in status['error']


def test_real_prfo_cap_is_bounded_nonconvergence_not_execution_failure(tmp_path):
    import pytest
    from types import SimpleNamespace
    from maple.function.dispatcher.dispatcher import Dispatcher
    atoms = _pure_test_atoms('prfo-cap-pes')
    atoms.calc.pes.evaluate_forces = lambda atoms, central_state=None: SimpleNamespace(
        central_state=SimpleNamespace(total_energy_eV=1.0),
        total_forces_eV_per_A=np.zeros_like(atoms.positions),
        evaluation_sha256='0' * 64,
    )
    output = tmp_path / 'prfo-cap.out'
    with pytest.raises(RuntimeError):
        Dispatcher()(SimpleNamespace(params={'method': 'prfo', 'max_iter': 0}),
                     'ts', atoms, str(output))
    record = json.loads((tmp_path / 'prfo-cap_status.json').read_text())
    assert record['termination_class'] == 'bounded_nonconvergence'
    assert record['termination_reason'] == 'maximum_iterations_reached'
    assert record['executed'] is True
    assert record['iterations'] == 0


def test_string_refinement_reuses_immutable_calculator_and_keeps_typed_failure(monkeypatch, tmp_path):
    import pytest
    from maple.function.dispatcher.ts.algorithm.string import GSM
    from maple.function.dispatcher.pure_nonmd_status import NonMDWorkflowFailure
    atoms = _pure_test_atoms('string-refinement-pes')
    atoms.calc.__deepcopy__ = lambda memo: pytest.fail('immutable PES must not be copied')
    class CappedPRFO:
        def __init__(self, *, atoms, **kwargs):
            self.atoms = atoms
        def run(self):
            raise NonMDWorkflowFailure(make_status(
                workflow='ts', method='prfo', converged=False,
                termination_reason='maximum_iterations_reached', iterations=3,
                final_metrics={'energy_hartree': -1.0}, atoms=self.atoms,
            ))
    import importlib
    monkeypatch.setattr(importlib.import_module('maple.function.dispatcher.ts.algorithm.PRFO'), 'PRFO', CappedPRFO)
    gsm = object.__new__(GSM)
    gsm.output = str(tmp_path / 'string.out')
    result = gsm.restart_run([atoms], 0, str(tmp_path / 'string'))
    assert result['iterations'] == 3
    assert result['termination_class'] == 'bounded_nonconvergence'


def test_cineb_initially_converged_path_has_finite_complete_metrics(tmp_path):
    from ase.calculators.singlepoint import SinglePointCalculator
    from maple.function.dispatcher.ts.algorithm.neb import NEB
    images = []
    for x, energy in ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)):
        atoms = Atoms('H', positions=[[x, 0, 0]])
        atoms.calc = SinglePointCalculator(atoms, energy=energy, free_energy=energy,
                                           forces=np.zeros((1, 3)))
        images.append(atoms)
    from maple.function.utility import Molecules
    neb = NEB(str(tmp_path / 'ci.out'), Molecules(images),
              paras={'n_images': 1, 'max_iter': 1, 'refine': 'cineb'})
    result = neb.restart_run(images, energies=[0.0, 1.0, 0.0])
    assert result['converged'] is True
    assert result['iterations'] == 0
    assert result['final_metrics']['rms_projected_force_hartree_per_angstrom'] == 0.0


def test_status_serializes_numpy_metrics_without_relaxing_nonfinite_guard(tmp_path):
    import pytest
    atoms = Atoms('H', positions=[[0, 0, 0]])
    status = make_status(workflow='ts', method='string', converged=False,
                         termination_reason='path_not_converged', iterations=1,
                         final_metrics={'growth_converged': np.bool_(True),
                                        'count': np.int64(3), 'forces': np.array([1.0, 2.0])},
                         atoms=atoms)
    path = write_status(tmp_path / 'numpy.out', status)
    read = json.loads(path.read_text())
    assert read['final_metrics'] == {'growth_converged': True, 'count': 3, 'forces': [1.0, 2.0]}
    status['final_metrics']['invalid'] = np.float64(np.nan)
    with pytest.raises(ValueError):
        write_status(tmp_path / 'invalid.out', status)
