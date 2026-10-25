"""Adversarial raw-record tests; synthetic receipts are never scientific evidence."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'docs/implicit-solvation/benchmarks'
sys.path.insert(0, str(BENCH))
import validate_aimnet2025_analytic as validator  # noqa: E402

TEST_POINT = {'synthetic_test_only': True, 'neighbor_graph_sha256': 'a'*64,
              'neighbor_packing_sha256': 'b'*64, 'neighbor_graph_scheme': 'row-multiset-with-padding-v1'}


def _probe(positions, direction, step):
    x = np.asarray(positions)
    v = np.asarray(direction).reshape(x.shape)
    return {'point_plus': deepcopy(TEST_POINT), 'point_minus': deepcopy(TEST_POINT),
            'positions_plus_angstrom': (x + step*v).tolist(),
            'positions_minus_angstrom': (x - step*v).tolist(),
            'forces_plus_hartree_per_angstrom': np.zeros_like(x).tolist(),
            'forces_minus_hartree_per_angstrom': np.zeros_like(x).tolist()}


@pytest.fixture
def receipt(monkeypatch):
    # Fixed 12-key matrix, deliberately tiny inputs. All live boundary mocks are
    # local to this fixture; production has no equivalent flag or CLI bypass.
    manifest = {'records': []}
    directions = {'records': []}
    for i, case in enumerate(validator.CASES):
        for geometry in ('base', 'displaced'):
            item = {'case': case, 'geometry': geometry, 'atomic_numbers': [1],
                    'positions_angstrom': [[i*.125, 0., 0.]], 'ordered_charges_e': [0.],
                    'source': {'synthetic_test_only': True}}
            manifest['records'].append(item)
            for index in range(3):
                directions['records'].append({'case': case, 'geometry': geometry,
                    'direction_index': index, 'vector': np.eye(3)[index].tolist()})
    controls, source_directions = [], []
    for geometry in ('base', 'displaced'):
        controls.append({'case': 'nitralin', 'geometry': geometry, 'atomic_numbers': [1, 1],
            'positions_angstrom': [[0., 0., 0.], [4.6, 0., 0.]], 'ordered_charges_e': [0., 0.],
            'source': {'synthetic_test_only': True}})
        for index in range(3):
            source_directions.append({'case': 'nitralin', 'geometry': geometry,
                'direction_index': index, 'vector': np.eye(6)[index].tolist()})
    frozen = {'manifest': manifest, 'directions': directions,
              'original_manifest': {'records': controls}, 'original_directions': {'records': source_directions},
              'cutoff_controls': {'records': controls}, 'snapshot': {'synthetic_test_only': True}}
    monkeypatch.setattr(validator, 'load_frozen_inputs', lambda: deepcopy(frozen))
    runtime = {'synthetic_test_only': True}
    monkeypatch.setattr(validator, 'runtime_snapshot', lambda: runtime)
    monkeypatch.setattr(validator, 'validate_runtime_snapshot', lambda value: (
        validator.require_equal(value, runtime, 'runtime test binding')))
    snapshot = {'git_head': 'a'*40, 'worktree_status': [],
                'files_sha256': {'synthetic': 'b'*64},
                'files_aggregate_sha256': validator.digest({'synthetic': 'b'*64}),
                'inventory_policy': 'git-ls-files-cached-plus-nonignored-untracked-v1',
                'excluded_data_only_paths': sorted(validator.qualification.QUALIFICATION_SOURCE_EXCLUDES)}
    snapshot['aggregate_sha256'] = validator.digest(snapshot)
    monkeypatch.setattr(validator.qualification, 'source_snapshot', lambda **kw: snapshot)
    monkeypatch.setattr(validator.qualification, 'validate_current_source_binding', lambda value: (
        validator.require_equal(value, snapshot, 'current source test binding')))
    monkeypatch.setattr(validator, 'checkpoint_snapshot', lambda path=None: {'path': '/test.pt',
        'sha256': validator.CHECKPOINT_SHA256, 'size_bytes': 8839102})
    monkeypatch.setattr(validator, 'validate_point_identity', lambda point, item, device: (
        validator.require_equal(point, deepcopy(TEST_POINT), 'point test binding')))
    monkeypatch.setattr(validator, 'validate_cutoff_point_identity', lambda point, item, device: (
        validator.require_equal(point, deepcopy(TEST_POINT), 'cutoff point test binding')))
    identity = {'synthetic_test_only': True, 'native_configuration': {'engine': 'synthetic'},
                'state_sha256': 'synth-state'}
    monkeypatch.setattr(validator, 'validate_identity', lambda value, device, runtime: (
        validator.require_equal(value, identity, 'adapter identity test binding')))
    artifact = {'schema_version': validator.SCHEMA_VERSION, 'artifact_type': validator.ARTIFACT_TYPE,
        'qualification_stage': validator.STAGE, 'candidate_git_sha': 'a'*40,
        'harness_sha256': validator.core.sha256_file(ROOT / validator.HARNESS_REL),
        'validator_sha256': validator.core.sha256_file(Path(validator.__file__)),
        'profile': validator.load_profile(), 'expected_native_identity': validator.EXPECTED_NATIVE_IDENTITY,
        'model_identities_before': {'cpu': identity, 'cuda': identity},
        'model_identities_after': {'cpu': identity, 'cuda': identity},
        'source_snapshot_before': snapshot,
        'source_snapshot_after': deepcopy(snapshot), 'inputs_before': frozen['snapshot'],
        'inputs_after': frozen['snapshot'], 'runtime_before': runtime, 'runtime_after': runtime,
        'checkpoint_before': validator.checkpoint_snapshot(),
        'checkpoint_after': validator.checkpoint_snapshot(),
        'gas_records': [], 'hessian_records': [], 'hvp_records': [], 'cutoff_control_records': []}
    for item in manifest['records']:
        for device in ('cpu', 'cuda'):
            common = {k: item[k] for k in ('case', 'geometry')}
            common.update(device=device, status='ok', input=deepcopy(item), identity=deepcopy(identity),
                          point_identity=deepcopy(TEST_POINT))
            gas = dict(common, energy_hartree=0., native_energy_hartree=0.,
                       forces_hartree_per_angstrom=[[0., 0., 0.]],
                       native_forces_hartree_per_angstrom=[[0., 0., 0.]],
                       raw_hessian_hartree_per_angstrom2=np.zeros((3, 3)).tolist())
            artifact['gas_records'].append(gas)
            for nonpolar in ('ace', 'none'):
                row = dict(common, nonpolar=nonpolar,
                    energy_hartree=0., forces_hartree_per_angstrom=[[0., 0., 0.]],
                    solvent_energy_hartree=0., solvent_forces_hartree_per_angstrom=[[0., 0., 0.]],
                    raw_hessian_hartree_per_angstrom2=np.zeros((3, 3)).tolist(),
                    gas_hessian_hartree_per_angstrom2=np.zeros((3, 3)).tolist(),
                    solvent_hessian_hartree_per_angstrom2=np.zeros((3, 3)).tolist(),
                    provider_identity={'runtime_provider': 'torch-obc2', 'model': 'obc2',
                                       'profile': 'obc2-mbondi2', 'nonpolar': nonpolar,
                                       'dtype': 'torch.float32', 'arithmetic_profile': 'obc2-fp32-v1'},
                    force_fd_ladder=[])
                for step in validator.STEPS:
                    probes = [dict(axis=axis, **_probe(item['positions_angstrom'], np.eye(3)[axis], step))
                              for axis in range(3)]
                    row['force_fd_ladder'].append({'step_angstrom': step, 'probes': probes,
                        'hessian_hartree_per_angstrom2': np.zeros((3, 3)).tolist()})
                artifact['hessian_records'].append(row)
                for direction_index in range(3):
                    v = np.eye(3)[direction_index].tolist()
                    artifact['hvp_records'].append(dict(common, nonpolar=nonpolar,
                        direction_index=direction_index, direction=v, hvp_hartree_per_angstrom2=[0.]*3,
                        energy_hartree=0., forces_hartree_per_angstrom=[[0., 0., 0.]],
                        direct_hvp_dense_hessian_forbidden=True,
                        directional_fd_ladder=[dict(step_angstrom=step,
                            **_probe(item['positions_angstrom'], v, step)) for step in validator.STEPS]))
    for section in ('gas_records', 'hessian_records', 'hvp_records'):
        for row in artifact[section]:
            if section == 'gas_records':
                row['native_reference_identity'] = {'implementation': 'aimnet.calculators.calculator.AIMNet2Calculator',
                    'operation': 'official forces=True,hessian=False; native input without requires_grad',
                    'checkpoint_sha256': validator.CHECKPOINT_SHA256,
                    'potential_configuration': {}, 'state_sha256': 'synth-state',
                    'observed_neighbor_graph_sha256': TEST_POINT['neighbor_graph_sha256'],
                    'observed_neighbor_packing_sha256': TEST_POINT['neighbor_packing_sha256'],
                    'neighbor_graph_scheme': validator.GRAPH_SCHEME}
            row['precision'] = dict(row['identity'], requested_dtype='float32', effective_dtype='float32',
                original_checkpoint_path='/test.pt', original_checkpoint_sha256=validator.CHECKPOINT_SHA256,
                in_memory_cast_only=False, analytic_prepared=True,
                solvent_dtype=None if section == 'gas_records' else 'float32', last_point=row['point_identity'])
            if section == 'hvp_records':
                row['hvp_provenance'] = {'composition': 'aimnet2025-gas-analytic+torch-obc2-directional',
                    'gas_ad_class': 'N32', 'gas_coordinate_dtype': 'float32', 'gas_gradient_dtype': 'float32',
                    'gas_hvp_dtype': 'float32', 'output_dtype': 'float64', 'gas_point': row['point_identity'],
                    'direction_quantization_max': 0., 'solvent_dtype': 'float32',
                    'solvent': {'synthetic_test_only': True, 'dtype': 'torch.float32', 'arithmetic_profile': 'obc2-fp32-v1'}}
    for item in controls:
        for device in ('cpu', 'cuda'):
            for nonpolar in ('ace', 'none'):
                point = deepcopy(TEST_POINT)
                precision = dict(identity, requested_dtype='float32', effective_dtype='float32',
                    original_checkpoint_path='/test.pt', original_checkpoint_sha256=validator.CHECKPOINT_SHA256,
                    in_memory_cast_only=False, analytic_prepared=True, solvent_dtype='float32', last_point=point)
                gas = {'identity': identity, 'point_identity': point,
                    'precision': dict(precision, solvent_dtype=None), 'energy_hartree': 0.,
                    'native_energy_hartree': 0., 'forces_hartree_per_angstrom': np.zeros((2, 3)).tolist(),
                    'native_forces_hartree_per_angstrom': np.zeros((2, 3)).tolist(),
                    'native_reference_identity': {'implementation': 'aimnet.calculators.calculator.AIMNet2Calculator',
                        'operation': 'official forces=True,hessian=False; native input without requires_grad',
                        'checkpoint_sha256': validator.CHECKPOINT_SHA256, 'potential_configuration': {},
                        'state_sha256': 'synth-state',
                    'observed_neighbor_graph_sha256': TEST_POINT['neighbor_graph_sha256'],
                    'observed_neighbor_packing_sha256': TEST_POINT['neighbor_packing_sha256'],
                    'neighbor_graph_scheme': validator.GRAPH_SCHEME}}
                row = {'case': 'nitralin', 'geometry': item['geometry'], 'device': device, 'nonpolar': nonpolar,
                    'status': 'ok', 'input': deepcopy(item), 'identity': identity, 'point_identity': point,
                    'precision': precision, 'gas_center': gas, 'energy_hartree': 0.,
                    'forces_hartree_per_angstrom': np.zeros((2, 3)).tolist(), 'solvent_energy_hartree': 0.,
                    'solvent_forces_hartree_per_angstrom': np.zeros((2, 3)).tolist(),
                    'provider_identity': {'runtime_provider': 'torch-obc2', 'model': 'obc2',
                        'profile': 'obc2-mbondi2', 'nonpolar': nonpolar,
                                       'dtype': 'torch.float32', 'arithmetic_profile': 'obc2-fp32-v1'}, 'refusals': []}
                for operation, index in [('GAS_H', -1), ('COMPOSED_H', -1)]+[('HVP', i) for i in range(3)]:
                    refusal = {'operation': operation, 'direction_index': index, 'status': 'rejected',
                        'error': {'module': 'maple.function.calculator.analytic_protocol',
                            'type': 'AnalyticCapabilityError',
                            'message': 'AIMNet2025 analytic curvature refused within native/original cutoff guard.',
                            'traceback': 'synthetic isolated expected refusal'},
                        'positions_angstrom': item['positions_angstrom'], 'point_identity': point,
                        'precision': dict(precision, solvent_dtype=None if operation == 'GAS_H' else 'float32'),
                        'forbidden_entrypoints': list(validator.CUTOFF_FORBIDDEN), 'fallback_calls': []}
                    if operation == 'HVP':
                        refusal['direction'] = np.eye(6)[index].tolist()
                    row['refusals'].append(refusal)
                artifact['cutoff_control_records'].append(row)
    # Actual-operand traces are synthetic only in this isolated schema fixture.
    from maple.function.calculator.analytic_protocol import named_array_sha256
    for section in ('hessian_records', 'cutoff_control_records'):
        for row in artifact[section]:
            shape = np.asarray(row['forces_hartree_per_angstrom']).shape
            row['same_call_center'] = {
                'schema': 'same-call-ef-operands-v1', 'finalize_calls': 1, 'solvent_calls': 1,
                'gas_source_unit': 'eV', 'gas_energy_source': 0.,
                'gas_forces_source_dtype': 'float64', 'gas_forces_source': np.zeros(shape).tolist(),
                'solvent_energy_hartree': row['solvent_energy_hartree'],
                'solvent_forces_hartree_per_angstrom': row['solvent_forces_hartree_per_angstrom'],
                'point_identity': deepcopy(row['point_identity'])}
            if section == 'hessian_records':
                row['same_call_hessian'] = {
                    'schema': 'same-call-hessian-operands-v1', 'gas_calls': 1, 'solvent_calls': 1,
                    'gas_hessian_sha256': named_array_sha256({'component': np.asarray(row['gas_hessian_hartree_per_angstrom2'])}),
                    'solvent_hessian_sha256': named_array_sha256({'component': np.asarray(row['solvent_hessian_hartree_per_angstrom2'])}),
                    'point_identity': deepcopy(row['point_identity'])}
    for row in artifact['hvp_records']:
        row['same_call_hvp'] = {'schema': 'same-call-hvp-operands-v1', 'scalar_calls': 1,
            'solvent_calls': 1, 'gas_source_unit': 'eV', 'gas_energy_source': 0.,
            'gas_derivatives': [{'dtype': 'float32', 'values': [[0., 0., 0.]]} for _ in range(2)],
            'solvent': {'energy_hartree': 0., 'forces_hartree_per_angstrom': [[0., 0., 0.]],
                        'hvp_hartree_per_angstrom2': [0., 0., 0.]},
            'point_identity': deepcopy(row['point_identity'])}
    # Populate stored summaries from raw data, not hand-written pass labels.
    validator.add_assessments(artifact)
    return validator.core.seal_artifact(artifact)


def _reseal(artifact):
    return validator.core.seal_artifact(artifact)


def test_replays_all_four_complete_raw_record_denominators(receipt):
    gates = validator.validate_artifact(receipt)
    assert {name: gate['passed_count'] for name, gate in gates.items()} == validator.EXPECTED_COUNTS


@pytest.mark.parametrize('section', ['gas_records', 'hessian_records', 'hvp_records'])
def test_rejects_a_missing_expected_cell(receipt, section):
    receipt[section].pop()
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('section', ['gas_records', 'hessian_records', 'hvp_records'])
def test_rejects_duplicate_cell_even_when_denominator_is_unchanged(receipt, section):
    receipt[section][-1] = deepcopy(receipt[section][0])
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('field', ['harness_sha256', 'validator_sha256'])
def test_rejects_current_code_fingerprint_mismatch(receipt, field):
    receipt[field] = 'f'*64
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('field', ['source_snapshot_after', 'inputs_after', 'runtime_after', 'checkpoint_after'])
def test_rejects_before_after_identity_drift(receipt, field):
    receipt[field] = {}
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_raw_hessian_tampering_despite_passing_summary(receipt):
    receipt['hessian_records'][0]['raw_hessian_hartree_per_angstrom2'][0][0] = .1
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_raw_force_probe_tampering_despite_passing_summary(receipt):
    receipt['hessian_records'][0]['force_fd_ladder'][1]['probes'][0]['forces_plus_hartree_per_angstrom'][0][0] = 1.
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_probe_coordinate_tampering(receipt):
    receipt['hessian_records'][0]['force_fd_ladder'][1]['probes'][0]['positions_plus_angstrom'][0][1] = .01
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '0'])
def test_rejects_nonfinite_or_nonnumeric_raw_matrix_entry(receipt, bad):
    receipt['hessian_records'][0]['raw_hessian_hartree_per_angstrom2'][0][0] = bad
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_wrong_matrix_shape(receipt):
    receipt['hessian_records'][0]['raw_hessian_hartree_per_angstrom2'].pop()
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_changed_fd_step_instead_of_choosing_best_step(receipt):
    receipt['hessian_records'][0]['force_fd_ladder'][2]['step_angstrom'] = .0001
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_a_nonfrozen_hvp_direction(receipt):
    receipt['hvp_records'][0]['direction'] = [0., 1., 0.]
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_directional_fd_tampering(receipt):
    receipt['hvp_records'][0]['directional_fd_ladder'][1]['forces_minus_hartree_per_angstrom'][0][0] = .1
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_wrong_adapter_identity(receipt):
    receipt['gas_records'][0]['identity'] = {}
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_missing_point_graph_identity(receipt):
    del receipt['gas_records'][0]['point_identity']
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_relabeling_native_precision_as_d64(receipt):
    receipt['profile']['gas_precision_class'] = 'D64'
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_claiming_installed_admission(receipt):
    receipt['qualification_stage'] = 'installed qualification'
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_counts_cuda_error_as_failure_without_dropping_expected_rows(receipt):
    row = receipt['gas_records'][1]
    row.update(status='error', error={'type': 'RuntimeError', 'message': 'CUDA unavailable', 'traceback': 'test'})
    for section in ('hessian_records', 'hvp_records'):
        for dependent in receipt[section]:
            if all(dependent[key] == row[key] for key in ('case', 'geometry', 'device')):
                dependent.update(status='error', error=deepcopy(row['error']))
    validator.add_assessments(receipt)
    assert receipt['gates']['aimnet2025_native_gas_identity']['expected_count'] == 24
    assert receipt['gates']['aimnet2025_native_gas_identity']['passed_count'] == 23
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_forged_gate_count(receipt):
    receipt['gates']['aimnet2025_direct_hvp']['passed_count'] = 143
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_unsealed_evidence(receipt):
    receipt['content_sha256'] = '0'*64
    with pytest.raises(ValueError):
        validator.validate_artifact(receipt)


def test_producer_refuses_overwrite_before_reading_models(tmp_path):
    path = tmp_path / 'existing.json'
    path.write_text('immutable')
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(FileExistsError):
        module.run(path)
    assert path.read_text() == 'immutable'


def test_coarse_fd_failure_is_retained_but_does_not_select_a_best_step(receipt):
    row = receipt['hessian_records'][0]
    coarse = row['force_fd_ladder'][0]
    coarse['probes'][0]['forces_plus_hartree_per_angstrom'][0][0] = 1.
    coarse['hessian_hartree_per_angstrom2'][0][0] = -250.
    validator.add_assessments(receipt)
    assert row['metrics']['force_fd'][0]['max'] == 250.
    assert validator.validate_artifact(_reseal(receipt))['aimnet2025_composed_hessian']['passed']


@pytest.mark.parametrize('index', [1, 2])
def test_each_frozen_finest_force_fd_step_must_pass(receipt, index):
    row = receipt['hessian_records'][0]
    step = row['force_fd_ladder'][index]['step_angstrom']
    row['force_fd_ladder'][index]['probes'][0]['forces_plus_hartree_per_angstrom'][0][0] = 2*step*.001
    row['force_fd_ladder'][index]['hessian_hartree_per_angstrom2'][0][0] = -.001
    validator.add_assessments(receipt)
    assert row['passed'] is False
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('field,value', [('native_energy_hartree', 2.001e-5),
    ('native_forces_hartree_per_angstrom', [[2.001e-6, 0., 0.]])])
def test_native_center_disagreement_fails_even_after_recomputing_labels(receipt, field, value):
    receipt['gas_records'][0][field] = value
    validator.add_assessments(receipt)
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_hvp_gate_uses_elementwise_tolerance_not_only_fd_rms(receipt):
    row = receipt['hvp_records'][0]
    row['hvp_hartree_per_angstrom2'][0] = 5.001e-6
    validator.add_assessments(receipt)
    assert row['passed'] is False
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_boolean_direction_index(receipt):
    receipt['hvp_records'][1]['direction_index'] = True
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_producer_rejects_dirty_source_before_initializing_a_model(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def dirty(**kwargs):
        assert kwargs == {'require_clean': True}
        raise RuntimeError('Formal qualification requires a clean committed tree')
    monkeypatch.setattr(validator.qualification, 'source_snapshot', dirty)
    monkeypatch.setattr(validator, 'load_frozen_inputs', lambda: pytest.fail('must reject before reading inputs'))
    with pytest.raises(RuntimeError, match='clean committed tree'):
        module.run(tmp_path / 'new.json')


def test_real_frozen_inputs_have_all_predeclared_vectors_and_array_hashes():
    frozen = validator.load_frozen_inputs()
    assert len(frozen['manifest']['records']) == 12
    assert len(frozen['directions']['records']) == 36


def test_producer_retains_nonfinite_failure_as_explicit_raw_token(tmp_path):
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failed = {'passed': False, 'raw': [float('nan'), float('inf'), -float('inf')]}
    module._write_new(tmp_path / 'failed.json', failed)
    import json
    preserved = json.loads((tmp_path / 'failed.json').read_text())
    assert preserved['raw'] == ['NaN', 'Infinity', '-Infinity']


@pytest.mark.parametrize('field,value', [('effective_dtype', 'float64'), ('solvent_dtype', 'float64'),
    ('original_checkpoint_sha256', 'f'*64), ('analytic_prepared', False)])
def test_rejects_per_operation_precision_or_checkpoint_spoof(receipt, field, value):
    receipt['hessian_records'][0]['precision'][field] = value
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_hvp_precision_provenance_spoof(receipt):
    receipt['hvp_records'][0]['hvp_provenance']['gas_hvp_dtype'] = 'float64'
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_native_reference_potential_spoof(receipt):
    receipt['gas_records'][0]['native_reference_identity'] = {'model': 'legacy aimnet2'}
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_rejects_all_pass_booleans_when_raw_native_energy_fails(receipt):
    receipt['gas_records'][0]['native_energy_hartree'] = .01
    assert receipt['gas_records'][0]['passed'] is True
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_nonfinite_producer_assessment_preserves_all_expected_rows(receipt):
    receipt['hessian_records'][0]['raw_hessian_hartree_per_angstrom2'][0][0] = float('nan')
    validator.add_assessments(receipt)
    assert receipt['expected_count'] == 224
    assert receipt['passed'] is False


def test_failed_model_postflight_can_never_be_sealed(receipt):
    receipt['postflight_errors'] = {'cpu': {'message': 'state drift'}}
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.fixture
def native_identity():
    """Portable complete native configuration; no inference or cache access."""
    from maple.function.calculator.aimnet import _aimnet2025_identity as pins
    metadata = dict(deepcopy(pins.MODEL_METADATA), family=None, supports_charged_systems=None,
                    has_embedded_d3ts=False)
    flags = {name: {} for name in ('afv', 'mlps', 'outputs', 'outputs.atomic_shift.shifts',
        'outputs.energy_mlp.mlp', 'outputs.energy_mlp.mlp.0', 'outputs.energy_mlp.mlp.2',
        'outputs.energy_mlp.mlp.4')}
    flags.update({'': {'d2features': True, 'num_charge_channels': 1}, 'aev': {'dmat_fill': 5.},
        'conv_a': {'d2features': True, 'do_vector': True},
        'conv_q': {'d2features': False, 'do_vector': True},
        'outputs.energy_mlp': {'key_in': 'aim', 'key_out': 'energy'},
        'outputs.atomic_shift': {'reduce_sum': False, 'key_in': 'energy', 'key_out': 'energy'},
        'outputs.atomic_sum': {'key_in': 'energy', 'key_out': 'energy'},
        'outputs.srcoulomb': {'key_in': 'charges', 'key_out': 'energy'},
        'outputs.energy_mlp.mlp.1': {'approximate': 'none'}})
    for index in range(3):
        flags[f'mlps.{index}'] = {}
        for layer in ([0, 2, 4] if index < 2 else [0, 2, 4, 6]):
            flags[f'mlps.{index}.{layer}'] = {}
        flags[f'mlps.{index}.1'] = {'approximate': 'none'}
    sr = {'rc': 4.599999904632568, 'envelope': 'exp', 'key_in': 'charges',
          'key_out': 'energy', 'factor': 7.199822675975274}
    config = {'metadata': metadata, 'engine': 'official aimnet public scalar; forces=False,hessian=False; requires_grad leaf',
        'train': False, 'compiled': False, 'device': 'cpu', 'lr': True, 'cutoff': 5.,
        'cutoff_lr': 'inf', 'coulomb_method': 'simple', 'coulomb_cutoff': 'inf',
        'd3_cutoff': 15., 'nb_threshold': 120,
        'neighbor_cutoffs': {'_nblist': 5., '_nblist_lr': 1000000., '_nblist_dftd3': None, '_nblist_coulomb': None},
        'input_dtypes': {'coord': 'torch.float32', 'numbers': 'torch.int32', 'charge': 'torch.float32'},
        'model_input_dtypes': ['torch.float32', 'torch.int64', 'torch.float32'], 'model_flags': flags,
        'coulomb': dict(sr, method='simple', subtract_sr=False), 'embedded_sr': sr,
        'd3': dict(pins.MODEL_METADATA['d3_params'], smoothing_on=12., smoothing_off=15., key_out='energy')}
    external = {'packages': dict(pins.PACKAGES), 'source_sha256': dict(pins.SOURCE_SHA256)}
    identity = {'model': 'aimnet2025', 'registry_key': pins.REGISTRY_KEY, 'profile_id': pins.PROFILE_ID,
        'device': 'cpu', 'checkpoint_sha256': validator.CHECKPOINT_SHA256,
        'effective_nn_dtype': 'float32', 'effective_ad_class': 'N32', 'atomic_shift_dtype': 'float64',
        'energy_accumulation_dtype': 'float64', 'd4': False, 'source_energy_unit': 'eV',
        'neighbor_graph_scheme': pins.NEIGHBOR_GRAPH_SCHEME,
        'precision_execution_policy': pins.PRECISION_EXECUTION_POLICY,
        'external_runtime': external, 'native_configuration': config,
        'state_sha256': validator.EXPECTED_NATIVE_IDENTITY['state_sha256']}
    runtime = {'external_runtime': external, 'devices': {'cuda': {'index': 0}}}
    return identity, runtime


def test_accepts_exact_pinned_complete_native_configuration(native_identity):
    identity, runtime = native_identity
    validator.validate_identity(identity, 'cpu', runtime)


@pytest.mark.parametrize('field', ['model', 'registry_key', 'profile_id', 'checkpoint_sha256',
    'effective_ad_class', 'external_runtime', 'native_configuration', 'state_sha256',
    'neighbor_graph_scheme', 'precision_execution_policy'])
def test_rejects_missing_structured_native_identity_field(native_identity, field):
    identity, runtime = native_identity
    del identity[field]
    with pytest.raises(ValueError):
        validator.validate_identity(identity, 'cpu', runtime)


@pytest.mark.parametrize('path,value', [
    (('effective_ad_class',), 'D64'), (('device',), 'cuda:0'), (('d4',), True),
    (('state_sha256',), 'f'*64),
    (('external_runtime', 'source_sha256', 'modules/lr.py'), 'f'*64),
    (('native_configuration', 'coulomb', 'factor'), 0.),
    (('native_configuration', 'd3', 's8'), 0.),
    (('native_configuration', 'neighbor_cutoffs', '_nblist'), 6.),
    (('native_configuration', 'model_flags', 'conv_a', 'do_vector'), False),
    (('native_configuration', 'compiled'), True)])
def test_rejects_nested_native_potential_or_source_spoof(native_identity, path, value):
    identity, runtime = native_identity
    # Keep independent expected runtime separate from the tampered receipt.
    runtime = deepcopy(runtime)
    current = identity
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError):
        validator.validate_identity(identity, 'cpu', runtime)


@pytest.fixture
def native_point():
    from maple.function.calculator.analytic_protocol import named_array_sha256
    x = np.array([[0., 0., 0.], [1.1, 0., 0.]], dtype=np.float64)
    z = np.array([1, 1], dtype=np.int64)
    native = x.astype(np.float32)
    point = {'geometry_sha256': named_array_sha256({'atomic_numbers': z, 'positions_angstrom': x}),
        'native_geometry_sha256': named_array_sha256({'atomic_numbers': z, 'positions_angstrom': native}),
        'native_coordinate_dtype': 'torch.float32', 'atomic_numbers': z.tolist(),
        'charge': 0., 'multiplicity': 1, 'total_energy_dtype': 'torch.float64',
        'max_coordinate_quantization_angstrom': float(np.max(np.abs(x-native.astype(np.float64)))),
        'neighbor_graph_sha256': 'a'*64, 'neighbor_packing_sha256': 'b'*64,
        'neighbor_graph_scheme': validator.GRAPH_SCHEME, 'active_cutoffs_angstrom': {'radial': 5.,
            'embedded_sr': 4.599999904632568, 'd3_smoothing_on': 12., 'd3_smoothing_off': 15.},
        'minimum_cutoff_gap_angstrom': 4.599999904632568-float(native[1, 0]),
        'cutoff_guard_margin_angstrom': .003,
        'intermediate_dtypes': {key: 'torch.float32' for key in ('coord', 'charge', 'mult', 'a',
            'd_ij', 'g_sv', '_delta_Q', 'charges_pre', '_dQ', 'charges', 'aim')},
        'response_settings': {'autocast_enabled': False, 'cuda_matmul_allow_tf32': False,
            'cudnn_allow_tf32': False, 'inference_mode_enabled': False, 'grad_enabled': True, 'device_type': 'cpu'}}
    point['intermediate_dtypes']['energy'] = 'torch.float64'
    return point, {'atomic_numbers': z.tolist(), 'positions_angstrom': x.tolist()}


def test_accepts_native_point_with_replayed_geometry_and_cutoff_margin(native_point):
    point, item = native_point
    validator.validate_point_identity(point, item, 'cpu')


@pytest.mark.parametrize('path,value', [
    (('native_coordinate_dtype',), 'torch.float64'),
    (('native_geometry_sha256',), 'f'*64),
    (('neighbor_graph_sha256',), None),
    (('minimum_cutoff_gap_angstrom',), 100.),
    (('intermediate_dtypes', 'energy'), 'torch.float32'),
    (('response_settings', 'autocast_enabled'), 'false'),
    (('response_settings', 'grad_enabled'), False),
    (('active_cutoffs_angstrom', 'd3_smoothing_off'), 14.)])
def test_rejects_actual_point_graph_precision_or_cutoff_spoof(native_point, path, value):
    point, item = native_point
    current = point
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError):
        validator.validate_point_identity(point, item, 'cpu')


@pytest.mark.parametrize('field,value', [('autocast_enabled', True), ('cuda_matmul_allow_tf32', True),
    ('cudnn_allow_tf32', True), ('inference_mode_enabled', True), ('grad_enabled', False)])
def test_runtime_rejects_forbidden_settings_even_when_snapshot_matches(monkeypatch, field, value):
    state = {'autocast_enabled': False, 'cuda_matmul_allow_tf32': False,
        'cudnn_allow_tf32': False, 'inference_mode_enabled': False, 'grad_enabled': True}
    runtime = {'default_dtype': 'torch.float32', 'settings': {'cpu': deepcopy(state), 'cuda': deepcopy(state)}}
    runtime['settings']['cuda'][field] = value
    monkeypatch.setattr(validator, 'runtime_snapshot', lambda: runtime)
    with pytest.raises(ValueError):
        validator.validate_runtime_snapshot(runtime)


def test_native_reference_uses_resolved_cuda_index(monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = {}
    def native(path, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'aimnet.calculators', SimpleNamespace(AIMNet2Calculator=native))
    module._construct_native_reference(SimpleNamespace(device='cuda:0'), '/test.pt')
    assert observed['device'] == 'cuda:0'


def test_failed_producer_retains_full_matrix_without_publishing_canonical(receipt, monkeypatch, tmp_path):
    from maple.function.calculator.aimnet import _aimnet2025_calculator as adapter
    import torch
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def unavailable(*args, **kwargs):
        raise RuntimeError('isolated unavailable model/device')
    monkeypatch.setattr(adapter, 'AIMNet2025Calculator', unavailable)
    canonical = tmp_path / 'candidate.json'
    backends = (torch.backends.cuda.matmul, torch.backends.cudnn, torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
    selectors_before = tuple(backend.fp32_precision for backend in backends)
    artifact = module.run(canonical)
    assert tuple(backend.fp32_precision for backend in backends) == selectors_before
    assert not canonical.exists()
    assert [len(artifact[k]) for k in ('gas_records', 'hessian_records', 'hvp_records', 'cutoff_control_records')] == [24, 48, 144, 8]
    assert artifact['expected_count'] == 224 and artifact['passed_count'] == 0
    assert Path(artifact['retained_output_path']).is_file()
    assert artifact['passed'] is False


def test_private_candidate_admission_patch_is_restored_after_constructor_error(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import maple.function.calculator.set_calculator as factory_module
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = factory_module._admit_independent_obc2_analytic_cell
    def fail():
        raise RuntimeError('isolated constructor error')
    monkeypatch.setattr(module, 'atoms_from_input', lambda item: object())
    monkeypatch.setattr(factory_module, 'SetCalculator', lambda *a, **kw: SimpleNamespace(set_calculator=fail))
    with pytest.raises(RuntimeError, match='isolated constructor error'):
        module._construct_composed({}, 'cpu', 'ace', tmp_path, '/test.pt', {}, {})
    assert factory_module._admit_independent_obc2_analytic_cell is original


def test_expected_source_snapshot_parameter_cannot_waive_clean_status(receipt):
    dirty = deepcopy(receipt['source_snapshot_before'])
    dirty['worktree_status'] = [' M source.py']
    with pytest.raises(ValueError):
        validator.validate_artifact(receipt, expected_source_snapshot=dirty)


def test_rejects_native_reference_neighbor_topology_spoof(receipt):
    receipt['gas_records'][0]['native_reference_identity']['observed_neighbor_graph_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_candidate_protocol_v2_is_explicit_without_schema1_compatibility():
    assert validator.ARTIFACT_TYPE == 'aimnet2025-candidate-qualification-v3'
    assert validator.EXPECTED_COUNTS['aimnet2025_cutoff_guard'] == 8


def test_v2_loader_preserves_old_sources_and_only_two_explicit_transforms():
    frozen = validator.load_frozen_inputs()
    old = {(r['case'], r['geometry']): r for r in frozen['original_manifest']['records']}
    for item in frozen['manifest']['records']:
        source = old[item['case'], item['geometry']]
        if item['case'] == 'nitralin':
            assert item['qualification_geometry_transform']['scale'] == .97
            assert validator.cutoff_gap(item) > .004
        else:
            assert item == source
    assert frozen['cutoff_controls']['records'] == [r for r in old.values() if r['case'] == 'nitralin']


def test_original_invalid_nitralin_cannot_be_relabelled_as_positive():
    frozen = validator.load_frozen_inputs()
    for original in frozen['cutoff_controls']['records']:
        with pytest.raises(ValueError):
            validator.validate_positive_geometry(original, original)


def test_original_source_file_bytes_remain_pinned_in_v2_snapshot():
    frozen = validator.load_frozen_inputs()
    for name, (expected, _) in validator.FROZEN_HASHES.items():
        relative = str((validator.CORPUS/'frozen-inputs'/name).relative_to(ROOT))
        assert frozen['snapshot']['files_sha256'][relative] == expected
        assert validator.core.sha256_file(ROOT/relative) == expected


@pytest.mark.parametrize('schema,kind', [(1, 'aimnet2025-candidate-qualification-v1'), (1, 'aimnet2025-candidate-qualification-v2')])
def test_schema1_cannot_be_implicitly_upgraded_to_protocol2(receipt, schema, kind):
    receipt.update(schema_version=schema, artifact_type=kind)
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'moved_source', 'input_position'])
def test_rejects_missing_duplicate_or_changed_original_control(receipt, mutation):
    rows = receipt['cutoff_control_records']
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    elif mutation == 'moved_source':
        rows[0]['input']['source'] = {'moved_to': 'unbound.mol2'}
    else:
        rows[0]['input']['positions_angstrom'][1][0] = 4.5
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('field,value', [('module', 'builtins'), ('type', 'RuntimeError'),
    ('message', 'AIMNet2025 composed analytic HVP requires parent receipt admission.'),
    ('message', 'CUDA unavailable')])
def test_unrelated_negative_exception_never_counts_as_expected_guard(receipt, field, value):
    receipt['cutoff_control_records'][0]['refusals'][0]['error'][field] = value
    validator.add_assessments(receipt)
    assert receipt['gates']['aimnet2025_cutoff_guard']['passed_count'] == 7
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_negative_guard_requires_all_three_original_unit_directions(receipt):
    receipt['cutoff_control_records'][0]['refusals'].pop()
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_negative_control_native_center_closure_is_recomputed_from_raw_forces(receipt):
    receipt['cutoff_control_records'][0]['gas_center']['native_forces_hartree_per_angstrom'][0][0] = .1
    validator.add_assessments(receipt)
    assert receipt['gates']['aimnet2025_cutoff_guard']['passed_count'] == 7
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_negative_control_cannot_reach_fd_or_dense_fallback(receipt):
    receipt['cutoff_control_records'][0]['refusals'][0]['fallback_calls'] = ['torch.autograd.grad']
    validator.add_assessments(receipt)
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_negative_refusal_counts_are_separate_from_48_computed_hessians(receipt):
    validator.validate_artifact(receipt)
    assert receipt['gates']['aimnet2025_composed_hessian']['passed_count'] == 48
    assert receipt['gates']['aimnet2025_cutoff_guard']['passed_count'] == 8
    assert receipt['expected_cutoff_refusals'] == receipt['observed_cutoff_refusals'] == 40


def _negative_point_from_fixture(native_point):
    from maple.function.calculator.analytic_protocol import named_array_sha256
    point, item = deepcopy(native_point)
    item['positions_angstrom'][1][0] = 4.6
    x = np.asarray(item['positions_angstrom'], dtype=np.float64)
    z = np.asarray(item['atomic_numbers'], dtype=np.int64)
    point.update(geometry_sha256=named_array_sha256({'atomic_numbers': z, 'positions_angstrom': x}),
        native_geometry_sha256=named_array_sha256({'atomic_numbers': z, 'positions_angstrom': x.astype(np.float32)}),
        max_coordinate_quantization_angstrom=float(np.max(np.abs(x-x.astype(np.float32).astype(np.float64)))),
        minimum_cutoff_gap_angstrom=validator.cutoff_gap(item))
    return point, item


def test_genuine_negative_point_is_rejected_as_positive_but_valid_as_control(native_point):
    point, item = _negative_point_from_fixture(native_point)
    with pytest.raises(ValueError, match='inside native cutoff guard'):
        validator.validate_point_identity(point, item, 'cpu')
    validator.validate_cutoff_point_identity(point, item, 'cpu')


@pytest.mark.parametrize('field,value', [('cutoff_guard_margin_angstrom', .004),
    ('cutoff_guard_margin_angstrom', '0.003'), ('minimum_cutoff_gap_angstrom', 1.)])
def test_negative_point_rejects_changed_guard_encoding_or_claimed_gap(native_point, field, value):
    point, item = _negative_point_from_fixture(native_point)
    point[field] = value
    with pytest.raises(ValueError):
        validator.validate_cutoff_point_identity(point, item, 'cpu')


def test_positive_point_cannot_pass_as_expected_negative(native_point):
    point, item = native_point
    with pytest.raises(ValueError):
        validator.validate_cutoff_point_identity(point, item, 'cpu')


def test_unavailable_cuda_control_remains_a_failed_row_in_full_denominator(receipt):
    row = receipt['cutoff_control_records'][2]
    row.update(status='error', error={'module': 'builtins', 'type': 'RuntimeError',
        'message': 'CUDA unavailable', 'traceback': 'synthetic failed device'})
    validator.add_assessments(receipt)
    assert receipt['gates']['aimnet2025_cutoff_guard'] == {'expected_count': 8, 'passed_count': 7, 'passed': False}
    assert receipt['expected_count'] == 224
    assert receipt['observed_cutoff_refusals'] == 35
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_control_refusal_rejects_changed_source_unit_direction(receipt):
    refusal = receipt['cutoff_control_records'][0]['refusals'][2]
    refusal['direction'] = np.eye(6)[1].tolist()
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_control_input_snapshot_drift_cannot_be_resealed(receipt):
    receipt['inputs_after'] = dict(receipt['inputs_after'], cutoff_controls_content_sha256='f'*64)
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_control_collector_records_and_blocks_gradient_fallback():
    from types import SimpleNamespace
    from ase import Atoms
    import torch
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    no_call = lambda *a, **kw: None
    provider = SimpleNamespace(hessian=no_call, kernel=SimpleNamespace(hessian=no_call))
    correction = SimpleNamespace(get_hessian=no_call, underlying_provider=provider)
    calc = SimpleNamespace(last_analytic_point_identity={'synthetic': True}, inference_precision_provenance={'synthetic': True})
    original_grad = torch.autograd.grad
    observed = module._capture_cutoff_refusal(calc, Atoms('H2', positions=[[0, 0, 0], [4.6, 0, 0]]),
        correction, 'GAS_H', -1, lambda: torch.autograd.grad(None, None))
    assert observed['fallback_calls'] == ['torch.autograd.grad']
    assert observed['error']['type'] == 'AssertionError'
    assert torch.autograd.grad is original_grad


def test_genuine_cutoff_refusals_cannot_mask_true_connectivity_mismatch(receipt, audit_graph, monkeypatch):
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    from maple.function.calculator.analytic_protocol import named_array_sha256
    row = receipt['cutoff_control_records'][0]
    changed = {name: value.copy() for name, value in audit_graph.items()}
    changed['nbmat'][0, 0] = 0  # A real entry/edge change, not a permutation.
    row['point_identity'].update(neighbor_graph_sha256=canonical_neighbor_graph_sha256(audit_graph),
        neighbor_packing_sha256=named_array_sha256(audit_graph))
    row['gas_center']['native_reference_identity'].update(
        observed_neighbor_graph_sha256=canonical_neighbor_graph_sha256(changed),
        observed_neighbor_packing_sha256=named_array_sha256(changed))
    # Keep this synthetic test focused on semantic connectivity; point physics
    # is separately covered with complete real-shaped native_point fixtures.
    monkeypatch.setattr(validator, 'validate_cutoff_point_identity', lambda *args: None)
    validator.add_assessments(receipt)
    assert receipt['gates']['aimnet2025_cutoff_guard']['passed_count'] == 7
    assert row['status'] == 'error'
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.fixture
def audit_graph():
    # Full capacities, dummy row, duplicate values and sentinels are intentional.
    return {'numbers': np.array([6, 1, 0], dtype=np.int32),
        'mol_idx': np.array([0, 0, 0], dtype=np.int64),
        'nbmat': np.array([[1, 2, 2, -1], [0, 2, 2, -1], [2, 2, 2, -1]], dtype=np.int32),
        'nbmat_lr': np.array([[1, 2, 2], [0, 2, 2], [2, 2, 2]], dtype=np.int32)}


def test_canonical_audit_preserves_full_row_multisets_without_mutating_inputs(audit_graph):
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    from maple.function.calculator.analytic_protocol import named_array_sha256
    before = {name: array.copy() for name, array in audit_graph.items()}
    permuted = {name: array.copy() for name, array in audit_graph.items()}
    permuted['nbmat'] = permuted['nbmat'][:, [3, 1, 0, 2]]
    permuted['nbmat_lr'] = permuted['nbmat_lr'][:, [2, 0, 1]]
    assert canonical_neighbor_graph_sha256(audit_graph) == canonical_neighbor_graph_sha256(permuted)
    assert named_array_sha256(audit_graph) != named_array_sha256(permuted)
    assert all(np.array_equal(audit_graph[name], array) for name, array in before.items())


@pytest.mark.parametrize('change', ['edge', 'row', 'duplicate', 'sentinel', 'capacity', 'dtype', 'other_array'])
def test_canonical_audit_does_not_hide_connectivity_layout_or_value_drift(audit_graph, change):
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    changed = {name: array.copy() for name, array in audit_graph.items()}
    if change == 'edge':
        changed['nbmat'][0, 0] = 0
    elif change == 'row':
        changed['nbmat'] = changed['nbmat'][[1, 0, 2]]
    elif change == 'duplicate':
        changed['nbmat'][0, 1] = 1
    elif change == 'sentinel':
        changed['nbmat'][0, 3] = -2
    elif change == 'capacity':
        changed['nbmat'] = changed['nbmat'][:, :-1]
    elif change == 'dtype':
        changed['nbmat'] = changed['nbmat'].astype(np.int64)
    else:
        changed['mol_idx'][0] = 1
    assert canonical_neighbor_graph_sha256(audit_graph) != canonical_neighbor_graph_sha256(changed)


@pytest.mark.parametrize('bad', [np.ones(3, dtype=np.int32), np.ones((3, 4), dtype=np.float32),
    np.ones((3, 2, 2), dtype=np.int32), np.ones((3, 4), dtype=bool)])
def test_canonical_audit_rejects_unsupported_neighbor_layout(audit_graph, bad):
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    audit_graph['nbmat'] = bad
    with pytest.raises(ValueError):
        canonical_neighbor_graph_sha256(audit_graph)


def test_point_identity_requires_explicit_semantic_scheme_and_raw_packing(native_point):
    point, item = native_point
    point['neighbor_graph_scheme'] = 'raw-order-v1'
    with pytest.raises(ValueError):
        validator.validate_point_identity(point, item, 'cpu')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_center_accepts_only_full_row_permutation_with_both_raw_packings(receipt, audit_graph, device):
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    from maple.function.calculator.analytic_protocol import named_array_sha256
    row = deepcopy(next(r for r in receipt['gas_records'] if r['device'] == device))
    permuted = {name: value.copy() for name, value in audit_graph.items()}
    permuted['nbmat'] = permuted['nbmat'][:, ::-1].copy()
    row['point_identity'].update(neighbor_graph_sha256=canonical_neighbor_graph_sha256(audit_graph),
        neighbor_packing_sha256=named_array_sha256(audit_graph), neighbor_graph_scheme=validator.GRAPH_SCHEME)
    row['native_reference_identity'].update(observed_neighbor_graph_sha256=canonical_neighbor_graph_sha256(permuted),
        observed_neighbor_packing_sha256=named_array_sha256(permuted), neighbor_graph_scheme=validator.GRAPH_SCHEME)
    metrics, passed = validator.native_center_metrics(row, row['input'])
    assert passed and metrics['energy_abs'] == 0.
    assert row['point_identity']['neighbor_packing_sha256'] != row['native_reference_identity']['observed_neighbor_packing_sha256']


@pytest.mark.parametrize('field,value', [('observed_neighbor_packing_sha256', None),
    ('observed_neighbor_packing_sha256', 'bad-sha'), ('neighbor_graph_scheme', 'raw-order-v1')])
def test_native_reference_rejects_missing_raw_packing_or_wrong_graph_scheme(receipt, field, value):
    receipt['gas_records'][0]['native_reference_identity'][field] = value
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('field', ['neighbor_packing_sha256', 'neighbor_graph_scheme'])
def test_center_point_and_probe_require_raw_packing_and_scheme(native_point, field):
    point, item = native_point
    del point[field]
    with pytest.raises(ValueError):
        validator.validate_point_identity(point, item, 'cpu')


def test_native_producer_hashes_its_independently_captured_arrays_without_mutation(monkeypatch, audit_graph):
    from types import SimpleNamespace
    from maple.function.calculator.aimnet import _aimnet2025_calculator as adapter
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    from maple.function.calculator.analytic_protocol import named_array_sha256
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    state = lambda: {'unit': torch.tensor(0., dtype=torch.float32)}
    backend = SimpleNamespace(model=SimpleNamespace(state_dict=state),
        external_coulomb=SimpleNamespace(state_dict=state), external_dftd3=SimpleNamespace(state_dict=state))
    monkeypatch.setattr(adapter, '_configuration', lambda native: {'engine': 'unit', 'device': 'cpu'})
    before = {name: value.copy() for name, value in audit_graph.items()}
    result = module._native_reference_identity(backend, audit_graph)
    assert result['observed_neighbor_graph_sha256'] == canonical_neighbor_graph_sha256(audit_graph)
    assert result['observed_neighbor_packing_sha256'] == named_array_sha256(audit_graph)
    assert result['neighbor_graph_scheme'] == validator.GRAPH_SCHEME
    assert all(np.array_equal(audit_graph[name], value) for name, value in before.items())


def test_negative_guard_sequence_uses_one_constructor_and_no_readmission(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ase import Atoms
    from maple.function.calculator.analytic_protocol import AnalyticCapabilityError
    spec = importlib.util.spec_from_file_location('aim2025_candidate', ROOT / validator.HARNESS_REL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atoms = Atoms('H2', positions=[[0., 0., 0.], [4.6, 0., 0.]])
    built = []
    def fresh():
        provider = SimpleNamespace(hessian=lambda *a: None, kernel=SimpleNamespace(hessian=lambda *a: None))
        correction = SimpleNamespace(underlying_provider=provider, get_hessian=lambda *a: None,
            evaluate=lambda *a, **kw: SimpleNamespace(energy_hartree=0., forces_hartree_per_angstrom=np.zeros((2, 3))))
        calc = SimpleNamespace(solvent_correction=correction, analytic_implicit_derivatives_admitted=True,
            last_analytic_point_identity=deepcopy(TEST_POINT), inference_precision_provenance={'synthetic': True})
        def refuse(*args):
            if not calc.analytic_implicit_derivatives_admitted:
                raise AnalyticCapabilityError('AIMNet2025 composed analytic HVP requires parent receipt admission.')
            # Final A policy preserves the cell on geometry-only refusal.
            raise AnalyticCapabilityError(validator.CUTOFF_ERROR_PREFIX)
        calc.get_hessian = calc.get_hvp = refuse
        return calc
    gas = fresh()
    def construct(*args):
        calc = fresh();built.append(calc)
        return calc, atoms.copy(), {'identity': {}, 'provider_identity': {}}
    monkeypatch.setattr(module, '_construct_composed', construct)
    monkeypatch.setattr(module, '_gas_center', lambda row, *a: None)
    monkeypatch.setattr(module, '_center_with_components', lambda *a: (0., np.zeros((2, 3)),
        {'solvent_energy_hartree': 0., 'solvent_forces_hartree_per_angstrom': np.zeros((2, 3)).tolist()}))
    monkeypatch.setattr(module, 'atoms_from_input', lambda item: atoms.copy())
    item = {'case': 'nitralin', 'geometry': 'base'}
    vectors = {('nitralin', 'base', i): {'vector': np.eye(6)[i].tolist()} for i in range(3)}
    row = {}
    module._cutoff_control_row(row, item, 'cpu', 'ace', tmp_path, '/unused.pt', {}, {}, gas, vectors)
    assert len(built) == 1
    assert all(op['error']['message'] == validator.CUTOFF_ERROR_PREFIX for op in row['refusals'])


def test_independent_native_constructor_is_ieee_scoped_and_restores_caller():
    # Isolate Torch's process-wide legacy flag setters from the rest of pytest.
    # Do not mix selector writes with the unchanged standard Boolean guard API.
    import subprocess
    script = r"""
from types import SimpleNamespace
import importlib.util, sys, torch
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('candidate', root/'tests/solvation/run_aimnet2025_candidate_cell.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
observed = {}
def native(*args, **kwargs):
    observed.update(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
    return SimpleNamespace()
sys.modules['aimnet.calculators'] = SimpleNamespace(AIMNet2Calculator=native)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
backends = (torch.backends.cuda.matmul, torch.backends.cudnn, torch.backends.cudnn.conv, torch.backends.cudnn.rnn)
before = tuple(backend.fp32_precision for backend in backends)
module._construct_native_reference(SimpleNamespace(device='cuda:0'), '/unused.pt')
assert observed == {'matmul': False, 'cudnn': False}, observed
assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
assert tuple(backend.fp32_precision for backend in backends) == before
print('SCOPED_OK')
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.strip().endswith('SCOPED_OK')


def test_full_artifact_preserves_different_raw_packings_when_semantic_hash_matches(receipt):
    original = receipt['gas_records'][0]['point_identity']['neighbor_packing_sha256']
    observed = receipt['gas_records'][0]['native_reference_identity']
    observed['observed_neighbor_packing_sha256'] = 'f'*64
    assert original != observed['observed_neighbor_packing_sha256']
    assert validator.validate_artifact(_reseal(receipt))['aimnet2025_native_gas_identity']['passed']


def test_canonical_equivalence_does_not_waive_exact_component_closure(receipt):
    row = next(r for r in receipt['cutoff_control_records'] if r['device'] == 'cuda')
    row['gas_center']['native_reference_identity']['observed_neighbor_packing_sha256'] = 'f'*64
    row['energy_hartree'] = 8.76e-9
    row['forces_hartree_per_angstrom'][0][0] = 7.01e-8
    validator.add_assessments(receipt)
    assert row['metrics']['observed_expected_refusals'] == 5
    assert row['passed'] is False
    assert receipt['gates']['aimnet2025_cutoff_guard']['passed_count'] == 7
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('section,trace', [
    ('hessian_records','same_call_center'), ('hessian_records','same_call_hessian'),
    ('hvp_records','same_call_hvp'), ('cutoff_control_records','same_call_center')])
def test_same_call_operand_trace_is_mandatory(receipt,section,trace):
    receipt[section][0].pop(trace)
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_independent_native_repeat_jitter_does_not_replace_exact_same_call_math(receipt):
    gas=receipt['gas_records'][0]
    gas['energy_hartree']=gas['native_energy_hartree']=1e-8
    gas['forces_hartree_per_angstrom']=gas['native_forces_hartree_per_angstrom']=[[1e-8]*3]
    gas['raw_hessian_hartree_per_angstrom2']=np.eye(3).__mul__(1e-7).tolist()
    validator.add_assessments(receipt)
    assert receipt['passed']
    assert receipt['hessian_records'][0]['metrics']['center_force_closure']['max']==0
    assert receipt['hessian_records'][0]['metrics']['independent_center_force']['max']==1e-8
    validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('section',['hessian_records','cutoff_control_records'])
def test_actual_component_error_below_native_budget_still_fails_exact_accounting(receipt,section):
    row=receipt[section][0]
    row['forces_hartree_per_angstrom'][0][0]=1e-8
    validator.add_assessments(receipt)
    assert not row['passed']  # Native cross-call budget is not the addition budget.
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_hvp_wrong_actual_operand_is_not_hidden_by_Hv_tolerance(receipt):
    row=receipt['hvp_records'][0]
    row['same_call_hvp']['gas_derivatives'][1]['values'][0][0]=float(np.float32(1e-6))
    validator.add_assessments(receipt)
    assert not row['passed']
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


@pytest.mark.parametrize('bad',[0,2,True,None])
def test_same_call_counts_are_exact_integers(receipt,bad):
    receipt['hessian_records'][0]['same_call_center']['finalize_calls']=bad
    with pytest.raises(ValueError):
        validator.validate_artifact(_reseal(receipt))


def test_joint_fp32_revision_keeps_old_profile_and_declares_revised_bounds():
    profile = validator.load_profile()
    assert validator.SCHEMA_VERSION == 3
    assert profile['profile_id'] == 'aimnet2025-obc2-fp32-engineering-v3'
    assert profile['solvent_dtype'] == 'float32'
    assert profile['supersedes_for_new_branch_only']['content_sha256'] == 'c118376bac0a58bbf1682abaf573f3c521c843cd52c912b388c3a839de96c835'
    assert 'user-authorized' in profile['status']
    assert validator.MAX_BOUND == .002 and validator.RMS_BOUND == .0001


def test_relaxed_absolute_gate_does_not_mask_large_relative_curvature_error():
    actual = np.eye(3)*.01
    reference = actual + np.eye(3)*.00009
    metrics = validator._response_difference(actual, reference)
    assert validator._bounded(metrics)
    assert not validator._response_bounded(metrics)


def test_spectral_diagnostic_preserves_raw_asymmetry():
    actual = np.array([[1., .0001], [0., 2.]])
    before = actual.copy()
    metrics = validator._response_difference(actual, actual.copy())
    np.testing.assert_array_equal(actual, before)
    assert metrics['symmetric_eigenvalue_max'] == 0.
    assert actual[0, 1] != actual[1, 0]


@pytest.mark.parametrize('reject', [False, True])
def test_replay_cli_establishes_and_restores_its_ieee_profile(monkeypatch, tmp_path, reject):
    import torch
    artifact = tmp_path / 'artifact.json'
    artifact.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['replay', '--validate-only', '--output', str(artifact)])
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', True)

    def validate(value):
        assert torch.is_grad_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        if reject:
            raise ValueError('raw evidence rejected')
        return {}

    monkeypatch.setattr(validator, 'validate_artifact', validate)
    with torch.no_grad():
        if reject:
            with pytest.raises(ValueError, match='raw evidence rejected'):
                validator.main()
        else:
            validator.main()
        assert not torch.is_grad_enabled()
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
