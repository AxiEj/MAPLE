#!/usr/bin/env python3
"""Replay the versioned joint FP32 engineering matrix, never installed admission.

No model inference or hydration labels are needed to validate raw derivatives.
The existing OBC/ANI gates are deliberately outside this additive validator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import validate_independent_obc2_default as qualification  # noqa: E402

core = qualification.core
GRAPH_SCHEME = 'row-multiset-with-padding-v1'
SCHEMA_VERSION = 3
ARTIFACT_TYPE = 'aimnet2025-candidate-qualification-v3'
STAGE = 'candidate qualification, not installed qualification'
HARNESS_REL = 'tests/solvation/run_aimnet2025_candidate_cell.py'
VALIDATOR_REL = 'docs/implicit-solvation/benchmarks/validate_aimnet2025_analytic.py'
CAMPAIGN = ROOT / '.omx/benchmarks/route1-aimnet2025-performance-20261025'
CORPUS = ROOT / '.omx/benchmarks/route1-aimnet2025-20261024'
CHECKPOINT = Path.home() / '.cache/aimnet/aimnet2_2025_b973c_d3_0.pt'
CHECKPOINT_SHA256 = '043ed5418a104e31f79462f8e5ebeca64a2d24422174f5d29f894d32271981b5'
PROFILE_ARCHIVE = HERE / 'aimnet2025-profiles'
PROFILE_PATH = PROFILE_ARCHIVE / 'JOINT_FP32_QUALIFICATION_PROFILE_V3.json'
PROFILE_CONTENT_SHA256 = '19060417d315e994d3cb4a87e0db261b62537b9c75b3cb0b2daa5d6c0a23352e'
FROZEN_HASHES = {
    'manifest.json': ('af9c95401d445af8f4ab1922a7222024efe05217f601525309e6768bb26d537f',
                      'be1235fc9bd2408660119c72827c45fc1411025754e53ab8a029f59fe47e4507'),
    'directions.json': ('db75d2d9dac8a0990ab1b058baaaf8d1407920898d4dcb6b3e73f342e2701da7',
                        'b6c2959540f9c114c842d710cca4cd1f5c0f9c059dada22fb250bd92df271ac6')}
V2_FROZEN_HASHES = {
    'manifest.json': ('39b5444fadcf910d17bf672f78119163986f56727f19fa8e8e63e5e0d4a96760',
                      '52dd06cbc60c936d08cedf55b02600c35fad5fd3429e067824cd295bf17d96c9'),
    'directions.json': ('a1d43b73ed7007aaf1b1218aa7c3e5bcf5baa2adf56932c30437bc812632fa81',
                        'd22b7be8db6081752a02ec124322d1b9bef456633cc1727237fed86b934d60fd'),
    'cutoff-controls.json': ('213568b8cdd0d5d011ba9209c159666073aa60114104fd88a79ea23e8935d32e',
                           'b66c398874b8d18e0fc427e5bab94c5b4e425028fca73c19c81f4405c3506cd2')}
CUTOFF_PLAN_REL = '.omx/checkpoints/route1-aimnet2025-20261024/CUTOFF_PROTOCOL_V2_PLAN.md'
CUTOFF_PLAN_SHA256 = 'f5c758b05879a6fe7a65128ecd345bcd7d3c4a5ab8d07dde50a6259142e33bee'
ACTIVE_CUTOFFS = (4.599999904632568, 5., 12., 15.)
CUTOFF_GUARD = .003
CUTOFF_ERROR_MODULE = 'maple.function.calculator.analytic_protocol'
CUTOFF_ERROR_CLASS = 'AnalyticCapabilityError'
CUTOFF_ERROR_PREFIX = 'AIMNet2025 analytic curvature refused within native/original cutoff guard.'
CUTOFF_FORBIDDEN = ('torch.autograd.grad', 'calculator_base.numerical_hessian_from_atoms',
                    'torch.autograd.functional.hessian', 'torch.func.hessian',
                    'solvent.get_hessian', 'solvent.provider.hessian', 'solvent.kernel.hessian')
CASES = ('water', 'methanol', 'ethane', 'benzene', 'methyl-hexanoate', 'nitralin')
DEVICES = ('cpu', 'cuda')
NONPOLAR = ('ace', 'none')
STEPS = (0.002, 0.001, 0.0005)
EXPECTED_COUNTS = {'aimnet2025_native_gas_identity': 24,
                   'aimnet2025_composed_hessian': 48, 'aimnet2025_direct_hvp': 144,
                   'aimnet2025_cutoff_guard': 8}
RMS_BOUND, MAX_BOUND = 1e-4, 2e-3
DIRECTIONAL_RMS_BOUND = 3e-4
RELATIVE_BOUND, NORM_FLOOR = .005, 1e-3
# Complete native configuration/state captured in the pre-formal two-device
# helper smoke; device is bound separately. These are identity pins, not gates.
EXPECTED_NATIVE_IDENTITY = {
    'checkpoint_sha256': CHECKPOINT_SHA256,
    'state_sha256': 'b6e471d65b6c53f58dddfd63becf45fb7bf6377276da745a536079d9a1ae9c47',
    'configuration_sha256_without_device': '71a53d4c66a29aea1b0343c68a65e56b487764ab234e324f46518b591501685c',
    'effective_ad_class': 'N32', 'effective_nn_dtype': 'float32',
    'atomic_shift_dtype': 'float64', 'energy_accumulation_dtype': 'float64'}


def digest(value):
    return core.sha256_bytes(core.canonical_json_bytes(value))


def require_equal(actual, expected, label):
    # JSON canonical equality also rejects bool-as-int and tuple/list surprises.
    if core.canonical_json_bytes(actual) != core.canonical_json_bytes(expected):
        raise ValueError(f'{label} mismatch')


def _mapping(value, label):
    if not isinstance(value, dict) or not value:
        raise ValueError(f'{label} must be a nonempty mapping')
    return value


def _sha(value, label):
    if not qualification._valid_sha256(value):
        raise ValueError(f'{label} SHA256 is malformed')


def array(value, shape, label):
    objects = np.asarray(value, dtype=object)
    if any(isinstance(v, (bool, str, bytes)) for v in objects.flat):
        raise ValueError(f'{label} contains nonnumeric entries')
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape != shape:
        raise ValueError(f'{label} must be a numeric array of shape {shape}')
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{label} contains nonfinite values')
    return result


def scalar(value, label):
    return float(array(value, (), label))


def _array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def load_profile():
    value = json.loads(PROFILE_PATH.read_text())
    require_equal(core.artifact_content_sha256(value), PROFILE_CONTENT_SHA256, 'joint FP32 profile')
    require_equal(value.get('content_sha256'), PROFILE_CONTENT_SHA256, 'joint FP32 self hash')
    protocol = value['based_on_preexisting_protocol']
    require_equal(core.sha256_file(PROFILE_ARCHIVE / Path(protocol['path']).name), protocol['sha256'], 'historical protocol file')
    previous = value['supersedes_for_new_branch_only']
    historical = json.loads((PROFILE_ARCHIVE / Path(previous['path']).name).read_text())
    require_equal(core.artifact_content_sha256(historical), previous['content_sha256'], 'unchanged historical N32 profile')
    require_equal(value['gates']['force_fd_hessian_rms_hartree_per_angstrom2'], RMS_BOUND, 'FP32 RMS gate')
    require_equal(value['gates']['force_fd_hessian_max_hartree_per_angstrom2'], MAX_BOUND, 'FP32 max gate')
    require_equal(value['gates']['force_fd_relative_l2'], RELATIVE_BOUND, 'FP32 relative gate')
    require_equal(value['gates']['force_fd_relative_norm_floor'], NORM_FLOOR, 'relative norm floor')
    require_equal(value['gates']['directional_force_fd_hvp_rms_hartree_per_angstrom2'], DIRECTIONAL_RMS_BOUND, 'directional RMS gate')
    intermediate = value['intermediate_profile']
    require_equal(core.sha256_file(PROFILE_ARCHIVE / Path(intermediate['path']).name), intermediate['file_sha256'], 'retained intermediate profile')
    require_equal(value['solvent_dtype'], 'float32', 'actual OBC kernel precision')
    return value


def _frozen_content_sha(value):
    # freeze_inputs.py predates the common helper and used ensure_ascii=True.
    # Replay that documented encoding without changing any frozen input bytes.
    payload = {k: v for k, v in value.items() if k != 'content_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def _load_corpus(folder, hashes):
    """Verify a pinned corpus without selecting or changing any geometry."""
    values, files = {}, {}
    for name, (file_sha, content_sha) in hashes.items():
        path = CORPUS / folder / name
        require_equal(core.sha256_file(path), file_sha, f'frozen {name} bytes')
        value = json.loads(path.read_text())
        require_equal(value.get('content_sha256'), content_sha, f'frozen {name} self hash')
        require_equal(_frozen_content_sha(value), content_sha, f'frozen {name} content')
        values[name] = value
        files[str(path.relative_to(ROOT))] = file_sha
    manifest, directions = values['manifest.json'], values['directions.json']
    expected = {(c, g) for c in CASES for g in ('base', 'displaced')}
    records = _index(manifest['records'], ('case', 'geometry'), expected)
    for item in records.values():
        z = np.asarray(item['atomic_numbers'])
        if z.dtype.kind not in 'iu' or z.ndim != 1 or not len(z) or np.any(z <= 0):
            raise ValueError('Frozen atomic numbers malformed')
        x = array(item['positions_angstrom'], (len(z), 3), 'frozen positions')
        q = array(item['ordered_charges_e'], (len(z),), 'frozen charges')
        require_equal(_array_sha(x), item['position_array_sha256'], 'frozen position array hash')
        require_equal(_array_sha(q), item['charge_array_sha256'], 'frozen charge array hash')
        require_equal(_frozen_content_sha(item), item['content_sha256'], 'frozen record self hash')
        provenance = [(item['source_mol2_relative_path'], item['source_mol2_sha256'])]
        source = item['source']
        if 'input_xyz' in source:
            provenance.append((source['input_xyz'], source['input_xyz_sha256']))
        if 'fixture_relative_path' in source:
            provenance.append((source['fixture_relative_path'], source['fixture_sha256']))
        for relative, expected_sha in provenance:
            path = (ROOT / relative).resolve()
            path.relative_to(ROOT)
            require_equal(core.sha256_file(path), expected_sha, 'frozen provenance file')
            files[relative] = expected_sha
    expected_directions = {key + (i,) for key in expected for i in range(3)}
    vectors = _index(directions['records'], ('case', 'geometry', 'direction_index'), expected_directions)
    for key, row in vectors.items():
        vector = array(row['vector'], (3*len(records[key[:2]]['atomic_numbers']),), 'frozen direction')
        if abs(np.linalg.norm(vector)-1.) > 1e-14:
            raise ValueError('Frozen direction must be normalized')
        require_equal(_array_sha(vector), row['vector_sha256'], 'frozen direction array hash')
    require_equal(directions['input_manifest_sha256'], hashes['manifest.json'][0], 'direction manifest binding')
    return {'manifest': manifest, 'directions': directions, 'cutoff_controls': values.get('cutoff-controls.json'),
            'snapshot': {'files_sha256': files, 'manifest_content_sha256': manifest['content_sha256'],
                         'directions_content_sha256': directions['content_sha256'],
                         'serialized_arrays_sha256': digest(manifest['records'])}}


def cutoff_gap(item):
    """Recompute the adapter's original/native-coordinate guard, not its label."""
    x = np.asarray(item['positions_angstrom'], dtype=np.float64)
    gaps = []
    for positions in (x, x.astype(np.float32).astype(np.float64)):
        for i in range(len(positions)-1):
            distance = np.linalg.norm(positions[i+1:]-positions[i], axis=1)
            gaps.extend(float(np.min(np.abs(distance-cutoff))) for cutoff in ACTIVE_CUTOFFS)
    return min(gaps) if gaps else None


def _selection_margins(x):
    margins = {}
    for name, dtype in (('original_float64', np.float64), ('native_float32', np.float32)):
        coordinates = np.asarray(x, dtype=dtype)
        distances = np.linalg.norm(coordinates[:, None, :]-coordinates[None, :, :], axis=-1)
        off_diagonal = ~np.eye(len(coordinates), dtype=bool)
        margins[name] = {str(c): float(np.min(np.abs(distances[off_diagonal]-c))) for c in ACTIVE_CUTOFFS}
    return margins


def validate_positive_geometry(item, original):
    gap = cutoff_gap(item)
    if gap is not None and gap <= CUTOFF_GUARD:
        raise ValueError('Original cutoff-ineligible geometry cannot be a positive numerical row')
    if item['case'] != 'nitralin':
        require_equal(item, original, 'ten literal original positive geometries')
        return
    transform = _mapping(item.get('qualification_geometry_transform'), 'explicit pre-formal geometry lineage')
    x = np.asarray(original['positions_angstrom'], dtype=np.float64)
    center = np.mean(x, axis=0)
    expected = (x-center)*.97+center  # Verify the already-frozen choice; never select a new one.
    require_equal(item['positions_angstrom'], expected.tolist(), 'fixed pre-formal .97 coordinates')
    require_equal(transform, {'kind': 'geometry-only-centroid-scale-before-formal-derivatives',
        'scale': .97, 'original_record_content_sha256': original['content_sha256'],
        'original_position_array_sha256': original['position_array_sha256'],
        'centroid_angstrom': center.tolist(), 'selection_margins': _selection_margins(expected),
        'selection_plan_relative_path': CUTOFF_PLAN_REL}, 'explicit fixed geometry transform')
    for values in transform['selection_margins'].values():
        if not all(value > .004 for value in values.values()):
            raise ValueError('Pre-formal selected geometry lacks its unchanged eligibility margin')
    unchanged = {k: v for k, v in item.items() if k not in
                 ('positions_angstrom', 'position_array_sha256', 'content_sha256', 'qualification_geometry_transform')}
    require_equal(unchanged, {k: v for k, v in original.items() if k not in
                  ('positions_angstrom', 'position_array_sha256', 'content_sha256')}, 'unchanged original elements/q/provenance')


def load_frozen_inputs():
    original = _load_corpus('frozen-inputs', FROZEN_HASHES)
    positive = _load_corpus('frozen-inputs-v2', V2_FROZEN_HASHES)
    manifest = positive['manifest']
    old = {(r['case'], r['geometry']): r for r in original['manifest']['records']}
    for item in manifest['records']:
        validate_positive_geometry(item, old[item['case'], item['geometry']])
    require_equal(manifest['original_manifest_relative_path'],
                  str((CORPUS/'frozen-inputs/manifest.json').relative_to(ROOT)), 'original manifest location')
    require_equal(manifest['original_manifest_sha256'], FROZEN_HASHES['manifest.json'][0], 'original manifest lineage')
    selection = manifest['geometry_selection']
    require_equal(selection['chosen_scale'], .97, 'fixed geometry-only scale')
    require_equal(selection['no_derivative_or_energy_results_used'], True, 'pre-formal geometric selection')
    require_equal(selection['minimum_original_and_native_margin_angstrom'], .004, 'selection-only margin')
    require_equal(selection['active_cutoffs_angstrom'], list(ACTIVE_CUTOFFS), 'unchanged active boundaries')
    require_equal([r['scale'] for r in selection['attempts']], [1.01, .99, 1.02, .98, 1.03, .97], 'six frozen attempts')
    require_equal([r['geometry_only_passed'] for r in selection['attempts']], [False]*5+[True], 'fixed first eligible choice')
    for before, after in zip(original['directions']['records'], positive['directions']['records']):
        for field in ('case', 'geometry', 'direction_index', 'seed', 'vector', 'vector_sha256'):
            require_equal(after[field], before[field], 'unchanged original frozen unit direction')
    controls = positive['cutoff_controls']
    require_equal(controls['records'], [r for r in old.values() if r['case'] == 'nitralin'], 'literal original cutoff controls')
    for name in ('manifest', 'directions'):
        require_equal(controls[f'original_{name}_relative_path'],
                      str((CORPUS/'frozen-inputs'/f'{name}.json').relative_to(ROOT)), 'negative source location')
        require_equal(controls[f'original_{name}_sha256'], FROZEN_HASHES[f'{name}.json'][0], 'negative source binding')
    require_equal(controls['runtime_cutoff_guard_margin_angstrom'], CUTOFF_GUARD, 'unchanged runtime guard')
    require_equal(controls['expected_device_profile_rows'], 8, 'negative control denominator')
    for item in controls['records']:
        if cutoff_gap(item) is None or cutoff_gap(item) > CUTOFF_GUARD:
            raise ValueError('Original negative control must remain inside unchanged cutoff guard')
    require_equal(core.sha256_file(ROOT/CUTOFF_PLAN_REL), CUTOFF_PLAN_SHA256, 'pre-formal plan bytes')
    snapshot = positive['snapshot']
    snapshot['files_sha256'].update(original['snapshot']['files_sha256'])
    snapshot['files_sha256'][CUTOFF_PLAN_REL] = CUTOFF_PLAN_SHA256
    snapshot.update(original_manifest_content_sha256=original['manifest']['content_sha256'],
        original_directions_content_sha256=original['directions']['content_sha256'],
        cutoff_controls_content_sha256=controls['content_sha256'],
        original_serialized_arrays_sha256=digest(original['manifest']['records']))
    return {**positive, 'original_manifest': original['manifest'], 'original_directions': original['directions'],
            'cutoff_controls': controls, 'snapshot': snapshot}


def checkpoint_snapshot(path=None):
    path = Path(path or CHECKPOINT).resolve()
    value = {'path': str(path), 'sha256': core.sha256_file(path), 'size_bytes': path.stat().st_size}
    require_equal(value['sha256'], CHECKPOINT_SHA256, 'official member0 checkpoint')
    require_equal(value['size_bytes'], 8839102, 'official member0 checkpoint size')
    return value


def runtime_snapshot():
    """Fresh external source and actual runtime settings; no inference or writes."""
    from importlib import metadata
    from maple.function.calculator.aimnet._aimnet2025_identity import bind_external_runtime
    from maple.function.calculator.analytic_protocol import torch_response_settings
    import torch
    external, _ = bind_external_runtime()
    package = Path(metadata.distribution('aimnet').locate_file('aimnet'))
    files = {str(p.relative_to(package)): core.sha256_file(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.yaml', '.pt')}
    return {'python': sys.version, 'external_runtime': external, 'aimnet_files_sha256': files,
            'default_dtype': str(torch.get_default_dtype()),
            'settings': {d: {k: json.loads(encoded) for k, encoded in torch_response_settings(torch.device(d))}
                         for d in DEVICES},
            'devices': {'cpu': {'type': 'cpu'}, 'cuda': {'available': torch.cuda.is_available(),
                'index': torch.cuda.current_device() if torch.cuda.is_available() else None,
                'name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'cuda_runtime': torch.version.cuda}}}


def validate_runtime_snapshot(value):
    require_equal(value, runtime_snapshot(), 'live full external/runtime fingerprint')
    require_equal(value['default_dtype'], 'torch.float32', 'native default dtype')
    for device in DEVICES:
        state = value['settings'][device]
        for key in ('autocast_enabled', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'inference_mode_enabled'):
            require_equal(state[key], False, f'{device} {key}')
        require_equal(state['grad_enabled'], True, 'grad-enabled public scalar')


def validate_identity(identity, device, runtime):
    from maple.function.calculator.aimnet._aimnet2025_identity import (
        MODEL_METADATA, PROFILE_ID, REGISTRY_KEY, NEIGHBOR_GRAPH_SCHEME, PRECISION_EXECUTION_POLICY)
    _mapping(identity, 'adapter cell identity')
    for key, expected in {'model': 'aimnet2025', 'registry_key': REGISTRY_KEY, 'profile_id': PROFILE_ID,
            'checkpoint_sha256': CHECKPOINT_SHA256, 'effective_nn_dtype': 'float32', 'effective_ad_class': 'N32',
            'atomic_shift_dtype': 'float64', 'energy_accumulation_dtype': 'float64', 'd4': False,
            'source_energy_unit': 'eV', 'external_runtime': runtime['external_runtime'],
            'neighbor_graph_scheme': NEIGHBOR_GRAPH_SCHEME, 'precision_execution_policy': PRECISION_EXECUTION_POLICY}.items():
        require_equal(identity.get(key), expected, f'adapter {key}')
    actual_device = identity.get('device')
    require_equal(actual_device, 'cpu' if device == 'cpu' else f"cuda:{runtime['devices']['cuda']['index']}", 'actual device')
    _sha(identity.get('state_sha256'), 'native state')
    config = _mapping(identity.get('native_configuration'), 'native complete potential')
    for key, expected in {'engine': 'official aimnet public scalar; forces=False,hessian=False; requires_grad leaf',
            'train': False, 'compiled': False, 'lr': True, 'cutoff': 5.0, 'cutoff_lr': 'inf',
            'coulomb_cutoff': 'inf', 'coulomb_method': 'simple', 'd3_cutoff': 15.0}.items():
        require_equal(config.get(key), expected, f'complete potential {key}')
    for key, expected in MODEL_METADATA.items():
        require_equal(config.get('metadata', {}).get(key), expected, f'native metadata {key}')
    require_equal(config.get('d3'), dict(MODEL_METADATA['d3_params'], smoothing_on=12.0,
                                      smoothing_off=15.0, key_out='energy'), 'complete D3BJ')
    for term in ('coulomb', 'embedded_sr'):
        c = _mapping(config.get(term), term)
        for key, expected in {'rc': MODEL_METADATA['coulomb_sr_rc'], 'envelope': 'exp',
                              'key_in': 'charges', 'key_out': 'energy'}.items():
            require_equal(c.get(key), expected, f'{term} {key}')
        scalar(c.get('factor'), f'{term} factor')
    require_equal(config['coulomb'].get('subtract_sr'), False, 'full Coulomb not SR subtraction')
    require_equal(config['coulomb'].get('method'), 'simple', 'full Coulomb method')
    require_equal(config.get('input_dtypes', {}).get('coord'), 'torch.float32', 'public native coordinates')
    if not config.get('model_input_dtypes') or config['model_input_dtypes'][0] != 'torch.float32':
        raise ValueError('Native NN dtype fingerprint missing')
    _mapping(config.get('neighbor_cutoffs'), 'neighbor cutoff configuration')
    require_equal(config.get('device'), actual_device, 'complete configuration device')
    unplaced = {k: v for k, v in config.items() if k != 'device'}
    require_equal(digest(unplaced), EXPECTED_NATIVE_IDENTITY['configuration_sha256_without_device'],
                  'complete native potential/configuration fingerprint')
    require_equal(identity['state_sha256'], EXPECTED_NATIVE_IDENTITY['state_sha256'],
                  'official native model/Coulomb/D3 state fingerprint')


def _validate_point_identity(point, item, device, *, cutoff_control):
    from maple.function.calculator.analytic_protocol import named_array_sha256
    _mapping(point, 'actual point/graph identity')
    x = np.asarray(item['positions_angstrom'], dtype='<f8')
    z = np.asarray(item['atomic_numbers'], dtype=np.int64)
    for key, expected in {'geometry_sha256': named_array_sha256({'atomic_numbers': z, 'positions_angstrom': x}),
            'native_geometry_sha256': named_array_sha256({'atomic_numbers': z, 'positions_angstrom': x.astype(np.float32)}),
            'native_coordinate_dtype': 'torch.float32', 'atomic_numbers': z.tolist(), 'charge': 0.,
            'multiplicity': 1, 'total_energy_dtype': 'torch.float64'}.items():
        # Charge may be represented as integer 0 by a future official API.
        if key == 'charge':
            require_equal(scalar(point.get(key), key), expected, key)
        else:
            require_equal(point.get(key), expected, f'point {key}')
    require_equal(point.get('max_coordinate_quantization_angstrom'),
                  float(np.max(np.abs(x-x.astype(np.float32).astype(np.float64)))), 'coordinate quantization')
    _sha(point.get('neighbor_graph_sha256'), 'actual semantic neighbor graph')
    _sha(point.get('neighbor_packing_sha256'), 'actual raw neighbor packing')
    require_equal(point.get('neighbor_graph_scheme'), GRAPH_SCHEME, 'explicit semantic graph audit scheme')
    intermediates = {key: 'torch.float32' for key in ('coord', 'charge', 'mult', 'a', 'd_ij', 'g_sv',
                        '_delta_Q', 'charges_pre', '_dQ', 'charges', 'aim')}
    intermediates['energy'] = 'torch.float64'
    require_equal(point.get('intermediate_dtypes'), intermediates, 'actual native intermediate precision')
    require_equal(point.get('active_cutoffs_angstrom'), {'radial': 5., 'embedded_sr': 4.599999904632568,
                  'd3_smoothing_on': 12., 'd3_smoothing_off': 15.}, 'complete active cutoffs')
    require_equal(point.get('cutoff_guard_margin_angstrom'), .003, 'native cutoff guard')
    expected_gap = cutoff_gap(item)
    require_equal(point.get('minimum_cutoff_gap_angstrom'), expected_gap, 'actual native/original cutoff gap')
    if cutoff_control:
        if expected_gap is None or expected_gap > CUTOFF_GUARD:
            raise ValueError('Negative control is not inside the unchanged cutoff guard')
    elif expected_gap is not None and expected_gap <= CUTOFF_GUARD:
        raise ValueError('Point is inside native cutoff guard')
    state = _mapping(point.get('response_settings'), 'actual response settings')
    for key in ('autocast_enabled', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'inference_mode_enabled'):
        require_equal(state.get(key), False, f'point {key}')
    require_equal(state.get('grad_enabled'), True, 'point grad enabled')
    require_equal(state.get('device_type'), device, 'point device')


def validate_point_identity(point, item, device):
    _validate_point_identity(point, item, device, cutoff_control=False)


def validate_cutoff_point_identity(point, item, device):
    _validate_point_identity(point, item, device, cutoff_control=True)


def _index(rows, fields, expected):
    if not isinstance(rows, list):
        raise ValueError('Raw records must be a list')
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Raw record must be a mapping')
        for field in fields:
            value = row.get(field)
            expected_type = int if field in ('axis', 'direction_index') else str
            if type(value) is not expected_type:
                raise ValueError(f'Raw cell {field} has the wrong scalar type')
        key = tuple(row.get(f) for f in fields)
        try:
            if key in result or key not in expected:
                raise ValueError(f'Duplicate or unexpected raw cell: {key}')
            result[key] = row
        except TypeError as exc:
            raise ValueError('Raw cell keys must be scalar identities') from exc
    if set(result) != expected:
        raise ValueError('Missing expected raw cells; full denominator is mandatory')
    return result


def _error_metrics(row):
    error = _mapping(row.get('error'), 'failed row error')
    for field in ('type', 'message', 'traceback'):
        if not isinstance(error.get(field), str) or not error[field]:
            raise ValueError('Failed row must preserve exception type/message/traceback')
    return {'execution_error': True}, False


def _delta(actual, reference):
    difference = actual-reference
    return {'rms': float(np.sqrt(np.mean(difference**2))), 'max': float(np.max(np.abs(difference)))}


def _bounded(metrics, factor=1., *, rms_bound=RMS_BOUND):
    return metrics['rms'] <= factor*rms_bound and metrics['max'] <= factor*MAX_BOUND


def _response_difference(actual, reference):
    """Add scale-aware checks; this never replaces absolute-error limits."""
    metrics = _delta(actual, reference)
    metrics['relative_l2'] = float(np.linalg.norm(actual-reference) / max(np.linalg.norm(actual), NORM_FLOOR))
    if actual.ndim == 2:
        # Raw H asymmetry has its own stricter gate. Symmetrization here is only
        # for the independent eigenvalue diagnostic, never HVP or composition.
        eigenvalues = np.linalg.eigvalsh((actual+actual.T)*.5)
        finite_difference = np.linalg.eigvalsh((reference+reference.T)*.5)
        metrics['symmetric_eigenvalue_max'] = float(np.max(np.abs(eigenvalues-finite_difference)))
    return metrics


def _response_bounded(metrics, *, directional=False):
    return (_bounded(metrics, rms_bound=DIRECTIONAL_RMS_BOUND if directional else RMS_BOUND)
            and metrics['relative_l2'] <= RELATIVE_BOUND
            and metrics.get('symmetric_eigenvalue_max', 0.) <= MAX_BOUND)


def _probe(row, item, direction, step, device):
    x = np.asarray(item['positions_angstrom'], dtype=np.float64)
    v = np.asarray(direction).reshape(x.shape)
    for sign, multiplier in (('plus', 1.), ('minus', -1.)):
        expected = x + multiplier*step*v
        actual = array(row.get(f'positions_{sign}_angstrom'), x.shape, 'actual FD inputs')
        if not np.array_equal(actual, expected):
            raise ValueError('FD probe must use exact frozen coordinates and predeclared displacement')
        probe_item = dict(item, positions_angstrom=expected.tolist())
        validate_point_identity(row.get(f'point_{sign}'), probe_item, device)
    plus = array(row.get('forces_plus_hartree_per_angstrom'), x.shape, 'raw FD plus force')
    minus = array(row.get('forces_minus_hartree_per_angstrom'), x.shape, 'raw FD minus force')
    return -(plus-minus).reshape(-1)/(2.*step)


def native_center_metrics(row, item):
    n = len(item['atomic_numbers'])
    e = scalar(row.get('energy_hartree'), 'gas energy')
    native = scalar(row.get('native_energy_hartree'), 'official native gas energy')
    f = array(row.get('forces_hartree_per_angstrom'), (n, 3), 'gas force')
    nf = array(row.get('native_forces_hartree_per_angstrom'), (n, 3), 'official native gas force')
    potential = {k: v for k, v in row['identity']['native_configuration'].items() if k != 'engine'}
    reference = _mapping(row.get('native_reference_identity'), 'independent observed native identity')
    _sha(reference.get('observed_neighbor_graph_sha256'), 'observed semantic native graph')
    _sha(reference.get('observed_neighbor_packing_sha256'), 'observed raw native packing')
    require_equal(reference, {
        'implementation': 'aimnet.calculators.calculator.AIMNet2Calculator',
        'operation': 'official forces=True,hessian=False; native input without requires_grad',
        'checkpoint_sha256': CHECKPOINT_SHA256, 'potential_configuration': potential,
        'observed_neighbor_graph_sha256': row['point_identity'].get('neighbor_graph_sha256'),
        'observed_neighbor_packing_sha256': reference['observed_neighbor_packing_sha256'],
        'neighbor_graph_scheme': GRAPH_SCHEME,
        'state_sha256': row['identity']['state_sha256']}, 'independent official native E/F identity')
    metrics = {'energy_abs': abs(e-native), 'force': _delta(f, nf)}
    return metrics, (metrics['energy_abs'] <= 2e-5 and metrics['force']['max'] <= 2e-6
                     and metrics['force']['rms'] <= 5e-7)


def gas_metrics(row, item):
    n = len(item['atomic_numbers'])
    array(row.get('raw_hessian_hartree_per_angstrom2'), (3*n, 3*n), 'gas raw Hessian')
    return native_center_metrics(row, item)


def _one_call(trace, key):
    if type(trace.get(key)) is not int or trace[key] != 1:
        raise ValueError(f'Same-call accounting requires exactly one {key}')


def _trace_point(trace, row, item, *, cutoff=False):
    point = _mapping(trace.get('point_identity'), 'observed operand point identity')
    check = validate_cutoff_point_identity if cutoff else validate_point_identity
    check(point, item, row['device'])
    require_equal(point.get('neighbor_graph_sha256'), row['point_identity'].get('neighbor_graph_sha256'),
                  'same mathematical center graph for observed operands')


def _same_call_center_metrics(row, item, *, cutoff=False):
    from maple.function.calculator.calculator_base import _convert_energy_force_units
    trace = _mapping(row.get('same_call_center'), 'actual same-call center operands')
    require_equal(trace.get('schema'), 'same-call-ef-operands-v1', 'center observation schema')
    _one_call(trace, 'finalize_calls'); _one_call(trace, 'solvent_calls')
    require_equal(trace.get('gas_source_unit'), 'eV', 'actual model source units')
    require_equal(trace.get('gas_forces_source_dtype'), 'float64', 'promotion before shared composition')
    shape = (len(item['atomic_numbers']), 3)
    raw_energy = scalar(trace.get('gas_energy_source'), 'actual finalizer gas energy')
    raw_forces = array(trace.get('gas_forces_source'), shape, 'actual finalizer gas forces')
    gas_e, gas_f = _convert_energy_force_units(raw_energy, raw_forces, source_unit='eV')
    se = scalar(trace.get('solvent_energy_hartree'), 'actual same-call solvent energy')
    sf = array(trace.get('solvent_forces_hartree_per_angstrom'), shape, 'actual same-call solvent forces')
    require_equal(se, row['solvent_energy_hartree'], 'recorded actual solvent energy return')
    require_equal(sf.tolist(), row['solvent_forces_hartree_per_angstrom'], 'recorded actual solvent force return')
    _trace_point(trace, row, item, cutoff=cutoff)
    return {'center_energy_closure': abs(row['energy_hartree']-(gas_e+se)),
            'center_force_closure': _delta(row['forces_hartree_per_angstrom'], gas_f+sf)}


def _validate_hessian_observation(row, item, gas_h, solvent_h):
    from maple.function.calculator.analytic_protocol import named_array_sha256
    trace = _mapping(row.get('same_call_hessian'), 'actual Hessian return observation')
    require_equal(trace.get('schema'), 'same-call-hessian-operands-v1', 'H observation schema')
    _one_call(trace, 'gas_calls'); _one_call(trace, 'solvent_calls')
    for key, value in [('gas_hessian_sha256', gas_h), ('solvent_hessian_sha256', solvent_h)]:
        require_equal(trace.get(key), named_array_sha256({'component': value}), 'observed actual H component')
    _trace_point(trace, row, item)


def _same_call_hvp_metrics(row, item):
    from maple.function.calculator.calculator_base import EV2HARTREE
    trace = _mapping(row.get('same_call_hvp'), 'actual HVP operand observation')
    require_equal(trace.get('schema'), 'same-call-hvp-operands-v1', 'HVP observation schema')
    require_equal(trace.get('gas_source_unit'), 'eV', 'HVP gas source unit')
    _one_call(trace, 'scalar_calls'); _one_call(trace, 'solvent_calls')
    derivatives = trace.get('gas_derivatives')
    if not isinstance(derivatives, list) or len(derivatives) != 2:
        raise ValueError('HVP observation needs both actual backwards, without another evaluation')
    shape = (len(item['atomic_numbers']), 3)
    values = []
    for entry in derivatives:
        require_equal(entry.get('dtype'), 'float32', 'native gas backward precision')
        value = array(entry.get('values'), shape, 'actual native gas backward')
        if not np.array_equal(value, value.astype(np.float32).astype(np.float64)):
            raise ValueError('Claimed native backward values are not float32-representable')
        values.append(value)
    solvent = _mapping(trace.get('solvent'), 'actual directional provider return')
    se = scalar(solvent.get('energy_hartree'), 'actual directional solvent energy')
    sf = array(solvent.get('forces_hartree_per_angstrom'), shape, 'actual directional solvent force')
    sv = array(solvent.get('hvp_hartree_per_angstrom2'), (np.prod(shape),), 'actual directional solvent HVP')
    gas_e = scalar(trace.get('gas_energy_source'), 'actual scalar gas energy') * EV2HARTREE
    _trace_point(trace, row, item)
    return {'same_call_energy_closure': abs(row['energy_hartree']-(gas_e+se)),
            'same_call_force_closure': _delta(row['forces_hartree_per_angstrom'], -values[0]*EV2HARTREE+sf),
            'same_call_hvp_closure': _delta(row['hvp_hartree_per_angstrom2'], values[1].reshape(-1)*EV2HARTREE+sv)}


def hessian_metrics(row, item, gas):
    n = len(item['atomic_numbers']); size = 3*n
    h = array(row.get('raw_hessian_hartree_per_angstrom2'), (size, size), 'composed raw Hessian')
    gh = array(row.get('gas_hessian_hartree_per_angstrom2'), h.shape, 'component gas Hessian')
    sh = array(row.get('solvent_hessian_hartree_per_angstrom2'), h.shape, 'component solvent Hessian')
    reference_h = array(gas['raw_hessian_hartree_per_angstrom2'], h.shape, 'independent gas H reference')
    _validate_hessian_observation(row, item, gh, sh)
    f = array(row.get('forces_hartree_per_angstrom'), (n, 3), 'composed force')
    sf = array(row.get('solvent_forces_hartree_per_angstrom'), f.shape, 'solvent force')
    gf = array(gas.get('forces_hartree_per_angstrom'), f.shape, 'center gas force')
    e = scalar(row.get('energy_hartree'), 'composed energy')
    se = scalar(row.get('solvent_energy_hartree'), 'solvent energy')
    provider = _mapping(row.get('provider_identity'), 'solvent provider identity')
    for key, expected in {'runtime_provider': 'torch-obc2', 'model': 'obc2',
                          'profile': 'obc2-mbondi2', 'nonpolar': row['nonpolar'],
                          'dtype': 'torch.float32', 'arithmetic_profile': 'obc2-fp32-v1'}.items():
        require_equal(provider.get(key), expected, f'solvent {key}')
    metrics = {'composition': _delta(h, gh+sh), **_same_call_center_metrics(row, item),
               'independent_center_energy_abs': abs(e-(gas['energy_hartree']+se)),
               'independent_center_force': _delta(f, gf+sf),
               'independent_gas_hessian': _delta(gh, reference_h),
               'raw_asymmetry': float(np.max(np.abs(h-h.T))),
               'translation': float(np.max(np.abs(h@np.tile(np.eye(3), (n, 1))))),
               'force_fd': []}
    ladder = row.get('force_fd_ladder')
    if not isinstance(ladder, list) or len(ladder) != 3:
        raise ValueError('Full fixed force FD ladder missing')
    matrices = []
    for entry, step in zip(ladder, STEPS):
        _mapping(entry, 'raw fixed FD ladder entry')
        require_equal(entry.get('step_angstrom'), step, 'predeclared FD step')
        probes = _index(entry.get('probes'), ('axis',), {(i,) for i in range(size)})
        fd = np.column_stack([_probe(probes[(axis,)], item, np.eye(size)[axis], step, row['device'])
                              for axis in range(size)])
        stored = array(entry.get('hessian_hartree_per_angstrom2'), h.shape, 'stored FD matrix')
        if not np.array_equal(stored, fd):
            raise ValueError('FD matrix does not replay from all signed force probes')
        matrices.append(fd)
        metrics['force_fd'].append(dict(step_angstrom=step, **_response_difference(h, fd)))
    metrics['finest_pair_difference'] = _delta(matrices[1], matrices[2])
    passed = (np.array_equal(h, gh+sh) and metrics['center_energy_closure'] <= 1e-12
        and metrics['center_force_closure']['max'] <= 1e-12
        and metrics['independent_center_energy_abs'] <= 2e-5
        and metrics['independent_center_force']['max'] <= 2e-6
        and metrics['independent_center_force']['rms'] <= 5e-7
        and np.allclose(gh, reference_h, atol=5e-6, rtol=5e-5)
        and metrics['raw_asymmetry'] <= 2e-6*max(1., float(np.max(np.abs(h))))
        and metrics['translation'] <= max(1e-5, 1e-5*float(np.linalg.norm(h)))
        and all(_response_bounded(m) for m in metrics['force_fd'][1:])
        and _bounded(metrics['finest_pair_difference'], 2.))
    return metrics, bool(passed)


def hvp_metrics(row, item, composed, vector):
    n = len(item['atomic_numbers']); size = 3*n
    v = array(row.get('direction'), (size,), 'direct HVP direction')
    require_equal(v.tolist(), vector['vector'], 'frozen HVP vector')
    h = array(composed.get('raw_hessian_hartree_per_angstrom2'), (size, size), 'RAW H for HVP')
    hvp = array(row.get('hvp_hartree_per_angstrom2'), (size,), 'direct HVP')
    dense = h@v  # NEVER symmetrize before this comparison.
    e = scalar(row.get('energy_hartree'), 'HVP center energy')
    f = array(row.get('forces_hartree_per_angstrom'), (n, 3), 'HVP center forces')
    center_f = array(composed.get('forces_hartree_per_angstrom'), f.shape, 'composed center force')
    require_equal(row.get('direct_hvp_dense_hessian_forbidden'), True, 'direct HVP dense trap')
    ladder = row.get('directional_fd_ladder')
    if not isinstance(ladder, list) or len(ladder) != 3:
        raise ValueError('Full fixed directional FD ladder missing')
    metrics = {'raw_Hv': _delta(hvp, dense), **_same_call_hvp_metrics(row, item),
               'center_energy_abs': abs(e-composed['energy_hartree']),
               'center_force': _delta(f, center_f), 'directional_fd': []}
    products = []
    for entry, step in zip(ladder, STEPS):
        _mapping(entry, 'raw fixed FD ladder entry')
        require_equal(entry.get('step_angstrom'), step, 'directional FD step')
        product = _probe(entry, item, v, step, row['device'])
        products.append(product)
        metrics['directional_fd'].append(dict(step_angstrom=step, **_response_difference(hvp, product)))
    metrics['finest_pair_difference'] = _delta(products[1], products[2])
    passed = (np.allclose(hvp, dense, atol=5e-6, rtol=5e-5)
        and metrics['same_call_energy_closure'] <= 1e-12
        and metrics['same_call_force_closure']['max'] <= 1e-12
        and metrics['same_call_hvp_closure']['max'] <= 1e-12
        and metrics['center_energy_abs'] <= 2e-5
        and metrics['center_force']['max'] <= 2e-6 and metrics['center_force']['rms'] <= 5e-7
        and all(_response_bounded(m, directional=True) for m in metrics['directional_fd'][1:])
        and _bounded(metrics['finest_pair_difference'], 2., rms_bound=DIRECTIONAL_RMS_BOUND))
    return metrics, bool(passed)


def validate_operation_provenance(row, section, artifact):
    identity = row['identity']
    precision = dict(identity, requested_dtype='float32', effective_dtype='float32',
        original_checkpoint_path=artifact['checkpoint_before']['path'],
        original_checkpoint_sha256=CHECKPOINT_SHA256, in_memory_cast_only=False,
        analytic_prepared=True, solvent_dtype=None if section == 'gas_records' else 'float32',
        last_point=row['point_identity'])
    require_equal(row.get('precision'), precision, 'actual operation precision/source fingerprint')
    if section == 'hvp_records':
        provenance = _mapping(row.get('hvp_provenance'), 'actual direct-HVP precision')
        for key, value in {'composition': 'aimnet2025-gas-analytic+torch-obc2-directional',
                'gas_ad_class': 'N32', 'gas_coordinate_dtype': 'float32', 'gas_gradient_dtype': 'float32',
                'gas_hvp_dtype': 'float32', 'output_dtype': 'float64', 'solvent_dtype': 'float32',
                'gas_point': row['point_identity']}.items():
            require_equal(provenance.get(key), value, f'HVP {key}')
        vector = np.asarray(row['direction'], dtype=np.float64)
        require_equal(provenance.get('direction_quantization_max'),
                      float(np.max(np.abs(vector-vector.astype(np.float32).astype(np.float64)))),
                      'native direction quantization')
        solvent = _mapping(provenance.get('solvent'), 'actual OBC32 directional provenance')
        require_equal(solvent.get('dtype'), 'torch.float32', 'HVP actual solvent kernel dtype')
        require_equal(solvent.get('arithmetic_profile'), 'obc2-fp32-v1', 'HVP actual solvent arithmetic profile')


def cutoff_guard_metrics(row, item, vectors, artifact):
    """Eight expected-rejection controls, never computed positive Hessians."""
    gap = cutoff_gap(item)
    if gap is None or gap > CUTOFF_GUARD:
        raise ValueError('Control original/native geometry is not within the fixed guard')
    gas = _mapping(row.get('gas_center'), 'finite native control center')
    if 'raw_hessian_hartree_per_angstrom2' in gas or 'raw_hessian_hartree_per_angstrom2' in row:
        raise ValueError('Expected-refusal controls cannot claim computed Hessians')
    validate_identity(gas.get('identity'), row['device'], artifact['runtime_before'])
    require_equal(gas['identity'], row['identity'], 'same complete control gas identity')
    validate_cutoff_point_identity(gas.get('point_identity'), item, row['device'])
    validate_operation_provenance(gas, 'gas_records', artifact)
    metrics, native_passed = native_center_metrics(gas, item)
    require_equal(row['point_identity'].get('neighbor_graph_sha256'),
                  gas['point_identity'].get('neighbor_graph_sha256'), 'same center native graph')
    n = len(item['atomic_numbers'])
    energy = scalar(row.get('energy_hartree'), 'composed control energy')
    forces = array(row.get('forces_hartree_per_angstrom'), (n, 3), 'finite composed control force')
    solvent_energy = scalar(row.get('solvent_energy_hartree'), 'control solvent energy')
    solvent_forces = array(row.get('solvent_forces_hartree_per_angstrom'), (n, 3), 'control solvent force')
    provider = _mapping(row.get('provider_identity'), 'control solvent identity')
    for field, expected in {'runtime_provider': 'torch-obc2', 'model': 'obc2',
            'profile': 'obc2-mbondi2', 'nonpolar': row['nonpolar'],
                          'dtype': 'torch.float32', 'arithmetic_profile': 'obc2-fp32-v1'}.items():
        require_equal(provider.get(field), expected, f'control {field}')
    metrics.update(cutoff_gap_angstrom=gap, **_same_call_center_metrics(row, item, cutoff=True),
        independent_center_energy_abs=abs(energy-(gas['energy_hartree']+solvent_energy)),
        independent_center_force=_delta(forces, np.asarray(gas['forces_hartree_per_angstrom'])+solvent_forces),
        expected_refusals=5, observed_expected_refusals=0)
    expected = {('GAS_H', -1), ('COMPOSED_H', -1)} | {('HVP', i) for i in range(3)}
    refusals = _index(row.get('refusals'), ('operation', 'direction_index'), expected)
    for (operation, index), observed in refusals.items():
        require_equal(observed.get('positions_angstrom'), item['positions_angstrom'], 'unmoved original control operation')
        validate_cutoff_point_identity(observed.get('point_identity'), item, row['device'])
        evidence = dict(row, point_identity=observed['point_identity'], precision=observed.get('precision'))
        validate_operation_provenance(evidence, 'gas_records' if operation == 'GAS_H' else 'cutoff_control_records', artifact)
        require_equal(observed.get('forbidden_entrypoints'), list(CUTOFF_FORBIDDEN), 'all FD/dense/AD fallback traps')
        calls = observed.get('fallback_calls')
        if not isinstance(calls, list):
            raise ValueError('Cutoff fallback-call trace missing')
        if operation == 'HVP':
            vector = vectors[(item['case'], item['geometry'], index)]['vector']
            require_equal(observed.get('direction'), vector, 'original source unit direction')
        error = _mapping(observed.get('error'), 'observed cutoff refusal error')
        if not isinstance(error.get('traceback'), str) or not error['traceback']:
            raise ValueError('Original cutoff refusal traceback missing')
        specific = (observed.get('status') == 'rejected' and calls == []
            and error.get('module') == CUTOFF_ERROR_MODULE and error.get('type') == CUTOFF_ERROR_CLASS
            and isinstance(error.get('message'), str) and error['message'].startswith(CUTOFF_ERROR_PREFIX))
        metrics['observed_expected_refusals'] += int(specific)
    passed = (native_passed and metrics['center_energy_closure'] <= 1e-12
              and metrics['center_force_closure']['max'] <= 1e-12
              and metrics['independent_center_energy_abs'] <= 2e-5
              and metrics['independent_center_force']['max'] <= 2e-6
              and metrics['independent_center_force']['rms'] <= 5e-7
              and metrics['observed_expected_refusals'] == 5)
    return metrics, bool(passed)


def _assess_rows(artifact, *, store=False):
    frozen = load_frozen_inputs()
    inputs = {(r['case'], r['geometry']): r for r in frozen['manifest']['records']}
    vectors = {(r['case'], r['geometry'], r['direction_index']): r for r in frozen['directions']['records']}
    controls = {(r['case'], r['geometry']): r for r in frozen['cutoff_controls']['records']}
    original_vectors = {(r['case'], r['geometry'], r['direction_index']): r
                        for r in frozen['original_directions']['records']}
    gas_keys = {k+(d,) for k in inputs for d in DEVICES}
    h_keys = {k+(p,) for k in gas_keys for p in NONPOLAR}
    sections = [('gas_records', ('case', 'geometry', 'device'), gas_keys),
                ('hessian_records', ('case', 'geometry', 'device', 'nonpolar'), h_keys),
                ('hvp_records', ('case', 'geometry', 'device', 'nonpolar', 'direction_index'),
                 {k+(i,) for k in h_keys for i in range(3)}),
                ('cutoff_control_records', ('case', 'geometry', 'device', 'nonpolar'),
                 {k+(d, p) for k in controls for d in DEVICES for p in NONPOLAR})]
    indices = {name: _index(artifact.get(name), fields, keys) for name, fields, keys in sections}
    counts = []
    for name, _, _ in sections:
        count = 0
        for key, row in indices[name].items():
            item = (controls if name == 'cutoff_control_records' else inputs)[key[:2]]
            require_equal(row.get('input'), item, 'complete frozen input/provenance')
            try:
                if row.get('status') == 'error':
                    metrics, passed = _error_metrics(row)
                elif row.get('status') == 'ok':
                    validate_identity(row.get('identity'), row['device'], artifact['runtime_before'])
                    require_equal(row['identity'], artifact['model_identities_before'][row['device']], 'same native cell identity')
                    point_validator = validate_cutoff_point_identity if name == 'cutoff_control_records' else validate_point_identity
                    point_validator(row.get('point_identity'), item, row['device'])
                    validate_operation_provenance(row, name, artifact)
                    if name == 'cutoff_control_records':
                        metrics, passed = cutoff_guard_metrics(row, item, original_vectors, artifact)
                    elif name == 'gas_records':
                        metrics, passed = gas_metrics(row, item)
                    elif name == 'hessian_records':
                        gas = indices['gas_records'][key[:3]]
                        if gas.get('status') != 'ok':
                            raise ValueError('Successful composed row missing successful gas reference')
                        metrics, passed = hessian_metrics(row, item, gas)
                    else:
                        composed = indices['hessian_records'][key[:4]]
                        if composed.get('status') != 'ok':
                            raise ValueError('Successful HVP row missing composed raw Hessian')
                        metrics, passed = hvp_metrics(row, item, composed, vectors[key[:2]+(key[4],)])
                else:
                    raise ValueError('Raw record status must be ok or error, never skip')
            except (ValueError, TypeError, KeyError) as exc:
                if not store:
                    raise ValueError('Malformed raw operation record') from exc
                row.update(status='error', error={'type': type(exc).__name__,
                    'message': str(exc) or repr(exc), 'traceback': traceback.format_exc()})
                metrics, passed = _error_metrics(row)
            if store:
                row.update(metrics=metrics, passed=bool(passed))
            else:
                require_equal(row.get('metrics'), metrics, 'raw numerical metric replay')
                require_equal(row.get('passed'), bool(passed), 'raw numerical verdict replay')
            count += int(passed)
        counts.append(count)
    return {gate: {'expected_count': expected, 'passed_count': count, 'passed': count == expected}
            for (gate, expected), count in zip(EXPECTED_COUNTS.items(), counts)}


def add_assessments(artifact):
    """Producer-only helper; compute every stored metric/verdict from raw arrays."""
    gates = _assess_rows(artifact, store=True)
    artifact.update(gates=gates, expected_count=sum(EXPECTED_COUNTS.values()),
                    passed_count=sum(g['passed_count'] for g in gates.values()),
                    passed=all(g['passed'] for g in gates.values()), expected_cutoff_refusals=40,
                    observed_cutoff_refusals=sum(row['metrics'].get('observed_expected_refusals', 0)
                        for row in artifact['cutoff_control_records']))
    return artifact


def assess_artifact(artifact, *, expected_source_snapshot=None):
    _mapping(artifact, 'candidate artifact')
    require_equal(core.artifact_content_sha256(artifact), artifact.get('content_sha256'), 'candidate self hash')
    for key, value in {'schema_version': SCHEMA_VERSION, 'artifact_type': ARTIFACT_TYPE, 'qualification_stage': STAGE,
            'profile': load_profile(), 'expected_native_identity': EXPECTED_NATIVE_IDENTITY, 'harness_sha256': core.sha256_file(ROOT / HARNESS_REL),
            'validator_sha256': core.sha256_file(ROOT / VALIDATOR_REL)}.items():
        require_equal(artifact.get(key), value, key)
    source = _mapping(artifact.get('source_snapshot_before'), 'clean committed source')
    require_equal(artifact.get('source_snapshot_after'), source, 'before/after full source inventory')
    require_equal(source.get('worktree_status'), [], 'clean committed source status')
    require_equal(artifact.get('candidate_git_sha'), source.get('git_head'), 'candidate Git SHA')
    _sha(source.get('files_aggregate_sha256'), 'source files aggregate')
    files = _mapping(source.get('files_sha256'), 'complete source inventory')
    for sha in files.values():
        _sha(sha, 'source file')
    require_equal(digest(files), source['files_aggregate_sha256'], 'source file aggregate replay')
    payload = {k: v for k, v in source.items() if k != 'aggregate_sha256'}
    require_equal(digest(payload), source.get('aggregate_sha256'), 'source snapshot self hash')
    qualification.validate_current_source_binding(source)
    if expected_source_snapshot is not None:
        require_equal(expected_source_snapshot.get('worktree_status'), [], 'expected source clean status')
        require_equal(expected_source_snapshot.get('files_sha256'), files, 'expected source inventory')
    require_equal(artifact.get('inputs_before'), load_frozen_inputs()['snapshot'], 'frozen full input snapshot')
    require_equal(artifact.get('inputs_after'), artifact['inputs_before'], 'before/after input snapshot')
    checkpoint = _mapping(artifact.get('checkpoint_before'), 'checkpoint snapshot')
    require_equal(checkpoint, checkpoint_snapshot(checkpoint.get('path')), 'live checkpoint fingerprint')
    require_equal(artifact.get('checkpoint_after'), checkpoint, 'before/after checkpoint snapshot')
    runtime = _mapping(artifact.get('runtime_before'), 'runtime snapshot')
    validate_runtime_snapshot(runtime)
    require_equal(artifact.get('runtime_after'), runtime, 'before/after full runtime/source fingerprint')
    identities = _mapping(artifact.get('model_identities_before'), 'actual per-device model identities')
    require_equal(sorted(identities), sorted(DEVICES), 'exact model identity devices')
    require_equal(artifact.get('model_identities_after'), identities, 'before/after native model fingerprint')
    for device, identity in identities.items():
        _mapping(identity, 'per-device model identity')
        if identity.get('status') == 'error':
            _error_metrics(identity)
        else:
            validate_identity(identity, device, runtime)
    gates = _assess_rows(artifact)
    require_equal(artifact.get('gates'), gates, 'recomputed gate counts')
    require_equal(artifact.get('expected_cutoff_refusals'), 40, 'separate expected cutoff refusals')
    require_equal(artifact.get('observed_cutoff_refusals'),
        sum(row['metrics'].get('observed_expected_refusals', 0) for row in artifact['cutoff_control_records']),
        'separate replayed cutoff refusal count')
    require_equal(artifact.get('expected_count'), sum(EXPECTED_COUNTS.values()), 'full expected denominator')
    require_equal(artifact.get('passed_count'), sum(g['passed_count'] for g in gates.values()), 'full passed count')
    require_equal(artifact.get('passed'), all(g['passed'] for g in gates.values()), 'overall replay verdict')
    return gates


def validate_artifact(artifact, *, expected_source_snapshot=None):
    """Sealer entry: validate raw proof, clean source binding, and all 216 positive + 8 control rows."""
    gates = assess_artifact(artifact, expected_source_snapshot=expected_source_snapshot)
    if artifact.get('postflight_errors') or artifact.get('validation_error'):
        raise ValueError('Candidate postflight/validation failure cannot be sealed')
    if not all(g['passed'] for g in gates.values()):
        raise ValueError('AIMNet2025 candidate matrix failed; no installed admission')
    return gates


def main():
    from types import SimpleNamespace
    from maple.function.calculator.aimnet._aimnet2025_calculator import _native_precision_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=CAMPAIGN / 'candidate-aimnet2025-v3.json')
    args = parser.parse_args()
    # Replaying an IEEE-profile artifact needs the same scoped settings as its
    # producer. Establish them explicitly and restore the operator's settings.
    with _native_precision_context(SimpleNamespace(), 'qualification-replay'):
        gates = validate_artifact(json.loads(args.output.read_text()))
    print(json.dumps({'path': str(args.output), 'gates': gates, 'stage': STAGE}, indent=2))


if __name__ == '__main__':
    main()
