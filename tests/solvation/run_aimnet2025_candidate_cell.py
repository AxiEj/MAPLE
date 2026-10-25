#!/usr/bin/env python3
"""Produce frozen AIMNet2025 N32 candidate evidence, NOT installed qualification.

Formal execution requires a clean committed source freeze. The single private
admission patch exists only inside factory construction and is always restored.
All force differences below are independent validation, never runtime fallback.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'docs/implicit-solvation/benchmarks', Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import validate_aimnet2025_analytic as validation  # noqa: E402
from aimnet2025_observation import (  # noqa: E402
    center as _center, center_with_components as _center_with_components,
    observe_hessian_components as _observe_hessian_components,
    observe_hvp_components as _observe_hvp_components,
)

OUTPUT = validation.CAMPAIGN / 'candidate-aimnet2025-v3.json'


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _error(exc):
    return {'module': type(exc).__module__, 'type': type(exc).__name__, 'message': str(exc) or repr(exc),
            'traceback': traceback.format_exc()}


def atoms_from_input(item):
    from maple.function.read.filereader.mol2_reader import MOL2Reader
    atoms = MOL2Reader(str(ROOT / item['source_mol2_relative_path']), charge=0, mult=1,
                       validate_charge=True)
    validation.require_equal(atoms.numbers.tolist(), item['atomic_numbers'], 'ordered frozen elements')
    validation.require_equal(atoms.get_initial_charges().tolist(), item['ordered_charges_e'], 'frozen q')
    # The immutable record explicitly names the displaced geometry; no MOL2
    # rewrite, geometry preparation, charge recomputation or renormalization.
    atoms.set_positions(item['positions_angstrom'])
    return atoms


def _probe(calculator, atoms, direction, step):
    x = atoms.positions.copy()
    v = np.asarray(direction).reshape(x.shape)
    result = {}
    try:
        for sign, multiplier in (('plus', 1.), ('minus', -1.)):
            actual = x + multiplier*step*v
            atoms.set_positions(actual)
            forces = array(atoms.get_forces())
            result.update({f'positions_{sign}_angstrom': atoms.positions.tolist(),
                           f'forces_{sign}_hartree_per_angstrom': forces.tolist(),
                           f'point_{sign}': deepcopy(calculator.last_analytic_point_identity)})
    finally:
        atoms.set_positions(x)
    return result


def _construct_composed(item, device, nonpolar, temporary, checkpoint, runtime, expected_identity):
    import maple.function.calculator.set_calculator as module
    from maple.function.calculator.aimnet._aimnet2025_calculator import AIMNet2025Calculator

    atoms = atoms_from_input(item)
    observed = {}

    def exact_test_gate(calculator, correction, *, model, device, d4):
        if type(calculator) is not AIMNet2025Calculator:
            raise RuntimeError('Candidate requires exact shipped AIMNet2025Calculator')
        if model != 'aimnet2025' or d4 is not False:
            raise RuntimeError('Candidate model/D4 identity mismatch')
        identity = calculator.analytic_cell_identity
        validation.validate_identity(identity, str(device).split(':', 1)[0], runtime)
        validation.require_equal(identity, expected_identity, 'factory native identity')
        provenance = correction.underlying_provider.provenance
        for key, value in {'runtime_provider': 'torch-obc2', 'model': 'obc2',
                'profile': 'obc2-mbondi2', 'nonpolar': nonpolar,
                'dtype': 'torch.float32', 'arithmetic_profile': 'obc2-fp32-v1'}.items():
            validation.require_equal(provenance.get(key), value, f'candidate OBC {key}')
        calculator.prepare_analytic_derivatives()
        observed.update(identity=deepcopy(identity), provider_identity=deepcopy(provenance),
                        precision=deepcopy(calculator.inference_precision_provenance))
        calculator.analytic_implicit_derivatives_admitted = True
        correction.analytic_task_derivatives_admitted = True

    with patch.object(module, '_admit_independent_obc2_analytic_cell', exact_test_gate):
        factory = module.SetCalculator(device, 'aimnet2025', str(temporary / nonpolar), atoms=atoms,
            implicit='gb', solvent='water', d4=False,
            model_options={'hessian': 'analytic', 'model_path': str(checkpoint), 'dtype': 'float32'},
            solvation_options={'implicit': 'water', 'method': 'gb', 'provider': 'torch',
                'model': 'obc2', 'profile': 'obc2-mbondi2', 'nonpolar': nonpolar, 'experimental': True},
            charge_options={'source': 'mol2', 'label': 'frozen-aimnet2025-candidate',
                            'mode': 'fixed', 'geometry': 'keep'})
        calculator = factory.set_calculator()
    if not observed:
        raise RuntimeError('Private candidate admission seam was not exercised')
    validation.require_equal(atoms.positions.tolist(), item['positions_angstrom'], 'factory coordinate preservation')
    validation.require_equal(atoms.get_initial_charges().tolist(), item['ordered_charges_e'], 'factory q preservation')
    return calculator, atoms, observed


def _construct_native_reference(calculator, checkpoint):
    from aimnet.calculators import AIMNet2Calculator
    from maple.function.calculator.aimnet._aimnet2025_calculator import _native_precision_context
    # Resolve CUDA's actual index once; do not compare an unspecified alias
    # against the adapter's exact cuda:0 cell fingerprint.
    with _native_precision_context(calculator, 'independent-native-construction'):
        return AIMNet2Calculator(str(checkpoint), device=str(calculator.device),
                                train=False, compile_model=False)


def _native_reference_identity(native, graph):
    from maple.function.calculator.aimnet._aimnet2025_identity import canonical_neighbor_graph_sha256
    from maple.function.calculator.aimnet._aimnet2025_calculator import _configuration
    from maple.function.calculator.analytic_protocol import named_array_sha256
    state = named_array_sha256({f'{label}.{name}': tensor.detach().cpu().numpy()
        for label, module in (('model', native.model), ('coulomb', native.external_coulomb),
                              ('d3', native.external_dftd3))
        for name, tensor in module.state_dict().items()})
    config = _configuration(native)
    return {'implementation': type(native).__module__+'.'+type(native).__name__,
        'operation': 'official forces=True,hessian=False; native input without requires_grad',
        'checkpoint_sha256': validation.CHECKPOINT_SHA256,
        'potential_configuration': {k: v for k, v in config.items() if k != 'engine'},
        'state_sha256': state,
        'observed_neighbor_graph_sha256': canonical_neighbor_graph_sha256(graph),
        'observed_neighbor_packing_sha256': named_array_sha256(graph),
        'neighbor_graph_scheme': validation.GRAPH_SCHEME}


def _gas_center(row, calculator, item):
    from maple.function.calculator.calculator_base import EV2HARTREE
    from maple.function.calculator.aimnet._aimnet2025_calculator import _native_precision_context
    atoms = atoms_from_input(item)
    energy, forces = _center(calculator, atoms)
    # The unmodified independent oracle starts cold for each molecular input.
    # Reusing its adaptive allocator after a smaller molecule can silently
    # truncate lists in the pinned upstream runtime (kept as a negative probe).
    native = _construct_native_reference(
        calculator, calculator.inference_precision_provenance['original_checkpoint_path'])
    # Separate official calculator, native forces (including official native D3
    # first-backward path). This is not the adapter's force evaluated twice.
    graph = {}
    def observe_native_input(module, args):
        for key, value in args[0].items():
            if key.startswith(('nbmat', 'mol_idx', 'numbers', 'shifts', 'mask_', 'nb_pad_mask')) and hasattr(value, 'detach'):
                graph[key] = value.detach().cpu().numpy().copy()
    hook = native.model.register_forward_pre_hook(observe_native_input)
    try:
        # A's shared IEEE audit scope, but this is still the separately loaded
        # official backend and its own captured forward inputs/outputs.
        with _native_precision_context(calculator, 'independent-native-forward'):
            reference = native({'coord': np.asarray(item['positions_angstrom'], dtype=np.float32),
                                'numbers': atoms.numbers, 'charge': 0., 'mult': 1}, forces=True, hessian=False)
    finally:
        hook.remove()
    row.update(energy_hartree=energy, forces_hartree_per_angstrom=forces.tolist(),
        native_energy_hartree=float(array(reference['energy']).sum()*EV2HARTREE),
        native_forces_hartree_per_angstrom=(array(reference['forces']).reshape(forces.shape)*EV2HARTREE).tolist(),
        identity=calculator.analytic_cell_identity,
        native_reference_identity=_native_reference_identity(native, graph))
    row['point_identity'] = deepcopy(calculator.last_analytic_point_identity)
    row['precision'] = deepcopy(calculator.inference_precision_provenance)


def _gas_row(row, calculator, item):
    _gas_center(row, calculator, item)
    atoms = atoms_from_input(item)
    row['raw_hessian_hartree_per_angstrom2'] = array(calculator.get_hessian(atoms)).tolist()
    row['point_identity'] = deepcopy(calculator.last_analytic_point_identity)
    row['precision'] = deepcopy(calculator.inference_precision_provenance)


def _hessian_row(row, calculator, atoms, observed, gas):
    row.update(identity=observed['identity'], provider_identity=observed['provider_identity'])
    energy, forces, center = _center_with_components(calculator, atoms)
    row.update(energy_hartree=energy, forces_hartree_per_angstrom=forces.tolist(), same_call_center=center)
    with _observe_hessian_components(calculator) as (htrace, operands):
        hessian = array(calculator.get_hessian(atoms))
    row.update(raw_hessian_hartree_per_angstrom2=hessian.tolist(),
               point_identity=deepcopy(calculator.last_analytic_point_identity),
               precision=deepcopy(calculator.inference_precision_provenance))
    row.update(solvent_energy_hartree=center['solvent_energy_hartree'],
        solvent_forces_hartree_per_angstrom=center['solvent_forces_hartree_per_angstrom'],
        solvent_hessian_hartree_per_angstrom2=operands['solvent'].tolist(),
        gas_hessian_hartree_per_angstrom2=operands['gas'].tolist(), same_call_hessian=htrace)
    ladder = row['force_fd_ladder'] = []
    for step in validation.STEPS:
        entry = {'step_angstrom': step, 'probes': []}
        ladder.append(entry)
        columns = []
        for axis in range(atoms.positions.size):
            probe = dict(axis=axis, **_probe(calculator, atoms, np.eye(atoms.positions.size)[axis], step))
            entry['probes'].append(probe)
            plus = np.asarray(probe['forces_plus_hartree_per_angstrom'])
            minus = np.asarray(probe['forces_minus_hartree_per_angstrom'])
            columns.append(-(plus-minus).reshape(-1)/(2.*step))
        entry['hessian_hartree_per_angstrom2'] = np.column_stack(columns).tolist()


def _hvp_row(row, calculator, atoms, vector):
    def forbidden_dense(*_args, **_kwargs):
        raise AssertionError('direct HVP materialized a dense Hessian')

    correction = calculator.solvent_correction
    provider = correction.underlying_provider
    row.update(identity=calculator.analytic_cell_identity, direction=vector['vector'])
    # Negative controls cover adapter/shared/official dense-H entrypoints. Force
    # finite differences run afterwards and never supply the actual HVP.
    with ExitStack() as stack:
        for owner, name in ((calculator, '_analytic_hessian'), (calculator, 'get_hessian'),
                            (correction, 'get_hessian'), (provider, 'hessian'),
                            (provider.kernel, 'hessian')):
            stack.enter_context(patch.object(owner, name, side_effect=forbidden_dense))
        if hasattr(calculator.backend, 'calculate_hessian'):
            stack.enter_context(patch.object(calculator.backend, 'calculate_hessian', side_effect=forbidden_dense))
        with _observe_hvp_components(calculator) as same_call:
            hvp, forces, energy = calculator.get_hvp(atoms, np.asarray(vector['vector']))
    row.update(hvp_hartree_per_angstrom2=array(hvp).reshape(-1).tolist(),
               forces_hartree_per_angstrom=array(forces).reshape(len(atoms), 3).tolist(),
               energy_hartree=float(array(energy)),
               direct_hvp_dense_hessian_forbidden=True,
               point_identity=deepcopy(calculator.last_analytic_point_identity),
               hvp_provenance=deepcopy(calculator.last_hvp_provenance),
               precision=deepcopy(calculator.inference_precision_provenance))
    row['same_call_hvp'] = same_call
    row['directional_fd_ladder'] = [dict(step_angstrom=step,
        **_probe(calculator, atoms, vector['vector'], step)) for step in validation.STEPS]


def _capture_cutoff_refusal(calculator, atoms, correction, operation, index, operation_call, direction=None):
    """Observe the real guard and forbid all derivative/FD/dense fallbacks."""
    import torch
    from maple.function.calculator import calculator_base
    calls = []
    def forbidden(label):
        def trap(*_args, **_kwargs):
            calls.append(label)
            raise AssertionError('cutoff control reached forbidden fallback: '+label)
        return trap
    targets = ((torch.autograd, 'grad'), (calculator_base, 'numerical_hessian_from_atoms'),
        (torch.autograd.functional, 'hessian'), (torch.func, 'hessian'),
        (correction, 'get_hessian'), (correction.underlying_provider, 'hessian'),
        (correction.underlying_provider.kernel, 'hessian'))
    observed = {'operation': operation, 'direction_index': index,
        'forbidden_entrypoints': list(validation.CUTOFF_FORBIDDEN), 'fallback_calls': calls}
    if direction is not None:
        observed['direction'] = list(direction)
    with ExitStack() as stack:
        for label, (owner, name) in zip(validation.CUTOFF_FORBIDDEN, targets):
            stack.enter_context(patch.object(owner, name, side_effect=forbidden(label)))
        try:
            operation_call()
            observed.update(status='returned', error={'module': 'builtins', 'type': 'NoException',
                'message': 'operation returned instead of refusing cutoff curvature', 'traceback': 'no exception'})
        except Exception as exc:
            observed.update(status='rejected', error=_error(exc))
    observed.update(positions_angstrom=atoms.positions.tolist(),
        point_identity=deepcopy(calculator.last_analytic_point_identity),
        precision=deepcopy(calculator.inference_precision_provenance))
    return observed


def _cutoff_control_row(row, item, device, nonpolar, temporary, checkpoint, runtime,
                        expected_identity, gas_calculator, vectors):
    gas = row['gas_center'] = {}
    _gas_center(gas, gas_calculator, item)  # Finite E/F only; no H/FD.
    calculator, atoms, observed = _construct_composed(item, device, nonpolar, temporary,
        checkpoint, runtime, expected_identity)
    energy, forces, center = _center_with_components(calculator, atoms)
    correction = calculator.solvent_correction
    row.update(identity=observed['identity'], provider_identity=observed['provider_identity'],
        energy_hartree=energy, forces_hartree_per_angstrom=forces.tolist(),
        solvent_energy_hartree=center['solvent_energy_hartree'],
        solvent_forces_hartree_per_angstrom=center['solvent_forces_hartree_per_angstrom'],
        same_call_center=center,
        point_identity=deepcopy(calculator.last_analytic_point_identity),
        precision=deepcopy(calculator.inference_precision_provenance), refusals=[])
    gas_atoms = atoms_from_input(item)
    row['refusals'].append(_capture_cutoff_refusal(gas_calculator, gas_atoms, correction,
        'GAS_H', -1, lambda: gas_calculator.get_hessian(gas_atoms)))
    row['refusals'].append(_capture_cutoff_refusal(calculator, atoms, correction,
        'COMPOSED_H', -1, lambda: calculator.get_hessian(atoms)))
    for index in range(3):
        vector = vectors[(item['case'], item['geometry'], index)]['vector']
        row['refusals'].append(_capture_cutoff_refusal(calculator, atoms, correction,
            'HVP', index, lambda: calculator.get_hvp(atoms, np.asarray(vector)), direction=vector))


def _preserve_nonfinite(value):
    if isinstance(value, dict):
        return {k: _preserve_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_preserve_nonfinite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return 'NaN' if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def _write_new(path, artifact):
    if artifact.get('passed') is False:
        safe = _preserve_nonfinite(artifact)
        artifact.clear()
        artifact.update(safe)
    validation.core.seal_artifact(artifact)
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL also handles races after the initial refusal, with no replacement.
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload)


def run(output: Path):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'candidate evidence is immutable: {output}')
    # This is deliberately before model import/initialization/inference.
    before = validation.qualification.source_snapshot(require_clean=True)
    from types import SimpleNamespace
    from maple.function.calculator.aimnet._aimnet2025_calculator import _native_precision_context
    # The same scoped IEEE policy as model operations, including exact caller
    # selector restoration. This audit host grants no calculator admission.
    with _native_precision_context(SimpleNamespace(), 'candidate-qualification'):
        return _run_in_profile(output, before)


def _run_in_profile(output, before):
    frozen = validation.load_frozen_inputs()
    checkpoint = validation.checkpoint_snapshot()
    profile = validation.load_profile()
    import torch
    from maple.function.calculator.aimnet._aimnet2025_calculator import AIMNet2025Calculator
    torch.set_num_threads(1)
    runtime = validation.runtime_snapshot()
    validation.validate_runtime_snapshot(runtime)
    calculators, identities = {}, {}
    for device in validation.DEVICES:
        try:
            calculator = AIMNet2025Calculator(device, model_path=checkpoint['path'], dtype='float32')
            calculator.prepare_analytic_derivatives()
            calculators[device] = calculator
            identities[device] = calculator.analytic_cell_identity
        except Exception as exc:
            identities[device] = {'status': 'error', 'error': _error(exc)}
    artifact = {'schema_version': validation.SCHEMA_VERSION, 'artifact_type': validation.ARTIFACT_TYPE,
        'qualification_stage': validation.STAGE, 'candidate_git_sha': before['git_head'],
        'harness_sha256': validation.core.sha256_file(Path(__file__).resolve()),
        'validator_sha256': validation.core.sha256_file(ROOT / validation.VALIDATOR_REL),
        'profile': profile, 'expected_native_identity': validation.EXPECTED_NATIVE_IDENTITY,
        'source_snapshot_before': before, 'inputs_before': frozen['snapshot'],
        'checkpoint_before': checkpoint, 'runtime_before': runtime, 'model_identities_before': identities,
        'gas_records': [], 'hessian_records': [], 'hvp_records': [], 'cutoff_control_records': []}
    vectors = {(r['case'], r['geometry'], r['direction_index']): r for r in frozen['directions']['records']}
    original_vectors = {(r['case'], r['geometry'], r['direction_index']): r
                        for r in frozen['original_directions']['records']}
    with tempfile.TemporaryDirectory(prefix='maple-aimnet2025-candidate-') as temporary:
        for item in frozen['manifest']['records']:
            for device in validation.DEVICES:
                common = {k: item[k] for k in ('case', 'geometry')}
                common.update(device=device, input=deepcopy(item), status='ok')
                gas = dict(common)
                artifact['gas_records'].append(gas)
                try:
                    if identities[device].get('status') == 'error':
                        raise RuntimeError('Native initialization failed: '+identities[device]['error']['message'])
                    _gas_row(gas, calculators[device], item)
                except Exception as exc:
                    gas.update(status='error', error=_error(exc))
                for nonpolar in validation.NONPOLAR:
                    row = dict(common, nonpolar=nonpolar)
                    artifact['hessian_records'].append(row)
                    calculator = atoms = None
                    try:
                        if gas['status'] != 'ok':
                            raise RuntimeError('Gas reference row failed; no valid composed closure')
                        calculator, atoms, observed = _construct_composed(item, device, nonpolar,
                            Path(temporary), checkpoint['path'], runtime, identities[device])
                        _hessian_row(row, calculator, atoms, observed, gas)
                    except Exception as exc:
                        row.update(status='error', error=_error(exc))
                    for index in range(3):
                        hvp = dict(common, nonpolar=nonpolar, direction_index=index)
                        artifact['hvp_records'].append(hvp)
                        try:
                            if row['status'] != 'ok':
                                raise RuntimeError('Composed Hessian row failed; raw H@v unavailable')
                            _hvp_row(hvp, calculator, atoms, vectors[(item['case'], item['geometry'], index)])
                        except Exception as exc:
                            hvp.update(status='error', error=_error(exc))
        for item in frozen['cutoff_controls']['records']:
            for device in validation.DEVICES:
                for nonpolar in validation.NONPOLAR:
                    row = {k: item[k] for k in ('case', 'geometry')}
                    row.update(device=device, nonpolar=nonpolar, input=deepcopy(item), status='ok')
                    artifact['cutoff_control_records'].append(row)
                    try:
                        if identities[device].get('status') == 'error':
                            raise RuntimeError('Native initialization failed: '+identities[device]['error']['message'])
                        _cutoff_control_row(row, item, device, nonpolar, Path(temporary), checkpoint['path'],
                            runtime, identities[device], calculators[device], original_vectors)
                    except Exception as exc:
                        row.update(status='error', error=_error(exc))
    # Re-read every snapshot independently, even when another snapshot fails.
    # prepare_analytic_derivatives rehashes the actual in-memory model state.
    for device, calculator in calculators.items():
        try:
            calculator.prepare_analytic_derivatives()
        except Exception as exc:
            artifact.setdefault('postflight_errors', {})[device] = _error(exc)
    readers = {
        'source_snapshot_after': lambda: validation.qualification.source_snapshot(require_clean=True),
        'inputs_after': lambda: validation.load_frozen_inputs()['snapshot'],
        'checkpoint_after': validation.checkpoint_snapshot,
        'runtime_after': validation.runtime_snapshot,
        'model_identities_after': lambda: {d: calculators[d].analytic_cell_identity if d in calculators
                                          else identities[d] for d in validation.DEVICES}}
    for name, reader in readers.items():
        try:
            artifact[name] = reader()
        except Exception as exc:
            artifact[name] = {'status': 'error', 'error': _error(exc)}
    try:
        validation.add_assessments(artifact)
        validation.core.seal_artifact(artifact)
        validation.assess_artifact(artifact)
        if artifact.get('postflight_errors'):
            raise RuntimeError('Actual model/source postflight rehash failed')
    except Exception as exc:
        artifact.update(passed=False, validation_error=_error(exc))
    destination = output if artifact['passed'] else output.with_name(output.stem+'.FAILED-'+uuid4().hex+'.json')
    artifact['retained_output_path'] = str(destination)
    validation.core.seal_artifact(artifact)
    _write_new(destination, artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.resolve().relative_to((ROOT / '.omx').resolve())
    artifact = run(args.output)
    print(json.dumps({'path': artifact['retained_output_path'], 'passed': artifact['passed'],
                      'gates': artifact.get('gates'), 'content_sha256': artifact['content_sha256']}, indent=2))
    if not artifact['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
