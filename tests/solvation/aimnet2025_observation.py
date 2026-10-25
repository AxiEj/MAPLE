"""Value-preserving observations of real, single-call composition operands.

This is candidate-test instrumentation, never an installed admission surface.
No component is reconstructed by subtracting the total or by another forward.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from unittest.mock import patch

import numpy as np


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def component_fingerprint(value):
    from maple.function.calculator.analytic_protocol import named_array_sha256
    return named_array_sha256({'component': array(value)})


def center(calculator, atoms):
    atoms.calc = calculator
    calculator.calculate(atoms, properties=['energy', 'forces'])
    return float(calculator.results['energy']), array(calculator.results['forces'])


def center_with_components(calculator, atoms):
    correction = calculator.solvent_correction
    original_finalize, original_evaluate = calculator._finalize_results, correction.evaluate
    trace = {'schema': 'same-call-ef-operands-v1', 'finalize_calls': 0, 'solvent_calls': 0}

    def finalize(*args, **kwargs):
        trace['finalize_calls'] += 1
        forces = np.asarray(kwargs['forces'])
        trace.update(gas_source_unit=kwargs.get('unit') or calculator.MODEL_ENERGY_UNIT,
                     gas_energy_source=float(kwargs['energy']), gas_forces_source=forces.tolist(),
                     gas_forces_source_dtype=str(forces.dtype),
                     point_identity=deepcopy(calculator.last_analytic_point_identity))
        return original_finalize(*args, **kwargs)

    def evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        trace['solvent_calls'] += 1
        trace.update(solvent_energy_hartree=float(result.energy_hartree),
                     solvent_forces_hartree_per_angstrom=array(result.forces_hartree_per_angstrom).tolist())
        return result

    with patch.object(calculator, '_finalize_results', side_effect=finalize), \
            patch.object(correction, 'evaluate', side_effect=evaluate):
        energy, forces = center(calculator, atoms)
    if trace['finalize_calls'] != 1 or trace['solvent_calls'] != 1:
        raise RuntimeError('Expected one actual finalizer and one solvent evaluation')
    return energy, forces, trace


@contextmanager
def observe_hessian_components(calculator):
    correction = calculator.solvent_correction
    original_gas, original_solvent = calculator._analytic_hessian, correction.get_hessian
    trace = {'schema': 'same-call-hessian-operands-v1', 'gas_calls': 0, 'solvent_calls': 0}
    operands = {}

    def gas(*args, **kwargs):
        result = original_gas(*args, **kwargs)
        trace['gas_calls'] += 1
        operands['gas'] = array(result).copy()
        trace.update(gas_hessian_sha256=component_fingerprint(result),
                     point_identity=deepcopy(calculator.last_analytic_point_identity))
        return result

    def solvent(*args, **kwargs):
        result = original_solvent(*args, **kwargs)
        trace['solvent_calls'] += 1
        operands['solvent'] = array(result).copy()
        trace['solvent_hessian_sha256'] = component_fingerprint(result)
        return result

    with patch.object(calculator, '_analytic_hessian', side_effect=gas), \
            patch.object(correction, 'get_hessian', side_effect=solvent):
        yield trace, operands
    if trace['gas_calls'] != 1 or trace['solvent_calls'] != 1:
        raise RuntimeError('Expected one actual gas and one solvent Hessian return')


@contextmanager
def observe_hvp_components(calculator):
    correction = calculator.solvent_correction
    original_scalar, original_directional = calculator._scalar, correction.get_directional_derivatives
    trace = {'schema': 'same-call-hvp-operands-v1', 'scalar_calls': 0,
             'solvent_calls': 0, 'gas_derivatives': []}
    with ExitStack() as stack:
        def scalar(*args, **kwargs):
            result = original_scalar(*args, **kwargs)
            energy, coordinates = result
            trace['scalar_calls'] += 1
            trace.update(gas_energy_source=float(array(energy)), gas_source_unit='eV',
                         point_identity=deepcopy(calculator.last_analytic_point_identity))
            def gradient(value):
                trace['gas_derivatives'].append({'dtype': str(value.dtype).removeprefix('torch.'),
                                                'values': array(value).tolist()})
                return None
            stack.callback(coordinates.register_hook(gradient).remove)
            return result

        def directional(*args, **kwargs):
            result = original_directional(*args, **kwargs)
            trace['solvent_calls'] += 1
            trace['solvent'] = {'energy_hartree': float(result.energy_hartree),
                'forces_hartree_per_angstrom': array(result.forces_hartree_per_angstrom).tolist(),
                'hvp_hartree_per_angstrom2': array(result.hvp_hartree_per_angstrom2).reshape(-1).tolist()}
            return result

        stack.enter_context(patch.object(calculator, '_scalar', side_effect=scalar))
        stack.enter_context(patch.object(correction, 'get_directional_derivatives', side_effect=directional))
        yield trace
    if trace['scalar_calls'] != 1 or trace['solvent_calls'] != 1 or len(trace['gas_derivatives']) != 2:
        raise RuntimeError('Expected one scalar, two actual gas backwards and one solvent return')
