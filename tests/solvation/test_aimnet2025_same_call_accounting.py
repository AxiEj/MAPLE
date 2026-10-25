"""Composition accounting observes one execution, not GPU repeat equality."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = spec_from_file_location("aimnet2025_accounting_producer", ROOT / "tests/solvation/run_aimnet2025_candidate_cell.py")
producer = module_from_spec(spec)
spec.loader.exec_module(producer)
from maple.function.calculator.calculator_base import EV2HARTREE


class Correction:
    def __init__(self):
        self.calls = 0

    def evaluate(self, atoms, need_forces=False, **kwargs):
        self.calls += 1
        return SimpleNamespace(energy_hartree=.125, forces_hartree_per_angstrom=np.full((1, 3), .25))


class JitterCalculator:
    MODEL_ENERGY_UNIT = "eV"

    def __init__(self):
        self.calls = 0
        self.solvent_correction = Correction()
        self.last_analytic_point_identity = {"actual_call": 0}

    def calculate(self, atoms, properties):
        self.calls += 1
        self.last_analytic_point_identity = {"actual_call": self.calls}
        self._finalize_results(atoms, energy=3. + self.calls * 1e-7,
                               forces=np.full((1, 3), 2. + self.calls * 1e-7))

    def _finalize_results(self, atoms, *, energy, forces):
        solvent = self.solvent_correction.evaluate(atoms, need_forces=True)
        factor = EV2HARTREE
        self.results = {"energy": energy * factor + solvent.energy_hartree,
                        "forces": forces * factor + solvent.forces_hartree_per_angstrom}


def test_center_has_one_real_forward_and_records_its_operands():
    calc = JitterCalculator()
    energy, force, trace = producer._center_with_components(calc, SimpleNamespace())
    assert calc.calls == calc.solvent_correction.calls == 1
    assert trace["finalize_calls"] == trace["solvent_calls"] == 1
    assert trace["point_identity"] == {"actual_call": 1}
    assert energy == trace["gas_energy_source"] * EV2HARTREE + .125
    np.testing.assert_array_equal(force, np.asarray(trace["gas_forces_source"]) * EV2HARTREE + .25)
    second_energy, second_force, _ = producer._center_with_components(calc, SimpleNamespace())
    assert second_energy != energy and not np.array_equal(second_force, force)
    assert calc.calls == 2  # Independent-repeat jitter was not suppressed.


def test_center_observer_restores_methods_when_backend_raises():
    calc = JitterCalculator()
    original = calc._finalize_results
    def fail(*args, **kwargs):
        raise RuntimeError("backend failure")
    calc.calculate = fail
    with pytest.raises(RuntimeError, match="backend failure"):
        producer._center_with_components(calc, SimpleNamespace())
    assert calc._finalize_results == original


def test_hvp_observer_passes_through_values_and_adds_no_backward():
    import torch
    class Model:
        def __init__(self):
            self.solvent_correction = SimpleNamespace(get_directional_derivatives=self.directional)
            self.last_analytic_point_identity = {"actual_call": 1}
            self.scalar_calls = self.backward_calls = 0
        def _scalar(self, atoms, **kwargs):
            self.scalar_calls += 1
            q = torch.tensor([[1., 2., 3.]], requires_grad=True)
            return (q*q).sum().double(), q
        def directional(self, *args):
            return SimpleNamespace(energy_hartree=.1, forces_hartree_per_angstrom=np.ones((1, 3)),
                                   hvp_hartree_per_angstrom2=np.ones(3), provenance={})
        def run(self):
            e, q = self._scalar(None)
            self.backward_calls += 1
            g = torch.autograd.grad(e, q, create_graph=True)[0]
            self.backward_calls += 1
            hv = torch.autograd.grad(g.sum(), q)[0]
            self.solvent_correction.get_directional_derivatives(None, np.ones(3))
            return e, g, hv
    model = Model()
    original = model._scalar
    with producer._observe_hvp_components(model) as trace:
        e, g, hv = model.run()
    assert model.scalar_calls == 1 and model.backward_calls == 2
    assert trace["scalar_calls"] == trace["solvent_calls"] == 1
    assert len(trace["gas_derivatives"]) == 2
    np.testing.assert_array_equal(trace["gas_derivatives"][0]["values"], g.detach().numpy())
    np.testing.assert_array_equal(trace["gas_derivatives"][1]["values"], hv.detach().numpy())
    assert model._scalar == original
