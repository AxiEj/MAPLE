"""WS2 — routing the per-step logged reads through ``evaluate_md_properties``
adds no backend forward pass, and the logged values equal the evaluator's.

The per-step backend-call counts are the operational form of the
"no-double-compute" gate.  They are exact and were cross-checked against the
pre-WS2 behaviour:

* NVE / NVT(v-rescale):  K + 1   (one t=0 force cache + one force eval/step;
  the logged PE read hits the integrator's cache)
* NPT(v-rescale + c-rescale):  3K + 1   (per step: pre-barostat pressure,
  post-barostat force refresh, and the final Velocity-Verlet force/stress
  evaluation; logged PE is folded into cached evaluator reads)
"""

import numpy as np
from ase.build import bulk
from ase.calculators.calculator import all_changes

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.evaluator import evaluate_md_properties
from maple.function.dispatcher.md.validation import MapleLJReferenceCalculator


class _CountingLJ(MapleLJReferenceCalculator):
    """LJ reference that counts backend forward passes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_calculate = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.n_calculate += 1
        super().calculate(atoms, properties, system_changes)


def _crystal_with_counter():
    # FCC Ar supercell with rc < the minimum-image radius (clean LJ reference).
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)
    atoms.calc = _CountingLJ(rc=4.0)
    return atoms


def _spy_last_log_step(sim):
    """Capture the kwargs of the last ``log_step`` call."""
    record = {}
    original = sim.logger.log_step

    def spy(**kwargs):
        record.update(kwargs)
        return original(**kwargs)

    sim.logger.log_step = spy
    return record


def test_nve_routing_adds_no_backend_call(tmp_path):
    atoms = _crystal_with_counter()
    steps = 5
    NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={
        "steps": steps, "timestep": 0.5, "temperature": 80.0, "random_seed": 1,
        "remove_com_every": 0, "verbose": 0, "log_every": 1, "traj_every": steps,
        "rst_every": 0,
    }).run()
    assert atoms.calc.n_calculate == steps + 1


def test_nvt_routing_adds_no_backend_call(tmp_path):
    atoms = _crystal_with_counter()
    steps = 5
    NVT(output=str(tmp_path / "nvt.out"), atoms=atoms, paras={
        "steps": steps, "timestep": 0.5, "temperature": 80.0, "thermostat": "v-rescale",
        "tau_t": 50.0, "random_seed": 2, "remove_com_every": 0, "verbose": 0,
        "log_every": 1, "traj_every": steps, "rst_every": 0,
    }).run()
    assert atoms.calc.n_calculate == steps + 1


def test_npt_routing_adds_no_backend_call(tmp_path):
    atoms = _crystal_with_counter()
    steps = 5
    NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={
        "steps": steps, "timestep": 0.5, "temperature": 80.0, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 50.0,
        "tau_p": 1000.0, "random_seed": 3, "remove_com_every": 0, "verbose": 0,
        "log_every": 1, "traj_every": steps, "rst_every": 0,
    }).run()
    assert atoms.calc.n_calculate == 3 * steps + 1


def test_nve_logged_energy_equals_evaluator_field(tmp_path):
    atoms = _crystal_with_counter()
    sim = NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={
        "steps": 3, "timestep": 0.5, "temperature": 80.0, "random_seed": 1,
        "remove_com_every": 0, "verbose": 0, "log_every": 1, "traj_every": 3,
        "rst_every": 0,
    })
    record = _spy_last_log_step(sim)
    sim.run()
    # atoms rest at the last logged geometry, so re-reading the single entry
    # point reproduces the logged PE (cache hit at the same configuration).
    assert record["potential_energy"] == evaluate_md_properties(atoms).energy_ha


def test_npt_logged_pressure_and_energy_equal_evaluator_fields(tmp_path):
    atoms = _crystal_with_counter()
    sim = NPT(output=str(tmp_path / "npt.out"), atoms=atoms, paras={
        "steps": 3, "timestep": 0.5, "temperature": 80.0, "pressure": 1.0,
        "thermostat": "v-rescale", "barostat": "c-rescale", "tau_t": 50.0,
        "tau_p": 1000.0, "random_seed": 3, "remove_com_every": 0, "verbose": 0,
        "log_every": 1, "traj_every": 3, "rst_every": 0,
    })
    record = _spy_last_log_step(sim)
    sim.run()
    props = evaluate_md_properties(atoms, need_stress=True, velocities_au=record["velocities"])
    assert record["potential_energy"] == props.energy_ha
    assert record["pressure"] == props.pressure_bar
