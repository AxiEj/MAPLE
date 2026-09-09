"""Effective MD dimensions are a saved contract, not a checkpoint-shape guess."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.rst_io import read_rst, write_rst
from maple.function.dispatcher.md.thermostat.vrescale import VRescaleThermostat
from maple.function.utility.rigid_body import rotational_dof


class FreeParticles(Calculator):
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces"]
        self.maple_pes_identity = {
            "backend": "free-particles-test",
            "model_fingerprint": {"algorithm": "sha256", "digest": "e" * 64, "source": "test"},
            "relevant_settings": {},
        }

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def bending_atoms():
    atoms = Atoms("HOH", positions=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
                  masses=[1, 16, 1], calculator=FreeParticles())
    atoms.info.update(charge=0, mult=1)
    atoms.arrays["velocities"] = np.array([[0, 1e-4, 0], [0, -1e-4 / 8, 0], [0, 1e-4, 0]])
    return atoms


def parameters(steps, **updates):
    result = {
        "steps": steps, "timestep": 0.1, "thermostat": "v-rescale", "temperature": 300,
        "tau_t": 100, "random_seed": 123, "remove_com": True, "remove_angular": True,
        "remove_com_every": 1, "remove_angular_every": 1, "rst_every": 2,
        "log_every": 100, "traj_every": 100, "verbose": 0, "init_velocities": False,
    }
    result.update(updates)
    return result


def test_linear_start_bends_without_changing_restart_thermostat(tmp_path):
    atoms = bending_atoms()
    masses = atoms.get_masses()[:, None]
    np.testing.assert_allclose((masses * atoms.arrays["velocities"]).sum(axis=0), 0)
    np.testing.assert_allclose(np.cross(atoms.positions, masses * atoms.arrays["velocities"]).sum(axis=0), 0)
    assert rotational_dof(atoms) == 2
    continuous = NVT(str(tmp_path / "continuous.out"), atoms, parameters(8))
    continuous.run()
    split = NVT(str(tmp_path / "split.out"), bending_atoms(), parameters(4))
    split.run()
    assert rotational_dof(split.atoms) == 3
    resumed = NVT(str(tmp_path / "split.out"), bending_atoms(), parameters(8, restart=True))
    resumed.run()
    expected = read_rst(tmp_path / "continuous_md.rst")
    actual = read_rst(tmp_path / "split_md.rst")
    assert isinstance(resumed.thermostat, VRescaleThermostat)
    assert isinstance(continuous.thermostat, VRescaleThermostat)
    assert resumed.thermostat._n_dof == continuous.thermostat._n_dof
    assert resumed.thermostat._ke_target == continuous.thermostat._ke_target
    np.testing.assert_array_equal(actual["positions"], expected["positions"])
    np.testing.assert_array_equal(actual["velocities"], expected["velocities"])
    assert actual["rng_state"] == expected["rng_state"]
    assert actual["dynamics_parameters"]["motion_subspace"]["n_dof"] == resumed._runtime_n_dof


@pytest.mark.parametrize("field", ["n_dof", "strategy", "rotational_dof_removed", "translational_dof_removed"])
@pytest.mark.parametrize("change", ["missing", "different"])
def test_exact_restart_requires_effective_dof_contract(tmp_path, field, change):
    source = NVT(str(tmp_path / "source.out"), bending_atoms(), parameters(2))
    source.run()
    path = tmp_path / "source_md.rst"
    state = read_rst(path)
    dynamics = state["dynamics_parameters"]
    assert field in dynamics["motion_subspace"]
    if change == "missing":
        del dynamics["motion_subspace"][field]
    else:
        dynamics["motion_subspace"][field] = "unknown" if field == "strategy" else 99
    # A valid checksum isolates semantic contract validation from corruption checks.
    write_rst(path, source.atoms, state["velocities"], state["step"], state["timestep"],
              state["ensemble"], state["energy"], rng_state=state["rng_state"],
              velocity_representation=state["velocity_representation"],
              pes_identity=state["pes_identity"], dynamics_parameters=dynamics)
    target = bending_atoms()
    original = target.positions.copy()
    resumed = NVT(str(tmp_path / "target.out"), target,
                  parameters(4, restart=True, rst_file=str(path)))
    with pytest.raises(RuntimeError, match="Dynamics|DOF|dynamics"):
        resumed.run()
    np.testing.assert_array_equal(target.positions, original)
    assert not list(tmp_path.glob("target_md*"))


@pytest.mark.parametrize("kind,expected", [
    ("linear", 3), ("bent", 3), ("dimer", 1), ("periodic", 6), ("anchored", 6),
])
def test_fixed_md_dimension_contract(kind, expected):
    from ase.constraints import FixAtoms

    from maple.function.dispatcher.md.utils import (
        get_persistent_motion_dof_policy,
        motion_subspace_identity,
    )

    atoms = bending_atoms()
    if kind == "bent":
        atoms.positions[1, 1] = 0.2
    elif kind == "dimer":
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    elif kind == "periodic":
        atoms.set_cell([10, 10, 10])
        atoms.pbc = True
    elif kind == "anchored":
        atoms.set_constraint(FixAtoms(indices=[0]))
    policy = get_persistent_motion_dof_policy(atoms, remove_com=True, remove_angular=True)
    identity = motion_subspace_identity(policy)
    assert identity["n_dof"] == expected
    assert identity["strategy"] == "flexible-cartesian-v1"


def test_old_dof_contract_can_only_start_a_new_run(tmp_path):
    source = NVT(str(tmp_path / "source.out"), bending_atoms(), parameters(2))
    source.run()
    path = tmp_path / "source_md.rst"
    state = read_rst(path)
    dynamics = state["dynamics_parameters"]
    dynamics["motion_subspace"] = {"com_excluded": True, "angular_excluded": True}
    write_rst(path, source.atoms, state["velocities"], state["step"], state["timestep"],
              state["ensemble"], state["energy"], rng_state=state["rng_state"],
              velocity_representation=state["velocity_representation"],
              pes_identity=state["pes_identity"], dynamics_parameters=dynamics)
    loaded = NVT(str(tmp_path / "loaded.out"), bending_atoms(),
                 parameters(2, load_state=True, rst_file=str(path)))
    loaded.run()
    assert loaded._runtime_n_dof == 3
    assert read_rst(tmp_path / "loaded_md.rst")["step"] == 2
