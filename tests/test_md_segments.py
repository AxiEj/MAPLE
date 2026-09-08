import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.rst_io import read_rst

IDENTITY = {
    "backend": "md-segment-test",
    "model_fingerprint": {
        "algorithm": "sha256",
        "digest": "e" * 64,
        "source": "test",
    },
    "relevant_settings": {"zero": True},
}


class Zero(Calculator):
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces"]
        self.maple_pes_identity = dict(IDENTITY)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def system():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    atoms.calc = Zero()
    atoms.arrays["velocities"] = np.array([[1e-4, 0, 0], [-1e-4, 0, 0]])
    return atoms


def run(output, steps, **updates):
    params = {
        "timestep": 0.1,
        "steps": steps,
        "traj_every": 5,
        "log_every": 5,
        "verbose": 0,
        "rst_every": 10,
        "remove_com": False,
        "remove_angular": False,
        "remove_com_every": 0,
        "remove_angular_every": 0,
    }
    params.update(updates)
    job = NVE(str(output), system(), params)
    job.run()
    return job


@pytest.mark.parametrize("traj_format", ["xyz", "dcd"])
def test_rollback_restart_creates_new_output_segment(tmp_path, traj_format):
    output = tmp_path / f"rollback-{traj_format}.out"
    run(output, 15, traj_format=traj_format)
    thermo = tmp_path / f"rollback-{traj_format}_md_thermo.dat"
    trajectory = tmp_path / f"rollback-{traj_format}_md_traj.{traj_format}"
    summary = tmp_path / f"rollback-{traj_format}_md_summary.txt"
    final = tmp_path / f"rollback-{traj_format}_final.xyz"
    original = {path: path.read_bytes() for path in (thermo, trajectory, summary, final)}
    previous = tmp_path / f"rollback-{traj_format}_md_prev.rst"
    assert read_rst(previous)["step"] == 10

    run(
        output,
        20,
        traj_format=traj_format,
        restart=True,
        rst_file=str(previous),
    )

    assert all(path.read_bytes() == content for path, content in original.items())
    segment_thermo = tmp_path / f"rollback-{traj_format}_md_seg0002_thermo.dat"
    segment_traj = tmp_path / f"rollback-{traj_format}_md_seg0002_traj.{traj_format}"
    segment_summary = tmp_path / f"rollback-{traj_format}_md_seg0002_summary.txt"
    segment_final = tmp_path / f"rollback-{traj_format}_md_seg0002_final.xyz"
    parent_snapshot = tmp_path / f"rollback-{traj_format}_md_seg0002_parent.rst"
    assert all(path.exists() for path in (segment_thermo, segment_traj, segment_summary, segment_final, parent_snapshot))
    assert read_rst(parent_snapshot)["step"] == 10
    text = segment_thermo.read_text()
    assert "Parent checkpoint step: 10" in text
    assert "Start step: 10" in text
    assert read_rst(tmp_path / f"rollback-{traj_format}_md.rst")["step"] == 20
