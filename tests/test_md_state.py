from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.rst_io import read_rst, write_rst

IDENTITY = {
    "backend": "md-state-test",
    "model_fingerprint": {
        "algorithm": "sha256",
        "digest": "d" * 64,
        "source": "test",
    },
    "relevant_settings": {"k": 0.02},
}


class Harmonic(Calculator):
    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy", "forces"]
        self.maple_pes_identity = dict(IDENTITY)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        positions = atoms.get_positions()
        self.results = {
            "energy": 0.01 * float(np.sum(positions**2)),
            "forces": -0.02 * positions,
        }


def water(*, linear=False):
    positions = (
        [[-0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]
        if linear
        else [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    )
    atoms = Atoms("HOH", positions=positions)
    atoms.info.update(charge=0, mult=1)
    atoms.calc = Harmonic()
    return atoms


def nvt_params(steps, **updates):
    params = {
        "timestep": 0.1,
        "steps": steps,
        "temperature": 300.0,
        "thermostat": "v-rescale",
        "tau_t": 100.0,
        "traj_every": 1000,
        "log_every": 1000,
        "verbose": 0,
        "restart": False,
        "load_state": False,
        "rst_every": 4,
        "remove_com": True,
        "remove_angular": True,
        "remove_com_every": 0,
        "remove_angular_every": 0,
        "random_seed": 271828,
    }
    params.update(updates)
    return params


def test_restart_configures_vrescale_from_restored_checkpoint_geometry(tmp_path):
    continuous = NVT(str(tmp_path / "continuous.out"), water(), nvt_params(8))
    continuous.run()
    expected = read_rst(tmp_path / "continuous_md.rst")

    split = NVT(str(tmp_path / "split.out"), water(), nvt_params(4))
    split.run()
    checkpoint = tmp_path / "split_md.rst"

    resumed = NVT(
        str(tmp_path / "split.out"),
        water(linear=True),
        nvt_params(8, restart=True, rst_file=str(checkpoint)),
    )
    resumed.run()
    actual = read_rst(tmp_path / "split_md.rst")

    assert resumed._runtime_n_dof == 3
    assert np.allclose(actual["positions"], expected["positions"], atol=1e-15)
    assert np.allclose(actual["velocities"], expected["velocities"], atol=1e-15)
    assert actual["rng_state"] == expected["rng_state"]
    segment_thermo = (tmp_path / "split_md_seg0002_thermo.dat").read_text()
    segment_summary = (tmp_path / "split_md_seg0002_summary.txt").read_text()
    assert "diagnostics are segment-local" in segment_thermo
    assert "Energy reporting basis:     vrescale-conserved" in segment_summary
    assert "Diagnostic scope:           current output segment" in segment_summary


@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_load_state_configures_from_candidate_geometry(tmp_path, ensemble):
    source = water(linear=True)
    checkpoint = tmp_path / f"{ensemble}.rst"
    write_rst(
        checkpoint,
        source,
        np.full((3, 3), 1.0e-4),
        step=4,
        timestep=0.1,
        ensemble=ensemble,
        energy=0.0,
        rng_state=None,
        velocity_representation="standard",
        pes_identity=IDENTITY,
        dynamics_parameters={"source": True},
    )
    params = {
        "steps": 1,
        "timestep": 0.1,
        "load_state": True,
        "rst_file": str(checkpoint),
        "verbose": 0,
        "traj_every": 1000,
        "log_every": 1000,
        "rst_every": 1000,
        "remove_com": True,
        "remove_angular": True,
        "remove_com_every": 0,
        "remove_angular_every": 0,
    }
    if ensemble == "nvt":
        params.update(thermostat="v-rescale", tau_t=100.0, random_seed=9)
        dynamics = NVT(str(tmp_path / "loaded.out"), water(), params)
    else:
        dynamics = NVE(str(tmp_path / "loaded.out"), water(), params)
    dynamics.run()
    assert dynamics._runtime_n_dof == 4


def test_failed_restart_does_not_mutate_live_atoms_or_create_segment(tmp_path):
    source = NVT(str(tmp_path / "source.out"), water(), nvt_params(4))
    source.run()
    target = water(linear=True)
    original = target.positions.copy()
    resumed = NVT(
        str(tmp_path / "target.out"),
        target,
        nvt_params(8, restart=True, rst_file=str(tmp_path / "source_md.rst"), tau_t=50.0),
    )
    with pytest.raises(RuntimeError, match="Dynamics-parameter mismatch"):
        resumed.run()
    assert np.array_equal(target.positions, original)
    assert not list(tmp_path.glob("target_md_seg*"))


def test_completed_restart_installs_checkpoint_without_opening_segment(tmp_path):
    output = tmp_path / "completed.out"
    source = NVT(str(output), water(), nvt_params(4))
    source.run()
    checkpoint = read_rst(tmp_path / "completed_md.rst")
    target = water(linear=True)
    target.positions[:] = 50.0
    resumed = NVT(
        str(output), target,
        nvt_params(4, restart=True, rst_file=str(tmp_path / "completed_md.rst")),
    )
    resumed.run()
    assert np.array_equal(resumed.atoms.positions, checkpoint["positions"])
    assert np.array_equal(resumed.atoms.arrays["velocities"], checkpoint["velocities"])
    assert not list(tmp_path.glob("completed_md_seg*"))


def test_load_state_uses_new_run_rng_not_checkpoint_rng(tmp_path):
    from maple.function.dispatcher.md.rst_io import get_rng_state_hex

    source = water()
    velocities = np.full((3, 3), 1.0e-4)
    paths = []
    for index, seed in enumerate((1, 2)):
        path = tmp_path / f"rng-{index}.rst"
        write_rst(
            path, source, velocities, 4, 0.1, "nvt", 0.0,
            rng_state=get_rng_state_hex(np.random.default_rng(seed)),
            velocity_representation="standard", pes_identity=IDENTITY,
            dynamics_parameters={"source": True},
        )
        paths.append(path)

    final = []
    for index, path in enumerate(paths):
        params = nvt_params(
            2, thermostat="langevin", friction=0.01, load_state=True,
            rst_file=str(path), restart=False, random_seed=1234,
        )
        job = NVT(str(tmp_path / f"load-rng-{index}.out"), water(), params)
        job.run()
        final.append(job.atoms.arrays["velocities"].copy())
    assert np.array_equal(final[0], final[1])


def test_exact_langevin_restart_matches_continuous_without_rebinding(tmp_path):
    params = nvt_params(
        8, thermostat="langevin", friction=0.01,
        remove_angular=False, remove_com=False,
    )
    continuous = NVT(str(tmp_path / "lg-cont.out"), water(), params)
    continuous.run()
    expected = read_rst(tmp_path / "lg-cont_md.rst")

    first = NVT(str(tmp_path / "lg-split.out"), water(), {**params, "steps": 4})
    first.run()
    resumed = NVT(
        str(tmp_path / "lg-split.out"), water(),
        {**params, "restart": True, "rst_file": str(tmp_path / "lg-split_md.rst")},
    )
    resumed.run()
    actual = read_rst(tmp_path / "lg-split_md.rst")
    assert np.array_equal(actual["positions"], expected["positions"])
    assert np.array_equal(actual["velocities"], expected["velocities"])
    assert actual["rng_state"] == expected["rng_state"]


def test_failed_carried_load_state_conversion_has_no_output_side_effect(tmp_path):
    from maple.function.calculator.electronic_state import electronic_state_identity

    class FailingForce(Harmonic):
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            raise RuntimeError("force probe failed")

    source = water()
    checkpoint = tmp_path / "carried.rst"
    write_rst(
        checkpoint, source, np.full((3, 3), 1.0e-4), 4, 0.1, "nvt", 0.0,
        rng_state=None, velocity_representation="lfmiddle_carried",
        pes_identity=electronic_state_identity(source),
        dynamics_parameters={"source": True},
    )
    target = water()
    target.calc = FailingForce()
    output = tmp_path / "failed-load.out"
    job = NVT(
        str(output), target,
        nvt_params(2, thermostat="langevin", load_state=True, rst_file=str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="force probe failed"):
        job.run()
    assert not output.exists()
    assert not list(tmp_path.glob("failed-load_md*"))


def test_npt_load_state_builds_components_on_restored_cell(tmp_path):
    from maple.function.dispatcher.md.ensemble.npt import NPT

    class StressHarmonic(Harmonic):
        def __init__(self):
            super().__init__()
            self.implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["stress"] = np.zeros(6)

    source = water()
    source.set_cell(np.diag([8.0, 9.0, 10.0]))
    source.set_pbc(True)
    source.calc = StressHarmonic()
    checkpoint = tmp_path / "npt-load.rst"
    write_rst(
        checkpoint, source, np.full((3, 3), 1e-4), 4, 0.1, "npt", 0.0,
        rng_state=None, velocity_representation="standard",
        pes_identity=IDENTITY, dynamics_parameters={"source": True},
    )
    target = source.copy()
    target.calc = StressHarmonic()
    target.set_cell(np.eye(3) * 20, scale_atoms=False)
    job = NPT(str(tmp_path / "npt-loaded.out"), target, {
        "steps": 1, "timestep": 0.1, "load_state": True,
        "rst_file": str(checkpoint), "verbose": 0,
        "thermostat": "v-rescale", "barostat": "berendsen",
        "compressibility": 0.0, "remove_com": False,
        "remove_com_every": 0, "traj_every": 1000,
        "log_every": 1000, "rst_every": 1000,
    })
    job.run()
    assert np.array_equal(job.atoms.cell.array, source.cell.array)
    assert job.thermostat.atoms is job.barostat.atoms is job.atoms


def test_restart_promotes_the_exact_bytes_that_were_validated(tmp_path, monkeypatch):
    source = NVT(str(tmp_path / "race-source.out"), water(), nvt_params(4))
    source.run()
    external = tmp_path / "external.rst"
    original_bytes = (tmp_path / "race-source_md.rst").read_bytes()
    external.write_bytes(original_bytes)

    import maple.function.dispatcher.md.ensemble.nvt as nvt_module

    original_validate = nvt_module.validate_prepared_restart
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path):
        nonlocal reads
        if path == external:
            reads += 1
        return original_read_bytes(path)

    def validate_then_replace_source(prepared, **kwargs):
        result = original_validate(prepared, **kwargs)
        prepared.source.write_text("replaced after validation")
        return result

    monkeypatch.setattr(nvt_module, "validate_prepared_restart", validate_then_replace_source)
    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    resumed = NVT(
        str(tmp_path / "race-target.out"), water(linear=True),
        nvt_params(8, restart=True, rst_file=str(external)),
    )
    resumed.run()
    parent = tmp_path / "race-target_md_seg0002_parent.rst"
    assert parent.read_bytes() == original_bytes
    assert read_rst(tmp_path / "race-target_md.rst")["step"] == 8
    assert reads == 1


def test_completed_external_restart_publishes_canonical_checkpoint(tmp_path):
    source = NVT(str(tmp_path / "complete-source.out"), water(), nvt_params(4))
    source.run()
    checkpoint = tmp_path / "complete-source_md.rst"
    expected = read_rst(checkpoint)

    target_output = tmp_path / "new-base.out"
    resumed = NVT(
        str(target_output), water(linear=True),
        nvt_params(4, restart=True, rst_file=str(checkpoint)),
    )
    resumed.run()
    canonical = tmp_path / "new-base_md.rst"
    assert read_rst(canonical)["step"] == expected["step"]
    assert np.array_equal(resumed.atoms.positions, expected["positions"])
    assert not list(tmp_path.glob("new-base_md_seg*"))


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_failed_final_checkpoint_keeps_prior_canonical_and_no_completion(
    tmp_path, monkeypatch, failure
):
    from maple.function.dispatcher.md import rst_io

    output = tmp_path / f"atomic-{failure}.out"
    first = NVT(str(output), water(), nvt_params(4))
    first.run()
    canonical = tmp_path / f"atomic-{failure}_md.rst"
    prior_bytes = canonical.read_bytes()
    completion_count = output.read_text().count("MD SIMULATION COMPLETED")

    if failure == "serialize":
        monkeypatch.setattr(
            rst_io,
            "_serialize_rst",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("serialize failed")),
        )
    else:
        original_replace = rst_io.os.replace

        def fail_canonical_replace(source, destination):
            if Path(destination) == canonical:
                raise OSError("replace failed")
            return original_replace(source, destination)

        monkeypatch.setattr(rst_io.os, "replace", fail_canonical_replace)

    resumed = NVT(
        str(output), water(),
        nvt_params(8, restart=True, rst_file=str(canonical), rst_every=1000),
    )
    with pytest.raises((RuntimeError, OSError), match=f"{failure} failed"):
        resumed.run()
    assert canonical.read_bytes() == prior_bytes
    assert read_rst(canonical)["step"] == 4
    assert output.read_text().count("MD SIMULATION COMPLETED") == completion_count


def test_completed_exact_restart_rejects_nonzero_frozen_velocity(tmp_path):
    from maple.function.calculator.electronic_state import electronic_state_identity
    from maple.function.dispatcher.md.rst_io import get_rng_state_hex

    atoms = water()
    atoms.set_constraint(FixAtoms(indices=[0]))
    probe = NVT(str(tmp_path / "probe.out"), atoms, nvt_params(4))
    dynamics_parameters = probe._build_actual_state(atoms)[3]
    velocities = np.full((3, 3), 1e-4)
    velocities[0] = [0.01, 0.02, 0.03]
    checkpoint = tmp_path / "ghost.rst"
    write_rst(
        checkpoint, atoms, velocities, 4, 0.1, "nvt", 0.0,
        rng_state=get_rng_state_hex(np.random.default_rng(8)),
        velocity_representation="standard",
        pes_identity=electronic_state_identity(atoms),
        dynamics_parameters=dynamics_parameters,
    )
    output = tmp_path / "ghost-target.out"
    job = NVT(
        str(output), atoms,
        nvt_params(4, restart=True, rst_file=str(checkpoint)),
    )
    with pytest.raises(RuntimeError, match="frozen.*velocity"):
        job.run()
    assert not (tmp_path / "ghost-target_md.rst").exists()
    assert not output.exists()


def test_load_state_projects_frozen_velocity_before_start(tmp_path):
    atoms = water()
    atoms.set_constraint(FixAtoms(indices=[0]))
    velocities = np.full((3, 3), 1e-4)
    velocities[0] = [0.01, 0.02, 0.03]
    checkpoint = tmp_path / "load-ghost.rst"
    write_rst(
        checkpoint, atoms, velocities, 4, 0.1, "nvt", 0.0,
        velocity_representation="standard", pes_identity=IDENTITY,
        dynamics_parameters={"source": True},
    )
    job = NVT(
        str(tmp_path / "load-ghost.out"), atoms,
        nvt_params(1, load_state=True, restart=False, rst_file=str(checkpoint)),
    )
    job.run()
    assert np.array_equal(job.atoms.arrays["velocities"][0], np.zeros(3))


def test_exact_restart_rejects_tampered_velocity_representation(tmp_path):
    source_output = tmp_path / "repr-source.out"
    source = NVT(
        str(source_output), water(),
        nvt_params(4, thermostat="langevin", friction=0.01),
    )
    source.run()
    original = tmp_path / "repr-source_md.rst"
    tampered = tmp_path / "repr-tampered.rst"
    tampered.write_text(
        original.read_text().replace(
            "velocity_representation = lfmiddle_carried",
            "velocity_representation = standard",
        )
    )

    output = tmp_path / "repr-target.out"
    resumed = NVT(
        str(output), water(),
        nvt_params(
            8, thermostat="langevin", friction=0.01,
            restart=True, rst_file=str(tampered),
        ),
    )
    with pytest.raises(RuntimeError, match="Velocity-representation mismatch"):
        resumed.run()
    assert not output.exists()
    assert not (tmp_path / "repr-target_md.rst").exists()
    assert not list(tmp_path.glob("repr-target_md_seg*"))
