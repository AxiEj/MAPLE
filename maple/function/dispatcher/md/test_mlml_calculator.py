import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator.mlml_calculator import MLMLCalculator
from maple.function.engine import engine
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.part_reader import PartReader


class ConstantCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy: float, forces):
        super().__init__()
        self._energy = float(energy)
        self._forces = np.asarray(forces, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self._energy
        self.results["forces"] = self._forces.copy()


def test_part_reader_supports_range_and_mixed_tokens(tmp_path):
    partition_file = tmp_path / "part.txt"
    partition_file.write_text("[CORE]\n1-3 7 10-12\n", encoding="utf-8")

    partition = PartReader(str(partition_file), natoms=12)

    assert partition["core_indices"] == [0, 1, 2, 6, 9, 10, 11]
    assert partition["env_indices"] == [3, 4, 5, 7, 8]


def test_part_reader_rejects_reversed_range(tmp_path):
    partition_file = tmp_path / "part.txt"
    partition_file.write_text("[CORE]\n10-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid range '10-1'.*line 2"):
        PartReader(str(partition_file), natoms=100)


def test_part_reader_rejects_invalid_token(tmp_path):
    partition_file = tmp_path / "part.txt"
    partition_file.write_text("[CORE]\n1-2-3\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid atom index token '1-2-3'.*line 2"):
        PartReader(str(partition_file), natoms=100)


def test_part_reader_detects_duplicate_after_range_expansion(tmp_path):
    partition_file = tmp_path / "part.txt"
    partition_file.write_text("[CORE]\n1-3 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate atom index 3"):
        PartReader(str(partition_file), natoms=10)


def test_part_reader_range_out_of_bounds(tmp_path):
    partition_file = tmp_path / "part.txt"
    partition_file.write_text("[CORE]\n99-101\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"out of range 1\.\.100.*line 2"):
        PartReader(str(partition_file), natoms=100)


def test_mlml_calculator_energy_and_force_composition(monkeypatch, tmp_path):
    atoms = Atoms("H3", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.8, 0.0]])
    partition_file = tmp_path / "part.txt"
    partition_file.write_text("[CORE]\n1 2\n\n[ENV]\n3\n", encoding="utf-8")

    low_full_forces = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    high_core_forces = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ])
    low_core_forces = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    calls = {"n": 0}

    def fake_set_calculator(self):
        calls["n"] += 1
        if calls["n"] == 1:
            return ConstantCalculator(energy=10.0, forces=low_full_forces)
        if calls["n"] == 2:
            return ConstantCalculator(energy=7.0, forces=high_core_forces)
        raise AssertionError("Unexpected extra SetClaculator.set_calculator call")

    monkeypatch.setattr(
        "maple.function.calculator.mlml_calculator.SetClaculator.set_calculator",
        fake_set_calculator,
    )

    mlml = MLMLCalculator(
        output=str(tmp_path / "mlml.out"),
        device=None,
        full_atoms=atoms,
        high_model="aimnet2",
        low_model="maceoff23m",
        partition_file=str(partition_file),
    )

    # low(core) should use low_calc (first calculator object), so patch its force/energy for core evaluation
    mlml.low_calc = ConstantCalculator(energy=4.0, forces=low_core_forces)
    # full low still read from low_calc; set back via sequential swap inside calculate by monkeypatching local copies is not needed
    # use a dedicated wrapper by overriding calculate inputs below through object replacement

    # Keep full low and core low distinct by replacing in order during calculate
    original_build_core = mlml._build_core_atoms

    state = {"stage": 0}

    def staged_low_energy_forces(atoms_obj):
        state["stage"] += 1
        if len(atoms_obj) == 3:
            return 10.0, low_full_forces
        return 4.0, low_core_forces

    class StagedLowCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, f = staged_low_energy_forces(atoms)
            self.results["energy"] = e
            self.results["forces"] = np.asarray(f, dtype=float)

    mlml.low_calc = StagedLowCalc()
    mlml.high_calc = ConstantCalculator(energy=7.0, forces=high_core_forces)

    mlml.calculate(atoms=atoms)

    assert np.isclose(mlml.results["mlml_energy_low_full"], 10.0)
    assert np.isclose(mlml.results["mlml_energy_high_core"], 7.0)
    assert np.isclose(mlml.results["mlml_energy_low_core"], 4.0)
    assert np.isclose(mlml.results["mlml_energy_correction"], 3.0)
    assert np.isclose(mlml.results["energy"], 13.0)

    expected_forces = np.array([
        [2.0, 0.0, 0.0],  # 1 + (3-2)
        [0.0, 4.0, 0.0],  # 1 + (4-1)
        [0.0, 0.0, 1.0],  # env unchanged
    ])
    assert np.allclose(mlml.results["forces"], expected_forces)

    # avoid linter unused warning for local helper kept for readability of test setup
    assert original_build_core is not None


def test_engine_mlml_initiator_forwards_per_model_options(monkeypatch, tmp_path):
    captured = {}

    class FakeMLMLCalculator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeCommandControl:
        def __init__(self):
            self.params = {
                "solv": {},
                "mlml_config": {
                    "high_model": "aimnet2",
                    "low_model": "maceoff23m",
                    "partition_file": str(tmp_path / "part.txt"),
                },
                "high_model_options": {"task": "omol", "size": "uma-s-1p1"},
                "low_model_options": {"model_path": "/tmp/low.pt"},
            }

        def get(self, key, default=None):
            return self.params.get(key, default)

    eng = engine()
    eng.jobtype = "mlml"
    eng.atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    eng.output = str(tmp_path / "mlml.out")
    eng.d4 = False
    eng.commandcontrol = FakeCommandControl()

    monkeypatch.setattr("maple.function.calculator.mlml_calculator.MLMLCalculator", FakeMLMLCalculator)

    eng._mlp_initiator(model=None, device=None)

    assert captured["high_model_options"] == {"task": "omol", "size": "uma-s-1p1"}
    assert captured["low_model_options"] == {"model_path": "/tmp/low.pt"}


def test_commandcontrol_loads_mdp_for_mlml_md_when_method_is_separate(tmp_path):
    mdp_file = tmp_path / "nve.mdp"
    mdp_file.write_text("timestep = 0.5\nsteps = 20\n", encoding="utf-8")

    cc = CommandControl.from_settings(
        [
            "#mlml",
            "#method=md",
            f"#mlml_config(high_model=aimnet2, low_model=maceoff23m, partition_file={tmp_path / 'part.txt'})",
            f"#mdp={mdp_file}",
        ],
        output_path=str(tmp_path / "parse.out"),
    )

    assert cc.get("timestep") == 0.5
    assert cc.get("steps") == 20
