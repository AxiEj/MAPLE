"""WS3 — MD provenance manifest."""

import json

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._ase_unit_contract import ASE_STRESS_UNIT
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.provenance import (
    MANIFEST_SCHEMA_VERSION,
    RST_SCHEMA_VERSION,
    build_md_manifest,
    collect_calculator_provenance,
    collect_environment_provenance,
    final_state_hash,
)
from maple.function.dispatcher.md.semantics import resolve_md_dof_policy


class _Calc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, pbc_capable=True):
        super().__init__()
        self.maple_model_name = "fake-pbc" if pbc_capable else "fake-cluster"
        self.maple_model_options = {"foundation": "test", "default_dtype": "float64"}
        self.maple_pbc_md_supported = pbc_capable
        self.maple_stress_supported = False

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def _periodic(calc):
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[6.0, 6.0, 6.0], pbc=True)
    atoms.calc = calc
    return atoms


def test_environment_provenance_keys():
    env = collect_environment_provenance()
    for key in ("maple_git_commit", "maple_git_branch", "python", "numpy", "ase"):
        assert key in env


def test_calculator_provenance_records_capabilities():
    calc = _Calc(pbc_capable=True)
    prov = collect_calculator_provenance(calc, model="fake-pbc", device="cpu")
    assert prov["model"] == "fake-pbc"
    assert prov["capabilities"]["pbc_md_supported"] is True
    assert prov["model_options"] == {"foundation": "test", "default_dtype": "float64"}
    assert prov["device"] == "cpu"


def test_final_state_hash_is_deterministic_and_position_sensitive():
    atoms = _periodic(_Calc())
    v = np.array([[0.01, 0.0, 0.0]])
    h1 = final_state_hash(atoms, v)
    h2 = final_state_hash(atoms, v)
    assert h1 == h2 and len(h1) == 64
    moved = atoms.copy()
    moved.positions[0, 0] += 0.1
    assert final_state_hash(moved, v) != h1


def test_build_manifest_schema_and_sections():
    atoms = _periodic(_Calc())
    policy = resolve_md_dof_policy(atoms, _Params(), "nve")
    from maple.function.dispatcher.md.provenance import build_run_context

    run_context = build_run_context(params=_Params(), dof_policy=policy, ensemble="nve")
    manifest = build_md_manifest(
        atoms=atoms,
        run_context=run_context,
        calc_provenance=collect_calculator_provenance(atoms.calc),
        rst_path="/tmp/x.rst",
        final_velocities=np.zeros((1, 3)),
    )
    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["rst_schema_version"] == RST_SCHEMA_VERSION
    for section in ("environment", "calculator", "system", "run", "final_state_hash"):
        assert section in manifest
    assert manifest["system"]["pbc"] == [True, True, True]
    assert manifest["run"]["dof_policy"]["runtime_n_dof"] == policy.runtime_n_dof
    assert manifest["run"]["partial_pbc"]["is_partial"] is False


def test_partial_pbc_flag_in_manifest():
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]], cell=[6.0, 6.0, 20.0], pbc=[True, True, False])
    atoms.calc = _Calc()
    manifest = build_md_manifest(
        atoms=atoms,
        run_context={"partial_pbc": {"allowed": True}},
        calc_provenance=None,
    )
    assert manifest["run"]["partial_pbc"]["is_partial"] is True


def test_nve_run_writes_manifest(tmp_path):
    atoms = _periodic(_Calc(pbc_capable=True))
    atoms.calc.maple_provenance = collect_calculator_provenance(atoms.calc, model="fake-pbc")
    atoms.arrays["velocities"] = np.array([[0.001, 0.0, 0.0]])
    NVE(
        output=str(tmp_path / "nve.out"),
        atoms=atoms,
        paras={
            "steps": 1, "timestep": 0.5, "init_velocities": False,
            "remove_com_every": 0, "verbose": 0, "log_every": 1,
            "traj_every": 1, "rst_every": 1,
        },
    ).run()

    manifest_file = tmp_path / "nve_md_manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text())
    assert manifest["run"]["ensemble"] == "nve"
    assert manifest["calculator"]["capabilities"]["pbc_md_supported"] is True
    assert manifest["system"]["pbc"] == [True, True, True]
    assert manifest["rst_path"].endswith("nve_md.rst")
    assert len(manifest["final_state_hash"]) == 64
    assert manifest["run"]["dof_policy"]["init_n_dof"] >= 1


def test_long_range_method_from_coulomb_option():
    calc = _Calc()
    calc.maple_model_options = {"coulomb": "ewald", "foundation": "test"}
    prov = collect_calculator_provenance(calc)
    assert prov["capabilities"]["long_range_method"] == "ewald"


def test_long_range_method_defaults_to_none():
    prov = collect_calculator_provenance(_Calc())
    assert prov["capabilities"]["long_range_method"] == "none"


def test_validation_artifact_id_surfaces_at_manifest_top_level():
    from maple.function.dispatcher.md.provenance import build_run_context

    atoms = _periodic(_Calc())

    class _ParamsWithId(_Params):
        validation_artifact_id = "md_acceptance_TEST_123"

    params = _ParamsWithId()
    run_context = build_run_context(
        params=params, dof_policy=resolve_md_dof_policy(atoms, params, "nve"), ensemble="nve"
    )
    manifest = build_md_manifest(atoms=atoms, run_context=run_context, calc_provenance=None)
    assert manifest["validation_artifact_id"] == "md_acceptance_TEST_123"
    # Surfaced once at the top level, not duplicated inside the run block.
    assert "validation_artifact_id" not in manifest["run"]


def test_nve_run_surfaces_validation_artifact_id(tmp_path):
    atoms = _periodic(_Calc(pbc_capable=True))
    atoms.calc.maple_provenance = collect_calculator_provenance(atoms.calc, model="fake-pbc")
    atoms.arrays["velocities"] = np.zeros((1, 3))
    NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={
        "steps": 1, "timestep": 0.5, "init_velocities": False, "remove_com_every": 0,
        "verbose": 0, "log_every": 1, "traj_every": 1, "rst_every": 1,
        "validation_artifact_id": "md_acceptance_TEST_123",
    }).run()
    manifest = json.loads((tmp_path / "nve_md_manifest.json").read_text())
    assert manifest["validation_artifact_id"] == "md_acceptance_TEST_123"
    # Banner records it too (start_simulation writes to the main output file).
    assert "md_acceptance_TEST_123" in (tmp_path / "nve.out").read_text()


def test_calc_provenance_falls_back_when_no_maple_provenance(tmp_path):
    from ase.build import bulk
    from maple.function.dispatcher.md.validation import MapleLJReferenceCalculator

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)
    atoms.calc = MapleLJReferenceCalculator(rc=4.0)
    assert not hasattr(atoms.calc, "maple_provenance")  # engine never annotated it
    atoms.arrays["velocities"] = np.zeros((len(atoms), 3))
    NVE(output=str(tmp_path / "nve.out"), atoms=atoms, paras={
        "steps": 1, "timestep": 0.5, "init_velocities": False, "remove_com_every": 0,
        "verbose": 0, "log_every": 1, "traj_every": 1, "rst_every": 1,
    }).run()
    manifest = json.loads((tmp_path / "nve_md_manifest.json").read_text())
    block = manifest["calculator"]
    assert block is not None and block["model"] == "lj-reference"
    caps = block["capabilities"]
    assert caps["neighbor_cutoff_A"] == pytest.approx(4.0)
    assert caps["stress_unit"] is not None
    assert caps["long_range_method"] == "none"


class _Params:
    remove_com = True
    remove_rotation = False
    remove_angular = False
    remove_com_every = 0
    remove_angular_every = 0
    thermostat = ""
    barostat = ""
    allow_partial_pbc = False
    validation_artifact_id = ""
    random_seed = 7
