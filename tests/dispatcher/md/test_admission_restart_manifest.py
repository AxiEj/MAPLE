import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator._ase_unit_contract import (
    ASE_STRESS_UNIT,
    MAPLE_ENERGY_UNIT,
    MAPLE_FORCE_UNIT,
)
from maple.function.dispatcher.md.ensemble.npt import NPT
from maple.function.dispatcher.md.ensemble.nve import NVE
from maple.function.dispatcher.md.ensemble.nvt import NVT
from maple.function.dispatcher.md.provenance import companion_manifest_for_rst
from maple.function.dispatcher.md.rst_io import write_rst
from maple.function.dispatcher.md.utils import VELOCITY_REPR_STANDARD


class _AdmissionCalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, *, cutoff=1.0, stress=True, model="admission-fake"):
        super().__init__()
        self.maple_model_name = model
        self.maple_pbc_md_supported = True
        self.maple_stress_supported = bool(stress)
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT
        self.maple_neighbor_cutoff = cutoff
        self.maple_model_options = {"coulomb": "none"}
        if stress:
            self.maple_stress_unit = ASE_STRESS_UNIT

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        if self.maple_stress_supported:
            self.results["stress"] = np.zeros(6)


class _ClusterCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.maple_model_name = "cluster-fake"
        self.maple_pbc_md_supported = False
        self.maple_stress_supported = False
        self.maple_energy_unit = MAPLE_ENERGY_UNIT
        self.maple_force_unit = MAPLE_FORCE_UNIT

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def _pbc_atoms(calc=None, *, cell=10.0, pbc=True):
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=[cell, cell, cell], pbc=pbc)
    atoms.calc = calc or _AdmissionCalc()
    return atoms


def _cluster_atoms(calc=None):
    atoms = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], pbc=False)
    atoms.arrays["velocities"] = np.zeros((len(atoms), 3))
    atoms.calc = calc or _ClusterCalc()
    return atoms


def _write_rst(path: Path, atoms: Atoms, *, ensemble="nve", step=1, timestep=0.1, velocities=None):
    if velocities is None:
        velocities = np.full((len(atoms), 3), 1.0e-4)
    write_rst(
        path,
        atoms=atoms,
        velocities=velocities,
        step=step,
        timestep=timestep,
        ensemble=ensemble,
        energy=0.0,
        velocity_representation=VELOCITY_REPR_STANDARD,
    )
    return path


def _manifest_for(atoms: Atoms, *, ensemble="nve", timestep=0.1, calc=None, **overrides):
    calc = calc or atoms.calc
    caps = {
        "energy_unit": getattr(calc, "maple_energy_unit", None),
        "force_unit": getattr(calc, "maple_force_unit", None),
        "neighbor_cutoff_A": getattr(calc, "maple_neighbor_cutoff", None),
        "long_range_method": (getattr(calc, "maple_model_options", {}) or {}).get("coulomb", "none"),
    }
    if getattr(calc, "maple_stress_supported", False):
        caps["stress_unit"] = getattr(calc, "maple_stress_unit", None)
    data = {
        "manifest_schema_version": "1.0.0",
        "system": {
            "n_atoms": len(atoms),
            "pbc": [bool(x) for x in atoms.pbc],
            "cell": np.asarray(atoms.cell.array, dtype=float).tolist(),
        },
        "run": {"ensemble": ensemble, "params": {"timestep": timestep}},
        "calculator": {
            "model": getattr(calc, "maple_model_name", None),
            "capabilities": caps,
        },
    }
    data.update(overrides)
    return data


def _write_companion_manifest(rst_path: Path, manifest: dict):
    path = companion_manifest_for_rst(rst_path)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def test_startup_mass_validation_rejects_zero_mass(tmp_path):
    atoms = _cluster_atoms()
    atoms.set_masses([0.0, 39.948])
    with pytest.raises(ValueError, match="atomic masses must be > 0"):
        NVE(output=str(tmp_path / "zero_mass.out"), atoms=atoms, paras={"steps": 0, "verbose": 0})


def test_load_state_revalidates_mass_after_rst_restore(tmp_path):
    source = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False))
    rst = _write_rst(tmp_path / "mass_md.rst", source, ensemble="nve")

    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False))
    sim = NVE(
        output=str(tmp_path / "mass_restart.out"),
        atoms=atoms,
        paras={"steps": 0, "verbose": 0, "load_state": True, "rst_file": str(rst)},
    )
    atoms.set_masses([0.0, 39.948])
    with pytest.raises(ValueError, match="NVE load_state admission: atomic masses"):
        sim.run()


@pytest.mark.parametrize("ensemble_cls,ensemble,extra", [
    (NVE, "nve", {}),
    (NVT, "nvt", {"thermostat": "langevin"}),
    (NPT, "npt", {"thermostat": "langevin"}),
])
@pytest.mark.parametrize("mode", ["restart", "load_state"])
def test_rst_restored_cell_cutoff_violation_rejected_for_all_ensembles(tmp_path, ensemble_cls, ensemble, extra, mode):
    restored = _pbc_atoms(_AdmissionCalc(cutoff=2.5, stress=(ensemble == "npt")), cell=4.0)
    rst = _write_rst(tmp_path / f"small_{ensemble}_md.rst", restored, ensemble=ensemble, step=1)

    atoms = _pbc_atoms(_AdmissionCalc(cutoff=2.5, stress=(ensemble == "npt")), cell=10.0)
    paras = {
        "steps": 2,
        "verbose": 0,
        "rst_file": str(rst),
        mode: True,
        "init_velocities": False,
        **extra,
    }
    with pytest.raises(ValueError, match=">= minimum-image radius"):
        ensemble_cls(output=str(tmp_path / f"{ensemble}_{mode}.out"), atoms=atoms, paras=paras).run()


def test_npt_load_state_rejects_restored_partial_pbc(tmp_path):
    restored = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=True), cell=10.0, pbc=[True, True, False])
    rst = _write_rst(tmp_path / "partial_md.rst", restored, ensemble="npt", step=1)
    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=True), cell=10.0, pbc=True)

    with pytest.raises(ValueError, match="NPT load_state admission: NPT ensemble requires full"):
        NPT(
            output=str(tmp_path / "partial.out"),
            atoms=atoms,
            paras={"steps": 0, "verbose": 0, "load_state": True, "rst_file": str(rst)},
        ).run()


def test_vrescale_zero_kinetic_energy_rejected_but_langevin_can_heat(tmp_path):
    zero_v = np.zeros((2, 3))

    vrescale_atoms = _cluster_atoms()
    vrescale_atoms.arrays["velocities"] = zero_v.copy()
    with pytest.raises(ValueError, match="thermostat=v-rescale requires non-zero active kinetic energy"):
        NVT(
            output=str(tmp_path / "zero_vrescale.out"),
            atoms=vrescale_atoms,
            paras={"steps": 1, "verbose": 0, "init_velocities": False, "thermostat": "v-rescale"},
        ).run()

    langevin_atoms = _cluster_atoms()
    langevin_atoms.arrays["velocities"] = zero_v.copy()
    NVT(
        output=str(tmp_path / "zero_langevin.out"),
        atoms=langevin_atoms,
        paras={
            "steps": 1,
            "verbose": 0,
            "init_velocities": False,
            "thermostat": "langevin",
            "random_seed": 7,
        },
    ).run()


def test_strict_restart_manifest_mismatch_hard_fails(tmp_path):
    source = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    rst = _write_rst(tmp_path / "strict_md.rst", source, ensemble="nve", step=1)
    manifest = _manifest_for(source, ensemble="nve")
    manifest["calculator"]["capabilities"]["neighbor_cutoff_A"] = 1.2
    _write_companion_manifest(rst, manifest)

    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    with pytest.raises(RuntimeError, match="Strict restart manifest consistency failed"):
        NVE(
            output=str(tmp_path / "strict.out"),
            atoms=atoms,
            paras={"steps": 2, "verbose": 0, "restart": True, "rst_file": str(rst)},
        ).run()


def test_load_state_manifest_mismatch_warns_and_continues(tmp_path):
    source = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    rst = _write_rst(tmp_path / "load_md.rst", source, ensemble="nve", step=1)
    manifest = _manifest_for(source, ensemble="nve")
    manifest["calculator"]["capabilities"]["long_range_method"] = "ewald"
    _write_companion_manifest(rst, manifest)

    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    NVE(
        output=str(tmp_path / "load.out"),
        atoms=atoms,
        paras={"steps": 1, "verbose": 0, "load_state": True, "rst_file": str(rst)},
    ).run()
    assert "load_state companion manifest consistency issue" in (tmp_path / "load.out").read_text()


def test_strict_restart_manifest_missing_fields_fails_unless_marked_legacy(tmp_path):
    source = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    rst = _write_rst(tmp_path / "missing_md.rst", source, ensemble="nve", step=1)
    _write_companion_manifest(rst, {"system": {"n_atoms": len(source)}})

    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    with pytest.raises(RuntimeError, match="missing required fields"):
        NVE(
            output=str(tmp_path / "missing.out"),
            atoms=atoms,
            paras={"steps": 2, "verbose": 0, "restart": True, "rst_file": str(rst)},
        ).run()

    legacy_rst = _write_rst(tmp_path / "legacy_md.rst", source, ensemble="nve", step=1)
    _write_companion_manifest(legacy_rst, {"manifest_status": "legacy"})
    legacy_atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    NVE(
        output=str(tmp_path / "legacy.out"),
        atoms=legacy_atoms,
        paras={"steps": 1, "verbose": 0, "load_state": True, "rst_file": str(legacy_rst)},
    ).run()
    assert "marked legacy/incomplete" in (tmp_path / "legacy.out").read_text()


def test_explicit_rst_file_uses_source_directory_companion_manifest(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    rst = _write_rst(source_dir / "external_md.rst", source, ensemble="nve", step=1)
    manifest = _manifest_for(source, ensemble="nve")
    manifest["calculator"]["model"] = "different-model"
    _write_companion_manifest(rst, manifest)

    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=False), cell=10.0)
    with pytest.raises(RuntimeError, match="calculator.model"):
        NVE(
            output=str(tmp_path / "current.out"),
            atoms=atoms,
            paras={"steps": 2, "verbose": 0, "restart": True, "rst_file": str(rst)},
        ).run()


def test_npt_pbc_banner_mentions_variable_cell_msd_caveat(tmp_path):
    atoms = _pbc_atoms(_AdmissionCalc(cutoff=1.0, stress=True), cell=10.0)
    NPT(output=str(tmp_path / "banner.out"), atoms=atoms, paras={"steps": 1, "verbose": 0}).run()
    text = (tmp_path / "banner.out").read_text()
    assert "fixed-cell NVE/NVT" in text
    assert "variable-cell NPT" in text
    assert "fractional displacements" in text
