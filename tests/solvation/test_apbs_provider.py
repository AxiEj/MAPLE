from __future__ import annotations

import importlib.util
import shutil
import subprocess

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import apbs_pb
from maple.function.calculator.extra_correction.implicit.apbs_pb import (
    APBSLPB,
    parse_apbs_print_energy,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


def test_apbs_input_contains_solvated_minus_reference_and_apolar(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = APBSLPB(atoms, atoms.get_initial_charges(), executable="apbs")
    rendered = provider.render_input("molecule.pqr")
    assert "lpbe" in rendered
    assert "print elecEnergy solv - ref end" in rendered
    assert "apolar name nonpolar" in rendered
    assert "print apolEnergy nonpolar end" in rendered
    assert "gamma 0.10500000" in rendered
    assert "press 0.00000000" in rendered
    assert rendered.count("swin 0.3") == 3
    assert provider.provenance["nonpolar"] == "sasa"


def test_apbs_print_energy_parser_reads_kj_per_mol_values():
    output = """
    PRINT ELEC ENERGY 1: -2.295900000000E+02 kJ/mol
    PRINT APOL ENERGY 1: 1.250000000000E+01 kJ/mol
    """
    assert parse_apbs_print_energy(output, "ELEC") == -229.59
    assert parse_apbs_print_energy(output, "APOL") == 12.5


def test_apbs_print_energy_parser_reads_official_global_net_output():
    output = """
      Global net ELEC energy = -2.295900000000E+02 kJ/mol
      Global net APOL energy = 1.250000000000E+01 kJ/mol
    """
    assert parse_apbs_print_energy(output, "ELEC") == -229.59
    assert parse_apbs_print_energy(output, "APOL") == 12.5


def test_apbs_print_energy_parser_accepts_generic_print_lines_in_order():
    output = """
    PRINT ENERGY: -2.295900000000E+02 kJ/mol
    PRINT ENERGY: 1.250000000000E+01 kJ/mol
    """
    assert parse_apbs_print_energy(output, "ELEC") == -229.59
    assert parse_apbs_print_energy(output, "APOL") == 12.5


def test_apbs_parser_never_reuses_one_polar_value_as_nonpolar():
    output = "PRINT ENERGY: -2.295900000000E+02 kJ/mol\n"
    with pytest.raises(ValueError, match="APOL"):
        parse_apbs_print_energy(output, "APOL")


def test_apbs_rejects_grid_dimension_that_apbs_would_round_down(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match=r"c\*32\+1"):
        APBSLPB(atoms, atoms.get_initial_charges(), grid_points=99)


def test_apbs_missing_executable_is_actionable(water_mol2, monkeypatch):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = APBSLPB(atoms, atoms.get_initial_charges())
    monkeypatch.setattr(apbs_pb.shutil, "which", lambda _name: None)
    with pytest.raises(ImportError, match="optional 'apbs' executable"):
        provider.evaluate(atoms)


def test_apbs_adapter_composes_polar_nonpolar_and_writes_audit(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    audit = tmp_path / "audit"
    provider = APBSLPB(atoms, atoms.get_initial_charges(), audit_dir=audit)
    monkeypatch.setattr(apbs_pb.shutil, "which", lambda _name: "/opt/apbs/bin/apbs")

    def fake_run(command, **_kwargs):
        if "--version" in command:
            assert _kwargs.get("cwd") is not None
            return subprocess.CompletedProcess(command, 0, stdout="APBS 3.4.1\n", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Global net ELEC energy = -2.000000000000E+02 kJ/mol\n"
                "Global net APOL energy = 1.000000000000E+01 kJ/mol\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(apbs_pb.subprocess, "run", fake_run)
    result = provider.evaluate(atoms)
    assert np.isclose(
        result.energy_hartree,
        result.components_hartree["polar"] + result.components_hartree["nonpolar"],
    )
    assert (audit / "apbs.in").is_file()
    assert (audit / "molecule.pqr").is_file()
    assert (audit / "apbs.result.json").is_file()
    assert result.provenance["provider_version"] == "3.4.1"


@pytest.mark.skipif(
    shutil.which("apbs") is None or importlib.util.find_spec("openmm") is None,
    reason="real APBS smoke requires the optional APBS and OpenMM providers",
)
def test_real_apbs_smoke_when_executable_is_available(water_mol2, tmp_path):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    result = APBSLPB(
        atoms,
        atoms.get_initial_charges(),
        audit_dir=tmp_path / "real-apbs-audit",
    ).evaluate(atoms)

    assert np.isfinite(result.energy_hartree)
    assert np.isclose(
        result.energy_hartree,
        result.components_hartree["polar"] + result.components_hartree["nonpolar"],
    )
    assert result.provenance["provider_version"] != "unknown-external"
