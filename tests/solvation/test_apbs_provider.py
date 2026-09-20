from __future__ import annotations

import importlib.util
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import apbs_pb
from maple.function.calculator.extra_correction.implicit.apbs_pb import (
    APBSLPB,
    parse_apbs_print_energy,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


def _install_fake_apbs(monkeypatch, tmp_path, calls):
    executable = tmp_path / "apbs"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(apbs_pb.shutil, "which", lambda _name: str(executable))

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        if "--version" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="APBS 3.4.1\n", stderr=""
            )
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
    return executable


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


def test_apbs_exposes_first_class_radius_and_nonpolar_providers(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = APBSLPB(atoms, atoms.get_initial_charges(), executable="apbs")

    assert provider.radius_provider.name == "openmm-mbondi2-radii"
    assert provider.radius_result.profile == "generic-mbondi2"
    assert provider.radius_result.radii_angstrom.shape == (len(atoms),)
    assert np.all(provider.radius_result.radii_angstrom > 0.0)
    assert provider.nonpolar_provider.name == "apbs-sasa"
    assert provider.nonpolar_provider.component_properties == frozenset({"energy"})
    assert provider.provenance["radius_provider"]["profile"] == "generic-mbondi2"
    assert provider.provenance["nonpolar_provider"]["name"] == "apbs-sasa"


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


def test_apbs_print_energy_parser_rejects_unnamed_positional_lines():
    output = """
    PRINT ENERGY: -2.295900000000E+02 kJ/mol
    PRINT ENERGY: 1.250000000000E+01 kJ/mol
    """
    with pytest.raises(ValueError, match="uniquely named"):
        parse_apbs_print_energy(output, "ELEC")


def test_apbs_parser_rejects_duplicate_named_results():
    output = """
    Global net ELEC energy = -2.295900000000E+02 kJ/mol
    Global net ELEC energy = -2.100000000000E+02 kJ/mol
    """
    with pytest.raises(ValueError, match="exactly one"):
        parse_apbs_print_energy(output, "ELEC")


def test_apbs_parser_rejects_duplicate_results_across_named_formats():
    output = """
    Global net ELEC energy = -2.295900000000E+02 kJ/mol
    PRINT ELEC ENERGY 1: -2.100000000000E+02 kJ/mol
    """
    with pytest.raises(ValueError, match="exactly one"):
        parse_apbs_print_energy(output, "ELEC")


def test_apbs_rejects_grid_dimension_that_apbs_would_round_down(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match=r"c\*32\+1"):
        APBSLPB(atoms, atoms.get_initial_charges(), grid_points=99)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("grid_spacing", np.nan),
        ("timeout", np.inf),
        ("probe_radius", np.nan),
        ("surface_tension", np.inf),
        ("pressure", np.nan),
    ],
)
def test_apbs_rejects_nonfinite_configuration(water_mol2, keyword, value):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match="finite"):
        APBSLPB(atoms, atoms.get_initial_charges(), **{keyword: value})


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
    calls = []
    _install_fake_apbs(monkeypatch, tmp_path, calls)
    provider = APBSLPB(atoms, atoms.get_initial_charges(), audit_dir=audit)
    result = provider.evaluate(atoms)
    assert np.isclose(
        result.energy_hartree,
        result.components_hartree["polar"] + result.components_hartree["nonpolar"],
    )
    assert (audit / "apbs.in").is_file()
    assert (audit / "molecule.pqr").is_file()
    assert (audit / "apbs.result.json").is_file()
    assert result.provenance["provider_version"] == "3.4.1"


def test_apbs_uses_one_version_probe_per_instance_and_one_solve_per_scalar(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    calls = []
    _install_fake_apbs(monkeypatch, tmp_path, calls)
    provider = APBSLPB(atoms, atoms.get_initial_charges())

    provider.evaluate(atoms)
    provider.evaluate(atoms)

    assert sum("--version" in command for command, _ in calls) == 1
    assert sum("--version" not in command for command, _ in calls) == 2


def test_apbs_path_drift_fails_before_another_solve(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    calls = []
    first = _install_fake_apbs(monkeypatch, tmp_path, calls)
    provider = APBSLPB(atoms, atoms.get_initial_charges())
    second = tmp_path / "apbs-replacement"
    second.write_bytes(first.read_bytes())
    second.chmod(0o755)
    monkeypatch.setattr(apbs_pb.shutil, "which", lambda _name: str(second))

    with pytest.raises(RuntimeError, match="path drifted"):
        provider.evaluate(atoms)

    assert sum("--version" not in command for command, _ in calls) == 0


def test_apbs_hash_drift_fails_before_another_solve(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    calls = []
    executable = _install_fake_apbs(monkeypatch, tmp_path, calls)
    provider = APBSLPB(atoms, atoms.get_initial_charges())
    executable.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="content drifted"):
        provider.evaluate(atoms)

    assert sum("--version" not in command for command, _ in calls) == 0


def test_apbs_audit_context_isolates_each_scalar_without_mutating_fallback(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    calls = []
    _install_fake_apbs(monkeypatch, tmp_path, calls)
    fallback = tmp_path / "fallback"
    provider = APBSLPB(
        atoms, atoms.get_initial_charges(), audit_dir=fallback
    )
    first = tmp_path / "scalar-1"
    second = tmp_path / "scalar-2"
    first.mkdir()
    second.mkdir()

    provider.evaluate(atoms, audit_context=SimpleNamespace(path=first))
    provider.evaluate(atoms, audit_context=SimpleNamespace(path=second))

    assert (first / "apbs.result.json").is_file()
    assert (second / "apbs.result.json").is_file()
    assert not fallback.exists()
    assert provider.audit_dir == fallback


def test_apbs_serialized_pair_rejects_collapsed_subquantum_displacements():
    provider = object.__new__(APBSLPB)
    center = np.zeros((1, 3))
    with pytest.raises(ValueError, match="collapses"):
        provider.validate_serialized_pair(center, center - 0.0004, center + 0.0004)


def test_apbs_serialized_pair_rejects_asymmetric_three_decimal_rounding():
    provider = object.__new__(APBSLPB)
    center = np.asarray([[0.0004, 0.0, 0.0]])
    minus = center.copy()
    plus = center.copy()
    minus[0, 0] -= 0.0006
    plus[0, 0] += 0.0006
    with pytest.raises(ValueError, match="symmetrically"):
        provider.validate_serialized_pair(center, minus, plus)


def test_apbs_serialized_pair_reports_effective_symmetric_step():
    provider = object.__new__(APBSLPB)
    center = np.zeros((1, 3))
    minus = center.copy()
    plus = center.copy()
    minus[0, 0] = -0.0011
    plus[0, 0] = 0.0011

    metadata = provider.validate_serialized_pair(
        center, minus, plus, requested_step_angstrom=0.0011
    )

    assert metadata["requested_step_angstrom"] == pytest.approx(0.0011)
    assert metadata["actual_symmetric_step_angstrom"] == pytest.approx(0.001)
    denominator = np.asarray(metadata["serialized_denominator_angstrom"])
    assert denominator[0, 0] == pytest.approx(0.002)


def test_apbs_revalidates_moving_grid_and_audits_displaced_serialized_center(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    calls = []
    _install_fake_apbs(monkeypatch, tmp_path, calls)
    audit = tmp_path / "moved-audit"
    provider = APBSLPB(atoms, atoms.get_initial_charges())
    moved = atoms.copy()
    moved.positions[:, 0] += 0.125

    audit.mkdir()
    result = provider.evaluate(moved, audit_context=SimpleNamespace(path=audit))

    bounds = result.provenance["grid_containment"]["grid_center"][
        "serialized_coordinate_bounds_angstrom"
    ]
    assert np.asarray(bounds["minimum"])[0] == pytest.approx(
        np.min(APBSLPB.serialized_coordinates(moved), axis=0)[0]
    )
    assert result.provenance["grid_containment"]["validated"] is True
    assert (audit / "apbs.command.json").is_file()


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
