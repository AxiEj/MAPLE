from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import charges
from maple.function.calculator.extra_correction.implicit.charges import prepare_charges
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


def test_missing_antechamber_dependency_is_actionable(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(charges.shutil, "which", lambda _name: None)
    with pytest.raises(ImportError, match=r"AmberTools24\+"):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "abcg2", "mode": "fixed"},
            tmp_path,
        )


def test_ambertools_failure_never_falls_back_to_qeq(water_mol2, tmp_path, monkeypatch):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )
    qeq_called = False

    def unexpected_qeq(*_args, **_kwargs):
        nonlocal qeq_called
        qeq_called = True
        raise AssertionError("QEq must not be used as a provider fallback")

    def failed_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 2, stdout="", stderr="provider failed"
        )

    monkeypatch.setattr(charges.QEqGTO, "solve", unexpected_qeq)
    monkeypatch.setattr(charges.subprocess, "run", failed_run)

    with pytest.raises(
        RuntimeError, match="Antechamber am1bcc charge generation failed"
    ):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )

    assert qeq_called is False
    assert not (tmp_path / "charges-am1bcc.mol2").exists()


@pytest.mark.parametrize("method, flag", [("am1bcc", "bcc"), ("abcg2", "abcg2")])
def test_ambertools_charge_orchestration_and_mapping_audit(
    water_mol2, tmp_path, monkeypatch, method, flag
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(input_path.read_text(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": method, "mode": "fixed", "geometry": "keep"},
        tmp_path,
    )
    command = result.provenance["command"]
    assert command[command.index("-c") + 1] == flag
    assert command[command.index("-an") + 1] == "n"
    assert command[command.index("-du") + 1] == "n"
    assert command[command.index("-seq") + 1] == "n"
    assert Path(result.provenance["atom_mapping"]).is_file()
    assert result.charges.tolist() == atoms.get_initial_charges().tolist()


def test_ambertools_charge_input_uses_current_atoms_geometry(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    current = atoms.get_positions() + np.asarray([2.0, -1.0, 0.5])
    atoms.set_positions(current)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        assert input_path != water_mol2.resolve()
        provider_input = MOL2Reader(str(input_path), charge=0, mult=1)
        np.testing.assert_allclose(provider_input.get_positions(), current, atol=1.0e-10)
        assert provider_input.info["mol2"]["atom_names"] == ["O1", "H1", "H2"]
        output_path.write_text(input_path.read_text(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": "am1bcc", "mode": "fixed", "geometry": "keep"},
        tmp_path,
    )

    np.testing.assert_allclose(result.reference_positions, current)
    np.testing.assert_allclose(result.provider_positions, current)
    assert Path(result.provenance["provider_input_mol2"]).is_file()


def test_ambertools_rejects_coarse_provider_coordinate_tokens(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        lines = input_path.read_text(encoding="utf-8").splitlines()
        section = ""
        for index, line in enumerate(lines):
            if line.startswith("@<TRIPOS>"):
                section = line[9:]
                continue
            fields = line.split()
            if section == "ATOM" and fields and fields[0] == "2":
                fields[2] = "1"
                lines[index] = " ".join(fields)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="coordinate token .* is coarser"):
        prepare_charges(
            atoms,
            {
                "source": "maple",
                "method": "am1bcc",
                "mode": "fixed",
                "geometry": "keep",
            },
            tmp_path,
        )


def test_ambertools_rejects_coarse_provider_charge_tokens(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        lines = input_path.read_text(encoding="utf-8").splitlines()
        section = ""
        replacement_charges = {"1": "-1", "2": "0", "3": "0"}
        for index, line in enumerate(lines):
            if line.startswith("@<TRIPOS>"):
                section = line[9:]
                continue
            fields = line.split()
            if section == "ATOM" and fields:
                fields[8] = replacement_charges[fields[0]]
                lines[index] = " ".join(fields)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="charge token .* is coarser"):
        prepare_charges(
            atoms,
            {
                "source": "maple",
                "method": "am1bcc",
                "mode": "fixed",
                "geometry": "keep",
            },
            tmp_path,
        )


def test_ambertools_charge_input_rejects_source_mol2_drift_before_execution(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    water_mol2.write_text(
        water_mol2.read_text(encoding="utf-8").replace("O.3", "O.2", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )
    provider_called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("Antechamber must not run after source drift")

    monkeypatch.setattr(charges.subprocess, "run", unexpected_run)

    with pytest.raises(ValueError, match="source MOL2 changed after MOL2Reader"):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )
    assert provider_called is False


def test_ambertools_charge_input_rejects_mutated_mol2_metadata_before_execution(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    atoms.info["mol2"]["atom_types"][0] = "O.2"
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )
    provider_called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("Antechamber must not run after metadata drift")

    monkeypatch.setattr(charges.subprocess, "run", unexpected_run)

    with pytest.raises(ValueError, match="metadata changed after MOL2Reader"):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )
    assert provider_called is False


def test_ambertools_charge_input_rejects_preparation_after_atom_reordering(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    reordered = atoms[[1, 0, 2]]
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    with pytest.raises(ValueError, match="source MOL2 atom IDs"):
        prepare_charges(
            reordered,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )


def test_ambertools_charge_output_permutation_is_mapped_by_atom_ids(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        lines = input_path.read_text().splitlines()
        atom_start = lines.index("@<TRIPOS>ATOM") + 1
        bond_start = lines.index("@<TRIPOS>BOND")
        atom_rows = [line for line in lines[atom_start:bond_start] if line.strip()]
        charges_by_atom_id = {
            "1": -0.800000,
            "2": 0.300000,
            "3": 0.500000,
        }
        rewritten = []
        for row in atom_rows:
            fields = row.split()
            fields[8] = f"{charges_by_atom_id[fields[0]]:.6f}"
            rewritten.append(" ".join(fields))
        lines[atom_start:bond_start] = [rewritten[0], rewritten[2], rewritten[1]]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": "am1bcc", "mode": "fixed", "geometry": "provider"},
        tmp_path,
    )

    assert result.charges.tolist() == pytest.approx([-0.8, 0.3, 0.5])
    np.testing.assert_allclose(result.provider_positions, atoms.get_positions())
    mapping = json.loads(Path(result.provenance["atom_mapping"]).read_text())
    assert [record["provider_index"] for record in mapping] == [0, 2, 1]


def test_ambertools_charge_output_atom_name_changes_fail_closed(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(
            input_path.read_text().replace("H1", "H000", 1),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="changed a MOL2 atom name"):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )


def test_ambertools_charge_output_changed_atom_id_fails(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        lines = input_path.read_text().splitlines()
        section = ""
        for index, line in enumerate(lines):
            if line.startswith("@<TRIPOS>"):
                section = line[9:]
                continue
            fields = line.split()
            if section == "ATOM" and len(fields) >= 2 and fields[0] == "2":
                fields[0] = "20"
                lines[index] = " ".join(fields)
            elif section == "BOND" and len(fields) >= 4 and fields[2] == "2":
                fields[2] = "20"
                lines[index] = " ".join(fields)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="changed, removed, or duplicated a MOL2 atom ID"):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )


def test_ambertools_charge_output_duplicate_bond_fails_closed(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        lines = input_path.read_text(encoding="utf-8").splitlines()
        molecule_index = lines.index("@<TRIPOS>MOLECULE")
        count_index = next(
            index
            for index in range(molecule_index + 2, len(lines))
            if lines[index].strip()
        )
        counts = lines[count_index].split()
        counts[1] = str(int(counts[1]) + 1)
        lines[count_index] = " ".join(counts)

        bond_index = lines.index("@<TRIPOS>BOND")
        bond_rows = []
        insertion_index = len(lines)
        for index in range(bond_index + 1, len(lines)):
            if lines[index].startswith("@<TRIPOS>"):
                insertion_index = index
                break
            if lines[index].strip():
                bond_rows.append(lines[index].split())
        duplicate = bond_rows[0]
        duplicate[0] = str(max(int(row[0]) for row in bond_rows) + 1)
        lines.insert(insertion_index, " ".join(duplicate))
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="Duplicate MOL2 bond between"):
        prepare_charges(
            atoms,
            {"source": "maple", "method": "am1bcc", "mode": "fixed"},
            tmp_path,
        )


def test_ambertools_relative_audit_directory_is_resolved_before_provider_run(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("-o") + 1])
        assert output_path.is_absolute()
        assert Path(kwargs["cwd"]).is_absolute()
        input_path = Path(command[command.index("-i") + 1])
        output_path.write_text(input_path.read_text(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": "am1bcc", "mode": "fixed"},
        Path("relative-audit"),
    )

    assert Path(result.provenance["output_mol2"]).is_file()


def test_ambertools_charge_serialization_rounding_is_repaired_and_audited(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        text = input_path.read_text()
        replacements = {
            "-0.834000000000": "-0.8340",
            "0.417000000000": "0.4170",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace("0.4170", "0.4169", 1)
        output_path.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": "am1bcc", "mode": "fixed", "geometry": "keep"},
        tmp_path,
    )

    assert result.charges.sum() == pytest.approx(0.0, abs=1.0e-12)
    assert result.provenance["provider_sum_e"] == pytest.approx(-0.0001)
    audit = json.loads(Path(result.provenance["charge_normalization"]).read_text())
    assert audit["strategy"] == "serialization-rounding-repair"
    assert audit["distribution"] == "uniform-all-atoms"
    assert audit["sqm_precharge_resolution_e"] == pytest.approx(0.001)
    assert audit[
        "maximum_accepted_mol2_charge_rounding_half_width_e"
    ] == pytest.approx(5.0e-5)
    assert audit["derived_rounding_bound_e"] == pytest.approx(0.00165)
    assert audit["correction_per_atom_e"] == pytest.approx(0.0001 / 3.0)
    assert result.provenance[
        "maximum_accepted_mol2_coordinate_rounding_half_width_angstrom"
    ] == pytest.approx(5.0e-5)


def test_ambertools_charge_residual_above_guard_fails(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        text = input_path.read_text()
        text = text.replace("-0.834000000000", "-0.8340")
        text = text.replace("0.417000000000", "0.4170")
        text = text.replace("0.4170", "0.4150", 1)
        output_path.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="serialization rounding bound"):
        prepare_charges(
            atoms,
            {
                "source": "maple",
                "method": "am1bcc",
                "mode": "fixed",
                "geometry": "keep",
            },
            tmp_path,
        )


def test_amber_pbsa_profile_stops_at_evidence_gate_before_charge_provider(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(charges.shutil, "which", lambda _name: None)
    with pytest.raises(NotImplementedError, match="optimized atom-type radii"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "maple", "method": "abcg2", "mode": "fixed"},
            {
                "method": "pb",
                "provider": "amber-pbsa",
                "profile": "abcg2-pbsa-2023",
                "nonpolar": "amber-pbsa",
                "experimental": True,
            },
            output=tmp_path / "job.out",
        )


def test_direct_api_rejects_unknown_charge_mode(tmp_path, water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    with pytest.raises(ValueError, match="Charge mode"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2", "mode": "responsive"},
            {"method": "gb", "model": "obc2", "experimental": True},
            output=tmp_path / "maple.out",
        )


def test_direct_api_rejects_unvalidated_chagb_charge_profile(tmp_path, water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    with pytest.raises(ValueError, match="requires fixed.*am1bcc"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2", "mode": "fixed", "geometry": "keep"},
            {
                "method": "gb",
                "provider": "ambertools",
                "model": "chagb",
                "profile": "chagb-bondi-pbsa-inp2",
                "nonpolar": "cavity-dispersion",
                "experimental": True,
            },
            output=tmp_path / "maple.out",
        )
