from __future__ import annotations

import json
from pathlib import Path
import subprocess

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
    assert Path(result.provenance["atom_mapping"]).is_file()
    assert result.charges.tolist() == atoms.get_initial_charges().tolist()


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


def test_ambertools_charge_rounding_residual_is_corrected_and_audited(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        charges.shutil, "which", lambda _name: "/opt/amber/bin/antechamber"
    )

    def fake_run(command, **_kwargs):
        input_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[command.index("-o") + 1])
        text = input_path.read_text().replace("-0.834000", "-0.836000")
        output_path.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    result = prepare_charges(
        atoms,
        {"source": "maple", "method": "am1bcc", "mode": "fixed", "geometry": "keep"},
        tmp_path,
    )

    assert result.charges.sum() == pytest.approx(0.0, abs=1.0e-12)
    assert result.provenance["provider_sum_e"] == pytest.approx(-0.002)
    audit = json.loads(Path(result.provenance["charge_normalization"]).read_text())
    assert audit["strategy"] == "uniform-all-atoms"
    assert audit["correction_per_atom_e"] == pytest.approx(0.002 / 3.0)


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
        text = input_path.read_text().replace("-0.834000", "-0.934000")
        output_path.write_text(text, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(charges.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="residual is too large"):
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
