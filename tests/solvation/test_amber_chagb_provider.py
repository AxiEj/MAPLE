from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit import amber_chagb
from maple.function.calculator.extra_correction.implicit.amber_chagb import (
    AmberToolsChaGB,
    KCAL_PER_MOL_PER_HARTREE,
    parse_gbnsr6_components,
    parse_pbsa_components,
    render_typed_mol2,
    require_no_frcmod_nonbonded_overrides,
)
from maple.function.calculator.extra_correction.implicit.charges import ChargeResult
from maple.function.calculator.extra_correction.implicit import (
    correction as correction_module,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from source_compatibility import validate_frozen_source


def test_component_parsers_use_final_provider_blocks():
    assert parse_gbnsr6_components(
        "EGB = -1.0 ESURF = 2.0\nEGB = -6.5 ESURF = 1.5\n"
    ) == {"polar": -6.5, "surface_tension": 1.5}
    assert parse_pbsa_components(
        "ECAVITY = 1.0 EDISPER = -0.5\n" "ECAVITY = 16.8 EDISPER = -14.7\n"
    ) == {"cavity": 16.8, "dispersion": -14.7}


def test_component_parsers_and_nonbonded_contract_fail_closed():
    with pytest.raises(ValueError, match="EGB/ESURF"):
        parse_gbnsr6_components("Etot = -4")
    with pytest.raises(ValueError, match="ECAVITY/EDISPER"):
        parse_pbsa_components("ENPOLAR = 1")
    require_no_frcmod_nonbonded_overrides("MASS\n\nNONBON\n\n")
    with pytest.raises(ValueError, match="non-GAFF/GAFF2"):
        require_no_frcmod_nonbonded_overrides("MASS\n\nNONBON\nxx 1.8240 0.1700\n")


def test_mol2_renderer_preserves_types_and_bonds_but_replaces_state(water_mol2):
    lines = (
        water_mol2.read_text(encoding="utf-8")
        .replace("USER_CHARGES", "NO_CHARGES")
        .splitlines()
    )
    source_lines = []
    in_atoms = False
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            in_atoms = line == "@<TRIPOS>ATOM"
        if in_atoms and line.strip() and not line.startswith("@<TRIPOS>"):
            line = " ".join(line.split()[:8])
        source_lines.append(line)
    source = "\n".join(source_lines) + "\n"
    positions = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    rendered = render_typed_mol2(source, positions, [-0.8, 0.4, 0.4])

    assert "USER_CHARGES" in rendered
    assert "NO_CHARGES" not in rendered
    assert "O.3" in rendered
    assert rendered.split("@<TRIPOS>BOND", 1)[1] == source.split("@<TRIPOS>BOND", 1)[1]
    atom_rows = (
        rendered.split("@<TRIPOS>ATOM\n", 1)[1]
        .split("@<TRIPOS>BOND", 1)[0]
        .splitlines()
    )
    assert [float(value) for value in atom_rows[0].split()[2:5]] == [1.0, 2.0, 3.0]
    assert [float(row.split()[8]) for row in atom_rows] == [-0.8, 0.4, 0.4]


def _fake_ambertools_bundle(path: Path) -> Path:
    path.mkdir()
    for name in ("gbnsr6", "pbsa", "parmchk2", "tleap"):
        executable = path / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    return path / "gbnsr6"


def test_provider_composes_only_egb_cavity_dispersion_and_audits(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    executable = _fake_ambertools_bundle(tmp_path / "amber-bin")
    audit = tmp_path / "audit"

    def fake_run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        label = Path(command[0]).name
        if label == "parmchk2":
            (cwd / "molecule.frcmod").write_text(
                "MASS\n\nBOND\n\nNONBON\n\n", encoding="utf-8"
            )
        elif label == "tleap":
            (cwd / "molecule.prmtop").write_text("topology", encoding="utf-8")
            (cwd / "molecule.inpcrd").write_text("coordinates", encoding="utf-8")
            (cwd / "leap.log").write_text("ok", encoding="utf-8")
        elif label == "gbnsr6":
            (cwd / "gbnsr6.out").write_text(
                "EGB = -6.2823 ESURF = 1.8698\n", encoding="utf-8"
            )
        elif label == "pbsa":
            (cwd / "pbsa.out").write_text(
                "ECAVITY = 20.9485 EDISPER = -18.7755\n",
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(amber_chagb.subprocess, "run", fake_run)
    provider = AmberToolsChaGB(
        atoms,
        atoms.get_initial_charges(),
        executable=str(executable),
        audit_dir=audit,
    )
    result = provider.evaluate(atoms)

    expected_kcal = -6.2823 + 20.9485 - 18.7755
    assert result.energy_hartree == pytest.approx(
        expected_kcal / KCAL_PER_MOL_PER_HARTREE
    )
    assert result.energy_hartree == pytest.approx(
        result.components_hartree["polar"] + result.components_hartree["nonpolar"]
    )
    assert result.provenance["gas_phase_mm_energy_used"] is False
    assert result.provenance["bonded_mm_energy_used"] is False
    assert result.provenance["execution_control"]["gbnsr6_in_process_serialization"]
    assert result.provenance["reported_formula"] == "EGB + ECAVITY + EDISPER"
    citations = result.provenance["citations"]
    assert any("10.1021/ct4010917" in citation for citation in citations)
    assert any("10.1021/ct200786m" in citation for citation in citations)
    assert all("10.1021/acs.jctc.4c01471" not in citation for citation in citations)
    ledger = result.provenance["parameter_source_ledger"]
    assert set(ledger["polar"]) == set(result.provenance["polar_parameters"])
    assert set(ledger["nonpolar"]) == set(result.provenance["nonpolar_parameters"])
    assert (audit / "molecule.mol2").is_file()
    assert (audit / "amber-chagb.commands.json").is_file()
    recorded = json.loads((audit / "amber-chagb.result.json").read_text())
    assert recorded["energy_hartree"] == pytest.approx(result.energy_hartree)


def test_chagb_rejects_source_mol2_drift_before_resolving_tools(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    water_mol2.write_text(
        water_mol2.read_text(encoding="utf-8").replace("O.3", "O.2", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source MOL2 changed after MOL2Reader"):
        AmberToolsChaGB(atoms, atoms.get_initial_charges())


def test_composition_boundary_selects_chagb_provider(water_mol2, tmp_path, monkeypatch):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    executable = _fake_ambertools_bundle(tmp_path / "amber-bin")

    monkeypatch.setattr(
        correction_module,
        "prepare_charges",
        lambda target, _options, _audit: ChargeResult(
            charges=target.get_initial_charges(),
            method="am1bcc",
            provenance={"source": "test", "method": "am1bcc"},
        ),
    )

    def fake_run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        label = Path(command[0]).name
        if label == "parmchk2":
            (cwd / "molecule.frcmod").write_text("MASS\n\nNONBON\n\n", encoding="utf-8")
        elif label == "tleap":
            (cwd / "molecule.prmtop").write_text("topology", encoding="utf-8")
            (cwd / "molecule.inpcrd").write_text("coordinates", encoding="utf-8")
        elif label == "gbnsr6":
            (cwd / "gbnsr6.out").write_text(
                "EGB = -1.0 ESURF = 0.0\n", encoding="utf-8"
            )
        elif label == "pbsa":
            (cwd / "pbsa.out").write_text(
                "ECAVITY = 0.2 EDISPER = -0.1\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(amber_chagb.subprocess, "run", fake_run)
    correction = correction_module.ImplicitSolvationCorrection(
        atoms,
        {
            "source": "maple",
            "method": "am1bcc",
            "mode": "fixed",
            "geometry": "keep",
        },
        {
            "method": "gb",
            "provider": "ambertools",
            "model": "chagb",
            "profile": "chagb-bondi-pbsa-inp2",
            "nonpolar": "cavity-dispersion",
            "executable": str(executable),
            "experimental": True,
        },
        output=tmp_path / "job.out",
    )

    assert isinstance(correction.provider, AmberToolsChaGB)
    assert correction.supported_properties == {"energy"}
    assert correction.evaluate(atoms).energy_hartree == pytest.approx(
        -0.9 / KCAL_PER_MOL_PER_HARTREE
    )
    manifest = json.loads((tmp_path / "job.out.implicit" / "manifest.json").read_text())
    assert manifest["route"]["prohibited_terms"]["gas_phase_mm_energy"] is False
    assert manifest["solvation"]["model"] == "chagb"


def test_provider_rejects_forces_before_running_external_tools(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    executable = _fake_ambertools_bundle(tmp_path / "amber-bin")
    provider = AmberToolsChaGB(
        atoms,
        atoms.get_initial_charges(),
        executable=str(executable),
    )

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("energy-only force rejection must happen before execution")

    monkeypatch.setattr(amber_chagb.subprocess, "run", unexpected_run)
    with pytest.raises(NotImplementedError, match="SP-energy-only"):
        provider.evaluate(atoms, need_forces=True)


def test_provider_failure_retains_external_logs(water_mol2, tmp_path, monkeypatch):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    executable = _fake_ambertools_bundle(tmp_path / "amber-bin")
    audit = tmp_path / "audit"

    def fake_run(command, **kwargs):
        cwd = Path(kwargs["cwd"])
        label = Path(command[0]).name
        if label == "parmchk2":
            (cwd / "molecule.frcmod").write_text("MASS\n\nNONBON\n\n", encoding="utf-8")
        elif label == "tleap":
            (cwd / "molecule.prmtop").write_text("topology", encoding="utf-8")
            (cwd / "molecule.inpcrd").write_text("coordinates", encoding="utf-8")
        elif label == "gbnsr6":
            return subprocess.CompletedProcess(
                command, 7, stdout="provider stdout", stderr="provider failed"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(amber_chagb.subprocess, "run", fake_run)
    provider = AmberToolsChaGB(
        atoms,
        atoms.get_initial_charges(),
        executable=str(executable),
        audit_dir=audit,
    )

    with pytest.raises(RuntimeError, match="gbnsr6 exited with code 7"):
        provider.evaluate(atoms)
    assert (audit / "gbnsr6.stderr.log").read_text() == "provider failed"
    command = json.loads((audit / "gbnsr6.command.json").read_text())
    assert command["returncode"] == 7


def test_frozen_runtime_parity_artifact_locks_sp_only_boundary():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/implicit-solvation/benchmarks"
        / "route1-chagb-runtime-provider-parity-2026-07-25.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    recorded = artifact.pop("content_sha256")
    payload = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert hashlib.sha256(payload).hexdigest() == recorded
    assert artifact["recorded_date"] == "2026-07-25"
    assert "SP-only" in artifact["claim_scope"]
    assert "not confirmation" in artifact["claim_scope"]
    topology = artifact["topology_contract_audit"]
    assert topology["case_count"] == 526
    assert topology["input_vs_antechamber_gaff2_atom_types_different"] == 95
    assert artifact["live_provider_parity"]["all_components_within_tolerance"]
    assert artifact["runtime_contract"]["supported_tasks"] == ["sp"]
    assert artifact["runtime_contract"]["gas_phase_mm_energy_used"] is False
    assert artifact["runtime_contract"]["hydration_label_fit_or_residual"] is False
    assert "POSIX" in artifact["runtime_contract"]["gbnsr6_execution_control"]
    smoke = artifact["mlip_composition_smoke"]
    assert smoke["model"] == "maceoff23m"
    assert smoke["closure_hartree"] == 0.0
    assert len(smoke["checkpoint_sha256"]) == 64
    implementation_update = artifact["runtime_implementation_update"]
    assert implementation_update["ambertools_provider_source_changed"] is False
    platform_audit = implementation_update["openmm_platform_audit"]
    assert platform_audit["all_checks_pass"] is True
    platform_audit_path = path.parent / platform_audit["path"]
    platform_artifact = json.loads(platform_audit_path.read_text(encoding="utf-8"))
    assert platform_artifact["content_sha256"] == platform_audit["content_sha256"]
    repository = Path(__file__).resolve().parents[2]
    historical_implementation = artifact["runtime_implementation"]
    assert set(historical_implementation) == {
        "maple/function/calculator/extra_correction/implicit/amber_chagb.py",
        "maple/function/calculator/extra_correction/implicit/correction.py",
        "maple/function/read/command_control.py",
    }
    assert all(len(value) == 64 for value in historical_implementation.values())
    frequency_update = artifact["frequency_contract_compatibility_update"]
    assert frequency_update["historical_runtime_hashes_preserved"] is True
    assert frequency_update["ambertools_provider_source_changed"] is False
    assert frequency_update["provider_formula_or_component_changed"] is False
    assert "OpenMM GB" in frequency_update["reason"]
    assert "SP-only" in frequency_update["reason"]
    assert all(
        len(value) == 64
        for value in frequency_update["current_runtime_implementation"].values()
    )
    identity_update = artifact["frozen_identity_contract_update"]
    assert identity_update["historical_runtime_hashes_preserved"] is True
    assert identity_update["ambertools_provider_source_changed"] is False
    assert identity_update["provider_formula_or_component_changed"] is False
    assert identity_update["fixed_charge_identity_validation_added"] is True
    assert identity_update["same_element_reordering_rejected"] is True
    assert identity_update["prebuilt_endpoint_gate_tightened"] is True
    assert "charges, radii, and topology" in identity_update["reason"]
    assert "OpenMM GB" in identity_update["reason"]
    assert "ACE/LCPO" in identity_update["reason"]
    for relative, expected in identity_update["current_runtime_implementation"].items():
        validation = validate_frozen_source(repository, relative, expected)
        assert validation["mode"] in {
            "exact-historical-freeze",
            "documented-postexecution-production-safety-change",
        }
