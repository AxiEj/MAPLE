from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maple.function.benchmarking import (
    AccuracyAdapterArtifactReceipt,
    AccuracyAdapterRegistration,
    BenchmarkQuantity,
    MolecularInputReceipt,
)
from maple.function.benchmarking import atomicese_accuracy
from maple.function.benchmarking import c3net_accuracy, pretrained_hub


def _xyz(path: Path) -> Path:
    path.write_text(
        "C 0.000000 0.000000 0.000000\n"
        "H 0.629118 0.629118 0.629118\n"
        "H -0.629118 -0.629118 0.629118\n"
        "H -0.629118 0.629118 -0.629118\n"
        "H 0.629118 -0.629118 -0.629118\n",
        encoding="utf-8",
    )
    return path


def _context(xyz_path: Path, binary: Path, *, solvent: str = "methanol"):
    receipt = MolecularInputReceipt.from_file(
        xyz_path,
        molecular_input_format="headerless_xyz",
    )
    return SimpleNamespace(
        target_quantity=BenchmarkQuantity.PROPERTY_PREDICTION,
        molecular_input=SimpleNamespace(
            molecular_input_format="headerless_xyz",
            solute_structure_identifier=receipt.solute_structure_identifier,
            read=receipt.read_verified_payload,
        ),
        solvent_components=(solvent,),
        solvent_mole_fractions=(1.0,),
        implementation_artifact_paths={
            "adapter_code": str(Path(atomicese_accuracy.__file__)),
            "binary": str(binary),
            "release_audit": str(atomicese_accuracy.ATOMICESE_RELEASE_AUDIT_PATH),
            "runner_code": str(Path(atomicese_accuracy.__file__)),
            "runtime_lock": str(atomicese_accuracy.ATOMICESE_FORMAL_RUNTIME_LOCK_PATH),
        },
    )


def _fake_binary(path: Path, *, diagnostic: bool = False) -> Path:
    extra = "printf 'diagnostic\\n' >&2\n" if diagnostic else ""
    path.write_text(
        "#!/bin/sh\n"
        f"{extra}"
        'printf \'Solvent name "%s" was read from the command line\\n\' "$3"\n'
        "printf 'Number of atoms in the solute:    5\\n'\n"
        "printf 'Total solvation free energy = -1.25 kcal/mol\\n'\n"
        "printf 'CPU time = 0.000 seconds.\\n'\n",
        encoding="utf-8",
    )
    return path


def test_headerless_xyz_receipt_is_strict_and_reverifies_graph(tmp_path):
    path = _xyz(tmp_path / "methane.xyz")
    receipt = MolecularInputReceipt.from_file(
        path,
        molecular_input_format="headerless_xyz",
    )

    assert receipt.solute_structure_identifier == "SMILES:C"
    assert receipt.read_verified_payload() == path.read_bytes()

    path.write_text("5\ncomment\n" + path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="source bytes changed"):
        receipt.read_verified_payload()
    with pytest.raises(ValueError, match="four whitespace-separated fields"):
        MolecularInputReceipt.from_file(
            path,
            molecular_input_format="headerless_xyz",
        )

    malformed = tmp_path / "malformed.xyz"
    malformed.write_text("C nan 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        MolecularInputReceipt.from_file(
            malformed,
            molecular_input_format="headerless_xyz",
        )


@pytest.mark.parametrize(
    ("supplied", "expected"),
    (
        ("methanol", "methanol"),
        ("ethanol", "ethanol"),
        ("octanol", "1-octanol"),
        ("hexadecane", "n-hexadecane"),
        ("hexane", "n-hexane"),
        ("dmf", "n,n-dimethylformamide"),
    ),
)
def test_atomicese_aliases_execute_only_a_temporary_binary_copy(
    monkeypatch,
    tmp_path,
    supplied,
    expected,
):
    binary = _fake_binary(tmp_path / "AtomicESE.x")
    original_mode = binary.stat().st_mode
    monkeypatch.setattr(
        atomicese_accuracy,
        "ATOMICESE_LINUX_BINARY_SHA256",
        hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    observed = []
    real_run = atomicese_accuracy.subprocess.run

    def recording_run(command, **kwargs):
        observed.append(tuple(command))
        return real_run(command, **kwargs)

    monkeypatch.setattr(atomicese_accuracy.subprocess, "run", recording_run)
    result = atomicese_accuracy.AtomicESEFormalAccuracyAdapter().predict(
        context=_context(_xyz(tmp_path / "methane.xyz"), binary, solvent=supplied)
    )

    assert result == -1.25
    assert observed[0][2:] == ("-solvent", expected)
    assert Path(observed[0][0]) != binary
    assert not Path(observed[0][0]).exists()
    assert binary.stat().st_mode == original_mode


def test_atomicese_rejects_water_mixtures_and_exit_zero_diagnostics(
    monkeypatch,
    tmp_path,
):
    binary = _fake_binary(tmp_path / "AtomicESE.x")
    monkeypatch.setattr(
        atomicese_accuracy,
        "ATOMICESE_LINUX_BINARY_SHA256",
        hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    adapter = atomicese_accuracy.AtomicESEFormalAccuracyAdapter()
    context = _context(_xyz(tmp_path / "methane.xyz"), binary, solvent="water")
    with pytest.raises(ValueError, match="no frozen alias"):
        adapter.predict(context=context)

    context.solvent_components = ("methanol", "ethanol")
    context.solvent_mole_fractions = (0.5, 0.5)
    with pytest.raises(ValueError, match="exactly one pure solvent"):
        adapter.predict(context=context)

    diagnostic = _fake_binary(tmp_path / "diagnostic.x", diagnostic=True)
    monkeypatch.setattr(
        atomicese_accuracy,
        "ATOMICESE_LINUX_BINARY_SHA256",
        hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
    )
    context = _context(
        _xyz(tmp_path / "methane-diagnostic.xyz"),
        diagnostic,
    )
    with pytest.raises(RuntimeError, match="stderr"):
        adapter.predict(context=context)


def test_atomicese_runtime_lock_and_adapter_specific_roles_are_frozen(tmp_path):
    runtime_lock = atomicese_accuracy.ATOMICESE_FORMAL_RUNTIME_LOCK_PATH
    payload = json.loads(runtime_lock.read_text(encoding="utf-8"))
    assert hashlib.sha256(runtime_lock.read_bytes()).hexdigest() == (
        atomicese_accuracy.ATOMICESE_FORMAL_RUNTIME_LOCK_SHA256
    )
    assert payload["source_revision"] == atomicese_accuracy.ATOMICESE_SOURCE_REVISION
    assert payload["source_tree"] == atomicese_accuracy.ATOMICESE_SOURCE_TREE
    assert payload["binary_sha256"] == (
        atomicese_accuracy.ATOMICESE_LINUX_BINARY_SHA256
    )
    assert payload["release_audit_sha256"] == (
        atomicese_accuracy.ATOMICESE_RELEASE_AUDIT_SHA256
    )
    assert payload["wall_times_recorded"] is False
    assert payload["solvent_aliases"]["ethanol"] == "ethanol"

    adapter = atomicese_accuracy.AtomicESEFormalAccuracyAdapter()
    binary = tmp_path / "binary"
    binary.write_bytes(b"binary")
    artifacts = (
        AccuracyAdapterArtifactReceipt.from_file(
            role="adapter_code",
            path=Path(atomicese_accuracy.__file__),
        ),
        AccuracyAdapterArtifactReceipt.from_file(role="binary", path=binary),
        AccuracyAdapterArtifactReceipt.from_file(
            role="release_audit",
            path=atomicese_accuracy.ATOMICESE_RELEASE_AUDIT_PATH,
        ),
        AccuracyAdapterArtifactReceipt.from_file(
            role="runner_code",
            path=Path(
                __import__(
                    "maple.function.benchmarking.pretrained_hub",
                    fromlist=["dummy"],
                ).__file__
            ),
        ),
        AccuracyAdapterArtifactReceipt.from_file(
            role="runtime_lock",
            path=runtime_lock,
        ),
    )
    registration = AccuracyAdapterRegistration.from_components(
        adapter_id="atomicese-test",
        adapter=adapter,
        implementation_artifacts=artifacts,
        configuration={},
    )
    assert registration.adapter_id == "atomicese-test"


def test_formal_registry_configures_c3net_and_atomicese_independently(monkeypatch):
    monkeypatch.delenv("MAPLE_C3NET_FORMAL_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("MAPLE_C3NET_FORMAL_SOURCE_BUNDLE", raising=False)
    monkeypatch.setenv("MAPLE_ATOMICESE_FORMAL_SOURCE_ROOT", "/atomicese")
    atomicese_registration = SimpleNamespace(adapter_id="atomicese-formal-test")
    monkeypatch.setattr(
        atomicese_accuracy,
        "build_atomicese_formal_accuracy_registration",
        lambda **kwargs: atomicese_registration,
    )
    assert pretrained_hub._configured_formal_accuracy_adapters() == {
        "atomicese-formal-test": atomicese_registration
    }

    monkeypatch.delenv("MAPLE_ATOMICESE_FORMAL_SOURCE_ROOT")
    monkeypatch.setenv("MAPLE_C3NET_FORMAL_SOURCE_ROOT", "/c3net")
    monkeypatch.setenv("MAPLE_C3NET_FORMAL_SOURCE_BUNDLE", "/c3net.bundle")
    c3net_registration = SimpleNamespace(adapter_id="c3net-formal-test")
    monkeypatch.setattr(
        c3net_accuracy,
        "build_c3net_formal_accuracy_registration",
        lambda **kwargs: c3net_registration,
    )
    assert pretrained_hub._configured_formal_accuracy_adapters() == {
        "c3net-formal-test": c3net_registration
    }

    monkeypatch.delenv("MAPLE_C3NET_FORMAL_SOURCE_BUNDLE")
    with pytest.raises(RuntimeError, match="requires both"):
        pretrained_hub._configured_formal_accuracy_adapters()
