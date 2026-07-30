from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / "docs/pretrained-solvation-hub/run_atomicese_flexisol_formal_accuracy.py"
)
RESULT_ROOT = (
    ROOT / "docs/pretrained-solvation-hub/benchmarks/"
    "atomicese-flexisol-formal-2026-07-31"
)
RESULT_PATH = RESULT_ROOT / "result.json"
REPRODUCIBILITY_PATH = (
    ROOT / "docs/pretrained-solvation-hub/benchmarks/"
    "atomicese-flexisol-formal-2026-07-31-reproducibility.json"
)
FROZEN_ARTIFACTS_AVAILABLE = RESULT_PATH.is_file() and REPRODUCIBILITY_PATH.is_file()


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "run_atomicese_flexisol_formal_accuracy",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_selection_identity_is_output_blind():
    module = _load_script()
    row = {
        "FlexiSol Name": "example",
        "Solvent": "ethanol",
        "Ref.": "10.example/reference",
        r"Value (\kcalpmole)": "-5.0",
    }
    changed_label = dict(row)
    changed_label[r"Value (\kcalpmole)"] = "999.0"

    assert module._candidate_identity(row, "CCO") == module._candidate_identity(
        changed_label,
        "CCO",
    )
    assert r"Value (\kcalpmole)" not in module.SELECTION_IDENTITY_FIELDS


def test_frozen_eligibility_inventory_includes_verified_ethanol_only():
    module = _load_script()
    assert module.ELIGIBLE_SOLVENT_ALIASES == {
        "methanol": "methanol",
        "octanol": "octanol",
        "hexadecane": "hexadecane",
        "hexane": "hexane",
        "dmf": "dmf",
        "ethanol": "ethanol",
    }
    assert "water" not in module.ELIGIBLE_SOLVENT_ALIASES
    assert module.EXPECTED_ELIGIBLE_ROW_COUNT == 76
    assert module.EXPECTED_ELIGIBLE_SOLVENT_COUNTS == {
        "ethanol": 1,
        "hexadecane": 13,
        "hexane": 1,
        "octanol": 61,
    }
    assert len(module.EXPECTED_PRIMARY_GROUPS) == 15


def test_mmff94_geometry_is_deterministic_headerless_and_fixed_precision(tmp_path):
    module = _load_script()
    record = {
        "record_id": "record",
        "canonical_smiles": "CCO",
        "selection_hash": hashlib.sha256(b"record").hexdigest(),
    }
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"

    first_metadata = module._write_mmff94_xyz(record, destination=first)
    second_metadata = module._write_mmff94_xyz(record, destination=second)

    assert first.read_bytes() == second.read_bytes()
    lines = first.read_text(encoding="ascii").splitlines()
    molecule = Chem.AddHs(Chem.MolFromSmiles(record["canonical_smiles"]))
    assert len(lines) == molecule.GetNumAtoms()
    coordinate = re.compile(r"^[A-Z][a-z]? -?\d+\.\d{10} -?\d+\.\d{10} -?\d+\.\d{10}$")
    assert all(coordinate.fullmatch(line) for line in lines)
    assert first_metadata == second_metadata
    assert first_metadata["force_field"] == "MMFF94"
    assert first_metadata["conformer_count"] == 1
    assert first_metadata["headerless_xyz"] is True
    assert first_metadata["coordinate_decimal_places"] == 10
    assert first_metadata["purpose"] == "MAPLE development geometry"
    assert first_metadata["author_manual_correction_reproduction"] is False


def test_mmff94_geometry_fails_closed_on_missing_parameters(monkeypatch, tmp_path):
    module = _load_script()
    record = {
        "record_id": "record",
        "canonical_smiles": "CCO",
        "selection_hash": hashlib.sha256(b"record").hexdigest(),
    }
    monkeypatch.setattr(module.AllChem, "MMFFHasAllMoleculeParams", lambda _mol: False)
    with pytest.raises(ValueError, match="MMFF94 parameters"):
        module._write_mmff94_xyz(record, destination=tmp_path / "record.xyz")


def test_run_rejects_reusing_an_output_directory_before_external_setup(tmp_path):
    module = _load_script()
    output_root = tmp_path / "existing-output"
    output_root.mkdir()
    sentinel = output_root / "do-not-overwrite.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="[Oo]utput directory.*exist"):
        module.run(
            flexisol_source_root=tmp_path / "unused-flexisol-source",
            atomicese_source_root=tmp_path / "unused-atomicese-source",
            output_directory=output_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert tuple(output_root.iterdir()) == (sentinel,)


@pytest.mark.skipif(
    not FROZEN_ARTIFACTS_AVAILABLE,
    reason="Production AtomicESE/FlexiSol panel was intentionally not executed.",
)
def test_frozen_result_is_complete_and_applies_strict_accuracy_gate():
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    accuracy = payload["accuracy"]
    errors = accuracy["per_record_errors"]

    assert payload["scientific_status"] == (
        "development_accuracy_only_overlap_unknown_not_final_holdout"
    )
    assert payload["leakage_audit"]["status"] == "overlap_unknown"
    assert accuracy["record_count"] == 15
    assert accuracy["evaluated_count"] == 15
    assert len(errors) == 15
    assert len({row["primary_functional_group"] for row in errors}) == 15
    assert all(
        {
            "experimental_kcal_mol",
            "predicted_kcal_mol",
            "signed_error_kcal_mol",
            "absolute_error_kcal_mol",
        }.issubset(row)
        for row in errors
    )
    maximum = max(row["absolute_error_kcal_mol"] for row in errors)
    assert accuracy["metrics"]["maximum_absolute_error_kcal_mol"] == pytest.approx(
        maximum
    )
    assert payload["accuracy_gate"]["passed"] is (maximum < 1.5)
    expected_status = (
        "pending_future_matched_qm_benchmark"
        if maximum < 1.5
        else "not_eligible_due_to_accuracy_failure"
    )
    assert payload["matched_qm_gate"]["status"] == expected_status
    assert "timing" not in json.dumps(payload).casefold()


@pytest.mark.skipif(
    not FROZEN_ARTIFACTS_AVAILABLE,
    reason="Production AtomicESE/FlexiSol panel was intentionally not executed.",
)
def test_frozen_inputs_are_exact_headerless_xyz_and_no_binary_is_published():
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    records = payload["selection"]["records"]
    assert len(records) == 15
    assert payload["selection"]["eligible_row_count"] == 76
    assert payload["selection"]["eligible_solvent_counts"] == {
        "ethanol": 1,
        "hexadecane": 13,
        "hexane": 1,
        "octanol": 61,
    }
    for record in records:
        path = RESULT_ROOT / "inputs" / f"{record['record_id']}.xyz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            record["molecular_input_sha256"]
        )
        assert len(path.read_text(encoding="ascii").splitlines()) == (
            record["geometry_generation"]["atom_count"]
        )
    assert not any(
        path.name in {"AtomicESE.x", "AtomicESE.exe"} for path in RESULT_ROOT.rglob("*")
    )


@pytest.mark.skipif(
    not FROZEN_ARTIFACTS_AVAILABLE,
    reason="Production AtomicESE/FlexiSol panel was intentionally not executed.",
)
def test_reproducibility_receipt_binds_the_complete_canonical_tree():
    receipt = json.loads(REPRODUCIBILITY_PATH.read_text(encoding="utf-8"))
    manifest = {
        path.relative_to(RESULT_ROOT)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(RESULT_ROOT.rglob("*"))
        if path.is_file()
    }
    manifest_digest = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    assert receipt["run_count"] == 2
    assert receipt["byte_identical"] is True
    assert receipt["canonical_tree_manifest"] == manifest
    assert receipt["canonical_tree_sha256"] == manifest_digest
    assert receipt["excluded_nondeterministic_fields"] == []
