from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "pretrained-solvation-hub" / "run_flexisol_release_audit.py"
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "flexisol-release-audit-2026-07-30.json"
)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("flexisol_release_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _dgsolv_csv(rows: int = 530) -> bytes:
    lines = [",".join(AUDIT.DGSOLV_HEADER)]
    solvents = (
        "water",
        "octanol",
        "hexadecane",
        "methanol",
        "ethanol",
        "dmf",
        "hexane",
    )
    for index in range(rows):
        solvent = solvents[index % len(solvents)]
        lines.append(
            f"name-{index},iupac-{index},{solvent},SMILES-{index},"
            f"{-index / 10:.1f},10.1000/source-{index % 11}"
        )
    return ("\n".join(lines) + "\n").encode()


def _logkab_csv(rows: int = 294) -> bytes:
    lines = [",".join(AUDIT.LOGKAB_HEADER)]
    solvent_pairs = (
        '"octanol,water"',
        '"toluene,water"',
        '"ethanol,water"',
        '"chloroform,water"',
        '"water,methanol"',
        '"benzene,water"',
    )
    for index in range(rows):
        pair = solvent_pairs[index % len(solvent_pairs)]
        lines.append(
            f"name-{index},iupac-{index},{pair},SMILES-{index},"
            f"{index / 100:.2f},10.1000/source-{index % 9}"
        )
    return ("\n".join(lines) + "\n").encode()


def _registry() -> bytes:
    return json.dumps(
        {
            "methods": {
                "alpb": {"method": "alpb", "type": "solv"},
                "directml": {"method": "directml", "type": "solv"},
                "cigin": {"method": "cigin", "type": "solv"},
                "el_gfn2": {"method": "el_gfn2", "type": "el"},
            }
        }
    ).encode()


def test_reference_parser_rejects_header_drift_and_nonfinite_values():
    with pytest.raises(ValueError, match="header"):
        AUDIT._read_reference_csv(
            b"wrong,header\n",
            label="synthetic",
            expected_header=AUDIT.DGSOLV_HEADER,
            solvent_column="Solvent",
            value_column=r"Value (\kcalpmole)",
            expected_rows=530,
        )

    data = _dgsolv_csv().replace(b"-0.1", b"nan", 1)
    with pytest.raises(ValueError, match="Non-finite"):
        AUDIT._read_reference_csv(
            data,
            label="synthetic",
            expected_header=AUDIT.DGSOLV_HEADER,
            solvent_column="Solvent",
            value_column=r"Value (\kcalpmole)",
            expected_rows=530,
        )


def test_reference_summary_counts_coverage_without_redistributing_rows():
    rows = AUDIT._read_reference_csv(
        _dgsolv_csv(),
        label="synthetic",
        expected_header=AUDIT.DGSOLV_HEADER,
        solvent_column="Solvent",
        value_column=r"Value (\kcalpmole)",
        expected_rows=530,
    )

    summary = AUDIT._reference_summary(rows, solvent_column="Solvent")

    assert summary["record_count"] == 530
    assert summary["unique_solvent_or_pair_count"] == 7
    assert summary["source_doi_count"] == 11
    assert summary["records_embedded"] is False
    assert "records" not in summary


def test_registry_audit_distinguishes_solvation_from_electronic_methods():
    summary = AUDIT._registry_summary(_registry())

    assert summary["registered_solvation_methods"] == [
        "alpb",
        "cigin",
        "directml",
    ]
    assert summary["registered_electronic_methods"] == ["el_gfn2"]
    assert summary["route4_models_already_registered"] == ["cigin", "directml"]


def test_revision_gate_fails_closed(monkeypatch, tmp_path):
    answers = iter(
        (
            b"0000000000000000000000000000000000000000\n",
            f"{AUDIT.SOURCE_TREE}\n".encode(),
        )
    )
    monkeypatch.setattr(AUDIT, "_git_bytes", lambda *args: next(answers))

    with pytest.raises(ValueError, match="revision"):
        AUDIT._verify_source_identity(tmp_path)


def test_tree_gate_fails_closed(monkeypatch, tmp_path):
    answers = iter(
        (
            f"{AUDIT.SOURCE_REVISION}\n".encode(),
            b"0000000000000000000000000000000000000000\n",
        )
    )
    monkeypatch.setattr(AUDIT, "_git_bytes", lambda *args: next(answers))

    with pytest.raises(ValueError, match="tree"):
        AUDIT._verify_source_identity(tmp_path)


def test_blob_gate_rejects_wrong_size_and_same_size_wrong_hash(monkeypatch, tmp_path):
    expected = AUDIT.PINNED_BLOBS[AUDIT.LICENSE_PATH]
    monkeypatch.setattr(AUDIT, "_git_bytes", lambda *args: b"wrong")
    with pytest.raises(ValueError, match="size"):
        AUDIT._read_verified_blob(tmp_path, AUDIT.LICENSE_PATH)

    monkeypatch.setattr(
        AUDIT,
        "_git_bytes",
        lambda *args: b"x" * expected["size_bytes"],
    )
    with pytest.raises(ValueError, match="SHA256"):
        AUDIT._read_verified_blob(tmp_path, AUDIT.LICENSE_PATH)


def test_blob_reads_are_bound_to_the_immutable_source_revision(monkeypatch, tmp_path):
    data = b"exact synthetic blob"
    monkeypatch.setitem(
        AUDIT.PINNED_BLOBS,
        AUDIT.LICENSE_PATH,
        {
            "size_bytes": len(data),
            "sha256": AUDIT._sha256_bytes(data),
        },
    )
    calls = []

    def fake_git_bytes(root, *args):
        calls.append(args)
        return data

    monkeypatch.setattr(AUDIT, "_git_bytes", fake_git_bytes)

    assert AUDIT._read_verified_blob(tmp_path, AUDIT.LICENSE_PATH) == data
    assert calls == [("show", f"{AUDIT.SOURCE_REVISION}:{AUDIT.LICENSE_PATH}")]


def test_audit_payload_is_deterministic_and_cannot_admit_a_final_holdout(
    monkeypatch, tmp_path
):
    blobs = {
        AUDIT.DGSOLV_PATH: _dgsolv_csv(),
        AUDIT.LOGKAB_PATH: _logkab_csv(),
        AUDIT.REGISTRY_PATH: _registry(),
        AUDIT.README_PATH: b"Correction\nhexadecane\n10.1039/D5SC06406F\n",
        AUDIT.LICENSE_PATH: b"MIT License\n",
    }
    monkeypatch.setattr(
        AUDIT,
        "_verify_source_identity",
        lambda root: {
            "revision": AUDIT.SOURCE_REVISION,
            "tree": AUDIT.SOURCE_TREE,
        },
    )
    monkeypatch.setattr(
        AUDIT,
        "_read_verified_blob",
        lambda root, path: blobs[path],
    )

    first = AUDIT.audit_release(tmp_path)
    second = AUDIT.audit_release(tmp_path)

    assert first == second
    assert first["acceptance_eligible"] is False
    assert first["candidate_external_confirmation"] is True
    assert first["strict_final_holdout"] is False
    assert first["validation_scope"]["models_scored"] == []
    assert first["validation_scope"]["model_predictions_read"] is False
    assert first["validation_scope"]["maximum_error_computed"] is False
    assert first["panels"]["dgsolv"]["unique_solvent_or_pair_count"] == 7
    assert first["coverage_gate"]["minimum_required_pure_solvents"] == 10
    assert first["coverage_gate"]["passes"] is False
    assert "records" not in first


def test_frozen_release_artifact_preserves_static_only_boundary():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["identity"]["revision"] == AUDIT.SOURCE_REVISION
    assert payload["identity"]["tree"] == AUDIT.SOURCE_TREE
    assert payload["identity"]["files"][AUDIT.DGSOLV_PATH] == {
        "sha256": AUDIT.PINNED_BLOBS[AUDIT.DGSOLV_PATH]["sha256"],
        "size_bytes": AUDIT.PINNED_BLOBS[AUDIT.DGSOLV_PATH]["size_bytes"],
        "verified": True,
    }
    assert payload["panels"]["dgsolv"]["record_count"] == 530
    assert payload["panels"]["dgsolv"]["solvent_or_pair_counts"] == {
        "dmf": 1,
        "ethanol": 5,
        "hexadecane": 24,
        "hexane": 1,
        "methanol": 5,
        "octanol": 197,
        "water": 297,
    }
    assert payload["panels"]["logkab"]["record_count"] == 294
    assert payload["registry"]["route4_models_already_registered"] == [
        "cigin",
        "directml",
    ]
    assert payload["acceptance_eligible"] is False
    assert payload["strict_final_holdout"] is False
    assert payload["validation_scope"]["model_predictions_read"] is False
    assert payload["validation_scope"]["models_scored"] == []
    assert payload["gpu_acceleration"]["relevant_to_static_audit"] is False
    assert "records" not in payload
