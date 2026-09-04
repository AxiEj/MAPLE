from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
RUNNER = BENCHMARKS / "run_mace_ef_cosmors_freesolv20.py"


def _json(name: str) -> dict[str, object]:
    return json.loads((BENCHMARKS / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_mace_ef_cosmors_freesolv20",
        RUNNER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exploratory_v1_failure_artifact_is_preserved_byte_for_byte():
    path = BENCHMARKS / "mace-ef-cosmors-freesolv20-diverse-v1.json"
    artifact = _json(path.name)

    assert _sha256(path) == (
        "40205791f7d81d058ff8bb7ce0c108e52a31fe4cd5b918b38a9c321404166a63"
    )
    assert artifact["status"] == "complete-with-failures"
    assert artifact["records"] == []
    assert len(artifact["failures"]) == 20
    assert {failure["error_type"] for failure in artifact["failures"]} == {
        "RuntimeError",
        "ValueError",
    }


def test_selection_v2_changes_only_the_documented_runtime_domain_failures():
    v1 = _json("mace-ef-cosmors-freesolv20-selection-v1.json")
    v2 = _json("mace-ef-cosmors-freesolv20-selection-v2.json")
    assert (
        _sha256(BENCHMARKS / "mace-ef-cosmors-freesolv20-selection-v1.json")
        == "8133478621c535c21fc5c7e5e2f7509260bac5ea3239e2640996dc9253df335d"
    )
    assert (
        _sha256(BENCHMARKS / "mace-ef-cosmors-freesolv20-selection-v2.json")
        == "df0fdb715b48d4d9cee399590385d1b23247112fd7ab57fb5d56d4b572c28c91"
    )
    changed_ordinals = [
        index + 1
        for index, (left, right) in enumerate(
            zip(v1["records"], v2["records"], strict=True)
        )
        if left != right
    ]
    assert changed_ordinals == [1, 19]
    assert v2["selection_policy"]["candidate_predictions_used"] is False
    assert (
        v2["selection_policy"]["experimental_values_used_to_rank_candidates"] is False
    )
    assert "methane" in v2["selection_revision_reason"]
    assert "phosphorus" in v2["selection_revision_reason"]


def test_selection_v3_is_a_byte_bound_record_identical_clean_replay():
    v2_path = BENCHMARKS / "mace-ef-cosmors-freesolv20-selection-v2.json"
    v2 = _json(v2_path.name)
    v3 = _json("mace-ef-cosmors-freesolv20-selection-v3.json")

    assert v3["status"] == "frozen-before-clean-replay"
    assert v3["records"] == v2["records"]
    assert v3["source_selection"] == {
        "record_changes": [],
        "selection_id": v2["selection_id"],
        "sha256": _sha256(v2_path),
    }
    manifest = v3["prepared_manifest"]
    assert manifest["sha256"] == (
        "e82f328d8b6a989dfc6367dc7b0c3019bbb09c0cb5929429202732e201d53bd7"
    )
    assert manifest["selected_record_manifest_sha256"] == (
        "c70e4f7feaeb46736ff21579a6bf6755a88a17fbb019024ccfc4fa4f8a415d94"
    )


def test_exploratory_v2_and_derived_results_are_frozen_as_unbound_evidence():
    expected = {
        "mace-ef-cosmors-freesolv20-diverse-v2.json": (
            "d4424d3a9fee178f553df7a2eb2615d2152eb343bd331b8f2c61ca16349a9e73"
        ),
        "mace-ef-cosmors-freesolv20-component-ablation-v1.json": (
            "deae9f0fff99e44e43cfae1109528979e410d6639fc25df47db7833e22422069"
        ),
        "mace-ef-cosmors-freesolv20-frozen-source-ablation-v1.json": (
            "22368e899009465c0834d89448a06a686672415e59ff56c1c66b4a6a219b4633"
        ),
        "mace-ef-cosmors-hbond-tail-audit-v1.json": (
            "90373c933e351854b556df5898cd1f76b713e38e88af1d78e18e280d2d1a4ece"
        ),
    }
    for filename, digest in expected.items():
        artifact = _json(filename)
        assert _sha256(BENCHMARKS / filename) == digest
        assert "execution_git_head" not in artifact
        assert "source_files_sha256" not in artifact

    primary = _json("mace-ef-cosmors-freesolv20-diverse-v2.json")
    assert primary["status"] == "complete"
    assert primary["summary"]["mae_kcal_mol"] == pytest.approx(9.191693016044956)
    assert primary["summary"]["maximum_absolute_error_kcal_mol"] == pytest.approx(
        52.25019974116495
    )
    assert len(primary["records"]) == 20

    tail = _json("mace-ef-cosmors-hbond-tail-audit-v1.json")
    assert len(tail["records"]) == 4
    assert {record["compound_id"] for record in tail["records"]} == {
        "mobley_1636752",
        "mobley_3034976",
        "mobley_20524",
        "mobley_4639255",
    }


def test_prepared_manifest_validation_fails_closed_on_byte_drift(
    runner: ModuleType,
    tmp_path: Path,
):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    artifact_hashes = {}
    for filename in ("database.json", "database.txt", "mol2files_gaff.tar.gz"):
        path = dataset_dir / filename
        path.write_bytes(f"frozen-{filename}".encode())
        artifact_hashes[filename] = _sha256(path)

    selected = [
        {
            "ordinal": index,
            "compound_id": f"record-{index:02d}",
            "chemical_class": f"class-{index:02d}",
            "ring_atom_count": 0,
        }
        for index in range(1, 21)
    ]
    candidates = [
        {
            "compound_id": item["compound_id"],
            "mol2_relative_path": (
                f"dataset/mol2files_gaff/{item['compound_id']}.mol2"
            ),
            "mol2_sha256": f"{item['ordinal']:064x}",
            "dataset_record_sha256": f"{item['ordinal'] + 20:064x}",
            "structure_group_sha256": f"{item['ordinal'] + 40:064x}",
            "experimental_kcal_mol": -float(item["ordinal"]),
            "experimental_uncertainty_kcal_mol": 0.6,
        }
        for item in selected
    ]
    prepared = {
        "candidate_count": 20,
        "candidates": candidates,
        "dataset_commit": "a" * 40,
        "protocol_id": "synthetic-freesolv",
        "protocol_fingerprint": "b" * 64,
        "dataset_artifact_sha256": artifact_hashes,
    }
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared, sort_keys=True), encoding="utf-8")
    candidate_map = {item["compound_id"]: item for item in candidates}
    selected_digest = runner._canonical_sha256(
        runner._selected_record_manifest(selected, candidate_map)
    )
    selection = {
        "dataset": {"expected_record_count": 20, "repository_commit": "a" * 40},
        "prepared_manifest": {
            "sha256": _sha256(prepared_path),
            "protocol_id": "synthetic-freesolv",
            "protocol_fingerprint": "b" * 64,
            "dataset_artifact_sha256": artifact_hashes,
            "selected_record_manifest_sha256": selected_digest,
        },
    }
    observed, digest = runner._validate_prepared_manifest(
        selection,
        selected,
        prepared,
        prepared_path=prepared_path,
        dataset_dir=dataset_dir,
    )
    assert set(observed) == {item["compound_id"] for item in selected}
    assert digest == selected_digest

    prepared_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256 drifted"):
        runner._validate_prepared_manifest(
            selection,
            selected,
            prepared,
            prepared_path=prepared_path,
            dataset_dir=dataset_dir,
        )


def test_runner_atomic_writer_never_leaves_temporary_file(
    runner: ModuleType,
    tmp_path: Path,
):
    output = tmp_path / "artifact.json"
    runner._write_json_atomic(output, {"status": "complete", "value": 3})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "complete",
        "value": 3,
    }
    assert not output.with_suffix(".json.tmp").exists()
