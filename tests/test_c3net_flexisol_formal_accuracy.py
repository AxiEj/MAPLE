from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
from rdkit import Chem

import maple.function.benchmarking.c3net_accuracy as c3net_accuracy
import maple.function.benchmarking.pretrained_hub as pretrained_hub
import maple.function.solvfe.c3net_property as c3net_property
from maple.function.benchmarking import MolecularInputReceipt

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / "docs/pretrained-solvation-hub/run_c3net_flexisol_formal_accuracy.py"
)
RESULT_ROOT = (
    ROOT / "docs/pretrained-solvation-hub/benchmarks/"
    "c3net-flexisol-formal-2026-07-30"
)
RESULT_PATH = RESULT_ROOT / "result.json"
RESULT_SHA256 = "fdd2f535b2e8be04788323283812f20c817e3641d783339f6b80b3c92db3884e"
REPRODUCIBILITY_PATH = (
    ROOT / "docs/pretrained-solvation-hub/benchmarks/"
    "c3net-flexisol-formal-2026-07-30-reproducibility.json"
)
REPRODUCIBILITY_SHA256 = (
    "d54f7b09848330bd51d9263355b7fdd7f08c284f02306bf1d9c666b80bf82a7c"
)
TAXONOMY_PATH = ROOT / "docs/pretrained-solvation-hub/functional-group-taxonomy-v1.json"


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "run_c3net_flexisol_formal_accuracy",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_selection_identity_is_output_blind():
    module = _load_script()
    row = {
        "FlexiSol Name": "ethanol",
        "Solvent": "water",
        "Ref.": "10.example/reference",
        r"Value (\kcalpmole)": "-5.0",
    }
    changed_label = dict(row)
    changed_label[r"Value (\kcalpmole)"] = "999.0"

    assert module._candidate_identity(row, "CCO") == module._candidate_identity(
        changed_label,
        "CCO",
    )


def test_run_rejects_reusing_an_output_directory_before_external_setup(tmp_path):
    module = _load_script()
    output_root = tmp_path / "existing-output"
    output_root.mkdir()
    sentinel = output_root / "do-not-overwrite.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="[Oo]utput directory.*exist"):
        module.run(
            flexisol_source_root=tmp_path / "unused-flexisol-source",
            c3net_source_root=tmp_path / "unused-c3net-source",
            c3net_source_bundle=tmp_path / "unused-c3net-bundle",
            output_directory=output_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert tuple(output_root.iterdir()) == (sentinel,)


def test_frozen_result_contains_19_distinct_groups_and_rejects_c3net_accuracy():
    assert hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest() == RESULT_SHA256
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    accuracy = payload["accuracy"]
    coverage = accuracy["functional_group_coverage"]
    errors = accuracy["per_record_errors"]

    assert payload["scientific_status"] == (
        "development_accuracy_only_overlap_unknown_not_final_holdout"
    )
    assert payload["leakage_audit"]["status"] == "overlap_unknown"
    assert accuracy["record_count"] == 19
    assert accuracy["evaluated_count"] == 19
    assert accuracy["accuracy_precision_policy"] == "float32"
    assert coverage["passes"] is True
    assert coverage["observed_count"] == 19
    assert len(set(coverage["observed_functional_groups"])) == 19
    assert len(errors) == 19
    assert len({row["primary_functional_group"] for row in errors}) == 19
    assert all(
        {
            "experimental_kcal_mol",
            "predicted_kcal_mol",
            "signed_error_kcal_mol",
            "absolute_error_kcal_mol",
        }.issubset(row)
        for row in errors
    )
    assert accuracy["metrics"]["mae_kcal_mol"] == pytest.approx(2.148012556778758)
    assert accuracy["metrics"]["rmse_kcal_mol"] == pytest.approx(2.9857217677885237)
    assert accuracy["metrics"]["maximum_absolute_error_kcal_mol"] == (
        pytest.approx(8.459287071228028)
    )
    maximum = max(errors, key=lambda row: row["absolute_error_kcal_mol"])
    assert maximum["primary_functional_group"] == "ester"
    assert maximum["experimental_kcal_mol"] == -5.6
    assert maximum["predicted_kcal_mol"] == pytest.approx(-14.059287071228027)
    assert payload["accuracy_gate"]["passed"] is False
    assert payload["speed_gate"]["status"] == ("not_eligible_due_to_accuracy_failure")


def test_frozen_result_binds_registered_artifacts_and_diverse_conformers():
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert len(tuple(path for path in RESULT_ROOT.rglob("*") if path.is_file())) == 21
    roles = {artifact["role"] for artifact in payload["implementation_artifacts"]}
    assert roles == {
        "adapter_code",
        "checkpoint",
        "dependency_lock",
        "embedding",
        "featurizer_code",
        "property_adapter_code",
        "runner_code",
        "runtime_lock",
    }
    assert {artifact["role"] for artifact in payload["panel_generation_artifacts"]} == {
        "panel_runner_code",
        "functional_group_taxonomy",
    }
    for artifact in payload["panel_generation_artifacts"]:
        assert artifact["execution_scope"] == "parent_only"
        assert artifact["registered_with_label_free_adapter"] is False
        assert artifact["mounted_in_prediction_sandbox"] is False

    selected = payload["selection"]["records"]
    assert len(selected) == 19
    assert len({record["primary_functional_group"] for record in selected}) == 19
    observed_conformer_counts = []
    for record in selected:
        path = RESULT_ROOT / "inputs" / f"{record['record_id']}.sdf"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            record["molecular_input_sha256"]
        )
        supplier = Chem.ForwardSDMolSupplier(
            str(path),
            sanitize=True,
            removeHs=True,
            strictParsing=True,
        )
        conformers = tuple(supplier)
        assert all(conformer is not None for conformer in conformers)
        assert 1 <= len(conformers) <= 5
        generation = record["conformer_generation"]
        assert generation["candidate_count"] == 50
        assert generation["conformer_count"] == len(conformers)
        assert generation["heavy_atom_rmsd_threshold_angstrom"] == pytest.approx(0.5)
        assert generation["random_seed"] == (
            int(record["selection_hash"][:8], 16) & 0x7FFFFFFF
        )
        assert len(generation["selected_conformer_ids"]) == len(conformers)
        assert len(set(generation["selected_conformer_ids"])) == len(conformers)
        assert all(
            isinstance(conformer_id, int)
            and 0 <= conformer_id < generation["candidate_count"]
            for conformer_id in generation["selected_conformer_ids"]
        )
        assert (
            len(generation["uff_optimization_statuses"])
            == generation["candidate_count"]
        )
        assert {
            status["conformer_id"] for status in generation["uff_optimization_statuses"]
        } == set(range(generation["candidate_count"]))
        observed_conformer_counts.append(len(conformers))
        receipt = MolecularInputReceipt.from_file(
            path,
            molecular_input_format="sdf_conformers",
        )
        assert receipt.molecular_input_sha256 == record["molecular_input_sha256"]
    assert any(count < 5 for count in observed_conformer_counts)


def test_frozen_result_binds_current_repository_code_and_registration():
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    artifacts = {
        artifact["role"]: artifact["artifact_sha256"]
        for artifact in payload["implementation_artifacts"]
    }
    current_files = {
        "adapter_code": Path(c3net_accuracy.__file__),
        "property_adapter_code": Path(c3net_property.__file__),
        "runner_code": Path(pretrained_hub.__file__),
        "runtime_lock": (
            ROOT / "docs/pretrained-solvation-hub/"
            "c3net-formal-runtime-lock-2026-07-30.json"
        ),
    }
    assert {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in current_files.items()
    } == {role: artifacts[role] for role in current_files}

    panel_generation_artifacts = {
        artifact["role"]: artifact["artifact_sha256"]
        for artifact in payload["panel_generation_artifacts"]
    }
    assert panel_generation_artifacts == {
        "panel_runner_code": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        "functional_group_taxonomy": hashlib.sha256(
            TAXONOMY_PATH.read_bytes()
        ).hexdigest(),
    }

    source_root = Path(
        os.environ.get(
            "MAPLE_C3NET_FORMAL_SOURCE_ROOT",
            "/home/axie/.cache/maple-benchmarks/c3net/source",
        )
    )
    source_bundle = Path(
        os.environ.get(
            "MAPLE_C3NET_FORMAL_SOURCE_BUNDLE",
            "/home/axie/.cache/maple-benchmarks/c3net/" "c3net-191d5a928fb4787d.bundle",
        )
    )
    if not source_root.is_dir() or not source_bundle.is_file():
        pytest.skip("Pinned local C3Net source and bundle are unavailable.")
    registration = c3net_accuracy.build_c3net_formal_accuracy_registration(
        source_root=source_root,
        source_bundle=source_bundle,
    )
    assert registration.compute_fingerprint() == payload["accuracy_adapter_fingerprint"]


def test_reproducibility_receipt_binds_the_complete_canonical_tree():
    assert hashlib.sha256(REPRODUCIBILITY_PATH.read_bytes()).hexdigest() == (
        REPRODUCIBILITY_SHA256
    )
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

    assert receipt["full_tree_byte_identical"] is True
    assert receipt["file_count"] == len(manifest) == 21
    assert receipt["relative_file_sha256"] == manifest
    assert receipt["file_manifest_sha256"] == manifest_digest
    assert receipt["result_sha256"] == RESULT_SHA256
    assert len(receipt["runs"]) == 2
    assert {
        (run["file_count"], run["file_manifest_sha256"]) for run in receipt["runs"]
    } == {(21, manifest_digest)}
    assert all(run["artifact_write_span_seconds"] > 0 for run in receipt["runs"])
    assert (
        receipt["diagnostics"]["artifact_span_is_wall_clock_or_performance_evidence"]
        is False
    )
    assert (
        receipt["diagnostics"]["timings_are_reproducibility_diagnostics_only"] is True
    )
    assert (
        receipt["diagnostics"]["timings_are_matched_qm_performance_evidence"] is False
    )
