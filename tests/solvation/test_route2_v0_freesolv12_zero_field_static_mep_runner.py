from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_route2_v0_freesolv12_zero_field_static_mep as runner  # noqa: E402

EXECUTION_PATH = BENCHMARK_DIR / (
    "route2-v0-freesolv12-zero-field-mace-static-surface-mep-"
    "execution-edfae78e.json"
)


def test_label_free_static_source_manifest_is_all_twelve_geometries():
    manifest = runner.load_static_mep_manifest(runner.MANIFEST_PATH)
    records = manifest["locked_records"]

    assert manifest["status"] == "preregistered-not-executed"
    assert len(records) == 12
    assert len({record["compound_id"] for record in records}) == 12
    assert {record["functional_group"] for record in records if record["functional_group"]} == {
        "primary-alcohol",
        "dialkyl-ether",
        "ketone",
        "carboxylic-acid-ester",
        "carbonitrile",
        "primary-aromatic-amine",
        "alkyl-chloride",
        "sulfoxide",
        "primary-carboxylic-acid-amide",
        "carboxylic-acid",
    }
    forbidden = {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "experimental_reference",
        "smiles",
    }
    assert not any(forbidden.intersection(record) for record in records)


def test_static_source_manifest_rejects_an_added_experimental_label(tmp_path):
    manifest = json.loads(runner.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["locked_records"][0]["experimental_kcal_mol"] = 0.0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exact no-label schema"):
        runner.load_static_mep_manifest(path)


def test_static_source_runner_uses_zero_field_source_and_not_a_continuum_ledger():
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert runner.ARTIFACT == "route2-v0-freesolv12-zero-field-mace-static-surface-mep-v1"
    assert "calculator.polar_state(atoms)" in source
    assert "point_multipole_potential" in source
    assert "continuum_or_solvation_energy_invoked\": False" in source
    assert "v0_permanent_reference_admitted\": False" in source
    assert "experimental_solvation_labels_read\": False" in source


def test_qm_interpreter_path_preserves_a_virtual_environment_symlink(tmp_path):
    target = tmp_path / "base-python"
    target.write_text("placeholder", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)

    observed = runner._executable_path_preserving_venv(venv_python)

    assert observed == venv_python.absolute()
    assert observed.is_symlink()


def test_public_static_source_execution_preserves_the_no_label_boundary():
    payload = json.loads(EXECUTION_PATH.read_text(encoding="utf-8"))
    records = payload["records"]

    assert payload["status"] == "complete-source-diagnostic-no-acceptance-threshold"
    assert len(records) == 12
    assert payload["execution"]["execution_git_head"].startswith("edfae78e")
    assert payload["frozen_protocol"]["functionalized_record_count"] == 10
    assert payload["disposition"]["experimental_solvation_labels_read"] is False
    assert payload["disposition"]["v0_permanent_reference_admitted"] is False
    assert not any(
        key.startswith("experimental_")
        for record in records
        for key in record
    )
    assert payload["observed_metrics"]["functionalized_records"] == {
        "maximum_weighted_relative_l2": pytest.approx(0.1611112244083607),
        "mean_weighted_relative_l2": pytest.approx(0.11377174378458857),
        "minimum_weighted_relative_l2": pytest.approx(0.07920875507302354),
        "record_count": 10,
        "worst_compound_id_by_weighted_relative_l2": "mobley_2198613",
    }
