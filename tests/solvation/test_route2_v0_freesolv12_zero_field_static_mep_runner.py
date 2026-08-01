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
