from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_route2_v0_freesolv12_localized_response as runner  # noqa: E402

COMPLETED_ARTIFACT = (
    BENCHMARK_DIR
    / "route2-v0-freesolv12-mace-localized-response-execution-62a8e413.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_localized_response_manifest_is_frozen_and_label_free():
    manifest = json.loads(runner.MANIFEST_PATH.read_text(encoding="utf-8"))
    records = manifest["locked_records"]

    assert manifest["status"] == "frozen-before-execution"
    assert len(records) == 12
    assert len({row["functional_group"] for row in records if row["functional_group"]}) == 10
    assert manifest["mode_protocol"]["mode_count"] == 4
    assert manifest["finite_field_protocol"] == {
        "field_steps_e": [0.0003, 0.001],
        "selected_reporting_step_e": 0.0003,
        "signs": [-1, 1],
    }
    forbidden = {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "experimental_reference",
        "smiles",
    }
    assert not any(forbidden.intersection(row) for row in records)

    completed = json.loads(COMPLETED_ARTIFACT.read_text(encoding="utf-8"))
    assert completed["status"] == "complete-response-gate-reject"
    assert completed["run_lock"]["manifest_sha256"] == _sha256(
        runner.MANIFEST_PATH
    )
    assert (
        completed["run_lock"]["source_files_sha256"]
        == manifest["execution_contract"]["source_sha256"]
    )

    # The preregistration has already produced a completed immutable result.
    # Current source evolution must be rejected by that old runner rather than
    # accommodated by changing its frozen hashes.
    with pytest.raises(RuntimeError, match="source changed after freeze"):
        runner._load_manifest(runner.MANIFEST_PATH)


def test_localized_response_runner_cannot_invoke_a_continuum_or_select_modes_from_results():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    manifest = json.loads(runner.MANIFEST_PATH.read_text(encoding="utf-8"))

    assert runner.ARTIFACT == "route2-v0-freesolv12-mace-localized-response-v1"
    assert "linearize_density_response" in source
    assert "select_farthest_exterior_point_charge_modes" in source
    assert "solvent_correction_evaluate" not in source
    assert "experimental_solvation_labels_read\": False" in source
    assert manifest["hard_constraints"]["continuum_or_pcm_invoked"] is False
    assert (
        manifest["hard_constraints"][
            "mode_or_record_subset_selected_after_execution"
        ]
        is False
    )


def test_qm_localized_response_uses_the_same_scalar_point_charge_perturbation():
    helper = (BENCHMARK_DIR / "route2_qm_localized_point_charge_response.py").read_text(
        encoding="utf-8"
    )

    assert "base_hcore - amplitude_e * source_integral" in helper
    assert "FIELD_STEPS_E = (3.0e-4, 1.0e-3)" in helper
    assert "electronic_surface_potential_hartree_per_e" in helper
    assert "route2_qm_surface_mep" in helper
