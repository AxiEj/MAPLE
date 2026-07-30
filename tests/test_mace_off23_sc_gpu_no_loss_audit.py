from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "run_mace_off23_sc_gpu_no_loss_audit.py"
)
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "mace-off23-sc-gpu-no-loss-audit-2026-07-30.json"
)
WATCHLIST = ROOT / "docs" / "pretrained-solvation-hub" / "research_watchlist.yaml"
MODEL_CARD = (
    ROOT / "maple" / "function" / "calculator" / "model_cards" / "mace-off23-sc.yaml"
)


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "mace_off23_sc_gpu_no_loss_audit",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _scientific_projection(payload):
    projection = deepcopy(payload)
    projection.pop("timing_diagnostic_not_an_admission_observable")
    return projection


def test_frozen_audit_is_negative_parity_evidence_not_accuracy():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["verdict"] == "fail_no_loss"
    assert payload["acceptance_eligible"] is False
    assert payload["accuracy_evaluation_performed"] is False
    assert payload["panel"] == {
        "accuracy_metric_reporting_allowed": False,
        "experimental_labels_used": False,
        "functional_group_accuracy_gate": {
            "minimum_distinct_functional_groups": 10,
            "minimum_record_count": 10,
            "observed_distinct_functional_groups": 0,
            "observed_record_count": 0,
            "passes": False,
            "record_assignments_predeclared": False,
        },
        "purpose": "numerical_cpu_cuda_rejection_only",
        "record_count": 1,
    }
    assert payload["gpu_admission"]["smaller_panel_can_reject_but_never_unlock"]
    assert payload["gpu_admission"]["allowed_tasks"] == []
    assert payload["gpu_admission"]["allowed_inference_modes"] == []
    assert (
        payload["scientific_scope"]["absolute_solvation_free_energy_evaluated"] is False
    )
    assert payload["scientific_scope"]["experimental_accuracy_evaluated"] is False


def test_frozen_identity_and_raw_output_aggregates_recompute():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    identity = payload["identity"]
    assert identity["checkpoint_revision"] == AUDIT.CHECKPOINT_REVISION
    assert identity["checkpoint_tree"] == AUDIT.CHECKPOINT_TREE
    assert identity["checkpoint"]["sha256"] == AUDIT.CHECKPOINT_SHA256
    assert identity["protocol_revision"] == AUDIT.PROTOCOL_REVISION
    assert identity["protocol_tree"] == AUDIT.PROTOCOL_TREE
    assert identity["waterbox"]["sha256"] == AUDIT.WATERBOX_SHA256
    assert identity["waterbox"]["atom_count"] == 192
    assert identity["waterbox"]["formula"] == "H128O64"

    energy = payload["observables"]["energy"]
    reference_energy = energy["reference_values_ev"]
    accelerator_energy = energy["accelerator_values_ev"]
    assert reference_energy == [-133210.23996324115]
    assert accelerator_energy == [-133210.2399632409]
    assert energy["observed_maximum_abs_difference_ev"] == abs(
        accelerator_energy[0] - reference_energy[0]
    )
    assert energy["observed_maximum_abs_difference_ev"] == pytest.approx(
        2.6193447411060333e-10
    )
    assert energy["passed_exact_equality"] is False

    record_id = "float64:official-waterbox-192"
    assert energy["reference_sha256"] == AUDIT._output_sha256(
        observable="energy",
        record_id=record_id,
        role="reference",
        values=reference_energy,
    )
    assert energy["accelerator_sha256"] == AUDIT._output_sha256(
        observable="energy",
        record_id=record_id,
        role="accelerator",
        values=accelerator_energy,
    )

    forces = payload["observables"]["forces"]
    reference_forces = np.asarray(
        forces["reference_values_ev_per_angstrom"],
        dtype=np.float64,
    )
    accelerator_forces = np.asarray(
        forces["accelerator_values_ev_per_angstrom"],
        dtype=np.float64,
    )
    difference = np.abs(accelerator_forces - reference_forces)
    assert reference_forces.shape == (576,)
    assert accelerator_forces.shape == (576,)
    assert forces["nonzero_difference_count"] == int(np.count_nonzero(difference))
    assert forces["nonzero_difference_count"] == 551
    assert forces["observed_maximum_abs_difference_ev_per_angstrom"] == float(
        difference.max()
    )
    assert forces["observed_maximum_abs_difference_ev_per_angstrom"] == pytest.approx(
        1.0769163338864018e-14
    )
    assert forces["passed_exact_equality"] is False
    assert forces["reference_sha256"] == AUDIT._output_sha256(
        observable="forces",
        record_id=record_id,
        role="reference",
        values=reference_forces.tolist(),
    )
    assert forces["accelerator_sha256"] == AUDIT._output_sha256(
        observable="forces",
        record_id=record_id,
        role="accelerator",
        values=accelerator_forces.tolist(),
    )

    manifest = payload["comparison_manifest"]
    assert (
        payload["comparison_manifest_sha256"]
        == hashlib.sha256(AUDIT._canonical_bytes(manifest)).hexdigest()
    )
    assert manifest["observable_counts"] == {"energy": 1, "forces": 576}


def test_watchlist_binds_negative_artifact_while_model_card_remains_closed():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    watchlist = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    candidate = next(
        model for model in watchlist["models"] if model["model_id"] == "mace-off23-sc"
    )
    gpu = candidate["gpu_no_loss_audit"]
    assert gpu["artifact"] == ARTIFACT.relative_to(WATCHLIST.parent).as_posix()
    assert gpu["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert gpu["gpu_admitted"] is False
    assert gpu["no_loss_parity_verified"] is False

    card = json.loads(MODEL_CARD.read_text(encoding="utf-8"))
    assert card["gpu_acceleration"]["no_loss_parity_verified"] is False
    assert card["gpu_acceleration"]["evidence_artifact"] is None
    assert card["gpu_acceleration"]["evidence_sha256"] is None
    assert card["gpu_acceleration"]["allowed_tasks"] == []
    assert card["capabilities"]["supports_absolute_solvation"] is False
    assert card["capabilities"]["supports_alchemical_lambda"] is False
    assert payload["gpu_admission"]["card_evidence_must_remain_null"] is True


def test_fresh_physical_audit_matches_frozen_scientific_projection(tmp_path):
    checkpoint_root = os.environ.get("MAPLE_MACE_OFF23_SC_CHECKPOINT_ROOT")
    protocol_root = os.environ.get("MAPLE_MACE_MD_PROTOCOL_ROOT")
    if not checkpoint_root or not protocol_root:
        pytest.skip(
            "Set MAPLE_MACE_OFF23_SC_CHECKPOINT_ROOT and "
            "MAPLE_MACE_MD_PROTOCOL_ROOT for the physical GPU audit."
        )

    output = tmp_path / "fresh.json"
    environment = dict(os.environ)
    environment.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        AUDIT.CUBLAS_WORKSPACE_CONFIG,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint-root",
            checkpoint_root,
            "--protocol-root",
            protocol_root,
            "--output",
            str(output),
        ],
        check=True,
        env=environment,
    )
    fresh = json.loads(output.read_text(encoding="utf-8"))
    frozen = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert _scientific_projection(fresh) == _scientific_projection(frozen)
