from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / (
    "docs/route2/preregistrations/"
    "auxiliary-density-confirmation-selection-v1.json"
)
DEVELOPMENT = ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-v0-freesolv12-zero-field-mace-static-surface-mep-prereg-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_confirmation_was_sealed_before_ladder_results() -> None:
    value = json.loads(SELECTION.read_text())
    creator = ROOT / value["source"]["creator_path"]

    assert value["artifact"] == "route2-auxiliary-density-confirmation-selection-v1"
    assert value["status"] == "locked-before-any-standard-basis-ladder-result"
    assert len(value["records"]) == 12
    assert value["source"]["creator_sha256"] == _sha256(creator)
    assert value["one_shot_contract"] == {
        "apply_only_to_first_development_passer": True,
        "confirmation_failure_closes_entire_frozen_ladder": True,
        "identical_numerical_and_reaction_metric_gates": True,
        "later_candidates_forbidden_after_confirmation_failure": True,
    }


def test_confirmation_is_identity_disjoint_and_contains_no_target_fields() -> None:
    value = json.loads(SELECTION.read_text())
    development = json.loads(DEVELOPMENT.read_text())
    selected_ids = {record["compound_id"] for record in value["records"]}
    development_ids = {
        record["compound_id"] for record in development["locked_records"]
    }
    forbidden = {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "predicted_kcal_mol",
    }

    assert selected_ids.isdisjoint(development_ids)
    assert len({record["smiles"] for record in value["records"]}) == 12
    assert all(forbidden.isdisjoint(record) for record in value["records"])
    assert value["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "model_output_read": False,
    }
