from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-matched-projector-top5-diagnostic-v1.json"
)


def test_matched_projector_ablation_is_not_promoted_to_v0_or_acceptance():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 1
    assert artifact["artifact"] == "route2-matched-projector-top5-diagnostic-v1"
    assert artifact["status"] == "diagnostic-only-not-an-acceptance-panel"
    assert "cannot establish a general accuracy ranking" in artifact["claim_boundary"]
    assert artifact["selection"] == {
        "source_commit": "c77eb6a295cbb39a1b6e66d21190a192cc8ef050",
        "source_path": (
            "docs/implicit-solvation/benchmarks/"
            "freesolv10-qeq-gto-full-h-2026-07-22.json"
        ),
        "source_sha256": (
            "893ad62e476e6375602d82231def6b0ad22cab60128c8068894daaf4a6ee8c4c"
        ),
        "ranking": "top five historical retired QEq-GTO/GBn2 absolute errors",
    }
    assert artifact["method_controls"] == {
        "only_intended_difference": (
            "reaction_field_projector local-jet versus exact-gto-v1"
        ),
        "same_cavity": "canonical SMD PCMSolver built-in water cavity",
        "same_checkpoint": "polar-1-m",
        "same_model_field_gauge": "atomic-center-mean-zero-v1",
        "same_nonpolar": "native water SMD CDS",
        "same_provider": "pcmsolver IEFPCM",
    }
    assert all(
        artifact["no_target_policy"][key] is False
        for key in (
            "post_training",
            "fine_tuning",
            "experimental_solvation_fit",
            "map_or_uq_calibration",
            "radius_adjustment",
            "half_coupling_adjustment",
            "post_selection_parameter_change",
            "experimental_labels_used_to_rank_current_method",
        )
    )


def test_matched_projector_ablation_locks_moleculewise_winners_and_boundaries():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    expected = (
        ("ethyl acetate", "exact_gto", 2.4346141145007345, 0.6963090734616988),
        ("aniline", "centered_local_jet", 0.02979557969629365, 0.2214856905372482),
        ("acetonitrile", "centered_local_jet", 0.20484405347525758, 0.7063579951124037),
        ("chloroethane", "centered_local_jet", 0.034574206809241614, 0.4985664145471602),
        ("methoxymethane", "centered_local_jet", 0.6068250170892664, 1.0762969614701723),
    )
    records = artifact["records"]
    assert len(records) == len(expected)
    for record, (name, winner, local_error, exact_error) in zip(records, expected):
        assert record["name"] == name
        assert record["winner"] == winner
        assert record["centered_local_jet"]["absolute_error_kcal_mol"] == pytest.approx(
            local_error, abs=1e-12
        )
        assert record["exact_gto"]["absolute_error_kcal_mol"] == pytest.approx(
            exact_error, abs=1e-12
        )
        local_contract = record["centered_local_jet"]["immutable_evaluation_record"][
            "source_receiver_contract"
        ]
        exact_contract = record["exact_gto"]["immutable_evaluation_record"][
            "source_receiver_contract"
        ]
        assert local_contract["continuum_pairing_established"] is True
        assert exact_contract["continuum_pairing_established"] is False
        assert local_contract["common_stationary_electronic_functional_established"] is False
        assert exact_contract["common_stationary_electronic_functional_established"] is False
        assert local_contract["public_capability"] == "experimental-energy-only"
        assert exact_contract["public_capability"] == "experimental-energy-only"

    summary = artifact["summary"]
    assert summary["exact_gto_wins"] == 1
    assert summary["local_jet_wins"] == 4
    assert summary["exact_gto_mean_absolute_error_kcal_mol"] == pytest.approx(
        0.6398032270257366, abs=1e-12
    )
    assert summary["local_jet_mean_absolute_error_kcal_mol"] == pytest.approx(
        0.6621305943141588, abs=1e-12
    )
