from __future__ import annotations

import json
from pathlib import Path

from tools.route2_release import run_mdp_polar_uniform_response_gate as gate

ROOT = Path(__file__).resolve().parents[2]


def test_uniform_response_gate_is_prospectively_bound_and_disjoint() -> None:
    path, prereg = gate._load_preregistration()
    records = gate._selected_records(prereg)
    prior = json.loads(
        (
            ROOT / "docs/route2/preregistrations/"
            "mdp-mbis-pcm-source-physical-gate-v1.json"
        ).read_text()
    )
    assert gate.parent._sha256_file(path) == (
        "834c8a9f06ed2974467049c58485cf3d4809ff77ba93aaf0193b07ba697d0ef5"
    )
    assert len(records) == len({record["molecule"] for record in records}) == 60
    assert not {record["molecule"] for record in records}.intersection(
        record["molecule"] for record in prior["selection"]["records"]
    )
    assert prereg["selection"]["selection_sha256"] == (
        "0b1e1de6bb569407453ffdd312e0ab49b66083d5498b0730bc94128389b17e83"
    )


def test_uniform_response_gate_binds_live_code_and_forbids_training() -> None:
    _path, prereg = gate._load_preregistration()
    inputs = prereg["inputs"]
    assert inputs["runner_sha256"] == gate.parent._sha256_file(Path(gate.__file__))
    assert inputs["response_manifold_implementation_sha256"] == (
        gate.parent._sha256_file(
            ROOT / "maple/solvation/release/uniform_response_manifold.py"
        )
    )
    claim = prereg["claim_boundary"]
    assert claim["post_training_or_fitting_performed"] is False
    assert claim["experimental_solvation_targets_read"] is False
    assert claim["source_or_energy_result_used_for_selection"] is False
    assert claim["selection_disjoint_from_prior_sixty_molecules"] is True


def test_decision_rule_cannot_select_response_from_mae_alone() -> None:
    _path, prereg = gate._load_preregistration()
    rules = prereg["prospective_decision_rules"]
    assert rules["fixed_source_energy_mae_kcal_mol_maximum"] == 1.5
    assert rules["response_mae_reduction_fraction_minimum"] == 0.1
    assert rules["response_paired_improvement_fraction_minimum"] == 0.6
    assert rules["fixed_source_energy_q95_kcal_mol_maximum"] == 3.0
    assert rules["fixed_source_energy_maximum_kcal_mol_maximum"] == 4.0
    assert "every response gate" in rules["selection_rule"]
