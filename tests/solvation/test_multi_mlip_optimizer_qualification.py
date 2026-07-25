from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
RUNNER = BENCHMARK_DIR / "run_multi_mlip_optimizer_qualification.py"
PROTOCOL = BENCHMARK_DIR / "multi_mlip_optimizer_qualification_protocol.json"
FAILURE_AUDIT = (
    BENCHMARK_DIR
    / "route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-flexible-"
    "preflight-failure-2026-07-25.json"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "multi_mlip_optimizer_qualification",
        RUNNER,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_optimizer_protocol_is_label_blind_and_freezes_all_three_mlips():
    runner = _load_runner()
    protocol, fingerprint, qrrho_protocol, _ = runner.load_protocol(PROTOCOL)

    assert len(fingerprint) == 64
    assert protocol["qualification_scope"]["model_ids"] == [
        "maceoff23m",
        "aimnet2",
        "ani2x",
    ]
    assert protocol["qualification_scope"]["phases"] == ["gas", "solution"]
    assert protocol["qualification_scope"]["expected_branch_count_per_candidate"] == 18
    assert protocol["qualification_scope"][
        "per_model_or_per_phase_optimizer_selection_allowed"
    ] is False
    assert protocol["label_boundary"] == {
        "experimental_labels_read": False,
        "forbidden_inputs": [
            "FreeSolv experimental values",
            "hydration residuals",
            "per-model accuracy metrics",
        ],
        "hydration_accuracy_evaluated": False,
        "selection_uses_only_optimizer_observables": True,
    }
    assert protocol["route_boundary"]["gas_phase_mm_energy"] is False
    assert protocol["route_boundary"]["hydration_label_residual"] is False
    assert [model["name"] for model in qrrho_protocol["models"]] == [
        "maceoff23m",
        "aimnet2",
        "ani2x",
    ]


def test_v8_flexible_failure_is_durable_and_closes_v8():
    runner = _load_runner()
    artifact = runner.load_json(FAILURE_AUDIT)
    runner._validate_self_hash(artifact, name="v8 flexible failure audit")

    assert artifact["status"] == "failed-closed-before-seal-or-label-scoring"
    assert artifact["model_execution"]["maceoff23m"]["status"] == (
        "success-diagnostic-only"
    )
    assert artifact["model_execution"]["aimnet2"]["status"] == "failure"
    assert artifact["model_execution"]["ani2x"]["attempted"] is False
    assert artifact["decision"]["v8_resume_allowed"] is False
    assert artifact["decision"]["v8_record_replacement_allowed"] is False
    assert artifact["decision"]["v8_aggregate_seal_allowed"] is False
    assert artifact["decision"]["v8_label_scoring_allowed"] is False
    failed = next(
        branch
        for branch in artifact["optimizer_failure"]["branch_diagnostics"]
        if branch["source_state_index"] == 16
    )
    assert failed["reported_step_count"] == 500
    assert failed["converged_under_frozen_force_gate"] is False
    assert failed["final_max_force_eV_per_angstrom"] > 0.01
    for evidence in artifact["raw_evidence"]:
        path = REPOSITORY_ROOT / evidence["path"]
        assert path.stat().st_size == evidence["size_bytes"]
        assert runner.sha256_file(path) == evidence["sha256"]


def test_global_optimizer_selection_is_fail_closed_and_model_neutral():
    runner = _load_runner()
    passing = {
        "candidate_id": "global",
        "candidate_order": 1,
        "all_branches_passed": True,
        "total_calculator_evaluations": 100,
        "total_optimizer_steps": 80,
    }
    faster_but_incomplete = {
        "candidate_id": "partial",
        "candidate_order": 0,
        "all_branches_passed": False,
        "total_calculator_evaluations": 1,
        "total_optimizer_steps": 1,
    }
    slower = {
        "candidate_id": "slower",
        "candidate_order": 2,
        "all_branches_passed": True,
        "total_calculator_evaluations": 101,
        "total_optimizer_steps": 50,
    }

    assert runner.select_global_candidate([faster_but_incomplete]) is None
    selected = runner.select_global_candidate(
        [faster_but_incomplete, slower, passing]
    )
    assert selected["candidate_id"] == "global"


def test_optimizer_candidate_factory_covers_only_frozen_algorithms(tmp_path):
    runner = _load_runner()
    protocol, _, _, _ = runner.load_protocol(PROTOCOL)
    from ase import Atoms
    from ase.calculators.emt import EMT

    for candidate in protocol["optimizer_candidates"]:
        atoms = Atoms(
            "Cu",
            positions=[[0.0, 0.0, 0.0]],
            calculator=EMT(),
        )
        optimizer = runner._make_optimizer(
            candidate,
            atoms=atoms,
            logfile=tmp_path / f"{candidate['candidate_id']}.log",
            trajectory=tmp_path / f"{candidate['candidate_id']}.traj",
        )
        assert optimizer is not None
