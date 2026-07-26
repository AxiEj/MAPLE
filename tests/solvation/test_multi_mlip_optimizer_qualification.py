from __future__ import annotations

from argparse import Namespace
import importlib.util
import math
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
RUNNER = BENCHMARK_DIR / "run_multi_mlip_optimizer_qualification.py"
PROTOCOL = BENCHMARK_DIR / "multi_mlip_optimizer_qualification_protocol.json"
RIGID_PREFLIGHT = (
    BENCHMARK_DIR
    / "route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-rigid-"
    "preflight-2026-07-25.json"
)
FAILURE_AUDIT = (
    BENCHMARK_DIR
    / "route1-multi-mlip-phase-specific-selected-minimum-rrho-v8-flexible-"
    "preflight-failure-2026-07-25.json"
)
QUALIFICATION_V2 = (
    BENCHMARK_DIR
    / "route1-multi-mlip-optimizer-qualification-v2-2026-07-26.json"
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
    assert protocol["qualification_gates"][
        "maximum_calculator_evaluations_per_branch"
    ] == 1000
    assert protocol["frozen_upstream"]["v1_interruption_audit"][
        "v1_records_reused"
    ] is False
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


@pytest.mark.parametrize(
    ("command_name", "arguments"),
    [
        (
            "validate_command",
            Namespace(protocol=str(PROTOCOL)),
        ),
        (
            "run_command",
            Namespace(protocol=str(PROTOCOL), work_dir="unused"),
        ),
        (
            "seal_command",
            Namespace(
                protocol=str(PROTOCOL),
                record_dir="unused",
                output="unused",
            ),
        ),
    ],
    ids=["validate", "run", "seal"],
)
def test_optimizer_cli_never_executes_or_seals_documented_source_drift(
    command_name, arguments
):
    runner = _load_runner()

    with pytest.raises(ValueError, match="historical-audit-only"):
        getattr(runner, command_name)(arguments)


def test_v8_rigid_preflight_is_durable_and_scope_isolated():
    runner = _load_runner()
    artifact = runner.load_json(RIGID_PREFLIGHT)
    runner._validate_self_hash(artifact, name="v8 rigid preflight audit")

    assert artifact["case_scope"] == ["mobley_1952272"]
    assert artifact["model_case_count"] == len(artifact["records"]) == 3
    assert {record["model"] for record in artifact["records"]} == set(
        artifact["models"]
    )
    assert {record["compound_id"] for record in artifact["records"]} == set(
        artifact["case_scope"]
    )

    durable_root = BENCHMARK_DIR / f"{RIGID_PREFLIGHT.stem}-raw"
    listed_paths = {evidence["path"] for evidence in artifact["raw_evidence"]}
    actual_paths = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in durable_root.rglob("*")
        if path.is_file()
    }
    assert listed_paths == actual_paths

    for evidence in artifact["raw_evidence"]:
        assert any(
            f"/{compound_id}/" in evidence["path"]
            or evidence["path"].endswith(f"--{compound_id}.json")
            for compound_id in artifact["case_scope"]
        )
        path = REPOSITORY_ROOT / evidence["path"]
        assert path.stat().st_size == evidence["size_bytes"]
        assert runner.sha256_file(path) == evidence["sha256"]


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


def test_calculator_evaluation_budget_fails_closed():
    runner = _load_runner()
    from ase import Atoms
    from ase.calculators.calculator import all_changes
    from ase.calculators.emt import EMT

    source = EMT()
    atoms = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    adapter = runner.CountingHartreeToEVCalculator(
        source,
        maximum_calculate_calls=1,
    )
    adapter.calculate(
        atoms=atoms,
        properties=["energy"],
        system_changes=all_changes,
    )
    assert adapter.calculate_calls == 1

    with pytest.raises(
        RuntimeError,
        match="per-branch calculator-evaluation budget exhausted",
    ):
        adapter.calculate(
            atoms=atoms,
            properties=["energy"],
            system_changes=all_changes,
        )


def test_v2_artifact_fails_closed_without_a_global_optimizer_policy():
    runner = _load_runner()
    artifact = runner.load_json(QUALIFICATION_V2)
    runner._validate_self_hash(artifact, name="optimizer qualification v2")

    assert artifact["protocol_id"] == (
        "maple-route1-multi-mlip-optimizer-qualification-v2"
    )
    assert artifact["status"] == "failed-closed-no-global-policy"
    assert artifact["selected_global_policy"] is None
    assert artifact["selection_evidence"] is None
    assert len(artifact["records"]) == 9
    assert {
        (record["candidate_id"], record["model"])
        for record in artifact["records"]
    } == {
        (candidate, model)
        for candidate in (
            "bfgs-linesearch",
            "lbfgs-linesearch",
            "fire2-abc",
        )
        for model in ("maceoff23m", "aimnet2", "ani2x")
    }

    summaries = {
        summary["candidate_id"]: summary
        for summary in artifact["candidate_summaries"]
    }
    assert set(summaries) == {
        "bfgs-linesearch",
        "lbfgs-linesearch",
        "fire2-abc",
    }
    assert all(
        summary["observed_branch_count"] == 18
        and summary["passed_branch_count"] == 12
        and summary["all_branches_passed"] is False
        for summary in summaries.values()
    )
    assert artifact["claim_boundary"][
        "optimizer_robustness_qualification_only"
    ]
    assert artifact["claim_boundary"]["hydration_accuracy_established"] is False
    assert artifact["claim_boundary"]["public_solvfe_eligible"] is False
    assert artifact["label_boundary"]["experimental_labels_read"] is False
    assert artifact["route_boundary"]["gas_phase_mm_energy"] is False
    assert artifact["route_boundary"]["hydration_label_residual"] is False


def test_v2_artifact_covers_all_durable_raw_evidence():
    runner = _load_runner()
    artifact = runner.load_json(QUALIFICATION_V2)
    durable_root = BENCHMARK_DIR / f"{QUALIFICATION_V2.stem}-raw"
    listed = {
        evidence["path"]: evidence for evidence in artifact["raw_evidence"]
    }
    actual = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in durable_root.rglob("*")
        if path.is_file()
    }

    assert set(listed) == actual
    for relative, evidence in listed.items():
        path = REPOSITORY_ROOT / relative
        assert path.stat().st_size == evidence["size_bytes"]
        assert runner.sha256_file(path) == evidence["sha256"]


def test_v2_failure_records_have_nonempty_finite_traces():
    runner = _load_runner()
    durable_root = BENCHMARK_DIR / f"{QUALIFICATION_V2.stem}-raw" / "records"
    failures = []

    for record_path in sorted(durable_root.glob("*.json")):
        record = runner.load_json(record_path)
        for phase in ("gas", "solution"):
            for branch in record["branches"][phase]:
                if branch["status"] != "failure":
                    continue
                failures.append(branch)
                assert branch["trace"]
                expected = all(
                    math.isfinite(float(row["energy_eV"]))
                    and math.isfinite(
                        float(row["maximum_force_eV_per_angstrom"])
                    )
                    for row in branch["trace"]
                )
                assert branch["finite_trace"] is expected

    assert failures


def test_route1_docs_close_v8_and_optimizer_v2_without_accuracy_claim():
    documents = [
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md",
        BENCHMARK_DIR / "README.md",
    ]
    normalized = " ".join(
        "\n".join(path.read_text(encoding="utf-8") for path in documents).split()
    )

    assert "failed-closed-no-global-policy" in normalized
    assert "no v9" in normalized
    assert "12/18" in normalized
    assert "per-model optimizer tuning" in normalized
    assert "successful branches only" in normalized
    assert "normative and executed v2 gate is `maximum_steps=1000`" in normalized
