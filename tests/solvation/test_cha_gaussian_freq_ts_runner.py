"""Fail-closed contracts for the analytic Gaussian-CHA workflow runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "docs/implicit-solvation/benchmarks/run_cha_gaussian_freq_ts.py"


def _runner():
    spec = importlib.util.spec_from_file_location("cha_freq_ts_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_externally_pinned_complete_and_has_exact_15_roster():
    runner = _runner()
    protocol, digest = runner.load_protocol()
    assert digest == runner.PROTOCOL_SHA256
    assert protocol["approved_plan_sha256"] == runner.PLAN_SHA256
    assert protocol["production_order"] == 64
    assert protocol["profile_id"] == runner.PROFILE_ID
    assert len(protocol["cases"]) == 15
    assert [case["row_id"] for case in protocol["cases"]] == [
        f"center-{center:02d}-sigma-{width:02d}"
        for center in range(5)
        for width in range(3)
    ]
    assert protocol["prfo_options"]["project_rigid_modes"] is True
    assert protocol["prfo_options"]["max_iter"] == 1
    assert protocol["dimer_options"]["use_hvp"] is True
    assert protocol["dimer_options"]["n_init"] == "given"
    assert protocol["dimer_options"]["max_iter"] == 1
    assert protocol["dimer_options"]["rot_max_iter"] == 1


@pytest.mark.parametrize(
    "workflow,mutation,match",
    [
        ("prfo", {"project_rigid_mode": True}, "unknown"),
        ("prfo", {"project_rigid_modes": False}, "project_rigid_modes"),
        ("dimer", {"use_hvp": False}, "use_hvp"),
        ("dimer", {"n_init": "force"}, "n_init"),
        ("frequency", {"method": "nonmw"}, "method"),
    ],
)
def test_closed_workflow_options_reject_typos_and_nonanalytic_substitutions(
    workflow, mutation, match
):
    runner = _runner()
    protocol, _ = runner.load_protocol()
    key = {
        "frequency": "frequency_options",
        "prfo": "prfo_options",
        "dimer": "dimer_options",
    }[workflow]
    values = dict(protocol[key])
    values.update(mutation)
    with pytest.raises(ValueError, match=match):
        runner.validate_workflow_options(workflow, values, protocol)


def test_effective_prfo_and_dimer_parameters_must_match_constructed_objects():
    runner = _runner()
    protocol, _ = runner.load_protocol()
    prfo = SimpleNamespace(**protocol["prfo_options"])
    runner.assert_effective_workflow_params("prfo", prfo, protocol["prfo_options"])
    prfo.max_iter = 2
    with pytest.raises(ValueError, match="effective prfo"):
        runner.assert_effective_workflow_params("prfo", prfo, protocol["prfo_options"])

    requested = {**protocol["dimer_options"], "n_given": [1.0] + [0.0] * 8}
    dimer = SimpleNamespace(**requested)
    runner.assert_effective_workflow_params("dimer", dimer, requested)
    dimer.use_hvp = False
    with pytest.raises(ValueError, match="effective dimer"):
        runner.assert_effective_workflow_params("dimer", dimer, requested)


def test_campaign_boundary_is_unique_direct_child_only(tmp_path):
    runner = _runner()
    root = runner.OUTPUT_ROOT
    assert not runner.output_allowed(root)
    assert runner.output_allowed(root / "campaign-20261002T010203Z")
    assert not runner.output_allowed(root / "campaign-not-utc")
    assert not runner.output_allowed(root / "planning")
    assert not runner.output_allowed(root / "campaign-20261002T010203Z" / "nested")


def test_source_snapshot_hashes_the_current_candidate_prfo():
    runner = _runner()
    path = "maple/function/dispatcher/ts/algorithm/PRFO.py"
    snapshot = runner._source_snapshot()
    assert snapshot[path] == runner.sha256_file(ROOT / path)


def test_analytic_center_uses_one_composed_energy_force_graph(monkeypatch):
    runner = _runner()
    import sys

    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    protocol, _ = runner.load_protocol()
    if not (runner.ROOT / protocol["topology_path"]).is_file():
        pytest.skip("private pinned water inputs are not installed")
    _, topology = runner.load_topology(protocol)
    atoms, correction, calculator = runner.fresh_analytic_system(
        protocol["cases"][0], topology
    )
    trace = runner.GraphTrace()
    with trace:
        runner._analytic_center(atoms, correction, calculator, [], trace)
    record = trace.receipt()
    assert record["annotations_match_observations"] is True
    composed = next(
        x for x in record["operations"] if x["operation_id"] == "center-composed-ef"
    )
    assert composed["gas_graphs"] == composed["solvent_graphs"] == 1


def test_dimer_call_budget_uses_counter_delta_and_rejects_dense_or_fd_markers():
    runner = _runner()
    runner.validate_dimer_execution(
        before={
            "direct_hvp_calls": 3,
            "gas_dense_hessian_calls": 1,
            "solvent_dense_hessian_calls": 0,
        },
        after={
            "direct_hvp_calls": 11,
            "gas_dense_hessian_calls": 1,
            "solvent_dense_hessian_calls": 0,
        },
        maximum_hvp_calls=8,
        derivative_mode="hvp",
    )
    with pytest.raises(ValueError, match="HVP call budget"):
        runner.validate_dimer_execution(
            before={
                "direct_hvp_calls": 0,
                "gas_dense_hessian_calls": 0,
                "solvent_dense_hessian_calls": 0,
            },
            after={
                "direct_hvp_calls": 9,
                "gas_dense_hessian_calls": 0,
                "solvent_dense_hessian_calls": 0,
            },
            maximum_hvp_calls=8,
            derivative_mode="hvp",
        )
    with pytest.raises(ValueError, match="dense Hessian|finite-difference"):
        runner.validate_dimer_execution(
            before={
                "direct_hvp_calls": 0,
                "gas_dense_hessian_calls": 0,
                "solvent_dense_hessian_calls": 0,
            },
            after={
                "direct_hvp_calls": 1,
                "gas_dense_hessian_calls": 1,
                "solvent_dense_hessian_calls": 0,
            },
            maximum_hvp_calls=8,
            derivative_mode="hvp",
        )


def test_validation_recomputes_rows_and_does_not_trust_stored_pass(
    monkeypatch, tmp_path
):
    runner = _runner()
    protocol, _ = runner.load_protocol()
    prereg = {
        **runner._dimer_call_sites(),
        "rows": protocol["cases"],
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "source_identity_sha256": "s" * 64,
        "input_identity_sha256": "i" * 64,
        "source_before": {
            path: "s" * 64
            for path in (
                "runner.py",
                "maple/function/calculator/mace/_mace_cha_analytic_calculator.py",
                "maple/function/calculator/extra_correction/implicit/gaussian_cha_analytic_correction.py",
                "maple/function/dispatcher/ts/algorithm/dimer.py",
            )
        },
        "input_before": {"protocol": "i" * 64},
    }
    for folder in ("derivatives", "freq", "prfo", "dimer", "resource"):
        (tmp_path / folder).mkdir()
    for case in protocol["cases"]:
        binding = {
            "case_identity": case,
            "preregistration_sha256": "p" * 64,
            "protocol_sha256": runner.PROTOCOL_SHA256,
            "source_identity_sha256": prereg["source_identity_sha256"],
            "input_identity_sha256": prereg["input_identity_sha256"],
        }
        for operation in ("energy-force", "dense-h", "direct-hvp"):
            (tmp_path / "resource" / f"{case['row_id']}-{operation}.json").write_text(
                json.dumps(
                    runner.seal(
                        {
                            **binding,
                            "phase": "resource",
                            "operation": operation,
                            "status": "FAILED",
                        }
                    )
                )
            )
        (tmp_path / "derivatives" / f"{case['row_id']}.json").write_text(
            json.dumps(
                runner.seal({**binding, "phase": "derivatives", "stored_passed": True})
            )
        )
        for phase in ("freq", "prfo", "dimer"):
            (tmp_path / phase / f"{case['row_id']}.json").write_text(
                json.dumps(
                    runner.seal(
                        {
                            **binding,
                            "phase": phase,
                            "status": "FAILED",
                            "fresh_forces_hartree_per_angstrom": np.ones(
                                (3, 3)
                            ).tolist(),
                        }
                    )
                )
            )

    class Validator:
        @staticmethod
        def validate_freq_ts_row(receipt, protocol, row):
            return {
                "row_id": row["row_id"],
                "passed": False,
                "status": "SECOND_DERIVATIVE_UNRESOLVED",
            }

        @staticmethod
        def summarize_freq_ts_rows(rows, expected):
            return {
                "row_count": len(rows),
                "passed_count": 0,
                "status": "SECOND_DERIVATIVE_UNRESOLVED",
            }

    monkeypatch.setattr(
        runner, "load_preregistration", lambda output, pin: (prereg, "p" * 64)
    )
    monkeypatch.setattr(runner, "load_validation_helper", lambda: Validator)
    monkeypatch.setattr(runner, "_source_snapshot", lambda: prereg["source_before"])
    monkeypatch.setattr(
        runner, "_input_snapshot", lambda protocol: prereg["input_before"]
    )
    result = runner.validate_campaign(tmp_path, "p" * 64, write_output=False)
    assert result["status"] != "ANALYTIC_FREQ_VALIDATED_TS_INTERFACES_ONLY"
    assert result["second_derivative"]["passed_count"] == 0


def test_resource_measurement_requires_a_single_explicit_case(monkeypatch, tmp_path):
    runner = _runner()
    monkeypatch.setattr(runner, "load_preregistration", lambda *args: ({}, "p" * 64))
    monkeypatch.setattr(runner, "load_topology", lambda *args: ({}, None))
    with pytest.raises(ValueError, match="single explicit"):
        runner.run_campaign(tmp_path, "p" * 64, phase="resource", operation="dense-h")


def test_kernel_peak_rss_parser_rejects_missing_or_wrong_units():
    runner = _runner()
    assert runner._parse_rss_status("VmRSS:\t10 kB\nVmHWM:\t12 kB\n") == {
        "rss_bytes": 10 * 1024,
        "peak_bytes": 12 * 1024,
    }
    for text in (
        "VmRSS: 10 kB",
        "VmRSS: 10 MB\nVmHWM: 12 MB",
        "VmRSS: -1 kB\nVmHWM: 0 kB",
    ):
        with pytest.raises(ValueError):
            runner._parse_rss_status(text)


def test_graph_trace_observes_failed_calls_instead_of_trusting_declared_counts(
    monkeypatch,
):
    import sys

    runner = _runner()

    def failing_graph():
        raise ValueError("graph failed")

    monkeypatch.setattr(
        runner, "_graph_code_map", lambda: {failing_graph.__code__: "solvent_graphs"}
    )
    monkeypatch.setattr(runner, "_graph_seams", lambda: {})
    prior = sys.getprofile()
    trace = runner.GraphTrace()
    with trace:
        trace.attempt("failure", solvent_graphs=99)
        with pytest.raises(ValueError, match="graph failed"):
            failing_graph()
    assert sys.getprofile() is prior
    receipt = trace.receipt()
    assert receipt["solvent_graphs"] == 1
    assert receipt["annotations_match_observations"] is False


def test_partial_numeric_samples_survive_a_later_failure():
    runner = _runner()
    trace = runner.GraphTrace()
    with trace.sample("force", {"point": 1}) as sample:
        sample["force"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="typed failure"):
        with trace.sample("force", {"point": 2}):
            raise ValueError("typed failure")
    assert [item["status"] for item in trace.partial_samples] == ["RETURNED", "FAILED"]
    assert trace.partial_samples[0]["force"] == [1.0, 2.0, 3.0]


def test_actual_producer_matches_raw_workflow_validator(tmp_path):
    runner = _runner()
    protocol, _ = runner.load_protocol()
    if not (runner.ROOT / protocol["topology_path"]).is_file():
        pytest.skip("private pinned water inputs are not installed")
    _, topology = runner.load_topology(protocol)
    runner.load_validation_helper()
    import importlib

    workflow = importlib.import_module("cha_gaussian_workflow_validation")
    protocol = {
        **protocol,
        **runner._dimer_call_sites(),
        "workflow_sources": {
            name: runner.sha256_file(ROOT / path)
            for name, path in (
                (
                    "gas_source_sha256",
                    "maple/function/calculator/mace/_mace_cha_analytic_calculator.py",
                ),
                (
                    "solvent_source_sha256",
                    "maple/function/calculator/extra_correction/implicit/gaussian_cha_analytic_correction.py",
                ),
                (
                    "dimer_source_sha256",
                    "maple/function/dispatcher/ts/algorithm/dimer.py",
                ),
            )
        },
    }
    case = protocol["cases"][0]
    atoms, correction, calculator = runner.fresh_analytic_system(case, topology)
    forces = atoms.get_forces()
    center = {
        "forces_hartree_per_angstrom": forces.tolist(),
        "energy_hartree": float(atoms.get_potential_energy(force_consistent=True)),
        "hessian_hartree_per_angstrom2": calculator.get_hessian(atoms).tolist(),
    }
    for phase, producer in (
        ("freq", runner.run_frequency_workflow),
        ("prfo", runner.run_prfo_workflow),
        ("dimer", runner.run_dimer_workflow),
    ):
        raw = producer(case, topology, tmp_path, protocol)
        verdict = workflow.validate_workflow_receipt(raw, phase, case, protocol, center)
        (tmp_path / f"{phase}-raw.json").write_text(
            json.dumps(runner.json_safe(raw), indent=2) + "\n"
        )
        assert verdict["passed"], verdict


def test_actual_derivative_producer_matches_validator(tmp_path):
    runner = _runner()
    protocol, _ = runner.load_protocol()
    if not (runner.ROOT / protocol["topology_path"]).is_file():
        pytest.skip("private pinned water inputs are not installed")
    mapping, topology = runner.load_topology(protocol)
    helper = runner.load_validation_helper()
    source, inputs = runner._source_snapshot(), runner._input_snapshot(protocol)
    prereg = {
        "source_identity_sha256": runner._identity(source),
        "input_identity_sha256": runner._identity(inputs),
        "development_only": True,
    }
    case = protocol["cases"][0]
    trace = runner.GraphTrace()
    try:
        raw = runner.produce_derivative_receipt(
            case, mapping, topology, prereg, runner._identity(prereg), protocol, trace
        )
        verdict = helper.validate_freq_ts_row(raw, {**protocol, **prereg}, case)
        (tmp_path / "raw.json").write_text(
            json.dumps(runner.json_safe(raw), indent=2) + "\n"
        )
        (tmp_path / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        assert verdict["passed"], verdict
    finally:
        (tmp_path / "partial-samples.json").write_text(
            json.dumps(
                runner.json_safe(
                    {"samples": trace.partial_samples, "counters": trace.receipt()}
                ),
                indent=2,
            )
            + "\n"
        )
