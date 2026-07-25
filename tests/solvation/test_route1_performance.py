from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
DOCUMENTATION_DIR = BENCHMARK_DIR.parent
SPEC = importlib.util.spec_from_file_location(
    "run_route1_performance",
    BENCHMARK_DIR / "run_route1_performance.py",
)
performance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(performance)


def test_timing_summary_reports_median_p95_and_mean():
    summary = performance.timing_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["n"] == 4
    assert summary["median_ms"] == 2.5
    assert summary["mean_ms"] == 2.5
    assert np.isclose(summary["p95_ms"], 3.85)


def test_timing_summary_rejects_empty_or_nonfinite_samples():
    with pytest.raises(ValueError, match="non-empty finite"):
        performance.timing_summary([])
    with pytest.raises(ValueError, match="non-empty finite"):
        performance.timing_summary([1.0, float("nan")])


def test_paired_benchmark_interleaves_and_counts_both_call_paths():
    calls = []

    first, second = performance.benchmark_paired_calls(
        lambda index: calls.append(("first", index)),
        lambda index: calls.append(("second", index)),
        samples=4,
        warmups=2,
        synchronize=lambda: None,
    )

    assert first["n"] == second["n"] == 4
    assert calls[:4] == [
        ("first", 0),
        ("second", 0),
        ("second", 1),
        ("first", 1),
    ]
    assert sum(name == "first" for name, _index in calls) == 6
    assert sum(name == "second" for name, _index in calls) == 6


def test_load_charge_vector_requires_one_finite_charge_per_atom(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"records":[{"compound_id":"case","am1bcc_charges_e":[-0.1,0.1]}]}',
        encoding="utf-8",
    )

    charges, record = performance.load_charge_vector(
        manifest,
        "case",
        atom_count=2,
    )
    assert np.allclose(charges, [-0.1, 0.1])
    assert record["compound_id"] == "case"

    with pytest.raises(ValueError, match="one finite AM1-BCC charge"):
        performance.load_charge_vector(manifest, "case", atom_count=3)


def test_gpu_trace_limits_mm_ratio_to_a_nonclaim_observation():
    trace = json.loads(
        (
            BENCHMARK_DIR / "route1-performance-methyl-hexanoate-2026-07-24.json"
        ).read_text(encoding="utf-8")
    )

    assert trace["artifact_type"] == "route1-warm-energy-force-local-trace"
    assert trace["task"] == "warm in-process energy+forces"
    assert trace["atom_count"] == 23
    assert (
        0.0
        < trace["ratios"]["reference_paired_combined_minus_gas_fraction_observed"]
        < 0.1
    )
    assert trace["ratios"]["reference_correction_time_fraction_vs_gas_mlip"] < 0.1
    observation = trace["mm_baseline_observations"]["reference"]
    assert observation["same_host_local_observation"] is False
    assert observation["not_production_throughput_comparison"] is True
    assert "not a local MM comparison or speed claim" in observation["interpretation"]
    assert trace["skipped_platforms"] == ["CUDA"]
    assert any("locally traceable" in item for item in trace["limitations"])


def test_cpu_trace_records_only_the_predeclared_same_resource_comparison():
    trace = json.loads(
        (
            BENCHMARK_DIR / "route1-performance-methyl-hexanoate-cpu-2026-07-24.json"
        ).read_text(encoding="utf-8")
    )

    observation = trace["mm_baseline_observations"]["reference"]
    assert trace["environment"]["device"] == "cpu"
    assert trace["environment"]["torch_num_threads"] == 1
    assert observation["same_host_local_observation"] is True
    assert observation["not_production_throughput_comparison"] is True
    assert observation["resource_policy"] == (
        "same-host CPU; PyTorch one thread; OpenMM Reference single-thread"
    )
    assert (
        "not a production-MM or general throughput comparison"
        in observation["interpretation"]
    )
    assert observation["mlip_plus_gb_over_mm_plus_gb"] > 1.0
    assert trace["ratios"]["reference_correction_time_fraction_vs_gas_mlip"] < 0.02
    paired = trace["ratios"]["reference_paired_combined_minus_gas_fraction_observed"]
    interpretation = trace["solvent_timing_observations"]["reference"]["interpretation"]
    if paired < 0:
        assert "do not interpret" in interpretation
    else:
        assert "positive local paired observation" in interpretation


@pytest.mark.parametrize(
    ("filename", "device", "same_host_local_observation"),
    [
        (
            "route1-performance-methyl-hexanoate-ani2x-2026-07-24.json",
            "cuda",
            False,
        ),
        (
            "route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json",
            "cpu",
            True,
        ),
    ],
)
def test_ani2x_traces_keep_solvent_overhead_and_mm_comparison_separate(
    filename,
    device,
    same_host_local_observation,
):
    trace = json.loads((BENCHMARK_DIR / filename).read_text(encoding="utf-8"))

    assert trace["route1"]["gas_model"] == "ani2x"
    assert trace["environment"]["device"] == device
    assert (
        trace["mm_baseline_observations"]["reference"]["same_host_local_observation"]
        is same_host_local_observation
    )
    assert trace["mm_baseline_observations"]["reference"][
        "not_production_throughput_comparison"
    ]
    assert 0.0 < trace["ratios"]["reference_correction_time_fraction_vs_gas_mlip"] < 0.3
    assert (
        0.0
        < trace["ratios"]["reference_paired_combined_minus_gas_fraction_observed"]
        < 0.3
    )
    if same_host_local_observation:
        assert trace["environment"]["torch_num_threads"] == 1
        assert (
            trace["mm_baseline_observations"]["reference"][
                "mlip_plus_gb_over_mm_plus_gb"
            ]
            > 1.0
        )


def test_performance_documentation_numbers_are_bound_to_frozen_artifacts():
    traces = {
        "mace_gpu": json.loads(
            (
                BENCHMARK_DIR / "route1-performance-methyl-hexanoate-2026-07-24.json"
            ).read_text(encoding="utf-8")
        ),
        "mace_cpu": json.loads(
            (
                BENCHMARK_DIR
                / "route1-performance-methyl-hexanoate-cpu-2026-07-24.json"
            ).read_text(encoding="utf-8")
        ),
        "ani_gpu": json.loads(
            (
                BENCHMARK_DIR
                / "route1-performance-methyl-hexanoate-ani2x-2026-07-24.json"
            ).read_text(encoding="utf-8")
        ),
        "ani_cpu": json.loads(
            (
                BENCHMARK_DIR
                / "route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json"
            ).read_text(encoding="utf-8")
        ),
    }
    documents = " ".join(
        path.read_text(encoding="utf-8")
        for path in (
            DOCUMENTATION_DIR / "ROUTE1_PRODUCT_SPEC.md",
            DOCUMENTATION_DIR / "VALIDATION_STATUS.md",
            BENCHMARK_DIR / "README.md",
        )
    )

    for trace in traces.values():
        paired = trace["ratios"][
            "reference_paired_combined_minus_gas_fraction_observed"
        ]
        correction = trace["ratios"]["reference_correction_time_fraction_vs_gas_mlip"]
        assert f"{100.0 * paired:.2f}%" in documents
        assert f"{100.0 * correction:.2f}%" in documents
        observation = trace["mm_baseline_observations"]["reference"]
        assert observation["not_production_throughput_comparison"]
        assert (
            "not a production-MM or general throughput comparison"
            in trace["claim_scope"]
        )

    for key in ("mace_cpu", "ani_cpu"):
        ratio = traces[key]["mm_baseline_observations"]["reference"][
            "mlip_plus_gb_over_mm_plus_gb"
        ]
        assert f"{ratio:.0f}x" in documents
