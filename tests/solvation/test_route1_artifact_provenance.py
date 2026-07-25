from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"


@pytest.mark.parametrize(
    ("artifact_name", "script_name"),
    [
        (
            "route1-compatibility-methyl-hexanoate-2026-07-24.json",
            "run_route1_compatibility.py",
        ),
        (
            "route1-task-matrix-methyl-hexanoate-2026-07-24.json",
            "run_route1_task_matrix.py",
        ),
        (
            "route1-performance-methyl-hexanoate-2026-07-24.json",
            "run_route1_performance.py",
        ),
        (
            "route1-performance-methyl-hexanoate-cpu-2026-07-24.json",
            "run_route1_performance.py",
        ),
        (
            "route1-performance-methyl-hexanoate-ani2x-2026-07-24.json",
            "run_route1_performance.py",
        ),
        (
            "route1-performance-methyl-hexanoate-ani2x-cpu-2026-07-24.json",
            "run_route1_performance.py",
        ),
    ],
)
def test_route1_local_trace_has_self_hash_and_command_provenance(
    artifact_name,
    script_name,
):
    artifact_path = BENCHMARK_DIR / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    recorded_content_hash = artifact.pop("content_sha256")
    content_payload = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert hashlib.sha256(content_payload).hexdigest() == recorded_content_hash
    command = artifact["command_provenance"]
    expected_script = f"docs/implicit-solvation/benchmarks/{script_name}"
    script_path = REPOSITORY_ROOT / expected_script
    assert command["script"] == expected_script
    assert (
        command["script_sha256"] == hashlib.sha256(script_path.read_bytes()).hexdigest()
    )
    argument_payload = (
        json.dumps(
            command["arguments"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(argument_payload).hexdigest() == command["arguments_sha256"]
    assert Path(command["arguments"]["output"]).name == artifact_name
    assert set(command["environment_variables"]) == {
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
    }
