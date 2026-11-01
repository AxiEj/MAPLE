import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import validate_route1_algorithm_contracts as validator  # noqa: E402
from maple.function.calculator import set_calculator as runtime


def _artifact():
    source = {
        "git_head": "1" * 40,
        "files_aggregate_sha256": "2" * 64,
        "files_sha256": {},
        "worktree_status": [],
    }
    rows = {}
    for spec in validator.CANONICAL_ROWS:
        test_file = spec.test_node.split("::", 1)[0]
        expected = validator.expected_outcome(spec)
        rows[spec.row_id] = {
            "capability_key": spec.capability,
            "implementation_files_sha256": {
                spec.implementation: hashlib.sha256(
                    (ROOT / spec.implementation).read_bytes()
                ).hexdigest()
            },
            "test_file": test_file,
            "test_node": spec.test_node,
            "test_file_sha256": hashlib.sha256(
                (ROOT / test_file).read_bytes()
            ).hexdigest(),
            "fixture_identity": validator.fixture_identity(spec),
            "expected": expected,
            "observed": expected,
            "passed": True,
        }
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-advanced-algorithm-contracts",
        "source_before": source,
        "source_after": source,
        "capability_keys": sorted(validator.EXPECTED_CAPABILITIES),
        "rows": rows,
    }
    artifact["content_sha256"] = hashlib.sha256(
        validator.canonical_payload(artifact)
    ).hexdigest()
    return artifact, source


def _reseal(artifact):
    artifact["content_sha256"] = hashlib.sha256(
        validator.canonical_payload(artifact)
    ).hexdigest()


def _receipt(artifact, source):
    qualification_sources = {}
    for row in artifact["rows"].values():
        qualification_sources.update(row["implementation_files_sha256"])
        qualification_sources[row["test_file"]] = row["test_file_sha256"]
    return {
        "candidate_git_sha": source["git_head"],
        "qualification_source_aggregate_sha256": source["files_aggregate_sha256"],
        "qualification_sources": qualification_sources,
        "algorithm_contracts": validator.validate_artifact(
            artifact, expected_source_snapshot=source
        ),
    }


def test_complete_source_bound_algorithm_contract_replays():
    artifact, source = _artifact()
    result = validator.validate_artifact(artifact, expected_source_snapshot=source)
    assert set(result["capability_keys"]) == validator.EXPECTED_CAPABILITIES
    assert result["source_git_sha"] == source["git_head"]


@pytest.mark.parametrize(
    "mutation",
    ["nonexistent-node", "swapped-node", "outcome-mismatch", "fixture-drift"],
)
def test_resealed_forged_algorithm_contract_refuses(mutation):
    artifact, source = _artifact()
    first, second = list(artifact["rows"].values())[:2]
    if mutation == "nonexistent-node":
        first["test_node"] = first["test_file"] + "::does_not_exist"
    elif mutation == "swapped-node":
        first["test_node"] = second["test_node"]
        first["test_file"] = second["test_file"]
        first["test_file_sha256"] = second["test_file_sha256"]
    elif mutation == "outcome-mismatch":
        first["observed"] = "forged-pass"
    else:
        first["fixture_identity"] = {"kind": "arbitrary"}
    _reseal(artifact)
    with pytest.raises(ValueError):
        validator.validate_artifact(artifact, expected_source_snapshot=source)


def test_sealer_replay_executes_each_canonical_node_once(monkeypatch):
    artifact, source = _artifact()
    calls = []

    class Completed:
        returncode = 0
        stdout = "passed"
        stderr = ""

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(validator.subprocess, "run", run)
    validator.validate_artifact(artifact, expected_source_snapshot=source, replay=True)
    assert [call[0][-1] for call in calls] == list(
        dict.fromkeys(spec.test_node for spec in validator.CANONICAL_ROWS)
    )


def test_sealer_replay_rejects_a_failing_canonical_node(monkeypatch):
    artifact, source = _artifact()

    class Failed:
        returncode = 1
        stdout = "1 failed"
        stderr = ""

    monkeypatch.setattr(validator.subprocess, "run", lambda *_args, **_kwargs: Failed())
    with pytest.raises(ValueError, match="algorithm contract replay failed"):
        validator.validate_artifact(
            artifact, expected_source_snapshot=source, replay=True
        )


def test_runtime_accepts_only_canonical_source_bound_contract_rows():
    artifact, source = _artifact()
    receipt = _receipt(artifact, source)
    assert runtime._validated_algorithm_capabilities(receipt) == (
        validator.EXPECTED_CAPABILITIES
    )

    row_id = validator.CANONICAL_ROWS[0].row_id
    receipt["algorithm_contracts"]["rows"][row_id]["test_node"] += "_forged"
    with pytest.raises(ValueError, match="test binding"):
        runtime._validated_algorithm_capabilities(receipt)


@pytest.mark.parametrize(
    "mutation", ["missing-capability", "missing-hostile", "source", "implementation"]
)
def test_incomplete_or_drifted_algorithm_contract_refuses(mutation):
    artifact, source = _artifact()
    if mutation == "missing-capability":
        artifact["rows"].pop("capability.irc.gs")
    elif mutation == "missing-hostile":
        artifact["rows"].pop("hostile.false-ts")
    elif mutation == "source":
        artifact["source_after"] = dict(source, git_head="3" * 40)
    else:
        next(iter(artifact["rows"].values()))["implementation_files_sha256"] = {
            "maple/function/dispatcher/method_registry.py": "0" * 64
        }
    _reseal(artifact)
    with pytest.raises(ValueError):
        validator.validate_artifact(artifact, expected_source_snapshot=source)
