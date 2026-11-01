import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import validate_route1_algorithm_contracts as validator  # noqa: E402


def _artifact():
    implementation = "maple/function/dispatcher/method_registry.py"
    test_file = "tests/test_method_registry.py"
    implementation_sha = hashlib.sha256((ROOT / implementation).read_bytes()).hexdigest()
    test_sha = hashlib.sha256((ROOT / test_file).read_bytes()).hexdigest()
    source = {
        "git_head": "1" * 40,
        "files_aggregate_sha256": "2" * 64,
        "files_sha256": {},
        "worktree_status": [],
    }
    rows = {}
    for capability in sorted(validator.EXPECTED_CAPABILITIES):
        rows[f"capability.{capability}"] = {
            "capability_key": capability,
            "implementation_files_sha256": {implementation: implementation_sha},
            "test_file": test_file,
            "test_node": test_file + "::test_registry_covers_public_methods_and_exact_dataclass_fields",
            "test_file_sha256": test_sha,
            "fixture_identity": {"kind": "synthetic"},
            "expected": "pass",
            "observed": "pass",
            "passed": True,
        }
    for row_id in validator.HOSTILE_ROW_IDS:
        rows[row_id] = dict(
            rows["capability.irc.gs"], capability_key=None,
        )
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


def test_complete_source_bound_algorithm_contract_replays():
    artifact, source = _artifact()
    result = validator.validate_artifact(artifact, expected_source_snapshot=source)
    assert set(result["capability_keys"]) == validator.EXPECTED_CAPABILITIES
    assert result["source_git_sha"] == source["git_head"]


@pytest.mark.parametrize("mutation", ["missing-capability", "missing-hostile", "source", "implementation"])
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
