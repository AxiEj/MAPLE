from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_topology_determinism_protocol.json"
PERFORMANCE_PROTOCOL_PATH = BENCHMARK_DIR / "route1_performance_matrix_protocol.json"
ARTIFACT_PATH = BENCHMARK_DIR / "route1-topology-determinism-screen-2026-07-26.json"
SPEC = importlib.util.spec_from_file_location(
    "run_route1_topology_determinism_screen",
    BENCHMARK_DIR / "run_route1_topology_determinism_screen.py",
)
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)
FROZEN_CONTENT_SHA256 = (
    "416c945995f9cf3b6de92680314a9d32438fb5470348bd55c904ed25578e6acf"
)


def _load_artifact():
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _reseal(artifact):
    artifact["content_sha256"] = screen.artifact_content_sha256(artifact)
    return artifact


def test_protocol_pins_label_blind_large_case_topology_screen():
    protocol, fingerprint = screen.load_and_validate_protocol(PROTOCOL_PATH)
    performance_protocol = json.loads(
        PERFORMANCE_PROTOCOL_PATH.read_text(encoding="utf-8")
    )

    assert len(fingerprint) == 64
    assert protocol["execution"]["repeat_count"] == 20
    assert protocol["selection_boundary"] == {
        "experimental_hydration_labels_used": False,
        "accuracy_claimed": False,
        "force_field_quality_claimed": False,
        "scope": (
            "Topology construction reproducibility only; it does not validate "
            "chemistry or performance."
        ),
    }
    accepted = next(
        candidate
        for candidate in protocol["candidates"]
        if candidate["role"] == "accepted_large_candidate"
    )
    assert (
        accepted["compound_id"]
        == performance_protocol["matrix"]["cases"][-1]["case_id"]
    )
    for candidate in protocol["candidates"]:
        path = REPOSITORY_ROOT / candidate["mol2_relative_path"]
        assert screen.sha256_file(path) == candidate["mol2_sha256"]


def test_protocol_rejects_float_repeat_count(tmp_path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["execution"]["repeat_count"] = 20.0
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON integer"):
        screen.load_and_validate_protocol(path)


def test_selection_decision_requires_both_expected_roles():
    assert screen._selection_decision(
        [
            {"role": "accepted_large_candidate", "deterministic": True},
            {"role": "rejected_large_candidate", "deterministic": False},
        ]
    ) == {
        "accepted_candidate_passed": True,
        "rejected_candidate_failed": True,
        "selection_supported": True,
    }

    with pytest.raises(ValueError, match="both candidate roles"):
        screen._selection_decision(
            [{"role": "accepted_large_candidate", "deterministic": True}]
        )


def test_frozen_screen_artifact_validates_offline_and_supports_selection():
    artifact = _load_artifact()
    protocol, fingerprint = screen.load_and_validate_protocol(PROTOCOL_PATH)

    screen.validate_artifact(
        artifact,
        protocol_path=PROTOCOL_PATH,
        artifact_path=ARTIFACT_PATH,
        expected_content_sha256=FROZEN_CONTENT_SHA256,
        verify_external_executables=False,
    )
    assert artifact["protocol_fingerprint"] == fingerprint
    assert artifact["selection_boundary"] == protocol["selection_boundary"]
    assert artifact["decision"] == {
        "accepted_candidate_passed": True,
        "rejected_candidate_failed": True,
        "selection_supported": True,
    }
    records = {record["compound_id"]: record for record in artifact["records"]}
    assert records["mobley_8124669"]["variant_count"] == 2
    assert records["mobley_8124669"]["deterministic"] is False
    assert records["mobley_2078467"]["variant_count"] == 1
    assert records["mobley_2078467"]["deterministic"] is True
    for record in records.values():
        assert record["repeat_count"] == 20
        assert len(record["runs"]) == 20
        assert sum(record["frcmod_sha256_histogram"].values()) == 20
        assert set(record["representative_frcmod_by_sha256"]) == set(
            record["frcmod_sha256_histogram"]
        )
        for digest, content in record["representative_frcmod_by_sha256"].items():
            assert screen.sha256_bytes(content.encode("utf-8")) == digest


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("schema_version",), 1.0),
        (("command_provenance", "script_sha256"), "0" * 64),
        (("executable", "path"), "/tmp/parmchk2"),
        (
            ("executable", "wrapped_program", "path"),
            "/tmp/wrapped_progs/parmchk2",
        ),
        (
            ("charge_manifest", "path"),
            "docs/implicit-solvation/benchmarks/README.md",
        ),
        (("charge_manifest", "sha256"), "0" * 64),
        (
            ("records", 0, "input", "source_mol2"),
            "docs/implicit-solvation/benchmarks/README.md",
        ),
        (("records", 0, "input", "source_mol2_sha256"), "0" * 64),
        (("records", 0, "runs", 0, "repeat_index"), 0.9),
        (
            (
                "records",
                0,
                "frcmod_sha256_histogram",
                "b8128e78fa5dfc03299a33ddfeca544820d6583afca3c7ba78dcf8951306c53c",
            ),
            6.0,
        ),
        (("records", 0, "variant_count"), 1),
        (("decision", "selection_supported"), False),
    ],
)
def test_resealed_semantic_tampering_is_rejected(field_path, replacement):
    artifact = copy.deepcopy(_load_artifact())
    target = artifact
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = replacement
    _reseal(artifact)

    with pytest.raises((TypeError, ValueError)):
        screen.validate_artifact(
            artifact,
            protocol_path=PROTOCOL_PATH,
            artifact_path=ARTIFACT_PATH,
            verify_external_executables=False,
        )


def test_resealed_executable_hash_tampering_is_rejected_when_binary_is_available():
    artifact = copy.deepcopy(_load_artifact())
    executable_path = Path(artifact["executable"]["path"])
    if not executable_path.is_file():
        pytest.skip("Frozen parmchk2 executable is unavailable.")
    artifact["executable"]["sha256"] = "0" * 64
    _reseal(artifact)

    with pytest.raises(ValueError, match="unavailable or changed"):
        screen.validate_artifact(
            artifact,
            protocol_path=PROTOCOL_PATH,
            artifact_path=ARTIFACT_PATH,
        )


def test_benchmark_documentation_discloses_rejected_topology_variants():
    documentation = (BENCHMARK_DIR / "README.md").read_text(encoding="utf-8")

    assert "20 independent invocations produced two" in documentation
    assert "`8/20` and `12/20`" in documentation
    assert "Ibuprofen produced one frcmod SHA256 in" in documentation
    assert "`20/20`" in documentation
    assert "route1-topology-determinism-screen-2026-07-26.json" in documentation
