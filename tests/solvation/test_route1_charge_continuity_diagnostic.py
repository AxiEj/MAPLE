from __future__ import annotations

import argparse
import importlib.util
import copy
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"

RUNNER_PATH = BENCHMARK_DIR / "run_route1_charge_continuity_diagnostic.py"
PROTOCOL_PATH = BENCHMARK_DIR / "route1_charge_continuity_diagnostic_protocol_v1.json"
SOURCE_MANIFEST_PATH = BENCHMARK_DIR / "route1_freesolv_reserve_source_manifest.json"
ENERGY_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-energy-2026-07-25.json"
SCORE_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-score-2026-07-25.json"
PAIR_PROTOCOL_PATH = BENCHMARK_DIR / "route1_charge_continuity_protocol_v1.json"
PAIR_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-charge-continuity-pairs-2026-07-29.json"
FROZEN_OUTPUT = BENCHMARK_DIR / "route1-freesolv-reserve-charge-continuity-diagnostic-2026-07-29.json"

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "route1_charge_continuity_diagnostic",
    RUNNER_PATH,
)
assert RUNNER_SPEC is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def _run_diagnostic(
    tmp_path: Path,
    *,
    protocol: Path = PROTOCOL_PATH,
    source_manifest: Path = SOURCE_MANIFEST_PATH,
    energy_artifact: Path = ENERGY_PATH,
    score_artifact: Path = SCORE_PATH,
    pair_protocol: Path = PAIR_PROTOCOL_PATH,
    pair_artifact: Path = PAIR_PATH,
    output_name: str = "route1-freesolv-reserve-charge-continuity-diagnostic-test.json",
):
    output = tmp_path / output_name
    args = argparse.Namespace(
        protocol=protocol,
        source_manifest=source_manifest,
        energy_artifact=energy_artifact,
        score_artifact=score_artifact,
        pair_protocol=pair_protocol,
        pair_artifact=pair_artifact,
        output=output,
    )
    return runner.run(args)


def _json_copy_with_mutation(path: Path, tmp_path: Path, transform) -> Path:
    data = runner.core.load_json(path)
    mutate_target = copy.deepcopy(data)
    transform(mutate_target)

    if (
        isinstance(mutate_target, dict)
        and "artifact_type" in mutate_target
        and "schema_version" in mutate_target
    ):
        mutate_target["content_sha256"] = (
            runner.core.artifact_content_sha256(mutate_target)
        )

    out = tmp_path / path.name
    runner.core.write_json_atomic(out, mutate_target)
    return out


def _aligned_protocol(
    tmp_path: Path,
    *,
    source_manifest: Path = SOURCE_MANIFEST_PATH,
    energy_artifact: Path = ENERGY_PATH,
    score_artifact: Path = SCORE_PATH,
    pair_protocol: Path = PAIR_PROTOCOL_PATH,
    pair_artifact: Path = PAIR_PATH,
) -> Path:
    protocol = runner.core.load_json(PROTOCOL_PATH)
    source_manifest_hash = runner.core.sha256_file(source_manifest)
    energy_object = runner.core.load_json(energy_artifact)
    score_object = runner.core.load_json(score_artifact)
    pair_artifact_object = runner.core.load_json(pair_artifact)

    protocol["source_evidence"] = {
        **protocol["source_evidence"],
        "source_manifest_sha256": source_manifest_hash,
        "energy_artifact_sha256": runner.core.sha256_file(energy_artifact),
        "energy_artifact_content_sha256": runner.core.artifact_content_sha256(
            energy_object
        ),
        "score_artifact_sha256": runner.core.sha256_file(score_artifact),
        "score_artifact_content_sha256": runner.core.artifact_content_sha256(
            score_object
        ),
        "pair_protocol_sha256": runner.core.sha256_file(pair_protocol),
        "pair_artifact_sha256": runner.core.sha256_file(pair_artifact),
        "pair_artifact_content_sha256": runner.core.artifact_content_sha256(
            pair_artifact_object
        ),
    }

    output = tmp_path / "route1_charge_continuity_diagnostic_protocol_v1.json"
    runner.core.write_json_atomic(output, protocol)
    return output


def _contains_forbidden_fields(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in forbidden:
                return True
            if _contains_forbidden_fields(nested, forbidden):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_fields(item, forbidden) for item in value)
    return False


def test_charge_continuity_diagnostic_reproducible_with_frozen_evidence(tmp_path: Path) -> None:
    protocol = _aligned_protocol(tmp_path)
    artifact = _run_diagnostic(
        tmp_path,
        protocol=protocol,
    )
    expected = runner.core.load_json(FROZEN_OUTPUT)

    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == expected["artifact_type"]
    assert artifact["case_count"] == expected["case_count"] == 116
    assert artifact["candidate_pair_count"] == expected["candidate_pair_count"] == 6670
    assert artifact["evaluable_pair_count"] == expected["evaluable_pair_count"]
    assert artifact["component_count"] == expected["component_count"] == 5
    assert artifact["diagnostic_summaries"].keys() == expected["diagnostic_summaries"].keys()

    all_summary = artifact["diagnostic_summaries"]["all_experimentally_distinct"]
    expected_all = expected["diagnostic_summaries"]["all_experimentally_distinct"]
    assert all_summary["pair_count"] == expected_all["pair_count"]
    assert all_summary["methods"]["am1bcc_obc2_ace"]["forced_sign_accuracy"] == pytest.approx(
        expected_all["methods"]["am1bcc_obc2_ace"]["forced_sign_accuracy"]
    )
    assert all_summary["methods"]["am1bcc_chagb_pbsa_cavity_dispersion"]["forced_sign_accuracy"] == pytest.approx(
        expected_all["methods"]["am1bcc_chagb_pbsa_cavity_dispersion"]["forced_sign_accuracy"]
    )
    assert (
        artifact["diagnostic_summaries"]["fixed_abs_delta_gte_0.5"]["pair_count"]
        == expected["diagnostic_summaries"]["fixed_abs_delta_gte_0.5"]["pair_count"]
    )
    assert (
        artifact["diagnostic_summaries"]["fixed_abs_delta_gte_1.0"]["pair_count"]
        == expected["diagnostic_summaries"]["fixed_abs_delta_gte_1.0"]["pair_count"]
    )
    assert (
        artifact["diagnostic_summaries"]["fixed_abs_delta_gte_2.0"]["pair_count"]
        == expected["diagnostic_summaries"]["fixed_abs_delta_gte_2.0"]["pair_count"]
    )
    assert (
        artifact["diagnostic_summaries"]["uncertainty_z_1.96"]["pair_count"]
        == expected["diagnostic_summaries"]["uncertainty_z_1.96"]["pair_count"]
    )

    for gate_name in (
        "fixed_abs_delta_gte_0.5",
        "fixed_abs_delta_gte_1.0",
        "fixed_abs_delta_gte_2.0",
        "uncertainty_z_1.96",
    ):
        summary = artifact["diagnostic_summaries"][gate_name]
        assert summary["pair_count"] == summary["endpoint_correctness_cross_tab"]["both_correct"] + summary[
            "endpoint_correctness_cross_tab"
        ]["both_wrong_or_tied"] + summary["endpoint_correctness_cross_tab"][
            "baseline_only_correct"
        ] + summary["endpoint_correctness_cross_tab"]["candidate_only_correct"]



def test_charge_continuity_diagnostic_is_deterministic_and_self_sealed(tmp_path: Path) -> None:
    protocol = _aligned_protocol(tmp_path)
    first = _run_diagnostic(
        tmp_path,
        protocol=protocol,
        output_name="route1-freesolv-reserve-charge-continuity-diagnostic-test.json",
    )
    second = _run_diagnostic(
        tmp_path,
        protocol=protocol,
        output_name="route1-freesolv-reserve-charge-continuity-diagnostic-test.json",
    )

    assert first["content_sha256"] == second["content_sha256"]
    assert first["content_sha256"] == runner.core.artifact_content_sha256(first)
    assert first["evaluable_pair_count"] == second["evaluable_pair_count"]


def test_charge_continuity_diagnostic_forbids_selection_and_certification(
    tmp_path: Path,
) -> None:
    artifact = _run_diagnostic(
        tmp_path,
        protocol=_aligned_protocol(tmp_path),
    )
    decision = artifact["decision"]

    assert decision["status"] == "diagnostic_signal_only_insufficient_for_threshold_or_certification"
    assert decision["endpoint_selection_allowed"] is False
    assert decision["charge_method_selection_allowed"] is False
    assert decision["energy_correction_allowed"] is False
    assert decision["abstention_threshold_selection_allowed"] is False
    assert decision["certified_ranking_available"] is False


def test_charge_continuity_diagnostic_excludes_per_pair_experimental_labels_and_inference_fields(
    tmp_path: Path,
) -> None:
    artifact = _run_diagnostic(
        tmp_path,
        protocol=_aligned_protocol(tmp_path),
    )

    assert not _contains_forbidden_fields(
        artifact, set(runner._FORBIDDEN_OUTPUT_FIELDS)
    )


def test_charge_continuity_diagnostic_chain_tamper_source_manifest_hash_is_rejected(tmp_path: Path) -> None:
    tampered_source = _json_copy_with_mutation(
        SOURCE_MANIFEST_PATH,
        tmp_path,
        lambda data: data.__setitem__("case_count", data["case_count"] + 1),
    )

    with pytest.raises(
        ValueError, match="Frozen source-manifest hash mismatch."
    ):
        _run_diagnostic(
            tmp_path,
            source_manifest=tampered_source,
            output_name="diagnostic_with_tampered_source.json",
        )


def test_charge_continuity_diagnostic_chain_tamper_energy_hash_is_rejected(tmp_path: Path) -> None:
    def _mutate_energy(record_data):
        record_data["records"][0]["predictions_kcal_mol"]["am1bcc_obc2_ace"] = (
            record_data["records"][0]["predictions_kcal_mol"]["am1bcc_obc2_ace"] + 1.0
        )

    tampered_energy = _json_copy_with_mutation(
        ENERGY_PATH,
        tmp_path,
        _mutate_energy,
    )

    with pytest.raises(
        ValueError, match="Frozen energy-artifact file hash mismatch."
    ):
        _run_diagnostic(
            tmp_path,
            protocol=_aligned_protocol(tmp_path),
            energy_artifact=tampered_energy,
            output_name="diagnostic_with_tampered_energy.json",
        )


def test_charge_continuity_diagnostic_chain_score_energy_mismatch_is_detected(tmp_path: Path) -> None:
    def _mutate_score(record_data):
        record_data["records"][0]["predictions_kcal_mol"]["am1bcc_obc2_ace"] = (
            record_data["records"][0]["predictions_kcal_mol"]["am1bcc_obc2_ace"] + 1.0
        )

    tampered_score = _json_copy_with_mutation(
        SCORE_PATH,
        tmp_path,
        _mutate_score,
    )
    protocol = _aligned_protocol(
        tmp_path,
        score_artifact=tampered_score,
    )

    with pytest.raises(
        ValueError, match="Score/energy prediction mismatch for"
    ):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            score_artifact=tampered_score,
            output_name="diagnostic_with_score_energy_mismatch.json",
        )


def test_charge_continuity_diagnostic_chain_tamper_pair_protocol_hash_is_rejected(tmp_path: Path) -> None:
    tampered_pair_protocol = _json_copy_with_mutation(
        PAIR_PROTOCOL_PATH,
        tmp_path,
        lambda data: data.__setitem__("protocol_id", "maple-route1-charge-continuity-pairs-v1-tampered"),
    )

    with pytest.raises(
        ValueError, match="Frozen pair-protocol hash mismatch."
    ):
        _run_diagnostic(
            tmp_path,
            protocol=_aligned_protocol(tmp_path),
            pair_protocol=tampered_pair_protocol,
            output_name="diagnostic_with_tampered_pair_protocol.json",
        )


def test_charge_continuity_diagnostic_chain_tamper_pair_artifact_hash_is_rejected(tmp_path: Path) -> None:
    tampered_pair_artifact = _json_copy_with_mutation(
        PAIR_PATH,
        tmp_path,
        lambda data: data.__setitem__(
            "evaluable_pair_count", data["evaluable_pair_count"] + 1
        ),
    )

    with pytest.raises(
        ValueError, match="Frozen pair-artifact file hash mismatch."
    ):
        _run_diagnostic(
            tmp_path,
            protocol=_aligned_protocol(tmp_path),
            pair_artifact=tampered_pair_artifact,
            output_name="diagnostic_with_tampered_pair_artifact.json",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data.__setitem__(
                "component_count", data["component_count"] + 1
            ),
            "Pair-artifact component count mismatch",
        ),
        (
            lambda data: data["components"][0].__setitem__(
                "member_count", data["components"][0]["member_count"] + 1
            ),
            "Pair graph component member count mismatch",
        ),
    ],
)
def test_charge_continuity_diagnostic_rejects_resealed_repinned_component_metadata(
    tmp_path: Path, mutation, message: str
) -> None:
    tampered_pair_artifact = _json_copy_with_mutation(
        PAIR_PATH, tmp_path, mutation
    )
    protocol = _aligned_protocol(
        tmp_path, pair_artifact=tampered_pair_artifact
    )

    with pytest.raises(ValueError, match=message):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_artifact=tampered_pair_artifact,
            output_name="diagnostic_with_semantic_component_tamper.json",
        )


def test_charge_continuity_diagnostic_rejects_resealed_repinned_pair_identity(
    tmp_path: Path,
) -> None:
    def _mutate_pair(data):
        data["pairs"][1]["first_id"] = data["pairs"][0]["first_id"]
        data["pairs"][1]["second_id"] = data["pairs"][0]["second_id"]

    tampered_pair_artifact = _json_copy_with_mutation(
        PAIR_PATH, tmp_path, _mutate_pair
    )
    protocol = _aligned_protocol(
        tmp_path, pair_artifact=tampered_pair_artifact
    )

    with pytest.raises(ValueError, match="Derived strict-MMP pair id mismatch"):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_artifact=tampered_pair_artifact,
            output_name="diagnostic_with_semantic_pair_tamper.json",
        )


@pytest.mark.parametrize("boundary_field", ["score_artifact_read", "series_artifact_read"])
def test_charge_continuity_diagnostic_rejects_resealed_repinned_label_boundary(
    tmp_path: Path, boundary_field: str
) -> None:
    tampered_pair_artifact = _json_copy_with_mutation(
        PAIR_PATH,
        tmp_path,
        lambda data: data["label_boundary"].__setitem__(
            boundary_field, True
        ),
    )
    protocol = _aligned_protocol(
        tmp_path, pair_artifact=tampered_pair_artifact
    )

    with pytest.raises(
        ValueError, match="crossed the experimental-label boundary"
    ):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_artifact=tampered_pair_artifact,
            output_name="diagnostic_with_label_boundary_tamper.json",
        )


def test_charge_continuity_protocol_and_artifact_claim_exactly_65_edges() -> None:
    protocol = runner.core.load_json(PROTOCOL_PATH)
    artifact = runner.core.load_json(FROZEN_OUTPUT)

    assert protocol["source_evidence"]["expected_pair_count"] == 65
    assert "65 strict heavy-atom matched-pair edges" in protocol["claim_scope"]
    assert artifact["evaluable_pair_count"] == 65
    assert "65 strict heavy-atom matched-pair edges" in artifact["claim_scope"]


def test_charge_continuity_diagnostic_carries_the_pair_mapping_boundary() -> None:
    protocol = runner.core.load_json(PROTOCOL_PATH)
    artifact = runner.core.load_json(FROZEN_OUTPUT)

    assert protocol["pair_mapping_boundary"] == runner.PAIR_MAPPING_BOUNDARY
    assert artifact["pair_mapping_boundary"] == runner.PAIR_MAPPING_BOUNDARY
    assert "single SMARTS returned by RDKit FindMCS" in protocol["claim_scope"]
    assert (
        "distinct non-isomorphic maximum-MCS patterns are not enumerated"
        in protocol["claim_scope"]
    )
    assert artifact["claim_scope"] == protocol["claim_scope"]


def test_charge_continuity_diagnostic_rejects_a_weakened_pair_mapping_boundary(
    tmp_path: Path,
) -> None:
    protocol = runner.core.load_json(_aligned_protocol(tmp_path))
    protocol["pair_mapping_boundary"][
        "distinct_maximum_mcs_patterns_audited"
    ] = True
    protocol_path = tmp_path / "weakened-pair-mapping-boundary.json"
    runner.core.write_json_atomic(protocol_path, protocol)

    with pytest.raises(
        ValueError, match="weakens its pair-mapping boundary"
    ):
        _run_diagnostic(
            tmp_path,
            protocol=protocol_path,
            output_name="diagnostic-with-weakened-mapping-boundary.json",
        )


def test_charge_continuity_reported_metric_contract_matches_output_exactly() -> None:
    protocol = runner.core.load_json(PROTOCOL_PATH)
    artifact = runner.core.load_json(FROZEN_OUTPUT)
    primary_metrics = set(
        protocol["diagnostic_statistics"]["reported_pair_metrics"]
    )
    accounting = {
        "pair_count",
        "correct_count",
        "wrong_count",
        "predicted_tie_count",
    }
    association = {
        "descriptive_spearman_charge_drift_vs_absolute_delta_delta_error"
    }

    for gate in artifact["diagnostic_summaries"].values():
        for method in gate["methods"].values():
            assert set(method) == primary_metrics | accounting | association


def test_charge_continuity_diagnostic_rejects_resealed_repinned_route_boundary(
    tmp_path: Path,
) -> None:
    tampered_pair_artifact = _json_copy_with_mutation(
        PAIR_PATH,
        tmp_path,
        lambda data: data["route1_boundary"].__setitem__(
            "fixed_charge", "forbidden"
        ),
    )
    protocol = _aligned_protocol(
        tmp_path, pair_artifact=tampered_pair_artifact
    )

    with pytest.raises(ValueError, match="violates the Route 1 boundary"):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_artifact=tampered_pair_artifact,
            output_name="diagnostic_with_route_boundary_tamper.json",
        )


def test_charge_continuity_diagnostic_rejects_repinned_pair_protocol_label_access(
    tmp_path: Path,
) -> None:
    pair_protocol = runner.core.load_json(PAIR_PROTOCOL_PATH)
    pair_protocol["evaluation_design"]["this_phase_reads_score_artifact"] = True
    pair_protocol_path = tmp_path / "pair-protocol.json"
    runner.core.write_json_atomic(pair_protocol_path, pair_protocol)

    pair_artifact = runner.core.load_json(PAIR_PATH)
    pair_artifact["protocol_sha256"] = runner.core.sha256_file(
        pair_protocol_path
    )
    pair_artifact["protocol_fingerprint"] = runner.core.sha256_bytes(
        runner.core.canonical_json_bytes(pair_protocol)
    )
    pair_artifact = runner.core.seal_artifact(
        {
            key: value
            for key, value in pair_artifact.items()
            if key != "content_sha256"
        }
    )
    pair_artifact_path = tmp_path / "pair-artifact.json"
    runner.core.write_json_atomic(pair_artifact_path, pair_artifact)
    protocol = _aligned_protocol(
        tmp_path,
        pair_protocol=pair_protocol_path,
        pair_artifact=pair_artifact_path,
    )

    with pytest.raises(ValueError, match="weakens the label-free phase"):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_protocol=pair_protocol_path,
            pair_artifact=pair_artifact_path,
            output_name="diagnostic_with_pair_protocol_label_access.json",
        )


def test_charge_continuity_diagnostic_rejects_repinned_pair_protocol_decision(
    tmp_path: Path,
) -> None:
    pair_protocol = runner.core.load_json(PAIR_PROTOCOL_PATH)
    pair_protocol["pre_registered_decision_rule"][
        "energy_correction_allowed"
    ] = True
    pair_protocol_path = tmp_path / "pair-protocol.json"
    runner.core.write_json_atomic(pair_protocol_path, pair_protocol)

    pair_artifact = runner.core.load_json(PAIR_PATH)
    pair_artifact["protocol_sha256"] = runner.core.sha256_file(
        pair_protocol_path
    )
    pair_artifact["protocol_fingerprint"] = runner.core.sha256_bytes(
        runner.core.canonical_json_bytes(pair_protocol)
    )
    pair_artifact = runner.core.seal_artifact(
        {
            key: value
            for key, value in pair_artifact.items()
            if key != "content_sha256"
        }
    )
    pair_artifact_path = tmp_path / "pair-artifact.json"
    runner.core.write_json_atomic(pair_artifact_path, pair_artifact)
    protocol = _aligned_protocol(
        tmp_path,
        pair_protocol=pair_protocol_path,
        pair_artifact=pair_artifact_path,
    )

    with pytest.raises(
        ValueError, match="Pair protocol permits an unsupported decision"
    ):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_protocol=pair_protocol_path,
            pair_artifact=pair_artifact_path,
            output_name="diagnostic_with_pair_protocol_decision.json",
        )


def test_charge_continuity_diagnostic_rejects_repinned_pair_artifact_decision(
    tmp_path: Path,
) -> None:
    def _mutate_decision(data):
        data["decision"]["energy_correction_allowed"] = True
        data["decision"]["certified_order_allowed"] = True

    pair_artifact_path = _json_copy_with_mutation(
        PAIR_PATH, tmp_path, _mutate_decision
    )
    protocol = _aligned_protocol(
        tmp_path, pair_artifact=pair_artifact_path
    )

    with pytest.raises(
        ValueError, match="Pair artifact permits an unsupported decision"
    ):
        _run_diagnostic(
            tmp_path,
            protocol=protocol,
            pair_artifact=pair_artifact_path,
            output_name="diagnostic_with_pair_artifact_decision.json",
        )
