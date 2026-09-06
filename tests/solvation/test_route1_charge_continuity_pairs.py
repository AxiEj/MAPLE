from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

import pytest
from rdkit import Chem

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import build_route1_charge_continuity_pairs as builder  # pyright: ignore[reportMissingImports]


PROTOCOL_PATH = BENCHMARK_DIR / "route1_charge_continuity_protocol_v1.json"
SOURCE_MANIFEST_PATH = (
    BENCHMARK_DIR / "route1_freesolv_reserve_source_manifest.json"
)
ENERGY_PATH = BENCHMARK_DIR / "route1-freesolv-reserve-energy-2026-07-25.json"
PAIR_ARTIFACT_PATH = (
    BENCHMARK_DIR
    / "route1-freesolv-reserve-charge-continuity-pairs-2026-07-29.json"
)
SOURCE_ROOT = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
PROTOCOL = builder.core.load_json(PROTOCOL_PATH)


def _build(output: Path, **overrides: Path) -> dict[str, object]:
    arguments = {
        "protocol": PROTOCOL_PATH,
        "source_manifest": SOURCE_MANIFEST_PATH,
        "energy_artifact": ENERGY_PATH,
        "source_root": SOURCE_ROOT,
        "output": output,
    }
    arguments.update(overrides)
    return builder.build(argparse.Namespace(**arguments))


@pytest.fixture(scope="module")
def rebuilt_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _build(
        tmp_path_factory.mktemp("charge-continuity-pairs")
        / "route1-freesolv-reserve-charge-continuity-pairs.json"
    )


def _without_run_specific_fields(artifact: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(artifact)
    normalized.pop("content_sha256", None)
    normalized.pop("command_provenance", None)
    return normalized


def _iter_keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _iter_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_keys(nested)


def _graph(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    molecule = Chem.RemoveHs(molecule)
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetIntProp("source_atom_index", index)
    return molecule


def _write_aligned_protocol(
    tmp_path: Path,
    *,
    source_manifest: Path = SOURCE_MANIFEST_PATH,
    energy_artifact: Path = ENERGY_PATH,
) -> Path:
    protocol = copy.deepcopy(PROTOCOL)
    energy = builder.core.load_json(energy_artifact)
    protocol["source_evidence"]["source_manifest_sha256"] = (
        builder.core.sha256_file(source_manifest)
    )
    protocol["source_evidence"]["energy_artifact_sha256"] = (
        builder.core.sha256_file(energy_artifact)
    )
    protocol["source_evidence"]["energy_artifact_content_sha256"] = (
        energy["content_sha256"]
    )
    path = tmp_path / "route1_charge_continuity_protocol_v1.json"
    builder.core.write_json_atomic(path, protocol)
    return path


def test_frozen_pair_artifact_is_reproducible_and_self_sealed(
    rebuilt_artifact: dict[str, object],
) -> None:
    frozen = builder.core.load_json(PAIR_ARTIFACT_PATH)

    assert frozen["content_sha256"] == builder.core.artifact_content_sha256(frozen)
    assert _without_run_specific_fields(rebuilt_artifact) == (
        _without_run_specific_fields(frozen)
    )
    assert frozen["case_count"] == 116
    assert frozen["candidate_pair_count"] == 6670
    assert frozen["evaluable_pair_count"] == 65
    assert frozen["excluded_pair_count"] == 6605
    assert frozen["excluded_reason_counts"] == {
        "mapping_enumeration_overflow": 9,
        "mapping_metric_variant": 1,
        "mapping_remote_count_variant": 1,
        "mcs_too_small": 4481,
        "no_strict_mmp_mapping": 2113,
    }
    assert frozen["mapping_variant_sensitivity_count"] == 2
    assert frozen["mapping_ambiguous_but_metric_invariant_count"] == 63
    assert frozen["endpoint_order_disagreement_count"] == 6
    assert sorted(
        (component["member_count"], component["edge_count"])
        for component in frozen["components"]
    ) == [(2, 1), (2, 1), (4, 5), (6, 6), (13, 52)]


def test_pair_phase_is_physically_label_free_and_has_no_score_or_series_inputs() -> None:
    frozen = builder.core.load_json(PAIR_ARTIFACT_PATH)
    source_evidence = PROTOCOL["source_evidence"]
    command_arguments = frozen["command_provenance"]["arguments"]

    assert set(source_evidence) == {
        "source_manifest",
        "source_manifest_sha256",
        "energy_artifact",
        "energy_artifact_sha256",
        "energy_artifact_content_sha256",
        "expected_case_count",
    }
    assert set(command_arguments) == {
        "phase",
        "protocol",
        "source_manifest",
        "energy_artifact",
        "source_root",
        "output",
    }
    assert frozen["label_boundary"] == {
        "experimental_labels_read": False,
        "score_artifact_read": False,
        "series_artifact_read": False,
        "historical_label_exposure": True,
    }
    forbidden = {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_errors_kcal_mol",
        "hydration_free_energy",
    }
    assert forbidden.isdisjoint(set(_iter_keys(PROTOCOL)))
    assert forbidden.isdisjoint(set(_iter_keys(frozen)))
    design = PROTOCOL["pair_audit_design"]
    assert design["mcs_pattern_enumeration"] == (
        "single_rdkit_findmcs_smarts_only"
    )
    assert design["distinct_maximum_mcs_patterns_audited"] is False
    assert frozen["pair_audit_design"] == design


def test_pair_and_component_identities_reconcile() -> None:
    frozen = builder.core.load_json(PAIR_ARTIFACT_PATH)
    pair_ids: set[str] = set()
    edges_by_component: dict[str, list[str]] = {
        component["component_id"]: [] for component in frozen["components"]
    }
    members_by_component = {
        component["component_id"]: set(component["member_ids"])
        for component in frozen["components"]
    }
    for pair in frozen["pairs"]:
        assert pair["first_id"] < pair["second_id"]
        assert pair["pair_id"] == builder._pair_id(
            pair["first_id"], pair["second_id"]
        )
        assert pair["pair_id"] not in pair_ids
        pair_ids.add(pair["pair_id"])
        assert {
            pair["first_id"],
            pair["second_id"],
        }.issubset(members_by_component[pair["component_id"]])
        edges_by_component[pair["component_id"]].append(pair["pair_id"])
        assert pair["mapping_enumeration_status"] == (
            "complete_for_single_rdkit_findmcs_smarts"
        )
        assert pair["remote_heavy_atom_count_interval"][0] == (
            pair["remote_heavy_atom_count_interval"][1]
        )
        for interval in pair["mapping_metric_intervals_e"].values():
            assert interval[0] == interval[1]

    assert len(pair_ids) == frozen["evaluable_pair_count"]
    for component in frozen["components"]:
        component_id = component["component_id"]
        members = sorted(component["member_ids"])
        expected_component_id = (
            "strict-mmp-component-v1:"
            + builder.core.sha256_bytes(builder.core.canonical_json_bytes(members))
        )
        assert component_id == expected_component_id
        assert component["edge_count"] == len(edges_by_component[component_id])
        assert component["pair_ids_sha256"] == builder.core.sha256_bytes(
            builder.core.canonical_json_bytes(
                sorted(edges_by_component[component_id])
            )
        )


def test_all_source_atoms_and_charge_vectors_are_explicitly_bound() -> None:
    frozen = builder.core.load_json(PAIR_ARTIFACT_PATH)
    energy = builder.core.load_json(ENERGY_PATH)
    energy_by_id = {row["compound_id"]: row for row in energy["records"]}

    assert len(frozen["molecule_audit"]) == 116
    for molecule in frozen["molecule_audit"]:
        energy_row = energy_by_id[molecule["compound_id"]]
        mapping = molecule["heavy_rdkit_to_source_and_charge_index"]
        assert molecule["charge_vector_sha256"] == (
            energy_row["am1bcc_charge_vector_sha256"]
        )
        assert molecule["all_atom_count"] == len(energy_row["am1bcc_charges_e"])
        assert molecule["heavy_atom_count"] == len(mapping)
        assert [row["heavy_rdkit_index"] for row in mapping] == list(
            range(len(mapping))
        )
        assert all(
            row["source_mol2_atom_index"] == row["charge_vector_index"]
            for row in mapping
        )


def test_strict_mmp_distance_three_is_included_but_distance_four_is_not() -> None:
    left = _graph("CCCCCF")
    right = _graph("CCCCCCl")
    design = copy.deepcopy(PROTOCOL["pair_audit_design"])

    reason, candidates = builder._strict_mapping_candidates(left, right, design)
    assert reason is None
    assert candidates
    assert all(candidate["remote_heavy_atom_count"] >= 2 for candidate in candidates)

    design["minimum_attachment_distance_bonds"] = 4
    reason, candidates = builder._strict_mapping_candidates(left, right, design)
    assert reason == "no_strict_mmp_mapping"
    assert candidates == []


def test_strict_mmp_rejects_two_disconnected_substituent_changes() -> None:
    left = _graph("FCCCCCCl")
    right = _graph("BrCCCCCI")

    reason, candidates = builder._strict_mapping_candidates(
        left,
        right,
        PROTOCOL["pair_audit_design"],
    )

    assert reason == "no_strict_mmp_mapping"
    assert candidates == []


def test_mapping_cross_product_overflow_fails_closed() -> None:
    left = _graph("CCCCC(F)C")
    right = _graph("CCCCC(Cl)C")
    design = copy.deepcopy(PROTOCOL["pair_audit_design"])
    design["maximum_mapping_enumeration"] = 1

    reason, candidates = builder._strict_mapping_candidates(left, right, design)

    assert reason == "mapping_enumeration_overflow"
    assert candidates == []


@pytest.mark.parametrize("mutation", ["length", "nan", "hash"])
def test_charge_vector_corruption_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    energy = builder.core.load_json(ENERGY_PATH)
    row = energy["records"][0]
    if mutation == "length":
        row["am1bcc_charges_e"].pop()
    elif mutation == "nan":
        row["am1bcc_charges_e"][0] = float("nan")
    else:
        row["am1bcc_charge_vector_sha256"] = "0" * 64
    row["content_sha256"] = builder.core.artifact_content_sha256(row)
    energy = builder.core.seal_artifact(
        {key: value for key, value in energy.items() if key != "content_sha256"}
    )
    energy_path = tmp_path / "energy.json"
    builder.core.write_json_atomic(energy_path, energy)
    protocol_path = _write_aligned_protocol(
        tmp_path, energy_artifact=energy_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "Charge-vector length mismatch"
            if mutation == "length"
            else "Nonfinite charge"
            if mutation == "nan"
            else "Charge-vector hash mismatch"
        ),
    ):
        builder.build_pair_artifact(
            protocol_path=protocol_path,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            energy_artifact_path=energy_path,
            source_root=SOURCE_ROOT,
        )


def test_label_bearing_energy_is_rejected_even_when_fully_resealed(
    tmp_path: Path,
) -> None:
    energy = builder.core.load_json(ENERGY_PATH)
    row = energy["records"][0]
    row["experimental_kcal_mol"] = -4.59
    row["content_sha256"] = builder.core.artifact_content_sha256(row)
    energy = builder.core.seal_artifact(
        {key: value for key, value in energy.items() if key != "content_sha256"}
    )
    energy_path = tmp_path / "label-bearing-energy.json"
    builder.core.write_json_atomic(energy_path, energy)
    protocol_path = _write_aligned_protocol(
        tmp_path, energy_artifact=energy_path
    )

    protocol, _fingerprint = builder.load_protocol(protocol_path)
    _manifest, source_by_id = builder._load_source_manifest(
        protocol, SOURCE_MANIFEST_PATH
    )
    with pytest.raises(
        ValueError, match="label field 'experimental_kcal_mol'"
    ):
        builder._load_energy_artifact(protocol, energy_path, source_by_id)


def test_pair_protocol_rejects_unknown_score_or_series_evidence(
    tmp_path: Path,
) -> None:
    protocol = copy.deepcopy(PROTOCOL)
    protocol["source_evidence"]["score_artifact"] = "forbidden.json"
    path = tmp_path / "protocol-with-score.json"
    builder.core.write_json_atomic(path, protocol)

    with pytest.raises(
        ValueError, match="Pair protocol source_evidence schema mismatch"
    ):
        builder.load_protocol(path)


def test_pinned_mol2_corruption_fails_closed(tmp_path: Path) -> None:
    manifest = builder.core.load_json(SOURCE_MANIFEST_PATH)
    first = min(manifest["records"], key=lambda row: row["compound_id"])
    relative_path = Path(first["source_mol2_relative_path"])
    source_root = tmp_path / "source"
    destination = source_root / relative_path
    destination.parent.mkdir(parents=True)
    shutil.copy2(SOURCE_ROOT / relative_path, destination)
    destination.write_bytes(destination.read_bytes() + b"\n# tampered\n")

    with pytest.raises(ValueError, match="Pinned MOL2 hash mismatch"):
        builder.build_pair_artifact(
            protocol_path=PROTOCOL_PATH,
            source_manifest_path=SOURCE_MANIFEST_PATH,
            energy_artifact_path=ENERGY_PATH,
            source_root=source_root,
        )


def test_manifest_record_order_does_not_change_pair_payload(tmp_path: Path) -> None:
    manifest = builder.core.load_json(SOURCE_MANIFEST_PATH)
    manifest["records"] = list(reversed(manifest["records"]))
    manifest_path = tmp_path / "manifest.json"
    builder.core.write_json_atomic(manifest_path, manifest)
    energy = builder.core.load_json(ENERGY_PATH)
    energy["source_manifest_sha256"] = builder.core.sha256_file(manifest_path)
    energy = builder.core.seal_artifact(
        {key: value for key, value in energy.items() if key != "content_sha256"}
    )
    energy_path = tmp_path / "energy.json"
    builder.core.write_json_atomic(energy_path, energy)
    protocol_path = _write_aligned_protocol(
        tmp_path,
        source_manifest=manifest_path,
        energy_artifact=energy_path,
    )
    reordered = builder.build_pair_artifact(
        protocol_path=protocol_path,
        source_manifest_path=manifest_path,
        energy_artifact_path=energy_path,
        source_root=SOURCE_ROOT,
    )
    frozen = builder.core.load_json(PAIR_ARTIFACT_PATH)

    assert reordered["pairs"] == frozen["pairs"]
    assert reordered["components"] == frozen["components"]
    assert reordered["molecule_audit"] == frozen["molecule_audit"]
    assert reordered["excluded_pairs_sha256"] == frozen["excluded_pairs_sha256"]


def test_mapping_variant_pairs_are_excluded_but_retained_as_label_free_sensitivity() -> None:
    frozen = builder.core.load_json(PAIR_ARTIFACT_PATH)
    evaluable_ids = {row["pair_id"] for row in frozen["pairs"]}
    sensitivity_ids = {
        row["pair_id"] for row in frozen["mapping_variant_sensitivity"]
    }

    assert sensitivity_ids
    assert evaluable_ids.isdisjoint(sensitivity_ids)
    assert {
        row["reason"] for row in frozen["mapping_variant_sensitivity"]
    } == {"mapping_metric_variant", "mapping_remote_count_variant"}
    assert any(
        any(interval[0] != interval[1] for interval in row[
            "mapping_metric_intervals_e"
        ].values())
        for row in frozen["mapping_variant_sensitivity"]
    )
