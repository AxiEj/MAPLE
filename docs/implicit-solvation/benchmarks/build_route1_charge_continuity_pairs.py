#!/usr/bin/env python3
"""Build a label-free strict-MMP charge-continuity sidecar for Route 1."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_charge_continuity_protocol_v1.json"
DEFAULT_SOURCE_MANIFEST = SCRIPT_DIR / "route1_freesolv_reserve_source_manifest.json"
DEFAULT_ENERGY_ARTIFACT = (
    SCRIPT_DIR / "route1-freesolv-reserve-energy-2026-07-25.json"
)
DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-freesolv-reserve-charge-continuity-pairs-2026-07-29.json"
)

ARTIFACT_TYPE = "route1-charge-continuity-label-free-pairs"
ENDPOINTS = (
    "am1bcc_obc2_ace",
    "am1bcc_chagb_pbsa_cavity_dispersion",
)
ROUTE1_BOUNDARY = {
    "name": "Additive fixed-charge PB/GB implicit solvation",
    "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
    "gas_phase_mm_energy": False,
    "hydration_label_residual": False,
    "mlip_retraining": False,
    "fixed_charge": "AM1-BCC",
}
_BOND_TYPES = {
    "1": Chem.BondType.SINGLE,
    "2": Chem.BondType.DOUBLE,
    "3": Chem.BondType.TRIPLE,
    "ar": Chem.BondType.AROMATIC,
}
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_errors_kcal_mol",
        "hydration_free_energy",
    }
)
_PROTOCOL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "claim_scope",
        "route1_boundary",
        "evaluation_design",
        "source_evidence",
        "pair_audit_design",
        "endpoints",
        "pre_registered_decision_rule",
    }
)
_EVALUATION_DESIGN_KEYS = frozenset(
    {
        "historical_label_exposure",
        "this_phase_reads_experimental_labels",
        "this_phase_reads_score_artifact",
        "this_phase_reads_series_artifact",
        "no_fit",
        "no_residual",
        "no_endpoint_selection",
        "no_threshold_tuning",
        "certified_ranking_available",
    }
)
_SOURCE_EVIDENCE_KEYS = frozenset(
    {
        "source_manifest",
        "source_manifest_sha256",
        "energy_artifact",
        "energy_artifact_sha256",
        "energy_artifact_content_sha256",
        "expected_case_count",
    }
)
_PAIR_AUDIT_DESIGN_KEYS = frozenset(
    {
        "source_root_relative_path",
        "rdkit_required_version",
        "mcs_timeout_seconds",
        "minimum_matched_heavy_atoms",
        "minimum_unmatched_heavy_atoms_each_side",
        "maximum_unmatched_heavy_atoms_each_side",
        "unmatched_component_count_each_side",
        "core_substituent_boundary_edge_count_each_side",
        "minimum_remote_heavy_atoms",
        "minimum_attachment_distance_bonds",
        "maximum_mapping_enumeration",
        "mapping_metric_tolerance_e",
        "mcs_atom_compare",
        "mcs_bond_compare",
        "ring_matches_ring_only",
        "complete_rings_only",
        "matched_core_connected",
        "substructure_matches_uniquified",
        "mcs_pattern_enumeration",
        "distinct_maximum_mcs_patterns_audited",
        "mapping_enumeration_scope",
        "mapping_selection",
        "hydrogens_in_charge_drift",
    }
)
_DECISION_RULE_KEYS = frozenset(
    {
        "metric_variant_structural_mapping_is_evaluable",
        "endpoint_selection_allowed",
        "charge_method_selection_allowed",
        "energy_correction_allowed",
        "abstention_threshold_selection_allowed",
        "certified_order_allowed",
        "failure_action",
    }
)
_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "protocol_id",
        "source_designation",
        "source_prepared_sha256",
        "case_count",
        "records",
    }
)
_SOURCE_RECORD_KEYS = frozenset(
    {
        "compound_id",
        "dataset_record_sha256",
        "source_mol2_relative_path",
        "source_mol2_sha256",
        "structure_group_sha256",
        "atom_count",
        "heavy_atom_count",
        "elements",
        "bins",
    }
)
_ENERGY_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "protocol_sha256",
        "protocol_fingerprint",
        "route1_contract",
        "source_designation",
        "source_manifest_sha256",
        "case_count",
        "endpoint_names",
        "provider_provenance",
        "record_file_sha256",
        "records",
        "command_provenance",
        "content_sha256",
    }
)
_ENERGY_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_id",
        "protocol_fingerprint",
        "compound_id",
        "dataset_record_sha256",
        "source_mol2_sha256",
        "am1bcc_charges_e",
        "am1bcc_charge_vector_sha256",
        "charge_provenance",
        "components_kcal_mol",
        "predictions_kcal_mol",
        "provider_provenance",
        "status",
        "content_sha256",
    }
)


def _relative_to_repository(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_object(path: Path) -> dict[str, Any]:
    value = core.load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{context} schema mismatch; missing={missing}, extra={extra}."
        )


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if not core._is_hex(text, 64):
        raise ValueError(f"{field} must be a lowercase SHA256.")
    return text


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _require_nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative.")
    return number


def _reject_label_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_LABEL_FIELDS:
                raise ValueError(f"Label-free input contains label field {key!r}.")
            _reject_label_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_label_fields(nested)


def _safe_child(root: Path, relative_value: object) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source path: {relative}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source path escapes its root: {relative}") from exc
    return result


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol_path = Path(path).resolve()
    protocol = _load_object(protocol_path)
    _require_exact_keys(protocol, _PROTOCOL_KEYS, context="Pair protocol")
    if protocol.get("schema_version") != 1:
        raise ValueError("Only charge-continuity protocol schema 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-charge-continuity-pairs-v1":
        raise ValueError("Unexpected charge-continuity pair protocol id.")
    if protocol.get("route1_boundary") != ROUTE1_BOUNDARY:
        raise ValueError("Charge-continuity protocol violates the Route 1 boundary.")

    design_flags = protocol.get("evaluation_design", {})
    if not isinstance(design_flags, dict):
        raise ValueError("Pair protocol evaluation_design must be an object.")
    _require_exact_keys(
        design_flags,
        _EVALUATION_DESIGN_KEYS,
        context="Pair protocol evaluation_design",
    )
    required_flags = {
        "historical_label_exposure": True,
        "this_phase_reads_experimental_labels": False,
        "this_phase_reads_score_artifact": False,
        "this_phase_reads_series_artifact": False,
        "no_fit": True,
        "no_residual": True,
        "no_endpoint_selection": True,
        "no_threshold_tuning": True,
        "certified_ranking_available": False,
    }
    if any(design_flags.get(key) is not value for key, value in required_flags.items()):
        raise ValueError("Charge-continuity pair protocol weakens label separation.")

    evidence = protocol.get("source_evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("Pair protocol source_evidence must be an object.")
    _require_exact_keys(
        evidence,
        _SOURCE_EVIDENCE_KEYS,
        context="Pair protocol source_evidence",
    )
    for field in (
        "source_manifest_sha256",
        "energy_artifact_sha256",
        "energy_artifact_content_sha256",
    ):
        _require_sha256(evidence.get(field), field=f"source_evidence.{field}")
    _require_positive_int(
        evidence.get("expected_case_count"),
        field="source_evidence.expected_case_count",
    )

    design = protocol.get("pair_audit_design", {})
    if not isinstance(design, dict):
        raise ValueError("Pair protocol pair_audit_design must be an object.")
    _require_exact_keys(
        design,
        _PAIR_AUDIT_DESIGN_KEYS,
        context="Pair protocol pair_audit_design",
    )
    for field in (
        "mcs_timeout_seconds",
        "minimum_matched_heavy_atoms",
        "minimum_unmatched_heavy_atoms_each_side",
        "maximum_unmatched_heavy_atoms_each_side",
        "unmatched_component_count_each_side",
        "core_substituent_boundary_edge_count_each_side",
        "minimum_remote_heavy_atoms",
        "minimum_attachment_distance_bonds",
        "maximum_mapping_enumeration",
    ):
        _require_positive_int(design.get(field), field=f"pair_audit_design.{field}")
    _require_nonnegative_float(
        design.get("mapping_metric_tolerance_e"),
        field="pair_audit_design.mapping_metric_tolerance_e",
    )
    if design.get("mcs_atom_compare") != "CompareElements":
        raise ValueError("The pair protocol must compare atom elements exactly.")
    if design.get("mcs_bond_compare") != "CompareOrderExact":
        raise ValueError("The pair protocol must compare bond orders exactly.")
    if design.get("ring_matches_ring_only") is not True:
        raise ValueError("ring_matches_ring_only must be true.")
    if design.get("complete_rings_only") is not True:
        raise ValueError("complete_rings_only must be true.")
    if design.get("matched_core_connected") is not True:
        raise ValueError("The matched heavy-atom core must remain connected.")
    if design.get("substructure_matches_uniquified") is not False:
        raise ValueError(
            "All embeddings of the returned MCS SMARTS must be enumerated."
        )
    if design.get("mcs_pattern_enumeration") != (
        "single_rdkit_findmcs_smarts_only"
    ):
        raise ValueError("The MCS-pattern enumeration boundary changed.")
    if design.get("distinct_maximum_mcs_patterns_audited") is not False:
        raise ValueError("Distinct maximum-MCS patterns are not audited.")
    if (
        design.get("mapping_enumeration_scope")
        != "cross_product_of_nonuniquified_embeddings_of_single_rdkit_findmcs_smarts"
    ):
        raise ValueError("The mapping-enumeration scope changed.")
    if (
        design.get("mapping_selection")
        != "all_strict_embeddings_of_returned_smarts_must_be_metric_invariant_then_lexicographic"
    ):
        raise ValueError("The structure-only mapping selection rule changed.")
    if design.get("hydrogens_in_charge_drift") is not False:
        raise ValueError("This protocol measures remote heavy-atom charges only.")
    if int(design["minimum_unmatched_heavy_atoms_each_side"]) > int(
        design["maximum_unmatched_heavy_atoms_each_side"]
    ):
        raise ValueError("The unmatched-heavy-atom bounds are inverted.")
    if rdBase.rdkitVersion != design.get("rdkit_required_version"):
        raise ValueError(
            "RDKit version mismatch: "
            f"expected {design.get('rdkit_required_version')}, "
            f"observed {rdBase.rdkitVersion}."
        )
    if tuple(protocol.get("endpoints", ())) != ENDPOINTS:
        raise ValueError("Charge-continuity endpoints or their order changed.")

    rule = protocol.get("pre_registered_decision_rule", {})
    if not isinstance(rule, dict):
        raise ValueError("Pair protocol decision rule must be an object.")
    _require_exact_keys(
        rule,
        _DECISION_RULE_KEYS,
        context="Pair protocol pre_registered_decision_rule",
    )
    if any(
        rule.get(key) is not False
        for key in (
            "metric_variant_structural_mapping_is_evaluable",
            "endpoint_selection_allowed",
            "charge_method_selection_allowed",
            "energy_correction_allowed",
            "abstention_threshold_selection_allowed",
            "certified_order_allowed",
        )
    ):
        raise ValueError("Charge-continuity protocol permits unsupported selection.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _load_source_manifest(
    protocol: Mapping[str, Any], path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["source_manifest_sha256"]:
        raise ValueError("Frozen source-manifest hash mismatch.")
    manifest = _load_object(path)
    _reject_label_fields(manifest)
    _require_exact_keys(
        manifest, _SOURCE_MANIFEST_KEYS, context="Source manifest"
    )
    records = manifest.get("records", [])
    expected = int(evidence["expected_case_count"])
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError("Source-manifest case count mismatch.")
    if manifest.get("case_count") != expected:
        raise ValueError("Source-manifest declared case count mismatch.")
    ids = [str(row.get("compound_id", "")) for row in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Source-manifest IDs must be nonempty and unique.")
    by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Malformed source-manifest record.")
        _require_exact_keys(
            row, _SOURCE_RECORD_KEYS, context="Source-manifest record"
        )
        compound_id = str(row["compound_id"])
        _require_sha256(row.get("source_mol2_sha256"), field="source_mol2_sha256")
        _require_positive_int(row.get("atom_count"), field="source atom_count")
        if not str(row.get("source_mol2_relative_path", "")):
            raise ValueError("Source record lacks a MOL2 path.")
        by_id[compound_id] = row
    return manifest, by_id


def _load_energy_artifact(
    protocol: Mapping[str, Any],
    path: Path,
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = protocol["source_evidence"]
    if core.sha256_file(path) != evidence["energy_artifact_sha256"]:
        raise ValueError("Frozen energy-artifact file hash mismatch.")
    artifact = _load_object(path)
    _reject_label_fields(artifact)
    _require_exact_keys(
        artifact, _ENERGY_ARTIFACT_KEYS, context="Energy artifact"
    )
    if core.artifact_content_sha256(artifact) != artifact.get("content_sha256"):
        raise ValueError("Frozen energy-artifact content seal mismatch.")
    if artifact.get("content_sha256") != evidence["energy_artifact_content_sha256"]:
        raise ValueError("Frozen energy-artifact content hash changed.")
    if (
        artifact.get("artifact_type")
        != "route1-freesolv-reserve-label-free-energy"
    ):
        raise ValueError("Unexpected energy-artifact type.")
    if artifact.get("route1_contract") != ROUTE1_BOUNDARY:
        raise ValueError("Energy artifact violates the Route 1 boundary.")
    if artifact.get("source_manifest_sha256") != evidence["source_manifest_sha256"]:
        raise ValueError("Energy artifact is bound to another source manifest.")
    if artifact.get("case_count") != evidence["expected_case_count"]:
        raise ValueError("Energy-artifact declared case count mismatch.")
    if tuple(artifact.get("endpoint_names", ())) != ENDPOINTS:
        raise ValueError("Energy-artifact endpoint declaration changed.")

    records = artifact.get("records", [])
    if not isinstance(records, list) or len(records) != len(source_by_id):
        raise ValueError("Energy-artifact record count mismatch.")
    ids = [str(row.get("compound_id", "")) for row in records]
    if len(ids) != len(set(ids)) or set(ids) != set(source_by_id):
        raise ValueError("Energy-artifact IDs do not match the source manifest.")

    by_id: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Malformed energy record.")
        _require_exact_keys(
            row, _ENERGY_RECORD_KEYS, context="Energy record"
        )
        compound_id = str(row["compound_id"])
        if core.artifact_content_sha256(row) != row.get("content_sha256"):
            raise ValueError(f"Energy-record content seal mismatch for {compound_id}.")
        if row.get("status") != "success":
            raise ValueError(f"Unsuccessful energy record for {compound_id}.")
        source = source_by_id[compound_id]
        for field in ("dataset_record_sha256", "source_mol2_sha256"):
            if row.get(field) != source.get(field):
                raise ValueError(f"Energy/source identity mismatch for {compound_id}.")
        charges = row.get("am1bcc_charges_e")
        if not isinstance(charges, list) or len(charges) != int(source["atom_count"]):
            raise ValueError(f"Charge-vector length mismatch for {compound_id}.")
        normalized = [float(charge) for charge in charges]
        if not all(math.isfinite(charge) for charge in normalized):
            raise ValueError(f"Nonfinite charge for {compound_id}.")
        expected_charge_hash = core.sha256_bytes(core.canonical_json_bytes(normalized))
        if row.get("am1bcc_charge_vector_sha256") != expected_charge_hash:
            raise ValueError(f"Charge-vector hash mismatch for {compound_id}.")
        predictions = row.get("predictions_kcal_mol")
        if not isinstance(predictions, dict) or set(predictions) != set(ENDPOINTS):
            raise ValueError(f"Endpoint membership mismatch for {compound_id}.")
        if not all(
            isinstance(predictions[name], (int, float))
            and not isinstance(predictions[name], bool)
            and math.isfinite(float(predictions[name]))
            for name in ENDPOINTS
        ):
            raise ValueError(f"Nonfinite endpoint prediction for {compound_id}.")
        by_id[compound_id] = row
    return artifact, by_id


def _load_heavy_graph(path: Path) -> tuple[Any, int, list[int]]:
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    graph = Chem.RWMol()
    for source_index, symbol in enumerate(atoms.get_chemical_symbols()):
        atom = Chem.Atom(symbol)
        atom.SetIntProp("source_atom_index", source_index)
        graph.AddAtom(atom)
    for left, right, token_value in atoms.info["mol2"]["bonds"]:
        token = str(token_value).lower()
        if token not in _BOND_TYPES:
            raise ValueError(f"Unsupported MOL2 bond type {token_value!r}.")
        graph.AddBond(int(left), int(right), _BOND_TYPES[token])
        if token == "ar":
            graph.GetAtomWithIdx(int(left)).SetIsAromatic(True)
            graph.GetAtomWithIdx(int(right)).SetIsAromatic(True)
    molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:  # RDKit exception types are not stable.
        raise ValueError(f"RDKit could not sanitize {path.name}: {exc}") from exc
    heavy = Chem.RemoveHs(molecule)
    heavy_to_source = [
        int(heavy.GetAtomWithIdx(index).GetIntProp("source_atom_index"))
        for index in range(heavy.GetNumAtoms())
    ]
    if len(heavy_to_source) != len(set(heavy_to_source)):
        raise ValueError(f"Heavy/source atom mapping is not one-to-one: {path.name}.")
    return heavy, len(atoms), heavy_to_source


def _component_count(molecule: Any, nodes: set[int]) -> int:
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                index = int(neighbor.GetIdx())
                if index in unseen:
                    unseen.remove(index)
                    stack.append(index)
    return count


def _boundary_edges(
    molecule: Any, core: set[int], unmatched: set[int]
) -> list[tuple[int, int]]:
    return sorted(
        {
            (core_index, int(neighbor.GetIdx()))
            for core_index in core
            for neighbor in molecule.GetAtomWithIdx(core_index).GetNeighbors()
            if int(neighbor.GetIdx()) in unmatched
        }
    )


def _distance(molecule: Any, first: int, second: int) -> int:
    if first == second:
        return 0
    return len(Chem.GetShortestPath(molecule, first, second)) - 1


def _mcs_parameters(design: Mapping[str, Any]) -> Any:
    parameters = rdFMCS.MCSParameters()
    parameters.Threshold = 1.0
    parameters.MaximizeBonds = True
    parameters.Timeout = int(design["mcs_timeout_seconds"])
    parameters.Verbose = False
    parameters.AtomTyper = rdFMCS.AtomCompare.CompareElements
    parameters.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    parameters.AtomCompareParameters.RingMatchesRingOnly = True
    parameters.AtomCompareParameters.CompleteRingsOnly = True
    parameters.BondCompareParameters.RingMatchesRingOnly = True
    parameters.BondCompareParameters.CompleteRingsOnly = True
    return parameters


def _strict_mapping_candidates(
    left: Any, right: Any, design: Mapping[str, Any]
) -> tuple[str | None, list[dict[str, Any]]]:
    result = rdFMCS.FindMCS([left, right], _mcs_parameters(design))
    if result.canceled:
        return "mcs_timeout", []
    if int(result.numAtoms) < int(design["minimum_matched_heavy_atoms"]):
        return "mcs_too_small", []
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        raise ValueError("RDKit returned an invalid MCS SMARTS.")
    if len(Chem.GetMolFrags(query)) != 1:
        return "mcs_disconnected", []

    limit = int(design["maximum_mapping_enumeration"])
    left_raw_matches = list(
        left.GetSubstructMatches(query, uniquify=False, maxMatches=limit + 1)
    )
    right_raw_matches = list(
        right.GetSubstructMatches(query, uniquify=False, maxMatches=limit + 1)
    )
    if len(left_raw_matches) > limit or len(right_raw_matches) > limit:
        return "mapping_enumeration_overflow", []
    left_matches = sorted(set(left_raw_matches))
    right_matches = sorted(set(right_raw_matches))
    if len(left_raw_matches) * len(right_raw_matches) > limit:
        return "mapping_enumeration_overflow", []

    left_all = set(range(left.GetNumAtoms()))
    right_all = set(range(right.GetNumAtoms()))
    minimum_unmatched = int(design["minimum_unmatched_heavy_atoms_each_side"])
    maximum_unmatched = int(design["maximum_unmatched_heavy_atoms_each_side"])
    expected_components = int(design["unmatched_component_count_each_side"])
    expected_boundary_edges = int(
        design["core_substituent_boundary_edge_count_each_side"]
    )
    remote_distance = int(design["minimum_attachment_distance_bonds"])
    minimum_remote = int(design["minimum_remote_heavy_atoms"])

    candidates: dict[tuple[object, ...], dict[str, Any]] = {}
    for left_match in left_matches:
        left_core = set(left_match)
        if _component_count(left, left_core) != 1:
            continue
        left_unmatched = left_all - left_core
        if not minimum_unmatched <= len(left_unmatched) <= maximum_unmatched:
            continue
        if _component_count(left, left_unmatched) != expected_components:
            continue
        left_edges = _boundary_edges(left, left_core, left_unmatched)
        if len(left_edges) != expected_boundary_edges:
            continue
        left_attachment = left_edges[0][0]

        for right_match in right_matches:
            right_core = set(right_match)
            if _component_count(right, right_core) != 1:
                continue
            right_unmatched = right_all - right_core
            if not minimum_unmatched <= len(right_unmatched) <= maximum_unmatched:
                continue
            if _component_count(right, right_unmatched) != expected_components:
                continue
            right_edges = _boundary_edges(right, right_core, right_unmatched)
            if len(right_edges) != expected_boundary_edges:
                continue
            right_attachment = right_edges[0][0]

            remote_pairs = [
                (int(left_index), int(right_index))
                for left_index, right_index in zip(left_match, right_match)
                if _distance(left, int(left_index), left_attachment)
                >= remote_distance
                and _distance(right, int(right_index), right_attachment)
                >= remote_distance
            ]
            if len(remote_pairs) < minimum_remote:
                continue
            source_pairs = tuple(
                (
                    int(
                        left.GetAtomWithIdx(left_index).GetIntProp(
                            "source_atom_index"
                        )
                    ),
                    int(
                        right.GetAtomWithIdx(right_index).GetIntProp(
                            "source_atom_index"
                        )
                    ),
                )
                for left_index, right_index in remote_pairs
            )
            candidate = {
                "left_match_heavy_indices": list(map(int, left_match)),
                "right_match_heavy_indices": list(map(int, right_match)),
                "left_unmatched_heavy_indices": sorted(left_unmatched),
                "right_unmatched_heavy_indices": sorted(right_unmatched),
                "left_attachment_heavy_index": left_attachment,
                "right_attachment_heavy_index": right_attachment,
                "left_attachment_source_index": int(
                    left.GetAtomWithIdx(left_attachment).GetIntProp(
                        "source_atom_index"
                    )
                ),
                "right_attachment_source_index": int(
                    right.GetAtomWithIdx(right_attachment).GetIntProp(
                        "source_atom_index"
                    )
                ),
                "remote_source_atom_pairs": [list(pair) for pair in source_pairs],
                "mcs_heavy_atom_count": int(result.numAtoms),
                "remote_heavy_atom_count": len(remote_pairs),
            }
            key = (
                tuple(candidate["left_match_heavy_indices"]),
                tuple(candidate["right_match_heavy_indices"]),
                source_pairs,
            )
            candidates[key] = candidate
    if not candidates:
        return "no_strict_mmp_mapping", []
    return None, [candidates[key] for key in sorted(candidates)]


def _charge_metrics(
    candidate: Mapping[str, Any],
    left_charges: Sequence[float],
    right_charges: Sequence[float],
) -> dict[str, float]:
    differences = [
        float(left_charges[left_index]) - float(right_charges[right_index])
        for left_index, right_index in candidate["remote_source_atom_pairs"]
    ]
    if not differences:
        raise ValueError("A strict mapping has no remote heavy atoms.")
    squared_sum = sum(value * value for value in differences)
    return {
        "remote_heavy_atom_charge_l2_e": math.sqrt(squared_sum),
        "remote_heavy_atom_charge_rms_e": math.sqrt(
            squared_sum / len(differences)
        ),
        "remote_heavy_atom_charge_max_abs_e": max(map(abs, differences)),
    }


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _pair_id(first_id: str, second_id: str) -> str:
    digest = core.sha256_bytes(core.canonical_json_bytes([first_id, second_id]))
    return f"strict-mmp-v1:{digest}"


def _pair_record(
    first_id: str,
    second_id: str,
    first_molecule: Mapping[str, Any],
    second_molecule: Mapping[str, Any],
    design: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    reason, candidates = _strict_mapping_candidates(
        first_molecule["graph"], second_molecule["graph"], design
    )
    pair_id = _pair_id(first_id, second_id)
    if reason is not None:
        return None, {"pair_id": pair_id, "reason": reason}

    metric_rows = [
        _charge_metrics(
            candidate,
            first_molecule["charges"],
            second_molecule["charges"],
        )
        for candidate in candidates
    ]
    tolerance = float(design["mapping_metric_tolerance_e"])
    intervals = {
        name: [
            min(row[name] for row in metric_rows),
            max(row[name] for row in metric_rows),
        ]
        for name in metric_rows[0]
    }
    remote_count_interval = [
        min(int(item["remote_heavy_atom_count"]) for item in candidates),
        max(int(item["remote_heavy_atom_count"]) for item in candidates),
    ]
    if remote_count_interval[0] != remote_count_interval[1]:
        return None, {
            "pair_id": pair_id,
            "reason": "mapping_remote_count_variant",
            "structural_mapping_count": len(candidates),
            "remote_heavy_atom_count_interval": remote_count_interval,
            "mapping_metric_intervals_e": intervals,
        }
    if any(maximum - minimum > tolerance for minimum, maximum in intervals.values()):
        return None, {
            "pair_id": pair_id,
            "reason": "mapping_metric_variant",
            "structural_mapping_count": len(candidates),
            "remote_heavy_atom_count_interval": remote_count_interval,
            "mapping_metric_intervals_e": intervals,
        }

    # Mapping choice is structural and lexicographic; charge values never break ties.
    selected = candidates[0]
    metrics = metric_rows[0]
    predicted_delta = {
        endpoint: float(first_molecule["predictions"][endpoint])
        - float(second_molecule["predictions"][endpoint])
        for endpoint in ENDPOINTS
    }
    endpoint_signs = {
        endpoint: _sign(predicted_delta[endpoint]) for endpoint in ENDPOINTS
    }
    return (
        {
            "pair_id": pair_id,
            "first_id": first_id,
            "second_id": second_id,
            "mapping_status": (
                "unique"
                if len(candidates) == 1
                else "structural_but_metric_invariant"
            ),
            "mapping_enumeration_status": (
                "complete_for_single_rdkit_findmcs_smarts"
            ),
            "structural_mapping_count": len(candidates),
            "remote_heavy_atom_count_interval": remote_count_interval,
            "mapping_metric_intervals_e": intervals,
            "first_source_mol2_sha256": first_molecule["source_mol2_sha256"],
            "second_source_mol2_sha256": second_molecule["source_mol2_sha256"],
            "first_charge_vector_sha256": first_molecule[
                "charge_vector_sha256"
            ],
            "second_charge_vector_sha256": second_molecule[
                "charge_vector_sha256"
            ],
            "mapping": selected,
            **metrics,
            "predicted_delta_kcal_mol": predicted_delta,
            "endpoint_order_sign": endpoint_signs,
            "endpoint_order_disagreement": len(set(endpoint_signs.values())) > 1,
        },
        None,
    )


def _connected_components(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    adjacency: dict[str, set[str]] = {}
    for pair in pairs:
        first = str(pair["first_id"])
        second = str(pair["second_id"])
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)

    components: list[dict[str, Any]] = []
    member_to_component: dict[str, str] = {}
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        members: list[str] = []
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        members.sort()
        component_pairs = [
            str(pair["pair_id"])
            for pair in pairs
            if pair["first_id"] in members and pair["second_id"] in members
        ]
        component_digest = core.sha256_bytes(core.canonical_json_bytes(members))
        component_id = f"strict-mmp-component-v1:{component_digest}"
        for member in members:
            member_to_component[member] = component_id
        components.append(
            {
                "component_id": component_id,
                "member_count": len(members),
                "edge_count": len(component_pairs),
                "member_ids": members,
                "pair_ids_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(sorted(component_pairs))
                ),
            }
        )
    components.sort(key=lambda row: str(row["component_id"]))
    return components, member_to_component


def build_pair_artifact(
    *,
    protocol_path: Path,
    source_manifest_path: Path,
    energy_artifact_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    protocol, fingerprint = load_protocol(protocol_path)
    _manifest, source_by_id = _load_source_manifest(
        protocol, source_manifest_path
    )
    energy, energy_by_id = _load_energy_artifact(
        protocol, energy_artifact_path, source_by_id
    )
    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    molecules: dict[str, dict[str, Any]] = {}
    molecule_audit: list[dict[str, Any]] = []
    for compound_id in sorted(source_by_id):
        source = source_by_id[compound_id]
        energy_row = energy_by_id[compound_id]
        mol2_path = _safe_child(root, source["source_mol2_relative_path"])
        if core.sha256_file(mol2_path) != source["source_mol2_sha256"]:
            raise ValueError(f"Pinned MOL2 hash mismatch for {compound_id}.")
        graph, atom_count, heavy_to_source = _load_heavy_graph(mol2_path)
        if atom_count != int(source["atom_count"]):
            raise ValueError(f"MOL2 atom-count mismatch for {compound_id}.")
        molecule = {
            "graph": graph,
            "charges": [float(value) for value in energy_row["am1bcc_charges_e"]],
            "predictions": {
                endpoint: energy_row["predictions_kcal_mol"][endpoint]
                for endpoint in ENDPOINTS
            },
            "source_mol2_sha256": source["source_mol2_sha256"],
            "charge_vector_sha256": energy_row["am1bcc_charge_vector_sha256"],
        }
        molecules[compound_id] = molecule
        molecule_audit.append(
            {
                "compound_id": compound_id,
                "source_mol2_sha256": source["source_mol2_sha256"],
                "charge_vector_sha256": energy_row[
                    "am1bcc_charge_vector_sha256"
                ],
                "all_atom_count": atom_count,
                "heavy_atom_count": graph.GetNumAtoms(),
                "heavy_rdkit_to_source_and_charge_index": [
                    {
                        "heavy_rdkit_index": index,
                        "source_mol2_atom_index": source_index,
                        "charge_vector_index": source_index,
                    }
                    for index, source_index in enumerate(heavy_to_source)
                ],
            }
        )

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    design = protocol["pair_audit_design"]
    for first_id, second_id in itertools.combinations(sorted(molecules), 2):
        record, exclusion = _pair_record(
            first_id,
            second_id,
            molecules[first_id],
            molecules[second_id],
            design,
        )
        if record is not None:
            accepted.append(record)
        elif exclusion is not None:
            excluded.append(exclusion)

    components, member_to_component = _connected_components(accepted)
    for record in accepted:
        first_component = member_to_component[str(record["first_id"])]
        second_component = member_to_component[str(record["second_id"])]
        if first_component != second_component:
            raise AssertionError("Accepted MMP edge crosses its graph component.")
        record["component_id"] = first_component
    accepted.sort(key=lambda row: str(row["pair_id"]))
    excluded.sort(key=lambda row: str(row["pair_id"]))
    exclusion_counts = Counter(row["reason"] for row in excluded)
    mapping_variant_sensitivity = [
        row
        for row in excluded
        if row["reason"]
        in {"mapping_metric_variant", "mapping_remote_count_variant"}
    ]

    candidate_count = len(molecules) * (len(molecules) - 1) // 2
    if candidate_count != len(accepted) + len(excluded):
        raise AssertionError("Strict-MMP candidate accounting does not close.")
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": core.sha256_file(protocol_path),
        "protocol_fingerprint": fingerprint,
        "route1_boundary": ROUTE1_BOUNDARY,
        "label_boundary": {
            "experimental_labels_read": False,
            "score_artifact_read": False,
            "series_artifact_read": False,
            "historical_label_exposure": True,
        },
        "source_manifest_sha256": protocol["source_evidence"][
            "source_manifest_sha256"
        ],
        "energy_artifact_sha256": protocol["source_evidence"][
            "energy_artifact_sha256"
        ],
        "energy_artifact_content_sha256": energy["content_sha256"],
        "rdkit_version": rdBase.rdkitVersion,
        "pair_audit_design": design,
        "case_count": len(molecules),
        "candidate_pair_count": candidate_count,
        "evaluable_pair_count": len(accepted),
        "excluded_pair_count": len(excluded),
        "excluded_reason_counts": dict(sorted(exclusion_counts.items())),
        "excluded_pairs_sha256": core.sha256_bytes(
            core.canonical_json_bytes(excluded)
        ),
        "mapping_variant_sensitivity_count": len(
            mapping_variant_sensitivity
        ),
        "mapping_variant_sensitivity": mapping_variant_sensitivity,
        "mapping_ambiguous_but_metric_invariant_count": sum(
            row["mapping_status"] == "structural_but_metric_invariant"
            for row in accepted
        ),
        "endpoint_order_disagreement_count": sum(
            bool(row["endpoint_order_disagreement"]) for row in accepted
        ),
        "component_count": len(components),
        "components": components,
        "molecule_audit": molecule_audit,
        "pairs": accepted,
        "decision": {
            "accuracy_claim_available": False,
            "charge_model_comparison_available": False,
            "threshold_selection_allowed": False,
            "endpoint_selection_allowed": False,
            "certified_ranking_available": False,
            "status": "label_free_charge_continuity_sidecar_only",
        },
    }
    _reject_label_fields(artifact)
    return core.seal_artifact(artifact)


def build(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    energy_artifact_path = Path(args.energy_artifact).resolve()
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    artifact = build_pair_artifact(
        protocol_path=protocol_path,
        source_manifest_path=source_manifest_path,
        energy_artifact_path=energy_artifact_path,
        source_root=source_root,
    )
    artifact["command_provenance"] = core.command_provenance(
        __file__,
        {
            "phase": "build-label-free-charge-continuity-pairs",
            "protocol": _relative_to_repository(protocol_path),
            "source_manifest": _relative_to_repository(source_manifest_path),
            "energy_artifact": _relative_to_repository(energy_artifact_path),
            "source_root": _relative_to_repository(source_root),
            "output": _relative_to_repository(output),
        },
        repository_root=REPOSITORY_ROOT,
    )
    artifact = core.seal_artifact(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "candidate_pair_count": artifact["candidate_pair_count"],
                "evaluable_pair_count": artifact["evaluable_pair_count"],
                "component_count": artifact["component_count"],
            },
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--energy-artifact", type=Path, default=DEFAULT_ENERGY_ARTIFACT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
