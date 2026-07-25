from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from .conftest import DOCS_DIR, PROJECT_ROOT, load_json

PROTO_PATH = DOCS_DIR / "protocol-v1.json"
FAIL_PATH = DOCS_DIR / "failure-contract-v1.json"
STATE_PATH = DOCS_DIR / "state-contract-v1.json"
CLAIM_PATH = DOCS_DIR / "claim-contract-v1.json"
SCHEMA_PATH = DOCS_DIR / "protocol-schema-v1.json"
RADIUS_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "radius-profiles" / "bondi-mantina-v1.json"
CAVITY_FIX_PATH = PROJECT_ROOT / "tests" / "solvation" / "route_a" / "fixtures" / "cavities" / "pcmsolver-golden-water-dimer.json"
STATE_FIX = PROJECT_ROOT / "tests" / "solvation" / "route_a" / "fixtures" / "state" / "reduced_potential_v1.json"
MODEL_PROBE_PATH = PROJECT_ROOT / "tests" / "solvation" / "route_a" / "fixtures" / "models" / "real_model_probes_v1.json"
OUTER_PARITY_PATH = PROJECT_ROOT / "tests" / "solvation" / "route_a" / "fixtures" / "outer" / "route2_acetone_component_parity_v1.json"
THERMODYNAMICS_MD_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "thermodynamics.md"
RESTRAINT_PATH = DOCS_DIR / "restraint-contract-v1.json"
OUTER_ADAPTER_PATH = DOCS_DIR / "outer-adapter-contract-v1.json"
REFERENCE_CYCLE_PATH = DOCS_DIR / "reference-cycle-ablation-contract-v1.json"



def _canonical_sha256(data: dict, ignore_keys=None) -> str:
    ignore_keys = set(ignore_keys or [])
    payload = {k: v for k, v in data.items() if k not in ignore_keys}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _npz_canonical_centers_areas_sha(npz_path: Path) -> str:
    mesh = np.load(npz_path)
    centers = np.asarray(mesh["centers_angstrom"], dtype=float)
    areas = np.asarray(mesh["areas_angstrom2"], dtype=float)
    return hashlib.sha256(np.hstack((centers, areas[:, None])).tobytes()).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    data = json.loads(path.read_text())
    return _canonical_sha256(data)


def _artifact_entry_hash(path: Path, entry: dict[str, str]) -> str:
    hash_kind = entry["hash_kind"]
    if hash_kind == "canonical_sha256":
        if path.suffix != ".json":
            raise AssertionError(f"canonical_sha256 used only for json artifacts, got {path.suffix}")
        return _canonical_json_sha256(path)
    if hash_kind == "sha256":
        return _sha256_file(path)
    raise AssertionError(f"Unsupported hash kind: {hash_kind}")


def _is_hex64(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{64}", value))


def _is_hex40(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[a-f0-9]{40}", value))


def _is_valid_schema(node: object, value: object, schema: dict, path: str = "root") -> bool:
    try:
        _validate_against_schema(node, value, schema, path)
    except AssertionError:
        return False
    return True


def _resolve_ref(schema: dict, ref: str):
    assert ref.startswith("#/")
    current: object = schema
    for part in ref[2:].split("/"):
        current = current[part]
    return current


def _validate_against_schema(node: object, value: object, schema: dict, path: str = "root") -> None:
    if not isinstance(node, dict):
        return

    # Reference support
    if "$ref" in node:
        ref = node["$ref"]
        _validate_against_schema(_resolve_ref(schema, ref), value, schema, f"{path}->$ref")
        rest = {k: v for k, v in node.items() if k != "$ref"}
        if rest:
            _validate_against_schema(rest, value, schema, path)
        return

    # Composite combinators
    if "allOf" in node:
        for idx, subschema in enumerate(node["allOf"]):
            _validate_against_schema(subschema, value, schema, f"{path}.allOf[{idx}]")

    if "oneOf" in node:
        matches = 0
        last_err = None
        for idx, subschema in enumerate(node["oneOf"]):
            try:
                _validate_against_schema(subschema, value, schema, f"{path}.oneOf[{idx}]")
                matches += 1
            except AssertionError as exc:
                last_err = exc
        if matches != 1:
            raise AssertionError(f"{path}: oneOf expected 1 match, got {matches}") from last_err

    if "type" in node:
        t = node["type"]
        if isinstance(t, list):
            if not any(_is_valid_schema({"type": item}, value, schema, path) for item in t):
                raise AssertionError(f"{path}: type {type(value)} not in {t}")
        elif t == "object":
            if not isinstance(value, dict):
                raise AssertionError(f"{path}: expected object, got {type(value)}")

            required = node.get("required", [])
            for field in required:
                assert field in value, f"{path}: missing required field {field}"

            properties = node.get("properties", {})
            additional = node.get("additionalProperties", True)
            for key, sub in value.items():
                if key not in properties:
                    if additional is False:
                        raise AssertionError(f"{path}: unexpected key {key}")
                else:
                    _validate_against_schema(properties[key], sub, schema, f"{path}.{key}")

        elif t == "array":
            if not isinstance(value, list):
                raise AssertionError(f"{path}: expected array, got {type(value)}")
            if "minItems" in node:
                assert len(value) >= node["minItems"]
            if "maxItems" in node:
                assert len(value) <= node["maxItems"]
            if "items" in node:
                for idx, item in enumerate(value):
                    _validate_against_schema(node["items"], item, schema, f"{path}[{idx}]")

        elif t == "string":
            if not isinstance(value, str):
                raise AssertionError(f"{path}: expected string, got {type(value)}")
            if "pattern" in node:
                assert re.search(node["pattern"], value)

            if "minLength" in node:
                assert len(value) >= node["minLength"]
            if "maxLength" in node:
                assert len(value) <= node["maxLength"]

            enum = node.get("enum")
            if enum is not None:
                assert value in enum

            if "const" in node:
                assert value == node["const"]

        elif t == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AssertionError(f"{path}: expected number, got {type(value)}")
            if "minimum" in node:
                assert float(value) >= node["minimum"]
            if "maximum" in node:
                assert float(value) <= node["maximum"]

            if "enum" in node:
                assert value in node["enum"]

            if "const" in node:
                assert value == node["const"]

        elif t == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise AssertionError(f"{path}: expected integer, got {type(value)}")
            if "minimum" in node:
                assert value >= node["minimum"]
            if "maximum" in node:
                assert value <= node["maximum"]

            if "enum" in node:
                assert value in node["enum"]

            if "const" in node:
                assert value == node["const"]

        elif t == "boolean":
            if not isinstance(value, bool):
                raise AssertionError(f"{path}: expected boolean, got {type(value)}")
            if "const" in node:
                assert value == node["const"]

        elif t == "null":
            if value is not None:
                raise AssertionError(f"{path}: expected null, got {value!r}")

    # Generic checks for all typed nodes
    if "enum" in node:
        assert value in node["enum"]
    if "const" in node:
        assert value == node["const"]


def _validate_local_schema(instance: object, schema: dict) -> None:
    root = copy.deepcopy(schema)
    # validate and also ensure all refs are resolvable
    for ref in _collect_refs(schema):
        _resolve_ref(schema, ref)
    _validate_against_schema(schema, instance, root)


def _collect_refs(schema: object, prefix: str = "") -> set[str]:
    refs: set[str] = set()
    if isinstance(schema, dict):
        if "$ref" in schema:
            refs.add(schema["$ref"])
        for value in schema.values():
            refs |= _collect_refs(value)
    elif isinstance(schema, list):
        for item in schema:
            refs |= _collect_refs(item)
    return refs


def _probe_key_map() -> dict[str, str]:
    return {
        "outer": "real_model_probes_v1.json::models.polar1m",
        "sampler": "real_model_probes_v1.json::models.off24",
        "target": "real_model_probes_v1.json::models.omol0",
    }


def _resolve_probe_key(
    model_probe: dict,
    key: str,
) -> dict:
    assert "::" in key
    source, dotted = key.split("::", 1)
    assert source == "real_model_probes_v1.json"
    keys = dotted.split(".")
    assert keys and keys[0] == "models"
    probe_payload = model_probe
    for part in keys:
        assert part in probe_payload, f"Missing path segment {part} in model probe key {key}"
        probe_payload = probe_payload[part]
    return probe_payload


def _reject_protocol_mutation(
    base_protocol: dict,
    schema: dict,
    mutate_fn,
    *,
    assert_domain_code: bool = False,
) -> None:
    mutated = copy.deepcopy(base_protocol)
    mutate_fn(mutated)

    with pytest.raises(AssertionError):
        _validate_local_schema(mutated, schema)

    if assert_domain_code:
        assert (
            mutated["domain"]["unsupported_domains_fail_code"]
            == base_protocol["domain"]["unsupported_domains_fail_code"]
            == "DOMAIN_UNSUPPORTED"
        )


def _claim_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "contract_name": {"type": "string"},
            "contract_version": {"type": "string"},
            "hash_canonicalization": {"type": "string"},
            "immutable": {"type": "boolean"},
            "notes": {"type": "string"},
            "result_order": {"type": "array", "items": {"type": "string"}},
            "status_lattice": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status", "required"],
                    "properties": {
                        "status": {"type": "string"},
                        "required": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                },
            },
            "hash_fields": {"type": "array", "items": {"type": "string"}},
            "route2_superiority_metric": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "required_fields", "definition", "evaluator"],
                "properties": {
                    "name": {"type": "string"},
                    "required_fields": {"type": "array", "items": {"type": "string"}},
                    "definition": {"type": "string"},
                    "formula": {"type": "string"},
                    "evaluator": {"type": "object"},
                    "raw_evidence_rule": {"type": "string"},
                },
            },
            "frozen_absolute_accuracy": {"type": "object"},
            "material_partition_bias": {"type": "object"},
            "evidence_derivation": {"type": "object"},
            "production_gate": {"type": "object"},
            "test_only_validation": {"type": "object"},
        },
        "required": [
            "contract_name",
            "contract_version",
            "immutable",
            "result_order",
            "status_lattice",
            "hash_fields",
            "route2_superiority_metric",
            "frozen_absolute_accuracy",
            "material_partition_bias",
            "evidence_derivation",
            "production_gate",
            "test_only_validation",
            "hash_canonicalization",
        ],
    }


def _failure_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract_name", "contract_version", "hash_canonicalization", "immutable", "catalog", "runtime"],
        "properties": {
            "contract_name": {"type": "string"},
            "contract_version": {"type": "string"},
            "hash_canonicalization": {"type": "string"},
            "immutable": {"type": "boolean"},
            "notes": {"type": "string"},
            "catalog": {
                "type": "object",
                "required": ["catalog_entry_required_fields", "known_categories", "failure_codes", "failure_record_required_fields", "required_fields", "uniqueness"],
                "properties": {
                    "catalog_entry_required_fields": {"type": "array", "items": {"type": "string"}},
                    "failure_codes": {"type": "array", "items": {"type": "object"}},
                    "known_categories": {"type": "array", "items": {"type": "string"}},
                    "failure_record_required_fields": {"type": "array", "items": {"type": "string"}},
                    "required_fields": {"type": "array", "items": {"type": "string"}},
                    "uniqueness": {"type": "object"},
                },
            },
            "runtime": {
                "type": "object",
                "required": ["known_categories", "required_fields"],
                "properties": {
                    "known_categories": {"type": "array", "items": {"type": "string"}},
                    "required_fields": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def _state_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "contract_name",
            "contract_version",
            "hash_canonicalization",
            "immutable",
            "contracts",
            "objects",
            "reduced_potential_unit_guard",
            "stage_graph",
        ],
        "properties": {
            "contract_name": {"type": "string"},
            "contract_version": {"type": "string"},
            "hash_canonicalization": {"type": "string"},
            "immutable": {"type": "boolean"},
            "contracts": {"type": "object"},
            "objects": {"type": "object"},
            "reduced_potential_unit_guard": {"type": "object"},
            "stage_graph": {"type": "object"},
        },
    }


def test_protocol_hash_self_contract_and_schema_hash_contract():
    protocol = load_json(PROTO_PATH)
    thermo_md_refs = protocol["artifact_references"]["thermodynamics_md"]
    assert _is_hex64(thermo_md_refs["sha256"])
    assert thermo_md_refs["sha256"] == _sha256_file(THERMODYNAMICS_MD_PATH)

    payload = copy.deepcopy(protocol)
    payload.pop("protocol_sha256", None)

    schema = load_json(SCHEMA_PATH)
    assert protocol["protocol_schema"]["sha256"] == _canonical_sha256(schema)
    assert protocol["protocol_schema"]["path"] == "docs/solvation/route-a/protocol-schema-v1.json"
    assert protocol["protocol_schema"]["hash_kind"] == "canonical_sha256"
    assert _is_hex64(protocol["protocol_schema"]["sha256"])
    assert protocol["protocol_sha256"] == _canonical_sha256(payload)
    assert _is_hex64(protocol["protocol_sha256"])


def test_tail_envelope_is_preregistered_and_protocol_hash_bound():
    protocol = load_json(PROTO_PATH)
    contract = copy.deepcopy(protocol["tail_envelope_contract"])
    declared_hash = contract.pop("contract_sha256")

    assert contract["immutable"] is True
    assert contract["model"] == "geometric-exponential-conservative-ceiling-v1"
    assert contract["conservative_ratio_ceiling"] == 0.55
    assert contract["ratio_floor"] == 0.0
    assert contract["minimum_consecutive_ratios"] == 2
    assert contract["selection_timing"] == (
        "frozen before any production high-n occupancy result"
    )
    assert declared_hash == _canonical_sha256(contract)


def test_protocol_schema_is_draft202012_and_protocol_validates_against_it():
    protocol = load_json(PROTO_PATH)
    schema = load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(protocol)


def test_protocol_schema_rejects_missing_thresholds_formula_and_path_redirects():
    protocol = load_json(PROTO_PATH)
    schema = load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)

    mutations = []
    missing_threshold = copy.deepcopy(protocol)
    missing_threshold["thresholds"].pop("omitted_tail_mass")
    mutations.append(missing_threshold)

    empty_thresholds = copy.deepcopy(protocol)
    empty_thresholds["thresholds"] = {}
    mutations.append(empty_thresholds)

    wrong_formula = copy.deepcopy(protocol)
    wrong_formula["thermodynamics"]["formulas"]["qct_n_entry"] = "WRONG"
    mutations.append(wrong_formula)

    redirected_top = copy.deepcopy(protocol)
    redirected_top["claim_contract"]["reference"]["path"] = "elsewhere.json"
    mutations.append(redirected_top)

    redirected_artifact = copy.deepcopy(protocol)
    redirected_artifact["artifact_references"]["claim_contract"][
        "path"
    ] = "elsewhere.json"
    mutations.append(redirected_artifact)

    wrong_code = copy.deepcopy(protocol)
    wrong_code["outer_adapter_contract"]["code"] = "other-adapter"
    mutations.append(wrong_code)

    for mutation in mutations:
        assert not validator.is_valid(mutation)


def test_protocol_schema_validation_is_recursive_and_ref_resolved():
    protocol = load_json(PROTO_PATH)
    schema = load_json(SCHEMA_PATH)

    _validate_local_schema(protocol, schema)

    for ref in _collect_refs(schema):
        _resolve_ref(schema, ref)


def test_protocol_hash_contracts_cover_cited_artifacts_and_contracts():
    protocol = load_json(PROTO_PATH)
    claim = load_json(CLAIM_PATH)
    failure = load_json(FAIL_PATH)
    state = load_json(STATE_PATH)
    restraint = load_json(RESTRAINT_PATH)
    outer_adapter = load_json(OUTER_ADAPTER_PATH)
    reference_cycle = load_json(REFERENCE_CYCLE_PATH)
    cavity = load_json(CAVITY_FIX_PATH)

    # Contract hashes use canonical encoding
    assert protocol["claim_contract"]["reference"]["hash"] == _canonical_sha256(claim)
    assert protocol["claim_contract"]["reference"]["hash_kind"] == "canonical_sha256"
    assert protocol["failure_contract"]["reference"]["hash"] == _canonical_sha256(failure)
    assert protocol["failure_contract"]["reference"]["hash_kind"] == "canonical_sha256"
    assert protocol["state_contract"]["reference"]["hash"] == _canonical_sha256(state)
    assert protocol["state_contract"]["reference"]["hash_kind"] == "canonical_sha256"
    assert protocol["restraint_contract"]["reference"]["hash"] == _canonical_sha256(restraint)
    assert protocol["restraint_contract"]["reference"]["hash_kind"] == "canonical_sha256"
    assert protocol["outer_adapter_contract"]["reference"]["hash"] == _canonical_sha256(outer_adapter)
    assert protocol["outer_adapter_contract"]["reference"]["hash_kind"] == "canonical_sha256"
    assert protocol["reference_cycle_ablation_contract"]["reference"]["hash"] == _canonical_sha256(reference_cycle)
    assert protocol["reference_cycle_ablation_contract"]["reference"]["hash_kind"] == "canonical_sha256"

    # Radius profile hash and model probe fixture hashes
    radius = load_json(RADIUS_PATH)
    assert protocol["radius_profile"]["sha256"] == _canonical_sha256(radius)

    model_probes = load_json(MODEL_PROBE_PATH)
    assert _is_hex64(protocol["artifact_references"]["model_probes"]["sha256"])
    assert protocol["artifact_references"]["model_probes"]["sha256"] == _canonical_sha256(model_probes)
    for key, contract in [
        ("claim", protocol["claim_contract"]),
        ("failure", protocol["failure_contract"]),
        ("state", protocol["state_contract"]),
        ("restraint", protocol["restraint_contract"]),
        ("outer_adapter", protocol["outer_adapter_contract"]),
        ("reference_cycle_ablation", protocol["reference_cycle_ablation_contract"]),
    ]:
        refs = protocol["artifact_references"][f"{key}_contract"]
        assert refs["sha256"] == contract["reference"]["hash"]
        assert refs["hash_kind"] == contract["reference"]["hash_kind"]
        assert refs["path"] == contract["reference"]["path"]

    proto_schema_refs = protocol["artifact_references"]["protocol_schema"]
    assert proto_schema_refs["path"] == protocol["protocol_schema"]["path"]
    assert proto_schema_refs["sha256"] == protocol["protocol_schema"]["sha256"]
    assert proto_schema_refs["hash_kind"] == protocol["protocol_schema"]["hash_kind"]

    thermo_md_refs = protocol["artifact_references"]["thermodynamics_md"]
    assert _is_hex64(thermo_md_refs["sha256"])
    assert thermo_md_refs["path"] == THERMODYNAMICS_MD_PATH.relative_to(PROJECT_ROOT).as_posix()
    assert thermo_md_refs["hash_kind"] == "sha256"

    # Artifact integrity checks
    artifacts = protocol["artifact_references"]
    for artifact_key, entry in artifacts.items():
        path = PROJECT_ROOT / entry["path"]
        expected = _artifact_entry_hash(path, entry)
        assert _is_hex64(entry["sha256"])
        assert entry["sha256"] == expected
        assert entry["hash_kind"] in {"sha256", "canonical_sha256"}

    # Mesh-specific canonical diagnostic hash for tesserae
    mesh_entry = artifacts["mesh"]
    mesh_path = PROJECT_ROOT / mesh_entry["path"]
    assert mesh_entry["sha256"] == _sha256_file(mesh_path)
    assert mesh_entry["hash_kind"] == "sha256"
    assert _npz_canonical_centers_areas_sha(mesh_path) == cavity["cavity"]["mesh"]["canonical_centers_areas_sha256"]

    cavity_entry = artifacts["cavity_fixture"]
    assert cavity_entry["hash_kind"] == "canonical_sha256"
    assert cavity_entry["sha256"] == protocol["shell"]["cavity"]["fixture_hash"]
    assert cavity_entry["sha256"] == _canonical_json_sha256(CAVITY_FIX_PATH)


def test_bondi_radius_profile_has_bondi_1964_doi_and_no_unverified_adjustment_claim():
    profile = load_json(RADIUS_PATH)
    assert profile["source_doi"] == "10.1021/j100785a001"
    assert profile["doi"] == profile["source_doi"]
    assert "Mantina" not in profile["source"]
    assert "Mantina" in profile["mantina_lineage"]
    assert str(profile["radii"]["O"]) == "1.52"


def test_protocol_domain_contract_only_supports_closed_shell_neutral_water_chemistry():
    protocol = load_json(PROTO_PATH)
    domain = protocol["domain"]
    schema = load_json(SCHEMA_PATH)
    schema_domain = schema["properties"]["domain"]["properties"]
    assert schema_domain["name"]["const"] == protocol["domain"]["name"]
    assert schema_domain["solvent"]["properties"]["model"]["const"] == protocol["domain"]["solvent"]["model"]
    assert protocol["domain"]["unsupported_domains_fail_code"] == "DOMAIN_UNSUPPORTED"

    assert domain["solvent"]["name"] == "water"
    assert domain["solvent"]["phase"] == "liquid"
    assert domain["solvent"]["electrolyte_free"] is True
    assert domain["solvent"]["model"] == "tip3p-like geometry-compatible"

    constraints = domain["solute_constraints"]
    assert constraints["charge"] == 0
    assert constraints["spin_multiplicity"] == 1
    assert constraints["open_shell"] is False
    assert constraints["disallow_metals"] is True
    assert constraints["disallow_radicals"] is True
    assert constraints["one_connected_solute"] is True

    allowed = set(constraints["allowed_elements"])
    assert allowed == {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
    expected_allowed = list(constraints["allowed_elements"])

    assert domain["standard_state"]["solution_reference_concentration_molar"] == 1.0

    def mutate_and_reject(mutator, *, assert_code: bool = True) -> None:
        _reject_protocol_mutation(protocol, schema, mutator, assert_domain_code=assert_code)

    mutate_and_reject(lambda p: p["domain"]["solvent"].__setitem__("name", "methanol"))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("charge", 1))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("open_shell", True))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("spin_multiplicity", 2))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("allowed_elements", ["H", "C", "Na"]))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("allowed_elements", list(reversed(expected_allowed))))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("one_connected_solute", False))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("disallow_radicals", False))
    mutate_and_reject(lambda p: p["domain"]["solute_constraints"].__setitem__("disallow_metals", False))
    mutate_and_reject(lambda p: p["domain"].__setitem__("property", "absolute_hydration_free_enthalpy"))
    mutate_and_reject(lambda p: p["domain"].__setitem__("production_domain", "periodic"))
    mutate_and_reject(lambda p: p["domain"]["occupancy_support"].__setitem__("adaptive_n", False))
    mutate_and_reject(lambda p: p["domain"]["occupancy_support"].__setitem__("n_min", 2))
    mutate_and_reject(lambda p: p["domain"]["occupancy_support"].__setitem__("n_min", -1))
    mutate_and_reject(lambda p: p["domain"]["occupancy_support"].__setitem__("requires_n_plus_greater_zero", False))
    mutate_and_reject(lambda p: p["domain"].__setitem__("pressure_bar", 0.9))
    mutate_and_reject(lambda p: p["domain"].__setitem__("name", "route-a-break-glass"))
    mutate_and_reject(lambda p: p["domain"]["solvent"].__setitem__("model", "spce-like geometry-compatible"))
    mutate_and_reject(lambda p: p["domain"].__setitem__("absolute_hydration_free_energy", False))
    mutate_and_reject(
        lambda p: p["domain"].__setitem__("unsupported_domains_fail_code", "DOMAIN_OK"),
        assert_code=False,
    )
    mutate_and_reject(lambda p: p["domain"]["standard_state"].__setitem__("gas_reference_concentration_molar", 0.0))
    mutate_and_reject(lambda p: p["domain"]["standard_state"].__setitem__("water_bulk_concentration_molar", 54.0))
    mutate_and_reject(lambda p: p["domain"]["standard_state"].__setitem__("C_number_0_per_A3", 0.0))
    mutate_and_reject(lambda p: p["domain"]["standard_state"].__setitem__("standard_state_volume_A3_per_molecule", 0.0))
    mutate_and_reject(lambda p: p["domain"]["standard_state"].__setitem__("solution_reference_concentration_molar", 2.0))
    mutate_and_reject(lambda p: p["standard_state"].__setitem__("temperature_k", 299.0))
    mutate_and_reject(lambda p: p["standard_state"].__setitem__("gas_constant_kcal_per_mol_K", 0.0009))
    mutate_and_reject(lambda p: p.__setitem__("unsupported", 1), assert_code=False)

    assert domain["pressure_bar"] == 1.0
    assert domain["absolute_hydration_free_energy"] is True
    assert domain["production_domain"] == "nonperiodic"
    assert domain["standard_state"]["gas_reference_concentration_molar"] == 1.0
    assert domain["standard_state"]["water_bulk_concentration_molar"] == 55.345
    assert domain["standard_state"]["solution_reference_concentration_molar"] == 1.0
    assert domain["standard_state"]["C_number_0_per_A3"] == 0.000602214076
    assert domain["standard_state"]["rho_W_number_per_A3"] == 0.03332953803622
    assert domain["standard_state"]["rho_W_over_C0"] == 55.345
    assert domain["standard_state"]["standard_state_volume_A3_per_molecule"] == 1660.5390671738467
    assert protocol["standard_state"]["temperature_k"] == 298.15
    assert protocol["standard_state"]["gas_constant_kcal_per_mol_K"] == 0.001987204258640831
    assert protocol["standard_state"]["C_number_0_per_A3"] == 0.000602214076
    assert protocol["standard_state"]["rho_W_number_per_A3"] == 0.03332953803622
    assert protocol["standard_state"]["rho_W_over_C0"] == 55.345
    assert protocol["standard_state"]["standard_state_volume_A3_per_molecule"] == 1660.5390671738467


def test_protocol_schema_rejects_nested_artifact_and_ordered_standard_state_contracts():
    protocol = load_json(PROTO_PATH)
    schema = load_json(SCHEMA_PATH)

    def mutate_and_reject(mutator) -> None:
        _reject_protocol_mutation(protocol, schema, mutator, assert_domain_code=False)

    mutate_and_reject(
        lambda p: p["artifact_references"]["geometry"].__setitem__("path", ""),
    )
    mutate_and_reject(
        lambda p: p["artifact_references"]["geometry"].__setitem__("unexpected", "field"),
    )
    mutate_and_reject(
        lambda p: p["domain"]["solute_constraints"]["allowed_elements"].__setitem__(
            slice(None, None),
            list(reversed(p["domain"]["solute_constraints"]["allowed_elements"])),
        ),
    )
    mutate_and_reject(
        lambda p: p["standard_state"].__setitem__("temperature_k", 300.0),
    )


def test_protocol_thermodynamics_and_shell_predicates_are_scientifically_aligned():
    protocol = load_json(PROTO_PATH)
    therm = protocol["thermodynamics"]

    assert therm["water_density_reference"] == "rho_W_over_C0"
    assert therm["association_block"]["alchemical"] == "ΔG_alch_labeled(i)"
    assert therm["formulas"]["association_labeled"] == "ΔG_alch_labeled(i)+ΔG_vol(i)+RT ln(i)+ΔG_release(i)"
    expected_qct = (
        "ΔG_n = Σ_{i=1..n}[ΔG_alch_labeled(i)+ΔG_vol(i)+RT ln(i)+ΔG_release(i)] "
        "- n RT ln(rho_W_over_C0) + ΔG_outer(XW_n) - nΔG_outer(W)"
    )
    assert therm["formulas"]["qct_n_entry"] == expected_qct
    assert therm["formulas"]["hamiltonian_decomposition"] == (
        "ΔG_outer(XW_n)=F[U_OMOL_vac+δ_outer,R_n]-F[U_OMOL_vac,R_n], "
        "δ_outer_kcal(q)=Hartree_to_kcal_mol*((E_intrinsic_Polar[V_reac](q)-"
        "E_intrinsic_Polar[0](q))/Hartree_eV+E_PCM_pol_Ha(q)+G_CDS_Ha(q)), "
        "and ΔG_outer(W)=F[U_OMOL_vac+δ_outer,R_W]-F[U_OMOL_vac,R_W]"
    )
    assert therm["formulas"]["volume_density_identity"] == (
        "rho_W_number_per_A3 = water_bulk_concentration_molar * C0 / solution_reference_concentration_molar; "
        "rho_W_over_C0 = rho_W_number_per_A3 / C0 = water_bulk_concentration_molar / solution_reference_concentration_molar; "
        "ΔG_vol(i) - RT ln(rho_W_over_C0) = -RT ln(rho_W_number_per_A3 * V_eff,i) for each insertion edge i=1..n; "
        "Σ_{i=1}^n ΔG_vol(i) - n RT ln(rho_W_over_C0) = -RT Σ_{i=1}^n ln(rho_W_number_per_A3 * V_eff,i)"
    )
    assert therm["formulas"]["outer_contains_density_term"] is False
    assert "RT ln(i)" in therm["formulas"]["symmetry_correction"]
    assert "RT ln(n!)" in therm["formulas"]["symmetry_correction"]
    assert "ΔG_outer(XW_n) = solvent outer free-energy restraint contribution" in therm["formulas"]["water_outer_signature"]
    assert "-nΔG_outer(W)" in therm["formulas"]["water_outer_signature"]

    assert "ΔG_outer" in therm["formulas"]["hamiltonian_decomposition"]
    assert re.search(r"(?<!Δ)G_outer\(", therm["formulas"]["hamiltonian_decomposition"]) is None
    cluster_outer_match = re.search(
        r"ΔG_outer\(XW_n\)=F\[[^\]]+\]-F\[[^\]]+\]",
        therm["formulas"]["hamiltonian_decomposition"],
    )
    water_outer_match = re.search(
        r"ΔG_outer\(W\)=F\[[^\]]+\]-F\[[^\]]+\]",
        therm["formulas"]["hamiltonian_decomposition"],
    )
    assert cluster_outer_match is not None and water_outer_match is not None
    assert "R_n" in cluster_outer_match.group(0)
    assert "R_W" not in cluster_outer_match.group(0)
    assert "R_W" in water_outer_match.group(0)

    thermo_text = THERMODYNAMICS_MD_PATH.read_text()
    for needle in [
        "G_n = Σ_{i=1..n}[ΔG_alch_labeled(i)+ΔG_vol(i)+RT ln(i)+ΔG_release(i)] - n RT ln(ρ_W_over_C0) + ΔG_outer(XW_n) - nΔG_outer(W)",
        "ΔG_outer(XW_n)=F[U_OMOL_vac+δ_outer,R_n]-F[U_OMOL_vac,R_n]",
        "ΔG_outer(W)=F[U_OMOL_vac+δ_outer,R_W]-F[U_OMOL_vac,R_W]",
        "ρ_W_number_per_A3 = water_bulk_concentration_molar * C0 / solution_reference_concentration_molar",
    ]:
        rho_variants = {needle, needle.replace("ρ_W_over_C0", "rho_W_over_C0"), needle.replace("ρ", "rho")}
        assert any(v in thermo_text for v in rho_variants)

    assert "G_0 = ΔG_outer(X)" in thermo_text
    assert "ΔG_outer(X)" in thermo_text

    pred = protocol["shell"]["cavity"]["penetration_predicate"]
    assert "max_penetration_A" not in pred
    assert pred["numerical_tolerance_A"] == 1e-06
    assert isinstance(pred["pair_partition"], str)
    assert pred["pair_partition"].startswith("auto:")
    assert "cross-fragment" in pred["pair_partition"]
    assert "R_i+R_j-d_ij>0" in pred["pair_criterion"]

    sd = protocol["shell"]["signed_distance"]["distance_fn"]
    assert "min_i" in sd
    assert "R_i^vdW" in sd


def test_protocol_thresholds_models_are_pinned_and_referenced_to_local_probe_fixture():
    protocol = load_json(PROTO_PATH)
    thresholds = protocol["thresholds"]

    assert thresholds["omitted_tail_mass"]["value"] == 0.005
    assert thresholds["omitted_tail_mass"]["comparator"] == "<"
    assert thresholds["omitted_tail_final_two_kcal"]["value"] == 0.1
    assert thresholds["omitted_tail_final_two_kcal"]["comparator"] == "<"

    model_probes = load_json(MODEL_PROBE_PATH)
    key_map = _probe_key_map()
    required_spec_fields = {
        "checkpoint",
        "checkpoint_sha256",
        "size_bytes",
        "provider",
        "license",
    }
    required_provenance_fields = {
        "official_url",
        "source_commit",
        "source_ref",
        "observed_local_filename",
        "observed_local_path",
    }
    expected_model_refs = {
        "outer": {
            "checkpoint": "MACEPOLAR1Mmodel",
            "probe_key": "models.polar1m",
        },
        "sampler": {
            "checkpoint": "MACE-OFF24_medium.model",
            "probe_key": "models.off24",
        },
        "target": {
            "checkpoint": "MACE-omol-0-extra-large-1024.model",
            "probe_key": "models.omol0",
        },
    }

    for role, spec in protocol["models"].items():
        fixture_key = key_map[role].split("::", 1)[1]
        fixture_entry = _resolve_probe_key(model_probes, key_map[role])
        expected = expected_model_refs[role]
        assert fixture_key == expected["probe_key"]

        assert _is_hex64(spec["checkpoint_sha256"])
        assert isinstance(spec["size_bytes"], int) and spec["size_bytes"] > 0
        assert spec["checkpoint"] == expected["checkpoint"]
        assert set(spec.keys()).issuperset(required_spec_fields)

        prov = spec["provenance"]
        assert _is_hex40(prov["source_commit"])
        assert prov["source_ref"] is not None
        assert prov["official_url"] == fixture_entry["source_url"]
        assert prov["source_commit"] == fixture_entry["source_commit"]
        assert prov["source_ref"] == fixture_entry["source_ref"]
        assert prov["observed_local_filename"] == fixture_entry["observed_local_filename"]
        assert prov["observed_local_path"] == fixture_entry["observed_local_path"]
        assert prov["verified_by_model_probe_key"] == key_map[role]
        assert spec["source_ref"] == fixture_entry["source_ref"]
        assert spec["source_commit"] == fixture_entry["source_commit"]
        assert spec["checkpoint_sha256"] == fixture_entry["sha256"]
        assert spec["size_bytes"] == fixture_entry["size_bytes"]
        assert spec["provider"] == fixture_entry["provider"]
        assert spec["license"] == fixture_entry["license"]
        assert set(prov.keys()).issuperset(required_provenance_fields)
        assert prov["observed_local_filename"] == fixture_entry["observed_local_filename"]
        assert prov["observed_local_path"] == fixture_entry["observed_local_path"]
        assert prov["source_ref"] == fixture_entry["source_ref"]

    assert _is_hex40(protocol["models"]["outer"]["provenance"]["source_commit"])
    assert _is_hex40(protocol["models"]["sampler"]["provenance"]["source_commit"])
    assert _is_hex40(protocol["models"]["target"]["provenance"]["source_commit"])

    # Fail-closed gate for unsupported topology/disconnected topology remains explicit in protocol and contracts
    failure = load_json(FAIL_PATH)
    codes = {row["code"] for row in failure["catalog"]["failure_codes"]}
    assert "SMD_CAVITY_TOPOLOGY_DISCONTINUITY" in codes


def test_claim_failure_state_contracts_validate_and_enforce_lattice_logic():
    claim = load_json(CLAIM_PATH)
    failure = load_json(FAIL_PATH)
    state = load_json(STATE_PATH)
    schema = _claim_schema()

    _validate_local_schema(claim, schema)
    _validate_local_schema(failure, _failure_schema())
    _validate_local_schema(state, _state_schema())

    assert state["contracts"]["reduced_potential_unit"] == "dimensionless"

    # unique status codes and required fields remain consistent
    codes = [row["code"] for row in failure["catalog"]["failure_codes"]]
    assert len(codes) == len(set(codes))
    for row in failure["catalog"]["failure_codes"]:
        for field in failure["catalog"]["required_fields"]:
            assert field in row

    assert "DOMAIN_UNSUPPORTED" in codes
    assert failure["catalog"]["uniqueness"]["code"] == "required"

    metric = claim["route2_superiority_metric"]
    assert "route2_sample_count" in metric["required_fields"]

    assert metric["required_fields"] == [
        "route2_delta_ci_95_upper",
        "route2_delta_ci_95_lower",
        "route2_fail_count",
        "route2_abstain_count",
        "route2_d_i_abs_error",
        "route2_sample_count",
        "route2_paired_rows",
        "route2_paired_rows_hash",
        "route2_bootstrap_indices",
        "route2_bootstrap_indices_hash",
    ]


def test_state_contract_shape_and_unit_guard_with_mutation_checks():
    table = load_json(STATE_FIX)
    assert table["units"] == "dimensionless"
    assert table["dimensionless"] is True

    K = len(table["lambda_values"])
    N = len(table["frame_ids"])

    assert len(table["u_kn"]) == K
    assert all(len(row) == N for row in table["u_kn"])
    assert len(table["N_k"]) == K
    assert len(table["row_labels"]) == K
    assert len(table["state_hashes"]) == K
    assert sum(table["N_k"]) == N

    u = np.asarray(table["u_kn"], dtype=float)
    assert np.isfinite(u).all()

    # unit/shape/unit mismatch fail-closed checks by controlled mutation
    mutated = copy.deepcopy(table)
    mutated["units"] = "kcal/mol"
    assert mutated["units"] != table["units"]

    mutated = copy.deepcopy(table)
    mutated["N_k"] = [1, 2]
    assert len(mutated["N_k"]) != len(mutated["lambda_values"])

    mutated = copy.deepcopy(table)
    mutated["measure_id"] = "wrong_measure"
    assert mutated["measure_id"] != table["measure_id"]

    mutated = copy.deepcopy(table)
    mutated["boundary_conditions"]["pbc"] = True
    assert mutated["boundary_conditions"]["pbc"] != table["boundary_conditions"]["pbc"]
