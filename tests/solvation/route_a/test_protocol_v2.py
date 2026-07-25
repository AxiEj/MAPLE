from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from maple.function.dispatcher.solvfe.protocol import (
    ProtocolIntegrityError,
    RouteAProtocol,
)

from .conftest import DOCS_DIR, PROJECT_ROOT


PROTOCOL_PATH = DOCS_DIR / "protocol-v2.json"
SCHEMA_PATH = DOCS_DIR / "protocol-schema-v2.json"
THERMODYNAMICS_PATH = DOCS_DIR / "thermodynamics-v2.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_v2_is_self_consistent_and_artifact_bound():
    protocol = RouteAProtocol.load(PROTOCOL_PATH, project_root=PROJECT_ROOT)

    assert protocol.data["protocol_version"] == "2.0.0"
    assert protocol.data["status"] == (
        "research-protocol-partially-implemented"
    )
    assert protocol.data["supersedes"]["protocol_version"] == "1.4.0"
    assert protocol.data["supersedes"]["scientific_reason"] == (
        "V1 used ordinary bare-solute Route2 outer solvation as G_0 and omitted "
        "the QCT empty-volume packing/conditioning term; its multi-n result is "
        "not eligible for scientific promotion."
    )
    assert protocol.verified_artifact_count >= 1


def test_protocol_v2_freezes_qct_conditioning_and_supermolecule_cavity():
    protocol = _load(PROTOCOL_PATH)
    qct = protocol["qct_conditioning"]
    cavity = protocol["supermolecule_outer_cavity"]
    thresholds = protocol["thresholds"]

    assert qct["identity"] == "cluster-qct-multistate"
    assert qct["packing_probability"]["symbol"] == "p_0(lambda_s)"
    assert qct["packing_probability"]["system"] == "pure_bulk_water"
    assert qct["packing_probability"]["required"] is True
    assert qct["n0_state"]["ordinary_bare_route2_outer_allowed"] is False
    assert qct["n0_state"]["conditioned_empty_shell_required"] is True
    assert qct["formulas"]["hydration_free_energy"] == (
        "mu_X_ex = -RT ln p_0(lambda_s) "
        "- RT ln sum_n exp[-beta A_n_tilde(lambda_s)]"
    )

    assert cavity["whole_supermolecule_inside"] is True
    assert cavity["construction_rule_same_for_all_n"] is True
    assert cavity["per_frame_gepol"] is True
    assert cavity["mesh_reused_across_frames"] is False
    assert cavity["base_radii_A"]["H"] == 1.2
    assert cavity["base_radii_A"]["C"] == 1.85
    assert cavity["base_radii_A"]["O"] == 1.52
    assert cavity["carbonyl_oxygen_override"] == {
        "atom_type": "o",
        "element": "O",
        "radius_A": 1.7,
    }
    assert cavity["radius_scale"] == 1.2
    assert cavity["tessera_area_A2"] == 0.2
    assert cavity["minimum_added_sphere_radius_A"] == 1.0
    assert cavity["full_ensemble_required"] is True
    assert cavity["frame_deletion_allowed"] is False
    assert cavity["native_and_pedra_warnings_fatal"] is True
    assert cavity["minimum_bridge_overlap_A"] > 0.0
    assert cavity["cds_policy"] == "excluded-from-v2-core-mandatory-ablation"
    assert thresholds["per_state_se_kcal_max"]["value"] == 0.2
    assert thresholds["packing_empty_samples_max_bias_min"]["value"] == 100
    assert thresholds["packing_se_kcal_max"]["value"] == 0.2
    assert thresholds["replica_min_count"]["value"] == 3
    assert thresholds["replica_pairwise_z_max"]["value"] == 2.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.pop("qct_conditioning"),
        lambda value: value["qct_conditioning"].pop("packing_probability"),
        lambda value: value["qct_conditioning"]["n0_state"].__setitem__(
            "ordinary_bare_route2_outer_allowed", True
        ),
        lambda value: value.pop("supermolecule_outer_cavity"),
        lambda value: value["supermolecule_outer_cavity"].__setitem__(
            "frame_deletion_allowed", True
        ),
        lambda value: value["supermolecule_outer_cavity"].__setitem__(
            "construction_rule_same_for_all_n", False
        ),
        lambda value: value["thresholds"]["replica_pairwise_z_max"].__setitem__(
            "value", 3.0
        ),
        lambda value: value["thresholds"].pop("per_state_se_kcal_max"),
    ],
)
def test_protocol_v2_schema_rejects_missing_or_weakened_scientific_contract(
    mutator,
):
    protocol = _load(PROTOCOL_PATH)
    schema = _load(SCHEMA_PATH)
    mutated = copy.deepcopy(protocol)
    mutator(mutated)

    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(mutated)


def test_protocol_v2_thermodynamics_rejects_v1_bare_g0_shortcut():
    text = THERMODYNAMICS_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "ordinary bare-solute Route 2 value is not the `n=0` QCT state" in text
    assert "p_0(lambda_s)" in text
    assert "same preregistered construction rule" in normalized
    assert "one whole-supermolecule cavity per frame" in normalized
    assert "Deleting failed frames is forbidden" in text
    assert "The `n=0` entry is" in text
    assert "`A_0_tilde = DeltaG_LR_super(X | n=0,lambda_s)`" in text
    assert text.count("G_0 = ΔG_outer(X)") == 1


def test_protocol_loader_rejects_v2_hash_drift(tmp_path: Path):
    protocol = _load(PROTOCOL_PATH)
    protocol["supermolecule_outer_cavity"]["frame_deletion_allowed"] = True
    drifted = tmp_path / "protocol-v2.json"
    drifted.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ProtocolIntegrityError, match="self hash mismatch"):
        RouteAProtocol.load(drifted, project_root=PROJECT_ROOT)
