from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/implicit-solvation/benchmarks/prepare_cha_continuum_water.py"
SOURCE_ROOT = (
    Path("/home/axie/MAPLE/MAPLE-implicitsolv-route1")
    / ".omx/benchmarks/route1-foundation-20260913/inputs-precision-v2/water"
)


def _module():
    spec = importlib.util.spec_from_file_location("prepare_cha_continuum_water", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_bundle_rejects_a_tampered_external_hash(tmp_path):
    prep = _module()
    source_paths = {
        name: SOURCE_ROOT / relative
        for name, relative in {
            "fixed_mol2": "fixed.mol2",
            "retyped_mol2": "charge-audit/charges-am1bcc.mol2",
            "mapping": "charge-audit/charges-am1bcc.mapping.json",
            "normalization": "charge-audit/charges-am1bcc.normalization.json",
        }.items()
    }
    pins = {name: _file_sha(path) for name, path in source_paths.items()}
    copied = tmp_path / "fixed.mol2"
    copied.write_bytes(source_paths["fixed_mol2"].read_bytes() + b"\n")
    source_paths["fixed_mol2"] = copied

    with pytest.raises(ValueError, match="fixed_mol2.*SHA256"):
        prep.load_source_bundle(source_paths, pins)


def test_type_merge_preserves_fixed_identity_coordinates_bonds_and_charges():
    prep = _module()
    fixed = prep.parse_mol2((SOURCE_ROOT / "fixed.mol2").read_text())
    retyped = prep.parse_mol2(
        (SOURCE_ROOT / "charge-audit/charges-am1bcc.mol2").read_text()
    )
    mapping = json.loads(
        (SOURCE_ROOT / "charge-audit/charges-am1bcc.mapping.json").read_text()
    )
    normalization = json.loads(
        (SOURCE_ROOT / "charge-audit/charges-am1bcc.normalization.json").read_text()
    )

    rendered, receipt = prep.merge_verified_gaff2_types(
        fixed, retyped, mapping, normalization
    )
    merged = prep.parse_mol2(rendered)

    assert [atom["id"] for atom in merged["atoms"]] == [1, 2, 3]
    assert [atom["name"] for atom in merged["atoms"]] == ["O1", "H2", "H3"]
    assert [atom["coordinates_angstrom"] for atom in merged["atoms"]] == [
        atom["coordinates_angstrom"] for atom in fixed["atoms"]
    ]
    assert [atom["atom_type"] for atom in merged["atoms"]] == ["oh", "ho", "ho"]
    assert [atom["charge_e"] for atom in merged["atoms"]] == pytest.approx(
        [-0.784666666667, 0.392333333333, 0.392333333333], abs=0.0
    )
    assert merged["bonds"] == fixed["bonds"]
    assert receipt["provider_charge_tokens_used"] is False
    assert receipt["mapping_verified"] is True
    assert receipt["normalization_verified"] is True


@pytest.mark.parametrize(
    "mutation", ["name", "order", "coordinate", "charge", "mapping"]
)
def test_type_merge_fails_closed_when_receipted_identity_changes(mutation):
    prep = _module()
    fixed = prep.parse_mol2((SOURCE_ROOT / "fixed.mol2").read_text())
    retyped = prep.parse_mol2(
        (SOURCE_ROOT / "charge-audit/charges-am1bcc.mol2").read_text()
    )
    mapping = json.loads(
        (SOURCE_ROOT / "charge-audit/charges-am1bcc.mapping.json").read_text()
    )
    normalization = json.loads(
        (SOURCE_ROOT / "charge-audit/charges-am1bcc.normalization.json").read_text()
    )
    fixed = copy.deepcopy(fixed)
    mapping = copy.deepcopy(mapping)
    if mutation == "name":
        fixed["atoms"][0]["name"] = "OX"
    elif mutation == "order":
        fixed["atoms"][0], fixed["atoms"][1] = fixed["atoms"][1], fixed["atoms"][0]
    elif mutation == "coordinate":
        fixed["atoms"][0]["coordinates_angstrom"][0] += 0.01
    elif mutation == "charge":
        fixed["atoms"][0]["charge_e"] += 0.01
    else:
        mapping[0]["provider_atom_name"] = "OX"
    with pytest.raises(ValueError):
        prep.merge_verified_gaff2_types(fixed, retyped, mapping, normalization)


def test_prmtop_charge_serialization_and_topology_loader_round_trip():
    prep = _module()
    source = [-0.784666666667, 0.392333333333, 0.392333333333]
    charge_tokens = [f"{charge * prep.AMBER_CHARGE_SCALE:16.8E}" for charge in source]
    prmtop = (
        "%FLAG CHARGE                                                                    \n"
        "%FORMAT(5E16.8)                                                                 \n"
        + "".join(charge_tokens)
        + "\n"
    )
    effective, raw_tokens = prep.parse_prmtop_charge_field(prmtop, atom_count=3)
    artifact = prep.build_topology_artifact(
        atom_ids=[1, 2, 3],
        atom_names=["O1", "H2", "H3"],
        elements=["O", "H", "H"],
        atomic_numbers=[8, 1, 1],
        gaff2_types=["oh", "ho", "ho"],
        bonds=[[0, 1, "1"], [0, 2, "1"]],
        source_charges=source,
        effective_charges=effective,
        source_mol2_sha256="a" * 64,
        prepared_prmtop_sha256="b" * 64,
        parameter_source_sha256="c" * 64,
        bondi_radii=[1.5, 1.2, 1.2],
        cha_radii=[1.88, 1.04, 1.04],
        lj_rmin=[1.82, 0.3019, 0.3019],
        lj_epsilon=[0.093, 0.0047, 0.0047],
    )

    assert raw_tokens == charge_tokens
    assert max(abs(a - b) for a, b in zip(source, effective)) < 1.0e-9
    topology = ContinuumChaTopology.from_mapping(
        artifact,
        expected_content_sha256=artifact["content_sha256"],
        expected_source_charge_sha256=artifact["source_charges_sha256"],
        expected_source_mol2_sha256="a" * 64,
    )
    assert topology.gaff2_types == ("oh", "ho", "ho")
    assert topology.cha_radii_angstrom == (1.88, 1.04, 1.04)
    assert topology.effective_charges_e == pytest.approx(source, abs=1.0e-9)
