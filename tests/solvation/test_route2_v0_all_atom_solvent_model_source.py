from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.route2_v0_all_atom_solvent_model_source import (
    V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_CONSTRUCTION,
    V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS,
    Route2V0AllAtomSolventModelSource,
    load_route2_v0_all_atom_solvent_model_source,
    parse_route2_v0_all_atom_solvent_model_source,
)
from maple.function.calculator.extra_correction.implicit.route2_v0_rism_molecular_source import (
    AMBER_ELECTROSTATIC_CHARGE_SCALE,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "docs/implicit-solvation/benchmarks/route2-v0-solvent-model-sources"
SCM_DOCUMENT_SHA256 = "da85b31eb884865c9e7a5af1a172df304b7e264ef016045461de0b5029fc1470"


@pytest.mark.parametrize(
    ("filename", "solvent_id", "atom_count", "mdl_sha256", "multiplicities"),
    [
        (
            "dichloromethane-scm-adf-3drism-v1.json",
            "dichloromethane",
            5,
            "efacc15b6d9fed5ba28be58916f66e76b1087149ffe7b2e261f98f4a913343ed",
            (1, 2, 2),
        ),
        (
            "chloroform-scm-adf-3drism-v1.json",
            "chloroform",
            5,
            "166d8323369bf456561cbd6403dcc16c5f625a90acb7e2bdc384b608ace4d389",
            (1, 1, 3),
        ),
    ],
)
def test_checked_in_all_atom_model_sources_are_exact_source_only_records(
    filename: str,
    solvent_id: str,
    atom_count: int,
    mdl_sha256: str,
    multiplicities: tuple[int, ...],
):
    source = load_route2_v0_all_atom_solvent_model_source(SOURCES / filename)

    assert isinstance(source, Route2V0AllAtomSolventModelSource)
    assert source.construction == V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_CONSTRUCTION
    assert source.status == V0_ALL_ATOM_SOLVENT_MODEL_SOURCE_STATUS
    assert source.solvent_id == solvent_id
    assert source.atom_count == atom_count
    assert source.document_sha256 == SCM_DOCUMENT_SHA256
    assert source.source_locator.startswith("SCM ADF 2026.1 3D-RISM documentation")
    assert source.total_charge_e == pytest.approx(0.0, abs=1.0e-15)
    assert tuple(site.multiplicity for site in source.site_types) == multiplicities
    assert all(site.mass_amu > 0.0 for site in source.site_types)
    assert all(site.sigma_angstrom > 0.0 for site in source.site_types)
    assert all(site.epsilon_kcal_per_mol > 0.0 for site in source.site_types)
    assert "not a liquid-state asset" in source.claim_boundary
    assert any(
        "not admit a physical liquid endpoint" in item for item in source.not_claimed
    )

    mdl_text = source.to_amber_mdl_text()
    assert source.amber_mdl_sha256() == mdl_sha256
    assert hashlib.sha256(mdl_text.encode("utf-8")).hexdigest() == mdl_sha256
    assert (SOURCES / f"{solvent_id}-scm-adf-3drism-v1.mdl").read_text(
        encoding="utf-8"
    ) == mdl_text
    assert "%FLAG ATMTYP" in mdl_text
    assert "%FLAG MULTI" in mdl_text
    assert "CH2" not in mdl_text
    assert "CHCl3" not in mdl_text


def test_mdl_serialization_preserves_source_charge_and_lj_conventions():
    source = load_route2_v0_all_atom_solvent_model_source(
        SOURCES / "dichloromethane-scm-adf-3drism-v1.json"
    )

    assert [site.label for site in source.site_types] == ["C", "H", "Cl"]
    assert [site.atomic_number for site in source.site_types] == [6, 1, 17]
    assert source.site_types[
        0
    ].charge_e * AMBER_ELECTROSTATIC_CHARGE_SCALE == pytest.approx(-6.6329172)
    assert source.site_types[1].rmin_half_angstrom == pytest.approx(1.28690274)
    assert source.site_types[2].rmin_half_angstrom == pytest.approx(2.00022737)

    coordinates = np.concatenate(
        [site.positions_angstrom for site in source.site_types],
        axis=0,
    )
    np.testing.assert_allclose(
        coordinates[1],
        np.array([-0.926635, 0.0, -1.406798]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        coordinates[4],
        np.array([0.0, 1.451416, 0.177892]),
        rtol=0.0,
        atol=0.0,
    )


def _payload(filename: str = "dichloromethane-scm-adf-3drism-v1.json") -> dict:
    return json.loads((SOURCES / filename).read_text(encoding="utf-8"))


def test_source_record_rejects_training_or_target_label_provenance():
    payload = _payload()
    payload["no_target_policy"]["fine_tuning"] = True

    with pytest.raises(ValueError, match="policy flags must be false"):
        parse_route2_v0_all_atom_solvent_model_source(json.dumps(payload))


def test_source_record_rejects_united_atom_mass_and_coincident_sites():
    united = _payload()
    united["site_types"][0]["mass_amu"] = 14.027
    with pytest.raises(ValueError, match="united-atom and virtual-site"):
        parse_route2_v0_all_atom_solvent_model_source(json.dumps(united))

    coincident = _payload()
    coincident["site_types"][1]["positions_angstrom"][1] = [
        0.0,
        0.0,
        -0.814053,
    ]
    with pytest.raises(ValueError, match="coincident sites"):
        parse_route2_v0_all_atom_solvent_model_source(json.dumps(coincident))

    mislabeled = _payload()
    mislabeled["site_types"][2]["label"] = "C"
    with pytest.raises(ValueError, match="element symbol 'Cl'"):
        parse_route2_v0_all_atom_solvent_model_source(json.dumps(mislabeled))


def test_source_record_requires_strict_schema_and_content_addressed_document():
    payload = _payload()
    payload["source"]["document_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="SHA-256"):
        parse_route2_v0_all_atom_solvent_model_source(json.dumps(payload))

    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="keys differ"):
        parse_route2_v0_all_atom_solvent_model_source(json.dumps(payload))
