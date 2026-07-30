from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from route2_v0_asset_fixture import (
    refresh_route2_v0_test_source_bindings,
    sha256_file,
    write_route2_v0_test_manifest,
)

from maple.function.calculator.extra_correction.implicit.route2_v0_solvent_asset import (
    V0_DEFAULT_SOLVENT_IDS,
    load_route2_v0_frozen_solvent_registry,
)


def test_frozen_solvent_asset_binds_hashed_bulk_and_liquid_provenance(tmp_path):
    manifest, _ = write_route2_v0_test_manifest(tmp_path)

    registry = load_route2_v0_frozen_solvent_registry(manifest)
    asset = registry.asset_for("water")

    assert asset.model_identifier == "cSPCE-test-control"
    assert asset.bulk_direct_correlation.metadata.site_names == ("O", "H1")
    assert asset.bulk_direct_correlation.metadata.temperature_kelvin == pytest.approx(
        298.0
    )
    assert asset.file_for("short_range_interaction").path.name == "cSPCE-source.json"
    assert asset.molecular_reference.rism_site_type_names == ("O", "H1", "H1")
    np.testing.assert_array_equal(
        asset.molecular_reference.molecular_reference.atomic_numbers,
        np.array([8, 1, 1]),
    )
    assert asset.molecular_reference.molecular_reference.provenance_label == (
        "frozen-site-model:" + asset.file_for("site_model").sha256
    )
    assert set(asset.excluded_target_label_sets) == {
        "mnsol",
        "freesolv",
        "development",
        "confirmation",
        "blind",
    }
    registry.verify_integrity()

    with pytest.raises(ValueError, match="missing default assets"):
        registry.require_default_solvent_panel()
    assert len(V0_DEFAULT_SOLVENT_IDS) == 11


def test_frozen_solvent_asset_rejects_mutation_and_target_label_use(tmp_path):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    registry = load_route2_v0_frozen_solvent_registry(manifest)

    (tmp_path / "model/cSPCE.mdl").write_text("changed model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.verify_integrity()
    with pytest.raises(ValueError, match="hash mismatch"):
        load_route2_v0_frozen_solvent_registry(manifest)

    payload["assets"][0]["provenance"]["target_solvation_labels_used"] = True
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly false"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_missing_source_or_path_escape(tmp_path):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    del payload["assets"][0]["source_files"]["short_range_interaction"]
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="short_range_interaction"):
        load_route2_v0_frozen_solvent_registry(manifest)

    manifest, payload = write_route2_v0_test_manifest(tmp_path / "separate")
    payload["assets"][0]["source_files"]["site_model"]["path"] = "../escape.mdl"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="relative path inside the asset root"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_unbound_or_nonphysical_molecular_reference(
    tmp_path,
):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    del payload["assets"][0]["molecular_reference"]
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="molecular reference"):
        load_route2_v0_frozen_solvent_registry(manifest)

    manifest, payload = write_route2_v0_test_manifest(tmp_path / "wrong-site-model")
    payload["assets"][0]["molecular_reference"]["site_model_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="site-model SHA-256"):
        load_route2_v0_frozen_solvent_registry(manifest)

    manifest, payload = write_route2_v0_test_manifest(tmp_path / "wrong-multiplicity")
    payload["assets"][0]["molecular_reference"]["rism_site_type_names"] = [
        "O",
        "O",
        "H1",
    ]
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="site map"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_precanonical_schema_v1_manifest(tmp_path):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    payload["schema_version"] = 1
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="registry schema"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_semantically_inconsistent_source_files(
    tmp_path,
):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    input_path = tmp_path / "bulk/cSPCE.inp"
    input_path.write_text(
        input_path.read_text(encoding="utf-8").replace(
            "CLOSURE='PSE3'", "CLOSURE='KH'"
        ),
        encoding="utf-8",
    )
    refresh_route2_v0_test_source_bindings(tmp_path, payload)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="closure"):
        load_route2_v0_frozen_solvent_registry(manifest)

    manifest, payload = write_route2_v0_test_manifest(tmp_path / "failed-self-test")
    thermo_path = tmp_path / "failed-self-test/bulk/cSPCE.thermo"
    thermo_path.write_text(
        thermo_path.read_text(encoding="utf-8").replace(
            "1.0000000000000000E-12", "1.0E-02"
        ),
        encoding="utf-8",
    )
    refresh_route2_v0_test_source_bindings(
        tmp_path / "failed-self-test",
        payload,
    )
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="SM free-energy identity"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_binds_generation_iterations_to_maxstep(tmp_path):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    input_path = tmp_path / "bulk/cSPCE.inp"
    input_path.write_text(
        input_path.read_text(encoding="utf-8").replace("MAXSTEP=100", "MAXSTEP=1"),
        encoding="utf-8",
    )
    provenance_path = tmp_path / "provenance/cSPCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for run in provenance["runs"]:
        transcript = run["transcript"].replace(
            "step=   1     Res=  5.0000000000000000E-13     MDIIS=  1",
            "step=   1     Res=  5.0000000000000000E-06     MDIIS=  1\n"
            "step=   2     Res=  5.0000000000000000E-13     MDIIS=  1",
        )
        run["transcript"] = transcript
        run["transcript_sha256"] = hashlib.sha256(
            transcript.encode("utf-8")
        ).hexdigest()
    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    refresh_route2_v0_test_source_bindings(tmp_path, payload)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="iterations exceed rism1d MAXSTEP"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_manifest_transcription_that_differs_from_mdl(
    tmp_path,
):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    payload["assets"][0]["molecular_reference"]["site_charges_e"] = [
        -0.7,
        0.35,
        0.35,
    ]
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="site charges must equal"):
        load_route2_v0_frozen_solvent_registry(manifest)


def test_frozen_solvent_asset_rejects_short_range_certificate_hash_drift(tmp_path):
    manifest, payload = write_route2_v0_test_manifest(tmp_path)
    provenance_path = tmp_path / "provenance/cSPCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["claim_boundary"] = "Changed but still valid and label-free."
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    payload["assets"][0]["source_files"]["provenance_statement"]["sha256"] = (
        sha256_file(provenance_path)
    )
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="exact provenance_statement source hash"):
        load_route2_v0_frozen_solvent_registry(manifest)
