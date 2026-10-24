"""Source-bound mbondi2/profile tests that never need an OpenMM installation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    mol2_identity_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SOURCE = (
    REPOSITORY_ROOT
    / ".omx/vendor-audits/gbmv2/openmm-src/wrappers/python/openmm/app/internal/customgbforces.py"
)
EXPECTED_UPSTREAM_SHA256 = (
    "f96a49a9e9ef0528e937481b1c1331b43ad03e6ec3018b478c642b301f07391a"
)


def _atoms_with_mol2(symbols, bonds):
    count = len(symbols)
    metadata = {
        "atom_ids": list(range(1, count + 1)),
        "atom_names": list(symbols),
        "atom_types": list(symbols),
        "subst_ids": [1] * count,
        "subst_names": ["MOL"] * count,
        "bonds": bonds,
        "component_ids": [0] * count,
        "component_count": 1,
    }
    metadata[MOL2_IDENTITY_SHA256_KEY] = mol2_identity_sha256(metadata)
    atoms = Atoms(symbols, positions=np.zeros((count, 3), dtype=np.float64))
    atoms.new_array(MOL2_ATOM_ID_ARRAY, np.arange(1, count + 1, dtype=np.int64))
    atoms.info["mol2"] = metadata
    return atoms


def test_packaged_manifest_is_neutral_source_bound_and_self_verifying():
    from maple.function.calculator.extra_correction.implicit.obc2_profile import (
        OBC2_FORMULA_MANIFEST_ID,
        load_obc2_profile_manifest,
        verify_obc2_profile_manifest,
    )

    manifest = load_obc2_profile_manifest()
    verified = verify_obc2_profile_manifest()

    assert OBC2_FORMULA_MANIFEST_ID == "openmm-8.5.2-obc2-mbondi2-v1"
    assert manifest["manifest_id"] == OBC2_FORMULA_MANIFEST_ID
    assert "ace" not in manifest["manifest_id"].lower()
    assert manifest["upstream"]["version"] == "8.5.2"
    assert manifest["upstream"]["source_sha256"] == EXPECTED_UPSTREAM_SHA256
    assert verified["manifest_id"] == OBC2_FORMULA_MANIFEST_ID
    assert verified["canonical_payload_sha256"] == manifest[
        "canonical_payload_sha256"
    ]
    assert verified["license_sha256"] == manifest["license"]["sha256"]


def test_packaged_license_preserves_required_openmm_notice():
    from maple.function.calculator.extra_correction.implicit.obc2_profile import (
        load_obc2_profile_license,
        load_obc2_profile_manifest,
    )

    license_text = load_obc2_profile_license()
    manifest = load_obc2_profile_manifest()

    assert "University of Virginia and the Authors" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
    assert hashlib.sha256(license_text.encode("utf-8")).hexdigest() == manifest[
        "license"
    ]["sha256"]


def test_frozen_upstream_extractor_matches_packaged_canonical_payload():
    from maple.function.calculator.extra_correction.implicit.obc2_profile import (
        extract_openmm_obc2_profile,
        load_obc2_profile_manifest,
    )

    assert hashlib.sha256(UPSTREAM_SOURCE.read_bytes()).hexdigest() == (
        EXPECTED_UPSTREAM_SHA256
    )
    extracted = extract_openmm_obc2_profile(UPSTREAM_SOURCE)
    manifest = load_obc2_profile_manifest()

    assert extracted["canonical_payload"] == manifest["canonical_payload"]
    assert extracted["source_sha256"] == EXPECTED_UPSTREAM_SHA256
    assert extracted["source_ranges"] == manifest["upstream"]["source_ranges"]


def test_manifest_verifier_fails_closed_on_payload_or_license_drift():
    from maple.function.calculator.extra_correction.implicit.obc2_profile import (
        load_obc2_profile_license,
        load_obc2_profile_manifest,
        verify_obc2_profile_manifest,
    )

    manifest = copy.deepcopy(load_obc2_profile_manifest())
    manifest["canonical_payload"]["offset_nm"] = 0.01
    with pytest.raises(ValueError, match="payload|SHA|digest"):
        verify_obc2_profile_manifest(manifest=manifest)

    with pytest.raises(ValueError, match="license|SHA|digest"):
        verify_obc2_profile_manifest(
            license_text=load_obc2_profile_license() + "tampered\n"
        )


def test_independent_radius_rules_cover_every_upstream_branch():
    from maple.function.calculator.extra_correction.implicit.obc2_topology import (
        OBC2Topology,
    )
    from maple.function.calculator.extra_correction.implicit.radii import (
        IndependentMbondi2RadiusProvider,
    )
    # H bonded to N and H bonded to C exercise the only connectivity-dependent
    # mbondi2 branch. Br and I exercise the upstream default radius/screen branch.
    symbols = ["N", "H", "C", "H", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
    bonds = [[0, 1, "1"], [0, 2, "1"], [2, 3, "1"], [2, 4, "1"], [2, 5, "1"],
             [2, 6, "1"], [2, 7, "1"], [2, 8, "1"], [2, 9, "1"],
             [2, 10, "1"], [2, 11, "1"]]
    topology = OBC2Topology.from_atoms(_atoms_with_mol2(symbols, bonds))
    result = IndependentMbondi2RadiusProvider().assign(topology)

    expected_angstrom = np.array(
        [1.55, 1.3, 1.7, 1.2, 1.5, 1.5, 2.1, 1.85, 1.8, 1.7, 1.5, 1.5]
    )
    np.testing.assert_array_equal(
        result.provider_parameters[:, 0], expected_angstrom / 10.0
    )
    np.testing.assert_array_equal(
        result.radii_angstrom, result.provider_parameters[:, 0] * 10.0
    )
    np.testing.assert_array_equal(
        result.provider_parameters[:, 1],
        np.array([0.79, 0.85, 0.72, 0.85, 0.85, 0.88, 0.8, 0.86, 0.96, 0.8, 0.8, 0.8]),
    )
    np.testing.assert_allclose(
        result.provider_parameters[:, 0],
        result.radii_angstrom / 10.0,
        rtol=0.0,
        atol=0.0,
    )
    assert result.profile == "obc2-mbondi2"
    assert result.provenance["provider"] == "maple-independent-mbondi2"
    assert result.provenance["formula_manifest_id"] == (
        "openmm-8.5.2-obc2-mbondi2-v1"
    )
    assert result.provider_parameters.flags.writeable is False
    assert result.radii_angstrom.flags.writeable is False


def test_manifest_json_is_canonicalizable_without_runtime_openmm_metadata():
    from maple.function.calculator.extra_correction.implicit.obc2_profile import (
        load_obc2_profile_manifest,
    )

    canonical = json.dumps(
        load_obc2_profile_manifest(), sort_keys=True, separators=(",", ":")
    )
    assert "openmm-8.5.2-obc2-mbondi2-v1" in canonical
    assert "source_path" not in canonical
