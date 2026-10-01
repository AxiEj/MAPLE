"""Label-free panel diagnostic for missing multi-probe SES clipping.

The optional frozen topology assets are not distributed with a clean clone.
This test characterizes the present local geometry, not a valid R6 endpoint.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_ses_geometry import (
    PROBE_ANGSTROM,
    triple_probe_centers,
)

_INPUTS = (
    Path(__file__).resolve().parents[2]
    / ".omx/benchmarks/route1-torch-full-cha/panel30/continuum_inputs"
)
_PINNED_MANIFEST_SHA256 = (
    "ac3c54b32e18868fb7194c996f5cef2de94667197ab9ebf005e27967a7836b84"
)
_PINNED_MANIFEST_FILE_SHA256 = (
    "344711992c25d89f8413306208ce0a1dcaf9910f02042aa98200364361a9ae9b"
)


def test_local_probe_center_overlap_is_common_on_frozen_panel():
    manifest_path = _INPUTS / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("optional frozen panel30 input assets are not installed")
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == _PINNED_MANIFEST_FILE_SHA256
    manifest = json.loads(manifest_bytes)
    assert manifest["content_sha256"] == _PINNED_MANIFEST_SHA256

    exposed_count = 0
    overlap_pairs = 0
    affected_molecules = 0
    for record in manifest["records"]:
        case_dir = _INPUTS / record["compound_id"]
        topology_path = case_dir / "topology.json"
        coordinates_path = case_dir / "coordinates.json"
        if not topology_path.is_file() or not coordinates_path.is_file():
            pytest.fail(f"incomplete frozen panel case: {record['compound_id']}")
        topology_bytes = topology_path.read_bytes()
        coordinates_bytes = coordinates_path.read_bytes()
        assert (
            hashlib.sha256(topology_bytes).hexdigest() == record["topology_file_sha256"]
        )
        assert (
            hashlib.sha256(coordinates_bytes).hexdigest()
            == record["coordinate_file_sha256"]
        )
        topology = json.loads(topology_bytes)
        coordinates = json.loads(coordinates_bytes)
        positions = torch.tensor(coordinates["positions_angstrom"], dtype=torch.float64)
        radii = torch.tensor(topology["cha_radii_angstrom"], dtype=torch.float64)
        assert len(positions) == record["atom_count"]
        expanded = radii + PROBE_ANGSTROM
        exposed = []
        for triple in combinations(range(len(positions)), 3):
            if any(
                torch.linalg.vector_norm(positions[first] - positions[second])
                >= expanded[first] + expanded[second]
                for first, second in combinations(triple, 2)
            ):
                continue
            try:
                geometry = triple_probe_centers(positions, radii, triple)
            except ValueError as exc:
                if "collinear" not in str(exc):
                    raise
                continue
            exposed.extend(
                center
                for center, locally_exposed in zip(
                    geometry.centers, geometry.locally_exposed
                )
                if bool(locally_exposed)
            )
        exposed_count += len(exposed)
        if len(exposed) > 1:
            close_pairs = int(
                (torch.pdist(torch.stack(exposed)) < 2.0 * PROBE_ANGSTROM).sum()
            )
            overlap_pairs += close_pairs
            affected_molecules += close_pairs > 0

    assert len(manifest["records"]) == 30
    assert (exposed_count, overlap_pairs, affected_molecules) == (320, 249, 18)
