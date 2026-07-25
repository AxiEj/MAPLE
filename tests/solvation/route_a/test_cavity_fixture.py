from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from .conftest import FIXTURE_DIR, PROJECT_ROOT, load_json

CAVITY_FIX = FIXTURE_DIR / "cavities" / "pcmsolver-golden-water-dimer.json"
TOPOLOGY_FIX = (
    FIXTURE_DIR / "cavities" / "pcmsolver_topology_transition_v1.json"
)
PROTO_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "protocol-v1.json"
RADIUS_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "radius-profiles" / "bondi-mantina-v1.json"
OUTER_CONTRACT_PATH = PROJECT_ROOT / "docs" / "solvation" / "route-a" / "outer-adapter-contract-v1.json"

EXPECTED_OVERLAP_PAIRS = [(0, 3), (1, 3), (1, 5)]

def _load_xyz(path: Path) -> Tuple[np.ndarray, list[str]]:
    lines = path.read_text().strip().splitlines()
    count = int(lines[0])
    atom_lines = lines[2 : 2 + count]
    symbols = []
    coords = []
    for line in atom_lines:
        s, x, y, z = line.split()
        symbols.append(s)
        coords.append([float(x), float(y), float(z)])
    return np.array(coords, dtype=float), symbols


def _pair_overlap_penetration_max(
    centers: np.ndarray,
    atom_positions: np.ndarray,
    radii: np.ndarray,
    pairs,
    tolerance: float,
) -> tuple[float, tuple[int, int], np.ndarray]:
    max_pen = -np.inf
    worst_pair = (-1, -1)
    worst_center = np.zeros(3)

    for t in centers:
        d = np.linalg.norm(t[None, :] - atom_positions, axis=1)
        signed = d - radii
        for i, j in pairs:
            p = min(-signed[i], -signed[j])
            if p > max_pen:
                max_pen = p
                worst_pair = (i, j)
                worst_center = t.copy()

    return max_pen, worst_pair, worst_center


def _native_surface_component_count(
    sphere_centers: np.ndarray,
    sphere_radii: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> int:
    native_spheres = np.column_stack([sphere_centers, sphere_radii])
    _, unique_indices = np.unique(
        np.rint(native_spheres / tolerance).astype(np.int64),
        axis=0,
        return_index=True,
    )
    centers = sphere_centers[unique_indices]
    radii = sphere_radii[unique_indices]
    n = len(centers)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(centers[i] - centers[j])
            if distance <= radii[i] + radii[j] + tolerance:
                _union(i, j)

    roots = {_find(i) for i in range(n)}
    return len(roots)


def test_pcmsolver_fixture_records_water_dimer_provenance_and_hashes():
    data = load_json(CAVITY_FIX)

    required = {
        "fixture_version",
        "system_name",
        "protocol_reference",
        "status",
        "reason",
        "source_runs",
        "geometry",
        "input",
        "cavity",
        "source_provenance",
        "predetermined_mesh_required",
        "protection_predicate",
    }
    assert required.issubset(data)
    assert data["system_name"] == "water-dimer"
    assert data["status"] == "AVAILABLE"
    assert data["source_runs"] == ["minradius1-run1", "minradius1-run2"]

    geom = data["geometry"]
    geom_path = PROJECT_ROOT / geom["path"]
    assert geom_path.exists()
    geom_bytes = geom_path.read_bytes()
    assert geom["sha256"] == hashlib.sha256(geom_bytes).hexdigest()
    assert geom["atoms"] == 6

    raw = data["input"]["raw"]
    parsed = data["input"]["parsed"]
    assert raw["sha256"] == hashlib.sha256((PROJECT_ROOT / raw["path"]).read_bytes()).hexdigest()
    assert parsed["sha256"] == hashlib.sha256((PROJECT_ROOT / parsed["path"]).read_bytes()).hexdigest()

    archive = data["cavity"]["archive"]
    archive_path = PROJECT_ROOT / archive["path"]
    assert archive["sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()

    mesh = data["cavity"]["mesh"]
    mesh_path = PROJECT_ROOT / mesh["path"]
    mesh_npz = np.load(mesh_path)
    centers = mesh_npz["centers_angstrom"]
    areas = mesh_npz["areas_angstrom2"]
    assert centers.shape == (mesh["tesserae"], 3)
    assert areas.shape == (mesh["tesserae"],)
    assert mesh["tesserae"] == 378
    assert mesh["area_sum_A2"] == pytest.approx(64.76408764745233, rel=0, abs=1e-12)
    assert mesh["canonical_centers_areas_sha256"] == "b7d8c88afbb9b713607395ca612865e7e3a4f52a38b63ba9b18bab220d39cb22"
    mesh_sha = hashlib.sha256(np.column_stack([centers, areas]).astype("<f8", order="C").tobytes()).hexdigest()
    assert mesh_sha == mesh["canonical_centers_areas_sha256"]

    assert data["cavity"]["penetrations"] == 0
    assert data["cavity"]["warnings"] == []

    assert data["cavity"]["component_graph_required"] == 1
    assert data["cavity"]["component_graph_required"] == data["runtime"]["component_count"]
    native = np.load(archive_path)
    assert _native_surface_component_count(
        native["elSphCenter"].T,
        native["elRadius"].reshape(-1),
    ) == data["runtime"]["component_count"]

    prov = data["source_provenance"]
    assert prov["geometry_coordinates"]["units"] == "angstrom"
    assert prov["input_raw"]["units"] == "angstrom"
    assert prov["input_parsed"]["units"] == "angstrom"
    assert prov["pcmsolver_native"]["coordinates"] == "bohr"
    assert prov["pcmsolver_native"]["areas"] in {"bohr2", "angstrom2"}
    assert prov["exported_mesh"]["centers"] == "angstrom"
    assert prov["exported_mesh"]["areas"] == "angstrom2"
    audit_ref = prov["two_run_audit"]
    audit = load_json(PROJECT_ROOT / audit_ref["path"])
    assert audit_ref["hash_kind"] == "canonical_sha256"
    assert audit_ref["sha256"] == hashlib.sha256(
        json.dumps(
            audit,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert len(audit["runs"]) == 2
    assert all(
        run["raw_input"]["sha256"] == raw["sha256"]
        and run["parsed_input"]["sha256"] == parsed["sha256"]
        and run["native_cavity_archive"]["sha256"] == archive["sha256"]
        and run["exported_mesh"]["canonical_centers_areas_sha256"]
        == mesh["canonical_centers_areas_sha256"]
        and run["warning_log"]["warning_count"] == 0
        for run in audit["runs"]
    )
    assert all(audit["cross_run_checks"].values())

    runtime = data["runtime"]
    assert runtime["engine"] == "pcmsolver"
    assert runtime["library_version"] == "1.3.0+27fb35f"
    assert runtime["library_commit"] == "27fb35f"
    assert runtime["library_sha256"] == "296b6f34a03789943ae8823b3790f36357c50c896497f21374ac16fe5a6c43a3"
    assert runtime["library_path_source_prefix"] == "bbd992d54ebeace528cf236dede1f0c56641defb"
    assert "27fb35f" in runtime["library_version"]

    protocol = load_json(PROTO_PATH)
    outer = load_json(OUTER_CONTRACT_PATH)
    cavity_profile = outer["cavity_profile"]
    assert data["protocol_reference"] == (
        f"route-a-protocol-v{protocol['protocol_version']}"
    )
    assert raw["tessera_fallback_area_A2"] == (
        cavity_profile["area_angstrom2"]
    )
    assert raw["fallback_minradius_A"] == (
        cavity_profile["minimum_added_sphere_radius_angstrom"]
    )
    assert mesh["source_tolerances_A"] == {
        "area": cavity_profile["area_angstrom2"],
        "minradius": cavity_profile["minimum_added_sphere_radius_angstrom"],
    }
    assert runtime["profile"]["warnings_fatal"] is (
        cavity_profile["native_and_pedra_warnings_fatal"]
    )


def test_executable_protected_overlap_predicate_and_synthetic_fail_case():
    proto = load_json(PROTO_PATH)
    fixture = load_json(CAVITY_FIX)
    pred = proto["shell"]["cavity"]["penetration_predicate"]
    tol = pred["numerical_tolerance_A"]

    radius = load_json(RADIUS_PATH)["radii"]
    geom, symbols = _load_xyz(PROJECT_ROOT / fixture["geometry"]["path"])
    radii = np.array([radius[s] for s in symbols], dtype=float)

    npz = np.load(PROJECT_ROOT / fixture["cavity"]["mesh"]["path"])
    centers = npz["centers_angstrom"]

    # first three atoms are solute fragment, last three are first-shell waters
    solute_idx = (0, 1, 2)
    shell_idx = (3, 4, 5)
    overlap_pairs = [
        (i, j)
        for i in solute_idx
        for j in shell_idx
        if radii[i] + radii[j] - np.linalg.norm(geom[i] - geom[j]) > 0
    ]

    assert overlap_pairs == EXPECTED_OVERLAP_PAIRS
    max_pen, worst_pair, worst_center = _pair_overlap_penetration_max(
        centers,
        geom,
        radii,
        overlap_pairs,
        tol,
    )

    assert max_pen <= tol
    assert max_pen == pytest.approx(-0.01408324119508686, rel=0, abs=1e-6)
    assert worst_pair == (1, 3)

    # record geometry-derived diagnostics for deterministic audit trail
    nearest_distance = np.linalg.norm(centers - worst_center[None, :], axis=1).min()
    assert nearest_distance >= 0.0

    # synthetic probe inside overlap lens must fail deterministically
    syn = fixture["cavity"]["synthetic_probe"]
    syn_center = np.array(syn["center_A"], dtype=float)
    assert np.array_equal(syn_center, np.array([1.4, 0.0, 0.0], dtype=float))

    d_syn = np.linalg.norm(syn_center - geom, axis=1)
    s = d_syn - radii
    syn_p = max(min(-s[i], -s[j]) for i, j in overlap_pairs)
    assert syn["expected_failure"] is True
    assert syn_p > tol
    assert syn_p == pytest.approx(0.1200000000000001, rel=0, abs=1e-12)
    assert syn["penetration_A"] == pytest.approx(0.1200000000000001, rel=0, abs=1e-12)
    assert syn["center_A"] == [1.4, 0.0, 0.0]
    assert [tuple(pair) for pair in syn["overlap_pairs"]] == EXPECTED_OVERLAP_PAIRS


def test_native_cavity_topology_transition_fails_closed_across_adjacent_frames():
    fixture = load_json(TOPOLOGY_FIX)
    metadata = fixture["metadata_npz"]
    metadata_path = PROJECT_ROOT / metadata["path"]
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == metadata["sha256"]
    data = np.load(metadata_path)

    positions = [
        data[f"frame{index}_positions_angstrom"]
        for index in range(len(fixture["frames"]))
    ]
    # The second trajectory frame is the first with only the first-shell water
    # translated; the solute coordinates and atom ordering remain invariant.
    assert np.array_equal(positions[0][:3], positions[1][:3])
    displacement = positions[1][3:] - positions[0][3:]
    assert displacement == pytest.approx(
        np.tile([0.8, 0.0, 0.0], (3, 1)),
        rel=0,
        abs=1.0e-12,
    )

    component_counts = []
    for index, frame in enumerate(fixture["frames"]):
        assert frame["warning_count"] == 0
        centers = data[f"frame{index}_elSphCenter_bohr"]
        radii = data[f"frame{index}_elRadius_bohr"]
        component_counts.append(
            _native_surface_component_count(centers, radii)
        )

    assert component_counts == fixture["expected_component_counts"] == [1, 2]
    assert fixture["expected_failure_code"] == (
        "SMD_CAVITY_TOPOLOGY_DISCONTINUITY"
    )


def test_shell_predicate_hints_are_no_longer_arbitrary_subtraction():
    proto = load_json(PROTO_PATH)
    pred = proto["shell"]["cavity"]["penetration_predicate"]

    # old scalar-clearance fields must not appear in protocol freeze
    assert "corridor_clearance_min_A" not in pred
    assert "solvent_tessera_penetration_tolerance_A" not in pred
    assert "smd_cavity_tolerance_A" not in pred

    assert pred["name"] == "protected-overlap-sphere-lens"
    assert pred["numerical_tolerance_A"] == 1e-6
    assert pred["s_i"] == "||t-r_i||-R_i"
    assert pred["penetration_depth"] == "min(-s_i,-s_j)"
