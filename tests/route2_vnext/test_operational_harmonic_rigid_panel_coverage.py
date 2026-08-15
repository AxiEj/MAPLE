from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "docs" / "route2" / "evidence"
PANEL_ASSET = (
    ROOT / "tools" / "route2_release" / "data" / "fixedbox590_pes_panel_v1.json"
)
EVIDENCE_GLOB = "operational-analytic-harmonic-rigid-*"
ASSET_SHA256 = "ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3"
SCALAR_ID = (
    "route2-operational-macepolar-analytic-gaussian-multipole-"
    "smoothharmonicgalerkin-cpcm-v1"
)
PROFILE_ID = (
    "route2-profile-operational-macepolar-analytic-gaussian-multipole-"
    "smoothharmonicgalerkin-cpcm-v1"
)
CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_equilibrium_rigid_panel_has_exact_contiguous_two_process_coverage():
    panel = _load(PANEL_ASSET)
    expected_ids = [molecule["molecule_id"] for molecule in panel["molecules"]]

    shards: list[tuple[int, dict[str, object], Path]] = []
    for evidence_dir in sorted(EVIDENCE_ROOT.glob(EVIDENCE_GLOB)):
        primary_path = evidence_dir / "measurements.json"
        replay_path = evidence_dir / "cold-replay.json"
        if not primary_path.is_file() or not replay_path.is_file():
            continue
        primary = _load(primary_path)
        replay = _load(replay_path)
        contract = primary["contract"]
        shards.append((contract["shard_start"], primary, evidence_dir))
        for key in ("contract", "molecules", "decision", "measurement_sha256"):
            assert replay[key] == primary[key]
        assert replay["runtime_seconds"] != primary["runtime_seconds"]

    shards.sort(key=lambda item: item[0])
    assert len(shards) == len(expected_ids) == 20
    assert [start for start, _, _ in shards] == list(range(20))

    observed_ids: list[str] = []
    energy_errors: dict[str, float] = {}
    force_errors: dict[str, float] = {}
    source_errors: dict[str, float] = {}
    field_errors: dict[str, float] = {}
    for start, artifact, evidence_dir in shards:
        contract = artifact["contract"]
        assert contract["shard_stop"] == start + 1
        assert contract["molecule_count"] == 20
        assert contract["asset_sha256"] == ASSET_SHA256
        assert contract["scalar_id"] == SCALAR_ID
        assert contract["profile_id"] == PROFILE_ID
        assert (
            contract["harmonic_configuration"]["laboratory_fixed_surface_grid"] is False
        )
        assert artifact["capabilities"] == CAPABILITIES
        assert artifact["decision"] == {
            "all_shard_gates_passed": True,
            "legacy_laboratory_grid_rehabilitated": False,
            "public_energy_admitted": False,
            "public_force_admitted": False,
            "tier_v_admitted": False,
        }

        assert len(artifact["molecules"]) == 1
        molecule = artifact["molecules"][0]
        molecule_id = molecule["molecule_id"]
        assert contract["shard_molecule_ids"] == [molecule_id]
        assert molecule["all_gates_passed"] is True
        assert molecule["cold_warm"]["gate_passed"] is True
        assert molecule["scalar_identity"]["gate_passed"] is True
        assert molecule["rigid_symmetry"]["all_gates_passed"] is True
        assert all(molecule["rigid_symmetry"]["gates"].values())
        assert (evidence_dir / "SHA256SUMS").is_file()

        observed_ids.append(molecule_id)
        rigid = molecule["rigid_symmetry"]
        energy_errors[molecule_id] = rigid["maximum_rotation_energy_abs_eV"]
        force_errors[molecule_id] = rigid["maximum_rotation_force_covariance_relative"]
        source_errors[molecule_id] = rigid[
            "maximum_rotation_source_covariance_relative"
        ]
        field_errors[molecule_id] = molecule["field_covariance"]["maximum_relative"]

    assert observed_ids == expected_ids
    assert max(energy_errors, key=energy_errors.get) == "benzene"
    assert energy_errors["benzene"] == pytest.approx(
        4.2611645767465234e-8, rel=0.0, abs=1.0e-20
    )
    assert max(force_errors, key=force_errors.get) == "pyridine"
    assert force_errors["pyridine"] == pytest.approx(
        6.934036634982884e-8, rel=0.0, abs=1.0e-20
    )
    assert max(source_errors, key=source_errors.get) == "nitromethane"
    assert source_errors["nitromethane"] == pytest.approx(
        9.046633806516271e-9, rel=0.0, abs=1.0e-21
    )
    assert max(field_errors, key=field_errors.get) == "chloroform"
    assert field_errors["chloroform"] == pytest.approx(
        5.0708117686345365e-8, rel=0.0, abs=1.0e-20
    )
