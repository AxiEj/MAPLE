from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "operational-analytic-harmonic-rigid-trans-butane-77abe42b"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"
HEAD = "77abe42b0622ea9f829ce00de213c89a0491cbb7"
CHECKPOINT = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT = "44102bdffb58e49e95694fe2e7993f661b2178aa23136c92208f2f8b818a3734"
CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILES = {
    "measurements.json": (
        "7e68af6f9f342103d04cfe9712aea92743b3f3bd7c49d7e4a122e1ba090c6ac4"
    ),
    "cold-replay.json": (
        "02985a242f0093d56420b97342eab3099821a95dffcef06cc220a7ada092e441"
    ),
    "README.md": "3c17d9a69dc6dbb15b09cac306f77353b8db0940fb34506c5306ae3276f0ccaf",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _assert_artifact(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-operational-analytic-harmonic-rigid-panel-shard-v1"
    )
    assert artifact["status"] == "rigid-panel-shard-passed-not-admitted"
    assert artifact["execution_git_head"] == HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT
    assert artifact["capabilities"] == CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT
    assert_source_files_match_execution_commit(ROOT, artifact)

    contract = artifact["contract"]
    assert contract["asset_sha256"] == (
        "ecaa309cb468f17449a78a16e6a79acd0a5a16cf982eb52b3fb8d5d1dcd80ca3"
    )
    assert contract["shard_start"] == 7
    assert contract["shard_stop"] == 8
    assert contract["shard_molecule_ids"] == ["trans-butane"]
    assert contract["rotation_count"] == 3
    assert contract["harmonic_configuration"]["laboratory_fixed_surface_grid"] is False
    assert artifact["decision"] == {
        "all_shard_gates_passed": True,
        "legacy_laboratory_grid_rehabilitated": False,
        "public_energy_admitted": False,
        "public_force_admitted": False,
        "tier_v_admitted": False,
    }

    molecule = artifact["molecules"][0]
    assert len(artifact["molecules"]) == 1
    assert molecule["molecule_id"] == "trans-butane"
    assert molecule["chemical_formula"] == "C4H10"
    assert molecule["all_gates_passed"] is True
    assert molecule["base"]["iterations"] == 22
    assert molecule["base"]["primal_residual"] == pytest.approx(
        7.652889002253314e-13, rel=0.0, abs=1.0e-26
    )
    assert molecule["base"]["adjoint_residual"] == pytest.approx(
        2.1753222515410796e-12, rel=0.0, abs=1.0e-25
    )
    assert molecule["cold_warm"]["gate_passed"] is True
    assert molecule["cold_warm"]["source_relative_difference"] == 0.0
    assert molecule["cold_warm"]["field_relative_difference"] == 0.0
    assert molecule["cold_warm"]["energy_absolute_difference_eV"] == 0.0
    assert molecule["scalar_identity"]["gate_passed"] is True

    rigid = molecule["rigid_symmetry"]
    assert rigid["all_gates_passed"] is True
    assert all(rigid["gates"].values())
    assert rigid["translation"]["energy_abs_eV"] == 0.0
    assert rigid["permutation"]["energy_abs_eV"] == 0.0
    assert rigid["maximum_rotation_energy_abs_eV"] == pytest.approx(
        2.6966517907567322e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_rotation_force_covariance_relative"] == pytest.approx(
        3.591746985795435e-8, rel=0.0, abs=1.0e-20
    )
    assert rigid["maximum_rotation_source_covariance_relative"] == pytest.approx(
        2.899118004565727e-9, rel=0.0, abs=1.0e-21
    )
    assert rigid["maximum_primal_residual"] == pytest.approx(
        7.65300346861557e-13, rel=0.0, abs=1.0e-26
    )
    assert rigid["maximum_adjoint_residual"] == pytest.approx(
        2.1753307182778503e-12, rel=0.0, abs=1.0e-25
    )
    assert molecule["field_covariance"]["maximum_relative"] == pytest.approx(
        7.601105449126282e-9, rel=0.0, abs=1.0e-21
    )

    measured = {key: artifact[key] for key in ("contract", "molecules", "decision")}
    assert canonical_json_sha256(measured) == MEASUREMENT


def test_trans_butane_rigid_artifacts_are_bound_and_replay_exactly():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_artifact(primary)
    _assert_artifact(replay)
    for key in ("contract", "molecules", "decision", "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_trans_butane_rigid_evidence_hashes_are_frozen():
    for name, expected in FILES.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILES
