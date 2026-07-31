from __future__ import annotations

import hashlib
import importlib
import pickle
import subprocess
from pathlib import Path

import pytest

UPSTREAM_REVISION = "1d85bcc065003e083d2e95ab7091cb1762eeb1bf"
VACUUM_MODEL_SHA256 = (
    "83618b7b4f6680e80936f68e5e2d02f9c6ff1c62750333ca28fc5733286347c4"
)
WATER_MODEL_SHA256 = (
    "56a72b33e7aa3b26e06f7792924f2630a62f1d0a776f2843f5871195e88e3e13"
)
DATABASE_SHA256 = (
    "8a1dd006a54f0986f58b967bf7046fb72293dafa66954ec8529cdc5e68b4d405"
)
FULL_TEST_MANIFEST_SHA256 = (
    "cf529c1ffb8940d54e849e9b13f5391b68a47124823c92675d59aab6815c1418"
)


def _protocol():
    return importlib.import_module("maple.function.solvfe.resolv_protocol")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit_repository(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "MAPLE tests"],
        check=True,
    )
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _artifact_fixture(tmp_path: Path):
    module = _protocol()
    upstream = tmp_path / "ReSolv"
    upstream.mkdir()
    revision = _commit_repository(upstream)
    vacuum = tmp_path / "U_vac.pkl"
    water = tmp_path / "U_wat.pkl"
    database = tmp_path / "database.pickle"
    vacuum.write_bytes(b"vacuum-model")
    water.write_bytes(b"water-model")
    database.write_bytes(b"database")
    trajectories = tmp_path / "precomputed_trajectories"
    trajectories.mkdir()
    (trajectories / "250_50ps_load_traj_mol_1_AC").write_bytes(b"vacuum-trajectory")
    (trajectories / "250_50ps_load_wat_traj_mol_1_AC").write_bytes(b"water-trajectory")
    artifacts = module.ReSolvArtifactSet(
        upstream_root=upstream,
        vacuum_model_path=vacuum,
        water_model_path=water,
        database_path=database,
        trajectory_root=trajectories,
    )
    expected_sha256 = {
        "vacuum_model": _sha256(vacuum),
        "water_model": _sha256(water),
        "database": _sha256(database),
    }
    return module, artifacts, revision, expected_sha256


def _official_upstream_root() -> Path:
    return Path.home() / ".cache/maple/resolv/ReSolv"


def test_official_revision_and_artifact_hashes_are_frozen() -> None:
    module = _protocol()
    assert module.RESOLV_UPSTREAM_REVISION == UPSTREAM_REVISION
    assert module.RESOLV_VACUUM_MODEL_SHA256 == VACUUM_MODEL_SHA256
    assert module.RESOLV_WATER_MODEL_SHA256 == WATER_MODEL_SHA256
    assert module.RESOLV_DATABASE_SHA256 == DATABASE_SHA256
    assert module.RESOLV_FULL_TEST_MANIFEST_SHA256 == FULL_TEST_MANIFEST_SHA256
    assert len(module.RESOLV_IMPORTED_MODULE_CODE_SHA256) == 64


def test_artifact_validation_accepts_a_matching_revision_and_hashes(tmp_path: Path) -> None:
    _, artifacts, revision, expected_sha256 = _artifact_fixture(tmp_path)
    receipt = artifacts.validate(
        expected_revision=revision,
        expected_sha256=expected_sha256,
        required_k_indices=(0,),
    )
    assert receipt["source_revision"] == revision
    assert receipt["artifact_sha256"] == expected_sha256


def test_artifact_validation_rejects_a_different_source_revision(tmp_path: Path) -> None:
    module, artifacts, _, expected_sha256 = _artifact_fixture(tmp_path)
    with pytest.raises(module.ReSolvProtocolError, match="revision"):
        artifacts.validate(
            expected_revision="0" * 40,
            expected_sha256=expected_sha256,
            required_k_indices=(0,),
        )


def test_artifact_validation_rejects_a_model_hash_mismatch(tmp_path: Path) -> None:
    module, artifacts, revision, expected_sha256 = _artifact_fixture(tmp_path)
    expected_sha256["water_model"] = "0" * 64
    with pytest.raises(module.ReSolvProtocolError, match="(?i)(sha|hash).*(water|U_wat)"):
        artifacts.validate(
            expected_revision=revision,
            expected_sha256=expected_sha256,
            required_k_indices=(0,),
        )


def test_artifact_validation_rejects_a_missing_endpoint_pair(tmp_path: Path) -> None:
    module, artifacts, revision, expected_sha256 = _artifact_fixture(tmp_path)
    artifacts.water_trajectory_path(0).unlink()
    with pytest.raises(module.ReSolvProtocolError, match="(?i)missing.*(pair|trajectory)"):
        artifacts.validate(
            expected_revision=revision,
            expected_sha256=expected_sha256,
            required_k_indices=(0,),
        )


def test_zero_based_database_index_maps_to_one_based_trajectory_name(
    tmp_path: Path,
) -> None:
    _, artifacts, _, _ = _artifact_fixture(tmp_path)
    assert artifacts.vacuum_trajectory_path(0).name == "250_50ps_load_traj_mol_1_AC"
    assert artifacts.water_trajectory_path(0).name == "250_50ps_load_wat_traj_mol_1_AC"
    assert artifacts.vacuum_trajectory_path(271).name == (
        "250_50ps_load_traj_mol_272_AC"
    )
    assert artifacts.water_trajectory_path(271).name == (
        "250_50ps_load_wat_traj_mol_272_AC"
    )


def test_official_manifest_has_exact_split_and_primary_group_coverage() -> None:
    module = _protocol()
    upstream = _official_upstream_root()
    if not upstream.is_dir():
        pytest.skip("pinned ReSolv checkout is not installed")
    artifacts = module.ReSolvArtifactSet(upstream_root=upstream)
    records = module.build_resolv_manifest(artifacts)
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "test", "failed")
    }
    test_records = [record for record in records if record["split"] == "test"]
    assert len(records) == 559
    assert counts == {"train": 375, "test": 162, "failed": 22}
    classified_groups = {
        record["primary_functional_group"]
        for record in test_records
        if record["primary_functional_group"] != "unclassified"
    }
    assert len(classified_groups) == 26
    assert len(classified_groups) >= 10
    assert sum(
        record["primary_functional_group"] == "unclassified"
        for record in test_records
    ) == 9


def test_verified_database_deserializes_the_same_bytes_that_were_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _protocol()
    database_path = tmp_path / "database.pickle"
    admitted = {"mobley_test": {"smiles": "C", "expt": 1.0}}
    serialized = pickle.dumps(admitted)
    database_path.write_bytes(serialized)
    monkeypatch.setattr(
        module,
        "RESOLV_DATABASE_SHA256",
        hashlib.sha256(serialized).hexdigest(),
    )

    original_read_bytes = Path.read_bytes
    read_count = 0

    def mutate_after_read(path: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        payload = original_read_bytes(path)
        path.write_bytes(pickle.dumps({"tampered": {}}))
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    assert module._load_verified_database(database_path) == admitted
    assert read_count == 1


def test_verified_source_parser_retains_the_bytes_matched_to_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _protocol()
    upstream = tmp_path / "ReSolv"
    upstream.mkdir()
    source = upstream / "source.py"
    source.write_text("values = [1, 2, 3]\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    subprocess.run(
        ["git", "-C", str(upstream), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(upstream), "config", "user.name", "MAPLE tests"],
        check=True,
    )
    subprocess.run(["git", "-C", str(upstream), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(upstream), "commit", "-qm", "source"], check=True)

    original_read_bytes = Path.read_bytes
    read_count = 0

    def mutate_after_read(path: Path) -> bytes:
        nonlocal read_count
        read_count += 1
        payload = original_read_bytes(path)
        path.write_text("values = [999]\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    retained = module._read_verified_head_blob(upstream, source)
    assert module._literal_assignment(
        source,
        "values",
        source_bytes=retained,
    ) == (1, 2, 3)
    assert read_count == 1


def test_protocol_metadata_is_water_only_at_298K_for_neutral_supported_elements() -> None:
    metadata = _protocol().RESOLV_PROTOCOL_METADATA
    assert metadata["solvent"] == "water"
    assert metadata["temperature_kelvin"] == pytest.approx(298.15)
    assert metadata["total_charge"] == 0
    assert metadata["multiplicity"] == 1
    assert set(metadata["elements"]) == {"H", "C", "N", "O", "S", "Cl"}


def test_protocol_is_not_an_ordinary_calculator_or_additive_continuum() -> None:
    metadata = _protocol().RESOLV_PROTOCOL_METADATA
    assert metadata["protocol_kind"] == "dedicated_hydration_free_energy"
    assert metadata["ordinary_calculator"] is False
    assert metadata["additive_continuum"] is False
    assert metadata["supports_opt_freq_md"] is False
