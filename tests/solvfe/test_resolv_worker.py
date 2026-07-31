from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import io
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

PINNED_SOURCE_REVISION = "1d85bcc065003e083d2e95ab7091cb1762eeb1bf"


def _worker():
    return importlib.import_module("maple.function.solvfe._resolv_worker")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, str]:
    path.write_bytes(payload)
    return {"path": str(path), "sha256": _sha256(path)}


def _source_archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, payload in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
    return output.getvalue()


def _request(tmp_path: Path, *, k_index: int = 7) -> dict[str, object]:
    worker = _worker()
    upstream = tmp_path / "ReSolv"
    upstream.mkdir()
    trajectories = upstream / worker.TRAJECTORY_DIRECTORY
    trajectories.mkdir(parents=True)
    trajectory_id = k_index + 1
    return {
        "upstream_root": str(upstream),
        "source_revision": PINNED_SOURCE_REVISION,
        "vacuum_model": {
            "path": str(
                tmp_path / worker.OFFICIAL_ARTIFACT_BASENAMES["vacuum_model"]
            ),
            "sha256": worker.OFFICIAL_VACUUM_MODEL_SHA256,
        },
        "database": {
            "path": str(tmp_path / worker.OFFICIAL_ARTIFACT_BASENAMES["database"]),
            "sha256": worker.OFFICIAL_DATABASE_SHA256,
        },
        "water_model": {
            "path": str(
                tmp_path / worker.OFFICIAL_ARTIFACT_BASENAMES["water_model"]
            ),
            "sha256": worker.OFFICIAL_WATER_MODEL_SHA256,
        },
        "record": {
            "k_index": k_index,
            "smiles": "C",
            "vacuum_trajectory": _write(
                trajectories / f"250_50ps_load_traj_mol_{trajectory_id}_AC",
                b"vacuum trajectory",
            ),
            "water_trajectory": _write(
                trajectories / f"250_50ps_load_wat_traj_mol_{trajectory_id}_AC",
                b"water trajectory",
            ),
        },
    }


def _call_leaf_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_worker_source_contains_no_fresh_conformer_calls() -> None:
    tree = ast.parse(inspect.getsource(_worker()))
    call_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (name := _call_leaf_name(node)) is not None
    }
    assert call_names.isdisjoint(
        {"EmbedMolecule", "EmbedMultipleConfs", "ETKDG", "ETKDGv2", "ETKDGv3"}
    )


def test_worker_source_contains_no_md_propagation_calls() -> None:
    tree = ast.parse(inspect.getsource(_worker()))
    call_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (name := _call_leaf_name(node)) is not None
    }
    assert call_names.isdisjoint(
        {
            "run_simulation",
            "simulate",
            "trajectory_generator",
            "trajectory_generator_init",
            "step",
            "scan",
        }
    )


def test_worker_verifies_every_artifact_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker()
    raw_request = _request(tmp_path)
    events: list[tuple[str, str]] = []

    def verify_source(upstream_root: Path, revision: str) -> None:
        assert upstream_root == (tmp_path / "ReSolv").resolve()
        events.append(("verify_source", revision))

    def verified_bytes(path: Path, expected: str, label: str) -> bytes:
        events.append(("verify_file", label))
        if path.is_file():
            return path.read_bytes()
        return f"verified:{label}".encode()

    def run_verified(request: dict[str, object]) -> dict[str, object]:
        events.append(("run", str(request["source_revision"])))
        return {"ok": True}

    monkeypatch.setattr(worker, "_verify_source", verify_source)
    monkeypatch.setattr(worker, "_verified_bytes", verified_bytes)
    monkeypatch.setattr(
        worker,
        "_head_blob_oid",
        lambda _root, relative: worker._git_blob_oid(
            (tmp_path / "ReSolv" / relative).read_bytes()
        ),
    )
    monkeypatch.setattr(worker, "_run_verified", run_verified)

    assert worker.run(raw_request) == {"ok": True}
    assert events == [
        ("verify_source", PINNED_SOURCE_REVISION),
        ("verify_file", "vacuum_model"),
        ("verify_file", "database"),
        ("verify_file", "water_model"),
        ("verify_file", "vacuum_trajectory[7]"),
        ("verify_file", "water_trajectory[7]"),
        ("run", PINNED_SOURCE_REVISION),
    ]


def test_worker_rejects_k_index_and_trajectory_name_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker()
    raw_request = _request(tmp_path, k_index=7)
    record = raw_request["record"]
    assert isinstance(record, dict)
    record["water_trajectory"] = _write(
        tmp_path / "ReSolv" / _worker().TRAJECTORY_DIRECTORY / "250_50ps_load_wat_traj_mol_9_AC",
        b"wrong trajectory id",
    )
    run_called = False

    monkeypatch.setattr(worker, "_verify_source", lambda *_: None)
    monkeypatch.setattr(worker, "_verified_bytes", lambda *_: b"verified")
    monkeypatch.setattr(
        worker, "_head_blob_oid", lambda *_: worker._git_blob_oid(b"verified")
    )

    def run_verified(_request: dict[str, object]) -> dict[str, object]:
        nonlocal run_called
        run_called = True
        return {"ok": True}

    monkeypatch.setattr(worker, "_run_verified", run_verified)

    with pytest.raises(RuntimeError, match="k_index=7"):
        worker.run(raw_request)
    assert run_called is False


def test_worker_defaults_to_fail_fast_for_record_errors(tmp_path: Path) -> None:
    request = _worker()._normalise_request(_request(tmp_path))
    assert request["continue_on_record_error"] is False


def test_worker_pins_source_revision_and_endpoint_snapshot_count() -> None:
    worker = _worker()
    assert worker.PINNED_SOURCE_REVISION == PINNED_SOURCE_REVISION
    assert worker.EXPECTED_SNAPSHOTS_PER_ENDPOINT == 40
    assert worker.OFFICIAL_ARTIFACT_HASHES == {
        "vacuum_model": "83618b7b4f6680e80936f68e5e2d02f9c6ff1c62750333ca28fc5733286347c4",
        "water_model": "56a72b33e7aa3b26e06f7792924f2630a62f1d0a776f2843f5871195e88e3e13",
        "database": "8a1dd006a54f0986f58b967bf7046fb72293dafa66954ec8529cdc5e68b4d405",
    }
    assert set(worker.OFFICIAL_ARTIFACT_BASENAMES) == {
        "vacuum_model",
        "water_model",
        "database",
    }


def test_worker_enables_x64_before_importing_jax() -> None:
    source = inspect.getsource(_worker()._run_from_snapshot)
    enable_environment = 'os.environ["JAX_ENABLE_X64"] = "True"'
    import_jax = 'importlib.import_module("jax")'
    enable_config = 'jax.config.update("jax_enable_x64", True)'
    assert enable_environment in source
    assert enable_config in source
    assert source.index(enable_environment) < source.index(import_jax)
    assert source.index(import_jax) < source.index(enable_config)
    assert source.index('os.environ["NVIDIA_TF32_OVERRIDE"] = "0"') < source.index(
        import_jax
    )
    assert 'jax.config.update("jax_default_matmul_precision", "highest")' in source
    assert '"reduced_float32_matmul_disabled": True' in source


@pytest.mark.parametrize("role", ["vacuum_model", "water_model"])
def test_model_role_requires_exact_official_basename_before_loading(
    role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker()
    raw = _request(tmp_path)
    asset = raw[role]
    assert isinstance(asset, dict)
    wrong_role = "water_model" if role == "vacuum_model" else "vacuum_model"
    asset["path"] = str(
        tmp_path / worker.OFFICIAL_ARTIFACT_BASENAMES[wrong_role]
    )
    loaded_labels: list[str] = []

    monkeypatch.setattr(worker, "_verify_source", lambda *_: None)

    def verified_bytes(_path: Path, _expected: str, label: str) -> bytes:
        loaded_labels.append(label)
        return b"unexpected"

    monkeypatch.setattr(worker, "_verified_bytes", verified_bytes)
    with pytest.raises(RuntimeError, match=f"{role} basename.*wrong artifact role"):
        worker._verify_everything(worker._normalise_request(raw))
    assert role not in loaded_labels


def test_self_reported_model_hash_cannot_admit_malicious_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker()
    raw_request = _request(tmp_path)
    malicious = tmp_path / worker.OFFICIAL_ARTIFACT_BASENAMES["vacuum_model"]
    malicious.write_bytes(b"malicious pickle")
    model = raw_request["vacuum_model"]
    assert isinstance(model, dict)
    model["sha256"] = _sha256(malicious)
    loader_called = False

    monkeypatch.setattr(worker, "_verify_source", lambda *_: None)

    def run_verified(_request: dict[str, object]) -> dict[str, object]:
        nonlocal loader_called
        loader_called = True
        return {"ok": True}

    monkeypatch.setattr(worker, "_run_verified", run_verified)
    with pytest.raises(RuntimeError, match="not the admitted official SHA-256"):
        worker.run(raw_request)
    assert loader_called is False


def test_non_head_trajectory_is_rejected_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker()
    raw_request = _request(tmp_path)
    loader_called = False

    monkeypatch.setattr(worker, "_verify_source", lambda *_: None)
    monkeypatch.setattr(
        worker,
        "_verified_bytes",
        lambda path, *_: path.read_bytes() if path.is_file() else b"verified",
    )
    monkeypatch.setattr(worker, "_head_blob_oid", lambda *_: "0" * 40)

    def run_verified(_request: dict[str, object]) -> dict[str, object]:
        nonlocal loader_called
        loader_called = True
        return {"ok": True}

    monkeypatch.setattr(worker, "_run_verified", run_verified)
    with pytest.raises(RuntimeError, match="not the exact blob from pinned HEAD"):
        worker.run(raw_request)
    assert loader_called is False


def test_deserialization_uses_preverified_bytes_not_paths() -> None:
    source = inspect.getsource(_worker()._run_from_snapshot)
    assert "_load_verified(verified_payloads" in source
    assert ".open(" not in source
    assert "dill.load(handle)" not in source


def test_bar_receipts_include_float64_intermediates_and_no_fake_uncertainty() -> None:
    source = inspect.getsource(_worker()._run_from_snapshot)
    for receipt in (
        "vacuum_on_vacuum",
        "water_on_vacuum",
        "vacuum_on_water",
        "water_on_water",
        "dimensionless_work_vacuum",
        "dimensionless_work_water",
        "stored_vacuum_traj_aux_energy_V0",
        "stored_water_traj_aux_energy_Vp",
        "recomputed_cross_rVp",
        "recomputed_cross_rV0",
        '"status": "not_computed"',
    ):
        assert receipt in source
    assert "must be float64" in source
    assert '"entropy_kcal_mol_k"' not in source
    assert '"entropy_term"' not in source
    assert '"upstream_entropy_like_raw_unverified_units"' in source


def _mock_runtime_versions(
    worker: Any, monkeypatch: pytest.MonkeyPatch, **overrides: str
) -> dict[str, str]:
    versions = dict(worker.ADMITTED_RUNTIME_VERSIONS)
    versions["jaxlib"] = "0.4.23"
    versions.update(overrides)
    monkeypatch.setattr(
        worker.importlib.metadata, "version", lambda name: versions[name]
    )
    return versions


def test_runtime_contract_accepts_only_exact_cpu_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    versions = _mock_runtime_versions(worker, monkeypatch)
    monkeypatch.setattr(worker.sys, "version", "3.10.20 admitted")
    receipt = worker._verify_runtime_versions("cpu")
    assert receipt["python"] == "3.10.20"
    assert receipt["packages"] == versions


def test_runtime_contract_locks_direct_upstream_import_dependencies() -> None:
    worker = _worker()
    assert worker.ADMITTED_RUNTIME_VERSIONS["coax"] == "0.1.13"
    assert worker.ADMITTED_RUNTIME_VERSIONS["ml-collections"] == "0.1.1"


def test_runtime_contract_rejects_python_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    _mock_runtime_versions(worker, monkeypatch)
    monkeypatch.setattr(worker.sys, "version", "3.10.19 drifted")
    with pytest.raises(RuntimeError, match="Python runtime drifted"):
        worker._verify_runtime_versions("cpu")


def test_runtime_contract_rejects_package_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    _mock_runtime_versions(worker, monkeypatch, numpy="1.26.3")
    monkeypatch.setattr(worker.sys, "version", "3.10.20 admitted")
    with pytest.raises(RuntimeError, match='"numpy"'):
        worker._verify_runtime_versions("cpu")


def test_runtime_contract_rejects_backend_specific_jaxlib_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    _mock_runtime_versions(worker, monkeypatch, jaxlib="0.4.23")
    monkeypatch.setattr(worker.sys, "version", "3.10.20 admitted")
    with pytest.raises(RuntimeError, match='"jaxlib"'):
        worker._verify_runtime_versions("gpu")


def test_expected_backend_is_fail_closed(tmp_path: Path) -> None:
    raw = _request(tmp_path)
    raw["expected_backend"] = "tpu"
    with pytest.raises(ValueError, match="cpu or gpu"):
        _worker()._normalise_request(raw)


def test_pinned_snapshot_wins_over_untracked_live_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker()
    live = tmp_path / "live"
    live.mkdir()
    sentinel = tmp_path / "executed"
    (live / "util.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    archive = _source_archive(
        {
            "chemtrain/__init__.py": b"SOURCE = 'snapshot'\n",
            "util/__init__.py": b"SOURCE = 'snapshot'\n",
            "jax_sgmc/__init__.py": b"SOURCE = 'snapshot'\n",
        }
    )
    monkeypatch.setattr(worker, "_git_archive_bytes", lambda _root: archive)
    sys.modules.pop("util", None)
    sys.path.insert(0, str(live))
    try:
        with worker._pinned_source_snapshot(live) as provenance:
            imported = importlib.import_module("util")
            assert imported.SOURCE == "snapshot"
            assert provenance["revision"] == PINNED_SOURCE_REVISION
            assert provenance["exported_roots"] == [
                "chemtrain",
                "jax_sgmc",
                "util",
            ]
            assert not sentinel.exists()
        assert "util" not in sys.modules
    finally:
        while str(live) in sys.path:
            sys.path.remove(str(live))
        sys.modules.pop("util", None)


def test_source_archive_rejects_links(tmp_path: Path) -> None:
    worker = _worker()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        entry = tarfile.TarInfo("util/escape")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../../escape"
        archive.addfile(entry)
    with pytest.raises(RuntimeError, match="entry type"):
        worker._extract_safe_source_archive(output.getvalue(), tmp_path / "out")


def test_source_archive_rejects_path_traversal(tmp_path: Path) -> None:
    worker = _worker()
    archive = _source_archive(
        {
            "chemtrain/__init__.py": b"",
            "util/../../escape.py": b"bad",
        }
    )
    with pytest.raises(RuntimeError, match="archive path"):
        worker._extract_safe_source_archive(archive, tmp_path / "out")
