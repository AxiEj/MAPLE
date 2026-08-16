from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from maple.solvation.release.evidence import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "route2_release"
GENERATOR_PATH = TOOLS / "generate_maple_cds_w1_stock_area_features.py"
AGGREGATOR_PATH = TOOLS / "aggregate_maple_cds_w1_stock_area_features.py"


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    try:
        specification = importlib.util.spec_from_file_location(name, path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


GENERATOR = _load(GENERATOR_PATH, "generate_maple_cds_w1_stock_area_features")
AGGREGATOR = _load(AGGREGATOR_PATH, "aggregate_maple_cds_w1_stock_area_features")


class _TargetTrap:
    @property
    def delta_g_kcal_mol(self):  # pragma: no cover - access is failure
        raise AssertionError("target was accessed")


def _item(index: int = 0):
    offset = 0.05 * index
    geometry = SimpleNamespace(
        atomic_numbers=(8, 1, 1),
        coordinates_angstrom=np.asarray(
            (
                (offset, 0.0, 0.0),
                (0.9572 + offset, 0.0, 0.0),
                (-0.239987 + offset, 0.927297, 0.0),
            ),
            dtype=float,
        ),
        sha256=f"{index + 1:064x}",
        charge=0,
        multiplicity=1,
    )
    return SimpleNamespace(
        canonical_solvent="water",
        opaque_record_id=f"{index + 101:064x}",
        eligible_record=SimpleNamespace(
            partition="development",
            geometry=geometry,
            record=_TargetTrap(),
        ),
    )


def _write_read_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _rewrite_read_only(path: Path, payload: object) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    _write_read_only(path, payload)


def _fake_feature_payload(
    *,
    water_ordinal: int,
    selection_index: int,
    item: object,
    preregistration_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    row = np.linspace(0.001, 0.018, 18) * (water_ordinal + 1)
    stock = float(
        row @ GENERATOR.SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    )
    payload: dict[str, object] = {
        "artifact": GENERATOR.ARTIFACT,
        "schema_version": 1,
        "status": "pass",
        "do_not_commit": True,
        "partition": "development",
        "canonical_solvent": "water",
        "dataset_loader_parsed_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_performed": False,
        "energy_only_diagnostic": True,
        "force_capability": False,
        "water_ordinal": water_ordinal,
        "selection_index": selection_index,
        "opaque_record_id": item.opaque_record_id,
        "geometry_sha256": item.eligible_record.geometry.sha256,
        "atom_count": 3,
        "symbols": ["O", "H", "H"],
        "preregistration_sha256": preregistration_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "design_parameter_names": list(GENERATOR.SMD_WATER_TENSION_PARAMETER_NAMES),
        "design_row_angstrom2_div_1000": row.tolist(),
        "stock_smd_reconstruction_kcal_mol": stock,
        "compiled_legacy_smd_kcal_mol": stock - 0.001,
        "stock_minus_compiled_legacy_kcal_mol": 0.001,
        "rotation_control_stock_error_kcal_mol": -0.002,
        "rotation_control_legacy_error_kcal_mol": 0.003,
        "rotated_stock_minus_compiled_legacy_kcal_mol": -0.004,
        "claim_boundary": GENERATOR.claim_boundary(),
    }
    payload["feature_sha256"] = GENERATOR._canonical_sha256(payload)
    return payload


def test_feature_record_is_exactly_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AGGREGATOR, "_feature_payload", _fake_feature_payload)
    path = tmp_path / "water-000.json"
    payload = _fake_feature_payload(
        water_ordinal=0,
        selection_index=4,
        item=_item(),
        preregistration_sha256="c" * 64,
        runtime_identity_sha256="d" * 64,
    )
    _write_read_only(path, payload)
    row, controls = AGGREGATOR._validate_feature_record(
        payload=payload,
        path=path,
        water_ordinal=0,
        selection_index=4,
        item=_item(),
        preregistration_sha256="c" * 64,
        runtime_identity_sha256="d" * 64,
    )
    np.testing.assert_array_equal(row, payload["design_row_angstrom2_div_1000"])
    assert controls["stock_minus_compiled_legacy_kcal_mol"] == pytest.approx(0.001)


def test_feature_record_rejects_rehashed_fabricated_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AGGREGATOR, "_feature_payload", _fake_feature_payload)
    path = tmp_path / "water-000.json"
    payload = _fake_feature_payload(
        water_ordinal=0,
        selection_index=4,
        item=_item(),
        preregistration_sha256="c" * 64,
        runtime_identity_sha256="d" * 64,
    )
    payload["design_row_angstrom2_div_1000"][0] += 10.0
    payload.pop("feature_sha256")
    payload["feature_sha256"] = GENERATOR._canonical_sha256(payload)
    _write_read_only(path, payload)
    with pytest.raises(AGGREGATOR.StockAreaAggregationError, match="reproduce"):
        AGGREGATOR._validate_feature_record(
            payload=payload,
            path=path,
            water_ordinal=0,
            selection_index=4,
            item=_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def test_matrix_and_control_diagnostics_are_target_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AGGREGATOR, "EXPECTED_WATER_RECORD_COUNT", 20)
    matrix = np.zeros((20, 18), dtype=float)
    matrix[:, :5] = np.arange(100, dtype=float).reshape(20, 5) + np.eye(20, 5)
    diagnostic = AGGREGATOR._matrix_diagnostics(matrix)
    assert diagnostic["shape"] == [20, 18]
    assert diagnostic["active_column_indices"] == [0, 1, 2, 3, 4]
    assert diagnostic["standardization"].startswith("divide-each")
    assert "Target-free" in diagnostic["claim_boundary"]
    controls = AGGREGATOR._control_diagnostics([{"parity": 0.1}, {"parity": -0.2}])
    assert controls["parity"]["maximum_absolute_kcal_mol"] == pytest.approx(0.2)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(ROOT), *arguments), text=True
    ).strip()


def _preregistration_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    for index, name in enumerate(GENERATOR.INPUT_FILE_NAMES):
        (input_root / name).write_bytes(f"input-{index}".encode())
    input_hashes = {
        name: sha256_file(input_root / name) for name in GENERATOR.INPUT_FILE_NAMES
    }
    parent_path = tmp_path / "parent.json"
    parent = {
        "artifact_id": GENERATOR.PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
        "source_git_head": "f" * 40,
        "input_root": str(input_root),
        **{
            field: input_hashes[filename]
            for filename, field in GENERATOR.PARENT_INPUT_HASH_FIELDS.items()
        },
    }
    _write_read_only(parent_path, parent)
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    preregistration_path = tmp_path / "preregistration.json"
    runtime_identity = {"contract": "test-runtime"}
    runtime_sha256 = GENERATOR._canonical_sha256(runtime_identity)
    pyscf_assets = {"contract": "test-pyscf-assets"}
    pyscf_assets_sha256 = GENERATOR._canonical_sha256(pyscf_assets)
    monkeypatch.setattr(
        AGGREGATOR,
        "_runtime_identity",
        lambda: (runtime_identity, runtime_sha256),
    )
    monkeypatch.setattr(
        AGGREGATOR,
        "_pyscf_runtime_assets",
        lambda: (pyscf_assets, pyscf_assets_sha256),
    )
    source_head = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    source_hashes = {
        name: sha256_file(ROOT / name) for name in GENERATOR.REQUIRED_SOURCE_FILE_NAMES
    }
    payload: dict[str, object] = {
        "artifact": GENERATOR.PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": GENERATOR.PREREGISTRATION_STATUS,
        "locked_at_utc": "2026-08-16T00:00:00+00:00",
        "partition": "development-water-only",
        "water_record_count": GENERATOR.EXPECTED_WATER_RECORD_COUNT,
        "water_identity_sha256": "a" * 64,
        "dataset_loader_parses_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read_by_feature_generator": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_permitted": False,
        "checkpoint_or_method_selection_permitted": False,
        "geometry_only_unit_canaries_precede_lock": True,
        "energy_only_diagnostic": True,
        "force_capability": False,
        "source_root": str(ROOT),
        "source_git_head": source_head,
        "source_git_tree": source_tree,
        "source_files_sha256": source_hashes,
        "feature_generator_sha256": source_hashes[
            GENERATOR_PATH.relative_to(ROOT).as_posix()
        ],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_sha256,
        "pyscf_runtime_assets": pyscf_assets,
        "pyscf_runtime_assets_sha256": pyscf_assets_sha256,
        "input_root": str(input_root),
        "input_files_sha256": input_hashes,
        "parent_hybrid_preregistration_path": str(parent_path),
        "parent_hybrid_preregistration_sha256": sha256_file(parent_path),
        "parent_hybrid_source_git_head": "f" * 40,
        "feature_output_dir": str(records_dir),
        "preregistration_path": str(preregistration_path),
        "area_definition": GENERATOR.stock_area_contract(),
        "linear_basis": GENERATOR.linear_basis_contract(),
        "numerical_choice_rationale": GENERATOR.numerical_choice_rationale(),
        "output_contract": GENERATOR.output_contract(),
        "claim_boundary": GENERATOR.claim_boundary(),
    }
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)
    _write_read_only(preregistration_path, payload)
    return SimpleNamespace(
        input_root=input_root,
        records_dir=records_dir,
        parent_path=parent_path,
        preregistration_path=preregistration_path,
        preregistration=payload,
        runtime_sha256=runtime_sha256,
    )


def test_preregistration_validation_binds_historical_sources_parent_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _preregistration_case(tmp_path, monkeypatch)
    observed = AGGREGATOR._validate_preregistration(
        preregistration=case.preregistration,
        preregistration_path=case.preregistration_path,
        source_root=ROOT,
        input_root=case.input_root,
        records_dir=case.records_dir,
    )
    assert observed == (sha256_file(case.preregistration_path), case.runtime_sha256)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.__setitem__("target_kcal_mol", [1.0]),
            "schema",
        ),
        (
            lambda payload: payload.__setitem__("force_capability", True),
            "force_capability",
        ),
        (
            lambda payload: payload["area_definition"].__setitem__(
                "grid_points_per_atom", 590
            ),
            "definition",
        ),
        (
            lambda payload: payload.__setitem__("runtime_identity_sha256", "0" * 64),
            "runtime digest",
        ),
    ),
)
def test_preregistration_rejects_rehashed_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    case = _preregistration_case(tmp_path, monkeypatch)
    payload = json.loads(json.dumps(case.preregistration))
    mutation(payload)
    payload.pop("self_sha256", None)
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)
    _rewrite_read_only(case.preregistration_path, payload)
    with pytest.raises(AGGREGATOR.StockAreaAggregationError, match=message):
        AGGREGATOR._validate_preregistration(
            preregistration=payload,
            preregistration_path=case.preregistration_path,
            source_root=ROOT,
            input_root=case.input_root,
            records_dir=case.records_dir,
        )


def test_preregistration_rejects_parent_input_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _preregistration_case(tmp_path, monkeypatch)
    parent = json.loads(case.parent_path.read_text())
    field = next(iter(GENERATOR.PARENT_INPUT_HASH_FIELDS.values()))
    parent[field] = "0" * 64
    _rewrite_read_only(case.parent_path, parent)
    payload = json.loads(case.preregistration_path.read_text())
    payload["parent_hybrid_preregistration_sha256"] = sha256_file(case.parent_path)
    payload.pop("self_sha256")
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)
    _rewrite_read_only(case.preregistration_path, payload)
    with pytest.raises(AGGREGATOR.StockAreaAggregationError, match="input binding"):
        AGGREGATOR._validate_preregistration(
            preregistration=payload,
            preregistration_path=case.preregistration_path,
            source_root=ROOT,
            input_root=case.input_root,
            records_dir=case.records_dir,
        )


class _Snapshot:
    root = ROOT
    head = "1" * 40
    tree = "2" * 40

    def assert_unchanged(self) -> None:
        return None


def _aggregate_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    preregistration_path = tmp_path / "preregistration.json"
    _write_read_only(
        preregistration_path,
        {"source_git_head": "3" * 40, "water_identity_sha256": "e" * 64},
    )
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    items = (_item(0), _item(1))
    water = ((4, items[0]), (9, items[1]))
    monkeypatch.setattr(AGGREGATOR, "EXPECTED_WATER_RECORD_COUNT", 2)
    monkeypatch.setattr(AGGREGATOR, "_feature_payload", _fake_feature_payload)
    monkeypatch.setattr(
        AGGREGATOR,
        "_validate_preregistration",
        lambda **_kwargs: ("c" * 64, "d" * 64),
    )
    monkeypatch.setattr(AGGREGATOR, "_load_selection", lambda **_kwargs: (None, items))
    monkeypatch.setattr(AGGREGATOR, "_water_records", lambda _rows: (water, "e" * 64))
    monkeypatch.setattr(
        AGGREGATOR,
        "RepositorySnapshot",
        SimpleNamespace(capture=lambda _root: _Snapshot()),
    )
    monkeypatch.setattr(
        AGGREGATOR,
        "collect_loaded_repository_sources",
        lambda _root, required_paths=(): tuple(required_paths),
    )
    monkeypatch.setattr(
        AGGREGATOR,
        "committed_source_hashes",
        lambda _snapshot, relative_paths: {
            relative: "f" * 64 for relative in relative_paths
        },
    )
    for ordinal, (selection_index, item) in enumerate(water):
        payload = _fake_feature_payload(
            water_ordinal=ordinal,
            selection_index=selection_index,
            item=item,
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )
        _write_read_only(records_dir / f"water-{ordinal:03d}.json", payload)
    output = tmp_path / "aggregate.json"
    return SimpleNamespace(
        args=argparse.Namespace(
            source_root=ROOT,
            input_root=input_root,
            preregistration=preregistration_path,
            records_dir=records_dir,
            output=output,
        ),
        records_dir=records_dir,
        output=output,
    )


def test_aggregate_closes_matrix_hashes_without_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _aggregate_case(tmp_path, monkeypatch)
    result = AGGREGATOR.aggregate(case.args)
    assert json.loads(case.output.read_text()) == result
    assert not case.output.stat().st_mode & 0o222
    assert result["record_count"] == 2
    assert result["experimental_targets_used_by_feature_aggregation"] is False
    assert result["hybrid_prediction_records_read"] is False
    assert result["confirmation_selection_manifest_opened"] is False
    without_digest = dict(result)
    assert without_digest.pop("aggregate_sha256") == canonical_json_sha256(
        without_digest
    )


def test_aggregate_rejects_output_collision_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _aggregate_case(tmp_path, monkeypatch)
    case.output.write_text("preexisting\n")
    with pytest.raises(FileExistsError):
        AGGREGATOR.aggregate(case.args)
    assert case.output.read_text() == "preexisting\n"


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_aggregate_requires_exact_filename_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    case = _aggregate_case(tmp_path, monkeypatch)
    if change == "missing":
        (case.records_dir / "water-001.json").unlink()
    else:
        (case.records_dir / "unexpected.json").write_text("{}\n")
    with pytest.raises(AGGREGATOR.StockAreaAggregationError, match="incomplete"):
        AGGREGATOR.aggregate(case.args)


def test_aggregator_help_does_not_import_pyscf_or_torch() -> None:
    code = f"""
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pyscf' or name.startswith('pyscf.') or name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('optional runtime imported during --help')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, {str(TOOLS)!r})
sys.argv = [{str(AGGREGATOR_PATH)!r}, '--help']
runpy.run_path({str(AGGREGATOR_PATH)!r}, run_name='__main__')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--records-dir" in result.stdout
