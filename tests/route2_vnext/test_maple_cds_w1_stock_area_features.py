from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
)

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = (
    ROOT / "tools" / "route2_release" / "generate_maple_cds_w1_stock_area_features.py"
)
CREATOR_PATH = (
    ROOT
    / "tools"
    / "route2_release"
    / "create_maple_cds_w1_stock_area_preregistration.py"
)


def _load_generator():
    specification = importlib.util.spec_from_file_location(
        "generate_maple_cds_w1_stock_area_features",
        GENERATOR_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _load_creator():
    sys.path.insert(0, str(CREATOR_PATH.parent))
    try:
        specification = importlib.util.spec_from_file_location(
            "create_maple_cds_w1_stock_area_preregistration",
            CREATOR_PATH,
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


CREATOR = _load_creator()


class _TargetTrap:
    @property
    def delta_g_kcal_mol(self):  # pragma: no cover - access fails the test
        raise AssertionError("experimental target was accessed")


def _water_item():
    geometry = SimpleNamespace(
        atomic_numbers=(8, 1, 1),
        coordinates_angstrom=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.9572, 0.0, 0.0),
                (-0.239987, 0.927297, 0.0),
            ),
            dtype=float,
        ),
        sha256="a" * 64,
        charge=0,
        multiplicity=1,
    )
    return SimpleNamespace(
        canonical_solvent="water",
        eligible_record=SimpleNamespace(
            partition="development",
            geometry=geometry,
            record=_TargetTrap(),
        ),
        opaque_record_id="b" * 64,
    )


def test_payload_is_target_blind_and_marks_energy_only_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    row = np.linspace(0.001, 0.018, 18)
    legacy = float(row @ stock) - 0.01
    calls: list[np.ndarray] = []

    def fake_row(_symbols, positions):
        calls.append(np.asarray(positions).copy())
        return row.copy(), legacy

    monkeypatch.setattr(GENERATOR, "_pyscf_stock_area_row", fake_row)
    payload = GENERATOR._feature_payload(
        water_ordinal=2,
        selection_index=7,
        item=_water_item(),
        preregistration_sha256="c" * 64,
        runtime_identity_sha256="d" * 64,
    )

    assert len(calls) == 2
    assert payload["experimental_targets_used_by_feature_computation"] is False
    assert payload["experimental_targets_emitted"] is False
    assert payload["hybrid_prediction_records_read"] is False
    assert payload["energy_only_diagnostic"] is True
    assert payload["force_capability"] is False
    assert payload["claim_boundary"] == GENERATOR.claim_boundary()
    assert payload["stock_minus_compiled_legacy_kcal_mol"] == pytest.approx(0.01)
    assert not GENERATOR.PROHIBITED_OUTPUT_KEYS.intersection(payload)
    without_digest = dict(payload)
    observed = without_digest.pop("feature_sha256")
    assert observed == GENERATOR._canonical_sha256(without_digest)


def test_payload_fails_closed_when_stock_energy_does_not_match_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = np.linspace(0.001, 0.018, 18)
    stock_energy = float(row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2)
    monkeypatch.setattr(
        GENERATOR,
        "_pyscf_stock_area_row",
        lambda _symbols, _positions: (row.copy(), stock_energy - 0.2),
    )
    with pytest.raises(GENERATOR.StockAreaFeatureError, match="close legacy"):
        GENERATOR._feature_payload(
            water_ordinal=0,
            selection_index=0,
            item=_water_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def test_payload_fails_closed_when_rotation_control_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    row = np.linspace(0.001, 0.018, 18)
    shifted = row.copy()
    shifted[0] += 0.01
    calls = iter(((row, float(row @ stock)), (shifted, float(shifted @ stock))))
    monkeypatch.setattr(
        GENERATOR,
        "_pyscf_stock_area_row",
        lambda _symbols, _positions: next(calls),
    )
    with pytest.raises(GENERATOR.StockAreaFeatureError, match="rotation control"):
        GENERATOR._feature_payload(
            water_ordinal=0,
            selection_index=0,
            item=_water_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def test_payload_fails_closed_when_legacy_rotation_control_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    row = np.linspace(0.001, 0.018, 18)
    stock_energy = float(row @ stock)
    calls = iter(((row, stock_energy), (row, stock_energy + 0.2)))
    monkeypatch.setattr(
        GENERATOR,
        "_pyscf_stock_area_row",
        lambda _symbols, _positions: next(calls),
    )
    with pytest.raises(GENERATOR.StockAreaFeatureError, match="Legacy SMD rotation"):
        GENERATOR._feature_payload(
            water_ordinal=0,
            selection_index=0,
            item=_water_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def test_payload_fails_closed_when_rotated_parity_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock = SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    row = np.linspace(0.001, 0.018, 18)
    original_energy = float(row @ stock)
    rotated = row.copy()
    rotated[0] += 0.04 / float(stock[0])
    rotated_energy = float(rotated @ stock)
    calls = iter(((row, original_energy), (rotated, original_energy - 0.04)))
    monkeypatch.setattr(
        GENERATOR,
        "_pyscf_stock_area_row",
        lambda _symbols, _positions: next(calls),
    )
    with pytest.raises(GENERATOR.StockAreaFeatureError, match="Rotated stock-area"):
        GENERATOR._feature_payload(
            water_ordinal=0,
            selection_index=0,
            item=_water_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )
    assert rotated_energy - original_energy == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("charge", "multiplicity", "message"),
    ((1, 1, "neutral"), (0, 2, "singlet")),
)
def test_payload_rejects_wrong_charge_or_spin(
    charge: int,
    multiplicity: int,
    message: str,
) -> None:
    item = _water_item()
    item.eligible_record.geometry.charge = charge
    item.eligible_record.geometry.multiplicity = multiplicity
    with pytest.raises(GENERATOR.StockAreaFeatureError, match=message):
        GENERATOR._feature_payload(
            water_ordinal=0,
            selection_index=0,
            item=item,
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def test_rotation_control_is_a_proper_orthogonal_matrix() -> None:
    rotation = GENERATOR._validate_rotation_matrix()
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=2.0e-15)


def test_real_pyscf_water_closes_compiled_legacy_energy() -> None:
    pytest.importorskip("pyscf")
    symbols = ("O", "H", "H")
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.239987, 0.927297, 0.0)),
        dtype=float,
    )
    row, legacy = GENERATOR._pyscf_stock_area_row(symbols, positions)
    stock = float(row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2)
    assert row.shape == (18,)
    assert abs(stock - legacy) <= GENERATOR.STOCK_ENERGY_PARITY_TOLERANCE_KCAL_MOL


def _write_read_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _preregistration_case(
    tmp_path: Path,
    *,
    parent_input_root: Path | None = None,
) -> tuple[Path, Path, Path, dict[str, object]]:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    for index, name in enumerate(GENERATOR.INPUT_FILE_NAMES):
        (input_root / name).write_bytes(f"input-{index}".encode())
    input_hashes = {
        name: hashlib.sha256((input_root / name).read_bytes()).hexdigest()
        for name in GENERATOR.INPUT_FILE_NAMES
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
        "input_root": str(
            input_root if parent_input_root is None else parent_input_root
        ),
        **{
            field: input_hashes[filename]
            for filename, field in GENERATOR.PARENT_INPUT_HASH_FIELDS.items()
        },
    }
    _write_read_only(parent_path, parent)
    runtime_identity, runtime_sha = GENERATOR._runtime_identity()
    pyscf_assets, pyscf_assets_sha = GENERATOR._pyscf_runtime_assets()
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in GENERATOR.REQUIRED_SOURCE_FILE_NAMES
    }
    preregistration_path = tmp_path / "preregistration.json"
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
        "source_git_head": "d" * 40,
        "source_git_tree": "e" * 40,
        "source_files_sha256": source_hashes,
        "feature_generator_sha256": source_hashes[
            GENERATOR_PATH.relative_to(ROOT).as_posix()
        ],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_sha,
        "pyscf_runtime_assets": pyscf_assets,
        "pyscf_runtime_assets_sha256": pyscf_assets_sha,
        "input_root": str(input_root),
        "input_files_sha256": input_hashes,
        "parent_hybrid_preregistration_path": str(parent_path),
        "parent_hybrid_preregistration_sha256": hashlib.sha256(
            parent_path.read_bytes()
        ).hexdigest(),
        "parent_hybrid_source_git_head": "f" * 40,
        "feature_output_dir": str(tmp_path / "features"),
        "preregistration_path": str(preregistration_path),
        "area_definition": GENERATOR.stock_area_contract(),
        "linear_basis": GENERATOR.linear_basis_contract(),
        "numerical_choice_rationale": GENERATOR.numerical_choice_rationale(),
        "output_contract": GENERATOR.output_contract(),
        "claim_boundary": GENERATOR.claim_boundary(),
    }
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)
    assert set(payload) == GENERATOR.preregistration_schema_keys()
    _write_read_only(preregistration_path, payload)
    return preregistration_path, parent_path, input_root, payload


def _fake_clean_git(_root: Path, *arguments: str) -> str:
    values = {
        ("rev-parse", "HEAD"): "d" * 40,
        ("rev-parse", "HEAD^{tree}"): "e" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    return values[arguments]


def test_preregistration_validation_binds_exact_schema_parent_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration_path, _parent, input_root, payload = _preregistration_case(tmp_path)
    monkeypatch.setattr(GENERATOR, "_git", _fake_clean_git)
    GENERATOR._validate_preregistration(
        preregistration=payload,
        preregistration_path=preregistration_path,
        source_root=ROOT,
        input_root=input_root,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.__setitem__(
                "experimental_target_values_accidentally_embedded", [1.0]
            ),
            "top-level schema",
        ),
        (
            lambda payload: payload.__setitem__("force_capability", True),
            "force_capability",
        ),
        (
            lambda payload: payload["pyscf_runtime_assets"].__setitem__(
                "pyscf_version", "patched"
            ),
            "PySCF runtime assets",
        ),
    ),
)
def test_preregistration_rejects_rehashed_claim_or_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    preregistration_path, _parent, input_root, original = _preregistration_case(
        tmp_path
    )
    payload = json.loads(json.dumps(original))
    mutation(payload)
    payload.pop("self_sha256", None)
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)
    monkeypatch.setattr(GENERATOR, "_git", _fake_clean_git)
    with pytest.raises(GENERATOR.StockAreaFeatureError, match=message):
        GENERATOR._validate_preregistration(
            preregistration=payload,
            preregistration_path=preregistration_path,
            source_root=ROOT,
            input_root=input_root,
        )


def test_preregistration_rejects_parent_input_root_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = tmp_path / "other-inputs"
    preregistration_path, _parent, input_root, payload = _preregistration_case(
        tmp_path,
        parent_input_root=other,
    )
    monkeypatch.setattr(GENERATOR, "_git", _fake_clean_git)
    with pytest.raises(GENERATOR.StockAreaFeatureError, match="input root"):
        GENERATOR._validate_preregistration(
            preregistration=payload,
            preregistration_path=preregistration_path,
            source_root=ROOT,
            input_root=input_root,
        )


@pytest.mark.parametrize("mode", ("missing", "mismatch"))
def test_parent_input_hash_binding_fails_closed(mode: str) -> None:
    input_hashes = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(GENERATOR.INPUT_FILE_NAMES)
    }
    parent = {
        field: input_hashes[filename]
        for filename, field in GENERATOR.PARENT_INPUT_HASH_FIELDS.items()
    }
    field = GENERATOR.PARENT_INPUT_HASH_FIELDS[GENERATOR.INPUT_FILE_NAMES[0]]
    if mode == "missing":
        parent.pop(field)
    else:
        parent[field] = "f" * 64
    with pytest.raises(GENERATOR.StockAreaFeatureError):
        GENERATOR._validate_parent_input_hashes(parent, input_hashes)


def test_creator_rejects_a_different_source_checkout(tmp_path: Path) -> None:
    with pytest.raises(GENERATOR.StockAreaFeatureError, match="own source checkout"):
        CREATOR.create(SimpleNamespace(source_root=tmp_path))


def test_exclusive_writer_is_read_only_and_does_not_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    GENERATOR._write_json_exclusive(path, {"status": "pass"})
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        GENERATOR._write_json_exclusive(path, {"status": "overwritten"})
    assert json.loads(path.read_text()) == {"status": "pass"}


@pytest.mark.parametrize("command", (GENERATOR_PATH, CREATOR_PATH))
def test_tools_expose_dependency_light_help(command: Path) -> None:
    code = f"""\nimport builtins, runpy, sys\noriginal = builtins.__import__\ndef guarded(name, *args, **kwargs):\n    if name == "pyscf" or name.startswith("pyscf.") or name == "torch" or name.startswith("torch."):\n        raise RuntimeError("optional runtime imported during --help")\n    return original(name, *args, **kwargs)\nbuiltins.__import__ = guarded\nsys.path.insert(0, {str(command.parent)!r})\nsys.argv = [{str(command)!r}, "--help"]\nrunpy.run_path({str(command)!r}, run_name="__main__")\n"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-root" in result.stdout
