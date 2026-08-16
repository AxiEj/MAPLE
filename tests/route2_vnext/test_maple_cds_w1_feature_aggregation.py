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

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2,
    SMD_WATER_TENSION_PARAMETER_NAMES,
)
from maple.solvation.release.evidence import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[2]
AGGREGATOR_PATH = ROOT / "tools/route2_release/aggregate_maple_cds_w1_features.py"
GENERATOR_PATH = ROOT / "tools/route2_release/generate_maple_cds_w1_features.py"


def _load(path: Path, name: str):
    tools = str(path.parent)
    sys.path.insert(0, tools)
    try:
        specification = importlib.util.spec_from_file_location(name, path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


GENERATOR = _load(GENERATOR_PATH, "generate_maple_cds_w1_features")
AGGREGATOR = _load(AGGREGATOR_PATH, "aggregate_maple_cds_w1_features")


def _item(index: int = 0):
    offset = 0.05 * index
    geometry = SimpleNamespace(
        atomic_numbers=(8, 1, 1),
        coordinates_angstrom=np.asarray(
            (
                (offset, 0.000000, 0.000000),
                (0.957200 + offset, 0.000000, 0.000000),
                (-0.239987 + offset, 0.927297, 0.000000),
            ),
            dtype=float,
        ),
        sha256=f"{index + 1:064x}",
    )
    return SimpleNamespace(
        canonical_solvent="water",
        opaque_record_id=f"{index + 101:064x}",
        eligible_record=SimpleNamespace(
            partition="development",
            geometry=geometry,
        ),
    )


def _write_read_only(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _rewrite_read_only(path: Path, payload: object) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    _write_read_only(path, payload)


def _feature(
    path: Path,
    *,
    water_ordinal: int = 0,
    selection_index: int = 4,
    item=None,
    preregistration_sha256: str = "c" * 64,
    runtime_identity_sha256: str = "d" * 64,
) -> dict[str, object]:
    payload = GENERATOR._feature_payload(
        water_ordinal=water_ordinal,
        selection_index=selection_index,
        item=_item() if item is None else item,
        preregistration_sha256=preregistration_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
    )
    _write_read_only(path, payload)
    return payload


def _rehash_feature(payload: dict[str, object]) -> None:
    payload.pop("feature_sha256", None)
    payload["feature_sha256"] = canonical_json_sha256(payload)


def test_feature_record_is_regenerated_from_frozen_geometry(tmp_path: Path) -> None:
    path = tmp_path / "water-000.json"
    payload = _feature(path)
    row = AGGREGATOR._validate_feature_record(
        payload=payload,
        path=path,
        water_ordinal=0,
        selection_index=4,
        item=_item(),
        preregistration_sha256="c" * 64,
        runtime_identity_sha256="d" * 64,
    )
    np.testing.assert_array_equal(
        row,
        np.asarray(payload["design_row_angstrom2_div_1000"], dtype=float),
    )


def test_feature_record_rejects_fabricated_row_even_after_rehash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "water-000.json"
    payload = _feature(path)
    fabricated = np.linspace(0.1, 1.8, 18)
    payload["design_row_angstrom2_div_1000"] = fabricated.tolist()
    payload["stock_smd_control_kcal_mol"] = float(
        fabricated @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    )
    _rehash_feature(payload)
    with pytest.raises(
        AGGREGATOR.W1FeatureAggregationError,
        match="exactly reproduce",
    ):
        AGGREGATOR._validate_feature_record(
            payload=payload,
            path=path,
            water_ordinal=0,
            selection_index=4,
            item=_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.__setitem__("experimental_targets_emitted", True),
        lambda payload: payload.__setitem__(
            "design_parameter_names",
            list(reversed(SMD_WATER_TENSION_PARAMETER_NAMES)),
        ),
        lambda payload: payload.__setitem__("unregistered_field", 1),
    ),
)
def test_feature_record_schema_drift_fails_after_self_rehash(
    tmp_path: Path,
    mutation,
) -> None:
    path = tmp_path / "water-000.json"
    payload = _feature(path)
    mutation(payload)
    _rehash_feature(payload)
    with pytest.raises(AGGREGATOR.W1FeatureAggregationError):
        AGGREGATOR._validate_feature_record(
            payload=payload,
            path=path,
            water_ordinal=0,
            selection_index=4,
            item=_item(),
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def test_matrix_diagnostics_reports_rank_support_without_targets(monkeypatch) -> None:
    monkeypatch.setattr(AGGREGATOR, "EXPECTED_WATER_RECORD_COUNT", 20)
    matrix = np.zeros((20, 18), dtype=float)
    matrix[:, :5] = np.arange(100, dtype=float).reshape(20, 5) + np.eye(20, 5)
    diagnostic = AGGREGATOR._matrix_diagnostics(matrix)
    assert diagnostic["shape"] == [20, 18]
    assert 1 <= diagnostic["rank"] <= 5
    assert diagnostic["condition_number_if_full_rank"] is None
    assert diagnostic["nonzero_record_count_by_column"][5:] == [0] * 13
    assert "Target-free" in diagnostic["claim_boundary"]


class _Snapshot:
    root = ROOT
    head = "1" * 40
    tree = "2" * 40

    def assert_unchanged(self) -> None:
        return None


def _source_hashes() -> dict[str, str]:
    return {
        relative: sha256_file(ROOT / relative)
        for relative in GENERATOR.REQUIRED_SOURCE_FILE_NAMES
    }


def _preregistration_payload(
    *,
    input_root: Path,
    records_dir: Path,
    preregistration_path: Path,
    parent_path: Path,
    water: tuple[tuple[int, object], ...],
    water_identity_sha256: str,
) -> dict[str, object]:
    runtime_identity, runtime_identity_sha256 = GENERATOR._runtime_identity()
    source_hashes = _source_hashes()
    maximum_active = GENERATOR._maximum_active_pair_factors(water)
    finite_product_degree = (
        GENERATOR.POSITIVE_BERNSTEIN_PAIR_DEGREE * maximum_active
        + 2 * GENERATOR.AREA_SURFACE_LMAX
    )
    payload: dict[str, object] = {
        "artifact": GENERATOR.PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-positive-parent-feature-evaluation",
        "locked_at_utc": "2026-08-16T00:00:00+00:00",
        "partition": "development-water-only",
        "water_record_count": len(water),
        "water_identity_sha256": water_identity_sha256,
        "dataset_loader_parses_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read_by_feature_generator": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_permitted": False,
        "checkpoint_or_method_selection_permitted": False,
        "source_root": str(ROOT),
        "source_git_head": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            text=True,
        ).strip(),
        "source_git_tree": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"),
            text=True,
        ).strip(),
        "source_files_sha256": source_hashes,
        "feature_generator_sha256": source_hashes[
            "tools/route2_release/generate_maple_cds_w1_features.py"
        ],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "input_root": str(input_root),
        "input_files_sha256": {
            name: sha256_file(input_root / name) for name in GENERATOR.INPUT_FILE_NAMES
        },
        "parent_hybrid_preregistration_path": str(parent_path),
        "parent_hybrid_preregistration_sha256": sha256_file(parent_path),
        "parent_hybrid_source_git_head": "3" * 40,
        "feature_output_dir": str(records_dir),
        "preregistration_path": str(preregistration_path),
        "area_definition": {
            "contract_id": (
                GENERATOR.POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID
            ),
            "exposure_contract_id": (GENERATOR.POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID),
            "profile_id": GENERATOR.MAPLE_CDS_W1_POSITIVE_BERNSTEIN_PROFILE_ID,
            "radii": "published-smd-sasa-radii-including-0.4-A-probe",
            "transition_width_angstrom2": (GENERATOR.AREA_TRANSITION_WIDTH_ANGSTROM2),
            "surface_lmax": GENERATOR.AREA_SURFACE_LMAX,
            "retained_parent_moment_lmax": 2 * GENERATOR.AREA_SURFACE_LMAX,
            "positive_parent_pair_degree": (GENERATOR.POSITIVE_BERNSTEIN_PAIR_DEGREE),
            "maximum_transition_factors": (
                GENERATOR.POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
            ),
            "maximum_integrand_degree": (
                GENERATOR.POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
            ),
            "dtype": GENERATOR.AREA_DTYPE,
            "device": GENERATOR.AREA_DEVICE,
            "area_measure": "a_i^2-integral-e_i-domega",
            "reconstructed_low_band_used_as_mask": False,
        },
        "linear_basis": {
            "contract": "published-aqueous-smd-linear-18-column-v1",
            "parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
            "stock_coefficients_cal_mol_angstrom2": (
                SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
            ),
            "stock_coefficients_sha256": canonical_json_sha256(
                SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
            ),
        },
        "numerical_choice_rationale": {
            "maximum_observed_active_pair_factors": maximum_active,
            "finite_product_degree_formula": (
                "pair_degree*active_pair_factors+2*surface_lmax"
            ),
            "positive_parent_pair_degree": GENERATOR.POSITIVE_BERNSTEIN_PAIR_DEGREE,
            "maximum_integrand_degree": finite_product_degree,
            "maximum_transition_factor_cap": (
                GENERATOR.POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
            ),
            "implementation_degree_limit": (
                GENERATOR.POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
            ),
            "selected_from_geometry_only": True,
            "accuracy_results_used": False,
        },
        "output_contract": {
            "one_exclusive_json_per_water_record": True,
            "coordinates_emitted": False,
            "experimental_targets_emitted": False,
            "hybrid_outputs_emitted": False,
            "fit_or_calibration_emitted": False,
        },
        "claim_boundary": "Synthetic target-blind feature preregistration.",
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    return payload


def _prepare_aggregate_case(tmp_path: Path, monkeypatch):
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    for index, name in enumerate(GENERATOR.INPUT_FILE_NAMES):
        (input_root / name).write_bytes(f"input-{index}".encode())
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    parent_path = tmp_path / "parent.json"
    parent = {
        "artifact_id": "route2-hybrid-smd-development-prereg-v3",
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": 505,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
        "source_git_head": "3" * 40,
        "input_root": str(input_root),
        "dataset_zip_sha256": sha256_file(input_root / "MNSolDatabase_v2012.zip"),
        "development_selection_sha256": sha256_file(
            input_root / "route2-mnsol-development-selection-v1.private.json"
        ),
        "pilot_selection_sha256": sha256_file(
            input_root / "route2-mnsol-pilot-selection-v1.json"
        ),
        "protocol_sha256": sha256_file(input_root / "route2-mnsol-protocol-v1.json"),
    }
    _write_read_only(parent_path, parent)

    items = (_item(0), _item(1))
    water = ((4, items[0]), (9, items[1]))
    water_identity_sha256 = "e" * 64
    preregistration_path = tmp_path / "preregistration.json"
    preregistration = _preregistration_payload(
        input_root=input_root,
        records_dir=records_dir,
        preregistration_path=preregistration_path,
        parent_path=parent_path,
        water=water,
        water_identity_sha256=water_identity_sha256,
    )
    _write_read_only(preregistration_path, preregistration)
    preregistration_sha256 = sha256_file(preregistration_path)
    for ordinal, (selection_index, item) in enumerate(water):
        _feature(
            records_dir / f"water-{ordinal:03d}.json",
            water_ordinal=ordinal,
            selection_index=selection_index,
            item=item,
            preregistration_sha256=preregistration_sha256,
            runtime_identity_sha256=preregistration["runtime_identity_sha256"],
        )

    monkeypatch.setattr(AGGREGATOR, "EXPECTED_WATER_RECORD_COUNT", len(water))
    monkeypatch.setattr(
        AGGREGATOR,
        "RepositorySnapshot",
        SimpleNamespace(capture=lambda _root: _Snapshot()),
    )
    monkeypatch.setattr(
        AGGREGATOR,
        "_load_selection",
        lambda **_kwargs: (object(), items),
    )
    monkeypatch.setattr(
        AGGREGATOR,
        "_water_records",
        lambda _rows: (water, water_identity_sha256),
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
    output = tmp_path / "aggregate.json"
    args = argparse.Namespace(
        source_root=ROOT,
        input_root=input_root,
        preregistration=preregistration_path,
        records_dir=records_dir,
        output=output,
    )
    return SimpleNamespace(
        args=args,
        input_root=input_root,
        records_dir=records_dir,
        parent_path=parent_path,
        preregistration_path=preregistration_path,
        preregistration=preregistration,
        output=output,
    )


def test_aggregate_closes_complete_matrix_hashes_and_no_target_claims(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    result = AGGREGATOR.aggregate(case.args)
    stored = json.loads(case.output.read_text())
    assert stored == result
    assert not case.output.stat().st_mode & 0o222
    assert result["record_count"] == 2
    assert result["experimental_targets_used_by_feature_aggregation"] is False
    assert result["experimental_targets_emitted"] is False
    assert result["hybrid_prediction_records_read"] is False
    assert result["confirmation_selection_manifest_opened"] is False
    without_digest = dict(result)
    assert without_digest.pop("aggregate_sha256") == canonical_json_sha256(
        without_digest
    )
    matrix = [record["design_row_angstrom2_div_1000"] for record in result["records"]]
    assert result["matrix_sha256"] == canonical_json_sha256(matrix)


def test_aggregate_rejects_output_collision_without_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    case.output.write_text("preexisting\n")
    with pytest.raises(FileExistsError):
        AGGREGATOR.aggregate(case.args)
    assert case.output.read_text() == "preexisting\n"


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_aggregate_requires_exact_record_filename_closure(
    tmp_path: Path,
    monkeypatch,
    change: str,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    if change == "missing":
        (case.records_dir / "water-001.json").unlink()
    else:
        (case.records_dir / "unexpected.json").write_text("{}\n")
    with pytest.raises(AGGREGATOR.W1FeatureAggregationError, match="incomplete"):
        AGGREGATOR.aggregate(case.args)


def test_aggregate_rejects_tampered_row_with_recomputed_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    path = case.records_dir / "water-001.json"
    payload = json.loads(path.read_text())
    row = np.asarray(payload["design_row_angstrom2_div_1000"], dtype=float)
    row[0] += 1.0
    payload["design_row_angstrom2_div_1000"] = row.tolist()
    payload["stock_smd_control_kcal_mol"] = float(
        row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2
    )
    _rehash_feature(payload)
    _rewrite_read_only(path, payload)
    with pytest.raises(
        AGGREGATOR.W1FeatureAggregationError,
        match="exactly reproduce",
    ):
        AGGREGATOR.aggregate(case.args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["area_definition"].__setitem__("surface_lmax", 3),
            "area definition",
        ),
        (
            lambda payload: payload.__setitem__("experimental_targets_emitted", True),
            "experimental_targets_emitted",
        ),
        (
            lambda payload: payload.__setitem__("runtime_identity_sha256", "0" * 64),
            "runtime digest",
        ),
    ),
)
def test_aggregate_rejects_rehashed_preregistration_contract_drift(
    tmp_path: Path,
    monkeypatch,
    mutation,
    message: str,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    payload = json.loads(case.preregistration_path.read_text())
    mutation(payload)
    payload.pop("self_sha256")
    payload["self_sha256"] = canonical_json_sha256(payload)
    _rewrite_read_only(case.preregistration_path, payload)
    with pytest.raises(AGGREGATOR.W1FeatureAggregationError, match=message):
        AGGREGATOR.aggregate(case.args)


def test_aggregate_rejects_input_drift_after_preregistration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    (case.input_root / GENERATOR.INPUT_FILE_NAMES[0]).write_bytes(b"drift")
    with pytest.raises(AGGREGATOR.W1FeatureAggregationError, match="input file"):
        AGGREGATOR.aggregate(case.args)


def test_aggregate_rejects_rebound_parent_with_wrong_input_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _prepare_aggregate_case(tmp_path, monkeypatch)
    parent = json.loads(case.parent_path.read_text())
    parent["dataset_zip_sha256"] = "0" * 64
    _rewrite_read_only(case.parent_path, parent)
    preregistration = json.loads(case.preregistration_path.read_text())
    preregistration["parent_hybrid_preregistration_sha256"] = sha256_file(
        case.parent_path
    )
    preregistration.pop("self_sha256")
    preregistration["self_sha256"] = canonical_json_sha256(preregistration)
    _rewrite_read_only(case.preregistration_path, preregistration)
    with pytest.raises(
        AGGREGATOR.W1FeatureAggregationError,
        match="parent input hash",
    ):
        AGGREGATOR.aggregate(case.args)


def test_aggregator_help_is_available_without_private_inputs() -> None:
    result = subprocess.run(
        (sys.executable, str(AGGREGATOR_PATH), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--records-dir" in result.stdout
    assert "--preregistration" in result.stdout
