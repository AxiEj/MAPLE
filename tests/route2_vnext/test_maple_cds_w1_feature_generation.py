from __future__ import annotations

import hashlib
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

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "route2_release"
GENERATOR_PATH = TOOLS / "generate_maple_cds_w1_features.py"
CREATOR_PATH = TOOLS / "create_maple_cds_w1_feature_preregistration.py"


def _load_generator():
    specification = importlib.util.spec_from_file_location(
        "generate_maple_cds_w1_features",
        GENERATOR_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


class _TargetTrap:
    @property
    def delta_g_kcal_mol(self):  # pragma: no cover - access is a test failure
        raise AssertionError("experimental target was accessed")


def _water_item(*, solvent: str = "water", partition: str = "development"):
    geometry = SimpleNamespace(
        atomic_numbers=(8, 1, 1),
        coordinates_angstrom=np.asarray(
            (
                (0.000000, 0.000000, 0.000000),
                (0.957200, 0.000000, 0.000000),
                (-0.239987, 0.927297, 0.000000),
            ),
            dtype=float,
        ),
        sha256="a" * 64,
    )
    eligible = SimpleNamespace(
        partition=partition,
        geometry=geometry,
        record=_TargetTrap(),
    )
    return SimpleNamespace(
        canonical_solvent=solvent,
        eligible_record=eligible,
        opaque_record_id="b" * 64,
    )


def test_feature_payload_is_target_blind_and_closes_stock_linear_control() -> None:
    payload = GENERATOR._feature_payload(
        water_ordinal=2,
        selection_index=7,
        item=_water_item(),
        preregistration_sha256="c" * 64,
        runtime_identity_sha256="d" * 64,
    )

    row = np.asarray(payload["design_row_angstrom2_div_1000"], dtype=float)
    assert row.shape == (18,)
    assert np.all(np.isfinite(row))
    assert payload["design_parameter_names"] == list(SMD_WATER_TENSION_PARAMETER_NAMES)
    expected_stock = float(row @ SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2)
    assert payload["stock_smd_control_kcal_mol"] == pytest.approx(
        expected_stock,
        abs=2.0e-14,
    )
    assert payload["dataset_loader_parsed_experimental_targets"] is True
    assert payload["experimental_targets_used_by_feature_computation"] is False
    assert payload["experimental_targets_emitted"] is False
    assert payload["hybrid_prediction_records_read"] is False
    assert payload["fitting_or_calibration_performed"] is False
    assert not GENERATOR.PROHIBITED_OUTPUT_KEYS.intersection(payload)
    without_digest = dict(payload)
    observed = without_digest.pop("feature_sha256")
    assert observed == GENERATOR._canonical_sha256(without_digest)


@pytest.mark.parametrize(
    ("item", "message"),
    (
        (_water_item(solvent="methanol"), "water records"),
        (_water_item(partition="confirmation"), "development records"),
    ),
)
def test_feature_payload_rejects_wrong_solvent_or_partition(item, message) -> None:
    with pytest.raises(GENERATOR.W1FeatureGenerationError, match=message):
        GENERATOR._feature_payload(
            water_ordinal=0,
            selection_index=0,
            item=item,
            preregistration_sha256="c" * 64,
            runtime_identity_sha256="d" * 64,
        )


def _preregistration_payload(
    *,
    tmp_path: Path,
    parent: Path,
    input_root: Path,
) -> tuple[Path, dict[str, object]]:
    for name in GENERATOR.INPUT_FILE_NAMES:
        (input_root / name).write_text(name, encoding="utf-8")
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in GENERATOR.REQUIRED_SOURCE_FILE_NAMES
    }
    runtime_identity, runtime_identity_sha256 = GENERATOR._runtime_identity()
    payload: dict[str, object] = {
        "artifact": GENERATOR.PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-first-feature-evaluation",
        "partition": "development-water-only",
        "water_record_count": GENERATOR.EXPECTED_WATER_RECORD_COUNT,
        "dataset_loader_parses_experimental_targets": True,
        "experimental_targets_used_by_feature_computation": False,
        "experimental_targets_emitted": False,
        "hybrid_prediction_records_read_by_feature_generator": False,
        "confirmation_selection_manifest_opened": False,
        "confirmation_records_selected_or_emitted": False,
        "fitting_or_calibration_permitted": False,
        "checkpoint_or_method_selection_permitted": False,
        "source_root": str(ROOT),
        "source_git_head": "d" * 40,
        "source_git_tree": "e" * 40,
        "feature_generator_sha256": source_hashes[
            GENERATOR_PATH.relative_to(ROOT).as_posix()
        ],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "input_root": str(input_root),
        "input_files_sha256": {
            name: hashlib.sha256((input_root / name).read_bytes()).hexdigest()
            for name in GENERATOR.INPUT_FILE_NAMES
        },
        "source_files_sha256": source_hashes,
        "parent_hybrid_preregistration_path": str(parent),
        "parent_hybrid_preregistration_sha256": hashlib.sha256(
            parent.read_bytes()
        ).hexdigest(),
        "parent_hybrid_source_git_head": "f" * 40,
        "feature_output_dir": str(tmp_path / "features"),
        "area_definition": {
            "contract_id": GENERATOR.SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID,
            "profile_id": GENERATOR.MAPLE_CDS_W1_HARMONIC_LINEAR_PROFILE_ID,
            "radii": "published-smd-sasa-radii-including-0.4-A-probe",
            "transition_width_angstrom2": GENERATOR.AREA_TRANSITION_WIDTH_ANGSTROM2,
            "exposure_lmax": GENERATOR.AREA_EXPOSURE_LMAX,
            "radial_quadrature_order": GENERATOR.AREA_RADIAL_QUADRATURE_ORDER,
            "dtype": GENERATOR.AREA_DTYPE,
            "device": GENERATOR.AREA_DEVICE,
            "area_measure": "a_i^2-integral-e_i-domega",
        },
        "linear_basis": {
            "contract": "published-aqueous-smd-linear-18-column-v1",
            "parameter_names": list(SMD_WATER_TENSION_PARAMETER_NAMES),
            "stock_coefficients_cal_mol_angstrom2": (
                SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
            ),
            "stock_coefficients_sha256": GENERATOR._canonical_sha256(
                SMD_WATER_STOCK_TENSION_COEFFICIENTS_CAL_MOL_ANGSTROM2.tolist()
            ),
        },
        "output_contract": {
            "one_exclusive_json_per_water_record": True,
            "coordinates_emitted": False,
            "experimental_targets_emitted": False,
            "hybrid_outputs_emitted": False,
            "fit_or_calibration_emitted": False,
        },
    }
    path = tmp_path / "preregistration.json"
    payload["preregistration_path"] = str(path)
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path, payload


def test_preregistration_validation_binds_head_inputs_sources_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "artifact_id": "route2-hybrid-smd-development-prereg-v3",
                "schema_version": 3,
                "status": "locked-before-first-v3-hybrid-evaluation",
                "partition": "development",
                "record_count": 505,
                "confirmation_partition_opened": False,
                "fitting_or_calibration_permitted": False,
                "source_git_head": "f" * 40,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    path, payload = _preregistration_payload(
        tmp_path=tmp_path,
        parent=parent,
        input_root=input_root,
    )

    def fake_git(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "d" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "e" * 40
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(GENERATOR, "_git", fake_git)
    GENERATOR._validate_preregistration(
        preregistration=payload,
        preregistration_path=path,
        source_root=ROOT,
        input_root=input_root,
    )

    (input_root / GENERATOR.INPUT_FILE_NAMES[0]).write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(GENERATOR.W1FeatureGenerationError, match="Input file"):
        GENERATOR._validate_preregistration(
            preregistration=payload,
            preregistration_path=path,
            source_root=ROOT,
            input_root=input_root,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.__setitem__("schema_version", 2), "schema_version"),
        (
            lambda payload: payload["output_contract"].__setitem__(
                "coordinates_emitted", True
            ),
            "output contract",
        ),
        (
            lambda payload: payload["runtime_identity"].__setitem__(
                "python", "different-runtime"
            ),
            "numerical runtime",
        ),
    ),
)
def test_preregistration_rejects_schema_output_or_runtime_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps(
            {
                "artifact_id": "route2-hybrid-smd-development-prereg-v3",
                "schema_version": 3,
                "status": "locked-before-first-v3-hybrid-evaluation",
                "partition": "development",
                "record_count": 505,
                "confirmation_partition_opened": False,
                "fitting_or_calibration_permitted": False,
                "source_git_head": "f" * 40,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    path, original = _preregistration_payload(
        tmp_path=tmp_path,
        parent=parent,
        input_root=input_root,
    )
    payload = json.loads(json.dumps(original))
    mutation(payload)
    payload.pop("self_sha256", None)
    payload["self_sha256"] = GENERATOR._canonical_sha256(payload)

    def fake_git(_root: Path, *arguments: str) -> str:
        values = {
            ("rev-parse", "HEAD"): "d" * 40,
            ("rev-parse", "HEAD^{tree}"): "e" * 40,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        return values[arguments]

    monkeypatch.setattr(GENERATOR, "_git", fake_git)
    with pytest.raises(GENERATOR.W1FeatureGenerationError, match=message):
        GENERATOR._validate_preregistration(
            preregistration=payload,
            preregistration_path=path,
            source_root=ROOT,
            input_root=input_root,
        )


def test_relative_file_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-w1.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(GENERATOR.W1FeatureGenerationError, match="inside"):
        GENERATOR._relative_file(tmp_path, "../outside-w1.txt", name="source")


def test_creator_rejects_a_different_source_checkout(tmp_path: Path) -> None:
    sys.path.insert(0, str(TOOLS))
    try:
        import create_maple_cds_w1_feature_preregistration as creator
    finally:
        sys.path.pop(0)
    with pytest.raises(
        creator.W1FeaturePreregistrationError,
        match="own source checkout",
    ):
        creator.create(SimpleNamespace(source_root=tmp_path))


def test_exclusive_feature_writer_makes_record_read_only(tmp_path: Path) -> None:
    path = tmp_path / "feature.json"
    GENERATOR._write_json_exclusive(path, {"status": "pass"})
    assert json.loads(path.read_text()) == {"status": "pass"}
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        GENERATOR._write_json_exclusive(path, {"status": "overwrite"})


@pytest.mark.parametrize("command", (GENERATOR_PATH, CREATOR_PATH))
def test_w1_feature_tools_expose_dependency_light_help(command: Path) -> None:
    result = subprocess.run(
        (sys.executable, str(command), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-root" in result.stdout
    assert "--input-root" in result.stdout


def test_generator_source_does_not_access_target_or_hybrid_record_attributes() -> None:
    text = GENERATOR_PATH.read_text(encoding="utf-8")
    assert ".delta_g_kcal_mol" not in text
    assert 'hybrid_prediction_records_read": True' not in text
    assert 'experimental_targets_used_by_feature_computation": True' not in text
