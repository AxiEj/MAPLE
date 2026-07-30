from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from maple.function.calculator.model_capabilities import load_model_provenance_card
from maple.function.solvfe import (
    C3NET_CHECKPOINT_SHA256,
    C3NET_SOLVENT_COUNT,
    C3NET_SOURCE_REVISION,
    C3NetConfigError,
    C3NetPropertyAdapter,
    C3NetRuntimeError,
)
from maple.function.solvfe.c3net_property import (
    _PREDICT_SCRIPT,
    _normalize_origin,
    _runtime_identity_payload,
    _verify_pinned_file,
)


def _adapter_without_upstream_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> C3NetPropertyAdapter:
    source_root = tmp_path / "C3Net"
    source_root.mkdir()
    monkeypatch.setattr(
        C3NetPropertyAdapter, "assert_source_identity", lambda self: None
    )
    return C3NetPropertyAdapter(source_root)


def test_property_card_is_not_an_ordinary_calculator_backend():
    card = load_model_provenance_card(
        "c3net", Path("maple/function/calculator/model_cards")
    )

    assert card.payload["source_revision"] == C3NET_SOURCE_REVISION
    assert card.capabilities.solvation_mode == "property_only"
    assert card.capabilities.energy is False
    assert card.capabilities.forces is False
    assert card.capabilities.supports_absolute_solvation is False
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("sp")


def test_prediction_subprocess_uses_deterministic_inference_mode():
    assert "random.seed(0)" in _PREDICT_SCRIPT
    assert "np.random.seed(0)" in _PREDICT_SCRIPT
    assert "torch.manual_seed(0)" in _PREDICT_SCRIPT
    assert "torch.set_num_threads(1)" in _PREDICT_SCRIPT
    assert "torch.use_deterministic_algorithms(True)" in _PREDICT_SCRIPT
    assert 'torch.set_float32_matmul_precision("highest")' in _PREDICT_SCRIPT
    assert "dataset.npzs = sorted(" in _PREDICT_SCRIPT
    assert "v2000_declared_stereo_identity" in _PREDICT_SCRIPT
    assert "declared_stereo_identity" in _PREDICT_SCRIPT
    assert "model.eval()" in _PREDICT_SCRIPT
    assert "with torch.inference_mode():" in _PREDICT_SCRIPT


def test_prediction_subprocess_preserves_raw_tensor_precision():
    assert 'output["y"].detach().cpu()' in _PREDICT_SCRIPT
    assert "prediction.txt" not in _PREDICT_SCRIPT
    assert "%5.3f" not in _PREDICT_SCRIPT


def test_prediction_subprocess_pins_deterministic_cpu_environment(
    tmp_path, monkeypatch
):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args,
            0,
            '{"value": 1.0}\n',
            "",
        )

    monkeypatch.setattr(subprocess, "run", run)

    assert adapter._run_script("print('{}')") == {"value": 1.0}
    assert {
        key: captured[key]
        for key in (
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
    } == {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }


def test_formal_runtime_manifest_matches_and_detects_dependency_drift(tmp_path):
    manifest = {
        **_runtime_identity_payload(),
        "required_controls": {
            "deterministic_algorithms": True,
            "interop_threads": 1,
            "matmul_precision": "highest",
            "numpy_random_seed": 0,
            "python_random_seed": 0,
            "torch_random_seed": 0,
            "torch_threads": 1,
        },
    }
    path = tmp_path / "runtime-lock.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    adapter = object.__new__(C3NetPropertyAdapter)
    adapter.runtime_manifest_path = path

    adapter.assert_runtime_identity()

    manifest["packages"]["numpy"] = "0.0-forged"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(C3NetConfigError, match="differs from"):
        adapter.assert_runtime_identity()


def test_predict_labels_the_scalar_result_as_property_only(tmp_path, monkeypatch):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    sdf = tmp_path / "solute.sdf"
    sdf.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "_run_script",
        lambda script, *arguments: {
            "prediction_kcal_mol": -21.389,
            "solvent": "water",
            "solvent_id": 102,
            "numpy_int_compatibility_shim": True,
        },
    )

    result = adapter.predict(sdf, "Water")

    assert result.predicted_solvation_free_energy_kcal_mol == pytest.approx(-21.389)
    assert result.unit == "kcal/mol"
    assert result.target_quantity == "property_prediction"
    assert result.property_only is True
    assert result.absolute_solvation_backend is False
    assert result.numpy_int_compatibility_shim is True


def test_property_heads_are_rejected_before_prediction(tmp_path, monkeypatch):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    sdf = tmp_path / "solute.sdf"
    sdf.write_text("placeholder", encoding="utf-8")

    with pytest.raises(C3NetConfigError, match="not solvation-free-energy solvents"):
        adapter.predict(sdf, "logp")


def test_conformer_prediction_uses_the_paper_mean_without_becoming_a_pes(
    tmp_path, monkeypatch
):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    sdf = tmp_path / "conformers.sdf"
    sdf.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "_run_script",
        lambda script, *arguments: {
            "predictions_kcal_mol": [-5.0, -4.0, -6.0],
            "solvent": "methanol",
            "solvent_id": 65,
            "numpy_int_compatibility_shim": False,
        },
    )

    result = adapter.predict_conformers(sdf, "methanol")

    assert result.predicted_solvation_free_energy_kcal_mol == pytest.approx(-5.0)
    assert result.component_predictions_kcal_mol == (-5.0, -4.0, -6.0)
    assert result.conformer_count == 3
    assert "arithmetic mean" in result.conformer_policy
    assert result.property_only is True
    assert result.absolute_solvation_backend is False


def test_available_solvents_requires_the_pinned_103_solvent_registry(
    tmp_path, monkeypatch
):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    solvents = ["water", *[f"solvent-{index}" for index in range(102)]]
    monkeypatch.setattr(
        adapter, "_run_script", lambda script, *arguments: {"solvents": solvents}
    )

    assert adapter.available_solvents() == tuple(solvents)
    assert len(adapter.available_solvents()) == C3NET_SOLVENT_COUNT

    monkeypatch.setattr(
        adapter, "_run_script", lambda script, *arguments: {"solvents": ["water"]}
    )
    with pytest.raises(C3NetRuntimeError, match="expected 103-solvent registry"):
        adapter.available_solvents()


def test_pinned_artifact_hash_mismatch_fails_closed(tmp_path):
    artifact = tmp_path / "checkpoint-1.pth.tar"
    artifact.write_bytes(b"not the public C3Net checkpoint")

    with pytest.raises(C3NetConfigError, match="SHA256 mismatch"):
        _verify_pinned_file(artifact, C3NET_CHECKPOINT_SHA256, "prediction checkpoint")


def test_source_identity_requires_the_official_clean_checkout(tmp_path, monkeypatch):
    source_root = tmp_path / "C3Net"
    source_root.mkdir()
    seen_artifacts: list[tuple[Path, str, str]] = []

    def git_output(_root, *args):
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): C3NET_SOURCE_REVISION,
            (
                "config",
                "--get",
                "remote.origin.url",
            ): "git@github.com:SehanLee/C3Net.git",
            ("status", "--porcelain", "--untracked-files=all"): "",
        }
        return values[args]

    monkeypatch.setattr("maple.function.solvfe.c3net_property._git_output", git_output)
    monkeypatch.setattr(
        "maple.function.solvfe.c3net_property._git_result",
        lambda _root, *args: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(
        "maple.function.solvfe.c3net_property._verify_pinned_file",
        lambda path, digest, label: seen_artifacts.append((path, digest, label)),
    )

    adapter = C3NetPropertyAdapter(source_root)

    assert adapter.source_root == source_root.resolve()
    assert [entry[2] for entry in seen_artifacts] == [
        "prediction checkpoint",
        "embedding artifact",
    ]
    assert _normalize_origin("git@github.com:SehanLee/C3Net.git") == (
        "https://github.com/sehanlee/c3net"
    )


def test_source_identity_rejects_tracked_modifications(tmp_path, monkeypatch):
    source_root = tmp_path / "C3Net"
    source_root.mkdir()

    def git_output(_root, *args):
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): C3NET_SOURCE_REVISION,
            (
                "config",
                "--get",
                "remote.origin.url",
            ): "https://github.com/SehanLee/C3Net",
            ("status", "--porcelain", "--untracked-files=all"): "",
        }
        return values[args]

    monkeypatch.setattr("maple.function.solvfe.c3net_property._git_output", git_output)
    monkeypatch.setattr(
        "maple.function.solvfe.c3net_property._git_result",
        lambda _root, *args: subprocess.CompletedProcess(args, 1, "", ""),
    )

    with pytest.raises(C3NetConfigError, match="tracked modifications"):
        C3NetPropertyAdapter(source_root)


def test_source_identity_rejects_untracked_import_overrides(tmp_path, monkeypatch):
    source_root = tmp_path / "C3Net"
    source_root.mkdir()

    def git_output(_root, *args):
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): C3NET_SOURCE_REVISION,
            (
                "config",
                "--get",
                "remote.origin.url",
            ): "https://github.com/SehanLee/C3Net",
            ("status", "--porcelain", "--untracked-files=all"): "?? prediction/json.py",
        }
        return values[args]

    monkeypatch.setattr("maple.function.solvfe.c3net_property._git_output", git_output)

    with pytest.raises(C3NetConfigError, match="tracked or untracked modifications"):
        C3NetPropertyAdapter(source_root)
