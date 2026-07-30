from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from maple.function.calculator.model_capabilities import load_model_provenance_card
from maple.function.solvfe import (
    CIGIN_CHECKPOINT_SHA256,
    CIGIN_SOURCE_REVISION,
    CIGINConfigError,
    CIGINPropertyAdapter,
    CIGINRuntimeError,
)
from maple.function.solvfe.cigin_property import _normalize_origin, _verify_pinned_file


def _adapter_without_upstream_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CIGINPropertyAdapter:
    source_root = tmp_path / "CIGIN"
    source_root.mkdir()
    monkeypatch.setattr(
        CIGINPropertyAdapter, "assert_source_identity", lambda self: None
    )
    return CIGINPropertyAdapter(source_root)


def test_property_card_is_not_an_ordinary_calculator_backend():
    card = load_model_provenance_card(
        "cigin", Path("maple/function/calculator/model_cards")
    )

    assert card.payload["source_revision"] == CIGIN_SOURCE_REVISION
    assert card.capabilities.solvation_mode == "property_only"
    assert card.capabilities.energy is False
    assert card.capabilities.forces is False
    assert card.capabilities.supports_absolute_solvation is False
    with pytest.raises(ValueError, match="explicitly forbids"):
        card.validate_task("sp")


def test_predict_canonicalizes_and_labels_the_scalar_result(tmp_path, monkeypatch):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_run_script",
        lambda script, *arguments: {
            "prediction_kcal_mol": -7.81,
            "solute_smiles": "NCCO",
            "solvent_smiles": "CO",
            "execution_device": "cpu",
        },
    )

    result = adapter.predict("OCCN", "CO")

    assert result.predicted_solvation_free_energy_kcal_mol == pytest.approx(-7.81)
    assert result.solute_smiles == "NCCO"
    assert result.solvent_smiles == "CO"
    assert result.unit == "kcal/mol"
    assert result.target_quantity == "property_prediction"
    assert result.property_only is True
    assert result.absolute_solvation_backend is False
    assert result.execution_device == "cpu"
    assert result.cuda_disabled_for_upstream_device_consistency is True


@pytest.mark.parametrize(
    ("solute", "solvent", "message"),
    [
        ("", "O", "must be non-empty"),
        ("C[NH3+]", "O", "restricted to neutral"),
        ("[CH3]", "O", "does not accept radical"),
        ("CC.O", "O", "does not accept disconnected"),
        ("CB", "O", "does not support solute element"),
    ],
)
def test_invalid_smiles_domains_fail_before_upstream_prediction(
    tmp_path, monkeypatch, solute, solvent, message
):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_run_script",
        lambda *_arguments: pytest.fail("upstream CIGIN runtime should not start"),
    )

    with pytest.raises(CIGINConfigError, match=message):
        adapter.predict(solute, solvent)


def test_upstream_cpu_runtime_environment_is_explicit_and_source_safe(
    tmp_path, monkeypatch
):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            args[0], 0, '{"prediction_kcal_mol": -1}\n', ""
        )

    monkeypatch.setattr("maple.function.solvfe.cigin_property.subprocess.run", fake_run)
    assert adapter._run_script("print('unused')") == {"prediction_kcal_mol": -1}

    environment = captured["env"]
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPATH"].split(":")[0] == str(
        adapter.source_root / "scripts"
    )
    assert environment["LD_LIBRARY_PATH"].split(":")[0] == str(Path(sys.prefix) / "lib")
    assert captured["cwd"] == adapter.source_root


def test_non_cpu_child_payload_is_rejected(tmp_path, monkeypatch):
    adapter = _adapter_without_upstream_checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_run_script",
        lambda script, *arguments: {
            "prediction_kcal_mol": -7.81,
            "solute_smiles": "NCCO",
            "solvent_smiles": "CO",
            "execution_device": "cuda",
        },
    )

    with pytest.raises(CIGINRuntimeError, match="mismatched identity or device"):
        adapter.predict("NCCO", "CO")


def test_pinned_artifact_hash_mismatch_fails_closed(tmp_path):
    artifact = tmp_path / "cigin.tar"
    artifact.write_bytes(b"not the public CIGIN checkpoint")

    with pytest.raises(CIGINConfigError, match="SHA256 mismatch"):
        _verify_pinned_file(artifact, CIGIN_CHECKPOINT_SHA256, "prediction checkpoint")


def test_source_identity_requires_the_official_clean_checkout(tmp_path, monkeypatch):
    source_root = tmp_path / "CIGIN"
    source_root.mkdir()
    seen_artifacts: list[tuple[Path, str, str]] = []

    def git_output(_root, *args):
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): CIGIN_SOURCE_REVISION,
            (
                "config",
                "--get",
                "remote.origin.url",
            ): "git@github.com:devalab/CIGIN.git",
            ("status", "--porcelain", "--untracked-files=all"): "",
        }
        return values[args]

    monkeypatch.setattr("maple.function.solvfe.cigin_property._git_output", git_output)
    monkeypatch.setattr(
        "maple.function.solvfe.cigin_property._git_result",
        lambda _root, *args: subprocess.CompletedProcess(args, 0, "", ""),
    )
    monkeypatch.setattr(
        "maple.function.solvfe.cigin_property._verify_pinned_file",
        lambda path, digest, label: seen_artifacts.append((path, digest, label)),
    )

    adapter = CIGINPropertyAdapter(source_root)

    assert adapter.source_root == source_root.resolve()
    assert [entry[2] for entry in seen_artifacts] == ["prediction checkpoint"]
    assert _normalize_origin("git@github.com:devalab/CIGIN.git") == (
        "https://github.com/devalab/cigin"
    )


def test_source_identity_rejects_untracked_import_overrides(tmp_path, monkeypatch):
    source_root = tmp_path / "CIGIN"
    source_root.mkdir()

    def git_output(_root, *args):
        values = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "HEAD"): CIGIN_SOURCE_REVISION,
            (
                "config",
                "--get",
                "remote.origin.url",
            ): "https://github.com/devalab/CIGIN",
            ("status", "--porcelain", "--untracked-files=all"): "?? scripts/models.pyc",
        }
        return values[args]

    monkeypatch.setattr("maple.function.solvfe.cigin_property._git_output", git_output)

    with pytest.raises(CIGINConfigError, match="tracked or untracked modifications"):
        CIGINPropertyAdapter(source_root)
