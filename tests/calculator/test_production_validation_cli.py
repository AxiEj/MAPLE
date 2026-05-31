import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "production_validation.py"
_SPEC = importlib.util.spec_from_file_location("production_validation", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
_parse_model_options = _MODULE._parse_model_options
_apply_validation_model_defaults = _MODULE._apply_validation_model_defaults


def _head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _fake_context(*, dirty: bool = False) -> dict:
    return {
        "artifact_id": "md_acceptance_TEST",
        "generated_utc": "20260526T000000Z",
        "environment": {
            "maple_git_commit": _head_commit(),
            "maple_git_branch": "fix/pbc",
            "maple_git_dirty": dirty,
        },
    }


def _patch_passing_matrix(monkeypatch, *, dirty: bool = False):
    def fake_run_acceptance_matrix(
        calc_factory, thresholds, workdir, *, quick=False, validation_artifact_id=None,
        validation_scope="production_npt",
    ):
        assert validation_artifact_id == "md_acceptance_TEST"
        assert validation_scope in {"production_npt", "mic_compatibility"}
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "nve_md_manifest.json").write_text(
            json.dumps({"validation_artifact_id": validation_artifact_id}) + "\n"
        )
        (workdir / "nve_md_summary.txt").write_text("summary\n")
        (workdir / "nve_md_thermo.dat").write_text("# thermo\n")
        return [
            SimpleNamespace(
                name="barostat_clamp_free",
                status="pass",
                passed=True,
                metrics={"clamp_count": 0},
                detail="ok",
            )
        ]

    monkeypatch.setattr(_MODULE, "make_report_context", lambda thresholds: _fake_context(dirty=dirty))
    monkeypatch.setattr(_MODULE, "run_acceptance_matrix", fake_run_acceptance_matrix)


def test_model_option_parser_coerces_supported_scalar_types():
    opts = _parse_model_options([
        "dispersion=true",
        "cutoff=12.0",
        "ewald_accuracy=1e-6",
        "max_steps=25",
        "head=omat",
    ])
    assert opts == {
        "dispersion": True,
        "cutoff": 12.0,
        "ewald_accuracy": 1e-6,
        "max_steps": 25,
        "head": "omat",
    }


def test_model_option_parser_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate --model-option key: cutoff"):
        _parse_model_options(["cutoff=12.0", "cutoff=10.0"])


def test_validation_defaults_use_float64_for_macepol_precision_gates():
    assert _apply_validation_model_defaults("macepol-pbc-small", {})["default_dtype"] == "float64"
    assert _apply_validation_model_defaults(
        "macepol-pbc-small", {"default_dtype": "float32"}
    )["default_dtype"] == "float32"
    assert _apply_validation_model_defaults("mace-mp-pbc-small", {}) == {}
    assert _apply_validation_model_defaults("aimnet2-pbc", {}) == {}


def test_help_text_preserves_single_target_gate_examples():
    completed = subprocess.run(
        ["python", str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "single-target production-validation gate" in completed.stdout
    assert "one passing report for every" in completed.stdout
    assert "validation/required_pbc_backends.toml" in completed.stdout
    assert "--validation-scope" in completed.stdout
    assert "python scripts/production_validation.py --model mace-mp-pbc-small --device cuda" in completed.stdout
    assert "python scripts/production_validation.py --model aimnet2-pbc --device cuda --model-option coulomb=ewald" in completed.stdout
    assert "python scripts/check_production_backend_matrix.py" in completed.stdout


def test_main_auto_artifact_dir_preserves_manifests_and_hashes(monkeypatch, tmp_path):
    _patch_passing_matrix(monkeypatch)

    outdir = tmp_path / "reports"
    assert _MODULE.main(["--outdir", str(outdir)]) == 0

    artifact_dir = outdir / "md_acceptance_TEST"
    run_workdir = artifact_dir / "runs"
    manifest = run_workdir / "nve_md_manifest.json"
    payload = json.loads((artifact_dir / "report.json").read_text())
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()

    assert manifest.exists()
    assert (run_workdir / "nve_md_summary.txt").exists()
    assert (run_workdir / "nve_md_thermo.dat").exists()
    assert payload["run_workdir"] == str(run_workdir)
    assert payload["manifest_files"] == [{"path": str(manifest), "sha256": manifest_hash}]
    assert payload["artifact_sha256"]["manifests"][str(manifest)] == manifest_hash
    assert payload["environment"]["maple_git_commit"] == _head_commit()
    assert payload["environment"]["maple_git_dirty"] is False
    assert payload["production_validated"] is True


def test_main_explicit_workdir_is_persistent_after_exit(monkeypatch, tmp_path):
    _patch_passing_matrix(monkeypatch)

    workdir = tmp_path / "explicit-runs"
    assert _MODULE.main(["--outdir", str(tmp_path / "reports"), "--workdir", str(workdir)]) == 0

    assert (workdir / "nve_md_manifest.json").exists()
    payload = json.loads((tmp_path / "reports" / "md_acceptance_TEST" / "report.json").read_text())
    assert payload["run_workdir"] == str(workdir)


def test_main_full_release_gate_rejects_dirty_environment(monkeypatch, tmp_path):
    _patch_passing_matrix(monkeypatch, dirty=True)

    assert _MODULE.main(["--outdir", str(tmp_path / "reports")]) == 1

    payload = json.loads((tmp_path / "reports" / "md_acceptance_TEST" / "report.json").read_text())
    assert payload["production_validated"] is False
    assert "environment.maple_git_dirty == false" in payload["release_gate"]["failed_criteria"]


def test_main_mic_compatibility_scope_is_not_production_validated(monkeypatch, tmp_path):
    _patch_passing_matrix(monkeypatch)

    outdir = tmp_path / "reports"
    assert _MODULE.main([
        "--outdir", str(outdir),
        "--validation-scope", "mic_compatibility",
    ]) == 0

    payload = json.loads((outdir / "md_acceptance_TEST" / "report.json").read_text())
    assert payload["validation_scope"] == "mic_compatibility"
    assert payload["validation_mode"] == "compatibility-validation"
    assert payload["compatibility_validated"] is True
    assert payload["production_validated"] is False
    assert payload["compatibility_gate"]["ready"] is True
    assert payload["release_gate"]["ready"] is False
