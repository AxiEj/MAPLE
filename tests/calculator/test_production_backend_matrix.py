import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_production_backend_matrix.py"
_SPEC = importlib.util.spec_from_file_location("check_production_backend_matrix", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _head_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _matrix(path: Path, targets: str, required_classes: str | None = None) -> Path:
    classes = required_classes or '["barostat_clamp_free", "stress_finite_difference"]'
    path.write_text(
        "minimum_thresholds_version = \"1.4.0\"\n"
        f"required_classes = {classes}\n\n"
        + targets
    )
    return path


def _stress_result(*, full_voigt: bool = True, sign_ok: bool = True) -> dict:
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    if not full_voigt:
        names = names[:-1]
    return {
        "name": "stress_finite_difference",
        "status": "pass",
        "passed": True,
        "metrics": {
            "components": [
                {
                    "component": name,
                    "stress_ev_per_ang3": 0.001,
                    "passed": True,
                    "per_delta": [
                        {
                            "delta": 0.001,
                            "fd_ev_per_ang3": 0.001,
                            "stress_ev_per_ang3": 0.001,
                            "abs_error_ev_per_ang3": 0.0,
                            "rel_error": 0.0,
                            "log10_abs_ratio_error": 0.0,
                            "sign_ok": sign_ok,
                            "abs_ok": True,
                            "rel_ok": True,
                            "log_ok": True,
                            "ok": True,
                        }
                    ],
                }
                for name in names
            ]
        },
        "detail": "full Voigt stress FD [xx, yy, zz, yz, xz, xy]",
    }


def _report(
    path: Path,
    *,
    model: str,
    model_options: dict,
    validation_mode: str = "production-validation",
    thresholds_version: str = "1.4.0",
    production_validated: bool = True,
    dirty: bool = False,
    full_voigt: bool = True,
    sign_ok: bool = True,
) -> Path:
    contract = {
        "energy_unit": "Ha",
        "force_unit": "Ha/A",
        "stress_unit": "eV/A^3",
        "neighbor_cutoff_A": 6.0,
        "local_descriptor_cutoff_A": None,
        "long_range_method": "none",
    }
    if model.startswith("aimnet2"):
        method = model_options.get("coulomb")
        contract["long_range_method"] = method
        if method in {"ewald", "pme"}:
            contract["local_descriptor_cutoff_A"] = 5.0
        if method == "dsf":
            contract["long_range_coulomb_cutoff_A"] = 12.0
    results = [
        {
            "name": "barostat_clamp_free",
            "status": "pass",
            "passed": True,
            "metrics": {"clamp_count": 0},
            "detail": "zero clamps",
        },
        _stress_result(full_voigt=full_voigt, sign_ok=sign_ok),
    ]
    payload = {
        "artifact_id": "md_acceptance_TEST",
        "thresholds_version": thresholds_version,
        "calculator": model,
        "environment": {
            "maple_git_commit": _head_commit(),
            "maple_git_dirty": dirty,
        },
        "overall": "PASS" if production_validated else "FAIL",
        "summary": {
            "n_pass": len(results),
            "n_fail": 0 if production_validated else 1,
            "n_skip": 0,
            "barostat_clamp_count": 0,
        },
        "results": results,
        "validation_mode": validation_mode,
        "production_validated": production_validated,
        "release_gate": {"ready": production_validated},
        "validation_target": {
            "model": model,
            "model_options": model_options,
            "target": model,
        },
        "model_options": model_options,
        "calculator_contract": contract,
        "manifest_files": [{"path": "run/nve_md_manifest.json", "sha256": "0" * 64}],
        "validation_system": {"dynamics": {"n_atoms": 24}, "stress_finite_difference": {"n_atoms": 24}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_required_backend_matrix_declares_review_scope():
    targets, minimum, classes = _MODULE._load_matrix(
        Path(__file__).resolve().parents[2] / "validation" / "required_pbc_backends.toml"
    )
    ids = {target.id for target in targets}
    assert minimum == "1.7.1"
    assert "stress_finite_difference" in classes
    assert "npt_com_pressure_invariance" in classes
    assert ids == {
        "aimnet2-pbc-dsf", "aimnet2-pbc-ewald", "aimnet2-pbc-pme",
        "aimnet2nse-pbc-dsf", "aimnet2nse-pbc-ewald", "aimnet2nse-pbc-pme",
        "mace-mp-pbc-small", "mace-mp-pbc-medium", "mace-mp-pbc-large",
        "macepol-pbc-small", "macepol-pbc-medium", "macepol-pbc-large",
        "uma-omat-default",
    }
    by_id = {target.id: target for target in targets}
    assert by_id["aimnet2-pbc-dsf"].model_options == {"coulomb": "dsf", "cutoff": 5.0}
    assert by_id["aimnet2nse-pbc-dsf"].model_options == {"coulomb": "dsf", "cutoff": 5.0}


def test_backend_matrix_accepts_complete_current_non_smoke_reports(tmp_path):
    matrix = _matrix(
        tmp_path / "matrix.toml",
        """
[[targets]]
id = "aimnet2-pbc-pme"
model = "aimnet2-pbc"
model_options = { coulomb = "pme" }

[[targets]]
id = "macepol-pbc-small"
model = "macepol-pbc-small"
model_options = { default_dtype = "float64" }
""",
    )
    reports = tmp_path / "reports"
    _report(reports / "aimnet" / "report.json", model="aimnet2-pbc", model_options={"coulomb": "pme"})
    _report(
        reports / "macepol" / "report.json",
        model="macepol-pbc-small",
        model_options={"default_dtype": "float64"},
    )

    ok, messages = _MODULE.check_matrix(matrix_path=matrix, reports_dir=reports)

    assert ok, messages
    assert any("[PASS] aimnet2-pbc-pme" in message for message in messages)
    assert any("[PASS] macepol-pbc-small" in message for message in messages)


def test_backend_matrix_rejects_smoke_or_dirty_reports(tmp_path, capsys):
    matrix = _matrix(
        tmp_path / "matrix.toml",
        """
[[targets]]
id = "mace-mp-pbc-small"
model = "mace-mp-pbc-small"
model_options = {}
""",
    )
    reports = tmp_path / "reports"
    _report(
        reports / "report.json",
        model="mace-mp-pbc-small",
        model_options={},
        validation_mode="quick-smoke",
        thresholds_version="smoke-1.3.1",
        dirty=True,
    )

    rc = _MODULE.main(["--matrix", str(matrix), "--reports-dir", str(reports)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "production backend matrix: FAIL" in out
    assert "validation_mode is not production-validation" in out
    assert "thresholds_version 'smoke-1.3.1'" in out
    assert "environment.maple_git_dirty is not false" in out


def test_backend_matrix_rejects_incomplete_or_sign_wrong_stress_fd(tmp_path):
    matrix = _matrix(
        tmp_path / "matrix.toml",
        """
[[targets]]
id = "uma-omat-default"
model = "uma"
model_options = { task = "omat", size = "uma-s-1p1" }
""",
    )
    reports = tmp_path / "reports"
    _report(
        reports / "report.json",
        model="uma",
        model_options={"task": "omat", "size": "uma-s-1p1"},
        full_voigt=False,
        sign_ok=False,
    )

    ok, messages = _MODULE.check_matrix(matrix_path=matrix, reports_dir=reports)
    text = "\n".join(messages)

    assert not ok
    assert "full ASE Voigt order" in text
    assert "sign mismatch" in text


def test_backend_matrix_does_not_let_extra_model_options_satisfy_default_target(tmp_path):
    matrix = _matrix(
        tmp_path / "matrix.toml",
        """
[[targets]]
id = "mace-mp-pbc-small"
model = "mace-mp-pbc-small"
model_options = {}
""",
    )
    reports = tmp_path / "reports"
    _report(
        reports / "report.json",
        model="mace-mp-pbc-small",
        model_options={"dispersion": True},
    )

    ok, messages = _MODULE.check_matrix(matrix_path=matrix, reports_dir=reports)
    text = "\n".join(messages)

    assert not ok
    assert "no matching report" in text
