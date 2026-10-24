from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests/solvation/run_independent_obc2_public_activation.py"
SPEC = importlib.util.spec_from_file_location("obc2_public_activation", HARNESS)
activation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(activation)
SP_OUTPUT = """Energy: -1.25 Hartree
Gradients (Hartree/Angstrom):
  Gradient = -Force
  Atom  El        Gx              Gy              Gz
  1 H 0.0 0.0 0.0

"""


def _write_manifest(cwd: Path, task: str, *, provider="torch-obc2", openmm=False):
    directory = cwd / f"{task}.out.implicit"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "solvation": {
                    "provider": provider,
                    "runtime_provider": provider,
                    "runtime_openmm_required": openmm,
                }
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_activation_declares_fresh_no_openmm_console_script_boundary():
    completed = subprocess.run(
        [sys.executable, str(HARNESS), "--describe"], cwd=ROOT,
        text=True, capture_output=True, check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    description = json.loads(completed.stdout)
    assert description["stage"] == "installed activation, not candidate qualification"
    assert description["console_script_only"] is True
    assert description["network_allowed"] is False
    assert description["openmm_allowed"] is False
    assert description["pythonpath_allowed"] is False
    assert description["system_site_packages"] is False
    assert description["task_order"] == list(activation.TASK_ORDER)
    assert description["post_opt_replay"] == "opt_final_sp"
    for key in ("venv", "runtime", "receipt"):
        Path(description[key]).resolve().relative_to(activation.RUN_ROOT.resolve())


def test_activation_matrix_has_every_required_public_task_once():
    assert activation.TASK_ORDER == (
        "sp", "opt", "scan", "nve", "nvt", "analytic_freq", "prfo",
        "direct_hvp_dimer", "numerical_freq", "force_difference_dimer",
    )


def test_rendered_inputs_use_default_provider_without_qualification_bypass(tmp_path):
    checkpoint = tmp_path / "ani2x.pt"
    mol2 = tmp_path / "water.mol2"
    for task in activation.TASK_ORDER:
        rendered = activation.render_input(task, mol2, checkpoint)
        assert "provider=" not in rendered
        assert "qualification" not in rendered.lower()
        assert "#device=cpu" in rendered
        assert "model=obc2" in rendered
        assert "profile=obc2-mbondi2" in rendered
        assert "nonpolar=ace" in rendered
    assert "hessian=analytic" in activation.render_input("analytic_freq", mol2, checkpoint)
    assert "use_hvp=true" in activation.render_input("direct_hvp_dimer", mol2, checkpoint)
    assert "hessian=numerical" in activation.render_input("numerical_freq", mol2, checkpoint)
    assert "use_hvp=false" in activation.render_input("force_difference_dimer", mol2, checkpoint)


def test_render_only_preserves_water_and_nonwater_input_identity(tmp_path):
    receipt = activation.RUN_ROOT / "pytest-render-only.json"
    runtime = activation.RUN_ROOT / "pytest-render-runtime"
    try:
        completed = subprocess.run(
            [sys.executable, str(HARNESS), "--render-only", "--receipt", str(receipt),
             "--runtime", str(runtime), "--venv", str(activation.RUN_ROOT / "unused-venv")],
            cwd=ROOT, text=True, capture_output=True, check=False, timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert {row["molecule"] for row in result["inputs"].values()} == {"water", "methanol"}
        assert result["inputs"]["analytic_freq"]["molecule_sha256"] == activation.WATER_MOL2_SHA256
        fixture = activation.sealed_fixture()
        methanol = next(row for row in fixture["records"] if (row["case"], row["nonpolar"], row["geometry"]) == ("methanol", "ace", "base"))
        assert result["inputs"]["sp"]["molecule_sha256"] == methanol["source_mol2_sha256"]
    finally:
        receipt.unlink(missing_ok=True)
        shutil.rmtree(runtime, ignore_errors=True)


def test_task_audit_requires_finite_output_and_independent_provider(tmp_path):
    (tmp_path / "sp.out").write_text(SP_OUTPUT, encoding="utf-8")
    _write_manifest(tmp_path, "sp")

    result = activation.audit_task("sp", tmp_path)

    assert result["passed"] is True
    assert result["provider"]["runtime_provider"] == "torch-obc2"
    assert result["provider"]["runtime_openmm_required"] is False


def test_task_audit_rejects_error_marker_even_with_readable_energy(tmp_path):
    (tmp_path / "sp.out").write_text(
        SP_OUTPUT + "ERROR: calculation failed late\n", encoding="utf-8"
    )
    _write_manifest(tmp_path, "sp")

    result = activation.audit_task("sp", tmp_path)

    assert result["passed"] is False
    assert result["fatal_markers"] == ["ERROR:"]


def test_task_audit_rejects_openmm_provider_provenance(tmp_path):
    (tmp_path / "sp.out").write_text(SP_OUTPUT, encoding="utf-8")
    _write_manifest(tmp_path, "sp", provider="openmm", openmm=True)

    result = activation.audit_task("sp", tmp_path)

    assert result["passed"] is False
    assert result["provider"]["passed"] is False


def test_analytic_task_requires_successful_derivative_use_audit(tmp_path):
    (tmp_path / "analytic_freq.out").write_text(
        "Frequency analysis completed\n", encoding="utf-8"
    )
    directory = _write_manifest(tmp_path, "analytic_freq")
    (directory / "derivative-use.json").write_text(
        json.dumps(
            {
                "status": "success",
                "derivative": "hessian",
                "runtime_provider": "torch-obc2",
            }
        ),
        encoding="utf-8",
    )

    result = activation.audit_task("analytic_freq", tmp_path)

    assert result["passed"] is True
    assert result["derivative_use"]["derivative"] == "hessian"


def test_scan_artifact_requires_at_least_three_frames(tmp_path):
    (tmp_path / "scan.out").write_text(
        "Scan completed! Total points: 3\n", encoding="utf-8"
    )
    _write_manifest(tmp_path, "scan")
    (tmp_path / "scan_scan_final.xyz").write_text(
        "1\nScanning combination 1/2: [1.0] Energy = -1.0\nH 0 0 0\n"
        "1\nScanning combination 2/2: [1.1] Energy = -0.9\nH 0 0 0\n",
        encoding="utf-8",
    )
    scan = activation.audit_task("scan", tmp_path)
    assert scan["passed"] is False
    assert scan["evidence"]["frame_count"] == 2


def test_md_artifacts_require_all_scheduled_steps_and_frames(tmp_path):
    (tmp_path / "nve.out").write_text(
        "NVE simulation completed successfully.\n", encoding="utf-8"
    )
    _write_manifest(tmp_path, "nve")
    (tmp_path / "nve_md_thermo.dat").write_text(
        "1 0.05 298.0 0.1 -1.1 -1.0\n", encoding="utf-8"
    )
    (tmp_path / "nve_md_traj.xyz").write_text(
        "1\nstep 1\nH 0 0 0\n", encoding="utf-8"
    )
    md = activation.audit_task("nve", tmp_path)
    assert md["passed"] is False
    assert md["evidence"]["thermo_steps"] == [1]
