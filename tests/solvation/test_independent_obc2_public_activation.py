from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

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

    result = activation.audit_task("sp", tmp_path, expected_atom_count=1)

    assert result["passed"] is True
    assert result["provider"]["runtime_provider"] == "torch-obc2"
    assert result["provider"]["runtime_openmm_required"] is False


def test_task_audit_rejects_error_marker_even_with_readable_energy(tmp_path):
    (tmp_path / "sp.out").write_text(
        SP_OUTPUT + "ERROR: calculation failed late\n", encoding="utf-8"
    )
    _write_manifest(tmp_path, "sp")

    result = activation.audit_task("sp", tmp_path, expected_atom_count=1)

    assert result["passed"] is False
    assert result["fatal_markers"] == ["ERROR:"]


def test_task_audit_rejects_openmm_provider_provenance(tmp_path):
    (tmp_path / "sp.out").write_text(SP_OUTPUT, encoding="utf-8")
    _write_manifest(tmp_path, "sp", provider="openmm", openmm=True)

    result = activation.audit_task("sp", tmp_path, expected_atom_count=1)

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


@pytest.mark.parametrize("rows", [
    "  1 H bad 0.0 0.0\n  2 H 0.0 0.0 0.0\n",
    "  1 H 0.0 0.0\n  2 H 0.0 0.0 0.0\n",
    "  1 H nan 0.0 0.0\n  2 H 0.0 0.0 0.0\n",
    "  1 H 0.0 0.0 0.0\n",
    "  1 H 0.0 0.0 0.0\n  1 H 0.0 0.0 0.0\n",
    "  2 H 0.0 0.0 0.0\n  1 H 0.0 0.0 0.0\n",
])
def test_sp_audit_rejects_incomplete_or_malformed_gradient_rows(tmp_path, rows):
    text = SP_OUTPUT.replace("  1 H 0.0 0.0 0.0\n", rows)
    (tmp_path / "sp.out").write_text(text)
    _write_manifest(tmp_path, "sp")
    assert activation.audit_task("sp", tmp_path, expected_atom_count=2)["passed"] is False


def test_sp_audit_accepts_complete_finite_3n_block(tmp_path):
    (tmp_path / "sp.out").write_text(SP_OUTPUT.replace(
        "  1 H 0.0 0.0 0.0\n", "  1 H 0.0 0.0 0.0\n  2 H 1e-3 -2.0 0.0\n"))
    _write_manifest(tmp_path, "sp")
    result = activation.audit_task("sp", tmp_path, expected_atom_count=2)
    assert result["passed"] is True
    assert result["evidence"]["force_component_count"] == 6


def _opt_frame(index, x):
    return f"1\nImage {index}  Energy = -1.0\nH {x} 0 0\n"


def _write_opt(tmp_path, *, trajectory=True, moving=True):
    (tmp_path / "opt.out").write_text(
        "Iteration: 1\nIteration: 2\nLBFGS did NOT converge after 2 iterations.\n")
    _write_manifest(tmp_path, "opt")
    (tmp_path / "opt_opt.xyz").write_text(_opt_frame(0, 0.2 if moving else 0))
    if trajectory:
        (tmp_path / "opt_opt_traj.xyz").write_text(
            _opt_frame(0, 0) + _opt_frame(1, 0.1 if moving else 0)
            + _opt_frame(2, 0.2 if moving else 0))


def test_opt_audit_rejects_final_xyz_without_multistep_evidence(tmp_path):
    _write_opt(tmp_path, trajectory=False)
    with pytest.raises(ValueError, match="trajectory"):
        activation.audit_task("opt", tmp_path)


def test_opt_audit_retains_bounded_nonconvergence_with_real_steps(tmp_path):
    _write_opt(tmp_path)
    result = activation.audit_task("opt", tmp_path)
    assert result["passed"] is True
    assert result["evidence"]["trajectory_frame_count"] == 3
    assert result["evidence"]["iteration_numbers"] == [1, 2]
    assert result["evidence"]["bounded_nonconvergence_recorded"] is True


def test_opt_audit_rejects_duplicate_stationary_frames(tmp_path):
    _write_opt(tmp_path, moving=False)
    assert activation.audit_task("opt", tmp_path)["passed"] is False


def test_opt_audit_rejects_final_geometry_not_matching_last_step(tmp_path):
    _write_opt(tmp_path)
    (tmp_path / "opt_opt.xyz").write_text(_opt_frame(0, 0.9))
    assert activation.audit_task("opt", tmp_path)["passed"] is False



@pytest.mark.parametrize("record", ["H 0 0", "H nan 0 0", "H 0 0 0 extra"])
def test_xyz_parser_rejects_incomplete_or_nonfinite_coordinates(tmp_path, record):
    path = tmp_path / "bad.xyz"
    path.write_text(f"1\nImage 0 Energy = -1.0\n{record}\n")
    with pytest.raises(ValueError):
        activation._xyz_frames(path)


def test_opt_audit_rejects_truncated_atom_inventory(tmp_path):
    _write_opt(tmp_path)
    assert activation.audit_task("opt", tmp_path, expected_atom_count=2)["passed"] is False


def test_opt_input_enables_both_iteration_records(tmp_path):
    assert "max_iter=2,verbose=1" in activation.render_input("opt", tmp_path / "m.mol2", tmp_path / "a.pt")


def test_sp_audit_requires_expected_input_atom_count(tmp_path):
    (tmp_path / "sp.out").write_text(SP_OUTPUT)
    _write_manifest(tmp_path, "sp")
    assert activation.audit_task("sp", tmp_path)["passed"] is False
