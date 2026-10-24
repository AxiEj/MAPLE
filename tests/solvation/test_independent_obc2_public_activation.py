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


@pytest.fixture
def final_checkout(tmp_path, monkeypatch):
    """Synthetic qualification data exists only in a disposable test checkout."""
    qualification, runtime = activation._binding_helpers()
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(activation, "ROOT", root)
    monkeypatch.setattr(qualification, "ROOT", root)
    monkeypatch.setattr(runtime, "__file__", str(root / "maple/function/calculator/set_calculator.py"))
    files = {
        "maple/__init__.py": "",
        "maple/function/calculator/set_calculator.py": "# runtime\n",
        "maple/function/calculator/extra_correction/implicit/data/obc2_profile_manifest.json": "{}\n",
        "maple/function/calculator/extra_correction/implicit/data/OPENMM_CUSTOMGBFORCES_LICENSE.txt": "license\n",
        "tests/solvation/harness.py": "# harness\n",
        "docs/historical.json": "{\"historical\": true}\n",
        "inputs/molecule.mol2": "input\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Activation Test")
    git("add", ".")
    git("commit", "-qm", "candidate")
    sources = qualification.source_snapshot()["files_sha256"]
    receipt = {
        "schema_version": 1,
        "artifact_type": "maple-independent-obc2-analytic-qualification",
        "status": "qualified",
        "qualification_sources": sources,
        "installed_runtime_sources": runtime._installed_runtime_source_inventory(),
    }
    receipt["content_sha256"] = activation.hashlib.sha256(
        runtime._canonical_receipt_payload(receipt)).hexdigest()
    receipt_path = root / activation.QUALIFICATION_RECEIPT_REL
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt))
    pointer = root / activation.POINTER_REL
    pointer.write_text(
        f'QUALIFICATION_RECEIPT_FILENAME = "{receipt_path.name}"\n'
        f'QUALIFICATION_RECEIPT_SHA256 = "{activation.sha256(receipt_path)}"\n')
    git("add", ".")
    git("commit", "-qm", "data-only qualification")
    wheel = tmp_path / "maple.whl"

    def make_wheel(*, omit=None, change=None):
        with activation.zipfile.ZipFile(wheel, "w") as archive:
            expected = {**runtime._installed_runtime_source_inventory(),
                        activation.POINTER_REL: "", activation.QUALIFICATION_RECEIPT_REL: ""}
            for relative in expected:
                if relative != omit:
                    archive.writestr(relative, b"stale" if relative == change else (root / relative).read_bytes())
        return wheel

    return root, git, make_wheel, root / "activation.json"


def test_final_binding_covers_checkout_and_final_data_but_not_unshipped_history(final_checkout):
    root, git, wheel, receipt = final_checkout
    binding = activation.final_source_binding(wheel(), receipt)
    assert binding["source_before"]["git_head"] == git("rev-parse", "HEAD")
    assert "docs/historical.json" in binding["source_before"]["files_sha256"]
    assert "docs/historical.json" not in binding["expected_wheel_runtime"]
    for relative, field in ((activation.POINTER_REL, "final_pointer"),
                            (activation.QUALIFICATION_RECEIPT_REL, "qualification_receipt")):
        assert binding[field]["sha256"] == activation.sha256(root / relative)
        assert binding["expected_wheel_runtime"][relative] == binding[field]["sha256"]


@pytest.mark.parametrize("relative", [
    "maple/function/calculator/set_calculator.py",
    "maple/function/calculator/extra_correction/implicit/data/obc2_profile_manifest.json",
    "maple/function/calculator/extra_correction/implicit/data/OPENMM_CUSTOMGBFORCES_LICENSE.txt",
    "maple/function/calculator/_independent_obc2_qualified_cell.py",
    "maple/function/calculator/data/independent_obc2_analytic_qualification.json",
])
@pytest.mark.parametrize("defect", ["omit", "change"])
def test_final_binding_rejects_missing_or_mismatched_wheel_member(final_checkout, relative, defect):
    _, _, wheel, receipt = final_checkout
    with pytest.raises(ValueError, match="wheel"):
        activation.final_source_binding(wheel(**{defect: relative}), receipt)


@pytest.mark.parametrize("relative", [
    "maple/function/calculator/_independent_obc2_qualified_cell.py",
    "maple/function/calculator/data/independent_obc2_analytic_qualification.json",
])
def test_final_binding_rejects_stale_pointer_or_receipt(final_checkout, relative):
    root, _, wheel, receipt = final_checkout
    candidate = wheel()
    with (root / relative).open("a") as handle:
        handle.write('\n# stale\n')
    with pytest.raises((ValueError, SyntaxError)):
        activation.final_source_binding(candidate, receipt)


def test_final_binding_rejects_receipt_from_another_checkout(final_checkout):
    root, _, wheel, receipt = final_checkout
    (root / "docs/historical.json").write_text("changed source")
    with pytest.raises(ValueError, match="qualification source"):
        activation.final_source_binding(wheel(), receipt)


@pytest.mark.parametrize("change", ["bytes", "missing", "new", "head"])
def test_final_binding_rejects_source_drift_after_activation(final_checkout, change):
    root, git, wheel, receipt = final_checkout
    binding = activation.final_source_binding(wheel(), receipt)
    path = root / "docs/historical.json"
    if change == "bytes":
        path.write_text("changed")
    elif change == "missing":
        path.unlink()
    elif change == "new":
        (root / "tests/new.py").write_text("# new\n")
    else:
        git("commit", "--allow-empty", "-qm", "new HEAD")
    with pytest.raises((ValueError, FileNotFoundError), match="source|drift"):
        activation.verify_final_source_binding(binding, receipt)


def test_final_binding_accepts_unchanged_source_and_excludes_only_activation_receipt(final_checkout):
    root, _, wheel, receipt = final_checkout
    receipt.write_text("old receipt")
    subprocess.run(["git", "add", receipt.name], cwd=root, check=True)
    binding = activation.final_source_binding(wheel(), receipt)
    receipt.write_text("new activation receipt")
    activation.verify_final_source_binding(binding, receipt)
    assert receipt.name not in binding["source_before"]["files_sha256"]
    assert binding["source_after"] == binding["source_before"]


def test_installed_binding_accepts_exact_runtime_from_purelib(final_checkout, tmp_path):
    _, _, wheel, receipt = final_checkout
    binding = activation.final_source_binding(wheel(), receipt)
    purelib = tmp_path / "site-packages"
    payload = {"runtime_file": str(purelib / "maple/function/calculator/set_calculator.py"),
               "runtime_sources": binding["expected_wheel_runtime"]}
    activation.verify_installed_binding(payload, purelib, binding)


@pytest.mark.parametrize("defect", ["origin", "missing", "mismatch", "extra"])
def test_installed_binding_rejects_wrong_origin_or_runtime_data(final_checkout, tmp_path, defect):
    root, _, wheel, receipt = final_checkout
    binding = activation.final_source_binding(wheel(), receipt)
    purelib = tmp_path / "site-packages"
    payload = {"runtime_file": str(purelib / "maple/function/calculator/set_calculator.py"),
               "runtime_sources": dict(binding["expected_wheel_runtime"])}
    if defect == "origin":
        payload["runtime_file"] = str(root / "maple/function/calculator/set_calculator.py")
    elif defect == "missing":
        payload["runtime_sources"].pop(activation.POINTER_REL)
    elif defect == "extra":
        payload["runtime_sources"]["maple/extra.py"] = "0" * 64
    else:
        payload["runtime_sources"][activation.QUALIFICATION_RECEIPT_REL] = "0" * 64
    with pytest.raises(ValueError, match="installed"):
        activation.verify_installed_binding(payload, purelib, binding)


@pytest.mark.parametrize("defect", ["unsealed", "missing_receipt", "stale_receipt_sha"])
def test_final_binding_rejects_unqualified_or_missing_final_data(final_checkout, defect):
    root, _, wheel, receipt = final_checkout
    candidate = wheel()
    if defect == "missing_receipt":
        (root / activation.QUALIFICATION_RECEIPT_REL).unlink()
    else:
        pointer = root / activation.POINTER_REL
        text = pointer.read_text()
        if defect == "unsealed":
            text = text.replace(f'"{activation.sha256(root / activation.QUALIFICATION_RECEIPT_REL)}"', "None")
        else:
            text = text.replace(activation.sha256(root / activation.QUALIFICATION_RECEIPT_REL), "0" * 64)
        pointer.write_text(text)
    with pytest.raises((ValueError, FileNotFoundError)):
        activation.final_source_binding(candidate, receipt)


def test_final_binding_rejects_wheel_drift_after_activation(final_checkout):
    _, _, wheel, receipt = final_checkout
    candidate = wheel()
    binding = activation.final_source_binding(candidate, receipt)
    with candidate.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ValueError, match="wheel drift"):
        activation.verify_final_source_binding(binding, receipt)


@pytest.mark.parametrize("relative", [
    "maple/function/calculator/_independent_obc2_qualified_cell.py",
    "maple/function/calculator/data/independent_obc2_analytic_qualification.json",
])
def test_final_binding_rejects_final_data_drift_after_activation(final_checkout, relative):
    root, _, wheel, receipt = final_checkout
    binding = activation.final_source_binding(wheel(), receipt)
    with (root / relative).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="source drift"):
        activation.verify_final_source_binding(binding, receipt)


def test_final_inventory_binds_ignored_checkout_inputs_before_and_after(final_checkout):
    root, _, wheel, receipt = final_checkout
    input_path = root / ".ignored-inputs/checkpoint.pt"
    input_path.parent.mkdir()
    input_path.write_bytes(b"frozen checkpoint")
    # Exclude through the local git info file, not a qualification source edit.
    (root / ".git/info/exclude").write_text(".ignored-inputs/\n")
    before = activation.final_source_snapshot(receipt, activation_inputs=(input_path,))
    binding = activation.final_source_binding(wheel(), receipt, source_before=before)
    assert before["files_sha256"][".ignored-inputs/checkpoint.pt"] == activation.sha256(input_path)
    input_path.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="source drift"):
        activation.verify_final_source_binding(binding, receipt)


@pytest.fixture
def prepared_main(final_checkout, monkeypatch):
    root, _, wheel, receipt = final_checkout
    checkpoint = root / "inputs/molecule.mol2"
    # Exercise orchestration only: no model loading, venv creation or installation.
    monkeypatch.setattr(activation, "ANI2X", checkpoint)
    monkeypatch.setattr(activation, "ANI2X_SHA256", activation.sha256(checkpoint))
    monkeypatch.setattr(activation, "RUN_ROOT", root)
    monkeypatch.setattr(activation, "sealed_fixture", lambda: {"records": [{
        "case": "methanol", "nonpolar": "ace", "geometry": "base", "atom_count": 1,
        "source_mol2_relative_path": "inputs/molecule.mol2",
        "source_mol2_sha256": activation.sha256(checkpoint),
    }]})
    monkeypatch.setattr(sys, "argv", [str(HARNESS), "--candidate-wheel", str(wheel()),
        "--receipt", str(receipt), "--runtime", str(root / "run"),
        "--venv", str(root / "venv"), "--wheelhouse", str(root / "wheelhouse")])
    (root / ".git/info/exclude").write_text("run/\nactivation.json\n")
    return root, receipt


def test_activation_failure_preserves_receipt_raw_logs_and_original_error(prepared_main, monkeypatch):
    root, receipt = prepared_main

    def fail(*args):
        activation.run_raw([sys.executable, "-c", "print('raw evidence');raise SystemExit(3)"],
                           root, activation.clean_environment(), root / "run/raw/install", 10)
        (root / "docs/historical.json").write_text("concurrent drift")
        raise RuntimeError("original install failure")

    monkeypatch.setattr(activation, "_activate", fail)
    with pytest.raises(RuntimeError, match="original install failure"):
        activation.main()
    result = json.loads(receipt.read_text())
    assert result["passed"] is False
    assert result["error"]["message"] == "original install failure"
    assert "source drift" in result["binding_error"]["message"]
    assert (root / "run/raw/install.stdout").read_text() == "raw evidence\n"
    assert (root / "run/raw/install.exitcode").read_text() == "3\n"


def test_activation_rejects_drift_even_when_task_runner_reports_success(prepared_main, monkeypatch):
    root, receipt = prepared_main

    def drift(*args):
        args[-1]["passed"] = True
        (root / "tests/solvation/harness.py").write_text("changed harness")

    monkeypatch.setattr(activation, "_activate", drift)
    with pytest.raises(ValueError, match="source drift"):
        activation.main()
    assert json.loads(receipt.read_text())["passed"] is False


def test_activation_checks_wheel_before_starting_installer(prepared_main, monkeypatch):
    _, receipt = prepared_main
    candidate = Path(sys.argv[sys.argv.index("--candidate-wheel") + 1])
    with activation.zipfile.ZipFile(candidate, "w") as archive:
        archive.writestr("maple/__init__.py", "stale")
    calls = []
    monkeypatch.setattr(activation, "_activate", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="wheel"):
        activation.main()
    assert calls == []
    assert json.loads(receipt.read_text())["failed_stage"] == "source-wheel-binding"


@pytest.mark.parametrize("render_only", [False, True])
def test_activation_refuses_to_erase_a_previous_failed_attempt(prepared_main, monkeypatch, render_only):
    root, receipt = prepared_main
    if render_only:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--render-only"])
    raw = root / "run/raw/install.stdout"
    raw.parent.mkdir(parents=True)
    raw.write_text("previous evidence")
    receipt.write_text('{"passed": false}')
    monkeypatch.setattr(activation, "_activate", lambda *args: pytest.fail("must not activate"))
    with pytest.raises(FileExistsError, match="activation"):
        activation.main()
    assert raw.read_text() == "previous evidence"
    assert receipt.read_text() == '{"passed": false}'


def test_installed_probe_hashes_actual_runtime_and_final_data(final_checkout, capsys):
    root, _, wheel, receipt = final_checkout
    binding = activation.final_source_binding(wheel(), receipt)
    exec("import json;" + activation.INSTALLED_BINDING_PROBE, {})
    payload = json.loads(capsys.readouterr().out)
    activation.verify_installed_binding(payload, root, binding)
    assert payload["runtime_sources"] == binding["expected_wheel_runtime"]


@pytest.mark.parametrize("defect", ["runtime", "sources", "seal"])
def test_final_binding_rejects_repointed_but_stale_qualification_receipt(final_checkout, defect):
    root, _, wheel, receipt = final_checkout
    _, runtime = activation._binding_helpers()
    path = root / activation.QUALIFICATION_RECEIPT_REL
    value = json.loads(path.read_text())
    if defect == "runtime":
        value["installed_runtime_sources"]["maple/__init__.py"] = "0" * 64
    elif defect == "sources":
        value["qualification_sources"]["docs/historical.json"] = "0" * 64
    else:
        value["status"] = "stale"
    if defect != "seal":
        value["content_sha256"] = activation.hashlib.sha256(
            runtime._canonical_receipt_payload(value)).hexdigest()
    path.write_text(json.dumps(value))
    (root / activation.POINTER_REL).write_text(
        f'QUALIFICATION_RECEIPT_FILENAME = "{path.name}"\n'
        f'QUALIFICATION_RECEIPT_SHA256 = "{activation.sha256(path)}"\n')
    with pytest.raises(ValueError, match="qualification"):
        activation.final_source_binding(wheel(), receipt)


def test_final_binding_rejects_source_changed_between_snapshot_and_wheel_check(final_checkout):
    root, _, wheel, receipt = final_checkout
    before = activation.final_source_snapshot(receipt)
    candidate = wheel()
    (root / "maple/function/calculator/set_calculator.py").write_text("changed runtime")
    with pytest.raises(ValueError, match="runtime inventory"):
        activation.final_source_binding(candidate, receipt, source_before=before)


def test_final_binding_rejects_unexpected_python_wheel_member(final_checkout):
    _, _, wheel, receipt = final_checkout
    candidate = wheel()
    with activation.zipfile.ZipFile(candidate, "a") as archive:
        archive.writestr("maple/unexpected.py", "# extra runtime\n")
    with pytest.raises(ValueError, match="wheel runtime member set"):
        activation.final_source_binding(candidate, receipt)


def test_render_only_never_requests_qualification_binding(prepared_main, monkeypatch, capsys):
    _, receipt = prepared_main
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--render-only"])
    monkeypatch.setattr(activation, "_binding_helpers",
                        lambda: pytest.fail("render-only must not load qualification helpers"))
    activation.main()
    result = json.loads(capsys.readouterr().out)
    assert "final_source_binding" not in result
    assert result["inputs"]
    assert receipt.is_file()
