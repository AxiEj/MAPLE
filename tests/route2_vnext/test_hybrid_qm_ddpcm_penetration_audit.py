from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/route2_release/run_hybrid_qm_ddpcm_penetration_audit.py"
SPEC = importlib.util.spec_from_file_location("hybrid_penetration_audit", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frozen_design_and_cell_matrix_are_exact() -> None:
    inputs = MODULE.load_frozen_inputs(ROOT)
    assert MODULE.sha256_file(inputs.design_path) == MODULE.DESIGN_SHA256
    assert MODULE.sha256_file(inputs.profile_path) == MODULE.PROFILE_SHA256
    assert [plan.key for plan in MODULE.basis_plans(inputs.design)] == [
        "compact",
        "baseline",
        "B1",
        "B2",
    ]
    assert MODULE.radius_scales(inputs.design) == (
        1.5,
        1.3,
        1.2,
        1.1,
        1.05,
        1.025,
        1.0,
    )
    assert MODULE.continuum_discretizations(inputs.design) == (
        (15, 1202),
        (17, 1730),
        (19, 2030),
    )
    cells = MODULE.planned_cells(inputs.design)
    assert len(cells) == 28
    assert cells[0] == {"basis_key": "compact", "radius_scale": 1.5}
    assert cells[-1] == {"basis_key": "B2", "radius_scale": 1.0}


def test_design_and_profile_hashes_fail_closed(tmp_path: Path) -> None:
    design = ROOT / MODULE.DESIGN_RELATIVE_PATH
    profile = ROOT / json.loads(design.read_text())["geometry"]["profile_input_path"]
    fake_root = tmp_path / "repo"
    fake_design = fake_root / MODULE.DESIGN_RELATIVE_PATH
    fake_profile = fake_root / profile.relative_to(ROOT)
    fake_design.parent.mkdir(parents=True)
    fake_profile.parent.mkdir(parents=True)
    fake_design.write_bytes(design.read_bytes())
    fake_profile.write_bytes(profile.read_bytes())
    MODULE.load_frozen_inputs(fake_root)
    fake_profile.write_text(fake_profile.read_text() + "\n")
    with pytest.raises(MODULE.AuditContractError, match="Profile input bytes changed"):
        MODULE.load_frozen_inputs(fake_root)


def test_augmented_basis_text_adds_exact_frozen_shells() -> None:
    inputs = MODULE.load_frozen_inputs(ROOT)
    plans = {plan.key: plan for plan in MODULE.basis_plans(inputs.design)}
    baseline = """spherical\n****\nH 0\nS 1 1.0\n  1.0 1.0\n****\nC 0\nS 1 1.0\n  2.0 1.0\n****\n"""
    text = MODULE.augmented_basis_text(baseline, plans["B1"])
    assert text.count("H     0") == 1
    assert text.count("C     0") == 1
    for element in ("H", "C"):
        for angular in ("s", "p"):
            for exponent in plans["B1"].added_exponents[element][angular]:
                assert f"{exponent:.17g}" in text
    assert text.count("P   1   1.00") == 2
    nested = MODULE.augmented_basis_text(baseline, plans["B2"])
    assert nested.count("P   1   1.00") == 4
    for element in ("H", "C"):
        for angular in ("s", "p"):
            assert len(plans["B2"].added_exponents[element][angular]) == 2
    with pytest.raises(MODULE.AuditContractError, match="Element C is absent"):
        MODULE.augmented_basis_text(baseline.replace("C 0", "N 0"), plans["B1"])


def test_electron_count_and_operator_metrics() -> None:
    overlap = np.diag([1.0, 4.0])
    density = np.diag([2.0, 0.5])
    assert MODULE.electron_count(density, overlap) == pytest.approx(4.0)
    operator = np.diag([2.0, 8.0])
    metrics = MODULE.orthonormal_operator_metrics(operator, overlap, cutoff=1.0e-12)
    assert metrics["retained_rank"] == 2
    assert metrics["frobenius"] == pytest.approx(np.sqrt(8.0))
    assert metrics["spectral"] == pytest.approx(2.0)
    with pytest.raises(MODULE.AuditContractError, match="Overlap must be symmetric"):
        MODULE.orthonormal_operator_metrics(
            operator, np.array([[1.0, 1.0], [0.0, 1.0]]), cutoff=1.0e-12
        )


def test_fixed_density_refinement_metrics_apply_frozen_q18_gates() -> None:
    overlap = np.diag([1.0, 4.0])
    base = np.diag([2.0, 8.0])
    refined = np.diag([2.0 + 5.0e-5, 8.0])
    result = MODULE.fixed_density_refinement_metrics(
        base_energy_hartree=-0.01,
        base_operator=base,
        refined_energy_hartree=-0.0100005,
        refined_operator=refined,
        overlap=overlap,
        cutoff=1.0e-12,
    )
    assert result["energy_absolute_difference_hartree"] == pytest.approx(5.0e-7)
    assert result["operator_relative_difference"] == pytest.approx(
        5.0e-5 / (2.0 + 5.0e-5)
    )
    assert result["energy_gate_pass"] is True
    assert result["operator_gate_pass"] is True

    failed = MODULE.fixed_density_refinement_metrics(
        base_energy_hartree=-0.01,
        base_operator=base,
        refined_energy_hartree=-0.010002,
        refined_operator=np.diag([2.001, 8.0]),
        overlap=overlap,
        cutoff=1.0e-12,
    )
    assert failed["energy_gate_pass"] is False
    assert failed["operator_gate_pass"] is False


def test_outlying_charge_union_of_spheres() -> None:
    density = np.array([1.0, 2.0, 3.0])
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    weights = np.array([0.5, 0.25, 0.1])
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([0.5, 0.5])
    result = MODULE.outlying_charge_from_points(
        density, points, weights, centers, radii
    )
    assert result["grid_electron_count"] == pytest.approx(1.3)
    assert result["outlying_charge_electron"] == pytest.approx(0.3)
    assert result["outlying_fraction"] == pytest.approx(0.3 / 1.3)


def test_validate_cli_is_optional_runtime_free_and_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "validate.json"
    code = (
        "import builtins,runpy,sys;"
        "real=builtins.__import__;"
        "builtins.__import__=lambda n,*a,**k: "
        "(_ for _ in ()).throw(ImportError(n)) if n.split('.')[0] in {'psi4','pyddx'} else real(n,*a,**k);"
        f"sys.argv={[str(RUNNER), '--repository', str(ROOT), 'validate', '--output', str(output)]!r};"
        f"runpy.run_path({str(RUNNER)!r},run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["measurement"]["planned_cell_count"] == 28
    assert payload["claim_boundary"]["experimental_solvation_targets_read"] is False
    assert payload["claim_boundary"]["terminal_orbital_stability_classified"] is False
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--repository",
            str(ROOT),
            "validate",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert output.is_file()
