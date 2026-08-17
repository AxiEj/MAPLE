from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT / "tools" / "route2_release"
RUNNERS = {
    "water_loop": RUNNER_DIR / "run_aimnet2_geometry_mediated_water_loop.py",
    "hvp": RUNNER_DIR / "run_aimnet2_geometry_mediated_hvp.py",
    "frequency": RUNNER_DIR / "run_aimnet2_geometry_mediated_frequency.py",
}
OLD_ARTIFACT_KINDS = {
    "water_loop": (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-water-bidirectional-loop"
    ),
    "hvp": (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-water-complete-hvp"
    ),
    "frequency": (
        "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
        "smooth-harmonic-stationary-water-dense-hessian-frequency"
    ),
}
OLD_CLAIM_MARKERS = {
    "water_loop": "conductor-reference scalar",
    "hvp": "finite-dielectric or nonpolar validation",
    "frequency": "conductor-reference point-charge research scalar",
}


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(
        f"route2_aimnet2_daily_{name}_runner", RUNNERS[name]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RUNNER_DIR))
    return module


@pytest.mark.parametrize("name", tuple(RUNNERS))
def test_daily_runner_cli_preserves_old_default_and_accepts_water_ddpcm(
    name, monkeypatch
):
    runner = _load_runner(name)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNERS[name]), "--checkpoint", "aimnet2.pt", "--output", "out.json"],
    )
    assert runner._parse_args().continuum == "harmonic-point"
    if name == "water_loop":
        assert runner._parse_args().nonpolar == "none"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNERS[name]),
            "--checkpoint",
            "aimnet2.pt",
            "--continuum",
            "harmonic-ddpcm-water",
            "--output",
            "out.json",
        ],
    )
    assert runner._parse_args().continuum == "harmonic-ddpcm-water"

    help_result = subprocess.run(
        (sys.executable, str(RUNNERS[name]), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--continuum {harmonic-point,harmonic-ddpcm-water}" in help_result.stdout
    if name == "water_loop":
        assert "--nonpolar {none,pyscf-smd-cds-water}" in help_result.stdout


@pytest.mark.parametrize("name", tuple(RUNNERS))
def test_daily_runner_artifact_labels_keep_old_evidence_and_close_new_claims(name):
    runner = _load_runner(name)
    assert runner._artifact_kind("harmonic-point") == OLD_ARTIFACT_KINDS[name]
    old_claim_boundary = runner._claim_boundary("harmonic-point")
    assert OLD_CLAIM_MARKERS[name] in old_claim_boundary
    assert "G_np is identically zero" not in old_claim_boundary

    artifact_kind = runner._artifact_kind("harmonic-ddpcm-water")
    claim_boundary = runner._claim_boundary("harmonic-ddpcm-water")
    assert "frozen-charge-water" in artifact_kind
    assert "harmonic-ddpcm" in artifact_kind
    assert "water-bound frozen-charge finite-dielectric" in claim_boundary
    assert "evaluated once per geometry" in claim_boundary
    assert "receives no continuum field" in claim_boundary
    assert "no electronic SCF" in claim_boundary
    assert "G_np is identically zero" in claim_boundary
    assert "admission" in claim_boundary


def test_frequency_runner_bound_transform_keeps_every_root_trial_in_domain():
    runner = _load_runner("frequency")
    bounds = np.asarray(
        runner.AIMNET2_GEOMETRY_MEDIATED_STATIONARY_WATER_INTERNAL_BOUNDS,
        dtype=float,
    )
    initial = runner.aimnet2_geometry_mediated_stationary_water_initial_coordinates()
    target = np.array((1.02, 1.01, np.deg2rad(115.0)))
    observed = []

    def function(coordinates):
        observed.append(np.asarray(coordinates, dtype=float))
        return np.asarray(coordinates, dtype=float) - target

    solved = runner._solve_stationary_root(
        function,
        lambda coordinates: np.eye(3),
        initial,
        "harmonic-ddpcm-water",
    )
    assert solved.success is True
    np.testing.assert_allclose(solved.x, target, atol=1.0e-12, rtol=0.0)
    assert observed
    assert all(
        np.all(values > bounds[:, 0]) and np.all(values < bounds[:, 1])
        for values in observed
    )


def test_water_loop_runner_names_total_ddpcm_smdcds_scalar_without_admission():
    runner = _load_runner("water_loop")
    artifact = runner._artifact_kind("harmonic-ddpcm-water", "pyscf-smd-cds-water")
    boundary = runner._claim_boundary("harmonic-ddpcm-water", "pyscf-smd-cds-water")
    assert "pyscf-smdcds" in artifact
    assert "official PySCF 2.13.1 SMD-CDS scalar" in boundary
    assert "field-independent" in boundary
    assert "not electronically iterated" in boundary
    assert "not strict original-SMD electrostatic equivalence" in boundary
    assert "same-scalar HVP" in boundary
    assert "admission" in boundary
