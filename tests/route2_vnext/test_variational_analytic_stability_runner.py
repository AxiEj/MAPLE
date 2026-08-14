from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "tools"
    / "route2_release"
    / "run_variational_analytic_stability_water_canary.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "route2_variational_analytic_stability_runner", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_is_checkpoint_free_and_all_capabilities_remain_closed():
    result = subprocess.run(
        (sys.executable, str(RUNNER), "--help"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--device" in result.stdout
    assert "--output" in result.stdout
    assert "does not admit Route-2 E/F/H/V/M" in result.stdout

    text = RUNNER.read_text(encoding="utf-8")
    assert "NO_CAPABILITIES" in text
    assert '"global_passivity_proven": False' in text
    assert '"global_root_uniqueness_proven": False' in text
    assert '"public_force_admitted": False' in text
    assert '"tier_v_admitted": False' in text


def test_protocol_thresholds_and_five_start_vectors_are_preregistered():
    runner = _load_runner()
    assert runner.SCHEMA_VERSION.endswith("-v1")
    assert runner.RANDOM_SEEDS == (20260816, 20260817)
    assert runner.ROOT_OPTIONS.tolerance == pytest.approx(2.0e-10)
    assert runner.ROOT_OPTIONS.max_iterations == 120
    assert runner.ROOT_OPTIONS.damping == pytest.approx(0.5)
    assert runner.ROOT_OPTIONS.history == 6
    assert runner.LINEARIZATION_RELATIVE_TOLERANCE == pytest.approx(1.0e-10)
    assert runner.STABILITY_THRESHOLDS.as_dict() == {
        "symmetry_relative": 1.0e-10,
        "curvature_sign_relative": 1.0e-8,
        "absolute_eigenvalue": 1.0e-12,
        "model_invertibility_relative": 1.0e-6,
        "feedback_imaginary_relative": 1.0e-10,
        "feedback_spectral_radius_maximum": 0.95,
        "residual_minimum_singular_relative": 1.0e-3,
        "combined_positive_relative": 1.0e-6,
    }

    starts = runner._starts(23)
    assert tuple(starts) == (
        "zero",
        "axis_positive_0p25",
        "random_20260816_positive_0p25",
        "random_20260816_negative_0p25",
        "random_20260817_positive_0p50",
    )
    expected = {
        "zero": (
            0.0,
            "0ec1bf45e06394e7a37ad49ef51b79a21fc9e73c6f9ee3eef71f1e3802815945",
        ),
        "axis_positive_0p25": (
            0.25,
            "9ade009efd7c18c9b0c22983518079a113270d06b2706f3911123266eb0f38e9",
        ),
        "random_20260816_positive_0p25": (
            0.25,
            "044f0d2565227d79980383fef9113880364233ce0a262e033017dd98e43a18c7",
        ),
        "random_20260816_negative_0p25": (
            0.25,
            "4a021d53c966e9dace54e41d65af1c4cbf86f60cb8156a2f724be555321fe875",
        ),
        "random_20260817_positive_0p50": (
            0.50,
            "d16dd13baa7a00957500509582d841c083558703b80f7b7e0d6797bb520527e5",
        ),
    }
    for label, vector in starts.items():
        norm, digest = expected[label]
        assert float(np.linalg.norm(vector)) == pytest.approx(norm, abs=1.0e-15)
        assert runner.canonical_json_sha256(vector.tolist()) == digest
    assert len({vector.tobytes() for vector in starts.values()}) == 5


def test_measurement_hash_input_structurally_excludes_wall_time():
    runner = _load_runner()
    records = {
        name: {"kind": name}
        for name in (
            "protocol",
            "identity",
            "geometry",
            "multi_start",
            "stability",
            "decision",
        )
    }
    measured = runner._measurement_record(**records)

    assert measured == records
    assert "runtime_seconds" not in measured
    assert "solve_timings_seconds" not in measured
    assert "derivative_timings_seconds" not in measured


def test_harmonic_configuration_matches_the_preceding_common_scalar_canary():
    runner = _load_runner()
    assert runner.SURFACE_LMAX == 1
    assert runner.EXPOSURE_LMAX == 2
    assert runner.TRANSITION_WIDTH_ANGSTROM2 == pytest.approx(0.18)
    assert runner.EXPOSURE_RADIAL_QUADRATURE_ORDER == 32
    assert runner.SOURCE_RADIAL_QUADRATURE_ORDER == 32
    assert runner.GREEN_RADIAL_QUADRATURE_ORDER == 32
    assert (
        "maple/solvation/models/runtime/analytic_gaussian_multipole.py"
        in runner.REQUIRED_SOURCE_PATHS
    )
    assert (
        "maple/solvation/release/variational_stability.py"
        in runner.REQUIRED_SOURCE_PATHS
    )
