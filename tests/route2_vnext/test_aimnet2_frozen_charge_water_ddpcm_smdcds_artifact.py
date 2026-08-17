from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_water_loop,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-water-ddpcm-smdcds-loop-51b4eff4"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "51b4eff47ca95fc114f420466b4a7f0e250e9c50"
EXECUTION_TREE = "1244222d4275b4c47ae4ee41dafedb4ee472c9d7"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
MEASUREMENT_SHA256 = "2bba943688afccc89a16b012eb6763437a835af6ccd060266d9382bb73397aaf"
SCALAR_ID = (
    "route2-candidate-aimnet2-frozen-charge-water-"
    "smoothharmonicgalerkin-ddpcm-pyscf-smdcds-v1"
)
PROFILE_ID = (
    "route2-profile-candidate-aimnet2-frozen-charge-water-"
    "smoothharmonicgalerkin-ddpcm-pyscf-smdcds-v1"
)
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "82ffd58ecdf3f2c9fde04e65c865f2203309542710010cfc6a468bb0ef1a6f92"
    ),
    "cold-replay.json": (
        "6a26ac9bbac39b45dc348b199a8c7ae8ac9d36dc36677e08c9d0af13f0e2f414"
    ),
    "README.md": "fab169b9b274423e83574125e4375ca051beb41ebdb2b1c63b82293a95894172",
}
MEASURED_KEYS = (
    "contract_version",
    "protocol",
    "identity",
    "forward_records",
    "reverse_records",
    "summary",
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-aimnet2-geometry-mediated-water-loop-artifact-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
        "smooth-harmonic-ddpcm-pyscf-smdcds-bidirectional-loop"
    )
    assert artifact["status"] == "diagnostic-gates-passed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-ddpcm-water"
    assert artifact["nonpolar_kind"] == "pyscf-smd-cds-water"
    assert artifact["dtype"] == "float64"
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert len(artifact["source_files_sha256"]) == 136

    protocol = artifact["protocol"]
    assert protocol["outer_charge_fixed_point"] is False
    assert protocol["fixed_geometry_mutual_polarization"] is False
    assert protocol["independent_forward_reverse_evaluations"] is True
    assert protocol["nonpolar_energy_gradient_same_component"] is True
    assert protocol["strict_original_smd_electrostatic_equivalence"] is False
    assert protocol["continuum"]["finite_dielectric_parameterization"] is True

    identity = artifact["identity"]
    assert identity["profile_id"] == PROFILE_ID
    assert identity["scalar_id"] == SCALAR_ID
    assert identity["nonpolar_provider_id"] == (
        "maple.route2.nonpolar.pyscf-smd-cds-energy-gradient.impl.v1"
    )
    assert identity["nonpolar_profile_id"] == (
        "pyscf-2.13.1-water-smd-cds-analytic-gradient-v1"
    )

    recomputed = summarize_aimnet2_geometry_mediated_water_loop(
        forward_records=artifact["forward_records"],
        reverse_records=artifact["reverse_records"],
        continuum_kind="harmonic-ddpcm-water",
        nonpolar_kind="pyscf-smd-cds-water",
    )
    assert recomputed == artifact["summary"]
    measured = {key: artifact[key] for key in MEASURED_KEYS}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256

    summary = recomputed
    assert summary["diagnostic_gates_passed"] is True
    assert all(summary["gates"].values())
    assert summary["same_scalar_nonpolar_energy_gradient"] is True
    assert summary["strict_original_smd_electrostatic_equivalence"] is False
    assert summary["capabilities"] == NO_CAPABILITIES
    assert summary["opt_admitted"] is False
    assert summary["freq_ts_irc_admitted"] is False
    assert summary["md_admitted"] is False
    assert summary["point_count_per_traversal"] == 17
    assert summary["forward"]["work"]["simpson_work_eV"] == pytest.approx(
        0.00014572688007338797, rel=0.0, abs=1.0e-20
    )
    assert summary["reverse"]["work"]["simpson_work_eV"] == pytest.approx(
        -0.00014572688007338883, rel=0.0, abs=1.0e-20
    )
    assert summary["forward_reverse_work_sum_eV"] == pytest.approx(
        -8.673617379884035e-19, rel=0.0, abs=1.0e-30
    )
    assert summary["maximum_same_coordinate_energy_error_eV"] == 0.0
    assert summary["maximum_same_coordinate_source_error"] == 0.0
    assert summary["maximum_same_coordinate_force_error_eV_per_A"] == 0.0
    assert summary["maximum_stationarity_absolute_residual"] == pytest.approx(
        2.0887116361891608e-14, rel=0.0, abs=1.0e-26
    )
    assert summary["maximum_surface_condition_number"] == pytest.approx(
        113.03902157292583, rel=0.0, abs=1.0e-12
    )
    assert summary["maximum_reciprocity_absolute_error_eV"] == pytest.approx(
        5.551115123125783e-15, rel=0.0, abs=1.0e-27
    )
    assert summary["maximum_charge_fd_absolute_error_eV_per_e"] == pytest.approx(
        2.6767477123712524e-12, rel=0.0, abs=1.0e-24
    )
    assert summary["maximum_source_gradient_half_error_eV_per_source_unit"] == 0.0
    assert summary["maximum_charge_gauge_vjp_norm_eV_per_A"] == 0.0

    for traversal in (artifact["forward_records"], artifact["reverse_records"]):
        for point in traversal:
            gradient = np.asarray(point["total_gradient_eV_per_A"])
            electrostatic = np.asarray(point["electrostatic_total_gradient_eV_per_A"])
            nonpolar = np.asarray(point["nonpolar_gradient_eV_per_A"])
            np.testing.assert_allclose(
                gradient, electrostatic + nonpolar, rtol=0.0, atol=2.0e-10
            )
            assert point["nonpolar"]["runtime_provenance"]["pyscf_version"] == (
                "2.13.1"
            )


def test_total_water_loop_artifact_recomputes_same_scalar_diagnostics():
    primary = _load(PRIMARY)
    _assert_measurement(primary)
    assert_source_files_match_execution_commit(ROOT, primary)


def test_total_water_loop_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(replay)
    for key in (*MEASURED_KEYS, "measurement_sha256"):
        assert replay[key] == primary[key]
    assert replay["exact_command"] != primary["exact_command"]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_total_water_loop_evidence_file_hashes_are_frozen():
    assert FILE_SHA256
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
