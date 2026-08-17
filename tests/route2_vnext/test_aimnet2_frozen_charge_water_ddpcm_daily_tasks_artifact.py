from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release import (
    canonical_json_sha256,
    summarize_aimnet2_geometry_mediated_frequency_water,
    summarize_aimnet2_geometry_mediated_hvp_water,
    summarize_aimnet2_geometry_mediated_water_loop,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "aimnet2-frozen-charge-water-ddpcm-daily-tasks-76d4d097"
)
EXECUTION_HEAD = "76d4d0974abf7a631d783f65723f87af1af88498"
EXECUTION_TREE = "58218130c36191cd2a5002ca41f96bfaa9bf1322"
CHECKPOINT_SHA256 = "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
MEASUREMENT_SHA256 = {
    "water-loop": "4c70022f570e2736e4b2f16312009971058461cf2a78776e3ba45e01738cd26f",
    "hvp": "60159b0b31ec0ae1ae308dd03175121d9c467078882965cbf9a72670d4d39dfb",
    "frequency": "2cb5aceee9d22c84bfe458bf9e9967d5c5cb61c9536f90a450246d97730c258c",
}
ARTIFACT_KIND = {
    "water-loop": (
        "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
        "smooth-harmonic-ddpcm-water-bidirectional-loop"
    ),
    "hvp": (
        "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
        "smooth-harmonic-ddpcm-water-complete-hvp"
    ),
    "frequency": (
        "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
        "smooth-harmonic-ddpcm-stationary-water-dense-hessian-frequency"
    ),
}
STATUS = {
    "water-loop": "diagnostic-gates-passed-not-admitted",
    "hvp": "diagnostic-hvp-gates-passed-not-admitted",
    "frequency": "diagnostic-frequency-gates-passed-not-admitted",
}
MEASURED_KEYS = {
    "water-loop": (
        "contract_version",
        "protocol",
        "identity",
        "forward_records",
        "reverse_records",
        "summary",
    ),
    "hvp": (
        "contract_version",
        "protocol",
        "identity",
        "admission_boundary",
        "center_record",
        "direction_records",
        "finite_difference_records",
        "summary",
    ),
    "frequency": (
        "contract_version",
        "protocol",
        "identity",
        "admission_boundary",
        "search_record",
        "center_record",
        "hessian_vector_records",
        "finite_difference_records",
        "summary",
    ),
}
FILE_SHA256 = {
    "primary-frequency.json.gz": (
        "aff9e81c04df47bf1270c989d19b3b3be28ee3e10db9efe47077267d2e2d81b7"
    ),
    "primary-hvp.json.gz": (
        "af00264dac7e911de0f001b82f026950980d5d3c9cd7a89ab091ee1f16d40050"
    ),
    "primary-water-loop.json.gz": (
        "b8801201bf54c028a604d0a2ce1f983113ae1e275c26783a35f26ce5844ef647"
    ),
    "cold-replay-frequency.json.gz": (
        "6d562b51a137995cf7be45f43581d0449cc4eb781057ace3285a6a2cb7821e75"
    ),
    "cold-replay-hvp.json.gz": (
        "868417fc60e97c30ce21af40d5c650b5e6239f18730cd52f55fba5bcb630e6a4"
    ),
    "cold-replay-water-loop.json.gz": (
        "8e6271cd25f61271c6f616ba6275186c4b40f0975ae3f602b783ea4014f993fc"
    ),
    "README.md": "f207e4c7cb46900731d70b219c3a1942bd7a241ccb2782b2a0e4c11f09c95fb7",
}


def _load(name: str, *, replay: bool = False) -> dict[str, object]:
    prefix = "cold-replay" if replay else "primary"
    with gzip.open(
        EVIDENCE / f"{prefix}-{name}.json.gz", "rt", encoding="utf-8"
    ) as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recompute_summary(name: str, artifact: dict[str, object]) -> dict[str, object]:
    if name == "water-loop":
        return summarize_aimnet2_geometry_mediated_water_loop(
            forward_records=artifact["forward_records"],
            reverse_records=artifact["reverse_records"],
            continuum_kind="harmonic-ddpcm-water",
        )
    if name == "hvp":
        return summarize_aimnet2_geometry_mediated_hvp_water(
            center_record=artifact["center_record"],
            direction_records=artifact["direction_records"],
            finite_difference_records=artifact["finite_difference_records"],
            continuum_kind="harmonic-ddpcm-water",
        )
    return summarize_aimnet2_geometry_mediated_frequency_water(
        search_record=artifact["search_record"],
        center_record=artifact["center_record"],
        hessian_vector_records=artifact["hessian_vector_records"],
        finite_difference_records=artifact["finite_difference_records"],
        continuum_kind="harmonic-ddpcm-water",
    )


def _assert_common(name: str, artifact: dict[str, object]) -> None:
    assert artifact["artifact_kind"] == ARTIFACT_KIND[name]
    assert artifact["status"] == STATUS[name]
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["execution_git_tree"] == EXECUTION_TREE
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["aimnet_runtime"] == "reconstructed-python-float64"
    assert artifact["continuum_kind"] == "harmonic-ddpcm-water"
    assert artifact["dtype"] == "float64"
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256[name]
    assert len(artifact["source_files_sha256"]) == 135

    protocol = artifact["protocol"]
    assert protocol["outer_charge_fixed_point"] is False
    assert protocol["fixed_geometry_mutual_polarization"] is False
    assert protocol["continuum"]["finite_dielectric_parameterization"] is True
    assert protocol["continuum"]["solvent"] == "water"
    assert protocol["continuum"]["dielectric"] == pytest.approx(78.355)
    assert protocol["continuum"]["aimnet2_source_evaluation"] == (
        "one-shot-per-geometry"
    )
    assert protocol["continuum"]["continuum_field_supplied_to_aimnet2"] is False
    assert protocol["continuum"]["electronic_scf_iteration"] is False

    identity = artifact["identity"]
    assert identity["scalar_id"] == (
        "route2-candidate-aimnet2-frozen-charge-water-"
        "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
    )
    assert identity["profile_id"] == (
        "route2-profile-candidate-aimnet2-frozen-charge-water-"
        "smoothharmonicgalerkin-ddpcm-electrostatic-v1"
    )
    assert identity["model_runtime"]["checkpoint_weights_changed"] is False
    assert identity["model_runtime"]["public_hessian"] is False
    assert identity["model_runtime"]["public_hvp"] is False
    assert identity["model_runtime"]["public_ase_calculator"] is False

    recomputed = _recompute_summary(name, artifact)
    assert recomputed == artifact["summary"]
    assert recomputed["diagnostic_gates_passed"] is True
    assert all(recomputed["gates"].values())
    assert recomputed["conductor_reference_only"] is False
    assert recomputed["finite_dielectric_parameterization"] is True
    assert recomputed["capabilities"] == NO_CAPABILITIES
    measured = {key: artifact[key] for key in MEASURED_KEYS[name]}
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256[name]


def test_primary_daily_task_artifacts_recompute_and_remain_closed():
    artifacts = {name: _load(name) for name in MEASURED_KEYS}
    for name, artifact in artifacts.items():
        _assert_common(name, artifact)
        assert_source_files_match_execution_commit(ROOT, artifact)


def test_daily_task_cold_replays_are_scientifically_identical():
    for name in MEASURED_KEYS:
        primary = _load(name)
        replay = _load(name, replay=True)
        _assert_common(name, replay)
        for key in (*MEASURED_KEYS[name], "measurement_sha256"):
            assert replay[key] == primary[key]
        assert replay["exact_command"] != primary["exact_command"]
        assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_daily_task_selected_numerical_results_are_frozen():
    water = _load("water-loop")["summary"]
    assert water["forward_reverse_work_sum_eV"] == pytest.approx(
        -1.100465205072787e-17, rel=0.0, abs=1.0e-30
    )
    assert water["maximum_stationarity_absolute_residual"] == pytest.approx(
        2.0887116361891608e-14, rel=0.0, abs=1.0e-28
    )
    assert water["minimum_point_source_shell_margin_A"] == pytest.approx(
        0.20407200707293016, rel=0.0, abs=1.0e-15
    )

    hvp = _load("hvp")["summary"]
    assert hvp["bilinear_symmetry"]["absolute_error_eV_per_A2"] == pytest.approx(
        7.105427357601002e-15, rel=0.0, abs=1.0e-29
    )
    assert hvp["finite_difference_error_norms"]["total_hvp_eV_per_A2"] == (
        pytest.approx(
            [0.000527159915140729, 0.00013179027720185125, 3.294758659925606e-05],
            rel=0.0,
            abs=1.0e-18,
        )
    )
    assert hvp["translation_zero_modes"]["total_hvp_norms_eV_per_A2"] == [
        0.0,
        0.0,
        0.0,
    ]

    frequency = _load("frequency")["summary"]
    assert frequency["search"]["result"]["solution"] == pytest.approx(
        [1.0207730339378311, 1.0207730339378314, 2.00873796263625],
        rel=0.0,
        abs=1.0e-14,
    )
    assert frequency["center"]["gradient_max_abs_eV_per_A"] == pytest.approx(
        6.7449165827224156e-15, rel=0.0, abs=1.0e-28
    )
    assert frequency["dense_hessian"]["symmetry_max_abs_eV_per_A2"] == (
        pytest.approx(1.126172990825879e-14, rel=0.0, abs=1.0e-28)
    )
    assert frequency["vibrational_analysis"]["frequencies_cm1"] == pytest.approx(
        [1663.9300726005338, 2672.8294837599124, 2900.466040251068],
        rel=0.0,
        abs=1.0e-10,
    )
    assert frequency["search"]["protocol"]["internal_coordinate_transform"] == (
        "componentwise-open-bound-tanh"
    )


def test_daily_task_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
