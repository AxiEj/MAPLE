from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "variational-analytic-stability-water-604ecfa2"
)
PRIMARY = EVIDENCE / "measurements.json"
REPLAY = EVIDENCE / "cold-replay.json"

EXECUTION_HEAD = "604ecfa2035f140750124ada13df55f5bd3a3164"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "56e7b23760fbd45aadccc9d1579d3d8647ccba9879a30db72f5fff49d9a5e1ff"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "measurements.json": (
        "516fb35e14b1e6fa06e1a0b2f692c87f12250576bf53fa287fbf93f44dac6cb6"
    ),
    "cold-replay.json": (
        "d6b3a53d90c2e66892730cb064043704e4bc49f0882085d4e415478694c683f3"
    ),
    "README.md": "60a530266d4ef96e98548ced60fc5c58afa2cc8876b42a38aa70f6ec9f4933f3",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_measurement(artifact: dict[str, object]) -> None:
    assert artifact["schema_version"] == (
        "route2-variational-analytic-gaussian-multipole-water-"
        "local-stability-canary-v1"
    )
    assert artifact["artifact_kind"] == (
        "disabled-real-checkpoint-analytic-model-local-stability-" "multistart-canary"
    )
    assert artifact["status"] == "local-stability-canary-failed-not-admitted"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)

    identity = artifact["identity"]
    assert identity["long_range_evaluator_profile"] == (
        "graph-longrange-analytic-gaussian-multipole-realspace-v1"
    )
    assert identity["conjugacy_sign"] == 1
    assert identity["scalar_id"] == (
        "route2-variational-macepolar-analytic-gaussian-multipole-"
        "energygradient-smoothharmonicgalerkin-cpcm-v1"
    )

    assert artifact["decision"] == {
        "all_local_stability_gates_passed": False,
        "global_passivity_proven": False,
        "global_root_uniqueness_proven": False,
        "multi_start_root_gate_passed": True,
        "one_water_local_stability_canary_passed": False,
        "public_force_admitted": False,
        "state_residual_factorization_gate_passed": True,
        "tier_v_admitted": False,
    }

    multi_start = artifact["multi_start"]
    assert multi_start["gate_passed"] is True
    assert len(multi_start["starts"]) == 5
    assert all(record["converged"] for record in multi_start["starts"].values())
    assert max(
        record["actual_unmixed_residual_norm"]
        for record in multi_start["starts"].values()
    ) == pytest.approx(1.720697910222882e-10, rel=0.0, abs=1.0e-24)
    assert multi_start["maxima"] == pytest.approx(
        {
            "reduced_relative_difference": 3.821619684420472e-10,
            "source_relative_difference": 3.821619668205479e-10,
            "field_relative_difference": 4.39964064689302e-10,
            "energy_absolute_difference_eV": 4.547473508864641e-13,
        },
        rel=0.0,
        abs=1.0e-24,
    )

    stability = artifact["stability"]
    gauge = stability["gauge_reduction"]
    assert gauge["reduced_fixed_charge_tangent_used"] is True
    assert gauge["total_charge_e"] == 0.0
    assert gauge["constant_potential_coupling_energy_eV"] == 0.0
    assert gauge["constant_potential_eV_per_e"] == pytest.approx(
        0.11740458051762441, rel=0.0, abs=1.0e-15
    )
    factorization = stability["state_residual_factorization"]
    assert factorization["gate_passed"] is True
    assert factorization["relative_frobenius_error"] == pytest.approx(
        4.8093797227417603e-17, rel=0.0, abs=1.0e-30
    )

    local = stability["local_stability"]
    assert local["admitted"] is False
    assert local["decisions"] == {
        "model_reciprocity_gate_passed": True,
        "model_passivity_gate_passed": False,
        "model_local_invertibility_gate_passed": False,
        "continuum_reciprocity_gate_passed": True,
        "continuum_nonpositive_curvature_gate_passed": True,
        "feedback_real_spectrum_gate_passed": True,
        "feedback_nonnegative_spectrum_gate_passed": False,
        "feedback_contraction_gate_passed": True,
        "residual_local_nonsingularity_gate_passed": True,
        "combined_hessian_positive_gate_passed": False,
        "all_local_stability_gates_passed": False,
    }
    model = local["model_response_jacobian"]
    eigenvalues = np.asarray(model["eigenvalues"], dtype=float)
    assert model["symmetry_relative_defect"] == pytest.approx(
        1.3495196506404244e-15, rel=0.0, abs=1.0e-28
    )
    assert np.count_nonzero(eigenvalues < -1.0e-12) == 5
    assert np.count_nonzero(np.abs(eigenvalues) <= 1.0e-12) == 6
    assert np.count_nonzero(eigenvalues > 1.0e-12) == 12
    assert model["minimum_eigenvalue"] == pytest.approx(
        -0.021567868753146407, rel=0.0, abs=1.0e-16
    )
    assert model["maximum_eigenvalue"] == pytest.approx(
        0.044638953966440714, rel=0.0, abs=1.0e-16
    )
    assert local["combined_hessian"] is None
    assert local["feedback"]["spectral_radius"] == pytest.approx(
        0.010081162126547305, rel=0.0, abs=1.0e-16
    )
    assert min(local["feedback"]["eigenvalues_real"]) < -1.0e-2
    assert local["residual_jacobian"][
        "minimum_to_maximum_singular_ratio"
    ] == pytest.approx(0.9726080031165828, rel=0.0, abs=1.0e-15)

    measured = {
        key: artifact[key]
        for key in (
            "protocol",
            "identity",
            "geometry",
            "multi_start",
            "stability",
            "decision",
        )
    }
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256


def test_analytic_stability_canary_is_source_bound_and_fail_closed():
    _assert_measurement(_load(PRIMARY))


def test_analytic_stability_cold_replay_is_scientifically_identical():
    primary = _load(PRIMARY)
    replay = _load(REPLAY)
    _assert_measurement(primary)
    _assert_measurement(replay)
    for key in (
        "protocol",
        "identity",
        "geometry",
        "multi_start",
        "stability",
        "decision",
        "measurement_sha256",
    ):
        assert replay[key] == primary[key]
    assert replay["runtime_seconds"] != primary["runtime_seconds"]


def test_analytic_stability_evidence_file_hashes_are_frozen():
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
    listed = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in (EVIDENCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert listed == FILE_SHA256
