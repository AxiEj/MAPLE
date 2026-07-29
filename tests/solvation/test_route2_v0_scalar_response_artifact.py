from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from artifact_source_binding import (
    assert_source_files_match_execution_commit,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-scalar-response-water-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v0_water_artifact_is_source_bound_and_stops_on_failed_gates():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == "route2-v0-scalar-response-water-v1"
    assert artifact["schema_version"] == 1
    assert artifact["execution_git_head"] == (
        "9913d43cb7b5389cb74904bc7ac026df2187ed39"
    )
    assert artifact["status"] == "fail"
    assert artifact["failed_gates"] == [
        "base_field_charge",
        "neutral_response_reciprocity",
        "neutral_response_passivity",
    ]
    assert_source_files_match_execution_commit(ROOT, artifact)

    prereg = artifact["preregistration"]
    prereg_path = ROOT / prereg["path"]
    assert _sha256(prereg_path) == prereg["sha256"]
    assert prereg["protocol_id"] == (
        "route2-v0-scalar-response-water-prereg-v1"
    )
    frozen = artifact["frozen_input"]
    assert _sha256(ROOT / frozen["path"]) == frozen["sha256"]

    identity = artifact["scientific_identity"]
    assert identity["construction"] == (
        "anchored-frozen-checkpoint-scalar-response-v0"
    )
    assert identity["field_interface"] == "local-potential-gradient-v1"
    assert identity["continuum_fixed_point_solved"] is False
    assert identity["public_route_changed"] is False
    assert identity["training_or_fine_tuning"] is False
    assert "Do not proceed to V1" in artifact["stop_condition"]

    passed = artifact["gates"]["passed"]
    assert passed == {
        "base_field_charge": False,
        "dipole_scalar_gradient": True,
        "monopole_scalar_gradient": True,
        "neutral_response_passivity": False,
        "neutral_response_reciprocity": False,
        "zero_field_anchor": True,
        "zero_field_charge": True,
    }

    scalar = artifact["scalar_response"]
    np.testing.assert_allclose(
        [
            scalar["zero_field_anchor_maximum_absolute_error_e"],
            scalar["zero"]["response_total_charge_e"],
            scalar["base"]["response_total_charge_e"],
            scalar["maximum_monopole_scalar_gradient_error_e"],
            scalar["maximum_dipole_scalar_gradient_error_e_angstrom"],
        ],
        [
            7.16093850883226e-15,
            -3.8163916471489756e-16,
            -5.609264602901808e-4,
            2.2330792981348213e-6,
            2.3429963055821013e-6,
        ],
        rtol=1.0e-12,
        atol=1.0e-16,
    )

    response = artifact["neutral_response"]
    assert response["dimension"] == 11
    np.testing.assert_allclose(
        [
            response["antisymmetric_to_symmetric_frobenius_ratio"],
            response["minimum_eigenvalue_ev"],
            response["maximum_eigenvalue_ev"],
        ],
        [
            5.618413050008762e-5,
            -0.02277592319825096,
            0.019709750294421874,
        ],
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    assert artifact["runtime"]["device"] == "cuda"
    assert artifact["runtime"]["v0_evaluation_count"] == 26
