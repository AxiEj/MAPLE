from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
PREREG = (
    BENCHMARKS
    / "route2-v0-atomic-independent-particle-source-acetone-prereg-v1.json"
)
RUNNER = BENCHMARKS / "run_route2_v0_atomic_independent_particle_source_acetone.py"
ATOMIC_RESPONSE = (
    ROOT / "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_independent_particle_response.py"
)
RESPONSE_KERNEL = (
    ROOT / "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_response_kernel.py"
)
POINTS = (
    BENCHMARKS / "reproducers/route2-v0-atomic-displacement-source-acetone-v1/"
    "frozen-exterior-qm-mep-points-bohr.npy"
)
POINTS_PROVENANCE = POINTS.with_name("frozen-exterior-qm-mep-points-provenance.json")
RAW_QM = (
    BENCHMARKS
    / "reproducers/route2-v0-mace-mdp-induced-source-acetone-v1/qm-induced-mep.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def test_atomic_independent_particle_source_preregistration_freezes_all_inputs():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == (
        "route2-v0-atomic-independent-particle-source-acetone-prereg-v1"
    )
    assert protocol["status"] == "frozen-before-execution"
    assert protocol["execution_contract"]["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "run_route2_v0_atomic_independent_particle_source_acetone.py": _sha256(
            RUNNER
        ),
        "maple/function/calculator/extra_correction/implicit/"
        "route2_v0_atomic_independent_particle_response.py": _sha256(
            ATOMIC_RESPONSE
        ),
        "maple/function/calculator/extra_correction/implicit/"
        "route2_v0_response_kernel.py": _sha256(RESPONSE_KERNEL),
    }
    for relative, expected_hash in protocol["execution_contract"][
        "input_sha256"
    ].items():
        assert _sha256(ROOT / relative) == expected_hash
    assert protocol["finite_field_protocol"] == {
        "directions": ["x", "y", "z"],
        "field_steps_au": [3.0e-4, 1.0e-3],
        "selected_reporting_step_au": 3.0e-4,
        "signs": [-1, 1],
    }
    assert protocol["scientific_falsification_gates"][
        "mep_response_relative_frobenius_max"
    ] == 0.2
    assert protocol["scientific_falsification_gates"][
        "mep_response_relative_direction_max"
    ] == 0.3


def test_atomic_independent_particle_source_preregistration_has_no_repair_route():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert all(value is False for value in protocol["hard_constraints"].values())
    assert "Do not alter transition selection" in protocol["decision_rule"]
    assert "solvation label" in protocol["decision_rule"]
    assert "accuracy panel" in protocol["decision_rule"]
    assert "MACE density" in protocol["source_representation"]["molecular_response"]
    assert "C=C0-LS0L" in protocol["source_representation"]["completion"]


def test_frozen_exterior_qm_mep_points_remain_complete_and_raw_qm_bound():
    provenance = json.loads(POINTS_PROVENANCE.read_text(encoding="utf-8"))
    raw_qm = json.loads(RAW_QM.read_text(encoding="utf-8"))
    points = np.asarray(np.load(POINTS, allow_pickle=False), dtype=float)

    assert points.shape == (516, 3)
    assert provenance["points"]["file_sha256"] == _sha256(POINTS)
    assert provenance["points"]["array_sha256"] == _sha256_array(points)
    assert provenance["origin"]["raw_qm_induced_mep_sha256"] == _sha256(RAW_QM)
    assert raw_qm["input"]["source_point_count"] == len(points)
    assert raw_qm["input"]["source_points_bohr_sha256"] == _sha256_array(points)
