from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
SCRIPT = BENCHMARK_DIR / "run_chagb_derivative_capability_audit.py"
ARTIFACT = (
    BENCHMARK_DIR / "route1-chagb-derivative-capability-audit-2026-07-25.json"
)
SPEC = importlib.util.spec_from_file_location("chagb_derivative_audit", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_debug_force_parser_preserves_zero_analytical_mismatch():
    parsed = AUDIT.parse_debug_force(
        """
        NUMERICAL, ANALYTICAL FORCES (diff) from atom      1
             1    0.05786533     0.00000000     0.05786533
             2   -0.55185254     0.00000000    -0.55185254
             3    0.30099780     0.00000000     0.30099780
        RMS force error =  0.631E+03
        """
    )

    assert parsed["atom_one_based"] == 1
    assert parsed["all_analytical_components_zero"] is True
    assert parsed["any_numerical_component_nonzero"] is True
    assert parsed["numerical_analytical_match"] is False
    assert parsed["reported_rms_force_error"] == 631.0


def test_frozen_chagb_derivative_audit_is_self_hashed():
    artifact = _artifact()
    recorded = artifact.pop("content_sha256")
    payload = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert hashlib.sha256(payload).hexdigest() == recorded
    assert artifact["recorded_date"] == "2026-07-25"
    assert artifact["route1_contract"]["formula"] == (
        "E_solution(R) = E_MLIP,gas(R) + "
        "G_polar(R,q_fixed) + G_nonpolar(R)"
    )
    assert artifact["route1_contract"]["gas_phase_mm_energy"] is False
    assert artifact["route1_contract"]["mlip_retraining"] is False
    assert artifact["route1_contract"]["hydration_label_fit_or_residual"] is False


def test_chagb_and_ar6_fail_closed_from_source_and_runtime_evidence():
    artifact = _artifact()
    source = artifact["amberclassic_source_audit"]
    probe = artifact["runtime_debug_probe"]["debug_force"]
    decision = artifact["decision"]

    assert source["driver_plumbing"]["calls_egb_with_force_array"] is True
    assert source["driver_plumbing"]["calls_nonpolar_force"] is True
    assert (
        source["polar_source"]["chagb_equation_force_array_argument_present"] is False
    )
    assert source["polar_source"]["chagb_equation_accumulates_epol"] is True
    assert source["nonpolar_source"]["active_np_cavity_derivative_call"] is False
    assert source["nonpolar_source"]["active_dispersion_force_assignments"] > 0
    assert (
        source["manual_boundary"][
            "states_gbnsr6_cannot_yet_be_used_in_dynamics"
        ]
        is True
    )
    assert source["ar6_igb9"]["msander_igb9_condition_count"] == 0
    assert source["ar6_igb9"]["msander_ar6_topology_flag_count"] == 0
    assert probe["all_analytical_components_zero"] is True
    assert probe["any_numerical_component_nonzero"] is True
    assert decision["runtime_probe_supports_rejection"] is True
    assert decision["chagb_supported_tasks"] == ["sp"]
    assert decision["chagb_opt_scan_md"] is False
    assert decision["ar6_igb9_runtime_candidate_available"] is False
