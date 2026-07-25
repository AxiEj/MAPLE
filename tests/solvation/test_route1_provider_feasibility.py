from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "docs/implicit-solvation/benchmarks"
)
AUDIT_PATH = BENCHMARK_DIR / "route1-provider-feasibility-2026-07-24.json"


def _load_audit():
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_provider_feasibility_audit_has_a_self_consistent_fingerprint():
    audit = _load_audit()
    recorded = audit.pop("content_sha256")
    payload = json.dumps(
        audit,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert hashlib.sha256(payload).hexdigest() == recorded
    assert audit["schema_version"] == 1
    assert audit["protocol"]["protocol_id"] == "route1-provider-feasibility-v1"
    assert all(
        len(fingerprint) == 64 for fingerprint in audit["source_fingerprints"].values()
    )


def test_provider_feasibility_audit_cannot_imply_product_readiness():
    audit = _load_audit()

    assert audit["route"]["name"] == "Additive fixed-charge PB/GB implicit solvation"
    assert audit["iwm_gb_2024"][
        "experimental_hydration_labels_used_for_parameter_optimization"
    ]
    assert not audit["iwm_gb_2024"]["released_force_api_identified_in_audit"]
    assert audit["openmm_agbnp_plugin"]["readme_contract"]["implemented_model"] == (
        "AGBNP1"
    )
    assert audit["openmm_agbnp_plugin"]["local_openmm_8_5_2_reference_probe"][
        "finite_energy_kj_per_mol"
    ] == pytest.approx(-28.472)
    assert not audit["openmm_agbnp3_plugin"][
        "generic_small_molecule_typing_provider_present"
    ]
    assert all(audit["protocol"]["product_admission_gates"].values())
    assert not audit["overall_decision"]["new_runtime_provider_added"]
