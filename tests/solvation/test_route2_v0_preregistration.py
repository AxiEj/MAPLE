from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREREG = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-scalar-response-water-prereg-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v0_water_protocol_is_frozen_before_execution():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == (
        "route2-v0-scalar-response-water-prereg-v1"
    )
    assert protocol["schema_version"] == 1
    assert protocol["status"] == "preregistered-before-execution"
    assert protocol["construction"]["field_interface"] == (
        "local-potential-gradient-v1"
    )
    assert protocol["construction"]["response_source"] == (
        "c_V0(f)=Q^-T grad_f W0(f)"
    )

    frozen = protocol["frozen_input"]
    source = ROOT / frozen["field_source_artifact"]
    assert source.is_file()
    assert _sha256(source) == frozen["field_source_artifact_sha256"]
    artifact = json.loads(source.read_text(encoding="utf-8"))
    assert artifact["checkpoint"]["sha256"] == frozen["checkpoint_sha256"]
    assert artifact["scientific_identity"]["reaction_field_projector"] == (
        frozen["reaction_field_projector"]
    )

    gates = protocol["gates"]
    assert gates["finite_difference_step"] == 1.0e-4
    assert gates["zero_field_anchor_maximum_absolute_error_e"] == 1.0e-10
    assert gates["neutral_response_maximum_eigenvalue_ev"] == 1.0e-6
    assert gates["neutral_response_antisymmetric_frobenius_ratio"] == 1.0e-5

    prohibited = " ".join(protocol["prohibited_actions"])
    assert "fine-tune" in prohibited
    assert "MAP" in prohibited
    assert "FreeSolv or MNSol" in prohibited
    assert "fall through to V1" in prohibited
    assert "Do not proceed to V1" in protocol["stop_condition"]
    assert "does not change the public Route 2" in protocol["claim_boundary"]
