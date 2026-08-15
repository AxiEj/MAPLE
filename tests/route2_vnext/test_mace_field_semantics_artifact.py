from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact_source_binding import assert_source_files_match_execution_commit
from maple.solvation.release.evidence import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "route2" / "evidence" / "mace-field-semantics-3014f1a6"
RUN1 = EVIDENCE / "run1.json"
RUN2 = EVIDENCE / "run2.json"

EXECUTION_HEAD = "3014f1a63cc966ac8186c1af78cde327b7519456"
CHECKPOINT_SHA256 = "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
MEASUREMENT_SHA256 = "e77ab89951e8eefec3480571fcc8df03026090f06aa15b817c464d033f230439"
SEMANTICS_MEASUREMENT_SHA256 = (
    "dde808e7dde56acaf99fef191e3df1c6bb2ce3253f248a9d4a6c682ff2815c95"
)
MANIFEST_SHA256 = "3b425ff8ae41d0817a1b1f06e275b3f09478c4a27793f7d53a649cee24f446c8"
NO_CAPABILITIES = {"E": False, "F": False, "H": False, "M": False, "V": False}
FILE_SHA256 = {
    "run1.json": "99427334621fddf6384a86afacb4793684239ab32eb667292ad81232d9be61fc",
    "run2.json": "e6f2db148012650703c0d299a64fe182ed3c9b50efbbcb578b2533201318cdbf",
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_uniform_replay(record: dict[str, object]) -> dict[str, object]:
    keys = (
        "source_max_abs_error",
        "energy_absolute_error_eV",
        "force_max_abs_error_eV_per_A",
        "dipole_max_abs_error_e_angstrom",
        "energy_derivative_absolute_error_e_angstrom",
        "source_path_matched",
        "passed",
        "branch_semantics",
    )
    return {key: record[key] for key in keys}


def _base_directional(record: dict[str, object]) -> dict[str, object]:
    keys = (
        "analytic_directional_derivative_eV",
        "finite_difference_values_eV",
        "steps",
        "minimum_absolute_error_eV",
        "passed",
    )
    return {key: record[key] for key in keys}


def test_real_checkpoint_field_semantics_are_hybrid_and_fail_closed():
    artifact = _load(RUN1)
    assert artifact["schema_version"] == "route2-mace-field-semantics-water-v1"
    assert artifact["status"] == "native-injection-omits-upstream-explicit-work"
    assert artifact["execution_git_head"] == EXECUTION_HEAD
    assert artifact["working_tree_clean"] is True
    assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert artifact["capabilities"] == NO_CAPABILITIES
    assert artifact["decision"] == {
        "native_injection_complete_external_enthalpy": False,
        "native_raw_graph_charging_identity_passed": True,
        "native_raw_graph_directional_derivative_passed": True,
        "phi0_physical_ledger_admitted": False,
        "phi1_delta_physical_ledger_admitted": False,
        "phi1_delta_semantics_complete": False,
        "source_pcm_physics_admitted": False,
        "upstream_and_native_energy_branches_identical": False,
        "zero_field_baseline_passed": True,
    }

    replay = artifact["uniform_field_replay"]
    assert replay["source_max_abs_error"] == 0.0
    assert replay["dipole_max_abs_error_e_angstrom"] == 0.0
    assert replay["branch_semantics"] == "hybrid-branch-dependent"
    assert replay["selected_work_sign"] == 1
    assert replay["explicit_work_identity_absolute_error_eV"] < 7.0e-14
    assert replay["energy_absolute_error_eV"] > 3.8e-3
    assert replay["force_max_abs_error_eV_per_A"] > 3.5e-3
    assert replay["energy_derivative_absolute_error_e_angstrom"] > 0.49

    directional = artifact["native_raw_directional_derivative"]
    assert directional["forward_reverse_passed"] is True
    assert directional["forward_reverse_absolute_error_eV"] < 2.0e-16
    assert directional["minimum_absolute_error_eV"] < 3.0e-10
    assert (
        artifact["native_raw_charging_path"]["endpoint_identity_absolute_error_eV"]
        < 1.0e-13
    )

    manifest = artifact["field_semantics_manifest"]
    assert manifest["configuration_sha256"] == MANIFEST_SHA256
    assert manifest["native_injection_explicit_work_included"] is False
    assert manifest["upstream_uniform_explicit_work_included"] is True
    assert manifest["phi1_semantics_complete"] is False
    assert manifest["evidence_measurement_sha256s"] == [SEMANTICS_MEASUREMENT_SHA256]
    assert artifact["field_semantics_measurement_sha256"] == (
        SEMANTICS_MEASUREMENT_SHA256
    )

    semantics_measurements = {
        "zero_field_baseline": artifact["zero_field_baseline"],
        "uniform_field_replay": _base_uniform_replay(replay),
        "upstream_native_energy_difference_eV": replay[
            "upstream_native_energy_difference_eV"
        ],
        "explicit_uniform_work_eV": replay["explicit_uniform_work_eV"],
        "work_sign_canaries": replay["work_sign_canaries"],
        "selected_work_sign": replay["selected_work_sign"],
        "explicit_work_identity_absolute_error_eV": replay[
            "explicit_work_identity_absolute_error_eV"
        ],
        "native_raw_reverse_directional_derivative": _base_directional(directional),
        "native_raw_forward_directional_derivative_eV": directional[
            "forward_mode_directional_derivative_eV"
        ],
        "native_raw_forward_reverse_absolute_error_eV": directional[
            "forward_reverse_absolute_error_eV"
        ],
        "native_raw_forward_reverse_passed": directional["forward_reverse_passed"],
        "native_raw_charging_path": artifact["native_raw_charging_path"],
        "external_python_sources": artifact["external_python_sources"],
    }
    assert canonical_json_sha256(semantics_measurements) == (
        SEMANTICS_MEASUREMENT_SHA256
    )
    measured = {
        key: artifact[key]
        for key in (
            "field_semantics_manifest",
            "protocol",
            "zero_field_baseline",
            "uniform_field_replay",
            "native_raw_directional_derivative",
            "native_raw_charging_path",
            "decision",
        )
    }
    assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
    assert canonical_json_sha256(measured) == MEASUREMENT_SHA256
    assert_source_files_match_execution_commit(ROOT, artifact)


def test_field_semantics_cold_replay_and_files_are_exact():
    first = _load(RUN1)
    second = _load(RUN2)
    for artifact in (first, second):
        assert artifact["execution_git_head"] == EXECUTION_HEAD
        assert artifact["checkpoint"]["sha256"] == CHECKPOINT_SHA256
        assert artifact["measurement_sha256"] == MEASUREMENT_SHA256
        assert artifact["capabilities"] == NO_CAPABILITIES
        assert_source_files_match_execution_commit(ROOT, artifact)
    for key in (
        "field_semantics_manifest",
        "field_semantics_measurement_sha256",
        "protocol",
        "zero_field_baseline",
        "uniform_field_replay",
        "native_raw_directional_derivative",
        "native_raw_charging_path",
        "decision",
        "measurement_sha256",
    ):
        assert second[key] == first[key]
    for name, expected in FILE_SHA256.items():
        assert _sha256(EVIDENCE / name) == expected
