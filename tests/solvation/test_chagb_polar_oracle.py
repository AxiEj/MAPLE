"""Portable numeric comparison with source-pinned native CHA polar algebra.

The fixture contains numbers and provenance only: no upstream source, executable,
experimental labels, complete coordinate forces, or hydration-score claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "data/chagb_polar_oracle_v1.json"
GENERATOR = ROOT / "docs/implicit-solvation/benchmarks/run_route1_chagb_polar_oracle.py"
SOURCE_MANIFEST = Path(__file__).parent / "data/chagb_at26_source_manifest_v1.json"
LOCAL_NATIVE_SOURCE = (
    ROOT / ".omx/vendor/route1-gbnsr6-at26-precision-20260905/pristine"
)
CONTENT_SHA256 = "f38bd061dfa5b717d0fbcc41e32414e4366ac30a6a63ca893163cfd2e3c9167d"
FULL_REFERENCE_EGB_KCAL_MOL = -6.282267527560081


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fixture():
    artifact = json.loads(FIXTURE.read_text())
    payload = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert hashlib.sha256(_canonical(payload).encode()).hexdigest() == CONTENT_SHA256
    assert artifact["content_sha256"] == CONTENT_SHA256
    return artifact


def test_oracle_provenance_and_full_native_bridge_are_frozen():
    artifact = _fixture()
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "route1-chagb-polar-algebra-native-oracle-v1"
    assert artifact["label_reads"] is False
    assert artifact["new_qm"] is False
    assert artifact["full_chain_complete"] is False
    assert artifact["source_manifest_file_count"] == 64
    assert artifact["source_manifest_sha256"] == (
        "fea10fa41adb6a5b36bd09d48bf74e3efb6d9f82a39025e7caa3a9f9c385a65c"
    )
    assert (
        hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
        == artifact["source_manifest_sha256"]
    )
    assert artifact["verified_source_file_count"] == 70
    assert (
        artifact["generator_sha256"]
        == hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    )
    assert artifact["two_clean_scratch_numeric_replays_identical"] is True
    assert len(artifact["runs"]) == 2
    assert len(artifact["cases"]) == 9
    assert artifact["pinned_full_reference_egb_kcal_mol"] == FULL_REFERENCE_EGB_KCAL_MOL
    native = artifact["full_native_bridge_expected"]
    algebra = artifact["cases"][0]["expected"]
    assert native["polar_kcal_mol"] == FULL_REFERENCE_EGB_KCAL_MOL
    assert abs(native["polar_kcal_mol"] - algebra["polar_kcal_mol"]) < 1e-9
    assert (
        artifact["runs"][0]["numeric_payload_sha256"]
        == artifact["runs"][1]["numeric_payload_sha256"]
    )
    for run in artifact["runs"]:
        assert (
            run["bridge_absolute_error_kcal_mol"]
            <= artifact["bridge_tolerance_kcal_mol"]
        )
        assert len(run["object_sha256"]) == 31
        assert set(run["static_link_input_sha256"]) == {
            "libamber_common.a",
            "libxblas-amb.a",
        }
        assert len(run["dynamic_library_sha256"]) >= 1
        assert run["commands"][0]["argv"][:3] == ["make", "-B", "-j1"]
        assert all(
            len(command["stdout_sha256"]) == len(command["stderr_sha256"]) == 64
            for command in run["commands"]
        )
    for field in (
        "object_sha256",
        "static_link_input_sha256",
        "dynamic_library_sha256",
        "binary_sha256",
    ):
        assert artifact["runs"][0][field] == artifact["runs"][1][field]


def test_local_native_oracle_rejects_unverified_build_products(tmp_path):
    if not LOCAL_NATIVE_SOURCE.is_dir():
        pytest.skip("Pinned local native source is unavailable.")
    verify = runpy.run_path(str(GENERATOR))["_verify_sources"]
    scratch = tmp_path / "source"
    shutil.copytree(LOCAL_NATIVE_SOURCE, scratch)
    verify(scratch, SOURCE_MANIFEST)
    stale_object = scratch / "src/gbnsr6/stale.o"
    stale_object.write_bytes(b"unverified prebuilt object")
    with pytest.raises(ValueError, match="unverified files"):
        verify(scratch, SOURCE_MANIFEST)
    stale_object.unlink()
    symlink = scratch / "src/gbnsr6/stale.mod"
    symlink.symlink_to(scratch / "src/gbnsr6/egb.F90")
    with pytest.raises(ValueError, match="symlink"):
        verify(scratch, SOURCE_MANIFEST)


@pytest.mark.parametrize("case_index", range(9))
def test_torch_polar_algebra_matches_native_oracle(case_index):
    torch = pytest.importorskip("torch")
    from maple.function.calculator.extra_correction.implicit.torch_chagb import (
        cha_polar_from_inverse_born,
    )

    case = _fixture()["cases"][case_index]
    inputs = case["inputs"]
    expected = case["expected"]
    tensor = lambda name: torch.tensor(inputs[name], dtype=torch.float64)
    result = cha_polar_from_inverse_born(
        tensor("positions_angstrom"),
        tensor("charges_e"),
        tensor("effective_cha_radii_angstrom"),
        tensor("unshifted_inverse_born_per_angstrom"),
    )
    for name, actual in (
        ("polar_kcal_mol", result.polar_kcal_mol),
        ("self_kcal_mol", result.self_kcal_mol),
        ("pair_kcal_mol", result.pair_kcal_mol),
    ):
        delta = abs(float(actual) - expected[name])
        assert delta <= 1e-9, (case["case_id"], name, delta)
        if abs(expected[name]) > 1e-6:
            assert delta / abs(expected[name]) <= 1e-10
    for name, actual in (
        ("electrostatic_size_angstrom", result.electrostatic_size_angstrom),
        ("inverse_born_shift_per_angstrom", result.inverse_born_shift_per_angstrom),
    ):
        assert abs(float(actual) - expected[name]) <= 1e-10, (case["case_id"], name)
    for name, actual in (
        ("born_radii_angstrom", result.born_radii_angstrom),
        ("effective_charges_e", result.effective_charges_e),
        ("cha_factors", result.cha_factors),
    ):
        deltas = [
            abs(value - reference)
            for value, reference in zip(actual.tolist(), expected[name])
        ]
        assert len(deltas) == len(expected[name])
        assert max(deltas) <= 1e-10, (case["case_id"], name, max(deltas))
    assert result.complete_coordinate_graph is False
