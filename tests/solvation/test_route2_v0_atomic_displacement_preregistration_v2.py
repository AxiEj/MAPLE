from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
PREREG = BENCHMARKS / "route2-v0-atomic-displacement-hf-def2-tzvpd-prereg-v2.json"
V1_AUDIT = (
    BENCHMARKS / "route2-v0-atomic-displacement-hf-def2-tzvpd-v1-"
    "preflight-reproducibility-audit.json"
)
V2_AUDIT = (
    BENCHMARKS / "route2-v0-atomic-displacement-hf-def2-tzvpd-v2-"
    "reproducibility-audit.json"
)
GENERATOR = BENCHMARKS / "generate_route2_v0_atomic_displacement_response_v2.py"
SOURCE = (
    ROOT / "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_displacement_response.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generator_module():
    specification = importlib.util.spec_from_file_location(
        "route2_v0_atomic_displacement_response_generator_v2",
        GENERATOR,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_atomic_displacement_v2_preregistration_freezes_the_reproducible_source():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert (
        protocol["protocol_id"]
        == "route2-v0-atomic-displacement-hf-def2-tzvpd-v2-prereg"
    )
    assert protocol["status"] == "frozen-before-asset-generation"
    assert protocol["v1_preflight_disposition"] == {
        "audit_path": (
            "docs/implicit-solvation/benchmarks/"
            "route2-v0-atomic-displacement-hf-def2-tzvpd-v1-"
            "preflight-reproducibility-audit.json"
        ),
        "verdict": "v1-not-admitted-because-thread-runtime-was-not-frozen",
    }
    contract = protocol["generation_contract"]
    assert contract["method"] == "PySCF AtomSphAverageRHF"
    assert contract["basis"] == "def2-TZVPD"
    assert contract["runtime"]["threads"] == 1
    assert contract["runtime"]["thread_environment"] == {
        "BLIS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert contract["deterministic_archive"] == {
        "compression": "ZIP_DEFLATED-level-9",
        "format": "npz",
        "member_order": "lexicographic-array-key",
        "zip_member_timestamp_utc": "1980-01-01T00:00:00+00:00",
    }


def test_atomic_displacement_v2_preregistration_binds_implementation_and_archive():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))
    generator = _generator_module()

    assert protocol["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/"
        "generate_route2_v0_atomic_displacement_response_v2.py": _sha256(GENERATOR),
        "maple/function/calculator/extra_correction/implicit/"
        "route2_v0_atomic_displacement_response.py": _sha256(SOURCE),
    }
    assert generator.THREAD_COUNT == 1
    assert (
        generator.DETERMINISTIC_ARCHIVE_CONTRACT
        == protocol["generation_contract"]["deterministic_archive"]
    )
    arrays = {
        "atomic_numbers": np.asarray([1, 6], dtype=np.int64),
        "radial_grid_bohr": np.asarray([0.0, 1.0, 2.0]),
    }
    encoded = generator._deterministic_npz_bytes(arrays)
    assert encoded == generator._deterministic_npz_bytes(arrays)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["atomic_numbers"], arrays["atomic_numbers"]
        )
        np.testing.assert_array_equal(
            archive["radial_grid_bohr"], arrays["radial_grid_bohr"]
        )


def test_atomic_displacement_v1_preflight_outputs_are_preserved_but_not_admitted():
    audit = json.loads(V1_AUDIT.read_text(encoding="utf-8"))

    assert audit["decision"] == {
        "reason": (
            "The v1 preregistration did not freeze the numerical thread runtime. "
            "Two executions at the same tracked source revision and Python/PySCF "
            "versions produce non-identical enclosed-electron tables, so a v1 table "
            "must not be selected for the next physical gate."
        ),
        "status": "preflight-not-admitted",
        "verdict": "freeze-v2-single-thread-runtime-before-any-source-falsifier",
    }
    for raw in audit["raw_outputs"].values():
        assert _sha256(ROOT / raw["manifest_path"]) == raw["manifest_sha256"]
        assert _sha256(ROOT / raw["table_path"]) == raw["table_sha256"]
    assert audit["hard_constraints"]["v1_outputs_deleted"] is False
    assert audit["hard_constraints"]["solvation_accuracy_panel_run"] is False


def test_atomic_displacement_v2_repeat_is_byte_identical_before_qm_mep_use():
    audit = json.loads(V2_AUDIT.read_text(encoding="utf-8"))

    assert audit["decision"]["status"] == "pass"
    assert (
        audit["decision"]["verdict"]
        == "admit-v2-free-atom-translation-tangent-asset-to-separately-"
        "preregistered-gas-phase-qm-mep-source-falsifier-only"
    )
    assert audit["deterministic_repeat"] == {
        "array_and_archive_byte_identity": True,
        "identical_element_records": True,
        "identical_runtime_evidence": True,
        "identical_table_sha256": "bc7b62eac01cc9d035257841d4533bcca504ecdb295e6665a5524b27e9db0319",
        "independent_execution_count": 2,
    }
    for execution in (audit["primary_execution"], audit["repeat_execution"]):
        assert (
            _sha256(ROOT / execution["manifest_path"]) == execution["manifest_sha256"]
        )
        assert _sha256(ROOT / execution["table_path"]) == execution["table_sha256"]
    assert audit["hard_constraints"]["solvation_accuracy_panel_run"] is False
