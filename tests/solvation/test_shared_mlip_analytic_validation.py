from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import validate_shared_mlip_analytic as validation


def test_matrix_keeps_every_named_cell_and_freezes_two_base_geometries():
    specs = validation.cell_specs()

    assert validation.GEOMETRIES == ("water", "methanol")
    assert {spec.model for spec in specs if spec.tier == "initial-cpu"} == {
        "ani2x",
        "aimnet2",
        "maceoff23m",
    }
    assert {spec.nonpolar for spec in specs} == {"ace", "none"}
    assert any(spec.model == "uma" and spec.variant == "uma-s-1p1" for spec in specs)
    assert any(spec.model == "uma" and spec.variant == "uma-s-1p2" for spec in specs)
    assert any(spec.device.startswith("cuda") for spec in specs)
    assert len({spec.profile_id for spec in specs}) == len(specs)


def test_matrix_keeps_every_original_named_mace_and_uma_variant():
    variants = {spec.variant for spec in validation.cell_specs()}

    assert {
        "maceoff23m.pt",
        "MACE-OFF23_small.model",
        "MACE-OFF23_medium.model",
        "MACE-OFF23_large.model",
        "MACE-OFF24_medium.model",
        "MACE-omol-0-extra-large-1024.model",
        "MACEPOLAR1Smodel",
        "MACEPOLAR1Mmodel",
        "MACEPOLAR1Lmodel",
        "uma-s-1p1",
        "uma-s-1p2",
        "uma-m-1p1",
    } <= variants


def test_three_direction_vectors_are_seeded_normalized_and_exactly_replayable():
    first = validation.normalized_directions(size=18, cell_index=2)
    second = validation.normalized_directions(size=18, cell_index=2)

    assert len(first) == validation.DIRECTION_COUNT == 3
    for (seed_a, vector_a), (seed_b, vector_b) in zip(first, second):
        assert seed_a == seed_b
        np.testing.assert_array_equal(vector_a, vector_b)
        assert np.isclose(np.linalg.norm(vector_a), 1.0, rtol=0.0, atol=1.0e-15)


def test_dtype_thresholds_freeze_native_steps_and_two_finest_pair_rule():
    assert validation.DTYPE_GATES["D64"]["force_fd_steps_angstrom"] == (
        1.0e-3,
        5.0e-4,
        2.5e-4,
    )
    assert validation.DTYPE_GATES["N32"]["force_fd_steps_angstrom"] == (
        2.0e-3,
        1.0e-3,
        5.0e-4,
    )
    assert validation.DTYPE_GATES["D64"]["force_fd_max"] == 2.0e-5
    assert validation.DTYPE_GATES["N32"]["force_fd_max"] == 2.0e-4


def test_partial_matrix_retains_rows_but_cannot_claim_release_pass():
    specs = validation.cell_specs()[:3]
    results = {
        specs[0].profile_id: validation.empty_cell_record(
            specs[0],
            status="QUALIFIED",
            checks={name: True for name in validation.REQUIRED_CHECKS},
            gas_profile={"component": "gas"},
            solvent_profile={"component": "solvent"},
            dependency_versions={"torch": "test"},
        ),
        specs[1].profile_id: validation.empty_cell_record(
            specs[1], status="FAILED", failures=["measured mismatch"]
        ),
    }
    artifact = validation.compile_matrix_report(specs=specs, results=results)

    assert artifact["artifact_type"] == "route1-analytic-cell-matrix"
    assert artifact["schema_version"] == 1
    assert artifact["passed"] is False
    assert len(artifact["cells"]) == 3
    assert [row["status"] for row in artifact["cells"]] == [
        "QUALIFIED",
        "FAILED",
        "NOT_EXECUTED",
    ]


def test_candidate_constructor_injects_only_the_private_factory_seam():
    captured = {}

    class FakeFactory:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    resolver = lambda gas, solvent: (gas, solvent)
    constructor = validation.candidate_factory_constructor(FakeFactory, resolver)
    constructor("cpu", "ani2x", "out.log", model_options={"hessian": "analytic"})

    assert captured["kwargs"].pop("_analytic_admission_resolver") is resolver
    assert captured["kwargs"] == {"model_options": {"hessian": "analytic"}}
    assert captured["args"] == ("cpu", "ani2x", "out.log")


def test_candidate_resolver_rejects_live_profile_drift():
    from maple.function.calculator.analytic_admission import cell_profile
    from maple.function.calculator.analytic_protocol import (
        AnalyticIdentityError,
        ScalarCellIdentity,
    )

    gas = ScalarCellIdentity(
        component="gas", implementation="test", model="test",
        checkpoint_path="/fixed/model.pt", checkpoint_sha256="a" * 64,
        device="cpu", effective_dtype="torch.float64", d4=False,
        source_energy_unit="hartree", cutoff_guard_policy=None,
        cutoff_guard_margin_angstrom=None,
    )
    solvent = ScalarCellIdentity(
        component="solvent", implementation="test", model="obc2",
        checkpoint_path=None, checkpoint_sha256=None, device="cpu",
        effective_dtype="torch.float64", d4=None, source_energy_unit="hartree",
        cutoff_guard_policy=None, cutoff_guard_margin_angstrom=None,
    )

    resolver = validation.exact_candidate_resolver(
        expected_gas_profile=cell_profile(gas),
        expected_solvent_profile=cell_profile(solvent),
        profile_id="CANDIDATE:test",
    )

    with np.testing.assert_raises_regex(AnalyticIdentityError, "predeclared"):
        resolver(gas, replace(solvent, model="drift"))


def test_installable_receipt_contains_exactly_one_fully_qualified_cell():
    specs = validation.cell_specs()[:1]
    results = {
        spec.profile_id: validation.empty_cell_record(
            spec,
            status="QUALIFIED",
            checks={name: True for name in validation.REQUIRED_CHECKS},
            gas_profile={"component": "gas", "model": spec.model},
            solvent_profile={"component": "solvent", "nonpolar": spec.nonpolar},
            dependency_versions={"torch": "test"},
        )
        for spec in specs
    }
    artifact = validation.compile_artifact(
        specs=specs,
        results=results,
        implementation_files_sha256={"maple/function/calculator/calculator_base.py": "a" * 64},
        gates={
            "solvent_validation": True,
            "isolated_install": True,
            "candidate_workflows": True,
        },
        exhaustive_matrix_sha256="b" * 64,
        validation_files_sha256={"tests/solvation/test.py": "c" * 64},
    )

    assert artifact["passed"] is True
    assert artifact["all_cells_qualified"] is True
    assert len(artifact["cells"]) == 1
    assert artifact["exhaustive_matrix_sha256"] == "b" * 64
    assert artifact["scope"] == "exact-qualified-cells-not-family-or-matrix"
    assert artifact["validation_files_sha256"] == {
        "tests/solvation/test.py": "c" * 64
    }


def test_installable_receipt_rejects_missing_exhaustive_matrix_binding():
    spec = validation.cell_specs()[0]

    with np.testing.assert_raises_regex(ValueError, "exhaustive matrix"):
        validation.compile_artifact(
            specs=[spec],
            results={},
            implementation_files_sha256={"maple/x.py": "a" * 64},
            gates={name: True for name in validation.REQUIRED_GATES},
            exhaustive_matrix_sha256="",
            validation_files_sha256={},
        )


def test_installable_receipt_rejects_nonqualified_or_multiple_cells():
    specs = validation.cell_specs()[:2]
    qualified = validation.empty_cell_record(
        specs[0], status="QUALIFIED",
        checks={name: True for name in validation.REQUIRED_CHECKS},
        gas_profile={"component": "gas"}, solvent_profile={"component": "solvent"},
        dependency_versions={"torch": "test"},
    )
    common = {
        "implementation_files_sha256": {"maple/x.py": "a" * 64},
        "validation_files_sha256": {"tests/x.py": "c" * 64},
        "gates": {name: True for name in validation.REQUIRED_GATES},
        "exhaustive_matrix_sha256": "b" * 64,
    }

    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        validation.compile_artifact(
            specs=specs, results={specs[0].profile_id: qualified}, **common
        )
    with np.testing.assert_raises_regex(ValueError, "fully qualified"):
        validation.compile_artifact(specs=specs[:1], results={}, **common)


def test_exhaustive_matrix_report_retains_nonqualified_terminal_evidence():
    specs = validation.cell_specs()[:3]
    results = {
        specs[0].profile_id: validation.empty_cell_record(specs[0], status="FAILED"),
        specs[1].profile_id: validation.empty_cell_record(
            specs[1], status="UNAVAILABLE", failures=["CUDA device absent"]
        ),
        specs[2].profile_id: validation.empty_cell_record(
            specs[2], status="FEASIBILITY_BLOCKED", failures=["graph unavailable"]
        ),
    }

    report = validation.compile_matrix_report(specs=specs, results=results)

    assert report["artifact_type"] == "route1-analytic-cell-matrix"
    assert report["passed"] is False
    assert [row["status"] for row in report["cells"]] == [
        "FAILED",
        "UNAVAILABLE",
        "FEASIBILITY_BLOCKED",
    ]


def test_source_manifest_excludes_only_activation_allowlist_and_evidence_data():
    paths = validation.implementation_paths()
    validation_paths = validation.validation_paths()

    assert "maple/function/calculator/_qualified_analytic_cells.py" not in paths
    assert not any(path.startswith(".omx/") for path in paths)
    assert not any(path.startswith("maple/function/calculator/data/") for path in paths)
    assert any("extra_correction/implicit/data/" in path for path in paths)
    assert "maple/function/calculator/set_calculator.py" in paths
    assert "maple/function/calculator/analytic_admission.py" in paths
    assert "maple/function/read/command_control.py" in paths
    assert "maple/function/engine.py" in paths
    assert any(path.startswith("maple/function/dispatcher/frequency/") for path in paths)
    assert any(path.startswith("maple/function/dispatcher/ts/") for path in paths)
    assert "docs/implicit-solvation/benchmarks/validate_shared_mlip_analytic.py" not in paths
    assert "docs/implicit-solvation/benchmarks/validate_shared_mlip_analytic.py" in validation_paths
    assert "tests/solvation/run_no_openmm_install.py" in validation_paths
    assert "pyproject.toml" in validation_paths
