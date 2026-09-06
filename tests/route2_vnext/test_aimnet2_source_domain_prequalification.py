from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "route2-aimnet2-mnsol653-source-domain-prequalification-protocol-v1.json"
)
RUNNER = (
    ROOT
    / "tools"
    / "route2_release"
    / "run_aimnet2_mnsol653_source_domain_prequalification.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("aimnet2_domain_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_artifact(runner):
    runtime_provenance = {
        "runtime_kind": "aimnet-reconstructed-float64-runtime-v4",
        "aimnet_package_version": "0.2.0",
        "aimnet_runtime_files_sha256": {"aimnet/models/base.py": "a" * 64},
        "package_versions": {"torch": "synthetic"},
    }
    numerical_runtime = {
        "python_implementation": "CPython",
        "python_version": "3.11.0",
        "numpy_version": "synthetic",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "threadpools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "num_threads": 1,
                "prefix": "libopenblas",
                "version": "synthetic",
                "threading_layer": "pthreads",
                "architecture": "synthetic",
                "library_path": "/synthetic/libopenblas.so",
                "library_sha256": "c" * 64,
            }
        ],
        "torch_version": "synthetic",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    selected = [
        {
            "ordinal": 0,
            "selection_index": 7,
            "partition": "development",
            "opaque_record_id": "1" * 64,
            "geometry_handle": "synthetic",
            "geometry_sha256": "2" * 64,
            "atom_count": 2,
            "atomic_numbers": [1, 6],
        }
    ]
    parity = {
        "energy_absolute_error_eV": 0.0,
        "charge_max_absolute_error_e": 0.0,
        "intrinsic_gradient_max_absolute_error_eV_per_A": 0.0,
        "charge_vjp_max_absolute_error_eV_per_A": 0.0,
        "repeat_energy_absolute_error_eV": 0.0,
        "repeat_charge_max_absolute_error_e": 0.0,
        "repeat_intrinsic_gradient_max_absolute_error_eV_per_A": 0.0,
        "repeat_charge_vjp_max_absolute_error_eV_per_A": 0.0,
        "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A": 0.0,
    }
    row = {
        **selected[0],
        "status": "measured",
        "energy_eV": -1.0,
        "raw_charges_e": [0.1, -0.1],
        "projected_charges_e": [0.1, -0.1],
        "raw_charge_residual_e": 0.0,
        "projected_charge_sum_e": 0.0,
        "intrinsic_gradient_norm_eV_per_A": 0.2,
        "charge_vjp_norm_eV_per_A": 0.1,
        "parity": parity,
        "gate_passed": True,
        "runtime_seconds": 1.0,
    }
    bindings = {
        "execution_git_commit": "3" * 40,
        "execution_git_tree": "4" * 40,
        "protocol_file_sha256": "5" * 64,
        "mnsol_dataset_zip_sha256": "6" * 64,
        "mnsol_protocol_sha256": "7" * 64,
        "prior_private_653_sha256": "8" * 64,
        "selection_manifest_file_sha256": "9" * 64,
        "selection_manifest_canonical_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_bytes": 1234,
        "runtime_kind": "aimnet-reconstructed-float64-runtime-v4",
        "runtime_provenance_sha256": runner.canonical_sha256(runtime_provenance),
        "numerical_runtime_sha256": runner.canonical_sha256(numerical_runtime),
        "covered_atomic_numbers": [1, 6],
        "selected_records": selected,
        "thresholds": runner._thresholds(),
        "claim_boundary": "synthetic source-domain boundary",
    }
    artifact = {
        "artifact": runner.ARTIFACT,
        "schema_version": 1,
        "status": "complete",
        "do_not_commit": True,
        **{
            key: copy.deepcopy(value)
            for key, value in bindings.items()
            if key != "selected_records"
        },
        "runtime_provenance": runtime_provenance,
        "numerical_runtime": numerical_runtime,
        "record_count": 1,
        "passed_record_count": 1,
        "failed_record_count": 0,
        "all_gates_passed": True,
        "rows": [row],
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
    }
    artifact["artifact_sha256"] = runner.canonical_sha256(artifact)
    return artifact, bindings


def test_source_domain_protocol_is_target_free_and_cannot_admit_capabilities():
    runner = _runner()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["status"] == "preregistered-not-executed"
    assert protocol["runtime"]["kind"] == "aimnet-reconstructed-float64-runtime-v4"
    assert protocol["selection"]["expected_atomic_numbers"] == [
        1,
        6,
        7,
        8,
        9,
        15,
        16,
        17,
        35,
        53,
    ]
    assert protocol["selection"]["expected_geometry_count"] == 15
    assert protocol["selection"]["selection_reads_experimental_targets"] is False
    assert (
        protocol["selection"]["selection_reads_aimnet2_or_continuum_outputs"] is False
    )
    assert (
        protocol["execution_contract"][
            "no_domain_widening_before_a_passing_hash_bound_artifact"
        ]
        is True
    )
    assert not any(protocol["capabilities"].values())
    assert protocol["claim_boundary"] == runner.SOURCE_DOMAIN_CLAIM_BOUNDARY
    assert protocol["execution_contract"] == runner.SOURCE_DOMAIN_EXECUTION_CONTRACT


def test_selector_completes_missing_element_by_geometry_hash_without_targets():
    runner = _runner()
    geometries = {
        "base-a": SimpleNamespace(sha256="b" * 64, atomic_numbers=(1, 6)),
        "base-b": SimpleNamespace(sha256="c" * 64, atomic_numbers=(1, 8)),
        "br-late": SimpleNamespace(sha256="f" * 64, atomic_numbers=(1, 6, 35)),
        "br-first": SimpleNamespace(sha256="a" * 64, atomic_numbers=(1, 35)),
    }
    dataset = SimpleNamespace(geometries=geometries)
    prior = [
        {
            "opaque_record_id": "01",
            "geometry_handle": "base-a",
            "geometry_sha256": "b" * 64,
        },
        {
            "opaque_record_id": "02",
            "geometry_handle": "base-b",
            "geometry_sha256": "c" * 64,
        },
        {
            "opaque_record_id": "03",
            "geometry_handle": "br-late",
            "geometry_sha256": "f" * 64,
        },
        {
            "opaque_record_id": "04",
            "geometry_handle": "br-first",
            "geometry_sha256": "a" * 64,
        },
    ]
    manifest = [
        {"opaque_record_id": "01", "geometry_sha256": "b" * 64},
        {"opaque_record_id": "02", "geometry_sha256": "c" * 64},
        {"opaque_record_id": "01", "geometry_sha256": "b" * 64},
    ]

    selected, atomic_numbers = runner.select_audit_geometries(
        dataset=dataset,
        prior_records=prior,
        manifest_records=manifest,
    )

    assert [record["geometry_handle"] for record in selected] == [
        "base-a",
        "base-b",
        "br-first",
    ]
    assert atomic_numbers == (1, 6, 8, 35)


@pytest.mark.parametrize("extra_candidate", (False, True))
def test_selector_one_geometry_can_complete_multiple_missing_elements(extra_candidate):
    runner = _runner()
    geometries = {
        "base": SimpleNamespace(sha256="a" * 64, atomic_numbers=(1, 6)),
        "shared": SimpleNamespace(sha256="b" * 64, atomic_numbers=(1, 17, 35)),
    }
    if extra_candidate:
        geometries["redundant"] = SimpleNamespace(
            sha256="c" * 64, atomic_numbers=(1, 35)
        )
    prior = [
        {
            "opaque_record_id": name,
            "geometry_handle": name,
            "geometry_sha256": geometry.sha256,
        }
        for name, geometry in geometries.items()
    ]
    selected, atomic_numbers = runner.select_audit_geometries(
        dataset=SimpleNamespace(geometries=geometries),
        prior_records=prior,
        manifest_records=[prior[0]],
    )
    assert [record["geometry_handle"] for record in selected] == ["base", "shared"]
    assert atomic_numbers == (1, 6, 17, 35)


def test_source_domain_reducer_hard_gates_repeat_but_not_embedded_d3_report():
    runner = _runner()
    parity = {
        "energy_absolute_error_eV": 0.0,
        "charge_max_absolute_error_e": 0.0,
        "charge_vjp_max_absolute_error_eV_per_A": 0.0,
        "repeat_energy_absolute_error_eV": 0.0,
        "repeat_charge_max_absolute_error_e": 0.0,
        "repeat_intrinsic_gradient_max_absolute_error_eV_per_A": 0.0,
        "repeat_charge_vjp_max_absolute_error_eV_per_A": 0.0,
        "ordinary_forward_intrinsic_gradient_report_only_max_absolute_error_eV_per_A": 1.0,
    }
    row = {
        "status": "measured",
        "raw_charge_residual_e": 0.0,
        "projected_charge_sum_e": 0.0,
        "parity": parity,
    }

    assert runner._row_passes(row, runner._thresholds()) is True
    parity["repeat_intrinsic_gradient_max_absolute_error_eV_per_A"] = 2.0e-7
    assert runner._row_passes(row, runner._thresholds()) is False


def test_source_domain_runner_parser_constructs():
    runner = _runner()
    parser = runner._parser()

    assert all(action.dest != "overwrite" for action in parser._actions)


def test_source_domain_runtime_identity_binds_external_runtime_files():
    runner = _runner()

    class Calculator:
        def __init__(self):
            self.payload = {
                "runtime_kind": "aimnet-reconstructed-float64-runtime-v4",
                "aimnet_package_version": "0.2.0",
                "aimnet_runtime_files_sha256": {"aimnet/models/base.py": "a" * 64},
                "package_versions": {"torch": "synthetic"},
            }

        def runtime_provenance(self):
            return self.payload

    calculator = Calculator()
    payload, first = runner._runtime_identity(calculator)
    calculator.payload["aimnet_runtime_files_sha256"]["aimnet/models/base.py"] = (
        "b" * 64
    )
    _changed_payload, second = runner._runtime_identity(calculator)

    assert payload["aimnet_package_version"] == "0.2.0"
    assert first != second


def test_source_domain_runtime_identity_rejects_nonfinite_metadata():
    runner = _runner()
    calculator = SimpleNamespace(
        runtime_provenance=lambda: {
            "runtime_kind": "aimnet-reconstructed-float64-runtime-v4",
            "bad": float("nan"),
        }
    )

    with pytest.raises(ValueError, match="finite JSON metadata"):
        runner._runtime_identity(calculator)


def test_source_domain_private_path_is_protected_by_tracked_gitignore():
    runner = _runner()
    path = ROOT / ".omx" / "benchmarks" / "source-private-path-test.json"

    assert runner._private_output(path) == path.resolve()
    with pytest.raises(ValueError, match="below repository .omx"):
        runner._private_output(ROOT / "source-private-path-test.json")


def test_source_domain_artifact_verifier_binds_runtime_and_rows():
    runner = _runner()
    artifact, bindings = _synthetic_artifact(runner)

    runner.validate_source_domain_artifact(
        artifact,
        expected_bindings=bindings,
    )


def test_source_domain_artifact_rejects_runtime_provenance_tamper():
    runner = _runner()
    artifact, bindings = _synthetic_artifact(runner)
    artifact["runtime_provenance"]["aimnet_runtime_files_sha256"][
        "aimnet/models/base.py"
    ] = ("c" * 64)
    artifact["artifact_sha256"] = runner.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="runtime provenance identity"):
        runner.validate_source_domain_artifact(
            artifact,
            expected_bindings=bindings,
        )


def test_source_domain_artifact_rejects_numerical_runtime_tamper():
    runner = _runner()
    artifact, bindings = _synthetic_artifact(runner)
    artifact["numerical_runtime"]["threadpools"][0]["num_threads"] = 2
    artifact["artifact_sha256"] = runner.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="numerical runtime identity"):
        runner.validate_source_domain_artifact(
            artifact,
            expected_bindings=bindings,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("energy_eV", float("nan")),
        ("intrinsic_gradient_norm_eV_per_A", -1.0),
        ("charge_vjp_norm_eV_per_A", -1.0),
    ),
)
def test_source_domain_artifact_rejects_invalid_response_scalars(field, value):
    runner = _runner()
    artifact, bindings = _synthetic_artifact(runner)
    artifact["rows"][0][field] = value

    with pytest.raises(ValueError, match="response scalars are invalid"):
        runner.validate_source_domain_artifact(
            artifact,
            expected_bindings=bindings,
        )


def test_source_domain_artifact_rejects_negative_parity_error():
    runner = _runner()
    artifact, bindings = _synthetic_artifact(runner)
    artifact["rows"][0]["parity"]["repeat_energy_absolute_error_eV"] = -1.0
    artifact["artifact_sha256"] = runner.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="parity ledger is malformed"):
        runner.validate_source_domain_artifact(
            artifact,
            expected_bindings=bindings,
        )


def test_source_domain_artifact_rejects_non_affine_charge_projection():
    runner = _runner()
    artifact, bindings = _synthetic_artifact(runner)
    artifact["rows"][0]["projected_charges_e"] = [0.2, -0.2]
    artifact["artifact_sha256"] = runner.canonical_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="affine charge projection"):
        runner.validate_source_domain_artifact(
            artifact,
            expected_bindings=bindings,
        )
