from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from maple.solvation.release.admission import (
    CLAIM_BOUNDARY_ID,
    COMPUTATION_SEAL_V2_SCHEMA,
    EXPOSURE_AWARE_PROTOCOL_LABEL,
    H1_V2_CAVITY_PROFILE_ID,
    H1_V2_CONTINUUM_PROFILE_ID,
    H1_V2_DISPLACEMENT_POLICY,
    H1_V2_EXACT_SCALAR,
    H1_V2_EXCLUDED_COMPONENTS,
    H1_V2_FORCE_DERIVATIVE,
    H1_V2_INCLUDED_COMPONENTS,
    H1_V2_INDUCED_SOURCE_KERNEL,
    H1_V2_LONG_RANGE_EVALUATOR_ID,
    H1_V2_ENDPOINT_REPLAY_SCOPE,
    H1_V2_PERMANENT_SOURCE_KERNEL,
    H1_V2_PROFILE_ID,
    H1_V2_RADII_PROVIDER_ID,
    H1_V2_RECEIVER_KERNEL,
    H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
    H1_V2_RECEIVER_SPACE_ID,
    H1_V2_ROOT_METHOD,
    H1_V2_SCALAR_ID,
    H1_V2_SECOND_START,
    H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
    H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
    H1_V2_STATE_ID,
    H1_V2_TOPOLOGY_POLICY,
    REQUIRED_DOMAIN_GUARDS,
    REQUIRED_NON_ADMISSIONS,
    REQUIRED_RUNTIME_GUARDS,
    REPLICATE_ADMISSION_RECORD_V2_SCHEMA,
    V1_ADMISSION_GATE_NAMES,
    ComputationSealV2,
    verified_pro_schema_amendment_contract,
)
from maple.solvation.release.evidence import canonical_json_sha256
from maple.solvation.release.evidence import RepositorySnapshot
import tools.route2_release.aggregate_mace_mdp_polar_harmonic_ef_revalidation as aggregator

SCRIPT = (
    Path(__file__).parents[2]
    / "tools/route2_release/aggregate_mace_mdp_polar_harmonic_ef_revalidation.py"
)
REAL_PREREGISTRATION = Path(__file__).parents[2] / (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-ef-revalidation-v2.json"
)
RUNNER_ID = "tools/route2_release/run_mace_mdp_polar_harmonic_ef_revalidation.py"
ADMISSION_ID = "maple/solvation/release/admission.py"
EVIDENCE_ID = "maple/solvation/release/evidence.py"
CAPABILITIES_ID = "maple/solvation/api/capabilities.py"
SYNTHETIC_EXECUTION_SOURCES = (
    aggregator.AGGREGATOR_ID,
    ADMISSION_ID,
    EVIDENCE_ID,
    CAPABILITIES_ID,
)
SHA_A, SHA_B, SHA_C, SHA_D = (character * 64 for character in "abcd")
REAL_EXECUTION_SOURCE_RELATIVES = aggregator._execution_source_relatives


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rehash(payload: dict[str, object], field: str) -> None:
    payload[field] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _science() -> dict[str, object]:
    return {
        "continuum": {
            "profile_id": H1_V2_CONTINUUM_PROFILE_ID,
            "transition_width_angstrom2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "source_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
        },
        "cavity": {
            "profile_id": H1_V2_CAVITY_PROFILE_ID,
            "radii_provider_id": H1_V2_RADII_PROVIDER_ID,
        },
        "source_receiver": {
            "permanent_source_kernel": H1_V2_PERMANENT_SOURCE_KERNEL,
            "induced_source_kernel": H1_V2_INDUCED_SOURCE_KERNEL,
            "receiver_kernel": H1_V2_RECEIVER_KERNEL,
            "long_range_evaluator_id": H1_V2_LONG_RANGE_EVALUATOR_ID,
            "source_space_id": H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
            "source_space_contract_sha256": H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            "receiver_space_id": H1_V2_RECEIVER_SPACE_ID,
            "receiver_space_contract_sha256": H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
        },
        "energy_ledger": {
            "exact_scalar": H1_V2_EXACT_SCALAR,
            "included_components": list(H1_V2_INCLUDED_COMPONENTS),
            "excluded_components": list(H1_V2_EXCLUDED_COMPONENTS),
        },
        "root_algorithm": {
            "method": H1_V2_ROOT_METHOD,
            "tolerance_ev": 1.0e-10,
            "maximum_iterations": 40,
            "second_start": H1_V2_SECOND_START,
            "total_charge_tolerance_e": 1.0e-8,
            "multi_start_field_tolerance_ev": 2.0e-9,
            "multi_start_energy_tolerance_ev": 1.0e-10,
            "evidence_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
        },
        "force_stencil": {
            "derivative": H1_V2_FORCE_DERIVATIVE,
            "coarse_step_angstrom": 5.0e-4,
            "fine_step_angstrom": 2.5e-4,
            "independent_step_angstrom": 1.25e-4,
            "maximum_local_error_ev_per_angstrom": 2.0e-4,
            "topology_policy": H1_V2_TOPOLOGY_POLICY,
            "displacement_policy": H1_V2_DISPLACEMENT_POLICY,
        },
    }


def _runtime() -> dict[str, object]:
    return {
        "python": "3.11.13",
        "implementation": "CPython",
        "platform": "Linux-test",
        "machine": "x86_64",
        "cpu_model": "Test CPU",
        "packages": {
            "maple": None,
            "ase": "3.26.0",
            "numpy": "2.1.0",
            "scipy": "1.15.0",
            "torch": "2.7.0",
            "mace-torch": "0.3.14",
            "graph-longrange": "0.1.0",
            "pyscf": None,
            "pyddx": "0.6.0",
        },
        "numpy": {"version": "2.1.0", "show_config": "blas=openblas"},
        "torch": {
            "version": "2.7.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "cuda_available": True,
            "devices": ["Test GPU"],
            "default_dtype": "torch.float64",
            "threads": 1,
            "interop_threads": 1,
            "driver_version": "555.42.02",
            "device_uuids": ["GPU-33333333-3333-4333-8333-333333333333"],
            "device_capabilities": ["8.9"],
            "device_multiprocessor_counts": [128],
            "deterministic_algorithms_enabled": True,
            "deterministic_debug_mode": 2,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_LAUNCH_BLOCKING": None,
            "PYTHONHASHSEED": "0",
        },
        "execution_device": "cuda",
        "execution_dtype": "float64",
    }


class Fixture:
    def __init__(self, root: Path, external: Path) -> None:
        self.root = root
        self.external = external
        self.prereg = root / "docs/preregistration-v2.json"
        self.seal = external / "seal.json"
        self.replicate_a = external / "replicate-a.json"
        self.replicate_b = external / "replicate-b.json"
        self.output = external / "aggregate.json"

    def arguments(self) -> tuple[Path, Path, Path, Path, Path]:
        return self.root, self.prereg, self.seal, self.replicate_a, self.replicate_b


@pytest.fixture
def valid_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.setattr(
        aggregator,
        "_execution_source_relatives",
        lambda repository: SYNTHETIC_EXECUTION_SOURCES,
    )
    root = tmp_path / "repository"
    external = tmp_path / "external"
    (root / "tools/route2_release").mkdir(parents=True)
    (root / "docs").mkdir()
    external.mkdir()
    aggregator_bytes = SCRIPT.read_bytes()
    runner_bytes = b"#!/usr/bin/env python3\n# frozen synthetic runner\n"
    (root / aggregator.AGGREGATOR_ID).write_bytes(aggregator_bytes)
    (root / RUNNER_ID).write_bytes(runner_bytes)
    execution_module_bytes = {
        ADMISSION_ID: Path(aggregator.admission_module.__file__).read_bytes(),
        EVIDENCE_ID: Path(aggregator.evidence_module.__file__).read_bytes(),
        CAPABILITIES_ID: Path(aggregator.capabilities_module.__file__).read_bytes(),
    }
    for relative, content in execution_module_bytes.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    prereg_payload = {
        "schema_version": aggregator.PREREGISTRATION_SCHEMA,
        "artifact_id": aggregator.PREREGISTRATION_ARTIFACT_ID,
        "status": aggregator.PREREGISTRATION_STATUS,
        "protocol_classification": aggregator.PREREGISTRATION_CLASSIFICATION,
        "prospective_h1_identity_contract": {
            "profile_id": H1_V2_PROFILE_ID,
            "scalar_id": H1_V2_SCALAR_ID,
            "state_id": H1_V2_STATE_ID,
            "claim_boundary_id": CLAIM_BOUNDARY_ID,
            "lifecycle": aggregator.IDENTITY_LIFECYCLE,
        },
        "target_capabilities": aggregator.EF_CAPABILITIES,
        "prospective_h1_scientific_settings": _science(),
        "runtime_guards": list(REQUIRED_RUNTIME_GUARDS),
        "domain_guards": list(REQUIRED_DOMAIN_GUARDS),
        "claim_boundary": {
            "id": CLAIM_BOUNDARY_ID,
            **aggregator.CLAIM_BOUNDARY_TEXT,
            "non_admissions": list(REQUIRED_NON_ADMISSIONS),
        },
        "frozen_execution_sequence": aggregator.FROZEN_EXECUTION_SEQUENCE,
        "sequence_invariants": aggregator.SEQUENCE_INVARIANTS,
        "post_freeze_prohibitions": aggregator.POST_FREEZE_PROHIBITIONS,
        "execution_state": aggregator.EXECUTION_STATE,
        "verified_pro_schema_amendment": verified_pro_schema_amendment_contract(),
        "parent_bindings": {
            "v1_preregistration": {
                "artifact_id": "synthetic-v1-preregistration",
                "relative_path": "docs/synthetic-v1-preregistration.json",
                "sha256": SHA_D,
            },
            "v1_parent_panel": {
                "artifact_id": "synthetic-v1-panel",
                "relative_path": "docs/synthetic-v1-panel.json",
                "sha256": SHA_C,
                "record_count": 4,
                "record_order": ["a", "b", "c", "d"],
            },
        },
    }
    prereg_bytes = (json.dumps(prereg_payload, sort_keys=True) + "\n").encode()
    prereg = root / "docs/preregistration-v2.json"
    prereg.write_bytes(prereg_bytes)
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(
        ["git", "-C", root, "config", "user.email", "test@example.test"], check=True
    )
    subprocess.run(
        ["git", "-C", root, "config", "user.name", "Aggregator Test"], check=True
    )
    subprocess.run(["git", "-C", root, "add", "."], check=True)
    subprocess.run(["git", "-C", root, "commit", "-qm", "frozen fixture"], check=True)
    head = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    source_ledger = {
        aggregator.AGGREGATOR_ID: _sha(aggregator_bytes),
        RUNNER_ID: _sha(runner_bytes),
        **{
            relative: _sha(content)
            for relative, content in execution_module_bytes.items()
        },
    }
    assets = {
        "benzene_mol2": SHA_D,
        "benzene_projection_result": SHA_A,
        "parent_panel": SHA_C,
        "v1_preregistration": SHA_D,
        "verified_pro_audit": (
            verified_pro_schema_amendment_contract()["verified_pro_audit"][
                "raw_file_sha256"
            ]
        ),
        "verified_pro_prompt": (
            "2834f81f9cbfa31538972539754a5f0f5478d6c563278fe456b539dd0db7df0e"
        ),
        "verified_pro_response": (
            "252323ec9b93fbefd6c77e9181addb5e2269ca73031497217f5c39d60ae6fbb2"
        ),
        "verified_pro_mode_verification": (
            "df55961b70f52a12b765cdffb354f3e0687da7277bc1050d52a8473e59d72583"
        ),
    }
    seal_payload: dict[str, object] = {
        "schema_id": COMPUTATION_SEAL_V2_SCHEMA,
        "artifact_id": "synthetic-h1-v2-computation-seal",
        "profile_id": H1_V2_PROFILE_ID,
        "scalar_id": H1_V2_SCALAR_ID,
        "state_id": H1_V2_STATE_ID,
        "git_head": head,
        "git_tree": tree,
        "git_clean": True,
        "preregistration_id": prereg_payload["artifact_id"],
        "preregistration_sha256": _sha(prereg_bytes),
        "runner_id": RUNNER_ID,
        "runner_sha256": source_ledger[RUNNER_ID],
        "aggregator_id": aggregator.AGGREGATOR_ID,
        "aggregator_sha256": source_ledger[aggregator.AGGREGATOR_ID],
        "computation_source_sha256s": source_ledger,
        "asset_sha256s": assets,
        "checkpoint_sha256s": {
            "mace_mdp_checkpoint": SHA_A,
            "mace_polar_checkpoint": SHA_B,
        },
        "provider_configuration_sha256s": {
            "hybrid_configuration": SHA_A,
            "hybrid_provenance": SHA_B,
            "continuum_configuration": SHA_C,
            "continuum_provenance": SHA_D,
            "cavity_configuration": SHA_A,
                "prepared_input_manifest": SHA_B,
                "force_panel_contract": SHA_C,
                "benzene_pes_configuration": "e" * 64,
                "water_pes_configuration": "f" * 64,
            },
        "scientific_settings": _science(),
        "runtime_fingerprint": _runtime(),
        "runtime_guards": list(REQUIRED_RUNTIME_GUARDS),
        "domain_guards": list(REQUIRED_DOMAIN_GUARDS),
        "exposure_aware_protocol_label": EXPOSURE_AWARE_PROTOCOL_LABEL,
        "claim_boundary_id": CLAIM_BOUNDARY_ID,
        "non_admissions": list(REQUIRED_NON_ADMISSIONS),
    }
    _rehash(seal_payload, "content_sha256")
    seal = ComputationSealV2.from_mapping(seal_payload)
    (external / "seal.json").write_text(json.dumps(seal_payload), encoding="utf-8")
    for label, uuid, timestamp in (
        ("a", "11111111-1111-4111-8111-111111111111", "2026-08-23T01:00:00+00:00"),
        ("b", "22222222-2222-4222-8222-222222222222", "2026-08-23T02:00:00+00:00"),
    ):
        payload: dict[str, object] = {
            "schema_id": REPLICATE_ADMISSION_RECORD_V2_SCHEMA,
            "artifact_id": f"synthetic-replicate-{label}",
            "label": label,
            "process_uuid": uuid,
            "process_started_at_utc": timestamp,
            "seal_id": seal.artifact_id,
            "seal_sha256": seal.content_sha256,
            "git_head": seal.git_head,
            "git_tree": seal.git_tree,
            "source_ledger_sha256": seal.source_ledger_sha256,
            "asset_ledger_sha256": seal.asset_ledger_sha256,
            "runtime_fingerprint_sha256": seal.runtime_fingerprint_sha256,
            "measurement_sha256": "f" * 64,
            "gate_results": {name: True for name in V1_ADMISSION_GATE_NAMES},
        }
        _rehash(payload, "artifact_sha256")
        (external / f"replicate-{label}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return Fixture(root, external)


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(valid_fixture: Fixture) -> ComputationSealV2:
    return ComputationSealV2.from_mapping(_payload(valid_fixture.seal))


def _main_arguments(valid_fixture: Fixture) -> list[str]:
    return [
        "--repo-root",
        str(valid_fixture.root),
        "--preregistration",
        str(valid_fixture.prereg),
        "--seal",
        str(valid_fixture.seal),
        "--replicate-a",
        str(valid_fixture.replicate_a),
        "--replicate-b",
        str(valid_fixture.replicate_b),
        "--output",
        str(valid_fixture.output),
    ]


def _publication_stability(
    valid_fixture: Fixture, *input_paths: Path
) -> aggregator._Stability:
    repository = RepositorySnapshot.capture(valid_fixture.root)
    captures = tuple(
        aggregator._read_json_once(path, role=f"test input {index}")
        for index, path in enumerate(input_paths)
    )
    return aggregator._Stability(repository, captures)


def _publish_test_artifact(
    valid_fixture: Fixture,
    *,
    stability: aggregator._Stability | None = None,
) -> bytes:
    encoded = b'{"content_sha256":"test-publication"}\n'
    aggregator._publish_external_output(
        valid_fixture.output,
        encoded,
        stability=(
            stability
            if stability is not None
            else _publication_stability(valid_fixture, valid_fixture.seal)
        ),
    )
    return encoded


def test_rejects_replicate_a_that_does_not_start_strictly_before_b() -> None:
    with pytest.raises(aggregator.AggregationInputError, match="strictly before"):
        aggregator._validate_process_start_order(
            "2026-08-23T01:00:00+00:00",
            "2026-08-23T00:59:59+00:00",
        )


def test_accepts_strictly_ordered_replicate_process_starts() -> None:
    aggregator._validate_process_start_order(
        "2026-08-23T01:00:00+00:00",
        "2026-08-23T02:00:00+00:00",
    )


@pytest.mark.parametrize(
    "field,mutation",
    [
        (
            "prospective_h1_scientific_settings",
            lambda payload: payload["continuum"].__setitem__("surface_lmax", 2),
        ),
        ("frozen_execution_sequence", lambda payload: payload.reverse()),
        (
            "sequence_invariants",
            lambda payload: payload.__setitem__(
                "mechanical_aggregator_has_no_discretionary_thresholds", False
            ),
        ),
        (
            "post_freeze_prohibitions",
            lambda payload: payload.__setitem__("fit", True),
        ),
        (
            "execution_state",
            lambda payload: payload.__setitem__("aggregate_result", {}),
        ),
    ],
    ids=("science", "sequence", "invariant", "prohibition", "state"),
)
def test_preregistration_contract_is_exact(
    valid_fixture: Fixture, field: str, mutation
) -> None:
    payload = _payload(valid_fixture.prereg)
    changed = copy.deepcopy(payload[field])
    mutation(changed)
    payload[field] = changed
    with pytest.raises(aggregator.AggregationInputError, match=f"{field} differs"):
        aggregator._validate_preregistration_contract(
            payload, seal=_seal(valid_fixture)
        )


def test_real_preregistration_artifact_matches_exact_aggregator_contract(
    valid_fixture: Fixture,
) -> None:
    preregistration_bytes = REAL_PREREGISTRATION.read_bytes()
    preregistration = json.loads(preregistration_bytes)
    seal_payload = _payload(valid_fixture.seal)
    actual_sha256 = _sha(preregistration_bytes)
    seal_payload["preregistration_id"] = preregistration["artifact_id"]
    seal_payload["preregistration_sha256"] = actual_sha256
    seal_payload["asset_sha256s"]["v1_preregistration"] = preregistration[
        "parent_bindings"
    ]["v1_preregistration"]["sha256"]
    seal_payload["asset_sha256s"]["parent_panel"] = preregistration["parent_bindings"][
        "v1_parent_panel"
    ]["sha256"]
    _rehash(seal_payload, "content_sha256")
    seal = ComputationSealV2.from_mapping(seal_payload)

    aggregator._validate_preregistration_contract(preregistration, seal=seal)


def test_typed_aggregate_includes_only_the_frozen_mechanical_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def from_mapping(payload, *, seal):
        captured.update(payload)
        return payload

    monkeypatch.setattr(
        aggregator,
        "AggregateAdmissionInputsV2",
        SimpleNamespace(from_mapping=from_mapping),
    )
    seal = SimpleNamespace(
        artifact_id="seal",
        content_sha256="a" * 64,
        profile_id="profile",
        scalar_id="scalar",
        state_id="state",
        runtime_guards=("runtime",),
        domain_guards=("domain",),
        claim_boundary_id="claim",
        non_admissions=("non-admission",),
    )
    replicate_a = SimpleNamespace(as_dict=lambda: {"label": "a"})
    replicate_b = SimpleNamespace(as_dict=lambda: {"label": "b"})
    aggregate = aggregator._typed_aggregate(seal, replicate_a, replicate_b)
    assert aggregate["aggregate_gate_results"] == {
        "two_independent_processes_same_measurement_sha256": True
    }
    assert captured == aggregate


@pytest.mark.parametrize(
    "asset_key,error",
    (
        ("v1_preregistration", "v1_preregistration asset differs"),
        ("parent_panel", "parent_panel asset differs"),
    ),
)
def test_rejects_parent_asset_mismatch_against_protocol_bindings(
    valid_fixture: Fixture, asset_key: str, error: str
) -> None:
    seal_payload = _payload(valid_fixture.seal)
    seal_payload["asset_sha256s"][asset_key] = "0" * 64
    _rehash(seal_payload, "content_sha256")
    seal = ComputationSealV2.from_mapping(seal_payload)

    with pytest.raises(aggregator.AggregationInputError, match=error):
        aggregator._validate_preregistration_contract(
            _payload(valid_fixture.prereg), seal=seal
        )


def test_rejects_cross_root_execution_and_missing_or_drifted_execution_sources(
    valid_fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = RepositorySnapshot.capture(valid_fixture.root)
    with pytest.raises(aggregator.AggregationInputError, match="repository root"):
        REAL_EXECUTION_SOURCE_RELATIVES(repository)

    monkeypatch.setattr(
        aggregator,
        "_execution_source_relatives",
        lambda snapshot: (*SYNTHETIC_EXECUTION_SOURCES, "missing/module.py"),
    )
    with pytest.raises(aggregator.AggregationInputError, match="omits executed"):
        aggregator.aggregate_revalidation(*valid_fixture.arguments())

    monkeypatch.setattr(
        aggregator,
        "_execution_source_relatives",
        lambda snapshot: SYNTHETIC_EXECUTION_SOURCES,
    )
    seal = _payload(valid_fixture.seal)
    seal["computation_source_sha256s"][ADMISSION_ID] = "0" * 64
    _rehash(seal, "content_sha256")
    valid_fixture.seal.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(aggregator.AggregationInputError, match="committed HEAD"):
        aggregator.aggregate_revalidation(*valid_fixture.arguments())


def test_rejects_dirty_repository_source_drift_and_preregistration_mismatch(
    valid_fixture: Fixture,
) -> None:
    (valid_fixture.root / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(aggregator.AggregationInputError, match="clean repository"):
        aggregator.aggregate_revalidation(*valid_fixture.arguments())
    (valid_fixture.root / "dirty.txt").unlink()

    runner = valid_fixture.root / RUNNER_ID
    runner.write_text("drift", encoding="utf-8")
    with pytest.raises(aggregator.AggregationInputError, match="clean repository"):
        aggregator.aggregate_revalidation(*valid_fixture.arguments())
    subprocess.run(
        ["git", "-C", valid_fixture.root, "checkout", "--", RUNNER_ID], check=True
    )

    seal = _payload(valid_fixture.seal)
    seal["preregistration_id"] = "wrong-preregistration"
    _rehash(seal, "content_sha256")
    valid_fixture.seal.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(aggregator.AggregationInputError, match="artifact_id differs"):
        aggregator.aggregate_revalidation(*valid_fixture.arguments())


def test_rejects_symlink_hardlink_and_output_aliases(valid_fixture: Fixture) -> None:
    original_seal = valid_fixture.seal
    symlink = valid_fixture.external / "seal-link.json"
    symlink.symlink_to(original_seal)
    with pytest.raises(aggregator.AggregationInputError, match="symbolic"):
        aggregator.aggregate_revalidation(
            valid_fixture.root,
            valid_fixture.prereg,
            symlink,
            valid_fixture.replicate_a,
            valid_fixture.replicate_b,
        )

    hardlink = valid_fixture.external / "seal-hardlink.json"
    os.link(original_seal, hardlink)
    with pytest.raises(aggregator.AggregationInputError, match="hard-linked"):
        aggregator.aggregate_revalidation(*valid_fixture.arguments())
    hardlink.unlink()

    with pytest.raises(aggregator.AggregationInputError, match="already exists"):
        aggregator._open_output_parent(original_seal, root=valid_fixture.root)
    with pytest.raises(aggregator.AggregationInputError, match="external"):
        aggregator._open_output_parent(
            valid_fixture.root / "output.json", root=valid_fixture.root
        )


def test_input_stability_detects_mutation(valid_fixture: Fixture) -> None:
    stability = _publication_stability(valid_fixture, valid_fixture.replicate_b)
    valid_fixture.replicate_b.write_text("{}", encoding="utf-8")
    with pytest.raises(aggregator.AggregationInputError, match="changed"):
        stability.assert_stable()


def test_cross_root_cli_returns_two(valid_fixture: Fixture) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *_main_arguments(valid_fixture)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "repository root" in result.stderr


def test_transactional_output_succeeds_once_and_rejects_preexisting_output(
    valid_fixture: Fixture,
) -> None:
    encoded = _publish_test_artifact(valid_fixture)
    assert valid_fixture.output.read_bytes() == encoded

    preserved = valid_fixture.output.read_bytes()
    with pytest.raises(aggregator.AggregationInputError, match="refusing to overwrite"):
        _publish_test_artifact(valid_fixture)
    assert valid_fixture.output.read_bytes() == preserved


@pytest.mark.parametrize(
    "hook_name", ("_after_anonymous_write", "_after_anonymous_verification")
)
def test_precommit_input_races_leave_no_output(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    hook_name: str,
) -> None:
    def mutate_input(descriptor: int) -> None:
        del descriptor
        valid_fixture.replicate_b.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(aggregator, hook_name, mutate_input)
    stability = _publication_stability(valid_fixture, valid_fixture.replicate_b)
    with pytest.raises(aggregator.AggregationInputError, match="changed"):
        _publish_test_artifact(valid_fixture, stability=stability)
    assert not valid_fixture.output.exists()
    assert not list(valid_fixture.external.glob(".*.tmp-*"))


def test_precommit_anonymous_inode_byte_race_fails_without_publication(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def corrupt_anonymous_inode(descriptor: int) -> None:
        os.pwrite(descriptor, b"corrupt", 0)

    monkeypatch.setattr(aggregator, "_after_anonymous_write", corrupt_anonymous_inode)
    with pytest.raises(aggregator.AggregationInputError, match="anonymous output"):
        _publish_test_artifact(valid_fixture)
    assert not valid_fixture.output.exists()
    assert not list(valid_fixture.external.glob(".*.tmp-*"))


def test_output_parent_retarget_is_detected_and_leaves_no_artifact(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parent = valid_fixture.external
    moved_parent = original_parent.with_name("external-moved")

    def retarget_parent(descriptor: int) -> None:
        del descriptor
        original_parent.rename(moved_parent)
        original_parent.symlink_to(moved_parent, target_is_directory=True)

    monkeypatch.setattr(aggregator, "_after_anonymous_write", retarget_parent)
    stability = _publication_stability(valid_fixture, valid_fixture.seal)
    with pytest.raises(aggregator.AggregationInputError, match="changed|symbolic"):
        _publish_test_artifact(valid_fixture, stability=stability)
    assert not (moved_parent / valid_fixture.output.name).exists()
    assert not list(moved_parent.glob(".*.tmp-*"))


def test_unsupported_o_tmpfile_fails_closed_without_output(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(aggregator.os, "O_TMPFILE", raising=False)
    with pytest.raises(
        aggregator.AggregationInputError, match="O_TMPFILE is unavailable"
    ):
        _publish_test_artifact(valid_fixture)
    assert not valid_fixture.output.exists()


def test_linkat_failure_leaves_anonymous_inode_unpublished(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_linkat(
        anonymous_descriptor: int, directory_descriptor: int, output_name: str
    ) -> None:
        del anonymous_descriptor, directory_descriptor, output_name
        raise aggregator.AggregationInputError("injected linkat failure")

    monkeypatch.setattr(aggregator, "_link_anonymous_noreplace", fail_linkat)
    with pytest.raises(
        aggregator.AggregationInputError, match="injected linkat failure"
    ):
        _publish_test_artifact(valid_fixture)
    assert not valid_fixture.output.exists()


def test_linkat_fallback_rejects_and_preserves_existing_destination(
    valid_fixture: Fixture,
) -> None:
    valid_fixture.output.write_bytes(b"preserve-existing")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(valid_fixture.external, directory_flags)
    anonymous_descriptor = aggregator._open_anonymous_output(directory_descriptor)
    try:
        with pytest.raises(aggregator.AggregationInputError, match="appeared"):
            aggregator._link_anonymous_noreplace(
                anonymous_descriptor,
                directory_descriptor,
                valid_fixture.output.name,
            )
    finally:
        os.close(anonymous_descriptor)
        os.close(directory_descriptor)
    assert valid_fixture.output.read_bytes() == b"preserve-existing"


def test_real_procfs_fallback_publishes_exact_open_inode(
    valid_fixture: Fixture,
) -> None:
    output_name = "real-anonymous-publication.json"
    expected = b'{"real":true}\n'
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(valid_fixture.external, directory_flags)
    anonymous_descriptor = aggregator._open_anonymous_output(directory_descriptor)
    try:
        assert os.write(anonymous_descriptor, expected) == len(expected)
        os.fsync(anonymous_descriptor)
        aggregator._verify_anonymous_descriptor(anonymous_descriptor, expected)
        aggregator._link_anonymous_noreplace(
            anonymous_descriptor, directory_descriptor, output_name
        )
        descriptor_stat = os.fstat(anonymous_descriptor)
        output_stat = os.stat(
            output_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        assert (output_stat.st_dev, output_stat.st_ino) == (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        )
    finally:
        os.close(anonymous_descriptor)
        os.close(directory_descriptor)
    assert (valid_fixture.external / output_name).read_bytes() == expected


def test_procfs_fallback_error_fails_closed(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = iter((errno.ENOENT, errno.EIO))
    monkeypatch.setattr(aggregator, "_call_linkat", lambda *arguments: next(errors))
    with pytest.raises(aggregator.AggregationInputError, match="proc/self/fd"):
        _publish_test_artifact(valid_fixture)
    assert not valid_fixture.output.exists()


def test_publication_structure_has_no_named_temp_or_rollback_surface() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "O_TMPFILE" in source
    assert "AT_EMPTY_PATH" in source
    assert ".tmp-" not in source
    assert "_rename_noreplace" not in source
    assert "_after_output_publish" not in source
    assert "_unlink_output_if_owned" not in source
    assert "os.unlink(output_name" not in source


def test_publication_commit_is_terminal(
    valid_fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stability = _publication_stability(valid_fixture, valid_fixture.replicate_a)

    real_link = aggregator._link_anonymous_noreplace

    def commit_then_external_tamper(
        anonymous_descriptor: int, directory_descriptor: int, output_name: str
    ) -> None:
        real_link(anonymous_descriptor, directory_descriptor, output_name)
        valid_fixture.replicate_a.write_text("external-tamper", encoding="utf-8")

    monkeypatch.setattr(
        aggregator, "_link_anonymous_noreplace", commit_then_external_tamper
    )
    encoded = _publish_test_artifact(valid_fixture, stability=stability)
    assert valid_fixture.output.read_bytes() == encoded
    assert valid_fixture.output.exists()
