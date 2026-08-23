from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from maple.solvation.release import accuracy_admission as accuracy
import tools.route2_release.run_mace_mdp_polar_hybrid_mnsol10_fullsolv as runner
import tools.route2_release.recover_mace_mdp_polar_hybrid_mnsol10_attempt as recovery
from tools.route2_release import _secure_artifacts as secure
from tools.route2_release import _hybrid_mnsol10_chain as chain_contract

ROOT = Path(__file__).parents[2]
PREREGISTRATION = ROOT / (
    "docs/implicit-solvation/benchmarks/"
    "route2-mace-mdp-polar-hybrid-mnsol10-fullsolv-prereg-v1.json"
)


def _prediction_record(index: int, *, gate_failure: str | None = None):
    predicted_eV = -1.0 - 0.01 * index
    predicted_kcal = predicted_eV * accuracy.EV_TO_KCAL_MOL
    polarization = predicted_kcal - 0.25
    gates = {
        "cold_wide_source": True,
        "cold_wide_field": True,
        "cold_wide_energy": True,
        "actual_residuals": True,
        "permanent_charge": True,
        "induced_charge": True,
        "combined_charge": True,
    }
    values = {
        "induced_source_max_abs_difference_e": 1.0e-12,
        "native_field_max_abs_difference_eV_per_e": 1.0e-12,
        "continuum_energy_abs_difference_eV": 1.0e-12,
        "cold_actual_residual_norm": 1.0e-12,
        "wide_actual_residual_norm": 2.0e-12,
        "permanent_charge_e": 2.0e-12,
        "induced_charge_e": -1.0e-12,
        "combined_charge_e": 1.0e-12,
    }
    if gate_failure == "cold_wide_source":
        values["induced_source_max_abs_difference_e"] = 3.0e-9
        gates[gate_failure] = False
    record = {
        "selection_index": index,
        "opaque_record_id": f"{index + 1:064x}",
        "canonical_solvent": f"solvent-{index}",
        "partition": "confirmation" if index < 8 else "development",
        "geometry_sha256": f"{index + 11:064x}",
        "normalized_geometry_sha256": f"{index + 21:064x}",
        "atom_count": 3,
        "solution_total_energy_eV": -4.0 + predicted_eV,
        "vacuum_energy_eV": -4.0,
        "predicted_delta_g_eV": predicted_eV,
        "predicted_delta_g_kcal_mol": predicted_kcal,
        "polarization_kcal_mol": polarization,
        "cds_kcal_mol": 0.25,
        **values,
        "cold_iterations": 4,
        "wide_iterations": 5,
        "gates": gates,
        "configuration_sha256": f"{index + 31:064x}",
        "continuum_configuration_sha256": f"{index + 41:064x}",
        "continuum_provenance_sha256": f"{index + 51:064x}",
        "equation_sha256": f"{index + 61:064x}",
        "cold_root_sha256": f"{index + 71:064x}",
        "wide_root_sha256": f"{index + 81:064x}",
        "base_ledger_sha256": f"{index + 91:064x}",
        "total_ledger_sha256": f"{index + 101:064x}",
        "solvent_term_configuration_sha256": f"{index + 111:064x}",
        "state_sha256": f"{index + 121:064x}",
    }
    accuracy.validate_prediction_record(record, expected_index=index)
    return record


def _label_free_bundle():
    records = []
    for index in range(10):
        numbers = [8, 1, 1]
        positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.7 + index * 1.0e-4, 0.5],
            [0.0, -0.7, 0.5],
        ]
        records.append(
            {
                "selection_index": index,
                "opaque_record_id": f"{index + 1:064x}",
                "canonical_solvent": f"solvent-{index}",
                "partition": "confirmation" if index < 8 else "development",
                "geometry_sha256": f"{index + 11:064x}",
                "normalized_geometry_sha256": accuracy.normalized_geometry_sha256(
                    atomic_numbers=numbers,
                    positions_angstrom=positions,
                    charge=0,
                    multiplicity=1,
                ),
                "atom_count": 3,
                "atomic_numbers": numbers,
                "positions_angstrom": positions,
                "charge": 0,
                "multiplicity": 1,
            }
        )
    payload = {
        "schema_id": accuracy.LABEL_FREE_INPUT_SCHEMA_ID,
        "artifact_id": "test-label-free-input",
        "created_at_utc": "2026-08-24T00:00:00+00:00",
        "git": {"head": "1" * 40, "tree": "2" * 40, "clean": True},
        "preregistration_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "selection_sha256": "c" * 64,
        "selection_fingerprint": "d" * 64,
        "dataset": {
            "protocol_id": "protocol",
            "protocol_fingerprint": "e" * 64,
            "table_sha256": "f" * 64,
            "normalized_bundle_sha256": "0" * 64,
            "temperature_k": 298.0,
            "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        },
        "record_count": 10,
        "records": records,
        "redistribution_guard": {
            "raw_rows_emitted": False,
            "entry_numbers_emitted": False,
            "geometry_handles_emitted": False,
            "solute_names_emitted": False,
            "formulas_emitted": False,
            "experimental_values_emitted": False,
        },
        "claim_boundary": "label-free test bundle",
    }
    return accuracy.with_content_sha256(payload)


def test_label_free_bundle_rejects_nested_target_and_coordinate_digest_drift():
    bundle = _label_free_bundle()
    accuracy.validate_label_free_input_bundle(bundle, expected_record_count=10)

    hidden = copy.deepcopy(bundle)
    del hidden["content_sha256"]
    hidden["dataset"]["target"] = -4.2
    hidden = accuracy.with_content_sha256(hidden)
    with pytest.raises(
        accuracy.AccuracyContractError, match="forbidden key|keys changed"
    ):
        accuracy.validate_label_free_input_bundle(hidden, expected_record_count=10)

    changed = copy.deepcopy(bundle)
    del changed["content_sha256"]
    changed["records"][0]["positions_angstrom"][0][0] = 0.1
    changed = accuracy.with_content_sha256(changed)
    with pytest.raises(accuracy.AccuracyContractError, match="geometry digest"):
        accuracy.validate_label_free_input_bundle(changed, expected_record_count=10)


def test_prediction_validator_recomputes_units_components_charge_and_gates():
    record = _prediction_record(0)
    for field, value, message in (
        ("predicted_delta_g_kcal_mol", -1.0, "eV-to-kcal"),
        ("cds_kcal_mol", 1.0, "polarization plus CDS"),
        ("combined_charge_e", 2.0e-12, "charge does not close"),
    ):
        changed = copy.deepcopy(record)
        changed[field] = value
        with pytest.raises(accuracy.AccuracyContractError, match=message):
            accuracy.validate_prediction_record(changed, expected_index=0)
    changed = copy.deepcopy(record)
    changed["gates"]["cold_wide_source"] = False
    with pytest.raises(accuracy.AccuracyContractError, match="mechanically derived"):
        accuracy.validate_prediction_record(changed, expected_index=0)


def _fake_prepared(tmp_path: Path):
    bundle = _label_free_bundle()
    preregistration = json.loads(PREREGISTRATION.read_text())

    class Stable:
        def assert_stable(self):
            return None

    capture = SimpleNamespace(sha256="a" * 64, assert_stable=lambda **_: None)
    return SimpleNamespace(
        repository=SimpleNamespace(
            root=ROOT,
            as_dict=lambda: {"head": "1" * 40, "tree": "2" * 40, "clean": True},
        ),
        seal=SimpleNamespace(
            execution_id="9" * 64,
            attempt_slot_id="6" * 64,
            content_sha256="8" * 64,
            claim_boundary="known-panel regression only",
            profile_id=preregistration["target_identity"]["profile_id"],
            scalar_id=preregistration["target_identity"]["scalar_id"],
            input_content_sha256=bundle["content_sha256"],
            source_files_sha256={"runner.py": "7" * 64},
        ),
        seal_capture=capture,
        input_capture=capture,
        preregistration_capture=capture,
        mdp_checkpoint_capture=capture,
        polar_checkpoint_capture=capture,
        stability=Stable(),
        preregistration=preregistration,
        input_bundle=bundle,
        accuracy=accuracy,
    )


@pytest.mark.parametrize("failed_index", [0, 5, 9])
def test_typed_failure_retains_every_completed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed_index: int
):
    prepared = _fake_prepared(tmp_path)
    published = {}
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_DIRECTORY", tmp_path / "runs")
    monkeypatch.setattr(
        runner,
        "_require_fresh_deterministic_process",
        lambda: runner.REQUIRED_ENVIRONMENT,
    )
    monkeypatch.setattr(runner, "_prepare_execution", lambda **_: prepared)
    monkeypatch.setattr(runner, "_validate_runtime_contract", lambda *_: None)
    monkeypatch.setattr(runner, "ensure_host_user_custody_directory", lambda *_: None)
    monkeypatch.setattr(runner, "claim_execution", lambda *_, **__: tmp_path / "claim")
    monkeypatch.setattr(runner, "capture_file", lambda *_, **__: prepared.seal_capture)

    def publish(path, value, **_kwargs):
        published["path"] = path
        published["value"] = value
        return path

    monkeypatch.setattr(runner, "publish_json_noreplace", publish)

    def science(_prepared, _contract, progress):
        for index in range(failed_index + 1):
            progress.current_index = index
            progress.current_record_id = f"{index + 1:064x}"
            progress.records.append(_prediction_record(index))
        raise runner.ScientificExecutionError("synthetic root failure")

    _path, terminal = runner.execute(
        seal_path=tmp_path / "seal.json", science_factory=science
    )
    assert terminal["status"] == "failure"
    assert [record["selection_index"] for record in terminal["records"]] == list(
        range(failed_index + 1)
    )
    assert terminal["failure"]["failed_selection_index"] == failed_index
    assert published["value"] == terminal


def test_invalid_candidate_record_leaves_validated_prefix_and_typed_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    prepared = _fake_prepared(tmp_path)
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "RUN_DIRECTORY", tmp_path / "runs")
    monkeypatch.setattr(
        runner,
        "_require_fresh_deterministic_process",
        lambda: runner.REQUIRED_ENVIRONMENT,
    )
    monkeypatch.setattr(runner, "_prepare_execution", lambda **_: prepared)
    monkeypatch.setattr(runner, "_validate_runtime_contract", lambda *_: None)
    monkeypatch.setattr(runner, "ensure_host_user_custody_directory", lambda *_: None)
    monkeypatch.setattr(runner, "claim_execution", lambda *_, **__: tmp_path / "claim")
    monkeypatch.setattr(runner, "capture_file", lambda *_, **__: prepared.seal_capture)
    monkeypatch.setattr(
        runner, "publish_json_noreplace", lambda path, _value, **_: path
    )

    def science(_prepared, _contract, progress):
        progress.records.extend(_prediction_record(index) for index in range(5))
        invalid = _prediction_record(5)
        invalid["predicted_delta_g_kcal_mol"] = 0.0
        progress.records.append(invalid)
        progress.current_index = 5
        progress.current_record_id = invalid["opaque_record_id"]

    _path, terminal = runner.execute(
        seal_path=tmp_path / "seal.json", science_factory=science
    )
    assert terminal["status"] == "failure"
    assert terminal["failure"]["stage"] == "candidate-record-validation"
    assert [record["selection_index"] for record in terminal["records"]] == list(
        range(5)
    )
    assert terminal["failure"]["failed_selection_index"] == 5


def test_execution_identity_has_one_canonical_claim_and_no_output_override():
    assert runner.ATTEMPT_SLOT_CLAIM_PATH == (
        Path.home()
        / ".cache/maple-route2-stage-claims-v1"
        / chain_contract.ATTEMPT_SLOT_STAGE_ID
        / "execution-claim.json"
    )
    parser_source = inspect.getsource(runner._arguments)
    assert "--private-output" not in parser_source
    assert "--public-output" not in parser_source
    assert "--preregistration" not in parser_source
    execute_source = inspect.getsource(runner.execute)
    assert (
        "run_directory = RUN_DIRECTORY / prepared.seal.execution_id" in execute_source
    )
    assert "claim_path = ATTEMPT_SLOT_CLAIM_PATH" in execute_source
    assert (
        'terminal_path = run_directory / "prediction-terminal.json"' in execute_source
    )


def test_attempt_slot_is_scientific_and_independent_of_git_execution_identity():
    preregistration = json.loads(PREREGISTRATION.read_text())
    source_validation = {
        "mnsol_source_file_sha256": "1" * 64,
        "dataset": {"normalized_bundle_sha256": "2" * 64},
        "derived_records_sha256": "3" * 64,
    }
    arguments = {
        "preregistration": preregistration,
        "source_validation": source_validation,
        "protocol_sha256": "4" * 64,
        "selection_sha256": "5" * 64,
        "mace_mdp_checkpoint_sha256": "6" * 64,
        "mace_polar_checkpoint_sha256": "7" * 64,
    }
    first = chain_contract.build_attempt_slot_identity(**arguments)
    changed_metadata = copy.deepcopy(preregistration)
    changed_metadata["created_at_utc"] = "2099-01-01T00:00:00Z"
    second = chain_contract.build_attempt_slot_identity(
        **{**arguments, "preregistration": changed_metadata}
    )
    assert first == second
    assert "git" not in first
    assert "source_files_sha256" not in first
    changed_selection = chain_contract.build_attempt_slot_identity(
        **{**arguments, "selection_sha256": "8" * 64}
    )
    assert changed_selection != first


def test_secure_publication_is_noreplace_and_checks_stability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "artifact.json"
    calls = []
    fsync_descriptors = []
    events = []
    real_fsync = secure.os.fsync
    real_write_all = secure._write_all
    real_linkat = secure._linkat

    def observed_write(descriptor, data):
        events.append("write")
        return real_write_all(descriptor, data)

    def observed_fsync(descriptor):
        fsync_descriptors.append(descriptor)
        events.append("fsync")
        return real_fsync(descriptor)

    def observed_linkat(*arguments):
        events.append("linkat")
        return real_linkat(*arguments)

    monkeypatch.setattr(secure, "_write_all", observed_write)
    monkeypatch.setattr(secure.os, "fsync", observed_fsync)
    monkeypatch.setattr(secure, "_linkat", observed_linkat)
    secure.publish_json_noreplace(
        output,
        {"b": 2, "a": [1, True]},
        root=tmp_path,
        stability=lambda: (calls.append("checked"), events.append("stability")),
    )
    assert calls == ["checked"]
    assert len(fsync_descriptors) >= 2
    assert events[:5] == ["write", "fsync", "stability", "linkat", "fsync"]
    assert output.read_bytes() == secure.canonical_json_bytes({"b": 2, "a": [1, True]})
    with pytest.raises(FileExistsError):
        secure.publish_json_noreplace(output, {"different": True}, root=tmp_path)
    assert secure.secure_publication_contract() == {
        "contract_id": "maple-route2-linux-o-tmpfile-linkat-fsync-noreplace-v1",
        "platform": "linux",
        "anonymous_inode": "O_TMPFILE",
        "file_durability_before_commit": "fsync(tmpfd)-required",
        "commit": "linkat-no-replace",
        "directory_durability_after_commit": "fsync(parent-dirfd)-required",
        "path_boundary": "retained-O_NOFOLLOW-parent-dirfd-plus-lexical-identity",
        "precommit_stability_callback": "required-when-evidence-has-inputs",
        "directory_creation": "mkdirat-each-component-plus-fsync-parent",
    }


def test_cold_publication_durably_syncs_each_new_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events = []
    real_mkdir = secure.os.mkdir
    real_fsync = secure.os.fsync

    def observed_mkdir(path, mode=0o777, *, dir_fd=None):
        events.append(f"mkdir:{path}")
        return real_mkdir(path, mode=mode, dir_fd=dir_fd)

    def observed_fsync(descriptor):
        events.append("fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(secure.os, "mkdir", observed_mkdir)
    monkeypatch.setattr(secure.os, "fsync", observed_fsync)
    output = tmp_path / "cold" / "nested" / "artifact.json"
    secure.publish_json_noreplace(output, {"value": 1}, root=tmp_path)
    mkdir_indices = [
        index for index, event in enumerate(events) if event.startswith("mkdir:")
    ]
    assert [events[index] for index in mkdir_indices] == [
        "mkdir:cold",
        "mkdir:nested",
    ]
    assert all(events[index + 1] == "fsync" for index in mkdir_indices)
    assert output.is_file()


def test_spoofed_home_cannot_move_host_global_stage_claim(tmp_path: Path):
    alternate_home = tmp_path / "alternate-home"
    alternate_home.mkdir()
    environment = dict(os.environ)
    environment["HOME"] = str(alternate_home)
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "from tools.route2_release._hybrid_mnsol10_chain import "
            "ATTEMPT_SLOT_CLAIM_PATH; print(ATTEMPT_SLOT_CLAIM_PATH)",
        ),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "HOME differs from the canonical passwd custody root" in result.stderr


def test_secure_publication_rejects_parent_retarget_and_source_drift(tmp_path: Path):
    parent = tmp_path / "evidence"
    parent.mkdir()
    output = parent / "artifact.json"

    def retarget_parent():
        parent.rename(tmp_path / "evidence-old")
        parent.mkdir()

    with pytest.raises(secure.SecureArtifactError, match="parent changed"):
        secure.publish_json_noreplace(
            output,
            {"value": 1},
            root=tmp_path,
            stability=retarget_parent,
        )
    assert not output.exists()

    drift_output = tmp_path / "drift" / "artifact.json"

    def source_drift():
        raise secure.SecureArtifactError("source drifted")

    with pytest.raises(secure.SecureArtifactError, match="source drifted"):
        secure.publish_json_noreplace(
            drift_output,
            {"value": 2},
            root=tmp_path,
            stability=source_drift,
        )
    assert not drift_output.exists()


def test_secure_publication_rejects_symlinked_output_parent(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(secure.SecureArtifactError, match="must not contain symlinks"):
        secure.publish_json_noreplace(
            linked / "artifact.json", {"value": 1}, root=tmp_path
        )


def test_public_projection_is_validated_and_contains_no_row_data():
    prediction_records = [_prediction_record(index) for index in range(10)]
    scored = []
    for index, record in enumerate(prediction_records):
        experimental = float(index) - 5.0
        signed = float(record["predicted_delta_g_kcal_mol"]) - experimental
        scored.append(
            {
                **record,
                "experimental_delta_g_kcal_mol": experimental,
                "signed_error_kcal_mol": signed,
                "absolute_error_kcal_mol": abs(signed),
            }
        )
    bundle = accuracy.build_scored_bundle(
        artifact_id="score",
        scored_at_utc="2026-08-24T00:00:00+00:00",
        claim_boundary="known panel only",
        profile_id="profile",
        scalar_id="scalar",
        prediction={
            "attempt_slot_id": "0" * 64,
            "execution_id": "1" * 64,
            "file_sha256": "2" * 64,
            "content_sha256": "3" * 64,
            "measurement_sha256": "4" * 64,
        },
        dataset={
            "protocol_id": "protocol",
            "protocol_fingerprint": "5" * 64,
            "protocol_sha256": "6" * 64,
            "selection_sha256": "7" * 64,
            "selection_fingerprint": "8" * 64,
            "table_sha256": "9" * 64,
            "normalized_bundle_sha256": "a" * 64,
            "source_artifact_sha256": "b" * 64,
            "temperature_k": 298.0,
            "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
        },
        scored_records=scored,
        scorer_source_sha256="c" * 64,
        expected_partition_counts={"confirmation": 8, "development": 2},
    )
    public = accuracy.public_accuracy_projection(bundle)
    assert public["attempt_slot_id"] == "0" * 64
    assert public["evidence_class"] == "exposure-aware-known-panel-regression"
    assert public["terminal_type"] == "complete"

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    public_keys = keys(public)
    for forbidden in (
        "records",
        "opaque_record_id",
        "experimental_delta_g_kcal_mol",
        "predicted_delta_g_kcal_mol",
    ):
        assert forbidden not in public_keys
    tampered = copy.deepcopy(bundle)
    del tampered["content_sha256"]
    tampered["dataset"]["records"] = [{"experimental_delta_g_kcal_mol": -5.0}]
    tampered = accuracy.with_content_sha256(tampered)
    with pytest.raises(accuracy.AccuracyContractError):
        accuracy.public_accuracy_projection(tampered)


def test_runner_is_label_free_and_scorer_has_no_model_or_continuum_import():
    runner_source = inspect.getsource(runner)
    assert "load_mnsol_v2012" not in runner_source
    assert "experimental_delta_g_kcal_mol" not in runner_source
    scorer_source = (
        ROOT / "tools/route2_release/score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py"
    ).read_text()
    assert "from maple.solvation.models" not in scorer_source
    assert "import maple.solvation.models" not in scorer_source
    assert "from maple.solvation.continuum" not in scorer_source
    assert "import maple.solvation.continuum" not in scorer_source
    assert "from maple.solvation.coupling" not in scorer_source
    assert "import maple.solvation.coupling" not in scorer_source


def test_downstream_chain_rebinds_claim_archive_and_score_identity():
    chain_source = (ROOT / "tools/route2_release/_hybrid_mnsol10_chain.py").read_text()
    assert 'claim.get("process_uuid") != prediction_runtime.get("process_uuid")' in (
        chain_source
    )
    assert 'claim.get("process_started_at_utc")' in chain_source
    assert 'prediction.get("started_at_utc")' in chain_source
    assert "dict(prediction_git) != repository.as_dict()" in chain_source

    scorer_source = (
        ROOT / "tools/route2_release/score_mace_mdp_polar_hybrid_mnsol10_fullsolv.py"
    ).read_text()
    assert 'source_validation.get("mnsol_source_file_sha256")' in scorer_source
    assert "dataset.source_artifact_sha256 != source_capture.sha256" in scorer_source

    publisher_source = (
        ROOT / "tools/route2_release/publish_mace_mdp_polar_hybrid_mnsol10_public.py"
    ).read_text()
    for binding in ("artifact_id", "profile_id", "scalar_id", "claim_boundary"):
        assert f'"{binding}": private.get("{binding}")' in publisher_source


def test_orphan_recovery_checks_exact_linux_process_identity():
    process_id = runner.os.getpid()
    start_ticks = secure.linux_process_start_ticks(process_id)
    assert recovery._claim_process_is_alive(
        {
            "boot_id": secure.linux_boot_id(),
            "process_id": process_id,
            "process_start_ticks": start_ticks,
        }
    )
    assert not recovery._claim_process_is_alive(
        {
            "boot_id": secure.linux_boot_id(),
            "process_id": 2**30,
            "process_start_ticks": 1,
        }
    )

    recovery_source = inspect.getsource(recovery)
    assert "capture_clean_repository" not in recovery_source
    assert "_load_accuracy_contract" in recovery_source
    assert 'Path("/proc/sys/kernel/random/boot_id")' in inspect.getsource(
        secure.linux_boot_id
    )
    assert not recovery._claim_process_is_alive(
        {
            "boot_id": "00000000-0000-0000-0000-000000000000",
            "process_id": process_id,
            "process_start_ticks": start_ticks,
        }
    )


def test_failed_attempt_has_strict_aggregate_only_public_projection(tmp_path: Path):
    prepared = _fake_prepared(tmp_path)
    terminal = runner._terminal_base(
        prepared=prepared,
        process_started_at_utc="2026-08-24T00:00:00+00:00",
        finished_at_utc="2026-08-24T00:01:00+00:00",
        records=[_prediction_record(0)],
        runtime={
            "environment": runner.REQUIRED_ENVIRONMENT,
            "process_uuid": "process-uuid",
            "process_id": 123,
            "process_start_ticks": 456,
        },
        timing_seconds={"total_wall": 60.0},
    )
    terminal["status"] = "failure"
    terminal["failure"] = {
        "stage": "record-gate",
        "error_type": "test.RecordGateFailure",
        "message": "synthetic failure",
        "failed_selection_index": 1,
        "failed_opaque_record_id": "2" * 64,
    }
    terminal = accuracy.with_content_sha256(terminal)
    projection = accuracy.prediction_failure_public_projection(
        terminal, terminal_file_sha256="f" * 64
    )
    assert projection["attempt_slot_closed"] is True
    assert projection["score_performed"] is False
    assert projection["failure"] == {
        "stage": "record-gate",
        "error_type": "test.RecordGateFailure",
        "panel_completion": "incomplete",
        "expected_record_count": 10,
    }
    encoded = json.dumps(projection, sort_keys=True)
    for forbidden in (
        "opaque_record_id",
        "predicted_delta_g_kcal_mol",
        "experimental_delta_g_kcal_mol",
        "signed_error_kcal_mol",
    ):
        assert forbidden not in encoded
