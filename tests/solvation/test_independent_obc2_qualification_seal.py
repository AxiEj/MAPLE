"""Synthetic sealer fixtures only: no real qualification evidence is generated."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import seal_independent_obc2_qualification as sealer  # noqa: E402

validation = sealer.validation


def _write_json(path, artifact):
    validation.core.seal_artifact(artifact)
    path.write_text(json.dumps(artifact, sort_keys=True) + "\n")


def _record(nonpolar):
    return {
        "nonpolar": nonpolar,
        "passed": True,
        "same_scalar_energy_abs_hartree": 0.0,
        "same_scalar_force_max_hartree_per_angstrom": 0.0,
        "hessian_asymmetry_max_hartree_per_angstrom2": 0.0,
        "hessian_composition_max_hartree_per_angstrom2": 0.0,
        "direct_hvp_dense_max_hartree_per_angstrom2": 0.0,
        "direct_hvp_dense_hessian_forbidden": True,
        "hessian_force_fd_ladder": [
            {"step_angstrom": step, "rms_hartree_per_angstrom2": 0.0,
             "max_hartree_per_angstrom2": 0.0, "passed": True}
            for step in (1e-6, 2e-6, 5e-6)
        ],
        "gate_identity": {
            "model": "ani2x", "device": "cpu", "dtype": "float64", "d4": False,
            "checkpoint_sha256": sealer.runtime._ANI2X_QUALIFIED_CHECKPOINT_SHA256,
            "provider": {
                "runtime_provider": "torch-obc2", "model": "obc2",
                "profile": "obc2-mbondi2", "nonpolar": nonpolar,
            },
        },
    }


@pytest.fixture
def fixture_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True
        ).strip()

    git("init", "-q")
    git("config", "user.email", "sealer-test@example.invalid")
    git("config", "user.name", "Sealer Fixture")
    (tmp_path / ".gitignore").write_text(".evidence/\n")
    (tmp_path / "runtime.py").write_text("# synthetic runtime fixture\n")
    harness = tmp_path / validation.CANDIDATE_CELL_HARNESS_REL
    harness.parent.mkdir(parents=True)
    harness.write_text("# synthetic candidate harness fixture\n")
    pointer = tmp_path / sealer.POINTER_REL
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        f'QUALIFICATION_RECEIPT_FILENAME = "{Path(sealer.RECEIPT_REL).name}"\n'
        "QUALIFICATION_RECEIPT_SHA256 = None\n"
    )
    git("add", ".")
    git("commit", "-qm", "synthetic frozen candidate")
    monkeypatch.setattr(sealer, "ROOT", tmp_path)
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    frozen = validation.source_snapshot()
    monkeypatch.setattr(
        sealer.runtime, "_installed_runtime_source_inventory",
        lambda: {"runtime.py": frozen["files_sha256"]["runtime.py"]},
    )

    candidate = {
        "schema_version": 1,
        "artifact_type": "independent-obc2-candidate-exact-cell-v1",
        "qualification_stage": "candidate qualification, not installed qualification",
        "nonpolar": ["ace", "none"], "expected_count": 2, "passed_count": 2,
        "passed": True, "model": "ani2x", "device": "cpu",
        "dtype": "float64", "d4": False,
        "checkpoint_sha256": sealer.runtime._ANI2X_QUALIFIED_CHECKPOINT_SHA256,
        "candidate_git_sha": frozen["git_head"],
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "candidate_source_snapshot": frozen,
        "records": [_record("ace"), _record("none")],
    }
    validation.core.seal_artifact(candidate)
    gates = {
        key: {"expected_count": count, "passed_count": passed, "passed": True}
        for key, (count, passed) in sealer.runtime._QUALIFICATION_GATE_COUNTS.items()
    }
    gates["analytic_hessian_16_fd"]["steps_angstrom"] = [0.001, 0.0005, 0.00025]
    gates["direct_hvp_144"]["dense_hessian_forbidden"] = True
    gates["candidate_exact_cell_public_composition"].update(
        nonpolar=["ace", "none"], evidence_content_sha256=candidate["content_sha256"]
    )
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-independent-obc2-validation-v1",
        "recorded_date": validation.RECORDED_DATE,
        "source_unchanged": True, "source_snapshot": frozen,
        "candidate_exact_cell": copy.deepcopy(candidate),
        "selection": {"selected_cases": list(validation.FULL_H_CASES)},
        "guarded_no_openmm": {"passed": True},
        "replay_526": {"record_count": 1052, "passed": True},
        "portable_48_ef_radius": {"comparison_count": 48, "passed": True},
        "full_hessian_matrix": {"record_count": 16, "passed": True},
        "direct_hvp_matrix": {"record_count": 144, "passed": True},
        "qualification_gates": gates, "passed": True,
    }
    evidence_dir = tmp_path / ".evidence"
    evidence_dir.mkdir()
    full_path = evidence_dir / "synthetic-full.json"
    candidate_path = evidence_dir / "synthetic-candidate.json"
    _write_json(full_path, artifact)
    _write_json(candidate_path, candidate)
    return tmp_path, git, artifact, candidate, full_path, candidate_path


@pytest.fixture
def io_repo(fixture_repo, monkeypatch):
    """Isolate receipt I/O; authoritative full-matrix behavior is tested below.

    Source lineage, exact candidate records, artifact seals and gate schema
    still run through the real validator. Only full scientific matrix replay
    is stubbed here, never in the operator or any qualification environment.
    """
    scientific_replay = Mock(name="authoritative_full_matrix_replay")
    monkeypatch.setattr(
        validation, "_validate_full_matrix_records", scientific_replay
    )
    return fixture_repo


def test_deterministic_receipt_matches_factory_schema_and_raw_evidence(io_repo):
    root, _git, _full, candidate, full_path, candidate_path = io_repo
    first = sealer.build_receipt(full_path, candidate_path)
    assert first == sealer.build_receipt(full_path, candidate_path)
    assert validation._validate_full_matrix_records.call_count == 2
    assert first["content_sha256"] == hashlib.sha256(
        sealer.runtime._canonical_receipt_payload(first)
    ).hexdigest()
    assert first["qualified_cells"] == sealer._qualified_cells()
    assert [cell["solvent"]["nonpolar"] for cell in first["qualified_cells"]] == [
        "ace", "none"
    ]
    assert len(first["qualification_gates"]) == 7
    assert first["candidate_git_sha"] == candidate["candidate_git_sha"]
    assert first["qualification_sources"] == candidate[
        "candidate_source_snapshot"
    ]["files_sha256"]
    for key, path in (
        ("candidate_validation", full_path), ("candidate_exact_cell", candidate_path)
    ):
        binding = first["evidence"][key]
        assert binding["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert binding["path"] == path.relative_to(root).as_posix()


def test_publish_receipt_before_pointer_and_refuse_even_identical_rerun(io_repo):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    receipt = sealer.seal_qualification(full_path, candidate_path)
    receipt_path = root / sealer.RECEIPT_REL
    pointer_path = root / sealer.POINTER_REL
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(pointer_path.read_text()).body
    }
    assert values["QUALIFICATION_RECEIPT_SHA256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert json.loads(receipt_path.read_bytes()) == receipt
    before = (receipt_path.read_bytes(), pointer_path.read_bytes())
    with pytest.raises(FileExistsError, match="immutable"):
        sealer.seal_qualification(full_path, candidate_path)
    assert before == (receipt_path.read_bytes(), pointer_path.read_bytes())


@pytest.mark.parametrize("mutation", ["failed_record", "hash", "missing_record", "identity"])
def test_invalid_candidate_evidence_never_publishes(io_repo, mutation):
    root, _git, full, candidate, full_path, candidate_path = io_repo
    if mutation == "failed_record":
        candidate["records"][1]["passed"] = False
    elif mutation == "hash":
        candidate["content_sha256"] = "0" * 64
    elif mutation == "missing_record":
        candidate["records"].pop()
    else:
        candidate["records"][0]["gate_identity"]["device"] = "cuda"
    if mutation != "hash":
        validation.core.seal_artifact(candidate)
    candidate_path.write_text(json.dumps(candidate))
    full["candidate_exact_cell"] = copy.deepcopy(candidate)
    _write_json(full_path, full)
    with pytest.raises(ValueError):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()
    assert b"= None" in (root / sealer.POINTER_REL).read_bytes()


@pytest.mark.parametrize("mutation", ["hash", "gate_count", "false_pass", "missing_gate"])
def test_invalid_full_evidence_never_publishes(io_repo, mutation):
    root, _git, full, _candidate, full_path, candidate_path = io_repo
    if mutation == "hash":
        full["content_sha256"] = "0" * 64
        full_path.write_text(json.dumps(full))
    else:
        if mutation == "gate_count":
            # Equal bogus numerator/denominator must not pass the runtime contract.
            gate = full["qualification_gates"]["sealed_oracle_1052_energy"]
            gate.update(expected_count=1, passed_count=1)
        elif mutation == "false_pass":
            full["passed"] = False
        else:
            full["qualification_gates"].pop("source_unchanged")
        _write_json(full_path, full)
    with pytest.raises(ValueError):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


def test_external_candidate_cannot_differ_from_embedded_evidence(io_repo):
    root, _git, _full, candidate, full_path, candidate_path = io_repo
    candidate["fixture_note"] = "self-sealed but not the embedded artifact"
    _write_json(candidate_path, candidate)
    with pytest.raises(ValueError, match="embedded"):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


@pytest.mark.parametrize("missing", ["full", "candidate", "pointer"])
def test_missing_input_or_pointer_does_not_publish(io_repo, missing):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    {"full": full_path, "candidate": candidate_path,
     "pointer": root / sealer.POINTER_REL}[missing].unlink()
    with pytest.raises(FileNotFoundError):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


@pytest.mark.parametrize("change", ["dirty", "committed", "unrelated"])
def test_source_drift_or_nonancestor_never_publishes(io_repo, change):
    root, git, _full, _candidate, full_path, candidate_path = io_repo
    if change == "unrelated":
        git("checkout", "--orphan", "unrelated")
        git("commit", "-qm", "same files unrelated history")
    else:
        (root / "runtime.py").write_text("# source changed\n")
        if change == "committed":
            git("add", ".")
            git("commit", "-qm", "changed source")
    with pytest.raises((ValueError, RuntimeError)):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


def test_runtime_inventory_must_match_frozen_checkout(io_repo, monkeypatch):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    monkeypatch.setattr(
        sealer.runtime, "_installed_runtime_source_inventory",
        lambda: {"runtime.py": "0" * 64},
    )
    with pytest.raises(ValueError, match="inventory"):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


@pytest.mark.parametrize("failure", ["receipt", "pointer"])
def test_partial_write_failure_never_qualifies_pointer(io_repo, monkeypatch, failure):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    original = sealer._publish
    before = (root / sealer.POINTER_REL).read_bytes()

    def fail_publish(path, raw, *, replace):
        if (failure == "pointer") == replace:
            raise OSError("synthetic publication failure")
        return original(path, raw, replace=replace)

    monkeypatch.setattr(sealer, "_publish", fail_publish)
    with pytest.raises(OSError, match="synthetic"):
        sealer.seal_qualification(full_path, candidate_path)
    assert (root / sealer.POINTER_REL).read_bytes() == before
    assert (root / sealer.RECEIPT_REL).exists() == (failure == "pointer")


def test_atomic_exclusive_receipt_install_cannot_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        sealer._publish(path, b"replacement", replace=False)
    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


def test_pointer_must_be_unqualified_data_only(io_repo):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    pointer = root / sealer.POINTER_REL
    pointer.write_text("raise RuntimeError('not data')\n")
    with pytest.raises(ValueError, match="two data assignments"):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


def test_candidate_gate_hash_must_bind_the_embedded_evidence(io_repo):
    root, _git, full, _candidate, full_path, candidate_path = io_repo
    full["qualification_gates"]["candidate_exact_cell_public_composition"][
        "evidence_content_sha256"
    ] = "0" * 64
    _write_json(full_path, full)
    with pytest.raises(ValueError, match="evidence hash"):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()


def test_atomic_pointer_replace_observes_complete_matching_receipt(io_repo, monkeypatch):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    replace = sealer.os.replace
    calls = []

    def checked_replace(temporary, destination):
        pointer = dict(
            (node.targets[0].id, ast.literal_eval(node.value))
            for node in ast.parse(Path(temporary).read_text()).body
        )
        assert pointer["QUALIFICATION_RECEIPT_SHA256"] == hashlib.sha256(
            (root / sealer.RECEIPT_REL).read_bytes()
        ).hexdigest()
        calls.append(destination)
        return replace(temporary, destination)

    monkeypatch.setattr(sealer.os, "replace", checked_replace)
    sealer.seal_qualification(full_path, candidate_path)
    assert calls == [root / sealer.POINTER_REL]


def test_failed_atomic_pointer_replace_leaves_unqualified_data_and_no_tempfile(
    io_repo, monkeypatch
):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    pointer = root / sealer.POINTER_REL
    before = pointer.read_bytes()

    def fail_replace(*_args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(sealer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        sealer.seal_qualification(full_path, candidate_path)
    assert pointer.read_bytes() == before
    assert (root / sealer.RECEIPT_REL).is_file()
    assert not list(pointer.parent.glob(f".{pointer.name}.*"))


@pytest.mark.parametrize("target", ["pointer", "receipt"])
def test_symlink_output_is_refused_without_touching_target(io_repo, target):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    destination = root / (
        sealer.POINTER_REL if target == "pointer" else sealer.RECEIPT_REL
    )
    victim = root / ".evidence" / "do-not-change"
    victim.write_text("preserve")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    destination.symlink_to(victim)
    with pytest.raises((ValueError, FileExistsError)):
        sealer.seal_qualification(full_path, candidate_path)
    assert victim.read_text() == "preserve"


def test_operator_main_parses_explicit_paths_and_reports_published_seal(
    io_repo, monkeypatch, capsys
):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    monkeypatch.setattr(
        sys, "argv", ["seal-independent-obc2", "--validation", str(full_path),
                      "--candidate", str(candidate_path)]
    )
    sealer.main()
    output = json.loads(capsys.readouterr().out)
    receipt = json.loads((root / sealer.RECEIPT_REL).read_bytes())
    assert output == {
        "receipt": sealer.RECEIPT_REL, "pointer": sealer.POINTER_REL,
        "content_sha256": receipt["content_sha256"],
    }


def test_summary_only_full_matrix_cannot_authorize_qualification(fixture_repo):
    """Preseal safety regression: claimed counts are not underlying evidence."""
    root, _git, full, _candidate, full_path, candidate_path = fixture_repo
    assert all(
        "records" not in full[name]
        for name in (
            "replay_526", "portable_48_ef_radius",
            "full_hessian_matrix", "direct_hvp_matrix",
        )
    )
    with pytest.raises(ValueError, match="replay_526 records incomplete: expected 1052"):
        sealer.seal_qualification(full_path, candidate_path)
    assert not (root / sealer.RECEIPT_REL).exists()
    assert b"= None" in (root / sealer.POINTER_REL).read_bytes()


def test_scientific_validator_failure_prevents_receipt_and_pointer_publication(io_repo):
    root, _git, _full, _candidate, full_path, candidate_path = io_repo
    validation._validate_full_matrix_records.side_effect = ValueError(
        "scientific evidence rejected"
    )
    pointer_before = (root / sealer.POINTER_REL).read_bytes()
    with pytest.raises(ValueError, match="scientific evidence rejected"):
        sealer.seal_qualification(full_path, candidate_path)
    validation._validate_full_matrix_records.assert_called_once()
    assert not (root / sealer.RECEIPT_REL).exists()
    assert (root / sealer.POINTER_REL).read_bytes() == pointer_before
