from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "pretrained-solvation-hub" / "run_atomicese_release_audit.py"
ARTIFACT = (
    ROOT
    / "docs"
    / "pretrained-solvation-hub"
    / "benchmarks"
    / "atomicese-release-audit-2026-07-31.json"
)


def _load_audit_module():
    specification = importlib.util.spec_from_file_location(
        "atomicese_release_audit",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_pinned_release_identity_covers_exact_four_files():
    assert AUDIT.SOURCE_REVISION == "31e643c7e8974497c78fc2fb6f3c17778d61fa10"
    assert AUDIT.SOURCE_TREE == "e08ce8bfaf73ca9c35c2b85929067e639d7bf048"
    assert tuple(AUDIT.PINNED_BLOBS) == AUDIT.EXPECTED_TREE_PATHS
    assert all(
        set(specification) == {"git_blob", "sha256", "size_bytes"}
        for specification in AUDIT.PINNED_BLOBS.values()
    )


def test_tree_parser_and_blob_gates_fail_closed():
    tree = AUDIT._parse_tree(
        b"100644 blob 0123456789012345678901234567890123456789\tfile\n"
    )
    assert tree == {"file": ("100644", "0123456789012345678901234567890123456789")}

    with pytest.raises(ValueError, match="size"):
        AUDIT._verify_blob(
            b"wrong",
            path="synthetic",
            specification={"size_bytes": 6, "sha256": "unused"},
        )
    with pytest.raises(ValueError, match="SHA256"):
        AUDIT._verify_blob(
            b"wrong",
            path="synthetic",
            specification={"size_bytes": 5, "sha256": "0" * 64},
        )


def test_release_identity_rejects_wrong_revision_and_dirty_checkout(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        AUDIT,
        "_git_bytes",
        lambda root, *arguments: (
            b"0" * 40 + b"\n"
            if arguments == ("rev-parse", "HEAD")
            else f"{AUDIT.SOURCE_TREE}\n".encode()
        ),
    )
    with pytest.raises(ValueError, match="revision"):
        AUDIT._verify_release(tmp_path)

    answers = iter(
        (
            f"{AUDIT.SOURCE_REVISION}\n".encode(),
            f"{AUDIT.SOURCE_TREE}\n".encode(),
            b"?? unexpected\n",
        )
    )
    monkeypatch.setattr(AUDIT, "_git_bytes", lambda root, *arguments: next(answers))
    with pytest.raises(ValueError, match="clean"):
        AUDIT._verify_release(tmp_path)


def test_sandbox_command_is_no_network_and_uses_only_temporary_copy(tmp_path):
    command = AUDIT._sandbox_command(
        tmp_path,
        (AUDIT.SAMPLE_PATH, "-solvent", "acetone"),
    )

    assert command[0] == "/usr/bin/bwrap"
    assert "--unshare-all" in command
    assert "--share-net" not in command
    assert "--ro-bind" in command
    assert str(tmp_path) in command
    assert "/work/AtomicESE.x" in command
    assert "/home/axie/.cache/maple-benchmarks/AtomicESE/AtomicESE.x" not in command


def _completed(stdout: str, stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["AtomicESE.x"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_sample_parser_and_error_semantics_are_fail_closed():
    successful = _completed(
        'Solvent name "acetone" was read from the command line\n'
        "Number of atoms in the solute:    5\n"
        "Total solvation free energy = -5.92 kcal/mol\n"
        "CPU time = 0.000 seconds.\n"
    )
    assert AUDIT._parse_sample_result(successful) == -5.92

    with pytest.raises(ValueError, match="differs"):
        AUDIT._parse_sample_result(
            _completed(
                'Solvent name "acetone" was read from the command line\n'
                "Number of atoms in the solute:    5\n"
                "Total solvation free energy = -5.91 kcal/mol\n"
                "CPU time = 0.000 seconds.\n"
            )
        )

    diagnostic = "Water is not implemented in this method."
    AUDIT._require_diagnostic(
        _completed(diagnostic, diagnostic),
        label="water",
        token=diagnostic,
    )
    with pytest.raises(ValueError, match="exit code"):
        AUDIT._require_diagnostic(
            _completed(diagnostic, diagnostic, returncode=1),
            label="water",
            token=diagnostic,
        )


def test_runtime_audit_repeats_scalar_and_records_no_wall_times(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(AUDIT, "BWRAP_PATH", tmp_path / "bwrap")
    AUDIT.BWRAP_PATH.write_bytes(b"stub")
    calls = []

    def fake_run(temporary_root, arguments):
        calls.append(tuple(arguments))
        if arguments[-1] == "water":
            message = "Water is not implemented in this method."
            return _completed(message, message)
        if arguments[-1] == "definitely_not_a_solvent":
            message = 'Solvent "definitely_not_a_solvent" not in the list.'
            return _completed(message, message)
        return _completed(
            'Solvent name "acetone" was read from the command line\n'
            "Number of atoms in the solute:    5\n"
            "Total solvation free energy = -5.92 kcal/mol\n"
            "CPU time = 0.000 seconds.\n"
        )

    monkeypatch.setattr(AUDIT, "_run_sandboxed", fake_run)
    runtime = AUDIT._audit_runtime(b"binary", b"sample")

    assert (
        calls[: AUDIT.REPEAT_COUNT]
        == [(AUDIT.SAMPLE_PATH, "-solvent", "acetone")] * AUDIT.REPEAT_COUNT
    )
    assert runtime["sample"]["repeat_count"] >= 3
    assert runtime["sample"]["scalar_result"] == -5.92
    assert runtime["sample"]["wall_times_recorded_as_admission_evidence"] is False
    assert runtime["failure_semantics"]["controlled_error_exit_code"] == 0
    assert runtime["failure_semantics"]["water_supported"] is False


def test_frozen_artifact_preserves_all_negative_admission_boundaries():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["acceptance_eligible"] is False
    assert payload["identity"]["revision"] == AUDIT.SOURCE_REVISION
    assert payload["identity"]["tree"] == AUDIT.SOURCE_TREE
    release = payload["release_boundary"]
    assert release["scope"] == "exact_pinned_official_git_tree_only"
    assert release["external_publication_or_supporting_information_audited"] is False
    supporting_information = release[
        "publication_declared_external_supporting_information"
    ]
    assert supporting_information["locator"].endswith("jcc70104-sup-0001-supinfo.zip")
    assert supporting_information["bytes_audited"] is False
    assert supporting_information["sha256"] is None
    assert release["license_file_present_in_pinned_git_tree"] is False
    assert release["source_code_present_in_pinned_git_tree"] is False
    assert release["checkpoint_identity_ledger_present_in_pinned_git_tree"] is False
    assert payload["cli_contract"]["standard_state_status"] == (
        "unknown_from_pinned_release"
    )
    assert payload["runtime"]["sample"]["repeat_count"] >= 3
    assert payload["runtime"]["sample"]["scalar_result"] == -5.92
    assert payload["runtime"]["failure_semantics"]["water_supported"] is False
    assert payload["runtime"]["failure_semantics"]["controlled_error_exit_code"] == 0
    compute = payload["packaged_compute_evidence"]
    assert compute["selected_gpu_or_cuda_ascii_markers_present"] is False
    assert compute["gpu_runtime_verified"] is False
    assert compute["packaged_gpu_capability_status"] == (
        "unverified_no_selected_gpu_or_cuda_ascii_markers"
    )
    assert payload["admission_gates"] == {
        "acceptance_eligible": False,
        "accuracy_admission_eligible": False,
        "gpu_admission_eligible": False,
        "matched_qm_speed_eligible": False,
    }
    assert payload["validation_scope"]["experimental_scoring_performed"] is False
    assert payload["validation_scope"]["wall_time_used_for_admission"] is False
    assert payload["capability_boundary"] == {
        "forces": False,
        "free_energy_sampling_protocol": False,
        "mlip": False,
        "potential_energy_surface": False,
        "scalar_solvation_estimator_only": True,
        "thermodynamic_protocol": False,
    }
    assert payload["route4_decision"]["integration_performed"] is False


def test_frozen_artifact_equals_fresh_pinned_release_audit():
    source_root = Path(
        os.environ.get(
            "MAPLE_ATOMICESE_AUDIT_ROOT",
            "/home/axie/.cache/maple-benchmarks/AtomicESE",
        )
    )
    if not source_root.is_dir() or not AUDIT.BWRAP_PATH.is_file():
        pytest.skip("Pinned AtomicESE checkout or bubblewrap is unavailable.")

    fresh = AUDIT.audit_release(source_root)
    rendered = json.dumps(fresh, indent=2, sort_keys=True) + "\n"

    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == fresh
    assert ARTIFACT.read_text(encoding="utf-8") == rendered
