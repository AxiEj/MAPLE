from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from maple.solvation.release import canonical_json_sha256
import tools.route2_release.audit_source_hash_lineage as audit_module
from tools.route2_release.audit_source_hash_lineage import (
    AuditInputError,
    audit_source_hash_lineage,
)


SCRIPT = Path(__file__).parents[2] / "tools/route2_release/audit_source_hash_lineage.py"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(
        ["git", "-C", root, "config", "user.email", "audit@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", root, "config", "user.name", "Audit Test"], check=True
    )
    (root / "source.py").write_bytes(b"print('stable')\n")
    (root / "nested").mkdir()
    (root / "nested" / "model.py").write_bytes(b"MODEL = 1\n")
    (root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    subprocess.run(["git", "-C", root, "add", "."], check=True)
    subprocess.run(["git", "-C", root, "commit", "-qm", "fixture"], check=True)
    return root


def _write_evidence(
    root: Path,
    ledger: dict[str, str],
    *,
    field: str = "source_files_sha256",
) -> Path:
    path = root / "evidence.json"
    path.write_text(
        json.dumps({"artifact_id": "test-evidence-v1", field: ledger}),
        encoding="utf-8",
    )
    return path


def _run_cli(
    repository: Path, evidence: Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repository),
            "--evidence",
            str(evidence),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )


def _working_tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def test_classifies_match_drift_and_missing_and_allows_dirty_repository(
    repository: Path,
) -> None:
    evidence = _write_evidence(
        repository,
        {
            "source.py": _sha256(b"print('stable')\n"),
            "nested/model.py": _sha256(b"different\n"),
            "removed.py": "0" * 64,
        },
    )

    result = audit_source_hash_lineage(repository, evidence)

    assert result["all_match"] is False
    assert result["counts"] == {
        "total": 3,
        "matching": 1,
        "drifted": 1,
        "missing": 1,
    }
    assert result["matching"] == ["source.py"]
    assert result["drifted"] == [
        {
            "path": "nested/model.py",
            "expected_sha256": _sha256(b"different\n"),
            "actual_sha256": _sha256(b"MODEL = 1\n"),
        }
    ]
    assert result["missing"] == [
        {"path": "removed.py", "expected_sha256": "0" * 64}
    ]
    assert result["repository"]["dirty"] is True


@pytest.mark.parametrize(
    "path,digest",
    [
        ("../escape.py", "0" * 64),
        ("/absolute.py", "0" * 64),
        ("C:/absolute.py", "0" * 64),
        ("source.py", "A" * 64),
        ("source.py", "abc"),
    ],
)
def test_rejects_invalid_paths_and_digests(
    repository: Path, path: str, digest: str
) -> None:
    evidence = _write_evidence(repository, {path: digest})

    with pytest.raises(AuditInputError):
        audit_source_hash_lineage(repository, evidence)


def test_custom_field_and_report_digest_are_deterministic(repository: Path) -> None:
    evidence = _write_evidence(
        repository,
        {"source.py": _sha256(b"print('stable')\n")},
        field="implementation_hashes",
    )

    first = audit_source_hash_lineage(
        repository, evidence, source_hash_field="implementation_hashes"
    )
    second = audit_source_hash_lineage(
        repository, evidence, source_hash_field="implementation_hashes"
    )

    assert first == second
    digest = first.pop("report_sha256")
    assert digest == canonical_json_sha256(first)
    assert len(digest) == 64


def test_cli_exit_codes_output_boundary_and_no_repository_mutation(
    repository: Path, tmp_path: Path
) -> None:
    evidence = _write_evidence(
        repository, {"source.py": _sha256(b"print('stable')\n")}
    )
    before = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    bytes_before = _working_tree_bytes(repository)
    output = tmp_path / "audit.json"

    matched = _run_cli(repository, evidence, "--output", str(output))
    after = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert matched.returncode == 0, matched.stderr
    assert json.loads(matched.stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert after == before
    assert _working_tree_bytes(repository) == bytes_before

    inside = _run_cli(
        repository, evidence, "--output", str(repository / "audit.json")
    )
    assert inside.returncode == 2
    assert not (repository / "audit.json").exists()

    (repository / "source.py").write_bytes(b"drifted\n")
    drifted = _run_cli(repository, evidence)
    assert drifted.returncode == 1
    assert json.loads(drifted.stdout)["all_match"] is False

    malformed = _write_evidence(repository, {"../escape.py": "0" * 64})
    invalid = _run_cli(repository, malformed)
    assert invalid.returncode == 2
    assert invalid.stdout == ""
    assert "error:" in invalid.stderr


def test_output_rejects_symlink_hardlinks_and_evidence_identity(
    repository: Path, tmp_path: Path
) -> None:
    evidence = tmp_path / "external-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "artifact_id": "external-v1",
                "source_files_sha256": {
                    "source.py": _sha256(b"print('stable')\n")
                },
            }
        ),
        encoding="utf-8",
    )

    assert _run_cli(
        repository, evidence, "--output", str(evidence)
    ).returncode == 2

    evidence_alias = tmp_path / "evidence-alias.json"
    os.link(evidence, evidence_alias)
    assert _run_cli(
        repository, evidence, "--output", str(evidence_alias)
    ).returncode == 2

    repository_alias = tmp_path / "repository-alias.py"
    os.link(repository / "source.py", repository_alias)
    assert _run_cli(
        repository, evidence, "--output", str(repository_alias)
    ).returncode == 2
    assert repository_alias.read_bytes() == b"print('stable')\n"

    target = tmp_path / "symlink-target.json"
    target.write_text("preserve me", encoding="utf-8")
    symlink = tmp_path / "output-link.json"
    symlink.symlink_to(target)
    assert _run_cli(
        repository, evidence, "--output", str(symlink)
    ).returncode == 2
    assert target.read_text(encoding="utf-8") == "preserve me"


def test_external_evidence_mutation_is_detected(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "external-evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "artifact_id": "external-v1",
                "source_files_sha256": {
                    "source.py": _sha256(b"print('stable')\n")
                },
            }
        ),
        encoding="utf-8",
    )

    def mutate_evidence() -> None:
        evidence.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(audit_module, "_after_initial_capture", mutate_evidence)
    with pytest.raises(AuditInputError, match="evidence artifact changed"):
        audit_source_hash_lineage(repository, evidence)


def test_dirty_untracked_source_race_after_output_is_detected(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    volatile = repository / "volatile.py"
    volatile.write_bytes(b"VALUE = 1\n")
    original = volatile.stat()
    evidence = _write_evidence(
        repository,
        {
            "source.py": _sha256(b"print('stable')\n"),
            "volatile.py": _sha256(b"VALUE = 1\n"),
        },
    )
    status_before = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def mutate_source_after_output() -> None:
        volatile.write_bytes(b"VALUE = 2\n")
        os.utime(volatile, ns=(original.st_atime_ns, original.st_mtime_ns))

    monkeypatch.setattr(
        audit_module, "_after_output_write", mutate_source_after_output
    )
    output = tmp_path / "race-audit.json"
    returncode = audit_module.main(
        [
            "--repo-root",
            str(repository),
            "--evidence",
            str(evidence),
            "--output",
            str(output),
        ]
    )

    status_after = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert returncode == 2
    assert "source file changed during audit" in capsys.readouterr().err
    assert status_after == status_before
    assert output.is_file()


def test_stable_source_symlink_is_rejected(repository: Path) -> None:
    ignored = repository / "ignored"
    ignored.mkdir()
    link = ignored / "source-link.py"
    link.symlink_to("../source.py")
    evidence = _write_evidence(
        repository,
        {"ignored/source-link.py": _sha256(b"print('stable')\n")},
    )

    with pytest.raises(AuditInputError, match="must not contain symbolic links"):
        audit_source_hash_lineage(repository, evidence)


def test_source_directory_retarget_from_inside_to_outside_is_detected(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = repository / "ignored"
    active = ignored / "active"
    active.mkdir(parents=True)
    (active / "source.py").write_bytes(b"ORIGIN = 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.py").write_bytes(b"OUTSIDE = 1\n")
    evidence = _write_evidence(
        repository,
        {"ignored/active/source.py": _sha256(b"ORIGIN = 1\n")},
    )
    status_before = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def retarget_outside() -> None:
        active.rename(ignored / "original")
        active.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(audit_module, "_after_initial_capture", retarget_outside)
    with pytest.raises(AuditInputError, match="must not contain symbolic links"):
        audit_source_hash_lineage(repository, evidence)
    status_after = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status_after == status_before


def test_source_directory_retarget_to_different_inside_target_is_detected(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ignored = repository / "ignored"
    active = ignored / "active"
    alternate = ignored / "alternate"
    active.mkdir(parents=True)
    alternate.mkdir()
    (active / "source.py").write_bytes(b"ORIGIN = 1\n")
    (alternate / "source.py").write_bytes(b"ALTERNATE = 1\n")
    evidence = _write_evidence(
        repository,
        {"ignored/active/source.py": _sha256(b"ORIGIN = 1\n")},
    )
    status_before = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def retarget_inside() -> None:
        active.rename(ignored / "original")
        active.symlink_to("alternate", target_is_directory=True)

    monkeypatch.setattr(audit_module, "_after_initial_capture", retarget_inside)
    with pytest.raises(AuditInputError, match="must not contain symbolic links"):
        audit_source_hash_lineage(repository, evidence)
    status_after = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status_after == status_before
