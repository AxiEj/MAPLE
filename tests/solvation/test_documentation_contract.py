from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_pilot_documentation_does_not_claim_a_certified_current_default():
    benchmark = (
        REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/README.md"
    ).read_text(encoding="utf-8")
    validation = (
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md"
    ).read_text(encoding="utf-8")

    assert "## Current default:" not in benchmark
    assert "temporary experimental default" in benchmark.lower()
    assert "current AM1-BCC/OBC-II/ACE default" not in validation


def test_qeq_documentation_freezes_it_as_explicit_experimental_only():
    overview = (
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(overview.split())

    assert "QEq/CQEq are frozen experimental research profiles" in overview
    assert "never selected as defaults or provider fallbacks" in normalized
