#!/usr/bin/env python
"""Verify that real-backend PBC-MD production reports cover the claimed matrix.

`production_validation.py` proves one calculator target at a time.  This script is
an aggregate release gate: it rejects a production PBC-MD claim unless every
required real backend/model-option target has a current, clean, non-smoke report
whose full acceptance matrix genuinely passed.

Examples
--------
    # Check validation/reports/ against validation/required_pbc_backends.toml
    python scripts/check_production_backend_matrix.py

    # Check an explicit report bundle produced on a release runner
    python scripts/check_production_backend_matrix.py --reports-dir /path/to/reports
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MATRIX = _REPO_ROOT / "validation" / "required_pbc_backends.toml"
_DEFAULT_REPORTS = _REPO_ROOT / "validation" / "reports"
_REQUIRED_STRESS_COMPONENTS = ["xx", "yy", "zz", "yz", "xz", "xy"]


@dataclass(frozen=True)
class Target:
    id: str
    model: str
    model_options: dict[str, Any]


@dataclass(frozen=True)
class ReportCandidate:
    path: Path
    payload: dict[str, Any]


def _git_head_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _version_tuple(value: Any) -> tuple[int, ...] | None:
    text = str(value or "")
    if text.startswith("smoke"):
        return None
    parts = text.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _version_at_least(actual: Any, minimum: Any) -> bool:
    actual_tuple = _version_tuple(actual)
    minimum_tuple = _version_tuple(minimum)
    if actual_tuple is None or minimum_tuple is None:
        return False
    length = max(len(actual_tuple), len(minimum_tuple))
    actual_tuple = actual_tuple + (0,) * (length - len(actual_tuple))
    minimum_tuple = minimum_tuple + (0,) * (length - len(minimum_tuple))
    return actual_tuple >= minimum_tuple


def _load_matrix(path: Path) -> tuple[list[Target], str, list[str]]:
    data = tomllib.loads(Path(path).read_text())
    targets = [
        Target(
            id=str(item["id"]),
            model=str(item["model"]),
            model_options=dict(item.get("model_options") or {}),
        )
        for item in data.get("targets", [])
    ]
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate target id(s): {', '.join(duplicates)}")
    return targets, str(data.get("minimum_thresholds_version", "")), list(data.get("required_classes", []))


def _iter_report_payloads(root: Path) -> Iterable[ReportCandidate]:
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict) and "results" in payload:
            yield ReportCandidate(path=path, payload=payload)


def _report_model(payload: dict[str, Any]) -> str | None:
    target = payload.get("validation_target") or {}
    return target.get("model") or payload.get("calculator")


def _report_model_options(payload: dict[str, Any]) -> dict[str, Any]:
    # Top-level ``model_options`` is the exact command-line target validated by
    # production_validation.py. ``validation_target.model_options`` may add
    # derived provenance such as an AIMNet DSF cutoff for audit readability, so
    # use it only as a compatibility fallback for older report files.
    options = payload.get("model_options")
    if options is None:
        options = (payload.get("validation_target") or {}).get("model_options")
    return dict(options or {})


def _options_match(report_options: dict[str, Any], expected: dict[str, Any]) -> bool:
    return report_options == expected


def _result_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in payload.get("results", [])
        if isinstance(item, dict)
    }


def _stress_fd_errors(stress: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not stress.get("passed") or stress.get("status") != "pass":
        errors.append("stress_finite_difference did not pass")
        return errors
    components = stress.get("metrics", {}).get("components")
    if not isinstance(components, list):
        return ["stress_finite_difference metrics.components missing"]
    names = [str(item.get("component")) for item in components]
    if names != _REQUIRED_STRESS_COMPONENTS:
        errors.append(
            "stress_finite_difference did not report full ASE Voigt order "
            f"{_REQUIRED_STRESS_COMPONENTS}: got {names}"
        )
    for item in components:
        name = str(item.get("component"))
        if item.get("passed") is not True:
            errors.append(f"stress component {name} did not pass")
        for delta in item.get("per_delta", []) or []:
            if delta.get("ok") is not True:
                errors.append(f"stress component {name} delta {delta.get('delta')} not ok")
            if delta.get("sign_ok") is not True:
                errors.append(f"stress component {name} delta {delta.get('delta')} sign mismatch")
    return errors


def _aimnet_cutoff_errors(target: Target, payload: dict[str, Any]) -> list[str]:
    if not target.model.startswith("aimnet2"):
        return []
    expected_method = str(target.model_options.get("coulomb", "")).lower()
    contract = payload.get("calculator_contract") or {}
    method = str(contract.get("long_range_method") or "").lower()
    errors: list[str] = []
    if expected_method and method != expected_method:
        errors.append(
            f"calculator_contract.long_range_method={method!r} does not match "
            f"target coulomb={expected_method!r}"
        )
    if expected_method in {"ewald", "pme"}:
        if contract.get("local_descriptor_cutoff_A") is None:
            errors.append("AIMNet Ewald/PME report must record local_descriptor_cutoff_A")
    elif expected_method == "dsf":
        if (
            contract.get("long_range_coulomb_cutoff_A") is None
            and contract.get("short_range_realspace_cutoff_A") is None
            and contract.get("neighbor_cutoff_A") is None
        ):
            errors.append("AIMNet DSF report must record the real-space cutoff admitted by MIC")
    return errors


def _validate_candidate(
    target: Target,
    candidate: ReportCandidate,
    *,
    minimum_thresholds_version: str,
    required_classes: list[str],
    head_commit: str | None,
    allow_old_commit: bool,
) -> list[str]:
    payload = candidate.payload
    errors: list[str] = []
    env = payload.get("environment") or {}
    summary = payload.get("summary") or {}
    release_gate = payload.get("release_gate") or {}
    results = _result_by_name(payload)

    if payload.get("validation_mode") != "production-validation":
        errors.append("validation_mode is not production-validation")
    if payload.get("production_validated") is not True:
        errors.append("production_validated is not true")
    if release_gate.get("ready") is not True:
        errors.append("release_gate.ready is not true")
    if payload.get("overall") != "PASS":
        errors.append("overall is not PASS")
    if not _version_at_least(payload.get("thresholds_version"), minimum_thresholds_version):
        errors.append(
            f"thresholds_version {payload.get('thresholds_version')!r} is below "
            f"required {minimum_thresholds_version!r} or is smoke"
        )
    if env.get("maple_git_dirty") is not False:
        errors.append("environment.maple_git_dirty is not false")
    if not allow_old_commit and head_commit and env.get("maple_git_commit") != head_commit:
        errors.append("environment.maple_git_commit does not equal current HEAD")
    if summary.get("n_fail") != 0:
        errors.append("summary.n_fail is not 0")
    if summary.get("n_skip") != 0:
        errors.append("summary.n_skip is not 0")
    if summary.get("barostat_clamp_count") != 0:
        errors.append("summary.barostat_clamp_count is not 0")

    missing = [name for name in required_classes if name not in results]
    if missing:
        errors.append(f"missing acceptance class(es): {', '.join(missing)}")
    for name in required_classes:
        result = results.get(name)
        if result is not None and (result.get("passed") is not True or result.get("status") != "pass"):
            errors.append(f"acceptance class {name} did not pass")

    stress = results.get("stress_finite_difference")
    if stress is None:
        errors.append("stress_finite_difference result missing")
    else:
        errors.extend(_stress_fd_errors(stress))

    clamp = results.get("barostat_clamp_free")
    if clamp is not None and (clamp.get("metrics") or {}).get("clamp_count") != 0:
        errors.append("barostat_clamp_free clamp_count is not 0")

    contract = payload.get("calculator_contract") or {}
    expected_units = {
        "energy_unit": "Ha",
        "force_unit": "Ha/A",
        "stress_unit": "eV/A^3",
    }
    for key, expected in expected_units.items():
        if contract.get(key) != expected:
            errors.append(f"calculator_contract.{key}={contract.get(key)!r}, expected {expected!r}")

    if not payload.get("manifest_files"):
        errors.append("manifest_files is empty; per-run provenance artifacts were not preserved")
    if not payload.get("validation_system"):
        errors.append("validation_system missing")

    report_options = _report_model_options(payload)
    if not _options_match(report_options, target.model_options):
        errors.append(
            f"report model_options {report_options!r} do not exactly match required {target.model_options!r}"
        )
    errors.extend(_aimnet_cutoff_errors(target, payload))
    return errors


def _find_matching_reports(target: Target, reports: list[ReportCandidate]) -> list[ReportCandidate]:
    matches = []
    for candidate in reports:
        payload = candidate.payload
        if _report_model(payload) != target.model:
            continue
        if not _options_match(_report_model_options(payload), target.model_options):
            continue
        matches.append(candidate)
    return matches


def check_matrix(
    *,
    matrix_path: Path,
    reports_dir: Path,
    allow_old_commit: bool = False,
) -> tuple[bool, list[str]]:
    targets, minimum_version, required_classes = _load_matrix(matrix_path)
    reports = list(_iter_report_payloads(reports_dir))
    head = _git_head_commit()
    messages: list[str] = []
    ok = True

    if not targets:
        return False, [f"no targets declared in {matrix_path}"]
    if not reports:
        return False, [f"no report JSON files found under {reports_dir}"]

    for target in targets:
        candidates = _find_matching_reports(target, reports)
        if not candidates:
            messages.append(f"[FAIL] {target.id}: no matching report for {target.model} {target.model_options}")
            ok = False
            continue
        candidate_messages = []
        target_ok = False
        for candidate in candidates:
            errors = _validate_candidate(
                target,
                candidate,
                minimum_thresholds_version=minimum_version,
                required_classes=required_classes,
                head_commit=head,
                allow_old_commit=allow_old_commit,
            )
            if not errors:
                messages.append(f"[PASS] {target.id}: {candidate.path}")
                target_ok = True
                break
            candidate_messages.append(f"{candidate.path}: " + "; ".join(errors))
        if not target_ok:
            ok = False
            messages.append(f"[FAIL] {target.id}: no valid production report")
            messages.extend(f"  - {message}" for message in candidate_messages)
    return ok, messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(_DEFAULT_MATRIX), help="required backend matrix TOML")
    parser.add_argument("--reports-dir", default=str(_DEFAULT_REPORTS), help="directory containing report JSON files")
    parser.add_argument(
        "--allow-old-commit",
        action="store_true",
        help="accept clean reports from older commits (for historical audits only; not a release gate)",
    )
    args = parser.parse_args(argv)

    try:
        ok, messages = check_matrix(
            matrix_path=Path(args.matrix),
            reports_dir=Path(args.reports_dir),
            allow_old_commit=bool(args.allow_old_commit),
        )
    except Exception as exc:
        print(f"production backend matrix: FAIL ({type(exc).__name__}: {exc})")
        return 2

    print(f"production backend matrix: {'PASS' if ok else 'FAIL'}")
    for message in messages:
        print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
