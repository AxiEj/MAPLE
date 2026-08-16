#!/usr/bin/env python3
"""Freeze the one-parameter MAPLE-CDS-W1 M2 candidate before fitting.

The creator reads only the already-locked hybrid-v3 preregistration and a
clean committed source tree.  It does not open prediction records, targets,
or the confirmation partition.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))

from maple.solvation.release.evidence import (
    RepositorySnapshot,
    canonical_json_sha256,
    committed_source_hashes,
    runtime_record,
)

PREREGISTRATION_ARTIFACT = "route2-maple-cds-w1-m2-prereg-v1"
PARENT_ARTIFACT = "route2-hybrid-smd-development-prereg-v3"
EXPECTED_WATER_COUNT = 306
EXPECTED_NONAQUEOUS_COUNT = 199
EXPECTED_RECORD_COUNT = EXPECTED_WATER_COUNT + EXPECTED_NONAQUEOUS_COUNT
ALPHA_MINIMUM = 0.0
ALPHA_MAXIMUM = 2.0
PRIMARY_MAE_THRESHOLD_KCAL_MOL = 1.5
MINIMUM_MAE_IMPROVEMENT_KCAL_MOL = 0.05

_SOURCE_FILES = (
    "maple/solvation/release/evidence.py",
    "tools/route2_release/analyze_hybrid_smd_components_v3.py",
    "tools/route2_release/create_maple_cds_w1_m2_preregistration.py",
    "tools/route2_release/fit_maple_cds_w1_m2_scale.py",
)

_TERMINAL_EVIDENCE_RELATIVE_PATHS = (
    "audits/preregistered-full-verification.json",
    "audits/independent-integrity-audit.json",
    "exits/aggregate.exit",
    "exits/integrity-audit.exit",
)


class M2PreregistrationError(RuntimeError):
    """Raised when the target-blind M2 contract cannot be frozen."""


def normalized_runtime_identity() -> tuple[dict[str, object], str]:
    """Return the timestamp-free numerical runtime bound by M2 evidence."""

    identity = runtime_record()
    identity.pop("generated_at_utc", None)
    return identity, canonical_json_sha256(identity)


def _outside_source(path: Path, source_root: Path) -> None:
    try:
        path.relative_to(source_root)
    except ValueError:
        return
    raise M2PreregistrationError("M2 preregistration must be outside the checkout.")


def _load_parent(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise M2PreregistrationError("Hybrid-v3 preregistration is not an object.")
    expected = {
        "artifact_id": PARENT_ARTIFACT,
        "schema_version": 3,
        "status": "locked-before-first-v3-hybrid-evaluation",
        "partition": "development",
        "record_count": EXPECTED_WATER_COUNT + EXPECTED_NONAQUEOUS_COUNT,
        "confirmation_partition_opened": False,
        "fitting_or_calibration_permitted": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise M2PreregistrationError(f"Parent field {key!r} drifted.")
    payload["_raw_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _write_json_exclusive_atomic(
    *, output: Path, payload: dict[str, object], snapshot: RepositorySnapshot
) -> None:
    """Atomically publish one read-only artifact without replacement."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        snapshot.assert_unchanged()
        os.link(temporary, output)
        directory_fd = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_preterminal_hybrid_evidence_state(
    *, parent: dict[str, object], evidence_root: Path
) -> dict[str, object]:
    """Bind M2 to a v3 evidence root before terminal evidence exists.

    Only record file names are observed.  Record contents, targets, predictions,
    aggregate metrics, and confirmation data are not opened by this operation.
    """

    resolved_root = evidence_root.expanduser().resolve(strict=True)
    expected_parent_path = resolved_root / "preregistration.json"
    if Path(str(parent.get("preregistration_path"))).resolve() != expected_parent_path:
        raise M2PreregistrationError(
            "Hybrid evidence root does not own the parent preregistration."
        )
    records_path = resolved_root / "records"
    if Path(str(parent.get("records_path"))).resolve() != records_path:
        raise M2PreregistrationError(
            "Hybrid evidence root does not own the parent record directory."
        )
    if not records_path.is_dir():
        raise M2PreregistrationError("Hybrid-v3 record directory is absent.")

    expected_names = {
        f"index-{index:03d}.json" for index in range(EXPECTED_RECORD_COUNT)
    }
    observed_names = sorted(path.name for path in records_path.glob("index-*.json"))
    unexpected_names = sorted(set(observed_names) - expected_names)
    if unexpected_names:
        raise M2PreregistrationError(
            "Hybrid-v3 record directory contains unexpected index files."
        )
    if len(observed_names) >= EXPECTED_RECORD_COUNT:
        raise M2PreregistrationError(
            "M2 must be preregistered before all 505 v3 records exist."
        )

    terminal_state = {
        relative: not (resolved_root / relative).exists()
        for relative in _TERMINAL_EVIDENCE_RELATIVE_PATHS
    }
    if not all(terminal_state.values()):
        raise M2PreregistrationError(
            "M2 must be preregistered before terminal v3 audit evidence exists."
        )
    return {
        "hybrid_evidence_root": str(resolved_root),
        "records_path": str(records_path),
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "record_file_count_at_lock": len(observed_names),
        "record_file_names_sha256_at_lock": canonical_json_sha256(observed_names),
        "record_file_names_observed": True,
        "record_contents_read": False,
        "terminal_evidence_absent": terminal_state,
    }


def create(args: argparse.Namespace) -> dict[str, object]:
    source_root = args.source_root.expanduser().resolve(strict=True)
    if source_root != _SOURCE_ROOT:
        raise M2PreregistrationError(
            "M2 creator must execute for the checkout containing the script."
        )
    parent_path = args.parent_hybrid_preregistration.expanduser().resolve(strict=True)
    evidence_root = args.hybrid_evidence_root.expanduser().resolve(strict=True)
    if parent_path != evidence_root / "preregistration.json":
        raise M2PreregistrationError(
            "Parent preregistration must be the preregistration of the bound "
            "hybrid evidence root."
        )
    output = args.output.expanduser().resolve()
    _outside_source(output, source_root)
    if output.exists():
        raise FileExistsError(output)

    snapshot = RepositorySnapshot.capture(source_root)
    parent = _load_parent(parent_path)
    parent_raw_sha256 = str(parent.pop("_raw_sha256"))
    evidence_state = _capture_preterminal_hybrid_evidence_state(
        parent=parent,
        evidence_root=evidence_root,
    )
    source_hashes = committed_source_hashes(snapshot, _SOURCE_FILES)
    runtime_identity, runtime_identity_sha256 = normalized_runtime_identity()
    payload: dict[str, object] = {
        "artifact": PREREGISTRATION_ARTIFACT,
        "schema_version": 1,
        "status": "locked-before-terminal-v3-evidence-and-first-m2-fit",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_profile_id": (
            "maple-cds-w1-stock-smd-global-scale-water-candidate-v1"
        ),
        "partition": "development-water-only",
        "expected_water_record_count": EXPECTED_WATER_COUNT,
        "expected_nonaqueous_record_count": EXPECTED_NONAQUEOUS_COUNT,
        "parent_hybrid_preregistration_path": str(parent_path),
        "parent_hybrid_preregistration_sha256": parent_raw_sha256,
        "parent_hybrid_source_git_head": parent["source_git_head"],
        "hybrid_evidence_state_at_lock": evidence_state,
        "source_root": str(source_root),
        "source_git_head": snapshot.head,
        "source_git_tree": snapshot.tree,
        "source_files_sha256": source_hashes,
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_identity_sha256,
        "experimental_targets_read_by_preregistration": False,
        "hybrid_prediction_records_read_by_preregistration": False,
        "hybrid_record_file_names_observed_by_preregistration": True,
        "confirmation_partition_opened": False,
        "fit_contract": {
            "target": "experimental_delta_g_kcal_mol-continuum_polarization_kcal_mol",
            "predictor": "stock_smd_cds_kcal_mol",
            "model": "continuum_polarization_kcal_mol+alpha*stock_smd_cds_kcal_mol",
            "objective": "unweighted-ordinary-least-squares",
            "closed_form": "alpha=sum(x*y)/sum(x*x)",
            "intercept": False,
            "record_scope": "all-and-only-306-water-development-records",
            "alpha_minimum_inclusive": ALPHA_MINIMUM,
            "alpha_maximum_inclusive": ALPHA_MAXIMUM,
            "out_of_domain_action": "reject-candidate-without-clipping",
            "standard_state": "1M-ideal-gas-to-1M-ideal-solution",
            "standard_state_correction_kcal_mol": 0.0,
        },
        "development_decision_rule": {
            "primary_mae_threshold_kcal_mol": PRIMARY_MAE_THRESHOLD_KCAL_MOL,
            "minimum_mae_improvement_over_better_of_m0_m1_kcal_mol": (
                MINIMUM_MAE_IMPROVEMENT_KCAL_MOL
            ),
            "both_conditions_required": True,
            "tail_metrics": "report-only-not-post-hoc-gates",
        },
        "nonaqueous_policy": "retain-stock-smd-cds-m1-unchanged",
        "confirmation_policy": (
            "remain-sealed-until-alpha-and-all-confirmation-gates-are-frozen"
        ),
        "claim_boundary": (
            "One preregistered water-only scale diagnostic/candidate. It changes "
            "neither hybrid electrostatics nor stock CDS geometry, has no fitted "
            "intercept, and does not establish force or production admission."
        ),
    }
    payload["self_sha256"] = canonical_json_sha256(payload)
    if (
        _capture_preterminal_hybrid_evidence_state(
            parent=parent,
            evidence_root=evidence_root,
        )
        != evidence_state
    ):
        raise M2PreregistrationError(
            "Hybrid-v3 evidence changed while M2 preregistration was captured."
        )
    _write_json_exclusive_atomic(output=output, payload=payload, snapshot=snapshot)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-hybrid-preregistration", type=Path, required=True)
    parser.add_argument("--hybrid-evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = create(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
