from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import finalize_mnsol_aimnet2_smooth_partition_ddpcm as finalizer  # noqa: E402
import run_mnsol_aimnet2_smooth_partition_ddpcm as runner  # noqa: E402


def _records() -> list[dict[str, object]]:
    return [
        {
            "signed_error_kcal_mol": 0.0,
            "smooth_minus_reference_kcal_mol": 0.0,
            "atom_count": 1,
            "partition": "development" if index < 505 else "confirmation",
            "canonical_solvent": "water",
            "geometry_handle": f"private-{index % 395:03d}",
            "factor_count": 0,
            "required_algebraic_degree": 8,
        }
        for index in range(runner.EXPECTED_RECORD_COUNT)
    ]


def _raw_public(records: list[dict[str, object]]) -> dict[str, object]:
    checkpoint = {"sha256": "a" * 64, "bytes": 1}
    return {
        "artifact": runner.ARTIFACT,
        "schema_version": 1,
        "complete_panel": True,
        "execution_git_head": "b" * 40,
        "record_count": runner.EXPECTED_RECORD_COUNT,
        "unique_geometry_count": 395,
        "source": {
            "prior_artifact": runner.PRIOR_ARTIFACT,
            "prior_private_sha256": "c" * 64,
            "prior_public_sha256": "d" * 64,
            "checkpoint": checkpoint,
            "continuum_field_supplied_to_aimnet2": False,
            "electronic_scf_iteration": False,
        },
        "method": {
            "continuum": "smooth-partition-harmonic-ddpcm",
            "surface_lmax": 4,
            "partition_lmax": runner.AIMNET2_SMOOTH_PARTITION_PARTITION_LMAX,
            "maximum_algebraic_degree": (
                runner.HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE
            ),
            "transition_width_angstrom2": 0.18,
            "nonpolar": "PySCF-2.13.1 SMD-CDS reused from frozen prior artifact",
            "fit_or_calibration": False,
        },
        "factor_degree_preflight": {
            "maximum_factor_count": 0,
            "maximum_required_algebraic_degree": 8,
            "algebraic_degree_bound": (
                runner.HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE
            ),
            "failed_record_count": 0,
        },
        "aggregate_metrics": runner._metrics(records),
        "partition_metrics": runner._grouped_metrics(
            records, lambda record: str(record["partition"])
        ),
        "solvent_metrics": runner._grouped_metrics(
            records, lambda record: str(record["canonical_solvent"])
        ),
        "atom_count_bin_metrics": runner._grouped_metrics(records, runner._atom_bin),
        "claim_boundary": "aggregate-only synthetic fixture",
    }


def test_raw_public_reducer_replays_all_metric_groups_and_method_identity():
    records = _records()
    raw = _raw_public(records)
    private = {
        "execution_git_head": "b" * 40,
        "prior_private_sha256": "c" * 64,
        "prior_public_sha256": "d" * 64,
    }
    prior_public = {"checkpoint": raw["source"]["checkpoint"]}

    finalizer._validate_raw_public(
        raw, private=private, records=records, prior_public=prior_public
    )

    tampered = deepcopy(raw)
    tampered["method"]["surface_lmax"] = 5
    with pytest.raises(ValueError, match="method identity drifted"):
        finalizer._validate_raw_public(
            tampered,
            private=private,
            records=records,
            prior_public=prior_public,
        )


def test_shard_manifest_and_measurement_git_blob_hashes_are_content_bound():
    entries = (("0000.json", "a" * 64), ("0001.json", "b" * 64))
    expected = hashlib.sha256(
        (
            json.dumps(list(entries), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    ).hexdigest()
    assert finalizer._sha256_manifest(entries) == expected

    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    path = (
        "docs/implicit-solvation/benchmarks/run_mnsol_aimnet2_smooth_partition_ddpcm.py"
    )
    blob = subprocess.run(
        ("git", "show", f"{head}:{path}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert finalizer._git_source_hashes(ROOT, head, (path,)) == {
        path: hashlib.sha256(blob).hexdigest()
    }
