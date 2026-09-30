#!/usr/bin/env python3
"""Generate the archival-only Route 1 source-compatibility receipt.

The generator rolls an immutable previous receipt forward.  It never changes
historical evidence, authorizes execution, or infers scientific parity from a
source hash.  Existing record metadata is copied verbatim; only hash history,
the current hash, and an explicitly declared subsequent-change annotation may
be added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark_core import (
    artifact_content_sha256,
    load_json,
    sha256_file,
    write_json_atomic,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[2]
PREVIOUS_RECEIPT = (
    BENCHMARK_DIR / "route1-production-safety-source-compatibility-2026-09-27.json"
)
OUTPUT_RECEIPT = (
    BENCHMARK_DIR / "route1-production-safety-source-compatibility-2026-09-30.json"
)
PREVIOUS_RECEIPT_SHA256 = (
    "4658b7dc8e0d7c1fa510e62e391d04aab9d076b084183b590384055b651fc17d"
)


SHARED_REWRITE_PATHS = frozenset(
    {
        "maple/function/calculator/aimnet/_aimnet2_calculator.py",
        "maple/function/calculator/ani/_ani_calculator.py",
        "maple/function/calculator/calculator_base.py",
        "maple/function/calculator/mace/_mace_calculator.py",
        "maple/function/calculator/set_calculator.py",
        "maple/function/read/command_control.py",
    }
)
INDEPENDENT_OBC2_PATHS = frozenset(
    {
        "maple/function/calculator/extra_correction/implicit/__init__.py",
        "maple/function/calculator/extra_correction/implicit/correction.py",
        "maple/function/calculator/extra_correction/implicit/obc2_parameters.py",
        "maple/function/calculator/extra_correction/implicit/openmm_gb.py",
        "maple/function/calculator/extra_correction/implicit/radii.py",
        "maple/function/calculator/extra_correction/implicit/torch_obc2.py",
    }
)
UMA_PATHS = frozenset({"maple/function/calculator/uma/_uma_calculator.py"})


NEW_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "path": "pyproject.toml",
        "historical_sha256": [
            "c2a783dfd40c20ebcf0d0a6048edc441ded0e24de692142e6a6df6d0ac7867f0"
        ],
        "change_class": "independent-obc2-extra-and-provenance-packaging",
        "reason": (
            "Adds the explicit independent OBC-II Torch extra and packages provenance "
            "data; it does not qualify a runtime cell or promote a scientific claim."
        ),
    },
    {
        "path": "maple/function/calculator/mace/_common.py",
        "historical_sha256": [
            "9d62c9129855ae19e978a93caea7df9a189451a22abd639fa0b09745ce00eb57"
        ],
        "change_class": "shared-analytic-operation-identity-adapter",
        "reason": (
            "Adds typed gas-operation identity support used by the separately "
            "qualified shared composition path. Historical batch and conformer "
            "results are not recomputed or admitted for reuse."
        ),
    },
    {
        "path": "maple/function/calculator/analytic_protocol.py",
        "historical_sha256": [],
        "change_class": "new-typed-analytic-identity-and-composition-algebra",
        "reason": "New post-execution protocol code; no historical numerical claim exists.",
    },
    {
        "path": "maple/function/calculator/solvent_composition.py",
        "historical_sha256": [],
        "change_class": "new-model-neutral-solvent-composition-boundary",
        "reason": "New post-execution orchestration code; no historical numerical claim exists.",
    },
    {
        "path": "maple/function/calculator/analytic_admission.py",
        "historical_sha256": [],
        "change_class": "new-exact-cell-analytic-admission-resolver",
        "reason": "New fail-closed admission code; it does not retroactively qualify old evidence.",
    },
    {
        "path": "maple/function/calculator/_qualified_analytic_cells.py",
        "historical_sha256": [],
        "change_class": "new-empty-or-reviewed-exact-cell-admission-data",
        "reason": "New admission data surface; historical execution remains unauthorized.",
    },
    {
        "path": "maple/function/calculator/extra_correction/implicit/independent_obc2.py",
        "historical_sha256": [],
        "change_class": "new-openmm-independent-obc2-provider",
        "reason": "New candidate provider implementation; no old parity or scientific claim is reused.",
    },
    {
        "path": "maple/function/calculator/extra_correction/implicit/obc2_profile.py",
        "historical_sha256": [],
        "change_class": "new-independent-obc2-profile-contract",
        "reason": "New post-execution profile code; historical evidence remains immutable.",
    },
    {
        "path": "maple/function/calculator/extra_correction/implicit/obc2_topology.py",
        "historical_sha256": [],
        "change_class": "new-independent-obc2-topology-contract",
        "reason": "New post-execution topology code; historical evidence remains immutable.",
    },
    {
        "path": "maple/function/calculator/uma/_analytic_adapter.py",
        "historical_sha256": [],
        "change_class": "new-uma-analytic-operation-adapter",
        "reason": "New integration code only; FAIR-Chem capability is not treated as qualification.",
    },
    {
        "path": "maple/function/calculator/extra_correction/implicit/data/__init__.py",
        "historical_sha256": [],
        "change_class": "new-packaged-independent-obc2-provenance-data",
        "reason": (
            "New package-data namespace only; it records implementation provenance "
            "and does not establish parity or a scientific claim."
        ),
    },
    {
        "path": "maple/function/calculator/extra_correction/implicit/data/obc2_profile_manifest.json",
        "historical_sha256": [],
        "change_class": "new-packaged-independent-obc2-provenance-data",
        "reason": (
            "New packaged OBC-II profile provenance only; historical results are "
            "not recomputed or admitted for reuse."
        ),
    },
    {
        "path": "maple/function/calculator/extra_correction/implicit/data/OPENMM_CUSTOMGBFORCES_LICENSE.txt",
        "historical_sha256": [],
        "change_class": "new-packaged-upstream-license-provenance",
        "reason": (
            "New packaged upstream license notice only; it carries no numerical or "
            "scientific qualification claim."
        ),
    },
    {
        "path": "tests/solvation/test_torch_obc2_artifacts.py",
        "historical_sha256": [
            "9866dc3c3d8f84b33f1f18ea428769f3427e7300dda47a0cc0ac3c5462cbfcfa"
        ],
        "change_class": "archival-test-current-evidence-separation",
        "reason": (
            "Replaces direct comparison of historical source hashes to revised code "
            "with compatibility-audited history plus mandatory fresh validation."
        ),
    },
    {
        "path": "tests/solvation/test_direct_solvent_contract.py",
        "historical_sha256": [
            "d2e74905fa6a8b6b7728bf4ea05332a93b21c10f21910cb414ef725a71995fa3"
        ],
        "change_class": "postexecution-shared-composition-contract-test-evolution",
        "reason": (
            "Test evolution for the shared solvent-composition boundary only; the "
            "historical OBC-II revalidation remains immutable archival evidence."
        ),
    },
    {
        "path": "tests/solvation/test_torch_obc2_parity.py",
        "historical_sha256": [
            "a578e09cf4f067b406dbdbd2c164c2668d27b8388ed18f0e45edd09b0fce4645"
        ],
        "change_class": "postexecution-independent-obc2-parity-test-evolution",
        "reason": (
            "Test evolution for the independent OBC-II candidate only; it does not "
            "authorize reuse of the historical validation claim."
        ),
    },
    {
        "path": "docs/implicit-solvation/benchmarks/generate_source_compatibility.py",
        "historical_sha256": [],
        "change_class": "new-deterministic-archival-receipt-generator",
        "reason": "Audit tooling only; it cannot authorize execution, sealing, or claim reuse.",
    },
)


def _subsequent_annotation(path: str) -> dict[str, str]:
    if path in SHARED_REWRITE_PATHS:
        return {
            "subsequent_change_class": "shared-model-neutral-solvent-composition-integration",
            "subsequent_reason": (
                "2026-09-30 source evolution connects existing gas backends to one "
                "typed solvent composition boundary. This is an audited "
                "post-execution change only and does not establish old numerical parity."
            ),
        }
    if path in INDEPENDENT_OBC2_PATHS:
        return {
            "subsequent_change_class": "openmm-independent-obc2-rewrite-integration",
            "subsequent_reason": (
                "2026-09-30 source evolution adds or connects the independent OBC-II "
                "candidate path. Historical results and scientific claims are not reused."
            ),
        }
    if path in UMA_PATHS:
        return {
            "subsequent_change_class": "uma-analytic-derivative-and-shared-solvent-integration",
            "subsequent_reason": (
                "2026-09-30 source evolution integrates operation-scoped FAIR-Chem "
                "derivatives and shared solvent composition. It records no qualification."
            ),
        }
    return {}


def _roll_record(record: dict[str, Any]) -> dict[str, Any]:
    rolled = dict(record)
    path = rolled["path"]
    source = REPOSITORY_ROOT / path
    if not source.is_file():
        raise FileNotFoundError(f"Compatibility source does not exist: {path}")
    current = sha256_file(source)
    history = list(rolled["historical_sha256"])
    previous_current = rolled["current_sha256"]
    if previous_current != current:
        history.append(previous_current)
        rolled.update(_subsequent_annotation(path))
    rolled["historical_sha256"] = list(dict.fromkeys(history))
    rolled["current_sha256"] = current
    return rolled


def build_receipt() -> dict[str, Any]:
    if sha256_file(PREVIOUS_RECEIPT) != PREVIOUS_RECEIPT_SHA256:
        raise ValueError("The immutable 2026-09-27 compatibility receipt changed.")
    previous = load_json(PREVIOUS_RECEIPT)
    records = [_roll_record(record) for record in previous["source_changes"]]
    prior_paths = {record["path"] for record in records}
    for declared in NEW_RECORDS:
        if declared["path"] in prior_paths:
            raise ValueError(f"New compatibility record duplicates {declared['path']}.")
        record = dict(declared)
        source = REPOSITORY_ROOT / record["path"]
        if not source.is_file():
            raise FileNotFoundError(
                f"Compatibility source does not exist: {record['path']}"
            )
        record.update(
            current_sha256=sha256_file(source),
            historical_numerical_results_recomputed=False,
            scientific_claim_reuse_authorized=False,
        )
        records.append(record)
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-postexecution-production-safety-source-compatibility",
        "recorded_date": "2026-09-30",
        "last_updated_date": "2026-09-30",
        "historical_evidence_bytes_unchanged": True,
        "frozen_results_recomputed": False,
        "scientific_claim_promoted": False,
        "route1_formula_changed": False,
        "gas_phase_mm_energy_added": False,
        "hydration_label_residual_added": False,
        "qrrho_v8_status_remains_failed_closed": True,
        "scope": (
            "Documents audited post-execution source evolution for independent OBC-II, "
            "shared gas-plus-solvent composition, exact-cell admission, and UMA analytic "
            "integration. It does not recompute historical results, establish numerical "
            "parity, authorize execution or scoring under an old freeze, or promote any "
            "scientific claim."
        ),
        "source_changes": records,
        "supersedes": {
            "file": PREVIOUS_RECEIPT.name,
            "file_sha256": PREVIOUS_RECEIPT_SHA256,
        },
    }
    artifact["content_sha256"] = artifact_content_sha256(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    artifact = build_receipt()
    if args.check:
        if not OUTPUT_RECEIPT.is_file() or load_json(OUTPUT_RECEIPT) != artifact:
            raise SystemExit("Source-compatibility receipt is not current.")
    else:
        write_json_atomic(OUTPUT_RECEIPT, artifact)
    print(
        json.dumps(
            {
                "path": str(OUTPUT_RECEIPT),
                "content_sha256": artifact["content_sha256"],
                "records": len(artifact["source_changes"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
