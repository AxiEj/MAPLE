#!/usr/bin/env python3
"""Run the fixed FreeSolv-10 CIGIN property-prediction development pilot.

This records direct scalar output from the official CIGIN checkpoint.  CIGIN
identifies FreeSolv as training provenance but does not publish record-level
training accounting, so this is overlap-unknown development evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.benchmarking import (  # noqa: E402
    BenchmarkIdentity,
    BenchmarkQuantity,
    ModelTrainingEvidence,
    ValidationPanel,
    ValidationStage,
    audit_training_overlap,
    summarize_runtime_smoke,
)
from maple.function.solvfe import (  # noqa: E402
    CIGIN_SOURCE_REVISION,
    CIGINPropertyAdapter,
)

PILOT_INPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "freesolv10-2026-07-22.json"
)
WATER_SMILES = "O"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _load_compounds(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compounds = payload.get("compounds")
    if not isinstance(compounds, list) or len(compounds) != 10:
        raise ValueError(
            "The fixed CIGIN FreeSolv pilot must contain exactly 10 records."
        )
    identifiers = [str(row.get("id", "")).strip() for row in compounds]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "The fixed CIGIN FreeSolv pilot has invalid compound identifiers."
        )
    return compounds


def _identity(compounds: list[dict[str, Any]]) -> BenchmarkIdentity:
    return BenchmarkIdentity(
        dataset="FreeSolv",
        dataset_version="0.52 / fixed MAPLE 10-molecule chemistry-stratified pilot",
        record_ids=tuple(str(row["id"]) for row in compounds),
        temperature_kelvin=298.15,
        standard_state="FreeSolv experimental convention; property prediction only",
        protonation_policy="neutral molecules encoded by the fixed FreeSolv pilot SMILES",
        tautomer_policy="as represented by the fixed FreeSolv pilot SMILES",
        conformer_policy="not applicable; CIGIN takes direct canonical SMILES",
        geometry_protocol="not applicable; no supplied coordinates or conformers",
        solvent_protocol="canonical water SMILES O passed to the exact CIGIN checkpoint",
        potential=f"CIGIN cigin.tar @ {CIGIN_SOURCE_REVISION}",
        solvation_backend="CIGIN direct scalar property head",
        cavity_model="not applicable to a scalar property predictor",
        sampling_protocol="not applicable; one deterministic CPU property prediction per SMILES pair",
        estimator="CIGIN direct scalar output (not TI/BAR/MBAR/FEP)",
        experimental_provenance="FreeSolv v0.52 entries pinned by the fixed pilot input",
        target_quantity=BenchmarkQuantity.PROPERTY_PREDICTION,
    )


def run(source_root: str | Path) -> dict[str, Any]:
    """Run the fixed development-only panel against one pinned CIGIN checkout."""

    compounds = _load_compounds(PILOT_INPUT)
    identity = _identity(compounds)
    training_evidence = ModelTrainingEvidence(
        training_datasets=("FreeSolv",),
        record_accounting_complete=False,
        evidence_source=(
            "The official CIGIN author page identifies FreeSolv training and reports "
            "an RMSE, but does not publish complete record-level accounting for cigin.tar."
        ),
    )
    panel = ValidationPanel(
        model_id="cigin-cigin.tar",
        stage=ValidationStage.DEVELOPMENT,
        identity=identity,
        training_evidence=training_evidence,
        selection_allowed=True,
    )
    leakage = audit_training_overlap(identity, training_evidence)
    adapter = CIGINPropertyAdapter(source_root)
    records: list[dict[str, Any]] = []
    for row in compounds:
        result = adapter.predict(str(row["smiles"]), WATER_SMILES)
        records.append(
            {
                "record_id": str(row["id"]),
                "name": str(row["name"]),
                "chemical_class": str(row["chemical_class"]),
                "input_smiles": str(row["smiles"]),
                "canonical_solute_smiles": result.solute_smiles,
                "canonical_solvent_smiles": result.solvent_smiles,
                "predicted_kcal_mol": result.predicted_solvation_free_energy_kcal_mol,
                "cigin": {
                    "source_revision": result.source_revision,
                    "checkpoint_sha256": result.checkpoint_sha256,
                    "unit": result.unit,
                    "execution_device": result.execution_device,
                    "cuda_disabled_for_upstream_device_consistency": (
                        result.cuda_disabled_for_upstream_device_consistency
                    ),
                },
            }
        )

    summary = summarize_runtime_smoke(identity, records)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "benchmark_status": (
            "runtime smoke only; chemical classes are not a functional-group "
            "accuracy taxonomy; "
            "known training dataset provenance with unresolved record overlap; "
            "not acceptance evidence"
        ),
        "acceptance_eligible": False,
        "input": {
            "path": str(PILOT_INPUT.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_file(PILOT_INPUT),
        },
        "validation_panel": {
            "model_id": panel.model_id,
            "stage": panel.stage.value,
            "selection_allowed": panel.selection_allowed,
            "identity": identity.canonical_payload(),
            "identity_fingerprint": identity.fingerprint,
            "training_evidence": {
                "training_datasets": list(training_evidence.training_datasets),
                "record_accounting_complete": training_evidence.record_accounting_complete,
                "evidence_source": training_evidence.evidence_source,
            },
            "leakage_audit": leakage.as_dict(),
        },
        "records": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", required=True, help="Pinned official CIGIN checkout"
    )
    parser.add_argument("--output", required=True, help="Output JSON artifact path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        parser.error(
            f"Refusing to overwrite existing output: {output}; pass --overwrite."
        )
    payload = run(args.source_root)
    _write_json_atomic(output, payload)
    print(
        "CIGIN FreeSolv-10 runtime smoke: "
        f"coverage={payload['summary']['coverage']:.3f}; no accuracy evaluation."
    )


if __name__ == "__main__":
    main()
