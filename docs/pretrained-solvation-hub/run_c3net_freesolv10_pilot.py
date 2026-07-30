#!/usr/bin/env python3
"""Run the fixed FreeSolv-10 C3Net property-prediction development pilot.

This script intentionally records a ``property_prediction`` benchmark with
``overlap_unknown``.  It does not report an absolute-solvation calculation or
an independent validation result.
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

from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

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
    C3NET_SOURCE_REVISION,
    C3NetPropertyAdapter,
)

PILOT_INPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
    / "freesolv10-2026-07-22.json"
)
ETKDG_SEED_BASE = 0xF00D


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
            "The fixed C3Net FreeSolv pilot must contain exactly 10 records."
        )
    identifiers = [str(row.get("id", "")).strip() for row in compounds]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "The fixed C3Net FreeSolv pilot has invalid compound identifiers."
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
        conformer_policy=(
            "one RDKit ETKDGv3 conformer per SMILES, seed 0xf00d plus row index, "
            "followed by UFF minimization"
        ),
        geometry_protocol="RDKit ETKDGv3 + UFF, generated at execution time",
        solvent_protocol="C3Net exact upstream identifier water",
        potential=f"C3Net checkpoint-1 @ {C3NET_SOURCE_REVISION}",
        solvation_backend="C3Net direct scalar property head",
        cavity_model="not applicable to a scalar property predictor",
        sampling_protocol="not applicable; one deterministic property prediction per supplied geometry",
        estimator="C3Net direct scalar output (not TI/BAR/MBAR/FEP)",
        experimental_provenance="FreeSolv v0.52 entries pinned by the fixed pilot input",
        target_quantity=BenchmarkQuantity.PROPERTY_PREDICTION,
    )


def _write_sdf(row: dict[str, Any], index: int, directory: Path) -> Path:
    identifier = str(row["id"])
    molecule = Chem.MolFromSmiles(str(row["smiles"]))
    if molecule is None:
        raise ValueError(f"Cannot parse fixed pilot SMILES for {identifier}.")
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = ETKDG_SEED_BASE + index
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise ValueError(f"ETKDGv3 embedding failed for {identifier}.")
    if AllChem.UFFOptimizeMolecule(molecule, maxIters=500) != 0:
        raise ValueError(f"UFF minimization did not converge for {identifier}.")
    path = directory / f"{identifier}.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    return path


def run(source_root: str | Path) -> dict[str, Any]:
    """Run the fixed development-only panel against one pinned C3Net checkout."""

    compounds = _load_compounds(PILOT_INPUT)
    identity = _identity(compounds)
    training_evidence = ModelTrainingEvidence(
        record_accounting_complete=False,
        evidence_source=(
            "C3Net arXiv:2309.15334 reports a random pair split but does not "
            "publish complete record-level training accounting for checkpoint-1."
        ),
    )
    panel = ValidationPanel(
        model_id="c3net-checkpoint-1",
        stage=ValidationStage.DEVELOPMENT,
        identity=identity,
        training_evidence=training_evidence,
        selection_allowed=True,
    )
    leakage = audit_training_overlap(identity, training_evidence)
    adapter = C3NetPropertyAdapter(source_root)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="maple-c3net-freesolv10-") as temporary:
        temporary_directory = Path(temporary)
        for index, row in enumerate(compounds):
            sdf_path = _write_sdf(row, index, temporary_directory)
            result = adapter.predict(sdf_path, "water")
            records.append(
                {
                    "record_id": str(row["id"]),
                    "name": str(row["name"]),
                    "chemical_class": str(row["chemical_class"]),
                    "smiles": str(row["smiles"]),
                    "predicted_kcal_mol": result.predicted_solvation_free_energy_kcal_mol,
                    "sdf_generation": {
                        "method": "RDKit ETKDGv3 followed by UFF minimization",
                        "random_seed": ETKDG_SEED_BASE + index,
                        "max_uff_iterations": 500,
                    },
                    "c3net": {
                        "source_revision": result.source_revision,
                        "checkpoint_sha256": result.checkpoint_sha256,
                        "embedding_sha256": result.embedding_sha256,
                        "solvent": result.solvent,
                        "solvent_id": result.solvent_id,
                        "unit": result.unit,
                        "numpy_int_compatibility_shim": result.numpy_int_compatibility_shim,
                    },
                }
            )

    summary = summarize_runtime_smoke(identity, records)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "benchmark_status": (
            "runtime smoke only; chemical classes are not a functional-group "
            "accuracy taxonomy; overlap-unknown and not acceptance evidence"
        ),
        "acceptance_eligible": False,
        "input": {
            "path": str(PILOT_INPUT.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256_file(PILOT_INPUT),
        },
        "software": {"rdkit_version": rdBase.rdkitVersion},
        "validation_panel": {
            "model_id": panel.model_id,
            "stage": panel.stage.value,
            "selection_allowed": panel.selection_allowed,
            "identity": identity.canonical_payload(),
            "identity_fingerprint": identity.fingerprint,
            "training_evidence": {
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
        "--source-root", required=True, help="Pinned official C3Net checkout"
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
        "C3Net FreeSolv-10 runtime smoke: "
        f"coverage={payload['summary']['coverage']:.3f}; no accuracy evaluation."
    )


if __name__ == "__main__":
    main()
