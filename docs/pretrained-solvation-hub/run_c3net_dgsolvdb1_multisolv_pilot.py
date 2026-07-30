#!/usr/bin/env python3
"""Run the fixed 11-solvent dGsolvDB1 C3Net development property pilot.

The input dataset is a public experimental compilation.  The script derives a
22-record panel by an output-blind hash rule, then uses the C3Net paper's
five-conformer arithmetic-mean convention.  It deliberately labels the result
as ``property_prediction`` and ``overlap_unknown``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger, rdBase  # noqa: E402
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
    C3NET_SUPPORTED_ELEMENTS,
    C3NetPropertyAdapter,
)

DATASET_URL = "https://zenodo.org/records/5792296"
DATASET_SHA256 = "d912a59697c4e8013ff77ac18e42ddff1167c6afe0e18efbb34ef8cb58f10d8b"
DATASET_MEMBER = "Solvation_data-1.0.0/all_data/dGsolvDB1_all.xlsx"
RECORDS_PER_SOLVENT = 2
CONFORMERS_PER_SOLUTE = 5
ETKDG_SEED_BASE = 0xC3E7

# Exact C3Net solvent names paired to output-blind canonical-SMILES lookup.
SOLVENT_SMILES = {
    "acetone": "CC(=O)C",
    "acetonitrile": "CC#N",
    "benzene": "c1ccccc1",
    "carbon_tetrachloride": "ClC(Cl)(Cl)Cl",
    "chloroform": "ClC(Cl)Cl",
    "dichloromethane": "ClCCl",
    "dimethylsulfoxide": "CS(C)=O",
    "ethanol": "CCO",
    "methanol": "CO",
    "tetrahydrofuran": "C1CCOC1",
    "toluene": "Cc1ccccc1",
}


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


def _canonical_smiles(smiles: object) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def _load_table(archive: Path) -> pd.DataFrame:
    if _sha256_file(archive) != DATASET_SHA256:
        raise ValueError(
            "dGsolvDB1 archive SHA256 does not match the pinned Zenodo file."
        )
    with zipfile.ZipFile(archive) as compressed:
        try:
            content = compressed.read(DATASET_MEMBER)
        except KeyError as exc:
            raise ValueError(
                f"Pinned dGsolvDB1 workbook member is missing: {DATASET_MEMBER}"
            ) from exc
    return pd.read_excel(io.BytesIO(content))


def _select_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    required_columns = {
        "smiles_solvent",
        "inchi_solvent",
        "smiles_solute",
        "inchi_solute",
        "dGsolv_avg [kcal/mol]",
        "dGsolv_std [kcal/mol]",
        "no. of data",
        "Source_all",
    }
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError("dGsolvDB1 workbook is missing columns: " + ", ".join(missing))

    canonical_solvent_tokens = {
        _canonical_smiles(smiles): token for token, smiles in SOLVENT_SMILES.items()
    }
    if None in canonical_solvent_tokens:
        raise RuntimeError(
            "Internal C3Net solvent mapping has an invalid SMILES string."
        )

    RDLogger.DisableLog("rdApp.*")
    grouped: dict[str, list[dict[str, Any]]] = {token: [] for token in SOLVENT_SMILES}
    for row in table.to_dict(orient="records"):
        solvent = canonical_solvent_tokens.get(_canonical_smiles(row["smiles_solvent"]))
        if solvent is None:
            continue
        molecule = Chem.MolFromSmiles(str(row["smiles_solute"]))
        if (
            molecule is None
            or Chem.GetFormalCharge(molecule) != 0
            or len(Chem.GetMolFrags(molecule)) != 1
            or not 2 <= molecule.GetNumHeavyAtoms() <= 20
            or any(atom.GetIsotope() for atom in molecule.GetAtoms())
            or not {atom.GetSymbol() for atom in molecule.GetAtoms()}.issubset(
                C3NET_SUPPORTED_ELEMENTS
            )
        ):
            continue
        try:
            experimental = float(row["dGsolv_avg [kcal/mol]"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(experimental):
            continue
        inchi_solvent = str(row["inchi_solvent"])
        inchi_solute = str(row["inchi_solute"])
        selection_hash = hashlib.sha256(
            f"{inchi_solvent}|{inchi_solute}".encode("utf-8")
        ).hexdigest()
        grouped[solvent].append(
            {
                "selection_hash": selection_hash,
                "record_id": hashlib.sha256(
                    f"c3net-dgsolvdb1|{solvent}|{inchi_solvent}|{inchi_solute}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "solvent": solvent,
                "solvent_smiles": str(row["smiles_solvent"]),
                "solvent_inchi": inchi_solvent,
                "solute_smiles": str(row["smiles_solute"]),
                "solute_inchi": inchi_solute,
                "experimental_kcal_mol": experimental,
                "experimental_std_kcal_mol": (
                    None
                    if pd.isna(row["dGsolv_std [kcal/mol]"])
                    else float(row["dGsolv_std [kcal/mol]"])
                ),
                "source_count": int(row["no. of data"]),
                "source_all": str(row["Source_all"]),
            }
        )

    selected: list[dict[str, Any]] = []
    for solvent in SOLVENT_SMILES:
        candidates = sorted(grouped[solvent], key=lambda row: row["selection_hash"])
        if len(candidates) < RECORDS_PER_SOLVENT:
            raise ValueError(
                f"dGsolvDB1 has fewer than {RECORDS_PER_SOLVENT} eligible records for {solvent}."
            )
        selected.extend(candidates[:RECORDS_PER_SOLVENT])
    return selected


def _identity(records: list[dict[str, Any]]) -> BenchmarkIdentity:
    return BenchmarkIdentity(
        dataset="dGsolvDB1",
        dataset_version="Solvation_data-1.0.0 / Zenodo 5792296",
        record_ids=tuple(record["record_id"] for record in records),
        temperature_kelvin=298.0,
        standard_state="1 mol/L gas and 1 mol/L liquid",
        protonation_policy="neutral structures as supplied by the pinned dGsolvDB1 workbook",
        tautomer_policy="as represented by the pinned dGsolvDB1 solute SMILES",
        conformer_policy=(
            "five RDKit ETKDGv3 conformers per solute; seed 0xc3e7 plus "
            "five times record index plus conformer index; each UFF-minimized; "
            "C3Net arithmetic mean"
        ),
        geometry_protocol="RDKit ETKDGv3 + UFF, generated at execution time",
        solvent_protocol="exact C3Net solvent identifier selected from pinned canonical SMILES",
        potential=f"C3Net checkpoint-1 @ {C3NET_SOURCE_REVISION}",
        solvation_backend="C3Net direct scalar property head",
        cavity_model="not applicable to a scalar property predictor",
        sampling_protocol="not applicable; five deterministic supplied conformers",
        estimator="C3Net paper arithmetic mean of five direct scalar predictions",
        experimental_provenance="dGsolvDB1 experimental 298 K, 1M/1M aggregates",
        target_quantity=BenchmarkQuantity.PROPERTY_PREDICTION,
    )


def _write_conformers(record: dict[str, Any], index: int, directory: Path) -> Path:
    base = Chem.MolFromSmiles(record["solute_smiles"])
    if base is None:
        raise ValueError(
            f"Cannot parse selected solute SMILES for {record['record_id']}."
        )
    path = directory / f"{record['record_id']}.sdf"
    writer = Chem.SDWriter(str(path))
    try:
        for conformer_index in range(CONFORMERS_PER_SOLUTE):
            molecule = Chem.AddHs(Chem.Mol(base))
            parameters = AllChem.ETKDGv3()
            parameters.randomSeed = (
                ETKDG_SEED_BASE + CONFORMERS_PER_SOLUTE * index + conformer_index
            )
            if AllChem.EmbedMolecule(molecule, parameters) != 0:
                raise ValueError(
                    f"ETKDGv3 embedding failed for {record['record_id']}, conformer {conformer_index}."
                )
            if AllChem.UFFOptimizeMolecule(molecule, maxIters=500) != 0:
                raise ValueError(
                    f"UFF minimization did not converge for {record['record_id']}, conformer {conformer_index}."
                )
            writer.write(molecule)
    finally:
        writer.close()
    return path


def run(source_root: str | Path, dataset_archive: str | Path) -> dict[str, Any]:
    """Execute the fixed 22-record, 11-solvent development-only panel."""

    records = _select_records(_load_table(Path(dataset_archive).expanduser()))
    identity = _identity(records)
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
    with tempfile.TemporaryDirectory(prefix="maple-c3net-dgsolvdb1-") as temporary:
        temporary_directory = Path(temporary)
        for index, record in enumerate(records):
            conformers = _write_conformers(record, index, temporary_directory)
            result = adapter.predict_conformers(conformers, record["solvent"])
            record["predicted_kcal_mol"] = (
                result.predicted_solvation_free_energy_kcal_mol
            )
            record["sdf_generation"] = {
                "method": "five RDKit ETKDGv3 conformers, each UFF minimized",
                "seed_base": ETKDG_SEED_BASE + CONFORMERS_PER_SOLUTE * index,
                "conformer_count": result.conformer_count,
                "max_uff_iterations": 500,
            }
            record["c3net"] = {
                "source_revision": result.source_revision,
                "checkpoint_sha256": result.checkpoint_sha256,
                "embedding_sha256": result.embedding_sha256,
                "solvent": result.solvent,
                "solvent_id": result.solvent_id,
                "unit": result.unit,
                "component_predictions_kcal_mol": list(
                    result.component_predictions_kcal_mol
                ),
                "numpy_int_compatibility_shim": result.numpy_int_compatibility_shim,
            }

    summary = summarize_runtime_smoke(identity, records)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "benchmark_status": (
            "runtime-only multi-solvent property-prediction smoke panel; "
            "functional-group accuracy coverage was not predeclared, so no "
            "accuracy metrics or acceptance claim are permitted"
        ),
        "acceptance_eligible": False,
        "input": {
            "archive_filename": Path(dataset_archive).expanduser().name,
            "source_url": DATASET_URL,
            "archive_sha256": DATASET_SHA256,
            "workbook_member": DATASET_MEMBER,
            "temperature_kelvin": 298.0,
            "standard_state": "1 mol/L gas and 1 mol/L liquid",
        },
        "selection": {
            "rule": "two lowest SHA256(inchi_solvent + '|' + inchi_solute) records per listed solvent after fixed neutral/domain filters; model outputs never enter selection",
            "solvents": SOLVENT_SMILES,
            "records_per_solvent": RECORDS_PER_SOLVENT,
            "conformers_per_solute": CONFORMERS_PER_SOLUTE,
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
    parser.add_argument(
        "--dataset-archive",
        required=True,
        help="Pinned Solvation_data-1.0.0.zip from Zenodo 5792296",
    )
    parser.add_argument("--output", required=True, help="Output JSON artifact path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        parser.error(
            f"Refusing to overwrite existing output: {output}; pass --overwrite."
        )
    payload = run(args.source_root, args.dataset_archive)
    _write_json_atomic(output, payload)
    print(
        "C3Net dGsolvDB1 11-solvent runtime smoke: "
        f"coverage={payload['summary']['coverage']:.3f}; "
        "accuracy metrics disabled because functional-group coverage was not "
        "predeclared."
    )


if __name__ == "__main__":
    main()
