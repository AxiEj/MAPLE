#!/usr/bin/env python3
"""Run the output-blind 19-functional-group C3Net/FlexiSol development panel.

This is a development accuracy panel with unknown checkpoint overlap.  It is
not MAPLE's final independent extrapolation set, and speed cannot admit the
model unless this accuracy gate passes first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

AUDITED_ON = "2026-07-30"
PAPER_DOI = "10.1039/D5SC06406F"
FLEXISOL_SOURCE_URL = "https://github.com/grimme-lab/flexisol"
FLEXISOL_SOURCE_REVISION = "7b44798f26c888ef541faa6143a813136921483f"
FLEXISOL_SOURCE_TREE = "8cf4879bc1435ca1061acfffd0daf10784445984"
FLEXISOL_REFERENCE_PATH = "data/references/dgsolv-references.csv"
FLEXISOL_REFERENCE_SHA256 = (
    "73a6f25ea9a9fae31bcb7e7c0fb04e452b0df2d24308e25419b98f01daa6ed5f"
)
FLEXISOL_REFERENCE_ROWS = 530
TAXONOMY_SHA256 = "46e073d0e253d7534b7f516f1cc2096ba0abfea84ddc170c95fc86634c917b36"
CONFORMER_CANDIDATE_COUNT = 50
CONFORMERS_PER_SOLUTE = 5
CONFORMER_RMSD_THRESHOLD_ANGSTROM = 0.5
MAXIMUM_ABSOLUTE_ERROR_LIMIT_KCAL_MOL = 1.5
SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
EXPECTED_PRIMARY_GROUPS = (
    "alcohol",
    "aldehyde",
    "amide",
    "amine",
    "carboxylic_acid",
    "epoxide",
    "ester",
    "ether",
    "ketone",
    "lactone",
    "nitrile",
    "nitro",
    "organohalogen",
    "phenol",
    "phosphate_ester",
    "sulfonamide",
    "terminal_alkyne",
    "thioether",
    "urea",
)
EXPECTED_SOLVENTS = frozenset(
    {"water", "octanol", "hexadecane", "hexane", "methanol", "ethanol", "dmf"}
)
EXPECTED_HEADER = (
    "FlexiSol Name",
    "IUPAC Name",
    "Solvent",
    "SMILES",
    r"Value (\kcalpmole)",
    "Ref.",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_output(source_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FlexiSol Git identity check failed: {detail}")
    return completed.stdout


def _read_verified_rows(source_root: Path) -> list[dict[str, str]]:
    revision = _git_output(source_root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git_output(source_root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if revision != FLEXISOL_SOURCE_REVISION or tree != FLEXISOL_SOURCE_TREE:
        raise ValueError(
            "FlexiSol source revision or tree differs from the frozen panel."
        )
    payload = _git_output(
        source_root,
        "show",
        f"{FLEXISOL_SOURCE_REVISION}:{FLEXISOL_REFERENCE_PATH}",
    )
    if _sha256_bytes(payload) != FLEXISOL_REFERENCE_SHA256:
        raise ValueError("FlexiSol experimental-reference CSV SHA256 mismatch.")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
        raise ValueError("FlexiSol experimental-reference CSV header changed.")
    rows: list[dict[str, str]] = []
    for line_number, raw in enumerate(reader, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"FlexiSol CSV line {line_number} is malformed.")
        row = {key: value.strip() for key, value in raw.items()}
        row["_line_number"] = str(line_number)
        rows.append(row)
    if len(rows) != FLEXISOL_REFERENCE_ROWS:
        raise ValueError(f"FlexiSol reference row count changed: observed {len(rows)}.")
    observed_solvents = {row["Solvent"].casefold() for row in rows}
    if observed_solvents != EXPECTED_SOLVENTS:
        raise ValueError("FlexiSol pure-solvent registry changed.")
    return rows


def _candidate_identity(row: Mapping[str, str], canonical_smiles: str) -> str:
    return "|".join(
        (
            row["FlexiSol Name"],
            row["Solvent"].casefold(),
            canonical_smiles,
            row["Ref."],
        )
    )


def _select_output_blind_records(
    rows: Sequence[Mapping[str, str]],
    *,
    taxonomy: Any,
    derive_functional_group_assignment: Any,
) -> list[dict[str, Any]]:
    """Select one min-hash eligible identity per primary group without labels."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    RDLogger.DisableLog("rdApp.*")
    for row in rows:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if (
            molecule is None
            or Chem.GetFormalCharge(molecule) != 0
            or len(Chem.GetMolFrags(molecule)) != 1
            or not 2 <= molecule.GetNumHeavyAtoms() <= 20
            or any(atom.GetIsotope() for atom in molecule.GetAtoms())
            or not {atom.GetSymbol() for atom in molecule.GetAtoms()}.issubset(
                SUPPORTED_ELEMENTS
            )
        ):
            continue
        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        identity_text = _candidate_identity(row, canonical_smiles)
        selection_hash = _sha256_bytes(identity_text.encode("utf-8"))
        provisional_id = f"flexisol-{selection_hash}"
        try:
            assignment = derive_functional_group_assignment(
                record_id=provisional_id,
                structure_identifier=f"SMILES:{canonical_smiles}",
                taxonomy=taxonomy,
                assignment_evidence=(
                    "frozen taxonomy SMARTS applied before any C3Net output"
                ),
            )
        except ValueError:
            continue
        candidates.setdefault(assignment.primary_group, []).append(
            {
                "selection_hash": selection_hash,
                "canonical_smiles": canonical_smiles,
                "primary_group": assignment.primary_group,
                "matched_groups": assignment.matched_groups,
                "row": dict(row),
            }
        )

    observed_groups = tuple(sorted(candidates))
    if observed_groups != EXPECTED_PRIMARY_GROUPS:
        raise ValueError(
            "Eligible FlexiSol primary-group inventory changed: "
            f"observed {observed_groups!r}."
        )
    selected: list[dict[str, Any]] = []
    for group in EXPECTED_PRIMARY_GROUPS:
        record = min(candidates[group], key=lambda item: item["selection_hash"])
        record["record_id"] = f"flexisol-{group}-{record['selection_hash'][:16]}"
        selected.append(record)
    return selected


def _write_conformer_input(
    record: Mapping[str, Any],
    *,
    destination: Path,
) -> tuple[Path, dict[str, Any]]:
    molecule = Chem.MolFromSmiles(record["canonical_smiles"])
    if molecule is None:
        raise ValueError(f"Selected record {record['record_id']} cannot be parsed.")
    molecule = Chem.AddHs(molecule)
    random_seed = int(record["selection_hash"][:8], 16) & 0x7FFFFFFF
    conformer_ids = tuple(
        int(conformer_id)
        for conformer_id in AllChem.EmbedMultipleConfs(
            molecule,
            numConfs=CONFORMER_CANDIDATE_COUNT,
            randomSeed=random_seed,
        )
    )
    if not conformer_ids:
        raise ValueError(
            f"RDKit failed to generate conformer candidates for "
            f"{record['record_id']}."
        )
    optimization_results = tuple(AllChem.UFFOptimizeMoleculeConfs(molecule))
    if len(optimization_results) != len(conformer_ids):
        raise ValueError(
            f"UFF status count changed for {record['record_id']}: "
            f"{len(optimization_results)} statuses for {len(conformer_ids)} "
            "candidates."
        )

    heavy_atom_molecule = Chem.RemoveHs(molecule)
    AllChem.AlignMolConformers(heavy_atom_molecule)
    selected_conformer_ids: list[int] = []
    for conformer_id in conformer_ids:
        if all(
            AllChem.GetConformerRMS(
                heavy_atom_molecule,
                conformer_id,
                selected_id,
                prealigned=True,
            )
            >= CONFORMER_RMSD_THRESHOLD_ANGSTROM
            for selected_id in selected_conformer_ids
        ):
            selected_conformer_ids.append(conformer_id)
        if len(selected_conformer_ids) >= CONFORMERS_PER_SOLUTE:
            break
    if not selected_conformer_ids:
        raise ValueError(
            f"No conformer survived deduplication for {record['record_id']}."
        )

    path = destination / f"{record['record_id']}.sdf"
    writer = Chem.SDWriter(str(path))
    try:
        for output_index, conformer_id in enumerate(selected_conformer_ids):
            molecule.SetProp("Conf_ID", str(output_index))
            writer.write(molecule, confId=conformer_id)
    finally:
        writer.close()
    return path, {
        "candidate_count": len(conformer_ids),
        "conformer_count": len(selected_conformer_ids),
        "selected_conformer_ids": selected_conformer_ids,
        "heavy_atom_rmsd_threshold_angstrom": (CONFORMER_RMSD_THRESHOLD_ANGSTROM),
        "random_seed": random_seed,
        "uff_optimization_statuses": [
            {
                "conformer_id": conformer_id,
                "status": int(status),
                "energy": float(energy),
            }
            for conformer_id, (status, energy) in zip(
                conformer_ids,
                optimization_results,
                strict=True,
            )
        ],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _run_into_output_directory(
    *,
    flexisol_source_root: str | Path,
    c3net_source_root: str | Path,
    c3net_source_bundle: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Execute the frozen development panel in a fresh configured process."""

    os.environ["MAPLE_C3NET_FORMAL_SOURCE_ROOT"] = str(
        Path(c3net_source_root).expanduser().resolve(strict=True)
    )
    os.environ["MAPLE_C3NET_FORMAL_SOURCE_BUNDLE"] = str(
        Path(c3net_source_bundle).expanduser().resolve(strict=True)
    )

    import maple.function.benchmarking.pretrained_hub as pretrained_hub
    from maple.function.benchmarking import (
        BenchmarkExperimentalReference,
        BenchmarkIdentity,
        BenchmarkQuantity,
        BenchmarkRecordInput,
        LeakageStatus,
        ModelTrainingEvidence,
        MolecularInputReceipt,
        SolventComposition,
        ValidationPanel,
        ValidationStage,
        audit_training_overlap,
        derive_functional_group_assignment,
        load_functional_group_taxonomy,
        run_accuracy_panel,
    )
    from maple.function.benchmarking.c3net_accuracy import (
        C3NET_FORMAL_ACCURACY_ADAPTER_ID,
    )

    try:
        registration = pretrained_hub._FORMAL_ACCURACY_ADAPTERS[
            C3NET_FORMAL_ACCURACY_ADAPTER_ID
        ]
    except KeyError as exc:
        raise RuntimeError(
            "C3Net formal registration must be configured before importing "
            "maple.function.benchmarking; launch this script in a fresh process."
        ) from exc

    repository_root = Path(__file__).resolve().parents[2]
    taxonomy_path = (
        repository_root
        / "docs/pretrained-solvation-hub/functional-group-taxonomy-v1.json"
    )
    taxonomy = load_functional_group_taxonomy(
        taxonomy_path,
        expected_sha256=TAXONOMY_SHA256,
    )
    rows = _read_verified_rows(
        Path(flexisol_source_root).expanduser().resolve(strict=True)
    )
    selected = _select_output_blind_records(
        rows,
        taxonomy=taxonomy,
        derive_functional_group_assignment=derive_functional_group_assignment,
    )

    output_root = Path(output_directory).expanduser().resolve()
    input_root = output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    assignments = {}
    for record in selected:
        input_path, conformer_generation = _write_conformer_input(
            record,
            destination=input_root,
        )
        record["conformer_generation"] = conformer_generation
        receipt = MolecularInputReceipt.from_file(
            input_path,
            molecular_input_format="sdf_conformers",
            locator=(
                f"flexisol-git:{FLEXISOL_SOURCE_REVISION}:"
                f"{FLEXISOL_REFERENCE_PATH}#L{record['row']['_line_number']};"
                f"generated:{input_path.name}"
            ),
        )
        expected_structure = "SMILES:" + Chem.MolToSmiles(
            Chem.MolFromSmiles(record["canonical_smiles"]),
            canonical=True,
            isomericSmiles=False,
        )
        if receipt.solute_structure_identifier != expected_structure:
            raise ValueError(
                f"Generated conformers changed graph identity for "
                f"{record['record_id']}."
            )
        receipts[record["record_id"]] = receipt
        assignments[record["record_id"]] = derive_functional_group_assignment(
            record_id=record["record_id"],
            structure_identifier=receipt.solute_structure_identifier,
            taxonomy=taxonomy,
            assignment_evidence=(
                "frozen taxonomy SMARTS applied to the exact generated SDF graph "
                "before any C3Net output"
            ),
        )

    reference_payload = {}
    for record in selected:
        row = record["row"]
        try:
            value = float(row[r"Value (\kcalpmole)"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Selected FlexiSol record {record['record_id']} has no finite label."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Selected FlexiSol record {record['record_id']} has no finite label."
            )
        reference_payload[record["record_id"]] = {
            "value": value,
            "unit": "kcal/mol",
            "provenance": (
                f"FlexiSol {FLEXISOL_SOURCE_REVISION} "
                f"{FLEXISOL_REFERENCE_PATH} line {row['_line_number']}; "
                f"original experimental reference {row['Ref.']}; "
                f"FlexiSol paper DOI {PAPER_DOI}"
            ),
        }
    reference_path = output_root / "experimental-references.json"
    _write_json_atomic(reference_path, reference_payload)

    record_ids = tuple(record["record_id"] for record in selected)
    identity = BenchmarkIdentity(
        dataset="FlexiSol-dGsolv",
        dataset_version=(
            f"paper {PAPER_DOI}; source revision {FLEXISOL_SOURCE_REVISION}"
        ),
        record_ids=record_ids,
        temperature_kelvin=298.15,
        standard_state=(
            "experimental convention curated by FlexiSol; values are not "
            "restandardized by MAPLE"
        ),
        protonation_policy="neutral, single-fragment dataset SMILES only",
        tautomer_policy="exact tautomer encoded by the pinned FlexiSol SMILES",
        conformer_policy=(
            "one to five deterministic RDKit conformers per solute, selected "
            "from 50 UFF-optimized candidates by greedy 0.5 Angstrom heavy-atom "
            "RMSD deduplication; arithmetic mean over the actual retained "
            "one-to-five unmodified C3Net predictions"
        ),
        geometry_protocol=(
            "deterministic output-blind adaptation of the upstream-structured "
            "RDKit protocol, not a historical exact replay: EmbedMultipleConfs "
            "default parameters with seed derived only from frozen record "
            "identity; UFFOptimizeMoleculeConfs defaults; RemoveHs; "
            "AlignMolConformers; greedy heavy-atom RMSD deduplication at 0.5 "
            "Angstrom; retain at most five"
        ),
        solvent_protocol=(
            "one pure FlexiSol solvent mapped by a frozen alias to the exact "
            "C3Net solvent registry"
        ),
        potential=f"C3Net checkpoint-1 @ {registration.compute_fingerprint()}",
        solvation_backend="C3Net direct scalar solvation-property head",
        cavity_model="not applicable to a scalar property predictor",
        sampling_protocol=(
            "no thermodynamic sampling; the actual one-to-five retained "
            "conformers are evaluated independently"
        ),
        estimator=(
            "arithmetic mean over the actual one-to-five retained C3Net scalar "
            "predictions"
        ),
        experimental_provenance=(
            f"FlexiSol standard-state experimental references, paper {PAPER_DOI}"
        ),
        target_quantity=BenchmarkQuantity.PROPERTY_PREDICTION,
        record_solvent_compositions=tuple(
            SolventComposition(
                record_id=record["record_id"],
                components=(record["row"]["Solvent"].casefold(),),
                mole_fractions=(1.0,),
            )
            for record in selected
        ),
        record_inputs=tuple(
            BenchmarkRecordInput.from_receipt(
                record_id=record_id,
                receipt=receipts[record_id],
            )
            for record_id in record_ids
        ),
        record_experimental_references=tuple(
            BenchmarkExperimentalReference.from_json_file(
                reference_path,
                record_id=record_id,
            )
            for record_id in record_ids
        ),
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
        functional_group_taxonomy=taxonomy,
        record_functional_groups=tuple(
            assignments[record_id] for record_id in record_ids
        ),
    )
    training_evidence = ModelTrainingEvidence(
        training_datasets=(),
        known_training_record_ids=(),
        explicitly_excluded_datasets=(),
        explicitly_excluded_record_ids=(),
        record_accounting_complete=False,
        evidence_source=(
            "The official C3Net source does not publish complete canonical "
            "record identities for checkpoint-1 training."
        ),
    )
    leakage = audit_training_overlap(identity, training_evidence)
    if leakage.status is not LeakageStatus.OVERLAP_UNKNOWN:
        raise RuntimeError(
            "FlexiSol/C3Net development panel must remain overlap_unknown."
        )
    ValidationPanel(
        model_id="c3net-checkpoint-1",
        stage=ValidationStage.DEVELOPMENT,
        identity=identity,
        training_evidence=training_evidence,
        selection_allowed=True,
    )

    accuracy = run_accuracy_panel(identity, bootstrap_samples=10_000, seed=0)
    maximum_error = accuracy["metrics"]["maximum_absolute_error_kcal_mol"]
    accuracy_gate_passed = (
        accuracy["functional_group_coverage"]["passes"]
        and maximum_error < MAXIMUM_ABSOLUTE_ERROR_LIMIT_KCAL_MOL
    )
    result = {
        "schema_version": 1,
        "audited_on": AUDITED_ON,
        "scientific_status": (
            "development_accuracy_only_overlap_unknown_not_final_holdout"
        ),
        "model_id": "c3net-checkpoint-1",
        "accuracy_adapter_id": registration.adapter_id,
        "accuracy_adapter_fingerprint": registration.compute_fingerprint(),
        "implementation_artifacts": [
            artifact.canonical_payload()
            for artifact in registration.implementation_artifacts
        ],
        "panel_generation_artifacts": [
            {
                "role": "panel_runner_code",
                "artifact_sha256": _sha256_bytes(Path(__file__).read_bytes()),
                "execution_scope": "parent_only",
                "registered_with_label_free_adapter": False,
                "mounted_in_prediction_sandbox": False,
            },
            {
                "role": "functional_group_taxonomy",
                "artifact_sha256": _sha256_bytes(taxonomy_path.read_bytes()),
                "execution_scope": "parent_only",
                "registered_with_label_free_adapter": False,
                "mounted_in_prediction_sandbox": False,
            },
        ],
        "dataset": {
            "name": "FlexiSol-dGsolv",
            "paper_doi": PAPER_DOI,
            "source_url": FLEXISOL_SOURCE_URL,
            "source_revision": FLEXISOL_SOURCE_REVISION,
            "source_tree": FLEXISOL_SOURCE_TREE,
            "reference_path": FLEXISOL_REFERENCE_PATH,
            "reference_sha256": FLEXISOL_REFERENCE_SHA256,
        },
        "selection": {
            "output_blind": True,
            "policy": (
                "neutral single-fragment 2-20-heavy-atom C3Net-domain rows; "
                "frozen taxonomy; one minimum SHA256 identity per primary group; "
                "experimental value and model output excluded from selection"
            ),
            "selected_record_count": len(selected),
            "selected_primary_group_count": len(EXPECTED_PRIMARY_GROUPS),
            "records": [
                {
                    "record_id": record["record_id"],
                    "selection_hash": record["selection_hash"],
                    "primary_functional_group": record["primary_group"],
                    "matched_functional_groups": list(record["matched_groups"]),
                    "flexisol_name": record["row"]["FlexiSol Name"],
                    "solvent": record["row"]["Solvent"].casefold(),
                    "canonical_smiles": record["canonical_smiles"],
                    "source_line": int(record["row"]["_line_number"]),
                    "source_reference": record["row"]["Ref."],
                    "molecular_input_sha256": receipts[
                        record["record_id"]
                    ].molecular_input_sha256,
                    "conformer_generation": record["conformer_generation"],
                }
                for record in selected
            ],
        },
        "leakage_audit": leakage.as_dict(),
        "accuracy_gate": {
            "evaluated_before_speed": True,
            "maximum_absolute_error_limit_kcal_mol": (
                MAXIMUM_ABSOLUTE_ERROR_LIMIT_KCAL_MOL
            ),
            "strict_inequality_required": True,
            "passed": accuracy_gate_passed,
        },
        "speed_gate": {
            "required_only_after_accuracy_passes": True,
            "matched_end_to_end_qm_speedup_required": ">1",
            "status": (
                "pending_matched_qm_benchmark"
                if accuracy_gate_passed
                else "not_eligible_due_to_accuracy_failure"
            ),
        },
        "accuracy": accuracy,
    }
    _write_json_atomic(output_root / "result.json", result)
    return result


def run(
    *,
    flexisol_source_root: str | Path,
    c3net_source_root: str | Path,
    c3net_source_bundle: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Execute the panel into a fresh directory published by atomic rename."""

    output_root = Path(output_directory).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output directory must not already exist: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )
    try:
        result = _run_into_output_directory(
            flexisol_source_root=flexisol_source_root,
            c3net_source_root=c3net_source_root,
            c3net_source_bundle=c3net_source_bundle,
            output_directory=temporary_root,
        )
        os.replace(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flexisol-source-root", required=True)
    parser.add_argument("--c3net-source-root", required=True)
    parser.add_argument("--c3net-source-bundle", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = run(
        flexisol_source_root=arguments.flexisol_source_root,
        c3net_source_root=arguments.c3net_source_root,
        c3net_source_bundle=arguments.c3net_source_bundle,
        output_directory=arguments.output_directory,
    )
    print(
        json.dumps(
            {
                "result_path": str(
                    Path(arguments.output_directory).expanduser().resolve()
                    / "result.json"
                ),
                "record_count": result["accuracy"]["record_count"],
                "functional_group_count": result["accuracy"][
                    "functional_group_coverage"
                ]["observed_count"],
                "mae_kcal_mol": result["accuracy"]["metrics"]["mae_kcal_mol"],
                "maximum_absolute_error_kcal_mol": result["accuracy"]["metrics"][
                    "maximum_absolute_error_kcal_mol"
                ],
                "accuracy_gate_passed": result["accuracy_gate"]["passed"],
                "speed_gate_status": result["speed_gate"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
