#!/usr/bin/env python3
"""Run the output-blind AtomicESE/FlexiSol development accuracy panel.

The panel is overlap-unknown development evidence, not a final independent
holdout.  It records no runtime measurements.  A matched-QM timing study is
eligible only if the strict, unrounded maximum absolute error is below
1.5 kcal/mol.
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
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

AUDITED_ON = "2026-07-31"
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
MAXIMUM_ABSOLUTE_ERROR_LIMIT_KCAL_MOL = 1.5
SUPPORTED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
ELIGIBLE_SOLVENT_ALIASES = {
    "methanol": "methanol",
    "octanol": "octanol",
    "hexadecane": "hexadecane",
    "hexane": "hexane",
    "dmf": "dmf",
    "ethanol": "ethanol",
}
SELECTION_IDENTITY_FIELDS = (
    "FlexiSol Name",
    "Solvent",
    "canonical_smiles",
    "Ref.",
)
EXPECTED_ELIGIBLE_ROW_COUNT = 76
EXPECTED_ELIGIBLE_SOLVENT_COUNTS = {
    "ethanol": 1,
    "hexadecane": 13,
    "hexane": 1,
    "octanol": 61,
}
EXPECTED_PRIMARY_GROUPS = (
    "alcohol",
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
XYZ_DECIMAL_PLACES = 10
MMFF_MAX_ITERATIONS = 1000
MMFF_NONBONDED_THRESHOLD = 100.0
EMBED_MAX_ATTEMPTS = 1000


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
    if {row["Solvent"].casefold() for row in rows} != EXPECTED_SOLVENTS:
        raise ValueError("FlexiSol pure-solvent registry changed.")
    return rows


def _candidate_identity(row: Mapping[str, str], canonical_smiles: str) -> str:
    """Return the frozen label-free identity used for SHA256 selection."""

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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the minimum identity SHA256 in every eligible primary group."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    eligible_solvents: Counter[str] = Counter()
    eligible_count = 0
    RDLogger.DisableLog("rdApp.*")
    for row in rows:
        solvent = row["Solvent"].casefold()
        if solvent not in ELIGIBLE_SOLVENT_ALIASES:
            continue
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
                    "frozen taxonomy SMARTS applied before any AtomicESE output"
                ),
            )
        except ValueError:
            continue
        eligible_count += 1
        eligible_solvents[solvent] += 1
        candidates.setdefault(assignment.primary_group, []).append(
            {
                "selection_hash": selection_hash,
                "canonical_smiles": canonical_smiles,
                "primary_group": assignment.primary_group,
                "matched_groups": assignment.matched_groups,
                "row": dict(row),
            }
        )

    observed_counts = dict(sorted(eligible_solvents.items()))
    if eligible_count != EXPECTED_ELIGIBLE_ROW_COUNT:
        raise ValueError(
            "Eligible FlexiSol row inventory changed: "
            f"expected {EXPECTED_ELIGIBLE_ROW_COUNT}, observed {eligible_count}."
        )
    if observed_counts != EXPECTED_ELIGIBLE_SOLVENT_COUNTS:
        raise ValueError(
            "Eligible FlexiSol solvent inventory changed: "
            f"observed {observed_counts!r}."
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
    return selected, {
        "eligible_row_count": eligible_count,
        "eligible_solvent_counts": observed_counts,
        "eligible_primary_group_count": len(observed_groups),
        "eligible_primary_group_candidate_counts": {
            group: len(candidates[group]) for group in observed_groups
        },
    }


def _write_mmff94_xyz(
    record: Mapping[str, Any],
    *,
    destination: Path,
) -> dict[str, Any]:
    """Write one deterministic, headerless MMFF94 development geometry."""

    molecule = Chem.MolFromSmiles(record["canonical_smiles"])
    if molecule is None:
        raise ValueError(f"Selected record {record['record_id']} cannot be parsed.")
    molecule = Chem.AddHs(molecule, addCoords=False)
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError(
            f"Selected record {record['record_id']} lacks complete MMFF94 parameters."
        )

    random_seed = int(record["selection_hash"][:8], 16) & 0x7FFFFFFF
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = random_seed
    parameters.clearConfs = True
    parameters.enforceChirality = True
    parameters.useRandomCoords = False
    parameters.maxIterations = EMBED_MAX_ATTEMPTS
    parameters.numThreads = 1
    parameters.pruneRmsThresh = -1.0
    parameters.useExpTorsionAnglePrefs = True
    parameters.useBasicKnowledge = True
    parameters.useSmallRingTorsions = False
    parameters.useMacrocycleTorsions = True
    parameters.useMacrocycle14config = True
    parameters.ETversion = 2
    conformer_id = int(AllChem.EmbedMolecule(molecule, parameters))
    if conformer_id < 0:
        raise ValueError(
            f"RDKit embedding failed for selected record {record['record_id']}."
        )
    status = int(
        AllChem.MMFFOptimizeMolecule(
            molecule,
            mmffVariant="MMFF94",
            maxIters=MMFF_MAX_ITERATIONS,
            nonBondedThresh=MMFF_NONBONDED_THRESHOLD,
            confId=conformer_id,
            ignoreInterfragInteractions=True,
        )
    )
    if status != 0:
        raise ValueError(
            f"MMFF94 optimization did not converge for {record['record_id']}: "
            f"status {status}."
        )
    properties = AllChem.MMFFGetMoleculeProperties(
        molecule,
        mmffVariant="MMFF94",
        mmffVerbosity=0,
    )
    if properties is None:
        raise ValueError(f"MMFF94 properties disappeared for {record['record_id']}.")
    force_field = AllChem.MMFFGetMoleculeForceField(
        molecule,
        properties,
        nonBondedThresh=MMFF_NONBONDED_THRESHOLD,
        confId=conformer_id,
        ignoreInterfragInteractions=True,
    )
    if force_field is None:
        raise ValueError(
            f"MMFF94 force field could not be built for {record['record_id']}."
        )
    energy = float(force_field.CalcEnergy())
    conformer = molecule.GetConformer(conformer_id)
    lines: list[str] = []
    coordinates: list[list[float]] = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        values = (float(position.x), float(position.y), float(position.z))
        if not math.isfinite(energy) or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError(
                f"MMFF94 produced non-finite output for {record['record_id']}."
            )
        coordinates.append(list(values))
        lines.append(
            f"{atom.GetSymbol()} "
            f"{values[0]:.{XYZ_DECIMAL_PLACES}f} "
            f"{values[1]:.{XYZ_DECIMAL_PLACES}f} "
            f"{values[2]:.{XYZ_DECIMAL_PLACES}f}"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return {
        "purpose": "MAPLE development geometry",
        "author_manual_correction_reproduction": False,
        "conformer_count": 1,
        "force_field": "MMFF94",
        "force_field_variant": "MMFF94",
        "optimization_status": status,
        "optimization_max_iterations": MMFF_MAX_ITERATIONS,
        "optimization_nonbonded_threshold_angstrom": MMFF_NONBONDED_THRESHOLD,
        "optimization_ignore_interfragment_interactions": True,
        "final_energy_kcal_mol": energy,
        "embedding_method": "RDKit ETKDGv3",
        "embedding_random_seed": random_seed,
        "embedding_num_threads": 1,
        "embedding_clear_conformers": True,
        "embedding_enforce_chirality": True,
        "embedding_use_random_coordinates": False,
        "embedding_max_iterations": EMBED_MAX_ATTEMPTS,
        "embedding_prune_rms_threshold_angstrom": -1.0,
        "embedding_use_experimental_torsion_preferences": True,
        "embedding_use_basic_knowledge": True,
        "embedding_use_small_ring_torsions": False,
        "embedding_use_macrocycle_torsions": True,
        "embedding_use_macrocycle_14_configuration": True,
        "embedding_et_version": 2,
        "atom_count": molecule.GetNumAtoms(),
        "headerless_xyz": True,
        "coordinate_decimal_places": XYZ_DECIMAL_PLACES,
        "coordinates_angstrom": coordinates,
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
    atomicese_source_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    os.environ["MAPLE_ATOMICESE_FORMAL_SOURCE_ROOT"] = str(
        Path(atomicese_source_root).expanduser().resolve(strict=True)
    )
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

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
    from maple.function.benchmarking.atomicese_accuracy import (
        ATOMICESE_FORMAL_ACCURACY_ADAPTER_ID,
        ATOMICESE_LINUX_BINARY_SHA256,
        ATOMICESE_SOURCE_REVISION,
        ATOMICESE_SOURCE_TREE,
    )

    try:
        registration = pretrained_hub._FORMAL_ACCURACY_ADAPTERS[
            ATOMICESE_FORMAL_ACCURACY_ADAPTER_ID
        ]
    except KeyError as exc:
        raise RuntimeError(
            "AtomicESE formal registration must be configured before importing "
            "maple.function.benchmarking; launch this script in a fresh process."
        ) from exc

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
    selected, eligible_inventory = _select_output_blind_records(
        rows,
        taxonomy=taxonomy,
        derive_functional_group_assignment=derive_functional_group_assignment,
    )

    output_root = Path(output_directory).expanduser().resolve()
    input_root = output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Any] = {}
    assignments: dict[str, Any] = {}
    for record in selected:
        input_path = input_root / f"{record['record_id']}.xyz"
        geometry_generation = _write_mmff94_xyz(record, destination=input_path)
        record["geometry_generation"] = geometry_generation
        receipt = MolecularInputReceipt.from_file(
            input_path,
            molecular_input_format="headerless_xyz",
            locator=(
                f"flexisol-git:{FLEXISOL_SOURCE_REVISION}:"
                f"{FLEXISOL_REFERENCE_PATH}#L{record['row']['_line_number']};"
                f"maple-development-geometry:{input_path.name}"
            ),
        )
        expected_graph = "SMILES:" + Chem.MolToSmiles(
            Chem.MolFromSmiles(record["canonical_smiles"]),
            canonical=True,
            isomericSmiles=False,
        )
        if receipt.solute_structure_identifier != expected_graph:
            raise ValueError(
                f"Generated XYZ changed graph identity for {record['record_id']}: "
                f"{receipt.solute_structure_identifier!r} != {expected_graph!r}."
            )
        receipts[record["record_id"]] = receipt
        assignments[record["record_id"]] = derive_functional_group_assignment(
            record_id=record["record_id"],
            structure_identifier=receipt.solute_structure_identifier,
            taxonomy=taxonomy,
            assignment_evidence=(
                "frozen taxonomy SMARTS applied to the exact generated headerless "
                "XYZ graph before any AtomicESE output"
            ),
        )

    reference_payload: dict[str, Any] = {}
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
    fingerprint = registration.compute_fingerprint()
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
            "one deterministic RDKit ETKDGv3 embedding followed by fully "
            "specified MMFF94 optimization per solute"
        ),
        geometry_protocol=(
            "MAPLE development geometry, not reproduction of the author "
            "manual-correction workflow: AddHs; one ETKDGv3 embed with frozen "
            "identity-derived seed and explicit parameters; MMFF94 with explicit "
            "iteration and nonbonded settings; headerless fixed-precision XYZ"
        ),
        solvent_protocol=(
            "one pure FlexiSol solvent restricted to a locally runtime-verified "
            "AtomicESE alias; water excluded"
        ),
        potential=f"AtomicESE packaged Linux scalar predictor @ {fingerprint}",
        solvation_backend="AtomicESE direct scalar solvation-property predictor",
        cavity_model="not applicable to a scalar property predictor",
        sampling_protocol="no thermodynamic sampling; one development geometry",
        estimator="one unmodified AtomicESE scalar prediction",
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
        accuracy_adapter_fingerprint=fingerprint,
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
            "The packaged AtomicESE release does not expose complete canonical "
            "training-record identities sufficient for row-level overlap audit."
        ),
    )
    leakage = audit_training_overlap(identity, training_evidence)
    if leakage.status is not LeakageStatus.OVERLAP_UNKNOWN:
        raise RuntimeError(
            "FlexiSol/AtomicESE development panel must remain overlap_unknown."
        )
    ValidationPanel(
        model_id="atomicese-packaged-linux",
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
        "model_id": "atomicese-packaged-linux",
        "accuracy_adapter_id": registration.adapter_id,
        "accuracy_adapter_fingerprint": fingerprint,
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
        "model_source": {
            "repository": "https://github.com/vyboishchikov/AtomicESE",
            "revision": ATOMICESE_SOURCE_REVISION,
            "tree": ATOMICESE_SOURCE_TREE,
            "linux_binary_sha256": ATOMICESE_LINUX_BINARY_SHA256,
            "binary_copied_into_repository": False,
        },
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
            "selection_identity_fields": list(SELECTION_IDENTITY_FIELDS),
            "labels_in_selection_identity": False,
            "predictions_in_selection_identity": False,
            "policy": (
                "neutral single-fragment isotope-free 2-20-heavy-atom "
                "AtomicESE-supported-element rows in locally verified non-water "
                "solvents; frozen taxonomy; one minimum SHA256 identity per "
                "eligible primary group"
            ),
            **eligible_inventory,
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
                    "geometry_generation": record["geometry_generation"],
                }
                for record in selected
            ],
        },
        "leakage_audit": leakage.as_dict(),
        "accuracy_gate": {
            "evaluated_before_matched_qm": True,
            "maximum_absolute_error_limit_kcal_mol": (
                MAXIMUM_ABSOLUTE_ERROR_LIMIT_KCAL_MOL
            ),
            "strict_inequality_required": True,
            "unrounded_maximum_used": True,
            "passed": accuracy_gate_passed,
        },
        "matched_qm_gate": {
            "required_only_after_accuracy_passes": True,
            "status": (
                "pending_future_matched_qm_benchmark"
                if accuracy_gate_passed
                else "not_eligible_due_to_accuracy_failure"
            ),
            "runtime_measurements_recorded": False,
        },
        "accuracy": accuracy,
    }
    _write_json_atomic(output_root / "result.json", result)
    return result


def run(
    *,
    flexisol_source_root: str | Path,
    atomicese_source_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Execute into a fresh directory and publish it by atomic rename."""

    output_root = Path(output_directory).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output directory must not already exist: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        result = _run_into_output_directory(
            flexisol_source_root=flexisol_source_root,
            atomicese_source_root=atomicese_source_root,
            output_directory=temporary_root,
        )
        os.replace(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return result


def _tree_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_twice_and_publish(
    *,
    flexisol_source_root: str | Path,
    atomicese_source_root: str | Path,
    output_directory: str | Path,
    reproducibility_path: str | Path,
) -> dict[str, Any]:
    """Run twice, require byte-identical trees, then publish both artifacts."""

    output_root = Path(output_directory).expanduser().resolve()
    receipt_path = Path(reproducibility_path).expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output directory must not already exist: {output_root}")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(
            f"Reproducibility receipt must not already exist: {receipt_path}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    first = Path(
        tempfile.mkdtemp(prefix=".atomicese-formal-first-", dir=output_root.parent)
    )
    second = Path(
        tempfile.mkdtemp(prefix=".atomicese-formal-second-", dir=output_root.parent)
    )
    try:
        first_result = _run_into_output_directory(
            flexisol_source_root=flexisol_source_root,
            atomicese_source_root=atomicese_source_root,
            output_directory=first,
        )
        _run_into_output_directory(
            flexisol_source_root=flexisol_source_root,
            atomicese_source_root=atomicese_source_root,
            output_directory=second,
        )
        first_manifest = _tree_manifest(first)
        second_manifest = _tree_manifest(second)
        if first_manifest != second_manifest:
            raise RuntimeError(
                "Repeated AtomicESE/FlexiSol formal runs were not byte-identical."
            )
        tree_sha256 = _sha256_bytes(
            json.dumps(
                first_manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        receipt = {
            "schema_version": 1,
            "audited_on": AUDITED_ON,
            "run_count": 2,
            "byte_identical": True,
            "excluded_nondeterministic_fields": [],
            "canonical_tree_manifest": first_manifest,
            "canonical_tree_sha256": tree_sha256,
        }
        os.replace(first, output_root)
        _write_json_atomic(receipt_path, receipt)
        return first_result
    except BaseException:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)
        if receipt_path.exists():
            receipt_path.unlink()
        raise
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flexisol-source-root", required=True)
    parser.add_argument("--atomicese-source-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--reproducibility-path", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = run_twice_and_publish(
        flexisol_source_root=arguments.flexisol_source_root,
        atomicese_source_root=arguments.atomicese_source_root,
        output_directory=arguments.output_directory,
        reproducibility_path=arguments.reproducibility_path,
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
                "matched_qm_gate_status": result["matched_qm_gate"]["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
