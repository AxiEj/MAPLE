#!/usr/bin/env python3
"""Run the preregistered label-blind Route 1 phase-specific qRRHO test.

The energy stage deliberately never opens a FreeSolv label source. It consumes
only the frozen state, topology, charge, solvent-correction, and checkpoint
artifacts named by the protocol. Experimental scoring belongs to a separate
post-seal program.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, cast

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from ase.units import Hartree
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from source_compatibility import (  # noqa: E402
    require_exact_frozen_sources,
    validate_frozen_source,
)

from maple.function.calculator.calculator_base import (  # noqa: E402
    numerical_hessian_from_atoms,
)
from maple.function.calculator.extra_correction.implicit.openmm_gb import (  # noqa: E402
    OpenMMGB,
)
from maple.function.calculator.set_calculator import SetCalculator  # noqa: E402
from maple.function.dispatcher.frequency.frequency import MWFrequency  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

KCAL_PER_HARTREE = 627.5094740631
KJ_PER_KCAL = 4.184
R_KCAL_MOL_K = 0.00198720425864083
FORBIDDEN_LABEL_KEYS = {
    "experimental_kcal_mol",
    "experimental_reference",
    "experimental_uncertainty_kcal_mol",
}


def _relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Scientific evidence must remain inside the repository: {path}."
        ) from exc


def _resolve_repository_file(path: str, *, name: str) -> Path:
    """Resolve one normalized repository-relative, non-escaping file."""

    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError(f"{name} must be a non-empty POSIX repository path.")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must not be absolute or escape the repository.")
    candidate = (REPOSITORY_ROOT / relative).resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the repository through a symlink.") from exc
    if not candidate.is_file():
        raise ValueError(f"{name} is missing or is not a regular file: {path}.")
    return candidate


def _contains_forbidden_label_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_LABEL_KEYS or _contains_forbidden_label_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_label_key(item) for item in value)
    return False


def _float_list_sha256(values: Iterable[float]) -> str:
    return sha256_bytes(canonical_json_bytes([float(value) for value in values]))


def _checkpoint_path(model: dict[str, Any]) -> Path:
    return (
        REPOSITORY_ROOT
        / "maple/function/calculator/model"
        / model["checkpoint_filename"]
    )


def _validate_self_hash(artifact: dict[str, Any], *, name: str) -> None:
    expected = artifact.get("content_sha256")
    if not isinstance(expected, str) or expected != artifact_content_sha256(artifact):
        raise ValueError(f"{name} has an invalid content SHA256.")


def select_protocol_source_cases(
    manifest_cases: Iterable[dict[str, Any]],
    *,
    target_state_counts: Iterable[int],
    maximum_atom_count: int,
) -> list[dict[str, Any]]:
    """Apply the frozen cost-bounded, label-blind source-selection rule."""

    cases = list(manifest_cases)
    targets = [int(value) for value in target_state_counts]
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("Source-selection targets must be positive integers.")
    if maximum_atom_count <= 0:
        raise ValueError("The source-selection atom ceiling must be positive.")
    selected = []
    used: set[str] = set()
    for target in targets:
        eligible = [
            case
            for case in cases
            if len(case["atomic_numbers"]) <= maximum_atom_count
            and case["compound_id"] not in used
        ]
        if not eligible:
            raise ValueError("The frozen source-selection rule exhausted candidates.")
        chosen = min(
            eligible,
            key=lambda case: (
                abs(math.log(int(case["state_count"])) - math.log(target)),
                len(case["atomic_numbers"]),
                case["compound_id"],
            ),
        )
        selected.append(chosen)
        used.add(chosen["compound_id"])
    return selected


def load_qrrho_protocol(
    path: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load and fully validate the preregistered protocol and label-free source."""

    protocol_path = Path(path).resolve()
    protocol = load_json(protocol_path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unsupported phase-specific qRRHO protocol schema.")
    if protocol.get("protocol_id") != (
        "maple-route1-multi-mlip-phase-specific-selected-minimum-rrho-v8"
    ):
        raise ValueError("The active phase-specific qRRHO protocol is not v8.")
    if protocol.get("source_partition") != "development":
        raise ValueError("The qRRHO falsification run must remain development-only.")
    if protocol.get("status") != (
        "preregistered-before-energy-evaluation-and-label-scoring"
    ):
        raise ValueError("The qRRHO protocol is not in its preregistered state.")
    route = protocol.get("route", {})
    if (
        route.get("gas_phase_mm_energy") is not False
        or route.get("hydration_label_residual") is not False
        or route.get("mlip_retraining") is not False
        or route.get("charge_refitting") is not False
    ):
        raise ValueError("The qRRHO protocol violates the strict Route 1 boundary.")
    solvation = protocol.get("solvation", {})
    if (
        solvation.get("charge_method") != "am1bcc"
        or solvation.get("charge_mode") != "fixed"
        or solvation.get("charge_source") != "frozen-am1bcc-json"
        or solvation.get("method") != "gb"
        or solvation.get("provider") != "openmm"
        or solvation.get("polar_model") != "obc2"
        or solvation.get("nonpolar_model") != "ace"
    ):
        raise ValueError("The qRRHO solvent endpoint is not frozen AM1-BCC/OBC-II/ACE.")
    thermochemistry = protocol.get("thermochemistry", {})
    if (
        thermochemistry.get("primary", {}).get("ilowfreq") != 0
        or thermochemistry.get("sensitivity_variants_are_not_primary") is not True
    ):
        raise ValueError("The v8 primary must remain local harmonic RRHO.")
    estimator = protocol.get("ensemble_estimator", {})
    claim_boundary = protocol.get("claim_boundary", {})
    if (
        estimator.get("estimator_identity") != "selected-minimum local-RRHO surrogate"
        or estimator.get("not_exact_cartesian_endpoint_ratio") is not True
        or claim_boundary.get("absolute_hydration_free_energy_estimator_under_test")
        is not False
        or claim_boundary.get("selected_minimum_local_rrho_surrogate_under_test")
        is not True
    ):
        raise ValueError("The v8 local-surrogate claim boundary changed.")
    hessian_specification = protocol.get("hessian_and_stationary_point", {})
    if (
        hessian_specification.get("imaginary_frequency_cutoff_cm1") != 0.0
        or hessian_specification.get("every_unique_minimum_must_be_valid") is not True
    ):
        raise ValueError("The v8 stationary-point fail-closed policy changed.")
    recovery = hessian_specification.get("imaginary_mode_recovery", {})
    if (
        recovery.get("enabled") is not True
        or recovery.get("directions") != [-1, 1]
        or recovery.get("maximum_recovery_cycles") != 1
        or float(recovery.get("maximum_atom_displacement_angstrom", math.nan))
        != 0.5
        or float(recovery.get("minimum_energy_lowering_kcal_mol", math.nan))
        != 0.001
    ):
        raise ValueError("The v8 imaginary-mode recovery policy changed.")
    qualification = hessian_specification.get(
        "label_blind_numerical_qualification",
        {},
    )
    qualification_path = _resolve_repository_file(
        qualification.get("artifact"),
        name="Label-blind Hessian qualification artifact",
    )
    if sha256_file(qualification_path) != qualification.get(
        "artifact_file_sha256"
    ):
        raise ValueError("The label-blind Hessian qualification file changed.")
    qualification_artifact = load_json(qualification_path)
    _validate_self_hash(
        qualification_artifact,
        name="Label-blind Hessian qualification artifact",
    )
    if (
        qualification_artifact.get("content_sha256")
        != qualification.get("artifact_content_sha256")
        or qualification_artifact.get("label_boundary", {}).get(
            "experimental_labels_read"
        )
        is not False
        or qualification.get("experimental_labels_read") is not False
    ):
        raise ValueError("The v8 Hessian qualification crossed the label boundary.")
    amendment_audit = protocol.get("engineering_amendment_audit", {})
    amendment_audit_path = _resolve_repository_file(
        amendment_audit.get("artifact"),
        name="v7 engineering-amendment audit artifact",
    )
    if sha256_file(amendment_audit_path) != amendment_audit.get(
        "artifact_file_sha256"
    ):
        raise ValueError("The v7 engineering-amendment audit file changed.")
    amendment_artifact = load_json(amendment_audit_path)
    _validate_self_hash(
        amendment_artifact,
        name="v7 engineering-amendment audit artifact",
    )
    if (
        amendment_artifact.get("content_sha256")
        != amendment_audit.get("artifact_content_sha256")
        or amendment_artifact.get("status")
        != "invalidated-before-seal-or-label-scoring"
        or amendment_audit.get("experimental_labels_read") is not False
        or amendment_audit.get("v7_records_reused") is not False
    ):
        raise ValueError("The v8 engineering amendment is not label-blind.")

    source = protocol.get("source", {})
    manifest_path = protocol_path.parent / source["manifest"]
    if sha256_file(manifest_path) != source["manifest_file_sha256"]:
        raise ValueError("The frozen source-manifest file SHA256 changed.")
    manifest = load_json(manifest_path)
    _validate_self_hash(manifest, name="Source manifest")
    if manifest["content_sha256"] != source["manifest_content_sha256"]:
        raise ValueError("The frozen source-manifest content SHA256 changed.")
    if manifest.get("source_partition") != "development":
        raise ValueError("The source manifest is not development-only.")
    if (
        manifest["label_boundary"]["prepared_state_manifest_contains_labels"]
        is not False
    ):
        raise ValueError("The source manifest crosses the label boundary.")

    selected_source_cases = select_protocol_source_cases(
        manifest["cases"],
        target_state_counts=source["selection_target_state_counts"],
        maximum_atom_count=int(source["maximum_atom_count"]),
    )
    if len(selected_source_cases) != int(source["expected_case_count"]):
        raise ValueError("The label-blind source case count changed.")
    if sum(int(case["state_count"]) for case in selected_source_cases) != int(
        source["expected_source_state_count"]
    ):
        raise ValueError("The label-blind source state count changed.")
    protocol_cases = protocol.get("cases", [])
    if [case["compound_id"] for case in protocol_cases] != [
        case["compound_id"] for case in selected_source_cases
    ]:
        raise ValueError("The preregistered case order changed.")

    source_by_id = {case["compound_id"]: case for case in selected_source_cases}
    for case in protocol_cases:
        source_case = source_by_id[case["compound_id"]]
        if int(case["source_state_count"]) != int(source_case["state_count"]):
            raise ValueError(f"State count changed for {case['compound_id']}.")
        symmetry_number = case.get("rotational_symmetry_number")
        if (
            not isinstance(symmetry_number, int)
            or isinstance(symmetry_number, bool)
            or symmetry_number <= 0
        ):
            raise ValueError(
                f"Rotational symmetry number is invalid for {case['compound_id']}."
            )
        state_path = REPOSITORY_ROOT / case["state_file"]
        if (
            case["state_file"] != source_case["state_file"]
            or case["state_file_sha256"] != source_case["state_file_sha256"]
            or sha256_file(state_path) != case["state_file_sha256"]
        ):
            raise ValueError(f"Frozen state file changed for {case['compound_id']}.")
        mol2_path = REPOSITORY_ROOT / case["mol2_file"]
        if (
            case["mol2_file"] != source_case["source"]["mol2"]
            or case["mol2_file_sha256"] != source_case["source"]["mol2_sha256"]
            or sha256_file(mol2_path) != case["mol2_file_sha256"]
        ):
            raise ValueError(f"Frozen MOL2 changed for {case['compound_id']}.")
        charge_path = REPOSITORY_ROOT / case["charge_file"]
        if sha256_file(charge_path) != case["charge_file_sha256"]:
            raise ValueError(
                f"Frozen AM1-BCC record changed for {case['compound_id']}."
            )
        charge_record = load_json(charge_path)
        if (
            charge_record.get("charge_method") != "am1bcc"
            or charge_record.get("mol2_sha256") != case["mol2_file_sha256"]
            or _float_list_sha256(charge_record["charges_e"])
            != case["charge_vector_sha256"]
        ):
            raise ValueError(
                f"Frozen AM1-BCC vector is inconsistent for {case['compound_id']}."
            )
        if _contains_forbidden_label_key(charge_record):
            raise ValueError(
                f"Charge record contains a label for {case['compound_id']}."
            )

    manifest_models = {model["name"]: model for model in manifest["models"]}
    models = protocol.get("models", [])
    if [model["name"] for model in models] != source["expected_models"]:
        raise ValueError("The preregistered MLIP order changed.")
    for model in models:
        source_model = manifest_models[model["name"]]
        checkpoint = _checkpoint_path(model)
        if (
            model["checkpoint_sha256"] != source_model["checkpoint_sha256"]
            or sha256_file(checkpoint) != model["checkpoint_sha256"]
        ):
            raise ValueError(f"Checkpoint changed for {model['name']}.")

    implementation = protocol.get("implementation_freeze", {})
    required_python = implementation.get("required_python")
    observed_python = platform.python_version()
    if observed_python != required_python:
        raise ValueError(
            "Required Python version changed: "
            f"expected {required_python}, got {observed_python}."
        )
    for name, expected in implementation["required_versions"].items():
        observed = importlib.metadata.version(name)
        if observed != expected:
            raise ValueError(
                f"Required {name} version changed: expected {expected}, got {observed}."
            )
    for record in implementation["source_files"]:
        validate_frozen_source(
            REPOSITORY_ROOT,
            record["path"],
            record["sha256"],
        )
    scoring = protocol.get("post_seal_scoring", {})
    scoring_script = REPOSITORY_ROOT / scoring["script"]
    if sha256_file(scoring_script) != scoring["script_sha256"]:
        raise ValueError("The preregistered post-seal scoring program changed.")
    frozen_sources = {
        record["path"]: record["sha256"] for record in implementation["source_files"]
    }
    if frozen_sources.get(scoring["script"]) != scoring["script_sha256"]:
        raise ValueError("The scoring-program freeze is internally inconsistent.")

    execution = protocol.get("execution", {})
    if execution.get("device") != "cuda:0":
        raise ValueError("The qRRHO numerical experiment is pinned to cuda:0.")
    if int(execution.get("batch_size", 0)) <= 0:
        raise ValueError("The qRRHO batch size must be positive.")
    if int(execution.get("source_energy_repeat_count", 0)) < 2:
        raise ValueError("Source-state energy evaluation needs at least two repeats.")
    repeat_tolerance = execution.get(
        "maximum_repeat_relative_energy_difference_kcal_mol"
    )
    if (
        not isinstance(repeat_tolerance, (int, float))
        or isinstance(repeat_tolerance, bool)
        or not math.isfinite(float(repeat_tolerance))
        or float(repeat_tolerance) <= 0.0
    ):
        raise ValueError("The source-energy repeat tolerance must be positive.")
    tolerance_provenance = execution.get("repeat_tolerance_provenance", {})
    prior_artifact_value = tolerance_provenance.get("source_artifact")
    if not isinstance(prior_artifact_value, str):
        raise ValueError("The repeat-tolerance source artifact path is invalid.")
    prior_artifact_path = _resolve_repository_file(
        prior_artifact_value,
        name="repeat-tolerance source artifact",
    )
    if sha256_file(prior_artifact_path) != tolerance_provenance.get(
        "source_artifact_file_sha256"
    ):
        raise ValueError("The repeat-tolerance source artifact file changed.")
    prior_artifact = load_json(prior_artifact_path)
    _validate_self_hash(prior_artifact, name="Repeat-tolerance source artifact")
    if prior_artifact.get("content_sha256") != tolerance_provenance.get(
        "source_artifact_content_sha256"
    ) or _contains_forbidden_label_key(prior_artifact):
        raise ValueError("The repeat-tolerance source artifact is not label-free.")
    observed_prior_maximum = max(
        float(record["maximum_repeat_relative_energy_difference_kcal_mol"])
        for record in prior_artifact["records"]
    )
    if (
        not math.isclose(
            observed_prior_maximum,
            float(
                tolerance_provenance[
                    "maximum_prior_label_free_repeat_difference_kcal_mol"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or float(repeat_tolerance) <= observed_prior_maximum
    ):
        raise ValueError("The repeat tolerance is inconsistent with its frozen basis.")
    if int(protocol["energy_artifact_gates"]["expected_model_case_count"]) != (
        len(models) * len(protocol_cases)
    ):
        raise ValueError("The expected model-case count is inconsistent.")
    if _contains_forbidden_label_key(protocol_cases):
        raise ValueError("The preregistered case table contains a label.")

    fingerprint = sha256_bytes(canonical_json_bytes(protocol))
    return protocol, fingerprint, manifest


def _load_case_sources(
    *,
    protocol_case: dict[str, Any],
    source_case: dict[str, Any],
) -> tuple[Atoms, np.ndarray, np.ndarray, dict[str, Any]]:
    state_path = REPOSITORY_ROOT / protocol_case["state_file"]
    positions = np.load(state_path, allow_pickle=False)
    expected_shape = tuple(source_case["state_shape"])
    if positions.shape != expected_shape or not np.isfinite(positions).all():
        raise ValueError(
            f"Invalid state array for {protocol_case['compound_id']}: "
            f"{positions.shape}, expected {expected_shape}."
        )

    mol2_path = REPOSITORY_ROOT / protocol_case["mol2_file"]
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    observed_numbers = [int(value) for value in atoms.get_atomic_numbers()]
    if observed_numbers != [int(value) for value in source_case["atomic_numbers"]]:
        raise ValueError(f"MOL2 atom order changed for {protocol_case['compound_id']}.")

    charge_path = REPOSITORY_ROOT / protocol_case["charge_file"]
    charge_record = load_json(charge_path)
    charges = np.asarray(charge_record["charges_e"], dtype=np.float64)
    if (
        charges.shape != (len(atoms),)
        or not np.isfinite(charges).all()
        or abs(float(charges.sum())) > 1.0e-10
    ):
        raise ValueError(
            f"Invalid neutral AM1-BCC vector for {protocol_case['compound_id']}."
        )
    atoms.set_initial_charges(charges)

    solvent = np.asarray(
        source_case["solvent_correction_kcal_mol"],
        dtype=np.float64,
    )
    if (
        solvent.shape != (len(positions),)
        or not np.isfinite(solvent).all()
        or _float_list_sha256(solvent) != source_case["solvent_correction_sha256"]
    ):
        raise ValueError(
            f"Frozen solvent corrections changed for {protocol_case['compound_id']}."
        )
    charge_evidence = {
        "charge_method": "am1bcc",
        "charge_record": protocol_case["charge_file"],
        "charge_record_sha256": protocol_case["charge_file_sha256"],
        "charge_vector_sha256": protocol_case["charge_vector_sha256"],
        "sum_e": float(charges.sum()),
        "mol2_file": protocol_case["mol2_file"],
        "mol2_file_sha256": protocol_case["mol2_file_sha256"],
        "provenance": charge_record.get("provenance", {}),
    }
    return atoms, positions, solvent, charge_evidence


def _heavy_atom_indices(symbols: list[str]) -> np.ndarray:
    indices = np.asarray(
        [index for index, symbol in enumerate(symbols) if symbol != "H"],
        dtype=int,
    )
    return indices if len(indices) else np.arange(len(symbols), dtype=int)


def _aligned_rmsd(
    left: np.ndarray,
    right: np.ndarray,
    atom_indices: np.ndarray,
) -> float:
    """Return a reflection-safe Kabsch RMSD over the selected atoms."""

    left_selected = np.asarray(left, dtype=np.float64)[atom_indices]
    right_selected = np.asarray(right, dtype=np.float64)[atom_indices]
    left_centered = left_selected - left_selected.mean(axis=0)
    right_centered = right_selected - right_selected.mean(axis=0)
    covariance = left_centered.T @ right_centered
    left_vectors, _singular_values, right_vectors = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left_vectors @ right_vectors))
    rotation = left_vectors @ correction @ right_vectors
    difference = left_centered @ rotation - right_centered
    return float(np.sqrt(np.square(difference).sum() / len(atom_indices)))


def select_diverse_seeds(
    *,
    positions_angstrom: np.ndarray,
    potential_hartree: np.ndarray,
    symbols: list[str],
    maximum_count: int,
    threshold_angstrom: float,
) -> list[dict[str, Any]]:
    """Select deterministic low-potential, heavy-atom-diverse phase seeds."""

    positions = np.asarray(positions_angstrom, dtype=np.float64)
    potential = np.asarray(potential_hartree, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[2] != 3
        or potential.shape != (len(positions),)
        or len(symbols) != positions.shape[1]
        or not np.isfinite(positions).all()
        or not np.isfinite(potential).all()
    ):
        raise ValueError("Seed positions and potentials have incompatible shapes.")
    if maximum_count <= 0 or threshold_angstrom <= 0.0:
        raise ValueError("Seed count and RMSD threshold must be positive.")

    indices = np.arange(len(positions))
    ranking = np.lexsort((indices, potential))
    heavy = _heavy_atom_indices(symbols)
    selected: list[dict[str, Any]] = []
    for index in ranking:
        distances = [
            _aligned_rmsd(
                positions[index],
                positions[accepted["source_state_index"]],
                heavy,
            )
            for accepted in selected
        ]
        if distances and min(distances) <= threshold_angstrom:
            continue
        selected.append(
            {
                "source_state_index": int(index),
                "phase_potential_hartree": float(potential[index]),
                "minimum_rmsd_to_prior_seed_angstrom": (
                    float(min(distances)) if distances else None
                ),
            }
        )
        if len(selected) == maximum_count:
            break
    return selected


class HartreeToEVCalculator(Calculator):
    """Expose MAPLE's Hartree contract to ASE optimizers in native eV."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, source_calculator):
        super().__init__()
        self.source_calculator = source_calculator

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        requested = set(properties or ["energy"])
        source_properties = ["energy"]
        if "forces" in requested:
            source_properties.append("forces")
        self.source_calculator.calculate(
            atoms,
            properties=source_properties,
            system_changes=system_changes,
        )
        source = self.source_calculator.results
        results: dict[str, Any] = {
            "energy": float(source["energy"]) * Hartree,
            "free_energy": float(source.get("free_energy", source["energy"])) * Hartree,
        }
        if "forces" in source:
            results["forces"] = np.asarray(source["forces"], dtype=np.float64) * Hartree
        self.results = results


class FrozenChargeOpenMMCorrection:
    """Composed-force correction backed by one pinned external charge vector."""

    mode = "fixed"
    supported_properties = frozenset({"energy", "forces"})

    def __init__(
        self,
        atoms: Atoms,
        charges: np.ndarray,
        *,
        charge_evidence: dict[str, Any],
        model: str,
        nonpolar: str,
        platform: str,
    ):
        self.provider = OpenMMGB(
            atoms,
            charges,
            model=model,
            nonpolar=nonpolar,
            platform=platform,
        )
        self.charge_evidence = dict(charge_evidence)

    def evaluate(
        self,
        atoms: Atoms,
        need_forces: bool = False,
        calculator=None,
    ):
        result = self.provider.evaluate(
            atoms,
            need_forces=need_forces,
            calculator=calculator,
        )
        result.provenance = {
            **result.provenance,
            "charge_provider": {
                "source": "frozen-am1bcc-json",
                **self.charge_evidence,
            },
        }
        return result


def _source_energy_result(source_calculator) -> dict[str, Any]:
    source = source_calculator.results
    total = float(source["energy"])
    if not math.isfinite(total):
        raise ValueError("MAPLE returned a non-finite energy.")
    structured = source.get("solvation")
    if structured is None:
        return {
            "total_energy_hartree": total,
            "gas_energy_hartree": total,
            "solvent_energy_hartree": 0.0,
            "solvent_components_hartree": {},
            "solvent_provenance": {},
        }
    gas = float(structured["gas_energy_hartree"])
    solvent = float(structured["energy_hartree"])
    components = {
        str(key): float(value)
        for key, value in structured["components_hartree"].items()
    }
    if abs(total - gas - solvent) > 1.0e-10:
        raise ValueError("The composed phase energy does not close.")
    if abs(solvent - sum(components.values())) > 1.0e-10:
        raise ValueError("The solvent component energies do not close.")
    return {
        "total_energy_hartree": total,
        "gas_energy_hartree": gas,
        "solvent_energy_hartree": solvent,
        "solvent_components_hartree": components,
        "solvent_provenance": dict(structured["provenance"]),
    }


def _maximum_force(forces_eV_per_angstrom: np.ndarray) -> float:
    forces = np.asarray(forces_eV_per_angstrom, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
        raise ValueError("Optimizer forces must be a finite (N, 3) array.")
    return float(np.linalg.norm(forces, axis=1).max())


def _run_optimization(
    *,
    template_atoms: Atoms,
    initial_positions: np.ndarray,
    source_state_index: int,
    source_calculator,
    optimizer_options: dict[str, Any],
    phase: str,
    branch_dir: Path,
) -> dict[str, Any]:
    atoms = template_atoms.copy()
    atoms.set_positions(initial_positions)
    atoms.calc = HartreeToEVCalculator(source_calculator)
    branch_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        initial_forces = atoms.get_forces()
        initial_source = _source_energy_result(source_calculator)
        optimizer = LBFGS(
            atoms,
            logfile=str(branch_dir / f"{phase}.log"),
            trajectory=str(branch_dir / f"{phase}.traj"),
            maxstep=float(optimizer_options["maxstep_angstrom"]),
        )
        optimizer_reported_converged = bool(
            optimizer.run(
                fmax=float(optimizer_options["fmax_eV_per_angstrom"]),
                steps=int(optimizer_options["max_steps"]),
            )
        )
        final_forces = atoms.get_forces()
        final_source = _source_energy_result(source_calculator)
        final_fmax = _maximum_force(final_forces)
        converged = bool(
            optimizer_reported_converged
            and final_fmax <= float(optimizer_options["fmax_eV_per_angstrom"]) + 1.0e-12
        )
        return {
            "status": "success",
            "phase": phase,
            "source_state_index": int(source_state_index),
            "converged": converged,
            "optimizer_reported_converged": optimizer_reported_converged,
            "steps": int(optimizer.nsteps),
            "elapsed_seconds": time.perf_counter() - start,
            "initial_max_force_eV_per_angstrom": _maximum_force(initial_forces),
            "final_max_force_eV_per_angstrom": final_fmax,
            "initial_energy": initial_source,
            "final_energy": final_source,
            "final_positions_angstrom": atoms.get_positions().tolist(),
        }
    except Exception as exc:
        return {
            "status": "failure",
            "phase": phase,
            "source_state_index": int(source_state_index),
            "converged": False,
            "elapsed_seconds": time.perf_counter() - start,
            "failure": {
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            },
        }


def deduplicate_optimized_minima(
    *,
    branches: list[dict[str, Any]],
    symbols: list[str],
    threshold_angstrom: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate converged minima without treating arrival count as degeneracy."""

    candidates = [
        branch
        for branch in branches
        if branch.get("status") == "success" and branch.get("converged") is True
    ]
    candidates.sort(
        key=lambda branch: (
            float(branch["final_energy"]["total_energy_hartree"]),
            int(branch["source_state_index"]),
        )
    )
    heavy = _heavy_atom_indices(symbols)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for branch in candidates:
        positions = np.asarray(
            branch["final_positions_angstrom"],
            dtype=np.float64,
        )
        distances = [
            _aligned_rmsd(
                positions,
                np.asarray(other["final_positions_angstrom"], dtype=np.float64),
                heavy,
            )
            for other in accepted
        ]
        if distances and min(distances) <= threshold_angstrom:
            rejected.append(
                {
                    "source_state_index": int(branch["source_state_index"]),
                    "duplicate_of_source_state_index": int(
                        accepted[int(np.argmin(distances))]["source_state_index"]
                    ),
                    "aligned_heavy_atom_rmsd_angstrom": float(min(distances)),
                }
            )
        else:
            accepted.append(branch)
    return accepted, rejected


def validate_selected_branch_accounting(
    branches: list[dict[str, Any]],
    *,
    duplicate_minima: list[dict[str, Any]],
    unique_minimum_count: int,
) -> dict[str, Any]:
    """Require every selected seed to resolve without silently dropping a branch."""

    if any(
        branch.get("status") != "success" or branch.get("converged") is not True
        for branch in branches
    ):
        raise ValueError(
            "Every selected optimization branch must succeed and converge before "
            "deduplication."
        )
    resolved = int(unique_minimum_count) + len(duplicate_minima)
    if resolved != len(branches):
        raise ValueError(
            "Selected optimization branches did not resolve exactly as unique "
            "minima or explicit duplicates."
        )
    return {
        "selected_seed_count": len(branches),
        "valid_unique_minimum_count": int(unique_minimum_count),
        "explicit_duplicate_count": len(duplicate_minima),
        "all_selected_seeds_resolved": True,
    }


def reconcile_post_analysis_deduplication(
    *,
    branches: list[dict[str, Any]],
    initially_unique: list[dict[str, Any]],
    minimum_analyses: list[dict[str, Any]],
    symbols: list[str],
    threshold_angstrom: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Recompute duplicate evidence after any imaginary-mode reoptimization."""

    unique, duplicates = deduplicate_optimized_minima(
        branches=branches,
        symbols=symbols,
        threshold_angstrom=threshold_angstrom,
    )
    initial_ids = {
        int(branch["source_state_index"]) for branch in initially_unique
    }
    final_ids = {int(branch["source_state_index"]) for branch in unique}
    if initial_ids != final_ids:
        raise ValueError(
            "Stationary-point recovery changed the unique-minimum membership; "
            "the model-case is fail-closed rather than silently changing its "
            "minimum accounting."
        )
    analysis_by_source = {
        int(analysis["source_state_index"]): analysis
        for analysis in minimum_analyses
    }
    ordered_analyses = [
        analysis_by_source[int(branch["source_state_index"])]
        for branch in unique
    ]
    accounting = validate_selected_branch_accounting(
        branches,
        duplicate_minima=duplicates,
        unique_minimum_count=len(unique),
    )
    return unique, duplicates, ordered_analyses, accounting


def select_vibrational_modes(
    frequencies_cm1: np.ndarray,
    *,
    atom_count: int,
    linear: bool,
    imaginary_frequency_cutoff_cm1: float,
    maximum_rigid_mode_leakage_cm1: float,
) -> dict[str, Any]:
    """Select the projected vibrational subspace and apply fail-closed gates."""

    frequencies = np.asarray(frequencies_cm1, dtype=np.float64)
    if frequencies.shape != (3 * atom_count,) or not np.isfinite(frequencies).all():
        raise ValueError("Projected frequency array has an invalid shape or value.")
    count = 3 * atom_count - (5 if linear else 6)
    if count <= 0:
        raise ValueError("The preregistered molecular qRRHO path needs vibrations.")
    indices = np.arange(len(frequencies))
    ranking = np.lexsort((indices, -np.abs(frequencies)))
    selected_indices = np.sort(ranking[:count])
    selected = frequencies[selected_indices]
    rigid_indices = np.sort(ranking[count:])
    rigid = frequencies[rigid_indices]
    negative = selected[selected < imaginary_frequency_cutoff_cm1]
    maximum_leakage = float(np.max(np.abs(rigid))) if len(rigid) else 0.0
    leakage_passed = maximum_leakage <= maximum_rigid_mode_leakage_cm1
    return {
        "linear": bool(linear),
        "expected_vibrational_mode_count": int(count),
        "selected_indices": selected_indices.astype(int).tolist(),
        "selected_frequencies_cm1": selected.tolist(),
        "thermochemistry_frequencies_cm1": selected.tolist(),
        "negative_selected_frequencies_cm1": negative.tolist(),
        "rigid_mode_indices": rigid_indices.astype(int).tolist(),
        "rigid_mode_frequencies_cm1": rigid.tolist(),
        "maximum_rigid_mode_leakage_cm1": maximum_leakage,
        "rigid_mode_leakage_gate_passed": leakage_passed,
        "valid_stationary_point": not len(negative) and leakage_passed,
    }


def validate_hessian_numerical_quality(
    diagnostics: dict[str, Any],
    *,
    specification: dict[str, Any],
) -> None:
    """Apply absolute-sanity and scale-aware raw Hessian quality gates."""

    keys = (
        "maximum_absolute_raw_hessian_hartree_per_angstrom2",
        "maximum_raw_asymmetry_hartree_per_angstrom2",
        "raw_hessian_frobenius_norm_hartree_per_angstrom2",
        "raw_asymmetry_frobenius_norm_hartree_per_angstrom2",
        "relative_raw_asymmetry_frobenius",
    )
    values = {key: float(diagnostics.get(key, math.nan)) for key in keys}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("Raw finite-difference Hessian diagnostics are invalid.")
    raw_norm = values["raw_hessian_frobenius_norm_hartree_per_angstrom2"]
    asymmetry_norm = values[
        "raw_asymmetry_frobenius_norm_hartree_per_angstrom2"
    ]
    expected_relative = asymmetry_norm / raw_norm if raw_norm else 0.0
    if not math.isclose(
        values["relative_raw_asymmetry_frobenius"],
        expected_relative,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise ValueError("The relative raw Hessian asymmetry is inconsistent.")
    if values["maximum_raw_asymmetry_hartree_per_angstrom2"] > float(
        specification[
            "maximum_absolute_raw_hessian_asymmetry_hartree_per_angstrom2"
        ]
    ):
        raise ValueError(
            "Raw finite-difference Hessian asymmetry exceeds the absolute "
            "sanity ceiling."
        )
    if values["relative_raw_asymmetry_frobenius"] > float(
        specification["maximum_relative_raw_asymmetry_frobenius"]
    ):
        raise ValueError(
            "Raw finite-difference Hessian asymmetry exceeds the scale-aware "
            "Frobenius gate."
        )


def stable_ensemble_free_energy(
    free_energies_kcal_mol: Iterable[float],
    *,
    temperature_kelvin: float,
) -> dict[str, Any]:
    """Return a stable unit-weight minimum ensemble free energy."""

    values = np.asarray(list(free_energies_kcal_mol), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("An ensemble requires finite one-dimensional free energies.")
    if not math.isfinite(temperature_kelvin) or temperature_kelvin <= 0.0:
        raise ValueError("Ensemble temperature must be positive and finite.")
    rt = R_KCAL_MOL_K * temperature_kelvin
    minimum = float(values.min())
    boltzmann = np.exp(-(values - minimum) / rt)
    partition = float(boltzmann.sum())
    weights = boltzmann / partition
    return {
        "free_energy_kcal_mol": minimum - rt * math.log(partition),
        "minimum_free_energy_kcal_mol": minimum,
        "unit_weight_partition_sum": partition,
        "normalized_weights": weights.tolist(),
        "effective_minimum_count": float(1.0 / np.square(weights).sum()),
        "maximum_normalized_weight": float(weights.max()),
    }


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.asarray(array, dtype=np.float64), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _requires_hessian_displacement_sensitivity(
    protocol: dict[str, Any],
    *,
    compound_id: str,
    model_name: str,
    phase: str,
    minimum_index: int,
) -> bool:
    sensitivity = protocol["hessian_and_stationary_point"]["displacement_sensitivity"]
    return (
        compound_id in sensitivity["compound_ids"]
        and model_name in sensitivity["model_names"]
        and phase in sensitivity["phases"]
        and minimum_index in sensitivity["unique_minimum_indices"]
    )


def _compute_hessian_analysis(
    *,
    atoms: Atoms,
    source_calculator,
    protocol: dict[str, Any],
    delta: float,
    hessian_path: Path,
    frequency_output: Path,
) -> dict[str, Any]:
    specification = protocol["hessian_and_stationary_point"]
    hessian, diagnostics = cast(
        tuple[np.ndarray, dict[str, float]],
        numerical_hessian_from_atoms(
            source_calculator,
            atoms,
            delta=delta,
            return_diagnostics=True,
        ),
    )
    validate_hessian_numerical_quality(
        diagnostics,
        specification=specification,
    )
    _write_npy_atomic(hessian_path, hessian)
    primary = protocol["thermochemistry"]["primary"]
    frequency = MWFrequency(
        str(frequency_output),
        atoms,
        temperature=float(protocol["thermochemistry"]["temperature_kelvin"]),
        ilowfreq=int(primary["ilowfreq"]),
        omega0_cm1=float(primary["omega0_cm1"]),
        nu_floor_cm1=float(primary["nu_floor_cm1"]),
        device="cpu",
    )
    frequency.verbosity = 0
    frequencies, modes = frequency.compute_frequencies(hessian)
    selection = select_vibrational_modes(
        frequencies,
        atom_count=len(atoms),
        linear=frequency._is_linear_molecule(
            tol=float(specification["linearity_moment_ratio_threshold"])
        ),
        imaginary_frequency_cutoff_cm1=float(
            specification["imaginary_frequency_cutoff_cm1"]
        ),
        maximum_rigid_mode_leakage_cm1=float(
            specification["maximum_rigid_mode_leakage_cm1"]
        ),
    )
    return {
        "hessian": hessian,
        "diagnostics": diagnostics,
        "path": hessian_path,
        "frequencies_cm1": np.asarray(frequencies, dtype=np.float64),
        "modes_cartesian": np.asarray(modes, dtype=np.float64),
        "selection": selection,
    }


def _recover_imaginary_mode_minimum(
    *,
    branch: dict[str, Any],
    initial_analysis: dict[str, Any],
    template_atoms: Atoms,
    source_calculator,
    protocol: dict[str, Any],
    phase: str,
    minimum_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = protocol["hessian_and_stationary_point"][
        "imaginary_mode_recovery"
    ]
    selection = initial_analysis["selection"]
    frequencies = initial_analysis["frequencies_cm1"]
    negative_indices = [
        int(index)
        for index in selection["selected_indices"]
        if float(frequencies[index]) < 0.0
    ]
    if not negative_indices:
        raise ValueError(
            "Stationary-point recovery is allowed only for a selected negative "
            "vibrational mode."
        )
    mode_index = min(negative_indices, key=lambda index: float(frequencies[index]))
    mode = np.asarray(
        initial_analysis["modes_cartesian"][mode_index],
        dtype=np.float64,
    ).reshape(len(template_atoms), 3)
    maximum_atom_norm = float(np.linalg.norm(mode, axis=1).max())
    if not math.isfinite(maximum_atom_norm) or maximum_atom_norm <= 0.0:
        raise ValueError("The imaginary normal mode cannot define a displacement.")
    amplitude = float(recovery["maximum_atom_displacement_angstrom"])
    displacement = mode * (amplitude / maximum_atom_norm)
    original_positions = np.asarray(
        branch["final_positions_angstrom"],
        dtype=np.float64,
    )
    trials = []
    for direction in (-1, 1):
        trial = _run_optimization(
            template_atoms=template_atoms,
            initial_positions=original_positions + direction * displacement,
            source_state_index=int(branch["source_state_index"]),
            source_calculator=source_calculator,
            optimizer_options=protocol["optimization"],
            phase=phase,
            branch_dir=minimum_dir
            / "stationary-point-recovery"
            / ("minus" if direction < 0 else "plus"),
        )
        trials.append({"direction": direction, "optimization": trial})
    converged = [
        trial
        for trial in trials
        if trial["optimization"].get("status") == "success"
        and trial["optimization"].get("converged") is True
    ]
    if not converged:
        raise ValueError(
            "Neither imaginary-mode displacement direction converged."
        )
    chosen = min(
        converged,
        key=lambda trial: (
            float(trial["optimization"]["final_energy"]["total_energy_hartree"]),
            int(trial["direction"]),
        ),
    )
    original_energy = float(branch["final_energy"]["total_energy_hartree"])
    chosen_energy = float(
        chosen["optimization"]["final_energy"]["total_energy_hartree"]
    )
    energy_lowering = (original_energy - chosen_energy) * KCAL_PER_HARTREE
    if energy_lowering < float(recovery["minimum_energy_lowering_kcal_mol"]):
        raise ValueError(
            "Imaginary-mode recovery did not reach a sufficiently lower "
            "stationary structure."
        )
    accepted = copy.deepcopy(chosen["optimization"])
    evidence = {
        "required": True,
        "status": "reoptimized-from-selected-imaginary-mode",
        "pre_recovery_optimization": copy.deepcopy(branch),
        "trigger_frequency_cm1": float(frequencies[mode_index]),
        "trigger_sorted_mode_index": mode_index,
        "maximum_atom_displacement_angstrom": amplitude,
        "trial_optimizations": trials,
        "selected_direction": int(chosen["direction"]),
        "energy_lowering_kcal_mol": energy_lowering,
        "minimum_required_energy_lowering_kcal_mol": float(
            recovery["minimum_energy_lowering_kcal_mol"]
        ),
    }
    return accepted, evidence


def _run_hessian_displacement_sensitivity(
    *,
    atoms: Atoms,
    source_calculator,
    protocol: dict[str, Any],
    compound_id: str,
    model_name: str,
    phase: str,
    minimum_index: int,
    minimum_dir: Path,
    reference_hessian: np.ndarray,
    reference_diagnostics: dict[str, float],
    reference_path: Path,
    reference_frequencies_cm1: np.ndarray,
    reference_selection: dict[str, Any],
) -> dict[str, Any]:
    specification = protocol["hessian_and_stationary_point"]
    sensitivity = specification["displacement_sensitivity"]
    required = _requires_hessian_displacement_sensitivity(
        protocol,
        compound_id=compound_id,
        model_name=model_name,
        phase=phase,
        minimum_index=minimum_index,
    )
    if not required:
        return {
            "required": False,
            "reason": "outside-preregistered-label-blind-preflight-subset",
        }

    reference_delta = float(sensitivity["reference_displacement_angstrom"])
    records = []
    reference_selected = np.asarray(
        reference_selection["thermochemistry_frequencies_cm1"],
        dtype=np.float64,
    )
    maximum_rms = 0.0
    for displacement in sensitivity["cartesian_displacements_angstrom"]:
        delta = float(displacement)
        if math.isclose(delta, reference_delta, rel_tol=0.0, abs_tol=0.0):
            hessian = reference_hessian
            diagnostics = reference_diagnostics
            hessian_path = reference_path
            frequencies = reference_frequencies_cm1
            selection = reference_selection
        else:
            hessian, diagnostics = cast(
                tuple[np.ndarray, dict[str, float]],
                numerical_hessian_from_atoms(
                    source_calculator,
                    atoms,
                    delta=delta,
                    return_diagnostics=True,
                ),
            )
            validate_hessian_numerical_quality(
                diagnostics,
                specification=specification,
            )
            hessian_path = minimum_dir / (
                f"hessian-delta-{delta:.4f}-hartree-per-angstrom2.npy"
            )
            _write_npy_atomic(hessian_path, hessian)
            primary = protocol["thermochemistry"]["primary"]
            frequency = MWFrequency(
                str(minimum_dir / f"frequency-delta-{delta:.4f}.out"),
                atoms,
                temperature=float(protocol["thermochemistry"]["temperature_kelvin"]),
                ilowfreq=int(primary["ilowfreq"]),
                omega0_cm1=float(primary["omega0_cm1"]),
                nu_floor_cm1=float(primary["nu_floor_cm1"]),
                device="cpu",
            )
            frequency.verbosity = 0
            frequencies, _ = frequency.compute_frequencies(hessian)
            selection = select_vibrational_modes(
                frequencies,
                atom_count=len(atoms),
                linear=frequency._is_linear_molecule(
                    tol=float(specification["linearity_moment_ratio_threshold"])
                ),
                imaginary_frequency_cutoff_cm1=float(
                    specification["imaginary_frequency_cutoff_cm1"]
                ),
                maximum_rigid_mode_leakage_cm1=float(
                    specification["maximum_rigid_mode_leakage_cm1"]
                ),
            )
        if not selection["valid_stationary_point"]:
            raise ValueError(
                "A Hessian displacement-sensitivity branch failed the stationary-"
                "point gate."
            )
        selected = np.asarray(
            selection["thermochemistry_frequencies_cm1"],
            dtype=np.float64,
        )
        rms = float(np.sqrt(np.mean(np.square(selected - reference_selected))))
        maximum_rms = max(maximum_rms, rms)
        records.append(
            {
                "cartesian_displacement_angstrom": delta,
                "hessian": {
                    **diagnostics,
                    "path": _relative_to_repository(hessian_path),
                    "sha256": sha256_file(hessian_path),
                    "shape": list(hessian.shape),
                },
                "all_projected_frequencies_cm1": np.asarray(
                    frequencies,
                    dtype=np.float64,
                ).tolist(),
                "vibrational_mode_selection": selection,
                "selected_mode_rms_difference_from_reference_cm1": rms,
            }
        )
    threshold = float(sensitivity["maximum_selected_mode_rms_difference_cm1"])
    if maximum_rms > threshold:
        raise ValueError(
            "The Hessian displacement-sensitivity selected-mode RMS gate failed."
        )
    return {
        "required": True,
        "reference_displacement_angstrom": reference_delta,
        "maximum_selected_mode_rms_difference_cm1": maximum_rms,
        "gate_threshold_cm1": threshold,
        "passed": True,
        "records": records,
    }


def _thermochemistry_variants(
    *,
    atoms: Atoms,
    thermochemistry_frequencies_cm1: np.ndarray,
    rotational_symmetry_number: int,
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    specification = protocol["thermochemistry"]
    variants = [
        {"id": "primary", **specification["primary"]},
        *specification["predeclared_sensitivity_variants"],
    ]
    results = {}
    for variant in variants:
        frequency = MWFrequency(
            str(output_dir / f"thermo-{variant['id']}.out"),
            atoms,
            temperature=float(specification["temperature_kelvin"]),
            symmetry_number=rotational_symmetry_number,
            ilowfreq=int(variant["ilowfreq"]),
            omega0_cm1=float(variant["omega0_cm1"]),
            nu_floor_cm1=float(
                variant.get(
                    "nu_floor_cm1",
                    specification["primary"]["nu_floor_cm1"],
                )
            ),
            device="cpu",
        )
        setattr(
            frequency,
            "alpha",
            int(variant.get("alpha", specification["primary"]["alpha"])),
        )
        frequency.verbosity = 0
        thermo = frequency.compute_internal_rotational_thermo(
            thermochemistry_frequencies_cm1
        )
        payload = asdict(thermo)
        payload["g_internal_rotational_kcal_mol"] = (
            thermo.g_correction_kjmol / KJ_PER_KCAL
        )
        results[variant["id"]] = {
            "definition": variant,
            "thermochemistry": payload,
        }
    return results


def _analyze_minimum(
    *,
    branch: dict[str, Any],
    template_atoms: Atoms,
    source_calculator,
    protocol: dict[str, Any],
    compound_id: str,
    model_name: str,
    rotational_symmetry_number: int,
    phase: str,
    minimum_index: int,
    audit_dir: Path,
) -> dict[str, Any]:
    atoms = template_atoms.copy()
    atoms.set_positions(branch["final_positions_angstrom"])
    atoms.calc = source_calculator
    specification = protocol["hessian_and_stationary_point"]
    minimum_dir = audit_dir / phase / f"minimum-{minimum_index:02d}"
    minimum_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        primary = _compute_hessian_analysis(
            atoms=atoms,
            source_calculator=source_calculator,
            protocol=protocol,
            delta=float(specification["cartesian_displacement_angstrom"]),
            hessian_path=minimum_dir / "hessian-hartree-per-angstrom2.npy",
            frequency_output=minimum_dir / "frequency-projection.out",
        )
        recovery_evidence: dict[str, Any] = {
            "required": False,
            "reason": "initial-optimization-is-a-valid-minimum",
        }
        if not primary["selection"]["valid_stationary_point"]:
            if not primary["selection"]["negative_selected_frequencies_cm1"]:
                raise ValueError(
                    "The projected rigid complement exceeds its leakage gate."
                )
            trigger_path = (
                minimum_dir
                / "hessian-before-stationary-point-recovery-hartree-per-angstrom2.npy"
            )
            os.replace(primary["path"], trigger_path)
            accepted, recovery_evidence = _recover_imaginary_mode_minimum(
                branch=branch,
                initial_analysis=primary,
                template_atoms=template_atoms,
                source_calculator=source_calculator,
                protocol=protocol,
                phase=phase,
                minimum_dir=minimum_dir,
            )
            recovery_evidence["trigger_hessian"] = {
                **primary["diagnostics"],
                "path": _relative_to_repository(trigger_path),
                "sha256": sha256_file(trigger_path),
                "shape": list(primary["hessian"].shape),
            }
            recovery_evidence["trigger_all_projected_frequencies_cm1"] = (
                primary["frequencies_cm1"].tolist()
            )
            recovery_evidence["trigger_vibrational_mode_selection"] = primary[
                "selection"
            ]
            branch.clear()
            branch.update(accepted)
            branch["stationary_point_recovery"] = {
                key: value
                for key, value in recovery_evidence.items()
                if not key.startswith("trigger_hessian")
                and not key.startswith("trigger_all_projected")
                and not key.startswith("trigger_vibrational")
            }
            atoms.set_positions(branch["final_positions_angstrom"])
            primary = _compute_hessian_analysis(
                atoms=atoms,
                source_calculator=source_calculator,
                protocol=protocol,
                delta=float(specification["cartesian_displacement_angstrom"]),
                hessian_path=minimum_dir / "hessian-hartree-per-angstrom2.npy",
                frequency_output=minimum_dir / "frequency-projection.out",
            )
        hessian = primary["hessian"]
        diagnostics = primary["diagnostics"]
        hessian_path = primary["path"]
        frequencies = primary["frequencies_cm1"]
        selection = primary["selection"]
        if not selection["valid_stationary_point"]:
            raise ValueError(
                "Selected vibrational subspace contains a negative mode or the "
                "projected rigid complement exceeds its leakage gate."
            )
        displacement_sensitivity = _run_hessian_displacement_sensitivity(
            atoms=atoms,
            source_calculator=source_calculator,
            protocol=protocol,
            compound_id=compound_id,
            model_name=model_name,
            phase=phase,
            minimum_index=minimum_index,
            minimum_dir=minimum_dir,
            reference_hessian=hessian,
            reference_diagnostics=diagnostics,
            reference_path=hessian_path,
            reference_frequencies_cm1=np.asarray(frequencies, dtype=np.float64),
            reference_selection=selection,
        )
        thermo_frequencies = np.asarray(
            selection["thermochemistry_frequencies_cm1"],
            dtype=np.float64,
        )
        variants = _thermochemistry_variants(
            atoms=atoms,
            thermochemistry_frequencies_cm1=thermo_frequencies,
            rotational_symmetry_number=rotational_symmetry_number,
            protocol=protocol,
            output_dir=minimum_dir,
        )
        return {
            "status": "valid",
            "source_state_index": int(branch["source_state_index"]),
            "elapsed_seconds": time.perf_counter() - start,
            "potential_energy": branch["final_energy"],
            "final_max_force_eV_per_angstrom": float(
                branch["final_max_force_eV_per_angstrom"]
            ),
            "final_positions_angstrom": branch["final_positions_angstrom"],
            "rotational_symmetry_number": rotational_symmetry_number,
            "hessian": {
                **diagnostics,
                "path": _relative_to_repository(hessian_path),
                "sha256": sha256_file(hessian_path),
                "shape": list(hessian.shape),
            },
            "all_projected_frequencies_cm1": np.asarray(
                frequencies,
                dtype=np.float64,
            ).tolist(),
            "vibrational_mode_selection": selection,
            "stationary_point_recovery": recovery_evidence,
            "hessian_displacement_sensitivity": displacement_sensitivity,
            "thermochemistry_variants": variants,
        }
    except Exception as exc:
        return {
            "status": "invalid",
            "source_state_index": int(branch["source_state_index"]),
            "elapsed_seconds": time.perf_counter() - start,
            "failure": {
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            },
        }


def _phase_ensemble(
    minima: list[dict[str, Any]],
    *,
    temperature_kelvin: float,
) -> dict[str, Any]:
    if not minima:
        raise ValueError("A phase has no valid unique qRRHO minimum.")
    variant_ids = list(minima[0]["thermochemistry_variants"])
    ensembles = {}
    per_minimum = {}
    for variant_id in variant_ids:
        values = []
        for minimum in minima:
            potential_kcal = (
                float(minimum["potential_energy"]["total_energy_hartree"])
                * KCAL_PER_HARTREE
            )
            correction = float(
                minimum["thermochemistry_variants"][variant_id]["thermochemistry"][
                    "g_internal_rotational_kcal_mol"
                ]
            )
            values.append(potential_kcal + correction)
        per_minimum[variant_id] = values
        ensembles[variant_id] = stable_ensemble_free_energy(
            values,
            temperature_kelvin=temperature_kelvin,
        )
    return {
        "valid_unique_minimum_count": len(minima),
        "per_minimum_free_energy_kcal_mol": per_minimum,
        "ensembles": ensembles,
    }


def _evaluate_state_energies(
    *,
    calculator,
    template_atoms: Atoms,
    positions: np.ndarray,
    batch_size: int,
    repeat_count: int,
    maximum_relative_difference_kcal_mol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    structures = []
    for state in positions:
        atoms = template_atoms.copy()
        atoms.set_positions(state)
        structures.append(atoms)
    repeats = []
    elapsed_seconds = []
    for _repeat in range(repeat_count):
        energies = []
        start = time.perf_counter()
        for begin in range(0, len(structures), batch_size):
            result = calculator.calculate_many(
                structures[begin : begin + batch_size],
                properties=("energy",),
            )
            if result.energies is None:
                raise RuntimeError("The MLIP batch path omitted requested energies.")
            energies.extend(np.asarray(result.energies, dtype=np.float64).tolist())
        elapsed_seconds.append(time.perf_counter() - start)
        output = np.asarray(energies, dtype=np.float64)
        if output.shape != (len(positions),) or not np.isfinite(output).all():
            raise ValueError(
                "The MLIP source-state energies are incomplete or non-finite."
            )
        repeats.append(output)
    relative = [(values - float(values.min())) * KCAL_PER_HARTREE for values in repeats]
    repeat_records = []
    maximum_repeat_difference = 0.0
    maximum_difference_location = {
        "repeat_index": 0,
        "state_index": 0,
        "signed_relative_energy_difference_kcal_mol": 0.0,
    }
    for repeat_index, (absolute_values, relative_values) in enumerate(
        zip(repeats, relative)
    ):
        repeat_records.append(
            {
                "repeat_index": repeat_index,
                "energy_hartree": absolute_values.tolist(),
                "energy_hartree_sha256": _float_list_sha256(absolute_values),
                "relative_energy_kcal_mol": relative_values.tolist(),
                "relative_energy_kcal_mol_sha256": _float_list_sha256(relative_values),
                "minimum_state_index": int(np.argmin(absolute_values)),
            }
        )
        if repeat_index:
            difference = relative_values - relative[0]
            state_index = int(np.argmax(np.abs(difference)))
            observed = float(abs(difference[state_index]))
            if observed > maximum_repeat_difference:
                maximum_repeat_difference = observed
                maximum_difference_location = {
                    "repeat_index": repeat_index,
                    "state_index": state_index,
                    "signed_relative_energy_difference_kcal_mol": float(
                        difference[state_index]
                    ),
                }
    if maximum_repeat_difference > maximum_relative_difference_kcal_mol:
        raise ValueError(
            "Repeated MLIP source-state relative energies differ by "
            f"{maximum_repeat_difference:.12g} kcal/mol, exceeding "
            f"{maximum_relative_difference_kcal_mol:.12g} kcal/mol."
        )
    return repeats[0], {
        "repeat_count": repeat_count,
        "elapsed_seconds": elapsed_seconds,
        "repeats": repeat_records,
        "maximum_repeat_relative_energy_difference_kcal_mol": (
            maximum_repeat_difference
        ),
        "maximum_difference_location": maximum_difference_location,
        "maximum_allowed_repeat_relative_energy_difference_kcal_mol": (
            maximum_relative_difference_kcal_mol
        ),
    }


def _attach_solution_correction(
    calculator,
    *,
    atoms: Atoms,
    charge_evidence: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    solvent = protocol["solvation"]
    calculator.solvent_correction = FrozenChargeOpenMMCorrection(
        atoms,
        np.asarray(atoms.get_initial_charges(), dtype=np.float64),
        charge_evidence=charge_evidence,
        model=solvent["polar_model"],
        nonpolar=solvent["nonpolar_model"],
        platform=solvent["platform"],
    )
    calculator.chargecalc = None


def _run_model_case(
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_manifest: dict[str, Any],
    model: dict[str, Any],
    protocol_case: dict[str, Any],
    calculator,
    environment: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    compound_id = protocol_case["compound_id"]
    rotational_symmetry_number = int(protocol_case["rotational_symmetry_number"])
    source_case = next(
        case for case in source_manifest["cases"] if case["compound_id"] == compound_id
    )
    case_dir = work_dir / "audit" / model["name"] / compound_id
    atoms, positions, solvent_kcal, charge_evidence = _load_case_sources(
        protocol_case=protocol_case,
        source_case=source_case,
    )
    calculator.solvent_correction = None
    gas_energy, energy_repeat_evidence = _evaluate_state_energies(
        calculator=calculator,
        template_atoms=atoms,
        positions=positions,
        batch_size=int(protocol["execution"]["batch_size"]),
        repeat_count=int(protocol["execution"]["source_energy_repeat_count"]),
        maximum_relative_difference_kcal_mol=float(
            protocol["execution"]["maximum_repeat_relative_energy_difference_kcal_mol"]
        ),
    )
    solution_potential = gas_energy + solvent_kcal / KCAL_PER_HARTREE
    symbols = list(atoms.get_chemical_symbols())
    sampling = protocol["phase_specific_sampling"]
    gas_seeds = select_diverse_seeds(
        positions_angstrom=positions,
        potential_hartree=gas_energy,
        symbols=symbols,
        maximum_count=int(sampling["maximum_seed_count_per_phase_model_case"]),
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )
    solution_seeds = select_diverse_seeds(
        positions_angstrom=positions,
        potential_hartree=solution_potential,
        symbols=symbols,
        maximum_count=int(sampling["maximum_seed_count_per_phase_model_case"]),
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )

    optimizer = protocol["optimization"]
    gas_branches = [
        _run_optimization(
            template_atoms=atoms,
            initial_positions=positions[seed["source_state_index"]],
            source_state_index=seed["source_state_index"],
            source_calculator=calculator,
            optimizer_options=optimizer,
            phase="gas",
            branch_dir=case_dir / f"state-{seed['source_state_index']:04d}",
        )
        for seed in gas_seeds
    ]
    gas_unique, gas_duplicates = deduplicate_optimized_minima(
        branches=gas_branches,
        symbols=symbols,
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )
    gas_accounting = validate_selected_branch_accounting(
        gas_branches,
        duplicate_minima=gas_duplicates,
        unique_minimum_count=len(gas_unique),
    )
    gas_minima = [
        _analyze_minimum(
            branch=branch,
            template_atoms=atoms,
            source_calculator=calculator,
            protocol=protocol,
            compound_id=compound_id,
            model_name=model["name"],
            rotational_symmetry_number=rotational_symmetry_number,
            phase="gas",
            minimum_index=index,
            audit_dir=case_dir,
        )
        for index, branch in enumerate(gas_unique)
    ]
    (
        gas_unique,
        gas_duplicates,
        gas_minima,
        gas_accounting,
    ) = reconcile_post_analysis_deduplication(
        branches=gas_branches,
        initially_unique=gas_unique,
        minimum_analyses=gas_minima,
        symbols=symbols,
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )

    _attach_solution_correction(
        calculator,
        atoms=atoms,
        charge_evidence=charge_evidence,
        protocol=protocol,
    )
    solution_branches = [
        _run_optimization(
            template_atoms=atoms,
            initial_positions=positions[seed["source_state_index"]],
            source_state_index=seed["source_state_index"],
            source_calculator=calculator,
            optimizer_options=optimizer,
            phase="solution",
            branch_dir=case_dir / f"state-{seed['source_state_index']:04d}",
        )
        for seed in solution_seeds
    ]
    solution_unique, solution_duplicates = deduplicate_optimized_minima(
        branches=solution_branches,
        symbols=symbols,
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )
    solution_accounting = validate_selected_branch_accounting(
        solution_branches,
        duplicate_minima=solution_duplicates,
        unique_minimum_count=len(solution_unique),
    )
    solution_minima = [
        _analyze_minimum(
            branch=branch,
            template_atoms=atoms,
            source_calculator=calculator,
            protocol=protocol,
            compound_id=compound_id,
            model_name=model["name"],
            rotational_symmetry_number=rotational_symmetry_number,
            phase="solution",
            minimum_index=index,
            audit_dir=case_dir,
        )
        for index, branch in enumerate(solution_unique)
    ]
    (
        solution_unique,
        solution_duplicates,
        solution_minima,
        solution_accounting,
    ) = reconcile_post_analysis_deduplication(
        branches=solution_branches,
        initially_unique=solution_unique,
        minimum_analyses=solution_minima,
        symbols=symbols,
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )
    calculator.solvent_correction = None

    valid_gas = [minimum for minimum in gas_minima if minimum["status"] == "valid"]
    valid_solution = [
        minimum for minimum in solution_minima if minimum["status"] == "valid"
    ]
    if len(valid_gas) != len(gas_minima) or len(valid_solution) != len(solution_minima):
        raise ValueError(
            "Every unique selected minimum must pass Hessian and stationary-point "
            "validation: "
            f"gas={len(valid_gas)}/{len(gas_minima)}, "
            f"solution={len(valid_solution)}/{len(solution_minima)}."
        )
    if not valid_gas or not valid_solution:
        raise ValueError("Each phase requires at least one valid unique minimum.")

    temperature = float(protocol["thermochemistry"]["temperature_kelvin"])
    gas_ensemble = _phase_ensemble(
        valid_gas,
        temperature_kelvin=temperature,
    )
    solution_ensemble = _phase_ensemble(
        valid_solution,
        temperature_kelvin=temperature,
    )
    variants = {}
    for variant_id in gas_ensemble["ensembles"]:
        gas_value = float(gas_ensemble["ensembles"][variant_id]["free_energy_kcal_mol"])
        solution_value = float(
            solution_ensemble["ensembles"][variant_id]["free_energy_kcal_mol"]
        )
        variants[variant_id] = {
            "gas_phase_ensemble_free_energy_kcal_mol": gas_value,
            "solution_phase_ensemble_free_energy_kcal_mol": solution_value,
            "hydration_prediction_kcal_mol": solution_value - gas_value,
        }
    zero_kelvin = (
        min(
            minimum["potential_energy"]["total_energy_hartree"]
            for minimum in valid_solution
        )
        - min(
            minimum["potential_energy"]["total_energy_hartree"] for minimum in valid_gas
        )
    ) * KCAL_PER_HARTREE

    record = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-phase-specific-qrrho-model-case",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "status": "success",
        "model": model,
        "compound_id": compound_id,
        "name": protocol_case["name"],
        "flexibility_bin": protocol_case["flexibility_bin"],
        "environment": environment,
        "source_evidence": {
            "source_manifest_content_sha256": source_manifest["content_sha256"],
            "state_file": protocol_case["state_file"],
            "state_file_sha256": protocol_case["state_file_sha256"],
            "source_state_count": len(positions),
            "charge": charge_evidence,
        },
        "source_state_evaluation": {
            "repeat_evidence": energy_repeat_evidence,
            "gas_energy_hartree": gas_energy.tolist(),
            "gas_energy_sha256": _float_list_sha256(gas_energy),
            "frozen_solvent_correction_kcal_mol": solvent_kcal.tolist(),
            "frozen_solvent_correction_sha256": _float_list_sha256(solvent_kcal),
            "solution_potential_hartree": solution_potential.tolist(),
            "solution_potential_sha256": _float_list_sha256(solution_potential),
        },
        "selected_phase_seeds": {
            "gas": gas_seeds,
            "solution": solution_seeds,
        },
        "phases": {
            "gas": {
                "optimization_branches": gas_branches,
                "duplicate_minima": gas_duplicates,
                "minimum_analyses": gas_minima,
                "branch_accounting": gas_accounting,
                "ensemble": gas_ensemble,
            },
            "solution": {
                "optimization_branches": solution_branches,
                "duplicate_minima": solution_duplicates,
                "minimum_analyses": solution_minima,
                "branch_accounting": solution_accounting,
                "ensemble": solution_ensemble,
            },
        },
        "prediction": {
            "variants": variants,
            "primary_hydration_prediction_kcal_mol": variants["primary"][
                "hydration_prediction_kcal_mol"
            ],
            "zero_kelvin_separately_optimized_minimum_kcal_mol": zero_kelvin,
            "single_reference_endpoint_baseline_kcal_mol": float(
                source_case["reference_endpoint_kcal_mol"]
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    if _contains_forbidden_label_key(record):
        raise ValueError("The qRRHO energy record contains a forbidden label key.")
    return seal_artifact(record)


def _finite_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


def _assert_equivalent(
    observed: Any,
    expected: Any,
    *,
    name: str,
    absolute_tolerance: float = 1.0e-10,
) -> None:
    """Compare deterministic scientific payloads with finite-float tolerance."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(f"{name} has different fields from its derivation.")
        for key in expected:
            _assert_equivalent(
                observed[key],
                expected[key],
                name=f"{name}.{key}",
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"{name} has a different list shape from its derivation.")
        for index, (observed_item, expected_item) in enumerate(zip(observed, expected)):
            _assert_equivalent(
                observed_item,
                expected_item,
                name=f"{name}[{index}]",
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise ValueError(f"{name} differs from its deterministic derivation.")
        return
    if isinstance(expected, (int, float, np.number)):
        if not isinstance(observed, (int, float, np.number)) or isinstance(
            observed, bool
        ):
            raise ValueError(f"{name} must be numeric.")
        observed_float = float(observed)
        expected_float = float(expected)
        if (
            not math.isfinite(observed_float)
            or not math.isfinite(expected_float)
            or not math.isclose(
                observed_float,
                expected_float,
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            )
        ):
            raise ValueError(f"{name} differs from its deterministic derivation.")
        return
    if observed != expected:
        raise ValueError(f"{name} differs from its deterministic derivation.")


def _validate_energy_result(
    value: Any,
    *,
    phase: str,
    name: str,
    protocol: dict[str, Any],
    charge_evidence: dict[str, Any],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a structured phase-energy record.")
    required = {
        "total_energy_hartree",
        "gas_energy_hartree",
        "solvent_energy_hartree",
        "solvent_components_hartree",
        "solvent_provenance",
    }
    if (
        set(value) != required
        or not isinstance(value["solvent_components_hartree"], dict)
        or not isinstance(value["solvent_provenance"], dict)
    ):
        raise ValueError(f"{name} has an invalid phase-energy schema.")
    scalars = [
        float(value["total_energy_hartree"]),
        float(value["gas_energy_hartree"]),
        float(value["solvent_energy_hartree"]),
        *[
            float(component)
            for component in value["solvent_components_hartree"].values()
        ],
    ]
    if not all(math.isfinite(item) for item in scalars):
        raise ValueError(f"{name} contains a non-finite energy.")
    if not math.isclose(
        scalars[0],
        scalars[1] + scalars[2],
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ) or not math.isclose(
        scalars[2],
        sum(scalars[3:]),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError(f"{name} does not close its gas/solvent decomposition.")
    if phase == "gas" and (
        abs(scalars[2]) > 1.0e-12
        or value["solvent_components_hartree"]
        or value["solvent_provenance"]
    ):
        raise ValueError(f"{name} contains a solvent term in the gas phase.")
    if phase == "solution":
        if set(value["solvent_components_hartree"]) != {"polar", "nonpolar"}:
            raise ValueError(
                f"{name} must contain exact OBC-II polar and ACE nonpolar terms."
            )
        provenance = value["solvent_provenance"]
        expected_provenance = {
            "provider": "openmm",
            "provider_version": protocol["implementation_freeze"]["required_versions"][
                "openmm"
            ],
            "method": protocol["solvation"]["method"],
            "platform": protocol["solvation"]["platform"],
            "model": protocol["solvation"]["polar_model"],
            "profile": protocol["solvation"]["profile"],
            "nonpolar": protocol["solvation"]["nonpolar_model"],
            "solvent": protocol["solvation"]["solvent"],
            "charge_provider": {
                "source": protocol["solvation"]["charge_source"],
                **charge_evidence,
            },
        }
        for key, expected in expected_provenance.items():
            _assert_equivalent(
                provenance.get(key),
                expected,
                name=f"{name} solvent provenance {key}",
            )


def _validate_repeat_evidence(
    evidence: Any,
    *,
    first_repeat_hartree: np.ndarray,
    protocol: dict[str, Any],
) -> None:
    execution = protocol["execution"]
    expected_count = int(execution["source_energy_repeat_count"])
    tolerance = float(execution["maximum_repeat_relative_energy_difference_kcal_mol"])
    if not isinstance(evidence, dict):
        raise ValueError("Source-energy repeat evidence is missing.")
    repeats = evidence.get("repeats")
    elapsed = evidence.get("elapsed_seconds")
    if (
        int(evidence.get("repeat_count", -1)) != expected_count
        or not isinstance(repeats, list)
        or len(repeats) != expected_count
        or not isinstance(elapsed, list)
        or len(elapsed) != expected_count
        or any(
            not math.isfinite(float(value)) or float(value) < 0.0 for value in elapsed
        )
    ):
        raise ValueError("Source-energy repeat evidence is incomplete.")

    relative_arrays = []
    for repeat_index, repeat in enumerate(repeats):
        absolute = _finite_array(
            repeat.get("energy_hartree"),
            shape=first_repeat_hartree.shape,
            name=f"repeat {repeat_index} absolute energies",
        )
        relative = (absolute - float(absolute.min())) * KCAL_PER_HARTREE
        recorded_relative = _finite_array(
            repeat.get("relative_energy_kcal_mol"),
            shape=first_repeat_hartree.shape,
            name=f"repeat {repeat_index} relative energies",
        )
        if (
            int(repeat.get("repeat_index", -1)) != repeat_index
            or repeat.get("energy_hartree_sha256") != _float_list_sha256(absolute)
            or repeat.get("relative_energy_kcal_mol_sha256")
            != _float_list_sha256(recorded_relative)
            or int(repeat.get("minimum_state_index", -1)) != int(np.argmin(absolute))
            or not np.allclose(
                recorded_relative,
                relative,
                rtol=0.0,
                atol=1.0e-10,
            )
        ):
            raise ValueError(f"Source-energy repeat {repeat_index} is inconsistent.")
        if repeat_index == 0 and not np.array_equal(absolute, first_repeat_hartree):
            raise ValueError("The retained gas energies are not repeat zero.")
        relative_arrays.append(relative)

    maximum = 0.0
    location = {
        "repeat_index": 0,
        "state_index": 0,
        "signed_relative_energy_difference_kcal_mol": 0.0,
    }
    for repeat_index, relative in enumerate(relative_arrays[1:], start=1):
        difference = relative - relative_arrays[0]
        state_index = int(np.argmax(np.abs(difference)))
        observed = float(abs(difference[state_index]))
        if observed > maximum:
            maximum = observed
            location = {
                "repeat_index": repeat_index,
                "state_index": state_index,
                "signed_relative_energy_difference_kcal_mol": float(
                    difference[state_index]
                ),
            }
    if not math.isclose(
        float(
            evidence.get(
                "maximum_repeat_relative_energy_difference_kcal_mol",
                math.nan,
            )
        ),
        maximum,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        float(
            evidence.get(
                "maximum_allowed_repeat_relative_energy_difference_kcal_mol",
                math.nan,
            )
        ),
        tolerance,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("The source-energy repeat gate was not derived correctly.")
    _assert_equivalent(
        evidence.get("maximum_difference_location"),
        location,
        name="source-energy maximum-difference location",
        absolute_tolerance=1.0e-12,
    )
    if maximum > tolerance:
        raise ValueError("The source-energy repeat gate failed.")


def _validate_optimization_branch(
    branch: Any,
    *,
    phase: str,
    source_state_index: int,
    atom_count: int,
    protocol: dict[str, Any],
    charge_evidence: dict[str, Any],
) -> None:
    if not isinstance(branch, dict):
        raise ValueError("An optimization branch is not structured.")
    if (
        branch.get("phase") != phase
        or int(branch.get("source_state_index", -1)) != source_state_index
    ):
        raise ValueError("An optimization branch changed phase or source seed.")
    elapsed = float(branch.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("An optimization branch has invalid elapsed time.")
    if branch.get("status") == "failure":
        failure = branch.get("failure")
        if not isinstance(failure, dict) or not failure.get("exception_class"):
            raise ValueError("A failed optimization branch lacks audit evidence.")
        return
    if branch.get("status") != "success":
        raise ValueError("An optimization branch has an unknown status.")
    positions = _finite_array(
        branch.get("final_positions_angstrom"),
        shape=(atom_count, 3),
        name="optimized positions",
    )
    del positions
    _validate_energy_result(
        branch.get("initial_energy"),
        phase=phase,
        name="initial phase energy",
        protocol=protocol,
        charge_evidence=charge_evidence,
    )
    _validate_energy_result(
        branch.get("final_energy"),
        phase=phase,
        name="final phase energy",
        protocol=protocol,
        charge_evidence=charge_evidence,
    )
    maximum_steps = int(protocol["optimization"]["max_steps"])
    steps = int(branch.get("steps", -1))
    initial_force = float(branch.get("initial_max_force_eV_per_angstrom", math.nan))
    final_force = float(branch.get("final_max_force_eV_per_angstrom", math.nan))
    if (
        steps < 0
        or steps > maximum_steps
        or not math.isfinite(initial_force)
        or initial_force < 0.0
        or not math.isfinite(final_force)
        or final_force < 0.0
    ):
        raise ValueError("An optimization branch has invalid convergence evidence.")
    reported = branch.get("optimizer_reported_converged")
    converged = branch.get("converged")
    if not isinstance(reported, bool) or not isinstance(converged, bool):
        raise ValueError("Optimization convergence flags must be Boolean.")
    expected_converged = bool(
        reported
        and final_force
        <= float(protocol["optimization"]["fmax_eV_per_angstrom"]) + 1.0e-12
    )
    if converged is not expected_converged:
        raise ValueError("Optimization convergence does not satisfy the force gate.")


def _validate_hessian_displacement_sensitivity(
    observed: Any,
    *,
    atoms: Atoms,
    primary_hessian_record: dict[str, Any],
    primary_frequencies_cm1: np.ndarray,
    primary_selection: dict[str, Any],
    protocol: dict[str, Any],
    compound_id: str,
    model_name: str,
    phase: str,
    minimum_index: int,
    evidence_root: Path,
) -> None:
    required = _requires_hessian_displacement_sensitivity(
        protocol,
        compound_id=compound_id,
        model_name=model_name,
        phase=phase,
        minimum_index=minimum_index,
    )
    if not required:
        _assert_equivalent(
            observed,
            {
                "required": False,
                "reason": "outside-preregistered-label-blind-preflight-subset",
            },
            name="Hessian displacement sensitivity",
        )
        return
    if not isinstance(observed, dict) or observed.get("required") is not True:
        raise ValueError("Required Hessian displacement sensitivity is missing.")
    specification = protocol["hessian_and_stationary_point"]
    sensitivity = specification["displacement_sensitivity"]
    expected_deltas = [
        float(value) for value in sensitivity["cartesian_displacements_angstrom"]
    ]
    records = observed.get("records")
    if not isinstance(records, list) or len(records) != len(expected_deltas):
        raise ValueError("Hessian displacement-sensitivity record count changed.")
    reference_delta = float(sensitivity["reference_displacement_angstrom"])
    selected_by_delta: dict[float, np.ndarray] = {}
    reported_rms: dict[float, float] = {}
    for record, delta in zip(records, expected_deltas):
        if not math.isclose(
            float(record.get("cartesian_displacement_angstrom", math.nan)),
            delta,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("A Hessian sensitivity displacement changed.")
        hessian_record = record.get("hessian")
        if not isinstance(hessian_record, dict):
            raise ValueError("A Hessian sensitivity branch lacks Hessian evidence.")
        if math.isclose(delta, reference_delta, rel_tol=0.0, abs_tol=0.0):
            _assert_equivalent(
                hessian_record,
                primary_hessian_record,
                name="Hessian sensitivity primary-reference anchor",
            )
        hessian_path_value = hessian_record.get("path")
        if not isinstance(hessian_path_value, str):
            raise ValueError(
                "A Hessian sensitivity branch has an invalid evidence path."
            )
        path = _resolve_repository_file(
            hessian_path_value,
            name="Hessian sensitivity evidence path",
        )
        try:
            path.relative_to(evidence_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "A Hessian sensitivity file escaped the sealed evidence root."
            ) from exc
        if sha256_file(path) != hessian_record.get("sha256"):
            raise ValueError("A Hessian sensitivity evidence file changed.")
        hessian = np.load(path, allow_pickle=False)
        expected_shape = (3 * len(atoms), 3 * len(atoms))
        if (
            hessian.shape != expected_shape
            or list(hessian.shape) != hessian_record.get("shape")
            or not np.isfinite(hessian).all()
            or not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError("A Hessian sensitivity matrix is invalid.")
        recorded_delta = float(
            hessian_record.get("cartesian_displacement_angstrom", math.nan)
        )
        if not math.isclose(
            recorded_delta,
            delta,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("A Hessian sensitivity displacement changed.")
        validate_hessian_numerical_quality(
            hessian_record,
            specification=specification,
        )
        primary = protocol["thermochemistry"]["primary"]
        frequency = MWFrequency(
            str(path.with_suffix(".sensitivity-validation.out")),
            atoms,
            temperature=float(protocol["thermochemistry"]["temperature_kelvin"]),
            ilowfreq=int(primary["ilowfreq"]),
            omega0_cm1=float(primary["omega0_cm1"]),
            nu_floor_cm1=float(primary["nu_floor_cm1"]),
            device="cpu",
        )
        frequency.verbosity = 0
        expected_frequencies, _ = frequency.compute_frequencies(hessian)
        observed_frequencies = _finite_array(
            record.get("all_projected_frequencies_cm1"),
            shape=(3 * len(atoms),),
            name="Hessian sensitivity projected frequencies",
        )
        if not np.allclose(
            observed_frequencies,
            expected_frequencies,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("Hessian sensitivity frequencies changed.")
        if math.isclose(delta, reference_delta, rel_tol=0.0, abs_tol=0.0):
            if not np.allclose(
                observed_frequencies,
                primary_frequencies_cm1,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(
                    "The Hessian sensitivity reference frequencies do not match "
                    "the primary Hessian."
                )
        expected_selection = select_vibrational_modes(
            expected_frequencies,
            atom_count=len(atoms),
            linear=frequency._is_linear_molecule(
                tol=float(specification["linearity_moment_ratio_threshold"])
            ),
            imaginary_frequency_cutoff_cm1=float(
                specification["imaginary_frequency_cutoff_cm1"]
            ),
            maximum_rigid_mode_leakage_cm1=float(
                specification["maximum_rigid_mode_leakage_cm1"]
            ),
        )
        _assert_equivalent(
            record.get("vibrational_mode_selection"),
            expected_selection,
            name="Hessian sensitivity mode selection",
            absolute_tolerance=1.0e-8,
        )
        if math.isclose(delta, reference_delta, rel_tol=0.0, abs_tol=0.0):
            _assert_equivalent(
                expected_selection,
                primary_selection,
                name="Hessian sensitivity primary-reference mode selection",
                absolute_tolerance=1.0e-8,
            )
        if not expected_selection["valid_stationary_point"]:
            raise ValueError("A Hessian sensitivity stationary-point gate failed.")
        selected_by_delta[delta] = np.asarray(
            expected_selection["thermochemistry_frequencies_cm1"],
            dtype=np.float64,
        )
        reported_rms[delta] = float(
            record.get(
                "selected_mode_rms_difference_from_reference_cm1",
                math.nan,
            )
        )
    reference = selected_by_delta[reference_delta]
    derived_rms = {
        delta: float(np.sqrt(np.mean(np.square(values - reference))))
        for delta, values in selected_by_delta.items()
    }
    for delta in expected_deltas:
        if not math.isclose(
            reported_rms[delta],
            derived_rms[delta],
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError("A Hessian sensitivity RMS diagnostic changed.")
    maximum = max(derived_rms.values())
    threshold = float(sensitivity["maximum_selected_mode_rms_difference_cm1"])
    expected_summary = {
        "reference_displacement_angstrom": reference_delta,
        "maximum_selected_mode_rms_difference_cm1": maximum,
        "gate_threshold_cm1": threshold,
        "passed": maximum <= threshold,
    }
    for key, value in expected_summary.items():
        _assert_equivalent(
            observed.get(key),
            value,
            name=f"Hessian displacement sensitivity {key}",
            absolute_tolerance=1.0e-8,
        )
    if not expected_summary["passed"]:
        raise ValueError("The Hessian displacement-sensitivity RMS gate failed.")


def _validate_stationary_point_recovery(
    observed: Any,
    *,
    branch: dict[str, Any],
    template_atoms: Atoms,
    protocol: dict[str, Any],
    evidence_root: Path,
) -> None:
    branch_summary = branch.get("stationary_point_recovery")
    if branch_summary is None:
        _assert_equivalent(
            observed,
            {
                "required": False,
                "reason": "initial-optimization-is-a-valid-minimum",
            },
            name="stationary-point recovery",
        )
        return
    if not isinstance(observed, dict):
        raise ValueError("Stationary-point recovery evidence is missing.")
    summary = {
        key: value
        for key, value in observed.items()
        if not key.startswith("trigger_hessian")
        and not key.startswith("trigger_all_projected")
        and not key.startswith("trigger_vibrational")
    }
    _assert_equivalent(
        branch_summary,
        summary,
        name="optimization/minimum stationary-point recovery",
        absolute_tolerance=1.0e-10,
    )
    recovery = protocol["hessian_and_stationary_point"][
        "imaginary_mode_recovery"
    ]
    if (
        observed.get("required") is not True
        or observed.get("status")
        != "reoptimized-from-selected-imaginary-mode"
        or not math.isclose(
            float(observed.get("maximum_atom_displacement_angstrom", math.nan)),
            float(recovery["maximum_atom_displacement_angstrom"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("Stationary-point recovery changed its frozen policy.")
    trials = observed.get("trial_optimizations")
    if (
        not isinstance(trials, list)
        or [trial.get("direction") for trial in trials] != [-1, 1]
    ):
        raise ValueError("Stationary-point recovery trial directions changed.")
    chosen_direction = int(observed.get("selected_direction", 0))
    if chosen_direction not in (-1, 1):
        raise ValueError("Stationary-point recovery selected an invalid direction.")
    chosen = next(
        trial["optimization"]
        for trial in trials
        if int(trial["direction"]) == chosen_direction
    )
    branch_without_recovery = {
        key: value
        for key, value in branch.items()
        if key != "stationary_point_recovery"
    }
    _assert_equivalent(
        branch_without_recovery,
        chosen,
        name="stationary-point accepted optimization",
        absolute_tolerance=1.0e-10,
    )
    pre_recovery = observed.get("pre_recovery_optimization")
    if not isinstance(pre_recovery, dict):
        raise ValueError("Stationary-point recovery lost its original optimization.")
    energy_lowering = (
        float(pre_recovery["final_energy"]["total_energy_hartree"])
        - float(chosen["final_energy"]["total_energy_hartree"])
    ) * KCAL_PER_HARTREE
    if not math.isclose(
        float(observed.get("energy_lowering_kcal_mol", math.nan)),
        energy_lowering,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ) or energy_lowering < float(recovery["minimum_energy_lowering_kcal_mol"]):
        raise ValueError("Stationary-point recovery energy lowering changed.")

    hessian_record = observed.get("trigger_hessian")
    if not isinstance(hessian_record, dict):
        raise ValueError("Stationary-point recovery lacks its trigger Hessian.")
    hessian_path_value = hessian_record.get("path")
    if not isinstance(hessian_path_value, str):
        raise ValueError("Stationary-point trigger Hessian path is invalid.")
    hessian_path = _resolve_repository_file(
        hessian_path_value,
        name="Stationary-point trigger Hessian",
    )
    try:
        hessian_path.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "The stationary-point trigger Hessian escaped its evidence root."
        ) from exc
    if sha256_file(hessian_path) != hessian_record.get("sha256"):
        raise ValueError("The stationary-point trigger Hessian changed.")
    hessian = np.load(hessian_path, allow_pickle=False)
    expected_shape = (3 * len(template_atoms), 3 * len(template_atoms))
    if (
        hessian.shape != expected_shape
        or list(hessian.shape) != hessian_record.get("shape")
        or not np.isfinite(hessian).all()
        or not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("The stationary-point trigger Hessian is invalid.")
    validate_hessian_numerical_quality(
        hessian_record,
        specification=protocol["hessian_and_stationary_point"],
    )
    atoms = template_atoms.copy()
    atoms.set_positions(pre_recovery["final_positions_angstrom"])
    primary = protocol["thermochemistry"]["primary"]
    frequency = MWFrequency(
        str(hessian_path.with_suffix(".recovery-validation.out")),
        atoms,
        temperature=float(protocol["thermochemistry"]["temperature_kelvin"]),
        ilowfreq=int(primary["ilowfreq"]),
        omega0_cm1=float(primary["omega0_cm1"]),
        nu_floor_cm1=float(primary["nu_floor_cm1"]),
        device="cpu",
    )
    frequency.verbosity = 0
    expected_frequencies, _ = frequency.compute_frequencies(hessian)
    observed_frequencies = _finite_array(
        observed.get("trigger_all_projected_frequencies_cm1"),
        shape=(3 * len(atoms),),
        name="stationary-point trigger frequencies",
    )
    if not np.allclose(
        observed_frequencies,
        expected_frequencies,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("Stationary-point trigger frequencies changed.")
    specification = protocol["hessian_and_stationary_point"]
    expected_selection = select_vibrational_modes(
        expected_frequencies,
        atom_count=len(atoms),
        linear=frequency._is_linear_molecule(
            tol=float(specification["linearity_moment_ratio_threshold"])
        ),
        imaginary_frequency_cutoff_cm1=float(
            specification["imaginary_frequency_cutoff_cm1"]
        ),
        maximum_rigid_mode_leakage_cm1=float(
            specification["maximum_rigid_mode_leakage_cm1"]
        ),
    )
    _assert_equivalent(
        observed.get("trigger_vibrational_mode_selection"),
        expected_selection,
        name="stationary-point trigger mode selection",
        absolute_tolerance=1.0e-8,
    )
    trigger_index = int(observed.get("trigger_sorted_mode_index", -1))
    if (
        trigger_index not in expected_selection["selected_indices"]
        or float(expected_frequencies[trigger_index]) >= 0.0
        or not math.isclose(
            float(observed.get("trigger_frequency_cm1", math.nan)),
            float(expected_frequencies[trigger_index]),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        )
    ):
        raise ValueError("Stationary-point recovery trigger mode changed.")


def _validate_minimum_analysis(
    minimum: Any,
    *,
    branch: dict[str, Any],
    template_atoms: Atoms,
    phase: str,
    protocol: dict[str, Any],
    compound_id: str,
    model_name: str,
    minimum_index: int,
    rotational_symmetry_number: int,
    evidence_root: Path,
) -> None:
    if not isinstance(minimum, dict):
        raise ValueError("A minimum analysis is not structured.")
    if int(minimum.get("source_state_index", -1)) != int(branch["source_state_index"]):
        raise ValueError("A minimum analysis changed its source branch.")
    elapsed = float(minimum.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("A minimum analysis has invalid elapsed time.")
    if minimum.get("status") == "invalid":
        failure = minimum.get("failure")
        if not isinstance(failure, dict) or not failure.get("exception_class"):
            raise ValueError("An invalid minimum lacks failure evidence.")
        return
    if minimum.get("status") != "valid":
        raise ValueError("A minimum analysis has an unknown status.")
    if int(minimum.get("rotational_symmetry_number", 0)) != (
        rotational_symmetry_number
    ):
        raise ValueError("A minimum changed the frozen rotational symmetry number.")

    atom_count = len(template_atoms)
    final_positions = _finite_array(
        minimum.get("final_positions_angstrom"),
        shape=(atom_count, 3),
        name="minimum positions",
    )
    if not np.array_equal(
        final_positions,
        np.asarray(branch["final_positions_angstrom"], dtype=np.float64),
    ):
        raise ValueError("A minimum does not match its optimized branch.")
    _assert_equivalent(
        minimum.get("potential_energy"),
        branch["final_energy"],
        name="minimum potential energy",
    )
    if not math.isclose(
        float(minimum.get("final_max_force_eV_per_angstrom", math.nan)),
        float(branch["final_max_force_eV_per_angstrom"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("A minimum changed the optimizer force evidence.")

    hessian_record = minimum.get("hessian")
    if not isinstance(hessian_record, dict):
        raise ValueError("A valid minimum lacks Hessian evidence.")
    hessian_path_value = hessian_record.get("path")
    if not isinstance(hessian_path_value, str):
        raise ValueError("A valid minimum has an invalid Hessian path.")
    hessian_path = _resolve_repository_file(
        hessian_path_value,
        name="Hessian evidence path",
    )
    try:
        hessian_path.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise ValueError("A Hessian escaped the sealed evidence root.") from exc
    if sha256_file(hessian_path) != hessian_record.get("sha256"):
        raise ValueError("A Hessian evidence file changed.")
    hessian = np.load(hessian_path, allow_pickle=False)
    expected_shape = (3 * atom_count, 3 * atom_count)
    if (
        hessian.shape != expected_shape
        or not np.isfinite(hessian).all()
        or list(hessian.shape) != hessian_record.get("shape")
        or not np.allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("A Hessian has invalid shape, values, or symmetry.")
    specification = protocol["hessian_and_stationary_point"]
    if not math.isclose(
        float(hessian_record.get("cartesian_displacement_angstrom", math.nan)),
        float(specification["cartesian_displacement_angstrom"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("The Hessian displacement changed.")
    validate_hessian_numerical_quality(
        hessian_record,
        specification=specification,
    )

    atoms = template_atoms.copy()
    atoms.set_positions(final_positions)
    primary = protocol["thermochemistry"]["primary"]
    frequency = MWFrequency(
        str(hessian_path.with_suffix(".validation.out")),
        atoms,
        temperature=float(protocol["thermochemistry"]["temperature_kelvin"]),
        ilowfreq=int(primary["ilowfreq"]),
        omega0_cm1=float(primary["omega0_cm1"]),
        nu_floor_cm1=float(primary["nu_floor_cm1"]),
        device="cpu",
    )
    frequency.verbosity = 0
    expected_frequencies, _ = frequency.compute_frequencies(hessian)
    observed_frequencies = _finite_array(
        minimum.get("all_projected_frequencies_cm1"),
        shape=(3 * atom_count,),
        name="projected frequencies",
    )
    if not np.allclose(
        observed_frequencies,
        expected_frequencies,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError("Projected frequencies do not match the sealed Hessian.")
    expected_selection = select_vibrational_modes(
        expected_frequencies,
        atom_count=atom_count,
        linear=frequency._is_linear_molecule(
            tol=float(specification["linearity_moment_ratio_threshold"])
        ),
        imaginary_frequency_cutoff_cm1=float(
            specification["imaginary_frequency_cutoff_cm1"]
        ),
        maximum_rigid_mode_leakage_cm1=float(
            specification["maximum_rigid_mode_leakage_cm1"]
        ),
    )
    _assert_equivalent(
        minimum.get("vibrational_mode_selection"),
        expected_selection,
        name="vibrational-mode selection",
        absolute_tolerance=1.0e-8,
    )
    if not expected_selection["valid_stationary_point"]:
        raise ValueError(
            "A valid minimum contains a selected negative mode or excessive rigid "
            "mode leakage."
        )
    _validate_stationary_point_recovery(
        minimum.get("stationary_point_recovery"),
        branch=branch,
        template_atoms=template_atoms,
        protocol=protocol,
        evidence_root=evidence_root,
    )
    _validate_hessian_displacement_sensitivity(
        minimum.get("hessian_displacement_sensitivity"),
        atoms=atoms,
        primary_hessian_record=hessian_record,
        primary_frequencies_cm1=expected_frequencies,
        primary_selection=expected_selection,
        protocol=protocol,
        compound_id=compound_id,
        model_name=model_name,
        phase=phase,
        minimum_index=minimum_index,
        evidence_root=evidence_root,
    )
    expected_variants = _thermochemistry_variants(
        atoms=atoms,
        thermochemistry_frequencies_cm1=np.asarray(
            expected_selection["thermochemistry_frequencies_cm1"],
            dtype=np.float64,
        ),
        rotational_symmetry_number=rotational_symmetry_number,
        protocol=protocol,
        output_dir=evidence_root,
    )
    _assert_equivalent(
        minimum.get("thermochemistry_variants"),
        expected_variants,
        name="minimum thermochemistry",
        absolute_tolerance=1.0e-8,
    )


def _validate_phase_record(
    phase_record: Any,
    *,
    selected_seeds: list[dict[str, Any]],
    template_atoms: Atoms,
    phase: str,
    protocol: dict[str, Any],
    compound_id: str,
    model_name: str,
    rotational_symmetry_number: int,
    evidence_root: Path,
    charge_evidence: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(phase_record, dict) or set(phase_record) != {
        "optimization_branches",
        "duplicate_minima",
        "minimum_analyses",
        "branch_accounting",
        "ensemble",
    }:
        raise ValueError(f"The {phase} phase record is incomplete.")
    branches = phase_record["optimization_branches"]
    if not isinstance(branches, list) or len(branches) != len(selected_seeds):
        raise ValueError(f"The {phase} optimization branch count changed.")
    for branch, seed in zip(branches, selected_seeds):
        _validate_optimization_branch(
            branch,
            phase=phase,
            source_state_index=int(seed["source_state_index"]),
            atom_count=len(template_atoms),
            protocol=protocol,
            charge_evidence=charge_evidence,
        )
    accepted, expected_duplicates = deduplicate_optimized_minima(
        branches=branches,
        symbols=list(template_atoms.get_chemical_symbols()),
        threshold_angstrom=float(
            protocol["optimized_minimum_deduplication"][
                "heavy_atom_rmsd_threshold_angstrom"
            ]
        ),
    )
    _assert_equivalent(
        phase_record["duplicate_minima"],
        expected_duplicates,
        name=f"{phase} duplicate-minimum evidence",
        absolute_tolerance=1.0e-10,
    )
    expected_accounting = validate_selected_branch_accounting(
        branches,
        duplicate_minima=expected_duplicates,
        unique_minimum_count=len(accepted),
    )
    _assert_equivalent(
        phase_record["branch_accounting"],
        expected_accounting,
        name=f"{phase} selected-branch accounting",
    )
    analyses = phase_record["minimum_analyses"]
    if not isinstance(analyses, list) or len(analyses) != len(accepted):
        raise ValueError(f"The {phase} minimum-analysis count changed.")
    for minimum_index, (minimum, branch) in enumerate(zip(analyses, accepted)):
        _validate_minimum_analysis(
            minimum,
            branch=branch,
            template_atoms=template_atoms,
            phase=phase,
            protocol=protocol,
            compound_id=compound_id,
            model_name=model_name,
            minimum_index=minimum_index,
            rotational_symmetry_number=rotational_symmetry_number,
            evidence_root=evidence_root,
        )
    valid = [minimum for minimum in analyses if minimum.get("status") == "valid"]
    if len(valid) != len(analyses) or not valid:
        raise ValueError(
            f"The {phase} phase must retain every unique minimum as valid."
        )
    expected_ensemble = _phase_ensemble(
        valid,
        temperature_kelvin=float(protocol["thermochemistry"]["temperature_kelvin"]),
    )
    _assert_equivalent(
        phase_record["ensemble"],
        expected_ensemble,
        name=f"{phase} ensemble",
        absolute_tolerance=1.0e-8,
    )
    return expected_ensemble


def validate_model_case_record(
    record: Any,
    *,
    protocol: dict[str, Any],
    fingerprint: str,
    source_manifest: dict[str, Any],
    model: dict[str, Any],
    protocol_case: dict[str, Any],
    evidence_root: Path,
    require_durable_evidence: bool = False,
) -> None:
    """Validate a complete scientific record before resume, seal, or scoring."""

    if not isinstance(record, dict):
        raise ValueError("A qRRHO model-case record must be a mapping.")
    _validate_self_hash(record, name="qRRHO model-case record")
    if _contains_forbidden_label_key(record):
        raise ValueError("A qRRHO model-case record crosses the label boundary.")
    required_identity = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-phase-specific-qrrho-model-case",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "status": "success",
        "model": model,
        "compound_id": protocol_case["compound_id"],
        "name": protocol_case["name"],
        "flexibility_bin": protocol_case["flexibility_bin"],
        "claim_boundary": protocol["claim_boundary"],
    }
    for key, expected in required_identity.items():
        _assert_equivalent(
            record.get(key),
            expected,
            name=f"model-case identity {key}",
        )

    environment = record.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("The model-case environment is missing.")
    required_environment = {
        "python": protocol["implementation_freeze"]["required_python"],
        "ase_version": protocol["implementation_freeze"]["required_versions"]["ase"],
        "numpy_version": protocol["implementation_freeze"]["required_versions"][
            "numpy"
        ],
        "torch_version": protocol["implementation_freeze"]["required_versions"][
            "torch"
        ],
        "device": protocol["execution"]["device"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "native_batch_energy_forces": True,
    }
    for key, expected in required_environment.items():
        _assert_equivalent(
            environment.get(key),
            expected,
            name=f"model-case environment {key}",
        )

    source_case = next(
        case
        for case in source_manifest["cases"]
        if case["compound_id"] == protocol_case["compound_id"]
    )
    template_atoms, positions, solvent_kcal, charge_evidence = _load_case_sources(
        protocol_case=protocol_case,
        source_case=source_case,
    )
    source_evidence = record.get("source_evidence")
    expected_source_evidence = {
        "source_manifest_content_sha256": source_manifest["content_sha256"],
        "state_file": protocol_case["state_file"],
        "state_file_sha256": protocol_case["state_file_sha256"],
        "source_state_count": len(positions),
        "charge": charge_evidence,
    }
    _assert_equivalent(
        source_evidence,
        expected_source_evidence,
        name="model-case source evidence",
    )

    evaluation = record.get("source_state_evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("The source-state evaluation is missing.")
    gas_energy = _finite_array(
        evaluation.get("gas_energy_hartree"),
        shape=(len(positions),),
        name="source gas energies",
    )
    frozen_solvent = _finite_array(
        evaluation.get("frozen_solvent_correction_kcal_mol"),
        shape=(len(positions),),
        name="frozen source solvent corrections",
    )
    solution_potential = _finite_array(
        evaluation.get("solution_potential_hartree"),
        shape=(len(positions),),
        name="source solution potentials",
    )
    if (
        evaluation.get("gas_energy_sha256") != _float_list_sha256(gas_energy)
        or evaluation.get("frozen_solvent_correction_sha256")
        != _float_list_sha256(frozen_solvent)
        or evaluation.get("solution_potential_sha256")
        != _float_list_sha256(solution_potential)
        or not np.array_equal(frozen_solvent, solvent_kcal)
        or not np.allclose(
            solution_potential,
            gas_energy + frozen_solvent / KCAL_PER_HARTREE,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("The source-state energy/correction decomposition changed.")
    _validate_repeat_evidence(
        evaluation.get("repeat_evidence"),
        first_repeat_hartree=gas_energy,
        protocol=protocol,
    )

    sampling = protocol["phase_specific_sampling"]
    threshold = float(
        protocol["optimized_minimum_deduplication"][
            "heavy_atom_rmsd_threshold_angstrom"
        ]
    )
    expected_seeds = {
        "gas": select_diverse_seeds(
            positions_angstrom=positions,
            potential_hartree=gas_energy,
            symbols=list(template_atoms.get_chemical_symbols()),
            maximum_count=int(sampling["maximum_seed_count_per_phase_model_case"]),
            threshold_angstrom=threshold,
        ),
        "solution": select_diverse_seeds(
            positions_angstrom=positions,
            potential_hartree=solution_potential,
            symbols=list(template_atoms.get_chemical_symbols()),
            maximum_count=int(sampling["maximum_seed_count_per_phase_model_case"]),
            threshold_angstrom=threshold,
        ),
    }
    _assert_equivalent(
        record.get("selected_phase_seeds"),
        expected_seeds,
        name="phase-specific seed selection",
        absolute_tolerance=1.0e-10,
    )
    phases = record.get("phases")
    if not isinstance(phases, dict) or set(phases) != {"gas", "solution"}:
        raise ValueError("The model-case phase records are incomplete.")
    gas_ensemble = _validate_phase_record(
        phases["gas"],
        selected_seeds=expected_seeds["gas"],
        template_atoms=template_atoms,
        phase="gas",
        protocol=protocol,
        compound_id=protocol_case["compound_id"],
        model_name=model["name"],
        rotational_symmetry_number=int(protocol_case["rotational_symmetry_number"]),
        evidence_root=evidence_root,
        charge_evidence=charge_evidence,
    )
    solution_ensemble = _validate_phase_record(
        phases["solution"],
        selected_seeds=expected_seeds["solution"],
        template_atoms=template_atoms,
        phase="solution",
        protocol=protocol,
        compound_id=protocol_case["compound_id"],
        model_name=model["name"],
        rotational_symmetry_number=int(protocol_case["rotational_symmetry_number"]),
        evidence_root=evidence_root,
        charge_evidence=charge_evidence,
    )

    variants = {}
    for variant_id in gas_ensemble["ensembles"]:
        gas_value = float(gas_ensemble["ensembles"][variant_id]["free_energy_kcal_mol"])
        solution_value = float(
            solution_ensemble["ensembles"][variant_id]["free_energy_kcal_mol"]
        )
        variants[variant_id] = {
            "gas_phase_ensemble_free_energy_kcal_mol": gas_value,
            "solution_phase_ensemble_free_energy_kcal_mol": solution_value,
            "hydration_prediction_kcal_mol": solution_value - gas_value,
        }
    valid_gas = [
        minimum
        for minimum in phases["gas"]["minimum_analyses"]
        if minimum["status"] == "valid"
    ]
    valid_solution = [
        minimum
        for minimum in phases["solution"]["minimum_analyses"]
        if minimum["status"] == "valid"
    ]
    zero_kelvin = (
        min(
            minimum["potential_energy"]["total_energy_hartree"]
            for minimum in valid_solution
        )
        - min(
            minimum["potential_energy"]["total_energy_hartree"] for minimum in valid_gas
        )
    ) * KCAL_PER_HARTREE
    expected_prediction = {
        "variants": variants,
        "primary_hydration_prediction_kcal_mol": variants["primary"][
            "hydration_prediction_kcal_mol"
        ],
        "zero_kelvin_separately_optimized_minimum_kcal_mol": zero_kelvin,
        "single_reference_endpoint_baseline_kcal_mol": float(
            source_case["reference_endpoint_kcal_mol"]
        ),
    }
    _assert_equivalent(
        record.get("prediction"),
        expected_prediction,
        name="model-case prediction",
        absolute_tolerance=1.0e-8,
    )
    if require_durable_evidence:
        hessian_hashes = sorted(
            {
                hessian_record["sha256"]
                for phase in ("gas", "solution")
                for minimum in phases[phase]["minimum_analyses"]
                for hessian_record in _minimum_hessian_records(minimum)
            }
        )
        durable = record.get("durable_evidence")
        if not isinstance(durable, dict):
            raise ValueError("A sealed record lacks durable-evidence provenance.")
        source_hash = durable.get("source_record_content_sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError("The source-record provenance hash is invalid.")
        _assert_equivalent(
            {
                key: durable.get(key)
                for key in (
                    "schema_version",
                    "hessian_storage",
                    "hessian_sha256",
                )
            },
            {
                "schema_version": 1,
                "hessian_storage": ("repository-relative-content-addressed-npy"),
                "hessian_sha256": hessian_hashes,
            },
            name="durable scientific evidence",
        )


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _minimum_hessian_records(minimum: dict[str, Any]) -> list[dict[str, Any]]:
    if minimum.get("status") != "valid":
        return []
    records = [minimum["hessian"]]
    recovery = minimum.get("stationary_point_recovery", {})
    if recovery.get("required") is True:
        records.append(recovery["trigger_hessian"])
    sensitivity = minimum.get("hessian_displacement_sensitivity", {})
    if sensitivity.get("required") is True:
        records.extend(record["hessian"] for record in sensitivity["records"])
    return records


def _make_durable_record(
    record: dict[str, Any],
    *,
    source_evidence_root: Path,
    durable_root: Path,
) -> dict[str, Any]:
    """Copy Hessians into content-addressed durable storage and reseal a record."""

    durable = copy.deepcopy(record)
    source_content_sha256 = durable.pop("content_sha256")
    hessian_hashes = []
    for phase in ("gas", "solution"):
        for minimum in durable["phases"][phase]["minimum_analyses"]:
            for hessian_record in _minimum_hessian_records(minimum):
                source = _resolve_repository_file(
                    hessian_record["path"],
                    name="source Hessian evidence path",
                )
                try:
                    source.relative_to(source_evidence_root.resolve())
                except ValueError as exc:
                    raise ValueError(
                        "A source Hessian escaped the model-case work directory."
                    ) from exc
                expected_sha256 = hessian_record["sha256"]
                if sha256_file(source) != expected_sha256:
                    raise ValueError("A source Hessian changed before durable sealing.")
                destination = durable_root / "hessians" / f"{expected_sha256}.npy"
                if destination.exists():
                    if (
                        not destination.is_file()
                        or sha256_file(destination) != expected_sha256
                    ):
                        raise ValueError(
                            "A content-addressed durable Hessian conflicts."
                        )
                else:
                    _copy_file_atomic(source, destination)
                    if sha256_file(destination) != expected_sha256:
                        raise ValueError("A durable Hessian copy failed verification.")
                hessian_record["path"] = _relative_to_repository(destination)
                hessian_hashes.append(expected_sha256)
    durable["durable_evidence"] = {
        "schema_version": 1,
        "source_record_content_sha256": source_content_sha256,
        "hessian_storage": "repository-relative-content-addressed-npy",
        "hessian_sha256": sorted(set(hessian_hashes)),
    }
    return seal_artifact(durable)


def _model_environment(
    *,
    calculator,
    model: dict[str, Any],
    device,
    load_seconds: float,
) -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ase_version": importlib.metadata.version("ase"),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "checkpoint_sha256": sha256_file(_checkpoint_path(model)),
        "calculator_class": (
            f"{type(calculator).__module__}.{type(calculator).__qualname__}"
        ),
        "calculator_dtype": str(getattr(calculator, "dtype", None)),
        "calculator_cutoff_angstrom": (
            float(getattr(calculator, "r_max"))
            if getattr(calculator, "r_max", None) is not None
            else None
        ),
        "native_batch_energy_forces": bool(
            getattr(calculator, "supports_batch_energy_forces", False)
        ),
        "model_load_seconds": float(load_seconds),
    }


def validate_command(args: argparse.Namespace) -> None:
    protocol, fingerprint, manifest = load_qrrho_protocol(args.protocol)
    require_exact_frozen_sources(
        REPOSITORY_ROOT,
        protocol["implementation_freeze"]["source_files"],
    )
    print(
        f"Validated {protocol['protocol_id']} fingerprint={fingerprint} "
        f"for {len(protocol['models'])} models x {len(protocol['cases'])} cases; "
        f"source={manifest['content_sha256']}."
    )


def run_command(args: argparse.Namespace) -> None:
    import torch

    protocol, fingerprint, manifest = load_qrrho_protocol(args.protocol)
    require_exact_frozen_sources(
        REPOSITORY_ROOT,
        protocol["implementation_freeze"]["source_files"],
    )
    device = torch.device(protocol["execution"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The preregistered cuda:0 execution device is unavailable.")
    work_dir = Path(args.work_dir).resolve()
    _relative_to_repository(work_dir)
    record_dir = work_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    model_filter = set(args.model or [])
    case_filter = set(args.case or [])
    models = [
        model
        for model in protocol["models"]
        if not model_filter or model["name"] in model_filter
    ]
    cases = [
        case
        for case in protocol["cases"]
        if not case_filter or case["compound_id"] in case_filter
    ]
    if len(models) != (len(model_filter) if model_filter else len(models)):
        raise ValueError("Unknown --model selection.")
    if len(cases) != (len(case_filter) if case_filter else len(cases)):
        raise ValueError("Unknown --case selection.")

    source_cases = {case["compound_id"]: case for case in manifest["cases"]}
    first_case = cases[0]
    first_atoms, _, _, _ = _load_case_sources(
        protocol_case=first_case,
        source_case=source_cases[first_case["compound_id"]],
    )
    for model in models:
        start = time.perf_counter()
        calculator = SetCalculator(
            device,
            model["name"],
            str(work_dir / f"{model['name']}.log"),
            atoms=first_atoms,
            model_options={"hessian": "numerical"},
        ).set_calculator()
        load_seconds = time.perf_counter() - start
        source_model = next(
            item for item in manifest["models"] if item["name"] == model["name"]
        )
        if [int(value) for value in calculator.atomic_numbers] != source_model[
            "supported_atomic_numbers"
        ]:
            raise ValueError(f"Model element domain changed for {model['name']}.")
        if not calculator.supports_batch_energy_forces:
            raise RuntimeError(f"{model['name']} lacks the required native batch path.")
        environment = _model_environment(
            calculator=calculator,
            model=model,
            device=device,
            load_seconds=load_seconds,
        )
        for case in cases:
            destination = record_dir / f"{model['name']}--{case['compound_id']}.json"
            if destination.is_file():
                existing = load_json(destination)
                try:
                    validate_model_case_record(
                        existing,
                        protocol=protocol,
                        fingerprint=fingerprint,
                        source_manifest=manifest,
                        model=model,
                        protocol_case=case,
                        evidence_root=work_dir,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Existing record is not resumable: {destination}."
                    ) from exc
                print(f"skip {model['name']}/{case['compound_id']}", flush=True)
                continue
            print(f"run {model['name']}/{case['compound_id']}", flush=True)
            try:
                record = _run_model_case(
                    protocol=protocol,
                    fingerprint=fingerprint,
                    source_manifest=manifest,
                    model=model,
                    protocol_case=case,
                    calculator=calculator,
                    environment=environment,
                    work_dir=work_dir,
                )
            except Exception as exc:
                calculator.solvent_correction = None
                record = seal_artifact(
                    {
                        "schema_version": 1,
                        "artifact_type": (
                            "route1-multi-mlip-phase-specific-qrrho-model-case"
                        ),
                        "protocol_id": protocol["protocol_id"],
                        "protocol_fingerprint": fingerprint,
                        "source_partition": "development",
                        "status": "failure",
                        "model": model,
                        "compound_id": case["compound_id"],
                        "failure": {
                            "exception_class": type(exc).__name__,
                            "reason": str(exc),
                        },
                    }
                )
            if record["status"] == "success":
                validate_model_case_record(
                    record,
                    protocol=protocol,
                    fingerprint=fingerprint,
                    source_manifest=manifest,
                    model=model,
                    protocol_case=case,
                    evidence_root=work_dir,
                )
            write_json_atomic(destination, record)
            print(
                f"{model['name']}/{case['compound_id']}: {record['status']}",
                flush=True,
            )
            if record["status"] != "success":
                raise RuntimeError(
                    f"Fail-closed qRRHO record written to {destination}."
                )


def seal_command(args: argparse.Namespace) -> None:
    protocol, fingerprint, manifest = load_qrrho_protocol(args.protocol)
    require_exact_frozen_sources(
        REPOSITORY_ROOT,
        protocol["implementation_freeze"]["source_files"],
    )
    record_dir = Path(args.record_dir).resolve()
    if not record_dir.is_dir():
        raise ValueError(f"Record directory is missing: {record_dir}.")
    _relative_to_repository(record_dir)
    output_path = Path(args.output).resolve()
    _relative_to_repository(output_path)
    if output_path.parent != SCRIPT_DIR:
        raise ValueError(
            "The sealed qRRHO aggregate and durable raw evidence must be written "
            "directly under docs/implicit-solvation/benchmarks."
        )
    durable_root = output_path.parent / f"{output_path.stem}-raw"
    _relative_to_repository(durable_root)
    durable_root.mkdir(parents=True, exist_ok=True)
    records = []
    for model in protocol["models"]:
        for case in protocol["cases"]:
            path = record_dir / f"{model['name']}--{case['compound_id']}.json"
            if not path.is_file():
                raise ValueError(f"Missing preregistered model-case record: {path}.")
            record = load_json(path)
            try:
                validate_model_case_record(
                    record,
                    protocol=protocol,
                    fingerprint=fingerprint,
                    source_manifest=manifest,
                    model=model,
                    protocol_case=case,
                    evidence_root=record_dir.parent,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid preregistered model-case record: {path}."
                ) from exc
            durable_record = _make_durable_record(
                record,
                source_evidence_root=record_dir.parent,
                durable_root=durable_root,
            )
            durable_path = (
                durable_root
                / "records"
                / f"{model['name']}--{case['compound_id']}.json"
            )
            write_json_atomic(durable_path, durable_record)
            validate_model_case_record(
                durable_record,
                protocol=protocol,
                fingerprint=fingerprint,
                source_manifest=manifest,
                model=model,
                protocol_case=case,
                evidence_root=durable_root,
                require_durable_evidence=True,
            )
            records.append(
                {
                    "model": model["name"],
                    "compound_id": case["compound_id"],
                    "record": _relative_to_repository(durable_path),
                    "record_file_sha256": sha256_file(durable_path),
                    "record_content_sha256": durable_record["content_sha256"],
                    "prediction": durable_record["prediction"],
                }
            )
    expected = int(protocol["energy_artifact_gates"]["expected_model_case_count"])
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} complete model-cases, found {len(records)}."
        )
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-multi-mlip-phase-specific-qrrho-energy-artifact",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": fingerprint,
        "source_partition": "development",
        "status": "sealed-before-label-scoring",
        "source_manifest_content_sha256": manifest["content_sha256"],
        "durable_raw_root": _relative_to_repository(durable_root),
        "durable_record_schema_version": 1,
        "model_case_count": len(records),
        "records": records,
        "claim_boundary": protocol["claim_boundary"],
        "command_provenance": command_provenance(
            __file__,
            {
                "protocol": args.protocol,
                "record_dir": args.record_dir,
                "output": args.output,
                "phase": "seal",
            },
            repository_root=REPOSITORY_ROOT,
        ),
    }
    if _contains_forbidden_label_key(artifact):
        raise ValueError("The sealed qRRHO energy artifact contains a label.")
    seal_artifact(artifact)
    write_json_atomic(output_path, artifact)
    print(f"Sealed {len(records)} label-free model-case records to " f"{output_path}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    default_protocol = SCRIPT_DIR / "multi_mlip_phase_specific_qrrho_protocol.json"

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate every frozen source, model, implementation, and invariant.",
    )
    validate_parser.add_argument("--protocol", default=str(default_protocol))
    validate_parser.set_defaults(handler=validate_command)

    run_parser = subparsers.add_parser(
        "run",
        help="Run or resume label-blind model-case records.",
    )
    run_parser.add_argument("--protocol", default=str(default_protocol))
    run_parser.add_argument(
        "--work-dir",
        default=(
            ".omx/benchmarks/"
            "route1-multi-mlip-phase-specific-selected-minimum-rrho-v8"
        ),
    )
    run_parser.add_argument("--model", action="append")
    run_parser.add_argument("--case", action="append")
    run_parser.set_defaults(handler=run_command)

    seal_parser = subparsers.add_parser(
        "seal",
        help="Seal the exact 18 complete records before any label scoring.",
    )
    seal_parser.add_argument("--protocol", default=str(default_protocol))
    seal_parser.add_argument(
        "--record-dir",
        default=(
            ".omx/benchmarks/"
            "route1-multi-mlip-phase-specific-selected-minimum-rrho-v8/records"
        ),
    )
    seal_parser.add_argument(
        "--output",
        default=(
            "docs/implicit-solvation/benchmarks/"
            "route1-multi-mlip-phase-specific-qrrho-2026-07-25.json"
        ),
    )
    seal_parser.set_defaults(handler=seal_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
