#!/usr/bin/env python3
"""Execute the preregistered terminal fixed-radial source experiment.

The finite candidate is selected from eight frozen QM surface-MEP records.
Four additional MEP records and all four matched QM/PCMSolver polarization
energies are excluded from selection.  No experimental solvation label, energy
ledger, force evidence, or public capability is read or admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time

from ase.units import Bohr
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (
    PCMSolverSession,
)
from maple.function.calculator.extra_correction.implicit.route2_static_surface_mep import (
    build_route2_smd_exterior_probe_surface,
    route2_weighted_surface_mep_discrepancy,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.route2_smd_profiles import (
    DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
)
from maple.solvation.api.profiles import (
    MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
)
from maple.solvation.coupling.radial_embedding import FixedRadialSourceEmbedding
from maple.solvation.models import (
    MACEPolarOriginalSourceNativeFieldAdapter,
    build_official_mace_polar_1_m_radial_gto_adapter,
)
from maple.solvation.release import (
    RadialEmbeddingCandidateMetric,
    RepositorySnapshot,
    canonical_json_sha256,
    checkpoint_record,
    collect_loaded_repository_sources,
    committed_source_hashes,
    runtime_record,
    select_fixed_radial_embedding,
    sha256_file,
    write_external_json_artifact,
)

SCHEMA_VERSION = "route2-mace-fixed-radial-embedding-terminal-v1"
PREREGISTRATION = (
    "docs/route2/preregistrations/"
    "mace-original-source-fixed-radial-embedding-v1.json"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
STATIC_MEP_RECORD_ROOT = (
    ".omx/benchmarks/route2-v0-freesolv12-zero-field-static-mep-edfae78e/records"
)
MOL2_DATASET_ROOT = ".omx/benchmarks/route2-macepolar-smd-smoke/dataset"
PCM_PANEL_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
PCM_CUTOFF_DIRECTORY = "cutoff-1e-12"
KCAL_PER_HARTREE = 627.5094740631
GEOMETRY_ATOL_ANGSTROM = 1.0e-8
SURFACE_ATOL_BOHR = 1.0e-12
TOTAL_CHARGE_ATOL_E = 1.0e-8
NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
REQUIRED_SOURCE_PATHS = (
    "maple/function/calculator/extra_correction/implicit/continuum_response.py",
    "maple/function/calculator/extra_correction/implicit/gto_density.py",
    "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
    "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
    "maple/function/calculator/extra_correction/implicit/route2_static_surface_mep.py",
    "maple/solvation/coupling/radial_embedding.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/models/mace_polar_separated.py",
    "maple/solvation/release/radial_embedding.py",
    "tools/route2_release/run_mace_fixed_radial_embedding_panel.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--pcmsolver-library", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACEPOLAR1Mmodel",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _repository_asset(
    repository: RepositorySnapshot, record: object, *, name: str
) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"{name} path record is invalid.")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{name} path is missing.")
    path = repository.root / raw_path
    _validated_sha(path, record.get("sha256"), name=name)
    return path


def _rebased_asset_path(asset_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Frozen asset path must be a non-empty string.")
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()
    try:
        marker = candidate.parts.index(".omx")
    except ValueError as exc:
        raise RuntimeError(f"Cannot rebase frozen asset path {raw_path!r}.") from exc
    result = asset_root.joinpath(*candidate.parts[marker:]).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _configure_determinism(torch: object) -> None:
    torch.manual_seed(20260815)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260815)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _array_sha256(values: object) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _molecular_dipole(source4: np.ndarray, positions: np.ndarray) -> np.ndarray:
    atomic_dipoles_xyz = source4[:, (3, 1, 2)]
    return np.sum(source4[:, 0, None] * positions + atomic_dipoles_xyz, axis=0)


def _token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _candidate_grid(
    preregistration: dict[str, object],
) -> tuple[tuple[str, FixedRadialSourceEmbedding, float], ...]:
    grid = preregistration.get("candidate_grid")
    if not isinstance(grid, dict):
        raise RuntimeError("Preregistration omits candidate_grid.")
    baseline_id = str(grid.get("baseline_candidate_id"))
    broad = float(preregistration["model_and_map"]["broad_sigma_angstrom"])
    narrow_sigmas = tuple(float(value) for value in grid["narrow_sigmas_angstrom"])
    narrow_weights = tuple(float(value) for value in grid["narrow_weights"])
    candidates: list[tuple[str, FixedRadialSourceEmbedding, float]] = [
        (baseline_id, FixedRadialSourceEmbedding((broad,), (1.0,)), 0.0)
    ]
    for sigma in narrow_sigmas:
        if not 0.0 < sigma < broad:
            raise RuntimeError("Narrow sigma grid must lie strictly below broad sigma.")
        for weight in narrow_weights:
            if not 0.0 < weight <= 1.0:
                raise RuntimeError("Narrow weight grid must lie in (0,1].")
            if weight == 1.0:
                candidate_id = f"pure-sigma-{_token(sigma)}"
                embedding = FixedRadialSourceEmbedding((sigma,), (1.0,))
            else:
                candidate_id = f"mix-sigma-{_token(sigma)}-weight-{_token(weight)}"
                embedding = FixedRadialSourceEmbedding(
                    (sigma, broad), (weight, 1.0 - weight)
                )
            distance = float(
                sum(
                    radial_weight * (radial_sigma - broad) ** 2
                    for radial_sigma, radial_weight in zip(
                        embedding.sigmas_angstrom, embedding.weights, strict=True
                    )
                )
            )
            candidates.append((candidate_id, embedding, distance))
    if len(candidates) != int(grid.get("candidate_count", -1)):
        raise RuntimeError("Generated candidate count differs from preregistration.")
    if len({candidate[0] for candidate in candidates}) != len(candidates):
        raise RuntimeError("Generated candidate IDs are not unique.")
    return tuple(candidates)


def _static_mep_case(
    *,
    record: dict[str, object],
    split: str,
    candidates: tuple[tuple[str, FixedRadialSourceEmbedding, float], ...],
    asset_root: Path,
    static_record_root: Path,
    dataset_root: Path,
    electronic: MACEPolarOriginalSourceNativeFieldAdapter,
    clearance_angstrom: float,
) -> tuple[dict[str, object], tuple[RadialEmbeddingCandidateMetric, ...]]:
    compound_id = str(record["compound_id"])
    frozen_record_path = static_record_root / f"{compound_id}.json"
    frozen_record = _load_json(frozen_record_path, name=f"{compound_id} MEP record")
    if frozen_record.get("compound_id") != compound_id:
        raise RuntimeError(f"{compound_id} frozen record identity drifted.")
    mol2 = dataset_root / str(record["mol2_archive_member"])
    _validated_sha(mol2, record.get("mol2_sha256"), name=f"{compound_id} MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)

    attempt = static_record_root / compound_id / "qm-attempt-001"
    qm_path = attempt / "qm-static-mep.npz"
    surface_path = attempt / "surface.npz"
    qm_record = frozen_record.get("qm_reference")
    if not isinstance(qm_record, dict):
        raise RuntimeError(f"{compound_id} omits QM reference identity.")
    _validated_sha(
        qm_path,
        qm_record.get("static_mep_npz_sha256"),
        name=f"{compound_id} QM MEP",
    )
    with np.load(qm_path, allow_pickle=False) as qm:
        reference = np.asarray(qm["surface_potential_hartree_per_e"], dtype=float)
        qm_total_charge = float(qm["total_charge_e"])
        qm_positions = np.asarray(qm["atom_positions_angstrom"], dtype=float)
        qm_numbers = np.asarray(qm["atomic_numbers"], dtype=int)
    points = np.asarray(
        np.load(surface_path, allow_pickle=False)["surface_points_bohr"], dtype=float
    )
    if not np.array_equal(qm_numbers, np.asarray(atoms.numbers, dtype=int)):
        raise RuntimeError(f"{compound_id} QM and MOL2 elements differ.")
    geometry_error = float(np.max(np.abs(qm_positions - atoms.positions)))
    if geometry_error > GEOMETRY_ATOL_ANGSTROM:
        raise RuntimeError(f"{compound_id} QM/MOL2 geometry mismatch.")
    surface_record = frozen_record.get("surface")
    if not isinstance(surface_record, dict):
        raise RuntimeError(f"{compound_id} omits frozen surface identity.")
    if _array_sha256(points) != surface_record.get("surface_points_sha256"):
        raise RuntimeError(f"{compound_id} frozen surface hash drifted.")
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        profile=DDPCM_MULTISOLVENT_SMD_DIRECT_PCM_V2_PROFILE,
    )
    replay = build_route2_smd_exterior_probe_surface(
        atoms.positions,
        radii,
        clearance_angstrom=clearance_angstrom,
    )
    surface_error = float(np.max(np.abs(replay.surface_points_bohr - points)))
    if surface_error > SURFACE_ATOL_BOHR:
        raise RuntimeError(f"{compound_id} static surface did not replay.")
    if reference.shape != (len(points),):
        raise RuntimeError(f"{compound_id} QM MEP length is invalid.")

    source = np.asarray(
        electronic.evaluate_source(atoms, np.zeros((len(atoms), 8))), dtype=float
    )
    if source.shape != (len(atoms), 4) or not np.all(np.isfinite(source)):
        raise RuntimeError(f"{compound_id} MACE source is invalid.")
    source_record = frozen_record.get("mace_zero_field_source")
    old_metrics = frozen_record.get("static_surface_mep_metrics")
    if not isinstance(source_record, dict) or not isinstance(old_metrics, dict):
        raise RuntimeError(f"{compound_id} omits historical source replay data.")
    dipole = _molecular_dipole(source, atoms.positions)
    old_dipole = np.asarray(source_record["molecular_dipole_e_angstrom"], dtype=float)
    dipole_replay_error = float(np.max(np.abs(dipole - old_dipole)))
    charge_replay_error = abs(
        float(np.sum(source[:, 0])) - float(source_record["total_charge_e"])
    )
    point_metrics = route2_weighted_surface_mep_discrepancy(
        point_multipole_potential(points, atoms.positions, source),
        reference,
        replay.quadrature_weights,
    )
    metric_replay_error = max(
        abs(float(point_metrics[name]) - float(old_metrics[name]))
        for name in point_metrics
    )
    current_charge_error = abs(float(np.sum(source[:, 0])) - qm_total_charge)
    if current_charge_error > TOTAL_CHARGE_ATOL_E:
        raise RuntimeError(f"{compound_id} current source violates total charge.")

    metrics: list[RadialEmbeddingCandidateMetric] = []
    candidate_payload: list[dict[str, object]] = []
    maximum_moment_preservation_error = 0.0
    for candidate_id, embedding, distance in candidates:
        expanded = embedding.expand(source)
        maximum_moment_preservation_error = max(
            maximum_moment_preservation_error,
            float(np.max(np.abs(np.sum(expanded, axis=1) - source))),
        )
        discrepancy = route2_weighted_surface_mep_discrepancy(
            embedding.surface_potential(points, atoms.positions, source),
            reference,
            replay.quadrature_weights,
        )
        metric = RadialEmbeddingCandidateMetric(
            candidate_id=candidate_id,
            compound_id=compound_id,
            split=split,
            embedding_distance_from_original=distance,
            **discrepancy,
        )
        metrics.append(metric)
        candidate_payload.append(
            {
                "candidate_id": candidate_id,
                "configuration_sha256": embedding.configuration_sha256,
                "embedding_distance_from_original_angstrom2": distance,
                **discrepancy,
            }
        )
    if maximum_moment_preservation_error > 2.0e-15:
        raise RuntimeError(f"{compound_id} radial map changed a source moment.")
    return (
        {
            "compound_id": compound_id,
            "name": str(record["name"]),
            "chemical_class": str(record["chemical_class"]),
            "functional_group": record["functional_group"],
            "split": split,
            "atom_count": len(atoms),
            "geometry_max_abs_error_angstrom": geometry_error,
            "surface_replay_max_abs_error_bohr": surface_error,
            "current_source_identity": {
                "source4_sha256": _array_sha256(source),
                "total_charge_absolute_error_to_qm_e": current_charge_error,
                "point_multipole_metrics": point_metrics,
            },
            "historical_legacy_source_diagnostic_only": {
                "identity_note": (
                    "The historical source used the upstream molecular-realspace "
                    "evaluator, whereas this experiment uses the current analytic-"
                    "Gaussian evaluator profile. Historical MACE predictions are "
                    "not a source-identity gate; only their frozen QM MEP assets "
                    "and geometry are reused."
                ),
                "historical_source4_sha256": source_record[
                    "density_coefficients_sha256"
                ],
                "bitwise_source_hash_replayed": bool(
                    _array_sha256(source)
                    == source_record["density_coefficients_sha256"]
                ),
                "charge_absolute_error_e": charge_replay_error,
                "dipole_max_abs_error_e_angstrom": dipole_replay_error,
                "point_multipole_metric_max_abs_error": metric_replay_error,
            },
            "maximum_moment_preservation_error": maximum_moment_preservation_error,
            "assets": {
                "mol2_sha256": sha256_file(mol2),
                "frozen_record_sha256": sha256_file(frozen_record_path),
                "qm_mep_sha256": sha256_file(qm_path),
                "surface_sha256": sha256_file(surface_path),
            },
            "candidate_metrics": candidate_payload,
        },
        tuple(metrics),
    )


def _pcm_case(
    *,
    record: dict[str, object],
    asset_root: Path,
    electronic: MACEPolarOriginalSourceNativeFieldAdapter,
    embedding: FixedRadialSourceEmbedding,
    pcmsolver_library: Path,
    energy_budget_kcal_per_mol: float,
) -> dict[str, object]:
    compound_id = str(record["compound_id"])
    panel_root = asset_root / PCM_PANEL_ROOT / compound_id
    result_path = panel_root / PCM_CUTOFF_DIRECTORY / "result.json"
    work = panel_root / PCM_CUTOFF_DIRECTORY / "work"
    result = _load_json(result_path, name=f"{compound_id} projection result")
    inputs = result.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("compound_id") != compound_id:
        raise RuntimeError(f"{compound_id} projection result identity is invalid.")
    mol2 = asset_root / str(record["mol2_path"])
    _validated_sha(mol2, record.get("mol2_sha256"), name=f"{compound_id} MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    qm_path = work / "qm-surface-mep.npz"
    surface_path = work / "surface.npz"
    with np.load(qm_path, allow_pickle=False) as qm:
        reference = np.asarray(qm["surface_potential_hartree_per_e"], dtype=float)
        qm_positions = np.asarray(qm["atom_positions_angstrom"], dtype=float)
        qm_numbers = np.asarray(qm["atomic_numbers"], dtype=int)
        qm_energy_source = str(qm["density_source"])
    if not np.array_equal(qm_numbers, np.asarray(atoms.numbers, dtype=int)):
        raise RuntimeError(f"{compound_id} PCM QM/MOL2 elements differ.")
    geometry_error = float(np.max(np.abs(qm_positions - atoms.positions)))
    if geometry_error > GEOMETRY_ATOL_ANGSTROM:
        raise RuntimeError(f"{compound_id} PCM QM/MOL2 geometry mismatch.")
    checkpoint_record_data = inputs.get("qm_checkpoint")
    pcm_input_record = inputs.get("parsed_pcm_input")
    if not isinstance(checkpoint_record_data, dict) or not isinstance(
        pcm_input_record, dict
    ):
        raise RuntimeError(f"{compound_id} omits frozen PCM inputs.")
    qm_checkpoint = _rebased_asset_path(asset_root, checkpoint_record_data.get("path"))
    pcm_input = _rebased_asset_path(asset_root, pcm_input_record.get("path"))
    _validated_sha(
        qm_checkpoint,
        checkpoint_record_data.get("sha256"),
        name=f"{compound_id} QM checkpoint",
    )
    _validated_sha(
        pcm_input,
        pcm_input_record.get("sha256"),
        name=f"{compound_id} PCM input",
    )
    radii = np.asarray(pcm_input_record["cavity_radii_angstrom"], dtype=float)
    source = np.asarray(
        electronic.evaluate_source(atoms, np.zeros((len(atoms), 8))), dtype=float
    )
    frozen_surface = np.asarray(
        np.load(surface_path, allow_pickle=False)["surface_points_bohr"], dtype=float
    )
    with tempfile.TemporaryDirectory(prefix=f"route2-radial-{compound_id}-") as workdir:
        previous = Path.cwd()
        try:
            os.chdir(workdir)
            with PCMSolverSession(
                np.asarray(atoms.numbers, dtype=float),
                atoms.positions / Bohr,
                pcm_input,
                library_path=pcmsolver_library,
            ) as session:
                points = np.asarray(session.cavity_centers_bohr, dtype=float)
                surface_error = float(np.max(np.abs(points - frozen_surface)))
                if surface_error > SURFACE_ATOL_BOHR:
                    raise RuntimeError(f"{compound_id} PCMSolver surface drifted.")
                response = PCMSolverExternalMEPCavityResponse(
                    session, cavity_radii_angstrom=radii
                )
                predicted = embedding.surface_potential(points, atoms.positions, source)
                qm_charge = np.asarray(
                    response.apply_energy_conjugate(reference), dtype=float
                )
                predicted_charge = np.asarray(
                    response.apply_energy_conjugate(predicted), dtype=float
                )
                error = predicted - reference
                error_charge = np.asarray(
                    response.apply_energy_conjugate(error), dtype=float
                )
                areas = np.asarray(session.cavity_areas_bohr2, dtype=float)
        finally:
            os.chdir(previous)
    qm_energy = 0.5 * float(np.vdot(reference, qm_charge))
    predicted_energy = 0.5 * float(np.vdot(predicted, predicted_charge))
    reference_norm = float(np.sqrt(max(0.0, -float(np.vdot(reference, qm_charge)))))
    error_norm = float(np.sqrt(max(0.0, -float(np.vdot(error, error_charge)))))
    energy_error_kcal = abs(predicted_energy - qm_energy) * KCAL_PER_HARTREE
    energy_bound_kcal = (
        reference_norm * error_norm + 0.5 * error_norm * error_norm
    ) * KCAL_PER_HARTREE
    if energy_error_kcal > energy_bound_kcal + 2.0e-9:
        raise RuntimeError(f"{compound_id} response-norm bound is inconsistent.")
    target = float(
        result["basis_results"]["one_radial"]["target_polarization_energy_hartree"]
    )
    if abs(qm_energy - target) > 2.0e-13:
        raise RuntimeError(f"{compound_id} target polarization energy drifted.")
    area_relative_mep = float(
        np.sqrt(np.sum(areas * error * error) / np.sum(areas * reference * reference))
    )
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "class": str(record["class"]),
        "atom_count": len(atoms),
        "geometry_max_abs_error_angstrom": geometry_error,
        "surface_replay_max_abs_error_bohr": surface_error,
        "qm_density_source": qm_energy_source,
        "qm_polarization_energy_kcal_per_mol": qm_energy * KCAL_PER_HARTREE,
        "predicted_polarization_energy_kcal_per_mol": (
            predicted_energy * KCAL_PER_HARTREE
        ),
        "polarization_energy_absolute_error_kcal_per_mol": energy_error_kcal,
        "polarization_energy_error_upper_bound_kcal_per_mol": energy_bound_kcal,
        "surface_mep_area_weighted_relative_l2_error": area_relative_mep,
        "fixed_source_energy_gate_passed": bool(
            energy_error_kcal < energy_budget_kcal_per_mol
        ),
        "assets": {
            "mol2_sha256": sha256_file(mol2),
            "projection_result_sha256": sha256_file(result_path),
            "qm_surface_mep_sha256": sha256_file(qm_path),
            "surface_points_sha256": sha256_file(surface_path),
            "qm_checkpoint_sha256": sha256_file(qm_checkpoint),
            "pcm_input_sha256": sha256_file(pcm_input),
        },
    }


def main() -> None:
    args = _parse_args()
    repository = RepositorySnapshot.capture(Path(__file__).parents[2])
    started = time.perf_counter()
    asset_root = args.asset_root.expanduser().resolve(strict=True)
    pcmsolver_library = args.pcmsolver_library.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("MACE checkpoint does not match the frozen official model.")
    preregistration_path = repository.root / PREREGISTRATION
    preregistration = _load_json(preregistration_path, name="radial preregistration")
    if (
        preregistration.get("artifact_id")
        != "route2-mace-original-source-fixed-radial-embedding-prereg-v1"
        or preregistration.get("status") != "frozen-before-transfer-execution"
        or preregistration.get("capabilities") != NO_CAPABILITIES
    ):
        raise RuntimeError(
            "Radial preregistration identity or capability state is invalid."
        )
    evidence = preregistration.get("input_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("Radial preregistration omits input evidence.")
    original_negative = _repository_asset(
        repository,
        evidence["original_source_pcm_negative_artifact"],
        name="negative source evidence",
    )
    farfield_positive = _repository_asset(
        repository,
        evidence["far_field_necessary_gate_artifact"],
        name="far-field evidence",
    )
    static_manifest_path = _repository_asset(
        repository, evidence["frozen_static_mep_manifest"], name="static MEP manifest"
    )
    pcm_manifest_path = _repository_asset(
        repository,
        evidence["matched_pcmsolver_preregistration"],
        name="PCMSolver preregistration",
    )
    if _load_json(original_negative, name="negative source evidence")["decision"][
        "original_four_channel_quantitative_pcm_source_admitted"
    ]:
        raise RuntimeError(
            "Original source negative gate unexpectedly admits the source."
        )
    if not _load_json(farfield_positive, name="far-field evidence")["decision"][
        "fixed_radial_embedding_research_authorized"
    ]:
        raise RuntimeError("Far-field evidence does not authorize this experiment.")

    static_manifest = _load_json(static_manifest_path, name="static MEP manifest")
    static_records = static_manifest.get("locked_records")
    if not isinstance(static_records, list) or len(static_records) != 12:
        raise RuntimeError("Expected twelve frozen static MEP records.")
    selection_contract = preregistration.get("selection")
    gates = preregistration.get("admission_gates")
    if not isinstance(selection_contract, dict) or not isinstance(gates, dict):
        raise RuntimeError("Preregistration omits selection or gates.")
    training_ids = tuple(selection_contract["training_compound_ids"])
    heldout_ids = tuple(selection_contract["heldout_from_selection_compound_ids"])
    if set(training_ids).intersection(heldout_ids) or set(
        training_ids + heldout_ids
    ) != {str(record["compound_id"]) for record in static_records}:
        raise RuntimeError("Static MEP split does not exactly cover the frozen panel.")
    candidates = _candidate_grid(preregistration)
    candidate_by_id = {
        candidate_id: embedding for candidate_id, embedding, _ in candidates
    }

    expected_pcm_sha = _load_json(pcm_manifest_path, name="PCMSolver preregistration")[
        "execution_contract"
    ]["pcmsolver_library_sha256"]
    _validated_sha(pcmsolver_library, expected_pcm_sha, name="PCMSolver shared library")

    import torch

    _configure_determinism(torch)
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    electronic = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    static_record_root = asset_root / STATIC_MEP_RECORD_ROOT
    dataset_root = asset_root / MOL2_DATASET_ROOT
    clearance = float(static_manifest["probe_surface"]["clearance_angstrom"])
    static_payload: list[dict[str, object]] = []
    all_metrics: list[RadialEmbeddingCandidateMetric] = []
    for record in static_records:
        compound_id = str(record["compound_id"])
        split = "train" if compound_id in training_ids else "heldout"
        payload, metrics = _static_mep_case(
            record=record,
            split=split,
            candidates=candidates,
            asset_root=asset_root,
            static_record_root=static_record_root,
            dataset_root=dataset_root,
            electronic=electronic,
            clearance_angstrom=clearance,
        )
        static_payload.append(payload)
        all_metrics.extend(metrics)
    baseline_id = str(preregistration["candidate_grid"]["baseline_candidate_id"])
    selection = select_fixed_radial_embedding(
        tuple(all_metrics),
        candidate_ids=tuple(candidate_by_id),
        training_compound_ids=training_ids,
        baseline_candidate_id=baseline_id,
    )
    selected_id = selection.selected_candidate_id
    selected = candidate_by_id[selected_id]
    metric_lookup = {
        (metric.candidate_id, metric.compound_id): metric for metric in all_metrics
    }
    selected_heldout_objective = float(
        np.mean(
            [
                metric_lookup[(selected_id, compound_id)].weighted_relative_l2 ** 2
                for compound_id in heldout_ids
            ]
        )
    )
    baseline_heldout_objective = float(
        np.mean(
            [
                metric_lookup[(baseline_id, compound_id)].weighted_relative_l2 ** 2
                for compound_id in heldout_ids
            ]
        )
    )
    heldout_rmse_ratios = {
        compound_id: float(
            metric_lookup[(selected_id, compound_id)].weighted_rmse_hartree_per_e
            / metric_lookup[(baseline_id, compound_id)].weighted_rmse_hartree_per_e
        )
        for compound_id in heldout_ids
    }
    training_gate = bool(
        selection.relative_training_improvement
        >= float(gates["minimum_training_relative_improvement"])
    )
    heldout_objective_gate = bool(
        selected_heldout_objective < baseline_heldout_objective
    )
    heldout_record_gate = bool(
        max(heldout_rmse_ratios.values())
        <= float(gates["maximum_per_heldout_weighted_rmse_ratio_to_baseline"])
    )
    static_mep_gate = bool(
        training_gate and heldout_objective_gate and heldout_record_gate
    )

    pcm_records: list[dict[str, object]] = []
    if static_mep_gate:
        pcm_manifest = _load_json(pcm_manifest_path, name="PCMSolver preregistration")
        records = pcm_manifest.get("records")
        if not isinstance(records, list) or len(records) != int(
            gates["matched_pcmsolver_case_count"]
        ):
            raise RuntimeError("PCMSolver terminal panel identity is invalid.")
        pcm_records = [
            _pcm_case(
                record=record,
                asset_root=asset_root,
                electronic=electronic,
                embedding=selected,
                pcmsolver_library=pcmsolver_library,
                energy_budget_kcal_per_mol=float(
                    gates[
                        "maximum_absolute_fixed_source_energy_error_kcal_per_mol_per_case"
                    ]
                ),
            )
            for record in sorted(records, key=lambda item: str(item["compound_id"]))
        ]
    pcm_gate = bool(
        pcm_records
        and len(pcm_records) == int(gates["matched_pcmsolver_case_count"])
        and all(record["fixed_source_energy_gate_passed"] for record in pcm_records)
    )
    source_profile_passed = bool(static_mep_gate and pcm_gate)
    if not static_mep_gate:
        status = "fixed-radial-embedding-fails-static-mep-gate"
    elif not pcm_gate:
        status = "fixed-radial-embedding-fails-pcmsolver-energy-gate"
    else:
        status = "fixed-radial-embedding-passes-terminal-source-gate"
    measurement = {
        "protocol": {
            "selection_uses_training_mep_only": True,
            "heldout_mep_used_for_selection": False,
            "pcmsolver_energy_used_for_selection": False,
            "experimental_solvation_labels_read": False,
            "source_map": "one universal convex normalized-Gaussian mixture preserving every original q/p block",
            "static_mep_asset_reuse": (
                "Only frozen QM MEP, geometry, and geometry-only surface assets "
                "are reused. Historical legacy-evaluator MACE predictions do not "
                "gate the current analytic-evaluator source."
            ),
            "static_mep_reference": "frozen omegaB97M-V/def2-TZVPD total MEP",
            "pcmsolver_reference": "frozen matched QM total MEP and identical intrinsic PCMSolver cavity",
        },
        "candidate_grid": [
            {
                "candidate_id": candidate_id,
                "sigmas_angstrom": list(embedding.sigmas_angstrom),
                "weights": list(embedding.weights),
                "configuration_sha256": embedding.configuration_sha256,
                "embedding_distance_from_original_angstrom2": distance,
            }
            for candidate_id, embedding, distance in candidates
        ],
        "selection": selection.as_dict(),
        "selected_embedding": {
            "candidate_id": selected_id,
            "sigmas_angstrom": list(selected.sigmas_angstrom),
            "weights": list(selected.weights),
            "configuration_sha256": selected.configuration_sha256,
        },
        "static_mep_records": static_payload,
        "static_mep_gate": {
            "training_gate_passed": training_gate,
            "heldout_objective_gate_passed": heldout_objective_gate,
            "heldout_record_gate_passed": heldout_record_gate,
            "selected_heldout_objective": selected_heldout_objective,
            "baseline_heldout_objective": baseline_heldout_objective,
            "heldout_weighted_rmse_ratios_to_baseline": heldout_rmse_ratios,
            "gate_passed": static_mep_gate,
        },
        "matched_pcmsolver_records": pcm_records,
        "matched_pcmsolver_gate": {
            "executed": bool(pcm_records),
            "case_pass_count": sum(
                bool(record["fixed_source_energy_gate_passed"])
                for record in pcm_records
            ),
            "maximum_energy_absolute_error_kcal_per_mol": (
                max(
                    float(record["polarization_energy_absolute_error_kcal_per_mol"])
                    for record in pcm_records
                )
                if pcm_records
                else None
            ),
            "gate_passed": pcm_gate,
        },
        "decision": {
            "fixed_radial_source_profile_passed": source_profile_passed,
            "selected_profile_may_be_frozen": source_profile_passed,
            "phi0_ledger_decomposition_authorized": source_profile_passed,
            "phi1_delta_ledger_decomposition_authorized": False,
            "transition_to_scalar_first_or_independent_variational_polarization": (
                not source_profile_passed
            ),
            "additional_radial_patches_authorized": False,
            "public_capability_admitted": False,
        },
    }
    measurement_sha256 = canonical_json_sha256(measurement)
    repository.assert_unchanged()
    source_paths = collect_loaded_repository_sources(
        repository.root, required_paths=REQUIRED_SOURCE_PATHS
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "terminal-fixed-radial-source-research-audit",
        "status": status,
        "claim_boundary": (
            "This artifact makes one preregistered low-capacity radial-source "
            "decision. A pass would authorize only decomposition of a new "
            "operational ledger; it cannot reopen the unchanged source, prove "
            "Tier V, inherit force evidence, or admit E/F/H/V/M."
        ),
        "capabilities": NO_CAPABILITIES,
        "exact_command": shlex.join(sys.argv),
        "execution_git_head": repository.head,
        "execution_git_tree": repository.tree,
        "working_tree_clean": repository.clean,
        "source_files_sha256": committed_source_hashes(repository, source_paths),
        "external_assets": {
            "asset_root": str(asset_root),
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": sha256_file(preregistration_path),
            },
            "input_evidence": {
                "original_negative_sha256": sha256_file(original_negative),
                "farfield_positive_sha256": sha256_file(farfield_positive),
                "static_manifest_sha256": sha256_file(static_manifest_path),
                "pcmsolver_manifest_sha256": sha256_file(pcm_manifest_path),
            },
            "mace_checkpoint": checkpoint_record(checkpoint),
            "pcmsolver_library": {
                "path": str(pcmsolver_library),
                "sha256": sha256_file(pcmsolver_library),
            },
        },
        "runtime": runtime_record(),
        "device": args.device,
        "dtype": str(radial.dtype).replace("torch.", ""),
        **measurement,
        "measurement_sha256": measurement_sha256,
        "runtime_seconds": time.perf_counter() - started,
    }
    repository.assert_unchanged()
    artifact = write_external_json_artifact(repository, args.output, payload)
    repository.assert_unchanged()
    print(
        "ROUTE2_MACE_FIXED_RADIAL_EMBEDDING="
        + json.dumps(
            {
                "artifact": artifact,
                "status": status,
                "selection": selection.as_dict(),
                "selected_embedding": measurement["selected_embedding"],
                "static_mep_gate": measurement["static_mep_gate"],
                "matched_pcmsolver_gate": measurement["matched_pcmsolver_gate"],
                "decision": measurement["decision"],
                "measurement_sha256": measurement_sha256,
                "capabilities": NO_CAPABILITIES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
