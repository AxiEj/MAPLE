#!/usr/bin/env python3
"""Decompose permanent-source error into anchor and multipole-order terms.

This target-free follow-up reuses the exact 60-molecule selection, cavity, and
runtime contract of ``run_mdp_mbis_pcm_source_gate.py``.  It answers two
prospectively frozen architecture questions:

* Is forcing exact MBIS q/p to the frozen MACE-MDP molecular dipole harmless?
* Does adding MBIS quadrupoles remove at least half of the q/p-to-q/p/Q/O
  cavity-MEP truncation error at every preregistered cavity radius?

No experimental solvation target, CDS term, or public capability is read or
admitted.  The v1 learned q/p head is retained only as a diagnostic comparator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.route2_release import run_mdp_mbis_pcm_source_gate as parent

PREREG_RELATIVE_PATH = Path(
    "docs/route2/preregistrations/mdp-mbis-source-error-decomposition-v1.json"
)
PREREG_SHA256 = "70aab9945717dee5a7a1b3d59f392a919a450f686826daf89a663a4695d91179"
ARTIFACT_ID = "route2-mdp-mbis-source-error-decomposition-v1"
RECORD_COUNT = 60


def _load_preregistrations() -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    root = parent._repository_root()
    parent_path, parent_prereg = parent._load_preregistration()
    path = (root / PREREG_RELATIVE_PATH).resolve(strict=True)
    parent._require(
        parent._sha256_file(path) == PREREG_SHA256,
        "Source-error decomposition preregistration bytes changed.",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    parent._require(
        isinstance(value, dict), "Decomposition preregistration must be an object."
    )
    parent._require(
        value.get("artifact")
        == "route2-mdp-mbis-source-error-decomposition-preregistration-v1",
        "Wrong decomposition preregistration identity.",
    )
    parent._require(
        value.get("status") == "locked-before-anchor-or-order-decomposition-results",
        "Decomposition preregistration was not prospectively locked.",
    )
    claim = value.get("claim_boundary")
    parent._require(isinstance(claim, dict), "Missing decomposition claim boundary.")
    for key in (
        "experimental_solvation_targets_read",
        "fitting_or_calibration_performed",
        "hybrid_integration_admitted",
        "mnsol_freesolv_or_cds_read",
        "public_capability_admitted",
    ):
        parent._require(
            claim.get(key) is False, f"Forbidden decomposition claim: {key}."
        )
    parent._require(
        claim.get("source_diagnostic_only") is True, "Source-only claim changed."
    )

    binding = value.get("parent_source_gate")
    parent._require(isinstance(binding, dict), "Missing parent source-gate binding.")
    parent._require(
        binding.get("preregistration_sha256") == parent._sha256_file(parent_path),
        "Parent preregistration binding changed.",
    )
    parent._require(
        binding.get("selection_count") == RECORD_COUNT, "Selection count changed."
    )
    aggregate_path = (root / str(binding.get("aggregate_path"))).resolve(strict=True)
    parent._require(
        parent._sha256_file(aggregate_path) == binding.get("aggregate_sha256"),
        "Parent aggregate bytes changed.",
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    parent._require(
        aggregate.get("status") == binding.get("parent_status") == "fail",
        "Parent status changed.",
    )
    parent._require(
        aggregate.get("record_count") == RECORD_COUNT, "Parent record count changed."
    )
    parent._require(
        aggregate.get("preregistration_sha256") == parent._sha256_file(parent_path),
        "Parent aggregate/preregistration binding changed.",
    )
    return path, value, parent_path, parent_prereg


def _project_qp_to_charge_and_dipole(
    *,
    positions_angstrom: object,
    charges_e: object,
    dipoles_eangstrom: object,
    target_charge_e: float,
    target_dipole_eangstrom: object,
    charge_sigma_e: float,
    dipole_sigma_eangstrom: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Metric-project q/p to exact molecular charge and dipole constraints."""

    positions = np.asarray(positions_angstrom, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64)
    dipoles = np.asarray(dipoles_eangstrom, dtype=np.float64)
    target_dipole = np.asarray(target_dipole_eangstrom, dtype=np.float64)
    count = len(charges)
    if (
        count < 1
        or positions.shape != (count, 3)
        or dipoles.shape != (count, 3)
        or target_dipole.shape != (3,)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(charges))
        or not np.all(np.isfinite(dipoles))
        or not np.all(np.isfinite(target_dipole))
        or not np.isfinite(target_charge_e)
        or not np.isfinite(charge_sigma_e)
        or not np.isfinite(dipole_sigma_eangstrom)
        or charge_sigma_e <= 0.0
        or dipole_sigma_eangstrom <= 0.0
    ):
        raise ValueError(
            "Projection inputs and metric scales must be finite and compatible."
        )

    source = np.concatenate((charges, dipoles.reshape(-1)))
    constraint = np.zeros((4, 4 * count), dtype=np.float64)
    constraint[0, :count] = 1.0
    constraint[1:, :count] = positions.T
    for atom_index in range(count):
        start = count + 3 * atom_index
        constraint[1:, start : start + 3] = np.eye(3)
    covariance = np.concatenate(
        (
            np.full(count, charge_sigma_e**2),
            np.full(3 * count, dipole_sigma_eangstrom**2),
        )
    )
    target = np.concatenate(([float(target_charge_e)], target_dipole))
    residual = target - constraint @ source
    schur = (constraint * covariance[None, :]) @ constraint.T
    condition = float(np.linalg.cond(schur))
    if not np.isfinite(condition):
        raise RuntimeError("Projection Schur matrix is singular.")
    multipliers = np.linalg.solve(schur, residual)
    projected = source + covariance * (constraint.T @ multipliers)
    closure = constraint @ projected
    if not np.allclose(closure, target, rtol=0.0, atol=2.0e-12):
        raise RuntimeError("Projected source does not close charge/dipole constraints.")
    return projected[:count], projected[count:].reshape(count, 3), condition


def _source_head_metric_scales(path: Path) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as payload:
        keys = frozenset(payload.files)
        parent._require(
            {"charge_sigma", "dipole_sigma"}.issubset(keys),
            "Source-head projection metric is missing.",
        )
        charge_sigma = float(payload["charge_sigma"])
        dipole_sigma = float(payload["dipole_sigma"])
    parent._require(
        np.isfinite(charge_sigma)
        and charge_sigma > 0.0
        and np.isfinite(dipole_sigma)
        and dipole_sigma > 0.0,
        "Source-head projection metric is invalid.",
    )
    return charge_sigma, dipole_sigma


def _molecular_dipole(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    dipoles_eangstrom: np.ndarray,
) -> np.ndarray:
    result = np.sum(charges_e[:, None] * positions_angstrom + dipoles_eangstrom, axis=0)
    parent._require(
        result.shape == (3,) and np.all(np.isfinite(result)),
        "Molecular dipole is invalid.",
    )
    return result


def _run_one(
    *,
    selection_index: int,
    selected: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    adapter: object,
    decomposition_prereg: Mapping[str, Any],
    decomposition_prereg_path: Path,
    parent_prereg: Mapping[str, Any],
    parent_prereg_path: Path,
    dataset: Path,
    checkpoint: Path,
    source_head: Path,
    charge_sigma: float,
    dipole_sigma: float,
    device: str,
) -> dict[str, object]:
    from ase import Atoms
    from ase.data import chemical_symbols
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_water_coulomb_radii,
    )
    from maple.solvation.continuum import SeparatedSourceDDXBackend
    from maple.solvation.release.cartesian_multipole_mep import (
        cartesian_atomic_multipole_potential,
    )

    numbers = arrays["numbers"]
    positions = arrays["positions_angstrom"]
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        info={"charge": 0, "multiplicity": 1},
    )
    state = adapter.evaluate_state(atoms)
    reference_qp_source = parent._raw_l1(
        arrays["charges_e"], arrays["dipoles_eangstrom"]
    )
    learned_source = state.source4_raw_l1
    anchored_charges, anchored_dipoles, anchor_condition = (
        _project_qp_to_charge_and_dipole(
            positions_angstrom=positions,
            charges_e=arrays["charges_e"],
            dipoles_eangstrom=arrays["dipoles_eangstrom"],
            target_charge_e=0.0,
            target_dipole_eangstrom=state.public_molecular_dipole_eangstrom,
            charge_sigma_e=charge_sigma,
            dipole_sigma_eangstrom=dipole_sigma,
        )
    )
    anchored_source = parent._raw_l1(anchored_charges, anchored_dipoles)
    symbols = tuple(chemical_symbols[int(number)] for number in numbers)
    base_radii = smd_water_coulomb_radii(symbols)

    mbis_dipole = _molecular_dipole(
        positions,
        arrays["charges_e"],
        arrays["dipoles_eangstrom"],
    )
    anchor_dipole = _molecular_dipole(positions, anchored_charges, anchored_dipoles)
    learned_dipole = _molecular_dipole(
        positions,
        state.charges_e,
        state.dipoles_eangstrom,
    )
    source_closure = {
        "mbis_total_charge_e": float(np.sum(arrays["charges_e"])),
        "mbis_molecular_dipole_eangstrom": mbis_dipole.tolist(),
        "spice_scf_dipole_eangstrom": arrays["scf_dipole_eangstrom"].tolist(),
        "mbis_vs_spice_scf_dipole_l2_eangstrom": float(
            np.linalg.norm(mbis_dipole - arrays["scf_dipole_eangstrom"])
        ),
        "mdp_public_molecular_dipole_eangstrom": state.public_molecular_dipole_eangstrom.tolist(),
        "mdp_vs_mbis_molecular_dipole_l2_eangstrom": float(
            np.linalg.norm(state.public_molecular_dipole_eangstrom - mbis_dipole)
        ),
        "anchor_total_charge_e": float(np.sum(anchored_charges)),
        "anchor_molecular_dipole_eangstrom": anchor_dipole.tolist(),
        "anchor_constraint_l2_error": float(
            np.linalg.norm(
                np.concatenate(
                    (
                        [np.sum(anchored_charges)],
                        anchor_dipole - state.public_molecular_dipole_eangstrom,
                    )
                )
            )
        ),
        "learned_molecular_dipole_eangstrom": learned_dipole.tolist(),
        "learned_constraint_l2_error": float(
            np.linalg.norm(learned_dipole - state.public_molecular_dipole_eangstrom)
        ),
        "anchor_projection_condition_number": anchor_condition,
    }

    mep_by_scale: dict[str, object] = {}
    fixed_energy: dict[str, float] | None = None
    for radius_scale in parent_prereg["cavity_and_continuum"]["radius_scales_for_mep"]:
        radii = base_radii * float(radius_scale)
        backend = SeparatedSourceDDXBackend(
            symbols,
            radii,
            continuum_model="pcm",
            dielectric=78.39,
            lmax=8,
            n_lebedev=194,
            solver_tolerance=1.0e-12,
            eta=0.1,
            n_proc=1,
        )
        prepared = backend.prepare(atoms, reference_qp_source)
        points = prepared.cavity_points_bohr
        owners = prepared.cavity_parent_indices
        weights = parent._lebedev_area_weights(
            points_bohr=points,
            owners=owners,
            positions_angstrom=positions,
            radii_angstrom=radii,
        )
        reference_qp = prepared.permanent_point_mep(reference_qp_source)
        reference_qpq = cartesian_atomic_multipole_potential(
            points_bohr=points,
            centers_angstrom=positions,
            charges_e=arrays["charges_e"],
            dipoles_eangstrom=arrays["dipoles_eangstrom"],
            quadrupoles_eangstrom2=arrays["quadrupoles_eangstrom2"],
        )
        reference_qpqo = cartesian_atomic_multipole_potential(
            points_bohr=points,
            centers_angstrom=positions,
            charges_e=arrays["charges_e"],
            dipoles_eangstrom=arrays["dipoles_eangstrom"],
            quadrupoles_eangstrom2=arrays["quadrupoles_eangstrom2"],
            octupoles_eangstrom3=arrays["octupoles_eangstrom3"],
        )
        anchored_mep = prepared.permanent_point_mep(anchored_source)
        learned_mep = prepared.permanent_point_mep(learned_source)
        scale_key = f"{float(radius_scale):.2f}"
        mep_by_scale[scale_key] = {
            "cavity_point_count": prepared.cavity_point_count,
            "cavity_topology_sha256": prepared.cavity_topology_sha256,
            "area_weight_sum_bohr2": float(np.sum(weights)),
            "reference_qp_vs_reference_qpq": parent._metric_payload(
                reference_qp, reference_qpq, weights
            ),
            "reference_qpq_vs_reference_qpqo": parent._metric_payload(
                reference_qpq, reference_qpqo, weights
            ),
            "reference_qp_vs_reference_qpqo": parent._metric_payload(
                reference_qp, reference_qpqo, weights
            ),
            "oracle_anchor_vs_reference_qp": parent._metric_payload(
                anchored_mep, reference_qp, weights
            ),
            "learned_vs_oracle_anchor": parent._metric_payload(
                learned_mep, anchored_mep, weights
            ),
            "learned_vs_reference_qp": parent._metric_payload(
                learned_mep, reference_qp, weights
            ),
        }
        if float(radius_scale) == float(
            parent_prereg["cavity_and_continuum"]["fixed_source_energy_radius_scale"]
        ):
            reference_energy = prepared.fixed_permanent_source_energy_ev(
                reference_qp_source
            )
            anchor_energy = prepared.fixed_permanent_source_energy_ev(anchored_source)
            learned_energy = prepared.fixed_permanent_source_energy_ev(learned_source)
            fixed_energy = {
                "reference_mbis_qp_eV": reference_energy,
                "oracle_anchor_eV": anchor_energy,
                "learned_v1_eV": learned_energy,
                "oracle_anchor_absolute_error_kcal_mol": abs(
                    anchor_energy - reference_energy
                )
                * parent.EV_TO_KCAL_MOL,
                "learned_v1_absolute_error_vs_reference_kcal_mol": abs(
                    learned_energy - reference_energy
                )
                * parent.EV_TO_KCAL_MOL,
                "learned_v1_absolute_difference_vs_oracle_kcal_mol": abs(
                    learned_energy - anchor_energy
                )
                * parent.EV_TO_KCAL_MOL,
            }
    parent._require(fixed_energy is not None, "Fixed-source energy scale was not run.")

    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-record",
        "selection_index": selection_index,
        "selection_identity": dict(selected),
        "decomposition_preregistration_sha256": parent._sha256_file(
            decomposition_prereg_path
        ),
        "parent_preregistration_sha256": parent._sha256_file(parent_prereg_path),
        "input_sha256": {
            "dataset": parent._sha256_file(dataset),
            "checkpoint": parent._sha256_file(checkpoint),
            "source_head": parent._sha256_file(source_head),
        },
        "runtime": parent._runtime_identity(device),
        "source_state_sha256": state.state_sha256,
        "source_adapter_configuration_sha256": state.configuration_sha256,
        "projection_metric": {
            "charge_sigma_e": charge_sigma,
            "dipole_sigma_eangstrom": dipole_sigma,
        },
        "source_closure": source_closure,
        "mep_by_radius_scale": mep_by_scale,
        "fixed_source_ddpcm_energy": fixed_energy,
        "claim_boundary": dict(decomposition_prereg["claim_boundary"]),
    }
    payload["record_sha256"] = parent._canonical_sha256(payload)
    return payload


def run_records(args: argparse.Namespace) -> None:
    decomp_path, decomp, parent_path, parent_prereg = _load_preregistrations()
    selected = parent._selected_records(parent_prereg)
    dataset = Path(args.dataset).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    source_head = Path(args.source_head).expanduser().resolve(strict=True)
    parent._validate_inputs(
        parent_prereg,
        dataset=dataset,
        checkpoint=checkpoint,
        source_head=source_head,
    )
    charge_sigma, dipole_sigma = _source_head_metric_scales(source_head)
    start, stop = int(args.start), int(args.stop)
    parent._require(0 <= start < stop <= len(selected), "Invalid record slice.")
    from maple.solvation.models import build_mace_mdp_mbis_source_adapter

    adapter = build_mace_mdp_mbis_source_adapter(
        checkpoint_path=checkpoint,
        source_head_path=source_head,
        device=args.device,
    )
    output = Path(args.output_dir).expanduser().resolve()
    for index in range(start, stop):
        arrays = parent._load_record_arrays(dataset, selected[index])
        payload = _run_one(
            selection_index=index,
            selected=selected[index],
            arrays=arrays,
            adapter=adapter,
            decomposition_prereg=decomp,
            decomposition_prereg_path=decomp_path,
            parent_prereg=parent_prereg,
            parent_prereg_path=parent_path,
            dataset=dataset,
            checkpoint=checkpoint,
            source_head=source_head,
            charge_sigma=charge_sigma,
            dipole_sigma=dipole_sigma,
            device=args.device,
        )
        path = output / "records" / f"record-{index:03d}.json"
        parent._write_json_exclusive(path, payload)
        print(json.dumps({"record": index, "path": str(path)}, sort_keys=True))


def _architecture_decision(
    *,
    qp_to_qpqo_rmse: Mapping[str, float],
    qpq_to_qpqo_rmse: Mapping[str, float],
    anchor_relative_rmse: Mapping[str, float],
    anchor_energy_mae_kcal_mol: float,
) -> dict[str, object]:
    scales = tuple(sorted(qp_to_qpqo_rmse))
    if scales != tuple(sorted(qpq_to_qpqo_rmse)) or scales != tuple(
        sorted(anchor_relative_rmse)
    ):
        raise ValueError(
            "Architecture decision metrics use inconsistent radius scales."
        )
    reductions: dict[str, float] = {}
    for scale in scales:
        baseline = float(qp_to_qpqo_rmse[scale])
        residual = float(qpq_to_qpqo_rmse[scale])
        if not np.isfinite(baseline) or baseline <= 0.0 or not np.isfinite(residual):
            raise ValueError(
                "Multipole-order RMSE values must be finite and compatible."
            )
        reductions[scale] = 1.0 - residual / baseline
    quadrupole_allowed = all(value >= 0.5 for value in reductions.values())
    anchor_mep_pass = all(
        np.isfinite(float(value)) and float(value) <= 0.05
        for value in anchor_relative_rmse.values()
    )
    anchor_energy_pass = (
        np.isfinite(anchor_energy_mae_kcal_mol) and anchor_energy_mae_kcal_mol <= 1.0
    )
    anchor_allowed = anchor_mep_pass and anchor_energy_pass
    if quadrupole_allowed:
        representation_decision = "q/p/Q-head-eligible-for-prototype"
    else:
        representation_decision = (
            "q/p/Q-terminally-insufficient-use-l3-or-density-potential-basis"
        )
    if anchor_allowed:
        anchor_decision = "hard-MDP-molecular-dipole-anchor-eligible"
    else:
        anchor_decision = "QM-dipole-closure-required-MDP-dipole-auxiliary-only"
    return {
        "quadrupole_rmse_reduction_fraction_by_radius_scale": reductions,
        "quadrupole_head_allowed": quadrupole_allowed,
        "hard_anchor_mep_gate": anchor_mep_pass,
        "hard_anchor_energy_gate": anchor_energy_pass,
        "hard_anchor_allowed": anchor_allowed,
        "representation_decision": representation_decision,
        "anchor_decision": anchor_decision,
    }


def aggregate(args: argparse.Namespace) -> None:
    decomp_path, decomp, parent_path, parent_prereg = _load_preregistrations()
    selected = parent._selected_records(parent_prereg)
    output = Path(args.output_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    runtime: object | None = None
    input_sha256: object | None = None
    projection_metric: object | None = None
    for index, identity in enumerate(selected):
        path = output / "records" / f"record-{index:03d}.json"
        parent._require(path.is_file(), f"Missing record {index}.")
        record = json.loads(path.read_text(encoding="utf-8"))
        digest = record.pop("record_sha256", None)
        parent._require(
            digest == parent._canonical_sha256(record), f"Record {index} hash mismatch."
        )
        record["record_sha256"] = digest
        parent._require(record["selection_index"] == index, "Record index mismatch.")
        parent._require(
            record["selection_identity"] == identity, "Record identity mismatch."
        )
        parent._require(
            record["decomposition_preregistration_sha256"]
            == parent._sha256_file(decomp_path),
            "Record decomposition-preregistration mismatch.",
        )
        parent._require(
            record["parent_preregistration_sha256"] == parent._sha256_file(parent_path),
            "Record parent-preregistration mismatch.",
        )
        if runtime is None:
            runtime = record["runtime"]
            input_sha256 = record["input_sha256"]
            projection_metric = record["projection_metric"]
        parent._require(record["runtime"] == runtime, "Record runtime mismatch.")
        parent._require(
            record["input_sha256"] == input_sha256, "Record input mismatch."
        )
        parent._require(
            record["projection_metric"] == projection_metric,
            "Record projection-metric mismatch.",
        )
        records.append(record)

    scales = [
        f"{float(scale):.2f}"
        for scale in parent_prereg["cavity_and_continuum"]["radius_scales_for_mep"]
    ]
    summary_by_scale: dict[str, object] = {}
    qp_to_qpqo_rmse: dict[str, float] = {}
    qpq_to_qpqo_rmse: dict[str, float] = {}
    anchor_relative_rmse: dict[str, float] = {}
    for scale in scales:
        qp_qpq = parent._global_metric(records, scale, "reference_qp_vs_reference_qpq")
        qpq_qpqo = parent._global_metric(
            records, scale, "reference_qpq_vs_reference_qpqo"
        )
        qp_qpqo = parent._global_metric(
            records, scale, "reference_qp_vs_reference_qpqo"
        )
        anchor_qp = parent._global_metric(
            records, scale, "oracle_anchor_vs_reference_qp"
        )
        learned_anchor = parent._global_metric(
            records, scale, "learned_vs_oracle_anchor"
        )
        learned_qp = parent._global_metric(records, scale, "learned_vs_reference_qp")
        qp_to_qpqo_rmse[scale] = qp_qpqo["global_area_weighted_rmse_hartree_per_e"]
        qpq_to_qpqo_rmse[scale] = qpq_qpqo["global_area_weighted_rmse_hartree_per_e"]
        anchor_relative_rmse[scale] = anchor_qp["global_relative_rmse"]
        summary_by_scale[scale] = {
            "reference_qp_vs_reference_qpq": qp_qpq,
            "reference_qpq_vs_reference_qpqo": qpq_qpqo,
            "reference_qp_vs_reference_qpqo": qp_qpqo,
            "oracle_anchor_vs_reference_qp": anchor_qp,
            "learned_vs_oracle_anchor": learned_anchor,
            "learned_vs_reference_qp": learned_qp,
        }

    anchor_energy_errors = np.asarray(
        [
            record["fixed_source_ddpcm_energy"]["oracle_anchor_absolute_error_kcal_mol"]
            for record in records
        ],
        dtype=np.float64,
    )
    learned_reference_errors = np.asarray(
        [
            record["fixed_source_ddpcm_energy"][
                "learned_v1_absolute_error_vs_reference_kcal_mol"
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    learned_anchor_differences = np.asarray(
        [
            record["fixed_source_ddpcm_energy"][
                "learned_v1_absolute_difference_vs_oracle_kcal_mol"
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    anchor_energy_mae = float(np.mean(anchor_energy_errors))
    decision = _architecture_decision(
        qp_to_qpqo_rmse=qp_to_qpqo_rmse,
        qpq_to_qpqo_rmse=qpq_to_qpqo_rmse,
        anchor_relative_rmse=anchor_relative_rmse,
        anchor_energy_mae_kcal_mol=anchor_energy_mae,
    )
    report: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "status": "complete-terminal-architecture-decision",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decomposition_preregistration_sha256": parent._sha256_file(decomp_path),
        "parent_preregistration_sha256": parent._sha256_file(parent_path),
        "record_count": len(records),
        "unique_molecule_count": len(
            {record["selection_identity"]["molecule"] for record in records}
        ),
        "runtime": runtime,
        "input_sha256": input_sha256,
        "projection_metric": projection_metric,
        "mep_summary_by_radius_scale": summary_by_scale,
        "fixed_source_ddpcm_energy": {
            "oracle_anchor_mae_kcal_mol": anchor_energy_mae,
            "oracle_anchor_q95_absolute_error_kcal_mol": float(
                np.quantile(anchor_energy_errors, 0.95)
            ),
            "oracle_anchor_maximum_absolute_error_kcal_mol": float(
                np.max(anchor_energy_errors)
            ),
            "learned_v1_mae_vs_reference_kcal_mol": float(
                np.mean(learned_reference_errors)
            ),
            "learned_v1_mae_vs_oracle_kcal_mol": float(
                np.mean(learned_anchor_differences)
            ),
        },
        "architecture_decision": decision,
        "claim_boundary": dict(decomp["claim_boundary"]),
        "record_sha256": [record["record_sha256"] for record in records],
    }
    report["aggregate_sha256"] = parent._canonical_sha256(report)
    path = output / "aggregate.json"
    parent._write_json_exclusive(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("records", "aggregate"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset",
        default=str(
            Path.home()
            / ".cache/maple/spice2-ntc1000-v1.1/spice_2_dataset_v1.1_ntc_1000.hdf5"
        ),
    )
    parser.add_argument(
        "--checkpoint", default=str(Path.home() / ".cache/mace/MACE-MDP.model")
    )
    parser.add_argument(
        "--source-head",
        default=str(
            parent._repository_root()
            / "docs/route2/evidence/mdp-mbis-source-head-prototype-20260817/source-head.npz"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=RECORD_COUNT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mode == "records":
        run_records(args)
    else:
        aggregate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
