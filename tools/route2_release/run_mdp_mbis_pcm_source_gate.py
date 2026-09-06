#!/usr/bin/env python3
"""Run the molecule-held-out MDP/MBIS PCM permanent-source physical gate.

The frozen experiment contains 60 neutral SPICE test-split molecules and one
deterministically selected configuration per molecule.  It reads no solvation
target, CDS value, or downstream error.  On identical SMD-water/ddPCM cavity
nodes it compares

* the original latent MACE-MDP atomwise q/p partition;
* the new molecule-held-out MBIS-supervised q/p source head; and
* independent SPICE-QM MBIS q/p and q/p/Q/O source labels.

The fixed-source ddPCM comparison uses SPICE MBIS q/p as the compact reference
through the exact same point-multipole operator.  The q/p/Q/O MEP is an
additional near-field truncation diagnostic, not a direct full-density QM MEP.
Nothing in this runner admits an energy, force, or solvation-accuracy claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

PREREG_RELATIVE_PATH = Path(
    "docs/route2/preregistrations/mdp-mbis-pcm-source-physical-gate-v1.json"
)
PREREG_SHA256 = "e3ed4ee5129c994325449bceafa36abc4f338d32e069c289c26ff14cea31cffb"
ARTIFACT_ID = "route2-mdp-mbis-pcm-source-physical-gate-v1"
RECORD_COUNT = 60
EV_TO_KCAL_MOL = 23.06054783061903


class SourceGateContractError(RuntimeError):
    """Raised when the frozen experiment contract no longer closes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceGateContractError(message)


def _sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _repository_root() -> Path:
    return REPOSITORY_ROOT


def _load_preregistration() -> tuple[Path, dict[str, Any]]:
    root = _repository_root()
    path = (root / PREREG_RELATIVE_PATH).resolve(strict=True)
    _require(_sha256_file(path) == PREREG_SHA256, "Preregistration bytes changed.")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "Preregistration must be an object.")
    _require(
        value.get("artifact")
        == "route2-mdp-mbis-pcm-source-physical-gate-preregistration-v1",
        "Wrong preregistration identity.",
    )
    _require(
        value.get("status") == "locked-before-source-mep-or-fixed-source-ddpcm-results",
        "Preregistration was not prospectively locked.",
    )
    claim = value.get("claim_boundary")
    _require(isinstance(claim, dict), "Missing claim boundary.")
    for key in (
        "pure_mace_polar_in_scope",
        "experimental_solvation_targets_read",
        "mnsol_or_freesolv_imported",
        "cds_or_standard_state_used",
        "source_or_mep_results_used_for_selection",
        "capability_admitted",
    ):
        _require(claim.get(key) is False, f"Forbidden preregistration claim: {key}.")
    _require(claim.get("hybrid_line_only") is True, "Hybrid-only scope changed.")
    return path, value


def _selected_records(prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    selection = prereg.get("selection")
    _require(isinstance(selection, dict), "Missing selection block.")
    records = selection.get("records")
    _require(
        isinstance(records, list) and len(records) == RECORD_COUNT,
        "Selection must contain exactly 60 records.",
    )
    normalized = [dict(record) for record in records]
    _require(
        _canonical_sha256(normalized) == selection.get("selection_sha256"),
        "Selection digest changed.",
    )
    return normalized


def _validate_inputs(
    prereg: Mapping[str, Any],
    *,
    dataset: Path,
    checkpoint: Path,
    source_head: Path,
) -> None:
    inputs = prereg.get("inputs")
    _require(isinstance(inputs, dict), "Missing input block.")
    for key, path in (
        ("spice_dataset_sha256", dataset),
        ("mace_mdp_checkpoint_sha256", checkpoint),
        ("mbis_source_head_sha256", source_head),
    ):
        _require(_sha256_file(path) == inputs.get(key), f"Input hash mismatch: {key}.")


def _raw_l1(charges: np.ndarray, dipoles: np.ndarray) -> np.ndarray:
    values = np.concatenate((charges[:, None], dipoles[:, (1, 2, 0)]), axis=1)
    _require(np.all(np.isfinite(values)), "Raw-l1 source is non-finite.")
    return values


def _lebedev_area_weights(
    *,
    points_bohr: np.ndarray,
    owners: np.ndarray,
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
) -> np.ndarray:
    from ase.units import Bohr
    from scipy.integrate._lebedev import lebedev_rule
    from scipy.spatial import cKDTree

    directions, rule_weights = lebedev_rule(23)
    rule = np.asarray(directions, dtype=np.float64).T
    parent_centers = positions_angstrom[owners] / Bohr
    parent_radii = radii_angstrom[owners] / Bohr
    observed = (points_bohr - parent_centers) / parent_radii[:, None]
    distance, index = cKDTree(rule).query(observed, k=1)
    _require(
        float(np.max(distance)) < 3.0e-12,
        "ddX cavity nodes do not match the frozen 194-point Lebedev rule.",
    )
    weights = np.asarray(rule_weights[index] * parent_radii**2, dtype=np.float64)
    _require(np.all(weights > 0.0), "Cavity area weights must be positive.")
    return weights


def _metric_payload(
    predicted: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    from maple.solvation.release.cartesian_multipole_mep import weighted_mep_metrics

    metric = weighted_mep_metrics(predicted, reference, weights)
    error = predicted - reference
    return metric.as_dict() | {
        "area_weight_sum_bohr2": float(np.sum(weights)),
        "area_weighted_squared_error_sum_hartree2_bohr2_per_e2": float(
            np.sum(weights * error**2)
        ),
        "area_weighted_reference_squared_sum_hartree2_bohr2_per_e2": float(
            np.sum(weights * reference**2)
        ),
    }


def _runtime_identity(device: str) -> dict[str, object]:
    import pyddx
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "pyddx": str(pyddx.__version__),
        "device": device,
    }


def _load_record_arrays(
    dataset: Path,
    selected: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    import h5py

    molecule = selected["molecule"]
    configuration = int(selected["configuration"])
    with h5py.File(dataset, "r") as handle:
        _require(molecule in handle, f"Missing SPICE molecule {molecule}.")
        group = handle[molecule]
        numbers = np.asarray(group["atomic_numbers"], dtype=np.int64).reshape(-1)
        positions_raw = np.asarray(group["positions"][configuration])
        _require(
            _array_sha256(numbers) == selected["atomic_numbers_sha256"],
            "Atomic-number identity changed.",
        )
        _require(
            _array_sha256(positions_raw) == selected["positions_raw_sha256"],
            "Geometry identity changed.",
        )
        return {
            "numbers": numbers,
            "positions_angstrom": np.asarray(positions_raw, dtype=np.float64) * 10.0,
            "charges_e": np.asarray(
                group["mbis_charges"][configuration, :, 0], dtype=np.float64
            ),
            "dipoles_eangstrom": np.asarray(
                group["mbis_dipoles"][configuration], dtype=np.float64
            )
            * 10.0,
            "quadrupoles_eangstrom2": np.asarray(
                group["mbis_quadrupoles"][configuration], dtype=np.float64
            )
            * 100.0,
            "octupoles_eangstrom3": np.asarray(
                group["mbis_octupoles"][configuration], dtype=np.float64
            )
            * 1000.0,
            "scf_dipole_eangstrom": np.asarray(
                group["scf_dipole"][configuration], dtype=np.float64
            )
            * 10.0,
        }


def _run_one(
    *,
    selection_index: int,
    selected: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    adapter: object,
    prereg: Mapping[str, Any],
    prereg_path: Path,
    dataset: Path,
    checkpoint: Path,
    source_head: Path,
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
    reference_source = _raw_l1(arrays["charges_e"], arrays["dipoles_eangstrom"])
    latent_source = _raw_l1(
        state.latent_mdp_charges_e,
        state.latent_mdp_dipoles_eangstrom,
    )
    learned_source = state.source4_raw_l1
    symbols = tuple(chemical_symbols[int(number)] for number in numbers)
    base_radii = smd_water_coulomb_radii(symbols)

    source_closure = {}
    for name, charges, dipoles, target_dipole in (
        (
            "spice_mbis_reference",
            arrays["charges_e"],
            arrays["dipoles_eangstrom"],
            arrays["scf_dipole_eangstrom"],
        ),
        (
            "original_latent_mdp",
            state.latent_mdp_charges_e,
            state.latent_mdp_dipoles_eangstrom,
            state.public_molecular_dipole_eangstrom,
        ),
        (
            "learned_mbis_head",
            state.charges_e,
            state.dipoles_eangstrom,
            state.public_molecular_dipole_eangstrom,
        ),
    ):
        reconstructed = np.sum(charges[:, None] * positions + dipoles, axis=0)
        source_closure[name] = {
            "total_charge_e": float(np.sum(charges)),
            "molecular_dipole_eangstrom": reconstructed.tolist(),
            "target_molecular_dipole_eangstrom": target_dipole.tolist(),
            "molecular_dipole_l2_error_eangstrom": float(
                np.linalg.norm(reconstructed - target_dipole)
            ),
        }

    mep_by_scale: dict[str, object] = {}
    fixed_energy: dict[str, float] | None = None
    for radius_scale in prereg["cavity_and_continuum"]["radius_scales_for_mep"]:
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
        prepared = backend.prepare(atoms, reference_source)
        points = prepared.cavity_points_bohr
        owners = prepared.cavity_parent_indices
        weights = _lebedev_area_weights(
            points_bohr=points,
            owners=owners,
            positions_angstrom=positions,
            radii_angstrom=radii,
        )
        reference_qp = prepared.permanent_point_mep(reference_source)
        analytic_qp = cartesian_atomic_multipole_potential(
            points_bohr=points,
            centers_angstrom=positions,
            charges_e=arrays["charges_e"],
            dipoles_eangstrom=arrays["dipoles_eangstrom"],
        )
        replay_error = float(np.max(np.abs(reference_qp - analytic_qp)))
        _require(replay_error < 3.0e-12, "ddX/Cartesian q/p MEP conventions differ.")
        reference_qpqo = cartesian_atomic_multipole_potential(
            points_bohr=points,
            centers_angstrom=positions,
            charges_e=arrays["charges_e"],
            dipoles_eangstrom=arrays["dipoles_eangstrom"],
            quadrupoles_eangstrom2=arrays["quadrupoles_eangstrom2"],
            octupoles_eangstrom3=arrays["octupoles_eangstrom3"],
        )
        latent_mep = prepared.permanent_point_mep(latent_source)
        learned_mep = prepared.permanent_point_mep(learned_source)
        scale_key = f"{float(radius_scale):.2f}"
        mep_by_scale[scale_key] = {
            "cavity_point_count": prepared.cavity_point_count,
            "cavity_topology_sha256": prepared.cavity_topology_sha256,
            "area_weight_sum_bohr2": float(np.sum(weights)),
            "ddx_vs_cartesian_reference_qp_max_abs_hartree_per_e": replay_error,
            "original_vs_reference_qp": _metric_payload(
                latent_mep, reference_qp, weights
            ),
            "learned_vs_reference_qp": _metric_payload(
                learned_mep, reference_qp, weights
            ),
            "original_vs_reference_qpqo": _metric_payload(
                latent_mep, reference_qpqo, weights
            ),
            "learned_vs_reference_qpqo": _metric_payload(
                learned_mep, reference_qpqo, weights
            ),
            "reference_qp_vs_reference_qpqo": _metric_payload(
                reference_qp, reference_qpqo, weights
            ),
        }
        if float(radius_scale) == float(
            prereg["cavity_and_continuum"]["fixed_source_energy_radius_scale"]
        ):
            reference_energy = prepared.fixed_permanent_source_energy_ev(
                reference_source
            )
            latent_energy = prepared.fixed_permanent_source_energy_ev(latent_source)
            learned_energy = prepared.fixed_permanent_source_energy_ev(learned_source)
            fixed_energy = {
                "reference_mbis_qp_eV": reference_energy,
                "original_latent_mdp_eV": latent_energy,
                "learned_mbis_head_eV": learned_energy,
                "original_absolute_error_kcal_mol": abs(
                    latent_energy - reference_energy
                )
                * EV_TO_KCAL_MOL,
                "learned_absolute_error_kcal_mol": abs(
                    learned_energy - reference_energy
                )
                * EV_TO_KCAL_MOL,
            }
    _require(fixed_energy is not None, "Fixed-source energy scale was not run.")

    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-record",
        "selection_index": selection_index,
        "selection_identity": dict(selected),
        "preregistration_sha256": _sha256_file(prereg_path),
        "input_sha256": {
            "dataset": _sha256_file(dataset),
            "checkpoint": _sha256_file(checkpoint),
            "source_head": _sha256_file(source_head),
        },
        "runtime": _runtime_identity(device),
        "source_state_sha256": state.state_sha256,
        "source_adapter_configuration_sha256": state.configuration_sha256,
        "source_closure": source_closure,
        "mep_by_radius_scale": mep_by_scale,
        "fixed_source_ddpcm_energy": fixed_energy,
        "claim_boundary": {
            "solvation_targets_read": False,
            "capability_admitted": False,
            "direct_full_density_qm_mep_claimed": False,
        },
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def run_records(args: argparse.Namespace) -> None:
    prereg_path, prereg = _load_preregistration()
    selected = _selected_records(prereg)
    dataset = Path(args.dataset).expanduser().resolve(strict=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    source_head = Path(args.source_head).expanduser().resolve(strict=True)
    _validate_inputs(
        prereg,
        dataset=dataset,
        checkpoint=checkpoint,
        source_head=source_head,
    )
    start, stop = int(args.start), int(args.stop)
    _require(0 <= start < stop <= len(selected), "Invalid record slice.")
    from maple.solvation.models import build_mace_mdp_mbis_source_adapter

    adapter = build_mace_mdp_mbis_source_adapter(
        checkpoint_path=checkpoint,
        source_head_path=source_head,
        device=args.device,
    )
    output = Path(args.output_dir).expanduser().resolve()
    for index in range(start, stop):
        arrays = _load_record_arrays(dataset, selected[index])
        payload = _run_one(
            selection_index=index,
            selected=selected[index],
            arrays=arrays,
            adapter=adapter,
            prereg=prereg,
            prereg_path=prereg_path,
            dataset=dataset,
            checkpoint=checkpoint,
            source_head=source_head,
            device=args.device,
        )
        path = output / "records" / f"record-{index:03d}.json"
        _write_json_exclusive(path, payload)
        print(json.dumps({"record": index, "path": str(path)}, sort_keys=True))


def _global_metric(
    records: list[Mapping[str, Any]], scale: str, key: str
) -> dict[str, float]:
    rows = [record["mep_by_radius_scale"][scale][key] for record in records]
    weight = sum(float(row["area_weight_sum_bohr2"]) for row in rows)
    error2 = sum(
        float(row["area_weighted_squared_error_sum_hartree2_bohr2_per_e2"])
        for row in rows
    )
    reference2 = sum(
        float(row["area_weighted_reference_squared_sum_hartree2_bohr2_per_e2"])
        for row in rows
    )
    return {
        "global_area_weighted_rmse_hartree_per_e": float(np.sqrt(error2 / weight)),
        "global_reference_rms_hartree_per_e": float(np.sqrt(reference2 / weight)),
        "global_relative_rmse": float(
            np.sqrt(error2 / max(reference2, np.finfo(float).tiny))
        ),
        "configuration_median_relative_rmse": float(
            np.median([row["relative_root_mean_square_error"] for row in rows])
        ),
        "configuration_q95_relative_rmse": float(
            np.quantile([row["relative_root_mean_square_error"] for row in rows], 0.95)
        ),
    }


def aggregate(args: argparse.Namespace) -> None:
    prereg_path, prereg = _load_preregistration()
    selected = _selected_records(prereg)
    output = Path(args.output_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for index, identity in enumerate(selected):
        path = output / "records" / f"record-{index:03d}.json"
        _require(path.is_file(), f"Missing record {index}.")
        record = json.loads(path.read_text(encoding="utf-8"))
        digest = record.pop("record_sha256", None)
        _require(digest == _canonical_sha256(record), f"Record {index} hash mismatch.")
        record["record_sha256"] = digest
        _require(record["selection_index"] == index, "Record index mismatch.")
        _require(record["selection_identity"] == identity, "Record identity mismatch.")
        _require(
            record["preregistration_sha256"] == _sha256_file(prereg_path),
            "Record preregistration mismatch.",
        )
        records.append(record)

    scales = [
        f"{float(scale):.2f}"
        for scale in prereg["cavity_and_continuum"]["radius_scales_for_mep"]
    ]
    mep_summary: dict[str, object] = {}
    qp_improved = 0
    qpo_no_worsening = True
    for scale in scales:
        original_qp = _global_metric(records, scale, "original_vs_reference_qp")
        learned_qp = _global_metric(records, scale, "learned_vs_reference_qp")
        original_qpo = _global_metric(records, scale, "original_vs_reference_qpqo")
        learned_qpo = _global_metric(records, scale, "learned_vs_reference_qpqo")
        truncation = _global_metric(records, scale, "reference_qp_vs_reference_qpqo")
        if scale == "1.00":
            qp_improved = sum(
                record["mep_by_radius_scale"][scale]["learned_vs_reference_qp"][
                    "root_mean_square_error_hartree_per_e"
                ]
                < record["mep_by_radius_scale"][scale]["original_vs_reference_qp"][
                    "root_mean_square_error_hartree_per_e"
                ]
                for record in records
            )
        qpo_no_worsening = qpo_no_worsening and (
            learned_qpo["global_area_weighted_rmse_hartree_per_e"]
            <= original_qpo["global_area_weighted_rmse_hartree_per_e"]
        )
        mep_summary[scale] = {
            "original_vs_reference_qp": original_qp,
            "learned_vs_reference_qp": learned_qp,
            "original_vs_reference_qpqo": original_qpo,
            "learned_vs_reference_qpqo": learned_qpo,
            "reference_qp_truncation_vs_qpqo": truncation,
        }

    original_energy_errors = np.asarray(
        [
            record["fixed_source_ddpcm_energy"]["original_absolute_error_kcal_mol"]
            for record in records
        ]
    )
    learned_energy_errors = np.asarray(
        [
            record["fixed_source_ddpcm_energy"]["learned_absolute_error_kcal_mol"]
            for record in records
        ]
    )
    primary_scale = mep_summary["1.00"]
    original_qp_rmse = primary_scale["original_vs_reference_qp"][
        "global_area_weighted_rmse_hartree_per_e"
    ]
    learned_qp_rmse = primary_scale["learned_vs_reference_qp"][
        "global_area_weighted_rmse_hartree_per_e"
    ]
    mep_reduction = 1.0 - learned_qp_rmse / original_qp_rmse
    energy_reduction = 1.0 - float(np.mean(learned_energy_errors)) / float(
        np.mean(original_energy_errors)
    )
    rules = prereg["prospective_decision_rules"]
    gates = {
        "primary_qp_mep_global_rmse_reduction": (
            mep_reduction
            >= rules["primary_qp_mep_global_rmse_reduction_fraction_minimum"]
        ),
        "primary_qp_mep_paired_configuration_fraction": (
            qp_improved / len(records)
            >= rules["primary_qp_mep_paired_configuration_improvement_fraction_minimum"]
        ),
        "fixed_source_ddpcm_energy_mae_reduction": (
            energy_reduction
            >= rules["fixed_source_ddpcm_energy_mae_reduction_fraction_minimum"]
        ),
        "higher_multipole_mep_no_worsening": qpo_no_worsening,
    }
    report: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "status": "pass" if all(gates.values()) else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": _sha256_file(prereg_path),
        "record_count": len(records),
        "unique_molecule_count": len(
            {r["selection_identity"]["molecule"] for r in records}
        ),
        "mep_summary_by_radius_scale": mep_summary,
        "fixed_source_ddpcm_energy": {
            "original_mae_kcal_mol": float(np.mean(original_energy_errors)),
            "learned_mae_kcal_mol": float(np.mean(learned_energy_errors)),
            "original_q95_absolute_error_kcal_mol": float(
                np.quantile(original_energy_errors, 0.95)
            ),
            "learned_q95_absolute_error_kcal_mol": float(
                np.quantile(learned_energy_errors, 0.95)
            ),
            "original_maximum_absolute_error_kcal_mol": float(
                np.max(original_energy_errors)
            ),
            "learned_maximum_absolute_error_kcal_mol": float(
                np.max(learned_energy_errors)
            ),
            "paired_improved_count": int(
                np.sum(learned_energy_errors < original_energy_errors)
            ),
            "mae_reduction_fraction": energy_reduction,
        },
        "primary_qp_mep_rmse_reduction_fraction": mep_reduction,
        "primary_qp_mep_paired_improved_count": qp_improved,
        "primary_qp_mep_paired_improved_fraction": qp_improved / len(records),
        "prospective_gates": gates,
        "claim_boundary": {
            "solvation_targets_read": False,
            "capability_admitted": False,
            "hybrid_integration_admitted": all(gates.values()),
            "direct_full_density_qm_mep_claimed": False,
        },
        "record_sha256": [record["record_sha256"] for record in records],
    }
    report["aggregate_sha256"] = _canonical_sha256(report)
    path = output / "aggregate.json"
    _write_json_exclusive(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("records", "aggregate"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset",
        default=str(
            Path.home() / ".cache/maple/spice2-ntc1000-v1.1/"
            "spice_2_dataset_v1.1_ntc_1000.hdf5"
        ),
    )
    parser.add_argument(
        "--checkpoint", default=str(Path.home() / ".cache/mace/MACE-MDP.model")
    )
    parser.add_argument(
        "--source-head",
        default=str(
            _repository_root()
            / "docs/route2/evidence/mdp-mbis-source-head-prototype-20260817/"
            "source-head.npz"
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
