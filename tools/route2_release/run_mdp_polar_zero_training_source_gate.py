#!/usr/bin/env python3
"""Evaluate a finite no-training MACE-MDP/MACE-POLAR permanent-source set.

The experiment changes neither official checkpoint and reads no experimental
solvation target.  It factorizes two questions on the frozen 60-molecule
SPICE/MBIS panel:

* atomwise source topology: latent MACE-MDP versus MACE-POLAR at zero field;
* near-field kernel: point multipoles versus the checkpoint-native 1.5 A
  Gaussian density.

The primary candidate projects the zero-field MACE-POLAR source to the exact
MACE-MDP total charge and public molecular dipole using the analytic Gaussian
Coulomb metric.  MBIS q/p and q/p/Q/O are evaluation-only references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release import run_mdp_mbis_pcm_source_gate as parent  # noqa: E402

PREREGISTRATION = REPO_ROOT / (
    "docs/route2/preregistrations/" "mdp-polar-zero-training-source-gate-v1.json"
)
DATASET = Path(
    "/home/axie/.cache/maple/spice2-ntc1000-v1.1/" "spice_2_dataset_v1.1_ntc_1000.hdf5"
)
MDP_CHECKPOINT = Path("/home/axie/.cache/mace/MACE-MDP.model")
POLAR_CHECKPOINT = Path("/home/axie/.cache/mace/MACEPOLAR1Mmodel")
ARTIFACT_ID = "route2-mdp-polar-zero-training-source-gate-v1"
PRIMARY_CANDIDATE = "polar_zero_mdp_qmu_gaussian"
CANDIDATES = (
    "mdp_latent_point",
    "mdp_latent_gaussian",
    "polar_zero_point",
    "polar_zero_gaussian",
    "polar_zero_mdp_qmu_point",
    PRIMARY_CANDIDATE,
)
REFERENCE_NAMES = ("reference_qp", "reference_qpqo")
EV_TO_KCAL_MOL = 23.06054783061903


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_preregistration() -> tuple[Path, dict[str, Any]]:
    path = PREREGISTRATION.resolve(strict=True)
    value = json.loads(path.read_text())
    parent._require(
        value.get("artifact")
        == "route2-mdp-polar-zero-training-source-gate-preregistration-v1",
        "Unknown no-training source-gate preregistration.",
    )
    parent._require(value.get("schema_version") == 1, "Schema version changed.")
    parent._require(
        value.get("status") == "locked-before-any-candidate-mep-or-energy-result",
        "No-training source gate is not prospectively locked.",
    )
    boundary = value.get("claim_boundary")
    parent._require(isinstance(boundary, dict), "Missing claim boundary.")
    expected_boundary = {
        "capability_admitted": False,
        "cds_or_standard_state_used": False,
        "experimental_solvation_targets_read": False,
        "hybrid_line_only": True,
        "mbis_labels_used_for_evaluation_only": True,
        "mnsol_or_freesolv_imported": False,
        "post_training_or_fitting_performed": False,
        "pure_mace_polar_in_scope": False,
        "solvation_result_used_to_select_projection_or_kernel": False,
    }
    parent._require(boundary == expected_boundary, "Claim boundary changed.")
    parent._require(
        tuple(value.get("candidate_order", ())) == CANDIDATES,
        "Candidate set/order changed.",
    )
    parent._require(
        value.get("primary_candidate") == PRIMARY_CANDIDATE,
        "Primary candidate changed.",
    )
    return path, value


def _validate_inputs(
    prereg: Mapping[str, Any],
    *,
    dataset: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
) -> None:
    inputs = prereg.get("inputs")
    parent._require(isinstance(inputs, dict), "Missing input bindings.")
    for key, path in (
        ("spice_dataset_sha256", dataset),
        ("mace_mdp_checkpoint_sha256", mdp_checkpoint),
        ("mace_polar_checkpoint_sha256", polar_checkpoint),
        (
            "parent_source_gate_preregistration_sha256",
            REPO_ROOT / "docs/route2/preregistrations/"
            "mdp-mbis-pcm-source-physical-gate-v1.json",
        ),
        (
            "gaussian_projection_implementation_sha256",
            REPO_ROOT / "maple/solvation/coupling/gaussian_coulomb_projection.py",
        ),
        ("runner_sha256", Path(__file__).resolve()),
    ):
        parent._require(
            parent._sha256_file(path) == inputs.get(key),
            f"Input hash mismatch: {key}.",
        )


def _selected_records(prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    binding = prereg.get("selection")
    parent._require(isinstance(binding, dict), "Missing selection binding.")
    parent_path = REPO_ROOT / str(binding.get("parent_preregistration"))
    parent_value = json.loads(parent_path.resolve(strict=True).read_text())
    parent._require(
        parent._sha256_file(parent_path)
        == binding.get("parent_preregistration_sha256"),
        "Parent selection preregistration hash changed.",
    )
    records = parent_value.get("selection", {}).get("records")
    parent._require(
        isinstance(records, list) and len(records) == binding.get("record_count") == 60,
        "No-training gate must use exactly the frozen 60 records.",
    )
    return [dict(record) for record in records]


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
        "polar_device": device,
        "mdp_device": "cpu",
    }


def _gaussian_problem_data(
    problem: object, source4: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from maple.solvation.continuum.separated_source_ddx import (
        embed_atomic_l1_in_first_radial_channel,
    )

    radial = embed_atomic_l1_in_first_radial_channel(source4)
    vector = radial.reshape(-1)
    phi = np.asarray(problem.phi_matrix @ vector, dtype=np.float64)
    psi = np.asarray(problem.psi_matrix @ vector, dtype=np.float64).reshape(
        int(problem.model.n_basis), len(source4)
    )
    return psi, phi


def _gaussian_energy_ev(problem: object, source4: np.ndarray) -> float:
    psi, phi = _gaussian_problem_data(problem, source4)
    _state, energy, _gradient = problem.solve_general(psi, phi)
    value = float(energy)
    parent._require(math.isfinite(value), "Gaussian fixed-source energy is invalid.")
    return value


def _source_closure(
    source4: np.ndarray,
    positions_angstrom: np.ndarray,
    target_charge: float,
    target_dipole: np.ndarray,
) -> dict[str, object]:
    from maple.function.calculator.extra_correction.implicit.gto_density import (
        cartesian_multipoles,
    )

    charges, dipoles = cartesian_multipoles(source4)
    molecular = np.sum(charges[:, None] * positions_angstrom + dipoles, axis=0)
    return {
        "total_charge_e": float(np.sum(charges)),
        "target_total_charge_e": float(target_charge),
        "total_charge_absolute_error_e": abs(float(np.sum(charges)) - target_charge),
        "molecular_dipole_eangstrom": molecular.tolist(),
        "target_molecular_dipole_eangstrom": target_dipole.tolist(),
        "molecular_dipole_l2_error_eangstrom": float(
            np.linalg.norm(molecular - target_dipole)
        ),
    }


def _run_one(
    *,
    selection_index: int,
    selected: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    mdp: object,
    polar: object,
    prereg: Mapping[str, Any],
    prereg_path: Path,
    dataset: Path,
    mdp_checkpoint: Path,
    polar_checkpoint: Path,
    device: str,
) -> dict[str, object]:
    from ase import Atoms
    from ase.data import chemical_symbols
    from maple.function.calculator.extra_correction.implicit.gto_density import (
        cartesian_multipoles,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_water_coulomb_radii,
    )
    from maple.solvation.continuum import SeparatedSourceDDXBackend
    from maple.solvation.coupling.gaussian_coulomb_projection import (
        project_charge_dipoles_in_gaussian_coulomb_metric,
    )
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
    mdp_state = mdp.evaluate(atoms)
    zero_field = np.zeros(polar.receiver_space.shape(len(atoms)), dtype=np.float64)
    polar_zero = polar.evaluate_source(atoms, zero_field)
    polar_charges, polar_dipoles = cartesian_multipoles(polar_zero)
    projection_rule = prereg["projection"]
    projected = project_charge_dipoles_in_gaussian_coulomb_metric(
        positions_angstrom=positions,
        charges_e=polar_charges,
        dipoles_eangstrom=polar_dipoles,
        target_total_charge_e=float(mdp_state.total_charge_e),
        target_molecular_dipole_eangstrom=mdp_state.public_dipole_eangstrom,
        sigma_angstrom=float(projection_rule["sigma_angstrom"]),
        maximum_gram_condition_number=float(
            projection_rule["maximum_gram_condition_number"]
        ),
        maximum_constraint_condition_number=float(
            projection_rule["maximum_constraint_condition_number"]
        ),
    )
    projected_source = parent._raw_l1(projected.charges_e, projected.dipoles_eangstrom)
    mdp_source = np.asarray(mdp_state.source4_raw_l1, dtype=np.float64)
    reference_source = parent._raw_l1(arrays["charges_e"], arrays["dipoles_eangstrom"])
    sources = {
        "mdp_latent": mdp_source,
        "polar_zero": polar_zero,
        "polar_zero_mdp_qmu": projected_source,
    }
    symbols = tuple(chemical_symbols[int(number)] for number in numbers)
    base_radii = smd_water_coulomb_radii(symbols)
    closures = {
        name: _source_closure(
            source,
            positions,
            float(mdp_state.total_charge_e),
            np.asarray(mdp_state.public_dipole_eangstrom),
        )
        for name, source in sources.items()
    }

    mep_by_scale: dict[str, object] = {}
    fixed_energy: dict[str, float] | None = None
    settings = prereg["cavity_and_continuum"]
    for radius_scale in settings["radius_scales_for_mep"]:
        radii = base_radii * float(radius_scale)
        backend = SeparatedSourceDDXBackend(
            symbols,
            radii,
            continuum_model="pcm",
            dielectric=float(settings["dielectric"]),
            lmax=int(settings["lmax"]),
            n_lebedev=int(settings["n_lebedev"]),
            solver_tolerance=float(settings["solver_tolerance"]),
            eta=float(settings["eta"]),
            n_proc=int(settings["n_proc"]),
        )
        prepared = backend.prepare(atoms, reference_source)
        problem = backend.radial_backend.prepare_problem(atoms)
        parent._require(
            problem.cavity_topology_sha256 == prepared.cavity_topology_sha256,
            "Point/Gaussian cavities differ.",
        )
        points = prepared.cavity_points_bohr
        owners = prepared.cavity_parent_indices
        weights = parent._lebedev_area_weights(
            points_bohr=points,
            owners=owners,
            positions_angstrom=positions,
            radii_angstrom=radii,
        )
        reference_qp = prepared.permanent_point_mep(reference_source)
        reference_qpqo = cartesian_atomic_multipole_potential(
            points_bohr=points,
            centers_angstrom=positions,
            charges_e=arrays["charges_e"],
            dipoles_eangstrom=arrays["dipoles_eangstrom"],
            quadrupoles_eangstrom2=arrays["quadrupoles_eangstrom2"],
            octupoles_eangstrom3=arrays["octupoles_eangstrom3"],
        )
        candidate_mep = {
            "mdp_latent_point": prepared.permanent_point_mep(mdp_source),
            "mdp_latent_gaussian": _gaussian_problem_data(problem, mdp_source)[1],
            "polar_zero_point": prepared.permanent_point_mep(polar_zero),
            "polar_zero_gaussian": _gaussian_problem_data(problem, polar_zero)[1],
            "polar_zero_mdp_qmu_point": prepared.permanent_point_mep(projected_source),
            PRIMARY_CANDIDATE: _gaussian_problem_data(problem, projected_source)[1],
        }
        references = {"reference_qp": reference_qp, "reference_qpqo": reference_qpqo}
        metrics = {
            candidate: {
                reference: parent._metric_payload(values, reference_values, weights)
                for reference, reference_values in references.items()
            }
            for candidate, values in candidate_mep.items()
        }
        scale_key = f"{float(radius_scale):.2f}"
        mep_by_scale[scale_key] = {
            "cavity_point_count": prepared.cavity_point_count,
            "cavity_topology_sha256": prepared.cavity_topology_sha256,
            "metrics": metrics,
            "reference_qp_vs_reference_qpqo": parent._metric_payload(
                reference_qp, reference_qpqo, weights
            ),
        }
        if float(radius_scale) == float(settings["fixed_source_energy_radius_scale"]):
            reference_energy = prepared.fixed_permanent_source_energy_ev(
                reference_source
            )
            energies = {
                "reference_mbis_qp_eV": reference_energy,
                "mdp_latent_point_eV": prepared.fixed_permanent_source_energy_ev(
                    mdp_source
                ),
                "mdp_latent_gaussian_eV": _gaussian_energy_ev(problem, mdp_source),
                "polar_zero_point_eV": prepared.fixed_permanent_source_energy_ev(
                    polar_zero
                ),
                "polar_zero_gaussian_eV": _gaussian_energy_ev(problem, polar_zero),
                "polar_zero_mdp_qmu_point_eV": (
                    prepared.fixed_permanent_source_energy_ev(projected_source)
                ),
                f"{PRIMARY_CANDIDATE}_eV": _gaussian_energy_ev(
                    problem, projected_source
                ),
            }
            fixed_energy = energies | {
                f"{candidate}_absolute_error_kcal_mol": abs(
                    energies[f"{candidate}_eV"] - reference_energy
                )
                * EV_TO_KCAL_MOL
                for candidate in CANDIDATES
            }
    parent._require(fixed_energy is not None, "Fixed-source energy scale was not run.")

    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-record",
        "selection_index": selection_index,
        "selection_identity": dict(selected),
        "preregistration_sha256": parent._sha256_file(prereg_path),
        "input_sha256": {
            "dataset": parent._sha256_file(dataset),
            "mace_mdp_checkpoint": parent._sha256_file(mdp_checkpoint),
            "mace_polar_checkpoint": parent._sha256_file(polar_checkpoint),
        },
        "runtime": _runtime_identity(device),
        "mdp_state_sha256": mdp_state.state_sha256,
        "polar_configuration_sha256": polar.configuration_sha256(),
        "projection": {
            "sigma_angstrom": float(projection_rule["sigma_angstrom"]),
            "gram_minimum_eigenvalue": projected.gram_minimum_eigenvalue,
            "gram_condition_number": projected.gram_condition_number,
            "constraint_condition_number": projected.constraint_condition_number,
            "constraint_max_abs_error": projected.constraint_max_abs_error,
            "coulomb_correction_norm": projected.coulomb_correction_norm,
        },
        "source_closure": closures,
        "mep_by_radius_scale": mep_by_scale,
        "fixed_source_ddpcm_energy": fixed_energy,
        "claim_boundary": prereg["claim_boundary"],
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def run_records(args: argparse.Namespace) -> None:
    prereg_path, prereg = _load_preregistration()
    selected = _selected_records(prereg)
    dataset = args.dataset.expanduser().resolve(strict=True)
    mdp_checkpoint = args.mdp_checkpoint.expanduser().resolve(strict=True)
    polar_checkpoint = args.polar_checkpoint.expanduser().resolve(strict=True)
    _validate_inputs(
        prereg,
        dataset=dataset,
        mdp_checkpoint=mdp_checkpoint,
        polar_checkpoint=polar_checkpoint,
    )
    parent._require(0 <= args.start <= args.stop <= len(selected), "Bad shard range.")

    import torch
    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=mdp_checkpoint,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    for index in range(args.start, args.stop):
        selected_record = selected[index]
        arrays = parent._load_record_arrays(dataset, selected_record)
        payload = _run_one(
            selection_index=index,
            selected=selected_record,
            arrays=arrays,
            mdp=mdp,
            polar=polar,
            prereg=prereg,
            prereg_path=prereg_path,
            dataset=dataset,
            mdp_checkpoint=mdp_checkpoint,
            polar_checkpoint=polar_checkpoint,
            device=args.device,
        )
        _write_json_exclusive(args.output_dir / f"record-{index:03d}.json", payload)


def _global_relative(
    records: list[dict[str, Any]], scale: str, candidate: str, reference: str
) -> float:
    error = 0.0
    norm = 0.0
    for record in records:
        metric = record["mep_by_radius_scale"][scale]["metrics"][candidate][reference]
        error += float(metric["area_weighted_squared_error_sum_hartree2_bohr2_per_e2"])
        norm += float(
            metric["area_weighted_reference_squared_sum_hartree2_bohr2_per_e2"]
        )
    return math.sqrt(error / max(norm, np.finfo(float).tiny))


def aggregate(args: argparse.Namespace) -> None:
    prereg_path, prereg = _load_preregistration()
    selected = _selected_records(prereg)
    _validate_inputs(
        prereg,
        dataset=args.dataset.expanduser().resolve(strict=True),
        mdp_checkpoint=args.mdp_checkpoint.expanduser().resolve(strict=True),
        polar_checkpoint=args.polar_checkpoint.expanduser().resolve(strict=True),
    )
    expected_names = {f"record-{index:03d}.json" for index in range(len(selected))}
    observed_names = {path.name for path in args.output_dir.glob("record-*.json")}
    parent._require(observed_names == expected_names, "Record file closure failed.")
    records: list[dict[str, Any]] = []
    expected_prereg_hash = parent._sha256_file(prereg_path)
    for index in range(len(selected)):
        payload = json.loads((args.output_dir / f"record-{index:03d}.json").read_text())
        digest = payload.pop("record_sha256", None)
        parent._require(digest == _canonical_sha256(payload), "Record hash mismatch.")
        payload["record_sha256"] = digest
        parent._require(
            payload.get("selection_index") == index, "Record index mismatch."
        )
        parent._require(
            payload.get("selection_identity") == selected[index],
            "Record identity mismatch.",
        )
        parent._require(
            payload.get("preregistration_sha256") == expected_prereg_hash,
            "Record preregistration mismatch.",
        )
        parent._require(
            payload.get("claim_boundary") == prereg["claim_boundary"],
            "Record claim boundary mismatch.",
        )
        records.append(payload)

    scales = tuple(
        f"{float(value):.2f}"
        for value in prereg["cavity_and_continuum"]["radius_scales_for_mep"]
    )
    global_mep = {
        scale: {
            candidate: {
                reference: _global_relative(records, scale, candidate, reference)
                for reference in REFERENCE_NAMES
            }
            for candidate in CANDIDATES
        }
        for scale in scales
    }
    energy_errors = {
        candidate: np.asarray(
            [
                record["fixed_source_ddpcm_energy"][
                    f"{candidate}_absolute_error_kcal_mol"
                ]
                for record in records
            ],
            dtype=np.float64,
        )
        for candidate in CANDIDATES
    }
    primary_errors = energy_errors[PRIMARY_CANDIDATE]
    baseline_errors = energy_errors["mdp_latent_point"]
    rules = prereg["prospective_decision_rules"]
    gates = {
        "closure": all(
            record["source_closure"]["polar_zero_mdp_qmu"][
                "total_charge_absolute_error_e"
            ]
            <= float(rules["closure_total_charge_absolute_error_e_maximum"])
            and record["source_closure"]["polar_zero_mdp_qmu"][
                "molecular_dipole_l2_error_eangstrom"
            ]
            <= float(rules["closure_molecular_dipole_l2_error_eangstrom_maximum"])
            for record in records
        ),
        "conditioning": all(
            record["projection"]["gram_condition_number"]
            <= float(prereg["projection"]["maximum_gram_condition_number"])
            and record["projection"]["constraint_condition_number"]
            <= float(prereg["projection"]["maximum_constraint_condition_number"])
            for record in records
        ),
        "fixed_source_energy_mae": float(np.mean(primary_errors))
        <= float(rules["fixed_source_ddpcm_energy_mae_kcal_mol_maximum"]),
        "paired_energy_improvement": float(np.mean(primary_errors < baseline_errors))
        >= float(rules["paired_energy_improvement_fraction_minimum"]),
        "mep_absolute": all(
            global_mep[scale][PRIMARY_CANDIDATE]["reference_qpqo"]
            <= float(
                rules["primary_qpqo_global_relative_rmse_maximum_by_radius"][scale]
            )
            for scale in scales
        ),
        "mep_baseline_reduction": all(
            1.0
            - global_mep[scale][PRIMARY_CANDIDATE]["reference_qpqo"]
            / global_mep[scale]["mdp_latent_point"]["reference_qpqo"]
            >= float(rules["primary_vs_mdp_point_qpqo_rmse_reduction_fraction_minimum"])
            for scale in scales
        ),
        "gaussian_kernel_no_worsening": all(
            global_mep[scale][PRIMARY_CANDIDATE]["reference_qpqo"]
            <= global_mep[scale]["polar_zero_mdp_qmu_point"]["reference_qpqo"]
            for scale in scales
        ),
    }
    passed = all(gates.values())
    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "preregistration_sha256": expected_prereg_hash,
        "record_count": len(records),
        "primary_candidate": PRIMARY_CANDIDATE,
        "global_relative_mep_rmse": global_mep,
        "fixed_source_ddpcm_energy": {
            candidate: {
                "mae_kcal_mol": float(np.mean(values)),
                "q95_kcal_mol": float(np.quantile(values, 0.95)),
                "maximum_kcal_mol": float(np.max(values)),
                "paired_improvement_fraction_vs_mdp_latent_point": float(
                    np.mean(values < baseline_errors)
                ),
            }
            for candidate, values in energy_errors.items()
        },
        "projection_conditioning": {
            "maximum_gram_condition_number": max(
                float(record["projection"]["gram_condition_number"])
                for record in records
            ),
            "maximum_constraint_condition_number": max(
                float(record["projection"]["constraint_condition_number"])
                for record in records
            ),
            "maximum_constraint_error": max(
                float(record["projection"]["constraint_max_abs_error"])
                for record in records
            ),
        },
        "gates": gates,
        "passed": passed,
        "decision": (
            "zero-training-projected-polar-gaussian-source-eligible-for-hybrid-integration-research"
            if passed
            else "zero-training-projected-polar-gaussian-source-rejected-under-frozen-gates"
        ),
        "claim_boundary": prereg["claim_boundary"],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(args.aggregate_output, payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("records", "aggregate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--mdp-checkpoint", type=Path, default=MDP_CHECKPOINT)
    parser.add_argument("--polar-checkpoint", type=Path, default=POLAR_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.mode == "records":
        run_records(args)
        return
    if args.aggregate_output is None:
        raise ValueError("--aggregate-output is required in aggregate mode.")
    args.aggregate_output = args.aggregate_output.expanduser().resolve()
    aggregate(args)


if __name__ == "__main__":
    main()
