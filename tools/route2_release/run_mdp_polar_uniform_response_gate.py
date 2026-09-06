#!/usr/bin/env python3
"""Prospectively test a zero-training MDP/POLAR response-manifold source.

The candidate starts from the frozen MACE-POLAR zero-field atomwise source and
moves it only along the same checkpoint's uniform-affine-field response
manifold until its molecular dipole matches the frozen MACE-MDP observable.
No parameter is trained and no solvation target is read.  SPICE MBIS q/p and
q/p/Q/O are evaluation-only references on a molecule-disjoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release import run_mdp_mbis_pcm_source_gate as parent  # noqa: E402
from tools.route2_release import (
    run_mdp_polar_zero_training_source_gate as prior,
)  # noqa: E402

PREREGISTRATION = REPO_ROOT / (
    "docs/route2/preregistrations/" "mdp-polar-uniform-response-source-gate-v1.json"
)
DATASET = prior.DATASET
MDP_CHECKPOINT = prior.MDP_CHECKPOINT
POLAR_CHECKPOINT = prior.POLAR_CHECKPOINT
ARTIFACT_ID = "route2-mdp-polar-uniform-response-source-gate-v1"
CANDIDATES = (
    "mdp_latent_point",
    "polar_zero_point",
    "polar_uniform_response_point",
)
REFERENCES = ("reference_qp", "reference_qpqo")
EV_TO_KCAL_MOL = prior.EV_TO_KCAL_MOL


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


def _load_preregistration() -> tuple[Path, dict[str, Any]]:
    path = PREREGISTRATION.resolve(strict=True)
    value = json.loads(path.read_text())
    parent._require(
        value.get("artifact")
        == "route2-mdp-polar-uniform-response-source-gate-preregistration-v1",
        "Wrong uniform-response preregistration identity.",
    )
    parent._require(value.get("schema_version") == 1, "Schema version changed.")
    parent._require(
        value.get("status")
        == "locked-before-any-disjoint-heldout-source-mep-or-energy-result",
        "Uniform-response gate was not prospectively locked.",
    )
    parent._require(
        tuple(value.get("candidate_order", ())) == CANDIDATES,
        "Uniform-response candidate order changed.",
    )
    expected_boundary = {
        "capability_admitted": False,
        "cds_or_standard_state_used": False,
        "experimental_solvation_targets_read": False,
        "hybrid_line_only": True,
        "mbis_labels_used_for_evaluation_only": True,
        "mnsol_or_freesolv_imported": False,
        "post_training_or_fitting_performed": False,
        "pure_mace_polar_in_scope": False,
        "selection_disjoint_from_prior_sixty_molecules": True,
        "source_or_energy_result_used_for_selection": False,
    }
    parent._require(
        value.get("claim_boundary") == expected_boundary,
        "Uniform-response claim boundary changed.",
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
            "prior_selection_preregistration_sha256",
            REPO_ROOT / "docs/route2/preregistrations/"
            "mdp-mbis-pcm-source-physical-gate-v1.json",
        ),
        (
            "response_manifold_implementation_sha256",
            REPO_ROOT / "maple/solvation/release/uniform_response_manifold.py",
        ),
        (
            "mace_mdp_adapter_sha256",
            REPO_ROOT / "maple/solvation/models/mace_mdp.py",
        ),
        (
            "mace_polar_separated_adapter_sha256",
            REPO_ROOT / "maple/solvation/models/mace_polar_separated.py",
        ),
        ("runner_sha256", Path(__file__).resolve()),
    ):
        parent._require(
            parent._sha256_file(path) == inputs.get(key),
            f"Input hash mismatch: {key}.",
        )


def _selected_records(prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    selection = prereg.get("selection")
    parent._require(isinstance(selection, dict), "Missing selection binding.")
    records = selection.get("records")
    parent._require(
        isinstance(records, list)
        and len(records) == selection.get("record_count") == 60,
        "Uniform-response gate requires exactly 60 held-out molecules.",
    )
    normalized = [dict(record) for record in records]
    parent._require(
        _canonical_sha256(normalized) == selection.get("selection_sha256"),
        "Held-out selection digest changed.",
    )
    prior_value = json.loads(
        (
            REPO_ROOT / "docs/route2/preregistrations/"
            "mdp-mbis-pcm-source-physical-gate-v1.json"
        ).read_text()
    )
    prior_names = {record["molecule"] for record in prior_value["selection"]["records"]}
    current_names = {record["molecule"] for record in normalized}
    parent._require(
        len(current_names) == 60 and not current_names.intersection(prior_names),
        "Held-out molecule identities overlap the prior sixty.",
    )
    return normalized


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

    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_water_coulomb_radii,
    )
    from maple.solvation.continuum import SeparatedSourceDDXBackend
    from maple.solvation.release.cartesian_multipole_mep import (
        cartesian_atomic_multipole_potential,
    )
    from maple.solvation.release.uniform_response_manifold import (
        solve_uniform_response_dipole_closure,
    )

    positions = arrays["positions_angstrom"]
    atoms = Atoms(
        numbers=arrays["numbers"],
        positions=positions,
        info={"charge": 0, "multiplicity": 1},
    )
    mdp_state = mdp.evaluate(atoms)
    zero_field = np.zeros(polar.receiver_space.shape(len(atoms)), dtype=np.float64)
    polar_zero = polar.evaluate_source(atoms, zero_field)
    solve_rules = prereg["uniform_response_solver"]
    closure = solve_uniform_response_dipole_closure(
        positions_angstrom=positions,
        target_molecular_dipole_eangstrom=mdp_state.public_dipole_eangstrom,
        evaluate_source=lambda field: polar.evaluate_source(atoms, field),
        source_jvp=lambda field, direction: polar.field_jvp(atoms, field, direction),
        dipole_tolerance_eangstrom=float(solve_rules["dipole_tolerance_eangstrom"]),
        charge_tolerance_e=float(solve_rules["charge_tolerance_e"]),
        maximum_iterations=int(solve_rules["maximum_iterations"]),
        maximum_gradient_norm_ev_per_e_angstrom=float(
            solve_rules["maximum_gradient_norm_ev_per_e_angstrom"]
        ),
        maximum_jacobian_condition_number=float(
            solve_rules["maximum_jacobian_condition_number"]
        ),
        minimum_jacobian_singular_value=float(
            solve_rules["minimum_jacobian_singular_value"]
        ),
        minimum_line_search_fraction=float(solve_rules["minimum_line_search_fraction"]),
    )
    response_source = closure.source4_raw_l1
    mdp_source = np.asarray(mdp_state.source4_raw_l1, dtype=np.float64)
    reference_source = parent._raw_l1(arrays["charges_e"], arrays["dipoles_eangstrom"])
    sources = {
        "mdp_latent_point": mdp_source,
        "polar_zero_point": polar_zero,
        "polar_uniform_response_point": response_source,
    }
    closures = {
        name: prior._source_closure(
            source,
            positions,
            float(mdp_state.total_charge_e),
            np.asarray(mdp_state.public_dipole_eangstrom),
        )
        for name, source in sources.items()
    }
    symbols = tuple(chemical_symbols[int(number)] for number in arrays["numbers"])
    base_radii = smd_water_coulomb_radii(symbols)
    settings = prereg["cavity_and_continuum"]
    mep_by_scale: dict[str, object] = {}
    fixed_energy: dict[str, float] | None = None
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
        points = prepared.cavity_points_bohr
        owners = prepared.cavity_parent_indices
        weights = parent._lebedev_area_weights(
            points_bohr=points,
            owners=owners,
            positions_angstrom=positions,
            radii_angstrom=radii,
        )
        references = {
            "reference_qp": prepared.permanent_point_mep(reference_source),
            "reference_qpqo": cartesian_atomic_multipole_potential(
                points_bohr=points,
                centers_angstrom=positions,
                charges_e=arrays["charges_e"],
                dipoles_eangstrom=arrays["dipoles_eangstrom"],
                quadrupoles_eangstrom2=arrays["quadrupoles_eangstrom2"],
                octupoles_eangstrom3=arrays["octupoles_eangstrom3"],
            ),
        }
        metrics = {
            candidate: {
                reference: parent._metric_payload(
                    prepared.permanent_point_mep(source), reference_values, weights
                )
                for reference, reference_values in references.items()
            }
            for candidate, source in sources.items()
        }
        key = f"{float(radius_scale):.2f}"
        mep_by_scale[key] = {
            "cavity_point_count": prepared.cavity_point_count,
            "cavity_topology_sha256": prepared.cavity_topology_sha256,
            "metrics": metrics,
        }
        if float(radius_scale) == float(settings["fixed_source_energy_radius_scale"]):
            reference_energy = prepared.fixed_permanent_source_energy_ev(
                reference_source
            )
            energies = {
                "reference_mbis_qp_eV": reference_energy,
                **{
                    f"{name}_eV": prepared.fixed_permanent_source_energy_ev(source)
                    for name, source in sources.items()
                },
            }
            fixed_energy = energies | {
                f"{candidate}_absolute_error_kcal_mol": abs(
                    energies[f"{candidate}_eV"] - reference_energy
                )
                * EV_TO_KCAL_MOL
                for candidate in CANDIDATES
            }
    parent._require(fixed_energy is not None, "Fixed-source energy was not run.")
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
        "runtime": prior._runtime_identity(device),
        "mdp_state_sha256": mdp_state.state_sha256,
        "polar_configuration_sha256": polar.configuration_sha256(),
        "uniform_response": {
            "converged": closure.converged,
            "iterations": closure.iterations,
            "potential_gradient_ev_per_e_angstrom": (
                closure.potential_gradient_ev_per_e_angstrom.tolist()
            ),
            "potential_gradient_norm_ev_per_e_angstrom": float(
                np.linalg.norm(closure.potential_gradient_ev_per_e_angstrom)
            ),
            "residual_norm_eangstrom": closure.residual_norm_eangstrom,
            "total_charge_change_e": closure.total_charge_change_e,
            "maximum_jacobian_condition_number": (
                closure.maximum_jacobian_condition_number
            ),
            "minimum_jacobian_singular_value": (
                closure.minimum_jacobian_singular_value
            ),
            "source_l2_shift": float(np.linalg.norm(response_source - polar_zero)),
            "maximum_atom_source_shift": float(
                np.max(np.linalg.norm(response_source - polar_zero, axis=1))
            ),
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
    parent._require(0 <= args.start <= args.stop <= len(selected), "Bad range.")

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
    mdp = build_mace_mdp_moment_adapter(checkpoint_path=mdp_checkpoint, device="cpu")
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=polar_checkpoint,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    for index in range(args.start, args.stop):
        arrays = parent._load_record_arrays(dataset, selected[index])
        payload = _run_one(
            selection_index=index,
            selected=selected[index],
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
        prior._write_json_exclusive(
            args.output_dir / f"record-{index:03d}.json", payload
        )


def _global_relative(
    records: list[dict[str, Any]],
    scale: str,
    candidate: str,
    reference: str,
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
    expected_names = {f"record-{index:03d}.json" for index in range(60)}
    observed_names = {path.name for path in args.output_dir.glob("record-*.json")}
    parent._require(observed_names == expected_names, "Record closure failed.")
    records: list[dict[str, Any]] = []
    prereg_hash = parent._sha256_file(prereg_path)
    for index in range(60):
        payload = json.loads((args.output_dir / f"record-{index:03d}.json").read_text())
        digest = payload.pop("record_sha256", None)
        parent._require(digest == _canonical_sha256(payload), "Record hash mismatch.")
        payload["record_sha256"] = digest
        parent._require(payload.get("selection_index") == index, "Index mismatch.")
        parent._require(
            payload.get("selection_identity") == selected[index],
            "Selection identity mismatch.",
        )
        parent._require(
            payload.get("preregistration_sha256") == prereg_hash,
            "Preregistration mismatch.",
        )
        parent._require(
            payload.get("claim_boundary") == prereg["claim_boundary"],
            "Claim boundary mismatch.",
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
                for reference in REFERENCES
            }
            for candidate in CANDIDATES
        }
        for scale in scales
    }
    errors = {
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
    baseline = errors["polar_zero_point"]
    response = errors["polar_uniform_response_point"]
    rules = prereg["prospective_decision_rules"]
    response_diagnostics = [record["uniform_response"] for record in records]
    common_response_gates = {
        "closure": all(
            diagnostic["converged"] is True
            and float(diagnostic["residual_norm_eangstrom"])
            <= float(rules["dipole_closure_residual_eangstrom_maximum"])
            and abs(float(diagnostic["total_charge_change_e"]))
            <= float(rules["charge_change_absolute_e_maximum"])
            for diagnostic in response_diagnostics
        ),
        "field_norm": max(
            float(diagnostic["potential_gradient_norm_ev_per_e_angstrom"])
            for diagnostic in response_diagnostics
        )
        <= float(rules["potential_gradient_norm_ev_per_e_angstrom_maximum"]),
        "conditioning": max(
            float(diagnostic["maximum_jacobian_condition_number"])
            for diagnostic in response_diagnostics
        )
        <= float(rules["jacobian_condition_number_maximum"]),
    }

    def absolute_energy_gates(values: np.ndarray) -> dict[str, bool]:
        return {
            "energy_mae": float(np.mean(values))
            <= float(rules["fixed_source_energy_mae_kcal_mol_maximum"]),
            "energy_q95": float(np.quantile(values, 0.95))
            <= float(rules["fixed_source_energy_q95_kcal_mol_maximum"]),
            "energy_maximum": float(np.max(values))
            <= float(rules["fixed_source_energy_maximum_kcal_mol_maximum"]),
        }

    polar_zero_gates = absolute_energy_gates(baseline) | {
        "paired_improvement_vs_mdp": float(
            np.mean(baseline < errors["mdp_latent_point"])
        )
        >= float(rules["paired_improvement_fraction_vs_mdp_minimum"]),
        "qpqo_mep_improvement_vs_mdp": all(
            global_mep[scale]["polar_zero_point"]["reference_qpqo"]
            < global_mep[scale]["mdp_latent_point"]["reference_qpqo"]
            for scale in scales
        ),
    }
    response_gates = (
        common_response_gates
        | absolute_energy_gates(response)
        | {
            "mae_material_improvement_vs_polar_zero": (
                1.0 - float(np.mean(response)) / float(np.mean(baseline))
                >= float(rules["response_mae_reduction_fraction_minimum"])
            ),
            "paired_improvement_vs_polar_zero": float(np.mean(response < baseline))
            >= float(rules["response_paired_improvement_fraction_minimum"]),
            "tail_no_worsening": (
                float(np.quantile(response, 0.95)) <= float(np.quantile(baseline, 0.95))
                and float(np.max(response)) <= float(np.max(baseline))
            ),
            "qpqo_mep_no_worsening": all(
                global_mep[scale]["polar_uniform_response_point"]["reference_qpqo"]
                <= global_mep[scale]["polar_zero_point"]["reference_qpqo"]
                for scale in scales
            ),
        }
    )
    polar_zero_passed = all(polar_zero_gates.values())
    response_passed = all(response_gates.values())
    if response_passed:
        decision = "select-polar-uniform-response-point-for-hybrid-research"
        selected_candidate = "polar_uniform_response_point"
    elif polar_zero_passed:
        decision = "select-polar-zero-point-for-hybrid-research"
        selected_candidate = "polar_zero_point"
    else:
        decision = "reject-both-polar-point-source-candidates"
        selected_candidate = None
    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "preregistration_sha256": prereg_hash,
        "record_count": len(records),
        "global_relative_mep_rmse": global_mep,
        "fixed_source_ddpcm_energy": {
            candidate: {
                "mae_kcal_mol": float(np.mean(values)),
                "q95_kcal_mol": float(np.quantile(values, 0.95)),
                "maximum_kcal_mol": float(np.max(values)),
            }
            for candidate, values in errors.items()
        },
        "uniform_response_summary": {
            "maximum_potential_gradient_norm_ev_per_e_angstrom": max(
                float(item["potential_gradient_norm_ev_per_e_angstrom"])
                for item in response_diagnostics
            ),
            "maximum_jacobian_condition_number": max(
                float(item["maximum_jacobian_condition_number"])
                for item in response_diagnostics
            ),
            "minimum_jacobian_singular_value": min(
                float(item["minimum_jacobian_singular_value"])
                for item in response_diagnostics
            ),
            "maximum_dipole_residual_eangstrom": max(
                float(item["residual_norm_eangstrom"]) for item in response_diagnostics
            ),
        },
        "polar_zero_gates": polar_zero_gates,
        "uniform_response_gates": response_gates,
        "polar_zero_passed": polar_zero_passed,
        "uniform_response_passed": response_passed,
        "selected_candidate": selected_candidate,
        "decision": decision,
        "claim_boundary": prereg["claim_boundary"],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    prior._write_json_exclusive(args.aggregate_output, payload)


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
