#!/usr/bin/env python3
"""Adversarial zero-training tail test of MDP-polarizability dipole closure.

The seven geometries were selected *before this candidate was evaluated* as the
largest fixed-source errors of the earlier, frozen MACE-POLAR uniform-response
source.  This is an explicitly opened-tail falsification diagnostic, not an
independent accuracy estimate or an admission panel.  It reads no experimental
solvation target and fits no parameter.
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
from tools.route2_release import (  # noqa: E402
    run_mdp_polar_zero_training_source_gate as prior,
)

PREREGISTRATION = REPO_ROOT / (
    "docs/route2/preregistrations/" "mdp-polarizability-dipole-closure-tail-v1.json"
)
ARTIFACT_ID = "route2-mdp-polarizability-dipole-closure-tail-v1"
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
        == "route2-mdp-polarizability-dipole-closure-tail-preregistration-v1",
        "Wrong tail preregistration identity.",
    )
    parent._require(value.get("schema_version") == 1, "Schema version changed.")
    parent._require(
        value.get("status")
        == "locked-after-prior-tail-opened-before-new-candidate-evaluation",
        "Tail candidate was not locked at the declared boundary.",
    )
    expected_boundary = {
        "adversarial_tail_selected_from_previously_opened_errors": True,
        "capability_admitted": False,
        "experimental_solvation_targets_read": False,
        "fitting_or_post_training_performed": False,
        "hybrid_line_only": True,
        "mbis_labels_used_for_evaluation_only": True,
        "pure_mace_polar_in_scope": False,
        "result_is_independent_accuracy_estimate": False,
    }
    parent._require(
        value.get("claim_boundary") == expected_boundary,
        "Tail claim boundary changed.",
    )
    return path, value


def _validate_inputs(prereg: Mapping[str, Any]) -> None:
    inputs = prereg.get("inputs")
    parent._require(isinstance(inputs, dict), "Missing input bindings.")
    paths = {
        "spice_dataset_sha256": prior.DATASET,
        "mace_mdp_checkpoint_sha256": prior.MDP_CHECKPOINT,
        "mace_polar_checkpoint_sha256": prior.POLAR_CHECKPOINT,
        "prior_aggregate_file_sha256": REPO_ROOT
        / "docs/route2/evidence/"
        / "mdp-polar-uniform-response-source-gate-20260817/aggregate.json",
        "closure_implementation_sha256": REPO_ROOT
        / "maple/solvation/release/mdp_polarizability_dipole_closure.py",
        "mace_mdp_adapter_sha256": REPO_ROOT / "maple/solvation/models/mace_mdp.py",
        "mace_polar_adapter_sha256": REPO_ROOT
        / "maple/solvation/models/mace_polar_separated.py",
        "runner_sha256": Path(__file__).resolve(),
    }
    for key, path in paths.items():
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
        and len(records) == selection.get("record_count") == 7,
        "Tail falsification requires exactly seven records.",
    )
    normalized = [dict(record) for record in records]
    parent._require(
        _canonical_sha256(normalized) == selection.get("selection_sha256"),
        "Tail selection digest changed.",
    )
    for record in normalized:
        source_path = REPO_ROOT / record["prior_record_path"]
        parent._require(
            parent._sha256_file(source_path) == record["prior_record_file_sha256"],
            "Prior record bytes changed.",
        )
        prior_record = json.loads(source_path.read_text())
        parent._require(
            prior_record["selection_identity"] == record["selection_identity"],
            "Prior record identity changed.",
        )
        parent._require(
            prior_record["record_sha256"] == record["prior_record_sha256"],
            "Prior record payload digest changed.",
        )
    return normalized


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _run_one(
    *,
    index: int,
    selected: Mapping[str, Any],
    mdp: object,
    polar: object,
    prereg: Mapping[str, Any],
    prereg_path: Path,
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
    from maple.solvation.release.mdp_polarizability_dipole_closure import (
        close_dipole_with_mdp_polarizability,
    )

    identity = selected["selection_identity"]
    arrays = parent._load_record_arrays(prior.DATASET, identity)
    positions = arrays["positions_angstrom"]
    atoms = Atoms(
        numbers=arrays["numbers"],
        positions=positions,
        info={"charge": 0, "multiplicity": 1},
    )
    mdp_state = mdp.evaluate(atoms)
    zero_field = np.zeros(polar.receiver_space.shape(len(atoms)), dtype=np.float64)
    polar_zero = polar.evaluate_source(atoms, zero_field)
    rules = prereg["closure_rules"]
    closure = close_dipole_with_mdp_polarizability(
        source4_raw_l1=polar_zero,
        positions_angstrom=positions,
        target_molecular_dipole_eangstrom=mdp_state.public_dipole_eangstrom,
        atomic_polarizabilities_eangstrom2_per_volt=(
            mdp_state.atomic_polarizabilities_eangstrom2_per_volt
        ),
        molecular_polarizability_eangstrom2_per_volt=(
            mdp_state.public_polarizability_eangstrom2_per_volt
        ),
        dipole_tolerance_eangstrom=float(rules["dipole_tolerance_eangstrom"]),
        charge_tolerance_e=float(rules["charge_tolerance_e"]),
        maximum_condition_number=float(rules["maximum_condition_number"]),
        maximum_equivalent_field_volt_per_angstrom=float(
            rules["maximum_equivalent_field_volt_per_angstrom"]
        ),
    )
    candidate = closure.source4_raw_l1
    reference = parent._raw_l1(arrays["charges_e"], arrays["dipoles_eangstrom"])
    symbols = tuple(chemical_symbols[int(number)] for number in arrays["numbers"])
    settings = prereg["cavity_and_continuum"]
    radii = smd_water_coulomb_radii(symbols) * float(settings["radius_scale"])
    backend = SeparatedSourceDDXBackend(
        symbols,
        radii,
        continuum_model="pcm",
        dielectric=float(settings["dielectric"]),
        lmax=int(settings["lmax"]),
        n_lebedev=int(settings["n_lebedev"]),
        solver_tolerance=float(settings["solver_tolerance"]),
        eta=float(settings["eta"]),
        n_proc=1,
    )
    prepared = backend.prepare(atoms, reference)
    reference_energy = prepared.fixed_permanent_source_energy_ev(reference)
    zero_energy = prepared.fixed_permanent_source_energy_ev(polar_zero)
    candidate_energy = prepared.fixed_permanent_source_energy_ev(candidate)
    points = prepared.cavity_points_bohr
    weights = parent._lebedev_area_weights(
        points_bohr=points,
        owners=prepared.cavity_parent_indices,
        positions_angstrom=positions,
        radii_angstrom=radii,
    )
    reference_qpqo = cartesian_atomic_multipole_potential(
        points_bohr=points,
        centers_angstrom=positions,
        charges_e=arrays["charges_e"],
        dipoles_eangstrom=arrays["dipoles_eangstrom"],
        quadrupoles_eangstrom2=arrays["quadrupoles_eangstrom2"],
        octupoles_eangstrom3=arrays["octupoles_eangstrom3"],
    )
    zero_mep = prepared.permanent_point_mep(polar_zero)
    candidate_mep = prepared.permanent_point_mep(candidate)
    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-record",
        "index": index,
        "selection_identity": identity,
        "preregistration_sha256": parent._sha256_file(prereg_path),
        "prior_record_sha256": selected["prior_record_sha256"],
        "mdp_state_sha256": mdp_state.state_sha256,
        "polar_configuration_sha256": polar.configuration_sha256(),
        "closure": {
            "state_sha256": closure.state_sha256,
            "residual_eangstrom": closure.closure_residual_eangstrom,
            "total_charge_change_e": closure.total_charge_change_e,
            "equivalent_field_volt_per_angstrom": (
                closure.equivalent_uniform_field_volt_per_angstrom.tolist()
            ),
            "equivalent_field_norm_volt_per_angstrom": float(
                np.linalg.norm(closure.equivalent_uniform_field_volt_per_angstrom)
            ),
            "molecular_polarizability_condition_number": (
                closure.molecular_polarizability_condition_number
            ),
        },
        "fixed_source_ddpcm": {
            "reference_mbis_qp_eV": reference_energy,
            "polar_zero_point_eV": zero_energy,
            "candidate_point_eV": candidate_energy,
            "polar_zero_absolute_error_kcal_mol": abs(zero_energy - reference_energy)
            * EV_TO_KCAL_MOL,
            "candidate_absolute_error_kcal_mol": abs(
                candidate_energy - reference_energy
            )
            * EV_TO_KCAL_MOL,
        },
        "qpqo_mep": {
            "polar_zero": parent._metric_payload(zero_mep, reference_qpqo, weights),
            "candidate": parent._metric_payload(candidate_mep, reference_qpqo, weights),
        },
        "claim_boundary": prereg["claim_boundary"],
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def run(args: argparse.Namespace) -> None:
    prereg_path, prereg = _load_preregistration()
    _validate_inputs(prereg)
    selected = _selected_records(prereg)
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
        checkpoint_path=prior.MDP_CHECKPOINT, device="cpu"
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=prior.POLAR_CHECKPOINT,
        device=args.device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)
    for index, record in enumerate(selected):
        payload = _run_one(
            index=index,
            selected=record,
            mdp=mdp,
            polar=polar,
            prereg=prereg,
            prereg_path=prereg_path,
        )
        _write_json_exclusive(args.output_dir / f"record-{index:03d}.json", payload)


def aggregate(args: argparse.Namespace) -> None:
    prereg_path, prereg = _load_preregistration()
    _validate_inputs(prereg)
    selected = _selected_records(prereg)
    expected_names = {f"record-{i:03d}.json" for i in range(len(selected))}
    observed_names = {path.name for path in args.output_dir.glob("record-*.json")}
    parent._require(observed_names == expected_names, "Record closure failed.")
    records = []
    prereg_hash = parent._sha256_file(prereg_path)
    for index, selection in enumerate(selected):
        payload = json.loads((args.output_dir / f"record-{index:03d}.json").read_text())
        digest = payload.pop("record_sha256", None)
        parent._require(digest == _canonical_sha256(payload), "Record hash failed.")
        payload["record_sha256"] = digest
        parent._require(payload.get("index") == index, "Record index failed.")
        parent._require(
            payload.get("selection_identity") == selection["selection_identity"],
            "Record identity failed.",
        )
        parent._require(
            payload.get("preregistration_sha256") == prereg_hash,
            "Record preregistration failed.",
        )
        records.append(payload)
    zero = np.asarray(
        [
            row["fixed_source_ddpcm"]["polar_zero_absolute_error_kcal_mol"]
            for row in records
        ]
    )
    candidate = np.asarray(
        [
            row["fixed_source_ddpcm"]["candidate_absolute_error_kcal_mol"]
            for row in records
        ]
    )
    zero_mep_error = sum(
        row["qpqo_mep"]["polar_zero"][
            "area_weighted_squared_error_sum_hartree2_bohr2_per_e2"
        ]
        for row in records
    )
    candidate_mep_error = sum(
        row["qpqo_mep"]["candidate"][
            "area_weighted_squared_error_sum_hartree2_bohr2_per_e2"
        ]
        for row in records
    )
    reference_norm = sum(
        row["qpqo_mep"]["candidate"][
            "area_weighted_reference_squared_sum_hartree2_bohr2_per_e2"
        ]
        for row in records
    )
    rules = prereg["terminal_falsification_rules"]
    gates = {
        "paired_improvement_count": int(np.count_nonzero(candidate < zero))
        >= int(rules["paired_improvement_count_minimum"]),
        "mean_relative_reduction": (
            1.0 - float(np.mean(candidate)) / float(np.mean(zero))
        )
        >= float(rules["mean_error_reduction_fraction_minimum"]),
        "maximum_no_worsening": float(np.max(candidate))
        <= float(np.max(zero)) + float(rules["maximum_worsening_tolerance_kcal_mol"]),
        "qpqo_mep_no_worsening": candidate_mep_error
        <= zero_mep_error * (1.0 + float(rules["mep_relative_worsening_tolerance"])),
        "closure": all(
            row["closure"]["residual_eangstrom"]
            <= float(prereg["closure_rules"]["dipole_tolerance_eangstrom"])
            and abs(row["closure"]["total_charge_change_e"])
            <= float(prereg["closure_rules"]["charge_tolerance_e"])
            for row in records
        ),
    }
    passed = all(gates.values())
    payload: dict[str, object] = {
        "artifact": f"{ARTIFACT_ID}-aggregate",
        "preregistration_sha256": prereg_hash,
        "record_count": len(records),
        "polar_zero": {
            "mae_kcal_mol": float(np.mean(zero)),
            "maximum_kcal_mol": float(np.max(zero)),
            "qpqo_global_relative_mep_rmse": math.sqrt(zero_mep_error / reference_norm),
        },
        "candidate": {
            "mae_kcal_mol": float(np.mean(candidate)),
            "maximum_kcal_mol": float(np.max(candidate)),
            "paired_improvement_count": int(np.count_nonzero(candidate < zero)),
            "mean_error_reduction_fraction": 1.0
            - float(np.mean(candidate)) / float(np.mean(zero)),
            "qpqo_global_relative_mep_rmse": math.sqrt(
                candidate_mep_error / reference_norm
            ),
        },
        "gates": gates,
        "passed_tail_falsification": passed,
        "decision": (
            "candidate-survives-opened-tail-falsification-requires-new-independent-gate"
            if passed
            else "candidate-rejected-by-opened-tail-falsification"
        ),
        "claim_boundary": prereg["claim_boundary"],
    }
    payload["aggregate_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(args.output_dir / "aggregate.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
