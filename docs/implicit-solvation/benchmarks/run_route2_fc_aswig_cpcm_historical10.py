#!/usr/bin/env python3
"""Benchmark experimental fixed-topology amplitude-SWIG C-PCM on FreeSolv-10.

This is a deliberately narrow numerical experiment.  It keeps the immutable
historical ten-class FreeSolv panel, fixed MOL2 conformers, MACE-POLAR
checkpoint, water dielectric, SMD Coulomb radii, and upstream PySCF SMD CDS
term frozen.  Only the electrostatic continuum discretization is changed:

* all atom/Lebedev candidates are retained;
* a C3 compact exposure amplitude ``g`` is used;
* C-PCM is solved in amplitude charges, ``q = G y``;
* the reported energy uses the direct PCM half-coupling ledger.

It does *not* claim to be a reparameterization of pruned PySCF SWIG, a
replacement for IEFPCM, a force-capable Route-2 provider, or a common
MACE--PCM electronic functional.  It is an energy-only structural and
accuracy diagnostic; FreeSolv labels are read only after all settings are
frozen in this file and the checked-in input lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from route2_v0_historical_freesolv10 import (  # pyright: ignore[reportImplicitRelativeImport]
    evaluate_historical_freesolv10_predictions,
    load_historical_freesolv10_manifest,
)

from maple.function.calculator.extra_correction.implicit.ddpcm_smd import (
    _DDPCM_ENGINE_SETTINGS,
)
from maple.function.calculator.extra_correction.implicit.pyscf_smd_cds import (
    pyscf_smd_cds,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
)
from maple.function.calculator.extra_correction.implicit.route2_fc_aswig_cpcm import (
    FixedTopologyAmplitudeSWIGCPCMResponse,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (
    route2_coulomb_radii,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.route2_energy_ledger import PCM_HALF_COUPLING_ONLY_V1
from maple.function.route2_smd_profiles import DDPCM_SMD_DIRECT_PCM_PROFILE

KCAL_PER_HARTREE = 627.5094740631
WATER_STATIC_DIELECTRIC = 78.39
DEFAULT_LEBEDEV_ORDER = 15
DEFAULT_PREPARED = (
    REPO_ROOT / ".omx" / "benchmarks" / "route2-macepolar-smd" / "prepared.json"
)
DEFAULT_WORK_DIR = (
    REPO_ROOT / ".omx" / "benchmarks" / "route2-fc-aswig-cpcm-historical10-20260731"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route2-fc-aswig-cpcm-historical10-experimental-v1.json"
)


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _settings() -> dict[str, Any]:
    """Reuse the pinned direct-ddPCM model/parser profile, not its continuum."""

    params = CommandControl.from_settings(
        [
            "#model=macepol-m",
            "#sp",
            (
                "#solv(implicit=water,method=smd,provider=pyddx,"
                f"profile={DDPCM_SMD_DIRECT_PCM_PROFILE},response=scf,"
                "standard_state=1m,experimental=true)"
            ),
        ]
    ).as_dict()
    if params["solv"]["profile"] != DDPCM_SMD_DIRECT_PCM_PROFILE:
        raise RuntimeError("Direct PCM parser profile drifted.")
    return params


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _load_selection(prepared_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    candidates = prepared.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("Prepared FreeSolv artifact has no candidates list.")
    by_id = {candidate["compound_id"]: candidate for candidate in candidates}
    lock = load_historical_freesolv10_manifest()
    selected: list[dict[str, Any]] = []
    for locked in lock["locked_records"]:
        compound_id = locked["compound_id"]
        try:
            candidate = by_id[compound_id]
        except KeyError as exc:
            raise ValueError(
                f"Prepared FreeSolv artifact is missing locked {compound_id}."
            ) from exc
        if candidate["mol2_sha256"] != locked["mol2_sha256"]:
            raise ValueError(f"MOL2 identity drifted for {compound_id}.")
        if (
            abs(
                float(candidate["experimental_kcal_mol"])
                - float(locked["experimental_kcal_mol"])
            )
            > 1.0e-12
        ):
            raise ValueError(f"Experimental value drifted for {compound_id}.")
        selected.append(candidate)
    return lock, selected


def _surface_record(response: FixedTopologyAmplitudeSWIGCPCMResponse, atom_count: int) -> dict[str, Any]:
    amplitudes = response.exposure_amplitudes
    expected_size = atom_count * int(response.runtime_provenance["grid_points_per_atom"])
    if response.surface_size != expected_size:
        raise RuntimeError(
            "Fixed-topology surface cardinality does not equal atoms times "
            "nodes-per-atom."
        )
    if response.surface_areas_bohr2.shape != (expected_size,):
        raise RuntimeError("Fixed-topology surface areas have an invalid size.")
    return {
        "surface_size": response.surface_size,
        "expected_fixed_surface_size": expected_size,
        "fixed_cardinality_verified": True,
        "fully_buried_candidate_count": int(np.count_nonzero(amplitudes == 0.0)),
        "partially_exposed_candidate_count": int(
            np.count_nonzero((amplitudes > 0.0) & (amplitudes < 1.0))
        ),
        "fully_exposed_candidate_count": int(np.count_nonzero(amplitudes == 1.0)),
        "minimum_exposure_amplitude": float(np.min(amplitudes)),
        "maximum_exposure_amplitude": float(np.max(amplitudes)),
        "zero_effective_area_candidate_count": int(
            np.count_nonzero(response.surface_areas_bohr2 == 0.0)
        ),
        "continuum_runtime_provenance": response.runtime_provenance,
    }


def _run_record(
    *,
    calculator,
    candidate: dict[str, Any],
    prepared_root: Path,
    lebedev_order: int,
    device: str,
) -> dict[str, Any]:
    compound_id = candidate["compound_id"]
    mol2_path = prepared_root / candidate["mol2_relative_path"]
    if _sha256(mol2_path) != candidate["mol2_sha256"]:
        raise ValueError(f"MOL2 file content drifted for {compound_id}.")
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    mol2 = atoms.info.get("mol2")
    if not isinstance(mol2, dict):
        raise TypeError(f"{compound_id}: input lacks MOL2 atom-type metadata.")
    atom_types = mol2.get("atom_types")
    radii = route2_coulomb_radii(
        atoms.get_chemical_symbols(),
        solvent="water",
        atom_types=atom_types,
        profile=DDPCM_SMD_DIRECT_PCM_PROFILE,
    )
    built_response: dict[str, FixedTopologyAmplitudeSWIGCPCMResponse] = {}

    def reaction_field_factory(current_atoms):
        response = FixedTopologyAmplitudeSWIGCPCMResponse(
            current_atoms.get_chemical_symbols(),
            current_atoms.get_positions(),
            radii,
            dielectric=WATER_STATIC_DIELECTRIC,
            lebedev_order=lebedev_order,
        )
        built_response["response"] = response
        return response.reaction_field_linear_map(current_atoms.get_positions())

    engine = Route2ContinuumEngine(
        reaction_field_factory=reaction_field_factory,
        cds_evaluator=lambda current_atoms: pyscf_smd_cds(
            current_atoms.get_chemical_symbols(),
            current_atoms.get_positions(),
            solvent="water",
        ),
        settings=replace(
            _DDPCM_ENGINE_SETTINGS,
            continuum_label=f"FC-aSWIG-CPCM-L{lebedev_order}",
        ),
    )
    calculator.reset()
    _sync(device)
    started = time.perf_counter()
    gas_state = engine.gas_state(calculator, atoms, need_forces=False)
    coupled = engine.solve_coupled_state(
        atoms,
        calculator,
        gas_state,
        provider_cache_signature=(
            "fixed-topology-aswig-cpcm",
            "cpcm-fc-aswig-v1-experimental",
            lebedev_order,
        ),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )
    components = engine.energy_components(
        gas_state,
        coupled,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )
    _sync(device)
    wall_seconds = time.perf_counter() - started
    try:
        response = built_response["response"]
    except KeyError as exc:
        raise AssertionError("Fixed-topology continuum factory was not called.") from exc
    if float(components["solute_polarization"]) != 0.0:
        raise RuntimeError(f"{compound_id}: direct PCM ledger included a MACE leaf.")
    history = coupled.history
    if not history:
        raise RuntimeError(f"{compound_id}: SCF returned no accepted history.")
    final = history[-1]
    density = coupled.root_density_coefficients
    total_charge = float(np.sum(density[:, 0]))
    if abs(total_charge) > 1.0e-8:
        raise RuntimeError(f"{compound_id}: root density violates total-charge gate.")
    field_delta_hartree = (
        float(coupled.solvent_state.energy_ev) - float(gas_state.energy_ev)
    ) / 27.211386245988
    direct_hartree = float(components["delta_g_solv"])
    direct_kcal = direct_hartree * KCAL_PER_HARTREE
    reconstructed_legacy_kcal = (direct_hartree + field_delta_hartree) * KCAL_PER_HARTREE
    experimental = float(candidate["experimental_kcal_mol"])
    return {
        "compound_id": compound_id,
        "name": candidate["name"],
        "functional_groups": candidate["functional_groups"],
        "experimental_kcal_mol": experimental,
        "direct_pcm_prediction_kcal_mol": direct_kcal,
        "direct_pcm_signed_error_kcal_mol": direct_kcal - experimental,
        "direct_pcm_absolute_error_kcal_mol": abs(direct_kcal - experimental),
        "same_state_reconstructed_legacy_prediction_kcal_mol": (
            reconstructed_legacy_kcal
        ),
        "same_state_reconstructed_legacy_signed_error_kcal_mol": (
            reconstructed_legacy_kcal - experimental
        ),
        "same_state_reconstructed_legacy_absolute_error_kcal_mol": abs(
            reconstructed_legacy_kcal - experimental
        ),
        "excluded_field_conditioned_mace_energy_change_kcal_mol": (
            field_delta_hartree * KCAL_PER_HARTREE
        ),
        "components_kcal_mol": {
            key: float(value) * KCAL_PER_HARTREE for key, value in components.items()
        },
        "scf_monitor": {
            "unmixed_density_residual_inf_e": coupled.density_residual_inf,
            "reaction_potential_change_ev": final["reaction_potential_change_ev"],
            "reaction_gradient_change_ev_per_angstrom": final[
                "reaction_gradient_change_ev_per_angstrom"
            ],
            "ledger_energy_residual_ev": final["energy_residual_ev"],
            "energy_residual_source": final["energy_residual_source"],
            "root_total_charge_e": total_charge,
            "raw_response_total_charge_e": final["raw_response_total_charge_e"],
            "iterations": len(history),
        },
        "surface": _surface_record(response, len(atoms)),
        "cavity_radii_angstrom": np.asarray(radii, dtype=float).tolist(),
        "gas_energy_ev": float(gas_state.energy_ev),
        "field_conditioned_mace_energy_ev": float(coupled.solvent_state.energy_ev),
        "energy_identity_error_ev": float(coupled.energy_identity_error_ev),
        "wall_seconds": wall_seconds,
    }


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    errors = np.asarray([row[f"{key}_signed_error_kcal_mol"] for row in rows])
    absolute_errors = np.abs(errors)
    worst_index = int(np.argmax(absolute_errors))
    return {
        "mae_kcal_mol": float(absolute_errors.mean()),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error_kcal_mol": float(errors.mean()),
        "maximum_absolute_error_kcal_mol": float(absolute_errors.max()),
        "worst_compound_id": rows[worst_index]["compound_id"],
        "worst_name": rows[worst_index]["name"],
        "records_at_or_above_1_5_kcal_mol": int(np.sum(absolute_errors >= 1.5)),
        "all_records_strictly_below_1_5_kcal_mol": bool(
            np.all(absolute_errors < 1.5)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lebedev-order", type=int, default=DEFAULT_LEBEDEV_ORDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.lebedev_order <= 0:
        raise ValueError("--lebedev-order must be positive.")
    prepared_path = args.prepared.resolve()
    prepared_root = prepared_path.parent
    work_dir = args.work_dir.resolve()
    output_path = args.output.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    records_dir = work_dir / "records"
    records_dir.mkdir(exist_ok=True)
    lock, selected = _load_selection(prepared_path)
    settings = _settings()
    selection = {
        "locked_manifest_sha256": _sha256(
            SCRIPT_DIR / "route2-v0-historical-freesolv10-regression-v1.json"
        ),
        "prepared_sha256": _sha256(prepared_path),
        "mace_parser_profile": DDPCM_SMD_DIRECT_PCM_PROFILE,
        "continuum_provider": "fixed-topology-aswig",
        "continuum_profile": "cpcm-fc-aswig-v1-experimental",
        "continuum_equation": "variational-amplitude-cpcm-v1",
        "lebedev_order": args.lebedev_order,
        "energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
        "compound_ids": [candidate["compound_id"] for candidate in selected],
    }
    selection_path = work_dir / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise RuntimeError("Selection, continuum, or ledger drifted in work dir.")
    else:
        _write_json(selection_path, selection)

    device = args.device
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    pending = [
        candidate
        for candidate in selected
        if not (records_dir / f"{candidate['compound_id']}.json").exists()
    ]
    calculator = None
    if pending:
        first_atoms = MOL2Reader(
            str(prepared_root / pending[0]["mol2_relative_path"]),
            charge=0,
            mult=1,
        )
        calculator = SetCalculator(
            device,
            settings["model"],
            str(work_dir / "model.out"),
            atoms=first_atoms,
            implicit="smd",
            solvent="water",
            model_options=settings.get("model_options"),
            solvation_options=settings["solv"],
            charge_options=settings.get("charge") or {},
        ).set_calculator()
        if str(calculator.dtype) != "torch.float64":
            raise RuntimeError("Historical FC-aSWIG experiment requires float64 MACE-POLAR.")

    for index, candidate in enumerate(selected, start=1):
        record_path = records_dir / f"{candidate['compound_id']}.json"
        if record_path.exists():
            continue
        if calculator is None:
            raise AssertionError("A calculator is required for a pending record.")
        try:
            record = _run_record(
                calculator=calculator,
                candidate=candidate,
                prepared_root=prepared_root,
                lebedev_order=args.lebedev_order,
                device=device,
            )
            record["status"] = "success"
            print(
                f"{index:02d}/10 {candidate['name']:<24.24} "
                f"direct={record['direct_pcm_prediction_kcal_mol']:8.3f} "
                f"abs={record['direct_pcm_absolute_error_kcal_mol']:6.3f} "
                f"M={record['surface']['surface_size']:4d}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - persist every per-record failure.
            record = {
                "compound_id": candidate["compound_id"],
                "name": candidate["name"],
                "status": "failure",
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            }
            print(
                f"{index:02d}/10 {candidate['name']:<24.24} FAILURE: {exc}",
                flush=True,
            )
        _write_json(record_path, record)

    records = [
        json.loads((records_dir / f"{candidate['compound_id']}.json").read_text())
        for candidate in selected
    ]
    failures = [record for record in records if record["status"] != "success"]
    if failures:
        summary: dict[str, Any] = {
            "success_count": len(records) - len(failures),
            "failure_count": len(failures),
        }
        direct_gate = None
        legacy_gate = None
    else:
        direct_gate = evaluate_historical_freesolv10_predictions(
            [
                {
                    "compound_id": row["compound_id"],
                    "predicted_kcal_mol": row["direct_pcm_prediction_kcal_mol"],
                }
                for row in records
            ]
        ).as_dict()
        legacy_gate = evaluate_historical_freesolv10_predictions(
            [
                {
                    "compound_id": row["compound_id"],
                    "predicted_kcal_mol": row[
                        "same_state_reconstructed_legacy_prediction_kcal_mol"
                    ],
                }
                for row in records
            ]
        ).as_dict()
        summary = {
            "success_count": len(records),
            "failure_count": 0,
            "direct_pcm": _summarize(records, "direct_pcm"),
            "same_state_reconstructed_legacy_control": _summarize(
                records,
                "same_state_reconstructed_legacy",
            ),
        }
    payload = {
        "schema_version": 1,
        "artifact": "route2-fc-aswig-cpcm-historical10-experimental-v1",
        "status": "frozen-ten-class-energy-only-experimental-not-route2v-acceptance",
        "completed_at_utc": _utc(),
        "selection": selection,
        "formula": (
            "Delta G = 0.5*<c_MACE-POLAR, f_reac_FC-aSWIG-CPCM> + G_CDS; "
            "E_MACE[V_reac]-E_MACE[gas] is audit-only"
        ),
        "claim_boundary": (
            "The changing variable is the fixed-cardinality C3-amplitude C-PCM "
            "continuum, not a fitted parameter. This tests all ten frozen "
            "FreeSolv identities and does not fit their labels. It remains "
            "an energy-only nonvariational MACE fixed point, retains upstream "
            "PySCF SMD CDS, does not establish a common electronic functional, "
            "does not establish forces or PES smoothness, and cannot satisfy "
            "Route2V acceptance by accuracy alone."
        ),
        "source_lock": {
            "panel_id": lock["panel_id"],
            "historical_7_041_record": lock["historical_observed_maximum_error"],
            "distinct_chemical_classes": [
                record["chemical_class"] for record in lock["locked_records"]
            ],
        },
        "summary": summary,
        "direct_pcm_historical10_gate": direct_gate,
        "same_state_reconstructed_legacy_control_gate": legacy_gate,
        "records": records,
    }
    _write_json(output_path, payload)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output_path}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
