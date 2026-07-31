#!/usr/bin/env python3
"""Run the direct-PCM Route-2 ledger on the immutable FreeSolv-10 gate.

The selected molecule identities, conformers, experimental values, ten
chemical classes, and the historic 7.041-kcal/mol ethyl-acetate record come
from the tracked V0 historical lock.  This runner selects none of them using
the new method's errors.  It uses a direct ddPCM profile so MACE-POLAR remains
the SCF density source while the reported electrostatic solvation energy is
only the ddPCM half-coupling; the excluded MACE field-energy change is retained
as an audit value and reconstructs a same-state legacy-ledger control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from route2_v0_historical_freesolv10 import (
    evaluate_historical_freesolv10_predictions,
    load_historical_freesolv10_manifest,
)

from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.function.route2_smd_profiles import DDPCM_SMD_DIRECT_PCM_PROFILE

KCAL_PER_HARTREE = 627.5094740631
DEFAULT_PREPARED = (
    REPO_ROOT / ".omx" / "benchmarks" / "route2-macepolar-smd" / "prepared.json"
)
DEFAULT_WORK_DIR = (
    REPO_ROOT / ".omx" / "benchmarks" / "route2-direct-pcm-ddpcm-historical10-20260731"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "route2-direct-pcm-ddpcm-historical10-diagnostic-v1.json"


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


def _run_record(
    *,
    calculator,
    candidate: dict[str, Any],
    settings: dict[str, Any],
    prepared_root: Path,
    work_dir: Path,
    device: str,
) -> dict[str, Any]:
    compound_id = candidate["compound_id"]
    mol2_path = prepared_root / candidate["mol2_relative_path"]
    if _sha256(mol2_path) != candidate["mol2_sha256"]:
        raise ValueError(f"MOL2 file content drifted for {compound_id}.")
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    audit_output = work_dir / "provider-audit" / compound_id / "maple.out"
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    calculator.solvent_correction = ImplicitSolvationCorrection(
        atoms,
        settings.get("charge") or {},
        settings["solv"],
        output=str(audit_output),
    )
    calculator.reset()
    atoms.calc = calculator
    _sync(device)
    started = time.perf_counter()
    combined_energy_hartree = float(atoms.get_potential_energy())
    _sync(device)
    wall_seconds = time.perf_counter() - started
    solvation = calculator.results.get("solvation")
    if not isinstance(solvation, dict):
        raise TypeError(f"{compound_id}: no integrated solvation payload.")
    components = solvation["components_hartree"]
    if float(components["solute_polarization"]) != 0.0:
        raise RuntimeError(
            f"{compound_id}: direct PCM profile reported a nonzero MACE leaf."
        )
    audit_dir = Path(calculator.solvent_correction.audit_dir)
    audit = json.loads((audit_dir / "route2-ddpcm-result.json").read_text())
    if audit.get("electrostatic_energy_ledger") != "pcm-half-coupling-only-v1":
        raise RuntimeError(f"{compound_id}: direct ledger provenance drifted.")
    field_delta_hartree = float(audit["field_conditioned_mace_energy_change_hartree"])
    direct_hartree = float(solvation["delta_g_solv_hartree"])
    reconstructed_legacy_hartree = direct_hartree + field_delta_hartree
    if (
        abs(
            combined_energy_hartree
            - (float(solvation["gas_energy_hartree"]) + direct_hartree)
        )
        > 1.0e-10
    ):
        raise RuntimeError(f"{compound_id}: direct integrated energy mismatch.")
    direct_kcal = direct_hartree * KCAL_PER_HARTREE
    legacy_kcal = reconstructed_legacy_hartree * KCAL_PER_HARTREE
    experimental = float(candidate["experimental_kcal_mol"])
    return {
        "compound_id": compound_id,
        "name": candidate["name"],
        "functional_groups": candidate["functional_groups"],
        "experimental_kcal_mol": experimental,
        "direct_pcm_prediction_kcal_mol": direct_kcal,
        "direct_pcm_signed_error_kcal_mol": direct_kcal - experimental,
        "direct_pcm_absolute_error_kcal_mol": abs(direct_kcal - experimental),
        "reconstructed_legacy_prediction_kcal_mol": legacy_kcal,
        "reconstructed_legacy_signed_error_kcal_mol": legacy_kcal - experimental,
        "reconstructed_legacy_absolute_error_kcal_mol": abs(legacy_kcal - experimental),
        "excluded_field_conditioned_mace_energy_change_kcal_mol": (
            field_delta_hartree * KCAL_PER_HARTREE
        ),
        "components_kcal_mol": {
            key: float(value) * KCAL_PER_HARTREE for key, value in components.items()
        },
        "combined_energy_hartree": combined_energy_hartree,
        "gas_energy_hartree": float(solvation["gas_energy_hartree"]),
        "wall_seconds": wall_seconds,
        "scf_iterations": int(solvation["provenance"]["iterations"]),
        "audit_directory": str(audit_dir),
    }


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    errors = np.asarray([row[f"{key}_signed_error_kcal_mol"] for row in rows])
    abs_errors = np.abs(errors)
    worst_index = int(np.argmax(abs_errors))
    return {
        "mae_kcal_mol": float(abs_errors.mean()),
        "rmse_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error_kcal_mol": float(errors.mean()),
        "maximum_absolute_error_kcal_mol": float(abs_errors.max()),
        "worst_compound_id": rows[worst_index]["compound_id"],
        "worst_name": rows[worst_index]["name"],
        "records_at_or_above_1_5_kcal_mol": int(np.sum(abs_errors >= 1.5)),
        "all_records_strictly_below_1_5_kcal_mol": bool(np.all(abs_errors < 1.5)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = _parser().parse_args()
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
        "profile": DDPCM_SMD_DIRECT_PCM_PROFILE,
        "compound_ids": [candidate["compound_id"] for candidate in selected],
    }
    selection_path = work_dir / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise RuntimeError("Selection or profile drifted in existing work dir.")
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
            raise RuntimeError("Historical direct PCM requires float64 MACE-POLAR.")

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
                settings=settings,
                prepared_root=prepared_root,
                work_dir=work_dir,
                device=device,
            )
            record["status"] = "success"
            print(
                f"{index:02d}/10 {candidate['name']:<24.24} "
                f"direct={record['direct_pcm_prediction_kcal_mol']:8.3f} "
                f"abs={record['direct_pcm_absolute_error_kcal_mol']:6.3f}",
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
        summary = {
            "success_count": len(records) - len(failures),
            "failure_count": len(failures),
        }
        direct_gate = None
        reconstructed_gate = None
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
        reconstructed_gate = evaluate_historical_freesolv10_predictions(
            [
                {
                    "compound_id": row["compound_id"],
                    "predicted_kcal_mol": row[
                        "reconstructed_legacy_prediction_kcal_mol"
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
                "reconstructed_legacy",
            ),
        }
    payload = {
        "schema_version": 1,
        "artifact": "route2-direct-pcm-ddpcm-historical10-diagnostic-v1",
        "status": "frozen-ten-class-energy-only-diagnostic-not-route2v-acceptance",
        "completed_at_utc": _utc(),
        "profile": DDPCM_SMD_DIRECT_PCM_PROFILE,
        "electrostatic_energy_ledger": "pcm-half-coupling-only-v1",
        "formula": (
            "Delta G = 0.5*<c_MACE-POLAR, f_reac_ddPCM> + G_CDS; "
            "E_MACE[V_reac]-E_MACE[gas] is audit-only"
        ),
        "claim_boundary": (
            "This tests the actual direct ddPCM implementation on the ten "
            "locked FreeSolv identities and does not fit experimental labels. "
            "It is still a nonvariational MACE fixed point and is energy-only; "
            "it does not establish a common stationary electronic functional, "
            "reciprocity, conservative forces, broad coverage, or Route2V "
            "acceptance."
        ),
        "source_lock": {
            "panel_id": lock["panel_id"],
            "historical_7_041_record": lock["historical_observed_maximum_error"],
            "distinct_chemical_classes": [
                record["chemical_class"] for record in lock["locked_records"]
            ],
        },
        "selection": selection,
        "summary": summary,
        "direct_pcm_historical10_gate": direct_gate,
        "same_state_reconstructed_legacy_control_gate": reconstructed_gate,
        "records": records,
    }
    _write_json(output_path, payload)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {output_path}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
