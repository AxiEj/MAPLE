#!/usr/bin/env python3
"""Extract frozen MACE-MDP geometry descriptors for the VQM24 train split.

The helper never opens a QM observable, PCM quantity, or solvation label.  It
stores the audited atomwise q/p/polarizability partitions solely as frozen
geometry descriptors and baselines for the scalar-head ablation sequence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from ase import Atoms
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.models.mace_mdp import (  # noqa: E402
    MACE_MDP_EXPECTED_CHECKPOINT_SHA256,
    build_mace_mdp_moment_adapter,
)


SELF_REPO_PATH = "tools/route2_release/run_vqm24_frozen_mdp_feature_batch.py"
PREREGISTRATION_ARTIFACT = "route2-vqm24-observable-training-batch-dense-prereg-v2"
ARTIFACT = "route2-vqm24-frozen-mdp-feature-batch-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def run(args: argparse.Namespace) -> dict[str, object]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if preregistration.get("artifact") != PREREGISTRATION_ARTIFACT:
        raise ValueError("Frozen MDP feature batch has the wrong parent.")
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != MACE_MDP_EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen MACE-MDP checkpoint identity changed.")
    adapter = build_mace_mdp_moment_adapter(
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    records = []
    started = time.perf_counter()
    for record in preregistration["records"]:
        selection = record["selection_record"]
        atoms = Atoms(
            numbers=np.asarray(selection["atomic_numbers"], dtype=np.int64),
            positions=np.asarray(selection["positions_angstrom"], dtype=np.float64),
        )
        state = adapter.evaluate(atoms)
        record_directory = output_directory / record["record_id"]
        record_directory.mkdir()
        npz_path = record_directory / "features.npz"
        np.savez(
            npz_path,
            atomic_numbers=state.atomic_numbers,
            positions_angstrom=state.positions_angstrom,
            charges_e=state.charges_e,
            atomic_dipoles_eangstrom=state.atomic_dipoles_eangstrom,
            atomic_polarizabilities_eangstrom2_per_volt=(
                state.atomic_polarizabilities_eangstrom2_per_volt
            ),
            source4_raw_l1=state.source4_raw_l1,
            public_dipole_eangstrom=state.public_dipole_eangstrom,
            public_polarizability_eangstrom2_per_volt=(
                state.public_polarizability_eangstrom2_per_volt
            ),
        )
        npz_path.chmod(0o444)
        record_json = {
            "record_id": record["record_id"],
            "selection_record_sha256": selection["record_sha256"],
            "model_input_sha256": state.model_input_sha256,
            "state_sha256": state.state_sha256,
            "npz_sha256": _sha256(npz_path),
            "atom_count": len(atoms),
            "total_charge_e": float(np.sum(state.charges_e)),
            "dipole_closure_max_abs_eangstrom": float(
                np.max(
                    np.abs(
                        np.sum(
                            state.charges_e[:, None] * state.positions_angstrom
                            + state.atomic_dipoles_eangstrom,
                            axis=0,
                        )
                        - state.public_dipole_eangstrom
                    )
                )
            ),
            "polarizability_closure_max_abs": float(
                np.max(
                    np.abs(
                        np.sum(
                            state.atomic_polarizabilities_eangstrom2_per_volt,
                            axis=0,
                        )
                        - state.public_polarizability_eangstrom2_per_volt
                    )
                )
            ),
        }
        json_path = record_directory / "features.json"
        json_path.write_text(json.dumps(record_json, indent=2, sort_keys=True) + "\n")
        json_path.chmod(0o444)
        records.append(
            {
                **record_json,
                "json_sha256": _sha256(json_path),
                "relative_directory": record["record_id"],
            }
        )
    records.sort(key=lambda record: record["record_id"])
    payload = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "complete-frozen-mdp-features",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "adapter_configuration_sha256": adapter.configuration_sha256(),
        },
        "records": records,
        "aggregate": {
            "record_count": len(records),
            "maximum_total_charge_abs_e": max(
                abs(record["total_charge_e"]) for record in records
            ),
            "maximum_dipole_closure_abs_eangstrom": max(
                record["dipole_closure_max_abs_eangstrom"] for record in records
            ),
            "maximum_polarizability_closure_abs": max(
                record["polarizability_closure_max_abs"] for record in records
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float64",
            "checkpoint_energy_role": "none-dipole-polarizability-model",
        },
        "claim_boundary": {
            "frozen_mdp_descriptors_generated": True,
            "original_atomic_partition_used_as_training_truth": False,
            "qm_observable_read": False,
            "fit_or_training_performed": False,
            "model_accuracy_measured": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    result_path = output_directory / "manifest.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
