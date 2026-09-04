#!/usr/bin/env python3
"""Replay all 20x8 response-cross arms from the committed profile bundle only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maple.function.cosmors_torch.cosmospace import (  # noqa: E402
    OPEN_COSMORS_24A_PARAMETERS,
)
from maple.function.cosmors_torch.surface import read_sigma_profile  # noqa: E402
from maple.function.cosmors_torch.thermodynamics import (  # noqa: E402
    open24a_solvation_free_energy,
)
from run_mace_ef_cosmors_freesolv20 import (  # noqa: E402
    _bind_source_files,
    _canonical_sha256,
    _git_blob_sha256,
    _repository_relative,
    _runtime_versions,
    _sha256_file,
    _write_json_atomic,
)

PRIMARY_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-diverse-v3.json"
FROZEN_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-frozen-source-ablation-v2.json"
CROSS_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-response-cross-v1.json"
PREREGISTRATION_PATH = (
    SCRIPT_DIR / "mace-ef-cosmors-freesolv20-response-cross-prereg-v1.json"
)
BUNDLE_MANIFEST_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-profile-bundle-v1.json"
BUNDLE_PATH = SCRIPT_DIR / "mace-ef-cosmors-freesolv20-profile-bundle-v1.tar.gz"
EPSILON_COEFFICIENT = 1.0e-30


def _assert_source_bound(artifact: dict[str, object]) -> None:
    hashes = artifact.get("source_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError("Input artifact lacks committed source-file binding.")
    commit = artifact.get("execution_git_head")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("Input artifact lacks a full execution Git commit.")
    for relative, expected in hashes.items():
        if _git_blob_sha256(commit, str(relative)) != expected:
            raise RuntimeError(
                f"Input artifact source {relative!r} is not bound at {commit}."
            )


def _to_device(profile, device: str):
    updates = {}
    for field in (
        "sigma_e_per_angstrom2",
        "sigma_orthogonal_e_per_angstrom2",
        "areas_angstrom2",
        "atomic_numbers",
        "hydrogen_bond_donor_weight",
        "hydrogen_bond_acceptor_weight",
        "ionic_contact_weight",
        "cavity_volume_angstrom3",
        "dielectric_energy_hartree",
        "molecular_charge_e",
    ):
        updates[field] = getattr(profile, field).to(device=device)
    return replace(profile, **updates)


def _load_profiles_from_bundle(
    bundle: Path,
    member_sha256: dict[str, str],
    *,
    device: str,
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    with TemporaryDirectory(prefix="maple-cosmors-profile-replay-") as temporary:
        root = Path(temporary)
        with tarfile.open(bundle, mode="r:gz") as archive:
            members = archive.getmembers()
            observed_names = [member.name for member in members]
            if observed_names != sorted(member_sha256):
                raise RuntimeError("Profile bundle members drifted from the manifest.")
            for member in members:
                if not member.isfile() or member.name.startswith(("/", "../")):
                    raise RuntimeError("Profile bundle contains an unsafe member.")
                if not (
                    member.name.startswith("scf/") or member.name.startswith("frozen/")
                ):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not read bundle member {member.name!r}.")
                payload = extracted.read()
                observed = hashlib.sha256(payload).hexdigest()
                if observed != member_sha256[member.name]:
                    raise RuntimeError(f"Bundle member {member.name!r} failed SHA256.")
                path = root / member.name.replace("/", "__")
                path.write_bytes(payload)
                profiles[member.name] = _to_device(
                    read_sigma_profile(path),
                    device,
                )
    return profiles


def _arm_state(name: str) -> tuple[str, str, str]:
    parts = name.split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid preregistered arm name {name!r}.")
    solute = parts[0].removesuffix("_solute")
    water = parts[1].removesuffix("_water")
    interaction = parts[2]
    if solute not in {"frozen", "self_consistent"}:
        raise ValueError(f"Unknown solute state in arm {name!r}.")
    if water not in {"frozen", "self_consistent"}:
        raise ValueError(f"Unknown water state in arm {name!r}.")
    if interaction not in {"full", "no_hydrogen_bond"}:
        raise ValueError(f"Unknown interaction model in arm {name!r}.")
    return solute, water, interaction


def run(
    *,
    output_path: Path,
    device: str,
    tolerance_kcal_mol: float,
    torch_threads: int,
) -> int:
    import torch

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_threads)
    primary = json.loads(PRIMARY_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    reference = json.loads(CROSS_PATH.read_text(encoding="utf-8"))
    preregistration = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    bundle_manifest = json.loads(BUNDLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if any(item.get("status") != "complete" for item in (primary, frozen, reference)):
        raise ValueError("Bundle replay requires complete input artifacts.")
    if bundle_manifest.get("status") != "complete":
        raise ValueError("Bundle replay requires a complete profile manifest.")
    if _sha256_file(BUNDLE_PATH) != bundle_manifest["archive"]["sha256"]:
        raise RuntimeError("Committed profile bundle failed SHA256 validation.")
    expected_inputs = {
        "primary": _sha256_file(PRIMARY_PATH),
        "frozen_control": _sha256_file(FROZEN_PATH),
        "response_cross": _sha256_file(CROSS_PATH),
    }
    for name, expected in expected_inputs.items():
        if bundle_manifest["input_artifacts"][name]["sha256"] != expected:
            raise RuntimeError(f"Profile bundle is not bound to {name}.")

    execution_git_head, source_files_sha256 = _bind_source_files(
        (
            Path(__file__),
            PREREGISTRATION_PATH,
            BUNDLE_MANIFEST_PATH,
            BUNDLE_PATH,
        )
    )
    for artifact in (primary, frozen, reference, bundle_manifest):
        _assert_source_bound(artifact)

    profiles = _load_profiles_from_bundle(
        BUNDLE_PATH,
        bundle_manifest["member_sha256"],
        device=device,
    )
    maximum_iterations = int(primary["model"]["cosmospace_maximum_iterations"])
    parameters = {
        "full": replace(
            OPEN_COSMORS_24A_PARAMETERS,
            maximum_iterations=maximum_iterations,
        ),
        "no_hydrogen_bond": replace(
            OPEN_COSMORS_24A_PARAMETERS,
            name="open24a-ablation-no-hydrogen-bond",
            hydrogen_bond_coefficient_j_angstrom2_per_mol_e2=EPSILON_COEFFICIENT,
            maximum_iterations=maximum_iterations,
        ),
    }
    reference_by_id = {record["compound_id"]: record for record in reference["records"]}
    water_profiles = {
        "frozen": profiles["frozen/water.torch-cosmors.json"],
        "self_consistent": profiles["scf/water.torch-cosmors.json"],
    }
    replay_records = []
    per_arm_differences: dict[str, list[float]] = {
        arm: [] for arm in preregistration["arms"]
    }
    for primary_record in primary["records"]:
        compound_id = primary_record["compound_id"]
        reference_record = reference_by_id[compound_id]
        solute_profiles = {
            "frozen": profiles[f"frozen/{compound_id}.torch-cosmors.json"],
            "self_consistent": profiles[f"scf/{compound_id}.torch-cosmors.json"],
        }
        arm_differences = {}
        for arm in preregistration["arms"]:
            solute_state, water_state, interaction = _arm_state(arm)
            result = open24a_solvation_free_energy(
                solute_profiles[solute_state],
                water_profiles[water_state],
                temperature_k=298.15,
                ring_atom_count=int(primary_record["ring_atom_count"]),
                solvent_liquid_molar_volume_cm3_mol=18.06863632,
                cosmospace_parameters=parameters[interaction],
            )
            replayed = float(result.delta_g_solvation_kcal_mol.detach().cpu())
            expected = float(reference_record["arms"][arm]["predicted_kcal_mol"])
            difference = replayed - expected
            arm_differences[arm] = difference
            per_arm_differences[arm].append(difference)
        replay_records.append(
            {
                "ordinal": int(primary_record["ordinal"]),
                "compound_id": compound_id,
                "arm_difference_kcal_mol": arm_differences,
            }
        )

    per_arm_maximum = {
        arm: float(np.max(np.abs(values)))
        for arm, values in per_arm_differences.items()
    }
    maximum = max(per_arm_maximum.values())
    passed = maximum <= tolerance_kcal_mol
    output: dict[str, object] = {
        "schema_version": 1,
        "artifact": "mace-ef-cosmors-freesolv20-response-cross-bundle-replay-v2",
        "status": "pass" if passed else "fail",
        "scientific_result": False,
        "asset_source": "committed-profile-bundle-only",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "execution_git_head": execution_git_head,
        "source_files_sha256": source_files_sha256,
        "runtime": {
            **_runtime_versions(),
            "device": device,
            "torch_threads": torch_threads,
        },
        "inputs": {
            "primary_sha256": expected_inputs["primary"],
            "frozen_control_sha256": expected_inputs["frozen_control"],
            "response_cross_sha256": expected_inputs["response_cross"],
            "profile_bundle_sha256": _sha256_file(BUNDLE_PATH),
            "profile_bundle_manifest_sha256": _sha256_file(BUNDLE_MANIFEST_PATH),
            "preregistration_sha256": _sha256_file(PREREGISTRATION_PATH),
        },
        "method": {
            "record_count": len(replay_records),
            "arm_count": len(preregistration["arms"]),
            "prediction_count": sum(
                len(record["arm_difference_kcal_mol"]) for record in replay_records
            ),
            "tolerance_kcal_mol": tolerance_kcal_mol,
            "full_parameter_sha256": _canonical_sha256(asdict(parameters["full"])),
            "no_hydrogen_bond_parameter_sha256": _canonical_sha256(
                asdict(parameters["no_hydrogen_bond"])
            ),
        },
        "maximum_absolute_prediction_difference_kcal_mol": maximum,
        "per_arm_maximum_absolute_difference_kcal_mol": per_arm_maximum,
        "records": replay_records,
        "claim_boundary": "Numerical replay of the committed 20x8 profile-level calculation only; no source-generation, accuracy, passivity, force, or admission claim is added.",
    }
    _write_json_atomic(output_path, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "prediction_count": output["method"]["prediction_count"],
                "maximum_absolute_prediction_difference_kcal_mol": maximum,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance-kcal-mol", type=float, default=1.0e-9)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if not np.isfinite(args.tolerance_kcal_mol) or args.tolerance_kcal_mol <= 0.0:
        parser.error("--tolerance-kcal-mol must be finite and positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    return run(
        output_path=args.output.expanduser().resolve(),
        device=str(args.device),
        tolerance_kcal_mol=float(args.tolerance_kcal_mol),
        torch_threads=int(args.torch_threads),
    )


if __name__ == "__main__":
    raise SystemExit(main())
