#!/usr/bin/env python3
"""Executable identity ledger for the isolated matrix-free research branch.

The ledger proves provenance and public unreachability.  It does not prove
model accuracy, complete any gate, or admit a scientific/public capability.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN_TRACKED = HERE / "source-freeze.json"
EXPECTED_FROZEN_MANIFEST_SHA256 = (
    "ab0776e30e4fdb857ccd470a4ecf704a91597ddce31054d40494c204aa1d5d61"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a"
)
EXPECTED_CHECKPOINT_SIZE = 68_133_235
RESEARCH_PROFILE = "pure-macepolar-native-fp64-ddpcm-matrixfree-research-v1"
RESEARCH_PUBLIC_MODULE = "maple.solvation.experimental.ddpcm_matrixfree_300"
ACTIVE_EMBEDDED_COLUMNS = (0, 2, 3, 4)
INACTIVE_EMBEDDED_COLUMNS = (1, 5, 6, 7)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def tensor_sha(value: Any) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)


def current_tracked_freeze() -> dict[str, object]:
    frozen = json.loads(FROZEN_TRACKED.read_text(encoding="utf-8"))
    names = tuple(sorted(frozen["sources"]))
    current = {
        name: sha256_file(ROOT / name) if (ROOT / name).is_file() else None
        for name in names
    }
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": current,
        "manifest_entry_count": len(current),
        "manifest_sha256": canonical_sha(current),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        ).hexdigest(),
    }


def verify_tracked_freeze() -> dict[str, object]:
    payload = FROZEN_TRACKED.read_bytes()
    manifest_digest = hashlib.sha256(payload).hexdigest()
    if manifest_digest != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise RuntimeError("The frozen source manifest itself changed.")
    frozen = json.loads(payload)
    current = current_tracked_freeze()
    if current["tracked_diff_sha256"] != hashlib.sha256(b"").hexdigest():
        raise RuntimeError(
            "Tracked worktree is dirty; research requires the frozen clean baseline."
        )
    if current["head"] != frozen["head"]:
        raise RuntimeError("Frozen research HEAD changed.")
    if current["sources"] != frozen["sources"]:
        changed = sorted(
            name
            for name, digest in frozen["sources"].items()
            if current["sources"].get(name) != digest
        )
        raise RuntimeError(f"Frozen 312-file tracked manifest changed: {changed[:8]}")
    current["frozen_manifest_file_sha256"] = manifest_digest
    return current


def research_python_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(HERE.glob("*.py"))
        if path.is_file()
    }


def public_unreachability() -> dict[str, object]:
    module_spec = importlib.util.find_spec(RESEARCH_PUBLIC_MODULE)
    import_failed = False
    import_error = None
    try:
        importlib.import_module(RESEARCH_PUBLIC_MODULE)
    except ModuleNotFoundError as error:
        import_failed = True
        import_error = str(error)
    from maple.function.route2_smd_profiles import (
        SUPPORTED_ROUTE2_SMD_PROFILES,
        route2_smd_profile_spec,
        route2_smd_profiles_for_provider,
    )

    profile_rejected = False
    profile_error = None
    try:
        route2_smd_profile_spec(RESEARCH_PROFILE)
    except ValueError as error:
        profile_rejected = True
        profile_error = str(error)
    research_registry_entries = sorted(
        name
        for name in SUPPORTED_ROUTE2_SMD_PROFILES
        if "matrixfree" in name.lower() or "research" in name.lower()
    )
    provider_rejected = False
    provider_error = None
    try:
        route2_smd_profiles_for_provider("matrixfree")
    except ValueError as error:
        provider_rejected = True
        provider_error = str(error)
    research_provider_entries = sorted(
        name
        for name in SUPPORTED_ROUTE2_SMD_PROFILES
        if any(
            marker in route2_smd_profile_spec(name).provider.lower()
            for marker in ("matrixfree", "research")
        )
    )
    passed = (
        module_spec is None
        and import_failed
        and profile_rejected
        and provider_rejected
        and not research_registry_entries
        and not research_provider_entries
    )
    if not passed:
        raise RuntimeError("Research implementation became reachable from public APIs.")
    return {
        "passed": True,
        "public_module": RESEARCH_PUBLIC_MODULE,
        "module_spec_is_none": module_spec is None,
        "module_import_failed": import_failed,
        "module_import_error": import_error,
        "research_profile": RESEARCH_PROFILE,
        "profile_rejected": profile_rejected,
        "profile_error": profile_error,
        "registry_research_entries": research_registry_entries,
        "provider_rejected": provider_rejected,
        "provider_error": provider_error,
        "registry_research_provider_entries": research_provider_entries,
    }


def resolve_local_checkpoint(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("MAPLE_MACE_POLAR_CHECKPOINT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            Path.home() / ".cache" / "mace" / "MACE-POLAR-1-M.model",
            Path.home() / ".cache" / "mace" / "polar-1-m.model",
        )
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            if resolved.stat().st_size != EXPECTED_CHECKPOINT_SIZE:
                continue
            if sha256_file(resolved) == EXPECTED_CHECKPOINT_SHA256:
                return resolved
    raise FileNotFoundError(
        "Official MACE-POLAR-1-M checkpoint is not locally available; "
        "downloads are prohibited for this ledger."
    )


def capture_v2_water_identity(checkpoint: Path) -> dict[str, object]:
    """Construct the actual CPU V2 water composition and freeze its identity."""

    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    from ase import Atoms
    from maple.solvation.api.units import HARTREE_TO_EV
    from maple.solvation.coupling.exact_gto import (
        mace_polar_learned_source_embedding_matrix,
    )
    from maple.solvation.experimental.mace_polar_response_native_fp64_v2 import (
        build_smd_mace_polar_response_native_fp64_v2_pes,
    )
    from maple.solvation.nonpolar.native_smd_cds_parameters import (
        COORDINATE_SCALE,
        LEGACY_HARTREE_TO_KCAL_MOL,
        NATIVE_LITERAL_CLASS_MANIFEST,
        UPSTREAM_MNSOL_SHA256,
    )

    checkpoint_before = sha256_file(checkpoint)
    atoms = Atoms(
        ("O", "H", "H"),
        positions=((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.239987, 0.927297, 0.0)),
    )
    atoms.info.update(charge=0, mult=1)
    pes = build_smd_mace_polar_response_native_fp64_v2_pes(
        atoms.get_chemical_symbols(),
        solvent="water",
        device="cpu",
        checkpoint_path=checkpoint,
    )
    configuration_before = pes.configuration_sha256()
    component_configuration = {
        "model": pes.model.configuration_sha256(),
        "ddpcm": pes.continuum.configuration_sha256(),
        "cds": pes.solvent_term.configuration_sha256(),
    }
    positions = torch.tensor(atoms.positions, dtype=torch.float64)
    vacuum, learned = pes.model.energy_source_torch(atoms, positions)
    embedding = torch.tensor(
        mace_polar_learned_source_embedding_matrix(), dtype=torch.float64
    )
    embedded = learned @ embedding.T
    inactive = embedded[:, INACTIVE_EMBEDDED_COLUMNS]
    if int(torch.count_nonzero(inactive)) != 0:
        raise RuntimeError("Frozen source embedding inactive columns are nonzero.")
    cds_state = pes.solvent_term.evaluate_torch(positions)
    polar = pes.continuum.energy_torch(positions, embedded)
    component_sum = float((vacuum + polar + cds_state.energy_eV).detach())
    replay_total = pes.get_potential_energy(atoms)
    decomposition_error = abs(component_sum - replay_total)
    if not math.isfinite(component_sum) or decomposition_error > 1e-10:
        raise RuntimeError(
            "V2 scalar component decomposition does not match total energy."
        )
    source_files = {
        name: sha256_file(ROOT / name)
        for name in (
            "maple/function/route2_solvents.py",
            "maple/solvation/api/units.py",
            "maple/solvation/nonpolar/native_smd_cds.py",
            "maple/solvation/nonpolar/native_smd_cds_parameters.py",
            "maple/solvation/nonpolar/legacy_smd_cds.py",
            "maple/solvation/surfaces/legacy_dareal.py",
        )
    }
    result = {
        "schema": "matrixfree-research-identity-ledger-v1",
        "water_geometry_sha256": canonical_sha(
            {
                "symbols": atoms.get_chemical_symbols(),
                "positions_angstrom": atoms.positions.tolist(),
            }
        ),
        "provider_id": pes.provider_id,
        "profile_id": pes.profile_id,
        "scalar_contract_id": pes.scalar_contract_id,
        "pes_configuration_sha256": configuration_before,
        "component_configuration_sha256": component_configuration,
        "ddpcm": {
            "configuration_payload": pes.continuum._configuration_payload(),
            "radii_angstrom": pes.continuum.radii_angstrom.tolist(),
            "dielectric": pes.continuum.dielectric,
            "lmax": pes.continuum.lmax,
            "n_lebedev": pes.continuum.n_lebedev,
            "eta": 0.1,
            "operator_order": ["A y = C rhs", "L z = y"],
        },
        "component_energies_eV": {
            "vacuum": float(vacuum.detach()),
            "ddpcm": float(polar.detach()),
            "cds": float(cds_state.energy_eV.detach()),
            "component_sum": component_sum,
            "public_programmatic_total_replay": replay_total,
            "sum_replay_absolute_error": decomposition_error,
            "tolerance": 1e-10,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_before,
            "size_bytes": checkpoint.stat().st_size,
        },
        "model_metadata": pes.model.metadata(),
        "source": {
            "raw_shape": list(learned.shape),
            "raw_total_charge_e": float(learned[:, 0].sum().detach()),
            "embedded_shape": list(embedded.shape),
            "raw_sha256": tensor_sha(learned),
            "embedded_sha256": tensor_sha(embedded),
            "vacuum_energy_sha256": tensor_sha(vacuum.reshape(1)),
            "embedding_matrix_sha256": tensor_sha(embedding),
            "active_columns": list(ACTIVE_EMBEDDED_COLUMNS),
            "inactive_columns": list(INACTIVE_EMBEDDED_COLUMNS),
            "inactive_nonzero_count": int(torch.count_nonzero(inactive)),
            "inactive_exact_zero": True,
        },
        "cds": {
            "configuration_sha256": component_configuration["cds"],
            "provenance_sha256": pes.solvent_term.provenance_sha256,
            "solvent": cds_state.diagnostics.solvent,
            "icds": cds_state.diagnostics.icds,
            "dareal_topology_sha256": cds_state.diagnostics.dareal.topology_sha256,
            "dareal_config": asdict(pes.solvent_term._config.dareal),
            "literal_class_manifest": dict(NATIVE_LITERAL_CLASS_MANIFEST),
            "upstream_mnsol_sha256": UPSTREAM_MNSOL_SHA256,
            "coordinate_scale": COORDINATE_SCALE,
            "legacy_hartree_to_kcal_mol": LEGACY_HARTREE_TO_KCAL_MOL,
            "hartree_to_ev": HARTREE_TO_EV,
            "source_files_sha256": source_files,
            "source_files_manifest_sha256": canonical_sha(source_files),
        },
        "runtime": {
            "device": "cpu",
            "torch_default_dtype": str(torch.get_default_dtype()),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "configuration_unchanged": pes.configuration_sha256() == configuration_before,
        "checkpoint_unchanged": sha256_file(checkpoint) == checkpoint_before,
        "scientific_admitted": False,
        "gate_complete": False,
        "ledger_is_model_accuracy_evidence": False,
    }
    if not result["configuration_unchanged"] or not result["checkpoint_unchanged"]:
        raise RuntimeError("V2 identity or checkpoint mutated during ledger capture.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload: dict[str, object] = {
        "schema": "matrixfree-research-identity-receipt-v1",
        "scientific_admitted": False,
        "gate_complete": False,
        "ledger_is_model_accuracy_evidence": False,
        "completed": False,
    }
    exit_status = 1
    try:
        payload["tracked_freeze"] = verify_tracked_freeze()
        payload["research_python_sha256"] = research_python_hashes()
        payload["public_unreachability"] = public_unreachability()
        checkpoint = resolve_local_checkpoint(args.checkpoint)
        payload["v2_water_identity"] = capture_v2_water_identity(checkpoint)
        payload["tracked_freeze_after"] = verify_tracked_freeze()
        payload["completed"] = True
        exit_status = 0
    except Exception as error:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    write_json_new(args.output, payload)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False))
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
