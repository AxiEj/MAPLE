#!/usr/bin/env python3
"""Create a metadata-only replacement for the failed matched-QM preregistration.

The v1 execution stopped before any induced-response measurement because an
NPZ geometry copied through rounded Angstrom text differed from the PySCF
checkpoint by 2.13e-9 Angstrom.  This creator accepts a replacement NPZ only
when every scientific array is bitwise unchanged and the sole modified array,
``atom_positions_angstrom``, is exactly reconstructed from the already frozen
checkpoint.  It then emits a new self-hashed preregistration for the unchanged
v1 runner without mutating the failed v1 evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, cast


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = (
    "tools/route2_release/"
    "create_mace_mdp_polar_qm_boundary_response_metadata_remediation.py"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _load_object(path: Path, *, name: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object.")
    return payload


def _validate_parent(parent: dict[str, object]) -> None:
    if (
        parent.get("artifact")
        != "route2-mace-mdp-polar-matched-qm-boundary-response-prereg-v1"
        or parent.get("schema_version") != 1
        or parent.get("status")
        != "frozen-before-first-qm-induced-boundary-response-execution"
    ):
        raise ValueError("parent matched-QM preregistration identity drifted.")
    unsigned = dict(parent)
    observed = unsigned.pop("preregistration_sha256", None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError("parent matched-QM preregistration self hash drifted.")
    source_hashes = parent.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("parent source manifest is missing.")
    for relative, expected in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("parent source manifest is malformed.")
        if _sha256_file(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"parent source {relative!r} drifted.")


def _validate_replacement(
    *,
    original: Path,
    replacement: Path,
    checkpoint: Path,
) -> dict[str, object]:
    import numpy as np
    from ase.units import Bohr
    from pyscf import lib

    with np.load(original) as handle:
        old = {name: np.array(handle[name], copy=True) for name in handle.files}
    with np.load(replacement) as handle:
        new = {name: np.array(handle[name], copy=True) for name in handle.files}
    if set(old) != set(new):
        raise ValueError("replacement QM MEP NPZ schema drifted.")
    for name, values in old.items():
        if name != "atom_positions_angstrom" and not np.array_equal(
            values, new[name]
        ):
            raise ValueError(f"replacement changed scientific array {name!r}.")
    molecule = lib.chkfile.load_mol(str(checkpoint))
    checkpoint_positions = (
        np.asarray(molecule.atom_coords(), dtype=np.float64) * Bohr
    )
    old_positions = np.asarray(old["atom_positions_angstrom"], dtype=np.float64)
    new_positions = np.asarray(new["atom_positions_angstrom"], dtype=np.float64)
    if not np.array_equal(new_positions, checkpoint_positions):
        raise ValueError("replacement positions are not checkpoint-exact.")
    maximum_delta = float(np.max(np.abs(old_positions - checkpoint_positions)))
    if not 2.0e-10 < maximum_delta < 5.0e-9:
        raise ValueError("geometry metadata difference is not roundoff-sized.")
    return {
        "reason": "v1 stopped before induced response on rounded geometry metadata",
        "change": "atom_positions_angstrom metadata only",
        "maximum_position_delta_angstrom": maximum_delta,
        "all_nonposition_npz_arrays_bitwise_unchanged": True,
        "replacement_positions_checkpoint_exact": True,
        "original_file_sha256": _sha256_file(original),
        "replacement_file_sha256": _sha256_file(replacement),
        "method_or_threshold_changed": False,
        "experimental_solvation_target_read": False,
    }


def create(args: argparse.Namespace) -> dict[str, object]:
    parent_path = args.parent.expanduser().resolve(strict=True)
    replacement = args.replacement.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = _load_object(parent_path, name="parent preregistration")
    _validate_parent(parent)
    inputs = parent.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("parent inputs are missing.")
    case = inputs.get("case")
    if not isinstance(case, dict):
        raise ValueError("parent case is missing.")
    files = case.get("files")
    if not isinstance(files, dict):
        raise ValueError("parent case files are missing.")
    original_record = files.get("frozen_qm_surface_mep")
    checkpoint_record = files.get("qm_checkpoint")
    if not isinstance(original_record, dict) or not isinstance(
        checkpoint_record, dict
    ):
        raise ValueError("parent QM MEP/checkpoint records are missing.")
    original = Path(str(original_record["path"])).resolve(strict=True)
    checkpoint = Path(str(checkpoint_record["path"])).resolve(strict=True)
    if _sha256_file(original) != original_record.get("sha256"):
        raise ValueError("parent frozen QM MEP bytes drifted.")
    if _sha256_file(checkpoint) != checkpoint_record.get("sha256"):
        raise ValueError("parent QM checkpoint bytes drifted.")
    remediation = _validate_replacement(
        original=original,
        replacement=replacement,
        checkpoint=checkpoint,
    )

    payload = cast(dict[str, Any], deepcopy(parent))
    payload.pop("preregistration_sha256", None)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["parent_failed_preregistration_path"] = str(parent_path)
    payload["parent_failed_preregistration_file_sha256"] = _sha256_file(parent_path)
    payload["metadata_only_remediation"] = remediation
    payload["inputs"]["case"]["files"]["frozen_qm_surface_mep"] = _file_record(
        replacement
    )
    payload["inputs"]["case"]["qm_mep_geometry_metadata_rebind"] = remediation
    payload["source_files_sha256"][SELF_REPO_PATH] = _sha256_file(
        SOURCE_ROOT / SELF_REPO_PATH
    )
    payload["decision_contract"][
        "metadata_only_remediation_after_prephysics_failure"
    ] = True
    payload["preregistration_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "file_sha256": _sha256_file(args.output.expanduser().resolve()),
                "preregistration_sha256": payload["preregistration_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
