#!/usr/bin/env python3
"""Bind a checkpoint-exact PCMSolver cavity-point metadata replay.

This is the second and final metadata-only repair of the matched-QM audit.
The parent v2 run passed the molecular-geometry check and stopped before any
response measurement because the frozen cavity-point coordinates inherited
the same rounded geometry.  All non-coordinate surface arrays and every
scientific protocol choice must remain bitwise unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, cast

from create_mace_mdp_polar_qm_boundary_response_metadata_remediation import (
    SOURCE_ROOT,
    _canonical_sha256,
    _file_record,
    _load_object,
    _sha256_file,
    _validate_parent,
)


SELF_REPO_PATH = (
    "tools/route2_release/"
    "create_mace_mdp_polar_qm_boundary_cavity_metadata_remediation.py"
)


def _validate_replacement(
    *,
    original: Path,
    replacement: Path,
    checkpoint: Path,
    pcm_input: Path,
    pcmsolver_library: Path,
) -> dict[str, object]:
    import numpy as np
    from pyscf import lib

    from maple.function.calculator.extra_correction.implicit.pcmsolver import (
        PCMSolverSession,
    )

    with np.load(original) as handle:
        old = {name: np.array(handle[name], copy=True) for name in handle.files}
    with np.load(replacement) as handle:
        new = {name: np.array(handle[name], copy=True) for name in handle.files}
    if set(old) != set(new):
        raise ValueError("replacement cavity NPZ schema drifted.")
    for name, values in old.items():
        if name != "surface_points_bohr" and not np.array_equal(values, new[name]):
            raise ValueError(f"replacement changed cavity array {name!r}.")

    molecule = lib.chkfile.load_mol(str(checkpoint))
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="route2-qm-cavity-remediation-") as td:
            os.chdir(td)
            with PCMSolverSession(
                np.asarray(molecule.atom_charges(), dtype=np.float64),
                np.asarray(molecule.atom_coords(), dtype=np.float64),
                pcm_input,
                library_path=pcmsolver_library,
            ) as session:
                replay = np.asarray(session.cavity_centers_bohr, dtype=np.float64)
    finally:
        os.chdir(previous)
    old_points = np.asarray(old["surface_points_bohr"], dtype=np.float64)
    new_points = np.asarray(new["surface_points_bohr"], dtype=np.float64)
    if not np.array_equal(new_points, replay):
        raise ValueError("replacement cavity points are not checkpoint replay-exact.")
    maximum_delta = float(np.max(np.abs(old_points - replay)))
    if not 1.0e-12 < maximum_delta < 1.0e-7:
        raise ValueError("cavity point metadata difference is not roundoff-sized.")
    return {
        "reason": "v2 stopped before response on rounded frozen cavity points",
        "change": "surface_points_bohr metadata only",
        "maximum_surface_point_delta_bohr": maximum_delta,
        "all_nonpoint_surface_arrays_bitwise_unchanged": True,
        "replacement_points_checkpoint_replay_exact": True,
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
    inputs = cast(dict[str, Any], parent["inputs"])
    case = cast(dict[str, Any], inputs["case"])
    files = cast(dict[str, Any], case["files"])
    runtime = cast(dict[str, Any], parent["runtime"])
    original_record = cast(dict[str, Any], files["frozen_surface"])
    checkpoint_record = cast(dict[str, Any], files["qm_checkpoint"])
    pcm_record = cast(dict[str, Any], files["pcm_input"])
    library_record = cast(dict[str, Any], runtime["pcmsolver_library"])
    original = Path(str(original_record["path"])).resolve(strict=True)
    checkpoint = Path(str(checkpoint_record["path"])).resolve(strict=True)
    pcm_input = Path(str(pcm_record["path"])).resolve(strict=True)
    library = Path(str(library_record["path"])).resolve(strict=True)
    for path, record, name in (
        (original, original_record, "surface"),
        (checkpoint, checkpoint_record, "checkpoint"),
        (pcm_input, pcm_record, "PCM input"),
        (library, library_record, "PCMSolver"),
    ):
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"parent {name} bytes drifted.")
    remediation = _validate_replacement(
        original=original,
        replacement=replacement,
        checkpoint=checkpoint,
        pcm_input=pcm_input,
        pcmsolver_library=library,
    )

    payload = cast(dict[str, Any], deepcopy(parent))
    payload.pop("preregistration_sha256", None)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    payload["parent_cavity_failed_preregistration_path"] = str(parent_path)
    payload["parent_cavity_failed_preregistration_file_sha256"] = _sha256_file(
        parent_path
    )
    payload["cavity_metadata_only_remediation"] = remediation
    payload["inputs"]["case"]["files"]["frozen_surface"] = _file_record(
        replacement
    )
    payload["inputs"]["case"]["cavity_geometry_metadata_rebind"] = remediation
    payload["source_files_sha256"][SELF_REPO_PATH] = _sha256_file(
        SOURCE_ROOT / SELF_REPO_PATH
    )
    payload["decision_contract"][
        "cavity_metadata_only_remediation_after_prephysics_failure"
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
