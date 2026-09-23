#!/usr/bin/env python3
"""Pinned upstream SvdW-DROP atom-area experiment; no model admission or fit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
INVALID_V1 = HERE.parent / "upstream-swig-bakeoff-20260923"
PANEL = (
    ROOT / ".omx/research/pure-torch-analytic-20260922/"
    "ten-molecule-panel-20260923/run-1"
)
MOIST_SOURCE = Path("/tmp/maple-moist-svdw-drop-main-20260923")
MOIST_COMMIT = "a0116e8ee369c546d48558e5e389e6f20bdff2e9"
NLEB = (110, 302, 590)
ROTATION_COUNT = 4
ROTATION_SEED = 20260923


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _check_moist() -> str:
    import subprocess
    import moist

    commit = subprocess.check_output(
        ["git", "-C", str(MOIST_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != MOIST_COMMIT:
        raise RuntimeError("Pinned MOIST source commit changed.")
    if not str(Path(moist.__file__).resolve()).startswith(
        str((MOIST_SOURCE / "build-maple/python/moist").resolve())
    ):
        raise RuntimeError("MOIST is not loaded from the isolated pinned build.")
    return commit


def prepare() -> None:
    if (HERE / "moist-protocol.json").exists():
        raise FileExistsError("MOIST protocol is already frozen.")
    commit = _check_moist()
    cases = json.loads((PANEL / "protocol.json").read_text())["cases"]
    if len(cases) != 10:
        raise RuntimeError("The frozen ten-molecule panel changed.")
    _write(
        HERE / "moist-protocol.json",
        {
            "status": "frozen-before-batch",
            "purpose": "MOIST SvdW-DROP atom-area/legacy-SMD-CDS substitution diagnostic",
            "case_names": [case["name"] for case in cases],
            "case_protocol_sha256": _sha(PANEL / "protocol.json"),
            "invalid_v1_protocol_sha256": _sha(INVALID_V1 / "moist-protocol.json"),
            "invalid_v1_result_sha256": _sha(INVALID_V1 / "moist-results.json"),
            "correction": "pass positions (natoms,3), then read back native sphxyz",
            "script_sha256": _sha(Path(__file__)),
            "moist_commit": commit,
            "moist_source_version_file": (MOIST_SOURCE / "VERSION").read_text().strip(),
            "moist_build": "isolated Meson Python extension, debug, no project install",
            "moist_cavity": "SvdW-DROP C API custom per-atom radii, default blend_k=2.0",
            "radii": "SMD legacy CDS Bondi/Mantina plus 0.4 Angstrom",
            "tensions": "unchanged legacy SMD atom tensions and solvent constant",
            "nleb": NLEB,
            "rotation_count": ROTATION_COUNT,
            "rotation_seed": ROTATION_SEED,
            "rotation_nleb": 302,
            "metrics": (
                "per-atom and total area vs DAREAL; unfit CDS energy change; "
                "four rigid-rotation area and CDS-energy deviations"
            ),
            "fit_performed": False,
            "experimental_accuracy_tested": False,
            "selection_rule": "all ten preregistered neutral geometries and all failures retained",
        },
    )


def _moist_areas(atoms: Atoms, radii_angstrom: np.ndarray, nleb: int) -> dict:
    """Only bridge missing from MOIST's high-level API: custom-radii constructor."""
    import moist
    from moist import library

    ffi, lib = library.ffi, library.lib
    radii_bohr = np.ascontiguousarray(radii_angstrom / Bohr, dtype=np.float64)
    radii_handle = library.error_check(lib.moist_new_custom_radii)()
    cavity_handle = ffi.NULL
    structure = None
    try:
        library.error_check(lib.moist_set_custom_radii_atoms)(
            radii_handle,
            len(atoms),
            ffi.cast("double *", ffi.from_buffer(radii_bohr)),
        )
        cavity_handle = library.error_check(lib.moist_new_drop_cavity_with_radii)(
            radii_handle, ffi.new("int *", nleb), *([ffi.NULL] * 13)
        )
        structure = moist.Structure(
            np.asarray(atoms.numbers, dtype=np.int32),
            np.ascontiguousarray(atoms.positions / Bohr, dtype=np.float64),
        )
        library.error_check(lib.moist_update_cavity)(
            cavity_handle, structure._as_handle().handle
        )
        borrowed = library.CavityHandle(cavity_handle)
        result = library.get_cavity_results(borrowed)
        centers_bohr = library.get_cavity_field(borrowed, "sphxyz")
        if centers_bohr.shape != (3, len(atoms)) or not np.allclose(
            centers_bohr.T * Bohr, atoms.positions, rtol=0.0, atol=1.0e-10
        ):
            raise RuntimeError("DROP native sphere centres differ from input geometry.")
        if result["nsph"] != len(atoms):
            raise RuntimeError("DROP sphere count differs from atom count.")
        if not np.allclose(result["radii"] * Bohr, radii_angstrom, atol=1.0e-10):
            raise RuntimeError("DROP did not use the requested SMD CDS radii.")
        if not np.all(result["converged"]):
            raise RuntimeError("At least one DROP projection did not converge.")
        areas = np.asarray(result["asph"] * Bohr**2, dtype=float)
        if abs(float(areas.sum()) - result["area"] * Bohr**2) > 1.0e-8:
            raise RuntimeError("DROP per-sphere areas do not sum to total area.")
        return {
            "atom_area_A2": areas.tolist(),
            "total_area_A2": float(areas.sum()),
            "surface_points": int(result["ngrid"]),
        }
    finally:
        if cavity_handle != ffi.NULL:
            lib.moist_delete_cavity(ffi.new("moist_cavity *", cavity_handle))
        lib.moist_delete_radii(ffi.new("moist_radii *", radii_handle))
        del structure
        gc.collect()


def run() -> None:
    import torch
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_sasa_radii,
    )
    from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

    protocol = json.loads((HERE / "moist-protocol.json").read_text())
    if (
        protocol["script_sha256"] != _sha(Path(__file__))
        or protocol["case_protocol_sha256"] != _sha(PANEL / "protocol.json")
        or protocol["moist_commit"] != _check_moist()
    ):
        raise RuntimeError("MOIST benchmark identity drifted.")
    if (HERE / "moist-results.json").exists():
        raise FileExistsError("MOIST result exists; do not overwrite it.")
    cases = json.loads((PANEL / "protocol.json").read_text())["cases"]
    rotations = Rotation.random(
        ROTATION_COUNT, random_state=np.random.default_rng(ROTATION_SEED)
    ).as_matrix()
    rows = []
    output = {
        "protocol_sha256": _sha(HERE / "moist-protocol.json"),
        "rows": rows,
        "fit_performed": False,
        "experimental_accuracy_tested": False,
    }
    for case in cases:
        row = {"name": case["name"], "grid": {}, "rotation": []}
        rows.append(row)
        print(case["name"], flush=True)
        try:
            atoms = Atoms(case["symbols"], positions=case["positions_angstrom"])
            symbols = tuple(atoms.get_chemical_symbols())
            radii = smd_sasa_radii(symbols)
            term = TorchLegacySMDCDS(symbols, "water", device="cpu")
            reference = term.evaluate_torch(
                torch.tensor(atoms.positions, dtype=torch.float64)
            )
            ref_area = reference.atom_areas_angstrom2.detach().numpy()
            tensions = reference.atom_tensions_cal_mol_angstrom2.detach().numpy()
            # This is the already-published legacy scalar's per-area coefficient,
            # not a newly fitted or optimized parameter.
            effective_tensions = tensions + term._cssigma
            row["radii_A"] = radii.tolist()
            row["dareal_atom_area_A2"] = ref_area.tolist()
            row["dareal_cds_eV"] = float(reference.energy_eV.detach())
            row["effective_tensions_cal_mol_A2"] = effective_tensions.tolist()
            for nleb in NLEB:
                key = str(nleb)
                try:
                    result = _moist_areas(atoms, radii, nleb)
                    area = np.asarray(result["atom_area_A2"])
                    energy = (
                        float(np.dot(area, effective_tensions))
                        * 0.001
                        * Hartree
                        / 627.5094740631
                    )
                    result.update(
                        cds_energy_eV=energy,
                        delta_cds_eV=energy - row["dareal_cds_eV"],
                        max_atom_area_delta_A2=float(np.max(np.abs(area - ref_area))),
                        total_area_delta_A2=float(area.sum() - ref_area.sum()),
                    )
                    row["grid"][key] = result
                except Exception as error:
                    row["grid"][key] = {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            if "cds_energy_eV" in row["grid"]["302"]:
                center = row["grid"]["302"]
                for rotation in rotations:
                    moved = atoms.copy()
                    moved.positions = atoms.positions @ rotation.T
                    try:
                        result = _moist_areas(moved, radii, 302)
                        area = np.asarray(result["atom_area_A2"])
                        energy = (
                            float(np.dot(area, effective_tensions))
                            * 0.001
                            * Hartree
                            / 627.5094740631
                        )
                        row["rotation"].append(
                            {
                                "max_atom_area_delta_A2": float(
                                    np.max(np.abs(area - center["atom_area_A2"]))
                                ),
                                "total_area_delta_A2": float(
                                    area.sum() - center["total_area_A2"]
                                ),
                                "cds_energy_delta_eV": energy - center["cds_energy_eV"],
                            }
                        )
                    except Exception as error:
                        row["rotation"].append(
                            {"error_type": type(error).__name__, "error": str(error)}
                        )
        except Exception as error:
            row["case_error"] = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
        _write(HERE / "moist-progress.json", output)
    _write(HERE / "moist-results.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run()
