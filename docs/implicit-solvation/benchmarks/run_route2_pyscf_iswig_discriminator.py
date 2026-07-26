#!/usr/bin/env python3
"""Locked fixed-density SWIG/ISWIG rigid-rotation discriminator.

This ignored research runner executes the exact budget registered in
docs/implicit-solvation/benchmarks/
route2-pyscf-iswig-discriminator-prereg-v1.json.  It does not load MACE,
iterate an ML-SCF root, evaluate CDS, or modify a public Route-2 provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve


ARTIFACT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = ARTIFACT_DIR.parents[2]
PREREGISTRATION_PATH = (
    REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-pyscf-iswig-discriminator-prereg-v1.json"
)
DATASET_ROOT = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/route2-macepolar-smd/dataset/mol2files_gaff"
)
STATE_ROOT = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/route2-macepolar-smd/provider-audit"
)
PYSCF_VALIDATION_PATH = Path(
    os.environ.get("PYSCF_VALIDATION_PATH", "")
).expanduser()
if not PYSCF_VALIDATION_PATH.is_dir():
    raise RuntimeError(
        "Set PYSCF_VALIDATION_PATH to the isolated PySCF 2.13.1 "
        "site-packages directory."
    )

sys.path.insert(0, str(PYSCF_VALIDATION_PATH))
sys.path.insert(0, str(REPOSITORY_ROOT))

from ase.units import Bohr  # noqa: E402
from pyscf import __version__ as pyscf_version, gto  # noqa: E402
from pyscf.dft import gen_grid  # noqa: E402
from pyscf.solvent import pcm as pyscf_pcm  # noqa: E402

from maple.function.calculator.extra_correction.implicit.gto_density import (  # noqa: E402
    density_reaction_coupling,
    density_to_external_field_order,
    external_field_to_density_order,
    point_asc_reaction_potential_gradient,
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    route2_water_coulomb_radii,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402


KCAL_PER_HARTREE = 627.5094740631


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=float)
    unit_axis /= np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    cross = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(unit_axis, unit_axis)
        + math.sin(angle) * cross
    )


def rotated_density(
    density_coefficients: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    external = density_to_external_field_order(density_coefficients)
    external[:, 1:] = external[:, 1:] @ rotation.T
    return external_field_to_density_order(external)


class _AtomIndexedSurfaceMolecule:
    """Expose atom indices as element keys so PySCF accepts per-atom radii."""

    def __init__(self, molecule: Any) -> None:
        self._molecule = molecule
        self.natm = int(molecule.natm)
        self.elements = tuple(range(self.natm))

    def atom_coords(self, unit: str = "B") -> np.ndarray:
        return np.asarray(self._molecule.atom_coords(unit=unit), dtype=float)


def molecule(
    symbols: tuple[str, ...],
    positions_angstrom: np.ndarray,
):
    return gto.M(
        atom=list(zip(symbols, positions_angstrom.tolist(), strict=True)),
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )


def parent_counts(surface: dict[str, Any], atom_count: int) -> list[int]:
    slices = surface["gslice_by_atom"]
    if len(slices) != atom_count:
        raise RuntimeError("PySCF surface atom slices changed cardinality.")
    counts = []
    cursor = 0
    for start, stop in slices:
        start = int(start)
        stop = int(stop)
        if start != cursor or stop < start:
            raise RuntimeError("PySCF surface atom slices are not contiguous.")
        counts.append(stop - start)
        cursor = stop
    if cursor != int(np.asarray(surface["grid_coords"]).shape[0]):
        raise RuntimeError("PySCF surface slices do not cover all points.")
    return counts


def evaluate(
    *,
    symbols: tuple[str, ...],
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    density_coefficients: np.ndarray,
    dielectric: float,
    lebedev_order: int,
    surface_method: str,
) -> dict[str, Any]:
    mol = molecule(symbols, positions_angstrom)
    grid_points_per_atom = int(gen_grid.LEBEDEV_ORDER[lebedev_order])
    surface = pyscf_pcm.gen_surface(
        _AtomIndexedSurfaceMolecule(mol),
        ng=grid_points_per_atom,
        rad=np.asarray(radii_angstrom, dtype=float) / Bohr,
        surface_discretization_method=surface_method,
    )
    _, area = pyscf_pcm.get_F_A(surface)
    D, S = pyscf_pcm.get_D_S(surface, with_S=True, with_D=True)
    f_epsilon = (dielectric - 1.0) / (dielectric + 1.0)
    DA = D * area
    K = S - f_epsilon / (2.0 * math.pi) * (DA @ S)
    R = -f_epsilon * (
        np.eye(K.shape[0]) - DA / (2.0 * math.pi)
    )
    lu_and_piv = lu_factor(K, check_finite=True)
    centers_bohr = np.asarray(surface["grid_coords"], dtype=float)
    mep = point_multipole_potential(
        centers_bohr,
        positions_angstrom,
        density_coefficients,
    )
    q = lu_solve(lu_and_piv, R @ mep, check_finite=True)
    v_k_inverse = lu_solve(
        lu_and_piv,
        mep,
        trans=1,
        check_finite=True,
    )
    q_symmetric = 0.5 * (q + R.T @ v_k_inverse)
    energy_hartree = 0.5 * float(np.dot(q_symmetric, mep))
    reaction_potential, reaction_gradient = (
        point_asc_reaction_potential_gradient(
            positions_angstrom,
            centers_bohr,
            q_symmetric,
        )
    )
    surface_coupling = float(np.dot(mep, q_symmetric))
    multipole_coupling = density_reaction_coupling(
        density_coefficients,
        reaction_potential,
        reaction_gradient,
    )
    identity_error = max(
        abs(2.0 * energy_hartree - surface_coupling),
        abs(surface_coupling - multipole_coupling),
    )
    values = (
        energy_hartree,
        identity_error,
        np.asarray(area),
        np.asarray(q_symmetric),
    )
    if not all(np.all(np.isfinite(value)) for value in values):
        raise RuntimeError("PySCF continuum solve produced non-finite data.")
    return {
        "energy_hartree": energy_hartree,
        "half_coupling_identity_error_hartree": identity_error,
        "parent_counts": parent_counts(surface, len(symbols)),
        "surface_size": int(centers_bohr.shape[0]),
    }


def load_case(spec: dict[str, Any]) -> dict[str, Any]:
    compound_id = str(spec["compound_id"])
    mol2_path = DATASET_ROOT / f"{compound_id}.mol2"
    density_path = (
        STATE_ROOT
        / compound_id
        / "maple.out.implicit/route2-state.npz"
    )
    if sha256(mol2_path) != spec["mol2_sha256"]:
        raise RuntimeError(f"{compound_id} MOL2 hash does not match the lock.")
    if sha256(density_path) != spec["density_npz_sha256"]:
        raise RuntimeError(
            f"{compound_id} density archive hash does not match the lock."
        )
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    symbols = tuple(atoms.get_chemical_symbols())
    atom_types = tuple(atoms.info["mol2"]["atom_types"])
    radii = route2_water_coulomb_radii(
        symbols,
        atom_types=atom_types,
        profile="smd-iefpcm-gaff2-o",
    )
    with np.load(density_path) as archive:
        density = np.asarray(
            archive["solvent_density_coefficients"],
            dtype=float,
        )
    return {
        "atoms": atoms,
        "density": density,
        "radii_angstrom": radii,
    }


def main() -> None:
    registration = json.loads(
        PREREGISTRATION_PATH.read_text(encoding="utf-8")
    )
    if registration["status"] != "pre-registered":
        raise RuntimeError("The ISWIG discriminator is not pre-registered.")
    if pyscf_version != "2.13.1":
        raise RuntimeError(
            f"Expected PySCF 2.13.1, found {pyscf_version}."
        )

    rotation_spec = registration["rotation"]
    axis = np.asarray(rotation_spec["axis"], dtype=float)
    angles = tuple(float(x) for x in rotation_spec["angles_radians"])
    records = []
    for case_spec in registration["fixed_inputs"]:
        loaded = load_case(case_spec)
        atoms = loaded["atoms"]
        base_positions = np.asarray(atoms.get_positions(), dtype=float)
        center = np.mean(base_positions, axis=0)
        centered = base_positions - center
        base_density = np.asarray(loaded["density"], dtype=float)
        for method in ("SWIG", "ISWIG"):
            orientations = []
            for angle in angles:
                rotation = rotation_matrix(axis, angle)
                positions = centered @ rotation.T + center
                density = rotated_density(base_density, rotation)
                result = evaluate(
                    symbols=tuple(atoms.get_chemical_symbols()),
                    positions_angstrom=positions,
                    radii_angstrom=np.asarray(
                        loaded["radii_angstrom"], dtype=float
                    ),
                    density_coefficients=density,
                    dielectric=float(registration["candidate"]["dielectric"]),
                    lebedev_order=int(
                        registration["candidate"]["lebedev_order"]
                    ),
                    surface_method=method,
                )
                orientations.append(
                    {
                        "angle_radians": angle,
                        **result,
                    }
                )
            energies = [
                float(item["energy_hartree"]) for item in orientations
            ]
            records.append(
                {
                    "compound_id": case_spec["compound_id"],
                    "name": case_spec["name"],
                    "surface_discretization_method": method,
                    "orientations": orientations,
                    "rotation_span_hartree": max(energies) - min(energies),
                    "rotation_span_kcal_mol": (
                        max(energies) - min(energies)
                    )
                    * KCAL_PER_HARTREE,
                }
            )

    expected_solves = int(
        registration["exact_evaluation_budget"]["continuum_scalar_solves"]
    )
    actual_solves = sum(
        len(record["orientations"]) for record in records
    )
    by_key = {
        (
            str(record["compound_id"]),
            str(record["surface_discretization_method"]),
        ): record
        for record in records
    }
    thresholds = registration["locked_gates"]
    candidate_records = [
        record
        for record in records
        if record["surface_discretization_method"] == "ISWIG"
    ]
    all_orientations = [
        item
        for record in records
        for item in record["orientations"]
    ]
    rotation_ratios: dict[str, float] = {}
    not_worse = True
    ratio_gate = True
    for candidate in candidate_records:
        compound_id = str(candidate["compound_id"])
        control = by_key[(compound_id, "SWIG")]
        control_span = float(control["rotation_span_kcal_mol"])
        candidate_span = float(candidate["rotation_span_kcal_mol"])
        ratio = (
            candidate_span / control_span
            if control_span > 0.0
            else math.inf
        )
        rotation_ratios[compound_id] = ratio
        not_worse &= candidate_span <= control_span
        ratio_gate &= ratio <= float(
            thresholds["candidate_rotation_span_ratio_to_control_maximum"]
        )

    gates = {
        "exact_evaluation_budget": actual_solves == expected_solves,
        "all_surface_solves_finite": all(
            math.isfinite(float(item["energy_hartree"]))
            for item in all_orientations
        ),
        "maximum_half_coupling_identity_error": max(
            float(item["half_coupling_identity_error_hartree"])
            for item in all_orientations
        )
        <= float(
            thresholds[
                "maximum_half_coupling_identity_error_hartree"
            ]
        ),
        "candidate_maximum_rotation_span": max(
            float(record["rotation_span_kcal_mol"])
            for record in candidate_records
        )
        <= float(
            thresholds["candidate_maximum_rotation_span_kcal_mol"]
        ),
        "candidate_parent_counts_constant_across_orientations": all(
            len(
                {
                    tuple(item["parent_counts"])
                    for item in record["orientations"]
                }
            )
            == 1
            for record in candidate_records
        ),
        "candidate_rotation_span_not_greater_than_control": not_worse,
        "candidate_rotation_span_ratio_to_control": ratio_gate,
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload = {
        "claim_boundary": registration["claim_boundary"],
        "failed_gates": failed,
        "gates": gates,
        "git_head": git_output("rev-parse", "HEAD"),
        "preregistration_path": str(
            PREREGISTRATION_PATH.relative_to(REPOSITORY_ROOT)
        ),
        "preregistration_sha256": sha256(PREREGISTRATION_PATH),
        "protocol_id": registration["protocol_id"],
        "pyscf": {
            "path": str(PYSCF_VALIDATION_PATH),
            "version": pyscf_version,
        },
        "records": records,
        "rotation_span_ratios_iswig_over_swig": rotation_ratios,
        "schema_version": 1,
        "status": "pass" if not failed else "fail",
        "stop_condition": registration["decision_rule"],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACT_DIR / "results.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "failed_gates": failed,
                "output": str(output_path),
                "rotation_span_ratios": rotation_ratios,
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
