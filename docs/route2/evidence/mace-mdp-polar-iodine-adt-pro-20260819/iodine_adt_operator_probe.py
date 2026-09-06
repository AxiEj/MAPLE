#!/usr/bin/env python3
"""Cold-replay the target-free iodine ADT operator comparison.

The operational ECP-valence shape is loaded from the repository registry.  The
NR and spin-free-X2C all-electron references are rebuilt from the frozen radial
densities with the same Gaussian projection kernel used by the production
asset.  No MNSol target or untracked ``/tmp`` file is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


GEOMETRY_REPO_PATH = (
    "docs/route2/evidence/mace-mdp-polar-iodine-adt-pro-20260819/"
    "iodine_operator_geometry.json"
)
GENERATOR_REPO_PATH = "tools/route2_release/generate_iodine_adt_assets.py"
OPERATOR_REPO_PATH = (
    "docs/route2/evidence/mace-mdp-polar-iodine-adt-pro-20260819/"
    "iodine_adt_operator_probe.py"
)
OUTPUT_REPO_PATH = (
    "docs/route2/evidence/mace-mdp-polar-iodine-adt-pro-20260819/"
    "iodine_adt_operator_probe_output.json"
)


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


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_geometry(source_root: Path) -> dict[str, object]:
    path = source_root / GEOMETRY_REPO_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "artifact",
        "schema_version",
        "claim_boundary",
        "source",
        "formula",
        "charge",
        "multiplicity",
        "solvent",
        "atomic_numbers",
        "coordinates_angstrom",
        "payload_sha256",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("Iodine operator geometry schema changed.")
    stored = payload.pop("payload_sha256")
    if stored != _canonical_sha256(payload):
        raise RuntimeError("Iodine operator geometry payload SHA256 mismatch.")
    payload["payload_sha256"] = stored
    if payload["claim_boundary"] != {
        "experimental_solvation_target_embedded": False,
        "geometry_only": True,
        "purpose": "target-free-iodine-adt-operator-comparison",
    }:
        raise RuntimeError("Iodine geometry claim boundary changed.")
    return payload


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def run(source_root: Path) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from ase import Atoms, __version__ as ase_version
    import pyddx
    import scipy

    from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
        GaussianMixtureAtom,
    )
    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.separated_source_adt_ddx import (
        prepare_canonical_adt_separated_ddx,
    )
    from maple.solvation.continuum.separated_source_ddx import (
        SeparatedSourceDDXBackend,
    )
    from maple.solvation.coupling.adt_radial_shape import (
        load_repository_adt_radial_shape_registry,
    )
    from maple.solvation.coupling.atomic_displacement_lift import (
        CanonicalAtomicDisplacementLift,
    )
    from maple.solvation.coupling.neutral_atom_penetration import (
        NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH,
        NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
        load_neutral_atom_penetration_mixtures,
    )

    generator = _load_module(
        source_root / GENERATOR_REPO_PATH, "route2_iodine_adt_asset_generator"
    )
    geometry = _load_geometry(source_root)
    numbers = np.asarray(geometry["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(geometry["coordinates_angstrom"], dtype=np.float64)
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        info={
            "charge": int(geometry["charge"]),
            "multiplicity": int(geometry["multiplicity"]),
        },
    )
    symbols = tuple(atoms.get_chemical_symbols())
    solvent = route2_solvent_spec(str(geometry["solvent"]))
    radii = smd_coulomb_radii(symbols, solvent=solvent.name)
    backend = SeparatedSourceDDXBackend(
        symbols,
        radii,
        continuum_model="pcm",
        dielectric=solvent.descriptors.dielectric,
        lmax=8,
        n_lebedev=1202,
        solver_tolerance=1.0e-12,
        eta=0.1,
        n_proc=1,
    )
    prepared = backend.prepare(atoms, np.zeros((len(atoms), 4), dtype=np.float64))
    neutral = dict(
        load_neutral_atom_penetration_mixtures(
            table_path=source_root / NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
            manifest_path=(
                source_root / NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH
            ),
        )
    )

    registry = load_repository_adt_radial_shape_registry(source_root=source_root)
    lifts = {
        "ecp": CanonicalAtomicDisplacementLift.from_radial_shapes(
            atomic_numbers=numbers,
            registry=registry,
        )
    }
    for label in ("nr", "x2c"):
        counts, exponents, _metrics = generator.project_audit_mixture(
            source_root, label
        )
        mixtures = dict(neutral)
        mixtures[53] = GaussianMixtureAtom(counts, exponents)
        lifts[label] = CanonicalAtomicDisplacementLift.from_mixtures(
            atomic_numbers=numbers,
            mixtures_by_atomic_number=mixtures,
            source_asset_sha256=generator.RAW_DENSITIES[label]["sha256"],
        )

    outputs: dict[str, dict[str, object]] = {}
    molecular_dipole = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    for label, lift in lifts.items():
        direct = prepare_canonical_adt_separated_ddx(prepared, lift)
        dipoles = lift.lift_molecular_dipole(molecular_dipole)
        state = direct.solve(np.zeros((len(atoms), 4), dtype=np.float64), dipoles)
        outputs[label] = {
            "lift": lift,
            "dipoles": dipoles,
            "state": state,
            "phi": direct.adt_problem_data.phi_matrix,
            "psi": direct.adt_problem_data.psi_matrix,
        }

    iodine_atoms = np.flatnonzero(numbers == 53)
    if iodine_atoms.size != 1:
        raise RuntimeError("Frozen iodine operator geometry must contain one iodine.")
    iodine_columns = np.concatenate(
        [3 * iodine_atoms[:, None] + np.arange(3, dtype=np.int64)[None, :]]
    ).reshape(-1)

    per_shape: dict[str, object] = {}
    for label, output in outputs.items():
        lift = output["lift"]
        state = output["state"]
        dipoles = np.asarray(output["dipoles"])
        per_shape[label] = {
            "lift_configuration_sha256": lift.configuration_sha256,
            "self_work_hartree_per_ebohr2_sha256": _array_sha256(
                lift.self_work_hartree_per_ebohr2
            ),
            "atomic_dipole_weights_sha256": _array_sha256(
                lift.atomic_dipole_weights
            ),
            "iodine_atomic_dipole_weight": float(
                lift.atomic_dipole_weights[iodine_atoms[0]]
            ),
            "iodine_allocated_unit_dipole": dipoles[iodine_atoms].tolist(),
            "polarization_energy_eV": float(state.polarization_energy_ev),
            "model_field_l2_norm": float(np.linalg.norm(state.model_field8)),
            "phi_sha256": _array_sha256(output["phi"]),
            "psi_sha256": _array_sha256(output["psi"]),
        }

    comparisons: dict[str, object] = {}
    for left_label, right_label in (("ecp", "x2c"), ("nr", "x2c")):
        left = outputs[left_label]
        right = outputs[right_label]
        left_phi = np.asarray(left["phi"])[:, iodine_columns]
        right_phi = np.asarray(right["phi"])[:, iodine_columns]
        left_energy = float(left["state"].polarization_energy_ev)
        right_energy = float(right["state"].polarization_energy_ev)
        comparisons[f"{left_label}_vs_{right_label}"] = {
            "iodine_phi_maximum_absolute_difference": float(
                np.max(np.abs(left_phi - right_phi))
            ),
            "iodine_phi_relative_frobenius_difference": float(
                np.linalg.norm(left_phi - right_phi) / np.linalg.norm(right_phi)
            ),
            "psi_exact_match": bool(
                np.array_equal(np.asarray(left["psi"]), np.asarray(right["psi"]))
            ),
            "atomic_dipole_allocation_maximum_absolute_difference": float(
                np.max(
                    np.abs(
                        np.asarray(left["dipoles"])
                        - np.asarray(right["dipoles"])
                    )
                )
            ),
            "polarization_energy_difference_eV": left_energy - right_energy,
            "polarization_energy_relative_difference": abs(
                left_energy - right_energy
            )
            / max(abs(right_energy), 1.0e-30),
        }

    result: dict[str, object] = {
        "artifact": "route2-iodine-adt-operator-probe-v2",
        "schema_version": 2,
        "claim_boundary": {
            "capability_admitted": False,
            "experimental_solvation_target_read": False,
            "mace_output_read": False,
            "purpose": "target-free-role-separated-iodine-adt-operator-audit",
            "running_505_modified": False,
        },
        "source": {
            "geometry_path": GEOMETRY_REPO_PATH,
            "geometry_file_sha256": _sha256(source_root / GEOMETRY_REPO_PATH),
            "geometry_payload_sha256": geometry["payload_sha256"],
            "generator_path": GENERATOR_REPO_PATH,
            "generator_sha256": _sha256(source_root / GENERATOR_REPO_PATH),
            "operator_probe_path": OPERATOR_REPO_PATH,
            "operator_probe_sha256": _sha256(source_root / OPERATOR_REPO_PATH),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "ase": ase_version,
            "pyddx": getattr(pyddx, "__version__", "unknown"),
        },
        "continuum": {
            "model": "pcm",
            "solvent": solvent.name,
            "dielectric": float(solvent.descriptors.dielectric),
            "lmax": 8,
            "n_lebedev": 1202,
            "solver_tolerance": 1.0e-12,
            "eta": 0.1,
            "cavity_radii_angstrom": np.asarray(radii).tolist(),
            "prepared_configuration_sha256": prepared.configuration_sha256,
            "cavity_topology_sha256": prepared.cavity_topology_sha256,
        },
        "per_shape": per_shape,
        "comparisons": comparisons,
        "decision": {
            "operational_shape": "ecp-25-electron-valence-pseudodensity",
            "reference_shape": "x2c-53-electron-all-electron-density",
            "neutral_penetration_from_ecp_shape": "forbidden",
        },
    }
    result["payload_sha256"] = _canonical_sha256(result)
    return result


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.expanduser().resolve()
    result = run(source_root)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = args.output or source_root / OUTPUT_REPO_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
