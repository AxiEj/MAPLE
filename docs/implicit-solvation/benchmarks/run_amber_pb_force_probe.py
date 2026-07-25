#!/usr/bin/env python3
"""Audit an Amber PBSA inp=2 exact-difference correction against its energy FD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np

try:
    import sander
    from parmed.amber import Rst7
except ImportError as exc:
    raise SystemExit(
        "This probe requires AmberTools' Python environment, for example "
        "`$AMBER_PREFIX/bin/python run_amber_pb_force_probe.py`."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_INPUT_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/route1-performance-20260724/mobley_1017962"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "amber-pb-inp2-force-probe-methyl-hexanoate-2026-07-24.json"
)
DEFAULT_COMPONENTS = ((0, 0), (6, 0), (7, 0), (8, 0))
DEFAULT_STEPS_ANGSTROM = (0.003, 0.01)
FORCE_TOLERANCE_KCAL_MOL_ANGSTROM = 0.05


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _vacuum_options():
    """Return identical MM terms with a zero-dielectric-contrast GB term."""
    options = sander.gas_input(5)
    options.igb = 5
    options.ipb = 0
    options.inp = 0
    options.gbsa = 0
    options.intdiel = 1.0
    options.extdiel = 1.0
    return options


def _pb_inp2_options():
    options = sander.gas_input(10)
    options.igb = 10
    options.ipb = 1
    options.inp = 2
    return options


def _energy_terms(energy) -> dict[str, float]:
    names = (
        "tot",
        "bond",
        "angle",
        "dihedral",
        "vdw",
        "elec",
        "vdw_14",
        "elec_14",
        "gb",
        "pb",
        "surf",
        "disp",
    )
    return {
        name: float(getattr(energy, name)) for name in names if hasattr(energy, name)
    }


def _coordinate_key(step_angstrom: float, atom_index: int, axis: int, sign: int) -> str:
    return f"{step_angstrom:.12g}:{atom_index}:{axis}:{sign:+d}"


def _evaluation_points(
    positions: np.ndarray,
    components: Iterable[tuple[int, int]],
    steps_angstrom: Iterable[float],
) -> list[tuple[str, np.ndarray]]:
    points = [("reference", positions.copy())]
    for step in steps_angstrom:
        for atom_index, axis in components:
            for sign in (-1, 1):
                displaced = positions.copy()
                displaced[atom_index, axis] += sign * step
                points.append(
                    (
                        _coordinate_key(step, atom_index, axis, sign),
                        displaced,
                    )
                )
    return points


def _evaluate_context(
    prmtop: Path,
    reference_positions: np.ndarray,
    points: list[tuple[str, np.ndarray]],
    options,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    # PBSA's library state is process-global. Reusing one initialized context
    # for every displaced point avoids unsupported repeated PB initialization.
    with sander.setup(str(prmtop), reference_positions, None, options) as context:
        for key, positions in points:
            context.positions = positions
            energy, forces = context.energy_forces()
            results[key] = {
                "energy": _energy_terms(energy),
                "forces_kcal_mol_angstrom": np.asarray(forces, dtype=float).tolist(),
            }
    return results


def _parse_components(raw: str, atom_count: int) -> tuple[tuple[int, int], ...]:
    components: list[tuple[int, int]] = []
    for token in raw.split(","):
        atom_text, axis_text = token.strip().split(":", maxsplit=1)
        atom_index = int(atom_text)
        axis = int(axis_text)
        if not 0 <= atom_index < atom_count:
            raise ValueError(f"Atom index outside [0, {atom_count}): {atom_index}.")
        if axis not in {0, 1, 2}:
            raise ValueError(f"Cartesian axis must be 0, 1, or 2: {axis}.")
        components.append((atom_index, axis))
    if not components or len(components) != len(set(components)):
        raise ValueError("Force components must be non-empty and unique.")
    return tuple(components)


def _parse_steps(raw: str) -> tuple[float, ...]:
    steps = tuple(float(value) for value in raw.split(","))
    if not steps or any(not np.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError("Finite-difference steps must be finite and positive.")
    return steps


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    prmtop = Path(args.prmtop).resolve()
    rst7_path = Path(args.rst7).resolve()
    if not prmtop.is_file() or not rst7_path.is_file():
        raise FileNotFoundError("Both --prmtop and --rst7 must exist.")
    restart = Rst7.open(str(rst7_path))
    positions = np.asarray(restart.coordinates, dtype=float).reshape((-1, 3))
    components = _parse_components(args.components, len(positions))
    steps = _parse_steps(args.steps)
    points = _evaluation_points(positions, components, steps)

    vacuum = _evaluate_context(prmtop, positions, points, _vacuum_options())
    solution = _evaluate_context(prmtop, positions, points, _pb_inp2_options())
    reference_vacuum = vacuum["reference"]
    reference_solution = solution["reference"]
    correction_forces = np.asarray(
        reference_solution["forces_kcal_mol_angstrom"], dtype=float
    ) - np.asarray(reference_vacuum["forces_kcal_mol_angstrom"], dtype=float)

    samples: list[dict[str, Any]] = []
    for step in steps:
        for atom_index, axis in components:
            minus_key = _coordinate_key(step, atom_index, axis, -1)
            plus_key = _coordinate_key(step, atom_index, axis, 1)
            minus_energy = (
                solution[minus_key]["energy"]["tot"]
                - vacuum[minus_key]["energy"]["tot"]
            )
            plus_energy = (
                solution[plus_key]["energy"]["tot"] - vacuum[plus_key]["energy"]["tot"]
            )
            finite_difference_force = -(plus_energy - minus_energy) / (2.0 * step)
            reported_force = float(correction_forces[atom_index, axis])
            error = abs(reported_force - finite_difference_force)
            samples.append(
                {
                    "step_angstrom": step,
                    "atom_index": atom_index,
                    "axis": axis,
                    "reported_exact_difference_force_kcal_mol_angstrom": (
                        reported_force
                    ),
                    "centered_energy_fd_force_kcal_mol_angstrom": (
                        finite_difference_force
                    ),
                    "absolute_error_kcal_mol_angstrom": error,
                    "within_tolerance": (error <= FORCE_TOLERANCE_KCAL_MOL_ANGSTROM),
                }
            )

    solution_energy = reference_solution["energy"]
    vacuum_energy = reference_vacuum["energy"]
    mm_component_tolerance = 1.0e-10
    mm_components = {}
    for name in (
        "bond",
        "angle",
        "dihedral",
        "vdw",
        "elec",
        "vdw_14",
        "elec_14",
    ):
        difference = solution_energy[name] - vacuum_energy[name]
        mm_components[name] = {
            "vacuum_kcal_mol": vacuum_energy[name],
            "solution_kcal_mol": solution_energy[name],
            "difference_kcal_mol": difference,
        }
    maximum_mm_component_difference = max(
        abs(component["difference_kcal_mol"]) for component in mm_components.values()
    )
    maximum_error = max(
        sample["absolute_error_kcal_mol_angstrom"] for sample in samples
    )
    library = Path(sys.prefix) / "lib/libsander.so"
    payload = {
        "schema_version": 1,
        "artifact_type": "route1-amber-pb-inp2-exact-difference-force-probe",
        "compound_id": "mobley_1017962",
        "compound_name": "methyl hexanoate",
        "atom_count": len(positions),
        "route1_boundary": {
            "formula": (
                "E_solution(R) = E_MLIP,gas(R) + "
                "G_polar(R,q_fixed) + G_nonpolar(R)"
            ),
            "internal_exact_difference": (
                "DeltaG_candidate(R) = E_MM,PB(inp=2)(R) - E_MM,vacuum(R)"
            ),
            "gas_phase_mm_energy_in_reported_potential": False,
            "hydration_label_fit_or_residual_model": False,
        },
        "input": {
            "prmtop": str(prmtop),
            "prmtop_sha256": sha256_file(prmtop),
            "rst7": str(rst7_path),
            "rst7_sha256": sha256_file(rst7_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sander_python_distribution": _version("sander"),
            "parmed": _version("ParmEd"),
            "libsander": str(library) if library.is_file() else None,
            "libsander_sha256": sha256_file(library) if library.is_file() else None,
        },
        "candidate_options": {
            "vacuum": {
                "igb": 5,
                "ipb": 0,
                "inp": 0,
                "gbsa": 0,
                "intdiel": 1.0,
                "extdiel": 1.0,
                "interpretation": (
                    "Identical MM topology with zero dielectric contrast and "
                    "nonpolar disabled."
                ),
            },
            "solution": {"igb": 10, "ipb": 1, "inp": 2},
        },
        "reference_geometry": {
            "vacuum_total_kcal_mol": vacuum_energy["tot"],
            "solution_total_kcal_mol": solution_energy["tot"],
            "exact_difference_kcal_mol": (
                solution_energy["tot"] - vacuum_energy["tot"]
            ),
            "solution_components_kcal_mol": {
                "pb": solution_energy["pb"],
                "cavity": solution_energy["surf"],
                "dispersion": solution_energy["disp"],
            },
            "mm_component_cancellation": {
                "tolerance_kcal_mol": mm_component_tolerance,
                "components": mm_components,
                "maximum_absolute_difference_kcal_mol": (
                    maximum_mm_component_difference
                ),
                "all_components_cancel_within_tolerance": (
                    maximum_mm_component_difference <= mm_component_tolerance
                ),
            },
        },
        "force_consistency": {
            "force_definition": ("F_candidate = F_MM,PB(inp=2) - F_MM,vacuum"),
            "finite_difference_definition": (
                "-d[DeltaG_candidate]/dR by centered energy differences"
            ),
            "tolerance_kcal_mol_angstrom": (FORCE_TOLERANCE_KCAL_MOL_ANGSTROM),
            "steps_angstrom": list(steps),
            "sampled_components": [
                {"atom_index": atom_index, "axis": axis}
                for atom_index, axis in components
            ],
            "samples": samples,
            "maximum_absolute_error_kcal_mol_angstrom": maximum_error,
            "all_samples_within_tolerance": all(
                sample["within_tolerance"] for sample in samples
            ),
        },
        "product_eligibility": {
            "single_point_reference_energy": True,
            "force_consistent_runtime": False,
            "optimization": False,
            "relaxed_scan": False,
            "default_provider": False,
        },
        "official_capability_boundary": {
            "amber25_manual": "https://ambermd.org/doc12/Amber25.pdf",
            "gbnsr6": ("The manual states that GBNSR6 cannot yet be used in dynamics."),
            "pbsa_inp2": (
                "The documented PB force path is electrostatic; the standalone "
                "force example disables nonpolar interactions."
            ),
            "chagb_primary_reference": ("https://doi.org/10.1021/ct4010917"),
        },
        "interpretation": (
            "Subtracting identical MM gas terms removes MM from the reported "
            "Route-1 potential algebraically, but it does not make the PBSA "
            "inp=2 energy and returned force a conservative pair. This candidate "
            "must remain a single-point reference and must not be wired into "
            "OPT or relaxed SCAN."
        ),
    }
    write_json_atomic(args.output, payload)
    print(f"Wrote Amber PB force probe to {Path(args.output).resolve()}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", default=str(DEFAULT_INPUT_DIR / "obc2.prmtop"))
    parser.add_argument("--rst7", default=str(DEFAULT_INPUT_DIR / "obc2.rst7"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--components",
        default=",".join(f"{atom}:{axis}" for atom, axis in DEFAULT_COMPONENTS),
        help="Comma-separated atom:axis pairs.",
    )
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS_ANGSTROM),
        help="Comma-separated centered finite-difference steps in angstrom.",
    )
    return parser


def main() -> None:
    run_probe(build_parser().parse_args())


if __name__ == "__main__":
    main()
