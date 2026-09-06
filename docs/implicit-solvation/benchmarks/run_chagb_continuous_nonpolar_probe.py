#!/usr/bin/env python3
"""Probe continuous PBSA cavity/dispersion energy and coordinate gradients.

This is an isolated same-functional numerical experiment.  It neither invokes
the CHA polar term nor promotes forces into MAPLE's runtime provider.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (
    canonical_json_bytes,
    seal_artifact,
    sha256_file,
    write_json_atomic,
)

from maple.function.calculator.extra_correction.implicit.sphere_union_dispersion import (
    dispersion_energy_and_gradient,
)
from maple.function.calculator.extra_correction.implicit.sphere_union_volume import (
    volume_and_gradient,
)

ARTIFACT_TYPE = "route1-chagb-continuous-nonpolar-probe"
PROFILE = (
    "experimental_same_functional_continuum_quadrature_not_legacy_discrete_endpoint"
)
CAVITY_PROBE_RADIUS = 1.3
CAVITY_SURFACE_TENSION = 0.0378
CAVITY_OFFSET = -0.5692
DISPERSION_PROBE_RADIUS = 0.557
OXYGEN_RMIN = 1.7683
OXYGEN_EPSILON = 0.1520
RHO_WATER = 0.03333 * 1.129
TWO_TO_NEGATIVE_ONE_SIXTH = 2.0 ** (-1.0 / 6.0)
SOURCE_ARCHIVE_SHA256 = (
    "5d46eef3c2bb7d5bf9e8c0c38add34406ea67e3f0e4097ac9d11d8a544538c9c"
)
SOURCE_FILES = {
    "pb_read.F90": "9458777de9b2cad57bbed82f01da7251373292acde96e805d13c8c87aab92ad4",
    "sa_driver.F90": "dfd7bf96496938d3bee900ae5f9a7c1e5875bb16eb86631c209cd0822951fbee",
    "np_force.F90": "91b3ce2d072e8c0f6eb99a9e3213793834cec49997ea7d4750081f9d81a9fd9f",
}


def _capability_boundary() -> dict[str, bool]:
    return {
        PROFILE: True,
        "nonpolar_energy": True,
        "nonpolar_coordinate_gradient": True,
        "full_chagb_energy": False,
        "full_chagb_force": False,
        "runtime_provider_promoted": False,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numeric_vector(name: str, value: object, count: int | None = None) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) for item in raw.flat):
        raise TypeError(f"{name} must contain numbers, not booleans.")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numbers.") from exc
    if result.ndim != 1 or (count is not None and result.shape != (count,)):
        expected = "a one-dimensional array" if count is None else f"shape ({count},)"
        raise ValueError(f"{name} must have {expected}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def validate_parameters(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("parameter input must be a JSON object.")
    required = {
        "positions_angstrom",
        "rmin_angstrom",
        "epsilon_kcal_mol",
        "prmtop_sha256",
        "parmed_version",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"parameter input is missing fields: {', '.join(missing)}")
    try:
        raw_positions = np.asarray(payload["positions_angstrom"], dtype=object)
    except ValueError:
        raw_positions = None
    if raw_positions is not None and any(
        isinstance(value, (bool, np.bool_)) for value in raw_positions.flat
    ):
        raise TypeError("positions_angstrom must contain numbers, not booleans.")
    try:
        positions = np.asarray(payload["positions_angstrom"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("positions_angstrom must contain numbers.") from exc
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0:
        raise ValueError("positions_angstrom must have nonempty shape (n, 3).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_angstrom must be finite.")
    rmin = _numeric_vector("rmin_angstrom", payload["rmin_angstrom"], len(positions))
    epsilon = _numeric_vector(
        "epsilon_kcal_mol", payload["epsilon_kcal_mol"], len(positions)
    )
    if np.any(rmin <= 0.0):
        raise ValueError(
            "rmin_angstrom must be positive; Amber zero-radius site skipping is "
            "outside this probe's validated mapping."
        )
    if np.any(epsilon < 0.0):
        raise ValueError("epsilon_kcal_mol must be nonnegative.")
    prmtop_sha256 = payload["prmtop_sha256"]
    if (
        not isinstance(prmtop_sha256, str)
        or len(prmtop_sha256) != 64
        or any(character not in "0123456789abcdef" for character in prmtop_sha256)
    ):
        raise ValueError("prmtop_sha256 must be a lowercase SHA256 digest.")
    parmed_version = payload["parmed_version"]
    if not isinstance(parmed_version, str) or not parmed_version.strip():
        raise ValueError("parmed_version must be a nonempty string.")
    return {
        "positions_angstrom": positions,
        "rmin_angstrom": rmin,
        "epsilon_kcal_mol": epsilon,
        "prmtop_sha256": prmtop_sha256,
        "parmed_version": parmed_version,
    }


def load_parameters(path: str | Path) -> dict[str, object]:
    source = Path(path)
    return validate_parameters(json.loads(source.read_text(encoding="utf-8")))


def evaluate_nonpolar(parameters: dict[str, object]) -> dict[str, object]:
    validated = validate_parameters(parameters)
    positions = np.asarray(validated["positions_angstrom"], dtype=float)
    rmin = np.asarray(validated["rmin_angstrom"], dtype=float)
    epsilon = np.asarray(validated["epsilon_kcal_mol"], dtype=float)

    cavity = volume_and_gradient(positions, rmin + CAVITY_PROBE_RADIUS)
    sigma = (rmin + OXYGEN_RMIN) * TWO_TO_NEGATIVE_ONE_SIXTH
    mixed_epsilon = np.sqrt(epsilon * OXYGEN_EPSILON)
    dispersion = dispersion_energy_and_gradient(
        positions,
        rmin + DISPERSION_PROBE_RADIUS,
        sigma,
        mixed_epsilon,
        RHO_WATER,
    )
    cavity_energy = CAVITY_SURFACE_TENSION * cavity.volume + CAVITY_OFFSET
    cavity_gradient = CAVITY_SURFACE_TENSION * cavity.gradient
    total_gradient = cavity_gradient + dispersion.gradient
    return {
        "components_kcal_mol": {
            "cavity": cavity_energy,
            "dispersion": dispersion.energy,
            "total_nonpolar": cavity_energy + dispersion.energy,
        },
        "component_coordinate_gradients_kcal_mol_angstrom": {
            "cavity": cavity_gradient,
            "dispersion": dispersion.gradient,
        },
        "coordinate_gradient_kcal_mol_angstrom": total_gradient,
        "forces_kcal_mol_angstrom": -total_gradient,
        "derived_site_parameters": {
            "cavity_radii_angstrom": rmin + CAVITY_PROBE_RADIUS,
            "dispersion_sas_radii_angstrom": rmin + DISPERSION_PROBE_RADIUS,
            "sigma_angstrom": sigma,
            "mixed_epsilon_kcal_mol": mixed_epsilon,
        },
        "quadrature": {
            "cavity": {
                "estimated_error": cavity.quadrature_error,
                **cavity.diagnostics,
            },
            "dispersion": {
                "estimated_error": dispersion.quadrature_error,
                **dispersion.diagnostics,
            },
        },
        "capability_boundary": _capability_boundary(),
    }


def _finite_difference(
    parameters: dict[str, object], coordinates: list[tuple[int, int]], step: float
) -> dict[str, object]:
    validated = validate_parameters(parameters)
    positions = np.asarray(validated["positions_angstrom"], dtype=float)
    reference = evaluate_nonpolar(validated)
    analytic = np.asarray(reference["coordinate_gradient_kcal_mol_angstrom"])
    records: list[dict[str, float | int]] = []
    for atom, axis in coordinates:
        if atom < 0 or atom >= len(positions) or axis not in (0, 1, 2):
            raise ValueError(f"finite-difference coordinate {(atom, axis)} is invalid.")
        plus = positions.copy()
        minus = positions.copy()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        plus_parameters = {**validated, "positions_angstrom": plus}
        minus_parameters = {**validated, "positions_angstrom": minus}
        plus_components = evaluate_nonpolar(plus_parameters)["components_kcal_mol"]
        minus_components = evaluate_nonpolar(minus_parameters)["components_kcal_mol"]
        if not isinstance(plus_components, dict) or not isinstance(
            minus_components, dict
        ):
            raise TypeError("nonpolar evaluation returned malformed components.")
        plus_energy = plus_components["total_nonpolar"]
        minus_energy = minus_components["total_nonpolar"]
        numerical = (float(plus_energy) - float(minus_energy)) / (2.0 * step)
        records.append(
            {
                "atom_index": atom,
                "axis": axis,
                "analytic_gradient": float(analytic[atom, axis]),
                "finite_difference_gradient": numerical,
                "absolute_difference": abs(numerical - float(analytic[atom, axis])),
            }
        )
    return {
        "requested": True,
        "step_angstrom": step,
        "coordinates": records,
        "maximum_absolute_difference": max(
            (record["absolute_difference"] for record in records), default=0.0
        ),
    }


def _physical_parameters() -> dict[str, object]:
    return {
        "pbsa_profile": "inp=2,use_rmin=1,use_sav=1,decompopt=2",
        "cavity_probe_radius_angstrom": CAVITY_PROBE_RADIUS,
        "cavity_surface_tension_kcal_mol_angstrom3": CAVITY_SURFACE_TENSION,
        "cavity_offset_kcal_mol": CAVITY_OFFSET,
        "dispersion_probe_radius_angstrom": DISPERSION_PROBE_RADIUS,
        "water_oxygen_rmin_angstrom": OXYGEN_RMIN,
        "water_oxygen_epsilon_kcal_mol": OXYGEN_EPSILON,
        "water_number_density_angstrom3": 0.03333,
        "effective_water_density_scale": 1.129,
        "effective_water_density_angstrom3": RHO_WATER,
        "sigma_formula": "(rmin + 1.7683) * 2**(-1/6)",
        "mixed_epsilon_formula": "sqrt(epsilon * 0.1520)",
    }


def _source_provenance() -> dict[str, object]:
    return {
        "distribution": "AmberTools 26 RC7",
        "archive_url": "https://ambermd.org/downloads/ambertools26_rc7.tar.bz2",
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "archive_sha256_verified": True,
        "source_files_sha256": SOURCE_FILES,
        "source_locations": {
            "defaults": "AmberTools/src/pbsa/pb_read.F90:726-730",
            "cavity_volume_radius": "AmberTools/src/pbsa/sa_driver.F90:1933",
            "cavity_energy": "AmberTools/src/pbsa/np_force.F90:92-93",
            "dispersion_surface": "AmberTools/src/pbsa/np_force.F90:75-78",
            "mixing_parameters": "AmberTools/src/pbsa/np_force.F90:107,120",
        },
        "note": (
            "The local extracted directory retains a legacy source-unverified name; "
            "the complete RC7 archive digest was subsequently verified."
        ),
    }


def _artifact(
    parameter_path: Path,
    parameters: dict[str, object],
    result: dict[str, object],
    finite_difference: dict[str, object],
    arguments: dict[str, object],
) -> dict[str, object]:
    volume_kernel = (
        REPOSITORY_ROOT
        / "maple/function/calculator/extra_correction/implicit/sphere_union_volume.py"
    )
    dispersion_kernel = (
        REPOSITORY_ROOT
        / "maple/function/calculator/extra_correction/implicit/sphere_union_dispersion.py"
    )
    safe_artifact = _json_safe(
        {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "status": "completed",
            "profile": PROFILE,
            "input": {
                "path": str(parameter_path),
                "sha256": sha256_file(parameter_path),
                "prmtop_sha256": parameters["prmtop_sha256"],
                "parmed_version": parameters["parmed_version"],
                "atom_count": len(np.asarray(parameters["positions_angstrom"])),
            },
            "physical_parameters": _physical_parameters(),
            "source_provenance": _source_provenance(),
            "implementation": {
                "script": SCRIPT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "volume_kernel": volume_kernel.relative_to(REPOSITORY_ROOT).as_posix(),
                "volume_kernel_sha256": sha256_file(volume_kernel),
                "dispersion_kernel": dispersion_kernel.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "dispersion_kernel_sha256": sha256_file(dispersion_kernel),
                "arguments": arguments,
            },
            "result": result,
            "finite_difference": finite_difference,
            "capability_boundary": result["capability_boundary"],
        }
    )
    if not isinstance(safe_artifact, dict):
        raise TypeError("internal artifact serialization failed.")
    return seal_artifact(safe_artifact)


def _failure_artifact(
    parameter_path: Path,
    arguments: dict[str, object],
    error: Exception,
) -> dict[str, object]:
    volume_kernel = (
        REPOSITORY_ROOT
        / "maple/function/calculator/extra_correction/implicit/sphere_union_volume.py"
    )
    dispersion_kernel = (
        REPOSITORY_ROOT
        / "maple/function/calculator/extra_correction/implicit/sphere_union_dispersion.py"
    )
    return seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "status": "failed",
            "profile": PROFILE,
            "input": {
                "path": str(parameter_path),
                "sha256": (
                    sha256_file(parameter_path) if parameter_path.is_file() else None
                ),
            },
            "physical_parameters": _physical_parameters(),
            "source_provenance": _source_provenance(),
            "implementation": {
                "script": SCRIPT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "volume_kernel_sha256": sha256_file(volume_kernel),
                "dispersion_kernel_sha256": sha256_file(dispersion_kernel),
                "arguments": arguments,
            },
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "capability_boundary": _capability_boundary(),
        }
    )


def _write_without_distinct_overwrite(path: Path, artifact: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json_bytes(existing) == canonical_json_bytes(artifact):
            return
        raise FileExistsError(
            f"refusing to overwrite distinct existing artifact: {path}"
        )
    write_json_atomic(path, artifact)


def _coordinate(value: str) -> tuple[int, int]:
    try:
        atom_text, axis_text = value.split(",", 1)
        return int(atom_text), int(axis_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("coordinate must be ATOM,AXIS") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fd-coordinate", action="append", type=_coordinate)
    group.add_argument("--all-3n-fd", action="store_true")
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parameter_path = args.parameters.resolve()
    output_path = args.output.resolve()
    arguments = {
        "parameters": str(parameter_path),
        "output": str(output_path),
        "fd_coordinate": args.fd_coordinate,
        "all_3n_fd": args.all_3n_fd,
        "fd_step": args.fd_step,
    }
    try:
        parameters = load_parameters(parameter_path)
        result = evaluate_nonpolar(parameters)
        if not math.isfinite(args.fd_step) or args.fd_step <= 0.0:
            raise ValueError("fd-step must be finite and positive.")
        atom_count = len(np.asarray(parameters["positions_angstrom"]))
        coordinates = args.fd_coordinate
        if args.all_3n_fd:
            coordinates = [
                (atom, axis) for atom in range(atom_count) for axis in range(3)
            ]
        finite_difference: dict[str, object] = (
            _finite_difference(parameters, coordinates, args.fd_step)
            if coordinates
            else {"requested": False}
        )
        artifact = _artifact(
            parameter_path,
            parameters,
            result,
            finite_difference,
            arguments,
        )
    except Exception as error:
        failure = _failure_artifact(parameter_path, arguments, error)
        _write_without_distinct_overwrite(output_path, failure)
        raise
    _write_without_distinct_overwrite(output_path, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
