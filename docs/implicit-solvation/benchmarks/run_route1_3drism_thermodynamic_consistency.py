#!/usr/bin/env python3
"""Run label-free consistency probes for the 3D-RISM research comparator."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, cast


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]
import route1_custom_solvent_asset_audit as asset_audit  # pyright: ignore[reportImplicitRelativeImport]
import run_route1_3drism_single_solvent_pilot as pilot  # pyright: ignore[reportImplicitRelativeImport]


PROTOCOL_ID = "maple-route1-3drism-thermodynamic-consistency-v1"
ROUTE_BOUNDARY = {
    "product_route1_formula": (
        "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)"
    ),
    "product_route1_formula_unchanged": True,
    "candidate_route_identity": "separate_3drism_research_comparator",
    "candidate_may_satisfy_product_route1_accuracy_gate": False,
    "experimental_solvation_labels_loaded": False,
    "experimental_residual_fit": False,
    "energy_calibration": False,
    "endpoint_selection_from_labels": False,
    "product_runtime_change": False,
}
ENERGY_KEYS = (
    "raw_excess_chemical_potential_kcal_mol",
    "gaussian_fluctuation_excess_chemical_potential_kcal_mol",
    "pc_plus_excess_chemical_potential_kcal_mol",
)
PMV_KEY = "partial_molar_volume_angstrom3"
FORBIDDEN_VALUE_FIELDS = frozenset(
    {
        "deltagsolv",
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_error_kcal_mol",
        "absolute_error_kcal_mol",
    }
)


def _forbid_value_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in FORBIDDEN_VALUE_FIELDS:
                raise ValueError(f"Forbidden value-bearing field: {key!r}.")
            _forbid_value_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _forbid_value_fields(nested)


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = core.load_json(path)
    if not isinstance(protocol, dict):
        raise ValueError("3D-RISM consistency protocol must be a JSON object.")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "research_comparator_only_no_accuracy_claim"
    ):
        raise ValueError("Unexpected 3D-RISM consistency protocol.")
    if protocol.get("route_boundary") != ROUTE_BOUNDARY:
        raise ValueError("3D-RISM consistency protocol changed the route boundary.")

    bound = protocol.get("bound_pilot")
    if not isinstance(bound, dict):
        raise ValueError("3D-RISM consistency protocol lacks a bound pilot.")
    for field in ("protocol", "artifact"):
        relative = pilot._safe_relative_path(bound.get(field), f"bound_pilot.{field}")
        if len(relative.parts) != 1:
            raise ValueError(f"bound_pilot.{field} must name one benchmark file.")
    for field in (
        "protocol_file_sha256",
        "artifact_file_sha256",
        "artifact_content_sha256",
    ):
        if not pilot._is_digest(bound.get(field)):
            raise ValueError(f"bound_pilot.{field} must be a SHA256 digest.")
    if (
        bound.get("required_admission_status")
        != "one_nonwater_solver_pilot_numerically_qualified_not_accuracy_validated"
    ):
        raise ValueError("Bound pilot admission status is not frozen.")

    reference = protocol.get("thermodynamic_reference")
    if reference != {
        "target_dataset": "MNSol-v2012",
        "target_temperature_kelvin": 298.0,
        "asset_temperature_kelvin": 298.15,
        "target_standard_state": (
            "Ben-Naim ideal gas 1 mol/L to ideal solution 1 mol/L"
        ),
        "neutral_excess_chemical_potential_standard_state_shift_kcal_mol": 0.0,
        "rt_ln_24_46_is_pressure_standard_state_conversion_not_applied": True,
        "pressure_correction_is_not_standard_state_conversion": True,
        "ions_require_an_additional_extrathermodynamic_convention": True,
        "fixed_geometry_is_not_an_absolute_solvation_free_energy_ensemble": True,
        "accuracy_temperature_eligible": False,
        "accuracy_ensemble_eligible": False,
    }:
        raise ValueError("Thermodynamic reference boundary is not frozen.")

    profile = protocol.get("numerical_profile")
    if profile != {
        "closure_3d": "kh",
        "grid_spacing_angstrom": 0.3,
        "buffer_angstrom": 14.0,
        "tolerance": 1e-05,
        "maximum_steps": 10000,
        "centering": 1,
        "raw_kh_is_a_probe_not_a_selected_accuracy_endpoint": True,
        "gf_pc_plus_and_partial_molar_volume_are_diagnostics": True,
    }:
        raise ValueError("3D-RISM consistency numerical profile is not frozen.")

    probes = protocol.get("probes")
    if not isinstance(probes, dict) or set(probes) != {
        "translation",
        "rotation",
        "bonded_parameter_independence",
        "zero_interaction_limit",
    }:
        raise ValueError("3D-RISM consistency probes are incomplete.")
    translation = probes["translation"]
    rotation = probes["rotation"]
    bonded = probes["bonded_parameter_independence"]
    ghost = probes["zero_interaction_limit"]
    if not all(isinstance(probe, dict) for probe in probes.values()):
        raise ValueError("Each 3D-RISM consistency probe must be an object.")
    if (
        translation.get("vector_angstrom") != [37.0, -19.0, 11.0]
        or _finite(
            translation.get("maximum_energy_difference_kcal_mol"),
            "translation energy threshold",
        )
        != 1e-08
        or _finite(
            translation.get("maximum_partial_molar_volume_difference_angstrom3"),
            "translation PMV threshold",
        )
        != 1e-06
    ):
        raise ValueError("Translation probe is not frozen.")
    if (
        rotation.get("euler_degrees_xyz") != [37.0, 53.0, 71.0]
        or rotation.get("rotation_about_centroid") is not True
        or _finite(
            rotation.get("maximum_energy_difference_kcal_mol"),
            "rotation energy threshold",
        )
        != 0.01
        or _finite(
            rotation.get("maximum_partial_molar_volume_difference_angstrom3"),
            "rotation PMV threshold",
        )
        != 0.5
    ):
        raise ValueError("Rotation probe is not frozen.")
    if (
        bonded.get("scaled_prmtop_flags")
        != [
            "BOND_FORCE_CONSTANT",
            "ANGLE_FORCE_CONSTANT",
            "DIHEDRAL_FORCE_CONSTANT",
        ]
        or bonded.get("scale_factor") != 1.5
        or bonded.get("required_unchanged_prmtop_flags")
        != [
            "CHARGE",
            "ATOM_TYPE_INDEX",
            "LENNARD_JONES_ACOEF",
            "LENNARD_JONES_BCOEF",
        ]
        or _finite(
            bonded.get("maximum_energy_difference_kcal_mol"),
            "bonded energy threshold",
        )
        != 1e-08
        or _finite(
            bonded.get("maximum_partial_molar_volume_difference_angstrom3"),
            "bonded PMV threshold",
        )
        != 1e-06
    ):
        raise ValueError("Bonded-independence probe is not frozen.")
    if (
        ghost.get("zeroed_prmtop_flags")
        != ["CHARGE", "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF"]
        or _finite(
            ghost.get("maximum_absolute_raw_kh_kcal_mol"),
            "ghost raw threshold",
        )
        != 0.005
        or _finite(
            ghost.get("maximum_absolute_gf_kcal_mol"),
            "ghost GF threshold",
        )
        != 0.005
        or ghost.get("pc_plus_is_not_required_to_vanish") is not True
        or ghost.get("partial_molar_volume_is_only_required_to_be_finite")
        is not True
    ):
        raise ValueError("Zero-interaction probe is not frozen.")

    admission = protocol.get("admission")
    if not isinstance(admission, dict) or (
        admission.get("all_numerical_probes_must_pass") is not True
        or admission.get("experimental_labels_must_remain_closed") is not True
        or admission.get("accuracy_admission_must_remain_false") is not True
        or admission.get("product_route1_admission_must_remain_false") is not True
        or admission.get("result_if_passed")
        != "research_comparator_consistency_qualified_accuracy_blocked"
        or admission.get("result_if_failed")
        != "research_comparator_consistency_failed_accuracy_blocked"
        or admission.get("next_gate")
        != "return_to_fixed_charge_chagb_pbsa_ranking_mainline"
    ):
        raise ValueError("3D-RISM consistency admission gate is not frozen.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _load_bound_pilot(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    bound = protocol["bound_pilot"]
    pilot_protocol_path = SCRIPT_DIR / bound["protocol"]
    pilot_artifact_path = SCRIPT_DIR / bound["artifact"]
    if (
        core.sha256_file(pilot_protocol_path) != bound["protocol_file_sha256"]
        or core.sha256_file(pilot_artifact_path) != bound["artifact_file_sha256"]
    ):
        raise ValueError("Bound 3D-RISM pilot files changed.")
    pilot_protocol, _ = pilot.load_protocol(pilot_protocol_path)
    artifact = core.load_json(pilot_artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Bound 3D-RISM pilot artifact must be an object.")
    if (
        artifact.get("content_sha256") != bound["artifact_content_sha256"]
        or artifact.get("content_sha256") != core.artifact_content_sha256(artifact)
        or artifact.get("admission", {}).get("status")
        != bound["required_admission_status"]
    ):
        raise ValueError("Bound 3D-RISM pilot artifact is invalid.")
    if (
        pilot_protocol["solvent_asset"]["manifest"]["solvent"][
            "temperature_kelvin"
        ]
        != protocol["thermodynamic_reference"]["asset_temperature_kelvin"]
    ):
        raise ValueError("Bound ethanol asset temperature changed.")

    reference_case = next(
        case
        for case in pilot_protocol["cases"]
        if case["id"] == pilot_protocol["admission"]["reference_case"]
    )
    if (
        reference_case["grid_spacing_angstrom"]
        != protocol["numerical_profile"]["grid_spacing_angstrom"]
        or reference_case["buffer_angstrom"]
        != protocol["numerical_profile"]["buffer_angstrom"]
        or reference_case["tolerance"]
        != protocol["numerical_profile"]["tolerance"]
    ):
        raise ValueError("Consistency profile no longer matches the bound reference.")
    grid_case = next(case for case in pilot_protocol["cases"] if case["id"] == "grid_sensitivity")
    if (
        protocol["probes"]["rotation"]["maximum_energy_difference_kcal_mol"]
        != 0.5 * grid_case["maximum_primary_difference_kcal_mol"]
    ):
        raise ValueError("Rotation energy threshold derivation changed.")
    reference_solver = next(
        case["solver"] for case in artifact["cases"] if case["id"] == "reference"
    )
    maximum_pmv_difference = max(
        abs(case["solver"][PMV_KEY] - reference_solver[PMV_KEY])
        for case in artifact["cases"]
        if case["id"] in {
            "grid_sensitivity",
            "buffer_sensitivity",
            "tolerance_sensitivity",
        }
    )
    derived_pmv_threshold = math.ceil(20.0 * maximum_pmv_difference) / 10.0
    if (
        protocol["probes"]["rotation"][
            "maximum_partial_molar_volume_difference_angstrom3"
        ]
        != derived_pmv_threshold
    ):
        raise ValueError("Rotation PMV threshold derivation changed.")
    return pilot_protocol, artifact, pilot_protocol_path, pilot_artifact_path


def _resolve_executables(amber_root_path: str | Path) -> tuple[Path, dict[str, Path]]:
    root, executables = pilot._resolve_amber_root(amber_root_path)
    for name in ("ambpdb", "parmed"):
        executable = root / "bin" / name
        if not executable.is_file():
            raise FileNotFoundError(f"AmberTools root is missing {name}.")
        executables[name] = executable
    return root, executables


def _read_restart(path: Path) -> tuple[str, list[tuple[float, float, float]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError("Amber restart is truncated.")
    try:
        atom_count = int(lines[1].split()[0])
        values = [
            float(token.replace("D", "E"))
            for token in " ".join(lines[2:]).split()
        ]
    except (IndexError, ValueError) as exc:
        raise ValueError("Amber restart is malformed.") from exc
    if atom_count <= 0 or len(values) != 3 * atom_count:
        raise ValueError("Amber restart must contain coordinates only.")
    coordinates = [
        (values[index], values[index + 1], values[index + 2])
        for index in range(0, len(values), 3)
    ]
    return lines[0], coordinates


def _write_restart(
    path: Path,
    title: str,
    coordinates: list[tuple[float, float, float]],
) -> None:
    flat = [value for coordinate in coordinates for value in coordinate]
    lines = [title, f"{len(coordinates):6d}"]
    for start in range(0, len(flat), 6):
        lines.append("".join(f"{value:12.7f}" for value in flat[start : start + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _matmul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return cast(
        tuple[tuple[float, float, float], ...],
        tuple(
            tuple(
                sum(left[row][inner] * right[inner][column] for inner in range(3))
                for column in range(3)
            )
            for row in range(3)
        ),
    )


def _rotation_matrix(
    euler_degrees_xyz: list[float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z = (math.radians(angle) for angle in euler_degrees_xyz)
    rx = ((1.0, 0.0, 0.0), (0.0, math.cos(x), -math.sin(x)), (0.0, math.sin(x), math.cos(x)))
    ry = ((math.cos(y), 0.0, math.sin(y)), (0.0, 1.0, 0.0), (-math.sin(y), 0.0, math.cos(y)))
    rz = ((math.cos(z), -math.sin(z), 0.0), (math.sin(z), math.cos(z), 0.0), (0.0, 0.0, 1.0))
    return _matmul(rz, _matmul(ry, rx))


def _transform_coordinates(
    coordinates: list[tuple[float, float, float]],
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[tuple[float, float, float], ...] | None = None,
) -> list[tuple[float, float, float]]:
    centroid = tuple(
        sum(coordinate[axis] for coordinate in coordinates) / len(coordinates)
        for axis in range(3)
    )
    matrix = rotation or ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    transformed: list[tuple[float, float, float]] = []
    for coordinate in coordinates:
        centered = tuple(coordinate[axis] - centroid[axis] for axis in range(3))
        rotated = tuple(
            sum(matrix[row][column] * centered[column] for column in range(3))
            for row in range(3)
        )
        transformed.append(
            cast(
                tuple[float, float, float],
                tuple(
                    rotated[axis] + centroid[axis] + translation[axis]
                    for axis in range(3)
                ),
            )
        )
    return transformed


def _maximum_pair_distance_change(
    first: list[tuple[float, float, float]],
    second: list[tuple[float, float, float]],
) -> float:
    if len(first) != len(second):
        raise ValueError("Coordinate sets differ in atom count.")
    maximum = 0.0
    for left in range(len(first)):
        for right in range(left):
            first_distance = math.dist(first[left], first[right])
            second_distance = math.dist(second[left], second[right])
            maximum = max(maximum, abs(first_distance - second_distance))
    return maximum


def _write_pdb(
    *,
    executable: Path,
    prmtop: Path,
    restart: Path,
    output: Path,
    cwd: Path,
    timeout_seconds: float,
) -> None:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable), "-p", prmtop.name],
            cwd=cwd,
            input=restart.read_bytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("ambpdb exceeded its timeout.") from exc
    if completed.returncode != 0 or not completed.stdout:
        raise RuntimeError(
            "ambpdb failed; "
            f"elapsed={time.perf_counter() - started:.3f}; "
            f"stderr_tail={completed.stderr[-2000:]!r}"
        )
    output.write_bytes(completed.stdout)


def _prmtop_sections(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    starts = [
        (index, line.split(maxsplit=1)[1].strip())
        for index, line in enumerate(lines)
        if line.startswith("%FLAG ")
    ]
    if not starts:
        raise ValueError("Amber prmtop contains no %FLAG sections.")
    sections: dict[str, str] = {}
    for position, (start, name) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        if name in sections:
            raise ValueError(f"Amber prmtop repeats section {name!r}.")
        sections[name] = "".join(lines[start + 1 : end])
    return sections


def _section_values(section: str) -> list[float]:
    lines = section.splitlines()
    if not lines or not lines[0].startswith("%FORMAT"):
        raise ValueError("Amber prmtop section lacks a %FORMAT record.")
    return [
        float(token.replace("D", "E"))
        for token in re.findall(pilot.FLOAT_PATTERN, "\n".join(lines[1:]))
    ]


def _section_evidence(
    original_path: Path,
    modified_path: Path,
    *,
    scaled_flags: list[str] | None = None,
    scale_factor: float | None = None,
    unchanged_flags: list[str] | None = None,
    zeroed_flags: list[str] | None = None,
) -> dict[str, Any]:
    original = _prmtop_sections(original_path)
    modified = _prmtop_sections(modified_path)
    evidence: dict[str, Any] = {}
    all_passed = True
    for flag in scaled_flags or []:
        before = _section_values(original[flag])
        after = _section_values(modified[flag])
        passed = (
            bool(before)
            and len(before) == len(after)
            and any(abs(value) > 0.0 for value in before)
            and scale_factor is not None
            and all(
                math.isclose(
                    after_value,
                    scale_factor * before_value,
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )
                for before_value, after_value in zip(before, after, strict=True)
            )
        )
        evidence[flag] = {
            "operation": "scaled",
            "original_section_sha256": core.sha256_bytes(original[flag].encode()),
            "modified_section_sha256": core.sha256_bytes(modified[flag].encode()),
            "passed": passed,
        }
        all_passed = all_passed and passed
    for flag in unchanged_flags or []:
        before = _section_values(original[flag])
        after = _section_values(modified[flag])
        passed = len(before) == len(after) and all(
            first == second for first, second in zip(before, after, strict=True)
        )
        evidence[flag] = {
            "operation": "required_unchanged",
            "original_section_sha256": core.sha256_bytes(original[flag].encode()),
            "modified_section_sha256": core.sha256_bytes(modified[flag].encode()),
            "passed": passed,
        }
        all_passed = all_passed and passed
    for flag in zeroed_flags or []:
        values = _section_values(modified[flag])
        passed = bool(values) and all(abs(value) <= 1e-12 for value in values)
        evidence[flag] = {
            "operation": "zeroed",
            "original_section_sha256": core.sha256_bytes(original[flag].encode()),
            "modified_section_sha256": core.sha256_bytes(modified[flag].encode()),
            "passed": passed,
        }
        all_passed = all_passed and passed
    return {"flags": evidence, "passed": all_passed}


def _run_parmed(
    *,
    executable: Path,
    work: Path,
    source_prmtop: Path,
    output_name: str,
    commands: list[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    script = work / f"{output_name}.parmed.in"
    output = work / f"{output_name}.prmtop"
    script.write_text(
        "\n".join([*commands, f"outparm {output.name}", "quit"]) + "\n",
        encoding="ascii",
    )
    stdout, stderr, elapsed = pilot._run(
        [
            str(executable),
            "-n",
            "-O",
            "-p",
            source_prmtop.name,
            "-i",
            script.name,
        ],
        cwd=work,
        timeout_seconds=timeout_seconds,
        label=f"parmed:{output_name}",
    )
    if not output.is_file():
        raise RuntimeError(f"ParmEd did not create {output.name}.")
    return {
        "path": output,
        "script_sha256": core.sha256_file(script),
        "stdout_sha256": core.sha256_bytes(stdout.encode()),
        "stderr_empty": not stderr.strip(),
        "elapsed_seconds": elapsed,
    }


def _run_rism_case(
    *,
    identifier: str,
    executables: dict[str, Path],
    work: Path,
    prmtop: Path,
    restart: Path,
    pdb: Path,
    xvv: Path,
    profile: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    grid = float(profile["grid_spacing_angstrom"])
    command = [
        str(executables["rism3d.snglpnt"]),
        "--pdb",
        pdb.name,
        "--prmtop",
        prmtop.name,
        "--rst",
        restart.name,
        "--xvv",
        str(xvv),
        "--closure",
        str(profile["closure_3d"]),
        "--buffer",
        str(profile["buffer_angstrom"]),
        "--grdspc",
        f"{grid},{grid},{grid}",
        "--tolerance",
        str(profile["tolerance"]),
        "--maxstep",
        str(profile["maximum_steps"]),
        "--centering",
        str(profile["centering"]),
        "--verbose",
        "1",
        "--gf",
        "--pc+",
    ]
    try:
        stdout, stderr, elapsed = pilot._run(
            command,
            cwd=work,
            timeout_seconds=timeout_seconds,
            label=f"rism3d:{identifier}",
        )
        parsed = pilot.parse_rism_output(stdout)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        return {
            "id": identifier,
            "completed": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "input": {
                "prmtop_sha256": core.sha256_file(prmtop),
                "restart_sha256": core.sha256_file(restart),
                "pdb_sha256": core.sha256_file(pdb),
            },
        }
    return {
        "id": identifier,
        "completed": True,
        "solver": parsed,
        "elapsed_seconds": elapsed,
        "stderr_empty": not stderr.strip(),
        "input": {
            "prmtop_sha256": core.sha256_file(prmtop),
            "restart_sha256": core.sha256_file(restart),
            "pdb_sha256": core.sha256_file(pdb),
        },
    }


def _scalar_differences(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, float], float]:
    energy = {
        key: abs(float(candidate[key]) - float(reference[key]))
        for key in ENERGY_KEYS
    }
    pmv = abs(float(candidate[PMV_KEY]) - float(reference[PMV_KEY]))
    return energy, pmv


def _comparison_gate(
    reference_case: dict[str, Any],
    candidate_case: dict[str, Any],
    *,
    maximum_energy_difference: float,
    maximum_pmv_difference: float,
) -> dict[str, Any]:
    if not reference_case.get("completed") or not candidate_case.get("completed"):
        return {"passed": False, "reason": "required_case_did_not_complete"}
    energy, pmv = _scalar_differences(
        reference_case["solver"], candidate_case["solver"]
    )
    passed = (
        max(energy.values()) <= maximum_energy_difference
        and pmv <= maximum_pmv_difference
    )
    return {
        "energy_differences_kcal_mol": energy,
        "maximum_energy_difference_kcal_mol": max(energy.values()),
        "partial_molar_volume_difference_angstrom3": pmv,
        "energy_threshold_kcal_mol": maximum_energy_difference,
        "partial_molar_volume_threshold_angstrom3": maximum_pmv_difference,
        "passed": passed,
    }


def run_qualification(
    *,
    protocol_path: Path,
    amber_root_path: Path,
    solvent_data_root_path: Path | None,
    output_path: Path,
    timeout_seconds: float,
    recorded_date: str,
) -> dict[str, Any]:
    protocol, protocol_fingerprint = _load_protocol(protocol_path)
    (
        base_protocol,
        base_artifact,
        bound_protocol_path,
        bound_artifact_path,
    ) = _load_bound_pilot(protocol)
    amber_root, executables = _resolve_executables(amber_root_path)

    source = base_protocol["solute"]
    source_mol2 = pilot._safe_benchmark_path(source["mol2"], "solute.mol2")
    if (
        not source_mol2.is_file()
        or core.sha256_file(source_mol2) != source["mol2_sha256"]
    ):
        raise ValueError("Pinned consistency-probe solute changed or is missing.")

    solvent = base_protocol["solvent_asset"]
    asset_contract_path = pilot._safe_benchmark_path(
        solvent["asset_contract"], "solvent_asset.asset_contract"
    )
    if (
        not asset_contract_path.is_file()
        or core.sha256_file(asset_contract_path)
        != solvent["asset_contract_sha256"]
    ):
        raise ValueError("Pinned consistency-probe asset contract changed.")
    asset_contract, asset_contract_fingerprint = asset_audit.load_contract(
        asset_contract_path
    )
    if solvent_data_root_path is None:
        solvent_data = (
            amber_root
            / "dat"
            / pilot._safe_relative_path(
                solvent["amber_data_subdirectory"],
                "solvent_asset.amber_data_subdirectory",
            )
        ).resolve()
    else:
        solvent_data = solvent_data_root_path.resolve()
    if not solvent_data.is_dir():
        raise FileNotFoundError("Pinned consistency-probe solvent data is missing.")

    with tempfile.TemporaryDirectory(
        prefix="maple-route1-3drism-consistency-"
    ) as temporary:
        work = Path(temporary)
        asset_work = work / "solvent"
        asset_work.mkdir()
        for entry in solvent["files"].values():
            source_asset = solvent_data / entry["name"]
            if (
                not source_asset.is_file()
                or core.sha256_file(source_asset) != entry["sha256"]
            ):
                raise ValueError(
                    f"Pinned consistency-probe asset changed: {entry['name']}."
                )
            shutil.copyfile(source_asset, asset_work / entry["name"])
        manifest_path = asset_work / "manifest.json"
        core.write_json_atomic(manifest_path, copy.deepcopy(solvent["manifest"]))
        solvent_asset_audit = asset_audit.audit_manifest(
            asset_contract, asset_contract_fingerprint, manifest_path
        )
        if (
            solvent_asset_audit["conclusion"]["status"]
            != base_protocol["admission"]["asset_audit_status_required"]
        ):
            raise ValueError("Solvent asset failed the frozen input audit.")

        topology_hashes = pilot._prepare_topology(
            work=work,
            source_mol2=source_mol2,
            executables=executables,
            timeout_seconds=timeout_seconds,
        )
        reference_prmtop = work / "molecule.prmtop"
        reference_restart = work / "molecule.inpcrd"
        reference_pdb = work / "molecule.pdb"
        title, coordinates = _read_restart(reference_restart)

        translation_vector = cast(
            tuple[float, float, float],
            tuple(protocol["probes"]["translation"]["vector_angstrom"]),
        )
        translated_coordinates = _transform_coordinates(
            coordinates, translation=translation_vector
        )
        translated_restart = work / "translated.inpcrd"
        translated_pdb = work / "translated.pdb"
        _write_restart(translated_restart, title, translated_coordinates)
        _write_pdb(
            executable=executables["ambpdb"],
            prmtop=reference_prmtop,
            restart=translated_restart,
            output=translated_pdb,
            cwd=work,
            timeout_seconds=timeout_seconds,
        )

        rotation = _rotation_matrix(
            protocol["probes"]["rotation"]["euler_degrees_xyz"]
        )
        rotated_coordinates = _transform_coordinates(
            coordinates, rotation=rotation
        )
        rotated_restart = work / "rotated.inpcrd"
        rotated_pdb = work / "rotated.pdb"
        _write_restart(rotated_restart, title, rotated_coordinates)
        _write_pdb(
            executable=executables["ambpdb"],
            prmtop=reference_prmtop,
            restart=rotated_restart,
            output=rotated_pdb,
            cwd=work,
            timeout_seconds=timeout_seconds,
        )
        coordinate_evidence = {
            "translation_maximum_pair_distance_change_angstrom": (
                _maximum_pair_distance_change(coordinates, translated_coordinates)
            ),
            "rotation_maximum_pair_distance_change_angstrom": (
                _maximum_pair_distance_change(coordinates, rotated_coordinates)
            ),
        }

        bonded_probe = protocol["probes"]["bonded_parameter_independence"]
        bonded_result = _run_parmed(
            executable=executables["parmed"],
            work=work,
            source_prmtop=reference_prmtop,
            output_name="bonded_scaled",
            commands=[
                f"scale {flag} {bonded_probe['scale_factor']}"
                for flag in bonded_probe["scaled_prmtop_flags"]
            ],
            timeout_seconds=timeout_seconds,
        )
        bonded_prmtop = cast(Path, bonded_result.pop("path"))
        bonded_section_evidence = _section_evidence(
            reference_prmtop,
            bonded_prmtop,
            scaled_flags=bonded_probe["scaled_prmtop_flags"],
            scale_factor=float(bonded_probe["scale_factor"]),
            unchanged_flags=bonded_probe["required_unchanged_prmtop_flags"],
        )

        ghost_probe = protocol["probes"]["zero_interaction_limit"]
        ghost_result = _run_parmed(
            executable=executables["parmed"],
            work=work,
            source_prmtop=reference_prmtop,
            output_name="zero_interaction",
            commands=[
                "change CHARGE @* 0 quiet",
                "scale LENNARD_JONES_ACOEF 0",
                "scale LENNARD_JONES_BCOEF 0",
            ],
            timeout_seconds=timeout_seconds,
        )
        ghost_prmtop = cast(Path, ghost_result.pop("path"))
        ghost_section_evidence = _section_evidence(
            reference_prmtop,
            ghost_prmtop,
            zeroed_flags=ghost_probe["zeroed_prmtop_flags"],
            unchanged_flags=["ATOM_TYPE_INDEX"],
        )

        xvv = asset_work / solvent["files"]["xvv"]["name"]
        profile = protocol["numerical_profile"]
        cases = {
            "reference": _run_rism_case(
                identifier="reference",
                executables=executables,
                work=work,
                prmtop=reference_prmtop,
                restart=reference_restart,
                pdb=reference_pdb,
                xvv=xvv,
                profile=profile,
                timeout_seconds=timeout_seconds,
            ),
            "translated": _run_rism_case(
                identifier="translated",
                executables=executables,
                work=work,
                prmtop=reference_prmtop,
                restart=translated_restart,
                pdb=translated_pdb,
                xvv=xvv,
                profile=profile,
                timeout_seconds=timeout_seconds,
            ),
            "rotated": _run_rism_case(
                identifier="rotated",
                executables=executables,
                work=work,
                prmtop=reference_prmtop,
                restart=rotated_restart,
                pdb=rotated_pdb,
                xvv=xvv,
                profile=profile,
                timeout_seconds=timeout_seconds,
            ),
            "bonded_scaled": _run_rism_case(
                identifier="bonded_scaled",
                executables=executables,
                work=work,
                prmtop=bonded_prmtop,
                restart=reference_restart,
                pdb=reference_pdb,
                xvv=xvv,
                profile=profile,
                timeout_seconds=timeout_seconds,
            ),
            "zero_interaction": _run_rism_case(
                identifier="zero_interaction",
                executables=executables,
                work=work,
                prmtop=ghost_prmtop,
                restart=reference_restart,
                pdb=reference_pdb,
                xvv=xvv,
                profile=profile,
                timeout_seconds=timeout_seconds,
            ),
        }

        reference_case = cases["reference"]
        translation_probe = protocol["probes"]["translation"]
        translation_gate = _comparison_gate(
            reference_case,
            cases["translated"],
            maximum_energy_difference=float(
                translation_probe["maximum_energy_difference_kcal_mol"]
            ),
            maximum_pmv_difference=float(
                translation_probe[
                    "maximum_partial_molar_volume_difference_angstrom3"
                ]
            ),
        )
        translation_gate["coordinate_identity_passed"] = (
            coordinate_evidence[
                "translation_maximum_pair_distance_change_angstrom"
            ]
            <= 1e-10
        )
        translation_gate["passed"] = bool(
            translation_gate["passed"]
            and translation_gate["coordinate_identity_passed"]
        )

        rotation_probe = protocol["probes"]["rotation"]
        rotation_gate = _comparison_gate(
            reference_case,
            cases["rotated"],
            maximum_energy_difference=float(
                rotation_probe["maximum_energy_difference_kcal_mol"]
            ),
            maximum_pmv_difference=float(
                rotation_probe[
                    "maximum_partial_molar_volume_difference_angstrom3"
                ]
            ),
        )
        rotation_gate["coordinate_identity_passed"] = (
            coordinate_evidence["rotation_maximum_pair_distance_change_angstrom"]
            <= 1e-10
        )
        rotation_gate["passed"] = bool(
            rotation_gate["passed"] and rotation_gate["coordinate_identity_passed"]
        )

        bonded_gate = _comparison_gate(
            reference_case,
            cases["bonded_scaled"],
            maximum_energy_difference=float(
                bonded_probe["maximum_energy_difference_kcal_mol"]
            ),
            maximum_pmv_difference=float(
                bonded_probe[
                    "maximum_partial_molar_volume_difference_angstrom3"
                ]
            ),
        )
        bonded_gate["topology_perturbation_passed"] = bonded_section_evidence[
            "passed"
        ]
        bonded_gate["passed"] = bool(
            bonded_gate["passed"]
            and bonded_gate["topology_perturbation_passed"]
        )

        ghost_case = cases["zero_interaction"]
        if ghost_case.get("completed"):
            ghost_solver = ghost_case["solver"]
            ghost_gate = {
                "absolute_raw_kh_kcal_mol": abs(
                    float(
                        ghost_solver[
                            "raw_excess_chemical_potential_kcal_mol"
                        ]
                    )
                ),
                "absolute_gf_kcal_mol": abs(
                    float(
                        ghost_solver[
                            "gaussian_fluctuation_excess_chemical_potential_kcal_mol"
                        ]
                    )
                ),
                "pc_plus_finite": math.isfinite(
                    float(
                        ghost_solver[
                            "pc_plus_excess_chemical_potential_kcal_mol"
                        ]
                    )
                ),
                "partial_molar_volume_finite": math.isfinite(
                    float(ghost_solver[PMV_KEY])
                ),
                "topology_perturbation_passed": ghost_section_evidence["passed"],
            }
            ghost_gate["passed"] = bool(
                ghost_gate["absolute_raw_kh_kcal_mol"]
                <= ghost_probe["maximum_absolute_raw_kh_kcal_mol"]
                and ghost_gate["absolute_gf_kcal_mol"]
                <= ghost_probe["maximum_absolute_gf_kcal_mol"]
                and ghost_gate["pc_plus_finite"]
                and ghost_gate["partial_molar_volume_finite"]
                and ghost_gate["topology_perturbation_passed"]
            )
        else:
            ghost_gate = {
                "passed": False,
                "reason": "zero_interaction_case_did_not_complete",
                "topology_perturbation_passed": ghost_section_evidence["passed"],
            }

        gates = {
            "translation_invariance": translation_gate,
            "proper_rotation_invariance": rotation_gate,
            "bonded_parameter_independence": bonded_gate,
            "zero_interaction_limit": ghost_gate,
        }
        numerical_passed = all(gate["passed"] for gate in gates.values())
        status_key = "result_if_passed" if numerical_passed else "result_if_failed"
        artifact: dict[str, Any] = {
            "artifact_type": "route1-3drism-thermodynamic-consistency-v1",
            "recorded_date": recorded_date,
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": protocol_fingerprint,
            "claim_scope": protocol["claim_scope"],
            "route_boundary": protocol["route_boundary"],
            "bound_pilot": {
                **protocol["bound_pilot"],
                "protocol_observed_file_sha256": core.sha256_file(
                    bound_protocol_path
                ),
                "artifact_observed_file_sha256": core.sha256_file(
                    bound_artifact_path
                ),
                "artifact_observed_content_sha256": base_artifact[
                    "content_sha256"
                ],
            },
            "thermodynamic_reference": protocol["thermodynamic_reference"],
            "numerical_profile": protocol["numerical_profile"],
            "solute": {
                "compound_id": source["compound_id"],
                "mol2_sha256": source["mol2_sha256"],
                "topology_model": source["topology_model"],
                "generated_topology_hashes": topology_hashes,
            },
            "solvent_asset_audit": solvent_asset_audit,
            "coordinate_evidence": coordinate_evidence,
            "topology_perturbations": {
                "bonded_scaled": {
                    **bonded_result,
                    "prmtop_sha256": core.sha256_file(bonded_prmtop),
                    "section_evidence": bonded_section_evidence,
                },
                "zero_interaction": {
                    **ghost_result,
                    "prmtop_sha256": core.sha256_file(ghost_prmtop),
                    "section_evidence": ghost_section_evidence,
                },
            },
            "cases": cases,
            "gates": gates,
            "containment": {
                "experimental_solvation_labels_loaded": False,
                "experimental_values_emitted": False,
                "fit_or_tuning_performed": False,
                "accuracy_endpoint_selected": False,
                "accuracy_admission": False,
                "product_route1_admission": False,
            },
            "conclusion": {
                "numerical_consistency_passed": numerical_passed,
                "accuracy_admission": False,
                "product_route1_admission": False,
                "status": protocol["admission"][status_key],
                "next_gate": protocol["admission"]["next_gate"],
            },
            "executable_sha256": {
                name: core.sha256_file(path)
                for name, path in executables.items()
            },
            "prohibited_interpretations": protocol[
                "prohibited_interpretations"
            ],
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "protocol": str(protocol_path),
                    "amber_root": str(amber_root),
                    "solvent_data_root": str(solvent_data),
                    "output": str(output_path),
                    "timeout_seconds": timeout_seconds,
                    "recorded_date": recorded_date,
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
        _forbid_value_fields(artifact)
        artifact = core.seal_artifact(artifact)
        core.write_json_atomic(output_path, artifact)
        return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amber-root", type=Path, required=True)
    parser.add_argument("--solvent-data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--recorded-date", required=True)
    args = parser.parse_args(argv)
    run_qualification(
        protocol_path=args.protocol,
        amber_root_path=args.amber_root,
        solvent_data_root_path=args.solvent_data_root,
        output_path=args.output,
        timeout_seconds=pilot._positive_number(
            args.timeout_seconds, "timeout_seconds"
        ),
        recorded_date=args.recorded_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
