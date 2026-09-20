#!/usr/bin/env python3
"""Run bounded physical/derivative validation for the ddX ddLPB reference path.

The optional APBS calculation is deliberately an independent, nonblocking raw
comparison.  This first harness does not turn a single finite-grid comparison
into a solver-agreement claim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (
    canonical_json_bytes,
    command_provenance,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

from maple.function.read.filereader.mol2_reader import MOL2Reader

BOHR_ANGSTROM = 0.529177210903
KJ_MOL_PER_HARTREE = 2625.4996394799
TEMPERATURE_K = 298.15
SOLUTE_EPSILON = 1.0
SOLVENT_EPSILON = 78.5
FD_STEPS_ANGSTROM = (1.0e-4, 5.0e-5)
SPHERE_ENERGY_ABS_TOLERANCE_HARTREE = 1.0e-12
SPHERE_FORCE_NORM_TOLERANCE_HARTREE_PER_ANGSTROM = 1.0e-10
FD_RMSE_TOLERANCE_HARTREE_PER_ANGSTROM = 5.0e-6
FD_MAX_TOLERANCE_HARTREE_PER_ANGSTROM = 2.0e-5
NET_FORCE_TOLERANCE_HARTREE_PER_ANGSTROM = 1.0e-8


def _json_safe(value: Any) -> Any:
    if not isinstance(value, type) and is_dataclass(value):
        return _json_safe(asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def charged_sphere_exact_energy_hartree(
    *,
    charge_e: float,
    radius_angstrom: float,
    solvent_kappa_inverse_angstrom: float,
    solute_epsilon: float = SOLUTE_EPSILON,
    solvent_epsilon: float = SOLVENT_EPSILON,
) -> float:
    """Return the analytic LPB reaction energy of one centered charged sphere."""
    radius_bohr = float(radius_angstrom) / BOHR_ANGSTROM
    kappa_inverse_bohr = float(solvent_kappa_inverse_angstrom) * BOHR_ANGSTROM
    return (
        1.0 / (float(solvent_epsilon) * (1.0 + kappa_inverse_bohr * radius_bohr))
        - 1.0 / float(solute_epsilon)
    ) * float(charge_e) ** 2 / (2.0 * radius_bohr)


def ionic_strength_molar(
    solvent_kappa_inverse_angstrom: float,
    *,
    temperature_kelvin: float = TEMPERATURE_K,
    solvent_epsilon: float = SOLVENT_EPSILON,
) -> float:
    """Map inverse-Angstrom kappa to symmetric 1:1 salt concentration."""
    from scipy import constants

    kappa_inverse_metre = float(solvent_kappa_inverse_angstrom) * 1.0e10
    return (
        kappa_inverse_metre**2
        * constants.epsilon_0
        * float(solvent_epsilon)
        * constants.k
        * float(temperature_kelvin)
        / (2.0 * constants.e**2 * constants.N_A * 1000.0)
    )


def kappa_mapping_record(
    solvent_kappa_inverse_angstrom: float,
    *,
    temperature_kelvin: float = TEMPERATURE_K,
    solvent_epsilon: float = SOLVENT_EPSILON,
) -> dict[str, Any]:
    """Record constants and the independent pyddx solvent_kappa round trip."""
    import pyddx
    import scipy
    from scipy import constants

    ionic_strength = ionic_strength_molar(
        solvent_kappa_inverse_angstrom,
        temperature_kelvin=temperature_kelvin,
        solvent_epsilon=solvent_epsilon,
    )
    pyddx_kappa_inverse_bohr = float(
        pyddx.solvent_kappa(
            [(1, ionic_strength), (-1, ionic_strength)],
            temperature_kelvin,
            solvent_epsilon,
        )
    )
    roundtrip = pyddx_kappa_inverse_bohr / BOHR_ANGSTROM
    requested = float(solvent_kappa_inverse_angstrom)
    return {
        "requested_kappa_inverse_angstrom": requested,
        "ionic_strength_molar": ionic_strength,
        "temperature_kelvin": float(temperature_kelvin),
        "solvent_epsilon": float(solvent_epsilon),
        "formula": (
            "I_molar=(kappa_A*1e10)^2*epsilon_0*epsilon_out*k_B*T/"
            "(2*e^2*N_A*1000)"
        ),
        "scipy_version": scipy.__version__,
        "scipy_constants": {
            "vacuum_electric_permittivity_F_m": constants.epsilon_0,
            "Boltzmann_constant_J_K": constants.k,
            "elementary_charge_C": constants.e,
            "Avogadro_constant_mol_inverse": constants.N_A,
        },
        "pyddx_version": importlib.metadata.version("pyddx"),
        "pyddx_solvent_kappa_inverse_bohr": pyddx_kappa_inverse_bohr,
        "roundtrip_kappa_inverse_angstrom": roundtrip,
        "relative_roundtrip_error": abs(roundtrip - requested) / abs(requested),
        # pyddx and SciPy embed independently rounded CODATA constants.
        "constants_relative_tolerance": 1.0e-8,
        "roundtrip_within_constants_tolerance": (
            abs(roundtrip - requested) / abs(requested) <= 1.0e-8
        ),
    }


def render_apbs_sphere_input(
    *,
    pqr_basename: str,
    ionic_strength_molar: float,
    spacing_angstrom: float = 0.25,
    points_per_axis: int = 97,
) -> str:
    """Render one matched hard-sphere LPB comparison without an APOLAR block."""
    if Path(pqr_basename).name != pqr_basename:
        raise ValueError("APBS PQR input must be a basename.")
    shared = f"""\
    mg-manual
    dime {points_per_axis} {points_per_axis} {points_per_axis}
    nlev 4
    grid {spacing_angstrom:.12f} {spacing_angstrom:.12f} {spacing_angstrom:.12f}
    gcent mol 1
    mol 1
    lpbe
    bcfl mdh
    pdie 1.000000
    chgm spl2
    srfm mol
    srad 0.000000
    swin 0.300000
    sdens 10.000000
    temp {TEMPERATURE_K:.6f}
    calcenergy total
    calcforce no"""
    ions = f"""\
    ion charge +1 conc {ionic_strength_molar:.12f} radius 0.000000
    ion charge -1 conc {ionic_strength_molar:.12f} radius 0.000000"""
    return f"""\
read
    mol pqr {pqr_basename}
end
elec name solv
{shared}
    sdie {SOLVENT_EPSILON:.6f}
{ions}
end
elec name ref
{shared}
    sdie {SOLUTE_EPSILON:.6f}
end
print elecEnergy solv - ref end
quit
"""


_ENERGY_PATTERN = re.compile(
    r"(?:Global\s+net\s+ELEC\s+energy\s*=|PRINT\s+ELEC\s+ENERGY[^:]*:)\s*"
    r"([-+0-9.Ee]+)\s*(kJ/mol)",
    re.IGNORECASE,
)
_DIAGNOSTIC_PATTERNS = {
    "ionic_strength": re.compile(r"ionic\s+strength", re.IGNORECASE),
    "temperature": re.compile(r"temperature|\btemp\b", re.IGNORECASE),
    "epsilon": re.compile(r"epsilon|dielectric", re.IGNORECASE),
    "debye_length": re.compile(r"debye", re.IGNORECASE),
    "xkappa": re.compile(r"xkappa", re.IGNORECASE),
}
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def parse_apbs_report(stdout: str) -> dict[str, Any]:
    """Parse APBS energy and retain actual diagnostic lines/printed precision."""
    energy_matches = _ENERGY_PATTERN.findall(stdout)
    if len(energy_matches) != 1:
        raise ValueError(
            "APBS output must contain exactly one named ELEC solvation-energy result."
        )
    printed_energy, unit = energy_matches[0]
    reported: dict[str, list[dict[str, Any]]] = {
        key: [] for key in _DIAGNOSTIC_PATTERNS
    }
    for line in stdout.splitlines():
        for key, pattern in _DIAGNOSTIC_PATTERNS.items():
            if not pattern.search(line):
                continue
            numbers = _NUMBER_PATTERN.findall(line)
            reported[key].append(
                {
                    "raw_line": line,
                    "printed": numbers[-1] if numbers else None,
                    "parsed_value": float(numbers[-1]) if numbers else None,
                }
            )
    return {
        "polar_energy": {
            "printed": printed_energy,
            "unit": unit,
            "value_kj_mol": float(printed_energy),
            "value_hartree": float(printed_energy) / KJ_MOL_PER_HARTREE,
        },
        "reported": reported,
    }


def assess_independent_crosscheck(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed unless a separately reviewed convergence analysis exists."""
    successful = [record for record in records if record.get("status") == "success"]
    required_grid_keys = {"spacing_angstrom", "domain_length_angstrom"}
    malformed = [
        record for record in successful if not required_grid_keys <= record.keys()
    ]
    spacings: list[float] = []
    domains: list[float] = []
    reason = "successful APBS record is missing grid or domain metadata"
    if not malformed:
        spacings = sorted(
            {float(record["spacing_angstrom"]) for record in successful}
        )
        domains = sorted(
            {float(record["domain_length_angstrom"]) for record in successful}
        )
        if len(successful) < 6:
            reason = "fewer than six successful APBS grid/domain calculations"
        elif len(spacings) < 3 or len(domains) < 2:
            reason = "missing three grid spacings and two domain sizes"
        else:
            reason = (
                "raw grid/domain evidence exists but no defensible asymptotic "
                "extrapolation and uncertainty review is implemented in this "
                "bounded harness"
            )
    return {
        "independent_crosscheck_complete": False,
        "solver_agreement_claim": False,
        "successful_record_count": len(successful),
        "distinct_grid_spacings_angstrom": spacings,
        "distinct_domain_lengths_angstrom": domains,
        "reason": reason,
    }


def _write_sphere_mol2(path: Path) -> None:
    path.write_text(
        """@<TRIPOS>MOLECULE
DDLPB_CHARGED_SPHERE
 1 0 1 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000000000 0.0000000000 0.0000000000 C.3 1 MOL 1.000000000000
@<TRIPOS>SUBSTRUCTURE
     1 MOL         1 TEMP              0 ****  ****    0 ROOT
""",
        encoding="utf-8",
    )


def _evaluate_timed(provider, atoms, *, need_forces: bool) -> tuple[Any, float]:
    started = time.perf_counter()
    result = provider.evaluate(atoms, need_forces=need_forces)
    return result, time.perf_counter() - started


def _finite_difference_record(
    provider,
    atoms,
    analytic_forces,
    step: float,
    *,
    progress: dict[str, Any],
) -> dict[str, Any]:
    finite_difference = np.empty((len(atoms), 3), dtype=np.float64)
    plus_energies = np.empty_like(finite_difference)
    minus_energies = np.empty_like(finite_difference)
    timings = np.empty((len(atoms), 3, 2), dtype=np.float64)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            plus.positions[atom_index, axis] += step
            minus.positions[atom_index, axis] -= step
            progress["active_stage"] = {
                "name": "molecular_finite_difference",
                "step_angstrom": step,
                "atom_index": atom_index,
                "axis": axis,
                "side": "plus",
            }
            plus_result, plus_time = _evaluate_timed(
                provider, plus, need_forces=False
            )
            progress["molecular_fd_component_evaluations"].append(
                {
                    **progress["active_stage"],
                    "status": "success",
                    "energy_hartree": float(plus_result.energy_hartree),
                    "timing_seconds": plus_time,
                }
            )
            progress["active_stage"] = {
                "name": "molecular_finite_difference",
                "step_angstrom": step,
                "atom_index": atom_index,
                "axis": axis,
                "side": "minus",
            }
            minus_result, minus_time = _evaluate_timed(
                provider, minus, need_forces=False
            )
            progress["molecular_fd_component_evaluations"].append(
                {
                    **progress["active_stage"],
                    "status": "success",
                    "energy_hartree": float(minus_result.energy_hartree),
                    "timing_seconds": minus_time,
                }
            )
            plus_energies[atom_index, axis] = plus_result.energy_hartree
            minus_energies[atom_index, axis] = minus_result.energy_hartree
            timings[atom_index, axis] = [plus_time, minus_time]
            finite_difference[atom_index, axis] = -(
                plus_result.energy_hartree - minus_result.energy_hartree
            ) / (2.0 * step)
    errors = np.asarray(analytic_forces) - finite_difference
    record = {
        "step_angstrom": step,
        "denominators_angstrom": [2.0 * step] * (len(atoms) * 3),
        "plus_energies_hartree": plus_energies,
        "minus_energies_hartree": minus_energies,
        "finite_difference_forces_hartree_per_angstrom": finite_difference,
        "analytic_forces_hartree_per_angstrom": analytic_forces,
        "signed_errors_hartree_per_angstrom": errors,
        "absolute_errors_hartree_per_angstrom": np.abs(errors),
        "rmse_hartree_per_angstrom": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_hartree_per_angstrom": float(np.max(np.abs(errors))),
        "evaluation_timings_seconds": timings,
    }
    progress["completed_fd_steps"].append(record)
    return record


def _provider_settings(provider) -> dict[str, Any]:
    if not hasattr(provider, "settings"):
        raise ValueError("ddLPB provider did not expose its mandatory settings.")
    return _json_safe(provider.settings)


def _run_core_validation(
    *,
    molecular_atoms,
    molecular_charges: np.ndarray,
    sphere_atoms,
    output_dir: Path,
    kappa: float,
    provider_class,
    settings_class,
    progress: dict[str, Any],
) -> dict[str, Any]:
    base_settings = settings_class()
    sphere_records = []
    first_sphere_provider = None
    for sphere_kappa in (kappa, 2.0 * kappa):
        progress["active_stage"] = {
            "name": "charged_sphere",
            "solvent_kappa_inverse_angstrom": sphere_kappa,
        }
        provider = provider_class(
            sphere_atoms,
            sphere_atoms.get_initial_charges(),
            solvent_kappa_inverse_angstrom=sphere_kappa,
            settings=base_settings,
            audit_dir=output_dir / f"sphere-kappa-{sphere_kappa:.12g}",
        )
        if first_sphere_provider is None:
            first_sphere_provider = provider
        _provider_settings(provider)
        result, elapsed = _evaluate_timed(provider, sphere_atoms, need_forces=True)
        force = np.asarray(result.forces_hartree_per_angstrom, dtype=np.float64)
        exact = charged_sphere_exact_energy_hartree(
            charge_e=float(provider.charges[0]),
            radius_angstrom=float(provider.radii[0]),
            solvent_kappa_inverse_angstrom=sphere_kappa,
            solute_epsilon=float(base_settings.solute_epsilon),
            solvent_epsilon=float(base_settings.solvent_epsilon),
        )
        sphere_record = {
                "solvent_kappa_inverse_angstrom": sphere_kappa,
                "charge_e": float(provider.charges[0]),
                "radius_angstrom": float(provider.radii[0]),
                "computed_energy_hartree": float(result.energy_hartree),
                "exact_energy_hartree": exact,
                "signed_error_hartree": float(result.energy_hartree - exact),
                "absolute_error_hartree": abs(float(result.energy_hartree - exact)),
                "forces_hartree_per_angstrom": force,
                "translation_force_norm_hartree_per_angstrom": float(
                    np.linalg.norm(np.sum(force, axis=0))
                ),
                "timing_seconds": elapsed,
                "components_hartree": result.components_hartree,
                "result_provenance": result.provenance,
                "provider_provenance": provider.provenance,
        }
        sphere_records.append(sphere_record)
        progress["completed_sphere_records"].append(sphere_record)
    assert first_sphere_provider is not None

    progress["active_stage"] = {"name": "molecular_working_analytic"}
    provider = provider_class(
        molecular_atoms,
        molecular_charges,
        solvent_kappa_inverse_angstrom=kappa,
        settings=base_settings,
        audit_dir=output_dir / "molecular-working",
    )
    working_settings_record = _provider_settings(provider)
    charges_match = np.array_equal(np.asarray(provider.charges), molecular_charges)
    radii = np.asarray(provider.radii, dtype=np.float64)
    input_match = {
        "charges_exact": bool(charges_match),
        "charge_vector_sha256": sha256_bytes(
            canonical_json_bytes(molecular_charges.tolist())
        ),
        "provider_charge_vector_sha256": sha256_bytes(
            canonical_json_bytes(np.asarray(provider.charges).tolist())
        ),
        "radii_shape_matches_atoms": radii.shape == (len(molecular_atoms),),
        "radii_finite_positive": bool(
            radii.shape == (len(molecular_atoms),)
            and np.isfinite(radii).all()
            and np.all(radii > 0.0)
        ),
        "radius_vector_angstrom": radii,
        "radius_vector_sha256": sha256_bytes(canonical_json_bytes(radii.tolist())),
    }
    if not all(
        input_match[key]
        for key in (
            "charges_exact",
            "radii_shape_matches_atoms",
            "radii_finite_positive",
        )
    ):
        raise ValueError(
            "ddLPB provider inputs do not match the fixed molecular input."
        )
    working, working_time = _evaluate_timed(provider, molecular_atoms, need_forces=True)
    analytic_forces = np.asarray(
        working.forces_hartree_per_angstrom, dtype=np.float64
    )
    if analytic_forces.shape != (len(molecular_atoms), 3):
        raise ValueError("ddLPB molecular force output has the wrong shape.")
    fd_records = [
        _finite_difference_record(
            provider,
            molecular_atoms,
            analytic_forces,
            step,
            progress=progress,
        )
        for step in FD_STEPS_ANGSTROM
    ]
    net_force_norm = float(np.linalg.norm(np.sum(analytic_forces, axis=0)))
    force_gate = {
        "rmse_at_both_steps": all(
            row["rmse_hartree_per_angstrom"] <= FD_RMSE_TOLERANCE_HARTREE_PER_ANGSTROM
            for row in fd_records
        ),
        "maximum_at_both_steps": all(
            row["maximum_absolute_error_hartree_per_angstrom"]
            <= FD_MAX_TOLERANCE_HARTREE_PER_ANGSTROM
            for row in fd_records
        ),
        "net_force": net_force_norm <= NET_FORCE_TOLERANCE_HARTREE_PER_ANGSTROM,
    }
    force_gate["passed"] = all(force_gate.values())

    strict_settings = replace(
        base_settings,
        lmax=11,
        n_lebedev=590,
        solver_tolerance=1.0e-12,
    )
    progress["active_stage"] = {"name": "molecular_refinement"}
    strict_provider = provider_class(
        molecular_atoms,
        molecular_charges,
        solvent_kappa_inverse_angstrom=kappa,
        settings=strict_settings,
        audit_dir=output_dir / "molecular-refined",
    )
    strict_settings_record = _provider_settings(strict_provider)
    strict, strict_time = _evaluate_timed(
        strict_provider, molecular_atoms, need_forces=True
    )
    strict_forces = np.asarray(strict.forces_hartree_per_angstrom, dtype=np.float64)
    refinement_finite = bool(
        math.isfinite(float(working.energy_hartree))
        and math.isfinite(float(strict.energy_hartree))
        and analytic_forces.shape == strict_forces.shape == (len(molecular_atoms), 3)
        and np.isfinite(analytic_forces).all()
        and np.isfinite(strict_forces).all()
    )
    refinement = {
        "working_settings": working_settings_record,
        "strict_settings": strict_settings_record,
        "working_energy_hartree": float(working.energy_hartree),
        "strict_energy_hartree": float(strict.energy_hartree),
        "working_components_hartree": working.components_hartree,
        "strict_components_hartree": strict.components_hartree,
        "energy_delta_strict_minus_working_hartree": float(
            strict.energy_hartree - working.energy_hartree
        ),
        "working_forces_hartree_per_angstrom": analytic_forces,
        "strict_forces_hartree_per_angstrom": strict_forces,
        "force_delta_hartree_per_angstrom": strict_forces - analytic_forces,
        "maximum_absolute_force_delta_hartree_per_angstrom": float(
            np.max(np.abs(strict_forces - analytic_forces))
        ),
        "timings_seconds": {"working": working_time, "strict": strict_time},
        "finite_complete_converged": refinement_finite,
        "promotion_threshold": None,
        "twenty_structure_panel_required_for_first_release": False,
        "working_result_provenance": working.provenance,
        "strict_result_provenance": strict.provenance,
    }
    sphere_gate = {
        "energy_at_both_kappas": all(
            row["absolute_error_hartree"] <= SPHERE_ENERGY_ABS_TOLERANCE_HARTREE
            for row in sphere_records
        ),
        "translation_force_at_both_kappas": all(
            row["translation_force_norm_hartree_per_angstrom"]
            <= SPHERE_FORCE_NORM_TOLERANCE_HARTREE_PER_ANGSTROM
            for row in sphere_records
        ),
    }
    sphere_gate["passed"] = all(sphere_gate.values())
    core_passed = bool(
        sphere_gate["passed"] and force_gate["passed"] and refinement_finite
    )
    progress["active_stage"] = {"name": "complete"}
    return {
        "core_validation": {
            "passed": core_passed,
            "apbs_is_nonblocking": True,
            "sphere_gate": sphere_gate,
            "molecular_force_gate": force_gate,
            "refinement_finite_complete": refinement_finite,
        },
        "charged_sphere_validation": sphere_records,
        "molecular_force_validation": {
            "analytic_energy_hartree": float(working.energy_hartree),
            "components_hartree": working.components_hartree,
            "analytic_forces_hartree_per_angstrom": analytic_forces,
            "net_force_hartree_per_angstrom": np.sum(analytic_forces, axis=0),
            "net_force_norm_hartree_per_angstrom": net_force_norm,
            "steps": fd_records,
            "thresholds": {
                "rmse_hartree_per_angstrom": FD_RMSE_TOLERANCE_HARTREE_PER_ANGSTROM,
                "maximum_absolute_error_hartree_per_angstrom": (
                    FD_MAX_TOLERANCE_HARTREE_PER_ANGSTROM
                ),
                "net_force_norm_hartree_per_angstrom": (
                    NET_FORCE_TOLERANCE_HARTREE_PER_ANGSTROM
                ),
            },
        },
        "refinement": refinement,
        "provider_input_match": input_match,
        "sphere_apbs_inputs": {
            "charge_e": float(first_sphere_provider.charges[0]),
            "radius_angstrom": float(first_sphere_provider.radii[0]),
            "ddx_energy_hartree": sphere_records[0]["computed_energy_hartree"],
        },
    }


def _run_apbs_crosscheck(
    *,
    executable: str | os.PathLike[str] | None,
    output_dir: Path,
    sphere_inputs: dict[str, float],
    kappa_mapping: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "role": "independent_optional_crosscheck",
        "blocking_core_provider_release": False,
        "same_bvp_scope": "one hard charged sphere only",
        "molecular_comparison_scope": "not_run",
        "physical_parameters_verified": False,
        "records": [],
    }
    if executable is None:
        base.update({"status": "not_requested"})
        base.update(assess_independent_crosscheck([]))
        return base

    apbs_dir = output_dir / "apbs-sphere"
    apbs_dir.mkdir()
    pqr = apbs_dir / "sphere.pqr"
    pqr.write_text(
        "ATOM      1 C1   MOL A   1      "
        f"{0.0:14.10f}{0.0:14.10f}{0.0:14.10f} "
        f"{sphere_inputs['charge_e']:16.12f} "
        f"{sphere_inputs['radius_angstrom']:14.10f}\n"
        "END\n",
        encoding="utf-8",
    )
    spacing = 0.25
    points = 97
    domain = (points - 1) * spacing
    input_path = apbs_dir / "sphere.in"
    input_path.write_text(
        render_apbs_sphere_input(
            pqr_basename=pqr.name,
            ionic_strength_molar=kappa_mapping["ionic_strength_molar"],
            spacing_angstrom=spacing,
            points_per_axis=points,
        ),
        encoding="utf-8",
    )
    executable_text = os.fspath(executable)
    resolved_executable = shutil.which(executable_text)
    executable_path = (
        Path(resolved_executable).resolve() if resolved_executable is not None else None
    )
    record: dict[str, Any] = {
        "spacing_angstrom": spacing,
        "points_per_axis": points,
        "domain_length_angstrom": domain,
        "command": [executable_text, input_path.name],
        "cwd": str(apbs_dir),
        "input_basename": input_path.name,
        "input_sha256": sha256_file(input_path),
        "pqr_sha256": sha256_file(pqr),
        "pqr_charge_e": sphere_inputs["charge_e"],
        "pqr_radius_angstrom": sphere_inputs["radius_angstrom"],
        "executable_resolved_path": (
            str(executable_path) if executable_path is not None else None
        ),
        "executable_sha256": (
            sha256_file(executable_path) if executable_path is not None else None
        ),
    }
    try:
        version = subprocess.run(
            [executable_text, "--version"],
            cwd=apbs_dir,
            text=True,
            capture_output=True,
            timeout=30.0,
            check=False,
        )
        record["version_probe"] = {
            "returncode": version.returncode,
            "stdout": version.stdout,
            "stderr": version.stderr,
        }
        completed = subprocess.run(
            [executable_text, input_path.name],
            cwd=apbs_dir,
            text=True,
            capture_output=True,
            timeout=3600.0,
            check=False,
        )
        record.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise RuntimeError(f"APBS exited with status {completed.returncode}")
        parsed = parse_apbs_report(completed.stdout)
        generated_diagnostics = []
        for generated in sorted(apbs_dir.glob("*")):
            if not generated.is_file() or generated in {pqr, input_path}:
                continue
            try:
                generated_text = generated.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            xkappa_lines = [
                line
                for line in generated_text.splitlines()
                if _DIAGNOSTIC_PATTERNS["xkappa"].search(line)
            ]
            if xkappa_lines:
                generated_diagnostics.append(
                    {
                        "basename": generated.name,
                        "sha256": sha256_file(generated),
                        "xkappa_lines": xkappa_lines,
                    }
                )
        apbs_energy = float(parsed["polar_energy"]["value_hartree"])
        record.update(
            {
                "status": "success",
                "parsed": parsed,
                "generated_diagnostics": generated_diagnostics,
                "signed_difference_apbs_minus_ddx_hartree": (
                    apbs_energy - sphere_inputs["ddx_energy_hartree"]
                ),
                "absolute_difference_hartree": abs(
                    apbs_energy - sphere_inputs["ddx_energy_hartree"]
                ),
            }
        )
        base["status"] = "raw_comparison_complete"
    except Exception as exc:  # noqa: BLE001 - optional external failure is evidence
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        base["status"] = "failed"
    base["records"] = [record]
    base.update(assess_independent_crosscheck([record]))
    return base


def run_validation(
    *,
    mol2_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    solvent_kappa_inverse_angstrom: float,
    apbs_executable: str | os.PathLike[str] | None = None,
    provider_class=None,
    settings_class=None,
) -> dict[str, Any]:
    """Run validation into a new directory and return its sealed JSON artifact."""
    kappa = float(solvent_kappa_inverse_angstrom)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError(
            "solvent_kappa_inverse_angstrom must be finite and strictly positive."
        )
    source = Path(mol2_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory must not already exist: {destination}")
    destination.mkdir(parents=True)
    molecular_copy = destination / "molecular-input.mol2"
    molecular_copy.write_bytes(source.read_bytes())
    sphere_path = destination / "sphere-input.mol2"
    _write_sphere_mol2(sphere_path)

    if provider_class is None or settings_class is None:
        from maple.function.calculator.extra_correction.implicit.ddx_lpb import (
            DDXLPB,
            DDXLPBSettings,
        )

        provider_class = provider_class or DDXLPB
        settings_class = settings_class or DDXLPBSettings
    molecular_atoms = MOL2Reader(str(molecular_copy), charge=None, mult=1)
    sphere_atoms = MOL2Reader(str(sphere_path), charge=1, mult=1)
    molecular_charges = np.asarray(
        molecular_atoms.get_initial_charges(), dtype=np.float64
    )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "route1-ddlpb-reference-validation",
        "claim_scope": (
            "bounded ddLPB polar E/F physical and derivative validation; no hydration-"
            "accuracy, product, performance, OPT/MD/TS, or automatic APBS-"
            "agreement claim"
        ),
        "inputs": {
            "molecular_mol2": {
                "original_path": str(source),
                "preserved_copy": molecular_copy.name,
                "sha256": sha256_file(molecular_copy),
                "bytes": molecular_copy.stat().st_size,
                "atom_count": len(molecular_atoms),
                "coordinates_angstrom": np.asarray(molecular_atoms.positions),
                "charges_e": molecular_charges,
                "mol2_identity": molecular_atoms.info.get("mol2"),
            },
            "sphere_mol2": {
                "preserved_copy": sphere_path.name,
                "sha256": sha256_file(sphere_path),
                "bytes": sphere_path.stat().st_size,
            },
            "solvent_kappa_inverse_angstrom": kappa,
            "finite_difference_steps_angstrom": list(FD_STEPS_ANGSTROM),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "command_provenance": command_provenance(
            __file__,
            {
                "mol2": str(source),
                "output_dir": str(destination),
                "solvent_kappa_inverse_angstrom": kappa,
                "apbs": None if apbs_executable is None else str(apbs_executable),
            },
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "PYTHONPATH",
            ),
        ),
    }
    core_progress: dict[str, Any] = {
        "active_stage": {"name": "not_started"},
        "completed_sphere_records": [],
        "molecular_fd_component_evaluations": [],
        "completed_fd_steps": [],
    }
    try:
        core = _run_core_validation(
            molecular_atoms=molecular_atoms,
            molecular_charges=molecular_charges,
            sphere_atoms=sphere_atoms,
            output_dir=destination,
            kappa=kappa,
            provider_class=provider_class,
            settings_class=settings_class,
            progress=core_progress,
        )
        artifact.update(core)
        artifact["inputs"]["provider_input_match"] = core["provider_input_match"]
        artifact.pop("provider_input_match")
        mapping = kappa_mapping_record(kappa)
        artifact["kappa_ionic_strength_mapping"] = mapping
        artifact["independent_apbs_crosscheck"] = _run_apbs_crosscheck(
            executable=apbs_executable,
            output_dir=destination,
            sphere_inputs=core["sphere_apbs_inputs"],
            kappa_mapping=mapping,
        )
        artifact.pop("sphere_apbs_inputs")
        artifact["overall_release_gate"] = {
            "passed": bool(artifact["core_validation"]["passed"]),
            "apbs_crosscheck_is_nonblocking": True,
            "independent_crosscheck_complete": artifact[
                "independent_apbs_crosscheck"
            ]["independent_crosscheck_complete"],
        }
        artifact["status"] = (
            "success" if artifact["overall_release_gate"]["passed"] else "failed"
        )
    except Exception as exc:
        failed_stage = dict(core_progress["active_stage"])
        failed_stage.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        core_progress["failed_stage"] = failed_stage
        core_progress["completion_counts"] = {
            "charged_sphere_records": len(
                core_progress["completed_sphere_records"]
            ),
            "finite_difference_component_sides": len(
                core_progress["molecular_fd_component_evaluations"]
            ),
            "complete_finite_difference_steps": len(
                core_progress["completed_fd_steps"]
            ),
        }
        artifact["status"] = "failed"
        artifact["core_validation"] = {
            "passed": False,
            "apbs_is_nonblocking": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        artifact["partial_core_evidence"] = core_progress
        sealed = seal_artifact(_json_safe(artifact))
        write_json_atomic(destination / "validation.json", sealed)
        raise RuntimeError(
            f"ddLPB core validation failed; partial evidence retained in {destination}"
        ) from exc
    sealed = seal_artifact(_json_safe(artifact))
    write_json_atomic(destination / "validation.json", sealed)
    return sealed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--solvent-kappa-inverse-angstrom", required=True, type=float
    )
    parser.add_argument(
        "--apbs",
        type=Path,
        default=None,
        help=(
            "Optional APBS executable for a nonblocking single-grid raw sphere "
            "comparison."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_validation(
        mol2_path=args.mol2,
        output_dir=args.output_dir,
        solvent_kappa_inverse_angstrom=args.solvent_kappa_inverse_angstrom,
        apbs_executable=args.apbs,
    )
    print(
        f"ddLPB validation {artifact['status']}: "
        f"{Path(args.output_dir).resolve() / 'validation.json'}"
    )
    return 0 if artifact["overall_release_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
