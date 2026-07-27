"""Bookkeeping and fixed-source solves for the MNSol response ablation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)

ABLATION_METHODS = (
    "aimnet2_fixed_l0",
    "mace_fixed_l0",
    "mace_fixed_l1",
    "mace_one_shot_l1",
    "mace_scf_l1",
)


def solve_fixed_multipole_continuum(
    reaction_field: Any,
    density_coefficients: np.ndarray,
    *,
    declared_total_charge_e: float = 0.0,
    charge_tolerance_e: float = 1.0e-8,
    energy_identity_tolerance_ev: float = 2.0e-10,
) -> dict[str, object]:
    """Solve one fixed ``l<=1`` source and verify the half-coupling identity."""

    coefficients = np.asarray(density_coefficients, dtype=float)
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] == 0
        or coefficients.shape[1] != 4
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError(
            "Fixed multipole coefficients must be finite with shape " "(n_atoms, 4)."
        )
    declared = float(declared_total_charge_e)
    charge_tolerance = float(charge_tolerance_e)
    identity_tolerance = float(energy_identity_tolerance_ev)
    if not math.isfinite(declared):
        raise ValueError("Declared fixed-source charge must be finite.")
    if not math.isfinite(charge_tolerance) or charge_tolerance <= 0.0:
        raise ValueError("Fixed-source charge tolerance must be positive.")
    if not math.isfinite(identity_tolerance) or identity_tolerance <= 0.0:
        raise ValueError("Energy-identity tolerance must be positive.")
    charge_error = abs(float(np.sum(coefficients[:, 0])) - declared)
    if charge_error > charge_tolerance:
        raise ValueError(
            "Fixed multipole source violates its declared total charge "
            f"(absolute error={charge_error:.3e} e)."
        )

    field = np.asarray(reaction_field.apply_scf(coefficients), dtype=float)
    if field.shape != coefficients.shape or not np.all(np.isfinite(field)):
        raise RuntimeError(
            "Fixed-source reaction field must be finite with shape "
            f"{coefficients.shape}; received {field.shape}."
        )
    energy_hartree = float(reaction_field.scf_polarization_energy_hartree(coefficients))
    if not math.isfinite(energy_hartree):
        raise RuntimeError("Fixed-source continuum energy is non-finite.")
    half_coupling_ev = 0.5 * MACE_POLAR_L1_PAIRING.pair(
        coefficients,
        field,
    )
    provider_energy_ev = energy_hartree * Hartree
    identity_error_ev = abs(half_coupling_ev - provider_energy_ev)
    if identity_error_ev > identity_tolerance:
        raise RuntimeError(
            "Fixed multipole continuum failed the half-coupling identity "
            f"(absolute error={identity_error_ev:.3e} eV)."
        )
    return {
        "reaction_field_values_ev": field.copy(),
        "polarization_energy_hartree": energy_hartree,
        "half_coupling_energy_ev": half_coupling_ev,
        "energy_identity_error_ev": identity_error_ev,
        "declared_total_charge_e": declared,
        "observed_total_charge_e": float(np.sum(coefficients[:, 0])),
    }


def compose_method_ledger(
    *,
    experimental_kcal_mol: float,
    solute_polarization_kcal_mol: float,
    continuum_polarization_kcal_mol: float,
    smd_cds_kcal_mol: float,
    wall_seconds: float,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Close one response-ablation energy ledger in kcal/mol."""

    values = {
        "experimental_kcal_mol": float(experimental_kcal_mol),
        "solute_polarization_kcal_mol": float(solute_polarization_kcal_mol),
        "continuum_polarization_kcal_mol": float(continuum_polarization_kcal_mol),
        "smd_cds_kcal_mol": float(smd_cds_kcal_mol),
        "wall_seconds": float(wall_seconds),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Response-ablation ledger values must be finite.")
    if values["wall_seconds"] < 0.0:
        raise ValueError("Response-ablation wall time cannot be negative.")

    electrostatic = (
        values["solute_polarization_kcal_mol"]
        + values["continuum_polarization_kcal_mol"]
    )
    total = electrostatic + values["smd_cds_kcal_mol"]
    signed_error = total - values["experimental_kcal_mol"]
    result: dict[str, object] = {
        "solute_polarization_kcal_mol": (values["solute_polarization_kcal_mol"]),
        "continuum_polarization_kcal_mol": (values["continuum_polarization_kcal_mol"]),
        "electrostatic_kcal_mol": electrostatic,
        "smd_cds_kcal_mol": values["smd_cds_kcal_mol"],
        "total_solvation_kcal_mol": total,
        "signed_error_kcal_mol": signed_error,
        "absolute_error_kcal_mol": abs(signed_error),
        "wall_seconds": values["wall_seconds"],
    }
    if extra:
        overlap = set(result).intersection(extra)
        if overlap:
            raise ValueError(
                "Response-ablation extra fields cannot replace ledger fields: "
                + ", ".join(sorted(overlap))
            )
        result.update(extra)
    return result


def aggregate_method_metrics(
    records: Sequence[Mapping[str, object]],
    method: str,
) -> dict[str, float | int]:
    """Aggregate one method without exposing row-level MNSol values."""

    if not records:
        raise ValueError("Response-ablation metrics require records.")
    method_rows = []
    experimental = []
    for record in records:
        methods = record.get("methods")
        if not isinstance(methods, Mapping) or method not in methods:
            raise ValueError(f"Response-ablation record is missing method {method!r}.")
        row = methods[method]
        if not isinstance(row, Mapping):
            raise ValueError("Response-ablation method rows must be mappings.")
        method_rows.append(row)
        experimental.append(float(record["experimental_delta_g_kcal_mol"]))

    def _array(field: str) -> np.ndarray:
        values = np.asarray(
            [float(row[field]) for row in method_rows],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Response-ablation field {field!r} must be finite.")
        return values

    errors = _array("signed_error_kcal_mol")
    predicted = _array("total_solvation_kcal_mol")
    solute = _array("solute_polarization_kcal_mol")
    continuum = _array("continuum_polarization_kcal_mol")
    electrostatic = _array("electrostatic_kcal_mol")
    cds = _array("smd_cds_kcal_mol")
    wall = _array("wall_seconds")
    experimental_array = np.asarray(experimental, dtype=float)
    if not np.all(np.isfinite(experimental_array)):
        raise ValueError("Experimental MNSol values must be finite.")

    return {
        "record_count": int(errors.size),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": float(np.mean(np.abs(errors))),
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(np.abs(errors))),
        "mean_predicted_delta_g_kcal_mol": float(np.mean(predicted)),
        "mean_experimental_delta_g_kcal_mol": float(np.mean(experimental_array)),
        "mean_solute_polarization_kcal_mol": float(np.mean(solute)),
        "mean_continuum_polarization_kcal_mol": float(np.mean(continuum)),
        "mean_electrostatic_kcal_mol": float(np.mean(electrostatic)),
        "mean_smd_cds_kcal_mol": float(np.mean(cds)),
        "total_wall_seconds": float(np.sum(wall)),
        "mean_wall_seconds": float(np.mean(wall)),
        "maximum_wall_seconds": float(np.max(wall)),
    }


def paired_method_comparison(
    records: Sequence[Mapping[str, object]],
    *,
    left: str,
    right: str,
) -> dict[str, float | int | str]:
    """Compare paired predictions on exactly the same selected records."""

    if not records:
        raise ValueError("Paired response comparison requires records.")
    left_rows = [record["methods"][left] for record in records]
    right_rows = [record["methods"][right] for record in records]
    left_errors = np.asarray(
        [float(row["absolute_error_kcal_mol"]) for row in left_rows],
        dtype=float,
    )
    right_errors = np.asarray(
        [float(row["absolute_error_kcal_mol"]) for row in right_rows],
        dtype=float,
    )
    energy_delta = np.asarray(
        [
            float(right_row["total_solvation_kcal_mol"])
            - float(left_row["total_solvation_kcal_mol"])
            for left_row, right_row in zip(left_rows, right_rows, strict=True)
        ],
        dtype=float,
    )
    if not (
        np.all(np.isfinite(left_errors))
        and np.all(np.isfinite(right_errors))
        and np.all(np.isfinite(energy_delta))
    ):
        raise ValueError("Paired response comparison requires finite values.")
    tolerance = 1.0e-12
    return {
        "left_method": left,
        "right_method": right,
        "record_count": int(len(records)),
        "right_minus_left_mean_energy_kcal_mol": float(np.mean(energy_delta)),
        "right_minus_left_mean_absolute_error_kcal_mol": float(
            np.mean(right_errors) - np.mean(left_errors)
        ),
        "right_lower_absolute_error_count": int(
            np.sum(right_errors < left_errors - tolerance)
        ),
        "left_lower_absolute_error_count": int(
            np.sum(left_errors < right_errors - tolerance)
        ),
        "absolute_error_tie_count": int(
            np.sum(np.abs(right_errors - left_errors) <= tolerance)
        ),
        "tie_tolerance_kcal_mol": tolerance,
    }


__all__ = [
    "ABLATION_METHODS",
    "aggregate_method_metrics",
    "compose_method_ledger",
    "paired_method_comparison",
    "solve_fixed_multipole_continuum",
]
