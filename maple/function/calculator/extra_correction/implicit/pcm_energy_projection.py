"""PCM-energy-norm projection into one fixed GTO Galerkin basis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import null_space

from .gto_galerkin import FixedCavityGTOGalerkinOperator


def _immutable(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PCMEnergyNormProjection:
    """One constrained affine projection and its exact diagnostic ledger."""

    coefficients: np.ndarray
    fitted_surface_potential_hartree_per_e: np.ndarray
    residual_surface_potential_hartree_per_e: np.ndarray
    target_constraints: np.ndarray
    achieved_constraints: np.ndarray
    constraint_residual_inf: float
    tangent_optimality_inf: float
    shifted_target_energy_norm_squared_hartree: float
    shifted_fit_energy_norm_squared_hartree: float
    residual_energy_norm_squared_hartree: float
    shifted_pythagorean_error_hartree: float
    target_polarization_energy_hartree: float
    fitted_polarization_energy_hartree: float
    polarization_energy_error_hartree: float

    def __post_init__(self) -> None:
        for name in (
            "coefficients",
            "fitted_surface_potential_hartree_per_e",
            "residual_surface_potential_hartree_per_e",
            "target_constraints",
            "achieved_constraints",
        ):
            object.__setattr__(
                self,
                name,
                _immutable(getattr(self, name), name=name),
            )
        for name in (
            "constraint_residual_inf",
            "tangent_optimality_inf",
            "shifted_target_energy_norm_squared_hartree",
            "shifted_fit_energy_norm_squared_hartree",
            "residual_energy_norm_squared_hartree",
            "shifted_pythagorean_error_hartree",
            "target_polarization_energy_hartree",
            "fitted_polarization_energy_hartree",
            "polarization_energy_error_hartree",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if (
                name.endswith("_inf")
                or "norm_squared" in name
                or name == "shifted_pythagorean_error_hartree"
            ):
                if value < -1.0e-12:
                    raise ValueError(f"{name} must be nonnegative.")
                value = max(0.0, value)
            object.__setattr__(self, name, value)


def project_surface_potential_in_pcm_energy_norm(
    operator: FixedCavityGTOGalerkinOperator,
    target_surface_potential_hartree_per_e: np.ndarray,
    *,
    total_charge_e: float,
    molecular_dipole_e_angstrom: np.ndarray,
    constraint_tolerance: float = 1.0e-10,
    optimality_tolerance: float = 1.0e-9,
    semidefinite_tolerance_hartree: float = 1.0e-10,
    pythagorean_tolerance_hartree: float = 1.0e-8,
) -> PCMEnergyNormProjection:
    """Project one MEP with exact charge/dipole constraints.

    The feasible coefficient set is affine.  Therefore the rigorous
    Pythagorean identity is evaluated after subtracting a particular feasible
    potential; no false absolute-energy identity is asserted for nonzero
    charge or dipole targets.
    """

    potential = np.asarray(
        target_surface_potential_hartree_per_e,
        dtype=float,
    )
    surface_operator = operator.surface_operator
    if potential.shape != (surface_operator.shape[0],) or not np.all(
        np.isfinite(potential)
    ):
        raise ValueError(
            "Target surface potential must be finite with one value per "
            "continuum point."
        )
    dipole = np.asarray(molecular_dipole_e_angstrom, dtype=float)
    if dipole.shape != (3,) or not np.all(np.isfinite(dipole)):
        raise ValueError("Target molecular dipole must be finite with shape (3,).")
    charge = float(total_charge_e)
    if not math.isfinite(charge):
        raise ValueError("Target molecular charge must be finite.")
    tolerances = (
        constraint_tolerance,
        optimality_tolerance,
        semidefinite_tolerance_hartree,
        pythagorean_tolerance_hartree,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("Projection tolerances must be finite and positive.")

    constraints = operator.basis.molecular_charge_dipole_constraints(
        operator.atom_positions_angstrom
    )
    target = np.concatenate(([charge], dipole))
    constraint_gram = constraints @ constraints.T
    particular = constraints.T @ np.linalg.solve(
        constraint_gram,
        target,
    )
    tangent = null_space(constraints)

    pcm_metric = -operator.galerkin_matrix_hartree
    minimum_metric_eigenvalue = float(np.min(np.linalg.eigvalsh(pcm_metric)))
    if minimum_metric_eigenvalue < -semidefinite_tolerance_hartree:
        raise RuntimeError(
            "Continuum response does not define a positive-semidefinite PCM "
            "energy norm in this GTO basis."
        )
    target_charge = operator.apply_surface_response(potential)
    linear_term = -(surface_operator.T @ target_charge)
    if tangent.shape[1]:
        reduced_metric = tangent.T @ pcm_metric @ tangent
        reduced_rhs = tangent.T @ (linear_term - pcm_metric @ particular)
        reduced_solution = np.linalg.lstsq(
            reduced_metric,
            reduced_rhs,
            rcond=1.0e-12,
        )[0]
        coefficient_vector = particular + tangent @ reduced_solution
    else:
        coefficient_vector = particular

    achieved = constraints @ coefficient_vector
    constraint_residual = float(np.max(np.abs(achieved - target)))
    if constraint_residual > constraint_tolerance:
        raise RuntimeError(
            "PCM-energy projection failed its exact molecular-moment " "constraints."
        )
    gradient = pcm_metric @ coefficient_vector - linear_term
    tangent_optimality = (
        0.0 if tangent.shape[1] == 0 else float(np.max(np.abs(tangent.T @ gradient)))
    )
    if tangent_optimality > optimality_tolerance:
        raise RuntimeError(
            "PCM-energy projection did not reach the constrained tangent " "optimum."
        )

    fitted = surface_operator @ coefficient_vector
    residual = potential - fitted
    particular_potential = surface_operator @ particular
    shifted_target = potential - particular_potential
    shifted_fit = fitted - particular_potential
    response_particular = operator.apply_surface_response(particular_potential)
    response_shifted_target = target_charge - response_particular
    response_shifted_fit = operator.apply_surface_response(shifted_fit)
    response_residual = response_shifted_target - response_shifted_fit

    def energy_norm_squared(
        values: np.ndarray,
        response_values: np.ndarray,
    ) -> float:
        result = -float(np.dot(values, response_values))
        if result < -semidefinite_tolerance_hartree:
            raise RuntimeError("PCM energy norm became negative beyond tolerance.")
        return max(0.0, result)

    shifted_target_norm = energy_norm_squared(
        shifted_target,
        response_shifted_target,
    )
    shifted_fit_norm = energy_norm_squared(
        shifted_fit,
        response_shifted_fit,
    )
    residual_norm = energy_norm_squared(
        residual,
        response_residual,
    )
    pythagorean_error = abs(shifted_target_norm - shifted_fit_norm - residual_norm)
    if pythagorean_error > pythagorean_tolerance_hartree:
        raise RuntimeError(
            "PCM-energy projection failed its shifted affine Pythagorean " "identity."
        )
    target_energy = 0.5 * float(np.dot(potential, target_charge))
    fitted_charge = operator.apply_surface_response(fitted)
    fitted_energy = 0.5 * float(np.dot(fitted, fitted_charge))
    if (
        target_energy > semidefinite_tolerance_hartree
        or fitted_energy > semidefinite_tolerance_hartree
    ):
        raise RuntimeError(
            "Continuum response produced a positive polarization energy "
            "beyond tolerance."
        )

    coefficients = operator.basis.unflatten(
        coefficient_vector,
        atom_count=operator.atom_count,
        name="projected_gto_coefficients",
    )
    return PCMEnergyNormProjection(
        coefficients=coefficients,
        fitted_surface_potential_hartree_per_e=fitted,
        residual_surface_potential_hartree_per_e=residual,
        target_constraints=target,
        achieved_constraints=achieved,
        constraint_residual_inf=constraint_residual,
        tangent_optimality_inf=tangent_optimality,
        shifted_target_energy_norm_squared_hartree=shifted_target_norm,
        shifted_fit_energy_norm_squared_hartree=shifted_fit_norm,
        residual_energy_norm_squared_hartree=residual_norm,
        shifted_pythagorean_error_hartree=pythagorean_error,
        target_polarization_energy_hartree=target_energy,
        fitted_polarization_energy_hartree=fitted_energy,
        polarization_energy_error_hartree=fitted_energy - target_energy,
    )
