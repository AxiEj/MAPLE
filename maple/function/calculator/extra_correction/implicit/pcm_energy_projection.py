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
    relative_spectral_cutoff: float
    reduced_dimension: int
    effective_rank: int
    discarded_mode_count: int
    maximum_eigenvalue_hartree: float
    retained_minimum_eigenvalue_hartree: float
    retained_condition_number: float
    coefficient_l2_norm: float
    coefficient_max_abs: float
    constraint_residual_inf: float
    retained_subspace_optimality_inf: float
    full_tangent_gradient_inf: float
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
            "relative_spectral_cutoff",
            "maximum_eigenvalue_hartree",
            "retained_minimum_eigenvalue_hartree",
            "retained_condition_number",
            "coefficient_l2_norm",
            "coefficient_max_abs",
            "constraint_residual_inf",
            "retained_subspace_optimality_inf",
            "full_tangent_gradient_inf",
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
                or name
                in {
                    "relative_spectral_cutoff",
                    "maximum_eigenvalue_hartree",
                    "retained_minimum_eigenvalue_hartree",
                    "coefficient_l2_norm",
                    "coefficient_max_abs",
                }
            ):
                if value < -1.0e-12:
                    raise ValueError(f"{name} must be nonnegative.")
                value = max(0.0, value)
            object.__setattr__(self, name, value)
        if not 0.0 < self.relative_spectral_cutoff <= 1.0:
            raise ValueError("relative_spectral_cutoff must be in the interval (0, 1].")
        for name in (
            "reduced_dimension",
            "effective_rank",
            "discarded_mode_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
            object.__setattr__(self, name, int(value))
        if self.effective_rank > self.reduced_dimension:
            raise ValueError("effective_rank cannot exceed reduced_dimension.")
        if self.discarded_mode_count != (self.reduced_dimension - self.effective_rank):
            raise ValueError(
                "discarded_mode_count must close the reduced spectral ledger."
            )
        if self.retained_condition_number < 1.0:
            raise ValueError("retained_condition_number must be at least one.")


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
    relative_spectral_cutoff: float = 1.0e-12,
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
    cutoff = float(relative_spectral_cutoff)
    if not math.isfinite(cutoff) or not 0.0 < cutoff <= 1.0:
        raise ValueError("The relative spectral cutoff must be finite and in (0, 1].")

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
        eigenvalues, eigenvectors = np.linalg.eigh(reduced_metric)
        maximum_eigenvalue = max(0.0, float(eigenvalues[-1]))
        retained = eigenvalues > cutoff * maximum_eigenvalue
        effective_rank = int(np.count_nonzero(retained))
        reduced_solution = np.zeros(tangent.shape[1], dtype=float)
        if effective_rank:
            reduced_solution = eigenvectors[:, retained] @ (
                (eigenvectors[:, retained].T @ reduced_rhs) / eigenvalues[retained]
            )
        coefficient_vector = particular + tangent @ reduced_solution
        retained_minimum_eigenvalue = (
            float(eigenvalues[retained][0]) if effective_rank else 0.0
        )
        retained_condition_number = (
            maximum_eigenvalue / retained_minimum_eigenvalue if effective_rank else 1.0
        )
    else:
        retained = np.empty(0, dtype=bool)
        effective_rank = 0
        maximum_eigenvalue = 0.0
        retained_minimum_eigenvalue = 0.0
        retained_condition_number = 1.0
        coefficient_vector = particular

    achieved = constraints @ coefficient_vector
    constraint_residual = float(np.max(np.abs(achieved - target)))
    if constraint_residual > constraint_tolerance:
        raise RuntimeError(
            "PCM-energy projection failed its exact molecular-moment " "constraints."
        )
    gradient = pcm_metric @ coefficient_vector - linear_term
    reduced_gradient = tangent.T @ gradient
    full_tangent_gradient = (
        0.0 if tangent.shape[1] == 0 else float(np.max(np.abs(reduced_gradient)))
    )
    retained_subspace_optimality = (
        0.0
        if effective_rank == 0
        else float(np.max(np.abs(eigenvectors[:, retained].T @ reduced_gradient)))
    )
    if retained_subspace_optimality > optimality_tolerance:
        raise RuntimeError(
            "PCM-energy projection did not reach the retained spectral "
            "subspace optimum."
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
        relative_spectral_cutoff=cutoff,
        reduced_dimension=int(tangent.shape[1]),
        effective_rank=effective_rank,
        discarded_mode_count=int(tangent.shape[1]) - effective_rank,
        maximum_eigenvalue_hartree=maximum_eigenvalue,
        retained_minimum_eigenvalue_hartree=retained_minimum_eigenvalue,
        retained_condition_number=retained_condition_number,
        coefficient_l2_norm=float(np.linalg.norm(coefficient_vector)),
        coefficient_max_abs=float(np.max(np.abs(coefficient_vector))),
        constraint_residual_inf=constraint_residual,
        retained_subspace_optimality_inf=retained_subspace_optimality,
        full_tangent_gradient_inf=full_tangent_gradient,
        shifted_target_energy_norm_squared_hartree=shifted_target_norm,
        shifted_fit_energy_norm_squared_hartree=shifted_fit_norm,
        residual_energy_norm_squared_hartree=residual_norm,
        shifted_pythagorean_error_hartree=pythagorean_error,
        target_polarization_energy_hartree=target_energy,
        fitted_polarization_energy_hartree=fitted_energy,
        polarization_energy_error_hartree=fitted_energy - target_energy,
    )
