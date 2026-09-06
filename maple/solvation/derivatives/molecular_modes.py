"""Internal molecular modes with explicit Richardson uncertainty intervals.

This module is intentionally independent of the legacy frequency projection.
It mass-weights the Cartesian Hessian before removing rigid motion, as required
for a physically meaningful generalized vibrational eigenproblem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _immutable_array(
    value: object, *, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array shape {array.shape} does not match required {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("molecular-mode inputs must contain only finite values.")
    copy = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64)
    copy = copy.reshape(array.shape)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, slots=True)
class MolecularModeAnalysis:
    """Mass-weighted internal eigensystem and a conservative error interval."""

    rigid_rank: int
    internal_dimension: int
    mass_weighted_hessian_eV_per_A2_amu: np.ndarray
    richardson_error_eV_per_A2: np.ndarray
    symmetric_error_envelope_eV_per_A2: np.ndarray
    rigid_basis_mass_weighted: np.ndarray
    internal_basis_mass_weighted: np.ndarray
    eigenvalues_eV_per_A2_amu: np.ndarray
    modes_cartesian: np.ndarray
    uncertainty_eV_per_A2_amu: float
    uncertainty_label: str
    statuses: tuple[str, ...]

    @property
    def resolved_negative_count(self) -> int:
        return self.statuses.count("negative")

    @property
    def resolved_positive_count(self) -> int:
        return self.statuses.count("positive")

    @property
    def uncertain_count(self) -> int:
        return self.statuses.count("uncertain")

    @property
    def eigenvalue_intervals_eV_per_A2_amu(
        self,
    ) -> tuple[tuple[float, float], ...]:
        epsilon = self.uncertainty_eV_per_A2_amu
        return tuple(
            (float(value - epsilon), float(value + epsilon))
            for value in self.eigenvalues_eV_per_A2_amu
        )

    @property
    def is_resolved_minimum(self) -> bool:
        return (
            self.internal_dimension > 0
            and self.resolved_positive_count == self.internal_dimension
        )

    @property
    def is_resolved_index_one(self) -> bool:
        return (
            self.resolved_negative_count == 1
            and self.resolved_positive_count == self.internal_dimension - 1
            and self.uncertain_count == 0
        )


def _mass_weighted_rigid_basis(
    positions_angstrom: np.ndarray,
    masses_amu: np.ndarray,
) -> np.ndarray:
    center_of_mass = np.average(positions_angstrom, axis=0, weights=masses_amu)
    centered = positions_angstrom - center_of_mass
    root_masses = np.sqrt(masses_amu)
    columns = []
    for axis in np.eye(3):
        columns.append((root_masses[:, None] * axis).reshape(-1))
    for axis in np.eye(3):
        # Infinitesimal rotation delta-r = omega cross r.
        columns.append((root_masses[:, None] * np.cross(axis, centered)).reshape(-1))
    return np.column_stack(columns)


def analyze_molecular_modes(
    *,
    positions_angstrom: object,
    masses_amu: object,
    hessian_eV_per_A2: object,
    richardson_error_eV_per_A2: object,
) -> MolecularModeAnalysis:
    """Diagonalize the internal mass-weighted Hessian with mode intervals.

    The scalar ``uncertainty_eV_per_A2_amu`` is
    ``||M^-1/2 B M^-1/2||_2`` with ``B=(R+R.T)/2``.  It is deliberately not
    projected before taking the norm: signed projection can cancel entries of
    the componentwise Richardson envelope and understate admissible error.
    """

    positions = _immutable_array(positions_angstrom)
    masses = _immutable_array(masses_amu)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_angstrom must have shape (N,3).")
    if masses.shape != (len(positions),) or np.any(masses <= 0.0):
        raise ValueError("masses_amu must be positive with shape (N,).")
    dimension = positions.size
    hessian = _immutable_array(hessian_eV_per_A2, shape=(dimension, dimension))
    errors = _immutable_array(richardson_error_eV_per_A2, shape=(dimension, dimension))
    if np.any(errors < 0.0):
        raise ValueError("Richardson component error estimates must be non-negative.")

    inv_root_mass = np.repeat(1.0 / np.sqrt(masses), 3)
    symmetric_hessian = 0.5 * (hessian + hessian.T)
    mass_weighted_hessian = (
        inv_root_mass[:, None] * symmetric_hessian * inv_root_mass[None, :]
    )

    rigid_candidates = _mass_weighted_rigid_basis(positions, masses)
    u, singular_values, _ = np.linalg.svd(rigid_candidates, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(rigid_candidates.shape) * largest
    rigid_rank = int(np.count_nonzero(singular_values > tolerance))
    rigid_basis = u[:, :rigid_rank]
    internal_basis = u[:, rigid_rank:]

    internal_hessian = internal_basis.T @ mass_weighted_hessian @ internal_basis
    internal_hessian = 0.5 * (internal_hessian + internal_hessian.T)
    eigenvalues, eigenvectors_internal = np.linalg.eigh(internal_hessian)
    modes_mass_weighted = internal_basis @ eigenvectors_internal
    modes_cartesian = (inv_root_mass[:, None] * modes_mass_weighted).T

    envelope = 0.5 * (errors + errors.T)
    mass_weighted_envelope = inv_root_mass[:, None] * envelope * inv_root_mass[None, :]
    uncertainty = float(np.linalg.norm(mass_weighted_envelope, ord=2))
    statuses = tuple(
        (
            "negative"
            if eigenvalue + uncertainty < 0.0
            else "positive" if eigenvalue - uncertainty > 0.0 else "uncertain"
        )
        for eigenvalue in eigenvalues
    )

    return MolecularModeAnalysis(
        rigid_rank=rigid_rank,
        internal_dimension=dimension - rigid_rank,
        mass_weighted_hessian_eV_per_A2_amu=_immutable_array(mass_weighted_hessian),
        richardson_error_eV_per_A2=errors,
        symmetric_error_envelope_eV_per_A2=_immutable_array(envelope),
        rigid_basis_mass_weighted=_immutable_array(rigid_basis),
        internal_basis_mass_weighted=_immutable_array(internal_basis),
        eigenvalues_eV_per_A2_amu=_immutable_array(eigenvalues),
        modes_cartesian=_immutable_array(modes_cartesian),
        uncertainty_eV_per_A2_amu=uncertainty,
        uncertainty_label="Richardson-derived numerical uncertainty estimate",
        statuses=statuses,
    )


def analyze_hessian_evaluation(atoms, evaluation) -> MolecularModeAnalysis:
    """Analyze one concrete immutable Richardson Hessian evaluation."""

    from .scalar_finite_difference import RichardsonScalarHessianEvaluation

    if not isinstance(evaluation, RichardsonScalarHessianEvaluation):
        raise TypeError("evaluation must be RichardsonScalarHessianEvaluation.")
    return analyze_molecular_modes(
        positions_angstrom=atoms.get_positions(),
        masses_amu=atoms.get_masses(),
        hessian_eV_per_A2=evaluation.hessian_eV_per_A2,
        richardson_error_eV_per_A2=evaluation.error_estimates_eV_per_A2,
    )


def hessian_numerical_diagnostics(evaluation) -> dict[str, object]:
    """Serialize ordered stencil evidence without discarding accepted events."""
    from .scalar_finite_difference import RichardsonScalarHessianEvaluation

    if not isinstance(evaluation, RichardsonScalarHessianEvaluation):
        raise TypeError("evaluation must be RichardsonScalarHessianEvaluation.")
    return {
        "evaluation_sha256": evaluation.evaluation_sha256,
        "derivative_policy_sha256": evaluation.derivative_policy_sha256,
        "coarse_step_angstrom": evaluation.coarse_step_angstrom,
        "fine_step_angstrom": evaluation.fine_step_angstrom,
        "maximum_richardson_error_eV_per_A2": evaluation.maximum_error_estimate_eV_per_A2,
        "maximum_raw_antisymmetry_eV_per_A2": evaluation.maximum_antisymmetry_eV_per_A2,
        "topology_observation_coverage": evaluation.topology_observation_coverage,
        "topology_guard_status": evaluation.topology_guard_status,
        "stencil_order_per_column": ["+h", "-h", "+h/2", "-h/2"],
        "displaced_force_sha256": list(evaluation.displaced_force_sha256),
        "displaced_topology_ids": list(evaluation.displaced_topology_ids),
        "displaced_topology_changed": list(evaluation.displaced_topology_changed),
        "topology_change_count": sum(evaluation.displaced_topology_changed),
        "energy_force_discrepancies_eV_per_A": list(
            evaluation.energy_force_discrepancies_eV_per_A
        ),
        "maximum_energy_force_discrepancy_eV_per_A": max(
            evaluation.energy_force_discrepancies_eV_per_A, default=0.0
        ),
    }


def hessian_numerical_summary(evaluation) -> str:
    """Compact log projection; complete ordered diagnostics remain available."""
    record = hessian_numerical_diagnostics(evaluation)
    return (
        f"Hessian evaluation SHA256: {record['evaluation_sha256']}\n"
        f"Stencil h/h2: {record['coarse_step_angstrom']:.6g}/"
        f"{record['fine_step_angstrom']:.6g} A; maximum Richardson: "
        f"{record['maximum_richardson_error_eV_per_A2']:.8e} eV/A^2; "
        f"raw antisymmetry: {record['maximum_raw_antisymmetry_eV_per_A2']:.8e} eV/A^2\n"
        f"Topology coverage: {record['topology_observation_coverage']}; "
        f"changed samples: {record['topology_change_count']}; "
        f"maximum endpoint work discrepancy: "
        f"{record['maximum_energy_force_discrepancy_eV_per_A']:.8e} eV/A"
    )


__all__ = [
    "MolecularModeAnalysis",
    "analyze_hessian_evaluation",
    "analyze_molecular_modes",
    "hessian_numerical_diagnostics",
    "hessian_numerical_summary",
]
