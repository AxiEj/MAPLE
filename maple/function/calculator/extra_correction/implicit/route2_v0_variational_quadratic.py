"""No-training quadratic variational-response kernel for Route-2 V0-Q.

The frozen MACE-POLAR checkpoint supplies only a zero-field density ``c0``.
It does *not* supply the field-dependent electronic response used here.  A
separately declared, symmetric coefficient-space curvature ``H`` instead
defines the induced-density cost

``0.5 * dc.T @ H @ dc``.

At fixed geometry, a reciprocal GTO Galerkin continuum supplies ``P`` and the
joint stationary state solves

``argmin_{u.T @ dc = 0} 0.5 dc.T H dc + 0.5 (c0 + dc).T P (c0 + dc)``.

This is deliberately a research kernel, not a public Route-2 profile or a
source of default physical parameters.  In particular, a caller must freeze
``H`` from an independently documented physical model before any chemistry is
scored.  The implementation rejects a non-symmetric curvature or a loss of
positive curvature in the charge-conserving subspace; it never repairs either
condition by mixing, damping, eigenvalue clipping, or an error fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gto_density import MACE_POLAR_DENSITY_SIGMA_ANGSTROM
from .gto_galerkin import (
    AtomCenteredL1GTOBasis,
    FixedCavityGTOGalerkinOperator,
    GTOGalerkinSnapshot,
)


V0_VARIATIONAL_QUADRATIC_CONSTRUCTION = "route2-v0-variational-quadratic-kkt-v1"


def _finite_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
    ):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    return array


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    result = np.array(_finite_array(values, name=name, shape=shape), copy=True)
    result.setflags(write=False)
    return result


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _relative_matrix_scale(matrix: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(matrix, ord=2)))


def _validated_symmetric_curvature(
    values: np.ndarray,
    *,
    coefficient_count: int,
    relative_tolerance: float,
) -> tuple[np.ndarray, float]:
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("Curvature symmetry tolerance must be finite and positive.")
    matrix = _finite_array(
        values,
        name="Electronic curvature",
        shape=(coefficient_count, coefficient_count),
    )
    antisymmetric_norm = float(np.linalg.norm(0.5 * (matrix - matrix.T), ord=2))
    if antisymmetric_norm > relative_tolerance * _relative_matrix_scale(matrix):
        raise ValueError(
            "Electronic curvature must be symmetric in the declared "
            "coefficient/dual pairing."
        )
    # Remove only roundoff after the material-asymmetry gate above has passed.
    return 0.5 * (matrix + matrix.T), antisymmetric_norm


def _neutral_basis(charge_constraint: np.ndarray) -> np.ndarray:
    vector = _finite_array(charge_constraint, name="Charge constraint")
    if vector.ndim != 1 or float(np.linalg.norm(vector)) <= 0.0:
        raise ValueError("Charge constraint must be one nonzero finite vector.")
    _, _, right_vectors = np.linalg.svd(vector[None, :], full_matrices=True)
    return right_vectors[1:, :].T.copy()


def _positive_neutral_curvature(
    matrix: np.ndarray,
    neutral_basis: np.ndarray,
    *,
    name: str,
    relative_tolerance: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    projected = neutral_basis.T @ matrix @ neutral_basis
    projected = 0.5 * (projected + projected.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    minimum = float(np.min(eigenvalues))
    threshold = relative_tolerance * _relative_matrix_scale(projected)
    if minimum <= threshold:
        raise RuntimeError(
            f"{name} is not positive definite in the charge-conserving "
            "subspace; the variational state is rejected."
        )
    return minimum, threshold, eigenvalues, eigenvectors


def _coefficient_block(
    values: np.ndarray,
    *,
    operator: FixedCavityGTOGalerkinOperator,
    name: str,
) -> np.ndarray:
    return operator.basis.validate_coefficients(
        values,
        atom_count=operator.atom_count,
        name=name,
    )


def embed_mace_polar_l1_density_in_single_radial_gto(
    density_coefficients: np.ndarray,
    basis: AtomCenteredL1GTOBasis,
) -> np.ndarray:
    """Embed the native MACE ``l<=1`` density without fitting or projection.

    MACE-POLAR's current density head has one Gaussian radial channel with
    width :data:`MACE_POLAR_DENSITY_SIGMA_ANGSTROM`.  A multi-radial GTO basis
    would require a density decomposition and is therefore rejected rather
    than invented here.
    """

    density = _finite_array(
        density_coefficients,
        name="MACE-POLAR l<=1 density",
    )
    if density.ndim != 2 or density.shape[0] == 0 or density.shape[1] != 4:
        raise ValueError(
            "MACE-POLAR l<=1 density must be finite with shape (n_atoms, 4)."
        )
    if basis.radial_count != 1:
        raise ValueError(
            "An unmodified MACE-POLAR density can be embedded only in a "
            "single-radial GTO basis."
        )
    if not math.isclose(
        basis.sigmas_angstrom[0],
        MACE_POLAR_DENSITY_SIGMA_ANGSTROM,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "The single GTO radial width must equal the frozen MACE-POLAR "
            "density width."
        )
    result = np.array(density[:, None, :], dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0VariationalQuadraticState:
    """One fixed-geometry joint stationary state of the V0-Q functional.

    All matrices use the declared GTO coefficient/dual pairing.  Thus the
    quantity ``0.5 * dc.T @ electronic_curvature @ dc`` is in Hartree even
    though monopole and dipole coordinates have different native dimensions.
    """

    frozen_density_coefficients: np.ndarray
    induced_density_coefficients: np.ndarray
    stationary_density_coefficients: np.ndarray
    external_coefficient_dual_hartree: np.ndarray
    charge_constraint_vector: np.ndarray
    electronic_curvature_coefficient_dual: np.ndarray
    joint_hessian_coefficient_dual: np.ndarray
    joint_external_dual_response: np.ndarray
    continuum_snapshot: GTOGalerkinSnapshot
    target_total_charge_e: float
    frozen_total_charge_e: float
    stationary_total_charge_e: float
    lagrange_multiplier_hartree_per_e: float
    electronic_induction_energy_hartree: float
    continuum_polarization_energy_hartree: float
    external_work_hartree: float
    solute_continuum_energy_hartree: float
    stationary_total_energy_hartree: float
    electronic_minimum_neutral_curvature: float
    joint_minimum_neutral_curvature: float
    curvature_stability_threshold: float
    electronic_curvature_antisymmetry_norm: float
    stationarity_residual_inf: float
    charge_constraint_residual_e: float
    construction: str = V0_VARIATIONAL_QUADRATIC_CONSTRUCTION

    def __post_init__(self) -> None:
        frozen = _finite_array(
            self.frozen_density_coefficients,
            name="Frozen density coefficients",
        )
        if frozen.ndim != 3 or frozen.shape[0] == 0 or frozen.shape[2] != 4:
            raise ValueError(
                "Frozen density coefficients must be finite with shape "
                "(n_atoms, n_radial, 4)."
            )
        shape = frozen.shape
        coefficient_count = int(np.prod(shape))
        object.__setattr__(
            self,
            "frozen_density_coefficients",
            _immutable_array(frozen, name="Frozen density coefficients"),
        )
        for name in (
            "induced_density_coefficients",
            "stationary_density_coefficients",
            "external_coefficient_dual_hartree",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), name=name, shape=shape),
            )
        for name in (
            "charge_constraint_vector",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(
                    getattr(self, name),
                    name=name,
                    shape=(coefficient_count,),
                ),
            )
        for name in (
            "electronic_curvature_coefficient_dual",
            "joint_hessian_coefficient_dual",
            "joint_external_dual_response",
        ):
            matrix = _immutable_array(
                getattr(self, name),
                name=name,
                shape=(coefficient_count, coefficient_count),
            )
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"{name} must be symmetric.")
            object.__setattr__(self, name, matrix)
        for name in (
            "target_total_charge_e",
            "frozen_total_charge_e",
            "stationary_total_charge_e",
            "lagrange_multiplier_hartree_per_e",
            "electronic_induction_energy_hartree",
            "continuum_polarization_energy_hartree",
            "external_work_hartree",
            "solute_continuum_energy_hartree",
            "stationary_total_energy_hartree",
            "electronic_minimum_neutral_curvature",
            "joint_minimum_neutral_curvature",
            "curvature_stability_threshold",
            "electronic_curvature_antisymmetry_norm",
            "stationarity_residual_inf",
            "charge_constraint_residual_e",
        ):
            value = _finite_scalar(getattr(self, name), name=name)
            if name in (
                "curvature_stability_threshold",
                "electronic_curvature_antisymmetry_norm",
                "stationarity_residual_inf",
                "charge_constraint_residual_e",
            ) and value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        if self.construction != V0_VARIATIONAL_QUADRATIC_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 quadratic construction.")
        if not np.allclose(
            self.stationary_density_coefficients,
            self.frozen_density_coefficients + self.induced_density_coefficients,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Stationary density must equal frozen plus induced density.")
        energy_tolerance = max(
            1.0e-12,
            1.0e-10 * abs(self.continuum_polarization_energy_hartree),
        )
        if abs(
            self.continuum_polarization_energy_hartree
            - self.continuum_snapshot.polarization_energy_hartree
        ) > energy_tolerance:
            raise ValueError(
                "Continuum energy must equal the matching GTO Galerkin snapshot."
            )
        if abs(
            self.solute_continuum_energy_hartree
            - (
                self.electronic_induction_energy_hartree
                + self.continuum_polarization_energy_hartree
            )
        ) > energy_tolerance:
            raise ValueError("Solute/continuum energy ledger is inconsistent.")
        if abs(
            self.stationary_total_energy_hartree
            - (self.solute_continuum_energy_hartree + self.external_work_hartree)
        ) > energy_tolerance:
            raise ValueError("Stationary total energy ledger is inconsistent.")
        response_eigenvalue = float(
            np.max(np.linalg.eigvalsh(self.joint_external_dual_response))
        )
        if response_eigenvalue > 1.0e-10:
            raise ValueError(
                "Joint external-field response must be nonpositive in the "
                "energy pairing."
            )


def solve_route2_v0_variational_quadratic(
    *,
    frozen_density_coefficients: np.ndarray,
    operator: FixedCavityGTOGalerkinOperator,
    electronic_curvature_coefficient_dual: np.ndarray,
    target_total_charge_e: float = 0.0,
    external_coefficient_dual_hartree: np.ndarray | None = None,
    total_charge_tolerance_e: float = 1.0e-10,
    curvature_symmetry_relative_tolerance: float = 1.0e-12,
    stability_relative_tolerance: float = 1.0e-10,
    stationarity_relative_tolerance: float = 1.0e-10,
) -> Route2V0VariationalQuadraticState:
    """Solve the charge-conserving V0-Q stationary system at fixed geometry.

    ``electronic_curvature_coefficient_dual`` must describe the *gas-phase*
    induced-density energy in exactly the same GTO coefficient/dual basis as
    ``operator``.  It is not inferred from MACE's rejected learned field
    update, and no default curvature is provided.  With an optional external
    coefficient dual ``f``, the scalar minimized is

    ``0.5 dc.T H dc + 0.5 (c0 + dc).T P (c0 + dc) + (c0 + dc).T f``.

    The returned response matrix is ``dc / df`` after continuum feedback.  It
    is generated from the inverse of the symmetric projected Hessian and is
    therefore reciprocal and nonpositive by construction.
    """

    if not isinstance(operator, FixedCavityGTOGalerkinOperator):
        raise TypeError("V0-Q requires a fixed-cavity reciprocal GTO operator.")
    if (
        not math.isfinite(total_charge_tolerance_e)
        or total_charge_tolerance_e <= 0.0
    ):
        raise ValueError("Total-charge tolerance must be finite and positive.")
    if (
        not math.isfinite(stability_relative_tolerance)
        or stability_relative_tolerance <= 0.0
    ):
        raise ValueError("Stability tolerance must be finite and positive.")
    if (
        not math.isfinite(stationarity_relative_tolerance)
        or stationarity_relative_tolerance <= 0.0
    ):
        raise ValueError("Stationarity tolerance must be finite and positive.")

    frozen_block = _coefficient_block(
        frozen_density_coefficients,
        operator=operator,
        name="Frozen density coefficients",
    )
    frozen_vector = np.asarray(frozen_block).reshape(-1)
    coefficient_count = operator.coefficient_count
    curvature, curvature_antisymmetry_norm = _validated_symmetric_curvature(
        electronic_curvature_coefficient_dual,
        coefficient_count=coefficient_count,
        relative_tolerance=curvature_symmetry_relative_tolerance,
    )
    continuum_matrix = operator.galerkin_matrix_hartree
    joint_hessian = curvature + continuum_matrix
    joint_hessian = 0.5 * (joint_hessian + joint_hessian.T)

    constraints = operator.basis.molecular_charge_dipole_constraints(
        operator.atom_positions_angstrom
    )
    charge_constraint = constraints[0]
    target_charge = _finite_scalar(target_total_charge_e, name="Target total charge")
    frozen_charge = float(np.dot(charge_constraint, frozen_vector))
    if abs(frozen_charge - target_charge) > total_charge_tolerance_e:
        raise ValueError(
            "Frozen density violates its declared total-charge constraint "
            f"(observed={frozen_charge:.16e} e, target={target_charge:.16e} e)."
        )
    neutral_basis = _neutral_basis(charge_constraint)
    (
        electronic_minimum,
        electronic_threshold,
        _,
        _,
    ) = _positive_neutral_curvature(
        curvature,
        neutral_basis,
        name="Electronic curvature",
        relative_tolerance=stability_relative_tolerance,
    )
    (
        joint_minimum,
        joint_threshold,
        joint_eigenvalues,
        joint_eigenvectors,
    ) = _positive_neutral_curvature(
        joint_hessian,
        neutral_basis,
        name="Joint electronic-continuum curvature",
        relative_tolerance=stability_relative_tolerance,
    )

    if external_coefficient_dual_hartree is None:
        external_block = np.zeros_like(frozen_block)
    else:
        external_block = _coefficient_block(
            external_coefficient_dual_hartree,
            operator=operator,
            name="External coefficient dual",
        )
    external_vector = np.asarray(external_block).reshape(-1)
    kkt_matrix = np.empty((coefficient_count + 1, coefficient_count + 1))
    kkt_matrix[:-1, :-1] = joint_hessian
    kkt_matrix[:-1, -1] = charge_constraint
    kkt_matrix[-1, :-1] = charge_constraint
    kkt_matrix[-1, -1] = 0.0
    right_hand_side = np.concatenate(
        (-continuum_matrix @ frozen_vector - external_vector, [0.0])
    )
    try:
        solution = np.linalg.solve(kkt_matrix, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("V0-Q KKT system is singular.") from error
    induced_vector = solution[:-1]
    lagrange_multiplier = float(solution[-1])
    stationary_vector = frozen_vector + induced_vector
    stationarity = (
        joint_hessian @ induced_vector
        + continuum_matrix @ frozen_vector
        + external_vector
        + lagrange_multiplier * charge_constraint
    )
    stationarity_residual = float(np.linalg.norm(stationarity, ord=np.inf))
    residual_scale = max(
        1.0,
        float(np.linalg.norm(right_hand_side[:-1], ord=np.inf)),
        float(np.linalg.norm(joint_hessian @ induced_vector, ord=np.inf)),
    )
    if stationarity_residual > stationarity_relative_tolerance * residual_scale:
        raise RuntimeError("V0-Q KKT stationarity residual exceeds its tolerance.")
    charge_residual = abs(float(np.dot(charge_constraint, induced_vector)))
    if charge_residual > total_charge_tolerance_e:
        raise RuntimeError("V0-Q induced density violates charge conservation.")

    # This spectral form is the inverse of the positive projected Hessian.  It
    # is a consequence of the scalar functional, not a symmetrization of a
    # learned response Jacobian.
    projected_modes = neutral_basis @ joint_eigenvectors
    joint_response = -(
        (projected_modes / joint_eigenvalues[None, :]) @ projected_modes.T
    )
    joint_response = 0.5 * (joint_response + joint_response.T)

    stationary_block = operator.basis.unflatten(
        stationary_vector,
        atom_count=operator.atom_count,
        name="Stationary density coefficients",
    )
    induced_block = operator.basis.unflatten(
        induced_vector,
        atom_count=operator.atom_count,
        name="Induced density coefficients",
    )
    snapshot = operator.snapshot(stationary_block)
    electronic_energy = 0.5 * float(induced_vector @ curvature @ induced_vector)
    continuum_energy = float(snapshot.polarization_energy_hartree)
    external_work = float(stationary_vector @ external_vector)
    solute_continuum_energy = electronic_energy + continuum_energy
    stationary_total_energy = solute_continuum_energy + external_work
    stationary_charge = float(np.dot(charge_constraint, stationary_vector))

    return Route2V0VariationalQuadraticState(
        frozen_density_coefficients=frozen_block,
        induced_density_coefficients=induced_block,
        stationary_density_coefficients=stationary_block,
        external_coefficient_dual_hartree=external_block,
        charge_constraint_vector=charge_constraint,
        electronic_curvature_coefficient_dual=curvature,
        joint_hessian_coefficient_dual=joint_hessian,
        joint_external_dual_response=joint_response,
        continuum_snapshot=snapshot,
        target_total_charge_e=target_charge,
        frozen_total_charge_e=frozen_charge,
        stationary_total_charge_e=stationary_charge,
        lagrange_multiplier_hartree_per_e=lagrange_multiplier,
        electronic_induction_energy_hartree=electronic_energy,
        continuum_polarization_energy_hartree=continuum_energy,
        external_work_hartree=external_work,
        solute_continuum_energy_hartree=solute_continuum_energy,
        stationary_total_energy_hartree=stationary_total_energy,
        electronic_minimum_neutral_curvature=electronic_minimum,
        joint_minimum_neutral_curvature=joint_minimum,
        curvature_stability_threshold=max(electronic_threshold, joint_threshold),
        electronic_curvature_antisymmetry_norm=curvature_antisymmetry_norm,
        stationarity_residual_inf=stationarity_residual,
        charge_constraint_residual_e=charge_residual,
    )


__all__ = [
    "Route2V0VariationalQuadraticState",
    "V0_VARIATIONAL_QUADRATIC_CONSTRUCTION",
    "embed_mace_polar_l1_density_in_single_radial_gto",
    "solve_route2_v0_variational_quadratic",
]
