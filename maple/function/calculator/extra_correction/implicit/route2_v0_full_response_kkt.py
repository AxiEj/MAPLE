"""Constrained common-scalar KKT solve for a full V0 response kernel.

For a frozen atomic independent-particle response completion ``C`` and its
exact GTO surface map ``B``, this module minimizes only the declared scalar

``G(x) = 0.5*x.T*C_plus*x + 0.5*(v0 + B*x).T*Q*(v0 + B*x) + x.T*f``

subject to the exact kernel-support rows ``N*x = 0``.  ``Q`` is the
energy-conjugate continuum response and ``v0`` is an optional *separately
provenanced* permanent surface potential.  The resulting KKT system is
symmetric.  The module neither invents a permanent density nor accepts a
non-reciprocal continuum, a shifted response spectrum, a fitted correction,
or a post-solve ledger.

It is a fixed-geometry full-response structural primitive.  A physical V0
calculation still needs a smooth coordinate-dependent continuum/cavity,
permanent stationary electronic source, nonpolar scalar, and envelope-force
proof.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
)
from .route2_v0_atomic_independent_particle_surface import (
    AtomicIndependentParticleSurfaceCoupling,
    BOHR_ANGSTROM,
)
from .route2_v0_response_kernel import (
    Route2V0MolecularMomentResponseKernelCompletion,
    Route2V0ResponseKernelCompletion,
)


V0_FULL_RESPONSE_KKT_CONSTRUCTION = "route2-v0-full-response-kkt-v1"


def _immutable_array(
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
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite and positive.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _infinity_norm(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _symmetric_matrix(
    values: np.ndarray,
    *,
    name: str,
    relative_tolerance: float,
) -> tuple[np.ndarray, float]:
    matrix = _immutable_array(values, name=name)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix.")
    antisymmetry = float(np.linalg.norm(0.5 * (matrix - matrix.T), ord=2))
    if antisymmetry > relative_tolerance * _matrix_scale(matrix):
        raise RuntimeError(f"{name} is not reciprocal in its declared pairing.")
    return 0.5 * (matrix + matrix.T), antisymmetry


def _support_basis(
    completion: (
        Route2V0ResponseKernelCompletion
        | Route2V0MolecularMomentResponseKernelCompletion
    ),
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    projector, projector_antisymmetry = _symmetric_matrix(
        completion.response_support_projector,
        name="Response support projector",
        relative_tolerance=relative_tolerance,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    support = eigenvalues > 0.5
    if not np.any(support):
        raise RuntimeError("Response completion has no declared support.")
    if np.any(eigenvalues[support] < 1.0 - relative_tolerance) or np.any(
        eigenvalues[~support] > relative_tolerance
    ):
        raise RuntimeError("Response support projector is not idempotent.")
    return eigenvectors[:, support], projector, projector_antisymmetry


def _validate_completion_matches_coupling(
    coupling: AtomicIndependentParticleSurfaceCoupling,
    completion: (
        Route2V0ResponseKernelCompletion
        | Route2V0MolecularMomentResponseKernelCompletion
    ),
    *,
    relative_tolerance: float,
) -> None:
    if not isinstance(
        completion,
        (
            Route2V0ResponseKernelCompletion,
            Route2V0MolecularMomentResponseKernelCompletion,
        ),
    ):
        raise TypeError("completion must be a V0 response-kernel completion.")
    baseline = coupling.baseline
    expected_covariance = baseline.baseline_response_covariance_coefficient_dual
    expected_map = baseline.atom_dipole_map_coefficient_to_ebohr
    if (
        completion.response_covariance_coefficient_dual.shape
        != (coupling.coefficient_count, coupling.coefficient_count)
        or completion.electronic_curvature_coefficient_dual.shape
        != (coupling.coefficient_count, coupling.coefficient_count)
        or not np.allclose(
            completion.baseline_response_covariance_coefficient_dual,
            expected_covariance,
            rtol=relative_tolerance,
            atol=relative_tolerance * _matrix_scale(expected_covariance),
        )
        or not np.allclose(
            completion.atom_dipole_map_coefficient_to_ebohr,
            expected_map,
            rtol=relative_tolerance,
            atol=relative_tolerance * _matrix_scale(expected_map),
        )
        or completion.charge_constraint_vector is not None
    ):
        raise ValueError(
            "The response completion does not match the intrinsic-neutral "
            "atomic independent-particle coefficient space."
        )


def validate_atomic_independent_particle_continuum(
    coupling: AtomicIndependentParticleSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
) -> None:
    """Require a fixed continuum to share the exact source geometry and surface."""

    if (
        getattr(continuum, "contract_version", None)
        != EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported external-MEP continuum-response contract.")
    if not bool(getattr(continuum, "energy_response_is_reciprocal", False)):
        raise ValueError("Full-response KKT requires a reciprocal continuum.")
    if int(getattr(continuum, "atom_count", -1)) != coupling.atom_count:
        raise ValueError("Continuum atom count does not match the atomic-IP source.")
    atomic_numbers = np.asarray(getattr(continuum, "atomic_numbers", None))
    if atomic_numbers.shape != coupling.baseline.atomic_numbers.shape or not np.array_equal(
        atomic_numbers,
        coupling.baseline.atomic_numbers,
    ):
        raise ValueError("Continuum atomic numbers do not match the atomic-IP source.")
    positions_bohr = np.asarray(continuum.reference_positions_bohr, dtype=float)
    points = np.asarray(continuum.surface_points_bohr, dtype=float)
    if (
        positions_bohr.shape != coupling.atom_positions_angstrom.shape
        or points.shape != coupling.surface_points_bohr.shape
        or not np.allclose(
            positions_bohr * BOHR_ANGSTROM,
            coupling.atom_positions_angstrom,
            rtol=0.0,
            atol=1.0e-12,
        )
        or not np.allclose(points, coupling.surface_points_bohr, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Continuum geometry or surface points do not match atomic-IP.")


@dataclass(frozen=True)
class Route2V0FullResponseKKTState:
    """One fixed-geometry constrained stationary full-response state."""

    induced_coefficients: np.ndarray
    support_constraint_multipliers: np.ndarray
    external_coefficient_dual_hartree: np.ndarray
    permanent_surface_potential_hartree_per_e: np.ndarray
    induced_surface_potential_hartree_per_e: np.ndarray
    total_surface_potential_hartree_per_e: np.ndarray
    permanent_surface_charge_e: np.ndarray
    induced_surface_charge_e: np.ndarray
    total_surface_charge_e: np.ndarray
    reaction_coefficient_dual_hartree: np.ndarray
    electronic_curvature_coefficient_dual: np.ndarray
    continuum_curvature_coefficient_dual: np.ndarray
    joint_curvature_coefficient_dual: np.ndarray
    external_response_coefficient_dual: np.ndarray
    electronic_induction_energy_hartree: float
    continuum_polarization_energy_hartree: float
    external_work_hartree: float
    stationary_total_energy_hartree: float
    source_duality_error_hartree: float
    continuum_linearity_error_e: float
    continuum_reciprocity_antisymmetry_hartree: float
    continuum_reciprocity_tolerance_hartree: float
    continuum_support_maximum_eigenvalue_hartree: float
    continuum_passivity_threshold_hartree: float
    kkt_residual_inf_hartree: float
    support_constraint_residual_inf: float
    stationarity_tolerance_hartree: float
    support_minimum_curvature_hartree: float
    support_stability_threshold_hartree: float
    support_projector_antisymmetry: float
    construction: str = V0_FULL_RESPONSE_KKT_CONSTRUCTION

    def __post_init__(self) -> None:
        coefficient_count = np.asarray(self.induced_coefficients).size
        if coefficient_count == 0:
            raise ValueError("Full-response KKT coefficients must be nonempty.")
        for name, shape in (
            ("induced_coefficients", (coefficient_count,)),
            ("external_coefficient_dual_hartree", (coefficient_count,)),
            ("reaction_coefficient_dual_hartree", (coefficient_count,)),
            ("electronic_curvature_coefficient_dual", (coefficient_count, coefficient_count)),
            ("continuum_curvature_coefficient_dual", (coefficient_count, coefficient_count)),
            ("joint_curvature_coefficient_dual", (coefficient_count, coefficient_count)),
            ("external_response_coefficient_dual", (coefficient_count, coefficient_count)),
        ):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), name=name, shape=shape))
        external_response, external_response_antisymmetry = _symmetric_matrix(
            self.external_response_coefficient_dual,
            name="External coefficient response",
            relative_tolerance=1.0e-10,
        )
        object.__setattr__(
            self,
            "external_response_coefficient_dual",
            _immutable_array(
                external_response,
                name="external_response_coefficient_dual",
                shape=(coefficient_count, coefficient_count),
            ),
        )
        if external_response_antisymmetry > 1.0e-10 * _matrix_scale(
            external_response
        ):
            raise ValueError("Full-response external response is not reciprocal.")
        surface_count = np.asarray(self.total_surface_potential_hartree_per_e).size
        if surface_count == 0:
            raise ValueError("Full-response KKT surface potential must be nonempty.")
        for name in (
            "permanent_surface_potential_hartree_per_e",
            "induced_surface_potential_hartree_per_e",
            "total_surface_potential_hartree_per_e",
            "permanent_surface_charge_e",
            "induced_surface_charge_e",
            "total_surface_charge_e",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), name=name, shape=(surface_count,)),
            )
        multipliers = np.asarray(self.support_constraint_multipliers, dtype=float)
        if multipliers.ndim != 1 or not np.all(np.isfinite(multipliers)):
            raise ValueError("support_constraint_multipliers must be a finite vector.")
        multipliers = np.array(multipliers, dtype=float, copy=True)
        multipliers.setflags(write=False)
        object.__setattr__(self, "support_constraint_multipliers", multipliers)
        if self.construction != V0_FULL_RESPONSE_KKT_CONSTRUCTION:
            raise ValueError("Unsupported full-response KKT construction.")
        for name in (
            "electronic_induction_energy_hartree",
            "continuum_polarization_energy_hartree",
            "external_work_hartree",
            "stationary_total_energy_hartree",
            "source_duality_error_hartree",
            "continuum_linearity_error_e",
            "continuum_reciprocity_antisymmetry_hartree",
            "continuum_reciprocity_tolerance_hartree",
            "continuum_support_maximum_eigenvalue_hartree",
            "continuum_passivity_threshold_hartree",
            "kkt_residual_inf_hartree",
            "support_constraint_residual_inf",
            "stationarity_tolerance_hartree",
            "support_minimum_curvature_hartree",
            "support_stability_threshold_hartree",
            "support_projector_antisymmetry",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if (
            self.continuum_reciprocity_antisymmetry_hartree
            > self.continuum_reciprocity_tolerance_hartree
        ):
            raise ValueError("Full-response continuum reciprocity is inconsistent.")
        if (
            self.continuum_support_maximum_eigenvalue_hartree
            > self.continuum_passivity_threshold_hartree
        ):
            raise ValueError("Full-response continuum passivity is inconsistent.")
        if self.kkt_residual_inf_hartree > self.stationarity_tolerance_hartree:
            raise ValueError("Full-response KKT stationarity residual is inconsistent.")
        if self.support_minimum_curvature_hartree <= self.support_stability_threshold_hartree:
            raise ValueError("Full-response KKT support curvature is not positive.")
        if np.max(np.linalg.eigvalsh(self.external_response_coefficient_dual)) > 1.0e-10:
            raise ValueError("Full-response external response must be passive.")
        expected_total = (
            self.electronic_induction_energy_hartree
            + self.continuum_polarization_energy_hartree
            + self.external_work_hartree
        )
        tolerance = 1.0e-10 * max(1.0, abs(expected_total))
        if abs(self.stationary_total_energy_hartree - expected_total) > tolerance:
            raise ValueError("Full-response KKT energy ledger is inconsistent.")


def _response_columns(
    coupling: AtomicIndependentParticleSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(
                continuum.apply_energy_conjugate(
                    coupling.surface_operator_hartree_per_e_per_coefficient[:, index]
                ),
                dtype=float,
            )
            for index in range(coupling.coefficient_count)
        ]
    )


def _surface_charge(
    continuum: ExternalMEPCavityResponse,
    potential: np.ndarray,
    *,
    surface_count: int,
) -> np.ndarray:
    charge = _immutable_array(
        continuum.apply_energy_conjugate(potential),
        name="energy_conjugate_surface_charge_e",
        shape=(surface_count,),
    )
    return charge


def solve_route2_v0_full_response_kkt(
    *,
    coupling: AtomicIndependentParticleSurfaceCoupling,
    completion: (
        Route2V0ResponseKernelCompletion
        | Route2V0MolecularMomentResponseKernelCompletion
    ),
    continuum: ExternalMEPCavityResponse,
    permanent_surface_potential_hartree_per_e: np.ndarray | None = None,
    external_coefficient_dual_hartree: np.ndarray | None = None,
    continuum_reciprocity_relative_tolerance: float = 1.0e-10,
    stability_relative_tolerance: float = 1.0e-10,
    stationarity_relative_tolerance: float = 1.0e-10,
) -> Route2V0FullResponseKKTState:
    """Solve the support-constrained stationary V0 full-response scalar.

    ``permanent_surface_potential_hartree_per_e`` is intentionally a surface
    input rather than a density model.  Passing it does not certify its source;
    it merely makes the exact cross term available once a permanent stationary
    source has independent provenance.
    """

    if not isinstance(coupling, AtomicIndependentParticleSurfaceCoupling):
        raise TypeError("coupling must be an atomic independent-particle surface map.")
    for name, value in (
        ("continuum_reciprocity_relative_tolerance", continuum_reciprocity_relative_tolerance),
        ("stability_relative_tolerance", stability_relative_tolerance),
        ("stationarity_relative_tolerance", stationarity_relative_tolerance),
    ):
        _finite_positive(value, name=name)
    validate_atomic_independent_particle_continuum(coupling, continuum)
    _validate_completion_matches_coupling(
        coupling,
        completion,
        relative_tolerance=1.0e-10,
    )
    coefficient_count = coupling.coefficient_count
    surface_count = coupling.surface_point_count
    permanent_potential = _immutable_array(
        np.zeros(surface_count)
        if permanent_surface_potential_hartree_per_e is None
        else permanent_surface_potential_hartree_per_e,
        name="permanent_surface_potential_hartree_per_e",
        shape=(surface_count,),
    )
    external = _immutable_array(
        np.zeros(coefficient_count)
        if external_coefficient_dual_hartree is None
        else external_coefficient_dual_hartree,
        name="external_coefficient_dual_hartree",
        shape=(coefficient_count,),
    )
    support_basis, _, support_projector_antisymmetry = _support_basis(
        completion,
        relative_tolerance=1.0e-10,
    )
    constraints = _immutable_array(
        completion.response_support_constraints,
        name="response_support_constraints",
    )
    if constraints.ndim != 2 or constraints.shape[1] != coefficient_count:
        raise ValueError("Response support constraints have an invalid coefficient shape.")
    constraint_count = constraints.shape[0]
    if constraint_count:
        rank = np.linalg.matrix_rank(constraints, tol=1.0e-10)
        if rank != constraint_count or constraint_count + support_basis.shape[1] != coefficient_count:
            raise RuntimeError("Response support constraints do not span the kernel null space.")
        if np.linalg.norm(constraints @ support_basis, ord=2) > 1.0e-10:
            raise RuntimeError("Response support constraints are not orthogonal to support.")

    zero_charge = _surface_charge(
        continuum,
        np.zeros(surface_count),
        surface_count=surface_count,
    )
    if _infinity_norm(zero_charge) > continuum_reciprocity_relative_tolerance:
        raise RuntimeError("Continuum response is affine, not linear in this source space.")
    columns = _response_columns(coupling, continuum)
    if columns.shape != (surface_count, coefficient_count) or not np.all(
        np.isfinite(columns)
    ):
        raise RuntimeError("Continuum response columns have an invalid shape.")
    raw_continuum = coupling.surface_operator_hartree_per_e_per_coefficient.T @ columns
    continuum_antisymmetry = float(
        np.linalg.norm(0.5 * (raw_continuum - raw_continuum.T), ord=2)
    )
    continuum_tolerance = continuum_reciprocity_relative_tolerance * _matrix_scale(
        raw_continuum
    )
    if continuum_antisymmetry > continuum_tolerance:
        raise RuntimeError("Continuum is not reciprocal in the atomic-IP source space.")
    # The material reciprocity gate above has passed; this only removes roundoff
    # before a symmetric KKT factorization and is not a response repair.
    continuum_curvature = 0.5 * (raw_continuum + raw_continuum.T)
    continuum_on_support = support_basis.T @ continuum_curvature @ support_basis
    continuum_maximum = float(np.max(np.linalg.eigvalsh(continuum_on_support)))
    continuum_passivity_threshold = (
        continuum_reciprocity_relative_tolerance
        * _matrix_scale(continuum_on_support)
    )
    if continuum_maximum > continuum_passivity_threshold:
        raise RuntimeError("Continuum is not passive in the atomic-IP source space.")
    electronic_curvature, _ = _symmetric_matrix(
        completion.electronic_curvature_coefficient_dual,
        name="Electronic response curvature",
        relative_tolerance=1.0e-10,
    )
    joint_curvature = electronic_curvature + continuum_curvature
    joint_curvature = 0.5 * (joint_curvature + joint_curvature.T)
    reduced_curvature = support_basis.T @ joint_curvature @ support_basis
    reduced_curvature = 0.5 * (reduced_curvature + reduced_curvature.T)
    minimum_curvature = float(np.min(np.linalg.eigvalsh(reduced_curvature)))
    stability_threshold = stability_relative_tolerance * _matrix_scale(reduced_curvature)
    if minimum_curvature <= stability_threshold:
        raise RuntimeError("Full-response KKT curvature is not positive on support.")

    permanent_charge = _surface_charge(
        continuum,
        permanent_potential,
        surface_count=surface_count,
    )
    reaction_from_permanent = coupling.surface_to_coefficient_dual(permanent_charge)
    source_dual = reaction_from_permanent + external
    if constraint_count:
        kkt_matrix = np.block(
            [
                [joint_curvature, constraints.T],
                [constraints, np.zeros((constraint_count, constraint_count))],
            ]
        )
        try:
            solution = np.linalg.solve(
                kkt_matrix,
                -np.concatenate((source_dual, np.zeros(constraint_count))),
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError("Full-response KKT linear system is singular.") from error
        induced = solution[:coefficient_count]
        multipliers = solution[coefficient_count:]
    else:
        try:
            induced = np.linalg.solve(joint_curvature, -source_dual)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("Full-response KKT curvature is singular.") from error
        multipliers = np.empty(0)
    induced = coupling.coefficients(induced, name="induced_coefficients")
    induced_potential = coupling.surface_potential(induced)
    total_potential = permanent_potential + induced_potential
    induced_charge = _surface_charge(
        continuum,
        induced_potential,
        surface_count=surface_count,
    )
    total_charge = _surface_charge(
        continuum,
        total_potential,
        surface_count=surface_count,
    )
    predicted_induced_charge = columns @ induced
    continuum_linearity_error = max(
        _infinity_norm(induced_charge - predicted_induced_charge),
        _infinity_norm(total_charge - permanent_charge - induced_charge),
    )
    linearity_tolerance = continuum_reciprocity_relative_tolerance * max(
        1.0,
        _infinity_norm(total_charge),
        _infinity_norm(permanent_charge),
        _infinity_norm(induced_charge),
    )
    if continuum_linearity_error > linearity_tolerance:
        raise RuntimeError("Continuum response is not linear in the atomic-IP source.")
    reaction_dual = coupling.surface_to_coefficient_dual(total_charge)
    residual = joint_curvature @ induced + source_dual
    if constraint_count:
        residual += constraints.T @ multipliers
        constraint_residual = _infinity_norm(constraints @ induced)
    else:
        constraint_residual = 0.0
    kkt_residual = _infinity_norm(residual)
    stationarity_tolerance = stationarity_relative_tolerance * max(
        1.0,
        _infinity_norm(joint_curvature @ induced),
        _infinity_norm(source_dual),
        _infinity_norm(constraints.T @ multipliers) if constraint_count else 0.0,
    )
    if kkt_residual > stationarity_tolerance or constraint_residual > stationarity_tolerance:
        raise RuntimeError("Full-response KKT stationarity failed.")
    source_duality_error = coupling.source_duality_error(induced, total_charge)
    duality_tolerance = 1.0e-10 * max(
        1.0,
        abs(float(induced_potential @ total_charge)),
        abs(float(induced @ reaction_dual)),
    )
    if source_duality_error > duality_tolerance:
        raise RuntimeError("Atomic-IP source/dual pairing failed.")
    try:
        inverse_reduced = np.linalg.solve(
            reduced_curvature,
            np.eye(reduced_curvature.shape[0]),
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Full-response support curvature is singular.") from error
    external_response = -(support_basis @ inverse_reduced @ support_basis.T)
    external_response = 0.5 * (external_response + external_response.T)
    electronic_energy = 0.5 * float(induced @ electronic_curvature @ induced)
    continuum_energy = 0.5 * float(total_potential @ total_charge)
    external_work = float(induced @ external)
    return Route2V0FullResponseKKTState(
        induced_coefficients=induced,
        support_constraint_multipliers=multipliers,
        external_coefficient_dual_hartree=external,
        permanent_surface_potential_hartree_per_e=permanent_potential,
        induced_surface_potential_hartree_per_e=induced_potential,
        total_surface_potential_hartree_per_e=total_potential,
        permanent_surface_charge_e=permanent_charge,
        induced_surface_charge_e=induced_charge,
        total_surface_charge_e=total_charge,
        reaction_coefficient_dual_hartree=reaction_dual,
        electronic_curvature_coefficient_dual=electronic_curvature,
        continuum_curvature_coefficient_dual=continuum_curvature,
        joint_curvature_coefficient_dual=joint_curvature,
        external_response_coefficient_dual=external_response,
        electronic_induction_energy_hartree=electronic_energy,
        continuum_polarization_energy_hartree=continuum_energy,
        external_work_hartree=external_work,
        stationary_total_energy_hartree=(
            electronic_energy + continuum_energy + external_work
        ),
        source_duality_error_hartree=source_duality_error,
        continuum_linearity_error_e=continuum_linearity_error,
        continuum_reciprocity_antisymmetry_hartree=continuum_antisymmetry,
        continuum_reciprocity_tolerance_hartree=continuum_tolerance,
        continuum_support_maximum_eigenvalue_hartree=continuum_maximum,
        continuum_passivity_threshold_hartree=continuum_passivity_threshold,
        kkt_residual_inf_hartree=kkt_residual,
        support_constraint_residual_inf=constraint_residual,
        stationarity_tolerance_hartree=stationarity_tolerance,
        support_minimum_curvature_hartree=minimum_curvature,
        support_stability_threshold_hartree=stability_threshold,
        support_projector_antisymmetry=support_projector_antisymmetry,
    )


__all__ = [
    "Route2V0FullResponseKKTState",
    "V0_FULL_RESPONSE_KKT_CONSTRUCTION",
    "solve_route2_v0_full_response_kkt",
    "validate_atomic_independent_particle_continuum",
]
