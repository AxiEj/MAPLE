"""No-training full response-kernel completions for Route-2 V0-RK.

The frozen MACE-MDP model supplies a molecular polarizability and an audited
map that partitions a molecular induced dipole across atoms.  It does *not*
identify the high-dimensional density response that controls a near-field
electrostatic potential.  This module therefore accepts an independently
source-bound, positive-semidefinite baseline density-response kernel ``C0``
and replaces only a declared moment covariance with the frozen MACE value.

With ``A`` mapping density coefficients to stacked atomic dipoles, ``W``
mapping a molecular dipole to stacked atomic dipoles, and ``alpha`` the
frozen molecular polarizability, the construction is

``S0 = A @ C0 @ A.T``
``L  = C0 @ A.T @ inv(S0)``
``C  = C0 - L @ S0 @ L.T + L @ (W @ alpha @ W.T) @ L.T``.

Provided ``C0 >= 0`` and ``S0 > 0``, this is a positive-semidefinite response
kernel with ``A @ C @ A.T = W @ alpha @ W.T``.  Its electronic scalar on the
numerical response support is ``0.5 * dc.T @ C^+ @ dc`` and consequently
``dc / df = -C`` before continuum feedback.  The construction is a fixed
low-rank moment constraint, not a fitted response correction.

``complete_route2_v0_response_kernel`` retains the original *stacked atomic*
construction as a structural control.  Its target ``W alpha W.T`` has rank at
most three, so it cannot be treated as a general nonuniform-field response.
``complete_molecular_moment_response_kernel`` instead constrains only the
total molecular dipole and retains the baseline conditional response outside
that three-dimensional moment subspace.

This module deliberately does not source ``C0``, infer a radial basis, load a
checkpoint, create a continuum, or score chemistry.  Those operations remain
separate admission gates because a mathematically valid completion is not by
itself physical response data or solvation accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


V0_RESPONSE_KERNEL_CONSTRUCTION = "route2-v0-response-kernel-completion-v1"
V0_MOLECULAR_MOMENT_RESPONSE_KERNEL_CONSTRUCTION = (
    "route2-v0-molecular-moment-response-kernel-completion-v1"
)


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


def _finite_scalar(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite scalar.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _symmetric_matrix(
    values: np.ndarray,
    *,
    name: str,
    dimension: int,
    relative_tolerance: float,
) -> tuple[np.ndarray, float]:
    matrix = _immutable_array(
        values,
        name=name,
        shape=(dimension, dimension),
    )
    antisymmetry = float(np.linalg.norm(0.5 * (matrix - matrix.T), ord=2))
    if antisymmetry > relative_tolerance * _matrix_scale(matrix):
        raise ValueError(f"{name} must be symmetric in its declared dual pairing.")
    # Only remove floating-point antisymmetry after the material gate passed.
    return 0.5 * (matrix + matrix.T), antisymmetry


def _positive_semidefinite_spectrum(
    matrix: np.ndarray,
    *,
    name: str,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    minimum = float(np.min(eigenvalues))
    threshold = relative_tolerance * _matrix_scale(matrix)
    if minimum < -threshold:
        raise ValueError(f"{name} must be positive semidefinite.")
    return eigenvalues, eigenvectors, minimum, threshold


def _positive_definite(
    matrix: np.ndarray,
    *,
    name: str,
    relative_tolerance: float,
) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(np.min(eigenvalues))
    threshold = relative_tolerance * _matrix_scale(matrix)
    if minimum <= threshold:
        raise ValueError(f"{name} must be positive definite.")
    return minimum, threshold


def _error_tolerance(
    *values: np.ndarray,
    relative_tolerance: float,
) -> float:
    return relative_tolerance * max(_matrix_scale(value) for value in values)


@dataclass(frozen=True)
class Route2V0ResponseKernelCompletion:
    """Certified no-fit replacement of a baseline atom-dipole covariance.

    ``response_covariance_coefficient_dual`` is the positive susceptibility
    ``C`` in ``dc = -C f``.  ``electronic_curvature_coefficient_dual`` is its
    Moore--Penrose inverse on the declared support. A future same-basis KKT
    must impose ``response_support_constraints @ dc = 0``.  When
    ``charge_constraint_vector`` is present, its generic total-charge KKT row
    is separate and the returned rows span only the remaining kernel
    directions.  When it is ``None``, every response mode is intrinsically
    neutral and the returned rows span the complete kernel.  Zero response
    modes are never regularized by an arbitrary curvature.
    """

    baseline_response_covariance_coefficient_dual: np.ndarray
    response_covariance_coefficient_dual: np.ndarray
    electronic_curvature_coefficient_dual: np.ndarray
    response_support_projector: np.ndarray
    response_support_constraints: np.ndarray
    atom_dipole_map_coefficient_to_ebohr: np.ndarray
    atomic_dipole_partition_molecular_to_ebohr: np.ndarray
    molecular_polarizability_bohr3: np.ndarray
    baseline_atom_dipole_covariance_bohr3: np.ndarray
    target_atom_dipole_covariance_bohr3: np.ndarray
    conditional_null_covariance_coefficient_dual: np.ndarray
    lifting_map_coefficient_per_ebohr: np.ndarray
    charge_constraint_vector: np.ndarray | None
    baseline_symmetry_error: float
    completed_symmetry_error: float
    baseline_minimum_eigenvalue: float
    completed_minimum_eigenvalue: float
    baseline_atom_covariance_minimum_eigenvalue: float
    partition_identity_error: float
    baseline_charge_response_error: float
    completed_charge_response_error: float
    atom_covariance_error: float
    molecular_polarizability_error: float
    moore_penrose_error: float
    charge_nullspace_projection_error: float
    numerical_relative_tolerance: float
    construction: str = V0_RESPONSE_KERNEL_CONSTRUCTION

    def __post_init__(self) -> None:
        baseline = _immutable_array(
            self.baseline_response_covariance_coefficient_dual,
            name="Baseline response covariance",
        )
        if baseline.ndim != 2 or baseline.shape[0] == 0:
            raise ValueError("Baseline response covariance must be a nonempty matrix.")
        coefficient_count = baseline.shape[0]
        if baseline.shape != (coefficient_count, coefficient_count):
            raise ValueError("Baseline response covariance must be square.")
        atom_map = _immutable_array(
            self.atom_dipole_map_coefficient_to_ebohr,
            name="Atom-dipole map",
        )
        if atom_map.ndim != 2 or atom_map.shape[1] != coefficient_count:
            raise ValueError(
                "Atom-dipole map must have one column per density coefficient."
            )
        if atom_map.shape[0] == 0 or atom_map.shape[0] % 3 != 0:
            raise ValueError(
                "Atom-dipole map must have shape (3*n_atoms, n_coefficients)."
            )
        atomic_count = atom_map.shape[0]
        for name, shape in (
            ("response_covariance_coefficient_dual", baseline.shape),
            ("electronic_curvature_coefficient_dual", baseline.shape),
            ("response_support_projector", baseline.shape),
            ("response_support_constraints", (None, coefficient_count)),
            (
                "atomic_dipole_partition_molecular_to_ebohr",
                (atomic_count, 3),
            ),
            ("molecular_polarizability_bohr3", (3, 3)),
            ("baseline_atom_dipole_covariance_bohr3", (atomic_count, atomic_count)),
            ("target_atom_dipole_covariance_bohr3", (atomic_count, atomic_count)),
            ("conditional_null_covariance_coefficient_dual", baseline.shape),
            ("lifting_map_coefficient_per_ebohr", (coefficient_count, atomic_count)),
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if shape[0] is None:
                if values.ndim != 2 or values.shape[1] != shape[1]:
                    raise ValueError(
                        f"{name} must be finite with shape (n, {shape[1]})."
                    )
                object.__setattr__(self, name, _immutable_array(values, name=name))
            else:
                object.__setattr__(
                    self,
                    name,
                    _immutable_array(values, name=name, shape=shape),
                )
        if self.charge_constraint_vector is not None:
            object.__setattr__(
                self,
                "charge_constraint_vector",
                _immutable_array(
                    self.charge_constraint_vector,
                    name="charge_constraint_vector",
                    shape=(coefficient_count,),
                ),
            )
        object.__setattr__(
            self,
            "baseline_response_covariance_coefficient_dual",
            baseline,
        )
        object.__setattr__(
            self,
            "atom_dipole_map_coefficient_to_ebohr",
            atom_map,
        )
        for name in (
            "baseline_symmetry_error",
            "completed_symmetry_error",
            "baseline_minimum_eigenvalue",
            "completed_minimum_eigenvalue",
            "baseline_atom_covariance_minimum_eigenvalue",
            "partition_identity_error",
            "baseline_charge_response_error",
            "completed_charge_response_error",
            "atom_covariance_error",
            "molecular_polarizability_error",
            "moore_penrose_error",
            "charge_nullspace_projection_error",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(
                    getattr(self, name),
                    name=name,
                    nonnegative=name.endswith("_error"),
                ),
            )
        object.__setattr__(
            self,
            "numerical_relative_tolerance",
            _finite_scalar(
                self.numerical_relative_tolerance,
                name="numerical_relative_tolerance",
                nonnegative=True,
            ),
        )
        if self.numerical_relative_tolerance == 0.0:
            raise ValueError("numerical_relative_tolerance must be positive.")
        if self.construction != V0_RESPONSE_KERNEL_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 response-kernel construction.")


@dataclass(frozen=True)
class Route2V0MolecularMomentResponseKernelCompletion:
    """No-fit completion that constrains only the total molecular dipole.

    ``atomic_dipole_map_coefficient_to_ebohr`` remains the actual map used by
    the AIPR surface coupling.  The enclosed three-row completion is built on
    ``molecular_dipole_map_coefficient_to_ebohr = G @ A`` rather than on the
    full stacked atomic map ``A``.  It therefore has the same common scalar,
    charge-neutrality, PSD, and Moore--Penrose guarantees as the structural
    control without asserting an unsupported full ``3N x 3N`` atomic target.
    """

    molecular_moment_completion: Route2V0ResponseKernelCompletion
    atom_dipole_map_coefficient_to_ebohr: np.ndarray
    molecular_dipole_map_coefficient_to_ebohr: np.ndarray
    construction: str = V0_MOLECULAR_MOMENT_RESPONSE_KERNEL_CONSTRUCTION

    def __post_init__(self) -> None:
        if self.construction != V0_MOLECULAR_MOMENT_RESPONSE_KERNEL_CONSTRUCTION:
            raise ValueError(
                "Unsupported Route-2 V0 molecular-moment response-kernel construction."
            )
        completion = self.molecular_moment_completion
        if not isinstance(completion, Route2V0ResponseKernelCompletion):
            raise TypeError("molecular_moment_completion must be a V0 completion.")
        coefficient_count = completion.response_covariance_coefficient_dual.shape[0]
        atom_map = _immutable_array(
            self.atom_dipole_map_coefficient_to_ebohr,
            name="Atom-dipole map",
        )
        if (
            atom_map.ndim != 2
            or atom_map.shape[0] == 0
            or atom_map.shape[0] % 3 != 0
            or atom_map.shape[1] != coefficient_count
        ):
            raise ValueError(
                "Atom-dipole map must have shape (3*n_atoms, n_coefficients)."
            )
        molecular_map = _immutable_array(
            self.molecular_dipole_map_coefficient_to_ebohr,
            name="Molecular-dipole map",
            shape=(3, coefficient_count),
        )
        atom_sum = np.hstack([np.eye(3) for _ in range(atom_map.shape[0] // 3)])
        expected_molecular_map = atom_sum @ atom_map
        tolerance = _error_tolerance(
            expected_molecular_map,
            molecular_map,
            relative_tolerance=completion.numerical_relative_tolerance,
        )
        if np.linalg.norm(expected_molecular_map - molecular_map, ord=2) > tolerance:
            raise ValueError(
                "Molecular-dipole map must be the exact sum of atomic dipoles."
            )
        if not np.allclose(
            completion.atom_dipole_map_coefficient_to_ebohr,
            molecular_map,
            rtol=0.0,
            atol=tolerance,
        ) or not np.allclose(
            completion.atomic_dipole_partition_molecular_to_ebohr,
            np.eye(3),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "Molecular-moment completion does not use the declared total-dipole map."
            )
        object.__setattr__(self, "atom_dipole_map_coefficient_to_ebohr", atom_map)
        object.__setattr__(
            self,
            "molecular_dipole_map_coefficient_to_ebohr",
            molecular_map,
        )

    @property
    def baseline_response_covariance_coefficient_dual(self) -> np.ndarray:
        return self.molecular_moment_completion.baseline_response_covariance_coefficient_dual

    @property
    def response_covariance_coefficient_dual(self) -> np.ndarray:
        return self.molecular_moment_completion.response_covariance_coefficient_dual

    @property
    def electronic_curvature_coefficient_dual(self) -> np.ndarray:
        return self.molecular_moment_completion.electronic_curvature_coefficient_dual

    @property
    def response_support_projector(self) -> np.ndarray:
        return self.molecular_moment_completion.response_support_projector

    @property
    def response_support_constraints(self) -> np.ndarray:
        return self.molecular_moment_completion.response_support_constraints

    @property
    def charge_constraint_vector(self) -> np.ndarray | None:
        return self.molecular_moment_completion.charge_constraint_vector

    @property
    def molecular_polarizability_bohr3(self) -> np.ndarray:
        return self.molecular_moment_completion.molecular_polarizability_bohr3

    @property
    def baseline_molecular_polarizability_bohr3(self) -> np.ndarray:
        return self.molecular_moment_completion.baseline_atom_dipole_covariance_bohr3

    @property
    def target_molecular_polarizability_bohr3(self) -> np.ndarray:
        return self.molecular_moment_completion.target_atom_dipole_covariance_bohr3

    @property
    def conditional_null_covariance_coefficient_dual(self) -> np.ndarray:
        return self.molecular_moment_completion.conditional_null_covariance_coefficient_dual

    @property
    def lifting_map_coefficient_per_ebohr(self) -> np.ndarray:
        return self.molecular_moment_completion.lifting_map_coefficient_per_ebohr

    @property
    def baseline_symmetry_error(self) -> float:
        return self.molecular_moment_completion.baseline_symmetry_error

    @property
    def completed_symmetry_error(self) -> float:
        return self.molecular_moment_completion.completed_symmetry_error

    @property
    def baseline_minimum_eigenvalue(self) -> float:
        return self.molecular_moment_completion.baseline_minimum_eigenvalue

    @property
    def completed_minimum_eigenvalue(self) -> float:
        return self.molecular_moment_completion.completed_minimum_eigenvalue

    @property
    def baseline_molecular_polarizability_minimum_eigenvalue(self) -> float:
        return self.molecular_moment_completion.baseline_atom_covariance_minimum_eigenvalue

    @property
    def baseline_charge_response_error(self) -> float:
        return self.molecular_moment_completion.baseline_charge_response_error

    @property
    def completed_charge_response_error(self) -> float:
        return self.molecular_moment_completion.completed_charge_response_error

    @property
    def molecular_polarizability_error(self) -> float:
        return self.molecular_moment_completion.molecular_polarizability_error

    @property
    def moore_penrose_error(self) -> float:
        return self.molecular_moment_completion.moore_penrose_error

    @property
    def charge_nullspace_projection_error(self) -> float:
        return self.molecular_moment_completion.charge_nullspace_projection_error

    @property
    def numerical_relative_tolerance(self) -> float:
        return self.molecular_moment_completion.numerical_relative_tolerance


def complete_route2_v0_response_kernel(
    *,
    baseline_response_covariance_coefficient_dual: np.ndarray,
    atom_dipole_map_coefficient_to_ebohr: np.ndarray,
    atomic_dipole_partition_molecular_to_ebohr: np.ndarray,
    molecular_polarizability_bohr3: np.ndarray,
    charge_constraint_vector: np.ndarray | None = None,
    numerical_relative_tolerance: float = 1.0e-10,
) -> Route2V0ResponseKernelCompletion:
    """Complete a physical response kernel without fitting a response value.

    The caller owns the provenance of ``C0``.  It must be a fixed,
    independently sourced response covariance in the exact coefficient/dual
    pairing used later for the electronic scalar and continuum.  This routine
    performs no parameter optimization, comparison with solvation data, or
    repair of a nonphysical input; it fails closed instead.
    """

    if (
        not math.isfinite(numerical_relative_tolerance)
        or numerical_relative_tolerance <= 0.0
    ):
        raise ValueError(
            "numerical_relative_tolerance must be finite and positive."
        )
    baseline_values = np.asarray(
        baseline_response_covariance_coefficient_dual,
        dtype=float,
    )
    if baseline_values.ndim != 2 or baseline_values.shape[0] == 0:
        raise ValueError(
            "Baseline response covariance must be a nonempty square matrix."
        )
    coefficient_count = baseline_values.shape[0]
    baseline, baseline_symmetry_error = _symmetric_matrix(
        baseline_values,
        name="Baseline response covariance",
        dimension=coefficient_count,
        relative_tolerance=numerical_relative_tolerance,
    )
    (
        _,
        _,
        baseline_minimum,
        _,
    ) = _positive_semidefinite_spectrum(
        baseline,
        name="Baseline response covariance",
        relative_tolerance=numerical_relative_tolerance,
    )
    if charge_constraint_vector is None:
        charge_constraint = None
        baseline_charge_error = 0.0
    else:
        charge_constraint = _immutable_array(
            charge_constraint_vector,
            name="Charge constraint vector",
            shape=(coefficient_count,),
        )
        if float(np.linalg.norm(charge_constraint)) == 0.0:
            raise ValueError("Charge constraint vector must be nonzero.")
        baseline_charge_error = float(np.linalg.norm(baseline @ charge_constraint))
        baseline_charge_tolerance = _error_tolerance(
            baseline,
            charge_constraint[:, None],
            relative_tolerance=numerical_relative_tolerance,
        )
        if baseline_charge_error > baseline_charge_tolerance:
            raise ValueError(
                "Baseline response covariance violates the exact charge-neutral "
                "response constraint."
            )

    atom_map = _immutable_array(
        atom_dipole_map_coefficient_to_ebohr,
        name="Atom-dipole map",
    )
    if atom_map.ndim != 2 or atom_map.shape[1] != coefficient_count:
        raise ValueError(
            "Atom-dipole map must have shape (3*n_atoms, n_coefficients)."
        )
    if atom_map.shape[0] == 0 or atom_map.shape[0] % 3 != 0:
        raise ValueError(
            "Atom-dipole map must have shape (3*n_atoms, n_coefficients)."
        )
    atomic_dimension = atom_map.shape[0]
    atom_count = atomic_dimension // 3
    partition = _immutable_array(
        atomic_dipole_partition_molecular_to_ebohr,
        name="Atomic dipole partition",
        shape=(atomic_dimension, 3),
    )
    molecular_polarizability, _ = _symmetric_matrix(
        molecular_polarizability_bohr3,
        name="Molecular polarizability",
        dimension=3,
        relative_tolerance=numerical_relative_tolerance,
    )
    _positive_definite(
        molecular_polarizability,
        name="Molecular polarizability",
        relative_tolerance=numerical_relative_tolerance,
    )
    atom_sum = np.hstack([np.eye(3) for _ in range(atom_count)])
    partition_identity_error = float(np.linalg.norm(atom_sum @ partition - np.eye(3)))
    partition_tolerance = _error_tolerance(
        atom_sum @ partition,
        np.eye(3),
        relative_tolerance=numerical_relative_tolerance,
    )
    if partition_identity_error > partition_tolerance:
        raise ValueError(
            "Atomic dipole partition must sum exactly to the molecular dipole."
        )

    baseline_atom_covariance, _ = _symmetric_matrix(
        atom_map @ baseline @ atom_map.T,
        name="Baseline atom-dipole covariance",
        dimension=atomic_dimension,
        relative_tolerance=numerical_relative_tolerance,
    )
    baseline_atom_minimum, _ = _positive_definite(
        baseline_atom_covariance,
        name="Baseline atom-dipole covariance",
        relative_tolerance=numerical_relative_tolerance,
    )
    target_atom_covariance, _ = _symmetric_matrix(
        partition @ molecular_polarizability @ partition.T,
        name="Target atom-dipole covariance",
        dimension=atomic_dimension,
        relative_tolerance=numerical_relative_tolerance,
    )
    _positive_semidefinite_spectrum(
        target_atom_covariance,
        name="Target atom-dipole covariance",
        relative_tolerance=numerical_relative_tolerance,
    )
    try:
        lifting = np.linalg.solve(
            baseline_atom_covariance,
            atom_map @ baseline,
        ).T
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Baseline atom-dipole covariance is singular.") from error
    conditional_null, _ = _symmetric_matrix(
        baseline - lifting @ baseline_atom_covariance @ lifting.T,
        name="Conditional null-space covariance",
        dimension=coefficient_count,
        relative_tolerance=numerical_relative_tolerance,
    )
    _positive_semidefinite_spectrum(
        conditional_null,
        name="Conditional null-space covariance",
        relative_tolerance=numerical_relative_tolerance,
    )
    completed, completed_symmetry_error = _symmetric_matrix(
        conditional_null + lifting @ target_atom_covariance @ lifting.T,
        name="Completed response covariance",
        dimension=coefficient_count,
        relative_tolerance=numerical_relative_tolerance,
    )
    (
        eigenvalues,
        eigenvectors,
        completed_minimum,
        support_threshold,
    ) = _positive_semidefinite_spectrum(
        completed,
        name="Completed response covariance",
        relative_tolerance=numerical_relative_tolerance,
    )

    if charge_constraint is None:
        completed_charge_error = 0.0
    else:
        completed_charge_error = float(np.linalg.norm(completed @ charge_constraint))
        completed_charge_tolerance = _error_tolerance(
            completed,
            charge_constraint[:, None],
            relative_tolerance=numerical_relative_tolerance,
        )
        if completed_charge_error > completed_charge_tolerance:
            raise RuntimeError(
                "Response-kernel completion lost the exact charge-neutral response "
                "constraint."
            )
    atom_covariance_error = float(
        np.linalg.norm(
            atom_map @ completed @ atom_map.T - target_atom_covariance,
            ord=2,
        )
    )
    atom_covariance_tolerance = _error_tolerance(
        atom_map @ completed @ atom_map.T,
        target_atom_covariance,
        relative_tolerance=numerical_relative_tolerance,
    )
    if atom_covariance_error > atom_covariance_tolerance:
        raise RuntimeError(
            "Response-kernel completion does not reproduce the target "
            "atom-dipole covariance."
        )
    molecular_dipole_map = atom_sum @ atom_map
    molecular_polarizability_error = float(
        np.linalg.norm(
            molecular_dipole_map @ completed @ molecular_dipole_map.T
            - molecular_polarizability,
            ord=2,
        )
    )
    molecular_polarizability_tolerance = _error_tolerance(
        molecular_dipole_map @ completed @ molecular_dipole_map.T,
        molecular_polarizability,
        relative_tolerance=numerical_relative_tolerance,
    )
    if molecular_polarizability_error > molecular_polarizability_tolerance:
        raise RuntimeError(
            "Response-kernel completion does not reproduce the frozen "
            "molecular polarizability."
        )

    support = eigenvalues > support_threshold
    if not np.any(support):
        raise RuntimeError("Completed response covariance has no numerical support.")
    support_vectors = eigenvectors[:, support]
    response_support_projector = support_vectors @ support_vectors.T
    electronic_curvature = (
        support_vectors / eigenvalues[support][None, :]
    ) @ support_vectors.T
    null_vectors = eigenvectors[:, ~support]
    if charge_constraint is None:
        # Every transition-density response mode is neutral by construction.
        # Therefore no generic charge KKT row exists in this coordinate space.
        charge_nullspace_projection_error = 0.0
        response_support_constraints = null_vectors.T
    else:
        charge_null_projection = null_vectors @ (null_vectors.T @ charge_constraint)
        charge_nullspace_projection_error = float(
            np.linalg.norm(charge_constraint - charge_null_projection)
        )
        charge_nullspace_tolerance = _error_tolerance(
            charge_constraint[:, None],
            charge_null_projection[:, None],
            relative_tolerance=numerical_relative_tolerance,
        )
        if charge_nullspace_projection_error > charge_nullspace_tolerance:
            raise RuntimeError(
                "Charge constraint is not contained in the completed response "
                "kernel null space."
            )
        _, _, charge_null_right_vectors = np.linalg.svd(
            (null_vectors.T @ charge_constraint)[None, :],
            full_matrices=True,
        )
        # The total-charge row is imposed by the generic KKT solver. Return only
        # the remaining kernel rows so that additional constraints stay independent.
        response_support_constraints = charge_null_right_vectors[1:] @ null_vectors.T
    moore_penrose_error = float(
        np.linalg.norm(
            electronic_curvature @ completed - response_support_projector,
            ord=2,
        )
    )
    moore_penrose_tolerance = _error_tolerance(
        electronic_curvature @ completed,
        response_support_projector,
        relative_tolerance=numerical_relative_tolerance,
    )
    if moore_penrose_error > moore_penrose_tolerance:
        raise RuntimeError(
            "Response-kernel curvature is not the Moore--Penrose inverse on "
            "the declared support."
        )

    return Route2V0ResponseKernelCompletion(
        baseline_response_covariance_coefficient_dual=baseline,
        response_covariance_coefficient_dual=completed,
        electronic_curvature_coefficient_dual=electronic_curvature,
        response_support_projector=response_support_projector,
        response_support_constraints=response_support_constraints,
        atom_dipole_map_coefficient_to_ebohr=atom_map,
        atomic_dipole_partition_molecular_to_ebohr=partition,
        molecular_polarizability_bohr3=molecular_polarizability,
        baseline_atom_dipole_covariance_bohr3=baseline_atom_covariance,
        target_atom_dipole_covariance_bohr3=target_atom_covariance,
        conditional_null_covariance_coefficient_dual=conditional_null,
        lifting_map_coefficient_per_ebohr=lifting,
        charge_constraint_vector=charge_constraint,
        baseline_symmetry_error=baseline_symmetry_error,
        completed_symmetry_error=completed_symmetry_error,
        baseline_minimum_eigenvalue=baseline_minimum,
        completed_minimum_eigenvalue=completed_minimum,
        baseline_atom_covariance_minimum_eigenvalue=baseline_atom_minimum,
        partition_identity_error=partition_identity_error,
        baseline_charge_response_error=baseline_charge_error,
        completed_charge_response_error=completed_charge_error,
        atom_covariance_error=atom_covariance_error,
        molecular_polarizability_error=molecular_polarizability_error,
        moore_penrose_error=moore_penrose_error,
        charge_nullspace_projection_error=charge_nullspace_projection_error,
        numerical_relative_tolerance=numerical_relative_tolerance,
    )


def complete_molecular_moment_response_kernel(
    *,
    baseline_response_covariance_coefficient_dual: np.ndarray,
    atom_dipole_map_coefficient_to_ebohr: np.ndarray,
    molecular_polarizability_bohr3: np.ndarray,
    charge_constraint_vector: np.ndarray | None = None,
    numerical_relative_tolerance: float = 1.0e-10,
) -> Route2V0MolecularMomentResponseKernelCompletion:
    """Replace only the total molecular-dipole covariance of ``C0``.

    Let ``A`` map coefficients to stacked atomic dipoles and let ``G`` sum
    those dipoles.  This routine completes the kernel with
    ``T = G @ A`` and the frozen target ``alpha``:

    ``S0 = T @ C0 @ T.T``
    ``L = C0 @ T.T @ inv(S0)``
    ``C = C0 - L @ S0 @ L.T + L @ alpha @ L.T``.

    Thus ``T @ C @ T.T = alpha`` while the baseline conditional covariance
    orthogonal to the *molecular* moment is retained.  Unlike the historical
    stacked-atomic control, this routine makes no unsupported claim about the
    full atom-dipole covariance and does not force that covariance to rank
    three.  It is still a structural primitive: it neither supplies a
    molecular charge-transfer model nor validates nonuniform QM response.
    """

    atom_map = _immutable_array(
        atom_dipole_map_coefficient_to_ebohr,
        name="Atom-dipole map",
    )
    baseline_values = np.asarray(
        baseline_response_covariance_coefficient_dual,
        dtype=float,
    )
    if baseline_values.ndim != 2 or baseline_values.shape[0] == 0:
        raise ValueError(
            "Baseline response covariance must be a nonempty square matrix."
        )
    coefficient_count = baseline_values.shape[0]
    if (
        atom_map.ndim != 2
        or atom_map.shape[1] != coefficient_count
        or atom_map.shape[0] == 0
        or atom_map.shape[0] % 3 != 0
    ):
        raise ValueError(
            "Atom-dipole map must have shape (3*n_atoms, n_coefficients)."
        )
    atom_sum = np.hstack([np.eye(3) for _ in range(atom_map.shape[0] // 3)])
    molecular_map = atom_sum @ atom_map

    # Reuse the fully audited algebra with one three-component "atom".  The
    # wrapper restores the true stacked atomic map for the AIPR/KKT coupling.
    molecular_completion = complete_route2_v0_response_kernel(
        baseline_response_covariance_coefficient_dual=(
            baseline_response_covariance_coefficient_dual
        ),
        atom_dipole_map_coefficient_to_ebohr=molecular_map,
        atomic_dipole_partition_molecular_to_ebohr=np.eye(3),
        molecular_polarizability_bohr3=molecular_polarizability_bohr3,
        charge_constraint_vector=charge_constraint_vector,
        numerical_relative_tolerance=numerical_relative_tolerance,
    )
    return Route2V0MolecularMomentResponseKernelCompletion(
        molecular_moment_completion=molecular_completion,
        atom_dipole_map_coefficient_to_ebohr=atom_map,
        molecular_dipole_map_coefficient_to_ebohr=molecular_map,
    )


__all__ = [
    "Route2V0MolecularMomentResponseKernelCompletion",
    "Route2V0ResponseKernelCompletion",
    "V0_MOLECULAR_MOMENT_RESPONSE_KERNEL_CONSTRUCTION",
    "V0_RESPONSE_KERNEL_CONSTRUCTION",
    "complete_molecular_moment_response_kernel",
    "complete_route2_v0_response_kernel",
]
