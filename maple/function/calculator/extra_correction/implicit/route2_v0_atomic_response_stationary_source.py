"""Stationary permanent-source construction from the frozen V0 atomic-IP basis.

The frozen atomic independent-particle (AIP) asset already supplies two
objects in one coefficient/dual space: an all-electron atomic reference
density and neutral transition-density modes.  The latter, after the frozen
MACE-MDP moment completion, defines the positive response covariance ``C``.
This module uses no fitted charge, width, response scale, or target label to
choose the otherwise-undetermined permanent affine origin.

For a frozen direct-sum atomic reference ``n_ref`` and a transition-density
coordinate ``x``, define the interaction dual

``b_m = integral tau_m(r) [-V_nuc,other(r) + J[n_other](r)] dr``.

The permanent correction is the constrained stationary point of the one
scalar

``F_ref(x) = E_ref + 0.5*x.T*C_plus*x + x.T*b``

on the declared response support.  It is therefore ``x0 = -C*b`` and not a
promolecular density relabelled as a molecular state.  The PySCF-dependent
part is intentionally lazy: it provides exact AO Coulomb integrals only when
building a concrete frozen-geometry reference.  Importing and testing the
algebra never requires a quantum-chemistry runtime.

This is a fixed-geometry candidate source constructor.  It is not yet a
continuum, force/PES implementation, universal molecular electronic
functional, or solvation result.  A registered static-QM-MEP falsifier and
subsequent broad physics gates decide whether it is admissible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .route2_v0_atomic_independent_particle_response import (
    Route2V0AtomicIndependentParticleBaseline,
    Route2V0AtomicIndependentParticleResponseTable,
)
from .route2_v0_atomic_independent_particle_surface import BOHR_ANGSTROM
from .route2_v0_response_kernel import (
    Route2V0MolecularMomentResponseKernelCompletion,
    Route2V0ResponseKernelCompletion,
)

V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION = (
    "route2-v0-atomic-response-stationary-permanent-source-v1"
)
_NUMERICAL_RELATIVE_TOLERANCE = 1.0e-10
V0_ATOMIC_RESPONSE_DENSITY_GRID_BUFFER_BOHR = 5.0
V0_ATOMIC_RESPONSE_DENSITY_GRID_SPACING_BOHR = 0.4


def _immutable_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array.") from error
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must have {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _infinity_norm(values: np.ndarray) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _positions(values: object, *, atom_count: int) -> np.ndarray:
    return _immutable_array(
        values,
        name="atom_positions_angstrom",
        shape=(atom_count, 3),
    )


def _atomic_numbers(values: object) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("atomic_numbers must be a nonempty finite vector.")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric) or np.any(rounded <= 0.0):
        raise ValueError("atomic_numbers must contain positive exact integers.")
    result = np.asarray(rounded, dtype=np.int64)
    result.setflags(write=False)
    return result


def _symmetric(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _immutable_array(values, name=name)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix.")
    antisymmetry = float(np.linalg.norm(0.5 * (matrix - matrix.T), ord=2))
    if antisymmetry > _NUMERICAL_RELATIVE_TOLERANCE * _matrix_scale(matrix):
        raise ValueError(f"{name} must be symmetric in its declared pairing.")
    result = 0.5 * (matrix + matrix.T)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Route2V0AtomicResponseStationaryState:
    """The support-constrained stationary correction to the atomic reference."""

    reference_coefficient_dual_hartree: np.ndarray
    projected_reference_coefficient_dual_hartree: np.ndarray
    permanent_response_coefficients: np.ndarray
    electronic_correction_energy_hartree: float
    stationary_energy_identity_error_hartree: float
    stationarity_residual_inf_hartree: float
    support_constraint_residual_inf: float
    support_minimum_curvature_hartree: float
    stationarity_tolerance_hartree: float
    construction: str = V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION

    def __post_init__(self) -> None:
        count = np.asarray(self.permanent_response_coefficients).size
        if count == 0:
            raise ValueError("Permanent response coefficients must be nonempty.")
        for name in (
            "reference_coefficient_dual_hartree",
            "projected_reference_coefficient_dual_hartree",
            "permanent_response_coefficients",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), name=name, shape=(count,)),
            )
        for name in (
            "electronic_correction_energy_hartree",
            "stationary_energy_identity_error_hartree",
            "stationarity_residual_inf_hartree",
            "support_constraint_residual_inf",
            "support_minimum_curvature_hartree",
            "stationarity_tolerance_hartree",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.support_minimum_curvature_hartree <= 0.0:
            raise ValueError("Support curvature must be positive.")
        if self.stationarity_tolerance_hartree <= 0.0:
            raise ValueError("Stationarity tolerance must be positive.")
        if self.stationarity_residual_inf_hartree > self.stationarity_tolerance_hartree:
            raise ValueError(
                "Permanent source is not stationary on the response support."
            )
        if self.construction != V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported atomic-response stationary construction.")


@dataclass(frozen=True)
class Route2V0AtomicResponsePermanentDensityAudit:
    """Conservation and AO-carrier diagnostics for one permanent density field.

    The response completion acts on a density expansion, not on a variational
    many-electron wavefunction.  Its AO product coefficient matrix is a carrier
    for ``n(r)`` and need not be a one-particle density matrix with Pauli
    occupations.  Pointwise density nonnegativity, not this carrier's metric
    spectrum, is the physical source gate.
    """

    density_matrix: np.ndarray
    electron_count_e: float
    reference_electron_count_e: float
    electron_count_error_e: float
    minimum_ao_metric_density_eigenvalue: float
    maximum_ao_metric_density_eigenvalue: float
    density_symmetry_error: float
    construction: str = V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION

    def __post_init__(self) -> None:
        density = _symmetric(self.density_matrix, name="permanent_density_matrix")
        object.__setattr__(self, "density_matrix", density)
        for name in (
            "electron_count_e",
            "reference_electron_count_e",
            "electron_count_error_e",
            "minimum_ao_metric_density_eigenvalue",
            "maximum_ao_metric_density_eigenvalue",
            "density_symmetry_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.electron_count_error_e < 0.0 or self.density_symmetry_error < 0.0:
            raise ValueError("Permanent-density errors must be nonnegative.")
        if self.construction != V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported atomic-response permanent-density audit.")


@dataclass(frozen=True)
class Route2V0AtomicResponsePyscfReference:
    """Frozen AO carrier for the all-electron atomic-reference source.

    ``molecule`` is a PySCF integral carrier.  It is intentionally typed as
    ``Any`` so importing this module does not make PySCF a MAPLE dependency.
    The stored density is the direct sum of the frozen spherical atomic HF
    ground-state densities; it becomes a candidate molecular state only after
    the stationary neutral correction returned by this module is applied.
    """

    response_table: Route2V0AtomicIndependentParticleResponseTable
    baseline: Route2V0AtomicIndependentParticleBaseline
    atom_positions_angstrom: np.ndarray
    molecule: Any
    atom_ao_slices: tuple[slice, ...]
    atomic_ground_density_matrices: tuple[np.ndarray, ...]
    ground_density_matrix: np.ndarray
    overlap_matrix: np.ndarray
    atomic_mo_metric_errors: np.ndarray
    reference_electron_count_e: float
    construction: str = V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION

    def __post_init__(self) -> None:
        if not isinstance(
            self.response_table,
            Route2V0AtomicIndependentParticleResponseTable,
        ):
            raise TypeError("response_table must be a frozen atomic-IP table.")
        if not isinstance(self.baseline, Route2V0AtomicIndependentParticleBaseline):
            raise TypeError("baseline must be a frozen atomic-IP direct sum.")
        atom_count = len(self.baseline.atomic_numbers)
        positions = _positions(self.atom_positions_angstrom, atom_count=atom_count)
        if len(self.atom_ao_slices) != atom_count:
            raise ValueError("The AO carrier requires one slice per atom.")
        expected_ao = 0
        local_densities: list[np.ndarray] = []
        for atom_index, ao_slice in enumerate(self.atom_ao_slices):
            if (
                ao_slice.step not in (None, 1)
                or ao_slice.start != expected_ao
                or ao_slice.stop is None
                or ao_slice.stop <= expected_ao
            ):
                raise ValueError("Atomic AO slices must be contiguous and complete.")
            response = self.response_table.response_for_atomic_number(
                int(self.baseline.atomic_numbers[atom_index])
            )
            width = ao_slice.stop - ao_slice.start
            density = _symmetric(
                np.asarray(self.atomic_ground_density_matrices[atom_index]),
                name="atomic_ground_density_matrix",
            )
            if (
                density.shape != (width, width)
                or width != response.mo_coefficients.shape[0]
            ):
                raise ValueError(
                    "Atomic ground density disagrees with its frozen AO block."
                )
            local_densities.append(density)
            expected_ao = ao_slice.stop
        if (
            not hasattr(self.molecule, "nao_nr")
            or int(self.molecule.nao_nr()) != expected_ao
        ):
            raise ValueError(
                "PySCF AO carrier disagrees with the frozen atomic blocks."
            )
        ground_density = _symmetric(
            self.ground_density_matrix, name="ground_density_matrix"
        )
        overlap = _symmetric(self.overlap_matrix, name="overlap_matrix")
        if (
            ground_density.shape != (expected_ao, expected_ao)
            or overlap.shape != ground_density.shape
        ):
            raise ValueError(
                "Global ground density and overlap must match the AO carrier."
            )
        for density, ao_slice in zip(local_densities, self.atom_ao_slices, strict=True):
            if not np.allclose(
                ground_density[ao_slice, ao_slice],
                density,
                rtol=0.0,
                atol=1.0e-13,
            ):
                raise ValueError(
                    "Global ground density does not retain an atomic block."
                )
        off_block = ground_density.copy()
        for ao_slice in self.atom_ao_slices:
            off_block[ao_slice, ao_slice] = 0.0
        if _infinity_norm(off_block) > 1.0e-13:
            raise ValueError("Frozen atomic ground density must remain a direct sum.")
        metric_errors = _immutable_array(
            self.atomic_mo_metric_errors,
            name="atomic_mo_metric_errors",
            shape=(atom_count,),
        )
        if np.any(metric_errors > _NUMERICAL_RELATIVE_TOLERANCE):
            raise ValueError("The PySCF AO carrier changed a frozen atomic MO metric.")
        reference_count = float(self.reference_electron_count_e)
        if not math.isfinite(reference_count) or reference_count <= 0.0:
            raise ValueError("reference_electron_count_e must be finite and positive.")
        actual_count = float(
            np.einsum("ij,ji->", ground_density, overlap, optimize=True)
        )
        if abs(actual_count - reference_count) > _NUMERICAL_RELATIVE_TOLERANCE * max(
            1.0, reference_count
        ):
            raise ValueError(
                "Atomic ground density does not preserve its electron count."
            )
        if self.construction != V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported atomic-response PySCF carrier.")
        object.__setattr__(self, "atom_positions_angstrom", positions)
        object.__setattr__(
            self, "atomic_ground_density_matrices", tuple(local_densities)
        )
        object.__setattr__(self, "ground_density_matrix", ground_density)
        object.__setattr__(self, "overlap_matrix", overlap)
        object.__setattr__(self, "atomic_mo_metric_errors", metric_errors)
        object.__setattr__(self, "reference_electron_count_e", reference_count)

    @property
    def atom_count(self) -> int:
        return len(self.baseline.atomic_numbers)

    @property
    def coefficient_count(self) -> int:
        return self.baseline.coefficient_count

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self.baseline.atomic_numbers.copy()

    def reference_coefficient_dual_hartree(self) -> np.ndarray:
        """Return the exact frozen-reference interaction dual ``b``.

        Only *other* atoms enter each local dual.  The sign is the electronic
        energy pairing: other nuclei are attractive and other electron-number
        density produces the positive Coulomb operator.  Self terms belong to
        the independently frozen atomic response curvature and are not added
        a second time.
        """

        try:
            from pyscf.scf import hf
        except ImportError as error:
            raise RuntimeError(
                "PySCF is required to evaluate the stationary atomic-reference dual."
            ) from error
        total_j, _ = hf.get_jk(
            self.molecule,
            self.ground_density_matrix,
            hermi=1,
            with_j=True,
            with_k=False,
        )
        total_j = _symmetric(total_j, name="reference_total_coulomb_matrix")
        self_densities = np.zeros(
            (
                self.atom_count,
                self.ground_density_matrix.shape[0],
                self.ground_density_matrix.shape[1],
            ),
            dtype=float,
        )
        for atom_index, (density, ao_slice) in enumerate(
            zip(self.atomic_ground_density_matrices, self.atom_ao_slices, strict=True)
        ):
            self_densities[atom_index, ao_slice, ao_slice] = density
        self_j, _ = hf.get_jk(
            self.molecule,
            self_densities,
            hermi=1,
            with_j=True,
            with_k=False,
        )
        self_j = np.asarray(self_j, dtype=float)
        if self_j.shape != self_densities.shape:
            raise RuntimeError(
                "PySCF did not return one Coulomb matrix per atomic density."
            )
        positions_bohr = self.atom_positions_angstrom / BOHR_ANGSTROM
        numbers = self.baseline.atomic_numbers
        dual = np.empty(self.coefficient_count, dtype=float)
        for atom_index, (coefficient_slice, ao_slice) in enumerate(
            zip(self.baseline.coefficient_slices, self.atom_ao_slices, strict=True)
        ):
            other_operator = (
                total_j[ao_slice, ao_slice] - self_j[atom_index, ao_slice, ao_slice]
            )
            for other_index, charge in enumerate(numbers):
                if other_index == atom_index:
                    continue
                with self.molecule.with_rinv_origin(positions_bohr[other_index]):
                    inverse_distance = self.molecule.intor("int1e_rinv")
                other_operator -= float(charge) * inverse_distance[ao_slice, ao_slice]
            response = self.response_table.response_for_atomic_number(
                int(numbers[atom_index])
            )
            transitions = response.transition_density_matrices()
            dual[coefficient_slice] = np.einsum(
                "mij,ij->m",
                transitions,
                other_operator,
                optimize=True,
            )
        return _immutable_array(
            dual,
            name="reference_coefficient_dual_hartree",
            shape=(self.coefficient_count,),
        )

    def permanent_density_audit(
        self,
        state: Route2V0AtomicResponseStationaryState,
    ) -> Route2V0AtomicResponsePermanentDensityAudit:
        """Apply the stationary neutral correction and audit its AO density."""

        if not isinstance(state, Route2V0AtomicResponseStationaryState):
            raise TypeError("state must be an atomic-response stationary state.")
        if len(state.permanent_response_coefficients) != self.coefficient_count:
            raise ValueError(
                "Stationary coefficients do not match the frozen atomic basis."
            )
        density = np.array(self.ground_density_matrix, dtype=float, copy=True)
        for coefficient_slice, ao_slice, number in zip(
            self.baseline.coefficient_slices,
            self.atom_ao_slices,
            self.baseline.atomic_numbers,
            strict=True,
        ):
            transitions = self.response_table.response_for_atomic_number(
                int(number)
            ).transition_density_matrices()
            density[ao_slice, ao_slice] += np.einsum(
                "m,mij->ij",
                state.permanent_response_coefficients[coefficient_slice],
                transitions,
                optimize=True,
            )
        density_symmetry_error = float(
            np.linalg.norm(0.5 * (density - density.T), ord=2)
        )
        density = 0.5 * (density + density.T)
        electron_count = float(
            np.einsum("ij,ji->", density, self.overlap_matrix, optimize=True)
        )
        overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(self.overlap_matrix)
        if np.min(overlap_eigenvalues) <= _NUMERICAL_RELATIVE_TOLERANCE * _matrix_scale(
            self.overlap_matrix
        ):
            raise RuntimeError("PySCF atomic AO overlap is not positive definite.")
        overlap_sqrt = (
            overlap_eigenvectors * np.sqrt(overlap_eigenvalues)
        ) @ overlap_eigenvectors.T
        metric_density_eigenvalues = np.linalg.eigvalsh(
            0.5
            * (
                overlap_sqrt @ density @ overlap_sqrt
                + (overlap_sqrt @ density @ overlap_sqrt).T
            )
        )
        return Route2V0AtomicResponsePermanentDensityAudit(
            density_matrix=density,
            electron_count_e=electron_count,
            reference_electron_count_e=self.reference_electron_count_e,
            electron_count_error_e=abs(
                electron_count - self.reference_electron_count_e
            ),
            minimum_ao_metric_density_eigenvalue=float(
                np.min(metric_density_eigenvalues)
            ),
            maximum_ao_metric_density_eigenvalue=float(
                np.max(metric_density_eigenvalues)
            ),
            density_symmetry_error=density_symmetry_error,
        )

    def density_positivity_grid_bohr(self) -> np.ndarray:
        """Return the fixed Cartesian density-source canary grid for this geometry.

        The box construction and spacing are numerical validation conventions,
        declared before any source comparison.  They are not cavity radii,
        continuum parameters, or error-selected density thresholds.
        """

        positions_bohr = self.atom_positions_angstrom / BOHR_ANGSTROM
        lower = (
            np.min(positions_bohr, axis=0) - V0_ATOMIC_RESPONSE_DENSITY_GRID_BUFFER_BOHR
        )
        upper = (
            np.max(positions_bohr, axis=0) + V0_ATOMIC_RESPONSE_DENSITY_GRID_BUFFER_BOHR
        )
        axes = tuple(
            lower[axis]
            + V0_ATOMIC_RESPONSE_DENSITY_GRID_SPACING_BOHR
            * np.arange(
                int(
                    np.ceil(
                        (upper[axis] - lower[axis])
                        / V0_ATOMIC_RESPONSE_DENSITY_GRID_SPACING_BOHR
                    )
                )
                + 1,
                dtype=float,
            )
            for axis in range(3)
        )
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        return _immutable_array(
            grid,
            name="density_positivity_grid_bohr",
            shape=grid.shape,
        )

    def electron_number_density_e_per_bohr3(
        self,
        density_audit: Route2V0AtomicResponsePermanentDensityAudit,
        points_bohr: np.ndarray,
        *,
        maximum_points_per_block: int = 4096,
    ) -> np.ndarray:
        """Evaluate the represented electron-number density without renormalizing it."""

        if not isinstance(density_audit, Route2V0AtomicResponsePermanentDensityAudit):
            raise TypeError("density_audit must be a permanent-density audit.")
        points = _immutable_array(points_bohr, name="points_bohr")
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise ValueError("points_bohr must have finite shape (n_points, 3).")
        if (
            isinstance(maximum_points_per_block, bool)
            or int(maximum_points_per_block) != maximum_points_per_block
            or maximum_points_per_block <= 0
        ):
            raise ValueError("maximum_points_per_block must be a positive integer.")
        try:
            from pyscf.dft import numint
        except ImportError as error:
            raise RuntimeError(
                "PySCF is required to evaluate an AO density field."
            ) from error
        density = np.empty(len(points), dtype=float)
        for start in range(0, len(points), int(maximum_points_per_block)):
            stop = min(start + int(maximum_points_per_block), len(points))
            ao_values = numint.eval_ao(self.molecule, points[start:stop])
            density[start:stop] = np.einsum(
                "pi,ij,pj->p",
                ao_values,
                density_audit.density_matrix,
                ao_values,
                optimize=True,
            )
        return _immutable_array(
            density,
            name="electron_number_density_e_per_bohr3",
            shape=(len(points),),
        )

    def total_permanent_potential_hartree_per_e(
        self,
        density_audit: Route2V0AtomicResponsePermanentDensityAudit,
        points_bohr: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the complete nuclear-minus-electron permanent MEP."""

        if not isinstance(density_audit, Route2V0AtomicResponsePermanentDensityAudit):
            raise TypeError("density_audit must be a permanent-density audit.")
        points = _immutable_array(points_bohr, name="points_bohr")
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise ValueError("points_bohr must have finite shape (n_points, 3).")
        positions_bohr = self.atom_positions_angstrom / BOHR_ANGSTROM
        charges = np.asarray(self.baseline.atomic_numbers, dtype=float)
        potential = np.empty(len(points), dtype=float)
        for index, point in enumerate(points):
            distances = np.linalg.norm(positions_bohr - point, axis=1)
            if np.any(distances <= 1.0e-12):
                raise ValueError(
                    "A permanent-MEP point coincides with a source nucleus."
                )
            nuclear = float(np.dot(charges, 1.0 / distances))
            with self.molecule.with_rinv_origin(point):
                inverse_distance = self.molecule.intor("int1e_rinv")
            electronic = -float(
                np.einsum(
                    "ij,ij->",
                    density_audit.density_matrix,
                    inverse_distance,
                    optimize=True,
                )
            )
            potential[index] = nuclear + electronic
        return _immutable_array(
            potential,
            name="total_permanent_potential_hartree_per_e",
            shape=(len(points),),
        )

    def total_permanent_dipole_e_bohr(
        self,
        density_audit: Route2V0AtomicResponsePermanentDensityAudit,
    ) -> np.ndarray:
        """Return the total nuclear-minus-electron permanent dipole."""

        if not isinstance(density_audit, Route2V0AtomicResponsePermanentDensityAudit):
            raise TypeError("density_audit must be a permanent-density audit.")
        positions_bohr = self.atom_positions_angstrom / BOHR_ANGSTROM
        nuclear = np.einsum(
            "a,ax->x",
            self.baseline.atomic_numbers,
            positions_bohr,
            optimize=True,
        )
        position_integrals = self.molecule.intor_symmetric("int1e_r", comp=3)
        electronic = np.einsum(
            "xij,ij->x",
            position_integrals,
            density_audit.density_matrix,
            optimize=True,
        )
        return _immutable_array(
            nuclear - electronic,
            name="total_permanent_dipole_e_bohr",
            shape=(3,),
        )


def solve_route2_v0_atomic_response_stationary_state(
    *,
    completion: (
        Route2V0ResponseKernelCompletion
        | Route2V0MolecularMomentResponseKernelCompletion
    ),
    reference_coefficient_dual_hartree: np.ndarray,
) -> Route2V0AtomicResponseStationaryState:
    """Solve the unmodified support-constrained atomic-response stationarity.

    The stationary equation is ``C_plus*x + P*b = 0`` with ``x`` restricted
    by the exact null/support rows already supplied by the response completion.
    No inverse is newly constructed here: ``C_plus`` and ``C`` must be the
    mutually certified objects from ``complete_route2_v0_response_kernel``.
    """

    if not isinstance(
        completion,
        (
            Route2V0ResponseKernelCompletion,
            Route2V0MolecularMomentResponseKernelCompletion,
        ),
    ):
        raise TypeError("completion must be a V0 response-kernel completion.")
    count = completion.response_covariance_coefficient_dual.shape[0]
    dual = _immutable_array(
        reference_coefficient_dual_hartree,
        name="reference_coefficient_dual_hartree",
        shape=(count,),
    )
    covariance = _symmetric(
        completion.response_covariance_coefficient_dual,
        name="response_covariance_coefficient_dual",
    )
    curvature = _symmetric(
        completion.electronic_curvature_coefficient_dual,
        name="electronic_curvature_coefficient_dual",
    )
    support = _symmetric(
        completion.response_support_projector, name="response_support_projector"
    )
    support_error = float(np.linalg.norm(support @ support - support, ord=2))
    if support_error > _NUMERICAL_RELATIVE_TOLERANCE * _matrix_scale(support):
        raise ValueError("Response support projector is not idempotent.")
    projected_dual = support @ dual
    coefficients = -(covariance @ projected_dual)
    stationarity = curvature @ coefficients + projected_dual
    constraints = np.asarray(completion.response_support_constraints, dtype=float)
    if constraints.ndim != 2 or constraints.shape[1] != count:
        raise ValueError("Response support constraints do not match the covariance.")
    support_residual = constraints @ coefficients
    support_eigenvalues, support_eigenvectors = np.linalg.eigh(support)
    supported = support_eigenvalues > 0.5
    if (
        not np.any(supported)
        or np.any(support_eigenvalues[supported] < 1.0 - _NUMERICAL_RELATIVE_TOLERANCE)
        or np.any(support_eigenvalues[~supported] > _NUMERICAL_RELATIVE_TOLERANCE)
    ):
        raise ValueError("Response support projector has an invalid spectrum.")
    support_vectors = support_eigenvectors[:, supported]
    minimum_curvature = float(
        np.min(np.linalg.eigvalsh(support_vectors.T @ curvature @ support_vectors))
    )
    tolerance = _NUMERICAL_RELATIVE_TOLERANCE * max(
        1.0,
        _matrix_scale(curvature) * _infinity_norm(coefficients),
        _infinity_norm(projected_dual),
    )
    if minimum_curvature <= tolerance:
        raise ValueError(
            "Electronic curvature is not positive on the declared support."
        )
    correction_energy = float(
        0.5 * coefficients @ curvature @ coefficients + coefficients @ dual
    )
    closed_form_energy = float(-0.5 * dual @ covariance @ dual)
    identity_error = abs(correction_energy - closed_form_energy)
    stationarity_residual = _infinity_norm(stationarity)
    support_constraint_residual = _infinity_norm(support_residual)
    if stationarity_residual > tolerance:
        raise RuntimeError("Atomic-response stationary solve failed its KKT equation.")
    if support_constraint_residual > tolerance:
        raise RuntimeError(
            "Atomic-response stationary solve left the response support."
        )
    if identity_error > tolerance:
        raise RuntimeError("Atomic-response stationary energy ledger is inconsistent.")
    return Route2V0AtomicResponseStationaryState(
        reference_coefficient_dual_hartree=dual,
        projected_reference_coefficient_dual_hartree=projected_dual,
        permanent_response_coefficients=coefficients,
        electronic_correction_energy_hartree=correction_energy,
        stationary_energy_identity_error_hartree=identity_error,
        stationarity_residual_inf_hartree=stationarity_residual,
        support_constraint_residual_inf=support_constraint_residual,
        support_minimum_curvature_hartree=minimum_curvature,
        stationarity_tolerance_hartree=tolerance,
    )


def build_route2_v0_atomic_response_pyscf_reference(
    *,
    response_table: Route2V0AtomicIndependentParticleResponseTable,
    atomic_numbers: np.ndarray,
    atom_positions_angstrom: np.ndarray,
) -> Route2V0AtomicResponsePyscfReference:
    """Build a hash-bound global AO carrier for the frozen atomic reference."""

    if not isinstance(response_table, Route2V0AtomicIndependentParticleResponseTable):
        raise TypeError("response_table must be a frozen atomic-IP response table.")
    numbers = _atomic_numbers(atomic_numbers)
    baseline = response_table.assemble(numbers)
    positions = _positions(atom_positions_angstrom, atom_count=len(numbers))
    try:
        from pyscf import gto
    except ImportError as error:
        raise RuntimeError(
            "PySCF is required to construct a global atomic-response AO carrier."
        ) from error
    positions_bohr = positions / BOHR_ANGSTROM
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    basis: dict[str, str] = {}
    for atom_index, (number, position) in enumerate(
        zip(numbers, positions_bohr, strict=True)
    ):
        response = response_table.response_for_atomic_number(int(number))
        label = f"{response.symbol}{atom_index + 1}"
        atoms.append((label, tuple(float(value) for value in position)))
        basis[label] = response.basis
    molecule = gto.M(
        atom=atoms,
        basis=basis,
        charge=0,
        # The AO integral carrier has no molecular SCF state.  PySCF still
        # requires a parity-compatible spin declaration when constructing it.
        spin=int(np.sum(numbers) % 2),
        unit="Bohr",
        symmetry=False,
        verbose=0,
    )
    ao_rows = np.asarray(molecule.aoslice_by_atom(), dtype=int)
    if ao_rows.shape != (len(numbers), 4):
        raise RuntimeError("PySCF did not expose one AO slice per atomic reference.")
    atom_slices = tuple(slice(int(row[2]), int(row[3])) for row in ao_rows)
    expected_ao = sum(
        response_table.response_for_atomic_number(int(number)).mo_coefficients.shape[0]
        for number in numbers
    )
    if molecule.nao_nr() != expected_ao:
        raise RuntimeError("PySCF AO count disagrees with the frozen atomic table.")
    overlap = molecule.intor_symmetric("int1e_ovlp")
    ground_density = np.zeros((expected_ao, expected_ao), dtype=float)
    local_densities: list[np.ndarray] = []
    metric_errors: list[float] = []
    reference_count = 0.0
    for number, ao_slice in zip(numbers, atom_slices, strict=True):
        response = response_table.response_for_atomic_number(int(number))
        coefficients = response.mo_coefficients
        occupations = response.orbital_occupations
        width = ao_slice.stop - ao_slice.start
        if coefficients.shape[0] != width:
            raise RuntimeError(
                "PySCF atom-local AO order disagrees with the frozen table."
            )
        local_overlap = overlap[ao_slice, ao_slice]
        metric_error = float(
            np.linalg.norm(
                coefficients.T @ local_overlap @ coefficients
                - np.eye(coefficients.shape[1]),
                ord=2,
            )
        )
        local_density = (coefficients * occupations) @ coefficients.T
        local_density = 0.5 * (local_density + local_density.T)
        ground_density[ao_slice, ao_slice] = local_density
        local_densities.append(local_density)
        metric_errors.append(metric_error)
        reference_count += float(np.sum(occupations))
    return Route2V0AtomicResponsePyscfReference(
        response_table=response_table,
        baseline=baseline,
        atom_positions_angstrom=positions,
        molecule=molecule,
        atom_ao_slices=atom_slices,
        atomic_ground_density_matrices=tuple(local_densities),
        ground_density_matrix=ground_density,
        overlap_matrix=overlap,
        atomic_mo_metric_errors=np.asarray(metric_errors, dtype=float),
        reference_electron_count_e=reference_count,
    )


__all__ = [
    "Route2V0AtomicResponsePermanentDensityAudit",
    "Route2V0AtomicResponsePyscfReference",
    "Route2V0AtomicResponseStationaryState",
    "V0_ATOMIC_RESPONSE_DENSITY_GRID_BUFFER_BOHR",
    "V0_ATOMIC_RESPONSE_DENSITY_GRID_SPACING_BOHR",
    "V0_ATOMIC_RESPONSE_STATIONARY_SOURCE_CONSTRUCTION",
    "build_route2_v0_atomic_response_pyscf_reference",
    "solve_route2_v0_atomic_response_stationary_state",
]
