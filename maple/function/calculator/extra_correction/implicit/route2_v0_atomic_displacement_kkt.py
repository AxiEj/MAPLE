"""Reduced common-scalar KKT kernel for the Route-2 V0-ADT induced source.

V0-ADT is a source-provenanced *induced* real-space electronic response.  At
fixed geometry its molecular induced dipole ``p`` has the frozen MACE-MDP
quadratic cost ``0.5 p.T alpha^-1 p``.  The atomic free-atom translation
tangent maps that same ``p`` to a surface potential with one dense matrix
``B``.  A reciprocal continuum response then yields the reduced scalar

``L(p; f) = 0.5 p.T alpha^-1 p + 0.5 (B p).T Q (B p) + p.T f``,

where ``Q`` is the continuum's energy-conjugate surface response and ``f`` is
an external dipole dual.  The continuum surface charge has already been
eliminated; this module therefore proves the induced-source/common-energy
part of the KKT construction, not a complete permanent-density or total
solvation model.

No radial width, response rescaling, field mixing, empirical label, continuum
parameter, or MACE weight is adjusted here.  A nonzero permanent source is
intentionally outside this response-only kernel until it has its own
source/dual provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
)
from .route2_v0_atomic_displacement_response import (
    BOHR_ANGSTROM,
    Route2V0AtomicDisplacementResponseTable,
    atomic_induced_dipoles,
)

V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_CONSTRUCTION = (
    "route2-v0-atomic-displacement-reduced-kkt-v1"
)


def _finite_vector(
    values: np.ndarray,
    *,
    name: str,
    length: int,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite with shape ({length},).")
    return vector


def _finite_matrix(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return matrix


def _immutable(values: np.ndarray, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_positive_integer_vector(
    values: np.ndarray,
    *,
    name: str,
    length: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite with shape ({length},).")
    numeric = np.asarray(raw, dtype=float)
    integer = np.rint(numeric)
    if np.any(integer != numeric) or np.any(integer < 1.0):
        raise ValueError(f"{name} must contain positive integers.")
    result = integer.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _relative_matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _infinity_norm(values: np.ndarray) -> float:
    """Return a deterministic infinity norm without importing test machinery."""

    return float(np.max(np.abs(values)))


def _symmetric_positive_definite(
    values: np.ndarray,
    *,
    name: str,
    relative_tolerance: float,
) -> tuple[np.ndarray, float]:
    matrix = _finite_matrix(values, name=name, shape=(3, 3))
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("Relative tolerance must be finite and positive.")
    antisymmetry = float(np.linalg.norm(0.5 * (matrix - matrix.T), ord=2))
    if antisymmetry > relative_tolerance * _relative_matrix_scale(matrix):
        raise ValueError(f"{name} is not reciprocal in its declared pairing.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    threshold = relative_tolerance * _relative_matrix_scale(symmetric)
    minimum = float(np.min(eigenvalues))
    if minimum <= threshold:
        raise ValueError(f"{name} is not positive definite.")
    return symmetric, antisymmetry


@dataclass(frozen=True)
class AtomicDisplacementSurfaceCoupling:
    """One exact matrix source/dual pairing for an induced V0-ADT source.

    ``surface_operator_hartree_per_e_per_ebohr`` is constructed by applying
    the physical real-space source to each Cartesian unit molecular dipole.
    Its transpose is the only permitted surface-to-dipole dual map.
    """

    response_table: Route2V0AtomicDisplacementResponseTable
    atomic_numbers: np.ndarray
    atom_positions_angstrom: np.ndarray
    atomic_dipole_weights: np.ndarray
    surface_points_bohr: np.ndarray
    surface_operator_hartree_per_e_per_ebohr: np.ndarray = field(init=False)
    atomic_partition_identity_error: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.response_table, Route2V0AtomicDisplacementResponseTable):
            raise TypeError(
                "response_table must be a Route2V0AtomicDisplacementResponseTable."
            )
        positions = np.asarray(self.atom_positions_angstrom, dtype=float)
        points = np.asarray(self.surface_points_bohr, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("Atomic positions must be finite with shape (n_atoms, 3).")
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("Surface points must be finite with shape (n_surface, 3).")
        atomic_numbers = _immutable_positive_integer_vector(
            self.atomic_numbers,
            name="Atomic numbers",
            length=len(positions),
        )
        weights = _finite_matrix(
            self.atomic_dipole_weights,
            name="Atomic dipole weights",
            shape=(len(positions), 3, 3),
        )
        partition_error = float(
            np.linalg.norm(np.sum(weights, axis=0) - np.eye(3), ord="fro")
        )
        if partition_error > 1.0e-10:
            raise ValueError(
                "Atomic dipole weights do not preserve the molecular dipole."
            )
        unsupported = sorted(
            set(atomic_numbers.tolist())
            - set(self.response_table.supported_atomic_numbers)
        )
        if unsupported:
            raise ValueError(
                "Atomic-displacement response table lacks elements " f"{unsupported}."
            )

        operator = np.column_stack(
            [
                self.response_table.induced_potential(
                    points,
                    atomic_numbers,
                    positions,
                    weights,
                    np.eye(3)[index],
                )
                for index in range(3)
            ]
        )
        if operator.shape != (len(points), 3) or not np.all(np.isfinite(operator)):
            raise RuntimeError("Atomic-displacement surface operator is invalid.")
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(
            self,
            "atom_positions_angstrom",
            _immutable(
                positions,
                name="atom_positions_angstrom",
                shape=positions.shape,
            ),
        )
        object.__setattr__(
            self,
            "atomic_dipole_weights",
            _immutable(
                weights,
                name="atomic_dipole_weights",
                shape=weights.shape,
            ),
        )
        object.__setattr__(
            self,
            "surface_points_bohr",
            _immutable(points, name="surface_points_bohr", shape=points.shape),
        )
        object.__setattr__(
            self,
            "surface_operator_hartree_per_e_per_ebohr",
            _immutable(
                operator,
                name="surface_operator_hartree_per_e_per_ebohr",
                shape=operator.shape,
            ),
        )
        object.__setattr__(self, "atomic_partition_identity_error", partition_error)

    @property
    def atom_count(self) -> int:
        return len(self.atomic_numbers)

    @property
    def surface_point_count(self) -> int:
        return len(self.surface_points_bohr)

    def surface_potential(
        self,
        molecular_induced_dipole_ebohr: np.ndarray,
    ) -> np.ndarray:
        """Map one molecular induced dipole to its physical surface potential."""

        dipole = _finite_vector(
            molecular_induced_dipole_ebohr,
            name="Molecular induced dipole",
            length=3,
        )
        return self.surface_operator_hartree_per_e_per_ebohr @ dipole

    def surface_to_dipole_dual(self, surface_charge_e: np.ndarray) -> np.ndarray:
        """Apply the exact transpose of :meth:`surface_potential`."""

        charge = _finite_vector(
            surface_charge_e,
            name="Surface charge",
            length=self.surface_point_count,
        )
        return self.surface_operator_hartree_per_e_per_ebohr.T @ charge

    def atomic_induced_dipoles(
        self,
        molecular_induced_dipole_ebohr: np.ndarray,
    ) -> np.ndarray:
        """Return the frozen atom partition used by the source map."""

        return atomic_induced_dipoles(
            self.atomic_dipole_weights,
            molecular_induced_dipole_ebohr,
        )


@dataclass(frozen=True)
class Route2V0AtomicDisplacementReducedKKTState:
    """One fixed-geometry stationary induced-response state.

    The state is intentionally response-only: it contains no permanent density
    or total-solvation ledger.  It does certify that the frozen induced source,
    electronic quadratic form, continuum response, and external dipole work
    are derivatives of one reduced scalar.
    """

    induced_dipole_ebohr: np.ndarray
    external_dipole_dual_hartree_per_ebohr: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    surface_charge_e: np.ndarray
    surface_operator_hartree_per_e_per_ebohr: np.ndarray
    electronic_curvature_hartree_per_ebohr2: np.ndarray
    continuum_curvature_hartree_per_ebohr2: np.ndarray
    joint_curvature_hartree_per_ebohr2: np.ndarray
    external_dipole_response_bohr3: np.ndarray
    electronic_induction_energy_hartree: float
    continuum_polarization_energy_hartree: float
    external_work_hartree: float
    stationary_total_energy_hartree: float
    electronic_polarizability_antisymmetry_norm_bohr3: float
    continuum_restricted_antisymmetry_norm_hartree_per_ebohr2: float
    source_duality_error_hartree: float
    stationarity_residual_inf: float
    stationarity_tolerance_hartree_per_ebohr: float
    electronic_minimum_curvature: float
    electronic_stability_threshold: float
    joint_minimum_curvature: float
    joint_stability_threshold: float
    construction: str = V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_CONSTRUCTION

    def __post_init__(self) -> None:
        surface_count = len(np.asarray(self.surface_potential_hartree_per_e))
        if surface_count == 0:
            raise ValueError("Surface potential must contain at least one value.")
        for name, shape in (
            ("induced_dipole_ebohr", (3,)),
            ("external_dipole_dual_hartree_per_ebohr", (3,)),
            ("surface_potential_hartree_per_e", (surface_count,)),
            ("surface_charge_e", (surface_count,)),
            ("surface_operator_hartree_per_e_per_ebohr", (surface_count, 3)),
            ("electronic_curvature_hartree_per_ebohr2", (3, 3)),
            ("continuum_curvature_hartree_per_ebohr2", (3, 3)),
            ("joint_curvature_hartree_per_ebohr2", (3, 3)),
            ("external_dipole_response_bohr3", (3, 3)),
        ):
            object.__setattr__(
                self,
                name,
                _immutable(getattr(self, name), name=name, shape=shape),
            )
        if self.construction != V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_CONSTRUCTION:
            raise ValueError(
                "Unsupported atomic-displacement reduced-KKT construction."
            )
        for name in (
            "electronic_induction_energy_hartree",
            "continuum_polarization_energy_hartree",
            "external_work_hartree",
            "stationary_total_energy_hartree",
            "electronic_polarizability_antisymmetry_norm_bohr3",
            "continuum_restricted_antisymmetry_norm_hartree_per_ebohr2",
            "source_duality_error_hartree",
            "stationarity_residual_inf",
            "stationarity_tolerance_hartree_per_ebohr",
            "electronic_minimum_curvature",
            "electronic_stability_threshold",
            "joint_minimum_curvature",
            "joint_stability_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if (
                name.endswith("norm_bohr3")
                or name.endswith("error_hartree")
                or name.endswith("residual_inf")
                or name.endswith("tolerance_hartree_per_ebohr")
                or name.endswith("stability_threshold")
            ):
                if value < 0.0:
                    raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        for name in (
            "electronic_curvature_hartree_per_ebohr2",
            "continuum_curvature_hartree_per_ebohr2",
            "joint_curvature_hartree_per_ebohr2",
            "external_dipole_response_bohr3",
        ):
            matrix = getattr(self, name)
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"{name} must be symmetric.")
        if not np.allclose(
            self.joint_curvature_hartree_per_ebohr2,
            self.electronic_curvature_hartree_per_ebohr2
            + self.continuum_curvature_hartree_per_ebohr2,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Joint curvature must equal electronic plus continuum curvature."
            )
        expected_surface = (
            self.surface_operator_hartree_per_e_per_ebohr
            @ self.induced_dipole_ebohr
        )
        if not np.allclose(
            self.surface_potential_hartree_per_e,
            expected_surface,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Surface potential must come from the same V0-ADT source map."
            )
        source_dual = (
            self.surface_operator_hartree_per_e_per_ebohr.T @ self.surface_charge_e
        )
        expected_continuum_dual = (
            self.continuum_curvature_hartree_per_ebohr2 @ self.induced_dipole_ebohr
        )
        dual_scale = max(
            1.0,
            float(np.linalg.norm(source_dual)),
            float(np.linalg.norm(expected_continuum_dual)),
        )
        if not np.allclose(
            source_dual,
            expected_continuum_dual,
            rtol=0.0,
            atol=1.0e-10 * dual_scale,
        ):
            raise ValueError(
                "Surface charge and continuum curvature do not share one source dual."
            )
        expected_residual = (
            self.electronic_curvature_hartree_per_ebohr2 @ self.induced_dipole_ebohr
            + source_dual
            + self.external_dipole_dual_hartree_per_ebohr
        )
        expected_residual_inf = _infinity_norm(expected_residual)
        residual_check_tolerance = max(
            1.0e-12,
            1.0e-10
            * max(
                1.0,
                float(
                    np.linalg.norm(
                        self.electronic_curvature_hartree_per_ebohr2
                        @ self.induced_dipole_ebohr
                    )
                ),
                float(np.linalg.norm(source_dual)),
                float(np.linalg.norm(self.external_dipole_dual_hartree_per_ebohr)),
            ),
        )
        if not math.isclose(
            self.stationarity_residual_inf,
            expected_residual_inf,
            rel_tol=0.0,
            abs_tol=residual_check_tolerance,
        ):
            raise ValueError(
                "Stationarity residual must match the common scalar derivative."
            )
        if (
            self.stationarity_residual_inf
            > self.stationarity_tolerance_hartree_per_ebohr
        ):
            raise ValueError(
                "Reduced-KKT state is not stationary within its declared tolerance."
            )
        expected_source_duality_error = abs(
            float(
                self.surface_potential_hartree_per_e @ self.surface_charge_e
                - self.induced_dipole_ebohr @ source_dual
            )
        )
        if not math.isclose(
            self.source_duality_error_hartree,
            expected_source_duality_error,
            rel_tol=0.0,
            abs_tol=max(1.0e-12, 1.0e-10 * dual_scale),
        ):
            raise ValueError(
                "Source duality error must match the exact source transpose."
            )
        expected_electronic = 0.5 * float(
            self.induced_dipole_ebohr
            @ self.electronic_curvature_hartree_per_ebohr2
            @ self.induced_dipole_ebohr
        )
        expected_continuum = 0.5 * float(
            self.surface_potential_hartree_per_e @ self.surface_charge_e
        )
        expected_external = float(
            self.induced_dipole_ebohr @ self.external_dipole_dual_hartree_per_ebohr
        )
        tolerance = max(
            1.0e-12,
            1.0e-10
            * max(
                1.0,
                abs(expected_electronic),
                abs(expected_continuum),
                abs(expected_external),
            ),
        )
        if (
            abs(self.electronic_induction_energy_hartree - expected_electronic)
            > tolerance
            or abs(self.continuum_polarization_energy_hartree - expected_continuum)
            > tolerance
            or abs(self.external_work_hartree - expected_external) > tolerance
            or abs(
                self.stationary_total_energy_hartree
                - (expected_electronic + expected_continuum + expected_external)
            )
            > tolerance
        ):
            raise ValueError("Reduced-KKT energy ledger is inconsistent.")
        electronic_minimum = float(
            np.min(np.linalg.eigvalsh(self.electronic_curvature_hartree_per_ebohr2))
        )
        if not math.isclose(
            self.electronic_minimum_curvature,
            electronic_minimum,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("Electronic minimum curvature is inconsistent.")
        if self.electronic_minimum_curvature <= self.electronic_stability_threshold:
            raise ValueError(
                "Electronic induced-dipole curvature is not positive definite."
            )
        joint_minimum = float(
            np.min(np.linalg.eigvalsh(self.joint_curvature_hartree_per_ebohr2))
        )
        if not math.isclose(
            self.joint_minimum_curvature,
            joint_minimum,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("Joint minimum curvature is inconsistent.")
        if self.joint_minimum_curvature <= self.joint_stability_threshold:
            raise ValueError("Joint reduced-KKT curvature is not positive definite.")
        inverse_check = (
            self.joint_curvature_hartree_per_ebohr2
            @ self.external_dipole_response_bohr3
            + np.eye(3)
        )
        if _infinity_norm(inverse_check) > 1.0e-10 * _relative_matrix_scale(
            self.joint_curvature_hartree_per_ebohr2
        ):
            raise ValueError("External dipole response is not the KKT Hessian inverse.")
        if np.max(np.linalg.eigvalsh(self.external_dipole_response_bohr3)) > 1.0e-10:
            raise ValueError("External dipole response must be passive.")


def _validated_continuum(
    coupling: AtomicDisplacementSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
) -> None:
    if (
        getattr(continuum, "contract_version", None)
        != EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
    ):
        raise ValueError("Unsupported external-MEP continuum-response contract.")
    if not bool(getattr(continuum, "energy_response_is_reciprocal", False)):
        raise ValueError("Atomic-displacement KKT requires a reciprocal continuum.")
    if int(getattr(continuum, "atom_count", -1)) != coupling.atom_count:
        raise ValueError("Continuum atom count does not match the V0-ADT source.")
    continuum_atomic_numbers = _immutable_positive_integer_vector(
        getattr(continuum, "atomic_numbers", None),
        name="Continuum atomic numbers",
        length=coupling.atom_count,
    )
    if not np.array_equal(continuum_atomic_numbers, coupling.atomic_numbers):
        raise ValueError("Continuum atomic numbers do not match the V0-ADT source.")
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
        raise ValueError("Continuum geometry or surface points do not match V0-ADT.")


def _continuum_columns(
    coupling: AtomicDisplacementSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
) -> np.ndarray:
    columns = np.column_stack(
        [
            _finite_vector(
                continuum.apply_energy_conjugate(
                    coupling.surface_operator_hartree_per_e_per_ebohr[:, index]
                ),
                name="Energy-conjugate surface charge",
                length=coupling.surface_point_count,
            )
            for index in range(3)
        ]
    )
    return columns


def solve_route2_v0_atomic_displacement_reduced_kkt(
    *,
    coupling: AtomicDisplacementSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
    polarizability_bohr3: np.ndarray,
    external_dipole_dual_hartree_per_ebohr: np.ndarray | None = None,
    polarizability_reciprocity_relative_tolerance: float = 1.0e-12,
    continuum_reciprocity_relative_tolerance: float = 1.0e-10,
    stability_relative_tolerance: float = 1.0e-10,
    stationarity_relative_tolerance: float = 1.0e-10,
) -> Route2V0AtomicDisplacementReducedKKTState:
    """Solve the response-only V0-ADT reduced stationary system.

    The external dual is optional because a zero-field state is a valid
    stationary diagnostic. It is deliberately a three-component molecular
    dipole dual, not a hidden high-dimensional field adapter.
    """

    if not isinstance(coupling, AtomicDisplacementSurfaceCoupling):
        raise TypeError("coupling must be an AtomicDisplacementSurfaceCoupling.")
    _validated_continuum(coupling, continuum)
    alpha, alpha_antisymmetry = _symmetric_positive_definite(
        polarizability_bohr3,
        name="Electronic polarizability",
        relative_tolerance=polarizability_reciprocity_relative_tolerance,
    )
    if (
        not math.isfinite(continuum_reciprocity_relative_tolerance)
        or continuum_reciprocity_relative_tolerance <= 0.0
        or not math.isfinite(stability_relative_tolerance)
        or stability_relative_tolerance <= 0.0
        or not math.isfinite(stationarity_relative_tolerance)
        or stationarity_relative_tolerance <= 0.0
    ):
        raise ValueError("Reduced-KKT tolerances must be finite and positive.")
    external = (
        np.zeros(3)
        if external_dipole_dual_hartree_per_ebohr is None
        else _finite_vector(
            external_dipole_dual_hartree_per_ebohr,
            name="External dipole dual",
            length=3,
        )
    )
    electronic_curvature = np.linalg.solve(alpha, np.eye(3))
    electronic_curvature = 0.5 * (electronic_curvature + electronic_curvature.T)
    electronic_eigenvalues = np.linalg.eigvalsh(electronic_curvature)
    electronic_minimum = float(np.min(electronic_eigenvalues))
    electronic_stability_threshold = (
        stability_relative_tolerance * _relative_matrix_scale(electronic_curvature)
    )
    if electronic_minimum <= electronic_stability_threshold:
        raise RuntimeError(
            "Electronic induced-dipole curvature is not positive definite."
        )

    response_columns = _continuum_columns(coupling, continuum)
    raw_continuum_curvature = (
        coupling.surface_operator_hartree_per_e_per_ebohr.T @ response_columns
    )
    continuum_antisymmetry = float(
        np.linalg.norm(
            0.5 * (raw_continuum_curvature - raw_continuum_curvature.T),
            ord=2,
        )
    )
    continuum_reciprocity_threshold = (
        continuum_reciprocity_relative_tolerance
        * _relative_matrix_scale(raw_continuum_curvature)
    )
    if continuum_antisymmetry > continuum_reciprocity_threshold:
        raise RuntimeError("Continuum is not reciprocal in the V0-ADT source subspace.")
    continuum_curvature = 0.5 * (raw_continuum_curvature + raw_continuum_curvature.T)
    joint_curvature = electronic_curvature + continuum_curvature
    joint_curvature = 0.5 * (joint_curvature + joint_curvature.T)
    joint_eigenvalues = np.linalg.eigvalsh(joint_curvature)
    joint_minimum = float(np.min(joint_eigenvalues))
    stability_threshold = stability_relative_tolerance * _relative_matrix_scale(
        joint_curvature
    )
    if joint_minimum <= stability_threshold:
        raise RuntimeError("V0-ADT reduced-KKT curvature is not positive definite.")

    induced = np.linalg.solve(joint_curvature, -external)
    potential = coupling.surface_potential(induced)
    surface_charge = _finite_vector(
        continuum.apply_energy_conjugate(potential),
        name="Energy-conjugate surface charge",
        length=coupling.surface_point_count,
    )
    source_dual = coupling.surface_to_dipole_dual(surface_charge)
    residual = electronic_curvature @ induced + source_dual + external
    residual_inf = _infinity_norm(residual)
    stationarity_threshold = stationarity_relative_tolerance * max(
        1.0,
        float(np.linalg.norm(electronic_curvature @ induced)),
        float(np.linalg.norm(source_dual)),
        float(np.linalg.norm(external)),
    )
    if residual_inf > stationarity_threshold:
        raise RuntimeError("V0-ADT reduced-KKT stationarity failed.")
    source_duality_error = abs(
        float(potential @ surface_charge - induced @ source_dual)
    )
    external_response = -np.linalg.solve(joint_curvature, np.eye(3))
    external_response = 0.5 * (external_response + external_response.T)
    return Route2V0AtomicDisplacementReducedKKTState(
        induced_dipole_ebohr=induced,
        external_dipole_dual_hartree_per_ebohr=external,
        surface_potential_hartree_per_e=potential,
        surface_charge_e=surface_charge,
        surface_operator_hartree_per_e_per_ebohr=(
            coupling.surface_operator_hartree_per_e_per_ebohr
        ),
        electronic_curvature_hartree_per_ebohr2=electronic_curvature,
        continuum_curvature_hartree_per_ebohr2=continuum_curvature,
        joint_curvature_hartree_per_ebohr2=joint_curvature,
        external_dipole_response_bohr3=external_response,
        electronic_induction_energy_hartree=0.5
        * float(induced @ electronic_curvature @ induced),
        continuum_polarization_energy_hartree=0.5 * float(potential @ surface_charge),
        external_work_hartree=float(induced @ external),
        stationary_total_energy_hartree=(
            0.5 * float(induced @ electronic_curvature @ induced)
            + 0.5 * float(potential @ surface_charge)
            + float(induced @ external)
        ),
        electronic_polarizability_antisymmetry_norm_bohr3=alpha_antisymmetry,
        continuum_restricted_antisymmetry_norm_hartree_per_ebohr2=(
            continuum_antisymmetry
        ),
        source_duality_error_hartree=source_duality_error,
        stationarity_residual_inf=residual_inf,
        stationarity_tolerance_hartree_per_ebohr=stationarity_threshold,
        electronic_minimum_curvature=electronic_minimum,
        electronic_stability_threshold=electronic_stability_threshold,
        joint_minimum_curvature=joint_minimum,
        joint_stability_threshold=stability_threshold,
    )


__all__ = [
    "V0_ATOMIC_DISPLACEMENT_REDUCED_KKT_CONSTRUCTION",
    "AtomicDisplacementSurfaceCoupling",
    "Route2V0AtomicDisplacementReducedKKTState",
    "solve_route2_v0_atomic_displacement_reduced_kkt",
]
