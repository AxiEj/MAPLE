"""Direct-sum frozen-source KKT gate for the no-training Route-2 V0 path.

The frozen MACE-POLAR point monopole/dipole source and the V0 atomic-
displacement tangent are represented in one *distributional* source space,
not by fitting the latter into an arbitrary Gaussian radial channel.  At fixed
geometry, write the surface potential as

``v(c0, p) = S_R @ c0 + B_R @ p``.

``c0`` is the unchanged zero-field MACE source and remains fixed.  ``p`` is
an induced molecular dipole whose physical V0-ADT source has the positive gas
curvature ``alpha^-1``.  With an energy-conjugate reciprocal continuum
response ``q = Q_R v``, the only scalar minimized here is

``g_R(p; c0) = 0.5 p.T alpha^-1 p + 0.5 v(c0, p).T Q_R v(c0, p)``.

This is an exact source/dual direct sum: ``S_R.T`` and ``B_R.T`` are generated
from the same surface matrices as the forward maps.  It is deliberately not a
full electronic density functional: the permanent MACE source is frozen, and
this fixed-geometry structural gate has no physical V0 cavity, nonpolar term,
force, PES, or solvation-accuracy claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .continuum_response import ExternalMEPCavityResponse
from .gto_density import point_multipole_potential
from .route2_v0_atomic_displacement_kkt import (
    AtomicDisplacementSurfaceCoupling,
    Route2V0AtomicDisplacementReducedKKTState,
    solve_route2_v0_atomic_displacement_reduced_kkt,
    validate_atomic_displacement_continuum,
)

V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_CONSTRUCTION = (
    "route2-v0-atomic-displacement-frozen-kkt-v1"
)


def _immutable(
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


def _relative_matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _infinity_norm(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


@dataclass(frozen=True)
class PointMultipoleSurfaceCoupling:
    """Exact point-multipole source matrix and transpose at one surface.

    The coefficient layout is the frozen raw MACE-POLAR ``(atom, [q,l1])``
    layout consumed by :func:`point_multipole_potential`.  The matrix is built
    column-by-column from that exact routine, so its transpose cannot drift
    from the permanent source convention.
    """

    atom_positions_angstrom: np.ndarray
    surface_points_bohr: np.ndarray
    surface_operator_hartree_per_e_per_coefficient: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        positions = np.asarray(self.atom_positions_angstrom, dtype=float)
        points = np.asarray(self.surface_points_bohr, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "Point-multipole positions must be finite with shape (n_atoms, 3)."
            )
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "Point-multipole surface points must be finite with shape "
                "(n_surface, 3)."
            )
        coefficient_count = 4 * len(positions)
        columns: list[np.ndarray] = []
        for index in range(coefficient_count):
            coefficient = np.zeros((len(positions), 4), dtype=float)
            coefficient.reshape(-1)[index] = 1.0
            columns.append(point_multipole_potential(points, positions, coefficient))
        operator = np.column_stack(columns)
        if operator.shape != (len(points), coefficient_count):
            raise RuntimeError("Point-multipole surface operator has an invalid shape.")
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
            "surface_points_bohr",
            _immutable(points, name="surface_points_bohr", shape=points.shape),
        )
        object.__setattr__(
            self,
            "surface_operator_hartree_per_e_per_coefficient",
            _immutable(
                operator,
                name="surface_operator_hartree_per_e_per_coefficient",
                shape=operator.shape,
            ),
        )

    @property
    def atom_count(self) -> int:
        return self.atom_positions_angstrom.shape[0]

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return (self.atom_count, 4)

    @property
    def surface_point_count(self) -> int:
        return self.surface_points_bohr.shape[0]

    def coefficients(self, values: np.ndarray, *, name: str) -> np.ndarray:
        return _immutable(values, name=name, shape=self.coefficient_shape)

    def surface_potential(self, coefficients: np.ndarray) -> np.ndarray:
        """Map raw frozen MACE coefficients to the declared surface MEP."""

        source = self.coefficients(coefficients, name="Point-multipole coefficients")
        return self.surface_operator_hartree_per_e_per_coefficient @ source.reshape(-1)

    def surface_to_coefficient_dual(self, surface_charge_e: np.ndarray) -> np.ndarray:
        """Apply the exact transpose of :meth:`surface_potential`."""

        charge = _immutable(
            surface_charge_e,
            name="Point-multipole surface charge",
            shape=(self.surface_point_count,),
        )
        return (
            self.surface_operator_hartree_per_e_per_coefficient.T @ charge
        ).reshape(self.coefficient_shape)


@dataclass(frozen=True)
class AtomicDisplacementDirectSumSurfaceCoupling:
    """One permanent-point plus induced-ADT source/dual space at fixed geometry."""

    permanent: PointMultipoleSurfaceCoupling
    induced: AtomicDisplacementSurfaceCoupling
    surface_operator_hartree_per_e: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.permanent, PointMultipoleSurfaceCoupling):
            raise TypeError("permanent must be a PointMultipoleSurfaceCoupling.")
        if not isinstance(self.induced, AtomicDisplacementSurfaceCoupling):
            raise TypeError("induced must be an AtomicDisplacementSurfaceCoupling.")
        if (
            self.permanent.atom_count != self.induced.atom_count
            or not np.array_equal(
                self.permanent.atom_positions_angstrom,
                self.induced.atom_positions_angstrom,
            )
            or not np.array_equal(
                self.permanent.surface_points_bohr,
                self.induced.surface_points_bohr,
            )
        ):
            raise ValueError(
                "Permanent point source and induced ADT source must share one geometry "
                "and surface."
            )
        operator = np.column_stack(
            (
                self.permanent.surface_operator_hartree_per_e_per_coefficient,
                self.induced.surface_operator_hartree_per_e_per_ebohr,
            )
        )
        object.__setattr__(
            self,
            "surface_operator_hartree_per_e",
            _immutable(
                operator,
                name="Direct-sum surface operator",
                shape=operator.shape,
            ),
        )

    @property
    def atom_count(self) -> int:
        return self.permanent.atom_count

    @property
    def surface_point_count(self) -> int:
        return self.permanent.surface_point_count

    def surface_potential(
        self,
        frozen_density_coefficients: np.ndarray,
        induced_dipole_ebohr: np.ndarray,
    ) -> np.ndarray:
        """Return ``S c0 + B p`` from the direct-sum source space."""

        return (
            self.permanent.surface_potential(frozen_density_coefficients)
            + self.induced.surface_potential(induced_dipole_ebohr)
        )

    def surface_to_dual(
        self,
        surface_charge_e: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact direct-sum transpose ``(S.T q, B.T q)``."""

        return (
            self.permanent.surface_to_coefficient_dual(surface_charge_e),
            self.induced.surface_to_dipole_dual(surface_charge_e),
        )


@dataclass(frozen=True)
class Route2V0AtomicDisplacementFrozenKKTState:
    """One fixed-geometry stationary state with a frozen permanent source.

    Only the induced dipole is stationary.  ``frozen_density_coefficients`` is
    an external zero-field MACE source and is retained with its exact continuum
    dual solely to certify one direct-sum source pairing.
    """

    frozen_density_coefficients: np.ndarray
    induced_dipole_ebohr: np.ndarray
    frozen_surface_potential_hartree_per_e: np.ndarray
    induced_surface_potential_hartree_per_e: np.ndarray
    total_surface_potential_hartree_per_e: np.ndarray
    frozen_surface_charge_e: np.ndarray
    induced_surface_charge_e: np.ndarray
    total_surface_charge_e: np.ndarray
    frozen_density_coefficient_reaction_dual_hartree: np.ndarray
    induced_dipole_reaction_dual_hartree_per_ebohr: np.ndarray
    electronic_curvature_hartree_per_ebohr2: np.ndarray
    joint_curvature_hartree_per_ebohr2: np.ndarray
    external_dipole_response_bohr3: np.ndarray
    electronic_induction_energy_hartree: float
    frozen_continuum_energy_hartree: float
    total_continuum_energy_hartree: float
    stationary_total_energy_hartree: float
    direct_sum_duality_error_hartree: float
    continuum_direct_sum_antisymmetry_norm: float
    continuum_direct_sum_reciprocity_tolerance: float
    continuum_linearity_error_e: float
    continuum_linearity_tolerance_e: float
    stationarity_residual_inf: float
    stationarity_tolerance_hartree_per_ebohr: float
    construction: str = V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_CONSTRUCTION

    def __post_init__(self) -> None:
        density = _immutable(
            self.frozen_density_coefficients,
            name="Frozen density coefficients",
        )
        if density.ndim != 2 or density.shape[0] == 0 or density.shape[1] != 4:
            raise ValueError(
                "Frozen density coefficients must have shape (n_atoms, 4)."
            )
        surface_count = self.total_surface_potential_hartree_per_e.size
        if surface_count == 0:
            raise ValueError("Frozen KKT surface arrays must be nonempty.")
        for name, shape in (
            ("induced_dipole_ebohr", (3,)),
            ("frozen_surface_potential_hartree_per_e", (surface_count,)),
            ("induced_surface_potential_hartree_per_e", (surface_count,)),
            ("total_surface_potential_hartree_per_e", (surface_count,)),
            ("frozen_surface_charge_e", (surface_count,)),
            ("induced_surface_charge_e", (surface_count,)),
            ("total_surface_charge_e", (surface_count,)),
            ("frozen_density_coefficient_reaction_dual_hartree", density.shape),
            ("induced_dipole_reaction_dual_hartree_per_ebohr", (3,)),
            ("electronic_curvature_hartree_per_ebohr2", (3, 3)),
            ("joint_curvature_hartree_per_ebohr2", (3, 3)),
            ("external_dipole_response_bohr3", (3, 3)),
        ):
            object.__setattr__(
                self,
                name,
                _immutable(getattr(self, name), name=name, shape=shape),
            )
        object.__setattr__(self, "frozen_density_coefficients", density)
        if self.construction != V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_CONSTRUCTION:
            raise ValueError(
                "Unsupported frozen-source atomic-displacement KKT construction."
            )
        for name in (
            "electronic_induction_energy_hartree",
            "frozen_continuum_energy_hartree",
            "total_continuum_energy_hartree",
            "stationary_total_energy_hartree",
            "direct_sum_duality_error_hartree",
            "continuum_direct_sum_antisymmetry_norm",
            "continuum_direct_sum_reciprocity_tolerance",
            "continuum_linearity_error_e",
            "continuum_linearity_tolerance_e",
            "stationarity_residual_inf",
            "stationarity_tolerance_hartree_per_ebohr",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(
                    getattr(self, name),
                    name=name,
                    nonnegative=(
                        name.endswith("error_hartree")
                        or name.endswith("antisymmetry_norm")
                        or name.endswith("reciprocity_tolerance")
                        or name.endswith("error_e")
                        or name.endswith("tolerance_e")
                        or name.endswith("residual_inf")
                        or name.endswith("tolerance_hartree_per_ebohr")
                    ),
                ),
            )
        if not np.allclose(
            self.total_surface_potential_hartree_per_e,
            self.frozen_surface_potential_hartree_per_e
            + self.induced_surface_potential_hartree_per_e,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Total surface potential must equal frozen plus induced sources."
            )
        if not np.allclose(
            self.total_surface_charge_e,
            self.frozen_surface_charge_e + self.induced_surface_charge_e,
            rtol=0.0,
            atol=self.continuum_linearity_tolerance_e,
        ):
            raise ValueError(
                "Continuum response violates the declared linear direct sum."
            )
        if (
            self.continuum_direct_sum_antisymmetry_norm
            > self.continuum_direct_sum_reciprocity_tolerance
        ):
            raise ValueError(
                "Continuum is not reciprocal in the direct-sum source space."
            )
        for name in (
            "electronic_curvature_hartree_per_ebohr2",
            "joint_curvature_hartree_per_ebohr2",
            "external_dipole_response_bohr3",
        ):
            matrix = getattr(self, name)
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"{name} must be symmetric.")
        residual = (
            self.electronic_curvature_hartree_per_ebohr2 @ self.induced_dipole_ebohr
            + self.induced_dipole_reaction_dual_hartree_per_ebohr
        )
        if not math.isclose(
            self.stationarity_residual_inf,
            _infinity_norm(residual),
            rel_tol=0.0,
            abs_tol=max(1.0e-12, self.stationarity_tolerance_hartree_per_ebohr),
        ):
            raise ValueError("Frozen KKT stationarity residual is inconsistent.")
        if (
            self.stationarity_residual_inf
            > self.stationarity_tolerance_hartree_per_ebohr
        ):
            raise ValueError("Frozen KKT state is not stationary within its tolerance.")
        expected_electronic = 0.5 * float(
            self.induced_dipole_ebohr
            @ self.electronic_curvature_hartree_per_ebohr2
            @ self.induced_dipole_ebohr
        )
        expected_continuum = 0.5 * float(
            self.total_surface_potential_hartree_per_e @ self.total_surface_charge_e
        )
        tolerance = max(
            1.0e-12,
            1.0e-10 * max(1.0, abs(expected_electronic), abs(expected_continuum)),
        )
        if (
            abs(self.electronic_induction_energy_hartree - expected_electronic)
            > tolerance
            or abs(self.total_continuum_energy_hartree - expected_continuum)
            > tolerance
            or abs(
                self.stationary_total_energy_hartree
                - (expected_electronic + expected_continuum)
            )
            > tolerance
        ):
            raise ValueError("Frozen KKT energy ledger is inconsistent.")


def _surface_charge(
    continuum: ExternalMEPCavityResponse,
    potential: np.ndarray,
    *,
    surface_count: int,
) -> np.ndarray:
    charge = _immutable(
        continuum.apply_energy_conjugate(potential),
        name="Energy-conjugate surface charge",
        shape=(surface_count,),
    )
    return charge


def _direct_sum_response_columns(
    coupling: AtomicDisplacementDirectSumSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
) -> np.ndarray:
    """Apply the declared continuum to every permanent and induced source column."""

    source_operator = coupling.surface_operator_hartree_per_e
    return np.column_stack(
        [
            _surface_charge(
                continuum,
                source_operator[:, index],
                surface_count=coupling.surface_point_count,
            )
            for index in range(source_operator.shape[1])
        ]
    )


def _validate_direct_sum_continuum(
    coupling: AtomicDisplacementDirectSumSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
    *,
    reciprocity_relative_tolerance: float,
    linearity_relative_tolerance: float,
) -> tuple[np.ndarray, float, float]:
    """Require a homogeneous reciprocal response on the complete V0 source sum.

    The reduced V0-ADT kernel checks only ``B.T @ Q @ B``.  A common scalar
    with a frozen permanent source also requires the cross block
    ``S.T @ Q @ B`` to be its transpose.  Sampling every ``[S | B]`` column
    checks that condition without assigning a GTO projection to the ADT
    tangent.
    """

    zero_charge = _surface_charge(
        continuum,
        np.zeros(coupling.surface_point_count),
        surface_count=coupling.surface_point_count,
    )
    zero_tolerance = linearity_relative_tolerance * max(
        1.0,
        _infinity_norm(zero_charge),
    )
    if _infinity_norm(zero_charge) > zero_tolerance:
        raise RuntimeError(
            "Continuum response is not linear in the direct-sum source."
        )

    response_columns = _direct_sum_response_columns(coupling, continuum)
    raw_curvature = coupling.surface_operator_hartree_per_e.T @ response_columns
    antisymmetry = float(
        np.linalg.norm(0.5 * (raw_curvature - raw_curvature.T), ord=2)
    )
    tolerance = reciprocity_relative_tolerance * _relative_matrix_scale(
        raw_curvature
    )
    if antisymmetry > tolerance:
        raise RuntimeError(
            "Continuum is not reciprocal in the direct-sum source space."
        )
    return response_columns, antisymmetry, tolerance


def _validated_frozen_density(
    coupling: AtomicDisplacementDirectSumSurfaceCoupling,
    density_coefficients: np.ndarray,
    *,
    target_total_charge_e: float,
    total_charge_tolerance_e: float,
) -> np.ndarray:
    density = coupling.permanent.coefficients(
        density_coefficients,
        name="Frozen MACE density coefficients",
    )
    target = _finite_scalar(target_total_charge_e, name="Target total charge")
    tolerance = _finite_scalar(
        total_charge_tolerance_e,
        name="Total-charge tolerance",
        nonnegative=True,
    )
    if tolerance == 0.0:
        raise ValueError("Total-charge tolerance must be positive.")
    observed = float(np.sum(density[:, 0]))
    if abs(observed - target) > tolerance:
        raise ValueError(
            "Frozen MACE density violates its total-charge constraint "
            f"(observed={observed:.16e} e, target={target:.16e} e)."
        )
    return density


def solve_route2_v0_atomic_displacement_frozen_kkt(
    *,
    coupling: AtomicDisplacementDirectSumSurfaceCoupling,
    continuum: ExternalMEPCavityResponse,
    frozen_density_coefficients: np.ndarray,
    polarizability_bohr3: np.ndarray,
    target_total_charge_e: float = 0.0,
    total_charge_tolerance_e: float = 1.0e-10,
    continuum_linearity_relative_tolerance: float = 1.0e-10,
    continuum_reciprocity_relative_tolerance: float = 1.0e-10,
) -> Route2V0AtomicDisplacementFrozenKKTState:
    """Stationarize V0-ADT induction around one frozen permanent MACE source.

    This calls the reduced KKT gate with the permanent-source continuum dual as
    the external dipole dual.  It then independently checks the full direct-sum
    source and continuum linearity.  No MACE field update or density projection
    occurs.
    """

    if not isinstance(coupling, AtomicDisplacementDirectSumSurfaceCoupling):
        raise TypeError(
            "coupling must be an AtomicDisplacementDirectSumSurfaceCoupling."
        )
    if (
        not math.isfinite(continuum_linearity_relative_tolerance)
        or continuum_linearity_relative_tolerance <= 0.0
        or not math.isfinite(continuum_reciprocity_relative_tolerance)
        or continuum_reciprocity_relative_tolerance <= 0.0
    ):
        raise ValueError(
            "Direct-sum continuum tolerances must be finite and positive."
        )
    density = _validated_frozen_density(
        coupling,
        frozen_density_coefficients,
        target_total_charge_e=target_total_charge_e,
        total_charge_tolerance_e=total_charge_tolerance_e,
    )
    validate_atomic_displacement_continuum(coupling.induced, continuum)
    (
        response_columns,
        direct_sum_antisymmetry,
        direct_sum_reciprocity_tolerance,
    ) = _validate_direct_sum_continuum(
        coupling,
        continuum,
        reciprocity_relative_tolerance=continuum_reciprocity_relative_tolerance,
        linearity_relative_tolerance=continuum_linearity_relative_tolerance,
    )
    frozen_potential = coupling.permanent.surface_potential(density)
    frozen_charge = _surface_charge(
        continuum,
        frozen_potential,
        surface_count=coupling.surface_point_count,
    )
    _, permanent_to_induced_dual = coupling.surface_to_dual(
        frozen_charge
    )
    reduced_state: Route2V0AtomicDisplacementReducedKKTState = (
        solve_route2_v0_atomic_displacement_reduced_kkt(
            coupling=coupling.induced,
            continuum=continuum,
            polarizability_bohr3=polarizability_bohr3,
            external_dipole_dual_hartree_per_ebohr=permanent_to_induced_dual,
        )
    )
    induced_potential = reduced_state.surface_potential_hartree_per_e
    induced_charge = reduced_state.surface_charge_e
    total_potential = frozen_potential + induced_potential
    total_charge = _surface_charge(
        continuum,
        total_potential,
        surface_count=coupling.surface_point_count,
    )
    frozen_total_dual, induced_total_dual = coupling.surface_to_dual(total_charge)
    direct_sum_coefficients = np.concatenate(
        (density.reshape(-1), reduced_state.induced_dipole_ebohr)
    )
    predicted_total_charge = response_columns @ direct_sum_coefficients
    predicted_frozen_charge = (
        response_columns[:, : density.size] @ density.reshape(-1)
    )
    predicted_induced_charge = response_columns[:, density.size :] @ (
        reduced_state.induced_dipole_ebohr
    )
    linearity_error = max(
        _infinity_norm(total_charge - predicted_total_charge),
        _infinity_norm(frozen_charge - predicted_frozen_charge),
        _infinity_norm(induced_charge - predicted_induced_charge),
        _infinity_norm(total_charge - frozen_charge - induced_charge),
    )
    linearity_tolerance = continuum_linearity_relative_tolerance * max(
        1.0,
        _infinity_norm(total_charge),
        _infinity_norm(frozen_charge),
        _infinity_norm(induced_charge),
    )
    if linearity_error > linearity_tolerance:
        raise RuntimeError(
            "Continuum response is not linear in the direct-sum source."
        )
    stationarity = (
        reduced_state.electronic_curvature_hartree_per_ebohr2
        @ reduced_state.induced_dipole_ebohr
        + induced_total_dual
    )
    stationarity_residual = _infinity_norm(stationarity)
    stationarity_tolerance = max(
        reduced_state.stationarity_tolerance_hartree_per_ebohr,
        1.0e-10
        * max(
            1.0,
            _infinity_norm(
                reduced_state.electronic_curvature_hartree_per_ebohr2
                @ reduced_state.induced_dipole_ebohr
            ),
            _infinity_norm(induced_total_dual),
        ),
    )
    if stationarity_residual > stationarity_tolerance:
        raise RuntimeError("Frozen V0-ADT KKT stationarity failed.")
    direct_sum_duality_error = abs(
        float(
            total_potential @ total_charge
            - density.reshape(-1) @ frozen_total_dual.reshape(-1)
            - reduced_state.induced_dipole_ebohr @ induced_total_dual
        )
    )
    duality_tolerance = 1.0e-10 * max(
        1.0,
        abs(float(total_potential @ total_charge)),
        abs(float(density.reshape(-1) @ frozen_total_dual.reshape(-1))),
        abs(float(reduced_state.induced_dipole_ebohr @ induced_total_dual)),
    )
    if direct_sum_duality_error > duality_tolerance:
        raise RuntimeError("Direct-sum source/dual pairing failed.")
    frozen_energy = 0.5 * float(frozen_potential @ frozen_charge)
    total_continuum_energy = 0.5 * float(total_potential @ total_charge)
    electronic_energy = 0.5 * float(
        reduced_state.induced_dipole_ebohr
        @ reduced_state.electronic_curvature_hartree_per_ebohr2
        @ reduced_state.induced_dipole_ebohr
    )
    return Route2V0AtomicDisplacementFrozenKKTState(
        frozen_density_coefficients=density,
        induced_dipole_ebohr=reduced_state.induced_dipole_ebohr,
        frozen_surface_potential_hartree_per_e=frozen_potential,
        induced_surface_potential_hartree_per_e=induced_potential,
        total_surface_potential_hartree_per_e=total_potential,
        frozen_surface_charge_e=frozen_charge,
        induced_surface_charge_e=induced_charge,
        total_surface_charge_e=total_charge,
        frozen_density_coefficient_reaction_dual_hartree=frozen_total_dual,
        induced_dipole_reaction_dual_hartree_per_ebohr=induced_total_dual,
        electronic_curvature_hartree_per_ebohr2=(
            reduced_state.electronic_curvature_hartree_per_ebohr2
        ),
        joint_curvature_hartree_per_ebohr2=(
            reduced_state.joint_curvature_hartree_per_ebohr2
        ),
        external_dipole_response_bohr3=reduced_state.external_dipole_response_bohr3,
        electronic_induction_energy_hartree=electronic_energy,
        frozen_continuum_energy_hartree=frozen_energy,
        total_continuum_energy_hartree=total_continuum_energy,
        stationary_total_energy_hartree=electronic_energy + total_continuum_energy,
        direct_sum_duality_error_hartree=direct_sum_duality_error,
        continuum_direct_sum_antisymmetry_norm=direct_sum_antisymmetry,
        continuum_direct_sum_reciprocity_tolerance=(
            direct_sum_reciprocity_tolerance
        ),
        continuum_linearity_error_e=linearity_error,
        continuum_linearity_tolerance_e=linearity_tolerance,
        stationarity_residual_inf=stationarity_residual,
        stationarity_tolerance_hartree_per_ebohr=stationarity_tolerance,
    )


__all__ = [
    "AtomicDisplacementDirectSumSurfaceCoupling",
    "PointMultipoleSurfaceCoupling",
    "Route2V0AtomicDisplacementFrozenKKTState",
    "V0_ATOMIC_DISPLACEMENT_FROZEN_KKT_CONSTRUCTION",
    "solve_route2_v0_atomic_displacement_frozen_kkt",
]
