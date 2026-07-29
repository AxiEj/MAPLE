"""Gas-phase uniform-field response of the no-training V0-Q QEq control.

This is a deliberately narrow physical diagnostic for the published-hardness,
monopole-only V0-Q tangent.  It evaluates the gas-phase quadratic curvature
in a uniform electric field without invoking MACE's learned field update,
fitting a scale, or adding a continuum term.  Its only purpose is to make the
curvature's molecular polarizability directly comparable with an independently
frozen QM finite-field calculation.

The control remains reference-shifted rather than full QEq: electronegativity
linear terms are absent, and all induced ``l=1`` channels remain unsupported.
It is not a public Route-2 response model or a solvation free-energy model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ase.units import Bohr
import numpy as np

from .route2_v0_qeq_monopole import (
    RAPPE_GODDARD_QEQ_TANGENT_CONSTRUCTION,
    SameBasisQEqHardnessCurvature,
    build_same_basis_qeq_hardness_curvature,
)


RAPPE_GODDARD_QEQ_GAS_RESPONSE_CONSTRUCTION = (
    "route2-v0-rappe-goddard-hardness-same-basis-monopole-gas-response-v1"
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
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _neutral_basis(atom_count: int) -> np.ndarray:
    if atom_count < 2:
        raise ValueError("Gas QEq response needs at least two monopole sites.")
    _, _, right_vectors = np.linalg.svd(
        np.ones((1, atom_count), dtype=float),
        full_matrices=True,
    )
    return right_vectors[1:, :].T.copy()


def _relative_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


@dataclass(frozen=True)
class QEqGasMonopoleFieldState:
    """One neutral monopole response to a uniform electric field in a.u."""

    field_hartree_per_e_bohr: np.ndarray
    induced_monopoles_e: np.ndarray
    induced_dipole_e_bohr: np.ndarray
    induction_energy_hartree: float
    charge_constraint_residual_e: float
    stationarity_residual_inf_hartree_per_e: float

    def __post_init__(self) -> None:
        field = _immutable_array(
            self.field_hartree_per_e_bohr,
            name="Uniform electric field",
            shape=(3,),
        )
        monopoles = _immutable_array(
            self.induced_monopoles_e,
            name="Induced monopoles",
        )
        dipole = _immutable_array(
            self.induced_dipole_e_bohr,
            name="Induced dipole",
            shape=(3,),
        )
        for name in (
            "induction_energy_hartree",
            "charge_constraint_residual_e",
            "stationarity_residual_inf_hartree_per_e",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                name != "induction_energy_hartree" and value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "field_hartree_per_e_bohr", field)
        object.__setattr__(self, "induced_monopoles_e", monopoles)
        object.__setattr__(self, "induced_dipole_e_bohr", dipole)


@dataclass(frozen=True)
class QEqGasMonopoleResponse:
    """Positive gas-phase response implied by one fixed QEq hardness matrix."""

    qeq_curvature: SameBasisQEqHardnessCurvature
    neutral_basis: np.ndarray
    dipole_map_e_bohr_per_e: np.ndarray
    polarizability_bohr3: np.ndarray
    electronic_minimum_neutral_curvature_hartree_per_e2: float
    stability_threshold_hartree_per_e2: float
    construction: str = RAPPE_GODDARD_QEQ_GAS_RESPONSE_CONSTRUCTION

    def __post_init__(self) -> None:
        atom_count = len(self.qeq_curvature.symbols)
        neutral_basis = _immutable_array(
            self.neutral_basis,
            name="Neutral monopole basis",
            shape=(atom_count, atom_count - 1),
        )
        if not np.allclose(
            np.ones(atom_count) @ neutral_basis,
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.allclose(
            neutral_basis.T @ neutral_basis,
            np.eye(atom_count - 1),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Neutral monopole basis must be orthonormal.")
        dipole_map = _immutable_array(
            self.dipole_map_e_bohr_per_e,
            name="Dipole map",
            shape=(3, atom_count),
        )
        polarizability = _immutable_array(
            self.polarizability_bohr3,
            name="QEq gas polarizability",
            shape=(3, 3),
        )
        if not np.allclose(polarizability, polarizability.T, rtol=0.0, atol=1.0e-12):
            raise ValueError("QEq gas polarizability must be symmetric.")
        if float(np.min(np.linalg.eigvalsh(polarizability))) < -1.0e-10:
            raise ValueError("QEq gas polarizability must be nonnegative.")
        minimum = float(self.electronic_minimum_neutral_curvature_hartree_per_e2)
        threshold = float(self.stability_threshold_hartree_per_e2)
        if (
            not math.isfinite(minimum)
            or not math.isfinite(threshold)
            or threshold <= 0.0
            or minimum <= threshold
        ):
            raise ValueError("Gas QEq curvature must be positive in neutral space.")
        if self.construction != RAPPE_GODDARD_QEQ_GAS_RESPONSE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 QEq gas-response construction.")
        object.__setattr__(self, "neutral_basis", neutral_basis)
        object.__setattr__(self, "dipole_map_e_bohr_per_e", dipole_map)
        object.__setattr__(self, "polarizability_bohr3", polarizability)
        object.__setattr__(
            self,
            "electronic_minimum_neutral_curvature_hartree_per_e2",
            minimum,
        )
        object.__setattr__(self, "stability_threshold_hartree_per_e2", threshold)

    def solve_uniform_field(
        self,
        field_hartree_per_e_bohr: np.ndarray,
    ) -> QEqGasMonopoleFieldState:
        """Minimize the neutral QEq tangent in one uniform external field."""

        field = _immutable_array(
            field_hartree_per_e_bohr,
            name="Uniform electric field",
            shape=(3,),
        )
        hardness = self.qeq_curvature.hardness_matrix_hartree_per_e2
        driving = self.dipole_map_e_bohr_per_e.T @ field
        reduced_hardness = self.neutral_basis.T @ hardness @ self.neutral_basis
        reduced_monopoles = np.linalg.solve(
            reduced_hardness,
            self.neutral_basis.T @ driving,
        )
        monopoles = self.neutral_basis @ reduced_monopoles
        dipole = self.dipole_map_e_bohr_per_e @ monopoles
        gradient = hardness @ monopoles - driving
        multiplier = -float(np.mean(gradient))
        residual = gradient + multiplier
        energy = 0.5 * float(monopoles @ hardness @ monopoles) - float(
            field @ dipole
        )
        expected_energy = -0.5 * float(field @ self.polarizability_bohr3 @ field)
        if abs(energy - expected_energy) > 1.0e-12 * max(1.0, abs(expected_energy)):
            raise RuntimeError("Gas QEq field energy disagrees with its polarizability.")
        if not np.allclose(
            dipole,
            self.polarizability_bohr3 @ field,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("Gas QEq field dipole disagrees with its polarizability.")
        return QEqGasMonopoleFieldState(
            field_hartree_per_e_bohr=field,
            induced_monopoles_e=monopoles,
            induced_dipole_e_bohr=dipole,
            induction_energy_hartree=energy,
            charge_constraint_residual_e=abs(float(np.sum(monopoles))),
            stationarity_residual_inf_hartree_per_e=float(
                np.max(np.abs(residual))
            ),
        )


def build_route2_v0_rappe_goddard_gas_monopole_response(
    symbols: tuple[str, ...] | list[str],
    atom_positions_angstrom: np.ndarray,
    *,
    stability_relative_tolerance: float = 1.0e-10,
) -> QEqGasMonopoleResponse:
    """Build the exact neutral QEq gas polarizability for one geometry.

    The dipole map uses coordinates relative to the arithmetic atom center.
    Because every response is exactly neutral, translating that origin leaves
    the response unchanged.  Fields are in ``Eh / (e a0)`` and the returned
    polarizability is in ``a0^3``.
    """

    if (
        not math.isfinite(stability_relative_tolerance)
        or stability_relative_tolerance <= 0.0
    ):
        raise ValueError("Gas QEq stability tolerance must be finite and positive.")
    curvature = build_same_basis_qeq_hardness_curvature(symbols, atom_positions_angstrom)
    atom_count = len(curvature.symbols)
    neutral_basis = _neutral_basis(atom_count)
    hardness = curvature.hardness_matrix_hartree_per_e2
    reduced_hardness = neutral_basis.T @ hardness @ neutral_basis
    reduced_hardness = 0.5 * (reduced_hardness + reduced_hardness.T)
    minimum = float(np.min(np.linalg.eigvalsh(reduced_hardness)))
    threshold = stability_relative_tolerance * _relative_scale(reduced_hardness)
    if minimum <= threshold:
        raise RuntimeError("Gas QEq curvature is not positive definite in neutral space.")
    positions_bohr = curvature.atom_positions_angstrom / Bohr
    centered_positions_bohr = positions_bohr - np.mean(positions_bohr, axis=0)
    dipole_map = centered_positions_bohr.T
    inverse_reduced_hardness = np.linalg.inv(reduced_hardness)
    polarizability = (
        dipole_map
        @ neutral_basis
        @ inverse_reduced_hardness
        @ neutral_basis.T
        @ dipole_map.T
    )
    polarizability = 0.5 * (polarizability + polarizability.T)
    return QEqGasMonopoleResponse(
        qeq_curvature=curvature,
        neutral_basis=neutral_basis,
        dipole_map_e_bohr_per_e=dipole_map,
        polarizability_bohr3=polarizability,
        electronic_minimum_neutral_curvature_hartree_per_e2=minimum,
        stability_threshold_hartree_per_e2=threshold,
    )


__all__ = [
    "RAPPE_GODDARD_QEQ_GAS_RESPONSE_CONSTRUCTION",
    "QEqGasMonopoleFieldState",
    "QEqGasMonopoleResponse",
    "build_route2_v0_rappe_goddard_gas_monopole_response",
]
