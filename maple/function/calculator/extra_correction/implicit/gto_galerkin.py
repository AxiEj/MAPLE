"""Same-basis atom-centred GTO/continuum Galerkin operators.

This module is a fixed-geometry research primitive.  It uses one normalized
``l<=1`` Gaussian basis for both the solute surface potential and the reaction
field dual, so the compressed continuum operator is

``P = S.T @ Q_energy @ S``.

``S`` maps GTO coefficients to the cavity MEP and ``Q_energy`` is the
energy-conjugate surface response supplied by the continuum provider.  The
module does not alter the current MACE-POLAR density space, public Route-2
profiles, forces, or solution-phase PES capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from ase.units import Bohr

from .continuum_response import (
    EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION,
    ExternalMEPCavityResponse,
)
from .electrostatic_pairing import MACE_POLAR_L1_PAIRING
from .gto_density import gaussian_multipole_potential

GTO_GALERKIN_LAYOUT = "atom-radial-raw-mace-l1-v1"


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AtomCenteredL1GTOBasis:
    """Normalized atom-centred Gaussian monopole/dipole basis."""

    sigmas_angstrom: tuple[float, ...]
    layout: str = GTO_GALERKIN_LAYOUT

    def __post_init__(self) -> None:
        sigmas = tuple(float(value) for value in self.sigmas_angstrom)
        if not sigmas or any(
            not math.isfinite(value) or value <= 0.0 for value in sigmas
        ):
            raise ValueError("GTO widths must be finite and positive.")
        if len(set(sigmas)) != len(sigmas):
            raise ValueError("GTO widths must be unique.")
        if self.layout != GTO_GALERKIN_LAYOUT:
            raise ValueError("Unsupported GTO Galerkin coefficient layout.")
        object.__setattr__(self, "sigmas_angstrom", sigmas)

    @property
    def radial_count(self) -> int:
        return len(self.sigmas_angstrom)

    def coefficient_shape(self, atom_count: int) -> tuple[int, int, int]:
        if isinstance(atom_count, bool) or int(atom_count) < 1:
            raise ValueError("GTO basis atom count must be positive.")
        return (int(atom_count), self.radial_count, 4)

    def validate_coefficients(
        self,
        values: np.ndarray,
        *,
        atom_count: int,
        name: str = "gto_coefficients",
    ) -> np.ndarray:
        return _immutable_array(
            values,
            name=name,
            shape=self.coefficient_shape(atom_count),
        )

    def flatten(
        self,
        values: np.ndarray,
        *,
        atom_count: int,
        name: str = "gto_coefficients",
    ) -> np.ndarray:
        coefficients = self.validate_coefficients(
            values,
            atom_count=atom_count,
            name=name,
        )
        return np.asarray(coefficients).reshape(-1).copy()

    def unflatten(
        self,
        values: np.ndarray,
        *,
        atom_count: int,
        name: str = "gto_dual",
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        shape = self.coefficient_shape(atom_count)
        if vector.shape != (int(np.prod(shape)),) or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{name} must be finite with shape {(int(np.prod(shape)),)}."
            )
        return vector.reshape(shape).copy()

    def surface_operator(
        self,
        surface_points_bohr: np.ndarray,
        atom_positions_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Return the dense source map ``S`` for one fixed geometry."""

        points = np.asarray(surface_points_bohr, dtype=float)
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "GTO Galerkin surface points must be finite with shape "
                "(n_surface, 3)."
            )
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "GTO Galerkin atom positions must be finite with shape " "(n_atoms, 3)."
            )

        shape = self.coefficient_shape(len(positions))
        operator = np.empty((len(points), int(np.prod(shape))), dtype=float)
        for atom_index, center in enumerate(positions):
            for radial_index, sigma in enumerate(self.sigmas_angstrom):
                for component in range(4):
                    unit = np.zeros((1, 4), dtype=float)
                    unit[0, component] = 1.0
                    column = np.ravel_multi_index(
                        (atom_index, radial_index, component),
                        shape,
                    )
                    operator[:, column] = gaussian_multipole_potential(
                        points,
                        center[None, :],
                        unit,
                        sigma_angstrom=sigma,
                    )
        operator.setflags(write=False)
        return operator

    def molecular_charge_dipole_constraints(
        self,
        atom_positions_angstrom: np.ndarray,
    ) -> np.ndarray:
        """Return ``A`` for exact total charge and Cartesian dipole moments."""

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] == 0
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "Constraint atom positions must be finite with shape " "(n_atoms, 3)."
            )
        shape = self.coefficient_shape(len(positions))
        constraints = np.zeros((4, int(np.prod(shape))), dtype=float)
        for atom_index, position in enumerate(positions):
            for radial_index in range(self.radial_count):
                monopole = np.ravel_multi_index(
                    (atom_index, radial_index, 0),
                    shape,
                )
                constraints[0, monopole] = 1.0
                constraints[1:, monopole] = position
                for cartesian_component, raw_component in enumerate(
                    MACE_POLAR_L1_PAIRING.density_to_field_indices[1:],
                    start=1,
                ):
                    constraints[
                        cartesian_component,
                        np.ravel_multi_index(
                            (atom_index, radial_index, raw_component),
                            shape,
                        ),
                    ] = 1.0
        constraints.setflags(write=False)
        return constraints


@dataclass(frozen=True)
class GTOGalerkinSnapshot:
    """One same-source/same-receiver continuum state."""

    coefficients: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    surface_charge_e: np.ndarray
    reaction_dual_hartree: np.ndarray
    polarization_energy_hartree: float
    half_coupling_identity_error_hartree: float

    def __post_init__(self) -> None:
        for name in (
            "coefficients",
            "surface_potential_hartree_per_e",
            "surface_charge_e",
            "reaction_dual_hartree",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), name=name),
            )
        for name in (
            "polarization_energy_hartree",
            "half_coupling_identity_error_hartree",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                name.endswith("error_hartree") and value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)


class FixedCavityGTOGalerkinOperator:
    """Compress one reciprocal continuum response into a GTO basis."""

    reciprocal_energy_pairing = True

    def __init__(
        self,
        response: ExternalMEPCavityResponse,
        atom_positions_angstrom: np.ndarray,
        basis: AtomCenteredL1GTOBasis,
        *,
        geometry_tolerance_angstrom: float = 1.0e-12,
        reciprocity_tolerance_hartree: float = 1.0e-10,
    ) -> None:
        if (
            getattr(response, "contract_version", None)
            != EXTERNAL_MEP_RESPONSE_CONTRACT_VERSION
        ):
            raise ValueError("Unsupported external-MEP continuum-response contract.")
        if not response.energy_response_is_reciprocal:
            raise ValueError(
                "GTO Galerkin compression requires a reciprocal "
                "energy-conjugate continuum response."
            )
        if (
            not math.isfinite(geometry_tolerance_angstrom)
            or geometry_tolerance_angstrom < 0.0
        ):
            raise ValueError("Geometry tolerance must be finite and nonnegative.")
        if (
            not math.isfinite(reciprocity_tolerance_hartree)
            or reciprocity_tolerance_hartree <= 0.0
        ):
            raise ValueError("Reciprocity tolerance must be finite and positive.")

        positions = np.asarray(atom_positions_angstrom, dtype=float)
        response_positions = (
            np.asarray(response.reference_positions_bohr, dtype=float) * Bohr
        )
        if (
            positions.shape != response_positions.shape
            or not np.all(np.isfinite(positions))
            or not np.allclose(
                positions,
                response_positions,
                rtol=0.0,
                atol=geometry_tolerance_angstrom,
            )
        ):
            raise ValueError(
                "GTO Galerkin geometry must match the fixed continuum cavity."
            )

        points = np.asarray(response.surface_points_bohr, dtype=float)
        surface_operator = basis.surface_operator(points, positions)
        response_columns = np.column_stack(
            [
                self._apply_response_static(
                    response,
                    surface_operator[:, index],
                    surface_size=len(points),
                )
                for index in range(surface_operator.shape[1])
            ]
        )
        raw_galerkin = surface_operator.T @ response_columns
        antisymmetric = 0.5 * (raw_galerkin - raw_galerkin.T)
        antisymmetric_norm = float(np.linalg.norm(antisymmetric, ord=2))
        scale = max(1.0, float(np.linalg.norm(raw_galerkin, ord=2)))
        if antisymmetric_norm > reciprocity_tolerance_hartree * scale:
            raise RuntimeError(
                "Continuum response violates the reciprocal GTO Galerkin " "contract."
            )
        galerkin = 0.5 * (raw_galerkin + raw_galerkin.T)

        self._response = response
        self._positions_angstrom = positions.copy()
        self._surface_points_bohr = points.copy()
        self.basis = basis
        self.atom_count = len(positions)
        self._surface_operator = np.array(surface_operator, copy=True)
        self._surface_response_columns = np.array(
            response_columns,
            copy=True,
        )
        self._galerkin_matrix = np.array(galerkin, copy=True)
        self._antisymmetric_norm_hartree = antisymmetric_norm
        for values in (
            self._positions_angstrom,
            self._surface_points_bohr,
            self._surface_operator,
            self._surface_response_columns,
            self._galerkin_matrix,
        ):
            values.setflags(write=False)

    @staticmethod
    def _apply_response_static(
        response: ExternalMEPCavityResponse,
        potential: np.ndarray,
        *,
        surface_size: int,
    ) -> np.ndarray:
        charge = np.asarray(
            response.apply_energy_conjugate(potential),
            dtype=float,
        )
        if charge.shape != (surface_size,) or not np.all(np.isfinite(charge)):
            raise RuntimeError(
                "Continuum surface response must be finite with one value per "
                "surface point."
            )
        return charge

    @property
    def coefficient_count(self) -> int:
        return self._galerkin_matrix.shape[0]

    @property
    def atom_positions_angstrom(self) -> np.ndarray:
        """Return the fixed atom positions bound to this cavity."""

        return self._positions_angstrom.copy()

    @property
    def surface_operator(self) -> np.ndarray:
        return self._surface_operator.copy()

    @property
    def galerkin_matrix_hartree(self) -> np.ndarray:
        return self._galerkin_matrix.copy()

    @property
    def antisymmetric_norm_hartree(self) -> float:
        return self._antisymmetric_norm_hartree

    @property
    def eigenvalues_hartree(self) -> np.ndarray:
        return np.linalg.eigvalsh(self._galerkin_matrix)

    @property
    def provenance(self) -> dict[str, object]:
        eigenvalues = self.eigenvalues_hartree
        return {
            "operator": "fixed-cavity-gto-galerkin-v1",
            "basis_layout": self.basis.layout,
            "sigmas_angstrom": list(self.basis.sigmas_angstrom),
            "source_receiver_basis_identical": True,
            "surface_response": "energy-conjugate",
            "atom_count": self.atom_count,
            "coefficient_count": self.coefficient_count,
            "surface_point_count": self._surface_operator.shape[0],
            "antisymmetric_norm_hartree": self.antisymmetric_norm_hartree,
            "maximum_eigenvalue_hartree": float(np.max(eigenvalues)),
            "minimum_eigenvalue_hartree": float(np.min(eigenvalues)),
            "fixed_geometry_only": True,
            "production_profile": False,
        }

    def _flat_coefficients(
        self,
        values: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        return self.basis.flatten(
            values,
            atom_count=self.atom_count,
            name=name,
        )

    def surface_potential(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        vector = self._flat_coefficients(
            coefficients,
            name="gto_coefficients",
        )
        return self._surface_operator @ vector

    def apply_surface_response(
        self,
        surface_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        potential = np.asarray(surface_potential_hartree_per_e, dtype=float)
        if potential.shape != (self._surface_operator.shape[0],) or not np.all(
            np.isfinite(potential)
        ):
            raise ValueError(
                "Surface potential must be finite with one value per point."
            )
        return self._apply_response_static(
            self._response,
            potential,
            surface_size=len(potential),
        )

    def surface_charge(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        vector = self._flat_coefficients(
            coefficients,
            name="gto_coefficients",
        )
        return self._surface_response_columns @ vector

    def surface_to_reaction_dual(
        self,
        surface_charge_e: np.ndarray,
    ) -> np.ndarray:
        charge = np.asarray(surface_charge_e, dtype=float)
        if charge.shape != (self._surface_operator.shape[0],) or not np.all(
            np.isfinite(charge)
        ):
            raise ValueError("Surface charge must be finite with one value per point.")
        return self.basis.unflatten(
            self._surface_operator.T @ charge,
            atom_count=self.atom_count,
            name="gto_reaction_dual",
        )

    def apply(
        self,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        vector = self._flat_coefficients(
            coefficients,
            name="gto_coefficients",
        )
        return self.basis.unflatten(
            self._galerkin_matrix @ vector,
            atom_count=self.atom_count,
            name="gto_reaction_dual",
        )

    def snapshot(
        self,
        coefficients: np.ndarray,
        *,
        identity_tolerance_hartree: float = 1.0e-10,
    ) -> GTOGalerkinSnapshot:
        if (
            not math.isfinite(identity_tolerance_hartree)
            or identity_tolerance_hartree <= 0.0
        ):
            raise ValueError("Energy identity tolerance must be finite and positive.")
        coefficient_block = self.basis.validate_coefficients(
            coefficients,
            atom_count=self.atom_count,
        )
        vector = np.asarray(coefficient_block).reshape(-1)
        potential = self._surface_operator @ vector
        charge = self.apply_surface_response(potential)
        reaction_dual_vector = self._galerkin_matrix @ vector
        surface_energy = 0.5 * float(np.dot(potential, charge))
        galerkin_energy = 0.5 * float(np.dot(vector, reaction_dual_vector))
        error = abs(surface_energy - galerkin_energy)
        if error > identity_tolerance_hartree:
            raise RuntimeError("GTO Galerkin half-coupling identity failed.")
        return GTOGalerkinSnapshot(
            coefficients=coefficient_block,
            surface_potential_hartree_per_e=potential,
            surface_charge_e=charge,
            reaction_dual_hartree=self.basis.unflatten(
                reaction_dual_vector,
                atom_count=self.atom_count,
            ),
            polarization_energy_hartree=surface_energy,
            half_coupling_identity_error_hartree=error,
        )
