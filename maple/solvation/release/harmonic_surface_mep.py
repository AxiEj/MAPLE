"""Matched QM-MEP projection onto one harmonic continuum boundary chart.

The production harmonic continuum contains no laboratory-fixed surface grid.
This module supplies a *reference-only* numerical integration grid for
projecting an externally evaluated molecular electrostatic potential onto the
same finite weighted test space used by that continuum.  The grid is therefore
not part of the continuum definition and cannot admit rotational covariance,
energy, or force capabilities.

For every atom-centred sphere, the projector computes the real-harmonic
coefficients of the supplied potential through ``physical_lmax`` and then
applies the exact continuum weighted-basis transpose.  Consequently the final
right-hand side has exactly the same coefficient ordering and units as
``B(R)c``.  The supplied potential must be the total molecular MEP (nuclear
minus electronic) in Hartree per unit positive test charge; no fitted atomic
charges are introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np
from ase.units import Bohr

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.continuum.harmonic_coefficients import (
    HARMONIC_GALERKIN_MAXIMUM_LMAX,
    _real_harmonic_design,
)

HARMONIC_SURFACE_MEP_PROJECTION_CONTRACT_VERSION = (
    "route2-harmonic-surface-mep-reference-projection-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _readonly(values: object, *, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(shape).copy()
    result.setflags(write=False)
    return result


def _sphere_rule(polar_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic product rule used only for reference integration."""

    order = _positive_integer(polar_order, name="polar_order")
    if order > 512:
        raise ValueError("polar_order exceeds the bounded reference contract.")
    cosine, polar_weights = np.polynomial.legendre.leggauss(order)
    azimuthal_count = 2 * order + 1
    phi = 2.0 * np.pi * np.arange(azimuthal_count) / azimuthal_count
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    directions = np.stack(
        (
            np.repeat(sine, azimuthal_count) * np.tile(np.cos(phi), order),
            np.repeat(sine, azimuthal_count) * np.tile(np.sin(phi), order),
            np.repeat(cosine, azimuthal_count),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_count), azimuthal_count
    )
    return directions, weights


@dataclass(frozen=True, slots=True, init=False)
class HarmonicSurfaceMEPProjection:
    """One immutable potential projection bound to one projector identity."""

    projector_configuration_sha256: str
    potential_hartree_per_e_sha256: str
    raw_harmonic_coefficients_sha256: str
    boundary_rhs_sha256: str
    state_sha256: str
    _raw_values: tuple[float, ...]
    _rhs_values: tuple[float, ...]
    atom_count: int
    physical_dimension_per_atom: int
    boundary_dimension: int

    def __init__(
        self,
        *,
        projector_configuration_sha256: str,
        potential_hartree_per_e_sha256: str,
        raw_harmonic_coefficients: object,
        boundary_rhs: object,
        atom_count: int,
        physical_dimension_per_atom: int,
    ) -> None:
        projector_digest = _digest(
            projector_configuration_sha256,
            name="projector_configuration_sha256",
        )
        potential_digest = _digest(
            potential_hartree_per_e_sha256,
            name="potential_hartree_per_e_sha256",
        )
        count = _positive_integer(atom_count, name="atom_count")
        physical_dimension = _positive_integer(
            physical_dimension_per_atom,
            name="physical_dimension_per_atom",
        )
        raw = np.asarray(raw_harmonic_coefficients, dtype=float)
        rhs = np.asarray(boundary_rhs, dtype=float)
        if raw.shape != (count, physical_dimension) or not np.all(np.isfinite(raw)):
            raise ValueError(
                "raw_harmonic_coefficients must be finite with the projector shape."
            )
        if rhs.ndim != 1 or rhs.size < 1 or not np.all(np.isfinite(rhs)):
            raise ValueError("boundary_rhs must be a non-empty finite vector.")
        raw_digest = _array_sha256(raw)
        rhs_digest = _array_sha256(rhs)
        state = _canonical_sha256(
            {
                "contract_version": HARMONIC_SURFACE_MEP_PROJECTION_CONTRACT_VERSION,
                "projector_configuration_sha256": projector_digest,
                "potential_hartree_per_e_sha256": potential_digest,
                "raw_harmonic_coefficients_sha256": raw_digest,
                "boundary_rhs_sha256": rhs_digest,
            }
        )
        object.__setattr__(self, "projector_configuration_sha256", projector_digest)
        object.__setattr__(self, "potential_hartree_per_e_sha256", potential_digest)
        object.__setattr__(self, "raw_harmonic_coefficients_sha256", raw_digest)
        object.__setattr__(self, "boundary_rhs_sha256", rhs_digest)
        object.__setattr__(self, "state_sha256", state)
        object.__setattr__(
            self, "_raw_values", tuple(float(value) for value in raw.flat)
        )
        object.__setattr__(self, "_rhs_values", tuple(float(value) for value in rhs))
        object.__setattr__(self, "atom_count", count)
        object.__setattr__(self, "physical_dimension_per_atom", physical_dimension)
        object.__setattr__(self, "boundary_dimension", int(rhs.size))

    @property
    def raw_harmonic_coefficients_ev_per_e(self) -> np.ndarray:
        return _readonly(
            self._raw_values,
            shape=(self.atom_count, self.physical_dimension_per_atom),
        )

    @property
    def boundary_rhs(self) -> np.ndarray:
        return _readonly(self._rhs_values, shape=(self.boundary_dimension,))

    def as_dict(self, *, include_values: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "contract_version": HARMONIC_SURFACE_MEP_PROJECTION_CONTRACT_VERSION,
            "projector_configuration_sha256": self.projector_configuration_sha256,
            "potential_hartree_per_e_sha256": self.potential_hartree_per_e_sha256,
            "raw_harmonic_coefficients_sha256": (self.raw_harmonic_coefficients_sha256),
            "boundary_rhs_sha256": self.boundary_rhs_sha256,
            "state_sha256": self.state_sha256,
            "atom_count": self.atom_count,
            "physical_dimension_per_atom": self.physical_dimension_per_atom,
            "boundary_dimension": self.boundary_dimension,
            "capability_admitted": False,
        }
        if include_values:
            result["raw_harmonic_coefficients_ev_per_e"] = (
                self.raw_harmonic_coefficients_ev_per_e.tolist()
            )
            result["boundary_rhs"] = self.boundary_rhs.tolist()
        return result


@dataclass(frozen=True, slots=True, init=False)
class HarmonicSurfaceMEPProjector:
    """Reference-only grid and exact weighted-basis projection contract."""

    atom_count: int
    physical_lmax: int
    surface_lmax: int
    polar_order: int
    azimuthal_count: int
    points_per_atom: int
    boundary_dimension: int
    geometry_sha256: str
    continuum_configuration_sha256: str
    continuum_provenance_sha256: str
    topology_sha256: str
    cavity_profile_id: str
    weighted_basis_sha256: str
    points_bohr_sha256: str
    angular_weights_sha256: str
    configuration_sha256: str
    _position_values: tuple[float, ...]
    _radius_values: tuple[float, ...]
    _direction_values: tuple[float, ...]
    _weight_values: tuple[float, ...]
    _weighted_basis_values: tuple[float, ...]

    def __init__(
        self,
        *,
        positions_angstrom: object,
        radii_angstrom: object,
        physical_lmax: int,
        surface_lmax: int,
        polar_order: int,
        weighted_basis_operator: object,
        geometry_sha256: str,
        continuum_configuration_sha256: str,
        continuum_provenance_sha256: str,
        topology_sha256: str,
        cavity_profile_id: str,
    ) -> None:
        positions = np.asarray(positions_angstrom, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[0] < 1
            or positions.shape[1] != 3
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError(
                "positions_angstrom must be finite with shape (atom_count,3)."
            )
        count = int(positions.shape[0])
        radii = np.asarray(radii_angstrom, dtype=float)
        if (
            radii.shape != (count,)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "radii_angstrom must be finite and positive with one value per atom."
            )
        physical = _nonnegative_integer(physical_lmax, name="physical_lmax")
        surface = _nonnegative_integer(surface_lmax, name="surface_lmax")
        if physical > HARMONIC_GALERKIN_MAXIMUM_LMAX:
            raise ValueError("physical_lmax exceeds the harmonic coefficient contract.")
        if surface > physical:
            raise ValueError("surface_lmax cannot exceed physical_lmax.")
        order = _positive_integer(polar_order, name="polar_order")
        directions, weights = _sphere_rule(order)
        physical_dimension = (physical + 1) ** 2
        surface_dimension = (surface + 1) ** 2
        weighted_basis = np.asarray(weighted_basis_operator, dtype=float)
        expected_rows = count * physical_dimension
        if (
            weighted_basis.ndim != 2
            or weighted_basis.shape[0] != expected_rows
            or weighted_basis.shape[1] != count * surface_dimension
            or not np.all(np.isfinite(weighted_basis))
        ):
            raise ValueError(
                "weighted_basis_operator is incompatible with the declared harmonic spaces."
            )
        # Atom ownership is part of the coefficient chart.  Reject a matrix
        # that silently couples one atom's test basis into another atom's
        # physical-sphere block.
        for atom_i in range(count):
            row = slice(atom_i * physical_dimension, (atom_i + 1) * physical_dimension)
            for atom_j in range(count):
                if atom_i == atom_j:
                    continue
                column = slice(
                    atom_j * surface_dimension,
                    (atom_j + 1) * surface_dimension,
                )
                if np.any(weighted_basis[row, column] != 0.0):
                    raise ValueError(
                        "weighted_basis_operator must preserve atom-major ownership."
                    )
        points_angstrom = (
            positions[:, None, :] + radii[:, None, None] * directions[None, :, :]
        )
        points_bohr = points_angstrom.reshape(-1, 3) / Bohr
        weighted_digest = _array_sha256(weighted_basis)
        points_digest = _array_sha256(points_bohr)
        weights_digest = _array_sha256(weights)
        geometry_digest = _digest(geometry_sha256, name="geometry_sha256")
        continuum_digest = _digest(
            continuum_configuration_sha256,
            name="continuum_configuration_sha256",
        )
        provenance_digest = _digest(
            continuum_provenance_sha256,
            name="continuum_provenance_sha256",
        )
        topology_digest = _digest(topology_sha256, name="topology_sha256")
        cavity = _text(cavity_profile_id, name="cavity_profile_id")
        configuration = _canonical_sha256(
            {
                "contract_version": HARMONIC_SURFACE_MEP_PROJECTION_CONTRACT_VERSION,
                "geometry_sha256": geometry_digest,
                "continuum_configuration_sha256": continuum_digest,
                "continuum_provenance_sha256": provenance_digest,
                "topology_sha256": topology_digest,
                "cavity_profile_id": cavity,
                "positions_angstrom_sha256": _array_sha256(positions),
                "radii_angstrom_sha256": _array_sha256(radii),
                "physical_lmax": physical,
                "surface_lmax": surface,
                "polar_order": order,
                "azimuthal_count": 2 * order + 1,
                "quadrature": "gauss-legendre-polar-x-uniform-azimuth-reference-only",
                "weighted_basis_sha256": weighted_digest,
                "points_bohr_sha256": points_digest,
                "angular_weights_sha256": weights_digest,
                "bohr_to_angstrom": format(Bohr, ".17g"),
                "hartree_to_ev": format(HARTREE_TO_EV, ".17g"),
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "atom_count", count)
        object.__setattr__(self, "physical_lmax", physical)
        object.__setattr__(self, "surface_lmax", surface)
        object.__setattr__(self, "polar_order", order)
        object.__setattr__(self, "azimuthal_count", 2 * order + 1)
        object.__setattr__(self, "points_per_atom", int(directions.shape[0]))
        object.__setattr__(self, "boundary_dimension", int(weighted_basis.shape[1]))
        object.__setattr__(self, "geometry_sha256", geometry_digest)
        object.__setattr__(self, "continuum_configuration_sha256", continuum_digest)
        object.__setattr__(self, "continuum_provenance_sha256", provenance_digest)
        object.__setattr__(self, "topology_sha256", topology_digest)
        object.__setattr__(self, "cavity_profile_id", cavity)
        object.__setattr__(self, "weighted_basis_sha256", weighted_digest)
        object.__setattr__(self, "points_bohr_sha256", points_digest)
        object.__setattr__(self, "angular_weights_sha256", weights_digest)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(
            self, "_position_values", tuple(float(value) for value in positions.flat)
        )
        object.__setattr__(
            self, "_radius_values", tuple(float(value) for value in radii)
        )
        object.__setattr__(
            self, "_direction_values", tuple(float(value) for value in directions.flat)
        )
        object.__setattr__(
            self, "_weight_values", tuple(float(value) for value in weights)
        )
        object.__setattr__(
            self,
            "_weighted_basis_values",
            tuple(float(value) for value in weighted_basis.flat),
        )

    @property
    def physical_dimension_per_atom(self) -> int:
        return (self.physical_lmax + 1) ** 2

    @property
    def surface_dimension_per_atom(self) -> int:
        return (self.surface_lmax + 1) ** 2

    @property
    def positions_angstrom(self) -> np.ndarray:
        return _readonly(self._position_values, shape=(self.atom_count, 3))

    @property
    def radii_angstrom(self) -> np.ndarray:
        return _readonly(self._radius_values, shape=(self.atom_count,))

    @property
    def directions(self) -> np.ndarray:
        return _readonly(self._direction_values, shape=(self.points_per_atom, 3))

    @property
    def angular_weights(self) -> np.ndarray:
        return _readonly(self._weight_values, shape=(self.points_per_atom,))

    @property
    def points_bohr(self) -> np.ndarray:
        points_angstrom = (
            self.positions_angstrom[:, None, :]
            + self.radii_angstrom[:, None, None] * self.directions[None, :, :]
        )
        result = np.ascontiguousarray(points_angstrom.reshape(-1, 3) / Bohr)
        result.setflags(write=False)
        if _array_sha256(result) != self.points_bohr_sha256:
            raise RuntimeError("harmonic MEP projection points drifted.")
        return result

    @property
    def weighted_basis_operator(self) -> np.ndarray:
        return _readonly(
            self._weighted_basis_values,
            shape=(
                self.atom_count * self.physical_dimension_per_atom,
                self.boundary_dimension,
            ),
        )

    @property
    def squared_exposure_angular_weights(self) -> np.ndarray:
        """Return non-negative reference weights derived from the finite exposure.

        The first surface-basis column is ``e * Y00``.  Multiplying it by
        ``sqrt(4*pi)`` recovers the finite exposure coefficients used by the
        actual weighted basis.  Squaring the evaluated exposure yields a
        non-negative pointwise diagnostic weight; it does not alter the RHS
        projection itself.
        """

        design = _real_harmonic_design(self.directions, lmax=self.physical_lmax)
        result = np.empty((self.atom_count, self.points_per_atom), dtype=float)
        physical_dimension = self.physical_dimension_per_atom
        surface_dimension = self.surface_dimension_per_atom
        weighted_basis = self.weighted_basis_operator
        for atom in range(self.atom_count):
            row = slice(atom * physical_dimension, (atom + 1) * physical_dimension)
            constant_column = atom * surface_dimension
            exposure_coefficients = (
                np.sqrt(4.0 * np.pi) * weighted_basis[row, constant_column]
            )
            exposure = design @ exposure_coefficients
            result[atom] = self.angular_weights * exposure * exposure
        result.setflags(write=False)
        return result

    def project(self, potential_hartree_per_e: object) -> HarmonicSurfaceMEPProjection:
        potential = np.asarray(potential_hartree_per_e, dtype=float)
        expected = (self.atom_count, self.points_per_atom)
        if potential.shape == (self.atom_count * self.points_per_atom,):
            potential = potential.reshape(expected)
        if potential.shape != expected or not np.all(np.isfinite(potential)):
            raise ValueError(
                "potential_hartree_per_e must be finite with one value per "
                "projector point."
            )
        design = _real_harmonic_design(self.directions, lmax=self.physical_lmax)
        raw = np.stack(
            [
                design.T @ (self.angular_weights * atom_potential * HARTREE_TO_EV)
                for atom_potential in potential
            ]
        )
        rhs = self.weighted_basis_operator.T @ raw.reshape(-1)
        return HarmonicSurfaceMEPProjection(
            projector_configuration_sha256=self.configuration_sha256,
            potential_hartree_per_e_sha256=_array_sha256(potential),
            raw_harmonic_coefficients=raw,
            boundary_rhs=rhs,
            atom_count=self.atom_count,
            physical_dimension_per_atom=self.physical_dimension_per_atom,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": HARMONIC_SURFACE_MEP_PROJECTION_CONTRACT_VERSION,
            "configuration_sha256": self.configuration_sha256,
            "geometry_sha256": self.geometry_sha256,
            "continuum_configuration_sha256": self.continuum_configuration_sha256,
            "continuum_provenance_sha256": self.continuum_provenance_sha256,
            "topology_sha256": self.topology_sha256,
            "cavity_profile_id": self.cavity_profile_id,
            "atom_count": self.atom_count,
            "physical_lmax": self.physical_lmax,
            "surface_lmax": self.surface_lmax,
            "polar_order": self.polar_order,
            "azimuthal_count": self.azimuthal_count,
            "points_per_atom": self.points_per_atom,
            "total_point_count": self.atom_count * self.points_per_atom,
            "boundary_dimension": self.boundary_dimension,
            "weighted_basis_sha256": self.weighted_basis_sha256,
            "points_bohr_sha256": self.points_bohr_sha256,
            "angular_weights_sha256": self.angular_weights_sha256,
            "quadrature_role": (
                "reference-integration-only; not part of continuum assembly"
            ),
            "capability_admitted": False,
        }


__all__ = [
    "HARMONIC_SURFACE_MEP_PROJECTION_CONTRACT_VERSION",
    "HarmonicSurfaceMEPProjection",
    "HarmonicSurfaceMEPProjector",
]
