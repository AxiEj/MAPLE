"""Exact AO-density/grid dual map for the Route-2 V0-AQ-C construction.

The smooth diffuse-continuum blocks use an electron density ``n(r_g)`` on a
regular Cartesian grid, whereas a stationary auxiliary electronic method
normally represents its state by a spin-summed AO density matrix ``D``.  This
module owns that otherwise easy-to-miss change of representation.  For AO
values ``chi[g, mu]`` at the grid nodes it defines

``n_D[g] = sum_mu_nu chi[g, mu] D[mu, nu] chi[g, nu]``

and the *only* compatible AO potential pullback

``V_u[mu, nu] = dV sum_g chi[g, mu] u[g] chi[g, nu]``.

Consequently, in the declared discrete pairing,

``dV sum_g n_D[g] u[g] = Tr[D V_u]``.

This is the bridge required to put an AO stationary electronic functional and
the density-defined V0-AQ-C reaction scalar in one common Euler equation.  It
does not provide an electronic functional, a solvent asset, a nuclear source,
or an SCF solver; it also does not identify an auxiliary density with MACE.
The caller must supply all of those independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

import numpy as np

from .route2_v0_structured_solvent import RegularCartesianGrid

V0_AUXILIARY_AO_GRID_CONSTRUCTION = "route2-v0-auxiliary-ao-grid-dual-v1"
V0_AUXILIARY_AO_GRID_SCOPE = "auxiliary-electron-density-dual-only-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return one immutable finite real array with an optional exact shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _symmetric_matrix(
    values: np.ndarray,
    *,
    name: str,
    dimension: int,
    relative_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Validate a material symmetric AO matrix without repairing it."""

    matrix = _immutable_array(values, name=name, shape=(dimension, dimension))
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    asymmetry = float(np.linalg.norm(matrix - matrix.T, ord=2))
    if asymmetry > relative_tolerance * scale:
        raise ValueError(f"{name} must be symmetric in the declared AO pairing.")
    return matrix


def _projection_fingerprint(
    *,
    grid: RegularCartesianGrid,
    ao_values: np.ndarray,
    overlap_matrix: np.ndarray,
) -> str:
    """Return the immutable identity of one AO/grid representation."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(grid.origin_bohr, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(grid.spacing_bohr, dtype=np.float64).tobytes())
    digest.update(np.asarray(grid.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(ao_values, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(overlap_matrix, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Route2V0AuxiliaryAODensityGridState:
    """One physical AO density evaluated on one immutable Cartesian grid.

    ``ao_electron_count_e`` is the AO-overlap count ``Tr[D S]``.  The separate
    ``grid_electron_count_e`` deliberately exposes finite-box/quadrature error
    rather than normalizing or rescaling the grid density after evaluation.
    """

    density_matrix_e: np.ndarray
    electron_density_e_per_bohr3: np.ndarray
    ao_electron_count_e: float
    grid_electron_count_e: float
    projection_fingerprint: str
    construction: str = V0_AUXILIARY_AO_GRID_CONSTRUCTION
    scope: str = V0_AUXILIARY_AO_GRID_SCOPE

    def __post_init__(self) -> None:
        density = _immutable_array(
            self.electron_density_e_per_bohr3,
            name="AO-grid electron density",
        )
        if density.ndim != 3 or density.size == 0 or not np.all(np.isfinite(density)):
            raise ValueError(
                "AO-grid electron density must be a nonempty finite 3D field."
            )
        if np.any(density < 0.0):
            raise ValueError("AO-grid electron density must be nonnegative.")
        raw_matrix = _immutable_array(
            self.density_matrix_e, name="AO-grid density matrix"
        )
        if (
            raw_matrix.ndim != 2
            or raw_matrix.shape[0] == 0
            or raw_matrix.shape[0] != raw_matrix.shape[1]
        ):
            raise ValueError("AO-grid density matrix must be a nonempty square matrix.")
        matrix = _symmetric_matrix(
            raw_matrix,
            name="AO-grid density matrix",
            dimension=raw_matrix.shape[0],
        )
        for name in ("ao_electron_count_e", "grid_electron_count_e"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.projection_fingerprint, str)
            or _DIGEST.fullmatch(self.projection_fingerprint) is None
        ):
            raise ValueError("AO-grid projection fingerprint is invalid.")
        if self.construction != V0_AUXILIARY_AO_GRID_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 auxiliary AO-grid construction.")
        if self.scope != V0_AUXILIARY_AO_GRID_SCOPE:
            raise ValueError("Unsupported Route-2 V0 auxiliary AO-grid scope.")
        electron = np.array(density, dtype=float, copy=True)
        electron.setflags(write=False)
        object.__setattr__(self, "density_matrix_e", matrix)
        object.__setattr__(self, "electron_density_e_per_bohr3", electron)

    @property
    def grid_electron_count_error_e(self) -> float:
        """Return finite-domain/grid quadrature error without correcting it."""

        return float(self.grid_electron_count_e - self.ao_electron_count_e)


@dataclass(frozen=True)
class Route2V0AuxiliaryAODensityGridProjection:
    """One exact discrete AO-density/grid source and dual pairing.

    ``ao_values`` has shape ``(grid.point_count, n_ao)`` in the C-order point
    layout returned by :meth:`RegularCartesianGrid.points_bohr`.  The supplied
    overlap is the electronic method's AO overlap, not a grid-fitted overlap.
    This distinction makes finite-domain representation error observable.
    """

    grid: RegularCartesianGrid
    ao_values: np.ndarray
    overlap_matrix: np.ndarray
    construction: str = V0_AUXILIARY_AO_GRID_CONSTRUCTION
    scope: str = V0_AUXILIARY_AO_GRID_SCOPE
    _projection_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RegularCartesianGrid):
            raise TypeError("Auxiliary AO-grid projection requires a regular grid.")
        if self.construction != V0_AUXILIARY_AO_GRID_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 auxiliary AO-grid construction.")
        if self.scope != V0_AUXILIARY_AO_GRID_SCOPE:
            raise ValueError("Unsupported Route-2 V0 auxiliary AO-grid scope.")
        values = _immutable_array(self.ao_values, name="AO values")
        if (
            values.ndim != 2
            or values.shape[0] != self.grid.point_count
            or values.shape[1] == 0
        ):
            raise ValueError(
                "AO values must have finite shape (grid.point_count, n_ao)."
            )
        overlap = _symmetric_matrix(
            self.overlap_matrix,
            name="AO overlap matrix",
            dimension=values.shape[1],
        )
        overlap_eigenvalues = np.linalg.eigvalsh(overlap)
        scale = max(1.0, float(np.linalg.norm(overlap, ord=2)))
        if float(np.min(overlap_eigenvalues)) <= 1.0e-12 * scale:
            raise ValueError("AO overlap matrix must be positive definite.")
        fingerprint = _projection_fingerprint(
            grid=self.grid,
            ao_values=values,
            overlap_matrix=overlap,
        )
        object.__setattr__(self, "ao_values", values)
        object.__setattr__(self, "overlap_matrix", overlap)
        object.__setattr__(self, "_projection_fingerprint", fingerprint)

    @property
    def ao_count(self) -> int:
        """Return the immutable number of AO basis functions."""

        return int(self.ao_values.shape[1])

    @property
    def projection_fingerprint(self) -> str:
        """Return the content identity required by evaluated density states."""

        return self._projection_fingerprint

    def density_matrix(
        self, values: np.ndarray, *, name: str = "AO density matrix"
    ) -> np.ndarray:
        """Validate one symmetric AO density or density-direction matrix."""

        return _symmetric_matrix(values, name=name, dimension=self.ao_count)

    def density_direction_e_per_bohr3(
        self, density_direction: np.ndarray
    ) -> np.ndarray:
        """Return the linear density map for a signed AO variation ``delta D``."""

        direction = self.density_matrix(
            density_direction,
            name="AO density direction",
        )
        density = np.einsum(
            "gi,ij,gj->g",
            self.ao_values,
            direction,
            self.ao_values,
            optimize=True,
        ).reshape(self.grid.shape)
        density = np.array(density, dtype=float, copy=True)
        density.setflags(write=False)
        return density

    def evaluate(
        self,
        density_matrix_e: np.ndarray,
    ) -> Route2V0AuxiliaryAODensityGridState:
        """Evaluate one nonnegative physical electron density on this grid."""

        matrix = self.density_matrix(density_matrix_e)
        density = self.density_direction_e_per_bohr3(matrix)
        scale = max(1.0, float(np.max(np.abs(density))))
        if float(np.min(density)) < -128.0 * np.finfo(float).eps * scale:
            raise ValueError("AO density matrix produces a negative grid density.")
        # Remove only IEEE roundoff below the mathematically nonnegative bound;
        # this is not a density normalization or a response correction.
        density = np.maximum(density, 0.0)
        density.setflags(write=False)
        ao_count = self.ao_electron_count_e(matrix)
        grid_count = self.grid_electron_count_e(density)
        return Route2V0AuxiliaryAODensityGridState(
            density_matrix_e=matrix,
            electron_density_e_per_bohr3=density,
            ao_electron_count_e=ao_count,
            grid_electron_count_e=grid_count,
            projection_fingerprint=self._projection_fingerprint,
        )

    def validate_state(self, state: Route2V0AuxiliaryAODensityGridState) -> None:
        """Reject a state from another projection or a forged density record."""

        if not isinstance(state, Route2V0AuxiliaryAODensityGridState):
            raise TypeError("AO-grid projection requires an AO-grid density state.")
        if state.projection_fingerprint != self._projection_fingerprint:
            raise ValueError("AO-grid density state does not match this projection.")
        expected = self.evaluate(state.density_matrix_e)
        tolerance = (
            256.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.max(np.abs(expected.electron_density_e_per_bohr3))),
            )
        )
        if not np.allclose(
            state.electron_density_e_per_bohr3,
            expected.electron_density_e_per_bohr3,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError(
                "AO-grid density state is inconsistent with its AO matrix."
            )
        for name in ("ao_electron_count_e", "grid_electron_count_e"):
            actual = float(getattr(state, name))
            reference = float(getattr(expected, name))
            tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(reference))
            if abs(actual - reference) > tolerance:
                raise ValueError(
                    f"AO-grid density state {name} is inconsistent with its AO matrix."
                )

    def ao_electron_count_e(self, density_matrix_e: np.ndarray) -> float:
        """Return the method-defined AO electron count ``Tr[D S]``."""

        matrix = self.density_matrix(density_matrix_e)
        count = float(np.einsum("ij,ji->", matrix, self.overlap_matrix, optimize=True))
        if not math.isfinite(count):
            raise RuntimeError("AO electron count is non-finite.")
        return count

    def grid_electron_count_e(self, electron_density_e_per_bohr3: np.ndarray) -> float:
        """Return the declared finite-grid electron count without renormalizing."""

        density = _immutable_array(
            electron_density_e_per_bohr3,
            name="AO-grid electron density",
            shape=self.grid.shape,
        )
        return float(self.grid.volume_element_bohr3 * np.sum(density))

    def density_potential_matrix_hartree(
        self,
        electron_density_potential_hartree_per_e: np.ndarray,
    ) -> np.ndarray:
        """Pull back ``delta A/delta n`` into its exact AO Fock matrix.

        The potential has units Hartree per elementary charge; the returned
        matrix has Hartree units and is the derivative under the AO trace
        pairing.  It is suitable only after the caller has established that
        the input potential belongs to the same scalar as the electronic
        functional.
        """

        potential = _immutable_array(
            electron_density_potential_hartree_per_e,
            name="Electron-density potential",
            shape=self.grid.shape,
        ).reshape(self.grid.point_count)
        matrix = self.grid.volume_element_bohr3 * np.einsum(
            "gi,g,gj->ij",
            self.ao_values,
            potential,
            self.ao_values,
            optimize=True,
        )
        matrix = _symmetric_matrix(
            matrix,
            name="AO density-potential pullback",
            dimension=self.ao_count,
        )
        return matrix

    def density_potential_pairing_error_hartree(
        self,
        density_matrix_e: np.ndarray,
        electron_density_potential_hartree_per_e: np.ndarray,
    ) -> float:
        """Return the exact discrete AO/grid duality residual."""

        matrix = self.density_matrix(density_matrix_e)
        density = self.density_direction_e_per_bohr3(matrix)
        potential = _immutable_array(
            electron_density_potential_hartree_per_e,
            name="Electron-density potential",
            shape=self.grid.shape,
        )
        grid_pairing = self.grid.volume_element_bohr3 * float(
            np.sum(density * potential)
        )
        ao_pairing = float(
            np.einsum(
                "ij,ji->",
                matrix,
                self.density_potential_matrix_hartree(potential),
                optimize=True,
            )
        )
        return abs(grid_pairing - ao_pairing)


__all__ = [
    "V0_AUXILIARY_AO_GRID_CONSTRUCTION",
    "V0_AUXILIARY_AO_GRID_SCOPE",
    "Route2V0AuxiliaryAODensityGridProjection",
    "Route2V0AuxiliaryAODensityGridState",
]
