"""Fixed-topology smooth aqueous SMD-CDS companion for Route 2.

The fixed-topology amplitude C-PCM continuum needs a non-electrostatic term
that is differentiable on the same kind of all-candidate surface.  The
standard PySCF SMD CDS bridge remains the preferred *energy* implementation,
but its independently built/pruned surface cannot certify a smooth Route-2
PES together with a fixed-cardinality continuum.

This module therefore evaluates the published *aqueous* SMD atomic-tension
functional on a fixed-topology, C3-amplitude surface:

``G_CDS = sum_A gamma_A(R) A_A(R)``.

The geometry-dependent aqueous SMD tensions are reused verbatim from
``smd_cds``.  Only the numerical realization of the accessible areas changes:
every atom/angular candidate remains in the discrete space and contributes
``base_area * g**2``.  This is a named smooth discretization, not a bitwise
replacement for PySCF/NWChem legacy SMD-CDS.  In particular,
``strict_original_smd_equivalence`` is deliberately false in provenance.

It is water-only because the pure-Python published tension implementation in
``smd_cds`` is water-specific.  Other solvents continue to use the optional
upstream PySCF SMD-CDS energy path until a similarly source-grounded smooth
nonpolar functional is available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from ase.units import Bohr

from .route2_fixed_topology_surface import (
    FixedTopologyAmplitudeSWIGSurface,
    build_fixed_topology_amplitude_swig_surface,
    load_pyscf_amplitude_swig_angular_grid,
)
from .smd_cds import (
    HARTREE_TO_KCAL_MOL,
    aqueous_atomic_surface_tension_position_vjp,
    aqueous_atomic_surface_tensions,
    smd_sasa_radii,
    validate_smd_symbols,
)


def _immutable_vector(values: np.ndarray, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {(length,)}; received {array.shape}."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_gradient(
    values: np.ndarray,
    *,
    atom_count: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (atom_count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(
            "position_gradient_hartree_per_angstrom must be finite with shape "
            f"{(atom_count, 3)}; received {array.shape}."
        )
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FixedTopologyAqueousSMDCDSResult:
    """One same-energy aqueous SMD-CDS value and analytic coordinate gradient."""

    energy_hartree: float
    energy_kcal_mol: float
    atom_areas_angstrom2: np.ndarray
    atom_tensions_cal_mol_angstrom2: np.ndarray
    position_gradient_hartree_per_angstrom: np.ndarray
    grid_points_per_atom: int
    surface_size: int

    def __post_init__(self) -> None:
        energy_hartree = float(self.energy_hartree)
        energy_kcal = float(self.energy_kcal_mol)
        if not math.isfinite(energy_hartree) or not math.isfinite(energy_kcal):
            raise ValueError("Fixed-topology aqueous SMD-CDS energy must be finite.")
        if abs(energy_kcal - energy_hartree * HARTREE_TO_KCAL_MOL) > max(
            1.0e-10,
            1.0e-10 * abs(energy_kcal),
        ):
            raise ValueError(
                "Fixed-topology aqueous SMD-CDS energy unit conversion is invalid."
            )
        areas = np.asarray(self.atom_areas_angstrom2, dtype=float)
        if areas.ndim != 1:
            raise ValueError("atom_areas_angstrom2 must be one-dimensional.")
        atom_count = int(areas.size)
        if atom_count == 0:
            raise ValueError("Fixed-topology aqueous SMD-CDS requires one atom.")
        object.__setattr__(
            self,
            "atom_areas_angstrom2",
            _immutable_vector(
                areas,
                name="atom_areas_angstrom2",
                length=atom_count,
            ),
        )
        object.__setattr__(
            self,
            "atom_tensions_cal_mol_angstrom2",
            _immutable_vector(
                self.atom_tensions_cal_mol_angstrom2,
                name="atom_tensions_cal_mol_angstrom2",
                length=atom_count,
            ),
        )
        object.__setattr__(
            self,
            "position_gradient_hartree_per_angstrom",
            _immutable_gradient(
                self.position_gradient_hartree_per_angstrom,
                atom_count=atom_count,
            ),
        )
        grid_points = int(self.grid_points_per_atom)
        if grid_points <= 0:
            raise ValueError("grid_points_per_atom must be positive.")
        if int(self.surface_size) != atom_count * grid_points:
            raise ValueError(
                "Fixed-topology aqueous SMD-CDS surface cardinality must equal "
                "atom_count * grid_points_per_atom."
            )
        object.__setattr__(self, "energy_hartree", energy_hartree)
        object.__setattr__(self, "energy_kcal_mol", energy_kcal)
        object.__setattr__(self, "grid_points_per_atom", grid_points)
        object.__setattr__(self, "surface_size", int(self.surface_size))


class FixedTopologyAqueousSMDCDS:
    """A frozen-geometry, all-candidate aqueous SMD-CDS discretization.

    The object retains one immutable surface state.  Calling :meth:`result`
    returns the energy and the derivative of exactly that scalar; neither an
    exposed-node mask nor a separate finite-difference area calculation is
    used.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        atom_positions_angstrom: np.ndarray,
        *,
        lebedev_order: int | None = None,
        _unit_sphere: np.ndarray | None = None,
        _switching_constant: float | None = None,
        _runtime_version: str | None = None,
    ) -> None:
        symbol_tuple = validate_smd_symbols(tuple(str(symbol) for symbol in symbols))
        atom_count = len(symbol_tuple)
        if atom_count == 0:
            raise ValueError("Fixed-topology aqueous SMD-CDS requires at least one atom.")
        positions = np.asarray(atom_positions_angstrom, dtype=float)
        if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                "atom_positions_angstrom must be finite with shape "
                f"{(atom_count, 3)}."
            )

        injected_grid = _unit_sphere is not None or _switching_constant is not None
        if injected_grid and (_unit_sphere is None or _switching_constant is None):
            raise ValueError(
                "_unit_sphere and _switching_constant must be supplied together."
            )
        if injected_grid:
            angular_grid = np.asarray(_unit_sphere, dtype=float)
            switching_constant = float(_switching_constant)
            runtime_version = str(_runtime_version or "injected-test-grid")
            grid_order: int | None = None
        else:
            if lebedev_order is None:
                raise ValueError("lebedev_order is required without an injected grid.")
            angular_grid, switching_constant, runtime_version = (
                load_pyscf_amplitude_swig_angular_grid(lebedev_order)
            )
            grid_order = int(lebedev_order)

        radii_angstrom = smd_sasa_radii(symbol_tuple)
        surface = build_fixed_topology_amplitude_swig_surface(
            positions / Bohr,
            radii_angstrom / Bohr,
            angular_grid,
            switching_constant=switching_constant,
        )
        self._symbols = symbol_tuple
        self._positions_angstrom = np.array(positions, dtype=float, copy=True)
        self._radii_angstrom = np.array(radii_angstrom, dtype=float, copy=True)
        self._surface = surface
        self._runtime_version = runtime_version
        self._lebedev_order = grid_order
        self._result: FixedTopologyAqueousSMDCDSResult | None = None

    @property
    def atom_count(self) -> int:
        return len(self._symbols)

    @property
    def surface(self) -> FixedTopologyAmplitudeSWIGSurface:
        """Return the immutable fixed-topology area surface state."""

        return self._surface

    @property
    def runtime_provenance(self) -> dict[str, str | int | bool | None]:
        return {
            "provider": "fixed-topology-aqueous-smd-cds",
            "profile": "aqueous-smd-cds-fixed-topology-c3-area-v1-experimental",
            "solvent": "water",
            "pyscf_version": self._runtime_version,
            "lebedev_order": self._lebedev_order,
            "grid_points_per_atom": self._surface.grid_points_per_atom,
            "surface_size": self._surface.surface_size,
            "surface_cardinality_policy": "all-candidates-retained-v1",
            "area_quadrature": "base-area-times-C3-amplitude-squared",
            "atomic_tension_function": "published-aqueous-smd-geometry-functions",
            "strict_original_smd_equivalence": False,
            "same_scalar_coordinate_derivative": True,
        }

    def result(self) -> FixedTopologyAqueousSMDCDSResult:
        """Return the scalar CDS value and its analytic position gradient."""

        cached = self._result
        if cached is not None:
            return cached

        tensions = aqueous_atomic_surface_tensions(
            self._symbols,
            self._positions_angstrom,
        )
        parent = self._surface.parent_atom_indices
        amplitudes = self._surface.exposure_amplitudes
        base_areas_bohr2 = self._surface.base_areas_bohr2
        node_areas_bohr2 = base_areas_bohr2 * amplitudes**2
        atom_areas_bohr2 = np.bincount(
            parent,
            weights=node_areas_bohr2,
            minlength=self.atom_count,
        )
        atom_areas_angstrom2 = atom_areas_bohr2 * Bohr**2
        energy_scale = 1.0 / (1000.0 * HARTREE_TO_KCAL_MOL)
        energy_hartree = float(np.dot(tensions, atom_areas_angstrom2) * energy_scale)

        # d/dg [gamma_A * base_area_i * g_i**2], including the bohr2-to-A2
        # conversion.  The surface VJP returns hartree/bohr.
        amplitude_cotangent = (
            2.0
            * tensions[parent]
            * base_areas_bohr2
            * amplitudes
            * (Bohr**2 * energy_scale)
        )
        area_gradient_hartree_per_bohr = (
            self._surface.exposure_amplitude_position_vjp(amplitude_cotangent)
        )
        area_gradient_hartree_per_angstrom = area_gradient_hartree_per_bohr / Bohr

        # The existing analytical SMD tension VJP differentiates the same
        # gamma_A(R) functions.  Its input areas are in A2, so it already
        # returns hartree/angstrom.
        tension_gradient_hartree_per_angstrom = (
            aqueous_atomic_surface_tension_position_vjp(
                self._symbols,
                self._positions_angstrom,
                atom_areas_angstrom2 * energy_scale,
            )
        )
        gradient = (
            area_gradient_hartree_per_angstrom
            + tension_gradient_hartree_per_angstrom
        )

        computed = FixedTopologyAqueousSMDCDSResult(
            energy_hartree=energy_hartree,
            energy_kcal_mol=energy_hartree * HARTREE_TO_KCAL_MOL,
            atom_areas_angstrom2=atom_areas_angstrom2,
            atom_tensions_cal_mol_angstrom2=tensions,
            position_gradient_hartree_per_angstrom=gradient,
            grid_points_per_atom=self._surface.grid_points_per_atom,
            surface_size=self._surface.surface_size,
        )
        self._result = computed
        return computed


__all__ = [
    "FixedTopologyAqueousSMDCDS",
    "FixedTopologyAqueousSMDCDSResult",
]
