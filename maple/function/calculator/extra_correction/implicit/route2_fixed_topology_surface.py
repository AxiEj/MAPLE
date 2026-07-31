"""Fixed-cardinality amplitude-SWIG surface construction for Route 2.

This module deliberately does *not* call :func:`pyscf.solvent.pcm.gen_surface`.
That upstream helper removes low-weight nodes, which changes the dimension of
the discrete PCM system under a small nuclear displacement.  Here every
atom/angular-grid candidate is retained and an exposure amplitude ``g`` makes
an occluded node decouple continuously instead.

The construction is an experimental C-PCM discretization.  Its switching
weight is ``F = g**2`` with ``g`` formed from a C3 endpoint-smooth amplitude;
it is therefore not presented as a bitwise reparameterization of PySCF's
currently pruned SWIG matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import numpy as np

from .pyscf_runtime import TESTED_PYSCF_VERSION, require_tested_pyscf_version


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate and freeze a finite floating-point array."""

    array = np.asarray(values, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def amplitude_switch(t: np.ndarray) -> np.ndarray:
    """Return the C3 compact exposure amplitude on a normalized clearance.

    ``35*t**4 - 84*t**5 + 70*t**6 - 20*t**7`` has value and its first three
    derivatives equal to zero at the buried endpoint, while retaining unit
    value and zero derivatives at the exposed endpoint.  The function is
    evaluated branchwise rather than through ratios such as ``dg / g`` so
    completely covered nodes are harmless.
    """

    values = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Normalized clearances must be finite.")
    result = np.zeros_like(values)
    transition = (values > 0.0) & (values < 1.0)
    x = values[transition]
    # The exact C3 polynomial is monotone on [0, 1].  Horner evaluation near
    # the exposed endpoint can exceed one by a few ulps, however, which would
    # falsely turn a fully exposed candidate into an invalid physical weight.
    # Clipping restores the exact closed interval and only touches roundoff;
    # it is not a geometric cutoff or a node-deletion rule.
    result[transition] = np.clip(
        x**4 * (35.0 - x * (84.0 - x * (70.0 - 20.0 * x))),
        0.0,
        1.0,
    )
    result[values >= 1.0] = 1.0
    return result


@dataclass(frozen=True)
class FixedTopologyAmplitudeSWIGSurface:
    """One all-candidate amplitude-SWIG surface at a fixed geometry.

    Coordinates and radii are in bohr.  ``surface_points_bohr`` always has
    ``atom_count * grid_points_per_atom`` rows; effective quadrature area may
    become zero, but the candidate itself is never removed.
    """

    reference_positions_bohr: np.ndarray
    radii_bohr: np.ndarray
    parent_atom_indices: np.ndarray
    surface_points_bohr: np.ndarray
    unit_directions: np.ndarray
    quadrature_weights: np.ndarray
    charge_exponents_bohr_inverse: np.ndarray
    exposure_amplitudes: np.ndarray
    base_areas_bohr2: np.ndarray
    effective_areas_bohr2: np.ndarray
    inner_radii_bohr: np.ndarray
    switching_widths_bohr: np.ndarray
    grid_points_per_atom: int

    def __post_init__(self) -> None:
        positions = _immutable_array(
            self.reference_positions_bohr,
            name="reference_positions_bohr",
        )
        if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] != 3:
            raise ValueError(
                "reference_positions_bohr must have finite shape (n_atoms, 3)."
            )
        atom_count = positions.shape[0]
        radii = _immutable_array(
            self.radii_bohr,
            name="radii_bohr",
            shape=(atom_count,),
        )
        if np.any(radii <= 0.0):
            raise ValueError("radii_bohr must be strictly positive.")
        grid_points = int(self.grid_points_per_atom)
        if grid_points <= 0:
            raise ValueError("grid_points_per_atom must be positive.")
        surface_size = atom_count * grid_points
        parents = np.asarray(self.parent_atom_indices, dtype=int)
        if parents.shape != (surface_size,) or np.any(parents < 0) or np.any(
            parents >= atom_count
        ):
            raise ValueError(
                "parent_atom_indices must identify one parent per candidate node."
            )
        parents = np.array(parents, dtype=int, copy=True)
        parents.setflags(write=False)
        points = _immutable_array(
            self.surface_points_bohr,
            name="surface_points_bohr",
            shape=(surface_size, 3),
        )
        directions = _immutable_array(
            self.unit_directions,
            name="unit_directions",
            shape=(surface_size, 3),
        )
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-12):
            raise ValueError("unit_directions must have unit Euclidean norm.")
        weights = _immutable_array(
            self.quadrature_weights,
            name="quadrature_weights",
            shape=(surface_size,),
        )
        if np.any(weights <= 0.0):
            raise ValueError("quadrature_weights must be strictly positive.")
        exponents = _immutable_array(
            self.charge_exponents_bohr_inverse,
            name="charge_exponents_bohr_inverse",
            shape=(surface_size,),
        )
        if np.any(exponents <= 0.0):
            raise ValueError("charge_exponents_bohr_inverse must be positive.")
        amplitudes = _immutable_array(
            self.exposure_amplitudes,
            name="exposure_amplitudes",
            shape=(surface_size,),
        )
        if np.any(amplitudes < 0.0) or np.any(amplitudes > 1.0):
            raise ValueError("exposure_amplitudes must lie in [0, 1].")
        base_areas = _immutable_array(
            self.base_areas_bohr2,
            name="base_areas_bohr2",
            shape=(surface_size,),
        )
        if np.any(base_areas <= 0.0):
            raise ValueError("base_areas_bohr2 must be strictly positive.")
        effective_areas = _immutable_array(
            self.effective_areas_bohr2,
            name="effective_areas_bohr2",
            shape=(surface_size,),
        )
        expected_areas = base_areas * amplitudes**2
        if not np.allclose(effective_areas, expected_areas, rtol=1.0e-13, atol=1.0e-15):
            raise ValueError("effective_areas_bohr2 must equal base_area * g**2.")
        inner_radii = _immutable_array(
            self.inner_radii_bohr,
            name="inner_radii_bohr",
            shape=(atom_count,),
        )
        widths = _immutable_array(
            self.switching_widths_bohr,
            name="switching_widths_bohr",
            shape=(atom_count,),
        )
        if np.any(widths <= 0.0):
            raise ValueError("switching_widths_bohr must be strictly positive.")
        expected_points = positions[parents] + radii[parents, None] * directions
        if not np.allclose(points, expected_points, rtol=0.0, atol=2.0e-12):
            raise ValueError(
                "surface_points_bohr must be the rigid atom-centred candidate grid."
            )

        object.__setattr__(self, "reference_positions_bohr", positions)
        object.__setattr__(self, "radii_bohr", radii)
        object.__setattr__(self, "parent_atom_indices", parents)
        object.__setattr__(self, "surface_points_bohr", points)
        object.__setattr__(self, "unit_directions", directions)
        object.__setattr__(self, "quadrature_weights", weights)
        object.__setattr__(self, "charge_exponents_bohr_inverse", exponents)
        object.__setattr__(self, "exposure_amplitudes", amplitudes)
        object.__setattr__(self, "base_areas_bohr2", base_areas)
        object.__setattr__(self, "effective_areas_bohr2", effective_areas)
        object.__setattr__(self, "inner_radii_bohr", inner_radii)
        object.__setattr__(self, "switching_widths_bohr", widths)
        object.__setattr__(self, "grid_points_per_atom", grid_points)

    @property
    def atom_count(self) -> int:
        return int(self.reference_positions_bohr.shape[0])

    @property
    def surface_size(self) -> int:
        return int(self.surface_points_bohr.shape[0])

    @property
    def switching_weights(self) -> np.ndarray:
        """Return ``F = g**2`` without ever deleting a candidate node."""

        return np.array(self.exposure_amplitudes**2, copy=True)


def build_fixed_topology_amplitude_swig_surface(
    atom_positions_bohr: np.ndarray,
    radii_bohr: np.ndarray,
    unit_sphere: np.ndarray,
    *,
    switching_constant: float,
) -> FixedTopologyAmplitudeSWIGSurface:
    """Build the all-candidate amplitude-SWIG surface.

    ``unit_sphere`` follows PySCF's angular-grid convention: its first three
    columns are directions and its last column has weights that sum to one.
    The returned quadrature weights include the conventional ``4*pi`` factor.
    """

    positions = np.asarray(atom_positions_bohr, dtype=float)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] != 3:
        raise ValueError("atom_positions_bohr must have shape (n_atoms, 3).")
    if not np.all(np.isfinite(positions)):
        raise ValueError("atom_positions_bohr must be finite.")
    atom_count = positions.shape[0]
    radii = np.asarray(radii_bohr, dtype=float)
    if radii.shape != (atom_count,) or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_bohr must be finite and positive with shape (n_atoms,).")
    angular = np.asarray(unit_sphere, dtype=float)
    if angular.ndim != 2 or angular.shape[0] == 0 or angular.shape[1] != 4:
        raise ValueError("unit_sphere must have finite shape (n_grid, 4).")
    if not np.all(np.isfinite(angular)):
        raise ValueError("unit_sphere must be finite.")
    directions_one_atom = angular[:, :3]
    direction_norms = np.linalg.norm(directions_one_atom, axis=1)
    if not np.allclose(direction_norms, 1.0, rtol=0.0, atol=2.0e-12):
        raise ValueError("unit_sphere directions must have unit Euclidean norm.")
    raw_weights = angular[:, 3]
    if np.any(raw_weights <= 0.0):
        raise ValueError("unit_sphere weights must be strictly positive.")
    constant = float(switching_constant)
    if not math.isfinite(constant) or constant <= 0.0:
        raise ValueError("switching_constant must be finite and positive.")

    grid_points = angular.shape[0]
    parent = np.repeat(np.arange(atom_count, dtype=int), grid_points)
    directions = np.tile(directions_one_atom, (atom_count, 1))
    weights = np.tile(4.0 * math.pi * raw_weights, atom_count)
    points = positions[parent] + radii[parent, None] * directions
    base_areas = weights * radii[parent] ** 2
    charge_exponents = constant / (radii[parent] * np.sqrt(weights))

    switching_widths = radii * math.sqrt(14.0 / grid_points)
    alpha = (
        0.5
        + radii / switching_widths
        - np.sqrt((radii / switching_widths) ** 2 - 1.0 / 28.0)
    )
    inner_radii = radii - alpha * switching_widths

    displacement = points[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    normalized_clearance = (
        distances - inner_radii[None, :]
    ) / switching_widths[None, :]
    normalized_clearance[np.arange(points.shape[0]), parent] = 1.0
    amplitudes_by_neighbor = amplitude_switch(normalized_clearance)
    amplitudes = np.prod(amplitudes_by_neighbor, axis=1)
    effective_areas = base_areas * amplitudes**2

    return FixedTopologyAmplitudeSWIGSurface(
        reference_positions_bohr=positions,
        radii_bohr=radii,
        parent_atom_indices=parent,
        surface_points_bohr=points,
        unit_directions=directions,
        quadrature_weights=weights,
        charge_exponents_bohr_inverse=charge_exponents,
        exposure_amplitudes=amplitudes,
        base_areas_bohr2=base_areas,
        effective_areas_bohr2=effective_areas,
        inner_radii_bohr=inner_radii,
        switching_widths_bohr=switching_widths,
        grid_points_per_atom=grid_points,
    )


def load_pyscf_amplitude_swig_angular_grid(
    lebedev_order: int,
) -> tuple[np.ndarray, float, str]:
    """Load the tested PySCF angular grid and SWIG width constant lazily."""

    try:
        pyscf = importlib.import_module("pyscf")
        gen_grid = importlib.import_module("pyscf.dft.gen_grid")
        pcm = importlib.import_module("pyscf.solvent.pcm")
    except ImportError as exc:
        raise ImportError(
            "Fixed-topology amplitude-SWIG requires optional PySCF "
            f"{TESTED_PYSCF_VERSION} for its tabulated angular grid."
        ) from exc
    version = require_tested_pyscf_version(
        getattr(pyscf, "__version__", "unknown"),
        feature="Fixed-topology amplitude-SWIG angular-grid construction",
    )
    order = int(lebedev_order)
    try:
        grid_points = int(gen_grid.LEBEDEV_ORDER[order])
        switching_constant = float(pcm.XI[grid_points])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The requested Lebedev order is not supported by the tested "
            "PySCF SWIG angular-grid and switching-width tables."
        ) from exc
    if not math.isfinite(switching_constant) or switching_constant <= 0.0:
        raise RuntimeError("PySCF returned an invalid SWIG switching constant.")
    angular = np.asarray(gen_grid.MakeAngularGrid(grid_points), dtype=float)
    if angular.shape != (grid_points, 4) or not np.all(np.isfinite(angular)):
        raise RuntimeError("PySCF returned an invalid angular grid.")
    return angular, switching_constant, version


__all__ = [
    "FixedTopologyAmplitudeSWIGSurface",
    "amplitude_switch",
    "build_fixed_topology_amplitude_swig_surface",
    "load_pyscf_amplitude_swig_angular_grid",
]
