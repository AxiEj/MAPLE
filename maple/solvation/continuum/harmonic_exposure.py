"""Smooth rotational-scalar overlap fields in harmonic coefficient space.

This module never samples a raw buried/exposed mask on a laboratory-fixed
surface grid.  Each pair factor is reduced to invariant one-dimensional
Legendre coefficients and an exact harmonic irrep action.  Products and
multiplication matrices are then contracted only after every operand has a
declared finite bandwidth, using a tensor-product rule exact for that finite
algebraic degree.

The result is a fixed-dimensional, content-addressed geometry descriptor.  It
is a prerequisite for a geometry-assembled harmonic Galerkin continuum, not a
continuum operator or a Route-2 force admission by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from .harmonic_coefficients import (
    HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID,
    HARMONIC_GALERKIN_MAXIMUM_LMAX,
    PerAtomHarmonicSpace,
    _bounded_lmax,
    _positive_int,
    _real_harmonic_design,
)

SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID = (
    "maple.route2.cavity.smooth-weighted-overlap-harmonic-coefficients.v1"
)
SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID = (
    "maple.route2.cavity.smooth-harmonic-exposure.impl.v1"
)
HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE = 128


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"|<f8|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).resolve().parent
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in ("harmonic_coefficients.py", "harmonic_exposure.py")
    )


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _bounded_radial_order(value: object) -> int:
    count = _positive_int(value, name="radial_quadrature_order")
    if count > 4096:
        raise ValueError("radial_quadrature_order exceeds the bounded contract.")
    return count


def _vector3(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (3,).")
    return np.array(result, dtype=float, copy=True)


def _coefficient_vector(value: object, *, lmax: int, name: str) -> np.ndarray:
    maximum = _bounded_lmax(lmax)
    expected = ((maximum + 1) ** 2,)
    result = np.asarray(value, dtype=float)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {expected}.")
    return np.array(result, dtype=float, copy=True)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _constant_coefficients(lmax: int, value: float) -> np.ndarray:
    maximum = _bounded_lmax(lmax)
    result = np.zeros((maximum + 1) ** 2, dtype=float)
    result[0] = float(value) * np.sqrt(4.0 * np.pi)
    return _readonly(result)


def smooth_flat_step(value: object) -> float | np.ndarray:
    """Return a compactly supported ``C-infinity`` step on ``[-1, 1]``.

    Values are exactly zero for ``x <= -1`` and exactly one for ``x >= 1``.
    The transition is symmetric, monotone, and flat to all orders at both
    endpoints.
    """

    values = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("smooth_flat_step input must be finite.")
    result = np.empty_like(values, dtype=float)
    result[values <= -1.0] = 0.0
    result[values >= 1.0] = 1.0
    interior = np.abs(values) < 1.0
    if np.any(interior):
        magnitude = np.abs(values[interior])
        positive_value = expit(1.0 / (1.0 - magnitude) - 1.0 / (1.0 + magnitude))
        result[interior] = np.where(
            values[interior] < 0.0, 1.0 - positive_value, positive_value
        )
    if result.ndim == 0:
        return float(result)
    return result


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    count = _bounded_radial_order(order)
    points, weights = np.polynomial.legendre.leggauss(count)
    return _readonly(points), _readonly(weights)


def _finite_band_sphere_rule(required_degree: int) -> tuple[np.ndarray, np.ndarray]:
    degree = _positive_int(required_degree, name="required_degree", allow_zero=True)
    if degree > HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE:
        raise ValueError(
            "finite harmonic contraction exceeds the bounded algebraic degree "
            f"({HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE})."
        )
    polar_order = degree // 2 + 1
    azimuthal_order = degree + 1
    cos_theta, polar_weights = np.polynomial.legendre.leggauss(polar_order)
    phi = 2.0 * np.pi * np.arange(azimuthal_order) / azimuthal_order
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    directions = np.stack(
        (
            np.repeat(sin_theta, azimuthal_order) * np.tile(np.cos(phi), polar_order),
            np.repeat(sin_theta, azimuthal_order) * np.tile(np.sin(phi), polar_order),
            np.repeat(cos_theta, azimuthal_order),
        ),
        axis=1,
    )
    weights = np.repeat(
        polar_weights * (2.0 * np.pi / azimuthal_order), azimuthal_order
    )
    return directions, weights


def _pair_factor(
    *,
    center_i: object,
    radius_i_angstrom: object,
    center_j: object,
    radius_j_angstrom: object,
    transition_width_angstrom2: object,
    lmax: int,
    radial_quadrature_order: int,
) -> tuple[str, np.ndarray]:
    first = _vector3(center_i, name="center_i")
    second = _vector3(center_j, name="center_j")
    radius_i = _positive_float(radius_i_angstrom, name="radius_i_angstrom")
    radius_j = _positive_float(radius_j_angstrom, name="radius_j_angstrom")
    width = _positive_float(
        transition_width_angstrom2, name="transition_width_angstrom2"
    )
    maximum = _bounded_lmax(lmax)
    order = _bounded_radial_order(radial_quadrature_order)
    displacement = second - first
    distance = float(np.linalg.norm(displacement))
    if not np.isfinite(distance) or distance <= 1.0e-14:
        raise ValueError("sphere centers must be distinct for harmonic exposure.")

    z_minimum = (distance - radius_i) ** 2 - radius_j**2
    z_maximum = (distance + radius_i) ** 2 - radius_j**2
    if z_minimum >= width:
        return "exposed", _constant_coefficients(maximum, 1.0)
    if z_maximum <= -width:
        return "buried", _constant_coefficients(maximum, 0.0)

    cosine, weights = _legendre_rule(order)
    signed_overlap = (
        radius_i**2 + distance**2 - 2.0 * radius_i * distance * cosine - radius_j**2
    )
    exposure = np.asarray(smooth_flat_step(signed_overlap / width), dtype=float)
    legendre = np.polynomial.legendre.legvander(cosine, maximum)
    zonal_coefficients = 2.0 * np.pi * ((weights * exposure) @ legendre)
    direction = displacement / distance
    design = _real_harmonic_design(direction[None, :], lmax=maximum)[0]
    result = np.zeros((maximum + 1) ** 2, dtype=float)
    for ell in range(maximum + 1):
        block = slice(ell * ell, (ell + 1) * (ell + 1))
        result[block] = zonal_coefficients[ell] * design[block]
    return "transition", _readonly(result)


def smooth_pair_exposure_coefficients(
    *,
    center_i: object,
    radius_i_angstrom: object,
    center_j: object,
    radius_j_angstrom: object,
    transition_width_angstrom2: object,
    lmax: int,
    radial_quadrature_order: int = 96,
) -> np.ndarray:
    """Project one rotationally scalar smooth pair exposure to harmonics.

    The only numerical integration is the invariant one-dimensional integral
    over ``u dot d_hat``.  Its error changes radial scalars but cannot select a
    laboratory orientation.
    """

    _, coefficients = _pair_factor(
        center_i=center_i,
        radius_i_angstrom=radius_i_angstrom,
        center_j=center_j,
        radius_j_angstrom=radius_j_angstrom,
        transition_width_angstrom2=transition_width_angstrom2,
        lmax=lmax,
        radial_quadrature_order=radial_quadrature_order,
    )
    return coefficients


def project_harmonic_product(
    factors: object,
    *,
    lmax: int,
) -> np.ndarray:
    """Project a product of declared finite-band factors back to ``lmax``.

    The tensor-product rule is exact for the full finite algebraic degree of
    the product and output test harmonic.  This is coefficient algebra, not
    sampling of an unprojected cavity field.
    """

    maximum = _bounded_lmax(lmax)
    try:
        raw_factors = tuple(factors)
    except TypeError as exc:
        raise TypeError("factors must be an iterable of coefficient vectors.") from exc
    if not raw_factors:
        return _constant_coefficients(maximum, 1.0)
    coefficients = tuple(
        _coefficient_vector(value, lmax=maximum, name=f"factors[{index}]")
        for index, value in enumerate(raw_factors)
    )
    if any(not np.any(value) for value in coefficients):
        return _constant_coefficients(maximum, 0.0)
    required_degree = (len(coefficients) + 1) * maximum
    directions, weights = _finite_band_sphere_rule(required_degree)
    design = _real_harmonic_design(directions, lmax=maximum)
    values = np.ones(directions.shape[0], dtype=float)
    for coefficient in coefficients:
        values *= design @ coefficient
    result = design.T @ (weights * values)
    return _readonly(result)


def harmonic_multiplication_matrix(
    exposure_coefficients: object,
    *,
    exposure_lmax: int,
    surface_lmax: int,
) -> np.ndarray:
    """Return ``P_L M_exposure P_L`` in the complete real-harmonic basis."""

    exposure_maximum = _bounded_lmax(exposure_lmax)
    surface_maximum = _bounded_lmax(surface_lmax)
    if exposure_maximum < 2 * surface_maximum:
        raise ValueError(
            "exposure_lmax must be at least twice surface_lmax so every "
            "exposure mode visible to P_L M_e P_L is represented."
        )
    coefficients = _coefficient_vector(
        exposure_coefficients,
        lmax=exposure_maximum,
        name="exposure_coefficients",
    )
    required_degree = exposure_maximum + 2 * surface_maximum
    directions, weights = _finite_band_sphere_rule(required_degree)
    exposure_design = _real_harmonic_design(directions, lmax=exposure_maximum)
    surface_design = _real_harmonic_design(directions, lmax=surface_maximum)
    exposure = exposure_design @ coefficients
    result = surface_design.T @ ((weights * exposure)[:, None] * surface_design)
    result = 0.5 * (result + result.T)
    return _readonly(result)


def _assemble_exposure(
    *,
    positions: np.ndarray,
    radii: np.ndarray,
    exposure_lmax: int,
    surface_lmax: int,
    transition_width_angstrom2: float,
    radial_quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int, str], ...]]:
    atom_count = positions.shape[0]
    exposure_dimension = (exposure_lmax + 1) ** 2
    surface_dimension = (surface_lmax + 1) ** 2
    exposure_values = np.empty((atom_count, exposure_dimension), dtype=float)
    multiplication = np.empty(
        (atom_count, surface_dimension, surface_dimension), dtype=float
    )
    pair_states: list[tuple[int, int, str]] = []
    for atom_i in range(atom_count):
        active_factors: list[np.ndarray] = []
        buried = False
        for atom_j in range(atom_count):
            if atom_i == atom_j:
                continue
            state, coefficients = _pair_factor(
                center_i=positions[atom_i],
                radius_i_angstrom=radii[atom_i],
                center_j=positions[atom_j],
                radius_j_angstrom=radii[atom_j],
                transition_width_angstrom2=transition_width_angstrom2,
                lmax=exposure_lmax,
                radial_quadrature_order=radial_quadrature_order,
            )
            pair_states.append((atom_i, atom_j, state))
            if state == "buried":
                buried = True
            elif state == "transition":
                active_factors.append(coefficients)
        if buried:
            coefficients = _constant_coefficients(exposure_lmax, 0.0)
        else:
            coefficients = project_harmonic_product(active_factors, lmax=exposure_lmax)
        exposure_values[atom_i] = coefficients
        multiplication[atom_i] = harmonic_multiplication_matrix(
            coefficients,
            exposure_lmax=exposure_lmax,
            surface_lmax=surface_lmax,
        )
    return exposure_values, multiplication, tuple(pair_states)


def _topology_sha256(
    *,
    atomic_numbers: tuple[int, ...],
    surface_space: PerAtomHarmonicSpace,
    exposure_space: PerAtomHarmonicSpace,
) -> str:
    return _sha(
        {
            "contract_id": SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID,
            "coefficient_contract_id": HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID,
            "atom_count": len(atomic_numbers),
            "atomic_numbers": atomic_numbers,
            "surface_space_sha256": surface_space.metadata_sha256(),
            "exposure_space_sha256": exposure_space.metadata_sha256(),
            "ownership": "labelled-atom-major-fixed-complete-irreps",
            "active_dimension_changes": False,
        }
    )


def _configuration_sha256(
    *,
    topology_sha256: str,
    radii_angstrom: tuple[float, ...],
    transition_width_angstrom2: float,
    surface_lmax: int,
    exposure_lmax: int,
    radial_quadrature_order: int,
) -> str:
    return _sha(
        {
            "topology_sha256": topology_sha256,
            "radii_angstrom": radii_angstrom,
            "transition_width_angstrom2": transition_width_angstrom2,
            "surface_lmax": surface_lmax,
            "exposure_lmax": exposure_lmax,
            "radial_quadrature_order": radial_quadrature_order,
            "maximum_lmax": HARMONIC_GALERKIN_MAXIMUM_LMAX,
            "maximum_algebraic_degree": (HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE),
            "pair_projection": "invariant-1d-legendre",
            "product_projection": "exact-declared-finite-band-contraction",
            "surface_point_mask": False,
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class SmoothHarmonicExposureSnapshot:
    """Immutable fixed-dimensional smooth weighted-overlap descriptor."""

    atomic_numbers: tuple[int, ...]
    radii_angstrom: tuple[float, ...]
    transition_width_angstrom2: float
    surface_lmax: int
    exposure_lmax: int
    radial_quadrature_order: int
    surface_space: PerAtomHarmonicSpace
    exposure_space: PerAtomHarmonicSpace
    pair_states: tuple[tuple[int, int, str], ...]
    topology_sha256: str
    configuration_sha256: str
    geometry_sha256: str
    provenance_sha256: str
    state_sha256: str
    _position_values: tuple[float, ...]
    _exposure_values: tuple[float, ...]
    _multiplication_values: tuple[float, ...]

    def __init__(
        self,
        *,
        atomic_numbers: object,
        positions_angstrom: object,
        radii_angstrom: object,
        transition_width_angstrom2: object,
        surface_lmax: int,
        exposure_lmax: int,
        radial_quadrature_order: int = 96,
    ) -> None:
        try:
            numbers = tuple(atomic_numbers)
        except TypeError as exc:
            raise TypeError("atomic_numbers must be an iterable.") from exc
        if not numbers or any(
            isinstance(number, bool)
            or not isinstance(number, (int, np.integer))
            or int(number) < 1
            for number in numbers
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        numbers = tuple(int(number) for number in numbers)
        positions = np.asarray(positions_angstrom, dtype=float)
        if positions.shape != (len(numbers), 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                "positions_angstrom must be finite with shape (atom_count, 3)."
            )
        positions = np.ascontiguousarray(positions, dtype=float)
        radii = np.asarray(radii_angstrom, dtype=float)
        if (
            radii.shape != (len(numbers),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError(
                "radii_angstrom must be finite and positive with shape "
                "(atom_count,)."
            )
        radii = np.ascontiguousarray(radii, dtype=float)
        width = _positive_float(
            transition_width_angstrom2, name="transition_width_angstrom2"
        )
        surface_maximum = _bounded_lmax(surface_lmax)
        exposure_maximum = _bounded_lmax(exposure_lmax)
        if exposure_maximum < 2 * surface_maximum:
            raise ValueError(
                "exposure_lmax must be at least twice surface_lmax so the "
                "Galerkin multiplication operator is complete."
            )
        order = _bounded_radial_order(radial_quadrature_order)
        surface_space = PerAtomHarmonicSpace(
            atom_count=len(numbers), lmax=surface_maximum
        )
        exposure_space = PerAtomHarmonicSpace(
            atom_count=len(numbers), lmax=exposure_maximum
        )
        exposure, multiplication, pair_states = _assemble_exposure(
            positions=positions,
            radii=radii,
            exposure_lmax=exposure_maximum,
            surface_lmax=surface_maximum,
            transition_width_angstrom2=width,
            radial_quadrature_order=order,
        )
        radii_tuple = tuple(float(value) for value in radii)
        topology = _topology_sha256(
            atomic_numbers=numbers,
            surface_space=surface_space,
            exposure_space=exposure_space,
        )
        configuration = _configuration_sha256(
            topology_sha256=topology,
            radii_angstrom=radii_tuple,
            transition_width_angstrom2=width,
            surface_lmax=surface_maximum,
            exposure_lmax=exposure_maximum,
            radial_quadrature_order=order,
        )
        geometry = _array_sha256(positions)
        provenance = _sha(
            {
                "provider_id": SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID,
                "configuration_sha256": configuration,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        state = _sha(
            {
                "configuration_sha256": configuration,
                "geometry_sha256": geometry,
                "provenance_sha256": provenance,
                "pair_states": pair_states,
                "exposure_coefficients_sha256": _array_sha256(exposure),
                "multiplication_matrices_sha256": _array_sha256(multiplication),
            }
        )

        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "radii_angstrom", radii_tuple)
        object.__setattr__(self, "transition_width_angstrom2", width)
        object.__setattr__(self, "surface_lmax", surface_maximum)
        object.__setattr__(self, "exposure_lmax", exposure_maximum)
        object.__setattr__(self, "radial_quadrature_order", order)
        object.__setattr__(self, "surface_space", surface_space)
        object.__setattr__(self, "exposure_space", exposure_space)
        object.__setattr__(self, "pair_states", pair_states)
        object.__setattr__(self, "topology_sha256", topology)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "state_sha256", state)
        object.__setattr__(
            self, "_position_values", tuple(float(value) for value in positions.flat)
        )
        object.__setattr__(
            self, "_exposure_values", tuple(float(value) for value in exposure.flat)
        )
        object.__setattr__(
            self,
            "_multiplication_values",
            tuple(float(value) for value in multiplication.flat),
        )

    @property
    def atom_count(self) -> int:
        return len(self.atomic_numbers)

    @property
    def positions_angstrom(self) -> np.ndarray:
        return _readonly(
            np.asarray(self._position_values, dtype=float).reshape(self.atom_count, 3)
        )

    @property
    def exposure_coefficients(self) -> np.ndarray:
        dimension = self.exposure_space.single_atom_dimension
        return _readonly(
            np.asarray(self._exposure_values, dtype=float).reshape(
                self.atom_count, dimension
            )
        )

    @property
    def multiplication_matrices(self) -> np.ndarray:
        dimension = self.surface_space.single_atom_dimension
        return _readonly(
            np.asarray(self._multiplication_values, dtype=float).reshape(
                self.atom_count, dimension, dimension
            )
        )

    @property
    def global_multiplication_operator(self) -> np.ndarray:
        block_dimension = self.surface_space.single_atom_dimension
        result = np.zeros(
            (self.surface_space.dimension, self.surface_space.dimension), dtype=float
        )
        for atom, block in enumerate(self.multiplication_matrices):
            section = slice(atom * block_dimension, (atom + 1) * block_dimension)
            result[section, section] = block
        return _readonly(result)

    @property
    def coefficient_action_is_so3_equivariant(self) -> bool:
        return True

    @property
    def smooth_on_declared_noncoincident_domain(self) -> bool:
        return True

    @property
    def coordinate_derivative_available(self) -> bool:
        return False

    @property
    def tier_v_admitted(self) -> bool:
        return False

    def validate(self) -> None:
        positions = self.positions_angstrom
        radii = np.asarray(self.radii_angstrom, dtype=float)
        exposure = self.exposure_coefficients
        multiplication = self.multiplication_matrices
        if not np.all(np.isfinite(exposure)) or not np.all(np.isfinite(multiplication)):
            raise RuntimeError("harmonic exposure state drifted to non-finite values.")
        if not np.array_equal(multiplication, multiplication.swapaxes(1, 2)):
            raise RuntimeError("harmonic exposure multiplication matrices drifted.")
        expected_surface_space = PerAtomHarmonicSpace(
            atom_count=self.atom_count, lmax=self.surface_lmax
        )
        expected_exposure_space = PerAtomHarmonicSpace(
            atom_count=self.atom_count, lmax=self.exposure_lmax
        )
        expected_topology = _topology_sha256(
            atomic_numbers=self.atomic_numbers,
            surface_space=expected_surface_space,
            exposure_space=expected_exposure_space,
        )
        expected_configuration = _configuration_sha256(
            topology_sha256=expected_topology,
            radii_angstrom=self.radii_angstrom,
            transition_width_angstrom2=self.transition_width_angstrom2,
            surface_lmax=self.surface_lmax,
            exposure_lmax=self.exposure_lmax,
            radial_quadrature_order=self.radial_quadrature_order,
        )
        expected_geometry = _array_sha256(positions)
        expected_provenance = _sha(
            {
                "provider_id": SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID,
                "configuration_sha256": expected_configuration,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        expected_state = _sha(
            {
                "configuration_sha256": expected_configuration,
                "geometry_sha256": expected_geometry,
                "provenance_sha256": expected_provenance,
                "pair_states": self.pair_states,
                "exposure_coefficients_sha256": _array_sha256(exposure),
                "multiplication_matrices_sha256": _array_sha256(multiplication),
            }
        )
        if (
            self.surface_space != expected_surface_space
            or self.exposure_space != expected_exposure_space
            or self.topology_sha256 != expected_topology
            or self.configuration_sha256 != expected_configuration
            or self.geometry_sha256 != expected_geometry
            or self.provenance_sha256 != expected_provenance
            or self.state_sha256 != expected_state
        ):
            raise RuntimeError("harmonic exposure identity or content drifted.")
        rebuilt_exposure, rebuilt_multiplication, rebuilt_states = _assemble_exposure(
            positions=positions,
            radii=radii,
            exposure_lmax=self.exposure_lmax,
            surface_lmax=self.surface_lmax,
            transition_width_angstrom2=self.transition_width_angstrom2,
            radial_quadrature_order=self.radial_quadrature_order,
        )
        if (
            rebuilt_states != self.pair_states
            or not np.array_equal(rebuilt_exposure, exposure)
            or not np.array_equal(rebuilt_multiplication, multiplication)
        ):
            raise RuntimeError("harmonic exposure derived geometry state drifted.")


def build_smooth_harmonic_exposure(
    *,
    atomic_numbers: object,
    positions_angstrom: object,
    radii_angstrom: object,
    transition_width_angstrom2: object,
    surface_lmax: int,
    exposure_lmax: int,
    radial_quadrature_order: int = 96,
) -> SmoothHarmonicExposureSnapshot:
    return SmoothHarmonicExposureSnapshot(
        atomic_numbers=atomic_numbers,
        positions_angstrom=positions_angstrom,
        radii_angstrom=radii_angstrom,
        transition_width_angstrom2=transition_width_angstrom2,
        surface_lmax=surface_lmax,
        exposure_lmax=exposure_lmax,
        radial_quadrature_order=radial_quadrature_order,
    )


__all__ = [
    "HARMONIC_EXPOSURE_MAXIMUM_ALGEBRAIC_DEGREE",
    "SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID",
    "SMOOTH_HARMONIC_EXPOSURE_PROVIDER_ID",
    "SmoothHarmonicExposureSnapshot",
    "build_smooth_harmonic_exposure",
    "harmonic_multiplication_matrix",
    "project_harmonic_product",
    "smooth_flat_step",
    "smooth_pair_exposure_coefficients",
]
