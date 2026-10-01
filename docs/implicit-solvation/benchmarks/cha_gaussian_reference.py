"""Independent NumPy/SciPy reference for Gaussian-sign continuum CHA.

This validation-only oracle deliberately shares only the frozen independent
NumPy/SciPy R6 reference and the existing NumPy sphere-union nonpolar kernels.
It does not import Torch, the Gaussian production scalar, its correction, or
optimizer/runtime helpers.  Geometry preparation is width-independent and may
be reused only through the immutable, identity-bound object returned here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import islice
import math
from types import MappingProxyType
from typing import Iterable, Mapping, cast

import numpy as np
from scipy import special

if __package__:
    from .cha_continuum_reference import (
        UnsupportedR6Geometry,
        r6_inverse_born_reference,
    )
else:
    from cha_continuum_reference import (
        UnsupportedR6Geometry,
        r6_inverse_born_reference,
    )
from maple.function.calculator.extra_correction.implicit.sphere_union_dispersion import (
    dispersion_energy_and_gradient,
)
from maple.function.calculator.extra_correction.implicit.sphere_union_volume import (
    volume_and_gradient,
)

AMBER_CHARGE_SCALE = 18.2223
ALPB_ALPHA = 0.571412
CHA_TAU = 1.47
CHA_ROH_ANGSTROM = 0.586
# Keep the native gb_read expression explicit rather than substituting 0.88.
EFFECTIVE_PROBE_ANGSTROM = float(1.4 - 0.52)
GAUSSIAN_SIZE_CAP_ANGSTROM = 9.5
CAVITY_PROBE_ANGSTROM = 1.3
CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 = 0.0378
CAVITY_OFFSET_KCAL_MOL = -0.5692
DISPERSION_PROBE_ANGSTROM = 0.557
WATER_OXYGEN_RMIN_ANGSTROM = 1.7683
WATER_OXYGEN_EPSILON_KCAL_MOL = 0.1520
WATER_DENSITY_PER_ANGSTROM3 = 0.03333 * 1.129
_TWO_TO_NEGATIVE_ONE_SIXTH = 2.0 ** (-1.0 / 6.0)


class GaussianChaSizeDomainError(ValueError):
    """Raised when the separately versioned strict size domain is left."""


class _ImmutableArray(np.ndarray):
    """Bytes-backed ndarray whose scientific shape metadata cannot be changed."""

    def __setattr__(self, name, value):
        if name in {"shape", "strides", "dtype", "data"}:
            raise AttributeError("immutable reference arrays have fixed metadata")
        super().__setattr__(name, value)

    def resize(self, *args, **kwargs):
        raise ValueError("immutable reference arrays cannot be resized")


@dataclass(frozen=True)
class GaussianR6RefinementLevel:
    azimuth_order: int
    inverse_cube_per_angstrom3: np.ndarray
    inverse_born_per_angstrom: np.ndarray
    gauss_closure_vector_angstrom2: np.ndarray
    inverse_cube_quad_error_estimate_per_angstrom3: float
    meridian_evaluations: int


@dataclass(frozen=True)
class GaussianReferenceGeometry:
    positions_angstrom: np.ndarray
    charges_e: np.ndarray
    cha_radii_angstrom: np.ndarray
    lj_rmin_angstrom: np.ndarray
    lj_epsilon_kcal_mol: np.ndarray
    inverse_cube_per_angstrom3: np.ndarray
    inverse_born_per_angstrom: np.ndarray
    effective_charges_e: np.ndarray
    electrostatic_size_angstrom: float
    cavity_volume_angstrom3: float
    cavity_kcal_mol: float
    dispersion_kcal_mol: float
    r6_levels: tuple[GaussianR6RefinementLevel, ...]
    r6_diagnostics: Mapping[str, object]
    diagnostics: Mapping[str, object]
    coordinates_sha256: str
    parameters_sha256: str
    identity_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class GaussianChaReferenceResult:
    energies_kcal_mol: Mapping[str, float]
    effective_charges_e: np.ndarray
    born_radii_angstrom: np.ndarray
    smoothed_signs: np.ndarray
    gaussian_tail: np.ndarray
    charge_sigma_e: np.ndarray
    shifted_inverse_born_per_angstrom: np.ndarray
    cha_factors: np.ndarray
    inverse_cube_per_angstrom3: np.ndarray
    inverse_born_per_angstrom: np.ndarray
    electrostatic_size_angstrom: float
    r6_levels: tuple[GaussianR6RefinementLevel, ...]
    r6_diagnostics: Mapping[str, object]
    diagnostics: Mapping[str, object]
    geometry_identity_sha256: str

    @property
    def polar_kcal_mol(self) -> float:
        return self.energies_kcal_mol["polar"]

    @property
    def cavity_kcal_mol(self) -> float:
        return self.energies_kcal_mol["cavity"]

    @property
    def dispersion_kcal_mol(self) -> float:
        return self.energies_kcal_mol["dispersion"]

    @property
    def total_kcal_mol(self) -> float:
        return self.energies_kcal_mol["total"]

    @property
    def weighted_signs_over_sigma(self) -> np.ndarray:
        """Compatibility name used by the production scalar diagnostics."""
        return self.charge_sigma_e

    @property
    def gaussian_erfc_tails(self) -> np.ndarray:
        """Compatibility name spelling out the direct-tail definition."""
        return self.gaussian_tail


def _readonly(values) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float64)
    payload = source.tobytes(order="C")
    return (
        np.frombuffer(payload, dtype=np.float64)
        .reshape(source.shape)
        .view(_ImmutableArray)
    )


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze(value))


def _positive_real(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive real number, not boolean.")
    try:
        number = float(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive real number.") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite real number.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return number


def _real_array(
    name: str, values, shape: tuple[int, ...], *, positive=False, nonnegative=False
):
    raw = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in raw.flat):
        raise TypeError(f"{name} must contain real numbers, not booleans.")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numbers.") from exc
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return np.array(result, copy=True)


def _orders(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(islice(iter(values), 4))
    if len(result) > 3:
        raise ValueError("azimuth_orders may contain at most three refinement levels.")
    if not result:
        raise ValueError("azimuth_orders must contain integers at least 8.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 8
        for value in result
    ):
        raise ValueError("azimuth_orders must contain integers at least 8.")
    normalized = tuple(int(value) for value in result)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("azimuth_orders must be unique and strictly increasing.")
    if normalized[-1] > 128:
        raise ValueError("public oracle azimuthal order is limited to at most 128.")
    return normalized


def _electrostatic_size(positions: np.ndarray, radii: np.ndarray) -> float:
    weights = radii**3
    mass = float(np.sum(weights))
    center = np.sum(weights[:, None] * positions, axis=0) / mass
    centered = positions - center
    second_moment = centered.T @ (weights[:, None] * centered)
    sphere_moment = (2.0 / 5.0) * float(np.sum(weights * radii**2))
    inertia = (np.trace(second_moment) + sphere_moment) * np.eye(3) - second_moment
    determinant = float(np.linalg.det(inertia))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise RuntimeError("CHA electrostatic inertia determinant is not positive.")
    size = math.sqrt(2.5 / mass) * determinant ** (1.0 / 6.0)
    if not math.isfinite(size) or size <= 0.0:
        raise RuntimeError("CHA electrostatic size is not positive and finite.")
    return size


def _effective_charges(
    positions: np.ndarray, charges: np.ndarray, born: np.ndarray
) -> np.ndarray:
    differences = positions[:, None, :] - positions[None, :, :]
    distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
    native_charges = charges * AMBER_CHARGE_SCALE
    born_products = born[:, None] * born[None, :]
    weights = np.exp(-CHA_TAU * distances_squared / born_products)
    return (weights @ native_charges) / AMBER_CHARGE_SCALE


def _hash_arrays(label: str, *arrays: np.ndarray, extras=()) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    for extra in extras:
        digest.update(repr(extra).encode("ascii"))
    return digest.hexdigest()


def _parameter_hash(
    charges: np.ndarray,
    radii: np.ndarray,
    rmin: np.ndarray,
    epsilon: np.ndarray,
    orders: tuple[int, ...],
    epsabs: float,
    epsrel: float,
) -> str:
    return _hash_arrays(
        "gaussian-cha-parameters-v1",
        charges,
        radii,
        rmin,
        epsilon,
        extras=(
            orders,
            epsabs,
            epsrel,
            AMBER_CHARGE_SCALE,
            ALPB_ALPHA,
            CHA_TAU,
            CHA_ROH_ANGSTROM,
            EFFECTIVE_PROBE_ANGSTROM,
            GAUSSIAN_SIZE_CAP_ANGSTROM,
            CAVITY_PROBE_ANGSTROM,
            CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3,
            CAVITY_OFFSET_KCAL_MOL,
            DISPERSION_PROBE_ANGSTROM,
            WATER_OXYGEN_RMIN_ANGSTROM,
            WATER_OXYGEN_EPSILON_KCAL_MOL,
            WATER_DENSITY_PER_ANGSTROM3,
        ),
    )


def _payload_hash(
    positions: np.ndarray,
    charges: np.ndarray,
    radii: np.ndarray,
    rmin: np.ndarray,
    epsilon: np.ndarray,
    inverse_cube: np.ndarray,
    inverse_born: np.ndarray,
    effective_charges: np.ndarray,
    levels: tuple[GaussianR6RefinementLevel, ...],
    *,
    electrostatic_size: float,
    cavity_volume: float,
    cavity_energy: float,
    dispersion_energy: float,
    coordinates_sha256: str,
    parameters_sha256: str,
    identity_sha256: str,
) -> str:
    level_arrays = tuple(
        array
        for level in levels
        for array in (
            level.inverse_cube_per_angstrom3,
            level.inverse_born_per_angstrom,
            level.gauss_closure_vector_angstrom2,
        )
    )
    level_metadata = tuple(
        (
            level.azimuth_order,
            level.inverse_cube_quad_error_estimate_per_angstrom3,
            level.meridian_evaluations,
        )
        for level in levels
    )
    return _hash_arrays(
        "gaussian-cha-prepared-payload-v1",
        positions,
        charges,
        radii,
        rmin,
        epsilon,
        inverse_cube,
        inverse_born,
        effective_charges,
        *level_arrays,
        extras=(
            electrostatic_size,
            cavity_volume,
            cavity_energy,
            dispersion_energy,
            coordinates_sha256,
            parameters_sha256,
            identity_sha256,
            level_metadata,
        ),
    )


def _assert_geometry_integrity(geometry: GaussianReferenceGeometry) -> None:
    try:
        stored_orders = cast(tuple[object, ...], geometry.diagnostics["azimuth_orders"])
        orders = tuple(int(cast(int, value)) for value in stored_orders)
        epsabs = float(cast(float, geometry.diagnostics["epsabs"]))
        epsrel = float(cast(float, geometry.diagnostics["epsrel"]))
        coordinates_sha256 = _hash_arrays(
            "gaussian-cha-coordinates-v1", geometry.positions_angstrom
        )
        parameters_sha256 = _parameter_hash(
            geometry.charges_e,
            geometry.cha_radii_angstrom,
            geometry.lj_rmin_angstrom,
            geometry.lj_epsilon_kcal_mol,
            orders,
            epsabs,
            epsrel,
        )
        identity_sha256 = hashlib.sha256(
            f"gaussian-cha-geometry-v1:{coordinates_sha256}:{parameters_sha256}".encode(
                "ascii"
            )
        ).hexdigest()
        payload_sha256 = _payload_hash(
            geometry.positions_angstrom,
            geometry.charges_e,
            geometry.cha_radii_angstrom,
            geometry.lj_rmin_angstrom,
            geometry.lj_epsilon_kcal_mol,
            geometry.inverse_cube_per_angstrom3,
            geometry.inverse_born_per_angstrom,
            geometry.effective_charges_e,
            geometry.r6_levels,
            electrostatic_size=geometry.electrostatic_size_angstrom,
            cavity_volume=geometry.cavity_volume_angstrom3,
            cavity_energy=geometry.cavity_kcal_mol,
            dispersion_energy=geometry.dispersion_kcal_mol,
            coordinates_sha256=coordinates_sha256,
            parameters_sha256=parameters_sha256,
            identity_sha256=identity_sha256,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "prepared Gaussian reference payload integrity failed"
        ) from exc
    if (
        coordinates_sha256 != geometry.coordinates_sha256
        or parameters_sha256 != geometry.parameters_sha256
        or identity_sha256 != geometry.identity_sha256
        or payload_sha256 != geometry.payload_sha256
    ):
        raise RuntimeError("prepared Gaussian reference payload integrity failed")


def _frozen_levels(levels) -> tuple[GaussianR6RefinementLevel, ...]:
    return tuple(
        GaussianR6RefinementLevel(
            azimuth_order=int(level.azimuth_order),
            inverse_cube_per_angstrom3=_readonly(level.inverse_cube_per_angstrom3),
            inverse_born_per_angstrom=_readonly(level.inverse_born_per_angstrom),
            gauss_closure_vector_angstrom2=_readonly(
                level.gauss_closure_vector_angstrom2
            ),
            inverse_cube_quad_error_estimate_per_angstrom3=float(
                level.inverse_cube_quad_error_estimate_per_angstrom3
            ),
            meridian_evaluations=int(level.meridian_evaluations),
        )
        for level in levels
    )


def prepare_gaussian_reference_geometry(
    positions,
    charges_e,
    cha_radii,
    lj_rmin,
    lj_epsilon,
    *,
    azimuth_orders: Iterable[int] = (64, 96, 128),
    epsabs: float = 1.0e-12,
    epsrel: float = 1.0e-12,
) -> GaussianReferenceGeometry:
    """Prepare immutable width-independent R6/Born and nonpolar geometry."""

    raw_positions = np.asarray(positions, dtype=object)
    if raw_positions.ndim != 2 or raw_positions.shape[1:] != (3,):
        raise ValueError("positions must have nonempty shape (N, 3).")
    count = raw_positions.shape[0]
    if count == 0:
        raise ValueError("positions must have nonempty shape (N, 3).")
    if count > 3:
        raise UnsupportedR6Geometry(
            "the independent exterior proof covers at most three sites."
        )
    centers = _real_array("positions", positions, (count, 3))
    charges = _real_array("charges_e", charges_e, (count,))
    radii = _real_array("cha_radii", cha_radii, (count,), positive=True)
    rmin = _real_array("lj_rmin", lj_rmin, (count,), positive=True)
    epsilon = _real_array("lj_epsilon", lj_epsilon, (count,), nonnegative=True)
    orders = _orders(azimuth_orders)
    absolute_tolerance = _positive_real("epsabs", epsabs)
    relative_tolerance = _positive_real("epsrel", epsrel)

    electrostatic_size = _electrostatic_size(centers, radii)
    # Fail closed at the cap.  The inertia expression can round a mathematically
    # exact one-site radius down by one ulp, so include a scale-aware roundoff
    # guard rather than accidentally admitting equality.
    size_cap_guard = 8.0 * np.finfo(np.float64).eps * GAUSSIAN_SIZE_CAP_ANGSTROM
    if electrostatic_size >= GAUSSIAN_SIZE_CAP_ANGSTROM - size_cap_guard:
        raise GaussianChaSizeDomainError(
            "Gaussian-CHA v1 requires electrostatic size strictly below 9.5 angstrom."
        )

    r6 = r6_inverse_born_reference(
        centers,
        radii,
        probe_angstrom=EFFECTIVE_PROBE_ANGSTROM,
        azimuth_orders=orders,
        epsabs=absolute_tolerance,
        epsrel=relative_tolerance,
    )
    inverse_born = np.asarray(r6.inverse_born_per_angstrom, dtype=np.float64)
    if np.any(inverse_born <= 0.0) or not np.all(np.isfinite(inverse_born)):
        raise RuntimeError("R6 inverse Born radii must be positive and finite.")
    born = 1.0 / inverse_born
    effective_charges = _effective_charges(centers, charges, born)

    cavity = volume_and_gradient(
        centers,
        rmin + CAVITY_PROBE_ANGSTROM,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    cavity_energy = (
        CAVITY_COEFFICIENT_KCAL_MOL_ANGSTROM3 * cavity.volume + CAVITY_OFFSET_KCAL_MOL
    )
    sigma = (rmin + WATER_OXYGEN_RMIN_ANGSTROM) * _TWO_TO_NEGATIVE_ONE_SIXTH
    mixed_epsilon = np.sqrt(epsilon * WATER_OXYGEN_EPSILON_KCAL_MOL)
    dispersion = dispersion_energy_and_gradient(
        centers,
        rmin + DISPERSION_PROBE_ANGSTROM,
        sigma,
        mixed_epsilon,
        WATER_DENSITY_PER_ANGSTROM3,
        phi_order=orders[-1],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    coordinates_sha256 = _hash_arrays("gaussian-cha-coordinates-v1", centers)
    parameters_sha256 = _parameter_hash(
        charges,
        radii,
        rmin,
        epsilon,
        orders,
        absolute_tolerance,
        relative_tolerance,
    )
    identity_sha256 = hashlib.sha256(
        f"gaussian-cha-geometry-v1:{coordinates_sha256}:{parameters_sha256}".encode(
            "ascii"
        )
    ).hexdigest()
    diagnostics = _freeze_mapping(
        {
            "cavity_mixed_vector_quad_error": cavity.quadrature_error,
            "dispersion_mixed_vector_quad_error": dispersion.quadrature_error,
            "mixed_vector_quad_errors_are_scalar_uncertainties": False,
            "error_estimates_are_rigorous_bounds": False,
            "shared_numpy_reference_helpers": (
                "cha_continuum_reference.r6_inverse_born_reference",
                "sphere_union_volume.volume_and_gradient",
                "sphere_union_dispersion.dispersion_energy_and_gradient",
            ),
            "torch_gaussian_runtime_helpers_used": False,
            "torch_ses_helpers_used": False,
            "incidental_torch_import_via_maple_package_initialization": True,
            "incidental_import_closure_note": (
                "Importing the shared NumPy helpers executes MAPLE calculator "
                "package initializers, which load torch and calculator dispatch "
                "modules; this oracle calls none of those incidental modules."
            ),
            "azimuth_orders": orders,
            "epsabs": absolute_tolerance,
            "epsrel": relative_tolerance,
        }
    )
    frozen_positions = _readonly(centers)
    frozen_charges = _readonly(charges)
    frozen_radii = _readonly(radii)
    frozen_rmin = _readonly(rmin)
    frozen_epsilon = _readonly(epsilon)
    frozen_inverse_cube = _readonly(r6.inverse_cube_per_angstrom3)
    frozen_inverse_born = _readonly(inverse_born)
    frozen_effective_charges = _readonly(effective_charges)
    frozen_levels = _frozen_levels(r6.levels)
    payload_sha256 = _payload_hash(
        frozen_positions,
        frozen_charges,
        frozen_radii,
        frozen_rmin,
        frozen_epsilon,
        frozen_inverse_cube,
        frozen_inverse_born,
        frozen_effective_charges,
        frozen_levels,
        electrostatic_size=electrostatic_size,
        cavity_volume=float(cavity.volume),
        cavity_energy=float(cavity_energy),
        dispersion_energy=float(dispersion.energy),
        coordinates_sha256=coordinates_sha256,
        parameters_sha256=parameters_sha256,
        identity_sha256=identity_sha256,
    )
    return GaussianReferenceGeometry(
        positions_angstrom=frozen_positions,
        charges_e=frozen_charges,
        cha_radii_angstrom=frozen_radii,
        lj_rmin_angstrom=frozen_rmin,
        lj_epsilon_kcal_mol=frozen_epsilon,
        inverse_cube_per_angstrom3=frozen_inverse_cube,
        inverse_born_per_angstrom=frozen_inverse_born,
        effective_charges_e=frozen_effective_charges,
        electrostatic_size_angstrom=electrostatic_size,
        cavity_volume_angstrom3=float(cavity.volume),
        cavity_kcal_mol=float(cavity_energy),
        dispersion_kcal_mol=float(dispersion.energy),
        r6_levels=frozen_levels,
        r6_diagnostics=_freeze_mapping(r6.diagnostics),
        diagnostics=diagnostics,
        coordinates_sha256=coordinates_sha256,
        parameters_sha256=parameters_sha256,
        identity_sha256=identity_sha256,
        payload_sha256=payload_sha256,
    )


def gaussian_cha_from_reference_geometry(
    geometry: GaussianReferenceGeometry, *, sigma_e
) -> GaussianChaReferenceResult:
    """Evaluate the independent Gaussian-sign CHA scalar on prepared geometry."""

    if not isinstance(geometry, GaussianReferenceGeometry):
        raise TypeError("geometry must be a GaussianReferenceGeometry.")
    _assert_geometry_integrity(geometry)
    width = _positive_real("sigma_e", sigma_e)
    centers = geometry.positions_angstrom
    charges = geometry.charges_e
    inverse_born = geometry.inverse_born_per_angstrom
    born = 1.0 / inverse_born
    effective_charges = geometry.effective_charges_e
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        charge_sigma = effective_charges / width
    if np.any(np.isnan(charge_sigma)):
        raise RuntimeError("Gaussian charge/width diagnostic produced NaN.")
    charge_sigma_tail_saturated = bool(np.isinf(charge_sigma).any())
    gaussian_argument = charge_sigma / math.sqrt(2.0)
    smoothed_signs = special.erf(gaussian_argument)
    # Do not obtain this diagnostic by subtracting erf from unity: direct erfc
    # retains the narrow far-tail quantity without catastrophic cancellation.
    gaussian_tail = special.erfc(np.abs(gaussian_argument))
    cha_factors = 1.0 + smoothed_signs * CHA_ROH_ANGSTROM / (
        born + EFFECTIVE_PROBE_ANGSTROM
    )
    if np.any(cha_factors <= 0.0) or not np.all(np.isfinite(cha_factors)):
        raise RuntimeError("Gaussian CHA factors are not positive and finite.")

    differences = centers[:, None, :] - centers[None, :, :]
    distances_squared = np.einsum("ijk,ijk->ij", differences, differences)
    native_charges = charges * AMBER_CHARGE_SCALE
    born_products = born[:, None] * born[None, :]
    beta = ALPB_ALPHA / 78.5
    dielectric = (1.0 - 1.0 / 78.5) / (1.0 + beta)
    size_term = beta / geometry.electrostatic_size_angstrom
    self_energy = (
        -0.5
        * dielectric
        * np.sum(native_charges**2 * (inverse_born / cha_factors + size_term))
    )
    first, second = np.triu_indices(len(centers), 1)
    pair_r2 = distances_squared[first, second]
    pair_born = born_products[first, second]
    radicand = (
        pair_r2
        + pair_born
        * np.exp(-0.25 * pair_r2 / pair_born)
        * cha_factors[first]
        * cha_factors[second]
    )
    if np.any(radicand <= 0.0) or not np.all(np.isfinite(radicand)):
        raise RuntimeError("Gaussian CHA pair radicands are not positive and finite.")
    pair_energy = -dielectric * np.sum(
        native_charges[first] * native_charges[second] * (radicand**-0.5 + size_term)
    )
    polar = float(self_energy + pair_energy)
    total = polar + geometry.cavity_kcal_mol + geometry.dispersion_kcal_mol
    energies = MappingProxyType(
        {
            "polar": polar,
            "cavity": geometry.cavity_kcal_mol,
            "dispersion": geometry.dispersion_kcal_mol,
            "total": float(total),
        }
    )
    diagnostics = _freeze_mapping(
        {
            "sigma_e": width,
            "gaussian_tail_definition": "scipy.special.erfc(abs(S_e)/(sqrt(2)*sigma_e))",
            "size_shift_per_angstrom": 0.0,
            "size_domain": "electrostatic_size_angstrom < 9.5",
            "geometry_reused_only_for_identical_identity": True,
            "coordinates_sha256": geometry.coordinates_sha256,
            "parameters_sha256": geometry.parameters_sha256,
            "payload_sha256": geometry.payload_sha256,
            "charge_sigma_tail_saturated": charge_sigma_tail_saturated,
        }
    )
    return GaussianChaReferenceResult(
        energies_kcal_mol=energies,
        effective_charges_e=_readonly(effective_charges),
        born_radii_angstrom=_readonly(born),
        smoothed_signs=_readonly(smoothed_signs),
        gaussian_tail=_readonly(gaussian_tail),
        charge_sigma_e=_readonly(charge_sigma),
        shifted_inverse_born_per_angstrom=_readonly(inverse_born),
        cha_factors=_readonly(cha_factors),
        inverse_cube_per_angstrom3=_readonly(geometry.inverse_cube_per_angstrom3),
        inverse_born_per_angstrom=_readonly(inverse_born),
        electrostatic_size_angstrom=geometry.electrostatic_size_angstrom,
        r6_levels=geometry.r6_levels,
        r6_diagnostics=geometry.r6_diagnostics,
        diagnostics=diagnostics,
        geometry_identity_sha256=geometry.identity_sha256,
    )


def gaussian_cha_reference(
    positions,
    charges_e,
    cha_radii,
    lj_rmin,
    lj_epsilon,
    *,
    sigma_e,
    azimuth_orders: Iterable[int] = (64, 96, 128),
    epsabs: float = 1.0e-12,
    epsrel: float = 1.0e-12,
) -> GaussianChaReferenceResult:
    """Prepare geometry and evaluate one Gaussian width without hidden caching."""

    geometry = prepare_gaussian_reference_geometry(
        positions,
        charges_e,
        cha_radii,
        lj_rmin,
        lj_epsilon,
        azimuth_orders=azimuth_orders,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return gaussian_cha_from_reference_geometry(geometry, sigma_e=sigma_e)


__all__ = [
    "GaussianChaReferenceResult",
    "GaussianChaSizeDomainError",
    "GaussianR6RefinementLevel",
    "GaussianReferenceGeometry",
    "gaussian_cha_from_reference_geometry",
    "gaussian_cha_reference",
    "prepare_gaussian_reference_geometry",
]
