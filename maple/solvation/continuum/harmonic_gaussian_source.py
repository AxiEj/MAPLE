"""Geometry-assembled radial-GTO source map in harmonic coefficient space.

The map is built from invariant one-dimensional Gaussian-potential integrals,
complete real harmonic irreps, and analytic SO(3) Lie-algebra derivatives.
No surface-point sampling or independently implemented receiver is used.  The
receiver is exactly the transpose of the weighted source map.

This module closes the harmonic ``S(R)`` source/intertwiner subproblem only.
It does not assemble the continuum Green operator ``A(R)`` and does not expose
Route-2 E/F/H/V/M capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.units import Bohr
from scipy.special import erf

from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SIGMAS_ANGSTROM,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

from .harmonic_coefficients import (
    HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID,
    PerAtomHarmonicSpace,
    _bounded_lmax,
    _positive_int,
    _real_harmonic_design,
    real_wigner_generators,
)
from .harmonic_exposure import SmoothHarmonicExposureSnapshot, _legendre_rule

HARMONIC_GAUSSIAN_SOURCE_CONTRACT_ID = (
    "maple.route2.coupling.radial-gto-to-harmonic-potential.v1"
)
HARMONIC_GAUSSIAN_SOURCE_PROVIDER_ID = (
    "maple.route2.coupling.harmonic-gaussian-source.impl.v1"
)


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
    continuum = Path(__file__).resolve().parent
    coupling = continuum.parent / "coupling"
    api = continuum.parent / "api"
    files = {
        "maple.solvation.continuum.harmonic_coefficients": (
            continuum / "harmonic_coefficients.py"
        ),
        "maple.solvation.continuum.harmonic_exposure": (
            continuum / "harmonic_exposure.py"
        ),
        "maple.solvation.continuum.harmonic_gaussian_source": Path(__file__),
        "maple.solvation.coupling.metrics": coupling / "metrics.py",
        "maple.solvation.coupling.spaces": coupling / "spaces.py",
        "maple.solvation.api.units": api / "units.py",
    }
    return tuple(
        (name, hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in sorted(files.items())
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _geometry(
    positions_angstrom: object, radii_angstrom: object
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_angstrom, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "positions_angstrom must be finite with shape (atom_count, 3)."
        )
    radii = np.asarray(radii_angstrom, dtype=float)
    if (
        radii.shape != (len(positions),)
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError(
            "radii_angstrom must be finite and positive with shape " "(atom_count,)."
        )
    return (
        np.ascontiguousarray(positions, dtype=float),
        np.ascontiguousarray(radii, dtype=float),
    )


def _gaussian_monopole_kernel_and_radial_derivative(
    radius_angstrom: np.ndarray, *, sigma_angstrom: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``K(r)`` and ``dK/dr_A`` in atomic-potential units."""

    radius = np.asarray(radius_angstrom, dtype=float)
    sigma = float(sigma_angstrom)
    if (
        not np.all(np.isfinite(radius))
        or np.any(radius < 0.0)
        or not np.isfinite(sigma)
        or sigma <= 0.0
    ):
        raise ValueError("Gaussian radii and sigma must be finite and non-negative.")
    radius_bohr = radius / Bohr
    sigma_bohr = sigma / Bohr
    x = radius_bohr / (np.sqrt(2.0) * sigma_bohr)
    kernel = np.empty_like(radius_bohr)
    derivative_bohr = np.empty_like(radius_bohr)
    small = np.abs(x) < 1.0e-3
    if np.any(small):
        x_small = x[small]
        x2 = x_small * x_small
        prefactor = np.sqrt(2.0 / np.pi) / sigma_bohr
        kernel[small] = prefactor * (
            1.0 - x2 / 3.0 + x2**2 / 10.0 - x2**3 / 42.0 + x2**4 / 216.0
        )
        derivative_bohr[small] = (
            prefactor
            / (np.sqrt(2.0) * sigma_bohr)
            * (
                -2.0 * x_small / 3.0
                + 2.0 * x_small**3 / 5.0
                - x_small**5 / 7.0
                + x_small**7 / 27.0
            )
        )
    regular = ~small
    if np.any(regular):
        radius_regular = radius_bohr[regular]
        x_regular = x[regular]
        erf_values = erf(x_regular)
        kernel[regular] = erf_values / radius_regular
        derivative_bohr[regular] = np.sqrt(2.0 / np.pi) * np.exp(-(x_regular**2)) / (
            sigma_bohr * radius_regular
        ) - erf_values / (radius_regular**2)
    return kernel, derivative_bohr / Bohr


def _zonal_coefficients_and_distance_derivative(
    *,
    target_radius_angstrom: float,
    distance_angstrom: float,
    sigma_angstrom: float,
    lmax: int,
    radial_quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    cosine, weights = _legendre_rule(radial_quadrature_order)
    radius_squared = (
        target_radius_angstrom**2
        + distance_angstrom**2
        - 2.0 * target_radius_angstrom * distance_angstrom * cosine
    )
    radius = np.sqrt(np.maximum(0.0, radius_squared))
    kernel, radial_derivative = _gaussian_monopole_kernel_and_radial_derivative(
        radius, sigma_angstrom=sigma_angstrom
    )
    radial_direction = np.zeros_like(radius)
    nonzero = radius > 1.0e-14
    radial_direction[nonzero] = (
        distance_angstrom - target_radius_angstrom * cosine[nonzero]
    ) / radius[nonzero]
    legendre = np.polynomial.legendre.legvander(cosine, lmax)
    coefficients = 2.0 * np.pi * ((weights * kernel) @ legendre)
    derivatives = (
        2.0 * np.pi * ((weights * radial_derivative * radial_direction) @ legendre)
    )
    return coefficients, derivatives


def _distinct_pair_block(
    *,
    displacement_angstrom: np.ndarray,
    target_radius_angstrom: float,
    sigma_angstrom: float,
    lmax: int,
    radial_quadrature_order: int,
) -> np.ndarray:
    distance = float(np.linalg.norm(displacement_angstrom))
    direction = displacement_angstrom / distance
    zonal, zonal_derivative = _zonal_coefficients_and_distance_derivative(
        target_radius_angstrom=target_radius_angstrom,
        distance_angstrom=distance,
        sigma_angstrom=sigma_angstrom,
        lmax=lmax,
        radial_quadrature_order=radial_quadrature_order,
    )
    harmonics = _real_harmonic_design(direction[None, :], lmax=lmax)[0]
    generators = real_wigner_generators(lmax=lmax)
    coefficient = np.empty((lmax + 1) ** 2, dtype=float)
    jacobian = np.empty(((lmax + 1) ** 2, 3), dtype=float)
    for ell in range(lmax + 1):
        section = slice(ell * ell, (ell + 1) * (ell + 1))
        coefficient[section] = zonal[ell] * harmonics[section]
    for axis in range(3):
        radial_change = direction[axis]
        direction_change = (np.eye(3)[axis] - direction * radial_change) / distance
        rotation_vector = np.cross(direction, direction_change)
        harmonic_change = sum(
            rotation_vector[generator_axis] * (generators[generator_axis] @ harmonics)
            for generator_axis in range(3)
        )
        for ell in range(lmax + 1):
            section = slice(ell * ell, (ell + 1) * (ell + 1))
            jacobian[section, axis] = (
                zonal_derivative[ell] * radial_change * harmonics[section]
                + zonal[ell] * harmonic_change[section]
            )
    block = np.empty(((lmax + 1) ** 2, 4), dtype=float)
    block[:, 0] = coefficient
    # Authoritative raw l=1 order is (m0,m1,m-1)=(y,z,x).
    block[:, 1] = jacobian[:, 1]
    block[:, 2] = jacobian[:, 2]
    block[:, 3] = jacobian[:, 0]
    return block


def _self_pair_block(
    *, target_radius_angstrom: float, sigma_angstrom: float, lmax: int
) -> np.ndarray:
    kernel, radial_derivative = _gaussian_monopole_kernel_and_radial_derivative(
        np.asarray([target_radius_angstrom]), sigma_angstrom=sigma_angstrom
    )
    block = np.zeros(((lmax + 1) ** 2, 4), dtype=float)
    block[0, 0] = np.sqrt(4.0 * np.pi) * kernel[0]
    if lmax >= 1:
        dipole_amplitude = -radial_derivative[0] * np.sqrt(4.0 * np.pi / 3.0)
        block[1:4, 1:4] = dipole_amplitude * np.eye(3)
    return block


def gaussian_harmonic_source_operator(
    *,
    positions_angstrom: object,
    radii_angstrom: object,
    surface_lmax: int,
    radial_quadrature_order: int = 128,
) -> np.ndarray:
    """Return the raw geometry-assembled ``S(R)`` in eV-dual units."""

    positions, radii = _geometry(positions_angstrom, radii_angstrom)
    maximum = _bounded_lmax(surface_lmax)
    order = _positive_int(radial_quadrature_order, name="radial_quadrature_order")
    if order > 4096:
        raise ValueError("radial_quadrature_order exceeds the bounded contract.")
    atom_count = len(positions)
    block_dimension = (maximum + 1) ** 2
    operator = np.zeros((atom_count * block_dimension, atom_count * 8), dtype=float)
    radial_layouts = ((0, (0, 2, 3, 4)), (1, (1, 5, 6, 7)))
    for target in range(atom_count):
        row = slice(target * block_dimension, (target + 1) * block_dimension)
        for source in range(atom_count):
            displacement = positions[source] - positions[target]
            distance = float(np.linalg.norm(displacement))
            for radial_index, columns in radial_layouts:
                sigma = MACE_POLAR_RADIAL_GTO_SIGMAS_ANGSTROM[radial_index]
                if distance <= 1.0e-14:
                    block = _self_pair_block(
                        target_radius_angstrom=float(radii[target]),
                        sigma_angstrom=sigma,
                        lmax=maximum,
                    )
                else:
                    block = _distinct_pair_block(
                        displacement_angstrom=displacement,
                        target_radius_angstrom=float(radii[target]),
                        sigma_angstrom=sigma,
                        lmax=maximum,
                        radial_quadrature_order=order,
                    )
                for local_column, public_component in enumerate(columns):
                    operator[row, source * 8 + public_component] = (
                        HARTREE_TO_EV * block[:, local_column]
                    )
    return _readonly(operator)


@dataclass(frozen=True, slots=True, init=False)
class HarmonicGaussianSourceSnapshot:
    """Immutable exposed radial-GTO source map and exact transpose receiver."""

    exposure: SmoothHarmonicExposureSnapshot
    radial_quadrature_order: int
    configuration_sha256: str
    provenance_sha256: str
    state_sha256: str
    _raw_source_values: tuple[float, ...]
    _source_values: tuple[float, ...]

    def __init__(
        self,
        exposure: SmoothHarmonicExposureSnapshot,
        *,
        radial_quadrature_order: int = 128,
    ) -> None:
        if not isinstance(exposure, SmoothHarmonicExposureSnapshot):
            raise TypeError("exposure must be SmoothHarmonicExposureSnapshot.")
        exposure.validate()
        order = _positive_int(radial_quadrature_order, name="radial_quadrature_order")
        if order > 4096:
            raise ValueError("radial_quadrature_order exceeds the bounded contract.")
        raw = gaussian_harmonic_source_operator(
            positions_angstrom=exposure.positions_angstrom,
            radii_angstrom=exposure.radii_angstrom,
            surface_lmax=exposure.surface_lmax,
            radial_quadrature_order=order,
        )
        weighted = exposure.global_multiplication_operator @ raw
        configuration = _sha(
            {
                "contract_id": HARMONIC_GAUSSIAN_SOURCE_CONTRACT_ID,
                "provider_id": HARMONIC_GAUSSIAN_SOURCE_PROVIDER_ID,
                "coefficient_contract_id": (HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID),
                "exposure_configuration_sha256": exposure.configuration_sha256,
                "surface_space_sha256": exposure.surface_space.metadata_sha256(),
                "source_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.metadata_hash()
                ),
                "field_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.metadata_hash()
                ),
                "pairing_sha256": MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
                "sigmas_angstrom": MACE_POLAR_RADIAL_GTO_SIGMAS_ANGSTROM,
                "radial_quadrature_order": order,
                "bohr_to_angstrom": format(Bohr, ".17g"),
                "hartree_to_ev": format(HARTREE_TO_EV, ".17g"),
                "receiver": "exact-transpose-of-weighted-source-operator",
                "capabilities": "none",
            }
        )
        provenance = _sha(
            {
                "configuration_sha256": configuration,
                "implementation_sha256": _implementation_sha256(),
                "legacy_adapter_dependency": False,
            }
        )
        state = _sha(
            {
                "configuration_sha256": configuration,
                "provenance_sha256": provenance,
                "exposure_state_sha256": exposure.state_sha256,
                "raw_source_operator_sha256": _array_sha256(raw),
                "source_operator_sha256": _array_sha256(weighted),
            }
        )
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "radial_quadrature_order", order)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "state_sha256", state)
        object.__setattr__(
            self, "_raw_source_values", tuple(float(value) for value in raw.flat)
        )
        object.__setattr__(
            self, "_source_values", tuple(float(value) for value in weighted.flat)
        )

    @property
    def atom_count(self) -> int:
        return self.exposure.atom_count

    @property
    def surface_lmax(self) -> int:
        return self.exposure.surface_lmax

    @property
    def surface_space(self) -> PerAtomHarmonicSpace:
        return self.exposure.surface_space

    @property
    def raw_source_operator(self) -> np.ndarray:
        return _readonly(
            np.asarray(self._raw_source_values, dtype=float).reshape(
                self.surface_space.dimension, self.atom_count * 8
            )
        )

    @property
    def source_operator(self) -> np.ndarray:
        return _readonly(
            np.asarray(self._source_values, dtype=float).reshape(
                self.surface_space.dimension, self.atom_count * 8
            )
        )

    @property
    def coordinate_derivative_available(self) -> bool:
        return False

    @property
    def tier_v_admitted(self) -> bool:
        return False

    def apply_source(self, source: object) -> np.ndarray:
        values = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.validate(
            source, atom_count=self.atom_count, name="source"
        )
        return self.source_operator @ values.reshape(-1)

    def apply_adjoint(self, surface_cotangent: object) -> np.ndarray:
        cotangent = np.asarray(surface_cotangent, dtype=float)
        if cotangent.shape != (self.surface_space.dimension,) or not np.all(
            np.isfinite(cotangent)
        ):
            raise ValueError(
                "surface_cotangent must be finite with one value per harmonic "
                "surface coefficient."
            )
        return (self.source_operator.T @ cotangent).reshape(self.atom_count, 8)

    def source_jvp(self, source_direction: object) -> np.ndarray:
        return self.apply_source(source_direction)

    def source_vjp(self, surface_cotangent: object) -> np.ndarray:
        return self.apply_adjoint(surface_cotangent)

    def validate(self) -> None:
        self.exposure.validate()
        raw = gaussian_harmonic_source_operator(
            positions_angstrom=self.exposure.positions_angstrom,
            radii_angstrom=self.exposure.radii_angstrom,
            surface_lmax=self.surface_lmax,
            radial_quadrature_order=self.radial_quadrature_order,
        )
        weighted = self.exposure.global_multiplication_operator @ raw
        expected_configuration = _sha(
            {
                "contract_id": HARMONIC_GAUSSIAN_SOURCE_CONTRACT_ID,
                "provider_id": HARMONIC_GAUSSIAN_SOURCE_PROVIDER_ID,
                "coefficient_contract_id": (HARMONIC_GALERKIN_COEFFICIENT_CONTRACT_ID),
                "exposure_configuration_sha256": (self.exposure.configuration_sha256),
                "surface_space_sha256": self.surface_space.metadata_sha256(),
                "source_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.metadata_hash()
                ),
                "field_space_sha256": (
                    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE.metadata_hash()
                ),
                "pairing_sha256": MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash(),
                "sigmas_angstrom": MACE_POLAR_RADIAL_GTO_SIGMAS_ANGSTROM,
                "radial_quadrature_order": self.radial_quadrature_order,
                "bohr_to_angstrom": format(Bohr, ".17g"),
                "hartree_to_ev": format(HARTREE_TO_EV, ".17g"),
                "receiver": "exact-transpose-of-weighted-source-operator",
                "capabilities": "none",
            }
        )
        expected_provenance = _sha(
            {
                "configuration_sha256": expected_configuration,
                "implementation_sha256": _implementation_sha256(),
                "legacy_adapter_dependency": False,
            }
        )
        expected_state = _sha(
            {
                "configuration_sha256": expected_configuration,
                "provenance_sha256": expected_provenance,
                "exposure_state_sha256": self.exposure.state_sha256,
                "raw_source_operator_sha256": _array_sha256(self.raw_source_operator),
                "source_operator_sha256": _array_sha256(self.source_operator),
            }
        )
        if (
            self.configuration_sha256 != expected_configuration
            or self.provenance_sha256 != expected_provenance
            or self.state_sha256 != expected_state
            or not np.array_equal(raw, self.raw_source_operator)
            or not np.array_equal(weighted, self.source_operator)
        ):
            raise RuntimeError("harmonic Gaussian source identity or content drifted.")


def build_harmonic_gaussian_source(
    exposure: SmoothHarmonicExposureSnapshot,
    *,
    radial_quadrature_order: int = 128,
) -> HarmonicGaussianSourceSnapshot:
    return HarmonicGaussianSourceSnapshot(
        exposure, radial_quadrature_order=radial_quadrature_order
    )


__all__ = [
    "HARMONIC_GAUSSIAN_SOURCE_CONTRACT_ID",
    "HARMONIC_GAUSSIAN_SOURCE_PROVIDER_ID",
    "HarmonicGaussianSourceSnapshot",
    "build_harmonic_gaussian_source",
    "gaussian_harmonic_source_operator",
]
