"""Disabled geometry assembly for a smooth weighted harmonic conductor model.

Let ``E(R)`` be the rectangular map from the retained surface-unknown space to
the complete product bandwidth of ``e_i(R,u) Y_lm(u)``.  Let ``K(R)`` be the
physical Coulomb single-layer matrix on that larger charge space and ``V(R)``
the Gaussian-source boundary potential.  This module assembles

``A = E.T K E`` and ``S = E.T V``.

The rectangular map is essential.  Replacing it by an invertible square
``P_L M_e P_L`` in both ``M K M`` and ``M V`` merely reparameterizes the
stationary solve and makes the energy independent of exposure.

This is a dense, bounded scientific reference for a regularized weighted
multi-sphere conductor functional.  The exact shell kernel is globally C1 but
not C2 at pair tangencies; a fully buried chart can also lose rank.  Analytic
coordinate pullbacks and all public E/F/H/V/M capabilities therefore remain
disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .harmonic_coefficients import PerAtomHarmonicSpace, _positive_int
from .harmonic_exposure import SmoothHarmonicExposureSnapshot
from .harmonic_galerkin import FixedHarmonicGalerkinSnapshot
from .harmonic_gaussian_source import (
    HarmonicGaussianSourceSnapshot,
    build_harmonic_gaussian_source,
)
from .harmonic_single_layer import (
    HARMONIC_SINGLE_LAYER_CONTRACT_ID,
    HARMONIC_SINGLE_LAYER_PROVIDER_ID,
    harmonic_single_layer_operator,
)

SMOOTH_WEIGHTED_HARMONIC_GALERKIN_CONTRACT_ID = (
    "maple.route2.continuum.smooth-weighted-harmonic-coulomb-galerkin.v1"
)
SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID = (
    "maple.route2.continuum.smooth-weighted-harmonic-galerkin.impl.v1"
)
MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE = 1.0e-12
MAXIMUM_REFERENCE_CONDITION_NUMBER = 1.0e12


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
        for name in (
            "harmonic_coefficients.py",
            "harmonic_exposure.py",
            "harmonic_galerkin.py",
            "harmonic_gaussian_source.py",
            "harmonic_single_layer.py",
            "harmonic_weighted_galerkin.py",
        )
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _bounded_order(value: object, *, name: str) -> int:
    result = _positive_int(value, name=name)
    if result > 4096:
        raise ValueError(f"{name} exceeds the bounded contract.")
    return result


@dataclass(frozen=True, slots=True, init=False)
class SmoothWeightedHarmonicGalerkinAssembly:
    """Immutable geometry-bound ``E.T K E``/``E.T V`` assembly evidence."""

    exposure: SmoothHarmonicExposureSnapshot
    source: HarmonicGaussianSourceSnapshot
    fixed_snapshot: FixedHarmonicGalerkinSnapshot
    source_radial_quadrature_order: int
    green_radial_quadrature_order: int
    minimum_basis_singular_value: float
    minimum_relative_basis_singular_value: float
    minimum_raw_eigenvalue: float
    minimum_surface_eigenvalue: float
    raw_condition_number: float
    surface_condition_number: float
    configuration_sha256: str
    provenance_sha256: str
    state_sha256: str
    _raw_single_layer_values: tuple[float, ...]

    def __init__(
        self,
        exposure: SmoothHarmonicExposureSnapshot,
        *,
        source_radial_quadrature_order: int = 128,
        green_radial_quadrature_order: int = 128,
    ) -> None:
        if not isinstance(exposure, SmoothHarmonicExposureSnapshot):
            raise TypeError("exposure must be SmoothHarmonicExposureSnapshot.")
        exposure.validate()
        source_order = _bounded_order(
            source_radial_quadrature_order,
            name="source_radial_quadrature_order",
        )
        green_order = _bounded_order(
            green_radial_quadrature_order,
            name="green_radial_quadrature_order",
        )
        source = build_harmonic_gaussian_source(
            exposure, radial_quadrature_order=source_order
        )
        weighted_basis = source.weighted_basis_operator
        singular_values = np.linalg.svd(weighted_basis, compute_uv=False)
        minimum_basis = float(singular_values[-1])
        relative_basis = minimum_basis / float(singular_values[0])
        if (
            not np.isfinite(relative_basis)
            or relative_basis <= MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE
        ):
            raise ValueError(
                "weighted basis lost full column rank or exceeded its condition "
                "gate; fully buried charts are not silently deleted."
            )

        raw_single_layer = harmonic_single_layer_operator(
            positions_angstrom=exposure.positions_angstrom,
            radii_angstrom=exposure.radii_angstrom,
            lmax=source.physical_charge_lmax,
            radial_quadrature_order=green_order,
        )
        raw_eigenvalues = np.linalg.eigvalsh(raw_single_layer)
        minimum_raw = float(raw_eigenvalues[0])
        raw_condition = float(raw_eigenvalues[-1] / raw_eigenvalues[0])
        if raw_condition > MAXIMUM_REFERENCE_CONDITION_NUMBER:
            raise ValueError(
                "raw harmonic single-layer condition number exceeds the "
                "fail-closed reference bound."
            )
        surface = weighted_basis.T @ raw_single_layer @ weighted_basis
        surface = 0.5 * (surface + surface.T)
        surface_eigenvalues = np.linalg.eigvalsh(surface)
        minimum_surface = float(surface_eigenvalues[0])
        surface_condition = float(surface_eigenvalues[-1] / minimum_surface)
        if (
            minimum_surface <= 1.0e-12
            or not np.isfinite(surface_condition)
            or surface_condition > MAXIMUM_REFERENCE_CONDITION_NUMBER
        ):
            raise ValueError(
                "weighted harmonic surface operator is not an admitted "
                "strictly positive-definite solve."
            )

        fixed = FixedHarmonicGalerkinSnapshot(
            atomic_numbers=exposure.atomic_numbers,
            coefficient_space=exposure.surface_space,
            surface_operator=surface,
            source_operator=source.source_operator,
            cavity_descriptor_sha256=exposure.state_sha256,
            assembly_contract_id=SMOOTH_WEIGHTED_HARMONIC_GALERKIN_CONTRACT_ID,
        )
        configuration = self._configuration_payload(
            exposure=exposure,
            source=source,
            source_order=source_order,
            green_order=green_order,
        )
        provenance = _sha(
            {
                "provider_id": SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID,
                "configuration_sha256": configuration,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        state = self._state_payload(
            configuration_sha256=configuration,
            provenance_sha256=provenance,
            exposure=exposure,
            source=source,
            fixed_snapshot=fixed,
            raw_single_layer=raw_single_layer,
            weighted_basis=weighted_basis,
            surface=surface,
        )

        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "fixed_snapshot", fixed)
        object.__setattr__(self, "source_radial_quadrature_order", source_order)
        object.__setattr__(self, "green_radial_quadrature_order", green_order)
        object.__setattr__(self, "minimum_basis_singular_value", minimum_basis)
        object.__setattr__(
            self, "minimum_relative_basis_singular_value", relative_basis
        )
        object.__setattr__(self, "minimum_raw_eigenvalue", minimum_raw)
        object.__setattr__(self, "minimum_surface_eigenvalue", minimum_surface)
        object.__setattr__(self, "raw_condition_number", raw_condition)
        object.__setattr__(self, "surface_condition_number", surface_condition)
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "provenance_sha256", provenance)
        object.__setattr__(self, "state_sha256", state)
        object.__setattr__(
            self,
            "_raw_single_layer_values",
            tuple(float(value) for value in raw_single_layer.flat),
        )
        self.validate()

    @staticmethod
    def _configuration_payload(
        *,
        exposure: SmoothHarmonicExposureSnapshot,
        source: HarmonicGaussianSourceSnapshot,
        source_order: int,
        green_order: int,
    ) -> str:
        return _sha(
            {
                "contract_id": SMOOTH_WEIGHTED_HARMONIC_GALERKIN_CONTRACT_ID,
                "provider_id": SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID,
                "exposure_configuration_sha256": exposure.configuration_sha256,
                "source_configuration_sha256": source.configuration_sha256,
                "basis_space_sha256": exposure.surface_space.metadata_sha256(),
                "physical_charge_space_sha256": (
                    source.physical_charge_space.metadata_sha256()
                ),
                "single_layer_contract_id": HARMONIC_SINGLE_LAYER_CONTRACT_ID,
                "single_layer_provider_id": HARMONIC_SINGLE_LAYER_PROVIDER_ID,
                "source_radial_quadrature_order": source_order,
                "green_radial_quadrature_order": green_order,
                "minimum_relative_basis_singular_value": (
                    MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE
                ),
                "maximum_reference_condition_number": (
                    MAXIMUM_REFERENCE_CONDITION_NUMBER
                ),
                "stationary_scalar": "-1/2 (S c)^T A^-1 (S c)",
                "assembly": "A=E.T K E; S=E.T V; E rectangular",
                "regularity": (
                    "C1-on-nonsingular-distinct-centre-domain; not-C2-at-"
                    "shell-tangency"
                ),
                "coordinate_derivative": "unavailable",
                "capabilities": "none",
            }
        )

    @staticmethod
    def _state_payload(
        *,
        configuration_sha256: str,
        provenance_sha256: str,
        exposure: SmoothHarmonicExposureSnapshot,
        source: HarmonicGaussianSourceSnapshot,
        fixed_snapshot: FixedHarmonicGalerkinSnapshot,
        raw_single_layer: np.ndarray,
        weighted_basis: np.ndarray,
        surface: np.ndarray,
    ) -> str:
        return _sha(
            {
                "configuration_sha256": configuration_sha256,
                "provenance_sha256": provenance_sha256,
                "exposure_state_sha256": exposure.state_sha256,
                "source_state_sha256": source.state_sha256,
                "fixed_snapshot_state_sha256": fixed_snapshot.state_sha256,
                "raw_single_layer_sha256": _array_sha256(raw_single_layer),
                "weighted_basis_sha256": _array_sha256(weighted_basis),
                "surface_operator_sha256": _array_sha256(surface),
                "source_operator_sha256": _array_sha256(source.source_operator),
            }
        )

    @property
    def basis_space(self) -> PerAtomHarmonicSpace:
        return self.exposure.surface_space

    @property
    def physical_charge_space(self) -> PerAtomHarmonicSpace:
        return self.source.physical_charge_space

    @property
    def weighted_basis_operator(self) -> np.ndarray:
        return self.source.weighted_basis_operator

    @property
    def raw_single_layer_operator(self) -> np.ndarray:
        dimension = self.physical_charge_space.dimension
        return _readonly(
            np.asarray(self._raw_single_layer_values, dtype=float).reshape(
                dimension, dimension
            )
        )

    @property
    def surface_operator(self) -> np.ndarray:
        return self.fixed_snapshot.surface_operator

    @property
    def source_operator(self) -> np.ndarray:
        return self.fixed_snapshot.source_operator

    @property
    def coordinate_derivative_available(self) -> bool:
        return False

    @property
    def tier_v_admitted(self) -> bool:
        return False

    def validate(self) -> None:
        self.exposure.validate()
        self.source.validate()
        self.fixed_snapshot.validate()
        if self.source.exposure.state_sha256 != self.exposure.state_sha256:
            raise RuntimeError("weighted harmonic source/exposure binding drifted.")
        raw = harmonic_single_layer_operator(
            positions_angstrom=self.exposure.positions_angstrom,
            radii_angstrom=self.exposure.radii_angstrom,
            lmax=self.physical_charge_space.lmax,
            radial_quadrature_order=self.green_radial_quadrature_order,
        )
        surface = self.weighted_basis_operator.T @ raw @ self.weighted_basis_operator
        surface = 0.5 * (surface + surface.T)
        expected_configuration = self._configuration_payload(
            exposure=self.exposure,
            source=self.source,
            source_order=self.source_radial_quadrature_order,
            green_order=self.green_radial_quadrature_order,
        )
        expected_provenance = _sha(
            {
                "provider_id": SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID,
                "configuration_sha256": expected_configuration,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        expected_state = self._state_payload(
            configuration_sha256=expected_configuration,
            provenance_sha256=expected_provenance,
            exposure=self.exposure,
            source=self.source,
            fixed_snapshot=self.fixed_snapshot,
            raw_single_layer=raw,
            weighted_basis=self.weighted_basis_operator,
            surface=surface,
        )
        singular_values = np.linalg.svd(self.weighted_basis_operator, compute_uv=False)
        raw_eigenvalues = np.linalg.eigvalsh(raw)
        surface_eigenvalues = np.linalg.eigvalsh(surface)
        expected_values = (
            float(singular_values[-1]),
            float(singular_values[-1] / singular_values[0]),
            float(raw_eigenvalues[0]),
            float(surface_eigenvalues[0]),
            float(raw_eigenvalues[-1] / raw_eigenvalues[0]),
            float(surface_eigenvalues[-1] / surface_eigenvalues[0]),
        )
        actual_values = (
            self.minimum_basis_singular_value,
            self.minimum_relative_basis_singular_value,
            self.minimum_raw_eigenvalue,
            self.minimum_surface_eigenvalue,
            self.raw_condition_number,
            self.surface_condition_number,
        )
        if (
            self.configuration_sha256 != expected_configuration
            or self.provenance_sha256 != expected_provenance
            or self.state_sha256 != expected_state
            or not np.array_equal(raw, self.raw_single_layer_operator)
            or not np.array_equal(surface, self.surface_operator)
            or not np.allclose(
                actual_values, expected_values, atol=1.0e-14, rtol=1.0e-13
            )
        ):
            raise RuntimeError("smooth weighted harmonic Galerkin state drifted.")


def build_smooth_weighted_harmonic_galerkin(
    exposure: SmoothHarmonicExposureSnapshot,
    *,
    source_radial_quadrature_order: int = 128,
    green_radial_quadrature_order: int = 128,
) -> SmoothWeightedHarmonicGalerkinAssembly:
    return SmoothWeightedHarmonicGalerkinAssembly(
        exposure,
        source_radial_quadrature_order=source_radial_quadrature_order,
        green_radial_quadrature_order=green_radial_quadrature_order,
    )


__all__ = [
    "MAXIMUM_REFERENCE_CONDITION_NUMBER",
    "MINIMUM_RELATIVE_BASIS_SINGULAR_VALUE",
    "SMOOTH_WEIGHTED_HARMONIC_GALERKIN_CONTRACT_ID",
    "SMOOTH_WEIGHTED_HARMONIC_GALERKIN_PROVIDER_ID",
    "SmoothWeightedHarmonicGalerkinAssembly",
    "build_smooth_weighted_harmonic_galerkin",
]
