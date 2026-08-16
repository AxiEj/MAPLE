"""Same-cavity smooth harmonic atomic areas for non-electrostatic terms.

The harmonic cavity constructs a relaxed exposed-area fraction ``e_i(u)`` by
applying a centred smooth Heaviside to signed sphere overlap and composing the
pair factors.  The same finite coefficient object attenuates the electrostatic
trial/test basis, but bilinearity of that operator does not turn ``e_i`` into a
square-root area amplitude.  The SMD-like geometric area is therefore

``A_i = a_i**2 * integral e_i(u) dOmega``.

For an orthonormal real-harmonic coefficient vector ``c_i``, the integral is
exactly ``sqrt(4*pi) * c_i[0]``.  No laboratory-fixed surface grid, pointwise
clipping, or active-set change is part of the definition.  A quadratic
``integral e_i**2`` would be a separately named participation measure, not the
geometric area implemented here.

This is a geometry descriptor for a separately versioned CDS scalar.  It does
not claim equivalence to the sharp SMD SASA or admit any public Route-2 energy
or derivative capability by itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import SMOOTH_HARMONIC_CAVITY_PROFILE_ID
from maple.solvation.coupling.state_equation import geometry_sha256

from .harmonic_coefficients import _bounded_lmax, _positive_int
from .harmonic_exposure import SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID
from .harmonic_positive_exposure import (
    POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID,
    POSITIVE_BERNSTEIN_EXPOSURE_PROVIDER_ID,
    POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE,
    POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS,
    POSITIVE_BERNSTEIN_PAIR_DEGREE,
    assemble_positive_bernstein_exposure,
)
from .harmonic_torch_primitives import _assemble_exposure_coefficients, _torch

SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID = (
    "maple.route2.cds-area.smooth-harmonic-exposure-integral.v1"
)
SMOOTH_HARMONIC_EXPOSURE_AREA_PROVIDER_ID = (
    "maple.route2.cds-area.smooth-harmonic-exposure-integral.impl.v1"
)
_AREA_MEASURE = "a_i^2-integral-e_i-domega"
_SQRT_FOUR_PI = float(np.sqrt(4.0 * np.pi))
POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID = (
    "maple.route2.cds-area.positive-bernstein-parent-integral.v1"
)
POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_PROVIDER_ID = (
    "maple.route2.cds-area.positive-bernstein-parent-integral.impl.v1"
)
POSITIVE_BERNSTEIN_HARMONIC_CAVITY_PROFILE_ID = (
    "positive-bernstein-parent-harmonic-cavity-candidate-v1"
)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).resolve().parent
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in (
            "harmonic_cds_area.py",
            "harmonic_coefficients.py",
            "harmonic_exposure.py",
            "harmonic_torch_primitives.py",
        )
    )


def _positive_implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).resolve().parent
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in (
            "harmonic_cds_area.py",
            "harmonic_coefficients.py",
            "harmonic_positive_exposure.py",
            "harmonic_torch_primitives.py",
        )
    )


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _bounded_order(value: object, *, name: str) -> int:
    result = _positive_int(value, name=name)
    if result > 4096:
        raise ValueError(f"{name} exceeds the bounded contract.")
    return result


def _torch_device_matches(actual: Any, configured: object) -> bool:
    torch = _torch()
    actual_device = torch.device(actual)
    configured_device = torch.device(configured)
    return actual_device.type == configured_device.type and (
        configured_device.index is None
        or actual_device.index == configured_device.index
    )


class SmoothHarmonicExposureArea:
    """Immutable differentiable ``integral e`` area descriptor."""

    __slots__ = (
        "_atomic_numbers",
        "_configuration_sha256",
        "_exposure_lmax",
        "_radial_order",
        "_radii_angstrom",
        "_sealed",
        "_torch_device",
        "_torch_dtype",
        "_transition_width_angstrom2",
    )

    provider_id = SMOOTH_HARMONIC_EXPOSURE_AREA_PROVIDER_ID
    contract_id = SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID
    cavity_profile_id = SMOOTH_HARMONIC_CAVITY_PROFILE_ID
    exposure_contract_id = SMOOTH_HARMONIC_EXPOSURE_CONTRACT_ID
    area_measure = _AREA_MEASURE
    laboratory_fixed_surface_grid = False
    pointwise_clipping = False
    active_set_changes = False
    coordinate_derivative_available = True
    capabilities = CapabilityStatus()

    def __init__(
        self,
        *,
        atomic_numbers: tuple[int, ...],
        radii_angstrom: tuple[float, ...],
        transition_width_angstrom2: float,
        exposure_lmax: int,
        radial_quadrature_order: int = 96,
        dtype: object,
        device: object,
    ) -> None:
        numbers = tuple(atomic_numbers)
        if not numbers or any(
            isinstance(number, bool) or not isinstance(number, int) or number < 1
            for number in numbers
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        radii = tuple(
            _positive_float(value, name=f"radii_angstrom[{index}]")
            for index, value in enumerate(radii_angstrom)
        )
        if len(radii) != len(numbers):
            raise ValueError("radii_angstrom must have one value per atom.")
        width = _positive_float(
            transition_width_angstrom2,
            name="transition_width_angstrom2",
        )
        maximum = _bounded_lmax(exposure_lmax)
        order = _bounded_order(
            radial_quadrature_order,
            name="radial_quadrature_order",
        )
        runtime_dtype = str(dtype)
        runtime_device = str(device)
        configuration = _sha(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "cavity_profile_id": self.cavity_profile_id,
                "exposure_contract_id": self.exposure_contract_id,
                "atomic_numbers": numbers,
                "radii_angstrom": radii,
                "transition_width_angstrom2": width,
                "exposure_lmax": maximum,
                "radial_quadrature_order": order,
                "area_measure": self.area_measure,
                "harmonic_basis": "orthonormal-real-l-ascending-m-ascending",
                "derivative_route": "same-exposure-coefficient-torch-graph",
                "laboratory_fixed_surface_grid": False,
                "runtime_dtype": runtime_dtype,
                "runtime_device": runtime_device,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "_atomic_numbers", numbers)
        object.__setattr__(self, "_radii_angstrom", radii)
        object.__setattr__(self, "_transition_width_angstrom2", width)
        object.__setattr__(self, "_exposure_lmax", maximum)
        object.__setattr__(self, "_radial_order", order)
        object.__setattr__(self, "_torch_dtype", runtime_dtype)
        object.__setattr__(self, "_torch_device", runtime_device)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "SmoothHarmonicExposureArea is immutable after construction."
            )
        object.__setattr__(self, name, value)

    @property
    def atomic_numbers(self) -> tuple[int, ...]:
        return self._atomic_numbers

    @property
    def radii_angstrom(self) -> tuple[float, ...]:
        return self._radii_angstrom

    @property
    def transition_width_angstrom2(self) -> float:
        return self._transition_width_angstrom2

    @property
    def exposure_lmax(self) -> int:
        return self._exposure_lmax

    @property
    def radial_quadrature_order(self) -> int:
        return self._radial_order

    @property
    def runtime_dtype(self) -> str:
        return self._torch_dtype

    @property
    def runtime_device(self) -> str:
        return self._torch_device

    def _dtype(self):
        torch = _torch()
        dtype = getattr(torch, self._torch_dtype.replace("torch.", ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("area dtype is not a Torch dtype.")
        return dtype

    def configuration_sha256(self) -> str:
        current = _sha(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "cavity_profile_id": self.cavity_profile_id,
                "exposure_contract_id": self.exposure_contract_id,
                "atomic_numbers": self._atomic_numbers,
                "radii_angstrom": self._radii_angstrom,
                "transition_width_angstrom2": self._transition_width_angstrom2,
                "exposure_lmax": self._exposure_lmax,
                "radial_quadrature_order": self._radial_order,
                "area_measure": self.area_measure,
                "harmonic_basis": "orthonormal-real-l-ascending-m-ascending",
                "derivative_route": "same-exposure-coefficient-torch-graph",
                "laboratory_fixed_surface_grid": False,
                "runtime_dtype": self._torch_dtype,
                "runtime_device": self._torch_device,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("smooth harmonic area configuration drifted.")
        return current

    def _positions_tensor(self, geometry: object, *, requires_grad: bool):
        torch = _torch()
        getter = getattr(geometry, "get_positions", None)
        values = getter() if callable(getter) else geometry
        number_getter = getattr(geometry, "get_atomic_numbers", None)
        if callable(number_getter):
            actual = tuple(int(value) for value in number_getter())
            if actual != self._atomic_numbers:
                raise ValueError("geometry atomic numbers do not match the area model.")
        positions = np.asarray(values, dtype=float)
        expected = (len(self._atomic_numbers), 3)
        if positions.shape != expected or not np.all(np.isfinite(positions)):
            raise ValueError(
                f"geometry positions must be finite with shape {expected}."
            )
        return torch.tensor(
            positions,
            dtype=self._dtype(),
            device=self._torch_device,
            requires_grad=requires_grad,
        )

    def _coefficients_torch(self, positions: Any):
        torch = _torch()
        if (
            not torch.is_tensor(positions)
            or positions.shape != (len(self._atomic_numbers), 3)
            or not torch.is_floating_point(positions)
            or positions.dtype != self._dtype()
            or not _torch_device_matches(positions.device, self._torch_device)
            or not bool(torch.isfinite(positions).all())
        ):
            raise ValueError("positions do not satisfy the harmonic area contract.")
        return _assemble_exposure_coefficients(
            positions,
            radii=self._radii_angstrom,
            transition_width=self._transition_width_angstrom2,
            exposure_lmax=self._exposure_lmax,
            radial_order=self._radial_order,
        )

    def _areas_torch(self, positions: Any):
        coefficients = self._coefficients_torch(positions)
        radii = positions.new_tensor(self._radii_angstrom)
        return radii * radii * _SQRT_FOUR_PI * coefficients[:, 0]

    def _validated_areas_torch(self, positions: Any):
        """Return the same differentiable area tensor after fail-closed checks."""

        torch = _torch()
        result = self._areas_torch(positions)
        expected = (len(self._atomic_numbers),)
        if result.shape != expected or not bool(torch.isfinite(result).all()):
            raise RuntimeError("harmonic atom areas are invalid.")
        radii = positions.new_tensor(self._radii_angstrom)
        full_areas = 4.0 * np.pi * radii * radii
        bound_tolerance = (
            256.0
            * torch.finfo(result.dtype).eps
            * torch.maximum(
                torch.ones_like(full_areas),
                full_areas,
            )
        )
        violates_bounds = torch.logical_or(
            result < -bound_tolerance,
            result > full_areas + bound_tolerance,
        )
        if bool(violates_bounds.any()):
            raise RuntimeError(
                "harmonic atom areas violated the positive-parent projection "
                "bounds; clipping is forbidden."
            )
        return result

    def exposure_coefficients(self, geometry: object) -> np.ndarray:
        """Return detached coefficients for audit/reference comparisons."""

        self.configuration_sha256()
        positions = self._positions_tensor(geometry, requires_grad=False)
        result = np.asarray(self._coefficients_torch(positions).detach().cpu())
        if not np.all(np.isfinite(result)):
            raise RuntimeError("harmonic exposure coefficients are non-finite.")
        return np.array(result, dtype=float, copy=True)

    def atom_areas_angstrom2(self, geometry: object) -> np.ndarray:
        """Evaluate per-atom relaxed geometric areas without clipping."""

        self.configuration_sha256()
        positions = self._positions_tensor(geometry, requires_grad=False)
        result = np.asarray(
            self._validated_areas_torch(positions).detach().cpu(),
            dtype=float,
        )
        return np.array(result, copy=True)

    def position_vjp(
        self,
        geometry: object,
        area_cotangent: object,
    ) -> np.ndarray:
        """Return ``d <area_cotangent, A(R)> / dR`` in cotangent/A."""

        self.configuration_sha256()
        cotangent = np.asarray(area_cotangent, dtype=float)
        expected = (len(self._atomic_numbers),)
        if cotangent.shape != expected or not np.all(np.isfinite(cotangent)):
            raise ValueError(f"area_cotangent must be finite with shape {expected}.")
        positions = self._positions_tensor(geometry, requires_grad=True)
        weights = positions.new_tensor(cotangent)
        scalar = _torch().dot(weights, self._validated_areas_torch(positions))
        if scalar.requires_grad:
            try:
                (gradient,) = _torch().autograd.grad(
                    scalar,
                    (positions,),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "harmonic area gradient graph is disconnected from positions."
                ) from exc
        else:  # pragma: no cover - positions normally own the graph
            gradient = None
        if gradient is None:
            gradient = _torch().zeros_like(positions)
        result = np.asarray(gradient.detach().cpu(), dtype=float)
        if result.shape != (len(self._atomic_numbers), 3) or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("harmonic area position VJP is invalid.")
        return np.array(result, copy=True)

    def validate_same_cavity_as(self, continuum: object) -> str:
        """Fail closed unless a continuum exposes the same cavity descriptor.

        Electrostatic source, dielectric, and surface-charge truncation may be
        separately versioned.  The quantities compared here are exactly those
        that define ``e_i(R,u)``.  The returned digest binds an evidence row to
        the full continuum configuration after this cavity-level equality has
        been proved.
        """

        self.configuration_sha256()
        expected = {
            "cavity_profile_id": self.cavity_profile_id,
            "exposure_contract_id": self.exposure_contract_id,
            "atomic_numbers": self._atomic_numbers,
            "radii_angstrom": self._radii_angstrom,
            "transition_width_angstrom2": self._transition_width_angstrom2,
            "exposure_lmax": self._exposure_lmax,
            "exposure_radial_quadrature_order": self._radial_order,
            "runtime_dtype": self._torch_dtype,
            "runtime_device": self._torch_device,
            "laboratory_fixed_surface_grid": False,
        }
        observed: dict[str, object] = {}
        for name in expected:
            if not hasattr(continuum, name):
                raise TypeError(
                    f"continuum does not expose required cavity field {name!r}."
                )
            observed[name] = getattr(continuum, name)
        for name in ("atomic_numbers", "radii_angstrom"):
            try:
                observed[name] = tuple(observed[name])
            except TypeError as exc:
                raise TypeError(f"continuum cavity field {name!r} is invalid.") from exc
        if observed != expected:
            raise ValueError("continuum and CDS area cavity descriptors differ.")
        configuration = getattr(continuum, "configuration_sha256", None)
        if not callable(configuration):
            raise TypeError("continuum must expose callable configuration_sha256().")
        digest = configuration()
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("continuum configuration_sha256 is invalid.")
        return digest

    def state_identity(self, geometry: object) -> dict[str, object]:
        """Return the exact configuration/geometry binding for evidence rows."""

        return {
            "provider_id": self.provider_id,
            "contract_id": self.contract_id,
            "configuration_sha256": self.configuration_sha256(),
            "geometry_sha256": geometry_sha256(geometry),
            "cavity_profile_id": self.cavity_profile_id,
            "area_measure": self.area_measure,
            "coordinate_derivative_available": True,
            "laboratory_fixed_surface_grid": False,
            "capabilities": "none",
        }


class PositiveBernsteinHarmonicExposureArea(SmoothHarmonicExposureArea):
    """Differentiable area from one positive finite Bernstein parent.

    ``surface_lmax`` is the continuum trial/test band.  The stored harmonic
    moments extend through ``2 * surface_lmax`` solely to assemble the exact
    Galerkin multiplication matrix; they are never reconstructed as a cavity
    mask.
    """

    __slots__ = ("_surface_lmax",)

    provider_id = POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_PROVIDER_ID
    contract_id = POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID
    cavity_profile_id = POSITIVE_BERNSTEIN_HARMONIC_CAVITY_PROFILE_ID
    exposure_contract_id = POSITIVE_BERNSTEIN_EXPOSURE_CONTRACT_ID
    parent_provider_id = POSITIVE_BERNSTEIN_EXPOSURE_PROVIDER_ID
    positive_parent_pair_degree = POSITIVE_BERNSTEIN_PAIR_DEGREE
    positive_parent_maximum_transition_factors = (
        POSITIVE_BERNSTEIN_MAXIMUM_TRANSITION_FACTORS
    )
    positive_parent_maximum_integrand_degree = (
        POSITIVE_BERNSTEIN_MAXIMUM_INTEGRAND_DEGREE
    )
    reconstructed_low_band_used_as_mask = False

    def __init__(
        self,
        *,
        atomic_numbers: tuple[int, ...],
        radii_angstrom: tuple[float, ...],
        transition_width_angstrom2: float,
        surface_lmax: int,
        dtype: object,
        device: object,
    ) -> None:
        numbers = tuple(atomic_numbers)
        if not numbers or any(
            isinstance(number, bool) or not isinstance(number, int) or number < 1
            for number in numbers
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        radii = tuple(
            _positive_float(value, name=f"radii_angstrom[{index}]")
            for index, value in enumerate(radii_angstrom)
        )
        if len(radii) != len(numbers):
            raise ValueError("radii_angstrom must have one value per atom.")
        width = _positive_float(
            transition_width_angstrom2,
            name="transition_width_angstrom2",
        )
        surface_maximum = _bounded_lmax(surface_lmax)
        moment_maximum = 2 * surface_maximum
        _bounded_lmax(moment_maximum)
        runtime_dtype = str(dtype)
        runtime_device = str(device)
        configuration = _sha(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "cavity_profile_id": self.cavity_profile_id,
                "exposure_contract_id": self.exposure_contract_id,
                "parent_provider_id": self.parent_provider_id,
                "atomic_numbers": numbers,
                "radii_angstrom": radii,
                "transition_width_angstrom2": width,
                "surface_lmax": surface_maximum,
                "retained_parent_moment_lmax": moment_maximum,
                "positive_parent_pair_degree": self.positive_parent_pair_degree,
                "positive_parent_maximum_transition_factors": (
                    self.positive_parent_maximum_transition_factors
                ),
                "positive_parent_maximum_integrand_degree": (
                    self.positive_parent_maximum_integrand_degree
                ),
                "area_measure": self.area_measure,
                "harmonic_basis": "orthonormal-real-l-ascending-m-ascending",
                "moment_operator": "W=P_L-M_parent-P_L",
                "derivative_route": "same-positive-parent-torch-graph",
                "reconstructed_low_band_used_as_mask": False,
                "laboratory_fixed_surface_grid": False,
                "runtime_dtype": runtime_dtype,
                "runtime_device": runtime_device,
                "implementation_sha256": _positive_implementation_sha256(),
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "_atomic_numbers", numbers)
        object.__setattr__(self, "_radii_angstrom", radii)
        object.__setattr__(self, "_transition_width_angstrom2", width)
        object.__setattr__(self, "_surface_lmax", surface_maximum)
        object.__setattr__(self, "_exposure_lmax", moment_maximum)
        object.__setattr__(self, "_radial_order", 0)
        object.__setattr__(self, "_torch_dtype", runtime_dtype)
        object.__setattr__(self, "_torch_device", runtime_device)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    @property
    def surface_lmax(self) -> int:
        return self._surface_lmax

    @property
    def retained_parent_moment_lmax(self) -> int:
        return self._exposure_lmax

    @property
    def radial_quadrature_order(self) -> None:
        return None

    def configuration_sha256(self) -> str:
        current = _sha(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "cavity_profile_id": self.cavity_profile_id,
                "exposure_contract_id": self.exposure_contract_id,
                "parent_provider_id": self.parent_provider_id,
                "atomic_numbers": self._atomic_numbers,
                "radii_angstrom": self._radii_angstrom,
                "transition_width_angstrom2": self._transition_width_angstrom2,
                "surface_lmax": self._surface_lmax,
                "retained_parent_moment_lmax": self._exposure_lmax,
                "positive_parent_pair_degree": self.positive_parent_pair_degree,
                "positive_parent_maximum_transition_factors": (
                    self.positive_parent_maximum_transition_factors
                ),
                "positive_parent_maximum_integrand_degree": (
                    self.positive_parent_maximum_integrand_degree
                ),
                "area_measure": self.area_measure,
                "harmonic_basis": "orthonormal-real-l-ascending-m-ascending",
                "moment_operator": "W=P_L-M_parent-P_L",
                "derivative_route": "same-positive-parent-torch-graph",
                "reconstructed_low_band_used_as_mask": False,
                "laboratory_fixed_surface_grid": False,
                "runtime_dtype": self._torch_dtype,
                "runtime_device": self._torch_device,
                "implementation_sha256": _positive_implementation_sha256(),
                "capabilities": "none",
            }
        )
        if current != self._configuration_sha256:
            raise RuntimeError("positive Bernstein area configuration drifted.")
        return current

    def _parent_tensors(self, positions: Any):
        torch = _torch()
        if (
            not torch.is_tensor(positions)
            or positions.shape != (len(self._atomic_numbers), 3)
            or not torch.is_floating_point(positions)
            or positions.dtype != self._dtype()
            or not _torch_device_matches(positions.device, self._torch_device)
            or not bool(torch.isfinite(positions).all())
        ):
            raise ValueError("positions do not satisfy the positive area contract.")
        return assemble_positive_bernstein_exposure(
            positions,
            radii=self._radii_angstrom,
            transition_width=self._transition_width_angstrom2,
            surface_lmax=self._surface_lmax,
        )

    def _coefficients_torch(self, positions: Any):
        return self._parent_tensors(positions).moments

    def multiplication_blocks(self, geometry: object) -> np.ndarray:
        """Return detached ``P_L M_e P_L`` blocks for continuum audits."""

        self.configuration_sha256()
        positions = self._positions_tensor(geometry, requires_grad=False)
        values = self._parent_tensors(positions).multiplication_blocks
        result = np.asarray(values.detach().cpu(), dtype=float)
        expected = (
            len(self._atomic_numbers),
            (self._surface_lmax + 1) ** 2,
            (self._surface_lmax + 1) ** 2,
        )
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise RuntimeError("positive Bernstein multiplication blocks are invalid.")
        return np.array(result, copy=True)

    def parent_diagnostics(self, geometry: object) -> dict[str, object]:
        """Return target-blind structural diagnostics for one geometry."""

        self.configuration_sha256()
        positions = self._positions_tensor(geometry, requires_grad=False)
        tensors = self._parent_tensors(positions)
        return {
            "transition_factor_counts": list(tensors.transition_factor_counts),
            "maximum_transition_factor_count": max(
                tensors.transition_factor_counts, default=0
            ),
            "maximum_integrand_degree": tensors.maximum_integrand_degree,
        }

    def validate_same_cavity_as(self, continuum: object) -> str:
        self.configuration_sha256()
        expected = {
            "cavity_profile_id": self.cavity_profile_id,
            "exposure_contract_id": self.exposure_contract_id,
            "atomic_numbers": self._atomic_numbers,
            "radii_angstrom": self._radii_angstrom,
            "transition_width_angstrom2": self._transition_width_angstrom2,
            "surface_lmax": self._surface_lmax,
            "positive_parent_pair_degree": self.positive_parent_pair_degree,
            "positive_parent_maximum_transition_factors": (
                self.positive_parent_maximum_transition_factors
            ),
            "runtime_dtype": self._torch_dtype,
            "runtime_device": self._torch_device,
            "laboratory_fixed_surface_grid": False,
        }
        observed: dict[str, object] = {}
        for name in expected:
            if not hasattr(continuum, name):
                raise TypeError(
                    f"continuum does not expose required cavity field {name!r}."
                )
            observed[name] = getattr(continuum, name)
        for name in ("atomic_numbers", "radii_angstrom"):
            observed[name] = tuple(observed[name])
        if observed != expected:
            raise ValueError("continuum and positive CDS parent descriptors differ.")
        configuration = getattr(continuum, "configuration_sha256", None)
        if not callable(configuration):
            raise TypeError("continuum must expose callable configuration_sha256().")
        digest = configuration()
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("continuum configuration_sha256 is invalid.")
        return digest

    def state_identity(self, geometry: object) -> dict[str, object]:
        diagnostics = self.parent_diagnostics(geometry)
        return {
            "provider_id": self.provider_id,
            "contract_id": self.contract_id,
            "configuration_sha256": self.configuration_sha256(),
            "geometry_sha256": geometry_sha256(geometry),
            "cavity_profile_id": self.cavity_profile_id,
            "exposure_contract_id": self.exposure_contract_id,
            "area_measure": self.area_measure,
            "coordinate_derivative_available": True,
            "laboratory_fixed_surface_grid": False,
            "reconstructed_low_band_used_as_mask": False,
            **diagnostics,
            "capabilities": "none",
        }


__all__ = [
    "POSITIVE_BERNSTEIN_HARMONIC_CAVITY_PROFILE_ID",
    "POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_CONTRACT_ID",
    "POSITIVE_BERNSTEIN_HARMONIC_EXPOSURE_AREA_PROVIDER_ID",
    "PositiveBernsteinHarmonicExposureArea",
    "SMOOTH_HARMONIC_EXPOSURE_AREA_CONTRACT_ID",
    "SMOOTH_HARMONIC_EXPOSURE_AREA_PROVIDER_ID",
    "SmoothHarmonicExposureArea",
]
