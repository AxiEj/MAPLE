"""Topology-preserving MDP/POLAR uniform-susceptibility composition.

MACE-POLAR owns the atomwise near-field source topology.  MACE-MDP contributes
only its supervised *molecular* polarizability.  No atomic MDP response is
inserted into the source.

Let ``U`` embed the three affine-potential coordinates ``g = grad(phi)`` into
the native eight-channel MACE-POLAR field.  The left inverse ``G`` recovers
``g`` by averaging only the duplicated native gradient channels; it never
mixes potential values with gradients of different physical units.  If
``C_P`` is the molecular-dipole Jacobian produced by the zero-field POLAR
source along ``U`` and ``alpha_D`` is the public MDP molecular polarizability,
define

``X = C_P^{-1} (-alpha_D)``

and transform the field passed to POLAR by

``T u = u + U (X - I) G u``.

Consequently ``T(U g) = U X g`` while every field in ``ker(G)`` is unchanged.
The corrected uniform molecular response is exactly
``C_P X = -alpha_D``.  The minus sign follows from the native coordinate being
``grad(phi)`` while the physical electric field is ``E = -grad(phi)``.

This is a zero-training, geometry-bound linear-algebra primitive.  It is not an
energy functional or a capability admission.  A model wrapper must evaluate
MACE-POLAR at ``T u`` and use :meth:`pullback_field_cotangent` for the exact
transpose chain rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from maple.solvation.release.uniform_response_manifold import (
    molecular_dipole_eangstrom,
)

_CONTRACT = "route2-polar-topology-mdp-molecular-susceptibility-v1"
_MAXIMUM_CONDITION_NUMBER = 1.0e8
_CHARGE_RESPONSE_ATOL_E = 1.0e-10
_NATIVE_GY_CHANNELS = (2, 5)
_NATIVE_GZ_CHANNELS = (3, 6)
_NATIVE_GX_CHANNELS = (4, 7)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode() + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    # Always detach from caller-owned storage before enforcing immutability.
    # ``np.ascontiguousarray`` may return the original array and would then
    # silently mark a caller's field read-only.
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _condition_number(values: np.ndarray, *, name: str) -> float:
    singular_values = np.linalg.svd(values, compute_uv=False)
    if singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
        raise ValueError(f"{name} does not have rank three.")
    condition = float(singular_values[0] / singular_values[-1])
    if not math.isfinite(condition) or condition > _MAXIMUM_CONDITION_NUMBER:
        raise ValueError(f"{name} condition number is too large.")
    return condition


def arithmetic_uniform_gradient_left_inverse(atom_count: int) -> np.ndarray:
    """Return the dimensionally valid native-gradient chart ``G``.

    The native layout is ``[V1,V2,gy1,gz1,gx1,gy2,gz2,gx2]``.  A normalized
    spherical convolution preserves an affine gradient, hence both radial
    gradient blocks equal ``g`` on every atom.  Averaging those ``2N`` copies
    is permutation invariant and ignores the constant-potential gauge.
    """

    if isinstance(atom_count, bool) or atom_count < 1:
        raise ValueError("atom_count must be a positive integer.")
    result = np.zeros((3, atom_count * 8), dtype=np.float64)
    weight = 1.0 / (2.0 * atom_count)
    for atom in range(atom_count):
        offset = atom * 8
        for channel in _NATIVE_GX_CHANNELS:
            result[0, offset + channel] = weight
        for channel in _NATIVE_GY_CHANNELS:
            result[1, offset + channel] = weight
        for channel in _NATIVE_GZ_CHANNELS:
            result[2, offset + channel] = weight
    return result


@dataclass(frozen=True, slots=True)
class UniformSusceptibilityFieldTransform:
    """One immutable field chart preserving POLAR's atomwise response span."""

    uniform_native_basis: np.ndarray
    uniform_gradient_left_inverse: np.ndarray
    polar_zero_uniform_source_jacobian: np.ndarray
    polar_uniform_molecular_dipole_jacobian: np.ndarray
    mdp_molecular_polarizability_eangstrom2_per_volt: np.ndarray
    uniform_coordinate_transform: np.ndarray
    transformed_uniform_source_jacobian: np.ndarray
    uniform_basis_condition_number: float
    polar_molecular_response_condition_number: float
    coordinate_transform_condition_number: float
    state_sha256: str = ""

    def __post_init__(self) -> None:
        basis = np.asarray(self.uniform_native_basis, dtype=np.float64)
        if basis.ndim != 3 or basis.shape[1:] != (8, 3) or len(basis) < 1:
            raise ValueError("uniform_native_basis must have shape (N, 8, 3).")
        atom_count = len(basis)
        basis = _readonly(basis, shape=(atom_count, 8, 3), name="uniform_native_basis")
        left = _readonly(
            self.uniform_gradient_left_inverse,
            shape=(3, atom_count * 8),
            name="uniform_gradient_left_inverse",
        )
        polar_source = _readonly(
            self.polar_zero_uniform_source_jacobian,
            shape=(atom_count, 4, 3),
            name="polar_zero_uniform_source_jacobian",
        )
        polar_molecular = _readonly(
            self.polar_uniform_molecular_dipole_jacobian,
            shape=(3, 3),
            name="polar_uniform_molecular_dipole_jacobian",
        )
        alpha = _readonly(
            self.mdp_molecular_polarizability_eangstrom2_per_volt,
            shape=(3, 3),
            name="mdp_molecular_polarizability_eangstrom2_per_volt",
        )
        transform = _readonly(
            self.uniform_coordinate_transform,
            shape=(3, 3),
            name="uniform_coordinate_transform",
        )
        transformed_source = _readonly(
            self.transformed_uniform_source_jacobian,
            shape=(atom_count, 4, 3),
            name="transformed_uniform_source_jacobian",
        )
        conditions = tuple(
            float(value)
            for value in (
                self.uniform_basis_condition_number,
                self.polar_molecular_response_condition_number,
                self.coordinate_transform_condition_number,
            )
        )
        if any(
            not math.isfinite(value) or value < 1.0 or value > _MAXIMUM_CONDITION_NUMBER
            for value in conditions
        ):
            raise ValueError("A susceptibility-transform condition number is invalid.")

        flat_basis = basis.reshape(atom_count * 8, 3)
        if not np.allclose(left @ flat_basis, np.eye(3), rtol=0.0, atol=2.0e-12):
            raise ValueError("uniform_gradient_left_inverse is not a left inverse.")
        if np.max(np.abs(np.sum(polar_source[:, 0, :], axis=0))) > (
            _CHARGE_RESPONSE_ATOL_E
        ):
            raise ValueError("MACE-POLAR uniform response changes total charge.")
        if not np.allclose(
            transformed_source,
            np.einsum("nsc,cd->nsd", polar_source, transform),
            rtol=0.0,
            atol=2.0e-14,
        ):
            raise ValueError("transformed uniform source Jacobian is inconsistent.")
        if not np.allclose(
            polar_molecular @ transform,
            -alpha,
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise ValueError("uniform molecular susceptibility does not close.")

        payload = {
            "contract": _CONTRACT,
            "uniform_native_basis_sha256": _array_sha256(basis),
            "uniform_gradient_left_inverse_sha256": _array_sha256(left),
            "polar_zero_uniform_source_jacobian_sha256": _array_sha256(polar_source),
            "polar_uniform_molecular_dipole_jacobian_sha256": _array_sha256(
                polar_molecular
            ),
            "mdp_molecular_polarizability_sha256": _array_sha256(alpha),
            "uniform_coordinate_transform_sha256": _array_sha256(transform),
            "transformed_uniform_source_jacobian_sha256": _array_sha256(
                transformed_source
            ),
            "uniform_basis_condition_number": conditions[0],
            "polar_molecular_response_condition_number": conditions[1],
            "coordinate_transform_condition_number": conditions[2],
        }
        expected = _canonical_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match the field transform.")
        object.__setattr__(self, "uniform_native_basis", basis)
        object.__setattr__(self, "uniform_gradient_left_inverse", left)
        object.__setattr__(self, "polar_zero_uniform_source_jacobian", polar_source)
        object.__setattr__(
            self, "polar_uniform_molecular_dipole_jacobian", polar_molecular
        )
        object.__setattr__(
            self, "mdp_molecular_polarizability_eangstrom2_per_volt", alpha
        )
        object.__setattr__(self, "uniform_coordinate_transform", transform)
        object.__setattr__(
            self, "transformed_uniform_source_jacobian", transformed_source
        )
        object.__setattr__(self, "uniform_basis_condition_number", conditions[0])
        object.__setattr__(
            self, "polar_molecular_response_condition_number", conditions[1]
        )
        object.__setattr__(self, "coordinate_transform_condition_number", conditions[2])
        object.__setattr__(self, "state_sha256", expected)

    @property
    def atom_count(self) -> int:
        return int(self.uniform_native_basis.shape[0])

    def _field(self, values: object, *, name: str) -> np.ndarray:
        return _readonly(values, shape=(self.atom_count, 8), name=name)

    def _source(self, values: object, *, name: str) -> np.ndarray:
        return _readonly(values, shape=(self.atom_count, 4), name=name)

    def uniform_coordinates(self, field: object) -> np.ndarray:
        """Return the least-squares affine-potential coordinates ``g``."""

        values = self._field(field, name="native field")
        result = self.uniform_gradient_left_inverse @ values.reshape(-1)
        result.setflags(write=False)
        return result

    def transform_field(self, field: object) -> np.ndarray:
        """Map an external native field into the field evaluated by POLAR."""

        values = self._field(field, name="native field")
        coordinates = self.uniform_coordinates(values)
        delta = (self.uniform_coordinate_transform - np.eye(3)) @ coordinates
        result = values + np.einsum("nfc,c->nf", self.uniform_native_basis, delta)
        result.setflags(write=False)
        return result

    def transform_field_direction(self, direction: object) -> np.ndarray:
        """Apply the exact JVP of :meth:`transform_field`."""

        return self.transform_field(direction)

    def pullback_field_cotangent(self, polar_field_cotangent: object) -> np.ndarray:
        """Apply the exact transpose of the geometry-fixed field transform."""

        cotangent = self._field(
            polar_field_cotangent, name="MACE-POLAR field cotangent"
        )
        flat_basis = self.uniform_native_basis.reshape(self.atom_count * 8, 3)
        uniform_cotangent = flat_basis.T @ cotangent.reshape(-1)
        coordinate_correction = (
            self.uniform_coordinate_transform - np.eye(3)
        ).T @ uniform_cotangent
        correction = self.uniform_gradient_left_inverse.T @ coordinate_correction
        result = cotangent + correction.reshape(self.atom_count, 8)
        result.setflags(write=False)
        return result

    def tangent_source_correction(self, field: object) -> np.ndarray:
        """Return the minimal zero-field-tangent source correction.

        This is the additive alternative

        ``J_P U (X-I) G u``

        to evaluating the nonlinear checkpoint at ``T u``.  It has the same
        corrected zero-field uniform susceptibility, but deliberately leaves
        every second and higher field derivative of the original checkpoint
        unchanged.  The method only exposes the correction; it does not choose
        this continuation or claim a physical scalar.
        """

        coordinates = self.uniform_coordinates(field)
        corrected_coordinates = (
            self.uniform_coordinate_transform - np.eye(3)
        ) @ coordinates
        result = np.einsum(
            "nsc,c->ns",
            self.polar_zero_uniform_source_jacobian,
            corrected_coordinates,
        )
        result.setflags(write=False)
        return result

    def tangent_corrected_source(
        self, polar_source_at_field: object, field: object
    ) -> np.ndarray:
        """Add the minimal tangent correction to ``MACE-POLAR(R, u)``."""

        source = self._source(
            polar_source_at_field, name="MACE-POLAR source at native field"
        )
        result = source + self.tangent_source_correction(field)
        result.setflags(write=False)
        return result

    def tangent_source_direction(self, direction: object) -> np.ndarray:
        """Apply the exact field-to-source JVP of the additive correction."""

        return self.tangent_source_correction(direction)

    def pullback_tangent_source_cotangent(self, source_cotangent: object) -> np.ndarray:
        """Apply the exact transpose of the additive correction Jacobian."""

        cotangent = self._source(source_cotangent, name="MACE-POLAR source cotangent")
        uniform_source_cotangent = np.einsum(
            "nsc,ns->c",
            self.polar_zero_uniform_source_jacobian,
            cotangent,
        )
        coordinate_cotangent = (
            self.uniform_coordinate_transform - np.eye(3)
        ).T @ uniform_source_cotangent
        correction = self.uniform_gradient_left_inverse.T @ coordinate_cotangent
        result = correction.reshape(self.atom_count, 8)
        result.setflags(write=False)
        return result


def prepare_uniform_susceptibility_field_transform(
    *,
    positions_angstrom: object,
    uniform_native_basis: object,
    polar_zero_uniform_source_jacobian: object,
    mdp_molecular_polarizability_eangstrom2_per_volt: object,
) -> UniformSusceptibilityFieldTransform:
    """Build one fail-closed topology-preserving susceptibility transform."""

    basis = np.asarray(uniform_native_basis, dtype=np.float64)
    if basis.ndim != 3 or basis.shape[1:] != (8, 3) or len(basis) < 1:
        raise ValueError("uniform_native_basis must have shape (N, 8, 3).")
    atom_count = len(basis)
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    if positions.shape != (atom_count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_angstrom must be finite with shape (N, 3).")
    polar_source = np.asarray(polar_zero_uniform_source_jacobian, dtype=np.float64)
    if polar_source.shape != (atom_count, 4, 3) or not np.all(
        np.isfinite(polar_source)
    ):
        raise ValueError("polar_zero_uniform_source_jacobian must be finite (N, 4, 3).")
    if np.max(np.abs(np.sum(polar_source[:, 0, :], axis=0))) > (
        _CHARGE_RESPONSE_ATOL_E
    ):
        raise ValueError("MACE-POLAR uniform response changes total charge.")

    alpha = np.asarray(
        mdp_molecular_polarizability_eangstrom2_per_volt, dtype=np.float64
    )
    if alpha.shape != (3, 3) or not np.all(np.isfinite(alpha)):
        raise ValueError(
            "mdp_molecular_polarizability_eangstrom2_per_volt must be finite "
            "with shape (3, 3)."
        )
    symmetric_alpha = 0.5 * (alpha + alpha.T)
    scale = max(1.0, float(np.linalg.norm(symmetric_alpha, ord=2)))
    if not np.allclose(alpha, symmetric_alpha, rtol=0.0, atol=1.0e-10 * scale):
        raise ValueError("MACE-MDP molecular polarizability is not symmetric.")
    if float(np.min(np.linalg.eigvalsh(symmetric_alpha))) <= 1.0e-12 * scale:
        raise ValueError("MACE-MDP molecular polarizability is not positive definite.")

    flat_basis = basis.reshape(atom_count * 8, 3)
    basis_condition = _condition_number(flat_basis, name="uniform_native_basis")
    left_inverse = arithmetic_uniform_gradient_left_inverse(atom_count)
    if not np.allclose(left_inverse @ flat_basis, np.eye(3), rtol=0.0, atol=2.0e-14):
        raise ValueError(
            "uniform_native_basis does not match the official native gradient layout."
        )
    polar_molecular = np.column_stack(
        [
            molecular_dipole_eangstrom(positions, polar_source[:, :, axis])
            for axis in range(3)
        ]
    )
    polar_condition = _condition_number(
        polar_molecular, name="MACE-POLAR uniform molecular response"
    )
    transform = np.linalg.solve(polar_molecular, -symmetric_alpha)
    transform_condition = _condition_number(
        transform, name="uniform susceptibility coordinate transform"
    )
    transformed_source = np.einsum("nsc,cd->nsd", polar_source, transform)

    return UniformSusceptibilityFieldTransform(
        uniform_native_basis=basis,
        uniform_gradient_left_inverse=left_inverse,
        polar_zero_uniform_source_jacobian=polar_source,
        polar_uniform_molecular_dipole_jacobian=polar_molecular,
        mdp_molecular_polarizability_eangstrom2_per_volt=symmetric_alpha,
        uniform_coordinate_transform=transform,
        transformed_uniform_source_jacobian=transformed_source,
        uniform_basis_condition_number=basis_condition,
        polar_molecular_response_condition_number=polar_condition,
        coordinate_transform_condition_number=transform_condition,
    )


__all__ = [
    "UniformSusceptibilityFieldTransform",
    "arithmetic_uniform_gradient_left_inverse",
    "prepare_uniform_susceptibility_field_transform",
]
