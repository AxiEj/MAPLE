"""Strongly convex latent-source scalar reference.

This module is the algebraic oracle for the observable-supervised hybrid head.
It defines no trained model and admits no capability.  The quadratic instance

``A(z) = 1/2 z.T H z - b.T z``

is minimized under one exact charge constraint after coupling an external
potential through ``eta(v)=B.T v``.  The field scalar is zero-anchored at
``v=0``.  Permanent and induced sources, response, and field Hessian all come
from this single constrained scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus


LATENT_SOURCE_FUNCTIONAL_PROVIDER_ID = (
    "maple.route2.models.strongly-convex-latent-source-reference.v1"
)
LATENT_SOURCE_FUNCTIONAL_CONTRACT_ID = (
    "route2-observable-supervised-convex-dual-source-functional-v1"
)


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _torch():
    return __import__("torch")


def quadratic_latent_anchored_energy_torch(
    *,
    hessian,
    linear_term,
    charge_covector,
    total_charge: float,
    field_to_source,
    field,
):
    """Return the differentiable zero-anchored constrained quadratic scalar.

    This function deliberately returns only the scalar. Permanent/induced
    sources and every response derivative must be obtained by differentiating
    this graph rather than by calling a separately implemented source routine.
    """

    torch = _torch()
    tensors = {
        "hessian": hessian,
        "linear_term": linear_term,
        "charge_covector": charge_covector,
        "field_to_source": field_to_source,
        "field": field,
    }
    if not all(torch.is_tensor(value) for value in tensors.values()):
        raise TypeError("quadratic latent inputs must be Torch tensors.")
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be square.")
    latent_dimension = int(hessian.shape[0])
    if (
        linear_term.shape != (latent_dimension,)
        or charge_covector.shape != (latent_dimension,)
        or field_to_source.ndim != 2
        or field_to_source.shape[1] != latent_dimension
        or field.shape != (field_to_source.shape[0],)
    ):
        raise ValueError("quadratic latent Torch shapes are inconsistent.")
    reference = hessian
    if not all(
        value.dtype == reference.dtype and value.device == reference.device
        for value in tensors.values()
    ):
        raise ValueError("quadratic latent Torch tensors must share dtype/device.")
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("quadratic latent Torch tensors must be finite.")
    charge_value = float(total_charge)
    if not math.isfinite(charge_value):
        raise ValueError("total_charge must be finite.")
    symmetric = 0.5 * (hessian + hessian.T)
    zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
    kkt = torch.cat(
        (
            torch.cat((symmetric, charge_covector[:, None]), dim=1),
            torch.cat((charge_covector[None, :], zero.reshape(1, 1)), dim=1),
        ),
        dim=0,
    )

    def minimum(values):
        eta = field_to_source.T @ values
        rhs = torch.cat(
            (
                linear_term + eta,
                torch.as_tensor(
                    [charge_value],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
            )
        )
        latent = torch.linalg.solve(kkt, rhs)[:-1]
        return (
            0.5 * torch.dot(latent, symmetric @ latent)
            - torch.dot(linear_term + eta, latent)
        )

    return minimum(field) - minimum(torch.zeros_like(field))


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class LatentSourceState:
    configuration_sha256: str
    field: np.ndarray
    latent_source: np.ndarray
    physical_source: np.ndarray
    anchored_energy: float
    charge_constraint_residual: float
    kkt_max_absolute_residual: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.configuration_sha256, str)
            or len(self.configuration_sha256) != 64
        ):
            raise ValueError("configuration_sha256 must be a SHA256 digest.")
        field = np.asarray(self.field, dtype=np.float64)
        latent = np.asarray(self.latent_source, dtype=np.float64)
        source = np.asarray(self.physical_source, dtype=np.float64)
        if field.ndim != 1 or latent.ndim != 1 or source.ndim != 1:
            raise ValueError("field, latent_source, and physical_source must be vectors.")
        field = _readonly(field, shape=field.shape, name="field")
        latent = _readonly(latent, shape=latent.shape, name="latent_source")
        source = _readonly(source, shape=source.shape, name="physical_source")
        energy = float(self.anchored_energy)
        constraint = float(self.charge_constraint_residual)
        kkt = float(self.kkt_max_absolute_residual)
        if (
            not math.isfinite(energy)
            or not math.isfinite(constraint)
            or constraint < 0.0
            or not math.isfinite(kkt)
            or kkt < 0.0
        ):
            raise ValueError("latent-source state diagnostics are invalid.")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "latent_source", latent)
        object.__setattr__(self, "physical_source", source)
        object.__setattr__(self, "anchored_energy", energy)
        object.__setattr__(self, "charge_constraint_residual", constraint)
        object.__setattr__(self, "kkt_max_absolute_residual", kkt)


class QuadraticLatentSourceFunctional:
    """Immutable constrained quadratic instance of the convex-dual contract."""

    __slots__ = (
        "_charge_covector",
        "_configuration_sha256",
        "_field_to_source",
        "_hessian",
        "_implementation_sha256",
        "_kkt_matrix",
        "_linear_term",
        "_tangent_inverse",
        "_total_charge",
        "_vacuum_minimum",
        "_vacuum_state",
        "capabilities",
    )

    provider_id = LATENT_SOURCE_FUNCTIONAL_PROVIDER_ID
    contract_id = LATENT_SOURCE_FUNCTIONAL_CONTRACT_ID
    scalar_first = True
    structurally_reciprocal = True
    structurally_passive = True

    def __init__(
        self,
        *,
        hessian: object,
        linear_term: object,
        charge_covector: object,
        total_charge: float,
        field_to_source: object,
    ) -> None:
        matrix = np.asarray(hessian, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("hessian must be a finite square matrix of size >=2.")
        latent_dimension = int(matrix.shape[0])
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues[0] <= 0.0:
            raise ValueError("latent-source hessian must be strictly positive definite.")
        linear = _readonly(
            linear_term,
            shape=(latent_dimension,),
            name="linear_term",
        )
        charge = _readonly(
            charge_covector,
            shape=(latent_dimension,),
            name="charge_covector",
        )
        if float(np.linalg.norm(charge)) <= 0.0:
            raise ValueError("charge_covector must be nonzero.")
        mapping = np.asarray(field_to_source, dtype=np.float64)
        if (
            mapping.ndim != 2
            or mapping.shape[1] != latent_dimension
            or mapping.shape[0] < 1
            or not np.all(np.isfinite(mapping))
        ):
            raise ValueError(
                "field_to_source must have finite shape (field_dimension, latent_dimension)."
            )
        charge_value = float(total_charge)
        if not math.isfinite(charge_value):
            raise ValueError("total_charge must be finite.")
        matrix = np.ascontiguousarray(matrix)
        matrix.setflags(write=False)
        mapping = np.ascontiguousarray(mapping)
        mapping.setflags(write=False)
        kkt = np.zeros((latent_dimension + 1, latent_dimension + 1))
        kkt[:latent_dimension, :latent_dimension] = matrix
        kkt[:latent_dimension, latent_dimension] = charge
        kkt[latent_dimension, :latent_dimension] = charge
        kkt.setflags(write=False)
        inverse = np.linalg.solve(matrix, np.eye(latent_dimension))
        inverse_charge = inverse @ charge
        tangent_inverse = inverse - np.outer(inverse_charge, inverse_charge) / float(
            np.vdot(charge, inverse_charge)
        )
        tangent_inverse = 0.5 * (tangent_inverse + tangent_inverse.T)
        tangent_inverse.setflags(write=False)

        object.__setattr__(self, "_hessian", matrix)
        object.__setattr__(self, "_linear_term", linear)
        object.__setattr__(self, "_charge_covector", charge)
        object.__setattr__(self, "_total_charge", charge_value)
        object.__setattr__(self, "_field_to_source", mapping)
        object.__setattr__(self, "_kkt_matrix", kkt)
        object.__setattr__(self, "_tangent_inverse", tangent_inverse)
        object.__setattr__(self, "capabilities", CapabilityStatus())
        implementation_sha256 = _implementation_sha256()
        object.__setattr__(self, "_implementation_sha256", implementation_sha256)
        configuration = _hash(
            {
                "provider_id": self.provider_id,
                "contract_id": self.contract_id,
                "hessian_sha256": _array_sha256(matrix),
                "linear_term_sha256": _array_sha256(linear),
                "charge_covector_sha256": _array_sha256(charge),
                "total_charge": charge_value,
                "field_to_source_sha256": _array_sha256(mapping),
                "strong_convexity_floor": float(eigenvalues[0]),
                "hessian_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                "implementation_sha256": implementation_sha256,
                "capability_admitted": False,
            }
        )
        object.__setattr__(self, "_configuration_sha256", configuration)
        vacuum_state, multiplier = self._solve_latent(np.zeros(mapping.shape[0]))
        vacuum_minimum = self._minimum_value(
            vacuum_state,
            np.zeros(mapping.shape[0]),
        )
        object.__setattr__(self, "_vacuum_state", vacuum_state)
        object.__setattr__(self, "_vacuum_minimum", vacuum_minimum)
        if abs(float(np.vdot(charge, vacuum_state)) - charge_value) > 2.0e-12:
            raise RuntimeError("vacuum latent source violates the charge constraint.")
        del multiplier

    @property
    def latent_dimension(self) -> int:
        return int(self._hessian.shape[0])

    @property
    def field_dimension(self) -> int:
        return int(self._field_to_source.shape[0])

    @property
    def latent_charge_covector(self) -> np.ndarray:
        return self._charge_covector.copy()

    @property
    def strong_convexity_floor(self) -> float:
        return float(np.linalg.eigvalsh(self._hessian)[0])

    @property
    def hessian_condition_number(self) -> float:
        eigenvalues = np.linalg.eigvalsh(self._hessian)
        return float(eigenvalues[-1] / eigenvalues[0])

    def configuration_sha256(self) -> str:
        if _implementation_sha256() != self._implementation_sha256:
            raise RuntimeError("latent-source functional implementation drifted.")
        return self._configuration_sha256

    def _field(self, values: object) -> np.ndarray:
        return _readonly(values, shape=(self.field_dimension,), name="field")

    def _solve_latent(self, field: object) -> tuple[np.ndarray, float]:
        values = self._field(field)
        rhs = np.concatenate(
            (
                self._linear_term + self._field_to_source.T @ values,
                [self._total_charge],
            )
        )
        solution = np.linalg.solve(self._kkt_matrix, rhs)
        return np.asarray(solution[:-1]), float(solution[-1])

    def _minimum_value(self, latent: np.ndarray, field: np.ndarray) -> float:
        eta = self._field_to_source.T @ field
        return float(
            0.5 * np.vdot(latent, self._hessian @ latent)
            - np.vdot(self._linear_term + eta, latent)
        )

    def solve(self, field: object) -> LatentSourceState:
        values = self._field(field)
        latent, multiplier = self._solve_latent(values)
        source = self._field_to_source @ latent
        minimum = self._minimum_value(latent, values)
        anchored = minimum - self._vacuum_minimum
        stationarity = (
            self._hessian @ latent
            + self._charge_covector * multiplier
            - self._linear_term
            - self._field_to_source.T @ values
        )
        return LatentSourceState(
            configuration_sha256=self.configuration_sha256(),
            field=values,
            latent_source=latent,
            physical_source=source,
            anchored_energy=anchored,
            charge_constraint_residual=abs(
                float(np.vdot(self._charge_covector, latent)) - self._total_charge
            ),
            kkt_max_absolute_residual=float(np.max(np.abs(stationarity))),
        )

    def permanent_source(self) -> np.ndarray:
        result = self._field_to_source @ self._vacuum_state
        result.setflags(write=False)
        return result

    def induced_source(self, field: object) -> np.ndarray:
        result = self.solve(field).physical_source - self.permanent_source()
        result.setflags(write=False)
        return result

    def source_jvp(self, field_direction: object) -> np.ndarray:
        direction = self._field(field_direction)
        return (
            self._field_to_source
            @ self._tangent_inverse
            @ self._field_to_source.T
            @ direction
        )

    def susceptibility(self) -> np.ndarray:
        response = (
            self._field_to_source
            @ self._tangent_inverse
            @ self._field_to_source.T
        )
        response = np.ascontiguousarray(0.5 * (response + response.T))
        response.setflags(write=False)
        return response

    def energy_field_gradient(self, field: object) -> np.ndarray:
        """Return ``dE/dv = -source`` under the registered coupling sign."""

        result = -np.asarray(self.solve(field).physical_source)
        result.setflags(write=False)
        return result

    def energy_field_hvp(self, field_direction: object) -> np.ndarray:
        """Return the negative-semidefinite field Hessian action."""

        result = -self.source_jvp(field_direction)
        result.setflags(write=False)
        return result


__all__ = [
    "LATENT_SOURCE_FUNCTIONAL_CONTRACT_ID",
    "LATENT_SOURCE_FUNCTIONAL_PROVIDER_ID",
    "LatentSourceState",
    "QuadraticLatentSourceFunctional",
    "quadratic_latent_anchored_energy_torch",
]
