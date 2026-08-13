"""Scalar-first field-energy differentiation on the fixed-charge field chart.

The only model-specific hook in :class:`FieldEnergyFunctional` is one Torch
scalar.  Source values and every first/second derivative exposed here are
generated from that same graph.  Release admission remains a separate registry
decision; a callable derivative does not open Route-2 Tier V.

Torch is imported lazily so the contracts remain usable in dependency-light
CI and by continuum-only tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from maple.solvation.coupling.metrics import PairingMetric
from maple.solvation.coupling.spaces import (
    FieldDualSpace,
    LinearChargeCoordinates,
    ReducedCoordinateContract,
    SourceSpace,
)
from .base import atom_count, model_charge_and_multiplicity


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _sign(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1):
        raise ValueError("conjugacy_sign must be exactly +1 or -1.")
    return value


def _torch():
    return __import__("torch")


@dataclass(frozen=True, slots=True)
class TorchGeometry:
    """One graph-local coordinate tensor plus its immutable model metadata."""

    atoms: object
    positions: Any

    def __post_init__(self) -> None:
        torch = _torch()
        if not torch.is_tensor(self.positions):
            raise TypeError("positions must be a Torch tensor.")
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != 3
            or self.positions.shape[0] < 1
            or not torch.is_floating_point(self.positions)
            or not bool(torch.isfinite(self.positions).all())
        ):
            raise ValueError(
                "positions must be finite floating point with shape (N,3)."
            )
        if len(self.atoms) != self.positions.shape[0]:  # type: ignore[arg-type]
            raise ValueError("atoms and positions must have the same atom count.")


@dataclass(frozen=True, slots=True)
class GaugeReducedDualityMap:
    """Content-addressed ``(Q,T,W,g)`` map for one source/field convention.

    With ``T_plus T = I`` we choose

    ``W = Q^-1 T_plus.T`` and ``g = Q^-1 a``.

    Therefore ``T.T Q W = I`` and ``<c,g>_Q = a.T c = q``.  The chart is
    matrix-free; dense ``T``/``W`` arrays are never formed in operational
    methods.
    """

    duality_map_id: str
    source_space: SourceSpace
    field_space: FieldDualSpace
    pairing_metric: PairingMetric
    coordinate_contract: ReducedCoordinateContract
    conjugacy_sign: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "duality_map_id",
            _text(self.duality_map_id, name="duality_map_id"),
        )
        if not isinstance(self.source_space, SourceSpace):
            raise TypeError("source_space must be SourceSpace.")
        if not isinstance(self.field_space, FieldDualSpace):
            raise TypeError("field_space must be FieldDualSpace.")
        if not isinstance(self.pairing_metric, PairingMetric):
            raise TypeError("pairing_metric must be PairingMetric.")
        if not isinstance(self.coordinate_contract, ReducedCoordinateContract):
            raise TypeError("coordinate_contract must be ReducedCoordinateContract.")
        if self.field_space.source_space != self.source_space:
            raise ValueError("field_space must be dual to source_space.")
        if self.field_space.pairing_metric != self.pairing_metric:
            raise ValueError("field_space and pairing_metric identities differ.")
        if self.coordinate_contract.source_space != self.source_space:
            raise ValueError("coordinate contract and source space identities differ.")
        object.__setattr__(self, "conjugacy_sign", _sign(self.conjugacy_sign))

    def metadata(self) -> dict[str, object]:
        return {
            "duality_map_id": self.duality_map_id,
            "source_space_sha256": self.source_space.metadata_hash(),
            "field_space_sha256": self.field_space.metadata_hash(),
            "pairing_sha256": self.pairing_metric.metadata_hash(),
            "coordinate_contract_id": self.coordinate_contract.contract_id,
            "coordinate_contract": self.coordinate_contract.metadata(),
            "charge_covector": list(self.source_space.effective_charge_weights),
            "gauge_direction": "g=Q^-1 a",
            "reduced_field_section": "W=Q^-1 T_plus.T",
            "conjugacy_sign": self.conjugacy_sign,
        }

    def configuration_sha256(self) -> str:
        return _hash(self.metadata())

    def coordinates(
        self, *, atom_count: int, total_charge: float
    ) -> LinearChargeCoordinates:
        coordinates = self.coordinate_contract.build(
            atom_count=atom_count,
            total_charge=total_charge,
        )
        if not isinstance(coordinates, LinearChargeCoordinates):
            raise TypeError("GaugeReducedDualityMap requires LinearChargeCoordinates.")
        self.coordinate_contract.validate(coordinates)
        return coordinates

    def reduce_field(
        self, field: object, *, atom_count: int, total_charge: float
    ) -> np.ndarray:
        values = self.field_space.validate(field, atom_count=atom_count)
        q_field = self.pairing_metric.field_to_source_dual(values)
        return self.coordinates(
            atom_count=atom_count, total_charge=total_charge
        ).reduce_source_cotangent(q_field)

    def decompose_field(
        self, field: object, *, atom_count: int, total_charge: float
    ) -> tuple[np.ndarray, float]:
        values = self.field_space.validate(field, atom_count=atom_count)
        coordinates = self.coordinates(atom_count=atom_count, total_charge=total_charge)
        q_field = self.pairing_metric.field_to_source_dual(values)
        reduced = coordinates.reduce_source_cotangent(q_field)
        section = coordinates.lift_reduced_cotangent(reduced)
        residual = q_field.reshape(-1) - section.reshape(-1)
        charge = np.tile(
            np.asarray(self.source_space.effective_charge_weights, dtype=float),
            atom_count,
        )
        gauge = float(np.vdot(charge, residual) / np.vdot(charge, charge))
        mismatch = residual - gauge * charge
        tolerance = (
            256.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(q_field)))
        )
        if float(np.linalg.norm(mismatch)) > tolerance:
            raise ValueError(
                "field is outside the registered reduced-plus-gauge chart."
            )
        return reduced, gauge

    def lift_field(
        self,
        reduced_field: object,
        *,
        atom_count: int,
        total_charge: float,
        gauge_potential: float = 0.0,
    ) -> np.ndarray:
        coordinates = self.coordinates(atom_count=atom_count, total_charge=total_charge)
        reduced = np.asarray(reduced_field, dtype=float)
        if reduced.shape != (coordinates.reduced_dimension,) or not np.all(
            np.isfinite(reduced)
        ):
            raise ValueError("reduced_field has the wrong finite dimension.")
        if not np.isfinite(gauge_potential):
            raise ValueError("gauge_potential must be finite.")
        section = coordinates.lift_reduced_cotangent(reduced)
        charge = np.tile(
            np.asarray(self.source_space.effective_charge_weights, dtype=float),
            (atom_count, 1),
        )
        source_dual = section + float(gauge_potential) * charge
        return self.pairing_metric.source_to_field_dual(source_dual)

    def source_from_reduced_covector(
        self,
        reduced_covector: object,
        *,
        atom_count: int,
        total_charge: float,
    ) -> np.ndarray:
        coordinates = self.coordinates(atom_count=atom_count, total_charge=total_charge)
        reduced = np.asarray(reduced_covector, dtype=float)
        if reduced.shape != (coordinates.reduced_dimension,) or not np.all(
            np.isfinite(reduced)
        ):
            raise ValueError("reduced source covector has the wrong finite dimension.")
        signed = self.conjugacy_sign * reduced
        reference_covector = coordinates.reduce_tangent(coordinates.c_ref)
        return coordinates.expand(signed - reference_covector)

    def field_cotangent_from_reduced(
        self,
        reduced_cotangent: object,
        *,
        atom_count: int,
        total_charge: float,
    ) -> np.ndarray:
        coordinates = self.coordinates(atom_count=atom_count, total_charge=total_charge)
        reduced = np.asarray(reduced_cotangent, dtype=float)
        if reduced.shape != (coordinates.reduced_dimension,) or not np.all(
            np.isfinite(reduced)
        ):
            raise ValueError("reduced cotangent has the wrong finite dimension.")
        source_block = coordinates.expand_direction(reduced)
        return self.pairing_metric.source_to_field_dual(source_block)

    def _constants(self, *, atom_count: int, total_charge: float, like: Any):
        torch = _torch()
        coordinates = self.coordinates(atom_count=atom_count, total_charge=total_charge)
        scales = torch.as_tensor(
            np.tile(np.asarray(coordinates.component_scales), atom_count),
            dtype=like.dtype,
            device=like.device,
        )
        charge = torch.as_tensor(
            np.tile(
                np.asarray(self.source_space.effective_charge_weights, dtype=float),
                atom_count,
            ),
            dtype=like.dtype,
            device=like.device,
        )
        scaled_constraint = scales * charge
        direction = scaled_constraint / torch.linalg.vector_norm(scaled_constraint)
        axis = torch.zeros_like(direction)
        axis[0] = 1.0
        reflector = axis - direction
        reflector_norm = torch.linalg.vector_norm(reflector)
        if float(reflector_norm.detach().cpu()) <= 64.0 * np.finfo(float).eps:
            reflector = torch.zeros_like(reflector)
        else:
            reflector = reflector / reflector_norm
        reference = scales * (
            float(total_charge)
            * scaled_constraint
            / torch.dot(scaled_constraint, scaled_constraint)
        )
        return coordinates, scales, charge, reflector, reference

    @staticmethod
    def _reflect(values: Any, reflector: Any):
        torch = _torch()
        if not torch.is_tensor(values) or not torch.is_tensor(reflector):
            raise TypeError("Householder operands must be Torch tensors.")
        dot = torch.sum(reflector * values, dim=-1, keepdim=True)
        return values - 2.0 * reflector * dot

    def _t_apply_torch(self, reduced: Any, *, atom_count: int, total_charge: float):
        torch = _torch()
        coordinates, scales, _, reflector, _ = self._constants(
            atom_count=atom_count, total_charge=total_charge, like=reduced
        )
        if reduced.shape != (coordinates.reduced_dimension,):
            raise ValueError("reduced tensor has the wrong dimension.")
        embedded = torch.cat((torch.zeros_like(reduced[:1]), reduced))
        return (scales * self._reflect(embedded, reflector)).reshape(
            atom_count, self.source_space.component_count
        )

    def _tplus_apply_torch(self, source: Any, *, atom_count: int, total_charge: float):
        coordinates, scales, _, reflector, _ = self._constants(
            atom_count=atom_count, total_charge=total_charge, like=source
        )
        if source.shape != self.source_space.shape(atom_count):
            raise ValueError("source tensor has the wrong shape.")
        reflected = self._reflect(source.reshape(-1) / scales, reflector)
        if reflected.shape[0] != coordinates.source_dimension:
            raise RuntimeError("internal reduced-coordinate dimension mismatch.")
        return reflected[1:]

    def _tplus_transpose_torch(
        self, reduced: Any, *, atom_count: int, total_charge: float
    ):
        torch = _torch()
        coordinates, scales, _, reflector, _ = self._constants(
            atom_count=atom_count, total_charge=total_charge, like=reduced
        )
        if reduced.shape != (coordinates.reduced_dimension,):
            raise ValueError("reduced tensor has the wrong dimension.")
        embedded = torch.cat((torch.zeros_like(reduced[:1]), reduced))
        return (self._reflect(embedded, reflector) / scales).reshape(
            atom_count, self.source_space.component_count
        )

    def _tt_apply_torch(
        self, source_cotangent: Any, *, atom_count: int, total_charge: float
    ):
        coordinates, scales, _, reflector, _ = self._constants(
            atom_count=atom_count,
            total_charge=total_charge,
            like=source_cotangent,
        )
        if source_cotangent.shape != self.source_space.shape(atom_count):
            raise ValueError("source cotangent tensor has the wrong shape.")
        reflected = self._reflect(scales * source_cotangent.reshape(-1), reflector)
        if reflected.shape[0] != coordinates.source_dimension:
            raise RuntimeError("internal cotangent dimension mismatch.")
        return reflected[1:]

    def _q_field_torch(self, field: Any):
        torch = _torch()
        indices = torch.as_tensor(
            self.pairing_metric.field_to_source_indices,
            dtype=torch.long,
            device=field.device,
        )
        return torch.index_select(field, -1, indices)

    def _qt_source_torch(self, source: Any):
        torch = _torch()
        inverse = np.argsort(self.pairing_metric.field_to_source_indices)
        indices = torch.as_tensor(inverse, dtype=torch.long, device=source.device)
        return torch.index_select(source, -1, indices)

    def lift_reduced_field_torch(
        self, reduced: Any, *, atom_count: int, total_charge: float
    ):
        section = self._tplus_transpose_torch(
            reduced, atom_count=atom_count, total_charge=total_charge
        )
        return self._qt_source_torch(section)

    def reduce_field_torch(self, field: Any, *, atom_count: int, total_charge: float):
        return self._tt_apply_torch(
            self._q_field_torch(field),
            atom_count=atom_count,
            total_charge=total_charge,
        )

    def project_affine_source_torch(
        self, source: Any, *, atom_count: int, total_charge: float
    ):
        _, _, _, _, reference = self._constants(
            atom_count=atom_count, total_charge=total_charge, like=source
        )
        reference = reference.reshape(atom_count, self.source_space.component_count)
        tangent = self._tplus_apply_torch(
            source - reference,
            atom_count=atom_count,
            total_charge=total_charge,
        )
        return reference + self._t_apply_torch(
            tangent, atom_count=atom_count, total_charge=total_charge
        )

    def source_from_reduced_covector_torch(
        self, reduced_covector: Any, *, atom_count: int, total_charge: float
    ):
        _, _, _, _, reference = self._constants(
            atom_count=atom_count,
            total_charge=total_charge,
            like=reduced_covector,
        )
        reference = reference.reshape(atom_count, self.source_space.component_count)
        reference_covector = self._tplus_apply_torch(
            reference,
            atom_count=atom_count,
            total_charge=total_charge,
        )
        tangent = self._t_apply_torch(
            self.conjugacy_sign * reduced_covector - reference_covector,
            atom_count=atom_count,
            total_charge=total_charge,
        )
        return reference + tangent


class FieldEnergyFunctional:
    """Base class whose public derivatives are sealed to one Torch scalar."""

    __slots__ = (
        "_duality_map",
        "_field_energy_sealed",
        "_torch_device",
        "_torch_dtype",
    )

    _FINAL_DERIVATIVE_METHODS = frozenset(
        {
            "energy_torch",
            "energy_eV",
            "energy_directional_derivative",
            "source_from_energy",
            "source_jvp",
            "source_vjp",
            "field_hvp",
            "mixed_coordinate_field_vjp",
            "fixed_field_coordinate_gradient",
        }
    )

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        forbidden = sorted(cls._FINAL_DERIVATIVE_METHODS.intersection(cls.__dict__))
        if forbidden:
            raise TypeError(
                "FieldEnergyFunctional derivatives are final and cannot be "
                f"overridden: {', '.join(forbidden)}."
            )
        if "_energy_torch" not in cls.__dict__:
            raise TypeError(
                "A field-energy subclass must implement _energy_torch only."
            )

    def __init__(
        self,
        *,
        duality_map: GaugeReducedDualityMap,
        dtype: object,
        device: object,
    ) -> None:
        if not isinstance(duality_map, GaugeReducedDualityMap):
            raise TypeError("duality_map must be GaugeReducedDualityMap.")
        self._duality_map = duality_map
        self._torch_dtype = dtype
        self._torch_device = device
        self._field_energy_sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_field_energy_sealed", False):
            raise AttributeError(
                "FieldEnergyFunctional is immutable after construction."
            )
        object.__setattr__(self, name, value)

    @property
    def duality_map(self) -> GaugeReducedDualityMap:
        return self._duality_map

    def _energy_torch(self, geometry: TorchGeometry, reduced_field: Any):
        raise NotImplementedError

    @staticmethod
    def _validated_total_charge(atoms: object, declared: object) -> float:
        if isinstance(declared, bool) or not isinstance(
            declared, (int, float, np.integer, np.floating)
        ):
            raise TypeError("total_charge must be a finite numeric scalar.")
        value = float(declared)
        expected = float(model_charge_and_multiplicity(atoms)[0])
        if not np.isfinite(value) or value != expected:
            raise ValueError(
                "total_charge must exactly match the canonical model input charge."
            )
        return expected

    def _geometry(self, atoms: object, *, requires_grad: bool) -> TorchGeometry:
        torch = _torch()
        dtype = self._torch_dtype
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, str(dtype).replace("torch.", ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("functional dtype is not a Torch dtype.")
        getter = getattr(atoms, "get_positions", None)
        if not callable(getter):
            raise TypeError("geometry must expose get_positions().")
        positions = np.asarray(getter(), dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] < 1
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("geometry positions must be finite with shape (N,3).")
        tensor = torch.tensor(
            positions,
            dtype=dtype,
            device=self._torch_device,
            requires_grad=requires_grad,
        )
        return TorchGeometry(atoms, tensor)

    def _reduced_tensor(
        self,
        reduced_field: object,
        *,
        atom_count: int,
        total_charge: float,
        requires_grad: bool,
    ):
        torch = _torch()
        coordinates = self.duality_map.coordinates(
            atom_count=atom_count, total_charge=total_charge
        )
        values = np.asarray(reduced_field, dtype=float)
        if values.shape != (coordinates.reduced_dimension,) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("reduced_field has the wrong finite dimension.")
        dtype = self._torch_dtype
        if not isinstance(dtype, torch.dtype):
            dtype = getattr(torch, str(dtype).replace("torch.", ""), None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("functional dtype is not a Torch dtype.")
        return torch.tensor(
            values,
            dtype=dtype,
            device=self._torch_device,
            requires_grad=requires_grad,
        )

    def energy_torch(self, geometry: TorchGeometry, reduced_field: Any):
        torch = _torch()
        if not isinstance(geometry, TorchGeometry):
            raise TypeError("geometry must be TorchGeometry.")
        if not torch.is_tensor(reduced_field) or reduced_field.ndim != 1:
            raise TypeError("reduced_field must be a one-dimensional Torch tensor.")
        energy = self._energy_torch(geometry, reduced_field)
        if not torch.is_tensor(energy) or energy.numel() != 1:
            raise TypeError("_energy_torch must return one scalar Torch tensor.")
        energy = energy.reshape(())
        if not bool(torch.isfinite(energy)):
            raise ValueError("field-energy scalar is non-finite.")
        return energy

    def energy_eV(
        self,
        atoms: object,
        reduced_field: object,
        *,
        total_charge: float,
        gauge_potential: float = 0.0,
    ) -> float:
        total_charge = self._validated_total_charge(atoms, total_charge)
        geometry = self._geometry(atoms, requires_grad=False)
        reduced = self._reduced_tensor(
            reduced_field,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )
        value = self.energy_torch(geometry, reduced)
        value = value + (
            self.duality_map.conjugacy_sign
            * float(total_charge)
            * float(gauge_potential)
        )
        result = float(value.detach().cpu())
        if not np.isfinite(result):
            raise ValueError("field energy is non-finite.")
        return result

    def energy_directional_derivative(
        self,
        atoms: object,
        reduced_field: object,
        reduced_direction: object,
        *,
        total_charge: float,
    ) -> float:
        """Forward-mode derivative of the same sealed scalar graph."""

        torch = _torch()
        total_charge = self._validated_total_charge(atoms, total_charge)
        geometry = self._geometry(atoms, requires_grad=False)
        reduced = self._reduced_tensor(
            reduced_field,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )
        direction = self._reduced_tensor(
            reduced_direction,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )

        def scalar(values):
            return self.energy_torch(geometry, values)

        _, directional = torch.autograd.functional.jvp(
            scalar,
            (reduced,),
            (direction,),
            create_graph=False,
            strict=False,
        )
        result = float(directional.detach().cpu())
        if not np.isfinite(result):
            raise RuntimeError("energy directional derivative is non-finite.")
        return result

    def _source_graph(
        self,
        atoms: object,
        reduced_field: object,
        *,
        total_charge: float,
        coordinate_grad: bool,
        create_graph: bool,
    ):
        torch = _torch()
        total_charge = self._validated_total_charge(atoms, total_charge)
        geometry = self._geometry(atoms, requires_grad=coordinate_grad)
        reduced = self._reduced_tensor(
            reduced_field,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=True,
        )
        energy = self.energy_torch(geometry, reduced)
        if energy.requires_grad:
            (covector,) = torch.autograd.grad(
                energy,
                (reduced,),
                create_graph=create_graph,
                allow_unused=True,
            )
        else:
            covector = None
        if covector is None:
            covector = torch.zeros_like(reduced)
        source = self.duality_map.source_from_reduced_covector_torch(
            covector,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
        )
        return geometry, reduced, energy, covector, source

    def source_from_energy(
        self, atoms: object, reduced_field: object, *, total_charge: float
    ) -> np.ndarray:
        _, _, _, _, source = self._source_graph(
            atoms,
            reduced_field,
            total_charge=total_charge,
            coordinate_grad=False,
            create_graph=False,
        )
        result = np.asarray(source.detach().cpu(), dtype=float).copy()
        return self.duality_map.source_space.validate(
            result, atom_count=len(atoms), name="energy-gradient source"  # type: ignore[arg-type]
        )

    def field_hvp(
        self,
        atoms: object,
        reduced_field: object,
        reduced_direction: object,
        *,
        total_charge: float,
    ) -> np.ndarray:
        torch = _torch()
        total_charge = self._validated_total_charge(atoms, total_charge)
        reduced = self._reduced_tensor(
            reduced_field,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )
        direction = self._reduced_tensor(
            reduced_direction,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )
        geometry = self._geometry(atoms, requires_grad=False)

        def scalar(values):
            return self.energy_torch(geometry, values)

        _, hvp = torch.autograd.functional.hvp(
            scalar,
            reduced,
            direction,
            create_graph=False,
            strict=False,
        )
        result = np.asarray(hvp.detach().cpu(), dtype=float).copy()
        if result.shape != np.asarray(reduced_field).shape or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("field HVP is non-finite or has the wrong shape.")
        return result

    def source_jvp(
        self,
        atoms: object,
        reduced_field: object,
        reduced_direction: object,
        *,
        total_charge: float,
    ) -> np.ndarray:
        hvp = self.field_hvp(
            atoms,
            reduced_field,
            reduced_direction,
            total_charge=total_charge,
        )
        coordinates = self.duality_map.coordinates(
            atom_count=len(atoms), total_charge=total_charge  # type: ignore[arg-type]
        )
        return coordinates.expand_direction(self.duality_map.conjugacy_sign * hvp)

    def source_vjp(
        self,
        atoms: object,
        reduced_field: object,
        source_cotangent: object,
        *,
        total_charge: float,
    ) -> np.ndarray:
        total_charge = self._validated_total_charge(atoms, total_charge)
        cotangent = self.duality_map.source_space.validate(
            source_cotangent,
            atom_count=atom_count(atoms),
            name="source_cotangent",
        )
        coordinates = self.duality_map.coordinates(
            atom_count=atom_count(atoms), total_charge=total_charge
        )
        reduced_direction = coordinates.reduce_source_cotangent(cotangent)
        result = self.duality_map.conjugacy_sign * self.field_hvp(
            atoms,
            reduced_field,
            reduced_direction,
            total_charge=total_charge,
        )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("source VJP is non-finite.")
        return result

    def mixed_coordinate_field_vjp(
        self,
        atoms: object,
        reduced_field: object,
        source_cotangent: object,
        *,
        total_charge: float,
    ) -> np.ndarray:
        torch = _torch()
        total_charge = self._validated_total_charge(atoms, total_charge)
        geometry, _, _, _, source = self._source_graph(
            atoms,
            reduced_field,
            total_charge=total_charge,
            coordinate_grad=True,
            create_graph=True,
        )
        cotangent = self.duality_map.source_space.validate(
            source_cotangent,
            atom_count=len(atoms),  # type: ignore[arg-type]
            name="source_cotangent",
        )
        cotangent_tensor = torch.as_tensor(
            cotangent, dtype=source.dtype, device=source.device
        )
        contraction = torch.sum(source * cotangent_tensor)
        if contraction.requires_grad:
            (position_cotangent,) = torch.autograd.grad(
                contraction,
                (geometry.positions,),
                create_graph=False,
                allow_unused=True,
            )
        else:
            position_cotangent = None
        if position_cotangent is None:
            position_cotangent = torch.zeros_like(geometry.positions)
        result = np.asarray(position_cotangent.detach().cpu(), dtype=float).copy()
        if result.shape != (len(atoms), 3) or not np.all(np.isfinite(result)):  # type: ignore[arg-type]
            raise RuntimeError("mixed coordinate/field VJP is invalid.")
        return result

    def fixed_field_coordinate_gradient(
        self,
        atoms: object,
        reduced_field: object,
        *,
        total_charge: float,
    ) -> np.ndarray:
        torch = _torch()
        total_charge = self._validated_total_charge(atoms, total_charge)
        geometry = self._geometry(atoms, requires_grad=True)
        reduced = self._reduced_tensor(
            reduced_field,
            atom_count=len(atoms),  # type: ignore[arg-type]
            total_charge=total_charge,
            requires_grad=False,
        )
        energy = self.energy_torch(geometry, reduced)
        if energy.requires_grad:
            (gradient,) = torch.autograd.grad(
                energy,
                (geometry.positions,),
                create_graph=False,
                allow_unused=True,
            )
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(geometry.positions)
        result = np.asarray(gradient.detach().cpu(), dtype=float).copy()
        if result.shape != (len(atoms), 3) or not np.all(np.isfinite(result)):  # type: ignore[arg-type]
            raise RuntimeError("fixed-field coordinate gradient is invalid.")
        return result


__all__ = [
    "FieldEnergyFunctional",
    "GaugeReducedDualityMap",
    "TorchGeometry",
]
