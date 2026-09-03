"""Fixed-dimensional differentiable segment COSMO surface in PyTorch.

This is the all-Torch solute-side bridge needed by COSMO-RS.  It uses a
fixed atom-centred angular rule, smooth overlap weights, Gaussian surface
charges, and an exact total-screening-charge constraint.  No external quantum
chemistry or COSMO-RS executable is called.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from .surface import BOHR_ANGSTROM, COSMOSurface

HARTREE_EV = 27.211386245988
TORCH_SEGMENT_COSMO_MODEL_ID = "torch-segment-cosmo-swig-v1"
TORCH_SEGMENT_COSMO_PROVIDER_ID = (
    "maple.cosmors.torch-segment-cosmo-smooth-fixed-grid.v1"
)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise ImportError("Torch segment COSMO requires PyTorch.") from exc
    return torch


def _sphere_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an antipodally paired equal-area Fibonacci sphere rule."""

    half_count = (degree + 1) ** 2
    index = np.arange(half_count, dtype=float)
    positive_z = (index + 0.5) / half_count
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * index
    radial = np.sqrt(np.maximum(0.0, 1.0 - positive_z**2))
    half = np.stack(
        (
            radial * np.cos(azimuth),
            radial * np.sin(azimuth),
            positive_z,
        ),
        axis=1,
    )
    directions = np.concatenate((half, -half), axis=0)
    weights = np.full(directions.shape[0], 4.0 * np.pi / directions.shape[0])
    return directions, weights


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TorchSegmentCOSMOConfig:
    atomic_numbers: tuple[int, ...]
    radii_angstrom: tuple[float, ...]
    angular_degree: int = 4
    transition_width_angstrom2: float = 0.08
    gaussian_width_factor: float = 4.90498088169
    maximum_relative_residual: float = 2.0e-10

    def __post_init__(self) -> None:
        numbers = tuple(self.atomic_numbers)
        radii = tuple(float(value) for value in self.radii_angstrom)
        if not numbers or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in numbers
        ):
            raise ValueError("atomic_numbers must be positive integers.")
        if len(radii) != len(numbers) or any(
            not math.isfinite(value) or value <= 0.0 for value in radii
        ):
            raise ValueError("One finite positive COSMO radius is required per atom.")
        if (
            isinstance(self.angular_degree, bool)
            or not isinstance(self.angular_degree, int)
            or not 2 <= self.angular_degree <= 20
        ):
            raise ValueError("angular_degree must be an integer in [2, 20].")
        for name in (
            "transition_width_angstrom2",
            "gaussian_width_factor",
            "maximum_relative_residual",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "radii_angstrom", radii)

    @property
    def points_per_atom(self) -> int:
        return 2 * (self.angular_degree + 1) ** 2

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": TORCH_SEGMENT_COSMO_MODEL_ID,
            "provider_id": TORCH_SEGMENT_COSMO_PROVIDER_ID,
            "atomic_numbers": list(self.atomic_numbers),
            "radii_angstrom": list(self.radii_angstrom),
            "angular_degree": self.angular_degree,
            "points_per_atom": self.points_per_atom,
            "transition_width_angstrom2": self.transition_width_angstrom2,
            "gaussian_width_factor": self.gaussian_width_factor,
            "maximum_relative_residual": self.maximum_relative_residual,
            "surface_topology": "fixed atom-centred antipodal Fibonacci rule",
            "overlap": "smooth logistic product; no segment deletion",
            "kernel": "SWIG Gaussian single-layer S",
            "charge_constraint": "sum(q_surface)=-sum(q_solute)",
            "dtype": "torch.float64",
        }

    @property
    def configuration_sha256(self) -> str:
        return _canonical_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class TorchSegmentCOSMOState:
    surface: COSMOSurface
    surface_potential_hartree_per_e: Any
    single_layer_matrix_bohr_inverse: Any
    lagrange_multiplier_hartree_per_e: Any
    relative_linear_residual: Any


class TorchSegmentCOSMO:
    """Differentiable conductor scalar and explicit screening surface."""

    model_id = TORCH_SEGMENT_COSMO_MODEL_ID
    provider_id = TORCH_SEGMENT_COSMO_PROVIDER_ID
    fixed_dimensions = True
    smooth_partition = True
    conductor_limit = True

    def __init__(self, config: TorchSegmentCOSMOConfig) -> None:
        if not isinstance(config, TorchSegmentCOSMOConfig):
            raise TypeError("config must be TorchSegmentCOSMOConfig.")
        self.config = config
        directions, weights = _sphere_rule(config.angular_degree)
        self._directions = directions
        self._weights = weights
        self.atom_count = len(config.atomic_numbers)

    def _validate_torch(self, positions: Any, source_raw: Any) -> None:
        torch = _torch()
        if not isinstance(positions, torch.Tensor) or not isinstance(
            source_raw, torch.Tensor
        ):
            raise TypeError("positions and source_raw must be Torch tensors.")
        if positions.dtype != torch.float64 or source_raw.dtype != torch.float64:
            raise TypeError("Torch segment COSMO requires torch.float64.")
        if positions.device != source_raw.device:
            raise ValueError("positions and source_raw must share one device.")
        if positions.shape != (self.atom_count, 3):
            raise ValueError("positions must have shape (N, 3).")
        if source_raw.shape != (self.atom_count, 4):
            raise ValueError("source_raw must have shape (N, 4).")
        if not bool(torch.isfinite(positions).all().detach()) or not bool(
            torch.isfinite(source_raw).all().detach()
        ):
            raise ValueError("Torch segment COSMO inputs must be finite.")
        if self.atom_count > 1:
            distances = torch.cdist(positions, positions)
            off_diagonal = ~torch.eye(
                self.atom_count, dtype=torch.bool, device=positions.device
            )
            if bool((distances[off_diagonal] <= 1.0e-8).any().detach()):
                raise ValueError("COSMO atom centres must be distinct.")

    @staticmethod
    def _point_multipole_potential(
        points_bohr: Any, positions_angstrom: Any, source_raw: Any
    ):
        torch = _torch()
        cartesian = source_raw[:, (0, 3, 1, 2)]
        charges = cartesian[:, 0]
        dipoles_bohr = cartesian[:, 1:] / BOHR_ANGSTROM
        centres_bohr = positions_angstrom / BOHR_ANGSTROM
        displacement = points_bohr[:, None, :] - centres_bohr[None, :, :]
        radii = torch.linalg.vector_norm(displacement, dim=-1)
        if bool((radii <= 1.0e-10).any().detach()):
            raise ValueError("A COSMO segment coincides with a source centre.")
        dipole_projection = torch.sum(displacement * dipoles_bohr[None, :, :], dim=-1)
        return torch.sum(
            charges[None, :] / radii + dipole_projection / radii**3,
            dim=1,
        )

    def surface_state_torch(
        self,
        positions_angstrom: Any,
        source_raw: Any,
        *,
        name: str = "mace-ef-solute",
    ) -> TorchSegmentCOSMOState:
        torch = _torch()
        self._validate_torch(positions_angstrom, source_raw)
        device = positions_angstrom.device
        directions = positions_angstrom.new_tensor(self._directions)
        angular_weights = positions_angstrom.new_tensor(self._weights)
        radii_angstrom = positions_angstrom.new_tensor(self.config.radii_angstrom)
        points = (
            positions_angstrom[:, None, :]
            + radii_angstrom[:, None, None] * directions[None, :, :]
        )
        point_count = self.config.points_per_atom
        flat_points = points.reshape(-1, 3)
        parents = (
            torch.arange(self.atom_count, dtype=torch.int64, device=device)[:, None]
            .expand(-1, point_count)
            .reshape(-1)
        )

        displacement = flat_points[:, None, :] - positions_angstrom[None, :, :]
        signed_distance2 = torch.sum(displacement.square(), dim=-1) - (
            radii_angstrom[None, :].square()
        )
        logits = signed_distance2 / self.config.transition_width_angstrom2
        self_mask = (
            parents[:, None]
            == torch.arange(self.atom_count, dtype=torch.int64, device=device)[None, :]
        )
        bounded_logits = 60.0 * torch.tanh(logits / 60.0)
        bounded_logits = torch.where(
            self_mask,
            torch.full_like(bounded_logits, 60.0),
            bounded_logits,
        )
        exposure = torch.exp(torch.nn.functional.logsigmoid(bounded_logits).sum(dim=1))
        repeated_weights = (
            angular_weights[None, :].expand(self.atom_count, -1).reshape(-1)
        )
        parent_radii_angstrom = radii_angstrom[parents]
        areas_angstrom2 = repeated_weights * parent_radii_angstrom.square() * exposure

        points_bohr = flat_points / BOHR_ANGSTROM
        parent_radii_bohr = parent_radii_angstrom / BOHR_ANGSTROM
        charge_exponent = self.config.gaussian_width_factor / (
            parent_radii_bohr * torch.sqrt(repeated_weights)
        )
        xi_i = charge_exponent[:, None]
        xi_j = charge_exponent[None, :]
        xi_ij = xi_i * xi_j / torch.sqrt(xi_i.square() + xi_j.square())
        pair_distances = torch.cdist(points_bohr, points_bohr)
        limiting_kernel = 2.0 * xi_ij / math.sqrt(math.pi)
        safe_distance = torch.where(
            pair_distances > 1.0e-12,
            pair_distances,
            torch.ones_like(pair_distances),
        )
        single_layer = torch.where(
            pair_distances > 1.0e-12,
            torch.erf(xi_ij * pair_distances) / safe_distance,
            limiting_kernel,
        )
        diagonal = charge_exponent * math.sqrt(2.0 / math.pi) / exposure
        indices = torch.arange(single_layer.shape[0], device=device)
        single_layer = single_layer.clone()
        single_layer[indices, indices] = diagonal
        single_layer = 0.5 * (single_layer + single_layer.T)
        if not bool(torch.isfinite(single_layer).all().detach()):
            raise FloatingPointError("Torch segment COSMO kernel is non-finite.")

        potential = self._point_multipole_potential(
            points_bohr,
            positions_angstrom,
            source_raw,
        )
        ones = potential.new_ones((potential.numel(), 1))
        kkt = torch.cat(
            (
                torch.cat((single_layer, ones), dim=1),
                torch.cat(
                    (
                        ones.T,
                        potential.new_zeros((1, 1)),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )
        rhs = torch.cat((-potential, -source_raw[:, 0].sum().reshape(1)))
        solution = torch.linalg.solve(kkt, rhs)
        screening_charge = solution[:-1]
        lagrange_multiplier = solution[-1]
        residual = kkt @ solution - rhs
        relative_residual = torch.linalg.vector_norm(residual) / torch.clamp(
            torch.linalg.vector_norm(kkt) * torch.linalg.vector_norm(solution)
            + torch.linalg.vector_norm(rhs),
            min=torch.finfo(torch.float64).tiny,
        )
        if not bool(torch.isfinite(solution).all().detach()):
            raise FloatingPointError("Torch segment COSMO solve is non-finite.")
        if float(relative_residual.detach()) > self.config.maximum_relative_residual:
            raise RuntimeError("Torch segment COSMO failed its linear residual gate.")
        dielectric_energy = 0.5 * torch.dot(potential, screening_charge)

        normals = directions[None, :, :].expand(self.atom_count, -1, -1).reshape(-1, 3)
        origin = positions_angstrom.mean(dim=0)
        cavity_volume = (
            torch.sum(
                areas_angstrom2 * torch.sum((flat_points - origin) * normals, dim=1)
            )
            / 3.0
        )
        if float(cavity_volume.detach()) <= 0.0:
            raise RuntimeError("Torch segment COSMO cavity volume is nonpositive.")
        surface = COSMOSurface(
            name=name,
            atomic_numbers=self.config.atomic_numbers,
            atom_positions_angstrom=positions_angstrom,
            segment_positions_angstrom=flat_points,
            segment_areas_angstrom2=areas_angstrom2,
            segment_parent_atom_indices=parents,
            segment_screening_charge_e=screening_charge,
            dielectric_energy_hartree=dielectric_energy,
            dielectric_energy_role="boundary-polarization-only",
            cavity_volume_angstrom3=cavity_volume,
            molecular_charge_e=source_raw[:, 0].sum(),
            source_identity=(
                f"{TORCH_SEGMENT_COSMO_PROVIDER_ID}:"
                f"{self.config.configuration_sha256}"
            ),
            screening_charge_constraint="exact-total-charge",
        )
        return TorchSegmentCOSMOState(
            surface=surface,
            surface_potential_hartree_per_e=potential,
            single_layer_matrix_bohr_inverse=single_layer,
            lagrange_multiplier_hartree_per_e=lagrange_multiplier,
            relative_linear_residual=relative_residual,
        )

    def _energy_ev_torch(self, positions_angstrom: Any, source_raw: Any):
        return (
            self.surface_state_torch(
                positions_angstrom, source_raw
            ).surface.dielectric_energy_hartree
            * HARTREE_EV
        )

    def _numpy_tensors(
        self,
        geometry: object,
        source: object,
        *,
        grad_geometry: bool,
        grad_source: bool,
    ):
        torch = _torch()
        geometry_array = np.asarray(geometry)
        source_array = np.asarray(source)
        if geometry_array.dtype != np.float64 or source_array.dtype != np.float64:
            raise TypeError("Torch segment COSMO NumPy inputs must use float64.")
        if geometry_array.shape != (self.atom_count, 3) or source_array.shape != (
            self.atom_count,
            4,
        ):
            raise ValueError("Torch segment COSMO NumPy input shapes are invalid.")
        positions = torch.tensor(
            geometry_array,
            dtype=torch.float64,
            requires_grad=grad_geometry,
        )
        source_raw = torch.tensor(
            source_array,
            dtype=torch.float64,
            requires_grad=grad_source,
        )
        return positions, source_raw

    def energy(self, geometry: object, source: object) -> float:
        positions, source_raw = self._numpy_tensors(
            geometry, source, grad_geometry=False, grad_source=False
        )
        return float(self._energy_ev_torch(positions, source_raw).detach())

    def drive_cartesian(self, geometry: object, source: object) -> np.ndarray:
        torch = _torch()
        positions, source_raw = self._numpy_tensors(
            geometry, source, grad_geometry=False, grad_source=True
        )
        raw_gradient = torch.autograd.grad(
            self._energy_ev_torch(positions, source_raw), source_raw
        )[0]
        return raw_gradient[:, (0, 3, 1, 2)].detach().numpy().copy()

    def coordinate_gradient(self, geometry: object, source: object) -> np.ndarray:
        torch = _torch()
        positions, source_raw = self._numpy_tensors(
            geometry, source, grad_geometry=True, grad_source=False
        )
        gradient = torch.autograd.grad(
            self._energy_ev_torch(positions, source_raw), positions
        )[0]
        return gradient.detach().numpy().copy()

    def surface(self, geometry: object, source: object, *, name: str) -> COSMOSurface:
        positions, source_raw = self._numpy_tensors(
            geometry, source, grad_geometry=False, grad_source=False
        )
        state = self.surface_state_torch(positions, source_raw, name=name)
        value = state.surface
        return COSMOSurface(
            name=value.name,
            atomic_numbers=value.atomic_numbers,
            atom_positions_angstrom=value.atom_positions_angstrom.detach(),
            segment_positions_angstrom=value.segment_positions_angstrom.detach(),
            segment_areas_angstrom2=value.segment_areas_angstrom2.detach(),
            segment_parent_atom_indices=value.segment_parent_atom_indices.detach(),
            segment_screening_charge_e=value.segment_screening_charge_e.detach(),
            dielectric_energy_hartree=value.dielectric_energy_hartree.detach(),
            dielectric_energy_role=value.dielectric_energy_role,
            cavity_volume_angstrom3=value.cavity_volume_angstrom3.detach(),
            molecular_charge_e=value.molecular_charge_e.detach(),
            source_identity=value.source_identity,
            screening_charge_constraint=value.screening_charge_constraint,
        )

    def execution_provenance(self) -> Mapping[str, object]:
        torch = _torch()
        return MappingProxyType(
            {
                **self.config.as_dict(),
                "configuration_sha256": self.config.configuration_sha256,
                "torch_version": str(torch.__version__),
                "external_executable_invoked": False,
                "parameterization_scope": "solute-side-conductor-surface",
                "open24a_parameterization_equivalence": False,
            }
        )


@dataclass(frozen=True, slots=True)
class _BoundTorchSegmentCOSMO:
    model: TorchSegmentCOSMO
    positions_angstrom: np.ndarray
    atom_count: int = field(init=False)
    label: str = field(default=TORCH_SEGMENT_COSMO_MODEL_ID, init=False)
    identity_tolerance_ev: float = field(default=5.0e-8, init=False)
    field_replay_tolerance: float = field(default=5.0e-8, init=False)
    requested_solver_tolerance: float | None = field(default=None, init=False)
    achieved_solver_residual: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        positions = np.array(self.positions_angstrom, dtype=np.float64, copy=True)
        if positions.shape != (self.model.atom_count, 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("Torch segment COSMO geometry is invalid.")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "atom_count", self.model.atom_count)

    def drive_cartesian(
        self, source_raw: np.ndarray, *, warm_start: bool
    ) -> np.ndarray:
        del warm_start
        return self.model.drive_cartesian(self.positions_angstrom, source_raw)

    def energy_ev(self, source_raw: np.ndarray) -> float:
        return self.model.energy(self.positions_angstrom, source_raw)

    def coordinate_gradient_ev_per_angstrom(self, source_raw: np.ndarray) -> np.ndarray:
        return self.model.coordinate_gradient(self.positions_angstrom, source_raw)

    def runtime_provenance(self) -> Mapping[str, object]:
        return self.model.execution_provenance()


@dataclass(frozen=True, slots=True)
class TorchSegmentCOSMOFunctional:
    model: TorchSegmentCOSMO
    atom_count: int = field(init=False)
    label: str = field(default=TORCH_SEGMENT_COSMO_MODEL_ID, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, TorchSegmentCOSMO):
            raise TypeError("model must be TorchSegmentCOSMO.")
        object.__setattr__(self, "atom_count", self.model.atom_count)

    def bind_geometry(self, positions_angstrom: np.ndarray) -> _BoundTorchSegmentCOSMO:
        return _BoundTorchSegmentCOSMO(self.model, positions_angstrom)

    def as_identity(self) -> Mapping[str, object]:
        return self.model.execution_provenance()


__all__ = [
    "HARTREE_EV",
    "TORCH_SEGMENT_COSMO_MODEL_ID",
    "TORCH_SEGMENT_COSMO_PROVIDER_ID",
    "TorchSegmentCOSMO",
    "TorchSegmentCOSMOConfig",
    "TorchSegmentCOSMOFunctional",
    "TorchSegmentCOSMOState",
]
