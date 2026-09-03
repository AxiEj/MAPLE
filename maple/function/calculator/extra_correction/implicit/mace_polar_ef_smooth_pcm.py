"""Private MACE-POLAR-EF coupling to the fixed-dimensional smooth PCM scalar.

This module connects two separately identified research components without
registering either one as a public MAPLE provider.  The supplied EF checkpoint
still fails the mandatory electronic passivity gate, so normal evaluation
stops before the smooth PCM fixed-point iteration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .mace_polar_ef import MACEPolarEFConfig
from .mace_polar_ef import MACEPolarEFEnergyModel
from .mace_polar_ef_stationary import (
    MACEPolarEFSCFSettings,
    MACEPolarEFStationaryCoupling,
    MACEPolarEFStationaryResult,
    canonical_sha256,
)
from .torch_smooth_pcm import TorchSmoothPCM

MACE_POLAR_EF_SMOOTH_PCM_CONTRACT_ID = (
    "maple.route2.coupling.mace-polar-ef-v2-torch-smooth-pcm-"
    "stationary-first-order.v1"
)


@dataclass(frozen=True, slots=True)
class _BoundSmoothPCM:
    model: TorchSmoothPCM
    positions_angstrom: np.ndarray
    atom_count: int = field(init=False)
    label: str = field(default="torch-smooth-pcm-v1", init=False)
    identity_tolerance_ev: float = field(default=2.0e-9, init=False)
    field_replay_tolerance: float = field(default=2.0e-9, init=False)
    requested_solver_tolerance: float | None = field(default=None, init=False)
    achieved_solver_residual: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        positions = np.array(
            self.positions_angstrom,
            dtype=np.float64,
            copy=True,
        )
        if positions.shape != (self.model.atom_count, 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("Smooth PCM geometry has an invalid shape or value.")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "atom_count", self.model.atom_count)

    def _source(self, source_raw: np.ndarray) -> np.ndarray:
        source = np.asarray(source_raw, dtype=np.float64)
        if source.shape != (self.atom_count, 4) or not np.all(np.isfinite(source)):
            raise ValueError(
                "Smooth PCM source must be finite with shape "
                f"({self.atom_count}, 4)."
            )
        return np.array(source, dtype=np.float64, copy=True)

    def drive_cartesian(
        self,
        source_raw: np.ndarray,
        *,
        warm_start: bool,
    ) -> np.ndarray:
        del warm_start
        return self.model.drive_cartesian(
            self.positions_angstrom,
            self._source(source_raw),
        )

    def energy_ev(self, source_raw: np.ndarray) -> float:
        return self.model.energy(
            self.positions_angstrom,
            self._source(source_raw),
        )

    def coordinate_gradient_ev_per_angstrom(
        self,
        source_raw: np.ndarray,
    ) -> np.ndarray:
        return self.model.coordinate_gradient(
            self.positions_angstrom,
            self._source(source_raw),
        )

    def runtime_provenance(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                **self.model.execution_provenance(),
                "geometry_binding": "float32-MACE-coordinates-promoted-to-float64",
                "warm_start_applicable": False,
                "requested_solver_tolerance": None,
                "achieved_solver_residual": None,
                "internal_linear_solve_policy": (
                    "singular-value-backward-error-amplification-gated"
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class SmoothPCMFunctional:
    model: TorchSmoothPCM
    atom_count: int = field(init=False)
    label: str = field(default="torch-smooth-pcm-v1", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, TorchSmoothPCM):
            raise TypeError("Smooth PCM functional requires TorchSmoothPCM.")
        object.__setattr__(self, "atom_count", self.model.atom_count)

    def bind_geometry(self, positions_angstrom: np.ndarray) -> _BoundSmoothPCM:
        return _BoundSmoothPCM(self.model, positions_angstrom)

    def as_identity(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "model_id": self.model.model_id,
                "provider_id": self.model.provider_id,
                "scalar_id": self.model.scalar_id,
                "configuration_contract_id": (self.model.configuration_contract_id),
                "configuration_sha256": self.model.configuration_sha256(),
                "atomic_numbers": list(self.model.atomic_numbers),
                "radii_angstrom": list(self.model.radii_angstrom),
                "smooth_partition": bool(self.model.smooth_partition),
                "fixed_dimensions": bool(self.model.fixed_dimensions),
                "finite_dielectric": bool(self.model.finite_dielectric),
                "dtype": "torch.float64",
                "device": "cpu",
                "historical_evidence_transferable": False,
                "publicly_registered": False,
                "release_admitted": False,
            }
        )


@dataclass(frozen=True, slots=True)
class MACEPolarEFSmoothPCMConfig:
    electronic: MACEPolarEFConfig
    continuum: TorchSmoothPCM
    scf: MACEPolarEFSCFSettings = field(default_factory=MACEPolarEFSCFSettings)
    continuum_functional: SmoothPCMFunctional = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, MACEPolarEFConfig):
            raise TypeError("electronic must be MACEPolarEFConfig.")
        if not isinstance(self.continuum, TorchSmoothPCM):
            raise TypeError("continuum must be TorchSmoothPCM.")
        if not isinstance(self.scf, MACEPolarEFSCFSettings):
            raise TypeError("scf must be MACEPolarEFSCFSettings.")
        if self.electronic.atomic_numbers != self.continuum.atomic_numbers:
            raise ValueError(
                "Electronic and smooth PCM atomic-number order must match."
            )
        object.__setattr__(
            self,
            "continuum_functional",
            SmoothPCMFunctional(self.continuum),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": MACE_POLAR_EF_SMOOTH_PCM_CONTRACT_ID,
            "electronic": self.electronic.as_identity(),
            "continuum": dict(self.continuum_functional.as_identity()),
            "scf": self.scf.as_dict(),
            "common_scalar": "E_MACE-EF(R,f)+U_smooth_PCM(R,c)-<c,f>",
            "source_equation": "c=Q*dE_MACE-EF/df",
            "field_equation": "f=Q^T*dU_smooth_PCM/dc",
            "coordinate_identity": "MACE float32 coordinates promoted to float64",
            "coupled_autograd_order": 1,
            "electronic_passivity_required": True,
            "publicly_registered": False,
            "release_admitted": False,
        }

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


class MACEPolarEFSmoothPCMCoupling(MACEPolarEFStationaryCoupling):
    """Stationary smooth-PCM connection guarded by EF passivity."""

    def __init__(
        self,
        config: MACEPolarEFSmoothPCMConfig,
        *,
        electronic_model: MACEPolarEFEnergyModel | None = None,
    ) -> None:
        if not isinstance(config, MACEPolarEFSmoothPCMConfig):
            raise TypeError("config must be MACEPolarEFSmoothPCMConfig.")
        super().__init__(config, electronic_model=electronic_model)


MACEPolarEFSmoothPCMResult = MACEPolarEFStationaryResult


__all__ = [
    "MACE_POLAR_EF_SMOOTH_PCM_CONTRACT_ID",
    "MACEPolarEFSmoothPCMConfig",
    "MACEPolarEFSmoothPCMCoupling",
    "MACEPolarEFSmoothPCMResult",
    "SmoothPCMFunctional",
]
