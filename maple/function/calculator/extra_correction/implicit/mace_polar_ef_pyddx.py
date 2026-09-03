"""Standard-pyddx adapter for the private MACE-POLAR-EF stationary core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from ase.units import Hartree

from .mace_polar_ef import MACEPolarEFConfig
from .mace_polar_ef_stationary import (
    MACEPolarEFPassivityError,
    MACEPolarEFSCFSettings,
    MACEPolarEFStationaryCoupling,
    MACEPolarEFStationaryResult,
    canonical_sha256,
)
from .pyddx_pcm_response import PyDDXPCMReactionFieldLinearMap
from .torch_pyddx import TorchPyDDXPCMConfig

MACE_POLAR_EF_PYDDX_CONTRACT_ID = (
    "maple.route2.coupling.mace-polar-ef-v2-pyddx-ddpcm-" "stationary-first-order.v1"
)


@dataclass(frozen=True, slots=True)
class _BoundPyDDXPCM:
    config: TorchPyDDXPCMConfig
    positions_angstrom: np.ndarray
    atom_count: int = field(init=False)
    label: str = field(default="standard-pyddx-ddpcm", init=False)
    identity_tolerance_ev: float = field(init=False)
    field_replay_tolerance: float = field(default=1.0e-9, init=False)
    requested_solver_tolerance: float | None = field(init=False)
    achieved_solver_residual: float | None = field(default=None, init=False)
    _reaction_field: PyDDXPCMReactionFieldLinearMap = field(init=False, repr=False)

    def __post_init__(self) -> None:
        positions = np.array(self.positions_angstrom, dtype=float, copy=True)
        if positions.shape != (self.config.atom_count, 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("pyddx geometry has an invalid shape or value.")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "atom_count", self.config.atom_count)
        object.__setattr__(
            self,
            "identity_tolerance_ev",
            max(1.0e-9, 100.0 * self.config.solver_tolerance),
        )
        object.__setattr__(
            self,
            "requested_solver_tolerance",
            self.config.solver_tolerance,
        )
        object.__setattr__(
            self,
            "_reaction_field",
            self.config.build(positions),
        )

    def _source(self, source_raw: np.ndarray) -> np.ndarray:
        source = np.asarray(source_raw, dtype=float)
        if source.shape != (self.atom_count, 4) or not np.all(np.isfinite(source)):
            raise ValueError(
                "pyddx source must be finite with shape " f"({self.atom_count}, 4)."
            )
        return np.array(source, dtype=float, copy=True)

    def drive_cartesian(
        self,
        source_raw: np.ndarray,
        *,
        warm_start: bool,
    ) -> np.ndarray:
        source = self._source(source_raw)
        method = "apply_scf" if warm_start else "apply"
        return np.asarray(
            getattr(self._reaction_field, method)(source),
            dtype=float,
        )

    def energy_ev(self, source_raw: np.ndarray) -> float:
        source = self._source(source_raw)
        return float(self._reaction_field.polarization_energy_hartree(source) * Hartree)

    def coordinate_gradient_ev_per_angstrom(
        self,
        source_raw: np.ndarray,
    ) -> np.ndarray:
        source = self._source(source_raw)
        return np.asarray(
            self._reaction_field.polarization_position_gradient_ev_per_angstrom(source),
            dtype=float,
        )

    def runtime_provenance(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self._reaction_field.runtime_provenance))


@dataclass(frozen=True, slots=True)
class PyDDXPCMFunctional:
    config: TorchPyDDXPCMConfig
    atom_count: int = field(init=False)
    label: str = field(default="standard-pyddx-ddpcm", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, TorchPyDDXPCMConfig):
            raise TypeError("pyddx functional requires TorchPyDDXPCMConfig.")
        object.__setattr__(self, "atom_count", self.config.atom_count)

    def bind_geometry(self, positions_angstrom: np.ndarray) -> _BoundPyDDXPCM:
        return _BoundPyDDXPCM(self.config, positions_angstrom)

    def as_identity(self) -> Mapping[str, object]:
        return MappingProxyType(self.config.as_dict())


@dataclass(frozen=True, slots=True)
class MACEPolarEFPyDDXConfig:
    electronic: MACEPolarEFConfig
    continuum: TorchPyDDXPCMConfig
    scf: MACEPolarEFSCFSettings = field(default_factory=MACEPolarEFSCFSettings)
    continuum_functional: PyDDXPCMFunctional = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, MACEPolarEFConfig):
            raise TypeError("electronic must be MACEPolarEFConfig.")
        if not isinstance(self.continuum, TorchPyDDXPCMConfig):
            raise TypeError("continuum must be TorchPyDDXPCMConfig.")
        if not isinstance(self.scf, MACEPolarEFSCFSettings):
            raise TypeError("scf must be MACEPolarEFSCFSettings.")
        if self.electronic.atom_count != self.continuum.atom_count:
            raise ValueError("Electronic and continuum atom counts must match.")
        object.__setattr__(
            self,
            "continuum_functional",
            PyDDXPCMFunctional(self.continuum),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": MACE_POLAR_EF_PYDDX_CONTRACT_ID,
            "electronic": self.electronic.as_identity(),
            "continuum": dict(self.continuum_functional.as_identity()),
            "scf": self.scf.as_dict(),
            "common_scalar": "E_MACE-EF(R,f)+G_ddPCM(R,c)-<c,f>",
            "source_equation": "c=Q*dE_MACE-EF/df",
            "field_equation": "f=dG_ddPCM/dc",
            "coordinate_identity": "MACE float32 coordinates promoted to float64",
            "autograd_order": 1,
            "publicly_registered": False,
            "release_admitted": False,
        }

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


class MACEPolarEFPyDDXCoupling(MACEPolarEFStationaryCoupling):
    """Stationary standard-ddPCM connection guarded by EF passivity."""

    def __init__(self, config: MACEPolarEFPyDDXConfig) -> None:
        if not isinstance(config, MACEPolarEFPyDDXConfig):
            raise TypeError("config must be MACEPolarEFPyDDXConfig.")
        super().__init__(config)


MACEPolarEFPyDDXResult = MACEPolarEFStationaryResult


__all__ = [
    "MACE_POLAR_EF_PYDDX_CONTRACT_ID",
    "MACEPolarEFPassivityError",
    "MACEPolarEFPyDDXConfig",
    "MACEPolarEFPyDDXCoupling",
    "MACEPolarEFPyDDXResult",
    "PyDDXPCMFunctional",
]
