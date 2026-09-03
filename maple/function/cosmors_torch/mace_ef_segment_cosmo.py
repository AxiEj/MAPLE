"""Stationary MACE-EF coupling to the Torch segment-COSMO scalar."""

from __future__ import annotations

from dataclasses import dataclass, field

from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
    MACEPolarEFConfig,
    MACEPolarEFEnergyModel,
)
from maple.function.calculator.extra_correction.implicit.mace_polar_ef_stationary import (
    MACEPolarEFSCFSettings,
    MACEPolarEFStationaryCoupling,
    MACEPolarEFStationaryResult,
    canonical_sha256,
)

from .segment_cosmo import TorchSegmentCOSMO, TorchSegmentCOSMOFunctional

MACE_POLAR_EF_SEGMENT_COSMO_CONTRACT_ID = (
    "maple.mace-polar-ef-v2.segment-cosmo.stationary-first-order.v1"
)


@dataclass(frozen=True, slots=True)
class MACEPolarEFSegmentCOSMOConfig:
    electronic: MACEPolarEFConfig
    continuum: TorchSegmentCOSMO
    scf: MACEPolarEFSCFSettings = field(default_factory=MACEPolarEFSCFSettings)
    continuum_functional: TorchSegmentCOSMOFunctional = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, MACEPolarEFConfig):
            raise TypeError("electronic must be MACEPolarEFConfig.")
        if not isinstance(self.continuum, TorchSegmentCOSMO):
            raise TypeError("continuum must be TorchSegmentCOSMO.")
        if self.electronic.atomic_numbers != self.continuum.config.atomic_numbers:
            raise ValueError("MACE-EF and segment-COSMO atom orders must match.")
        object.__setattr__(
            self,
            "continuum_functional",
            TorchSegmentCOSMOFunctional(self.continuum),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": MACE_POLAR_EF_SEGMENT_COSMO_CONTRACT_ID,
            "electronic": self.electronic.as_identity(),
            "continuum": dict(self.continuum_functional.as_identity()),
            "scf": self.scf.as_dict(),
            "common_scalar": "E_MACE-EF(R,f)+U_segment-COSMO(R,c)-<c,f>",
            "source_equation": "c=Q*dE_MACE-EF/df",
            "field_equation": "f=Q^T*dU_segment-COSMO/dc",
            "external_executable_invoked": False,
            "open24a_parameterization_equivalence": False,
        }

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


class MACEPolarEFSegmentCOSMOCoupling(MACEPolarEFStationaryCoupling):
    def __init__(
        self,
        config: MACEPolarEFSegmentCOSMOConfig,
        *,
        electronic_model: MACEPolarEFEnergyModel | None = None,
    ) -> None:
        if not isinstance(config, MACEPolarEFSegmentCOSMOConfig):
            raise TypeError("config must be MACEPolarEFSegmentCOSMOConfig.")
        super().__init__(config, electronic_model=electronic_model)

    def surface_from_result(
        self,
        positions_angstrom,
        result: MACEPolarEFStationaryResult,
        *,
        name: str,
    ):
        if not isinstance(result, MACEPolarEFStationaryResult):
            raise TypeError("result must be MACEPolarEFStationaryResult.")
        return self.config.continuum.surface(
            positions_angstrom,
            result.source_raw,
            name=name,
        )


__all__ = [
    "MACE_POLAR_EF_SEGMENT_COSMO_CONTRACT_ID",
    "MACEPolarEFSegmentCOSMOConfig",
    "MACEPolarEFSegmentCOSMOCoupling",
]
