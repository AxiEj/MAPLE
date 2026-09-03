"""Differentiable conductor-limit sibling of the smooth ddPCM scalar."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .torch_smooth_pcm import TorchSmoothPCM
from .torch_smooth_pcm.identity import sha256_payload

TORCH_SMOOTH_COSMO_MODEL_ID = "torch-smooth-cosmo-v1"
TORCH_SMOOTH_COSMO_PROVIDER_ID = (
    "maple.route2.continuum.torch-smooth-cosmo-schwarz.provider.v1"
)
TORCH_SMOOTH_COSMO_SCALAR_ID = (
    "maple.route2.scalar.torch-smooth-cosmo-schwarz-conductor.v1"
)
TORCH_SMOOTH_COSMO_CONFIGURATION_CONTRACT_ID = (
    "maple.route2.continuum.torch-smooth-cosmo-schwarz-config.v1"
)


class TorchSmoothCOSMO(TorchSmoothPCM):
    """Smooth fixed-dimensional COSMO at the binary64 conductor limit.

    The largest finite float makes ``(epsilon+1)/(epsilon-1)`` exactly one,
    so the finite-dielectric operator equals its conductor operator without
    introducing ``inf/inf`` into the differentiable graph.
    """

    model_id = TORCH_SMOOTH_COSMO_MODEL_ID
    provider_id = TORCH_SMOOTH_COSMO_PROVIDER_ID
    scalar_id = TORCH_SMOOTH_COSMO_SCALAR_ID
    configuration_contract_id = TORCH_SMOOTH_COSMO_CONFIGURATION_CONTRACT_ID
    finite_dielectric = False
    conductor_limit = True

    def __init__(self, **kwargs) -> None:
        if "dielectric" in kwargs:
            raise ValueError("TorchSmoothCOSMO owns the conductor dielectric limit.")
        super().__init__(dielectric=float(np.finfo(np.float64).max), **kwargs)

    def _configuration_payload(self) -> dict[str, object]:
        payload = super()._configuration_payload()
        payload.update(
            {
                "continuum_model": "COSMO",
                "conductor_limit": True,
                "finite_dielectric": False,
                "dielectric_representation": "binary64-float-max-exact-jump-limit",
            }
        )
        return payload

    def execution_provenance(self) -> dict[str, object]:
        record = super().execution_provenance()
        record.pop("execution_provenance_sha256", None)
        path = Path(__file__).resolve()
        record.update(
            {
                "continuum_model": "COSMO",
                "conductor_limit": True,
                "torch_smooth_cosmo_source_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
        )
        record["execution_provenance_sha256"] = sha256_payload(record)
        return record


__all__ = [
    "TORCH_SMOOTH_COSMO_CONFIGURATION_CONTRACT_ID",
    "TORCH_SMOOTH_COSMO_MODEL_ID",
    "TORCH_SMOOTH_COSMO_PROVIDER_ID",
    "TORCH_SMOOTH_COSMO_SCALAR_ID",
    "TorchSmoothCOSMO",
]
