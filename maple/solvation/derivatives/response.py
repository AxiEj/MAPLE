"""Immutable result contract for structured stationary-response Hessians."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any

import numpy as np

from .analytic import (
    _REGISTERED_TORCH_V3_SCALAR_PREFIX,
    _canonical_bytes,
    _digest,
    _freeze_json,
    _immutable_float64_array,
    _nonempty_string,
    _plain_json,
)

RESPONSE_HESSIAN_DERIVATIVE_POLICY = MappingProxyType(
    {
        "policy_id": "maple-structured-stationary-response-hessian-v1",
        "derivative_method": "structured-stationary-response",
        "coordinate_finite_differences": False,
        "hessian_symmetrization": "none",
        "numerical_uncertainty": "unavailable-unless-independently-audited",
    }
)
RESPONSE_HESSIAN_DERIVATIVE_POLICY_SHA256 = hashlib.sha256(
    _canonical_bytes(dict(RESPONSE_HESSIAN_DERIVATIVE_POLICY))
).hexdigest()


@dataclass(frozen=True, slots=True)
class ResponseHessianEvaluation:
    """One hash-bound response evaluation retaining the raw Hessian."""

    hessian_eV_per_A2: np.ndarray
    forces_eV_per_A: np.ndarray
    energy_eV: float
    geometry_sha256: str
    configuration_sha256: str
    scalar_contract_id: str
    provider_id: str
    profile_id: str
    device: str
    component_energies_eV: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    evaluation_sha256: str = ""
    maximum_antisymmetry_eV_per_A2: float = field(init=False)
    derivative_policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        forces = _immutable_float64_array(self.forces_eV_per_A, name="forces_eV_per_A")
        if forces.ndim != 2 or forces.shape[0] == 0 or forces.shape[1] != 3:
            raise ValueError("forces_eV_per_A must have non-empty shape (N,3).")
        hessian = _immutable_float64_array(
            self.hessian_eV_per_A2,
            name="Hessian",
            shape=(forces.size, forces.size),
        )
        energy = float(self.energy_eV)
        if not np.isfinite(energy):
            raise ValueError("energy_eV must be finite.")
        geometry_hash = _digest(self.geometry_sha256, name="geometry_sha256")
        configuration_hash = _digest(
            self.configuration_sha256, name="configuration_sha256"
        )
        scalar_id = _nonempty_string(self.scalar_contract_id, name="scalar_contract_id")
        if scalar_id.startswith(
            _REGISTERED_TORCH_V3_SCALAR_PREFIX
        ) and scalar_id.endswith("-v3"):
            raise ValueError(
                "Torch-v3 autograd results cannot be relabeled as structured response."
            )
        provider_id = _nonempty_string(self.provider_id, name="provider_id")
        profile_id = _nonempty_string(self.profile_id, name="profile_id")
        device = _nonempty_string(self.device, name="device")

        if not isinstance(self.component_energies_eV, Mapping):
            raise TypeError("component_energies_eV must be a mapping.")
        components: dict[str, float] = {}
        for raw_name, raw_value in self.component_energies_eV.items():
            name = _nonempty_string(raw_name, name="component energy name")
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("component energies must be finite.")
            components[name] = value
        if not components:
            raise ValueError("component_energies_eV must not be empty.")
        frozen_components = MappingProxyType(dict(sorted(components.items())))
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping.")
        diagnostics = _freeze_json(self.diagnostics, path="diagnostics")
        antisymmetry = float(np.max(np.abs(hessian - hessian.T)))
        payload = {
            "derivative_policy_sha256": RESPONSE_HESSIAN_DERIVATIVE_POLICY_SHA256,
            "hessian_eV_per_A2": hessian.tolist(),
            "forces_eV_per_A": forces.tolist(),
            "energy_eV": energy,
            "geometry_sha256": geometry_hash,
            "configuration_sha256": configuration_hash,
            "scalar_contract_id": scalar_id,
            "provider_id": provider_id,
            "profile_id": profile_id,
            "device": device,
            "dtype": "float64",
            "component_energies_eV": dict(frozen_components),
            "diagnostics": _plain_json(diagnostics),
            "maximum_antisymmetry_eV_per_A2": antisymmetry,
        }
        expected_hash = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        if self.evaluation_sha256 and self.evaluation_sha256 != expected_hash:
            raise ValueError(
                "evaluation_sha256 does not match structured-response content."
            )

        object.__setattr__(self, "hessian_eV_per_A2", hessian)
        object.__setattr__(self, "forces_eV_per_A", forces)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "geometry_sha256", geometry_hash)
        object.__setattr__(self, "configuration_sha256", configuration_hash)
        object.__setattr__(self, "scalar_contract_id", scalar_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "component_energies_eV", frozen_components)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "maximum_antisymmetry_eV_per_A2", antisymmetry)
        object.__setattr__(
            self,
            "derivative_policy_sha256",
            RESPONSE_HESSIAN_DERIVATIVE_POLICY_SHA256,
        )
        object.__setattr__(self, "evaluation_sha256", expected_hash)

    @property
    def derivative_method(self) -> str:
        return "structured-stationary-response"

    @property
    def dtype(self) -> str:
        return "float64"

    @property
    def numerical_uncertainty_eV_per_A2(self) -> None:
        return None


__all__ = [
    "RESPONSE_HESSIAN_DERIVATIVE_POLICY",
    "RESPONSE_HESSIAN_DERIVATIVE_POLICY_SHA256",
    "ResponseHessianEvaluation",
]
