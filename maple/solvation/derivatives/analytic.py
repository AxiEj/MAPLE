"""Immutable result contract for coordinate-connected analytic Hessians.

This contract is deliberately separate from the numerical Richardson result.
It records what a Torch autograd evaluation actually produced without
inventing displacement steps or numerical error estimates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

ANALYTIC_HESSIAN_DERIVATIVE_POLICY = MappingProxyType(
    {
        "policy_id": "maple-coordinate-connected-torch-autograd-hessian-v1",
        "derivative_method": "torch-autograd",
        "coordinate_finite_differences": False,
        "hessian_symmetrization": "none",
        "numerical_uncertainty": "unavailable-unless-independently-audited",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


ANALYTIC_HESSIAN_DERIVATIVE_POLICY_SHA256 = hashlib.sha256(
    _canonical_bytes(dict(ANALYTIC_HESSIAN_DERIVATIVE_POLICY))
).hexdigest()

_REGISTERED_TORCH_V3_SCALAR_PREFIX = (
    "route2-experimental-pure-macepolar-frozen-point-l1-ddpcm-smd-torch-"
)
_REGISTERED_TORCH_V3_VERIFICATION_EVIDENCE_ID = (
    "route2-pure-macepolar-torch-analytic-v3-contract-tests"
)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA256 digest.")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _immutable_float64_array(
    value: object, *, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match required {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    immutable = np.frombuffer(
        np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape(array.shape)
    immutable.flags.writeable = False
    return immutable


def _freeze_json(value: object, *, path: str) -> object:
    """Normalize JSON-compatible diagnostics into recursively immutable data."""

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings.")
            normalized[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(normalized)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        normalized_float = float(value)
        if not np.isfinite(normalized_float):
            raise ValueError(f"{path} must contain only finite values.")
        return normalized_float
    raise TypeError(f"{path} must contain only JSON-compatible values.")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AnalyticHessianEvaluation:
    """One immutable analytic energy/force/Hessian evaluation.

    ``hessian_eV_per_A2`` is retained exactly as evaluated so raw
    antisymmetry remains observable. Consumers may symmetrize only for the
    eigensolve and must not rewrite this record.
    """

    hessian_eV_per_A2: np.ndarray
    forces_eV_per_A: np.ndarray
    energy_eV: float
    geometry_sha256: str
    configuration_sha256: str
    scalar_contract_id: str
    device: str
    component_energies_eV: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    provider_id: str | None = None
    profile_id: str | None = None
    verification_evidence_id: str | None = None
    evaluation_sha256: str = ""
    maximum_antisymmetry_eV_per_A2: float = field(init=False)
    derivative_policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        forces = _immutable_float64_array(self.forces_eV_per_A, name="forces_eV_per_A")
        if forces.ndim != 2 or forces.shape[1] != 3 or forces.shape[0] == 0:
            raise ValueError("forces_eV_per_A must have non-empty shape (N,3).")
        dimension = forces.size
        hessian = _immutable_float64_array(
            self.hessian_eV_per_A2,
            name="Hessian",
            shape=(dimension, dimension),
        )
        energy = float(self.energy_eV)
        if not np.isfinite(energy):
            raise ValueError("energy_eV must be finite.")
        geometry_sha256 = _digest(self.geometry_sha256, name="geometry_sha256")
        configuration_sha256 = _digest(
            self.configuration_sha256, name="configuration_sha256"
        )
        scalar_contract_id = _nonempty_string(
            self.scalar_contract_id, name="scalar_contract_id"
        )
        device = _nonempty_string(self.device, name="device")
        provider_id = self._optional_identity(self.provider_id, name="provider_id")
        profile_id = self._optional_identity(self.profile_id, name="profile_id")
        verification_evidence_id = self._optional_identity(
            self.verification_evidence_id, name="verification_evidence_id"
        )
        registered_torch_v3 = scalar_contract_id.startswith(
            _REGISTERED_TORCH_V3_SCALAR_PREFIX
        ) and scalar_contract_id.endswith("-v3")
        if registered_torch_v3 and any(
            value is None
            for value in (provider_id, profile_id, verification_evidence_id)
        ):
            raise ValueError(
                "registered Torch v3 analytic results require non-empty "
                "provider_id, profile_id, and verification_evidence_id."
            )
        if (
            registered_torch_v3
            and verification_evidence_id
            != _REGISTERED_TORCH_V3_VERIFICATION_EVIDENCE_ID
        ):
            raise ValueError(
                "registered Torch v3 analytic results require the bound contract "
                "verification evidence ID."
            )

        if not isinstance(self.component_energies_eV, Mapping):
            raise TypeError("component_energies_eV must be a mapping.")
        if not self.component_energies_eV:
            raise ValueError("component_energies_eV must not be empty.")
        components: dict[str, float] = {}
        for name, value in self.component_energies_eV.items():
            component_name = _nonempty_string(name, name="component energy name")
            component_energy = float(value)
            if not np.isfinite(component_energy):
                raise ValueError("component energies must be finite.")
            components[component_name] = component_energy
        frozen_components = MappingProxyType(dict(sorted(components.items())))

        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping.")
        frozen_diagnostics = _freeze_json(self.diagnostics, path="diagnostics")
        antisymmetry = float(np.max(np.abs(hessian - hessian.T)))
        payload = {
            "derivative_policy_sha256": ANALYTIC_HESSIAN_DERIVATIVE_POLICY_SHA256,
            "hessian_eV_per_A2": hessian.tolist(),
            "forces_eV_per_A": forces.tolist(),
            "energy_eV": energy,
            "geometry_sha256": geometry_sha256,
            "configuration_sha256": configuration_sha256,
            "scalar_contract_id": scalar_contract_id,
            "provider_id": provider_id,
            "profile_id": profile_id,
            "verification_evidence_id": verification_evidence_id,
            "device": device,
            "dtype": "float64",
            "component_energies_eV": dict(frozen_components),
            "diagnostics": _plain_json(frozen_diagnostics),
            "maximum_antisymmetry_eV_per_A2": antisymmetry,
        }
        expected_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        if self.evaluation_sha256 and self.evaluation_sha256 != expected_sha256:
            raise ValueError(
                "evaluation_sha256 does not match analytic Hessian content."
            )

        object.__setattr__(self, "hessian_eV_per_A2", hessian)
        object.__setattr__(self, "forces_eV_per_A", forces)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "geometry_sha256", geometry_sha256)
        object.__setattr__(self, "configuration_sha256", configuration_sha256)
        object.__setattr__(self, "scalar_contract_id", scalar_contract_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "verification_evidence_id", verification_evidence_id)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "component_energies_eV", frozen_components)
        object.__setattr__(self, "diagnostics", frozen_diagnostics)
        object.__setattr__(self, "maximum_antisymmetry_eV_per_A2", antisymmetry)
        object.__setattr__(
            self,
            "derivative_policy_sha256",
            ANALYTIC_HESSIAN_DERIVATIVE_POLICY_SHA256,
        )
        object.__setattr__(self, "evaluation_sha256", expected_sha256)

    @property
    def derivative_method(self) -> str:
        return "torch-autograd"

    @staticmethod
    def _optional_identity(value: object, *, name: str) -> str | None:
        if value is None:
            return None
        return _nonempty_string(value, name=name)

    @property
    def dtype(self) -> str:
        return "float64"

    @property
    def numerical_uncertainty_eV_per_A2(self) -> None:
        return None


__all__ = [
    "ANALYTIC_HESSIAN_DERIVATIVE_POLICY",
    "ANALYTIC_HESSIAN_DERIVATIVE_POLICY_SHA256",
    "AnalyticHessianEvaluation",
]
