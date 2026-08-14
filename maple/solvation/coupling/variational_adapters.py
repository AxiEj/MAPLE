"""Derivative-generated adapters for scalar-first Route 2 state equations.

These adapters do not infer a functional from response methods.  They expose
only derivatives sealed to :class:`FieldEnergyFunctional` and
:class:`ContinuumEnergyFunctional` scalar graphs, while preserving immutable
provider/configuration/provenance bindings required by ``ReducedStateEquation``.
No public scientific capability is admitted here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.continuum.functional import ContinuumEnergyFunctional
from maple.solvation.models.base import atom_count, model_charge_and_multiplicity
from maple.solvation.models.field_energy import FieldEnergyFunctional

from .state_equation import provider_behavior_sha256

SCALAR_FIRST_ELECTRONIC_ADAPTER_ID = (
    "maple.route2.variational.scalar-first-electronic-state-adapter.v1"
)
SCALAR_FIRST_CONTINUUM_ADAPTER_ID = (
    "maple.route2.variational.scalar-first-continuum-state-adapter.v1"
)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _stable_id(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty stable identifier.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _stable_id(value, name=name).lower()
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _configuration(functional: object, *, name: str) -> str:
    method = getattr(functional, "configuration_sha256", None)
    if not callable(method):
        raise TypeError(f"{name}.configuration_sha256 must be callable.")
    return _digest(method(), name=f"{name} configuration")


def _capabilities_are_closed(functional: object, *, name: str) -> None:
    capabilities = getattr(functional, "capabilities", None)
    if not isinstance(capabilities, CapabilityStatus):
        raise TypeError(f"{name}.capabilities must be CapabilityStatus.")
    if capabilities.enabled_tiers:
        raise ValueError(
            f"{name} cannot enter the disabled common-stationarity kernel with "
            "an admitted capability."
        )


class ScalarFirstElectronicResponseAdapter:
    """Expose one sealed field scalar through the response-equation protocol."""

    __slots__ = (
        "_configuration_sha256",
        "_functional",
        "_sealed",
        "coupling_id",
        "field_space",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
        "source_space",
    )

    def __init__(self, functional: FieldEnergyFunctional) -> None:
        if not isinstance(functional, FieldEnergyFunctional):
            raise TypeError("functional must be FieldEnergyFunctional.")
        _capabilities_are_closed(functional, name="electronic functional")
        provider = _stable_id(
            getattr(functional, "provider_id", None),
            name="electronic functional provider_id",
        )
        model_profile = _stable_id(
            getattr(functional, "model_profile_id", None),
            name="electronic functional model_profile_id",
        )
        coupling = _stable_id(
            getattr(functional, "coupling_id", None),
            name="electronic functional coupling_id",
        )
        provenance = _digest(
            getattr(functional, "provenance_sha256", None),
            name="electronic functional provenance_sha256",
        )
        configuration = _configuration(functional, name="electronic functional")
        behavior = provider_behavior_sha256(
            functional,
            (
                "configuration_sha256",
                "_energy_torch",
                "energy_eV",
                "source_from_energy",
                "source_jvp",
                "source_vjp",
                "mixed_coordinate_field_vjp",
                "fixed_field_coordinate_gradient",
            ),
            label="scalar_first_electronic_functional",
        )
        duality = functional.duality_map
        payload = {
            "adapter_id": SCALAR_FIRST_ELECTRONIC_ADAPTER_ID,
            "functional_provider_id": provider,
            "functional_configuration_sha256": configuration,
            "functional_provenance_sha256": provenance,
            "functional_behavior_sha256": behavior,
            "duality_map_sha256": duality.configuration_sha256(),
            "model_profile_id": model_profile,
            "coupling_id": coupling,
            "implementation_sha256": _implementation_sha256(),
            "capabilities": "none",
        }
        adapter_configuration = _sha(payload)
        adapter_provenance = _sha(
            {
                "adapter_id": SCALAR_FIRST_ELECTRONIC_ADAPTER_ID,
                "configuration_sha256": adapter_configuration,
                "functional_provenance_sha256": provenance,
            }
        )
        object.__setattr__(self, "_functional", functional)
        object.__setattr__(self, "provider_id", SCALAR_FIRST_ELECTRONIC_ADAPTER_ID)
        object.__setattr__(self, "model_profile_id", model_profile)
        object.__setattr__(self, "coupling_id", coupling)
        object.__setattr__(self, "provenance_sha256", adapter_provenance)
        object.__setattr__(self, "source_space", duality.source_space)
        object.__setattr__(self, "field_space", duality.field_space)
        object.__setattr__(self, "_configuration_sha256", adapter_configuration)
        object.__setattr__(self, "_sealed", True)
        self.configuration_sha256()

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("scalar-first electronic adapter is immutable.")
        object.__setattr__(self, name, value)

    @property
    def functional(self) -> FieldEnergyFunctional:
        self.configuration_sha256()
        return self._functional

    def _current_configuration_sha256(self) -> str:
        functional = self._functional
        return _sha(
            {
                "adapter_id": SCALAR_FIRST_ELECTRONIC_ADAPTER_ID,
                "functional_provider_id": getattr(functional, "provider_id"),
                "functional_configuration_sha256": _configuration(
                    functional, name="electronic functional"
                ),
                "functional_provenance_sha256": getattr(
                    functional, "provenance_sha256"
                ),
                "functional_behavior_sha256": provider_behavior_sha256(
                    functional,
                    (
                        "configuration_sha256",
                        "_energy_torch",
                        "energy_eV",
                        "source_from_energy",
                        "source_jvp",
                        "source_vjp",
                        "mixed_coordinate_field_vjp",
                        "fixed_field_coordinate_gradient",
                    ),
                    label="scalar_first_electronic_functional",
                ),
                "duality_map_sha256": functional.duality_map.configuration_sha256(),
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        _capabilities_are_closed(self._functional, name="electronic functional")
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("scalar-first electronic configuration drifted.")
        return current

    @staticmethod
    def _charge(geometry: object) -> float:
        return float(model_charge_and_multiplicity(geometry)[0])

    def evaluate_source(self, geometry: object, field: np.ndarray) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        charge = self._charge(geometry)
        reduced, _ = self._functional.duality_map.decompose_field(
            field, atom_count=count, total_charge=charge
        )
        return self._functional.source_from_energy(
            geometry, reduced, total_charge=charge
        )

    def field_jvp(
        self,
        geometry: object,
        field: np.ndarray,
        field_direction: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        charge = self._charge(geometry)
        reduced, _ = self._functional.duality_map.decompose_field(
            field, atom_count=count, total_charge=charge
        )
        direction = self._functional.duality_map.reduce_field(
            field_direction, atom_count=count, total_charge=charge
        )
        return self._functional.source_jvp(
            geometry, reduced, direction, total_charge=charge
        )

    def field_vjp(
        self,
        geometry: object,
        field: np.ndarray,
        source_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        charge = self._charge(geometry)
        reduced, _ = self._functional.duality_map.decompose_field(
            field, atom_count=count, total_charge=charge
        )
        reduced_cotangent = self._functional.source_vjp(
            geometry,
            reduced,
            source_cotangent,
            total_charge=charge,
        )
        return self._functional.duality_map.field_cotangent_from_reduced(
            reduced_cotangent,
            atom_count=count,
            total_charge=charge,
        )

    def coordinate_vjp(
        self,
        geometry: object,
        field: np.ndarray,
        source_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        charge = self._charge(geometry)
        reduced, _ = self._functional.duality_map.decompose_field(
            field, atom_count=count, total_charge=charge
        )
        result = np.asarray(
            self._functional.mixed_coordinate_field_vjp(
                geometry,
                reduced,
                source_cotangent,
                total_charge=charge,
            ),
            dtype=float,
        )
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise ValueError(
                "electronic scalar mixed coordinate VJP must be finite (N,3)."
            )
        return result.reshape(-1).copy()


class ScalarFirstContinuumResponseAdapter:
    """Expose one sealed continuum scalar through the response protocol."""

    __slots__ = (
        "_configuration_sha256",
        "_functional",
        "_sealed",
        "cavity_profile_id",
        "configuration_contract_id",
        "continuum_profile_id",
        "coupling_id",
        "field_space",
        "provenance_sha256",
        "provider_id",
        "scalar_id",
        "source_space",
    )

    fixed_topology = True
    linear_response = True
    reciprocal = True

    def __init__(self, functional: ContinuumEnergyFunctional) -> None:
        if not isinstance(functional, ContinuumEnergyFunctional):
            raise TypeError("functional must be ContinuumEnergyFunctional.")
        _capabilities_are_closed(functional, name="continuum functional")
        for declaration in (
            "scalar_first",
            "fixed_topology",
            "linear_response",
            "reciprocal",
        ):
            if getattr(functional, declaration, False) is not True:
                raise ValueError(
                    "strict fixed-cavity state requires continuum "
                    f"{declaration}=True."
                )
        provider = _stable_id(
            getattr(functional, "provider_id", None),
            name="continuum functional provider_id",
        )
        scalar_id = _stable_id(
            getattr(functional, "scalar_id", None),
            name="continuum functional scalar_id",
        )
        continuum_profile = _stable_id(
            getattr(functional, "continuum_profile_id", None),
            name="continuum functional continuum_profile_id",
        )
        cavity_profile = _stable_id(
            getattr(functional, "cavity_profile_id", None),
            name="continuum functional cavity_profile_id",
        )
        configuration_contract = _stable_id(
            getattr(
                functional,
                "configuration_contract_id",
                UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
            ),
            name="continuum functional configuration_contract_id",
        )
        coupling = _stable_id(
            getattr(functional, "coupling_id", None),
            name="continuum functional coupling_id",
        )
        provenance = _digest(
            getattr(functional, "provenance_sha256", None),
            name="continuum functional provenance_sha256",
        )
        configuration = _configuration(functional, name="continuum functional")
        behavior = provider_behavior_sha256(
            functional,
            (
                "configuration_sha256",
                "_energy_torch",
                "energy_eV",
                "drive",
                "source_jvp",
                "source_vjp",
                "mixed_coordinate_source_vjp",
                "coordinate_partial",
            ),
            label="scalar_first_continuum_functional",
        )
        adapter_configuration = _sha(
            {
                "adapter_id": SCALAR_FIRST_CONTINUUM_ADAPTER_ID,
                "functional_provider_id": provider,
                "functional_configuration_sha256": configuration,
                "functional_provenance_sha256": provenance,
                "functional_behavior_sha256": behavior,
                "scalar_id": scalar_id,
                "continuum_profile_id": continuum_profile,
                "cavity_profile_id": cavity_profile,
                "configuration_contract_id": configuration_contract,
                "coupling_id": coupling,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )
        adapter_provenance = _sha(
            {
                "adapter_id": SCALAR_FIRST_CONTINUUM_ADAPTER_ID,
                "configuration_sha256": adapter_configuration,
                "functional_provenance_sha256": provenance,
            }
        )
        object.__setattr__(self, "_functional", functional)
        object.__setattr__(self, "provider_id", SCALAR_FIRST_CONTINUUM_ADAPTER_ID)
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(self, "continuum_profile_id", continuum_profile)
        object.__setattr__(self, "cavity_profile_id", cavity_profile)
        object.__setattr__(self, "configuration_contract_id", configuration_contract)
        object.__setattr__(self, "coupling_id", coupling)
        object.__setattr__(self, "provenance_sha256", adapter_provenance)
        object.__setattr__(self, "source_space", functional.source_space)
        object.__setattr__(self, "field_space", functional.field_space)
        object.__setattr__(self, "_configuration_sha256", adapter_configuration)
        object.__setattr__(self, "_sealed", True)
        self.configuration_sha256()

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("scalar-first continuum adapter is immutable.")
        object.__setattr__(self, name, value)

    @property
    def functional(self) -> ContinuumEnergyFunctional:
        self.configuration_sha256()
        return self._functional

    def _current_configuration_sha256(self) -> str:
        functional = self._functional
        return _sha(
            {
                "adapter_id": SCALAR_FIRST_CONTINUUM_ADAPTER_ID,
                "functional_provider_id": getattr(functional, "provider_id"),
                "functional_configuration_sha256": _configuration(
                    functional, name="continuum functional"
                ),
                "functional_provenance_sha256": getattr(
                    functional, "provenance_sha256"
                ),
                "functional_behavior_sha256": provider_behavior_sha256(
                    functional,
                    (
                        "configuration_sha256",
                        "_energy_torch",
                        "energy_eV",
                        "drive",
                        "source_jvp",
                        "source_vjp",
                        "mixed_coordinate_source_vjp",
                        "coordinate_partial",
                    ),
                    label="scalar_first_continuum_functional",
                ),
                "scalar_id": self.scalar_id,
                "continuum_profile_id": self.continuum_profile_id,
                "cavity_profile_id": self.cavity_profile_id,
                "configuration_contract_id": self.configuration_contract_id,
                "coupling_id": self.coupling_id,
                "implementation_sha256": _implementation_sha256(),
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        _capabilities_are_closed(self._functional, name="continuum functional")
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("scalar-first continuum configuration drifted.")
        return current

    def evaluate_field(self, geometry: object, source: np.ndarray) -> np.ndarray:
        self.configuration_sha256()
        return self._functional.drive(geometry, source)

    def source_jvp(
        self,
        geometry: object,
        source: np.ndarray,
        source_direction: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._functional.source_jvp(geometry, source, source_direction)

    def source_vjp(
        self,
        geometry: object,
        source: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        return self._functional.source_vjp(geometry, source, field_cotangent)

    def coordinate_vjp(
        self,
        geometry: object,
        source: np.ndarray,
        field_cotangent: np.ndarray,
    ) -> np.ndarray:
        self.configuration_sha256()
        result = np.asarray(
            self._functional.mixed_coordinate_source_vjp(
                geometry, source, field_cotangent
            ),
            dtype=float,
        )
        count = np.asarray(source).shape[0]
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise ValueError(
                "continuum scalar mixed coordinate VJP must be finite (N,3)."
            )
        return result.reshape(-1).copy()


__all__ = [
    "SCALAR_FIRST_CONTINUUM_ADAPTER_ID",
    "SCALAR_FIRST_ELECTRONIC_ADAPTER_ID",
    "ScalarFirstContinuumResponseAdapter",
    "ScalarFirstElectronicResponseAdapter",
]
