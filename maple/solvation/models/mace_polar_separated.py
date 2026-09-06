"""MACE-POLAR source4/native-field8 operational adapter and semantics canary.

This module is deliberately model-specific at the edge and exposes a generic
separated response surface to the Route-2 kernel.  It never promotes the
checkpoint's original source to the gradient of its intrinsic field energy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np

from maple.solvation.coupling.exact_gto import (
    embed_mace_polar_learned_source,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.separated_operators import (
    MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE,
    NativeFieldSpace,
    SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID,
)
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE, SourceSpace

from .base import atom_count
from .mace_polar import MACEPolarRadialGTOModelAdapter

MACE_POLAR_SEPARATED_RESPONSE_PROVIDER_ID = (
    "maple.route2.model.mace-polar-original-source4-native-field8.impl.v1"
)
MACE_POLAR_NATIVE_SEMANTICS_CONTRACT = "route2-mace-polar-native-semantics-v1"

_LEARNED_SOURCE_INDICES = (0, 2, 3, 4)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    result = _nonempty(value, name=name).lower()
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _finite_tuple(values: object, *, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return tuple(float(value) for value in array)


def _jsonable_mapping(values: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, dict):
        raise TypeError(f"{name} must be a dictionary.")
    result = tuple(sorted((str(key), str(value)) for key, value in values.items()))
    if any(not key.strip() or not value.strip() for key, value in result):
        raise ValueError(f"{name} keys and values must be non-empty.")
    return result


@dataclass(frozen=True, slots=True)
class NativeSemanticsCanary:
    """Content-addressed facts read from one loaded MACE-POLAR checkpoint."""

    checkpoint_sha256: str
    model_provider_id: str
    model_profile_id: str
    model_class: str
    source_sigmas_angstrom: tuple[float, ...]
    source_max_l: int
    source_normalization: str
    receiver_sigmas_angstrom: tuple[float, ...]
    receiver_max_l: int
    receiver_normalization: str
    receiver_feature_layout: str
    projection_matrix_shape: tuple[int, int]
    projection_matrix_sha256: str
    field_transform_sha256: str
    source_component_count: int
    receiver_component_count: int
    spin_channel_count: int
    recursion_steps: int
    response_update_module_count: int
    interaction_count: int
    field_feature_norms: tuple[float, ...]
    field_norm_factor: float
    field_self_interaction: bool
    electrostatic_self_interaction: bool
    add_local_electron_energy: bool
    fixedpoint_update_config: tuple[tuple[str, str], ...]
    field_readout_config: tuple[tuple[str, str], ...]
    source_receiver_spaces_distinct: bool
    applied_field_energy_sign_verified: bool = False
    intrinsic_energy_is_complete_external_enthalpy: bool = False
    contract_version: str = MACE_POLAR_NATIVE_SEMANTICS_CONTRACT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _digest(self.checkpoint_sha256, name="checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "projection_matrix_sha256",
            _digest(self.projection_matrix_sha256, name="projection_matrix_sha256"),
        )
        object.__setattr__(
            self,
            "field_transform_sha256",
            _digest(self.field_transform_sha256, name="field_transform_sha256"),
        )
        for name in (
            "model_provider_id",
            "model_profile_id",
            "model_class",
            "source_normalization",
            "receiver_normalization",
            "receiver_feature_layout",
            "contract_version",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        for name in ("source_sigmas_angstrom", "receiver_sigmas_angstrom"):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or any(
                not np.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} must contain positive finite widths.")
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "field_feature_norms",
            _finite_tuple(self.field_feature_norms, name="field_feature_norms"),
        )
        for name in (
            "source_max_l",
            "receiver_max_l",
            "source_component_count",
            "receiver_component_count",
            "spin_channel_count",
            "recursion_steps",
            "response_update_module_count",
            "interaction_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        shape = tuple(self.projection_matrix_shape)
        if len(shape) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in shape
        ):
            raise ValueError(
                "projection_matrix_shape must contain two positive integers."
            )
        object.__setattr__(self, "projection_matrix_shape", shape)
        if not np.isfinite(self.field_norm_factor) or self.field_norm_factor <= 0.0:
            raise ValueError("field_norm_factor must be finite and positive.")
        for name in (
            "field_self_interaction",
            "electrostatic_self_interaction",
            "add_local_electron_energy",
            "source_receiver_spaces_distinct",
            "applied_field_energy_sign_verified",
            "intrinsic_energy_is_complete_external_enthalpy",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be exactly bool.")
        if len(self.field_feature_norms) != self.receiver_component_count:
            raise ValueError("field_feature_norms must match receiver component count.")

    @classmethod
    def from_adapter(
        cls, adapter: MACEPolarRadialGTOModelAdapter
    ) -> "NativeSemanticsCanary":
        if not isinstance(adapter, MACEPolarRadialGTOModelAdapter):
            raise TypeError("adapter must be MACEPolarRadialGTOModelAdapter.")
        adapter.configuration_sha256()
        calculator = adapter._calculator
        model = getattr(calculator, "model", None)
        source_basis = getattr(
            getattr(model, "coulomb_energy", None), "density_basis", None
        )
        if source_basis is None:
            raise RuntimeError("loaded checkpoint does not expose its source basis.")
        spec = calculator.route2_gto_field_projection_spec()
        projection = np.asarray(spec.upstream_matrix, dtype=float)
        if projection.ndim != 2 or not np.all(np.isfinite(projection)):
            raise RuntimeError("checkpoint receiver projection is invalid.")
        provenance = getattr(spec, "provenance", None)
        if not isinstance(provenance, dict):
            raise RuntimeError("checkpoint receiver projection provenance is absent.")
        feature_layout = _nonempty(
            provenance.get("feature_layout"), name="receiver feature layout"
        )
        upstream_matrix_sha256 = _digest(
            getattr(spec, "upstream_matrix_sha256", None),
            name="checkpoint upstream projection SHA256",
        )
        if provenance.get("upstream_matrix_sha256") != upstream_matrix_sha256:
            raise RuntimeError(
                "checkpoint receiver projection provenance does not bind the live matrix."
            )
        field_irreps = getattr(model, "field_irreps", None)
        potential_irreps = getattr(model, "potential_irreps", None)
        field_dimension = int(getattr(field_irreps, "dim", projection.shape[0]))
        potential_dimension = int(getattr(potential_irreps, "dim", 2 * field_dimension))
        if field_dimension < 1 or potential_dimension % field_dimension:
            raise RuntimeError("checkpoint spin-potential irreps are inconsistent.")
        feature_norms_raw = getattr(model, "field_feature_norms", None)
        if hasattr(feature_norms_raw, "detach"):
            feature_norms_raw = feature_norms_raw.detach().cpu().numpy()
        source_sigmas = tuple(float(value) for value in source_basis.sigmas)
        receiver_sigmas = tuple(float(value) for value in spec.receiver_sigmas_angstrom)
        source_components = len(source_sigmas) * (int(source_basis.max_l) + 1) ** 2
        receiver_components = len(receiver_sigmas) * (int(spec.receiver_max_l) + 1) ** 2
        return cls(
            checkpoint_sha256=adapter.provenance.checkpoint_sha256,
            model_provider_id=adapter.provider_id,
            model_profile_id=adapter.model_profile_id,
            model_class=f"{type(model).__module__}.{type(model).__qualname__}",
            source_sigmas_angstrom=source_sigmas,
            source_max_l=int(source_basis.max_l),
            source_normalization=str(source_basis.normalize),
            receiver_sigmas_angstrom=receiver_sigmas,
            receiver_max_l=int(spec.receiver_max_l),
            receiver_normalization=str(spec.receiver_normalization),
            receiver_feature_layout=feature_layout,
            projection_matrix_shape=tuple(int(value) for value in projection.shape),
            projection_matrix_sha256=upstream_matrix_sha256,
            field_transform_sha256=adapter.field_transform.configuration_sha256(),
            source_component_count=source_components,
            receiver_component_count=receiver_components,
            spin_channel_count=potential_dimension // field_dimension,
            recursion_steps=int(getattr(model, "num_recursion_steps")),
            response_update_module_count=len(model.field_dependent_charges_maps),
            interaction_count=len(model.interactions),
            field_feature_norms=_finite_tuple(
                feature_norms_raw, name="checkpoint field_feature_norms"
            ),
            field_norm_factor=float(getattr(model, "field_norm_factor")),
            field_self_interaction=bool(getattr(model, "field_si")),
            electrostatic_self_interaction=bool(
                getattr(model, "include_electrostatic_self_interaction")
            ),
            add_local_electron_energy=bool(getattr(model, "add_local_electron_energy")),
            fixedpoint_update_config=_jsonable_mapping(
                getattr(model, "_fixedpoint_update_config"),
                name="fixedpoint_update_config",
            ),
            field_readout_config=_jsonable_mapping(
                getattr(model, "_field_readout_config"),
                name="field_readout_config",
            ),
            source_receiver_spaces_distinct=(
                source_components != receiver_components
                or source_sigmas != receiver_sigmas
                or str(source_basis.normalize) != str(spec.receiver_normalization)
            ),
        )

    @property
    def separated_operational_contract_complete(self) -> bool:
        """Whether runtime metadata is sufficient to construct C4/B/A/L/U8.

        This does not verify the energy ledger sign and does not admit Tier V.
        """

        return (
            self.source_component_count == 4
            and self.receiver_component_count == 8
            and self.projection_matrix_shape == (8, 4)
            and self.source_max_l == self.receiver_max_l == 1
            and self.recursion_steps == self.response_update_module_count
            and self.spin_channel_count == 2
            and self.source_receiver_spaces_distinct
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "configuration_sha256": self.configuration_sha256(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_provider_id": self.model_provider_id,
            "model_profile_id": self.model_profile_id,
            "model_class": self.model_class,
            "source_sigmas_angstrom": list(self.source_sigmas_angstrom),
            "source_max_l": self.source_max_l,
            "source_normalization": self.source_normalization,
            "receiver_sigmas_angstrom": list(self.receiver_sigmas_angstrom),
            "receiver_max_l": self.receiver_max_l,
            "receiver_normalization": self.receiver_normalization,
            "receiver_feature_layout": self.receiver_feature_layout,
            "projection_matrix_shape": list(self.projection_matrix_shape),
            "projection_matrix_sha256": self.projection_matrix_sha256,
            "field_transform_sha256": self.field_transform_sha256,
            "source_component_count": self.source_component_count,
            "receiver_component_count": self.receiver_component_count,
            "spin_channel_count": self.spin_channel_count,
            "recursion_steps": self.recursion_steps,
            "response_update_module_count": self.response_update_module_count,
            "interaction_count": self.interaction_count,
            "field_feature_norms": list(self.field_feature_norms),
            "field_norm_factor": self.field_norm_factor,
            "field_self_interaction": self.field_self_interaction,
            "electrostatic_self_interaction": self.electrostatic_self_interaction,
            "add_local_electron_energy": self.add_local_electron_energy,
            "fixedpoint_update_config": dict(self.fixedpoint_update_config),
            "field_readout_config": dict(self.field_readout_config),
            "source_receiver_spaces_distinct": self.source_receiver_spaces_distinct,
            "separated_operational_contract_complete": (
                self.separated_operational_contract_complete
            ),
            "applied_field_energy_sign_verified": (
                self.applied_field_energy_sign_verified
            ),
            "intrinsic_energy_is_complete_external_enthalpy": (
                self.intrinsic_energy_is_complete_external_enthalpy
            ),
            "claim_boundary": (
                "Runtime metadata proves the 4-source/8-receiver category split. "
                "It does not select Phi0 versus Phi1 and does not prove Tier V."
            ),
        }

    def configuration_sha256(self) -> str:
        payload = {
            key: value for key, value in self.as_dict_without_configuration().items()
        }
        return canonical_metadata_sha256(payload)

    def as_dict_without_configuration(self) -> dict[str, object]:
        result = {field: getattr(self, field) for field in self.__dataclass_fields__}
        result["source_sigmas_angstrom"] = list(self.source_sigmas_angstrom)
        result["receiver_sigmas_angstrom"] = list(self.receiver_sigmas_angstrom)
        result["projection_matrix_shape"] = list(self.projection_matrix_shape)
        result["field_feature_norms"] = list(self.field_feature_norms)
        result["fixedpoint_update_config"] = dict(self.fixedpoint_update_config)
        result["field_readout_config"] = dict(self.field_readout_config)
        result["separated_operational_contract_complete"] = (
            self.separated_operational_contract_complete
        )
        return result


class MACEPolarOriginalSourceNativeFieldAdapter:
    """Expose original four-channel source response to native eight-channel field."""

    __slots__ = (
        "_base",
        "_configuration_sha256",
        "_sealed",
        "coupling_id",
        "model_profile_id",
        "provenance_sha256",
        "provider_id",
    )

    source_space: SourceSpace = ATOMIC_L1_SOURCE_SPACE
    receiver_space: NativeFieldSpace = MACE_POLAR_NATIVE_RADIAL_FIELD_SPACE
    capabilities = ()
    variational_functional_admitted = False
    coordinate_derivative_available = True

    def __init__(self, base: MACEPolarRadialGTOModelAdapter) -> None:
        if not isinstance(base, MACEPolarRadialGTOModelAdapter):
            raise TypeError("base must be MACEPolarRadialGTOModelAdapter.")
        base.configuration_sha256()
        object.__setattr__(self, "_base", base)
        object.__setattr__(
            self, "provider_id", MACE_POLAR_SEPARATED_RESPONSE_PROVIDER_ID
        )
        object.__setattr__(self, "model_profile_id", base.model_profile_id)
        object.__setattr__(
            self, "coupling_id", SEPARATED_MACE_POLAR_HARMONIC_COUPLING_ID
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            canonical_metadata_sha256(
                {
                    "provider_id": self.provider_id,
                    "base_provenance_sha256": base.provenance_sha256,
                    "implementation_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "source_space_sha256": self.source_space.metadata_hash(),
                    "receiver_space_sha256": self.receiver_space.metadata_hash(),
                    "capabilities": "none",
                }
            ),
        )
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "MACEPolarOriginalSourceNativeFieldAdapter is immutable."
            )
        object.__setattr__(self, name, value)

    @property
    def base(self) -> MACEPolarRadialGTOModelAdapter:
        self.configuration_sha256()
        return self._base

    @property
    def checkpoint_sha256(self) -> str:
        self.configuration_sha256()
        return self._base.provenance.checkpoint_sha256

    @property
    def long_range_evaluator_profile(self) -> str:
        self.configuration_sha256()
        return self._base.release_contract.long_range_evaluator_profile

    @property
    def field_energy_pairing_sha256(self) -> str:
        self.configuration_sha256()
        return MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash()

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "mace-polar-original-source4-native-field8-adapter-v1",
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "provenance_sha256": self.provenance_sha256,
                "base_configuration_sha256": self._base.configuration_sha256(),
                "base_provenance_sha256": self._base.provenance_sha256,
                "checkpoint_sha256": self._base.provenance.checkpoint_sha256,
                "source_space_sha256": self.source_space.metadata_hash(),
                "receiver_space_sha256": self.receiver_space.metadata_hash(),
                "field_energy_pairing_sha256": (
                    MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash()
                ),
                "learned_source_indices": list(_LEARNED_SOURCE_INDICES),
                "capabilities": "none",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("separated MACE-POLAR response configuration drifted.")
        return current

    def _field(self, geometry: object, field: object) -> np.ndarray:
        return self.receiver_space.validate(
            field, atom_count=atom_count(geometry), name="native receiver field"
        )

    @staticmethod
    def _extract(radial_source: object, *, atom_count_value: int) -> np.ndarray:
        values = np.asarray(radial_source, dtype=float)
        if values.shape != (atom_count_value, 8) or not np.all(np.isfinite(values)):
            raise ValueError("radial source must be finite with shape (N,8).")
        return ATOMIC_L1_SOURCE_SPACE.validate(
            values[:, _LEARNED_SOURCE_INDICES], atom_count=atom_count_value
        )

    @staticmethod
    def _embed(source4: object, *, atom_count_value: int) -> np.ndarray:
        values = ATOMIC_L1_SOURCE_SPACE.validate(
            source4, atom_count=atom_count_value, name="source4"
        )
        return embed_mace_polar_learned_source(values)

    def evaluate_source(self, geometry: object, field: object) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        state = self._base.evaluate_source(
            geometry,
            self._field(geometry, field),
            need_fixed_field_forces=False,
        )
        return self._extract(state.source, atom_count_value=count)

    def vacuum_energy_ev(self, geometry: object) -> float:
        """Return the exact zero-field checkpoint energy used by Phi0 ledgers."""

        self.configuration_sha256()
        state = self._base.evaluate_vacuum(geometry, need_forces=False)
        value = float(state.energy_eV)
        if not np.isfinite(value):
            raise RuntimeError("MACE-POLAR vacuum energy is non-finite.")
        return value

    def vacuum_forces_ev_per_angstrom(self, geometry: object) -> np.ndarray:
        """Return the exact zero-field checkpoint force for operational ledgers."""

        self.configuration_sha256()
        count = atom_count(geometry)
        state = self._base.evaluate_vacuum(geometry, need_forces=True)
        result = np.asarray(state.forces_eV_per_A, dtype=float)
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(
                "MACE-POLAR vacuum force must be finite with shape (N,3)."
            )
        return result.copy()

    def field_jvp(
        self, geometry: object, field: object, field_direction: object
    ) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        direction = self.receiver_space.validate(
            field_direction, atom_count=count, name="native field direction"
        )
        result = self._base.source_jvp(
            geometry, self._field(geometry, field), direction
        )
        return self._extract(result, atom_count_value=count)

    def field_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        cotangent8 = self._embed(source_cotangent, atom_count_value=count)
        result = self._base.source_vjp(
            geometry, self._field(geometry, field), cotangent8
        )
        return self.receiver_space.validate(
            result, atom_count=count, name="native field cotangent"
        )

    def coordinate_vjp(
        self, geometry: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        self.configuration_sha256()
        count = atom_count(geometry)
        result = np.asarray(
            self._base.source_position_vjp(
                geometry,
                self._field(geometry, field),
                self._embed(source_cotangent, atom_count_value=count),
            ),
            dtype=float,
        )
        if result.shape != (count, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError("separated source coordinate VJP must be finite (N,3).")
        return result.copy()

    def conditioned_raw_energy_ev(self, geometry: object, field: object) -> float:
        """Return the native-injection branch's uninterpreted raw scalar.

        Until the upstream replay and work-sign canaries are complete, this is
        deliberately not named an internal energy or external enthalpy.
        """

        self.configuration_sha256()
        return self._base.intrinsic_energy_ev(geometry, self._field(geometry, field))

    def conditioned_raw_energy_field_gradient(
        self, geometry: object, field: object
    ) -> np.ndarray:
        self.configuration_sha256()
        result = self._base.intrinsic_energy_field_gradient(
            geometry, self._field(geometry, field)
        )
        return self.receiver_space.validate(
            result,
            atom_count=atom_count(geometry),
            name="intrinsic energy native-field gradient",
        )

    def conditioned_raw_energy_directional_derivative(
        self,
        geometry: object,
        field: object,
        field_direction: object,
    ) -> float:
        self.configuration_sha256()
        count = atom_count(geometry)
        return self._base.intrinsic_energy_field_directional_derivative(
            geometry,
            self._field(geometry, field),
            self.receiver_space.validate(
                field_direction,
                atom_count=count,
                name="native field direction",
            ),
        )

    # Compatibility aliases for audit artifacts created before the raw-energy
    # semantics split.  New ledgers and canaries consume the conditioned_raw_*
    # names so that an unverified physical interpretation is not encoded in an
    # API method name.
    intrinsic_energy_ev = conditioned_raw_energy_ev
    intrinsic_energy_field_gradient = conditioned_raw_energy_field_gradient

    def dense_source_jacobian(self, geometry: object, field: object) -> np.ndarray:
        """Return audit-only ``d source4 / d native-field8``."""

        self.configuration_sha256()
        count = atom_count(geometry)
        radial = np.asarray(
            self._base.dense_source_jacobian(geometry, self._field(geometry, field)),
            dtype=float,
        )
        row_indices = np.concatenate(
            [
                np.asarray(_LEARNED_SOURCE_INDICES) + 8 * atom_index
                for atom_index in range(count)
            ]
        )
        result = radial[row_indices, :]
        expected = (count * 4, count * 8)
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise RuntimeError("separated dense source Jacobian has an invalid shape.")
        result = result.copy()
        result.setflags(write=False)
        return result


__all__ = [
    "MACE_POLAR_NATIVE_SEMANTICS_CONTRACT",
    "MACE_POLAR_SEPARATED_RESPONSE_PROVIDER_ID",
    "MACEPolarOriginalSourceNativeFieldAdapter",
    "NativeSemanticsCanary",
]
