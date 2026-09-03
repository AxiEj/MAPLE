"""Registered checkpoint contracts for atomwise-potential MACE-EF models.

Continuum coupling depends on a small, explicit electronic-model contract.
Adding another compatible checkpoint therefore requires a new immutable spec,
not a copy of the smooth-PCM, stationary-SCF, force, or Hessian implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MACEPolarEFCheckpointSpec:
    """Byte, schema, and electrostatic-input identity for one checkpoint."""

    name: str
    model_id: str
    model_family: str
    checkpoint_sha256: str
    checkpoint_size: int
    wrapper_source_fragments: tuple[str, ...]
    forward_schema_fragments: tuple[str, ...]
    supported_atomic_numbers: tuple[int, ...]
    node_attribute_width: int
    cutoff_angstrom: float
    device_type: str
    device_index: int
    tensor_dtype: str
    spin_input_semantics: str
    field_input: str
    checkpoint_vector_input: str
    wrapper_internal_vector: str
    gradient_to_checkpoint_vector_scale: float
    autograd_order: int = 1

    def __post_init__(self) -> None:
        for name in (
            "name",
            "model_id",
            "model_family",
            "checkpoint_sha256",
            "device_type",
            "tensor_dtype",
            "spin_input_semantics",
            "field_input",
            "checkpoint_vector_input",
            "wrapper_internal_vector",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be one lowercase SHA-256.")
        if self.checkpoint_size <= 0:
            raise ValueError("checkpoint_size must be positive.")
        if not self.wrapper_source_fragments or not self.forward_schema_fragments:
            raise ValueError("Checkpoint wrapper and schema fragments are required.")
        if (
            not self.supported_atomic_numbers
            or tuple(sorted(set(self.supported_atomic_numbers)))
            != self.supported_atomic_numbers
        ):
            raise ValueError("supported_atomic_numbers must be sorted and unique.")
        if self.cutoff_angstrom <= 0.0:
            raise ValueError("cutoff_angstrom must be positive.")
        if self.node_attribute_width < max(self.supported_atomic_numbers):
            raise ValueError("node_attribute_width must cover every supported element.")
        if self.device_type not in {"cpu", "cuda"} or self.device_index < 0:
            raise ValueError("Checkpoint device contract is invalid.")
        if self.tensor_dtype != "float32":
            raise ValueError("The current MACE-EF adapter supports float32 only.")
        if self.spin_input_semantics not in {"multiplicity", "two-s"}:
            raise ValueError("spin_input_semantics must be multiplicity or two-s.")
        if self.gradient_to_checkpoint_vector_scale not in {-1.0, 1.0}:
            raise ValueError("gradient_to_checkpoint_vector_scale must be -1 or +1.")
        if self.autograd_order not in {1, 2}:
            raise ValueError("autograd_order must be 1 or 2.")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size": self.checkpoint_size,
            "wrapper_source_fragments": list(self.wrapper_source_fragments),
            "forward_schema_fragments": list(self.forward_schema_fragments),
            "supported_atomic_numbers": list(self.supported_atomic_numbers),
            "node_attribute_width": self.node_attribute_width,
            "cutoff_angstrom": self.cutoff_angstrom,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "tensor_dtype": self.tensor_dtype,
            "spin_input_semantics": self.spin_input_semantics,
            "field_input": self.field_input,
            "checkpoint_vector_input": self.checkpoint_vector_input,
            "wrapper_internal_vector": self.wrapper_internal_vector,
            "gradient_to_checkpoint_vector_scale": (
                self.gradient_to_checkpoint_vector_scale
            ),
            "autograd_order": self.autograd_order,
        }


_CHECKPOINT_SPECS: dict[str, MACEPolarEFCheckpointSpec] = {}


def register_mace_polar_ef_checkpoint_spec(
    spec: MACEPolarEFCheckpointSpec,
) -> MACEPolarEFCheckpointSpec:
    """Register one exact checkpoint contract without permitting replacement."""

    if not isinstance(spec, MACEPolarEFCheckpointSpec):
        raise TypeError("spec must be MACEPolarEFCheckpointSpec.")
    key = spec.name.strip().lower()
    existing = _CHECKPOINT_SPECS.get(key)
    if existing is not None and existing != spec:
        raise ValueError(f"MACE-EF checkpoint spec {key!r} is already registered.")
    _CHECKPOINT_SPECS[key] = spec
    return spec


def mace_polar_ef_checkpoint_spec(name: object) -> MACEPolarEFCheckpointSpec:
    key = str(name).strip().lower()
    try:
        return _CHECKPOINT_SPECS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_CHECKPOINT_SPECS))
        raise ValueError(
            f"Unknown MACE-EF checkpoint spec {name!r}; registered: {supported}."
        ) from exc


def registered_mace_polar_ef_checkpoint_specs() -> (
    Mapping[str, MACEPolarEFCheckpointSpec]
):
    return MappingProxyType(dict(_CHECKPOINT_SPECS))


MACE_POLAR_EF_V2_CHECKPOINT_SPEC = register_mace_polar_ef_checkpoint_spec(
    MACEPolarEFCheckpointSpec(
        name="mace-polar-ef-v2",
        model_id="mace-polar-ef-v2-energy-functional",
        model_family="mace-polar-ef-v2",
        checkpoint_sha256=(
            "4f820d381d06bbb37b02574c38da7203" "e5d429407fa2a512231fc08cdbb69b6b"
        ),
        checkpoint_size=34_581_570,
        wrapper_source_fragments=(
            "class PolarMACEEFTraceable",
            "external_field0 = torch.neg(external_field)",
            "external_potential_values = torch.unsqueeze(external_potential, -1)",
        ),
        forward_schema_fragments=(
            "Tensor external_field",
            "Tensor external_potential",
            "Tensor local_or_ghost",
            "-> ((Tensor, Tensor, Tensor))",
        ),
        supported_atomic_numbers=tuple(range(1, 84)),
        node_attribute_width=83,
        cutoff_angstrom=6.0,
        device_type="cuda",
        device_index=0,
        tensor_dtype="float32",
        spin_input_semantics="multiplicity",
        field_input="atomwise [potential, grad_x, grad_y, grad_z]",
        checkpoint_vector_input="physical electric field E=-grad(V)",
        wrapper_internal_vector="grad(V) after its built-in negation",
        gradient_to_checkpoint_vector_scale=-1.0,
        autograd_order=1,
    )
)


__all__ = [
    "MACE_POLAR_EF_V2_CHECKPOINT_SPEC",
    "MACEPolarEFCheckpointSpec",
    "mace_polar_ef_checkpoint_spec",
    "register_mace_polar_ef_checkpoint_spec",
    "registered_mace_polar_ef_checkpoint_specs",
]
