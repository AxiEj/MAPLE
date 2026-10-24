"""Typed identities and values for same-scalar analytic composition.

This module deliberately knows nothing about a particular MLIP or solvent
provider.  Adapters stamp each operation with the identity produced by the
actual graph used for that operation; composition then validates identity,
shape, unit, and finiteness before adding gas and solvent values once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from typing import Any, Literal

import numpy as np


AnalyticOperation = Literal["energy_forces", "hessian", "hvp"]
Component = Literal["gas", "solvent"]
EnergyUnit = Literal["hartree", "eV"]
FrozenOptions = tuple[tuple[str, str], ...]
NamedCutoffs = tuple[tuple[str, float], ...]
FileStatIdentity = tuple[int, int, int, int]

_OPERATIONS = frozenset({"energy_forces", "hessian", "hvp"})
_HEX_DIGITS = frozenset("0123456789abcdef")
NEIGHBOR_CUTOFF_GUARD_POLICY = "neighbor-cutoff-guard-v1"
NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM = 0.003


class AnalyticIdentityError(ValueError):
    """An operation no longer matches its admitted immutable scalar identity."""


class AnalyticCapabilityError(NotImplementedError):
    """A requested analytic operation is not qualified for the exact cell."""


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Identity options cannot contain non-finite floats.")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise TypeError("Identity option mapping keys must be non-empty strings.")
            result[key] = _canonical_json_value(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        )
    raise TypeError(
        "Identity options must contain JSON-like scalar, mapping, or sequence values; "
        f"received {type(value).__name__}."
    )


def freeze_identity_options(options: Mapping[str, Any] | None) -> FrozenOptions:
    """Return a deterministic immutable representation of identity options."""
    if options is None:
        return ()
    if not isinstance(options, Mapping):
        raise TypeError("Identity options must be a mapping.")
    frozen = []
    for key in sorted(options):
        if not isinstance(key, str) or not key:
            raise TypeError("Identity option keys must be non-empty strings.")
        encoded = json.dumps(
            _canonical_json_value(options[key]),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        frozen.append((key, encoded))
    return tuple(frozen)


def _validate_frozen_options(value: FrozenOptions, *, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be produced by freeze_identity_options().")
    keys = []
    for entry in value:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not entry[0]
            or not isinstance(entry[1], str)
        ):
            raise TypeError(f"{name} must contain (non-empty string, JSON string) pairs.")
        key, encoded = entry
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}[{key!r}] is not valid JSON.") from exc
        canonical = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if encoded != canonical:
            raise ValueError(f"{name}[{key!r}] is not canonical JSON.")
        keys.append(key)
    if keys != sorted(set(keys)):
        raise ValueError(f"{name} keys must be unique and sorted.")


def _canonical_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{name} cannot contain object-dtype values.")
    if array.dtype.byteorder in {"=", ">"} and array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


def named_array_sha256(arrays: Mapping[str, Any]) -> str:
    """Hash named arrays independent of mapping order and tensor device.

    Tensor-owning adapters must pass detached CPU arrays from the exact graph
    used by the operation.  Names, dtypes, shapes, and C-order bytes are all
    included, so a changed edge order or sentinel layout changes the identity.
    """
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("named_array_sha256 requires a non-empty mapping.")
    digest = hashlib.sha256(b"maple-named-array-v1\0")
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise TypeError("Named-array keys must be non-empty strings.")
        array = _canonical_array(arrays[name], name=name)
        header = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def geometry_sha256(atoms) -> str:
    """Hash ordered atomic numbers and exact float64 Cartesian coordinates."""
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype="<i8")
    positions = np.asarray(atoms.get_positions(), dtype="<f8")
    if numbers.ndim != 1 or positions.shape != (numbers.size, 3):
        raise ValueError("Geometry identity requires ordered Z and an (N, 3) position array.")
    if not np.isfinite(positions).all():
        raise ValueError("Geometry identity cannot contain non-finite coordinates.")
    return named_array_sha256({"atomic_numbers": numbers, "positions_angstrom": positions})


def file_stat_identity(stat_result) -> FileStatIdentity:
    """Return the filesystem identity used to bind loaded checkpoint bytes."""
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


@dataclass(frozen=True, slots=True)
class CheckpointFileIdentity:
    path: str
    stat_identity: FileStatIdentity
    sha256: str

    @classmethod
    def bind(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_stat_identity: FileStatIdentity | None = None,
    ) -> "CheckpointFileIdentity":
        realpath = os.path.realpath(os.fspath(path))
        digest = hashlib.sha256()
        with open(realpath, "rb") as handle:
            before = file_stat_identity(os.fstat(handle.fileno()))
            if expected_stat_identity is not None and before != expected_stat_identity:
                raise AnalyticIdentityError(
                    "Checkpoint file identity changed after the model was loaded."
                )
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = file_stat_identity(os.fstat(handle.fileno()))
        if after != before or file_stat_identity(os.stat(realpath)) != after:
            raise AnalyticIdentityError("Checkpoint file changed while it was being hashed.")
        return cls(realpath, after, digest.hexdigest())

    def require_unchanged(self) -> None:
        try:
            current = file_stat_identity(os.stat(self.path))
        except OSError as exc:
            raise AnalyticIdentityError(
                f"Bound checkpoint is no longer readable: {self.path}"
            ) from exc
        if current != self.stat_identity:
            raise AnalyticIdentityError(
                "Checkpoint file identity changed after analytic admission."
            )


def _torch_precision_state(device) -> dict[str, Any]:
    import torch

    device_type = getattr(device, "type", None) or str(device).split(":", 1)[0]
    try:
        autocast = bool(torch.is_autocast_enabled(str(device_type)))
    except TypeError:  # pragma: no cover - compatibility with older supported Torch
        autocast = bool(torch.is_autocast_enabled())
    return {
        "autocast_enabled": autocast,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "device_type": str(device_type),
        "grad_enabled": bool(torch.is_grad_enabled()),
        "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
    }


def torch_response_settings(device) -> FrozenOptions:
    """Record active Torch AD/AMP/TF32 state without changing global settings."""
    return freeze_identity_options(_torch_precision_state(device))


def require_no_amp_tf32(device) -> None:
    """Fail closed when AMP/autocast or CUDA TF32 could enter analytic AD."""
    state = _torch_precision_state(device)
    if state["autocast_enabled"]:
        raise AnalyticCapabilityError("Analytic profiles do not admit autocast/AMP.")
    if state["inference_mode_enabled"]:
        raise AnalyticCapabilityError(
            "Analytic profiles require a differentiable graph, not inference mode."
        )
    if state["device_type"] == "cuda" and (
        state["cuda_matmul_allow_tf32"] or state["cudnn_allow_tf32"]
    ):
        raise AnalyticCapabilityError("Analytic CUDA profiles do not admit TF32.")


def torch_model_runtime_identity(model) -> tuple[tuple[Any, ...], ...]:
    """Snapshot live parameter/buffer object, dtype, device, and mutation state."""
    rows = []
    for kind, iterator_name in (
        ("parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    ):
        iterator = getattr(model, iterator_name, None)
        if not callable(iterator):
            raise TypeError(f"Torch model does not expose {iterator_name}().")
        for name, tensor in iterator():
            rows.append(
                (
                    kind,
                    str(name),
                    tuple(int(value) for value in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                    int(tensor.data_ptr()),
                    int(tensor._version),
                )
            )
    return tuple(sorted(rows, key=lambda row: (row[0], row[1])))


def _require_sha256(value: str | None, *, name: str, optional: bool = False) -> None:
    if value is None and optional:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        qualifier = " or None" if optional else ""
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest{qualifier}.")


@dataclass(frozen=True, slots=True)
class ScalarCellIdentity:
    component: Component
    implementation: str
    model: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    device: str
    effective_dtype: str
    d4: bool | None
    source_energy_unit: EnergyUnit
    cutoff_guard_policy: str | None
    cutoff_guard_margin_angstrom: float | None
    options: FrozenOptions = ()

    def __post_init__(self) -> None:
        if self.component not in {"gas", "solvent"}:
            raise ValueError("Scalar component must be 'gas' or 'solvent'.")
        for name in ("implementation", "model", "device", "effective_dtype"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string.")
        if (self.checkpoint_path is None) != (self.checkpoint_sha256 is None):
            raise ValueError("Checkpoint path and SHA256 must either both be set or both be None.")
        if self.checkpoint_path is not None:
            if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
                raise ValueError("checkpoint_path must be a non-empty string.")
            _require_sha256(self.checkpoint_sha256, name="checkpoint_sha256")
        if self.component == "gas" and self.checkpoint_path is None:
            raise ValueError("Gas analytic cells require an exact checkpoint path and SHA256.")
        if self.d4 is not None and not isinstance(self.d4, bool):
            raise TypeError("d4 must be bool or None.")
        if self.source_energy_unit not in {"hartree", "eV"}:
            raise ValueError("source_energy_unit must be 'hartree' or 'eV'.")
        if (self.cutoff_guard_policy is None) != (
            self.cutoff_guard_margin_angstrom is None
        ):
            raise ValueError("Cutoff guard policy and margin must be specified together.")
        if self.cutoff_guard_policy is not None:
            if self.cutoff_guard_policy != NEIGHBOR_CUTOFF_GUARD_POLICY:
                raise ValueError(
                    f"Unsupported cutoff guard policy: {self.cutoff_guard_policy!r}."
                )
            margin = float(self.cutoff_guard_margin_angstrom)
            if margin != NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM:
                raise ValueError(
                    "neighbor-cutoff-guard-v1 requires the preregistered 0.003 A margin."
                )
        _validate_frozen_options(self.options, name="options")


@dataclass(frozen=True, slots=True)
class ScalarPointIdentity:
    cell: ScalarCellIdentity
    geometry_sha256: str
    atomic_numbers: tuple[int, ...]
    charge: float
    multiplicity: int
    atom_topology_sha256: str | None
    neighbor_graph_sha256: str | None
    active_cutoffs_angstrom: NamedCutoffs = ()
    minimum_cutoff_gap_angstrom: float | None = None
    response_settings: FrozenOptions = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cell, ScalarCellIdentity):
            raise TypeError("cell must be a ScalarCellIdentity.")
        _require_sha256(self.geometry_sha256, name="geometry_sha256")
        if (
            not isinstance(self.atomic_numbers, tuple)
            or not self.atomic_numbers
            or any(
                isinstance(number, bool)
                or not isinstance(number, int)
                or number <= 0
                for number in self.atomic_numbers
            )
        ):
            raise ValueError("atomic_numbers must be a non-empty tuple of positive integers.")
        if not math.isfinite(float(self.charge)):
            raise ValueError("charge must be finite.")
        if (
            isinstance(self.multiplicity, bool)
            or not isinstance(self.multiplicity, int)
            or self.multiplicity <= 0
        ):
            raise ValueError("multiplicity must be a positive integer.")
        _require_sha256(
            self.atom_topology_sha256,
            name="atom_topology_sha256",
            optional=True,
        )
        _require_sha256(
            self.neighbor_graph_sha256,
            name="neighbor_graph_sha256",
            optional=True,
        )
        if not isinstance(self.active_cutoffs_angstrom, tuple):
            raise TypeError("active_cutoffs_angstrom must be an immutable tuple.")
        names = []
        finite_cutoff_present = False
        for entry in self.active_cutoffs_angstrom:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
            ):
                raise TypeError("Active cutoffs must be (non-empty name, value) pairs.")
            name, raw_value = entry
            value = float(raw_value)
            if math.isnan(value) or value <= 0.0:
                raise ValueError("Active cutoffs must be positive or positive infinity.")
            finite_cutoff_present |= math.isfinite(value)
            names.append(name)
        if names != sorted(set(names)):
            raise ValueError("Active cutoff names must be unique and sorted.")
        if self.minimum_cutoff_gap_angstrom is not None:
            gap = float(self.minimum_cutoff_gap_angstrom)
            if math.isnan(gap) or gap < 0.0:
                raise ValueError(
                    "minimum_cutoff_gap_angstrom must be non-negative or positive infinity."
                )
        elif finite_cutoff_present:
            raise ValueError("Finite active cutoffs require an observed minimum cutoff gap.")
        _validate_frozen_options(self.response_settings, name="response_settings")

    @property
    def atom_count(self) -> int:
        return len(self.atomic_numbers)


@dataclass(frozen=True, slots=True)
class CompositePointIdentity:
    gas_identity: ScalarPointIdentity
    solvent_identity: ScalarPointIdentity

    def __post_init__(self) -> None:
        gas = self.gas_identity
        solvent = self.solvent_identity
        if gas.cell.component != "gas" or solvent.cell.component != "solvent":
            raise AnalyticIdentityError(
                "Composite identity requires gas and solvent component identities."
            )
        if gas.geometry_sha256 != solvent.geometry_sha256:
            raise AnalyticIdentityError(
                "Gas and solvent operations do not share the same center geometry."
            )
        if gas.atomic_numbers != solvent.atomic_numbers:
            raise AnalyticIdentityError("Gas and solvent atom order identities differ.")
        if gas.charge != solvent.charge or gas.multiplicity != solvent.multiplicity:
            raise AnalyticIdentityError(
                "Gas and solvent charge/multiplicity identities differ."
            )

    @property
    def atom_count(self) -> int:
        return self.gas_identity.atom_count


def _validate_operation_name(operation: str) -> None:
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown analytic operation: {operation!r}.")


def _read_only_numeric_array(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"{name} must be a real numeric array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _finite_real_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a real scalar, not bool or an array.")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def require_derivative_cutoff_guard(
    point: ScalarPointIdentity,
    operation: Literal["hessian", "hvp"],
) -> None:
    """Apply the preregistered graph-discontinuity refusal before derivative AD."""
    if operation not in {"hessian", "hvp"}:
        raise ValueError("Cutoff guard applies only to hessian or hvp operations.")
    margin = point.cell.cutoff_guard_margin_angstrom
    if margin is None:
        return
    gap = point.minimum_cutoff_gap_angstrom
    if gap is None:
        raise AnalyticIdentityError(
            "Guarded H/HVP identity is missing the observed cutoff gap."
        )
    if gap <= margin:
        raise AnalyticCapabilityError(
            f"Analytic {operation} refused: minimum distance-to-cutoff "
            f"{gap:.17g} A is within the preregistered {margin:.17g} A guard."
        )


@dataclass(frozen=True, slots=True)
class AnalyticOperationResult:
    operation: AnalyticOperation
    identity: ScalarPointIdentity | CompositePointIdentity
    energy_hartree: float | None = None
    forces_hartree_per_angstrom: np.ndarray | None = None
    hessian_hartree_per_angstrom2: np.ndarray | None = None
    direction: np.ndarray | None = None
    hvp_hartree_per_angstrom2: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_operation_name(self.operation)
        if not isinstance(self.identity, (ScalarPointIdentity, CompositePointIdentity)):
            raise TypeError("Analytic operation identity has the wrong type.")
        count = self.identity.atom_count
        forces = self.forces_hartree_per_angstrom
        hessian = self.hessian_hartree_per_angstrom2
        direction = self.direction
        hvp = self.hvp_hartree_per_angstrom2

        if self.operation == "energy_forces":
            if self.energy_hartree is None:
                raise ValueError("energy_forces requires a finite energy_hartree.")
            _finite_real_scalar(self.energy_hartree, name="energy_hartree")
            if hessian is not None or direction is not None or hvp is not None:
                raise ValueError(
                    "energy_forces cannot contain hessian, direction, or HVP values."
                )
        elif self.operation == "hessian":
            if any(value is not None for value in (self.energy_hartree, forces, direction, hvp)):
                raise ValueError("hessian operations can contain only a Hessian value.")
            if hessian is None:
                raise ValueError("hessian operations require a Hessian value.")
        else:
            if hessian is not None:
                raise ValueError("hvp operations cannot contain a full Hessian.")
            if (
                self.energy_hartree is None
                or forces is None
                or direction is None
                or hvp is None
            ):
                raise ValueError("hvp operations require finite HVP, direction, forces, and energy.")
            _finite_real_scalar(self.energy_hartree, name="energy_hartree")

        if forces is not None:
            forces_array = _read_only_numeric_array(
                forces, name="forces_hartree_per_angstrom"
            )
            if forces_array.shape != (count, 3):
                raise ValueError("forces_hartree_per_angstrom must have shape (N, 3).")
            object.__setattr__(self, "forces_hartree_per_angstrom", forces_array)
        if hessian is not None:
            hessian_array = _read_only_numeric_array(
                hessian, name="hessian_hartree_per_angstrom2"
            )
            if hessian_array.shape != (3 * count, 3 * count):
                raise ValueError("Hessian must have shape (3N, 3N).")
            object.__setattr__(self, "hessian_hartree_per_angstrom2", hessian_array)
        if direction is not None:
            direction_array = _read_only_numeric_array(direction, name="direction")
            if direction_array.shape != (3 * count,):
                raise ValueError("direction must have shape (3N,).")
            object.__setattr__(self, "direction", direction_array)
        if hvp is not None:
            hvp_array = _read_only_numeric_array(
                hvp, name="hvp_hartree_per_angstrom2"
            )
            if hvp_array.shape != (3 * count,):
                raise ValueError("hvp_hartree_per_angstrom2 must have shape (3N,).")
            object.__setattr__(self, "hvp_hartree_per_angstrom2", hvp_array)

        if self.operation in {"hessian", "hvp"}:
            point = (
                self.identity.gas_identity
                if isinstance(self.identity, CompositePointIdentity)
                else self.identity
            )
            require_derivative_cutoff_guard(point, self.operation)

    @classmethod
    def energy_forces(
        cls,
        *,
        identity: ScalarPointIdentity | CompositePointIdentity,
        energy_hartree: float,
        forces_hartree_per_angstrom: Any | None,
    ) -> "AnalyticOperationResult":
        return cls(
            operation="energy_forces",
            identity=identity,
            energy_hartree=energy_hartree,
            forces_hartree_per_angstrom=forces_hartree_per_angstrom,
        )

    @classmethod
    def hessian(
        cls,
        *,
        identity: ScalarPointIdentity | CompositePointIdentity,
        hessian_hartree_per_angstrom2: Any,
    ) -> "AnalyticOperationResult":
        return cls(
            operation="hessian",
            identity=identity,
            hessian_hartree_per_angstrom2=hessian_hartree_per_angstrom2,
        )

    @classmethod
    def hvp(
        cls,
        *,
        identity: ScalarPointIdentity | CompositePointIdentity,
        direction: Any,
        hvp_hartree_per_angstrom2: Any,
        forces_hartree_per_angstrom: Any,
        energy_hartree: float,
    ) -> "AnalyticOperationResult":
        return cls(
            operation="hvp",
            identity=identity,
            direction=direction,
            hvp_hartree_per_angstrom2=hvp_hartree_per_angstrom2,
            forces_hartree_per_angstrom=forces_hartree_per_angstrom,
            energy_hartree=energy_hartree,
        )


def _validate_admission_fields(
    profile_id: str,
    evidence_id: str,
    operations: frozenset[str],
) -> None:
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile_id must be a non-empty installed profile name.")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("evidence_id must be a non-empty stable evidence name.")
    if not isinstance(operations, frozenset) or not operations:
        raise TypeError("operations must be a non-empty frozenset.")
    if not operations.issubset(_OPERATIONS):
        raise ValueError(f"Unknown admitted operations: {sorted(operations - _OPERATIONS)}")
    if operations.intersection({"hessian", "hvp"}) and "energy_forces" not in operations:
        raise ValueError("Hessian/HVP admission requires same-scalar energy/force admission.")


@dataclass(frozen=True, slots=True)
class AnalyticDerivativeAdmission:
    profile_id: str
    evidence_id: str
    cell: ScalarCellIdentity
    operations: frozenset[AnalyticOperation]

    def __post_init__(self) -> None:
        _validate_admission_fields(self.profile_id, self.evidence_id, self.operations)
        if not isinstance(self.cell, ScalarCellIdentity):
            raise TypeError("Analytic derivative admission requires a scalar cell.")

    def require(self, point: ScalarPointIdentity, operation: AnalyticOperation) -> None:
        _validate_operation_name(operation)
        if not isinstance(point, ScalarPointIdentity) or point.cell != self.cell:
            raise AnalyticIdentityError(
                f"Analytic {self.profile_id!r} cell identity does not match the operation."
            )
        if operation not in self.operations:
            raise AnalyticCapabilityError(
                f"Analytic profile {self.profile_id!r} does not admit {operation}."
            )


@dataclass(frozen=True, slots=True)
class AnalyticCompositionAdmission:
    profile_id: str
    evidence_id: str
    gas_admission: AnalyticDerivativeAdmission
    solvent_admission: AnalyticDerivativeAdmission
    operations: frozenset[AnalyticOperation]

    def __post_init__(self) -> None:
        _validate_admission_fields(self.profile_id, self.evidence_id, self.operations)
        if self.gas_admission.cell.component != "gas":
            raise AnalyticIdentityError("Composition admission gas cell is not gas.")
        if self.solvent_admission.cell.component != "solvent":
            raise AnalyticIdentityError("Composition admission solvent cell is not solvent.")
        common = self.gas_admission.operations & self.solvent_admission.operations
        if not self.operations.issubset(common):
            raise AnalyticCapabilityError(
                "Composition operations exceed a component admission."
            )

    def require(
        self,
        gas_point: ScalarPointIdentity,
        solvent_point: ScalarPointIdentity,
        operation: AnalyticOperation,
    ) -> None:
        _validate_operation_name(operation)
        try:
            self.gas_admission.require(gas_point, operation)
        except AnalyticIdentityError as exc:
            raise AnalyticIdentityError(
                "Composite gas cell identity does not match its admission."
            ) from exc
        try:
            self.solvent_admission.require(solvent_point, operation)
        except AnalyticIdentityError as exc:
            raise AnalyticIdentityError(
                "Composite solvent cell identity does not match its admission."
            ) from exc
        if operation not in self.operations:
            raise AnalyticCapabilityError(
                f"Composite profile {self.profile_id!r} does not admit {operation}."
            )
        CompositePointIdentity(gas_point, solvent_point)


def _require_gas_operation(
    result: AnalyticOperationResult,
    operation: AnalyticOperation,
) -> ScalarPointIdentity:
    if result.operation != operation:
        raise ValueError(
            f"Expected a {operation!r} gas operation, received {result.operation!r}."
        )
    if not isinstance(result.identity, ScalarPointIdentity):
        raise AnalyticIdentityError(
            "Expected a gas component operation; an already composed result cannot be "
            "used as gas input."
        )
    if result.identity.cell.component != "gas":
        raise AnalyticIdentityError("Operation identity is not a gas component.")
    return result.identity


def _require_solvent_point(point: ScalarPointIdentity) -> None:
    if not isinstance(point, ScalarPointIdentity) or point.cell.component != "solvent":
        raise AnalyticIdentityError("Solvent result requires a solvent scalar identity.")


def compose_energy_forces(
    gas: AnalyticOperationResult,
    solvent,
    solvent_identity: ScalarPointIdentity,
) -> AnalyticOperationResult:
    gas_identity = _require_gas_operation(gas, "energy_forces")
    _require_solvent_point(solvent_identity)
    identity = CompositePointIdentity(gas_identity, solvent_identity)
    solvent_energy = _finite_real_scalar(
        solvent.energy_hartree,
        name="solvent energy_hartree",
    )
    gas_forces = gas.forces_hartree_per_angstrom
    solvent_forces = solvent.forces_hartree_per_angstrom
    if gas_forces is None:
        if solvent_forces is not None:
            raise ValueError("Energy-only composition received eager solvent forces.")
        forces = None
    else:
        if solvent_forces is None:
            raise AnalyticCapabilityError(
                "The solvent operation omitted forces required by gas composition."
            )
        solvent_forces = _read_only_numeric_array(
            solvent_forces,
            name="solvent forces_hartree_per_angstrom",
        )
        if solvent_forces.shape != (gas_identity.atom_count, 3):
            raise ValueError("Solvent forces must have shape (N, 3).")
        forces = np.asarray(gas_forces) + solvent_forces
    return AnalyticOperationResult.energy_forces(
        identity=identity,
        energy_hartree=float(gas.energy_hartree) + solvent_energy,
        forces_hartree_per_angstrom=forces,
    )


def compose_hessian(
    gas: AnalyticOperationResult,
    solvent_hessian_hartree_per_angstrom2: Any,
    solvent_identity: ScalarPointIdentity,
) -> AnalyticOperationResult:
    gas_identity = _require_gas_operation(gas, "hessian")
    _require_solvent_point(solvent_identity)
    identity = CompositePointIdentity(gas_identity, solvent_identity)
    solvent_hessian = _read_only_numeric_array(
        solvent_hessian_hartree_per_angstrom2,
        name="solvent_hessian_hartree_per_angstrom2",
    )
    expected = (3 * identity.atom_count, 3 * identity.atom_count)
    if solvent_hessian.shape != expected:
        raise ValueError("Solvent Hessian must have shape (3N, 3N).")
    return AnalyticOperationResult.hessian(
        identity=identity,
        hessian_hartree_per_angstrom2=(
            np.asarray(gas.hessian_hartree_per_angstrom2) + solvent_hessian
        ),
    )


def compose_hvp(
    gas: AnalyticOperationResult,
    solvent,
    solvent_identity: ScalarPointIdentity,
) -> AnalyticOperationResult:
    gas_identity = _require_gas_operation(gas, "hvp")
    _require_solvent_point(solvent_identity)
    identity = CompositePointIdentity(gas_identity, solvent_identity)
    solvent_hvp = _read_only_numeric_array(
        solvent.hvp_hartree_per_angstrom2,
        name="solvent hvp_hartree_per_angstrom2",
    )
    solvent_forces = _read_only_numeric_array(
        solvent.forces_hartree_per_angstrom,
        name="solvent forces_hartree_per_angstrom",
    )
    if solvent_hvp.shape != (3 * identity.atom_count,):
        raise ValueError("Solvent HVP must have shape (3N,).")
    if solvent_forces.shape != (identity.atom_count, 3):
        raise ValueError("Solvent forces must have shape (N, 3).")
    solvent_energy = _finite_real_scalar(
        solvent.energy_hartree,
        name="solvent energy_hartree",
    )
    return AnalyticOperationResult.hvp(
        identity=identity,
        direction=gas.direction,
        hvp_hartree_per_angstrom2=(
            np.asarray(gas.hvp_hartree_per_angstrom2)
            + solvent_hvp
        ),
        forces_hartree_per_angstrom=(
            np.asarray(gas.forces_hartree_per_angstrom)
            + solvent_forces
        ),
        energy_hartree=float(gas.energy_hartree) + solvent_energy,
    )


def identity_record(
    identity: ScalarPointIdentity | CompositePointIdentity,
) -> dict[str, Any]:
    """Return a detached JSON-compatible audit record for an immutable identity."""
    return asdict(identity)


__all__ = [
    "AnalyticCapabilityError",
    "AnalyticCompositionAdmission",
    "AnalyticDerivativeAdmission",
    "AnalyticIdentityError",
    "AnalyticOperationResult",
    "CheckpointFileIdentity",
    "CompositePointIdentity",
    "NEIGHBOR_CUTOFF_GUARD_MARGIN_ANGSTROM",
    "NEIGHBOR_CUTOFF_GUARD_POLICY",
    "ScalarCellIdentity",
    "ScalarPointIdentity",
    "compose_energy_forces",
    "compose_hessian",
    "compose_hvp",
    "file_stat_identity",
    "freeze_identity_options",
    "geometry_sha256",
    "identity_record",
    "named_array_sha256",
    "require_no_amp_tf32",
    "require_derivative_cutoff_guard",
    "torch_response_settings",
    "torch_model_runtime_identity",
]
