"""Geometry-mediated AIMNet2 adapter for the Route-2 model contracts.

The adapter reuses MAPLE's existing AIMNet2 calculator APIs.  It embeds the
checkpoint's neural-charge-equilibration monopoles as ``[q, 0, 0, 0]`` in the
authoritative atomic ``l<=1`` source space and exposes the model-native
coordinate VJP of those charges.  The source is deliberately independent of
the continuum field: this closes ``R -> q(R) -> PCM`` geometry response, not
fixed-geometry electronic mutual polarization.

No checkpoint bytes are distributed here.  The default contract binds the
local asset used by the historical MAPLE canaries by SHA256 while recording
that its original upstream release identity has not yet been recovered.
Consequently this provider is diagnostic and admits no public capability.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

from maple.solvation.api.profiles import (
    AIMNET2_FROZEN_CHARGE_MODEL_PROFILE_ID,
    AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
    AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
)
from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)
from maple.solvation.coupling.state_equation import provider_behavior_sha256

from .base import (
    ElectronicSourceState,
    ModelDomain,
    ModelProvenance,
    VacuumState,
    array_sha256,
    atom_count,
    model_charge_and_multiplicity,
    model_input_sha256,
)

AIMNET2_WB97M_D3_CHECKPOINT_SHA256 = (
    "85ba59d8c78eb4d3185f6b1614df79706427f7ca72f53f2d90e365a1723d953d"
)
AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES = 11_613_188


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(calculator: object) -> tuple[tuple[str, str], ...]:
    paths: dict[str, Path] = {"adapter": Path(__file__).resolve()}
    seen = {Path(__file__).resolve()}
    for index, cls in enumerate(type(calculator).__mro__):
        try:
            raw = inspect.getsourcefile(cls)
        except (TypeError, OSError):
            raw = None
        if raw is None:
            continue
        path = Path(raw).resolve()
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        paths[f"calculator_mro_{index}_{cls.__name__}"] = path
    if len(paths) == 1:
        raise RuntimeError("AIMNet2 calculator source is unavailable for provenance.")
    return tuple((label, _sha256_file(path)) for label, path in sorted(paths.items()))


def _runtime_provenance_sha256(calculator: object) -> str | None:
    provider = getattr(calculator, "runtime_provenance", None)
    if provider is None:
        return None
    if not callable(provider):
        raise TypeError("AIMNet2 runtime_provenance must be callable when present.")
    payload = provider()
    if not isinstance(payload, dict):
        raise TypeError("AIMNet2 runtime_provenance() must return a dictionary.")
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "AIMNet2 runtime provenance must be finite JSON-compatible metadata."
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


def _calculator_behavior_methods(calculator: object) -> tuple[str, ...]:
    methods = ("charge_state", "charge_position_response")
    if callable(getattr(calculator, "charge_position_second_order", None)):
        methods += ("charge_position_second_order",)
    return methods


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AIMNet2NeighborTopologyState:
    """Rigid-motion-invariant identity of AIMNet2's hard neighbor graphs."""

    configuration_sha256: str
    topology_sha256: str
    active_pairs_by_graph: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    minimum_cutoff_margin_angstrom: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "configuration_sha256": self.configuration_sha256,
            "topology_sha256": self.topology_sha256,
            "active_pairs_by_graph": {
                name: [list(pair) for pair in pairs]
                for name, pairs in self.active_pairs_by_graph
            },
            "minimum_cutoff_margin_angstrom": self.minimum_cutoff_margin_angstrom,
        }


@dataclass(frozen=True, slots=True)
class AIMNet2GeometryMediatedSecondOrder:
    """Second-order source ledger for the point-l0 AIMNet2 adapter.

    ``source_position_jvp`` is ``J_c h``.  The contracted source Hessian holds
    the supplied source cotangent fixed and is therefore
    ``D_R[J_c.T v][h]``.  Only the l0 channel may be nonzero because this
    adapter does not reinterpret AIMNet2 point charges as atomic dipoles.
    """

    source: np.ndarray
    source_cotangent: np.ndarray
    coordinate_direction: np.ndarray
    intrinsic_energy_gradient_eV_per_A: np.ndarray
    source_position_vjp_eV_per_A: np.ndarray
    source_position_jvp: np.ndarray
    intrinsic_energy_hvp_eV_per_A2: np.ndarray
    contracted_source_hessian_eV_per_A2: np.ndarray
    standard_decomposed_energy_absolute_error_eV: float
    standard_decomposed_charge_max_absolute_error_e: float
    standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A: float
    standard_decomposed_charge_vjp_max_absolute_error_eV_per_A: float
    charge_tangent_residual_e_per_A: float

    def __post_init__(self) -> None:
        raw_source = np.asarray(self.source, dtype=float)
        if raw_source.ndim != 2 or raw_source.shape[1] != 4:
            raise ValueError("source must have shape (N,4).")
        count = raw_source.shape[0]
        arrays = {
            "source": ((count, 4), raw_source),
            "source_cotangent": ((count, 4), self.source_cotangent),
            "coordinate_direction": ((count, 3), self.coordinate_direction),
            "intrinsic_energy_gradient_eV_per_A": (
                (count, 3),
                self.intrinsic_energy_gradient_eV_per_A,
            ),
            "source_position_vjp_eV_per_A": (
                (count, 3),
                self.source_position_vjp_eV_per_A,
            ),
            "source_position_jvp": ((count, 4), self.source_position_jvp),
            "intrinsic_energy_hvp_eV_per_A2": (
                (count, 3),
                self.intrinsic_energy_hvp_eV_per_A2,
            ),
            "contracted_source_hessian_eV_per_A2": (
                (count, 3),
                self.contracted_source_hessian_eV_per_A2,
            ),
        }
        frozen = {
            name: _readonly(values, shape=shape, name=name)
            for name, (shape, values) in arrays.items()
        }
        if not np.array_equal(frozen["source"][:, 1:], np.zeros((count, 3))):
            raise ValueError("AIMNet2 second-order source must be exactly point-l0.")
        if not np.array_equal(
            frozen["source_position_jvp"][:, 1:], np.zeros((count, 3))
        ):
            raise ValueError("AIMNet2 source-position JVP must be exactly point-l0.")
        for name, values in frozen.items():
            object.__setattr__(self, name, values)
        for name in (
            "standard_decomposed_energy_absolute_error_eV",
            "standard_decomposed_charge_max_absolute_error_e",
            "standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A",
            "standard_decomposed_charge_vjp_max_absolute_error_eV_per_A",
            "charge_tangent_residual_e_per_A",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if abs(float(np.sum(frozen["source_position_jvp"][:, 0]))) > 1.0e-10:
            raise ValueError(
                "AIMNet2 source-position JVP violates total-charge tangency."
            )


@dataclass(frozen=True, slots=True)
class AIMNet2CheckpointContract:
    """Content-addressed local-checkpoint and inference boundary."""

    provider_id: str
    model_profile_id: str
    checkpoint_identifier: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    model_name: str
    coulomb_method: str
    inference_dtype: str
    supported_atomic_numbers: tuple[int, ...]
    upstream_repository: str
    publication_doi: str
    upstream_version: str
    upstream_commit: str
    checkpoint_origin_status: str

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_profile_id",
            "checkpoint_identifier",
            "model_name",
            "coulomb_method",
            "inference_dtype",
            "upstream_repository",
            "publication_doi",
            "upstream_version",
            "upstream_commit",
            "checkpoint_origin_status",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, name="checkpoint_sha256"),
        )
        if (
            isinstance(self.checkpoint_size_bytes, bool)
            or not isinstance(self.checkpoint_size_bytes, int)
            or self.checkpoint_size_bytes < 1
        ):
            raise ValueError("checkpoint_size_bytes must be a positive integer.")
        numbers = tuple(self.supported_atomic_numbers)
        if (
            not numbers
            or tuple(sorted(set(numbers))) != numbers
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in numbers
            )
        ):
            raise ValueError(
                "supported_atomic_numbers must be unique sorted positive integers."
            )
        object.__setattr__(self, "supported_atomic_numbers", numbers)

    def metadata(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_profile_id": self.model_profile_id,
            "checkpoint_identifier": self.checkpoint_identifier,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "model_name": self.model_name,
            "coulomb_method": self.coulomb_method,
            "inference_dtype": self.inference_dtype,
            "supported_atomic_numbers": list(self.supported_atomic_numbers),
            "upstream_repository": self.upstream_repository,
            "publication_doi": self.publication_doi,
            "upstream_version": self.upstream_version,
            "upstream_commit": self.upstream_commit,
            "checkpoint_origin_status": self.checkpoint_origin_status,
        }

    @property
    def sha256(self) -> str:
        return _hash(self.metadata())


AIMNET2_WB97M_D3_LOCAL_CHECKPOINT_CONTRACT = AIMNet2CheckpointContract(
    provider_id="maple.route2.model.aimnet2-geometry-mediated.impl.v1",
    model_profile_id=AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
    checkpoint_identifier="aimnet2-wb97m-d3-local-sha256-bound",
    checkpoint_sha256=AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
    checkpoint_size_bytes=AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES,
    model_name="aimnet2",
    coulomb_method="simple",
    inference_dtype="float32",
    # This is the subset exercised by the retained MAPLE AIMNet2/ddPCM panels,
    # not a claim about the full published AIMNet2 chemical domain.
    supported_atomic_numbers=(1, 6, 7, 8),
    upstream_repository="https://github.com/isayevlab/aimnetcentral",
    publication_doi="10.1039/D4SC08572H",
    upstream_version="AIMNet2 wB97M-D3 local TorchScript asset",
    upstream_commit="unresolved-local-checkpoint-origin",
    checkpoint_origin_status="hash-bound-local-asset-upstream-release-origin-unresolved",
)

AIMNET2_WB97M_D3_RECONSTRUCTED_FLOAT64_CONTRACT = AIMNet2CheckpointContract(
    provider_id="maple.route2.model.aimnet2-geometry-mediated-float64.impl.v1",
    model_profile_id=AIMNET2_GEOMETRY_MEDIATED_MODEL_PROFILE_ID,
    checkpoint_identifier="aimnet2-wb97m-d3-local-sha256-bound",
    checkpoint_sha256=AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
    checkpoint_size_bytes=AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES,
    model_name="aimnet2",
    coulomb_method="simple",
    inference_dtype="float64",
    supported_atomic_numbers=(1, 6, 7, 8),
    upstream_repository="https://github.com/isayevlab/aimnetcentral",
    publication_doi="10.1039/D4SC08572H",
    upstream_version="aimnet==0.2.0 Python reconstruction of legacy wB97M-D3",
    upstream_commit="unresolved-upstream-commit-runtime-files-sha256-bound",
    checkpoint_origin_status=(
        "hash-bound-local-asset-upstream-release-origin-unresolved"
    ),
)

AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT = AIMNet2CheckpointContract(
    provider_id="maple.route2.model.aimnet2-frozen-charge-water-float64.impl.v1",
    model_profile_id=AIMNET2_FROZEN_CHARGE_MODEL_PROFILE_ID,
    checkpoint_identifier="aimnet2-wb97m-d3-local-sha256-bound",
    checkpoint_sha256=AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
    checkpoint_size_bytes=AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES,
    model_name="aimnet2",
    coulomb_method="simple",
    inference_dtype="float64",
    supported_atomic_numbers=(1, 6, 7, 8),
    upstream_repository="https://github.com/isayevlab/aimnetcentral",
    publication_doi="10.1039/D4SC08572H",
    upstream_version="aimnet==0.2.0 Python reconstruction of legacy wB97M-D3",
    upstream_commit="unresolved-upstream-commit-runtime-files-sha256-bound",
    checkpoint_origin_status=(
        "hash-bound-local-asset-upstream-release-origin-unresolved"
    ),
)

AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT = (
    AIMNet2CheckpointContract(
        provider_id=(
            "maple.route2.model.aimnet2-frozen-charge-multisolvent-float64.impl.v1"
        ),
        model_profile_id=AIMNET2_FROZEN_CHARGE_MODEL_PROFILE_ID,
        checkpoint_identifier="aimnet2-wb97m-d3-local-sha256-bound",
        checkpoint_sha256=AIMNET2_WB97M_D3_CHECKPOINT_SHA256,
        checkpoint_size_bytes=AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES,
        model_name="aimnet2",
        coulomb_method="simple",
        inference_dtype="float64",
        supported_atomic_numbers=(1, 6, 7, 8),
        upstream_repository="https://github.com/isayevlab/aimnetcentral",
        publication_doi="10.1039/D4SC08572H",
        upstream_version="aimnet==0.2.0 Python reconstruction of legacy wB97M-D3",
        upstream_commit="unresolved-upstream-commit-runtime-files-sha256-bound",
        checkpoint_origin_status=(
            "hash-bound-local-asset-upstream-release-origin-unresolved"
        ),
    )
)


class AIMNet2GeometryMediatedModelAdapter:
    """Vacuum/source/coordinate-response adapter with zero field response."""

    __slots__ = (
        "_calculator",
        "_calculator_runtime_provenance_sha256",
        "_checkpoint_path",
        "_checkpoint_stat",
        "_configuration_sha256",
        "_contract",
        "_neighbor_cutoffs_angstrom",
        "_raw_long_range_cutoff_angstrom",
        "_sealed",
        "coordinate_frame_policy",
        "coupling_id",
        "device",
        "domain",
        "dtype",
        "field_convention",
        "model_profile_id",
        "provenance",
        "provenance_sha256",
        "provider_id",
    )

    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE
    field_independent = True
    electronic_mutual_polarization = False
    variational_functional_admitted = False

    def __init__(
        self,
        calculator: object,
        contract: AIMNet2CheckpointContract = (
            AIMNET2_WB97M_D3_LOCAL_CHECKPOINT_CONTRACT
        ),
    ) -> None:
        if not isinstance(contract, AIMNet2CheckpointContract):
            raise TypeError("contract must be AIMNet2CheckpointContract.")
        for name in ("charge_state", "charge_position_response"):
            if not callable(getattr(calculator, name, None)):
                raise TypeError(f"AIMNet2 calculator must expose callable {name}().")
        if getattr(calculator, "model_name", None) != contract.model_name:
            raise ValueError(
                "AIMNet2 calculator model name does not match the contract."
            )
        if getattr(calculator, "_supports_atom_charges", None) is not True:
            raise ValueError(
                "AIMNet2 calculator does not expose admitted point charges."
            )
        if getattr(calculator, "_coulomb_method", None) != contract.coulomb_method:
            raise ValueError("AIMNet2 Coulomb method does not match the contract.")
        if getattr(calculator, "solvent_correction", None) is not None:
            raise ValueError(
                "AIMNet2 geometry-mediated vacuum/source inference must not include "
                "a calculator-internal solvent correction."
            )
        calculator_dtype = str(getattr(calculator, "inference_dtype", "float32"))
        if calculator_dtype != contract.inference_dtype:
            raise ValueError(
                "AIMNet2 calculator inference dtype does not match the contract."
            )
        short_cutoff = float(getattr(calculator, "cutoff", np.nan))
        raw_long_cutoff = float(getattr(calculator, "cutoff_lr", np.nan))
        if not math.isfinite(short_cutoff) or short_cutoff <= 0.0:
            raise ValueError("AIMNet2 short-range cutoff must be finite and positive.")
        if math.isnan(raw_long_cutoff) or raw_long_cutoff <= 0.0:
            raise ValueError(
                "AIMNet2 long-range cutoff must be positive or positive infinity."
            )
        effective_long_cutoff = (
            short_cutoff if math.isinf(raw_long_cutoff) else raw_long_cutoff
        )
        checkpoint = Path(str(getattr(calculator, "model_path", ""))).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("AIMNet2 checkpoint path is unavailable.")
        stat = checkpoint.stat()
        if (
            stat.st_size != contract.checkpoint_size_bytes
            or _sha256_file(checkpoint) != contract.checkpoint_sha256
        ):
            raise ValueError("AIMNet2 checkpoint bytes do not match the contract.")
        device = _text(str(getattr(calculator, "device", "")), name="device")
        domain = ModelDomain(contract.supported_atomic_numbers, (0, 0), (1,))
        files = _source_files(calculator)
        runtime_provenance_sha256 = _runtime_provenance_sha256(calculator)
        provenance = ModelProvenance(
            provider_id=contract.provider_id,
            model_profile_id=contract.model_profile_id,
            model_family="AIMNet2-NQE-geometry-mediated-point-l0",
            checkpoint_sha256=contract.checkpoint_sha256,
            upstream_version=contract.upstream_version,
            upstream_commit=contract.upstream_commit,
            inference_code_sha256=_hash(
                {
                    "source_files_sha256": files,
                    "runtime_provenance_sha256": runtime_provenance_sha256,
                }
            ),
            dtype=contract.inference_dtype,
            device=device,
            domain=domain,
            field_convention=self.field_space.field_convention,
            coordinate_frame_policy=(
                "laboratory Cartesian Angstrom input; nonperiodic molecule; fixed "
                "atom identity/order; source depends on geometry and declared total "
                "charge but not on continuum field"
            ),
        )
        object.__setattr__(self, "_calculator", calculator)
        object.__setattr__(
            self,
            "_calculator_runtime_provenance_sha256",
            runtime_provenance_sha256,
        )
        object.__setattr__(self, "_contract", contract)
        object.__setattr__(
            self,
            "_neighbor_cutoffs_angstrom",
            (
                ("short_range", short_cutoff),
                ("long_range_effective", effective_long_cutoff),
            ),
        )
        object.__setattr__(self, "_raw_long_range_cutoff_angstrom", raw_long_cutoff)
        object.__setattr__(self, "_checkpoint_path", checkpoint)
        object.__setattr__(
            self,
            "_checkpoint_stat",
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
        )
        object.__setattr__(self, "provider_id", provenance.provider_id)
        object.__setattr__(self, "model_profile_id", provenance.model_profile_id)
        object.__setattr__(
            self,
            "coupling_id",
            AIMNET2_POINT_L0_GEOMETRY_MEDIATED_COUPLING_ID,
        )
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "provenance_sha256", provenance.sha256)
        object.__setattr__(self, "dtype", provenance.dtype)
        object.__setattr__(self, "device", provenance.device)
        object.__setattr__(self, "domain", provenance.domain)
        object.__setattr__(self, "field_convention", provenance.field_convention)
        object.__setattr__(
            self, "coordinate_frame_policy", provenance.coordinate_frame_policy
        )
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("AIMNet2GeometryMediatedModelAdapter is immutable.")
        object.__setattr__(self, name, value)

    @property
    def checkpoint_contract(self) -> AIMNet2CheckpointContract:
        return self._contract

    def _current_configuration_sha256(self) -> str:
        stat = self._checkpoint_path.stat()
        if (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        ) != self._checkpoint_stat:
            raise RuntimeError(
                "AIMNet2 checkpoint file identity changed after construction."
            )
        return _hash(
            {
                "schema": "route2-aimnet2-geometry-mediated-model-adapter-v3",
                "contract": self._contract.metadata(),
                "source_files_sha256": _source_files(self._calculator),
                "runtime_provenance_sha256": _runtime_provenance_sha256(
                    self._calculator
                ),
                "calculator_behavior_sha256": provider_behavior_sha256(
                    self._calculator,
                    _calculator_behavior_methods(self._calculator),
                    label="aimnet2_calculator",
                ),
                "calculator_second_order_response_available": callable(
                    getattr(self._calculator, "charge_position_second_order", None)
                ),
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "provenance_sha256": self.provenance_sha256,
                "device": str(getattr(self._calculator, "device", "")),
                "calculator_inference_dtype": str(
                    getattr(self._calculator, "inference_dtype", "float32")
                ),
                "model_name": getattr(self._calculator, "model_name", None),
                "coulomb_method": getattr(self._calculator, "_coulomb_method", None),
                "neighbor_cutoffs_angstrom": {
                    name: cutoff for name, cutoff in self._neighbor_cutoffs_angstrom
                },
                "raw_long_range_cutoff_angstrom": (
                    "infinity"
                    if math.isinf(self._raw_long_range_cutoff_angstrom)
                    else self._raw_long_range_cutoff_angstrom
                ),
                "solvent_correction_absent": (
                    getattr(self._calculator, "solvent_correction", None) is None
                ),
                "source_space_sha256": self.source_space.metadata_hash(),
                "field_space_sha256": self.field_space.metadata_hash(),
                "field_independent": self.field_independent,
                "electronic_mutual_polarization": self.electronic_mutual_polarization,
            }
        )

    def configuration_sha256(self) -> str:
        if (
            _runtime_provenance_sha256(self._calculator)
            != self._calculator_runtime_provenance_sha256
        ):
            raise ValueError("AIMNet2 runtime provenance drifted.")
        current_short = float(getattr(self._calculator, "cutoff", np.nan))
        current_long = float(getattr(self._calculator, "cutoff_lr", np.nan))
        if current_short != self._neighbor_cutoffs_angstrom[0][1] or (
            math.isinf(current_long) != math.isinf(self._raw_long_range_cutoff_angstrom)
            or (
                not math.isinf(current_long)
                and current_long != self._raw_long_range_cutoff_angstrom
            )
        ):
            raise ValueError("AIMNet2 neighbor-cutoff configuration drifted.")
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise ValueError(
                "AIMNet2 adapter configuration drifted after construction."
            )
        return current

    def calculator_runtime_provenance(self) -> dict[str, object]:
        """Return the validated external calculator-runtime provenance payload."""

        self.configuration_sha256()
        provider = getattr(self._calculator, "runtime_provenance", None)
        if not callable(provider):
            raise NotImplementedError(
                "This AIMNet2 calculator exposes no runtime provenance payload."
            )
        payload = provider()
        if not isinstance(payload, dict):
            raise TypeError("AIMNet2 runtime_provenance() must return a dictionary.")
        if _runtime_provenance_sha256(self._calculator) != (
            self._calculator_runtime_provenance_sha256
        ):
            raise ValueError("AIMNet2 runtime provenance drifted.")
        return copy.deepcopy(payload)

    def neighbor_topology(self, atoms: object) -> AIMNet2NeighborTopologyState:
        """Return hard neighbor membership and distance from every cutoff event.

        The dense AIMNet2 wrapper uses ``distance <= cutoff``.  The hash binds
        pair membership for both short- and effective long-range graphs but is
        invariant to rigid translations and rotations.  A validation harness
        must additionally require a positive margin and identical hashes over
        every finite-difference or optimizer trial stencil.
        """

        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        getter = getattr(atoms, "get_positions", None)
        if not callable(getter):
            raise TypeError("geometry must expose callable get_positions().")
        positions = np.asarray(getter(), dtype=float)
        count = atom_count(atoms)
        if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("geometry positions must be finite with shape (N,3).")
        active: list[tuple[str, tuple[tuple[int, int], ...]]] = []
        margins: list[float] = []
        for name, cutoff in self._neighbor_cutoffs_angstrom:
            pairs: list[tuple[int, int]] = []
            for first in range(count):
                for second in range(first + 1, count):
                    distance = float(
                        np.linalg.norm(positions[first] - positions[second])
                    )
                    if distance <= cutoff:
                        pairs.append((first, second))
                    margins.append(abs(distance - cutoff))
            active.append((name, tuple(pairs)))
        payload = {
            "schema": "route2-aimnet2-hard-neighbor-topology-v1",
            "configuration_sha256": self.configuration_sha256(),
            "atomic_numbers": [int(value) for value in atoms.get_atomic_numbers()],
            "active_pairs_by_graph": {
                name: [list(pair) for pair in pairs] for name, pairs in active
            },
        }
        return AIMNet2NeighborTopologyState(
            configuration_sha256=self.configuration_sha256(),
            topology_sha256=_hash(payload),
            active_pairs_by_graph=tuple(active),
            minimum_cutoff_margin_angstrom=min(margins) if margins else None,
        )

    def last_source_response_parity(self) -> dict[str, float]:
        """Return the source runtime's most recent first-order parity ledger."""

        self.configuration_sha256()
        provider = getattr(self._calculator, "last_ordinary_decomposed_parity", None)
        if not callable(provider):
            raise NotImplementedError(
                "This AIMNet2 source runtime exposes no first-order parity ledger."
            )
        raw = provider()
        if not isinstance(raw, dict) or not raw:
            raise ValueError("AIMNet2 source-response parity ledger is malformed.")
        result: dict[str, float] = {}
        for name, value in raw.items():
            number = float(value)
            if (
                not isinstance(name, str)
                or not name
                or not math.isfinite(number)
                or number < 0.0
            ):
                raise ValueError(
                    "AIMNet2 source-response parity entries must be named, "
                    "finite, and non-negative."
                )
            result[name] = number
        return result

    def _charge_state(self, atoms: object) -> object:
        self.configuration_sha256()
        charge, _ = self.domain.validate_atoms(atoms)
        state = self._calculator.charge_state(atoms)
        self._validate_charge_state(state, atoms, charge)
        return state

    def _validate_charge_state(
        self, state: object, atoms: object, charge: int
    ) -> np.ndarray:
        count = atom_count(atoms)
        values = np.asarray(getattr(state, "charges_e", None), dtype=float)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "AIMNet2 charge state must contain one finite charge per atom."
            )
        if not np.isfinite(float(getattr(state, "energy_ev", np.nan))):
            raise ValueError("AIMNet2 charge-state energy must be finite.")
        if float(getattr(state, "requested_total_charge_e", np.nan)) != float(charge):
            raise ValueError(
                "AIMNet2 charge state does not match the declared total charge."
            )
        if abs(float(np.sum(values)) - float(charge)) > 1.0e-10:
            raise ValueError("AIMNet2 projected charges do not conserve total charge.")
        if getattr(state, "model_name", None) != self._contract.model_name:
            raise ValueError("AIMNet2 charge-state model identity drifted.")
        return np.array(values, copy=True)

    def _response(self, atoms: object, charge_cotangent: np.ndarray) -> object:
        self.configuration_sha256()
        charge, _ = self.domain.validate_atoms(atoms)
        count = atom_count(atoms)
        cotangent = np.asarray(charge_cotangent, dtype=float)
        if cotangent.shape != (count,) or not np.all(np.isfinite(cotangent)):
            raise ValueError("AIMNet2 charge cotangent must be finite with shape (N,).")
        response = self._calculator.charge_position_response(atoms, cotangent.copy())
        self._validate_charge_state(
            getattr(response, "charge_state", None), atoms, charge
        )
        stored = np.asarray(
            getattr(response, "charge_cotangent_ev_per_e", None), dtype=float
        )
        if stored.shape != (count,) or not np.allclose(
            stored, cotangent, rtol=0.0, atol=0.0
        ):
            raise ValueError("AIMNet2 response changed the supplied charge cotangent.")
        for name in (
            "intrinsic_energy_gradient_ev_per_angstrom",
            "charge_position_vjp_ev_per_angstrom",
        ):
            values = np.asarray(getattr(response, name, None), dtype=float)
            if values.shape != (count, 3) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"AIMNet2 response {name} must be finite with shape (N,3)."
                )
        return response

    def _second_order_response(
        self,
        atoms: object,
        charge_cotangent: np.ndarray,
        coordinate_direction: np.ndarray,
    ) -> object:
        self.configuration_sha256()
        charge, _ = self.domain.validate_atoms(atoms)
        count = atom_count(atoms)
        cotangent = np.asarray(charge_cotangent, dtype=float)
        direction = np.asarray(coordinate_direction, dtype=float)
        if cotangent.shape != (count,) or not np.all(np.isfinite(cotangent)):
            raise ValueError("AIMNet2 charge cotangent must be finite with shape (N,).")
        if direction.shape != (count, 3) or not np.all(np.isfinite(direction)):
            raise ValueError(
                "AIMNet2 coordinate direction must be finite with shape (N,3)."
            )
        provider = getattr(self._calculator, "charge_position_second_order", None)
        if not callable(provider):
            raise NotImplementedError(
                "This AIMNet2 runtime has no source-bound second-order response."
            )
        response = provider(atoms, cotangent.copy(), direction.copy())
        self._validate_charge_state(
            getattr(response, "charge_state", None), atoms, charge
        )
        stored_cotangent = np.asarray(
            getattr(response, "charge_cotangent_ev_per_e", None), dtype=float
        )
        stored_direction = np.asarray(
            getattr(response, "coordinate_direction", None), dtype=float
        )
        if stored_cotangent.shape != (count,) or not np.array_equal(
            stored_cotangent, cotangent
        ):
            raise ValueError(
                "AIMNet2 second-order response changed the supplied charge cotangent."
            )
        if stored_direction.shape != (count, 3) or not np.array_equal(
            stored_direction, direction
        ):
            raise ValueError(
                "AIMNet2 second-order response changed the coordinate direction."
            )
        array_shapes = {
            "intrinsic_energy_gradient_ev_per_angstrom": (count, 3),
            "charge_position_vjp_ev_per_angstrom": (count, 3),
            "charge_position_jvp_e_per_angstrom": (count,),
            "intrinsic_energy_hvp_ev_per_angstrom2": (count, 3),
            "contracted_charge_hessian_ev_per_angstrom2": (count, 3),
        }
        for name, shape in array_shapes.items():
            values = np.asarray(getattr(response, name, None), dtype=float)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"AIMNet2 second-order response {name} must be finite "
                    f"with shape {shape}."
                )
        for name in (
            "standard_decomposed_energy_absolute_error_ev",
            "standard_decomposed_charge_max_absolute_error_e",
            "standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom",
            "standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom",
            "charge_tangent_residual_e_per_angstrom",
        ):
            value = float(getattr(response, name, np.nan))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"AIMNet2 second-order response {name} must be non-negative."
                )
        charge_jvp = np.asarray(
            response.charge_position_jvp_e_per_angstrom, dtype=float
        )
        if abs(float(np.sum(charge_jvp))) > 1.0e-10:
            raise ValueError("AIMNet2 second-order response violates charge tangency.")
        return response

    @staticmethod
    def _embed_charges(charges: np.ndarray) -> np.ndarray:
        source = np.zeros((charges.size, 4), dtype=float)
        source[:, 0] = charges
        return source

    def evaluate_vacuum(self, atoms: object, *, need_forces: bool) -> VacuumState:
        count = atom_count(atoms)
        if need_forces:
            response = self._response(atoms, np.zeros(count))
            charge_state = response.charge_state
            forces = -np.asarray(
                response.intrinsic_energy_gradient_ev_per_angstrom, dtype=float
            )
        else:
            charge_state = self._charge_state(atoms)
            forces = None
        return VacuumState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            count,
            float(charge_state.energy_ev),
            need_forces,
            forces,
        )

    def evaluate_source(
        self, atoms: object, field: object, *, need_fixed_field_forces: bool
    ) -> ElectronicSourceState:
        count = atom_count(atoms)
        values = self.field_space.validate(field, atom_count=count)
        if need_fixed_field_forces:
            response = self._response(atoms, np.zeros(count))
            charge_state = response.charge_state
            forces = -np.asarray(
                response.intrinsic_energy_gradient_ev_per_angstrom, dtype=float
            )
        else:
            charge_state = self._charge_state(atoms)
            forces = None
        source = self._embed_charges(
            self._validate_charge_state(
                charge_state, atoms, model_charge_and_multiplicity(atoms)[0]
            )
        )
        return ElectronicSourceState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            array_sha256(values, name="field"),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            count,
            source,
            need_fixed_field_forces,
            forces,
        )

    def source_jvp(
        self, atoms: object, field: object, field_direction: object
    ) -> np.ndarray:
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        count = atom_count(atoms)
        self.field_space.validate(field, atom_count=count)
        self.field_space.validate(
            field_direction, atom_count=count, name="field_direction"
        )
        return np.zeros(self.source_space.shape(count), dtype=float)

    def source_vjp(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        self.configuration_sha256()
        self.domain.validate_atoms(atoms)
        count = atom_count(atoms)
        self.field_space.validate(field, atom_count=count)
        self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        return np.zeros(self.field_space.shape(count), dtype=float)

    def source_position_vjp(
        self, atoms: object, field: object, source_cotangent: object
    ) -> np.ndarray:
        count = atom_count(atoms)
        self.field_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        response = self._response(atoms, cotangent[:, 0])
        return np.asarray(
            response.charge_position_vjp_ev_per_angstrom, dtype=float
        ).copy()

    def source_position_second_order(
        self,
        atoms: object,
        field: object,
        source_cotangent: object,
        coordinate_direction: object,
    ) -> AIMNet2GeometryMediatedSecondOrder:
        """Return ``J_c h``, ``H_E h``, and fixed-cotangent ``D(J_c.T v)h``."""

        count = atom_count(atoms)
        self.field_space.validate(field, atom_count=count)
        cotangent = self.source_space.validate(
            source_cotangent, atom_count=count, name="source_cotangent"
        )
        direction = _readonly(
            coordinate_direction,
            shape=(count, 3),
            name="coordinate_direction",
        )
        response = self._second_order_response(atoms, cotangent[:, 0], direction)
        charges = self._validate_charge_state(
            response.charge_state,
            atoms,
            model_charge_and_multiplicity(atoms)[0],
        )
        source_jvp = np.zeros((count, 4), dtype=float)
        source_jvp[:, 0] = np.asarray(
            response.charge_position_jvp_e_per_angstrom, dtype=float
        )
        return AIMNet2GeometryMediatedSecondOrder(
            source=self._embed_charges(charges),
            source_cotangent=np.array(cotangent, copy=True),
            coordinate_direction=np.array(direction, copy=True),
            intrinsic_energy_gradient_eV_per_A=(
                response.intrinsic_energy_gradient_ev_per_angstrom
            ),
            source_position_vjp_eV_per_A=(response.charge_position_vjp_ev_per_angstrom),
            source_position_jvp=source_jvp,
            intrinsic_energy_hvp_eV_per_A2=(
                response.intrinsic_energy_hvp_ev_per_angstrom2
            ),
            contracted_source_hessian_eV_per_A2=(
                response.contracted_charge_hessian_ev_per_angstrom2
            ),
            standard_decomposed_energy_absolute_error_eV=(
                response.standard_decomposed_energy_absolute_error_ev
            ),
            standard_decomposed_charge_max_absolute_error_e=(
                response.standard_decomposed_charge_max_absolute_error_e
            ),
            standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A=(
                response.standard_decomposed_intrinsic_gradient_max_absolute_error_ev_per_angstrom
            ),
            standard_decomposed_charge_vjp_max_absolute_error_eV_per_A=(
                response.standard_decomposed_charge_vjp_max_absolute_error_ev_per_angstrom
            ),
            charge_tangent_residual_e_per_A=(
                response.charge_tangent_residual_e_per_angstrom
            ),
        )


__all__ = [
    "AIMNET2_WB97M_D3_FROZEN_CHARGE_MULTISOLVENT_FLOAT64_CONTRACT",
    "AIMNET2_WB97M_D3_CHECKPOINT_SHA256",
    "AIMNET2_WB97M_D3_CHECKPOINT_SIZE_BYTES",
    "AIMNET2_WB97M_D3_LOCAL_CHECKPOINT_CONTRACT",
    "AIMNET2_WB97M_D3_FROZEN_CHARGE_WATER_FLOAT64_CONTRACT",
    "AIMNET2_WB97M_D3_RECONSTRUCTED_FLOAT64_CONTRACT",
    "AIMNet2CheckpointContract",
    "AIMNet2GeometryMediatedSecondOrder",
    "AIMNet2GeometryMediatedModelAdapter",
    "AIMNet2NeighborTopologyState",
]
