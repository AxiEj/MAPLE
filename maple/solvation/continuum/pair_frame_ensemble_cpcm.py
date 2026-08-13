"""Rotationally equivariant ordered-pair-frame ensemble C-PCM candidate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np
from ase.data import atomic_numbers

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import (
    PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID,
)
from maple.solvation.api.scalar_registry import OPERATIONAL_CPCM_ELECTROSTATIC_V1
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.coupling.spaces import (
    MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE,
    MACE_POLAR_RADIAL_GTO_SOURCE_SPACE,
)

from .conjugate_fixed_topology_cpcm import (
    ConjugateRadialGTOFixedTopologyCPCMBackend,
    WATER_CPCM_DIELECTRIC,
)
from .fixed_topology_cpcm import CONTINUUM_PROFILE_ID
from .pair_frame_geometry import (
    PAIR_FRAME_POLICY,
    PairFrame,
    frame_activity_sha256,
    frame_data,
    frame_topology_sha256,
    geometry_sha256,
    pair_frame_vjp,
)
from .pair_frame_state import (
    PAIR_FRAME_CPCM_COUPLING_ID,
    PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM,
    PAIR_FRAME_CPCM_PROVIDER_ID,
    PairFrameCPCMState,
    pair_frame_state_sha256,
)
from maple.solvation.surfaces.fixed_topology import CAVITY_PROFILE_ID

PAIR_FRAME_CONTINUUM_PROFILE_ID = CONTINUUM_PROFILE_ID
PAIR_FRAME_CAVITY_PROFILE_ID = CAVITY_PROFILE_ID
PAIR_FRAME_CPCM_LEBEDEV_ORDER = 17
_ROTATION_CONTRACT_TOLERANCE = 5.0e-12


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation_sha256() -> tuple[tuple[str, str], ...]:
    directory = Path(__file__).parent
    names = (
        "pair_frame_ensemble_cpcm.py",
        "pair_frame_geometry.py",
        "pair_frame_state.py",
    )
    return tuple(
        (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        for name in names
    )


def _positions(geometry: Any, symbols: tuple[str, ...]) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else geometry
    result = np.asarray(values, dtype=float)
    if result.shape != (len(symbols), 3) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"geometry positions must be finite with shape ({len(symbols)}, 3)."
        )
    number_getter = getattr(geometry, "get_atomic_numbers", None)
    if callable(number_getter):
        expected = np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=int)
        actual = np.asarray(number_getter(), dtype=int)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(
                "ASE geometry atomic numbers do not match backend symbols."
            )
    return np.array(result, copy=True)


def _rotate_radial_gto_blocks(values: object, rotation: object) -> np.ndarray:
    """Apply a proper rotation to both raw real-spherical l=1 blocks."""

    result = np.asarray(values, dtype=float)
    transform = np.asarray(rotation, dtype=float)
    if (
        result.ndim != 2
        or result.shape != (result.shape[0], 8)
        or result.shape[0] < 1
        or not np.all(np.isfinite(result))
        or transform.shape != (3, 3)
        or not np.all(np.isfinite(transform))
        or not np.allclose(
            transform @ transform.T,
            np.eye(3),
            atol=_ROTATION_CONTRACT_TOLERANCE,
            rtol=0.0,
        )
        or not np.isclose(
            np.linalg.det(transform),
            1.0,
            atol=_ROTATION_CONTRACT_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise ValueError("values/rotation do not satisfy the radial-GTO contract.")
    result = np.array(result, copy=True)
    for raw_indices in ((2, 3, 4), (5, 6, 7)):
        cartesian = result[:, raw_indices][:, (2, 0, 1)]
        result[:, raw_indices] = (cartesian @ transform.T)[:, (1, 2, 0)]
    return result


def _raw_vector_blocks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 8 or not np.all(np.isfinite(array)):
        raise ValueError("radial values must be finite with shape (atom_count, 8).")
    first = array[:, (2, 3, 4)][:, (2, 0, 1)]
    second = array[:, (5, 6, 7)][:, (2, 0, 1)]
    return first, second


def _orientation_transform_bar(
    values_lab: np.ndarray, cotangent_body: np.ndarray
) -> np.ndarray:
    result = np.zeros((3, 3), dtype=float)
    for lab, bar in zip(
        _raw_vector_blocks(values_lab),
        _raw_vector_blocks(cotangent_body),
        strict=True,
    ):
        result += lab.T @ bar
    return result


@dataclass(frozen=True, slots=True)
class _FrameOperator:
    frame: PairFrame
    body_positions: np.ndarray
    orientation: np.ndarray
    centered_positions: np.ndarray
    body_matrix: np.ndarray


def _readonly_array(values: object) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


class OrderedPairFrameEnsembleRadialGTOCPCMBackend:
    """Disabled equivariant average of one body-frame amplitude-SWIG C-PCM map.

    The frame list contains every ordered atom pair.  Numerically singular
    pairs remain present as zero-weight members, so member and node ownership
    never changes.  The normalized weights and per-frame C-PCM energies are
    part of one declared half-coupling scalar.  This is a new ensemble
    discretization, not a claim of equivalence to one conventional cavity.
    """

    __slots__ = (
        "_body_backend",
        "_configuration",
        "_configuration_sha256",
        "_nuclear_charges",
        "_operator_cache_key",
        "_operator_cache_lock",
        "_operator_cache_value",
        "_provenance_sha256",
        "_sealed",
        "_symbols",
    )
    provider_id = PAIR_FRAME_CPCM_PROVIDER_ID
    continuum_profile_id = PAIR_FRAME_CONTINUUM_PROFILE_ID
    cavity_profile_id = PAIR_FRAME_CAVITY_PROFILE_ID
    configuration_contract_id = PAIR_FRAME_WATER_CPCM_110_CONFIGURATION_CONTRACT_ID
    coupling_id = PAIR_FRAME_CPCM_COUPLING_ID
    scalar_id = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    source_space = MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
    field_space = MACE_POLAR_RADIAL_GTO_FIELD_DUAL_SPACE
    pairing = MACE_POLAR_RADIAL_GTO_PAIRING
    capabilities = CapabilityStatus()
    fixed_topology = True
    linear_response = True
    reciprocal = True
    source_dependent_geometry = False
    ensemble_discretization = True

    def __init__(self, symbols: Sequence[str]) -> None:
        normalized = tuple(str(symbol) for symbol in symbols)
        if len(normalized) < 3:
            raise ValueError("ordered pair-frame C-PCM requires at least three atoms.")
        charges = np.asarray(
            [atomic_numbers[symbol] for symbol in normalized], dtype=float
        )
        body = ConjugateRadialGTOFixedTopologyCPCMBackend(
            normalized,
            smd_water_coulomb_radii(normalized),
            dielectric=WATER_CPCM_DIELECTRIC,
            lebedev_order=PAIR_FRAME_CPCM_LEBEDEV_ORDER,
        )
        if body.surface_provider.unit_sphere.shape != (
            PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM,
            4,
        ):
            raise RuntimeError(
                "pinned pair-frame grid did not contain 110 points/atom."
            )
        configuration = (
            normalized,
            tuple(float(value) for value in charges),
            body.configuration_sha256(),
            body.provenance_sha256,
            self.provider_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.configuration_contract_id,
            self.coupling_id,
            self.scalar_id,
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            self.pairing.metadata_hash(),
            PAIR_FRAME_POLICY,
            frame_topology_sha256(
                len(normalized), PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM
            ),
            HARTREE_TO_EV,
            _implementation_sha256(),
        )
        configuration_sha256 = _sha(configuration)
        provenance_sha256 = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": configuration_sha256,
                "body_provider_id": body.provider_id,
                "body_provenance_sha256": body.provenance_sha256,
                "ensemble_policy": PAIR_FRAME_POLICY,
                "scientific_status": (
                    "new-ensemble-discretization-not-conventional-single-cpcm"
                ),
            }
        )
        object.__setattr__(self, "_symbols", normalized)
        object.__setattr__(
            self, "_nuclear_charges", tuple(float(value) for value in charges)
        )
        object.__setattr__(self, "_body_backend", body)
        object.__setattr__(self, "_operator_cache_key", None)
        object.__setattr__(self, "_operator_cache_lock", threading.RLock())
        object.__setattr__(self, "_operator_cache_value", None)
        object.__setattr__(self, "_configuration", configuration)
        object.__setattr__(self, "_configuration_sha256", configuration_sha256)
        object.__setattr__(self, "_provenance_sha256", provenance_sha256)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ordered pair-frame C-PCM backend is immutable.")
        object.__setattr__(self, name, value)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def nuclear_charges(self) -> np.ndarray:
        result = np.asarray(self._nuclear_charges, dtype=float)
        result.setflags(write=False)
        return result

    @property
    def body_backend(self) -> ConjugateRadialGTOFixedTopologyCPCMBackend:
        return self._body_backend

    @property
    def provenance_sha256(self) -> str:
        return self._provenance_sha256

    def _current_configuration(self) -> tuple[object, ...]:
        return (
            self.symbols,
            tuple(float(value) for value in self.nuclear_charges),
            self.body_backend.configuration_sha256(),
            self.body_backend.provenance_sha256,
            self.provider_id,
            self.continuum_profile_id,
            self.cavity_profile_id,
            self.configuration_contract_id,
            self.coupling_id,
            self.scalar_id,
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            self.pairing.metadata_hash(),
            PAIR_FRAME_POLICY,
            frame_topology_sha256(
                len(self.symbols), PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM
            ),
            HARTREE_TO_EV,
            _implementation_sha256(),
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if (
            current != self._configuration
            or _sha(current) != self._configuration_sha256
        ):
            raise RuntimeError("ordered pair-frame C-PCM configuration drifted.")
        expected_provenance = _sha(
            {
                "provider_id": self.provider_id,
                "configuration_sha256": self._configuration_sha256,
                "body_provider_id": self.body_backend.provider_id,
                "body_provenance_sha256": self.body_backend.provenance_sha256,
                "ensemble_policy": PAIR_FRAME_POLICY,
                "scientific_status": (
                    "new-ensemble-discretization-not-conventional-single-cpcm"
                ),
            }
        )
        if expected_provenance != self._provenance_sha256:
            raise RuntimeError("ordered pair-frame C-PCM provenance drifted.")
        return self._configuration_sha256

    def _validated_source(self, values: object, *, name: str) -> np.ndarray:
        return self.source_space.validate(
            values, atom_count=len(self.symbols), name=name
        )

    @staticmethod
    def _geometry_cache_key(positions: np.ndarray) -> tuple[tuple[int, ...], bytes]:
        canonical = np.ascontiguousarray(positions, dtype="<f8")
        return canonical.shape, canonical.tobytes(order="C")

    def _frame_operators(self, geometry: Any):
        self.configuration_sha256()
        positions = _positions(geometry, self.symbols)
        key = self._geometry_cache_key(positions)
        with self._operator_cache_lock:
            if key == self._operator_cache_key:
                cached = self._operator_cache_value
                if cached is None:  # pragma: no cover - invariant guard
                    raise RuntimeError("pair-frame operator cache is missing.")
                return cached
        centroid, frames, total_weight = frame_data(positions, self.nuclear_charges)
        centered = positions - centroid
        operators: list[_FrameOperator] = []
        for frame in frames:
            orientation = frame.orientation
            body_positions = centered @ orientation
            body_matrix = self.body_backend.reaction_field_matrix(body_positions)
            operators.append(
                _FrameOperator(
                    frame,
                    _readonly_array(body_positions),
                    _readonly_array(orientation),
                    _readonly_array(centered),
                    body_matrix,
                )
            )
        result = (
            _readonly_array(positions),
            _readonly_array(centroid),
            tuple(operators),
            total_weight,
        )
        with self._operator_cache_lock:
            object.__setattr__(self, "_operator_cache_key", key)
            object.__setattr__(self, "_operator_cache_value", result)
        return result

    def _frames(self, geometry: Any):
        self.configuration_sha256()
        positions = _positions(geometry, self.symbols)
        centroid, frames, total_weight = frame_data(positions, self.nuclear_charges)
        return positions, centroid, frames, total_weight

    def evaluate_field(self, geometry: Any, source: object) -> np.ndarray:
        _positions_value, _centroid, operators, total_weight = self._frame_operators(
            geometry
        )
        values = self._validated_source(source, name="source")
        result = np.zeros_like(values)
        for operator in operators:
            if operator.frame.weight == 0.0:
                continue
            body_source = _rotate_radial_gto_blocks(values, operator.orientation.T)
            body_field = (operator.body_matrix @ body_source.reshape(-1)).reshape(
                values.shape
            )
            result += operator.frame.weight * _rotate_radial_gto_blocks(
                body_field, operator.orientation
            )
        return self.field_space.validate(
            result / total_weight,
            atom_count=len(self.symbols),
            name="ordered pair-frame reaction field",
        )

    field = evaluate_field

    def energy(self, geometry: Any, source: object) -> float:
        values = self._validated_source(source, name="source")
        return float(
            0.5 * self.pairing.pair(values, self.evaluate_field(geometry, values))
        )

    def build_state(self, geometry: Any, source: object) -> PairFrameCPCMState:
        values = self._validated_source(source, name="source")
        positions, _centroid, operators, _total_weight = self._frame_operators(geometry)
        field = self.evaluate_field(geometry, values)
        energy = float(0.5 * self.pairing.pair(values, field))
        frames = tuple(operator.frame for operator in operators)
        geometry_digest = geometry_sha256(positions, self.symbols)
        topology_sha256 = frame_topology_sha256(
            len(self.symbols), PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM
        )
        activity_sha256 = frame_activity_sha256(frames)
        active_frame_count = sum(frame.weight > 0.0 for frame in frames)
        payload = {
            "contract": "ordered-pair-frame-ensemble-cpcm-state-v1",
            "provider_id": self.provider_id,
            "configuration_sha256": self.configuration_sha256(),
            "provenance_sha256": self.provenance_sha256,
            "geometry_sha256": geometry_digest,
            "frame_topology_sha256": topology_sha256,
            "frame_activity_sha256": activity_sha256,
            "frame_count": len(frames),
            "active_frame_count": active_frame_count,
            "source": values.tolist(),
            "field": field.tolist(),
            "polarization_energy_eV": energy,
        }
        return PairFrameCPCMState(
            provider_id=self.provider_id,
            configuration_sha256=self.configuration_sha256(),
            provenance_sha256=self.provenance_sha256,
            geometry_sha256=geometry_digest,
            frame_topology_sha256=topology_sha256,
            frame_activity_sha256=activity_sha256,
            state_hash=pair_frame_state_sha256(payload),
            source_values=tuple(float(value) for value in values.reshape(-1)),
            field_values=tuple(float(value) for value in field.reshape(-1)),
            polarization_energy_eV=energy,
            atom_count=len(self.symbols),
            frame_count=len(frames),
            active_frame_count=active_frame_count,
        )

    def source_jvp(
        self, geometry: Any, source: object, source_direction: object
    ) -> np.ndarray:
        self._validated_source(source, name="source")
        direction = self._validated_source(source_direction, name="source direction")
        return self.evaluate_field(geometry, direction)

    def source_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        self._validated_source(source, name="source")
        cotangent = self.field_space.validate(
            field_cotangent, atom_count=len(self.symbols), name="field cotangent"
        )
        # Every member is reciprocal and radial rotations are orthogonal under Q=I.
        return self.evaluate_field(geometry, cotangent)

    def coordinate_vjp(
        self, geometry: Any, source: object, field_cotangent: object
    ) -> np.ndarray:
        positions, _centroid, operators, total_weight = self._frame_operators(geometry)
        values = self._validated_source(source, name="source")
        cotangent = self.field_space.validate(
            field_cotangent, atom_count=len(self.symbols), name="field cotangent"
        )
        members: list[tuple[float, np.ndarray, np.ndarray, np.ndarray] | None] = []
        for operator in operators:
            if operator.frame.weight == 0.0:
                members.append(None)
                continue
            body_source = _rotate_radial_gto_blocks(values, operator.orientation.T)
            body_cotangent = _rotate_radial_gto_blocks(
                cotangent, operator.orientation.T
            )
            body_field = (operator.body_matrix @ body_source.reshape(-1)).reshape(
                values.shape
            )
            members.append(
                (
                    float(np.vdot(body_cotangent, body_field)),
                    body_source,
                    body_cotangent,
                    body_field,
                )
            )
        average = (
            sum(
                operator.frame.weight * member[0]
                for operator, member in zip(operators, members, strict=True)
                if member is not None
            )
            / total_weight
        )
        result = np.zeros_like(positions)
        charge_weights = self.nuclear_charges / float(np.sum(self.nuclear_charges))
        for operator, member in zip(operators, members, strict=True):
            frame = operator.frame
            if member is None:
                continue
            member_value, body_source, body_cotangent, body_field = member
            alpha = frame.weight / total_weight
            body_position_bar = self.body_backend.coordinate_vjp(
                operator.body_positions,
                body_source,
                body_cotangent,
            ).reshape(positions.shape)
            body_source_bar = self.body_backend.source_vjp(
                operator.body_positions,
                body_source,
                body_cotangent,
            )
            direct = body_position_bar @ operator.orientation.T
            result += alpha * (
                direct - charge_weights[:, None] * np.sum(direct, axis=0)
            )
            orientation_bar = operator.centered_positions.T @ body_position_bar
            orientation_bar += _orientation_transform_bar(values, body_source_bar)
            orientation_bar += _orientation_transform_bar(cotangent, body_field)
            weight_bar = (member_value - average) / total_weight
            result += pair_frame_vjp(
                frame,
                alpha * orientation_bar,
                weight_bar,
                self.nuclear_charges,
            )
        if result.shape != positions.shape or not np.all(np.isfinite(result)):
            raise RuntimeError("ordered pair-frame coordinate VJP is invalid.")
        return result.reshape(-1).copy()

    def frame_diagnostics(self, geometry: Any) -> dict[str, object]:
        _positions_value, _centroid, frames, total_weight = self._frames(geometry)
        positive = tuple(frame for frame in frames if frame.weight > 0.0)
        return {
            "ensemble_policy": PAIR_FRAME_POLICY,
            "frame_topology_sha256": frame_topology_sha256(
                len(self.symbols), PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM
            ),
            "frame_activity_sha256": frame_activity_sha256(frames),
            "ordered_pair_frame_count": len(frames),
            "ordered_noncollinear_frame_count": len(positive),
            "zero_weight_singular_frame_count": len(frames) - len(positive),
            "total_weight": total_weight,
            "minimum_positive_weight": min(frame.weight for frame in positive),
            "maximum_weight": max(frame.weight for frame in positive),
        }

    @property
    def runtime_provenance(self) -> tuple[tuple[str, str], ...]:
        self.configuration_sha256()
        return tuple(
            sorted(
                {
                    "provider_id": self.provider_id,
                    "continuum_profile_id": self.continuum_profile_id,
                    "cavity_profile_id": self.cavity_profile_id,
                    "configuration_contract_id": self.configuration_contract_id,
                    "configuration_sha256": self.configuration_sha256(),
                    "provenance_sha256": self.provenance_sha256,
                    "ensemble_policy": PAIR_FRAME_POLICY,
                    "frame_topology_sha256": frame_topology_sha256(
                        len(self.symbols), PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM
                    ),
                    "scientific_status": (
                        "new-ensemble-discretization-not-conventional-single-cpcm"
                    ),
                    "coordinate_derivative": (
                        "complete-member-plus-frame-plus-normalized-weight-vjp"
                    ),
                    "capabilities": "none-pending-full-real-stack-gates",
                }.items()
            )
        )


def build_pair_frame_water_cpcm_110_candidate(
    symbols: Sequence[str],
) -> OrderedPairFrameEnsembleRadialGTOCPCMBackend:
    """Build the exact disabled equivariant water candidate without knobs."""

    return OrderedPairFrameEnsembleRadialGTOCPCMBackend(tuple(symbols))


__all__ = [
    "PAIR_FRAME_CPCM_PROVIDER_ID",
    "PAIR_FRAME_CONTINUUM_PROFILE_ID",
    "PAIR_FRAME_CAVITY_PROFILE_ID",
    "PAIR_FRAME_CPCM_LEBEDEV_ORDER",
    "PAIR_FRAME_CPCM_GRID_POINTS_PER_ATOM",
    "PairFrameCPCMState",
    "OrderedPairFrameEnsembleRadialGTOCPCMBackend",
    "build_pair_frame_water_cpcm_110_candidate",
]
