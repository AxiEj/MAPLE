"""Pure geometry and first-order pullbacks for ordered-pair body frames."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

PAIR_FRAME_POLICY = (
    "all-ordered-pairs-zero-weight-singular-extension-" "nuclear-charge-centroid-s4-v1"
)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PairFrame:
    first: int
    second: int
    displacement: tuple[float, float, float]
    centroid_offset: tuple[float, float, float]
    first_axis: tuple[float, float, float]
    second_axis: tuple[float, float, float]
    third_axis: tuple[float, float, float]
    cross_norm_squared: float
    weight: float
    singular_extension: bool

    @property
    def orientation(self) -> np.ndarray:
        return np.column_stack((self.first_axis, self.second_axis, self.third_axis))


def _fallback_axes(first_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded frame for an exactly zero-weight singular member."""

    reference = np.eye(3)[int(np.argmin(np.abs(first_axis)))]
    third_axis = np.cross(first_axis, reference)
    third_axis /= np.linalg.norm(third_axis)
    return np.cross(third_axis, first_axis), third_axis


def frame_topology_sha256(atom_count: int, nodes_per_atom: int) -> str:
    return _sha(
        {
            "contract": "ordered-pair-frame-topology-v1",
            "atom_count": atom_count,
            "members": [
                [first, second]
                for first in range(atom_count)
                for second in range(atom_count)
                if first != second
            ],
            "nodes_per_atom_per_member": nodes_per_atom,
        }
    )


def frame_activity_sha256(frames: tuple[PairFrame, ...]) -> str:
    return _sha(
        {
            "contract": "ordered-pair-frame-numerical-activity-v1",
            "active_members": [
                [frame.first, frame.second] for frame in frames if frame.weight > 0.0
            ],
        }
    )


def geometry_sha256(positions: np.ndarray, symbols: tuple[str, ...]) -> str:
    canonical = np.ascontiguousarray(positions, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"ordered-pair-frame-geometry-v1\0")
    digest.update(json.dumps(symbols, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(str(canonical.shape).encode())
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def frame_data(
    positions: np.ndarray, nuclear_charges: np.ndarray
) -> tuple[np.ndarray, tuple[PairFrame, ...], float]:
    charge_sum = float(np.sum(nuclear_charges))
    if not math.isfinite(charge_sum) or charge_sum <= 0.0:
        raise ValueError("nuclear-charge centroid requires positive nuclear charge.")
    centroid = np.sum(nuclear_charges[:, None] * positions, axis=0) / charge_sum
    frames: list[PairFrame] = []
    for first in range(len(positions)):
        for second in range(len(positions)):
            if first == second:
                continue
            displacement = positions[second] - positions[first]
            displacement_norm = float(np.linalg.norm(displacement))
            if not math.isfinite(displacement_norm) or displacement_norm <= 0.0:
                raise ValueError("ordered pair-frame atoms must not coincide.")
            midpoint = 0.5 * (positions[first] + positions[second])
            offset = centroid - midpoint
            cross = np.cross(displacement, offset)
            cross_norm = float(np.linalg.norm(cross))
            singularity_scale = displacement_norm * float(np.linalg.norm(offset))
            first_axis = displacement / displacement_norm
            cross_norm_squared = cross_norm * cross_norm
            singular = singularity_scale == 0.0 or cross_norm == 0.0
            if singular:
                second_axis, third_axis = _fallback_axes(first_axis)
                weight = 0.0
            else:
                third_axis = cross / cross_norm
                second_axis = np.cross(third_axis, first_axis)
                weight = cross_norm_squared**4
            if not math.isfinite(weight) or weight < 0.0:
                raise RuntimeError("ordered pair-frame weight is invalid.")
            frames.append(
                PairFrame(
                    first,
                    second,
                    tuple(float(value) for value in displacement),
                    tuple(float(value) for value in offset),
                    tuple(float(value) for value in first_axis),
                    tuple(float(value) for value in second_axis),
                    tuple(float(value) for value in third_axis),
                    cross_norm_squared,
                    weight,
                    singular,
                )
            )
    total_weight = float(sum(frame.weight for frame in frames))
    if (
        len(frames) != len(positions) * (len(positions) - 1)
        or not math.isfinite(total_weight)
        or total_weight <= 0.0
    ):
        raise ValueError(
            "ordered pair-frame ensemble requires a non-collinear molecular geometry."
        )
    return centroid, tuple(frames), total_weight


def pair_frame_vjp(
    frame: PairFrame,
    orientation_bar: np.ndarray,
    weight_bar: float,
    nuclear_charges: np.ndarray,
) -> np.ndarray:
    displacement = np.asarray(frame.displacement)
    offset = np.asarray(frame.centroid_offset)
    first_axis = np.asarray(frame.first_axis)
    third_axis = np.asarray(frame.third_axis)
    first_bar = np.array(orientation_bar[:, 0], copy=True)
    second_bar = np.asarray(orientation_bar[:, 1])
    third_bar = np.array(orientation_bar[:, 2], copy=True)

    first_bar += np.cross(second_bar, third_axis)
    third_bar += np.cross(first_axis, second_bar)
    cross = np.cross(displacement, offset)
    cross_norm = float(np.linalg.norm(cross))
    cross_bar = (third_bar - third_axis * np.dot(third_axis, third_bar)) / cross_norm
    displacement_bar = np.cross(offset, cross_bar)
    offset_bar = np.cross(cross_bar, displacement)
    displacement_norm = float(np.linalg.norm(displacement))
    displacement_bar += (
        first_bar - first_axis * np.dot(first_axis, first_bar)
    ) / displacement_norm

    factor = float(weight_bar) * 8.0 * frame.cross_norm_squared**3
    displacement_dot_offset = float(np.dot(displacement, offset))
    displacement_bar += factor * (
        np.dot(offset, offset) * displacement - displacement_dot_offset * offset
    )
    offset_bar += factor * (
        np.dot(displacement, displacement) * offset
        - displacement_dot_offset * displacement
    )

    result = np.zeros((len(nuclear_charges), 3), dtype=float)
    result[frame.first] -= displacement_bar
    result[frame.second] += displacement_bar
    result += nuclear_charges[:, None] / float(np.sum(nuclear_charges)) * offset_bar
    result[frame.first] -= 0.5 * offset_bar
    result[frame.second] -= 0.5 * offset_bar
    return result


__all__ = [
    "PAIR_FRAME_POLICY",
    "PairFrame",
    "frame_activity_sha256",
    "frame_data",
    "frame_topology_sha256",
    "geometry_sha256",
    "pair_frame_vjp",
]
