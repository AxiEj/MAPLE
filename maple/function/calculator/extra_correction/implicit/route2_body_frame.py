"""Pure Johnson--Gill--Pople standard-frame transform and reverse VJP.

This production-library module has no continuum-provider, MACE, or MAPLE parser
dependency.  It defines a local differentiable chart of the nuclear-charge
principal-axis frame for nondegenerate geometries.  It does *not* make a
variable-topology continuum force-admissible; callers must fail closed when the
relative eigengap guard or the local signed-axis gauge is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class JGP94Frame:
    """Validated local standard frame for one molecular geometry."""

    center: np.ndarray
    centered_positions: np.ndarray
    moment_tensor: np.ndarray
    eigenvalues: np.ndarray
    orientation: np.ndarray
    body_positions: np.ndarray
    relative_minimum_eigengap: float
    orthogonality_error: float
    determinant_error: float
    reference_axis_overlaps: np.ndarray | None


@dataclass(frozen=True)
class JGP94FrameVJP:
    """Reverse contraction through body positions and body dipoles."""

    position_cotangent: np.ndarray
    dipole_cotangent: np.ndarray
    direct_centered_position_cotangent: np.ndarray
    orientation_position_cotangent: np.ndarray
    total_centered_position_cotangent: np.ndarray
    orientation_cotangent: np.ndarray
    moment_cotangent: np.ndarray
    eigenbasis_moment_cotangent: np.ndarray


def _validated_positions(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be finite with shape (n, 3).")
    return array


def _validated_nuclear_charges(
    values: np.ndarray,
    *,
    atom_count: int,
) -> np.ndarray:
    charges = np.asarray(values, dtype=float)
    if (
        charges.shape != (atom_count,)
        or not np.all(np.isfinite(charges))
        or np.any(charges <= 0.0)
    ):
        raise ValueError(
            "nuclear_charges must be finite and positive with shape "
            f"{(atom_count,)}."
        )
    return charges


def _proper_signed_axis_alignment(
    orientation: np.ndarray,
    reference_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_orientation, dtype=float)
    if reference.shape != (3, 3) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_orientation must be finite with shape (3, 3).")
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for first_sign in (-1.0, 1.0):
        for second_sign in (-1.0, 1.0):
            third_sign = first_sign * second_sign
            signed = orientation @ np.diag(
                [first_sign, second_sign, third_sign]
            )
            overlaps = np.einsum("ij,ij->j", signed, reference)
            score = float(np.sum(overlaps))
            candidates.append((score, signed, overlaps))
    _, signed_orientation, overlaps = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    return signed_orientation, overlaps


def build_jgp94_frame(
    positions: np.ndarray,
    nuclear_charges: np.ndarray,
    *,
    minimum_relative_eigengap: float,
    reference_orientation: np.ndarray | None = None,
    minimum_reference_axis_overlap: float | None = None,
) -> JGP94Frame:
    """Construct one fail-closed nuclear-charge principal-axis frame."""

    coordinates = _validated_positions(positions, name="positions")
    charges = _validated_nuclear_charges(
        nuclear_charges,
        atom_count=coordinates.shape[0],
    )
    eigengap_guard = float(minimum_relative_eigengap)
    if not math.isfinite(eigengap_guard) or eigengap_guard <= 0.0:
        raise ValueError(
            "minimum_relative_eigengap must be finite and positive."
        )
    if minimum_reference_axis_overlap is not None:
        overlap_guard = float(minimum_reference_axis_overlap)
        if not math.isfinite(overlap_guard) or not 0.0 < overlap_guard <= 1.0:
            raise ValueError(
                "minimum_reference_axis_overlap must lie in (0, 1]."
            )
        if reference_orientation is None:
            raise ValueError(
                "An axis-overlap guard requires a reference orientation."
            )
    else:
        overlap_guard = None

    total_charge = float(np.sum(charges))
    center = np.sum(charges[:, None] * coordinates, axis=0) / total_charge
    centered = coordinates - center
    moment = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for charge, displacement in zip(charges, centered, strict=True):
        moment += charge * (
            float(np.dot(displacement, displacement)) * identity
            - np.outer(displacement, displacement)
        )

    eigenvalues, orientation = np.linalg.eigh(moment)
    if float(np.linalg.det(orientation)) < 0.0:
        orientation[:, -1] *= -1.0

    axis_overlaps = None
    if reference_orientation is not None:
        orientation, axis_overlaps = _proper_signed_axis_alignment(
            orientation,
            reference_orientation,
        )
        if overlap_guard is not None and float(np.min(axis_overlaps)) < overlap_guard:
            raise RuntimeError(
                "The JGP94 frame left the predeclared local signed-axis gauge."
            )

    eigenvalue_gaps = np.diff(eigenvalues)
    eigenvalue_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    relative_eigengap = float(np.min(eigenvalue_gaps) / eigenvalue_scale)
    if relative_eigengap < eigengap_guard:
        raise RuntimeError(
            "The JGP94 frame is outside the nondegenerate eigengap domain "
            f"({relative_eigengap:.6e} < {eigengap_guard:.6e})."
        )

    orthogonality_error = float(
        np.max(np.abs(orientation.T @ orientation - identity))
    )
    determinant_error = abs(float(np.linalg.det(orientation)) - 1.0)
    if orthogonality_error > 1.0e-12 or determinant_error > 1.0e-12:
        raise RuntimeError("The JGP94 orientation is not a proper orthogonal frame.")

    return JGP94Frame(
        center=center,
        centered_positions=centered,
        moment_tensor=moment,
        eigenvalues=eigenvalues,
        orientation=orientation,
        body_positions=centered @ orientation,
        relative_minimum_eigengap=relative_eigengap,
        orthogonality_error=orthogonality_error,
        determinant_error=determinant_error,
        reference_axis_overlaps=axis_overlaps,
    )


def body_dipoles(
    dipoles: np.ndarray,
    frame: JGP94Frame,
) -> np.ndarray:
    """Transform row-vector Cartesian dipoles into the body frame."""

    cartesian = _validated_positions(dipoles, name="dipoles")
    if cartesian.shape[0] != frame.centered_positions.shape[0]:
        raise ValueError("dipoles must contain one row per frame atom.")
    return cartesian @ frame.orientation


def jgp94_frame_vjp(
    frame: JGP94Frame,
    nuclear_charges: np.ndarray,
    dipoles: np.ndarray,
    body_position_cotangent: np.ndarray,
    body_dipole_cotangent: np.ndarray,
    *,
    additional_orientation_cotangent: np.ndarray | None = None,
) -> JGP94FrameVJP:
    """Contract the reverse derivative of ``Y=C@O, p_b=p@O``.

    ``additional_orientation_cotangent`` is the derivative of an otherwise
    external scalar with respect to the proper orientation ``O``.  It is
    needed by consumers that rotate a vector-valued *output* back to the lab
    frame, rather than only rotating input dipoles into the body frame.  The
    optional term makes that use explicit instead of encouraging wrappers to
    reimplement the sensitive eigenvector VJP.
    """

    atom_count = frame.centered_positions.shape[0]
    charges = _validated_nuclear_charges(
        nuclear_charges,
        atom_count=atom_count,
    )
    cartesian_dipoles = _validated_positions(dipoles, name="dipoles")
    if cartesian_dipoles.shape[0] != atom_count:
        raise ValueError("dipoles must contain one row per atom.")
    position_bar = _validated_positions(
        body_position_cotangent,
        name="body_position_cotangent",
    )
    dipole_bar = _validated_positions(
        body_dipole_cotangent,
        name="body_dipole_cotangent",
    )
    if position_bar.shape[0] != atom_count or dipole_bar.shape[0] != atom_count:
        raise ValueError("Body cotangents must contain one row per atom.")

    if additional_orientation_cotangent is None:
        additional_orientation_bar = np.zeros((3, 3), dtype=float)
    else:
        additional_orientation_bar = np.asarray(
            additional_orientation_cotangent,
            dtype=float,
        )
        if (
            additional_orientation_bar.shape != (3, 3)
            or not np.all(np.isfinite(additional_orientation_bar))
        ):
            raise ValueError(
                "additional_orientation_cotangent must be finite with shape (3, 3)."
            )

    orientation = frame.orientation
    centered = frame.centered_positions
    direct_centered_bar = position_bar @ orientation.T
    orientation_bar = (
        centered.T @ position_bar
        + cartesian_dipoles.T @ dipole_bar
        + additional_orientation_bar
    )

    eigenbasis_orientation_bar = orientation.T @ orientation_bar
    skew_orientation_bar = 0.5 * (
        eigenbasis_orientation_bar - eigenbasis_orientation_bar.T
    )
    eigenbasis_moment_bar = np.zeros((3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            if row != column:
                denominator = (
                    frame.eigenvalues[column] - frame.eigenvalues[row]
                )
                eigenbasis_moment_bar[row, column] = (
                    skew_orientation_bar[row, column] / denominator
                )
    symmetry_error = float(
        np.max(
            np.abs(
                eigenbasis_moment_bar - eigenbasis_moment_bar.T
            )
        )
    )
    if symmetry_error > 1.0e-12:
        raise RuntimeError("The eigenbasis moment cotangent is not symmetric.")

    moment_bar = (
        orientation
        @ eigenbasis_moment_bar
        @ orientation.T
    )
    moment_bar = 0.5 * (moment_bar + moment_bar.T)
    trace_moment_bar = float(np.trace(moment_bar))
    orientation_centered_bar = np.empty_like(centered)
    for atom_index, (charge, displacement) in enumerate(
        zip(charges, centered, strict=True)
    ):
        orientation_centered_bar[atom_index] = 2.0 * charge * (
            trace_moment_bar * displacement - displacement @ moment_bar
        )

    total_centered_bar = direct_centered_bar + orientation_centered_bar
    weights = charges / float(np.sum(charges))
    position_cotangent = (
        total_centered_bar
        - weights[:, None] * np.sum(total_centered_bar, axis=0)
    )
    dipole_cotangent = dipole_bar @ orientation.T
    arrays = (
        position_cotangent,
        dipole_cotangent,
        direct_centered_bar,
        orientation_centered_bar,
        total_centered_bar,
        orientation_bar,
        moment_bar,
        eigenbasis_moment_bar,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("The JGP94 frame VJP produced non-finite values.")

    return JGP94FrameVJP(
        position_cotangent=position_cotangent,
        dipole_cotangent=dipole_cotangent,
        direct_centered_position_cotangent=direct_centered_bar,
        orientation_position_cotangent=orientation_centered_bar,
        total_centered_position_cotangent=total_centered_bar,
        orientation_cotangent=orientation_bar,
        moment_cotangent=moment_bar,
        eigenbasis_moment_cotangent=eigenbasis_moment_bar,
    )


__all__ = [
    "JGP94Frame",
    "JGP94FrameVJP",
    "body_dipoles",
    "build_jgp94_frame",
    "jgp94_frame_vjp",
]
