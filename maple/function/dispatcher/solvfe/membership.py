from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from ase.units import kB

from .protocol import canonical_sha256


PERIODIC_IMAGE_LOG_TOLERANCE = 50.0
PERIODIC_IMAGE_MAX_SHELLS = 4


def membership_surface_identity_hash(
    *,
    solute_indices: tuple[int, ...],
    solute_atom_map_hash: str,
    solute_measure_hash: str,
    membership_definition_hash: str,
) -> str:
    """Boundary-independent identity of the shared smooth-union surface."""

    return canonical_sha256(
        {
            "contract_id": "membership-surface-identity-v4",
            "solute_indices": [int(index) for index in solute_indices],
            "solute_atom_map_hash": solute_atom_map_hash,
            "solute_measure_hash": solute_measure_hash,
            "membership_definition_hash": membership_definition_hash,
            "coordinate": (
                "log-sum-exp smooth minimum of atom-sphere signed distances"
            ),
            "complete_solvent_molecule_rule": "oxygen membership",
        }
    )


@dataclass(frozen=True)
class SoftCutoffEvaluation:
    membership: np.ndarray
    nonmembership: np.ndarray
    member_potential_ev: np.ndarray
    empty_potential_ev: np.ndarray
    member_derivative_ev_per_angstrom: np.ndarray
    empty_derivative_ev_per_angstrom: np.ndarray


@dataclass(frozen=True)
class SmoothSurfaceGeometry:
    signed_distances_angstrom: np.ndarray
    center_weights: np.ndarray
    center_gradient_vectors: np.ndarray
    periodic_image_shells: int


@dataclass(frozen=True)
class SoftCutoffMembership:
    """One complementary soft-cutoff definition for Route A QCT.

    For signed surface distance ``d``, the membership is

    ``b(d) = 1 / (1 + exp((d - lambda_s) / R))``.

    Association and empty-shell fields are derived from the same ``b``:

    ``u_member = -kT ln(b)`` and ``u_empty = -kT ln(1 - b)``.
    """

    vdw_radii_angstrom: Mapping[int, float]
    lambda_s_angstrom: float
    softness_angstrom: float
    surface_smoothing_angstrom: float
    shell_boundary_id: str

    def __post_init__(self) -> None:
        radii = {
            int(number): float(radius)
            for number, radius in self.vdw_radii_angstrom.items()
        }
        if not radii or any(
            number <= 0
            or not math.isfinite(radius)
            or radius <= 0.0
            for number, radius in radii.items()
        ):
            raise ValueError(
                "Soft-cutoff vdW radii must be finite positive values."
            )
        for name, value in (
            ("lambda_s_angstrom", self.lambda_s_angstrom),
            ("softness_angstrom", self.softness_angstrom),
            (
                "surface_smoothing_angstrom",
                self.surface_smoothing_angstrom,
            ),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if (
            not isinstance(self.shell_boundary_id, str)
            or not self.shell_boundary_id
        ):
            raise ValueError("shell_boundary_id must be a non-empty string.")
        object.__setattr__(
            self,
            "vdw_radii_angstrom",
            MappingProxyType(radii),
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "soft-cutoff-membership-v2",
                "surface_coordinate": (
                    "d_s=-tau*log(sum_i exp[-d_i/tau]); "
                    "d_i=|r-R_i|-r_i_vdw"
                ),
                "membership": (
                    "b(d)=1/(1+exp((d-lambda_s)/R))"
                ),
                "member_field": "u_member=-kT*ln(b)",
                "empty_field": "u_empty=-kT*ln(1-b)",
                "shell_boundary_id": self.shell_boundary_id,
                "lambda_s_angstrom": float(self.lambda_s_angstrom),
                "softness_angstrom": float(self.softness_angstrom),
                "surface_smoothing_angstrom": float(
                    self.surface_smoothing_angstrom
                ),
                "vdw_radii_angstrom": {
                    str(number): radius
                    for number, radius in sorted(
                        self.vdw_radii_angstrom.items()
                    )
                },
            }
        )

    def evaluate_signed_distances(
        self,
        signed_distances_angstrom: np.ndarray,
        *,
        temperature_k: float,
    ) -> SoftCutoffEvaluation:
        distances = np.asarray(signed_distances_angstrom, dtype=float)
        if distances.size == 0 or not np.all(np.isfinite(distances)):
            raise ValueError(
                "signed_distances_angstrom must be a non-empty finite array."
            )
        if (
            isinstance(temperature_k, (bool, np.bool_))
            or not math.isfinite(float(temperature_k))
            or float(temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")

        with np.errstate(over="ignore", invalid="ignore"):
            z = (
                distances - float(self.lambda_s_angstrom)
            ) / float(self.softness_angstrom)
        if not np.all(np.isfinite(z)):
            raise ValueError(
                "Soft-cutoff scaled distances exceed the finite range."
            )

        member_softplus = np.logaddexp(0.0, z)
        empty_softplus = np.logaddexp(0.0, -z)
        membership = np.exp(-member_softplus)
        nonmembership = np.exp(-empty_softplus)
        thermal_energy_ev = kB * float(temperature_k)
        inverse_softness = 1.0 / float(self.softness_angstrom)

        values = {
            "membership": membership,
            "nonmembership": nonmembership,
            "member_potential_ev": thermal_energy_ev * member_softplus,
            "empty_potential_ev": thermal_energy_ev * empty_softplus,
            "member_derivative_ev_per_angstrom": (
                thermal_energy_ev * inverse_softness * nonmembership
            ),
            "empty_derivative_ev_per_angstrom": (
                -thermal_energy_ev * inverse_softness * membership
            ),
        }
        immutable: dict[str, np.ndarray] = {}
        for name, value in values.items():
            array = np.array(value, dtype=float, copy=True, order="C")
            if not np.all(np.isfinite(array)):
                raise RuntimeError(
                    f"SOFT_MEMBERSHIP_NONFINITE: {name} is non-finite."
                )
            array.setflags(write=False)
            immutable[name] = array
        return SoftCutoffEvaluation(**immutable)


def soft_occupancy_weights(
    memberships: np.ndarray,
    *,
    active_occupancy_max: int | None = None,
) -> np.ndarray:
    """Return soft integer-occupancy weights by polynomial recurrence.

    The last axis enumerates solvent molecules.  The returned last axis
    enumerates either every occupancy or the preregistered active occupancies
    followed by one aggregate overflow bin.
    """

    values = np.asarray(memberships, dtype=float)
    if (
        values.ndim < 1
        or values.shape[-1] == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            "memberships must be a finite non-empty array with values in [0, 1]."
        )

    molecule_count = values.shape[-1]
    if active_occupancy_max is not None:
        if (
            isinstance(active_occupancy_max, (bool, np.bool_))
            or int(active_occupancy_max) != active_occupancy_max
            or not 0 <= int(active_occupancy_max) <= molecule_count
        ):
            raise ValueError(
                "active_occupancy_max must be an integer in "
                "[0, solvent_molecule_count]."
            )
        active_max = int(active_occupancy_max)
        # Last entry is the aggregate n > active_max tail. Once probability
        # reaches it, subsequent Bernoulli factors cannot return it to the
        # active support.
        weights = np.zeros(
            values.shape[:-1] + (active_max + 2,),
            dtype=float,
        )
        weights[..., 0] = 1.0
        for index in range(molecule_count):
            membership = values[..., index, np.newaxis]
            active = weights[..., : active_max + 1]
            updated = np.zeros_like(weights)
            updated[..., 0] = (
                active[..., 0] * (1.0 - membership[..., 0])
            )
            if active_max > 0:
                updated[..., 1 : active_max + 1] = (
                    active[..., 1:] * (1.0 - membership)
                    + active[..., :-1] * membership
                )
            updated[..., -1] = (
                weights[..., -1]
                + active[..., -1] * membership[..., 0]
            )
            weights = updated
    else:
        weights = np.ones(values.shape[:-1] + (1,), dtype=float)
        for index in range(molecule_count):
            membership = values[..., index, np.newaxis]
            updated = np.zeros(
                values.shape[:-1] + (weights.shape[-1] + 1,),
                dtype=float,
            )
            updated[..., :-1] += weights * (1.0 - membership)
            updated[..., 1:] += weights * membership
            weights = updated

    normalization = np.sum(weights, axis=-1)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.allclose(normalization, 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError(
            "SOFT_OCCUPANCY_INVALID: recurrence lost probability normalization."
        )
    weights.setflags(write=False)
    return weights


def smooth_surface_geometry(
    *,
    point_positions_angstrom: np.ndarray,
    center_positions_angstrom: np.ndarray,
    center_radii_angstrom: np.ndarray,
    surface_smoothing_angstrom: float,
    cell_angstrom: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> SmoothSurfaceGeometry:
    """Evaluate a differentiable log-sum-exp union of atom spheres.

    If ``d_i = |r - R_i| - r_i`` is the signed distance to atom sphere
    ``i``, the shared Route A surface coordinate is

    ``d_s = -tau log(sum_i exp(-d_i / tau))``.

    Its point gradient is the softmax-weighted sum of the individual sphere
    gradients. Under periodic boundary conditions, the same log-sum-exp is
    evaluated over a symmetric lattice-image shell. The shell is enlarged
    until omitted image weights are below the hash-bound numerical tolerance,
    avoiding the minimum-image cut-locus force cusp.
    """

    points = np.asarray(point_positions_angstrom, dtype=float)
    centers = np.asarray(center_positions_angstrom, dtype=float)
    radii = np.asarray(center_radii_angstrom, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) == 0
        or centers.ndim != 2
        or centers.shape[1:] != (3,)
        or len(centers) == 0
        or radii.shape != (len(centers),)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError(
            "Surface points/centers must be finite non-empty (N, 3) arrays "
            "with one positive radius per center."
        )
    if (
        isinstance(surface_smoothing_angstrom, (bool, np.bool_))
        or not math.isfinite(float(surface_smoothing_angstrom))
        or float(surface_smoothing_angstrom) <= 0.0
    ):
        raise ValueError(
            "surface_smoothing_angstrom must be finite and positive."
        )
    smoothing = float(surface_smoothing_angstrom)

    periodic = cell_angstrom is not None or pbc is not None
    if periodic:
        if cell_angstrom is None or pbc is None:
            raise ValueError(
                "Periodic surface geometry requires both cell_angstrom and pbc."
            )
        cell = np.asarray(cell_angstrom, dtype=float)
        periodic_axes = np.asarray(pbc, dtype=bool)
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or abs(float(np.linalg.det(cell))) <= 1.0e-12
            or periodic_axes.shape != (3,)
            or not bool(np.all(periodic_axes))
        ):
            raise ValueError(
                "Periodic surface geometry requires a finite nonsingular "
                "cell and three-dimensional PBC."
            )
    else:
        cell = None
        periodic_axes = None

    displacement_vectors = points[:, None, :] - centers[None, :, :]
    periodic_image_shells = 0
    if periodic:
        singular_values = np.linalg.svd(cell, compute_uv=False)
        smallest_scale = float(np.min(singular_values))
        largest_scale = float(np.max(singular_values))
        nearest_distance_upper_bound = (
            0.5 * math.sqrt(3.0) * largest_scale - float(np.max(radii))
        )
        required_separation = (
            smoothing * PERIODIC_IMAGE_LOG_TOLERANCE
            + float(np.max(radii))
            + nearest_distance_upper_bound
        )
        periodic_image_shells = max(
            1,
            int(
                math.ceil(
                    required_separation / smallest_scale - 0.5
                )
            ),
        )
        if periodic_image_shells > PERIODIC_IMAGE_MAX_SHELLS:
            raise ValueError(
                "PERIODIC_IMAGE_SHELL_UNSUPPORTED: the cell, radii, and "
                "surface smoothing require more lattice images than the "
                "validated Route A adapter permits."
            )
        inverse_cell = np.linalg.inv(cell)
        fractional = displacement_vectors @ inverse_cell
        fractional -= np.floor(fractional + 0.5)
        wrapped_displacements = fractional @ cell
        image_range = np.arange(
            -periodic_image_shells,
            periodic_image_shells + 1,
            dtype=float,
        )
        image_shifts = np.stack(
            np.meshgrid(
                image_range,
                image_range,
                image_range,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        image_vectors = image_shifts @ cell
        displacement_vectors = (
            wrapped_displacements[:, :, None, :]
            + image_vectors[None, None, :, :]
        )
    else:
        displacement_vectors = displacement_vectors[:, :, None, :]
    distances = np.linalg.norm(displacement_vectors, axis=3)
    if np.any(distances <= 1.0e-12):
        raise RuntimeError(
            "SURFACE_GEOMETRY_SINGULAR: point coincides with a surface "
            "center."
        )

    signed_by_center = distances - radii[None, :, None]
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = -signed_by_center / smoothing
    if not np.all(np.isfinite(scaled)):
        raise ValueError(
            "Smooth-surface scaled distances exceed the finite range."
        )
    row_max = np.max(scaled, axis=(1, 2), keepdims=True)
    exponentials = np.exp(scaled - row_max)
    normalizers = np.sum(exponentials, axis=(1, 2), keepdims=True)
    image_weights = exponentials / normalizers
    signed_distances = -smoothing * (
        row_max[:, 0, 0] + np.log(normalizers[:, 0, 0])
    )
    unit_vectors = displacement_vectors / distances[..., None]
    center_weights = np.sum(image_weights, axis=2)
    center_gradient_vectors = np.sum(
        image_weights[..., None] * unit_vectors,
        axis=2,
    )

    immutable_values = []
    for value in (
        signed_distances,
        center_weights,
        center_gradient_vectors,
    ):
        array = np.array(value, dtype=float, copy=True, order="C")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(
                "SMOOTH_SURFACE_NONFINITE: geometry evaluation is non-finite."
            )
        array.setflags(write=False)
        immutable_values.append(array)
    return SmoothSurfaceGeometry(
        signed_distances_angstrom=immutable_values[0],
        center_weights=immutable_values[1],
        center_gradient_vectors=immutable_values[2],
        periodic_image_shells=periodic_image_shells,
    )


__all__ = [
    "PERIODIC_IMAGE_LOG_TOLERANCE",
    "PERIODIC_IMAGE_MAX_SHELLS",
    "SmoothSurfaceGeometry",
    "SoftCutoffEvaluation",
    "SoftCutoffMembership",
    "membership_surface_identity_hash",
    "smooth_surface_geometry",
    "soft_occupancy_weights",
]
