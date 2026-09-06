"""Cartesian atomic-multipole electrostatic potentials for source audits.

The tensors accepted here are *raw Cartesian moments* of atom-centred net
charge distributions, not traceless chemistry quadrupoles.  Through octupole
order the potential of one atom-centred distribution is the Cartesian Taylor
expansion

``q/r + p_i r_i/r^3 + Q_ij (3 r_i r_j-r^2 delta_ij)/(2 r^5) + ...``.

All inputs are converted to atomic units before the contraction and the result
is Hartree per positive elementary test charge.  The implementation is used by
target-independent physical source diagnostics; it does not admit a solvation
energy or force capability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.units import Bohr


def _finite_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "" if shape is None else f" with shape {shape}"
        raise ValueError(f"{name} must be finite{expected}.")
    return np.ascontiguousarray(array)


def cartesian_atomic_multipole_potential(
    *,
    points_bohr: object,
    centers_angstrom: object,
    charges_e: object,
    dipoles_eangstrom: object,
    quadrupoles_eangstrom2: object | None = None,
    octupoles_eangstrom3: object | None = None,
    minimum_distance_bohr: float = 1.0e-8,
) -> np.ndarray:
    """Return the raw-Cartesian multipole MEP through optional octupoles.

    ``quadrupoles_eangstrom2`` and ``octupoles_eangstrom3`` must be full raw
    tensors with shapes ``(N,3,3)`` and ``(N,3,3,3)``.  No symmetrisation or
    traceless conversion is silently applied.  This matches the SPICE MBIS
    tensor storage contract and keeps convention errors observable.
    """

    points = _finite_array(points_bohr, name="points_bohr")
    centers = _finite_array(centers_angstrom, name="centers_angstrom")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_bohr must have nonempty shape (M,3).")
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("centers_angstrom must have nonempty shape (N,3).")
    atom_count = len(centers)
    charges = _finite_array(charges_e, name="charges_e", shape=(atom_count,))
    dipoles = _finite_array(
        dipoles_eangstrom,
        name="dipoles_eangstrom",
        shape=(atom_count, 3),
    )
    quadrupoles = (
        None
        if quadrupoles_eangstrom2 is None
        else _finite_array(
            quadrupoles_eangstrom2,
            name="quadrupoles_eangstrom2",
            shape=(atom_count, 3, 3),
        )
    )
    octupoles = (
        None
        if octupoles_eangstrom3 is None
        else _finite_array(
            octupoles_eangstrom3,
            name="octupoles_eangstrom3",
            shape=(atom_count, 3, 3, 3),
        )
    )
    threshold = float(minimum_distance_bohr)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_distance_bohr must be finite and positive.")

    centers_bohr = centers / Bohr
    dipoles_bohr = dipoles / Bohr
    quadrupoles_bohr2 = None if quadrupoles is None else quadrupoles / Bohr**2
    octupoles_bohr3 = None if octupoles is None else octupoles / Bohr**3
    potential = np.zeros(len(points), dtype=np.float64)

    for atom_index in range(atom_count):
        displacement = points - centers_bohr[atom_index]
        radius2 = np.einsum("pi,pi->p", displacement, displacement)
        if np.any(radius2 < threshold**2):
            raise ValueError("Evaluation point is too close to a multipole centre.")
        inverse_radius = 1.0 / np.sqrt(radius2)
        inverse_radius3 = inverse_radius**3
        potential += charges[atom_index] * inverse_radius
        potential += (displacement @ dipoles_bohr[atom_index]) * inverse_radius3

        if quadrupoles_bohr2 is not None:
            quadrupole = quadrupoles_bohr2[atom_index]
            d_q_d = np.einsum("pi,ij,pj->p", displacement, quadrupole, displacement)
            potential += (
                0.5 * (3.0 * d_q_d - radius2 * np.trace(quadrupole)) * inverse_radius**5
            )

        if octupoles_bohr3 is not None:
            octupole = octupoles_bohr3[atom_index]
            cubic = np.einsum(
                "pi,pj,pk,ijk->p",
                displacement,
                displacement,
                displacement,
                octupole,
                optimize=True,
            )
            trace_ij = np.einsum("iik->k", octupole)
            trace_ik = np.einsum("iji->j", octupole)
            trace_jk = np.einsum("ijj->i", octupole)
            trace_term = displacement @ (trace_ij + trace_ik + trace_jk)
            potential += (
                15.0 * cubic * inverse_radius**7 - 3.0 * trace_term * inverse_radius**5
            ) / 6.0

    if not np.all(np.isfinite(potential)):
        raise RuntimeError("Multipole potential is non-finite.")
    return potential


@dataclass(frozen=True, slots=True)
class WeightedMEPMetrics:
    """Area-weighted MEP error metrics on one shared point set."""

    root_mean_square_error_hartree_per_e: float
    mean_absolute_error_hartree_per_e: float
    maximum_absolute_error_hartree_per_e: float
    reference_root_mean_square_hartree_per_e: float
    relative_root_mean_square_error: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.root_mean_square_error_hartree_per_e,
                self.mean_absolute_error_hartree_per_e,
                self.maximum_absolute_error_hartree_per_e,
                self.reference_root_mean_square_hartree_per_e,
                self.relative_root_mean_square_error,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Weighted MEP metrics must be finite and non-negative.")

    def as_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field)) for field in self.__dataclass_fields__
        }


def weighted_mep_metrics(
    predicted_hartree_per_e: object,
    reference_hartree_per_e: object,
    area_weights_bohr2: object,
) -> WeightedMEPMetrics:
    """Return normalized positive-area-weighted MEP error metrics."""

    predicted = _finite_array(predicted_hartree_per_e, name="predicted")
    reference = _finite_array(reference_hartree_per_e, name="reference")
    weights = _finite_array(area_weights_bohr2, name="area_weights_bohr2")
    if (
        predicted.ndim != 1
        or reference.shape != predicted.shape
        or weights.shape != predicted.shape
        or len(predicted) == 0
        or np.any(weights <= 0.0)
    ):
        raise ValueError("MEPs and positive area weights must share shape (M,).")
    normalized = weights / np.sum(weights)
    error = predicted - reference
    rmse = float(np.sqrt(np.sum(normalized * error**2)))
    reference_rms = float(np.sqrt(np.sum(normalized * reference**2)))
    return WeightedMEPMetrics(
        root_mean_square_error_hartree_per_e=rmse,
        mean_absolute_error_hartree_per_e=float(np.sum(normalized * np.abs(error))),
        maximum_absolute_error_hartree_per_e=float(np.max(np.abs(error))),
        reference_root_mean_square_hartree_per_e=reference_rms,
        relative_root_mean_square_error=rmse
        / max(reference_rms, np.finfo(np.float64).tiny),
    )


__all__ = [
    "WeightedMEPMetrics",
    "cartesian_atomic_multipole_potential",
    "weighted_mep_metrics",
]
