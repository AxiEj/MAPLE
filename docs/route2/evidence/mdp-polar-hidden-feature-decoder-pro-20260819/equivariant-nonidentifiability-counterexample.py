"""Executable l=2 counterexample for hidden-feature density identifiability.

This is a finite-dimensional mathematical witness, not a fitted source model.
"""

from __future__ import annotations

import json
import math

import numpy as np


def fibonacci_sphere(count: int) -> np.ndarray:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = golden * index
        points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return np.asarray(points, dtype=np.float64)


def proper_rotation() -> np.ndarray:
    raw = np.asarray(
        [[0.31, -0.83, 0.47], [0.79, 0.02, -0.61], [0.52, 0.56, 0.64]],
        dtype=np.float64,
    )
    q, _ = np.linalg.qr(raw)
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def quadrupole_potential(points: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    radii = np.linalg.norm(points, axis=1)
    return 0.5 * np.einsum("ni,ij,nj->n", points, tensor, points) / radii**5


def main() -> None:
    # h2 is one hidden l=2 feature in symmetric-traceless Cartesian form.
    h2 = np.diag([-0.5, -0.5, 1.0])
    assert abs(np.trace(h2)) < 1.0e-15

    # D_0 ignores h2. D_t adds t*h2 through the identity l=2 intertwiner.
    # Both have q=1, p=0, and zero uniform-field derivative of p, hence the
    # same Q, molecular dipole, and polarizability in this counterexample.
    points = 2.3 * fibonacci_sphere(590)
    weights = np.full(points.shape[0], 4.0 * math.pi / points.shape[0])
    monopole = 1.0 / np.linalg.norm(points, axis=1)
    v0 = monopole
    v1 = monopole + quadrupole_potential(points, h2)

    # A=I is an invertible positive surface operator. It is enough to show that
    # two sources with identical public observables can give different values
    # of a legitimate stationary quadratic continuum scalar.
    g0 = -0.5 * float(np.dot(weights * v0, v0))
    g1 = -0.5 * float(np.dot(weights * v1, v1))

    rotation = proper_rotation()
    rotated_points = points @ rotation.T
    rotated_h2 = rotation @ h2 @ rotation.T
    rotated_v1 = 1.0 / np.linalg.norm(rotated_points, axis=1) + quadrupole_potential(
        rotated_points, rotated_h2
    )

    report = {
        "schema_version": 1,
        "same_total_charge": True,
        "same_molecular_dipole": True,
        "same_uniform_field_polarizability": True,
        "surface_mep_max_abs_difference": float(np.max(np.abs(v1 - v0))),
        "stationary_quadratic_energy_0": g0,
        "stationary_quadratic_energy_1": g1,
        "stationary_quadratic_energy_difference": g1 - g0,
        "rotation_covariance_max_abs_error": float(np.max(np.abs(rotated_v1 - v1))),
        "conclusion": (
            "distinct SO(3)-equivariant decoders preserve Q/mu/alpha but change "
            "surface MEP and a nondegenerate continuum scalar"
        ),
    }
    if report["surface_mep_max_abs_difference"] <= 1.0e-6:
        raise RuntimeError("Counterexample did not change the surface potential.")
    if abs(report["stationary_quadratic_energy_difference"]) <= 1.0e-6:
        raise RuntimeError("Counterexample did not change the continuum scalar.")
    if report["rotation_covariance_max_abs_error"] > 1.0e-12:
        raise RuntimeError("The l=2 decoder witness is not rotation covariant.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
