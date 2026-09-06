"""Fixed-width Gaussian traceless-quadrupole electrostatic kernel.

For the normalized Gaussian monopole kernel

``f(r) = erf(r/(sqrt(2)*sigma)) / r``,

the potential of a raw symmetric-traceless second moment is

``1/2 Q_ij d_i d_j f``.

The five coefficient channels use the repository's orthonormal STF Cartesian
basis and physical units ``e*angstrom^2``.  NumPy and Torch implementations
share the same formula; the Torch path is differentiable with respect to probe
and centre coordinates.
"""

from __future__ import annotations

import math

import numpy as np
from ase.units import Bohr
from scipy.special import erf

from .point_quadrupole import TRACELESS_QUADRUPOLE_BASIS


def _radial_l2_kernel_numpy(radius: np.ndarray, sigma_bohr: float) -> np.ndarray:
    a = 1.0 / (math.sqrt(2.0) * sigma_bohr)
    ar = a * radius
    exponential = np.exp(-(ar * ar))
    error_function = erf(ar)
    c = 2.0 * a / math.sqrt(math.pi)
    inverse = 1.0 / radius
    first = c * exponential * inverse - error_function * inverse**2
    second = (
        -2.0 * a * a * c * exponential
        - 2.0 * c * exponential * inverse**2
        + 2.0 * error_function * inverse**3
    )
    return (second - first * inverse) * inverse**2


def gaussian_traceless_quadrupole_surface_operator(
    *,
    points_bohr: object,
    centers_angstrom: object,
    sigma_angstrom: float,
    minimum_distance_bohr: float = 1.0e-6,
) -> np.ndarray:
    points = np.asarray(points_bohr, dtype=np.float64)
    centers = np.asarray(centers_angstrom, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("points_bohr must have finite nonempty shape (M,3).")
    if (
        centers.ndim != 2
        or centers.shape[0] == 0
        or centers.shape[1] != 3
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError("centers_angstrom must have finite nonempty shape (N,3).")
    sigma = float(sigma_angstrom)
    threshold = float(minimum_distance_bohr)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_angstrom must be finite and positive.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_distance_bohr must be finite and positive.")
    displacement = points[:, None, :] - centers[None, :, :] / Bohr
    radius = np.linalg.norm(displacement, axis=2)
    if np.any(radius <= threshold):
        raise ValueError("A Gaussian-l2 probe is below the separation guard.")
    contraction = np.einsum(
        "pni,aij,pnj->pna",
        displacement,
        TRACELESS_QUADRUPOLE_BASIS,
        displacement,
        optimize=True,
    )
    radial = _radial_l2_kernel_numpy(radius, sigma / Bohr)
    operator = 0.5 * contraction * radial[:, :, None] / Bohr**2
    result = np.ascontiguousarray(operator.reshape(len(points), -1))
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Gaussian-l2 surface operator is nonfinite.")
    return result


def gaussian_traceless_quadrupole_potential_torch(
    *,
    points_bohr,
    centers_angstrom,
    coefficients_eangstrom2,
    sigma_angstrom: float,
    minimum_distance_bohr: float = 1.0e-6,
):
    torch = __import__("torch")
    values = (points_bohr, centers_angstrom, coefficients_eangstrom2)
    if not all(torch.is_tensor(value) and torch.is_floating_point(value) for value in values):
        raise TypeError("Gaussian-l2 Torch inputs must be floating tensors.")
    reference = points_bohr
    if not all(
        value.dtype == reference.dtype and value.device == reference.device
        for value in values
    ):
        raise ValueError("Gaussian-l2 Torch inputs must share dtype and device.")
    atom_count = int(centers_angstrom.shape[0]) if centers_angstrom.ndim == 2 else -1
    if (
        points_bohr.ndim != 2
        or points_bohr.shape[0] == 0
        or points_bohr.shape[1] != 3
        or tuple(centers_angstrom.shape) != (atom_count, 3)
        or tuple(coefficients_eangstrom2.shape) != (atom_count, 5)
        or atom_count < 1
        or not all(bool(torch.isfinite(value).all()) for value in values)
    ):
        raise ValueError("Gaussian-l2 Torch input shapes or values are invalid.")
    sigma = float(sigma_angstrom)
    threshold = float(minimum_distance_bohr)
    if not math.isfinite(sigma) or sigma <= 0.0 or not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Gaussian-l2 width and separation guard must be positive.")
    bohr = torch.as_tensor(Bohr, dtype=reference.dtype, device=reference.device)
    displacement = points_bohr[:, None, :] - centers_angstrom[None, :, :] / bohr
    radius = torch.linalg.vector_norm(displacement, dim=2)
    if bool(torch.any(radius <= threshold)):
        raise ValueError("A Gaussian-l2 probe is below the separation guard.")
    basis = torch.as_tensor(
        np.array(TRACELESS_QUADRUPOLE_BASIS, copy=True),
        dtype=reference.dtype,
        device=reference.device,
    )
    tensors = torch.einsum("na,aij->nij", coefficients_eangstrom2, basis)
    contraction = torch.einsum(
        "pni,nij,pnj->pn",
        displacement,
        tensors,
        displacement,
    )
    sigma_bohr = torch.as_tensor(
        sigma,
        dtype=reference.dtype,
        device=reference.device,
    ) / bohr
    a = 1.0 / (math.sqrt(2.0) * sigma_bohr)
    ar = a * radius
    exponential = torch.exp(-(ar * ar))
    error_function = torch.erf(ar)
    c = 2.0 * a / math.sqrt(math.pi)
    inverse = radius.reciprocal()
    first = c * exponential * inverse - error_function * inverse.square()
    second = (
        -2.0 * a.square() * c * exponential
        - 2.0 * c * exponential * inverse.square()
        + 2.0 * error_function * inverse.pow(3)
    )
    radial = (second - first * inverse) * inverse.square()
    return 0.5 * torch.sum(contraction * radial, dim=1) / bohr.square()


__all__ = [
    "gaussian_traceless_quadrupole_potential_torch",
    "gaussian_traceless_quadrupole_surface_operator",
]
