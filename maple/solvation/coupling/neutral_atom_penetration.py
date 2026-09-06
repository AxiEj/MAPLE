"""Analytic neutral-atom electrostatic penetration primitives.

This module converts one positive normalized spherical electron-density
Gaussian mixture into the electrostatic potential of the corresponding
neutral atom,

``Z / r - V_e(r) = sum_k N_k * erfc(sqrt(alpha_k) * r) / r``.

The correction is spherical, carries exactly zero total charge and dipole at
long range, and contains no fitted solvation parameter.  It is deliberately a
small mathematical primitive: selecting it for a Route-2 source profile is a
separate preregistered scientific decision.

Coordinates are in bohr and the returned potential is in atomic units
(``hartree / e``).  Coincident evaluation with the point nucleus is singular
and therefore fails closed rather than being clipped or regularized.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.special import erfc

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)

_MINIMUM_RADIUS_BOHR = 1.0e-12
NEUTRAL_ATOM_PENETRATION_ARTIFACT = (
    "route2-neutral-atom-penetration-gaussian-mixture-v1"
)
NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-neutral-atom-penetration-gaussian-mixture-v1.npz"
)
NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "route2-neutral-atom-penetration-gaussian-mixture-v1.json"
)
NEUTRAL_ATOM_PENETRATION_TABLE_SHA256 = (
    "0189bed53f03216c50c8f4e9f3cd7d5bbfcb4320b1e5a4672cd063691b6a5c06"
)
NEUTRAL_ATOM_PENETRATION_MANIFEST_SHA256 = (
    "5bae8bf23febf62fb48c5672269c2321ede0a3fc7966657ef17b28d934fc7963"
)
NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256 = (
    "98103440d7c88228d0c70ac59060be81647b7d858884908df071385ea9ec2d7a"
)
NEUTRAL_ATOM_PENETRATION_SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    return _canonical_sha256(
        {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": value.tobytes(order="C").hex(),
            }
            for key, value in arrays.items()
        }
    )


def load_neutral_atom_penetration_mixtures(
    *,
    table_path: str | Path,
    manifest_path: str | Path,
) -> Mapping[int, GaussianMixtureAtom]:
    """Load and fully verify the frozen neutral-atom penetration asset."""

    table = Path(table_path)
    manifest = Path(manifest_path)
    if _sha256(manifest) != NEUTRAL_ATOM_PENETRATION_MANIFEST_SHA256:
        raise ValueError("Neutral-atom penetration manifest SHA256 mismatch.")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("artifact") != NEUTRAL_ATOM_PENETRATION_ARTIFACT:
        raise ValueError("Neutral-atom penetration artifact identity is invalid.")
    if payload.get("schema_version") != 1 or payload.get("status") != (
        "atomic-density-and-analytic-kernel-only-solvation-gate-pending"
    ):
        raise ValueError("Neutral-atom penetration asset status is invalid.")
    declared_payload_hash = payload.pop("manifest_payload_sha256", None)
    if declared_payload_hash != _canonical_sha256(payload):
        raise ValueError("Neutral-atom penetration manifest payload hash is invalid.")
    table_record = payload.get("table")
    if table_record != {
        "path": NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH,
        "sha256": NEUTRAL_ATOM_PENETRATION_TABLE_SHA256,
        "content_sha256": NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256,
    }:
        raise ValueError("Neutral-atom penetration table record is invalid.")
    if _sha256(table) != NEUTRAL_ATOM_PENETRATION_TABLE_SHA256:
        raise ValueError("Neutral-atom penetration table SHA256 mismatch.")
    with np.load(table, allow_pickle=False) as archive:
        expected_keys = {
            "atomic_numbers",
            "component_offsets",
            "electron_counts",
            "gaussian_exponents_bohr2",
        }
        if set(archive.files) != expected_keys:
            raise ValueError("Neutral-atom penetration table schema is invalid.")
        arrays = {
            key: np.ascontiguousarray(archive[key])
            for key in (
                "atomic_numbers",
                "component_offsets",
                "electron_counts",
                "gaussian_exponents_bohr2",
            )
        }
    if _array_content_sha256(arrays) != NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256:
        raise ValueError("Neutral-atom penetration table content hash is invalid.")
    numbers = np.asarray(arrays["atomic_numbers"], dtype=np.int64)
    offsets = np.asarray(arrays["component_offsets"], dtype=np.int64)
    counts = np.asarray(arrays["electron_counts"], dtype=float)
    exponents = np.asarray(arrays["gaussian_exponents_bohr2"], dtype=float)
    if tuple(numbers.tolist()) != NEUTRAL_ATOM_PENETRATION_SUPPORTED_ATOMIC_NUMBERS:
        raise ValueError("Neutral-atom penetration element coverage is invalid.")
    if (
        offsets.shape != (len(numbers) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(counts)
        or np.any(np.diff(offsets) <= 0)
        or counts.shape != exponents.shape
        or not np.all(np.isfinite(counts))
        or not np.all(np.isfinite(exponents))
    ):
        raise ValueError("Neutral-atom penetration packed arrays are invalid.")
    mixtures = {}
    for index, atomic_number in enumerate(numbers):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        mixture = GaussianMixtureAtom(counts[start:stop], exponents[start:stop])
        if abs(mixture.electron_count - float(atomic_number)) > 5.0e-12:
            raise ValueError("Neutral-atom penetration mixture is not neutral.")
        mixtures[int(atomic_number)] = mixture
    return MappingProxyType(mixtures)


def _points(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] != 3
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be finite with shape (n, 3).")
    return result


def _atomic_numbers(values: object, *, atom_count: int) -> np.ndarray:
    raw = np.asarray(values)
    numeric = np.asarray(raw, dtype=float)
    if numeric.shape != (atom_count,) or not np.all(np.isfinite(numeric)):
        raise ValueError(f"atomic_numbers must have shape ({atom_count},).")
    rounded = np.rint(numeric)
    if np.any(rounded != numeric) or np.any(rounded < 1.0):
        raise ValueError("atomic_numbers must contain positive integers.")
    return rounded.astype(np.int64, copy=False)


def _mixture(
    mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom],
    atomic_number: int,
) -> GaussianMixtureAtom:
    try:
        result = mixtures_by_atomic_number[int(atomic_number)]
    except KeyError as exc:
        raise ValueError(
            f"No neutral-atom penetration mixture is available for Z={atomic_number}."
        ) from exc
    if not isinstance(result, GaussianMixtureAtom):
        raise TypeError("neutral-atom penetration entries must be GaussianMixtureAtom.")
    if abs(result.electron_count - float(atomic_number)) > 5.0e-12:
        raise ValueError(
            f"Neutral-atom mixture for Z={atomic_number} is not normalized to Z."
        )
    return result


def neutral_atom_penetration_potential_and_gradient(
    *,
    points_bohr: object,
    centers_bohr: object,
    atomic_numbers: object,
    mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom],
) -> tuple[np.ndarray, np.ndarray]:
    """Return neutral-atom penetration potential and spatial gradient.

    The gradient is with respect to each evaluation point in bohr and has
    shape ``(n_points, 3)``.  Atomic components are accumulated in the supplied
    center order and Gaussian components in their frozen mixture order, so the
    value does not depend on BLAS reduction scheduling.
    """

    points = _points(points_bohr, name="points_bohr")
    centers = _points(centers_bohr, name="centers_bohr")
    numbers = _atomic_numbers(atomic_numbers, atom_count=len(centers))
    potential = np.zeros(len(points), dtype=float)
    gradient = np.zeros((len(points), 3), dtype=float)

    for center, atomic_number in zip(centers, numbers, strict=True):
        mixture = _mixture(mixtures_by_atomic_number, int(atomic_number))
        displacement = points - center
        radius_squared = np.einsum("pi,pi->p", displacement, displacement)
        radius = np.sqrt(radius_squared)
        if np.any(radius <= _MINIMUM_RADIUS_BOHR):
            raise ValueError(
                "Neutral-atom penetration potential is singular at a nucleus."
            )
        inverse_radius = 1.0 / radius
        inverse_radius_squared = inverse_radius * inverse_radius
        inverse_radius_cubed = inverse_radius_squared * inverse_radius

        for electron_count, exponent in zip(
            mixture.electron_counts,
            mixture.gaussian_exponents_bohr2,
            strict=True,
        ):
            count = float(electron_count)
            root_exponent = math.sqrt(float(exponent))
            scaled_radius = root_exponent * radius
            complement = erfc(scaled_radius)
            gaussian = np.exp(-(scaled_radius * scaled_radius))
            potential += count * complement * inverse_radius
            radial_over_radius = -count * (
                (2.0 * root_exponent / math.sqrt(math.pi))
                * gaussian
                * inverse_radius_squared
                + complement * inverse_radius_cubed
            )
            gradient += displacement * radial_over_radius[:, None]

    if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("Neutral-atom penetration evaluation is non-finite.")
    potential.setflags(write=False)
    gradient.setflags(write=False)
    return potential, gradient


def neutral_atom_penetration_potential(
    *,
    points_bohr: object,
    centers_bohr: object,
    atomic_numbers: object,
    mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom],
) -> np.ndarray:
    """Return only the neutral-atom penetration potential."""

    potential, _gradient = neutral_atom_penetration_potential_and_gradient(
        points_bohr=points_bohr,
        centers_bohr=centers_bohr,
        atomic_numbers=atomic_numbers,
        mixtures_by_atomic_number=mixtures_by_atomic_number,
    )
    return potential


__all__ = [
    "NEUTRAL_ATOM_PENETRATION_ARTIFACT",
    "NEUTRAL_ATOM_PENETRATION_CONTENT_SHA256",
    "NEUTRAL_ATOM_PENETRATION_MANIFEST_REPO_PATH",
    "NEUTRAL_ATOM_PENETRATION_MANIFEST_SHA256",
    "NEUTRAL_ATOM_PENETRATION_SUPPORTED_ATOMIC_NUMBERS",
    "NEUTRAL_ATOM_PENETRATION_TABLE_REPO_PATH",
    "NEUTRAL_ATOM_PENETRATION_TABLE_SHA256",
    "load_neutral_atom_penetration_mixtures",
    "neutral_atom_penetration_potential",
    "neutral_atom_penetration_potential_and_gradient",
]
