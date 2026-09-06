"""Canonical, zero-training atomic-density-translation source lift.

The frozen MACE-MDP checkpoint supplies one *molecular* polarizability tensor.
It does not determine an atomwise induced-source partition.  This module
provides the missing source-physical construction without using the checkpoint's
latent atomwise polarizability decomposition.

For each element, the same frozen positive spherical Gaussian-mixture density
used by the neutral-atom penetration asset defines an infinitesimal translation
tangent.  Atomic induced dipoles are allocated by minimizing the block-diagonal
vacuum Coulomb self-work of those tangents subject to reproducing the requested
molecular dipole.  For atom ``A`` the metric is isotropic,

``H_A = kappa_A I``.

The unique minimum-work allocation is therefore

``p_A = kappa_A**-1 / sum_B(kappa_B**-1) * p_mol``.

No response scale, geometry-local rank completion, fitted atomic weight, PCM
target, or solvation label enters this construction.  The result is an
operational source lift, not a common MACE/continuum variational functional and
not a capability admission.

All coordinates passed to the spatial kernels are in bohr.  Dipoles are in
``e bohr`` and potentials in ``hartree / e``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.special import erf

from maple.function.calculator.extra_correction.implicit.route2_atomic_reference_density import (
    GaussianMixtureAtom,
)
from maple.solvation.coupling.adt_radial_shape import ADTRadialShapeRegistry

CANONICAL_ADT_LIFT_CONTRACT = "route2-canonical-adt-free-atom-coulomb-self-work-lift-v1"
CANONICAL_ADT_LIFT_COUPLING_ID = (
    "route2-coupling-canonical-adt-free-atom-translation-v1"
)
ROLE_SEPARATED_ADT_LIFT_CONTRACT = (
    "route2-canonical-adt-role-separated-radial-shape-coulomb-self-work-lift-v2"
)
ROLE_SEPARATED_ADT_LIFT_COUPLING_ID = (
    "route2-coupling-canonical-adt-role-separated-radial-shape-translation-v2"
)

_SMALL_SCALED_RADIUS = 1.0e-3
_DIPOLE_CLOSURE_ATOL = 2.0e-13


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


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 digest.") from exc
    return value.lower()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _positive_atomic_numbers(values: object) -> np.ndarray:
    raw = np.asarray(values)
    numeric = np.asarray(raw, dtype=np.float64)
    if (
        numeric.ndim != 1
        or numeric.size == 0
        or not np.all(np.isfinite(numeric))
        or not np.array_equal(numeric, np.rint(numeric))
        or np.any(numeric < 1.0)
    ):
        raise ValueError("atomic_numbers must contain positive integers.")
    integer = np.ascontiguousarray(numeric, dtype=np.int64)
    return np.frombuffer(integer.tobytes(), dtype=np.int64)


def _points(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] != 3
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be finite with shape (n, 3).")
    return array


def _mixture_copy(
    mixtures: Mapping[int, GaussianMixtureAtom], atomic_number: int
) -> GaussianMixtureAtom:
    try:
        value = mixtures[int(atomic_number)]
    except KeyError as exc:
        raise ValueError(
            f"No frozen ADT Gaussian mixture exists for Z={atomic_number}."
        ) from exc
    if not isinstance(value, GaussianMixtureAtom):
        raise TypeError("ADT mixture entries must be GaussianMixtureAtom.")
    if abs(value.electron_count - float(atomic_number)) > 5.0e-12:
        raise ValueError(
            f"ADT Gaussian mixture for Z={atomic_number} is not normalized."
        )
    return GaussianMixtureAtom(
        np.asarray(value.electron_counts, dtype=np.float64),
        np.asarray(value.gaussian_exponents_bohr2, dtype=np.float64),
    )


def gaussian_translation_dipole_self_work(
    mixture: GaussianMixtureAtom,
) -> float:
    """Return ``kappa`` in ``0.5 * kappa * |p|**2`` for one atom.

    The result follows from the Fourier-space Coulomb quadratic form of the
    translated spherical density.  If

    ``n(k) = sum_a N_a exp(-k**2/(4 alpha_a))``

    and the translation tangent is normalized to molecular dipole ``p``, then

    ``kappa = 4/(3 sqrt(pi) Z**2) * sum_ab N_a N_b
              * (alpha_a alpha_b/(alpha_a+alpha_b))**(3/2)``.
    """

    if not isinstance(mixture, GaussianMixtureAtom):
        raise TypeError("mixture must be GaussianMixtureAtom.")
    counts = np.asarray(mixture.electron_counts, dtype=np.float64)
    exponents = np.asarray(mixture.gaussian_exponents_bohr2, dtype=np.float64)
    electron_count = float(np.sum(counts))
    if not math.isfinite(electron_count) or electron_count <= 0.0:
        raise ValueError("mixture electron count must be finite and positive.")
    pair_scale = (
        exponents[:, None]
        * exponents[None, :]
        / (exponents[:, None] + exponents[None, :])
    ) ** 1.5
    result = (
        4.0
        / (3.0 * math.sqrt(math.pi) * electron_count**2)
        * float(np.sum(counts[:, None] * counts[None, :] * pair_scale))
    )
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("Gaussian translation self-work is not positive finite.")
    return result


def gaussian_enclosed_electrons_over_r3(
    mixture: GaussianMixtureAtom, radius_bohr: object
) -> np.ndarray:
    """Return ``N(r)/r**3`` with its analytic finite ``r -> 0`` limit."""

    if not isinstance(mixture, GaussianMixtureAtom):
        raise TypeError("mixture must be GaussianMixtureAtom.")
    radius = np.asarray(radius_bohr, dtype=np.float64)
    if not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius_bohr must be finite and nonnegative.")
    result = np.zeros_like(radius, dtype=np.float64)
    for count, exponent in zip(
        mixture.electron_counts,
        mixture.gaussian_exponents_bohr2,
        strict=True,
    ):
        alpha = float(exponent)
        root_alpha = math.sqrt(alpha)
        scaled = root_alpha * radius
        small = scaled < _SMALL_SCALED_RADIUS
        regular = ~small
        if np.any(regular):
            x = scaled[regular]
            enclosed = erf(x) - (2.0 / math.sqrt(math.pi)) * x * np.exp(-(x * x))
            result[regular] += float(count) * enclosed / radius[regular] ** 3
        if np.any(small):
            r2 = radius[small] ** 2
            # erf(x) - 2*x*exp(-x^2)/sqrt(pi), divided by r^3.
            series = (2.0 / math.sqrt(math.pi)) * (
                (2.0 / 3.0) * alpha**1.5
                - (2.0 / 5.0) * alpha**2.5 * r2
                + (1.0 / 7.0) * alpha**3.5 * r2**2
                - (1.0 / 27.0) * alpha**4.5 * r2**3
            )
            result[small] += float(count) * series
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise RuntimeError("enclosed-electron radial factor is invalid.")
    return result


def gaussian_electron_potential(
    mixture: GaussianMixtureAtom, displacement_bohr: object
) -> np.ndarray:
    """Return the positive electron-density Coulomb potential ``V_e``."""

    displacement = _points(displacement_bohr, name="displacement_bohr")
    radius = np.linalg.norm(displacement, axis=1)
    result = np.empty(len(radius), dtype=np.float64)
    nonzero = radius > 0.0
    result[~nonzero] = sum(
        float(count) * 2.0 * math.sqrt(float(exponent) / math.pi)
        for count, exponent in zip(
            mixture.electron_counts,
            mixture.gaussian_exponents_bohr2,
            strict=True,
        )
    )
    if np.any(nonzero):
        values = np.zeros(np.count_nonzero(nonzero), dtype=np.float64)
        r = radius[nonzero]
        for count, exponent in zip(
            mixture.electron_counts,
            mixture.gaussian_exponents_bohr2,
            strict=True,
        ):
            values += float(count) * erf(math.sqrt(float(exponent)) * r) / r
        result[nonzero] = values
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Gaussian electron potential is non-finite.")
    return result


@dataclass(frozen=True, slots=True)
class CanonicalAtomicDisplacementLift:
    """Immutable minimum-self-work lift of a molecular induced dipole."""

    atomic_numbers: np.ndarray
    mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom]
    source_asset_sha256: str
    self_work_hartree_per_ebohr2: np.ndarray
    atomic_dipole_weights: np.ndarray
    effective_electron_counts_by_atomic_number: Mapping[int, float] | None = None
    radial_shape_registry_sha256: str | None = None
    configuration_sha256: str = ""

    def __post_init__(self) -> None:
        numbers = _positive_atomic_numbers(self.atomic_numbers)
        asset = _digest(self.source_asset_sha256, name="source_asset_sha256")
        role_separated = self.radial_shape_registry_sha256 is not None
        if role_separated != (
            self.effective_electron_counts_by_atomic_number is not None
        ):
            raise ValueError(
                "role-separated ADT requires both effective counts and registry SHA256."
            )
        unique_numbers = sorted(set(numbers.tolist()))
        if role_separated:
            registry_sha256 = _digest(
                self.radial_shape_registry_sha256,
                name="radial_shape_registry_sha256",
            )
            effective_values = self.effective_electron_counts_by_atomic_number
            if not isinstance(effective_values, Mapping):
                raise TypeError(
                    "effective_electron_counts_by_atomic_number must be a mapping."
                )
            frozen_mixtures: dict[int, GaussianMixtureAtom] = {}
            effective_counts: dict[int, float] = {}
            for number in unique_numbers:
                try:
                    mixture = self.mixtures_by_atomic_number[number]
                    effective = float(effective_values[number])
                except KeyError as exc:
                    raise ValueError(
                        f"No role-separated ADT radial shape exists for Z={number}."
                    ) from exc
                if not isinstance(mixture, GaussianMixtureAtom):
                    raise TypeError("ADT mixture entries must be GaussianMixtureAtom.")
                if not math.isfinite(effective) or effective <= 0.0:
                    raise ValueError(
                        "ADT effective electron counts must be positive finite."
                    )
                if abs(mixture.electron_count - effective) > 5.0e-12:
                    raise ValueError(
                        f"ADT radial shape for Z={number} is not normalized to N_eff."
                    )
                frozen_mixtures[number] = GaussianMixtureAtom(
                    np.asarray(mixture.electron_counts, dtype=np.float64),
                    np.asarray(mixture.gaussian_exponents_bohr2, dtype=np.float64),
                )
                effective_counts[number] = effective
        else:
            registry_sha256 = None
            frozen_mixtures = {
                number: _mixture_copy(self.mixtures_by_atomic_number, number)
                for number in unique_numbers
            }
            effective_counts = {number: float(number) for number in unique_numbers}
        expected_self_work = np.asarray(
            [
                gaussian_translation_dipole_self_work(frozen_mixtures[int(number)])
                for number in numbers
            ],
            dtype=np.float64,
        )
        self_work = _readonly(
            self.self_work_hartree_per_ebohr2,
            shape=(len(numbers),),
            name="self_work_hartree_per_ebohr2",
        )
        if not np.allclose(self_work, expected_self_work, rtol=2.0e-14, atol=0.0):
            raise ValueError(
                "ADT self-work does not match the frozen Gaussian mixtures."
            )
        inverse = 1.0 / self_work
        expected_weights = inverse / float(np.sum(inverse))
        weights = _readonly(
            self.atomic_dipole_weights,
            shape=(len(numbers),),
            name="atomic_dipole_weights",
        )
        if (
            np.any(weights <= 0.0)
            or not np.allclose(weights, expected_weights, rtol=2.0e-14, atol=0.0)
            or abs(float(np.sum(weights)) - 1.0) > _DIPOLE_CLOSURE_ATOL
        ):
            raise ValueError("ADT weights are not the unique minimum-self-work lift.")
        mixture_payload = {
            str(number): {
                "electron_counts": frozen_mixtures[number].electron_counts.tolist(),
                "gaussian_exponents_bohr2": (
                    frozen_mixtures[number].gaussian_exponents_bohr2.tolist()
                ),
            }
            for number in frozen_mixtures
        }
        if role_separated:
            expected = _canonical_sha256(
                {
                    "contract": ROLE_SEPARATED_ADT_LIFT_CONTRACT,
                    "coupling_id": ROLE_SEPARATED_ADT_LIFT_COUPLING_ID,
                    "source_asset_sha256": asset,
                    "radial_shape_registry_sha256": registry_sha256,
                    "atomic_numbers": numbers.tolist(),
                    "effective_electron_counts": {
                        str(number): effective_counts[number]
                        for number in unique_numbers
                    },
                    "mixtures": mixture_payload,
                    "self_work_hartree_per_ebohr2": self_work.tolist(),
                    "atomic_dipole_weights": weights.tolist(),
                    "allocation": "unique-block-coulomb-self-work-minimum",
                    "radial_shape_role": "adt_translation_shape",
                    "neutral_penetration_semantics": "not-implied",
                    "capabilities": "none",
                }
            )
        else:
            # Keep the historical v1 payload exact.  Its hash is part of prior
            # non-iodine evidence and must not change merely because v2 exists.
            expected = _canonical_sha256(
                {
                    "contract": CANONICAL_ADT_LIFT_CONTRACT,
                    "coupling_id": CANONICAL_ADT_LIFT_COUPLING_ID,
                    "source_asset_sha256": asset,
                    "atomic_numbers": numbers.tolist(),
                    "mixtures": mixture_payload,
                    "self_work_hartree_per_ebohr2": self_work.tolist(),
                    "atomic_dipole_weights": weights.tolist(),
                    "allocation": "unique-block-coulomb-self-work-minimum",
                    "capabilities": "none",
                }
            )
        if self.configuration_sha256 and self.configuration_sha256 != expected:
            raise ValueError("configuration_sha256 does not match the ADT lift.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(
            self, "mixtures_by_atomic_number", MappingProxyType(frozen_mixtures)
        )
        object.__setattr__(self, "source_asset_sha256", asset)
        object.__setattr__(self, "self_work_hartree_per_ebohr2", self_work)
        object.__setattr__(self, "atomic_dipole_weights", weights)
        object.__setattr__(
            self,
            "effective_electron_counts_by_atomic_number",
            MappingProxyType(effective_counts),
        )
        object.__setattr__(self, "radial_shape_registry_sha256", registry_sha256)
        object.__setattr__(self, "configuration_sha256", expected)

    @classmethod
    def from_mixtures(
        cls,
        *,
        atomic_numbers: object,
        mixtures_by_atomic_number: Mapping[int, GaussianMixtureAtom],
        source_asset_sha256: str,
    ) -> "CanonicalAtomicDisplacementLift":
        numbers = _positive_atomic_numbers(atomic_numbers)
        frozen = {
            number: _mixture_copy(mixtures_by_atomic_number, number)
            for number in sorted(set(numbers.tolist()))
        }
        self_work = np.asarray(
            [
                gaussian_translation_dipole_self_work(frozen[int(number)])
                for number in numbers
            ],
            dtype=np.float64,
        )
        inverse = 1.0 / self_work
        return cls(
            atomic_numbers=numbers,
            mixtures_by_atomic_number=frozen,
            source_asset_sha256=source_asset_sha256,
            self_work_hartree_per_ebohr2=self_work,
            atomic_dipole_weights=inverse / float(np.sum(inverse)),
        )

    @classmethod
    def from_radial_shapes(
        cls,
        *,
        atomic_numbers: object,
        registry: ADTRadialShapeRegistry,
    ) -> "CanonicalAtomicDisplacementLift":
        """Build the v2 lift from explicitly role-separated ADT shapes."""

        if not isinstance(registry, ADTRadialShapeRegistry):
            raise TypeError("registry must be ADTRadialShapeRegistry.")
        numbers = _positive_atomic_numbers(atomic_numbers)
        shapes = {
            number: registry.shape(number) for number in sorted(set(numbers.tolist()))
        }
        mixtures = {number: shape.mixture for number, shape in shapes.items()}
        effective = {
            number: shape.effective_electron_count for number, shape in shapes.items()
        }
        self_work = np.asarray(
            [
                gaussian_translation_dipole_self_work(mixtures[int(number)])
                for number in numbers
            ],
            dtype=np.float64,
        )
        inverse = 1.0 / self_work
        return cls(
            atomic_numbers=numbers,
            mixtures_by_atomic_number=mixtures,
            source_asset_sha256=registry.configuration_sha256,
            self_work_hartree_per_ebohr2=self_work,
            atomic_dipole_weights=inverse / float(np.sum(inverse)),
            effective_electron_counts_by_atomic_number=effective,
            radial_shape_registry_sha256=registry.configuration_sha256,
        )

    @property
    def contract_id(self) -> str:
        if self.radial_shape_registry_sha256 is None:
            return CANONICAL_ADT_LIFT_CONTRACT
        return ROLE_SEPARATED_ADT_LIFT_CONTRACT

    @property
    def coupling_id(self) -> str:
        if self.radial_shape_registry_sha256 is None:
            return CANONICAL_ADT_LIFT_COUPLING_ID
        return ROLE_SEPARATED_ADT_LIFT_COUPLING_ID

    @property
    def atom_count(self) -> int:
        return int(len(self.atomic_numbers))

    @property
    def metric_matrix(self) -> np.ndarray:
        result = np.kron(
            np.diag(self.self_work_hartree_per_ebohr2), np.eye(3, dtype=np.float64)
        )
        result.setflags(write=False)
        return result

    @property
    def molecular_dipole_map(self) -> np.ndarray:
        result = np.tile(np.eye(3, dtype=np.float64), (1, self.atom_count))
        result.setflags(write=False)
        return result

    @property
    def minimum_metric_right_inverse(self) -> np.ndarray:
        result = np.vstack(
            [
                weight * np.eye(3, dtype=np.float64)
                for weight in self.atomic_dipole_weights
            ]
        )
        result.setflags(write=False)
        return result

    def lift_molecular_dipole(self, molecular_dipole_ebohr: object) -> np.ndarray:
        dipole = _readonly(
            molecular_dipole_ebohr,
            shape=(3,),
            name="molecular_dipole_ebohr",
        )
        result = np.einsum("a,i->ai", self.atomic_dipole_weights, dipole, optimize=True)
        if not np.allclose(np.sum(result, axis=0), dipole, rtol=0.0, atol=2.0e-14):
            raise RuntimeError("canonical ADT lift did not preserve molecular dipole.")
        result.setflags(write=False)
        return result

    def tangent_potential(
        self,
        *,
        points_bohr: object,
        centers_bohr: object,
        atomic_dipoles_ebohr: object,
    ) -> np.ndarray:
        """Return the exact analytic free-atom translation tangent potential."""

        points = _points(points_bohr, name="points_bohr")
        centers = _points(centers_bohr, name="centers_bohr")
        if len(centers) != self.atom_count:
            raise ValueError("centers_bohr atom count does not match the ADT lift.")
        dipoles = _readonly(
            atomic_dipoles_ebohr,
            shape=(self.atom_count, 3),
            name="atomic_dipoles_ebohr",
        )
        potential = np.zeros(len(points), dtype=np.float64)
        for center, number, dipole in zip(
            centers, self.atomic_numbers, dipoles, strict=True
        ):
            displacement = points - center
            radius = np.linalg.norm(displacement, axis=1)
            radial = (
                gaussian_enclosed_electrons_over_r3(
                    self.mixtures_by_atomic_number[int(number)], radius
                )
                / self.effective_electron_counts_by_atomic_number[int(number)]
            )
            potential += np.einsum("pi,i->p", displacement, dipole) * radial
        if not np.all(np.isfinite(potential)):
            raise RuntimeError("ADT tangent potential is non-finite.")
        potential.setflags(write=False)
        return potential

    def exact_translated_potential_difference(
        self,
        *,
        points_bohr: object,
        centers_bohr: object,
        atomic_dipoles_ebohr: object,
    ) -> np.ndarray:
        """Return the finite electron-cloud translation relative to zero.

        The displacement associated with atomic dipole ``p_A`` is
        ``d_A = -p_A/N_eff,A``.  For the historical all-electron shapes
        ``N_eff,A = Z_A``; an explicitly role-separated ECP valence shape uses
        its positive effective electron count instead.  Nuclei remain fixed.
        This method exists only for the target-free tangent-support gate;
        production response remains the analytic tangent.
        """

        points = _points(points_bohr, name="points_bohr")
        centers = _points(centers_bohr, name="centers_bohr")
        if len(centers) != self.atom_count:
            raise ValueError("centers_bohr atom count does not match the ADT lift.")
        dipoles = _readonly(
            atomic_dipoles_ebohr,
            shape=(self.atom_count, 3),
            name="atomic_dipoles_ebohr",
        )
        difference = np.zeros(len(points), dtype=np.float64)
        for center, number, dipole in zip(
            centers, self.atomic_numbers, dipoles, strict=True
        ):
            mixture = self.mixtures_by_atomic_number[int(number)]
            displacement = (
                -dipole / self.effective_electron_counts_by_atomic_number[int(number)]
            )
            original = gaussian_electron_potential(mixture, points - center)
            translated = gaussian_electron_potential(
                mixture, points - (center + displacement)
            )
            # Electron charge is negative: -V_e(translated) - [-V_e(original)].
            difference += original - translated
        if not np.all(np.isfinite(difference)):
            raise RuntimeError("finite ADT translation potential is non-finite.")
        difference.setflags(write=False)
        return difference

    def atomic_surface_operator(
        self, *, points_bohr: object, centers_bohr: object
    ) -> np.ndarray:
        """Return ``D_phi`` from atom Cartesian dipoles to surface MEP."""

        points = _points(points_bohr, name="points_bohr")
        centers = _points(centers_bohr, name="centers_bohr")
        if len(centers) != self.atom_count:
            raise ValueError("centers_bohr atom count does not match the ADT lift.")
        operator = np.empty((len(points), self.atom_count * 3), dtype=np.float64)
        for column in range(self.atom_count * 3):
            dipoles = np.zeros((self.atom_count, 3), dtype=np.float64)
            dipoles.reshape(-1)[column] = 1.0
            operator[:, column] = self.tangent_potential(
                points_bohr=points,
                centers_bohr=centers,
                atomic_dipoles_ebohr=dipoles,
            )
        operator.setflags(write=False)
        return operator


__all__ = [
    "CANONICAL_ADT_LIFT_CONTRACT",
    "CANONICAL_ADT_LIFT_COUPLING_ID",
    "ROLE_SEPARATED_ADT_LIFT_CONTRACT",
    "ROLE_SEPARATED_ADT_LIFT_COUPLING_ID",
    "CanonicalAtomicDisplacementLift",
    "gaussian_electron_potential",
    "gaussian_enclosed_electrons_over_r3",
    "gaussian_translation_dipole_self_work",
]
