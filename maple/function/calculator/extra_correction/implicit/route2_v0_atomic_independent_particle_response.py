"""Frozen atomic independent-particle response assets for Route-2 V0-RK.

This module contains the *source-side* construction of a full, neutral
baseline response kernel.  A spherical atomic Hartree--Fock calculation gives
canonical orbitals ``phi_p`` with occupations ``n_p``.  For every occupied to
less-occupied pair ``p < q`` it stores the neutral transition density

``tau_pq(r) = phi_p(r) phi_q(r) + phi_q(r) phi_p(r)``.

Its independent-particle static response coefficient is

``w_pq = (n_p - n_q) / (2 * (epsilon_q - epsilon_p)) > 0``.

The coefficient/dual pairing is ``f_pq = integral tau_pq(r) V(r) dr`` and the
induced density amplitude is ``x_pq = -w_pq f_pq``.  Thus the baseline
covariance is diagonal and positive, while every mode is intrinsically charge
neutral by orbital orthogonality.  A later V0-RK Schur-complement completion
can replace only the atom-dipole covariance with the frozen MACE-MDP moment
covariance without fitting radial response modes.

The asset is deliberately not a molecular response model, a continuum, a
solute energy, or a solvation result.  Isolated-atom independent-particle
response omits molecular screening, bonding, and charge transfer.  It must
therefore pass the preregistered multi-molecule QM spatial-response gates
before it can be admitted to a common scalar or any accuracy calculation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np


V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_ARTIFACT = (
    "route2-v0-atomic-independent-particle-hf-def2-tzvpd-v1"
)
V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_CONSTRUCTION = (
    "spherical-atomic-hf-independent-particle-transition-density-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _immutable_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
    ):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _immutable_float_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    return _immutable_array(np.asarray(values, dtype=float), name=name, shape=shape)


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive scalar.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _integer_vector(values: np.ndarray, *, name: str, length: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (length,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be finite with shape ({length},).")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric):
        raise ValueError(f"{name} must contain exact integer indices.")
    result = np.asarray(rounded, dtype=np.int64)
    result.setflags(write=False)
    return result


def _validate_atomic_number(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Atomic number must be a positive integer.")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError("Atomic number must be a positive integer.")
    return result


@dataclass(frozen=True)
class Route2V0AtomicIndependentParticleResponse:
    """One frozen spherical-atom independent-particle response basis.

    The coefficient coordinates are amplitudes of the stored transition
    densities.  They are intrinsically neutral, so downstream induced-response
    KKT systems must not invent a redundant total-charge coordinate.
    """

    atomic_number: int
    symbol: str
    basis: str
    spin_2s: int
    mo_coefficients: np.ndarray
    orbital_energies_hartree: np.ndarray
    orbital_occupations: np.ndarray
    transition_lower_indices: np.ndarray
    transition_upper_indices: np.ndarray
    transition_weights_hartree_inverse: np.ndarray
    transition_dipoles_ebohr: np.ndarray
    transition_charges_e: np.ndarray
    numerical_relative_tolerance: float = 1.0e-10
    construction: str = V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_CONSTRUCTION

    def __post_init__(self) -> None:
        atomic_number = _validate_atomic_number(self.atomic_number)
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("Atomic response symbol must be nonempty.")
        if not isinstance(self.basis, str) or not self.basis:
            raise ValueError("Atomic response basis must be nonempty.")
        if isinstance(self.spin_2s, bool) or int(self.spin_2s) != self.spin_2s:
            raise ValueError("Atomic response spin_2s must be an integer.")
        tolerance = _finite_positive(
            self.numerical_relative_tolerance,
            name="numerical_relative_tolerance",
        )
        coefficients = _immutable_float_array(
            self.mo_coefficients,
            name="Atomic MO coefficients",
        )
        if coefficients.ndim != 2 or coefficients.shape[0] == 0:
            raise ValueError("Atomic MO coefficients must be a nonempty matrix.")
        orbital_count = coefficients.shape[1]
        energies = _immutable_float_array(
            self.orbital_energies_hartree,
            name="Atomic orbital energies",
            shape=(orbital_count,),
        )
        occupations = _immutable_float_array(
            self.orbital_occupations,
            name="Atomic orbital occupations",
            shape=(orbital_count,),
        )
        if np.any(occupations < 0.0):
            raise ValueError("Atomic orbital occupations must be nonnegative.")
        if np.any(np.diff(energies) < -tolerance * _matrix_scale(energies)):
            raise ValueError("Atomic orbital energies must be in canonical order.")
        transition_count = np.asarray(self.transition_weights_hartree_inverse).size
        if transition_count == 0:
            raise ValueError("Atomic response must contain at least one transition.")
        lower = _integer_vector(
            self.transition_lower_indices,
            name="Atomic transition lower indices",
            length=transition_count,
        )
        upper = _integer_vector(
            self.transition_upper_indices,
            name="Atomic transition upper indices",
            length=transition_count,
        )
        if (
            np.any(lower < 0)
            or np.any(upper >= orbital_count)
            or np.any(lower >= upper)
        ):
            raise ValueError("Atomic transition indices must be ordered MO pairs.")
        response_weights = _immutable_float_array(
            self.transition_weights_hartree_inverse,
            name="Atomic transition response weights",
            shape=(transition_count,),
        )
        if np.any(response_weights <= 0.0):
            raise ValueError("Atomic transition response weights must be positive.")
        occupation_differences = occupations[lower] - occupations[upper]
        energy_gaps = energies[upper] - energies[lower]
        if np.any(occupation_differences <= 0.0) or np.any(energy_gaps <= 0.0):
            raise ValueError(
                "Atomic transition pairs must go from more occupied to less "
                "occupied canonical orbitals."
            )
        expected_weights = occupation_differences / (2.0 * energy_gaps)
        if not np.allclose(
            response_weights,
            expected_weights,
            rtol=tolerance,
            atol=tolerance * _matrix_scale(expected_weights),
        ):
            raise ValueError(
                "Atomic transition weights must equal the independent-particle "
                "occupation-over-gap response exactly."
            )
        dipoles = _immutable_float_array(
            self.transition_dipoles_ebohr,
            name="Atomic transition dipoles",
            shape=(transition_count, 3),
        )
        charges = _immutable_float_array(
            self.transition_charges_e,
            name="Atomic transition charges",
            shape=(transition_count,),
        )
        charge_tolerance = tolerance * max(1.0, float(np.linalg.norm(charges)))
        if float(np.max(np.abs(charges))) > charge_tolerance:
            raise ValueError("Atomic transition-density modes must be neutral.")
        atom_covariance = dipoles.T @ (response_weights[:, None] * dipoles)
        atom_covariance = 0.5 * (atom_covariance + atom_covariance.T)
        minimum = float(np.min(np.linalg.eigvalsh(atom_covariance)))
        if minimum <= tolerance * _matrix_scale(atom_covariance):
            raise ValueError(
                "Atomic independent-particle dipole covariance must be positive "
                "definite."
            )
        if self.construction != V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_CONSTRUCTION:
            raise ValueError("Unsupported atomic independent-particle construction.")
        object.__setattr__(self, "atomic_number", atomic_number)
        object.__setattr__(self, "spin_2s", int(self.spin_2s))
        object.__setattr__(self, "mo_coefficients", coefficients)
        object.__setattr__(self, "orbital_energies_hartree", energies)
        object.__setattr__(self, "orbital_occupations", occupations)
        object.__setattr__(self, "transition_lower_indices", lower)
        object.__setattr__(self, "transition_upper_indices", upper)
        object.__setattr__(
            self,
            "transition_weights_hartree_inverse",
            response_weights,
        )
        object.__setattr__(self, "transition_dipoles_ebohr", dipoles)
        object.__setattr__(self, "transition_charges_e", charges)
        object.__setattr__(self, "numerical_relative_tolerance", tolerance)

    @property
    def transition_count(self) -> int:
        """Return the number of neutral atomic transition-density modes."""

        return len(self.transition_weights_hartree_inverse)

    @property
    def atomic_polarizability_bohr3(self) -> np.ndarray:
        """Return the positive atomic independent-particle dipole covariance."""

        result = self.transition_dipoles_ebohr.T @ (
            self.transition_weights_hartree_inverse[:, None]
            * self.transition_dipoles_ebohr
        )
        result = 0.5 * (result + result.T)
        result.setflags(write=False)
        return result

    @property
    def atom_dipole_map_coefficient_to_ebohr(self) -> np.ndarray:
        """Map response amplitudes to the induced electronic dipole.

        ``transition_dipoles_ebohr`` is the first moment of an *electron
        number* density.  The electronic dipole is its negative.
        """

        result = -self.transition_dipoles_ebohr.T.copy()
        result.setflags(write=False)
        return result

    def transition_density_matrices(self) -> np.ndarray:
        """Return AO transition-density matrices for Coulomb source evaluation."""

        lower = self.mo_coefficients[:, self.transition_lower_indices].T
        upper = self.mo_coefficients[:, self.transition_upper_indices].T
        result = np.einsum("mi,mj->mij", lower, upper, optimize=True)
        result += np.einsum("mi,mj->mij", upper, lower, optimize=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0AtomicIndependentParticleBaseline:
    """Direct sum of atomic transition-density response bases at one geometry."""

    atomic_numbers: np.ndarray
    response_weights_hartree_inverse: np.ndarray
    transition_density_dipoles_ebohr: np.ndarray
    atom_dipole_map_coefficient_to_ebohr: np.ndarray
    transition_charges_e: np.ndarray
    coefficient_slices: tuple[slice, ...]

    def __post_init__(self) -> None:
        numbers = np.asarray(self.atomic_numbers)
        if numbers.ndim != 1 or numbers.size == 0:
            raise ValueError("Atomic numbers must be a nonempty one-dimensional array.")
        parsed_numbers = np.asarray(
            [_validate_atomic_number(number) for number in numbers], dtype=np.int64
        )
        parsed_numbers.setflags(write=False)
        response_weights = _immutable_float_array(
            self.response_weights_hartree_inverse,
            name="Baseline response weights",
        )
        if response_weights.ndim != 1 or response_weights.size == 0:
            raise ValueError("Baseline response weights must be nonempty.")
        if np.any(response_weights <= 0.0):
            raise ValueError("Baseline response weights must be positive.")
        coefficient_count = response_weights.size
        dipoles = _immutable_float_array(
            self.transition_density_dipoles_ebohr,
            name="Baseline transition-density dipoles",
            shape=(coefficient_count, 3),
        )
        atom_map = _immutable_float_array(
            self.atom_dipole_map_coefficient_to_ebohr,
            name="Baseline atom-dipole map",
            shape=(3 * len(parsed_numbers), coefficient_count),
        )
        charges = _immutable_float_array(
            self.transition_charges_e,
            name="Baseline transition charges",
            shape=(coefficient_count,),
        )
        if float(np.max(np.abs(charges))) > 1.0e-10:
            raise ValueError("Baseline response modes must be intrinsically neutral.")
        if len(self.coefficient_slices) != len(parsed_numbers):
            raise ValueError("Baseline response requires one coefficient slice per atom.")
        expected_start = 0
        for atom_index, coefficient_slice in enumerate(self.coefficient_slices):
            if (
                coefficient_slice.step not in (None, 1)
                or coefficient_slice.start != expected_start
                or coefficient_slice.stop is None
                or coefficient_slice.stop <= expected_start
            ):
                raise ValueError("Baseline coefficient slices must be contiguous.")
            other_rows = np.delete(atom_map, slice(3 * atom_index, 3 * atom_index + 3), 0)
            if np.any(np.abs(other_rows[:, coefficient_slice]) > 1.0e-12):
                raise ValueError("Atomic response blocks must remain direct sums.")
            expected_start = coefficient_slice.stop
        if expected_start != coefficient_count:
            raise ValueError("Baseline coefficient slices must cover every response mode.")
        atom_covariance = atom_map @ (response_weights[:, None] * atom_map.T)
        if float(np.min(np.linalg.eigvalsh(0.5 * (atom_covariance + atom_covariance.T)))) <= 0.0:
            raise ValueError("Baseline atom-dipole covariance must be positive definite.")
        object.__setattr__(self, "atomic_numbers", parsed_numbers)
        object.__setattr__(
            self,
            "response_weights_hartree_inverse",
            response_weights,
        )
        object.__setattr__(self, "transition_density_dipoles_ebohr", dipoles)
        object.__setattr__(self, "atom_dipole_map_coefficient_to_ebohr", atom_map)
        object.__setattr__(self, "transition_charges_e", charges)

    @property
    def coefficient_count(self) -> int:
        return len(self.response_weights_hartree_inverse)

    @property
    def baseline_response_covariance_coefficient_dual(self) -> np.ndarray:
        """Return the positive diagonal independent-particle covariance ``C0``."""

        result = np.diag(self.response_weights_hartree_inverse)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class Route2V0AtomicIndependentParticleResponseTable:
    """Hash-bound collection of independently generated atomic response assets."""

    responses_by_atomic_number: Mapping[int, Route2V0AtomicIndependentParticleResponse]
    table_sha256: str
    manifest_sha256: str
    construction: str = V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_CONSTRUCTION

    def __post_init__(self) -> None:
        if self.construction != V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_CONSTRUCTION:
            raise ValueError("Unsupported atomic independent-particle table.")
        if not isinstance(self.table_sha256, str) or len(self.table_sha256) != 64:
            raise ValueError("Atomic independent-particle table SHA256 is invalid.")
        if not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64:
            raise ValueError("Atomic independent-particle manifest SHA256 is invalid.")
        parsed: dict[int, Route2V0AtomicIndependentParticleResponse] = {}
        for number, response in self.responses_by_atomic_number.items():
            atomic_number = _validate_atomic_number(number)
            if not isinstance(response, Route2V0AtomicIndependentParticleResponse):
                raise TypeError("Atomic independent-particle table entries are invalid.")
            if atomic_number != response.atomic_number:
                raise ValueError("Atomic independent-particle table key disagrees with asset.")
            parsed[atomic_number] = response
        if not parsed:
            raise ValueError("Atomic independent-particle table must contain an asset.")
        object.__setattr__(
            self,
            "responses_by_atomic_number",
            MappingProxyType(dict(sorted(parsed.items()))),
        )

    @property
    def supported_atomic_numbers(self) -> tuple[int, ...]:
        return tuple(self.responses_by_atomic_number)

    def response_for_atomic_number(
        self,
        atomic_number: int,
    ) -> Route2V0AtomicIndependentParticleResponse:
        number = _validate_atomic_number(atomic_number)
        try:
            return self.responses_by_atomic_number[number]
        except KeyError as error:
            raise ValueError(
                f"Atomic independent-particle response has no asset for Z={number}."
            ) from error

    def assemble(
        self,
        atomic_numbers: np.ndarray,
    ) -> Route2V0AtomicIndependentParticleBaseline:
        """Return the exact direct sum of the requested atomic response blocks."""

        raw_numbers = np.asarray(atomic_numbers)
        if raw_numbers.ndim != 1 or raw_numbers.size == 0:
            raise ValueError("Atomic numbers must be a nonempty one-dimensional array.")
        numbers = [_validate_atomic_number(number) for number in raw_numbers]
        responses = [self.response_for_atomic_number(number) for number in numbers]
        total = sum(response.transition_count for response in responses)
        response_weights = np.empty(total, dtype=float)
        transition_dipoles = np.empty((total, 3), dtype=float)
        transition_charges = np.empty(total, dtype=float)
        atom_map = np.zeros((3 * len(responses), total), dtype=float)
        slices: list[slice] = []
        start = 0
        for atom_index, response in enumerate(responses):
            stop = start + response.transition_count
            coefficient_slice = slice(start, stop)
            slices.append(coefficient_slice)
            response_weights[coefficient_slice] = (
                response.transition_weights_hartree_inverse
            )
            transition_dipoles[coefficient_slice] = response.transition_dipoles_ebohr
            transition_charges[coefficient_slice] = response.transition_charges_e
            atom_map[3 * atom_index : 3 * atom_index + 3, coefficient_slice] = (
                response.atom_dipole_map_coefficient_to_ebohr
            )
            start = stop
        return Route2V0AtomicIndependentParticleBaseline(
            atomic_numbers=np.asarray(numbers, dtype=np.int64),
            response_weights_hartree_inverse=response_weights,
            transition_density_dipoles_ebohr=transition_dipoles,
            atom_dipole_map_coefficient_to_ebohr=atom_map,
            transition_charges_e=transition_charges,
            coefficient_slices=tuple(slices),
        )


def build_route2_v0_atomic_independent_particle_response(
    *,
    atomic_number: int,
    symbol: str,
    basis: str,
    spin_2s: int,
    mo_coefficients: np.ndarray,
    orbital_energies_hartree: np.ndarray,
    orbital_occupations: np.ndarray,
    overlap_matrix: np.ndarray,
    position_integrals_ebohr: np.ndarray,
    numerical_relative_tolerance: float = 1.0e-10,
) -> Route2V0AtomicIndependentParticleResponse:
    """Build an atomic IP response from canonical orbitals without fitting.

    ``overlap_matrix`` and ``position_integrals_ebohr`` are the exact AO
    integrals of the frozen atomic calculation.  The routine sorts canonical
    orbitals by energy, retains every positive occupation-over-gap transition,
    and rejects a nonorthonormal or nonspherical numerical source instead of
    altering any response weight.
    """

    tolerance = _finite_positive(
        numerical_relative_tolerance,
        name="numerical_relative_tolerance",
    )
    coefficients = _immutable_float_array(
        mo_coefficients,
        name="Atomic MO coefficients",
    )
    if coefficients.ndim != 2 or coefficients.shape[0] == 0:
        raise ValueError("Atomic MO coefficients must be a nonempty matrix.")
    orbital_count = coefficients.shape[1]
    energies = _immutable_float_array(
        orbital_energies_hartree,
        name="Atomic orbital energies",
        shape=(orbital_count,),
    )
    occupations = _immutable_float_array(
        orbital_occupations,
        name="Atomic orbital occupations",
        shape=(orbital_count,),
    )
    overlap = _immutable_float_array(
        overlap_matrix,
        name="Atomic AO overlap matrix",
        shape=(coefficients.shape[0], coefficients.shape[0]),
    )
    position_integrals = _immutable_float_array(
        position_integrals_ebohr,
        name="Atomic AO position integrals",
        shape=(3, coefficients.shape[0], coefficients.shape[0]),
    )
    orthonormality_error = float(
        np.linalg.norm(coefficients.T @ overlap @ coefficients - np.eye(orbital_count), ord=2)
    )
    if orthonormality_error > tolerance * _matrix_scale(overlap):
        raise ValueError("Atomic canonical orbitals are not overlap-orthonormal.")
    order = np.argsort(energies, kind="stable")
    coefficients = coefficients[:, order]
    energies = energies[order]
    occupations = occupations[order]
    lower_indices: list[int] = []
    upper_indices: list[int] = []
    response_weights: list[float] = []
    transition_dipoles: list[np.ndarray] = []
    transition_charges: list[float] = []
    energy_scale = _matrix_scale(energies)
    for lower in range(orbital_count):
        for upper in range(lower + 1, orbital_count):
            occupation_difference = occupations[lower] - occupations[upper]
            energy_gap = energies[upper] - energies[lower]
            if occupation_difference <= tolerance or energy_gap <= tolerance * energy_scale:
                continue
            transition_density = (
                np.outer(coefficients[:, lower], coefficients[:, upper])
                + np.outer(coefficients[:, upper], coefficients[:, lower])
            )
            lower_indices.append(lower)
            upper_indices.append(upper)
            response_weights.append(occupation_difference / (2.0 * energy_gap))
            transition_dipoles.append(
                np.einsum(
                    "ij,kij->k",
                    transition_density,
                    position_integrals,
                    optimize=True,
                )
            )
            transition_charges.append(
                float(np.einsum("ij,ij", transition_density, overlap, optimize=True))
            )
    if not response_weights:
        raise RuntimeError("Atomic independent-particle response has no transitions.")
    response = Route2V0AtomicIndependentParticleResponse(
        atomic_number=atomic_number,
        symbol=symbol,
        basis=basis,
        spin_2s=spin_2s,
        mo_coefficients=coefficients,
        orbital_energies_hartree=energies,
        orbital_occupations=occupations,
        transition_lower_indices=np.asarray(lower_indices, dtype=np.int64),
        transition_upper_indices=np.asarray(upper_indices, dtype=np.int64),
        transition_weights_hartree_inverse=np.asarray(response_weights, dtype=float),
        transition_dipoles_ebohr=np.asarray(transition_dipoles, dtype=float),
        transition_charges_e=np.asarray(transition_charges, dtype=float),
        numerical_relative_tolerance=tolerance,
    )
    covariance = response.atomic_polarizability_bohr3
    isotropy_error = float(
        np.linalg.norm(covariance - np.trace(covariance) * np.eye(3) / 3.0, ord=2)
    )
    if isotropy_error > tolerance * _matrix_scale(covariance):
        raise ValueError(
            "Spherical atomic response does not have an isotropic dipole covariance."
        )
    return response


def load_route2_v0_atomic_independent_particle_response_table(
    *,
    table_path: Path,
    manifest_path: Path,
) -> Route2V0AtomicIndependentParticleResponseTable:
    """Load and hash-verify a frozen atomic independent-particle table."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read atomic independent-particle manifest: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Atomic independent-particle manifest must be an object.")
    if manifest.get("artifact") != V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_ARTIFACT:
        raise ValueError("Atomic independent-particle manifest artifact is invalid.")
    if manifest.get("status") != "pass":
        raise ValueError("Atomic independent-particle manifest is not admitted.")
    table_record = manifest.get("table")
    if not isinstance(table_record, dict) or table_record.get("sha256") != _sha256(table_path):
        raise ValueError("Atomic independent-particle table hash disagrees with manifest.")
    raw_results = manifest.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("Atomic independent-particle manifest has no element results.")
    results_by_number: dict[int, dict[str, object]] = {}
    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError("Atomic independent-particle manifest result is invalid.")
        number = _validate_atomic_number(result.get("atomic_number"))
        if number in results_by_number:
            raise ValueError("Atomic independent-particle manifest repeats an element.")
        results_by_number[number] = result
    try:
        archive = np.load(table_path, allow_pickle=False)
    except OSError as error:
        raise ValueError(f"Cannot read atomic independent-particle table: {error}") from error
    with archive:
        try:
            atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
        except KeyError as error:
            raise ValueError("Atomic independent-particle table omits atomic_numbers.") from error
        if sorted(atomic_numbers.tolist()) != sorted(results_by_number):
            raise ValueError("Atomic independent-particle table and manifest disagree.")
        responses: dict[int, Route2V0AtomicIndependentParticleResponse] = {}
        for number in atomic_numbers.tolist():
            result = results_by_number[number]
            prefix = f"Z{number}"
            try:
                response = Route2V0AtomicIndependentParticleResponse(
                    atomic_number=number,
                    symbol=str(result["symbol"]),
                    basis=str(manifest["generation_contract"]["basis"]),
                    spin_2s=int(result["spin_2s"]),
                    mo_coefficients=archive[f"mo_coefficients_{prefix}"],
                    orbital_energies_hartree=archive[f"orbital_energies_hartree_{prefix}"],
                    orbital_occupations=archive[f"orbital_occupations_{prefix}"],
                    transition_lower_indices=archive[f"transition_lower_indices_{prefix}"],
                    transition_upper_indices=archive[f"transition_upper_indices_{prefix}"],
                    transition_weights_hartree_inverse=archive[
                        f"transition_weights_hartree_inverse_{prefix}"
                    ],
                    transition_dipoles_ebohr=archive[
                        f"transition_dipoles_ebohr_{prefix}"
                    ],
                    transition_charges_e=archive[f"transition_charges_e_{prefix}"],
                    numerical_relative_tolerance=float(
                        manifest["generation_contract"]["numerical_relative_tolerance"]
                    ),
                )
            except KeyError as error:
                raise ValueError(
                    "Atomic independent-particle table omits a required element array."
                ) from error
            expected_hashes = result.get("array_sha256")
            if not isinstance(expected_hashes, dict):
                raise ValueError("Atomic independent-particle array hashes are missing.")
            actual_hashes = {
                "mo_coefficients": _sha256_array(response.mo_coefficients),
                "orbital_energies_hartree": _sha256_array(
                    response.orbital_energies_hartree
                ),
                "orbital_occupations": _sha256_array(response.orbital_occupations),
                "transition_lower_indices": _sha256_array(
                    response.transition_lower_indices
                ),
                "transition_upper_indices": _sha256_array(
                    response.transition_upper_indices
                ),
                "transition_weights_hartree_inverse": _sha256_array(
                    response.transition_weights_hartree_inverse
                ),
                "transition_dipoles_ebohr": _sha256_array(
                    response.transition_dipoles_ebohr
                ),
                "transition_charges_e": _sha256_array(response.transition_charges_e),
            }
            if actual_hashes != expected_hashes:
                raise ValueError(
                    "Atomic independent-particle element arrays disagree with manifest."
                )
            responses[number] = response
    return Route2V0AtomicIndependentParticleResponseTable(
        responses_by_atomic_number=responses,
        table_sha256=_sha256(table_path),
        manifest_sha256=_sha256(manifest_path),
    )


__all__ = [
    "Route2V0AtomicIndependentParticleBaseline",
    "Route2V0AtomicIndependentParticleResponse",
    "Route2V0AtomicIndependentParticleResponseTable",
    "V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_ARTIFACT",
    "V0_ATOMIC_INDEPENDENT_PARTICLE_RESPONSE_CONSTRUCTION",
    "build_route2_v0_atomic_independent_particle_response",
    "load_route2_v0_atomic_independent_particle_response_table",
]
