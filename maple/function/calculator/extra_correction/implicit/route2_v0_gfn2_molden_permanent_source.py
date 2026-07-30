"""Strict zero-field GFN2-xTB MOLDEN permanent-source adapter for Route-2 V0.

GFN2-xTB is retained here *only* as an independently stationary, zero-field
permanent electrostatic reference.  The old external-point-charge response
screen remains rejected and this module exposes no xTB field response,
implicit-solvent calculation, fitted charge, radius, or solvation correction.

The adapter reconstructs the valence AO density matrix printed by a closed-shell
GFN2-xTB ``--molden --json`` single point.  It verifies the printed MO metric,
valence electron count, effective-core total charge, and the xTB-reported
molecular dipole before the result may serve as the affine origin ``c0`` of a
separately specified Route-2 V0 quadratic response functional.

The source is deliberately narrow: only all-``s``/``p`` contracted Cartesian
Gaussian MOLDEN exports are admitted, which is the GFN2-xTB basis domain used
by the initial H/C/N/O/S/Cl V0 chemistry.  Other shells and open shells fail
closed rather than being silently reinterpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .route2_v0_auxiliary_ao_grid import Route2V0AuxiliaryAODensityGridProjection
from .route2_v0_structured_solvent import RegularCartesianGrid

V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION = (
    "route2-v0-gfn2-molden-permanent-source-v1"
)
V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE = (
    "zero-field-permanent-source-only-not-xtb-response-v1"
)

_MOLDEN_METRIC_RELATIVE_TOLERANCE = 5.0e-8
_MOLDEN_ELECTRON_COUNT_RELATIVE_TOLERANCE = 5.0e-8
_XTB_DIPOLE_ROUND_TRIP_TOLERANCE_E_BOHR = 1.0e-6
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SECTION = re.compile(r"^\s*\[([^]]+)\]\s*(.*)$", re.IGNORECASE)
_PARAMETER_ELEMENT = re.compile(r"^\s*\$Z\s*=\s*(\d+)\b", re.IGNORECASE)
_AO_DECLARATION = re.compile(r"^\s*ao\s*=\s*([^\s]+)", re.IGNORECASE)
_ORBITAL_PRINCIPAL = re.compile(r"(\d+)[spdfghi]", re.IGNORECASE)

# Closed-shell electron counts below the first explicitly modelled principal
# shell.  GFN2's ``ao=`` record declares that first shell; the table is a
# periodic-shell identity, not a fitted source parameter.
_CLOSED_SHELL_CORE_ELECTRONS = {
    1: 0,
    2: 2,
    3: 10,
    4: 18,
    5: 36,
    6: 54,
    7: 86,
}


@dataclass(frozen=True)
class _CartesianGaussianAO:
    """One unnormalised contracted Cartesian Gaussian from a MOLDEN export."""

    center_bohr: np.ndarray
    powers: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _immutable_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array.") from error
    if (shape is not None and array.shape != shape) or not np.all(np.isfinite(array)):
        expected = "a finite array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_integer_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if (shape is not None and raw.shape != shape) or not np.all(np.isfinite(raw)):
        expected = "a finite integer array" if shape is None else f"shape {shape}"
        raise ValueError(f"{name} must be finite with {expected}.")
    numeric = np.asarray(raw, dtype=float)
    rounded = np.rint(numeric)
    if np.any(rounded != numeric):
        raise ValueError(f"{name} must contain integers.")
    result = rounded.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite real number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _matrix_scale(values: np.ndarray) -> float:
    return max(1.0, float(np.linalg.norm(values, ord=2)))


def _immutable_mapping(
    values: Mapping[int, float], *, name: str
) -> Mapping[int, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a nonempty mapping.")
    frozen: dict[int, float] = {}
    for raw_number, raw_charge in values.items():
        if isinstance(raw_number, bool):
            raise ValueError(f"{name} atomic numbers must be positive integers.")
        number = int(raw_number)
        if number < 1:
            raise ValueError(f"{name} atomic numbers must be positive integers.")
        charge = _finite_scalar(raw_charge, name=f"{name} charge for Z={number}")
        if charge <= 0.0:
            raise ValueError(f"{name} effective core charges must be positive.")
        frozen[number] = charge
    return MappingProxyType(frozen)


def _section_indices(lines: list[str]) -> dict[str, int]:
    sections: dict[str, int] = {}
    for index, line in enumerate(lines):
        matched = _SECTION.match(line)
        if matched is None:
            continue
        name = matched.group(1).strip().lower()
        if name in sections:
            raise ValueError(f"MOLDEN export repeats [{name}] section.")
        sections[name] = index
    required = {"atoms", "gto", "mo"}
    if not required.issubset(sections):
        missing = sorted(required - set(sections))
        raise ValueError(f"MOLDEN export omits required sections {missing}.")
    if not (sections["atoms"] < sections["gto"] < sections["mo"]):
        raise ValueError("MOLDEN sections must occur in [Atoms], [GTO], [MO] order.")
    return sections


def _parse_atoms(
    lines: list[str], *, start: int, stop: int
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    header = _SECTION.match(lines[start])
    if header is None or header.group(1).strip().lower() != "atoms":
        raise ValueError("Cannot parse MOLDEN [Atoms] header.")
    unit = header.group(2).strip().upper()
    if unit != "AU":
        raise ValueError("Route-2 V0 accepts only MOLDEN [Atoms] AU coordinates.")
    symbols: list[str] = []
    numbers: list[int] = []
    positions: list[list[float]] = []
    expected_index = 1
    for line in lines[start + 1 : stop]:
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 6:
            raise ValueError("MOLDEN atom records must have six fields.")
        symbol = fields[0]
        try:
            index = int(fields[1])
            atomic_number = int(fields[2])
            position = [float(value) for value in fields[3:6]]
        except ValueError as error:
            raise ValueError("MOLDEN atom record is invalid.") from error
        if (
            index != expected_index
            or atomic_number < 1
            or not np.all(np.isfinite(position))
        ):
            raise ValueError(
                "MOLDEN atom indices, atomic numbers, or positions are invalid."
            )
        symbols.append(symbol)
        numbers.append(atomic_number)
        positions.append(position)
        expected_index += 1
    if not symbols:
        raise ValueError("MOLDEN [Atoms] section is empty.")
    return (
        tuple(symbols),
        _immutable_integer_array(numbers, name="MOLDEN atomic numbers"),
        _immutable_array(
            positions, name="MOLDEN atom positions", shape=(len(symbols), 3)
        ),
    )


def _parse_gto(
    lines: list[str],
    *,
    start: int,
    stop: int,
    atom_positions_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    atom_count = atom_positions_bohr.shape[0]
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    index = start + 1
    seen_atoms: set[int] = set()
    current_atom: int | None = None
    while index < stop:
        stripped = lines[index].strip()
        index += 1
        if not stripped:
            current_atom = None
            continue
        fields = stripped.split()
        if current_atom is None:
            if len(fields) != 2:
                raise ValueError("MOLDEN [GTO] atom header is invalid.")
            try:
                atom_index = int(fields[0])
                shell_index = int(fields[1])
            except ValueError as error:
                raise ValueError("MOLDEN [GTO] atom header is invalid.") from error
            if (
                shell_index != 0
                or not 1 <= atom_index <= atom_count
                or atom_index in seen_atoms
            ):
                raise ValueError("MOLDEN [GTO] atom blocks are invalid or repeated.")
            current_atom = atom_index - 1
            seen_atoms.add(current_atom)
            continue
        if len(fields) != 3:
            raise ValueError("MOLDEN shell header is invalid.")
        shell = fields[0].lower()
        if shell not in {"s", "p"}:
            raise ValueError(
                "Route-2 V0 GFN2 MOLDEN adapter accepts only explicit s/p shells."
            )
        try:
            primitive_count = int(fields[1])
            scale = float(fields[2])
        except ValueError as error:
            raise ValueError("MOLDEN shell header is invalid.") from error
        if primitive_count < 1 or not math.isfinite(scale) or scale != 1.0:
            raise ValueError(
                "MOLDEN shell primitive count must be positive and scale must equal one."
            )
        exponents: list[float] = []
        coefficients: list[float] = []
        for _ in range(primitive_count):
            if index >= stop:
                raise ValueError("MOLDEN shell is truncated.")
            primitive = lines[index].split()
            index += 1
            if len(primitive) < 2:
                raise ValueError("MOLDEN primitive record is invalid.")
            try:
                exponent = float(primitive[0])
                coefficient = float(primitive[1])
            except ValueError as error:
                raise ValueError("MOLDEN primitive record is invalid.") from error
            if (
                not math.isfinite(exponent)
                or not math.isfinite(coefficient)
                or exponent <= 0.0
            ):
                raise ValueError(
                    "MOLDEN primitive exponents and coefficients are invalid."
                )
            exponents.append(exponent)
            coefficients.append(coefficient)
        powers = ((0, 0, 0),) if shell == "s" else ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for power in powers:
            rows.append(
                (
                    atom_positions_bohr[current_atom],
                    np.asarray(power, dtype=np.int64),
                    np.asarray(exponents, dtype=float),
                    np.asarray(coefficients, dtype=float),
                )
            )
    if seen_atoms != set(range(atom_count)):
        raise ValueError("MOLDEN [GTO] does not declare every atom exactly once.")
    if not rows:
        raise ValueError("MOLDEN [GTO] contains no supported AO functions.")
    return (
        _immutable_array(
            [row[0] for row in rows],
            name="MOLDEN AO centres",
            shape=(len(rows), 3),
        ),
        _immutable_integer_array(
            [row[1] for row in rows],
            name="MOLDEN AO powers",
            shape=(len(rows), 3),
        ),
        tuple(_immutable_array(row[2], name="MOLDEN AO exponents") for row in rows),
        tuple(_immutable_array(row[3], name="MOLDEN AO coefficients") for row in rows),
    )


def _parse_mos(
    lines: list[str],
    *,
    start: int,
    ao_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = start + 1
    orbitals: list[np.ndarray] = []
    occupations: list[float] = []
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if not lines[index].strip().lower().startswith("sym="):
            raise ValueError("MOLDEN [MO] orbital must start with Sym=.")
        index += 1
        metadata: dict[str, str] = {}
        while index < len(lines):
            stripped = lines[index].strip()
            lowered = stripped.lower()
            if lowered.startswith("sym="):
                break
            if "=" not in stripped:
                break
            key, value = stripped.split("=", 1)
            key = key.strip().lower()
            if key not in {"ene", "spin", "occup"}:
                break
            if key in metadata:
                raise ValueError("MOLDEN [MO] repeats orbital metadata.")
            metadata[key] = value.strip()
            index += 1
        if set(metadata) != {"ene", "spin", "occup"}:
            raise ValueError("MOLDEN [MO] must declare Ene, Spin, and Occup.")
        try:
            energy = float(metadata["ene"])
            occupation = float(metadata["occup"])
        except ValueError as error:
            raise ValueError("MOLDEN [MO] energy or occupation is invalid.") from error
        if not math.isfinite(energy) or not math.isfinite(occupation):
            raise ValueError("MOLDEN [MO] energy or occupation is non-finite.")
        if metadata["spin"].strip().lower() != "alpha":
            raise ValueError(
                "Route-2 V0 GFN2 MOLDEN adapter requires closed-shell Alpha MOs."
            )
        if (
            occupation < 0.0
            or occupation > 2.0
            or min(abs(occupation), abs(occupation - 2.0)) > 1.0e-12
        ):
            raise ValueError(
                "Route-2 V0 GFN2 MOLDEN adapter requires integer 0/2 occupations."
            )
        coefficient = np.zeros(ao_count, dtype=float)
        seen_indices: set[int] = set()
        while index < len(lines) and not lines[index].strip().lower().startswith(
            "sym="
        ):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError("MOLDEN [MO] coefficient record is invalid.")
            try:
                ao_index = int(fields[0])
                value = float(fields[1])
            except ValueError as error:
                raise ValueError(
                    "MOLDEN [MO] coefficient record is invalid."
                ) from error
            if (
                not 1 <= ao_index <= ao_count
                or ao_index in seen_indices
                or not math.isfinite(value)
            ):
                raise ValueError("MOLDEN [MO] coefficient index or value is invalid.")
            coefficient[ao_index - 1] = value
            seen_indices.add(ao_index)
            index += 1
        orbitals.append(coefficient)
        occupations.append(occupation)
    if len(orbitals) != ao_count:
        raise ValueError(
            "Route-2 V0 requires a complete square GFN2 MOLDEN MO coefficient matrix."
        )
    return (
        _immutable_array(
            np.column_stack(orbitals),
            name="MOLDEN MO coefficients",
            shape=(ao_count, ao_count),
        ),
        _immutable_array(occupations, name="MOLDEN MO occupations", shape=(ao_count,)),
    )


def _one_dimension_gaussian_moment(
    *,
    left_power: int,
    right_power: int,
    coordinate_power: int,
    left_exponent: float,
    right_exponent: float,
    left_center: float,
    right_center: float,
) -> float:
    """Return one exact Cartesian Gaussian moment for angular powers <= one."""

    exponent_sum = left_exponent + right_exponent
    product_center = (
        left_exponent * left_center + right_exponent * right_center
    ) / exponent_sum
    result = 0.0
    for left_y_power in range(left_power + 1):
        left_coefficient = math.comb(left_power, left_y_power) * (
            product_center - left_center
        ) ** (left_power - left_y_power)
        for right_y_power in range(right_power + 1):
            right_coefficient = math.comb(right_power, right_y_power) * (
                product_center - right_center
            ) ** (right_power - right_y_power)
            for coordinate_y_power in range(coordinate_power + 1):
                coordinate_coefficient = math.comb(
                    coordinate_power, coordinate_y_power
                ) * product_center ** (coordinate_power - coordinate_y_power)
                total_power = left_y_power + right_y_power + coordinate_y_power
                if total_power % 2:
                    continue
                half_power = total_power // 2
                double_factorial = 1
                for value in range(1, 2 * half_power, 2):
                    double_factorial *= value
                gaussian_moment = (
                    double_factorial
                    * math.sqrt(math.pi)
                    / (2**half_power * exponent_sum ** (half_power + 0.5))
                )
                result += (
                    left_coefficient
                    * right_coefficient
                    * coordinate_coefficient
                    * gaussian_moment
                )
    return result


def _primitive_cartesian_moment(
    *,
    left_center: np.ndarray,
    left_powers: np.ndarray,
    left_exponent: float,
    right_center: np.ndarray,
    right_powers: np.ndarray,
    right_exponent: float,
    coordinate_powers: np.ndarray,
) -> float:
    exponent_sum = left_exponent + right_exponent
    reduced_exponent = left_exponent * right_exponent / exponent_sum
    prefactor = math.exp(
        -reduced_exponent
        * float(np.dot(left_center - right_center, left_center - right_center))
    )
    result = prefactor
    for axis in range(3):
        result *= _one_dimension_gaussian_moment(
            left_power=int(left_powers[axis]),
            right_power=int(right_powers[axis]),
            coordinate_power=int(coordinate_powers[axis]),
            left_exponent=left_exponent,
            right_exponent=right_exponent,
            left_center=float(left_center[axis]),
            right_center=float(right_center[axis]),
        )
    return result


def _contracted_cartesian_moment(
    left: _CartesianGaussianAO,
    right: _CartesianGaussianAO,
    *,
    coordinate_powers: np.ndarray,
) -> float:
    result = 0.0
    for left_exponent, left_coefficient in zip(
        left.exponents, left.coefficients, strict=True
    ):
        for right_exponent, right_coefficient in zip(
            right.exponents, right.coefficients, strict=True
        ):
            result += float(
                left_coefficient * right_coefficient
            ) * _primitive_cartesian_moment(
                left_center=left.center_bohr,
                left_powers=left.powers,
                left_exponent=float(left_exponent),
                right_center=right.center_bohr,
                right_powers=right.powers,
                right_exponent=float(right_exponent),
                coordinate_powers=coordinate_powers,
            )
    return result


@dataclass(frozen=True)
class Route2V0GFN2MoldenAODensity:
    """Closed-shell GFN2 MOLDEN valence density with exact AO algebra checks."""

    atom_symbols: tuple[str, ...]
    atomic_numbers: np.ndarray
    atom_positions_bohr: np.ndarray
    ao_centers_bohr: np.ndarray
    ao_powers: np.ndarray
    ao_exponents: tuple[np.ndarray, ...]
    ao_coefficients: tuple[np.ndarray, ...]
    molecular_orbital_coefficients: np.ndarray
    molecular_orbital_occupations: np.ndarray
    molden_sha256: str
    construction: str = V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION
    scope: str = V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE
    overlap_matrix: np.ndarray = field(init=False, repr=False)
    density_matrix: np.ndarray = field(init=False, repr=False)
    mo_metric_error: float = field(init=False)
    valence_electron_count_e: float = field(init=False)
    valence_electron_count_error_e: float = field(init=False)

    def __post_init__(self) -> None:
        atom_count = len(self.atom_symbols)
        if atom_count == 0 or any(not symbol.strip() for symbol in self.atom_symbols):
            raise ValueError("MOLDEN atom symbols must be nonempty.")
        atomic_numbers = _immutable_integer_array(
            self.atomic_numbers,
            name="MOLDEN atomic numbers",
            shape=(atom_count,),
        )
        if np.any(atomic_numbers < 1):
            raise ValueError("MOLDEN atomic numbers must be positive.")
        atom_positions = _immutable_array(
            self.atom_positions_bohr,
            name="MOLDEN atom positions",
            shape=(atom_count, 3),
        )
        centers = _immutable_array(
            self.ao_centers_bohr,
            name="MOLDEN AO centres",
        )
        if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] != 3:
            raise ValueError("MOLDEN AO centres must have shape (n_ao, 3).")
        ao_count = centers.shape[0]
        powers = _immutable_integer_array(
            self.ao_powers,
            name="MOLDEN AO powers",
            shape=(ao_count, 3),
        )
        if (
            np.any(powers < 0)
            or np.any(powers > 1)
            or np.any(np.sum(powers, axis=1) > 1)
        ):
            raise ValueError(
                "Route-2 V0 MOLDEN AO powers must be Cartesian s/p powers."
            )
        if len(self.ao_exponents) != ao_count or len(self.ao_coefficients) != ao_count:
            raise ValueError("MOLDEN AO primitive arrays do not match the AO count.")
        exponents: list[np.ndarray] = []
        coefficients: list[np.ndarray] = []
        for index, (raw_exponents, raw_coefficients) in enumerate(
            zip(self.ao_exponents, self.ao_coefficients, strict=True)
        ):
            exponent = _immutable_array(
                raw_exponents, name=f"MOLDEN AO {index} exponents"
            )
            coefficient = _immutable_array(
                raw_coefficients,
                name=f"MOLDEN AO {index} coefficients",
                shape=exponent.shape,
            )
            if exponent.ndim != 1 or exponent.size == 0 or np.any(exponent <= 0.0):
                raise ValueError(
                    "MOLDEN AO primitive exponents must be positive vectors."
                )
            exponents.append(exponent)
            coefficients.append(coefficient)
        coefficients_mo = _immutable_array(
            self.molecular_orbital_coefficients,
            name="MOLDEN MO coefficients",
            shape=(ao_count, ao_count),
        )
        occupations = _immutable_array(
            self.molecular_orbital_occupations,
            name="MOLDEN MO occupations",
            shape=(ao_count,),
        )
        if (
            np.any(occupations < 0.0)
            or np.any(occupations > 2.0)
            or np.any(
                np.minimum(np.abs(occupations), np.abs(occupations - 2.0)) > 1.0e-12
            )
        ):
            raise ValueError("Route-2 V0 requires closed-shell 0/2 MO occupations.")
        if (
            not isinstance(self.molden_sha256, str)
            or _DIGEST.fullmatch(self.molden_sha256) is None
        ):
            raise ValueError("MOLDEN source SHA256 is invalid.")
        if self.construction != V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported GFN2 MOLDEN permanent-source construction.")
        if self.scope != V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE:
            raise ValueError("Unsupported GFN2 MOLDEN permanent-source scope.")
        aos = tuple(
            _CartesianGaussianAO(
                center_bohr=centers[index],
                powers=powers[index],
                exponents=exponents[index],
                coefficients=coefficients[index],
            )
            for index in range(ao_count)
        )
        overlap = np.empty((ao_count, ao_count), dtype=float)
        coordinate_powers = np.zeros(3, dtype=np.int64)
        for left_index, left in enumerate(aos):
            for right_index in range(left_index + 1):
                value = _contracted_cartesian_moment(
                    left,
                    aos[right_index],
                    coordinate_powers=coordinate_powers,
                )
                overlap[left_index, right_index] = value
                overlap[right_index, left_index] = value
        overlap = _immutable_array(
            overlap, name="MOLDEN AO overlap", shape=(ao_count, ao_count)
        )
        eigenvalues = np.linalg.eigvalsh(overlap)
        if float(np.min(eigenvalues)) <= 1.0e-12 * _matrix_scale(overlap):
            raise ValueError("MOLDEN AO overlap must be positive definite.")
        mo_metric = coefficients_mo.T @ overlap @ coefficients_mo
        metric_error = float(np.linalg.norm(mo_metric - np.eye(ao_count), ord=2))
        if metric_error > _MOLDEN_METRIC_RELATIVE_TOLERANCE:
            raise ValueError(
                "MOLDEN MO coefficients do not satisfy the declared AO metric."
            )
        density = _immutable_array(
            (coefficients_mo * occupations) @ coefficients_mo.T,
            name="MOLDEN valence density matrix",
            shape=(ao_count, ao_count),
        )
        valence_count = float(np.einsum("ij,ji->", density, overlap, optimize=True))
        occupation_count = float(np.sum(occupations))
        count_error = abs(valence_count - occupation_count)
        if count_error > _MOLDEN_ELECTRON_COUNT_RELATIVE_TOLERANCE * max(
            1.0, abs(occupation_count)
        ):
            raise ValueError(
                "MOLDEN valence density does not reproduce its occupations."
            )
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "atom_positions_bohr", atom_positions)
        object.__setattr__(self, "ao_centers_bohr", centers)
        object.__setattr__(self, "ao_powers", powers)
        object.__setattr__(self, "ao_exponents", tuple(exponents))
        object.__setattr__(self, "ao_coefficients", tuple(coefficients))
        object.__setattr__(self, "molecular_orbital_coefficients", coefficients_mo)
        object.__setattr__(self, "molecular_orbital_occupations", occupations)
        object.__setattr__(self, "overlap_matrix", overlap)
        object.__setattr__(self, "density_matrix", density)
        object.__setattr__(self, "mo_metric_error", metric_error)
        object.__setattr__(self, "valence_electron_count_e", valence_count)
        object.__setattr__(self, "valence_electron_count_error_e", count_error)

    @property
    def ao_count(self) -> int:
        """Return the number of declared valence AO functions."""

        return int(self.ao_centers_bohr.shape[0])

    @property
    def atom_count(self) -> int:
        """Return the number of nuclei in the permanent reference."""

        return int(self.atomic_numbers.size)

    def ao_values(self, points_bohr: np.ndarray) -> np.ndarray:
        """Evaluate the exact printed contracted Cartesian AO functions."""

        points = _immutable_array(points_bohr, name="MOLDEN AO evaluation points")
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise ValueError(
                "MOLDEN AO evaluation points must have shape (n_points, 3)."
            )
        values = np.empty((points.shape[0], self.ao_count), dtype=float)
        for index, (center, powers, exponents, coefficients) in enumerate(
            zip(
                self.ao_centers_bohr,
                self.ao_powers,
                self.ao_exponents,
                self.ao_coefficients,
                strict=True,
            )
        ):
            displacement = points - center
            radial_squared = np.einsum("pi,pi->p", displacement, displacement)
            radial = np.zeros(points.shape[0], dtype=float)
            for exponent, coefficient in zip(exponents, coefficients, strict=True):
                radial += coefficient * np.exp(-exponent * radial_squared)
            values[:, index] = (
                np.prod(displacement ** powers[np.newaxis, :], axis=1) * radial
            )
        values.setflags(write=False)
        return values

    def coordinate_integrals_bohr(self) -> np.ndarray:
        """Return exact AO coordinate matrices for the printed s/p basis."""

        matrices = np.empty((3, self.ao_count, self.ao_count), dtype=float)
        aos = tuple(
            _CartesianGaussianAO(
                center_bohr=self.ao_centers_bohr[index],
                powers=self.ao_powers[index],
                exponents=self.ao_exponents[index],
                coefficients=self.ao_coefficients[index],
            )
            for index in range(self.ao_count)
        )
        for axis in range(3):
            coordinate_powers = np.zeros(3, dtype=np.int64)
            coordinate_powers[axis] = 1
            for left_index, left in enumerate(aos):
                for right_index in range(left_index + 1):
                    value = _contracted_cartesian_moment(
                        left,
                        aos[right_index],
                        coordinate_powers=coordinate_powers,
                    )
                    matrices[axis, left_index, right_index] = value
                    matrices[axis, right_index, left_index] = value
        matrices.setflags(write=False)
        return matrices

    def valence_electronic_dipole_e_bohr(self) -> np.ndarray:
        """Return ``Tr[D r]``; total dipole subtracts this electron moment."""

        value = np.einsum(
            "xij,ji->x",
            self.coordinate_integrals_bohr(),
            self.density_matrix,
            optimize=True,
        )
        result = _immutable_array(
            value,
            name="MOLDEN valence electronic dipole",
            shape=(3,),
        )
        return result

    def grid_projection(
        self,
        grid: RegularCartesianGrid,
        *,
        maximum_points_per_block: int = 100_000,
    ) -> Route2V0AuxiliaryAODensityGridProjection:
        """Return one exact AO/grid dual projection without rescaling density."""

        if not isinstance(grid, RegularCartesianGrid):
            raise TypeError(
                "GFN2 MOLDEN grid projection requires a regular Cartesian grid."
            )
        if isinstance(maximum_points_per_block, bool) or not isinstance(
            maximum_points_per_block, (int, np.integer)
        ):
            raise TypeError("MOLDEN grid block size must be an integer.")
        block_size = int(maximum_points_per_block)
        if block_size < 1:
            raise ValueError("MOLDEN grid block size must be positive.")
        points = grid.points_bohr()
        values = np.empty((grid.point_count, self.ao_count), dtype=float)
        for start in range(0, grid.point_count, block_size):
            stop = min(grid.point_count, start + block_size)
            values[start:stop] = self.ao_values(points[start:stop])
        return Route2V0AuxiliaryAODensityGridProjection(
            grid=grid,
            ao_values=values,
            overlap_matrix=self.overlap_matrix,
        )


def load_route2_v0_gfn2_molden_ao_density(path: Path) -> Route2V0GFN2MoldenAODensity:
    """Parse one strict closed-shell GFN2-xTB MOLDEN permanent-source export."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read GFN2 MOLDEN export: {error}") from error
    if not lines or lines[0].strip().lower() != "[molden format]":
        raise ValueError("GFN2 MOLDEN export must start with [Molden Format].")
    sections = _section_indices(lines)
    symbols, numbers, positions = _parse_atoms(
        lines,
        start=sections["atoms"],
        stop=sections["gto"],
    )
    centers, powers, exponents, coefficients = _parse_gto(
        lines,
        start=sections["gto"],
        stop=sections["mo"],
        atom_positions_bohr=positions,
    )
    mo_coefficients, occupations = _parse_mos(
        lines,
        start=sections["mo"],
        ao_count=centers.shape[0],
    )
    return Route2V0GFN2MoldenAODensity(
        atom_symbols=symbols,
        atomic_numbers=numbers,
        atom_positions_bohr=positions,
        ao_centers_bohr=centers,
        ao_powers=powers,
        ao_exponents=exponents,
        ao_coefficients=coefficients,
        molecular_orbital_coefficients=mo_coefficients,
        molecular_orbital_occupations=occupations,
        molden_sha256=_sha256(path),
    )


@dataclass(frozen=True)
class Route2V0GFN2EffectiveCoreModel:
    """GFN2 effective core charges derived from the bound shipped parameter file."""

    effective_core_charge_by_atomic_number: Mapping[int, float]
    parameter_file_sha256: str
    parameter_name: str = "GFN2-xTB"
    construction: str = V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION

    def __post_init__(self) -> None:
        charges = _immutable_mapping(
            self.effective_core_charge_by_atomic_number,
            name="GFN2 effective-core model",
        )
        if (
            not isinstance(self.parameter_file_sha256, str)
            or _DIGEST.fullmatch(self.parameter_file_sha256) is None
        ):
            raise ValueError("GFN2 parameter file SHA256 is invalid.")
        if self.parameter_name != "GFN2-xTB":
            raise ValueError(
                "Route-2 V0 accepts only the declared GFN2-xTB core model."
            )
        if self.construction != V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported GFN2 effective-core construction.")
        object.__setattr__(self, "effective_core_charge_by_atomic_number", charges)

    def charges_for(self, atomic_numbers: np.ndarray) -> np.ndarray:
        """Return the parameter-file-derived effective core charge at every atom."""

        numbers = _immutable_integer_array(
            atomic_numbers,
            name="GFN2 effective-core atomic numbers",
        )
        if numbers.ndim != 1 or numbers.size == 0 or np.any(numbers < 1):
            raise ValueError(
                "GFN2 effective-core atomic numbers must be positive vectors."
            )
        unsupported = sorted(
            set(int(number) for number in numbers)
            - set(self.effective_core_charge_by_atomic_number)
        )
        if unsupported:
            raise ValueError(
                "GFN2 parameter file has no effective core declaration for "
                f"atomic numbers {unsupported}."
            )
        values = np.asarray(
            [
                self.effective_core_charge_by_atomic_number[int(number)]
                for number in numbers
            ],
            dtype=float,
        )
        values.setflags(write=False)
        return values


def load_route2_v0_gfn2_effective_core_model(
    path: Path,
) -> Route2V0GFN2EffectiveCoreModel:
    """Derive GFN2 effective core charges from its version-bound ``ao=`` records."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read GFN2 parameter file: {error}") from error
    if not any(line.strip().lower() == "name gfn2-xtb" for line in lines):
        raise ValueError("Parameter file does not identify GFN2-xTB.")
    charge_by_atomic_number: dict[int, float] = {}
    current_number: int | None = None
    current_ao: str | None = None
    for line in [*lines, "$end"]:
        matched = _PARAMETER_ELEMENT.match(line)
        if matched is not None:
            if current_number is not None:
                raise ValueError("GFN2 parameter file element block is unterminated.")
            current_number = int(matched.group(1))
            current_ao = None
            continue
        if line.strip().lower() == "$end" and current_number is not None:
            if current_ao is None:
                raise ValueError(
                    f"GFN2 parameter file omits ao= declaration for Z={current_number}."
                )
            principals = [
                int(value) for value in _ORBITAL_PRINCIPAL.findall(current_ao)
            ]
            if not principals:
                raise ValueError(
                    f"GFN2 parameter file has invalid ao= declaration for Z={current_number}."
                )
            first_principal = min(principals)
            if first_principal not in _CLOSED_SHELL_CORE_ELECTRONS:
                raise ValueError(
                    f"GFN2 parameter file declares unsupported principal shell {first_principal}."
                )
            effective_core = (
                current_number - _CLOSED_SHELL_CORE_ELECTRONS[first_principal]
            )
            if effective_core <= 0:
                raise ValueError(
                    f"GFN2 parameter file gives nonpositive effective core for Z={current_number}."
                )
            charge_by_atomic_number[current_number] = float(effective_core)
            current_number = None
            current_ao = None
            continue
        if current_number is not None:
            declaration = _AO_DECLARATION.match(line)
            if declaration is not None:
                if current_ao is not None:
                    raise ValueError(
                        f"GFN2 parameter file repeats ao= declaration for Z={current_number}."
                    )
                current_ao = declaration.group(1)
    if not charge_by_atomic_number:
        raise ValueError("GFN2 parameter file has no effective-core declarations.")
    return Route2V0GFN2EffectiveCoreModel(
        effective_core_charge_by_atomic_number=charge_by_atomic_number,
        parameter_file_sha256=_sha256(path),
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object.")
    return value


def _reported_vector(value: object, *, name: str) -> np.ndarray:
    result = _immutable_array(value, name=name, shape=(3,))
    return result


def _zero_field_gfn2_stdout_is_admissible(stdout: str) -> bool:
    lowered = stdout.lower()
    return (
        "convergence criteria satisfied" in lowered
        and re.search(r"gbsa\s+solvation\s+false", stdout, re.IGNORECASE) is not None
        and re.search(r"pc\s+potential\s+false", stdout, re.IGNORECASE) is not None
    )


@dataclass(frozen=True)
class Route2V0GFN2MoldenPermanentReference:
    """One source-bound, zero-field GFN2 valence-plus-effective-core reference."""

    density: Route2V0GFN2MoldenAODensity
    effective_core_model: Route2V0GFN2EffectiveCoreModel
    effective_core_charges_e: np.ndarray
    reported_total_dipole_e_bohr: np.ndarray
    reconstructed_total_dipole_e_bohr: np.ndarray
    total_effective_charge_e: float
    reported_total_energy_hartree: float
    xtb_version: str
    xtbout_json_sha256: str
    stdout_sha256: str
    expected_net_charge_e: float
    dipole_round_trip_error_e_bohr: float
    construction: str = V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION
    scope: str = V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.density, Route2V0GFN2MoldenAODensity):
            raise TypeError("GFN2 permanent reference requires its MOLDEN AO density.")
        if not isinstance(self.effective_core_model, Route2V0GFN2EffectiveCoreModel):
            raise TypeError(
                "GFN2 permanent reference requires its effective-core model."
            )
        cores = _immutable_array(
            self.effective_core_charges_e,
            name="GFN2 effective-core charges",
            shape=(self.density.atom_count,),
        )
        expected_cores = self.effective_core_model.charges_for(
            self.density.atomic_numbers
        )
        if not np.array_equal(cores, expected_cores):
            raise ValueError(
                "GFN2 permanent reference core charges do not match its parameter file."
            )
        reported = _reported_vector(
            self.reported_total_dipole_e_bohr,
            name="GFN2 reported total dipole",
        )
        reconstructed = _reported_vector(
            self.reconstructed_total_dipole_e_bohr,
            name="GFN2 reconstructed total dipole",
        )
        expected_charge = _finite_scalar(
            self.expected_net_charge_e,
            name="GFN2 expected net charge",
        )
        observed_charge = float(np.sum(cores) - self.density.valence_electron_count_e)
        reported_charge = _finite_scalar(
            self.total_effective_charge_e,
            name="GFN2 total effective charge",
        )
        charge_tolerance = _MOLDEN_ELECTRON_COUNT_RELATIVE_TOLERANCE * max(
            1.0,
            abs(expected_charge),
            abs(observed_charge),
        )
        if (
            abs(reported_charge - observed_charge) > charge_tolerance
            or abs(expected_charge - observed_charge) > charge_tolerance
        ):
            raise ValueError(
                "GFN2 permanent reference effective charge is inconsistent."
            )
        energy = _finite_scalar(
            self.reported_total_energy_hartree,
            name="GFN2 reported total energy",
        )
        if not isinstance(self.xtb_version, str) or not self.xtb_version.strip():
            raise ValueError("GFN2 permanent reference xTB version is invalid.")
        for name in ("xtbout_json_sha256", "stdout_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"GFN2 permanent reference {name} is invalid.")
        dipole_error = _finite_scalar(
            self.dipole_round_trip_error_e_bohr,
            name="GFN2 dipole round-trip error",
        )
        actual_error = float(np.linalg.norm(reconstructed - reported, ord=2))
        if not math.isclose(dipole_error, actual_error, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "GFN2 permanent-reference dipole round-trip error is inconsistent."
            )
        if dipole_error > _XTB_DIPOLE_ROUND_TRIP_TOLERANCE_E_BOHR:
            raise ValueError(
                "GFN2 MOLDEN source does not reproduce the xTB total dipole."
            )
        if self.construction != V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported GFN2 permanent-reference construction.")
        if self.scope != V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE:
            raise ValueError("Unsupported GFN2 permanent-reference scope.")
        object.__setattr__(self, "effective_core_charges_e", cores)
        object.__setattr__(self, "reported_total_dipole_e_bohr", reported)
        object.__setattr__(self, "reconstructed_total_dipole_e_bohr", reconstructed)
        object.__setattr__(self, "total_effective_charge_e", observed_charge)
        object.__setattr__(self, "reported_total_energy_hartree", energy)
        object.__setattr__(self, "expected_net_charge_e", expected_charge)
        object.__setattr__(self, "dipole_round_trip_error_e_bohr", actual_error)

    @property
    def nuclear_effective_core_dipole_e_bohr(self) -> np.ndarray:
        """Return ``sum_a Z_eff,a R_a`` for this valence-electron source."""

        result = np.einsum(
            "a,ax->x",
            self.effective_core_charges_e,
            self.density.atom_positions_bohr,
            optimize=True,
        )
        result = _immutable_array(
            result,
            name="GFN2 effective-core dipole",
            shape=(3,),
        )
        return result

    def grid_projection(
        self,
        grid: RegularCartesianGrid,
        *,
        maximum_points_per_block: int = 100_000,
    ) -> Route2V0AuxiliaryAODensityGridProjection:
        """Return the valence-density AO/grid dual map; core charge remains explicit."""

        return self.density.grid_projection(
            grid,
            maximum_points_per_block=maximum_points_per_block,
        )


def load_route2_v0_gfn2_molden_permanent_reference(
    *,
    molden_path: Path,
    xtbout_json_path: Path,
    stdout_path: Path,
    parameter_file_path: Path,
    expected_net_charge_e: float = 0.0,
) -> Route2V0GFN2MoldenPermanentReference:
    """Load and certify a zero-field closed-shell GFN2 MOLDEN permanent source.

    The xTB stdout must certify a converged SCC calculation without GBSA or
    external point charges.  The xTB JSON must identify GFN2-xTB, a closed
    shell, its number of valence electrons, and the reported total dipole.
    """

    density = load_route2_v0_gfn2_molden_ao_density(molden_path)
    core_model = load_route2_v0_gfn2_effective_core_model(parameter_file_path)
    payload = _load_json_object(xtbout_json_path, label="GFN2 xTB JSON output")
    if payload.get("method") != "GFN2-xTB":
        raise ValueError("xTB JSON output does not identify GFN2-xTB.")
    version = payload.get("xtb version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("xTB JSON output has no xTB version.")
    expected_orbitals = payload.get("number of molecular orbitals")
    expected_electrons = payload.get("number of electrons")
    unpaired = payload.get("number of unpaired electrons")
    if (
        isinstance(expected_orbitals, bool)
        or int(expected_orbitals) != density.ao_count
        or isinstance(unpaired, bool)
        or int(unpaired) != 0
    ):
        raise ValueError("xTB JSON output is not a matching closed-shell MOLDEN state.")
    if isinstance(expected_electrons, bool):
        raise ValueError("xTB JSON output electron count is invalid.")
    electron_count = _finite_scalar(expected_electrons, name="xTB JSON electron count")
    if abs(electron_count - density.valence_electron_count_e) > (
        _MOLDEN_ELECTRON_COUNT_RELATIVE_TOLERANCE * max(1.0, abs(electron_count))
    ):
        raise ValueError(
            "MOLDEN valence density does not match xTB JSON electron count."
        )
    reported_dipole = _reported_vector(
        payload.get("dipole / a.u."),
        name="xTB JSON total dipole",
    )
    energy = _finite_scalar(payload.get("total energy"), name="xTB JSON total energy")
    command = payload.get("program call")
    if not isinstance(command, str):
        raise ValueError("xTB JSON output has no program call.")
    lowered_command = command.lower()
    required_tokens = ("--gfn", "2", "--molden", "--json")
    if not all(token in lowered_command for token in required_tokens):
        raise ValueError(
            "xTB JSON program call does not declare zero-field GFN2 MOLDEN output."
        )
    forbidden_tokens = (
        "--alpb",
        "--gbsa",
        "--cosmo",
        "--tmcosmo",
        "--cpcmx",
        "--input",
        "--efield",
        "pcharge",
    )
    if any(token in lowered_command for token in forbidden_tokens):
        raise ValueError(
            "xTB JSON program call includes a forbidden field or solvent interface."
        )
    try:
        stdout = stdout_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot read GFN2 xTB stdout: {error}") from error
    if not _zero_field_gfn2_stdout_is_admissible(stdout):
        raise ValueError(
            "xTB stdout does not certify converged zero-field, non-solvated, non-embedded GFN2 SCC."
        )
    core_charges = core_model.charges_for(density.atomic_numbers)
    reconstructed_dipole = _immutable_array(
        np.einsum(
            "a,ax->x",
            core_charges,
            density.atom_positions_bohr,
            optimize=True,
        )
        - density.valence_electronic_dipole_e_bohr(),
        name="GFN2 reconstructed total dipole",
        shape=(3,),
    )
    expected_charge = _finite_scalar(
        expected_net_charge_e,
        name="GFN2 expected net charge",
    )
    return Route2V0GFN2MoldenPermanentReference(
        density=density,
        effective_core_model=core_model,
        effective_core_charges_e=core_charges,
        reported_total_dipole_e_bohr=reported_dipole,
        reconstructed_total_dipole_e_bohr=reconstructed_dipole,
        total_effective_charge_e=float(
            np.sum(core_charges) - density.valence_electron_count_e
        ),
        reported_total_energy_hartree=energy,
        xtb_version=version,
        xtbout_json_sha256=_sha256(xtbout_json_path),
        stdout_sha256=_sha256(stdout_path),
        expected_net_charge_e=expected_charge,
        dipole_round_trip_error_e_bohr=float(
            np.linalg.norm(reconstructed_dipole - reported_dipole, ord=2)
        ),
    )


__all__ = [
    "V0_GFN2_MOLDEN_PERMANENT_SOURCE_CONSTRUCTION",
    "V0_GFN2_MOLDEN_PERMANENT_SOURCE_SCOPE",
    "Route2V0GFN2EffectiveCoreModel",
    "Route2V0GFN2MoldenAODensity",
    "Route2V0GFN2MoldenPermanentReference",
    "load_route2_v0_gfn2_effective_core_model",
    "load_route2_v0_gfn2_molden_ao_density",
    "load_route2_v0_gfn2_molden_permanent_reference",
]
