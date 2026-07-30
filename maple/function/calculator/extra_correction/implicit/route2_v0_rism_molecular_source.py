"""Fail-closed molecular-source audit for Route-2 V0 RISM assets.

The AMBER MDL model, its ``rism1d`` input, and the generated XVV metadata are
three serializations of one molecular liquid.  This module requires them to
agree on molecular composition, geometry, state, grid, and AMBER's documented
unit conversions before the source can be bound to a frozen solvent asset.
It performs no fitting, calibration, training, or solvent-selection logic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.units import Bohr

from .route2_v0_rism_bulk import (
    Route2V0RismXvvMetadata,
    parse_rism1d_xvv_metadata,
)

V0_RISM_MOLECULAR_SOURCE_CONSTRUCTION = "route2-v0-rism-molecular-source-v1"
AMBER_ELECTROSTATIC_CHARGE_SCALE = 18.2223
BOLTZMANN_KCAL_MOL_K = 0.00198720425864083
AVOGADRO_PER_ANGSTROM3_PER_MOLAR = 6.02214076e-4
# AmberTools text MDL/input files and XVV output round independently.  The
# bundled cSPC/E source differs by about 1.5e-5 in reduced charge and less than
# 1e-6 in density, so this is a source-serialization tolerance, not a fitted
# solvent parameter.
SOURCE_SERIALIZATION_RELATIVE_TOLERANCE = 5.0e-5


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite.")
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    number = _finite_float(value, name=name)
    integer = int(number)
    if number != integer or integer <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _integer(value: object, *, name: str) -> int:
    number = _finite_float(value, name=name)
    integer = int(number)
    if number != integer:
        raise ValueError(f"{name} must be an integer.")
    return integer


def _immutable_float_array(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...],
    nonnegative: bool = False,
) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and have shape {shape}.") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and have shape {shape}.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _immutable_atomic_numbers(values: object, *, count: int) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.shape != (count,)
        or not np.issubdtype(array.dtype, np.integer)
        or np.any(array <= 0)
        or np.any(array >= len(atomic_masses))
    ):
        raise ValueError(
            "Molecular-source atomic numbers must be valid positive integers."
        )
    result = np.array(array, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _flag_sections(text: str) -> dict[str, list[str]]:
    """Split AMBER ``%FLAG`` data while ignoring format/comment records."""

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("%FLAG"):
            fields = upper.split(maxsplit=1)
            if len(fields) != 2 or not fields[1].strip():
                raise ValueError("Malformed AMBER %FLAG record.")
            current = fields[1].strip()
            if current in sections:
                raise ValueError(f"Duplicate AMBER %{current} section.")
            sections[current] = []
            continue
        if upper.startswith(("%FORMAT", "%COMMENT", "%VERSION")):
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _required_section(sections: dict[str, list[str]], name: str) -> list[str]:
    values = sections.get(name)
    if not values:
        raise ValueError(f"AMBER source must contain %{name}.")
    return values


def _tokens(lines: Iterable[str], *, name: str) -> list[str]:
    values = [token for line in lines for token in line.split()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def _floats(lines: Iterable[str], *, name: str) -> np.ndarray:
    return np.asarray(
        [_finite_float(token, name=name) for token in _tokens(lines, name=name)],
        dtype=float,
    )


def _integers(lines: Iterable[str], *, name: str) -> np.ndarray:
    return np.asarray(
        [_positive_integer(token, name=name) for token in _tokens(lines, name=name)],
        dtype=np.int64,
    )


def _element_candidates(site_name: str) -> tuple[str, ...]:
    raw = _nonempty_string(site_name, name="AMBER site name")
    upper = raw.upper()
    if re.fullmatch(r"(?:EP|LP|DU|DUM|MW|M|X)\d*", upper):
        raise ValueError(f"AMBER site name {site_name!r} is a virtual-site pattern.")
    letters = "".join(character for character in raw if character.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer an element from AMBER site {site_name!r}.")
    candidates: list[str] = []
    if len(letters) >= 2:
        symbol = letters[0].upper() + letters[1].lower()
        if symbol in atomic_numbers:
            candidates.append(symbol)
    symbol = letters[0].upper()
    if symbol in atomic_numbers and symbol not in candidates:
        candidates.append(symbol)
    if not candidates:
        raise ValueError(f"Cannot infer an element from AMBER site {site_name!r}.")
    return tuple(candidates)


def _atomic_number_from_site_mass(site_name: str, mass_amu: float) -> int:
    if not math.isfinite(mass_amu) or mass_amu <= 0.0:
        raise ValueError("All Route-2 V0 solvent sites must be massive atomic sites.")
    matches: list[tuple[float, int]] = []
    for symbol in _element_candidates(site_name):
        number = atomic_numbers[symbol]
        expected = float(atomic_masses[number])
        absolute = abs(mass_amu - expected)
        relative = absolute / max(1.0, expected)
        if absolute <= 0.11 or relative <= 0.06:
            matches.append((relative, number))
    if not matches:
        raise ValueError(
            f"AMBER site {site_name!r} mass is not an all-atom element mass; "
            "united-atom and virtual-site sources are not admitted."
        )
    matches.sort()
    return matches[0][1]


def _relative_allclose(left: object, right: object, *, tolerance: float) -> bool:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return False
    scale = np.maximum(1.0e-12, np.maximum(np.abs(left_array), np.abs(right_array)))
    return bool(np.all(np.abs(left_array - right_array) <= tolerance * scale))


def _atom_distance_signature(
    *,
    atom_index: int,
    positions_angstrom: np.ndarray,
    atom_site_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Return one label-resolved rigid-distance signature."""

    buckets: dict[str, list[float]] = {}
    for other, label in enumerate(atom_site_names):
        if other != atom_index:
            buckets.setdefault(label, []).append(
                float(
                    np.linalg.norm(
                        positions_angstrom[atom_index] - positions_angstrom[other]
                    )
                )
            )
    return {key: np.sort(values) for key, values in buckets.items()}


def _same_distance_signature(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    tolerance_angstrom: float,
) -> bool:
    return bool(
        left.keys() == right.keys()
        and all(
            np.allclose(
                left[label],
                right[label],
                rtol=0.0,
                atol=tolerance_angstrom,
            )
            for label in left
        )
    )


def _proper_rotation_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    """Return best labelled RMSD using translations and a proper rotation only."""

    left_centered = left - np.mean(left, axis=0)
    right_centered = right - np.mean(right, axis=0)
    left_rank = np.linalg.matrix_rank(left_centered, tol=1.0e-12)
    right_rank = np.linalg.matrix_rank(right_centered, tol=1.0e-12)
    if left_rank != right_rank:
        return math.inf
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return float(
            np.sqrt(np.mean(np.sum((left_centered - right_centered) ** 2, axis=1)))
        )
    covariance = left_centered.T @ right_centered
    left_singular, _, right_singular = np.linalg.svd(covariance)
    rotation = left_singular @ right_singular
    if np.linalg.det(rotation) < 0.0:
        left_singular[:, -1] *= -1.0
        rotation = left_singular @ right_singular
    displacement = left_centered @ rotation - right_centered
    return float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1))))


def _same_labelled_geometry(
    left_positions: np.ndarray,
    left_names: tuple[str, ...],
    right_positions: np.ndarray,
    right_names: tuple[str, ...],
    *,
    require_proper_rotation: bool,
) -> bool:
    """Match labelled pairwise distances, optionally requiring proper rotation."""

    tolerance = 5.0e-7
    if (
        left_positions.shape != right_positions.shape
        or len(left_names) != len(right_names)
        or sorted(left_names) != sorted(right_names)
    ):
        return False
    count = len(left_names)
    left_signatures = tuple(
        _atom_distance_signature(
            atom_index=index,
            positions_angstrom=left_positions,
            atom_site_names=left_names,
        )
        for index in range(count)
    )
    right_signatures = tuple(
        _atom_distance_signature(
            atom_index=index,
            positions_angstrom=right_positions,
            atom_site_names=right_names,
        )
        for index in range(count)
    )
    candidates = {
        left: tuple(
            right
            for right in range(count)
            if left_names[left] == right_names[right]
            and _same_distance_signature(
                left_signatures[left],
                right_signatures[right],
                tolerance_angstrom=tolerance,
            )
        )
        for left in range(count)
    }
    if any(not values for values in candidates.values()):
        return False
    order = tuple(
        sorted(range(count), key=lambda index: (len(candidates[index]), index))
    )
    mapping = np.full(count, -1, dtype=np.int64)
    used = np.zeros(count, dtype=bool)

    def visit(depth: int) -> bool:
        if depth == count:
            if not require_proper_rotation:
                return True
            reordered_right = right_positions[mapping]
            return _proper_rotation_rmsd(left_positions, reordered_right) <= tolerance
        left = order[depth]
        for right in candidates[left]:
            if used[right]:
                continue
            compatible = True
            for previous_depth in range(depth):
                previous_left = order[previous_depth]
                previous_right = int(mapping[previous_left])
                left_distance = np.linalg.norm(
                    left_positions[left] - left_positions[previous_left]
                )
                right_distance = np.linalg.norm(
                    right_positions[right] - right_positions[previous_right]
                )
                if abs(float(left_distance - right_distance)) > tolerance:
                    compatible = False
                    break
            if not compatible:
                continue
            mapping[left] = right
            used[right] = True
            if visit(depth + 1):
                return True
            used[right] = False
            mapping[left] = -1
        return False

    return visit(0)


def same_labelled_distance_geometry(
    left_positions: np.ndarray,
    left_names: tuple[str, ...],
    right_positions: np.ndarray,
    right_names: tuple[str, ...],
) -> bool:
    """Require labelled pairwise-distance congruence, allowing reflection.

    AmberTools may choose independent signs for principal inertia axes when it
    serializes XVV coordinates.  That output convention can reflect an MDL
    geometry without changing the labelled molecular distance matrix.
    """

    return _same_labelled_geometry(
        left_positions,
        left_names,
        right_positions,
        right_names,
        require_proper_rotation=False,
    )


def same_labelled_rigid_geometry(
    left_positions: np.ndarray,
    left_names: tuple[str, ...],
    right_positions: np.ndarray,
    right_names: tuple[str, ...],
) -> bool:
    """Require labelled congruence under translation and proper rotation.

    Pairwise distances alone cannot distinguish a chiral molecule from its
    mirror image.  The canonical manifest-to-MDL boundary therefore applies a
    proper Kabsch check after label-constrained distance matching.
    """

    return _same_labelled_geometry(
        left_positions,
        left_names,
        right_positions,
        right_names,
        require_proper_rotation=True,
    )


def _neutral_charge_projection(charges_e: np.ndarray) -> np.ndarray:
    """Project serialized atom charges onto exact molecular neutrality.

    AMBER MDL charges are rounded decimal serializations of a neutral model.
    Once the residual has passed the source-rounding gate, the unique
    least-squares correction is the orthogonal projection

    ``q_neutral = q_serialized - 1 (1^T q_serialized) / N``.

    The final component absorbs only the floating-point summation residue so
    the serialized array has an exact zero sum.  This is a fixed algebraic
    constraint projection, not an accuracy fit.
    """

    charges = np.asarray(charges_e, dtype=float)
    projected = charges - float(np.sum(charges)) / charges.size
    projected[-1] -= float(np.sum(projected))
    result = np.array(projected, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _namelist_blocks(text: str) -> dict[str, str]:
    """Return distinct AMBER namelist bodies with inline comments removed."""

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        for marker in ("!", "#"):
            if marker in line:
                line = line.split(marker, 1)[0]
        if line.strip():
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    pattern = re.compile(
        r"&([A-Za-z][A-Za-z0-9_]*)\b(.*?)(?:^\s*/\s*$|&END\b)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    blocks: dict[str, str] = {}
    matches = list(pattern.finditer(cleaned))
    remainder = "".join(
        cleaned[end:start]
        for end, start in zip(
            (0, *(match.end() for match in matches)),
            (*(match.start() for match in matches), len(cleaned)),
            strict=True,
        )
    ).strip()
    if remainder:
        raise ValueError(f"rism1d input contains unparsed content {remainder!r}.")
    for match in matches:
        name = match.group(1).upper()
        if name in blocks:
            raise ValueError(f"rism1d input contains duplicate &{name} blocks.")
        blocks[name] = match.group(2)
    return blocks


def _assignments(block: str, *, name: str) -> dict[str, str]:
    pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s]+))",
        re.IGNORECASE,
    )
    result: dict[str, str] = {}
    for match in pattern.finditer(block):
        key = match.group(1).lower()
        value = next(
            candidate for candidate in match.groups()[1:] if candidate is not None
        ).strip()
        if key in result:
            raise ValueError(f"{name} contains duplicate {key!r} assignments.")
        if not value:
            raise ValueError(f"{name}.{key} must be nonempty.")
        result[key] = value
    remainder = pattern.sub("", block).replace(",", "").strip()
    if remainder:
        raise ValueError(f"{name} contains unparsed content {remainder!r}.")
    if not result:
        raise ValueError(f"{name} must contain assignments.")
    return result


def _required_assignment(values: dict[str, str], key: str, *, block: str) -> str:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"{block} must declare {key.upper()}.") from exc


@dataclass(frozen=True)
class Route2V0Rism1dInput:
    """Validated single-component molecular ``rism1d`` input state."""

    theory: str
    closure: str
    radial_point_count: int
    radial_spacing_angstrom: float
    temperature_kelvin: float
    component_count: int
    dielectric_constant: float
    molecular_number_density_angstrom3: float
    coulomb_smear_angstrom: float
    residual_tolerance: float
    self_test: int
    output_list: str
    maximum_steps: int
    model: str

    def __post_init__(self) -> None:
        theory = _nonempty_string(self.theory, name="rism1d theory").upper()
        closure = _nonempty_string(self.closure, name="rism1d closure").upper()
        model = _nonempty_string(self.model, name="rism1d model")
        points = _positive_integer(self.radial_point_count, name="rism1d NR")
        components = _positive_integer(self.component_count, name="rism1d NSP")
        spacing = _finite_float(self.radial_spacing_angstrom, name="rism1d DR")
        temperature = _finite_float(self.temperature_kelvin, name="rism1d temperature")
        dielectric = _finite_float(self.dielectric_constant, name="rism1d dielectric")
        density = _finite_float(
            self.molecular_number_density_angstrom3,
            name="rism1d molecular density",
        )
        smear = _finite_float(self.coulomb_smear_angstrom, name="rism1d SMEAR")
        tolerance = _finite_float(
            self.residual_tolerance,
            name="rism1d residual tolerance",
        )
        self_test = _integer(self.self_test, name="rism1d SELFTEST")
        output_list = _nonempty_string(
            self.output_list,
            name="rism1d OUTLIST",
        ).lower()
        maximum_steps = _positive_integer(
            self.maximum_steps,
            name="rism1d MAXSTEP",
        )
        if theory not in {"DRISM", "XRISM"}:
            raise ValueError("rism1d THEORY must be DRISM or XRISM.")
        if components != 1:
            raise ValueError("Route-2 V0 molecular sources require NSP=1.")
        if points < 2 or spacing <= 0.0:
            raise ValueError("rism1d radial grid must be positive and nontrivial.")
        if (
            temperature <= 0.0
            or dielectric <= 1.0
            or density <= 0.0
            or smear <= 0.0
            or tolerance <= 0.0
        ):
            raise ValueError("rism1d thermodynamic state and SMEAR must be physical.")
        if self_test != -1:
            raise ValueError("Route-2 V0 requires rism1d SELFTEST=-1.")
        if not {"x", "c"}.issubset(output_list):
            raise ValueError("Route-2 V0 rism1d OUTLIST must request XVV and Cvv.")
        object.__setattr__(self, "theory", theory)
        object.__setattr__(self, "closure", closure)
        object.__setattr__(self, "radial_point_count", points)
        object.__setattr__(self, "radial_spacing_angstrom", spacing)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "component_count", components)
        object.__setattr__(self, "dielectric_constant", dielectric)
        object.__setattr__(self, "molecular_number_density_angstrom3", density)
        object.__setattr__(self, "coulomb_smear_angstrom", smear)
        object.__setattr__(self, "residual_tolerance", tolerance)
        object.__setattr__(self, "self_test", self_test)
        object.__setattr__(self, "output_list", output_list)
        object.__setattr__(self, "maximum_steps", maximum_steps)
        object.__setattr__(self, "model", model)

    @property
    def nr(self) -> int:
        return self.radial_point_count

    @property
    def dr(self) -> float:
        return self.radial_spacing_angstrom

    @property
    def dielectric(self) -> float:
        return self.dielectric_constant

    @property
    def density(self) -> float:
        return self.molecular_number_density_angstrom3


@dataclass(frozen=True)
class Route2V0MolecularRismSource:
    """One all-atom MDL/input/XVV source identity."""

    atomic_numbers: np.ndarray
    site_charges_e: np.ndarray
    serialized_site_charges_e: np.ndarray
    reference_positions_bohr: np.ndarray
    atom_site_names: tuple[str, ...]
    per_atom_lj_epsilon_kcal_per_mol: np.ndarray
    per_atom_lj_rmin_half_angstrom: np.ndarray
    per_atom_masses_amu: np.ndarray
    rism1d_input: Route2V0Rism1dInput
    metadata: Route2V0RismXvvMetadata
    construction: str = V0_RISM_MOLECULAR_SOURCE_CONSTRUCTION

    def __post_init__(self) -> None:
        if self.construction != V0_RISM_MOLECULAR_SOURCE_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 V0 molecular-source construction.")
        names = tuple(
            _nonempty_string(value, name="Molecular-source atom-site name")
            for value in self.atom_site_names
        )
        count = len(names)
        if count == 0:
            raise ValueError("Molecular source must contain at least one atomic site.")
        numbers = _immutable_atomic_numbers(self.atomic_numbers, count=count)
        charges = _immutable_float_array(
            self.site_charges_e,
            name="Molecular-source site charges",
            shape=(count,),
        )
        serialized_charges = _immutable_float_array(
            self.serialized_site_charges_e,
            name="Serialized molecular-source site charges",
            shape=(count,),
        )
        positions = _immutable_float_array(
            self.reference_positions_bohr,
            name="Molecular-source positions",
            shape=(count, 3),
        )
        epsilon = _immutable_float_array(
            self.per_atom_lj_epsilon_kcal_per_mol,
            name="Molecular-source LJ epsilon",
            shape=(count,),
            nonnegative=True,
        )
        rmin_half = _immutable_float_array(
            self.per_atom_lj_rmin_half_angstrom,
            name="Molecular-source LJ Rmin/2",
            shape=(count,),
            nonnegative=True,
        )
        masses = _immutable_float_array(
            self.per_atom_masses_amu,
            name="Molecular-source masses",
            shape=(count,),
        )
        expected_numbers = np.asarray(
            [
                _atomic_number_from_site_mass(name, float(mass))
                for name, mass in zip(names, masses, strict=True)
            ],
            dtype=np.int64,
        )
        if not np.array_equal(numbers, expected_numbers):
            raise ValueError(
                "Molecular-source atomic numbers disagree with MDL names/masses."
            )
        serialized_scale = max(1.0, float(np.sum(np.abs(serialized_charges))))
        if abs(float(np.sum(serialized_charges))) > 1.0e-8 * serialized_scale:
            raise ValueError(
                "Serialized molecular-source charges exceed rounding error."
            )
        if float(np.sum(charges)) != 0.0:
            raise ValueError("Molecular-source site charges must be exactly neutral.")
        if not np.allclose(
            charges,
            _neutral_charge_projection(serialized_charges),
            rtol=0.0,
            atol=np.finfo(float).eps,
        ):
            raise ValueError(
                "Molecular-source site charges must be the fixed neutral projection "
                "of the serialized MDL charges."
            )
        if np.any((epsilon > 0.0) & (rmin_half <= 0.0)):
            raise ValueError("A nonzero LJ epsilon requires a positive Rmin/2.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "site_charges_e", charges)
        object.__setattr__(self, "serialized_site_charges_e", serialized_charges)
        object.__setattr__(self, "reference_positions_bohr", positions)
        object.__setattr__(self, "atom_site_names", names)
        object.__setattr__(self, "per_atom_lj_epsilon_kcal_per_mol", epsilon)
        object.__setattr__(self, "per_atom_lj_rmin_half_angstrom", rmin_half)
        object.__setattr__(self, "per_atom_masses_amu", masses)


def parse_route2_v0_rism1d_input(text: str) -> Route2V0Rism1dInput:
    """Parse the real two-block AMBER ``rism1d`` input convention."""

    blocks = _namelist_blocks(text)
    if set(blocks) != {"PARAMETERS", "SPECIES"}:
        raise ValueError("rism1d input must contain exactly &PARAMETERS and &SPECIES.")
    parameters = _assignments(blocks["PARAMETERS"], name="&PARAMETERS")
    species = _assignments(blocks["SPECIES"], name="&SPECIES")
    units = _required_assignment(species, "units", block="&SPECIES").lower()
    if units != "m":
        raise ValueError("Route-2 V0 currently requires rism1d DENSITY units of M.")
    density_molar = _finite_float(
        _required_assignment(species, "density", block="&SPECIES"),
        name="rism1d DENSITY",
    )
    return Route2V0Rism1dInput(
        theory=_required_assignment(parameters, "theory", block="&PARAMETERS"),
        closure=_required_assignment(parameters, "closure", block="&PARAMETERS"),
        radial_point_count=_positive_integer(
            _required_assignment(parameters, "nr", block="&PARAMETERS"),
            name="rism1d NR",
        ),
        radial_spacing_angstrom=_finite_float(
            _required_assignment(parameters, "dr", block="&PARAMETERS"),
            name="rism1d DR",
        ),
        temperature_kelvin=_finite_float(
            _required_assignment(parameters, "temperature", block="&PARAMETERS"),
            name="rism1d TEMPERATURE",
        ),
        component_count=_positive_integer(
            _required_assignment(parameters, "nsp", block="&PARAMETERS"),
            name="rism1d NSP",
        ),
        dielectric_constant=_finite_float(
            _required_assignment(parameters, "dieps", block="&PARAMETERS"),
            name="rism1d DIEPS",
        ),
        molecular_number_density_angstrom3=(
            density_molar * AVOGADRO_PER_ANGSTROM3_PER_MOLAR
        ),
        coulomb_smear_angstrom=_finite_float(
            _required_assignment(parameters, "smear", block="&PARAMETERS"),
            name="rism1d SMEAR",
        ),
        residual_tolerance=_finite_float(
            _required_assignment(parameters, "tolerance", block="&PARAMETERS"),
            name="rism1d TOLERANCE",
        ),
        self_test=_integer(
            _required_assignment(parameters, "selftest", block="&PARAMETERS"),
            name="rism1d SELFTEST",
        ),
        output_list=_required_assignment(parameters, "outlist", block="&PARAMETERS"),
        maximum_steps=_positive_integer(
            _required_assignment(parameters, "maxstep", block="&PARAMETERS"),
            name="rism1d MAXSTEP",
        ),
        model=_required_assignment(species, "model", block="&SPECIES"),
    )


def parse_route2_v0_rism_molecular_source(
    *,
    mdl_text: str,
    rism1d_input: str,
    xvv_text: str,
    site_model_filename: str,
) -> Route2V0MolecularRismSource:
    """Parse and cross-check one AMBER MDL/input/XVV molecular source."""

    expected_filename = Path(
        _nonempty_string(site_model_filename, name="Site-model filename")
    ).name
    if expected_filename != site_model_filename:
        raise ValueError("Site-model filename must be one basename without a path.")

    mdl = _flag_sections(mdl_text)
    pointers = _integers(_required_section(mdl, "POINTERS"), name="MDL POINTERS")
    if pointers.size < 2:
        raise ValueError("MDL POINTERS must declare atom and site-type counts.")
    atom_count, site_type_count = (int(value) for value in pointers[:2])
    type_order = _integers(_required_section(mdl, "ATMTYP"), name="MDL ATMTYP")
    if type_order.shape != (site_type_count,) or sorted(type_order.tolist()) != list(
        range(1, site_type_count + 1)
    ):
        raise ValueError("MDL ATMTYP must be a one-based site-type permutation.")
    type_order = type_order - 1

    site_names = tuple(_tokens(_required_section(mdl, "ATMNAME"), name="MDL ATMNAME"))
    if len(site_names) != site_type_count:
        raise ValueError("MDL ATMNAME count must equal the site-type count.")
    multiplicity = _integers(_required_section(mdl, "MULTI"), name="MDL MULTI")
    if (
        multiplicity.shape != (site_type_count,)
        or int(np.sum(multiplicity)) != atom_count
    ):
        raise ValueError("MDL MULTI must account for every atomic site.")
    masses = _immutable_float_array(
        _floats(_required_section(mdl, "MASS"), name="MDL MASS"),
        name="MDL MASS",
        shape=(site_type_count,),
    )
    charges_amber = _immutable_float_array(
        _floats(_required_section(mdl, "CHG"), name="MDL CHG"),
        name="MDL CHG",
        shape=(site_type_count,),
    )
    epsilon = _immutable_float_array(
        _floats(_required_section(mdl, "LJEPSILON"), name="MDL LJEPSILON"),
        name="MDL LJEPSILON",
        shape=(site_type_count,),
        nonnegative=True,
    )
    rmin_half = _immutable_float_array(
        _floats(_required_section(mdl, "LJSIGMA"), name="MDL LJSIGMA"),
        name="MDL LJSIGMA",
        shape=(site_type_count,),
        nonnegative=True,
    )
    if np.any((epsilon > 0.0) & (rmin_half <= 0.0)):
        raise ValueError("An MDL site with nonzero LJ epsilon needs positive Rmin/2.")
    coordinates_angstrom = _immutable_float_array(
        _floats(_required_section(mdl, "COORD"), name="MDL COORD").reshape((-1, 3)),
        name="MDL COORD",
        shape=(atom_count, 3),
    )

    atom_type_indices = np.concatenate(
        [np.repeat(index, multiplicity[index]) for index in type_order]
    ).astype(np.int64)
    atom_site_names = tuple(site_names[index] for index in atom_type_indices)
    per_atom_masses = masses[atom_type_indices]
    atomic_numbers_out = np.asarray(
        [
            _atomic_number_from_site_mass(name, float(mass))
            for name, mass in zip(atom_site_names, per_atom_masses, strict=True)
        ],
        dtype=np.int64,
    )
    serialized_per_atom_charges_e = (
        charges_amber[atom_type_indices] / AMBER_ELECTROSTATIC_CHARGE_SCALE
    )
    per_atom_charges_e = _neutral_charge_projection(serialized_per_atom_charges_e)
    weighted_charge = float(np.dot(charges_amber, multiplicity))
    weighted_scale = max(1.0, float(np.dot(np.abs(charges_amber), multiplicity)))
    if abs(weighted_charge) > 1.0e-8 * weighted_scale:
        raise ValueError("MDL solvent model must be electrically neutral.")

    metadata = parse_rism1d_xvv_metadata(xvv_text)
    xvv = _flag_sections(xvv_text)
    if metadata.component_count != 1:
        raise ValueError("Route-2 V0 molecular sources require one XVV species.")
    if metadata.site_names != site_names:
        raise ValueError("MDL ATMNAME and XVV ATOM_NAME must match exactly.")
    if not np.array_equal(metadata.site_multiplicity, multiplicity):
        raise ValueError("MDL MULTI and XVV MTV must match exactly.")

    xvv_masses = _floats(_required_section(xvv, "MASS"), name="XVV MASS")
    xvv_epsilon = _floats(_required_section(xvv, "EPSV"), name="XVV EPSV")
    xvv_rmin_half = _floats(_required_section(xvv, "RMIN2V"), name="XVV RMIN2V")
    species_types = _integers(_required_section(xvv, "NVSP"), name="XVV NVSP")
    species_density = _floats(_required_section(xvv, "RHOSP"), name="XVV RHOSP")
    xvv_coordinates = _immutable_float_array(
        _floats(_required_section(xvv, "COORD"), name="XVV COORD").reshape((-1, 3)),
        name="XVV COORD",
        shape=(atom_count, 3),
    )
    if (
        xvv_masses.shape != (site_type_count,)
        or xvv_epsilon.shape != (site_type_count,)
        or xvv_rmin_half.shape != (site_type_count,)
        or species_types.shape != (1,)
        or species_density.shape != (1,)
        or int(species_types[0]) != site_type_count
    ):
        raise ValueError(
            "XVV molecular metadata dimensions disagree with MDL POINTERS."
        )
    if not _relative_allclose(
        masses,
        xvv_masses,
        tolerance=SOURCE_SERIALIZATION_RELATIVE_TOLERANCE,
    ):
        raise ValueError("MDL and XVV site masses disagree.")

    thermal_energy = BOLTZMANN_KCAL_MOL_K * metadata.temperature_kelvin
    if not _relative_allclose(
        charges_amber / math.sqrt(thermal_energy),
        metadata.site_charges_sqrt_kT_angstrom,
        tolerance=SOURCE_SERIALIZATION_RELATIVE_TOLERANCE,
    ):
        raise ValueError("MDL CHG and XVV QV reduced-unit values disagree.")
    if not _relative_allclose(
        epsilon / thermal_energy,
        xvv_epsilon,
        tolerance=SOURCE_SERIALIZATION_RELATIVE_TOLERANCE,
    ):
        raise ValueError("MDL LJEPSILON and XVV EPSV reduced-unit values disagree.")
    if not _relative_allclose(
        rmin_half,
        xvv_rmin_half,
        tolerance=SOURCE_SERIALIZATION_RELATIVE_TOLERANCE,
    ):
        raise ValueError("MDL LJSIGMA and XVV RMIN2V values disagree.")

    if not same_labelled_distance_geometry(
        coordinates_angstrom,
        atom_site_names,
        xvv_coordinates,
        atom_site_names,
    ):
        raise ValueError("MDL and XVV labelled molecular geometries disagree.")

    molecular_density = metadata.bulk_number_density_angstrom3 / multiplicity
    if not _relative_allclose(
        molecular_density,
        np.full(site_type_count, species_density[0]),
        tolerance=SOURCE_SERIALIZATION_RELATIVE_TOLERANCE,
    ):
        raise ValueError("XVV RHOV/MTV and RHOSP molecular densities disagree.")

    control = parse_route2_v0_rism1d_input(rism1d_input)
    if Path(control.model).name != expected_filename:
        raise ValueError("rism1d MODEL must name the hash-bound MDL source file.")
    scalar_checks = (
        (control.temperature_kelvin, metadata.temperature_kelvin, "temperature"),
        (control.dielectric_constant, metadata.dielectric_constant, "dielectric"),
        (control.radial_spacing_angstrom, metadata.radial_spacing_angstrom, "DR"),
        (control.coulomb_smear_angstrom, metadata.coulomb_smear_angstrom, "SMEAR"),
        (control.molecular_number_density_angstrom3, species_density[0], "density"),
    )
    for observed, expected, label in scalar_checks:
        if not _relative_allclose(
            [observed],
            [expected],
            tolerance=SOURCE_SERIALIZATION_RELATIVE_TOLERANCE,
        ):
            raise ValueError(f"rism1d input and XVV {label} disagree.")
    if control.radial_point_count != metadata.radial_point_count:
        raise ValueError("rism1d input and XVV NR disagree.")
    if control.component_count != metadata.component_count:
        raise ValueError("rism1d input and XVV component counts disagree.")

    return Route2V0MolecularRismSource(
        atomic_numbers=atomic_numbers_out,
        site_charges_e=per_atom_charges_e,
        serialized_site_charges_e=serialized_per_atom_charges_e,
        reference_positions_bohr=coordinates_angstrom / Bohr,
        atom_site_names=atom_site_names,
        per_atom_lj_epsilon_kcal_per_mol=epsilon[atom_type_indices],
        per_atom_lj_rmin_half_angstrom=rmin_half[atom_type_indices],
        per_atom_masses_amu=per_atom_masses,
        rism1d_input=control,
        metadata=metadata,
    )


def load_route2_v0_rism_molecular_source(
    *,
    mdl_path: str | Path,
    rism1d_input_path: str | Path,
    xvv_path: str | Path,
) -> Route2V0MolecularRismSource:
    """Load and audit one MDL/input/XVV source triple."""

    mdl = Path(mdl_path)
    return parse_route2_v0_rism_molecular_source(
        mdl_text=mdl.read_text(encoding="utf-8"),
        rism1d_input=Path(rism1d_input_path).read_text(encoding="utf-8"),
        xvv_text=Path(xvv_path).read_text(encoding="utf-8"),
        site_model_filename=mdl.name,
    )


__all__ = [
    "AMBER_ELECTROSTATIC_CHARGE_SCALE",
    "AVOGADRO_PER_ANGSTROM3_PER_MOLAR",
    "BOLTZMANN_KCAL_MOL_K",
    "SOURCE_SERIALIZATION_RELATIVE_TOLERANCE",
    "V0_RISM_MOLECULAR_SOURCE_CONSTRUCTION",
    "Route2V0MolecularRismSource",
    "Route2V0Rism1dInput",
    "load_route2_v0_rism_molecular_source",
    "parse_route2_v0_rism1d_input",
    "parse_route2_v0_rism_molecular_source",
    "same_labelled_distance_geometry",
    "same_labelled_rigid_geometry",
]
