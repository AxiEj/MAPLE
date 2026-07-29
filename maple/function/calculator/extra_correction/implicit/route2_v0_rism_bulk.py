"""Frozen 1D-RISM bulk-correlation assets for the Route-2 V0-FD-S path.

This module parses only *bulk-solvent* output from ``rism1d``: the companion
``.xvv`` metadata and the radial ``.cvv`` direct-correlation table.  It never
accepts an AMBER solute topology, atom charges, Lennard-Jones parameters, or a
3D-RISM solute calculation.  The result is therefore a source-bound liquid
asset that may later couple to the MACE Gaussian grid potential.

A raw ionic-site direct correlation has the analytic large-distance form
``c_ab(r) = c_ab^sr(r) - q_a q_b / r`` in Amber's ``QV`` convention.  A finite
Cartesian FFT cannot consume that raw tail safely: wrapping it changes the
operator and subtracting it at ``r = 0`` invents a value.  The parser exposes
only the positive-radius short-range remainder and deliberately supplies no
3D interpolation until a separately declared Coulomb/Ewald operator exists.

This is an asset boundary, not a total-solvation backend, a physical solvent
registry, or an accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION = (
    "route2-v0-rism1d-bulk-direct-correlation-v1"
)


def _immutable_real_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
) -> np.ndarray:
    """Return one finite immutable real array with the declared shape."""

    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or not np.all(np.isfinite(array))
        or (positive and np.any(array <= 0.0))
    ):
        expected = "a finite array" if shape is None else f"a finite array with {shape}"
        if positive:
            expected += " and strictly positive entries"
        raise ValueError(f"{name} must be {expected}.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _site_names(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a stable ordered RISM-site name sequence."""

    names = tuple(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("RISM site names must be nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("RISM site names must be unique.")
    return names


def _integer_at_least(value: object, *, name: str, minimum: int) -> int:
    """Return one strict integer rather than silently truncating a control value."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer of at least {minimum}.") from exc
    if integer != value or integer < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return integer


def _fortran_floats(lines: list[str], *, name: str) -> list[float]:
    """Read whitespace-separated Fortran floating-point fields."""

    values: list[float] = []
    for line in lines:
        for token in line.split():
            try:
                value = float(token.replace("D", "E").replace("d", "e"))
            except ValueError as exc:
                raise ValueError(
                    f"{name} contains a nonnumeric field: {token!r}."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} contains a non-finite field.")
            values.append(value)
    return values


def _xvv_flag_data(text: str) -> dict[str, list[str]]:
    """Return metadata blocks preceding the potentially large ``XVV`` payload."""

    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("%FLAG"):
            flag = line.split(maxsplit=1)
            if len(flag) != 2:
                raise ValueError("Malformed RISM XVV flag.")
            current = flag[1].strip()
            if current == "XVV":
                break
            blocks.setdefault(current, [])
            continue
        if current is None or line.startswith("%FORMAT") or line.startswith("%COMMENT"):
            continue
        if line:
            blocks[current].append(line)
    return blocks


def _require_xvv_block(
    blocks: dict[str, list[str]],
    name: str,
) -> list[str]:
    """Return one required XVV metadata block."""

    values = blocks.get(name)
    if not values:
        raise ValueError(f"RISM XVV metadata is missing %{name}.")
    return values


@dataclass(frozen=True)
class Route2V0RismXvvMetadata:
    """Physical-state metadata read from a frozen 1D-RISM ``.xvv`` file.

    ``site_charges_sqrt_kT_angstrom`` retains Amber's native ``QV`` scale.  It
    is intentionally not reinterpreted as an elementary-charge vector here;
    the matching radial direct-correlation table uses the same convention.
    """

    site_names: tuple[str, ...]
    site_multiplicity: np.ndarray
    bulk_number_density_angstrom3: np.ndarray
    site_charges_sqrt_kT_angstrom: np.ndarray
    temperature_kelvin: float
    dielectric_constant: float
    radial_spacing_angstrom: float
    radial_point_count: int
    component_count: int
    construction: str = V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION

    def __post_init__(self) -> None:
        names = _site_names(self.site_names)
        count = len(names)
        if np.iscomplexobj(self.site_multiplicity):
            raise ValueError("RISM site multiplicity must be real-valued.")
        try:
            multiplicity_values = np.asarray(self.site_multiplicity, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RISM site multiplicity must contain positive integer entries."
            ) from exc
        if (
            multiplicity_values.shape != (count,)
            or not np.all(np.isfinite(multiplicity_values))
            or np.any(multiplicity_values <= 0.0)
            or np.any(multiplicity_values != np.floor(multiplicity_values))
        ):
            raise ValueError(
                "RISM site multiplicity must contain positive integer entries."
            )
        multiplicity = np.array(multiplicity_values, dtype=int, copy=True)
        multiplicity.setflags(write=False)
        density = _immutable_real_array(
            self.bulk_number_density_angstrom3,
            name="RISM bulk site number density",
            shape=(count,),
            positive=True,
        )
        charges = _immutable_real_array(
            self.site_charges_sqrt_kT_angstrom,
            name="RISM QV site charge",
            shape=(count,),
        )
        charge_scale = max(1.0, float(np.sum(np.abs(charges * multiplicity))))
        if abs(float(np.dot(charges, multiplicity))) > 1.0e-10 * charge_scale:
            raise ValueError(
                "RISM bulk site charges must be neutral after multiplicity."
            )
        temperature = float(self.temperature_kelvin)
        dielectric = float(self.dielectric_constant)
        spacing = float(self.radial_spacing_angstrom)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("RISM temperature must be finite and positive in K.")
        if not math.isfinite(dielectric) or dielectric <= 0.0:
            raise ValueError("RISM dielectric constant must be finite and positive.")
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(
                "RISM radial spacing must be finite and positive in Angstrom."
            )
        point_count = _integer_at_least(
            self.radial_point_count,
            name="RISM radial point count",
            minimum=2,
        )
        component_count = _integer_at_least(
            self.component_count,
            name="RISM component count",
            minimum=1,
        )
        if self.construction != V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 RISM bulk construction.")
        object.__setattr__(self, "site_names", names)
        object.__setattr__(self, "site_multiplicity", multiplicity)
        object.__setattr__(self, "bulk_number_density_angstrom3", density)
        object.__setattr__(self, "site_charges_sqrt_kT_angstrom", charges)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "dielectric_constant", dielectric)
        object.__setattr__(self, "radial_spacing_angstrom", spacing)
        object.__setattr__(self, "radial_point_count", point_count)
        object.__setattr__(self, "component_count", component_count)

    @property
    def site_count(self) -> int:
        """Return the number of distinct bulk-site types."""

        return len(self.site_names)


@dataclass(frozen=True)
class Route2V0RismShortRangeDirectCorrelation:
    """Positive-radius direct correlation after an analytic Coulomb split.

    The omitted origin is intentional: a raw RISM table may contain a finite
    smeared value at zero separation while ``q_a q_b / r`` is singular.  A
    later Cartesian backend must define that value through its declared
    Coulomb/Ewald discretization, not interpolation by convenience.
    """

    metadata: Route2V0RismXvvMetadata
    radii_angstrom: np.ndarray
    values_dimensionless: np.ndarray
    construction: str = V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION

    def __post_init__(self) -> None:
        radii = _immutable_real_array(
            self.radii_angstrom,
            name="RISM short-range radial grid",
            positive=True,
        )
        if radii.ndim != 1 or np.any(np.diff(radii) <= 0.0):
            raise ValueError("RISM short-range radii must be strictly increasing.")
        values = _immutable_real_array(
            self.values_dimensionless,
            name="RISM short-range direct correlation",
            shape=(self.metadata.site_count, self.metadata.site_count, radii.size),
        )
        if not np.allclose(values, np.swapaxes(values, 0, 1), rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "RISM short-range direct correlation must be site-symmetric."
            )
        if self.construction != V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 RISM bulk construction.")
        object.__setattr__(self, "radii_angstrom", radii)
        object.__setattr__(self, "values_dimensionless", values)


@dataclass(frozen=True)
class Route2V0RismBulkDirectCorrelation:
    """A frozen radial bulk direct-correlation table from 1D-RISM.

    This object is deliberately not convertible to
    :class:`Route2V0SiteHNCAsset` yet.  Raw ``c(r)`` carries an unscreened
    site-charge tail and needs an energy-conjugate Coulomb operator before it
    can be represented on a finite Cartesian grid.
    """

    metadata: Route2V0RismXvvMetadata
    radii_angstrom: np.ndarray
    values_dimensionless: np.ndarray
    construction: str = V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION

    def __post_init__(self) -> None:
        radii = _immutable_real_array(
            self.radii_angstrom,
            name="RISM radial grid",
        )
        if radii.ndim != 1 or radii.size != self.metadata.radial_point_count:
            raise ValueError("RISM radial grid must match the XVV point count.")
        if radii[0] < 0.0 or np.any(np.diff(radii) <= 0.0):
            raise ValueError(
                "RISM radii must start at zero or above and increase strictly."
            )
        spacing_error = float(
            np.max(np.abs(np.diff(radii) - self.metadata.radial_spacing_angstrom))
        )
        spacing_scale = max(1.0, abs(self.metadata.radial_spacing_angstrom))
        if spacing_error > 1.0e-12 * spacing_scale:
            raise ValueError("RISM Cvv radial spacing must match XVV DR.")
        values = _immutable_real_array(
            self.values_dimensionless,
            name="RISM direct correlation",
            shape=(self.metadata.site_count, self.metadata.site_count, radii.size),
        )
        if not np.allclose(values, np.swapaxes(values, 0, 1), rtol=0.0, atol=1.0e-12):
            raise ValueError("RISM direct correlation must be site-symmetric.")
        if self.construction != V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION:
            raise ValueError("Unsupported Route-2 RISM bulk construction.")
        object.__setattr__(self, "radii_angstrom", radii)
        object.__setattr__(self, "values_dimensionless", values)

    def coulomb_tail_residual(self, *, minimum_radius_angstrom: float) -> float:
        """Return ``max |c(r) + q_a q_b/r|`` over a declared tail region.

        The result is measured in the native dimensionless RISM convention;
        the caller, not this parser, must pre-register an admissible tolerance
        for a specific bulk asset.
        """

        minimum = float(minimum_radius_angstrom)
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("RISM Coulomb-tail radius must be finite and positive.")
        indices = self.radii_angstrom >= minimum
        if not np.any(indices):
            raise ValueError("RISM Coulomb-tail region has no radial samples.")
        radii = self.radii_angstrom[indices]
        charges = self.metadata.site_charges_sqrt_kT_angstrom
        tail = -np.multiply.outer(charges, charges)[..., None] / radii
        return float(np.max(np.abs(self.values_dimensionless[..., indices] - tail)))

    def split_coulomb_long_range(self) -> Route2V0RismShortRangeDirectCorrelation:
        """Return ``c^sr = c + q_a q_b/r`` on strictly positive radii only."""

        indices = self.radii_angstrom > 0.0
        if not np.any(indices):
            raise ValueError("RISM direct correlation has no positive radial samples.")
        radii = self.radii_angstrom[indices]
        charges = self.metadata.site_charges_sqrt_kT_angstrom
        short_range = self.values_dimensionless[..., indices] + (
            np.multiply.outer(charges, charges)[..., None] / radii
        )
        return Route2V0RismShortRangeDirectCorrelation(
            metadata=self.metadata,
            radii_angstrom=radii,
            values_dimensionless=short_range,
        )


def parse_rism1d_xvv_metadata(text: str) -> Route2V0RismXvvMetadata:
    """Parse the metadata portion of an Amber-compatible 1D-RISM ``.xvv`` file."""

    blocks = _xvv_flag_data(text)
    pointers = _fortran_floats(
        _require_xvv_block(blocks, "POINTERS"), name="XVV POINTERS"
    )
    if len(pointers) < 3 or any(value != int(value) for value in pointers[:3]):
        raise ValueError("RISM XVV POINTERS must contain integer NR, NV, and NSP.")
    point_count, site_count, component_count = (int(value) for value in pointers[:3])
    if point_count < 2 or site_count < 1 or component_count < 1:
        raise ValueError("RISM XVV POINTERS contain invalid dimensions.")

    thermo = _fortran_floats(_require_xvv_block(blocks, "THERMO"), name="XVV THERMO")
    if len(thermo) < 5:
        raise ValueError(
            "RISM XVV THERMO must contain temperature, dielectric, and DR."
        )
    names = tuple(" ".join(_require_xvv_block(blocks, "ATOM_NAME")).split())
    if len(names) != site_count:
        raise ValueError("RISM XVV ATOM_NAME count does not match NV.")
    density = _fortran_floats(_require_xvv_block(blocks, "RHOV"), name="XVV RHOV")
    charges = _fortran_floats(_require_xvv_block(blocks, "QV"), name="XVV QV")
    multiplicity = _fortran_floats(_require_xvv_block(blocks, "MTV"), name="XVV MTV")
    if (
        len(density) != site_count
        or len(charges) != site_count
        or len(multiplicity) != site_count
    ):
        raise ValueError("RISM XVV RHOV/QV/MTV count does not match NV.")
    return Route2V0RismXvvMetadata(
        site_names=names,
        site_multiplicity=np.asarray(multiplicity),
        bulk_number_density_angstrom3=np.asarray(density),
        site_charges_sqrt_kT_angstrom=np.asarray(charges),
        temperature_kelvin=thermo[0],
        dielectric_constant=thermo[1],
        radial_spacing_angstrom=thermo[4],
        radial_point_count=point_count,
        component_count=component_count,
    )


def _cvv_pairs(
    lines: list[str],
    *,
    site_names: tuple[str, ...],
) -> list[tuple[int, int]]:
    """Read the site-pair labels from the human-readable Cvv header."""

    known = {name: index for index, name in enumerate(site_names)}
    expected_count = len(site_names) * (len(site_names) + 1) // 2
    pair_pattern = re.compile(r"([^\s:]+):([^\s:]+)")
    for line in reversed(lines):
        if not line.lstrip().startswith("#"):
            continue
        labels = pair_pattern.findall(line)
        if len(labels) != expected_count:
            continue
        try:
            pairs = [(known[left], known[right]) for left, right in labels]
        except KeyError:
            continue
        canonical = {tuple(sorted(pair)) for pair in pairs}
        if len(canonical) == expected_count:
            return pairs
    raise ValueError("RISM Cvv header lacks one complete known site-pair list.")


def parse_rism1d_direct_correlation(
    text: str,
    *,
    metadata: Route2V0RismXvvMetadata,
) -> Route2V0RismBulkDirectCorrelation:
    """Parse an Amber-compatible radial ``.cvv`` table against XVV metadata."""

    lines = text.splitlines()
    pairs = _cvv_pairs(lines, site_names=metadata.site_names)
    rows: list[list[float]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = _fortran_floats([line], name="RISM Cvv data")
        if len(values) != len(pairs) + 1:
            raise ValueError("RISM Cvv row does not match its site-pair header.")
        rows.append(values)
    if len(rows) != metadata.radial_point_count:
        raise ValueError("RISM Cvv row count does not match XVV NR.")
    table = np.asarray(rows, dtype=float)
    radii = table[:, 0]
    direct = np.empty(
        (metadata.site_count, metadata.site_count, metadata.radial_point_count),
        dtype=float,
    )
    assigned: set[tuple[int, int]] = set()
    for column, (left, right) in enumerate(pairs, start=1):
        key = tuple(sorted((left, right)))
        if key in assigned:
            raise ValueError("RISM Cvv header repeats a site pair.")
        assigned.add(key)
        direct[left, right] = table[:, column]
        direct[right, left] = table[:, column]
    return Route2V0RismBulkDirectCorrelation(
        metadata=metadata,
        radii_angstrom=radii,
        values_dimensionless=direct,
    )


def load_rism1d_bulk_direct_correlation(
    *,
    xvv_path: str | Path,
    cvv_path: str | Path,
) -> Route2V0RismBulkDirectCorrelation:
    """Load one frozen 1D-RISM bulk asset from matching XVV and Cvv files."""

    xvv = Path(xvv_path)
    cvv = Path(cvv_path)
    metadata = parse_rism1d_xvv_metadata(xvv.read_text(encoding="utf-8"))
    return parse_rism1d_direct_correlation(
        cvv.read_text(encoding="utf-8"),
        metadata=metadata,
    )


__all__ = [
    "Route2V0RismBulkDirectCorrelation",
    "Route2V0RismShortRangeDirectCorrelation",
    "Route2V0RismXvvMetadata",
    "V0_RISM_BULK_DIRECT_CORRELATION_CONSTRUCTION",
    "load_rism1d_bulk_direct_correlation",
    "parse_rism1d_direct_correlation",
    "parse_rism1d_xvv_metadata",
]
