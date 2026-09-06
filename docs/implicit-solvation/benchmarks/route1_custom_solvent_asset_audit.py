#!/usr/bin/env python3
"""Fail-closed, label-free audit for a prospective molecular 3D-RISM solvent.

The audit deliberately validates *inputs*, not solvation accuracy.  A 3D-RISM
susceptibility (``.xvv``) is not a scalar dielectric: it is tied to a molecular
site model, a thermodynamic state, and a 1D-RISM calculation.  This script
checks those ties before a future Route 1 research provider is allowed to see
an MNSol label or expose a solvent option.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]


MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_XVV_GRID_POINTS = 1_048_576
FLOAT_TOLERANCE = 1.0e-8
BOLTZMANN_KCAL_MOL_K = 0.00198720425864083
AVOGADRO_PER_ANGSTROM_CUBED_PER_MOLAR = 6.02214076e-4
# Amber stores electrostatic charges in ``e * 18.2223`` in MDL files.  This is
# an exact file-format convention, not a solvent-dependent proportionality.
AMBER_ELECTROSTATIC_CHARGE_SCALE = 18.2223
# Amber's text MDL and precomputed XVV payloads round independently.  The
# cSPCE asset distributed with AmberTools 26 differs from a direct conversion
# by about 1.5e-5 relative, so retain a narrow explicit conversion tolerance.
REDUCED_UNIT_RELATIVE_TOLERANCE = 5.0e-5
# The bundled cSPCE input rounds 55.34496... M to ``55.345`` while its XVV
# retains more digits in number-density units.  This tolerance covers that
# documented text-input quantization, not an adjustable solvent parameter.
DENSITY_CONVERSION_RELATIVE_TOLERANCE = 5.0e-5
ROUTE1_BOUNDARY = {
    "name": "Additive fixed-charge PB/GB implicit solvation",
    "formula": "E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)",
    "gas_phase_mm_energy": False,
    "hydration_label_residual": False,
    "mlip_retraining": False,
    "fixed_charge": "AM1-BCC",
}
REQUIRED_CONTAINMENT = {
    "experimental_values_loaded": False,
    "experimental_residual_fit": False,
    "endpoint_selection": False,
    "energy_calibration": False,
    "runtime_provider_enabled": False,
    "accuracy_claim": "none",
}
CONTRACT_KEYS = {
    "schema_version",
    "protocol_id",
    "benchmark_kind",
    "claim_scope",
    "route1_boundary",
    "required_physical_inputs",
    "containment",
    "literature",
}
MANIFEST_KEYS = {
    "schema_version",
    "manifest_kind",
    "solvent",
    "site_types",
    "site_coordinates_angstrom",
    "rism",
    "assets",
    "provenance",
    "containment",
}
BASE_ASSET_NAMES = frozenset({"mdl", "rism1d_input", "xvv"})
GENERATION_EVIDENCE_ASSET_NAMES = frozenset(
    {"rism1d_stdout", "generation_evidence"}
)
GENERATION_EVIDENCE_CONTAINMENT = {
    "accuracy_claim": "none",
    "endpoint_selection": False,
    "experimental_residual_fit": False,
    "experimental_solvation_labels_loaded": False,
    "runtime_provider_enabled": False,
}


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite JSON number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _as_positive(value: object, field: str) -> float:
    result = _as_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive.")
    return result


def _as_nonnegative(value: object, field: str) -> float:
    result = _as_float(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative.")
    return result


def _as_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string.")
    return value.strip()


def _digest(value: object, field: str) -> str:
    result = _as_nonempty_string(value, field).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256 digest.")
    return result


def _strict_keys(value: object, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object.")
    unexpected = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unexpected or missing:
        details: list[str] = []
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        if missing:
            details.append("missing=" + ", ".join(missing))
        raise ValueError(f"Invalid {field} keys: " + "; ".join(details))
    return value


def _read_json(path: str | Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object.")
    return value


def load_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the immutable, no-label asset-audit contract."""
    contract = _strict_keys(_read_json(path, "contract"), CONTRACT_KEYS, "contract")
    if contract["schema_version"] != 1:
        raise ValueError("Only custom-solvent asset contract schema version 1 is supported.")
    if contract["benchmark_kind"] != "route1-custom-solvent-3drism-asset-audit":
        raise ValueError("Unexpected custom-solvent asset benchmark_kind.")
    _as_nonempty_string(contract["protocol_id"], "contract.protocol_id")
    _as_nonempty_string(contract["claim_scope"], "contract.claim_scope")
    if contract["route1_boundary"] != ROUTE1_BOUNDARY:
        raise ValueError("Custom-solvent asset contract must preserve the Route 1 boundary.")
    if contract["containment"] != REQUIRED_CONTAINMENT:
        raise ValueError("Custom-solvent asset contract must remain label-free and fail closed.")
    required = contract["required_physical_inputs"]
    if not isinstance(required, dict) or set(required) != {
        "molecular_site_model",
        "thermodynamic_state",
        "susceptibility",
        "standard_state",
    }:
        raise ValueError("Custom-solvent contract physical input categories are incomplete.")
    if not isinstance(contract["literature"], dict) or not contract["literature"]:
        raise ValueError("Custom-solvent contract requires literature provenance.")
    return contract, core.sha256_bytes(core.canonical_json_bytes(contract))


def _safe_relative_path(value: object, field: str) -> Path:
    text = _as_nonempty_string(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.name != text.split("/")[-1]:
        raise ValueError(f"{field} must be a safe relative asset path.")
    return Path(path.as_posix())


def _load_asset(root: Path, value: object, *, expected_suffix: str, field: str) -> tuple[Path, bytes, str]:
    asset = _strict_keys(value, {"relative_path", "sha256"}, field)
    relative_path = _safe_relative_path(asset["relative_path"], f"{field}.relative_path")
    if relative_path.suffix.lower() != expected_suffix:
        raise ValueError(f"{field}.relative_path must end in {expected_suffix!r}.")
    digest = _digest(asset["sha256"], f"{field}.sha256")
    root_resolved = root.resolve()
    path = (root_resolved / relative_path).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{field}.relative_path escapes the manifest directory.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Missing {field} asset: {relative_path.as_posix()}")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError(f"{field} asset has an invalid size.")
    payload = path.read_bytes()
    observed = core.sha256_bytes(payload)
    if observed != digest:
        raise ValueError(f"{field} SHA-256 mismatch.")
    return relative_path, payload, observed


def _decode_ascii(payload: bytes, field: str) -> str:
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field} must be ASCII text.") from exc


def _flag_values(text: str, flag: str) -> list[str]:
    match = re.search(
        rf"^%FLAG {re.escape(flag)}\r?\n%FORMAT\([^\n]+\)\r?\n"
        rf"(?P<values>.*?)(?=^%(?:FLAG|COMMENT) |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing %FLAG {flag}.")
    values = match.group("values").split()
    if not values:
        raise ValueError(f"Empty %FLAG {flag}.")
    return values


def _float_values(text: str, flag: str, count: int) -> list[float]:
    values = _flag_values(text, flag)
    if len(values) < count:
        raise ValueError(f"%FLAG {flag} has too few values.")
    try:
        parsed = [float(value) for value in values[:count]]
    except ValueError as exc:
        raise ValueError(f"%FLAG {flag} contains a non-numeric value.") from exc
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"%FLAG {flag} contains a non-finite value.")
    return parsed


def _int_values(text: str, flag: str, count: int) -> list[int]:
    values = _flag_values(text, flag)
    if len(values) < count:
        raise ValueError(f"%FLAG {flag} has too few values.")
    result: list[int] = []
    for value in values[:count]:
        try:
            integer = int(value)
        except ValueError as exc:
            raise ValueError(f"%FLAG {flag} contains a non-integer value.") from exc
        if str(integer) != value.strip() and not re.fullmatch(r"[+-]?0*\d+", value):
            raise ValueError(f"%FLAG {flag} contains a non-integer value.")
        result.append(integer)
    return result


def _coordinates(values: Iterable[float], count: int, field: str) -> list[tuple[float, float, float]]:
    sequence = list(values)
    if len(sequence) < 3 * count:
        raise ValueError(f"{field} has too few Cartesian coordinates.")
    return [
        (sequence[3 * index], sequence[3 * index + 1], sequence[3 * index + 2])
        for index in range(count)
    ]


def _pair_distances(coordinates: list[tuple[float, float, float]]) -> list[float]:
    """Return the indexed distance matrix in expanded-site order.

    Expanded-site order is part of the audited model contract.  Deliberately
    do not sort these distances: sorting would erase site identity and let an
    H/O/CH3 coordinate permutation pass as the same molecular model.
    """
    return [
        math.dist(left, right)
        for index, left in enumerate(coordinates)
        for right in coordinates[index + 1 :]
    ]


def _oriented_volumes(
    coordinates: list[tuple[float, float, float]],
) -> list[float]:
    """Return indexed scalar triple products for every four-site tuple.

    Pair distances fix a labelled geometry only up to reflection.  These
    oriented volumes preserve translation and proper-rotation invariance while
    rejecting a reflected non-planar site model.  Planar and three-site models
    have no chirality to distinguish, so their list is empty or all zero.
    """

    volumes: list[float] = []
    for first in range(len(coordinates)):
        ax, ay, az = coordinates[first]
        for second in range(first + 1, len(coordinates)):
            bx, by, bz = coordinates[second]
            ab = (bx - ax, by - ay, bz - az)
            for third in range(second + 1, len(coordinates)):
                cx, cy, cz = coordinates[third]
                ac = (cx - ax, cy - ay, cz - az)
                for fourth in range(third + 1, len(coordinates)):
                    dx, dy, dz = coordinates[fourth]
                    ad = (dx - ax, dy - ay, dz - az)
                    cross = (
                        ac[1] * ad[2] - ac[2] * ad[1],
                        ac[2] * ad[0] - ac[0] * ad[2],
                        ac[0] * ad[1] - ac[1] * ad[0],
                    )
                    volumes.append(
                        ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]
                    )
    return volumes


def _match_distances(
    left: list[tuple[float, float, float]],
    right: list[tuple[float, float, float]],
    field: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{field} site-coordinate counts disagree.")
    left_distances = _pair_distances(left)
    right_distances = _pair_distances(right)
    left_volumes = _oriented_volumes(left)
    right_volumes = _oriented_volumes(right)
    if (
        len(left_distances) != len(right_distances)
        or any(
            abs(first - second) > FLOAT_TOLERANCE
            for first, second in zip(left_distances, right_distances, strict=True)
        )
        or len(left_volumes) != len(right_volumes)
        or any(
            abs(first - second) > FLOAT_TOLERANCE
            for first, second in zip(left_volumes, right_volumes, strict=True)
        )
    ):
        raise ValueError(f"{field} site geometry disagrees.")


def _amber_charge_match(reference_e: list[float], observed: list[float], field: str) -> None:
    """Require the exact Amber ``e * 18.2223`` MDL charge convention."""
    expected = [charge * AMBER_ELECTROSTATIC_CHARGE_SCALE for charge in reference_e]
    if len(observed) != len(expected) or any(
        abs(value - target) > FLOAT_TOLERANCE * max(1.0, abs(value), abs(target))
        for value, target in zip(observed, expected, strict=True)
    ):
        raise ValueError(
            f"{field} charges disagree with the fixed Amber e*"
            f"{AMBER_ELECTROSTATIC_CHARGE_SCALE:g} convention."
        )


def _reduced_unit_match(
    source_values: Iterable[float],
    reduced_values: Iterable[float],
    *,
    temperature_kelvin: float,
    exponent: float,
    field: str,
) -> None:
    """Check Amber MDL -> XVV unit conversion through the declared temperature.

    Amber MDL charge and Lennard--Jones epsilon fields use energy-based units,
    while XVV stores QV in ``sqrt(kT A)`` and EPSV in ``kT``.  A mere pattern
    check would let a temperature-inconsistent susceptibility through.
    """
    thermal_energy = BOLTZMANN_KCAL_MOL_K * temperature_kelvin
    source = list(source_values)
    reduced = list(reduced_values)
    expected = [value / thermal_energy**exponent for value in source]
    if len(reduced) != len(expected) or any(
        abs(value - reference)
        > REDUCED_UNIT_RELATIVE_TOLERANCE * max(1.0e-12, abs(value), abs(reference))
        for value, reference in zip(reduced, expected, strict=True)
    ):
        raise ValueError(f"{field} disagrees.")


def _density_conversion_match(value: float, reference: float, field: str) -> None:
    if abs(value - reference) > DENSITY_CONVERSION_RELATIVE_TOLERANCE * max(
        1.0e-12, abs(value), abs(reference)
    ):
        raise ValueError(f"{field} disagrees.")


def _parse_mdl(text: str) -> dict[str, Any]:
    pointers = _int_values(text, "POINTERS", 2)
    actual_site_count, site_type_count = pointers
    names = _flag_values(text, "ATMNAME")[:site_type_count]
    multiplicities = _int_values(text, "MULTI", site_type_count)
    masses = _float_values(text, "MASS", site_type_count)
    charges = _float_values(text, "CHG", site_type_count)
    epsilons = _float_values(text, "LJEPSILON", site_type_count)
    sizes = _float_values(text, "LJSIGMA", site_type_count)
    if actual_site_count <= 0 or site_type_count <= 0 or sum(multiplicities) != actual_site_count:
        raise ValueError("MDL POINTERS/MULTI site counts disagree.")
    if any(multiplicity <= 0 for multiplicity in multiplicities):
        raise ValueError("MDL MULTI values must be positive.")
    if any(mass < 0.0 for mass in masses) or all(mass == 0.0 for mass in masses):
        raise ValueError("MDL masses must be non-negative with at least one massive site.")
    if any(epsilon < 0.0 or size < 0.0 for epsilon, size in zip(epsilons, sizes, strict=True)):
        raise ValueError("MDL Lennard-Jones parameters must be non-negative.")
    if any(epsilon > 0.0 and size == 0.0 for epsilon, size in zip(epsilons, sizes, strict=True)):
        raise ValueError("An MDL site with nonzero Lennard-Jones epsilon needs a positive size.")
    coordinates = _coordinates(
        _float_values(text, "COORD", 3 * actual_site_count), actual_site_count, "MDL COORD"
    )
    weighted_charge = sum(
        charge * multiplicity for charge, multiplicity in zip(charges, multiplicities, strict=True)
    )
    charge_scale = sum(
        abs(charge * multiplicity)
        for charge, multiplicity in zip(charges, multiplicities, strict=True)
    )
    if abs(weighted_charge) > FLOAT_TOLERANCE * max(1.0, charge_scale):
        raise ValueError("MDL solvent model must be electrically neutral.")
    return {
        "actual_site_count": actual_site_count,
        "site_type_count": site_type_count,
        "names": names,
        "multiplicities": multiplicities,
        "masses": masses,
        "charges": charges,
        "epsilons": epsilons,
        "sizes": sizes,
        "coordinates": coordinates,
    }


def _parse_xvv(text: str) -> dict[str, Any]:
    grid_points, site_type_count, species_count = _int_values(text, "POINTERS", 3)
    if grid_points <= 0 or grid_points > MAX_XVV_GRID_POINTS or site_type_count <= 0 or species_count <= 0:
        raise ValueError("XVV POINTERS are outside the audited range.")
    names = _flag_values(text, "ATOM_NAME")[:site_type_count]
    multiplicities = _int_values(text, "MTV", site_type_count)
    per_species_types = _int_values(text, "NVSP", species_count)
    if any(value <= 0 for value in multiplicities) or any(value <= 0 for value in per_species_types):
        raise ValueError("XVV multiplicities and species site counts must be positive.")
    if sum(per_species_types) != site_type_count:
        raise ValueError("XVV NVSP does not account for every site type.")
    thermo = _float_values(text, "THERMO", 6)
    masses = _float_values(text, "MASS", site_type_count)
    species_densities = _float_values(text, "RHOSP", species_count)
    site_densities = _float_values(text, "RHOV", site_type_count)
    charges = _float_values(text, "QV", site_type_count)
    epsilons = _float_values(text, "EPSV", site_type_count)
    sizes = _float_values(text, "RMIN2V", site_type_count)
    actual_site_count = sum(multiplicities)
    coordinates = _coordinates(
        _float_values(text, "COORD", 3 * actual_site_count), actual_site_count, "XVV COORD"
    )
    if any(mass < 0.0 for mass in masses) or all(mass == 0.0 for mass in masses):
        raise ValueError("XVV masses must be non-negative with at least one massive site.")
    if any(epsilon < 0.0 or size < 0.0 for epsilon, size in zip(epsilons, sizes, strict=True)):
        raise ValueError("XVV Lennard-Jones parameters must be non-negative.")
    if any(epsilon > 0.0 and size == 0.0 for epsilon, size in zip(epsilons, sizes, strict=True)):
        raise ValueError("An XVV site with nonzero Lennard-Jones epsilon needs a positive size.")
    if any(density <= 0.0 for density in species_densities) or any(density <= 0.0 for density in site_densities):
        raise ValueError("XVV densities must be positive.")
    if thermo[0] <= 0.0 or thermo[1] <= 1.0:
        raise ValueError("XVV temperature and dielectric constant are invalid.")
    correlation_values = _flag_values(text, "XVV")
    expected_correlation_count = (
        grid_points * site_type_count * site_type_count
    )
    if len(correlation_values) != expected_correlation_count:
        raise ValueError(
            "XVV correlation array length disagrees with POINTERS: "
            f"expected {expected_correlation_count}, observed "
            f"{len(correlation_values)}."
        )
    for value in correlation_values:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(
                "XVV correlation array contains a non-numeric value."
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError("XVV correlation array contains a non-finite value.")
    return {
        "grid_points": grid_points,
        "site_type_count": site_type_count,
        "species_count": species_count,
        "actual_site_count": actual_site_count,
        "names": names,
        "multiplicities": multiplicities,
        "per_species_types": per_species_types,
        "temperature_kelvin": thermo[0],
        "dielectric_constant": thermo[1],
        "grid_spacing_angstrom": thermo[4],
        "masses": masses,
        "species_densities": species_densities,
        "site_densities": site_densities,
        "charges": charges,
        "epsilons": epsilons,
        "sizes": sizes,
        "coordinates": coordinates,
    }


def _density_to_angstrom_minus3(
    value: float,
    units: str,
    *,
    molecular_mass_amu: float,
) -> float:
    """Convert the documented Amber ``rism1d`` density units to number density."""
    normalized = units.lower().replace("å", "a")
    if normalized in {"1/a^3", "1/a3"}:
        return value
    if normalized == "m":
        return value * AVOGADRO_PER_ANGSTROM_CUBED_PER_MOLAR
    if normalized == "mm":
        return value * 1.0e-3 * AVOGADRO_PER_ANGSTROM_CUBED_PER_MOLAR
    if normalized == "g/cm3":
        return (
            value
            * 1000.0
            * AVOGADRO_PER_ANGSTROM_CUBED_PER_MOLAR
            / molecular_mass_amu
        )
    if normalized == "kg/m3":
        return (
            value
            * AVOGADRO_PER_ANGSTROM_CUBED_PER_MOLAR
            / molecular_mass_amu
        )
    raise ValueError("1D-RISM input density units are not supported by the Amber contract.")


def _parse_rism1d_input(
    text: str,
    *,
    mdl_relative_path: Path,
    molecular_mass_amu: float,
) -> dict[str, Any]:
    def required_pattern(pattern: str, name: str) -> str:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"1D-RISM input does not declare {name}.")
        return match.group(1)

    theory = required_pattern(r"\bTHEORY\s*=\s*['\"]?([A-Za-z0-9_+-]+)", "theory").lower()
    closure = required_pattern(r"\bCLOSUR(?:E)?\s*=\s*['\"]?([A-Za-z0-9_+()-]+)", "closure").lower()
    temperature = float(required_pattern(r"\bTEMPER(?:ATURE)?\s*=\s*([-+0-9.eE]+)", "temperature"))
    dielectric = float(required_pattern(r"\bDIEPS\s*=\s*([-+0-9.eE]+)", "dielectric"))
    grid_spacing = float(required_pattern(r"\bDR\s*=\s*([-+0-9.eE]+)", "grid spacing"))
    grid_points = int(required_pattern(r"\bNR\s*=\s*([0-9]+)", "grid points"))
    density_value = float(required_pattern(r"\bDENSITY\s*=\s*([-+0-9.eE]+)", "species density"))
    density_units = required_pattern(r"\bUNITS\s*=\s*['\"]?([A-Za-z0-9/^_-]+)", "species-density units").lower()
    if not re.search(r"\bMODEL(?:S)?\s*=", text, flags=re.IGNORECASE):
        raise ValueError("1D-RISM input does not declare a molecular model.")
    model_stem = re.escape(mdl_relative_path.stem)
    if not re.search(model_stem, text, flags=re.IGNORECASE):
        raise ValueError("1D-RISM input does not reference the audited MDL model.")
    if theory not in {"drism", "xrism"}:
        raise ValueError("1D-RISM input theory must be DRISM or XRISM.")
    species_density = _density_to_angstrom_minus3(
        density_value,
        density_units,
        molecular_mass_amu=molecular_mass_amu,
    )
    if not (
        temperature > 0.0
        and dielectric > 1.0
        and grid_spacing > 0.0
        and grid_points > 0
        and species_density > 0.0
    ):
        raise ValueError("1D-RISM input thermodynamic state is invalid.")
    return {
        "theory": theory,
        "closure": closure,
        "temperature_kelvin": temperature,
        "dielectric_constant": dielectric,
        "grid_spacing_angstrom": grid_spacing,
        "grid_points": grid_points,
        "species_density_angstrom_minus3": species_density,
        "density_input_value": density_value,
        "density_input_units": density_units,
    }


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _strict_keys(_read_json(path, "manifest"), MANIFEST_KEYS, "manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("Only custom-solvent manifest schema version 1 is supported.")
    if manifest["manifest_kind"] != "route1-custom-solvent-3drism-asset":
        raise ValueError("Unexpected custom-solvent manifest_kind.")
    if manifest["containment"] != REQUIRED_CONTAINMENT:
        raise ValueError("Custom-solvent manifest must remain label-free and fail closed.")

    solvent = _strict_keys(
        manifest["solvent"],
        {
            "canonical_name",
            "temperature_kelvin",
            "pressure_bar",
            "dielectric_constant",
            "species_density_angstrom_minus3",
            "standard_state",
        },
        "manifest.solvent",
    )
    _as_nonempty_string(solvent["canonical_name"], "manifest.solvent.canonical_name")
    _as_positive(solvent["temperature_kelvin"], "manifest.solvent.temperature_kelvin")
    _as_positive(solvent["pressure_bar"], "manifest.solvent.pressure_bar")
    if _as_float(solvent["dielectric_constant"], "manifest.solvent.dielectric_constant") <= 1.0:
        raise ValueError("manifest.solvent.dielectric_constant must exceed one.")
    _as_positive(
        solvent["species_density_angstrom_minus3"],
        "manifest.solvent.species_density_angstrom_minus3",
    )
    standard_state = _strict_keys(
        solvent["standard_state"],
        {"provider_output", "benchmark_target", "conversion_status"},
        "manifest.solvent.standard_state",
    )
    if standard_state != {
        "provider_output": "excess_chemical_potential",
        "benchmark_target": "1M_ideal_gas_to_1M_ideal_solution",
        "conversion_status": "not_yet_applied",
    }:
        raise ValueError("Custom-solvent asset must preserve the explicit pre-score standard-state boundary.")

    site_types = manifest["site_types"]
    if not isinstance(site_types, list) or not site_types:
        raise ValueError("manifest.site_types must be a non-empty list.")
    names: set[str] = set()
    declared_sites: list[dict[str, Any]] = []
    for index, site in enumerate(site_types):
        site_object = _strict_keys(
            site,
            {
                "name",
                "multiplicity",
                "mass_amu",
                "charge_e",
                "lj_epsilon_kcal_mol",
                "lj_size_angstrom",
            },
            f"manifest.site_types[{index}]",
        )
        name = _as_nonempty_string(site_object["name"], f"manifest.site_types[{index}].name")
        if name in names:
            raise ValueError("manifest.site_types names must be unique.")
        names.add(name)
        multiplicity = site_object["multiplicity"]
        if type(multiplicity) is not int or multiplicity <= 0:
            raise ValueError(f"manifest.site_types[{index}].multiplicity must be a positive integer.")
        declared_sites.append(
            {
                "name": name,
                "multiplicity": multiplicity,
                "mass_amu": _as_nonnegative(site_object["mass_amu"], f"manifest.site_types[{index}].mass_amu"),
                "charge_e": _as_float(site_object["charge_e"], f"manifest.site_types[{index}].charge_e"),
                "lj_epsilon_kcal_mol": _as_nonnegative(
                    site_object["lj_epsilon_kcal_mol"],
                    f"manifest.site_types[{index}].lj_epsilon_kcal_mol",
                ),
                # The schema keeps its original compact key, but its physical
                # meaning is Amber MDL Rmin/2 in angstrom, not conventional
                # 12-6 sigma.  Source sigma must be converted by 2**(-5/6).
                "lj_size_angstrom": _as_nonnegative(
                    site_object["lj_size_angstrom"],
                    f"manifest.site_types[{index}].lj_size_angstrom",
                ),
            }
        )
        if (
            declared_sites[-1]["lj_epsilon_kcal_mol"] > 0.0
            and declared_sites[-1]["lj_size_angstrom"] == 0.0
        ):
            raise ValueError(
                f"manifest.site_types[{index}] has nonzero Lennard-Jones epsilon but zero size."
            )
    if all(site["mass_amu"] == 0.0 for site in declared_sites):
        raise ValueError("manifest.site_types must contain at least one massive site.")
    if abs(sum(site["charge_e"] * site["multiplicity"] for site in declared_sites)) > FLOAT_TOLERANCE:
        raise ValueError("manifest.site_types must define an electrically neutral solvent molecule.")

    coordinates = manifest["site_coordinates_angstrom"]
    expected_coordinates = sum(site["multiplicity"] for site in declared_sites)
    if not isinstance(coordinates, list) or len(coordinates) != expected_coordinates:
        raise ValueError("manifest.site_coordinates_angstrom count must equal site multiplicities.")
    declared_coordinates: list[tuple[float, float, float]] = []
    for index, coordinate in enumerate(coordinates):
        if not isinstance(coordinate, list) or len(coordinate) != 3:
            raise ValueError(f"manifest.site_coordinates_angstrom[{index}] must be a three-vector.")
        declared_coordinates.append(
            (
                _as_float(coordinate[0], f"manifest.site_coordinates_angstrom[{index}][0]"),
                _as_float(coordinate[1], f"manifest.site_coordinates_angstrom[{index}][1]"),
                _as_float(coordinate[2], f"manifest.site_coordinates_angstrom[{index}][2]"),
            )
        )

    rism = _strict_keys(
        manifest["rism"],
        {"theory", "closure", "grid_spacing_angstrom", "grid_points"},
        "manifest.rism",
    )
    theory = _as_nonempty_string(rism["theory"], "manifest.rism.theory").lower()
    closure = _as_nonempty_string(rism["closure"], "manifest.rism.closure").lower()
    if theory not in {"drism", "xrism"}:
        raise ValueError("manifest.rism.theory must be DRISM or XRISM.")
    if not re.fullmatch(r"(?:kh|hnc|py|vm0?|pse\d+)", closure):
        raise ValueError("manifest.rism.closure is not an accepted 1D-RISM closure.")
    _as_positive(rism["grid_spacing_angstrom"], "manifest.rism.grid_spacing_angstrom")
    if type(rism["grid_points"]) is not int or not 0 < rism["grid_points"] <= MAX_XVV_GRID_POINTS:
        raise ValueError("manifest.rism.grid_points is outside the audited range.")

    assets = manifest["assets"]
    if not isinstance(assets, dict):
        raise TypeError("manifest.assets must be a JSON object.")
    asset_names = frozenset(assets)
    allowed_asset_sets = {
        BASE_ASSET_NAMES,
        BASE_ASSET_NAMES | GENERATION_EVIDENCE_ASSET_NAMES,
    }
    if asset_names not in allowed_asset_sets:
        raise ValueError(
            "manifest.assets must contain the three physical inputs and either "
            "both or neither of rism1d_stdout/generation_evidence."
        )
    provenance = _strict_keys(
        manifest["provenance"],
        {"molecular_model_reference", "susceptibility_generation"},
        "manifest.provenance",
    )
    _as_nonempty_string(provenance["molecular_model_reference"], "manifest.provenance.molecular_model_reference")
    generation = _strict_keys(
        provenance["susceptibility_generation"],
        {"generator", "executable_version", "source_inputs_immutable"},
        "manifest.provenance.susceptibility_generation",
    )
    if _as_nonempty_string(generation["generator"], "manifest.provenance.susceptibility_generation.generator") != "rism1d":
        raise ValueError("Custom-solvent susceptibility must be generated by rism1d.")
    _as_nonempty_string(generation["executable_version"], "manifest.provenance.susceptibility_generation.executable_version")
    if generation["source_inputs_immutable"] is not True:
        raise ValueError("Custom-solvent susceptibility source inputs must be immutable.")

    return {
        "raw": manifest,
        "solvent": solvent,
        "site_types": declared_sites,
        "coordinates": declared_coordinates,
        "rism": {**rism, "theory": theory, "closure": closure},
        "assets": assets,
        "provenance": provenance,
    }


def _allclose(left: Iterable[float], right: Iterable[float], field: str) -> None:
    first = list(left)
    second = list(right)
    if len(first) != len(second) or any(
        abs(value - expected) > FLOAT_TOLERANCE * max(1.0, abs(value), abs(expected))
        for value, expected in zip(first, second, strict=True)
    ):
        raise ValueError(f"{field} disagrees.")


def _generation_convergence_record(
    value: object,
    field: str,
) -> dict[str, float | int]:
    record = _strict_keys(
        value,
        {"iterations", "final_residual", "requested_tolerance"},
        field,
    )
    iterations = record["iterations"]
    if type(iterations) is not int or iterations <= 0:
        raise ValueError(f"{field}.iterations must be a positive integer.")
    residual = _as_nonnegative(record["final_residual"], f"{field}.final_residual")
    tolerance = _as_positive(
        record["requested_tolerance"],
        f"{field}.requested_tolerance",
    )
    if residual > tolerance:
        raise ValueError(f"{field} did not reach its requested tolerance.")
    return {
        "iterations": iterations,
        "final_residual": residual,
        "requested_tolerance": tolerance,
    }


def _last_rism_step(text: str, field: str) -> tuple[int, float]:
    matches = re.findall(
        r"^step=\s*(\d+)\s+Res=\s*([-+0-9.eE]+)(?:\s|$)",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        raise ValueError(f"rism1d stdout has no {field} convergence steps.")
    iteration_text, residual_text = matches[-1]
    residual = float(residual_text)
    if not math.isfinite(residual):
        raise ValueError(f"rism1d stdout has a non-finite {field} residual.")
    return int(iteration_text), residual


def _validate_generation_evidence(
    payload: bytes,
    stdout_payload: bytes,
    assets: dict[str, tuple[Path, bytes, str]],
) -> dict[str, Any]:
    """Validate a sealed local generation record and its cross-hashed stdout.

    This binds the committed files and convergence log.  It deliberately does
    not claim executable replay or numerical regeneration parity; a future
    reproducibility gate must rerun the pinned executable for that stronger
    statement.
    """

    try:
        evidence_value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("generation evidence must be valid UTF-8 JSON.") from exc
    evidence = _strict_keys(
        evidence_value,
        {
            "schema_version",
            "artifact_type",
            "assets",
            "claim_scope",
            "containment",
            "content_sha256",
            "convergence",
            "generator",
            "recorded_date",
        },
        "generation evidence",
    )
    if (
        evidence["schema_version"] != 1
        or evidence["artifact_type"] != "route1-rism1d-generation-evidence-v1"
    ):
        raise ValueError("Unexpected rism1d generation-evidence schema.")
    _as_nonempty_string(evidence["claim_scope"], "generation evidence.claim_scope")
    if evidence["containment"] != GENERATION_EVIDENCE_CONTAINMENT:
        raise ValueError("Generation evidence must remain label-free and fail closed.")
    content_digest = _digest(
        evidence["content_sha256"],
        "generation evidence.content_sha256",
    )
    if content_digest != core.artifact_content_sha256(evidence):
        raise ValueError("Generation evidence content SHA-256 mismatch.")
    recorded_date = _as_nonempty_string(
        evidence["recorded_date"],
        "generation evidence.recorded_date",
    )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", recorded_date):
        raise ValueError("Generation evidence recorded_date must be ISO YYYY-MM-DD.")

    generator = _strict_keys(
        evidence["generator"],
        {
            "executable_sha256",
            "name",
            "original_invocation",
            "reproduction_invocation",
            "reproduction_note",
        },
        "generation evidence.generator",
    )
    executable_digest = _digest(
        generator["executable_sha256"],
        "generation evidence.generator.executable_sha256",
    )
    for field in (
        "name",
        "original_invocation",
        "reproduction_invocation",
        "reproduction_note",
    ):
        _as_nonempty_string(generator[field], f"generation evidence.generator.{field}")

    evidence_assets = _strict_keys(
        evidence["assets"],
        {"mdl", "input", "xvv", "stdout"},
        "generation evidence.assets",
    )
    asset_mapping = {
        "mdl": "mdl",
        "input": "rism1d_input",
        "xvv": "xvv",
        "stdout": "rism1d_stdout",
    }
    for evidence_name, manifest_name in asset_mapping.items():
        entry = _strict_keys(
            evidence_assets[evidence_name],
            {"name", "sha256"},
            f"generation evidence.assets.{evidence_name}",
        )
        relative_path, _, observed_digest = assets[manifest_name]
        if (
            _as_nonempty_string(
                entry["name"],
                f"generation evidence.assets.{evidence_name}.name",
            )
            != relative_path.name
            or _digest(
                entry["sha256"],
                f"generation evidence.assets.{evidence_name}.sha256",
            )
            != observed_digest
        ):
            raise ValueError(
                f"Generation evidence {evidence_name} asset binding disagrees."
            )

    convergence = _strict_keys(
        evidence["convergence"],
        {"exit_status", "primary_rism", "temperature_derivative_rism"},
        "generation evidence.convergence",
    )
    if convergence["exit_status"] != 0:
        raise ValueError("Generation evidence requires a successful rism1d exit.")
    primary = _generation_convergence_record(
        convergence["primary_rism"],
        "generation evidence.convergence.primary_rism",
    )
    derivative = _generation_convergence_record(
        convergence["temperature_derivative_rism"],
        "generation evidence.convergence.temperature_derivative_rism",
    )

    stdout_text = _decode_ascii(stdout_payload, "rism1d stdout")
    derivative_marker = "relaxing RISM DT:"
    if stdout_text.count(derivative_marker) != 1:
        raise ValueError(
            "rism1d stdout must contain exactly one temperature-derivative marker."
        )
    primary_text, derivative_text = stdout_text.split(derivative_marker)
    observed_primary = _last_rism_step(primary_text, "primary RISM")
    observed_derivative = _last_rism_step(
        derivative_text,
        "temperature-derivative RISM",
    )
    for observed, declared, field in (
        (observed_primary, primary, "primary RISM"),
        (observed_derivative, derivative, "temperature-derivative RISM"),
    ):
        if observed[0] != declared["iterations"] or not math.isclose(
            observed[1],
            float(declared["final_residual"]),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"Generation evidence and rism1d stdout {field} convergence disagree."
            )

    return {
        "bound": True,
        "evidence_sha256": assets["generation_evidence"][2],
        "stdout_sha256": assets["rism1d_stdout"][2],
        "executable_sha256": executable_digest,
        "primary_rism": primary,
        "temperature_derivative_rism": derivative,
        "regeneration_parity_validated": False,
    }


def audit_manifest(
    contract: dict[str, Any],
    contract_fingerprint: str,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Return a sealed, label-free asset-integrity artifact for one solvent."""
    loaded = _load_manifest(manifest_path)
    root = Path(manifest_path).resolve().parent
    assets: dict[str, tuple[Path, bytes, str]] = {
        "mdl": _load_asset(root, loaded["assets"]["mdl"], expected_suffix=".mdl", field="manifest.assets.mdl"),
        "rism1d_input": _load_asset(
            root,
            loaded["assets"]["rism1d_input"],
            expected_suffix=".inp",
            field="manifest.assets.rism1d_input",
        ),
        "xvv": _load_asset(root, loaded["assets"]["xvv"], expected_suffix=".xvv", field="manifest.assets.xvv"),
    }
    if GENERATION_EVIDENCE_ASSET_NAMES <= set(loaded["assets"]):
        assets["rism1d_stdout"] = _load_asset(
            root,
            loaded["assets"]["rism1d_stdout"],
            expected_suffix=".out",
            field="manifest.assets.rism1d_stdout",
        )
        assets["generation_evidence"] = _load_asset(
            root,
            loaded["assets"]["generation_evidence"],
            expected_suffix=".json",
            field="manifest.assets.generation_evidence",
        )
    mdl = _parse_mdl(_decode_ascii(assets["mdl"][1], "MDL asset"))
    input_parameters = _parse_rism1d_input(
        _decode_ascii(assets["rism1d_input"][1], "1D-RISM input asset"),
        mdl_relative_path=assets["mdl"][0],
        molecular_mass_amu=sum(
            mass * multiplicity
            for mass, multiplicity in zip(mdl["masses"], mdl["multiplicities"], strict=True)
        ),
    )
    xvv = _parse_xvv(_decode_ascii(assets["xvv"][1], "XVV asset"))

    declared_sites = loaded["site_types"]
    declared_names = [site["name"] for site in declared_sites]
    declared_multiplicities = [site["multiplicity"] for site in declared_sites]
    declared_masses = [site["mass_amu"] for site in declared_sites]
    declared_charges = [site["charge_e"] for site in declared_sites]
    declared_epsilons = [site["lj_epsilon_kcal_mol"] for site in declared_sites]
    declared_sizes = [site["lj_size_angstrom"] for site in declared_sites]
    if mdl["names"] != declared_names or xvv["names"] != declared_names:
        raise ValueError("Manifest, MDL, and XVV site names must agree exactly.")
    if mdl["multiplicities"] != declared_multiplicities or xvv["multiplicities"] != declared_multiplicities:
        raise ValueError("Manifest, MDL, and XVV site multiplicities must agree exactly.")
    _allclose(mdl["masses"], declared_masses, "Manifest and MDL masses")
    _allclose(xvv["masses"], declared_masses, "Manifest and XVV masses")
    _allclose(mdl["epsilons"], declared_epsilons, "Manifest and MDL Lennard-Jones epsilons")
    _allclose(mdl["sizes"], declared_sizes, "Manifest and MDL Lennard-Jones sizes")
    _allclose(xvv["sizes"], declared_sizes, "Manifest and XVV Lennard-Jones sizes")
    _amber_charge_match(declared_charges, mdl["charges"], "Manifest and MDL")
    _reduced_unit_match(
        mdl["charges"],
        xvv["charges"],
        temperature_kelvin=xvv["temperature_kelvin"],
        exponent=0.5,
        field="MDL and XVV charges",
    )
    _reduced_unit_match(
        mdl["epsilons"],
        xvv["epsilons"],
        temperature_kelvin=xvv["temperature_kelvin"],
        exponent=1.0,
        field="MDL and XVV Lennard-Jones epsilons",
    )
    _match_distances(loaded["coordinates"], mdl["coordinates"], "Manifest and MDL")
    _match_distances(mdl["coordinates"], xvv["coordinates"], "MDL and XVV")
    if xvv["site_type_count"] != len(declared_sites) or xvv["species_count"] != 1:
        raise ValueError("Custom-solvent XVV must represent exactly one declared solvent species.")
    if xvv["grid_points"] != loaded["rism"]["grid_points"]:
        raise ValueError("Manifest and XVV grid-point counts disagree.")
    if input_parameters["grid_points"] != loaded["rism"]["grid_points"]:
        raise ValueError("Manifest and 1D-RISM input grid-point counts disagree.")
    _allclose(
        [
            loaded["rism"]["grid_spacing_angstrom"],
            input_parameters["grid_spacing_angstrom"],
        ],
        [xvv["grid_spacing_angstrom"], xvv["grid_spacing_angstrom"]],
        "Manifest, 1D-RISM input, and XVV grid spacing",
    )
    if input_parameters["theory"] != loaded["rism"]["theory"]:
        raise ValueError("Manifest and 1D-RISM theories disagree.")
    if input_parameters["closure"] != loaded["rism"]["closure"]:
        raise ValueError("Manifest and 1D-RISM closures disagree.")
    _allclose(
        [loaded["solvent"]["temperature_kelvin"], input_parameters["temperature_kelvin"]],
        [xvv["temperature_kelvin"], xvv["temperature_kelvin"]],
        "Manifest, 1D-RISM input, and XVV temperature",
    )
    _allclose(
        [loaded["solvent"]["dielectric_constant"], input_parameters["dielectric_constant"]],
        [xvv["dielectric_constant"], xvv["dielectric_constant"]],
        "Manifest, 1D-RISM input, and XVV dielectric constant",
    )
    species_density = _as_float(
        loaded["solvent"]["species_density_angstrom_minus3"],
        "manifest.solvent.species_density_angstrom_minus3",
    )
    _allclose(xvv["species_densities"], [species_density], "Manifest and XVV species density")
    _density_conversion_match(
        input_parameters["species_density_angstrom_minus3"],
        species_density,
        "Manifest and 1D-RISM input species density",
    )
    _allclose(
        xvv["site_densities"],
        [species_density * multiplicity for multiplicity in declared_multiplicities],
        "XVV site densities and manifest multiplicities",
    )
    if "generation_evidence" in assets:
        generation_evidence = _validate_generation_evidence(
            assets["generation_evidence"][1],
            assets["rism1d_stdout"][1],
            assets,
        )
    else:
        generation_evidence = {
            "bound": False,
            "regeneration_parity_validated": False,
        }

    artifact = {
        "artifact_type": "route1-custom-solvent-3drism-asset-audit-v1",
        "contract_id": contract["protocol_id"],
        "contract_fingerprint": contract_fingerprint,
        "route1_boundary": contract["route1_boundary"],
        "containment": contract["containment"],
        "solvent": {
            "canonical_name": loaded["solvent"]["canonical_name"],
            "temperature_kelvin": loaded["solvent"]["temperature_kelvin"],
            "pressure_bar": loaded["solvent"]["pressure_bar"],
            "dielectric_constant": loaded["solvent"]["dielectric_constant"],
            "species_density_angstrom_minus3": species_density,
            "standard_state": loaded["solvent"]["standard_state"],
        },
        "molecular_site_model": {
            "site_type_count": len(declared_sites),
            "actual_site_count": len(loaded["coordinates"]),
            "site_names": declared_names,
            "multiplicities": declared_multiplicities,
            "net_charge_e": sum(
                site["charge_e"] * site["multiplicity"] for site in declared_sites
            ),
            "molecular_model_reference": loaded["provenance"]["molecular_model_reference"],
        },
        "rism": {
            "theory": loaded["rism"]["theory"],
            "closure": loaded["rism"]["closure"],
            "grid_spacing_angstrom": loaded["rism"]["grid_spacing_angstrom"],
            "grid_points": loaded["rism"]["grid_points"],
            "density_input": {
                "value": input_parameters["density_input_value"],
                "units": input_parameters["density_input_units"],
            },
            "generator": loaded["provenance"]["susceptibility_generation"]["generator"],
            "executable_version": loaded["provenance"]["susceptibility_generation"]["executable_version"],
            "generation_evidence": generation_evidence,
        },
        "assets": {
            name: {
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": len(payload),
            }
            for name, (relative, payload, digest) in assets.items()
        },
        "conclusion": {
            "status": "physical_asset_consistent_not_accuracy_validated",
            "asset_consistency_scope": (
                "metadata_and_exact_finite_xvv_payload_structure"
            ),
            "generation_evidence_bound": generation_evidence["bound"],
            "regeneration_parity_validated": False,
            "runtime_provider_enabled": False,
            "experimental_values_loaded": False,
            "accuracy_claim": "none",
            "next_gate": "numerical_3drism_provider_parity_then_prospective_per_solvent_and_external_accuracy_validation",
        },
    }
    return core.seal_artifact(artifact)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract, fingerprint = load_contract(args.contract)
    artifact = audit_manifest(contract, fingerprint, args.manifest)
    artifact["command_provenance"] = core.command_provenance(
        __file__,
        {
            "contract": str(args.contract),
            "manifest_name": args.manifest.name,
            "manifest_sha256": core.sha256_file(args.manifest),
            "output": str(args.output),
        },
        repository_root=REPOSITORY_ROOT,
    )
    core.seal_artifact(artifact)
    core.write_json_atomic(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
