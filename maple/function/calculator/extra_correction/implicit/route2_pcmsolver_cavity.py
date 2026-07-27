"""Versioned PCMSolver cavity input and effective-runtime contracts.

The legacy PCMSolver parser rejects an exact zero probe even though the 1.1.12
C++ core supports it.  This adapter retains the scientific input, parses one
pinned positive-probe surrogate, patches only the parsed probe scalar to zero,
and validates both the machine input and the effective C++ runtime report.
"""

from __future__ import annotations

import re

import numpy as np

from ....route2_solvents import route2_solvent_spec

PCM_TESSERA_AREA_ANGSTROM2 = 0.2
PCM_STABILITY_FALLBACK_TESSERA_AREA_ANGSTROM2 = 0.28
PCM_INTRINSIC_TESSERA_AREA_ANGSTROM2 = 0.28
PCM_STABILITY_FALLBACK_MIN_RADIUS_ANGSTROM = 0.30
PCM_INTRINSIC_MIN_RADIUS_BOHR = 100.0
PCMSOLVER_CODATA_2010_BOHR_ANGSTROM = 0.52917721092
PCM_INTRINSIC_MIN_RADIUS_ANGSTROM = (
    PCM_INTRINSIC_MIN_RADIUS_BOHR * PCMSOLVER_CODATA_2010_BOHR_ANGSTROM
)
PCM_SMD_WATER_DIELECTRIC = float(route2_solvent_spec("water").descriptors.dielectric)
PCM_SMD_WATER_OPTICAL_DIELECTRIC = float(
    route2_solvent_spec("water").descriptors.refractive_index**2
)
PCM_PARSER_SURROGATE_PROBE_RADIUS_ANGSTROM = 0.1


def _pcm_input_text(
    atom_count: int,
    radii_angstrom: np.ndarray,
    *,
    tessera_area_angstrom2: float = PCM_TESSERA_AREA_ANGSTROM2,
    minimum_added_sphere_radius_angstrom: float | None = None,
    dielectric_policy: str = "pcmsolver-water-keyword",
) -> str:
    """Return a human-readable PCMSolver input with explicit per-atom radii.

    ``MODE=ATOMS`` is intentional: it lets the C API initialize the molecular
    geometry from the host arrays while replacing every built-in radius with
    the SMD radius.  ``MODE=EXPLICIT`` instead turns the sphere list into dummy
    unit-charge atoms in the v1.1 C API.
    """

    radii = np.asarray(radii_angstrom, dtype=float)
    if radii.shape != (atom_count,) or not np.all(np.isfinite(radii)):
        raise ValueError("PCMSolver radii must be one finite value per atom.")
    area = float(tessera_area_angstrom2)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("PCMSolver tessera area must be finite and positive.")
    minimum_radius = (
        None
        if minimum_added_sphere_radius_angstrom is None
        else float(minimum_added_sphere_radius_angstrom)
    )
    if minimum_radius is not None and (
        not np.isfinite(minimum_radius) or minimum_radius <= 0.0
    ):
        raise ValueError(
            "PCMSolver minimum added-sphere radius must be finite and positive."
        )
    atom_indices = ", ".join(str(index) for index in range(1, atom_count + 1))
    radius_values = ", ".join(f"{radius:.10f}" for radius in radii)
    minimum_radius_line = (
        "" if minimum_radius is None else f"  MINRADIUS = {minimum_radius:.10f}\n"
    )
    if dielectric_policy == "pcmsolver-water-keyword":
        medium = (
            "  SOLVENT = WATER\n"
            "  NONEQUILIBRIUM = FALSE\n"
            "  MATRIXSYMM = TRUE\n"
            "  DIAGONALINTEGRATOR = COLLOCATION\n"
            "  DIAGONALSCALING = 1.07\n"
        )
    elif dielectric_policy == "explicit-smd-water-78.355-v1":
        medium = (
            "  SOLVENT = EXPLICIT\n"
            "  PROBERADIUS = 0.0000000000\n"
            "  GREEN<INSIDE>\n"
            "  {\n"
            "    TYPE = VACUUM\n"
            "    DER = DERIVATIVE\n"
            "  }\n"
            "  GREEN<OUTSIDE>\n"
            "  {\n"
            "    TYPE = UNIFORMDIELECTRIC\n"
            f"    EPS = {PCM_SMD_WATER_DIELECTRIC:.10f}\n"
            f"    EPSDYN = {PCM_SMD_WATER_OPTICAL_DIELECTRIC:.10f}\n"
            "    DER = DERIVATIVE\n"
            "  }\n"
            "  NONEQUILIBRIUM = FALSE\n"
            "  MATRIXSYMM = TRUE\n"
            "  DIAGONALINTEGRATOR = COLLOCATION\n"
            "  DIAGONALSCALING = 1.07\n"
        )
    else:
        raise ValueError(
            f"Unsupported PCMSolver dielectric policy: {dielectric_policy}."
        )
    return (
        "UNITS = ANGSTROM\n"
        "CODATA = 2010\n"
        "CAVITY\n"
        "{\n"
        "  TYPE = GEPOL\n"
        f"  AREA = {area:.10f}\n"
        "  SCALING = FALSE\n"
        f"{minimum_radius_line}"
        "  MODE = ATOMS\n"
        f"  ATOMS = [{atom_indices}]\n"
        f"  RADII = [{radius_values}]\n"
        "}\n"
        "MEDIUM\n"
        "{\n"
        "  SOLVERTYPE = IEFPCM\n"
        f"{medium}"
        "}\n"
    )


def _machine_keyword_values(
    text: str,
    kind: str,
    name: str,
) -> list[str]:
    """Read explicitly-set scalar values from PCMSolver's pinned flat format."""

    lines = text.splitlines()
    values: list[str] = []
    for index, line in enumerate(lines):
        fields = line.strip().split()
        if (
            len(fields) != 4
            or fields[0] != kind
            or fields[1] != name
            or fields[3] != "True"
        ):
            continue
        try:
            count = int(fields[2])
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed PCMSolver machine keyword header: {line!r}."
            ) from exc
        stop = index + 1 + count
        if stop > len(lines):
            raise RuntimeError(f"PCMSolver machine keyword {name} is truncated.")
        values.extend(value.strip() for value in lines[index + 1 : stop])
    return values


def _unique_machine_scalar(text: str, kind: str, name: str) -> str:
    values = _machine_keyword_values(text, kind, name)
    if len(values) != 1:
        raise RuntimeError(
            f"Expected exactly one explicitly-set PCMSolver {name}; "
            f"found {len(values)}."
        )
    return values[0]


def _validate_intrinsic_pcm_machine_input(text: str) -> dict[str, object]:
    """Fail closed unless the parsed input is the locked intrinsic-cavity model."""

    solvent = _unique_machine_scalar(text, "STR", "SOLVENT")
    probe_radius = float(_unique_machine_scalar(text, "DBL", "PROBERADIUS"))
    minimum_radius = float(_unique_machine_scalar(text, "DBL", "MINRADIUS"))
    scaling_text = _unique_machine_scalar(text, "BOOL", "SCALING")
    mode = _unique_machine_scalar(text, "STR", "MODE")
    types = _machine_keyword_values(text, "STR", "TYPE")
    static_dielectric = float(_unique_machine_scalar(text, "DBL", "EPS"))
    dynamic_dielectric = float(_unique_machine_scalar(text, "DBL", "EPSDYN"))

    expected_types = ["GEPOL", "VACUUM", "UNIFORMDIELECTRIC"]
    if solvent != "EXPLICIT":
        raise RuntimeError("Intrinsic PCMSolver input must use SOLVENT=EXPLICIT.")
    if probe_radius != 0.0:
        raise RuntimeError(
            "Intrinsic PCMSolver input must have exactly zero probe radius."
        )
    if not np.isclose(
        minimum_radius,
        PCM_INTRINSIC_MIN_RADIUS_BOHR,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            "Intrinsic PCMSolver input must disable added spheres with "
            "MINRADIUS=100 bohr."
        )
    if scaling_text != "False" or mode != "ATOMS":
        raise RuntimeError(
            "Intrinsic PCMSolver input must use unscaled MODE=ATOMS radii."
        )
    if types != expected_types:
        raise RuntimeError(
            "Intrinsic PCMSolver input must use GEPOL with vacuum inside "
            "and a uniform dielectric outside."
        )
    if not np.isclose(
        static_dielectric,
        PCM_SMD_WATER_DIELECTRIC,
        rtol=0.0,
        atol=1.0e-12,
    ) or not np.isclose(
        dynamic_dielectric,
        PCM_SMD_WATER_OPTICAL_DIELECTRIC,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Intrinsic PCMSolver input does not contain the locked SMD water "
            "static/optical dielectrics."
        )
    return {
        "solvent": solvent,
        "probe_radius_bohr": probe_radius,
        "minimum_added_sphere_radius_bohr": minimum_radius,
        "scaling": False,
        "mode": mode,
        "cavity_type": types[0],
        "inside_green_type": types[1],
        "outside_green_type": types[2],
        "outside_static_dielectric": static_dielectric,
        "outside_dynamic_dielectric": dynamic_dielectric,
    }


def _patch_intrinsic_pcm_machine_input(
    parsed_text: str,
) -> tuple[str, dict[str, object]]:
    """Apply the sole pinned-parser compatibility patch: probe 0.1 A -> 0."""

    lines = parsed_text.splitlines(keepends=True)
    headers = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "DBL PROBERADIUS 1 True"
    ]
    if len(headers) != 1:
        raise RuntimeError(
            "Expected exactly one PROBERADIUS in parsed PCMSolver input; "
            f"found {len(headers)}."
        )
    value_index = headers[0] + 1
    if value_index >= len(lines):
        raise RuntimeError("Parsed PCMSolver PROBERADIUS value is missing.")
    try:
        parser_probe = float(lines[value_index].strip())
    except ValueError as exc:
        raise RuntimeError("Parsed PCMSolver PROBERADIUS is not numeric.") from exc
    if parser_probe <= 0.0:
        raise RuntimeError(
            "The parser-surrogate PROBERADIUS must be positive before the "
            "single zero-probe compatibility patch."
        )
    expected_parser_probe_bohr = (
        PCM_PARSER_SURROGATE_PROBE_RADIUS_ANGSTROM / PCMSOLVER_CODATA_2010_BOHR_ANGSTROM
    )
    if not np.isclose(
        parser_probe,
        expected_parser_probe_bohr,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError(
            "Parsed PCMSolver PROBERADIUS does not match the pinned 0.1 A "
            "parser surrogate; refusing an unrecognized machine patch."
        )
    newline = "\n" if lines[value_index].endswith("\n") else ""
    lines[value_index] = f"0.0{newline}"
    patched = "".join(lines)
    return patched, _validate_intrinsic_pcm_machine_input(patched)


def _intrinsic_pcm_parser_surrogate(raw_text: str) -> str:
    marker = "PROBERADIUS = 0.0000000000"
    if raw_text.count(marker) != 1:
        raise RuntimeError(
            "Intrinsic PCMSolver scientific input must contain exactly one "
            "zero PROBERADIUS."
        )
    return raw_text.replace(
        marker,
        ("PROBERADIUS = " f"{PCM_PARSER_SURROGATE_PROBE_RADIUS_ANGSTROM:.10f}"),
    )


def _validate_intrinsic_pcm_runtime_info(
    text: str,
    *,
    atom_count: int,
) -> dict[str, object]:
    """Verify the effective C++ cavity, not only the machine input text."""

    probe_matches = re.findall(
        r"Solvent probe radius\s*=\s*" r"([-+0-9.eE]+)\s*Ang",
        text,
    )
    sphere_matches = re.findall(
        r"Number of spheres\s*=\s*(\d+)\s*"
        r"\[initial\s*=\s*(\d+);\s*added\s*=\s*(\d+)\]",
        text,
    )
    outside_sections = text.split(".... Outside", maxsplit=1)
    if len(probe_matches) != 1 or len(sphere_matches) != 1:
        raise RuntimeError(
            "PCMSolver runtime report did not expose one unambiguous cavity "
            "probe/sphere summary."
        )
    if len(outside_sections) != 2:
        raise RuntimeError(
            "PCMSolver runtime report did not expose the outside medium."
        )
    outside_permittivity = re.findall(
        r"Permittivity\s*=\s*([-+0-9.eE]+)",
        outside_sections[1],
    )
    if len(outside_permittivity) != 1:
        raise RuntimeError(
            "PCMSolver runtime report did not expose one outside permittivity."
        )

    probe_angstrom = float(probe_matches[0])
    total_spheres, initial_spheres, added_spheres = (
        int(value) for value in sphere_matches[0]
    )
    dielectric = float(outside_permittivity[0])
    if probe_angstrom != 0.0:
        raise RuntimeError(
            "PCMSolver runtime did not apply the zero-probe intrinsic cavity."
        )
    if (
        added_spheres != 0
        or initial_spheres != atom_count
        or total_spheres != atom_count
    ):
        raise RuntimeError(
            "PCMSolver runtime added spheres or changed the one-sphere-per-atom "
            "intrinsic cavity."
        )
    if not np.isclose(
        dielectric,
        PCM_SMD_WATER_DIELECTRIC,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "PCMSolver runtime did not apply the locked SMD water dielectric."
        )
    return {
        "probe_radius_angstrom": probe_angstrom,
        "sphere_count": total_spheres,
        "initial_sphere_count": initial_spheres,
        "added_sphere_count": added_spheres,
        "outside_static_dielectric": dielectric,
    }
