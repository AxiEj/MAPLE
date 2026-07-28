"""Version-locked QM-reference boundary for openCOSMO-RS 24a results.

COSMO-RS is a sigma-profile/statistical-thermodynamics workflow.  It is not a
continuum-equation option for MAPLE's Route-2 ``#solv`` calculator, so this
module validates external ORCA/openCOSMO-RS assets without registering a
calculator provider. It is an oracle for the experimental MLIP surface bridge
in :mod:`maple.function.mlip_cosmo_rs`, not the target MLIP--continuum method.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

HARTREE_TO_KCAL_MOL = 627.5094740631
OPEN_COSMORS_24A_PARAMETERIZATION = "openCOSMO-RS 24a"

_REFERENCE_TEMPERATURE_K = 298.15
_MINIMUM_SURFACE_SEGMENT_AREA_ANGSTROM2 = 0.01
_ENERGY_UNIT_TOLERANCE_KCAL_MOL = 5.0e-6
_SAFE_ORCA_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_SOLVENT_ALIAS_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_ELEMENT_SYMBOL_RE = re.compile(r"[A-Z][a-z]?\Z")


@dataclass(frozen=True)
class _VerifiedAsset:
    path: Path
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: str | Path, *, role: str) -> "_VerifiedAsset":
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"COSMO-RS {role} must be an existing regular file.")
        return cls(
            path=resolved,
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
            size_bytes=resolved.stat().st_size,
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class OpenCOSMORS24aInputBundle:
    """Three immutable QC assets required by the openCOSMO-RS 24a workflow."""

    solute_gas_output: _VerifiedAsset
    solute_conductor_surface: _VerifiedAsset
    solvent_conductor_surface: _VerifiedAsset
    orca_major_version: int
    solute_charge: int
    solute_multiplicity: int
    solvent_charge: int
    solvent_multiplicity: int
    temperature_k: float
    functional: str
    basis: str
    parameterization: str
    uses_parameterized_cavity_radii: bool
    minimum_surface_segment_area_angstrom2: float

    @classmethod
    def from_paths(
        cls,
        *,
        solute_gas_output: str | Path,
        solute_conductor_surface: str | Path,
        solvent_conductor_surface: str | Path,
        orca_major_version: int,
        solute_charge: int,
        solute_multiplicity: int,
        solvent_charge: int,
        solvent_multiplicity: int,
        temperature_k: float,
        functional: str,
        basis: str,
        parameterization: str,
        uses_parameterized_cavity_radii: bool,
        minimum_surface_segment_area_angstrom2: float,
    ) -> "OpenCOSMORS24aInputBundle":
        if str(functional).upper() != "BP86" or str(basis).lower() != "def2-tzvpd":
            raise ValueError("openCOSMO-RS 24a requires the BP86/def2-TZVPD QC level.")
        if int(orca_major_version) != 6:
            raise ValueError("openCOSMO-RS 24a requires the audited ORCA 6 workflow.")
        if not _is_close(temperature_k, _REFERENCE_TEMPERATURE_K):
            raise ValueError("openCOSMO-RS 24a is locked to 298.15 K.")
        if (
            int(solute_charge) != 0
            or int(solute_multiplicity) != 1
            or int(solvent_charge) != 0
            or int(solvent_multiplicity) != 1
        ):
            raise ValueError(
                "This boundary accepts only neutral closed-shell solute and solvent assets."
            )
        if parameterization != OPEN_COSMORS_24A_PARAMETERIZATION:
            raise ValueError(
                f"Parameterization must be {OPEN_COSMORS_24A_PARAMETERIZATION!r}."
            )
        if uses_parameterized_cavity_radii is not True:
            raise ValueError(
                "openCOSMO-RS 24a requires its parameterized cavity radii."
            )
        if not _is_close(
            minimum_surface_segment_area_angstrom2,
            _MINIMUM_SURFACE_SEGMENT_AREA_ANGSTROM2,
        ):
            raise ValueError(
                "openCOSMO-RS 24a requires the 0.01 angstrom squared "
                "minimum surface-segment area."
            )

        gas = _VerifiedAsset.from_path(
            solute_gas_output,
            role="solute gas output",
        )
        solute_surface = _VerifiedAsset.from_path(
            solute_conductor_surface,
            role="solute conductor surface",
        )
        solvent_surface = _VerifiedAsset.from_path(
            solvent_conductor_surface,
            role="solvent conductor surface",
        )
        if len({gas.path, solute_surface.path, solvent_surface.path}) != 3:
            raise ValueError("COSMO-RS requires three distinct input assets.")

        return cls(
            solute_gas_output=gas,
            solute_conductor_surface=solute_surface,
            solvent_conductor_surface=solvent_surface,
            orca_major_version=6,
            solute_charge=0,
            solute_multiplicity=1,
            solvent_charge=0,
            solvent_multiplicity=1,
            temperature_k=_REFERENCE_TEMPERATURE_K,
            functional="BP86",
            basis="def2-TZVPD",
            parameterization=OPEN_COSMORS_24A_PARAMETERIZATION,
            uses_parameterized_cavity_radii=True,
            minimum_surface_segment_area_angstrom2=(
                _MINIMUM_SURFACE_SEGMENT_AREA_ANGSTROM2
            ),
        )

    @classmethod
    def from_orca_run(
        cls,
        work_directory: str | Path,
        stem: str,
    ) -> "OpenCOSMORS24aInputBundle":
        """Bind the three audited assets emitted by one ORCA 6 COSMO-RS run."""

        if not _SAFE_ORCA_STEM_RE.fullmatch(str(stem)):
            raise ValueError("ORCA COSMO-RS stem must be one safe basename.")
        workdir = Path(work_directory).expanduser().resolve()
        if not workdir.is_dir():
            raise ValueError("ORCA COSMO-RS work directory must exist.")
        return cls.from_paths(
            solute_gas_output=workdir / f"{stem}.solute_vac.lastout",
            solute_conductor_surface=workdir / f"{stem}.solute.orcacosmo",
            solvent_conductor_surface=workdir / f"{stem}.solvent.orcacosmo",
            orca_major_version=6,
            solute_charge=0,
            solute_multiplicity=1,
            solvent_charge=0,
            solvent_multiplicity=1,
            temperature_k=_REFERENCE_TEMPERATURE_K,
            functional="BP86",
            basis="def2-TZVPD",
            parameterization=OPEN_COSMORS_24A_PARAMETERIZATION,
            uses_parameterized_cavity_radii=True,
            minimum_surface_segment_area_angstrom2=(
                _MINIMUM_SURFACE_SEGMENT_AREA_ANGSTROM2
            ),
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "scientific_family": "COSMO-RS",
            "continuum_equation_switch": False,
            "implementation_boundary": "external-orca-opencosmors-result-v1",
            "parameterization": self.parameterization,
            "orca_major_version": self.orca_major_version,
            "qc_level": {
                "functional": self.functional,
                "basis": self.basis,
            },
            "temperature_k": self.temperature_k,
            "system_domain": {
                "solute_charge": self.solute_charge,
                "solute_multiplicity": self.solute_multiplicity,
                "solvent_charge": self.solvent_charge,
                "solvent_multiplicity": self.solvent_multiplicity,
            },
            "surface_contract": {
                "uses_parameterized_cavity_radii": (
                    self.uses_parameterized_cavity_radii
                ),
                "minimum_segment_area_angstrom2": (
                    self.minimum_surface_segment_area_angstrom2
                ),
            },
            "assets": {
                "solute_gas_output": self.solute_gas_output.as_manifest(),
                "solute_conductor_surface": (
                    self.solute_conductor_surface.as_manifest()
                ),
                "solvent_conductor_surface": (
                    self.solvent_conductor_surface.as_manifest()
                ),
            },
        }


@dataclass(frozen=True)
class OpenCOSMORSSolvationResult:
    delta_g_solvation_hartree: float
    delta_g_solvation_kcal_mol: float
    temperature_k: float
    input_manifest: dict[str, Any]
    output_sha256: str


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_TEMPERATURE_RE = re.compile(
    rf"Reference\s+temperature\s*:\s*({_NUMBER})\s*K",
    re.IGNORECASE,
)
_SOLVATION_RE = re.compile(
    rf"Free\s+energy\s+of\s+solvation\s*\(dGsolv\)\s*:\s*"
    rf"({_NUMBER})\s*Eh\s+({_NUMBER})\s*kcal/mol",
    re.IGNORECASE,
)


def render_orca_opencosmors24a_input(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Sequence[float]],
    *,
    solvent_alias: str,
    maxcore_mb: int = 2000,
    nprocs: int = 1,
) -> str:
    """Render one neutral-singlet fixed-geometry ORCA 6 COSMO-RS input."""

    if isinstance(symbols, (str, bytes)):
        raise ValueError("COSMO-RS symbols must be a non-empty sequence.")
    try:
        normalized_symbols = tuple(str(symbol) for symbol in symbols)
    except TypeError as exc:
        raise ValueError("COSMO-RS symbols must be a non-empty sequence.") from exc
    if not normalized_symbols:
        raise ValueError("COSMO-RS symbols must be a non-empty sequence.")
    if any(
        _ELEMENT_SYMBOL_RE.fullmatch(symbol) is None for symbol in normalized_symbols
    ):
        raise ValueError("COSMO-RS symbols must be canonical element symbols.")
    if not _SAFE_SOLVENT_ALIAS_RE.fullmatch(str(solvent_alias)):
        raise ValueError("COSMO-RS solvent alias contains unsafe characters.")
    if isinstance(maxcore_mb, bool) or not isinstance(maxcore_mb, int):
        raise ValueError("COSMO-RS maxcore_mb must be a positive integer.")
    if maxcore_mb <= 0:
        raise ValueError("COSMO-RS maxcore_mb must be a positive integer.")
    if isinstance(nprocs, bool) or not isinstance(nprocs, int) or nprocs <= 0:
        raise ValueError("COSMO-RS nprocs must be a positive integer.")

    if isinstance(coordinates_angstrom, (str, bytes)):
        raise ValueError("COSMO-RS coordinates must be an N by 3 sequence.")
    try:
        rows = tuple(
            tuple(float(value) for value in row) for row in coordinates_angstrom
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "COSMO-RS coordinates must be an N by 3 numeric sequence."
        ) from exc
    if len(rows) != len(normalized_symbols) or any(len(row) != 3 for row in rows):
        raise ValueError("COSMO-RS coordinates must match symbols with shape N by 3.")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("COSMO-RS coordinates must be finite.")

    lines = [f"! COSMORS({solvent_alias})"]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.extend((f"%maxcore {maxcore_mb}", "* xyz 0 1"))
    lines.extend(
        f"{symbol:<2} {x: .15f} {y: .15f} {z: .15f}"
        for symbol, (x, y, z) in zip(normalized_symbols, rows, strict=True)
    )
    lines.append("*")
    return "\n".join(lines) + "\n"


def validate_orca_opencosmors_completion(output: str) -> None:
    """Reject partial ORCA workflows even when the host process returned zero."""

    normalized = str(output)
    if normalized.upper().count("OPENCOSMO-RS CALCULATION") != 1:
        raise ValueError("Expected exactly one ORCA OPENCOSMO-RS CALCULATION block.")
    if normalized.upper().count("****ORCA TERMINATED NORMALLY****") != 1:
        raise ValueError(
            "ORCA COSMO-RS output did not terminate normally exactly once."
        )
    lowered = normalized.lower()
    for failure_marker in ("error termination", "unable to open file"):
        if failure_marker in lowered:
            raise ValueError(
                f"ORCA COSMO-RS output contains failure marker {failure_marker!r}."
            )
    if len(_TEMPERATURE_RE.findall(normalized)) != 1:
        raise ValueError("ORCA COSMO-RS output omitted its unique temperature result.")
    if len(_SOLVATION_RE.findall(normalized)) != 1:
        raise ValueError("ORCA COSMO-RS output omitted its unique dGsolv result.")


def parse_orca_opencosmors_solvation_output(
    output: str,
    *,
    inputs: OpenCOSMORS24aInputBundle,
) -> OpenCOSMORSSolvationResult:
    """Parse one version-locked ORCA openCOSMO-RS solvation result."""

    validate_orca_opencosmors_completion(output)

    temperature_matches = _TEMPERATURE_RE.findall(output)
    energy_matches = _SOLVATION_RE.findall(output)
    if len(temperature_matches) != 1 or len(energy_matches) != 1:
        raise ValueError(
            "Expected exactly one COSMO-RS reference temperature and dGsolv result."
        )

    temperature_k = _parse_number(temperature_matches[0])
    if not _is_close(temperature_k, inputs.temperature_k):
        raise ValueError("COSMO-RS output temperature does not match its input bundle.")

    hartree = _parse_number(energy_matches[0][0])
    kcal_mol = _parse_number(energy_matches[0][1])
    converted = hartree * HARTREE_TO_KCAL_MOL
    if abs(converted - kcal_mol) > _ENERGY_UNIT_TOLERANCE_KCAL_MOL:
        raise ValueError(
            "COSMO-RS Hartree/kcal values are inconsistent: "
            f"{converted:.9f} != {kcal_mol:.9f} kcal/mol."
        )

    return OpenCOSMORSSolvationResult(
        delta_g_solvation_hartree=hartree,
        delta_g_solvation_kcal_mol=kcal_mol,
        temperature_k=temperature_k,
        input_manifest=inputs.as_manifest(),
        output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
    )


def _parse_number(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _is_close(left: float, right: float, *, tolerance: float = 1.0e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance
