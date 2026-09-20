"""External APBS adapter for aqueous linearized Poisson--Boltzmann SP jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .common import KJ_PER_MOL_PER_HARTREE, build_openmm_topology
from .nonpolar import APBSSASANonpolarProvider
from .radii import OpenMMMbondi2RadiusProvider
from .result import SolvationResult

_PRINT_PATTERNS = (
    r"Global\s+net\s+{kind}\s+energy\s*=\s*([-+0-9.Ee]+)\s*kJ/mol",
    r"PRINT\s+{kind}\s+ENERGY(?:\s+\d+)?\s*:\s*([-+0-9.Ee]+)\s*kJ/mol",
    r"{kind}\s+ENERGY[^\n:]*:\s*([-+0-9.Ee]+)\s*kJ/mol",
)


def _sha256_if_file(path: str | os.PathLike[str]) -> str | None:
    candidate = Path(path)
    try:
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _canonical_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def parse_apbs_print_energy(output: str, kind: str) -> float:
    """Parse a named PRINT result from APBS stdout, returned in kJ/mol."""
    kind = str(kind).upper()
    if kind not in {"ELEC", "APOL"}:
        raise ValueError("APBS energy kind must be ELEC or APOL.")
    matches: list[str] = []
    for line in output.splitlines():
        for template in _PRINT_PATTERNS:
            match = re.search(
                template.format(kind=re.escape(kind)),
                line,
                re.IGNORECASE,
            )
            if match:
                matches.append(match.group(1))
                break
    if len(matches) > 1:
        raise ValueError(
            f"APBS provider output contains {len(matches)} named PRINT "
            f"{kind} ENERGY results; exactly one is required."
        )
    if matches:
        value = float(matches[0])
        if not np.isfinite(value):
            raise ValueError(f"APBS PRINT {kind} ENERGY result is non-finite.")
        return value
    raise ValueError(
        f"Could not find a uniquely named APBS PRINT {kind} ENERGY result "
        "in provider output."
    )


def _apbs_version(
    executable: str,
    timeout: float,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            text=True,
            capture_output=True,
            cwd=cwd,
            timeout=min(timeout, 30.0),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown-external"
    text = completed.stdout + "\n" + completed.stderr
    match = re.search(
        r"\bAPBS(?:\s+version)?\s+v?([0-9]+(?:\.[0-9]+)+)", text, re.IGNORECASE
    )
    if match:
        return match.group(1)
    standalone = re.search(r"(?m)^\s*([0-9]+(?:\.[0-9]+)+)\s*$", text)
    return standalone.group(1) if standalone else "unknown-external"


class APBSLPB:
    """APBS LPBE polar correction plus APBS apolar energy.

    APBS polar solvation is evaluated as the documented solvated-minus-reference
    pair of ELEC calculations.  The initial release intentionally requests no
    PB forces; grid-converged force certification is a separate release gate.
    """

    supported_properties = frozenset({"energy"})

    def __init__(
        self,
        atoms,
        charges,
        *,
        executable: str = "apbs",
        grid_spacing: float = 0.33,
        grid_points: int = 97,
        probe_radius: float = 1.4,
        surface_tension: float = 0.105,
        pressure: float = 0.0,
        timeout: float = 3600.0,
        audit_dir: str | os.PathLike[str] | None = None,
    ):
        self.atoms = atoms.copy()
        self.charges = np.asarray(charges, dtype=np.float64).copy()
        if self.charges.shape != (len(atoms),) or not np.isfinite(self.charges).all():
            raise ValueError("APBS LPB requires one finite partial charge per atom.")
        topology = build_openmm_topology(atoms)
        self.radius_provider = OpenMMMbondi2RadiusProvider()
        self.radius_result = self.radius_provider.assign(topology)
        self.radii = self.radius_result.radii_angstrom.copy()
        self.nonpolar_provider = APBSSASANonpolarProvider(
            probe_radius=probe_radius,
            surface_tension=surface_tension,
            pressure=pressure,
        )
        self.executable = executable
        self.grid_spacing = float(grid_spacing)
        if isinstance(grid_points, bool) or not isinstance(
            grid_points, (int, np.integer)
        ):
            raise ValueError("APBS grid_points must be an integer.")
        self.grid_points = int(grid_points)
        self.probe_radius = self.nonpolar_provider.probe_radius
        self.surface_tension = self.nonpolar_provider.surface_tension
        self.pressure = self.nonpolar_provider.pressure
        self.timeout = float(timeout)
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        if self.grid_points < 33 or (self.grid_points - 1) % 32 != 0:
            raise ValueError(
                "APBS grid_points must have the nlev=4 form c*32+1 "
                "(for example 65, 97, 129, or 161)."
            )
        if not np.isfinite(self.grid_spacing) or self.grid_spacing <= 0:
            raise ValueError("APBS grid_spacing must be finite and positive.")
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("APBS timeout must be finite and positive.")
        self._validate_grid_contains(atoms)
        self._executable_path = self._resolve_executable()
        self._executable_sha256 = _sha256_if_file(self._executable_path)
        if self._executable_sha256 is None:
            raise ImportError(
                f"APBS executable is not a readable regular file: {self._executable_path}"
            )
        with tempfile.TemporaryDirectory(prefix="maple-apbs-version-") as tmp:
            self._provider_version = _apbs_version(
                str(self._executable_path), self.timeout, cwd=tmp
            )

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "provider": "apbs",
            "method": "lpb",
            "profile": "generic-mbondi2",
            "radii": "mbondi2",
            "nonpolar": "sasa",
            "radius_provider": self.radius_result.provenance,
            "nonpolar_provider": self.nonpolar_provider.provenance,
            "surface_tension_kj_mol_a2": self.surface_tension,
            "pressure_kj_mol_a3": self.pressure,
            "bulk_solvent_density_a3": 0.0,
            "grid_spacing_angstrom": self.grid_spacing,
            "grid_points": self.grid_points,
            "solvent_dielectric": 78.5,
            "solute_dielectric": 1.0,
            "energy_only": True,
            "grid_convergence_required_per_system": True,
            "default_grid_is_not_production_certification": True,
            "provider_version": self._provider_version,
            "executable": str(self._executable_path),
            "executable_sha256": self._executable_sha256,
        }

    def _resolve_executable(self) -> Path:
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise ImportError(
                "APBS LPB requires the optional 'apbs' executable; "
                f"'{self.executable}' was not found on PATH."
            )
        return Path(resolved).resolve()

    def _verify_executable_identity(self) -> dict[str, str]:
        resolved = self._resolve_executable()
        if resolved != self._executable_path:
            raise RuntimeError(
                "APBS executable path drifted after provider construction: "
                f"expected {self._executable_path}, resolved {resolved}."
            )
        executable_hash = _sha256_if_file(resolved)
        if executable_hash is None or executable_hash != self._executable_sha256:
            raise RuntimeError(
                "APBS executable content drifted after provider construction: "
                f"expected SHA256 {self._executable_sha256}, got {executable_hash}."
            )
        return {
            "path": str(resolved),
            "sha256": executable_hash,
            "version": self._provider_version,
        }

    @staticmethod
    def _audit_path(audit_context, fallback: Path | None) -> Path | None:
        if audit_context is None:
            return fallback
        try:
            path = Path(audit_context.path)
        except (AttributeError, TypeError) as exc:
            raise TypeError("audit_context must expose a filesystem path.") from exc
        if not path.is_dir():
            raise ValueError("audit_context.path must be a pre-created directory.")
        return path.resolve()

    def grid_containment_metadata(self, atoms) -> dict[str, object]:
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError("APBS LPB requires finite Nx3 coordinates.")
        serialized_positions = self.serialized_coordinates(positions)
        molecular_span = np.ptp(serialized_positions, axis=0)
        # Include the largest atom sphere, the solvent probe, and the two-grid
        # SPL2 charge-support margin documented by APBS on each boundary.
        boundary = (
            float(np.max(self.radii)) + self.probe_radius + 2.0 * self.grid_spacing
        )
        required = molecular_span + 2.0 * boundary
        available = (self.grid_points - 1) * self.grid_spacing
        margins = available - required
        if np.any(required > available):
            minimum = int(np.ceil(float(np.max(required)) / self.grid_spacing)) + 1
            suggested = ((max(minimum, 33) - 1 + 31) // 32) * 32 + 1
            raise ValueError(
                "APBS grid does not enclose the molecule, radii, probe surface, and SPL2 margin: "
                f"available={available:.3f} A, required={float(np.max(required)):.3f} A. "
                f"Use grid_points>={suggested} at grid_spacing={self.grid_spacing:g} A."
            )
        return {
            "validated": True,
            "grid_center": {
                "input_directive": "gcent mol 1",
                "moves_with_serialized_pqr_geometry": True,
                "serialized_coordinate_bounds_angstrom": {
                    "minimum": np.min(serialized_positions, axis=0).tolist(),
                    "maximum": np.max(serialized_positions, axis=0).tolist(),
                },
            },
            "grid_points": [self.grid_points] * 3,
            "grid_spacing_angstrom": [self.grid_spacing] * 3,
            "available_length_angstrom": [available] * 3,
            "requested_molecular_span_angstrom": np.ptp(
                positions, axis=0
            ).tolist(),
            "serialized_molecular_span_angstrom": molecular_span.tolist(),
            "boundary_allowance_angstrom": boundary,
            "required_length_angstrom": required.tolist(),
            "enclosure_margin_angstrom": margins.tolist(),
            "surface_discretization": "srfm mol",
            "charge_discretization": "chgm spl2",
        }

    def _validate_grid_contains(self, atoms) -> None:
        self.grid_containment_metadata(atoms)

    @staticmethod
    def serialized_coordinates(atoms_or_positions) -> np.ndarray:
        if hasattr(atoms_or_positions, "get_positions"):
            positions = atoms_or_positions.get_positions()
        else:
            positions = atoms_or_positions
        requested = np.asarray(positions, dtype=np.float64)
        if (
            requested.ndim != 2
            or requested.shape[1:] != (3,)
            or not np.isfinite(requested).all()
        ):
            raise ValueError("APBS serialized coordinates require finite Nx3 values.")
        return np.asarray(
            [[float(f"{value:.3f}") for value in row] for row in requested],
            dtype=np.float64,
        )

    def coordinate_serialization_metadata(
        self, atoms_or_positions
    ) -> dict[str, object]:
        if hasattr(atoms_or_positions, "get_positions"):
            positions = atoms_or_positions.get_positions()
        else:
            positions = atoms_or_positions
        requested = np.asarray(positions, dtype=np.float64)
        serialized = self.serialized_coordinates(requested)
        serialized_text = "\n".join(
            "".join(f"{value:8.3f}" for value in row) for row in requested
        )
        return {
            "requested_coordinates_angstrom": requested.tolist(),
            "requested_coordinates_sha256": _canonical_array_sha256(requested),
            "serialized_coordinates_angstrom": serialized.tolist(),
            "serialized_coordinates_sha256": hashlib.sha256(
                serialized_text.encode("ascii")
            ).hexdigest(),
            "pqr_coordinate_decimals": 3,
        }

    def validate_serialized_pair(
        self,
        center,
        minus,
        plus,
        *,
        requested_step_angstrom: float | None = None,
        symmetry_tolerance_angstrom: float = 1.0e-12,
    ) -> dict[str, object]:
        center_xyz = self.serialized_coordinates(center)
        minus_xyz = self.serialized_coordinates(minus)
        plus_xyz = self.serialized_coordinates(plus)
        if center_xyz.shape != minus_xyz.shape or center_xyz.shape != plus_xyz.shape:
            raise ValueError("APBS serialized coordinate pair shapes do not match.")
        delta_minus = center_xyz - minus_xyz
        delta_plus = plus_xyz - center_xyz
        moved = (np.abs(delta_minus) > symmetry_tolerance_angstrom) | (
            np.abs(delta_plus) > symmetry_tolerance_angstrom
        )
        if not np.any(moved):
            raise ValueError(
                "APBS displaced pair collapses to the same three-decimal PQR coordinates."
            )
        if not np.allclose(
            delta_minus[moved],
            delta_plus[moved],
            rtol=0.0,
            atol=symmetry_tolerance_angstrom,
        ):
            raise ValueError(
                "APBS serialized displaced coordinates do not straddle the "
                "serialized center symmetrically."
            )
        actual_steps = delta_plus[moved]
        result = {
            "center": self.coordinate_serialization_metadata(center),
            "minus": self.coordinate_serialization_metadata(minus),
            "plus": self.coordinate_serialization_metadata(plus),
            "serialized_minus_displacement_angstrom": delta_minus.tolist(),
            "serialized_plus_displacement_angstrom": delta_plus.tolist(),
            "serialized_denominator_angstrom": (plus_xyz - minus_xyz).tolist(),
            "actual_symmetric_step_angstrom": (
                float(actual_steps[0])
                if np.allclose(
                    actual_steps, actual_steps[0], rtol=0.0, atol=1.0e-12
                )
                else actual_steps.tolist()
            ),
            "symmetry_tolerance_angstrom": symmetry_tolerance_angstrom,
        }
        if requested_step_angstrom is not None:
            result["requested_step_angstrom"] = float(requested_step_angstrom)
        return result

    def validate_outer_displacements(
        self, center, displaced_pairs
    ) -> list[dict[str, object]]:
        """Preflight all outer minus/plus centers before nested force launches."""
        self._validate_grid_contains(center)
        records = []
        for pair in displaced_pairs:
            if len(pair) != 2:
                raise ValueError(
                    "Each APBS outer displacement must be a minus/plus pair."
                )
            minus, plus = pair
            self._validate_grid_contains(minus)
            self._validate_grid_contains(plus)
            records.append(self.validate_serialized_pair(center, minus, plus))
        return records

    def write_pqr(self, path: str | os.PathLike[str], atoms=None) -> None:
        atoms = self.atoms if atoms is None else atoms
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError("APBS PQR output requires finite Nx3 coordinates.")
        metadata = atoms.info.get("mol2", {})
        names = metadata.get("atom_names") or atoms.get_chemical_symbols()
        # This is deliberately synthetic: MOL2 molecule names are not residue
        # types and must not trigger provider-specific biomolecular typing.
        residue = "MOL"
        with Path(path).open("w", encoding="utf-8") as handle:
            for index, (name, xyz, charge, radius) in enumerate(
                zip(names, positions, self.charges, self.radii), start=1
            ):
                handle.write(
                    f"ATOM  {index:5d} {str(name):<4.4s} {residue:>3.3s} A   1    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f} {charge:9.5f} {radius:8.4f}\n"
                )
            handle.write("END\n")

    def render_input(self, pqr_name: str) -> str:
        n = self.grid_points
        h = self.grid_spacing
        shared = f"""\
    mg-manual
    dime {n} {n} {n}
    nlev 4
    grid {h:.6f} {h:.6f} {h:.6f}
    gcent mol 1
    mol 1
    lpbe
    bcfl mdh
    pdie 1.0
    chgm spl2
    srfm mol
    srad {self.probe_radius:.6f}
    swin 0.3
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce no"""
        return f"""\
read
    mol pqr {pqr_name}
end
elec name solv
{shared}
    sdie 78.5
end
elec name ref
{shared}
    sdie 1.0
end
{self.nonpolar_provider.render_input_block()}
print elecEnergy solv - ref end
print apolEnergy nonpolar end
quit
"""

    def evaluate(
        self, atoms, need_forces: bool = False, calculator=None, audit_context=None
    ) -> SolvationResult:
        if need_forces:
            raise NotImplementedError(
                "APBS LPB is SP-energy-only until grid-converged force parity is certified."
            )
        audit_dir = self._audit_path(audit_context, self.audit_dir)
        grid_metadata = self.grid_containment_metadata(atoms)
        coordinate_metadata = self.coordinate_serialization_metadata(atoms)
        executable_identity = self._verify_executable_identity()
        resolved = executable_identity["path"]
        with tempfile.TemporaryDirectory(prefix="maple-apbs-") as tmp:
            work = Path(tmp)
            pqr = work / "molecule.pqr"
            inp = work / "apbs.in"
            self.write_pqr(pqr, atoms)
            inp.write_text(self.render_input(pqr.name), encoding="utf-8")
            completed = subprocess.run(
                [resolved, inp.name],
                cwd=work,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            post_solve_identity = self._verify_executable_identity()
            if audit_dir is not None:
                audit_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pqr, audit_dir / pqr.name)
                shutil.copy2(inp, audit_dir / inp.name)
                (audit_dir / "apbs.stdout.log").write_text(
                    completed.stdout, encoding="utf-8"
                )
                (audit_dir / "apbs.stderr.log").write_text(
                    completed.stderr, encoding="utf-8"
                )
                (audit_dir / "apbs.command.json").write_text(
                    json.dumps(
                        {
                            "command": [resolved, inp.name],
                            "returncode": completed.returncode,
                            "executable_identity_before_solve": executable_identity,
                            "executable_identity_after_solve": post_solve_identity,
                            "coordinate_serialization": coordinate_metadata,
                            "grid_containment": grid_metadata,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"APBS exited with code {completed.returncode}; inspect the implicit-solvent audit logs."
                )
            polar_kj = parse_apbs_print_energy(completed.stdout, "ELEC")
            nonpolar_kj = parse_apbs_print_energy(completed.stdout, "APOL")
        provenance = {
            **self.provenance,
            "provider_version": self._provider_version,
            "executable": resolved,
            "executable_sha256": self._executable_sha256,
            "executable_identity_after_solve": post_solve_identity,
            "coordinate_serialization": coordinate_metadata,
            "grid_containment": grid_metadata,
            "citations": [
                "Baker et al., PNAS 2001, DOI:10.1073/pnas.181342398",
                "Jurrus et al., Protein Sci. 2018, DOI:10.1002/pro.3280",
                "Wagoner and Baker, PNAS 2006, DOI:10.1073/pnas.0600118103",
            ],
        }
        result = SolvationResult(
            energy_hartree=(polar_kj + nonpolar_kj) / KJ_PER_MOL_PER_HARTREE,
            components_hartree={
                "polar": polar_kj / KJ_PER_MOL_PER_HARTREE,
                "nonpolar": nonpolar_kj / KJ_PER_MOL_PER_HARTREE,
            },
            provenance=provenance,
        )
        if audit_dir is not None:
            (audit_dir / "apbs.result.json").write_text(
                json.dumps(
                    {
                        "energy_hartree": result.energy_hartree,
                        "components_hartree": result.components_hartree,
                        "provenance": result.provenance,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return result
