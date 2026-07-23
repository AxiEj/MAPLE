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

from .openmm_gb import KJ_PER_MOL_PER_HARTREE, build_openmm_topology
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


def parse_apbs_print_energy(output: str, kind: str) -> float:
    """Parse a named PRINT result from APBS stdout, returned in kJ/mol."""
    kind = str(kind).upper()
    for template in _PRINT_PATTERNS:
        match = re.search(template.format(kind=re.escape(kind)), output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    generic = re.findall(
        r"PRINT\s+ENERGY[^\n:]*:\s*([-+0-9.Ee]+)\s*kJ/mol",
        output,
        re.IGNORECASE,
    )
    if kind == "ELEC" and generic:
        return float(generic[0])
    if kind == "APOL" and len(generic) >= 2:
        return float(generic[-1])
    raise ValueError(f"Could not find APBS PRINT {kind} ENERGY result in provider output.")


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
    match = re.search(r"\bAPBS(?:\s+version)?\s+v?([0-9]+(?:\.[0-9]+)+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    standalone = re.search(r"(?m)^\s*([0-9]+(?:\.[0-9]+)+)\s*$", text)
    return standalone.group(1) if standalone else "unknown-external"


def _mbondi2_radii_angstrom(atoms) -> np.ndarray:
    try:
        from openmm.app.internal.customgbforces import GBSAOBC1Force
    except ImportError as exc:
        raise ImportError(
            "The APBS generic-mbondi2 profile uses OpenMM's upstream mbondi2 parameter assignment; "
            "install `maple[implicit-gb]`."
        ) from exc
    topology = build_openmm_topology(atoms)
    params = np.asarray(GBSAOBC1Force.getStandardParameters(topology), dtype=np.float64)
    return params[:, 0] * 10.0


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
        self.radii = _mbondi2_radii_angstrom(atoms)
        self.executable = executable
        self.grid_spacing = float(grid_spacing)
        if isinstance(grid_points, bool) or not isinstance(grid_points, (int, np.integer)):
            raise ValueError("APBS grid_points must be an integer.")
        self.grid_points = int(grid_points)
        self.probe_radius = float(probe_radius)
        self.surface_tension = float(surface_tension)
        self.pressure = float(pressure)
        self.timeout = float(timeout)
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None
        if self.grid_points < 33 or (self.grid_points - 1) % 32 != 0:
            raise ValueError(
                "APBS grid_points must have the nlev=4 form c*32+1 "
                "(for example 65, 97, 129, or 161)."
            )
        if self.grid_spacing <= 0:
            raise ValueError("APBS grid_spacing must be positive.")
        if self.probe_radius < 0:
            raise ValueError("APBS probe_radius must be non-negative.")
        if self.surface_tension < 0 or self.pressure < 0:
            raise ValueError("APBS surface_tension and pressure must be non-negative.")
        if self.timeout <= 0:
            raise ValueError("APBS timeout must be positive.")
        self._validate_grid_contains(atoms)

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "provider": "apbs",
            "method": "lpb",
            "profile": "generic-mbondi2",
            "radii": "mbondi2",
            "nonpolar": "sasa",
            "surface_tension_kj_mol_a2": self.surface_tension,
            "pressure_kj_mol_a3": self.pressure,
            "bulk_solvent_density_a3": 0.0,
            "grid_spacing_angstrom": self.grid_spacing,
            "grid_points": self.grid_points,
            "solvent_dielectric": 78.5,
            "solute_dielectric": 1.0,
        }

    def _validate_grid_contains(self, atoms) -> None:
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        molecular_span = np.ptp(positions, axis=0)
        # Include the largest atom sphere, the solvent probe, and the two-grid
        # SPL2 charge-support margin documented by APBS on each boundary.
        boundary = float(np.max(self.radii)) + self.probe_radius + 2.0 * self.grid_spacing
        required = molecular_span + 2.0 * boundary
        available = (self.grid_points - 1) * self.grid_spacing
        if np.any(required > available):
            minimum = int(np.ceil(float(np.max(required)) / self.grid_spacing)) + 1
            suggested = ((max(minimum, 33) - 1 + 31) // 32) * 32 + 1
            raise ValueError(
                "APBS grid does not enclose the molecule, radii, probe surface, and SPL2 margin: "
                f"available={available:.3f} A, required={float(np.max(required)):.3f} A. "
                f"Use grid_points>={suggested} at grid_spacing={self.grid_spacing:g} A."
            )

    def write_pqr(self, path: str | os.PathLike[str], atoms=None) -> None:
        atoms = self.atoms if atoms is None else atoms
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
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
apolar name nonpolar
    mol 1
    srfm sacc
    srad {self.probe_radius:.6f}
    swin 0.3
    sdens 10.0
    gamma {self.surface_tension:.8f}
    press {self.pressure:.8f}
    bconc 0.0
    dpos 0.05
    grid 0.5 0.5 0.5
    temp 298.15
    calcenergy total
    calcforce no
end
print elecEnergy solv - ref end
print apolEnergy nonpolar end
quit
"""

    def evaluate(self, atoms, need_forces: bool = False, calculator=None) -> SolvationResult:
        if need_forces:
            raise NotImplementedError(
                "APBS LPB is SP-energy-only until grid-converged force parity is certified."
            )
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise ImportError(
                f"APBS LPB requires the optional 'apbs' executable; '{self.executable}' was not found on PATH."
            )
        with tempfile.TemporaryDirectory(prefix="maple-apbs-") as tmp:
            work = Path(tmp)
            version = _apbs_version(resolved, self.timeout, cwd=work)
            executable_hash = _sha256_if_file(resolved)
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
            if self.audit_dir is not None:
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pqr, self.audit_dir / pqr.name)
                shutil.copy2(inp, self.audit_dir / inp.name)
                (self.audit_dir / "apbs.stdout.log").write_text(completed.stdout, encoding="utf-8")
                (self.audit_dir / "apbs.stderr.log").write_text(completed.stderr, encoding="utf-8")
                (self.audit_dir / "apbs.command.json").write_text(
                    json.dumps(
                        {
                            "command": [resolved, inp.name],
                            "returncode": completed.returncode,
                            "provider_version": version,
                            "executable_sha256": executable_hash,
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
            "provider_version": version,
            "executable": resolved,
            "executable_sha256": executable_hash,
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
        if self.audit_dir is not None:
            (self.audit_dir / "apbs.result.json").write_text(
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
