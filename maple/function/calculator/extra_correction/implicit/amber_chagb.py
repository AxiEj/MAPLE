"""AmberTools CHA-GB plus cavity/dispersion as an energy-only correction."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

import numpy as np

from .common import KJ_PER_MOL_PER_HARTREE
from .result import SolvationResult

KCAL_PER_MOL_PER_HARTREE = KJ_PER_MOL_PER_HARTREE / 4.184
_GBNSR6_PATTERN = re.compile(
    r"EGB\s*=\s*([-+0-9.Ee]+).*?ESURF\s*=\s*([-+0-9.Ee]+)",
    re.DOTALL,
)
_PBSA_PATTERN = re.compile(r"ECAVITY\s*=\s*([-+0-9.Ee]+)\s+EDISPER\s*=\s*([-+0-9.Ee]+)")
_GBNSR6_LOCK = threading.Lock()

_POLAR_PARAMETERS = {
    "epsin": 1.0,
    "epsout": 78.5,
    "istrng_molar": 0.0,
    "dprob_angstrom": 1.4,
    "space_angstrom": 0.3,
    "arcres_angstrom": 0.2,
    "alpb": 1,
    "chagb": 1,
    "radiopt": 0,
    "roh_angstrom": 0.586,
    "tau": 1.47,
    "rbornstat": 0,
}
_NONPOLAR_PARAMETERS = {
    "inp": 2,
    "istrng_millimolar": 0.0,
    "fillratio": 1.5,
    "saopt": 1,
    "decompopt": 2,
    "use_rmin": 1,
    "sprob_angstrom": 0.557,
    "vprob_angstrom": 1.3,
    "rhow_effect": 1.129,
    "use_sav": 1,
    "cavity_surften": 0.0378,
    "cavity_offset_kcal_mol": -0.5692,
}


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gbnsr6_components(text: str) -> dict[str, float]:
    """Read the final CHA-GB polar and surface-tension energy block."""
    matches = _GBNSR6_PATTERN.findall(text)
    if not matches:
        raise ValueError("GBNSR6 output does not contain final EGB/ESURF components.")
    polar, surface = (float(value) for value in matches[-1])
    if not math.isfinite(polar) or not math.isfinite(surface):
        raise ValueError("GBNSR6 returned a non-finite energy component.")
    return {"polar": polar, "surface_tension": surface}


def parse_pbsa_components(text: str) -> dict[str, float]:
    """Read the final PBSA cavity and dispersion energy block."""
    matches = _PBSA_PATTERN.findall(text)
    if not matches:
        raise ValueError(
            "PBSA output does not contain final ECAVITY/EDISPER components."
        )
    cavity, dispersion = (float(value) for value in matches[-1])
    if not math.isfinite(cavity) or not math.isfinite(dispersion):
        raise ValueError("PBSA returned a non-finite energy component.")
    return {"cavity": cavity, "dispersion": dispersion}


def require_no_frcmod_nonbonded_overrides(text: str) -> None:
    """Reject atom types outside the locked GAFF/GAFF2 nonbonded table."""
    lines = text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "NONBON"
        )
    except StopIteration as exc:
        raise ValueError("parmchk2 frcmod does not contain a NONBON section.") from exc
    overrides = [line for line in lines[start + 1 :] if line.strip()]
    if overrides:
        raise ValueError(
            "Input MOL2 atom types require non-GAFF/GAFF2 nonbonded overrides: "
            + "; ".join(overrides)
        )


def render_typed_mol2(
    source_text: str,
    positions_angstrom,
    charges_e,
) -> str:
    """Preserve input atom types/connectivity while replacing geometry/charges."""
    positions = np.asarray(positions_angstrom, dtype=np.float64)
    charges = np.asarray(charges_e, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or charges.shape != (positions.shape[0],)
        or not np.isfinite(positions).all()
        or not np.isfinite(charges).all()
    ):
        raise ValueError(
            "CHA-GB requires finite Nx3 coordinates and one finite fixed charge per atom."
        )

    output: list[str] = []
    section = ""
    atom_index = 0
    molecule_nonempty_index = 0
    for line in source_text.splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[9:].strip().upper()
            molecule_nonempty_index = 0
            output.append(line)
            continue
        if section == "MOLECULE" and line.strip():
            if molecule_nonempty_index == 3:
                output.append("USER_CHARGES")
            else:
                output.append(line)
            molecule_nonempty_index += 1
            continue
        if section == "ATOM" and line.strip():
            fields = line.split()
            if len(fields) < 6:
                raise ValueError("Malformed MOL2 ATOM row.")
            if atom_index >= len(charges):
                raise ValueError(
                    "MOL2 contains more atoms than the fixed-charge vector."
                )
            if len(fields) == 6:
                fields.extend(("1", "MOL"))
            elif len(fields) == 7:
                fields.append("MOL")
            fields[2:5] = [f"{coordinate:.10f}" for coordinate in positions[atom_index]]
            if len(fields) == 8:
                fields.append(f"{charges[atom_index]:.12f}")
            else:
                fields[8] = f"{charges[atom_index]:.12f}"
            output.append(" ".join(fields))
            atom_index += 1
            continue
        output.append(line)
    if atom_index != len(charges):
        raise ValueError(
            f"MOL2/charge length mismatch: {atom_index} atoms, {len(charges)} charges."
        )
    return "\n".join(output) + "\n"


def _ambertools_version(executable: Path) -> str:
    conda_meta = executable.parent.parent / "conda-meta"
    if conda_meta.is_dir():
        for path in sorted(conda_meta.glob("ambertools-*.json")):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(metadata.get("name", "")).lower() == "ambertools" and metadata.get(
                "version"
            ):
                return str(metadata["version"])
    return "unknown-external"


def _resolve_bundle(executable: str) -> dict[str, Path]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise ImportError(
            "AmberTools CHA-GB requires gbnsr6, pbsa, parmchk2, and tleap from "
            f"one installation; {executable!r} was not found."
        )
    gbnsr6 = Path(os.path.abspath(resolved))
    directory = gbnsr6.parent
    bundle = {"gbnsr6": gbnsr6}
    for name in ("pbsa", "parmchk2", "tleap"):
        candidate = directory / name
        if not candidate.is_file():
            raise ImportError(
                "AmberTools CHA-GB requires gbnsr6, pbsa, parmchk2, and tleap "
                f"from one installation; missing {candidate}."
            )
        bundle[name] = candidate.absolute()
    return bundle


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    label: str,
    audit_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=False,
    )
    stdout_path = cwd / f"{label}.stdout.log"
    stderr_path = cwd / f"{label}.stderr.log"
    command_path = cwd / f"{label}.command.json"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    command_path.write_text(
        json.dumps(
            {
                "label": label,
                "command": command,
                "returncode": completed.returncode,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        for source in (stdout_path, stderr_path, command_path):
            shutil.copy2(source, audit_dir / source.name)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-1000:].strip()
        raise RuntimeError(
            f"AmberTools {label} exited with code {completed.returncode}: {detail}"
        )
    return completed


class AmberToolsChaGB:
    """Locked CHA-GB EGB plus PBSA ECAVITY+EDISPER, without MM gas energy."""

    supported_properties = frozenset({"energy"})

    def __init__(
        self,
        atoms,
        charges,
        *,
        executable: str = "gbnsr6",
        timeout: float = 3600.0,
        audit_dir: str | os.PathLike[str] | None = None,
    ):
        metadata = atoms.info.get("mol2")
        if not metadata:
            raise ValueError("AmberTools CHA-GB requires a typed MOL2 input.")
        source_path = Path(metadata["path"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        self.source_path = source_path
        self.source_text = source_path.read_text(encoding="utf-8")
        self.symbols = list(atoms.get_chemical_symbols())
        self.atom_types = list(metadata.get("atom_types") or [])
        if len(self.atom_types) != len(atoms):
            raise ValueError("MOL2 atom-type metadata does not match the atom count.")
        self.charges = np.asarray(charges, dtype=np.float64).copy()
        if self.charges.shape != (len(atoms),) or not np.isfinite(self.charges).all():
            raise ValueError("AmberTools CHA-GB requires one finite charge per atom.")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("AmberTools CHA-GB timeout must be positive.")
        self.executables = _resolve_bundle(executable)
        self.audit_dir = Path(audit_dir).resolve() if audit_dir is not None else None
        self.gbnsr6_lock_path = (
            Path(tempfile.gettempdir())
            / f"maple-gbnsr6-{_sha256_file(self.executables['gbnsr6'])[:16]}.lock"
        )
        self._provenance = {
            "provider": "ambertools",
            "provider_version": _ambertools_version(self.executables["gbnsr6"]),
            "method": "gb",
            "model": "chagb",
            "profile": "chagb-bondi-pbsa-inp2",
            "nonpolar": "cavity-dispersion",
            "radii": "bondi",
            "topology_force_field": "gaff2",
            "topology_atom_types": "preserved-from-input-mol2",
            "input_atom_types": self.atom_types,
            "input_mol2": str(self.source_path),
            "input_mol2_sha256": _sha256_file(self.source_path),
            "polar_component": "gbnsr6:EGB",
            "nonpolar_components": ["pbsa:ECAVITY", "pbsa:EDISPER"],
            "polar_parameters": dict(_POLAR_PARAMETERS),
            "nonpolar_parameters": dict(_NONPOLAR_PARAMETERS),
            "gas_phase_mm_energy_used": False,
            "bonded_mm_energy_used": False,
            "provider_substitution": False,
            "supported_properties": ["energy"],
            "execution_control": {
                "gbnsr6_in_process_serialization": True,
                "gbnsr6_posix_interprocess_serialization": os.name == "posix",
                "reason": (
                    "AmberTools 26 GBNSR6 allocator failures were observed under "
                    "concurrent independent processes; identical serial runs passed."
                ),
            },
            "executables": {
                name: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for name, path in sorted(self.executables.items())
            },
            "citations": [
                "Mukhopadhyay et al., JCTC 2014, DOI:10.1021/ct4010917",
                "Aguilar et al., JCTC 2025, DOI:10.1021/acs.jctc.4c01471",
                "Tan, Tan, and Luo, JPCB 2007, DOI:10.1021/jp073399n",
            ],
        }

    @property
    def provenance(self) -> dict[str, object]:
        return dict(self._provenance)

    @staticmethod
    def _gbnsr6_input() -> str:
        polar = _POLAR_PARAMETERS
        return (
            "MAPLE Route-1 AM1-BCC CHA-GB polar correction\n"
            "&cntrl\n"
            "  inp=1,\n"
            "/\n"
            "&gb\n"
            f"  epsin={polar['epsin']}, epsout={polar['epsout']}, "
            f"istrng={polar['istrng_molar']},\n"
            f"  dprob={polar['dprob_angstrom']}, space={polar['space_angstrom']}, "
            f"arcres={polar['arcres_angstrom']},\n"
            f"  alpb={polar['alpb']}, chagb={polar['chagb']}, "
            f"radiopt={polar['radiopt']},\n"
            f"  ROH={polar['roh_angstrom']}, tau={polar['tau']}, "
            f"rbornstat={polar['rbornstat']},\n"
            "  cavity_surften=0.005,\n"
            "/\n"
        )

    @staticmethod
    def _pbsa_input() -> str:
        model = _NONPOLAR_PARAMETERS
        return (
            "MAPLE Route-1 PBSA cavity/dispersion correction\n"
            "&cntrl\n"
            f"  inp={model['inp']},\n"
            "/\n"
            "&pb\n"
            f"  npbverb=1, istrng={model['istrng_millimolar']}, "
            f"fillratio={model['fillratio']}, saopt={model['saopt']},\n"
            f"  decompopt={model['decompopt']}, use_rmin={model['use_rmin']}, "
            f"sprob={model['sprob_angstrom']}, vprob={model['vprob_angstrom']},\n"
            f"  rhow_effect={model['rhow_effect']}, use_sav={model['use_sav']},\n"
            f"  cavity_surften={model['cavity_surften']}, "
            f"cavity_offset={model['cavity_offset_kcal_mol']},\n"
            "/\n"
        )

    def _write_audit(
        self,
        work: Path,
        commands: list[dict[str, object]],
        result: SolvationResult,
    ) -> None:
        if self.audit_dir is None:
            return
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "molecule.mol2",
            "molecule.frcmod",
            "tleap.in",
            "leap.log",
            "gbnsr6.in",
            "gbnsr6.out",
            "pbsa.in",
            "pbsa.out",
            "parmchk2.stdout.log",
            "parmchk2.stderr.log",
            "tleap.stdout.log",
            "tleap.stderr.log",
            "gbnsr6.stdout.log",
            "gbnsr6.stderr.log",
            "pbsa.stdout.log",
            "pbsa.stderr.log",
        ):
            source = work / name
            if source.is_file():
                shutil.copy2(source, self.audit_dir / name)
        (self.audit_dir / "amber-chagb.commands.json").write_text(
            json.dumps(commands, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.audit_dir / "amber-chagb.result.json").write_text(
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

    def _copy_to_audit(self, path: Path) -> None:
        if self.audit_dir is None or not path.is_file():
            return
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self.audit_dir / path.name)

    @contextmanager
    def _serialized_gbnsr6(self):
        with _GBNSR6_LOCK:
            if os.name != "posix":
                yield
                return
            import fcntl

            with self.gbnsr6_lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def evaluate(
        self,
        atoms,
        need_forces: bool = False,
        calculator=None,
    ) -> SolvationResult:
        if need_forces:
            raise NotImplementedError(
                "AmberTools CHA-GB/cavity-dispersion is SP-energy-only; "
                "no energy-consistent solvent forces are available."
            )
        if list(atoms.get_chemical_symbols()) != self.symbols:
            raise ValueError("CHA-GB atom count/order changed after provider setup.")

        commands: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="maple-chagb-") as temporary:
            work = Path(temporary)
            mol2 = work / "molecule.mol2"
            mol2.write_text(
                render_typed_mol2(
                    self.source_text,
                    atoms.get_positions(),
                    self.charges,
                ),
                encoding="utf-8",
            )
            self._copy_to_audit(mol2)
            frcmod = work / "molecule.frcmod"
            command = [
                str(self.executables["parmchk2"]),
                "-i",
                mol2.name,
                "-f",
                "mol2",
                "-o",
                frcmod.name,
                "-s",
                "gaff2",
            ]
            completed = _run(
                command,
                cwd=work,
                timeout=self.timeout,
                label="parmchk2",
                audit_dir=self.audit_dir,
            )
            self._copy_to_audit(frcmod)
            commands.append(
                {
                    "label": "parmchk2",
                    "command": command,
                    "returncode": completed.returncode,
                }
            )
            require_no_frcmod_nonbonded_overrides(frcmod.read_text(encoding="utf-8"))

            leap_input = work / "tleap.in"
            leap_input.write_text(
                "\n".join(
                    [
                        "source leaprc.gaff2",
                        "set default PBradii bondi",
                        f"MOL = loadmol2 {mol2.name}",
                        f"loadamberparams {frcmod.name}",
                        "saveamberparm MOL molecule.prmtop molecule.inpcrd",
                        "quit",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self._copy_to_audit(leap_input)
            command = [str(self.executables["tleap"]), "-f", leap_input.name]
            completed = _run(
                command,
                cwd=work,
                timeout=self.timeout,
                label="tleap",
                audit_dir=self.audit_dir,
            )
            self._copy_to_audit(work / "leap.log")
            commands.append(
                {
                    "label": "tleap",
                    "command": command,
                    "returncode": completed.returncode,
                }
            )
            prmtop = work / "molecule.prmtop"
            inpcrd = work / "molecule.inpcrd"
            if not prmtop.is_file() or not inpcrd.is_file():
                raise RuntimeError(
                    "AmberTools tleap did not create molecule.prmtop/molecule.inpcrd."
                )

            gb_input = work / "gbnsr6.in"
            gb_output = work / "gbnsr6.out"
            gb_input.write_text(self._gbnsr6_input(), encoding="utf-8")
            self._copy_to_audit(gb_input)
            command = [
                str(self.executables["gbnsr6"]),
                "-O",
                "-i",
                gb_input.name,
                "-o",
                gb_output.name,
                "-p",
                prmtop.name,
                "-c",
                inpcrd.name,
            ]
            with self._serialized_gbnsr6():
                completed = _run(
                    command,
                    cwd=work,
                    timeout=self.timeout,
                    label="gbnsr6",
                    audit_dir=self.audit_dir,
                )
            self._copy_to_audit(gb_output)
            commands.append(
                {
                    "label": "gbnsr6",
                    "command": command,
                    "returncode": completed.returncode,
                }
            )
            polar = parse_gbnsr6_components(gb_output.read_text(encoding="utf-8"))[
                "polar"
            ]

            pb_input = work / "pbsa.in"
            pb_output = work / "pbsa.out"
            pb_input.write_text(self._pbsa_input(), encoding="utf-8")
            self._copy_to_audit(pb_input)
            command = [
                str(self.executables["pbsa"]),
                "-O",
                "-i",
                pb_input.name,
                "-o",
                pb_output.name,
                "-p",
                prmtop.name,
                "-c",
                inpcrd.name,
            ]
            completed = _run(
                command,
                cwd=work,
                timeout=self.timeout,
                label="pbsa",
                audit_dir=self.audit_dir,
            )
            self._copy_to_audit(pb_output)
            commands.append(
                {
                    "label": "pbsa",
                    "command": command,
                    "returncode": completed.returncode,
                }
            )
            nonpolar = parse_pbsa_components(pb_output.read_text(encoding="utf-8"))
            cavity = nonpolar["cavity"]
            dispersion = nonpolar["dispersion"]
            total = polar + cavity + dispersion
            provenance = {
                **self.provenance,
                "generated_mol2_sha256": _sha256_file(mol2),
                "reported_formula": "EGB + ECAVITY + EDISPER",
            }
            result = SolvationResult(
                energy_hartree=total / KCAL_PER_MOL_PER_HARTREE,
                components_hartree={
                    "polar": polar / KCAL_PER_MOL_PER_HARTREE,
                    "nonpolar": (cavity + dispersion) / KCAL_PER_MOL_PER_HARTREE,
                    "cavity": cavity / KCAL_PER_MOL_PER_HARTREE,
                    "dispersion": dispersion / KCAL_PER_MOL_PER_HARTREE,
                },
                provenance=provenance,
            )
            self._write_audit(work, commands, result)
            return result
