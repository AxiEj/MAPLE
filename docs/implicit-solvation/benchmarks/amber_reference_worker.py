#!/usr/bin/env python3
"""Generate independent Amber GB/LCPO energy and force references.

Run this script with the pinned AmberTools Python interpreter.  It delegates
topology/radii construction to parmchk2/tleap and energy/force evaluation to
the upstream Python-sander API; it does not reimplement GB or LCPO equations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from parmed.amber import AmberParm, Rst7
import sander


MODELS = {
    "hct": {"igb": 1, "radii": "mbondi", "profile": "hct-mbondi"},
    "obc1": {"igb": 2, "radii": "mbondi2", "profile": "obc1-mbondi2"},
    "obc2": {"igb": 5, "radii": "mbondi2", "profile": "obc2-mbondi2"},
    "gbn": {"igb": 7, "radii": "bondi", "profile": "gbn-bondi"},
    "gbn2": {"igb": 8, "radii": "mbondi3", "profile": "gbn2-mbondi3"},
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def run_command(command: list[str], *, cwd: Path, stem: str) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    (cwd / f"{stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cwd / f"{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    write_json(
        cwd / f"{stem}.command.json",
        {"command": command, "returncode": completed.returncode},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def normalize_mol2_charges(source: Path, destination: Path, total_charge: int) -> dict[str, Any]:
    lines = source.read_text(encoding="utf-8").splitlines()
    in_atoms = False
    atom_indices: list[int] = []
    charges: list[float] = []
    positions: list[list[float]] = []
    atom_names: list[str] = []
    for index, line in enumerate(lines):
        upper = line.upper()
        if upper.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if upper.startswith("@<TRIPOS>"):
            in_atoms = False
            continue
        if not in_atoms or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"MOL2 atom record lacks a partial charge: {line!r}")
        atom_indices.append(index)
        atom_names.append(fields[1])
        positions.append([float(fields[2]), float(fields[3]), float(fields[4])])
        charges.append(float(fields[8]))
    if not charges:
        raise ValueError(f"MOL2 contains no atoms: {source}")
    raw_sum = float(sum(charges))
    correction = (float(total_charge) - raw_sum) / len(charges)
    used = [charge + correction for charge in charges]
    for line_index, charge in zip(atom_indices, used):
        fields = lines[line_index].split()
        fields[8] = f"{charge:.10f}"
        lines[line_index] = " ".join(fields)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "atom_names": atom_names,
        "positions_angstrom": positions,
        "provider_charges_e": charges,
        "provider_sum_e": raw_sum,
        "correction_per_atom_e": correction,
        "normalized_charges_e": used,
        "normalized_sum_e": float(sum(used)),
    }


def sander_options(igb: int, *, external_dielectric: float, gbsa: int):
    options = sander.gas_input()
    options.igb = int(igb)
    options.gbsa = int(gbsa)
    options.cut = 999.0
    options.extdiel = float(external_dielectric)
    options.intdiel = 1.0
    options.saltcon = 0.0
    return options


def sander_energy_force(prmtop: Path, inpcrd: Path, igb: int, *, extdiel: float, gbsa: int):
    coordinates = Rst7.open(str(inpcrd))
    options = sander_options(igb, external_dielectric=extdiel, gbsa=gbsa)
    with sander.setup(str(prmtop), coordinates.coordinates, coordinates.box, options) as context:
        energy, force = context.energy_forces()
    return energy, np.asarray(force, dtype=np.float64)


def ambertools_version(prefix: Path) -> str:
    records = sorted((prefix / "conda-meta").glob("ambertools-*.json"))
    if len(records) != 1:
        return "unknown"
    return str(json.loads(records[0].read_text(encoding="utf-8"))["version"])


def generate(args: argparse.Namespace) -> None:
    cases_path = Path(args.cases).resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if cases.get("schema_version") != 1 or not cases.get("cases"):
        raise ValueError("Amber parity cases require schema_version=1 and a non-empty cases list.")
    output = Path(args.output).resolve()
    audit_root = output.parent / "audit"
    prefix = Path(sys.executable).resolve().parent.parent
    binaries = {
        name: prefix / "bin" / name for name in ("parmchk2", "tleap", "sander")
    }
    for name, path in binaries.items():
        if not path.is_file():
            raise FileNotFoundError(f"AmberTools executable is absent: {path}")

    generated_cases: list[dict[str, Any]] = []
    for case in cases["cases"]:
        source_mol2 = (cases_path.parent / case["mol2"]).resolve()
        if sha256_file(source_mol2) != case["mol2_sha256"]:
            raise ValueError(f"Pinned MOL2 hash mismatch for {case['case_id']}.")
        case_root = audit_root / case["case_id"]
        case_root.mkdir(parents=True, exist_ok=True)
        normalized_mol2 = case_root / "normalized.mol2"
        charge_audit = normalize_mol2_charges(
            source_mol2, normalized_mol2, int(case["charge"])
        )
        frcmod = case_root / "frcmod.gaff2"
        run_command(
            [
                str(binaries["parmchk2"]),
                "-i",
                str(normalized_mol2),
                "-f",
                "mol2",
                "-o",
                str(frcmod),
                "-s",
                "gaff2",
            ],
            cwd=case_root,
            stem="parmchk2",
        )
        model_records: dict[str, Any] = {}
        topology_charges: list[float] | None = None
        for model, settings in MODELS.items():
            model_root = case_root / model
            model_root.mkdir(parents=True, exist_ok=True)
            prmtop = model_root / "system.prmtop"
            inpcrd = model_root / "system.inpcrd"
            leap_input = model_root / "tleap.in"
            leap_input.write_text(
                "\n".join(
                    [
                        "source leaprc.gaff2",
                        f"set default PBradii {settings['radii']}",
                        f"MOL = loadmol2 {normalized_mol2}",
                        f"loadamberparams {frcmod}",
                        f"saveamberparm MOL {prmtop} {inpcrd}",
                        "quit",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            run_command(
                [str(binaries["tleap"]), "-f", str(leap_input)],
                cwd=model_root,
                stem="tleap",
            )
            topology = AmberParm(str(prmtop))
            charges = [float(atom.charge) for atom in topology.atoms]
            atom_names = [str(atom.name) for atom in topology.atoms]
            coordinates = np.asarray(
                Rst7.open(str(inpcrd)).coordinates, dtype=np.float64
            ).reshape((-1, 3))
            submitted = np.asarray(charge_audit["positions_angstrom"], dtype=np.float64)
            if atom_names != charge_audit["atom_names"]:
                raise ValueError(f"tleap changed atom order for {case['case_id']}/{model}.")
            if coordinates.shape != submitted.shape or not np.allclose(
                coordinates, submitted, rtol=0.0, atol=5.0e-4
            ):
                raise ValueError(f"tleap changed geometry for {case['case_id']}/{model}.")
            if topology_charges is None:
                topology_charges = charges
            elif not np.allclose(topology_charges, charges, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"tleap charges changed across radii profiles for {case['case_id']}.")

            reference_energy, reference_force = sander_energy_force(
                prmtop, inpcrd, settings["igb"], extdiel=1.0, gbsa=0
            )
            polar_energy, polar_force = sander_energy_force(
                prmtop, inpcrd, settings["igb"], extdiel=78.5, gbsa=0
            )
            complete_energy, complete_force = sander_energy_force(
                prmtop, inpcrd, settings["igb"], extdiel=78.5, gbsa=1
            )
            if abs(float(reference_energy.gb)) > 1.0e-10 or abs(float(reference_energy.surf)) > 1.0e-10:
                raise ValueError("The external-dielectric=1 Amber force reference is not solvent-free.")
            polar = float(polar_energy.gb)
            nonpolar = float(complete_energy.surf)
            model_records[model] = {
                **settings,
                "polar": polar,
                "nonpolar_lcpo": nonpolar,
                "total_lcpo": polar + nonpolar,
                "polar_force_kcal_mol_angstrom": (polar_force - reference_force).tolist(),
                "total_lcpo_force_kcal_mol_angstrom": (
                    complete_force - reference_force
                ).tolist(),
                "prmtop": os.path.relpath(prmtop, output.parent),
                "prmtop_sha256": sha256_file(prmtop),
                "inpcrd": os.path.relpath(inpcrd, output.parent),
                "inpcrd_sha256": sha256_file(inpcrd),
                "energy_settings": {
                    "solute_dielectric": 1.0,
                    "solvent_dielectric": 78.5,
                    "saltcon": 0.0,
                    "cut_angstrom": 999.0,
                    "gbsa": 1,
                    "surface_tension_kcal_mol_angstrom2": 0.005,
                },
            }
        generated_cases.append(
            {
                **case,
                "source_mol2": case["mol2"],
                "normalized_mol2": os.path.relpath(normalized_mol2, output.parent),
                "normalized_mol2_sha256": sha256_file(normalized_mol2),
                "charge_normalization": charge_audit,
                "charges_e": topology_charges,
                "models": model_records,
            }
        )

    manifest = {
        "schema_version": 1,
        "provider": "amber",
        "provider_version": ambertools_version(prefix),
        "generator": "AmberTools parmchk2/tleap plus Python-sander",
        "generator_python": sys.executable,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cases_source": os.path.relpath(cases_path, output.parent),
        "cases_source_sha256": sha256_file(cases_path),
        "binary_sha256": {name: sha256_file(path) for name, path in binaries.items()},
        "models": MODELS,
        "nonpolar_reference": "Amber gbsa=1 LCPO; OpenMM ACE is not treated as identical",
        "cases": generated_cases,
    }
    write_json(output, manifest)
    print(f"Wrote {len(generated_cases)} independent Amber cases to {output}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
