#!/usr/bin/env python3
"""Audit whether the locked AmberTools CHA-GB endpoint has usable derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MOL2 = (
    SCRIPT_DIR
    / "route1-multi-mlip-obc2-ti-raw/inputs/mobley_1017962.mol2"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-chagb-derivative-capability-audit-2026-07-25.json"
)
DEFAULT_AMBERCLASSIC_SOURCE = Path("/tmp/amberclassic-route1-audit-WupUOT")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_hashed_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    unsigned = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    payload = {**payload, "content_sha256": hashlib.sha256(unsigned).hexdigest()}
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def run_checked(command: list[str], *, cwd: Path, label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=False,
    )
    (cwd / f"{label}.stdout").write_text(completed.stdout, encoding="utf-8")
    (cwd / f"{label}.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-1200:].strip()
        raise RuntimeError(f"{label} failed with code {completed.returncode}: {detail}")


def require_no_nonbonded_overrides(frcmod: Path) -> None:
    lines = frcmod.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "NONBON")
    except StopIteration as exc:
        raise ValueError("parmchk2 output has no NONBON section.") from exc
    overrides = [line for line in lines[start + 1 :] if line.strip()]
    if overrides:
        raise ValueError(
            "The audit molecule requires non-GAFF/GAFF2 NONBON overrides: "
            + "; ".join(overrides)
        )


def parse_debug_force(text: str) -> dict[str, Any]:
    block = re.search(
        r"NUMERICAL,\s*ANALYTICAL FORCES.*?atom\s+(\d+)(.*?)"
        r"RMS force error\s*=\s*(" + FLOAT + ")",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if block is None:
        raise ValueError("GBNSR6 output has no numerical/analytical force block.")
    rows = re.findall(
        rf"^\s*([123])\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$",
        block.group(2),
        flags=re.MULTILINE,
    )
    if [int(row[0]) for row in rows] != [1, 2, 3]:
        raise ValueError("GBNSR6 force audit did not report all three axes.")
    numerical = [float(row[1]) for row in rows]
    analytical = [float(row[2]) for row in rows]
    differences = [float(row[3]) for row in rows]
    return {
        "atom_one_based": int(block.group(1)),
        "numerical_native_units": numerical,
        "analytical_native_units": analytical,
        "reported_difference_native_units": differences,
        "reported_rms_force_error": float(block.group(3)),
        "all_analytical_components_zero": all(value == 0.0 for value in analytical),
        "any_numerical_component_nonzero": any(value != 0.0 for value in numerical),
        "numerical_analytical_match": all(
            abs(numerical_value - analytical_value) <= 1.0e-8
            for numerical_value, analytical_value in zip(numerical, analytical)
        ),
        "unit_note": (
            "Values are retained in GBNSR6 debugf native output units; the "
            "eligibility decision uses only zero/nonzero and equality checks."
        ),
    }


def _uncommented_fortran(text: str) -> str:
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def _subroutine(text: str, name: str) -> str:
    match = re.search(
        rf"(?ims)^\s*subroutine\s+{re.escape(name)}\b.*?"
        rf"^\s*end\s+subroutine\s+{re.escape(name)}\b",
        text,
    )
    if match is None:
        raise ValueError(f"Missing expected upstream subroutine {name}.")
    return match.group(0)


def audit_amberclassic(source: Path) -> dict[str, Any]:
    files = {
        "force": source / "src/gbnsr6/force.F90",
        "egb": source / "src/gbnsr6/egb.F90",
        "nonpolar": source / "src/gbnsr6/np_force.F90",
        "driver": source / "src/gbnsr6/gbnsr6.F90",
        "developer_readme": source / "src/gbnsr6/README.txt",
        "manual_source": source / "doc/AmberClassic.lyx",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AmberClassic audit sources: " + ", ".join(missing))

    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    commit_date = subprocess.check_output(
        ["git", "-C", str(source), "show", "-s", "--format=%cI", "HEAD"],
        text=True,
    ).strip()
    egb = files["egb"].read_text(encoding="utf-8")
    nonpolar = files["nonpolar"].read_text(encoding="utf-8")
    force = files["force"].read_text(encoding="utf-8")
    readme = files["developer_readme"].read_text(encoding="utf-8")
    manual = files["manual_source"].read_text(encoding="utf-8")

    gb_equation = _uncommented_fortran(_subroutine(egb, "gb_equation")).lower()
    chagb_equation = _uncommented_fortran(
        _subroutine(egb, "chagb_equation")
    ).lower()
    nonpolar_active = _uncommented_fortran(nonpolar).lower()
    force_active = _uncommented_fortran(force).lower()

    msander = source / "src/msander"
    msander_files = sorted(path for path in msander.rglob("*") if path.is_file())
    msander_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in msander_files
    ).lower()
    igb9_pattern = re.compile(r"\bigb\s*(?:=|==|\.eq\.)\s*9\b")
    ar6_topology_flags = ("chunks_ar6", "necks_ar6", "volume_ar6")

    return {
        "repository": "https://github.com/Amber-MD/AmberClassic",
        "commit": commit,
        "commit_date": commit_date,
        "source_file_sha256": {
            str(path.relative_to(source)): sha256_file(path)
            for path in files.values()
        },
        "driver_plumbing": {
            "calls_egb_with_force_array": bool(
                "call egb" in force_active
                and re.search(r",\s*x\s*,\s*f\s*,", force_active)
            ),
            "calls_nonpolar_force": "call np_force" in force_active,
            "interpretation": (
                "Driver plumbing alone is not evidence that every endpoint "
                "component writes its derivative."
            ),
        },
        "polar_source": {
            "gb_equation_force_array_argument_present": bool(
                re.search(r"\b(?:f|pbfrc)\s*\(", gb_equation)
            ),
            "chagb_equation_force_array_argument_present": bool(
                re.search(r"\b(?:f|pbfrc)\s*\(", chagb_equation)
            ),
            "gb_equation_accumulates_epol": "epol = epol" in gb_equation,
            "chagb_equation_accumulates_epol": "epol = epol" in chagb_equation,
            "classification": "energy-only-in-inspected-equation-path",
        },
        "nonpolar_source": {
            "cavity_energy_uses_prtsas_or_prtsav": (
                "enbrfcav = cavity_surften * prtsas" in nonpolar_active
                and "enbrfcav = cavity_surften * prtsav" in nonpolar_active
            ),
            "active_np_cavity_derivative_call": bool(
                re.search(r"^\s*call\s+np_cavity\b", nonpolar_active, re.MULTILINE)
            ),
            "active_dispersion_force_assignments": len(
                re.findall(
                    r"^\s*f\s*\(\s*[123]\s*,.*?=",
                    nonpolar_active,
                    re.MULTILINE,
                )
            ),
            "classification": "cavity-energy-only-dispersion-partial-force",
        },
        "manual_boundary": {
            "states_gbnsr6_cannot_yet_be_used_in_dynamics": (
                "gbnsr6 can not yet be used in dynamics" in manual.lower()
            ),
        },
        "ar6_igb9": {
            "developer_readme_states_functionality_not_used_now": (
                "this functionality is not used now" in readme.lower()
            ),
            "developer_readme_mentions_future_igb9": "igb9" in readme.lower(),
            "msander_igb9_condition_count": len(igb9_pattern.findall(msander_text)),
            "msander_ar6_topology_flag_count": sum(
                msander_text.count(flag) for flag in ar6_topology_flags
            ),
            "classification": "not-exposed-by-current-msander-source",
        },
    }


def run_probe(mol2: Path, gbnsr6: Path) -> dict[str, Any]:
    bundle = {
        name: gbnsr6.parent / name for name in ("gbnsr6", "parmchk2", "tleap")
    }
    missing = [str(path) for path in bundle.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AmberTools executables: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="maple-chagb-force-audit-") as raw:
        work = Path(raw)
        target_mol2 = work / "molecule.mol2"
        shutil.copy2(mol2, target_mol2)
        frcmod = work / "molecule.frcmod"
        run_checked(
            [
                str(bundle["parmchk2"]),
                "-i",
                target_mol2.name,
                "-f",
                "mol2",
                "-o",
                frcmod.name,
                "-s",
                "gaff2",
            ],
            cwd=work,
            label="parmchk2",
        )
        require_no_nonbonded_overrides(frcmod)
        (work / "tleap.in").write_text(
            "\n".join(
                [
                    "source leaprc.gaff2",
                    "set default PBradii bondi",
                    "MOL = loadmol2 molecule.mol2",
                    "loadamberparams molecule.frcmod",
                    "saveamberparm MOL molecule.prmtop molecule.inpcrd",
                    "quit",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run_checked(
            [str(bundle["tleap"]), "-f", "tleap.in"],
            cwd=work,
            label="tleap",
        )
        debug_input = (
            "MAPLE Route-1 CHA-GB derivative capability audit\n"
            "&cntrl\n"
            "  imin=1, maxcyc=1, ntmin=0, inp=1,\n"
            "/\n"
            "&gb\n"
            "  epsin=1.0, epsout=78.5, istrng=0.0,\n"
            "  dprob=1.4, space=0.3, arcres=0.2,\n"
            "  alpb=1, chagb=1, radiopt=0,\n"
            "  ROH=0.586, tau=1.47, rbornstat=0,\n"
            "  cavity_surften=0.005,\n"
            "/\n"
            "&debugf\n"
            "  do_debugf=1, neglgdel=3, atomn(1)=1,\n"
            "/\n"
        )
        (work / "force-debug.in").write_text(debug_input, encoding="utf-8")
        run_checked(
            [
                str(bundle["gbnsr6"]),
                "-O",
                "-i",
                "force-debug.in",
                "-o",
                "force-debug.out",
                "-p",
                "molecule.prmtop",
                "-c",
                "molecule.inpcrd",
            ],
            cwd=work,
            label="gbnsr6",
        )
        output = work / "force-debug.out"
        parsed = parse_debug_force(output.read_text(encoding="utf-8"))
        return {
            "molecule": {
                "compound_id": "mobley_1017962",
                "name": "methyl hexanoate",
                "mol2": str(mol2),
                "mol2_sha256": sha256_file(mol2),
            },
            "input_sha256": hashlib.sha256(debug_input.encode("utf-8")).hexdigest(),
            "output_sha256": sha256_file(output),
            "executables": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in bundle.items()
            },
            "debug_force": parsed,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mol2 = Path(args.mol2).resolve()
    gbnsr6 = Path(args.gbnsr6).resolve()
    amberclassic = Path(args.amberclassic_source).resolve()
    if not mol2.is_file() or not gbnsr6.is_file():
        raise FileNotFoundError("Both --mol2 and --gbnsr6 must exist.")
    source = audit_amberclassic(amberclassic)
    probe = run_probe(mol2, gbnsr6)
    debug = probe["debug_force"]
    payload = {
        "schema_version": 1,
        "recorded_date": "2026-07-25",
        "artifact_type": "route1-chagb-derivative-capability-audit",
        "claim_scope": (
            "Source-pinned and one-molecule runtime audit of derivative "
            "availability for the locked AmberTools CHA-GB path. It is a "
            "capability rejection audit, not an accuracy benchmark."
        ),
        "route1_contract": {
            "name": "Additive fixed-charge PB/GB implicit solvation",
            "role": "Baseline/Product Route",
            "formula": (
                "E_solution(R) = E_MLIP,gas(R) + "
                "G_polar(R,q_fixed) + G_nonpolar(R)"
            ),
            "gas_phase_mm_energy": False,
            "mlip_retraining": False,
            "hydration_label_fit_or_residual": False,
        },
        "amberclassic_source_audit": source,
        "runtime_debug_probe": probe,
        "decision": {
            "chagb_full_endpoint_derivative_verified": False,
            "chagb_supported_tasks": ["sp"],
            "chagb_opt_scan_md": False,
            "ar6_igb9_runtime_candidate_available": False,
            "reason": (
                "The inspected polar equations accumulate energy without a "
                "force-array argument, the cavity derivative call is inactive, "
                "only partial dispersion force assignments remain, the official "
                "manual excludes GBNSR6 dynamics, and debugf reports nonzero "
                "numerical components beside zero analytical components."
            ),
            "runtime_probe_supports_rejection": (
                debug["all_analytical_components_zero"]
                and debug["any_numerical_component_nonzero"]
                and not debug["numerical_analytical_match"]
            ),
            "policy": (
                "Keep AmberTools CHA-GB/PBSA cavity-dispersion energy-only and "
                "fail closed before OPT, relaxed SCAN, MD, or force requests."
            ),
        },
        "primary_sources": {
            "amberclassic_commit": (
                "https://github.com/Amber-MD/AmberClassic/tree/"
                + source["commit"]
            ),
            "gbnsr6_original": "https://doi.org/10.1021/acs.jcim.7b00192",
            "ar6": "https://doi.org/10.1021/ct100392h",
        },
    }
    write_hashed_json(args.output, payload)
    print(f"Wrote CHA-GB derivative audit to {Path(args.output).resolve()}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", default=str(DEFAULT_MOL2))
    parser.add_argument(
        "--gbnsr6",
        default="/home/axie/miniconda3/envs/maple-ambertools/bin/gbnsr6",
    )
    parser.add_argument(
        "--amberclassic-source", default=str(DEFAULT_AMBERCLASSIC_SOURCE)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
