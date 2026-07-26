#!/usr/bin/env python3
"""Audit AmberTools MLSES PB energy, force, and local small-molecule timing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    command_provenance,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

DEFAULT_PBSA = "/home/axie/miniconda3/envs/maple-ambertools/bin/pbsa"
DEFAULT_INPUT_DIR = (
    REPOSITORY_ROOT
    / ".omx/benchmarks/route1-performance-20260724/mobley_1017962"
)
DEFAULT_WORK_DIR = (
    REPOSITORY_ROOT / ".omx/benchmarks/route1-mlses-pb-feasibility-20260726"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-mlses-pb-feasibility-probe-2026-07-26.json"
)

EXPECTED_PBSA_SHA256 = (
    "1cb42f0464e031b5f4331b8ae335cb0c130ddad50305956695efb62a19b2defd"
)
EXPECTED_PRMTOP_SHA256 = (
    "571cb2ddf776943390f5e458c53c1e97da69bd30a4564f200d2182feb28a428e"
)
EXPECTED_RST7_SHA256 = (
    "a4e4695fdc67bc74d990664db25bfe4de2b899015ff2e94f33e8b2a8ca536967"
)
AMBER26_MANUAL_SHA256 = (
    "d10151cfd32c9d0a7c3a1d1e0869f6d8645640447d8b8a9744a28ac844479f14"
)

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
PBSA_VERSION = re.compile(r"PBSA VERSION\s+([^\s]+)")
EPB = re.compile(rf"\bEPB\s*=\s*({FLOAT})")
INTERNAL_SECONDS = re.compile(rf"\| Total time\s+({FLOAT})")

GRIDS = (
    {
        "grid_id": "coarse-0.50-fill1.25",
        "space_angstrom": 0.5,
        "fillratio": 1.25,
        "arcres_angstrom": None,
        "fscale": None,
    },
    {
        "grid_id": "fine-0.25-fill2.00",
        "space_angstrom": 0.25,
        "fillratio": 2.0,
        "arcres_angstrom": 0.125,
        "fscale": 4,
    },
)
LEGAL_FORCE_PAIRS = (
    {"eneopt": 1, "frcopt": 1},
    {"eneopt": 2, "frcopt": 2},
    {"eneopt": 2, "frcopt": 3},
)
SURFACES = (
    {"surface_id": "classical-ses", "sasopt": 0},
    {"surface_id": "mlses-geniuses", "sasopt": 3},
)


def _resolve_file(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} was not found: {path}")
    return path


def _verify_input(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
    allow_other_inputs: bool,
) -> str:
    observed = sha256_file(path)
    if not allow_other_inputs and observed != expected_sha256:
        raise ValueError(
            f"{name} changed: expected {expected_sha256}, observed {observed}."
        )
    return observed


def _render_input(
    *,
    title: str,
    surface: dict[str, Any],
    grid: dict[str, Any],
    eneopt: int,
    frcopt: int,
) -> str:
    optional: list[str] = []
    if surface["sasopt"] == 3:
        optional.append("mlses_opt=0")
    if grid["arcres_angstrom"] is not None:
        optional.append(f"arcres={grid['arcres_angstrom']}")
    if grid["fscale"] is not None:
        optional.append(f"fscale={grid['fscale']}")
    optional_text = ", ".join(optional)
    if optional_text:
        optional_text += ","
    return f"""\
{title}
&cntrl
 ntx=1, imin=1, ipb=2, inp=0,
/
&pb
 npbverb=1, istrng=0.0,
 epsout=78.5, epsin=1.0, dprob=1.4,
 radiopt=0, sasopt={surface["sasopt"]}, {optional_text}
 fillratio={grid["fillratio"]}, nfocus=1, space={grid["space_angstrom"]},
 accept=0.000001, maxitn=10000,
 solvopt=3, npbopt=0, bcopt=6,
 eneopt={eneopt}, frcopt={frcopt},
 cutnb=15, cutsa=8, cutfd=7,
/
"""


def _error_lines(output: str) -> list[str]:
    return [
        line.strip()
        for line in output.splitlines()
        if "PB Bomb" in line or "PBSA BOMB" in line
    ]


def _run_job(
    *,
    job_dir: Path,
    pbsa: Path,
    prmtop: Path,
    rst7: Path,
    surface: dict[str, Any],
    grid: dict[str, Any],
    eneopt: int,
    frcopt: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    (job_dir / "top.prmtop").symlink_to(prmtop)
    (job_dir / "coords.rst7").symlink_to(rst7)
    input_path = job_dir / "input.in"
    input_path.write_text(
        _render_input(
            title=(
                "Route 1 MLSES PB feasibility: "
                f"{surface['surface_id']} {grid['grid_id']} "
                f"eneopt={eneopt} frcopt={frcopt}"
            ),
            surface=surface,
            grid=grid,
            eneopt=eneopt,
            frcopt=frcopt,
        ),
        encoding="utf-8",
    )

    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(pbsa),
            "-O",
            "-i",
            input_path.name,
            "-o",
            "output.out",
            "-p",
            "top.prmtop",
            "-c",
            "coords.rst7",
        ],
        cwd=job_dir,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    output_path = job_dir / "output.out"
    output = (
        output_path.read_text(encoding="utf-8", errors="replace")
        if output_path.is_file()
        else ""
    )
    force_path = job_dir / "force.dat"
    epb_values = [float(value) for value in EPB.findall(output)]
    internal_values = [
        float(value) for value in INTERNAL_SECONDS.findall(output)
    ]
    versions = PBSA_VERSION.findall(output)
    stderr_bytes = completed.stderr.encode("utf-8")
    stdout_bytes = completed.stdout.encode("utf-8")

    return {
        "surface_id": surface["surface_id"],
        "sasopt": surface["sasopt"],
        "grid": grid,
        "eneopt": eneopt,
        "frcopt": frcopt,
        "returncode": completed.returncode,
        "termination_signal": (
            -completed.returncode if completed.returncode < 0 else None
        ),
        "wall_seconds": wall_seconds,
        "pbsa_internal_seconds": internal_values[-1] if internal_values else None,
        "reaction_field_energy_kcal_mol": (
            epb_values[-1] if epb_values else None
        ),
        "pbsa_reported_version": versions[-1] if versions else None,
        "force_file": {
            "exists": force_path.is_file(),
            "size_bytes": force_path.stat().st_size if force_path.is_file() else 0,
            "sha256": sha256_file(force_path) if force_path.is_file() else None,
        },
        "error_lines": _error_lines(output),
        "input_sha256": sha256_file(input_path),
        "output_sha256": (
            sha256_file(output_path) if output_path.is_file() else None
        ),
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "stderr_excerpt": completed.stderr[-1000:],
    }


def _force_matrix(
    *,
    work_dir: Path,
    pbsa: Path,
    prmtop: Path,
    rst7: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for grid in GRIDS:
        for pair in LEGAL_FORCE_PAIRS:
            for surface in SURFACES:
                job_id = (
                    f"{grid['grid_id']}--e{pair['eneopt']}-f{pair['frcopt']}"
                    f"--{surface['surface_id']}"
                )
                records.append(
                    _run_job(
                        job_dir=work_dir / "forces" / job_id,
                        pbsa=pbsa,
                        prmtop=prmtop,
                        rst7=rst7,
                        surface=surface,
                        grid=grid,
                        eneopt=pair["eneopt"],
                        frcopt=pair["frcopt"],
                        timeout_seconds=timeout_seconds,
                    )
                )
    return records


def _energy_timing(
    *,
    work_dir: Path,
    pbsa: Path,
    prmtop: Path,
    rst7: Path,
    repeats: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for grid in GRIDS:
        surface_records: dict[str, list[dict[str, Any]]] = {}
        for surface in SURFACES:
            records: list[dict[str, Any]] = []
            for repeat in range(repeats):
                job_id = (
                    f"{grid['grid_id']}--{surface['surface_id']}"
                    f"--repeat-{repeat + 1:02d}"
                )
                record = _run_job(
                    job_dir=work_dir / "energy-timing" / job_id,
                    pbsa=pbsa,
                    prmtop=prmtop,
                    rst7=rst7,
                    surface=surface,
                    grid=grid,
                    eneopt=2,
                    frcopt=0,
                    timeout_seconds=timeout_seconds,
                )
                if (
                    record["returncode"] != 0
                    or record["reaction_field_energy_kcal_mol"] is None
                    or record["pbsa_internal_seconds"] is None
                ):
                    raise RuntimeError(
                        f"Energy/timing job failed: {job_id}: {record}"
                    )
                records.append(record)
            surface_records[surface["surface_id"]] = records

        classical = surface_records["classical-ses"]
        mlses = surface_records["mlses-geniuses"]
        classical_seconds = statistics.median(
            float(record["pbsa_internal_seconds"]) for record in classical
        )
        mlses_seconds = statistics.median(
            float(record["pbsa_internal_seconds"]) for record in mlses
        )
        classical_energy = float(
            classical[0]["reaction_field_energy_kcal_mol"]
        )
        mlses_energy = float(mlses[0]["reaction_field_energy_kcal_mol"])
        summaries.append(
            {
                "grid": grid,
                "repeats_per_surface": repeats,
                "records": surface_records,
                "classical_ses": {
                    "median_pbsa_internal_seconds": classical_seconds,
                    "reaction_field_energy_kcal_mol": classical_energy,
                },
                "mlses_geniuses": {
                    "median_pbsa_internal_seconds": mlses_seconds,
                    "reaction_field_energy_kcal_mol": mlses_energy,
                },
                "classical_over_mlses_speed_ratio": (
                    classical_seconds / mlses_seconds
                ),
                "mlses_minus_classical_energy_kcal_mol": (
                    mlses_energy - classical_energy
                ),
            }
        )
    return summaries


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    pbsa = _resolve_file(args.pbsa, name="PBSA executable")
    prmtop = _resolve_file(args.prmtop, name="Amber topology")
    rst7 = _resolve_file(args.rst7, name="Amber coordinates")
    pbsa_hash = _verify_input(
        pbsa,
        expected_sha256=EXPECTED_PBSA_SHA256,
        name="PBSA executable",
        allow_other_inputs=args.allow_other_inputs,
    )
    prmtop_hash = _verify_input(
        prmtop,
        expected_sha256=EXPECTED_PRMTOP_SHA256,
        name="Amber topology",
        allow_other_inputs=args.allow_other_inputs,
    )
    rst7_hash = _verify_input(
        rst7,
        expected_sha256=EXPECTED_RST7_SHA256,
        name="Amber coordinates",
        allow_other_inputs=args.allow_other_inputs,
    )
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")

    work_dir = Path(args.work_dir).resolve()
    force_records = _force_matrix(
        work_dir=work_dir,
        pbsa=pbsa,
        prmtop=prmtop,
        rst7=rst7,
        timeout_seconds=args.timeout,
    )
    timing = _energy_timing(
        work_dir=work_dir,
        pbsa=pbsa,
        prmtop=prmtop,
        rst7=rst7,
        repeats=args.repeats,
        timeout_seconds=args.timeout,
    )

    classical_force_records = [
        record
        for record in force_records
        if record["surface_id"] == "classical-ses"
    ]
    mlses_force_records = [
        record
        for record in force_records
        if record["surface_id"] == "mlses-geniuses"
    ]
    classical_controls_pass = all(
        record["returncode"] == 0
        and record["force_file"]["exists"]
        and record["force_file"]["size_bytes"] > 0
        for record in classical_force_records
    )
    mlses_force_supported = all(
        record["returncode"] == 0
        and record["force_file"]["exists"]
        and record["force_file"]["size_bytes"] > 0
        for record in mlses_force_records
    )
    mlses_faster_at_every_grid = all(
        result["classical_over_mlses_speed_ratio"] > 1.0 for result in timing
    )
    timings_finite = all(
        math.isfinite(float(result["classical_over_mlses_speed_ratio"]))
        and float(result["classical_over_mlses_speed_ratio"]) > 0.0
        for result in timing
    )

    canonical_inputs = (
        pbsa_hash == EXPECTED_PBSA_SHA256
        and prmtop_hash == EXPECTED_PRMTOP_SHA256
        and rst7_hash == EXPECTED_RST7_SHA256
        and args.repeats == 3
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "route1-mlses-pb-feasibility-probe",
        "protocol": {
            "protocol_id": "route1-mlses-pb-feasibility-v1",
            "status": "post-exploratory-label-free-capability-audit",
            "canonical_inputs": canonical_inputs,
            "force_matrix": {
                "surfaces": list(SURFACES),
                "grids": list(GRIDS),
                "legal_energy_force_pairs": list(LEGAL_FORCE_PAIRS),
            },
            "energy_timing": {
                "eneopt": 2,
                "frcopt": 0,
                "repeats_per_surface_and_grid": args.repeats,
                "wall_time_used_for_selection": False,
                "local_small_molecule_observation_only": True,
            },
        },
        "route1_boundary": {
            "formula": (
                "E_solution(R) = E_MLIP,gas(R) + "
                "G_polar(R,q_fixed) + G_nonpolar(R)"
            ),
            "candidate_changes_only_pb_dielectric_surface_generation": True,
            "hydration_labels_read": False,
            "hydration_label_fit_or_residual_model": False,
            "gas_phase_mm_energy_in_reported_potential": False,
            "gas_mlip_retrained": False,
        },
        "source_basis": {
            "primary_paper": {
                "doi": "10.1021/acs.jctc.1c00492",
                "url": "https://pubmed.ncbi.nlm.nih.gov/34516109/",
                "training_target": (
                    "Classical solvent-excluded-surface level-set geometry, "
                    "not hydration free energies or molecular forces."
                ),
                "reported_classical_ses_agreement_percent": 95.0,
                "reported_average_reaction_field_energy_deviation_percent": 1.0,
                "reported_cpu_surface_speedup": 2.5,
            },
            "amber26_manual": {
                "url": "https://ambermd.org/doc12/Amber26.pdf",
                "sha256": AMBER26_MANUAL_SHA256,
                "last_updated": "2026-06-22",
                "mlses_single_point_example": {
                    "ipb": 2,
                    "sasopt": 3,
                    "eneopt": 1,
                    "frcopt": 0,
                },
                "force_example_disables_nonpolar": True,
                "force_example_does_not_enable_mlses": True,
            },
        },
        "input": {
            "compound_id": "mobley_1017962",
            "compound_name": "methyl hexanoate",
            "atom_count": 23,
            "prmtop": {
                "path": str(prmtop),
                "sha256": prmtop_hash,
            },
            "rst7": {
                "path": str(rst7),
                "sha256": rst7_hash,
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pbsa": {
                "path": str(pbsa),
                "sha256": pbsa_hash,
                "conda_package": "ambertools 26.0 cuda_None nompi",
                "reported_version": next(
                    (
                        record["pbsa_reported_version"]
                        for record in force_records
                        if record["pbsa_reported_version"]
                    ),
                    None,
                ),
                "gpu_pbsa_executable_found": any(
                    candidate.is_file()
                    for candidate in (
                        pbsa.with_name("pbsa.cuda"),
                        pbsa.with_name("pbsa.MPI.cuda"),
                    )
                ),
            },
        },
        "command_provenance": command_provenance(
            __file__,
            {
                "pbsa": str(pbsa),
                "prmtop": str(prmtop),
                "rst7": str(rst7),
                "work_dir": str(work_dir),
                "repeats": args.repeats,
                "timeout": args.timeout,
                "allow_other_inputs": args.allow_other_inputs,
            },
            repository_root=REPOSITORY_ROOT,
        ),
        "force_capability": {
            "records": force_records,
            "classical_ses_controls_pass": classical_controls_pass,
            "mlses_all_legal_force_pairs_pass": mlses_force_supported,
            "finite_difference_gate_reached": mlses_force_supported,
            "interpretation": (
                "The same executable, topology, grids, and every legal "
                "ENEOPT/FRCOPT pair emit nonempty force.dat controls with "
                "classical SES. MLSES either terminates by signal or aborts "
                "while mapping dielectric-boundary force to atoms, so no "
                "candidate force exists to compare with an energy finite "
                "difference."
            ),
        },
        "energy_timing": {
            "records": timing,
            "all_ratios_finite": timings_finite,
            "mlses_faster_than_classical_at_every_grid": (
                mlses_faster_at_every_grid
            ),
            "claim_boundary": (
                "Three standalone CPU process repeats on one 23-atom molecule; "
                "not a general PBSA, GPU, or large-system performance claim."
            ),
        },
        "decision": {
            "status": (
                "rejected-no-atom-resolved-force-or-local-small-molecule-speedup"
            ),
            "learned_surface_is_route1_residual_cheating": False,
            "force_consistent_runtime": False,
            "optimization": False,
            "relaxed_scan": False,
            "md": False,
            "single_point_provider_added": False,
            "full_freesolv_screen_opened": False,
            "new_dependency_added": False,
            "default_provider_changed": False,
            "reason": (
                "MLSES is an admissible frozen geometry surrogate in principle, "
                "not a hydration-label residual. The maintained local AmberTools "
                "runtime supplies no atom-resolved MLSES force for any legal "
                "force pair, and it is slower than classical SES in both local "
                "small-molecule grid observations. It therefore supplies no "
                "Route-1 product advantage over the existing force-capable "
                "OBC-II/ACE or SP-only CHA-GB/cavity-dispersion profiles."
            ),
        },
    }
    if not classical_controls_pass:
        raise RuntimeError("Classical SES force controls did not pass.")
    if not timings_finite:
        raise RuntimeError("Energy/timing ratios are invalid.")

    output = Path(args.output).resolve()
    write_json_atomic(output, seal_artifact(payload))
    print(f"Wrote MLSES PB feasibility probe to {output}.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbsa", default=DEFAULT_PBSA)
    parser.add_argument(
        "--prmtop",
        default=str(DEFAULT_INPUT_DIR / "obc2.prmtop"),
    )
    parser.add_argument(
        "--rst7",
        default=str(DEFAULT_INPUT_DIR / "obc2.rst7"),
    )
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--allow-other-inputs",
        action="store_true",
        help="Permit a separately labelled noncanonical binary/topology comparison.",
    )
    return parser


def main() -> None:
    run_probe(build_parser().parse_args())


if __name__ == "__main__":
    main()
