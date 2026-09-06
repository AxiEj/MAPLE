#!/usr/bin/env python3
"""Run one label-free, fixed-geometry single-solvent 3D-RISM numerical pilot."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # pyright: ignore[reportImplicitRelativeImport]
import route1_custom_solvent_asset_audit as asset_audit  # pyright: ignore[reportImplicitRelativeImport]


PROTOCOL_ID = "maple-route1-3drism-single-solvent-pilot-v1"
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
REQUIRED_EXECUTABLES = ("parmchk2", "tleap", "rism3d.snglpnt")
REQUIRED_ROUTE_BOUNDARY = {
    "gas_phase_mm_energy_reported": False,
    "bonded_mm_energy_reported": False,
    "experimental_solvation_labels_loaded": False,
    "experimental_residual_fit": False,
    "energy_calibration": False,
    "endpoint_selection_from_labels": False,
    "mlip_retraining": False,
    "product_runtime_change": False,
    "solute_charge_source": "frozen AM1-BCC charges",
}


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} is not a safe relative path.")
    return Path(path.as_posix())


def _safe_benchmark_path(value: object, field: str) -> Path:
    relative = _safe_relative_path(value, field)
    resolved = (SCRIPT_DIR / relative).resolve()
    try:
        resolved.relative_to(SCRIPT_DIR)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the benchmark directory.") from exc
    return resolved


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive number.")
    return number


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = core.load_json(path)
    if not isinstance(protocol, dict):
        raise ValueError("3D-RISM pilot protocol must be a JSON object.")
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected 3D-RISM pilot protocol.")
    if protocol.get("status") != "research_only_not_accuracy_validated":
        raise ValueError("3D-RISM pilot protocol must remain research-only.")
    if protocol.get("route1_boundary") != REQUIRED_ROUTE_BOUNDARY:
        raise ValueError("3D-RISM pilot protocol violates the Route 1 boundary.")

    solute = protocol.get("solute")
    if not isinstance(solute, dict):
        raise ValueError("3D-RISM pilot solute definition is missing.")
    _safe_benchmark_path(solute.get("mol2"), "solute.mol2")
    if (
        not _is_digest(solute.get("mol2_sha256"))
        or solute.get("topology_is_solute_solvent_interaction_input_only") is not True
    ):
        raise ValueError("3D-RISM pilot solute provenance is incomplete.")

    solvent = protocol.get("solvent_asset")
    if not isinstance(solvent, dict):
        raise ValueError("3D-RISM pilot solvent asset definition is missing.")
    _safe_benchmark_path(solvent.get("asset_contract"), "solvent_asset.asset_contract")
    if not _is_digest(solvent.get("asset_contract_sha256")):
        raise ValueError("3D-RISM pilot asset-contract digest is invalid.")
    _safe_relative_path(
        solvent.get("amber_data_subdirectory"),
        "solvent_asset.amber_data_subdirectory",
    )
    files = solvent.get("files")
    base_files = {"mdl", "rism1d_input", "xvv"}
    generation_evidence_files = {"rism1d_stdout", "generation_evidence"}
    if not isinstance(files, dict) or set(files) not in (
        base_files,
        base_files | generation_evidence_files,
    ):
        raise ValueError("3D-RISM pilot solvent file set is incomplete.")
    for name, entry in files.items():
        if not isinstance(entry, dict):
            raise ValueError(f"solvent_asset.files.{name} must be an object.")
        filename = _safe_relative_path(
            entry.get("name"), f"solvent_asset.files.{name}.name"
        )
        if len(filename.parts) != 1 or not _is_digest(entry.get("sha256")):
            raise ValueError(f"solvent_asset.files.{name} is invalid.")
    manifest = solvent.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("3D-RISM pilot embedded asset manifest is missing.")
    manifest_assets = manifest.get("assets")
    if not isinstance(manifest_assets, dict) or set(manifest_assets) != set(files):
        raise ValueError("Embedded asset manifest file set is incomplete.")
    for name, entry in files.items():
        manifest_entry = manifest_assets[name]
        if not isinstance(manifest_entry, dict) or manifest_entry != {
            "relative_path": entry["name"],
            "sha256": entry["sha256"],
        }:
            raise ValueError("Embedded asset manifest does not match pinned files.")

    solver = protocol.get("solver")
    if not isinstance(solver, dict) or (
        solver.get("executable") != "rism3d.snglpnt"
        or solver.get("topology_executables") != ["parmchk2", "tleap"]
        or solver.get("closure_3d") != "kh"
        or solver.get("primary_quantity") != "rism_excessChemicalPotential"
        or solver.get("primary_convention") != "raw_kh_excess_chemical_potential"
        or solver.get("pc_plus_is_primary") is not False
        or solver.get("standard_state_conversion_applied") is not False
        or solver.get("forces_requested") is not False
        or solver.get("solute_potential_energy_reported") is not False
    ):
        raise ValueError("3D-RISM pilot solver profile is invalid.")
    if type(solver.get("maximum_steps")) is not int or solver["maximum_steps"] <= 0:
        raise ValueError("3D-RISM pilot maximum_steps must be a positive integer.")

    cases = protocol.get("cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("3D-RISM pilot requires a reference and sensitivity cases.")
    identifiers: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object.")
        identifier = case.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("3D-RISM pilot case identifiers must be unique.")
        identifiers.add(identifier)
        _positive_number(case.get("grid_spacing_angstrom"), f"cases[{index}].grid")
        _positive_number(case.get("buffer_angstrom"), f"cases[{index}].buffer")
        _positive_number(case.get("tolerance"), f"cases[{index}].tolerance")
        maximum = case.get("maximum_primary_difference_kcal_mol")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum))
            or float(maximum) < 0.0
        ):
            raise ValueError("3D-RISM pilot comparison threshold is invalid.")
    admission = protocol.get("admission")
    if not isinstance(admission, dict) or admission.get("reference_case") not in identifiers:
        raise ValueError("3D-RISM pilot reference case is invalid.")
    if not isinstance(admission.get("next_gate"), str) or not admission["next_gate"]:
        raise ValueError("3D-RISM pilot next gate must be explicit.")
    reference = next(case for case in cases if case["id"] == admission["reference_case"])
    if (
        reference.get("comparison_to_reference") != "self"
        or float(reference["maximum_primary_difference_kcal_mol"]) != 0.0
        or admission.get("all_cases_must_converge") is not True
        or admission.get("all_case_thresholds_must_pass") is not True
        or admission.get("asset_audit_status_required")
        != "physical_asset_consistent_not_accuracy_validated"
    ):
        raise ValueError("3D-RISM pilot admission gate is invalid.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    label: str,
) -> tuple[str, str, float]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{label} exceeded {timeout_seconds:.1f} seconds.") from exc
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}; "
            f"stdout_tail={completed.stdout[-2000:]!r}; "
            f"stderr_tail={completed.stderr[-2000:]!r}"
        )
    return completed.stdout, completed.stderr, elapsed


def _single_match(pattern: str, text: str, field: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {field}, observed {len(matches)}.")
    return matches[0]


def _float_line(text: str, label: str) -> float:
    match = _single_match(
        rf"^{re.escape(label)}\s+(?P<value>{FLOAT_PATTERN})(?:\s|$)",
        text,
        label,
    )
    value = float(match.group("value"))
    if not math.isfinite(value):
        raise ValueError(f"{label} is non-finite.")
    return value


def parse_rism_output(text: str) -> dict[str, Any]:
    """Parse only the RISM quantities admitted by the pilot protocol."""
    convergence = _single_match(
        r"^\|RXRISM converged in\s+(?P<steps>\d+)\s+steps\s*$",
        text,
        "RXRISM convergence record",
    )
    grid = _single_match(
        r"^\|grid size:\s+(?P<x>\d+)\s+X\s+(?P<y>\d+)\s+X\s+(?P<z>\d+)\s*$",
        text,
        "3D grid size",
    )
    spacing = _single_match(
        rf"^\|grid spacing \[A\]:\s+(?P<x>{FLOAT_PATTERN})\s+X\s+"
        rf"(?P<y>{FLOAT_PATTERN})\s+X\s+(?P<z>{FLOAT_PATTERN})\s*$",
        text,
        "3D grid spacing",
    )
    effective_buffer = _single_match(
        rf"^\|effective buffer \[A\]:\s+(?P<x>{FLOAT_PATTERN}),\s+"
        rf"(?P<y>{FLOAT_PATTERN}),\s+(?P<z>{FLOAT_PATTERN})\s*$",
        text,
        "effective buffer",
    )
    return {
        "converged": True,
        "iterations": int(convergence.group("steps")),
        "grid_points_xyz": [
            int(grid.group("x")),
            int(grid.group("y")),
            int(grid.group("z")),
        ],
        "actual_grid_spacing_angstrom_xyz": [
            float(spacing.group(axis)) for axis in ("x", "y", "z")
        ],
        "effective_buffer_angstrom_xyz": [
            float(effective_buffer.group(axis)) for axis in ("x", "y", "z")
        ],
        "raw_excess_chemical_potential_kcal_mol": _float_line(
            text, "rism_excessChemicalPotential"
        ),
        "gaussian_fluctuation_excess_chemical_potential_kcal_mol": _float_line(
            text, "rism_excessChemicalPotentialGF"
        ),
        "pc_plus_excess_chemical_potential_kcal_mol": _float_line(
            text, "rism_excessChemicalPotentialPCPLUS"
        ),
        "partial_molar_volume_angstrom3": _float_line(
            text, "rism_partialMolarVolume"
        ),
    }


def _resolve_amber_root(path: str | Path) -> tuple[Path, dict[str, Path]]:
    root = Path(path).expanduser().resolve()
    executables = {name: root / "bin" / name for name in REQUIRED_EXECUTABLES}
    missing = [name for name, executable in executables.items() if not executable.is_file()]
    if missing:
        raise FileNotFoundError(f"AmberTools root is missing executables: {missing}.")
    return root, executables


def _prepare_topology(
    *,
    work: Path,
    source_mol2: Path,
    executables: dict[str, Path],
    timeout_seconds: float,
) -> dict[str, dict[str, str]]:
    mol2 = work / "molecule.mol2"
    shutil.copyfile(source_mol2, mol2)
    _run(
        [
            str(executables["parmchk2"]),
            "-i",
            mol2.name,
            "-f",
            "mol2",
            "-o",
            "molecule.frcmod",
            "-s",
            "gaff2",
        ],
        cwd=work,
        timeout_seconds=timeout_seconds,
        label="parmchk2",
    )
    leap_input = work / "tleap.in"
    leap_input.write_text(
        "\n".join(
            [
                "source leaprc.gaff2",
                "set default PBradii mbondi2",
                "MOL = loadmol2 molecule.mol2",
                "loadamberparams molecule.frcmod",
                "saveamberparm MOL molecule.prmtop molecule.inpcrd",
                "savepdb MOL molecule.pdb",
                "quit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _run(
        [str(executables["tleap"]), "-f", leap_input.name],
        cwd=work,
        timeout_seconds=timeout_seconds,
        label="tleap",
    )
    outputs = ("molecule.mol2", "molecule.frcmod", "molecule.prmtop", "molecule.inpcrd", "molecule.pdb")
    missing = [name for name in outputs if not (work / name).is_file()]
    if missing:
        raise RuntimeError(f"Topology preparation did not create: {missing}.")
    return {name: _generated_file_hashes(work / name) for name in outputs}


def _generated_file_hashes(path: Path) -> dict[str, str]:
    """Return byte and reproducibility hashes for one generated topology file.

    Amber ``tleap`` writes the wall-clock date into the first ``%VERSION`` line
    of a prmtop.  That header is not part of the molecular topology and makes a
    raw file digest differ across otherwise identical runs.  Retain that raw
    digest as provenance, but use a second explicitly normalized digest for
    reproducibility comparisons.
    """

    payload = path.read_bytes()
    normalization = "none"
    reproducibility_payload = payload
    if path.suffix.lower() == ".prmtop":
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Generated Amber prmtop must be ASCII text.") from exc
        pattern = re.compile(
            r"^(%VERSION[^\r\n]*?\bDATE\s*=\s*)"
            r"\d{2}/\d{2}/\d{2}[ \t]+\d{2}:\d{2}:\d{2}[ \t]*(\r?)$",
            flags=re.MULTILINE,
        )
        normalized, count = pattern.subn(
            lambda match: f"{match.group(1)}<NORMALIZED>{match.group(2)}",
            text,
        )
        if count != 1:
            raise ValueError(
                "Generated Amber prmtop must contain exactly one dated "
                "%VERSION header."
            )
        reproducibility_payload = normalized.encode("ascii")
        normalization = "amber_prmtop_version_date_header_v1"
    return {
        "byte_sha256": core.sha256_bytes(payload),
        "reproducibility_sha256": core.sha256_bytes(reproducibility_payload),
        "normalization": normalization,
    }


def run_pilot(
    *,
    protocol_path: Path,
    amber_root_path: Path,
    solvent_data_root_path: Path | None,
    output_path: Path,
    timeout_seconds: float,
    recorded_date: str,
) -> dict[str, Any]:
    protocol, protocol_fingerprint = load_protocol(protocol_path)
    amber_root, executables = _resolve_amber_root(amber_root_path)

    source = protocol["solute"]
    source_mol2 = _safe_benchmark_path(source["mol2"], "solute.mol2")
    if not source_mol2.is_file() or core.sha256_file(source_mol2) != source["mol2_sha256"]:
        raise ValueError("Pinned pilot solute MOL2 changed or is missing.")

    solvent = protocol["solvent_asset"]
    asset_contract_path = _safe_benchmark_path(
        solvent["asset_contract"], "solvent_asset.asset_contract"
    )
    if (
        not asset_contract_path.is_file()
        or core.sha256_file(asset_contract_path) != solvent["asset_contract_sha256"]
    ):
        raise ValueError("Pinned custom-solvent asset contract changed or is missing.")
    asset_contract, asset_contract_fingerprint = asset_audit.load_contract(
        asset_contract_path
    )
    if solvent_data_root_path is None:
        solvent_data = (
            amber_root / "dat" / _safe_relative_path(
                solvent["amber_data_subdirectory"],
                "solvent_asset.amber_data_subdirectory",
            )
        ).resolve()
        try:
            solvent_data.relative_to((amber_root / "dat").resolve())
        except ValueError as exc:
            raise ValueError("Amber solvent data directory escapes AmberTools dat.") from exc
    else:
        solvent_data = solvent_data_root_path.resolve()
        if not solvent_data.is_dir():
            raise FileNotFoundError(
                f"Explicit solvent asset directory does not exist: {solvent_data}"
            )

    with tempfile.TemporaryDirectory(prefix="maple-route1-3drism-pilot-") as temporary:
        work = Path(temporary)
        asset_work = work / "solvent"
        asset_work.mkdir()
        for entry in solvent["files"].values():
            source_asset = solvent_data / entry["name"]
            if (
                not source_asset.is_file()
                or core.sha256_file(source_asset) != entry["sha256"]
            ):
                raise ValueError(
                    f"Pinned solvent asset changed or is missing: {entry['name']}."
                )
            shutil.copyfile(source_asset, asset_work / entry["name"])
        manifest_path = asset_work / "manifest.json"
        core.write_json_atomic(manifest_path, copy.deepcopy(solvent["manifest"]))
        solvent_asset_audit = asset_audit.audit_manifest(
            asset_contract,
            asset_contract_fingerprint,
            manifest_path,
        )
        required_asset_status = protocol["admission"]["asset_audit_status_required"]
        if solvent_asset_audit["conclusion"]["status"] != required_asset_status:
            raise ValueError("Solvent asset did not pass the required physical-input audit.")

        topology_hashes = _prepare_topology(
            work=work,
            source_mol2=source_mol2,
            executables=executables,
            timeout_seconds=timeout_seconds,
        )
        records: list[dict[str, Any]] = []
        xvv_path = asset_work / solvent["files"]["xvv"]["name"]
        for case in protocol["cases"]:
            grid = float(case["grid_spacing_angstrom"])
            command = [
                str(executables["rism3d.snglpnt"]),
                "--pdb",
                "molecule.pdb",
                "--prmtop",
                "molecule.prmtop",
                "--rst",
                "molecule.inpcrd",
                "--xvv",
                str(xvv_path),
                "--closure",
                protocol["solver"]["closure_3d"],
                "--buffer",
                str(case["buffer_angstrom"]),
                "--grdspc",
                f"{grid},{grid},{grid}",
                "--tolerance",
                str(case["tolerance"]),
                "--maxstep",
                str(protocol["solver"]["maximum_steps"]),
                "--verbose",
                "1",
                "--gf",
                "--pc+",
            ]
            stdout, stderr, elapsed = _run(
                command,
                cwd=work,
                timeout_seconds=timeout_seconds,
                label=f"rism3d:{case['id']}",
            )
            parsed = parse_rism_output(stdout)
            records.append(
                {
                    "id": case["id"],
                    "requested": {
                        "grid_spacing_angstrom": grid,
                        "buffer_angstrom": float(case["buffer_angstrom"]),
                        "tolerance": float(case["tolerance"]),
                    },
                    "comparison_to_reference": case["comparison_to_reference"],
                    "maximum_primary_difference_kcal_mol": float(
                        case["maximum_primary_difference_kcal_mol"]
                    ),
                    "solver": parsed,
                    "elapsed_seconds": elapsed,
                    "stderr_empty": not stderr.strip(),
                }
            )

        reference_id = protocol["admission"]["reference_case"]
        reference = next(record for record in records if record["id"] == reference_id)
        reference_energy = reference["solver"][
            "raw_excess_chemical_potential_kcal_mol"
        ]
        for record in records:
            difference = abs(
                record["solver"]["raw_excess_chemical_potential_kcal_mol"]
                - reference_energy
            )
            record["primary_difference_from_reference_kcal_mol"] = difference
            record["primary_difference_gate_passed"] = (
                difference <= record["maximum_primary_difference_kcal_mol"]
            )

        all_converged = all(record["solver"]["converged"] for record in records)
        all_thresholds = all(
            record["primary_difference_gate_passed"] for record in records
        )
        passed = all_converged and all_thresholds
        status_key = "result_if_passed" if passed else "result_if_failed"
        artifact: dict[str, Any] = {
            "artifact_type": "route1-3drism-single-solvent-pilot-v1",
            "recorded_date": recorded_date,
            "protocol_id": protocol["protocol_id"],
            "protocol_fingerprint": protocol_fingerprint,
            "claim_scope": protocol["claim_scope"],
            "route1_boundary": protocol["route1_boundary"],
            "solute": {
                "compound_id": source["compound_id"],
                "mol2": source["mol2"],
                "mol2_sha256": source["mol2_sha256"],
                "topology_model": source["topology_model"],
                "generated_topology_hashes": topology_hashes,
            },
            "solvent_asset_audit": solvent_asset_audit,
            "solver": protocol["solver"],
            "executable_sha256": {
                name: core.sha256_file(path) for name, path in executables.items()
            },
            "cases": records,
            "admission": {
                "all_cases_converged": all_converged,
                "all_case_thresholds_passed": all_thresholds,
                "passed": passed,
                "status": protocol["admission"][status_key],
                "accuracy_claim": "none",
                "runtime_provider_enabled": False,
                "next_gate": protocol["admission"]["next_gate"],
            },
            "prohibited_interpretations": protocol["prohibited_interpretations"],
            "command_provenance": core.command_provenance(
                __file__,
                {
                    "protocol": str(protocol_path),
                    "amber_root": str(amber_root),
                    "solvent_data_root": str(solvent_data),
                    "output": str(output_path),
                    "timeout_seconds": timeout_seconds,
                    "recorded_date": recorded_date,
                },
                repository_root=REPOSITORY_ROOT,
            ),
        }
        core.seal_artifact(artifact)
        core.write_json_atomic(output_path, artifact)
        return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amber-root", type=Path, required=True)
    parser.add_argument(
        "--solvent-data-root",
        type=Path,
        help=(
            "Optional explicit directory containing the pinned MDL, 1D input, "
            "and XVV. If omitted, use amber_root/dat/<protocol subdirectory>."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--recorded-date", default="2026-07-29")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be a finite positive number.")
    run_pilot(
        protocol_path=args.protocol,
        amber_root_path=args.amber_root,
        solvent_data_root_path=args.solvent_data_root,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        recorded_date=args.recorded_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
