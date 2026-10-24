"""Exact Cartesian AM1 starting geometry, then native BCC without another QM run.

Only newly inferred Cartesian topology uses this boundary. AM1 optimization and
the installed Antechamber default namelist are retained. Original MLIP geometry,
SQM's serialized optimized geometry and BCC's AC geometry are different records.
Raw tool outputs are never rewritten; the final original-geometry MOL2 is a new
MAPLE preparation artifact, not an alleged unmodified AmberTools output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from .ambertools import ResolvedAmberTools, file_sha256, run_ambertools

# Upstream wac()/rac() serializes coordinates to .001 A. This is a bound on
# downstream BCC serialization ONLY, not a charge-calculation geometry tolerance.
AC_SERIALIZATION_HALF_WIDTH_ANGSTROM = 5.0e-4


def _geometry(elements, positions, representation: str) -> dict[str, Any]:
    record = {
        "elements": list(elements),
        "positions_angstrom": np.asarray(positions).tolist(),
        "representation": representation,
    }
    record["sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def _default_sqm_header(template: Path) -> str:
    """Retain and check the installed default, rather than inventing QM settings."""
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    end = next((i for i, line in enumerate(lines) if line.strip() == "/"), None)
    if end is None:
        raise ValueError("Antechamber SQM template has no namelist terminator.")
    header = "".join(lines[: end + 1])
    parameters = dict(re.findall(r"(\w+)\s*=\s*([^,\s]+)", header.lower()))
    # This contract is qualified against AmberTools' ordinary AM1-BCC protocol.
    # A different installed default is a method decision, not a hidden fallback.
    expected = {
        "qm_theory": "'am1'",
        "grms_tol": "0.0005",
        "scfconv": "1.d-10",
        "ndiis_attempts": "700",
        "qmcharge": "0",
    }
    if parameters != expected:
        raise ValueError(
            f"Unsupported AmberTools default AM1 protocol: {parameters!r}; "
            "no alternative charge method or optimization settings were used."
        )
    return header


def _completed_sqm_geometry(output: Path, atoms) -> dict[str, Any]:
    text = output.read_text(encoding="utf-8")
    if (
        "Calculation Completed" not in text
        or "geometry converged !" not in text
        or text.count("Final Structure") != 1
    ):
        raise ValueError("Native SQM did not complete a converged AM1 optimization.")
    before, final = text.split("Final Structure")
    charge_blocks = before.split("Mulliken Charge\n")
    if len(charge_blocks) != 2:
        raise ValueError("SQM must contain exactly one final Mulliken charge block.")
    rows = charge_blocks[1].split("Total Mulliken Charge")[0].strip().splitlines()
    charge_rows = [line.split() for line in rows]
    symbols = atoms.get_chemical_symbols()
    if (
        len(charge_rows) != len(atoms)
        or any(len(row) != 3 for row in charge_rows)
        or [int(row[0]) for row in charge_rows] != list(range(1, len(atoms) + 1))
        or [row[1] for row in charge_rows] != symbols
        or not np.isfinite([float(row[2]) for row in charge_rows]).all()
    ):
        raise ValueError("SQM Mulliken atom count/order/elements/charges are invalid.")
    coordinate_rows = [
        line.split()
        for line in final.splitlines()
        if re.match(r"\s*QMMM:\s+\d+\s+\d+\s", line)
    ]
    if (
        len(coordinate_rows) != len(atoms)
        or any(len(row) != 7 for row in coordinate_rows)
        or [int(row[1]) for row in coordinate_rows] != list(range(1, len(atoms) + 1))
        or [int(row[2]) for row in coordinate_rows] != list(range(1, len(atoms) + 1))
        or [row[3] for row in coordinate_rows] != symbols
    ):
        raise ValueError("SQM final geometry changed atom count/order/elements.")
    positions = np.asarray(
        [[float(value) for value in row[4:]] for row in coordinate_rows]
    )
    if not np.isfinite(positions).all():
        raise ValueError("SQM final geometry is non-finite.")
    return _geometry(
        symbols,
        positions,
        "SQM final-structure text tokens; not full internal optimizer precision",
    )


def _ac_geometry(path: Path, atoms) -> dict[str, Any]:
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    ]
    metadata = atoms.info["mol2"]
    if (
        len(rows) != len(atoms)
        or any(len(row) < 10 for row in rows)
        or [int(row[1]) for row in rows] != metadata["atom_ids"]
        or [row[2] for row in rows] != metadata["atom_names"]
    ):
        raise ValueError("BCC AC serialization changed atom IDs/names/count/order.")
    if any(
        not re.fullmatch(r"[+-]?\d+\.\d{3}", token)
        for row in rows
        for token in row[5:8]
    ):
        raise ValueError(
            "BCC AC coordinates do not use upstream .001-A serialization tokens."
        )
    positions = np.asarray([[float(value) for value in row[5:8]] for row in rows])
    slack = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(atoms.positions)) * 8
    if not np.isfinite(positions).all() or np.any(
        np.abs(positions - atoms.positions)
        > AC_SERIALIZATION_HALF_WIDTH_ANGSTROM + slack
    ):
        raise ValueError("BCC AC geometry exceeded upstream .001-A serialization.")
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        **_geometry(
            atoms.get_chemical_symbols(), positions, "upstream AC .001-A serialization"
        ),
    }


def run_cartesian_sqm_bcc(
    atoms,
    method: str,
    typed_input: Path,
    audit_dir: Path,
    tool: ResolvedAmberTools,
    installation: dict[str, Any],
    timeout: float,
) -> tuple[Path, list[str], np.ndarray, dict[str, Any]]:
    """Return raw BCC output, command, AC reference positions and honest audit.

    A private fresh working directory excludes stale SQM/AC artifacts. The
    Antechamber sqmout branch reads completed Mulliken charges, while -ao crd
    supplies original geometry for standard BCC connectivity/equivalence rules.
    """
    work = Path(tempfile.mkdtemp(prefix="native-charge-", dir=audit_dir))
    topology_hash = file_sha256(typed_input)
    common = [
        "-nc",
        "0",
        "-m",
        "1",
        "-at",
        "gaff2",
        "-an",
        "n",
        "-du",
        "n",
        "-seq",
        "n",
    ]
    template = work / "sqm.template.in"
    run_ambertools(
        [
            tool.executable,
            "-i",
            str(typed_input),
            "-fi",
            "mol2",
            "-o",
            str(template),
            "-fo",
            "sqmcrt",
            "-j",
            "0",
            *common,
            "-pf",
            "y",
        ],
        installation=tool,
        audit_dir=audit_dir,
        stem="sqm-template",
        timeout=timeout,
        provenance=installation,
        working_directory=work,
    )
    header = _default_sqm_header(template)
    exact_input = work / "sqm-exact.in"
    rows = [
        f"{z} {name} " + " ".join(format(float(value), ".17g") for value in position)
        for z, name, position in zip(
            atoms.numbers, atoms.info["mol2"]["atom_names"], atoms.positions
        )
    ]
    # Native getsqmx() uses an 80-character line and an 8-character atom label.
    # Do not claim exact input if the Fortran reader would truncate a row.
    if any(len(row) > 80 or len(row.split()[1]) > 8 for row in rows):
        raise ValueError(
            "Exact Cartesian SQM row exceeds native 80-character/8-character-name input limits."
        )
    exact_input.write_text(header + "\n".join(rows) + "\n\n", encoding="utf-8")
    roundtrip = np.asarray(
        [[float(value) for value in row.split()[2:]] for row in rows]
    )
    if roundtrip.tobytes() != atoms.positions.tobytes():
        raise ValueError("Exact SQM input failed original-coordinate round-trip.")
    input_hash = file_sha256(exact_input)
    exact_output = work / "sqm-exact.out"
    sqm_executable = Path(tool.executable).parent / "sqm"
    sqm_identity = {
        "path": str(sqm_executable),
        "sha256": file_sha256(sqm_executable) if sqm_executable.is_file() else None,
    }
    sqm_command = [
        str(sqm_executable),
        "-O",
        "-i",
        str(exact_input),
        "-o",
        str(exact_output),
    ]
    run_ambertools(
        sqm_command,
        installation=tool,
        audit_dir=audit_dir,
        stem="sqm",
        timeout=timeout,
        provenance={
            **installation,
            "native_sqm": sqm_identity,
            "input_sha256": input_hash,
        },
        working_directory=work,
    )
    optimized_geometry = _completed_sqm_geometry(exact_output, atoms)
    if file_sha256(exact_input) != input_hash:
        raise ValueError("Native SQM modified its exact original-coordinate input.")
    output_hash = file_sha256(exact_output)
    output = work / f"charges-{method}.mol2"
    command = [
        tool.executable,
        "-i",
        str(exact_output),
        "-fi",
        "sqmout",
        "-a",
        str(typed_input),
        "-fa",
        "mol2",
        "-ao",
        "crd",
        "-o",
        str(output),
        "-fo",
        "mol2",
        "-c",
        {"am1bcc": "bcc", "abcg2": "abcg2"}[method],
        *common,
        "-pf",
        "n",
    ]
    run_ambertools(
        command,
        installation=tool,
        audit_dir=audit_dir,
        stem=f"charges-{method}",
        timeout=timeout,
        provenance={**installation, "completed_sqm_sha256": output_hash},
        working_directory=work,
    )
    if (
        (work / "sqm.in").exists()
        or (work / "sqm.out").exists()
        or file_sha256(exact_input) != input_hash
        or file_sha256(exact_output) != output_hash
        or file_sha256(typed_input) != topology_hash
    ):
        raise ValueError(
            "BCC regenerated SQM or modified the completed exact-coordinate QM/topology artifacts."
        )
    serialized = [_ac_geometry(path, atoms) for path in sorted(work.glob("*.AC*"))]
    precharge_ac = work / "ANTECHAMBER_AM1BCC_PRE.AC"
    reference = next(
        (item for item in serialized if item["path"] == str(precharge_ac)), None
    )
    if reference is None:
        raise ValueError("BCC did not preserve its raw precharge AC geometry audit.")
    reference_positions = np.asarray(reference["positions_angstrom"], dtype=np.float64)
    provenance = {
        "contract": "exact-Cartesian-native-AM1-optimization-then-sqmout-BCC-v1",
        "sqm_template": str(template),
        "sqm_template_sha256": file_sha256(template),
        "sqm_default_namelist": header,
        "native_sqm": sqm_identity,
        "sqm_input": str(exact_input),
        "sqm_input_sha256": input_hash,
        "sqm_output": str(exact_output),
        "sqm_output_sha256": output_hash,
        "sqm_command": sqm_command,
        "sqm_execution_count": 1,
        "second_sqm_generation_detected": False,
        "sqm_starting_geometry": _geometry(
            atoms.get_chemical_symbols(),
            atoms.positions,
            "exact original float64 Cartesian input",
        ),
        "charge_evaluation_geometry": "AM1-optimized-not-original-MLIP-geometry",
        "optimized_geometry": optimized_geometry,
        "bcc_coordinate_source": "original-geometry additional MOL2; native downstream AC serialization",
        "bcc_serialized_geometries": serialized,
        "source_to_bcc_serialized_maximum_error_angstrom": float(
            np.max(np.abs(reference_positions - atoms.positions))
        ),
        "ac_serialization_half_width_angstrom": AC_SERIALIZATION_HALF_WIDTH_ANGSTROM,
        "geometry_keep_scope": "final MLIP/solvent geometry only; AM1 optimization remains enabled",
        "bcc_output_mol2": str(output),
        "bcc_output_mol2_sha256": file_sha256(output),
    }
    (audit_dir / "cartesian-sqm.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return output, command, reference_positions, provenance
