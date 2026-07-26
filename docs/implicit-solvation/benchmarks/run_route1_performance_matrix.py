#!/usr/bin/env python3
"""Run or assemble the fair Route 1 multi-MLIP SP performance matrix."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import os
import platform as host_platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core import (  # pyright: ignore[reportImplicitRelativeImport]
    artifact_content_sha256,
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from run_route1_performance import (  # pyright: ignore[reportImplicitRelativeImport]
    load_charge_vector,
    timing_summary,
)

EXPECTED_SIZE_BINS = {"small", "medium", "large"}
EXPECTED_MODELS = {"maceoff23m", "aimnet2", "ani2x"}
HEX = frozenset("0123456789abcdef")
WARM_TIMING_KEYS = {
    "gas_mlip_for_overhead",
    "route1_combined_for_overhead",
    "route1_combined_for_mm",
    "route1_correction_only",
    "named_mm_obc2_ace",
}
CONSTRUCTION_COLD_TIMING_KEYS = {
    "gas_mlip_for_overhead",
    "route1_combined_for_overhead",
    "route1_combined_for_mm",
    "named_mm_obc2_ace",
}


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and set(text) <= HEX


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return cast(Mapping[str, Any], value)


def _as_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a JSON number.")
    return float(value)


def _as_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a JSON integer.")
    return value


def _portable_artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_artifact_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _exact_json_equal(first: object, second: object) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


def load_and_validate_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Performance protocol must be a schema-version-1 object.")
    if protocol.get("protocol_id") != "maple-route1-fair-sp-performance-matrix-v1":
        raise ValueError("Unexpected Route 1 performance protocol identifier.")

    matrix = _require_mapping(protocol.get("matrix"), "Protocol matrix")
    models = matrix.get("models")
    if (
        not isinstance(models, list)
        or len(models) != len(set(models))
        or set(models) != EXPECTED_MODELS
    ):
        raise ValueError("Protocol must declare the exact three-MLIP model set.")

    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("Protocol must declare exactly three molecule-size cases.")
    case_ids: set[str] = set()
    size_bins: set[str] = set()
    for raw_case in cases:
        case = _require_mapping(raw_case, "Protocol matrix case")
        case_id = str(case.get("case_id", ""))
        size_bin = str(case.get("size_bin", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(f"Invalid or duplicate performance case: {case_id!r}.")
        if size_bin not in EXPECTED_SIZE_BINS or size_bin in size_bins:
            raise ValueError(f"Invalid or duplicate molecule-size bin: {size_bin!r}.")
        atom_count = _as_int(
            case.get("atom_count"), f"matrix case {case_id}.atom_count"
        )
        if atom_count <= 0:
            raise ValueError(f"Case {case_id} requires a positive atom count.")
        heavy = _as_int(
            case.get("heavy_atom_count"), f"matrix case {case_id}.heavy_atom_count"
        )
        if heavy <= 0 or heavy > atom_count:
            raise ValueError(f"Case {case_id} has an invalid heavy-atom count.")
        elements = case.get("elements")
        if (
            not isinstance(elements, list)
            or not elements
            or elements != sorted(set(elements))
        ):
            raise ValueError(f"Case {case_id} requires sorted unique elements.")
        relative = Path(str(case.get("mol2_relative_path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix != ".mol2"
        ):
            raise ValueError(f"Case {case_id} has an unsafe MOL2 relative path.")
        if not _is_sha256(case.get("mol2_sha256")):
            raise ValueError(f"Case {case_id} requires a pinned MOL2 SHA256.")
        case_ids.add(case_id)
        size_bins.add(size_bin)
    if size_bins != EXPECTED_SIZE_BINS:
        raise ValueError("Protocol must cover small, medium, and large exactly once.")

    inputs = _require_mapping(protocol.get("inputs"), "Protocol inputs")
    for key in (
        "charge_manifest_sha256",
        "benchmark_helper_sha256",
    ):
        if not _is_sha256(inputs.get(key)):
            raise ValueError(f"Protocol input {key} requires a pinned SHA256.")
    for key in (
        "input_root_relative_path",
        "charge_manifest_relative_path",
        "benchmark_helper_relative_path",
    ):
        relative = Path(str(inputs.get(key, "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Protocol input {key} requires a safe relative path.")
    input_root = REPOSITORY_ROOT / inputs["input_root_relative_path"]
    charge_manifest = REPOSITORY_ROOT / inputs["charge_manifest_relative_path"]
    benchmark_helper = REPOSITORY_ROOT / inputs["benchmark_helper_relative_path"]
    if (
        not input_root.is_dir()
        or not charge_manifest.is_file()
        or sha256_file(charge_manifest) != inputs["charge_manifest_sha256"]
        or not benchmark_helper.is_file()
        or sha256_file(benchmark_helper) != inputs["benchmark_helper_sha256"]
    ):
        raise ValueError("Protocol input files are unavailable or do not match hashes.")
    for raw_case in cases:
        case = _require_mapping(raw_case, "Protocol matrix case")
        mol2_path = input_root / case["mol2_relative_path"]
        if not mol2_path.is_file() or sha256_file(mol2_path) != case["mol2_sha256"]:
            raise ValueError(
                f"Protocol case {case['case_id']} MOL2 is unavailable or changed."
            )

    timing = _require_mapping(protocol.get("timing"), "Protocol timing")
    for name in ("warm", "construction_cold"):
        record = _require_mapping(timing.get(name), f"timing.{name}")
        if _as_int(record.get("samples"), f"timing.{name}.samples") <= 0:
            raise ValueError(f"timing.{name}.samples must be positive.")
        if _as_int(record.get("warmups"), f"timing.{name}.warmups") < 0:
            raise ValueError(f"timing.{name}.warmups must be non-negative.")
        if not str(record.get("definition", "")).strip():
            raise ValueError(f"timing.{name}.definition must be explicit.")
    construction_cold = _require_mapping(
        timing["construction_cold"], "timing.construction_cold"
    )
    if (
        _as_int(
            construction_cold.get("warmups"),
            "timing.construction_cold.warmups",
        )
        != 0
    ):
        raise ValueError("Construction-cold timing must not perform warmup calls.")

    policy = _require_mapping(
        protocol.get("resource_policy"), "Protocol resource policy"
    )
    if policy.get("device") != "cpu" or policy.get("openmm_platform") != "CPU":
        raise ValueError("Version 1 freezes the fair comparison to CPU/OpenMM CPU.")
    if _as_int(policy.get("torch_threads"), "resource_policy.torch_threads") != 1:
        raise ValueError("Version 1 requires one PyTorch thread.")
    if (
        _as_int(
            policy.get("torch_interop_threads"),
            "resource_policy.torch_interop_threads",
        )
        != 1
    ):
        raise ValueError("Version 1 requires one PyTorch inter-op thread.")
    if (
        _as_int(
            policy.get("maximum_native_threadpool_threads"),
            "resource_policy.maximum_native_threadpool_threads",
        )
        != 1
    ):
        raise ValueError("Version 1 requires single-threaded native pools.")
    maximum_load = _as_float(
        policy.get("maximum_one_minute_load_per_logical_cpu"),
        "resource_policy.maximum_one_minute_load_per_logical_cpu",
    )
    if not np.isfinite(maximum_load) or maximum_load <= 0.0:
        raise ValueError("Version 1 requires a positive finite host-load gate.")
    if policy.get("required_environment") != {
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }:
        raise ValueError("Version 1 requires one MKL, OMP, and OpenBLAS thread.")
    if policy.get("openmm_platform_properties") != {
        "DeterministicForces": "true",
        "Threads": "1",
    }:
        raise ValueError("Version 1 requires deterministic single-thread OpenMM CPU.")

    comparator = _require_mapping(
        protocol.get("mm_comparator"), "Protocol MM comparator"
    )
    if (
        comparator.get("implicit_solvent") != "OBC2"
        or comparator.get("sasa_method") != "ACE"
        or comparator.get("expected_gbsa_force_class") != "GBSAOBCForce"
        or not np.isclose(
            _as_float(
                comparator.get("expected_surface_area_energy_kj_mol_nm2"),
                "mm_comparator.expected_surface_area_energy_kj_mol_nm2",
            ),
            2.25936,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "Version 1 requires explicit OpenMM OBC2/ACE comparator semantics."
        )

    acceptance = _require_mapping(protocol.get("acceptance"), "Protocol acceptance")
    required_cells = len(models) * len(cases)
    if (
        _as_int(
            acceptance.get("required_cell_count"),
            "acceptance.required_cell_count",
        )
        != required_cells
    ):
        raise ValueError(
            "Required cell count does not match the model/case cross product."
        )
    threshold_keys = (
        "maximum_energy_closure_hartree",
        "maximum_force_closure_hartree_per_angstrom",
        "maximum_topology_charge_difference_e",
        "maximum_topology_coordinate_difference_angstrom",
    )
    for key in threshold_keys:
        value = _as_float(acceptance.get(key), f"acceptance.{key}")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Acceptance threshold {key} must be positive and finite.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _find_case(protocol: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    matches = [
        case for case in protocol["matrix"]["cases"] if case["case_id"] == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate performance case {case_id!r}.")
    return dict(matches[0])


def named_mm_comparison_eligibility(
    *,
    device: str,
    torch_threads: int,
    torch_interop_threads: int,
    native_threadpools: Sequence[Mapping[str, Any]],
    maximum_native_threadpool_threads: int,
    required_environment: Mapping[str, str],
    observed_environment: Mapping[str, str | None],
    route1_platform: str,
    route1_properties: Mapping[str, str],
    mm_platform: str,
    mm_properties: Mapping[str, str],
) -> dict[str, Any]:
    reasons: list[str] = []
    if device != "cpu":
        reasons.append("gas MLIP device is not CPU")
    if torch_threads != 1:
        reasons.append("PyTorch is not single-threaded")
    if torch_interop_threads != 1:
        reasons.append("PyTorch inter-op pool is not single-threaded")
    if not native_threadpools:
        reasons.append("native thread pools were not observed")
    for record in native_threadpools:
        if int(record.get("num_threads", 0)) > maximum_native_threadpool_threads:
            reasons.append(
                f"native thread pool {record.get('prefix')!r} uses "
                f"{record.get('num_threads')!r} threads"
            )
    for name, expected in sorted(required_environment.items()):
        if observed_environment.get(name) != expected:
            reasons.append(
                f"{name}={observed_environment.get(name)!r}, expected {expected!r}"
            )
    if route1_platform.lower() != "cpu":
        reasons.append("Route 1 solvent platform is not OpenMM CPU")
    if mm_platform.lower() != route1_platform.lower():
        reasons.append("Route 1 and MM use different OpenMM platforms")
    normalized_route1 = {
        str(key): str(value) for key, value in route1_properties.items()
    }
    normalized_mm = {str(key): str(value) for key, value in mm_properties.items()}
    if normalized_route1 != normalized_mm:
        reasons.append("Route 1 and MM OpenMM platform properties differ")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "same_host": True,
        "same_task": "single-point energy+forces",
        "same_openmm_platform": route1_platform.lower() == mm_platform.lower(),
        "same_openmm_platform_properties": normalized_route1 == normalized_mm,
        "runtime_only_not_accuracy_parity": True,
    }


def _native_threadpool_snapshot(
    stage: str,
    *,
    maximum_threads: int,
) -> dict[str, Any]:
    from threadpoolctl import threadpool_info  # pyright: ignore[reportMissingImports]

    pools = []
    for record in threadpool_info():
        pools.append(
            {
                "user_api": record.get("user_api"),
                "internal_api": record.get("internal_api"),
                "num_threads": int(record.get("num_threads", 0)),
                "prefix": record.get("prefix"),
                "filepath": record.get("filepath"),
                "version": record.get("version"),
                "threading_layer": record.get("threading_layer"),
                "architecture": record.get("architecture"),
            }
        )
    pools.sort(key=lambda record: (str(record["filepath"]), str(record["prefix"])))
    if not pools:
        raise RuntimeError("No native thread pools were observable for timing.")
    oversized = [record for record in pools if record["num_threads"] > maximum_threads]
    if oversized:
        raise RuntimeError(
            f"Timing {stage} found native thread pools above {maximum_threads}: "
            f"{oversized}."
        )
    return {
        "stage": stage,
        "maximum_threads": maximum_threads,
        "pools": pools,
        "passed": True,
    }


def _host_load_snapshot(stage: str, maximum_per_logical_cpu: float) -> dict[str, Any]:
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count < 1:
        raise RuntimeError("Cannot determine logical CPU count for timing isolation.")
    if not hasattr(os, "getloadavg"):
        raise RuntimeError("Host load averages are unavailable for timing isolation.")
    load_1m, load_5m, load_15m = os.getloadavg()
    load_per_logical_cpu = load_1m / logical_cpu_count
    if load_per_logical_cpu > maximum_per_logical_cpu:
        raise RuntimeError(
            f"Timing {stage} host load {load_per_logical_cpu:.6f} per logical CPU "
            f"exceeds the protocol limit {maximum_per_logical_cpu:.6f}."
        )
    return {
        "stage": stage,
        "logical_cpu_count": logical_cpu_count,
        "load_average_1m": load_1m,
        "load_average_5m": load_5m,
        "load_average_15m": load_15m,
        "load_per_logical_cpu": load_per_logical_cpu,
        "maximum_load_per_logical_cpu": maximum_per_logical_cpu,
        "passed": True,
    }


def _host_fingerprint() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "node": host_platform.node(),
        "machine": host_platform.machine(),
        "processor": host_platform.processor(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
    }


def _derive_paired_comparisons(
    timings: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    warm = timings["warm"]
    cold = timings["construction_cold"]
    return {
        "warm": {
            "combined_vs_gas": _paired_comparison(
                warm["gas_mlip_for_overhead"],
                warm["route1_combined_for_overhead"],
                first_name="gas_mlip",
                second_name="route1_combined",
            ),
            "combined_vs_mm": _paired_comparison(
                warm["route1_combined_for_mm"],
                warm["named_mm_obc2_ace"],
                first_name="route1_combined",
                second_name="named_mm_obc2_ace",
            ),
        },
        "construction_cold": {
            "combined_vs_gas": _paired_comparison(
                cold["gas_mlip_for_overhead"],
                cold["route1_combined_for_overhead"],
                first_name="gas_mlip",
                second_name="route1_combined",
            ),
            "combined_vs_mm": _paired_comparison(
                cold["route1_combined_for_mm"],
                cold["named_mm_obc2_ace"],
                first_name="route1_combined",
                second_name="named_mm_obc2_ace",
            ),
        },
    }


def _derive_ratios(
    timings: Mapping[str, Mapping[str, Mapping[str, Any]]],
    paired_comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, float | bool]:
    warm = timings["warm"]
    warm_gas_median = float(warm["gas_mlip_for_overhead"]["median_ms"])
    warm_overhead = paired_comparisons["warm"]["combined_vs_gas"]
    warm_mm = paired_comparisons["warm"]["combined_vs_mm"]
    cold_overhead = paired_comparisons["construction_cold"]["combined_vs_gas"]
    cold_mm = paired_comparisons["construction_cold"]["combined_vs_mm"]
    return {
        "warm_combined_minus_gas_fraction_observed": (
            warm_overhead["median_second_over_first_minus_one"]
        ),
        "warm_correction_over_gas": (
            float(warm["route1_correction_only"]["median_ms"]) / warm_gas_median
        ),
        "warm_route1_over_named_mm": warm_mm["median_first_over_second"],
        "warm_route1_faster_than_named_mm_observed": (
            warm_mm["median_first_over_second"] < 1.0
        ),
        "construction_cold_combined_minus_gas_fraction_observed": (
            cold_overhead["median_second_over_first_minus_one"]
        ),
        "construction_cold_route1_over_named_mm": (cold_mm["median_first_over_second"]),
        "construction_cold_route1_faster_than_named_mm_observed": (
            cold_mm["median_first_over_second"] < 1.0
        ),
    }


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    label: str,
    timeout: int = 180,
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    (cwd / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cwd / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{label} failed with return code {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    return {
        "label": label,
        "command": list(command),
        "returncode": completed.returncode,
    }


def _mol2_atom_names(source_text: str) -> list[str]:
    names: list[str] = []
    section = ""
    for line in source_text.splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[9:].strip().upper()
            continue
        if section == "ATOM" and line.strip():
            fields = line.split()
            if len(fields) < 6:
                raise ValueError("Malformed MOL2 atom row.")
            names.append(fields[1])
    if not names:
        raise ValueError("MOL2 contains no atoms.")
    return names


def _topology_semantic_sha256(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix == ".prmtop":
        lines = payload.splitlines(keepends=True)
        if not lines or not lines[0].startswith(b"%VERSION"):
            raise ValueError("Amber prmtop lacks the expected version header.")
        version_prefix = lines[0].split(b"DATE =", 1)[0].rstrip()
        payload = version_prefix + b"\n" + b"".join(lines[1:])
    return sha256_bytes(payload)


def prepare_named_mm_topology(
    *,
    source_mol2: Path,
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    ambertools_bin: Path,
    work_dir: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from openmm import NonbondedForce, unit
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile, NoCutoff

    from maple.function.calculator.extra_correction.implicit.amber_chagb import (
        render_typed_mol2,
        require_no_frcmod_nonbonded_overrides,
    )

    executables = {
        name: (ambertools_bin / name).resolve() for name in ("parmchk2", "tleap")
    }
    for executable in executables.values():
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(
                f"AmberTools executable is unavailable: {executable}"
            )

    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Named-MM topology work directory is not empty: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    source_text = source_mol2.read_text(encoding="utf-8")
    typed_mol2 = work_dir / "molecule.mol2"
    typed_mol2.write_text(
        render_typed_mol2(source_text, positions_angstrom, charges_e),
        encoding="utf-8",
    )
    topology_protocol = protocol["topology"]
    commands = [
        _run_command(
            [
                str(executables["parmchk2"]),
                "-i",
                typed_mol2.name,
                "-f",
                "mol2",
                "-o",
                "molecule.frcmod",
                "-s",
                str(topology_protocol["parmchk2_mode"]),
            ],
            cwd=work_dir,
            label="parmchk2",
        )
    ]
    frcmod = work_dir / "molecule.frcmod"
    require_no_frcmod_nonbonded_overrides(frcmod.read_text(encoding="utf-8"))
    leap_input = work_dir / "tleap.in"
    leap_input.write_text(
        "\n".join(
            [
                f"source {topology_protocol['tleap_source']}",
                f"set default PBradii {topology_protocol['pb_radii']}",
                f"MOL = loadmol2 {typed_mol2.name}",
                f"loadamberparams {frcmod.name}",
                "saveamberparm MOL molecule.prmtop molecule.inpcrd",
                "quit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    commands.append(
        _run_command(
            [str(executables["tleap"]), "-f", leap_input.name],
            cwd=work_dir,
            label="tleap",
        )
    )

    prmtop_path = work_dir / "molecule.prmtop"
    inpcrd_path = work_dir / "molecule.inpcrd"
    if not prmtop_path.is_file() or not inpcrd_path.is_file():
        raise RuntimeError("tleap did not create molecule.prmtop and molecule.inpcrd.")
    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    expected_names = _mol2_atom_names(source_text)
    observed_atoms = list(prmtop.topology.atoms())
    observed_names = [atom.name for atom in observed_atoms]
    if observed_names != expected_names:
        raise ValueError("tleap changed atom names or atom order.")

    openmm_unit = cast(Any, unit)
    observed_positions = np.asarray(
        cast(Any, inpcrd.positions).value_in_unit(openmm_unit.angstrom),
        dtype=np.float64,
    )
    coordinate_max_abs = float(
        np.max(np.abs(observed_positions - np.asarray(positions_angstrom)))
    )
    acceptance = protocol["acceptance"]
    if coordinate_max_abs > float(
        acceptance["maximum_topology_coordinate_difference_angstrom"]
    ):
        raise ValueError(
            f"tleap changed coordinates by {coordinate_max_abs:.3e} Angstrom."
        )

    gas_system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None)
    nonbonded = next(
        (
            force
            for force in gas_system.getForces()
            if isinstance(force, NonbondedForce)
        ),
        None,
    )
    if nonbonded is None:
        raise ValueError("Prepared GAFF2 topology contains no NonbondedForce.")
    observed_charges = np.asarray(
        [
            nonbonded.getParticleParameters(index)[0].value_in_unit(
                openmm_unit.elementary_charge
            )
            for index in range(nonbonded.getNumParticles())
        ],
        dtype=np.float64,
    )
    charge_max_abs = float(np.max(np.abs(observed_charges - charges_e)))
    if charge_max_abs > float(acceptance["maximum_topology_charge_difference_e"]):
        raise ValueError(
            f"Prepared topology changed AM1-BCC charges by {charge_max_abs:.3e} e."
        )

    files = {}
    for path in (
        typed_mol2,
        frcmod,
        leap_input,
        prmtop_path,
        inpcrd_path,
        work_dir / "parmchk2.stdout.log",
        work_dir / "parmchk2.stderr.log",
        work_dir / "tleap.stdout.log",
        work_dir / "tleap.stderr.log",
    ):
        files[path.name] = {
            "path": _portable_artifact_path(path),
            "sha256": sha256_file(path),
            "semantic_sha256": _topology_semantic_sha256(path),
        }
    leap_log = work_dir / "leap.log"
    if leap_log.is_file():
        files[leap_log.name] = {
            "path": _portable_artifact_path(leap_log),
            "sha256": sha256_file(leap_log),
            "semantic_sha256": _topology_semantic_sha256(leap_log),
        }
    return {
        "prmtop_path": _portable_artifact_path(prmtop_path),
        "inpcrd_path": _portable_artifact_path(inpcrd_path),
        "commands": commands,
        "files": files,
        "executables": {
            name: {
                "path": _portable_artifact_path(path),
                "sha256": sha256_file(path),
            }
            for name, path in executables.items()
        },
        "validation": {
            "atom_count": len(observed_atoms),
            "atom_names_preserved": True,
            "coordinate_max_abs_angstrom": coordinate_max_abs,
            "charge_max_abs_e": charge_max_abs,
            "frcmod_nonbon_overrides": 0,
        },
    }


def _create_mm_context(
    prmtop,
    *,
    platform_name: str,
    properties: Mapping[str, str],
    sasa_method: str,
):
    from openmm import Context, Platform, VerletIntegrator, unit
    from openmm.app import OBC2, NoCutoff

    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
        implicitSolvent=OBC2,
        sasaMethod=sasa_method,
        constraints=None,
    )
    openmm_unit = cast(Any, unit)
    integrator = VerletIntegrator(0.001 * openmm_unit.picoseconds)
    openmm_platform = Platform.getPlatformByName(platform_name)
    requested = {str(key): str(value) for key, value in properties.items()}
    context = Context(system, integrator, openmm_platform, requested)
    observed = {
        name: openmm_platform.getPropertyValue(context, name)
        for name in sorted(requested)
    }
    return context, integrator, observed, system


def _validate_obc2_ace_system(
    system,
    *,
    expected_force_class: str,
    expected_surface_area_energy_kj_mol_nm2: float,
) -> dict[str, Any]:
    from openmm import GBSAOBCForce, unit

    forces = [force for force in system.getForces() if isinstance(force, GBSAOBCForce)]
    if len(forces) != 1:
        raise ValueError(
            "Named MM comparator requires exactly one OpenMM GBSAOBCForce."
        )
    force = forces[0]
    force_class = type(force).__name__
    openmm_unit = cast(Any, unit)
    surface_area_energy = float(
        cast(Any, force.getSurfaceAreaEnergy()).value_in_unit(
            openmm_unit.kilojoule_per_mole / openmm_unit.nanometer**2
        )
    )
    if force_class != expected_force_class or not np.isclose(
        surface_area_energy,
        expected_surface_area_energy_kj_mol_nm2,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Named MM comparator OBC2/ACE implementation does not match the protocol."
        )
    return {
        "implicit_solvent": "OBC2",
        "sasa_method": "ACE",
        "force_class": force_class,
        "force_count": len(forces),
        "surface_area_energy_kj_mol_nm2": surface_area_energy,
    }


def _materialize_mm_energy_forces(state, *, atom_count: int, label: str) -> None:
    from openmm import unit

    openmm_unit = cast(Any, unit)
    energy = float(
        cast(Any, state.getPotentialEnergy()).value_in_unit(
            openmm_unit.kilojoule_per_mole
        )
    )
    forces = np.asarray(
        cast(Any, state.getForces(asNumpy=True)).value_in_unit(
            openmm_unit.kilojoule_per_mole / openmm_unit.nanometer
        ),
        dtype=np.float64,
    )
    if not np.isfinite(energy):
        raise ValueError(f"{label} returned a non-finite energy.")
    if forces.shape != (atom_count, 3) or not np.isfinite(forces).all():
        raise ValueError(
            f"{label} returned invalid forces with shape {forces.shape!r}."
        )


def _checkpoint_record(model: str) -> dict[str, Any]:
    filename = {
        "maceoff23m": "maceoff23m.pt",
        "aimnet2": "aimnet2.pt",
        "ani2x": "ani2x.pt",
    }[model]
    path = (REPOSITORY_ROOT / "maple/function/calculator/model" / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _protocol_conformance(
    *,
    protocol: Mapping[str, Any],
    device: str,
    torch_threads: int,
    torch_interop_threads: int,
    openmm_platform: str,
    warm_samples: int,
    warmups: int,
    cold_samples: int,
) -> dict[str, Any]:
    policy = protocol["resource_policy"]
    timing = protocol["timing"]
    reasons: list[str] = []
    expected = {
        "device": policy["device"],
        "torch_threads": int(policy["torch_threads"]),
        "torch_interop_threads": int(policy["torch_interop_threads"]),
        "openmm_platform": policy["openmm_platform"],
        "warm_samples": int(timing["warm"]["samples"]),
        "warmups": int(timing["warm"]["warmups"]),
        "cold_samples": int(timing["construction_cold"]["samples"]),
    }
    observed = {
        "device": device,
        "torch_threads": torch_threads,
        "torch_interop_threads": torch_interop_threads,
        "openmm_platform": openmm_platform,
        "warm_samples": warm_samples,
        "warmups": warmups,
        "cold_samples": cold_samples,
    }
    for key, value in expected.items():
        if observed[key] != value:
            reasons.append(f"{key}={observed[key]!r}, expected {value!r}")
    for name, value in policy["required_environment"].items():
        if os.environ.get(name) != value:
            reasons.append(f"{name}={os.environ.get(name)!r}, expected {value!r}")
    return {
        "conformant": not reasons,
        "reasons": reasons,
        "expected": expected,
        "observed": observed,
    }


def _positive_timing(summary: Mapping[str, Any]) -> bool:
    try:
        count = _as_int(summary.get("n"), "timing.n")
        values = [
            _as_float(summary.get(key), f"timing.{key}")
            for key in ("median_ms", "p95_ms", "mean_ms")
        ]
    except TypeError:
        return False
    return count > 0 and all(np.isfinite(value) and value > 0.0 for value in values)


def _timing_summary_with_samples(
    milliseconds: Sequence[float],
) -> dict[str, float | int | list[float]]:
    samples = [float(value) for value in milliseconds]
    summary = timing_summary(samples)
    return {**summary, "samples_ms": samples}


def benchmark_calls_with_samples(
    function,
    *,
    samples: int,
    warmups: int,
    synchronize,
) -> dict[str, float | int | list[float]]:
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative.")
    for index in range(warmups):
        function(index)
    synchronize()
    milliseconds = []
    for index in range(samples):
        synchronize()
        started = time.perf_counter()
        function(index)
        synchronize()
        milliseconds.append((time.perf_counter() - started) * 1000.0)
    return _timing_summary_with_samples(milliseconds)


def benchmark_paired_calls_with_samples(
    first,
    second,
    *,
    samples: int,
    warmups: int,
    synchronize,
) -> tuple[
    dict[str, float | int | list[float]],
    dict[str, float | int | list[float]],
]:
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative.")

    def timed_call(function, index: int) -> float:
        synchronize()
        started = time.perf_counter()
        function(index)
        synchronize()
        return (time.perf_counter() - started) * 1000.0

    for index in range(warmups):
        if index % 2:
            second(index)
            first(index)
        else:
            first(index)
            second(index)
    synchronize()

    first_ms = []
    second_ms = []
    for index in range(samples):
        if index % 2:
            second_ms.append(timed_call(second, index))
            first_ms.append(timed_call(first, index))
        else:
            first_ms.append(timed_call(first, index))
            second_ms.append(timed_call(second, index))
    return _timing_summary_with_samples(first_ms), _timing_summary_with_samples(
        second_ms
    )


def benchmark_construction_cold_calls(
    function,
    *,
    samples: int,
    synchronize,
) -> dict[str, float | int | list[float]]:
    """Time construction plus first evaluation, then release outside the timer."""
    if samples < 1:
        raise ValueError("Construction-cold samples must be positive.")
    milliseconds = []
    for index in range(samples):
        synchronize()
        started = time.perf_counter()
        retained = function(index)
        synchronize()
        milliseconds.append((time.perf_counter() - started) * 1000.0)
        del retained
        gc.collect()
    return _timing_summary_with_samples(milliseconds)


def benchmark_paired_construction_cold_calls(
    first,
    second,
    *,
    samples: int,
    synchronize,
) -> tuple[
    dict[str, float | int | list[float]],
    dict[str, float | int | list[float]],
]:
    """Interleave two construction-cold paths without timing object teardown."""
    if samples < 1:
        raise ValueError("Construction-cold samples must be positive.")

    def timed_call(function, index: int) -> float:
        synchronize()
        started = time.perf_counter()
        retained = function(index)
        synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        del retained
        gc.collect()
        return elapsed

    first_ms = []
    second_ms = []
    for index in range(samples):
        if index % 2:
            second_ms.append(timed_call(second, index))
            first_ms.append(timed_call(first, index))
        else:
            first_ms.append(timed_call(first, index))
            second_ms.append(timed_call(second, index))
    return _timing_summary_with_samples(first_ms), _timing_summary_with_samples(
        second_ms
    )


def _paired_comparison(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    first_name: str,
    second_name: str,
) -> dict[str, Any]:
    first_ms = np.asarray(first.get("samples_ms"), dtype=np.float64)
    second_ms = np.asarray(second.get("samples_ms"), dtype=np.float64)
    if (
        first_ms.ndim != 1
        or second_ms.shape != first_ms.shape
        or not len(first_ms)
        or not np.isfinite(first_ms).all()
        or not np.isfinite(second_ms).all()
        or np.any(first_ms <= 0.0)
        or np.any(second_ms <= 0.0)
    ):
        raise ValueError("Paired timing samples must be aligned, positive, and finite.")
    difference = first_ms - second_ms
    first_over_second = first_ms / second_ms
    second_over_first = second_ms / first_ms
    return {
        "first": first_name,
        "second": second_name,
        "execution_order_by_sample": [
            "first_then_second" if index % 2 == 0 else "second_then_first"
            for index in range(len(first_ms))
        ],
        "first_minus_second_ms_by_sample": difference.tolist(),
        "first_over_second_by_sample": first_over_second.tolist(),
        "second_over_first_by_sample": second_over_first.tolist(),
        "median_first_minus_second_ms": float(np.median(difference)),
        "median_first_over_second": float(np.median(first_over_second)),
        "median_second_over_first": float(np.median(second_over_first)),
        "median_second_over_first_minus_one": float(np.median(second_over_first - 1.0)),
    }


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    import torch  # pyright: ignore[reportMissingImports]
    from ase.calculators.calculator import all_changes
    from openmm.app import AmberPrmtopFile

    from maple.function.calculator.extra_correction.implicit.openmm_gb import (
        DEFAULT_CPU_PLATFORM_PROPERTIES,
    )
    from maple.function.calculator.set_calculator import SetCalculator
    from maple.function.read.filereader.mol2_reader import MOL2Reader

    protocol, protocol_hash = load_and_validate_protocol(args.protocol)
    if args.model not in protocol["matrix"]["models"]:
        raise ValueError(f"Model {args.model!r} is not in the frozen matrix.")
    case = _find_case(protocol, args.case_id)
    policy = protocol["resource_policy"]
    device_name = args.device or str(policy["device"])
    openmm_platform = args.openmm_platform or str(policy["openmm_platform"])
    torch_threads = (
        int(args.torch_threads)
        if args.torch_threads is not None
        else int(policy["torch_threads"])
    )
    warm_samples = (
        int(args.warm_samples)
        if args.warm_samples is not None
        else int(protocol["timing"]["warm"]["samples"])
    )
    warmups = (
        int(args.warmups)
        if args.warmups is not None
        else int(protocol["timing"]["warm"]["warmups"])
    )
    cold_samples = (
        int(args.cold_samples)
        if args.cold_samples is not None
        else int(protocol["timing"]["construction_cold"]["samples"])
    )
    if min(warm_samples, cold_samples) < 1 or warmups < 0:
        raise ValueError("Timing samples must be positive and warmups non-negative.")
    maximum_load = float(policy["maximum_one_minute_load_per_logical_cpu"])
    host_load_entry = _host_load_snapshot("entry", maximum_load)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(int(policy["torch_interop_threads"]))
    device = torch.device(device_name)

    input_root = _resolve_artifact_path(protocol["inputs"]["input_root_relative_path"])
    charge_manifest = _resolve_artifact_path(
        protocol["inputs"]["charge_manifest_relative_path"]
    )
    mol2_path = input_root / case["mol2_relative_path"]
    if sha256_file(mol2_path) != case["mol2_sha256"]:
        raise ValueError("Performance-case MOL2 hash does not match the protocol.")
    if sha256_file(charge_manifest) != protocol["inputs"]["charge_manifest_sha256"]:
        raise ValueError("Charge-manifest hash does not match the protocol.")
    helper = REPOSITORY_ROOT / protocol["inputs"]["benchmark_helper_relative_path"]
    if sha256_file(helper) != protocol["inputs"]["benchmark_helper_sha256"]:
        raise ValueError("Pinned benchmark-helper hash has changed.")

    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1)
    if len(atoms) != int(case["atom_count"]):
        raise ValueError("Performance-case atom count does not match the protocol.")
    symbols = atoms.get_chemical_symbols()
    if sorted(set(symbols)) != case["elements"]:
        raise ValueError("Performance-case element set does not match the protocol.")
    if sum(symbol != "H" for symbol in symbols) != int(case["heavy_atom_count"]):
        raise ValueError(
            "Performance-case heavy-atom count does not match the protocol."
        )
    charges, charge_record = load_charge_vector(
        charge_manifest, case["case_id"], atom_count=len(atoms)
    )
    if charge_record.get("source_mol2_sha256") != case["mol2_sha256"]:
        raise ValueError("Charge record is not bound to the selected MOL2.")
    atoms.set_initial_charges(charges)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64).copy()

    work_dir = Path(args.work_dir).resolve() / case["case_id"] / args.model
    topology_dir = work_dir / "topology"
    topology = prepare_named_mm_topology(
        source_mol2=mol2_path,
        positions_angstrom=positions,
        charges_e=charges,
        ambertools_bin=Path(args.ambertools_bin).resolve(),
        work_dir=topology_dir,
        protocol=protocol,
    )
    prmtop = AmberPrmtopFile(str(_resolve_artifact_path(topology["prmtop_path"])))

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def displaced_positions(index: int) -> np.ndarray:
        displaced = positions.copy()
        displaced[0, 0] += (-1.0 if index % 2 == 0 else 1.0) * 1.0e-5
        return displaced

    logs = work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def make_calculator(target_atoms, *, combined: bool, log_name: str):
        kwargs: dict[str, Any] = {}
        if combined:
            target_atoms.set_initial_charges(charges)
            kwargs = {
                "implicit": "gb",
                "solvent": "water",
                "charge_options": {
                    "source": "mol2",
                    "label": "am1bcc-frozen-performance-matrix",
                },
                "solvation_options": {
                    "method": "gb",
                    "model": "obc2",
                    "nonpolar": "ace",
                    "platform": openmm_platform,
                    "experimental": True,
                },
            }
        return SetCalculator(
            device,
            args.model,
            str(logs / log_name),
            atoms=target_atoms,
            **kwargs,
        ).set_calculator()

    gas_calculator = make_calculator(atoms, combined=False, log_name="warm-gas.log")
    combined_calculator = make_calculator(
        atoms, combined=True, log_name="warm-combined.log"
    )
    solvent = combined_calculator.solvent_correction
    provider_provenance = getattr(solvent.provider, "provenance", None)
    raw_solvent_provenance = (
        provider_provenance() if callable(provider_provenance) else provider_provenance
    )
    solvent_provenance = dict(
        _require_mapping(raw_solvent_provenance, "Route 1 provider provenance")
    )
    raw_platform_properties = _require_mapping(
        solvent_provenance.get("platform_properties"),
        "Route 1 provider platform properties",
    )
    route1_properties = {
        str(key): str(value) for key, value in raw_platform_properties.items()
    }
    expected_properties = {
        str(key): str(value)
        for key, value in policy["openmm_platform_properties"].items()
    }
    if (
        openmm_platform.lower() == "cpu"
        and expected_properties != DEFAULT_CPU_PLATFORM_PROPERTIES
    ):
        raise ValueError("Protocol CPU properties differ from the product default.")

    comparator = protocol["mm_comparator"]
    mm_context, _mm_integrator, mm_properties, mm_system = _create_mm_context(
        prmtop,
        platform_name=openmm_platform,
        properties=expected_properties if openmm_platform.lower() == "cpu" else {},
        sasa_method=str(comparator["sasa_method"]),
    )
    mm_implicit_validation = _validate_obc2_ace_system(
        mm_system,
        expected_force_class=str(comparator["expected_gbsa_force_class"]),
        expected_surface_area_energy_kj_mol_nm2=float(
            comparator["expected_surface_area_energy_kj_mol_nm2"]
        ),
    )

    def set_positions(index: int) -> np.ndarray:
        displaced = displaced_positions(index)
        atoms.set_positions(displaced)
        return displaced

    def gas_call(index: int) -> None:
        set_positions(index)
        gas_calculator.calculate(
            atoms, properties=["energy", "forces"], system_changes=all_changes
        )

    def combined_call(index: int) -> None:
        set_positions(index)
        combined_calculator.calculate(
            atoms, properties=["energy", "forces"], system_changes=all_changes
        )

    def correction_call(index: int) -> None:
        set_positions(index)
        solvent.evaluate(atoms, need_forces=True)

    def mm_call(index: int) -> None:
        from openmm import unit

        openmm_unit = cast(Any, unit)
        displaced = set_positions(index)
        mm_context.setPositions(displaced * 0.1 * openmm_unit.nanometer)
        state = mm_context.getState(getEnergy=True, getForces=True)
        _materialize_mm_energy_forces(
            state,
            atom_count=len(atoms),
            label="Named MM comparator",
        )

    gas_call(0)
    gas_energy = float(gas_calculator.results["energy"])
    gas_forces = np.asarray(gas_calculator.results["forces"], dtype=np.float64).copy()
    combined_call(0)
    combined_energy = float(combined_calculator.results["energy"])
    combined_forces = np.asarray(
        combined_calculator.results["forces"], dtype=np.float64
    ).copy()
    solvent_result = combined_calculator.solvation_result
    structured = combined_calculator.results["solvation"]
    structured_energy_closure = abs(
        combined_energy
        - float(structured["gas_energy_hartree"])
        - float(solvent_result.energy_hartree)
    )
    independent_gas_energy_difference = abs(
        gas_energy - float(structured["gas_energy_hartree"])
    )
    solvent_forces = np.asarray(
        solvent_result.forces_hartree_per_angstrom, dtype=np.float64
    )
    implied_gas_forces = combined_forces - solvent_forces
    independent_gas_force_max = float(np.max(np.abs(gas_forces - implied_gas_forces)))
    energy_closure = abs(
        combined_energy - gas_energy - float(solvent_result.energy_hartree)
    )
    force_closure = float(np.max(np.abs(combined_forces - gas_forces - solvent_forces)))
    acceptance = protocol["acceptance"]
    energy_checks = (
        energy_closure,
        structured_energy_closure,
        independent_gas_energy_difference,
    )
    force_checks = (force_closure, independent_gas_force_max)
    if max(energy_checks) > float(acceptance["maximum_energy_closure_hartree"]) or max(
        force_checks
    ) > float(acceptance["maximum_force_closure_hartree_per_angstrom"]):
        raise ValueError("Complete Route 1 energy/force composition did not close.")

    maximum_native_threads = int(policy["maximum_native_threadpool_threads"])
    native_threadpools_preflight = _native_threadpool_snapshot(
        "timing_preflight", maximum_threads=maximum_native_threads
    )
    host_load_timing_preflight = _host_load_snapshot("timing_preflight", maximum_load)
    warm_gas, warm_combined_for_overhead = benchmark_paired_calls_with_samples(
        gas_call,
        combined_call,
        samples=warm_samples,
        warmups=warmups,
        synchronize=synchronize,
    )
    warm_combined_for_mm, warm_mm = benchmark_paired_calls_with_samples(
        combined_call,
        mm_call,
        samples=warm_samples,
        warmups=warmups,
        synchronize=synchronize,
    )
    warm_correction = benchmark_calls_with_samples(
        correction_call,
        samples=warm_samples,
        warmups=warmups,
        synchronize=synchronize,
    )

    cold_gas_atoms = [copy.deepcopy(atoms) for _ in range(cold_samples)]
    cold_combined_overhead_atoms = [copy.deepcopy(atoms) for _ in range(cold_samples)]
    cold_combined_mm_atoms = [copy.deepcopy(atoms) for _ in range(cold_samples)]
    for index in range(cold_samples):
        cold_gas_atoms[index].set_positions(displaced_positions(index))
        for target in (
            cold_combined_overhead_atoms[index],
            cold_combined_mm_atoms[index],
        ):
            target.set_positions(displaced_positions(index))
            target.set_initial_charges(charges)

    def cold_gas_call(index: int):
        calculator = make_calculator(
            cold_gas_atoms[index],
            combined=False,
            log_name=f"cold-{index}-gas.log",
        )
        calculator.calculate(
            cold_gas_atoms[index],
            properties=["energy", "forces"],
            system_changes=all_changes,
        )
        return calculator

    def cold_combined_overhead_call(index: int):
        calculator = make_calculator(
            cold_combined_overhead_atoms[index],
            combined=True,
            log_name=f"cold-{index}-combined-overhead.log",
        )
        calculator.calculate(
            cold_combined_overhead_atoms[index],
            properties=["energy", "forces"],
            system_changes=all_changes,
        )
        return calculator

    def cold_combined_mm_call(index: int):
        calculator = make_calculator(
            cold_combined_mm_atoms[index],
            combined=True,
            log_name=f"cold-{index}-combined-mm.log",
        )
        calculator.calculate(
            cold_combined_mm_atoms[index],
            properties=["energy", "forces"],
            system_changes=all_changes,
        )
        return calculator

    def cold_mm_call(index: int):
        from openmm import unit

        openmm_unit = cast(Any, unit)
        context, integrator, _properties, system = _create_mm_context(
            prmtop,
            platform_name=openmm_platform,
            properties=expected_properties if openmm_platform.lower() == "cpu" else {},
            sasa_method=str(comparator["sasa_method"]),
        )
        context.setPositions(displaced_positions(index) * 0.1 * openmm_unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        _materialize_mm_energy_forces(
            state,
            atom_count=len(atoms),
            label="Cold named MM comparator",
        )
        return context, integrator, system

    cold_gas, cold_combined_for_overhead = benchmark_paired_construction_cold_calls(
        cold_gas_call,
        cold_combined_overhead_call,
        samples=cold_samples,
        synchronize=synchronize,
    )
    cold_combined_for_mm, cold_mm = benchmark_paired_construction_cold_calls(
        cold_combined_mm_call,
        cold_mm_call,
        samples=cold_samples,
        synchronize=synchronize,
    )
    native_threadpools_postflight = _native_threadpool_snapshot(
        "postflight", maximum_threads=maximum_native_threads
    )
    host_load_postflight = _host_load_snapshot("postflight", maximum_load)

    conformance = _protocol_conformance(
        protocol=protocol,
        device=device_name,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        openmm_platform=openmm_platform,
        warm_samples=warm_samples,
        warmups=warmups,
        cold_samples=cold_samples,
    )
    observed_environment = {
        name: os.environ.get(name) for name in sorted(policy["required_environment"])
    }
    eligibility = named_mm_comparison_eligibility(
        device=device_name,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        native_threadpools=native_threadpools_preflight["pools"],
        maximum_native_threadpool_threads=maximum_native_threads,
        required_environment=policy["required_environment"],
        observed_environment=observed_environment,
        route1_platform=str(solvent_provenance["platform"]),
        route1_properties=route1_properties,
        mm_platform=openmm_platform,
        mm_properties=mm_properties,
    )
    timings = {
        "warm": {
            "gas_mlip_for_overhead": warm_gas,
            "route1_combined_for_overhead": warm_combined_for_overhead,
            "route1_combined_for_mm": warm_combined_for_mm,
            "route1_correction_only": warm_correction,
            "named_mm_obc2_ace": warm_mm,
        },
        "construction_cold": {
            "gas_mlip_for_overhead": cold_gas,
            "route1_combined_for_overhead": cold_combined_for_overhead,
            "route1_combined_for_mm": cold_combined_for_mm,
            "named_mm_obc2_ace": cold_mm,
        },
    }
    for timing_class in timings.values():
        for name, summary in timing_class.items():
            if not _positive_timing(summary):
                raise ValueError(f"Invalid timing summary for {name}.")

    paired_comparisons = _derive_paired_comparisons(timings)
    ratios = _derive_ratios(timings, paired_comparisons)
    timing_observations = {
        "warm_solvent_overhead": {
            "fraction_observed": ratios["warm_combined_minus_gas_fraction_observed"],
            "interpretation": (
                "positive paired local observation; no uncertainty interval"
                if ratios["warm_combined_minus_gas_fraction_observed"] >= 0.0
                else "paired difference is unresolved at this noise level; "
                "the negative value is not a speedup"
            ),
        },
        "construction_cold_solvent_overhead": {
            "fraction_observed": ratios[
                "construction_cold_combined_minus_gas_fraction_observed"
            ],
            "interpretation": (
                "positive paired local observation; no uncertainty interval"
                if ratios["construction_cold_combined_minus_gas_fraction_observed"]
                >= 0.0
                else "paired difference is unresolved at this noise level; "
                "the negative value is not a speedup"
            ),
        },
    }

    available_platforms = []
    from openmm import Platform

    for index in range(Platform.getNumPlatforms()):
        available_platforms.append(Platform.getPlatform(index).getName())
    result = {
        "schema_version": 1,
        "artifact_type": "route1-performance-matrix-cell",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "command_provenance": command_provenance(
            __file__,
            vars(args),
            repository_root=REPOSITORY_ROOT,
            environment_variables=(
                "CUDA_VISIBLE_DEVICES",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ),
        ),
        "claim_scope": protocol["claim_scope"],
        "case": case,
        "model": args.model,
        "checkpoint": _checkpoint_record(args.model),
        "protocol_conformance": conformance,
        "route1": {
            **protocol["route1"],
            "platform": solvent_provenance["platform"],
            "platform_properties": route1_properties,
            "provenance": solvent_provenance,
        },
        "named_mm_comparator": {
            **protocol["mm_comparator"],
            "platform": openmm_platform,
            "platform_properties": mm_properties,
            "implicit_model_validation": mm_implicit_validation,
            "topology": topology,
        },
        "comparison_eligibility": eligibility,
        "composition_checks": {
            "energy_closure_hartree": energy_closure,
            "structured_energy_closure_hartree": structured_energy_closure,
            "force_closure_hartree_per_angstrom_max": force_closure,
            "independent_gas_energy_difference_hartree": independent_gas_energy_difference,
            "independent_gas_force_difference_hartree_per_angstrom_max": (
                independent_gas_force_max
            ),
        },
        "timings": timings,
        "paired_comparisons": paired_comparisons,
        "ratios": ratios,
        "timing_observations": timing_observations,
        "environment": {
            "host_platform": host_platform.platform(),
            "host_fingerprint": _host_fingerprint(),
            "python": host_platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "openmm": importlib.metadata.version("openmm"),
            "numpy": np.__version__,
            "available_openmm_platforms": available_platforms,
            "required_thread_environment": observed_environment,
            "timing_isolation": {
                "evidence_scope": "entry_and_timing_endpoint_snapshots_only",
                "continuous_host_isolation_monitored": False,
                "whole_run_host_isolation_proven": False,
                "entry": host_load_entry,
                "timing_preflight": host_load_timing_preflight,
                "postflight": host_load_postflight,
                "endpoint_snapshots_passed": True,
            },
            "native_threadpools": {
                "evidence_scope": "timing_endpoint_snapshots_only",
                "timing_preflight": native_threadpools_preflight,
                "postflight": native_threadpools_postflight,
                "endpoint_snapshots_passed": True,
            },
        },
        "inputs": {
            "mol2": _portable_artifact_path(mol2_path),
            "mol2_sha256": sha256_file(mol2_path),
            "charge_manifest": _portable_artifact_path(charge_manifest),
            "charge_manifest_sha256": sha256_file(charge_manifest),
        },
        "limitations": list(protocol["limitations"]),
    }
    return seal_artifact(result)


def _validate_timing_block(
    block: Mapping[str, Any],
    *,
    expected_keys: set[str],
    expected_samples: int,
    label: str,
) -> None:
    if set(block) != expected_keys:
        raise ValueError(
            f"{label} timing keys differ from the frozen protocol: {sorted(block)}."
        )
    for name in sorted(expected_keys):
        raw_summary = block[name]
        if not isinstance(raw_summary, Mapping):
            raise TypeError(f"Invalid {label} timing summary for {name}.")
        summary = cast(Mapping[str, Any], raw_summary)
        if not _positive_timing(summary):
            raise ValueError(f"Invalid {label} timing summary for {name}.")
        if _as_int(summary["n"], f"{label}.{name}.n") != expected_samples:
            raise ValueError(
                f"{label} timing sample count for {name} is {summary['n']}, "
                f"expected {expected_samples}."
            )
        samples = summary.get("samples_ms")
        if not isinstance(samples, list) or len(samples) != expected_samples:
            raise ValueError(f"{label} timing raw samples for {name} are incomplete.")
        numeric_samples = [
            _as_float(value, f"{label}.{name}.samples_ms[{index}]")
            for index, value in enumerate(samples)
        ]
        recomputed = _timing_summary_with_samples(numeric_samples)
        if summary != recomputed:
            raise ValueError(f"{label} timing summary for {name} is not reproducible.")


def _validate_host_load_evidence(
    evidence: Mapping[str, Any],
    *,
    maximum_per_logical_cpu: float,
    expected_logical_cpu_count: int,
) -> None:
    if (
        evidence.get("evidence_scope") != "entry_and_timing_endpoint_snapshots_only"
        or evidence.get("continuous_host_isolation_monitored") is not False
        or evidence.get("whole_run_host_isolation_proven") is not False
        or evidence.get("endpoint_snapshots_passed") is not True
    ):
        raise ValueError("Matrix cell has invalid host-load evidence boundaries.")
    for stage in ("entry", "timing_preflight", "postflight"):
        snapshot = evidence.get(stage)
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"Matrix cell is missing host-load {stage} evidence.")
        snapshot_record = cast(Mapping[str, Any], snapshot)
        load_1m = _as_float(
            snapshot_record.get("load_average_1m", np.nan),
            f"host-load.{stage}.load_average_1m",
        )
        load_per_cpu = _as_float(
            snapshot_record.get("load_per_logical_cpu", np.nan),
            f"host-load.{stage}.load_per_logical_cpu",
        )
        observed_limit = _as_float(
            snapshot_record.get("maximum_load_per_logical_cpu", np.nan),
            f"host-load.{stage}.maximum_load_per_logical_cpu",
        )
        logical_cpu_count = _as_int(
            snapshot_record.get("logical_cpu_count", 0),
            f"host-load.{stage}.logical_cpu_count",
        )
        if (
            snapshot_record.get("stage") != stage
            or snapshot_record.get("passed") is not True
            or logical_cpu_count != expected_logical_cpu_count
            or not np.isfinite(load_1m)
            or not np.isfinite(load_per_cpu)
            or load_per_cpu < 0.0
            or not np.isclose(
                load_per_cpu,
                load_1m / logical_cpu_count,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            or not np.isclose(
                observed_limit,
                maximum_per_logical_cpu,
                rtol=0.0,
                atol=0.0,
            )
            or load_per_cpu > maximum_per_logical_cpu
        ):
            raise ValueError(f"Matrix cell host-load {stage} evidence is invalid.")
        for name in ("load_average_1m", "load_average_5m", "load_average_15m"):
            value = _as_float(
                snapshot_record.get(name, np.nan), f"host-load.{stage}.{name}"
            )
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Matrix cell host-load {stage} has invalid {name}.")


def _validate_native_threadpool_evidence(
    evidence: Mapping[str, Any],
    *,
    maximum_threads: int,
) -> None:
    if (
        evidence.get("evidence_scope") != "timing_endpoint_snapshots_only"
        or evidence.get("endpoint_snapshots_passed") is not True
    ):
        raise ValueError("Matrix cell native-threadpool evidence is invalid.")
    for stage in ("timing_preflight", "postflight"):
        raw_snapshot = evidence.get(stage)
        if not isinstance(raw_snapshot, Mapping):
            raise TypeError(
                f"Matrix cell native-threadpool {stage} evidence is invalid."
            )
        snapshot = cast(Mapping[str, Any], raw_snapshot)
        if (
            snapshot.get("stage") != stage
            or snapshot.get("passed") is not True
            or _as_int(
                snapshot.get("maximum_threads", 0),
                f"native-threadpool.{stage}.maximum_threads",
            )
            != maximum_threads
        ):
            raise ValueError(
                f"Matrix cell native-threadpool {stage} evidence is invalid."
            )
        pools = snapshot.get("pools")
        if not isinstance(pools, list) or not pools:
            raise ValueError(
                f"Matrix cell native-threadpool {stage} records are missing."
            )
        for record in pools:
            record_mapping = _require_mapping(record, f"native-threadpool.{stage}.pool")
            threads = _as_int(
                record_mapping.get("num_threads", 0),
                f"native-threadpool.{stage}.num_threads",
            )
            if threads < 1 or threads > maximum_threads:
                raise ValueError(
                    f"Matrix cell native-threadpool {stage} exceeds the limit."
                )


def _validate_cell_for_assembly(
    protocol: Mapping[str, Any],
    protocol_hash: str,
    cell: Mapping[str, Any],
    *,
    verify_external_dependencies: bool,
) -> tuple[str, str]:
    if (
        cell.get("schema_version") != 1
        or cell.get("artifact_type") != "route1-performance-matrix-cell"
    ):
        raise ValueError("Matrix assembly received an invalid cell artifact.")
    if cell.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("Matrix cell uses a different protocol identifier.")
    if cell.get("protocol_sha256") != protocol_hash:
        raise ValueError("Matrix cell uses a different protocol hash.")
    if cell.get("content_sha256") != artifact_content_sha256(cell):
        raise ValueError("Matrix cell content hash is invalid.")
    if cell.get("claim_scope") != protocol["claim_scope"]:
        raise ValueError("Matrix cell claim scope differs from the protocol.")
    if cell.get("limitations") != protocol["limitations"]:
        raise ValueError("Matrix cell limitations differ from the protocol.")

    case_record = cell.get("case")
    if not isinstance(case_record, Mapping):
        raise TypeError("Matrix cell case metadata is missing.")
    case_id = str(case_record.get("case_id", ""))
    model = str(cell.get("model", ""))
    expected_case = _find_case(protocol, case_id)
    if (
        not _exact_json_equal(case_record, expected_case)
        or model not in protocol["matrix"]["models"]
    ):
        raise ValueError("Matrix cell case/model metadata differs from the protocol.")
    policy = protocol["resource_policy"]
    timing = protocol["timing"]

    try:
        command = _require_mapping(
            cell.get("command_provenance"), "Matrix cell command provenance"
        )
    except TypeError as exc:
        raise ValueError(
            "Matrix cell was not produced by the current frozen runner."
        ) from exc
    command_arguments = command.get("arguments")
    if (
        command.get("script")
        != "docs/implicit-solvation/benchmarks/run_route1_performance_matrix.py"
        or command.get("script_sha256") != sha256_file(__file__)
        or not isinstance(command_arguments, Mapping)
        or command.get("arguments_sha256")
        != sha256_bytes(canonical_json_bytes(command_arguments))
    ):
        raise ValueError("Matrix cell was not produced by the current frozen runner.")
    expected_argument_values = {
        "command": "run-cell",
        "case_id": case_id,
        "model": model,
    }
    if any(
        command_arguments.get(key) != value
        for key, value in expected_argument_values.items()
    ):
        raise ValueError("Matrix cell command arguments differ from its case/model.")
    protocol_argument = Path(str(command_arguments.get("protocol", "")))
    if not protocol_argument.is_absolute():
        protocol_argument = REPOSITORY_ROOT / protocol_argument
    if (
        not protocol_argument.is_file()
        or load_and_validate_protocol(protocol_argument)[1] != protocol_hash
    ):
        raise ValueError("Matrix cell command references a different protocol.")
    optional_argument_expectations = {
        "device": policy["device"],
        "torch_threads": _as_int(
            policy["torch_threads"], "resource_policy.torch_threads"
        ),
        "openmm_platform": policy["openmm_platform"],
        "warm_samples": _as_int(
            protocol["timing"]["warm"]["samples"], "timing.warm.samples"
        ),
        "warmups": _as_int(
            protocol["timing"]["warm"]["warmups"], "timing.warm.warmups"
        ),
        "cold_samples": _as_int(
            protocol["timing"]["construction_cold"]["samples"],
            "timing.construction_cold.samples",
        ),
    }
    for key, expected_value in optional_argument_expectations.items():
        observed_value = command_arguments.get(key)
        if observed_value is None:
            continue
        if type(expected_value) is int:
            observed_value = _as_int(
                observed_value, f"command_provenance.arguments.{key}"
            )
        if observed_value != expected_value:
            raise ValueError(f"Matrix cell command override {key} is nonconformant.")
    command_environment = _require_mapping(
        command.get("environment_variables"),
        "Matrix cell command environment variables",
    )
    if any(
        command_environment.get(name) != expected_value
        for name, expected_value in policy["required_environment"].items()
    ):
        raise ValueError("Matrix cell command environment violates resource policy.")
    checkpoint = _require_mapping(cell.get("checkpoint"), "Matrix cell checkpoint")
    expected_checkpoint_filename = {
        "maceoff23m": "maceoff23m.pt",
        "aimnet2": "aimnet2.pt",
        "ani2x": "ani2x.pt",
    }[model]
    if (
        set(checkpoint) != {"path", "sha256", "size_bytes"}
        or Path(str(checkpoint.get("path", ""))).name != expected_checkpoint_filename
        or not _is_sha256(checkpoint.get("sha256"))
        or _as_int(checkpoint.get("size_bytes"), "checkpoint.size_bytes") < 1
    ):
        raise ValueError("Matrix cell checkpoint provenance is invalid.")
    if verify_external_dependencies:
        current_checkpoint = _checkpoint_record(model)
        if (
            checkpoint.get("sha256") != current_checkpoint["sha256"]
            or checkpoint["size_bytes"] != current_checkpoint["size_bytes"]
        ):
            raise ValueError("Matrix cell checkpoint differs from the current model.")

    expected_conformance = {
        "device": policy["device"],
        "torch_threads": _as_int(
            policy["torch_threads"], "resource_policy.torch_threads"
        ),
        "torch_interop_threads": _as_int(
            policy["torch_interop_threads"],
            "resource_policy.torch_interop_threads",
        ),
        "openmm_platform": policy["openmm_platform"],
        "warm_samples": _as_int(timing["warm"]["samples"], "timing.warm.samples"),
        "warmups": _as_int(timing["warm"]["warmups"], "timing.warm.warmups"),
        "cold_samples": _as_int(
            timing["construction_cold"]["samples"],
            "timing.construction_cold.samples",
        ),
    }
    conformance = _require_mapping(
        cell.get("protocol_conformance"), "Matrix cell protocol conformance"
    )
    if (
        conformance.get("conformant") is not True
        or conformance.get("reasons") != []
        or not _exact_json_equal(conformance.get("expected"), expected_conformance)
        or not _exact_json_equal(conformance.get("observed"), expected_conformance)
    ):
        raise ValueError("Nonconformant matrix cell.")

    environment = _require_mapping(cell.get("environment"), "Matrix cell environment")
    observed_thread_environment = cast(
        Mapping[str, str | None],
        _require_mapping(
            environment.get("required_thread_environment"),
            "Matrix cell required thread environment",
        ),
    )
    if (
        environment.get("device") != policy["device"]
        or _as_int(
            environment.get("torch_num_threads", 0),
            "environment.torch_num_threads",
        )
        != _as_int(policy["torch_threads"], "resource_policy.torch_threads")
        or _as_int(
            environment.get("torch_num_interop_threads", 0),
            "environment.torch_num_interop_threads",
        )
        != _as_int(
            policy["torch_interop_threads"],
            "resource_policy.torch_interop_threads",
        )
        or observed_thread_environment != policy["required_environment"]
    ):
        raise ValueError("Matrix cell environment differs from the resource policy.")
    raw_host_fingerprint = environment.get("host_fingerprint", {})
    if not isinstance(raw_host_fingerprint, Mapping):
        raise TypeError("Matrix cell host fingerprint is incomplete.")
    host_fingerprint = cast(Mapping[str, Any], raw_host_fingerprint)
    logical_cpu_count = _as_int(
        host_fingerprint.get("logical_cpu_count", 0),
        "environment.host_fingerprint.logical_cpu_count",
    )
    cpu_affinity = host_fingerprint.get("cpu_affinity")
    if (
        not str(host_fingerprint.get("node", "")).strip()
        or not str(host_fingerprint.get("cpu_model", "")).strip()
        or logical_cpu_count < 1
        or not isinstance(cpu_affinity, list)
        or not cpu_affinity
    ):
        raise ValueError("Matrix cell host fingerprint is incomplete.")
    if any(type(cpu_index) is not int or cpu_index < 0 for cpu_index in cpu_affinity):
        raise TypeError("Matrix cell CPU affinity must contain JSON integers.")
    for key in ("host_platform", "python", "torch", "openmm", "numpy"):
        if not str(environment.get(key, "")).strip():
            raise ValueError(f"Matrix cell software field {key} is incomplete.")
    _validate_host_load_evidence(
        _require_mapping(
            environment.get("timing_isolation"),
            "Matrix cell timing-isolation evidence",
        ),
        maximum_per_logical_cpu=_as_float(
            policy["maximum_one_minute_load_per_logical_cpu"],
            "resource_policy.maximum_one_minute_load_per_logical_cpu",
        ),
        expected_logical_cpu_count=logical_cpu_count,
    )
    native_threadpool_evidence = _require_mapping(
        environment.get("native_threadpools"),
        "Matrix cell native-threadpool evidence",
    )
    _validate_native_threadpool_evidence(
        native_threadpool_evidence,
        maximum_threads=_as_int(
            policy["maximum_native_threadpool_threads"],
            "resource_policy.maximum_native_threadpool_threads",
        ),
    )

    route1 = _require_mapping(cell.get("route1"), "Matrix cell Route 1 evidence")
    for key, value in protocol["route1"].items():
        if route1.get(key) != value:
            raise ValueError(f"Matrix cell Route 1 field {key} differs from protocol.")
    expected_properties = {
        str(key): str(value)
        for key, value in policy["openmm_platform_properties"].items()
    }
    route1_provenance = route1.get("provenance")
    if (
        str(route1.get("platform", "")).lower() != "cpu"
        or route1.get("platform_properties") != expected_properties
        or not isinstance(route1_provenance, Mapping)
    ):
        raise ValueError("Matrix cell Route 1 platform evidence is invalid.")
    try:
        radius_provider = _require_mapping(
            route1_provenance.get("radius_provider"),
            "Matrix cell Route 1 radius provider provenance",
        )
        nonpolar_provider = _require_mapping(
            route1_provenance.get("nonpolar_provider"),
            "Matrix cell Route 1 nonpolar provider provenance",
        )
    except TypeError as exc:
        raise ValueError("Matrix cell Route 1 provider provenance is invalid.") from exc
    required_route1_provenance = {
        "method": "gb",
        "model": "obc2",
        "amber_igb": 5,
        "profile": "obc2-mbondi2",
        "provider": "openmm",
        "provider_version": environment["openmm"],
        "radii": "mbondi2",
        "nonpolar": "ace",
        "platform": "CPU",
        "platform_properties": expected_properties,
        "solvent": "water",
        "solute_dielectric": 1.0,
        "solvent_dielectric": 78.5,
        "energy_force_evaluations_per_call": 1,
    }
    if any(
        route1_provenance.get(key) != value
        for key, value in required_route1_provenance.items()
    ) or (
        radius_provider.get("name") != "openmm-amber-gb-radii"
        or radius_provider.get("provider") != "openmm"
        or radius_provider.get("profile") != "obc2-mbondi2"
        or radius_provider.get("radii") != "mbondi2"
        or nonpolar_provider.get("name") != "openmm-ace"
        or nonpolar_provider.get("provider") != "openmm"
        or nonpolar_provider.get("profile") != "ace"
        or nonpolar_provider.get("component_properties") != ["energy", "forces"]
    ):
        raise ValueError("Matrix cell Route 1 provider provenance is invalid.")

    comparator = _require_mapping(
        cell.get("named_mm_comparator"), "Matrix cell named-MM comparator"
    )
    for key, value in protocol["mm_comparator"].items():
        if comparator.get(key) != value:
            raise ValueError(
                f"Matrix cell named-MM comparator field {key} differs from protocol."
            )
    expected_implicit_validation = {
        "implicit_solvent": protocol["mm_comparator"]["implicit_solvent"],
        "sasa_method": protocol["mm_comparator"]["sasa_method"],
        "force_class": protocol["mm_comparator"]["expected_gbsa_force_class"],
        "force_count": 1,
        "surface_area_energy_kj_mol_nm2": protocol["mm_comparator"][
            "expected_surface_area_energy_kj_mol_nm2"
        ],
    }
    implicit_validation = _require_mapping(
        comparator.get("implicit_model_validation"),
        "Matrix cell named-MM implicit-model validation",
    )
    if (
        comparator.get("platform") != policy["openmm_platform"]
        or comparator.get("platform_properties") != expected_properties
        or not _exact_json_equal(
            implicit_validation,
            expected_implicit_validation,
        )
    ):
        raise ValueError("Matrix cell named-MM platform/model evidence is invalid.")

    topology = _require_mapping(
        comparator.get("topology"), "Matrix cell named-MM topology"
    )
    topology_validation = _require_mapping(
        topology.get("validation"), "Matrix cell named-MM topology validation"
    )
    acceptance = protocol["acceptance"]
    charge_difference = _as_float(
        topology_validation.get("charge_max_abs_e"),
        "named_mm_comparator.topology.validation.charge_max_abs_e",
    )
    coordinate_difference = _as_float(
        topology_validation.get("coordinate_max_abs_angstrom"),
        "named_mm_comparator.topology.validation.coordinate_max_abs_angstrom",
    )
    if (
        _as_int(
            topology_validation.get("atom_count", 0),
            "named_mm_comparator.topology.validation.atom_count",
        )
        != expected_case["atom_count"]
        or topology_validation.get("atom_names_preserved") is not True
        or _as_int(
            topology_validation.get("frcmod_nonbon_overrides", -1),
            "named_mm_comparator.topology.validation.frcmod_nonbon_overrides",
        )
        != 0
        or not np.isfinite(charge_difference)
        or charge_difference < 0.0
        or charge_difference
        > _as_float(
            acceptance["maximum_topology_charge_difference_e"],
            "acceptance.maximum_topology_charge_difference_e",
        )
        or not np.isfinite(coordinate_difference)
        or coordinate_difference < 0.0
        or coordinate_difference
        > _as_float(
            acceptance["maximum_topology_coordinate_difference_angstrom"],
            "acceptance.maximum_topology_coordinate_difference_angstrom",
        )
    ):
        raise ValueError("Matrix cell named-MM topology validation failed.")
    topology_files = _require_mapping(
        topology.get("files"), "Matrix cell named-MM topology files"
    )
    for name in (
        "molecule.mol2",
        "molecule.frcmod",
        "molecule.prmtop",
        "molecule.inpcrd",
    ):
        record = _require_mapping(
            topology_files.get(name), f"Matrix cell named-MM topology file {name}"
        )
        if not _is_sha256(record.get("sha256")) or not _is_sha256(
            record.get("semantic_sha256")
        ):
            raise ValueError(
                f"Matrix cell named-MM topology file {name} lacks valid hashes."
            )
        topology_path = _resolve_artifact_path(record.get("path", ""))
        if (
            not topology_path.is_file()
            or sha256_file(topology_path) != record["sha256"]
            or _topology_semantic_sha256(topology_path) != record["semantic_sha256"]
        ):
            raise ValueError(
                f"Matrix cell named-MM topology file {name} is unavailable or "
                "does not match its recorded hashes."
            )
    executables = _require_mapping(
        topology.get("executables"), "Matrix cell AmberTools executables"
    )
    if set(executables) != {"parmchk2", "tleap"}:
        raise ValueError("Matrix cell AmberTools executable hashes are invalid.")
    executable_records = {
        name: _require_mapping(record, f"Matrix cell AmberTools executable {name}")
        for name, record in executables.items()
    }
    if any(
        set(record) != {"path", "sha256"}
        or not str(record.get("path", "")).strip()
        or not _is_sha256(record.get("sha256"))
        for record in executable_records.values()
    ):
        raise ValueError("Matrix cell AmberTools executable hashes are invalid.")
    if verify_external_dependencies:
        for name, record in executable_records.items():
            executable_path = _resolve_artifact_path(record.get("path", ""))
            if (
                not executable_path.is_file()
                or not os.access(executable_path, os.X_OK)
                or sha256_file(executable_path) != record["sha256"]
            ):
                raise ValueError(
                    f"Matrix cell AmberTools executable {name} is unavailable or "
                    "does not match its recorded hash."
                )

    native_preflight = _require_mapping(
        native_threadpool_evidence.get("timing_preflight"),
        "Matrix cell native-threadpool timing-preflight evidence",
    )
    native_preflight_pools = native_preflight.get("pools")
    if not isinstance(native_preflight_pools, list):
        raise TypeError("Matrix cell native-threadpool records must be a list.")
    expected_eligibility = named_mm_comparison_eligibility(
        device=str(environment["device"]),
        torch_threads=_as_int(
            environment["torch_num_threads"], "environment.torch_num_threads"
        ),
        torch_interop_threads=_as_int(
            environment["torch_num_interop_threads"],
            "environment.torch_num_interop_threads",
        ),
        native_threadpools=native_preflight_pools,
        maximum_native_threadpool_threads=_as_int(
            policy["maximum_native_threadpool_threads"],
            "resource_policy.maximum_native_threadpool_threads",
        ),
        required_environment=policy["required_environment"],
        observed_environment=observed_thread_environment,
        route1_platform=str(route1["platform"]),
        route1_properties=route1["platform_properties"],
        mm_platform=str(comparator["platform"]),
        mm_properties=comparator["platform_properties"],
    )
    if cell.get("comparison_eligibility") != expected_eligibility:
        raise ValueError("Ineligible or inconsistent named-MM comparison cell.")

    composition = _require_mapping(
        cell.get("composition_checks"), "Matrix cell composition checks"
    )
    energy_keys = (
        "energy_closure_hartree",
        "structured_energy_closure_hartree",
        "independent_gas_energy_difference_hartree",
    )
    force_keys = (
        "force_closure_hartree_per_angstrom_max",
        "independent_gas_force_difference_hartree_per_angstrom_max",
    )
    if set(composition) != set(energy_keys) | set(force_keys):
        raise ValueError("Matrix cell composition-check schema is incomplete.")
    for key in energy_keys:
        value = _as_float(composition[key], f"composition_checks.{key}")
        if (
            not np.isfinite(value)
            or value < 0.0
            or value
            > _as_float(
                acceptance["maximum_energy_closure_hartree"],
                "acceptance.maximum_energy_closure_hartree",
            )
        ):
            raise ValueError(f"Matrix cell energy composition check {key} failed.")
    for key in force_keys:
        value = _as_float(composition[key], f"composition_checks.{key}")
        if (
            not np.isfinite(value)
            or value < 0.0
            or value
            > _as_float(
                acceptance["maximum_force_closure_hartree_per_angstrom"],
                "acceptance.maximum_force_closure_hartree_per_angstrom",
            )
        ):
            raise ValueError(f"Matrix cell force composition check {key} failed.")

    inputs = _require_mapping(cell.get("inputs"), "Matrix cell inputs")
    expected_inputs = {
        "mol2": (
            _resolve_artifact_path(protocol["inputs"]["input_root_relative_path"])
            / expected_case["mol2_relative_path"],
            expected_case["mol2_sha256"],
        ),
        "charge_manifest": (
            _resolve_artifact_path(protocol["inputs"]["charge_manifest_relative_path"]),
            protocol["inputs"]["charge_manifest_sha256"],
        ),
    }
    if set(inputs) != {
        "mol2",
        "mol2_sha256",
        "charge_manifest",
        "charge_manifest_sha256",
    }:
        raise ValueError("Matrix cell source hashes differ from the protocol.")
    for name, (expected_path, expected_hash) in expected_inputs.items():
        recorded_path = _resolve_artifact_path(inputs.get(name, ""))
        recorded_hash = inputs.get(f"{name}_sha256")
        if (
            recorded_path != expected_path.resolve()
            or recorded_hash != expected_hash
            or not recorded_path.is_file()
            or sha256_file(recorded_path) != recorded_hash
        ):
            raise ValueError(
                f"Matrix cell source {name} is unavailable or does not match "
                "the protocol."
            )

    timings = _require_mapping(cell.get("timings"), "Matrix cell timings")
    if set(timings) != {"warm", "construction_cold"}:
        raise ValueError("Matrix cell timing classes differ from the protocol.")
    _validate_timing_block(
        timings["warm"],
        expected_keys=WARM_TIMING_KEYS,
        expected_samples=_as_int(timing["warm"]["samples"], "timing.warm.samples"),
        label="warm",
    )
    _validate_timing_block(
        timings["construction_cold"],
        expected_keys=CONSTRUCTION_COLD_TIMING_KEYS,
        expected_samples=_as_int(
            timing["construction_cold"]["samples"],
            "timing.construction_cold.samples",
        ),
        label="construction-cold",
    )
    expected_paired_comparisons = _derive_paired_comparisons(timings)
    if not _exact_json_equal(
        cell.get("paired_comparisons"), expected_paired_comparisons
    ):
        raise ValueError("Matrix cell paired comparisons are not reproducible.")
    expected_ratios = _derive_ratios(timings, expected_paired_comparisons)
    observed_ratios = _require_mapping(cell.get("ratios"), "Matrix cell ratios")
    if set(observed_ratios) != set(expected_ratios):
        raise ValueError("Matrix cell ratio schema differs from the runner.")
    for key, expected in expected_ratios.items():
        observed = observed_ratios[key]
        if isinstance(expected, bool):
            if observed is not expected:
                raise ValueError(f"Matrix cell ratio decision {key} is inconsistent.")
        elif not np.isclose(
            _as_float(observed, f"ratios.{key}"),
            _as_float(expected, f"expected_ratios.{key}"),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(f"Matrix cell ratio {key} is inconsistent.")
    return case_id, model


def assemble_matrix(
    protocol: Mapping[str, Any],
    protocol_hash: str,
    cells: Sequence[Mapping[str, Any]],
    *,
    verify_external_dependencies: bool = True,
) -> dict[str, Any]:
    expected = {
        (case["case_id"], model)
        for case in protocol["matrix"]["cases"]
        for model in protocol["matrix"]["models"]
    }
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in cells:
        key = _validate_cell_for_assembly(
            protocol,
            protocol_hash,
            cell,
            verify_external_dependencies=verify_external_dependencies,
        )
        if key in observed:
            raise ValueError(f"Duplicate matrix cell: {key}.")
        observed[key] = cell
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(f"Matrix coverage mismatch; missing={missing}, extra={extra}.")

    host_signatures = {
        sha256_bytes(
            canonical_json_bytes(
                {
                    key: cell["environment"][key]
                    for key in (
                        "host_platform",
                        "host_fingerprint",
                        "python",
                        "torch",
                        "torch_num_threads",
                        "torch_num_interop_threads",
                        "device",
                        "openmm",
                        "numpy",
                        "available_openmm_platforms",
                        "required_thread_environment",
                    )
                }
            )
        )
        for cell in observed.values()
    }
    if len(host_signatures) != 1:
        raise ValueError("Matrix cells do not share one host/software/resource policy.")

    executable_signatures = {
        sha256_bytes(
            canonical_json_bytes(cell["named_mm_comparator"]["topology"]["executables"])
        )
        for cell in observed.values()
    }
    if len(executable_signatures) != 1:
        raise ValueError("Matrix cells use different AmberTools executables.")

    for model in protocol["matrix"]["models"]:
        checkpoint_signatures = {
            sha256_bytes(
                canonical_json_bytes(observed[(case["case_id"], model)]["checkpoint"])
            )
            for case in protocol["matrix"]["cases"]
        }
        if len(checkpoint_signatures) != 1:
            raise ValueError(f"Matrix cells use different {model} checkpoints.")

    core_topology_names = (
        "molecule.mol2",
        "molecule.frcmod",
        "molecule.prmtop",
        "molecule.inpcrd",
    )
    for case in protocol["matrix"]["cases"]:
        topology_signatures = set()
        for model in protocol["matrix"]["models"]:
            files = observed[(case["case_id"], model)]["named_mm_comparator"][
                "topology"
            ]["files"]
            if any(name not in files for name in core_topology_names):
                raise ValueError(
                    f"Matrix case {case['case_id']} lacks comparator topology files."
                )
            topology_signatures.add(
                sha256_bytes(
                    canonical_json_bytes(
                        {
                            name: files[name]["semantic_sha256"]
                            for name in core_topology_names
                        }
                    )
                )
            )
        if len(topology_signatures) != 1:
            raise ValueError(
                f"Matrix case {case['case_id']} has model-dependent MM topology."
            )

    records = []
    for case in protocol["matrix"]["cases"]:
        for model in protocol["matrix"]["models"]:
            cell = observed[(case["case_id"], model)]
            records.append(
                {
                    "case_id": case["case_id"],
                    "size_bin": case["size_bin"],
                    "atom_count": case["atom_count"],
                    "model": model,
                    "cell_content_sha256": cell["content_sha256"],
                    "timings": copy.deepcopy(cell["timings"]),
                    "paired_comparisons": copy.deepcopy(cell["paired_comparisons"]),
                    "ratios": copy.deepcopy(cell["ratios"]),
                    "composition_checks": copy.deepcopy(cell["composition_checks"]),
                    "provenance": {
                        "command_provenance": copy.deepcopy(cell["command_provenance"]),
                        "checkpoint": copy.deepcopy(cell["checkpoint"]),
                        "environment": copy.deepcopy(cell["environment"]),
                        "inputs": copy.deepcopy(cell["inputs"]),
                        "named_mm_topology": copy.deepcopy(
                            cell["named_mm_comparator"]["topology"]
                        ),
                    },
                }
            )
    summary = {
        "cell_count": len(records),
        "warm_route1_faster_than_named_mm_count": sum(
            bool(record["ratios"]["warm_route1_faster_than_named_mm_observed"])
            for record in records
        ),
        "construction_cold_route1_faster_than_named_mm_count": sum(
            bool(
                record["ratios"][
                    "construction_cold_route1_faster_than_named_mm_observed"
                ]
            )
            for record in records
        ),
        "all_cells_protocol_conformant": True,
        "all_named_mm_comparisons_eligible": True,
        "all_cells_same_host_software_resource_policy": True,
        "all_case_mm_topologies_model_invariant": True,
        "universal_faster_than_mm_claim_admitted": False,
        "universal_negligible_solvent_overhead_claim_admitted": False,
    }
    return seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-fair-sp-performance-matrix",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_hash,
            "claim_scope": protocol["claim_scope"],
            "records": records,
            "summary": summary,
            "limitations": list(protocol["limitations"]),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-cell", help="Run one model/case matrix cell.")
    run.add_argument("--protocol", required=True)
    run.add_argument("--case-id", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--ambertools-bin", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--device")
    run.add_argument("--torch-threads", type=int)
    run.add_argument("--openmm-platform")
    run.add_argument("--warm-samples", type=int)
    run.add_argument("--warmups", type=int)
    run.add_argument("--cold-samples", type=int)
    run.add_argument("--output", required=True)

    assemble = subparsers.add_parser(
        "assemble", help="Verify and assemble the exact nine-cell matrix."
    )
    assemble.add_argument("--protocol", required=True)
    assemble.add_argument("--cell", action="append", required=True)
    assemble.add_argument("--output", required=True)

    verify_frozen = subparsers.add_parser(
        "verify-frozen",
        help="Reassemble committed evidence without local checkpoints/AmberTools.",
    )
    verify_frozen.add_argument("--protocol", required=True)
    verify_frozen.add_argument("--cell", action="append", required=True)
    verify_frozen.add_argument("--matrix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-cell":
        result = run_cell(args)
    elif args.command == "assemble":
        protocol, protocol_hash = load_and_validate_protocol(args.protocol)
        cells = [load_json(path) for path in args.cell]
        result = assemble_matrix(protocol, protocol_hash, cells)
    else:
        protocol, protocol_hash = load_and_validate_protocol(args.protocol)
        cells = [load_json(path) for path in args.cell]
        frozen = load_json(args.matrix)
        result = assemble_matrix(
            protocol,
            protocol_hash,
            cells,
            verify_external_dependencies=False,
        )
        if result != frozen:
            raise ValueError("Frozen performance matrix does not reassemble exactly.")
        print(
            json.dumps(
                {"valid": True, "content_sha256": result["content_sha256"]},
                indent=2,
            )
        )
        return
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
