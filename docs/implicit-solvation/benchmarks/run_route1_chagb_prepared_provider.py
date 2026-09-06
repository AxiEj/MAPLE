#!/usr/bin/env python3
"""Compare one-time CHA-GB topology preparation with the legacy fresh path."""

from __future__ import annotations

import argparse
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_core import (  # noqa: E402
    artifact_content_sha256,
    canonical_json_bytes,
    command_provenance,
    load_json,
    seal_artifact,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from maple.function.calculator.extra_correction.implicit.amber_chagb import (  # noqa: E402
    AmberToolsChaGB,
    KCAL_PER_MOL_PER_HARTREE,
    _run,
    parse_gbnsr6_components,
    parse_pbsa_components,
    render_typed_mol2,
    require_no_frcmod_nonbonded_overrides,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

PROTOCOL_ID = "maple-route1-chagb-prepared-provider-v1"
ARTIFACT_TYPE = "route1-chagb-prepared-provider-parity"
COMPONENTS = ("polar", "cavity", "dispersion", "total")


def _safe_benchmark_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe benchmark-relative path: {value!r}.")
    path = (SCRIPT_DIR / relative).resolve()
    try:
        path.relative_to(SCRIPT_DIR)
    except ValueError as exc:
        raise ValueError(f"Benchmark path escapes its directory: {value!r}.") from exc
    return path


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def load_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    protocol = load_json(path)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("Unexpected prepared-provider protocol.")
    route = protocol.get("route_contract", {})
    if (
        route.get("gas_phase_mm_energy_used") is not False
        or route.get("bonded_mm_energy_used") is not False
        or route.get("hydration_label_fit_or_residual") is not False
        or route.get("fixed_charge_geometry") != "keep"
        or route.get("supported_tasks") != ["sp"]
        or route.get("force_requests_fail_closed") is not True
    ):
        raise ValueError("Prepared-provider protocol violates the Route 1 boundary.")
    source = protocol.get("source", {})
    if (
        not isinstance(source.get("mol2"), str)
        or not _is_sha256(source.get("mol2_sha256"))
        or source.get("label_reads") is not False
    ):
        raise ValueError("Prepared-provider protocol source is invalid.")
    cases = protocol.get("coordinate_cases")
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("Prepared-provider protocol requires source plus warm cases.")
    identifiers: set[str] = set()
    for case in cases:
        identifier = case.get("id")
        translation = np.asarray(case.get("translation_angstrom"), dtype=np.float64)
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or translation.shape != (3,)
            or not np.isfinite(translation).all()
        ):
            raise ValueError("Prepared-provider coordinate case is invalid.")
        displacements = case.get("atom_displacements", [])
        if not isinstance(displacements, list):
            raise ValueError("Prepared-provider atom displacement list is invalid.")
        displacement_indices: set[int] = set()
        for displacement in displacements:
            index = displacement.get("atom_index")
            delta = np.asarray(displacement.get("delta_angstrom"), dtype=np.float64)
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index in displacement_indices
                or delta.shape != (3,)
                or not np.isfinite(delta).all()
            ):
                raise ValueError("Prepared-provider atom displacement is invalid.")
            displacement_indices.add(index)
        identifiers.add(identifier)
    comparison = protocol.get("comparison", {})
    if (
        float(comparison.get("component_parity_tolerance_kcal_mol", 0.0)) <= 0.0
        or int(comparison.get("warm_coordinate_count", -1)) != len(cases) - 1
        or comparison.get("persistent_external_worker_claimed") is not False
    ):
        raise ValueError("Prepared-provider comparison contract is invalid.")
    return protocol, sha256_bytes(canonical_json_bytes(protocol))


def _resolve_amber_bin(path: str | Path) -> Path:
    amber_bin = Path(path).expanduser().resolve()
    required = ("gbnsr6", "pbsa", "parmchk2", "tleap")
    missing = [name for name in required if not (amber_bin / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"AmberTools bin directory is incomplete: {amber_bin}; missing {missing}."
        )
    return amber_bin


def _legacy_fresh_topology_evaluate(
    provider: AmberToolsChaGB,
    positions_angstrom: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Execute the pre-cache provider workflow without modifying its formula."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="maple-chagb-legacy-reference-") as temporary:
        work = Path(temporary)
        mol2 = work / "molecule.mol2"
        mol2.write_text(
            render_typed_mol2(
                provider.source_text,
                positions_angstrom,
                provider.charges,
            ),
            encoding="utf-8",
        )
        frcmod = work / "molecule.frcmod"
        _run(
            [
                str(provider.executables["parmchk2"]),
                "-i",
                mol2.name,
                "-f",
                "mol2",
                "-o",
                frcmod.name,
                "-s",
                "gaff2",
            ],
            cwd=work,
            timeout=provider.timeout,
            label="legacy-parmchk2",
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
        _run(
            [str(provider.executables["tleap"]), "-f", leap_input.name],
            cwd=work,
            timeout=provider.timeout,
            label="legacy-tleap",
        )
        prmtop = work / "molecule.prmtop"
        inpcrd = work / "molecule.inpcrd"
        if not prmtop.is_file() or not inpcrd.is_file():
            raise RuntimeError("Legacy tleap did not create topology/coordinates.")

        gb_input = work / "gbnsr6.in"
        gb_output = work / "gbnsr6.out"
        gb_input.write_text(provider._gbnsr6_input(), encoding="utf-8")
        with provider._serialized_gbnsr6():
            _run(
                [
                    str(provider.executables["gbnsr6"]),
                    "-O",
                    "-i",
                    gb_input.name,
                    "-o",
                    gb_output.name,
                    "-p",
                    prmtop.name,
                    "-c",
                    inpcrd.name,
                ],
                cwd=work,
                timeout=provider.timeout,
                label="legacy-gbnsr6",
            )
        pb_input = work / "pbsa.in"
        pb_output = work / "pbsa.out"
        pb_input.write_text(provider._pbsa_input(), encoding="utf-8")
        _run(
            [
                str(provider.executables["pbsa"]),
                "-O",
                "-i",
                pb_input.name,
                "-o",
                pb_output.name,
                "-p",
                prmtop.name,
                "-c",
                inpcrd.name,
            ],
            cwd=work,
            timeout=provider.timeout,
            label="legacy-pbsa",
        )
        polar = parse_gbnsr6_components(gb_output.read_text(encoding="utf-8"))["polar"]
        nonpolar = parse_pbsa_components(pb_output.read_text(encoding="utf-8"))
    components = {
        "polar": float(polar),
        "cavity": float(nonpolar["cavity"]),
        "dispersion": float(nonpolar["dispersion"]),
    }
    components["total"] = float(sum(components.values()))
    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("Legacy reference returned a non-finite component.")
    return components, time.perf_counter() - started


def _prepared_components(result) -> dict[str, float]:
    components = {
        "polar": float(result.components_hartree["polar"] * KCAL_PER_MOL_PER_HARTREE),
        "cavity": float(result.components_hartree["cavity"] * KCAL_PER_MOL_PER_HARTREE),
        "dispersion": float(
            result.components_hartree["dispersion"] * KCAL_PER_MOL_PER_HARTREE
        ),
        "total": float(result.energy_hartree * KCAL_PER_MOL_PER_HARTREE),
    }
    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("Prepared provider returned a non-finite component.")
    return components


def _case_positions(source_positions: np.ndarray, case: dict[str, Any]) -> np.ndarray:
    positions = np.asarray(source_positions, dtype=np.float64) + np.asarray(
        case["translation_angstrom"], dtype=np.float64
    )
    for displacement in case.get("atom_displacements", []):
        index = int(displacement["atom_index"])
        if index >= len(positions):
            raise ValueError(
                f"Coordinate case atom index {index} exceeds {len(positions)} atoms."
            )
        positions[index] += np.asarray(displacement["delta_angstrom"], dtype=np.float64)
    return positions


def run(
    *,
    protocol_path: Path,
    output_path: Path,
    amber_bin: Path,
    timeout: float,
    recorded_date: str,
) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(protocol_path)
    source = protocol["source"]
    source_path = _safe_benchmark_path(source["mol2"])
    if not source_path.is_file() or sha256_file(source_path) != source["mol2_sha256"]:
        raise ValueError("Pinned prepared-provider source MOL2 changed or is missing.")
    atoms = MOL2Reader(str(source_path), charge=0, mult=1)
    provider = AmberToolsChaGB(
        atoms,
        atoms.get_initial_charges(),
        executable=str(amber_bin / "gbnsr6"),
        timeout=timeout,
    )
    provider_provenance = provider.provenance
    try:
        source_positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        records: list[dict[str, Any]] = []
        tolerance = float(protocol["comparison"]["component_parity_tolerance_kcal_mol"])
        for case in protocol["coordinate_cases"]:
            translation = np.asarray(case["translation_angstrom"], dtype=np.float64)
            positions = _case_positions(source_positions, case)
            legacy, legacy_seconds = _legacy_fresh_topology_evaluate(provider, positions)
            started = time.perf_counter()
            prepared_result = provider.evaluate_coordinates(positions)
            prepared_seconds = time.perf_counter() - started
            prepared = _prepared_components(prepared_result)
            delta = {key: prepared[key] - legacy[key] for key in COMPONENTS}
            records.append(
                {
                    "coordinate_id": case["id"],
                    "translation_angstrom": translation.tolist(),
                    "atom_displacements": case.get("atom_displacements", []),
                    "legacy_fresh_topology": {
                        "components_kcal_mol": legacy,
                        "wall_seconds": legacy_seconds,
                    },
                    "prepared_coordinate_only": {
                        "components_kcal_mol": prepared,
                        "wall_seconds": prepared_seconds,
                        "prepared_topology": prepared_result.provenance[
                            "prepared_topology"
                        ],
                    },
                    "prepared_minus_legacy_kcal_mol": delta,
                    "maximum_absolute_component_delta_kcal_mol": max(
                        abs(value) for value in delta.values()
                    ),
                }
            )
    finally:
        provider.close()

    if not all(
        record["maximum_absolute_component_delta_kcal_mol"] <= tolerance
        for record in records
    ):
        raise RuntimeError("Prepared topology did not preserve legacy components.")
    cache_hits = [
        bool(record["prepared_coordinate_only"]["prepared_topology"]["cache_hit"])
        for record in records
    ]
    if cache_hits != [False] + [True] * (len(records) - 1):
        raise RuntimeError("Prepared-topology cache behavior changed.")
    legacy_seconds = [record["legacy_fresh_topology"]["wall_seconds"] for record in records]
    prepared_seconds = [
        record["prepared_coordinate_only"]["wall_seconds"] for record in records
    ]
    warm_seconds = prepared_seconds[1:]
    result = seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "recorded_date": recorded_date,
            "claim_scope": protocol["claim_scope"],
            "protocol": {
                "path": protocol_path.relative_to(SCRIPT_DIR).as_posix(),
                "sha256": sha256_file(protocol_path),
                "fingerprint": protocol_sha256,
            },
            "command": command_provenance(
                __file__,
                {
                    "protocol": protocol_path.relative_to(SCRIPT_DIR).as_posix(),
                    "output": output_path.relative_to(SCRIPT_DIR).as_posix(),
                    "amber_bin": str(amber_bin),
                    "timeout": timeout,
                    "recorded_date": recorded_date,
                },
                repository_root=REPOSITORY_ROOT,
            ),
            "source": {
                **source,
                "resolved_path": source_path.relative_to(SCRIPT_DIR).as_posix(),
            },
            "implementation": {
                "prepared_provider": "maple/function/calculator/extra_correction/implicit/amber_chagb.py",
                "prepared_provider_sha256": sha256_file(
                    REPOSITORY_ROOT
                    / "maple/function/calculator/extra_correction/implicit/amber_chagb.py"
                ),
                "legacy_reference": "fresh parmchk2 + tleap per coordinate; GBNSR6 + PBSA unchanged",
                "persistent_external_worker_claimed": False,
            },
            "route_contract": protocol["route_contract"],
            "execution": {
                "profile": provider_provenance["profile"],
                "radii": provider_provenance["radii"],
                "executables": provider_provenance["executables"],
                "gbnsr6_serialization": provider_provenance["execution_control"],
            },
            "prepared_topology": records[0]["prepared_coordinate_only"][
                "prepared_topology"
            ],
            "component_parity": {
                "tolerance_kcal_mol": tolerance,
                "all_components_within_tolerance": True,
                "maximum_absolute_delta_kcal_mol": max(
                    record["maximum_absolute_component_delta_kcal_mol"]
                    for record in records
                ),
                "components": list(COMPONENTS),
            },
            "cache_contract": {
                "expected_cache_hits": [False] + [True] * (len(records) - 1),
                "observed_cache_hits": cache_hits,
                "topology_preparation_count": 1,
                "coordinate_only_evaluation_count": len(records),
            },
            "latency_seconds": {
                "legacy_fresh_topology_by_coordinate": legacy_seconds,
                "prepared_by_coordinate": prepared_seconds,
                "prepared_cold_start": prepared_seconds[0],
                "prepared_warm_by_coordinate": warm_seconds,
                "prepared_warm_mean": float(np.mean(warm_seconds)),
                "prepared_warm_throughput_evaluations_per_second": float(
                    len(warm_seconds) / sum(warm_seconds)
                ),
                "legacy_mean": float(np.mean(legacy_seconds)),
            },
            "records": records,
            "prohibited_interpretations": protocol["prohibited_interpretations"],
        }
    )
    write_json_atomic(output_path, result)
    return result


def validate_artifact(path: str | Path) -> dict[str, Any]:
    artifact = load_json(path)
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type") != ARTIFACT_TYPE
        or artifact.get("content_sha256") != artifact_content_sha256(artifact)
        or artifact.get("component_parity", {}).get("all_components_within_tolerance")
        is not True
        or artifact.get("cache_contract", {}).get("topology_preparation_count") != 1
        or artifact.get("implementation", {}).get("persistent_external_worker_claimed")
        is not False
    ):
        raise ValueError("Prepared-provider parity artifact is invalid.")
    hits = artifact["cache_contract"]["observed_cache_hits"]
    if hits != [False] + [True] * (len(hits) - 1):
        raise ValueError("Prepared-provider artifact has invalid cache evidence.")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=SCRIPT_DIR / "route1_chagb_prepared_provider_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "route1-chagb-prepared-provider-parity-2026-07-29.json",
    )
    parser.add_argument("--amber-bin", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--recorded-date", default="2026-07-29")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        validate_artifact(args.output)
        return
    if args.amber_bin is None:
        parser.error("--amber-bin is required unless --verify is used")
    amber_bin = _resolve_amber_bin(args.amber_bin)
    run(
        protocol_path=args.protocol.resolve(),
        output_path=args.output.resolve(),
        amber_bin=amber_bin,
        timeout=float(args.timeout),
        recorded_date=args.recorded_date,
    )


if __name__ == "__main__":
    main()
