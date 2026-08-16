#!/usr/bin/env python3
"""Run frozen MNSol-10 gates for explicit MACE-POLAR source embeddings.

The public artifact contains aggregate statistics only.  The row-level MNSol
values remain below ``.omx`` because the database is user supplied and is not
redistributed by MAPLE.  This runner deliberately reuses the already frozen,
experiment-blind ten-record selection; it neither selects records nor tunes a
continuum parameter after observing model output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Sequence

from ase import Atoms
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "docs" / "implicit-solvation" / "benchmarks"
for search_path in (REPO_ROOT, BENCHMARK_DIR):
    text = str(search_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from benchmark_core import canonical_json_bytes, sha256_file, write_json_atomic
from mnsol_dataset import load_mnsol_protocol, load_mnsol_v2012
from mnsol_pilot import validate_frozen_mnsol_pilot_selection

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    HARTREE_TO_KCAL_MOL,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.experimental.mace_polar_frozen_ddx import (
    build_smd_mace_polar_frozen_ddx_pes,
    build_smd_mace_polar_frozen_point_ddx_pes,
)
from maple.solvation.models import build_official_mace_polar_1_m_radial_gto_adapter


RADIAL_GTO_EMBEDDING = "radial-gto"
POINT_L1_EMBEDDING = "point-l1"
SUPPORTED_SOURCE_EMBEDDINGS = (RADIAL_GTO_EMBEDDING, POINT_L1_EMBEDDING)
PROFILE_IDENTITIES = {
    RADIAL_GTO_EMBEDDING: {
        "artifact_id": "route2-mnsol10-macepolar-radial-gto-frozen-ddpcm-smd-v1",
        "profile_id": "route2-pure-macepolar-radial-gto-frozen-ddpcm-smd-v1",
        "private_name": "mnsol10-radial-gto-private-v1.json",
        "public_name": "mace-polar-radial-gto-mnsol10-accuracy-v1.json",
    },
    POINT_L1_EMBEDDING: {
        "artifact_id": "route2-mnsol10-macepolar-point-l1-frozen-ddpcm-smd-v1",
        "profile_id": "route2-pure-macepolar-point-l1-frozen-ddpcm-smd-v1",
        "private_name": "mnsol10-point-l1-private-v1.json",
        "public_name": "mace-polar-point-l1-mnsol10-accuracy-v1.json",
    },
}
FULL_PANEL_RECORD_COUNT = 10
TARGET_MAE_KCAL_MOL = 1.5
EV_TO_KCAL_MOL = HARTREE_TO_KCAL_MOL / HARTREE_TO_EV
DEFAULT_PROTOCOL = BENCHMARK_DIR / "route2-mnsol-protocol-v1.json"
DEFAULT_SELECTION = BENCHMARK_DIR / "route2-mnsol-pilot-selection-v1.json"
SOURCE_FILES = (
    "GOAL.md",
    "maple/solvation/api/units.py",
    "maple/solvation/continuum/radial_gto_ddx.py",
    "maple/solvation/continuum/mace_polar_point_ddx.py",
    "maple/solvation/continuum/separated_source_ddx.py",
    "maple/solvation/coupling/exact_gto.py",
    "maple/solvation/coupling/gaussian_multipole_derivatives.py",
    "maple/solvation/coupling/metrics.py",
    "maple/solvation/coupling/spaces.py",
    "maple/solvation/experimental/mace_polar_frozen_ddx.py",
    "maple/solvation/models/mace_polar.py",
    "maple/solvation/solvent_terms.py",
    "docs/implicit-solvation/benchmarks/benchmark_core.py",
    "docs/implicit-solvation/benchmarks/mnsol_dataset.py",
    "docs/implicit-solvation/benchmarks/mnsol_partition.py",
    "docs/implicit-solvation/benchmarks/mnsol_pilot.py",
    "docs/implicit-solvation/benchmarks/route2-mnsol-protocol-v1.json",
    "docs/implicit-solvation/benchmarks/route2-mnsol-pilot-selection-v1.json",
    "tools/route2_release/run_mace_polar_radial_gto_mnsol10.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _require_private_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    private_root = (REPO_ROOT / ".omx").resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain below {private_root}.") from exc
    return resolved


def _profile_identity(source_embedding: str) -> dict[str, str]:
    try:
        return PROFILE_IDENTITIES[source_embedding]
    except KeyError as exc:  # pragma: no cover - argparse owns the public path
        raise ValueError(f"Unsupported source embedding: {source_embedding!r}.") from exc


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    identity = _profile_identity(args.source_embedding)
    private = (
        REPO_ROOT / ".omx" / "route2" / identity["private_name"]
        if args.private_output is None
        else Path(args.private_output)
    )
    public = (
        REPO_ROOT / "docs" / "route2" / "evidence" / identity["public_name"]
        if args.public_output is None
        else Path(args.public_output)
    )
    return private, public


def _git_identity() -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "head": head,
        "dirty": bool(status),
        "porcelain_sha256": _sha256_bytes(status.encode("utf-8")),
    }


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source identity file is missing: {path}")
        result[relative] = sha256_file(path)
    return result


def _atoms(selected: object) -> Atoms:
    item = selected.eligible_record
    geometry = item.geometry
    return Atoms(
        numbers=np.asarray(geometry.atomic_numbers, dtype=int),
        positions=np.asarray(geometry.coordinates_angstrom, dtype=float),
        info={"charge": int(item.record.charge), "mult": int(geometry.multiplicity)},
    )


def _metrics(records: Sequence[dict[str, object]]) -> dict[str, object]:
    if len(records) != FULL_PANEL_RECORD_COUNT:
        raise ValueError(
            f"MNSol-10 aggregate requires {FULL_PANEL_RECORD_COUNT} records."
        )
    errors = np.asarray([row["signed_error_kcal_mol"] for row in records], dtype=float)
    predictions = np.asarray(
        [row["predicted_delta_g_kcal_mol"] for row in records], dtype=float
    )
    experiment = np.asarray(
        [row["experimental_delta_g_kcal_mol"] for row in records], dtype=float
    )
    if not (
        np.all(np.isfinite(errors))
        and np.all(np.isfinite(predictions))
        and np.all(np.isfinite(experiment))
    ):
        raise ValueError("MNSol-10 metrics require finite values.")
    absolute = np.abs(errors)
    mae = float(np.mean(absolute))
    return {
        "record_count": len(records),
        "solvent_count": len({str(row["canonical_solvent"]) for row in records}),
        "mean_signed_error_kcal_mol": float(np.mean(errors)),
        "mean_absolute_error_kcal_mol": mae,
        "root_mean_square_error_kcal_mol": float(np.sqrt(np.mean(errors**2))),
        "maximum_absolute_error_kcal_mol": float(np.max(absolute)),
        "mean_predicted_delta_g_kcal_mol": float(np.mean(predictions)),
        "mean_experimental_delta_g_kcal_mol": float(np.mean(experiment)),
        "mean_polarization_kcal_mol": float(
            np.mean([row["polarization_kcal_mol"] for row in records])
        ),
        "mean_cds_kcal_mol": float(
            np.mean([row["cds_kcal_mol"] for row in records])
        ),
        "target_mae_kcal_mol": TARGET_MAE_KCAL_MOL,
        "target_mae_pass": mae <= TARGET_MAE_KCAL_MOL,
    }


def _runtime() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in (
                "ase",
                "mace-torch",
                "numpy",
                "pyddx",
                "pyscf",
                "scipy",
                "torch",
            )
        },
    }


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    source_embedding = str(args.source_embedding)
    identity = _profile_identity(source_embedding)
    artifact_id = identity["artifact_id"]
    profile_id = identity["profile_id"]
    source = Path(args.source).resolve()
    protocol_path = Path(args.protocol).resolve()
    selection_path = Path(args.selection).resolve()
    private_argument, public_argument = _output_paths(args)
    private_output = _require_private_path(
        private_argument, label="row-level MNSol output"
    )
    public_output = public_argument.resolve()
    if public_output == private_output:
        raise ValueError("Public and private MNSol outputs must be distinct.")

    protocol = load_mnsol_protocol(protocol_path)
    dataset = load_mnsol_v2012(source, protocol)
    selection_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    selection = validate_frozen_mnsol_pilot_selection(
        selection_manifest, dataset, protocol
    )
    if len(selection) != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError("The frozen pilot no longer contains exactly ten records.")

    import torch

    torch.set_num_threads(int(args.torch_threads))
    started = time.perf_counter()
    model_started = time.perf_counter()
    model = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device,
        checkpoint_path=args.checkpoint,
    )
    model_load_seconds = time.perf_counter() - model_started

    rows: list[dict[str, object]] = []
    configuration_hashes: dict[str, str] = {}
    for ordinal, selected in enumerate(selection, start=1):
        atoms = _atoms(selected)
        item = selected.eligible_record
        model.domain.validate_atoms(atoms)
        print(
            f"[{ordinal}/{len(selection)}] solvent={selected.canonical_solvent} "
            f"atoms={len(atoms)}",
            flush=True,
        )
        builder = (
            build_smd_mace_polar_frozen_ddx_pes
            if source_embedding == RADIAL_GTO_EMBEDDING
            else build_smd_mace_polar_frozen_point_ddx_pes
        )
        pes = builder(
            model,
            atoms.get_chemical_symbols(),
            solvent=selected.canonical_solvent,
            lmax=args.lmax,
            n_lebedev=args.n_lebedev,
            solver_tolerance=args.solver_tolerance,
            eta=args.eta,
            n_proc=args.n_proc,
        )
        sample_started = time.perf_counter()
        state = pes.solve(atoms)
        wall_seconds = time.perf_counter() - sample_started
        predicted = state.solvation_energy_eV * EV_TO_KCAL_MOL
        experimental = float(item.record.delta_g_kcal_mol)
        configuration_hashes[selected.canonical_solvent] = pes.configuration_sha256()
        rows.append(
            {
                "selection_index": ordinal - 1,
                "canonical_solvent": selected.canonical_solvent,
                "mnsol_solvent": item.record.solvent,
                "partition": item.partition,
                "opaque_record_id": selected.opaque_record_id,
                "entry_number": item.record.entry_number,
                "geometry_handle": item.record.geometry_handle,
                "geometry_sha256": item.geometry.sha256,
                "solute_name": item.record.solute_name,
                "formula": item.record.formula,
                "atom_count": len(atoms),
                "experimental_delta_g_kcal_mol": experimental,
                "predicted_delta_g_kcal_mol": predicted,
                "signed_error_kcal_mol": predicted - experimental,
                "absolute_error_kcal_mol": abs(predicted - experimental),
                "polarization_kcal_mol": (
                    state.polarization_energy_eV * EV_TO_KCAL_MOL
                ),
                "cds_kcal_mol": state.cds_energy_eV * EV_TO_KCAL_MOL,
                "source_total_charge_e": state.source_total_charge_e,
                "state_sha256": state.state_sha256,
                "continuum_state_sha256": state.continuum_state_sha256,
                "solvent_term_state_sha256": state.solvent_term_state_sha256,
                "topology_id": state.topology_id,
                "wall_seconds": wall_seconds,
            }
        )

    aggregate = _metrics(rows)
    if aggregate["solvent_count"] != FULL_PANEL_RECORD_COUNT:
        raise RuntimeError("The frozen MNSol-10 panel must retain ten solvents.")
    maximum_charge_error = max(abs(float(row["source_total_charge_e"])) for row in rows)
    if maximum_charge_error > 2.0e-12:
        raise RuntimeError("A frozen MACE-POLAR source violated neutral charge.")

    total_wall_seconds = time.perf_counter() - started
    source_hashes = _source_hashes()
    profile_payload = {
        "artifact_id": artifact_id,
        "profile_id": profile_id,
        "source_embedding": source_embedding,
        "model_configuration_sha256": model.configuration_sha256(),
        "model_provenance_sha256": model.provenance_sha256,
        "continuum_model": "pcm",
        "lmax": args.lmax,
        "n_lebedev": args.n_lebedev,
        "solver_tolerance": args.solver_tolerance,
        "eta": args.eta,
        "n_proc": args.n_proc,
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "protocol_fingerprint": protocol.fingerprint,
        "source_files_sha256": source_hashes,
    }
    profile_sha256 = _sha256_bytes(canonical_json_bytes(profile_payload))
    checkpoint = {
        "sha256": model.provenance.checkpoint_sha256,
        "model_family": model.provenance.model_family,
        "model_profile_id": model.provenance.model_profile_id,
        "upstream_version": model.provenance.upstream_version,
        "upstream_commit": model.provenance.upstream_commit,
        "dtype": model.provenance.dtype,
        "device": model.provenance.device,
        "domain": asdict(model.provenance.domain),
    }
    common = {
        "artifact_id": artifact_id,
        "schema_version": 1,
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "git": _git_identity(),
        "dataset": {
            "name": "MNSol",
            "version": "2012",
            "table_sha256": dataset.table_sha256,
            "normalized_bundle_sha256": dataset.normalized_bundle_sha256,
            "temperature_k": protocol.temperature_k,
            "standard_state": protocol.standard_state,
        },
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint,
        "selection_artifact": selection_path.name,
        "selection_artifact_sha256": sha256_file(selection_path),
        "selection_fingerprint": selection_manifest["selection_fingerprint"],
        "checkpoint": checkpoint,
        "scientific_identity": {
            "solute_model": "official unmodified MACE-POLAR-1-M checkpoint",
            "solute_source": (
                "zero-field learned l<=1 source evaluated as the checkpoint "
                "sigma=1.5-A Gaussian density"
                if source_embedding == RADIAL_GTO_EMBEDDING
                else "zero-field learned l<=1 moments evaluated through a "
                "no-fit point-multipole SourceEmbedding"
            ),
            "source_embedding": source_embedding,
            "continuum": (
                "pyddx ddPCM radial-GTO general source"
                if source_embedding == RADIAL_GTO_EMBEDDING
                else "pyddx ddPCM point-l<=1 multipole source"
            ),
            "cavity": "PySCF-2.13.1 SMD solvent-dependent Coulomb radii",
            "dielectric": "PySCF-2.13.1 SMD solvent registry",
            "nonpolar": "PySCF-2.13.1 SMD-CDS",
            "energy_ledger": "G_ddPCM[frozen zero-field source] + G_SMD-CDS",
            "field_conditioned_model_energy_included": False,
            "mutual_ml_continuum_polarization": False,
            "strict_tier_v": False,
        },
        "continuum_parameters": {
            "model": "pcm",
            "lmax": args.lmax,
            "n_lebedev": args.n_lebedev,
            "solver_tolerance": args.solver_tolerance,
            "eta": args.eta,
            "n_proc": args.n_proc,
        },
        "aggregate_metrics": aggregate,
        "gates": {
            "complete_frozen_panel": len(rows) == FULL_PANEL_RECORD_COUNT,
            "ten_distinct_solvents": aggregate["solvent_count"]
            == FULL_PANEL_RECORD_COUNT,
            "source_charge": maximum_charge_error <= 2.0e-12,
            "mae_at_most_1_5_kcal_mol": aggregate["target_mae_pass"],
        },
        "configuration_sha256_by_solvent": configuration_hashes,
        "source_files_sha256": source_hashes,
        "runtime": _runtime(),
        "timing_seconds": {
            "model_load": model_load_seconds,
            "record_total": float(sum(float(row["wall_seconds"]) for row in rows)),
            "total_wall": total_wall_seconds,
        },
        "claim_boundary": (
            "Frozen experiment-blind MNSol-10 multisolvent accuracy gate for "
            f"the exact pure MACE-POLAR {source_embedding} ddPCM plus SMD-CDS "
            "profile. Passing the 1.5 kcal/mol MAE target does not by itself "
            "admit broad MNSol generalization, self-consistent MACE-POLAR, "
            "hybrid MACE-MDP+MACE-POLAR, Tier V, forces, Hessians, OPT, or MD."
        ),
    }
    public = {
        **common,
        "visibility": "public-aggregate-only",
        "row_level_data_emitted": False,
        "record_count": len(rows),
        "partition_counts": dict(Counter(str(row["partition"]) for row in rows)),
        "status": "pass" if all(common["gates"].values()) else "fail",
    }
    private = {
        **common,
        "visibility": "private-user-supplied-mnsol-row-level",
        "do_not_commit": True,
        "public_output": str(public_output),
        "records": rows,
        "status": public["status"],
    }
    write_json_atomic(private_output, private)
    write_json_atomic(public_output, public)
    return private, public


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--source-embedding",
        choices=SUPPORTED_SOURCE_EMBEDDINGS,
        default=RADIAL_GTO_EMBEDDING,
    )
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--device", default=os.environ.get("MAPLE_ROUTE2_MACE_DEVICE", "cpu"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--lmax", type=int, default=15)
    parser.add_argument("--n-lebedev", type=int, default=1202)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--n-proc", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _private, public = run(args)
    print(json.dumps(public["aggregate_metrics"], indent=2, sort_keys=True))
    _private_path, public_path = _output_paths(args)
    print(f"public artifact: {public_path.resolve()}")
    return 0 if public["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
