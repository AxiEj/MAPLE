#!/usr/bin/env python3
"""Label-free, failure-complete execution on the frozen MNSol development inputs.

The archive's experimental table is never opened.  Outputs are geometry-level
model quantities, not experimental free-energy errors.  Restricted data and
row-level artifacts belong in a private output directory, not in Git.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
from zipfile import ZipFile

import numpy as np
from ase import Atoms

ARCHIVE_SHA = "6e2b25175d41f5c0a319ca7ec50c4586fe88ecfbd096264811fc1e8e77d6c4cc"
SELECTION_SHA = "d05130aa8f38d0d1c292b43d769bb021f0b6ce7d2c29cb0af7479246913ec6a4"
ALIASES = {
    "hexane": "n-hexane",
    "dimethylformamide": "dmf",
    "dimethylsulfoxide": "dmso",
    "tetrahydrofuran": "thf",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--geometry-parser-root", type=Path, required=True)
    parser.add_argument("--anysolv-source", type=Path, required=True)
    parser.add_argument("--model", choices=("anysolv", "consolv"), default="anysolv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output / "records.jsonl"
    if ledger_path.exists():
        raise FileExistsError(
            "Use a new output directory; existing ledgers are never overwritten."
        )
    archive = args.inputs / "MNSolDatabase_v2012.zip"
    selection_path = args.inputs / "route2-mnsol-development-selection-v1.private.json"
    for path, expected in ((archive, ARCHIVE_SHA), (selection_path, SELECTION_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    selection = json.loads(selection_path.read_text())
    rows = selection["selected_records"]
    if len(rows) != 505 or any(row["partition"] != "development" for row in rows):
        raise ValueError("Only the exact 505-record development partition is accepted.")
    sys.path.insert(0, str(args.geometry_parser_root.resolve()))
    from mnsol_dataset import _parse_geometry
    from maple.function.calculator.extra_correction.implicit.anysolv import (
        AnySolvBackend,
    )

    if args.model == "consolv":
        import torch
        from maple.function.calculator.consolv._consolv_calculator import (
            ConSolvCalculator,
            CONSOLV_SOLVENTS,
            CONSOLV_DEFAULT_MODEL_PATH,
        )
        from maple.function.calculator.consolv._torch_model import (
            load_consolv_torch_checkpoint,
        )

        vacuum_path = CONSOLV_DEFAULT_MODEL_PATH.parent / "consolv-vacuum-torch-v1.pt"
        if (
            digest(vacuum_path)
            != "d0369560cd34312604d8f4df3a002c7544c8867236ce50a6808c75794a5195f0"
        ):
            raise ValueError("ConSolv paired vacuum checkpoint mismatch.")
        vacuum_model = load_consolv_torch_checkpoint(
            vacuum_path, expected_state="vacuum"
        )
        vacuum_energies = {}

    wanted = {row["geometry_sha256"] for row in rows}
    geometries = {}
    with ZipFile(archive) as zf:
        for member in zf.infolist():
            parts = PurePosixPath(member.filename).parts
            if (
                "all_solutes" not in parts
                or "__MACOSX" in parts
                or not member.filename.endswith(".xyz")
            ):
                continue
            payload = zf.read(member)
            key = hashlib.sha256(payload).hexdigest()
            if key in wanted:
                geometry = _parse_geometry(PurePosixPath(member.filename).stem, payload)
                if geometry.charge != 0 or geometry.multiplicity != 1:
                    raise ValueError("Frozen development electronic state mismatch.")
                geometries[key] = geometry
    if set(geometries) != wanted or len(geometries) != 312:
        raise ValueError("Failed to recover all 312 frozen development geometries.")

    started = time.monotonic()
    cls = AnySolvBackend if args.model == "anysolv" else ConSolvCalculator
    source_file = Path(sys.modules[cls.__module__].__file__)
    initial_source_hash = digest(source_file)
    receipt = {
        "model": "anysolv-moe" if args.model == "anysolv" else "consolv-torch-paired",
        "status": "running",
        "planned_records": 505,
        "unique_geometries": 312,
        "archive_sha256": ARCHIVE_SHA,
        "selection_sha256": SELECTION_SHA,
        "selection_fingerprint": selection["selection_fingerprint"],
        "quantity_kind": "geometry_level_energy_correction",
        "experimental_table_opened": False,
        "confirmation_labels_opened": False,
        "accuracy_metrics": None,
        "backend_source_sha256": initial_source_hash,
        "geometry_parser_sha256": digest(
            args.geometry_parser_root / "mnsol_dataset.py"
        ),
    }
    write_json(args.output / "manifest.json", receipt)
    if args.model == "consolv":
        receipt["vacuum_checkpoint_sha256"] = digest(vacuum_path)
        receipt["torch_model_source_sha256"] = digest(
            Path(sys.modules[load_consolv_torch_checkpoint.__module__].__file__)
        )
        write_json(args.output / "manifest.json", receipt)
    print(
        f"STARTED {args.model}: 505 development records / 312 exact geometries",
        flush=True,
    )
    backends = {}
    counts = Counter()
    with ledger_path.open("x", buffering=1) as ledger:
        for i, row in enumerate(rows):
            record = dict(row)
            item_started = time.monotonic()
            try:
                geometry = geometries[row["geometry_sha256"]]
                if len(geometry.atomic_numbers) != row["atom_count"]:
                    raise ValueError("Atom count differs from frozen selection.")
                canonical = row["canonical_solvent"]
                solvent = (
                    ALIASES.get(canonical, canonical)
                    if args.model == "anysolv"
                    else {"tetrahydrofuran": "thf"}.get(canonical, canonical)
                )
                atoms = Atoms(
                    numbers=geometry.atomic_numbers,
                    positions=geometry.coordinates_angstrom,
                    info={"charge": geometry.charge, "mult": geometry.multiplicity},
                )
                if args.model == "anysolv":
                    if solvent not in backends:
                        backends[solvent] = AnySolvBackend(
                            solvent=solvent,
                            source_root=str(args.anysolv_source),
                            device="cpu",
                            execution_task="sp",
                        )
                    result = backends[solvent].evaluate(atoms, need_forces=False)
                    value = float(result.energy_hartree)
                    checkpoint_sha = result.provenance["checkpoint_sha256"]
                else:
                    if solvent not in CONSOLV_SOLVENTS:
                        raise NotImplementedError(f"unsupported_solvent: {canonical}")
                    if solvent not in backends:
                        backends[solvent] = ConSolvCalculator(
                            "cpu", native_solvent=solvent, execution_task="sp"
                        )
                    atoms.calc = backends[solvent]
                    solution_hartree = float(atoms.get_potential_energy())
                    key = row["geometry_sha256"]
                    if key not in vacuum_energies:
                        with torch.no_grad():
                            vacuum_energies[key] = (
                                float(
                                    vacuum_model.energy_ev(
                                        torch.tensor(
                                            geometry.atomic_numbers, dtype=torch.long
                                        ),
                                        torch.tensor(
                                            geometry.coordinates_angstrom,
                                            dtype=torch.float64,
                                        ),
                                        19,
                                    )
                                )
                                / 27.211386245988
                            )
                    value = solution_hartree - vacuum_energies[key]
                    checkpoint_sha = backends[solvent].provenance["checkpoint_sha256"]
                    record.update(
                        solution_energy_hartree=solution_hartree,
                        vacuum_energy_hartree=vacuum_energies[key],
                    )
                if not np.isfinite(value):
                    raise ValueError("Nonfinite correction")
                record.update(
                    status="success",
                    correction_hartree=value,
                    checkpoint_sha256=checkpoint_sha,
                    model_solvent=solvent,
                )
            except Exception as exc:
                # A failure is a retained ledger row, never a missing denominator.
                record.update(
                    status="failed", error_type=type(exc).__name__, error=str(exc)
                )
            record["seconds"] = time.monotonic() - item_started
            counts[record["status"]] += 1
            ledger.write(json.dumps(record, allow_nan=False) + "\n")
            progress = {
                "completed_records": i + 1,
                "planned_records": 505,
                "counts": dict(counts),
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json(args.output / "progress.json", progress)
            if i == 0 or (i + 1) % 25 == 0:
                print(json.dumps(progress), flush=True)
    drift = digest(source_file) != initial_source_hash
    receipt.update(status="complete" if not drift else "source_drift", **progress)
    receipt["source_changed_during_run"] = drift
    receipt["model_provenance_by_solvent"] = {
        s: b.provenance for s, b in backends.items()
    }
    write_json(args.output / "result.json", receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("status", "completed_records", "counts", "elapsed_seconds")
            }
        ),
        flush=True,
    )
    return int(drift or bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
