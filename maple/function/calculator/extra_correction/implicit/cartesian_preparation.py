"""Topology-only Cartesian inference, with exact source-coordinate restoration.

AmberTools 26 topology inference passes through 0.001 A AC serialization.
That intermediate is permitted ONLY for topology inference, never as the
charge-generation input geometry. The
normal MOL2 charge-provider precision/identity checks remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from maple.function.read.filereader.mol2_reader import MOL2Reader
from .ambertools import ResolvedAmberTools, file_sha256, run_ambertools

INFERENCE_WARNING = (
    "Topology and GAFF2 atom/bond types were inferred by AmberTools from Cartesian "
    "coordinates, not supplied by the user. Verify chemical connectivity and "
    "protonation; no atoms/hydrogens were added, removed or reordered. Original "
    "MLIP/solvent coordinates stay exact (geometry=keep). Charges use standard "
    "AM1 optimization starting from the exact original coordinates, not an AM1 "
    "single point at unchanged geometry. Optimized and downstream serialized "
    "geometries are separately audited."
)
# Separate topology-only intermediate bound; NOT a charge-provider tolerance.
GCRT_TOPOLOGY_ROUNDING_HALF_WIDTH_ANGSTROM = 5.0e-4


def validate_cartesian_domain(atoms, options: dict[str, Any]) -> None:
    if options.get("geometry", "keep") != "keep":
        raise ValueError(
            "Cartesian implicit-solvent preparation requires geometry=keep; geometry changes are not permitted."
        )
    if np.any(atoms.pbc):
        raise ValueError("Cartesian implicit-solvent preparation is non-periodic.")
    if (
        not len(atoms)
        or not np.isfinite(atoms.positions).all()
        or np.any(atoms.numbers <= 0)
    ):
        raise ValueError(
            "Cartesian preparation requires finite coordinates and explicit elements for every atom."
        )


def preparation_record(atoms) -> dict[str, Any]:
    geometry = {
        "elements": atoms.get_chemical_symbols(),
        "positions_angstrom": atoms.positions.tolist(),
        "charge": int(atoms.info["charge"]),
        "multiplicity": int(atoms.info["mult"]),
    }
    encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "status": "running",
        "warning": INFERENCE_WARNING,
        "source": dict(
            atoms.info.get("_maple_cartesian_source", {"kind": "Cartesian-Atoms"})
        ),
        "source_geometry": geometry,
        "source_geometry_sha256": hashlib.sha256(encoded).hexdigest(),
        "topology_only_gcrt_rounding_half_width_angstrom": GCRT_TOPOLOGY_ROUNDING_HALF_WIDTH_ANGSTROM,
        "charge_input_geometry": "exact-original-Cartesian-SQM-starting-geometry",
        "charge_evaluation_geometry": "standard-AM1-optimized-geometry-not-original-MLIP-geometry",
        "atom_rewrite_permitted": False,
        "charge_geometry_validated": False,
    }


def write_preparation_record(audit_dir: Path, record: dict[str, Any]) -> None:
    (audit_dir / "preparation.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def canonical_mol2(metadata: dict[str, Any], positions, charges) -> str:
    """Serialize newly inferred topology with round-trip float64 geometry/charges.

    This serializer is ONLY for new Cartesian inputs. Existing user MOL2
    serialization, identity, atom-type and topology semantics are unchanged.
    """
    lines = [
        "@<TRIPOS>MOLECULE",
        "MAPLE_PREPARED",
        f'{len(positions)} {len(metadata["bonds"])} 1 0 0',
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for index, (position, charge) in enumerate(zip(positions, charges)):
        xyz = " ".join(format(float(value), ".17g") for value in position)
        lines.append(
            f'{metadata["atom_ids"][index]} {metadata["atom_names"][index]} {xyz} '
            f'{metadata["atom_types"][index]} 1 MOL {float(charge):.17g}'
        )
    lines.append("@<TRIPOS>BOND")
    for bond_id, (first, second, bond_type) in enumerate(metadata["bonds"], 1):
        lines.append(
            f'{bond_id} {metadata["atom_ids"][int(first)]} '
            f'{metadata["atom_ids"][int(second)]} {bond_type}'
        )
    lines += ["@<TRIPOS>SUBSTRUCTURE", "1 MOL 1 TEMP 0 **** **** 0 ROOT"]
    return "\n".join(lines) + "\n"


def _validate_inferred_topology(source, inferred) -> None:
    metadata = inferred.info["mol2"]
    if len(inferred) != len(source) or metadata["atom_ids"] != list(
        range(1, len(source) + 1)
    ):
        raise ValueError(
            "Antechamber topology inference changed atom IDs, count or order."
        )
    if not np.array_equal(inferred.numbers, source.numbers):
        raise ValueError(
            "Antechamber topology inference changed an atom element or order."
        )
    error = np.abs(inferred.positions - source.positions)
    slack = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(source.positions)) * 8
    if np.any(error > GCRT_TOPOLOGY_ROUNDING_HALF_WIDTH_ANGSTROM + slack):
        raise ValueError(
            "Antechamber topology coordinates changed beyond the topology-only GCRT rounding bound; atom order is unsafe."
        )
    # MOL2Reader already enforces one connected graph, bounded references, and
    # unique atom IDs/bonds. Explicit hydrogens need one single-bond neighbor;
    # no hydrogen addition, removal or protonation repair is permitted.
    neighbors: list[list[int]] = [[] for _ in source]
    for first, second, bond_type in metadata["bonds"]:
        i, j = int(first), int(second)
        neighbors[i].append(j)
        neighbors[j].append(i)

        if (source.numbers[i] == 1 or source.numbers[j] == 1) and str(bond_type) != "1":
            raise ValueError("Inferred hydrogen bond must have a single bond order.")
    for index in np.flatnonzero(source.numbers == 1):
        if len(neighbors[index]) != 1:
            raise ValueError(
                "Every explicit hydrogen must have exactly one inferred bonded neighbor; no hydrogen repair is permitted."
            )


def infer_cartesian_topology(
    atoms,
    options: dict[str, Any],
    audit_dir: Path,
    tool: ResolvedAmberTools,
    installation: dict[str, Any],
    record: dict[str, Any],
):
    source = record["source"]
    if source.get("path") and source.get("sha256"):
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(
                "Cartesian source file changed after input parsing; refusing preparation."
            )
    snapshot = audit_dir / "source-geometry.json"
    snapshot.write_text(
        json.dumps(record["source_geometry"], indent=2), encoding="utf-8"
    )
    gcrt = audit_dir / "topology.input.gcrt"
    rows = [
        "# Cartesian topology inference only; no Gaussian execution",
        "",
        "MAPLE topology",
        "",
        "0 1",
    ]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        rows.append(
            symbol + " " + " ".join(format(float(value), ".17g") for value in position)
        )
    gcrt.write_text("\n".join(rows) + "\n\n", encoding="utf-8")
    output = audit_dir / "topology.inferred.mol2"
    command = [
        tool.executable,
        "-i",
        str(gcrt),
        "-fi",
        "gcrt",
        "-o",
        str(output),
        "-fo",
        "mol2",
        "-j",
        "4",
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
        "-pf",
        "y",
    ]
    record.update(
        installation=installation,
        discovery=tool.discovery,
        command=command,
        input_gcrt=str(gcrt),
        input_gcrt_sha256=file_sha256(gcrt),
        source_geometry_file=str(snapshot),
        source_geometry_file_sha256=file_sha256(snapshot),
    )
    write_preparation_record(audit_dir, record)
    run_ambertools(
        command,
        installation=tool,
        audit_dir=audit_dir,
        stem="topology",
        timeout=float(options.get("timeout", 3600.0)),
        provenance=installation,
    )
    record.update(inferred_mol2=str(output), inferred_mol2_sha256=file_sha256(output))
    inferred = MOL2Reader(str(output), charge=0, mult=1)
    _validate_inferred_topology(atoms, inferred)
    canonical = audit_dir / "topology.original-coordinates.mol2"
    canonical.write_text(
        canonical_mol2(inferred.info["mol2"], atoms.positions, np.zeros(len(atoms))),
        encoding="utf-8",
    )
    prepared = MOL2Reader(str(canonical), charge=0, mult=1)
    if prepared.positions.tobytes() != atoms.positions.tobytes():
        raise ValueError(
            "Canonical topology did not preserve exact source coordinates."
        )
    record.update(
        original_coordinates_restored=True,
        connected_graph_verified=True,
        hydrogen_neighbors_verified=True,
        canonical_mol2=str(canonical),
        canonical_mol2_sha256=file_sha256(canonical),
        maximum_topology_intermediate_coordinate_error_angstrom=float(
            np.max(np.abs(inferred.positions - atoms.positions))
        ),
    )
    write_preparation_record(audit_dir, record)
    return prepared
