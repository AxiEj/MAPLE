#!/usr/bin/env python3
"""Build a label-free structural series manifest for Route 1 ranking audits.

The current FreeSolv diagnostic uses deterministic RDKit Bemis--Murcko
scaffolds.  It is deliberately *not* called a binding/congeneric-series
manifest: a real binding benchmark must provide target, pose, protonation, and
series membership independently before the same ranking contract can certify a
claim.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
import sys
from typing import Any

import rdkit
from rdkit import Chem, RDLogger

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_core as core  # noqa: E402
from maple.function.read.filereader.mol2_reader import MOL2Reader  # noqa: E402

_murcko_scaffold = import_module("rdkit.Chem.Scaffolds.MurckoScaffold")
_get_scaffold_for_mol = _murcko_scaffold.GetScaffoldForMol  # type: ignore[attr-defined]

DEFAULT_PROTOCOL = SCRIPT_DIR / "route1_rank_protocol_v1.json"
DEFAULT_SOURCE_MANIFEST = SCRIPT_DIR / "route1_freesolv_reserve_source_manifest.json"
DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT / ".omx/benchmarks/neutral-water-freesolv-route1-20260723"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "route1-freesolv-reserve-rank-series-2026-07-29.json"
)

_BOND_TYPES = {
    "1": Chem.BondType.SINGLE,
    "2": Chem.BondType.DOUBLE,
    "3": Chem.BondType.TRIPLE,
    "ar": Chem.BondType.AROMATIC,
}
_FORBIDDEN_LABEL_FIELD_NAMES = frozenset(
    {
        "experimental_kcal_mol",
        "experimental_uncertainty_kcal_mol",
        "signed_errors_kcal_mol",
        "predictions_kcal_mol",
        "hydration_free_energy",
    }
)


def _relative_to_repository(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _reject_label_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_LABEL_FIELD_NAMES:
                raise ValueError(f"Structure source manifest contains label field {key!r}.")
            _reject_label_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_label_fields(nested)


def _rdkit_from_mol2(path: Path):
    """Create an RDKit graph from the pinned MOL2 topology, not its labels."""
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    graph = Chem.RWMol()
    for symbol in atoms.get_chemical_symbols():
        graph.AddAtom(Chem.Atom(symbol))
    for left, right, bond_label in atoms.info["mol2"]["bonds"]:
        token = str(bond_label).lower()
        try:
            bond_type = _BOND_TYPES[token]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported MOL2 bond type {bond_label!r} in {path.name}."
            ) from exc
        graph.AddBond(int(left), int(right), bond_type)
        if token == "ar":
            graph.GetAtomWithIdx(int(left)).SetIsAromatic(True)
            graph.GetAtomWithIdx(int(right)).SetIsAromatic(True)
    molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:  # RDKit's exception hierarchy is not stable.
        raise ValueError(f"RDKit cannot sanitize pinned MOL2 {path.name}: {exc}") from exc
    return Chem.RemoveHs(molecule)


def _series_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = core.load_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Only Route 1 ranking protocol schema version 1 is supported.")
    if protocol.get("protocol_id") != "maple-route1-rank-contract-v1":
        raise ValueError("Unexpected Route 1 ranking protocol id.")
    return protocol, core.sha256_bytes(core.canonical_json_bytes(protocol))


def build_series_manifest(
    *,
    source_manifest: str | Path,
    source_root: str | Path,
    expected_source_manifest_sha256: str,
    min_series_size: int,
    protocol_id: str,
    protocol_sha256: str,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    """Derive scaffold memberships only from hash-pinned MOL2 structures."""
    source_manifest_path = Path(source_manifest).resolve()
    if core.sha256_file(source_manifest_path) != expected_source_manifest_sha256:
        raise ValueError("Pinned label-free source manifest hash mismatch.")
    manifest = core.load_json(source_manifest_path)
    _reject_label_fields(manifest)
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Structure source manifest requires non-empty records.")
    if int(manifest.get("case_count", -1)) != len(records):
        raise ValueError("Structure source manifest case_count does not reconcile.")
    ids = [str(row.get("compound_id", "")) for row in records]
    if not all(ids) or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Structure source manifest ids must be sorted and unique.")
    if min_series_size < 2:
        raise ValueError("A ranking series must contain at least two members.")

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    disable_log = getattr(RDLogger, "DisableLog", None)
    if callable(disable_log):
        disable_log("rdApp.*")
    derived_records: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        relative = Path(str(row.get("source_mol2_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Structure source manifest contains an unsafe MOL2 path.")
        mol2_path = (root / relative).resolve()
        try:
            mol2_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Structure MOL2 path escapes the declared source root.") from exc
        expected_hash = str(row.get("source_mol2_sha256", ""))
        if core.sha256_file(mol2_path) != expected_hash:
            raise ValueError(f"Pinned MOL2 hash mismatch: {row['compound_id']}.")
        molecule = _rdkit_from_mol2(mol2_path)
        scaffold = _get_scaffold_for_mol(molecule)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        derived = {
            "compound_id": row["compound_id"],
            "source_mol2_sha256": expected_hash,
            "scaffold_sha256": (
                core.sha256_bytes(scaffold_smiles.encode("utf-8"))
                if scaffold_smiles
                else None
            ),
            "scaffold_heavy_atom_count": int(scaffold.GetNumHeavyAtoms()),
        }
        derived_records.append(derived)
        if scaffold_smiles:
            groups.setdefault(str(derived["scaffold_sha256"]), []).append(derived)

    series: list[dict[str, Any]] = []
    series_ids: dict[str, str] = {}
    for scaffold_hash, members in sorted(groups.items()):
        members.sort(key=lambda row: str(row["compound_id"]))
        if len(members) < min_series_size:
            continue
        series_id = f"murcko-scaffold-v1:{scaffold_hash}"
        series_ids[scaffold_hash] = series_id
        series.append(
            {
                "series_id": series_id,
                "series_kind": "structure-derived-murcko-scaffold-diagnostic",
                "scaffold_sha256": scaffold_hash,
                "scaffold_heavy_atom_count": members[0]["scaffold_heavy_atom_count"],
                "member_count": len(members),
                "member_ids": [row["compound_id"] for row in members],
            }
        )

    for derived in derived_records:
        scaffold_hash = derived["scaffold_sha256"]
        if scaffold_hash is None:
            derived["series_id"] = None
            derived["unassigned_reason"] = "empty_murcko_scaffold"
        elif str(scaffold_hash) not in series_ids:
            derived["series_id"] = None
            derived["unassigned_reason"] = "singleton_murcko_scaffold"
        else:
            derived["series_id"] = series_ids[str(scaffold_hash)]

    artifact = core.seal_artifact(
        {
            "schema_version": 1,
            "artifact_type": "route1-rank-label-free-structure-series",
            "protocol_id": protocol_id,
            "protocol_sha256": protocol_sha256,
            "protocol_fingerprint": protocol_fingerprint,
            "source_manifest_sha256": expected_source_manifest_sha256,
            "series_definition": {
                "kind": "rdkit-bemis-murcko-from-pinned-mol2-v1",
                "rdkit_version": rdkit.__version__,
                "min_series_size": min_series_size,
                "labels_read": False,
                "binding_or_congeneric_series_proven": False,
            },
            "case_count": len(derived_records),
            "evaluable_series_count": len(series),
            "evaluable_member_count": sum(item["member_count"] for item in series),
            "records": derived_records,
            "series": series,
        }
    )
    if "experimental" in json.dumps(artifact, sort_keys=True).lower():
        raise AssertionError("Label-free series artifact leaked experimental content.")
    return artifact


def build(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = Path(args.protocol).resolve()
    protocol, fingerprint = _series_protocol(protocol_path)
    evidence = protocol["source_evidence"]
    derivation = protocol["series_derivation"]
    observed_rdkit = rdkit.__version__
    if observed_rdkit != derivation["rdkit_required_version"]:
        raise ValueError(
            "RDKit version mismatch: "
            f"expected {derivation['rdkit_required_version']}, observed {observed_rdkit}."
        )
    artifact = build_series_manifest(
        source_manifest=args.source_manifest,
        source_root=args.source_root,
        expected_source_manifest_sha256=evidence["source_manifest_sha256"],
        min_series_size=int(derivation["min_series_size"]),
        protocol_id=protocol["protocol_id"],
        protocol_sha256=core.sha256_file(protocol_path),
        protocol_fingerprint=fingerprint,
    )
    output = Path(args.output).resolve()
    artifact["command_provenance"] = core.command_provenance(
        __file__,
        {
            "phase": "build-label-free-structure-series",
            "protocol": _relative_to_repository(protocol_path),
            "source_manifest": _relative_to_repository(Path(args.source_manifest)),
            "source_root": _relative_to_repository(Path(args.source_root)),
            "output": _relative_to_repository(output),
        },
        repository_root=REPOSITORY_ROOT,
    )
    artifact = core.seal_artifact(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    core.write_json_atomic(output, artifact)
    print(
        json.dumps(
            {
                "series_artifact": str(output),
                "file_sha256": core.sha256_file(output),
                "content_sha256": artifact["content_sha256"],
                "evaluable_series_count": artifact["evaluable_series_count"],
                "evaluable_member_count": artifact["evaluable_member_count"],
            },
            indent=2,
        )
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
