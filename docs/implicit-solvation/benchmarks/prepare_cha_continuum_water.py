#!/usr/bin/env python3
"""Prepare one fixed water topology for the differentiable CHA study.

The fixed, normalized charges and coordinates remain authoritative.  The
separately audited AmberTools MOL2 contributes GAFF2 atom types only.  This
script runs only ``parmchk2`` and ``tleap`` to construct the topology; it never
runs a charge generator, SQM, QM program, network client, or model downloader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ORIGINAL_ROOT = Path("/home/axie/MAPLE/MAPLE-implicitsolv-route1")
DEFAULT_PROTOCOL = (
    ORIGINAL_ROOT
    / ".omx/benchmarks/route1-cha-analytic-v1-20260930/prospective-protocol-v1.json"
)
DEFAULT_OUTPUT = DEFAULT_PROTOCOL.parent / "water-preparation-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "3fe641f0e9bb8a572e7736b8f3cb3ce2ac8a8f8f29ce6e022a0f8caf805cad64"
)
AMBER_ROOT = Path("/home/axie/miniconda3/envs/maple-ambertools")
AMBER_BIN = AMBER_ROOT / "bin"
PARMCHK2 = AMBER_BIN / "parmchk2"
TLEAP = AMBER_BIN / "tleap"
PARMED_PYTHON = AMBER_BIN / "python"
GAFF2_DAT = AMBER_ROOT / "dat/leap/parm/gaff2.dat"
LEAPRC_GAFF2 = AMBER_ROOT / "dat/leap/cmd/leaprc.gaff2"
SOURCE_LEDGER = (
    ORIGINAL_ROOT / ".omx/benchmarks/route1-torch-full-cha/build_continuum_inputs.py"
)
EXPECTED_SOURCE_LEDGER_SHA256 = (
    "3d8412e85105f159d4e107c050007c100459a62140773ad035f9456c66af4fe1"
)
EXPECTED_NATIVE_SHA256 = {
    "parmchk2": "c2ef0906b71c64e5277673bcba943512daf75ef334ad9421c0045b6b80b6d7cc",
    "tleap": "f1c526e0a24ad1fefc6c82be1279aa33eff5f9557d577c2236aa81437dfb52ac",
    "parmed_python": "344f37820b3545664b93c0dc07f4ed970e8121252d9ada5d2390eb45981891c1",
    "gaff2_dat": "2244b58627a85693776d12a0155a292f72b65b4398a03b87958136ad7460c67f",
    "leaprc_gaff2": "47d749188bf70b2c4b878ce5a66cc3a94a4baaa8c359f725acc795cf35a71900",
}
REQUIRED_SOURCE_KEYS = (
    "fixed_mol2",
    "retyped_mol2",
    "mapping",
    "normalization",
    "preparation",
)
EXPECTED_WATER_TYPES = ("oh", "ho", "ho")
EXPECTED_CHA_RADII = {"O": 1.36 + 0.52, "H": 0.52 + 0.52}
EXPECTED_GAFF2_LJ = {"oh": (1.8200, 0.0930), "ho": (0.3019, 0.0047)}
AMBER_CHARGE_SCALE = 18.2223
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + ("\n" if newline else "")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes(
            {key: item for key, item in value.items() if key != "content_sha256"}
        )
    )


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(canonical_bytes(value, newline=True))


def load_protocol(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("prospective protocol SHA256 differs from the external pin")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "route1-cha-analytic-branch-study-v1":
        raise ValueError("unexpected prospective protocol identity")
    assets = protocol.get("source_assets")
    if not isinstance(assets, dict) or any(
        key not in assets for key in REQUIRED_SOURCE_KEYS
    ):
        raise ValueError("prospective protocol lacks the required source assets")
    return protocol


def load_source_bundle(
    paths: Mapping[str, Path], pins: Mapping[str, str]
) -> dict[str, bytes]:
    bundle: dict[str, bytes] = {}
    for name in REQUIRED_SOURCE_KEYS:
        path = Path(paths[name])
        expected = pins[name]
        if _DIGEST.fullmatch(expected) is None:
            raise ValueError(f"{name} external SHA256 pin is malformed")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{name} source SHA256 mismatch")
        bundle[name] = path.read_bytes()
    return bundle


def parse_mol2(text: str) -> dict[str, Any]:
    atoms: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    section = ""
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if not line.strip():
            continue
        fields = line.split()
        if section == "@<TRIPOS>ATOM":
            if len(fields) < 9:
                raise ValueError("malformed MOL2 atom row")
            atoms.append(
                {
                    "id": int(fields[0]),
                    "name": fields[1],
                    "coordinates_angstrom": [float(item) for item in fields[2:5]],
                    "coordinate_tokens": fields[2:5],
                    "atom_type": fields[5],
                    "substructure_id": int(fields[6]),
                    "substructure_name": fields[7],
                    "charge_e": float(fields[8]),
                    "charge_token": fields[8],
                }
            )
        elif section == "@<TRIPOS>BOND":
            if len(fields) < 4:
                raise ValueError("malformed MOL2 bond row")
            bonds.append(
                {
                    "id": int(fields[0]),
                    "atom1_id": int(fields[1]),
                    "atom2_id": int(fields[2]),
                    "bond_type": fields[3],
                }
            )
    if not atoms:
        raise ValueError("MOL2 contains no atoms")
    return {"atoms": atoms, "bonds": bonds}


def _element(atom: Mapping[str, Any]) -> str:
    match = re.match(r"[A-Za-z]+", str(atom["name"]))
    if match is None:
        raise ValueError("atom name does not identify an element")
    symbol = match.group(0)
    return symbol[0].upper() + symbol[1:].lower()


def merge_verified_gaff2_types(
    fixed: Mapping[str, Any],
    retyped: Mapping[str, Any],
    mapping: list[dict[str, Any]],
    normalization: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    fixed_atoms = fixed["atoms"]
    typed_atoms = retyped["atoms"]
    if len(fixed_atoms) != 3 or len(typed_atoms) != 3 or len(mapping) != 3:
        raise ValueError("water source, retyping receipt, or mapping is incomplete")
    if tuple(atom["atom_type"] for atom in typed_atoms) != EXPECTED_WATER_TYPES:
        raise ValueError("audited water GAFF2 types changed")
    used_charges = [float(value) for value in normalization.get("used_charges_e", [])]
    provider_charges = [
        float(value) for value in normalization.get("provider_charges_e", [])
    ]
    if len(used_charges) != 3 or len(provider_charges) != 3:
        raise ValueError("charge-normalization receipt is incomplete")
    if abs(sum(provider_charges) + 0.001) > 1.0e-12 or abs(sum(used_charges)) > 1.0e-12:
        raise ValueError("charge-normalization receipt changed")

    for index, (source, typed, receipt) in enumerate(
        zip(fixed_atoms, typed_atoms, mapping, strict=True)
    ):
        expected = {
            "input_index": index,
            "provider_index": index,
            "input_atom_id": source["id"],
            "provider_atom_id": typed["id"],
            "input_atom_name": source["name"],
            "provider_atom_name": typed["name"],
            "element": _element(source),
            "input_atom_type": source["atom_type"],
            "provider_atom_type": typed["atom_type"],
            "atom_type_retyped": True,
        }
        if receipt != expected:
            raise ValueError(f"atom mapping receipt changed at index {index}")
        if source["id"] != index + 1 or typed["id"] != index + 1:
            raise ValueError("atom order is not contiguous and identical")
        if source["coordinates_angstrom"] != typed["coordinates_angstrom"]:
            raise ValueError(
                "retyping source coordinates differ from fixed coordinates"
            )
        if abs(float(source["charge_e"]) - used_charges[index]) > 5.0e-13:
            raise ValueError("fixed charge differs from normalization receipt")
    if fixed["bonds"] != retyped["bonds"]:
        raise ValueError("retyping changed the fixed bond graph")

    lines = [
        "@<TRIPOS>MOLECULE",
        "water",
        "3 2 1 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for source, typed in zip(fixed_atoms, typed_atoms, strict=True):
        x, y, z = source["coordinate_tokens"]
        lines.append(
            f"{source['id']} {source['name']} {x} {y} {z} {typed['atom_type']} "
            f"{source['substructure_id']} {source['substructure_name']} {source['charge_token']}"
        )
    lines.append("@<TRIPOS>BOND")
    for bond in fixed["bonds"]:
        lines.append(
            f"{bond['id']} {bond['atom1_id']} {bond['atom2_id']} {bond['bond_type']}"
        )
    lines.extend(["@<TRIPOS>SUBSTRUCTURE", "1 MOL 1 RESIDUE 1 A MOL 1", ""])
    receipt = {
        "mapping_verified": True,
        "normalization_verified": True,
        "provider_charge_tokens_used": False,
        "transferred_fields": ["gaff2_atom_type"],
        "preserved_fields": [
            "atom_id",
            "atom_name",
            "coordinates_angstrom",
            "bond_graph",
            "normalized_fixed_charge_e",
        ],
        "source_charge_sum_e": sum(float(atom["charge_e"]) for atom in fixed_atoms),
        "provider_raw_charge_sum_e": sum(
            float(atom["charge_e"]) for atom in typed_atoms
        ),
    }
    return "\n".join(lines), receipt


def parse_prmtop_charge_field(
    text: str, *, atom_count: int
) -> tuple[list[float], list[str]]:
    lines = text.splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "%FLAG CHARGE"
        )
    except ValueError as exc:
        raise ValueError("prmtop lacks %FLAG CHARGE") from exc
    except StopIteration as exc:
        raise ValueError("prmtop lacks %FLAG CHARGE") from exc
    if start + 1 >= len(lines) or lines[start + 1].strip() != "%FORMAT(5E16.8)":
        raise ValueError("unexpected prmtop CHARGE serialization format")
    tokens: list[str] = []
    for line in lines[start + 2 :]:
        if line.startswith("%FLAG"):
            break
        tokens.extend(
            token
            for offset in range(0, len(line), 16)
            if (token := line[offset : offset + 16]) and token.strip()
        )
    if len(tokens) != atom_count:
        raise ValueError("prmtop CHARGE token count differs from atom count")
    values = [float(token) / AMBER_CHARGE_SCALE for token in tokens]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("prmtop contains a non-finite charge")
    return values, tokens


def build_topology_artifact(
    *,
    atom_ids: list[int],
    atom_names: list[str],
    elements: list[str],
    atomic_numbers: list[int],
    gaff2_types: list[str],
    bonds: list[list[Any]],
    source_charges: list[float],
    effective_charges: list[float],
    source_mol2_sha256: str,
    prepared_prmtop_sha256: str,
    parameter_source_sha256: str,
    bondi_radii: list[float],
    cha_radii: list[float],
    lj_rmin: list[float],
    lj_epsilon: list[float],
) -> dict[str, Any]:
    artifact = {
        "schema_version": 1,
        "profile": "chagb-r6-pbsa-continuum-v1",
        "atom_ids": atom_ids,
        "atom_names": atom_names,
        "elements": elements,
        "atomic_numbers": atomic_numbers,
        "gaff2_types": gaff2_types,
        "bonds": bonds,
        "declared_charge_e": round(sum(source_charges)),
        "source_charges_e": source_charges,
        "effective_charges_e": effective_charges,
        "source_charges_sha256": sha256_bytes(
            canonical_bytes(source_charges, newline=True)
        ),
        "effective_charges_sha256": sha256_bytes(
            canonical_bytes(effective_charges, newline=True)
        ),
        "source_mol2_sha256": source_mol2_sha256,
        "prepared_prmtop_sha256": prepared_prmtop_sha256,
        "parameter_source_sha256": parameter_source_sha256,
        "serialization_profile": "ambertools26-prmtop-charge-5e16.8-v1",
        "bondi_radii_angstrom": bondi_radii,
        "cha_radii_angstrom": cha_radii,
        "lj_rmin_angstrom": lj_rmin,
        "lj_epsilon_kcal_mol": lj_epsilon,
    }
    artifact["content_sha256"] = content_sha256(artifact)
    return artifact


def _run(
    command: list[str], *, cwd: Path, stem: str, allowed: set[str]
) -> dict[str, Any]:
    if command[0] not in allowed:
        raise ValueError(f"refused unapproved executable: {command[0]}")
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    stdout_path = cwd / f"{stem}.stdout.log"
    stderr_path = cwd / f"{stem}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    receipt = {
        "argv": command,
        "cwd": ".",
        "returncode": completed.returncode,
        "stdout_file": stdout_path.name,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_file": stderr_path.name,
        "stderr_sha256": sha256_file(stderr_path),
    }
    write_json_new(cwd / f"{stem}.command.json", receipt)
    if completed.returncode:
        raise RuntimeError(
            f"{stem} failed ({completed.returncode}): "
            + (completed.stderr or completed.stdout)[-1200:].strip()
        )
    return receipt


def _require_empty_nonbon(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "NONBON")
    except StopIteration as exc:
        raise ValueError("parmchk2 frcmod lacks NONBON section") from exc
    if any(line.strip() for line in lines[start + 1 :]):
        raise ValueError("parmchk2 emitted forbidden nonbonded overrides")


def _inpcrd_tokens(path: Path, atom_count: int) -> list[list[str]]:
    flattened: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines()[2:]:
        flattened.extend(
            token.strip()
            for offset in range(0, len(line), 12)
            if (token := line[offset : offset + 12]).strip()
        )
    if len(flattened) < atom_count * 3:
        raise ValueError("inpcrd contains too few coordinate tokens")
    return [flattened[index : index + 3] for index in range(0, atom_count * 3, 3)]


def _read_parmed(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    code = r"""import json, parmed, sys
from parmed.periodic_table import Element
t = parmed.load_file(sys.argv[1], xyz=sys.argv[2])
print(json.dumps({
 "parmed_version": parmed.__version__,
 "python_executable": sys.executable,
 "atoms": [{"index": a.idx, "name": a.name, "atom_type": a.type,
             "atomic_number": int(a.atomic_number), "element": Element[int(a.atomic_number)],
             "charge_e": float(a.charge), "bondi_radius_angstrom": float(a.solvent_radius),
             "rmin_angstrom": float(a.rmin), "epsilon_kcal_mol": float(a.epsilon)} for a in t.atoms],
 "bonds": [[min(b.atom1.idx,b.atom2.idx), max(b.atom1.idx,b.atom2.idx)] for b in t.bonds],
 "coordinates_angstrom": t.coordinates.tolist(),
}, sort_keys=True, separators=(",", ":")))"""
    command = [
        str(PARMED_PYTHON),
        "-c",
        code,
        "molecule.prmtop",
        "molecule.inpcrd",
    ]
    receipt = _run(
        command,
        cwd=case_dir,
        stem="parmed-readback",
        allowed={str(PARMED_PYTHON)},
    )
    native = json.loads((case_dir / "parmed-readback.stdout.log").read_text())
    if native["parmed_version"] != "4.3.1":
        raise ValueError("unexpected ParmEd version")
    return native, receipt


def _gaff2_nonbonded(path: Path, required_types: set[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] in required_types:
            try:
                candidate = [float(fields[1]), float(fields[2])]
            except ValueError:
                continue
            if tuple(candidate) == EXPECTED_GAFF2_LJ[fields[0]]:
                found[fields[0]] = candidate
    if set(found) != required_types:
        raise ValueError("native GAFF2 nonbonded parameters were not found exactly")
    return found


def _provenance() -> dict[str, dict[str, Any]]:
    paths = {
        "parmchk2": PARMCHK2,
        "tleap": TLEAP,
        "parmed_python": PARMED_PYTHON,
        "gaff2_dat": GAFF2_DAT,
        "leaprc_gaff2": LEAPRC_GAFF2,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != EXPECTED_NATIVE_SHA256[name]:
            raise ValueError(f"{name} SHA256 differs from the external pin")
        result[name] = {"path": str(path), "sha256": observed}
    return result


def prepare(protocol_path: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"immutable destination already exists: {destination}")
    protocol = load_protocol(protocol_path, expected_sha256=EXPECTED_PROTOCOL_SHA256)
    assets = protocol["source_assets"]
    paths = {name: Path(assets[name]["path"]) for name in REQUIRED_SOURCE_KEYS}
    pins = {name: assets[name]["sha256"] for name in REQUIRED_SOURCE_KEYS}
    bundle = load_source_bundle(paths, pins)
    if sha256_file(SOURCE_LEDGER) != EXPECTED_SOURCE_LEDGER_SHA256:
        raise ValueError("frozen CHA radius source ledger SHA256 mismatch")
    fixed = parse_mol2(bundle["fixed_mol2"].decode("utf-8"))
    retyped = parse_mol2(bundle["retyped_mol2"].decode("utf-8"))
    mapping = json.loads(bundle["mapping"])
    normalization = json.loads(bundle["normalization"])
    merged_text, merge_receipt = merge_verified_gaff2_types(
        fixed, retyped, mapping, normalization
    )
    provenance = _provenance()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".water-preparation-v1-", dir=destination.parent)
    )
    try:
        mol2 = temporary / "molecule.mol2"
        mol2.write_text(merged_text, encoding="utf-8")
        parmchk = _run(
            [
                str(PARMCHK2),
                "-i",
                mol2.name,
                "-f",
                "mol2",
                "-o",
                "molecule.frcmod",
                "-s",
                "gaff2",
            ],
            cwd=temporary,
            stem="parmchk2",
            allowed={str(PARMCHK2), str(TLEAP)},
        )
        _require_empty_nonbon(temporary / "molecule.frcmod")
        leap_input = temporary / "tleap.in"
        leap_input.write_text(
            "source leaprc.gaff2\n"
            "set default PBradii bondi\n"
            "MOL = loadmol2 molecule.mol2\n"
            "loadamberparams molecule.frcmod\n"
            "saveamberparm MOL molecule.prmtop molecule.inpcrd\n"
            "quit\n",
            encoding="utf-8",
        )
        tleap = _run(
            [str(TLEAP), "-f", leap_input.name],
            cwd=temporary,
            stem="tleap",
            allowed={str(PARMCHK2), str(TLEAP)},
        )
        native, parmed_receipt = _read_parmed(temporary)
        source_charges = [float(atom["charge_e"]) for atom in fixed["atoms"]]
        effective_charges, raw_charge_tokens = parse_prmtop_charge_field(
            (temporary / "molecule.prmtop").read_text(encoding="ascii"), atom_count=3
        )
        native_charges = [float(atom["charge_e"]) for atom in native["atoms"]]
        if (
            max(
                abs(a - b)
                for a, b in zip(effective_charges, native_charges, strict=True)
            )
            > 1.0e-14
        ):
            raise ValueError("raw prmtop and independent ParmEd charges disagree")
        if [atom["name"] for atom in native["atoms"]] != ["O1", "H2", "H3"]:
            raise ValueError("native topology atom names/order changed")
        if tuple(atom["atom_type"] for atom in native["atoms"]) != EXPECTED_WATER_TYPES:
            raise ValueError("native topology atom types changed")
        source_bonds = sorted(
            [
                min(bond["atom1_id"], bond["atom2_id"]) - 1,
                max(bond["atom1_id"], bond["atom2_id"]) - 1,
            ]
            for bond in fixed["bonds"]
        )
        if sorted(native["bonds"]) != source_bonds:
            raise ValueError("native topology bond graph changed")
        coordinate_tokens = _inpcrd_tokens(temporary / "molecule.inpcrd", 3)
        coordinates = [[float(value) for value in row] for row in coordinate_tokens]
        fixed_coordinates = [atom["coordinates_angstrom"] for atom in fixed["atoms"]]
        coordinate_delta = max(
            abs(a - b)
            for row_a, row_b in zip(coordinates, fixed_coordinates, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
        if coordinate_delta != 0.0:
            raise ValueError(
                "inpcrd coordinates differ from the quantized fixed source"
            )
        gaff2_parameters = _gaff2_nonbonded(GAFF2_DAT, set(EXPECTED_WATER_TYPES))
        for atom in native["atoms"]:
            expected_rmin, expected_epsilon = gaff2_parameters[atom["atom_type"]]
            if (
                abs(atom["rmin_angstrom"] - expected_rmin) > 1.0e-9
                or abs(atom["epsilon_kcal_mol"] - expected_epsilon) > 1.0e-9
            ):
                raise ValueError("prepared LJ parameters differ from native gaff2.dat")

        parameter_source = {
            "gaff2_dat_sha256": provenance["gaff2_dat"]["sha256"],
            "leaprc_gaff2_sha256": provenance["leaprc_gaff2"]["sha256"],
            "cha_radius_source_path": str(SOURCE_LEDGER),
            "cha_radius_source_sha256": sha256_file(SOURCE_LEDGER),
            "cha_base_radii_angstrom": {"O": 1.36, "H": 0.52},
            "cha_rs_angstrom": 0.52,
            "native_gaff2_nonbonded": gaff2_parameters,
        }
        topology = build_topology_artifact(
            atom_ids=[1, 2, 3],
            atom_names=[atom["name"] for atom in native["atoms"]],
            elements=[atom["element"] for atom in native["atoms"]],
            atomic_numbers=[atom["atomic_number"] for atom in native["atoms"]],
            gaff2_types=[atom["atom_type"] for atom in native["atoms"]],
            bonds=[
                [
                    min(bond["atom1_id"], bond["atom2_id"]) - 1,
                    max(bond["atom1_id"], bond["atom2_id"]) - 1,
                    bond["bond_type"],
                ]
                for bond in fixed["bonds"]
            ],
            source_charges=source_charges,
            effective_charges=effective_charges,
            source_mol2_sha256=pins["fixed_mol2"],
            prepared_prmtop_sha256=sha256_file(temporary / "molecule.prmtop"),
            parameter_source_sha256=sha256_bytes(canonical_bytes(parameter_source)),
            bondi_radii=[atom["bondi_radius_angstrom"] for atom in native["atoms"]],
            cha_radii=[EXPECTED_CHA_RADII[atom["element"]] for atom in native["atoms"]],
            lj_rmin=[atom["rmin_angstrom"] for atom in native["atoms"]],
            lj_epsilon=[atom["epsilon_kcal_mol"] for atom in native["atoms"]],
        )
        repository_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repository_root))
        from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
            ContinuumChaTopology,
        )

        ContinuumChaTopology.from_mapping(
            topology,
            expected_content_sha256=topology["content_sha256"],
            expected_source_charge_sha256=topology["source_charges_sha256"],
            expected_source_mol2_sha256=pins["fixed_mol2"],
        )
        coordinate_manifest = {
            "schema_version": 1,
            "topology_sha256": topology["content_sha256"],
            "atom_ids": [1, 2, 3],
            "positions_angstrom": coordinates,
        }
        coordinate_manifest["coordinate_sha256"] = sha256_bytes(
            canonical_bytes(coordinate_manifest)
        )
        write_json_new(temporary / "topology.json", topology)
        write_json_new(temporary / "coordinates.json", coordinate_manifest)

        native_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "preparation-record.json"
        }
        record = {
            "schema_version": 1,
            "artifact_type": "route1-cha-water-topology-preparation-v1",
            "status": "success",
            "profile": topology["profile"],
            "labels_read_or_emitted": False,
            "new_qm_or_charge_generation_performed": False,
            "invoked_topology_executables": ["parmchk2", "tleap"],
            "protocol": {
                "path": str(protocol_path),
                "sha256": EXPECTED_PROTOCOL_SHA256,
                "protocol_id": protocol["protocol_id"],
            },
            "source_assets": {
                name: {"path": str(paths[name]), "sha256": pins[name]}
                for name in REQUIRED_SOURCE_KEYS
            },
            "merge_receipt": merge_receipt,
            "commands": {
                "parmchk2": parmchk,
                "tleap": tleap,
                "parmed_readback": parmed_receipt,
            },
            "executables_and_parameter_inputs": provenance,
            "parameter_source": parameter_source,
            "raw_prmtop_charge_tokens": raw_charge_tokens,
            "validation": {
                "atom_names_order_types_exact": True,
                "bond_connectivity_exact": True,
                "coordinates_exact": True,
                "fixed_charge_source_preserved": True,
                "raw_prmtop_charge_field_verified": True,
                "independent_parmed_readback_verified": True,
                "native_gaff2_parameters_verified": True,
                "source_charge_sum_e": sum(source_charges),
                "effective_charge_sum_e": sum(effective_charges),
                "maximum_abs_source_to_effective_charge_delta_e": max(
                    abs(a - b)
                    for a, b in zip(source_charges, effective_charges, strict=True)
                ),
                "maximum_abs_source_to_inpcrd_coordinate_delta_angstrom": coordinate_delta,
            },
            "topology_content_sha256": topology["content_sha256"],
            "coordinate_sha256": coordinate_manifest["coordinate_sha256"],
            "native_files": native_files,
        }
        record["content_sha256"] = content_sha256(record)
        write_json_new(temporary / "preparation-record.json", record)
        os.replace(temporary, destination)
    except BaseException:
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = prepare(args.protocol.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": record["status"],
                "topology_content_sha256": record["topology_content_sha256"],
                "coordinate_sha256": record["coordinate_sha256"],
                **record["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
