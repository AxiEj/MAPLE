"""Fixed charge providers used by implicit solvation.

QEq-GTO is shared with MAPLE's legacy charge boundary and retains both
charge-dependent hydrogen corrections from Rappe--Goddard QEq.  Its first
implicit-solvation domain is H, C, N, O, F, P, S, Cl, Br, and I.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from maple.function.read.filereader.mol2_reader import (
    MOL2_ATOM_ID_ARRAY,
    MOL2_IDENTITY_SHA256_KEY,
    MOL2Reader,
    mol2_identity_sha256,
)

from ..charge.qeq import QEqGTO
from .amber_chagb import render_typed_mol2


MOL2_CHARGE_TOL = 1.0e-4
# AmberTools SQM serializes the Mulliken precharges consumed by the AM1-BCC
# and ABCG2 stages to three decimal places.  The final MOL2 then introduces
# its own token rounding.  A valid closure bound therefore scales with atom
# count instead of using the former molecule-independent 0.01 e allowance.
ANTECHAMBER_PRECHARGE_RESOLUTION_E = 1.0e-3
# AmberTools 26 writes MOL2 coordinates and charges with at least four decimal
# places.  Provider-controlled tokens may not enlarge the error budget by
# claiming a coarser precision.
MAX_ANTECHAMBER_COORDINATE_HALF_WIDTH_ANGSTROM = 5.0e-5
MAX_ANTECHAMBER_CHARGE_HALF_WIDTH_E = 5.0e-5
QEQ_EXPERIMENTAL_PROVENANCE = {
    "scientific_status": "experimental",
    "accuracy_certified": False,
    "default_eligible": False,
    "selection_policy": "explicit-only-no-fallback",
}


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mol2_atom_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    section = ""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.upper().startswith("@<TRIPOS>"):
            section = line[9:].strip().upper()
            continue
        if section != "ATOM" or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(
                "Antechamber output must contain a serialized charge token for "
                f"every MOL2 atom; malformed row: {line!r}."
            )
        records.append(
            {
                "atom_id": int(fields[0]),
                "atom_name": fields[1],
                "coordinate_tokens": fields[2:5],
                "atom_type": fields[5],
                "charge_token": fields[8],
            }
        )
    if not records:
        raise ValueError("Antechamber output does not contain MOL2 ATOM records.")
    return records


def _decimal_rounding_half_width(token: str) -> float:
    try:
        value = Decimal(str(token))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal token in Antechamber MOL2 output: {token!r}.") from exc
    if not value.is_finite():
        raise ValueError(f"Non-finite decimal token in Antechamber MOL2 output: {token!r}.")
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f"Invalid finite decimal exponent in token: {token!r}.")
    return float(abs(Decimal(1).scaleb(exponent)) / 2)


def _bounded_decimal_rounding_half_width(
    token: str,
    *,
    quantity: str,
    maximum_half_width: float,
) -> float:
    half_width = _decimal_rounding_half_width(token)
    if half_width > maximum_half_width:
        raise ValueError(
            f"Antechamber {quantity} token {token!r} is coarser than the "
            "verified AmberTools 26 MOL2 precision."
        )
    return half_width


def _labeled_bond_graph(
    bonds,
    *,
    provider_to_input: dict[int, int] | None = None,
) -> Counter[tuple[int, int, str]]:
    graph: Counter[tuple[int, int, str]] = Counter()
    for first, second, bond_type in bonds:
        first = int(first)
        second = int(second)
        if provider_to_input is not None:
            try:
                first = provider_to_input[first]
                second = provider_to_input[second]
            except KeyError as exc:
                raise ValueError(
                    "Antechamber bond graph references an unmapped atom."
                ) from exc
        low, high = sorted((first, second))
        graph[(low, high, str(bond_type).strip().casefold())] += 1
    return graph


def _metadata_vector(
    metadata: dict[str, Any],
    name: str,
    atom_count: int,
) -> list[Any]:
    values = list(metadata.get(name) or [])
    if len(values) != atom_count:
        raise ValueError(f"MOL2 metadata must contain one {name} value per atom.")
    return values


def _validate_mol2_atom_identity(atoms, metadata: dict[str, Any]) -> list[int]:
    atom_ids = [
        int(value)
        for value in _metadata_vector(metadata, "atom_ids", len(atoms))
    ]
    identity = atoms.arrays.get(MOL2_ATOM_ID_ARRAY)
    if identity is None:
        raise ValueError(
            "MOL2 atom-identity provenance is missing; reload the structure "
            "with MAPLE's MOL2Reader before preparing charges."
        )
    identity = np.asarray(identity)
    if (
        identity.shape != (len(atoms),)
        or not np.issubdtype(identity.dtype, np.integer)
        or tuple(int(value) for value in identity) != tuple(atom_ids)
    ):
        raise ValueError(
            "Current ASE atom order no longer matches the source MOL2 atom IDs; "
            "refusing unsafe charge preparation."
        )
    return atom_ids


def _ambertools_installation(resolved_executable: str) -> dict[str, str | None]:
    executable_path = Path(resolved_executable)
    executable_hash = None
    try:
        if executable_path.is_file():
            import hashlib

            digest = hashlib.sha256()
            with executable_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            executable_hash = digest.hexdigest()
    except OSError:
        executable_hash = None

    version = "unknown-external"
    conda_meta = executable_path.parent.parent / "conda-meta"
    if conda_meta.is_dir():
        for metadata_path in sorted(conda_meta.glob("ambertools-*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if str(metadata.get("name", "")).lower() == "ambertools" and metadata.get("version"):
                version = str(metadata["version"])
                break
    return {
        "provider_version": version,
        "executable": str(executable_path),
        "executable_sha256": executable_hash,
    }


@dataclass
class ChargeResult:
    charges: np.ndarray
    method: str
    mode: str = "fixed"
    reference_positions: np.ndarray | None = None
    provider_positions: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _molecular_charge_and_mult(atoms) -> tuple[int, int]:
    if "charge" not in atoms.info or "mult" not in atoms.info:
        raise ValueError(
            "Implicit solvation requires an explicit 'charge multiplicity' line before MOL2."
        )
    charge_raw = atoms.info["charge"]
    mult_raw = atoms.info["mult"]
    try:
        charge_number = float(charge_raw)
        mult_number = float(mult_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Molecular charge and multiplicity must be integers.") from exc
    if not charge_number.is_integer() or not mult_number.is_integer():
        raise ValueError("Molecular charge and multiplicity must be integers.")
    charge = int(charge_number)
    mult = int(mult_number)
    if charge != 0 or mult != 1:
        raise ValueError(
            "The first supported implicit-solvation domain is neutral closed-shell molecules "
            "(charge=0, multiplicity=1)."
        )
    return charge, mult


def _mol2_charges(atoms, label: str | None) -> ChargeResult:
    metadata = atoms.info.get("mol2")
    if (
        not metadata
        or not metadata.get("charges_present")
        or str(metadata.get("charge_type", "NO_CHARGES")).upper() == "NO_CHARGES"
    ):
        raise ValueError("#charge(source=mol2) requires per-atom charges in the MOL2 ATOM section.")
    charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    total_charge, _ = _molecular_charge_and_mult(atoms)
    error = abs(float(charges.sum()) - total_charge)
    if error > MOL2_CHARGE_TOL:
        raise ValueError(
            "MOL2 partial-charge sum does not match the declared molecular charge: "
            f"sum={charges.sum():.8f}, declared={total_charge}. Charges are never silently renormalized."
        )
    return ChargeResult(
        charges=charges.copy(),
        method=label or "mol2-fixed",
        provenance={
            "source": "mol2",
            "label": label or "unspecified-fixed-charge",
            "path": metadata["path"],
            "charge_type": metadata.get("charge_type"),
            "sum_e": float(charges.sum()),
            "connected_component_count": int(metadata.get("component_count", 1)),
            "component_charge_sums_e": metadata.get("component_charge_sums_e"),
        },
    )


def _ambertools_charges(atoms, options: dict[str, Any], audit_dir: Path) -> ChargeResult:
    audit_dir = audit_dir.resolve()
    method = str(options["method"]).lower()
    charge_method = {"am1bcc": "bcc", "abcg2": "abcg2"}[method]
    geometry_policy = str(options.get("geometry", "keep")).lower()
    if geometry_policy not in {"keep", "provider"}:
        raise ValueError("Antechamber charge geometry must be 'keep' or 'provider'.")
    executable = str(options.get("executable", "antechamber"))
    resolved = shutil.which(executable)
    if resolved is None:
        raise ImportError(
            f"Charge method '{method}' requires AmberTools Antechamber. Install AmberTools24+ "
            f"and ensure '{executable}' is on PATH."
        )
    installation = _ambertools_installation(resolved)
    total_charge, mult = _molecular_charge_and_mult(atoms)
    metadata = atoms.info.get("mol2")
    if not metadata:
        raise ValueError(f"Charge method '{method}' requires a MOL2 input with explicit topology.")
    if metadata.get(MOL2_IDENTITY_SHA256_KEY) != mol2_identity_sha256(metadata):
        raise ValueError(
            "MOL2 atom/type/substructure/topology metadata changed after "
            "MOL2Reader; refusing unsafe charge preparation."
        )
    input_atom_ids = _validate_mol2_atom_identity(atoms, metadata)
    input_atom_names = _metadata_vector(metadata, "atom_names", len(atoms))
    input_atom_types = _metadata_vector(metadata, "atom_types", len(atoms))
    source_path = Path(metadata["path"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != metadata.get("source_sha256"):
        raise ValueError(
            "The source MOL2 changed after MOL2Reader froze its atom types and "
            "topology; refusing unsafe charge preparation."
        )
    source_text = source_bytes.decode("utf-8", errors="replace")
    audit_dir.mkdir(parents=True, exist_ok=True)
    provider_input_path = audit_dir / f"charges-{method}.input.mol2"
    input_charges = np.asarray(atoms.get_initial_charges(), dtype=np.float64)
    if input_charges.shape != (len(atoms),) or not np.isfinite(input_charges).all():
        input_charges = np.zeros(len(atoms), dtype=np.float64)
    provider_input_path.write_text(
        render_typed_mol2(
            source_text,
            atoms.get_positions(),
            input_charges,
        ),
        encoding="utf-8",
    )
    output_path = audit_dir / f"charges-{method}.mol2"
    command = [
        resolved,
        "-i",
        str(provider_input_path),
        "-fi",
        "mol2",
        "-o",
        str(output_path),
        "-fo",
        "mol2",
        "-c",
        charge_method,
        "-nc",
        str(total_charge),
        "-m",
        str(mult),
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
    completed = subprocess.run(
        command,
        cwd=audit_dir,
        text=True,
        capture_output=True,
        timeout=float(options.get("timeout", 3600.0)),
        check=False,
    )
    (audit_dir / f"charges-{method}.command.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": completed.returncode,
                "source_mol2": str(source_path),
                "source_mol2_sha256": source_sha256,
                "provider_input_mol2": str(provider_input_path),
                "provider_input_mol2_sha256": _sha256_file(provider_input_path),
                **installation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (audit_dir / f"charges-{method}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (audit_dir / f"charges-{method}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"Antechamber {method} charge generation failed with exit code {completed.returncode}; "
            f"see {audit_dir}."
        )

    provider_input_atoms = MOL2Reader(
        str(provider_input_path),
        charge=total_charge,
        mult=mult,
    )
    provider_atoms = MOL2Reader(str(output_path), charge=total_charge, mult=mult)
    provider_metadata = provider_atoms.info["mol2"]
    if (
        not provider_metadata.get("charges_present")
        or str(provider_metadata.get("charge_type", "NO_CHARGES")).upper() == "NO_CHARGES"
    ):
        raise ValueError("Antechamber output does not contain per-atom partial charges.")

    provider_atom_ids = [
        int(value)
        for value in _metadata_vector(
            provider_metadata,
            "atom_ids",
            len(provider_atoms),
        )
    ]
    if (
        len(set(provider_atom_ids)) != len(provider_atom_ids)
        or set(provider_atom_ids) != set(input_atom_ids)
    ):
        raise ValueError(
            "Antechamber changed, removed, or duplicated a MOL2 atom ID; "
            "refusing unsafe charge mapping."
        )
    provider_index_by_atom_id = {
        atom_id: provider_index
        for provider_index, atom_id in enumerate(provider_atom_ids)
    }
    provider_order = [
        provider_index_by_atom_id[atom_id] for atom_id in input_atom_ids
    ]
    provider_to_input = {
        provider_index: input_index
        for input_index, provider_index in enumerate(provider_order)
    }

    input_symbols = list(atoms.get_chemical_symbols())
    provider_symbols = list(provider_atoms.get_chemical_symbols())
    mapped_provider_symbols = [provider_symbols[index] for index in provider_order]
    if mapped_provider_symbols != input_symbols:
        raise ValueError(
            "Antechamber changed an atom element after identity mapping; "
            "refusing unsafe charge mapping."
        )
    input_graph = _labeled_bond_graph(metadata.get("bonds", []))
    provider_graph = _labeled_bond_graph(
        provider_metadata.get("bonds", []),
        provider_to_input=provider_to_input,
    )
    if provider_graph != input_graph:
        raise ValueError(
            "Antechamber changed labeled bond order/aromatic connectivity; "
            "refusing unsafe charge mapping."
        )

    provider_records = _mol2_atom_records(output_path)
    if len(provider_records) != len(provider_order):
        raise ValueError("Antechamber output atom-record count changed unexpectedly.")
    mapped_provider_positions = np.asarray(provider_atoms.get_positions(), dtype=np.float64)[
        provider_order
    ]
    if geometry_policy == "keep":
        expected_positions = np.asarray(
            provider_input_atoms.get_positions(),
            dtype=np.float64,
        )
        coordinate_half_widths = np.asarray(
            [
                [
                    _bounded_decimal_rounding_half_width(
                        token,
                        quantity="coordinate",
                        maximum_half_width=(
                            MAX_ANTECHAMBER_COORDINATE_HALF_WIDTH_ANGSTROM
                        ),
                    )
                    for token in provider_records[provider_index][
                        "coordinate_tokens"
                    ]
                ]
                for provider_index in provider_order
            ],
            dtype=np.float64,
        )
        coordinate_error = np.abs(mapped_provider_positions - expected_positions)
        if np.any(coordinate_error > coordinate_half_widths + 1.0e-10):
            maximum_error = float(np.max(coordinate_error))
            raise ValueError(
                "Antechamber changed coordinates under geometry=keep beyond "
                f"serialized output precision (maximum error {maximum_error:.8g} A)."
            )

    provider_names = _metadata_vector(
        provider_metadata,
        "atom_names",
        len(provider_atoms),
    )
    mapped_provider_names = [provider_names[index] for index in provider_order]
    if mapped_provider_names != [str(name) for name in input_atom_names]:
        raise ValueError(
            "Antechamber changed a MOL2 atom name despite disabled name and "
            "sequence rewriting; refusing unsafe charge mapping."
        )
    provider_atom_types = _metadata_vector(
        provider_metadata,
        "atom_types",
        len(provider_atoms),
    )
    mapping = []
    for input_index, provider_index in enumerate(provider_order):
        mapping.append(
            {
                "input_index": input_index,
                "provider_index": provider_index,
                "input_atom_id": input_atom_ids[input_index],
                "provider_atom_id": provider_atom_ids[provider_index],
                "input_atom_name": input_atom_names[input_index],
                "provider_atom_name": provider_names[provider_index],
                "element": input_symbols[input_index],
                "input_atom_type": input_atom_types[input_index],
                "provider_atom_type": provider_atom_types[provider_index],
                "atom_type_retyped": (
                    input_atom_types[input_index] != provider_atom_types[provider_index]
                ),
            }
        )
    mapping_path = audit_dir / f"charges-{method}.mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    raw_charges = np.asarray(
        provider_atoms.get_initial_charges(),
        dtype=np.float64,
    )[provider_order]
    charge_tokens = [
        provider_records[provider_index]["charge_token"]
        for provider_index in provider_order
    ]
    charge_half_widths = [
        _bounded_decimal_rounding_half_width(
            token,
            quantity="charge",
            maximum_half_width=MAX_ANTECHAMBER_CHARGE_HALF_WIDTH_E,
        )
        for token in charge_tokens
    ]
    raw_sum = float(raw_charges.sum())
    residual = float(total_charge) - raw_sum
    floating_slack = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.sum(np.abs(raw_charges))))
        * max(8, len(raw_charges))
    )
    precharge_half_widths = [
        ANTECHAMBER_PRECHARGE_RESOLUTION_E / 2.0
    ] * len(raw_charges)
    rounding_bound = float(
        sum(precharge_half_widths) + sum(charge_half_widths) + floating_slack
    )
    if abs(residual) > rounding_bound:
        raise ValueError(
            "Antechamber output charge residual exceeds the provider serialization "
            "rounding bound: "
            f"sum={raw_sum:.8f}, declared={total_charge}, "
            f"residual={residual:.8g} e, derived_bound={rounding_bound:.8g} e."
        )
    correction_per_atom = residual / len(raw_charges)
    charges = raw_charges + correction_per_atom
    normalization_path = audit_dir / f"charges-{method}.normalization.json"
    normalization_path.write_text(
        json.dumps(
            {
                "strategy": "serialization-rounding-repair",
                "distribution": "uniform-all-atoms",
                "scientific_role": (
                    "AmberTools SQM-precharge and MOL2-text serialization "
                    "closure; not a charge model or hydration residual model"
                ),
                "declared_total_e": total_charge,
                "provider_total_e": raw_sum,
                "provider_residual_e": residual,
                "serialized_charge_tokens": charge_tokens,
                "sqm_precharge_resolution_e": ANTECHAMBER_PRECHARGE_RESOLUTION_E,
                "per_atom_sqm_precharge_half_width_e": precharge_half_widths,
                "per_atom_mol2_rounding_half_width_e": charge_half_widths,
                "maximum_accepted_mol2_charge_rounding_half_width_e": (
                    MAX_ANTECHAMBER_CHARGE_HALF_WIDTH_E
                ),
                "derived_bound_formula": (
                    "sum(SQM precharge half-widths) + "
                    "sum(MOL2 token half-widths) + floating-point slack"
                ),
                "floating_point_slack_e": floating_slack,
                "derived_rounding_bound_e": rounding_bound,
                "correction_per_atom_e": correction_per_atom,
                "used_total_e": float(charges.sum()),
                "provider_charges_e": raw_charges.tolist(),
                "used_charges_e": charges.tolist(),
                "citation": (
                    "Loeffler et al., J. Chem. Inf. Model. 2015, "
                    "DOI:10.1021/acs.jcim.5b00368"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ChargeResult(
        charges=charges,
        method=method,
        reference_positions=np.asarray(atoms.get_positions()).copy(),
        provider_positions=mapped_provider_positions.copy(),
        provenance={
            "source": "ambertools-antechamber",
            "method": method,
            **installation,
            "command": command,
            "source_mol2": str(source_path),
            "source_mol2_sha256": source_sha256,
            "provider_input_mol2": str(provider_input_path),
            "provider_input_mol2_sha256": _sha256_file(provider_input_path),
            "output_mol2": str(output_path),
            "atom_mapping": str(mapping_path),
            "charge_normalization": str(normalization_path),
            "geometry_policy": geometry_policy,
            "maximum_accepted_mol2_coordinate_rounding_half_width_angstrom": (
                MAX_ANTECHAMBER_COORDINATE_HALF_WIDTH_ANGSTROM
            ),
            "maximum_accepted_mol2_charge_rounding_half_width_e": (
                MAX_ANTECHAMBER_CHARGE_HALF_WIDTH_E
            ),
            "identity_mapping": (
                "MOL2-atom-id-plus-atom-name-plus-labeled-graph"
            ),
            "labeled_bond_graph_verified": True,
            "provider_atom_names_verified": True,
            "provider_atom_type_retyping_count": sum(
                record["atom_type_retyped"] for record in mapping
            ),
            "provider_sum_e": raw_sum,
            "provider_residual_e": residual,
            "derived_rounding_bound_e": rounding_bound,
            "correction_per_atom_e": correction_per_atom,
            "sum_e": float(charges.sum()),
            "citations": [
                "Jakalian et al., J. Comput. Chem. 2002, DOI:10.1002/jcc.10128"
                if method == "am1bcc"
                else "ABCG2, J. Chem. Theory Comput. 2025, DOI:10.1021/acs.jctc.5c00038"
            ],
        },
    )


def prepare_charges(atoms, options: dict[str, Any], audit_dir: str | os.PathLike[str]) -> ChargeResult:
    """Prepare a charge record once at the reference geometry."""
    source = str(options.get("source", "")).lower()
    audit_path = Path(audit_dir)
    if source == "mol2":
        return _mol2_charges(atoms, options.get("label"))
    if source != "maple":
        raise ValueError("#charge source must be 'mol2' or 'maple'.")

    method = str(options.get("method", "")).lower()
    if method in {"am1bcc", "abcg2"}:
        return _ambertools_charges(atoms, options, audit_path)
    if method == "qeq-gto":
        total_charge, _ = _molecular_charge_and_mult(atoms)
        solver = QEqGTO()
        charges = solver.solve(atoms, total_charge=total_charge)
        return ChargeResult(
            charges=charges,
            method=method,
            mode=str(options.get("mode", "fixed")),
            reference_positions=np.asarray(atoms.get_positions()).copy(),
            provenance={
                "source": "maple",
                "method": "qeq-gto",
                "mode": str(options.get("mode", "fixed")),
                **QEQ_EXPERIMENTAL_PROVENANCE,
                "parameter_file": str(solver.data_file),
                "profile": "rappe-goddard-gto-full-h-scf",
                "lambda_scale": solver.lambda_scale,
                "damping": solver.damping,
                "tolerance_e": solver.tolerance,
                "hydrogen_idempotential_update": True,
                "hydrogen_screening_exponent_update": True,
                "iterations": solver.last_iterations,
                "max_delta_e": solver.last_max_delta,
                "kkt_residual_ev": solver.last_kkt_residual,
                "sum_e": float(charges.sum()),
                "citations": [
                    "Rappe and Goddard, Charge equilibration for molecular dynamics "
                    "simulations, J. Phys. Chem. 1991, DOI:10.1021/j100161a070",
                    "Chen and Martinez, Charge conservation in electronegativity "
                    "equalization, J. Chem. Phys. 2009, DOI:10.1063/1.3183167",
                ],
                "validation_scope": (
                    "fixed mode uses original-QEq SCF; polarizable mode switches to the "
                    "consistent-QEq nonlinear solver in the correction layer"
                ),
            },
        )
    raise ValueError(f"Unsupported MAPLE charge method: {method!r}.")
