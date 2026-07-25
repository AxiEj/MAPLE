"""Fixed charge providers used by implicit solvation.

QEq-GTO is shared with MAPLE's legacy charge boundary and retains both
charge-dependent hydrogen corrections from Rappe--Goddard QEq.  Its first
implicit-solvation domain is H, C, N, O, F, P, S, Cl, Br, and I.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from ..charge.qeq import QEqGTO


MOL2_CHARGE_TOL = 1.0e-4
AMBERTOOLS_CHARGE_RESIDUAL_LIMIT_E = 1.0e-2
QEQ_EXPERIMENTAL_PROVENANCE = {
    "scientific_status": "experimental",
    "accuracy_certified": False,
    "default_eligible": False,
    "selection_policy": "explicit-only-no-fallback",
}


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
    mol2_path = Path(metadata["path"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_path = audit_dir / f"charges-{method}.mol2"
    command = [
        resolved,
        "-i", str(mol2_path), "-fi", "mol2",
        "-o", str(output_path), "-fo", "mol2",
        "-c", charge_method,
        "-nc", str(total_charge),
        "-m", str(mult),
        "-at", "gaff2",
        "-pf", "y",
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

    from maple.function.read.filereader.mol2_reader import MOL2Reader

    provider_atoms = MOL2Reader(str(output_path), charge=total_charge, mult=mult)
    if provider_atoms.get_chemical_symbols() != atoms.get_chemical_symbols():
        raise ValueError("Antechamber changed atom count/order; refusing unsafe charge mapping.")
    input_bonds = {tuple(sorted((int(i), int(j)))) for i, j, _ in metadata.get("bonds", [])}
    provider_metadata = provider_atoms.info["mol2"]
    provider_bonds = {
        tuple(sorted((int(i), int(j)))) for i, j, _ in provider_metadata.get("bonds", [])
    }
    if provider_bonds != input_bonds:
        raise ValueError("Antechamber changed bond connectivity; refusing unsafe charge mapping.")
    mapping = [
        {
            "index": index,
            "element": symbol,
            "input_atom_name": input_name,
            "provider_atom_name": provider_name,
        }
        for index, (symbol, input_name, provider_name) in enumerate(
            zip(
                atoms.get_chemical_symbols(),
                metadata.get("atom_names", atoms.get_chemical_symbols()),
                provider_metadata.get("atom_names", provider_atoms.get_chemical_symbols()),
            )
        )
    ]
    mapping_path = audit_dir / f"charges-{method}.mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    if (
        not provider_metadata.get("charges_present")
        or str(provider_metadata.get("charge_type", "NO_CHARGES")).upper() == "NO_CHARGES"
    ):
        raise ValueError("Antechamber output does not contain per-atom partial charges.")
    raw_charges = np.asarray(provider_atoms.get_initial_charges(), dtype=np.float64)
    raw_sum = float(raw_charges.sum())
    residual = float(total_charge) - raw_sum
    if abs(residual) > AMBERTOOLS_CHARGE_RESIDUAL_LIMIT_E:
        raise ValueError(
            "Antechamber output charge residual is too large for a precision correction: "
            f"sum={raw_sum:.8f}, declared={total_charge}, "
            f"limit={AMBERTOOLS_CHARGE_RESIDUAL_LIMIT_E:.4f} e."
        )
    correction_per_atom = residual / len(raw_charges)
    charges = raw_charges + correction_per_atom
    normalization_path = audit_dir / f"charges-{method}.normalization.json"
    normalization_path.write_text(
        json.dumps(
            {
                "strategy": "uniform-all-atoms",
                "declared_total_e": total_charge,
                "provider_total_e": raw_sum,
                "provider_residual_e": residual,
                "correction_per_atom_e": correction_per_atom,
                "used_total_e": float(charges.sum()),
                "provider_charges_e": raw_charges.tolist(),
                "used_charges_e": charges.tolist(),
                "citation": "Loeffler et al., J. Chem. Inf. Model. 2015, DOI:10.1021/acs.jcim.5b00368",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ChargeResult(
        charges=charges,
        method=method,
        reference_positions=np.asarray(atoms.get_positions()).copy(),
        provider_positions=np.asarray(provider_atoms.get_positions()).copy(),
        provenance={
            "source": "ambertools-antechamber",
            "method": method,
            **installation,
            "command": command,
            "output_mol2": str(output_path),
            "atom_mapping": str(mapping_path),
            "charge_normalization": str(normalization_path),
            "geometry_policy": options.get("geometry", "keep"),
            "provider_sum_e": raw_sum,
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
