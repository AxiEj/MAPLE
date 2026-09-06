"""Audit SPICE MBIS charge/dipole units and molecular closure without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(path: Path) -> dict[str, object]:
    """Return closure statistics for every supported neutral configuration."""

    import h5py

    charge_errors: list[float] = []
    dipole_component_errors: list[float] = []
    molecule_count = 0
    configuration_count = 0
    atom_count = 0
    field_dtypes: dict[str, set[str]] = {
        "positions": set(),
        "mbis_charges": set(),
        "mbis_dipoles": set(),
        "scf_dipole": set(),
    }
    supported = set(SUPPORTED_ATOMIC_NUMBERS)
    with h5py.File(path, "r") as handle:
        for name in sorted(handle):
            group = handle[name]
            required = {
                "atomic_numbers",
                "positions",
                "mbis_charges",
                "mbis_dipoles",
                "scf_dipole",
                "total_charge",
            }
            if not required.issubset(group):
                continue
            numbers = np.asarray(group["atomic_numbers"], dtype=int).reshape(-1)
            if not set(numbers).issubset(supported):
                continue
            for key in field_dtypes:
                field_dtypes[key].add(str(group[key].dtype))
            positions = np.asarray(group["positions"], dtype=float) * 10.0
            charges = np.asarray(group["mbis_charges"], dtype=float)[..., 0]
            dipoles = np.asarray(group["mbis_dipoles"], dtype=float) * 10.0
            molecular = np.asarray(group["scf_dipole"], dtype=float) * 10.0
            total = np.asarray(group["total_charge"], dtype=float).reshape(-1)
            if not (
                len(positions)
                == len(charges)
                == len(dipoles)
                == len(molecular)
                == len(total)
            ):
                raise RuntimeError(f"SPICE record {name} has inconsistent lengths.")
            selected = np.flatnonzero(np.rint(total) == 0.0)
            if not len(selected):
                continue
            molecule_count += 1
            for index in selected:
                q = charges[index]
                p = dipoles[index]
                r = positions[index]
                mu = molecular[index]
                if (
                    q.shape != (len(numbers),)
                    or p.shape != (len(numbers), 3)
                    or r.shape != (len(numbers), 3)
                    or mu.shape != (3,)
                    or not all(np.all(np.isfinite(x)) for x in (q, p, r, mu))
                ):
                    raise RuntimeError(f"SPICE record {name}/{index} is malformed.")
                charge_errors.append(float(np.sum(q) - total[index]))
                dipole_component_errors.extend(
                    (np.sum(q[:, None] * r + p, axis=0) - mu).tolist()
                )
                configuration_count += 1
                atom_count += len(numbers)
    if not configuration_count:
        raise RuntimeError("No supported neutral SPICE source labels were found.")
    q_error = np.asarray(charge_errors, dtype=float)
    mu_error = np.asarray(dipole_component_errors, dtype=float)
    result = {
        "eligible_molecule_count": molecule_count,
        "eligible_configuration_count": configuration_count,
        "eligible_atom_count": atom_count,
        "field_dtypes": {key: sorted(values) for key, values in field_dtypes.items()},
        "charge_closure_rmse_e": float(np.sqrt(np.mean(q_error**2))),
        "charge_closure_max_abs_e": float(np.max(np.abs(q_error))),
        "dipole_component_closure_rmse_eangstrom": float(
            np.sqrt(np.mean(mu_error**2))
        ),
        "dipole_component_closure_mae_eangstrom": float(
            np.mean(np.abs(mu_error))
        ),
        "dipole_component_closure_max_abs_eangstrom": float(
            np.max(np.abs(mu_error))
        ),
    }
    result["gates"] = {
        "charge_units_and_closure": result["charge_closure_max_abs_e"] <= 1.0e-3,
        "dipole_units_and_closure": (
            result["dipole_component_closure_max_abs_eangstrom"] <= 4.0e-3
        ),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    dataset = arguments.dataset.expanduser().resolve(strict=True)
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    measurements = audit_dataset(dataset)
    gates = measurements.pop("gates")
    if not all(gates.values()):
        raise RuntimeError(f"SPICE MBIS label integrity gates failed: {gates}")
    report = {
        "artifact": "route2-spice-mbis-source-label-integrity-v1",
        "claim_boundary": {
            "solvation_targets_read": False,
            "fitting_performed": False,
            "label_accuracy_proven": False,
            "unit_and_internal_closure_only": True,
        },
        "dataset_sha256": _sha256_file(dataset),
        "script_sha256": _sha256_file(Path(__file__)),
        "measurements": measurements,
        "gates": gates,
        "interpretation": (
            "SPICE stores these labels at finite precision.  The closure residuals "
            "validate the documented e/nm-to-e/Angstrom conversion and dataset "
            "internal consistency; they are not a QM source-accuracy benchmark."
        ),
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
