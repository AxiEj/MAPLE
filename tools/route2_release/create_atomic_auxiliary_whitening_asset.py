#!/usr/bin/env python3
"""Create fixed ETB-beta2 isolated-atom Coulomb-whitening assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SELF_REPO_PATH = "tools/route2_release/create_atomic_auxiliary_whitening_asset.py"
WHITENING_REPO_PATH = (
    "maple/solvation/reference/atomic_auxiliary_whitening.py"
)
ARTIFACT = "route2-etb2-isolated-atom-coulomb-whitening-asset-v1"
ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode()
        + b"\0"
        + array.tobytes(order="C")
    ).hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    path.chmod(0o444)


def _write_npy_exclusive(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, np.asarray(values, dtype="<f8"), allow_pickle=False)
    path.chmod(0o444)


def create(args: argparse.Namespace) -> dict[str, Any]:
    from pyscf import df, gto
    import pyscf

    from maple.solvation.reference.atomic_auxiliary_whitening import (
        build_atomic_coulomb_whitening,
    )

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = []
    for symbol in ELEMENTS:
        spin = int(gto.charge(symbol) % 2)
        molecule = gto.M(
            atom=f"{symbol} 0 0 0",
            basis="def2-tzvpd",
            spin=spin,
            charge=0,
            verbose=0,
        )
        basis = df.aug_etb(molecule, beta=2.0)
        auxiliary = df.addons.make_auxmol(molecule, auxbasis=basis)
        whitening = build_atomic_coulomb_whitening(auxiliary)
        basis_path = output / f"{symbol}-etb2-basis.json"
        transform_path = output / f"{symbol}-coulomb-whitening.npy"
        _write_json_exclusive(basis_path, basis)
        _write_npy_exclusive(transform_path, whitening.transform)
        records.append(
            {
                "element": symbol,
                "atomic_number": int(gto.charge(symbol)),
                "isolated_atom_spin": spin,
                "auxiliary_dimension": whitening.dimension,
                "basis_file": basis_path.name,
                "basis_file_sha256": _sha256(basis_path),
                "transform_file": transform_path.name,
                "transform_file_sha256": _sha256(transform_path),
                "transform_array_sha256": _array_sha256(whitening.transform),
                "full_identity_max_absolute_error": (
                    whitening.full_identity_max_absolute_error
                ),
                "forbidden_angular_coupling_max_absolute_value": (
                    whitening.forbidden_angular_coupling_max_absolute_value
                ),
                "blocks": [
                    {
                        "angular_momentum": block.angular_momentum,
                        "radial_dimension": block.radial_dimension,
                        "normalized_condition_number": (
                            block.normalized_condition_number
                        ),
                        "whitened_identity_max_absolute_error": (
                            block.whitened_identity_max_absolute_error
                        ),
                    }
                    for block in whitening.blocks
                ],
            }
        )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "frozen-before-any-molecular-local-whitening-audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files_sha256": {
            SELF_REPO_PATH: _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            WHITENING_REPO_PATH: _sha256(SOURCE_ROOT / WHITENING_REPO_PATH),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "pyscf_init_sha256": _sha256(Path(pyscf.__file__).resolve(strict=True)),
        },
        "construction": {
            "orbital_basis_seed": "def2-tzvpd",
            "auxiliary_generator": "pyscf.df.aug_etb(beta=2.0)",
            "metric": "isolated-atom Coulomb metric",
            "transform": "canonical symmetric inverse square root per (element,l)",
            "angular_contract": "same radial transform for every m; no l mixing",
            "mode_removal_or_regularization": False,
            "molecular_geometry_or_cavity_used": False,
        },
        "elements": list(ELEMENTS),
        "records": records,
        "claim_boundary": {
            "experimental_solvation_target_read": False,
            "molecular_qm_or_model_output_read": False,
            "fit_or_training_performed": False,
            "capability_admitted": False,
        },
    }
    payload["asset_sha256"] = _canonical_sha256(payload)
    _write_json_exclusive(output / "asset.json", payload)
    output.chmod(0o555)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
