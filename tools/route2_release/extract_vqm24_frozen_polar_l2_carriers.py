#!/usr/bin/env python3
"""Extract two target-free zero-field MACE-POLAR l2 geometry carriers.

The checkpoint has two layers of ``512x0e + 512x1o`` node features and no
native l2 output.  This helper performs a train-only PCA in l1 multiplicity
space, retaining two Cartesian vector channels, then forms STF ``v tensor v``
carriers.  No source, response, QM observable, PCM, or solvation value is read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from ase import Atoms
import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.coupling.point_quadrupole import (  # noqa: E402
    traceless_quadrupole_coefficients,
)
from maple.solvation.models.mace_polar import (  # noqa: E402
    build_official_mace_polar_1_m_radial_gto_adapter,
)


SELF_REPO_PATH = "tools/route2_release/extract_vqm24_frozen_polar_l2_carriers.py"
ARTIFACT = "route2-vqm24-frozen-polar-l2-carrier-batch-v1"
RAW_VECTOR_MULTIPLICITY = 1024
RETAINED_VECTOR_CHANNELS = 2


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


def _vector_features(calculator, atoms: Atoms) -> np.ndarray:
    output = calculator._model_forward(
        calculator._batch_dict(atoms),
        training=False,
    )
    node_features = output.get("node_feats")
    if not torch.is_tensor(node_features):
        raise RuntimeError("MACE-POLAR zero-field forward omitted node_feats.")
    values = np.asarray(node_features.detach().cpu(), dtype=np.float64)
    atom_count = len(atoms)
    if values.shape != (atom_count, 4096):
        raise RuntimeError("MACE-POLAR node-feature layout changed from two 2048 blocks.")
    first_raw = values[:, 512:2048].reshape(atom_count, 512, 3)
    second_raw = values[:, 2560:4096].reshape(atom_count, 512, 3)
    raw = np.concatenate((first_raw, second_raw), axis=1)
    # e3nn real l=1 order is y,z,x for this checkpoint.
    return np.ascontiguousarray(raw[..., (2, 0, 1)])


def _fix_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    result = np.array(vectors, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if len(preregistration["records"]) != 32:
        raise RuntimeError("POLAR l2 carrier extraction requires the frozen 32 train records.")
    adapter = build_official_mace_polar_1_m_radial_gto_adapter(
        device=args.device,
        checkpoint_path=checkpoint_path,
    )
    calculator = adapter._calculator
    model = calculator.model
    if str(model.products[0].linear.irreps_out) != "512x0e+512x1o":
        raise RuntimeError("MACE-POLAR hidden irreps changed.")
    if int(model.num_interactions) != 2:
        raise RuntimeError("MACE-POLAR interaction count changed.")
    started = time.perf_counter()
    raw_records = []
    covariance = np.zeros(
        (RAW_VECTOR_MULTIPLICITY, RAW_VECTOR_MULTIPLICITY),
        dtype=np.float64,
    )
    vector_count = 0
    for record in preregistration["records"]:
        selection = record["selection_record"]
        atoms = Atoms(
            numbers=np.asarray(selection["atomic_numbers"], dtype=np.int64),
            positions=np.asarray(selection["positions_angstrom"], dtype=np.float64),
        )
        vectors = _vector_features(calculator, atoms)
        covariance += np.einsum("ncm,ndm->cd", vectors, vectors, optimize=True)
        vector_count += vectors.shape[0] * 3
        raw_records.append((record["record_id"], selection["record_sha256"], atoms, vectors))
    covariance /= vector_count
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    retained = _fix_eigenvector_signs(eigenvectors[:, -RETAINED_VECTOR_CHANNELS:])
    retained_values = eigenvalues[-RETAINED_VECTOR_CHANNELS:]
    if retained_values[0] <= 0.0:
        raise RuntimeError("Frozen POLAR multiplicity covariance has nonpositive top modes.")

    output_directory.mkdir(parents=True, exist_ok=False)
    records = []
    for record_id, selection_sha256, atoms, vectors in raw_records:
        compressed = np.einsum("ncm,ck->nkm", vectors, retained, optimize=True)
        tensors = np.einsum("nki,nkj->nkij", compressed, compressed, optimize=True)
        carriers = np.stack(
            [traceless_quadrupole_coefficients(tensors[:, channel]) for channel in range(2)],
            axis=1,
        )
        directory = output_directory / record_id
        directory.mkdir()
        npz_path = directory / "carriers.npz"
        np.savez(
            npz_path,
            atomic_numbers=atoms.numbers,
            positions_angstrom=atoms.positions,
            l2_carriers=carriers,
        )
        npz_path.chmod(0o444)
        records.append(
            {
                "record_id": record_id,
                "selection_record_sha256": selection_sha256,
                "npz_sha256": _sha256(npz_path),
                "carrier_sha256": hashlib.sha256(
                    np.ascontiguousarray(carriers).view(np.uint8)
                ).hexdigest(),
                "atom_count": len(atoms),
            }
        )
    pca_path = output_directory / "multiplicity-pca.npz"
    np.savez(
        pca_path,
        covariance_eigenvalues=eigenvalues,
        retained_eigenvalues=retained_values,
        retained_vectors=retained,
    )
    pca_path.chmod(0o444)
    payload = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "complete-frozen-polar-l2-carriers",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "adapter_configuration_sha256": adapter.configuration_sha256(),
            "pca_npz_sha256": _sha256(pca_path),
        },
        "feature_contract": {
            "zero_field_only": True,
            "hidden_irreps_per_layer": "512x0e+512x1o",
            "layer_count": 2,
            "raw_l1_multiplicity": RAW_VECTOR_MULTIPLICITY,
            "retained_l1_channels": RETAINED_VECTOR_CHANNELS,
            "raw_real_l1_order": "y,z,x",
            "stored_vector_order": "x,y,z",
            "l2_construction": "STF(v tensor v)",
            "pca_target_used": False,
        },
        "records": sorted(records, key=lambda record: record["record_id"]),
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": args.device,
            "dtype": "float64",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "frozen_zero_field_polar_geometry_carriers_generated": True,
            "original_polar_source_or_response_used": False,
            "qm_observable_read": False,
            "fit_or_training_performed": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "model_accuracy_measured": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
