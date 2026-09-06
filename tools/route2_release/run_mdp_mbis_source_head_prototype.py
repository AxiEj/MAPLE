"""Fit a target-independent MBIS source readout on frozen MACE-MDP features.

This is a research prototype for a *new* hybrid source identity.  It never
reads solvation data.  The frozen MACE-MDP backbone supplies invariant/vector
node features and a molecular dipole; independent SPICE MBIS atomic charges
and dipoles supervise two small ridge readouts.  A joint charge/dipole metric
projection then enforces total charge and the MACE-MDP molecular dipole exactly.

The script intentionally stops before PCM or accuracy admission.  Its output
can only establish whether a separately source-supervised frozen-backbone head
is technically viable on molecule-held-out QM data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import numpy as np


SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
LAMBDA_GRID = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
SCALAR_CHANNELS_PER_LAYER = 128
VECTOR_CHANNELS_PER_LAYER = 128
PRODUCT_WIDTH = 1152


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecule_split(name: str) -> str:
    """Return a deterministic molecule-level 70/10/20 split."""

    if not isinstance(name, str) or not name:
        raise ValueError("Molecule name must be nonempty.")
    bucket = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % 10
    if bucket < 7:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    """Return the unique finite ridge solution for one scalar target."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if (
        x.ndim != 2
        or y.shape != (x.shape[0],)
        or x.shape[0] == 0
        or x.shape[1] == 0
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or not np.isfinite(ridge)
        or ridge <= 0.0
    ):
        raise ValueError("Ridge inputs must be finite, nonempty, and compatible.")
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += float(ridge)
    weights = np.linalg.solve(gram, x.T @ y)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError("Ridge solution is not finite.")
    return weights


def charge_dipole_constraints(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    dipoles_eangstrom: np.ndarray,
) -> np.ndarray:
    """Return ``[Q, mu_x, mu_y, mu_z]`` in e/e-Angstrom units."""

    positions = np.asarray(positions_angstrom, dtype=float)
    charges = np.asarray(charges_e, dtype=float)
    dipoles = np.asarray(dipoles_eangstrom, dtype=float)
    atom_count = len(charges)
    if (
        positions.shape != (atom_count, 3)
        or dipoles.shape != (atom_count, 3)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(charges))
        or not np.all(np.isfinite(dipoles))
    ):
        raise ValueError("Source arrays must be finite and atom aligned.")
    return np.concatenate(
        ([float(np.sum(charges))], np.sum(charges[:, None] * positions + dipoles, axis=0))
    )


def project_charge_dipole_source(
    *,
    positions_angstrom: np.ndarray,
    raw_charges_e: np.ndarray,
    raw_dipoles_eangstrom: np.ndarray,
    total_charge_e: float,
    molecular_dipole_eangstrom: np.ndarray,
    charge_sigma_e: float,
    dipole_sigma_eangstrom: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project a raw source to exact molecular constraints in an uncertainty metric.

    The minimised dimensionless objective is

    ``||dq/charge_sigma||^2 + ||dp/dipole_sigma||^2``.

    Local dipole variables make the four-row constraint full rank for every
    nonempty geometry, including atoms and linear/symmetric molecules.  The
    strictly positive metric makes the projection unique.  Because both metric
    blocks are atom-scalar, the construction is permutation/translation/rotation
    covariant when the molecular target is transformed consistently.
    """

    positions = np.asarray(positions_angstrom, dtype=float)
    charges = np.asarray(raw_charges_e, dtype=float)
    dipoles = np.asarray(raw_dipoles_eangstrom, dtype=float)
    target_dipole = np.asarray(molecular_dipole_eangstrom, dtype=float)
    atom_count = len(charges)
    if atom_count == 0:
        raise ValueError("At least one atom is required.")
    if (
        positions.shape != (atom_count, 3)
        or dipoles.shape != (atom_count, 3)
        or target_dipole.shape != (3,)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(charges))
        or not np.all(np.isfinite(dipoles))
        or not np.all(np.isfinite(target_dipole))
        or not np.isfinite(total_charge_e)
        or not np.isfinite(charge_sigma_e)
        or not np.isfinite(dipole_sigma_eangstrom)
        or charge_sigma_e <= 0.0
        or dipole_sigma_eangstrom <= 0.0
    ):
        raise ValueError("Projection inputs and metric scales must be finite.")

    source = np.concatenate((charges, dipoles.reshape(-1)))
    constraint = np.zeros((4, 4 * atom_count), dtype=float)
    constraint[0, :atom_count] = 1.0
    constraint[1:, :atom_count] = positions.T
    for atom_index in range(atom_count):
        start = atom_count + 3 * atom_index
        constraint[1:, start : start + 3] = np.eye(3)
    covariance = np.concatenate(
        (
            np.full(atom_count, charge_sigma_e**2),
            np.full(3 * atom_count, dipole_sigma_eangstrom**2),
        )
    )
    target = np.concatenate(([float(total_charge_e)], target_dipole))
    residual = target - constraint @ source
    schur = (constraint * covariance[None, :]) @ constraint.T
    condition = float(np.linalg.cond(schur))
    if not np.isfinite(condition):
        raise RuntimeError("Constraint projection Schur matrix is singular.")
    multipliers = np.linalg.solve(schur, residual)
    projected = source + covariance * (constraint.T @ multipliers)
    projected_charges = projected[:atom_count]
    projected_dipoles = projected[atom_count:].reshape(atom_count, 3)
    closure = charge_dipole_constraints(
        positions, projected_charges, projected_dipoles
    )
    if not np.allclose(closure, target, rtol=0.0, atol=2.0e-12):
        raise RuntimeError("Charge/dipole projection did not close.")
    return projected_charges, projected_dipoles, condition


@dataclass(frozen=True)
class _Sample:
    molecule: str
    configuration: int
    split: str
    atomic_numbers: np.ndarray
    positions_angstrom: np.ndarray
    mbis_charges_e: np.ndarray
    mbis_dipoles_eangstrom: np.ndarray
    scf_dipole_eangstrom: np.ndarray
    total_charge_e: float


@dataclass(frozen=True)
class _Features:
    sample: _Sample
    scalar: np.ndarray
    vector: np.ndarray
    mdp_charges_e: np.ndarray
    mdp_dipoles_eangstrom: np.ndarray
    mdp_molecular_dipole_eangstrom: np.ndarray


def _load_samples(dataset_path: Path) -> list[_Sample]:
    import h5py

    supported = set(SUPPORTED_ATOMIC_NUMBERS)
    samples: list[_Sample] = []
    with h5py.File(dataset_path, "r") as handle:
        for molecule in sorted(handle):
            group = handle[molecule]
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
            positions = np.asarray(group["positions"], dtype=float) * 10.0
            charges = np.asarray(group["mbis_charges"], dtype=float)[..., 0]
            dipoles = np.asarray(group["mbis_dipoles"], dtype=float) * 10.0
            molecular = np.asarray(group["scf_dipole"], dtype=float) * 10.0
            total_charge = np.asarray(group["total_charge"], dtype=float).reshape(-1)
            if not (
                len(positions)
                == len(charges)
                == len(dipoles)
                == len(molecular)
                == len(total_charge)
            ):
                raise ValueError(f"SPICE record {molecule} has inconsistent lengths.")
            if not np.all(np.rint(total_charge) == 0.0):
                continue
            split = molecule_split(molecule)
            for index in range(len(positions)):
                sample = _Sample(
                    molecule=molecule,
                    configuration=index,
                    split=split,
                    atomic_numbers=numbers.copy(),
                    positions_angstrom=positions[index].copy(),
                    mbis_charges_e=charges[index].copy(),
                    mbis_dipoles_eangstrom=dipoles[index].copy(),
                    scf_dipole_eangstrom=molecular[index].copy(),
                    total_charge_e=float(total_charge[index]),
                )
                arrays: Iterable[np.ndarray] = (
                    sample.positions_angstrom,
                    sample.mbis_charges_e,
                    sample.mbis_dipoles_eangstrom,
                    sample.scf_dipole_eangstrom,
                )
                if not all(np.all(np.isfinite(array)) for array in arrays):
                    raise ValueError(f"SPICE record {molecule}/{index} is nonfinite.")
                samples.append(sample)
    if not samples or {sample.split for sample in samples} != {
        "train",
        "validation",
        "test",
    }:
        raise RuntimeError("Eligible SPICE samples do not populate every split.")
    return samples


def _extract_features(
    samples: list[_Sample], *, checkpoint: Path, device: str, batch_size: int
) -> tuple[list[_Features], dict[str, object]]:
    import torch
    from ase import Atoms
    from mace import data as mace_data
    from mace.calculators import MACECalculator
    from mace.tools import torch_geometric, torch_tools

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    calculator = MACECalculator(
        model_paths=str(checkpoint),
        model_type="DipolePolarizabilityMACE",
        default_dtype="float64",
        device=device,
    )
    model = calculator.models[0]
    if tuple(int(value) for value in model.atomic_numbers.tolist()) != (
        SUPPORTED_ATOMIC_NUMBERS
    ):
        raise RuntimeError("MACE-MDP element order changed.")
    products = tuple(model.products)
    if len(products) != 2:
        raise RuntimeError("MACE-MDP product-layer count changed.")
    key_specification = mace_data.KeySpecification(
        info_keys=calculator.info_keys, arrays_keys=calculator.arrays_keys
    )
    result: list[_Features] = []
    for offset in range(0, len(samples), batch_size):
        selected = samples[offset : offset + batch_size]
        graphs = []
        for sample in selected:
            atoms = Atoms(
                numbers=sample.atomic_numbers,
                positions=sample.positions_angstrom,
                info={"charge": 0, "mult": 1},
            )
            with torch_tools.default_dtype(calculator.default_dtype):
                config = mace_data.config_from_atoms(
                    atoms,
                    key_specification=key_specification,
                    head_name=calculator.head,
                )
                graphs.append(
                    mace_data.AtomicData.from_config(
                        config,
                        z_table=calculator.z_table,
                        cutoff=calculator.r_max,
                        heads=calculator.available_heads,
                    )
                )
        batch = torch_geometric.Batch.from_data_list(graphs).to(calculator.device)
        captured: list[object] = []

        def capture(_module: object, _inputs: object, output: object) -> None:
            captured.append(output)

        handles = [product.register_forward_hook(capture) for product in products]
        try:
            with torch.no_grad():
                output = model(
                    batch.to_dict(),
                    compute_dielectric_derivatives=False,
                    training=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 2 or any(
            tuple(value.shape)[1:] != (PRODUCT_WIDTH,) for value in captured
        ):
            raise RuntimeError("MACE-MDP hidden representation changed.")
        scalar = torch.cat(
            [value[:, :SCALAR_CHANNELS_PER_LAYER] for value in captured], dim=1
        ).detach().cpu().numpy()
        vector = torch.cat(
            [
                value[
                    :,
                    SCALAR_CHANNELS_PER_LAYER : SCALAR_CHANNELS_PER_LAYER
                    + 3 * VECTOR_CHANNELS_PER_LAYER,
                ].reshape(-1, VECTOR_CHANNELS_PER_LAYER, 3)
                for value in captured
            ],
            dim=1,
        ).detach().cpu().numpy()
        charges = output["charges"].detach().cpu().numpy()
        dipoles = output["atomic_dipoles"].detach().cpu().numpy()
        molecular = output["dipole"].detach().cpu().numpy()
        ptr = batch.ptr.detach().cpu().numpy()
        for local_index, sample in enumerate(selected):
            start, stop = int(ptr[local_index]), int(ptr[local_index + 1])
            result.append(
                _Features(
                    sample=sample,
                    scalar=np.asarray(scalar[start:stop], dtype=float),
                    vector=np.asarray(vector[start:stop], dtype=float),
                    mdp_charges_e=np.asarray(charges[start:stop], dtype=float),
                    mdp_dipoles_eangstrom=np.asarray(dipoles[start:stop], dtype=float),
                    mdp_molecular_dipole_eangstrom=np.asarray(
                        molecular[local_index], dtype=float
                    ),
                )
            )
    runtime = {
        "torch_version": torch.__version__,
        "device": device,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "mace_model_type": type(model).__name__,
        "hidden_product_width": PRODUCT_WIDTH,
    }
    return result, runtime


def _stack_scalar(
    records: list[_Features], split: str, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = [record for record in records if record.sample.split == split]
    scalar = np.concatenate([record.scalar for record in selected])
    numbers = np.concatenate([record.sample.atomic_numbers for record in selected])
    one_hot = numbers[:, None] == np.asarray(SUPPORTED_ATOMIC_NUMBERS)[None, :]
    features = np.concatenate(((scalar - mean) / scale, one_hot.astype(float)), axis=1)
    targets = np.concatenate([record.sample.mbis_charges_e for record in selected])
    return features, targets


def _stack_vector(
    records: list[_Features], split: str, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = [record for record in records if record.sample.split == split]
    vectors = np.concatenate([record.vector for record in selected]) / scale[None, :, None]
    features = vectors.transpose(0, 2, 1).reshape(-1, vectors.shape[1])
    targets = np.concatenate(
        [record.sample.mbis_dipoles_eangstrom for record in selected]
    ).reshape(-1)
    return features, targets


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def _head_prediction(
    record: _Features,
    *,
    scalar_mean: np.ndarray,
    scalar_scale: np.ndarray,
    vector_scale: np.ndarray,
    charge_weights: np.ndarray,
    dipole_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    one_hot = (
        record.sample.atomic_numbers[:, None]
        == np.asarray(SUPPORTED_ATOMIC_NUMBERS)[None, :]
    )
    scalar = np.concatenate(
        (
            (record.scalar - scalar_mean) / scalar_scale,
            one_hot.astype(float),
        ),
        axis=1,
    )
    charges = scalar @ charge_weights
    vectors = record.vector / vector_scale[None, :, None]
    dipoles = np.einsum("adc,d->ac", vectors, dipole_weights)
    return charges, dipoles


def _evaluate(
    records: list[_Features],
    split: str,
    *,
    scalar_mean: np.ndarray,
    scalar_scale: np.ndarray,
    vector_scale: np.ndarray,
    charge_weights: np.ndarray,
    dipole_weights: np.ndarray,
    charge_sigma: float,
    dipole_sigma: float,
) -> dict[str, float | int]:
    selected = [record for record in records if record.sample.split == split]
    original_q: list[np.ndarray] = []
    original_p: list[np.ndarray] = []
    raw_q: list[np.ndarray] = []
    raw_p: list[np.ndarray] = []
    projected_q: list[np.ndarray] = []
    projected_p: list[np.ndarray] = []
    target_q: list[np.ndarray] = []
    target_p: list[np.ndarray] = []
    mdp_mu: list[np.ndarray] = []
    scf_mu: list[np.ndarray] = []
    projected_mu: list[np.ndarray] = []
    max_charge_error = 0.0
    max_condition = 0.0
    atom_count = 0
    for record in selected:
        charges, dipoles = _head_prediction(
            record,
            scalar_mean=scalar_mean,
            scalar_scale=scalar_scale,
            vector_scale=vector_scale,
            charge_weights=charge_weights,
            dipole_weights=dipole_weights,
        )
        q_projected, p_projected, condition = project_charge_dipole_source(
            positions_angstrom=record.sample.positions_angstrom,
            raw_charges_e=charges,
            raw_dipoles_eangstrom=dipoles,
            total_charge_e=record.sample.total_charge_e,
            molecular_dipole_eangstrom=record.mdp_molecular_dipole_eangstrom,
            charge_sigma_e=charge_sigma,
            dipole_sigma_eangstrom=dipole_sigma,
        )
        closure = charge_dipole_constraints(
            record.sample.positions_angstrom, q_projected, p_projected
        )
        max_charge_error = max(
            max_charge_error, abs(closure[0] - record.sample.total_charge_e)
        )
        max_condition = max(max_condition, condition)
        atom_count += len(charges)
        original_q.append(record.mdp_charges_e)
        original_p.append(record.mdp_dipoles_eangstrom)
        raw_q.append(charges)
        raw_p.append(dipoles)
        projected_q.append(q_projected)
        projected_p.append(p_projected)
        target_q.append(record.sample.mbis_charges_e)
        target_p.append(record.sample.mbis_dipoles_eangstrom)
        mdp_mu.append(record.mdp_molecular_dipole_eangstrom)
        scf_mu.append(record.sample.scf_dipole_eangstrom)
        projected_mu.append(closure[1:])
    arrays = {
        "original_q": np.concatenate(original_q),
        "original_p": np.concatenate(original_p),
        "raw_q": np.concatenate(raw_q),
        "raw_p": np.concatenate(raw_p),
        "projected_q": np.concatenate(projected_q),
        "projected_p": np.concatenate(projected_p),
        "target_q": np.concatenate(target_q),
        "target_p": np.concatenate(target_p),
        "mdp_mu": np.asarray(mdp_mu),
        "scf_mu": np.asarray(scf_mu),
        "projected_mu": np.asarray(projected_mu),
    }
    return {
        "molecule_count": len({record.sample.molecule for record in selected}),
        "configuration_count": len(selected),
        "atom_count": atom_count,
        "original_charge_mae_e": _mae(arrays["original_q"], arrays["target_q"]),
        "original_charge_rmse_e": _rmse(arrays["original_q"], arrays["target_q"]),
        "learned_raw_charge_mae_e": _mae(arrays["raw_q"], arrays["target_q"]),
        "learned_raw_charge_rmse_e": _rmse(arrays["raw_q"], arrays["target_q"]),
        "learned_projected_charge_mae_e": _mae(
            arrays["projected_q"], arrays["target_q"]
        ),
        "learned_projected_charge_rmse_e": _rmse(
            arrays["projected_q"], arrays["target_q"]
        ),
        "original_dipole_component_mae_eangstrom": _mae(
            arrays["original_p"], arrays["target_p"]
        ),
        "original_dipole_component_rmse_eangstrom": _rmse(
            arrays["original_p"], arrays["target_p"]
        ),
        "learned_raw_dipole_component_mae_eangstrom": _mae(
            arrays["raw_p"], arrays["target_p"]
        ),
        "learned_raw_dipole_component_rmse_eangstrom": _rmse(
            arrays["raw_p"], arrays["target_p"]
        ),
        "learned_projected_dipole_component_mae_eangstrom": _mae(
            arrays["projected_p"], arrays["target_p"]
        ),
        "learned_projected_dipole_component_rmse_eangstrom": _rmse(
            arrays["projected_p"], arrays["target_p"]
        ),
        "mdp_molecular_dipole_component_rmse_vs_scf_eangstrom": _rmse(
            arrays["mdp_mu"], arrays["scf_mu"]
        ),
        "projected_molecular_dipole_component_rmse_vs_mdp_eangstrom": _rmse(
            arrays["projected_mu"], arrays["mdp_mu"]
        ),
        "projected_molecular_dipole_component_rmse_vs_scf_eangstrom": _rmse(
            arrays["projected_mu"], arrays["scf_mu"]
        ),
        "maximum_total_charge_closure_error_e": max_charge_error,
        "maximum_projection_schur_condition_number": max_condition,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    preregistration = json.loads(preregistration_path.read_text())
    if preregistration.get("artifact") != "mdp-mbis-source-head-prototype-prereg-v1":
        raise ValueError("Wrong preregistration artifact.")
    expected = preregistration.get("inputs_sha256")
    actual = {
        "dataset": _sha256_file(dataset),
        "checkpoint": _sha256_file(checkpoint),
        "script": _sha256_file(Path(__file__).resolve()),
    }
    if expected != actual:
        raise ValueError(f"Input/source hashes do not match preregistration: {actual}")
    samples = _load_samples(dataset)
    features, runtime = _extract_features(
        samples, checkpoint=checkpoint, device=args.device, batch_size=args.batch_size
    )

    train_scalar = np.concatenate(
        [record.scalar for record in features if record.sample.split == "train"]
    )
    scalar_mean = np.mean(train_scalar, axis=0)
    scalar_scale = np.std(train_scalar, axis=0)
    scalar_scale = np.maximum(scalar_scale, 1.0e-12)
    train_vector = np.concatenate(
        [record.vector for record in features if record.sample.split == "train"]
    )
    vector_scale = np.sqrt(np.mean(train_vector**2, axis=(0, 2)))
    vector_scale = np.maximum(vector_scale, 1.0e-12)
    q_train_x, q_train_y = _stack_scalar(
        features, "train", scalar_mean, scalar_scale
    )
    q_validation_x, q_validation_y = _stack_scalar(
        features, "validation", scalar_mean, scalar_scale
    )
    p_train_x, p_train_y = _stack_vector(features, "train", vector_scale)
    p_validation_x, p_validation_y = _stack_vector(
        features, "validation", vector_scale
    )
    q_candidates = []
    p_candidates = []
    for ridge in LAMBDA_GRID:
        q_weights = fit_ridge(q_train_x, q_train_y, ridge)
        p_weights = fit_ridge(p_train_x, p_train_y, ridge)
        q_candidates.append(
            (float(_rmse(q_validation_x @ q_weights, q_validation_y)), ridge, q_weights)
        )
        p_candidates.append(
            (float(_rmse(p_validation_x @ p_weights, p_validation_y)), ridge, p_weights)
        )
    q_validation_rmse, q_ridge, q_weights = min(
        q_candidates, key=lambda item: (item[0], item[1])
    )
    p_validation_rmse, p_ridge, p_weights = min(
        p_candidates, key=lambda item: (item[0], item[1])
    )
    charge_sigma = max(_rmse(q_train_x @ q_weights, q_train_y), 1.0e-6)
    dipole_sigma = max(_rmse(p_train_x @ p_weights, p_train_y), 1.0e-6)
    metrics = {
        split: _evaluate(
            features,
            split,
            scalar_mean=scalar_mean,
            scalar_scale=scalar_scale,
            vector_scale=vector_scale,
            charge_weights=q_weights,
            dipole_weights=p_weights,
            charge_sigma=charge_sigma,
            dipole_sigma=dipole_sigma,
        )
        for split in ("train", "validation", "test")
    }
    report = {
        "artifact": "mdp-mbis-source-head-prototype-result-v1",
        "claim_boundary": {
            "spice_qm_targets_read": True,
            "solvation_targets_read": False,
            "mnsol_or_freesolv_imported": False,
            "frozen_mdp_backbone": True,
            "new_source_identity": True,
            "pcm_or_solvation_accuracy_evaluated": False,
            "capability_admitted": False,
        },
        "inputs_sha256": actual,
        "preregistration_sha256": _sha256_file(preregistration_path),
        "dataset": {
            "eligible_configuration_count": len(samples),
            "eligible_molecule_count": len({sample.molecule for sample in samples}),
            "supported_atomic_numbers": list(SUPPORTED_ATOMIC_NUMBERS),
            "neutral_only": True,
            "split": "sha256(molecule) modulo 10: train 0-6, validation 7, test 8-9",
        },
        "head": {
            "scalar_feature_count": int(q_train_x.shape[1]),
            "vector_feature_count": int(p_train_x.shape[1]),
            "charge_ridge": q_ridge,
            "dipole_ridge": p_ridge,
            "charge_validation_rmse_e": q_validation_rmse,
            "dipole_validation_rmse_eangstrom": p_validation_rmse,
            "projection_charge_sigma_e": charge_sigma,
            "projection_dipole_sigma_eangstrom": dipole_sigma,
            "projection_metric": (
                "sum(dq/projection_charge_sigma)^2 + "
                "sum(dp/projection_dipole_sigma)^2"
            ),
        },
        "metrics": metrics,
        "runtime": runtime,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        np.savez_compressed(
            temporary / "source_head.npz",
            scalar_mean=scalar_mean,
            scalar_scale=scalar_scale,
            vector_scale=vector_scale,
            charge_weights=q_weights,
            dipole_weights=p_weights,
            charge_sigma=np.asarray(charge_sigma),
            dipole_sigma=np.asarray(dipole_sigma),
            atomic_numbers=np.asarray(SUPPORTED_ATOMIC_NUMBERS),
        )
        report["model_npz_sha256"] = _sha256_file(temporary / "source_head.npz")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), "test": metrics["test"]}, sort_keys=True))


if __name__ == "__main__":
    main()

