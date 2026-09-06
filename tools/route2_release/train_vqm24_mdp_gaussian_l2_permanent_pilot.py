#!/usr/bin/env python3
"""Train the minimal Q-A MDP/Gaussian-l2 permanent-source pilot.

This is a bounded development fit on the frozen 32 train formulas.  The model
uses MDP q/p/alpha only as frozen geometry descriptors, preserves total charge
through local antisymmetric transfer, predicts l2 tensors from equivariant
carriers, and optimizes full-MEP/dipole observables only.  Rotated audit probes
never enter the gradient.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
from typing import Any

from ase.units import Bohr, Hartree
import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.coupling.exact_gto import (  # noqa: E402
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.gaussian_quadrupole import (  # noqa: E402
    gaussian_traceless_quadrupole_surface_operator,
)
from maple.solvation.coupling.point_quadrupole import (  # noqa: E402
    traceless_quadrupole_coefficients,
)


SELF_REPO_PATH = (
    "tools/route2_release/train_vqm24_mdp_gaussian_l2_permanent_pilot.py"
)
ARTIFACT = "route2-vqm24-mdp-gaussian-l2-permanent-pilot-v1"
ELEMENTS = (1, 6, 7, 8, 9, 15, 16, 17, 35)
ELEMENT_INDEX = {number: index for index, number in enumerate(ELEMENTS)}
SIGMA_L2_ANGSTROM = 1.5
PAIR_CUTOFF_ANGSTROM = 5.0
PAIR_RADIAL_SCALES_ANGSTROM = (1.5, 3.0)
SEED = 20260827
EPOCHS = 2000
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4
MEP_LOSS_WEIGHT = 0.90
DIPOLE_LOSS_WEIGHT = 0.10


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


def _pair_carriers(positions: np.ndarray) -> np.ndarray:
    atom_count = len(positions)
    carriers = np.zeros((atom_count, 2, 5), dtype=np.float64)
    identity = np.eye(3)
    for atom_i in range(atom_count):
        for atom_j in range(atom_count):
            if atom_i == atom_j:
                continue
            displacement = positions[atom_j] - positions[atom_i]
            distance = float(np.linalg.norm(displacement))
            if not 0.0 < distance < PAIR_CUTOFF_ANGSTROM:
                continue
            direction = displacement / distance
            cutoff = 0.5 * (1.0 + math.cos(math.pi * distance / PAIR_CUTOFF_ANGSTROM))
            tensor = np.outer(direction, direction) - identity / 3.0
            coefficients = traceless_quadrupole_coefficients(tensor[None])[0]
            for channel, scale in enumerate(PAIR_RADIAL_SCALES_ANGSTROM):
                carriers[atom_i, channel] += (
                    cutoff * math.exp(-distance / scale) * coefficients
                )
    return carriers


def _record_data(
    record: dict[str, Any],
    *,
    mdp_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    record_id = record["record_id"]
    surface_path = Path(record["inputs"]["surface"]["path"])
    with np.load(surface_path, allow_pickle=False) as surface:
        points = np.asarray(surface["surface_points_bohr"], dtype=np.float64)
        positions = np.asarray(surface["atom_positions_angstrom"], dtype=np.float64)
        weights = np.asarray(surface["quadrature_weights"], dtype=np.float64)
        partitions = np.asarray(surface["partition_indices"], dtype=np.int64)
        numbers = np.asarray(surface["atomic_numbers"], dtype=np.int64)
    with np.load(record["outputs"]["npz"], allow_pickle=False) as target_state:
        target_mep = np.asarray(
            target_state["total_surface_mep_hartree_per_e"],
            dtype=np.float64,
        )
        target_dipole = np.asarray(target_state["total_dipole_e_bohr"], dtype=np.float64)
    with np.load(mdp_root / record_id / "features.npz", allow_pickle=False) as mdp:
        charges = np.asarray(mdp["charges_e"], dtype=np.float64)
        dipoles = np.asarray(mdp["atomic_dipoles_eangstrom"], dtype=np.float64)
        alpha = np.asarray(
            mdp["atomic_polarizabilities_eangstrom2_per_volt"],
            dtype=np.float64,
        )
    if not np.array_equal(numbers, np.asarray(record["selection_record"]["atomic_numbers"])):
        raise RuntimeError("MDP/permanent target atom identities differ.")
    radial_operator = MACEPolarRadialGTOCoupling().surface_operator(
        FixedSurfaceGeometry(positions, points)
    ) / Hartree
    l2_operator = gaussian_traceless_quadrupole_surface_operator(
        points_bohr=points,
        centers_angstrom=positions,
        sigma_angstrom=SIGMA_L2_ANGSTROM,
    )
    alpha_coefficients = traceless_quadrupole_coefficients(alpha)
    pair_coefficients = _pair_carriers(positions)
    element_indices = np.asarray([ELEMENT_INDEX[int(number)] for number in numbers])
    fit = partitions == 0
    audit = partitions == 1
    normalized_weights = {
        "fit": weights[fit] / np.sum(weights[fit]),
        "audit": weights[audit] / np.sum(weights[audit]),
    }
    return {
        "record_id": record_id,
        "numbers": torch.as_tensor(numbers, dtype=torch.long, device=device),
        "element_indices": torch.as_tensor(
            element_indices, dtype=torch.long, device=device
        ),
        "positions": torch.as_tensor(positions, dtype=torch.float64, device=device),
        "charges": torch.as_tensor(charges, dtype=torch.float64, device=device),
        "dipoles": torch.as_tensor(dipoles, dtype=torch.float64, device=device),
        "alpha_l2": torch.as_tensor(
            alpha_coefficients, dtype=torch.float64, device=device
        ),
        "pair_l2": torch.as_tensor(
            pair_coefficients, dtype=torch.float64, device=device
        ),
        "radial_operator": torch.as_tensor(
            radial_operator, dtype=torch.float64, device=device
        ),
        "l2_operator": torch.as_tensor(l2_operator, dtype=torch.float64, device=device),
        "target_mep": torch.as_tensor(target_mep, dtype=torch.float64, device=device),
        "target_dipole": torch.as_tensor(
            target_dipole * Bohr, dtype=torch.float64, device=device
        ),
        "fit_mask": torch.as_tensor(fit, dtype=torch.bool, device=device),
        "audit_mask": torch.as_tensor(audit, dtype=torch.bool, device=device),
        "fit_weights": torch.as_tensor(
            normalized_weights["fit"], dtype=torch.float64, device=device
        ),
        "audit_weights": torch.as_tensor(
            normalized_weights["audit"], dtype=torch.float64, device=device
        ),
    }


class PermanentPilot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta_q = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.beta_p = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.dipole_scale = torch.nn.Parameter(
            torch.zeros((len(ELEMENTS),), dtype=torch.float64)
        )
        self.radial_null = torch.nn.Parameter(
            torch.zeros((len(ELEMENTS),), dtype=torch.float64)
        )
        self.electronegativity = torch.nn.Parameter(
            torch.zeros((len(ELEMENTS),), dtype=torch.float64)
        )
        self.l2_carrier_gain = torch.nn.Parameter(
            torch.zeros((len(ELEMENTS), 3), dtype=torch.float64)
        )

    def forward(self, data: dict[str, Any]):
        dtype = data["charges"].dtype
        device = data["charges"].device
        moment = torch.full((2,), 0.5, dtype=dtype, device=device)
        radial_null_vector = torch.tensor(
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
            dtype=dtype,
            device=device,
        )
        element = data["element_indices"]
        positions = data["positions"]
        atom_count = len(element)
        delta_charge = torch.zeros(atom_count, dtype=dtype, device=device)
        chi = self.electronegativity[element]
        for atom_i in range(atom_count):
            for atom_j in range(atom_i + 1, atom_count):
                distance = torch.linalg.vector_norm(
                    positions[atom_j] - positions[atom_i]
                )
                cutoff = torch.where(
                    distance < PAIR_CUTOFF_ANGSTROM,
                    0.5
                    * (
                        1.0
                        + torch.cos(
                            math.pi * distance / PAIR_CUTOFF_ANGSTROM
                        )
                    ),
                    torch.zeros_like(distance),
                )
                transfer = cutoff * (chi[atom_j] - chi[atom_i])
                delta_charge = delta_charge.index_add(
                    0,
                    torch.tensor([atom_i, atom_j], device=device),
                    torch.stack((transfer, -transfer)),
                )
        physical_charge = data["charges"] + delta_charge
        q_mix = moment + self.beta_q * radial_null_vector
        q_radial = (
            physical_charge[:, None] * q_mix[None, :]
            + self.radial_null[element, None] * radial_null_vector[None, :]
        )
        physical_dipole = data["dipoles"] * (
            1.0 + self.dipole_scale[element, None]
        )
        p_mix = moment + self.beta_p * radial_null_vector
        p_radial = physical_dipole[:, None, :] * p_mix[None, :, None]
        source = torch.zeros((atom_count, 8), dtype=dtype, device=device)
        source[:, :2] = q_radial
        # Cartesian x,y,z -> public raw y,z,x.
        source[:, 2:5] = p_radial[:, 0, (1, 2, 0)]
        source[:, 5:8] = p_radial[:, 1, (1, 2, 0)]
        gains = self.l2_carrier_gain[element]
        quadrupole = (
            gains[:, 0, None] * data["alpha_l2"]
            + gains[:, 1, None] * data["pair_l2"][:, 0]
            + gains[:, 2, None] * data["pair_l2"][:, 1]
        )
        mep = (
            data["radial_operator"] @ source.reshape(-1)
            + data["l2_operator"] @ quadrupole.reshape(-1)
        )
        molecular_dipole = torch.sum(
            physical_charge[:, None] * positions + physical_dipole,
            dim=0,
        )
        return mep, molecular_dipole, source, quadrupole


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    mdp_root = args.mdp_features.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    mdp_manifest = json.loads((mdp_root / "manifest.json").read_text())
    if len(preregistration["records"]) != 32 or len(mdp_manifest["records"]) != 32:
        raise RuntimeError("Permanent pilot requires 32 frozen records and descriptors.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA permanent-head training is unavailable.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    records = [
        _record_data(record, mdp_root=mdp_root, device=device)
        for record in preregistration["records"]
    ]
    mep_scale = float(
        np.median(
            [
                math.sqrt(
                    float(
                        torch.sum(
                            record["fit_weights"]
                            * record["target_mep"][record["fit_mask"]].square()
                        ).cpu()
                    )
                )
                for record in records
            ]
        )
    )
    dipole_scale = float(
        np.median(
            [float(torch.linalg.vector_norm(record["target_dipole"]).cpu()) for record in records]
        )
    )
    if mep_scale <= 0.0 or dipole_scale <= 0.0:
        raise RuntimeError("Permanent-head train scales must be positive.")
    model = PermanentPilot().to(device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        mep_loss = torch.zeros((), dtype=torch.float64, device=device)
        dipole_loss = torch.zeros((), dtype=torch.float64, device=device)
        for record in records:
            prediction, dipole, _source, _quadrupole = model(record)
            fit = record["fit_mask"]
            mep_loss = mep_loss + torch.sum(
                record["fit_weights"]
                * (prediction[fit] - record["target_mep"][fit]).square()
            ) / (mep_scale * mep_scale)
            dipole_loss = dipole_loss + torch.sum(
                (dipole - record["target_dipole"]).square()
            ) / (dipole_scale * dipole_scale)
        mep_loss = mep_loss / len(records)
        dipole_loss = dipole_loss / len(records)
        loss = MEP_LOSS_WEIGHT * mep_loss + DIPOLE_LOSS_WEIGHT * dipole_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "mep_loss": float(mep_loss.detach().cpu()),
                    "dipole_loss": float(dipole_loss.detach().cpu()),
                }
            )
    metrics = []
    with torch.no_grad():
        for record in records:
            prediction, dipole, source, quadrupole = model(record)
            item = {"record_id": record["record_id"]}
            for name in ("fit", "audit"):
                mask = record[f"{name}_mask"]
                weights = record[f"{name}_weights"]
                error = prediction[mask] - record["target_mep"][mask]
                item[f"{name}_mep_relative"] = float(
                    torch.sqrt(torch.sum(weights * error.square()))
                    / torch.sqrt(
                        torch.sum(weights * record["target_mep"][mask].square())
                    )
                )
            item["dipole_relative"] = float(
                torch.linalg.vector_norm(dipole - record["target_dipole"])
                / torch.linalg.vector_norm(record["target_dipole"])
            )
            item["charge_residual_e"] = float(
                torch.abs(torch.sum(source[:, :2]))
            )
            item["quadrupole_norm"] = float(torch.linalg.vector_norm(quadrupole))
            metrics.append(item)
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_directory / "head.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "elements": ELEMENTS,
            "sigma_l2_angstrom": SIGMA_L2_ANGSTROM,
            "pair_cutoff_angstrom": PAIR_CUTOFF_ANGSTROM,
            "pair_radial_scales_angstrom": PAIR_RADIAL_SCALES_ANGSTROM,
            "mep_scale": mep_scale,
            "dipole_scale": dipole_scale,
        },
        checkpoint_path,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "complete-development-pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_path": str(preregistration_path.relative_to(SOURCE_ROOT)),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "mdp_feature_manifest_sha256": _sha256(mdp_root / "manifest.json"),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "architecture": {
            "name": "Q-A-element-linear-carriers-development-pilot",
            "trainable_parameter_count": parameter_count,
            "elements": list(ELEMENTS),
            "l2_kernel": f"fixed-width Gaussian sigma={SIGMA_L2_ANGSTROM} A",
            "pair_cutoff_angstrom": PAIR_CUTOFF_ANGSTROM,
            "pair_radial_scales_angstrom": list(PAIR_RADIAL_SCALES_ANGSTROM),
            "charge_conservation": "local antisymmetric transfer plus radial-null shape",
            "backbone_parameters_trainable": False,
        },
        "training": {
            "device": str(device),
            "dtype": "float64",
            "seed": SEED,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "mep_loss_weight": MEP_LOSS_WEIGHT,
            "dipole_loss_weight": DIPOLE_LOSS_WEIGHT,
            "mep_scale_hartree_per_e": mep_scale,
            "dipole_scale_eangstrom": dipole_scale,
            "audit_points_used_in_gradient": False,
            "history": history,
        },
        "metrics": metrics,
        "aggregate": {
            "mean_fit_mep_relative": float(
                np.mean([item["fit_mep_relative"] for item in metrics])
            ),
            "mean_audit_mep_relative": float(
                np.mean([item["audit_mep_relative"] for item in metrics])
            ),
            "maximum_audit_mep_relative": max(
                item["audit_mep_relative"] for item in metrics
            ),
            "mean_dipole_relative": float(
                np.mean([item["dipole_relative"] for item in metrics])
            ),
            "maximum_charge_residual_e": max(
                item["charge_residual_e"] for item in metrics
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "claim_boundary": {
            "development_train_formulas_only": True,
            "validation_or_blind_formula_opened": False,
            "coefficient_label_used": False,
            "response_target_used": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "model_selected_or_admitted": False,
            "capability_admitted": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--mdp-features", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
