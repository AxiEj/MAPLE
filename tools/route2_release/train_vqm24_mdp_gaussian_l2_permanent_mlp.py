#!/usr/bin/env python3
"""Train the bounded local-invariant Q-A permanent-source head."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tools.route2_release import (  # noqa: E402
    train_vqm24_mdp_gaussian_l2_permanent_pilot as base,
)


SELF_REPO_PATH = "tools/route2_release/train_vqm24_mdp_gaussian_l2_permanent_mlp.py"
ARTIFACT = "route2-vqm24-mdp-gaussian-l2-permanent-local-mlp-v1"
HIDDEN_WIDTH = 16
ELEMENT_EMBEDDING_DIMENSION = 4
NEIGHBOR_SCALE_ANGSTROM = 2.0


class PermanentLocalMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.element_embedding = torch.nn.Embedding(
            len(base.ELEMENTS),
            ELEMENT_EMBEDDING_DIMENSION,
            dtype=torch.float64,
        )
        input_dimension = 2 * ELEMENT_EMBEDDING_DIMENSION + 5
        self.atom_head = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, HIDDEN_WIDTH, dtype=torch.float64),
            torch.nn.SiLU(),
            torch.nn.Linear(HIDDEN_WIDTH, 6, dtype=torch.float64),
        )
        self.beta_q = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.beta_p = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        torch.nn.init.normal_(self.element_embedding.weight, std=0.05)
        torch.nn.init.zeros_(self.atom_head[-1].weight)
        torch.nn.init.zeros_(self.atom_head[-1].bias)

    def _atom_outputs(self, data: dict[str, Any]) -> torch.Tensor:
        element = data["element_indices"]
        positions = data["positions"]
        embedding = self.element_embedding(element)
        neighbor = torch.zeros_like(embedding)
        atom_count = len(element)
        for atom_i in range(atom_count):
            for atom_j in range(atom_count):
                if atom_i == atom_j:
                    continue
                distance = torch.linalg.vector_norm(
                    positions[atom_j] - positions[atom_i]
                )
                cutoff = torch.where(
                    distance < base.PAIR_CUTOFF_ANGSTROM,
                    0.5
                    * (
                        1.0
                        + torch.cos(
                            math.pi * distance / base.PAIR_CUTOFF_ANGSTROM
                        )
                    ),
                    torch.zeros_like(distance),
                )
                weight = cutoff * torch.exp(-distance / NEIGHBOR_SCALE_ANGSTROM)
                neighbor = neighbor.index_add(
                    0,
                    torch.tensor([atom_i], device=positions.device),
                    (weight * embedding[atom_j])[None, :],
                )
        alpha = data["alpha_l2"]
        alpha_norm = torch.linalg.vector_norm(alpha, dim=1, keepdim=True)
        scalar_features = torch.cat(
            (
                embedding,
                neighbor,
                data["charges"][:, None],
                torch.linalg.vector_norm(data["dipoles"], dim=1, keepdim=True),
                alpha_norm,
                torch.sum(data["pair_l2"][:, 0].square(), dim=1, keepdim=True),
                torch.sum(data["pair_l2"][:, 1].square(), dim=1, keepdim=True),
            ),
            dim=1,
        )
        return self.atom_head(scalar_features)

    def forward(self, data: dict[str, Any]):
        outputs = self._atom_outputs(data)
        radial_null_output = outputs[:, 0]
        dipole_scale = outputs[:, 1]
        electronegativity = outputs[:, 2]
        l2_gain = outputs[:, 3:6]
        positions = data["positions"]
        atom_count = len(positions)
        dtype = positions.dtype
        device = positions.device
        delta_charge = torch.zeros(atom_count, dtype=dtype, device=device)
        for atom_i in range(atom_count):
            for atom_j in range(atom_i + 1, atom_count):
                distance = torch.linalg.vector_norm(
                    positions[atom_j] - positions[atom_i]
                )
                cutoff = torch.where(
                    distance < base.PAIR_CUTOFF_ANGSTROM,
                    0.5
                    * (
                        1.0
                        + torch.cos(
                            math.pi * distance / base.PAIR_CUTOFF_ANGSTROM
                        )
                    ),
                    torch.zeros_like(distance),
                )
                transfer = cutoff * (
                    electronegativity[atom_j] - electronegativity[atom_i]
                )
                delta_charge = delta_charge.index_add(
                    0,
                    torch.tensor([atom_i, atom_j], device=device),
                    torch.stack((transfer, -transfer)),
                )
        moment = torch.full((2,), 0.5, dtype=dtype, device=device)
        radial_null = torch.tensor(
            [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)],
            dtype=dtype,
            device=device,
        )
        physical_charge = data["charges"] + delta_charge
        q_radial = (
            physical_charge[:, None] * (moment + self.beta_q * radial_null)[None]
            + radial_null_output[:, None] * radial_null[None]
        )
        physical_dipole = data["dipoles"] * (1.0 + dipole_scale[:, None])
        p_radial = physical_dipole[:, None, :] * (
            moment + self.beta_p * radial_null
        )[None, :, None]
        source = torch.zeros((atom_count, 8), dtype=dtype, device=device)
        source[:, :2] = q_radial
        source[:, 2:5] = p_radial[:, 0, (1, 2, 0)]
        source[:, 5:8] = p_radial[:, 1, (1, 2, 0)]
        quadrupole = (
            l2_gain[:, 0, None] * data["alpha_l2"]
            + l2_gain[:, 1, None] * data["pair_l2"][:, 0]
            + l2_gain[:, 2, None] * data["pair_l2"][:, 1]
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training is unavailable.")
    random.seed(base.SEED)
    np.random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(base.SEED)
    records = [
        base._record_data(record, mdp_root=mdp_root, device=device)
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
    model = PermanentLocalMLP().to(device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 512:
        raise RuntimeError("Q-A local MLP exceeds the frozen 512-parameter cap.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base.LEARNING_RATE,
        weight_decay=base.WEIGHT_DECAY,
    )
    history = []
    for epoch in range(base.EPOCHS):
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
        mep_loss /= len(records)
        dipole_loss /= len(records)
        loss = base.MEP_LOSS_WEIGHT * mep_loss + base.DIPOLE_LOSS_WEIGHT * dipole_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == base.EPOCHS - 1:
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
            item["charge_residual_e"] = float(torch.abs(torch.sum(source[:, :2])))
            item["quadrupole_norm"] = float(torch.linalg.vector_norm(quadrupole))
            metrics.append(item)
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_directory / "head.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "elements": base.ELEMENTS,
            "mep_scale": mep_scale,
            "dipole_scale": dipole_scale,
        },
        checkpoint_path,
    )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": "complete-development-pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": base._sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "base_runner_sha256": base._sha256(SOURCE_ROOT / base.SELF_REPO_PATH),
            "preregistration_file_sha256": base._sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "mdp_feature_manifest_sha256": base._sha256(mdp_root / "manifest.json"),
            "checkpoint_sha256": base._sha256(checkpoint_path),
        },
        "architecture": {
            "name": "Q-A-local-invariant-MLP-development-pilot",
            "trainable_parameter_count": parameter_count,
            "hidden_width": HIDDEN_WIDTH,
            "element_embedding_dimension": ELEMENT_EMBEDDING_DIMENSION,
            "neighbor_scale_angstrom": NEIGHBOR_SCALE_ANGSTROM,
            "new_message_passing": False,
            "backbone_parameters_trainable": False,
        },
        "training": {
            "device": str(device),
            "dtype": "float64",
            "seed": base.SEED,
            "epochs": base.EPOCHS,
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
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
    payload["result_sha256"] = base._canonical_sha256(payload)
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
