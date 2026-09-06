#!/usr/bin/env python3
"""One-time O2 L-BFGS optimization closure for the frozen P13 A3 pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from maple.solvation.models.passive_p13_response import (  # noqa: E402
    p13_induced_source_analytic_torch,
    p13_response_mep_torch,
    radial_source_molecular_dipole_torch,
    radial_source_total_charge_torch,
)
from tools.route2_release import (  # noqa: E402
    train_vqm24_mdp_p13_passive_response_pilot as pilot,
)


SELF_REPO_PATH = "tools/route2_release/continue_vqm24_mdp_p13_a3_lbfgs.py"
ARTIFACT = "route2-vqm24-mdp-p13-a3-o2-lbfgs-closure-v1"
PREREGISTRATION_ARTIFACT = (
    "route2-vqm24-mdp-p13-a3-o2-lbfgs-closure-prereg-v1"
)
MAXIMUM_STEPS = 500
MAXIMUM_OBJECTIVE_EVALUATIONS = 5000
PLATEAU_BLOCK_STEPS = 50
PLATEAU_BLOCK_COUNT = 3
PLATEAU_RELATIVE_IMPROVEMENT = 0.0025
LBFGS_HISTORY_SIZE = 100
LBFGS_LEARNING_RATE = 1.0
LBFGS_MAX_EVAL_PER_STEP = 10
LBFGS_TOLERANCE_GRAD = 1.0e-12
LBFGS_TOLERANCE_CHANGE = 1.0e-14


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


class AnalyticA3Response(pilot.MDPBlockPassiveP13Response):
    """Frozen A3 architecture using the exact derivative of the same factor."""

    def __init__(self) -> None:
        super().__init__(include_edges=True)

    def forward(self, data: dict[str, Any], *, create_graph: bool):
        del create_graph
        embedding, values = self._atom_coefficients(data)
        radial_gain = values[:, 0]
        vector_coefficients = values[:, 1:9].reshape(-1, 2, 2, 2)
        l2_gain = values[:, 9]
        edge_coefficients = self._edge_coefficients(
            data=data,
            embedding=embedding,
        )
        energies, radial_source, l2_source = p13_induced_source_analytic_torch(
            radial_fields=data["radial_fields"],
            l2_fields=data["l2_fields"],
            positions_angstrom=data["positions"],
            edge_index=data["edge_index"],
            atomic_polarizability=data["polarizabilities"],
            radial_gain=radial_gain,
            vector_coefficients=vector_coefficients,
            edge_coefficients=edge_coefficients,
            edge_cutoff=data["edge_cutoff"],
            scalar_radial_reshape=self.scalar_radial_reshape,
            vector_radial_reshape=self.vector_radial_reshape,
            l2_gain=l2_gain,
        )
        mep = p13_response_mep_torch(
            radial_source=radial_source,
            l2_source=l2_source,
            radial_operator=data["radial_operator"],
            l2_operator=data["l2_operator"],
        )
        dipole = radial_source_molecular_dipole_torch(
            radial_source=radial_source,
            positions_angstrom=data["positions"],
        )
        return {
            "energies": energies,
            "radial_source": radial_source,
            "l2_source": l2_source,
            "mep": mep,
            "dipole": dipole,
        }


@dataclass(frozen=True)
class ObjectiveValues:
    total: torch.Tensor
    data: torch.Tensor
    mep: torch.Tensor
    dipole: torch.Tensor
    regularization: torch.Tensor


def _objective(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    *,
    mep_scale: float,
    dipole_scale: float,
    create_graph: bool,
) -> ObjectiveValues:
    reference = next(model.parameters())
    mep_loss = torch.zeros((), dtype=reference.dtype, device=reference.device)
    dipole_loss = torch.zeros_like(mep_loss)
    for record in records:
        prediction = model(record, create_graph=create_graph)
        fit = record["fit_mask"]
        mep_loss = mep_loss + torch.mean(
            torch.sum(
                record["fit_weights"]
                * (
                    prediction["mep"][:, fit]
                    - record["target_mep"][:, fit]
                ).square(),
                dim=1,
            )
        ) / (mep_scale * mep_scale)
        dipole_loss = dipole_loss + torch.mean(
            torch.sum(
                (prediction["dipole"] - record["target_dipole"]).square(),
                dim=1,
            )
        ) / (dipole_scale * dipole_scale)
    mep_loss = mep_loss / len(records)
    dipole_loss = dipole_loss / len(records)
    data_loss = (
        pilot.MEP_LOSS_WEIGHT * mep_loss
        + pilot.DIPOLE_LOSS_WEIGHT * dipole_loss
    )
    regularization = 0.5 * pilot.WEIGHT_DECAY * sum(
        torch.sum(parameter.square()) for parameter in model.parameters()
    )
    return ObjectiveValues(
        total=data_loss + regularization,
        data=data_loss,
        mep=mep_loss,
        dipole=dipole_loss,
        regularization=regularization,
    )


def _float_objective(values: ObjectiveValues) -> dict[str, float]:
    return {
        name: float(getattr(values, name).detach().cpu())
        for name in ("total", "data", "mep", "dipole", "regularization")
    }


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (
                parameter.grad.reshape(-1)
                if parameter.grad is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter in model.parameters()
        ]
    )


def _backend_preflight(
    *,
    state_dict: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    mep_scale: float,
    dipole_scale: float,
    device: torch.device,
) -> dict[str, float | bool]:
    reference = pilot.MDPBlockPassiveP13Response(include_edges=True).to(device=device)
    analytic = AnalyticA3Response().to(device=device)
    reference.load_state_dict(state_dict, strict=True)
    analytic.load_state_dict(state_dict, strict=True)
    maximum_energy = 0.0
    maximum_radial = 0.0
    maximum_l2 = 0.0
    maximum_identity = 0.0
    maximum_charge = 0.0
    for record in records:
        expected = reference(record, create_graph=True)
        actual = analytic(record, create_graph=True)
        maximum_energy = max(
            maximum_energy,
            float(torch.max(torch.abs(actual["energies"] - expected["energies"]))),
        )
        maximum_radial = max(
            maximum_radial,
            float(
                torch.max(
                    torch.abs(actual["radial_source"] - expected["radial_source"])
                )
            ),
        )
        maximum_l2 = max(
            maximum_l2,
            float(torch.max(torch.abs(actual["l2_source"] - expected["l2_source"]))),
        )
        work = torch.sum(
            record["radial_fields"] * actual["radial_source"], dim=(1, 2)
        ) + torch.sum(record["l2_fields"] * actual["l2_source"], dim=(1, 2))
        maximum_identity = max(
            maximum_identity,
            float(torch.max(torch.abs(2.0 * actual["energies"] - work))),
        )
        maximum_charge = max(
            maximum_charge,
            float(
                torch.max(
                    torch.abs(
                        radial_source_total_charge_torch(actual["radial_source"])
                    )
                )
            ),
        )
    reference.zero_grad(set_to_none=True)
    analytic.zero_grad(set_to_none=True)
    reference_objective = _objective(
        reference,
        records,
        mep_scale=mep_scale,
        dipole_scale=dipole_scale,
        create_graph=True,
    )
    analytic_objective = _objective(
        analytic,
        records,
        mep_scale=mep_scale,
        dipole_scale=dipole_scale,
        create_graph=True,
    )
    reference_objective.total.backward()
    analytic_objective.total.backward()
    expected_gradient = _gradient_vector(reference)
    actual_gradient = _gradient_vector(analytic)
    maximum_gradient = float(torch.max(torch.abs(actual_gradient - expected_gradient)))
    relative_gradient = float(
        torch.linalg.vector_norm(actual_gradient - expected_gradient)
        / max(float(torch.linalg.vector_norm(expected_gradient)), torch.finfo(torch.float64).tiny)
    )
    source_scale = max(
        float(torch.max(torch.abs(expected_gradient))),
        1.0,
    )
    result: dict[str, float | bool] = {
        "maximum_energy_absolute": maximum_energy,
        "maximum_radial_source_absolute": maximum_radial,
        "maximum_l2_source_absolute": maximum_l2,
        "maximum_energy_work_identity_absolute": maximum_identity,
        "maximum_induced_charge_absolute": maximum_charge,
        "maximum_parameter_gradient_absolute": maximum_gradient,
        "parameter_gradient_relative": relative_gradient,
        "parameter_gradient_reference_scale": source_scale,
    }
    result["passed"] = bool(
        maximum_energy <= 1.0e-11
        and maximum_radial <= 1.0e-11
        and maximum_l2 <= 1.0e-11
        and maximum_identity <= 1.0e-11
        and maximum_charge <= 1.0e-12
        and maximum_gradient <= 1.0e-9
        and relative_gradient <= 1.0e-8
    )
    return result


def _plateau(best: list[float]) -> tuple[bool, tuple[float, float, float] | None]:
    required = PLATEAU_BLOCK_COUNT * PLATEAU_BLOCK_STEPS
    if len(best) <= required:
        return False, None
    k = len(best) - 1
    boundaries = [k - required, k - 2 * PLATEAU_BLOCK_STEPS, k - PLATEAU_BLOCK_STEPS, k]
    improvements = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        denominator = max(abs(best[start]), np.finfo(float).tiny)
        improvements.append((best[start] - best[end]) / denominator)
    return all(value < PLATEAU_RELATIVE_IMPROVEMENT for value in improvements), tuple(improvements)


def run(args: argparse.Namespace) -> dict[str, Any]:
    preregistration_path = args.preregistration.expanduser().resolve(strict=True)
    mdp_root = args.mdp_features.expanduser().resolve(strict=True)
    pilot_root = args.pilot_result.expanduser().resolve(strict=True)
    output_directory = args.output_directory.expanduser().resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    preregistration = json.loads(preregistration_path.read_text())
    if (
        preregistration.get("artifact") != PREREGISTRATION_ARTIFACT
        or preregistration.get("status") != "locked-before-o2-continuation"
    ):
        raise ValueError("O2 preregistration is invalid.")
    if _sha256(SOURCE_ROOT / SELF_REPO_PATH) != preregistration["source"][
        "runner_sha256"
    ]:
        raise RuntimeError("O2 runner changed after preregistration.")
    response_entry = preregistration["source"]["analytic_response"]
    if _sha256(SOURCE_ROOT / response_entry["path"]) != response_entry["sha256"]:
        raise RuntimeError("Analytic response backend changed after freeze.")
    pilot_result_path = pilot_root / "result.json"
    checkpoint_path = pilot_root / "A3" / "head.pt"
    if (
        _sha256(pilot_result_path)
        != preregistration["parent_pilot"]["result_sha256"]
        or _sha256(checkpoint_path)
        != preregistration["parent_pilot"]["a3_checkpoint_sha256"]
    ):
        raise RuntimeError("Frozen A3 parent changed before O2.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA O2 continuation is unavailable.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_mean = np.asarray(checkpoint["atom_feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(checkpoint["atom_feature_scale"], dtype=np.float64)
    parent_path = SOURCE_ROOT / preregistration["observable_parent"]["path"]
    parent = json.loads(parent_path.read_text())
    records = [
        pilot._record_data(
            record,
            mdp_root=mdp_root,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            device=device,
        )
        for record in parent["records"]
    ]
    model = AnalyticA3Response().to(device=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    mep_scale = float(checkpoint["mep_scale"])
    dipole_scale = float(checkpoint["dipole_scale"])
    preflight = _backend_preflight(
        state_dict=checkpoint["state_dict"],
        records=records,
        mep_scale=mep_scale,
        dipole_scale=dipole_scale,
        device=device,
    )
    if not preflight["passed"]:
        raise RuntimeError("Analytic O2 backend failed the frozen equivalence preflight.")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LEARNING_RATE,
        max_iter=1,
        max_eval=LBFGS_MAX_EVAL_PER_STEP,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    objective_evaluations = 0
    initial = _objective(
        model,
        records,
        mep_scale=mep_scale,
        dipole_scale=dipole_scale,
        create_graph=False,
    )
    objective_evaluations += 1
    initial_values = _float_objective(initial)
    best_values = dict(initial_values)
    best_state = _clone_state(model)
    best_history = [initial_values["total"]]
    history: list[dict[str, Any]] = [
        {"step": 0, "objective_evaluations": 1, **initial_values}
    ]
    plateau_reached = False
    plateau_improvements = None
    termination = "hard-step-cap"

    def closure():
        nonlocal objective_evaluations
        optimizer.zero_grad(set_to_none=True)
        values = _objective(
            model,
            records,
            mep_scale=mep_scale,
            dipole_scale=dipole_scale,
            create_graph=False,
        )
        objective_evaluations += 1
        if objective_evaluations > MAXIMUM_OBJECTIVE_EVALUATIONS:
            raise RuntimeError("O2 objective-evaluation cap exceeded.")
        values.total.backward()
        return values.total

    for step in range(1, MAXIMUM_STEPS + 1):
        optimizer.step(closure)
        if objective_evaluations >= MAXIMUM_OBJECTIVE_EVALUATIONS:
            termination = "hard-objective-evaluation-cap"
            break
        with torch.no_grad():
            current = _objective(
                model,
                records,
                mep_scale=mep_scale,
                dipole_scale=dipole_scale,
                create_graph=False,
            )
        objective_evaluations += 1
        current_values = _float_objective(current)
        if current_values["total"] < best_values["total"]:
            best_values = dict(current_values)
            best_state = _clone_state(model)
        best_history.append(min(best_history[-1], current_values["total"]))
        plateau_reached, plateau_improvements = _plateau(best_history)
        history.append(
            {
                "step": step,
                "objective_evaluations": objective_evaluations,
                "best_total": best_history[-1],
                **current_values,
                "plateau_block_relative_improvements": (
                    list(plateau_improvements)
                    if plateau_improvements is not None
                    else None
                ),
            }
        )
        if plateau_reached:
            termination = "preregistered-plateau"
            break
    model.load_state_dict(best_state, strict=True)
    output_directory.mkdir(parents=True, exist_ok=False)
    checkpoint_output = output_directory / "head.pt"
    torch.save(
        {
            **checkpoint,
            "state_dict": best_state,
            "parent_a3_checkpoint_sha256": _sha256(checkpoint_path),
            "optimization": "bounded-full-batch-LBFGS-strong-wolfe-v1",
        },
        checkpoint_output,
    )
    metrics = None
    aggregate = None
    gates = None
    gate_passed = False
    if plateau_reached:
        metrics, aggregate = pilot._candidate_metrics(model=model, records=records)
        gates = pilot._gate_candidate(aggregate, preregistration["gates"])
        gate_passed = all(gates.values())
    status = (
        "pass-a3-o2-optimization-closure"
        if plateau_reached and gate_passed
        else (
            "fail-a3-o2-frozen-gates"
            if plateau_reached
            else "fail-a3-o2-no-plateau"
        )
    )
    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "runner_path": SELF_REPO_PATH,
            "runner_sha256": _sha256(SOURCE_ROOT / SELF_REPO_PATH),
            "preregistration_file_sha256": _sha256(preregistration_path),
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "parent_pilot_result_sha256": _sha256(pilot_result_path),
            "parent_a3_checkpoint_sha256": _sha256(checkpoint_path),
            "output_checkpoint_sha256": _sha256(checkpoint_output),
        },
        "backend_preflight": preflight,
        "optimization": {
            "optimizer": "torch.optim.LBFGS",
            "line_search": "strong_wolfe",
            "maximum_steps": MAXIMUM_STEPS,
            "maximum_objective_evaluations": MAXIMUM_OBJECTIVE_EVALUATIONS,
            "accepted_outer_steps": len(history) - 1,
            "objective_evaluations": objective_evaluations,
            "termination": termination,
            "plateau_reached": plateau_reached,
            "plateau_block_relative_improvements": (
                list(plateau_improvements)
                if plateau_improvements is not None
                else None
            ),
            "initial_objective": initial_values,
            "best_objective": best_values,
            "history": history,
        },
        "metrics": metrics,
        "aggregate": aggregate,
        "gates": gates,
        "gate_passed": gate_passed,
        "decision": {
            "A3_selected_for_grouped_cv": gate_passed,
            "A4_authorized": plateau_reached and not gate_passed,
            "cross_factor_C_authorized": False,
            "validation_or_blind_opened": False,
            "maple_capability_admitted": False,
        },
        "claim_boundary": {
            "same_A3_architecture_and_objective": True,
            "analytic_source_parameterized_separately": False,
            "development_train_formulas_only": True,
            "audit_used_for_early_stopping": False,
            "grouped_formula_cv_completed": False,
            "validation_or_blind_formula_opened": False,
            "experimental_solvation_target_read": False,
            "pcm_or_cavity_used": False,
            "model_selected_or_admitted": False,
            "maple_capability_admitted": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "dtype": "float64",
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    shutil.copy2(preregistration_path, output_directory / "preregistration.json")
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path.chmod(0o444)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--mdp-features", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
