#!/usr/bin/env python3
"""Fixed-geometry V2 conductor SCRF roots and unmodified local-response tests.

Uses a private immutable checkpoint spec, the stock stationary core and stock
Anderson step utility. No production sources, registry, tensors, response
spectrum, or physical output charges are edited/projected by this diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_mace_ef_cos_input_semantics import _source_hashes, write_json_once
from audit_mace_ef_v2_trace_inputs import (
    verified_bytes,
    state_digest,
    SOURCE_MEMBER,
    CHECKPOINT_SHA256,
)
from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING as PAIR,
)

CANDIDATE_SHA256 = "0cac472aa566363a00f5f3fbc41e7fb8bc5e1f5bb8aa8cba654ebac3fe4dc902"


def validate_parent_binding(parent, protocol):
    receipts = [v for v in parent["variants"] if v["mode"] == "centered_v"]
    if len(receipts) != 1:
        raise ValueError("expected exactly one centered-V parent receipt")
    receipt = receipts[0]
    if not (
        receipt["checkpoint_sha256"] == protocol["candidate_sha256"] == CANDIDATE_SHA256
        and receipt["state_dict_sha256"]
        == protocol["candidate_state_dict_sha256"]
        == parent["state_dict_sha256"]
        and receipt["changed_members"] == [SOURCE_MEMBER]
        and receipt["parent_checkpoint_sha256"] == CHECKPOINT_SHA256
    ):
        raise ValueError(
            "qualified parent evidence does not bind this candidate program"
        )
    primary = [r for r in parent["records"] if r["mode"] == "centered_v"]
    if not primary or not all(
        r["gate_assessment"]["all_executed_checks_passed"] for r in primary
    ):
        raise ValueError("candidate input canaries are not qualified")


def tangent_basis(atom_count):
    from maple.function.calculator.extra_correction.implicit.route2_response import (
        FixedChargeCoordinates,
    )

    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, (int, np.integer))
        or atom_count < 1
    ):
        raise ValueError("atom_count must be a positive integer")
    coordinates = FixedChargeCoordinates(atom_count)
    return np.column_stack(
        [
            PAIR.density_to_field_order(coordinates.expand(v)).reshape(-1)
            for v in np.eye(coordinates.dimension)
        ]
    )


def fd_jacobian(function, point, directions, step):
    point, directions = np.asarray(point, dtype=float), np.asarray(
        directions, dtype=float
    )
    if (
        point.ndim != 1
        or not point.size
        or directions.ndim != 2
        or directions.shape[0] != point.size
        or not directions.shape[1]
        or not np.isfinite(point).all()
        or not np.isfinite(directions).all()
        or not np.isfinite(step)
        or step <= 0
    ):
        raise ValueError("invalid finite-difference inputs")
    columns = []
    for direction in directions.T:
        plus = np.asarray(function(point + step * direction), dtype=float)
        minus = np.asarray(function(point - step * direction), dtype=float)
        if (
            plus.ndim != 1
            or plus.shape != minus.shape
            or not np.isfinite(plus).all()
            or not np.isfinite(minus).all()
        ):
            raise ValueError("invalid finite-difference function output")
        columns.append((plus - minus) / (2 * step))
    return np.column_stack(columns)


def local_metrics(
    K,
    H,
    J_fd,
    *,
    support_absolute=1e-10,
    support_relative=1e-10,
    reciprocity_tolerance=0.01,
    positive_curvature_tolerance=0.01,
    continuum_reciprocity_relative_tolerance=1e-8,
):
    K, H, J_fd = (np.asarray(x, dtype=float) for x in (K, H, J_fd))
    if (
        K.ndim != 2
        or not K.shape[0]
        or K.shape[0] != K.shape[1]
        or H.shape != K.shape
        or J_fd.shape != K.shape
        or not all(np.isfinite(x).all() for x in (K, H, J_fd))
    ):
        raise ValueError("local matrices must be finite, square and shape-matched")
    tolerances = (
        support_absolute,
        support_relative,
        reciprocity_tolerance,
        positive_curvature_tolerance,
        continuum_reciprocity_relative_tolerance,
    )
    if not np.isfinite(tolerances).all() or min(tolerances) < 0:
        raise ValueError("local diagnostic tolerances must be nonnegative")
    anti_K = float(np.linalg.norm(0.5 * (K - K.T)))
    if anti_K > continuum_reciprocity_relative_tolerance * max(
        1.0, float(np.linalg.norm(K, 2))
    ):
        raise ValueError("continuum reciprocity failed")
    eigen_T, vectors_T = np.linalg.eigh(-0.5 * (K + K.T))
    threshold = support_absolute + support_relative * max(0.0, float(eigen_T[-1]))
    if eigen_T[0] < -threshold:
        raise ValueError("continuum has a nonpassive positive-energy direction")
    support = eigen_T > threshold
    if not np.any(support):
        raise ValueError("empty positive continuum-response support")
    factor = vectors_T[:, support] * np.sqrt(eigen_T[support])
    symmetric_H = 0.5 * (H + H.T)
    anti_H = float(np.linalg.norm(0.5 * (H - H.T)))
    gamma_raw = -factor.T @ H @ factor
    C = np.eye(int(support.sum())) + factor.T @ symmetric_H @ factor
    C_eigen = np.linalg.eigvalsh(C)
    H_eigen = np.linalg.eigvalsh(symmetric_H)
    J = H @ K  # retain full neutral tangent, including cross-support response
    singular = np.linalg.svd(np.eye(len(J)) - J, compute_uv=False)
    direct_singular = np.linalg.svd(np.eye(len(J)) - J_fd, compute_uv=False)
    singular_flag = singular[-1] <= 1e-12 * max(1.0, float(singular[0]))
    return {
        "continuum_eigenvalues_T": eigen_T.tolist(),
        "continuum_support_threshold": threshold,
        "continuum_support_rank": int(support.sum()),
        "continuum_support_nullity": int((~support).sum()),
        "continuum_antisymmetry_norm": anti_K,
        "electronic_eigenvalues": H_eigen.tolist(),
        "electronic_maximum_eigenvalue": float(H_eigen[-1]),
        "electronic_passivity_passed": bool(
            H_eigen[-1] <= positive_curvature_tolerance
        ),
        "symmetric_spectrum_nonpositive": bool(H_eigen[-1] <= 0),
        "passivity_gate_meaning": "legacy numerical tolerance, not a certificate of strict physical concavity",
        "electronic_antisymmetry_norm": anti_H,
        "electronic_reciprocity_passed": anti_H <= reciprocity_tolerance,
        "joint_normalized_curvature": C.tolist(),
        "joint_eigenvalues": C_eigen.tolist(),
        "joint_minimum_eigenvalue": float(C_eigen[0]),
        "joint_positive": bool(C_eigen[0] > 0),
        "weighted_feedback_antisymmetry_norm": float(
            np.linalg.norm(0.5 * (gamma_raw - gamma_raw.T), 2)
        ),
        "chain_jacobian_raw": J.tolist(),
        "fixed_point_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(J)))),
        "residual_sigma_min": float(singular[-1]),
        "residual_condition_number": (
            None if singular_flag else float(singular[0] / singular[-1])
        ),
        "direct_fixed_point_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(J_fd)))
        ),
        "direct_residual_sigma_min": float(direct_singular[-1]),
        "jacobian_chain_relative_error": float(
            np.linalg.norm(J - J_fd, 2)
            / max(1.0, np.linalg.norm(J, 2), np.linalg.norm(J_fd, 2))
        ),
    }


def private_candidate_spec(checkpoint):
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef_specs import (
        MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
    )

    payload = verified_bytes(checkpoint, CANDIDATE_SHA256)
    # A new instance, deliberately NOT passed to the global registration function.
    return replace(
        MACE_POLAR_EF_V2_CHECKPOINT_SPEC,
        name="mace-polar-ef-v2-centered-v-trace-diagnostic-v1",
        model_id="mace-polar-ef-v2-centered-v-trace-diagnostic-v1",
        model_family="mace-polar-ef-v2-centered-v-trace-diagnostic",
        checkpoint_sha256=CANDIDATE_SHA256,
        checkpoint_size=len(payload),
        field_input="atomwise [V,grad(V)]; graph-mean-centered internal V; raw terminal qV",
    )


class ElectronicMonitor:
    def __init__(self, coupling, positions):
        import torch

        self.coupling = coupling
        self.positions = torch.tensor(
            positions, device=coupling.electronic.device, dtype=torch.float32
        )
        self.states = []
        self.label = "unlabeled"

    def sample(self, field):
        field = self.coupling._validated_field(field, name="audit field")
        energy, source, density = self.coupling._electronic_source(
            self.positions, field
        )
        charge = self.coupling.config.electronic.total_charge
        record = {
            "label": self.label,
            "field_cartesian": field.tolist(),
            "actual_float32_field_cartesian": np.asarray(
                field, dtype=np.float32
            ).tolist(),
            "energy_ev": energy,
            "source_raw": source.tolist(),
            "density_diagnostic": density.tolist(),
            "source_charge_error_e": abs(float(source[:, 0].sum()) - charge),
            "auxiliary_charge_error_e": abs(float(density[:, 0].sum()) - charge),
        }
        self.states.append(record)  # retain failing raw states too
        source = self.coupling._validated_source(source)
        if (
            record["auxiliary_charge_error_e"]
            > self.coupling.config.scf.charge_drift_tolerance
        ):
            raise RuntimeError("auxiliary charge closure failed")
        return energy, source


def solve_seed(monitor, continuum, seed):
    """Cold-start test harness; the numerical step implementation is stock.

    The primary root is still stock coupling.evaluate(). This short harness
    varies initialization without pretending a seed is an electronic gradient.
    """
    from maple.function.calculator.extra_correction.implicit.route2_fixed_point import (
        FixedPointSample,
        next_fixed_point_density,
        SAFEGUARDED_ANDERSON_SOLVER,
    )

    coupling, settings = monitor.coupling, monitor.coupling.config.scf
    source = coupling._validated_source(seed)
    history, trace = [], []
    for iteration in range(1, settings.maximum_iterations + 1):
        field = continuum.drive_cartesian(source, warm_start=False)
        _, mapped = monitor.sample(field)
        residual = mapped - source
        maximum = float(np.max(np.abs(residual)))
        trace.append(
            {
                "iteration": iteration,
                "source_raw": source.tolist(),
                "maximum_residual": maximum,
            }
        )
        if maximum <= settings.source_residual_tolerance:
            return {
                "status": "converged",
                "source_raw": source.tolist(),
                "field_cartesian": field.tolist(),
                "iterations": iteration,
                "trace": trace,
            }
        history.append(FixedPointSample(density=source, residual=residual))
        step = next_fixed_point_density(
            history,
            solver=SAFEGUARDED_ANDERSON_SOLVER,
            mixing=settings.mixing,
            anderson_depth=settings.anderson_depth,
            anderson_regularization=settings.anderson_regularization,
            anderson_coefficient_l1_limit=settings.anderson_coefficient_l1_limit,
            anderson_step_ratio_limit=settings.anderson_step_ratio_limit,
        )
        trace[-1]["step_method"] = step.method
        trace[-1]["step_fallback_reason"] = step.fallback_reason
        trace[-1]["step_coefficient_l1"] = step.coefficient_l1
        source = coupling._validated_source(step.density)
    return {
        "status": "not-converged",
        "trace": trace,
        "last_source_raw": source.tolist(),
    }


def ledger(monitor, continuum, source):
    field = continuum.drive_cartesian(source, warm_start=False)
    energy, mapped = monitor.sample(field)
    U = float(continuum.energy_ev(source))
    pair = PAIR.pair(source, field)
    mapped_field = continuum.drive_cartesian(mapped, warm_start=False)
    return {
        "electronic_energy_ev": energy,
        "continuum_energy_ev": U,
        "source_field_pairing_ev": pair,
        "total_energy_ev": energy + U - pair,
        "half_identity_error_ev": abs(U - 0.5 * pair),
        "fresh_source_residual": float(np.max(np.abs(mapped - source))),
        "mapped_source_field_residual": float(np.max(np.abs(mapped_field - field))),
        "field_cartesian": field.tolist(),
    }


def audit_local(monitor, continuum, source, protocol):
    n = len(source)
    B = tangent_basis(n)
    d = np.tile(
        [
            protocol["metric"]["charge_scale_e"],
            *([protocol["metric"]["dipole_scale_e_angstrom"]] * 3),
        ],
        n,
    )
    E0 = protocol["metric"]["energy_scale_ev"]
    field = continuum.drive_cartesian(source, warm_start=False)
    scaled_source = PAIR.density_to_field_order(source).reshape(-1) / d
    scaled_field = d * field.reshape(-1) / E0
    zero = continuum.drive_cartesian(np.zeros_like(source), warm_start=False)
    if np.max(np.abs(zero)) > 1e-10:
        raise RuntimeError("continuum drive is not linear through zero")
    K_columns = np.column_stack(
        [
            d
            * continuum.drive_cartesian(
                PAIR.field_to_density_order((d * direction).reshape(n, 4)),
                warm_start=False,
            ).reshape(-1)
            / E0
            for direction in B.T
        ]
    )
    K = B.T @ K_columns

    def response(v):
        _, raw = monitor.sample((E0 * v / d).reshape(n, 4))
        return PAIR.density_to_field_order(raw).reshape(-1) / d

    def feedback(a):
        raw = PAIR.field_to_density_order((d * a).reshape(n, 4))
        raw = monitor.coupling._validated_source(raw)
        _, mapped = monitor.sample(continuum.drive_cartesian(raw, warm_start=False))
        return PAIR.density_to_field_order(mapped).reshape(-1) / d

    scans = []
    partial_result = {
        "tangent_basis_cartesian": B.tolist(),
        "source_metric_diagonal": d.tolist(),
        "continuum_K": K.tolist(),
        "continuum_field_columns": K_columns.tolist(),
        "scan": scans,
        "all_local_gates_passed": False,
        "physical_concavity_certified": False,
    }
    for field_step, source_step in zip(
        protocol["field_fd_steps"], protocol["source_fd_steps"], strict=True
    ):
        entry = {"field_step": field_step, "source_step": source_step}
        scans.append(entry)
        try:
            monitor.label = f"root-H-h={field_step}"
            full_H_columns = fd_jacobian(response, scaled_field, B, field_step)
            H = B.T @ full_H_columns
            entry.update(
                {
                    "raw_H": H.tolist(),
                    "full_H_columns": full_H_columns.tolist(),
                    "H_charge_tangent_leakage_norm": float(
                        np.linalg.norm(full_H_columns[0::4].sum(axis=0)) / np.sqrt(n)
                    ),
                }
            )
            monitor.label = f"root-J-h={source_step}"
            full_J_columns = fd_jacobian(feedback, scaled_source, B, source_step)
            J = B.T @ full_J_columns
            entry.update(
                {
                    "raw_J_fd": J.tolist(),
                    "full_J_columns": full_J_columns.tolist(),
                    "J_charge_tangent_leakage_norm": float(
                        np.linalg.norm(full_J_columns[0::4].sum(axis=0)) / np.sqrt(n)
                    ),
                }
            )
            entry["metrics"] = local_metrics(
                K,
                H,
                J,
                support_absolute=protocol["support_absolute"],
                support_relative=protocol["support_relative"],
                reciprocity_tolerance=protocol[
                    "electronic_reciprocity_absolute_tolerance"
                ],
                positive_curvature_tolerance=protocol[
                    "positive_electronic_curvature_tolerance"
                ],
                continuum_reciprocity_relative_tolerance=protocol[
                    "continuum_reciprocity_relative_tolerance"
                ],
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            entry["interpretation_error"] = _error(exc)
    if any("interpretation_error" in entry for entry in scans):
        return {
            **partial_result,
            "gate_results": {"local_metrics_completed": False},
            "scope": "partial failure evidence; no local stability conclusion",
        }
    fine, previous = scans[-1], scans[-2]
    Hfine, Hprevious = np.asarray(fine["raw_H"]), np.asarray(previous["raw_H"])
    m = fine["metrics"]
    C_error = float(
        np.linalg.norm(
            np.asarray(m["joint_normalized_curvature"])
            - previous["metrics"]["joint_normalized_curvature"],
            2,
        )
    ) + max(
        m["weighted_feedback_antisymmetry_norm"],
        previous["metrics"]["weighted_feedback_antisymmetry_norm"],
    )
    step_error = float(
        np.linalg.norm(Hfine - Hprevious, 2)
        / max(1.0, np.linalg.norm(Hfine, 2), np.linalg.norm(Hprevious, 2))
    )
    monitor.label = "root-gauge-base"
    base_energy, base_source = monitor.sample(field)
    gauge = []
    for offset in protocol["gauge_offsets_ev_per_e"]:
        shifted = field.copy()
        shifted[:, 0] += offset
        monitor.label = f"root-gauge-{offset}"
        energy, shifted_source = monitor.sample(shifted)
        gauge.append(
            {
                "offset": offset,
                "energy_error_ev": abs(
                    energy
                    - base_energy
                    - monitor.coupling.config.electronic.total_charge * offset
                ),
                "source_max_abs_change": float(
                    np.max(np.abs(shifted_source - base_source))
                ),
            }
        )
    passed = {
        "electronic_passivity": m["electronic_passivity_passed"],
        "electronic_reciprocity": m["electronic_reciprocity_passed"],
        "finite_difference_resolution": step_error
        <= protocol["fd_step_relative_tolerance"],
        "jacobian_chain": m["jacobian_chain_relative_error"]
        <= protocol["fd_chain_relative_tolerance"],
        "positive_joint_curvature_resolved": m["joint_minimum_eigenvalue"] > C_error,
        "root_gauge": all(
            g["energy_error_ev"] <= protocol["gauge_energy_abs_tolerance_ev"]
            and g["source_max_abs_change"] <= protocol["gauge_source_abs_tolerance"]
            for g in gauge
        ),
    }
    return {
        **partial_result,
        "gauge": gauge,
        "H_step_relative_error": step_error,
        "C_resolution_proxy": C_error,
        "gate_results": passed,
        "all_local_gates_passed": all(passed.values()),
        "gate_interpretation": "numerical tolerances only; positive electronic modes must be reported separately",
        "scope": "local supported minimum and selected root basin; not full saddle positivity/global uniqueness/admission",
    }


def _error(exc):
    return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}


def confirm_extremal_mode(monitor, continuum, source, local, protocol, confirmation):
    """Independently difference a mode using actual continuum-generated fields."""
    if not local.get("scan") or "metrics" not in local["scan"][-1]:
        return {
            "status": "not-tested",
            "reason": "local matrix interpretation unavailable",
        }
    B = np.asarray(local["tangent_basis_cartesian"])
    d = np.asarray(local["source_metric_diagonal"])
    K = np.asarray(local["continuum_K"])
    H = np.asarray(local["scan"][-1]["raw_H"])
    E0 = protocol["metric"]["energy_scale_ev"]
    eigen_T, vectors_T = np.linalg.eigh(-0.5 * (K + K.T))
    threshold = protocol["support_absolute"] + protocol["support_relative"] * max(
        0.0, float(eigen_T[-1])
    )
    R = vectors_T[:, eigen_T > threshold]
    values, vectors = np.linalg.eigh(R.T @ (0.5 * (H + H.T)) @ R)
    mode = R @ vectors[:, -1]
    if mode[np.argmax(np.abs(mode))] < 0:
        mode = -mode
    source_mode = R @ np.linalg.solve(R.T @ K @ R, R.T @ mode)
    delta_raw = PAIR.field_to_density_order(
        (d * (B @ source_mode)).reshape(source.shape)
    )
    field_direction = continuum.drive_cartesian(delta_raw, warm_start=False)
    realized = B.T @ (d * field_direction.reshape(-1) / E0)
    projection_error = float(np.linalg.norm(realized - mode))
    if projection_error > confirmation["projection_tolerance"]:
        raise RuntimeError("extremal field mode is not reproduced by continuum")
    base_field = continuum.drive_cartesian(source, warm_start=False)
    derivatives = []
    for index in range(confirmation["center_repeats"]):
        monitor.label = f"mode-center-repeat:{index}"
        _, raw = monitor.sample(base_field)
        derivatives.append(PAIR.pair(raw, field_direction) / E0)
    scans = []
    for step in confirmation["steps"]:
        sides = []
        for sign in (1, -1):
            perturbed_source = monitor.coupling._validated_source(
                source + sign * step * delta_raw
            )
            actual_field = continuum.drive_cartesian(perturbed_source, warm_start=False)
            error = float(
                np.max(
                    np.abs(actual_field - (base_field + sign * step * field_direction))
                )
            )
            if error > confirmation["field_linearity_tolerance"]:
                raise RuntimeError("continuum ray is not linear")
            monitor.label = f"mode-h={step}:sign={sign}"
            energy, raw = monitor.sample(actual_field)
            sides.append(
                {
                    "sign": sign,
                    "derivative": PAIR.pair(raw, field_direction) / E0,
                    "energy_ev": energy,
                    "electronic_state_index": len(monitor.states) - 1,
                    "field_linearity_error": error,
                }
            )
        scans.append(
            {
                "step": step,
                "sides": sides,
                "curvature": (sides[0]["derivative"] - sides[1]["derivative"])
                / (2 * step),
            }
        )
    cartesian_mode = (B @ mode).reshape(source.shape)
    return {
        "status": "completed",
        "predicted_curvature": float(values[-1]),
        "reduced_field_mode": mode.tolist(),
        "source_mode_raw": delta_raw.tolist(),
        "field_direction_cartesian": field_direction.tolist(),
        "realization_error": projection_error,
        "potential_component_squared_norm": float(np.sum(cartesian_mode[:, 0] ** 2)),
        "gradient_component_squared_norm": float(np.sum(cartesian_mode[:, 1:] ** 2)),
        "center_derivative_repeats": derivatives,
        "scan": scans,
        "all_measured_curvatures_positive": all(row["curvature"] > 0 for row in scans),
        "claim_boundary": "local direction confirmation only; original numerical thresholds are unchanged",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_bytes = args.preregistration.read_bytes()
    protocol = json.loads(protocol_bytes)
    confirmation_bytes = args.confirmation.read_bytes() if args.confirmation else None
    confirmation = json.loads(confirmation_bytes) if confirmation_bytes else None
    if (
        confirmation
        and confirmation["base_preregistration_sha256"]
        != hashlib.sha256(protocol_bytes).hexdigest()
    ):
        raise ValueError("confirmation is not bound to this base protocol")
    if protocol["candidate_sha256"] != CANDIDATE_SHA256:
        raise ValueError("unknown candidate")
    parent = json.loads(
        verified_bytes(args.input_audit, protocol["input_audit_sha256"])
    )
    validate_parent_binding(parent, protocol)
    spec = private_candidate_spec(args.checkpoint)
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef import (
        MACEPolarEFConfig,
        MACEPolarEFEnergyModel,
    )
    from maple.function.calculator.extra_correction.implicit.mace_polar_ef_stationary import (
        MACEPolarEFSCFSettings,
    )
    from maple.function.cosmors_torch.mace_ef_segment_cosmo import (
        MACEPolarEFSegmentCOSMOConfig,
        MACEPolarEFSegmentCOSMOCoupling,
    )
    from maple.function.cosmors_torch.segment_cosmo import (
        TorchSegmentCOSMO,
        TorchSegmentCOSMOConfig,
    )
    from maple.function.mlip_cosmo_rs import open_cosmors_24a_cavity_radii
    from ase.data import chemical_symbols
    import torch

    sources = [
        Path(__file__),
        HERE / "audit_mace_ef_cos_input_semantics.py",
        HERE / "audit_mace_ef_v2_trace_inputs.py",
    ]
    sources += [
        ROOT / "maple/function/calculator/extra_correction/implicit" / n
        for n in (
            "mace_polar_ef.py",
            "mace_polar_ef_specs.py",
            "mace_polar_ef_stationary.py",
            "route2_fixed_point.py",
            "route2_response.py",
            "electrostatic_pairing.py",
        )
    ]
    sources += [
        ROOT / "maple/function/cosmors_torch" / n
        for n in ("segment_cosmo.py", "mace_ef_segment_cosmo.py", "surface.py")
    ]
    sources.append(ROOT / "maple/function/mlip_cosmo_rs.py")
    hashes = _source_hashes(sources)
    records = []
    for molecule in protocol["molecules"]:
        print(f"SCRF {molecule['name']}", flush=True)
        row = {"molecule": molecule, "seed_runs": []}
        monitor = model = coupling = None
        try:
            electronic_config = MACEPolarEFConfig(
                checkpoint_path=str(args.checkpoint),
                checkpoint_spec=spec,
                atomic_numbers=tuple(molecule["atomic_numbers"]),
                total_charge=molecule["total_charge"],
                spin_multiplicity=molecule["multiplicity"],
            )
            model = MACEPolarEFEnergyModel(electronic_config)
            if state_digest(model.model) != protocol["candidate_state_dict_sha256"]:
                raise RuntimeError("candidate tensors differ")
            continuum_model = TorchSegmentCOSMO(
                TorchSegmentCOSMOConfig(
                    atomic_numbers=tuple(molecule["atomic_numbers"]),
                    radii_angstrom=tuple(
                        open_cosmors_24a_cavity_radii(
                            [chemical_symbols[z] for z in molecule["atomic_numbers"]]
                        )
                    ),
                    angular_degree=protocol["angular_degree"],
                )
            )
            if (
                continuum_model.config.configuration_sha256
                != protocol["continuum_configuration_sha256"][molecule["name"]]
            ):
                raise RuntimeError("cavity configuration mismatch")
            config = MACEPolarEFSegmentCOSMOConfig(
                electronic=electronic_config,
                continuum=continuum_model,
                scf=MACEPolarEFSCFSettings(**protocol["scf_settings"]),
            )
            coupling = MACEPolarEFSegmentCOSMOCoupling(config, electronic_model=model)
            positions = coupling._canonical_positions(
                np.asarray(molecule["positions_angstrom"])
            )
            row.update(
                {
                    "canonical_positions_angstrom": positions.tolist(),
                    "configuration": config.as_dict(),
                    "configuration_sha256": config.configuration_sha256,
                }
            )
            continuum = config.continuum_functional.bind_geometry(positions)
            monitor = ElectronicMonitor(coupling, positions)
            monitor.label = "gas-source"
            _, gas = monitor.sample(np.zeros((len(positions), 4)))
            try:
                stock = coupling.evaluate(positions)
                row["stock_root"] = {
                    "status": "converged",
                    "source_raw": stock.source_raw.tolist(),
                    "iterations": stock.iterations,
                    "total_energy_ev": stock.total_energy_ev,
                    "provenance": dict(stock.provenance),
                }
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                row["stock_root"] = _error(exc)
            seeds = [
                (f"gas-scale-{scale}", scale * gas) for scale in protocol["seed_scales"]
            ]
            perturb = PAIR.field_to_density_order(
                tangent_basis(len(gas))[:, 0].reshape(gas.shape)
            )
            seeds.append(
                (
                    "gas-plus-neutral",
                    gas + protocol["extra_neutral_seed_amplitude"] * perturb,
                )
            )
            for name, seed in seeds:
                monitor.label = f"seed:{name}"
                try:
                    result = solve_seed(monitor, continuum, seed)
                    if result["status"] == "converged":
                        monitor.label = f"ledger:{name}"
                        result["ledger"] = ledger(
                            monitor, continuum, np.asarray(result["source_raw"])
                        )
                    row["seed_runs"].append(
                        {"name": name, "initial_source_raw": seed.tolist(), **result}
                    )
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    row["seed_runs"].append({"name": name, **_error(exc)})
            if row["stock_root"]["status"] == "converged":
                source = np.asarray(row["stock_root"]["source_raw"])
                row["seed_agreement"] = [
                    {
                        "name": s["name"],
                        "source_max_abs": float(
                            np.max(np.abs(np.asarray(s["source_raw"]) - source))
                        ),
                        "total_energy_abs_ev": abs(
                            s["ledger"]["total_energy_ev"]
                            - row["stock_root"]["total_energy_ev"]
                        ),
                    }
                    for s in row["seed_runs"]
                    if s["status"] == "converged"
                ]
                row["root_repeatability_passed"] = (
                    len(row["seed_agreement"]) == len(seeds)
                    and all(
                        s["source_max_abs"] <= protocol["root_source_max_abs_tolerance"]
                        and s["total_energy_abs_ev"]
                        <= protocol["root_energy_abs_tolerance_ev"]
                        for s in row["seed_agreement"]
                    )
                    and all(
                        s["ledger"]["half_identity_error_ev"]
                        <= continuum.identity_tolerance_ev
                        and s["ledger"]["fresh_source_residual"]
                        <= config.scf.source_residual_tolerance
                        for s in row["seed_runs"]
                        if s["status"] == "converged"
                    )
                )
                row["local_stability"] = audit_local(
                    monitor, continuum, source, protocol
                )
                if confirmation:
                    row["extremal_mode_confirmation"] = confirm_extremal_mode(
                        monitor,
                        continuum,
                        source,
                        row["local_stability"],
                        protocol,
                        confirmation,
                    )
            if state_digest(model.model) != protocol["candidate_state_dict_sha256"]:
                raise RuntimeError("execution mutated candidate tensors")
            row["status"] = "diagnostics-completed"
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row.update(_error(exc))
        if monitor is not None:
            row["electronic_states"] = monitor.states
        records.append(row)
        if model is not None:
            row["candidate_state_sha256_after"] = state_digest(model.model)
            if (
                row["candidate_state_sha256_after"]
                != protocol["candidate_state_dict_sha256"]
            ):
                raise RuntimeError(
                    "candidate state changed in a failed or completed diagnostic"
                )
        del monitor, model, coupling
    if (
        hashes != _source_hashes(sources)
        or args.preregistration.read_bytes() != protocol_bytes
        or (args.confirmation and args.confirmation.read_bytes() != confirmation_bytes)
    ):
        raise RuntimeError("audit source/protocol drift")
    write_json_once(
        args.output,
        {
            "schema_version": 1,
            "artifact": "mace-ef-v2-scrf-water-methanol-v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_spec": spec.as_dict(),
            "input_audit_sha256": protocol["input_audit_sha256"],
            "candidate_state_dict_sha256": protocol["candidate_state_dict_sha256"],
            "confirmation_preregistration_sha256": (
                hashlib.sha256(confirmation_bytes).hexdigest()
                if confirmation_bytes
                else None
            ),
            "runtime": {
                "python": sys.executable,
                "numpy": str(np.__version__),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
            "preregistration_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "source_files_sha256": hashes,
            "records": records,
            "release_admitted": False,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


if __name__ == "__main__":
    main()
