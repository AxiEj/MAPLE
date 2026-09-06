#!/usr/bin/env python3
"""Target-free finite-field audit of two MDP/POLAR response continuations.

The MACE-MDP molecular polarizability fixes only the zero-field uniform
susceptibility.  It does not determine how the correction should continue to
finite or nonuniform fields.  This audit therefore compares, without reading a
QM/MBIS/solvation value during the measurement, the two explicit continuations

``input``
    ``M_P(Tu)``;

``tangent``
    ``M_P(u) + J_P(0) U (X-I) G u``.

Both close the same zero-field molecular polarizability.  The reported finite
field passivity, reciprocity, nonlinear remainder, and mixed-response metrics
are structural diagnostics only, not chemical-accuracy evidence.  The seven
geometries are an already-opened tail selection whose preregistration contains
prior accuracy values; only its immutable geometry identities are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.route2_release import (  # noqa: E402
    run_mdp_polar_zero_training_source_gate as prior,
)
from tools.route2_release.audit_mdp_polar_uniform_susceptibility_covariance import (  # noqa: E402
    _load_geometry,
)

SELECTION_PREREGISTRATION = REPO_ROOT / (
    "docs/route2/preregistrations/mdp-polar-ewald-gauge-l3-tail-v1.json"
)
IMPLEMENTATION = REPO_ROOT / (
    "maple/solvation/release/uniform_susceptibility_replacement.py"
)
ARTIFACT = "route2-target-free-mdp-polar-uniform-nonlinearity-audit-v1"
AMPLITUDES_V_PER_ANGSTROM = (0.005, 0.02, 0.05, 0.08)
_DIRECTION_ROWS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, -2.0, 0.5),
)
NONUNIFORM_GRADIENT_RMS_V_PER_ANGSTROM = 0.02


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _directions() -> np.ndarray:
    values = np.asarray(_DIRECTION_ROWS, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _relative(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(np.linalg.norm(right_array), np.finfo(np.float64).tiny)
    )


def _gradient_only_nonuniform_field(atom_count: int, *, seed: int) -> np.ndarray:
    """Return a dimensionally homogeneous field in ``ker(G)``."""

    rng = np.random.default_rng(seed)
    gradients = rng.normal(size=(atom_count, 3))
    gradients -= np.mean(gradients, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(gradients * gradients, axis=1))))
    gradients *= NONUNIFORM_GRADIENT_RMS_V_PER_ANGSTROM / rms
    field = np.zeros((atom_count, 8), dtype=np.float64)
    field[:, (4, 2, 3)] = gradients
    field[:, (7, 5, 6)] = gradients
    return field


def _molecular_jacobian(
    positions: np.ndarray, source_jacobian: np.ndarray
) -> np.ndarray:
    from maple.solvation.release.uniform_response_manifold import (
        molecular_dipole_eangstrom,
    )

    return np.column_stack(
        [
            molecular_dipole_eangstrom(positions, source_jacobian[:, :, axis])
            for axis in range(3)
        ]
    )


def _source_and_uniform_jacobian(
    *, strategy: str, atoms, positions, polar, chart, field, basis
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "input":
        evaluated_field = chart.transform_field(field)
        source = polar.evaluate_source(atoms, evaluated_field)
        jacobian = np.stack(
            [
                polar.field_jvp(
                    atoms,
                    evaluated_field,
                    chart.transform_field_direction(basis[:, :, axis]),
                )
                for axis in range(3)
            ],
            axis=-1,
        )
    elif strategy == "tangent":
        source = chart.tangent_corrected_source(
            polar.evaluate_source(atoms, field), field
        )
        jacobian = np.stack(
            [
                polar.field_jvp(atoms, field, basis[:, :, axis])
                + chart.tangent_source_direction(basis[:, :, axis])
                for axis in range(3)
            ],
            axis=-1,
        )
    else:  # pragma: no cover - internal closed vocabulary
        raise ValueError(f"unknown strategy {strategy!r}")
    if source.shape != (len(positions), 4) or jacobian.shape != (
        len(positions),
        4,
        3,
    ):
        raise RuntimeError("response continuation returned an invalid shape.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(jacobian)):
        raise RuntimeError("response continuation returned a non-finite value.")
    return np.asarray(source), np.asarray(jacobian)


def _finite_field_record(
    *,
    strategy: str,
    atoms,
    positions: np.ndarray,
    polar,
    chart,
    zero_source: np.ndarray,
    basis: np.ndarray,
    alpha_mdp: np.ndarray,
    amplitude: float,
    direction: np.ndarray,
) -> dict[str, Any]:
    from maple.solvation.release.uniform_response_manifold import (
        affine_uniform_native_field,
        molecular_dipole_eangstrom,
    )

    gradient = amplitude * direction
    field = affine_uniform_native_field(positions, gradient)
    source, source_jacobian = _source_and_uniform_jacobian(
        strategy=strategy,
        atoms=atoms,
        positions=positions,
        polar=polar,
        chart=chart,
        field=field,
        basis=basis,
    )
    molecular_jacobian = _molecular_jacobian(positions, source_jacobian)
    alpha = -molecular_jacobian
    symmetric_alpha = 0.5 * (alpha + alpha.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_alpha)
    induced_source = source - zero_source
    linear_source = np.einsum(
        "nsc,c->ns", chart.transformed_uniform_source_jacobian, gradient
    )
    induced_dipole = molecular_dipole_eangstrom(positions, source) - (
        molecular_dipole_eangstrom(positions, zero_source)
    )
    linear_dipole = -(alpha_mdp @ gradient)
    return {
        "strategy": strategy,
        "amplitude_v_per_angstrom": amplitude,
        "direction": direction.tolist(),
        "charge_drift_e": float(np.sum(induced_source[:, 0])),
        "induced_source_norm": float(np.linalg.norm(induced_source)),
        "nonlinear_source_remainder_norm": float(
            np.linalg.norm(induced_source - linear_source)
        ),
        "relative_nonlinear_source_remainder": float(
            np.linalg.norm(induced_source - linear_source)
            / max(np.linalg.norm(linear_source), np.finfo(np.float64).tiny)
        ),
        "induced_dipole_norm_eangstrom": float(np.linalg.norm(induced_dipole)),
        "relative_induced_dipole_deviation_from_linear_mdp": _relative(
            induced_dipole, linear_dipole
        ),
        "finite_field_alpha_reciprocity_defect": float(
            np.linalg.norm(alpha - alpha.T)
            / max(np.linalg.norm(alpha), np.finfo(np.float64).tiny)
        ),
        "finite_field_symmetric_alpha_eigenvalues": eigenvalues.tolist(),
        "finite_field_minimum_symmetric_alpha_eigenvalue": float(eigenvalues[0]),
        "relative_symmetric_alpha_deviation_from_mdp": _relative(
            symmetric_alpha, alpha_mdp
        ),
    }


def _mixed_response_record(*, atoms, positions, polar, chart, basis, seed: int):
    zero = np.zeros((len(positions), 8), dtype=np.float64)
    nonuniform = _gradient_only_nonuniform_field(len(positions), seed=seed)
    if np.linalg.norm(chart.uniform_coordinates(nonuniform)) > 1.0e-13:
        raise RuntimeError("nonuniform diagnostic field is not in ker(G).")
    direction = np.asarray([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    uniform_direction = np.einsum("nfc,c->nf", basis, direction)
    original = polar.field_jvp(atoms, nonuniform, uniform_direction) - polar.field_jvp(
        atoms, zero, uniform_direction
    )
    transformed_direction = chart.transform_field_direction(uniform_direction)
    input_mixed = polar.field_jvp(
        atoms, nonuniform, transformed_direction
    ) - polar.field_jvp(atoms, zero, transformed_direction)
    tangent_mixed = (
        polar.field_jvp(atoms, nonuniform, uniform_direction)
        + chart.tangent_source_direction(uniform_direction)
        - polar.field_jvp(atoms, zero, uniform_direction)
        - chart.tangent_source_direction(uniform_direction)
    )
    return {
        "nonuniform_gradient_rms_v_per_angstrom": (
            NONUNIFORM_GRADIENT_RMS_V_PER_ANGSTROM
        ),
        "original_mixed_response_norm": float(np.linalg.norm(original)),
        "input_relative_change_to_original_mixed_response": _relative(
            input_mixed, original
        ),
        "tangent_relative_change_to_original_mixed_response": _relative(
            tangent_mixed, original
        ),
    }


def _summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in ("input", "tangent"):
        selected = [record for record in records if record["strategy"] == strategy]
        result[strategy] = {
            "record_count": len(selected),
            "maximum_absolute_charge_drift_e": max(
                abs(record["charge_drift_e"]) for record in selected
            ),
            "maximum_relative_nonlinear_source_remainder": max(
                record["relative_nonlinear_source_remainder"] for record in selected
            ),
            "maximum_relative_induced_dipole_deviation_from_linear_mdp": max(
                record["relative_induced_dipole_deviation_from_linear_mdp"]
                for record in selected
            ),
            "maximum_finite_field_alpha_reciprocity_defect": max(
                record["finite_field_alpha_reciprocity_defect"] for record in selected
            ),
            "minimum_finite_field_symmetric_alpha_eigenvalue": min(
                record["finite_field_minimum_symmetric_alpha_eigenvalue"]
                for record in selected
            ),
            "maximum_relative_symmetric_alpha_deviation_from_mdp": max(
                record["relative_symmetric_alpha_deviation_from_mdp"]
                for record in selected
            ),
        }
    return result


def run(*, device: str) -> dict[str, Any]:
    preregistration = json.loads(SELECTION_PREREGISTRATION.read_text())
    selections = preregistration["selection"]["records"]

    import torch
    from ase import Atoms

    from maple.solvation.api.profiles import (
        MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    from maple.solvation.models import (
        MACEPolarOriginalSourceNativeFieldAdapter,
        build_mace_mdp_moment_adapter,
        build_official_mace_polar_1_m_radial_gto_adapter,
    )
    from maple.solvation.release.uniform_response_manifold import (
        affine_uniform_native_field,
    )
    from maple.solvation.release.uniform_susceptibility_replacement import (
        prepare_uniform_susceptibility_field_transform,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mdp = build_mace_mdp_moment_adapter(
        checkpoint_path=prior.MDP_CHECKPOINT, device="cpu"
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=prior.POLAR_CHECKPOINT,
        device=device,
        long_range_evaluator_profile=MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID,
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)

    directions = _directions()
    finite_records: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []
    for index, selected in enumerate(selections):
        identity = selected["selection_identity"]
        numbers, positions = _load_geometry(identity)
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            info={"charge": 0, "multiplicity": 1},
        )
        alpha_mdp = mdp.evaluate(atoms).public_polarizability_eangstrom2_per_volt
        zero = np.zeros((len(atoms), 8), dtype=np.float64)
        zero_source = polar.evaluate_source(atoms, zero)
        basis = np.stack(
            [
                affine_uniform_native_field(positions, np.eye(3)[axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        polar_jacobian = np.stack(
            [polar.field_jvp(atoms, zero, basis[:, :, axis]) for axis in range(3)],
            axis=-1,
        )
        chart = prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=basis,
            polar_zero_uniform_source_jacobian=polar_jacobian,
            mdp_molecular_polarizability_eangstrom2_per_volt=alpha_mdp,
        )
        records_for_geometry: list[dict[str, Any]] = []
        for amplitude in AMPLITUDES_V_PER_ANGSTROM:
            for direction in directions:
                pair: dict[str, dict[str, Any]] = {}
                for strategy in ("input", "tangent"):
                    record = _finite_field_record(
                        strategy=strategy,
                        atoms=atoms,
                        positions=positions,
                        polar=polar,
                        chart=chart,
                        zero_source=zero_source,
                        basis=basis,
                        alpha_mdp=alpha_mdp,
                        amplitude=amplitude,
                        direction=direction,
                    )
                    record["selection_index"] = index
                    pair[strategy] = record
                    finite_records.append(record)
                    records_for_geometry.append(record)
                input_source_norm = pair["input"]["induced_source_norm"]
                tangent_source_norm = pair["tangent"]["induced_source_norm"]
                pair_difference = abs(input_source_norm - tangent_source_norm) / max(
                    tangent_source_norm, np.finfo(np.float64).tiny
                )
                pair["input"][
                    "relative_induced_source_norm_difference_to_tangent"
                ] = pair_difference
                pair["tangent"][
                    "relative_induced_source_norm_difference_to_input"
                ] = pair_difference
        geometry_records.append(
            {
                "selection_index": index,
                "selection_identity": identity,
                "chart_state_sha256": chart.state_sha256,
                "coordinate_transform_condition_number": (
                    chart.coordinate_transform_condition_number
                ),
                "mixed_uniform_nonuniform_response": _mixed_response_record(
                    atoms=atoms,
                    positions=positions,
                    polar=polar,
                    chart=chart,
                    basis=basis,
                    seed=2026081800 + index,
                ),
                "finite_field_record_count": len(records_for_geometry),
            }
        )

    summary = _summaries(finite_records)
    summary["input"]["maximum_relative_change_to_original_mixed_response"] = max(
        record["mixed_uniform_nonuniform_response"][
            "input_relative_change_to_original_mixed_response"
        ]
        for record in geometry_records
    )
    summary["tangent"]["maximum_relative_change_to_original_mixed_response"] = max(
        record["mixed_uniform_nonuniform_response"][
            "tangent_relative_change_to_original_mixed_response"
        ]
        for record in geometry_records
    )

    payload: dict[str, Any] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "claim_boundary": {
            "accuracy_or_capability_admitted": False,
            "finite_field_mdp_polarizability_treated_as_reference": False,
            "fitting_or_post_training_performed": False,
            "experimental_or_mbis_values_used_by_measurement": False,
            "opened_tail_selection_contains_prior_accuracy_values": True,
            "only_selection_identity_used_from_opened_tail": True,
            "pure_mace_polar_in_scope": False,
        },
        "scientific_question": (
            "Which zero-training continuation makes the smaller unsupported "
            "change beyond the shared zero-field MDP molecular polarizability?"
        ),
        "input_sha256": {
            "dataset": _sha256_file(prior.DATASET),
            "mace_mdp_checkpoint": _sha256_file(prior.MDP_CHECKPOINT),
            "mace_polar_checkpoint": _sha256_file(prior.POLAR_CHECKPOINT),
            "selection_preregistration": _sha256_file(SELECTION_PREREGISTRATION),
            "susceptibility_implementation": _sha256_file(IMPLEMENTATION),
            "runner": _sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
        },
        "protocol": {
            "amplitudes_v_per_angstrom": list(AMPLITUDES_V_PER_ANGSTROM),
            "directions": directions.tolist(),
            "nonuniform_gradient_rms_v_per_angstrom": (
                NONUNIFORM_GRADIENT_RMS_V_PER_ANGSTROM
            ),
            "input_continuation": "M_P(Tu)",
            "tangent_continuation": "M_P(u)+J_P(0)U(X-I)Gu",
        },
        "geometry_records": geometry_records,
        "finite_field_records": finite_records,
        "summary": summary,
    }
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
