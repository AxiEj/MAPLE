#!/usr/bin/env python3
"""Audit the Pro-proposed work-dual MDP/POLAR tangent correction.

This target-free audit asks a deliberately narrow question.  MACE-MDP supplies
only its supervised molecular polarizability, while MACE-POLAR retains the
atomwise source topology and zero-field uniform source tangent.  Is the
nearest additive tangent correction compatible with the exact physical
source--field work pairing, without using a solvation or MBIS target?

The audit does not fit, damp, clip, or select using an accuracy value.  A pass
would only justify a later finite-field/root audit.  It would not admit an
energy, force, common functional, or chemical-accuracy capability.
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
SUSCEPTIBILITY_IMPLEMENTATION = REPO_ROOT / (
    "maple/solvation/release/uniform_susceptibility_replacement.py"
)
PRO_EVIDENCE = REPO_ROOT / (
    "docs/route2/evidence/mdp-polar-uniform-susceptibility-pro-20260817"
)
ARTIFACT = "route2-target-free-mdp-polar-work-dual-tangent-audit-v1"

LEFT_INVERSE_TOLERANCE = 1.0e-10
CHARGE_RESPONSE_TOLERANCE_E = 1.0e-10
MOLECULAR_CLOSURE_TOLERANCE = 1.0e-10
CORRECTION_RECIPROCITY_TOLERANCE = 1.0e-10
POLAR_MOLECULAR_SKEW_TOLERANCE = 1.0e-8


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


def _relative_difference(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(np.linalg.norm(right_array), np.finfo(np.float64).tiny)
    )


def _molecular_dipole_matrix(positions_angstrom: np.ndarray) -> np.ndarray:
    """Materialize the exact ``source4 -> molecular dipole`` map ``C``."""

    from maple.solvation.release.uniform_response_manifold import (
        molecular_dipole_eangstrom,
    )

    atom_count = len(positions_angstrom)
    result = np.empty((3, atom_count * 4), dtype=np.float64)
    for column in range(atom_count * 4):
        source = np.zeros((atom_count, 4), dtype=np.float64)
        source.reshape(-1)[column] = 1.0
        result[:, column] = molecular_dipole_eangstrom(
            positions_angstrom,
            source,
        )
    return result


def _physical_work_pairing(atom_count: int) -> np.ndarray:
    """Return ``W`` in ``source4.T @ W @ native_field8``.

    The original MACE-POLAR learned source occupies the published first radial
    GTO block.  The public radial pairing is identity in raw-dual ordering, so
    the exact rectangular work map is the transpose of the fixed ``4 -> 8``
    source embedding, repeated atomwise.
    """

    from maple.solvation.coupling.exact_gto import (
        mace_polar_learned_source_embedding_matrix,
    )
    from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING

    source_embedding = mace_polar_learned_source_embedding_matrix()
    block = source_embedding.T @ MACE_POLAR_RADIAL_GTO_PAIRING.block
    return np.kron(np.eye(atom_count, dtype=np.float64), block)


def _uniform_source_tangent(
    polar: object, atoms: object, basis: np.ndarray
) -> np.ndarray:
    zero = np.zeros((len(atoms), 8), dtype=np.float64)
    columns = [
        np.asarray(
            polar.field_jvp(atoms, zero, basis[:, :, axis]),
            dtype=np.float64,
        ).reshape(-1)
        for axis in range(3)
    ]
    result = np.column_stack(columns)
    if result.shape != (len(atoms) * 4, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("MACE-POLAR uniform source tangent is invalid.")
    return result


def _record(
    *,
    selection_index: int,
    selection_identity: dict[str, object],
    positions: np.ndarray,
    basis: np.ndarray,
    uniform_source_tangent: np.ndarray,
    arithmetic_chart: np.ndarray,
    alpha_mdp: np.ndarray,
) -> dict[str, Any]:
    atom_count = len(positions)
    uniform_basis = basis.reshape(atom_count * 8, 3)
    molecular_map = _molecular_dipole_matrix(positions)
    work_pairing = _physical_work_pairing(atom_count)
    polar_molecular = molecular_map @ uniform_source_tangent
    symmetric_polar = 0.5 * (polar_molecular + polar_molecular.T)
    skew_polar = 0.5 * (polar_molecular - polar_molecular.T)

    work_rhs = uniform_source_tangent.T @ work_pairing
    work_chart = np.linalg.solve(polar_molecular.T, work_rhs)
    symmetric_difference = -alpha_mdp - symmetric_polar
    tangent_coordinates = np.linalg.solve(polar_molecular, symmetric_difference)
    correction = uniform_source_tangent @ tangent_coordinates @ work_chart
    correction_work_jacobian = work_pairing.T @ correction
    corrected_molecular = polar_molecular + polar_molecular @ tangent_coordinates

    charge_response = uniform_source_tangent.reshape(atom_count, 4, 3)[:, 0, :]
    polar_scale_2 = max(
        np.linalg.norm(symmetric_polar, ord=2),
        np.finfo(np.float64).tiny,
    )
    correction_scale = max(
        np.linalg.norm(correction_work_jacobian),
        np.finfo(np.float64).tiny,
    )
    record: dict[str, Any] = {
        "selection_index": selection_index,
        "selection_identity": selection_identity,
        "atom_count": atom_count,
        "polar_uniform_molecular_jacobian": polar_molecular.tolist(),
        "mdp_molecular_polarizability_eangstrom2_per_volt": alpha_mdp.tolist(),
        "polar_molecular_skew_relative_2norm": float(
            np.linalg.norm(skew_polar, ord=2) / polar_scale_2
        ),
        "polar_molecular_skew_relative_frobenius": float(
            np.linalg.norm(skew_polar)
            / max(np.linalg.norm(symmetric_polar), np.finfo(np.float64).tiny)
        ),
        "arithmetic_chart_work_duality_defect": _relative_difference(
            polar_molecular.T @ arithmetic_chart,
            work_rhs,
        ),
        "work_chart_left_inverse_error": float(
            np.linalg.norm(work_chart @ uniform_basis - np.eye(3))
        ),
        "maximum_uniform_charge_response_e": float(
            np.max(np.abs(np.sum(charge_response, axis=0)))
        ),
        "symmetric_molecular_closure_error": _relative_difference(
            0.5 * (corrected_molecular + corrected_molecular.T),
            -alpha_mdp,
        ),
        "remaining_molecular_skew_relative": float(
            np.linalg.norm(0.5 * (corrected_molecular - corrected_molecular.T))
            / max(np.linalg.norm(alpha_mdp), np.finfo(np.float64).tiny)
        ),
        "correction_work_reciprocity_defect": float(
            np.linalg.norm(correction_work_jacobian - correction_work_jacobian.T)
            / correction_scale
        ),
        "relative_uniform_tangent_correction_norm": float(
            np.linalg.norm(uniform_source_tangent @ tangent_coordinates)
            / max(
                np.linalg.norm(uniform_source_tangent),
                np.finfo(np.float64).tiny,
            )
        ),
        "polar_molecular_condition_number": float(np.linalg.cond(polar_molecular)),
    }
    record["gates"] = {
        "polar_molecular_skew": (
            record["polar_molecular_skew_relative_2norm"]
            <= POLAR_MOLECULAR_SKEW_TOLERANCE
        ),
        "work_chart_left_inverse": (
            record["work_chart_left_inverse_error"] <= LEFT_INVERSE_TOLERANCE
        ),
        "charge_response": (
            record["maximum_uniform_charge_response_e"] <= CHARGE_RESPONSE_TOLERANCE_E
        ),
        "symmetric_molecular_closure": (
            record["symmetric_molecular_closure_error"] <= MOLECULAR_CLOSURE_TOLERANCE
        ),
        "correction_reciprocity": (
            record["correction_work_reciprocity_defect"]
            <= CORRECTION_RECIPROCITY_TOLERANCE
        ),
    }
    return record


def run(*, device: str) -> dict[str, object]:
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
        checkpoint_path=prior.MDP_CHECKPOINT,
        device="cpu",
    )
    radial = build_official_mace_polar_1_m_radial_gto_adapter(
        checkpoint_path=prior.POLAR_CHECKPOINT,
        device=device,
        long_range_evaluator_profile=(
            MACE_POLAR_ANALYTIC_GAUSSIAN_MULTIPOLE_EVALUATOR_ID
        ),
    )
    polar = MACEPolarOriginalSourceNativeFieldAdapter(radial)

    records: list[dict[str, Any]] = []
    for selection_index, selected in enumerate(selections):
        selection_identity = selected["selection_identity"]
        numbers, positions = _load_geometry(selection_identity)
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            info={"charge": 0, "multiplicity": 1},
        )
        basis = np.stack(
            [
                affine_uniform_native_field(positions, np.eye(3)[axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        uniform_tangent = _uniform_source_tangent(polar, atoms, basis)
        alpha = np.asarray(
            mdp.evaluate(atoms).public_polarizability_eangstrom2_per_volt,
            dtype=np.float64,
        )
        chart = prepare_uniform_susceptibility_field_transform(
            positions_angstrom=positions,
            uniform_native_basis=basis,
            polar_zero_uniform_source_jacobian=uniform_tangent.reshape(
                len(atoms), 4, 3
            ),
            mdp_molecular_polarizability_eangstrom2_per_volt=alpha,
        )
        records.append(
            _record(
                selection_index=selection_index,
                selection_identity=selection_identity,
                positions=positions,
                basis=basis,
                uniform_source_tangent=uniform_tangent,
                arithmetic_chart=chart.uniform_gradient_left_inverse,
                alpha_mdp=alpha,
            )
        )

    gate_names = tuple(records[0]["gates"])
    gate_summary = {
        gate: all(bool(record["gates"][gate]) for record in records)
        for gate in gate_names
    }
    payload: dict[str, object] = {
        "artifact": ARTIFACT,
        "schema_version": 1,
        "claim_boundary": {
            "accuracy_or_capability_admitted": False,
            "fitting_or_post_training_performed": False,
            "experimental_solvation_or_mbis_values_used": False,
            "opened_tail_selection_contains_prior_accuracy_values": True,
            "only_selection_identity_used_from_opened_tail": True,
            "full_native_work_reciprocity_audited": False,
            "finite_field_and_coupled_root_gates_audited": False,
            "pure_mace_polar_in_scope": False,
        },
        "scientific_question": (
            "Does the nearest work-dual additive tangent correction survive "
            "the zero-field molecular skew and exact work-duality gates?"
        ),
        "decision_rule": {
            "continue": (
                "All zero-field gates pass; proceed to finite-field passivity, "
                "full native reciprocity, field support, and coupled-root audits."
            ),
            "stop": (
                "Any material polar molecular skew terminates the Pro-proposed "
                "strict work-dual zero-training candidate; do not open an "
                "accuracy panel for that candidate."
            ),
        },
        "thresholds": {
            "polar_molecular_skew_relative_2norm": (POLAR_MOLECULAR_SKEW_TOLERANCE),
            "work_chart_left_inverse_error": LEFT_INVERSE_TOLERANCE,
            "maximum_uniform_charge_response_e": (CHARGE_RESPONSE_TOLERANCE_E),
            "symmetric_molecular_closure_error": (MOLECULAR_CLOSURE_TOLERANCE),
            "correction_work_reciprocity_defect": (CORRECTION_RECIPROCITY_TOLERANCE),
        },
        "input_sha256": {
            "dataset": _sha256_file(prior.DATASET),
            "mace_mdp_checkpoint": _sha256_file(prior.MDP_CHECKPOINT),
            "mace_polar_checkpoint": _sha256_file(prior.POLAR_CHECKPOINT),
            "selection_preregistration": _sha256_file(SELECTION_PREREGISTRATION),
            "susceptibility_implementation": _sha256_file(
                SUSCEPTIBILITY_IMPLEMENTATION
            ),
            "pro_prompt": _sha256_file(PRO_EVIDENCE / "q20-prompt.txt"),
            "pro_answer": _sha256_file(PRO_EVIDENCE / "q20-answer.md"),
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
        "records": records,
        "summary": {
            "record_count": len(records),
            "maximum_polar_molecular_skew_relative_2norm": max(
                record["polar_molecular_skew_relative_2norm"] for record in records
            ),
            "maximum_arithmetic_chart_work_duality_defect": max(
                record["arithmetic_chart_work_duality_defect"] for record in records
            ),
            "maximum_work_chart_left_inverse_error": max(
                record["work_chart_left_inverse_error"] for record in records
            ),
            "maximum_symmetric_molecular_closure_error": max(
                record["symmetric_molecular_closure_error"] for record in records
            ),
            "maximum_correction_work_reciprocity_defect": max(
                record["correction_work_reciprocity_defect"] for record in records
            ),
            "maximum_relative_uniform_tangent_correction_norm": max(
                record["relative_uniform_tangent_correction_norm"] for record in records
            ),
            "gates": gate_summary,
            "all_zero_field_gates_pass": all(gate_summary.values()),
        },
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
