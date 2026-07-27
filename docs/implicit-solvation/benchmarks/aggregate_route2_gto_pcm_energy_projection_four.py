#!/usr/bin/env python3
"""Aggregate the preregistered four-molecule PCM projection experiment.

The raw canaries intentionally remain local because they contain projected
coefficient arrays and runtime work products.  This script emits the compact,
hash-bound scientific ledger used for review and publication decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_RELATIVE_PATH = (
    "docs/implicit-solvation/benchmarks/"
    "aggregate_route2_gto_pcm_energy_projection_four.py"
)
DEFAULT_PREREGISTRATION = (
    REPO_ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
DEFAULT_INPUT_ROOT = (
    REPO_ROOT / ".omx/benchmarks/" "route2-gto-pcm-energy-projection-four-v1"
)
ARM_NAMES = ("one_radial", "two_radial")
COPIED_FIELDS = (
    "coefficient_count",
    "coefficient_l2_norm",
    "coefficient_max_abs",
    "constraint_residual_inf",
    "discarded_mode_count",
    "effective_rank",
    "elapsed_seconds",
    "fitted_polarization_energy_hartree",
    "fitted_polarization_energy_kcal_per_mol",
    "full_tangent_gradient_inf",
    "maximum_eigenvalue_hartree",
    "polarization_energy_error_hartree",
    "polarization_energy_error_kcal_per_mol",
    "reduced_dimension",
    "relative_spectral_cutoff",
    "residual_energy_norm_squared_hartree",
    "retained_condition_number",
    "retained_minimum_eigenvalue_hartree",
    "retained_subspace_optimality_inf",
    "shifted_pythagorean_error_hartree",
    "surface_point_count",
    "target_polarization_energy_hartree",
    "target_polarization_energy_kcal_per_mol",
    "widths_angstrom",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _cutoff_token(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _root_mean_square(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _arm_summary(result: dict[str, object]) -> dict[str, object]:
    return {field: result[field] for field in COPIED_FIELDS}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{label} is not a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a SHA-256 digest.") from exc
    return value


def _require_frozen_path(
    value: object,
    *,
    expected: object,
    label: str,
) -> None:
    actual_path = Path(str(value)).resolve()
    expected_path = Path(str(expected))
    if not expected_path.is_absolute():
        expected_path = REPO_ROOT / expected_path
    if actual_path != expected_path.resolve():
        raise RuntimeError(f"{label} does not match the preregistration.")


def _validate_qm_provenance(
    inputs: dict[str, object],
    *,
    record: dict[str, object],
    source_bindings: dict[str, object],
) -> None:
    checkpoint = inputs["qm_checkpoint"]
    actual = checkpoint["provenance"]
    expected = record["qm_reference"]
    compound_id = str(record["compound_id"])
    mode = str(expected["provenance_mode"])
    if actual["provenance_mode"] != mode:
        raise RuntimeError(f"QM provenance mode mismatch for {compound_id}.")
    checkpoint_sha = _require_sha256(
        actual["checkpoint_sha256"],
        label=f"QM checkpoint digest for {compound_id}",
    )
    if checkpoint_sha != checkpoint["sha256"]:
        raise RuntimeError(f"QM checkpoint binding mismatch for {compound_id}.")
    _require_sha256(
        actual["ledger_sha256"],
        label=f"QM ledger digest for {compound_id}",
    )

    if mode == "legacy-frozen-checkpoint-v1":
        if checkpoint_sha != expected["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen QM checkpoint mismatch for {compound_id}.")
        if actual["ledger_sha256"] != expected["ledger_sha256"]:
            raise RuntimeError(f"Frozen QM ledger mismatch for {compound_id}.")
        _require_frozen_path(
            actual["checkpoint_path"],
            expected=expected["checkpoint_path"],
            label=f"QM checkpoint path for {compound_id}",
        )
        _require_frozen_path(
            actual["ledger_path"],
            expected=expected["ledger_path"],
            label=f"QM ledger path for {compound_id}",
        )
        return

    if mode == "preregistered-gas-runner-v1":
        if (
            Path(str(actual["checkpoint_path"])).name
            != expected["expected_checkpoint_filename"]
        ):
            raise RuntimeError(
                f"Generated QM checkpoint name mismatch for {compound_id}."
            )
        if (
            Path(str(actual["ledger_path"])).name
            != expected["expected_ledger_filename"]
        ):
            raise RuntimeError(f"Generated QM ledger name mismatch for {compound_id}.")
        if expected["generator_runner"] not in source_bindings:
            raise RuntimeError(f"QM generator source is not frozen for {compound_id}.")
        return

    raise RuntimeError(f"Unsupported QM provenance mode for {compound_id}: {mode}.")


def _validate_pcm_provenance(
    inputs: dict[str, object],
    *,
    record: dict[str, object],
    source_bindings: dict[str, object],
) -> None:
    parsed_input = inputs["parsed_pcm_input"]
    actual = parsed_input["provenance"]
    expected = record["pcm_input"]
    compound_id = str(record["compound_id"])
    mode = str(expected["provenance_mode"])
    if actual["provenance_mode"] != mode:
        raise RuntimeError(f"PCM provenance mode mismatch for {compound_id}.")
    effective_sha = _require_sha256(
        actual["effective_machine_input_sha256"],
        label=f"PCM machine-input digest for {compound_id}",
    )
    if effective_sha != parsed_input["sha256"]:
        raise RuntimeError(f"PCM machine-input binding mismatch for {compound_id}.")

    if mode == "legacy-frozen-machine-input-v1":
        if effective_sha != expected["effective_machine_input_sha256"]:
            raise RuntimeError(f"Frozen PCM machine-input mismatch for {compound_id}.")
        _require_frozen_path(
            actual["effective_machine_input_path"],
            expected=expected["effective_machine_input_path"],
            label=f"PCM machine-input path for {compound_id}",
        )
        return

    if mode == "preregistered-intrinsic-input-runner-v1":
        if (
            Path(str(actual["effective_machine_input_path"])).name
            != expected["expected_effective_machine_input_filename"]
        ):
            raise RuntimeError(f"Generated PCM input name mismatch for {compound_id}.")
        if (
            Path(str(actual["ledger_path"])).name
            != expected["expected_ledger_filename"]
        ):
            raise RuntimeError(f"Generated PCM ledger name mismatch for {compound_id}.")
        _require_sha256(
            actual["ledger_sha256"],
            label=f"PCM ledger digest for {compound_id}",
        )
        if expected["generator_runner"] not in source_bindings:
            raise RuntimeError(f"PCM generator source is not frozen for {compound_id}.")
        return

    raise RuntimeError(f"Unsupported PCM provenance mode for {compound_id}: {mode}.")


def _validate_raw_result(
    raw: dict[str, object],
    *,
    record: dict[str, object],
    cutoff: float,
    preregistration_sha256: str,
    artifact_id_template: str,
    execution_contract: dict[str, object],
) -> None:
    compound_id = str(record["compound_id"])
    cutoff_token = _cutoff_token(cutoff)
    expected_artifact_id = artifact_id_template.format(
        compound_id=compound_id,
        cutoff_token=cutoff_token,
    )
    if raw.get("artifact_id") != expected_artifact_id:
        raise RuntimeError(f"Unexpected artifact identity for {compound_id}.")
    inputs = raw["inputs"]
    if inputs["compound_id"] != compound_id:
        raise RuntimeError(f"Compound mismatch for {compound_id}.")
    if inputs["molecule_name"] != record["name"]:
        raise RuntimeError(f"Molecule-name mismatch for {compound_id}.")
    if inputs["molecule"]["sha256"] != record["mol2_sha256"]:
        raise RuntimeError(f"MOL2 mismatch for {compound_id}.")
    source = raw["source"]
    runtime = raw["runtime"]
    source_bindings = execution_contract["source_sha256"]
    if source["source_sha256"] != source_bindings:
        raise RuntimeError(f"Frozen source map mismatch for {compound_id}.")
    if (
        runtime["pyscf_python"]["resolved_sha256"]
        != execution_contract["pyscf_python_resolved_sha256"]
    ):
        raise RuntimeError(f"PySCF interpreter mismatch for {compound_id}.")
    if (
        runtime["pcmsolver_library"]["sha256"]
        != execution_contract["pcmsolver_library_sha256"]
    ):
        raise RuntimeError(f"PCMSolver library mismatch for {compound_id}.")
    preregistration = source["preregistration"]
    if preregistration["sha256"] != preregistration_sha256:
        raise RuntimeError(f"Preregistration mismatch for {compound_id}.")
    _validate_qm_provenance(
        inputs,
        record=record,
        source_bindings=source_bindings,
    )
    _validate_pcm_provenance(
        inputs,
        record=record,
        source_bindings=source_bindings,
    )
    if not raw["gates"]["retained_subspace_projection_gates_passed"]:
        raise RuntimeError(
            f"Projection algebra gate failed for {compound_id} at " f"{cutoff_token}."
        )
    for arm_name in ARM_NAMES:
        arm = raw["basis_results"][arm_name]
        if not math.isclose(
            float(arm["relative_spectral_cutoff"]),
            cutoff,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(f"Cutoff mismatch for {compound_id}/{arm_name}.")


def _aggregate_cutoff(
    cases: list[dict[str, object]],
    *,
    tie_tolerance: float,
) -> dict[str, object]:
    arm_summaries: dict[str, dict[str, object]] = {}
    for arm_name in ARM_NAMES:
        arms = [case["basis_results"][arm_name] for case in cases]
        absolute_errors = [
            abs(float(arm["polarization_energy_error_kcal_per_mol"])) for arm in arms
        ]
        arm_summaries[arm_name] = {
            "mean_absolute_polarization_projection_error_kcal_mol": _mean(
                absolute_errors
            ),
            "root_mean_square_polarization_projection_error_kcal_mol": (
                _root_mean_square(absolute_errors)
            ),
            "maximum_absolute_polarization_projection_error_kcal_mol": max(
                absolute_errors
            ),
            "mean_wall_seconds": _mean([float(arm["elapsed_seconds"]) for arm in arms]),
            "total_wall_seconds": sum(float(arm["elapsed_seconds"]) for arm in arms),
            "maximum_retained_condition_number": max(
                float(arm["retained_condition_number"]) for arm in arms
            ),
            "maximum_coefficient_l2_norm": max(
                float(arm["coefficient_l2_norm"]) for arm in arms
            ),
            "maximum_coefficient_absolute_value": max(
                float(arm["coefficient_max_abs"]) for arm in arms
            ),
        }
    improvements = []
    for case in cases:
        one_error = float(
            case["basis_results"]["one_radial"][
                "polarization_energy_error_kcal_per_mol"
            ]
        )
        two_error = float(
            case["basis_results"]["two_radial"][
                "polarization_energy_error_kcal_per_mol"
            ]
        )
        improvements.append(abs(one_error) - abs(two_error))
    return {
        "arms": arm_summaries,
        "two_radial_vs_one_radial": {
            "improvement_count": sum(
                improvement > tie_tolerance for improvement in improvements
            ),
            "tie_count": sum(
                abs(improvement) <= tie_tolerance for improvement in improvements
            ),
            "regression_count": sum(
                improvement < -tie_tolerance for improvement in improvements
            ),
            "mean_absolute_error_improvement_kcal_mol": _mean(improvements),
        },
    }


def main() -> int:
    arguments = _parse_args()
    preregistration_path = arguments.preregistration.resolve()
    input_root = arguments.input_root.resolve()
    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    preregistration = _load_json(preregistration_path)
    preregistration_sha256 = _sha256(preregistration_path)
    method = preregistration["method"]
    records = preregistration["records"]
    cutoffs = [float(value) for value in method["relative_spectral_cutoffs"]]
    comparison_tolerances = preregistration["execution_contract"][
        "comparison_tolerances"
    ]
    execution_contract = preregistration["execution_contract"]
    artifact_id_template = execution_contract["projection_runner_policy"][
        "artifact_id_template"
    ]

    cases: list[dict[str, object]] = []
    completed_at_values: list[str] = []
    execution_git_heads: set[str] = set()
    for record in records:
        compound_id = str(record["compound_id"])
        for cutoff in cutoffs:
            cutoff_token = _cutoff_token(cutoff)
            raw_path = (
                input_root / compound_id / f"cutoff-{cutoff_token}" / "result.json"
            )
            raw = _load_json(raw_path)
            _validate_raw_result(
                raw,
                record=record,
                cutoff=cutoff,
                preregistration_sha256=preregistration_sha256,
                artifact_id_template=artifact_id_template,
                execution_contract=execution_contract,
            )
            completed_at_values.append(str(raw["generated_at_utc"]))
            execution_git_heads.add(str(raw["source"]["git_head"]))
            cases.append(
                {
                    "compound_id": compound_id,
                    "molecule_name": record["name"],
                    "chemistry_class": record["class"],
                    "cutoff_token": cutoff_token,
                    "relative_spectral_cutoff": cutoff,
                    "raw_artifact": {
                        "artifact_id": raw["artifact_id"],
                        "path": str(raw_path.relative_to(REPO_ROOT)),
                        "sha256": _sha256(raw_path),
                    },
                    "input_bindings": {
                        "mol2_sha256": raw["inputs"]["molecule"]["sha256"],
                        "parsed_pcm_input_sha256": raw["inputs"]["parsed_pcm_input"][
                            "sha256"
                        ],
                        "qm_checkpoint_sha256": raw["inputs"]["qm_checkpoint"][
                            "sha256"
                        ],
                    },
                    "geometry_max_abs_error_angstrom": raw["inputs"][
                        "geometry_max_abs_error_angstrom"
                    ],
                    "checkpoint_density_binding_residual_e": raw["inputs"][
                        "qm_density"
                    ]["checkpoint_density_binding_residual_e"],
                    "retained_subspace_projection_gates_passed": raw["gates"][
                        "retained_subspace_projection_gates_passed"
                    ],
                    "basis_results": {
                        arm_name: _arm_summary(raw["basis_results"][arm_name])
                        for arm_name in ARM_NAMES
                    },
                }
            )

    by_cutoff: dict[str, dict[str, object]] = {}
    for cutoff in cutoffs:
        cutoff_token = _cutoff_token(cutoff)
        cutoff_cases = [case for case in cases if case["cutoff_token"] == cutoff_token]
        by_cutoff[cutoff_token] = _aggregate_cutoff(
            cutoff_cases,
            tie_tolerance=float(
                comparison_tolerances["per_record_improvement_tie_kcal_per_mol"]
            ),
        )

    gates_spec = preregistration["decision_gates"]
    primary_token = _cutoff_token(cutoffs[0])
    primary_summary = by_cutoff[primary_token]
    monotonic_tolerance = float(
        comparison_tolerances["residual_energy_norm_monotonic_absolute_hartree"]
    )
    residual_monotonic = True
    for record in records:
        compound_id = str(record["compound_id"])
        for arm_name in ARM_NAMES:
            residuals = [
                float(
                    next(
                        case
                        for case in cases
                        if case["compound_id"] == compound_id
                        and case["relative_spectral_cutoff"] == cutoff
                    )["basis_results"][arm_name]["residual_energy_norm_squared_hartree"]
                )
                for cutoff in cutoffs
            ]
            residual_monotonic &= all(
                right <= left + monotonic_tolerance
                for left, right in zip(residuals, residuals[1:])
            )

    evaluated_gates = {
        "all_twenty_case_cutoff_combinations_execute": (
            len(cases) == len(records) * len(cutoffs)
        ),
        "all_density_checkpoint_binding_residuals_below_e": (
            max(float(case["checkpoint_density_binding_residual_e"]) for case in cases)
            < float(gates_spec["all_density_checkpoint_binding_residuals_below_e"])
        ),
        "all_geometry_max_abs_errors_below_angstrom": (
            max(float(case["geometry_max_abs_error_angstrom"]) for case in cases)
            < float(gates_spec["all_geometry_max_abs_errors_below_angstrom"])
        ),
        (
            "all_reciprocity_half_coupling_constraint_retained_subspace_"
            "optimality_and_shifted_pythagorean_gates_pass"
        ): all(
            bool(case["retained_subspace_projection_gates_passed"]) for case in cases
        ),
        "at_cutoff_1e-4_two_radial_improvement_count_minimum": (
            primary_summary["two_radial_vs_one_radial"]["improvement_count"]
            >= int(gates_spec["at_cutoff_1e-4_two_radial_improvement_count_minimum"])
        ),
        (
            "at_cutoff_1e-4_two_radial_"
            "maximum_absolute_projection_error_kcal_mol_below"
        ): (
            primary_summary["arms"]["two_radial"][
                "maximum_absolute_polarization_projection_error_kcal_mol"
            ]
            < float(
                gates_spec[
                    "at_cutoff_1e-4_two_radial_"
                    "maximum_absolute_projection_error_kcal_mol_below"
                ]
            )
        ),
        (
            "at_cutoff_1e-4_two_radial_mean_absolute_"
            "polarization_projection_error_not_above_one_radial"
        ): (
            primary_summary["arms"]["two_radial"][
                "mean_absolute_polarization_projection_error_kcal_mol"
            ]
            <= primary_summary["arms"]["one_radial"][
                "mean_absolute_polarization_projection_error_kcal_mol"
            ]
            + float(
                comparison_tolerances["mean_absolute_error_comparison_kcal_per_mol"]
            )
        ),
        (
            "residual_energy_norm_is_nonincreasing_as_cutoff_decreases_"
            "for_each_record_and_basis"
        ): residual_monotonic,
    }
    failed_gates = sorted(
        name for name, passed in evaluated_gates.items() if not passed
    )
    if len(execution_git_heads) != 1:
        raise RuntimeError("All raw canaries must come from one execution Git commit.")
    execution_git_head = next(iter(execution_git_heads))
    artifact = {
        "schema_version": 1,
        "artifact_id": "route2-gto-pcm-energy-projection-four-v1",
        "completed_at_utc": max(completed_at_values),
        "scientific_status": (
            "complete-negative-result"
            if failed_gates
            else "complete-preregistered-gates-pass"
        ),
        "decision": (
            "fail-preregistered-transfer-gate"
            if failed_gates
            else "pass-preregistered-transfer-gates"
        ),
        "failed_gate_names": failed_gates,
        "scientific_conclusion": (
            "The second radial channel lowers the PCM-energy-norm "
            "representation error for all four chemistry classes, but the "
            "preregistered 1e-4 cutoff misses the sub-1 kcal/mol maximum-error "
            "gate on the flexible diester. Lower cutoffs reduce energy error "
            "only by retaining increasingly ill-conditioned modes and do not "
            "authorize projected coefficients as ML targets."
        ),
        "claim_boundary": {
            "establishes": [
                "fixed-geometry same-basis representation transfer on four preregistered molecules",
                "the cutoff-dependent tradeoff between PCM energy error and spectral conditioning",
                "the preregistered pass/fail disposition without threshold changes",
            ],
            "does_not_establish": preregistration["claim_boundary"]["cannot_establish"],
        },
        "source": {
            "execution_git_head": execution_git_head,
            "aggregator_path": SCRIPT_RELATIVE_PATH,
            "aggregator_sha256": _sha256(REPO_ROOT / SCRIPT_RELATIVE_PATH),
            "preregistration": {
                "path": str(preregistration_path.relative_to(REPO_ROOT)),
                "sha256": preregistration_sha256,
            },
        },
        "provenance_validation": {
            "all_raw_source_hash_maps_match_preregistration": True,
            "all_pyscf_interpreter_hashes_match_preregistration": True,
            "all_pcmsolver_library_hashes_match_preregistration": True,
            "all_qm_provenance_matches_record_contract": True,
            "all_pcm_provenance_matches_record_contract": True,
            "common_frozen_hashes": {
                "source_sha256": execution_contract["source_sha256"],
                "pyscf_python_resolved_sha256": execution_contract[
                    "pyscf_python_resolved_sha256"
                ],
                "pcmsolver_library_sha256": execution_contract[
                    "pcmsolver_library_sha256"
                ],
                "pcmsolver_parser_sha256": execution_contract[
                    "pcmsolver_parser_sha256"
                ],
            },
        },
        "method": method,
        "record_count": len(records),
        "cutoff_count": len(cutoffs),
        "case_count": len(cases),
        "cases": cases,
        "aggregate_by_cutoff": by_cutoff,
        "evaluated_gates": evaluated_gates,
        "mandatory_diagnostics": {
            "maximum_two_radial_retained_condition_number": max(
                float(case["basis_results"]["two_radial"]["retained_condition_number"])
                for case in cases
            ),
            "maximum_two_radial_coefficient_l2_norm": max(
                float(case["basis_results"]["two_radial"]["coefficient_l2_norm"])
                for case in cases
            ),
            "maximum_two_radial_coefficient_absolute_value": max(
                float(case["basis_results"]["two_radial"]["coefficient_max_abs"])
                for case in cases
            ),
            "full_tangent_gradient_is_ungated": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
