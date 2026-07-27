from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-v1.json"
)
PREREGISTRATION = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
AGGREGATOR = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "aggregate_route2_gto_pcm_energy_projection_four.py"
)
ARTIFACT_SHA256 = "9969c3a9ba07ad349d0ef460f84a84e9b03e5d2b7021a8a7e9bf43c86f2480e8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def aggregator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aggregate_route2_gto_pcm_energy_projection_four",
        AGGREGATOR,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_raw(
    record: dict[str, object],
    execution_contract: dict[str, object],
    *,
    preregistration_sha256: str,
) -> dict[str, object]:
    qm_spec = record["qm_reference"]
    qm_mode = qm_spec["provenance_mode"]
    if qm_mode == "legacy-frozen-checkpoint-v1":
        qm_provenance = {
            "provenance_mode": qm_mode,
            "checkpoint_path": str(ROOT / qm_spec["checkpoint_path"]),
            "checkpoint_sha256": qm_spec["checkpoint_sha256"],
            "ledger_path": str(ROOT / qm_spec["ledger_path"]),
            "ledger_sha256": qm_spec["ledger_sha256"],
        }
    else:
        qm_provenance = {
            "provenance_mode": qm_mode,
            "checkpoint_path": (
                f"/synthetic/{qm_spec['expected_checkpoint_filename']}"
            ),
            "checkpoint_sha256": "a" * 64,
            "ledger_path": f"/synthetic/{qm_spec['expected_ledger_filename']}",
            "ledger_sha256": "b" * 64,
        }

    pcm_spec = record["pcm_input"]
    pcm_mode = pcm_spec["provenance_mode"]
    if pcm_mode == "legacy-frozen-machine-input-v1":
        pcm_provenance = {
            "provenance_mode": pcm_mode,
            "effective_machine_input_path": str(
                ROOT / pcm_spec["effective_machine_input_path"]
            ),
            "effective_machine_input_sha256": (
                pcm_spec["effective_machine_input_sha256"]
            ),
        }
    else:
        pcm_provenance = {
            "provenance_mode": pcm_mode,
            "effective_machine_input_path": (
                f"/synthetic/{pcm_spec['expected_effective_machine_input_filename']}"
            ),
            "effective_machine_input_sha256": "c" * 64,
            "ledger_path": f"/synthetic/{pcm_spec['expected_ledger_filename']}",
            "ledger_sha256": "d" * 64,
        }

    return {
        "artifact_id": (
            "route2-gto-pcm-energy-projection-"
            f"{record['compound_id']}-cutoff-1e-4-v1"
        ),
        "source": {
            "source_sha256": copy.deepcopy(execution_contract["source_sha256"]),
            "preregistration": {"sha256": preregistration_sha256},
        },
        "runtime": {
            "pyscf_python": {
                "resolved_sha256": execution_contract["pyscf_python_resolved_sha256"]
            },
            "pcmsolver_library": {
                "sha256": execution_contract["pcmsolver_library_sha256"]
            },
        },
        "inputs": {
            "compound_id": record["compound_id"],
            "molecule_name": record["name"],
            "molecule": {"sha256": record["mol2_sha256"]},
            "qm_checkpoint": {
                "sha256": qm_provenance["checkpoint_sha256"],
                "provenance": qm_provenance,
            },
            "parsed_pcm_input": {
                "sha256": pcm_provenance["effective_machine_input_sha256"],
                "provenance": pcm_provenance,
            },
        },
        "gates": {"retained_subspace_projection_gates_passed": True},
        "basis_results": {
            arm_name: {"relative_spectral_cutoff": 1.0e-4}
            for arm_name in ("one_radial", "two_radial")
        },
    }


def test_four_molecule_projection_artifact_is_locked_and_negative():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    assert artifact["schema_version"] == 1
    assert artifact["artifact_id"] == ("route2-gto-pcm-energy-projection-four-v1")
    assert artifact["scientific_status"] == "complete-negative-result"
    assert artifact["decision"] == "fail-preregistered-transfer-gate"
    assert artifact["failed_gate_names"] == [
        "at_cutoff_1e-4_two_radial_" "maximum_absolute_projection_error_kcal_mol_below"
    ]
    assert artifact["record_count"] == 4
    assert artifact["cutoff_count"] == 5
    assert artifact["case_count"] == 20


def test_four_molecule_projection_artifact_binds_sources_and_raw_cases():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["source"]["aggregator_sha256"] == _sha256(AGGREGATOR)
    assert artifact["source"]["preregistration"]["sha256"] == _sha256(PREREGISTRATION)
    assert artifact["source"]["execution_git_head"] == (
        "a3cab00d89e78ca0fb9b734fcbaaf4c64c0d0438"
    )
    cases = artifact["cases"]
    assert (
        len({(case["compound_id"], case["relative_spectral_cutoff"]) for case in cases})
        == 20
    )
    assert all(len(case["raw_artifact"]["sha256"]) == 64 for case in cases)
    assert all(case["retained_subspace_projection_gates_passed"] for case in cases)
    provenance = artifact["provenance_validation"]
    execution_contract = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))[
        "execution_contract"
    ]
    assert provenance == {
        "all_pcm_provenance_matches_record_contract": True,
        "all_pcmsolver_library_hashes_match_preregistration": True,
        "all_pyscf_interpreter_hashes_match_preregistration": True,
        "all_qm_provenance_matches_record_contract": True,
        "all_raw_source_hash_maps_match_preregistration": True,
        "common_frozen_hashes": {
            "source_sha256": execution_contract["source_sha256"],
            "pyscf_python_resolved_sha256": execution_contract[
                "pyscf_python_resolved_sha256"
            ],
            "pcmsolver_library_sha256": execution_contract["pcmsolver_library_sha256"],
            "pcmsolver_parser_sha256": execution_contract["pcmsolver_parser_sha256"],
        },
    }


def test_four_molecule_projection_aggregate_records_accuracy_stability_tradeoff():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    summaries = artifact["aggregate_by_cutoff"]

    primary = summaries["1e-4"]
    assert primary["two_radial_vs_one_radial"]["improvement_count"] == 4
    assert primary["arms"]["one_radial"][
        "mean_absolute_polarization_projection_error_kcal_mol"
    ] == pytest.approx(1.2364961326922692)
    assert primary["arms"]["two_radial"][
        "mean_absolute_polarization_projection_error_kcal_mol"
    ] == pytest.approx(0.7535668337330764)
    assert primary["arms"]["two_radial"][
        "maximum_absolute_polarization_projection_error_kcal_mol"
    ] == pytest.approx(1.5371996446955856)

    less_regularized = summaries["1e-12"]["arms"]["two_radial"]
    assert less_regularized[
        "mean_absolute_polarization_projection_error_kcal_mol"
    ] == pytest.approx(0.08552664167618854)
    assert less_regularized["maximum_retained_condition_number"] > 9.0e11
    assert less_regularized["maximum_coefficient_l2_norm"] > 1.0e4
    assert artifact["mandatory_diagnostics"][
        "maximum_two_radial_coefficient_absolute_value"
    ] == pytest.approx(9834.64659296467)


def test_four_molecule_projection_gates_are_recomputed_not_relabelled():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    gates = artifact["evaluated_gates"]

    assert gates == {
        "all_density_checkpoint_binding_residuals_below_e": True,
        "all_geometry_max_abs_errors_below_angstrom": True,
        (
            "all_reciprocity_half_coupling_constraint_retained_subspace_"
            "optimality_and_shifted_pythagorean_gates_pass"
        ): True,
        "all_twenty_case_cutoff_combinations_execute": True,
        (
            "at_cutoff_1e-4_two_radial_"
            "maximum_absolute_projection_error_kcal_mol_below"
        ): False,
        "at_cutoff_1e-4_two_radial_improvement_count_minimum": True,
        (
            "at_cutoff_1e-4_two_radial_mean_absolute_"
            "polarization_projection_error_not_above_one_radial"
        ): True,
        (
            "residual_energy_norm_is_nonincreasing_as_cutoff_decreases_"
            "for_each_record_and_basis"
        ): True,
    }
    assert (
        "hydration free-energy or experimental accuracy"
        in artifact["claim_boundary"]["does_not_establish"]
    )


def test_aggregate_uses_absolute_error_for_signed_projection_errors(
    aggregator_module: ModuleType,
):
    cases = [
        {
            "basis_results": {
                "one_radial": {
                    "polarization_energy_error_kcal_per_mol": -2.0,
                    "elapsed_seconds": 1.0,
                    "retained_condition_number": 2.0,
                    "coefficient_l2_norm": 3.0,
                    "coefficient_max_abs": 4.0,
                },
                "two_radial": {
                    "polarization_energy_error_kcal_per_mol": -1.0,
                    "elapsed_seconds": 1.0,
                    "retained_condition_number": 2.0,
                    "coefficient_l2_norm": 3.0,
                    "coefficient_max_abs": 4.0,
                },
            }
        }
    ]

    summary = aggregator_module._aggregate_cutoff(cases, tie_tolerance=1.0e-12)

    assert summary["arms"]["one_radial"][
        "mean_absolute_polarization_projection_error_kcal_mol"
    ] == pytest.approx(2.0)
    assert summary["arms"]["two_radial"][
        "maximum_absolute_polarization_projection_error_kcal_mol"
    ] == pytest.approx(1.0)
    assert summary["two_radial_vs_one_radial"] == {
        "improvement_count": 1,
        "tie_count": 0,
        "regression_count": 0,
        "mean_absolute_error_improvement_kcal_mol": pytest.approx(1.0),
    }


def test_raw_projection_validation_fails_closed_on_provenance_tampering(
    aggregator_module: ModuleType,
):
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    execution_contract = preregistration["execution_contract"]
    preregistration_sha256 = _sha256(PREREGISTRATION)
    artifact_id_template = execution_contract["projection_runner_policy"][
        "artifact_id_template"
    ]

    for record in preregistration["records"]:
        raw = _synthetic_raw(
            record,
            execution_contract,
            preregistration_sha256=preregistration_sha256,
        )
        aggregator_module._validate_raw_result(
            raw,
            record=record,
            cutoff=1.0e-4,
            preregistration_sha256=preregistration_sha256,
            artifact_id_template=artifact_id_template,
            execution_contract=execution_contract,
        )

    acetone = preregistration["records"][1]
    raw = _synthetic_raw(
        acetone,
        execution_contract,
        preregistration_sha256=preregistration_sha256,
    )
    tamperings = (
        ("source map", ("source", "source_sha256"), {"tampered": "0" * 64}),
        (
            "PySCF interpreter",
            ("runtime", "pyscf_python", "resolved_sha256"),
            "0" * 64,
        ),
        (
            "PCMSolver library",
            ("runtime", "pcmsolver_library", "sha256"),
            "0" * 64,
        ),
        (
            "QM checkpoint",
            ("inputs", "qm_checkpoint", "provenance", "checkpoint_sha256"),
            "0" * 64,
        ),
        (
            "PCM input",
            (
                "inputs",
                "parsed_pcm_input",
                "provenance",
                "effective_machine_input_sha256",
            ),
            "0" * 64,
        ),
    )
    for _, path, replacement in tamperings:
        tampered = copy.deepcopy(raw)
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(RuntimeError, match="mismatch"):
            aggregator_module._validate_raw_result(
                tampered,
                record=acetone,
                cutoff=1.0e-4,
                preregistration_sha256=preregistration_sha256,
                artifact_id_template=artifact_id_template,
                execution_contract=execution_contract,
            )

    generated_path_tamperings = (
        (
            preregistration["records"][0],
            ("inputs", "qm_checkpoint", "provenance", "ledger_path"),
        ),
        (
            preregistration["records"][3],
            ("inputs", "parsed_pcm_input", "provenance", "ledger_path"),
        ),
    )
    for record, path in generated_path_tamperings:
        tampered = _synthetic_raw(
            record,
            execution_contract,
            preregistration_sha256=preregistration_sha256,
        )
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "/synthetic/wrong-ledger-name.json"
        with pytest.raises(RuntimeError, match="name mismatch"):
            aggregator_module._validate_raw_result(
                tampered,
                record=record,
                cutoff=1.0e-4,
                preregistration_sha256=preregistration_sha256,
                artifact_id_template=artifact_id_template,
                execution_contract=execution_contract,
            )
