from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "docs"
    / "route2"
    / "preregistrations"
    / "vqm24-observable-training-batch-v1.json"
)
DENSE_PREREGISTRATION = (
    ROOT
    / "docs"
    / "route2"
    / "preregistrations"
    / "vqm24-observable-training-batch-dense-v2.json"
)
PILOT = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "vqm24-observable-training-pilot-fluorine-20260827"
)
GAS_EVIDENCE = (
    ROOT
    / "docs"
    / "route2"
    / "evidence"
    / "vqm24-observable-training-gas-20260827"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vqm24_observable_training_batch_is_frozen_and_input_bound() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())

    assert preregistration["status"] == "locked-before-observable-training-qm-execution"
    assert preregistration["selection"]["split"] == "train"
    assert preregistration["selection"]["record_count"] == 32
    assert len(preregistration["records"]) == 32
    assert Counter(
        record["selection_record"]["stratum"]
        for record in preregistration["records"]
    ) == {
        "base-hcno": 12,
        "bromine": 4,
        "chlorine": 4,
        "fluorine": 4,
        "phosphorus": 4,
        "sulfur": 4,
    }
    assert len(
        {
            record["selection_record"]["chemical_formula_hill"]
            for record in preregistration["records"]
        }
    ) == 32
    for record in preregistration["records"]:
        assert record["selection_record"]["split"] == "train"
        for input_record in record["inputs"].values():
            assert _sha256(Path(input_record["path"])) == input_record["sha256"]
    for source_name in ("creator", "surface_builder", "mode_selector"):
        source = preregistration["source"]
        path = ROOT / source[f"{source_name}_path"]
        assert _sha256(path) == source[f"{source_name}_sha256"]
    assert _sha256(
        ROOT / preregistration["source"]["observable_runner"]["path"]
    ) == preregistration["source"]["observable_runner"]["sha256"]
    assert _sha256(
        ROOT / preregistration["source"]["shared_finite_field_runner"]["path"]
    ) == preregistration["source"]["shared_finite_field_runner"]["sha256"]


def test_vqm24_observable_training_target_contract_excludes_proxy_labels() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text())

    assert preregistration["probe_protocol"] == {
        "atomic_radius": "ASE standard van der Waals radii",
        "candidate_directions": 26,
        "cavity_or_solvent_used": False,
        "clearance_angstrom": 1.0,
        "field_step_e": 0.001,
        "mode_count": 4,
        "signs": [-1, 1],
    }
    assert preregistration["target_contract"] == {
        "density_or_partition_coefficient": False,
        "experimental_solvation_quantity": False,
        "external_nuclear_coupling_included": True,
        "numerical_energy_curvature": False,
        "pcm_or_cavity_quantity": False,
        "perturbed_full_molecular_mep": True,
        "perturbed_molecular_dipole": True,
        "perturbed_total_external_enthalpy": True,
        "vqm24_energy_or_atomization_quantity": False,
        "zero_field_energy": True,
        "zero_field_full_molecular_mep": True,
        "zero_field_molecular_dipole": True,
    }
    assert preregistration["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "model_accuracy_measured": False,
        "pcm_or_cavity_label_used": False,
        "vqm24_energy_target_used": False,
    }


def test_vqm24_dense_v2_supersedes_sparse_mep_before_observable_execution() -> None:
    dense = json.loads(DENSE_PREREGISTRATION.read_text())
    parent = json.loads(PREREGISTRATION.read_text())

    assert dense["status"] == "locked-before-dense-observable-qm-execution"
    assert dense["parent"] == {
        "disposition": (
            "gas execution retained; sparse observable surface superseded "
            "before any observable record execution"
        ),
        "file_sha256": _sha256(PREREGISTRATION),
        "path": "docs/route2/preregistrations/vqm24-observable-training-batch-v1.json",
        "preregistration_sha256": parent["preregistration_sha256"],
    }
    assert len(dense["records"]) == 32
    assert dense["target_contract"]["dense_mep_fit_partition"] is True
    assert dense["target_contract"]["dense_mep_audit_partition"] is True
    assert dense["target_contract"]["sparse_26_direction_mep_target"] is False
    for source_name in ("creator", "dense_builder", "observable_runner"):
        assert _sha256(ROOT / dense["source"][f"{source_name}_path"]) == dense[
            "source"
        ][f"{source_name}_sha256"]
    for record in dense["records"]:
        atom_count = len(record["selection_record"]["atomic_numbers"])
        surface_record = record["inputs"]["surface"]
        modes_record = record["inputs"]["modes"]
        surface_path = Path(surface_record["path"])
        modes_path = Path(modes_record["path"])
        assert _sha256(surface_path) == surface_record["sha256"]
        assert _sha256(modes_path) == modes_record["sha256"]
        assert surface_record["fit_point_count"] > 8 * atom_count
        assert surface_record["audit_point_count"] > 8 * atom_count
        with np.load(surface_path, allow_pickle=False) as surface, np.load(
            modes_path, allow_pickle=False
        ) as modes:
            partitions = surface["partition_indices"]
            source_indices = modes["source_surface_indices"]
            assert set(partitions.tolist()) == {0, 1}
            assert np.all(partitions[source_indices] == 0)
            np.testing.assert_allclose(
                modes["source_points_bohr"],
                surface["surface_points_bohr"][source_indices],
                rtol=0.0,
                atol=0.0,
            )


def test_vqm24_observable_training_pilot_replays_gate_a_and_conjugacy() -> None:
    result = json.loads((PILOT / "result.json").read_text())
    source = result["source"]

    assert result["status"] == "pass-observable-record-pilot"
    assert source["sealer_sha256"] == _sha256(ROOT / source["sealer_path"])
    assert source["runner_sha256"] == _sha256(ROOT / source["runner_path"])
    assert all(result["gates"].values())
    assert result["metrics"]["mep_large_step_replay_relative"] == pytest.approx(
        2.0571410557733556e-09, abs=0.0
    )
    assert result["metrics"]["dipole_large_step_replay_relative"] == pytest.approx(
        5.798142352905037e-10, abs=0.0
    )
    assert result["metrics"][
        "electronic_energy_large_step_replay_abs_hartree"
    ] == pytest.approx(5.684341886080801e-13, abs=0.0)
    assert result["metrics"][
        "enthalpy_slope_vs_zero_source_mep_relative"
    ] == pytest.approx(1.912931778751673e-07, abs=0.0)
    assert result["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "model_accuracy_measured": False,
        "observable_record_runner_validated_on_opened_pilot": True,
        "pcm_or_cavity_used": False,
        "vqm24_energy_target_used": False,
    }


def test_vqm24_observable_training_gas_stage_is_complete_but_not_a_response() -> None:
    result = json.loads((GAS_EVIDENCE / "result.json").read_text())
    source = result["source"]

    assert result["status"] == "pass-observable-training-gas"
    assert source["sealer_sha256"] == _sha256(ROOT / source["sealer_path"])
    assert source["preregistration_file_sha256"] == _sha256(
        ROOT / source["preregistration_path"]
    )
    assert result["aggregate"] == {
        "maximum_density_binding_residual_inf": pytest.approx(
            2.220446049250313e-16, abs=0.0
        ),
        "maximum_scf_cycles": 20,
        "minimum_scf_cycles": 15,
        "record_count": 32,
        "total_elapsed_seconds": pytest.approx(9245.931632104002, abs=0.0),
    }
    assert len(result["records"]) == 32
    assert all(result["gates"].values())
    assert result["claim_boundary"] == {
        "capability_admitted": False,
        "experimental_solvation_target_read": False,
        "fit_or_training_performed": False,
        "independent_qm_gas_checkpoints_generated": True,
        "model_accuracy_measured": False,
        "observable_response_generated": False,
        "pcm_or_cavity_used": False,
        "vqm24_energy_target_used": False,
    }
