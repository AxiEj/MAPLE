"""Current-source bounded FP32 evidence is separate from historical v3."""

import json
from pathlib import Path

import numpy as np

from tools.route2_release.run_pure_mace_polar_fp32_smoke import source_manifest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ID = "route2-pure-macepolar-torch-fp32-v4-engineering-tests"


def test_current_fp32_evidence_has_all_four_cases_and_no_scientific_admission():
    record = json.loads(
        (ROOT / "docs/route2/evidence" / f"{EVIDENCE_ID}.json").read_text()
    )
    assert record["artifact_id"] == EVIDENCE_ID
    assert record["source_manifest"] == source_manifest()
    assert record["source_unchanged"] is True
    assert record["scientific_release_admitted"] is False
    assert record["physical_force_accuracy_admitted"] is False
    assert record["experimental_solvation_accuracy_admitted"] is False
    assert record["dataset_labels_used"] is False
    assert {(c["solvent"], c["device"]) for c in record["cases"]} == {
        (solvent, device)
        for solvent in ("water", "hexane")
        for device in ("cpu", "cuda:0")
    }
    assert len(record["cases"]) == 4
    for case in record["cases"]:
        assert case["pass"] is True
        assert case["computation_dtype"] == "float32"
        assert case["storage_dtype"] == "float64"
        diagnostics = case["diagnostics"]
        receipt = diagnostics["ddpcm"]["execution_receipt"]
        assert receipt["dtype"] == "torch.float32"
        assert receipt["lmax"] == 5
        assert receipt["n_lebedev"] == 194
        assert receipt["solve_residual_tolerance"] == 100 * np.finfo(np.float32).eps
        assert (
            case["hvp_dense_defect_eV_per_A2"]
            < record["hvp_dense_max_budget_eV_per_A2"]
        )
        assert (
            diagnostics["raw_hessian_antisymmetry_eV_per_A2"]
            <= diagnostics["hessian_antisymmetry_budget_eV_per_A2"]
        )
        assert (
            diagnostics["model_provenance"][
                "upstream_explicit_fp64_reductions_preserved"
            ]
            is True
        )
        assert "/home/" not in json.dumps(diagnostics["model_provenance"])
