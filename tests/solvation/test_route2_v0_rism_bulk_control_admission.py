from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-v0-rism1d-cspce-bulk-control-v1.json"
)


def test_rism_bulk_control_is_bulk_only_and_cannot_claim_a_route2_solvent_result():
    control = json.loads(CONTROL.read_text(encoding="utf-8"))

    assert control["protocol_id"] == "route2-v0-rism1d-cspce-bulk-control-v1"
    assert control["status"] == (
        "bulk-only-parser-control-not-a-physical-route2-solvent-asset"
    )
    assert "not a MACE-coupled 3D-RISM calculation" in control["claim_boundary"]
    assert "not a solvation-energy or accuracy result" in control["claim_boundary"]
    assert "no solute PDB, prmtop" in control["generator"]["input_mode"]
    assert control["generator"]["parameters"]["theory"] == "DRISM"
    assert control["generator"]["parameters"]["closure"] == "PSE3"
    assert control["generator"]["parameters"]["observed_final_residual"] < 1.0e-12
    assert control["parsed_bulk_state"]["site_names"] == ["O", "H1"]
    assert control["parsed_bulk_state"]["site_multiplicity"] == [1, 2]
    assert control["parsed_bulk_state"]["raw_cvv_shape"] == [2, 2, 16384]
    assert (
        control["parsed_bulk_state"][
            "observed_max_coulomb_tail_residual_at_r_ge_400_angstrom"
        ]
        < 1.0e-8
    )
    assert "excluded from c_sr" in control["mathematical_boundary"]["origin_policy"]
    assert (
        "cannot enter the finite Cartesian HNC FFT"
        in control["mathematical_boundary"]["cartesian_policy"]
    )
    assert control["hard_constraints"] == {
        "target_solvation_labels_read_before_control": False,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "solute_gaff_or_am1bcc_substitution": False,
        "raw_cvv_fft_interpolation": False,
        "coulomb_origin_imputation": False,
    }
    assert "eleven-solvent physical asset coverage" in control["not_claimed"]
    assert (
        "solvation-energy, MAE, or all-record accuracy result" in control["not_claimed"]
    )
