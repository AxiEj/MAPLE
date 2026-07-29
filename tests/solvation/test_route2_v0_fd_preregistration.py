from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREREG = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-fd-multisolvent-prereg-v1.json"
)


def test_v0_fixed_density_multisolvent_protocol_locks_the_no_fit_boundary():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == "route2-v0-fd-multisolvent-prereg-v1"
    assert protocol["status"] == "preregistered-before-physics-execution"
    construction = protocol["construction"]
    assert construction["name"] == "route2-v0-fixed-density-v1"
    assert "argmin_sigma" in construction["continuum_state"]
    assert "0.5" in construction["electrostatic_energy"]
    assert construction["solute_response"] == (
        "forbidden: c0 is not updated from a solvent reaction field"
    )

    constraints = protocol["hard_constraints"]
    assert constraints == {
        "official_checkpoint_unmodified": True,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "density_projection": False,
        "response_tempering_or_eigenvalue_clipping": False,
        "per_molecule_or_per_solvent_method_selection_from_error": False,
        "legacy_public_route_changed": False,
    }
    prohibited = " ".join(protocol["prohibited_actions"])
    assert "symmetrizing a response Jacobian" in prohibited
    assert "best of several methods per record" in prohibited


def test_v0_fixed_density_protocol_requires_more_than_ten_solvents_and_max_gates():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))
    solvent_contract = protocol["solvent_information_contract"]

    assert solvent_contract["minimum_registered_solvent_count"] == 11
    assert len(solvent_contract["registered_default_solvents"]) == 11
    assert "methanol" in solvent_contract["registered_default_solvents"]
    assert "fail closed" in solvent_contract["dielectric_only_custom_solvent_policy"]
    requirements = solvent_contract["total_free_energy_custom_solvent_requirements"]
    assert any("bulk number density" in item for item in requirements)
    assert any("dispersion" in item for item in requirements)

    coverage = protocol["benchmark_coverage_contract"]
    assert coverage["runtime_supported_solvent_count"] == 11
    assert coverage["mnsol_v2012_neutral_absolute_solvent_count"] == 10
    assert "methanol" not in coverage["mnsol_v2012_neutral_absolute_solvents"]
    assert "no neutral absolute methanol rows" in coverage["mnsol_methanol_limitation"]
    assert coverage["methanol_neutral_absolute_extension"].startswith(
        "required-before-11-solvent-chemistry-scoring"
    )
    assert "No statement" in coverage["accuracy_claim_rule"]

    sequence = " ".join(protocol["validation_sequence"])
    assert "every development record" in sequence
    assert "strictly below 1.5 kcal/mol" in sequence
    assert "strictly below 1.0 kcal/mol" in sequence
    assert "external final blind dataset" in sequence
