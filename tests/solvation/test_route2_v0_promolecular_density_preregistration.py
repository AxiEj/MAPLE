from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-v0-promolecular-atomic-hf-def2-tzvpd-prereg-v1.json"
)
RUNNER = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "generate_route2_v0_promolecular_density.py"
)


def test_promolecular_density_preregistration_locks_the_independent_source():
    protocol = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == (
        "route2-v0-promolecular-atomic-hf-def2-tzvpd-v1-prereg"
    )
    assert protocol["status"] == "frozen-before-asset-generation"
    contract = protocol["generation_contract"]
    assert contract["method"] == "PySCF AtomSphAverageRHF"
    assert contract["pyscf_version"] == "2.13.1"
    assert contract["basis"] == "def2-TZVPD"
    assert contract["radial_grid"] == {
        "coordinate_rule": "r_max * (i / (n - 1))**2",
        "point_count": 2049,
        "radius_max_bohr": 30.0,
    }

    elements = contract["elements"]
    assert [(entry["symbol"], entry["atomic_number"]) for entry in elements] == [
        ("H", 1),
        ("C", 6),
        ("N", 7),
        ("O", 8),
        ("S", 16),
        ("Cl", 17),
    ]
    assert [entry["neutral_electron_count"] for entry in elements] == [
        1,
        6,
        7,
        8,
        16,
        17,
    ]


def test_promolecular_density_preregistration_forbids_model_or_label_substitution():
    protocol = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))

    assert protocol["hard_constraints"] == {
        "mace_density_relabelled_as_electron_density": False,
        "mace_density_absolute_value_or_clipping": False,
        "post_training": False,
        "fine_tuning": False,
        "experimental_solvation_fit": False,
        "map_or_uq_calibration": False,
        "target_record_or_error_selected_parameter": False,
        "public_route_changed": False,
    }
    declared_use = protocol["declared_use"]
    assert "MACE Gaussian source" in declared_use["electrostatic_channel"]
    assert "positive promolecular density" in declared_use["short_range_channel"]
    assert "not a total solvent model" in declared_use["scope_limit"]
    source_hashes = protocol["source_sha256"]
    assert source_hashes == {
        "docs/implicit-solvation/benchmarks/"
        "generate_route2_v0_promolecular_density.py": source_hashes[
            "docs/implicit-solvation/benchmarks/"
            "generate_route2_v0_promolecular_density.py"
        ]
    }
    assert len(source_hashes[next(iter(source_hashes))]) == 64
    assert RUNNER.exists()
