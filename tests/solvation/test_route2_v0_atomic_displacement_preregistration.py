from __future__ import annotations

import hashlib
import json
from pathlib import Path

from maple.function.calculator.extra_correction.implicit.route2_v0_atomic_displacement_response import (
    V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
PREREG = BENCHMARKS / "route2-v0-atomic-displacement-hf-def2-tzvpd-prereg-v1.json"
GENERATOR = BENCHMARKS / "generate_route2_v0_atomic_displacement_response.py"
SOURCE = (
    ROOT / "maple/function/calculator/extra_correction/implicit/"
    "route2_v0_atomic_displacement_response.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_displacement_preregistration_freezes_the_physical_source_contract():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert (
        protocol["protocol_id"] == V0_ATOMIC_DISPLACEMENT_RESPONSE_ARTIFACT + "-prereg"
    )
    assert protocol["status"] == "frozen-before-asset-generation"
    assert protocol["generation_contract"]["method"] == "PySCF AtomSphAverageRHF"
    assert protocol["generation_contract"]["basis"] == "def2-TZVPD"
    assert protocol["generation_contract"]["response_construction"].startswith(
        "For each neutral spherical atomic electron density"
    )
    assert protocol["generation_contract"]["elements"] == [
        {
            "atomic_number": 1,
            "neutral_electron_count": 1,
            "spin_2s": 1,
            "symbol": "H",
        },
        {
            "atomic_number": 6,
            "neutral_electron_count": 6,
            "spin_2s": 2,
            "symbol": "C",
        },
        {
            "atomic_number": 7,
            "neutral_electron_count": 7,
            "spin_2s": 3,
            "symbol": "N",
        },
        {
            "atomic_number": 8,
            "neutral_electron_count": 8,
            "spin_2s": 2,
            "symbol": "O",
        },
        {
            "atomic_number": 16,
            "neutral_electron_count": 16,
            "spin_2s": 2,
            "symbol": "S",
        },
        {
            "atomic_number": 17,
            "neutral_electron_count": 17,
            "spin_2s": 1,
            "symbol": "Cl",
        },
    ]


def test_atomic_displacement_preregistration_binds_source_and_forbids_repair():
    protocol = json.loads(PREREG.read_text(encoding="utf-8"))

    assert protocol["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/generate_route2_v0_atomic_displacement_response.py": _sha256(
            GENERATOR
        ),
        "maple/function/calculator/extra_correction/implicit/route2_v0_atomic_displacement_response.py": _sha256(
            SOURCE
        ),
    }
    assert protocol["hard_constraints"] == {
        "atomic_response_fit_to_qm_mep": False,
        "continuum_or_pcm_invoked": False,
        "experimental_solvation_fit": False,
        "fine_tuning": False,
        "mace_density_relabelled_as_electron_density": False,
        "mace_mdp_treated_as_energy_or_force_model": False,
        "map_or_uq_calibration": False,
        "post_training": False,
        "public_route_changed": False,
        "solvation_labels_read": False,
        "width_or_multipole_selected_from_error": False,
    }
    assert "does not permit a GTO projection" in protocol["decision_rule"]
