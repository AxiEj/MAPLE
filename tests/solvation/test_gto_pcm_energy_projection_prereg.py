from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-prereg-v1.json"
)
COMPLETED_ARTIFACT = (
    ROOT
    / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-four-v1.json"
)
PREREG_SHA256 = "84eaee61470affcb113de964e4aea3fdec15bb22c9aa23bda100828e704e80ef"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gto_pcm_energy_projection_four_record_prereg_is_locked():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))

    assert _sha256(PREREG) == PREREG_SHA256
    assert prereg["schema_version"] == 1
    assert prereg["artifact_id"] == ("route2-gto-pcm-energy-projection-four-prereg-v1")
    assert prereg["status"] == "frozen-before-transfer-execution"
    records = prereg["records"]
    assert [record["compound_id"] for record in records] == [
        "mobley_3053621",
        "mobley_3867265",
        "mobley_3034976",
        "mobley_352111",
    ]
    assert len({record["class"] for record in records}) == 4
    assert all(len(record["mol2_sha256"]) == 64 for record in records)
    assert [record["qm_reference"]["provenance_mode"] for record in records] == [
        "preregistered-gas-runner-v1",
        "legacy-frozen-checkpoint-v1",
        "preregistered-gas-runner-v1",
        "legacy-frozen-checkpoint-v1",
    ]
    assert [record["pcm_input"]["provenance_mode"] for record in records] == [
        "legacy-frozen-machine-input-v1",
        "legacy-frozen-machine-input-v1",
        "legacy-frozen-machine-input-v1",
        "preregistered-intrinsic-input-runner-v1",
    ]

    method = prereg["method"]
    assert method["basis_arms_angstrom"] == {
        "one_radial": [1.5],
        "two_radial": [1.5, 3.0],
    }
    assert method["relative_spectral_cutoffs"] == [
        1.0e-4,
        1.0e-6,
        1.0e-8,
        1.0e-10,
        1.0e-12,
    ]
    assert method["record_order"] == [record["compound_id"] for record in records]

    execution = prereg["execution_contract"]
    assert set(execution["source_sha256"]) == {
        (
            "docs/implicit-solvation/benchmarks/"
            "run_route2_gto_pcm_energy_projection_canary.py"
        ),
        "docs/implicit-solvation/benchmarks/route2_qm_surface_mep.py",
        "docs/implicit-solvation/benchmarks/run_route2_qm_gas_checkpoint.py",
        "docs/implicit-solvation/benchmarks/run_route2_intrinsic_pcm_input.py",
        (
            "maple/function/calculator/extra_correction/implicit/"
            "continuum_response.py"
        ),
        (
            "maple/function/calculator/extra_correction/implicit/"
            "electrostatic_pairing.py"
        ),
        "maple/function/calculator/extra_correction/implicit/gto_density.py",
        "maple/function/calculator/extra_correction/implicit/gto_galerkin.py",
        (
            "maple/function/calculator/extra_correction/implicit/"
            "pcm_energy_projection.py"
        ),
        "maple/function/calculator/extra_correction/implicit/pcmsolver.py",
        (
            "maple/function/calculator/extra_correction/implicit/"
            "route2_pcmsolver_cavity.py"
        ),
        "maple/function/calculator/extra_correction/implicit/smd.py",
        "maple/function/calculator/extra_correction/implicit/smd_cds.py",
        "maple/function/read/filereader/mol2_reader.py",
        "maple/function/route2_smd_profiles.py",
    }
    assert all(len(value) == 64 for value in execution["source_sha256"].values())
    completed = json.loads(COMPLETED_ARTIFACT.read_text(encoding="utf-8"))
    assert completed["scientific_status"] == "complete-negative-result"
    assert completed["decision"] == "fail-preregistered-transfer-gate"
    provenance = completed["provenance_validation"]
    assert provenance["all_raw_source_hash_maps_match_preregistration"] is True
    assert (
        provenance["common_frozen_hashes"]["source_sha256"]
        == execution["source_sha256"]
    )
    # The completed result, rather than today's checkout, owns the historical
    # frozen source snapshot.  Later production refactors must make the old
    # runner fail closed without invalidating or rewriting that result.
    drifted_sources = {
        relative
        for relative, expected_sha in execution["source_sha256"].items()
        if _sha256(ROOT / relative) != expected_sha
    }
    assert {
        "maple/function/calculator/extra_correction/implicit/smd.py",
        "maple/function/route2_smd_profiles.py",
    } <= drifted_sources
    assert execution["gas_runner_effective_arguments"] == {
        "basis": "def2-tzvpd",
        "grid_level": 3,
        "max_memory_mb": 8000,
        "nlc_grid_level": 3,
        "nlc_grid_profile": "pyscf-official-50x194-sg1",
        "threads": 8,
    }
    assert execution["comparison_tolerances"] == {
        "maximum_error_threshold_is_strict": True,
        "mean_absolute_error_comparison_kcal_per_mol": 1.0e-9,
        "per_record_improvement_tie_kcal_per_mol": 1.0e-9,
        "residual_energy_norm_monotonic_absolute_hartree": 1.0e-12,
    }

    gates = prereg["decision_gates"]
    assert (
        gates[
            "at_cutoff_1e-4_two_radial_maximum_absolute_projection_error_kcal_mol_below"
        ]
        == 1.0
    )
    assert gates["at_cutoff_1e-4_two_radial_improvement_count_minimum"] == 3
    assert (
        gates[
            "residual_energy_norm_is_nonincreasing_as_cutoff_decreases_for_each_record_and_basis"
        ]
        is True
    )
    assert (
        "hydration free-energy or experimental accuracy"
        in prereg["claim_boundary"]["cannot_establish"]
    )
    assert gates["full_tangent_gradient_is_ungated_diagnostic"] is True
