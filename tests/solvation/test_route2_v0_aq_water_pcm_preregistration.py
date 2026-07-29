from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs/implicit-solvation/benchmarks"
PREREGISTRATION = BENCHMARKS / "route2-v0-aq-water-pcm-prereg-v1.json"
GEOMETRY = BENCHMARKS / "route2-v0-aq-water-geometry-v1.json"
RUNNER = BENCHMARKS / "run_route2_v0_auxiliary_qm_pcm_water.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def test_v0_aq_water_pcm_preregistration_binds_the_control_before_execution():
    protocol = _protocol()

    assert protocol["protocol_id"] == "route2-v0-aq-water-pcm-prereg-v1"
    assert protocol["status"] == "frozen-before-execution"
    assert "does not evaluate MACE gas energy" in protocol["claim_boundary"]
    assert "total solvation free energy" in protocol["claim_boundary"]
    assert protocol["qm_method"] == {
        "implementation": "PySCF",
        "pyscf_version": "2.13.1",
        "electronic_structure": "omegaB97M-V",
        "pyscf_xc_token": "wb97m-v",
        "basis": "def2-tzvpd",
        "reference": "RKS",
        "density_fitting": True,
        "charge": 0,
        "spin": 0,
        "semilocal_grid_level": 3,
        "nonlocal_grid_profile": "50x194-SG1",
        "scf_energy_tolerance_hartree": 1.0e-10,
        "scf_gradient_tolerance": 1.0e-7,
        "maximum_scf_cycles": 100,
    }
    assert protocol["pcm_control"] == {
        "method": "IEFPCM",
        "dielectric_constant": 78.3553,
        "vdw_scale": 1.0,
        "probe_radius_angstrom": 0.0,
        "lebedev_order": 17,
        "surface_discretization_method": "SWIG",
        "standard_state_energy_included": False,
        "empirical_cds_included": False,
    }


def test_v0_aq_water_pcm_preregistration_source_and_geometry_are_content_bound():
    protocol = _protocol()
    contract = protocol["execution_contract"]

    assert contract["source_sha256"] == {
        "docs/implicit-solvation/benchmarks/run_route2_v0_auxiliary_qm_pcm_water.py": _sha256(
            RUNNER
        )
    }
    assert contract["input_sha256"] == {
        "docs/implicit-solvation/benchmarks/route2-v0-aq-water-geometry-v1.json": _sha256(
            GEOMETRY
        )
    }
    runtime = contract["runtime"]
    assert runtime["pyscf_version"] == "2.13.1"
    assert runtime["threads"] == 8
    assert runtime["max_memory_mb"] == 8000


def test_v0_aq_water_pcm_preregistration_rejects_semantic_shortcuts():
    protocol = _protocol()
    ledger = protocol["fixed_energy_ledger"]
    forbidden = " ".join(ledger["forbidden"])

    assert "A_aux^(PCM-SCF)-A_aux^(gas-SCF)" in ledger["auxiliary_difference"]
    assert (
        "E_MACE,gas+[A_aux^(PCM-SCF)-A_aux^(gas-SCF)]"
        in ledger["future_mace_composite"]
    )
    assert "isolated implementation-specific PCM energy component" in forbidden
    assert "MACE field response" in forbidden
    assert "post-training" in forbidden
    assert "fine-tuning" in forbidden
    assert "target-label use" in forbidden
    assert "No force, PES, optimization" in protocol["force_boundary"]


def test_v0_aq_water_pcm_runner_never_uses_an_isolated_pcm_component():
    source = RUNNER.read_text(encoding="utf-8")

    assert "total_stationary_energy_hartree" in source
    assert "A_aux^(PCM-SCF)-A_aux^(gas-SCF)" in source
    assert "with_solvent.e" not in source
    assert "SMD CDS or another empirical non-electrostatic term" in source
