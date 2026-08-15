from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maple.solvation.reference.replay import audit_complete_reference_replay

CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_case(*, energy_shift: float = 0.0) -> dict[str, object]:
    identity = {
        "cavity_profile_id": "cavity-v1",
        "continuum_configuration_sha256": "1" * 64,
        "continuum_equation_id": "continuum-v1",
        "continuum_protocol_sha256": "2" * 64,
        "continuum_provenance_sha256": "3" * 64,
        "contract_version": "reference-v1",
        "dielectric": 78.355,
        "electrostatics_only": True,
        "experimental_solvation_labels_used": False,
        "fixed_nuclear_geometry": True,
        "geometry_sha256": "4" * 64,
        "molecule_group_sha256": "5" * 64,
        "nonpolar_terms_included": False,
        "pcm_density_reoptimized_in_vacuum": False,
        "pcm_density_self_consistent": True,
        "qm_runtime_manifest_sha256": "6" * 64,
        "reference_method_id": "qm-v1",
        "reference_protocol_sha256": "7" * 64,
        "spin_multiplicity": 1,
        "standard_state_terms_included": False,
        "topology_sha256": "8" * 64,
        "total_charge": 0,
        "vacuum_density_self_consistent": True,
        "vacuum_density_sha256": "9" * 64,
    }
    return {
        **identity,
        "solute_distortion_energy_eV": 0.2 + energy_shift,
        "continuum_stabilization_energy_eV": -0.8 + energy_shift,
        "total_electrostatic_solvation_energy_eV": -0.6 + energy_shift,
        "pcm_electrostatic_total_energy_eV": -100.6 + energy_shift,
        "polarized_density_vacuum_energy_eV": -99.8 + energy_shift,
        "pcm_density_sha256": ("a" if energy_shift == 0.0 else "b") * 64,
        "reference_boundary_rhs_sha256": ("c" if energy_shift == 0.0 else "d") * 64,
        "content_sha256": ("e" if energy_shift == 0.0 else "f") * 64,
        "reference_artifact_sha256": ("0" if energy_shift == 0.0 else "a") * 64,
    }


def _write_run(
    root: Path,
    *,
    preregistration_sha256: str,
    energy_shift: float,
    density_shift: float,
    scientific_sha: str,
) -> None:
    case_id = "case-1"
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    reference = _reference_case(energy_shift=energy_shift)
    case = {
        "schema_version": "route2-matched-qm-pcmsolver-decomposition-run-v3",
        "status": "pass-matched-electrostatic-reference-case",
        "capabilities": CAPABILITIES,
        "reference_case": reference,
    }
    case_path = case_dir / "case.json"
    case_path.write_text(json.dumps(case, sort_keys=True), encoding="utf-8")
    np.savez(
        case_dir / "surface.npz",
        surface_points_bohr=np.asarray([[1.0, 0.0, 0.0]]),
        surface_areas_bohr2=np.asarray([0.5]),
    )
    np.savez(
        case_dir / "pcm-state.npz",
        atomic_numbers=np.asarray([1]),
        atom_positions_bohr=np.asarray([[0.0, 0.0, 0.0]]),
        ao_density_matrix=np.asarray([[2.0 + density_shift]]),
        total_surface_mep_hartree_per_e=np.asarray([0.4 + density_shift]),
        apparent_surface_charge_e=np.asarray([-0.1 + density_shift]),
        polarization_energy_hartree=np.asarray(-0.02 + density_shift),
    )
    result = {
        "schema_version": "route2-matched-qm-pcmsolver-decomposition-run-v3",
        "status": "pass-complete-matched-electrostatic-reference-panel",
        "panel_complete": True,
        "capabilities": CAPABILITIES,
        "repository": {"head": "a" * 40, "tree": "b" * 40, "clean": True},
        "runtime": {"threads": 8},
        "pcmsolver_library": {"sha256": "c" * 64},
        "source_sha256": {"runner.py": "d" * 64},
        "preregistration": {"sha256": preregistration_sha256},
        "requested_case_ids": [case_id],
        "full_preregistered_case_ids": [case_id],
        "case_records": [
            {
                "case_id": case_id,
                "case_artifact": {"sha256": _sha256(case_path)},
                "reference_case": reference,
            }
        ],
        "scientific_payload_sha256": scientific_sha,
    }
    (root / "result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    preregistration = {
        "protocol_id": "route2-matched-qm-pcmsolver-four-prereg-v3",
        "status": "frozen-before-execution",
        "case_ids": ["case-1"],
        "numerical_gates": {
            "repeat_energy_abs_eV_max": 1.0e-8,
            "repeat_pcm_density_inf_max": 1.0e-8,
            "repeat_boundary_mep_inf_hartree_per_e_max": 1.0e-10,
            "repeat_asc_inf_e_max": 1.0e-10,
            "repeat_polarization_energy_abs_hartree_max": 1.0e-10,
        },
    }
    prereg = tmp_path / "prereg.json"
    prereg.write_text(json.dumps(preregistration, sort_keys=True), encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(
        first,
        preregistration_sha256=_sha256(prereg),
        energy_shift=0.0,
        density_shift=0.0,
        scientific_sha="1" * 64,
    )
    _write_run(
        second,
        preregistration_sha256=_sha256(prereg),
        energy_shift=1.0e-11,
        density_shift=1.0e-12,
        scientific_sha="2" * 64,
    )
    return first, second, prereg


def test_replay_accepts_exact_identity_with_bounded_float_differences(
    tmp_path: Path,
) -> None:
    first, second, prereg = _write_fixture(tmp_path)
    audit = audit_complete_reference_replay(first, second, prereg)
    assert audit["status"] == "pass-tolerance-bounded-complete-panel-replay"
    assert audit["capabilities"] == CAPABILITIES
    assert audit["raw_scientific_payload_sha256_equal"] is False
    assert audit["observed_maxima"]["energy_abs_eV"] == pytest.approx(1.0e-11)
    assert audit["observed_maxima"]["pcm_density_inf"] == pytest.approx(1.0e-12)


def test_replay_rejects_changed_exact_identity(tmp_path: Path) -> None:
    first, second, prereg = _write_fixture(tmp_path)
    case_path = second / "case-1/case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["reference_case"]["topology_sha256"] = "0" * 64
    case_path.write_text(json.dumps(case, sort_keys=True), encoding="utf-8")
    result_path = second / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["case_records"][0]["case_artifact"]["sha256"] = _sha256(case_path)
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="topology_sha256"):
        audit_complete_reference_replay(first, second, prereg)


def test_replay_rejects_numerical_drift_over_registered_gate(
    tmp_path: Path,
) -> None:
    first, second, prereg = _write_fixture(tmp_path)
    state_path = second / "case-1/pcm-state.npz"
    state = dict(np.load(state_path))
    state["ao_density_matrix"] = np.asarray([[2.0 + 2.0e-8]])
    np.savez(state_path, **state)
    with pytest.raises(RuntimeError, match="pcm_density_inf"):
        audit_complete_reference_replay(first, second, prereg)


def test_replay_module_does_not_import_optional_qm_runtime() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pyscf' or name.startswith('pyscf.'):
        raise RuntimeError('optional QM runtime imported')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import maple.solvation.reference.replay
"""
    subprocess.run([sys.executable, "-c", code], check=True)
