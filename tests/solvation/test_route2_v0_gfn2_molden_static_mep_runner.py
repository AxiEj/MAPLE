"""Unit guards for the preregistered GFN2 static-MEP runner mechanics."""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = (
    _REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/"
    "run_route2_v0_gfn2_molden_static_mep_acetone.py"
)
_RUNNER = runpy.run_path(str(_RUNNER_PATH))


def test_static_mep_metrics_use_frozen_global_norms_without_point_selection():
    relative_frobenius = _RUNNER["_relative_frobenius"]
    relative_max_abs = _RUNNER["_relative_max_abs"]
    candidate = np.array([2.0, -1.0, 4.0])
    reference = np.array([1.0, -2.0, 2.0])

    assert relative_frobenius(candidate, reference) == pytest.approx(
        np.linalg.norm(candidate - reference) / np.linalg.norm(reference)
    )
    assert relative_max_abs(candidate, reference) == pytest.approx(1.0)


def test_static_mep_representation_checks_require_every_registered_identity():
    checks = _RUNNER["_representation_checks"](
        {
            "candidate_representation": {
                "pyscf_vs_molden_overlap_relative_frobenius": 2.0e-10,
                "pyscf_vs_molden_overlap_max_abs": 3.0e-10,
                "pyscf_ao_metric_error": 4.0e-10,
                "pyscf_vs_molden_electronic_dipole_error_e_bohr": 5.0e-9,
                "pyscf_vs_molden_total_dipole_error_e_bohr": 6.0e-9,
            },
            "qm_reference": {
                "checkpoint_ao_density": {
                    "mo_metric_error": 7.0e-10,
                    "electron_count_error_e": 8.0e-10,
                },
                "total_charge_e": -9.0e-10,
                "geometry_max_abs_error_bohr": 1.0e-10,
            },
        },
        {
            "candidate_pyscf_overlap_relative_frobenius_max": 5.0e-8,
            "candidate_pyscf_overlap_max_abs_max": 5.0e-8,
            "candidate_pyscf_mo_metric_max": 5.0e-8,
            "candidate_pyscf_electronic_dipole_e_bohr_max": 1.0e-6,
            "candidate_pyscf_total_dipole_e_bohr_max": 1.0e-6,
            "qm_checkpoint_mo_metric_max": 1.0e-7,
            "qm_checkpoint_electron_count_e_max": 1.0e-7,
            "qm_checkpoint_total_charge_e_max": 1.0e-7,
            "geometry_max_abs_error_bohr_max": 1.0e-8,
        },
    )

    assert set(checks) == {
        "candidate_pyscf_overlap_relative_frobenius",
        "candidate_pyscf_overlap_max_abs",
        "candidate_pyscf_mo_metric",
        "candidate_pyscf_electronic_dipole_e_bohr",
        "candidate_pyscf_total_dipole_e_bohr",
        "qm_checkpoint_mo_metric",
        "qm_checkpoint_electron_count_e",
        "qm_checkpoint_total_charge_e",
        "geometry_max_abs_error_bohr",
    }
    assert all(check["passes"] for check in checks.values())


def test_static_mep_checkpoint_density_must_match_the_prior_qm_record():
    raw_path = (
        _REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks/reproducers/"
        "route2-v0-mace-mdp-induced-source-acetone-v1/qm-induced-mep.json"
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    expected = raw["zero_field"]["density_sha256"]
    validator = _RUNNER["_validate_qm_density_binding"]

    validator({"qm_reference": {"checkpoint_ao_density": {"density_sha256": expected}}})
    with pytest.raises(RuntimeError, match="not bound"):
        validator(
            {"qm_reference": {"checkpoint_ao_density": {"density_sha256": "0" * 64}}}
        )


def test_static_mep_helper_runtime_is_version_bound():
    validator = _RUNNER["_validate_helper_runtime"]

    validator({"qm_reference": {"pyscf_version": "2.13.1"}})
    with pytest.raises(RuntimeError, match="frozen PySCF"):
        validator({"qm_reference": {"pyscf_version": "2.12.0"}})
