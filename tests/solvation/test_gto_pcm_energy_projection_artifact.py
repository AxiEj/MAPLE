from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-gto-pcm-energy-projection-acetone-v1.json"
)
ARTIFACT_SHA256 = "9f5d13432ee5a3759bcc45ec6b0e6e845d3eb977d16cd7f3578c9813f3660823"


def _sha256(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_gto_pcm_energy_projection_artifact_is_immutable_and_source_bound():
    raw = ARTIFACT.read_bytes()
    artifact = json.loads(raw)
    assert _sha256(raw) == ARTIFACT_SHA256
    assert artifact["schema_version"] == 1
    assert artifact["artifact_id"] == ("route2-gto-pcm-energy-projection-acetone-v1")
    assert artifact["scientific_status"] == (
        "fixed-geometry representation feasibility canary; "
        "not a variational MACE-POLAR model"
    )

    source = artifact["source"]
    source_head = source["git_head"]
    assert source_head == "4959e588ccf19111990454bd9287fbefcbace5d9"
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_head, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    for relative_path, expected_sha256 in source["source_sha256"].items():
        blob = subprocess.run(
            ["git", "show", f"{source_head}:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert _sha256(blob) == expected_sha256

    assert "--density" not in source["helper_command"]
    assert source["runner_argv"][1].endswith(
        "run_route2_gto_pcm_energy_projection_canary.py"
    )
    assert source["helper_command"][1].endswith("route2_qm_surface_mep.py")

    inputs = artifact["inputs"]
    assert inputs["compound_id"] == "mobley_3867265"
    assert inputs["geometry_max_abs_error_angstrom"] < 1.0e-8
    assert inputs["qm_total_charge_e"] == pytest.approx(0.0, abs=1.0e-12)
    assert inputs["qm_density"] == {
        "checkpoint_density_binding_residual_e": 0.0,
        "checkpoint_mo_orthonormality_inf": pytest.approx(
            1.0043371057196852e-11,
            abs=1.0e-20,
        ),
        "checkpoint_occupation_sum_e": 32.0,
        "checkpoint_occupied_orbital_count": 16,
        "source": "checkpoint:scf/mo_coeff+scf/mo_occ",
    }
    for key in ("molecule", "parsed_pcm_input", "qm_checkpoint"):
        assert len(inputs[key]["sha256"]) == 64

    runtime = artifact["runtime"]
    assert runtime["pyscf"] == "2.13.1"
    assert runtime["pcmsolver_library"]["sha256"] == (
        "296b6f34a03789943ae8823b3790f36357c50c896497f21374ac16fe5a6c43a3"
    )


def test_gto_pcm_energy_projection_acetone_gate_is_locked():
    artifact = _artifact()
    one = artifact["basis_results"]["one_radial"]
    two = artifact["basis_results"]["two_radial"]

    assert one["widths_angstrom"] == [1.5]
    assert two["widths_angstrom"] == [1.5, 3.0]
    assert one["coefficient_count"] == 40
    assert two["coefficient_count"] == 80
    assert one["surface_point_count"] == two["surface_point_count"] == 516
    assert one["target_polarization_energy_kcal_per_mol"] == pytest.approx(
        -6.6433389623537735,
        abs=1.0e-12,
    )
    assert two["target_polarization_energy_kcal_per_mol"] == pytest.approx(
        one["target_polarization_energy_kcal_per_mol"],
        abs=1.0e-12,
    )
    assert one["fitted_polarization_energy_kcal_per_mol"] == pytest.approx(
        -6.232217191905792,
        abs=1.0e-12,
    )
    assert two["fitted_polarization_energy_kcal_per_mol"] == pytest.approx(
        -6.576596044147263,
        abs=1.0e-12,
    )
    assert one["polarization_energy_error_kcal_per_mol"] == pytest.approx(
        0.4111217704479819,
        abs=1.0e-12,
    )
    assert two["polarization_energy_error_kcal_per_mol"] == pytest.approx(
        0.06674291820651013,
        abs=1.0e-12,
    )

    for record in (one, two):
        assert record["constraint_residual_inf"] < 1.0e-10
        assert record["tangent_optimality_inf"] < 1.0e-9
        assert record["shifted_pythagorean_error_hartree"] < 1.0e-8
        assert record["operator"]["antisymmetric_norm_hartree"] < 1.0e-12
        assert record["operator"]["maximum_eigenvalue_hartree"] <= 1.0e-10
        assert math.isfinite(record["elapsed_seconds"])
        assert record["elapsed_seconds"] > 0.0

    assert artifact["gates"] == {
        "absolute_error_improvement_kcal_per_mol": pytest.approx(
            0.3443788522414718,
            abs=1.0e-12,
        ),
        "algebraic_projection_gates_passed": True,
        "one_radial_absolute_error_kcal_per_mol": pytest.approx(
            0.4111217704479819,
            abs=1.0e-12,
        ),
        "two_radial_absolute_error_kcal_per_mol": pytest.approx(
            0.06674291820651013,
            abs=1.0e-12,
        ),
        "two_radial_improves_energy_projection": True,
    }
    assert (
        "hydration free-energy accuracy"
        in artifact["claim_boundary"]["does_not_certify"]
    )
    assert (
        "forces or a solution-phase PES"
        in artifact["claim_boundary"]["does_not_certify"]
    )
