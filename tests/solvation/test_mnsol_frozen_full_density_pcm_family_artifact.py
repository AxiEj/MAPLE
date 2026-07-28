from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-frozen-full-density-pcm-family-v2.json"
)
LEARNED_SMOKE_ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-salted-learned-water-pcm-smoke-v1.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    payload = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_frozen_full_density_artifact_is_complete_private_safe_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == ("route2-mnsol-frozen-full-density-pcm-family-v2")
    assert artifact["schema_version"] == 1
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["do_not_commit"] is False
    assert artifact["complete_panel"] is True
    assert artifact["execution_git_head"] == (
        "f9c354f6ace13abaac1c18e39a6f4361eedf60be"
    )
    assert artifact["preregistration"]["sha256"] == (
        "aa121700c6bfc6d8b5d61e08b9845a5392f84063e83dd660208edf0eebb46cef"
    )
    assert artifact["selection"]["record_count"] == 10
    assert artifact["selection"]["solvent_count"] == 10
    assert len(artifact["selection"]["functional_group_coverage"]) == 10
    assert artifact["experimental_reference"]["row_level_data_emitted"] is False

    identity = artifact["scientific_identity"]
    assert identity["learned_salted_prediction_included"] is False
    assert identity["strict_original_smd_equivalence"] is False
    assert artifact["runtime"]["salted_runtime_imported"] is False
    assert artifact["salted_upstream"] == {
        "package_version": "3.0.0",
        "repository": "https://github.com/andreagrisafi/SALTED",
        "reviewed_git_commit": "fd5adaa1c682ae7f4ce580798d79bfe499783784",
    }

    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "experimental_delta_g_kcal_mol",
        "opaque_record_id",
        "records",
        "density_matrix",
        "ri_coefficients",
    }
    assert forbidden.isdisjoint(_all_keys(artifact))

    for relative_path, digest in artifact["source_files_sha256"].items():
        assert _git_blob_sha256(artifact["execution_git_head"], relative_path) == digest


@pytest.mark.parametrize(
    ("equation", "mace_mae", "ao_mae", "ri_mae", "ao_wins", "ri_wins"),
    (
        (
            "iefpcm",
            0.8780065988654894,
            0.486692316052709,
            0.4790567961950307,
            8,
            8,
        ),
        (
            "cpcm",
            0.9127298962577448,
            0.5914300990984641,
            0.6152723811847742,
            8,
            7,
        ),
        (
            "cosmo",
            0.8737904414643213,
            0.4836313776346154,
            0.4760219639739144,
            9,
            9,
        ),
    ),
)
def test_frozen_full_density_metrics_and_paired_improvement_are_immutable(
    equation: str,
    mace_mae: float,
    ao_mae: float,
    ri_mae: float,
    ao_wins: int,
    ri_wins: int,
):
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    metrics = artifact["aggregate_metrics"]

    mace = f"mace_fixed_l1__pyscf_swig_{equation}"
    ao = f"qm_ao_frozen__pyscf_swig_{equation}"
    ri = f"qm_ri_salted_space_frozen__pyscf_swig_{equation}"

    assert metrics[mace]["mean_absolute_error_kcal_mol"] == pytest.approx(
        mace_mae, abs=1.0e-12
    )
    assert metrics[ao]["mean_absolute_error_kcal_mol"] == pytest.approx(
        ao_mae, abs=1.0e-12
    )
    assert metrics[ri]["mean_absolute_error_kcal_mol"] == pytest.approx(
        ri_mae, abs=1.0e-12
    )
    assert metrics[ao]["record_count"] == metrics[ri]["record_count"] == 10

    ao_pair = artifact["paired_method_comparisons"][f"{mace}__to__{ao}"]
    ri_pair = artifact["paired_method_comparisons"][f"{mace}__to__{ri}"]
    assert ao_pair["right_lower_absolute_error_count"] == ao_wins
    assert ri_pair["right_lower_absolute_error_count"] == ri_wins
    assert ao_pair["right_minus_left_mean_absolute_error_kcal_mol"] == pytest.approx(
        ao_mae - mace_mae, abs=1.0e-12
    )
    assert ri_pair["right_minus_left_mean_absolute_error_kcal_mol"] == pytest.approx(
        ri_mae - mace_mae, abs=1.0e-12
    )


def test_salted_compatible_ri_representation_is_close_to_direct_ao_density():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    summary = artifact["representation_summary"]

    assert summary["record_count"] == 10
    assert summary["mean_AO_to_RI_surface_MEP_RMSE_hartree_per_e"] == (
        pytest.approx(0.00029179557303945553, abs=1.0e-15)
    )
    assert summary["maximum_AO_to_RI_surface_MEP_RMSE_hartree_per_e"] == (
        pytest.approx(0.0003446103640544204, abs=1.0e-15)
    )
    assert summary["maximum_RI_raw_charge_residual_e"] == pytest.approx(
        0.0011292162193115018, abs=1.0e-15
    )
    assert summary["maximum_RI_projected_charge_residual_e"] < 1.5e-14

    comparisons = artifact["paired_method_comparisons"]
    for equation, expected_delta in (
        ("iefpcm", -0.007635519857678297),
        ("cpcm", 0.023842282086310118),
        ("cosmo", -0.007609413660701003),
    ):
        ao = f"qm_ao_frozen__pyscf_swig_{equation}"
        ri = f"qm_ri_salted_space_frozen__pyscf_swig_{equation}"
        assert comparisons[f"{ao}__to__{ri}"][
            "right_minus_left_mean_absolute_error_kcal_mol"
        ] == pytest.approx(expected_delta, abs=1.0e-12)


def test_learned_salted_water_smoke_proves_connector_not_transferability():
    artifact = json.loads(LEARNED_SMOKE_ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == "route2-salted-learned-water-pcm-smoke-v1"
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["post_hoc_engineering_smoke"] is True
    assert artifact["record_count"] == 2
    assert artifact["row_level_data_emitted"] is False
    assert artifact["scientific_identity"]["solute_source"] == (
        "learned SALTED 3.0.0 RI Gaussian density"
    )
    assert artifact["scientific_identity"]["model_scope"] == ("water monomers only")
    assert artifact["training"]["training_molecule_count"] == 8
    assert artifact["training"]["validation_density_rmse_percent"] == (
        pytest.approx(12.22, abs=1.0e-12)
    )
    assert artifact["aggregate"][
        "mean_surface_mep_rmse_hartree_per_e"
    ] == pytest.approx(0.0010032837724772382, abs=1.0e-15)
    assert artifact["aggregate"][
        "mean_absolute_pcm_polarization_error_kcal_mol"
    ] == pytest.approx(0.11713546603682401, abs=1.0e-12)
    assert artifact["aggregate"]["maximum_projected_charge_residual_e"] < 2.0e-15

    forbidden = {
        "configuration_index",
        "coordinates",
        "records",
        "coefficients",
        "reference_polarization_kcal_mol",
        "learned_salted_polarization_kcal_mol",
    }
    assert forbidden.isdisjoint(_all_keys(artifact))
