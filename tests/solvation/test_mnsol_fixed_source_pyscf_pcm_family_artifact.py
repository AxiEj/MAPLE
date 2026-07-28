from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_source_binding import assert_source_files_match_execution_commit

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "docs/implicit-solvation/benchmarks/"
    "route2-mnsol-fixed-source-pyscf-pcm-family-v1.json"
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_pcm_family_artifact_is_complete_private_safe_and_source_bound():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["artifact"] == ("route2-mnsol-fixed-source-pyscf-pcm-family-v1")
    assert artifact["execution_git_head"] == (
        "da9d3ba112b7ba2e1f0a681c9b28b3aa32f19c39"
    )
    assert artifact["visibility"] == "public-aggregate-only"
    assert artifact["do_not_commit"] is False
    assert artifact["complete_panel"] is True
    assert artifact["preregistration"]["sha256"] == (
        "90d84d89d96d1db14ceb2a2e2d5d4569f9544177ac72ac64b500570ca315da54"
    )
    assert artifact["selection"]["record_count"] == 10
    assert artifact["selection"]["solvent_count"] == 10
    assert len(artifact["selection"]["functional_group_coverage"]) == 10
    assert artifact["experimental_reference"]["doi"] == "10.13020/3eks-j059"
    assert artifact["experimental_reference"]["row_level_data_emitted"] is False

    forbidden = {
        "entry_number",
        "geometry_handle",
        "solute_name",
        "formula",
        "experimental_delta_g_kcal_mol",
        "aimnet2_charges_e",
        "mace_gas_density_coefficients",
        "records",
    }
    assert forbidden.isdisjoint(_all_keys(artifact))
    assert_source_files_match_execution_commit(ROOT, artifact)


@pytest.mark.parametrize(
    ("method", "mae", "rmse", "maximum"),
    (
        (
            "aimnet2_fixed_l0__pyscf_swig_iefpcm",
            1.1297309378064637,
            1.3547201942623288,
            2.272731605819758,
        ),
        (
            "aimnet2_fixed_l0__pyscf_swig_cpcm",
            1.0395967316091395,
            1.2952799603566252,
            2.243410047319247,
        ),
        (
            "aimnet2_fixed_l0__pyscf_swig_cosmo",
            1.1142242850072284,
            1.3518731506298027,
            2.28554524031538,
        ),
        (
            "mace_fixed_l1__pyscf_swig_iefpcm",
            0.8780065988654894,
            1.0020735142451092,
            1.6458079867776407,
        ),
        (
            "mace_fixed_l1__pyscf_swig_cpcm",
            0.9127298962577448,
            1.0291292727050958,
            1.6455778130061316,
        ),
        (
            "mace_fixed_l1__pyscf_swig_cosmo",
            0.8737904414643213,
            1.0003500577251874,
            1.6457345128327945,
        ),
    ),
)
def test_pcm_family_aggregate_metrics_are_immutable(
    method: str,
    mae: float,
    rmse: float,
    maximum: float,
):
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    metrics = artifact["aggregate_metrics"][method]

    assert metrics["record_count"] == 10
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(
        mae,
        abs=1.0e-12,
    )
    assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
        rmse,
        abs=1.0e-12,
    )
    assert metrics["maximum_absolute_error_kcal_mol"] == pytest.approx(
        maximum,
        abs=1.0e-12,
    )


def test_pcm_family_equation_and_source_differences_remain_descriptive():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    comparisons = artifact["paired_method_comparisons"]

    aimnet_cpcm = comparisons[
        "aimnet2_fixed_l0__pyscf_swig_iefpcm__to__" "aimnet2_fixed_l0__pyscf_swig_cpcm"
    ]
    assert aimnet_cpcm["right_lower_absolute_error_count"] == 8
    assert aimnet_cpcm[
        "right_minus_left_mean_absolute_error_kcal_mol"
    ] == pytest.approx(-0.09013420619732426, abs=1.0e-12)

    mace_cosmo = comparisons[
        "mace_fixed_l1__pyscf_swig_iefpcm__to__" "mace_fixed_l1__pyscf_swig_cosmo"
    ]
    assert mace_cosmo["left_lower_absolute_error_count"] == 5
    assert mace_cosmo["right_lower_absolute_error_count"] == 5
    assert mace_cosmo["right_minus_left_mean_absolute_error_kcal_mol"] == pytest.approx(
        -0.004216157401168097, abs=1.0e-12
    )

    source_shift = comparisons[
        "aimnet2_fixed_l0__pyscf_swig_iefpcm__to__" "mace_fixed_l1__pyscf_swig_iefpcm"
    ]
    assert source_shift["right_lower_absolute_error_count"] == 7
    assert source_shift[
        "right_minus_left_mean_absolute_error_kcal_mol"
    ] == pytest.approx(-0.25172433894097435, abs=1.0e-12)

    identity = artifact["scientific_identity"]
    assert identity["strict_original_smd_equivalence"] is False
    assert identity["cosmo_rs_included"] is False
    assert artifact["timing_seconds"]["status"] == (
        "metadata-only-not-a-randomized-speed-ranking"
    )
    assert artifact["same_surface_summary"][
        "maximum_point_difference_bohr"
    ] == pytest.approx(0.0, abs=0.0)
    assert artifact["same_surface_summary"][
        "maximum_area_difference_bohr2"
    ] == pytest.approx(0.0, abs=0.0)
