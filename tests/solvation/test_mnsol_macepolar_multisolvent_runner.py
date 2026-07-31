from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_mnsol_macepolar_multisolvent_pilot as runner  # noqa: E402


def _record(
    *,
    experimental: float = -4.0,
    ddpcm_total: float = -4.5,
    ddcosmo_total: float = -4.1,
) -> dict[str, object]:
    def _method(total: float) -> dict[str, float | int]:
        signed_error = total - experimental
        return {
            "total_solvation_kcal_mol": total,
            "signed_error_kcal_mol": signed_error,
            "absolute_error_kcal_mol": abs(signed_error),
            "solute_polarization_kcal_mol": 0.5,
            "continuum_polarization_kcal_mol": total - 1.0,
            "electrostatic_kcal_mol": total - 0.5,
            "smd_cds_energy_kcal_mol": 0.5,
            "scf_iterations": 7,
            "unmixed_density_residual_inf_e": 1.0e-12,
            "half_coupling_identity_error_ev": 2.0e-14,
        }

    return {
        "experimental_delta_g_kcal_mol": experimental,
        "methods": {
            "ddpcm": _method(ddpcm_total),
            "ddcosmo": _method(ddcosmo_total),
        },
    }


def test_single_record_smoke_selection_preserves_full_panel_contract():
    full_selection = tuple(f"record-{index}" for index in range(10))

    indexed, complete = runner._indexed_selection(full_selection, None)
    assert complete is True
    assert indexed == list(enumerate(full_selection))

    indexed, complete = runner._indexed_selection(full_selection, 3)
    assert complete is False
    assert indexed == [(3, "record-3")]

    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        runner._indexed_selection(full_selection, 10)
    with pytest.raises(RuntimeError, match="exactly 10"):
        runner._indexed_selection(full_selection[:-1], None)


def test_single_record_smoke_metrics_are_descriptive_not_ten_only():
    records = [_record()]

    metrics = runner._metrics(records, "ddcosmo")
    paired = runner._paired_method_comparison(records)

    assert metrics["record_count"] == 1
    assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(-0.1)
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(0.1)
    assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(0.1)
    assert paired["record_count"] == 1
    assert paired["ddcosmo_lower_absolute_error_count"] == 1
    assert paired["ddpcm_lower_absolute_error_count"] == 0

    with pytest.raises(RuntimeError, match="finite nonempty"):
        runner._metrics([], "ddpcm")
    with pytest.raises(RuntimeError, match="finite nonempty"):
        runner._paired_method_comparison([])


def test_energy_ledger_selects_a_locked_paired_equation_comparison():
    legacy_profiles = runner.method_profiles_for_energy_ledger(
        runner.LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
    )
    direct_profiles = runner.method_profiles_for_energy_ledger(
        runner.PCM_HALF_COUPLING_ONLY_V1
    )

    assert tuple(method for method, _profile in legacy_profiles) == (
        "ddpcm",
        "ddcosmo",
    )
    assert tuple(method for method, _profile in direct_profiles) == (
        "ddpcm",
        "ddcosmo",
    )
    assert legacy_profiles != direct_profiles
    assert runner.artifact_name_for_energy_ledger(
        runner.LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
    ) == runner.ARTIFACT_NAME
    assert runner.artifact_name_for_energy_ledger(
        runner.PCM_HALF_COUPLING_ONLY_V1
    ) == runner.DIRECT_PCM_ARTIFACT_NAME


def test_parser_keeps_legacy_control_default_and_accepts_direct_pcm():
    common = [
        "--source",
        "source.tsv",
        "--protocol",
        "protocol.json",
        "--selection",
        "selection.json",
        "--private-output",
        ".omx/private.json",
        "--public-output",
        ".omx/public.json",
        "--work-dir",
        ".omx/work",
    ]
    parser = runner._build_parser()

    assert parser.parse_args(common).energy_ledger == (
        runner.LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1
    )
    assert parser.parse_args(
        [
            *common,
            "--energy-ledger",
            runner.PCM_HALF_COUPLING_ONLY_V1,
        ]
    ).energy_ledger == runner.PCM_HALF_COUPLING_ONLY_V1


def test_single_record_smoke_forces_both_outputs_below_omx():
    private = ROOT / ".omx/test-smoke/private.json"
    public = ROOT / ".omx/test-smoke/summary.json"
    work = ROOT / ".omx/test-smoke/work"

    resolved = runner._validated_output_paths(
        private_output=private,
        public_output=public,
        work_dir=work,
        complete_panel=False,
    )
    assert resolved == (private.resolve(), public.resolve(), work.resolve())

    with pytest.raises(ValueError, match="Single-record derived"):
        runner._validated_output_paths(
            private_output=private,
            public_output=ROOT / "docs/smoke-summary.json",
            work_dir=work,
            complete_panel=False,
        )

    full_paths = runner._validated_output_paths(
        private_output=private,
        public_output=ROOT / "docs/full-summary.json",
        work_dir=work,
        complete_panel=True,
    )
    assert full_paths[1] == (ROOT / "docs/full-summary.json").resolve()
