from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase.units import Hartree

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from mnsol_response_ablation import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    aggregate_method_metrics,
    compose_method_ledger,
    paired_method_comparison,
    solve_fixed_multipole_continuum,
)
from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (  # noqa: E402
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.calculator_base import EV2HARTREE  # noqa: E402
from maple.function.calculator.extra_correction.implicit.smd_cds import (  # noqa: E402
    HARTREE_TO_KCAL_MOL,
)
import run_mnsol_macepolar_response_ablation as response_runner  # pyright: ignore[reportMissingImports]  # noqa: E402


def _record(experimental: float, left_total: float, right_total: float):
    def _method(total: float):
        return compose_method_ledger(
            experimental_kcal_mol=experimental,
            solute_polarization_kcal_mol=1.0,
            continuum_polarization_kcal_mol=total - 1.5,
            smd_cds_kcal_mol=0.5,
            wall_seconds=2.0,
        )

    return {
        "experimental_delta_g_kcal_mol": experimental,
        "methods": {
            "left": _method(left_total),
            "right": _method(right_total),
        },
    }


def test_response_ablation_ledger_closes_components_and_error():
    result = compose_method_ledger(
        experimental_kcal_mol=-4.0,
        solute_polarization_kcal_mol=1.25,
        continuum_polarization_kcal_mol=-5.0,
        smd_cds_kcal_mol=-0.75,
        wall_seconds=3.5,
        extra={"identity_error_ev": 2.0e-14},
    )

    assert result["electrostatic_kcal_mol"] == pytest.approx(-3.75)
    assert result["total_solvation_kcal_mol"] == pytest.approx(-4.5)
    assert result["signed_error_kcal_mol"] == pytest.approx(-0.5)
    assert result["absolute_error_kcal_mol"] == pytest.approx(0.5)
    assert result["identity_error_ev"] == pytest.approx(2.0e-14)


def test_response_ablation_metrics_and_paired_counts():
    records = [
        _record(-4.0, -4.5, -4.1),
        _record(-2.0, -1.0, -2.3),
    ]

    metrics = aggregate_method_metrics(records, "right")
    paired = paired_method_comparison(
        records,
        left="left",
        right="right",
    )

    assert metrics["record_count"] == 2
    assert metrics["mean_signed_error_kcal_mol"] == pytest.approx(-0.2)
    assert metrics["mean_absolute_error_kcal_mol"] == pytest.approx(0.2)
    assert metrics["root_mean_square_error_kcal_mol"] == pytest.approx(
        (0.1**2 + 0.3**2) ** 0.5 / 2**0.5
    )
    assert metrics["total_wall_seconds"] == pytest.approx(4.0)
    assert paired["right_minus_left_mean_energy_kcal_mol"] == pytest.approx(-0.45)
    assert paired["right_lower_absolute_error_count"] == 2
    assert paired["left_lower_absolute_error_count"] == 0


def test_response_ablation_rejects_nonfinite_and_field_replacement():
    with pytest.raises(ValueError, match="must be finite"):
        compose_method_ledger(
            experimental_kcal_mol=float("nan"),
            solute_polarization_kcal_mol=0.0,
            continuum_polarization_kcal_mol=-1.0,
            smd_cds_kcal_mol=0.0,
            wall_seconds=1.0,
        )
    with pytest.raises(ValueError, match="cannot replace"):
        compose_method_ledger(
            experimental_kcal_mol=-1.0,
            solute_polarization_kcal_mol=0.0,
            continuum_polarization_kcal_mol=-1.0,
            smd_cds_kcal_mol=0.0,
            wall_seconds=1.0,
            extra={"signed_error_kcal_mol": 0.0},
        )


def test_fixed_multipole_solve_checks_charge_and_half_coupling():
    coefficients = np.asarray(
        [
            [0.2, 0.3, -0.4, 0.1],
            [-0.2, -0.1, 0.2, 0.5],
        ],
        dtype=float,
    )
    field = np.asarray(
        [
            [-1.0, 0.2, -0.3, 0.4],
            [0.5, -0.6, 0.7, -0.8],
        ],
        dtype=float,
    )

    class _ReactionField:
        def apply_scf(self, values):
            assert np.array_equal(values, coefficients)
            return field

        def scf_polarization_energy_hartree(self, values):
            return 0.5 * MACE_POLAR_L1_PAIRING.pair(values, field) / Hartree

    state = solve_fixed_multipole_continuum(
        _ReactionField(),
        coefficients,
    )

    assert state["reaction_field_values_ev"] == pytest.approx(field)
    assert state["energy_identity_error_ev"] == pytest.approx(0.0)
    assert state["observed_total_charge_e"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="declared total charge"):
        solve_fixed_multipole_continuum(
            _ReactionField(),
            coefficients + np.asarray([[0.1, 0, 0, 0], [0, 0, 0, 0]]),
        )


def test_response_ablation_uses_maple_energy_conversion_contract():
    assert response_runner.EV_TO_KCAL_MOL == pytest.approx(
        EV2HARTREE * HARTREE_TO_KCAL_MOL,
        rel=0.0,
        abs=0.0,
    )


def test_response_ablation_continuum_arm_is_explicit_and_defaults_to_ddpcm():
    parser = response_runner._build_parser()
    default_args = parser.parse_args(
        [
            "--source",
            "source.zip",
            "--protocol",
            "protocol.json",
            "--selection",
            "selection.json",
            "--aimnet2-checkpoint",
            "aimnet2.pt",
            "--private-output",
            ".omx/private.json",
            "--public-output",
            ".omx/public.json",
            "--work-dir",
            ".omx/work",
        ]
    )

    assert default_args.continuum_equation == "ddpcm"
    assert response_runner._continuum_arm("ddpcm").profile == (
        response_runner.DDPCM_MULTISOLVENT_SMD_PROFILE
    )
    assert response_runner._continuum_arm("ddcosmo").profile == (
        response_runner.DDCOSMO_MULTISOLVENT_SMD_PROFILE
    )
    assert (
        response_runner._continuum_arm("ddcosmo").reaction_field_type
        is response_runner.PyDDXCOSMOReactionFieldLinearMap
    )

    with pytest.raises(ValueError, match="Unsupported continuum equation"):
        response_runner._continuum_arm("cpcm")


def test_response_ablation_reaction_field_uses_selected_profile_and_backend(
    monkeypatch,
):
    captured = {}
    expected_radii = np.asarray([1.2, 1.85], dtype=float)

    class _Atoms:
        def get_chemical_symbols(self):
            return ["H", "C"]

        def get_positions(self):
            return np.zeros((2, 3), dtype=float)

    def _radii(symbols, *, solvent, profile):
        captured["symbols"] = symbols
        captured["solvent"] = solvent
        captured["profile"] = profile
        return expected_radii

    def _reaction(positions, radii, **kwargs):
        captured["positions"] = positions
        captured["radii"] = radii
        captured["reaction_kwargs"] = kwargs
        return "reaction"

    monkeypatch.setattr(response_runner, "route2_coulomb_radii", _radii)
    monkeypatch.setattr(
        response_runner,
        "route2_solvent_spec",
        lambda solvent: SimpleNamespace(
            descriptors=SimpleNamespace(dielectric=12.5)
        ),
    )
    arm = response_runner.ContinuumArm(
        equation="ddcosmo",
        display_name="ddCOSMO",
        profile="selected-profile",
        reaction_field_type=_reaction,
    )

    reaction, radii = response_runner._reaction_field(
        _Atoms(),
        "toluene",
        arm,
    )

    assert reaction == "reaction"
    assert radii is expected_radii
    assert captured["profile"] == "selected-profile"
    assert captured["solvent"] == "toluene"
    assert captured["reaction_kwargs"]["dielectric"] == pytest.approx(12.5)


def test_response_ablation_mace_bootstrap_profile_drift_fails_closed(
    monkeypatch,
):
    arm = response_runner._continuum_arm("ddcosmo")

    class _Runtime:
        @staticmethod
        def _settings(solvent, profile):
            return {
                "model": "mace",
                "d4": False,
                "model_options": None,
                "charge": {},
                "solv": {"profile": profile},
            }

    class _CompatibleMACE:
        @staticmethod
        def build_implicit_solvent_kwargs(solvation_options):
            return {"long_range_evaluator_profile": "shared"}

    monkeypatch.setattr(
        response_runner,
        "MACEPolCalculator",
        _CompatibleMACE,
    )
    response_runner._validate_mace_bootstrap_profile_compatibility(
        _Runtime,
        solvent="water",
        continuum_arm=arm,
    )

    class _DriftedMACE:
        @staticmethod
        def build_implicit_solvent_kwargs(solvation_options):
            return {
                "long_range_evaluator_profile": solvation_options["profile"]
            }

    monkeypatch.setattr(response_runner, "MACEPolCalculator", _DriftedMACE)
    with pytest.raises(RuntimeError, match="cannot isolate"):
        response_runner._validate_mace_bootstrap_profile_compatibility(
            _Runtime,
            solvent="water",
            continuum_arm=arm,
        )


def test_response_ablation_single_record_outputs_remain_private():
    private = REPOSITORY_ROOT / ".omx/response-ablation/private.json"
    public = REPOSITORY_ROOT / ".omx/response-ablation/summary.json"
    work = REPOSITORY_ROOT / ".omx/response-ablation/work"

    assert response_runner._validated_output_paths(
        private_output=private,
        public_output=public,
        work_dir=work,
        complete_panel=False,
    ) == (private.resolve(), public.resolve(), work.resolve())

    with pytest.raises(ValueError, match="Single-record response-ablation"):
        response_runner._validated_output_paths(
            private_output=private,
            public_output=REPOSITORY_ROOT / "docs/single-record.json",
            work_dir=work,
            complete_panel=False,
        )


def test_response_ablation_shard_disclosure_and_indices_are_explicit():
    assert response_runner._row_level_data_emitted(complete_panel=False) is True
    assert response_runner._row_level_data_emitted(complete_panel=True) is False
    assert response_runner._selection_indices(
        [{"selection_index": 2}, {"selection_index": 17}]
    ) == [2, 17]

    with pytest.raises(TypeError, match="Selection indices"):
        response_runner._selection_indices([{"selection_index": True}])


def test_response_ablation_stage_bound_is_private_and_method_complete():
    assert response_runner._validated_evaluated_methods(
        maximum_response_stage="one-shot",
        partition_shard=True,
    ) == response_runner.ABLATION_METHODS[:-1]
    assert response_runner._validated_evaluated_methods(
        maximum_response_stage="scf",
        partition_shard=False,
    ) == response_runner.ABLATION_METHODS

    with pytest.raises(ValueError, match="private frozen MNSol partition"):
        response_runner._validated_evaluated_methods(
            maximum_response_stage="one-shot",
            partition_shard=False,
        )


def _scf_result(*, gas_energy_ev: float = -10.0) -> dict[str, object]:
    return {
        "gas_energy_hartree": gas_energy_ev * response_runner.EV2HARTREE,
        "smd_cds_energy_kcal_mol": 0.4,
        "solute_polarization_kcal_mol": 0.2,
        "continuum_polarization_kcal_mol": -1.1,
        "timing_seconds": {"public_energy": 3.0},
        "scf_iterations": 7,
        "unmixed_density_residual_inf_e": 1.0e-13,
        "half_coupling_identity_error_ev": 2.0e-14,
    }


def test_response_ablation_audits_provider_failure_before_reraising():
    failures: list[Exception] = []

    def _provider_failure():
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        response_runner._run_audited_scf_stage(
            evaluator=_provider_failure,
            on_failure=failures.append,
            gas_energy_ev=-10.0,
            cds_energy_kcal_mol=0.4,
            experimental_kcal_mol=-0.5,
        )

    assert len(failures) == 1
    assert str(failures[0]) == "provider failed"


def test_response_ablation_audits_returned_scf_identity_failure():
    failures: list[Exception] = []
    inconsistent = _scf_result()
    inconsistent["gas_energy_hartree"] = 0.0

    with pytest.raises(RuntimeError, match="MACE gas energies"):
        response_runner._run_audited_scf_stage(
            evaluator=lambda: inconsistent,
            on_failure=failures.append,
            gas_energy_ev=-10.0,
            cds_energy_kcal_mol=0.4,
            experimental_kcal_mol=-0.5,
        )

    assert len(failures) == 1
    assert "MACE gas energies" in str(failures[0])
