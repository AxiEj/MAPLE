from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "docs/implicit-solvation/benchmarks"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_core as core
import run_ddx_pcm_force_probe as force_probe
import run_ddx_pcm_screen as screen

PROTOCOL_PATH = BENCHMARK_DIR / "ddx_pcm_protocol.json"
ENERGY_PATH = BENCHMARK_DIR / "route1-ddx-ddpcm-energy-screen-2026-07-25.json"
FORCE_PATH = (
    BENCHMARK_DIR / "route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json"
)
SUMMARY_PATH = BENCHMARK_DIR / "route1-ddx-ddpcm-development-summary-2026-07-25.json"


def test_ddx_protocol_preserves_route1_and_label_boundaries():
    protocol = screen.load_protocol(PROTOCOL_PATH)

    assert protocol["route1_boundary"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "formula": ("E_solution(R)=E_MLIP,gas(R)+G_polar(R,q_fixed)+G_nonpolar(R)"),
        "gas_phase_mm_energy": False,
        "hydration_label_residual": False,
        "mlip_retraining": False,
        "fixed_charge": "AM1-BCC",
    }
    assert (
        protocol["label_use_boundary"]["energy_phase_reads_experimental_labels"]
        is False
    )
    assert protocol["label_use_boundary"]["experimental_fit_or_residual"] is False
    assert (
        protocol["external_provider"]["isolated_build"]["project_dependency_added"]
        is False
    )
    assert protocol["physical_model"]["screened_model"] == "ddPCM"
    assert (
        protocol["physical_model"]["interpretation"]
        == "zero-ionic-strength ddLPB limit"
    )


def test_ddx_protocol_fails_closed_if_a_residual_is_enabled(tmp_path: Path):
    protocol = core.load_json(PROTOCOL_PATH)
    protocol["route1_boundary"]["hydration_label_residual"] = True
    path = tmp_path / "invalid-ddx-protocol.json"
    core.write_json_atomic(path, protocol)

    with pytest.raises(ValueError, match="Route 1 boundary"):
        screen.load_protocol(path)


def test_ddx_force_conversion_uses_both_native_terms_and_negates_gradient(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeModel:
        def __init__(self, _kind, centers, _radii, **_kwargs):
            self.atom_count = centers.shape[1]

        def multipole_electrostatics(self, _multipoles):
            return {"phi": np.zeros(self.atom_count)}

        def multipole_psi(self, _multipoles):
            return np.zeros(self.atom_count)

    class FakeState:
        def __init__(self, model, _psi, _phi):
            self.atom_count = model.atom_count

        def ddrun(self, _electrostatics, tol):
            assert tol == pytest.approx(1.0e-10)
            return 2.0, np.ones((3, self.atom_count))

        def multipole_force_terms(self, _multipoles):
            return np.full((3, self.atom_count), 2.0)

    class FakePyddx:
        Model = FakeModel
        State = FakeState

    monkeypatch.setattr(
        force_probe.screen,
        "_require_pyddx",
        lambda _version: FakePyddx,
    )
    energy, force = force_probe.ddpcm_energy_force_hartree_per_angstrom(
        np.zeros((2, 3)),
        np.asarray([0.2, -0.2]),
        np.asarray([1.5, 1.5]),
        lmax=3,
        n_lebedev=50,
        solvent_epsilon=78.5,
        solver_tolerance=1.0e-10,
        expected_version="0.8.0",
    )

    assert energy == pytest.approx(2.0)
    assert force == pytest.approx(np.full((2, 3), -3.0 * screen.BOHR_PER_ANGSTROM))


def test_pyddx_remains_outside_maple_runtime_dependencies_and_providers():
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runtime_dir = (
        REPOSITORY_ROOT / "maple/function/calculator/extra_correction/implicit"
    )

    assert "pyddx" not in pyproject.lower()
    for path in runtime_dir.glob("*.py"):
        assert "pyddx" not in path.read_text(encoding="utf-8").lower()


def test_frozen_ddx_energy_artifact_is_complete_and_label_free():
    protocol = screen.load_protocol(PROTOCOL_PATH)
    artifact = core.load_json(ENERGY_PATH)

    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["protocol_sha256"] == core.sha256_file(PROTOCOL_PATH)
    assert artifact["command_provenance"]["script"] == (
        "docs/implicit-solvation/benchmarks/run_ddx_pcm_screen.py"
    )
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_ddx_pcm_screen.py"
    )
    assert artifact["route1_contract"] == protocol["route1_boundary"]
    assert artifact["case_count"] == 526
    assert artifact["success_count"] == 526
    assert artifact["failure_count"] == 0
    assert len(artifact["records"]) == 526
    assert len({record["compound_id"] for record in artifact["records"]}) == 526
    assert "experimental" not in json.dumps(artifact["records"], sort_keys=True).lower()
    for record in artifact["records"]:
        assert record["status"] == "success"
        assert set(record["predictions"]) == set(screen.CANDIDATE_METHODS)
        for prediction in record["predictions"].values():
            assert prediction["total_kcal_mol"] == pytest.approx(
                prediction["polar_kcal_mol"]
                + prediction["openmm_ace_nonpolar_kcal_mol"]
            )


def test_frozen_ddx_force_artifact_covers_all_components_and_passes_derivative_gate():
    artifact = core.load_json(FORCE_PATH)

    assert core.artifact_content_sha256(artifact) == artifact["content_sha256"]
    assert artifact["protocol_sha256"] == core.sha256_file(PROTOCOL_PATH)
    assert artifact["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_ddx_pcm_force_probe.py"
    )
    assert artifact["experimental_labels_read"] is False
    assert artifact["molecule"]["atom_count"] == 23
    assert artifact["molecule"]["mol2_sha256"] == (
        "783df578c85819b79b59537197a0850e074426d2e9eef5bfef020c9a4849ce23"
    )
    assert artifact["force_gate"]["passed"] is True
    assert len(artifact["profiles"]) == 2
    for profile in artifact["profiles"]:
        assert set(profile["finite_difference"]) == {"0.003", "0.010"}
        for result in profile["finite_difference"].values():
            assert result["component_count"] == 69
            assert len(result["components"]) == 69
        primary = profile["finite_difference"]["0.003"]
        assert primary["rmse_kj_mol_angstrom"] < 0.001
        assert primary["maximum_absolute_error_kj_mol_angstrom"] < 0.003
        assert profile["net_force_norm_kj_mol_angstrom"] < 1.0e-10

    vdw = artifact["profiles"][0]
    assert vdw["name"] == "ddpcm_vdw_mbondi2_l7_n194"
    assert vdw["candidate"]["energy_kj_mol"] == pytest.approx(-31.33367301565967)
    assert vdw["finite_difference"]["0.003"]["rmse_kj_mol_angstrom"] == pytest.approx(
        0.0004044021348176307, abs=1.0e-12
    )


def test_frozen_ddx_convergence_and_performance_close_the_product_gate():
    artifact = core.load_json(FORCE_PATH)
    convergence = artifact["numerical_convergence_gate"]
    performance = artifact["performance_comparison"]

    assert convergence["experimental_labels_read"] is False
    assert convergence["case_count"] == 20
    assert convergence["passed"] is True
    assert convergence["summary"]["p90_absolute_kcal_mol"] == pytest.approx(
        0.13124795202570988
    )
    assert convergence["summary"]["maximum_absolute_kcal_mol"] == pytest.approx(
        0.18204954833719503
    )
    assert performance["ddpcm_vdw_force_vs_openmm_obc2_ace_correction_ratio"] > 300.0
    assert performance["ddpcm_vdw_force_vs_maceoff23m_gas_gpu_ratio"] > 6.0
    assert performance["faster_than_bare_mm_claim_allowed"] is False
    assert artifact["decision"]["runtime_provider_added"] is False
    assert artifact["decision"]["project_dependency_added"] is False


def test_frozen_ddx_development_summary_rejects_promotion_without_a_residual():
    summary = core.load_json(SUMMARY_PATH)

    assert core.artifact_content_sha256(summary) == summary["content_sha256"]
    assert summary["protocol_sha256"] == core.sha256_file(PROTOCOL_PATH)
    assert summary["energy_artifact_sha256"] == core.sha256_file(ENERGY_PATH)
    assert summary["force_artifact_sha256"] == core.sha256_file(FORCE_PATH)
    assert summary["command_provenance"]["script_sha256"] == core.sha256_file(
        BENCHMARK_DIR / "run_ddx_pcm_screen.py"
    )
    assert summary["case_count"] == 526
    assert summary["label_use_boundary"]["experimental_fit_or_residual"] is False
    assert summary["methods"]["am1bcc_obc2_ace"]["mae"] == pytest.approx(
        1.7603512076636891
    )
    assert summary["methods"]["ddpcm_vdw_mbondi2_l7_n194"]["mae"] == pytest.approx(
        1.7824286182381819
    )
    assert summary["methods"]["ddpcm_vdw_mbondi2_l7_n194"]["rmse"] == pytest.approx(
        2.8808346770440867
    )
    assert summary["methods"]["ddpcm_vdw_mbondi2_l7_n194"][
        "max_absolute_error"
    ] == pytest.approx(18.813418254059094)
    gain = summary["paired_absolute_error_gain"]["ddpcm_vdw_mbondi2_l7_n194"][
        "am1bcc_obc2_ace"
    ]
    assert gain["mean_mae_gain_kcal_mol"] == pytest.approx(-0.022077410574492782)
    assert gain["bootstrap_ci"][0] < 0.0 < gain["bootstrap_ci"][1]
    assert summary["product_gates"] == {
        "numerical_convergence_gate_passed": True,
        "complete_polar_force_gate_passed": True,
        "accuracy_materiality_gate_passed": False,
        "performance_gate_passed": False,
        "independent_confirmation_passed": False,
        "product_admission_passed": False,
    }
    assert summary["decision"]["disposition"] == "reject-product-provider"
    assert summary["decision"]["runtime_provider_added"] is False
    assert summary["decision"]["project_dependency_added"] is False
    assert summary["decision"]["production_default_changed"] is False


def test_route1_docs_keep_ddx_as_a_rejected_external_reference_candidate():
    paths = (
        REPOSITORY_ROOT / "docs/implicit-solvation/ROUTE1_PRODUCT_SPEC.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/README.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/VALIDATION_STATUS.md",
        REPOSITORY_ROOT / "docs/implicit-solvation/FORMULAS_AND_REFERENCES.md",
        BENCHMARK_DIR / "README.md",
    )
    normalized = " ".join(
        " ".join(path.read_text(encoding="utf-8").split()) for path in paths
    )

    assert "ddX/ddPCM conservative-provider audit" in normalized
    assert "route1-ddx-ddpcm-energy-screen-2026-07-25.json" in normalized
    assert "route1-ddx-ddpcm-force-probe-methyl-hexanoate-2026-07-25.json" in normalized
    assert "0.000404/0.001504 kJ/mol/A" in normalized
    assert "1.782/2.881" in normalized
    assert "roughly `360x` slower" in normalized
    assert "no `pyddx` dependency or MAPLE provider is introduced" in normalized


@pytest.mark.parametrize("path", [ENERGY_PATH, FORCE_PATH, SUMMARY_PATH])
def test_ddx_artifact_files_have_stable_sha256_shape(path: Path):
    assert len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64
