"""Complete-catalog contracts for the pure-Torch V4 multisolvent route."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_coulomb_radii,
)
from maple.function.route2_smd_profiles import (
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
    PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    route2_smd_profile_spec,
)
from maple.function.route2_solvents import SMD_SOLVENTS, SUPPORTED_SMD_SOLVENTS
from maple.solvation.continuum.torch_ddpcm import TorchDDPCM
from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS

SYMBOLS = ("O", "H", "H")
POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239, 0.9266, 0.0]],
    dtype=np.float32,
)
ROOT = Path(__file__).resolve().parents[2]
MULTISOLVENT_EVIDENCE = (
    ROOT / "docs/route2/evidence/"
    "route2-pure-macepolar-torch-fp32-v4-multisolvent-component-smoke-v1.json"
)
FORMER_TOPOLOGY_REJECTIONS = {
    "1-hexyne",
    "2-butanol",
    "2-methyl-2-propanol",
    "2-propanol",
    "acetonitrile",
    "aniline",
    "benzylalcohol",
    "cyclopentanol",
    "z-1-2-dichloroethene",
    "n-methylformamide-e-zmixture",
}


def test_v4_profiles_bind_the_complete_named_smd_catalog():
    assert len(SMD_SOLVENTS) == 179
    assert SUPPORTED_SMD_SOLVENTS == frozenset(SMD_SOLVENTS)
    for profile in (
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CPU_V4_PROFILE,
        PURE_MACEPOLAR_FROZEN_POINT_L1_DDPCM_SMD_TORCH_CUDA_V4_PROFILE,
    ):
        assert route2_smd_profile_spec(profile).supported_solvents == (
            SUPPORTED_SMD_SOLVENTS
        )


@pytest.mark.filterwarnings("ignore:ddPCM FP32 topology.*:RuntimeWarning")
def test_every_catalog_solvent_constructs_strict_fp32_components_on_cpu():
    positions = torch.tensor(POSITIONS, dtype=torch.float32, requires_grad=True)
    source = torch.zeros((3, 8), dtype=torch.float32)
    source[:, 0] = torch.tensor([-0.8, 0.4, 0.4], dtype=torch.float32)
    warned = set()
    for name, specification in SMD_SOLVENTS.items():
        continuum = TorchDDPCM(
            SYMBOLS,
            smd_coulomb_radii(SYMBOLS, solvent=name),
            dielectric=specification.descriptors.dielectric,
            lmax=5,
            n_lebedev=194,
            dtype=torch.float32,
            device="cpu",
        )
        cds = TorchLegacySMDCDS(SYMBOLS, name, dtype=torch.float32, device="cpu")
        diagnostics = continuum.diagnostics(positions, source)
        assert diagnostics["assessment_gate_pass"]
        assert diagnostics["assessment_gate_failures"] == []
        assert diagnostics["fp32_partition"]["hard_failures"] == ()
        if diagnostics["topology_warnings"]:
            warned.add(name)
        energy = continuum.energy_torch(positions, source) + cds.energy_torch(positions)
        gradient = torch.autograd.grad(energy, positions)[0]
        assert energy.dtype is torch.float32
        assert bool(torch.isfinite(energy))
        assert bool(torch.isfinite(gradient).all())
        assert continuum.dielectric == specification.descriptors.dielectric
        assert continuum.lmax == 5 and continuum.n_lebedev == 194
        assert cds.provenance["solvent"] == name
        positions = positions.detach().requires_grad_(True)
    assert FORMER_TOPOLOGY_REJECTIONS <= warned


@pytest.mark.parametrize(
    "solvent",
    [
        "water",
        "methanol",
        "acetonitrile",
        "acetone",
        "dimethylsulfoxide",
        "benzene",
        "chloroform",
        "hexane",
    ],
)
def test_v4_builder_propagates_catalog_descriptors_without_model_load(
    monkeypatch, solvent
):
    import maple.solvation.continuum.torch_ddpcm as continuum_module
    import maple.solvation.experimental.mace_polar_torch as builder_module
    import maple.solvation.models.mace_polar_fp32 as model_module
    import maple.solvation.nonpolar.legacy_smd_cds as cds_module

    captured = {}

    def continuum(symbols, radii, **kwargs):
        captured["continuum"] = (tuple(symbols), np.asarray(radii), kwargs)
        return object()

    def cds(symbols, selected_solvent, **kwargs):
        captured["cds"] = (tuple(symbols), selected_solvent, kwargs)
        return object()

    monkeypatch.setattr(continuum_module, "TorchDDPCM", continuum)
    monkeypatch.setattr(cds_module, "TorchLegacySMDCDS", cds)
    monkeypatch.setattr(
        model_module,
        "build_official_mace_polar_fp32_adapter",
        lambda **kwargs: ("model", kwargs),
    )
    monkeypatch.setattr(builder_module, "MACEPolarTorchPES", lambda **kwargs: kwargs)

    result = builder_module.build_smd_mace_polar_torch_pes(
        SYMBOLS, solvent=solvent, device="cpu"
    )
    specification = SMD_SOLVENTS[solvent]
    _, radii, continuum_kwargs = captured["continuum"]
    assert continuum_kwargs["dielectric"] == specification.descriptors.dielectric
    assert continuum_kwargs["dtype"] is torch.float32
    assert continuum_kwargs["lmax"] == 5
    assert continuum_kwargs["n_lebedev"] == 194
    np.testing.assert_allclose(radii, smd_coulomb_radii(SYMBOLS, solvent=solvent))
    assert captured["cds"] == (
        SYMBOLS,
        solvent,
        {"device": "cpu", "dtype": torch.float32},
    )
    assert result["solvent"] == solvent
    assert result["requested_profile"].endswith("torch-cpu-v4")


def test_smoke_hvp_selection_is_catalog_bound_and_deterministic():
    from tools.route2_release.run_pure_mace_polar_multisolvent_smoke import (
        _representative_hvp_solvents,
    )

    names = tuple(sorted(SMD_SOLVENTS))
    selected = _representative_hvp_solvents(names, 8)
    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert set(selected) <= set(names)
    assert selected == _representative_hvp_solvents(names, 8)


def test_published_multisolvent_artifact_reports_the_complete_population():
    from tools.route2_release.run_pure_mace_polar_multisolvent_smoke import (
        ARTIFACT_ID,
        _json_sha256,
        _source_manifest,
    )

    record = json.loads(MULTISOLVENT_EVIDENCE.read_text())
    assert record["artifact_id"] == ARTIFACT_ID
    assert record["catalog_cardinality"] == 179
    assert record["catalog_solvents"] == sorted(SMD_SOLVENTS)
    assert record["catalog_identity_sha256"] == _json_sha256(
        {
            name: {
                "pyscf_smd_name": spec.pyscf_smd_name,
                "mnsol_name": spec.mnsol_name,
                "descriptors": spec.descriptors.as_pyscf_tuple(),
            }
            for name, spec in SMD_SOLVENTS.items()
        }
    )
    assert record["source_unchanged"] is True
    assert record["source_manifest"] == _source_manifest()
    assert record["dataset_labels_used"] is False
    assert record["fit_performed"] is False
    assert record["new_fitted_parameters_introduced"] is False
    assert record["scientific_release_admitted"] is False
    assert record["experimental_solvation_accuracy_admitted"] is False

    assert record["runtime"]["devices_requested"] == ["cpu", "cuda:0"]
    assert record["summary"] == {
        "expected_cases": 358,
        "recorded_cases": 358,
        "successes": 358,
        "failures": 0,
        "complete_population_reported": True,
    }
    assert len(record["cases"]) == 358
    assert len(record["devices"]) == 2

    for device_record in record["devices"]:
        device = device_record["device"]
        device_cases = [case for case in record["cases"] if case["device"] == device]
        successful = [case for case in device_cases if case["status"] == "success"]
        failed = [case for case in device_cases if case["status"] == "failure"]
        assert len(device_cases) == 179
        assert {case["solvent"] for case in device_cases} == set(SMD_SOLVENTS)
        assert len(successful) == 179
        for case in successful:
            spec = SMD_SOLVENTS[case["solvent"]]
            assert case["descriptors"] == list(spec.descriptors.as_pyscf_tuple())
            assert case["dielectric"] == spec.descriptors.dielectric
        assert not failed
        by_name = {case["solvent"]: case for case in successful}
        for name in FORMER_TOPOLOGY_REJECTIONS:
            assert by_name[name]["topology_warning_codes"]
            assert by_name[name]["assessment_gate_failures"] == []
            assert by_name[name]["fp32_partition"]["hard_failures"] == []
        assert device_record["model_constructed_once"] is True
        assert device_record["model_forward_count"] == 1
        assert device_record["frozen_source_reused_across_solvents"] is True
        assert len(device_record["source_sha256"]) == 64
        assert len(device_record["model_configuration_sha256"]) == 64
        assert (
            abs(device_record["learned_source_charge_e"])
            <= device_record["learned_source_charge_atol_e"]
        )
        assert {case["source_sha256"] for case in successful} == {
            device_record["source_sha256"]
        }
