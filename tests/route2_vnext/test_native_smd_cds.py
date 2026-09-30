from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct

from ase.build import molecule
import numpy as np
import pytest
import torch

from maple.function.route2_solvents import SMDSolventDescriptors
from maple.solvation.nonpolar.legacy_smd_cds import TorchLegacySMDCDS
import maple.solvation.nonpolar.native_smd_cds as native_cds_module
from maple.solvation.nonpolar.native_smd_cds import TorchNativeSMDCDS
from maple.solvation.nonpolar.native_smd_cds_parameters import (
    AQ_SIGMA,
    BONDI_MANTINA_ANGSTROM,
    NATIVE_LITERAL_CLASS_MANIFEST,
    PROBE_RADIUS_ANGSTROM,
    REFERENCE_DISTANCE_ANGSTROM,
    SIGMA_MOL,
    UPSTREAM_MNSOL_SHA256,
    nonaqueous_parameters,
    promote_fortran_real_literal,
)
from maple.solvation.surfaces.legacy_dareal import LegacyDAREALTopologyError

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docs/route2/evidence/route2-native-smd-cds-v2-fixtures.json"
)
FIXTURE_SHA256 = "a8c942889229dd1eef70276bab5109588673ac1c1912f2d6b1680f6ff1f24e02"
FROZEN_STEPS = {"2e-05", "1e-05", "5e-06"}
GEOMETRY_SHA256 = {
    "regular-water": "8123ad30efaef9e9a0a26ecfdfe7a746ff8dbf8409e101babe8cfde699e784ea",
    "regular-methane": "7a94bc1b3540dd283c1bf7941d7627b6147be4f8e8f61d50a2191173ad8d07e1",
    "asymmetric-ethanol": "86337a3e3a776cb8d8f9069bdbb4385c13cce6a88853bcad2eae939f9cb21802",
    "sealed-acetone10": "d5f33d507454fdd7b6f72d9ce0c2c07514cd01ce3a236efaade6e6d40cd87614",
}
BASE_CASE_IDS = {
    f"{geometry}--{solvent}"
    for geometry in GEOMETRY_SHA256
    for solvent in ("water", "ethanol", "hexane")
}
HVP_CASE_IDS = {
    f"{geometry}--{solvent}"
    for geometry in GEOMETRY_SHA256
    for solvent in ("ethanol", "hexane")
}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _validate_fixture_inventory(fixture: dict) -> None:
    assert fixture["schema"] == "route2-native-smd-cds-v2-fixtures-v1"
    assert fixture["source"] == {
        "audit_steps_angstrom": [2.0e-5, 1.0e-5, 5.0e-6],
        "direction_seed": 20260922,
        "geometry_panel_sha256": (
            "a1b4adee9471884ebac8c55e1b93fc837e95eccd65878df324e1b30ec1b904d2"
        ),
        "installed_libsolvent_sha256": (
            "6b5ac2a3e60e71c71f905c9506ddac773ae3ffa2582d0ee7c6915520d2f53bdf"
        ),
        "upstream_mnsol_sha256": UPSTREAM_MNSOL_SHA256,
    }
    cases = fixture["cases"]
    assert len(cases) == 12
    assert {record["id"] for record in cases} == BASE_CASE_IDS
    assert len({record["id"] for record in cases}) == len(cases)
    hvp_ids = {
        record["id"]
        for record in cases
        if "directional_hvp_fd_eV_angstrom2" in record["native"]
    }
    assert hvp_ids == HVP_CASE_IDS
    for record in cases:
        geometry = record["geometry"]
        assert geometry["geometry_sha256"] == GEOMETRY_SHA256[geometry["name"]]
        topology = record["topology"]
        assert set(topology["steps"]) == FROZEN_STEPS
        for row in topology["steps"].values():
            assert row["plus"] == row["minus"] == topology["base"]
        if record["id"] in HVP_CASE_IDS:
            assert set(record["native"]["directional_hvp_fd_eV_angstrom2"]) == (
                FROZEN_STEPS
            )


def _case(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.tensor(
        record["geometry"]["positions_angstrom"],
        dtype=torch.float64,
        requires_grad=True,
    )
    direction = torch.tensor(record["direction"], dtype=torch.float64)
    return coordinates, direction


def test_default_real_promotion_and_double_literal_classes_are_explicit():
    promoted = struct.unpack("!f", struct.pack("!f", 1.20))[0]
    assert promote_fortran_real_literal(1.20) == promoted
    assert BONDI_MANTINA_ANGSTROM["H"] == promoted
    assert AQ_SIGMA[6] == promote_fortran_real_literal(129.74)
    assert SIGMA_MOL == tuple(
        promote_fortran_real_literal(value) for value in (0.35, 0.0, -4.19, -6.68)
    )
    assert PROBE_RADIUS_ANGSTROM == 0.4
    assert PROBE_RADIUS_ANGSTROM != promote_fortran_real_literal(0.4)
    assert REFERENCE_DISTANCE_ANGSTROM[("C", "C")] == 1.84
    assert REFERENCE_DISTANCE_ANGSTROM[("C", "C")] != (
        promote_fortran_real_literal(1.84)
    )
    assert NATIVE_LITERAL_CLASS_MANIFEST["probe_radius"] == "binary64-D0"
    assert NATIVE_LITERAL_CLASS_MANIFEST["bondi"] == "binary32-promoted-to-binary64"
    assert len(UPSTREAM_MNSOL_SHA256) == 64


def test_fixture_hash_and_complete_inventory_are_frozen():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    _validate_fixture_inventory(_fixture())


def test_fixture_inventory_rejects_case_and_step_thinning():
    missing_case = copy.deepcopy(_fixture())
    missing_case["cases"].pop()
    with pytest.raises(AssertionError):
        _validate_fixture_inventory(missing_case)

    missing_step = copy.deepcopy(_fixture())
    missing_step["cases"][0]["topology"]["steps"].pop("2e-05")
    with pytest.raises(AssertionError):
        _validate_fixture_inventory(missing_step)

    missing_hvp_step = copy.deepcopy(_fixture())
    hvp_case = next(
        record for record in missing_hvp_step["cases"] if record["id"] in HVP_CASE_IDS
    )
    hvp_case["native"]["directional_hvp_fd_eV_angstrom2"].pop("5e-06")
    with pytest.raises(AssertionError):
        _validate_fixture_inventory(missing_hvp_step)


def test_parameter_tables_are_immutable_and_source_formula_signs_are_retained():
    with pytest.raises(TypeError):
        AQ_SIGMA[6] = 0.0
    descriptors = SMDSolventDescriptors(1.3, 1.3, 0.2, 0.3, 20.0, 2.0, 0.2, 0.3)
    parameters = nonaqueous_parameters(descriptors)
    terms = parameters.molecular_terms_cal_mol_angstrom2
    assert terms[0] > 0.0
    assert terms[1] == 0.0
    assert terms[2] < 0.0
    assert terms[3] < 0.0
    assert (
        parameters.cssigma_cal_mol_angstrom2
        == ((terms[0] + terms[1]) + terms[2]) + terms[3]
    )
    with pytest.raises(TypeError):
        parameters.sigma[6] = 0.0


def test_native_synthetic_descriptor_sign_control_matches_source_prediction():
    control = _fixture()["formula_sign_control"]
    observed = control["observed_delta_kcal_mol"]
    residual = control["observed_minus_expected_kcal_mol"]
    assert observed["positive_gamma"] > 0.0
    assert observed["negative_phi"] < 0.0
    assert observed["negative_psi"] < 0.0
    assert max(abs(value) for value in residual.values()) < 3.0e-16


def test_native_provider_is_distinct_narrow_and_import_has_no_dtype_side_effect():
    assert not issubclass(TorchNativeSMDCDS, TorchLegacySMDCDS)
    before = torch.get_default_dtype()
    functional = TorchNativeSMDCDS(("O", "H", "H"), "water")
    assert torch.get_default_dtype() is before
    assert functional.device == torch.device("cpu")
    assert len(functional.configuration_sha256()) == 64
    with pytest.raises(ValueError, match="H, C, and O"):
        TorchNativeSMDCDS(("N", "H", "H"), "water")
    with pytest.raises(ValueError, match="water, ethanol, and hexane"):
        TorchNativeSMDCDS(("O", "H", "H"), "methanol")


def test_runtime_configuration_mutation_is_detected_before_evaluation():
    functional = TorchNativeSMDCDS(("O", "H", "H"), "water")
    functional._radii[0] += 0.01
    with pytest.raises(RuntimeError, match="configuration mutated"):
        functional.configuration_sha256()
    coordinates = torch.tensor(molecule("H2O").positions, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="configuration mutated"):
        functional.evaluate_torch(coordinates)


def test_runtime_configuration_mutation_is_detected_after_kernel_entry(monkeypatch):
    functional = TorchNativeSMDCDS(("O", "H", "H"), "water")
    original = native_cds_module.legacy_dareal_areas_torch

    def mutating_kernel(*args, **kwargs):
        result = original(*args, **kwargs)
        functional._cssigma += 1.0
        return result

    monkeypatch.setattr(native_cds_module, "legacy_dareal_areas_torch", mutating_kernel)
    coordinates = torch.tensor(molecule("H2O").positions, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="configuration mutated"):
        functional.evaluate_torch(coordinates)


def test_runtime_unit_constant_drift_is_detected(monkeypatch):
    functional = TorchNativeSMDCDS(("O", "H", "H"), "water")
    monkeypatch.setattr(
        native_cds_module,
        "HARTREE_TO_EV",
        native_cds_module.HARTREE_TO_EV + 1.0e-8,
    )
    with pytest.raises(RuntimeError, match="configuration mutated"):
        functional.configuration_sha256()


def test_provenance_binds_new_and_reused_sources():
    functional = TorchNativeSMDCDS(("O", "H", "H"), "water")
    provenance = functional.provenance
    assert provenance["provider"] == "torch-native-smd-cds"
    assert (
        provenance["implementation_version"] == "torch-native-smd-cds-pyscf-2.13.1-v2"
    )
    assert provenance["upstream_sha256"] == UPSTREAM_MNSOL_SHA256
    sources = dict(json.loads(provenance["source_files_sha256"]))
    assert set(sources) == {
        "maple.function.route2_solvents",
        "maple.solvation.api.units",
        "maple.solvation.nonpolar.legacy_smd_cds.atomic_tensions",
        "maple.solvation.nonpolar.native_smd_cds",
        "maple.solvation.nonpolar.native_smd_cds_parameters",
        "maple.solvation.surfaces.legacy_dareal",
    }
    with pytest.raises(TypeError):
        provenance["provider"] = "mutated"


@pytest.mark.parametrize("record", _fixture()["cases"], ids=lambda row: row["id"])
def test_native_scalar_area_gradient_and_directional_hvp_fixture(record):
    coordinates, direction = _case(record)
    functional = TorchNativeSMDCDS(record["geometry"]["symbols"], record["solvent"])
    state = functional.evaluate_torch(coordinates)
    gradient = torch.autograd.grad(state.energy_eV, coordinates, create_graph=True)[0]
    hvp = torch.autograd.grad(torch.sum(gradient * direction), coordinates)[0]
    native = record["native"]
    area = float(state.atom_areas_angstrom2.detach().sum())
    energy_kcal = float(state.energy_kcal_mol.detach())
    energy_ev = float(state.energy_eV.detach())
    assert abs(area - native["area_angstrom2"]) <= 1.0e-10
    assert abs(energy_kcal - native["energy_kcal_mol"]) <= 1.0e-10
    assert abs(energy_ev - native["energy_eV_maple"]) <= 1.0e-12
    np.testing.assert_allclose(
        gradient.detach().numpy(),
        native["gradient_eV_angstrom"],
        rtol=0.0,
        atol=1.0e-10,
    )
    assert state.diagnostics.dareal.topology_sha256 == record["topology"]["base"]

    if "directional_hvp_fd_eV_angstrom2" not in native:
        return
    errors = {}
    for step, reference in native["directional_hvp_fd_eV_angstrom2"].items():
        errors[step] = float(
            np.max(np.abs(hvp.detach().numpy() - np.asarray(reference)))
        )
        topology = record["topology"]["steps"][step]
        assert topology["plus"] == topology["minus"] == record["topology"]["base"]
    if record["geometry"]["name"] == "asymmetric-ethanol":
        assert min(errors.values()) > 1.0e-8
        ratio = errors["1e-05"] / errors["5e-06"]
        assert 3.5 <= ratio <= 4.5
        richardson = (
            4.0 * np.asarray(native["directional_hvp_fd_eV_angstrom2"]["5e-06"])
            - np.asarray(native["directional_hvp_fd_eV_angstrom2"]["1e-05"])
        ) / 3.0
        assert float(np.max(np.abs(hvp.detach().numpy() - richardson))) <= 1.0e-8
    else:
        assert max(errors.values()) <= 1.0e-8


def test_exact_shared_point_still_fails_closed():
    atoms = molecule("CH3COCH3")
    functional = TorchNativeSMDCDS(atoms.get_chemical_symbols(), "water")
    with pytest.raises(LegacyDAREALTopologyError, match="four-sphere shared-point"):
        functional.energy_torch(torch.tensor(atoms.positions, dtype=torch.float64))
