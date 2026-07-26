from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import MOL2Reader

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openmm") is None,
    reason="OpenMM optional dependency is not installed",
)


@pytest.mark.parametrize("model", ["hct", "obc1", "obc2", "gbn", "gbn2"])
def test_all_five_amber_gb_models_return_finite_energy_and_force(water_mol2, model):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(atoms, atoms.get_initial_charges(), model=model, nonpolar="ace")
    result = provider.evaluate(atoms, need_forces=True)

    assert np.isfinite(result.energy_hartree)
    assert result.forces_hartree_per_angstrom.shape == (3, 3)
    assert np.isfinite(result.forces_hartree_per_angstrom).all()
    assert np.isclose(
        result.components_hartree["polar"] + result.components_hartree["nonpolar"],
        result.energy_hartree,
        atol=1e-10,
    )


@pytest.mark.parametrize("model", ["hct", "obc1", "obc2", "gbn", "gbn2"])
def test_ace_derivative_components_match_an_independent_polar_context(
    water_mol2,
    model,
):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    charges = atoms.get_initial_charges()
    total = OpenMMGB(atoms, charges, model=model, nonpolar="ace").evaluate(atoms)
    polar = OpenMMGB(atoms, charges, model=model, nonpolar="none").evaluate(atoms)

    assert total.components_hartree["polar"] == pytest.approx(
        polar.energy_hartree,
        abs=2.0e-10,
    )
    assert total.components_hartree["nonpolar"] == pytest.approx(
        total.energy_hartree - polar.energy_hartree,
        abs=2.0e-10,
    )


def test_ace_term_discovery_contract_is_versioned_and_explicit(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import (
        NONPOLAR_SCALE_PARAMETER,
        OpenMMGB,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="ace"
    )
    force = provider._total.force

    assert provider.provenance["provider_version"] == importlib.metadata.version(
        "openmm"
    )
    compatibility = provider.provenance["private_api_compatibility"]
    assert compatibility["verified_openmm_version"] == "8.5.2"
    assert compatibility["observed_openmm_version"] == "8.5.2"
    assert compatibility["unverified_versions_fail_closed"] is True
    assert [
        force.getGlobalParameterName(index)
        for index in range(force.getNumGlobalParameters())
    ] == [NONPOLAR_SCALE_PARAMETER]
    assert force.getGlobalParameterDefaultValue(0) == 1.0
    assert [
        force.getEnergyParameterDerivativeName(index)
        for index in range(force.getNumEnergyParameterDerivatives())
    ] == [NONPOLAR_SCALE_PARAMETER]
    tagged_terms = [
        str(force.getEnergyTermParameters(index)[0])
        for index in range(force.getNumEnergyTerms())
        if str(force.getEnergyTermParameters(index)[0]).startswith(
            f"{NONPOLAR_SCALE_PARAMETER}*("
        )
    ]
    assert len(tagged_terms) == 1


def test_private_openmm_adapter_rejects_unverified_versions(monkeypatch):
    from maple.function.calculator.extra_correction.implicit import openmm_compat

    monkeypatch.setattr(openmm_compat, "openmm_version", lambda: "8.6.0")
    with pytest.raises(ImportError, match="verified only against OpenMM 8.5.2"):
        openmm_compat.customgbforces_module()


def test_ace_term_discovery_fails_closed_when_upstream_adds_multiple_terms():
    from maple.function.calculator.extra_correction.implicit.openmm_gb import (
        _tag_unique_added_energy_term,
    )

    class FakeForce:
        def __init__(self, terms):
            self.terms = list(terms)

        def getNumEnergyTerms(self):
            return len(self.terms)

        def getEnergyTermParameters(self, index):
            return self.terms[index]

    baseline = FakeForce([("polar", 0)])
    ambiguous = FakeForce([("polar", 0), ("ace-1", 0), ("ace-2", 0)])

    with pytest.raises(RuntimeError, match="exactly one added energy term"):
        _tag_unique_added_energy_term(
            ambiguous,
            baseline,
            "maple_nonpolar_scale",
        )


def test_openmm_gb_force_matches_finite_difference(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="ace"
    )
    result = provider.evaluate(atoms, need_forces=True)
    h = 1.0e-2
    plus = atoms.copy()
    plus.positions[1, 0] += h
    minus = atoms.copy()
    minus.positions[1, 0] -= h
    fd_force = -(
        provider.evaluate(plus).energy_hartree - provider.evaluate(minus).energy_hartree
    ) / (2 * h)
    assert np.isclose(result.forces_hartree_per_angstrom[1, 0], fd_force, atol=2e-6)


def test_ace_component_uses_one_total_context_evaluation(water_mol2, monkeypatch):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="ace"
    )

    def reject_duplicate_polar_evaluation(*_args, **_kwargs):
        raise AssertionError("ACE decomposition should not evaluate the polar context.")

    monkeypatch.setattr(provider._polar, "evaluate", reject_duplicate_polar_evaluation)
    result = provider.evaluate(atoms, need_forces=True)

    assert np.isfinite(result.energy_hartree)
    assert np.isfinite(result.forces_hartree_per_angstrom).all()
    assert result.components_hartree["nonpolar"] > 0.0
    assert np.isclose(
        result.components_hartree["polar"] + result.components_hartree["nonpolar"],
        result.energy_hartree,
        atol=1.0e-12,
    )


def test_openmm_gb_exposes_first_class_radius_and_nonpolar_providers(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="ace"
    )

    assert provider.radius_provider.name == "openmm-amber-gb-radii"
    assert provider.radius_result.profile == "obc2-mbondi2"
    assert provider.radius_result.radii_angstrom.shape == (len(atoms),)
    assert provider.radius_result.provider_parameters.shape[0] == len(atoms)
    assert np.all(provider.radius_result.radii_angstrom > 0.0)
    assert provider.nonpolar_provider.name == "openmm-ace"
    assert provider.nonpolar_provider.component_properties == frozenset(
        {"energy", "forces"}
    )
    assert provider.provenance["component_decomposition"] == (
        "single-context OpenMM energy-parameter derivative"
    )
    assert provider.provenance["energy_force_evaluations_per_call"] == 1
    assert provider.provenance["radius_provider"]["profile"] == "obc2-mbondi2"
    assert provider.provenance["nonpolar_provider"]["name"] == "openmm-ace"


def test_fixed_charge_runtime_manifest_freezes_route1_identity(water_mol2, tmp_path):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "method": "gb",
            "model": "obc2",
            "nonpolar": "ace",
            "experimental": True,
        },
        output=tmp_path / "job.out",
    )
    manifest = json.loads(
        (correction.audit_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["route"] == {
        "name": "Additive fixed-charge PB/GB implicit solvation",
        "role": "Baseline/Product Route",
        "formula": (
            "E_solution(R) = E_MLIP,gas(R) + " "G_polar(R,q_fixed) + G_nonpolar(R)"
        ),
        "fixed_charge": True,
        "product_contract": True,
        "prohibited_terms": {
            "gas_phase_mm_energy": False,
            "retraining": False,
            "hydration_label_residual": False,
        },
    }
    assert manifest["energy_composition"] == manifest["route"]["formula"]
    assert manifest["solvation"]["platform"] == "CPU"
    assert manifest["solvation"]["platform_properties"] == {
        "DeterministicForces": "true",
        "Threads": "1",
    }


def test_fixed_charge_correction_rejects_atom_identity_or_topology_drift(
    water_mol2,
    tmp_path,
):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "method": "gb",
            "model": "obc2",
            "nonpolar": "ace",
            "experimental": True,
        },
        output=tmp_path / "identity.out",
    )

    changed_element = atoms.copy()
    changed_element.numbers[0] = 9
    with pytest.raises(ValueError, match="frozen atom identity"):
        correction.evaluate(changed_element)

    reordered = atoms[[1, 0, 2]]
    with pytest.raises(ValueError, match="frozen atom identity"):
        correction.evaluate(reordered)

    shortened = atoms[:-1]
    with pytest.raises(ValueError, match="frozen atom identity"):
        correction.evaluate(shortened)

    changed_topology = atoms.copy()
    changed_topology.info["mol2"] = json.loads(
        json.dumps(changed_topology.info["mol2"])
    )
    changed_topology.info["mol2"]["bonds"][0][2] = "2"
    with pytest.raises(ValueError, match="frozen MOL2 topology"):
        correction.evaluate(changed_topology)


def test_fixed_charge_correction_rejects_same_element_atom_reordering(tmp_path):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    path = Path(__file__).parent / "data/amber_gb_reference/mol2/mobley_1017962.mol2"
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "method": "gb",
            "model": "obc2",
            "nonpolar": "ace",
            "experimental": True,
        },
        output=tmp_path / "same-element-identity.out",
    )
    atom_types = atoms.info["mol2"]["atom_types"]
    first, second = next(
        (first, second)
        for first in range(len(atoms))
        for second in range(first + 1, len(atoms))
        if atoms.numbers[first] == atoms.numbers[second]
        and atom_types[first] != atom_types[second]
    )
    permutation = list(range(len(atoms)))
    permutation[first], permutation[second] = permutation[second], permutation[first]
    reordered = atoms[permutation]

    assert tuple(reordered.numbers) == tuple(atoms.numbers)
    assert reordered.info["mol2"] == atoms.info["mol2"]
    with pytest.raises(ValueError, match="frozen atom identity"):
        correction.evaluate(reordered)


def test_fixed_charge_correction_rejects_atom_reordering_before_preparation(
    water_mol2,
    tmp_path,
):
    from maple.function.calculator.extra_correction.implicit.correction import (
        ImplicitSolvationCorrection,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    reordered = atoms[[1, 0, 2]]

    with pytest.raises(ValueError, match="source MOL2 atom IDs"):
        ImplicitSolvationCorrection(
            reordered,
            {"source": "mol2", "mode": "fixed", "geometry": "keep"},
            {
                "method": "gb",
                "model": "obc2",
                "nonpolar": "ace",
                "experimental": True,
            },
            output=tmp_path / "preparation-order.out",
        )


def test_disabled_nonpolar_provider_declares_no_component_properties():
    from maple.function.calculator.extra_correction.implicit.nonpolar import (
        OpenMMNonpolarProvider,
    )

    provider = OpenMMNonpolarProvider("none")

    assert provider.component_properties == frozenset()
    assert provider.provenance["component_properties"] == []


def test_mol2_name_cannot_trigger_biomolecular_residue_typing(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import (
        build_openmm_topology,
    )

    text = water_mol2.read_text().replace("WATER\n", "DA\n", 1)
    water_mol2.write_text(text)
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    topology = build_openmm_topology(atoms)
    assert [residue.name for residue in topology.residues()] == ["MOL"]


def test_openmm_platform_selection_is_case_insensitive(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar="ace",
        platform="reference",
    )
    assert np.isfinite(provider.evaluate(atoms).energy_hartree)


def test_openmm_product_default_is_single_thread_deterministic_cpu(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar="ace",
    )

    assert provider.provenance["platform"] == "CPU"
    assert provider.provenance["platform_properties"] == {
        "DeterministicForces": "true",
        "Threads": "1",
    }


def test_explicit_reference_platform_remains_property_free_control(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar="ace",
        platform="Reference",
    )

    assert provider.provenance["platform"] == "Reference"
    assert provider.provenance["platform_properties"] == {}


def test_lcpo_has_an_explicit_openmm_version_boundary(methanol_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(methanol_mol2), charge=0, mult=1)
    major_minor = tuple(
        int(part) for part in importlib.metadata.version("openmm").split(".")[:2]
    )
    if major_minor < (8, 5):
        with pytest.raises(ImportError, match="OpenMM>=8.5"):
            OpenMMGB(atoms, atoms.get_initial_charges(), nonpolar="lcpo")
    else:
        result = OpenMMGB(atoms, atoms.get_initial_charges(), nonpolar="lcpo").evaluate(
            atoms, need_forces=True
        )
        assert np.isfinite(result.energy_hartree)
        assert np.isfinite(result.forces_hartree_per_angstrom).all()
        assert result.components_hartree["nonpolar"] > 0.0


def test_lcpo_complete_force_matches_finite_difference(methanol_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    major_minor = tuple(
        int(part) for part in importlib.metadata.version("openmm").split(".")[:2]
    )
    if major_minor < (8, 5):
        pytest.skip("LCPO requires OpenMM>=8.5")
    atoms = MOL2Reader(str(methanol_mol2), charge=0, mult=1)
    provider = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="lcpo"
    )
    result = provider.evaluate(atoms, need_forces=True)
    h = 1.0e-2
    plus = atoms.copy()
    plus.positions[1, 0] += h
    minus = atoms.copy()
    minus.positions[1, 0] -= h
    fd_force = -(
        provider.evaluate(plus).energy_hartree - provider.evaluate(minus).energy_hartree
    ) / (2 * h)
    assert np.isclose(result.forces_hartree_per_angstrom[1, 0], fd_force, atol=2e-6)


def test_lcpo_uses_gaff_nitro_oxygen_types_for_amber_energy_and_force():
    from maple.function.calculator.extra_correction.implicit.common import (
        KJ_PER_MOL_PER_HARTREE,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    data = Path(__file__).parent / "data/amber_gb_reference"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    case = next(
        item for item in manifest["cases"] if item["case_id"] == "4-nitroaniline"
    )
    atoms = MOL2Reader(str(data / case["mol2"]), charge=0, mult=1)
    provider = OpenMMGB(
        atoms,
        np.asarray(case["charges_e"]),
        model="obc2",
        nonpolar="lcpo",
        platform="Reference",
    )

    installation = provider.provenance["nonpolar_installation"]
    assert installation["typed_atom_indices"] == [8, 9]
    assert installation["adjusted_atom_indices"] == [8, 9]

    result = provider.evaluate(atoms, need_forces=True)
    reference = case["models"]["obc2"]
    kcal_per_hartree = KJ_PER_MOL_PER_HARTREE / 4.184
    nonpolar_kcal_mol = result.components_hartree["nonpolar"] * kcal_per_hartree
    total_kcal_mol = result.energy_hartree * kcal_per_hartree
    force_kcal_mol_angstrom = result.forces_hartree_per_angstrom * kcal_per_hartree

    assert nonpolar_kcal_mol == pytest.approx(reference["nonpolar_lcpo"], abs=1.0e-12)
    assert total_kcal_mol == pytest.approx(reference["total_lcpo"], abs=1.0e-3)
    assert (
        np.max(
            np.abs(
                force_kcal_mol_angstrom
                - np.asarray(reference["total_lcpo_force_kcal_mol_angstrom"])
            )
        )
        < 1.0e-3
    )


def test_openmm_gb_rejects_invalid_dynamic_charge_array(water_mol2):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = OpenMMGB(atoms, atoms.get_initial_charges())

    with pytest.raises(ValueError, match="one finite partial charge per atom"):
        provider.evaluate(atoms, charges=np.array([0.0, 0.0]))


def test_gbn2_phosphorus_fails_closed_instead_of_using_openmm_default_parameters():
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    path = Path(__file__).parent / "data/gbn2_phosphorus_negative.mol2"
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    charges = atoms.get_initial_charges()
    with pytest.raises(NotImplementedError, match="phosphorus-specific"):
        OpenMMGB(atoms, charges, model="gbn2", nonpolar="ace")


def test_gbn2_sulfur_fails_closed_instead_of_using_inexact_descreening():
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    path = Path(__file__).parent / "data/amber_gb_reference/mol2/mobley_2725215.mol2"
    atoms = MOL2Reader(str(path), charge=0, mult=1)
    charges = atoms.get_initial_charges()

    with pytest.raises(NotImplementedError, match="negative screening radius"):
        OpenMMGB(atoms, charges, model="gbn2", nonpolar="none")


def test_gbn2_generic_mol_ester_uses_amber_mbondi3_radius_and_force():
    from maple.function.calculator.extra_correction.implicit.common import (
        KJ_PER_MOL_PER_HARTREE,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    data = Path(__file__).parent / "data/amber_gb_reference"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    case = next(
        item for item in manifest["cases"] if item["case_id"] == "methyl-hexanoate"
    )
    atoms = MOL2Reader(str(data / case["mol2"]), charge=0, mult=1)
    provider = OpenMMGB(
        atoms,
        np.asarray(case["charges_e"]),
        model="gbn2",
        nonpolar="none",
        platform="Reference",
    )

    assert provider.radius_result.radii_angstrom[6] == pytest.approx(1.5)
    assert provider.provenance["radius_provider"]["parameter_adjustments"][
        "adjusted_atom_indices"
    ] == [6]

    result = provider.evaluate(atoms, need_forces=True)
    reference = case["models"]["gbn2"]
    kcal_per_hartree = KJ_PER_MOL_PER_HARTREE / 4.184
    energy_kcal_mol = result.energy_hartree * kcal_per_hartree
    force_kcal_mol_angstrom = result.forces_hartree_per_angstrom * kcal_per_hartree

    assert energy_kcal_mol == pytest.approx(reference["polar"], abs=1.0e-3)
    assert (
        np.max(
            np.abs(
                force_kcal_mol_angstrom
                - np.asarray(reference["polar_force_kcal_mol_angstrom"])
            )
        )
        < 1.0e-3
    )
