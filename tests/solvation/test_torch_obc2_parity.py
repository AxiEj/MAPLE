from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from maple.function.read.filereader.mol2_reader import MOL2Reader

pytest.importorskip("openmm", reason="OpenMM optional dependency is not installed")
torch = pytest.importorskip(
    "torch", reason="Torch optional dependency is not installed"
)

DATA_DIR = Path(__file__).parent / "data" / "amber_gb_reference" / "audit"
FROZEN_CORPUS_CASES = (
    "benzene",
    "methanol",
    "bromobenzene",
    "methyl-hexanoate",
    "iodobenzene",
    "glucose",
    "4-nitroaniline",
    "dimethyl-sulfide",
    "fluorobenzene",
    "nitralin",
    "aniline",
    "chlorobenzene",
)

ENERGY_ABS_HARTREE = 2.0e-10
ENERGY_REL = 1.0e-9
FORCE_MAX_HARTREE_PER_ANGSTROM = 2.0e-7
FORCE_RMS_HARTREE_PER_ANGSTROM = 5.0e-8
CPU_OPERATIONAL_ENERGY_ABS_HARTREE = 2.0e-8


def _atoms(case: str, water_mol2):
    path = (
        water_mol2
        if case == "water"
        else DATA_DIR / "methyl-hexanoate" / "normalized.mol2"
    )
    return MOL2Reader(str(path), charge=0, mult=1)


def _backends(atoms, nonpolar: str):
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import (
        TorchOBC2,
    )

    reference = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar=nonpolar,
        platform="Reference",
    )
    parameters = build_obc2_parameters(
        reference.charges,
        reference.radius_result,
        nonpolar=nonpolar,
    )
    candidate = TorchOBC2(
        parameters,
        device="cpu",
        dtype=torch.float64,
    )
    return reference, candidate, parameters


def _assert_close(actual: np.ndarray | float, expected: np.ndarray | float, *, atol):
    np.testing.assert_allclose(actual, expected, rtol=ENERGY_REL, atol=atol)


def test_shared_parameters_are_deeply_immutable_and_owned(water_mol2):
    atoms = _atoms("water", water_mol2)
    reference, _candidate, parameters = _backends(atoms, "ace")

    original_charge = float(parameters.charges[0])
    reference.charges[0] += 0.125
    reference.radius_result.provider_parameters[0, 0] += 0.125

    assert parameters.charges.flags.writeable is False
    assert parameters.provider_parameters.flags.writeable is False
    assert parameters.offset_radii_nm.flags.writeable is False
    assert parameters.scaled_offset_radii_nm.flags.writeable is False
    assert parameters.charges[0] == original_charge
    with pytest.raises(ValueError, match="read-only"):
        parameters.charges[0] = 0.0
    with pytest.raises((TypeError, FrozenInstanceError)):
        parameters.nonpolar = "none"
    with pytest.raises(TypeError):
        parameters.provenance["mutated"] = True
    nested = next(
        value for value in parameters.provenance.values() if isinstance(value, Mapping)
    )
    with pytest.raises(TypeError):
        nested["mutated"] = True


@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_shared_parameters_match_finalized_openmm_particles(water_mol2, nonpolar):
    atoms = _atoms("water", water_mol2)
    reference, _candidate, parameters = _backends(atoms, nonpolar)
    finalized = np.asarray(
        [
            list(reference._total.force.getParticleParameters(index))
            for index in range(reference._total.force.getNumParticles())
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(finalized[:, 0], parameters.charges)
    np.testing.assert_allclose(
        finalized[:, 1], parameters.offset_radii_nm, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        finalized[:, 2],
        parameters.scaled_offset_radii_nm,
        rtol=0.0,
        atol=1.0e-15,
    )


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_torch_matches_openmm_component_energies(water_mol2, case, nonpolar):
    atoms = _atoms(case, water_mol2)
    reference, candidate, _parameters = _backends(atoms, nonpolar)

    expected = reference.evaluate(atoms, need_forces=False)
    actual = candidate.evaluate(atoms, need_forces=False)

    _assert_close(
        actual.energy_hartree, expected.energy_hartree, atol=ENERGY_ABS_HARTREE
    )
    _assert_close(
        actual.components_hartree["polar"],
        expected.components_hartree["polar"],
        atol=ENERGY_ABS_HARTREE,
    )
    _assert_close(
        actual.components_hartree["nonpolar"],
        expected.components_hartree["nonpolar"],
        atol=ENERGY_ABS_HARTREE,
    )
    assert actual.energy_hartree == pytest.approx(
        actual.components_hartree["polar"] + actual.components_hartree["nonpolar"],
        abs=1.0e-12,
    )


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_torch_matches_all_openmm_cartesian_forces(water_mol2, case, nonpolar):
    atoms = _atoms(case, water_mol2)
    reference, candidate, _parameters = _backends(atoms, nonpolar)

    expected = reference.evaluate(atoms, need_forces=True)
    actual = candidate.evaluate(atoms, need_forces=True)
    difference = (
        actual.forces_hartree_per_angstrom - expected.forces_hartree_per_angstrom
    )

    assert np.max(np.abs(difference)) <= FORCE_MAX_HARTREE_PER_ANGSTROM
    assert np.sqrt(np.mean(difference**2)) <= FORCE_RMS_HARTREE_PER_ANGSTROM


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
def test_torch_matches_openmm_ace_force_component(water_mol2, case):
    atoms = _atoms(case, water_mol2)
    openmm_total, torch_total, _parameters = _backends(atoms, "ace")
    openmm_polar, torch_polar, _parameters = _backends(atoms, "none")

    expected = (
        openmm_total.evaluate(atoms, need_forces=True).forces_hartree_per_angstrom
        - openmm_polar.evaluate(atoms, need_forces=True).forces_hartree_per_angstrom
    )
    actual = (
        torch_total.evaluate(atoms, need_forces=True).forces_hartree_per_angstrom
        - torch_polar.evaluate(atoms, need_forces=True).forces_hartree_per_angstrom
    )
    difference = actual - expected

    assert np.max(np.abs(difference)) <= FORCE_MAX_HARTREE_PER_ANGSTROM
    assert np.sqrt(np.mean(difference**2)) <= FORCE_RMS_HARTREE_PER_ANGSTROM


@pytest.mark.parametrize("case", FROZEN_CORPUS_CASES)
@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_torch_matches_reference_on_frozen_corpus_and_displacement(
    water_mol2, case, nonpolar
):
    atoms = MOL2Reader(str(DATA_DIR / case / "normalized.mol2"), charge=0, mult=1)
    reference, candidate, _parameters = _backends(atoms, nonpolar)
    displaced = atoms.copy()
    generator = np.random.default_rng(20260922 + FROZEN_CORPUS_CASES.index(case))
    displaced.positions += generator.normal(scale=2.0e-4, size=(len(atoms), 3))

    for geometry in (atoms, displaced):
        expected = reference.evaluate(geometry, need_forces=True)
        actual = candidate.evaluate(geometry, need_forces=True)
        assert actual.energy_hartree == pytest.approx(
            expected.energy_hartree, abs=ENERGY_ABS_HARTREE, rel=ENERGY_REL
        )
        difference = (
            actual.forces_hartree_per_angstrom - expected.forces_hartree_per_angstrom
        )
        assert np.max(np.abs(difference)) <= FORCE_MAX_HARTREE_PER_ANGSTROM
        assert np.sqrt(np.mean(difference**2)) <= FORCE_RMS_HARTREE_PER_ANGSTROM


def test_torch_energy_force_translation_and_rotation_covariance(water_mol2):
    atoms = _atoms("methyl-hexanoate", water_mol2)
    _reference, candidate, _parameters = _backends(atoms, "ace")
    baseline = candidate.evaluate(atoms, need_forces=True)
    np.testing.assert_allclose(
        baseline.forces_hartree_per_angstrom.sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-13,
    )

    translated = atoms.copy()
    translated.positions += np.array([3.25, -1.75, 0.625])
    translated_result = candidate.evaluate(translated, need_forces=True)
    assert translated_result.energy_hartree == pytest.approx(
        baseline.energy_hartree, abs=1.0e-14
    )
    np.testing.assert_allclose(
        translated_result.forces_hartree_per_angstrom,
        baseline.forces_hartree_per_angstrom,
        rtol=0.0,
        atol=1.0e-13,
    )

    axis = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated_result = candidate.evaluate(rotated, need_forces=True)
    assert rotated_result.energy_hartree == pytest.approx(
        baseline.energy_hartree, abs=1.0e-14
    )
    np.testing.assert_allclose(
        rotated_result.forces_hartree_per_angstrom,
        baseline.forces_hartree_per_angstrom @ rotation.T,
        rtol=0.0,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_torch_remains_operationally_consistent_with_openmm_cpu(
    water_mol2, case, nonpolar
):
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = _atoms(case, water_mol2)
    _oracle, candidate, _parameters = _backends(atoms, nonpolar)
    cpu = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar=nonpolar,
        platform="CPU",
    )

    expected = cpu.evaluate(atoms, need_forces=True)
    actual = candidate.evaluate(atoms, need_forces=True)
    difference = (
        actual.forces_hartree_per_angstrom - expected.forces_hartree_per_angstrom
    )

    assert actual.energy_hartree == pytest.approx(
        expected.energy_hartree, abs=CPU_OPERATIONAL_ENERGY_ABS_HARTREE
    )
    assert np.max(np.abs(difference)) <= FORCE_MAX_HARTREE_PER_ANGSTROM
    assert np.sqrt(np.mean(difference**2)) <= FORCE_RMS_HARTREE_PER_ANGSTROM


@pytest.mark.parametrize("case", ["water", "methyl-hexanoate"])
def test_nonpolar_none_returns_exactly_zero_ace(water_mol2, case):
    atoms = _atoms(case, water_mol2)
    _reference, candidate, _parameters = _backends(atoms, "none")

    result = candidate.evaluate(atoms, need_forces=True)

    assert result.components_hartree["nonpolar"] == 0.0
    assert result.energy_hartree == result.components_hartree["polar"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.__setitem__((0, 0), np.nan), "finite"),
        (lambda values: values.__setitem__((0, 0), 0.0), "positive"),
    ],
)
def test_parameter_builder_rejects_invalid_radius_domains(
    water_mol2, mutation, message
):
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = _atoms("water", water_mol2)
    reference = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="none"
    )
    invalid = reference.radius_result.provider_parameters.copy()
    mutation(invalid)
    invalid_result = type(reference.radius_result)(
        profile=reference.radius_result.profile,
        radii_angstrom=invalid[:, 0] * 10.0,
        provider_parameters=invalid,
        provenance=reference.radius_result.provenance,
    )

    with pytest.raises(ValueError, match=message):
        build_obc2_parameters(reference.charges, invalid_result, nonpolar="none")


def test_parameter_builder_rejects_unsupported_nonpolar_profile(water_mol2):
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB

    atoms = _atoms("water", water_mol2)
    reference = OpenMMGB(
        atoms, atoms.get_initial_charges(), model="obc2", nonpolar="none"
    )

    with pytest.raises(ValueError, match="ace.*none|none.*ace"):
        build_obc2_parameters(
            reference.charges, reference.radius_result, nonpolar="lcpo"
        )


def test_torch_derivative_backend_rejects_unverified_openmm_source(
    water_mol2, monkeypatch
):
    from maple.function.calculator.extra_correction.implicit import openmm_compat
    from maple.function.calculator.extra_correction.implicit.obc2_parameters import (
        build_obc2_parameters,
    )
    from maple.function.calculator.extra_correction.implicit.openmm_gb import OpenMMGB
    from maple.function.calculator.extra_correction.implicit.torch_obc2 import (
        TorchOBC2,
    )

    atoms = _atoms("water", water_mol2)
    reference = OpenMMGB(
        atoms,
        atoms.get_initial_charges(),
        model="obc2",
        nonpolar="ace",
        platform="Reference",
    )
    parameters = build_obc2_parameters(
        reference.charges, reference.radius_result, nonpolar="ace"
    )
    monkeypatch.setattr(openmm_compat, "VERIFIED_CUSTOMGBFORCES_SHA256", "0" * 64)
    assert np.isfinite(reference.evaluate(atoms).energy_hartree)

    with pytest.raises(ImportError, match="customgbforces.py SHA256"):
        TorchOBC2(parameters, device="cpu", dtype=torch.float64)
