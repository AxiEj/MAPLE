"""Independent provider, direct API, and parser/factory contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from maple.function.read.command_control import CommandControl
from maple.function.read.filereader.mol2_reader import MOL2Reader


def _parse(*lines):
    return CommandControl.from_settings(list(lines)).as_dict()


@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_provider_owns_native_energy_force_hessian_and_hvp(water_mol2, nonpolar):
    from maple.function.calculator.extra_correction.implicit.independent_obc2 import (
        IndependentOBC2,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = IndependentOBC2(
        atoms,
        atoms.get_initial_charges(),
        nonpolar=nonpolar,
        device="cpu",
    )

    assert provider.supported_properties == frozenset(
        {"energy", "forces", "hessian", "hvp"}
    )
    result = provider.evaluate(atoms, need_forces=True)
    hessian = provider.hessian(atoms)
    direction = np.arange(1, 3 * len(atoms) + 1, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    directional = provider.directional_derivatives(atoms, direction)

    assert np.isfinite(result.energy_hartree)
    assert np.isfinite(result.forces_hartree_per_angstrom).all()
    assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
    np.testing.assert_allclose(
        directional.hvp_hartree_per_angstrom2,
        hessian @ direction,
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert directional.energy_hartree == pytest.approx(
        result.energy_hartree, rel=0.0, abs=1.0e-14
    )
    np.testing.assert_allclose(
        directional.forces_hartree_per_angstrom,
        result.forces_hartree_per_angstrom,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_provider_provenance_separates_runtime_preparation_oracle_and_scalar(water_mol2):
    from maple.function.calculator.extra_correction.implicit.independent_obc2 import (
        IndependentOBC2,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = IndependentOBC2(atoms, atoms.get_initial_charges(), nonpolar="none")
    provenance = provider.provenance

    assert provenance["provider"] == "torch-obc2"
    assert provenance["runtime_provider"] == "torch-obc2"
    assert provenance["preparation_provider"] == "maple-independent-mbondi2"
    assert provenance["oracle_provider"] is None
    assert provenance["model"] == "obc2"
    assert provenance["profile"] == "obc2-mbondi2"
    assert provenance["nonpolar"] == "none"
    assert "ace" not in provenance["formula_manifest_id"].lower()
    assert provenance["dtype"] == "torch.float64"
    assert provenance["topology_sha256"] == provider.topology.identity_sha256


def test_provider_rejects_changed_bound_atom_or_topology_identity(water_mol2):
    from maple.function.calculator.extra_correction.implicit.independent_obc2 import (
        IndependentOBC2,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = IndependentOBC2(atoms, atoms.get_initial_charges())

    changed_numbers = atoms.copy()
    changed_numbers.numbers[0] = 7
    with pytest.raises(ValueError, match="bound|topology|identity"):
        provider.evaluate(changed_numbers)

    changed_topology = atoms.copy()
    changed_topology.info["mol2"] = dict(changed_topology.info["mol2"])
    changed_topology.info["mol2"]["bonds"] = [[0, 1, "1"]]
    with pytest.raises(ValueError, match="bound|topology|identity|changed"):
        provider.evaluate(changed_topology)


def test_direct_hvp_does_not_materialize_full_hessian(water_mol2, monkeypatch):
    from maple.function.calculator.extra_correction.implicit.independent_obc2 import (
        IndependentOBC2,
    )

    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = IndependentOBC2(atoms, atoms.get_initial_charges())

    def forbidden(*_args, **_kwargs):
        pytest.fail("direct HVP materialized a dense Hessian")

    monkeypatch.setattr(provider.kernel, "hessian", forbidden)
    direction = np.ones(3 * len(atoms), dtype=np.float64)
    assert np.isfinite(provider.hvp(atoms, direction)).all()


def test_parser_accepts_explicit_torch_without_changing_openmm_default():
    independent = _parse(
        "#model=ani2x",
        "#sp",
        "#charge(source=mol2)",
        (
            "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
            "profile=obc2-mbondi2,nonpolar=ace,experimental=true)"
        ),
    )
    legacy = _parse(
        "#model=ani2x",
        "#sp",
        "#charge(source=mol2)",
        "#solv(implicit=water,method=gb,experimental=true)",
    )

    assert independent["solv"] == {
        "implicit": "water",
        "method": "gb",
        "provider": "torch",
        "model": "obc2",
        "profile": "obc2-mbondi2",
        "nonpolar": "ace",
        "experimental": True,
    }
    assert legacy["solv"]["provider"] == "openmm"


@pytest.mark.parametrize("nonpolar", ["ace", "none"])
def test_parser_accepts_independent_analytic_profile_without_openmm_platform(nonpolar):
    params = _parse(
        "#model=ani2x(hessian=analytic)",
        "#freq(method=mw)",
        "#charge(source=mol2)",
        (
            "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
            f"profile=obc2-mbondi2,nonpolar={nonpolar},experimental=true)"
        ),
    )
    assert params["solv"]["provider"] == "torch"
    assert params["solv"]["nonpolar"] == nonpolar
    assert "platform" not in params["solv"]


@pytest.mark.parametrize(
    "option",
    ["platform=Reference", "precision=double", "device_index=0", "opencl_platform_index=0"],
)
def test_parser_rejects_every_openmm_only_option_for_torch(option):
    with pytest.raises(ValueError, match="OpenMM.*provider=openmm|does not support"):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=mol2)",
            (
                "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
                f"nonpolar=ace,{option},experimental=true)"
            ),
        )


@pytest.mark.parametrize(
    "selector",
    [
        "model=hct,profile=hct-mbondi,nonpolar=ace",
        "model=obc1,profile=obc1-mbondi2,nonpolar=ace",
        "model=obc2,profile=obc2-mbondi2,nonpolar=lcpo",
    ],
)
def test_parser_rejects_non_obc2_or_non_ace_none_torch_profiles(selector):
    with pytest.raises(ValueError, match="provider=torch|OBC|obc2|nonpolar"):
        _parse(
            "#model=ani2x",
            "#sp",
            "#charge(source=mol2)",
            f"#solv(implicit=water,method=gb,provider=torch,{selector},experimental=true)",
        )

