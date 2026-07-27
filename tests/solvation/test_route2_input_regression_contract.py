"""Regression contract: Route 2 restrictions must not disable base MAPLE APIs."""

from __future__ import annotations

import pytest
from ase import Atoms

from maple.function.calculator.set_calculator import SetCalculator
from maple.function.read.command_control import CommandControl


def _parse(*lines: str) -> dict[str, object]:
    return CommandControl.from_settings(list(lines)).as_dict()


def test_gas_phase_charge_configuration_remains_available_outside_route2():
    """A Route-2-only rejection must not become a branch-global rejection."""

    params = _parse(
        "#model=aimnet2",
        "#sp",
        "#charge(source=mol2)",
    )

    assert params["charge"] == {"source": "mol2"}
    assert "solv" not in params


def test_enhance_gbsa_contract_remains_available_alongside_route2():
    """Adding SMD must extend, not replace, the enhance GBSA selector."""

    params = _parse(
        "#model=aimnet2",
        "#sp",
        "#solv(implicit=water,method=gbsa,experimental=true)",
    )

    assert params["solv"] == {
        "implicit": "water",
        "method": "gbsa",
        "experimental": True,
    }


def test_calculator_factory_keeps_enhance_gbsa_available(tmp_path):
    atoms = Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    builder = SetCalculator(
        device="cpu",
        model="aimnet2",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="gbsa",
        solvent="water",
        solvation_options={
            "implicit": "water",
            "method": "gbsa",
            "experimental": True,
        },
    )

    builder._validate_solvent_config()


def test_route2_domain_rejection_precedes_model_discovery(tmp_path):
    atoms = Atoms("NO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    builder = SetCalculator(
        device="cpu",
        model="macepol-m",
        output=str(tmp_path / "job.out"),
        atoms=atoms,
        implicit="smd",
        solvent="water",
        solvation_options={
            "implicit": "water",
            "method": "smd",
            "experimental": True,
        },
    )
    builder._discover_calculator_class = lambda _name: pytest.fail(
        "Model discovery must not run before Route-2 domain validation."
    )

    with pytest.raises(ValueError, match="electron-count parity"):
        builder._build_calculator()


def test_route2_still_rejects_an_independent_charge_provider():
    with pytest.raises(ValueError, match=r"remove #charge"):
        _parse(
            "#model=macepol-m",
            "#sp",
            "#charge(source=mol2)",
            "#solv(implicit=water,method=smd,experimental=true)",
        )
