import importlib
import sys
import tomllib
from pathlib import Path

from setuptools import find_packages


PBC_CLASS_CONTRACT = {
    "aimnet2-pbc": (
        "maple.function.calculator.aimnet._aimnet2_official_pbc_calculator",
        "AIMNet2OfficialPBCCalculator",
    ),
    "aimnet2nse-pbc": (
        "maple.function.calculator.aimnet._aimnet2_official_pbc_calculator",
        "AIMNet2OfficialPBCCalculator",
    ),
    "mace-mp-pbc-small": (
        "maple.function.calculator.mace._mace_official_pbc_calculator",
        "MACEOfficialPBCCalculator",
    ),
    "mace-mp-pbc-medium": (
        "maple.function.calculator.mace._mace_official_pbc_calculator",
        "MACEOfficialPBCCalculator",
    ),
    "mace-mp-pbc-large": (
        "maple.function.calculator.mace._mace_official_pbc_calculator",
        "MACEOfficialPBCCalculator",
    ),
}


def test_pbc_calculator_classes_import_without_optional_extras(monkeypatch):
    for optional_name in ("aimnet", "aimnet.calculators", "mace", "mace.calculators"):
        monkeypatch.delitem(sys.modules, optional_name, raising=False)

    for module_name, class_name in PBC_CLASS_CONTRACT.values():
        module = importlib.import_module(module_name)

        assert getattr(module, class_name).__name__ == class_name


def test_pyproject_declares_pbc_optional_extras():
    pyproject = _read_pyproject()

    extras = pyproject["project"]["optional-dependencies"]
    assert extras["pbc-aimnet"] == ["aimnet[ase]"]
    assert extras["pbc-mace"] == ["mace-torch>=0.3.14,<0.4"]
    assert "full" in extras
    assert "minimal" in extras


def test_setuptools_package_discovery_includes_calculator_subpackages():
    pyproject = _read_pyproject()
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]

    assert package_find["where"] == ["."]
    assert package_find["include"] == ["maple*"]

    packages = set(
        find_packages(
            where=package_find["where"][0],
            include=package_find["include"],
        )
    )
    assert "maple.function.calculator.aimnet" in packages
    assert "maple.function.calculator.mace" in packages


def _read_pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text())
