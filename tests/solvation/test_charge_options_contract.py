"""One charge-option identity contract, before tools or caller-state changes."""

from copy import deepcopy

import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import charges, correction

INVALID = [
    {"source": "banana"},
    {"source": "mol2", "method": "abcg2"},
    {"source": "mol2", "method": None},
    {"source": "mol2", "geometry": "provider"},
    {"source": "mol2", "timeout": 1.0},
    {"source": "mol2", "label": None},
    {"source": "mol2", "unknown": True},
    {"source": "maple", "method": "bogus"},
    {"source": "maple", "method": "am1bcc", "label": "pretend"},
    {"source": "maple", "method": "am1bcc", "timeout": float("inf")},
]


@pytest.mark.parametrize("options", INVALID)
@pytest.mark.parametrize("entrypoint", ["correction", "prepare"])
def test_invalid_charge_identity_rejects_before_any_side_effect(
    tmp_path, monkeypatch, options, entrypoint
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid charge identity reached a provider")

    monkeypatch.setattr(charges, "_mol2_charges", forbidden)
    monkeypatch.setattr(charges, "_ambertools_charges", forbidden)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    before = set(atoms.arrays)
    with pytest.raises(ValueError):
        if entrypoint == "correction":
            correction.ImplicitSolvationCorrection(
                atoms,
                options,
                {"method": "gb", "provider": "openmm", "experimental": True},
                output=tmp_path / "bad.out",
            )
        else:
            charges.prepare_charges(atoms, options, tmp_path / "audit")
    assert set(atoms.arrays) == before
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options",
    [
        {
            "source": "MOL2",
            "mode": "FIXED",
            "geometry": "KEEP",
            "label": "source charge",
        },
        {"source": "maple"},
        {"source": "MAPLE", "method": "ABCG2", "geometry": "PROVIDER", "timeout": 4.0},
    ],
)
def test_cli_and_direct_paths_share_normalized_charge_identity(
    options, monkeypatch, tmp_path
):
    from maple.function.read.charge_options import normalize_charge_options
    from maple.function.read.command_control import CommandControl

    original = deepcopy(options)
    expected = normalize_charge_options(options)
    parsed = {"charge": deepcopy(options)}
    CommandControl._validate_charge(parsed, required=True, output_path=None)
    assert parsed["charge"] == expected
    assert options == original
    sentinel = object()
    if expected["source"] == "mol2":

        def prepare(atoms, label):
            assert label == expected["label"]
            return sentinel

        monkeypatch.setattr(charges, "_mol2_charges", prepare)
    else:

        def prepare(atoms, actual, audit):
            assert actual == expected
            return sentinel

        monkeypatch.setattr(charges, "_ambertools_charges", prepare)
    assert charges.prepare_charges(Atoms("H"), options, tmp_path) is sentinel
    assert options == original


def test_cli_rejects_explicit_null_mol2_method_without_mutating_options():
    from maple.function.read.command_control import CommandControl

    options = {"source": "mol2", "method": None}
    params = {"charge": options}
    with pytest.raises(ValueError, match="does not accept method"):
        CommandControl._validate_charge(params, required=True, output_path=None)
    assert params["charge"] is options
    assert options == {"source": "mol2", "method": None}
