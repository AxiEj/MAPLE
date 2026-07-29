from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.calculator_base import EV2HARTREE


class HarmonicDelegate:
    def __init__(self):
        self.results = {}

    def calculate(self, atoms, properties, system_changes):
        positions = np.asarray(atoms.get_positions(), dtype=float)
        self.results = {"energy": float(np.sum(positions**2))}
        if "forces" in properties:
            self.results["forces"] = -2.0 * positions


class InvalidDelegate:
    def __init__(self, failure_mode):
        self.failure_mode = failure_mode
        self.results = {}

    def calculate(self, atoms, properties, system_changes):
        del system_changes
        self.results = {
            "energy": np.nan if self.failure_mode == "energy" else 0.0,
        }
        if "forces" in properties:
            self.results["forces"] = (
                np.full((len(atoms), 3), np.inf)
                if self.failure_mode == "forces"
                else np.zeros((len(atoms), 3))
            )


def _single_h_mol2():
    return {
        "atom_names": ["H1"],
        "atom_types": ["H"],
        "bonds": [],
    }


@pytest.mark.parametrize(
    ("calculator_path", "sha_symbol", "atoms_factory"),
    [
        (
            "maple.function.calculator.mace._maceoff24_calculator.MACEOFF24MediumCalculator",
            "MACE_OFF24_MEDIUM_SHA256",
            lambda: Atoms("H", positions=[[0.4, -0.2, 0.1]]),
        ),
        (
            "maple.function.calculator.aceff._aceff2_calculator.AceFF2Calculator",
            "ACEFF2_SHA256",
            lambda: Atoms("H", positions=[[0.4, -0.2, 0.1]]),
        ),
    ],
)
def test_ordinary_adapters_delegate_units_forces_and_numerical_hessian(
    tmp_path, monkeypatch, calculator_path, sha_symbol, atoms_factory
):
    module_name, class_name = calculator_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    calculator_class = getattr(module, class_name)
    checkpoint = tmp_path / "official.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(module, "_sha256", lambda _path: getattr(module, sha_symbol))

    atoms = atoms_factory()
    atoms.info.update(charge=0, mult=1)
    atoms.info["mol2"] = _single_h_mol2()
    calc = calculator_class(
        device="cpu",
        model_path=str(checkpoint),
        calculator_factory=lambda *_args: HarmonicDelegate(),
    )

    np.testing.assert_allclose(
        calc.get_forces(atoms),
        -2.0 * atoms.positions * EV2HARTREE,
    )
    calc.calculate(atoms, properties=("hessian",))
    np.testing.assert_allclose(
        calc.results["hessian"],
        np.eye(3) * 2.0 * EV2HARTREE,
        atol=1.0e-10,
    )


def test_mace_off24_rejects_user_signed_card_and_non_neutral_input(
    tmp_path, monkeypatch
):
    from maple.function.calculator.mace import _maceoff24_calculator as mace_mod

    checkpoint = tmp_path / "mace.model"
    checkpoint.write_bytes(b"mace")
    monkeypatch.setattr(
        mace_mod, "_sha256", lambda _path: mace_mod.MACE_OFF24_MEDIUM_SHA256
    )
    bad_card = tmp_path / "bad.yaml"
    bad_card.write_text(
        json.dumps(
            {
                "model_id": "mace-off24-medium",
                "version": "MACE-OFF24(M)",
                "source": {
                    "revision": mace_mod.MACE_OFF24_OFFICIAL_IDENTITY["source_revision"]
                },
                "checkpoint_sha256": mace_mod.MACE_OFF24_MEDIUM_SHA256,
            }
        )
    )

    with pytest.raises(
        ValueError, match="does not accept user-supplied model_card_path"
    ):
        mace_mod.MACEOFF24MediumCalculator(
            "cpu",
            model_path=str(checkpoint),
            model_card_path=str(bad_card),
            calculator_factory=lambda *_args: HarmonicDelegate(),
        )

    calc = mace_mod.MACEOFF24MediumCalculator(
        "cpu",
        model_path=str(checkpoint),
        calculator_factory=lambda *_args: HarmonicDelegate(),
    )
    charged = Atoms("H", positions=[[0, 0, 0]])
    charged.info.update(charge=1, mult=1)
    with pytest.raises(ValueError, match="neutral singlets"):
        calc.get_potential_energy(charged)


@pytest.mark.parametrize(
    ("module_name", "class_name", "sha_symbol"),
    [
        (
            "maple.function.calculator.mace._maceoff24_calculator",
            "MACEOFF24MediumCalculator",
            "MACE_OFF24_MEDIUM_SHA256",
        ),
        (
            "maple.function.calculator.aceff._aceff2_calculator",
            "AceFF2Calculator",
            "ACEFF2_SHA256",
        ),
    ],
)
def test_ordinary_adapters_require_explicit_charge_and_multiplicity(
    tmp_path,
    monkeypatch,
    module_name,
    class_name,
    sha_symbol,
):
    module = __import__(module_name, fromlist=[class_name])
    calculator_class = getattr(module, class_name)
    checkpoint = tmp_path / f"{class_name}.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(module, "_sha256", lambda _path: getattr(module, sha_symbol))
    calc = calculator_class(
        "cpu",
        model_path=str(checkpoint),
        calculator_factory=lambda *_args: HarmonicDelegate(),
    )

    for present_field, present_value in (("charge", 0), ("mult", 1)):
        atoms = Atoms("H", positions=[[0, 0, 0]])
        atoms.info[present_field] = present_value
        atoms.info["mol2"] = _single_h_mol2()
        with pytest.raises(ValueError, match=r"explicit atoms\.info"):
            calc.get_potential_energy(atoms)


@pytest.mark.parametrize(
    ("module_name", "class_name", "sha_symbol"),
    [
        (
            "maple.function.calculator.mace._maceoff24_calculator",
            "MACEOFF24MediumCalculator",
            "MACE_OFF24_MEDIUM_SHA256",
        ),
        (
            "maple.function.calculator.aceff._aceff2_calculator",
            "AceFF2Calculator",
            "ACEFF2_SHA256",
        ),
    ],
)
@pytest.mark.parametrize(
    ("failure_mode", "message"), [("energy", "energy"), ("forces", "forces")]
)
def test_ordinary_adapters_reject_nonfinite_delegate_results(
    tmp_path,
    monkeypatch,
    module_name,
    class_name,
    sha_symbol,
    failure_mode,
    message,
):
    module = __import__(module_name, fromlist=[class_name])
    calculator_class = getattr(module, class_name)
    checkpoint = tmp_path / f"{class_name}-{failure_mode}.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(module, "_sha256", lambda _path: getattr(module, sha_symbol))
    calc = calculator_class(
        "cpu",
        model_path=str(checkpoint),
        calculator_factory=lambda *_args: InvalidDelegate(failure_mode),
    )
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.info.update(charge=0, mult=1, mol2=_single_h_mol2())

    with pytest.raises(ValueError, match=message):
        if failure_mode == "forces":
            calc.get_forces(atoms)
        else:
            calc.get_potential_energy(atoms)


def test_aceff_requires_official_card_charge_range_singlet_and_mol2_topology(
    tmp_path, monkeypatch
):
    from maple.function.calculator.aceff import _aceff2_calculator as ace_mod

    checkpoint = tmp_path / "ace.ckpt"
    checkpoint.write_bytes(b"aceff")
    monkeypatch.setattr(ace_mod, "_sha256", lambda _path: ace_mod.ACEFF2_SHA256)
    bad_card = tmp_path / "bad.yaml"
    bad_card.write_text("{}")

    with pytest.raises(
        ValueError, match="does not accept user-supplied model_card_path"
    ):
        ace_mod.AceFF2Calculator(
            "cpu",
            model_path=str(checkpoint),
            model_card_path=str(bad_card),
            calculator_factory=lambda *_args: HarmonicDelegate(),
        )

    calc = ace_mod.AceFF2Calculator(
        "cpu",
        model_path=str(checkpoint),
        calculator_factory=lambda *_args: HarmonicDelegate(),
    )

    missing = Atoms("H", positions=[[0, 0, 0]])
    missing.info.update(charge=0, mult=1)
    with pytest.raises(ValueError, match="requires explicit MOL2 topology"):
        calc.get_potential_energy(missing)

    out_of_range = missing.copy()
    out_of_range.info.update(charge=3, mult=1, mol2=_single_h_mol2())
    with pytest.raises(ValueError, match=r"\{-2,-1,0,1,2\}"):
        calc.get_potential_energy(out_of_range)

    fragments = Atoms("HH", positions=[[0, 0, 0], [10, 0, 0]])
    fragments.info.update(
        charge=0,
        mult=1,
        mol2={
            "atom_names": ["H1", "H2"],
            "atom_types": ["H", "H"],
            "bonds": [],
        },
    )
    with pytest.raises(ValueError, match="one small-molecule fragment"):
        calc.get_potential_energy(fragments)


def test_default_ordinary_model_cards_are_json_and_fail_closed():
    from maple.function.calculator.aceff._aceff2_calculator import ACEFF2_CARD
    from maple.function.calculator.mace._maceoff24_calculator import MACE_OFF24_CARD

    mace = json.loads(MACE_OFF24_CARD.read_text(encoding="utf-8"))
    ace = json.loads(ACEFF2_CARD.read_text(encoding="utf-8"))
    assert mace["checkpoint_sha256"].startswith("e5ccf583")
    assert "mace_off(model='medium')" in mace["notes"][0]
    assert MACE_OFF24_CARD.read_text(encoding="utf-8").count('"elements"') == 1
    assert ace["checkpoint_sha256"].startswith("877af6bf")
    assert ace["capabilities"]["supports_multifragment"] is False
    assert ace["capabilities"]["requires_topology"] is True


@pytest.mark.parametrize(
    ("module_name", "class_name", "sha_symbol", "canonical", "alias"),
    [
        (
            "maple.function.calculator.mace._maceoff24_calculator",
            "MACEOFF24MediumCalculator",
            "MACE_OFF24_MEDIUM_SHA256",
            "mace-off24-medium",
            "maceoff24medium",
        ),
        (
            "maple.function.calculator.aceff._aceff2_calculator",
            "AceFF2Calculator",
            "ACEFF2_SHA256",
            "aceff-2.0",
            "aceff20",
        ),
    ],
)
def test_fixed_ordinary_adapters_reject_forged_identity_and_report_canonical_name(
    tmp_path,
    monkeypatch,
    module_name,
    class_name,
    sha_symbol,
    canonical,
    alias,
):
    module = __import__(module_name, fromlist=[class_name])
    calculator_class = getattr(module, class_name)
    checkpoint = tmp_path / f"{class_name}.ckpt"
    checkpoint.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(module, "_sha256", lambda _path: getattr(module, sha_symbol))

    with pytest.raises(ValueError, match="pinned to model identity"):
        calculator_class(
            "cpu",
            model="forged-identity",
            model_path=str(checkpoint),
            calculator_factory=lambda *_args: HarmonicDelegate(),
        )

    calc = calculator_class(
        "cpu",
        model=alias,
        model_path=str(checkpoint),
        calculator_factory=lambda *_args: HarmonicDelegate(),
    )
    assert calc.model_name == canonical


@pytest.mark.parametrize(
    ("requested", "expected_class"),
    [
        ("mace-off24-medium", "MACEOFF24MediumCalculator"),
        ("maceoff24medium", "MACEOFF24MediumCalculator"),
        ("aceff-2.0", "AceFF2Calculator"),
        ("aceff20", "AceFF2Calculator"),
    ],
)
def test_set_calculator_discovers_new_builtin_names(
    tmp_path,
    requested,
    expected_class,
):
    from maple.function.calculator.set_calculator import SetCalculator

    setter = SetCalculator(
        device="cpu",
        model=requested,
        output=str(tmp_path / "maple.out"),
    )
    assert setter._discover_calculator_class(requested).__name__ == expected_class
