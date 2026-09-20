from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from math import pi, sqrt
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit import ddx_lpb
from maple.function.calculator.extra_correction.implicit.ddx_lpb import (
    BOHR_ANGSTROM,
    DDXLPB,
    PROFILE_NAME,
    DDXLPBSettings,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader


def _single_carbon() -> Atoms:
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]])
    atoms.info["mol2"] = {
        "atom_names": ["C1"],
        "atom_types": ["C.3"],
        "atom_ids": [1],
        "component_ids": [0],
        "component_count": 1,
        "bonds": [],
    }
    atoms.new_array("_maple_mol2_atom_id", np.asarray([1], dtype=np.int64))
    return atoms


def test_settings_are_frozen_and_lock_the_default_profile():
    settings = DDXLPBSettings()
    assert PROFILE_NAME == "ddlpb-union-mbondi2-v1"
    assert settings == DDXLPBSettings(
        lmax=9,
        n_lebedev=302,
        eta=0.1,
        shift=0.0,
        solver_tolerance=1.0e-10,
        max_iterations=200,
        jacobi_n_diis=20,
        n_proc=1,
        solvent_epsilon=78.5,
        solute_epsilon=1.0,
        enable_fmm=False,
    )
    with pytest.raises(FrozenInstanceError):
        settings.lmax = 11  # type: ignore[misc]
    assert replace(settings, lmax=11, n_lebedev=590).lmax == 11


def test_fake_native_contract_uses_units_both_gradient_terms_and_atomic_audit(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    charges = atoms.get_initial_charges()
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, kind, centers, radii, **kwargs):
            calls["kind"] = kind
            calls["centers"] = np.asarray(centers)
            calls["radii"] = np.asarray(radii)
            calls["model_kwargs"] = kwargs
            self.atom_count = centers.shape[1]

        def multipole_electrostatics(self, multipoles):
            calls["multipoles"] = np.asarray(multipoles)
            return {
                "phi": np.zeros(self.atom_count),
                "e": np.zeros((3, self.atom_count)),
                "g": np.zeros((3, 3, self.atom_count)),
            }

        def multipole_psi(self, _multipoles):
            return np.zeros((1, self.atom_count))

    class FakeState:
        def __init__(self, model, psi, phi, electric_field):
            calls["state_args"] = (psi, phi, electric_field)
            self.atom_count = model.atom_count
            self.is_solved = True
            self.is_solved_adjoint = True
            self.x_n_iter = 7
            self.s_n_iter = 9

        def ddrun(self, field, *, tol):
            calls["ddrun"] = (field, tol)
            return 2.0, np.ones((3, self.atom_count))

        def multipole_force_terms(self, _multipoles):
            return np.full((3, self.atom_count), 2.0)

    fake = SimpleNamespace(
        __version__="0.8.0", __file__=__file__, Model=FakeModel, State=FakeState
    )
    monkeypatch.setattr(ddx_lpb, "_require_pyddx", lambda: (fake, "f" * 64))

    audit = tmp_path / "audit"
    provider = DDXLPB(
        atoms,
        charges,
        solvent_kappa_inverse_angstrom=0.025,
        audit_dir=audit,
    )
    result = provider.evaluate(atoms, need_forces=True)

    assert calls["kind"] == "lpb"
    assert calls["centers"] == pytest.approx(atoms.positions.T / BOHR_ANGSTROM)
    assert calls["radii"] == pytest.approx(provider.radii / BOHR_ANGSTROM)
    kwargs = calls["model_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["solvent_kappa"] == pytest.approx(0.025 * BOHR_ANGSTROM)
    assert kwargs["enable_force"] is True
    assert kwargs["enable_fmm"] is False
    assert calls["multipoles"] == pytest.approx(charges.reshape(1, -1) / sqrt(4 * pi))
    ddrun_call = calls["ddrun"]
    assert isinstance(ddrun_call, tuple)
    assert ddrun_call[1] == pytest.approx(1.0e-10)
    assert result.energy_hartree == 2.0
    assert result.components_hartree == {"polar": 2.0}
    assert result.forces_hartree_per_angstrom == pytest.approx(
        np.full((len(atoms), 3), -3.0 / BOHR_ANGSTROM)
    )
    assert result.provenance["solver"]["x_n_iter"] == 7
    assert result.provenance["solver"]["s_n_iter"] == 9
    assert result.provenance["native_runtime"]["binary_sha256"] == "f" * 64
    recorded = json.loads((audit / "ddx-lpb.result.json").read_text())
    assert recorded["energy_hartree"] == 2.0


def test_energy_only_disables_native_force_and_omits_force_output(
    water_mol2, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    flags: list[bool] = []

    class FakeModel:
        def __init__(self, _kind, centers, _radii, **kwargs):
            flags.append(kwargs["enable_force"])
            self.n = centers.shape[1]

        def multipole_electrostatics(self, _multipoles):
            return {"phi": np.zeros(self.n), "e": np.zeros((3, self.n))}

        def multipole_psi(self, _multipoles):
            return np.zeros((1, self.n))

    class FakeState:
        is_solved = True
        is_solved_adjoint = False
        x_n_iter = 3
        s_n_iter = 0

        def __init__(self, *_args):
            pass

        def ddrun(self, _field, *, tol):
            return 1.25

    fake = SimpleNamespace(
        __version__="0.8.0", __file__=__file__, Model=FakeModel, State=FakeState
    )
    monkeypatch.setattr(ddx_lpb, "_require_pyddx", lambda: (fake, "0" * 64))
    result = DDXLPB(
        atoms,
        atoms.get_initial_charges(),
        solvent_kappa_inverse_angstrom=0.02,
    ).evaluate(atoms)
    assert flags == [False]
    assert result.energy_hartree == 1.25
    assert result.forces_hartree_per_angstrom is None
    assert result.provenance["solver"]["s_n_iter"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"solvent_kappa_inverse_angstrom": True}, "not a boolean"),
        ({"solvent_kappa_inverse_angstrom": 0.0}, "strictly positive"),
        ({"solvent_kappa_inverse_angstrom": np.inf}, "finite"),
        (
            {
                "solvent_kappa_inverse_angstrom": 0.02,
                "settings": replace(DDXLPBSettings(), enable_fmm=True),
            },
            "enable_fmm",
        ),
        (
            {
                "solvent_kappa_inverse_angstrom": 0.02,
                "settings": replace(DDXLPBSettings(), solute_epsilon=2.0),
            },
            "solute_epsilon",
        ),
        (
            {
                "solvent_kappa_inverse_angstrom": 0.02,
                "settings": replace(DDXLPBSettings(), n_lebedev=51),
            },
            "Lebedev",
        ),
    ],
)
def test_invalid_controls_fail_before_native_import(
    water_mol2, monkeypatch, kwargs, message
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    monkeypatch.setattr(
        ddx_lpb,
        "_require_pyddx",
        lambda: (_ for _ in ()).throw(AssertionError("native import was reached")),
    )
    with pytest.raises(ValueError, match=message):
        DDXLPB(atoms, atoms.get_initial_charges(), **kwargs)


def test_invalid_charge_shape_and_periodicity_are_rejected(water_mol2):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    with pytest.raises(ValueError, match="one finite partial charge"):
        DDXLPB(atoms, [0.0], solvent_kappa_inverse_angstrom=0.02)
    atoms.pbc = True
    with pytest.raises(ValueError, match="nonperiodic"):
        DDXLPB(atoms, atoms.get_initial_charges(), solvent_kappa_inverse_angstrom=0.02)


def test_readonly_input_views_and_atom_identity_gate(water_mol2, monkeypatch):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = DDXLPB(
        atoms, atoms.get_initial_charges(), solvent_kappa_inverse_angstrom=0.02
    )
    with pytest.raises(ValueError, match="read-only"):
        provider.charges[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        provider.radii[0] = 1.0
    mutated = atoms.copy()
    mutated.numbers[0] = 6
    monkeypatch.setattr(
        ddx_lpb,
        "_require_pyddx",
        lambda: (_ for _ in ()).throw(AssertionError("native import was reached")),
    )
    with pytest.raises(ValueError, match="frozen atom identity"):
        provider.evaluate(mutated)
    same_element_reorder = atoms[[0, 2, 1]]
    with pytest.raises(ValueError, match="frozen atom identity"):
        provider.evaluate(same_element_reorder)


def test_missing_and_wrong_native_versions_fail_with_stable_types(monkeypatch):
    real_import = ddx_lpb.importlib.import_module

    def missing(name):
        if name == "pyddx":
            raise ImportError("missing")
        return real_import(name)

    monkeypatch.setattr(ddx_lpb.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="ddX LPB native import failed"):
        ddx_lpb._require_pyddx()

    monkeypatch.setattr(
        ddx_lpb.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__version__="0.9.0", __file__=__file__),
    )
    with pytest.raises(ImportError, match="requires pyddx==0.8.0"):
        ddx_lpb._require_pyddx()


def test_failed_or_malformed_native_results_never_write_success_audit(
    water_mol2, tmp_path, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    audit = tmp_path / "audit"

    class FakeModel:
        def __init__(self, _kind, centers, _radii, **_kwargs):
            self.n = centers.shape[1]

        def multipole_electrostatics(self, _multipoles):
            return {
                "phi": np.zeros(self.n),
                "e": np.zeros((3, self.n)),
                "g": np.zeros((3, 3, self.n)),
            }

        def multipole_psi(self, _multipoles):
            return np.zeros((1, self.n))

    class FakeState:
        is_solved = False
        is_solved_adjoint = False
        x_n_iter = 200
        s_n_iter = 200

        def __init__(self, *_args):
            pass

        def ddrun(self, _field, *, tol):
            return np.nan, np.zeros((3, len(atoms)))

    fake = SimpleNamespace(
        __version__="0.8.0", __file__=__file__, Model=FakeModel, State=FakeState
    )
    monkeypatch.setattr(ddx_lpb, "_require_pyddx", lambda: (fake, "0" * 64))
    provider = DDXLPB(
        atoms,
        atoms.get_initial_charges(),
        solvent_kappa_inverse_angstrom=0.02,
        audit_dir=audit,
    )
    with pytest.raises(RuntimeError, match="forward solve did not converge"):
        provider.evaluate(atoms, need_forces=True)
    assert not (audit / "ddx-lpb.result.json").exists()


def test_native_stage_errors_are_chained_and_never_fall_back(water_mol2, monkeypatch):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    class ExplodingModel:
        def __init__(self, *_args, **_kwargs):
            raise OSError("native model error")

    fake = SimpleNamespace(
        __version__="0.8.0",
        __file__=__file__,
        Model=ExplodingModel,
        State=object,
    )
    monkeypatch.setattr(ddx_lpb, "_require_pyddx", lambda: (fake, "0" * 64))
    provider = DDXLPB(
        atoms,
        atoms.get_initial_charges(),
        solvent_kappa_inverse_angstrom=0.02,
    )
    with pytest.raises(RuntimeError, match="model construction failed") as caught:
        provider.evaluate(atoms)
    assert isinstance(caught.value.__cause__, OSError)


def test_force_path_rejects_missing_field_gradient_and_wrong_native_shape(
    water_mol2, monkeypatch
):
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)

    class FakeModel:
        include_gradient = False

        def __init__(self, _kind, centers, _radii, **_kwargs):
            self.n = centers.shape[1]

        def multipole_electrostatics(self, _multipoles):
            field = {"phi": np.zeros(self.n), "e": np.zeros((3, self.n))}
            if self.include_gradient:
                field["g"] = np.zeros((3, 3, self.n))
            return field

        def multipole_psi(self, _multipoles):
            return np.zeros((1, self.n))

    class FakeState:
        is_solved = True
        is_solved_adjoint = True
        x_n_iter = 2
        s_n_iter = 2

        def __init__(self, *_args):
            pass

        def ddrun(self, _field, *, tol):
            return 1.0, np.zeros((2, len(atoms)))

        def multipole_force_terms(self, _multipoles):
            raise AssertionError("wrong primary gradient must stop force composition")

    fake = SimpleNamespace(
        __version__="0.8.0", __file__=__file__, Model=FakeModel, State=FakeState
    )
    monkeypatch.setattr(ddx_lpb, "_require_pyddx", lambda: (fake, "0" * 64))
    provider = DDXLPB(
        atoms,
        atoms.get_initial_charges(),
        solvent_kappa_inverse_angstrom=0.02,
    )
    with pytest.raises(ValueError, match="incomplete phi/e/g"):
        provider.evaluate(atoms, need_forces=True)
    FakeModel.include_gradient = True
    with pytest.raises(ValueError, match=r"expected shape \(3, N\)"):
        provider.evaluate(atoms, need_forces=True)


def test_real_charged_sphere_matches_lpb_solution_and_translation_force():
    pytest.importorskip("pyddx")
    atoms = _single_carbon()
    q = np.asarray([1.0])
    for kappa in (0.02 / BOHR_ANGSTROM, 0.1 / BOHR_ANGSTROM):
        provider = DDXLPB(atoms, q, solvent_kappa_inverse_angstrom=kappa)
        result = provider.evaluate(atoms, need_forces=True)
        radius_bohr = provider.radii[0] / BOHR_ANGSTROM
        kappa_bohr = kappa * BOHR_ANGSTROM
        expected = (1.0 / (78.5 * (1.0 + kappa_bohr * radius_bohr)) - 1.0) / (
            2.0 * radius_bohr
        )
        assert result.energy_hartree == pytest.approx(expected, abs=1.0e-12)
        assert result.forces_hartree_per_angstrom is not None
        assert np.linalg.norm(result.forces_hartree_per_angstrom) <= 1.0e-10


def test_real_water_all_coordinate_forces_match_two_finite_difference_steps(water_mol2):
    pytest.importorskip("pyddx")
    atoms = MOL2Reader(str(water_mol2), charge=0, mult=1)
    provider = DDXLPB(
        atoms,
        atoms.get_initial_charges(),
        solvent_kappa_inverse_angstrom=0.02 / BOHR_ANGSTROM,
    )
    result = provider.evaluate(atoms, need_forces=True)
    analytic = result.forces_hartree_per_angstrom
    assert analytic is not None
    assert np.linalg.norm(analytic.sum(axis=0)) <= 1.0e-8
    for step in (1.0e-4, 5.0e-5):
        finite_difference = np.empty_like(analytic)
        for atom_index in range(len(atoms)):
            for axis in range(3):
                plus = atoms.copy()
                minus = atoms.copy()
                plus.positions[atom_index, axis] += step
                minus.positions[atom_index, axis] -= step
                finite_difference[atom_index, axis] = -(
                    provider.evaluate(plus).energy_hartree
                    - provider.evaluate(minus).energy_hartree
                ) / (2.0 * step)
        error = analytic - finite_difference
        assert np.sqrt(np.mean(error**2)) <= 5.0e-6
        assert np.max(np.abs(error)) <= 2.0e-5
