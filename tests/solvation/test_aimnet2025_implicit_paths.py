from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.calculator import BatchResult
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
    _ATOM_IDENTITY_ARRAY,
)
from maple.function.dispatcher.curvature import bofill_update
from maple.function.dispatcher.implicit_path import validate_path_parameters
from maple.function.dispatcher.irc.algorithm.gs import GS, GSParams
from maple.function.dispatcher.ts.algorithm.neb import NEB, NEBParams, improved_tangent
from maple.function.dispatcher.ts.algorithm.string import GSM, GSMParams
from maple.function.utility import Molecules
from maple.function.read.command_control import CommandControl


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.get_positions().reshape(-1)
        energy = float(x @ x)
        self.results = {"energy": energy, "free_energy": energy, "forces": (-2.0 * x).reshape(-1, 3)}

    def get_hessian(self, atoms):
        n3 = 3 * len(atoms)
        hessian = np.eye(n3)
        hessian[-1, -1] = -1.0
        return hessian

    def calculate_many(self, images, properties=("energy", "forces")):
        energies, forces = [], []
        for image in images:
            x = image.get_positions().reshape(-1)
            energies.append(float(x @ x))
            forces.append((-2.0 * x).reshape(-1, 3))
        return BatchResult(energies=energies, forces=forces)


class LinearYCalculator(QuadraticCalculator):
    def calculate_many(self, images, properties=("energy", "forces")):
        energies = [float(image.get_positions()[:, 1].sum()) for image in images]
        forces = []
        for image in images:
            force = np.zeros((len(image), 3), dtype=np.float64)
            force[:, 1] = -1.0
            forces.append(force)
        return BatchResult(energies=energies, forces=forces)


def _two_atom_endpoints():
    left = Atoms("HH", positions=[[-0.5, 0, 0], [0.5, 0, 0]])
    right = Atoms("HH", positions=[[-0.8, 0, 0], [0.8, 0, 0]])
    calculator = QuadraticCalculator()
    left.calc = right.calc = calculator
    return left, right


def test_bofill_null_secant_preserves_finite_symmetric_hessian():
    hessian = np.diag([1.0, 2.0])
    step = np.array([0.2, -0.1])
    result = bofill_update(hessian, step, hessian @ step)
    np.testing.assert_array_equal(result, hessian)
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "params, error",
    [
        ({"neb": {"max_iter": True}}, "integer"),
        ({"neb": {"max_iter": -1}}, "nonnegative"),
        ({"neb": {"step0": float("nan")}}, "finite"),
        ({"neb": {"grow_step": 0.1}}, "Unknown"),
        ({"string": {"refine": "stringts"}}, "refine=None"),
    ],
)
def test_path_parameter_validation_rejects_invalid_or_cross_method_keys(params, error):
    method = next(iter(params))
    cls = NEBParams if method == "neb" else GSMParams
    with pytest.raises((TypeError, ValueError), match=error):
        validate_path_parameters(method, params, cls)


@pytest.mark.parametrize("method", ["neb", "string"])
def test_command_boundary_accepts_only_exact_aimnet_torch_obc2_path(method):
    control = CommandControl.from_settings([
        "#model=aimnet2025(hessian=analytic)",
        f"#ts(method={method},n_images=3,max_iter=0)" if method == "neb" else f"#ts(method={method},n_images=3,max_iter_grow=0,refine=None)",
        "#charge(source=mol2,mode=fixed,geometry=keep)",
        "#solv(implicit=water,method=gb,provider=torch,model=obc2,profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
    ])
    assert control.task == "ts"
    assert control.params["method"] == method


def test_command_boundary_rejects_cross_method_key_before_model_work():
    with pytest.raises(ValueError, match="Unknown TS parameter"):
        CommandControl.from_settings([
            "#model=aimnet2025(hessian=analytic)",
            "#ts(method=neb,n_images=3,max_iter_grow=0)",
            "#charge(source=mol2,mode=fixed,geometry=keep)",
            "#solv(implicit=water,method=gb,provider=torch,model=obc2,profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
        ])


def test_command_boundary_accepts_exact_implicit_gs_irc():
    control = CommandControl.from_settings([
        "#model=aimnet2025(hessian=analytic)",
        "#irc(method=gs,max_steps=0,project_rigid_modes=true)",
        "#charge(source=mol2,mode=fixed,geometry=keep)",
        "#solv(implicit=water,method=gb,provider=torch,model=obc2,profile=obc2-mbondi2,nonpolar=none,experimental=true)",
    ])
    assert control.task == "irc"
    assert control.params["project_rigid_modes"] is True


def test_gs_zero_budget_returns_exact_input_ts_once_and_restores_geometry(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.2]])
    atoms.calc = QuadraticCalculator()
    original = atoms.get_positions().copy()
    result = GS(
        atoms,
        str(tmp_path / "gs.out"),
        params=GSParams(max_steps=0, write_traj=False, print_each=False),
    ).run()
    assert result["converged"] is False
    assert result["termination_reason"] == "maximum_steps"
    assert result["forward"]["records"] == result["backward"]["records"] == []
    assert len(result["ordered_path"]) == 1
    np.testing.assert_array_equal(result["ordered_path"][0]["x"], original)
    np.testing.assert_array_equal(atoms.get_positions(), original)


def test_gs_rejects_nonpositive_masses_before_displacement(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.2]], masses=[0.0])
    atoms.calc = QuadraticCalculator()
    with pytest.raises(ValueError, match="positive atomic masses"):
        GS(atoms, str(tmp_path / "gs.out"), params=GSParams(max_steps=0)).run()


def test_string_endpoint_step_is_internal_3n_not_centroid_translation(tmp_path):
    left, right = _two_atom_endpoints()
    string = GSM(str(tmp_path / "string.out"), left, right, paras={"string": {"n_images": 3}})
    guide = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    before = left.get_positions().copy()
    string._endpoint_direct_step(left, guide, 0.2)
    displacement = left.get_positions() - before
    np.testing.assert_allclose(displacement.mean(axis=0), 0.0, atol=1e-15)
    assert np.linalg.norm(displacement) == pytest.approx(0.2)


def test_string_refine_none_returns_path_without_prfo(tmp_path, monkeypatch):
    left, right = _two_atom_endpoints()
    string = GSM(
        str(tmp_path / "string.out"),
        left,
        right,
        paras={"string": {"n_images": 3, "max_iter_grow": 0, "refine": None}},
    )

    def predictable(last, direction, _energy, _label):
        trial = last.copy()
        trial.calc = last.calc
        trial.set_positions(last.get_positions() + 0.05 * direction.reshape(-1, 3))
        return trial, 0.05, 0.0

    monkeypatch.setattr(string, "_predict_and_relax_one", predictable)
    monkeypatch.setattr(
        string,
        "restart_run",
        lambda *_args, **_kwargs: pytest.fail("refine=None must not invoke PRFO"),
    )
    result = string.run()
    assert result["converged"] is False
    assert result["termination_reason"] in {"growth_limit", "merged_path"}
    assert len(result["images"]) == 3


def test_neb_zero_budget_reports_force_based_nonconvergence_and_one_batch(tmp_path):
    left, right = _two_atom_endpoints()
    calculator = LinearYCalculator()
    left.calc = right.calc = calculator
    calls = 0
    original = calculator.calculate_many

    def counted(images, properties=("energy", "forces")):
        nonlocal calls
        calls += 1
        return original(images, properties)

    calculator.calculate_many = counted
    result = NEB(
        str(tmp_path / "neb.out"),
        Molecules([left, right]),
        paras={"neb": {"n_images": 1, "max_iter": 0, "ifidpp": 0, "refine": None}},
    ).run()
    assert result["iterations"] == 0
    assert result["converged"] is False
    assert result["termination_reason"] == "maximum_iterations"
    assert calls == 1


def test_neb_degenerate_tangent_refuses_to_invent_an_axis():
    positions = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="degenerate"):
        improved_tangent(positions, positions, positions, 0.0, 0.0, 0.0)


def test_bind_compatible_geometries_is_atomic_on_invalid_second_image():
    first, second = _two_atom_endpoints()
    correction = ImplicitSolvationCorrection.__new__(ImplicitSolvationCorrection)
    correction._frozen_atomic_numbers = (1, 1)
    correction._frozen_atom_identity = (10, 20)
    correction._frozen_mol2_topology_signature = None
    correction._frozen_masses = tuple(float(value) for value in first.get_masses())
    correction._frozen_charges = (0.0, 0.0)
    first.set_initial_charges([0.0, 0.0])
    second.set_initial_charges([0.1, -0.1])
    with pytest.raises(ValueError, match="charge vector"):
        correction.bind_compatible_geometries([first, second])
    assert _ATOM_IDENTITY_ARRAY not in first.arrays
    assert _ATOM_IDENTITY_ARRAY not in second.arrays


def test_bound_path_rejects_live_charge_and_mass_mutation():
    first, second = _two_atom_endpoints()
    correction = ImplicitSolvationCorrection.__new__(ImplicitSolvationCorrection)
    correction._frozen_atomic_numbers = (1, 1)
    correction._frozen_atom_identity = (10, 20)
    correction._frozen_mol2_topology_signature = None
    correction._frozen_masses = tuple(float(value) for value in first.get_masses())
    correction._frozen_charges = (0.0, 0.0)
    for image in (first, second):
        image.set_initial_charges([0.0, 0.0])
        image.info.update(charge=0, mult=1)
    correction.bind_compatible_geometries([first, second])
    second.set_initial_charges([0.1, -0.1])
    with pytest.raises(ValueError, match="frozen charge vector"):
        correction._validate_bound_atoms(second)
    second.set_initial_charges([0.0, 0.0])
    second.set_masses([1.0, 2.0])
    with pytest.raises(ValueError, match="positive mass vector"):
        correction._validate_bound_atoms(second)


def test_v5_receipt_pointer_is_unqualified_and_v4_remains_historical():
    import inspect

    from maple.function.calculator import _independent_obc2_qualified_cell as pointer
    from maple.function.calculator import set_calculator as runtime

    assert pointer.QUALIFICATION_RECEIPT_FILENAME.endswith("_v5.json")
    assert pointer.QUALIFICATION_RECEIPT_SHA256 is None
    inventory_source = inspect.getsource(runtime._installed_runtime_source_inventory)
    assert "independent_obc2_analytic_qualification_v4.json" in inventory_source
    assert "independent_obc2_analytic_qualification_v5.json" in inventory_source
