from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

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
from maple.function.dispatcher.implicit_path import (
    bind_admitted_implicit_path,
    validate_path_parameters,
)
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
            force[:, 1] = np.linspace(-1.0, 1.0, len(image))
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


PATH_PARAMETER_CASES = {
    "gs": {
        "target_mode": (2, 0),
        "step_length_bohr": (0.2, 0.0),
        "max_steps": (0, -1),
        "max_micro_cycles": (0, -1),
        "micro_step_thresh": (0.002, 0.0),
        "hessian_recalc": (3, -1),
        "hessian_update": ("bfgs", "sr1"),
        "f_max_th": (0.003, 0.0),
        "f_rms_th": (0.001, 0.0),
        "print_each": (False, 0),
        "write_traj": (False, 0),
        "project_rigid_modes": (True, 1),
    },
    "neb": {
        "n_images": (1, 0),
        "k_min": (0.02, 0.0),
        "k_max": (0.4, 0.0),
        "use_dynamic_k": (False, 0),
        "k_decay": (0.25, 0.0),
        "max_iter": (0, -1),
        "lbfgs_m": (5, 0),
        "step0": (0.01, 0.0),
        "ifidpp": (0, 2),
        "neb_f_max_th": (0.01, 0.0),
        "neb_f_rms_th": (0.006, 0.0),
        "initial_opt": (True, 1),
        "refine": (None, "cineb"),
    },
    "string": {
        "grow_step": (0.2, 0.0),
        "grow_step_min": (0.01, 0.0),
        "grow_backtrack": (0.25, 1.0),
        "grow_relax_iters": (0, -1),
        "grow_relax_tol": (0.003, 0.0),
        "grow_energy_tol": (0.002, 0.0),
        "pause_imbalance": (0, -1),
        "refine_end_every": (0, -1),
        "merge_threshold": (0.4, 0.0),
        "n_images": (2, 1),
        "refine": (None, "stringts"),
        "max_iter_grow": (0, -1),
        "log_every": (2, 0),
        "guide_conn_min": (0.5, 1.1),
        "guide_d_switch": (2.0, 0.0),
        "guide_inc_trigger": (0, -1),
    },
}

INACTIVE_PATH_PARAMETERS = {
    "gs": set(),
    "neb": {"cineb_f_max_th", "cineb_f_rms_th", "cilbfgs_m", "cistep0"},
    "string": {
        "max_iter_relax", "string_f_max_th", "string_f_rms_th",
        "lbfgs_memory", "lbfgs_curvature", "lbfgs_max_step", "reparam_every",
        "cistring_f_max_th", "cistring_f_rms_th",
    },
}


@pytest.mark.parametrize("method, params_class", [
    ("gs", GSParams), ("neb", NEBParams), ("string", GSMParams),
])
def test_path_parameter_schema_covers_every_dataclass_field(method, params_class):
    import maple.function.dispatcher.implicit_path as contract

    schema = contract.PATH_PARAMETER_SCHEMAS[method]
    assert set(schema) == {field.name for field in fields(params_class)}
    assert {name for name, rule in schema.items() if rule.inactive_reason} == (
        INACTIVE_PATH_PARAMETERS[method]
    )


@pytest.mark.parametrize("method, params_class", [
    ("gs", GSParams), ("neb", NEBParams), ("string", GSMParams),
])
def test_path_parameter_defaults_remain_valid_and_implicit(method, params_class):
    assert validate_path_parameters(method, {method: {}}, params_class) == {}


@pytest.mark.parametrize(
    "method,key,valid,invalid",
    [
        (method, key, values[0], values[1])
        for method, cases in PATH_PARAMETER_CASES.items()
        for key, values in cases.items()
    ],
)
def test_each_active_path_parameter_rule_accepts_valid_and_rejects_invalid(
    method, key, valid, invalid
):
    params_class = {"gs": GSParams, "neb": NEBParams, "string": GSMParams}[method]
    assert validate_path_parameters(
        method, {method: {key: valid}}, params_class
    ) == {key: valid}
    with pytest.raises((TypeError, ValueError)):
        validate_path_parameters(method, {method: {key: invalid}}, params_class)


@pytest.mark.parametrize(
    "method,key",
    [
        (method, key)
        for method, keys in INACTIVE_PATH_PARAMETERS.items()
        for key in sorted(keys)
    ],
)
def test_each_inactive_path_parameter_rejects_explicit_default(method, key):
    params_class = {"gs": GSParams, "neb": NEBParams, "string": GSMParams}[method]
    default = getattr(params_class(), key)
    with pytest.raises(ValueError, match="inactive"):
        validate_path_parameters(method, {method: {key: default}}, params_class)


@pytest.mark.parametrize("method, params_class", [
    ("gs", GSParams), ("neb", NEBParams), ("string", GSMParams),
])
def test_every_active_path_rule_enforces_its_declared_type(method, params_class):
    import maple.function.dispatcher.implicit_path as contract

    for key, rule in contract.PATH_PARAMETER_SCHEMAS[method].items():
        if rule.inactive_reason:
            continue
        if rule.kind in {"int", "optional_int"}:
            invalid_values = (True,)
            error = "integer"
        elif rule.kind == "real":
            invalid_values = (float("nan"), float("inf"), float("-inf"))
            error = "finite"
        elif rule.kind == "bool":
            invalid_values = (1,)
            error = "boolean"
        elif rule.kind == "enum":
            invalid_values = (1,)
            error = "string"
        elif rule.kind == "none":
            invalid_values = ("enabled",)
            error = "requires"
        else:
            raise AssertionError(f"uncovered rule kind: {rule.kind}")
        for invalid in invalid_values:
            with pytest.raises((TypeError, ValueError), match=error):
                validate_path_parameters(
                    method, {method: {key: invalid}}, params_class
                )


def test_optional_hessian_recalculation_preserves_explicit_none():
    assert validate_path_parameters(
        "gs", {"gs": {"hessian_recalc": None}}, GSParams
    ) == {"hessian_recalc": None}


@pytest.mark.parametrize(
    "task_line,error",
    [
        ("#ts(method=neb,n_images=1,max_iter=0,cistep0=0.005,refine=None)", "inactive"),
        ("#ts(method=string,n_images=3,max_iter_grow=0,max_iter_relax=300,refine=None)", "inactive"),
        ("#ts(method=string,n_images=3,max_iter_grow=0,grow_backtrack=1,refine=None)", "grow_backtrack"),
        ("#ts(method=string,n_images=3,max_iter_grow=0,guide_d_switch=0,refine=None)", "guide_d_switch"),
    ],
)
def test_command_boundary_rejects_inactive_and_invalid_path_parameters(task_line, error):
    with pytest.raises((TypeError, ValueError), match=error):
        CommandControl.from_settings([
            "#model=aimnet2025(hessian=analytic)", task_line,
            "#charge(source=mol2,mode=fixed,geometry=keep)",
            "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
            "profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
        ])


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


def test_gs_force_convergence_precedes_step_too_small(tmp_path, monkeypatch):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = QuadraticCalculator()
    gs = GS(
        atoms,
        str(tmp_path / "gs.out"),
        params=GSParams(
            step_length_bohr=1e-16,
            max_steps=1,
            max_micro_cycles=0,
            write_traj=False,
            print_each=False,
        ),
    )
    gs._D = np.ones(3)
    gs._step_len_mw = gs.p.step_length_bohr
    force_calls = iter([
        np.ones(3),
        np.ones(3),
        np.zeros(3),
    ])
    monkeypatch.setattr(gs, "_energy_forces_from_mw", lambda _coords: (0.0, next(force_calls)))
    monkeypatch.setattr(gs, "_get_hessian_cart", lambda: np.eye(3))

    result = gs._one_side(
        forward=True,
        sign=1.0,
        q_ts_cart=np.zeros(3),
        v_neg_mw=np.array([1.0, 0.0, 0.0]),
        E_ts=0.0,
    )

    assert result["converged"] is True
    assert result["termination_reason"] == "force_converged"


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
    calculator.solvent_correction = SimpleNamespace(_path_frozen_masses=(1.0, 1.0))
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
    correction._path_frozen_masses = None
    correction._frozen_charges = (0.0, 0.0)
    first.set_initial_charges([0.0, 0.0])
    second.set_initial_charges([0.1, -0.1])
    first.info.update(charge=0, mult=1)
    second.info.update(charge=0, mult=1)
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
    correction._path_frozen_masses = None
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


def test_neb_improved_tangent_uses_absolute_extrema_weights():
    rm1 = np.array([[0.0, 0.0, 0.0]])
    current = np.array([[1.0, 0.0, 0.0]])
    rp1 = np.array([[1.0, 2.0, 0.0]])
    expected_max = np.array([1.0, 3.0, 0.0])
    expected_max /= np.linalg.norm(expected_max)
    np.testing.assert_allclose(
        improved_tangent(rm1, current, rp1, 0.0, 3.0, 1.0), expected_max
    )
    expected_min = np.array([3.0, 4.0, 0.0])
    expected_min /= np.linalg.norm(expected_min)
    np.testing.assert_allclose(
        improved_tangent(rm1, current, rp1, 3.0, 0.0, 2.0), expected_min
    )


@pytest.mark.parametrize(
    "method, params, error",
    [
        ("string", {"string": {"n_images": 1}}, "at least 2"),
        ("string", {"string": {"log_every": 0}}, "positive"),
        ("neb", {"neb": {"ifidpp": 2}}, "0 or 1"),
        ("gs", {"gs": {"hessian_recalc": True}}, "integer"),
        ("neb", {"neb": {"k_min": 0.4, "k_max": 0.3}}, "must not exceed"),
        ("string", {"string": {"grow_step_min": 0.4, "grow_step": 0.3}}, "must not exceed"),
    ],
)
def test_path_parameter_validation_rejects_method_specific_invalid_values(method, params, error):
    cls = {"gs": GSParams, "neb": NEBParams, "string": GSMParams}[method]
    with pytest.raises((TypeError, ValueError), match=error):
        validate_path_parameters(method, params, cls)


@pytest.mark.parametrize("method", ["neb", "string"])
def test_command_boundary_requires_explicit_analytic_hessian_for_all_new_paths(method):
    task = (
        f"#ts(method={method},n_images=3,max_iter=0)"
        if method == "neb"
        else f"#ts(method={method},n_images=3,max_iter_grow=0,refine=None)"
    )
    with pytest.raises(ValueError, match="hessian=analytic"):
        CommandControl.from_settings([
            "#model=aimnet2025",
            task,
            "#charge(source=mol2,mode=fixed,geometry=keep)",
            "#solv(implicit=water,method=gb,provider=torch,model=obc2,profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
        ])


def test_gas_neb_and_string_evaluate_distinct_per_image_calculators():
    left = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    right = Atoms("H", positions=[[1.0, 0.0, 0.0]])
    left.calc = QuadraticCalculator()
    right.calc = QuadraticCalculator()
    neb_energies, _ = NEB._evaluate_images([left, right])
    string_energies, _ = GSM._evaluate_images([left, right])
    assert neb_energies == string_energies == [0.0, 1.0]


@pytest.mark.parametrize("algorithm", [NEB, GSM])
def test_gas_path_with_shared_calculator_keeps_per_image_ase_evaluation(algorithm):
    left, right = _two_atom_endpoints()
    calculator = left.calc
    calls = 0

    def forbidden_batch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("ordinary gas paths must not use calculate_many")

    calculator.calculate_many = forbidden_batch
    energies, forces = algorithm._evaluate_images([left, right])
    assert energies == pytest.approx([0.5, 1.28])
    assert [force.shape for force in forces] == [(2, 3), (2, 3)]
    assert calls == 0


def test_unbound_single_point_does_not_freeze_isotope_masses():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    atoms.set_initial_charges([0.0])
    atoms.new_array(_ATOM_IDENTITY_ARRAY, np.array([7], dtype=np.int64))
    correction = ImplicitSolvationCorrection.__new__(ImplicitSolvationCorrection)
    correction._frozen_atomic_numbers = (1,)
    correction._frozen_atom_identity = (7,)
    correction._frozen_mol2_topology_signature = None
    correction._frozen_charges = (0.0,)
    correction._path_frozen_masses = None
    atoms.set_masses([2.014])
    correction._validate_bound_atoms(atoms)


def test_direct_path_binding_rejects_invalid_second_image_without_tokens():
    first, second = _two_atom_endpoints()
    for image in (first, second):
        image.info.update(charge=0, mult=1)
        image.set_initial_charges([0.0, 0.0])
    second.positions[0, 0] = np.nan
    correction = ImplicitSolvationCorrection.__new__(ImplicitSolvationCorrection)
    correction._frozen_atomic_numbers = (1, 1)
    correction._frozen_atom_identity = (10, 20)
    correction._frozen_mol2_topology_signature = None
    correction._frozen_charges = (0.0, 0.0)
    correction._path_frozen_masses = None
    with pytest.raises(ValueError, match="finite Cartesian"):
        correction.bind_compatible_geometries([first, second])
    assert _ATOM_IDENTITY_ARRAY not in first.arrays
    assert _ATOM_IDENTITY_ARRAY not in second.arrays


@pytest.mark.parametrize("container", [list, Molecules])
@pytest.mark.parametrize("method", ["neb", "string"])
def test_transition_state_path_containers_reach_algorithm_boundary(
    tmp_path, monkeypatch, container, method
):
    import maple.function.dispatcher.ts.ts as ts_module
    import maple.function.dispatcher.ts.algorithm as algorithms

    left, right = _two_atom_endpoints()
    left.calc.solvent_correction = SimpleNamespace()
    monkeypatch.setattr(ts_module, "bind_admitted_implicit_path", lambda *_args, **_kwargs: True)
    expected = {"method": method}
    algorithm = algorithms.NEB if method == "neb" else algorithms.GSM
    monkeypatch.setattr(algorithm, "run", lambda _self: expected)
    params = {"method": method, "n_images": 2, "refine": None}
    params["max_iter" if method == "neb" else "max_iter_grow"] = 0
    result = ts_module.TransitionState(
        str(tmp_path / f"{method}.out"), container([left, right]), params, method
    ).run()
    assert result is expected


@pytest.mark.parametrize("method", ["neb", "string"])
def test_implicit_path_log_describes_path_generation_not_ts_candidate_search(
    tmp_path, monkeypatch, method
):
    import maple.function.dispatcher.ts.ts as ts_module
    import maple.function.dispatcher.ts.algorithm as algorithms

    left, right = _two_atom_endpoints()
    left.calc.solvent_correction = SimpleNamespace()
    monkeypatch.setattr(ts_module, "bind_admitted_implicit_path", lambda *_args, **_kwargs: True)
    algorithm = algorithms.NEB if method == "neb" else algorithms.GSM
    monkeypatch.setattr(algorithm, "run", lambda _self: {"status": "done"})
    params = {"method": method, "n_images": 2, "refine": None}
    params["max_iter" if method == "neb" else "max_iter_grow"] = 0
    output = tmp_path / f"{method}.out"

    ts_module.TransitionState(str(output), [left, right], params, method).run()

    log = output.read_text()
    assert "path-generation capability" in log
    assert "TS candidate-search capability" not in log


def test_transition_state_string_rejects_more_than_two_inputs(tmp_path):
    import maple.function.dispatcher.ts.ts as ts_module

    left, right = _two_atom_endpoints()
    third = right.copy()
    third.calc = right.calc
    with pytest.raises(ValueError, match="exactly two"):
        ts_module.TransitionState(
            str(tmp_path / "string.out"), [left, right, third],
            {"method": "string", "n_images": 3, "max_iter_grow": 0}, "string"
        ).run()


def test_string_first_bilateral_growth_consumes_one_round(tmp_path, monkeypatch):
    left, right = _two_atom_endpoints()
    string = GSM(
        str(tmp_path / "string.out"), left, right,
        paras={"string": {"n_images": 3, "max_iter_grow": 1, "merge_threshold": 0.01}},
    )
    calls = 0

    def unchanged(last, _direction, _energy, _label):
        nonlocal calls
        calls += 1
        return string._copy_with_inherit(last), 0.0, 0.0

    monkeypatch.setattr(string, "_predict_and_relax_one", unchanged)
    result = string.run()
    assert calls == 2
    assert result["growth_iterations"] == 1
    assert result["termination_reason"] == "growth_limit"
    assert (tmp_path / "string_gsm_equal_arc_candidate.xyz").is_file()
    assert not (tmp_path / "string_gsm_mep.xyz").exists()


def test_cache_context_restores_absence_values_and_exceptions():
    from maple.function.dispatcher.implicit_path import preserve_path_calculator_caches

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = QuadraticCalculator()
    atoms.calc = calculator
    del calculator.atoms
    calculator.results = {"energy": 7.0}
    calculator.solvation_result = "before"
    assert not hasattr(calculator, "_evaluated_correction")
    with pytest.raises(RuntimeError, match="injected"):
        with preserve_path_calculator_caches([atoms], enabled=True):
            calculator.results = {"energy": 9.0}
            calculator.solvation_result = "during"
            calculator._evaluated_correction = object()
            calculator.atoms = atoms.copy()
            raise RuntimeError("injected")
    assert calculator.results == {"energy": 7.0}
    assert calculator.solvation_result == "before"
    assert not hasattr(calculator, "_evaluated_correction")
    assert not hasattr(calculator, "atoms")


def test_gs_one_macrostep_counts_one_iteration_and_keeps_exact_ts(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.2]])
    atoms.calc = QuadraticCalculator()
    input_positions = atoms.get_positions().copy()
    result = GS(
        atoms, str(tmp_path / "gs.out"),
        params=GSParams(max_steps=1, max_micro_cycles=0, write_traj=False, print_each=False),
    ).run()
    assert result["forward"]["iterations"] == 1
    assert result["backward"]["iterations"] == 1
    assert result["iterations"] == 2
    matches = [np.array_equal(record["x"], input_positions) for record in result["ordered_path"]]
    assert sum(matches) == 1
    assert "forward:" in result["termination_reason"]


def test_gas_irc_keeps_normal_caller_geometry_and_cache_behavior(tmp_path, monkeypatch):
    import maple.function.dispatcher.irc.algorithm as algorithms
    from maple.function.dispatcher.irc.irc import IRC

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.2]])
    atoms.calc = QuadraticCalculator()
    original_calc = atoms.calc

    class CapturingGS:
        def __init__(self, received, **_kwargs):
            assert received is atoms
            self.received = received

        def run(self):
            self.received.positions[0, 0] = 4.0
            self.received.calc.results = {"energy": 12.0}
            return {"status": "gas"}

    monkeypatch.setattr(algorithms, "GS", CapturingGS)
    result = IRC({"gs": {"max_steps": 1}}, str(tmp_path / "gas.out"), atoms, "gs").run()
    assert result == {"status": "gas"}
    assert atoms.positions[0, 0] == 4.0
    assert atoms.calc is original_calc
    assert atoms.calc.results == {"energy": 12.0}


def test_final_implicit_preflight_rejects_postprocessed_constraint_before_metadata(tmp_path):
    from ase.constraints import FixAtoms
    from maple.function.read.input_reader import InputReader

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info.update(charge=0, mult=1)
    atoms.set_initial_charges([0.0])
    atoms.set_constraint(FixAtoms(indices=[0]))
    reader = InputReader()
    reader.input = str(tmp_path / "input.inp")
    (tmp_path / "input.inp").write_text("synthetic\n")
    reader.output = str(tmp_path / "input.out")
    reader.command_control = SimpleNamespace(
        task="ts",
        params={
            "method": "neb",
            "charge": {"source": "maple", "geometry": "keep"},
        },
    )
    with pytest.raises(ValueError, match="constraints"):
        reader._validate_final_implicit_images(
            [atoms, atoms.copy()], {"implicit": "water"}
        )
    for image in (atoms,):
        assert "_maple_charge_options" not in image.info
        assert "_maple_solvation_options" not in image.info


def _preflight_reader(tmp_path, task, method, *, inner=None):
    from maple.function.read.input_reader import InputReader

    reader = InputReader()
    reader.input = str(tmp_path / f"{task}-{method}.inp")
    Path(reader.input).write_text("synthetic\n")
    reader.output = str(tmp_path / f"{task}-{method}.out")
    reader.command_control = SimpleNamespace(
        task=task,
        params={
            "method": method,
            "charge": {"source": "maple", "geometry": "keep"},
            "solv": {"inner": inner},
        },
    )
    return reader


def _implicit_atom(*, constrained=False):
    atom = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atom.info.update(charge=0, mult=1)
    atom.set_initial_charges([0.0])
    if constrained:
        from ase.constraints import FixAtoms
        atom.set_constraint(FixAtoms(indices=[0]))
    return atom


@pytest.mark.parametrize(
    "task,method,count,error",
    [
        ("irc", "gs", 2, "exactly one"),
        ("ts", "neb", 1, "at least two"),
        ("ts", "string", 1, "exactly two"),
        ("ts", "string", 3, "exactly two"),
    ],
)
def test_new_implicit_path_preflight_uses_requested_task_and_method(
    tmp_path, task, method, count, error
):
    reader = _preflight_reader(tmp_path, task, method)
    images = [_implicit_atom() for _ in range(count)]
    value = images[0] if count == 1 else images
    with pytest.raises(ValueError, match=error):
        reader._validate_final_implicit_images(value, {"implicit": "water"})


@pytest.mark.parametrize("task,method", [("opt", ""), ("freq", "mw"), ("ts", "prfo")])
def test_legacy_single_geometry_implicit_tasks_keep_constraints(
    tmp_path, task, method
):
    reader = _preflight_reader(tmp_path, task, method)
    atom = _implicit_atom(constrained=True)
    reader._validate_final_implicit_images(atom, {"implicit": "water"})
    assert atom.constraints


def test_prebuilt_inner_component_guard_remains_before_postprocessing(tmp_path, monkeypatch):
    from maple.function.read.input_reader import InputReader

    source = (
        Path(__file__).parent
        / "data" / "amber_gb_reference" / "mol2" / "mobley_3053621.mol2"
    )
    input_path = tmp_path / "prebuilt.inp"
    input_path.write_text("\n".join([
        "#model=ani2x", "#opt", "#charge(source=mol2,mode=fixed,geometry=keep)",
        "#solv(implicit=water,inner=prebuilt,method=gb,provider=openmm,model=obc2,experimental=true)",
        "", "0 1", f"MOL2 {source}", "", "POST setcharge 0", "",
    ]))
    post_calls = 0

    def forbidden_post(*_args, **_kwargs):
        nonlocal post_calls
        post_calls += 1

    monkeypatch.setattr(InputReader, "post_processing_command", forbidden_post)
    with pytest.raises(ValueError, match="at least two connected components"):
        InputReader()(str(input_path), str(tmp_path / "prebuilt.out"))
    assert post_calls == 0


def test_gs_success_result_has_no_unreachable_numerical_failure_status_branch():
    import inspect
    from maple.function.dispatcher.irc.algorithm.gs import GS

    source = inspect.getsource(GS.run)
    assert '== "numerical_failure"' not in source


@pytest.mark.parametrize("method", ["neb", "string"])
def test_direct_implicit_path_requires_analytic_model_enrollment(method):
    from maple.function.calculator.aimnet._aimnet2025_calculator import (
        AIMNet2025Calculator,
    )

    correction = SimpleNamespace(
        solvation_options={
            "provider": "torch",
            "model": "obc2",
            "profile": "obc2-mbondi2",
            "nonpolar": "ace",
        },
        mode="fixed",
        inner_mode=None,
        analytic_task_derivatives_admitted=True,
        bind_compatible_geometries=lambda _images: pytest.fail(
            "binding must not start before analytic enrollment passes"
        ),
    )
    calculator = AIMNet2025Calculator.__new__(AIMNet2025Calculator)
    calculator.solvent_correction = correction
    calculator.analytic_implicit_derivatives_admitted = True
    calculator.hessian = None
    left, right = _two_atom_endpoints()
    left.calc = right.calc = calculator
    with pytest.raises(ValueError, match="analytic enrolled"):
        bind_admitted_implicit_path([left, right], method)


def _write_two_mol2_neb_input(tmp_path, first, second):
    input_path = tmp_path / "path.inp"
    input_path.write_text(
        "\n".join([
            "#model=aimnet2025(hessian=analytic)",
            "#ts(method=neb,n_images=1,max_iter=0,ifidpp=0,refine=None)",
            "#charge(source=mol2,mode=fixed,geometry=keep)",
            "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
            "profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
            "", "0 1", f"MOL2 {first}", "", "0 1", f"MOL2 {second}", "",
        ])
    )
    return input_path


def test_engine_rejects_missing_neb_endpoint_before_model_construction(
    tmp_path, monkeypatch
):
    from maple.function.engine import engine

    source = (
        Path(__file__).parent
        / "data" / "amber_gb_reference" / "mol2" / "mobley_3053621.mol2"
    )
    input_path = tmp_path / "one-endpoint.inp"
    input_path.write_text("\n".join([
        "#model=aimnet2025(hessian=analytic)",
        "#ts(method=neb,n_images=1,max_iter=0,ifidpp=0,refine=None)",
        "#charge(source=mol2,mode=fixed,geometry=keep)",
        "#solv(implicit=water,method=gb,provider=torch,model=obc2,"
        "profile=obc2-mbondi2,nonpolar=ace,experimental=true)",
        "", "0 1", f"MOL2 {source}", "",
    ]))
    job = engine()
    model_calls = 0

    def forbidden_model(*_args):
        nonlocal model_calls
        model_calls += 1

    monkeypatch.setattr(job, "_mlp_initiator", forbidden_model)
    with pytest.raises(ValueError, match="at least two endpoint"):
        job(str(input_path), str(tmp_path / "one-endpoint.out"))
    assert model_calls == 0


def test_engine_preflights_invalid_second_image_before_model_construction(
    tmp_path, monkeypatch
):
    from maple.function.engine import engine

    data = Path(__file__).parent / "data" / "amber_gb_reference" / "mol2"
    input_path = _write_two_mol2_neb_input(
        tmp_path, data / "mobley_3053621.mol2", data / "mobley_2725215.mol2"
    )
    job = engine()
    model_calls = 0

    def forbidden_model(*_args):
        nonlocal model_calls
        model_calls += 1

    monkeypatch.setattr(job, "_mlp_initiator", forbidden_model)
    with pytest.raises(ValueError, match="preserve atom"):
        job(str(input_path), str(tmp_path / "path.out"))
    assert model_calls == 0


def test_engine_prepares_one_model_provider_and_parameterization_for_path(
    tmp_path, monkeypatch
):
    from maple.function.engine import engine

    source = (
        Path(__file__).parent
        / "data" / "amber_gb_reference" / "mol2" / "mobley_3053621.mol2"
    )
    input_path = _write_two_mol2_neb_input(tmp_path, source, source)
    job = engine()
    calls = {"model": 0, "provider": 0, "preparation": 0, "binding": 0}
    calculator = SimpleNamespace()

    def initiate(*_args):
        calls["model"] += 1
        calls["provider"] += 1
        calls["preparation"] += 1

        def bind(images):
            calls["binding"] += 1
            assert len(images) == 2

        calculator.solvent_correction = SimpleNamespace(
            bind_compatible_geometries=bind
        )
        job.calulator = calculator

    def dispatch(_control, _jobtype, atoms, _output, extra=None):
        assert extra == {}
        assert all(image.calc is calculator for image in atoms.multiatoms)
        return {"status": "reached"}

    monkeypatch.setattr(job, "_mlp_initiator", initiate)
    monkeypatch.setattr(job, "_jobtype_dispatcher", dispatch)
    job(str(input_path), str(tmp_path / "path.out"))
    assert calls == {"model": 1, "provider": 1, "preparation": 1, "binding": 1}
