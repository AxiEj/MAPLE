from dataclasses import dataclass

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.result_contracts import PathStageResult, RunStatus
from maple.function.dispatcher.ts.algorithm import autoneb as autoneb_module
from maple.function.dispatcher.ts.algorithm.autoneb import (
    AutoNEB,
    AutoNEBParams,
    select_insertion_interval,
)


class _QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = float(atoms.positions[0, 0])
        energy = x * x
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        forces[0, 0] = -2.0 * x
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


def _image(x, calculator, *, role):
    atoms = Atoms("H", positions=[[x, 0.0, 0.0]])
    atoms.calc = calculator
    atoms.info.update(
        {
            "path_role": role,
            "source_identity": "sealed-source",
            "prepared_parameter_identity": "prepared-parameters",
            "topology_identity": "frozen-topology",
            "charge_identity": "frozen-charge",
        }
    )
    atoms.new_array("frozen_mass", np.array([1.0]))
    return atoms


def _stage_result(images, *, status, maximum, used, candidate=None):
    energies = [float(image.positions[0, 0] ** 2) for image in images]
    forces = [np.zeros((len(image), 3), dtype=np.float64) for image in images]
    return PathStageResult.capture(
        status=status,
        images=images,
        energies_hartree=energies,
        forces_hartree_per_angstrom=forces,
        budgets={"maximum_iterations": maximum, "used_iterations": used},
        termination_reason=status.value,
        candidate=candidate,
    )


@dataclass
class _FakeNEBParams:
    max_iter: int = 0
    neb_f_max_th: float = 0.0
    neb_f_rms_th: float = 0.0
    cineb_max_iter: int = 0
    cineb_f_max_th: float = 0.0
    cineb_f_rms_th: float = 0.0
    ci_displacement_tol: float = 0.0


def _install_stage_oracle(monkeypatch, statuses):
    calls = []
    remaining = iter(statuses)

    def run_neb_stage(images, *, params, stage_kind, output):
        status = next(remaining)
        calls.append((stage_kind, params, output, tuple(images)))
        maximum = params.max_iter if stage_kind == "base" else params.cineb_max_iter
        candidate = images[1] if stage_kind == "ci" else None
        return _stage_result(
            images,
            status=status,
            maximum=maximum,
            used=maximum,
            candidate=candidate,
        )

    monkeypatch.setattr(
        autoneb_module,
        "_load_neb_stage_api",
        lambda: (_FakeNEBParams, run_neb_stage),
        raising=False,
    )
    return calls


def test_pinned_ase_327_policy_selects_spatial_and_energy_dominant_intervals():
    spatial_positions = [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[3.0, 0.0, 0.0]]),
        np.array([[4.0, 0.0, 0.0]]),
    ]
    assert select_insertion_interval(
        spatial_positions,
        [0.0, 0.1, 1.0],
        space_energy_ratio=1.0,
    ) == (0, "space")
    assert select_insertion_interval(
        spatial_positions,
        [0.0, 0.1, 1.0],
        space_energy_ratio=2.0,
    ) == (1, "energy")


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_images", 4),
        ("ang_iter", 2),
        ("max_paths", 2),
        ("do_final_refine", True),
        ("max_steps", 0),
    ],
)
def test_obsolete_tree_and_phantom_budget_fields_are_rejected(tmp_path, field, value):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]
    with pytest.raises(ValueError, match=field):
        AutoNEB(str(tmp_path / "autoneb.out"), images, {field: value})


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"n_max": 2, "climb": True}, "interior"),
        ({"n_max": 1}, "at least"),
        ({"max_iter": -1}, "max_iter"),
        ({"fmax": np.nan}, "finite"),
        ({"climb": 1}, "boolean"),
        ({"interpolation": "ase"}, "interpolation"),
    ],
)
def test_parameter_domain_fails_closed(changes, match):
    with pytest.raises((TypeError, ValueError), match=match):
        AutoNEBParams(**changes)


def test_n_max_must_cover_the_supplied_path(tmp_path):
    calculator = _QuadraticCalculator()
    images = [
        _image(0.0, calculator, role="reactant"),
        _image(0.5, calculator, role="interior"),
        _image(1.0, calculator, role="product"),
    ]
    with pytest.raises(ValueError, match="initial image count"):
        AutoNEB(
            str(tmp_path / "autoneb.out"),
            images,
            {"n_max": 2, "climb": False},
        )


def test_zero_budget_is_unmoved_bounded_nonconvergence(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]
    before = [image.positions.copy() for image in images]
    monkeypatch.setattr(
        autoneb_module,
        "_load_neb_stage_api",
        lambda: pytest.fail("zero budget must not invoke the NEB stage kernel"),
        raising=False,
    )

    result = AutoNEB(
        str(tmp_path / "autoneb.out"),
        images,
        {"n_max": 4, "max_iter": 0, "climb": False},
    ).run()

    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert result.initial_image_count == 2
    assert result.final_image_count == 2
    assert result.insertion_count == 0
    assert result.base_stage_count == 0
    assert result.ci_stage is None
    assert result.certified_ts is False
    assert all(np.array_equal(image.positions, expected) for image, expected in zip(result.images, before))


def test_insertions_preserve_one_calculator_metadata_and_endpoint_roles(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    left = _image(0.0, calculator, role="reactant")
    right = _image(1.0, calculator, role="product")
    calls = _install_stage_oracle(monkeypatch, [RunStatus.CONVERGED, RunStatus.CONVERGED])

    result = AutoNEB(
        str(tmp_path / "autoneb.out"),
        [left, right],
        {"n_max": 4, "max_iter": 3, "climb": False, "interpolation": "linear"},
    ).run()

    assert result.status is RunStatus.CONVERGED
    assert result.insertion_count == 2
    assert result.base_stage_count == 2
    assert [call[0] for call in calls] == ["base", "base"]
    assert all(image.calc is calculator for image in result.images)
    assert result.images[0].info["path_role"] == "reactant"
    assert result.images[-1].info["path_role"] == "product"
    for image in result.images[1:-1]:
        assert image.info["path_role"] == "autoneb_inserted"
        assert image.info["source_identity"] == "sealed-source"
        assert image.info["prepared_parameter_identity"] == "prepared-parameters"
        assert image.info["topology_identity"] == "frozen-topology"
        assert image.info["charge_identity"] == "frozen-charge"
        assert np.array_equal(image.arrays["frozen_mass"], np.array([1.0]))


def test_nonconverged_growth_stage_stops_growth_and_blocks_ci(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]
    calls = _install_stage_oracle(monkeypatch, [RunStatus.BOUNDED_NONCONVERGED])

    result = AutoNEB(
        str(tmp_path / "autoneb.out"),
        images,
        {"n_max": 4, "max_iter": 2, "climb": True},
    ).run()

    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert result.final_image_count == 3
    assert result.insertion_count == 1
    assert result.base_stage_count == 1
    assert result.ci_stage is None
    assert [call[0] for call in calls] == ["base"]
    assert result.certified_ts is False


def test_ci_uses_separate_budget_only_after_all_growth_stages_converge(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]
    calls = _install_stage_oracle(
        monkeypatch,
        [RunStatus.CONVERGED, RunStatus.CONVERGED, RunStatus.BOUNDED_NONCONVERGED],
    )

    result = AutoNEB(
        str(tmp_path / "autoneb.out"),
        images,
        {
            "n_max": 4,
            "max_iter": 2,
            "fmax": 0.02,
            "rms_fmax": 0.01,
            "climb": True,
            "ci_max_iter": 7,
            "ci_fmax": 0.004,
            "ci_rms_fmax": 0.003,
            "ci_displacement_tol": 0.002,
        },
    ).run()

    assert [call[0] for call in calls] == ["base", "base", "ci"]
    assert calls[0][1].max_iter == 2
    assert calls[0][1].neb_f_max_th == pytest.approx(0.02)
    assert calls[0][1].neb_f_rms_th == pytest.approx(0.01)
    assert calls[-1][1].cineb_max_iter == 7
    assert calls[-1][1].cineb_f_max_th == pytest.approx(0.004)
    assert calls[-1][1].cineb_f_rms_th == pytest.approx(0.003)
    assert calls[-1][1].ci_displacement_tol == pytest.approx(0.002)
    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert result.ci_stage is not None
    assert result.candidate is not None
    assert result.certified_ts is False


def test_climb_false_never_calls_ci(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]
    calls = _install_stage_oracle(monkeypatch, [RunStatus.CONVERGED])

    result = AutoNEB(
        str(tmp_path / "autoneb.out"),
        images,
        {"n_max": 3, "max_iter": 2, "climb": False},
    ).run()

    assert [call[0] for call in calls] == ["base"]
    assert result.status is RunStatus.CONVERGED
    assert result.ci_stage is None
    assert result.certified_ts is False


def test_zero_ci_budget_is_separate_bounded_stage(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]
    calls = _install_stage_oracle(monkeypatch, [RunStatus.CONVERGED])

    result = AutoNEB(
        str(tmp_path / "autoneb.out"),
        images,
        {"n_max": 3, "max_iter": 2, "climb": True, "ci_max_iter": 0},
    ).run()

    assert [call[0] for call in calls] == ["base"]
    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert result.ci_stage is not None
    assert dict(result.ci_stage.budgets) == {
        "maximum_iterations": 0,
        "used_iterations": 0,
    }
    assert result.certified_ts is False


def test_stage_kernel_cannot_move_fixed_endpoint(monkeypatch, tmp_path):
    calculator = _QuadraticCalculator()
    images = [_image(0.0, calculator, role="reactant"), _image(1.0, calculator, role="product")]

    def run_neb_stage(stage_images, *, params, stage_kind, output):
        stage_images[0].positions[0, 0] = -1.0
        return _stage_result(
            stage_images,
            status=RunStatus.CONVERGED,
            maximum=params.max_iter,
            used=params.max_iter,
        )

    monkeypatch.setattr(
        autoneb_module,
        "_load_neb_stage_api",
        lambda: (_FakeNEBParams, run_neb_stage),
    )
    with pytest.raises(RuntimeError, match="fixed AutoNEB endpoint"):
        AutoNEB(
            str(tmp_path / "autoneb.out"),
            images,
            {"n_max": 3, "max_iter": 2, "climb": False},
        ).run()
