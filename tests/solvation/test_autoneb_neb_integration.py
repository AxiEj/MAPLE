from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.result_contracts import RunStatus
from maple.function.dispatcher.ts.algorithm.autoneb import AutoNEB


class _QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = float(atoms.positions[0, 0])
        energy = x * x
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        forces[0, 0] = -2.0 * x
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


class _PerpendicularCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = float(-np.sum(atoms.positions[:, 2]))
        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        forces[:, 2] = 1.0
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


def _endpoints(calculator):
    result = []
    for x, role in ((0.0, "reactant"), (1.0, "product")):
        image = Atoms("H", positions=[[x, 0.0, 0.0]], masses=[1.25])
        image.calc = calculator
        image.info.update(
            {
                "path_role": role,
                "source_identity": "analytic-autoneb-integration-v1",
                "prepared_parameter_identity": "analytic-parameters-v1",
                "topology_identity": "H",
                "charge_identity": "neutral",
            }
        )
        image.new_array("path_atom_identity", np.array([17], dtype=np.int64))
        result.append(image)
    return result


def test_real_neb_base_stage_returns_aligned_standard_autoneb_result(tmp_path):
    calculator = _QuadraticCalculator()
    output = tmp_path / "base.out"

    result = AutoNEB(
        str(output),
        _endpoints(calculator),
        {
            "n_max": 3,
            "max_iter": 1,
            "fmax": 1.0e6,
            "rms_fmax": 1.0e6,
            "climb": False,
            "interpolation": "linear",
        },
    ).run()

    assert result.status is RunStatus.CONVERGED
    assert result.base_stage_count == 1
    assert result.insertion_count == 1
    assert result.final_image_count == 3
    assert result.ci_stage is None
    assert result.certified_ts is False
    assert dict(result.final_stage.budgets) == {
        "maximum_iterations": 1,
        "used_iterations": 0,
    }
    assert result.energies_hartree.shape == (3,)
    assert len(result.forces_hartree_per_angstrom) == 3
    assert all(image.calc is calculator for image in result.images)
    assert [image.info["path_role"] for image in result.images] == [
        "reactant",
        "autoneb_inserted",
        "product",
    ]
    assert Path(str(output).replace(".out", "_image_traj.xyz")).is_file()
    assert not list(tmp_path.glob("*ts*.xyz"))


def test_real_ci_stage_preserves_index_and_displacement_evidence(tmp_path):
    output = tmp_path / "ci.out"

    result = AutoNEB(
        str(output),
        _endpoints(_QuadraticCalculator()),
        {
            "n_max": 3,
            "max_iter": 1,
            "fmax": 1.0e6,
            "rms_fmax": 1.0e6,
            "climb": True,
            "ci_max_iter": 1,
            "ci_fmax": 1.0e6,
            "ci_rms_fmax": 1.0e6,
            "ci_displacement_tol": 1.0e-12,
            "interpolation": "linear",
        },
    ).run()

    assert result.status is RunStatus.CONVERGED
    assert result.ci_stage is result.final_stage
    assert result.candidate is not None
    assert result.certified_ts is False
    assert dict(result.ci_stage.budgets) == {
        "maximum_iterations": 1,
        "used_iterations": 0,
    }
    diagnostics = getattr(result.ci_stage, "_stage_diagnostics")
    assert diagnostics["climbing_image_index"] == 1
    assert diagnostics["ci_max_displacement_angstrom"] == pytest.approx(0.0)
    np.testing.assert_array_equal(
        result.candidate.positions,
        result.images[diagnostics["climbing_image_index"]].positions,
    )
    assert Path(str(output).replace(".out", "_cineb_traj.xyz")).is_file()
    assert not list(tmp_path.glob("*ts*.xyz"))


def test_real_bounded_base_stage_stops_growth_and_blocks_ci(tmp_path):
    output = tmp_path / "bounded.out"

    result = AutoNEB(
        str(output),
        _endpoints(_PerpendicularCalculator()),
        {
            "n_max": 4,
            "max_iter": 1,
            "fmax": 1.0e-12,
            "rms_fmax": 1.0e-12,
            "climb": True,
            "ci_max_iter": 4,
            "interpolation": "linear",
        },
    ).run()

    assert result.status is RunStatus.BOUNDED_NONCONVERGED
    assert result.base_stage_count == 1
    assert result.insertion_count == 1
    assert result.final_image_count == 3
    assert result.ci_stage is None
    assert result.certified_ts is False
    assert dict(result.final_stage.budgets) == {
        "maximum_iterations": 1,
        "used_iterations": 1,
    }
    assert not Path(str(output).replace(".out", "_cineb_traj.xyz")).exists()
    assert not list(tmp_path.glob("*ts*.xyz"))
