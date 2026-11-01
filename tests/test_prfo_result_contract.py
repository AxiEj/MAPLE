from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from maple.function.dispatcher.result_contracts import PRFOResult, PRFOStatus
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.dispatcher.ts.ts import TransitionState


class Quadratic(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, reference, hessian):
        super().__init__()
        self.reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        self._hessian = np.asarray(hessian, dtype=np.float64)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = atoms.positions.reshape(-1) - self.reference
        gradient = self._hessian @ displacement
        energy = float(0.5 * displacement @ gradient)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -gradient.reshape(-1, 3),
        }

    def get_hessian(self, atoms):
        return self._hessian.copy()


def _stationary_hydrogen(*, negative_modes: int) -> Atoms:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    diagonal = np.ones(3)
    diagonal[:negative_modes] = -1.0
    atoms.calc = Quadratic(atoms.positions, np.diag(diagonal))
    return atoms


def test_certified_success(tmp_path):
    source = _stationary_hydrogen(negative_modes=1)
    result = PRFO(
        str(tmp_path / "certified.out"),
        source,
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert isinstance(result, PRFOResult)
    assert result.status is PRFOStatus.CERTIFIED_TS
    assert result.certified_ts is True
    assert result.force_displacement_converged is True
    assert result.geometry_sha256
    assert result.forces_hartree_per_angstrom.flags.writeable is False
    assert result.curvature["negative_mode_count"] == 1
    assert (tmp_path / "certified_prfo_ts.xyz").is_file()
    frozen_positions = result.atoms.get_positions().copy()
    source.positions[0, 0] = 3.0
    np.testing.assert_array_equal(result.atoms.get_positions(), frozen_positions)
    np.testing.assert_array_equal(result.forces_hartree_per_angstrom, 0.0)


def test_budget_returns_typed_result(tmp_path):
    result = PRFO(
        str(tmp_path / "budget.out"),
        _stationary_hydrogen(negative_modes=1),
        paras={"max_iter": 0},
    ).run()

    assert isinstance(result, PRFOResult)
    assert result.status is PRFOStatus.BOUNDED_NONCONVERGED
    assert result.certified_ts is False
    assert result.iterations == 0


@pytest.mark.parametrize("negative_modes", [0, 2])
def test_failure_never_writes_ts_file(tmp_path, negative_modes):
    result = PRFO(
        str(tmp_path / "minimum.out"),
        _stationary_hydrogen(negative_modes=negative_modes),
        paras={"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
    ).run()

    assert result.status is PRFOStatus.REJECTED_NOT_INDEX_ONE
    assert result.certified_ts is False
    assert not (tmp_path / "minimum_prfo_ts.xyz").exists()
    assert (tmp_path / "minimum_prfo_candidate.xyz").is_file()


def test_transition_state_union_contract(tmp_path):
    certified = TransitionState(
        str(tmp_path / "public-certified.out"),
        _stationary_hydrogen(negative_modes=1),
        {"max_iter": 1, "trust_radius": 0.02, "trust_min": 0.02},
        method="prfo",
    )
    geometry = certified.run()
    assert isinstance(geometry, Atoms)
    assert certified.last_result.status is PRFOStatus.CERTIFIED_TS

    bounded = TransitionState(
        str(tmp_path / "public-bounded.out"),
        _stationary_hydrogen(negative_modes=1),
        {"max_iter": 0},
        method="prfo",
    )
    result = bounded.run()
    assert isinstance(result, PRFOResult)
    assert bounded.last_result is result


def test_all_repo_callers_migrated():
    root = Path(__file__).resolve().parents[1]
    owned_callers = [
        root / "maple/function/dispatcher/ts/ts.py",
        root / "maple/function/dispatcher/ts/algorithm/neb.py",
        root / "maple/function/dispatcher/ts/algorithm/string.py",
    ]
    for path in owned_callers:
        text = path.read_text()
        assert "prfo.run()" in text
        assert "certified_ts" in text or "last_result" in text
    assert "images.insert(hei + 1, ts_opt)" not in owned_callers[1].read_text()
    assert "images_ts.insert" not in owned_callers[2].read_text()
    benchmark_callers = [
        root / "docs/implicit-solvation/benchmarks/run_mlip_device_canary.py",
        root
        / "docs/implicit-solvation/benchmarks/run_route1_stationary_validation.py",
    ]
    for path in benchmark_callers:
        text = path.read_text()
        assert "prfo_result" in text
        assert "certified_ts" in text
