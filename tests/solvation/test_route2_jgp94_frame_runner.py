from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

BENCHMARK_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "implicit-solvation"
    / "benchmarks"
)
if str(BENCHMARK_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIRECTORY))

from run_route2_jgp94_frame_vjp_canary import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    Hartree,
    _coordinate_gradient_with_retained_state,
    _state_energy_ev,
)


class _FakeState:
    def __init__(self, atom_count: int) -> None:
        self._atom_count = atom_count

    def solvation_force_terms(self, electrostatics: Any) -> np.ndarray:
        del electrostatics
        return np.ones((3, self._atom_count), dtype=float)

    def multipole_force_terms(self, multipoles: Any) -> np.ndarray:
        del multipoles
        return np.full((3, self._atom_count), 2.0, dtype=float)

    def energy(self) -> float:
        return 0.25


class _FakeResponse:
    def __init__(self, atom_count: int) -> None:
        self.atom_count = atom_count
        self._scf_state: _FakeState | None = None
        self.solve_arguments: dict[str, bool] | None = None

    def _validated_density(
        self,
        values: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        assert name == "density_coefficients"
        return np.asarray(values, dtype=float)

    def _solve_state(
        self,
        density: np.ndarray,
        *,
        coordinate_derivative: bool,
        solve_adjoint: bool,
        warm_start: bool,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], _FakeState]:
        assert density.shape == (self.atom_count, 4)
        self.solve_arguments = {
            "coordinate_derivative": coordinate_derivative,
            "solve_adjoint": solve_adjoint,
            "warm_start": warm_start,
        }
        state = _FakeState(self.atom_count)
        self._scf_state = state
        return np.zeros_like(density), {}, state


def test_coordinate_gradient_retains_the_state_used_for_energy_accounting() -> None:
    response: Any = _FakeResponse(atom_count=2)
    density = np.zeros((2, 4), dtype=float)

    gradient = _coordinate_gradient_with_retained_state(response, density)

    assert response.solve_arguments == {
        "coordinate_derivative": True,
        "solve_adjoint": True,
        "warm_start": True,
    }
    np.testing.assert_allclose(gradient, np.full((2, 3), 3.0))
    assert _state_energy_ev(response) == 0.25 * Hartree
