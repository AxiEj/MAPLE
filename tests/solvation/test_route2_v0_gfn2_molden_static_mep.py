"""Unit tests for the validation-only GFN2 MOLDEN static-MEP helper."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PATH = (
    _REPOSITORY_ROOT
    / "docs/implicit-solvation/benchmarks/route2_v0_gfn2_molden_static_mep.py"
)
_MODULE = runpy.run_path(str(_HELPER_PATH))
_TO_PYSCF_INPUT = _MODULE["molden_shells_to_pyscf_input"]


def _normalization(angular_momentum: int, exponents: np.ndarray) -> np.ndarray:
    return np.full_like(exponents, 2.0 + angular_momentum, dtype=float)


def test_molden_s_and_p_shells_are_transferred_without_a_fitted_scale():
    centres = np.zeros((4, 3))
    exponents = (
        np.array([2.0, 1.0]),
        np.array([3.0, 1.5]),
        np.array([3.0, 1.5]),
        np.array([3.0, 1.5]),
    )
    coefficients = (
        np.array([4.0, 6.0]),
        np.array([9.0, 12.0]),
        np.array([9.0, 12.0]),
        np.array([9.0, 12.0]),
    )
    atoms, basis = _TO_PYSCF_INPUT(
        atom_symbols=("H",),
        atom_positions_bohr=np.zeros((1, 3)),
        ao_centers_bohr=centres,
        ao_powers=np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))),
        ao_exponents=exponents,
        ao_coefficients=coefficients,
        primitive_radial_normalization=_normalization,
    )

    assert atoms == [("H1", (0.0, 0.0, 0.0))]
    assert basis == {
        "H1": [
            [0, [2.0, 2.0], [1.0, 3.0]],
            [1, [3.0, 3.0], [1.5, 4.0]],
        ]
    }


def test_molden_p_shell_component_order_fails_closed():
    with pytest.raises(ValueError, match="canonical s/p order"):
        _TO_PYSCF_INPUT(
            atom_symbols=("H",),
            atom_positions_bohr=np.zeros((1, 3)),
            ao_centers_bohr=np.zeros((4, 3)),
            ao_powers=np.array(((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1))),
            ao_exponents=(
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
            ),
            ao_coefficients=(
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
            ),
            primitive_radial_normalization=_normalization,
        )
