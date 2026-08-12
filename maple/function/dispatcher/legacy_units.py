"""Private ASE-to-legacy unit boundary for MAPLE job implementations.

Raw calculators remain honest ASE calculators: eV, eV/Angstrom, stress in
eV/Angstrom^3, and Cartesian Hessians in eV/Angstrom^2.  Existing MAPLE job
algorithms historically consume the corresponding Hartree-based quantities.
This non-ASE view performs that boundary conversion without mutating the raw
calculator or its ``results`` mapping.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from ..calculator.calculator_base import EV2HARTREE


_ENERGY_PROPERTIES = frozenset({"energy", "free_energy"})
_FORCE_PROPERTIES = frozenset({"forces", "force"})
_STRESS_PROPERTIES = frozenset({"stress", "stresses", "virial", "virials"})
_SCALED_RESULT_PROPERTIES = (
    _ENERGY_PROPERTIES | _FORCE_PROPERTIES | _STRESS_PROPERTIES | {"hessian"}
)


def _scaled(value):
    return np.asarray(value) * EV2HARTREE if isinstance(value, np.ndarray) else value * EV2HARTREE


class LegacyHartreeJobView:
    """Duck-typed calculator view used only inside legacy MAPLE jobs.

    This deliberately does not inherit :class:`ase.calculators.Calculator`.
    Stress follows its energy-density dimension: public eV/Angstrom^3 becomes
    legacy Hartree/Angstrom^3.  Virial follows energy and is scaled likewise.
    Nested ``solvation`` data is already explicitly Hartree-labelled and is
    therefore passed through unchanged.
    """

    __slots__ = ("_calculator",)

    def __init__(self, calculator):
        if isinstance(calculator, LegacyHartreeJobView):
            raise ValueError("LegacyHartreeJobView must not be double-wrapped.")
        self._calculator = calculator

    @property
    def raw_calculator(self):
        return self._calculator

    @property
    def results(self) -> dict:
        raw = getattr(self._calculator, "results", {}) or {}
        return {
            key: _scaled(value) if key in _SCALED_RESULT_PROPERTIES else value
            for key, value in raw.items()
        }

    @property
    def implemented_properties(self):
        return getattr(self._calculator, "implemented_properties", ())

    def calculate(self, atoms=None, properties=None, system_changes=None):
        kwargs = {}
        if system_changes is not None:
            kwargs["system_changes"] = system_changes
        self._calculator.calculate(atoms, properties, **kwargs)
        return self.results

    def get_property(self, name, atoms=None, allow_calculation=True):
        value = self._calculator.get_property(name, atoms, allow_calculation)
        return _scaled(value) if str(name).lower() in _SCALED_RESULT_PROPERTIES else value

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return _scaled(
            self._calculator.get_potential_energy(
                atoms, force_consistent=force_consistent
            )
        )

    def get_forces(self, atoms=None):
        return _scaled(self._calculator.get_forces(atoms))

    def get_stress(self, atoms=None, *args, **kwargs):
        return _scaled(self._calculator.get_stress(atoms, *args, **kwargs))

    def get_hessian(self, atoms, *args, **kwargs):
        return _scaled(self._calculator.get_hessian(atoms, *args, **kwargs))

    def get_hvp(self, atoms, direction, *args, **kwargs):
        hvp, forces, energy = self._calculator.get_hvp(
            atoms, direction, *args, **kwargs
        )
        return _scaled(hvp), _scaled(forces), _scaled(energy)

    def __getattr__(self, name):
        return getattr(self._calculator, name)


def _iter_atoms(value):
    if hasattr(value, "multiatoms"):
        yield from value.multiatoms
    elif isinstance(value, list):
        yield from value
    else:
        yield value


@contextmanager
def legacy_hartree_job_calculators(atoms) -> Iterator[None]:
    """Temporarily install legacy views, restoring every raw calculator."""

    bindings = []
    seen = set()
    try:
        for item in _iter_atoms(atoms):
            if item is None or id(item) in seen:
                continue
            seen.add(id(item))
            calculator = getattr(item, "calc", None)
            if calculator is None or isinstance(calculator, LegacyHartreeJobView):
                continue
            bindings.append((item, calculator))
            item.calc = LegacyHartreeJobView(calculator)
        yield
    finally:
        for item, calculator in reversed(bindings):
            item.calc = calculator
