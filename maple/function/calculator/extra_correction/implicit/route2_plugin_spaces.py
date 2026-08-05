"""Electronic source/field spaces and explicit continuum adjoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .electrostatic_pairing import MACE_POLAR_L1_PAIRING
from .route2_plugin_errors import PluginContractError


def _nonempty(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _readonly_finite(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ElectronicSourceSpace:
    """Auditable finite source coordinate space.

    ``shape_for_atom_count`` is intentionally explicit so the same contract can
    describe atom multipoles, GTO coefficients, and a fixed-grid density.  A
    source's total-charge functional is optional only for source spaces that
    cannot represent a charge constraint; such a space cannot enter the KKT
    route.
    """

    name: str
    representation: str
    source_unit: str
    shape_for_atom_count: Callable[[int], tuple[int, ...]]
    total_charge: Callable[[np.ndarray], float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, name="source-space name"))
        object.__setattr__(
            self,
            "representation",
            _nonempty(self.representation, name="source-space representation"),
        )
        object.__setattr__(
            self,
            "source_unit",
            _nonempty(self.source_unit, name="source-space unit"),
        )
        if not callable(self.shape_for_atom_count):
            raise TypeError("source-space shape_for_atom_count must be callable.")
        if self.total_charge is not None and not callable(self.total_charge):
            raise TypeError("source-space total_charge must be callable or None.")

    def expected_shape(self, atom_count: int) -> tuple[int, ...]:
        if (
            isinstance(atom_count, bool)
            or not isinstance(atom_count, int)
            or atom_count < 1
        ):
            raise ValueError("source-space atom_count must be a positive integer.")
        shape = tuple(self.shape_for_atom_count(atom_count))
        if not shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in shape
        ):
            raise ValueError(
                "source-space shape_for_atom_count returned an invalid shape."
            )
        return shape

    def validate_source(
        self,
        values: object,
        *,
        atom_count: int,
        name: str = "electronic source",
    ) -> np.ndarray:
        array = _readonly_finite(values, name=name)
        expected = self.expected_shape(atom_count)
        if array.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected} in source space {self.name!r}; "
                f"received {array.shape}."
            )
        return array

    def constrained_total_charge(self, values: object, *, atom_count: int) -> float:
        source = self.validate_source(values, atom_count=atom_count)
        if self.total_charge is None:
            raise PluginContractError(
                f"Source space {self.name!r} has no total-charge functional."
            )
        charge = float(self.total_charge(source))
        if not np.isfinite(charge):
            raise ValueError(
                "source-space total-charge functional returned non-finite value."
            )
        return charge


@dataclass(frozen=True)
class FieldDualSpace:
    """Dual field coordinate space paired with an electronic source space."""

    name: str
    representation: str
    potential_unit: str
    gradient_unit: str
    gauge: str
    shape_for_atom_count: Callable[[int], tuple[int, ...]]
    pairing: Callable[[np.ndarray, np.ndarray], float]

    def __post_init__(self) -> None:
        for name in (
            "name",
            "representation",
            "potential_unit",
            "gradient_unit",
            "gauge",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        if not callable(self.shape_for_atom_count) or not callable(self.pairing):
            raise TypeError("field-dual shape and pairing must be callable.")

    def validate_field(
        self,
        values: object,
        *,
        atom_count: int,
        name: str = "dual field",
    ) -> np.ndarray:
        array = _readonly_finite(values, name=name)
        expected = tuple(self.shape_for_atom_count(atom_count))
        if not expected or any(
            not isinstance(size, int) or size < 1 for size in expected
        ):
            raise ValueError(
                "field-dual shape_for_atom_count returned an invalid shape."
            )
        if array.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected} in field dual {self.name!r}; "
                f"received {array.shape}."
            )
        return array

    def pair(
        self,
        source: object,
        field: object,
        *,
        source_space: ElectronicSourceSpace,
        atom_count: int,
    ) -> float:
        source_values = source_space.validate_source(source, atom_count=atom_count)
        field_values = self.validate_field(field, atom_count=atom_count)
        result = float(self.pairing(source_values, field_values))
        if not np.isfinite(result):
            raise ValueError("source/field pairing must be finite.")
        return result


@dataclass(frozen=True)
class CouplingAdjointEvidence:
    """Numerical evidence for one explicit ``B``/``B*`` pairing check."""

    source_to_surface: float
    source_from_surface: float
    absolute_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ContinuumCoupling:
    """Explicit source--continuum operators and their mathematical metadata.

    ``source_to_surface`` is :math:`B_R`; ``surface_to_field_dual`` is
    :math:`B_R^*`.  The latter must return coordinates in ``field_dual_space``
    rather than source coordinates.  This stops a transposed array with the
    right shape from being mistaken for an adjoint operator.
    """

    name: str
    source_space: ElectronicSourceSpace
    field_dual_space: FieldDualSpace
    source_to_surface: Callable[[np.ndarray], np.ndarray]
    surface_to_field_dual: Callable[[np.ndarray], np.ndarray]
    surface_pairing: Callable[[np.ndarray, np.ndarray], float]
    surface_unit: str
    total_charge_constraint: str
    gauge: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, name="coupling name"))
        object.__setattr__(
            self, "surface_unit", _nonempty(self.surface_unit, name="surface unit")
        )
        object.__setattr__(
            self,
            "total_charge_constraint",
            _nonempty(self.total_charge_constraint, name="total-charge constraint"),
        )
        object.__setattr__(self, "gauge", _nonempty(self.gauge, name="coupling gauge"))
        if not isinstance(self.source_space, ElectronicSourceSpace):
            raise TypeError("continuum coupling needs an ElectronicSourceSpace.")
        if not isinstance(self.field_dual_space, FieldDualSpace):
            raise TypeError("continuum coupling needs a FieldDualSpace.")
        if self.gauge != self.field_dual_space.gauge:
            raise ValueError(
                "continuum coupling and field dual must declare the same gauge."
            )
        for name in ("source_to_surface", "surface_to_field_dual", "surface_pairing"):
            if not callable(getattr(self, name)):
                raise TypeError(f"continuum coupling {name} must be callable.")

    def verify_adjoint(
        self,
        source: object,
        surface: object,
        *,
        atom_count: int,
        relative_tolerance: float = 1.0e-10,
        absolute_tolerance: float = 1.0e-12,
    ) -> CouplingAdjointEvidence:
        if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
            raise ValueError("adjoint tolerances must be non-negative.")
        source_values = self.source_space.validate_source(source, atom_count=atom_count)
        surface_values = _readonly_finite(surface, name="continuum surface test vector")
        mapped_surface = _readonly_finite(
            self.source_to_surface(source_values), name="B_R electronic source"
        )
        if mapped_surface.shape != surface_values.shape:
            raise ValueError(
                "B_R electronic source and continuum surface test vector must have the same shape."
            )
        dual_values = self.field_dual_space.validate_field(
            self.surface_to_field_dual(surface_values),
            atom_count=atom_count,
            name="B_R* surface field",
        )
        lhs = float(self.surface_pairing(mapped_surface, surface_values))
        rhs = self.field_dual_space.pair(
            source_values,
            dual_values,
            source_space=self.source_space,
            atom_count=atom_count,
        )
        if not np.isfinite(lhs) or not np.isfinite(rhs):
            raise ValueError("continuum adjoint pairing must be finite.")
        tolerance = absolute_tolerance + relative_tolerance * max(abs(lhs), abs(rhs))
        error = abs(lhs - rhs)
        return CouplingAdjointEvidence(
            source_to_surface=lhs,
            source_from_surface=rhs,
            absolute_error=error,
            tolerance=tolerance,
            passed=error <= tolerance,
        )


ATOMIC_L1_PLUGIN_SOURCE_SPACE = ElectronicSourceSpace(
    name="maple-atomic-net-monopole-real-spherical-l1-v1",
    representation="atom-centred net monopole plus real-spherical l=1 multipoles",
    source_unit="e, e*angstrom",
    shape_for_atom_count=lambda atom_count: (atom_count, 4),
    total_charge=lambda source: float(np.sum(source[:, 0])),
)
ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE = FieldDualSpace(
    name="maple-atomic-l1-local-potential-gradient-dual-v1",
    representation="nodewise [potential, potential_gradient_x, potential_gradient_y, potential_gradient_z]",
    potential_unit="eV/e",
    gradient_unit="eV/(e*angstrom)",
    gauge="continuum-zero-at-infinity",
    shape_for_atom_count=lambda atom_count: (atom_count, 4),
    pairing=MACE_POLAR_L1_PAIRING.pair,
)

__all__ = [
    "ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE",
    "ATOMIC_L1_PLUGIN_SOURCE_SPACE",
    "ContinuumCoupling",
    "CouplingAdjointEvidence",
    "ElectronicSourceSpace",
    "FieldDualSpace",
]
