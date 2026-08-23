"""Operational Phi0 scalar and implicit force for the hybrid source root."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2,
    get_scalar_definition,
)
from maple.solvation.api.state_registry import (
    PERMANENT_INDUCED_SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
)

from .adjoint import AdjointOptions, solve_reduced_adjoint
from .linearization import SeparatedReducedLinearization
from .permanent_induced_state import PermanentInducedOperationalStateEquation
from .separated_fixed_point import SeparatedFixedPointState
from .separated_ledgers import (
    OperationalLedgerEvaluation,
    SeparatedOperationalGradientResult,
)
from .state_equation import geometry_sha256, provider_behavior_sha256


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _rows(values: object) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("coordinate gradient must be a finite atom-by-3 array.")
    return tuple(tuple(float(value) for value in row) for row in array)


class HybridHarmonicDDPCMPhi0Ledger:
    """Frozen ``E_vac^POLAR + G_ddPCM(point p, Gaussian d*)`` ledger."""

    __slots__ = (
        "_configuration_sha256",
        "_sealed",
        "equation",
        "scalar_id",
    )

    implementation_entry_point = (
        "maple.solvation.coupling.permanent_induced_ledgers:"
        "HybridHarmonicDDPCMPhi0Ledger"
    )

    def __init__(self, equation: PermanentInducedOperationalStateEquation) -> None:
        if not isinstance(equation, PermanentInducedOperationalStateEquation):
            raise TypeError(
                "equation must be PermanentInducedOperationalStateEquation."
            )
        scalar_id = (
            OPERATIONAL_MACE_MDP_POLAR_HYBRID_PHI0_SMOOTH_HARMONIC_DDPCM_V2
        )
        definition = get_scalar_definition(scalar_id)
        if definition.implementation_entry_point != self.implementation_entry_point:
            raise ValueError("scalar registry entry point does not match the ledger.")
        if definition.state_equation_id != (
            PERMANENT_INDUCED_SEPARATED_OPERATIONAL_STATE_EQUATION_ID
        ):
            raise ValueError("scalar has the wrong state equation.")
        if definition.enabled or definition.admitted_capabilities.enabled_tiers:
            raise ValueError("hybrid ddPCM Phi0 must remain disabled.")
        if equation.continuum.scalar_id != scalar_id:
            raise ValueError("continuum snapshot is bound to a different scalar.")
        for name in (
            "vacuum_energy_ev",
            "vacuum_forces_ev_per_angstrom",
            "permanent_source_position_vjp",
        ):
            if not callable(getattr(equation.hybrid, name, None)):
                raise TypeError(f"hybrid ledger requires callable {name}().")
        equation.fingerprint_sha256()
        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("HybridHarmonicDDPCMPhi0Ledger is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "scalar_id": self.scalar_id,
                    "registry_formula": get_scalar_definition(
                        self.scalar_id
                    ).exact_formula,
                    "implementation_entry_point": self.implementation_entry_point,
                    "equation_sha256": self.equation.fingerprint_sha256(),
                    "hybrid_vacuum_behavior_sha256": provider_behavior_sha256(
                        self.equation.hybrid,
                        (
                            "configuration_sha256",
                            "vacuum_energy_ev",
                            "vacuum_forces_ev_per_angstrom",
                        ),
                        label="hybrid_phi0_vacuum",
                    ),
                    "capabilities": "none",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid ddPCM Phi0 ledger configuration drifted.")
        return current

    def _validate_geometry(self, geometry: object) -> None:
        if geometry_sha256(geometry) != self.equation.continuum.geometry_sha256:
            raise ValueError("geometry does not match the ledger continuum snapshot.")
        if geometry_sha256(geometry) != self.equation.anchor.geometry_sha256:
            raise ValueError("geometry does not match the ledger source anchor.")

    def evaluate_root(
        self,
        geometry: object,
        reduced_coordinates: object,
        *,
        root_tolerance: float,
    ) -> OperationalLedgerEvaluation:
        self.configuration_sha256()
        self._validate_geometry(geometry)
        if not np.isfinite(root_tolerance) or root_tolerance <= 0.0:
            raise ValueError("root_tolerance must be finite and positive.")
        y = np.asarray(reduced_coordinates, dtype=float)
        if y.shape != (self.equation.reduced_dimension,) or not np.all(np.isfinite(y)):
            raise ValueError("reduced coordinates have an invalid shape.")
        residual = self.equation.residual(geometry, y)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > root_tolerance:
            raise ValueError("ledger refuses a state above the root tolerance.")
        vacuum = float(self.equation.hybrid.vacuum_energy_ev(geometry))
        continuum = float(
            self.equation.continuum.continuum_energy_eV(
                self.equation.permanent_source,
                self.equation.source(y),
            )
        )
        components = (
            ("macepolar_vacuum_energy", vacuum),
            (
                "point_permanent_gaussian_induced_smooth_harmonic_ddpcm_energy",
                continuum,
            ),
        )
        return OperationalLedgerEvaluation(
            scalar_id=self.scalar_id,
            state_equation_id=self.equation.state_equation_id,
            geometry_sha256=geometry_sha256(geometry),
            equation_sha256=self.equation.fingerprint_sha256(),
            ledger_sha256=self.configuration_sha256(),
            field_semantics_sha256=self.equation.hybrid.configuration_sha256(),
            reduced_coordinates_sha256=_array_sha256(y),
            root_residual_norm=residual_norm,
            root_tolerance=float(root_tolerance),
            components_eV=components,
            total_energy_eV=vacuum + continuum,
        )

    def _validate_primal_state(
        self, geometry: object, state: SeparatedFixedPointState
    ) -> None:
        if not isinstance(state, SeparatedFixedPointState) or not state.converged:
            raise ValueError("implicit differentiation requires a converged state.")
        identities = {
            "state_equation_id": self.equation.state_equation_id,
            "geometry_sha256": geometry_sha256(geometry),
            "equation_sha256": self.equation.fingerprint_sha256(),
            "source_space_sha256": self.equation.source_space.metadata_hash(),
            "receiver_space_sha256": self.equation.receiver_space.metadata_hash(),
            "continuum_configuration_sha256": (
                self.equation.continuum.configuration_sha256()
            ),
        }
        for name, expected in identities.items():
            if getattr(state, name) != expected:
                raise ValueError(f"primal state has a mismatched {name}.")
        y = state.y_array()
        current = (
            self.equation.source(y),
            self.equation.boundary_state(y),
            self.equation.field(y),
            self.equation.residual(geometry, y),
        )
        stored = (
            state.source_array(),
            state.boundary_state_array(),
            state.field_array(),
            np.asarray(state.actual_unmixed_residual, dtype=float),
        )
        if any(not np.array_equal(left, right) for left, right in zip(current, stored)):
            raise ValueError("stored hybrid root leaves do not replay exactly.")
        if float(np.linalg.norm(current[-1])) > state.primal_tolerance:
            raise ValueError("primal residual exceeds its recorded tolerance.")

    def reduced_gradient(
        self, geometry: object, reduced_coordinates: object
    ) -> np.ndarray:
        self.configuration_sha256()
        self._validate_geometry(geometry)
        y = np.asarray(reduced_coordinates, dtype=float)
        if y.shape != (self.equation.reduced_dimension,) or not np.all(np.isfinite(y)):
            raise ValueError("reduced coordinates have an invalid shape.")
        _permanent, induced = self.equation.continuum.continuum_source_gradients(
            self.equation.permanent_source,
            self.equation.source(y),
        )
        return self.equation.coordinates.reduce_source_cotangent(induced)

    def direct_coordinate_gradient(
        self, geometry: object, reduced_coordinates: object
    ) -> np.ndarray:
        """Return ``partial_R Phi0`` at fixed induced reduced coordinates."""

        self.configuration_sha256()
        self._validate_geometry(geometry)
        y = np.asarray(reduced_coordinates, dtype=float)
        induced = self.equation.source(y)
        permanent_gradient, _induced_gradient = (
            self.equation.continuum.continuum_source_gradients(
                self.equation.permanent_source, induced
            )
        )
        vacuum = -np.asarray(
            self.equation.hybrid.vacuum_forces_ev_per_angstrom(geometry),
            dtype=float,
        )
        fixed_continuum = np.asarray(
            self.equation.continuum.continuum_energy_position_gradient(
                self.equation.permanent_source, induced
            ),
            dtype=float,
        )
        moving_permanent = np.asarray(
            self.equation.hybrid.permanent_source_position_vjp(
                geometry, self.equation.anchor, permanent_gradient
            ),
            dtype=float,
        )
        expected = (self.equation.coordinates.atom_count, 3)
        terms = (vacuum, fixed_continuum, moving_permanent)
        if any(term.shape != expected for term in terms) or any(
            not np.all(np.isfinite(term)) for term in terms
        ):
            raise ValueError(
                "hybrid direct-coordinate terms must be finite with shape "
                f"{expected}."
            )
        return vacuum + fixed_continuum + moving_permanent

    def implicit_gradient(
        self,
        geometry: object,
        state: SeparatedFixedPointState,
        *,
        adjoint_options: AdjointOptions = AdjointOptions(),
    ) -> SeparatedOperationalGradientResult:
        self.configuration_sha256()
        self._validate_primal_state(geometry, state)
        evaluation = self.evaluate_root(
            geometry,
            state.y_array(),
            root_tolerance=state.primal_tolerance,
        )
        linearization = SeparatedReducedLinearization.at(
            self.equation, geometry, state.y_array()
        )
        adjoint = solve_reduced_adjoint(
            linearization,
            self.reduced_gradient(geometry, state.y_array()),
            options=adjoint_options,
        )
        direct = self.direct_coordinate_gradient(geometry, state.y_array())
        residual_pullback = np.asarray(
            self.equation.coordinate_vjp(
                geometry, state.y_array(), adjoint.solution_array()
            ),
            dtype=float,
        )
        total = direct - residual_pullback
        return SeparatedOperationalGradientResult(
            ledger=evaluation,
            adjoint=adjoint,
            direct_coordinate_gradient_eV_per_angstrom=_rows(direct),
            residual_coordinate_pullback_eV_per_angstrom=_rows(
                residual_pullback
            ),
            total_coordinate_gradient_eV_per_angstrom=_rows(total),
        )


__all__ = ["HybridHarmonicDDPCMPhi0Ledger"]
