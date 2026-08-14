"""Fail-closed common-stationarity kernel for scalar-first Route 2 models.

For conjugacy sign ``s`` the stationary Lagrangian is

``L_s(R,c,u) = E(R,u) - s <c,u>_Q + s G(R,c)``.

Both response maps are generated from their respective sealed scalar graphs by
:mod:`maple.solvation.coupling.variational_adapters`.  The public registry
remains disabled: a converged internal root and envelope-gradient diagnostic do
not admit E/F/H/V/M or a MAPLE workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.api.profiles import get_solvation_profile
from maple.solvation.api.scalar_registry import get_scalar_definition
from maple.solvation.api.state_registry import VARIATIONAL_STATE_EQUATION_ID
from maple.solvation.continuum.functional import ContinuumEnergyFunctional
from maple.solvation.models.base import atom_count, model_charge_and_multiplicity
from maple.solvation.models.field_energy import FieldEnergyFunctional

from .fixed_point import FixedPointOptions, FixedPointState, solve_fixed_point
from .metrics import PairingMetric, get_pairing_metric
from .spaces import get_coordinate_contract, get_field_dual_space, get_source_space
from .state_equation import ReducedStateEquation, geometry_sha256
from .variational_adapters import (
    ScalarFirstContinuumResponseAdapter,
    ScalarFirstElectronicResponseAdapter,
    _capabilities_are_closed,
    _configuration,
    _sha,
    _stable_id,
)

VARIATIONAL_COMMON_FUNCTIONAL_ENTRY_POINT = (
    "maple.solvation.coupling.variational_state:VariationalCommonFunctional"
)


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _rows(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite numeric scalar, not bool.")
    array = np.asarray(value)
    if array.shape != ():
        raise TypeError(f"{name} must be a scalar.")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True, slots=True)
class VariationalStationaryEnergy:
    scalar_id: str
    profile_id: str
    conjugacy_sign: int
    model_energy_eV: float
    continuum_energy_eV: float
    coupling_energy_eV: float
    gauge_potential_eV_per_e: float
    reduced_residual: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scalar_id", _stable_id(self.scalar_id, name="scalar_id")
        )
        object.__setattr__(
            self, "profile_id", _stable_id(self.profile_id, name="profile_id")
        )
        if self.conjugacy_sign not in (-1, 1):
            raise ValueError("conjugacy_sign must be exactly +1 or -1.")
        for name in (
            "model_energy_eV",
            "continuum_energy_eV",
            "coupling_energy_eV",
            "gauge_potential_eV_per_e",
        ):
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name=name),
            )
        residual = np.asarray(self.reduced_residual, dtype=float)
        if residual.ndim != 1 or residual.size < 1 or not np.all(np.isfinite(residual)):
            raise ValueError("reduced_residual must be a non-empty finite vector.")
        object.__setattr__(
            self,
            "reduced_residual",
            tuple(float(value) for value in residual),
        )

    @property
    def total_energy_eV(self) -> float:
        sign = self.conjugacy_sign
        return float(
            self.model_energy_eV
            - sign * self.coupling_energy_eV
            + sign * self.continuum_energy_eV
        )

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.reduced_residual))


@dataclass(frozen=True, slots=True)
class VariationalEnvelopeGradient:
    scalar: VariationalStationaryEnergy
    model_coordinate_gradient_eV_per_A: tuple[tuple[float, float, float], ...]
    continuum_coordinate_gradient_eV_per_A: tuple[tuple[float, float, float], ...]
    total_coordinate_gradient_eV_per_A: tuple[tuple[float, float, float], ...]
    admitted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scalar, VariationalStationaryEnergy):
            raise TypeError("scalar must be VariationalStationaryEnergy.")
        arrays = tuple(
            np.asarray(getattr(self, name), dtype=float)
            for name in (
                "model_coordinate_gradient_eV_per_A",
                "continuum_coordinate_gradient_eV_per_A",
                "total_coordinate_gradient_eV_per_A",
            )
        )
        if (
            arrays[0].ndim != 2
            or arrays[0].shape[1] != 3
            or any(array.shape != arrays[0].shape for array in arrays)
            or any(not np.all(np.isfinite(array)) for array in arrays)
        ):
            raise ValueError(
                "coordinate gradients must be finite matching (N,3) arrays."
            )
        for name, array in zip(
            (
                "model_coordinate_gradient_eV_per_A",
                "continuum_coordinate_gradient_eV_per_A",
                "total_coordinate_gradient_eV_per_A",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, _rows(array))
        expected = arrays[0] + self.scalar.conjugacy_sign * arrays[1]
        if not np.allclose(arrays[2], expected, rtol=1.0e-13, atol=1.0e-14):
            raise ValueError(
                "total envelope gradient does not close its scalar leaves."
            )
        if self.admitted is not False:
            raise ValueError("the current variational envelope gradient is unadmitted.")

    @property
    def forces_eV_per_A(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            tuple(-value for value in row)
            for row in self.total_coordinate_gradient_eV_per_A
        )


@dataclass(frozen=True, slots=True)
class VariationalCommonFunctional:
    """One disabled common scalar bound to a derivative-generated state root."""

    equation: ReducedStateEquation
    model: FieldEnergyFunctional
    continuum: ContinuumEnergyFunctional
    scalar_id: str
    profile_id: str
    total_charge: float
    metric: PairingMetric
    capabilities: CapabilityStatus = CapabilityStatus()
    _construction_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.equation, ReducedStateEquation):
            raise TypeError("equation must be ReducedStateEquation.")
        if not isinstance(self.model, FieldEnergyFunctional):
            raise TypeError("model must be FieldEnergyFunctional.")
        if not isinstance(self.continuum, ContinuumEnergyFunctional):
            raise TypeError("continuum must be ContinuumEnergyFunctional.")
        if not isinstance(self.metric, PairingMetric):
            raise TypeError("metric must be PairingMetric.")
        if not isinstance(self.capabilities, CapabilityStatus):
            raise TypeError("capabilities must be CapabilityStatus.")
        if self.capabilities.enabled_tiers:
            raise ValueError("common functional capabilities must remain closed.")
        object.__setattr__(
            self,
            "total_charge",
            _finite_scalar(self.total_charge, name="total_charge"),
        )

        definition = get_scalar_definition(self.scalar_id)
        profile = get_solvation_profile(self.profile_id)
        if profile.enabled or profile.capabilities.enabled_tiers:
            raise ValueError(
                "this internal common-stationarity profile must be disabled."
            )
        if (
            profile.scalar_id != self.scalar_id
            or definition.state_equation_id != VARIATIONAL_STATE_EQUATION_ID
            or profile.state_equation_id != VARIATIONAL_STATE_EQUATION_ID
            or self.equation.state_equation_id != VARIATIONAL_STATE_EQUATION_ID
        ):
            raise ValueError("variational scalar/profile/state identities disagree.")
        if definition.implementation_entry_point != (
            VARIATIONAL_COMMON_FUNCTIONAL_ENTRY_POINT
        ):
            raise ValueError("registered scalar does not bind this implementation.")
        if definition.nonpolar_profile != "none" or profile.nonpolar_profile != "none":
            raise ValueError(
                "first common-stationarity kernel must be electrostatic-only."
            )
        if self.model is not self.equation.electronic.functional:
            raise ValueError("equation is bound to a different model scalar.")
        if self.continuum is not self.equation.continuum.functional:
            raise ValueError("equation is bound to a different continuum scalar.")
        if self.equation.electronic.model_profile_id != profile.model_profile:
            raise ValueError("model profile does not match the registry.")
        if self.equation.continuum.continuum_profile_id != profile.continuum_profile:
            raise ValueError("continuum profile does not match the registry.")
        if self.equation.continuum.cavity_profile_id != profile.cavity_profile:
            raise ValueError("cavity profile does not match the registry.")
        if (
            self.equation.continuum.configuration_contract_id
            != profile.continuum_configuration_contract_id
        ):
            raise ValueError("continuum configuration contract does not match.")
        if (
            self.equation.electronic.coupling_id != profile.coupling_id
            or self.equation.continuum.coupling_id != profile.coupling_id
        ):
            raise ValueError("model/continuum coupling identity does not match.")
        if getattr(self.continuum, "scalar_id", None) != self.scalar_id:
            raise ValueError("continuum scalar identity does not match.")

        registered_source = get_source_space(profile.source_space_id)
        registered_field = get_field_dual_space(profile.field_space_id)
        registered_pairing = get_pairing_metric(profile.pairing_id)
        get_coordinate_contract(profile.coordinate_contract_id).validate(
            self.equation.coordinates
        )
        if (
            self.equation.source_space.metadata_hash()
            != registered_source.metadata_hash()
            or self.equation.field_space.metadata_hash()
            != registered_field.metadata_hash()
            or self.metric.metadata_hash() != registered_pairing.metadata_hash()
            or self.metric.metadata_hash()
            != self.equation.field_space.pairing_metric.metadata_hash()
        ):
            raise ValueError(
                "common scalar source/field/pairing registry binding failed."
            )
        coordinates = self.equation.coordinates
        if coordinates.total_charge != self.total_charge:
            raise ValueError("coordinate total charge does not match the scalar.")
        _capabilities_are_closed(self.model, name="model")
        _capabilities_are_closed(self.continuum, name="continuum")
        object.__setattr__(
            self, "_construction_fingerprint", self._current_fingerprint_sha256()
        )

    @property
    def conjugacy_sign(self) -> int:
        return self.model.duality_map.conjugacy_sign

    def _current_fingerprint_sha256(self) -> str:
        payload = {
            "schema": "route2-disabled-variational-common-functional-v1",
            "implementation_sha256": _implementation_sha256(),
            "scalar_id": self.scalar_id,
            "profile_id": self.profile_id,
            "registry_formula": get_scalar_definition(self.scalar_id).exact_formula,
            "equation_sha256": self.equation.fingerprint_sha256(),
            "model_configuration_sha256": _configuration(self.model, name="model"),
            "continuum_configuration_sha256": _configuration(
                self.continuum, name="continuum"
            ),
            "metric_sha256": self.metric.metadata_hash(),
            "duality_map_sha256": self.model.duality_map.configuration_sha256(),
            "conjugacy_sign": self.conjugacy_sign,
            "total_charge": self.total_charge,
            "stationary_lagrangian": "E-s<c,u>_Q+sG",
            "capabilities": "none",
        }
        return _sha(payload)

    def fingerprint_sha256(self) -> str:
        if self.model is not self.equation.electronic.functional:
            raise RuntimeError("variational model/equation binding drifted.")
        if self.continuum is not self.equation.continuum.functional:
            raise RuntimeError("variational continuum/equation binding drifted.")
        if self.metric.metadata_hash() != self.continuum.pairing.metadata_hash():
            raise RuntimeError("variational pairing/continuum binding drifted.")
        _capabilities_are_closed(self.model, name="model")
        _capabilities_are_closed(self.continuum, name="continuum")
        current = self._current_fingerprint_sha256()
        if self._construction_fingerprint and current != self._construction_fingerprint:
            raise RuntimeError("variational common-functional configuration drifted.")
        return current

    def solve_state(
        self,
        geometry: object,
        *,
        root_context_id: str,
        initial_y: object | None = None,
        options: FixedPointOptions = FixedPointOptions(),
        require_convergence: bool = True,
    ) -> FixedPointState:
        self._validate_geometry_charge(geometry)
        return solve_fixed_point(
            self.equation,
            geometry,
            scalar_id=self.scalar_id,
            profile_id=self.profile_id,
            scalar_binding=self,
            root_context_id=root_context_id,
            initial_y=initial_y,
            options=options,
            require_convergence=require_convergence,
        )

    def _validate_geometry_charge(self, geometry: object) -> None:
        if atom_count(geometry) != self.equation.coordinates.atom_count:
            raise ValueError("geometry atom count does not match the state equation.")
        charge = float(model_charge_and_multiplicity(geometry)[0])
        if charge != self.total_charge:
            raise ValueError("geometry charge does not match the state equation.")

    def evaluate(self, geometry: object, y: object) -> VariationalStationaryEnergy:
        self.fingerprint_sha256()
        self._validate_geometry_charge(geometry)
        evaluated = self.equation.evaluate(geometry, y)
        source = np.asarray(evaluated.source, dtype=float)
        field = np.asarray(evaluated.field, dtype=float)
        reduced_field, gauge = self.model.duality_map.decompose_field(
            field,
            atom_count=source.shape[0],
            total_charge=self.total_charge,
        )
        model_energy = self.model.energy_eV(
            geometry,
            reduced_field,
            total_charge=self.total_charge,
            gauge_potential=gauge,
        )
        continuum_energy = self.continuum.energy_eV(geometry, source)
        coupling_energy = self.metric.pair(source, field)
        scalar = VariationalStationaryEnergy(
            scalar_id=self.scalar_id,
            profile_id=self.profile_id,
            conjugacy_sign=self.conjugacy_sign,
            model_energy_eV=float(model_energy),
            continuum_energy_eV=float(continuum_energy),
            coupling_energy_eV=float(coupling_energy),
            gauge_potential_eV_per_e=float(gauge),
            reduced_residual=tuple(float(value) for value in evaluated.residual),
        )
        if not np.isfinite(scalar.total_energy_eV):
            raise ValueError("variational common scalar is non-finite.")
        return scalar

    def evaluate_energy(self, geometry: object, y: object) -> float:
        return self.evaluate(geometry, y).total_energy_eV

    def _validate_state(self, geometry: object, state: FixedPointState) -> None:
        self.fingerprint_sha256()
        self._validate_geometry_charge(geometry)
        if not isinstance(state, FixedPointState) or not state.converged:
            raise ValueError("common-functional evaluation requires a converged state.")
        if state.state_equation_id != VARIATIONAL_STATE_EQUATION_ID:
            raise ValueError("state does not use the variational equation identity.")
        if state.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("state is bound to a different geometry.")
        if state.equation_sha256 != self.equation.fingerprint_sha256():
            raise ValueError("state is bound to a different equation/provider.")
        if state.scalar_id != self.scalar_id or state.profile_id != self.profile_id:
            raise ValueError("state is bound to a different scalar/profile.")
        if state.scalar_sha256 != self.fingerprint_sha256():
            raise ValueError("state is bound to a different scalar configuration.")
        current = self.equation.evaluate(geometry, state.y)
        if not np.array_equal(np.asarray(current.source), state.source_array()):
            raise ValueError("stored source does not match the bound equation.")
        if not np.array_equal(np.asarray(current.field), state.field_array()):
            raise ValueError("stored field does not match the bound equation.")
        current_residual = np.asarray(current.residual)
        if not np.array_equal(
            current_residual, np.asarray(state.actual_unmixed_residual)
        ):
            raise ValueError("stored residual does not match the bound equation.")
        if float(np.linalg.norm(current_residual)) > state.primal_tolerance:
            raise ValueError("state exceeds its recorded primal tolerance.")

    def evaluate_state(
        self, geometry: object, state: FixedPointState
    ) -> VariationalStationaryEnergy:
        self._validate_state(geometry, state)
        return self.evaluate(geometry, state.y)

    def envelope_coordinate_gradient(
        self, geometry: object, state: FixedPointState
    ) -> VariationalEnvelopeGradient:
        """Return the unadmitted stationary-envelope diagnostic.

        ``Q``, ``T``, the charge covector, and the gauge section are static in
        this v1 kernel.  A geometry-dependent duality map must add its explicit
        pullback before it can use this method.
        """

        self._validate_state(geometry, state)
        scalar = self.evaluate(geometry, state.y)
        source = state.source_array()
        field = state.field_array()
        reduced_field, _ = self.model.duality_map.decompose_field(
            field,
            atom_count=source.shape[0],
            total_charge=self.total_charge,
        )
        model_gradient = np.asarray(
            self.model.fixed_field_coordinate_gradient(
                geometry,
                reduced_field,
                total_charge=self.total_charge,
            ),
            dtype=float,
        )
        continuum_gradient = np.asarray(
            self.continuum.coordinate_partial(geometry, source), dtype=float
        )
        if (
            model_gradient.shape != continuum_gradient.shape
            or model_gradient.shape != (source.shape[0], 3)
            or not np.all(np.isfinite(model_gradient))
            or not np.all(np.isfinite(continuum_gradient))
        ):
            raise ValueError("model and continuum coordinate partials must match.")
        total = model_gradient + self.conjugacy_sign * continuum_gradient
        return VariationalEnvelopeGradient(
            scalar=scalar,
            model_coordinate_gradient_eV_per_A=_rows(model_gradient),
            continuum_coordinate_gradient_eV_per_A=_rows(continuum_gradient),
            total_coordinate_gradient_eV_per_A=_rows(total),
            admitted=False,
        )


def build_variational_common_functional(
    model: FieldEnergyFunctional,
    continuum: ContinuumEnergyFunctional,
    geometry: object,
    *,
    scalar_id: str,
    profile_id: str,
) -> VariationalCommonFunctional:
    """Bind derivative-generated model and continuum maps to one state scalar."""

    if not isinstance(model, FieldEnergyFunctional):
        raise TypeError("model must be FieldEnergyFunctional.")
    if not isinstance(continuum, ContinuumEnergyFunctional):
        raise TypeError("continuum must be ContinuumEnergyFunctional.")
    count = atom_count(geometry)
    total_charge = float(model_charge_and_multiplicity(geometry)[0])
    coordinates = model.duality_map.coordinates(
        atom_count=count, total_charge=total_charge
    )
    electronic = ScalarFirstElectronicResponseAdapter(model)
    response = ScalarFirstContinuumResponseAdapter(continuum)
    equation = ReducedStateEquation(
        coordinates,
        electronic,
        response,
        state_equation_id=VARIATIONAL_STATE_EQUATION_ID,
    )
    return VariationalCommonFunctional(
        equation=equation,
        model=model,
        continuum=continuum,
        scalar_id=scalar_id,
        profile_id=profile_id,
        total_charge=total_charge,
        metric=continuum.pairing,
    )


__all__ = [
    "VARIATIONAL_COMMON_FUNCTIONAL_ENTRY_POINT",
    "VariationalCommonFunctional",
    "VariationalEnvelopeGradient",
    "VariationalStationaryEnergy",
    "build_variational_common_functional",
]
