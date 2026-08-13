"""Canonical operational scalar for the fixed-topology electrostatic profile."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_CPCM_ELECTROSTATIC_V1,
    get_scalar_definition,
)
from maple.solvation.api.profiles import (
    OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1,
    UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
    get_solvation_profile,
)

from .adjoint import AdjointOptions, AdjointResult, solve_reduced_adjoint
from .fixed_point import FixedPointState
from .linearization import ReducedLinearization
from .metrics import PairingMetric, get_pairing_metric
from .spaces import (
    get_coordinate_contract,
    get_field_dual_space,
    get_source_space,
)
from .state_equation import (
    ContinuumResponseProvider,
    ReducedStateEquation,
    geometry_sha256,
    provider_behavior_sha256,
)


@runtime_checkable
class VacuumScalarProvider(Protocol):
    """Vacuum scalar and its partial Cartesian derivative."""

    provider_id: str
    model_profile_id: str
    provenance_sha256: str

    def configuration_sha256(self) -> str: ...

    def evaluate_energy(self, geometry: Any) -> float: ...
    def coordinate_gradient(self, geometry: Any) -> np.ndarray: ...


@runtime_checkable
class FixedTopologyReciprocalLinearContinuum(ContinuumResponseProvider, Protocol):
    """Provider declaration required by the canonical Phase-3 scalar."""

    fixed_topology: bool
    linear_response: bool
    reciprocal: bool


def _source_array(
    values: object, atom_count: int, component_count: int, name: str
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    expected = (atom_count, component_count)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must be finite with shape {expected}; received {result.shape}."
        )
    return np.array(result, copy=True)


def _source_dual_of_field(field: np.ndarray, metric: PairingMetric) -> np.ndarray:
    return np.einsum("ij,nj->ni", metric.block, field)


def _field_cotangent_of_source(source: np.ndarray, metric: PairingMetric) -> np.ndarray:
    return np.einsum("ji,nj->ni", metric.block, source)


def _validate_continuum_pairing_identity(
    continuum: ContinuumResponseProvider, metric: PairingMetric
) -> None:
    source_space = getattr(continuum, "source_space", None)
    field_space = getattr(continuum, "field_space", None)
    if source_space is None or field_space is None:
        raise TypeError(
            "Continuum must declare source_space and field_space identities."
        )
    if (
        source_space.components != metric.source_components
        or source_space.units != metric.source_units
    ):
        raise ValueError(
            "Continuum source-space identity does not match the pairing metric."
        )
    if field_space.source_space.metadata_hash() != source_space.metadata_hash():
        raise ValueError(
            "Continuum field space is paired with a different source identity."
        )
    if field_space.pairing_metric.metadata_hash() != metric.metadata_hash():
        raise ValueError("Continuum field pairing identity does not match exactly.")
    if (
        field_space.components != metric.field_components
        or field_space.units != metric.field_units
    ):
        raise ValueError("Continuum field order/units do not match the pairing metric.")


def _continuum_metric(
    continuum: ContinuumResponseProvider,
    metric: PairingMetric | None,
) -> PairingMetric:
    if metric is not None:
        if not isinstance(metric, PairingMetric):
            raise TypeError("metric must be a PairingMetric or None.")
        return metric
    field_space = getattr(continuum, "field_space", None)
    inferred = getattr(field_space, "pairing_metric", None)
    if not isinstance(inferred, PairingMetric):
        raise TypeError(
            "metric=None requires a continuum with an authoritative field pairing."
        )
    return inferred


@dataclass(frozen=True)
class HalfCouplingEvaluation:
    energy: float
    direct_source_gradient: tuple[tuple[float, ...], ...]
    response_source_gradient: tuple[tuple[float, ...], ...]
    total_source_gradient: tuple[tuple[float, ...], ...]
    coordinate_gradient: tuple[float, ...]

    def total_source_gradient_array(self) -> np.ndarray:
        result = np.asarray(self.total_source_gradient, dtype=float)
        result.setflags(write=False)
        return result

    def coordinate_gradient_array(self) -> np.ndarray:
        result = np.asarray(self.coordinate_gradient, dtype=float)
        result.setflags(write=False)
        return result


def nonlinear_half_coupling(
    continuum: ContinuumResponseProvider,
    geometry: Any,
    source: object,
    *,
    metric: PairingMetric | None = None,
) -> HalfCouplingEvaluation:
    """Evaluate ``0.5 <c,P_R(c)>_Q`` with direct plus ``J_P.T`` response.

    The expression is valid for nonlinear response maps.  For an admitted
    linear reciprocal map the two source-gradient halves are identical and
    their sum is ``Q P_R(c)``.
    """

    metric = _continuum_metric(continuum, metric)
    _validate_continuum_pairing_identity(continuum, metric)
    values = np.asarray(source, dtype=float)
    if values.ndim != 2 or values.shape[1] != metric.component_count:
        raise ValueError("source must have shape (atom_count, metric.component_count).")
    if not np.all(np.isfinite(values)):
        raise ValueError("source must be finite.")
    atom_count = values.shape[0]
    field = _source_array(
        continuum.evaluate_field(geometry, values),
        atom_count,
        metric.component_count,
        "continuum field",
    )
    energy = 0.5 * metric.pair(values, field)
    direct = 0.5 * _source_dual_of_field(field, metric)
    field_cotangent = 0.5 * _field_cotangent_of_source(values, metric)
    response = _source_array(
        continuum.source_vjp(geometry, values, field_cotangent),
        atom_count,
        metric.component_count,
        "half-coupling response source gradient",
    )
    coordinate = np.asarray(
        continuum.coordinate_vjp(geometry, values, field_cotangent), dtype=float
    )
    if coordinate.ndim != 1 or not np.all(np.isfinite(coordinate)):
        raise ValueError("continuum coordinate VJP must be a finite vector.")
    total = direct + response
    return HalfCouplingEvaluation(
        energy=float(energy),
        direct_source_gradient=tuple(tuple(float(v) for v in row) for row in direct),
        response_source_gradient=tuple(
            tuple(float(v) for v in row) for row in response
        ),
        total_source_gradient=tuple(tuple(float(v) for v in row) for row in total),
        coordinate_gradient=tuple(float(v) for v in coordinate),
    )


def reciprocal_linear_half_coupling(
    continuum: ContinuumResponseProvider,
    geometry: Any,
    source: object,
    *,
    metric: PairingMetric | None = None,
    reciprocity_tolerance: float = 1.0e-10,
) -> HalfCouplingEvaluation:
    """Canonical linear reciprocal half coupling, rejecting broken reciprocity."""

    for declaration in ("fixed_topology", "linear_response", "reciprocal"):
        if getattr(continuum, declaration, False) is not True:
            raise ValueError(
                "Canonical operational half coupling requires an explicitly "
                f"declared {declaration}=True continuum provider."
            )

    result = nonlinear_half_coupling(continuum, geometry, source, metric=metric)
    direct = np.asarray(result.direct_source_gradient)
    response = np.asarray(result.response_source_gradient)
    error = float(np.linalg.norm(direct - response))
    scale = max(1.0, float(np.linalg.norm(direct)), float(np.linalg.norm(response)))
    if error > reciprocity_tolerance * scale:
        raise ValueError(
            "Continuum response is not reciprocal under the authoritative pairing: "
            f"gradient-half error={error:.6e}."
        )
    return result


@dataclass(frozen=True)
class OperationalScalarEvaluation:
    scalar_id: str
    vacuum_energy: float
    continuum_energy: float
    total_energy: float
    reduced_gradient: tuple[float, ...]
    direct_coordinate_gradient: tuple[float, ...]


@dataclass(frozen=True)
class OperationalGradientResult:
    scalar: OperationalScalarEvaluation
    adjoint: AdjointResult
    total_coordinate_gradient: tuple[float, ...]

    @property
    def forces(self) -> tuple[float, ...]:
        return tuple(-value for value in self.total_coordinate_gradient)


@dataclass(frozen=True)
class OperationalElectrostaticScalar:
    """``E_vac + 0.5<c,P_R(c)>_Q`` evaluated along the admitted root."""

    equation: ReducedStateEquation
    vacuum: VacuumScalarProvider
    scalar_id: str = OPERATIONAL_CPCM_ELECTROSTATIC_V1
    profile_id: str = OPERATIONAL_CPCM_ELECTROSTATIC_PROFILE_V1
    metric: PairingMetric | None = None
    _construction_fingerprint: str = ""

    def __post_init__(self) -> None:
        definition = get_scalar_definition(self.scalar_id)
        profile = get_solvation_profile(self.profile_id)
        registered_source = get_source_space(profile.source_space_id)
        registered_field = get_field_dual_space(profile.field_space_id)
        registered_metric = get_pairing_metric(profile.pairing_id)
        registered_coordinates = get_coordinate_contract(profile.coordinate_contract_id)
        registered_coordinates.validate(self.equation.coordinates)
        metric = registered_metric if self.metric is None else self.metric
        if not isinstance(metric, PairingMetric):
            raise TypeError("metric must be a PairingMetric or None.")
        object.__setattr__(self, "metric", metric)
        if (
            profile.scalar_id != self.scalar_id
            or profile.state_equation_id != self.equation.state_equation_id
        ):
            raise ValueError(
                "Operational scalar/profile/state registry binding is inconsistent."
            )
        if definition.implementation_entry_point != (
            "maple.solvation.coupling.energy:OperationalElectrostaticScalar"
        ):
            raise ValueError(
                "Registered scalar is not implemented by OperationalElectrostaticScalar."
            )
        if definition.nonpolar_profile != "none":
            raise ValueError(
                "OperationalElectrostaticScalar cannot implement a nonpolar scalar."
            )
        if self.equation.electronic.model_profile_id != profile.model_profile:
            raise ValueError(
                "Electronic model-profile ID does not match the authoritative profile."
            )
        if self.equation.continuum.continuum_profile_id != profile.continuum_profile:
            raise ValueError(
                "Continuum profile ID does not match the authoritative profile."
            )
        if self.equation.continuum.cavity_profile_id != profile.cavity_profile:
            raise ValueError(
                "Continuum cavity-profile ID does not match the authoritative profile."
            )
        continuum_configuration_contract_id = getattr(
            self.equation.continuum,
            "configuration_contract_id",
            UNBOUND_CONTINUUM_CONFIGURATION_CONTRACT_ID,
        )
        if (
            continuum_configuration_contract_id
            != profile.continuum_configuration_contract_id
        ):
            raise ValueError(
                "Continuum physical-configuration contract does not match the "
                "authoritative profile."
            )
        if (
            getattr(self.equation.continuum, "scalar_id", self.scalar_id)
            != self.scalar_id
        ):
            raise ValueError(
                "Continuum scalar ID does not match the operational scalar identity."
            )
        if (
            self.equation.electronic.coupling_id != profile.coupling_id
            or self.equation.continuum.coupling_id != profile.coupling_id
        ):
            raise ValueError(
                "Source/receiver coupling ID does not match the authoritative profile."
            )
        if self.vacuum.model_profile_id != profile.model_profile:
            raise ValueError(
                "Vacuum model-profile ID does not match the authoritative profile."
            )
        if definition.state_equation_id != self.equation.state_equation_id:
            raise ValueError("Scalar registry and state equation IDs do not match.")
        if self.metric.metadata_hash() != registered_metric.metadata_hash():
            raise ValueError(
                "Operational scalar metric does not match the profile registry."
            )
        if (
            self.metric.metadata_hash()
            != self.equation.field_space.pairing_metric.metadata_hash()
        ):
            raise ValueError(
                "Operational scalar metric must equal the equation field pairing."
            )
        if (
            self.equation.source_space.metadata_hash()
            != self.equation.field_space.source_space.metadata_hash()
        ):
            raise ValueError(
                "Operational scalar source and field identities do not match."
            )
        if (
            self.equation.source_space.metadata_hash()
            != registered_source.metadata_hash()
            or self.equation.field_space.metadata_hash()
            != registered_field.metadata_hash()
        ):
            raise ValueError(
                "Canonical operational scalar requires the exact authoritative "
                "source/field identities registered by its profile."
            )
        for name in ("provider_id", "provenance_sha256"):
            value = getattr(self.vacuum, name, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"vacuum.{name} must be a non-empty stable identity.")
        digest = self.vacuum.provenance_sha256.lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "vacuum.provenance_sha256 must contain 64 hexadecimal digits."
            )
        if not callable(getattr(self.vacuum, "configuration_sha256", None)):
            raise TypeError("vacuum.configuration_sha256 must be callable.")
        object.__setattr__(
            self, "_construction_fingerprint", self._current_fingerprint_sha256()
        )

    def _current_fingerprint_sha256(self) -> str:
        payload = {
            "scalar_id": self.scalar_id,
            "profile_id": self.profile_id,
            "registry_formula": get_scalar_definition(self.scalar_id).exact_formula,
            "equation_sha256": self.equation.fingerprint_sha256(),
            "metric_sha256": self.metric.metadata_hash(),
            "vacuum": {
                "provider_id": self.vacuum.provider_id,
                "model_profile_id": self.vacuum.model_profile_id,
                "provenance_sha256": self.vacuum.provenance_sha256,
                "configuration_sha256": self.vacuum.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.vacuum,
                    (
                        "configuration_sha256",
                        "evaluate_energy",
                        "coordinate_gradient",
                    ),
                    label="vacuum",
                ),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint_sha256(self) -> str:
        current = self._current_fingerprint_sha256()
        if self._construction_fingerprint and current != self._construction_fingerprint:
            raise ValueError(
                "Operational scalar/provider configuration drifted after construction."
            )
        return current

    def evaluate_energy(self, geometry: Any, y: object) -> float:
        """Evaluate only the canonical scalar, without any derivative work.

        Real-stack finite-difference panels require many energy-only displaced
        roots.  This is not a second ledger: it is the same registered
        ``E_vac + 0.5<c,P_R(c)>_Q`` entry point, deliberately stopping before
        source VJP and coordinate VJP evaluation.
        """

        self.fingerprint_sha256()
        for declaration in ("fixed_topology", "linear_response", "reciprocal"):
            if getattr(self.equation.continuum, declaration, False) is not True:
                raise ValueError(
                    "Canonical operational energy requires an explicitly "
                    f"declared {declaration}=True continuum provider."
                )
        _validate_continuum_pairing_identity(self.equation.continuum, self.metric)
        reduced = self.equation._y(y)
        source = self.equation.coordinates.expand(reduced)
        field = _source_array(
            self.equation.continuum.evaluate_field(geometry, source),
            source.shape[0],
            self.metric.component_count,
            "continuum field",
        )
        continuum_energy = 0.5 * self.metric.pair(source, field)
        vacuum_energy = float(self.vacuum.evaluate_energy(geometry))
        total = vacuum_energy + continuum_energy
        if not all(
            np.isfinite(value) for value in (vacuum_energy, continuum_energy, total)
        ):
            raise ValueError("Operational scalar energy terms must be finite.")
        return float(total)

    def evaluate(self, geometry: Any, y: object) -> OperationalScalarEvaluation:
        self.fingerprint_sha256()
        reduced = self.equation._y(y)
        source = self.equation.coordinates.expand(reduced)
        continuum = reciprocal_linear_half_coupling(
            self.equation.continuum, geometry, source, metric=self.metric
        )
        vacuum_energy = float(self.vacuum.evaluate_energy(geometry))
        if not np.isfinite(vacuum_energy):
            raise ValueError("vacuum energy must be finite.")
        vacuum_gradient = np.asarray(
            self.vacuum.coordinate_gradient(geometry), dtype=float
        )
        continuum_gradient = continuum.coordinate_gradient_array()
        if vacuum_gradient.shape != continuum_gradient.shape or not np.all(
            np.isfinite(vacuum_gradient)
        ):
            raise ValueError(
                "Vacuum and continuum coordinate gradients must be finite and equally shaped."
            )
        source_gradient = continuum.total_source_gradient_array()
        reduced_gradient = self.equation.coordinates.reduce_source_cotangent(
            source_gradient
        )
        direct_coordinate_gradient = vacuum_gradient + continuum_gradient
        return OperationalScalarEvaluation(
            scalar_id=self.scalar_id,
            vacuum_energy=vacuum_energy,
            continuum_energy=continuum.energy,
            total_energy=vacuum_energy + continuum.energy,
            reduced_gradient=tuple(float(value) for value in reduced_gradient),
            direct_coordinate_gradient=tuple(
                float(value) for value in direct_coordinate_gradient
            ),
        )

    def implicit_gradient(
        self,
        geometry: Any,
        state: FixedPointState,
        *,
        adjoint_options: AdjointOptions = AdjointOptions(),
    ) -> OperationalGradientResult:
        if not state.converged:
            raise ValueError(
                "Implicit differentiation requires a converged primal state."
            )
        if state.state_equation_id != self.equation.state_equation_id:
            raise ValueError(
                "Primal state and operational scalar use different state equations."
            )
        if state.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("Primal state is bound to a different geometry.")
        if state.equation_sha256 != self.equation.fingerprint_sha256():
            raise ValueError(
                "Primal state is bound to a different provider/equation identity."
            )
        if state.scalar_id != self.scalar_id or state.profile_id != self.profile_id:
            raise ValueError(
                "Primal state is bound to a different scalar/profile identity."
            )
        if state.scalar_sha256 != self.fingerprint_sha256():
            raise ValueError(
                "Primal state is bound to a different scalar/vacuum configuration."
            )
        current = self.equation.evaluate(geometry, state.y)
        current_source = np.asarray(current.source)
        current_field = np.asarray(current.field)
        current_residual = np.asarray(current.residual)
        if not np.array_equal(current_source, state.source_array()):
            raise ValueError(
                "Stored primal source does not match the bound equation evaluation."
            )
        if not np.array_equal(current_field, state.field_array()):
            raise ValueError(
                "Stored primal field does not match the bound equation evaluation."
            )
        if not np.array_equal(
            current_residual, np.asarray(state.actual_unmixed_residual)
        ):
            raise ValueError(
                "Stored primal residual does not match the bound equation evaluation."
            )
        physical_residual_norm = float(np.linalg.norm(current_residual))
        if physical_residual_norm > state.primal_tolerance:
            raise ValueError(
                "Bound primal state exceeds its recorded physical residual tolerance."
            )
        scalar = self.evaluate(geometry, state.y)
        linearization = ReducedLinearization.at(self.equation, geometry, state.y)
        adjoint = solve_reduced_adjoint(
            linearization, scalar.reduced_gradient, options=adjoint_options
        )
        residual_coordinate_pullback = self.equation.coordinate_vjp(
            geometry, state.y, adjoint.solution
        )
        direct = np.asarray(scalar.direct_coordinate_gradient)
        if direct.shape != residual_coordinate_pullback.shape:
            raise ValueError(
                "Scalar and residual coordinate gradients must be equally shaped."
            )
        total = direct - residual_coordinate_pullback
        return OperationalGradientResult(
            scalar=scalar,
            adjoint=adjoint,
            total_coordinate_gradient=tuple(float(value) for value in total),
        )


__all__ = [
    "HalfCouplingEvaluation",
    "FixedTopologyReciprocalLinearContinuum",
    "OperationalElectrostaticScalar",
    "OperationalGradientResult",
    "OperationalScalarEvaluation",
    "VacuumScalarProvider",
    "nonlinear_half_coupling",
    "reciprocal_linear_half_coupling",
]
