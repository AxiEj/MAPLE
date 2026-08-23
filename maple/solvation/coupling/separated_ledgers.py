"""Two explicit, disabled operational energy ledgers for the separated root.

The fixed point selects a state; it does not uniquely select an energy.  These
classes therefore keep the current vacuum-plus-continuum ledger (Phi0) and the
vacuum-normalized raw conditioned-energy ledger (Phi1Delta) as different IDs.
Neither class asserts that the original checkpoint source is an energy
gradient, and neither enables a public capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.scalar_registry import (
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1,
    OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1,
    get_scalar_definition,
)
from maple.solvation.api.state_registry import (
    SEPARATED_OPERATIONAL_STATE_EQUATION_ID,
)
from maple.solvation.coupling.metrics import MACE_POLAR_RADIAL_GTO_PAIRING
from maple.solvation.release.field_semantics import FieldSemanticsManifest

from .separated_state import SeparatedOperationalStateEquation
from .separated_fixed_point import SeparatedFixedPointState
from .adjoint import AdjointOptions, AdjointResult, solve_reduced_adjoint
from .linearization import SeparatedReducedLinearization
from .state_equation import geometry_sha256, provider_behavior_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@runtime_checkable
class SeparatedVacuumScalarProvider(Protocol):
    provider_id: str
    model_profile_id: str
    provenance_sha256: str

    def configuration_sha256(self) -> str: ...

    def evaluate_energy(self, geometry: Any) -> float: ...


@dataclass(frozen=True, slots=True)
class OperationalLedgerEvaluation:
    scalar_id: str
    state_equation_id: str
    geometry_sha256: str
    equation_sha256: str
    ledger_sha256: str
    field_semantics_sha256: str
    reduced_coordinates_sha256: str
    root_residual_norm: float
    root_tolerance: float
    components_eV: tuple[tuple[str, float], ...]
    total_energy_eV: float

    def __post_init__(self) -> None:
        for name in (
            "scalar_id",
            "state_equation_id",
            "geometry_sha256",
            "equation_sha256",
            "ledger_sha256",
            "field_semantics_sha256",
            "reduced_coordinates_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "geometry_sha256",
            "equation_sha256",
            "ledger_sha256",
            "field_semantics_sha256",
            "reduced_coordinates_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        if not self.components_eV:
            raise ValueError("components_eV must be non-empty.")
        if len({name for name, _ in self.components_eV}) != len(self.components_eV):
            raise ValueError("ledger component names must be unique.")
        if any(
            not isinstance(name, str) or not name.strip() or not np.isfinite(value)
            for name, value in self.components_eV
        ):
            raise ValueError("ledger components must have names and finite values.")
        if not np.isfinite(self.total_energy_eV):
            raise ValueError("total_energy_eV must be finite.")
        if not np.isclose(
            self.total_energy_eV,
            sum(value for _, value in self.components_eV),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("ledger total does not close its immutable components.")
        if (
            not np.isfinite(self.root_residual_norm)
            or self.root_residual_norm < 0.0
            or not np.isfinite(self.root_tolerance)
            or self.root_tolerance <= 0.0
            or self.root_residual_norm > self.root_tolerance
        ):
            raise ValueError("ledger evaluation requires a converged root residual.")

    def as_dict(self) -> dict[str, object]:
        return {
            "scalar_id": self.scalar_id,
            "state_equation_id": self.state_equation_id,
            "geometry_sha256": self.geometry_sha256,
            "equation_sha256": self.equation_sha256,
            "ledger_sha256": self.ledger_sha256,
            "field_semantics_sha256": self.field_semantics_sha256,
            "reduced_coordinates_sha256": self.reduced_coordinates_sha256,
            "root_residual_norm": self.root_residual_norm,
            "root_tolerance": self.root_tolerance,
            "components_eV": dict(self.components_eV),
            "total_energy_eV": self.total_energy_eV,
            "capabilities": "none",
        }


@dataclass(frozen=True, slots=True)
class SeparatedOperationalGradientResult:
    """Complete operational first derivative of one bound separated scalar."""

    ledger: OperationalLedgerEvaluation
    adjoint: AdjointResult
    direct_coordinate_gradient_eV_per_angstrom: tuple[tuple[float, ...], ...]
    residual_coordinate_pullback_eV_per_angstrom: tuple[tuple[float, ...], ...]
    total_coordinate_gradient_eV_per_angstrom: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ledger, OperationalLedgerEvaluation):
            raise TypeError("ledger must be OperationalLedgerEvaluation.")
        if not isinstance(self.adjoint, AdjointResult) or not self.adjoint.converged:
            raise ValueError("a converged adjoint result is required.")
        arrays = tuple(
            np.asarray(getattr(self, name), dtype=float)
            for name in (
                "direct_coordinate_gradient_eV_per_angstrom",
                "residual_coordinate_pullback_eV_per_angstrom",
                "total_coordinate_gradient_eV_per_angstrom",
            )
        )
        if (
            any(array.ndim != 2 or array.shape[1] != 3 for array in arrays)
            or any(not np.all(np.isfinite(array)) for array in arrays)
            or arrays[0].shape != arrays[1].shape
            or arrays[0].shape != arrays[2].shape
        ):
            raise ValueError("coordinate gradients must be finite atom-by-3 arrays.")
        if not np.allclose(arrays[2], arrays[0] - arrays[1], rtol=0.0, atol=1.0e-12):
            raise ValueError("total gradient does not close direct minus residual VJP.")

    def gradient_array(self) -> np.ndarray:
        result = np.asarray(
            self.total_coordinate_gradient_eV_per_angstrom, dtype=float
        )
        result.setflags(write=False)
        return result

    def forces_array(self) -> np.ndarray:
        result = -np.asarray(
            self.total_coordinate_gradient_eV_per_angstrom, dtype=float
        )
        result.setflags(write=False)
        return result


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


class _SeparatedLedgerBase:
    __slots__ = (
        "_configuration_sha256",
        "_sealed",
        "equation",
        "field_semantics_manifest",
        "scalar_id",
        "vacuum",
    )

    implementation_entry_point = ""

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
        vacuum: SeparatedVacuumScalarProvider | None,
        field_semantics_manifest: FieldSemanticsManifest,
        scalar_id: str,
    ) -> None:
        if not isinstance(equation, SeparatedOperationalStateEquation):
            raise TypeError("equation must be SeparatedOperationalStateEquation.")
        definition = get_scalar_definition(scalar_id)
        if definition.state_equation_id != SEPARATED_OPERATIONAL_STATE_EQUATION_ID:
            raise ValueError("scalar is not bound to the separated state equation.")
        if definition.implementation_entry_point != self.implementation_entry_point:
            raise ValueError("scalar registry entry point does not match the ledger.")
        if equation.continuum.scalar_id != scalar_id:
            raise ValueError("continuum snapshot is bound to a different scalar ID.")
        if definition.enabled or definition.admitted_capabilities.enabled_tiers:
            raise ValueError("separated operational ledgers must remain disabled.")
        if not isinstance(field_semantics_manifest, FieldSemanticsManifest):
            raise TypeError("field_semantics_manifest must be FieldSemanticsManifest.")
        if (
            field_semantics_manifest.model_provider_id
            != equation.electronic.provider_id
        ):
            raise ValueError("field-semantics model provider does not match equation.")
        if (
            field_semantics_manifest.model_profile_id
            != equation.electronic.model_profile_id
        ):
            raise ValueError("field-semantics model profile does not match equation.")
        if (
            field_semantics_manifest.adapter_configuration_sha256
            != equation.electronic.configuration_sha256()
            or field_semantics_manifest.adapter_provenance_sha256
            != equation.electronic.provenance_sha256
        ):
            raise ValueError(
                "field-semantics adapter identity does not match equation."
            )
        if (
            getattr(equation.electronic, "checkpoint_sha256", None)
            != field_semantics_manifest.checkpoint_sha256
        ):
            raise ValueError("field-semantics checkpoint does not match equation.")
        canonical_pairing_sha256 = MACE_POLAR_RADIAL_GTO_PAIRING.metadata_hash()
        if (
            getattr(equation.electronic, "field_energy_pairing_sha256", None)
            != canonical_pairing_sha256
            or field_semantics_manifest.pairing_metric_sha256
            != canonical_pairing_sha256
        ):
            raise ValueError("field-semantics native-field pairing does not match.")
        if (
            field_semantics_manifest.source_space_sha256
            != equation.electronic.source_space.metadata_hash()
            or field_semantics_manifest.native_field_space_sha256
            != equation.electronic.receiver_space.metadata_hash()
        ):
            raise ValueError("field-semantics source/receiver spaces do not match.")
        if vacuum is not None:
            for name in ("provider_id", "model_profile_id", "provenance_sha256"):
                value = getattr(vacuum, name, None)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"vacuum.{name} must be a non-empty identity.")
            configuration = getattr(vacuum, "configuration_sha256", None)
            if not callable(configuration) or not callable(
                getattr(vacuum, "evaluate_energy", None)
            ):
                raise TypeError(
                    "vacuum must expose configuration_sha256 and evaluate_energy."
                )
            if vacuum.model_profile_id != equation.electronic.model_profile_id:
                raise ValueError(
                    "vacuum and response must bind the same model profile."
                )
            configuration()
        equation.fingerprint_sha256()
        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "vacuum", vacuum)
        object.__setattr__(self, "field_semantics_manifest", field_semantics_manifest)
        object.__setattr__(self, "scalar_id", scalar_id)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("operational ledger is immutable.")
        object.__setattr__(self, name, value)

    def _current_configuration_sha256(self) -> str:
        electronic_energy_behavior = "not-consumed-by-this-ledger"
        if self.scalar_id == (
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
        ):
            electronic_energy_behavior = provider_behavior_sha256(
                self.equation.electronic,
                ("configuration_sha256", "conditioned_raw_energy_ev"),
                label="separated_conditioned_raw_energy",
            )
        vacuum_record: dict[str, str] | None = None
        if self.vacuum is not None:
            vacuum_record = {
                "provider_id": self.vacuum.provider_id,
                "model_profile_id": self.vacuum.model_profile_id,
                "provenance_sha256": self.vacuum.provenance_sha256,
                "configuration_sha256": self.vacuum.configuration_sha256(),
                "behavior_sha256": provider_behavior_sha256(
                    self.vacuum,
                    ("configuration_sha256", "evaluate_energy"),
                    label="separated_vacuum",
                ),
            }
        payload = {
            "scalar_id": self.scalar_id,
            "registry_formula": get_scalar_definition(self.scalar_id).exact_formula,
            "implementation_entry_point": self.implementation_entry_point,
            "equation_sha256": self.equation.fingerprint_sha256(),
            "field_semantics_sha256": (
                self.field_semantics_manifest.configuration_sha256()
            ),
            "electronic_energy_behavior_sha256": electronic_energy_behavior,
            "vacuum": vacuum_record,
            "capabilities": "none",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("separated operational ledger configuration drifted.")
        return current

    def _components(
        self,
        geometry: object,
        source: np.ndarray,
        boundary_state: np.ndarray,
        native_field: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        raise NotImplementedError

    def evaluate_root(
        self,
        geometry: object,
        reduced_coordinates: object,
        *,
        root_tolerance: float,
    ) -> OperationalLedgerEvaluation:
        self.configuration_sha256()
        if not np.isfinite(root_tolerance) or root_tolerance <= 0.0:
            raise ValueError("root_tolerance must be finite and positive.")
        if geometry_sha256(geometry) != self.equation.continuum.geometry_sha256:
            raise ValueError("geometry does not match the ledger continuum snapshot.")
        y = np.asarray(reduced_coordinates, dtype=float)
        if y.shape != (self.equation.reduced_dimension,) or not np.all(np.isfinite(y)):
            raise ValueError("reduced_coordinates have an invalid shape.")
        residual = self.equation.residual(geometry, y)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > root_tolerance:
            raise ValueError(
                "operational ledger refuses a state above the root tolerance."
            )
        source = self.equation.source(y)
        boundary = self.equation.boundary_state(y)
        field = self.equation.continuum.native_field_from_boundary(boundary)
        components = self._components(geometry, source, boundary, field)
        total = float(sum(value for _, value in components))
        return OperationalLedgerEvaluation(
            scalar_id=self.scalar_id,
            state_equation_id=self.equation.state_equation_id,
            geometry_sha256=geometry_sha256(geometry),
            equation_sha256=self.equation.fingerprint_sha256(),
            ledger_sha256=self.configuration_sha256(),
            field_semantics_sha256=(
                self.field_semantics_manifest.configuration_sha256()
            ),
            reduced_coordinates_sha256=_array_sha256(y),
            root_residual_norm=residual_norm,
            root_tolerance=float(root_tolerance),
            components_eV=components,
            total_energy_eV=total,
        )

    def _validate_primal_state(
        self, geometry: object, state: SeparatedFixedPointState
    ) -> None:
        if not isinstance(state, SeparatedFixedPointState):
            raise TypeError("state must be SeparatedFixedPointState.")
        if not state.converged:
            raise ValueError("implicit differentiation requires a converged root.")
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
        source = self.equation.source(y)
        boundary = self.equation.boundary_state(y)
        field = self.equation.continuum.native_field_from_boundary(boundary)
        residual = self.equation.residual(geometry, y)
        for name, current, stored in (
            ("source", source, state.source_array()),
            ("boundary state", boundary, state.boundary_state_array()),
            ("native field", field, state.field_array()),
            ("residual", residual, state.actual_unmixed_residual),
        ):
            if not np.array_equal(current, np.asarray(stored, dtype=float)):
                raise ValueError(f"stored primal {name} does not match the equation.")
        if float(np.linalg.norm(residual)) > state.primal_tolerance:
            raise ValueError("primal residual exceeds its recorded tolerance.")


class _FrozenVacuumContinuumLedgerBase(_SeparatedLedgerBase):
    """Shared Phi0 implementation using the continuum-owned scalar contract."""

    continuum_component_name = "smooth_harmonic_continuum_stationary_energy"

    def _components(
        self,
        geometry: object,
        source: np.ndarray,
        boundary_state: np.ndarray,
        native_field: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        del boundary_state, native_field
        if self.vacuum is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Phi0 vacuum provider is absent.")
        vacuum = float(self.vacuum.evaluate_energy(geometry))
        continuum = float(self.equation.continuum.continuum_energy_eV(source))
        if not np.isfinite(vacuum) or not np.isfinite(continuum):
            raise RuntimeError("Phi0 ledger produced a non-finite component.")
        return (
            ("macepolar_vacuum_energy", vacuum),
            (self.continuum_component_name, continuum),
        )

    def reduced_gradient(
        self, geometry: object, reduced_coordinates: object
    ) -> np.ndarray:
        """Return ``T.T @ dG/dc`` for the selected Phi0 ledger.

        The fixed-point state and the scalar are deliberately distinct.  This
        method supplies the exact reduced scalar gradient required on the
        right-hand side of the operational implicit adjoint; it does not make
        the electronic state equation stationary or variational.
        """

        self.configuration_sha256()
        if geometry_sha256(geometry) != self.equation.continuum.geometry_sha256:
            raise ValueError("geometry does not match the ledger continuum snapshot.")
        y = np.asarray(reduced_coordinates, dtype=float)
        if y.shape != (self.equation.reduced_dimension,) or not np.all(
            np.isfinite(y)
        ):
            raise ValueError("reduced_coordinates have an invalid shape.")
        source_gradient = self.equation.continuum.continuum_source_gradient(
            self.equation.source(y)
        )
        result = self.equation.coordinates.reduce_source_cotangent(
            source_gradient
        )
        if result.shape != (self.equation.reduced_dimension,) or not np.all(
            np.isfinite(result)
        ):
            raise RuntimeError("Phi0 reduced gradient is invalid.")
        return result


class FrozenVacuumContinuumLedger(_FrozenVacuumContinuumLedgerBase):
    """Legacy smooth-harmonic CPCM Phi0 ledger."""

    implementation_entry_point = (
        "maple.solvation.coupling.separated_ledgers:FrozenVacuumContinuumLedger"
    )

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
        vacuum: SeparatedVacuumScalarProvider,
        field_semantics_manifest: FieldSemanticsManifest,
    ) -> None:
        super().__init__(
            equation=equation,
            vacuum=vacuum,
            field_semantics_manifest=field_semantics_manifest,
            scalar_id=(
                OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
            ),
        )



class HarmonicDDPCMFrozenVacuumLedger(_FrozenVacuumContinuumLedgerBase):
    """Phi0 for finite-dielectric smooth harmonic ddPCM.

    The continuum provider owns ``G_ddPCM=1/2 c.T C X``.  This class does not
    reinterpret the composite ddPCM boundary variables as a symmetric CPCM
    surface charge and therefore does not reuse the old CPCM half-coupling
    formula or scalar identity.
    """

    implementation_entry_point = (
        "maple.solvation.coupling.separated_ledgers:"
        "HarmonicDDPCMFrozenVacuumLedger"
    )
    continuum_component_name = "smooth_harmonic_ddpcm_stationary_energy"

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
        vacuum: SeparatedVacuumScalarProvider,
        field_semantics_manifest: FieldSemanticsManifest,
    ) -> None:
        scalar_id = equation.continuum.scalar_id
        if scalar_id not in (
            OPERATIONAL_MACEPOLAR_SEPARATED_PHI0_SMOOTH_HARMONIC_DDPCM_V1,
            OPERATIONAL_MACEPOLAR_GTO1P5_NATIVEFIELD8_SMOOTH_HARMONIC_DDPCM_PHI0_V2,
        ):
            raise ValueError("continuum is not bound to a supported ddPCM Phi0 scalar.")
        super().__init__(
            equation=equation,
            vacuum=vacuum,
            field_semantics_manifest=field_semantics_manifest,
            scalar_id=scalar_id,
        )

    def direct_coordinate_gradient(
        self, geometry: object, reduced_coordinates: object
    ) -> np.ndarray:
        """Return ``partial_R(E_vac+G_ddPCM)`` at fixed reduced state."""

        self.configuration_sha256()
        coordinate_gradient = getattr(self.vacuum, "coordinate_gradient", None)
        continuum_gradient = getattr(
            self.equation.continuum,
            "continuum_energy_position_gradient",
            None,
        )
        if not callable(coordinate_gradient) or not callable(continuum_gradient):
            raise TypeError(
                "Phi0 force requires vacuum and continuum coordinate gradients."
            )
        y = np.asarray(reduced_coordinates, dtype=float)
        if y.shape != (self.equation.reduced_dimension,) or not np.all(
            np.isfinite(y)
        ):
            raise ValueError("reduced_coordinates have an invalid shape.")
        vacuum = np.asarray(coordinate_gradient(geometry), dtype=float)
        continuum = np.asarray(
            continuum_gradient(self.equation.source(y)), dtype=float
        )
        expected = (self.equation.coordinates.atom_count, 3)
        if (
            vacuum.shape != expected
            or continuum.shape != expected
            or not np.all(np.isfinite(vacuum))
            or not np.all(np.isfinite(continuum))
        ):
            raise ValueError(
                "vacuum and continuum coordinate gradients must be finite "
                f"with shape {expected}."
            )
        return vacuum + continuum

    def implicit_gradient(
        self,
        geometry: object,
        state: SeparatedFixedPointState,
        *,
        adjoint_options: AdjointOptions = AdjointOptions(),
    ) -> SeparatedOperationalGradientResult:
        """Differentiate the registered Phi0 scalar along its operational root."""

        self.configuration_sha256()
        self._validate_primal_state(geometry, state)
        ledger = self.evaluate_root(
            geometry,
            state.y_array(),
            root_tolerance=state.primal_tolerance,
        )
        reduced_gradient = self.reduced_gradient(geometry, state.y_array())
        linearization = SeparatedReducedLinearization.at(
            self.equation, geometry, state.y_array()
        )
        adjoint = solve_reduced_adjoint(
            linearization, reduced_gradient, options=adjoint_options
        )
        direct = self.direct_coordinate_gradient(geometry, state.y_array())
        residual_pullback = np.asarray(
            self.equation.coordinate_vjp(
                geometry, state.y_array(), adjoint.solution_array()
            ),
            dtype=float,
        )
        if residual_pullback.shape != direct.shape or not np.all(
            np.isfinite(residual_pullback)
        ):
            raise ValueError(
                "residual and direct coordinate gradients must have equal shape."
            )
        total = direct - residual_pullback
        as_tuple = lambda values: tuple(  # noqa: E731
            tuple(float(value) for value in row) for row in values
        )
        return SeparatedOperationalGradientResult(
            ledger=ledger,
            adjoint=adjoint,
            direct_coordinate_gradient_eV_per_angstrom=as_tuple(direct),
            residual_coordinate_pullback_eV_per_angstrom=as_tuple(
                residual_pullback
            ),
            total_coordinate_gradient_eV_per_angstrom=as_tuple(total),
        )


class NormalizedPhi1DeltaLedger(_SeparatedLedgerBase):
    """Phi1Delta: vacuum plus raw conditioned-energy change and self energy."""

    implementation_entry_point = (
        "maple.solvation.coupling.separated_ledgers:NormalizedPhi1DeltaLedger"
    )

    def __init__(
        self,
        *,
        equation: SeparatedOperationalStateEquation,
        vacuum: SeparatedVacuumScalarProvider,
        field_semantics_manifest: FieldSemanticsManifest,
    ) -> None:
        if not callable(
            getattr(equation.electronic, "conditioned_raw_energy_ev", None)
        ):
            raise TypeError(
                "Phi1Delta electronic provider must expose conditioned_raw_energy_ev."
            )
        super().__init__(
            equation=equation,
            vacuum=vacuum,
            field_semantics_manifest=field_semantics_manifest,
            scalar_id=(
                OPERATIONAL_MACEPOLAR_SEPARATED_PHI1_SMOOTH_HARMONIC_GALERKIN_CPCM_V1
            ),
        )

    def _components(
        self,
        geometry: object,
        source: np.ndarray,
        boundary_state: np.ndarray,
        native_field: np.ndarray,
    ) -> tuple[tuple[str, float], ...]:
        del source
        if self.vacuum is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Phi1Delta vacuum provider is absent.")
        vacuum = float(self.vacuum.evaluate_energy(geometry))
        zero_field = np.zeros_like(native_field)
        conditioned = float(
            self.equation.electronic.conditioned_raw_energy_ev(geometry, native_field)
        )
        conditioned_zero = float(
            self.equation.electronic.conditioned_raw_energy_ev(geometry, zero_field)
        )
        conditioned_delta = conditioned - conditioned_zero
        self_energy = 0.5 * float(
            np.vdot(
                boundary_state,
                self.equation.continuum.surface_operator @ boundary_state,
            )
        )
        if not np.all(np.isfinite((vacuum, conditioned_delta, self_energy))):
            raise RuntimeError("Phi1Delta ledger produced a non-finite component.")
        return (
            ("macepolar_vacuum_energy", vacuum),
            ("macepolar_conditioned_raw_energy_difference", conditioned_delta),
            ("continuum_polarization_self_energy", self_energy),
        )


# Compatibility import only.  The registered scalar and implementation entry
# point use the vacuum-normalized class name above.
ExternalEnthalpyOperationalLedger = NormalizedPhi1DeltaLedger


__all__ = [
    "ExternalEnthalpyOperationalLedger",
    "FrozenVacuumContinuumLedger",
    "HarmonicDDPCMFrozenVacuumLedger",
    "NormalizedPhi1DeltaLedger",
    "OperationalLedgerEvaluation",
    "SeparatedOperationalGradientResult",
    "SeparatedVacuumScalarProvider",
]
