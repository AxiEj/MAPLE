"""Self-consistent canonical-ADT MDP/POLAR hybrid with separated ddX.

This target-free research evaluator closes the operational state equation

``u = K_R(delta_c_radial(u), p_ADT(u); c_perm)``

without collapsing the three physical source categories.  It provides exact
matrix-free JVP/VJP actions of the fixed-geometry state map, deterministic
five-start root replay, and a bounded dense local root certificate.  The
selected scalar is still the explicit operational ledger

``E_op = E_vac^POLAR(R) + G_ddX(R, c_MDP, delta_c_radial, p_ADT)``.

The original MACE-POLAR field energy is not claimed to be conjugate to the
source, and no public E/F/H/V/M capability is admitted.  Coordinate derivatives
remain blocked until the MDP permanent source, geometry-dependent ADT chart,
and moving ddX cavity are differentiated together.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Protocol, runtime_checkable

import numpy as np

from maple.solvation.api.capabilities import CapabilityStatus
from maple.solvation.continuum.separated_source_adt_ddx import (
    CanonicalADTSeparatedDDXState,
    PreparedCanonicalADTSeparatedDDX,
    prepare_canonical_adt_separated_ddx,
)
from maple.solvation.continuum.separated_source_ddx import SeparatedSourceDDXBackend
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.spaces import ATOMIC_L1_SOURCE_SPACE
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.models.base import atom_count
from maple.solvation.models.mace_mdp_polar_adt import (
    CanonicalADTInducedComponents,
    MACEPolarZeroFieldPointPermanentSource,
    MDP_POLAR_CANONICAL_ADT_CONTRACT,
    MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT,
    MDPPolarCanonicalADTResponse,
)
from maple.solvation.release.root_well_posedness import (
    RootWellPosednessCertificate,
    certify_root_well_posedness,
    dense_state_map_jacobian,
)

CANONICAL_ADT_HYBRID_DDX_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-canonical-adt-ddx.impl.v1"
)
CANONICAL_ADT_HYBRID_DDX_PROFILE_ID = (
    "route2-research-mace-mdp-point-polar-residual-adt-ddx-operational-v1"
)
CANONICAL_ADT_HYBRID_DDX_ROOT_CONTRACT = (
    "mace-mdp-polar-canonical-adt-ddx-five-start-anderson-root-v1"
)
ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-role-separated-adt-ddx.impl.v2"
)
ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID = "route2-research-mace-mdp-point-polar-residual-role-separated-adt-ddx-operational-v2"
POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID = (
    "maple.route2.experimental.macepolar-zero-point-mdp-alpha-"
    "role-separated-adt-ddx.impl.v3"
)
POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID = (
    "route2-research-macepolar-zero-point-mdp-alpha-polar-residual-"
    "role-separated-adt-ddx-operational-v3"
)

ROOT_TOLERANCE_EV = 1.0e-10
ROOT_REPLAY_FIELD_ATOL_EV = 3.0e-9
ROOT_REPLAY_ENERGY_ATOL_EV = 2.0e-10
MAX_ROOT_ITERATIONS = 100
ANDERSON_HISTORY = 6
ANDERSON_DAMPING = 0.70
MAX_DENSE_AUDIT_DIMENSION = 128
TOTAL_CHARGE_ATOL_E = 1.0e-8

_ROOT_STATE_CONTRACT = "canonical-adt-hybrid-ddx-root-state-v1"
_START_CONTRACT = "canonical-adt-hybrid-ddx-root-start-v1"
_SCREEN_STATE_CONTRACT = "canonical-adt-hybrid-ddx-zero-start-screen-state-v1"


def _array_sha(values: object) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(
        f"{array.dtype.str}:{array.shape}".encode() + b"\0" + array.tobytes(order="C")
    ).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA256 digest.") from exc
    return value.lower()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


@runtime_checkable
class PermanentSourceProvider(Protocol):
    provider_id: str
    model_profile_id: str
    source_space: object

    def configuration_sha256(self) -> str: ...

    def evaluate_source(self, geometry: object) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class CanonicalADTRootStart:
    """One deterministic root replay outcome."""

    label: str
    initial_field_sha256: str
    converged_field_sha256: str
    continuum_state_sha256: str
    iterations: int
    residual_norm_ev: float
    polarization_energy_ev: float
    record_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("root start label must be non-empty.")
        for name in (
            "initial_field_sha256",
            "converged_field_sha256",
            "continuum_state_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if (
            type(self.iterations) is not int
            or not 1 <= self.iterations <= MAX_ROOT_ITERATIONS
        ):
            raise ValueError("root iterations are outside the configured bound.")
        residual = float(self.residual_norm_ev)
        energy = float(self.polarization_energy_ev)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("root residual must be finite and non-negative.")
        if not math.isfinite(energy):
            raise ValueError("root polarization energy must be finite.")
        expected = canonical_metadata_sha256(
            {
                "contract": _START_CONTRACT,
                "label": self.label,
                "initial_field_sha256": self.initial_field_sha256,
                "converged_field_sha256": self.converged_field_sha256,
                "continuum_state_sha256": self.continuum_state_sha256,
                "iterations": self.iterations,
                "residual_norm_ev": residual,
                "polarization_energy_ev": energy,
            }
        )
        if self.record_sha256 and self.record_sha256 != expected:
            raise ValueError("record_sha256 does not match the root-start record.")
        object.__setattr__(self, "residual_norm_ev", residual)
        object.__setattr__(self, "polarization_energy_ev", energy)
        object.__setattr__(self, "record_sha256", expected)


@dataclass(frozen=True, slots=True)
class CanonicalADTHybridDDXState:
    """Five-start self-consistent state of the operational scalar."""

    geometry_sha256: str
    evaluator_configuration_sha256: str
    permanent_source4: np.ndarray
    native_field8: np.ndarray
    radial_residual_source4: np.ndarray
    adt_atomic_dipoles_eangstrom: np.ndarray
    continuum_state_sha256: str
    vacuum_energy_ev: float
    polarization_energy_ev: float
    primal_residual_ev: float
    root_starts: tuple[CanonicalADTRootStart, ...]
    root_sha256: str = ""

    def __post_init__(self) -> None:
        geometry = _digest(self.geometry_sha256, name="geometry_sha256")
        configuration = _digest(
            self.evaluator_configuration_sha256,
            name="evaluator_configuration_sha256",
        )
        continuum = _digest(self.continuum_state_sha256, name="continuum_state_sha256")
        permanent = np.asarray(self.permanent_source4, dtype=np.float64)
        if permanent.ndim != 2 or permanent.shape[1] != 4:
            raise ValueError("permanent_source4 must have shape (N,4).")
        count = len(permanent)
        permanent = _readonly(permanent, shape=(count, 4), name="permanent source")
        field = _readonly(self.native_field8, shape=(count, 8), name="native field")
        radial = _readonly(
            self.radial_residual_source4,
            shape=(count, 4),
            name="radial residual source",
        )
        dipoles = _readonly(
            self.adt_atomic_dipoles_eangstrom,
            shape=(count, 3),
            name="ADT atomic dipoles",
        )
        vacuum = float(self.vacuum_energy_ev)
        polarization = float(self.polarization_energy_ev)
        residual = float(self.primal_residual_ev)
        if not all(math.isfinite(value) for value in (vacuum, polarization, residual)):
            raise ValueError("root energies and residual must be finite.")
        if residual < 0.0 or residual >= ROOT_TOLERANCE_EV:
            raise ValueError("root residual does not satisfy the frozen tolerance.")
        starts = tuple(self.root_starts)
        if len(starts) != 5 or any(
            not isinstance(item, CanonicalADTRootStart) for item in starts
        ):
            raise ValueError("root_starts must contain exactly five records.")
        if len({item.label for item in starts}) != 5:
            raise ValueError("root start labels must be unique.")
        expected = canonical_metadata_sha256(
            {
                "contract": _ROOT_STATE_CONTRACT,
                "root_contract": CANONICAL_ADT_HYBRID_DDX_ROOT_CONTRACT,
                "geometry_sha256": geometry,
                "evaluator_configuration_sha256": configuration,
                "permanent_source_sha256": _array_sha(permanent),
                "native_field_sha256": _array_sha(field),
                "radial_residual_source_sha256": _array_sha(radial),
                "adt_atomic_dipoles_sha256": _array_sha(dipoles),
                "continuum_state_sha256": continuum,
                "vacuum_energy_ev": vacuum,
                "polarization_energy_ev": polarization,
                "primal_residual_ev": residual,
                "root_start_sha256": [item.record_sha256 for item in starts],
            }
        )
        if self.root_sha256 and self.root_sha256 != expected:
            raise ValueError("root_sha256 does not match the canonical ADT state.")
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "evaluator_configuration_sha256", configuration)
        object.__setattr__(self, "permanent_source4", permanent)
        object.__setattr__(self, "native_field8", field)
        object.__setattr__(self, "radial_residual_source4", radial)
        object.__setattr__(self, "adt_atomic_dipoles_eangstrom", dipoles)
        object.__setattr__(self, "continuum_state_sha256", continuum)
        object.__setattr__(self, "vacuum_energy_ev", vacuum)
        object.__setattr__(self, "polarization_energy_ev", polarization)
        object.__setattr__(self, "primal_residual_ev", residual)
        object.__setattr__(self, "root_starts", starts)
        object.__setattr__(self, "root_sha256", expected)

    @property
    def total_energy_ev(self) -> float:
        return self.vacuum_energy_ev + self.polarization_energy_ev


@dataclass(frozen=True, slots=True)
class CanonicalADTZeroStartScreenState:
    """One-start state for fast development screening, never admission.

    The five-start :meth:`MACE_MDPPolarCanonicalADTDDXEnergy.solve` remains the
    authoritative root replay.  This smaller state exists only to reject poor
    chemistry hypotheses before paying five times the checkpoint cost on a
    large development panel.
    """

    geometry_sha256: str
    evaluator_configuration_sha256: str
    permanent_source4: np.ndarray
    native_field8: np.ndarray
    radial_residual_source4: np.ndarray
    adt_atomic_dipoles_eangstrom: np.ndarray
    continuum_state_sha256: str
    vacuum_energy_ev: float
    polarization_energy_ev: float
    primal_residual_ev: float
    iterations: int
    screen_sha256: str = ""

    def __post_init__(self) -> None:
        geometry = _digest(self.geometry_sha256, name="geometry_sha256")
        configuration = _digest(
            self.evaluator_configuration_sha256,
            name="evaluator_configuration_sha256",
        )
        continuum = _digest(self.continuum_state_sha256, name="continuum_state_sha256")
        permanent = np.asarray(self.permanent_source4, dtype=np.float64)
        if permanent.ndim != 2 or permanent.shape[1] != 4:
            raise ValueError("permanent_source4 must have shape (N,4).")
        count = len(permanent)
        permanent = _readonly(permanent, shape=(count, 4), name="permanent source")
        field = _readonly(self.native_field8, shape=(count, 8), name="native field")
        radial = _readonly(
            self.radial_residual_source4,
            shape=(count, 4),
            name="radial residual source",
        )
        dipoles = _readonly(
            self.adt_atomic_dipoles_eangstrom,
            shape=(count, 3),
            name="ADT atomic dipoles",
        )
        vacuum = float(self.vacuum_energy_ev)
        polarization = float(self.polarization_energy_ev)
        residual = float(self.primal_residual_ev)
        if not all(math.isfinite(value) for value in (vacuum, polarization, residual)):
            raise ValueError("screen energies and residual must be finite.")
        if residual < 0.0 or residual >= ROOT_TOLERANCE_EV:
            raise ValueError("screen residual does not satisfy the frozen tolerance.")
        if (
            type(self.iterations) is not int
            or not 1 <= self.iterations <= MAX_ROOT_ITERATIONS
        ):
            raise ValueError("screen iterations are outside the configured bound.")
        expected = canonical_metadata_sha256(
            {
                "contract": _SCREEN_STATE_CONTRACT,
                "root_contract": CANONICAL_ADT_HYBRID_DDX_ROOT_CONTRACT,
                "geometry_sha256": geometry,
                "evaluator_configuration_sha256": configuration,
                "permanent_source_sha256": _array_sha(permanent),
                "native_field_sha256": _array_sha(field),
                "radial_residual_source_sha256": _array_sha(radial),
                "adt_atomic_dipoles_sha256": _array_sha(dipoles),
                "continuum_state_sha256": continuum,
                "vacuum_energy_ev": vacuum,
                "polarization_energy_ev": polarization,
                "primal_residual_ev": residual,
                "iterations": self.iterations,
                "claim_boundary": "zero-start-development-screen-not-admission",
            }
        )
        if self.screen_sha256 and self.screen_sha256 != expected:
            raise ValueError("screen_sha256 does not match the screen state.")
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "evaluator_configuration_sha256", configuration)
        object.__setattr__(self, "permanent_source4", permanent)
        object.__setattr__(self, "native_field8", field)
        object.__setattr__(self, "radial_residual_source4", radial)
        object.__setattr__(self, "adt_atomic_dipoles_eangstrom", dipoles)
        object.__setattr__(self, "continuum_state_sha256", continuum)
        object.__setattr__(self, "vacuum_energy_ev", vacuum)
        object.__setattr__(self, "polarization_energy_ev", polarization)
        object.__setattr__(self, "primal_residual_ev", residual)
        object.__setattr__(self, "screen_sha256", expected)

    @property
    def total_energy_ev(self) -> float:
        return self.vacuum_energy_ev + self.polarization_energy_ev


@dataclass(frozen=True, slots=True)
class _SolvedStart:
    initial_field: np.ndarray
    field: np.ndarray
    components: CanonicalADTInducedComponents
    continuum_state: CanonicalADTSeparatedDDXState
    residual_norm: float
    iterations: int


class MACE_MDPPolarCanonicalADTDDXEnergy:
    """Geometry-bound canonical-ADT hybrid/ddX operational scalar."""

    __slots__ = (
        "_configuration_sha256",
        "_continuum",
        "_direct_sum",
        "_geometry_sha256",
        "_permanent",
        "_permanent_source4",
        "_profile_id",
        "_provider_id",
        "_response",
        "_sealed",
    )

    capabilities = CapabilityStatus()
    coordinate_derivative_available = False
    force_available = False
    variational_functional_admitted = False

    def __init__(
        self,
        geometry: object,
        *,
        permanent: PermanentSourceProvider,
        response: MDPPolarCanonicalADTResponse,
        continuum: SeparatedSourceDDXBackend,
    ) -> None:
        if not isinstance(response, MDPPolarCanonicalADTResponse):
            raise TypeError("response must be MDPPolarCanonicalADTResponse.")
        polar_zero_permanent = isinstance(
            permanent, MACEPolarZeroFieldPointPermanentSource
        )
        if response.contract_id == MDP_POLAR_CANONICAL_ADT_CONTRACT:
            if polar_zero_permanent:
                raise ValueError(
                    "POLAR zero-field permanent source requires the current "
                    "role-separated ADT response contract."
                )
            provider_id = CANONICAL_ADT_HYBRID_DDX_PROVIDER_ID
            profile_id = CANONICAL_ADT_HYBRID_DDX_PROFILE_ID
        elif response.contract_id == MDP_POLAR_ROLE_SEPARATED_ADT_CONTRACT:
            if polar_zero_permanent:
                provider_id = POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID
                profile_id = POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
            else:
                provider_id = ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID
                profile_id = ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID
        else:
            raise ValueError("response exposes an unsupported ADT contract identity.")
        if not isinstance(continuum, SeparatedSourceDDXBackend):
            raise TypeError("continuum must be SeparatedSourceDDXBackend.")
        for name in ("configuration_sha256", "evaluate_source"):
            if not callable(getattr(permanent, name, None)):
                raise TypeError(f"permanent source provider requires {name}().")
        if getattr(permanent, "source_space", None) != ATOMIC_L1_SOURCE_SPACE:
            raise ValueError(
                "permanent provider must use the canonical atomic-l1 space."
            )
        if polar_zero_permanent:
            if (
                permanent.response_configuration_sha256
                != response.configuration_sha256()
            ):
                raise ValueError(
                    "POLAR zero-field permanent source and induced response "
                    "must be views of the same frozen response object."
                )
        else:
            permanent_mdp = getattr(permanent, "mdp_configuration_sha256", None)
            response_mdp = response.mdp_configuration_sha256
            if permanent_mdp != response_mdp:
                raise ValueError(
                    "permanent moments and molecular polarizability must come "
                    "from the same frozen MDP checkpoint/runtime."
                )
        count = atom_count(geometry)
        if count != len(continuum.symbols):
            raise ValueError("geometry and separated ddX atom counts differ.")
        permanent.configuration_sha256()
        response.configuration_sha256()
        source = ATOMIC_L1_SOURCE_SPACE.validate(
            permanent.evaluate_source(geometry),
            atom_count=count,
            name="hybrid permanent source",
        )
        if abs(float(np.sum(source[:, 0]))) > TOTAL_CHARGE_ATOL_E:
            raise ValueError(
                "canonical ADT research profile supports neutral sources only."
            )
        prepared = continuum.prepare(geometry, source)
        chart = response.chart_for_geometry(geometry)
        if polar_zero_permanent and not np.array_equal(
            source, chart.polar_zero_source4
        ):
            raise RuntimeError(
                "POLAR zero-field permanent view does not exactly match the "
                "response chart."
            )
        direct_sum = prepare_canonical_adt_separated_ddx(prepared, chart.adt_lift)
        object.__setattr__(self, "_provider_id", provider_id)
        object.__setattr__(self, "_profile_id", profile_id)
        configuration = canonical_metadata_sha256(
            {
                "provider_id": self.provider_id,
                "profile_id": self.profile_id,
                "root_contract": CANONICAL_ADT_HYBRID_DDX_ROOT_CONTRACT,
                "geometry_sha256": geometry_sha256(geometry),
                "permanent_provider_id": getattr(permanent, "provider_id", None),
                "permanent_model_profile_id": getattr(
                    permanent, "model_profile_id", None
                ),
                "permanent_configuration_sha256": permanent.configuration_sha256(),
                "response_provider_id": response.provider_id,
                "response_model_profile_id": response.model_profile_id,
                "response_configuration_sha256": response.configuration_sha256(),
                "continuum_provider_id": continuum.provider_id,
                "continuum_configuration_sha256": continuum.configuration_sha256(),
                "direct_sum_configuration_sha256": direct_sum.configuration_sha256(),
                "permanent_source_sha256": _array_sha(source),
                "permanent_source_role": (
                    "macepolar-zero-field-point-monopoles-dipoles"
                    if polar_zero_permanent
                    else "mace-mdp-latent-point-monopoles-dipoles"
                ),
                "root_tolerance_ev": ROOT_TOLERANCE_EV,
                "root_replay_field_atol_ev": ROOT_REPLAY_FIELD_ATOL_EV,
                "root_replay_energy_atol_ev": ROOT_REPLAY_ENERGY_ATOL_EV,
                "maximum_root_iterations": MAX_ROOT_ITERATIONS,
                "anderson_history": ANDERSON_HISTORY,
                "anderson_damping": ANDERSON_DAMPING,
                "start_policy": "zero,permanent,twice,negative,seeded-transverse-v1",
                "ledger": "polar-zero-field-vacuum-plus-ddx-polarization",
                "model_drive": "radial-external-mep-receiver",
                "coordinate_derivative_available": False,
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "_geometry_sha256", geometry_sha256(geometry))
        object.__setattr__(self, "_permanent", permanent)
        object.__setattr__(self, "_response", response)
        object.__setattr__(self, "_continuum", continuum)
        object.__setattr__(self, "_permanent_source4", source.copy())
        object.__setattr__(self, "_direct_sum", direct_sum)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACE_MDPPolarCanonicalADTDDXEnergy is immutable.")
        object.__setattr__(self, name, value)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def profile_id(self) -> str:
        return self._profile_id

    @property
    def permanent_source4(self) -> np.ndarray:
        return _readonly(
            self._permanent_source4,
            shape=(self._direct_sum.atom_count, 4),
            name="permanent source",
        )

    @property
    def direct_sum(self) -> PreparedCanonicalADTSeparatedDDX:
        return self._direct_sum

    def configuration_sha256(self) -> str:
        self._permanent.configuration_sha256()
        self._response.configuration_sha256()
        self._continuum.configuration_sha256()
        self._direct_sum.configuration_sha256()
        return self._configuration_sha256

    def _validate_geometry(self, geometry: object) -> int:
        if geometry_sha256(geometry) != self._geometry_sha256:
            raise ValueError("evaluator is bound to a different geometry.")
        return atom_count(geometry)

    def state_map(
        self, geometry: object, field: object
    ) -> tuple[
        np.ndarray, CanonicalADTInducedComponents, CanonicalADTSeparatedDDXState
    ]:
        count = self._validate_geometry(geometry)
        values = self._response.receiver_space.validate(
            field, atom_count=count, name="native field"
        )
        components = self._response.evaluate_components(geometry, values)
        continuum_state = self._direct_sum.solve(
            components.radial_residual_source4,
            components.adt_atomic_dipoles_eangstrom,
        )
        target = self._response.receiver_space.validate(
            continuum_state.model_field8,
            atom_count=count,
            name="ddX native model field",
        )
        return target, components, continuum_state

    def state_map_jvp(
        self,
        geometry: object,
        field: object,
        field_direction: object,
    ) -> np.ndarray:
        count = self._validate_geometry(geometry)
        values = self._response.receiver_space.validate(field, atom_count=count)
        direction = self._response.receiver_space.validate(
            field_direction, atom_count=count, name="native field direction"
        )
        components = self._response.field_jvp_components(geometry, values, direction)
        return self._response.receiver_space.validate(
            self._direct_sum.model_field_jvp(
                components.radial_residual_source4,
                components.adt_atomic_dipoles_eangstrom,
            ),
            atom_count=count,
            name="state-map JVP",
        )

    def state_map_vjp(
        self,
        geometry: object,
        field: object,
        field_cotangent: object,
    ) -> np.ndarray:
        count = self._validate_geometry(geometry)
        values = self._response.receiver_space.validate(field, atom_count=count)
        cotangent = self._response.receiver_space.validate(
            field_cotangent, atom_count=count, name="state-map cotangent"
        )
        branches = self._direct_sum.model_field_vjp(cotangent)
        return self._response.receiver_space.validate(
            self._response.field_vjp_components(
                geometry,
                values,
                radial_source_cotangent=branches.radial_residual_source4,
                adt_atomic_dipole_cotangent=(branches.adt_atomic_dipoles_eangstrom),
            ),
            atom_count=count,
            name="state-map VJP",
        )

    @staticmethod
    def _anderson_step(
        x_history: list[np.ndarray],
        f_history: list[np.ndarray],
        mapped: np.ndarray,
    ) -> np.ndarray:
        depth = min(ANDERSON_HISTORY, len(f_history) - 1)
        if depth == 0:
            return x_history[-1] + ANDERSON_DAMPING * f_history[-1]
        start = len(f_history) - depth - 1
        delta_x = np.column_stack(
            [
                x_history[index + 1] - x_history[index]
                for index in range(start, len(x_history) - 1)
            ]
        )
        delta_f = np.column_stack(
            [
                f_history[index + 1] - f_history[index]
                for index in range(start, len(f_history) - 1)
            ]
        )
        gamma = np.linalg.lstsq(delta_f, f_history[-1], rcond=None)[0]
        accelerated = mapped - (delta_x + delta_f) @ gamma
        return x_history[-1] + ANDERSON_DAMPING * (accelerated - x_history[-1])

    def _solve_from(self, geometry: object, initial_field: object) -> _SolvedStart:
        count = self._validate_geometry(geometry)
        initial = self._response.receiver_space.validate(
            initial_field, atom_count=count, name="initial native field"
        )
        field = np.array(initial, dtype=np.float64, copy=True)
        x_history: list[np.ndarray] = []
        f_history: list[np.ndarray] = []
        for iteration in range(1, MAX_ROOT_ITERATIONS + 1):
            target, components, continuum_state = self.state_map(geometry, field)
            residual = target - field
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm < ROOT_TOLERANCE_EV:
                return _SolvedStart(
                    initial_field=_readonly(
                        initial,
                        shape=(count, 8),
                        name="initial native field",
                    ),
                    field=_readonly(field, shape=(count, 8), name="root field"),
                    components=components,
                    continuum_state=continuum_state,
                    residual_norm=residual_norm,
                    iterations=iteration,
                )
            x_history.append(field.reshape(-1).copy())
            f_history.append(residual.reshape(-1).copy())
            candidate = self._anderson_step(
                x_history,
                f_history,
                target.reshape(-1),
            )
            if not np.all(np.isfinite(candidate)):
                raise RuntimeError("canonical ADT Anderson step is non-finite.")
            field = candidate.reshape(count, 8)
        raise RuntimeError(
            f"canonical ADT root did not converge in {MAX_ROOT_ITERATIONS} iterations."
        )

    def _initial_fields(self, geometry: object) -> tuple[tuple[str, np.ndarray], ...]:
        count = self._validate_geometry(geometry)
        zero = np.zeros((count, 8), dtype=np.float64)
        permanent = self._direct_sum.solve(
            np.zeros((count, 4), dtype=np.float64),
            np.zeros((count, 3), dtype=np.float64),
        ).model_field8
        seed = int(self._configuration_sha256[:16], 16) % (2**32)
        random = np.random.default_rng(seed).normal(size=(count, 8))
        random /= max(float(np.linalg.norm(random)), np.finfo(float).tiny)
        scale = max(float(np.linalg.norm(permanent)), 1.0e-3)
        transverse = permanent + scale * random
        return (
            ("zero", zero),
            ("permanent", permanent),
            ("twice-permanent", 2.0 * permanent),
            ("negative-permanent", -permanent),
            ("seeded-transverse", transverse),
        )

    def solve(self, geometry: object) -> CanonicalADTHybridDDXState:
        self.configuration_sha256()
        solved = tuple(
            (label, self._solve_from(geometry, initial))
            for label, initial in self._initial_fields(geometry)
        )
        reference = solved[0][1]
        records: list[CanonicalADTRootStart] = []
        for label, current in solved:
            field_difference = float(np.max(np.abs(current.field - reference.field)))
            energy_difference = abs(
                current.continuum_state.polarization_energy_ev
                - reference.continuum_state.polarization_energy_ev
            )
            if field_difference > ROOT_REPLAY_FIELD_ATOL_EV:
                raise RuntimeError("canonical ADT starts found different roots.")
            if energy_difference > ROOT_REPLAY_ENERGY_ATOL_EV:
                raise RuntimeError("canonical ADT starts disagree in energy.")
            records.append(
                CanonicalADTRootStart(
                    label=label,
                    initial_field_sha256=_array_sha(current.initial_field),
                    converged_field_sha256=_array_sha(current.field),
                    continuum_state_sha256=current.continuum_state.state_sha256,
                    iterations=current.iterations,
                    residual_norm_ev=current.residual_norm,
                    polarization_energy_ev=(
                        current.continuum_state.polarization_energy_ev
                    ),
                )
            )
        return CanonicalADTHybridDDXState(
            geometry_sha256=self._geometry_sha256,
            evaluator_configuration_sha256=self._configuration_sha256,
            permanent_source4=self._permanent_source4,
            native_field8=reference.field,
            radial_residual_source4=(reference.components.radial_residual_source4),
            adt_atomic_dipoles_eangstrom=(
                reference.components.adt_atomic_dipoles_eangstrom
            ),
            continuum_state_sha256=reference.continuum_state.state_sha256,
            vacuum_energy_ev=self._response.vacuum_energy_ev(geometry),
            polarization_energy_ev=(reference.continuum_state.polarization_energy_ev),
            primal_residual_ev=reference.residual_norm,
            root_starts=tuple(records),
        )

    def solve_zero_start_screen(
        self, geometry: object
    ) -> CanonicalADTZeroStartScreenState:
        """Solve one zero start for a fast, explicitly non-admitting screen."""

        self.configuration_sha256()
        count = self._validate_geometry(geometry)
        solved = self._solve_from(geometry, np.zeros((count, 8), dtype=np.float64))
        return CanonicalADTZeroStartScreenState(
            geometry_sha256=self._geometry_sha256,
            evaluator_configuration_sha256=self._configuration_sha256,
            permanent_source4=self._permanent_source4,
            native_field8=solved.field,
            radial_residual_source4=solved.components.radial_residual_source4,
            adt_atomic_dipoles_eangstrom=(
                solved.components.adt_atomic_dipoles_eangstrom
            ),
            continuum_state_sha256=solved.continuum_state.state_sha256,
            vacuum_energy_ev=self._response.vacuum_energy_ev(geometry),
            polarization_energy_ev=solved.continuum_state.polarization_energy_ev,
            primal_residual_ev=solved.residual_norm,
            iterations=solved.iterations,
        )

    def local_root_certificate(
        self,
        geometry: object,
        state: CanonicalADTHybridDDXState | None = None,
    ) -> RootWellPosednessCertificate:
        central = self.solve(geometry) if state is None else state
        if not isinstance(central, CanonicalADTHybridDDXState):
            raise TypeError("state must be CanonicalADTHybridDDXState.")
        if (
            central.geometry_sha256 != self._geometry_sha256
            or central.evaluator_configuration_sha256 != self.configuration_sha256()
        ):
            raise ValueError("root state belongs to another evaluator.")
        shape = central.native_field8.shape
        dimension = int(np.prod(shape))
        jacobian = dense_state_map_jacobian(
            lambda direction: self.state_map_jvp(
                geometry,
                central.native_field8,
                np.asarray(direction, dtype=np.float64).reshape(shape),
            ).reshape(-1),
            dimension=dimension,
            maximum_dimension=MAX_DENSE_AUDIT_DIMENSION,
        )
        return certify_root_well_posedness(jacobian)

    def molecular_response_passivity_eigenvalues(self, geometry: object) -> np.ndarray:
        """Return eigenvalues of the supervised canonical molecular alpha.

        Positive values certify passivity only for the three-dimensional
        uniform ADT response selected by the construction.  They do not prove
        that the full nonlinear coupled state map is globally contractive.
        """

        self._validate_geometry(geometry)
        alpha = self._response.chart_for_geometry(
            geometry
        ).mdp_molecular_polarizability_eangstrom2_per_volt
        result = np.linalg.eigvalsh(0.5 * (alpha + alpha.T))
        if result.shape != (3,) or not np.all(np.isfinite(result)) or result[0] <= 0.0:
            raise RuntimeError("canonical molecular polarizability is not passive.")
        result.setflags(write=False)
        return result

    def coordinate_vjp(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise NotImplementedError(
            "The canonical ADT hybrid/ddX complete coordinate VJP is not implemented."
        )


__all__ = [
    "ANDERSON_DAMPING",
    "ANDERSON_HISTORY",
    "CANONICAL_ADT_HYBRID_DDX_PROFILE_ID",
    "CANONICAL_ADT_HYBRID_DDX_PROVIDER_ID",
    "CANONICAL_ADT_HYBRID_DDX_ROOT_CONTRACT",
    "CanonicalADTHybridDDXState",
    "CanonicalADTZeroStartScreenState",
    "CanonicalADTRootStart",
    "MACE_MDPPolarCanonicalADTDDXEnergy",
    "MAX_ROOT_ITERATIONS",
    "ROOT_TOLERANCE_EV",
    "ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID",
    "ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID",
    "POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROFILE_ID",
    "POLAR_ZERO_ROLE_SEPARATED_ADT_HYBRID_DDX_PROVIDER_ID",
]
