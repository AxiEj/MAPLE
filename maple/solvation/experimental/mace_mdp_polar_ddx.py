"""MACE-MDP permanent moments + MACE-POLAR induced response with ddX.

This experimental energy path keeps the chemically successful hybrid source
semantics while replacing the GEPOL surface solver:

* MACE-MDP permanent ``(q,p)`` values enter ddX as point multipoles;
* only the MACE-POLAR induced increment enters through the 1.5-A Gaussian
  source channel;
* the response model receives the full 1.5/3.0-A external-MEP boundary field;
  this model drive is deliberately distinct from the operational ledger's
  source gradient because Gaussian tails violate ddX's compact-support
  assumption.

The selected operational ledger remains ``E_vac + E_ddX,pol``.  It is not a
claim that the original MACE-POLAR source is the gradient of its raw
field-conditioned energy.  The developer surface exposes E, the analytic
block-adjoint F, a molecular virial, and Richardson HVP/H derived from that
same force.  Registry and workflow admission remain closed until matched
chemistry, covariance, distorted-PES, derivative-accuracy, and root-regularity
evidence are all bound to this exact profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from maple.solvation.continuum.separated_source_ddx import (
    SeparatedSourceDDXBackend,
    embed_atomic_l1_in_first_radial_channel,
    extract_atomic_l1_first_radial_cotangent,
)
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives import (
    RichardsonScalarForce,
    RichardsonScalarForceComponentEvaluation,
    RichardsonScalarForceEvaluation,
    RichardsonScalarHessian,
    RichardsonScalarHessianEvaluation,
    RichardsonScalarHVPEvaluation,
    ScalarEnergySample,
    ScalarForceSample,
)
from maple.solvation.experimental.mace_polar_frozen_ddx import (
    MolecularVirialEvaluation,
)
from maple.solvation.models.base import atom_count
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
    PermanentInducedSourceAnchor,
)

HYBRID_DDX_SCALAR_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-separated-ddx-scalar.impl.v2"
)
HYBRID_DDX_PES_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-separated-ddx-pes.impl.v1"
)
HYBRID_DDX_ROOT_CONTRACT_ID = "mace-mdp-polar-separated-ddx-two-start-root-v2"
ROOT_TOLERANCE_EV = 1.0e-10
ROOT_REPLAY_FIELD_ATOL_EV = 2.0e-9
ROOT_REPLAY_ENERGY_ATOL_EV = 1.0e-10
TOTAL_CHARGE_ATOL_E = 1.0e-8
MAX_ROOT_ITERATIONS = 40
REQUIRED_LONG_RANGE_EVALUATOR = (
    "graph-longrange-analytic-gaussian-multipole-realspace-v1"
)
NUMERICAL_FORCE_COARSE_STEP_ANGSTROM = 5.0e-4
NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM = 2.0e-4
ADJOINT_TOLERANCE_EV = 1.0e-10
MAX_ADJOINT_ITERATIONS = 200


def _array_sha256(values: object, *, name: str) -> str:
    array = np.asarray(values)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite array.")
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode()
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array, dtype=float)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class HybridDDXEnergyState:
    """Immutable two-start self-consistent state for the operational scalar."""

    geometry_sha256: str
    evaluator_configuration_sha256: str
    anchor_state_sha256: str
    continuum_state_sha256: str
    native_field_ev: np.ndarray
    induced_source4: np.ndarray
    total_source4: np.ndarray
    vacuum_energy_ev: float
    polarization_energy_ev: float
    primal_residual_ev: float
    cold_iterations: int
    wide_iterations: int
    replay_field_max_abs_difference_ev: float
    replay_energy_abs_difference_ev: float
    root_sha256: str = ""

    def __post_init__(self) -> None:
        field = np.asarray(self.native_field_ev, dtype=float)
        induced = np.asarray(self.induced_source4, dtype=float)
        total = np.asarray(self.total_source4, dtype=float)
        if field.ndim != 2 or field.shape[1] != 8:
            raise ValueError("native_field_ev must have shape (N,8).")
        count = len(field)
        field = _readonly(field, shape=(count, 8), name="native field")
        induced = _readonly(induced, shape=(count, 4), name="induced source")
        total = _readonly(total, shape=(count, 4), name="total source")
        scalars: dict[str, float] = {}
        for name in (
            "vacuum_energy_ev",
            "polarization_energy_ev",
            "primal_residual_ev",
            "replay_field_max_abs_difference_ev",
            "replay_energy_abs_difference_ev",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if "residual" in name or "difference" in name:
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative.")
            scalars[name] = value
        for name in ("cold_iterations", "wide_iterations"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_ROOT_ITERATIONS:
                raise ValueError(f"{name} is outside the root iteration bound.")
        payload = {
            "contract": HYBRID_DDX_ROOT_CONTRACT_ID,
            "geometry_sha256": self.geometry_sha256,
            "evaluator_configuration_sha256": self.evaluator_configuration_sha256,
            "anchor_state_sha256": self.anchor_state_sha256,
            "continuum_state_sha256": self.continuum_state_sha256,
            "native_field_sha256": _array_sha256(field, name="native field"),
            "induced_source_sha256": _array_sha256(induced, name="induced source"),
            "total_source_sha256": _array_sha256(total, name="total source"),
            **scalars,
            "cold_iterations": self.cold_iterations,
            "wide_iterations": self.wide_iterations,
        }
        expected = canonical_metadata_sha256(payload)
        if self.root_sha256 and self.root_sha256 != expected:
            raise ValueError("root_sha256 does not match the state contents.")
        object.__setattr__(self, "native_field_ev", field)
        object.__setattr__(self, "induced_source4", induced)
        object.__setattr__(self, "total_source4", total)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "root_sha256", expected)

    @property
    def total_energy_ev(self) -> float:
        return self.vacuum_energy_ev + self.polarization_energy_ev


@dataclass(frozen=True, slots=True)
class HybridDDXForceEvaluation:
    """Complete block-adjoint force leaves for the operational scalar."""

    central_state: HybridDDXEnergyState
    vacuum_forces_ev_per_angstrom: np.ndarray
    continuum_fixed_source_forces_ev_per_angstrom: np.ndarray
    permanent_source_forces_ev_per_angstrom: np.ndarray
    induced_source_forces_ev_per_angstrom: np.ndarray
    model_field_geometry_forces_ev_per_angstrom: np.ndarray
    adjoint_field_ev: np.ndarray
    adjoint_residual_ev: float
    adjoint_iterations: int
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.central_state, HybridDDXEnergyState):
            raise TypeError("central_state must be HybridDDXEnergyState.")
        count = len(self.central_state.native_field_ev)
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "vacuum_forces_ev_per_angstrom",
            "continuum_fixed_source_forces_ev_per_angstrom",
            "permanent_source_forces_ev_per_angstrom",
            "induced_source_forces_ev_per_angstrom",
            "model_field_geometry_forces_ev_per_angstrom",
        ):
            arrays[name] = _readonly(getattr(self, name), shape=(count, 3), name=name)
        adjoint = _readonly(
            self.adjoint_field_ev,
            shape=self.central_state.native_field_ev.shape,
            name="adjoint_field_ev",
        )
        residual = float(self.adjoint_residual_ev)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("adjoint_residual_ev must be finite and non-negative.")
        if (
            type(self.adjoint_iterations) is not int
            or not 0 <= self.adjoint_iterations <= MAX_ADJOINT_ITERATIONS
        ):
            raise ValueError("adjoint_iterations is outside the configured bound.")
        payload = {
            "contract": "mace-mdp-polar-separated-ddx-block-adjoint-force-v1",
            "central_root_sha256": self.central_state.root_sha256,
            **{name: _array_sha256(value, name=name) for name, value in arrays.items()},
            "adjoint_field_sha256": _array_sha256(adjoint, name="adjoint field"),
            "adjoint_residual_ev": residual,
            "adjoint_iterations": self.adjoint_iterations,
        }
        expected = canonical_metadata_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("force evaluation_sha256 does not match its contents.")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "adjoint_field_ev", adjoint)
        object.__setattr__(self, "adjoint_residual_ev", residual)
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def total_forces_ev_per_angstrom(self) -> np.ndarray:
        values = np.add.reduce(
            (
                self.vacuum_forces_ev_per_angstrom,
                self.continuum_fixed_source_forces_ev_per_angstrom,
                self.permanent_source_forces_ev_per_angstrom,
                self.induced_source_forces_ev_per_angstrom,
                self.model_field_geometry_forces_ev_per_angstrom,
            )
        )
        values.setflags(write=False)
        return values


class MACE_MDPPolarHybridDDXEnergy:
    """One geometry-bound operational hybrid/ddX scalar evaluator."""

    __slots__ = (
        "_anchor",
        "_configuration_sha256",
        "_continuum",
        "_geometry_sha256",
        "_hybrid",
        "_prepared",
        "_sealed",
    )

    provider_id = HYBRID_DDX_SCALAR_PROVIDER_ID
    force_derivative_kind = "analytic-block-implicit-adjoint-v1"

    def __init__(
        self,
        geometry: object,
        *,
        hybrid: PermanentAnchoredInducedSourceModel,
        continuum: SeparatedSourceDDXBackend,
    ) -> None:
        if not isinstance(hybrid, PermanentAnchoredInducedSourceModel):
            raise TypeError("hybrid must be PermanentAnchoredInducedSourceModel.")
        if not isinstance(continuum, SeparatedSourceDDXBackend):
            raise TypeError("continuum must be SeparatedSourceDDXBackend.")
        if hybrid.long_range_evaluator_profile != REQUIRED_LONG_RANGE_EVALUATOR:
            raise ValueError(
                "hybrid ddX energy requires the analytic Gaussian-multipole "
                "MACE-POLAR evaluator."
            )
        count = atom_count(geometry)
        if count != len(continuum.symbols):
            raise ValueError("geometry and separated ddX atom counts differ.")
        anchor = hybrid.prepare(geometry)
        prepared = continuum.prepare(geometry, anchor.permanent_source4)
        configuration = canonical_metadata_sha256(
            {
                "provider_id": self.provider_id,
                "root_contract_id": HYBRID_DDX_ROOT_CONTRACT_ID,
                "geometry_sha256": geometry_sha256(geometry),
                "hybrid_configuration_sha256": hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": hybrid.provenance_sha256,
                "continuum_configuration_sha256": continuum.configuration_sha256(),
                "continuum_provenance_sha256": continuum.provenance_sha256,
                "permanent_source_kernel": hybrid.permanent_source_kernel,
                "induced_source_kernel": hybrid.induced_source_kernel,
                "radial_embedding": "first-width-atomic-l1-to-eight-channel",
                "root_tolerance_ev": ROOT_TOLERANCE_EV,
                "root_replay_field_atol_ev": ROOT_REPLAY_FIELD_ATOL_EV,
                "root_replay_energy_atol_ev": ROOT_REPLAY_ENERGY_ATOL_EV,
                "maximum_root_iterations": MAX_ROOT_ITERATIONS,
                "ledger": "vacuum-energy-plus-ddx-polarization",
                "model_drive": "external-mep-phi-adjoint-not-ledger-gradient",
                "capabilities": "none",
            }
        )
        object.__setattr__(self, "_geometry_sha256", geometry_sha256(geometry))
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_continuum", continuum)
        object.__setattr__(self, "_anchor", anchor)
        object.__setattr__(self, "_prepared", prepared)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACE_MDPPolarHybridDDXEnergy is immutable.")
        object.__setattr__(self, name, value)

    @property
    def anchor(self) -> PermanentInducedSourceAnchor:
        return self._anchor

    @property
    def cavity_topology_sha256(self) -> str:
        return self._prepared.cavity_topology_sha256

    @property
    def coordinate_derivative_available(self) -> bool:
        return self._hybrid.coordinate_derivative_available

    @property
    def force_available(self) -> bool:
        return self.coordinate_derivative_available

    def configuration_sha256(self) -> str:
        self._hybrid.configuration_sha256()
        self._continuum.configuration_sha256()
        return self._configuration_sha256

    def _validate_geometry(self, geometry: object) -> int:
        if geometry_sha256(geometry) != self._geometry_sha256:
            raise ValueError("evaluator is bound to a different geometry.")
        return atom_count(geometry)

    def _solve_from(
        self, geometry: object, initial_field: np.ndarray
    ) -> dict[str, Any]:
        count = self._validate_geometry(geometry)
        field = self._hybrid.receiver_space.validate(
            initial_field, atom_count=count, name="initial native field"
        )
        for iteration in range(1, MAX_ROOT_ITERATIONS + 1):
            induced = self._hybrid.induced_source(geometry, self._anchor, field)
            continuum_state = self._prepared.solve(
                embed_atomic_l1_in_first_radial_channel(induced)
            )
            target = self._hybrid.receiver_space.validate(
                continuum_state.model_field,
                atom_count=count,
                name="ddX radial receiver field",
            )
            residual = float(np.linalg.norm(target - field))
            field = target
            if residual < ROOT_TOLERANCE_EV:
                break
        else:
            raise RuntimeError(
                f"hybrid ddX root did not converge in {MAX_ROOT_ITERATIONS} iterations."
            )
        induced = self._hybrid.induced_source(geometry, self._anchor, field)
        continuum_state = self._prepared.solve(
            embed_atomic_l1_in_first_radial_channel(induced)
        )
        target = self._hybrid.receiver_space.validate(
            continuum_state.model_field,
            atom_count=count,
            name="ddX radial receiver field",
        )
        return {
            "iterations": iteration,
            # ``field`` is the input at which ``induced`` and
            # ``continuum_state`` were evaluated.  ``target`` is retained only
            # through the reported residual; substituting it here would bind
            # mutually inconsistent root leaves.
            "field": field,
            "induced": induced,
            "continuum_state": continuum_state,
            "residual_ev": float(np.linalg.norm(target - field)),
        }

    def solve(self, geometry: object) -> HybridDDXEnergyState:
        self.configuration_sha256()
        count = self._validate_geometry(geometry)
        zero = np.zeros(self._hybrid.receiver_space.shape(count), dtype=float)
        permanent_field = self._prepared.solve(
            np.zeros((count, 8), dtype=float)
        ).model_field
        cold = self._solve_from(geometry, zero)
        wide = self._solve_from(geometry, 2.0 * permanent_field)
        field_difference = float(np.max(np.abs(cold["field"] - wide["field"])))
        energy_difference = abs(
            cold["continuum_state"].polarization_energy_ev
            - wide["continuum_state"].polarization_energy_ev
        )
        if field_difference > ROOT_REPLAY_FIELD_ATOL_EV:
            raise RuntimeError("hybrid ddX starts found different roots.")
        if energy_difference > ROOT_REPLAY_ENERGY_ATOL_EV:
            raise RuntimeError("hybrid ddX starts disagree in energy.")
        if float(cold["residual_ev"]) >= ROOT_TOLERANCE_EV:
            raise RuntimeError("hybrid ddX root residual exceeds tolerance.")
        total_source = self._hybrid.total_source(geometry, self._anchor, cold["field"])
        if (
            abs(float(np.sum(total_source[:, 0])) - self._anchor.total_charge_e)
            > TOTAL_CHARGE_ATOL_E
        ):
            raise RuntimeError("hybrid ddX root violates fixed total charge.")
        continuum_state = cold["continuum_state"]
        return HybridDDXEnergyState(
            geometry_sha256=self._geometry_sha256,
            evaluator_configuration_sha256=self.configuration_sha256(),
            anchor_state_sha256=self._anchor.state_sha256,
            continuum_state_sha256=continuum_state.state_sha256,
            native_field_ev=cold["field"],
            induced_source4=cold["induced"],
            total_source4=total_source,
            vacuum_energy_ev=self._hybrid.vacuum_energy_ev(geometry),
            polarization_energy_ev=continuum_state.polarization_energy_ev,
            primal_residual_ev=float(cold["residual_ev"]),
            cold_iterations=int(cold["iterations"]),
            wide_iterations=int(wide["iterations"]),
            replay_field_max_abs_difference_ev=field_difference,
            replay_energy_abs_difference_ev=energy_difference,
        )

    def _validate_central_state(
        self, geometry: object, state: HybridDDXEnergyState
    ) -> None:
        if not isinstance(state, HybridDDXEnergyState):
            raise TypeError("central_state must be HybridDDXEnergyState.")
        if state.geometry_sha256 != self._geometry_sha256:
            raise ValueError("central_state belongs to a different geometry.")
        if state.evaluator_configuration_sha256 != self.configuration_sha256():
            raise ValueError("central_state belongs to a different evaluator.")
        if state.anchor_state_sha256 != self._anchor.state_sha256:
            raise ValueError("central_state belongs to a different source anchor.")

    def evaluate_forces(
        self,
        geometry: object,
        *,
        central_state: HybridDDXEnergyState | None = None,
    ) -> HybridDDXForceEvaluation:
        """Differentiate the selected operational ledger by block adjoint."""

        if not self.coordinate_derivative_available:
            raise NotImplementedError(
                "Hybrid response has no complete coordinate derivative; "
                "analytic block-adjoint force is unavailable."
            )
        self.configuration_sha256()
        self._validate_geometry(geometry)
        state = self.solve(geometry) if central_state is None else central_state
        self._validate_central_state(geometry, state)
        radial_source = embed_atomic_l1_in_first_radial_channel(state.induced_source4)
        continuum_state, fixed_gradient = (
            self._prepared.solve_with_fixed_source_energy_derivatives(radial_source)
        )
        if continuum_state.state_sha256 != state.continuum_state_sha256:
            raise RuntimeError("ddX state did not replay for analytic force.")

        energy_source_cotangent = extract_atomic_l1_first_radial_cotangent(
            continuum_state.energy_source_gradient
        )
        rhs = self._hybrid.field_vjp(
            geometry,
            self._anchor,
            state.native_field_ev,
            energy_source_cotangent,
        )
        dimension = rhs.size
        iterations = 0

        def transpose_residual_action(flat_values: np.ndarray) -> np.ndarray:
            field_cotangent = np.asarray(flat_values, dtype=float).reshape(
                state.native_field_ev.shape
            )
            radial_cotangent = self._prepared.radial_vjp(field_cotangent)
            source_cotangent = extract_atomic_l1_first_radial_cotangent(
                radial_cotangent
            )
            response = self._hybrid.field_vjp(
                geometry,
                self._anchor,
                state.native_field_ev,
                source_cotangent,
            )
            return (field_cotangent - response).reshape(-1)

        operator = LinearOperator(
            (dimension, dimension),
            matvec=transpose_residual_action,
            dtype=np.float64,
        )

        def count_iteration(_residual: object) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = gmres(
            operator,
            rhs.reshape(-1),
            atol=ADJOINT_TOLERANCE_EV,
            rtol=0.0,
            restart=min(50, dimension),
            maxiter=MAX_ADJOINT_ITERATIONS,
            callback=count_iteration,
            callback_type="pr_norm",
        )
        adjoint = np.asarray(solution, dtype=float).reshape(state.native_field_ev.shape)
        true_residual = float(
            np.linalg.norm(
                transpose_residual_action(adjoint.reshape(-1)) - rhs.reshape(-1)
            )
        )
        if info != 0 or true_residual > 10.0 * ADJOINT_TOLERANCE_EV:
            raise RuntimeError(
                "hybrid ddX adjoint did not satisfy its true residual: "
                f"info={info}, residual={true_residual:.6e} eV."
            )

        implicit_source_cotangent = extract_atomic_l1_first_radial_cotangent(
            self._prepared.radial_vjp(adjoint)
        )
        permanent_cotangent = (
            continuum_state.permanent_energy_gradient + implicit_source_cotangent
        )
        induced_cotangent = energy_source_cotangent + implicit_source_cotangent
        permanent_gradient = self._hybrid.permanent_source_position_vjp(
            geometry, self._anchor, permanent_cotangent
        )
        induced_gradient = self._hybrid.induced_source_position_vjp(
            geometry,
            self._anchor,
            state.native_field_ev,
            induced_cotangent,
        )
        model_field_geometry_gradient = self._prepared.model_field_coordinate_vjp(
            radial_source, adjoint
        )
        vacuum_forces = self._hybrid.vacuum_forces_ev_per_angstrom(geometry)
        return HybridDDXForceEvaluation(
            central_state=state,
            vacuum_forces_ev_per_angstrom=vacuum_forces,
            continuum_fixed_source_forces_ev_per_angstrom=-fixed_gradient,
            permanent_source_forces_ev_per_angstrom=-permanent_gradient,
            induced_source_forces_ev_per_angstrom=-induced_gradient,
            model_field_geometry_forces_ev_per_angstrom=(
                -model_field_geometry_gradient
            ),
            adjoint_field_ev=adjoint,
            adjoint_residual_ev=true_residual,
            adjoint_iterations=iterations,
        )


class MACE_MDPPolarHybridDDXPES:
    """Rebuildable operational scalar with an analytic conservative force.

    The production derivative is the complete block implicit adjoint of the
    separated point-permanent/Gaussian-induced scalar.  Independently
    re-solved Richardson derivatives remain available as an audit oracle.
    Molecular virials and error-estimated Richardson HVP/H reuse that force;
    they do not introduce another energy ledger.  The profile remains
    experimental because finite ddX exposed-node topology is not a global
    smoothness or SO(3) guarantee, and chemistry, derivative-accuracy, root
    regularity, and workflow admission remain open.
    """

    __slots__ = (
        "_cavity_radii_angstrom",
        "_configuration_sha256",
        "_continuum_settings",
        "_force_backend",
        "_hessian_backend",
        "_hybrid",
        "_sealed",
        "_symbols",
    )

    provider_id = HYBRID_DDX_PES_PROVIDER_ID
    force_derivative_kind = "analytic-block-implicit-adjoint-v1"
    numerical_force_audit_kind = "same-scalar-richardson-v1"

    def __init__(
        self,
        *,
        hybrid: PermanentAnchoredInducedSourceModel,
        symbols: object,
        cavity_radii_angstrom: object,
        continuum_model: str,
        dielectric: float,
        lmax: int,
        n_lebedev: int,
        solver_tolerance: float = 1.0e-12,
        eta: float = 0.1,
        n_proc: int = 1,
        force_backend: RichardsonScalarForce | None = None,
        hessian_backend: RichardsonScalarHessian | None = None,
    ) -> None:
        if not isinstance(hybrid, PermanentAnchoredInducedSourceModel):
            raise TypeError("hybrid must be PermanentAnchoredInducedSourceModel.")
        normalized_symbols = tuple(str(value) for value in symbols)
        if not normalized_symbols:
            raise ValueError("symbols must be nonempty.")
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        if (
            radii.shape != (len(normalized_symbols),)
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("cavity radii must be positive with shape (N,).")
        settings = {
            "continuum_model": str(continuum_model),
            "dielectric": float(dielectric),
            "lmax": int(lmax),
            "n_lebedev": int(n_lebedev),
            "solver_tolerance": float(solver_tolerance),
            "eta": float(eta),
            "n_proc": int(n_proc),
        }
        # Construction is also the authoritative validation of the ddX
        # method/settings contract.  The disposable object performs no solve.
        SeparatedSourceDDXBackend(normalized_symbols, radii, **settings)
        backend = force_backend or RichardsonScalarForce(
            coarse_step_angstrom=NUMERICAL_FORCE_COARSE_STEP_ANGSTROM,
            maximum_error_eV_per_A=NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
        )
        if not isinstance(backend, RichardsonScalarForce):
            raise TypeError("force_backend must be RichardsonScalarForce.")
        hessian = hessian_backend or RichardsonScalarHessian()
        if not isinstance(hessian, RichardsonScalarHessian):
            raise TypeError("hessian_backend must be RichardsonScalarHessian.")
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        radii.setflags(write=False)
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_symbols", normalized_symbols)
        object.__setattr__(self, "_cavity_radii_angstrom", radii)
        object.__setattr__(self, "_continuum_settings", tuple(sorted(settings.items())))
        object.__setattr__(self, "_force_backend", backend)
        object.__setattr__(self, "_hessian_backend", hessian)
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid ddX PES is immutable.")
        object.__setattr__(self, name, value)

    @property
    def force_backend(self) -> RichardsonScalarForce:
        return self._force_backend

    @property
    def hessian_backend(self) -> RichardsonScalarHessian:
        return self._hessian_backend

    @property
    def coordinate_derivative_available(self) -> bool:
        return self._hybrid.coordinate_derivative_available

    @property
    def force_available(self) -> bool:
        return self.coordinate_derivative_available

    def _current_configuration(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-separated-ddx-operational-pes-v1",
                "provider_id": self.provider_id,
                "scalar_provider_id": HYBRID_DDX_SCALAR_PROVIDER_ID,
                "hybrid_configuration_sha256": self._hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": self._hybrid.provenance_sha256,
                "symbols": list(self._symbols),
                "cavity_radii_angstrom": self._cavity_radii_angstrom.tolist(),
                "continuum_settings": dict(self._continuum_settings),
                "model_drive": "external-mep-phi-adjoint-not-ledger-gradient",
                "energy_ledger": "vacuum-energy-plus-ddx-polarization",
                "force_derivative_kind": self.force_derivative_kind,
                "numerical_force_audit_kind": self.numerical_force_audit_kind,
                "coarse_step_angstrom": self._force_backend.coarse_step_angstrom,
                "fine_step_angstrom": self._force_backend.fine_step_angstrom,
                "maximum_error_eV_per_A": self._force_backend.maximum_error_eV_per_A,
                "capabilities": "experimental-callable-not-release-admitted",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid ddX PES configuration drifted.")
        return current

    def _validate_geometry(self, geometry: object) -> None:
        getter = getattr(geometry, "get_chemical_symbols", None)
        if not callable(getter):
            raise TypeError("geometry must expose get_chemical_symbols().")
        if tuple(str(value) for value in getter()) != self._symbols:
            raise ValueError("geometry symbols differ from the PES configuration.")

    def _continuum(self) -> SeparatedSourceDDXBackend:
        return SeparatedSourceDDXBackend(
            self._symbols,
            self._cavity_radii_angstrom,
            **dict(self._continuum_settings),
        )

    def _evaluator(self, geometry: object) -> MACE_MDPPolarHybridDDXEnergy:
        self.configuration_sha256()
        self._validate_geometry(geometry)
        return MACE_MDPPolarHybridDDXEnergy(
            geometry,
            hybrid=self._hybrid,
            continuum=self._continuum(),
        )

    def solve(self, geometry: object) -> HybridDDXEnergyState:
        return self._evaluator(geometry).solve(geometry)

    @staticmethod
    def _sample_from_state(
        state: HybridDDXEnergyState,
        *,
        topology_id: str,
    ) -> ScalarEnergySample:
        return ScalarEnergySample(
            energy_eV=state.total_energy_ev,
            state_sha256=state.root_sha256,
            topology_id=topology_id,
            topology_observation_coverage="complete",
            unobservable_topology_components=(),
        )

    @classmethod
    def _force_sample_from_evaluation(
        cls,
        evaluation: HybridDDXForceEvaluation,
        *,
        topology_id: str,
    ) -> ScalarForceSample:
        return ScalarForceSample(
            energy_sample=cls._sample_from_state(
                evaluation.central_state,
                topology_id=topology_id,
            ),
            forces_eV_per_A=evaluation.total_forces_ev_per_angstrom,
            evaluation_sha256=evaluation.evaluation_sha256,
        )

    def sample(self, geometry: object) -> ScalarEnergySample:
        evaluator = self._evaluator(geometry)
        state = evaluator.solve(geometry)
        return self._sample_from_state(
            state,
            topology_id=evaluator.cavity_topology_sha256,
        )

    def topology_id(self, geometry: object) -> str:
        """Return the observed ddX cavity topology for this geometry."""

        return self._evaluator(geometry).cavity_topology_sha256

    def get_potential_energy(self, geometry: object) -> float:
        return self.solve(geometry).total_energy_ev

    def evaluate_forces(
        self,
        geometry: object,
        *,
        central_state: HybridDDXEnergyState | None = None,
    ) -> HybridDDXForceEvaluation:
        evaluator = self._evaluator(geometry)
        state = evaluator.solve(geometry) if central_state is None else central_state
        return evaluator.evaluate_forces(geometry, central_state=state)

    def _validated_force_evaluation(
        self,
        geometry: object,
        evaluation: HybridDDXForceEvaluation,
    ) -> tuple[HybridDDXForceEvaluation, str]:
        if not isinstance(evaluation, HybridDDXForceEvaluation):
            raise TypeError("force_evaluation must be HybridDDXForceEvaluation.")
        evaluator = self._evaluator(geometry)
        evaluator._validate_central_state(geometry, evaluation.central_state)
        replayed = evaluator.evaluate_forces(
            geometry,
            central_state=evaluation.central_state,
        )
        if (
            replayed.evaluation_sha256 != evaluation.evaluation_sha256
            or not np.array_equal(
                replayed.total_forces_ev_per_angstrom,
                evaluation.total_forces_ev_per_angstrom,
            )
        ):
            raise ValueError(
                "cached force evaluation did not replay for the current "
                "provider and geometry."
            )
        return replayed, evaluator.cavity_topology_sha256

    def _evaluated_force_sample(
        self,
        geometry: object,
        evaluation: HybridDDXForceEvaluation | None = None,
    ) -> tuple[HybridDDXForceEvaluation, ScalarForceSample]:
        if evaluation is None:
            evaluator = self._evaluator(geometry)
            state = evaluator.solve(geometry)
            evaluated = evaluator.evaluate_forces(
                geometry,
                central_state=state,
            )
            topology_id = evaluator.cavity_topology_sha256
        else:
            evaluated, topology_id = self._validated_force_evaluation(
                geometry,
                evaluation,
            )
        return evaluated, self._force_sample_from_evaluation(
            evaluated,
            topology_id=topology_id,
        )

    def force_sample(self, geometry: object) -> ScalarForceSample:
        _evaluation, sample = self._evaluated_force_sample(geometry)
        return sample

    def get_forces(self, geometry: object) -> np.ndarray:
        return np.array(
            self.evaluate_forces(geometry).total_forces_ev_per_angstrom, copy=True
        )

    def molecular_virial(
        self,
        geometry: object,
        *,
        origin_angstrom: object | None = None,
        force_evaluation: HybridDDXForceEvaluation | None = None,
    ) -> MolecularVirialEvaluation:
        self._validate_geometry(geometry)
        if force_evaluation is None:
            evaluated = self.evaluate_forces(geometry)
        else:
            evaluated, _topology_id = self._validated_force_evaluation(
                geometry,
                force_evaluation,
            )
        return MolecularVirialEvaluation.from_conservative_force(
            force_evaluation_sha256=evaluated.evaluation_sha256,
            positions_angstrom=getattr(geometry, "positions", None),
            forces_eV_per_A=evaluated.total_forces_ev_per_angstrom,
            origin_angstrom=origin_angstrom,
        )

    def hessian_vector_product(
        self,
        geometry: object,
        direction: object,
        *,
        central_force: HybridDDXForceEvaluation | None = None,
    ) -> RichardsonScalarHVPEvaluation:
        self._validate_geometry(geometry)
        _center, sample = self._evaluated_force_sample(
            geometry,
            central_force,
        )
        return self._hessian_backend.evaluate_hvp(
            self,
            geometry,
            direction,
            central_sample=sample,
        )

    def evaluate_hessian(
        self,
        geometry: object,
        *,
        central_force: HybridDDXForceEvaluation | None = None,
    ) -> RichardsonScalarHessianEvaluation:
        self._validate_geometry(geometry)
        _center, sample = self._evaluated_force_sample(
            geometry,
            central_force,
        )
        return self._hessian_backend.evaluate(
            self,
            geometry,
            central_sample=sample,
        )

    def get_hessian(self, geometry: object) -> np.ndarray:
        return np.array(
            self.evaluate_hessian(geometry).hessian_eV_per_A2,
            copy=True,
        )

    def numerical_force(self, geometry: object) -> RichardsonScalarForceEvaluation:
        return self._force_backend.evaluate(self, geometry)

    def numerical_force_component(
        self,
        geometry: object,
        *,
        atom_index: int,
        axis_index: int,
    ) -> RichardsonScalarForceComponentEvaluation:
        return self._force_backend.evaluate_component(
            self,
            geometry,
            atom_index=atom_index,
            axis_index=axis_index,
        )


__all__ = [
    "HYBRID_DDX_PES_PROVIDER_ID",
    "HYBRID_DDX_ROOT_CONTRACT_ID",
    "HYBRID_DDX_SCALAR_PROVIDER_ID",
    "HybridDDXEnergyState",
    "HybridDDXForceEvaluation",
    "MACE_MDPPolarHybridDDXEnergy",
    "MACE_MDPPolarHybridDDXPES",
]
