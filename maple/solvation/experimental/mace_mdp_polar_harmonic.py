"""MACE-MDP/MACE-POLAR smooth-harmonic operational scalar and force.

This is the fixed-coefficient-topology replacement for the failed GEPOL
finite-difference force candidate.  The selected scalar is

``E = E_vac^POLAR - f_eps/2 b(u*)^T A(R)^-1 b(u*)``

with a point-harmonic permanent MACE-MDP source, a Gaussian-harmonic induced
MACE-POLAR increment, and the checkpoint-native two-width radial receiver.
The public force is the matrix-free implicit-adjoint total derivative of this
same operational scalar.  A full re-solved Richardson stencil remains an
independent diagnostic oracle.  This is not a Tier-V common variational
functional.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import platform
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.api.provenance import (
    ProvenanceBundle,
    ProvenanceRecord,
    RuntimeProvenance,
)
from maple.solvation.api.result import EnergyComponent, ForceComponent, Route2Result
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1,
)
from maple.solvation.continuum.harmonic_point_source import (
    HARMONIC_POINT_SOURCE_CONTRACT_ID,
    HARMONIC_POINT_SOURCE_PROVIDER_ID,
    harmonic_point_source_implementation_sha256,
    point_harmonic_source_operator,
)
from maple.solvation.continuum.harmonic_torch_functional import (
    SmoothWeightedHarmonicGalerkinFunctionalCandidate,
)
from maple.solvation.continuum.harmonic_torch_primitives import (
    _assemble_point_source,
    _torch,
)
from maple.solvation.coupling.exact_gto import (
    mace_polar_learned_source_embedding_matrix,
)
from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives import (
    RichardsonScalarForce,
    RichardsonScalarForceComponentEvaluation,
    RichardsonScalarForceEvaluation,
    ScalarEnergySample,
)
from maple.solvation.models.base import atom_count, model_charge_and_multiplicity
from maple.solvation.models.mace_mdp_polar_hybrid import (
    MACE_MDP_POLAR_HYBRID_PROFILE_ID,
    MACE_MDP_POLAR_HYBRID_PROVIDER_ID,
    PermanentAnchoredInducedSourceModel,
    PermanentInducedSourceAnchor,
)

PROFILE_ID = (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_PROFILE_V1
)
SCALAR_ID = EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_SMOOTH_HARMONIC_GALERKIN_ELECTROSTATIC_V1
ROOT_CONTRACT = "mace-mdp-polar-hybrid-smooth-harmonic-two-start-root-v1"
ROOT_TOLERANCE_EV = 1.0e-10
ROOT_REPLAY_FIELD_ATOL_EV = 2.0e-9
ROOT_REPLAY_ENERGY_ATOL_EV = 1.0e-10
TOTAL_CHARGE_ATOL_E = 1.0e-8
MAX_ROOT_ITERATIONS = 40
REQUIRED_LONG_RANGE_EVALUATOR = (
    "graph-longrange-analytic-gaussian-multipole-realspace-v1"
)
HYBRID_HARMONIC_SCALAR_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-smooth-harmonic-scalar.impl.v1"
)
NUMERICAL_FORCE_COARSE_STEP_ANGSTROM = 5.0e-4
NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM = 2.0e-4
ANALYTIC_ADJOINT_TOLERANCE_EV = 1.0e-10
MAX_ANALYTIC_ADJOINT_ITERATIONS = 200
ADMITTED_CONTINUUM_SETTINGS = tuple(
    sorted(
        {
            "transition_width_angstrom2": 0.18,
            "surface_lmax": 1,
            "exposure_lmax": 2,
            "exposure_radial_quadrature_order": 32,
            "source_radial_quadrature_order": 32,
            "green_radial_quadrature_order": 32,
        }.items()
    )
)
ADMITTED_HYBRID_CONFIGURATION_SHA256 = (
    "ad866e18797d12614c98bc04e4060674a6d3e9801d6ae83f45a515cfbfc68aae"
)
ADMITTED_HYBRID_PROVENANCE_SHA256 = (
    "439e7b3585e2bcfb828bd346e163614ee1571b78895ce36536ddabfc633ef74e"
)
ADMITTED_DTYPE = "float64"
ADMITTED_DEVICE = "cuda"


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
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(shape)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True)
class HybridHarmonicEnergyState:
    """Immutable root and scalar leaves for one geometry."""

    geometry_sha256: str
    evaluator_configuration_sha256: str
    anchor_state_sha256: str
    continuum_state_sha256: str
    native_field_ev: np.ndarray
    induced_source4: np.ndarray
    total_source4: np.ndarray
    boundary_rhs: np.ndarray
    boundary_state: np.ndarray
    vacuum_energy_ev: float
    polarization_energy_ev: float
    primal_residual_ev: float
    cold_iterations: int
    wide_iterations: int
    replay_field_max_abs_difference_ev: float
    replay_energy_abs_difference_ev: float
    coefficient_topology_id: str
    root_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "geometry_sha256",
            "evaluator_configuration_sha256",
            "anchor_state_sha256",
            "continuum_state_sha256",
            "coefficient_topology_id",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        field = np.asarray(self.native_field_ev, dtype=float)
        total = np.asarray(self.total_source4, dtype=float)
        induced = np.asarray(self.induced_source4, dtype=float)
        rhs = np.asarray(self.boundary_rhs, dtype=float)
        sigma = np.asarray(self.boundary_state, dtype=float)
        if field.ndim != 2 or field.shape[1] != 8:
            raise ValueError("native_field_ev must have shape (N,8).")
        if total.shape != (field.shape[0], 4) or induced.shape != total.shape:
            raise ValueError("hybrid source arrays must have shape (N,4).")
        if rhs.ndim != 1 or sigma.shape != rhs.shape:
            raise ValueError("boundary rhs/state must be matching vectors.")
        field = _readonly(field, shape=field.shape, name="native field")
        total = _readonly(total, shape=total.shape, name="total source")
        induced = _readonly(induced, shape=induced.shape, name="induced source")
        rhs = _readonly(rhs, shape=rhs.shape, name="boundary rhs")
        sigma = _readonly(sigma, shape=sigma.shape, name="boundary state")
        scalars: dict[str, float] = {}
        for name in (
            "vacuum_energy_ev",
            "polarization_energy_ev",
            "primal_residual_ev",
            "replay_field_max_abs_difference_ev",
            "replay_energy_abs_difference_ev",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if ("residual" in name or "difference" in name) and value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            scalars[name] = value
        for name in ("cold_iterations", "wide_iterations"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_ROOT_ITERATIONS:
                raise ValueError(f"{name} is outside the frozen iteration bound.")
        payload = {
            "contract": ROOT_CONTRACT,
            "geometry_sha256": self.geometry_sha256,
            "evaluator_configuration_sha256": self.evaluator_configuration_sha256,
            "anchor_state_sha256": self.anchor_state_sha256,
            "continuum_state_sha256": self.continuum_state_sha256,
            "native_field_sha256": _array_sha256(field, name="native field"),
            "induced_source_sha256": _array_sha256(induced, name="induced source"),
            "total_source_sha256": _array_sha256(total, name="total source"),
            "boundary_rhs_sha256": _array_sha256(rhs, name="boundary rhs"),
            "boundary_state_sha256": _array_sha256(sigma, name="boundary state"),
            **scalars,
            "cold_iterations": self.cold_iterations,
            "wide_iterations": self.wide_iterations,
            "coefficient_topology_id": self.coefficient_topology_id,
        }
        expected = canonical_metadata_sha256(payload)
        if self.root_sha256 and self.root_sha256 != expected:
            raise ValueError("root_sha256 does not match the root content.")
        object.__setattr__(self, "native_field_ev", field)
        object.__setattr__(self, "total_source4", total)
        object.__setattr__(self, "induced_source4", induced)
        object.__setattr__(self, "boundary_rhs", rhs)
        object.__setattr__(self, "boundary_state", sigma)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "root_sha256", expected)

    @property
    def total_energy_ev(self) -> float:
        return self.vacuum_energy_ev + self.polarization_energy_ev


@dataclass(frozen=True, slots=True)
class HybridHarmonicAnalyticForceEvaluation:
    """Contraction-only implicit-adjoint derivative of the operational scalar."""

    central_state: HybridHarmonicEnergyState
    vacuum_forces_ev_per_angstrom: np.ndarray
    continuum_fixed_source_forces_ev_per_angstrom: np.ndarray
    permanent_source_forces_ev_per_angstrom: np.ndarray
    induced_source_forces_ev_per_angstrom: np.ndarray
    model_field_geometry_forces_ev_per_angstrom: np.ndarray
    adjoint_field_ev: np.ndarray
    adjoint_residual_ev: float
    adjoint_iterations: int

    def __post_init__(self) -> None:
        if not isinstance(self.central_state, HybridHarmonicEnergyState):
            raise TypeError("central_state must be HybridHarmonicEnergyState.")
        count = self.central_state.total_source4.shape[0]
        for name in (
            "vacuum_forces_ev_per_angstrom",
            "continuum_fixed_source_forces_ev_per_angstrom",
            "permanent_source_forces_ev_per_angstrom",
            "induced_source_forces_ev_per_angstrom",
            "model_field_geometry_forces_ev_per_angstrom",
        ):
            value = _readonly(getattr(self, name), shape=(count, 3), name=name)
            object.__setattr__(self, name, value)
        adjoint = _readonly(
            self.adjoint_field_ev,
            shape=self.central_state.native_field_ev.shape,
            name="adjoint_field_ev",
        )
        residual = float(self.adjoint_residual_ev)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("adjoint_residual_ev must be finite and non-negative.")
        if (
            type(self.adjoint_iterations) is not int
            or not 0 <= self.adjoint_iterations <= MAX_ANALYTIC_ADJOINT_ITERATIONS
        ):
            raise ValueError("adjoint_iterations is outside the configured bound.")
        object.__setattr__(self, "adjoint_field_ev", adjoint)
        object.__setattr__(self, "adjoint_residual_ev", residual)

    @property
    def total_forces_ev_per_angstrom(self) -> np.ndarray:
        return sum(
            (
                self.vacuum_forces_ev_per_angstrom,
                self.continuum_fixed_source_forces_ev_per_angstrom,
                self.permanent_source_forces_ev_per_angstrom,
                self.induced_source_forces_ev_per_angstrom,
                self.model_field_geometry_forces_ev_per_angstrom,
            ),
            start=np.zeros_like(self.vacuum_forces_ev_per_angstrom),
        )


class MACE_MDPPolarHybridSmoothHarmonicEnergy:
    """One geometry-bound hybrid scalar evaluator."""

    __slots__ = (
        "_anchor",
        "_configuration_sha256",
        "_continuum",
        "_geometry_sha256",
        "_hybrid",
        "_induced_operator",
        "_permanent_rhs",
        "_point_source_implementation_sha256",
        "_receiver_operator",
        "_screening_factor",
        "_sealed",
        "_surface_operator",
    )

    profile_id = PROFILE_ID
    force_available = True
    coordinate_derivative_available = True

    def __init__(
        self,
        geometry: object,
        *,
        hybrid: PermanentAnchoredInducedSourceModel,
        continuum: SmoothWeightedHarmonicGalerkinFunctionalCandidate,
    ) -> None:
        if not isinstance(hybrid, PermanentAnchoredInducedSourceModel):
            raise TypeError("hybrid must be PermanentAnchoredInducedSourceModel.")
        if not isinstance(continuum, SmoothWeightedHarmonicGalerkinFunctionalCandidate):
            raise TypeError(
                "continuum must be SmoothWeightedHarmonicGalerkinFunctionalCandidate."
            )
        if continuum.scalar_id != SCALAR_ID:
            raise ValueError("harmonic continuum is bound to a different scalar.")
        if hybrid.long_range_evaluator_profile != REQUIRED_LONG_RANGE_EVALUATOR:
            raise ValueError(
                "experimental hybrid energy requires the analytic Gaussian-"
                "multipole MACE-POLAR evaluator."
            )
        count = atom_count(geometry)
        positions = np.asarray(getattr(geometry, "positions", None), dtype=float)
        if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("geometry positions must be finite with shape (N,3).")
        matrices = continuum.debug_geometry_matrices(geometry)
        weighted_basis = np.asarray(matrices["weighted_basis"], dtype=float)
        surface = np.asarray(matrices["surface_operator"], dtype=float)
        source8 = np.asarray(matrices["source_operator"], dtype=float)
        if source8.shape[1] != count * 8 or surface.shape[0] != source8.shape[0]:
            raise RuntimeError("harmonic source/surface dimensions are inconsistent.")
        point_raw = point_harmonic_source_operator(
            positions_angstrom=positions,
            radii_angstrom=continuum.radii_angstrom,
            surface_lmax=continuum.physical_lmax,
            radial_quadrature_order=continuum.source_radial_quadrature_order,
        )
        point_operator = weighted_basis.T @ point_raw
        embedding = np.kron(np.eye(count), mace_polar_learned_source_embedding_matrix())
        induced_operator = source8 @ embedding
        screening_factor = continuum.cpcm_screening_factor
        receiver = -screening_factor * source8.T
        anchor = hybrid.prepare(geometry)
        permanent_rhs = point_operator @ anchor.permanent_source4.reshape(-1)
        configuration = canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-hybrid-smooth-harmonic-energy-v1",
                "profile_id": self.profile_id,
                "scalar_id": SCALAR_ID,
                "geometry_sha256": geometry_sha256(geometry),
                "hybrid_configuration_sha256": hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": hybrid.provenance_sha256,
                "continuum_configuration_sha256": continuum.configuration_sha256(),
                "continuum_provenance_sha256": continuum.provenance_sha256,
                "coefficient_topology_sha256": continuum.topology_sha256(),
                "point_source_contract_id": HARMONIC_POINT_SOURCE_CONTRACT_ID,
                "point_source_provider_id": HARMONIC_POINT_SOURCE_PROVIDER_ID,
                "point_source_implementation_sha256": (
                    harmonic_point_source_implementation_sha256()
                ),
                "surface_operator_sha256": _array_sha256(surface, name="surface"),
                "induced_operator_sha256": _array_sha256(
                    induced_operator, name="induced operator"
                ),
                "receiver_operator_sha256": _array_sha256(
                    receiver, name="receiver operator"
                ),
                "cpcm_screening_factor": screening_factor,
                "permanent_rhs_sha256": _array_sha256(
                    permanent_rhs, name="permanent rhs"
                ),
                "root_contract": ROOT_CONTRACT,
            }
        )
        for array in (surface, induced_operator, receiver, permanent_rhs):
            array.setflags(write=False)
        object.__setattr__(self, "_geometry_sha256", geometry_sha256(geometry))
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_continuum", continuum)
        object.__setattr__(self, "_anchor", anchor)
        object.__setattr__(self, "_surface_operator", surface)
        object.__setattr__(self, "_induced_operator", induced_operator)
        object.__setattr__(self, "_receiver_operator", receiver)
        object.__setattr__(self, "_screening_factor", screening_factor)
        object.__setattr__(self, "_permanent_rhs", permanent_rhs)
        object.__setattr__(
            self,
            "_point_source_implementation_sha256",
            harmonic_point_source_implementation_sha256(),
        )
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid smooth-harmonic evaluator is immutable.")
        object.__setattr__(self, name, value)

    @property
    def anchor(self) -> PermanentInducedSourceAnchor:
        return self._anchor

    def configuration_sha256(self) -> str:
        self._hybrid.configuration_sha256()
        self._continuum.configuration_sha256()
        self._continuum.topology_sha256()
        if (
            harmonic_point_source_implementation_sha256()
            != self._point_source_implementation_sha256
        ):
            raise RuntimeError("harmonic point-source implementation drifted.")
        return self._configuration_sha256

    def _joint_torch_graph(
        self,
        geometry: object,
        permanent_source4: object,
        induced_source4: object,
        *,
        coordinate_grad: bool,
        source_grad: bool,
    ):
        """Build the heterogeneous point/Gaussian scalar and receiver graph."""

        count = self._validate_geometry(geometry)
        permanent = self._hybrid.source_space.validate(
            permanent_source4,
            atom_count=count,
            name="permanent source",
        )
        induced = self._hybrid.source_space.validate(
            induced_source4,
            atom_count=count,
            name="induced source",
        )
        positions = self._continuum._positions_tensor(
            geometry,
            atom_count=count,
            requires_grad=coordinate_grad,
        )
        torch = _torch()
        permanent_tensor = torch.tensor(
            permanent,
            dtype=positions.dtype,
            device=positions.device,
            requires_grad=source_grad,
        )
        induced_tensor = torch.tensor(
            induced,
            dtype=positions.dtype,
            device=positions.device,
            requires_grad=source_grad,
        )
        weighted_basis, _, _, surface, gaussian_source8 = (
            self._continuum._assemble_torch(positions)
        )
        raw_point = _assemble_point_source(
            positions,
            radii=self._continuum.radii_angstrom,
            lmax=self._continuum.physical_lmax,
            radial_order=self._continuum.source_radial_quadrature_order,
        )
        point_operator = weighted_basis.T @ raw_point
        embedding = positions.new_tensor(
            np.kron(
                np.eye(count),
                mace_polar_learned_source_embedding_matrix(),
            )
        )
        induced_operator = gaussian_source8 @ embedding
        rhs = point_operator @ permanent_tensor.reshape(
            -1
        ) + induced_operator @ induced_tensor.reshape(-1)
        sigma = torch.linalg.solve(surface, rhs)
        screening = self._screening_factor
        energy = -0.5 * screening * (rhs @ sigma)
        field = (-screening * gaussian_source8.T @ sigma).reshape(count, 8)
        return positions, permanent_tensor, induced_tensor, energy, field

    @staticmethod
    def _validate_joint_graph_replay(
        state: HybridHarmonicEnergyState,
        energy: object,
        field: object,
    ) -> None:
        energy_value = float(energy.detach().cpu())
        field_values = np.asarray(field.detach().cpu(), dtype=float)
        if not np.isclose(
            energy_value,
            state.polarization_energy_ev,
            rtol=2.0e-11,
            atol=2.0e-11,
        ):
            raise RuntimeError("Torch harmonic scalar did not replay the root energy.")
        if not np.allclose(
            field_values,
            state.native_field_ev,
            rtol=2.0e-10,
            atol=2.0e-10,
        ):
            raise RuntimeError("Torch harmonic receiver did not replay the root field.")

    def _energy_source_gradients(
        self,
        geometry: object,
        state: HybridHarmonicEnergyState,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, permanent, induced, energy, field = self._joint_torch_graph(
            geometry,
            self._anchor.permanent_source4,
            state.induced_source4,
            coordinate_grad=False,
            source_grad=True,
        )
        self._validate_joint_graph_replay(state, energy, field)
        permanent_gradient, induced_gradient = _torch().autograd.grad(
            energy,
            (permanent, induced),
            create_graph=False,
            allow_unused=False,
        )
        return (
            np.asarray(permanent_gradient.detach().cpu(), dtype=float).copy(),
            np.asarray(induced_gradient.detach().cpu(), dtype=float).copy(),
        )

    def _field_source_vjp(
        self,
        geometry: object,
        state: HybridHarmonicEnergyState,
        field_cotangent: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        cotangent = self._hybrid.receiver_space.validate(
            field_cotangent,
            atom_count=state.total_source4.shape[0],
            name="harmonic receiver cotangent",
        )
        _, permanent, induced, energy, field = self._joint_torch_graph(
            geometry,
            self._anchor.permanent_source4,
            state.induced_source4,
            coordinate_grad=False,
            source_grad=True,
        )
        self._validate_joint_graph_replay(state, energy, field)
        cotangent_tensor = field.new_tensor(cotangent)
        contraction = _torch().sum(field * cotangent_tensor)
        permanent_gradient, induced_gradient = _torch().autograd.grad(
            contraction,
            (permanent, induced),
            create_graph=False,
            allow_unused=False,
        )
        return (
            np.asarray(permanent_gradient.detach().cpu(), dtype=float).copy(),
            np.asarray(induced_gradient.detach().cpu(), dtype=float).copy(),
        )

    def _coordinate_partials(
        self,
        geometry: object,
        state: HybridHarmonicEnergyState,
        field_cotangent: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        cotangent = self._hybrid.receiver_space.validate(
            field_cotangent,
            atom_count=state.total_source4.shape[0],
            name="harmonic receiver cotangent",
        )
        positions, _, _, energy, field = self._joint_torch_graph(
            geometry,
            self._anchor.permanent_source4,
            state.induced_source4,
            coordinate_grad=True,
            source_grad=False,
        )
        self._validate_joint_graph_replay(state, energy, field)
        (energy_gradient,) = _torch().autograd.grad(
            energy,
            (positions,),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        field_contraction = _torch().sum(field * field.new_tensor(cotangent))
        (field_gradient,) = _torch().autograd.grad(
            field_contraction,
            (positions,),
            create_graph=False,
            allow_unused=False,
        )
        return (
            np.asarray(energy_gradient.detach().cpu(), dtype=float).copy(),
            np.asarray(field_gradient.detach().cpu(), dtype=float).copy(),
        )

    def _validate_geometry(self, geometry: object) -> int:
        if geometry_sha256(geometry) != self._geometry_sha256:
            raise ValueError("evaluator is bound to a different geometry.")
        return atom_count(geometry)

    def _solve_from(
        self, geometry: object, initial_field: np.ndarray
    ) -> dict[str, object]:
        count = self._validate_geometry(geometry)
        field = self._hybrid.receiver_space.validate(
            initial_field, atom_count=count, name="initial native field"
        )
        for iteration in range(1, MAX_ROOT_ITERATIONS + 1):
            induced = self._hybrid.induced_source(geometry, self._anchor, field)
            rhs = self._permanent_rhs + self._induced_operator @ induced.reshape(-1)
            sigma = np.linalg.solve(self._surface_operator, rhs)
            target = (self._receiver_operator @ sigma).reshape(count, 8)
            residual = float(np.linalg.norm(target - field))
            field = target
            if residual < ROOT_TOLERANCE_EV:
                break
        else:
            raise RuntimeError(
                f"hybrid harmonic root did not converge in {MAX_ROOT_ITERATIONS} iterations."
            )
        induced = self._hybrid.induced_source(geometry, self._anchor, field)
        rhs = self._permanent_rhs + self._induced_operator @ induced.reshape(-1)
        sigma = np.linalg.solve(self._surface_operator, rhs)
        target = (self._receiver_operator @ sigma).reshape(count, 8)
        final_residual = float(np.linalg.norm(target - field))
        energy = -0.5 * self._screening_factor * float(np.vdot(rhs, sigma))
        if not np.isfinite(energy):
            raise RuntimeError("hybrid harmonic scalar is non-finite.")
        return {
            "iterations": iteration,
            "field": field,
            "induced": induced,
            "rhs": rhs,
            "sigma": sigma,
            "energy_ev": energy,
            "residual_ev": final_residual,
        }

    def solve(self, geometry: object) -> HybridHarmonicEnergyState:
        self.configuration_sha256()
        count = self._validate_geometry(geometry)
        permanent_sigma = np.linalg.solve(self._surface_operator, self._permanent_rhs)
        permanent_field = (self._receiver_operator @ permanent_sigma).reshape(count, 8)
        cold = self._solve_from(geometry, np.zeros((count, 8), dtype=float))
        wide = self._solve_from(geometry, 2.0 * permanent_field)
        field_difference = float(
            np.max(np.abs(np.asarray(cold["field"]) - np.asarray(wide["field"])))
        )
        energy_difference = abs(float(cold["energy_ev"]) - float(wide["energy_ev"]))
        if field_difference > ROOT_REPLAY_FIELD_ATOL_EV:
            raise RuntimeError("hybrid harmonic starts found different roots.")
        if energy_difference > ROOT_REPLAY_ENERGY_ATOL_EV:
            raise RuntimeError("hybrid harmonic starts disagree in energy.")
        if float(cold["residual_ev"]) >= ROOT_TOLERANCE_EV:
            raise RuntimeError("hybrid harmonic root residual exceeds tolerance.")
        total_source = self._hybrid.total_source(
            geometry, self._anchor, np.asarray(cold["field"])
        )
        charge_error = abs(
            float(np.sum(total_source[:, 0])) - self._anchor.total_charge_e
        )
        if charge_error > TOTAL_CHARGE_ATOL_E:
            raise RuntimeError("hybrid harmonic root violates fixed total charge.")
        return HybridHarmonicEnergyState(
            geometry_sha256=self._geometry_sha256,
            evaluator_configuration_sha256=self.configuration_sha256(),
            anchor_state_sha256=self._anchor.state_sha256,
            continuum_state_sha256=canonical_metadata_sha256(
                {
                    "continuum_configuration_sha256": (
                        self._continuum.configuration_sha256()
                    ),
                    "geometry_sha256": self._geometry_sha256,
                    "surface_operator_sha256": _array_sha256(
                        self._surface_operator, name="surface"
                    ),
                }
            ),
            native_field_ev=np.asarray(cold["field"]),
            induced_source4=np.asarray(cold["induced"]),
            total_source4=total_source,
            boundary_rhs=np.asarray(cold["rhs"]),
            boundary_state=np.asarray(cold["sigma"]),
            vacuum_energy_ev=self._hybrid.vacuum_energy_ev(geometry),
            polarization_energy_ev=float(cold["energy_ev"]),
            primal_residual_ev=float(cold["residual_ev"]),
            cold_iterations=int(cold["iterations"]),
            wide_iterations=int(wide["iterations"]),
            replay_field_max_abs_difference_ev=field_difference,
            replay_energy_abs_difference_ev=energy_difference,
            coefficient_topology_id=self._continuum.topology_sha256(),
        )

    def evaluate_analytic_forces(
        self,
        geometry: object,
        *,
        central_state: HybridHarmonicEnergyState | None = None,
    ) -> HybridHarmonicAnalyticForceEvaluation:
        """Differentiate the operational scalar by an implicit adjoint.

        This is the public derivative route after independent comparison with
        the admitted Richardson force.  It makes no common-functional claim.
        """

        if not self._hybrid.coordinate_derivative_available:
            raise NotImplementedError(
                "Hybrid response has no complete coordinate derivative."
            )
        state = self.solve(geometry) if central_state is None else central_state
        if not isinstance(state, HybridHarmonicEnergyState):
            raise TypeError("central_state must be HybridHarmonicEnergyState.")
        if state.geometry_sha256 != self._geometry_sha256:
            raise ValueError("central_state belongs to a different geometry.")
        if state.evaluator_configuration_sha256 != self.configuration_sha256():
            raise ValueError("central_state belongs to a different evaluator.")

        energy_permanent, energy_induced = self._energy_source_gradients(
            geometry, state
        )
        rhs = self._hybrid.field_vjp(
            geometry,
            self._anchor,
            state.native_field_ev,
            energy_induced,
        )
        dimension = rhs.size
        iterations = 0

        def transpose_residual_action(flat_values: np.ndarray) -> np.ndarray:
            field_cotangent = np.asarray(flat_values, dtype=float).reshape(
                state.native_field_ev.shape
            )
            _, induced_cotangent = self._field_source_vjp(
                geometry, state, field_cotangent
            )
            response = self._hybrid.field_vjp(
                geometry,
                self._anchor,
                state.native_field_ev,
                induced_cotangent,
            )
            return (field_cotangent - response).reshape(-1)

        operator = LinearOperator(
            (dimension, dimension),
            matvec=transpose_residual_action,
            dtype=np.float64,
        )
        restart = min(50, dimension)

        def count_iteration(_residual: object) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = gmres(
            operator,
            rhs.reshape(-1),
            atol=ANALYTIC_ADJOINT_TOLERANCE_EV,
            rtol=0.0,
            restart=restart,
            maxiter=MAX_ANALYTIC_ADJOINT_ITERATIONS,
            callback=count_iteration,
            # ``legacy`` makes ``maxiter`` count inner Krylov iterations,
            # matching the explicit bound stored in the immutable result.
            callback_type="legacy",
        )
        adjoint = np.asarray(solution, dtype=float).reshape(state.native_field_ev.shape)
        true_residual = float(
            np.linalg.norm(
                transpose_residual_action(adjoint.reshape(-1)) - rhs.reshape(-1)
            )
        )
        if info != 0 or true_residual > 10.0 * ANALYTIC_ADJOINT_TOLERANCE_EV:
            raise RuntimeError(
                "hybrid harmonic adjoint did not satisfy its true residual: "
                f"info={info}, residual={true_residual:.6e} eV."
            )

        implicit_permanent, implicit_induced = self._field_source_vjp(
            geometry, state, adjoint
        )
        permanent_gradient = self._hybrid.permanent_source_position_vjp(
            geometry,
            self._anchor,
            energy_permanent + implicit_permanent,
        )
        induced_gradient = self._hybrid.induced_source_position_vjp(
            geometry,
            self._anchor,
            state.native_field_ev,
            energy_induced + implicit_induced,
        )
        continuum_gradient, field_geometry_gradient = self._coordinate_partials(
            geometry, state, adjoint
        )
        return HybridHarmonicAnalyticForceEvaluation(
            central_state=state,
            vacuum_forces_ev_per_angstrom=(
                self._hybrid.vacuum_forces_ev_per_angstrom(geometry)
            ),
            continuum_fixed_source_forces_ev_per_angstrom=-continuum_gradient,
            permanent_source_forces_ev_per_angstrom=-permanent_gradient,
            induced_source_forces_ev_per_angstrom=-induced_gradient,
            model_field_geometry_forces_ev_per_angstrom=-field_geometry_gradient,
            adjoint_field_ev=adjoint,
            adjoint_residual_ev=true_residual,
            adjoint_iterations=iterations,
        )


class MACE_MDPPolarHybridSmoothHarmonicPES:
    """Geometry-resolved smooth scalar with an implicit-adjoint total force."""

    __slots__ = (
        "_atomic_numbers",
        "_cavity_radii_angstrom",
        "_configuration_sha256",
        "_continuum_settings",
        "_device",
        "_dielectric",
        "_dtype",
        "_force_backend",
        "_hybrid",
        "_sealed",
    )

    provider_id = HYBRID_HARMONIC_SCALAR_PROVIDER_ID
    profile_id = PROFILE_ID
    force_available = True
    coordinate_derivative_available = True
    force_derivative_kind = "operational-implicit-adjoint-total-derivative-v1"

    def __init__(
        self,
        *,
        hybrid: PermanentAnchoredInducedSourceModel,
        atomic_numbers: object,
        cavity_radii_angstrom: object,
        dtype: object,
        device: object,
        dielectric: float | None = None,
        transition_width_angstrom2: float = 0.18,
        surface_lmax: int = 1,
        exposure_lmax: int = 2,
        exposure_radial_quadrature_order: int = 32,
        source_radial_quadrature_order: int = 32,
        green_radial_quadrature_order: int = 32,
        force_backend: RichardsonScalarForce | None = None,
    ) -> None:
        if not isinstance(hybrid, PermanentAnchoredInducedSourceModel):
            raise TypeError("hybrid must be PermanentAnchoredInducedSourceModel.")
        numbers = np.asarray(atomic_numbers, dtype=float)
        if (
            numbers.ndim != 1
            or numbers.size == 0
            or not np.all(np.isfinite(numbers))
            or not np.array_equal(numbers, np.rint(numbers))
            or np.any(numbers < 1.0)
        ):
            raise ValueError("atomic_numbers must contain positive integers.")
        numbers = np.ascontiguousarray(numbers, dtype=np.int64)
        radii = np.asarray(cavity_radii_angstrom, dtype=float)
        if (
            radii.shape != numbers.shape
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("cavity_radii_angstrom must be positive with shape (N,).")
        if dielectric is None:
            normalized_dielectric = None
        else:
            if isinstance(dielectric, bool):
                raise TypeError("dielectric must be a real scalar or None.")
            normalized_dielectric = float(dielectric)
            if not np.isfinite(normalized_dielectric) or normalized_dielectric < 1.0:
                raise ValueError("dielectric must be finite and at least one.")
        settings = {
            "transition_width_angstrom2": float(transition_width_angstrom2),
            "surface_lmax": int(surface_lmax),
            "exposure_lmax": int(exposure_lmax),
            "exposure_radial_quadrature_order": int(exposure_radial_quadrature_order),
            "source_radial_quadrature_order": int(source_radial_quadrature_order),
            "green_radial_quadrature_order": int(green_radial_quadrature_order),
        }
        backend = force_backend or RichardsonScalarForce(
            coarse_step_angstrom=NUMERICAL_FORCE_COARSE_STEP_ANGSTROM,
            maximum_error_eV_per_A=NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
        )
        if not isinstance(backend, RichardsonScalarForce):
            raise TypeError("force_backend must be RichardsonScalarForce.")
        numbers.setflags(write=False)
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        radii.setflags(write=False)
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_atomic_numbers", numbers)
        object.__setattr__(self, "_cavity_radii_angstrom", radii)
        object.__setattr__(self, "_dtype", dtype)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_dielectric", normalized_dielectric)
        object.__setattr__(self, "_continuum_settings", tuple(sorted(settings.items())))
        object.__setattr__(self, "_force_backend", backend)
        object.__setattr__(
            self,
            "_configuration_sha256",
            self._current_configuration_sha256(),
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid smooth-harmonic PES is immutable.")
        object.__setattr__(self, name, value)

    @property
    def force_backend(self) -> RichardsonScalarForce:
        return self._force_backend

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-hybrid-smooth-harmonic-pes-v2",
                "provider_id": self.provider_id,
                "profile_id": self.profile_id,
                "scalar_id": SCALAR_ID,
                "hybrid_configuration_sha256": self._hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": self._hybrid.provenance_sha256,
                "atomic_numbers": self._atomic_numbers.tolist(),
                "cavity_radii_angstrom": self._cavity_radii_angstrom.tolist(),
                "dtype": str(self._dtype),
                "device": str(self._device),
                "dielectric": self._dielectric,
                "continuum_settings": dict(self._continuum_settings),
                "force_derivative_kind": self.force_derivative_kind,
                "analytic_adjoint_tolerance_eV": ANALYTIC_ADJOINT_TOLERANCE_EV,
                "maximum_analytic_adjoint_iterations": (
                    MAX_ANALYTIC_ADJOINT_ITERATIONS
                ),
                "richardson_role": "independent-diagnostic-oracle",
                "coarse_step_angstrom": self._force_backend.coarse_step_angstrom,
                "fine_step_angstrom": self._force_backend.fine_step_angstrom,
                "maximum_error_eV_per_A": self._force_backend.maximum_error_eV_per_A,
                "point_source_contract_id": HARMONIC_POINT_SOURCE_CONTRACT_ID,
                "point_source_implementation_sha256": (
                    harmonic_point_source_implementation_sha256()
                ),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid smooth-harmonic PES configuration drifted.")
        return current

    def _validate_geometry(self, geometry: object) -> None:
        numbers = np.asarray(getattr(geometry, "numbers", None), dtype=float)
        positions = np.asarray(getattr(geometry, "positions", None), dtype=float)
        if not np.array_equal(numbers, self._atomic_numbers.astype(float)):
            raise ValueError("geometry elements differ from the PES configuration.")
        if positions.shape != (len(self._atomic_numbers), 3) or not np.all(
            np.isfinite(positions)
        ):
            raise ValueError("geometry positions must be finite with shape (N,3).")

    def _validate_admitted_runtime(self, geometry: object) -> None:
        """Reject any runtime not covered by the replicated E/F artifact."""

        self.configuration_sha256()
        if model_charge_and_multiplicity(geometry) != (0, 1):
            raise RuntimeError(
                "The admitted hybrid harmonic profile is restricted to neutral singlets."
            )
        if (
            self._hybrid.provider_id != MACE_MDP_POLAR_HYBRID_PROVIDER_ID
            or self._hybrid.model_profile_id != MACE_MDP_POLAR_HYBRID_PROFILE_ID
            or self._hybrid.long_range_evaluator_profile
            != REQUIRED_LONG_RANGE_EVALUATOR
        ):
            raise RuntimeError(
                "Hybrid model identity is outside the admitted checkpoint profile."
            )
        if (
            self._hybrid.configuration_sha256() != ADMITTED_HYBRID_CONFIGURATION_SHA256
            or self._hybrid.provenance_sha256 != ADMITTED_HYBRID_PROVENANCE_SHA256
        ):
            raise RuntimeError(
                "Hybrid providers/checkpoints differ from the admitted evidence."
            )
        if self._continuum_settings != ADMITTED_CONTINUUM_SETTINGS:
            raise RuntimeError(
                "Harmonic continuum settings differ from the admitted evidence."
            )
        if self._dielectric is not None:
            raise RuntimeError(
                "Finite-dielectric harmonic electrostatics is outside the admitted "
                "conductor-limit evidence."
            )
        if (
            self._force_backend.coarse_step_angstrom
            != NUMERICAL_FORCE_COARSE_STEP_ANGSTROM
            or self._force_backend.maximum_error_eV_per_A
            != NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
        ):
            raise RuntimeError("Numerical-force settings differ from the admission.")
        if str(self._dtype) != ADMITTED_DTYPE or str(self._device) != ADMITTED_DEVICE:
            raise RuntimeError(
                "The replicated admission is restricted to float64 on cuda."
            )
        symbols_method = getattr(geometry, "get_chemical_symbols", None)
        if not callable(symbols_method):
            raise TypeError("admitted geometry must expose get_chemical_symbols().")
        expected_radii = np.asarray(
            smd_water_coulomb_radii(symbols_method()), dtype=float
        )
        if not np.array_equal(self._cavity_radii_angstrom, expected_radii):
            raise RuntimeError(
                "Cavity radii differ from the admitted SMD-water Coulomb radii."
            )

    def _continuum(self) -> SmoothWeightedHarmonicGalerkinFunctionalCandidate:
        return SmoothWeightedHarmonicGalerkinFunctionalCandidate(
            atomic_numbers=tuple(int(value) for value in self._atomic_numbers),
            radii_angstrom=tuple(float(value) for value in self._cavity_radii_angstrom),
            dtype=self._dtype,
            device=self._device,
            dielectric=self._dielectric,
            scalar_id=SCALAR_ID,
            **dict(self._continuum_settings),
        )

    def _evaluator(self, geometry: object) -> MACE_MDPPolarHybridSmoothHarmonicEnergy:
        self.configuration_sha256()
        self._validate_geometry(geometry)
        return MACE_MDPPolarHybridSmoothHarmonicEnergy(
            geometry, hybrid=self._hybrid, continuum=self._continuum()
        )

    def solve(self, geometry: object) -> HybridHarmonicEnergyState:
        return self._evaluator(geometry).solve(geometry)

    @staticmethod
    def _sample_from_state(state: HybridHarmonicEnergyState) -> ScalarEnergySample:
        return ScalarEnergySample(
            energy_eV=state.total_energy_ev,
            state_sha256=state.root_sha256,
            topology_id=state.coefficient_topology_id,
            topology_observation_coverage="complete",
            unobservable_topology_components=(),
        )

    @staticmethod
    def _validate_central_state(
        geometry: object, state: HybridHarmonicEnergyState
    ) -> None:
        if not isinstance(state, HybridHarmonicEnergyState):
            raise TypeError("central_state must be HybridHarmonicEnergyState.")
        if state.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("central_state is bound to a different geometry.")

    def sample(self, geometry: object) -> ScalarEnergySample:
        return self._sample_from_state(self.solve(geometry))

    def numerical_force(
        self,
        geometry: object,
        *,
        central_state: HybridHarmonicEnergyState | None = None,
    ) -> RichardsonScalarForceEvaluation:
        state = self.solve(geometry) if central_state is None else central_state
        self._validate_central_state(geometry, state)
        return self._force_backend.evaluate(
            self, geometry, central_sample=self._sample_from_state(state)
        )

    def analytic_force(
        self,
        geometry: object,
        *,
        central_state: HybridHarmonicEnergyState | None = None,
    ) -> HybridHarmonicAnalyticForceEvaluation:
        """Evaluate the public analytic-adjoint force with immutable leaves."""

        evaluator = self._evaluator(geometry)
        state = evaluator.solve(geometry) if central_state is None else central_state
        self._validate_central_state(geometry, state)
        return evaluator.evaluate_analytic_forces(geometry, central_state=state)

    def analytic_force_diagnostic(
        self,
        geometry: object,
        *,
        central_state: HybridHarmonicEnergyState | None = None,
    ) -> HybridHarmonicAnalyticForceEvaluation:
        """Compatibility alias retained for pre-admission validation callers."""

        return self.analytic_force(geometry, central_state=central_state)

    def numerical_force_component(
        self,
        geometry: object,
        *,
        atom_index: int,
        axis_index: int,
        central_state: HybridHarmonicEnergyState | None = None,
    ) -> RichardsonScalarForceComponentEvaluation:
        state = self.solve(geometry) if central_state is None else central_state
        self._validate_central_state(geometry, state)
        return self._force_backend.evaluate_component(
            self,
            geometry,
            atom_index=atom_index,
            axis_index=axis_index,
            central_sample=self._sample_from_state(state),
        )

    def _provenance(self) -> ProvenanceBundle:
        continuum = self._continuum()
        return ProvenanceBundle(
            model=ProvenanceRecord(
                identity=self._hybrid.provider_id,
                kind="model",
                version="hybrid-permanent-induced/1",
                sha256=self._hybrid.provenance_sha256,
                metadata=(
                    ("configuration_sha256", self._hybrid.configuration_sha256()),
                ),
            ),
            continuum=ProvenanceRecord(
                identity=continuum.provider_id,
                kind="continuum",
                version="smooth-harmonic/1",
                sha256=continuum.provenance_sha256,
                metadata=(("configuration_sha256", continuum.configuration_sha256()),),
            ),
            cavity=ProvenanceRecord(
                identity=continuum.cavity_profile_id,
                kind="cavity",
                version="smooth-weighted-overlap/1",
                sha256=continuum.topology_sha256(),
            ),
            runtime=RuntimeProvenance(
                python=sys.version.split()[0],
                platform=platform.platform(),
                dependencies=(
                    ("ase", _package_version("ase")),
                    ("mace-torch", _package_version("mace-torch")),
                    ("maple", _package_version("maple")),
                    ("numpy", _package_version("numpy")),
                    ("torch", _package_version("torch")),
                ),
            ),
        )

    def evaluate(self, geometry: object, *, need_forces: bool = False) -> Route2Result:
        if type(need_forces) is not bool:
            raise TypeError("need_forces must be a bool.")
        profile = get_solvation_profile(self.profile_id)
        if not profile.enabled or not profile.capabilities.energy:
            raise RuntimeError(
                "The experimental hybrid smooth-harmonic scalar has not passed admission."
            )
        if need_forces and not profile.capabilities.conservative_force:
            raise RuntimeError(
                "The experimental hybrid smooth-harmonic force has not passed admission."
            )
        self._validate_admitted_runtime(geometry)
        state = self.solve(geometry)
        force_evaluation = (
            self.analytic_force(geometry, central_state=state) if need_forces else None
        )
        force_components = ()
        if force_evaluation is not None:
            force_components = tuple(
                ForceComponent(
                    name,
                    tuple(tuple(float(value) for value in row) for row in values),
                )
                for name, values in (
                    (
                        "macepolar_zero_field_vacuum_force",
                        force_evaluation.vacuum_forces_ev_per_angstrom,
                    ),
                    (
                        "continuum_fixed_source_coordinate_force",
                        force_evaluation.continuum_fixed_source_forces_ev_per_angstrom,
                    ),
                    (
                        "mdp_permanent_source_coordinate_force",
                        force_evaluation.permanent_source_forces_ev_per_angstrom,
                    ),
                    (
                        "macepolar_induced_source_coordinate_force",
                        force_evaluation.induced_source_forces_ev_per_angstrom,
                    ),
                    (
                        "macepolar_native_field_geometry_force",
                        force_evaluation.model_field_geometry_forces_ev_per_angstrom,
                    ),
                )
            )
        return Route2Result(
            atom_count=atom_count(geometry),
            profile_id=self.profile_id,
            energy_components=(
                EnergyComponent(
                    "macepolar_zero_field_vacuum_energy", state.vacuum_energy_ev
                ),
                EnergyComponent(
                    "hybrid_smooth_harmonic_half_coupling_electrostatic",
                    state.polarization_energy_ev,
                ),
            ),
            force_components=force_components,
            provenance=self._provenance(),
            primal_residual=state.primal_residual_ev,
            adjoint_residual=(
                None
                if force_evaluation is None
                else force_evaluation.adjoint_residual_ev
            ),
            force_error_estimate_eV_per_A=None,
            root_identity="zero-field-and-twice-permanent-field-starts-agree-v1",
            root_sha256=state.root_sha256,
            evidence_artifact_ids=profile.evidence_artifact_ids,
            admitted_domain=(
                ("accuracy", "not-admitted"),
                ("capability", "experimental-electrostatic-E-and-analytic-F"),
                ("charge_spin", "neutral-singlet"),
                ("component", "electrostatic-polarization-only"),
                ("continuum", "smooth-fixed-coefficient-harmonic-galerkin"),
                ("device", ADMITTED_DEVICE),
                ("dtype", ADMITTED_DTYPE),
                ("force", self.force_derivative_kind),
                ("model_binding_sha256", ADMITTED_HYBRID_CONFIGURATION_SHA256),
                ("nonpolar", "excluded"),
                ("solvent", "water-cavity-conductor-limit-electrostatic"),
            ),
            warnings=(
                "Experimental electrostatic scalar and implicit-adjoint same-scalar "
                "force; not a complete solvation free energy or accuracy admission.",
                "Hessians, frequencies, and molecular dynamics remain unavailable.",
                "Strict Tier V is not claimed; this is an operational root/ledger.",
            ),
            fail_closed=False,
        )

    def get_forces(self, geometry: object) -> np.ndarray:
        result = self.evaluate(geometry, need_forces=True)
        forces = result.total_forces_eV_per_A
        if forces is None:  # pragma: no cover - registry/result invariant
            raise RuntimeError("Admitted force result omitted force leaves.")
        return np.asarray(forces, dtype=float)


__all__ = [
    "ADMITTED_CONTINUUM_SETTINGS",
    "ADMITTED_DEVICE",
    "ADMITTED_DTYPE",
    "ADMITTED_HYBRID_CONFIGURATION_SHA256",
    "ADMITTED_HYBRID_PROVENANCE_SHA256",
    "HYBRID_HARMONIC_SCALAR_PROVIDER_ID",
    "HybridHarmonicAnalyticForceEvaluation",
    "HybridHarmonicEnergyState",
    "MACE_MDPPolarHybridSmoothHarmonicEnergy",
    "MACE_MDPPolarHybridSmoothHarmonicPES",
    "NUMERICAL_FORCE_COARSE_STEP_ANGSTROM",
    "NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM",
    "PROFILE_ID",
    "ROOT_REPLAY_ENERGY_ATOL_EV",
    "ROOT_REPLAY_FIELD_ATOL_EV",
    "ROOT_TOLERANCE_EV",
    "SCALAR_ID",
]
