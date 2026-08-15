"""MACE-MDP/MACE-POLAR smooth-harmonic operational scalar and force.

This is the fixed-coefficient-topology replacement for the failed GEPOL
finite-difference force candidate.  The selected scalar is

``E = E_vac^POLAR - 1/2 b(u*)^T A(R)^-1 b(u*)``

with a point-harmonic permanent MACE-MDP source, a Gaussian-harmonic induced
MACE-POLAR increment, and the checkpoint-native two-width radial receiver.
The self-consistent root and the full scalar are rebuilt at every Richardson
stencil point.  This remains an operational scalar, not a Tier-V common
variational functional.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import platform
import sys

import numpy as np

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
from maple.solvation.models.base import atom_count
from maple.solvation.models.mace_mdp_polar_hybrid import (
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
        "_sealed",
        "_surface_operator",
    )

    profile_id = PROFILE_ID
    force_available = False
    coordinate_derivative_available = False

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
        receiver = -source8.T
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
        energy = -0.5 * float(np.vdot(rhs, sigma))
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


class MACE_MDPPolarHybridSmoothHarmonicPES:
    """Geometry-resolved smooth scalar with error-estimated numerical forces."""

    __slots__ = (
        "_atomic_numbers",
        "_cavity_radii_angstrom",
        "_configuration_sha256",
        "_continuum_settings",
        "_device",
        "_dtype",
        "_force_backend",
        "_hybrid",
        "_sealed",
    )

    provider_id = HYBRID_HARMONIC_SCALAR_PROVIDER_ID
    profile_id = PROFILE_ID
    force_available = True
    coordinate_derivative_available = False
    force_derivative_kind = "numerical-scalar-gradient-richardson-v1"

    def __init__(
        self,
        *,
        hybrid: PermanentAnchoredInducedSourceModel,
        atomic_numbers: object,
        cavity_radii_angstrom: object,
        dtype: object,
        device: object,
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
                "contract": "mace-mdp-polar-hybrid-smooth-harmonic-pes-v1",
                "provider_id": self.provider_id,
                "profile_id": self.profile_id,
                "scalar_id": SCALAR_ID,
                "hybrid_configuration_sha256": self._hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": self._hybrid.provenance_sha256,
                "atomic_numbers": self._atomic_numbers.tolist(),
                "cavity_radii_angstrom": self._cavity_radii_angstrom.tolist(),
                "dtype": str(self._dtype),
                "device": str(self._device),
                "continuum_settings": dict(self._continuum_settings),
                "force_derivative_kind": self.force_derivative_kind,
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

    def _continuum(self) -> SmoothWeightedHarmonicGalerkinFunctionalCandidate:
        return SmoothWeightedHarmonicGalerkinFunctionalCandidate(
            atomic_numbers=tuple(int(value) for value in self._atomic_numbers),
            radii_angstrom=tuple(float(value) for value in self._cavity_radii_angstrom),
            dtype=self._dtype,
            device=self._device,
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
        state = self.solve(geometry)
        force_evaluation = (
            self.numerical_force(geometry, central_state=state) if need_forces else None
        )
        force_components = ()
        force_error = None
        if force_evaluation is not None:
            force_components = (
                ForceComponent(
                    "hybrid_smooth_harmonic_scalar_numerical_gradient",
                    tuple(
                        tuple(float(value) for value in row)
                        for row in force_evaluation.forces_eV_per_A
                    ),
                ),
            )
            force_error = force_evaluation.maximum_error_estimate_eV_per_A
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
            adjoint_residual=None,
            force_error_estimate_eV_per_A=force_error,
            root_identity="zero-field-and-twice-permanent-field-starts-agree-v1",
            root_sha256=state.root_sha256,
            evidence_artifact_ids=profile.evidence_artifact_ids,
            admitted_domain=(
                ("accuracy", "not-admitted"),
                ("capability", "experimental-electrostatic-E-and-numerical-F"),
                ("charge_spin", "neutral-singlet"),
                ("component", "electrostatic-polarization-only"),
                ("continuum", "smooth-fixed-coefficient-harmonic-galerkin"),
                ("force", self.force_derivative_kind),
                ("nonpolar", "excluded"),
            ),
            warnings=(
                "Experimental electrostatic scalar and numerical scalar-gradient "
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
    "HYBRID_HARMONIC_SCALAR_PROVIDER_ID",
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
