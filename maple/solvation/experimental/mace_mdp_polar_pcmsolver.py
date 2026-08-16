"""MACE-MDP/MACE-POLAR/PCMSolver operational scalar and numerical force.

The scalar is deliberately narrow::

    E_exp = E_vac^MACE-POLAR + 1/2 v_total.T q_total

where the MACE-MDP permanent moments use the exterior point-multipole kernel,
only the field-induced MACE-POLAR increment uses the 1.5-A Gaussian kernel,
and the surface charge returns through the checkpoint-native 1.5/3.0-A radial
receiver.  The field-conditioned raw MACE-POLAR energy, MACE-MDP molecular
polarizability, nonpolar terms, Hessians, and MD are not part of this profile.

PCMSolver's public C API exposes no nuclear derivative for this external-MEP
construction.  The force fallback therefore rebuilds the cavity, resolves the
self-consistent root, and differentiates *this exact scalar* with a fourth-
order Richardson central stencil.  A topology change or excessive local
stencil error fails closed.  This is an explicitly numerical operational force,
not an analytic adjoint and not a strict common variational functional.

This module exposes a public result only when the authoritative registry has
admitted the experimental energy/force tier.  The numerical root and force can
be exercised before admission to produce bounded evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
from typing import Any

from ase.units import Bohr
import numpy as np

from maple.function.calculator.extra_correction.implicit.continuum_response import (
    ExternalMEPCavityResponse,
    PCMSolverExternalMEPCavityResponse,
)
from maple.function.calculator.extra_correction.implicit.gto_density import (
    point_multipole_potential,
)
from maple.function.calculator.extra_correction.implicit.gto_galerkin import (
    AtomCenteredL1GTOBasis,
)
from maple.function.calculator.extra_correction.implicit.pcmsolver import (
    PCMSolverSession,
)
from maple.solvation.api.profiles import (
    EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1,
    get_solvation_profile,
)
from maple.solvation.api.provenance import (
    ProvenanceBundle,
    ProvenanceRecord,
    RuntimeProvenance,
)
from maple.solvation.api.result import EnergyComponent, ForceComponent, Route2Result
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
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

PROFILE_ID = EXPERIMENTAL_MACE_MDP_POLAR_HYBRID_PCMSOLVER_ELECTROSTATIC_PROFILE_V1
ROOT_CONTRACT = "mace-mdp-polar-hybrid-pcmsolver-two-start-root-v1"
ROOT_TOLERANCE_EV = 1.0e-10
ROOT_REPLAY_FIELD_ATOL_EV = 2.0e-9
ROOT_REPLAY_ENERGY_ATOL_EV = 1.0e-10
TOTAL_CHARGE_ATOL_E = 1.0e-8
MAX_ROOT_ITERATIONS = 40
REQUIRED_LONG_RANGE_EVALUATOR = (
    "graph-longrange-analytic-gaussian-multipole-realspace-v1"
)
HYBRID_PCMSOLVER_SCALAR_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-pcmsolver-scalar.impl.v1"
)
NUMERICAL_FORCE_COARSE_STEP_ANGSTROM = 5.0e-4
NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM = 2.0e-4
_PCMSOLVER_WORKING_DIRECTORY_LOCK = threading.RLock()


def _array_sha256(values: object, *, name: str) -> str:
    array = np.asarray(values)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite array.")
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode()
    return hashlib.sha256(header + b"\0" + contiguous.tobytes()).hexdigest()


def _file_sha256(path: Path, *, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is unavailable: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
class HybridPCMSolverEnergyState:
    """Immutable, geometry-bound two-start root and scalar leaves."""

    geometry_sha256: str
    evaluator_configuration_sha256: str
    anchor_state_sha256: str
    native_field_ev: np.ndarray
    induced_source4: np.ndarray
    total_source4: np.ndarray
    surface_potential_hartree_per_e: np.ndarray
    surface_charge_e: np.ndarray
    vacuum_energy_ev: float
    polarization_energy_ev: float
    primal_residual_ev: float
    cold_iterations: int
    wide_iterations: int
    replay_field_max_abs_difference_ev: float
    replay_energy_abs_difference_ev: float
    cavity_topology_id: str
    root_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "geometry_sha256",
            "evaluator_configuration_sha256",
            "anchor_state_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 digest.")
        if (
            not isinstance(self.cavity_topology_id, str)
            or not self.cavity_topology_id.strip()
        ):
            raise ValueError("cavity_topology_id must be a non-empty string.")
        field = np.asarray(self.native_field_ev, dtype=float)
        source = np.asarray(self.total_source4, dtype=float)
        induced = np.asarray(self.induced_source4, dtype=float)
        potential = np.asarray(self.surface_potential_hartree_per_e, dtype=float)
        charge = np.asarray(self.surface_charge_e, dtype=float)
        if field.ndim != 2 or field.shape[1] != 8:
            raise ValueError("native_field_ev must have shape (N,8).")
        if source.shape != (field.shape[0], 4):
            raise ValueError("total_source4 must have shape (N,4).")
        if induced.shape != source.shape:
            raise ValueError("induced_source4 must match total_source4.")
        if potential.ndim != 1 or charge.shape != potential.shape:
            raise ValueError("surface potential and charge must be matching vectors.")
        field = _readonly(field, shape=field.shape, name="native_field_ev")
        source = _readonly(source, shape=source.shape, name="total_source4")
        induced = _readonly(induced, shape=source.shape, name="induced_source4")
        potential = _readonly(
            potential, shape=potential.shape, name="surface_potential"
        )
        charge = _readonly(charge, shape=charge.shape, name="surface_charge")
        finite_scalars = {}
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
            if "residual" in name or "difference" in name:
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative.")
            finite_scalars[name] = value
        for name in ("cold_iterations", "wide_iterations"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= MAX_ROOT_ITERATIONS:
                raise ValueError(f"{name} is outside the frozen iteration bound.")
        payload = {
            "contract": ROOT_CONTRACT,
            "geometry_sha256": self.geometry_sha256,
            "evaluator_configuration_sha256": (self.evaluator_configuration_sha256),
            "anchor_state_sha256": self.anchor_state_sha256,
            "native_field_sha256": _array_sha256(field, name="native field"),
            "induced_source_sha256": _array_sha256(induced, name="induced source"),
            "total_source_sha256": _array_sha256(source, name="total source"),
            "surface_potential_sha256": _array_sha256(
                potential, name="surface potential"
            ),
            "surface_charge_sha256": _array_sha256(charge, name="surface charge"),
            **finite_scalars,
            "cold_iterations": self.cold_iterations,
            "wide_iterations": self.wide_iterations,
            "cavity_topology_id": self.cavity_topology_id,
        }
        expected = canonical_metadata_sha256(payload)
        if self.root_sha256 and self.root_sha256 != expected:
            raise ValueError("root_sha256 does not match the root content.")
        object.__setattr__(self, "native_field_ev", field)
        object.__setattr__(self, "induced_source4", induced)
        object.__setattr__(self, "total_source4", source)
        object.__setattr__(self, "surface_potential_hartree_per_e", potential)
        object.__setattr__(self, "surface_charge_e", charge)
        object.__setattr__(self, "cavity_topology_id", self.cavity_topology_id.strip())
        for name, value in finite_scalars.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "root_sha256", expected)

    @property
    def total_energy_ev(self) -> float:
        return self.vacuum_energy_ev + self.polarization_energy_ev


class MACE_MDPPolarHybridPCMSolverEnergy:
    """One geometry-bound experimental Route-2 scalar evaluator."""

    __slots__ = (
        "_anchor",
        "_configuration_sha256",
        "_geometry_sha256",
        "_hybrid",
        "_induced_operator_hartree",
        "_permanent_potential_hartree",
        "_radial_coupling",
        "_receiver_operator_ev",
        "_response",
        "_response_configuration_sha256",
        "_cavity_configuration_sha256",
        "_sealed",
    )

    profile_id = PROFILE_ID
    force_available = False
    coordinate_derivative_available = False

    def __init__(
        self,
        geometry: object,
        *,
        hybrid: PermanentAnchoredInducedSourceModel,
        response: ExternalMEPCavityResponse,
    ) -> None:
        if not isinstance(hybrid, PermanentAnchoredInducedSourceModel):
            raise TypeError("hybrid must be PermanentAnchoredInducedSourceModel.")
        for name in (
            "configuration_sha256",
            "cavity_configuration_sha256",
            "apply_energy_conjugate",
            "solve",
        ):
            if not callable(getattr(response, name, None)):
                raise TypeError(f"response requires callable {name}().")
        if not bool(getattr(response, "energy_response_is_reciprocal", False)):
            raise ValueError(
                "experimental energy requires reciprocal PCMSolver response."
            )
        if hybrid.long_range_evaluator_profile != REQUIRED_LONG_RANGE_EVALUATOR:
            raise ValueError(
                "experimental hybrid energy requires the analytic Gaussian-"
                "multipole MACE-POLAR evaluator."
            )
        count = atom_count(geometry)
        positions = np.asarray(getattr(geometry, "positions"), dtype=float)
        numbers = np.asarray(getattr(geometry, "numbers"), dtype=float)
        if count != int(getattr(response, "atom_count", -1)):
            raise ValueError("geometry and PCMSolver response atom counts differ.")
        if not np.array_equal(numbers, np.asarray(response.atomic_numbers)):
            raise ValueError("geometry and PCMSolver response elements differ.")
        reference_positions = np.asarray(response.reference_positions_bohr) * Bohr
        if not np.allclose(positions, reference_positions, rtol=0.0, atol=1.0e-10):
            raise ValueError("geometry and PCMSolver cavity coordinates differ.")
        response.configuration_sha256()
        response.cavity_configuration_sha256()
        anchor = hybrid.prepare(geometry)
        points = np.asarray(response.surface_points_bohr, dtype=float)
        induced = AtomCenteredL1GTOBasis((1.5,)).surface_operator(points, positions)
        radial = MACEPolarRadialGTOCoupling()
        receiver = radial.surface_operator(FixedSurfaceGeometry(positions, points))
        permanent = point_multipole_potential(
            points, positions, anchor.permanent_source4
        )
        expected_induced_shape = (len(points), count * 4)
        expected_receiver_shape = (len(points), count * 8)
        if induced.shape != expected_induced_shape:
            raise RuntimeError("induced GTO operator has the wrong shape.")
        if receiver.shape != expected_receiver_shape:
            raise RuntimeError("radial receiver operator has the wrong shape.")
        configuration = canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-hybrid-pcmsolver-energy-evaluator-v1",
                "profile_id": self.profile_id,
                "geometry_sha256": geometry_sha256(geometry),
                "hybrid_configuration_sha256": hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": hybrid.provenance_sha256,
                "response_configuration_sha256": response.configuration_sha256(),
                "cavity_configuration_sha256": (response.cavity_configuration_sha256()),
                "radial_coupling_configuration_sha256": (radial.configuration_sha256()),
                "permanent_kernel": hybrid.permanent_source_kernel,
                "induced_kernel": hybrid.induced_source_kernel,
                "long_range_evaluator": hybrid.long_range_evaluator_profile,
                "root_contract": ROOT_CONTRACT,
                "root_tolerance_ev": ROOT_TOLERANCE_EV,
                "root_replay_field_atol_ev": ROOT_REPLAY_FIELD_ATOL_EV,
                "root_replay_energy_atol_ev": ROOT_REPLAY_ENERGY_ATOL_EV,
                "maximum_root_iterations": MAX_ROOT_ITERATIONS,
                "induced_operator_sha256": _array_sha256(
                    induced, name="induced operator"
                ),
                "receiver_operator_sha256": _array_sha256(
                    receiver, name="receiver operator"
                ),
                "permanent_potential_sha256": _array_sha256(
                    permanent, name="permanent potential"
                ),
            }
        )
        induced.setflags(write=False)
        receiver.setflags(write=False)
        permanent.setflags(write=False)
        object.__setattr__(self, "_geometry_sha256", geometry_sha256(geometry))
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_response", response)
        object.__setattr__(
            self,
            "_response_configuration_sha256",
            response.configuration_sha256(),
        )
        object.__setattr__(
            self,
            "_cavity_configuration_sha256",
            response.cavity_configuration_sha256(),
        )
        object.__setattr__(self, "_anchor", anchor)
        object.__setattr__(self, "_radial_coupling", radial)
        object.__setattr__(self, "_induced_operator_hartree", induced)
        object.__setattr__(self, "_receiver_operator_ev", receiver)
        object.__setattr__(self, "_permanent_potential_hartree", permanent)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid PCMSolver energy evaluator is immutable.")
        object.__setattr__(self, name, value)

    @property
    def anchor(self) -> PermanentInducedSourceAnchor:
        return self._anchor

    def configuration_sha256(self) -> str:
        self._hybrid.configuration_sha256()
        if self._response.configuration_sha256() != self._response_configuration_sha256:
            raise RuntimeError("PCMSolver response configuration drifted.")
        if (
            self._response.cavity_configuration_sha256()
            != self._cavity_configuration_sha256
        ):
            raise RuntimeError("PCMSolver cavity configuration drifted.")
        self._radial_coupling.configuration_sha256()
        return self._configuration_sha256

    def _validate_geometry(self, geometry: object) -> int:
        if geometry_sha256(geometry) != self._geometry_sha256:
            raise ValueError("evaluator is bound to a different geometry/cavity.")
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
            potential = self._permanent_potential_hartree + (
                self._induced_operator_hartree @ induced.reshape(-1)
            )
            charge = np.asarray(
                self._response.apply_energy_conjugate(potential), dtype=float
            )
            target = (self._receiver_operator_ev.T @ charge).reshape(count, 8)
            residual = float(np.linalg.norm(target - field))
            field = target
            if residual < ROOT_TOLERANCE_EV:
                break
        else:
            raise RuntimeError(
                f"hybrid root did not converge in {MAX_ROOT_ITERATIONS} iterations."
            )
        induced = self._hybrid.induced_source(geometry, self._anchor, field)
        potential = self._permanent_potential_hartree + (
            self._induced_operator_hartree @ induced.reshape(-1)
        )
        continuum_state = self._response.solve(potential)
        charge = np.asarray(
            continuum_state.energy_conjugate_surface_charge_e, dtype=float
        )
        target = (self._receiver_operator_ev.T @ charge).reshape(count, 8)
        final_residual = float(np.linalg.norm(target - field))
        energy_ev = float(continuum_state.polarization_energy_hartree) * (HARTREE_TO_EV)
        half_coupling_ev = 0.5 * float(np.vdot(potential, charge)) * HARTREE_TO_EV
        if not np.isclose(energy_ev, half_coupling_ev, rtol=1.0e-10, atol=1.0e-11):
            raise RuntimeError("PCMSolver energy violates the half-coupling identity.")
        return {
            "iterations": iteration,
            "field": field,
            "induced": induced,
            "potential": potential,
            "charge": charge,
            "energy_ev": energy_ev,
            "residual_ev": final_residual,
        }

    def _cavity_topology_id(self) -> str:
        """Return a geometry-independent discrete GEPOL branch signature."""

        points_angstrom = np.asarray(self._response.surface_points_bohr) * Bohr
        centers = np.asarray(self._response.reference_positions_bohr) * Bohr
        radii = np.asarray(self._response.cavity_radii_angstrom)
        shell_error = np.abs(
            np.linalg.norm(points_angstrom[:, None, :] - centers[None, :, :], axis=2)
            - radii[None, :]
        )
        owners = np.argmin(shell_error, axis=1)
        return canonical_metadata_sha256(
            {
                "contract": "pcmsolver-gepol-cardinality-owner-sequence-v1",
                "surface_size": int(len(points_angstrom)),
                "owner_sequence": owners.tolist(),
            }
        )

    def solve(self, geometry: object) -> HybridPCMSolverEnergyState:
        """Solve two deterministic starts and return the frozen scalar state."""

        self.configuration_sha256()
        count = self._validate_geometry(geometry)
        zero = np.zeros((count, 8), dtype=float)
        permanent_charge = np.asarray(
            self._response.apply_energy_conjugate(self._permanent_potential_hartree),
            dtype=float,
        )
        permanent_field = (self._receiver_operator_ev.T @ permanent_charge).reshape(
            count, 8
        )
        cold = self._solve_from(geometry, zero)
        wide = self._solve_from(geometry, 2.0 * permanent_field)
        field_difference = float(np.max(np.abs(cold["field"] - wide["field"])))
        energy_difference = abs(float(cold["energy_ev"]) - float(wide["energy_ev"]))
        if field_difference > ROOT_REPLAY_FIELD_ATOL_EV:
            raise RuntimeError("hybrid deterministic starts found different roots.")
        if energy_difference > ROOT_REPLAY_ENERGY_ATOL_EV:
            raise RuntimeError("hybrid deterministic starts disagree in energy.")
        if float(cold["residual_ev"]) >= ROOT_TOLERANCE_EV:
            raise RuntimeError("hybrid root residual exceeds the admitted tolerance.")
        total_source = self._hybrid.total_source(geometry, self._anchor, cold["field"])
        charge_error = abs(
            float(np.sum(total_source[:, 0])) - self._anchor.total_charge_e
        )
        if charge_error > TOTAL_CHARGE_ATOL_E:
            raise RuntimeError("hybrid root violates fixed total charge.")
        return HybridPCMSolverEnergyState(
            geometry_sha256=self._geometry_sha256,
            evaluator_configuration_sha256=self.configuration_sha256(),
            anchor_state_sha256=self._anchor.state_sha256,
            native_field_ev=cold["field"],
            induced_source4=cold["induced"],
            total_source4=total_source,
            surface_potential_hartree_per_e=cold["potential"],
            surface_charge_e=cold["charge"],
            vacuum_energy_ev=self._hybrid.vacuum_energy_ev(geometry),
            polarization_energy_ev=float(cold["energy_ev"]),
            primal_residual_ev=float(cold["residual_ev"]),
            cold_iterations=int(cold["iterations"]),
            wide_iterations=int(wide["iterations"]),
            replay_field_max_abs_difference_ev=field_difference,
            replay_energy_abs_difference_ev=energy_difference,
            cavity_topology_id=self._cavity_topology_id(),
        )

    def _provenance(self) -> ProvenanceBundle:
        return ProvenanceBundle(
            model=ProvenanceRecord(
                identity=self._hybrid.provider_id,
                kind="model",
                version="hybrid-permanent-induced/1",
                sha256=self._hybrid.provenance_sha256,
                metadata=(
                    ("configuration_sha256", self._hybrid.configuration_sha256()),
                    ("long_range_evaluator", self._hybrid.long_range_evaluator_profile),
                ),
            ),
            continuum=ProvenanceRecord(
                identity=str(self._response.provider_id),
                kind="continuum",
                version="external-mep/1",
                sha256=self._response.configuration_sha256(),
                metadata=(("profile", str(self._response.continuum_profile_id)),),
            ),
            cavity=ProvenanceRecord(
                identity=str(self._response.cavity_profile_id),
                kind="cavity",
                version="input-defined/1",
                sha256=self._response.cavity_configuration_sha256(),
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
        """Return the admitted public E-only result or fail closed."""

        if type(need_forces) is not bool:
            raise TypeError("need_forces must be a bool.")
        if need_forces:
            raise NotImplementedError(
                "The geometry-bound evaluator cannot move its PCMSolver cavity; "
                "use MACE_MDPPolarHybridPCMSolverPES for numerical force."
            )
        profile = get_solvation_profile(self.profile_id)
        if not profile.enabled or not profile.capabilities.energy:
            raise RuntimeError(
                "The experimental hybrid energy profile has not passed admission."
            )
        state = self.solve(geometry)
        return Route2Result(
            atom_count=atom_count(geometry),
            profile_id=self.profile_id,
            energy_components=(
                EnergyComponent(
                    "macepolar_zero_field_vacuum_energy",
                    state.vacuum_energy_ev,
                ),
                EnergyComponent(
                    "hybrid_pcmsolver_half_coupling_electrostatic",
                    state.polarization_energy_ev,
                ),
            ),
            provenance=self._provenance(),
            primal_residual=state.primal_residual_ev,
            adjoint_residual=None,
            root_identity=("zero-field-and-twice-permanent-field-starts-agree-v1"),
            root_sha256=state.root_sha256,
            evidence_artifact_ids=profile.evidence_artifact_ids,
            admitted_domain=(
                ("accuracy", "not-admitted"),
                ("capability", "experimental-single-point-energy-only"),
                ("charge_spin", "neutral-singlet"),
                ("component", "electrostatic-polarization-only"),
                ("continuum", "explicit-content-addressed-pcmsolver-input"),
                ("nonpolar", "excluded"),
            ),
            warnings=(
                "Experimental electrostatic single-point energy only; this is not "
                "a complete solvation free energy or an accuracy admission.",
                "Forces, geometry optimization, frequencies, and molecular dynamics "
                "are unavailable and fail closed.",
                "Strict Tier V is not claimed; the root and selected scalar are "
                "operational rather than a common variational functional.",
            ),
            fail_closed=False,
        )

    def get_forces(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise NotImplementedError(
            "A geometry-bound PCMSolver evaluator cannot move its cavity. Use "
            "MACE_MDPPolarHybridPCMSolverPES for numerical scalar-gradient forces."
        )


class MACE_MDPPolarHybridPCMSolverPES:
    """Geometry-resolved scalar family with error-estimated numerical forces."""

    __slots__ = (
        "_atomic_numbers",
        "_cavity_radii_angstrom",
        "_configuration_sha256",
        "_force_backend",
        "_hybrid",
        "_library_path",
        "_library_sha256",
        "_parsed_input_path",
        "_parsed_input_sha256",
        "_sealed",
    )

    provider_id = HYBRID_PCMSOLVER_SCALAR_PROVIDER_ID
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
        parsed_input_path: str | Path,
        pcmsolver_library_path: str | Path,
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
        parsed = Path(parsed_input_path).expanduser().resolve(strict=True)
        library = Path(pcmsolver_library_path).expanduser().resolve(strict=True)
        parsed_sha = _file_sha256(parsed, name="PCMSolver parsed input")
        library_sha = _file_sha256(library, name="PCMSolver shared library")
        backend = force_backend or RichardsonScalarForce(
            coarse_step_angstrom=NUMERICAL_FORCE_COARSE_STEP_ANGSTROM,
            maximum_error_eV_per_A=NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
        )
        if not isinstance(backend, RichardsonScalarForce):
            raise TypeError("force_backend must be RichardsonScalarForce.")
        configuration = canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-hybrid-pcmsolver-pes-v1",
                "provider_id": self.provider_id,
                "profile_id": self.profile_id,
                "hybrid_configuration_sha256": hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": hybrid.provenance_sha256,
                "atomic_numbers": numbers.tolist(),
                "cavity_radii_angstrom": radii.tolist(),
                "parsed_input_sha256": parsed_sha,
                "library_sha256": library_sha,
                "force_derivative_kind": self.force_derivative_kind,
                "coarse_step_angstrom": backend.coarse_step_angstrom,
                "fine_step_angstrom": backend.fine_step_angstrom,
                "maximum_error_eV_per_A": backend.maximum_error_eV_per_A,
            }
        )
        numbers.setflags(write=False)
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        radii.setflags(write=False)
        object.__setattr__(self, "_hybrid", hybrid)
        object.__setattr__(self, "_atomic_numbers", numbers)
        object.__setattr__(self, "_cavity_radii_angstrom", radii)
        object.__setattr__(self, "_parsed_input_path", parsed)
        object.__setattr__(self, "_library_path", library)
        object.__setattr__(self, "_parsed_input_sha256", parsed_sha)
        object.__setattr__(self, "_library_sha256", library_sha)
        object.__setattr__(self, "_force_backend", backend)
        object.__setattr__(self, "_configuration_sha256", configuration)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid PCMSolver PES is immutable.")
        object.__setattr__(self, name, value)

    @property
    def force_backend(self) -> RichardsonScalarForce:
        return self._force_backend

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-hybrid-pcmsolver-pes-v1",
                "provider_id": self.provider_id,
                "profile_id": self.profile_id,
                "hybrid_configuration_sha256": self._hybrid.configuration_sha256(),
                "hybrid_provenance_sha256": self._hybrid.provenance_sha256,
                "atomic_numbers": self._atomic_numbers.tolist(),
                "cavity_radii_angstrom": self._cavity_radii_angstrom.tolist(),
                "parsed_input_sha256": _file_sha256(
                    self._parsed_input_path, name="PCMSolver parsed input"
                ),
                "library_sha256": _file_sha256(
                    self._library_path, name="PCMSolver shared library"
                ),
                "force_derivative_kind": self.force_derivative_kind,
                "coarse_step_angstrom": self._force_backend.coarse_step_angstrom,
                "fine_step_angstrom": self._force_backend.fine_step_angstrom,
                "maximum_error_eV_per_A": (self._force_backend.maximum_error_eV_per_A),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid PCMSolver PES configuration drifted.")
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

    @contextmanager
    def _open_evaluator(self, geometry: object):
        self.configuration_sha256()
        self._validate_geometry(geometry)
        with _PCMSOLVER_WORKING_DIRECTORY_LOCK:
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="route2-hybrid-pes-") as work:
                try:
                    os.chdir(work)
                    with PCMSolverSession(
                        self._atomic_numbers.astype(float),
                        np.asarray(geometry.positions, dtype=float) / Bohr,
                        self._parsed_input_path,
                        library_path=self._library_path,
                    ) as session:
                        response = PCMSolverExternalMEPCavityResponse(
                            session,
                            cavity_radii_angstrom=self._cavity_radii_angstrom,
                        )
                        yield MACE_MDPPolarHybridPCMSolverEnergy(
                            geometry,
                            hybrid=self._hybrid,
                            response=response,
                        )
                finally:
                    os.chdir(previous)

    def solve(self, geometry: object) -> HybridPCMSolverEnergyState:
        with self._open_evaluator(geometry) as evaluator:
            return evaluator.solve(geometry)

    @staticmethod
    def _sample_from_state(state: HybridPCMSolverEnergyState) -> ScalarEnergySample:
        return ScalarEnergySample(
            energy_eV=state.total_energy_ev,
            state_sha256=state.root_sha256,
            topology_id=state.cavity_topology_id,
            topology_observation_coverage="complete",
            unobservable_topology_components=(),
        )

    @staticmethod
    def _validate_central_state(
        geometry: object, state: HybridPCMSolverEnergyState
    ) -> None:
        if not isinstance(state, HybridPCMSolverEnergyState):
            raise TypeError("central_state must be HybridPCMSolverEnergyState.")
        if state.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("central_state is bound to a different geometry.")

    def sample(self, geometry: object) -> ScalarEnergySample:
        return self._sample_from_state(self.solve(geometry))

    def numerical_force(
        self,
        geometry: object,
        *,
        central_state: HybridPCMSolverEnergyState | None = None,
    ) -> RichardsonScalarForceEvaluation:
        state = self.solve(geometry) if central_state is None else central_state
        self._validate_central_state(geometry, state)
        return self._force_backend.evaluate(
            self,
            geometry,
            central_sample=self._sample_from_state(state),
        )

    def numerical_force_component(
        self,
        geometry: object,
        *,
        atom_index: int,
        axis_index: int,
        central_state: HybridPCMSolverEnergyState | None = None,
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
        return ProvenanceBundle(
            model=ProvenanceRecord(
                identity=self._hybrid.provider_id,
                kind="model",
                version="hybrid-permanent-induced/1",
                sha256=self._hybrid.provenance_sha256,
                metadata=(
                    ("configuration_sha256", self._hybrid.configuration_sha256()),
                    (
                        "long_range_evaluator",
                        self._hybrid.long_range_evaluator_profile,
                    ),
                ),
            ),
            continuum=ProvenanceRecord(
                identity=self.provider_id,
                kind="continuum",
                version="external-mep-pes/1",
                sha256=self.configuration_sha256(),
                metadata=(
                    ("library_sha256", self._library_sha256),
                    (
                        "profile",
                        "pcmsolver-symmetric-external-mep-electrostatic-v1",
                    ),
                ),
            ),
            cavity=ProvenanceRecord(
                identity="pcmsolver-input-defined-gepol-cavity-v1",
                kind="cavity",
                version="input-defined/1",
                sha256=self._parsed_input_sha256,
                metadata=(
                    (
                        "radii_sha256",
                        _array_sha256(self._cavity_radii_angstrom, name="cavity radii"),
                    ),
                ),
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
                "The experimental hybrid scalar profile has not passed admission."
            )
        if need_forces and not profile.capabilities.conservative_force:
            raise RuntimeError(
                "The experimental hybrid numerical force has not passed admission."
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
                    "hybrid_operational_scalar_numerical_gradient",
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
                    "macepolar_zero_field_vacuum_energy",
                    state.vacuum_energy_ev,
                ),
                EnergyComponent(
                    "hybrid_pcmsolver_half_coupling_electrostatic",
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
                (
                    "capability",
                    "experimental-electrostatic-energy-and-numerical-force",
                ),
                ("charge_spin", "neutral-singlet"),
                ("component", "electrostatic-polarization-only"),
                ("continuum", "explicit-content-addressed-pcmsolver-input"),
                ("force", self.force_derivative_kind),
                ("nonpolar", "excluded"),
            ),
            warnings=(
                "Experimental electrostatic scalar and numerical scalar-gradient "
                "force; this is not a complete solvation free energy or an "
                "accuracy admission.",
                "The force rebuilds PCMSolver and resolves the root at every "
                "Richardson stencil point; topology changes or excessive stencil "
                "error fail closed.",
                "Hessians, frequencies, and molecular dynamics remain unavailable.",
                "Strict Tier V is not claimed; the root and selected scalar are "
                "operational rather than a common variational functional.",
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
    "HYBRID_PCMSOLVER_SCALAR_PROVIDER_ID",
    "HybridPCMSolverEnergyState",
    "MACE_MDPPolarHybridPCMSolverEnergy",
    "MACE_MDPPolarHybridPCMSolverPES",
    "MAX_ROOT_ITERATIONS",
    "NUMERICAL_FORCE_COARSE_STEP_ANGSTROM",
    "NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM",
    "PROFILE_ID",
    "ROOT_REPLAY_ENERGY_ATOL_EV",
    "ROOT_REPLAY_FIELD_ATOL_EV",
    "ROOT_TOLERANCE_EV",
    "TOTAL_CHARGE_ATOL_E",
]
