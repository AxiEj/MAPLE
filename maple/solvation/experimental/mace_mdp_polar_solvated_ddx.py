"""Add an independent solvent term to the hybrid MACE-MDP/MACE-POLAR ddX PES.

The electronic fixed point and the additive CDS/nonpolar scalar remain separate
modules.  This composition therefore closes the full operational ledger without
teaching the hybrid response solver anything about SMD, and without duplicating
the existing :mod:`maple.solvation.solvent_terms` implementation.

The declared scalar is

``E = E_vac + G_ddX,pol + G_solvent``.

Its analytic force is the existing block-adjoint hybrid force minus the
independent solvent-term gradient.  Molecular virials and Richardson HVP/H are
derived from that total force.  PySCF SMD-CDS does not expose its internal
surface topology, so HVP/H remain strict by default and require an explicitly
experimental observed-components-only policy when that term is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from maple.solvation.coupling.operator import canonical_metadata_sha256
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives import (
    RichardsonScalarForce,
    RichardsonScalarForceEvaluation,
    RichardsonScalarHessian,
    RichardsonScalarHessianEvaluation,
    RichardsonScalarHVPEvaluation,
    ScalarEnergySample,
    ScalarForceSample,
    normalize_topology_observation,
)
from maple.solvation.experimental.mace_mdp_polar_ddx import (
    HybridDDXEnergyState,
    HybridDDXForceEvaluation,
    MACE_MDPPolarHybridDDXPES,
)
from maple.solvation.experimental.mace_polar_frozen_ddx import (
    MolecularVirialEvaluation,
)
from maple.solvation.models.mace_mdp_polar_hybrid import (
    PermanentAnchoredInducedSourceModel,
)
from maple.solvation.solvent_terms import (
    PySCFSMDCDSTerm,
    SolventEnergyState,
    SolventEnergyTerm,
)

HYBRID_SOLVATED_DDX_PES_PROVIDER_ID = (
    "maple.route2.experimental.mace-mdp-polar-separated-ddx-smd.impl.v1"
)
HYBRID_SOLVATED_DDX_SCALAR_CONTRACT_ID = (
    "mace-mdp-permanent-plus-mace-polar-induced-ddx-plus-solvent-term-v1"
)
HYBRID_SOLVATED_DDX_DAILY_PROFILE_ID = (
    "route2-experimental-mace-mdp-polar-separated-ddx-smd-daily-v1"
)
ENERGY_REPLAY_ATOL_EV = 2.0e-10
_MODULE_PATH = Path(__file__).resolve()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    result = np.ascontiguousarray(array, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class HybridSolvatedDDXEnergyState:
    """Content-addressed full hybrid ddX plus solvent-term energy ledger."""

    provider_id: str
    scalar_contract_id: str
    configuration_sha256: str
    geometry_sha256: str
    electrostatic_state: HybridDDXEnergyState
    solvent_state: SolventEnergyState
    topology_id: str
    topology_observation_coverage: str
    unobservable_topology_components: tuple[str, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if self.provider_id != HYBRID_SOLVATED_DDX_PES_PROVIDER_ID:
            raise ValueError("hybrid solvated ddX provider identity is invalid.")
        if self.scalar_contract_id != HYBRID_SOLVATED_DDX_SCALAR_CONTRACT_ID:
            raise ValueError("hybrid solvated ddX scalar contract is invalid.")
        configuration = _digest(self.configuration_sha256, name="configuration_sha256")
        geometry = _digest(self.geometry_sha256, name="geometry_sha256")
        if not isinstance(self.electrostatic_state, HybridDDXEnergyState):
            raise TypeError("electrostatic_state must be HybridDDXEnergyState.")
        if not isinstance(self.solvent_state, SolventEnergyState):
            raise TypeError("solvent_state must be SolventEnergyState.")
        if self.electrostatic_state.geometry_sha256 != geometry:
            raise ValueError("electrostatic state geometry identity differs.")
        if self.solvent_state.geometry_sha256 != geometry:
            raise ValueError("solvent state geometry identity differs.")
        if self.solvent_state.gradient_eV_per_A is not None:
            raise ValueError("energy state must contain an energy-only solvent state.")
        topology = _text(self.topology_id, name="topology_id")
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        payload = {
            "contract": "mace-mdp-polar-solvated-ddx-energy-state-v1",
            "provider_id": self.provider_id,
            "scalar_contract_id": self.scalar_contract_id,
            "configuration_sha256": configuration,
            "geometry_sha256": geometry,
            "electrostatic_root_sha256": self.electrostatic_state.root_sha256,
            "solvent_state_sha256": self.solvent_state.state_sha256,
            "topology_id": topology,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
        }
        expected = canonical_metadata_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match energy-state contents.")
        object.__setattr__(self, "configuration_sha256", configuration)
        object.__setattr__(self, "geometry_sha256", geometry)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
        object.__setattr__(self, "state_sha256", expected)

    @property
    def vacuum_energy_eV(self) -> float:
        return self.electrostatic_state.vacuum_energy_ev

    @property
    def polarization_energy_eV(self) -> float:
        return self.electrostatic_state.polarization_energy_ev

    @property
    def cds_energy_eV(self) -> float:
        return self.solvent_state.energy_eV

    @property
    def solvation_energy_eV(self) -> float:
        return self.polarization_energy_eV + self.cds_energy_eV

    @property
    def total_energy_eV(self) -> float:
        return self.vacuum_energy_eV + self.solvation_energy_eV

    @property
    def energy_sample(self) -> ScalarEnergySample:
        return ScalarEnergySample(
            energy_eV=self.total_energy_eV,
            state_sha256=self.state_sha256,
            topology_id=self.topology_id,
            topology_observation_coverage=self.topology_observation_coverage,
            unobservable_topology_components=self.unobservable_topology_components,
        )


@dataclass(frozen=True, slots=True)
class HybridSolvatedDDXForceEvaluation:
    """Block-adjoint hybrid force plus the independent solvent-term force."""

    central_state: HybridSolvatedDDXEnergyState
    electrostatic_evaluation: HybridDDXForceEvaluation
    solvent_gradient_state: SolventEnergyState
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.central_state, HybridSolvatedDDXEnergyState):
            raise TypeError("central_state must be HybridSolvatedDDXEnergyState.")
        if not isinstance(self.electrostatic_evaluation, HybridDDXForceEvaluation):
            raise TypeError(
                "electrostatic_evaluation must be HybridDDXForceEvaluation."
            )
        if not isinstance(self.solvent_gradient_state, SolventEnergyState):
            raise TypeError("solvent_gradient_state must be SolventEnergyState.")
        if (
            self.electrostatic_evaluation.central_state.root_sha256
            != self.central_state.electrostatic_state.root_sha256
        ):
            raise ValueError("electrostatic force root differs from central state.")
        reference = self.central_state.solvent_state
        gradient_state = self.solvent_gradient_state
        for name in (
            "provider_id",
            "configuration_sha256",
            "geometry_sha256",
            "topology_id",
            "topology_observation_coverage",
            "unobservable_topology_components",
        ):
            if getattr(gradient_state, name) != getattr(reference, name):
                raise ValueError("solvent gradient state identity did not replay.")
        if abs(gradient_state.energy_eV - reference.energy_eV) > ENERGY_REPLAY_ATOL_EV:
            raise ValueError("solvent energy did not replay for force evaluation.")
        gradient = gradient_state.gradient_eV_per_A
        if gradient is None:
            raise ValueError("solvent gradient state contains no gradient.")
        _readonly(
            gradient,
            shape=self.electrostatic_evaluation.total_forces_ev_per_angstrom.shape,
            name="solvent gradient",
        )
        payload = {
            "contract": "mace-mdp-polar-solvated-ddx-force-v1",
            "central_state_sha256": self.central_state.state_sha256,
            "electrostatic_evaluation_sha256": (
                self.electrostatic_evaluation.evaluation_sha256
            ),
            "solvent_gradient_state_sha256": gradient_state.state_sha256,
        }
        expected = canonical_metadata_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match force contents.")
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def solvent_forces_eV_per_A(self) -> np.ndarray:
        gradient = np.asarray(self.solvent_gradient_state.gradient_eV_per_A)
        result = np.ascontiguousarray(-gradient, dtype=np.float64)
        result.setflags(write=False)
        return result

    @property
    def total_forces_eV_per_A(self) -> np.ndarray:
        result = np.ascontiguousarray(
            self.electrostatic_evaluation.total_forces_ev_per_angstrom
            + self.solvent_forces_eV_per_A,
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @property
    def force_sample(self) -> ScalarForceSample:
        return ScalarForceSample(
            energy_sample=self.central_state.energy_sample,
            forces_eV_per_A=self.total_forces_eV_per_A,
            evaluation_sha256=self.evaluation_sha256,
        )


class MACE_MDPPolarHybridSolvatedDDXPES:
    """Composable full-solvation scalar with derivative access when complete.

    The solvent term cannot repair a missing electrostatic coordinate
    derivative.  Force-, virial-, HVP-, and Hessian-level entry points therefore
    inherit the electrostatic provider's explicit capability instead of
    advertising a class-wide constant.
    """

    __slots__ = (
        "_configuration_sha256",
        "_electrostatic_pes",
        "_hessian_backend",
        "_numerical_force_backend",
        "_sealed",
        "_solvent_term",
    )

    provider_id = HYBRID_SOLVATED_DDX_PES_PROVIDER_ID
    scalar_contract_id = HYBRID_SOLVATED_DDX_SCALAR_CONTRACT_ID
    daily_profile_id = HYBRID_SOLVATED_DDX_DAILY_PROFILE_ID
    scientific_status = (
        "experimental-callable-E; conditional-complete-coordinate-derivative-"
        "F-molecular-virial-HVP-H; "
        "hybrid-full-solvation-accuracy-and-release-admission-pending"
    )

    def __init__(
        self,
        *,
        electrostatic_pes: object,
        solvent_term: SolventEnergyTerm,
        numerical_force_backend: RichardsonScalarForce | None = None,
        hessian_backend: RichardsonScalarHessian | None = None,
    ) -> None:
        for owner, names in (
            (
                electrostatic_pes,
                (
                    "configuration_sha256",
                    "topology_id",
                    "solve",
                    "evaluate_forces",
                ),
            ),
            (solvent_term, ("configuration_sha256", "evaluate")),
        ):
            for name in names:
                if not callable(getattr(owner, name, None)):
                    raise TypeError(f"provider requires callable {name}().")
        numerical = numerical_force_backend or RichardsonScalarForce(
            coarse_step_angstrom=5.0e-4,
            maximum_error_eV_per_A=2.0e-4,
        )
        hessian = hessian_backend or RichardsonScalarHessian(
            maximum_topology_step_reductions=6,
        )
        if not isinstance(numerical, RichardsonScalarForce):
            raise TypeError("numerical_force_backend must be RichardsonScalarForce.")
        if not isinstance(hessian, RichardsonScalarHessian):
            raise TypeError("hessian_backend must be RichardsonScalarHessian.")
        electrostatic_pes.configuration_sha256()
        solvent_term.configuration_sha256()
        object.__setattr__(self, "_electrostatic_pes", electrostatic_pes)
        object.__setattr__(self, "_solvent_term", solvent_term)
        object.__setattr__(self, "_numerical_force_backend", numerical)
        object.__setattr__(self, "_hessian_backend", hessian)
        object.__setattr__(self, "_configuration_sha256", self._current_configuration())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("hybrid solvated ddX PES is immutable.")
        object.__setattr__(self, name, value)

    @property
    def hessian_backend(self) -> RichardsonScalarHessian:
        return self._hessian_backend

    @property
    def coordinate_derivative_available(self) -> bool:
        declared = getattr(
            self._electrostatic_pes,
            "coordinate_derivative_available",
            None,
        )
        if type(declared) is not bool:
            raise RuntimeError(
                "electrostatic PES must declare a boolean complete-coordinate-"
                "derivative capability."
            )
        return declared

    @property
    def force_available(self) -> bool:
        return self.coordinate_derivative_available

    def _current_configuration(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "mace-mdp-polar-solvated-ddx-operational-pes-v1",
                "provider_id": self.provider_id,
                "scalar_contract_id": self.scalar_contract_id,
                "electrostatic_pes_provider_id": getattr(
                    self._electrostatic_pes, "provider_id", None
                ),
                "electrostatic_configuration_sha256": (
                    self._electrostatic_pes.configuration_sha256()
                ),
                "solvent_term_provider_id": getattr(
                    self._solvent_term, "provider_id", None
                ),
                "solvent_term_configuration_sha256": (
                    self._solvent_term.configuration_sha256()
                ),
                "energy_ledger": "vacuum-plus-ddx-polarization-plus-solvent-term",
                "force_derivative_kind": (
                    "analytic-block-adjoint-plus-solvent-gradient-v1"
                ),
                "coordinate_derivative_available": (
                    self.coordinate_derivative_available
                ),
                "numerical_force_policy_sha256": (
                    self._numerical_force_backend.policy_sha256()
                ),
                "hessian_policy_sha256": self._hessian_backend.policy_sha256(),
                "implementation_sha256": hashlib.sha256(
                    _MODULE_PATH.read_bytes()
                ).hexdigest(),
                "capabilities": "experimental-callable-not-release-admitted",
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration()
        if current != self._configuration_sha256:
            raise RuntimeError("hybrid solvated ddX PES configuration drifted.")
        return current

    def solve(self, geometry: object) -> HybridSolvatedDDXEnergyState:
        configuration = self.configuration_sha256()
        electrostatic = self._electrostatic_pes.solve(geometry)
        if not isinstance(electrostatic, HybridDDXEnergyState):
            raise TypeError(
                "electrostatic PES solve() must return HybridDDXEnergyState."
            )
        solvent = self._solvent_term.evaluate(geometry, need_gradient=False)
        if not isinstance(solvent, SolventEnergyState):
            raise TypeError("solvent term evaluate() must return SolventEnergyState.")
        geometry_digest = geometry_sha256(geometry)
        if (
            electrostatic.geometry_sha256 != geometry_digest
            or solvent.geometry_sha256 != geometry_digest
        ):
            raise RuntimeError("component geometry identity did not replay.")
        electrostatic_topology = _text(
            self._electrostatic_pes.topology_id(geometry),
            name="electrostatic topology",
        )
        topology = canonical_metadata_sha256(
            {
                "electrostatic_topology_id": electrostatic_topology,
                "solvent_term_topology_id": solvent.topology_id,
            }
        )
        coverage = (
            "complete"
            if solvent.topology_observation_coverage == "complete"
            else "partial"
        )
        return HybridSolvatedDDXEnergyState(
            provider_id=self.provider_id,
            scalar_contract_id=self.scalar_contract_id,
            configuration_sha256=configuration,
            geometry_sha256=geometry_digest,
            electrostatic_state=electrostatic,
            solvent_state=solvent,
            topology_id=topology,
            topology_observation_coverage=coverage,
            unobservable_topology_components=(solvent.unobservable_topology_components),
        )

    def _validated_energy_state(
        self,
        geometry: object,
        state: HybridSolvatedDDXEnergyState,
    ) -> HybridSolvatedDDXEnergyState:
        if not isinstance(state, HybridSolvatedDDXEnergyState):
            raise TypeError("central_state must be HybridSolvatedDDXEnergyState.")
        replayed = self.solve(geometry)
        if replayed.state_sha256 != state.state_sha256:
            raise ValueError(
                "central solvated state did not replay for the current provider "
                "and geometry."
            )
        return replayed

    def sample(self, geometry: object) -> ScalarEnergySample:
        return self.solve(geometry).energy_sample

    def get_potential_energy(self, geometry: object) -> float:
        return self.solve(geometry).total_energy_eV

    def get_solvation_energy(self, geometry: object) -> float:
        return self.solve(geometry).solvation_energy_eV

    def evaluate_forces(
        self,
        geometry: object,
        *,
        central_state: HybridSolvatedDDXEnergyState | None = None,
    ) -> HybridSolvatedDDXForceEvaluation:
        if not self.coordinate_derivative_available:
            raise NotImplementedError(
                "electrostatic response has no complete coordinate derivative; "
                "the combined analytic force is unavailable."
            )
        state = (
            self.solve(geometry)
            if central_state is None
            else self._validated_energy_state(geometry, central_state)
        )
        electrostatic = self._electrostatic_pes.evaluate_forces(
            geometry,
            central_state=state.electrostatic_state,
        )
        if not isinstance(electrostatic, HybridDDXForceEvaluation):
            raise TypeError(
                "electrostatic PES evaluate_forces() must return "
                "HybridDDXForceEvaluation."
            )
        solvent = self._solvent_term.evaluate(geometry, need_gradient=True)
        return HybridSolvatedDDXForceEvaluation(
            central_state=state,
            electrostatic_evaluation=electrostatic,
            solvent_gradient_state=solvent,
        )

    def _validated_force_evaluation(
        self,
        geometry: object,
        evaluation: HybridSolvatedDDXForceEvaluation,
    ) -> HybridSolvatedDDXForceEvaluation:
        if not isinstance(evaluation, HybridSolvatedDDXForceEvaluation):
            raise TypeError(
                "force_evaluation must be HybridSolvatedDDXForceEvaluation."
            )
        replayed = self.evaluate_forces(
            geometry,
            central_state=evaluation.central_state,
        )
        if (
            replayed.evaluation_sha256 != evaluation.evaluation_sha256
            or not np.array_equal(
                replayed.total_forces_eV_per_A,
                evaluation.total_forces_eV_per_A,
            )
        ):
            raise ValueError(
                "cached force evaluation did not replay for the current provider "
                "and geometry."
            )
        return replayed

    def force_sample(self, geometry: object) -> ScalarForceSample:
        return self.evaluate_forces(geometry).force_sample

    def get_forces(self, geometry: object) -> np.ndarray:
        return np.array(self.evaluate_forces(geometry).total_forces_eV_per_A, copy=True)

    def numerical_force_audit(
        self,
        geometry: object,
        *,
        central_state: HybridSolvatedDDXEnergyState | None = None,
    ) -> RichardsonScalarForceEvaluation:
        state = (
            self.solve(geometry)
            if central_state is None
            else self._validated_energy_state(geometry, central_state)
        )
        return self._numerical_force_backend.evaluate(
            self,
            geometry,
            central_sample=state.energy_sample,
        )

    def molecular_virial(
        self,
        geometry: object,
        *,
        origin_angstrom: object | None = None,
        force_evaluation: HybridSolvatedDDXForceEvaluation | None = None,
    ) -> MolecularVirialEvaluation:
        evaluated = (
            self.evaluate_forces(geometry)
            if force_evaluation is None
            else self._validated_force_evaluation(geometry, force_evaluation)
        )
        return MolecularVirialEvaluation.from_conservative_force(
            force_evaluation_sha256=evaluated.evaluation_sha256,
            positions_angstrom=getattr(geometry, "positions", None),
            forces_eV_per_A=evaluated.total_forces_eV_per_A,
            origin_angstrom=origin_angstrom,
        )

    def hessian_vector_product(
        self,
        geometry: object,
        direction: object,
        *,
        central_force: HybridSolvatedDDXForceEvaluation | None = None,
    ) -> RichardsonScalarHVPEvaluation:
        evaluated = (
            self.evaluate_forces(geometry)
            if central_force is None
            else self._validated_force_evaluation(geometry, central_force)
        )
        return self._hessian_backend.evaluate_hvp(
            self,
            geometry,
            direction,
            central_sample=evaluated.force_sample,
        )

    def evaluate_hessian(
        self,
        geometry: object,
        *,
        central_force: HybridSolvatedDDXForceEvaluation | None = None,
    ) -> RichardsonScalarHessianEvaluation:
        evaluated = (
            self.evaluate_forces(geometry)
            if central_force is None
            else self._validated_force_evaluation(geometry, central_force)
        )
        return self._hessian_backend.evaluate(
            self,
            geometry,
            central_sample=evaluated.force_sample,
        )

    def get_hessian(self, geometry: object) -> np.ndarray:
        return np.array(
            self.evaluate_hessian(geometry).hessian_eV_per_A2,
            copy=True,
        )


def build_smd_mace_mdp_polar_hybrid_ddx_pes(
    hybrid: PermanentAnchoredInducedSourceModel,
    symbols: tuple[str, ...] | list[str],
    *,
    solvent: str,
    continuum_model: str = "pcm",
    lmax: int = 15,
    n_lebedev: int = 1202,
    solver_tolerance: float = 1.0e-12,
    eta: float = 0.1,
    n_proc: int = 1,
    numerical_force_backend: RichardsonScalarForce | None = None,
    hessian_backend: RichardsonScalarHessian | None = None,
) -> MACE_MDPPolarHybridSolvatedDDXPES:
    """Build the named-solvent hybrid ddX + official PySCF SMD-CDS scalar."""

    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec

    normalized = tuple(symbols)
    specification = route2_solvent_spec(solvent)
    electrostatic = MACE_MDPPolarHybridDDXPES(
        hybrid=hybrid,
        symbols=normalized,
        cavity_radii_angstrom=smd_coulomb_radii(
            normalized,
            solvent=specification.name,
        ),
        continuum_model=continuum_model,
        dielectric=specification.descriptors.dielectric,
        lmax=lmax,
        n_lebedev=n_lebedev,
        solver_tolerance=solver_tolerance,
        eta=eta,
        n_proc=n_proc,
    )
    return MACE_MDPPolarHybridSolvatedDDXPES(
        electrostatic_pes=electrostatic,
        solvent_term=PySCFSMDCDSTerm(normalized, specification.name),
        numerical_force_backend=numerical_force_backend,
        hessian_backend=hessian_backend,
    )


__all__ = [
    "HYBRID_SOLVATED_DDX_DAILY_PROFILE_ID",
    "HYBRID_SOLVATED_DDX_PES_PROVIDER_ID",
    "HYBRID_SOLVATED_DDX_SCALAR_CONTRACT_ID",
    "HybridSolvatedDDXEnergyState",
    "HybridSolvatedDDXForceEvaluation",
    "MACE_MDPPolarHybridSolvatedDDXPES",
    "build_smd_mace_mdp_polar_hybrid_ddx_pes",
]
