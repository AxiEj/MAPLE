"""Pure MACE-POLAR frozen-source ddX/SMD scalar with E/F/virial/HVP/H access.

This module intentionally does not build a MACE-MDP permanent source and does
not solve a coupled electronic/continuum fixed point.  It defines one explicit
geometry scalar

``E(R) = E_vac^MACE-POLAR(R) + G_ddX(R, c0(R)) + G_CDS(R)``,

where ``c0`` is the unmodified zero-field MACE-POLAR source.  Its force is the
complete chain rule of that scalar,

``F = F_vac - partial_R G_ddX - (dc0/dR)^T partial_c G_ddX - dG_CDS/dR``.

Molecular virials follow from the same forces.  HVPs and full Hessians are
obtained by error-estimated Richardson differentiation of those conservative
forces.  Callable experimental derivatives are deliberately separated from
release admission: accuracy, rotational covariance, and broad Hessian panels
must still bind this exact model/evaluator/continuum/CDS identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    SASA_GRID_POINTS,
    smd_water_cds_fibonacci_swig_inspired,
    smd_water_cds_fibonacci_swig_inspired_position_gradient,
    validate_smd_symbols,
)
from maple.solvation.api.units import HARTREE_TO_EV
from maple.solvation.api.scalar_registry import (
    EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
)
from maple.solvation.coupling.operator import (
    canonical_metadata_sha256,
    source_files_sha256,
)
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
from maple.solvation.models.base import (
    atom_count,
    model_charge_and_multiplicity,
    model_input_sha256,
)
from maple.solvation.solvent_terms import (
    PySCFSMDCDSTerm,
    SolventEnergyState,
    SolventEnergyTerm,
)

PURE_FROZEN_DDX_PROVIDER_ID = (
    "maple.route2.experimental.mace-polar-frozen-source-ddx-smd.impl.v1"
)
PURE_FROZEN_DDX_SCALAR_CONTRACT_ID = (
    "mace-polar-zero-field-frozen-source-ddx-plus-differentiable-cds-v1"
)
SMOOTH_SMD_WATER_CDS_PROVIDER_ID = (
    "maple.route2.experimental.smd-water-fibonacci-swig-inspired-cds.impl.v1"
)
SCIENTIFIC_STATUS = (
    "experimental-callable-E-F-molecular-virial-HVP-H; "
    "accuracy-evidence-is-scalar-profile-specific; release-admission-pending"
)
SOURCE_CHARGE_ATOL_E = 1.0e-8
ENERGY_REPLAY_ATOL_EV = 2.0e-10
_MODULE_PATH = Path(__file__).resolve()
_MAPLE_ROOT = _MODULE_PATH.parents[2]
_SMD_CDS_PATH = (
    _MAPLE_ROOT
    / "function"
    / "calculator"
    / "extra_correction"
    / "implicit"
    / "smd_cds.py"
)


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


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _readonly(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape(shape)


def _symbols(geometry: object) -> tuple[str, ...]:
    getter = getattr(geometry, "get_chemical_symbols", None)
    if not callable(getter):
        raise TypeError("geometry must expose callable get_chemical_symbols().")
    values = tuple(str(value) for value in getter())
    if not values or len(values) != atom_count(geometry):
        raise ValueError("geometry chemical symbols are invalid.")
    return values


def _positions(geometry: object) -> np.ndarray:
    getter = getattr(geometry, "get_positions", None)
    values = getter() if callable(getter) else getattr(geometry, "positions", None)
    result = np.asarray(values, dtype=float)
    expected = (atom_count(geometry), 3)
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"geometry positions must be finite with shape {expected}.")
    return np.array(result, copy=True)


def _registered_point_profile_bindings_match(
    continuum: object,
    solvent_term: object,
) -> bool:
    """Return whether providers realize the exact registered point profile."""

    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.mace_polar_point_ddx import (
        MACEPolarPointEmbeddedDDXBackend,
    )

    if (
        type(continuum) is not MACEPolarPointEmbeddedDDXBackend
        or type(solvent_term) is not PySCFSMDCDSTerm
    ):
        return False
    try:
        specification = route2_solvent_spec(solvent_term.solvent)
        radial = continuum._separated.radial_backend
        expected_radii = smd_coulomb_radii(
            continuum.symbols,
            solvent=specification.name,
        )
        return bool(
            continuum.symbols == solvent_term.symbols
            and np.array_equal(continuum.cavity_radii_angstrom, expected_radii)
            and radial._continuum_model == "pcm"
            and radial._dielectric == specification.descriptors.dielectric
            and radial._lmax == 15
            and radial._n_lebedev == 1202
            and radial._solver_tolerance == 1.0e-12
            and radial._eta == 0.1
            and radial._n_proc == 1
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _registered_point_model_bindings_match(model: object) -> bool:
    """Return whether ``model`` is the exact evidenced official CPU adapter."""

    from maple.solvation.models.mace_polar import (
        MACEPolarRadialGTOModelAdapter,
        OFFICIAL_MACE_POLAR_1_M_CONTRACT,
    )

    if type(model) is not MACEPolarRadialGTOModelAdapter:
        return False
    try:
        provenance = model.provenance
        domain = model.domain
        release = model.release_contract
        expected_provider = (
            f"{OFFICIAL_MACE_POLAR_1_M_CONTRACT.provider_id}.radial-gto.v1"
        )
        return bool(
            release == OFFICIAL_MACE_POLAR_1_M_CONTRACT
            and model.provider_id == expected_provider
            and model.model_profile_id
            == OFFICIAL_MACE_POLAR_1_M_CONTRACT.model_profile_id
            and model.dtype == "float64"
            and model.device == "cpu"
            and provenance.provider_id == model.provider_id
            and provenance.model_profile_id == model.model_profile_id
            and provenance.model_family == "MACE-POLAR-1-radial-GTO-response"
            and provenance.checkpoint_sha256
            == OFFICIAL_MACE_POLAR_1_M_CONTRACT.checkpoint_sha256
            and provenance.upstream_version
            == (
                "mace-torch=="
                f"{OFFICIAL_MACE_POLAR_1_M_CONTRACT.mace_torch_version};"
                "graph-longrange=="
                f"{OFFICIAL_MACE_POLAR_1_M_CONTRACT.graph_longrange_version}"
            )
            and provenance.upstream_commit
            == OFFICIAL_MACE_POLAR_1_M_CONTRACT.upstream_commit
            and model.provenance_sha256 == provenance.sha256
            and provenance.dtype == model.dtype
            and provenance.device == model.device
            and provenance.domain == domain
            and model.field_convention == provenance.field_convention
            and model.coordinate_frame_policy == provenance.coordinate_frame_policy
            and tuple(domain.atomic_numbers) == tuple(range(1, 84))
            and tuple(domain.total_charge_range) == (0, 0)
            and tuple(domain.spin_multiplicities) == (1,)
            and model.configuration_sha256()
        )
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class SmoothSMDWaterCDS:
    """Differentiable SMD-water CDS candidate using the existing smooth grid."""

    symbols: tuple[str, ...]
    grid_points: int = SASA_GRID_POINTS
    provider_id: str = SMOOTH_SMD_WATER_CDS_PROVIDER_ID

    def __post_init__(self) -> None:
        symbols = validate_smd_symbols(tuple(self.symbols))
        if type(self.grid_points) is not int or self.grid_points < 12:
            raise ValueError("grid_points must be an integer of at least 12.")
        if self.provider_id != SMOOTH_SMD_WATER_CDS_PROVIDER_ID:
            raise ValueError("Smooth SMD-water CDS provider identity is fixed.")
        object.__setattr__(self, "symbols", symbols)

    def configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": "smd-water-fibonacci-swig-inspired-cds-v1",
                "provider_id": self.provider_id,
                "symbols": list(self.symbols),
                "grid_points": self.grid_points,
                "unit_conversion_hartree_to_eV": HARTREE_TO_EV,
                "implementation_files_sha256": dict(
                    source_files_sha256(
                        {
                            "experimental/mace_polar_frozen_ddx.py": _MODULE_PATH,
                            "function/calculator/extra_correction/implicit/smd_cds.py": (
                                _SMD_CDS_PATH
                            ),
                        }
                    )
                ),
                "scientific_status": (
                    "smooth-force-candidate; not exact published Lebedev-SWIG"
                ),
            }
        )

    def evaluate(self, geometry: object, *, need_gradient: bool) -> SolventEnergyState:
        if type(need_gradient) is not bool:
            raise TypeError("need_gradient must be a bool.")
        if _symbols(geometry) != self.symbols:
            raise ValueError("geometry symbols differ from the CDS configuration.")
        positions = _positions(geometry)
        result = smd_water_cds_fibonacci_swig_inspired(
            self.symbols, positions, grid_points=self.grid_points
        )
        gradient = None
        if need_gradient:
            gradient = HARTREE_TO_EV * (
                smd_water_cds_fibonacci_swig_inspired_position_gradient(
                    self.symbols,
                    positions,
                    grid_points=self.grid_points,
                )
            )
        return SolventEnergyState(
            provider_id=self.provider_id,
            configuration_sha256=self.configuration_sha256(),
            geometry_sha256=geometry_sha256(geometry),
            topology_id=f"smooth-fibonacci-grid-{self.grid_points}",
            atom_count=len(self.symbols),
            energy_eV=float(result.energy_hartree * HARTREE_TO_EV),
            gradient_eV_per_A=gradient,
            topology_observation_coverage="complete",
            unobservable_topology_components=(),
        )


@dataclass(frozen=True, slots=True)
class MACEPolarFrozenDDXEnergyState:
    """Immutable energy ledger for one pure frozen-source geometry."""

    provider_id: str
    scalar_contract_id: str
    configuration_sha256: str
    geometry_sha256: str
    model_input_sha256: str
    model_provider_id: str
    model_provenance_sha256: str
    continuum_provider_id: str
    continuum_state_sha256: str
    solvent_term_provider_id: str
    solvent_term_state_sha256: str
    topology_id: str
    atom_count: int
    source_values: np.ndarray
    source_total_charge_e: float
    vacuum_energy_eV: float
    polarization_energy_eV: float
    cds_energy_eV: float
    topology_observation_coverage: str = "unobservable"
    unobservable_topology_components: tuple[str, ...] = (
        "legacy-unspecified-topology-observation",
    )
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if self.provider_id != PURE_FROZEN_DDX_PROVIDER_ID:
            raise ValueError("pure frozen-source provider identity is invalid.")
        scalar_contract = _text(
            self.scalar_contract_id,
            name="scalar_contract_id",
        )
        if scalar_contract not in (
            PURE_FROZEN_DDX_SCALAR_CONTRACT_ID,
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
        ):
            raise ValueError("pure frozen-source scalar contract is invalid.")
        for name in (
            "configuration_sha256",
            "geometry_sha256",
            "model_input_sha256",
            "model_provenance_sha256",
            "continuum_state_sha256",
            "solvent_term_state_sha256",
        ):
            _digest(getattr(self, name), name=name)
        for name in (
            "model_provider_id",
            "continuum_provider_id",
            "solvent_term_provider_id",
            "topology_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        coverage, components = normalize_topology_observation(
            self.topology_observation_coverage,
            self.unobservable_topology_components,
        )
        if type(self.atom_count) is not int or self.atom_count < 1:
            raise ValueError("atom_count must be a positive integer.")
        source = _readonly(
            self.source_values,
            shape=(self.atom_count, 8),
            name="source_values",
        )
        charge = _finite(self.source_total_charge_e, name="source_total_charge_e")
        vacuum = _finite(self.vacuum_energy_eV, name="vacuum_energy_eV")
        polarization = _finite(
            self.polarization_energy_eV, name="polarization_energy_eV"
        )
        cds = _finite(self.cds_energy_eV, name="cds_energy_eV")
        payload = {
            "contract": "mace-polar-frozen-ddx-energy-state-v2",
            "provider_id": self.provider_id,
            "scalar_contract_id": scalar_contract,
            "configuration_sha256": self.configuration_sha256,
            "geometry_sha256": self.geometry_sha256,
            "model_input_sha256": self.model_input_sha256,
            "model_provider_id": self.model_provider_id,
            "model_provenance_sha256": self.model_provenance_sha256,
            "continuum_provider_id": self.continuum_provider_id,
            "continuum_state_sha256": self.continuum_state_sha256,
            "solvent_term_provider_id": self.solvent_term_provider_id,
            "solvent_term_state_sha256": self.solvent_term_state_sha256,
            "topology_id": self.topology_id,
            "topology_observation_coverage": coverage,
            "unobservable_topology_components": list(components),
            "atom_count": self.atom_count,
            "source_values": source.tolist(),
            "source_total_charge_e": charge,
            "vacuum_energy_eV": vacuum,
            "polarization_energy_eV": polarization,
            "cds_energy_eV": cds,
        }
        expected = canonical_metadata_sha256(payload)
        if self.state_sha256 and self.state_sha256 != expected:
            raise ValueError("state_sha256 does not match energy-state contents.")
        object.__setattr__(self, "source_values", source)
        object.__setattr__(self, "scalar_contract_id", scalar_contract)
        object.__setattr__(self, "topology_observation_coverage", coverage)
        object.__setattr__(self, "unobservable_topology_components", components)
        object.__setattr__(self, "source_total_charge_e", charge)
        object.__setattr__(self, "vacuum_energy_eV", vacuum)
        object.__setattr__(self, "polarization_energy_eV", polarization)
        object.__setattr__(self, "cds_energy_eV", cds)
        object.__setattr__(self, "state_sha256", expected)

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
            unobservable_topology_components=(self.unobservable_topology_components),
        )


@dataclass(frozen=True, slots=True)
class MACEPolarFrozenDDXForceEvaluation:
    """Analytic chain-rule force leaves for the frozen-source scalar."""

    central_state: MACEPolarFrozenDDXEnergyState
    vacuum_forces_eV_per_A: np.ndarray
    continuum_fixed_source_forces_eV_per_A: np.ndarray
    source_geometry_forces_eV_per_A: np.ndarray
    cds_forces_eV_per_A: np.ndarray
    evaluation_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.central_state, MACEPolarFrozenDDXEnergyState):
            raise TypeError("central_state must be MACEPolarFrozenDDXEnergyState.")
        shape = (self.central_state.atom_count, 3)
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "vacuum_forces_eV_per_A",
            "continuum_fixed_source_forces_eV_per_A",
            "source_geometry_forces_eV_per_A",
            "cds_forces_eV_per_A",
        ):
            arrays[name] = _readonly(getattr(self, name), shape=shape, name=name)
        payload = {
            "contract": "mace-polar-frozen-ddx-chain-rule-force-v1",
            "central_state_sha256": self.central_state.state_sha256,
            **{name: values.tolist() for name, values in arrays.items()},
        }
        expected = canonical_metadata_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match force contents.")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "evaluation_sha256", expected)

    @property
    def total_forces_eV_per_A(self) -> np.ndarray:
        return np.add.reduce(
            (
                self.vacuum_forces_eV_per_A,
                self.continuum_fixed_source_forces_eV_per_A,
                self.source_geometry_forces_eV_per_A,
                self.cds_forces_eV_per_A,
            )
        )

    @property
    def force_sample(self) -> ScalarForceSample:
        return ScalarForceSample(
            energy_sample=self.central_state.energy_sample,
            forces_eV_per_A=self.total_forces_eV_per_A,
            evaluation_sha256=self.evaluation_sha256,
        )


@dataclass(frozen=True, slots=True)
class MolecularVirialEvaluation:
    """Origin-declared molecular virial from the conservative force."""

    force_evaluation_sha256: str
    origin_angstrom: np.ndarray
    net_force_eV_per_A: np.ndarray
    raw_virial_eV: np.ndarray
    symmetric_virial_eV: np.ndarray
    strain_gradient_eV: np.ndarray
    maximum_antisymmetry_eV: float
    evaluation_sha256: str = ""

    @classmethod
    def from_conservative_force(
        cls,
        *,
        force_evaluation_sha256: str,
        positions_angstrom: object,
        forces_eV_per_A: object,
        origin_angstrom: object | None = None,
    ) -> "MolecularVirialEvaluation":
        positions = np.asarray(positions_angstrom, dtype=float)
        forces = np.asarray(forces_eV_per_A, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or forces.shape != positions.shape
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(forces))
        ):
            raise ValueError(
                "positions_angstrom and forces_eV_per_A must be finite "
                "matching (N,3) arrays."
            )
        origin = (
            np.mean(positions, axis=0)
            if origin_angstrom is None
            else np.asarray(origin_angstrom, dtype=float)
        )
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin_angstrom must be finite with shape (3,).")
        raw = forces.T @ (positions - origin)
        return cls(
            force_evaluation_sha256=force_evaluation_sha256,
            origin_angstrom=origin,
            net_force_eV_per_A=np.sum(forces, axis=0),
            raw_virial_eV=raw,
            symmetric_virial_eV=0.5 * (raw + raw.T),
            strain_gradient_eV=-raw,
            maximum_antisymmetry_eV=float(np.max(np.abs(raw - raw.T))),
        )

    def __post_init__(self) -> None:
        _digest(self.force_evaluation_sha256, name="force_evaluation_sha256")
        origin = _readonly(self.origin_angstrom, shape=(3,), name="origin_angstrom")
        net = _readonly(self.net_force_eV_per_A, shape=(3,), name="net_force_eV_per_A")
        raw = _readonly(self.raw_virial_eV, shape=(3, 3), name="raw_virial_eV")
        symmetric = _readonly(
            self.symmetric_virial_eV, shape=(3, 3), name="symmetric_virial_eV"
        )
        strain = _readonly(
            self.strain_gradient_eV, shape=(3, 3), name="strain_gradient_eV"
        )
        if not np.array_equal(symmetric, 0.5 * (raw + raw.T)):
            raise ValueError("symmetric_virial_eV must symmetrize raw_virial_eV.")
        if not np.array_equal(strain, -raw):
            raise ValueError("strain_gradient_eV must equal -raw_virial_eV.")
        antisymmetry = _finite(
            self.maximum_antisymmetry_eV, name="maximum_antisymmetry_eV"
        )
        if antisymmetry != float(np.max(np.abs(raw - raw.T))):
            raise ValueError("maximum_antisymmetry_eV is inconsistent.")
        payload = {
            "contract": "molecular-virial-from-conservative-force-v1",
            "force_evaluation_sha256": self.force_evaluation_sha256,
            "origin_angstrom": origin.tolist(),
            "net_force_eV_per_A": net.tolist(),
            "raw_virial_eV": raw.tolist(),
            "symmetric_virial_eV": symmetric.tolist(),
            "strain_gradient_eV": strain.tolist(),
            "maximum_antisymmetry_eV": antisymmetry,
        }
        expected = canonical_metadata_sha256(payload)
        if self.evaluation_sha256 and self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not match virial contents.")
        object.__setattr__(self, "origin_angstrom", origin)
        object.__setattr__(self, "net_force_eV_per_A", net)
        object.__setattr__(self, "raw_virial_eV", raw)
        object.__setattr__(self, "symmetric_virial_eV", symmetric)
        object.__setattr__(self, "strain_gradient_eV", strain)
        object.__setattr__(self, "maximum_antisymmetry_eV", antisymmetry)
        object.__setattr__(self, "evaluation_sha256", expected)


class MACEPolarFrozenSourceDDXPES:
    """Callable pure MACE-POLAR frozen-source ddX/SMD potential surface."""

    __slots__ = (
        "_configuration_sha256",
        "_continuum",
        "_hessian_backend",
        "_model",
        "_numerical_force_backend",
        "_sealed",
        "_scalar_contract_id",
        "_solvent_term",
        "_symbols",
    )

    provider_id = PURE_FROZEN_DDX_PROVIDER_ID
    scientific_status = SCIENTIFIC_STATUS

    def __init__(
        self,
        *,
        model: object,
        continuum: object,
        solvent_term: SolventEnergyTerm,
        scalar_contract_id: str = PURE_FROZEN_DDX_SCALAR_CONTRACT_ID,
        numerical_force_backend: RichardsonScalarForce | None = None,
        hessian_backend: RichardsonScalarHessian | None = None,
    ) -> None:
        provenance = getattr(model, "provenance", None)
        model_family = str(getattr(provenance, "model_family", ""))
        if not model_family.startswith("MACE-POLAR") or "MACE-MDP" in model_family:
            raise TypeError(
                "model must be a pure MACE-POLAR radial-GTO source provider."
            )
        for owner, names in (
            (
                model,
                (
                    "configuration_sha256",
                    "evaluate_vacuum",
                    "evaluate_source",
                    "source_position_vjp",
                ),
            ),
            (
                continuum,
                (
                    "configuration_sha256",
                    "build_state",
                    "build_state_with_fixed_source_coordinate_gradient",
                ),
            ),
            (solvent_term, ("configuration_sha256", "evaluate")),
        ):
            for name in names:
                if not callable(getattr(owner, name, None)):
                    raise TypeError(f"provider requires callable {name}().")
        symbols = tuple(getattr(continuum, "symbols", ()))
        if not symbols:
            raise ValueError("continuum must bind a non-empty symbol sequence.")
        if getattr(model, "coupling_id", None) != getattr(
            continuum, "coupling_id", None
        ):
            raise ValueError("model and continuum coupling identities differ.")
        for name in ("source_space", "field_space"):
            model_space = getattr(model, name, None)
            continuum_space = getattr(continuum, name, None)
            if not callable(
                getattr(model_space, "metadata_hash", None)
            ) or not callable(getattr(continuum_space, "metadata_hash", None)):
                raise TypeError(f"model and continuum require fingerprintable {name}.")
            if model_space.metadata_hash() != continuum_space.metadata_hash():
                raise ValueError(f"model and continuum {name} identities differ.")
        numerical = numerical_force_backend or RichardsonScalarForce(
            coarse_step_angstrom=5.0e-4,
            maximum_error_eV_per_A=2.0e-4,
        )
        hessian = hessian_backend or RichardsonScalarHessian()
        if not isinstance(numerical, RichardsonScalarForce):
            raise TypeError("numerical_force_backend must be RichardsonScalarForce.")
        if not isinstance(hessian, RichardsonScalarHessian):
            raise TypeError("hessian_backend must be RichardsonScalarHessian.")
        scalar_contract = _text(scalar_contract_id, name="scalar_contract_id")
        if scalar_contract not in (
            PURE_FROZEN_DDX_SCALAR_CONTRACT_ID,
            EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1,
        ):
            raise ValueError("scalar_contract_id is not supported by this PES.")
        if (
            scalar_contract == EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
            and not _registered_point_model_bindings_match(model)
        ):
            raise ValueError(
                "the registered point-l1 scalar requires the exact official "
                "MACE-POLAR-1-M radial-GTO CPU/float64 model binding."
            )
        if (
            scalar_contract == EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
            and not _registered_point_profile_bindings_match(
                continuum,
                solvent_term,
            )
        ):
            raise ValueError(
                "the registered point-l1 scalar requires the exact point-ddPCM "
                "lmax=15/n_lebedev=1202/eta=0.1 and PySCF SMD-CDS bindings."
            )
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_continuum", continuum)
        object.__setattr__(self, "_solvent_term", solvent_term)
        object.__setattr__(self, "_scalar_contract_id", scalar_contract)
        object.__setattr__(self, "_symbols", symbols)
        object.__setattr__(self, "_numerical_force_backend", numerical)
        object.__setattr__(self, "_hessian_backend", hessian)
        object.__setattr__(
            self, "_configuration_sha256", self._current_configuration_sha256()
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MACEPolarFrozenSourceDDXPES is immutable.")
        object.__setattr__(self, name, value)

    @property
    def model(self) -> object:
        return self._model

    @property
    def continuum(self) -> object:
        return self._continuum

    @property
    def solvent_term(self) -> SolventEnergyTerm:
        return self._solvent_term

    @property
    def scalar_contract_id(self) -> str:
        return self._scalar_contract_id

    def _current_configuration_sha256(self) -> str:
        return canonical_metadata_sha256(
            {
                "contract": self.scalar_contract_id,
                "provider_id": self.provider_id,
                "model_provider_id": getattr(self._model, "provider_id", None),
                "model_configuration_sha256": self._model.configuration_sha256(),
                "model_provenance_sha256": getattr(
                    self._model, "provenance_sha256", None
                ),
                "continuum_provider_id": getattr(self._continuum, "provider_id", None),
                "continuum_configuration_sha256": (
                    self._continuum.configuration_sha256()
                ),
                "solvent_term_provider_id": self._solvent_term.provider_id,
                "solvent_term_configuration_sha256": (
                    self._solvent_term.configuration_sha256()
                ),
                "symbols": list(self._symbols),
                "source_charge_atol_e": SOURCE_CHARGE_ATOL_E,
                "energy_replay_atol_eV": ENERGY_REPLAY_ATOL_EV,
                "ledger": (
                    "vacuum-MACE-POLAR + frozen-zero-field-source-ddX + "
                    "additive-differentiable-solvent-term"
                ),
                "derivative": "complete-explicit-chain-rule-v1",
                "scientific_status": self.scientific_status,
                "implementation_files_sha256": dict(
                    source_files_sha256(
                        {"experimental/mace_polar_frozen_ddx.py": _MODULE_PATH}
                    )
                ),
            }
        )

    def configuration_sha256(self) -> str:
        current = self._current_configuration_sha256()
        if current != self._configuration_sha256:
            raise RuntimeError("pure frozen-source PES configuration drifted.")
        return current

    def _validate_geometry(self, geometry: object) -> int:
        count = atom_count(geometry)
        if count != len(self._symbols) or _symbols(geometry) != self._symbols:
            raise ValueError("geometry differs from the configured continuum symbols.")
        return count

    def _validated_energy_state(
        self,
        geometry: object,
        state: MACEPolarFrozenDDXEnergyState,
    ) -> MACEPolarFrozenDDXEnergyState:
        if not isinstance(state, MACEPolarFrozenDDXEnergyState):
            raise TypeError("central_state must be MACEPolarFrozenDDXEnergyState.")
        if state.configuration_sha256 != self.configuration_sha256():
            raise ValueError("central_state belongs to a different PES configuration.")
        if state.scalar_contract_id != self.scalar_contract_id:
            raise ValueError("central_state belongs to a different scalar contract.")
        if state.geometry_sha256 != geometry_sha256(geometry):
            raise ValueError("central_state is bound to a different geometry.")
        replayed = self.solve(geometry)
        if replayed.state_sha256 != state.state_sha256:
            raise ValueError(
                "cached central energy state did not replay for the current "
                "provider and geometry."
            )
        return replayed

    def _validated_force_evaluation(
        self,
        geometry: object,
        evaluation: MACEPolarFrozenDDXForceEvaluation,
    ) -> MACEPolarFrozenDDXForceEvaluation:
        if not isinstance(evaluation, MACEPolarFrozenDDXForceEvaluation):
            raise TypeError(
                "force_evaluation must be MACEPolarFrozenDDXForceEvaluation."
            )
        state = self._validated_energy_state(geometry, evaluation.central_state)
        replayed = self.evaluate_forces(geometry, central_state=state)
        if (
            replayed.evaluation_sha256 != evaluation.evaluation_sha256
            or not np.array_equal(
                replayed.total_forces_eV_per_A,
                evaluation.total_forces_eV_per_A,
            )
        ):
            raise ValueError(
                "cached force evaluation did not replay for the current "
                "provider and geometry."
            )
        return replayed

    def _zero_field(self, count: int) -> np.ndarray:
        raw_count = getattr(self._model.field_space, "component_count", None)
        components = (
            int(raw_count)
            if raw_count is not None
            else len(tuple(getattr(self._model.field_space, "components", ())))
        )
        if components != 8:
            raise RuntimeError("pure frozen-source route requires eight radial fields.")
        return np.zeros((count, components), dtype=float)

    def _source_charge(self, source: np.ndarray, count: int) -> float:
        total_charge = getattr(self._model.source_space, "total_charge", None)
        if not callable(total_charge):
            raise TypeError("model source space must expose total_charge().")
        return float(total_charge(source, atom_count=count))

    def solve(self, geometry: object) -> MACEPolarFrozenDDXEnergyState:
        self.configuration_sha256()
        count = self._validate_geometry(geometry)
        zero = self._zero_field(count)
        vacuum = self._model.evaluate_vacuum(geometry, need_forces=False)
        source_state = self._model.evaluate_source(
            geometry, zero, need_fixed_field_forces=False
        )
        source = np.asarray(getattr(source_state, "source", None), dtype=float)
        if source.shape != (count, 8) or not np.all(np.isfinite(source)):
            raise RuntimeError("MACE-POLAR zero-field source is invalid.")
        source_charge = self._source_charge(source, count)
        declared_charge, _multiplicity = model_charge_and_multiplicity(geometry)
        if abs(source_charge - declared_charge) > SOURCE_CHARGE_ATOL_E:
            raise RuntimeError(
                "MACE-POLAR zero-field source violates the declared total charge: "
                f"{source_charge:.6e} vs {declared_charge:d} e."
            )
        continuum_state = self._continuum.build_state(geometry, source)
        continuum_source = np.asarray(
            getattr(continuum_state, "source", None), dtype=float
        )
        if not np.array_equal(continuum_source, source):
            raise RuntimeError("ddX state did not retain the exact zero-field source.")
        solvent_state = self._solvent_term.evaluate(geometry, need_gradient=False)
        if not isinstance(solvent_state, SolventEnergyState):
            raise TypeError("solvent_term.evaluate() must return SolventEnergyState.")
        continuum_topology = getattr(continuum_state, "cavity_topology_sha256", None)
        if not isinstance(continuum_topology, str) or not continuum_topology.strip():
            raise RuntimeError("ddX state did not expose its cavity topology identity.")
        topology_id = canonical_metadata_sha256(
            {
                "ddx_cavity_topology_sha256": continuum_topology,
                "solvent_term_topology_id": solvent_state.topology_id,
            }
        )
        solvent_coverage = solvent_state.topology_observation_coverage
        topology_coverage = "complete" if solvent_coverage == "complete" else "partial"
        unobservable_components = solvent_state.unobservable_topology_components
        return MACEPolarFrozenDDXEnergyState(
            provider_id=self.provider_id,
            scalar_contract_id=self.scalar_contract_id,
            configuration_sha256=self.configuration_sha256(),
            geometry_sha256=geometry_sha256(geometry),
            model_input_sha256=model_input_sha256(geometry),
            model_provider_id=getattr(self._model, "provider_id"),
            model_provenance_sha256=getattr(self._model, "provenance_sha256"),
            continuum_provider_id=getattr(self._continuum, "provider_id"),
            continuum_state_sha256=getattr(continuum_state, "state_hash"),
            solvent_term_provider_id=solvent_state.provider_id,
            solvent_term_state_sha256=solvent_state.state_sha256,
            topology_id=topology_id,
            atom_count=count,
            source_values=source,
            source_total_charge_e=source_charge,
            vacuum_energy_eV=float(getattr(vacuum, "energy_eV")),
            polarization_energy_eV=float(
                getattr(continuum_state, "polarization_energy_ev")
            ),
            cds_energy_eV=solvent_state.energy_eV,
            topology_observation_coverage=topology_coverage,
            unobservable_topology_components=unobservable_components,
        )

    def sample(self, geometry: object) -> ScalarEnergySample:
        return self.solve(geometry).energy_sample

    def energy_state(self, geometry: object) -> MACEPolarFrozenDDXEnergyState:
        return self.solve(geometry)

    def get_potential_energy(self, geometry: object) -> float:
        return self.solve(geometry).total_energy_eV

    def get_solvation_energy(self, geometry: object) -> float:
        return self.solve(geometry).solvation_energy_eV

    def evaluate_forces(
        self,
        geometry: object,
        *,
        central_state: MACEPolarFrozenDDXEnergyState | None = None,
    ) -> MACEPolarFrozenDDXForceEvaluation:
        self.configuration_sha256()
        count = self._validate_geometry(geometry)
        state = (
            self.solve(geometry)
            if central_state is None
            else self._validated_energy_state(geometry, central_state)
        )
        zero = self._zero_field(count)
        vacuum = self._model.evaluate_vacuum(geometry, need_forces=True)
        vacuum_forces = np.asarray(
            getattr(vacuum, "forces_eV_per_A", None), dtype=float
        )
        if vacuum_forces.shape != (count, 3) or not np.all(np.isfinite(vacuum_forces)):
            raise RuntimeError("MACE-POLAR vacuum forces are unavailable or invalid.")
        if abs(float(getattr(vacuum, "energy_eV")) - state.vacuum_energy_eV) > (
            ENERGY_REPLAY_ATOL_EV
        ):
            raise RuntimeError("MACE-POLAR vacuum energy did not replay for force.")
        continuum_state, fixed_gradient = (
            self._continuum.build_state_with_fixed_source_coordinate_gradient(
                geometry, state.source_values
            )
        )
        if getattr(continuum_state, "state_hash") != state.continuum_state_sha256:
            raise RuntimeError("ddX state did not replay for force evaluation.")
        reaction_field = np.asarray(
            getattr(continuum_state, "reaction_field", None), dtype=float
        )
        source_gradient = np.asarray(
            self._model.source_position_vjp(geometry, zero, reaction_field),
            dtype=float,
        )
        solvent_state = self._solvent_term.evaluate(geometry, need_gradient=True)
        if (
            solvent_state.provider_id != state.solvent_term_provider_id
            or solvent_state.geometry_sha256 != state.geometry_sha256
        ):
            raise RuntimeError("additive solvent state identity did not replay.")
        if abs(solvent_state.energy_eV - state.cds_energy_eV) > ENERGY_REPLAY_ATOL_EV:
            raise RuntimeError("additive solvent energy did not replay for force.")
        cds_gradient = solvent_state.gradient_eV_per_A
        if cds_gradient is None:
            raise RuntimeError("solvent term did not provide its requested gradient.")
        return MACEPolarFrozenDDXForceEvaluation(
            central_state=state,
            vacuum_forces_eV_per_A=vacuum_forces,
            continuum_fixed_source_forces_eV_per_A=-np.asarray(
                fixed_gradient, dtype=float
            ),
            source_geometry_forces_eV_per_A=-source_gradient,
            cds_forces_eV_per_A=-cds_gradient,
        )

    def force_sample(self, geometry: object) -> ScalarForceSample:
        return self.evaluate_forces(geometry).force_sample

    def get_forces(self, geometry: object) -> np.ndarray:
        return np.array(self.evaluate_forces(geometry).total_forces_eV_per_A, copy=True)

    def numerical_force_audit(
        self,
        geometry: object,
        *,
        central_state: MACEPolarFrozenDDXEnergyState | None = None,
    ) -> RichardsonScalarForceEvaluation:
        self._validate_geometry(geometry)
        state = (
            self.solve(geometry)
            if central_state is None
            else self._validated_energy_state(geometry, central_state)
        )
        return self._numerical_force_backend.evaluate(
            self, geometry, central_sample=state.energy_sample
        )

    def molecular_virial(
        self,
        geometry: object,
        *,
        origin_angstrom: object | None = None,
        force_evaluation: MACEPolarFrozenDDXForceEvaluation | None = None,
    ) -> MolecularVirialEvaluation:
        self._validate_geometry(geometry)
        evaluated = (
            self.evaluate_forces(geometry)
            if force_evaluation is None
            else self._validated_force_evaluation(geometry, force_evaluation)
        )
        return MolecularVirialEvaluation.from_conservative_force(
            force_evaluation_sha256=evaluated.evaluation_sha256,
            positions_angstrom=_positions(geometry),
            forces_eV_per_A=evaluated.total_forces_eV_per_A,
            origin_angstrom=origin_angstrom,
        )

    def hessian_vector_product(
        self,
        geometry: object,
        direction: object,
        *,
        central_force: MACEPolarFrozenDDXForceEvaluation | None = None,
    ) -> RichardsonScalarHVPEvaluation:
        self._validate_geometry(geometry)
        center = (
            self.evaluate_forces(geometry)
            if central_force is None
            else self._validated_force_evaluation(geometry, central_force)
        )
        return self._hessian_backend.evaluate_hvp(
            self,
            geometry,
            direction,
            central_sample=center.force_sample,
        )

    def evaluate_hessian(
        self,
        geometry: object,
        *,
        central_force: MACEPolarFrozenDDXForceEvaluation | None = None,
    ) -> RichardsonScalarHessianEvaluation:
        self._validate_geometry(geometry)
        center = (
            self.evaluate_forces(geometry)
            if central_force is None
            else self._validated_force_evaluation(geometry, central_force)
        )
        return self._hessian_backend.evaluate(
            self,
            geometry,
            central_sample=center.force_sample,
        )

    def get_hessian(self, geometry: object) -> np.ndarray:
        return np.array(self.evaluate_hessian(geometry).hessian_eV_per_A2, copy=True)


def build_water_mace_polar_frozen_ddx_pes(
    model: object,
    symbols: tuple[str, ...] | list[str],
    *,
    cds_grid_points: int = SASA_GRID_POINTS,
    numerical_force_backend: RichardsonScalarForce | None = None,
    hessian_backend: RichardsonScalarHessian | None = None,
) -> MACEPolarFrozenSourceDDXPES:
    """Build the frozen water/ddPCM-194 route without loading another model."""

    from maple.solvation.continuum.radial_gto_ddx import (
        build_water_radial_gto_ddpcm_194_candidate,
    )

    normalized = tuple(symbols)
    return MACEPolarFrozenSourceDDXPES(
        model=model,
        continuum=build_water_radial_gto_ddpcm_194_candidate(normalized),
        solvent_term=SmoothSMDWaterCDS(normalized, grid_points=cds_grid_points),
        numerical_force_backend=numerical_force_backend,
        hessian_backend=hessian_backend,
    )


def build_smd_mace_polar_frozen_ddx_pes(
    model: object,
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
) -> MACEPolarFrozenSourceDDXPES:
    """Build one multi-solvent frozen-source ddX + official SMD-CDS PES.

    The solvent controls only independently registered SMD descriptors,
    Coulomb radii, dielectric, and the upstream PySCF CDS term.  The MACE-POLAR
    checkpoint and radial source/receiver semantics are unchanged.
    """

    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.radial_gto_ddx import RadialGTODDXBackend

    normalized = tuple(symbols)
    specification = route2_solvent_spec(solvent)
    method = str(continuum_model).strip().lower()
    continuum = RadialGTODDXBackend(
        normalized,
        smd_coulomb_radii(normalized, solvent=specification.name),
        continuum_model=method,
        dielectric=specification.descriptors.dielectric,
        lmax=lmax,
        n_lebedev=n_lebedev,
        solver_tolerance=solver_tolerance,
        eta=eta,
        n_proc=n_proc,
        configuration_contract_id=("mace-polar-frozen-source-smd-radial-gto-ddx-v1"),
    )
    return MACEPolarFrozenSourceDDXPES(
        model=model,
        continuum=continuum,
        solvent_term=PySCFSMDCDSTerm(normalized, specification.name),
        numerical_force_backend=numerical_force_backend,
        hessian_backend=hessian_backend,
    )


def build_smd_mace_polar_frozen_point_ddx_pes(
    model: object,
    symbols: tuple[str, ...] | list[str],
    *,
    solvent: str,
    lmax: int = 15,
    n_lebedev: int = 1202,
    solver_tolerance: float = 1.0e-12,
    eta: float = 0.1,
    n_proc: int = 1,
    numerical_force_backend: RichardsonScalarForce | None = None,
    hessian_backend: RichardsonScalarHessian | None = None,
) -> MACEPolarFrozenSourceDDXPES:
    """Build the no-fit point-multipole SourceEmbedding ddPCM/SMD PES.

    This profile preserves the official learned ``(q,l=1)`` coefficients but
    does not claim that a point multipole is the checkpoint electron density.
    It is a separate continuum coupling identity selected by the frozen MNSol
    source-representation gate. Physical discretization overrides remain
    callable under the generic frozen-source scalar contract; only the exact
    ``lmax=15/n_lebedev=1202/eta=0.1`` identity receives the registered
    point-profile scalar ID. The registered identity also fixes
    ``solver_tolerance=1e-12`` and ``n_proc=1``.
    """

    from maple.function.calculator.extra_correction.implicit.smd_cds import (
        smd_coulomb_radii,
    )
    from maple.function.route2_solvents import route2_solvent_spec
    from maple.solvation.continuum.mace_polar_point_ddx import (
        MACEPolarPointEmbeddedDDXBackend,
    )

    normalized = tuple(symbols)
    specification = route2_solvent_spec(solvent)
    continuum = MACEPolarPointEmbeddedDDXBackend(
        normalized,
        smd_coulomb_radii(normalized, solvent=specification.name),
        dielectric=specification.descriptors.dielectric,
        lmax=lmax,
        n_lebedev=n_lebedev,
        solver_tolerance=solver_tolerance,
        eta=eta,
        n_proc=n_proc,
    )
    adaptive_hessian = hessian_backend or RichardsonScalarHessian(
        maximum_topology_step_reductions=6,
    )
    scalar_contract_id = (
        EXPERIMENTAL_PURE_MACEPOLAR_POINT_L1_DDPCM_SMD_V1
        if (
            lmax == 15
            and n_lebedev == 1202
            and solver_tolerance == 1.0e-12
            and eta == 0.1
            and n_proc == 1
        )
        else PURE_FROZEN_DDX_SCALAR_CONTRACT_ID
    )
    return MACEPolarFrozenSourceDDXPES(
        model=model,
        continuum=continuum,
        solvent_term=PySCFSMDCDSTerm(normalized, specification.name),
        scalar_contract_id=scalar_contract_id,
        numerical_force_backend=numerical_force_backend,
        hessian_backend=adaptive_hessian,
    )


__all__ = [
    "MACEPolarFrozenDDXEnergyState",
    "MACEPolarFrozenDDXForceEvaluation",
    "MACEPolarFrozenSourceDDXPES",
    "MolecularVirialEvaluation",
    "PURE_FROZEN_DDX_PROVIDER_ID",
    "PURE_FROZEN_DDX_SCALAR_CONTRACT_ID",
    "SCIENTIFIC_STATUS",
    "SMOOTH_SMD_WATER_CDS_PROVIDER_ID",
    "SmoothSMDWaterCDS",
    "SolventEnergyState",
    "SolventEnergyTerm",
    "build_smd_mace_polar_frozen_ddx_pes",
    "build_smd_mace_polar_frozen_point_ddx_pes",
    "build_water_mace_polar_frozen_ddx_pes",
]
