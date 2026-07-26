from __future__ import annotations

import hashlib
import math
import re
from dataclasses import InitVar, dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.special import logsumexp

from .association_conditioning import SoftEffectiveVolumeEstimate
from .occupancy_analysis import (
    SoftOccupancyAnalysisInput,
    SoftOccupancyDistribution,
    SoftOccupancyEstimate,
)
from .packing_analysis import (
    PackingAnalysisInput,
    SoftPackingEstimate,
    soft_packing_result_hash,
)
from .protocol import canonical_sha256


_R_KCAL_PER_MOL_K = 0.001987204258640831
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MATRIX_TOLERANCE = 1.0e-10
_EXACT_PROFILE_TOKEN = object()
_QCT_ARTIFACT_TOKEN = object()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _require_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash.")


def _validate_covariance(
    covariance: np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    if (
        values.shape != (size, size)
        or not np.all(np.isfinite(values))
        or not np.allclose(
            values,
            values.T,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(f"{name} must be a finite symmetric matrix.")
    symmetric = 0.5 * (values + values.T)
    if float(np.min(np.linalg.eigvalsh(symmetric))) < -1.0e-12:
        raise ValueError(f"{name} must be positive semidefinite.")
    immutable = np.array(
        symmetric,
        dtype=float,
        order="C",
        copy=True,
    )
    immutable.setflags(write=False)
    return immutable


def _qct_variable_labels(count: int) -> tuple[str, ...]:
    return tuple(
        [f"p[{n}]" for n in range(count)]
        + [f"p[n>={count}]"]
        + [f"x[{n}]" for n in range(count)]
        + [f"x[n>={count}]"]
        + [f"A[{n}]_kcal_mol" for n in range(count)]
        + [f"g[{n}]_kcal_mol" for n in range(count)]
    )


@dataclass(frozen=True)
class ConditionedQCTProfile:
    """The v3 conditioned row free energies ``A_n`` and their covariance.

    The master identity is

    ``mu = -RT ln(p0) - RT ln sum_n exp(-beta A_n)``.

    A profile may be exact for a finite enumerable system or an explicitly
    declared MAPLE cluster-continuum approximation.  Those roles are never
    conflated.
    """

    occupancies: tuple[int, ...]
    free_energies_kcal_mol: np.ndarray
    covariance_of_mean: np.ndarray
    membership_definition_hash: str
    observation_volume_hash: str
    periodic_boundary_adapter_hash: str
    boundary_measure_bridge_hash: str | None
    solute_measure_hash: str
    water_hamiltonian_hash: str
    reference_hamiltonian_hash: str
    coupled_hamiltonian_hash: str
    hamiltonian_bridge_hash: str
    temperature_k: float
    boundary_conditions: str
    source_artifact_hash: str
    estimator: str
    approximation_role: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "ConditionedQCTProfile must be constructed by its "
                "validated factory."
            )

    @classmethod
    def create(
        cls,
        *,
        free_energies_kcal_mol: np.ndarray,
        covariance_of_mean: np.ndarray,
        membership_definition_hash: str,
        observation_volume_hash: str,
        periodic_boundary_adapter_hash: str,
        boundary_measure_bridge_hash: str | None,
        solute_measure_hash: str,
        water_hamiltonian_hash: str,
        reference_hamiltonian_hash: str,
        coupled_hamiltonian_hash: str,
        hamiltonian_bridge_hash: str,
        temperature_k: float,
        boundary_conditions: str,
        source_artifact_hash: str,
        estimator: str,
        approximation_role: str,
        _exact_profile_token: object | None = None,
    ) -> "ConditionedQCTProfile":
        energies = np.asarray(free_energies_kcal_mol, dtype=float)
        if (
            energies.ndim != 1
            or len(energies) < 2
            or not np.all(np.isfinite(energies))
        ):
            raise ValueError(
                "Conditioned QCT free energies must be a finite vector "
                "including n=0 and at least one nonzero occupancy."
            )
        covariance = _validate_covariance(
            covariance_of_mean,
            size=len(energies),
            name="Conditioned QCT covariance",
        )
        for name in (
            "membership_definition_hash",
            "observation_volume_hash",
            "periodic_boundary_adapter_hash",
            "solute_measure_hash",
            "water_hamiltonian_hash",
            "reference_hamiltonian_hash",
            "coupled_hamiltonian_hash",
            "hamiltonian_bridge_hash",
            "source_artifact_hash",
        ):
            _require_hash(locals()[name], name)
        if boundary_measure_bridge_hash is not None:
            _require_hash(
                boundary_measure_bridge_hash,
                "boundary_measure_bridge_hash",
            )
        if (
            isinstance(temperature_k, (bool, np.bool_))
            or not math.isfinite(float(temperature_k))
            or float(temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")
        if boundary_conditions not in {
            "periodic-3d",
            "nonperiodic-cluster",
        }:
            raise ValueError(
                "Conditioned QCT boundary conditions must be periodic-3d "
                "or nonperiodic-cluster."
            )
        if not isinstance(estimator, str) or not estimator:
            raise ValueError("estimator must be a non-empty string.")
        if approximation_role not in {
            "exact-enumerable-bridge",
            "periodic-explicit-reference",
            "maple-cluster-continuum-v3",
        }:
            raise ValueError(
                "Unsupported conditioned QCT approximation_role."
            )
        if (
            approximation_role == "maple-cluster-continuum-v3"
            and boundary_measure_bridge_hash is None
        ):
            raise ValueError(
                "Cluster-continuum profiles require a boundary-measure "
                "bridge."
            )
        if (
            approximation_role != "maple-cluster-continuum-v3"
            and boundary_measure_bridge_hash is not None
        ):
            raise ValueError(
                "Periodic profiles cannot claim a nonperiodic boundary "
                "bridge."
            )
        if (
            approximation_role != "maple-cluster-continuum-v3"
            and _exact_profile_token is not _EXACT_PROFILE_TOKEN
        ):
            raise ValueError(
                "Exact periodic conditioned profiles must be constructed "
                "from a verified reference distribution and conditional-"
                "coupling bridge."
            )

        immutable_energies = np.array(
            energies,
            dtype=float,
            order="C",
            copy=True,
        )
        immutable_energies.setflags(write=False)
        occupancies = tuple(range(len(energies)))
        content_hash = canonical_sha256(
            {
                "contract_id": "conditioned-qct-profile-v3",
                "occupancies": list(occupancies),
                "free_energies_sha256": _array_sha256(
                    immutable_energies
                ),
                "covariance_of_mean_sha256": _array_sha256(covariance),
                "membership_definition_hash": membership_definition_hash,
                "observation_volume_hash": observation_volume_hash,
                "periodic_boundary_adapter_hash": (
                    periodic_boundary_adapter_hash
                ),
                "boundary_measure_bridge_hash": (
                    boundary_measure_bridge_hash
                ),
                "solute_measure_hash": solute_measure_hash,
                "water_hamiltonian_hash": water_hamiltonian_hash,
                "reference_hamiltonian_hash": reference_hamiltonian_hash,
                "coupled_hamiltonian_hash": coupled_hamiltonian_hash,
                "hamiltonian_bridge_hash": hamiltonian_bridge_hash,
                "temperature_k": float(temperature_k),
                "boundary_conditions": boundary_conditions,
                "source_artifact_hash": source_artifact_hash,
                "estimator": estimator,
                "approximation_role": approximation_role,
            }
        )
        return cls(
            occupancies=occupancies,
            free_energies_kcal_mol=immutable_energies,
            covariance_of_mean=covariance,
            membership_definition_hash=membership_definition_hash,
            observation_volume_hash=observation_volume_hash,
            periodic_boundary_adapter_hash=periodic_boundary_adapter_hash,
            boundary_measure_bridge_hash=boundary_measure_bridge_hash,
            solute_measure_hash=solute_measure_hash,
            water_hamiltonian_hash=water_hamiltonian_hash,
            reference_hamiltonian_hash=reference_hamiltonian_hash,
            coupled_hamiltonian_hash=coupled_hamiltonian_hash,
            hamiltonian_bridge_hash=hamiltonian_bridge_hash,
            temperature_k=float(temperature_k),
            boundary_conditions=boundary_conditions,
            source_artifact_hash=source_artifact_hash,
            estimator=estimator,
            approximation_role=approximation_role,
            content_hash=content_hash,
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )


@dataclass(frozen=True)
class ConditionalCouplingProfile:
    """Periodic fixed-occupancy coupling free energies ``g_n``."""

    occupancies: tuple[int, ...]
    free_energies_kcal_mol: np.ndarray
    covariance_of_mean: np.ndarray
    membership_definition_hash: str
    observation_volume_hash: str
    periodic_boundary_adapter_hash: str
    solute_measure_hash: str
    water_hamiltonian_hash: str
    reference_hamiltonian_hash: str
    coupled_hamiltonian_hash: str
    hamiltonian_bridge_hash: str
    temperature_k: float
    source_artifact_hash: str
    estimator: str
    provenance_role: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "ConditionalCouplingProfile must be constructed by its "
                "validated factory."
            )

    @classmethod
    def create(
        cls,
        *,
        free_energies_kcal_mol: np.ndarray,
        covariance_of_mean: np.ndarray,
        membership_definition_hash: str,
        observation_volume_hash: str,
        periodic_boundary_adapter_hash: str,
        solute_measure_hash: str,
        water_hamiltonian_hash: str,
        reference_hamiltonian_hash: str,
        coupled_hamiltonian_hash: str,
        hamiltonian_bridge_hash: str,
        temperature_k: float,
        source_artifact_hash: str,
        estimator: str,
        _exact_source_token: object | None = None,
    ) -> "ConditionalCouplingProfile":
        energies = np.asarray(free_energies_kcal_mol, dtype=float)
        if (
            energies.ndim != 1
            or len(energies) < 2
            or not np.all(np.isfinite(energies))
        ):
            raise ValueError(
                "Conditional coupling free energies must be a finite vector "
                "including n=0 and at least n=1."
            )
        covariance = _validate_covariance(
            covariance_of_mean,
            size=len(energies),
            name="Conditional coupling covariance",
        )
        for name in (
            "membership_definition_hash",
            "observation_volume_hash",
            "periodic_boundary_adapter_hash",
            "solute_measure_hash",
            "water_hamiltonian_hash",
            "reference_hamiltonian_hash",
            "coupled_hamiltonian_hash",
            "hamiltonian_bridge_hash",
            "source_artifact_hash",
        ):
            _require_hash(locals()[name], name)
        if (
            isinstance(temperature_k, (bool, np.bool_))
            or not math.isfinite(float(temperature_k))
            or float(temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")
        if not isinstance(estimator, str) or not estimator:
            raise ValueError("estimator must be a non-empty string.")
        exact_estimator = (
            estimator == "exact finite enumeration of Z_X,n/Z_0,n"
        )
        if exact_estimator and _exact_source_token is not _EXACT_PROFILE_TOKEN:
            raise ValueError(
                "Exact conditional-coupling provenance is restricted to "
                "the internal finite-system oracle."
            )
        if not exact_estimator and _exact_source_token is not None:
            raise ValueError(
                "Exact-source provenance cannot be attached to an empirical "
                "conditional-coupling estimator."
            )
        provenance_role = (
            "finite-state-oracle"
            if exact_estimator
            else "sampled-or-constructed"
        )
        immutable = np.array(energies, dtype=float, order="C", copy=True)
        immutable.setflags(write=False)
        occupancies = tuple(range(len(immutable)))
        content_hash = canonical_sha256(
            {
                "contract_id": "conditional-coupling-profile-v3",
                "occupancies": list(occupancies),
                "free_energies_sha256": _array_sha256(immutable),
                "covariance_of_mean_sha256": _array_sha256(covariance),
                "membership_definition_hash": membership_definition_hash,
                "observation_volume_hash": observation_volume_hash,
                "periodic_boundary_adapter_hash": (
                    periodic_boundary_adapter_hash
                ),
                "solute_measure_hash": solute_measure_hash,
                "water_hamiltonian_hash": water_hamiltonian_hash,
                "reference_hamiltonian_hash": reference_hamiltonian_hash,
                "coupled_hamiltonian_hash": coupled_hamiltonian_hash,
                "hamiltonian_bridge_hash": hamiltonian_bridge_hash,
                "temperature_k": float(temperature_k),
                "boundary_conditions": "periodic-3d",
                "source_artifact_hash": source_artifact_hash,
                "estimator": estimator,
                "provenance_role": provenance_role,
            }
        )
        return cls(
            occupancies=occupancies,
            free_energies_kcal_mol=immutable,
            covariance_of_mean=covariance,
            membership_definition_hash=membership_definition_hash,
            observation_volume_hash=observation_volume_hash,
            periodic_boundary_adapter_hash=periodic_boundary_adapter_hash,
            solute_measure_hash=solute_measure_hash,
            water_hamiltonian_hash=water_hamiltonian_hash,
            reference_hamiltonian_hash=reference_hamiltonian_hash,
            coupled_hamiltonian_hash=coupled_hamiltonian_hash,
            hamiltonian_bridge_hash=hamiltonian_bridge_hash,
            temperature_k=float(temperature_k),
            source_artifact_hash=source_artifact_hash,
            estimator=estimator,
            provenance_role=provenance_role,
            content_hash=content_hash,
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )


def build_periodic_qct_profile(
    *,
    reference: SoftOccupancyDistribution,
    conditional_coupling: ConditionalCouplingProfile,
    covariance_of_mean: np.ndarray,
    approximation_role: str,
    _exact_source_token: object | None = None,
) -> ConditionedQCTProfile:
    """Construct ``A_n`` so every nonzero ``p_n`` enters the exact bridge."""

    if not isinstance(reference, SoftOccupancyDistribution):
        raise ValueError(
            "reference must be a SoftOccupancyDistribution."
        )
    if not isinstance(conditional_coupling, ConditionalCouplingProfile):
        raise ValueError(
            "conditional_coupling must be a ConditionalCouplingProfile."
        )
    if approximation_role not in {
        "exact-enumerable-bridge",
        "periodic-explicit-reference",
    }:
        raise ValueError(
            "Periodic conditioned profile role must be exact enumerable or "
            "periodic explicit reference."
        )
    if approximation_role == "exact-enumerable-bridge":
        if _exact_source_token is not _EXACT_PROFILE_TOKEN:
            raise ValueError(
                "Exact enumerable profiles are restricted to the internal "
                "finite-state oracle."
            )
        if (
            reference.provenance_role != "finite-state-oracle"
            or conditional_coupling.provenance_role
            != "finite-state-oracle"
            or reference.tail_probability != 0.0
            or np.any(reference.covariance_of_mean != 0.0)
            or np.any(conditional_coupling.covariance_of_mean != 0.0)
            or reference.source_artifact_hash
            != conditional_coupling.source_artifact_hash
        ):
            raise ValueError(
                "Exact enumerable profiles require matching zero-uncertainty "
                "finite-oracle artifacts."
            )
    elif _exact_source_token is not None:
        raise ValueError(
            "Exact-source token is invalid for empirical periodic profiles."
        )
    if reference.occupancies != conditional_coupling.occupancies:
        raise ValueError("Periodic QCT bridge occupancy supports differ.")
    comparisons = (
        (
            reference.membership_definition_hash,
            conditional_coupling.membership_definition_hash,
        ),
        (
            reference.observation_volume_hash,
            conditional_coupling.observation_volume_hash,
        ),
        (
            reference.boundary_adapter_hash,
            conditional_coupling.periodic_boundary_adapter_hash,
        ),
        (
            reference.solute_measure_hash,
            conditional_coupling.solute_measure_hash,
        ),
        (
            reference.water_hamiltonian_hash,
            conditional_coupling.water_hamiltonian_hash,
        ),
        (
            reference.system_hamiltonian_hash,
            conditional_coupling.reference_hamiltonian_hash,
        ),
    )
    if any(left != right for left, right in comparisons):
        raise ValueError("Periodic QCT reference/conditional hashes differ.")
    if not math.isclose(
        reference.temperature_k,
        conditional_coupling.temperature_k,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Periodic QCT bridge temperatures differ.")
    rt = _R_KCAL_PER_MOL_K * reference.temperature_k
    probabilities = reference.probabilities
    rows = conditional_coupling.free_energies_kcal_mol - rt * np.log(
        probabilities / probabilities[0]
    )
    source_artifact_hash = canonical_sha256(
        {
            "contract_id": "periodic-conditioned-qct-build-v3",
            "reference_distribution_hash": reference.content_hash,
            "conditional_coupling_profile_hash": (
                conditional_coupling.content_hash
            ),
            "identity": "A_n=g_n-RT ln[p_n/p_0]",
            "approximation_role": approximation_role,
        }
    )
    return ConditionedQCTProfile.create(
        free_energies_kcal_mol=rows,
        covariance_of_mean=covariance_of_mean,
        membership_definition_hash=reference.membership_definition_hash,
        observation_volume_hash=reference.observation_volume_hash,
        periodic_boundary_adapter_hash=reference.boundary_adapter_hash,
        boundary_measure_bridge_hash=None,
        solute_measure_hash=reference.solute_measure_hash,
        water_hamiltonian_hash=reference.water_hamiltonian_hash,
        reference_hamiltonian_hash=reference.system_hamiltonian_hash,
        coupled_hamiltonian_hash=(
            conditional_coupling.coupled_hamiltonian_hash
        ),
        hamiltonian_bridge_hash=(
            conditional_coupling.hamiltonian_bridge_hash
        ),
        temperature_k=reference.temperature_k,
        boundary_conditions="periodic-3d",
        source_artifact_hash=source_artifact_hash,
        estimator=(
            "verified periodic identity A_n=g_n-RT ln[p_n/p_0]"
        ),
        approximation_role=approximation_role,
        _exact_profile_token=_EXACT_PROFILE_TOKEN,
    )


def _build_exact_periodic_qct_profile(
    *,
    reference: SoftOccupancyDistribution,
    conditional_coupling: ConditionalCouplingProfile,
    covariance_of_mean: np.ndarray,
) -> ConditionedQCTProfile:
    """Internal finite-oracle constructor for exact periodic rows."""

    return build_periodic_qct_profile(
        reference=reference,
        conditional_coupling=conditional_coupling,
        covariance_of_mean=covariance_of_mean,
        approximation_role="exact-enumerable-bridge",
        _exact_source_token=_EXACT_PROFILE_TOKEN,
    )


@dataclass(frozen=True)
class BoundaryMeasureBridge:
    """Explicit bridge between one surface and two boundary adapters."""

    membership_surface_hash: str
    periodic_adapter_hash: str
    nonperiodic_adapter_hash: str
    reference_distribution_hash: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "BoundaryMeasureBridge must be constructed by its "
                "validated factory."
            )
        expected = canonical_sha256(
            {
                "contract_id": "boundary-measure-bridge-v3",
                "membership_surface_hash": self.membership_surface_hash,
                "periodic_adapter_hash": self.periodic_adapter_hash,
                "nonperiodic_adapter_hash": self.nonperiodic_adapter_hash,
                "reference_distribution_hash": (
                    self.reference_distribution_hash
                ),
                "scientific_role": (
                    "shared surface identity with explicit boundary "
                    "approximation residual"
                ),
            }
        )
        if self.content_hash != expected:
            raise ValueError(
                "BoundaryMeasureBridge content hash does not match its data."
            )

    @classmethod
    def create(
        cls,
        *,
        reference: SoftOccupancyDistribution,
        nonperiodic_adapter_hash: str,
    ) -> "BoundaryMeasureBridge":
        if (
            not isinstance(reference, SoftOccupancyDistribution)
            or reference.ensemble_role != "reference-product"
        ):
            raise ValueError(
                "Boundary bridge requires a validated reference-product "
                "occupancy distribution."
            )
        membership_surface_hash = reference.observation_volume_hash
        periodic_adapter_hash = reference.boundary_adapter_hash
        for name in (
            "membership_surface_hash",
            "periodic_adapter_hash",
            "nonperiodic_adapter_hash",
        ):
            _require_hash(locals()[name], name)
        if periodic_adapter_hash == nonperiodic_adapter_hash:
            raise ValueError(
                "Periodic and nonperiodic boundary adapters must be distinct."
            )
        content_hash = canonical_sha256(
            {
                "contract_id": "boundary-measure-bridge-v3",
                "membership_surface_hash": membership_surface_hash,
                "periodic_adapter_hash": periodic_adapter_hash,
                "nonperiodic_adapter_hash": nonperiodic_adapter_hash,
                "reference_distribution_hash": reference.content_hash,
                "scientific_role": (
                    "shared surface identity with explicit boundary "
                    "approximation residual"
                ),
            }
        )
        return cls(
            membership_surface_hash=membership_surface_hash,
            periodic_adapter_hash=periodic_adapter_hash,
            nonperiodic_adapter_hash=nonperiodic_adapter_hash,
            reference_distribution_hash=reference.content_hash,
            content_hash=content_hash,
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )


@dataclass(frozen=True)
class ClusterQCTBuild:
    """Auditable construction of ``A_n`` from labeled cluster edges."""

    profile: ConditionedQCTProfile
    rows: tuple[Mapping[str, Any], ...]
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "ClusterQCTBuild must be constructed by the labeled-cluster "
                "builder."
            )


def build_cluster_qct_profile(
    *,
    reference_distribution: SoftOccupancyDistribution,
    edge_alchemical_free_energies_kcal_mol: np.ndarray,
    effective_volume_estimates: tuple[
        SoftEffectiveVolumeEstimate,
        ...,
    ],
    edge_release_free_energies_kcal_mol: np.ndarray,
    outer_cluster_free_energies_kcal_mol: np.ndarray,
    outer_water_free_energy_kcal_mol: float,
    water_number_density_per_angstrom3: float,
    covariance_of_mean: np.ndarray,
    temperature_k: float,
    membership_definition_hash: str,
    observation_volume_hash: str,
    solute_measure_hash: str,
    water_hamiltonian_hash: str,
    reference_hamiltonian_hash: str,
    coupled_hamiltonian_hash: str,
    outer_contract_hash: str,
    boundary_measure_bridge: BoundaryMeasureBridge,
    hamiltonian_bridge_hash: str,
    joint_provenance_graph_hash: str,
    covariance_method: str,
    approximation_role: str,
) -> ClusterQCTBuild:
    """Build every v3 row with density, volume and ``n!`` counted once."""

    alchemical = np.asarray(
        edge_alchemical_free_energies_kcal_mol,
        dtype=float,
    )
    if (
        not isinstance(reference_distribution, SoftOccupancyDistribution)
        or reference_distribution.ensemble_role != "reference-product"
    ):
        raise ValueError(
            "Cluster QCT construction requires a validated reference-product "
            "occupancy distribution."
        )
    if not isinstance(boundary_measure_bridge, BoundaryMeasureBridge):
        raise ValueError(
            "boundary_measure_bridge must be a BoundaryMeasureBridge."
        )
    volume_artifacts = tuple(effective_volume_estimates)
    if any(
        not isinstance(value, SoftEffectiveVolumeEstimate)
        for value in volume_artifacts
    ):
        raise ValueError(
            "effective_volume_estimates must contain only "
            "SoftEffectiveVolumeEstimate artifacts."
        )
    volumes = np.asarray(
        [value.volume_angstrom3 for value in volume_artifacts],
        dtype=float,
    )
    releases = np.asarray(
        edge_release_free_energies_kcal_mol,
        dtype=float,
    )
    outer_cluster = np.asarray(
        outer_cluster_free_energies_kcal_mol,
        dtype=float,
    )
    edge_count = len(alchemical)
    if (
        alchemical.ndim != 1
        or edge_count < 1
        or len(volume_artifacts) != edge_count
        or volumes.shape != (edge_count,)
        or releases.shape != (edge_count,)
        or outer_cluster.shape != (edge_count + 1,)
        or not np.all(np.isfinite(alchemical))
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0.0)
        or not np.all(np.isfinite(releases))
        or not np.all(np.isfinite(outer_cluster))
    ):
        raise ValueError(
            "Cluster QCT construction requires nmax finite alchemical, "
            "positive V_eff and release edges plus nmax+1 outer rows."
        )
    if boundary_measure_bridge.membership_surface_hash != (
        observation_volume_hash
    ):
        raise ValueError(
            "Boundary bridge does not match the cluster membership surface."
        )
    if (
        boundary_measure_bridge.reference_distribution_hash
        != reference_distribution.content_hash
        or boundary_measure_bridge.periodic_adapter_hash
        != reference_distribution.boundary_adapter_hash
        or reference_distribution.observation_volume_hash
        != observation_volume_hash
        or reference_distribution.membership_definition_hash
        != membership_definition_hash
        or reference_distribution.solute_measure_hash
        != solute_measure_hash
        or reference_distribution.water_hamiltonian_hash
        != water_hamiltonian_hash
        or reference_distribution.system_hamiltonian_hash
        != reference_hamiltonian_hash
    ):
        raise ValueError(
            "Boundary bridge/reference distribution does not match the "
            "cluster QCT construction."
        )
    if not math.isclose(
        reference_distribution.temperature_k,
        float(temperature_k),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "Reference distribution and cluster QCT temperatures differ."
        )
    for artifact in volume_artifacts:
        if (
            artifact.membership_definition_hash
            != membership_definition_hash
            or artifact.observation_volume_hash
            != observation_volume_hash
            or artifact.boundary_adapter_hash
            != boundary_measure_bridge.nonperiodic_adapter_hash
        ):
            raise ValueError(
                "Effective-volume artifact does not match the cluster "
                "membership/boundary bridge."
            )
    for name, value in (
        ("outer_water_free_energy_kcal_mol", outer_water_free_energy_kcal_mol),
        (
            "water_number_density_per_angstrom3",
            water_number_density_per_angstrom3,
        ),
        ("temperature_k", temperature_k),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or (
                float(value) <= 0.0
                if name != "outer_water_free_energy_kcal_mol"
                else False
            )
        ):
            qualifier = "finite" if name.startswith("outer_water") else (
                "finite and positive"
            )
            raise ValueError(f"{name} must be {qualifier}.")
    for name in (
        "outer_contract_hash",
        "hamiltonian_bridge_hash",
        "joint_provenance_graph_hash",
    ):
        _require_hash(locals()[name], name)
    if covariance_method not in {
        "joint-provenance-graph-bootstrap",
        "exact-enumeration",
    }:
        raise ValueError(
            "Cluster QCT covariance must come from a joint provenance-graph "
            "bootstrap or exact enumeration."
        )
    if approximation_role != "maple-cluster-continuum-v3":
        raise ValueError(
            "The nonperiodic cluster builder is always the declared "
            "maple-cluster-continuum-v3 approximation."
        )

    rt = _R_KCAL_PER_MOL_K * float(temperature_k)
    density = float(water_number_density_per_angstrom3)
    rows: list[Mapping[str, Any]] = []
    free_energies: list[float] = []
    for occupancy in range(edge_count + 1):
        alchemical_total = float(np.sum(alchemical[:occupancy]))
        volume_density = float(
            -rt * np.sum(np.log(density * volumes[:occupancy]))
        )
        symmetry = float(rt * math.lgamma(occupancy + 1.0))
        release = float(np.sum(releases[:occupancy]))
        outer_cycle = float(
            outer_cluster[occupancy]
            - occupancy * float(outer_water_free_energy_kcal_mol)
        )
        total = (
            alchemical_total
            + volume_density
            + symmetry
            + release
            + outer_cycle
        )
        row = MappingProxyType(
            {
                "occupancy": occupancy,
                "labeled_alchemical_kcal_mol": alchemical_total,
                "volume_density_kcal_mol": volume_density,
                "symmetry_kcal_mol": symmetry,
                "release_kcal_mol": release,
                "outer_cycle_kcal_mol": outer_cycle,
                "effective_volume_artifact_hashes": [
                    artifact.content_hash
                    for artifact in volume_artifacts[:occupancy]
                ],
                "A_n_kcal_mol": total,
            }
        )
        rows.append(row)
        free_energies.append(total)

    source_artifact_hash = canonical_sha256(
        {
            "contract_id": "labeled-cluster-qct-build-v3",
            "reference_distribution_hash": (
                reference_distribution.content_hash
            ),
            "edge_alchemical_sha256": _array_sha256(alchemical),
            "effective_volume_artifact_hashes": [
                artifact.content_hash for artifact in volume_artifacts
            ],
            "edge_release_sha256": _array_sha256(releases),
            "outer_cluster_sha256": _array_sha256(outer_cluster),
            "outer_water_free_energy_kcal_mol": float(
                outer_water_free_energy_kcal_mol
            ),
            "water_number_density_per_angstrom3": density,
            "temperature_k": float(temperature_k),
            "outer_contract_hash": outer_contract_hash,
            "boundary_measure_bridge_hash": (
                boundary_measure_bridge.content_hash
            ),
            "hamiltonian_bridge_hash": hamiltonian_bridge_hash,
            "joint_provenance_graph_hash": joint_provenance_graph_hash,
            "covariance_method": covariance_method,
            "counting_convention": (
                "labeled sequential edges plus RT ln(n!) and one "
                "-RT ln(rho_W V_eff) per edge"
            ),
        }
    )
    profile = ConditionedQCTProfile.create(
        free_energies_kcal_mol=np.asarray(free_energies),
        covariance_of_mean=covariance_of_mean,
        membership_definition_hash=membership_definition_hash,
        observation_volume_hash=observation_volume_hash,
        periodic_boundary_adapter_hash=(
            boundary_measure_bridge.periodic_adapter_hash
        ),
        boundary_measure_bridge_hash=boundary_measure_bridge.content_hash,
        solute_measure_hash=solute_measure_hash,
        water_hamiltonian_hash=water_hamiltonian_hash,
        reference_hamiltonian_hash=reference_hamiltonian_hash,
        coupled_hamiltonian_hash=coupled_hamiltonian_hash,
        hamiltonian_bridge_hash=hamiltonian_bridge_hash,
        temperature_k=temperature_k,
        boundary_conditions="nonperiodic-cluster",
        source_artifact_hash=source_artifact_hash,
        estimator=(
            "sequential labeled cluster association with one density-volume "
            "factor, one n! conversion and one outer reference cycle"
        ),
        approximation_role=approximation_role,
    )
    content_hash = canonical_sha256(
        {
            "contract_id": "cluster-qct-build-result-v3",
            "profile_hash": profile.content_hash,
            "source_artifact_hash": source_artifact_hash,
            "rows": [dict(row) for row in rows],
        }
    )
    return ClusterQCTBuild(
        profile=profile,
        rows=tuple(rows),
        content_hash=content_hash,
        _factory_token=_QCT_ARTIFACT_TOKEN,
    )


@dataclass(frozen=True)
class PackingOccupancyBridge:
    """Proof that packing and reference occupancy use one sampled measure."""

    role: str
    reference_distribution_hash: str
    reference_estimate_hash: str
    reference_analysis_input_hash: str
    packing_analysis_input_hash: str
    packing_result_hash: str
    reduced_potential_table_hash: str
    periodic_boundary_adapter_hash: str
    p0_reference: float
    p0_from_free_energy: float
    p0_from_expectation: float
    agreement_abs_tolerance: float
    agreement_z_max: float
    maximum_standardized_residual: float
    status: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    @staticmethod
    def _preimage_from_values(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "contract_id": "packing-occupancy-bridge-v3",
            "role": values["role"],
            "reference_distribution_hash": (
                values["reference_distribution_hash"]
            ),
            "reference_estimate_hash": values["reference_estimate_hash"],
            "reference_analysis_input_hash": (
                values["reference_analysis_input_hash"]
            ),
            "packing_analysis_input_hash": (
                values["packing_analysis_input_hash"]
            ),
            "packing_result_hash": values["packing_result_hash"],
            "reduced_potential_table_hash": (
                values["reduced_potential_table_hash"]
            ),
            "periodic_boundary_adapter_hash": (
                values["periodic_boundary_adapter_hash"]
            ),
            "p0_reference": values["p0_reference"],
            "p0_from_free_energy": values["p0_from_free_energy"],
            "p0_from_expectation": values["p0_from_expectation"],
            "agreement_abs_tolerance": values["agreement_abs_tolerance"],
            "agreement_z_max": values["agreement_z_max"],
            "maximum_standardized_residual": (
                values["maximum_standardized_residual"]
            ),
            "status": values["status"],
        }

    def _content_preimage(self) -> dict[str, Any]:
        return self._preimage_from_values(self.__dict__)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "PackingOccupancyBridge must be constructed by its "
                "validated factory."
            )
        if self.content_hash != canonical_sha256(self._content_preimage()):
            raise ValueError(
                "PackingOccupancyBridge content hash does not match its data."
            )

    @classmethod
    def create(
        cls,
        *,
        packing_input: PackingAnalysisInput,
        packing_result: SoftPackingEstimate,
        reference_input: SoftOccupancyAnalysisInput,
        reference_estimate: SoftOccupancyEstimate,
        agreement_abs_tolerance: float,
        agreement_z_max: float,
    ) -> "PackingOccupancyBridge":
        if not isinstance(packing_input, PackingAnalysisInput):
            raise ValueError("packing_input must be a PackingAnalysisInput.")
        if not isinstance(reference_input, SoftOccupancyAnalysisInput):
            raise ValueError(
                "reference_input must be a SoftOccupancyAnalysisInput."
            )
        if not isinstance(reference_estimate, SoftOccupancyEstimate):
            raise ValueError(
                "reference_estimate must be a SoftOccupancyEstimate."
            )
        reference = reference_estimate.distribution
        if reference.ensemble_role != "reference-product":
            raise ValueError("Packing bridge requires reference-product p(n).")
        if (
            reference.source_artifact_hash != reference_input.content_hash
            or reference_estimate.analysis_input_hash
            != reference_input.content_hash
        ):
            raise ValueError(
                "Reference estimate does not belong to reference_input."
            )
        if (
            not isinstance(packing_result, SoftPackingEstimate)
            or packing_result.get("status") != "passed"
            or packing_result.analysis_input_hash != packing_input.content_hash
            or packing_result.get("packing_analysis_input_sha256")
            != packing_input.content_hash
            or packing_result.get("reduced_potential_table_sha256")
            != packing_input.table.state_hash
        ):
            raise ValueError(
                "Packing bridge requires one passed, hash-bound packing result."
            )
        if (
            packing_result.content_hash
            != soft_packing_result_hash(packing_result)
            or packing_result["packing_result_sha256"]
            != packing_result.content_hash
        ):
            raise ValueError(
                "Packing result content hash does not match its data."
            )
        if packing_input.table.state_hash != reference_input.table.state_hash:
            raise ValueError(
                "Packing and reference occupancy must share the exact reduced-"
                "potential table and frame ordering."
            )
        schedule = packing_input.schedule
        contract = reference_input.contract
        comparisons = (
            ("target index", schedule.target_state_index, contract.target_state_index),
            ("target label", schedule.target_state.label, contract.target_state_label),
            (
                "sampling measure",
                schedule.conditioning_measure_id,
                contract.sampling_measure_id,
            ),
            (
                "membership definition",
                schedule.membership_definition_hash,
                contract.membership_definition_hash,
            ),
            (
                "observation volume",
                schedule.observation_volume_hash,
                contract.observation_volume_hash,
            ),
            (
                "boundary adapter",
                schedule.boundary_adapter_hash,
                contract.boundary_adapter_hash,
            ),
            (
                "solute measure",
                schedule.solute_measure_hash,
                contract.solute_measure_hash,
            ),
            (
                "water Hamiltonian",
                schedule.water_hamiltonian_hash,
                contract.water_hamiltonian_hash,
            ),
            (
                "boundary conditions",
                schedule.boundary_conditions,
                contract.boundary_conditions,
            ),
            (
                "active occupancy maximum",
                schedule.active_occupancy_max,
                contract.active_occupancy_max,
            ),
        )
        for label, scheduled, observed in comparisons:
            if scheduled != observed:
                raise ValueError(f"Packing/occupancy {label} mismatch.")
        if not math.isclose(
            schedule.temperature_k,
            contract.temperature_k,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Packing/occupancy temperature mismatch.")
        with np.errstate(divide="ignore"):
            occupancy_log_empty = np.log(
                reference_input.occupancy_weights[:, 0]
            )
        if not np.allclose(
            occupancy_log_empty,
            packing_input.log_empty_weights,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Packing empty weights and occupancy omega_0 differ."
            )
        for name, value in (
            ("agreement_abs_tolerance", agreement_abs_tolerance),
            ("agreement_z_max", agreement_z_max),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")

        p0_reference = float(reference.probabilities[0])
        p0_from_free_energy = float(
            packing_result["p0_from_free_energy"]
        )
        p0_from_expectation = float(
            packing_result["p0_from_reweighted_expectation"]
        )
        free_difference_se = float(
            packing_result["p0_estimator_difference_standard_error"]
        )
        free_allowed = max(
            float(agreement_abs_tolerance),
            float(agreement_z_max) * free_difference_se,
        )
        free_residual = abs(p0_reference - p0_from_free_energy)
        expectation_residual = abs(
            p0_reference - p0_from_expectation
        )
        failed = (
            free_residual > free_allowed
            or expectation_residual > float(agreement_abs_tolerance)
        )
        standardized = (
            free_residual / free_difference_se
            if free_difference_se > 0.0
            else (0.0 if free_residual == 0.0 else math.inf)
        )
        numerical_residual = (
            expectation_residual / float(agreement_abs_tolerance)
        )
        status = "failed" if failed else "passed"
        values = {
            "role": "shared-packing-occupancy-analysis",
            "reference_distribution_hash": reference.content_hash,
            "reference_estimate_hash": reference_estimate.content_hash,
            "reference_analysis_input_hash": reference_input.content_hash,
            "packing_analysis_input_hash": packing_input.content_hash,
            "packing_result_hash": packing_result.content_hash,
            "reduced_potential_table_hash": packing_input.table.state_hash,
            "periodic_boundary_adapter_hash": (
                reference.boundary_adapter_hash
            ),
            "p0_reference": p0_reference,
            "p0_from_free_energy": p0_from_free_energy,
            "p0_from_expectation": p0_from_expectation,
            "agreement_abs_tolerance": float(agreement_abs_tolerance),
            "agreement_z_max": float(agreement_z_max),
            "maximum_standardized_residual": max(
                standardized,
                numerical_residual,
            ),
            "status": status,
        }
        return cls(
            **values,
            content_hash=canonical_sha256(
                cls._preimage_from_values(values)
            ),
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )

    @classmethod
    def exact_enumeration(
        cls,
        reference: SoftOccupancyDistribution,
    ) -> "PackingOccupancyBridge":
        if not isinstance(reference, SoftOccupancyDistribution):
            raise ValueError(
                "reference must be a SoftOccupancyDistribution."
            )
        if reference.provenance_role != "finite-state-oracle":
            raise ValueError(
                "Exact packing evidence is restricted to the finite-state "
                "enumerator."
            )
        if (
            reference.tail_probability != 0.0
            or np.any(reference.covariance_of_mean != 0.0)
        ):
            raise ValueError(
                "Exact packing evidence requires zero tail and zero "
                "sampling covariance."
            )
        p0 = float(reference.probabilities[0])
        values = {
            "role": "exact-enumeration",
            "reference_distribution_hash": reference.content_hash,
            "reference_estimate_hash": reference.source_artifact_hash,
            "reference_analysis_input_hash": (
                reference.source_artifact_hash
            ),
            "packing_analysis_input_hash": reference.source_artifact_hash,
            "packing_result_hash": reference.source_artifact_hash,
            "reduced_potential_table_hash": (
                reference.source_artifact_hash
            ),
            "periodic_boundary_adapter_hash": (
                reference.boundary_adapter_hash
            ),
            "p0_reference": p0,
            "p0_from_free_energy": p0,
            "p0_from_expectation": p0,
            "agreement_abs_tolerance": 0.0,
            "agreement_z_max": 0.0,
            "maximum_standardized_residual": 0.0,
            "status": "passed",
        }
        return cls(
            **values,
            content_hash=canonical_sha256(
                cls._preimage_from_values(values)
            ),
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )


@dataclass(frozen=True)
class QCTBootstrapEvidence:
    """Replicate-level provenance for one shared QCT covariance."""

    replicate_values: np.ndarray
    covariance: np.ndarray
    variable_labels: tuple[str, ...]
    reference_distribution_hash: str
    coupled_distribution_hash: str
    conditioned_profile_hash: str
    conditional_coupling_profile_hash: str
    provenance_graph_hash: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    @staticmethod
    def _content_hash_from_values(
        *,
        replicate_values: np.ndarray,
        covariance: np.ndarray,
        variable_labels: tuple[str, ...],
        reference_distribution_hash: str,
        coupled_distribution_hash: str,
        conditioned_profile_hash: str,
        conditional_coupling_profile_hash: str,
        provenance_graph_hash: str,
    ) -> str:
        return canonical_sha256(
            {
                "contract_id": "qct-bootstrap-evidence-v3",
                "replicate_values_sha256": _array_sha256(
                    replicate_values
                ),
                "covariance_sha256": _array_sha256(covariance),
                "variable_labels": list(variable_labels),
                "reference_distribution_hash": (
                    reference_distribution_hash
                ),
                "coupled_distribution_hash": (
                    coupled_distribution_hash
                ),
                "conditioned_profile_hash": conditioned_profile_hash,
                "conditional_coupling_profile_hash": (
                    conditional_coupling_profile_hash
                ),
                "provenance_graph_hash": provenance_graph_hash,
            }
        )

    def _expected_content_hash(self, covariance: np.ndarray) -> str:
        return self._content_hash_from_values(
            replicate_values=self.replicate_values,
            covariance=covariance,
            variable_labels=self.variable_labels,
            reference_distribution_hash=self.reference_distribution_hash,
            coupled_distribution_hash=self.coupled_distribution_hash,
            conditioned_profile_hash=self.conditioned_profile_hash,
            conditional_coupling_profile_hash=(
                self.conditional_coupling_profile_hash
            ),
            provenance_graph_hash=self.provenance_graph_hash,
        )

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "QCTBootstrapEvidence must be constructed by its "
                "validated factory."
            )
        if (
            self.replicate_values.ndim != 2
            or self.replicate_values.shape[0] < 2
            or self.replicate_values.shape[1] != len(self.variable_labels)
            or not np.all(np.isfinite(self.replicate_values))
            or len(set(self.variable_labels)) != len(self.variable_labels)
            or any(
                not isinstance(label, str) or not label
                for label in self.variable_labels
            )
        ):
            raise ValueError(
                "QCTBootstrapEvidence requires finite replicate vectors "
                "with one uniquely labeled column per QCT variable."
            )
        for name in (
            "reference_distribution_hash",
            "coupled_distribution_hash",
            "conditioned_profile_hash",
            "conditional_coupling_profile_hash",
            "provenance_graph_hash",
        ):
            _require_hash(getattr(self, name), name)
        expected_covariance = np.atleast_2d(
            np.cov(self.replicate_values, rowvar=False, ddof=1)
        )
        if (
            self.covariance.shape != expected_covariance.shape
            or not np.allclose(
                self.covariance,
                expected_covariance,
                rtol=0.0,
                atol=1.0e-12,
            )
            or self.content_hash
            != self._expected_content_hash(expected_covariance)
        ):
            raise ValueError(
                "QCTBootstrapEvidence does not match its replicate data."
            )

    @classmethod
    def create(
        cls,
        *,
        replicate_columns: Mapping[str, np.ndarray],
        reference: SoftOccupancyDistribution,
        coupled: SoftOccupancyDistribution,
        profile: ConditionedQCTProfile,
        conditional_coupling: ConditionalCouplingProfile,
        provenance_graph_hash: str,
    ) -> "QCTBootstrapEvidence":
        _validate_ledger_compatibility(
            reference,
            coupled,
            profile,
            conditional_coupling,
        )
        labels = _qct_variable_labels(len(profile.occupancies))
        if (
            not isinstance(replicate_columns, Mapping)
            or set(replicate_columns) != set(labels)
        ):
            raise ValueError(
                "QCT bootstrap evidence columns must exactly match the "
                "canonical ledger variable labels."
            )
        columns = tuple(
            np.asarray(replicate_columns[label], dtype=float)
            for label in labels
        )
        if (
            any(column.ndim != 1 for column in columns)
            or len({len(column) for column in columns}) != 1
            or len(columns[0]) < 2
            or any(not np.all(np.isfinite(column)) for column in columns)
        ):
            raise ValueError(
                "QCT bootstrap evidence requires at least two finite "
                "replicate values for every canonical ledger variable."
            )
        _require_hash(provenance_graph_hash, "provenance_graph_hash")
        immutable_values = np.array(
            np.column_stack(columns),
            dtype=float,
            order="C",
            copy=True,
        )
        covariance = np.atleast_2d(
            np.cov(immutable_values, rowvar=False, ddof=1)
        )
        immutable_values.setflags(write=False)
        covariance.setflags(write=False)
        reference_hash = reference.content_hash
        coupled_hash = coupled.content_hash
        profile_hash = profile.content_hash
        conditional_hash = conditional_coupling.content_hash
        content_hash = cls._content_hash_from_values(
            replicate_values=immutable_values,
            covariance=covariance,
            variable_labels=labels,
            reference_distribution_hash=reference_hash,
            coupled_distribution_hash=coupled_hash,
            conditioned_profile_hash=profile_hash,
            conditional_coupling_profile_hash=conditional_hash,
            provenance_graph_hash=provenance_graph_hash,
        )
        return cls(
            replicate_values=immutable_values,
            covariance=covariance,
            variable_labels=labels,
            reference_distribution_hash=reference_hash,
            coupled_distribution_hash=coupled_hash,
            conditioned_profile_hash=profile_hash,
            conditional_coupling_profile_hash=conditional_hash,
            provenance_graph_hash=provenance_graph_hash,
            content_hash=content_hash,
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )


@dataclass(frozen=True)
class QCTJointCovariance:
    """Full covariance of ``[p, p_tail, x, x_tail, A, g]``."""

    matrix: np.ndarray
    variable_labels: tuple[str, ...]
    reference_distribution_hash: str
    coupled_distribution_hash: str
    conditioned_profile_hash: str
    conditional_coupling_profile_hash: str
    relation: str
    evidence_hash: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def _expected_content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "qct-joint-covariance-v3",
                "matrix_sha256": _array_sha256(self.matrix),
                "variable_labels": list(self.variable_labels),
                "reference_distribution_hash": (
                    self.reference_distribution_hash
                ),
                "coupled_distribution_hash": (
                    self.coupled_distribution_hash
                ),
                "conditioned_profile_hash": self.conditioned_profile_hash,
                "conditional_coupling_profile_hash": (
                    self.conditional_coupling_profile_hash
                ),
                "relation": self.relation,
                "evidence_hash": self.evidence_hash,
            }
        )

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _QCT_ARTIFACT_TOKEN:
            raise ValueError(
                "QCTJointCovariance must be constructed by its validated "
                "factory."
            )
        if self.content_hash != self._expected_content_hash():
            raise ValueError(
                "QCTJointCovariance content hash does not match its data."
            )

    @classmethod
    def create(
        cls,
        *,
        reference: SoftOccupancyDistribution,
        coupled: SoftOccupancyDistribution,
        profile: ConditionedQCTProfile,
        conditional_coupling: ConditionalCouplingProfile,
        matrix: np.ndarray,
        relation: str,
        evidence_hash: str,
        bootstrap_evidence: QCTBootstrapEvidence | None = None,
    ) -> "QCTJointCovariance":
        _validate_ledger_compatibility(
            reference,
            coupled,
            profile,
            conditional_coupling,
        )
        count = len(profile.occupancies)
        probability_count = count + 1
        total_count = 4 * count + 2
        labels = _qct_variable_labels(count)
        proposed = np.asarray(matrix, dtype=float)
        if (
            proposed.shape != (total_count, total_count)
            or not np.all(np.isfinite(proposed))
            or not np.allclose(
                proposed,
                proposed.T,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                "QCT joint covariance must be a finite symmetric matrix."
            )
        if relation not in {
            "exact-enumeration",
            "independent-block-diagonal",
            "shared-provenance-graph-bootstrap",
        }:
            raise ValueError("Unsupported QCT covariance relation.")
        _require_hash(evidence_hash, "evidence_hash")
        if relation == "shared-provenance-graph-bootstrap":
            if (
                not isinstance(bootstrap_evidence, QCTBootstrapEvidence)
                or evidence_hash != bootstrap_evidence.content_hash
                or bootstrap_evidence.variable_labels != labels
                or bootstrap_evidence.reference_distribution_hash
                != reference.content_hash
                or bootstrap_evidence.coupled_distribution_hash
                != coupled.content_hash
                or bootstrap_evidence.conditioned_profile_hash
                != profile.content_hash
                or bootstrap_evidence.conditional_coupling_profile_hash
                != conditional_coupling.content_hash
                or bootstrap_evidence.replicate_values.shape[1]
                != total_count
                or bootstrap_evidence.covariance.shape != proposed.shape
                or not np.allclose(
                    proposed,
                    bootstrap_evidence.covariance,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise ValueError(
                    "Shared QCT covariance requires the matching replicate-"
                    "level bootstrap evidence artifact."
                )
        elif bootstrap_evidence is not None:
            raise ValueError(
                "Bootstrap evidence is only valid for shared bootstrap "
                "covariance."
            )

        expected_blocks = (
            reference.covariance_of_mean,
            coupled.covariance_of_mean,
            profile.covariance_of_mean,
            conditional_coupling.covariance_of_mean,
        )
        block_slices = (
            slice(0, probability_count),
            slice(probability_count, 2 * probability_count),
            slice(2 * probability_count, 2 * probability_count + count),
            slice(2 * probability_count + count, total_count),
        )
        for block_slice, expected in zip(
            block_slices,
            expected_blocks,
            strict=True,
        ):
            observed = proposed[block_slice, block_slice]
            if not np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "QCT joint covariance diagonal blocks must match their "
                    "source artifacts."
                )

        if relation == "independent-block-diagonal":
            off_diagonal = np.array(proposed, copy=True)
            for block_slice in block_slices:
                off_diagonal[block_slice, block_slice] = 0.0
            if np.max(np.abs(off_diagonal)) > _MATRIX_TOLERANCE:
                raise ValueError(
                    "Independent block-diagonal covariance contains "
                    "cross-block terms."
                )
        elif relation == "exact-enumeration":
            exact_source_hashes = {
                reference.source_artifact_hash,
                coupled.source_artifact_hash,
                conditional_coupling.source_artifact_hash,
                evidence_hash,
            }
            if (
                reference.provenance_role != "finite-state-oracle"
                or coupled.provenance_role != "finite-state-oracle"
                or conditional_coupling.provenance_role
                != "finite-state-oracle"
                or profile.approximation_role
                != "exact-enumerable-bridge"
                or len(exact_source_hashes) != 1
            ):
                raise ValueError(
                    "Exact-enumeration covariance requires one matching "
                    "finite-oracle provenance artifact."
                )
        reference_null = np.zeros(total_count)
        reference_null[:probability_count] = 1.0
        coupled_null = np.zeros(total_count)
        coupled_null[
            probability_count : 2 * probability_count
        ] = 1.0
        if (
            np.max(np.abs(proposed @ reference_null))
            > _MATRIX_TOLERANCE
            or np.max(np.abs(proposed @ coupled_null))
            > _MATRIX_TOLERANCE
        ):
            raise ValueError(
                "QCT joint covariance violates an occupancy normalization "
                "nullspace."
            )
        if (
            relation == "exact-enumeration"
            and np.max(np.abs(proposed)) > _MATRIX_TOLERANCE
        ):
            raise ValueError(
                "Exact-enumeration covariance requires zero uncertainty."
            )
        covariance = _validate_covariance(
            proposed,
            size=total_count,
            name="QCT joint covariance",
        )

        content_hash = canonical_sha256(
            {
                "contract_id": "qct-joint-covariance-v3",
                "matrix_sha256": _array_sha256(covariance),
                "variable_labels": list(labels),
                "reference_distribution_hash": reference.content_hash,
                "coupled_distribution_hash": coupled.content_hash,
                "conditioned_profile_hash": profile.content_hash,
                "conditional_coupling_profile_hash": (
                    conditional_coupling.content_hash
                ),
                "relation": relation,
                "evidence_hash": evidence_hash,
            }
        )
        return cls(
            matrix=covariance,
            variable_labels=labels,
            reference_distribution_hash=reference.content_hash,
            coupled_distribution_hash=coupled.content_hash,
            conditioned_profile_hash=profile.content_hash,
            conditional_coupling_profile_hash=(
                conditional_coupling.content_hash
            ),
            relation=relation,
            evidence_hash=evidence_hash,
            content_hash=content_hash,
            _factory_token=_QCT_ARTIFACT_TOKEN,
        )

    @classmethod
    def block_diagonal(
        cls,
        *,
        reference: SoftOccupancyDistribution,
        coupled: SoftOccupancyDistribution,
        profile: ConditionedQCTProfile,
        conditional_coupling: ConditionalCouplingProfile,
        relation: str,
        evidence_hash: str,
    ) -> "QCTJointCovariance":
        count = len(profile.occupancies)
        if relation not in {
            "exact-enumeration",
            "independent-block-diagonal",
        }:
            raise ValueError(
                "block_diagonal cannot claim shared bootstrap provenance."
            )
        probability_count = count + 1
        total_count = 4 * count + 2
        matrix = np.zeros((total_count, total_count), dtype=float)
        matrix[:probability_count, :probability_count] = (
            reference.covariance_of_mean
        )
        matrix[
            probability_count : 2 * probability_count,
            probability_count : 2 * probability_count,
        ] = (
            coupled.covariance_of_mean
        )
        a_start = 2 * probability_count
        g_start = a_start + count
        matrix[a_start:g_start, a_start:g_start] = (
            profile.covariance_of_mean
        )
        matrix[g_start:, g_start:] = (
            conditional_coupling.covariance_of_mean
        )
        return cls.create(
            reference=reference,
            coupled=coupled,
            profile=profile,
            conditional_coupling=conditional_coupling,
            matrix=matrix,
            relation=relation,
            evidence_hash=evidence_hash,
        )


def _validate_ledger_compatibility(
    reference: SoftOccupancyDistribution,
    coupled: SoftOccupancyDistribution,
    profile: ConditionedQCTProfile,
    conditional_coupling: ConditionalCouplingProfile,
) -> None:
    if not isinstance(reference, SoftOccupancyDistribution):
        raise ValueError(
            "reference must be a SoftOccupancyDistribution."
        )
    if not isinstance(coupled, SoftOccupancyDistribution):
        raise ValueError(
            "coupled must be a SoftOccupancyDistribution."
        )
    if not isinstance(profile, ConditionedQCTProfile):
        raise ValueError("profile must be a ConditionedQCTProfile.")
    if not isinstance(conditional_coupling, ConditionalCouplingProfile):
        raise ValueError(
            "conditional_coupling must be a ConditionalCouplingProfile."
        )
    if reference.ensemble_role != "reference-product":
        raise ValueError(
            "reference distribution must use the reference-product ensemble."
        )
    if coupled.ensemble_role != "coupled-solution":
        raise ValueError(
            "coupled distribution must use the coupled-solution ensemble."
        )
    if not (
        reference.occupancies
        == coupled.occupancies
        == profile.occupancies
        == conditional_coupling.occupancies
    ):
        raise ValueError(
            "QCT occupancy supports must be identical and include n=0."
        )
    comparisons = (
        (
            "membership definition",
            reference.membership_definition_hash,
            coupled.membership_definition_hash,
            profile.membership_definition_hash,
            conditional_coupling.membership_definition_hash,
        ),
        (
            "observation volume",
            reference.observation_volume_hash,
            coupled.observation_volume_hash,
            profile.observation_volume_hash,
            conditional_coupling.observation_volume_hash,
        ),
        (
            "periodic boundary adapter",
            reference.boundary_adapter_hash,
            coupled.boundary_adapter_hash,
            profile.periodic_boundary_adapter_hash,
            conditional_coupling.periodic_boundary_adapter_hash,
        ),
        (
            "solute measure",
            reference.solute_measure_hash,
            coupled.solute_measure_hash,
            profile.solute_measure_hash,
            conditional_coupling.solute_measure_hash,
        ),
        (
            "water Hamiltonian",
            reference.water_hamiltonian_hash,
            coupled.water_hamiltonian_hash,
            profile.water_hamiltonian_hash,
            conditional_coupling.water_hamiltonian_hash,
        ),
        (
            "reference Hamiltonian",
            reference.system_hamiltonian_hash,
            reference.system_hamiltonian_hash,
            profile.reference_hamiltonian_hash,
            conditional_coupling.reference_hamiltonian_hash,
        ),
        (
            "coupled Hamiltonian",
            coupled.system_hamiltonian_hash,
            coupled.system_hamiltonian_hash,
            profile.coupled_hamiltonian_hash,
            conditional_coupling.coupled_hamiltonian_hash,
        ),
    )
    for label, *values in comparisons:
        if len(set(values)) != 1:
            raise ValueError(f"QCT {label} hashes do not match.")
    if not math.isclose(
        reference.temperature_k,
        coupled.temperature_k,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("QCT occupancy temperatures do not match.")
    if not math.isclose(
        reference.temperature_k,
        profile.temperature_k,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("QCT profile temperature does not match occupancy.")
    if not math.isclose(
        reference.temperature_k,
        conditional_coupling.temperature_k,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "QCT conditional-coupling temperature does not match occupancy."
        )
    if (
        profile.hamiltonian_bridge_hash
        != conditional_coupling.hamiltonian_bridge_hash
    ):
        raise ValueError("QCT Hamiltonian-bridge hashes do not match.")
    if (
        reference.boundary_conditions
        != coupled.boundary_conditions
    ):
        raise ValueError("QCT occupancy boundary conditions do not match.")
    if (
        profile.approximation_role == "maple-cluster-continuum-v3"
        and profile.boundary_conditions != "nonperiodic-cluster"
    ) or (
        profile.approximation_role == "periodic-explicit-reference"
        and profile.boundary_conditions != "periodic-3d"
    ):
        raise ValueError(
            "QCT profile boundary conditions do not match its approximation "
            "role."
        )


def _variance(
    gradient: np.ndarray,
    covariance: np.ndarray,
) -> float:
    value = float(gradient @ covariance @ gradient)
    if value < -1.0e-10:
        raise RuntimeError(
            "QCT_COVARIANCE_PROPAGATION_INVALID: negative propagated "
            "variance."
        )
    return max(value, 0.0)


def evaluate_soft_qct_ledger(
    *,
    reference: SoftOccupancyDistribution,
    coupled: SoftOccupancyDistribution,
    profile: ConditionedQCTProfile,
    conditional_coupling: ConditionalCouplingProfile,
    packing_bridge: PackingOccupancyBridge,
    joint_covariance: QCTJointCovariance,
    closure_abs_tolerance_kcal_mol: float,
    closure_z_max: float,
    bridge_abs_tolerance_kcal_mol: float,
    bridge_z_max: float,
    occupancy_abs_tolerance: float,
    tail_probability_max: float,
) -> dict[str, Any]:
    """Evaluate v3 multi-``n``, direct-``n=0`` and fixed-``n`` identities."""

    _validate_ledger_compatibility(
        reference,
        coupled,
        profile,
        conditional_coupling,
    )
    if not isinstance(packing_bridge, PackingOccupancyBridge):
        raise ValueError(
            "packing_bridge must be a PackingOccupancyBridge."
        )
    if (
        packing_bridge.reference_distribution_hash != reference.content_hash
        or packing_bridge.periodic_boundary_adapter_hash
        != reference.boundary_adapter_hash
        or not math.isclose(
            packing_bridge.p0_reference,
            float(reference.probabilities[0]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or packing_bridge.status != "passed"
    ):
        raise ValueError(
            "QCT ledger requires a passed packing bridge for this reference "
            "distribution."
        )
    if not isinstance(joint_covariance, QCTJointCovariance):
        raise ValueError(
            "joint_covariance must be a QCTJointCovariance."
        )
    if packing_bridge.role == "exact-enumeration":
        if joint_covariance.relation not in {
            "exact-enumeration",
            "independent-block-diagonal",
        }:
            raise ValueError(
                "Exact-enumeration packing evidence is restricted to the "
                "finite exact oracle or an explicitly independent "
                "downstream covariance."
            )
    elif packing_bridge.role != "shared-packing-occupancy-analysis":
        raise ValueError("Unsupported packing-occupancy bridge role.")
    expected_hashes = (
        (
            joint_covariance.reference_distribution_hash,
            reference.content_hash,
        ),
        (
            joint_covariance.coupled_distribution_hash,
            coupled.content_hash,
        ),
        (
            joint_covariance.conditioned_profile_hash,
            profile.content_hash,
        ),
        (
            joint_covariance.conditional_coupling_profile_hash,
            conditional_coupling.content_hash,
        ),
    )
    if any(observed != expected for observed, expected in expected_hashes):
        raise ValueError(
            "QCT joint covariance does not belong to the supplied artifacts."
        )
    for name, value, allow_zero in (
        (
            "closure_abs_tolerance_kcal_mol",
            closure_abs_tolerance_kcal_mol,
            False,
        ),
        ("closure_z_max", closure_z_max, False),
        (
            "bridge_abs_tolerance_kcal_mol",
            bridge_abs_tolerance_kcal_mol,
            False,
        ),
        ("bridge_z_max", bridge_z_max, False),
        (
            "occupancy_abs_tolerance",
            occupancy_abs_tolerance,
            False,
        ),
        ("tail_probability_max", tail_probability_max, False),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
        ):
            raise ValueError(f"{name} must be finite and positive.")

    count = len(profile.occupancies)
    probability_count = count + 1
    a_start = 2 * probability_count
    g_start = a_start + count
    total_count = 4 * count + 2
    p = reference.probabilities
    x = coupled.probabilities
    a = profile.free_energies_kcal_mol
    g = conditional_coupling.free_energies_kcal_mol
    covariance = joint_covariance.matrix
    rt = _R_KCAL_PER_MOL_K * reference.temperature_k
    beta = 1.0 / rt
    p0 = float(p[0])

    log_row_sum = float(logsumexp(-beta * a))
    packing = -rt * math.log(p0)
    coupled_active_mass = float(np.sum(x))
    multi_n = (
        packing
        - rt * log_row_sum
        + rt * math.log(coupled_active_mass)
    )
    fixed_n = a + rt * (np.log(x) - math.log(p0))
    reconstructed_x = np.exp(
        math.log(p0) + beta * (multi_n - a)
    )
    direct_n0 = float(fixed_n[0])

    gradient_multi = np.zeros(total_count)
    gradient_multi[0] = -rt / p0
    gradient_multi[
        probability_count : probability_count + count
    ] = rt / coupled_active_mass
    row_weights = np.exp(-beta * a - log_row_sum)
    gradient_multi[a_start:g_start] = row_weights
    multi_variance = _variance(gradient_multi, covariance)

    fixed_gradients = np.zeros((count, total_count))
    fixed_variances = np.zeros(count)
    closure_residuals = fixed_n - multi_n
    closure_standard_errors = np.zeros(count)
    for occupancy in range(count):
        gradient = fixed_gradients[occupancy]
        gradient[0] = -rt / p0
        gradient[probability_count + occupancy] = rt / x[occupancy]
        gradient[a_start + occupancy] = 1.0
        fixed_variances[occupancy] = _variance(gradient, covariance)
        residual_gradient = gradient - gradient_multi
        closure_standard_errors[occupancy] = math.sqrt(
            _variance(residual_gradient, covariance)
        )

    allowed_closure = np.maximum(
        float(closure_abs_tolerance_kcal_mol),
        float(closure_z_max) * closure_standard_errors,
    )
    occupancy_residuals = reconstructed_x - x
    exact_periodic_rows = g - rt * (
        np.log(p) - math.log(p0)
    )
    bridge_residuals = a - exact_periodic_rows
    bridge_standard_errors = np.zeros(count)
    for occupancy in range(count):
        gradient = np.zeros(total_count)
        gradient[a_start + occupancy] = 1.0
        gradient[g_start + occupancy] = -1.0
        gradient[occupancy] += rt / p[occupancy]
        gradient[0] -= rt / p0
        bridge_standard_errors[occupancy] = math.sqrt(
            _variance(gradient, covariance)
        )
    allowed_bridge = np.maximum(
        float(bridge_abs_tolerance_kcal_mol),
        float(bridge_z_max) * bridge_standard_errors,
    )
    failure_codes: list[str] = []
    if np.any(np.abs(closure_residuals) > allowed_closure):
        failure_codes.append("QCT_FIXED_N_CLOSURE_FAILED")
    if (
        abs(float(np.sum(reconstructed_x)) - coupled_active_mass)
        > float(occupancy_abs_tolerance)
        or np.max(np.abs(occupancy_residuals))
        > float(occupancy_abs_tolerance)
    ):
        failure_codes.append("QCT_OCCUPANCY_RECONSTRUCTION_FAILED")
    if np.any(np.abs(bridge_residuals) > allowed_bridge):
        failure_codes.append("QCT_PERIODIC_BRIDGE_FAILED")
    if (
        reference.tail_probability > float(tail_probability_max)
        or coupled.tail_probability > float(tail_probability_max)
    ):
        failure_codes.append("QCT_OCCUPANCY_TAIL_TOO_LARGE")

    packing_gradient = np.zeros(total_count)
    packing_gradient[0] = -rt / p0
    chemical_gradient = np.zeros(total_count)
    chemical_gradient[probability_count] = rt / x[0]
    n0_profile_gradient = np.zeros(total_count)
    n0_profile_gradient[a_start] = 1.0

    result_preimage = {
        "contract_id": "soft-qct-ledger-result-v3",
        "reference_distribution_hash": reference.content_hash,
        "coupled_distribution_hash": coupled.content_hash,
        "conditioned_profile_hash": profile.content_hash,
        "conditional_coupling_profile_hash": (
            conditional_coupling.content_hash
        ),
        "packing_bridge_hash": packing_bridge.content_hash,
        "joint_covariance_hash": joint_covariance.content_hash,
        "thresholds": {
            "closure_abs_tolerance_kcal_mol": float(
                closure_abs_tolerance_kcal_mol
            ),
            "closure_z_max": float(closure_z_max),
            "bridge_abs_tolerance_kcal_mol": float(
                bridge_abs_tolerance_kcal_mol
            ),
            "bridge_z_max": float(bridge_z_max),
            "occupancy_abs_tolerance": float(
                occupancy_abs_tolerance
            ),
            "tail_probability_max": float(tail_probability_max),
        },
    }
    return {
        "status": "passed" if not failure_codes else "failed",
        "failure_codes": failure_codes,
        "scientific_identity": (
            "mu=-RT ln(p0)-RT ln sum_active exp(-beta A_n)"
            "+RT ln(1-x_tail); "
            "mu_n=A_n+RT ln(x_n/p0)"
        ),
        "approximation_role": profile.approximation_role,
        "hamiltonian_bridge_hash": profile.hamiltonian_bridge_hash,
        "packing_bridge_hash": packing_bridge.content_hash,
        "profile_boundary_conditions": profile.boundary_conditions,
        "closure_interpretation": (
            "cluster-continuum bridge residual and falsification gate"
            if profile.approximation_role == "maple-cluster-continuum-v3"
            else (
                "exact enumerable Hamiltonian-bridge identity"
                if profile.boundary_conditions == "nonperiodic-cluster"
                else "exact same-Hamiltonian identity"
            )
        ),
        "occupancies": profile.occupancies,
        "packing_free_energy_kcal_mol": packing,
        "packing_standard_error_kcal_mol": math.sqrt(
            _variance(packing_gradient, covariance)
        ),
        "n0_conditioned_free_energy_kcal_mol": float(a[0]),
        "n0_conditioned_standard_error_kcal_mol": math.sqrt(
            _variance(n0_profile_gradient, covariance)
        ),
        "n0_chemical_free_energy_kcal_mol": rt * math.log(float(x[0])),
        "n0_chemical_standard_error_kcal_mol": math.sqrt(
            _variance(chemical_gradient, covariance)
        ),
        "direct_n0_free_energy_kcal_mol": direct_n0,
        "direct_n0_standard_error_kcal_mol": math.sqrt(
            fixed_variances[0]
        ),
        "multi_n_free_energy_kcal_mol": multi_n,
        "multi_n_standard_error_kcal_mol": math.sqrt(multi_variance),
        "fixed_n_free_energies_kcal_mol": fixed_n,
        "fixed_n_standard_errors_kcal_mol": np.sqrt(fixed_variances),
        "fixed_n_minus_multi_n_kcal_mol": closure_residuals,
        "fixed_n_minus_multi_n_standard_errors_kcal_mol": (
            closure_standard_errors
        ),
        "fixed_n_closure_allowed_kcal_mol": allowed_closure,
        "conditioned_row_logsum_weights": row_weights,
        "reconstructed_coupled_probabilities": reconstructed_x,
        "coupled_probability_residuals": occupancy_residuals,
        "reconstructed_probability_sum": float(
            np.sum(reconstructed_x)
        ),
        "coupled_active_probability_mass": coupled_active_mass,
        "tail_normalization_correction_kcal_mol": (
            rt * math.log(coupled_active_mass)
        ),
        "reference_tail_probability": reference.tail_probability,
        "coupled_tail_probability": coupled.tail_probability,
        "exact_periodic_conditioned_rows_kcal_mol": exact_periodic_rows,
        "cluster_minus_periodic_bridge_kcal_mol": bridge_residuals,
        "cluster_minus_periodic_bridge_standard_errors_kcal_mol": (
            bridge_standard_errors
        ),
        "cluster_minus_periodic_bridge_allowed_kcal_mol": allowed_bridge,
        "reference_distribution_hash": reference.content_hash,
        "coupled_distribution_hash": coupled.content_hash,
        "conditioned_profile_hash": profile.content_hash,
        "conditional_coupling_profile_hash": (
            conditional_coupling.content_hash
        ),
        "joint_covariance_hash": joint_covariance.content_hash,
        "ledger_result_hash": canonical_sha256(result_preimage),
        "thresholds": result_preimage["thresholds"],
    }


__all__ = [
    "BoundaryMeasureBridge",
    "ClusterQCTBuild",
    "ConditionalCouplingProfile",
    "ConditionedQCTProfile",
    "PackingOccupancyBridge",
    "QCTBootstrapEvidence",
    "QCTJointCovariance",
    "build_cluster_qct_profile",
    "build_periodic_qct_profile",
    "evaluate_soft_qct_ledger",
]
