from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .occupancy_analysis import SoftOccupancyDistribution
from .protocol import canonical_sha256
from .qct_ledger import (
    ConditionalCouplingProfile,
    ConditionedQCTProfile,
    build_periodic_qct_profile,
)


_R_KCAL_PER_MOL_K = 0.001987204258640831


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FiniteSoftQCTResult:
    """Exact enumeration used to falsify the Route A v3 ledger."""

    reference_distribution: SoftOccupancyDistribution
    coupled_distribution: SoftOccupancyDistribution
    conditioned_profile: ConditionedQCTProfile
    conditional_coupling_profile: ConditionalCouplingProfile
    exact_excess_free_energy_kcal_mol: float
    source_artifact_hash: str
    content_hash: str


@dataclass(frozen=True)
class FiniteLabeledClusterOracle:
    """Independent partition-product oracle for labeled cluster counting."""

    free_energies_kcal_mol: np.ndarray
    labeled_partition_weights: np.ndarray
    unlabeled_partition_weights: np.ndarray


def enumerate_labeled_cluster_partition(
    *,
    edge_coupling_boltzmann_factors: np.ndarray,
    edge_effective_volumes_angstrom3: np.ndarray,
    edge_release_boltzmann_factors: np.ndarray,
    outer_cluster_boltzmann_factors: np.ndarray,
    outer_water_boltzmann_factor: float,
    water_number_density_per_angstrom3: float,
    temperature_k: float,
) -> FiniteLabeledClusterOracle:
    """Enumerate primitive labeled products, then divide by ``n!`` once."""

    coupling = np.asarray(edge_coupling_boltzmann_factors, dtype=float)
    volumes = np.asarray(edge_effective_volumes_angstrom3, dtype=float)
    release = np.asarray(edge_release_boltzmann_factors, dtype=float)
    outer = np.asarray(outer_cluster_boltzmann_factors, dtype=float)
    edge_count = len(coupling)
    if (
        coupling.ndim != 1
        or edge_count < 1
        or volumes.shape != (edge_count,)
        or release.shape != (edge_count,)
        or outer.shape != (edge_count + 1,)
        or not np.all(np.isfinite(coupling))
        or not np.all(np.isfinite(volumes))
        or not np.all(np.isfinite(release))
        or not np.all(np.isfinite(outer))
        or np.any(coupling <= 0.0)
        or np.any(volumes <= 0.0)
        or np.any(release <= 0.0)
        or np.any(outer <= 0.0)
        or not math.isfinite(float(outer_water_boltzmann_factor))
        or float(outer_water_boltzmann_factor) <= 0.0
        or not math.isfinite(float(water_number_density_per_angstrom3))
        or float(water_number_density_per_angstrom3) <= 0.0
        or not math.isfinite(float(temperature_k))
        or float(temperature_k) <= 0.0
    ):
        raise ValueError(
            "Finite labeled-cluster oracle requires positive primitive "
            "partition factors."
        )
    density = float(water_number_density_per_angstrom3)
    labeled = np.empty(edge_count + 1, dtype=float)
    unlabeled = np.empty(edge_count + 1, dtype=float)
    for occupancy in range(edge_count + 1):
        labeled[occupancy] = (
            float(np.prod(coupling[:occupancy]))
            * float(np.prod(density * volumes[:occupancy]))
            * float(np.prod(release[:occupancy]))
            * float(outer[occupancy])
            / float(outer_water_boltzmann_factor) ** occupancy
        )
        unlabeled[occupancy] = labeled[occupancy] / math.factorial(
            occupancy
        )
    rt = _R_KCAL_PER_MOL_K * float(temperature_k)
    free_energies = -rt * np.log(unlabeled)
    for values in (labeled, unlabeled, free_energies):
        values.setflags(write=False)
    return FiniteLabeledClusterOracle(
        free_energies_kcal_mol=free_energies,
        labeled_partition_weights=labeled,
        unlabeled_partition_weights=unlabeled,
    )


def enumerate_finite_soft_qct(
    *,
    reference_energies_kcal_mol: np.ndarray,
    coupling_energies_kcal_mol: np.ndarray,
    soft_occupancy_weights: np.ndarray,
    temperature_k: float,
    membership_definition_hash: str,
    observation_volume_hash: str,
    solute_measure_hash: str,
    water_hamiltonian_hash: str,
    reference_hamiltonian_hash: str,
    coupled_hamiltonian_hash: str,
) -> FiniteSoftQCTResult:
    """Enumerate exact reference/coupled partitions and every occupancy row.

    For fractional soft occupancy ``omega_n(q)``,

    ``Z_{0,n}=sum_q omega_n(q) exp[-beta U_0(q)]`` and
    ``Z_{X,n}=sum_q omega_n(q) exp[-beta U_X(q)]``.

    The exact conditioned rows are

    ``A_n = -RT ln(Z_X,n/Z_0,n) - RT ln[p_n/p_0]``.
    """

    reference_energies = np.asarray(
        reference_energies_kcal_mol,
        dtype=float,
    )
    coupling_energies = np.asarray(
        coupling_energies_kcal_mol,
        dtype=float,
    )
    occupancy_weights = np.asarray(
        soft_occupancy_weights,
        dtype=float,
    )
    if (
        reference_energies.ndim != 1
        or len(reference_energies) < 2
        or coupling_energies.shape != reference_energies.shape
        or occupancy_weights.ndim != 2
        or occupancy_weights.shape[0] != len(reference_energies)
        or occupancy_weights.shape[1] < 2
        or not np.all(np.isfinite(reference_energies))
        or not np.all(np.isfinite(coupling_energies))
        or not np.all(np.isfinite(occupancy_weights))
        or np.any(occupancy_weights < 0.0)
        or np.any(occupancy_weights > 1.0)
        or not np.allclose(
            np.sum(occupancy_weights, axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            "Finite QCT enumeration requires finite reference/coupling "
            "energies and frame-normalized soft occupancy weights."
        )
    if (
        isinstance(temperature_k, (bool, np.bool_))
        or not math.isfinite(float(temperature_k))
        or float(temperature_k) <= 0.0
    ):
        raise ValueError("temperature_k must be finite and positive.")

    rt = _R_KCAL_PER_MOL_K * float(temperature_k)
    beta = 1.0 / rt
    log_reference_weights = -beta * reference_energies
    log_coupled_weights = -beta * (
        reference_energies + coupling_energies
    )
    log_z_reference = float(logsumexp(log_reference_weights))
    log_z_coupled = float(logsumexp(log_coupled_weights))

    occupancy_count = occupancy_weights.shape[1]
    log_z_reference_n = np.empty(occupancy_count)
    log_z_coupled_n = np.empty(occupancy_count)
    for occupancy in range(occupancy_count):
        column = occupancy_weights[:, occupancy]
        if not np.any(column > 0.0):
            raise ValueError(
                "Every finite QCT occupancy requires nonzero partition support."
            )
        with np.errstate(divide="ignore"):
            log_omega = np.log(column)
        log_z_reference_n[occupancy] = logsumexp(
            log_reference_weights + log_omega
        )
        log_z_coupled_n[occupancy] = logsumexp(
            log_coupled_weights + log_omega
        )

    reference_probabilities = np.exp(
        log_z_reference_n - log_z_reference
    )
    coupled_probabilities = np.exp(
        log_z_coupled_n - log_z_coupled
    )
    conditional_coupling = -rt * (
        log_z_coupled_n - log_z_reference_n
    )
    exact_excess = -rt * (log_z_coupled - log_z_reference)

    source_artifact_hash = canonical_sha256(
        {
            "contract_id": "finite-soft-qct-enumeration-v3",
            "reference_energies_sha256": _array_sha256(
                reference_energies
            ),
            "coupling_energies_sha256": _array_sha256(
                coupling_energies
            ),
            "soft_occupancy_weights_sha256": _array_sha256(
                occupancy_weights
            ),
            "temperature_k": float(temperature_k),
            "membership_definition_hash": membership_definition_hash,
            "observation_volume_hash": observation_volume_hash,
            "solute_measure_hash": solute_measure_hash,
            "water_hamiltonian_hash": water_hamiltonian_hash,
            "reference_hamiltonian_hash": reference_hamiltonian_hash,
            "coupled_hamiltonian_hash": coupled_hamiltonian_hash,
        }
    )
    bridge_hash = canonical_sha256(
        {
            "contract_id": "finite-soft-qct-identity-bridge-v3",
            "reference_hamiltonian_hash": reference_hamiltonian_hash,
            "coupled_hamiltonian_hash": coupled_hamiltonian_hash,
            "boundary_conditions": "periodic-3d",
        }
    )
    zero_probability_covariance = np.zeros(
        (occupancy_count + 1, occupancy_count + 1),
        dtype=float,
    )
    zero_energy_covariance = np.zeros(
        (occupancy_count, occupancy_count),
        dtype=float,
    )
    reference_distribution = SoftOccupancyDistribution.create(
        ensemble_role="reference-product",
        probabilities=reference_probabilities,
        tail_probability=0.0,
        covariance_of_mean=zero_probability_covariance,
        membership_definition_hash=membership_definition_hash,
        observation_volume_hash=observation_volume_hash,
        solute_measure_hash=solute_measure_hash,
        water_hamiltonian_hash=water_hamiltonian_hash,
        system_hamiltonian_hash=reference_hamiltonian_hash,
        temperature_k=temperature_k,
        boundary_conditions="periodic-3d",
        source_artifact_hash=source_artifact_hash,
        estimator="exact finite-state enumeration",
    )
    coupled_distribution = SoftOccupancyDistribution.create(
        ensemble_role="coupled-solution",
        probabilities=coupled_probabilities,
        tail_probability=0.0,
        covariance_of_mean=zero_probability_covariance,
        membership_definition_hash=membership_definition_hash,
        observation_volume_hash=observation_volume_hash,
        solute_measure_hash=solute_measure_hash,
        water_hamiltonian_hash=water_hamiltonian_hash,
        system_hamiltonian_hash=coupled_hamiltonian_hash,
        temperature_k=temperature_k,
        boundary_conditions="periodic-3d",
        source_artifact_hash=source_artifact_hash,
        estimator="exact finite-state enumeration",
    )
    conditional_coupling_profile = ConditionalCouplingProfile.create(
        free_energies_kcal_mol=conditional_coupling,
        covariance_of_mean=zero_energy_covariance,
        membership_definition_hash=membership_definition_hash,
        observation_volume_hash=observation_volume_hash,
        solute_measure_hash=solute_measure_hash,
        water_hamiltonian_hash=water_hamiltonian_hash,
        reference_hamiltonian_hash=reference_hamiltonian_hash,
        coupled_hamiltonian_hash=coupled_hamiltonian_hash,
        hamiltonian_bridge_hash=bridge_hash,
        temperature_k=temperature_k,
        source_artifact_hash=source_artifact_hash,
        estimator="exact finite enumeration of Z_X,n/Z_0,n",
    )
    conditioned_profile = build_periodic_qct_profile(
        reference=reference_distribution,
        conditional_coupling=conditional_coupling_profile,
        covariance_of_mean=zero_energy_covariance,
        approximation_role="exact-enumerable-bridge",
    )
    immutable_conditional = np.array(
        conditional_coupling,
        dtype=float,
        order="C",
        copy=True,
    )
    immutable_conditional.setflags(write=False)
    content_hash = canonical_sha256(
        {
            "contract_id": "finite-soft-qct-result-v3",
            "source_artifact_hash": source_artifact_hash,
            "reference_distribution_hash": (
                reference_distribution.content_hash
            ),
            "coupled_distribution_hash": coupled_distribution.content_hash,
            "conditioned_profile_hash": conditioned_profile.content_hash,
            "conditional_coupling_profile_hash": (
                conditional_coupling_profile.content_hash
            ),
            "conditional_coupling_sha256": _array_sha256(
                immutable_conditional
            ),
            "exact_excess_free_energy_kcal_mol": exact_excess,
        }
    )
    return FiniteSoftQCTResult(
        reference_distribution=reference_distribution,
        coupled_distribution=coupled_distribution,
        conditioned_profile=conditioned_profile,
        conditional_coupling_profile=conditional_coupling_profile,
        exact_excess_free_energy_kcal_mol=float(exact_excess),
        source_artifact_hash=source_artifact_hash,
        content_hash=content_hash,
    )


__all__ = [
    "FiniteLabeledClusterOracle",
    "FiniteSoftQCTResult",
    "enumerate_finite_soft_qct",
    "enumerate_labeled_cluster_partition",
]
