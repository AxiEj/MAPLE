from __future__ import annotations

import hashlib
import math
import re
from dataclasses import InitVar, dataclass
from typing import Any

import numpy as np
from ase.units import kB

from .analysis import ReducedPotentialTable
from .protocol import canonical_sha256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIMPLEX_TOLERANCE = 1.0e-10
_PSD_TOLERANCE = 1.0e-12
_OCCUPANCY_ARTIFACT_TOKEN = object()
_EXACT_OCCUPANCY_TOKEN = object()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _require_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash.")


def _require_pymbar() -> Any:
    try:
        import pymbar
    except ImportError as exc:
        raise RuntimeError(
            "Route A soft-occupancy analysis requires the optional "
            "'pymbar' dependency."
        ) from exc
    return pymbar


@dataclass(frozen=True)
class SoftOccupancyContract:
    """Scientific identity of one periodic soft-occupancy target ensemble."""

    ensemble_role: str
    target_state_index: int
    sampling_measure_id: str
    membership_definition_hash: str
    observation_volume_hash: str
    boundary_adapter_hash: str
    solute_measure_hash: str
    water_hamiltonian_hash: str
    system_hamiltonian_hash: str
    sample_independence_hash: str
    temperature_k: float
    boundary_conditions: str
    active_occupancy_max: int
    reconstruction_bin_index: int
    sample_independence_method: str = "subsampled-uncorrelated"
    target_state_label: str = "target"

    def __post_init__(self) -> None:
        if self.ensemble_role not in {
            "reference-product",
            "coupled-solution",
        }:
            raise ValueError(
                "ensemble_role must be reference-product or coupled-solution."
            )
        if (
            isinstance(self.target_state_index, (bool, np.bool_))
            or int(self.target_state_index) != self.target_state_index
            or int(self.target_state_index) < 0
        ):
            raise ValueError(
                "target_state_index must be a non-negative integer."
            )
        for name in (
            "sampling_measure_id",
            "target_state_label",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.sample_independence_method != "subsampled-uncorrelated":
            raise ValueError(
                "Soft-occupancy MBAR currently requires independently "
                "subsampled frames."
            )
        for name, value in (
            ("active_occupancy_max", self.active_occupancy_max),
            ("reconstruction_bin_index", self.reconstruction_bin_index),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        if int(self.active_occupancy_max) < 1:
            raise ValueError(
                "active_occupancy_max must include n=0 and at least n=1."
            )
        # The final bin is the aggregate n > active_occupancy_max tail.
        bin_count = int(self.active_occupancy_max) + 2
        if int(self.reconstruction_bin_index) >= bin_count:
            raise ValueError(
                "reconstruction_bin_index must identify an active or tail bin."
            )
        for name in (
            "membership_definition_hash",
            "observation_volume_hash",
            "boundary_adapter_hash",
            "solute_measure_hash",
            "water_hamiltonian_hash",
            "system_hamiltonian_hash",
            "sample_independence_hash",
        ):
            _require_hash(getattr(self, name), name)
        if (
            isinstance(self.temperature_k, (bool, np.bool_))
            or not math.isfinite(float(self.temperature_k))
            or float(self.temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")
        if self.boundary_conditions != "periodic-3d":
            raise ValueError(
                "Soft occupancy requires periodic-3d boundary conditions."
            )
        object.__setattr__(
            self,
            "target_state_index",
            int(self.target_state_index),
        )
        object.__setattr__(
            self,
            "active_occupancy_max",
            int(self.active_occupancy_max),
        )
        object.__setattr__(
            self,
            "reconstruction_bin_index",
            int(self.reconstruction_bin_index),
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(
            {
                "contract_id": "soft-occupancy-target-v3",
                "ensemble_role": self.ensemble_role,
                "target_state_index": self.target_state_index,
                "target_state_label": self.target_state_label,
                "sampling_measure_id": self.sampling_measure_id,
                "membership_definition_hash": (
                    self.membership_definition_hash
                ),
                "observation_volume_hash": self.observation_volume_hash,
                "boundary_adapter_hash": self.boundary_adapter_hash,
                "solute_measure_hash": self.solute_measure_hash,
                "water_hamiltonian_hash": self.water_hamiltonian_hash,
                "system_hamiltonian_hash": self.system_hamiltonian_hash,
                "sample_independence_hash": self.sample_independence_hash,
                "sample_independence_method": (
                    self.sample_independence_method
                ),
                "active_occupancy_max": self.active_occupancy_max,
                "tail_bin": f"n>{self.active_occupancy_max}",
                "reconstruction_bin_index": self.reconstruction_bin_index,
                "physical_solute_water_interactions": (
                    self.ensemble_role == "coupled-solution"
                ),
                "temperature_k": float(self.temperature_k),
                "boundary_conditions": self.boundary_conditions,
            }
        )


@dataclass(frozen=True)
class SoftOccupancyAnalysisInput:
    """MBAR table plus framewise ``omega_n`` values for one explicit target."""

    table: ReducedPotentialTable
    contract: SoftOccupancyContract
    occupancy_weights: np.ndarray
    content_hash: str

    @classmethod
    def create(
        cls,
        *,
        table: ReducedPotentialTable,
        contract: SoftOccupancyContract,
        occupancy_weights: np.ndarray,
    ) -> "SoftOccupancyAnalysisInput":
        if not isinstance(table, ReducedPotentialTable):
            raise ValueError("table must be a ReducedPotentialTable.")
        if not isinstance(contract, SoftOccupancyContract):
            raise ValueError("contract must be a SoftOccupancyContract.")
        if contract.target_state_index >= len(table.row_labels):
            raise ValueError(
                "target_state_index does not identify a table state."
            )
        if (
            table.row_labels[contract.target_state_index]
            != contract.target_state_label
        ):
            raise ValueError(
                "Soft-occupancy target state label does not match the table."
            )
        if table.measure_id != contract.sampling_measure_id:
            raise ValueError(
                "Soft-occupancy sampling measure does not match the table."
            )
        if table.boundary_conditions != contract.boundary_conditions:
            raise ValueError(
                "Soft-occupancy boundary conditions do not match the table."
            )
        expected_beta = 1.0 / (kB * float(contract.temperature_k))
        if not math.isclose(
            float(table.beta),
            expected_beta,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Soft-occupancy table beta does not match the temperature."
            )

        raw_weights = np.asarray(occupancy_weights, dtype=float)
        frame_count = table.u_kn.shape[1]
        if (
            raw_weights.ndim != 2
            or raw_weights.shape[0] != frame_count
            or raw_weights.shape[1]
            <= contract.active_occupancy_max + 1
            or not np.all(np.isfinite(raw_weights))
            or np.any(raw_weights < 0.0)
            or np.any(raw_weights > 1.0)
        ):
            raise ValueError(
                "occupancy_weights must be a finite frame-probability array "
                "that extends beyond the preregistered active support."
            )
        normalization = np.sum(raw_weights, axis=1)
        if not np.allclose(
            normalization,
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Framewise soft-occupancy normalization must equal one."
            )

        active_count = contract.active_occupancy_max + 1
        weights = np.column_stack(
            (
                raw_weights[:, :active_count],
                np.sum(raw_weights[:, active_count:], axis=1),
            )
        )
        if np.any(np.all(weights == 0.0, axis=0)):
            raise ValueError(
                "Every preregistered active/tail occupancy bin must have "
                "nonzero frame support."
            )
        immutable = np.array(weights, dtype=float, order="C", copy=True)
        immutable.setflags(write=False)
        content_hash = canonical_sha256(
            {
                "contract_id": "soft-occupancy-analysis-input-v3",
                "reduced_potential_table_hash": table.state_hash,
                "soft_occupancy_contract_hash": contract.content_hash,
                "raw_occupancy_weights_sha256": _array_sha256(raw_weights),
                "occupancy_weights_sha256": _array_sha256(immutable),
            }
        )
        return cls(
            table=table,
            contract=contract,
            occupancy_weights=immutable,
            content_hash=content_hash,
        )

    @property
    def target_state_index(self) -> int:
        return self.contract.target_state_index


@dataclass(frozen=True)
class SoftOccupancyDistribution:
    """Active ``p_tilde(n)``/``x_tilde(n)`` plus one normalized tail bin."""

    ensemble_role: str
    occupancies: tuple[int, ...]
    probabilities: np.ndarray
    tail_probability: float
    tail_lower_bound: int
    covariance_of_mean: np.ndarray
    membership_definition_hash: str
    observation_volume_hash: str
    boundary_adapter_hash: str
    solute_measure_hash: str
    water_hamiltonian_hash: str
    system_hamiltonian_hash: str
    temperature_k: float
    boundary_conditions: str
    source_artifact_hash: str
    estimator: str
    provenance_role: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _OCCUPANCY_ARTIFACT_TOKEN:
            raise ValueError(
                "SoftOccupancyDistribution must be constructed by its "
                "validated factory."
            )

    @classmethod
    def create(
        cls,
        *,
        ensemble_role: str,
        probabilities: np.ndarray,
        tail_probability: float,
        covariance_of_mean: np.ndarray,
        membership_definition_hash: str,
        observation_volume_hash: str,
        boundary_adapter_hash: str,
        solute_measure_hash: str,
        water_hamiltonian_hash: str,
        system_hamiltonian_hash: str,
        temperature_k: float,
        boundary_conditions: str,
        source_artifact_hash: str,
        estimator: str,
        _exact_source_token: object | None = None,
    ) -> "SoftOccupancyDistribution":
        if ensemble_role not in {"reference-product", "coupled-solution"}:
            raise ValueError(
                "ensemble_role must be reference-product or coupled-solution."
            )
        probabilities_array = np.asarray(probabilities, dtype=float)
        if (
            isinstance(tail_probability, (bool, np.bool_))
            or not math.isfinite(float(tail_probability))
            or not 0.0 <= float(tail_probability) < 1.0
        ):
            raise ValueError(
                "Soft-occupancy tail probability must be finite and in [0, 1)."
            )
        if (
            probabilities_array.ndim != 1
            or len(probabilities_array) < 2
            or not np.all(np.isfinite(probabilities_array))
            or np.any(probabilities_array <= 0.0)
            or not math.isclose(
                float(np.sum(probabilities_array)) + float(tail_probability),
                1.0,
                rel_tol=0.0,
                abs_tol=_SIMPLEX_TOLERANCE,
            )
        ):
            raise ValueError(
                "Soft-occupancy probabilities must be finite, strictly "
                "positive, and normalized together with the tail bin."
            )
        covariance_array = np.asarray(covariance_of_mean, dtype=float)
        count = len(probabilities_array)
        if (
            covariance_array.shape != (count + 1, count + 1)
            or not np.all(np.isfinite(covariance_array))
            or not np.allclose(
                covariance_array,
                covariance_array.T,
                rtol=0.0,
                atol=_PSD_TOLERANCE,
            )
        ):
            raise ValueError(
                "Soft-occupancy covariance must be a finite symmetric matrix."
            )
        symmetric = 0.5 * (covariance_array + covariance_array.T)
        if float(np.min(np.linalg.eigvalsh(symmetric))) < -_PSD_TOLERANCE:
            raise ValueError(
                "Soft-occupancy covariance must be positive semidefinite."
            )
        if np.max(np.abs(symmetric @ np.ones(count + 1))) > _SIMPLEX_TOLERANCE:
            raise ValueError(
                "Soft-occupancy covariance violates the simplex nullspace."
            )
        for name in (
            "membership_definition_hash",
            "observation_volume_hash",
            "boundary_adapter_hash",
            "solute_measure_hash",
            "water_hamiltonian_hash",
            "system_hamiltonian_hash",
            "source_artifact_hash",
        ):
            _require_hash(locals()[name], name)
        if (
            isinstance(temperature_k, (bool, np.bool_))
            or not math.isfinite(float(temperature_k))
            or float(temperature_k) <= 0.0
        ):
            raise ValueError("temperature_k must be finite and positive.")
        if boundary_conditions != "periodic-3d":
            raise ValueError(
                "Soft-occupancy distributions require periodic-3d boundaries."
            )
        if not isinstance(estimator, str) or not estimator:
            raise ValueError("estimator must be a non-empty string.")
        exact_estimator = estimator == "exact finite-state enumeration"
        if exact_estimator and _exact_source_token is not _EXACT_OCCUPANCY_TOKEN:
            raise ValueError(
                "Exact finite-state occupancy provenance is restricted to "
                "the internal finite-system oracle."
            )
        if not exact_estimator and _exact_source_token is not None:
            raise ValueError(
                "Exact-source provenance cannot be attached to an empirical "
                "occupancy estimator."
            )
        provenance_role = (
            "finite-state-oracle"
            if exact_estimator
            else "sampled-or-constructed"
        )

        immutable_probabilities = np.array(
            probabilities_array,
            dtype=float,
            order="C",
            copy=True,
        )
        immutable_covariance = np.array(
            symmetric,
            dtype=float,
            order="C",
            copy=True,
        )
        immutable_probabilities.setflags(write=False)
        immutable_covariance.setflags(write=False)
        occupancies = tuple(range(count))
        content_hash = canonical_sha256(
            {
                "contract_id": "soft-occupancy-distribution-v3",
                "ensemble_role": ensemble_role,
                "occupancies": list(occupancies),
                "tail_lower_bound": count,
                "tail_probability": float(tail_probability),
                "probabilities_sha256": _array_sha256(
                    immutable_probabilities
                ),
                "covariance_of_mean_sha256": _array_sha256(
                    immutable_covariance
                ),
                "membership_definition_hash": membership_definition_hash,
                "observation_volume_hash": observation_volume_hash,
                "boundary_adapter_hash": boundary_adapter_hash,
                "solute_measure_hash": solute_measure_hash,
                "water_hamiltonian_hash": water_hamiltonian_hash,
                "system_hamiltonian_hash": system_hamiltonian_hash,
                "temperature_k": float(temperature_k),
                "boundary_conditions": boundary_conditions,
                "source_artifact_hash": source_artifact_hash,
                "estimator": estimator,
                "provenance_role": provenance_role,
            }
        )
        return cls(
            ensemble_role=ensemble_role,
            occupancies=occupancies,
            probabilities=immutable_probabilities,
            tail_probability=float(tail_probability),
            tail_lower_bound=count,
            covariance_of_mean=immutable_covariance,
            membership_definition_hash=membership_definition_hash,
            observation_volume_hash=observation_volume_hash,
            boundary_adapter_hash=boundary_adapter_hash,
            solute_measure_hash=solute_measure_hash,
            water_hamiltonian_hash=water_hamiltonian_hash,
            system_hamiltonian_hash=system_hamiltonian_hash,
            temperature_k=float(temperature_k),
            boundary_conditions=boundary_conditions,
            source_artifact_hash=source_artifact_hash,
            estimator=estimator,
            provenance_role=provenance_role,
            content_hash=content_hash,
            _factory_token=_OCCUPANCY_ARTIFACT_TOKEN,
        )


@dataclass(frozen=True)
class SoftOccupancyEstimate:
    distribution: SoftOccupancyDistribution
    raw_covariance_of_mean: np.ndarray
    normalization_error: float
    covariance_projection_max_abs: float
    analysis_input_hash: str
    content_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _OCCUPANCY_ARTIFACT_TOKEN:
            raise ValueError(
                "SoftOccupancyEstimate must be constructed by the "
                "soft-occupancy estimator."
            )


def estimate_soft_occupancy(
    analysis_input: SoftOccupancyAnalysisInput,
) -> SoftOccupancyEstimate:
    """Estimate every soft occupancy and their full simplex covariance."""

    if not isinstance(analysis_input, SoftOccupancyAnalysisInput):
        raise ValueError(
            "analysis_input must be a SoftOccupancyAnalysisInput."
        )
    table = analysis_input.table
    target = analysis_input.target_state_index
    pymbar = _require_pymbar()
    estimator = pymbar.MBAR(
        table.u_kn,
        np.asarray(table.N_k, dtype=int),
        verbose=False,
    )
    # Estimate only K-1 independent simplex coordinates.  The omitted
    # coordinate is explicitly preregistered instead of silently choosing the
    # extremely small far-tail state in a large periodic water box.
    full_bin_count = analysis_input.occupancy_weights.shape[1]
    reconstruction_index = (
        analysis_input.contract.reconstruction_bin_index
    )
    independent_indices = tuple(
        index
        for index in range(full_bin_count)
        if index != reconstruction_index
    )
    independent_observables = np.array(
        analysis_input.occupancy_weights[:, independent_indices].T,
        dtype=float,
        order="C",
        copy=True,
    )
    result = estimator.compute_multiple_expectations(
        independent_observables,
        table.u_kn[target],
        compute_uncertainty=True,
        compute_covariance=True,
        return_theta=True,
    )
    independent_probabilities = np.atleast_1d(
        np.asarray(result["mu"], dtype=float)
    )
    independent_count = full_bin_count - 1
    theta = np.atleast_2d(np.asarray(result["Theta"], dtype=float))
    independent_covariance = np.array(
        theta[:independent_count, :independent_count]
        + theta[independent_count:, independent_count:]
        - theta[:independent_count, independent_count:]
        - theta[independent_count:, :independent_count],
        dtype=float,
        order="C",
        copy=True,
    )
    sigma = np.atleast_1d(np.asarray(result["sigma"], dtype=float))
    if (
        independent_probabilities.shape != (independent_count,)
        or sigma.shape != (independent_count,)
        or theta.shape != (2 * independent_count, 2 * independent_count)
        or independent_covariance.shape
        != (independent_count, independent_count)
        or not np.all(np.isfinite(independent_probabilities))
        or not np.all(np.isfinite(sigma))
        or not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(independent_covariance))
    ):
        raise RuntimeError(
            "SOFT_OCCUPANCY_ESTIMATE_INVALID: PyMBAR returned invalid shapes "
            "or non-finite values."
        )
    independent_covariance = 0.5 * (
        independent_covariance + independent_covariance.T
    )
    if not np.allclose(
        np.diag(independent_covariance),
        np.square(sigma),
        rtol=1.0e-10,
        atol=1.0e-15,
    ):
        raise RuntimeError(
            "SOFT_OCCUPANCY_COVARIANCE_INVALID: covariance reconstructed "
            "from PyMBAR Theta does not match its reported uncertainties."
        )

    reconstructed_probability = 1.0 - float(
        np.sum(independent_probabilities)
    )
    probabilities = np.empty(full_bin_count, dtype=float)
    probabilities[list(independent_indices)] = independent_probabilities
    probabilities[reconstruction_index] = reconstructed_probability
    normalization_error = abs(float(np.sum(probabilities)) - 1.0)
    if (
        normalization_error > _SIMPLEX_TOLERANCE
        or np.any(probabilities <= 0.0)
    ):
        raise RuntimeError(
            "SOFT_OCCUPANCY_NORMALIZATION_FAILED: estimated probabilities "
            "must be positive and sum to one."
        )
    probabilities = probabilities / np.sum(probabilities)

    count = full_bin_count
    simplex_transform = np.zeros((count, independent_count), dtype=float)
    for column, row in enumerate(independent_indices):
        simplex_transform[row, column] = 1.0
    simplex_transform[reconstruction_index, :] = -1.0
    raw_covariance = (
        simplex_transform
        @ independent_covariance
        @ simplex_transform.T
    )
    projector = np.eye(count) - np.ones((count, count)) / count
    symmetric = 0.5 * (raw_covariance + raw_covariance.T)
    projected = projector @ symmetric @ projector
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    if float(np.min(eigenvalues)) < -1.0e-9:
        raise RuntimeError(
            "SOFT_OCCUPANCY_COVARIANCE_INVALID: covariance is not positive "
            "semidefinite."
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * eigenvalues) @ eigenvectors.T
    projected = projector @ (0.5 * (projected + projected.T)) @ projector
    covariance_projection = float(
        np.max(np.abs(projected - raw_covariance))
    )
    if covariance_projection > 1.0e-8:
        raise RuntimeError(
            "SOFT_OCCUPANCY_COVARIANCE_PROJECTION_TOO_LARGE: PyMBAR "
            "covariance violates the probability simplex."
        )

    contract = analysis_input.contract
    distribution = SoftOccupancyDistribution.create(
        ensemble_role=contract.ensemble_role,
        probabilities=probabilities[:-1],
        tail_probability=float(probabilities[-1]),
        covariance_of_mean=projected,
        membership_definition_hash=contract.membership_definition_hash,
        observation_volume_hash=contract.observation_volume_hash,
        boundary_adapter_hash=contract.boundary_adapter_hash,
        solute_measure_hash=contract.solute_measure_hash,
        water_hamiltonian_hash=contract.water_hamiltonian_hash,
        system_hamiltonian_hash=contract.system_hamiltonian_hash,
        temperature_k=contract.temperature_k,
        boundary_conditions=contract.boundary_conditions,
        source_artifact_hash=analysis_input.content_hash,
        estimator=(
            "PyMBAR.compute_multiple_expectations with full covariance; "
            "preregistered active support plus aggregate tail bin; input "
            "samples bound to an independent-sample contract"
        ),
    )
    immutable_raw = np.array(
        raw_covariance,
        dtype=float,
        order="C",
        copy=True,
    )
    immutable_raw.setflags(write=False)
    content_hash = canonical_sha256(
        {
            "contract_id": "soft-occupancy-estimate-v3",
            "analysis_input_hash": analysis_input.content_hash,
            "distribution_hash": distribution.content_hash,
            "raw_covariance_sha256": _array_sha256(immutable_raw),
            "normalization_error": normalization_error,
            "covariance_projection_max_abs": covariance_projection,
        }
    )
    return SoftOccupancyEstimate(
        distribution=distribution,
        raw_covariance_of_mean=immutable_raw,
        normalization_error=normalization_error,
        covariance_projection_max_abs=covariance_projection,
        analysis_input_hash=analysis_input.content_hash,
        content_hash=content_hash,
        _factory_token=_OCCUPANCY_ARTIFACT_TOKEN,
    )


__all__ = [
    "SoftOccupancyAnalysisInput",
    "SoftOccupancyContract",
    "SoftOccupancyDistribution",
    "SoftOccupancyEstimate",
    "estimate_soft_occupancy",
]
