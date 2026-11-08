"""Immutable V4 frequency quantities derived from one identified Hessian."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.derivatives.computed_stationary import (
    _array_sha256,
    _assess_internal_operator,
)
from maple.solvation.models.base import (
    model_charge_and_multiplicity,
    model_input_sha256,
)


def _immutable_array(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"frequency quantity array must have finite shape {shape}.")
    result = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64)
    result = result.reshape(shape)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class CartesianCurvatureResult:
    """Full-3N eigensystem of the stored symmetrized Cartesian Hessian."""

    eigenvalues_eV_per_A2: np.ndarray
    eigenvectors_cartesian: np.ndarray
    normalized_eigenpair_residual: float
    eigenvector_orthogonality_defect: float
    computed_resolution_radius_eV_per_A2: float
    quality_gate: float
    computed_operator_quality_passed: bool
    raw_maximum_antisymmetry_eV_per_A2: float
    geometry_sha256: str
    model_input_sha256: str
    hessian_sha256: str
    stored_operator_sha256: str
    evaluation_sha256: str
    configuration_sha256: str
    scalar_contract_id: str
    provider_id: str | None
    profile_id: str | None
    verification_evidence_id: str | None
    computation_dtype: str
    device: str
    charge: int
    multiplicity: int
    eigenvalue_unit: str = "eV/angstrom^2"
    eigenvector_basis: str = "full_cartesian_3N"
    stored_operator: str = "symmetrized_cartesian_hessian"
    thermochemistry_available: bool = False
    independent_hessian_uncertainty: None = None
    interval_certified: None = None

    def __post_init__(self) -> None:
        dimension = np.asarray(self.eigenvalues_eV_per_A2).size
        object.__setattr__(
            self,
            "eigenvalues_eV_per_A2",
            _immutable_array(self.eigenvalues_eV_per_A2, shape=(dimension,)),
        )
        object.__setattr__(
            self,
            "eigenvectors_cartesian",
            _immutable_array(self.eigenvectors_cartesian, shape=(dimension, dimension)),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "quantity": "cartesian_curvature",
            "eigenvalues": self.eigenvalues_eV_per_A2.tolist(),
            "eigenvectors": self.eigenvectors_cartesian.tolist(),
            "eigenvalue_unit": self.eigenvalue_unit,
            "eigenvector_basis": self.eigenvector_basis,
            "stored_operator": self.stored_operator,
            "thermochemistry_available": self.thermochemistry_available,
            "computed_operator_qa": {
                "normalized_eigenpair_residual": self.normalized_eigenpair_residual,
                "eigenvector_orthogonality_defect": (
                    self.eigenvector_orthogonality_defect
                ),
                "computed_resolution_radius_eV_per_A2": (
                    self.computed_resolution_radius_eV_per_A2
                ),
                "quality_gate": self.quality_gate,
                "passed": self.computed_operator_quality_passed,
            },
            "raw_maximum_antisymmetry_eV_per_A2": (
                self.raw_maximum_antisymmetry_eV_per_A2
            ),
            "independent_hessian_uncertainty": self.independent_hessian_uncertainty,
            "interval_certified": self.interval_certified,
            "geometry_sha256": self.geometry_sha256,
            "model_input_sha256": self.model_input_sha256,
            "hessian_sha256": self.hessian_sha256,
            "stored_operator_sha256": self.stored_operator_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "configuration_sha256": self.configuration_sha256,
            "scalar_contract_id": self.scalar_contract_id,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "verification_evidence_id": self.verification_evidence_id,
            "computation_dtype": self.computation_dtype,
            "device": self.device,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
        }


@dataclass(frozen=True, slots=True)
class PhysicalMassWeightedResult:
    """Existing physical internal-mode analysis and RRHO result, copied by value."""

    frequencies_cm1: np.ndarray
    modes_cartesian: np.ndarray
    eigenvalues_eV_per_A2_amu: np.ndarray
    rigid_rank: int
    interval_statuses: tuple[str, ...]
    uncertainty_eV_per_A2_amu: float | None
    thermochemistry: tuple[tuple[str, float], ...]
    masses_amu: np.ndarray
    mass_sha256: str
    hessian_sha256: str
    evaluation_sha256: str
    temperature_K: float
    pressure_kPa: float
    ilowfreq: int
    omega0_cm1: float
    nu_floor_cm1: float
    symmetry_number: int
    mode_basis: str = "cartesian_displacements"
    mode_normalization: str = "Q.T@M@Q=I"
    thermochemistry_available: bool = True

    def __post_init__(self) -> None:
        mode_count = np.asarray(self.frequencies_cm1).size
        cartesian_dimension = np.asarray(self.modes_cartesian).shape[1]
        object.__setattr__(
            self,
            "frequencies_cm1",
            _immutable_array(self.frequencies_cm1, shape=(mode_count,)),
        )
        object.__setattr__(
            self,
            "modes_cartesian",
            _immutable_array(
                self.modes_cartesian, shape=(mode_count, cartesian_dimension)
            ),
        )
        object.__setattr__(
            self,
            "eigenvalues_eV_per_A2_amu",
            _immutable_array(self.eigenvalues_eV_per_A2_amu, shape=(mode_count,)),
        )
        object.__setattr__(
            self,
            "masses_amu",
            _immutable_array(self.masses_amu, shape=(cartesian_dimension // 3,)),
        )
        if _array_sha256(self.masses_amu) != self.mass_sha256:
            raise ValueError("mass_sha256 does not match masses_amu.")

    def to_record(self) -> dict[str, Any]:
        return {
            "quantity": "physical_mass_weighted_frequency",
            "frequencies_cm1": self.frequencies_cm1.tolist(),
            "frequency_unit": "cm^-1",
            "modes_cartesian": self.modes_cartesian.tolist(),
            "eigenvalues_eV_per_A2_amu": self.eigenvalues_eV_per_A2_amu.tolist(),
            "eigenvalue_unit": "eV/(angstrom^2*amu)",
            "rigid_rank": self.rigid_rank,
            "interval_statuses": list(self.interval_statuses),
            "uncertainty_eV_per_A2_amu": self.uncertainty_eV_per_A2_amu,
            "thermochemistry_available": self.thermochemistry_available,
            "thermochemistry": dict(self.thermochemistry),
            "thermochemistry_scope": (
                "ideal_gas_rrho_correction_not_full_solution_free_energy"
            ),
            "masses_amu": self.masses_amu.tolist(),
            "mass_sha256": self.mass_sha256,
            "hessian_sha256": self.hessian_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "temperature_K": self.temperature_K,
            "pressure_kPa": self.pressure_kPa,
            "ilowfreq": self.ilowfreq,
            "omega0_cm1": self.omega0_cm1,
            "nu_floor_cm1": self.nu_floor_cm1,
            "symmetry_number": self.symmetry_number,
            "mode_basis": self.mode_basis,
            "mode_normalization": self.mode_normalization,
        }


@dataclass(frozen=True, slots=True)
class FrequencyQuantityResult:
    """V4-owned result for Cartesian curvature or combined frequency quantities."""

    canonical_method: str
    cartesian_curvature: CartesianCurvatureResult
    physical_mass_weighted: PhysicalMassWeightedResult | None = None
    deprecated_alias: str | None = None

    def __post_init__(self) -> None:
        if self.canonical_method == "cartesian-curvature":
            if self.physical_mass_weighted is not None:
                raise ValueError("cartesian-curvature forbids physical MW quantities.")
            if self.deprecated_alias not in {None, "nonmw"}:
                raise ValueError("cartesian-curvature alias must be nonmw or absent.")
        elif self.canonical_method == "both":
            if self.physical_mass_weighted is None:
                raise ValueError("both requires physical MW quantities.")
            if self.deprecated_alias is not None:
                raise ValueError("both does not accept a deprecated alias.")
            if (
                self.physical_mass_weighted.hessian_sha256
                != self.cartesian_curvature.hessian_sha256
                or self.physical_mass_weighted.evaluation_sha256
                != self.cartesian_curvature.evaluation_sha256
            ):
                raise ValueError(
                    "both requires physical and Cartesian quantities from one Hessian."
                )
        else:
            raise ValueError("Unsupported frequency quantity result variant.")

    def to_final_metrics(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "canonical_method": self.canonical_method,
            "deprecated_alias": self.deprecated_alias,
            "cartesian_curvature": self.cartesian_curvature.to_record(),
        }
        if self.physical_mass_weighted is not None:
            record["physical_mass_weighted"] = self.physical_mass_weighted.to_record()
        return record


def analyze_cartesian_curvature(atoms, context) -> CartesianCurvatureResult:
    """Analyze the stored Cartesian Hessian without mass weighting or projection."""

    from maple.function.calculator.route2 import is_identified_hessian_evaluation

    if not is_identified_hessian_evaluation(context):
        raise TypeError(
            "Cartesian curvature requires a calculator-issued Hessian context."
        )
    evaluation = context.evaluation
    dimension = 3 * len(atoms)
    raw_hessian = _immutable_array(
        evaluation.hessian_eV_per_A2, shape=(dimension, dimension)
    )
    if evaluation.geometry_sha256 != geometry_sha256(atoms):
        raise ValueError("Hessian geometry identity does not match atoms.")
    current_model_input = model_input_sha256(atoms)
    charge, multiplicity = model_charge_and_multiplicity(atoms)
    if (
        context.model_input_sha256 != current_model_input
        or context.charge != charge
        or context.multiplicity != multiplicity
        or context.configuration_sha256 != evaluation.configuration_sha256
        or context.profile_id != evaluation.profile_id
        or context.scalar_contract_id != evaluation.scalar_contract_id
        or context.device != evaluation.device
    ):
        raise ValueError("identified Hessian context does not match its evaluation.")
    symmetric_hessian = 0.5 * (raw_hessian + raw_hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_hessian)
    qa = _assess_internal_operator(
        symmetric_hessian,
        rigid_internal_overlap=np.zeros((0, dimension), dtype=np.float64),
        eigensystem=(eigenvalues, eigenvectors),
        cartesian_dimension=dimension,
    )
    return CartesianCurvatureResult(
        eigenvalues_eV_per_A2=qa.computed_eigenvalues,
        eigenvectors_cartesian=eigenvectors,
        normalized_eigenpair_residual=qa.normalized_eigenpair_residual,
        eigenvector_orthogonality_defect=qa.eigenvector_orthogonality_defect,
        computed_resolution_radius_eV_per_A2=qa.computed_resolution_radius,
        quality_gate=qa.quality_gate,
        computed_operator_quality_passed=qa.computed_operator_quality_passed,
        raw_maximum_antisymmetry_eV_per_A2=(evaluation.maximum_antisymmetry_eV_per_A2),
        geometry_sha256=evaluation.geometry_sha256,
        model_input_sha256=current_model_input,
        hessian_sha256=_array_sha256(raw_hessian),
        stored_operator_sha256=_array_sha256(symmetric_hessian),
        evaluation_sha256=evaluation.evaluation_sha256,
        configuration_sha256=evaluation.configuration_sha256,
        scalar_contract_id=evaluation.scalar_contract_id,
        provider_id=evaluation.provider_id,
        profile_id=evaluation.profile_id,
        verification_evidence_id=evaluation.verification_evidence_id,
        computation_dtype=evaluation.computation_dtype,
        device=evaluation.device,
        charge=charge,
        multiplicity=multiplicity,
    )
