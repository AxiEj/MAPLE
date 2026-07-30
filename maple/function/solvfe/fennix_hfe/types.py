"""Typed, unit-explicit records for the FeNNix-Bio1 alchemical kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

FENNIX_GRAPH_SOFTCORE_PROVENANCE = "pinned_fennol_source_default"
FENNIX_REPULSION_POWER_PROVENANCE = "pinned_fennol_source_default"
FENNIX_REPULSION_SOFTCORE_PROVENANCE = (
    "maple_output_blind_reconstruction_not_paper_exact"
)
FENNIX_ALCHEMICAL_RECONSTRUCTION_SCOPE = (
    "maple_owned_softcore_reconstruction_mechanics_not_paper_reproduction"
)
FENNIX_KERNEL_SCIENTIFIC_SCOPE = (
    FENNIX_ALCHEMICAL_RECONSTRUCTION_SCOPE
    + "_not_hfe_not_accuracy_not_gpu_admission_not_performance"
)
FENNIX_REPULSION_NLH_COEFFICIENTS_PROVENANCE = (
    "pinned_nlh_coeffs_data_float32_origin_promoted_to_float64_for_model_execution"
)
FENNIX_PACKAGE_TREE_FILE_COUNT = 71
FENNIX_PACKAGE_TREE_SHA256 = (
    "b0ff2b138cdfa5f406b332ddcf12380bd981cee0be3986715827e6d7dacb4262"
)


@dataclass(frozen=True)
class FeNNixAlchemicalParameters:
    """Explicit MAPLE softcore reconstruction; not a paper-exact protocol."""

    graph_softcore_v_angstrom: float = 0.5
    repulsion_softcore_angstrom: float = 0.5
    repulsion_power_m: int = 2
    graph_softcore_provenance: str = field(
        default=FENNIX_GRAPH_SOFTCORE_PROVENANCE,
        init=False,
    )
    repulsion_power_provenance: str = field(
        default=FENNIX_REPULSION_POWER_PROVENANCE,
        init=False,
    )
    repulsion_softcore_provenance: str = field(
        default=FENNIX_REPULSION_SOFTCORE_PROVENANCE,
        init=False,
    )
    scientific_scope: str = field(
        default=FENNIX_ALCHEMICAL_RECONSTRUCTION_SCOPE,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in ("graph_softcore_v_angstrom", "repulsion_softcore_angstrom"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise ValueError(f"FeNNix {name} must be a finite positive number.")
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"FeNNix {name} must be a finite positive number.")
            object.__setattr__(self, name, value)
        power = self.repulsion_power_m
        if (
            isinstance(power, bool)
            or not isinstance(power, Integral)
            or int(power) <= 0
        ):
            raise ValueError("FeNNix repulsion_power_m must be a positive integer.")
        object.__setattr__(self, "repulsion_power_m", int(power))


@dataclass(frozen=True)
class FeNNixLambdaState:
    """Two-stage paper coupling state; all lambda values are dimensionless."""

    progress: float
    lambda_e: float
    lambda_v: float
    active_derivative: str
    active_derivative_scale: float

    @classmethod
    def from_progress(cls, progress: float) -> "FeNNixLambdaState":
        try:
            value = float(progress)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "FeNNix progress must be a finite number in [0, 1]."
            ) from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError("FeNNix progress must be a finite number in [0, 1].")
        if value <= 0.5:
            return cls(value, 0.0, 2.0 * value, "lambda_v", 2.0)
        return cls(value, 2.0 * value - 1.0, 1.0, "lambda_e", 2.0)

    def chain_derivative(
        self,
        denergy_dlambda_e: float,
        denergy_dlambda_v: float,
    ) -> float:
        """Return dE/dprogress for the active half-window."""

        active = (
            float(denergy_dlambda_v)
            if self.active_derivative == "lambda_v"
            else float(denergy_dlambda_e)
        )
        value = self.active_derivative_scale * active
        if not math.isfinite(value):
            raise ValueError("FeNNix progress derivative must be finite.")
        return value


@dataclass(frozen=True)
class FeNNixAlchemicalSystem:
    """One neutral solute in pure periodic water (angstrom input units)."""

    atomic_numbers: tuple[int, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    cell_angstrom: tuple[tuple[float, float, float], ...]
    molecule_ids: tuple[int, ...]
    solute_atom_indices: tuple[int, ...]
    pbc: tuple[bool, bool, bool] = (True, True, True)
    total_charge_e: int = 0
    multiplicity: int = 1
    solute_charge_e: int = 0
    solute_multiplicity: int = 1
    neighbor_skin_angstrom: float = 1.0


@dataclass(frozen=True)
class FeNNixParameterTreeReceipt:
    """Content fingerprints binding an unchanged checkpoint to its derived tree."""

    original_checkpoint_sha256: str
    original_parameter_fingerprint: str
    derived_parameter_fingerprint: str
    original_floating_dtypes: tuple[str, ...]
    derived_floating_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class FeNNixKernelIdentity:
    """Immutable provenance receipt for a derived in-memory float64 Hamiltonian."""

    checkpoint_sha256: str
    source_revision: str
    checkpoint_source_revision: str
    runtime_distribution: str
    runtime_version: str
    runtime_source_sha256: Mapping[str, str]
    runtime_package_versions: Mapping[str, str]
    fennol_package_tree_sha256: str
    fennol_package_tree_file_count: int
    original_parameter_tree_fingerprint: str
    derived_parameter_tree_fingerprint: str
    jax_enable_x64: bool
    matmul_precision: str
    tf32_enabled: bool
    repulsion_nlh_coefficients_provenance: str = (
        FENNIX_REPULSION_NLH_COEFFICIENTS_PROVENANCE
    )
    alchemical_parameters: FeNNixAlchemicalParameters = field(
        default_factory=FeNNixAlchemicalParameters
    )
    fixed_species_encoding_float64_sha256: str = ""
    model_id: str = "fennix-bio1"
    model_variant: str = "medium"
    energy_unit: str = "eV"
    length_unit: str = "angstrom"
    gpu_production_admitted: bool = False
    scientific_scope: str = FENNIX_KERNEL_SCIENTIFIC_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.alchemical_parameters, FeNNixAlchemicalParameters):
            raise TypeError(
                "FeNNix identity requires a FeNNixAlchemicalParameters contract."
            )
        if self.gpu_production_admitted:
            raise ValueError("FeNNix GPU production admission must remain false.")
        if (
            self.fennol_package_tree_sha256 != FENNIX_PACKAGE_TREE_SHA256
            or self.fennol_package_tree_file_count != FENNIX_PACKAGE_TREE_FILE_COUNT
        ):
            raise ValueError(
                "FeNNix identity must bind the exact pinned FeNNol package tree."
            )
        if (
            self.repulsion_nlh_coefficients_provenance
            != FENNIX_REPULSION_NLH_COEFFICIENTS_PROVENANCE
        ):
            raise ValueError(
                "FeNNix identity must disclose the pinned RepulsionNLH "
                "float32-origin coefficient construction."
            )
        if self.scientific_scope != FENNIX_KERNEL_SCIENTIFIC_SCOPE:
            raise ValueError(
                "FeNNix identity must retain the MAPLE reconstruction-only scope."
            )
        object.__setattr__(
            self,
            "runtime_source_sha256",
            MappingProxyType(dict(sorted(self.runtime_source_sha256.items()))),
        )
        object.__setattr__(
            self,
            "runtime_package_versions",
            MappingProxyType(dict(sorted(self.runtime_package_versions.items()))),
        )

    def verify_observed(self, observed: "FeNNixKernelIdentity") -> None:
        """Reject any drift from an expected frozen Hamiltonian identity."""

        checks = (
            ("checkpoint", self.checkpoint_sha256, observed.checkpoint_sha256),
            ("source", self.source_revision, observed.source_revision),
            (
                "checkpoint source",
                self.checkpoint_source_revision,
                observed.checkpoint_source_revision,
            ),
            ("runtime", self.runtime_distribution, observed.runtime_distribution),
            ("runtime", self.runtime_version, observed.runtime_version),
            (
                "runtime source",
                dict(self.runtime_source_sha256),
                dict(observed.runtime_source_sha256),
            ),
            (
                "runtime package",
                dict(self.runtime_package_versions),
                dict(observed.runtime_package_versions),
            ),
            (
                "FeNNol package tree",
                self.fennol_package_tree_sha256,
                observed.fennol_package_tree_sha256,
            ),
            (
                "FeNNol package tree file count",
                self.fennol_package_tree_file_count,
                observed.fennol_package_tree_file_count,
            ),
            (
                "original parameter fingerprint",
                self.original_parameter_tree_fingerprint,
                observed.original_parameter_tree_fingerprint,
            ),
            (
                "derived parameter fingerprint",
                self.derived_parameter_tree_fingerprint,
                observed.derived_parameter_tree_fingerprint,
            ),
            (
                "fixed species encoding fingerprint",
                self.fixed_species_encoding_float64_sha256,
                observed.fixed_species_encoding_float64_sha256,
            ),
            ("model", self.model_id, observed.model_id),
            ("model variant", self.model_variant, observed.model_variant),
            ("energy unit", self.energy_unit, observed.energy_unit),
            ("length unit", self.length_unit, observed.length_unit),
            (
                "RepulsionNLH coefficient provenance",
                self.repulsion_nlh_coefficients_provenance,
                observed.repulsion_nlh_coefficients_provenance,
            ),
            (
                "alchemical reconstruction",
                self.alchemical_parameters,
                observed.alchemical_parameters,
            ),
            (
                "scientific scope",
                self.scientific_scope,
                observed.scientific_scope,
            ),
        )
        for label, expected, actual in checks:
            if expected != actual:
                raise ValueError(f"FeNNix {label} identity mismatch.")
        if not observed.jax_enable_x64:
            raise ValueError("FeNNix identity requires JAX x64/float64.")
        if observed.matmul_precision.lower() != "highest":
            raise ValueError("FeNNix identity requires highest matmul precision.")
        if observed.tf32_enabled:
            raise ValueError("FeNNix identity forbids TF32.")

    def verify_compute_dtypes(self, dtypes: tuple[str, ...]) -> None:
        """Admit only explicit float64 floating-point compute observations."""

        normalized = tuple(str(value).strip().lower() for value in dtypes)
        forbidden = {"float16", "float32", "bfloat16", "tf32", "f16", "f32", "bf16"}
        if "float64" not in normalized or any(
            value in forbidden for value in normalized
        ):
            raise ValueError(
                "FeNNix compute dtype audit requires float64 and forbids TF32."
            )
        if any(
            value not in {"float64", "int32", "int64", "bool"} for value in normalized
        ):
            raise ValueError("FeNNix compute dtype audit encountered an unknown dtype.")


@dataclass(frozen=True)
class FeNNixKernelResult:
    """One scalar-Hamiltonian evaluation with explicit output units."""

    lambda_state: FeNNixLambdaState
    identity: FeNNixKernelIdentity
    energy_ev: float
    forces_ev_per_angstrom: tuple[tuple[float, float, float], ...]
    cell_gradient_ev_per_angstrom: tuple[tuple[float, float, float], ...]
    virial_ev: tuple[tuple[float, float, float], ...]
    denergy_dlambda_e_ev: float
    denergy_dlambda_v_ev: float
    denergy_dprogress_ev: float
    alchemical_parameters: FeNNixAlchemicalParameters = field(
        default_factory=FeNNixAlchemicalParameters
    )
    mechanics_validated: bool = True
    gpu_admitted: bool = False
    hfe_admitted: bool = False
    accuracy_admitted: bool = False
    performance_admitted: bool = False
    scope: str = FENNIX_KERNEL_SCIENTIFIC_SCOPE

    def __post_init__(self) -> None:
        if not self.mechanics_validated:
            raise ValueError("FeNNix kernel results must validate mechanics.")
        if any(
            (
                self.gpu_admitted,
                self.hfe_admitted,
                self.accuracy_admitted,
                self.performance_admitted,
            )
        ):
            raise ValueError(
                "FeNNix mechanics results cannot unlock GPU, HFE, accuracy, "
                "or performance admission."
            )
        if self.alchemical_parameters != self.identity.alchemical_parameters:
            raise ValueError(
                "FeNNix result alchemical parameters must match its kernel identity."
            )
        if self.scope != self.identity.scientific_scope:
            raise ValueError("FeNNix result scope must match its kernel identity.")

    @property
    def energy(self) -> float:
        return self.energy_ev

    @property
    def forces(self):
        return np.asarray(self.forces_ev_per_angstrom, dtype=np.float64)

    @property
    def cell_gradient(self):
        return np.asarray(self.cell_gradient_ev_per_angstrom, dtype=np.float64)

    @property
    def virial(self):
        return np.asarray(self.virial_ev, dtype=np.float64)

    @property
    def dE_dlambda_e(self) -> float:
        return self.denergy_dlambda_e_ev

    @property
    def dE_dlambda_v(self) -> float:
        return self.denergy_dlambda_v_ev

    @property
    def dE_dprogress(self) -> float:
        return self.denergy_dprogress_ev
