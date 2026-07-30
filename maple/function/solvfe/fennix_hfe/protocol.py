"""Frozen MAPLE FeNNix fixed-window TI protocol contract.

This module records an output-blind MAPLE reconstruction.  It is deliberately
not represented as a reproduction of the unpublished FeNNix HFE run inputs and
contains no sampler or experimental observations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .types import FeNNixAlchemicalParameters, FeNNixKernelIdentity


class FeNNixProtocolProvenance(str, Enum):
    """Allowed evidence categories for every frozen protocol field."""

    PAPER_EXPLICIT = "paper_explicit"
    SOURCE_DEFAULT = "source_default"
    METHOD_REFERENCE = "method_reference"
    MAPLE_RECONSTRUCTION = "maple_reconstruction"


_RECONSTRUCTION = FeNNixProtocolProvenance.MAPLE_RECONSTRUCTION
_PAPER = FeNNixProtocolProvenance.PAPER_EXPLICIT
_METHOD = FeNNixProtocolProvenance.METHOD_REFERENCE
_SOURCE = FeNNixProtocolProvenance.SOURCE_DEFAULT

_REPULSION_LAMBDA_V = tuple(index / 10.0 for index in range(11))
_ELECTRONIC_LAMBDA_E = tuple(index / 10.0 for index in range(11))
_SENSITIVITY_REPULSION_ALPHA_ANGSTROM = (0.4, 0.6)
_PROTOCOL_SCOPE = (
    "maple_owned_fixed_window_ti_reconstruction_not_fennix_paper_reproduction"
)


def _canonical_value(value: Any, *, path: str) -> Any:
    """Convert immutable protocol state to deterministic finite JSON values."""

    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_value(
                getattr(value, item.name), path=f"{path}.{item.name}"
            )
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key)
            if normalized in result:
                raise ValueError(f"{path} contains colliding mapping keys.")
            result[normalized] = _canonical_value(item, path=f"{path}.{normalized}")
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (tuple, list)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains non-canonical state {type(value).__name__!r}.")


def _require_exact(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"FeNNix HFE protocol field {name} is frozen at {expected!r}; "
            f"received {observed!r}."
        )


def _require_finite_real(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"FeNNix HFE protocol field {name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        raise ValueError(
            f"FeNNix HFE protocol field {name} must be finite and positive."
        )
    return normalized


def _require_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(
            f"FeNNix HFE protocol field {name} must be a positive integer."
        )
    return int(value)


@dataclass(frozen=True)
class FeNNixHFEProtocol:
    """Immutable fixed-window TI reconstruction bound to one exact kernel.

    The two native-lambda legs share the state ``(lambda_e=0, lambda_v=1)``.
    All admission flags are intentionally closed; this object defines work to
    execute and quality gates to evaluate, not a successful HFE result.
    """

    kernel_identity: FeNNixKernelIdentity
    protocol_scope: str = _PROTOCOL_SCOPE
    estimator: str = "fixed_window_thermodynamic_integration"
    repulsion_leg_lambda_v: tuple[float, ...] = _REPULSION_LAMBDA_V
    repulsion_leg_lambda_e: float = 0.0
    electronic_leg_lambda_e: tuple[float, ...] = _ELECTRONIC_LAMBDA_E
    electronic_leg_lambda_v: float = 1.0
    shared_native_lambda_state: tuple[float, float] = (0.0, 1.0)
    timestep_femtoseconds: float = 1.0
    integrator: str = "BAOAB_Langevin"
    temperature_kelvin: float = 298.15
    ensemble: str = "NVT"
    langevin_friction_per_picosecond: float = 1.0
    equilibration_picoseconds: float = 400.0
    production_picoseconds: float = 4000.0
    observable_interval_picoseconds: float = 0.1
    restart_interval_picoseconds: float = 1.0
    walker_count: int = 3
    window_orders: tuple[str, ...] = ("ascending", "descending")
    random_seeds: tuple[int, ...] = (2026073101, 2026073102, 2026073103)
    constraints_enabled: bool = False
    barostat_enabled: bool = False
    runtime_com_projection_enabled: bool = False
    ev_to_kcal_per_mol: float = 23.06054783061903

    minimum_effective_samples_per_window_walker: float = 200.0
    minimum_independent_equivalent_blocks: int = 20
    minimum_block_length_picoseconds: float = 100.0
    block_length_tau_int_multiplier: float = 5.0
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 2026073199
    production_half_mean_max_combined_se: float = 2.0
    full_vs_final_half_ti_abs_kcal_per_mol: float = 0.20
    full_vs_final_half_ti_se_multiplier: float = 1.0
    simpson_vs_trapezoid_leg_abs_kcal_per_mol: float = 0.10
    simpson_vs_trapezoid_leg_se_multiplier: float = 0.25
    minimum_adjacent_bidirectional_overlap: float = 0.03
    minimum_directional_reweighting_ess: float = 50.0
    disconnected_window_pair_forbidden: bool = True
    bar_vs_ti_abs_kcal_per_mol: float = 0.20
    bar_vs_ti_combined_se_multiplier: float = 2.0
    all_walkers_must_pass: bool = True
    walker_pairwise_max_z: float = 2.0
    walker_max_span_kcal_per_mol: float = 0.50
    ascending_vs_descending_abs_kcal_per_mol: float = 0.20
    ascending_vs_descending_combined_se_multiplier: float = 2.0

    endpoint_energy_absolute_tolerance_ev: float = 1.0e-8
    endpoint_energy_relative_tolerance: float = 1.0e-10
    endpoint_force_tolerance_ev_per_angstrom: float = 1.0e-7
    p1_ordinary_energy_force_identity_required: bool = True
    p0_factorization_identity_required: bool = True
    analytic_lambda_finite_difference_gate_required: bool = True
    softcore_must_vanish_at_lambda_v_one: bool = True
    reaction_forbidden: bool = True
    proton_transfer_forbidden: bool = True
    molecular_identity_change_forbidden: bool = True
    water_connectivity_change_forbidden: bool = True

    sensitivity_graph_softcore_angstrom: float = 0.5
    sensitivity_repulsion_softcore_variants_angstrom: tuple[float, ...] = (
        _SENSITIVITY_REPULSION_ALPHA_ANGSTROM
    )
    sensitivity_variants_output_blind: bool = True
    sensitivity_experimental_selection_forbidden: bool = True
    sensitivity_variants_must_be_finite: bool = True
    sensitivity_variants_must_pass_all_gates: bool = True
    sensitivity_total_hfe_shift_abs_kcal_per_mol: float = 0.20
    sensitivity_total_hfe_shift_combined_se_multiplier: float = 2.0

    hfe_admitted: bool = False
    accuracy_admitted: bool = False
    gpu_admitted: bool = False
    performance_admitted: bool = False

    FIELD_PROVENANCE: ClassVar[Mapping[str, FeNNixProtocolProvenance]] = (
        MappingProxyType(
            {
                "kernel_identity": _SOURCE,
                "protocol_scope": _RECONSTRUCTION,
                "estimator": _METHOD,
                "repulsion_leg_lambda_v": _RECONSTRUCTION,
                "repulsion_leg_lambda_e": _PAPER,
                "electronic_leg_lambda_e": _RECONSTRUCTION,
                "electronic_leg_lambda_v": _PAPER,
                "shared_native_lambda_state": _PAPER,
                "timestep_femtoseconds": _PAPER,
                "integrator": _PAPER,
                "temperature_kelvin": _RECONSTRUCTION,
                "ensemble": _RECONSTRUCTION,
                "langevin_friction_per_picosecond": _RECONSTRUCTION,
                "equilibration_picoseconds": _RECONSTRUCTION,
                "production_picoseconds": _RECONSTRUCTION,
                "observable_interval_picoseconds": _RECONSTRUCTION,
                "restart_interval_picoseconds": _RECONSTRUCTION,
                "walker_count": _RECONSTRUCTION,
                "window_orders": _RECONSTRUCTION,
                "random_seeds": _RECONSTRUCTION,
                "constraints_enabled": _RECONSTRUCTION,
                "barostat_enabled": _RECONSTRUCTION,
                "runtime_com_projection_enabled": _RECONSTRUCTION,
                "ev_to_kcal_per_mol": _METHOD,
                "minimum_effective_samples_per_window_walker": _RECONSTRUCTION,
                "minimum_independent_equivalent_blocks": _RECONSTRUCTION,
                "minimum_block_length_picoseconds": _RECONSTRUCTION,
                "block_length_tau_int_multiplier": _METHOD,
                "bootstrap_replicates": _RECONSTRUCTION,
                "bootstrap_seed": _RECONSTRUCTION,
                "production_half_mean_max_combined_se": _RECONSTRUCTION,
                "full_vs_final_half_ti_abs_kcal_per_mol": _RECONSTRUCTION,
                "full_vs_final_half_ti_se_multiplier": _RECONSTRUCTION,
                "simpson_vs_trapezoid_leg_abs_kcal_per_mol": _RECONSTRUCTION,
                "simpson_vs_trapezoid_leg_se_multiplier": _RECONSTRUCTION,
                "minimum_adjacent_bidirectional_overlap": _METHOD,
                "minimum_directional_reweighting_ess": _METHOD,
                "disconnected_window_pair_forbidden": _METHOD,
                "bar_vs_ti_abs_kcal_per_mol": _RECONSTRUCTION,
                "bar_vs_ti_combined_se_multiplier": _RECONSTRUCTION,
                "all_walkers_must_pass": _RECONSTRUCTION,
                "walker_pairwise_max_z": _RECONSTRUCTION,
                "walker_max_span_kcal_per_mol": _RECONSTRUCTION,
                "ascending_vs_descending_abs_kcal_per_mol": _RECONSTRUCTION,
                "ascending_vs_descending_combined_se_multiplier": _RECONSTRUCTION,
                "endpoint_energy_absolute_tolerance_ev": _RECONSTRUCTION,
                "endpoint_energy_relative_tolerance": _RECONSTRUCTION,
                "endpoint_force_tolerance_ev_per_angstrom": _RECONSTRUCTION,
                "p1_ordinary_energy_force_identity_required": _PAPER,
                "p0_factorization_identity_required": _PAPER,
                "analytic_lambda_finite_difference_gate_required": _METHOD,
                "softcore_must_vanish_at_lambda_v_one": _PAPER,
                "reaction_forbidden": _RECONSTRUCTION,
                "proton_transfer_forbidden": _RECONSTRUCTION,
                "molecular_identity_change_forbidden": _RECONSTRUCTION,
                "water_connectivity_change_forbidden": _RECONSTRUCTION,
                "sensitivity_graph_softcore_angstrom": _SOURCE,
                "sensitivity_repulsion_softcore_variants_angstrom": _RECONSTRUCTION,
                "sensitivity_variants_output_blind": _RECONSTRUCTION,
                "sensitivity_experimental_selection_forbidden": _RECONSTRUCTION,
                "sensitivity_variants_must_be_finite": _RECONSTRUCTION,
                "sensitivity_variants_must_pass_all_gates": _RECONSTRUCTION,
                "sensitivity_total_hfe_shift_abs_kcal_per_mol": _RECONSTRUCTION,
                "sensitivity_total_hfe_shift_combined_se_multiplier": _RECONSTRUCTION,
                "hfe_admitted": _RECONSTRUCTION,
                "accuracy_admitted": _RECONSTRUCTION,
                "gpu_admitted": _RECONSTRUCTION,
                "performance_admitted": _RECONSTRUCTION,
            }
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_identity, FeNNixKernelIdentity):
            raise TypeError("FeNNix HFE protocol requires a FeNNixKernelIdentity.")
        self.kernel_identity.verify_observed(self.kernel_identity)
        if self.kernel_identity.model_id != "fennix-bio1":
            raise ValueError("FeNNix HFE protocol requires the FeNNix-Bio1 model.")
        if self.kernel_identity.model_variant not in {"small", "medium"}:
            raise ValueError(
                "FeNNix HFE protocol requires a pinned Bio1 S or M variant."
            )
        if (
            self.kernel_identity.energy_unit != "eV"
            or self.kernel_identity.length_unit != "angstrom"
        ):
            raise ValueError("FeNNix HFE protocol requires eV/angstrom kernel units.")
        if self.kernel_identity.alchemical_parameters != FeNNixAlchemicalParameters():
            raise ValueError(
                "FeNNix HFE base protocol requires the frozen 0.5 angstrom, m=2 "
                "MAPLE alchemical reconstruction."
            )

        expected = {
            "protocol_scope": _PROTOCOL_SCOPE,
            "estimator": "fixed_window_thermodynamic_integration",
            "repulsion_leg_lambda_v": _REPULSION_LAMBDA_V,
            "repulsion_leg_lambda_e": 0.0,
            "electronic_leg_lambda_e": _ELECTRONIC_LAMBDA_E,
            "electronic_leg_lambda_v": 1.0,
            "shared_native_lambda_state": (0.0, 1.0),
            "timestep_femtoseconds": 1.0,
            "integrator": "BAOAB_Langevin",
            "temperature_kelvin": 298.15,
            "ensemble": "NVT",
            "langevin_friction_per_picosecond": 1.0,
            "equilibration_picoseconds": 400.0,
            "production_picoseconds": 4000.0,
            "observable_interval_picoseconds": 0.1,
            "restart_interval_picoseconds": 1.0,
            "walker_count": 3,
            "window_orders": ("ascending", "descending"),
            "random_seeds": (2026073101, 2026073102, 2026073103),
            "constraints_enabled": False,
            "barostat_enabled": False,
            "runtime_com_projection_enabled": False,
            "ev_to_kcal_per_mol": 23.06054783061903,
            "minimum_effective_samples_per_window_walker": 200.0,
            "minimum_independent_equivalent_blocks": 20,
            "minimum_block_length_picoseconds": 100.0,
            "block_length_tau_int_multiplier": 5.0,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 2026073199,
            "production_half_mean_max_combined_se": 2.0,
            "full_vs_final_half_ti_abs_kcal_per_mol": 0.20,
            "full_vs_final_half_ti_se_multiplier": 1.0,
            "simpson_vs_trapezoid_leg_abs_kcal_per_mol": 0.10,
            "simpson_vs_trapezoid_leg_se_multiplier": 0.25,
            "minimum_adjacent_bidirectional_overlap": 0.03,
            "minimum_directional_reweighting_ess": 50.0,
            "disconnected_window_pair_forbidden": True,
            "bar_vs_ti_abs_kcal_per_mol": 0.20,
            "bar_vs_ti_combined_se_multiplier": 2.0,
            "all_walkers_must_pass": True,
            "walker_pairwise_max_z": 2.0,
            "walker_max_span_kcal_per_mol": 0.50,
            "ascending_vs_descending_abs_kcal_per_mol": 0.20,
            "ascending_vs_descending_combined_se_multiplier": 2.0,
            "endpoint_energy_absolute_tolerance_ev": 1.0e-8,
            "endpoint_energy_relative_tolerance": 1.0e-10,
            "endpoint_force_tolerance_ev_per_angstrom": 1.0e-7,
            "p1_ordinary_energy_force_identity_required": True,
            "p0_factorization_identity_required": True,
            "analytic_lambda_finite_difference_gate_required": True,
            "softcore_must_vanish_at_lambda_v_one": True,
            "reaction_forbidden": True,
            "proton_transfer_forbidden": True,
            "molecular_identity_change_forbidden": True,
            "water_connectivity_change_forbidden": True,
            "sensitivity_graph_softcore_angstrom": 0.5,
            "sensitivity_repulsion_softcore_variants_angstrom": (0.4, 0.6),
            "sensitivity_variants_output_blind": True,
            "sensitivity_experimental_selection_forbidden": True,
            "sensitivity_variants_must_be_finite": True,
            "sensitivity_variants_must_pass_all_gates": True,
            "sensitivity_total_hfe_shift_abs_kcal_per_mol": 0.20,
            "sensitivity_total_hfe_shift_combined_se_multiplier": 2.0,
            "hfe_admitted": False,
            "accuracy_admitted": False,
            "gpu_admitted": False,
            "performance_admitted": False,
        }
        actual_names = {item.name for item in fields(self)}
        if actual_names != {"kernel_identity", *expected}:
            raise RuntimeError("FeNNix HFE protocol provenance/field contract drifted.")
        if actual_names != set(self.FIELD_PROVENANCE):
            raise RuntimeError("Every FeNNix HFE protocol field must carry provenance.")

        for name, frozen in expected.items():
            value = getattr(self, name)
            if isinstance(frozen, bool):
                if not isinstance(value, bool):
                    raise ValueError(
                        f"FeNNix HFE protocol field {name} must be a boolean."
                    )
            elif isinstance(frozen, int):
                value = _require_positive_integer(name, value)
            elif isinstance(frozen, float):
                value = _require_finite_real(name, value, positive=frozen > 0.0)
            _require_exact(name, value, frozen)

        if (
            len(self.random_seeds) != self.walker_count
            or len(set(self.random_seeds)) != 3
        ):
            raise ValueError("FeNNix HFE walkers require three unique fixed seeds.")
        if set(self.window_orders) != {"ascending", "descending"}:
            raise ValueError(
                "FeNNix HFE protocol requires ascending and descending orders."
            )
        if self.observable_interval_picoseconds < self.timestep_femtoseconds / 1000.0:
            raise ValueError("FeNNix HFE observable interval is shorter than one step.")
        if self.restart_interval_picoseconds < self.observable_interval_picoseconds:
            raise ValueError(
                "FeNNix HFE restart interval is shorter than observation interval."
            )
        if (
            self.sensitivity_graph_softcore_angstrom
            in self.sensitivity_repulsion_softcore_variants_angstrom
        ):
            raise ValueError(
                "FeNNix sensitivity variants must bracket, not repeat, the base alpha."
            )

        # Force complete canonicalization during construction so unsupported or
        # non-finite nested kernel identity state fails before any run can start.
        self.canonical_json()

    @staticmethod
    def _finite_uncertainty(name: str, value: float) -> float:
        normalized = _require_finite_real(name, value)
        if normalized < 0.0:
            raise ValueError(f"FeNNix HFE {name} must be nonnegative.")
        return normalized

    def block_length_picoseconds(self, tau_int_picoseconds: float) -> float:
        """Apply the frozen max(100 ps, 5*tau_int) blocking rule."""

        tau_int = self._finite_uncertainty("tau_int_picoseconds", tau_int_picoseconds)
        return max(
            self.minimum_block_length_picoseconds,
            self.block_length_tau_int_multiplier * tau_int,
        )

    def endpoint_energy_tolerance_ev(self, ordinary_energy_ev: float) -> float:
        """Apply max(1e-8 eV, 1e-10*abs(E)) at either endpoint."""

        energy = _require_finite_real("ordinary_energy_ev", ordinary_energy_ev)
        return max(
            self.endpoint_energy_absolute_tolerance_ev,
            self.endpoint_energy_relative_tolerance * abs(energy),
        )

    def full_vs_final_half_ti_tolerance(self, standard_error: float) -> float:
        """Return max(0.20 kcal/mol, 1 SE)."""

        error = self._finite_uncertainty("standard_error", standard_error)
        return max(
            self.full_vs_final_half_ti_abs_kcal_per_mol,
            self.full_vs_final_half_ti_se_multiplier * error,
        )

    def simpson_vs_trapezoid_leg_tolerance(self, standard_error: float) -> float:
        """Return max(0.10 kcal/mol, 0.25 SE) for one native leg."""

        error = self._finite_uncertainty("standard_error", standard_error)
        return max(
            self.simpson_vs_trapezoid_leg_abs_kcal_per_mol,
            self.simpson_vs_trapezoid_leg_se_multiplier * error,
        )

    def bar_vs_ti_tolerance(self, combined_standard_error: float) -> float:
        """Return max(0.20 kcal/mol, 2 combined SE)."""

        error = self._finite_uncertainty(
            "combined_standard_error", combined_standard_error
        )
        return max(
            self.bar_vs_ti_abs_kcal_per_mol,
            self.bar_vs_ti_combined_se_multiplier * error,
        )

    def ascending_vs_descending_tolerance(
        self, combined_standard_error: float
    ) -> float:
        """Return max(0.20 kcal/mol, 2 combined SE)."""

        error = self._finite_uncertainty(
            "combined_standard_error", combined_standard_error
        )
        return max(
            self.ascending_vs_descending_abs_kcal_per_mol,
            self.ascending_vs_descending_combined_se_multiplier * error,
        )

    def sensitivity_hfe_shift_tolerance(self, combined_standard_error: float) -> float:
        """Return the output-blind alpha sensitivity admission limit."""

        error = self._finite_uncertainty(
            "combined_standard_error", combined_standard_error
        )
        return max(
            self.sensitivity_total_hfe_shift_abs_kcal_per_mol,
            self.sensitivity_total_hfe_shift_combined_se_multiplier * error,
        )

    @property
    def native_lambda_states(self) -> tuple[tuple[float, float], ...]:
        """Return the 21 unique native states in physical coupling order."""

        repulsion = tuple((0.0, value) for value in self.repulsion_leg_lambda_v)
        electronic = tuple((value, 1.0) for value in self.electronic_leg_lambda_e[1:])
        return repulsion + electronic

    def canonical_payload(self) -> dict[str, Any]:
        """Return every field paired with its frozen provenance category."""

        return {
            "schema": "maple.fennix_hfe_protocol.v1",
            "fields": {
                item.name: {
                    "provenance": self.FIELD_PROVENANCE[item.name].value,
                    "value": _canonical_value(
                        getattr(self, item.name), path=f"protocol.{item.name}"
                    ),
                }
                for item in fields(self)
            },
        }

    def canonical_json(self) -> str:
        """Return deterministic ASCII JSON suitable for durable receipts."""

        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def canonical_sha256(self) -> str:
        """Bind all values, provenance labels, and exact kernel identity."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def __hash__(self) -> int:
        return int(self.canonical_sha256()[:16], 16)


__all__ = ["FeNNixHFEProtocol", "FeNNixProtocolProvenance"]
