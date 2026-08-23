"""Deterministic scientific measurement engine for harmonic hybrid E/F gates.

This module owns only the numerical measurement preimage shared by release
protocols.  It does not assign protocol identity, admit capabilities, capture
runtime provenance, or publish evidence artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
import re

from ase import Atoms
import numpy as np

from maple.function.calculator.extra_correction.implicit.smd_cds import (
    smd_water_coulomb_radii,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader
from maple.solvation.derivatives import (
    RichardsonScalarForce,
    RichardsonScalarForceComponentEvaluation,
    RichardsonScalarForceEvaluation,
    ScalarEnergySample,
)
from maple.solvation.coupling.state_equation import geometry_sha256
from maple.solvation.experimental.mace_mdp_polar_harmonic import (
    HybridHarmonicEnergyState,
    HybridHarmonicSolveDiagnostics,
    MACE_MDPPolarHybridSmoothHarmonicPES,
    NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM,
    ROOT_REPLAY_ENERGY_ATOL_EV,
    ROOT_REPLAY_FIELD_ATOL_EV,
    ROOT_TOLERANCE_EV,
)
from maple.solvation.release.evidence import canonical_json_sha256, sha256_file

PANEL_RELATIVE_ROOT = ".omx/benchmarks/route2-gto-pcm-energy-projection-four-v1"
CUTOFF_DIRECTORY = "cutoff-1e-12"
TOTAL_CHARGE_ATOL_E = 1.0e-8
HARMONIC_EF_REPLICATE_GATE_NAMES = (
    "clean_content_addressed_execution",
    "benzene_predecessor_failure_coordinate_passes_h_h2_h4",
    "water_full_cartesian_h_h2_force_is_finite",
    "all_local_richardson_error_estimates_below_2e-4_ev_per_angstrom",
    "independent_h4_directional_derivative_below_5e-4_ev_per_angstrom",
    "translation_energy_and_net_force_gates_pass",
    "rotation_energy_and_force_covariance_gates_pass",
    "closed_loop_work_plus_numerical_error_bound_gate_passes",
    "all_roots_converge_from_both_starts",
    "all_final_residuals_and_total_charge_errors_pass",
)
HARMONIC_EF_MEASUREMENT_SCOPE = "historical-v1-measurement-not-v2-admission-preimage"
HARMONIC_EF_V1_PROJECTION_SCOPE = (
    "archive-only-v1-shape-adapter-compatibility-non-admitting"
)
HARMONIC_EF_H0_NON_IDENTIFIABILITY = (
    "historical H0 artifacts omit displaced energies, per-start fields, and "
    "residual vectors; no rich H0 preimage can be reconstructed"
)
RICH_HARMONIC_EF_V2_EXECUTION_ENABLED = True
RICH_HARMONIC_EF_STATE_LEAF_REQUIRED_FIELDS = (
    "schema_id",
    "state_leaf_sha256",
    "legacy_root_sha256",
    "geometry_sha256",
    "provider_configuration_sha256",
    "prepared_pes_configuration_sha256",
    "topology_id",
    "target_charge_e",
    "audit_coefficient_sum_charge_e",
    "source_coefficient_basis_id",
    "source_space_contract_sha256",
    "source_coefficient_order",
    "receiver_space_id",
    "receiver_space_contract_sha256",
    "receiver_component_order",
    "audit_projection_contract",
    "numeric_array_encoding_contract",
    "endpoint_replay_scope",
    "permanent_point_source4",
    "polar_zero_reference_source4",
    "polar_final_source4",
    "induced_gto_source4",
    "audit_coefficient_sum4",
    "array_content_sha256s",
    "vacuum_energy_eV",
    "polarization_energy_eV",
    "total_energy_eV",
    "cold_start",
    "wide_start",
)
HARMONIC_EF_FORCE_PANEL_THRESHOLDS = MappingProxyType(
    {
        "force_coarse_step_angstrom": 5.0e-4,
        "force_fine_step_angstrom": 2.5e-4,
        "independent_force_step_angstrom": 1.25e-4,
        "force_maximum_local_error_estimate_ev_per_angstrom": 2.0e-4,
        "root_tolerance_ev": 1.0e-10,
        "maximum_root_iterations": 40,
        "total_charge_tolerance_e": 1.0e-8,
        "multi_start_field_tolerance_ev": 2.0e-9,
        "multi_start_energy_tolerance_ev": 1.0e-10,
        "maximum_independent_directional_error_ev_per_angstrom": 5.0e-4,
        "maximum_benzene_h4_difference_ev_per_angstrom": 2.0e-4,
        "maximum_translation_energy_error_ev": 1.0e-7,
        "maximum_translation_force_relative_error": 2.0e-5,
        "maximum_translation_force_absolute_error_ev_per_angstrom": 5.0e-5,
        "maximum_net_force_norm_ev_per_angstrom": 1.0e-3,
        "maximum_rotation_energy_error_ev": 1.0e-7,
        "maximum_rotation_force_relative_error": 2.0e-5,
        "maximum_rotation_force_absolute_error_ev_per_angstrom": 5.0e-5,
        "closed_loop_half_width_angstrom": 1.0e-3,
        "maximum_closed_loop_work_abs_ev": 2.0e-7,
    }
)
_FROZEN_FORCE_PANEL_IDENTITY = MappingProxyType(
    {
        "gepol_regression_compound_id": "mobley_3053621",
        "gepol_regression_cartesian_dof": (0, 0),
        "water_geometry_angstrom": (
            (0.0, 0.0, 0.0),
            (0.9572, 0.0, 0.0),
            (-0.239, 0.9266, 0.0),
        ),
        "independent_direction_seed": 20260816,
        "translation_angstrom": (4.2, -3.1, 1.7),
        "rotation_seed": 20260817,
        "closed_loop_cartesian_dofs": ((0, 0), (1, 1)),
    }
)
__all__ = [
    "HARMONIC_EF_FORCE_PANEL_THRESHOLDS",
    "HARMONIC_EF_H0_NON_IDENTIFIABILITY",
    "HARMONIC_EF_MEASUREMENT_SCOPE",
    "HARMONIC_EF_REPLICATE_GATE_NAMES",
    "HARMONIC_EF_V1_PROJECTION_SCOPE",
    "H1PreparedHarmonicEFPESV2",
    "PreparedHarmonicEFBenzeneInputV2",
    "PreparedHarmonicEFInputsV2",
    "PreparedHarmonicEFWaterInputV2",
    "RICH_HARMONIC_EF_V2_EXECUTION_ENABLED",
    "RICH_HARMONIC_EF_STATE_LEAF_REQUIRED_FIELDS",
    "RichHarmonicEFStateRecorderV2",
    "RichHarmonicEFStateLeafUnavailable",
    "capture_rich_harmonic_ef_state_leaf_v2",
    "record_benzene_predecessor_stencil_v2",
    "record_benzene_system_v2",
    "record_cartesian_panel_v2",
    "record_closed_loop_v2",
    "record_component_stencil_v2",
    "record_rotated_cartesian_panel_v2",
    "record_translated_cartesian_panel_v2",
    "record_water_directional_v2",
    "record_water_system_v2",
    "project_rich_harmonic_ef_raw_tree_to_v1_science_v2",
    "merge_rich_harmonic_ef_recordings_v2",
    "rich_harmonic_ef_state_leaf_missing_fields_v2",
    "derive_harmonic_ef_replicate_gates",
    "run_harmonic_ef_measurement",
    "run_rich_harmonic_ef_raw_measurement_v2",
    "validate_serialized_rich_state_recording_v2",
    "validate_harmonic_ef_science_measurement",
]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_H1_PREPARED_PES_ADAPTER_CONTRACT_ID = (
    "maple-route2-h1-prepared-harmonic-pes-adapter-v1"
)
_H1_PREPARED_PES_PROVIDER_ID = "maple.route2.h1-v2.prepared-harmonic-pes-adapter.v1"


def _exact_mapping(value: object, keys: set[str], *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{name} fields are invalid: {'; '.join(details)}.")
    return value


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be exactly bool.")
    return value


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a JSON number, not bool or text.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _finite_array(value: object, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, bool) or not isinstance(item, (int, float, np.number)):
            raise TypeError(f"{name} must contain only JSON numbers.")
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return result


def _require_close(actual: object, expected: float, *, name: str) -> None:
    value = _finite_scalar(actual, name=name)
    tolerance = 2.0e-15 * max(1.0, abs(expected))
    if not np.isclose(value, expected, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"engine-derived gates: {name} contradicts its detailed measurement."
        )


def _require_array_close(
    actual: np.ndarray, expected: np.ndarray, *, name: str
) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=2.0e-15):
        raise ValueError(
            f"engine-derived gates: {name} contradicts its frozen panel identity."
        )


_RUNTIME_THRESHOLD_KEYS = frozenset(
    {
        "force_coarse_step_angstrom",
        "force_fine_step_angstrom",
        "independent_force_step_angstrom",
        "force_maximum_local_error_estimate_ev_per_angstrom",
        "root_tolerance_ev",
        "maximum_root_iterations",
        "total_charge_tolerance_e",
        "multi_start_field_tolerance_ev",
        "multi_start_energy_tolerance_ev",
    }
)


def _resolved_runtime_contract(
    value: dict[str, object] | None,
) -> dict[str, object]:
    result = {
        key: item
        for key, item in HARMONIC_EF_FORCE_PANEL_THRESHOLDS.items()
        if key in _RUNTIME_THRESHOLD_KEYS
    }
    if value is not None:
        result.update(value)
    return result


def _resolved_force_panel(value: dict[str, object] | None) -> dict[str, object]:
    result = {
        key: item
        for key, item in HARMONIC_EF_FORCE_PANEL_THRESHOLDS.items()
        if key not in _RUNTIME_THRESHOLD_KEYS
    }
    result.update(
        {
            "gepol_regression_compound_id": _FROZEN_FORCE_PANEL_IDENTITY[
                "gepol_regression_compound_id"
            ],
            "gepol_regression_cartesian_dof": list(
                _FROZEN_FORCE_PANEL_IDENTITY["gepol_regression_cartesian_dof"]
            ),
            "water_geometry_angstrom": [
                list(row)
                for row in _FROZEN_FORCE_PANEL_IDENTITY["water_geometry_angstrom"]
            ],
            "independent_direction_seed": _FROZEN_FORCE_PANEL_IDENTITY[
                "independent_direction_seed"
            ],
            "translation_angstrom": list(
                _FROZEN_FORCE_PANEL_IDENTITY["translation_angstrom"]
            ),
            "rotation_seed": _FROZEN_FORCE_PANEL_IDENTITY["rotation_seed"],
            "closed_loop_cartesian_dofs": [
                list(dof)
                for dof in _FROZEN_FORCE_PANEL_IDENTITY["closed_loop_cartesian_dofs"]
            ],
        }
    )
    if value is not None:
        result.update(value)
    return result


def _sha256_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _prepared_positions(
    value: object, *, atom_count: int, name: str
) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (atom_count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({atom_count}, 3).")
    return tuple(tuple(float(component) for component in row) for row in array)


def _prepared_radii(value: object, *, atom_count: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if (
        array.shape != (atom_count,)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise ValueError(f"{name} must contain {atom_count} positive finite radii.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class PreparedHarmonicEFBenzeneInputV2:
    """Parsing-free immutable benzene identity for the future rich-v2 core."""

    compound_id: str
    name: str
    atomic_numbers: tuple[int, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    cavity_radii_angstrom: tuple[float, ...]
    mol2_sha256: str
    projection_result_sha256: str
    predecessor_dof: tuple[int, int]

    def __post_init__(self) -> None:
        if self.compound_id != "mobley_3053621" or not self.name:
            raise ValueError("prepared benzene identity is not the frozen record.")
        numbers = tuple(self.atomic_numbers)
        if (
            len(numbers) != 12
            or any(type(number) is not int or number <= 0 for number in numbers)
            or self.charge != 0
            or self.multiplicity != 1
        ):
            raise ValueError("prepared benzene atomic/domain identity is invalid.")
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(
            self,
            "positions_angstrom",
            _prepared_positions(
                self.positions_angstrom, atom_count=12, name="benzene positions"
            ),
        )
        object.__setattr__(
            self,
            "cavity_radii_angstrom",
            _prepared_radii(
                self.cavity_radii_angstrom,
                atom_count=12,
                name="benzene cavity radii",
            ),
        )
        _sha256_text(self.mol2_sha256, name="benzene mol2_sha256")
        _sha256_text(
            self.projection_result_sha256,
            name="benzene projection_result_sha256",
        )
        if tuple(self.predecessor_dof) != (0, 0):
            raise ValueError("prepared benzene predecessor_dof must be (0, 0).")
        object.__setattr__(self, "predecessor_dof", (0, 0))

    def __deepcopy__(self, memo: dict[int, object]):
        del memo
        return self


@dataclass(frozen=True, slots=True)
class PreparedHarmonicEFWaterInputV2:
    """Parsing-free immutable water identity for the future rich-v2 core."""

    atomic_numbers: tuple[int, int, int]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    cavity_radii_angstrom: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            tuple(self.atomic_numbers) != (8, 1, 1)
            or self.charge != 0
            or self.multiplicity != 1
        ):
            raise ValueError("prepared water atomic/domain identity is invalid.")
        positions = _prepared_positions(
            self.positions_angstrom, atom_count=3, name="water positions"
        )
        expected = np.asarray(_FROZEN_FORCE_PANEL_IDENTITY["water_geometry_angstrom"])
        if not np.array_equal(np.asarray(positions), expected):
            raise ValueError("prepared water geometry drifted from the frozen panel.")
        object.__setattr__(self, "atomic_numbers", (8, 1, 1))
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(
            self,
            "cavity_radii_angstrom",
            _prepared_radii(
                self.cavity_radii_angstrom,
                atom_count=3,
                name="water cavity radii",
            ),
        )

    def __deepcopy__(self, memo: dict[int, object]):
        del memo
        return self


def _immutable_prepared_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _immutable_prepared_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_immutable_prepared_json(item) for item in value)
    return value


def _mutable_prepared_json(value: object) -> object:
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _mutable_prepared_json(item) for key, item in dict(value).items()}
    if isinstance(value, tuple):
        return [_mutable_prepared_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PreparedHarmonicEFInputsV2:
    """I/O-free prepared system bundle with sealed contract identities."""

    benzene: PreparedHarmonicEFBenzeneInputV2
    water: PreparedHarmonicEFWaterInputV2
    v1_preregistration_sha256: str
    parent_panel_sha256: str
    force_panel_contract: dict[str, object]
    force_panel_contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.benzene, PreparedHarmonicEFBenzeneInputV2):
            raise TypeError("benzene must be a prepared benzene input.")
        if not isinstance(self.water, PreparedHarmonicEFWaterInputV2):
            raise TypeError("water must be a prepared water input.")
        for name in (
            "v1_preregistration_sha256",
            "parent_panel_sha256",
            "force_panel_contract_sha256",
        ):
            _sha256_text(getattr(self, name), name=name)
        expected_panel = {
            "gepol_regression_compound_id": "mobley_3053621",
            "gepol_regression_cartesian_dof": [0, 0],
            "water_geometry_angstrom": [
                list(row)
                for row in _FROZEN_FORCE_PANEL_IDENTITY["water_geometry_angstrom"]
            ],
            "water_full_cartesian_force": True,
            "independent_direction_seed": 20260816,
            "maximum_independent_directional_error_ev_per_angstrom": 5.0e-4,
            "maximum_benzene_h4_difference_ev_per_angstrom": 2.0e-4,
            "translation_angstrom": [4.2, -3.1, 1.7],
            "maximum_translation_energy_error_ev": 1.0e-7,
            "maximum_translation_force_relative_error": 2.0e-5,
            "maximum_translation_force_absolute_error_ev_per_angstrom": 5.0e-5,
            "maximum_net_force_norm_ev_per_angstrom": 1.0e-3,
            "rotation_seed": 20260817,
            "maximum_rotation_energy_error_ev": 1.0e-7,
            "maximum_rotation_force_relative_error": 2.0e-5,
            "maximum_rotation_force_absolute_error_ev_per_angstrom": 5.0e-5,
            "closed_loop_cartesian_dofs": [[0, 0], [1, 1]],
            "closed_loop_half_width_angstrom": 1.0e-3,
            "maximum_closed_loop_work_abs_ev": 2.0e-7,
        }
        normalized_panel = json.loads(
            json.dumps(self.force_panel_contract, sort_keys=True, allow_nan=False)
        )
        if normalized_panel != expected_panel:
            raise ValueError("force_panel_contract differs from the frozen panel.")
        if canonical_json_sha256(normalized_panel) != self.force_panel_contract_sha256:
            raise ValueError("force_panel_contract_sha256 does not match content.")
        object.__setattr__(
            self,
            "force_panel_contract",
            _immutable_prepared_json(normalized_panel),
        )

    def __deepcopy__(self, memo: dict[int, object]):
        del memo
        return self


def _prepared_system_identity_payload(
    prepared: PreparedHarmonicEFBenzeneInputV2 | PreparedHarmonicEFWaterInputV2,
) -> dict[str, object]:
    if isinstance(prepared, PreparedHarmonicEFBenzeneInputV2):
        return {
            "compound_id": prepared.compound_id,
            "name": prepared.name,
            "atomic_numbers": list(prepared.atomic_numbers),
            "positions_angstrom": [list(row) for row in prepared.positions_angstrom],
            "charge": prepared.charge,
            "multiplicity": prepared.multiplicity,
            "cavity_radii_angstrom": list(prepared.cavity_radii_angstrom),
            "mol2_sha256": prepared.mol2_sha256,
            "projection_result_sha256": prepared.projection_result_sha256,
            "predecessor_dof": list(prepared.predecessor_dof),
        }
    if isinstance(prepared, PreparedHarmonicEFWaterInputV2):
        return {
            "atomic_numbers": list(prepared.atomic_numbers),
            "positions_angstrom": [list(row) for row in prepared.positions_angstrom],
            "charge": prepared.charge,
            "multiplicity": prepared.multiplicity,
            "cavity_radii_angstrom": list(prepared.cavity_radii_angstrom),
        }
    raise TypeError("prepared system identity type is unsupported.")


class H1PreparedHarmonicEFPESV2:
    """Immutable non-public H1 identity adapter over the truthful v1 PES."""

    __slots__ = (
        "_base",
        "_configuration_sha256",
        "_prepared",
        "_prepared_input_identity_sha256",
        "_sealed",
        "_system_role",
        "_underlying_configuration_sha256",
    )

    provider_id = _H1_PREPARED_PES_PROVIDER_ID

    def __init__(
        self,
        base: MACE_MDPPolarHybridSmoothHarmonicPES,
        prepared: PreparedHarmonicEFBenzeneInputV2 | PreparedHarmonicEFWaterInputV2,
        *,
        system_role: str,
    ) -> None:
        if not isinstance(base, MACE_MDPPolarHybridSmoothHarmonicPES):
            raise TypeError("base must be a real harmonic v1 PES implementation.")
        if system_role not in {"benzene", "water"}:
            raise ValueError("system_role must be exactly benzene or water.")
        expected_type = (
            PreparedHarmonicEFBenzeneInputV2
            if system_role == "benzene"
            else PreparedHarmonicEFWaterInputV2
        )
        if not isinstance(prepared, expected_type):
            raise TypeError("prepared input type disagrees with adapter system_role.")
        if tuple(base.prepared_atomic_numbers) != tuple(prepared.atomic_numbers):
            raise ValueError(
                "underlying PES atomic numbers differ from prepared input."
            )
        if tuple(base.prepared_cavity_radii_angstrom) != tuple(
            prepared.cavity_radii_angstrom
        ):
            raise ValueError("underlying PES cavity radii differ from prepared input.")
        if (
            base.prepared_charge != prepared.charge
            or base.prepared_multiplicity != prepared.multiplicity
        ):
            raise ValueError("underlying PES domain differs from prepared input.")
        from maple.solvation.release.admission import (
            H1_V2_PROFILE_ID,
            H1_V2_SCALAR_ID,
            H1_V2_STATE_ID,
        )

        underlying_configuration = base.configuration_sha256()
        input_identity = canonical_json_sha256(
            _prepared_system_identity_payload(prepared)
        )
        payload = {
            "contract_id": _H1_PREPARED_PES_ADAPTER_CONTRACT_ID,
            "provider_id": self.provider_id,
            "profile_id": H1_V2_PROFILE_ID,
            "scalar_id": H1_V2_SCALAR_ID,
            "state_id": H1_V2_STATE_ID,
            "system_role": system_role,
            "underlying_provider_id": base.prepared_provider_id,
            "underlying_profile_id": base.prepared_profile_id,
            "underlying_scalar_id": base.prepared_scalar_id,
            "underlying_configuration_sha256": underlying_configuration,
            "prepared_input_identity_sha256": input_identity,
        }
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_prepared", prepared)
        object.__setattr__(self, "_system_role", system_role)
        object.__setattr__(
            self, "_underlying_configuration_sha256", underlying_configuration
        )
        object.__setattr__(self, "_prepared_input_identity_sha256", input_identity)
        object.__setattr__(
            self, "_configuration_sha256", canonical_json_sha256(payload)
        )
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("H1 prepared PES adapter is immutable.")
        object.__setattr__(self, name, value)

    @property
    def system_role(self) -> str:
        return self._system_role

    @property
    def prepared_atomic_numbers(self) -> tuple[int, ...]:
        return tuple(self._prepared.atomic_numbers)

    @property
    def prepared_cavity_radii_angstrom(self) -> tuple[float, ...]:
        return tuple(self._prepared.cavity_radii_angstrom)

    @property
    def prepared_charge(self) -> int:
        return self._prepared.charge

    @property
    def prepared_multiplicity(self) -> int:
        return self._prepared.multiplicity

    @property
    def prepared_profile_id(self) -> str:
        from maple.solvation.release.admission import H1_V2_PROFILE_ID

        return H1_V2_PROFILE_ID

    @property
    def prepared_scalar_id(self) -> str:
        from maple.solvation.release.admission import H1_V2_SCALAR_ID

        return H1_V2_SCALAR_ID

    @property
    def prepared_state_id(self) -> str:
        from maple.solvation.release.admission import H1_V2_STATE_ID

        return H1_V2_STATE_ID

    @property
    def prepared_provider_id(self) -> str:
        return self.provider_id

    @property
    def prepared_input_identity_sha256(self) -> str:
        return self._prepared_input_identity_sha256

    @property
    def underlying_provider_id(self) -> str:
        return self._base.prepared_provider_id

    @property
    def underlying_profile_id(self) -> str:
        return self._base.prepared_profile_id

    @property
    def underlying_scalar_id(self) -> str:
        return self._base.prepared_scalar_id

    @property
    def underlying_configuration_sha256(self) -> str:
        return self._underlying_configuration_sha256

    @property
    def prepared_configuration_sha256(self) -> str:
        return self.configuration_sha256()

    def configuration_sha256(self) -> str:
        if self._base.configuration_sha256() != self._underlying_configuration_sha256:
            raise RuntimeError("underlying v1 PES configuration changed.")
        return self._configuration_sha256

    def solve_with_diagnostics(self, geometry: object):
        self.configuration_sha256()
        result = self._base.solve_with_diagnostics(geometry)
        self.configuration_sha256()
        return result


class RichHarmonicEFStateLeafUnavailable(RuntimeError):
    """Raised while the solver lacks approved per-start/source raw leaves."""


def rich_harmonic_ef_state_leaf_missing_fields_v2(
    state: object,
    diagnostics: object | None = None,
    prepared_pes_configuration_sha256: object | None = None,
) -> tuple[str, ...]:
    """Report exact approved StateLeaf fields missing from typed solve outputs."""

    typed_state = isinstance(state, HybridHarmonicEnergyState)
    typed_diagnostics = isinstance(diagnostics, HybridHarmonicSolveDiagnostics)
    available = {
        "schema_id": True,
        "state_leaf_sha256": True,
        "legacy_root_sha256": typed_state,
        "geometry_sha256": typed_state,
        "provider_configuration_sha256": typed_state,
        "prepared_pes_configuration_sha256": (
            isinstance(prepared_pes_configuration_sha256, str)
            and _SHA256_PATTERN.fullmatch(prepared_pes_configuration_sha256) is not None
        ),
        "topology_id": typed_state,
        "target_charge_e": typed_diagnostics,
        "audit_coefficient_sum_charge_e": typed_diagnostics,
        "source_coefficient_basis_id": typed_diagnostics,
        "source_space_contract_sha256": typed_diagnostics,
        "source_coefficient_order": typed_diagnostics,
        "receiver_space_id": typed_diagnostics,
        "receiver_space_contract_sha256": typed_diagnostics,
        "receiver_component_order": typed_diagnostics,
        "audit_projection_contract": typed_diagnostics,
        "numeric_array_encoding_contract": typed_diagnostics,
        "endpoint_replay_scope": typed_diagnostics,
        "permanent_point_source4": typed_diagnostics,
        "polar_zero_reference_source4": typed_diagnostics,
        "polar_final_source4": typed_diagnostics,
        "induced_gto_source4": typed_diagnostics,
        "audit_coefficient_sum4": typed_diagnostics,
        "array_content_sha256s": typed_diagnostics,
        "vacuum_energy_eV": typed_state,
        "polarization_energy_eV": typed_state,
        "total_energy_eV": typed_state,
        "cold_start": typed_diagnostics,
        "wide_start": typed_diagnostics,
    }
    return tuple(
        field
        for field in RICH_HARMONIC_EF_STATE_LEAF_REQUIRED_FIELDS
        if not available[field]
    )


def capture_rich_harmonic_ef_state_leaf_v2(
    state: object,
    diagnostics: object,
    *,
    prepared_pes_configuration_sha256: object,
) -> dict[str, object]:
    """Capture and independently validate one rich-v2 state-leaf mapping."""

    missing = rich_harmonic_ef_state_leaf_missing_fields_v2(
        state, diagnostics, prepared_pes_configuration_sha256
    )
    if missing:
        raise RichHarmonicEFStateLeafUnavailable(
            "rich-v2 StateLeaf capture requires typed legacy state plus complete "
            f"solve diagnostics; missing approved fields: {', '.join(missing)}"
        )
    assert isinstance(state, HybridHarmonicEnergyState)
    assert isinstance(diagnostics, HybridHarmonicSolveDiagnostics)
    prepared_pes_configuration = _sha256_text(
        prepared_pes_configuration_sha256,
        name="prepared_pes_configuration_sha256",
    )

    # Lazy import avoids a release-contract/measurement-engine import cycle.
    from maple.solvation.release.admission import (
        H1_V2_ENDPOINT_REPLAY_SCOPE,
        H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT,
        H1_V2_RECEIVER_COMPONENT_ORDER,
        H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
        H1_V2_RECEIVER_SPACE_ID,
        H1_V2_RECEIVER_UNITS,
        H1_V2_SOURCE_COEFFICIENT_BASIS_ID,
        H1_V2_SOURCE_COEFFICIENT_ORDER,
        H1_V2_SOURCE_COEFFICIENT_UNITS,
        H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
        STATE_LEAF_V2_SCHEMA,
        StateLeafV2,
        canonical_numeric_array_sha256_v2,
        canonical_source_monopole_sum_v2,
        h1_v2_audit_coefficient_sum_contract,
    )

    if diagnostics.source_basis_id != H1_V2_SOURCE_COEFFICIENT_BASIS_ID:
        raise ValueError("diagnostic source basis does not match rich-v2.")
    if diagnostics.source_component_order != H1_V2_SOURCE_COEFFICIENT_ORDER:
        raise ValueError("diagnostic source component order does not match rich-v2.")
    if diagnostics.source_space_contract_sha256 != H1_V2_SOURCE_SPACE_CONTRACT_SHA256:
        raise ValueError("diagnostic source-space digest does not match rich-v2.")
    if diagnostics.receiver_basis_id != H1_V2_RECEIVER_SPACE_ID:
        raise ValueError("diagnostic receiver basis does not match rich-v2.")
    if (
        diagnostics.receiver_space_contract_sha256
        != H1_V2_RECEIVER_SPACE_CONTRACT_SHA256
    ):
        raise ValueError("diagnostic receiver-space digest does not match rich-v2.")
    roles = dict(diagnostics.source_role_identities)
    if roles.get("audit_coefficient_sum4") != (
        "non-operational permanent+induced coefficient audit projection"
    ):
        raise ValueError("diagnostic coefficient-audit role is unsupported.")
    if not np.array_equal(state.induced_source4, diagnostics.induced_source4):
        raise ValueError("legacy state and diagnostics induced sources disagree.")
    if not np.array_equal(
        state.total_source4, diagnostics.audit_coefficient_sum4
    ):
        raise ValueError("legacy state and diagnostics coefficient audits disagree.")
    cold = diagnostics.cold_start
    wide = diagnostics.wide_start
    if not np.array_equal(state.native_field_ev, cold.final_native_field_ev):
        raise ValueError("legacy state and diagnostics selected root field disagree.")
    if state.polarization_energy_ev != cold.final_polarization_energy_ev:
        raise ValueError("legacy state and diagnostics selected root energy disagree.")
    if state.primal_residual_ev != cold.final_residual_norm_ev:
        raise ValueError("legacy state and diagnostics selected residual disagree.")
    if state.cold_iterations != cold.iterations:
        raise ValueError("legacy state and diagnostics cold iterations disagree.")
    if state.wide_iterations != wide.iterations:
        raise ValueError("legacy state and diagnostics wide iterations disagree.")
    field_difference = float(
        np.max(np.abs(cold.final_native_field_ev - wide.final_native_field_ev))
    )
    if state.replay_field_max_abs_difference_ev != field_difference:
        raise ValueError(
            "legacy state and diagnostics replay field difference disagree."
        )
    energy_difference = abs(
        cold.final_polarization_energy_ev - wide.final_polarization_energy_ev
    )
    if state.replay_energy_abs_difference_ev != energy_difference:
        raise ValueError(
            "legacy state and diagnostics replay energy difference disagree."
        )

    def start_payload(start: object) -> dict[str, object]:
        return {
            "initial_state_sha256": start.initial_state_sha256,
            "converged": start.converged,
            "iterations": start.iterations,
            "final_native_field_eV": start.final_native_field_ev.tolist(),
            "final_residual_eV": start.final_residual_ev.tolist(),
            "final_polarization_energy_eV": start.final_polarization_energy_ev,
            "final_total_energy_eV": (
                state.vacuum_energy_ev + start.final_polarization_energy_ev
            ),
        }

    cold_payload = start_payload(diagnostics.cold_start)
    wide_payload = start_payload(diagnostics.wide_start)
    array_inputs = {
        "permanent_point_source4": (
            diagnostics.permanent_source4,
            "permanent-point-source4",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "polar_zero_reference_source4": (
            diagnostics.response_zero_source4,
            "polar-zero-reference-source4-nonoperational",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "polar_final_source4": (
            diagnostics.response_final_source4,
            "polar-final-source4",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "induced_gto_source4": (
            diagnostics.induced_source4,
            "induced-gto1p5-source4",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "audit_coefficient_sum4": (
            diagnostics.audit_coefficient_sum4,
            "audit-coefficient-sum4-nonoperational",
            H1_V2_SOURCE_SPACE_CONTRACT_SHA256,
            H1_V2_SOURCE_COEFFICIENT_UNITS,
        ),
        "cold_start.final_native_field_eV": (
            diagnostics.cold_start.final_native_field_ev,
            "cold-final-native-field8",
            H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
            H1_V2_RECEIVER_UNITS,
        ),
        "cold_start.final_residual_eV": (
            diagnostics.cold_start.final_residual_ev,
            "cold-final-actual-residual-field8",
            H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
            H1_V2_RECEIVER_UNITS,
        ),
        "wide_start.final_native_field_eV": (
            diagnostics.wide_start.final_native_field_ev,
            "wide-final-native-field8",
            H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
            H1_V2_RECEIVER_UNITS,
        ),
        "wide_start.final_residual_eV": (
            diagnostics.wide_start.final_residual_ev,
            "wide-final-actual-residual-field8",
            H1_V2_RECEIVER_SPACE_CONTRACT_SHA256,
            H1_V2_RECEIVER_UNITS,
        ),
    }
    array_content_sha256s = {
        name: canonical_numeric_array_sha256_v2(
            values,
            semantic_type=semantic_type,
            geometry_sha256=state.geometry_sha256,
            channel_space_contract_sha256=space_sha256,
            units=units,
        )
        for name, (values, semantic_type, space_sha256, units) in array_inputs.items()
    }
    payload: dict[str, object] = {
        "schema_id": STATE_LEAF_V2_SCHEMA,
        "legacy_root_sha256": state.root_sha256,
        "geometry_sha256": state.geometry_sha256,
        "provider_configuration_sha256": state.evaluator_configuration_sha256,
        "prepared_pes_configuration_sha256": prepared_pes_configuration,
        "topology_id": state.coefficient_topology_id,
        "target_charge_e": diagnostics.target_charge_e,
        "audit_coefficient_sum_charge_e": canonical_source_monopole_sum_v2(
            diagnostics.audit_coefficient_sum4
        ),
        "source_coefficient_basis_id": diagnostics.source_basis_id,
        "source_space_contract_sha256": diagnostics.source_space_contract_sha256,
        "source_coefficient_order": list(diagnostics.source_component_order),
        "receiver_space_id": diagnostics.receiver_basis_id,
        "receiver_space_contract_sha256": (
            diagnostics.receiver_space_contract_sha256
        ),
        "receiver_component_order": list(H1_V2_RECEIVER_COMPONENT_ORDER),
        "audit_projection_contract": h1_v2_audit_coefficient_sum_contract(),
        "numeric_array_encoding_contract": H1_V2_NUMERIC_ARRAY_ENCODING_CONTRACT,
        "endpoint_replay_scope": H1_V2_ENDPOINT_REPLAY_SCOPE,
        "permanent_point_source4": diagnostics.permanent_source4.tolist(),
        "polar_zero_reference_source4": diagnostics.response_zero_source4.tolist(),
        "polar_final_source4": diagnostics.response_final_source4.tolist(),
        "induced_gto_source4": diagnostics.induced_source4.tolist(),
        "audit_coefficient_sum4": diagnostics.audit_coefficient_sum4.tolist(),
        "array_content_sha256s": array_content_sha256s,
        "vacuum_energy_eV": state.vacuum_energy_ev,
        "polarization_energy_eV": state.polarization_energy_ev,
        "total_energy_eV": state.total_energy_ev,
        "cold_start": cold_payload,
        "wide_start": wide_payload,
    }
    mapping = {
        **payload,
        "state_leaf_sha256": canonical_json_sha256(payload),
    }
    return StateLeafV2.from_mapping(mapping).as_dict()


def validate_serialized_rich_state_recording_v2(
    value: object,
) -> dict[str, object]:
    """Validate and copy a deduplicated rich state/event recording."""

    from maple.solvation.release.admission import SolveEventV2, StateLeafV2

    raw = _exact_mapping(
        value, {"state_leaves", "solve_events"}, name="rich state recording"
    )
    raw_leaves = raw["state_leaves"]
    if not isinstance(raw_leaves, dict) or not raw_leaves:
        raise ValueError("state_leaves must be a nonempty digest mapping.")
    leaves = {}
    normalized_leaves = {}
    for digest, mapping in raw_leaves.items():
        _sha256_text(digest, name="state leaf mapping key")
        if not isinstance(mapping, dict):
            raise TypeError("serialized state leaf must be a JSON object.")
        leaf = StateLeafV2.from_mapping(mapping)
        if leaf.state_leaf_sha256 != digest:
            raise ValueError("state leaf mapping key disagrees with leaf digest.")
        leaves[digest] = leaf
        normalized_leaves[digest] = leaf.as_dict()
    raw_events = raw["solve_events"]
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("solve_events must be a nonempty ordered JSON array.")
    normalized_events = []
    for index, mapping in enumerate(raw_events):
        if not isinstance(mapping, dict):
            raise TypeError("serialized solve event must be a JSON object.")
        event = SolveEventV2.from_mapping(
            mapping, state_leaves=leaves, expected_index=index
        )
        normalized_events.append(
            {
                "schema_id": event.schema_id,
                "solve_event_sha256": event.solve_event_sha256,
                "event_index": event.event_index,
                "state_leaf_sha256": event.state_leaf_sha256,
                "geometry_sha256": event.geometry_sha256,
                "provider_configuration_sha256": (event.provider_configuration_sha256),
                "topology_id": event.topology_id,
            }
        )
    referenced_leaf_digests = {
        event["state_leaf_sha256"] for event in normalized_events
    }
    if set(normalized_leaves) != referenced_leaf_digests:
        raise ValueError(
            "state_leaves must equal the exact solve-event referenced digest set."
        )
    return json.loads(
        json.dumps(
            {
                "state_leaves": normalized_leaves,
                "solve_events": normalized_events,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


class RichHarmonicEFStateRecorderV2:
    """No-cache solve adapter with deduplicated leaves and ordered events."""

    __slots__ = ("_pes", "_solve_events", "_state_leaves")

    def __init__(self, pes: object) -> None:
        for name in ("configuration_sha256", "solve_with_diagnostics"):
            if not callable(getattr(pes, name, None)):
                raise TypeError(f"prepared harmonic PES requires callable {name}().")
        _sha256_text(
            pes.configuration_sha256(), name="prepared harmonic PES configuration"
        )
        self._pes = pes
        self._state_leaves: dict[str, dict[str, object]] = {}
        self._solve_events: list[dict[str, object]] = []

    @property
    def provider_id(self) -> str:
        return str(getattr(self._pes, "provider_id", ""))

    @property
    def event_count(self) -> int:
        return len(self._solve_events)

    def configuration_sha256(self) -> str:
        return _sha256_text(
            self._pes.configuration_sha256(),
            name="prepared harmonic PES configuration",
        )

    def solve(self, geometry: Atoms) -> HybridHarmonicEnergyState:
        """Execute one real solve and append one event; never reuse a state."""

        self.configuration_sha256()
        state, diagnostics = self._pes.solve_with_diagnostics(geometry)
        leaf_mapping = capture_rich_harmonic_ef_state_leaf_v2(
            state,
            diagnostics,
            prepared_pes_configuration_sha256=self.configuration_sha256(),
        )
        digest = str(leaf_mapping["state_leaf_sha256"])
        prior = self._state_leaves.get(digest)
        if prior is not None and prior != leaf_mapping:
            raise RuntimeError("state-leaf digest collision with unequal content.")
        self._state_leaves.setdefault(digest, leaf_mapping)

        from maple.solvation.release.admission import (
            SOLVE_EVENT_V2_SCHEMA,
            SolveEventV2,
            StateLeafV2,
        )

        index = len(self._solve_events)
        event_payload = {
            "schema_id": SOLVE_EVENT_V2_SCHEMA,
            "event_index": index,
            "state_leaf_sha256": digest,
            "geometry_sha256": leaf_mapping["geometry_sha256"],
            "provider_configuration_sha256": leaf_mapping[
                "provider_configuration_sha256"
            ],
            "topology_id": leaf_mapping["topology_id"],
        }
        event_mapping = {
            **event_payload,
            "solve_event_sha256": canonical_json_sha256(event_payload),
        }
        leaf = StateLeafV2.from_mapping(leaf_mapping)
        SolveEventV2.from_mapping(
            event_mapping,
            state_leaves={digest: leaf},
            expected_index=index,
        )
        self._solve_events.append(event_mapping)
        return state

    def sample(self, geometry: Atoms) -> ScalarEnergySample:
        state = self.solve(geometry)
        return ScalarEnergySample(
            energy_eV=state.total_energy_ev,
            state_sha256=state.root_sha256,
            topology_id=state.coefficient_topology_id,
        )

    def serialize(self) -> dict[str, object]:
        """Return a validated deep copy; caller mutation cannot affect recorder."""

        return validate_serialized_rich_state_recording_v2(
            {
                "state_leaves": self._state_leaves,
                "solve_events": self._solve_events,
            }
        )


def _validated_recorder_views(
    recorder: RichHarmonicEFStateRecorderV2,
):
    from maple.solvation.release.admission import SolveEventV2, StateLeafV2

    serialized = recorder.serialize()
    leaves = {
        digest: StateLeafV2.from_mapping(mapping)
        for digest, mapping in serialized["state_leaves"].items()
    }
    events = tuple(
        SolveEventV2.from_mapping(
            mapping,
            state_leaves=leaves,
            expected_index=index,
        )
        for index, mapping in enumerate(serialized["solve_events"])
    )
    return leaves, events


def _displaced_geometry(
    geometry: Atoms, *, atom_index: int, axis_index: int, displacement: float
) -> Atoms:
    if type(atom_index) is not int or not 0 <= atom_index < len(geometry):
        raise ValueError("atom_index is outside the geometry.")
    if type(axis_index) is not int or not 0 <= axis_index < 3:
        raise ValueError("axis_index must be zero, one, or two.")
    value = _finite_scalar(displacement, name="Cartesian displacement")
    result = geometry.copy()
    result.positions[atom_index, axis_index] += value
    return result


def record_component_stencil_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    center_event_index: int,
    atom_index: int,
    axis_index: int,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record +h,-h,+h/2,-h/2 after one explicit existing center event."""

    from maple.solvation.release.admission import ComponentStencilV2

    if not isinstance(recorder, RichHarmonicEFStateRecorderV2):
        raise TypeError("recorder must be RichHarmonicEFStateRecorderV2.")
    coarse = _finite_scalar(coarse_step_angstrom, name="coarse_step_angstrom")
    if coarse <= 0.0:
        raise ValueError("coarse_step_angstrom must be positive.")
    leaves, events = _validated_recorder_views(recorder)
    if type(center_event_index) is not int or not 0 <= center_event_index < len(events):
        raise ValueError("center_event_index must reference an existing solve event.")
    if events[center_event_index].geometry_sha256 != geometry_sha256(geometry):
        raise ValueError("center event is not bound to the supplied geometry.")
    indices = []
    for displacement in (coarse, -coarse, 0.5 * coarse, -0.5 * coarse):
        event_index = recorder.event_count
        recorder.solve(
            _displaced_geometry(
                geometry,
                atom_index=atom_index,
                axis_index=axis_index,
                displacement=displacement,
            )
        )
        indices.append(event_index)
    mapping = {
        "atom_index": atom_index,
        "axis_index": axis_index,
        "coarse_step_angstrom": coarse,
        "fine_step_angstrom": 0.5 * coarse,
        "center_event_index": center_event_index,
        "plus_h_event_index": indices[0],
        "minus_h_event_index": indices[1],
        "plus_h2_event_index": indices[2],
        "minus_h2_event_index": indices[3],
    }
    leaves, events = _validated_recorder_views(recorder)
    ComponentStencilV2.from_mapping(mapping, solve_events=events, state_leaves=leaves)
    return json.loads(json.dumps(mapping, sort_keys=True, allow_nan=False))


def record_cartesian_panel_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    label: str,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record one center and nine atom-major/axis-minor component stencils."""

    from maple.solvation.release.admission import CartesianPanelV2

    if len(geometry) != 3:
        raise ValueError("rich-v2 Cartesian panel requires exactly three atoms.")
    center_event_index = recorder.event_count
    recorder.solve(geometry.copy())
    components = [
        record_component_stencil_v2(
            recorder,
            geometry,
            center_event_index=center_event_index,
            atom_index=atom_index,
            axis_index=axis_index,
            coarse_step_angstrom=coarse_step_angstrom,
        )
        for atom_index in range(3)
        for axis_index in range(3)
    ]
    mapping = {"label": label, "components": components}
    leaves, events = _validated_recorder_views(recorder)
    CartesianPanelV2.from_mapping(mapping, solve_events=events, state_leaves=leaves)
    return json.loads(json.dumps(mapping, sort_keys=True, allow_nan=False))


def record_benzene_predecessor_stencil_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    center_event_index: int,
    atom_index: int,
    axis_index: int,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record the v1 component plus exact audit-only +h/4,-h/4 events."""

    from maple.solvation.release.admission import BenzenePredecessorStencilV2

    component = record_component_stencil_v2(
        recorder,
        geometry,
        center_event_index=center_event_index,
        atom_index=atom_index,
        axis_index=axis_index,
        coarse_step_angstrom=coarse_step_angstrom,
    )
    independent = 0.25 * float(coarse_step_angstrom)
    indices = []
    for displacement in (independent, -independent):
        event_index = recorder.event_count
        recorder.solve(
            _displaced_geometry(
                geometry,
                atom_index=atom_index,
                axis_index=axis_index,
                displacement=displacement,
            )
        )
        indices.append(event_index)
    mapping = {
        **component,
        "independent_step_angstrom": independent,
        "plus_h4_event_index": indices[0],
        "minus_h4_event_index": indices[1],
        "audit_only": True,
    }
    leaves, events = _validated_recorder_views(recorder)
    BenzenePredecessorStencilV2.from_mapping(
        mapping, solve_events=events, state_leaves=leaves
    )
    return json.loads(json.dumps(mapping, sort_keys=True, allow_nan=False))


def record_water_directional_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    independent_step_angstrom: float,
) -> dict[str, object]:
    """Record the frozen seeded directional +step,-step solve events."""

    step = _finite_scalar(independent_step_angstrom, name="independent_step_angstrom")
    if step <= 0.0:
        raise ValueError("independent_step_angstrom must be positive.")
    seed = 20260816
    direction = np.random.default_rng(seed).normal(size=geometry.positions.shape)
    direction /= np.linalg.norm(direction)
    indices = []
    for sign in (1.0, -1.0):
        event_index = recorder.event_count
        recorder.solve(
            _copy_with_positions(geometry, geometry.positions + sign * step * direction)
        )
        indices.append(event_index)
    return json.loads(
        json.dumps(
            {
                "seed": seed,
                "normalized_direction": direction.tolist(),
                "step_angstrom": step,
                "plus_event_index": indices[0],
                "minus_event_index": indices[1],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def record_translated_cartesian_panel_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record the exact frozen translation and its full Cartesian panel."""

    vector = np.asarray([4.2, -3.1, 1.7], dtype=float)
    translated = _copy_with_positions(geometry, geometry.positions + vector)
    panel = record_cartesian_panel_v2(
        recorder,
        translated,
        label="translated",
        coarse_step_angstrom=coarse_step_angstrom,
    )
    return json.loads(
        json.dumps(
            {
                "translation_angstrom": vector.tolist(),
                "panel": panel,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def record_rotated_cartesian_panel_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record the frozen proper rotation and its full Cartesian panel."""

    seed = 20260817
    rotation = _proper_rotation(seed)
    centroid = np.mean(geometry.positions, axis=0)
    rotated_positions = (geometry.positions - centroid) @ rotation.T + centroid
    rotated = _copy_with_positions(geometry, rotated_positions)
    panel = record_cartesian_panel_v2(
        recorder,
        rotated,
        label="rotated",
        coarse_step_angstrom=coarse_step_angstrom,
    )
    return json.loads(
        json.dumps(
            {
                "seed": seed,
                "rotation_matrix": rotation.tolist(),
                "panel": panel,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def record_closed_loop_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record the frozen four-edge counterclockwise raw loop graph."""

    dofs = ((0, 0), (1, 1))
    half_width = 1.0e-3
    (atom_x, axis_x), (atom_y, axis_y) = dofs
    edge_contract = (
        ("bottom", 0.0, -half_width, atom_x, axis_x, 2.0 * half_width),
        ("right", half_width, 0.0, atom_y, axis_y, 2.0 * half_width),
        ("top", 0.0, half_width, atom_x, axis_x, -2.0 * half_width),
        ("left", -half_width, 0.0, atom_y, axis_y, -2.0 * half_width),
    )
    edges = []
    for label, x, y, atom, axis, displacement in edge_contract:
        midpoint = geometry.copy()
        midpoint.positions[atom_x, axis_x] += x
        midpoint.positions[atom_y, axis_y] += y
        center_event_index = recorder.event_count
        recorder.solve(midpoint)
        component = record_component_stencil_v2(
            recorder,
            midpoint,
            center_event_index=center_event_index,
            atom_index=atom,
            axis_index=axis,
            coarse_step_angstrom=coarse_step_angstrom,
        )
        edges.append(
            {
                "label": label,
                "midpoint_offsets_angstrom": [x, y],
                "displacement_angstrom": displacement,
                "component": component,
            }
        )
    return json.loads(
        json.dumps(
            {
                "cartesian_dofs": [list(dof) for dof in dofs],
                "half_width_angstrom": half_width,
                "orientation": "counterclockwise",
                "edges": edges,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def record_benzene_system_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    atom_index: int,
    axis_index: int,
    coarse_step_angstrom: float,
) -> dict[str, object]:
    """Record the historical eight-occurrence benzene system graph."""

    if recorder.event_count != 0:
        raise ValueError("benzene system recorder must start with zero events.")
    recorder.solve(geometry.copy())
    predecessor = record_benzene_predecessor_stencil_v2(
        recorder,
        geometry,
        center_event_index=0,
        atom_index=atom_index,
        axis_index=axis_index,
        coarse_step_angstrom=coarse_step_angstrom,
    )
    reported_center_event_index = recorder.event_count
    recorder.solve(geometry.copy())
    if recorder.event_count != 8 or reported_center_event_index != 7:
        raise RuntimeError("benzene system occurrence count drifted from eight.")
    return json.loads(
        json.dumps(
            {
                "event_range": [0, 8],
                "predecessor_stencil": predecessor,
                "reported_center_event_index": 7,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def record_water_system_v2(
    recorder: RichHarmonicEFStateRecorderV2,
    geometry: Atoms,
    *,
    coarse_step_angstrom: float,
    independent_step_angstrom: float,
) -> dict[str, object]:
    """Record the historical 133-occurrence water system graph."""

    if recorder.event_count != 0:
        raise ValueError("water system recorder must start with zero events.")
    base = record_cartesian_panel_v2(
        recorder,
        geometry,
        label="base",
        coarse_step_angstrom=coarse_step_angstrom,
    )
    directional = record_water_directional_v2(
        recorder,
        geometry,
        independent_step_angstrom=independent_step_angstrom,
    )
    translation = record_translated_cartesian_panel_v2(
        recorder, geometry, coarse_step_angstrom=coarse_step_angstrom
    )
    rotation = record_rotated_cartesian_panel_v2(
        recorder, geometry, coarse_step_angstrom=coarse_step_angstrom
    )
    loop = record_closed_loop_v2(
        recorder, geometry, coarse_step_angstrom=coarse_step_angstrom
    )
    if recorder.event_count != 133:
        raise RuntimeError("water system occurrence count drifted from 133.")
    return json.loads(
        json.dumps(
            {
                "event_range": [0, 133],
                "base_panel": base,
                "directional": directional,
                "translation": translation,
                "rotation": rotation,
                "closed_loop": loop,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


_COMPONENT_EVENT_INDEX_FIELDS = (
    "center_event_index",
    "plus_h_event_index",
    "minus_h_event_index",
    "plus_h2_event_index",
    "minus_h2_event_index",
)
_COMPONENT_MAPPING_FIELDS = frozenset(
    {
        "atom_index",
        "axis_index",
        "coarse_step_angstrom",
        "fine_step_angstrom",
        *_COMPONENT_EVENT_INDEX_FIELDS,
    }
)


def _remap_component_mapping(value: object, *, event_offset: int) -> dict[str, object]:
    raw = _exact_mapping(value, set(_COMPONENT_MAPPING_FIELDS), name="component")
    return {
        key: (
            int(raw[key]) + event_offset
            if key in _COMPONENT_EVENT_INDEX_FIELDS
            else raw[key]
        )
        for key in raw
    }


def _remap_panel_mapping(value: object, *, event_offset: int) -> dict[str, object]:
    raw = _exact_mapping(value, {"label", "components"}, name="Cartesian panel")
    components = raw["components"]
    if not isinstance(components, list):
        raise TypeError("Cartesian panel components must be a JSON array.")
    return {
        "label": raw["label"],
        "components": [
            _remap_component_mapping(item, event_offset=event_offset)
            for item in components
        ],
    }


def _remap_benzene_system(value: object, *, event_offset: int) -> dict[str, object]:
    raw = _exact_mapping(
        value,
        {"event_range", "predecessor_stencil", "reported_center_event_index"},
        name="benzene system",
    )
    if raw["event_range"] != [0, 8] or raw["reported_center_event_index"] != 7:
        raise ValueError("benzene local occurrence range/center is invalid.")
    stencil = _exact_mapping(
        raw["predecessor_stencil"],
        set(_COMPONENT_MAPPING_FIELDS)
        | {
            "independent_step_angstrom",
            "plus_h4_event_index",
            "minus_h4_event_index",
            "audit_only",
        },
        name="benzene predecessor stencil",
    )
    component = _remap_component_mapping(
        {key: stencil[key] for key in _COMPONENT_MAPPING_FIELDS},
        event_offset=event_offset,
    )
    return {
        "event_range": [event_offset, event_offset + 8],
        "predecessor_stencil": {
            **component,
            "independent_step_angstrom": stencil["independent_step_angstrom"],
            "plus_h4_event_index": int(stencil["plus_h4_event_index"]) + event_offset,
            "minus_h4_event_index": int(stencil["minus_h4_event_index"]) + event_offset,
            "audit_only": stencil["audit_only"],
        },
        "reported_center_event_index": event_offset + 7,
    }


def _remap_water_system(value: object, *, event_offset: int) -> dict[str, object]:
    raw = _exact_mapping(
        value,
        {
            "event_range",
            "base_panel",
            "directional",
            "translation",
            "rotation",
            "closed_loop",
        },
        name="water system",
    )
    if raw["event_range"] != [0, 133]:
        raise ValueError("water local occurrence range is invalid.")
    directional = _exact_mapping(
        raw["directional"],
        {
            "seed",
            "normalized_direction",
            "step_angstrom",
            "plus_event_index",
            "minus_event_index",
        },
        name="directional",
    )
    translation = _exact_mapping(
        raw["translation"], {"translation_angstrom", "panel"}, name="translation"
    )
    rotation = _exact_mapping(
        raw["rotation"], {"seed", "rotation_matrix", "panel"}, name="rotation"
    )
    loop = _exact_mapping(
        raw["closed_loop"],
        {"cartesian_dofs", "half_width_angstrom", "orientation", "edges"},
        name="closed loop",
    )
    edges = loop["edges"]
    if not isinstance(edges, list):
        raise TypeError("closed-loop edges must be a JSON array.")
    remapped_edges = []
    for edge_value in edges:
        edge = _exact_mapping(
            edge_value,
            {
                "label",
                "midpoint_offsets_angstrom",
                "displacement_angstrom",
                "component",
            },
            name="closed-loop edge",
        )
        remapped_edges.append(
            {
                "label": edge["label"],
                "midpoint_offsets_angstrom": edge["midpoint_offsets_angstrom"],
                "displacement_angstrom": edge["displacement_angstrom"],
                "component": _remap_component_mapping(
                    edge["component"], event_offset=event_offset
                ),
            }
        )
    return {
        "event_range": [event_offset, event_offset + 133],
        "base_panel": _remap_panel_mapping(
            raw["base_panel"], event_offset=event_offset
        ),
        "directional": {
            "seed": directional["seed"],
            "normalized_direction": directional["normalized_direction"],
            "step_angstrom": directional["step_angstrom"],
            "plus_event_index": int(directional["plus_event_index"]) + event_offset,
            "minus_event_index": int(directional["minus_event_index"]) + event_offset,
        },
        "translation": {
            "translation_angstrom": translation["translation_angstrom"],
            "panel": _remap_panel_mapping(
                translation["panel"], event_offset=event_offset
            ),
        },
        "rotation": {
            "seed": rotation["seed"],
            "rotation_matrix": rotation["rotation_matrix"],
            "panel": _remap_panel_mapping(rotation["panel"], event_offset=event_offset),
        },
        "closed_loop": {
            "cartesian_dofs": loop["cartesian_dofs"],
            "half_width_angstrom": loop["half_width_angstrom"],
            "orientation": loop["orientation"],
            "edges": remapped_edges,
        },
    }


def merge_rich_harmonic_ef_recordings_v2(
    benzene_recording: object,
    water_recording: object,
    *,
    benzene_system: object,
    water_system: object,
    prepared_water_positions: tuple[tuple[float, float, float], ...],
) -> dict[str, object]:
    """Merge benzene then water event graphs with explicit index remapping."""

    from maple.solvation.release.admission import (
        BenzenePredecessorStencilV2,
        BenzeneSystemV2,
        CartesianPanelV2,
        ClosedLoopV2,
        DirectionalCheckV2,
        RigidRotationV2,
        RigidTranslationV2,
        SOLVE_EVENT_V2_SCHEMA,
        SolveEventV2,
        StateLeafV2,
        WaterSystemV2,
    )

    recordings = (
        validate_serialized_rich_state_recording_v2(benzene_recording),
        validate_serialized_rich_state_recording_v2(water_recording),
    )
    if len(recordings[0]["solve_events"]) != 8:
        raise ValueError("benzene recording must contain exactly eight occurrences.")
    if len(recordings[1]["solve_events"]) != 133:
        raise ValueError("water recording must contain exactly 133 occurrences.")
    global_leaves: dict[str, dict[str, object]] = {}
    global_events: list[dict[str, object]] = []
    offsets = []
    for recording in recordings:
        offsets.append(len(global_events))
        for digest, leaf in recording["state_leaves"].items():
            prior = global_leaves.get(digest)
            if prior is not None and prior != leaf:
                raise RuntimeError("global state-leaf digest collision.")
            global_leaves.setdefault(digest, leaf)
        for event in recording["solve_events"]:
            event_index = len(global_events)
            payload = {
                "schema_id": SOLVE_EVENT_V2_SCHEMA,
                "event_index": event_index,
                "state_leaf_sha256": event["state_leaf_sha256"],
                "geometry_sha256": event["geometry_sha256"],
                "provider_configuration_sha256": event["provider_configuration_sha256"],
                "topology_id": event["topology_id"],
            }
            global_events.append(
                {**payload, "solve_event_sha256": canonical_json_sha256(payload)}
            )
    leaf_objects = {
        digest: StateLeafV2.from_mapping(mapping)
        for digest, mapping in global_leaves.items()
    }
    event_objects = tuple(
        SolveEventV2.from_mapping(
            event,
            state_leaves=leaf_objects,
            expected_index=index,
        )
        for index, event in enumerate(global_events)
    )
    validate_serialized_rich_state_recording_v2(
        {"state_leaves": global_leaves, "solve_events": global_events}
    )
    systems = {
        "benzene": _remap_benzene_system(benzene_system, event_offset=offsets[0]),
        "water": _remap_water_system(water_system, event_offset=offsets[1]),
    }
    BenzenePredecessorStencilV2.from_mapping(
        systems["benzene"]["predecessor_stencil"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
    )
    base_panel = CartesianPanelV2.from_mapping(
        systems["water"]["base_panel"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
    )
    DirectionalCheckV2.from_mapping(
        systems["water"]["directional"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
        base_panel=base_panel,
        prepared_water_positions=prepared_water_positions,
    )
    RigidTranslationV2.from_mapping(
        systems["water"]["translation"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
        prepared_water_positions=prepared_water_positions,
    )
    RigidRotationV2.from_mapping(
        systems["water"]["rotation"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
        prepared_water_positions=prepared_water_positions,
    )
    ClosedLoopV2.from_mapping(
        systems["water"]["closed_loop"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
        prepared_water_positions=prepared_water_positions,
    )
    BenzeneSystemV2.from_mapping(
        systems["benzene"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
        prepared_pes_configuration_sha256=leaf_objects[
            event_objects[0].state_leaf_sha256
        ].prepared_pes_configuration_sha256,
    )
    WaterSystemV2.from_mapping(
        systems["water"],
        solve_events=event_objects,
        state_leaves=leaf_objects,
        prepared_water_positions=prepared_water_positions,
        prepared_pes_configuration_sha256=leaf_objects[
            event_objects[8].state_leaf_sha256
        ].prepared_pes_configuration_sha256,
    )
    return json.loads(
        json.dumps(
            {
                "state_leaves": global_leaves,
                "solve_events": global_events,
                "systems": systems,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


_PREPARED_PROVIDER_CONTRACT_FIELDS = {
    "profile_id",
    "scalar_id",
    "state_id",
    "checkpoint_sha256s",
    "provider_configuration_sha256s",
}
_PREPARED_PROVIDER_CONFIGURATION_KEYS = {
    "hybrid_configuration",
    "hybrid_provenance",
    "continuum_configuration",
    "continuum_provenance",
    "cavity_configuration",
    "prepared_input_manifest",
    "force_panel_contract",
    "benzene_pes_configuration",
    "water_pes_configuration",
}


def _normalized_prepared_provider_contract(value: object) -> dict[str, object]:
    raw = _exact_mapping(
        value,
        _PREPARED_PROVIDER_CONTRACT_FIELDS,
        name="prepared_provider_contract",
    )
    for field in ("profile_id", "scalar_id", "state_id"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise ValueError(f"prepared_provider_contract.{field} must be nonempty.")
    checkpoints = _exact_mapping(
        raw["checkpoint_sha256s"],
        {"mace_mdp_checkpoint", "mace_polar_checkpoint"},
        name="prepared provider checkpoints",
    )
    configurations = _exact_mapping(
        raw["provider_configuration_sha256s"],
        _PREPARED_PROVIDER_CONFIGURATION_KEYS,
        name="prepared provider configurations",
    )
    for name, digest in (*checkpoints.items(), *configurations.items()):
        _sha256_text(digest, name=f"prepared provider digest {name}")
    if len(set((*checkpoints.values(), *configurations.values()))) != (
        len(checkpoints) + len(configurations)
    ):
        raise ValueError("prepared provider contract contains digest aliases.")
    return json.loads(json.dumps(raw, sort_keys=True, allow_nan=False))


def _prepared_pes_identity(
    pes: object,
    prepared: PreparedHarmonicEFBenzeneInputV2 | PreparedHarmonicEFWaterInputV2,
    *,
    name: str,
    expected_profile_id: str,
    expected_scalar_id: str,
    expected_state_id: str,
    expected_configuration_sha256: str,
) -> dict[str, str]:
    """Validate the explicit prepared provider identity without hash inference."""

    if not isinstance(pes, H1PreparedHarmonicEFPESV2):
        raise TypeError(f"{name} PES must be H1PreparedHarmonicEFPESV2.")
    if pes.system_role != name:
        raise ValueError(f"{name} PES adapter has the wrong system role.")
    configuration = getattr(pes, "configuration_sha256", None)
    solve = getattr(pes, "solve_with_diagnostics", None)
    if not callable(configuration) or not callable(solve):
        raise TypeError(f"{name} PES omits its prepared solve/configuration interface.")
    digest = _sha256_text(configuration(), name=f"{name} PES configuration")
    numbers = getattr(pes, "prepared_atomic_numbers", None)
    radii = getattr(pes, "prepared_cavity_radii_angstrom", None)
    charge = getattr(pes, "prepared_charge", None)
    multiplicity = getattr(pes, "prepared_multiplicity", None)
    profile_id = getattr(pes, "prepared_profile_id", None)
    scalar_id = getattr(pes, "prepared_scalar_id", None)
    provider_id = getattr(pes, "prepared_provider_id", None)
    state_id = getattr(pes, "prepared_state_id", None)
    prepared_configuration = getattr(pes, "prepared_configuration_sha256", None)
    if numbers is None or radii is None or charge is None or multiplicity is None:
        raise TypeError(f"{name} PES omits explicit prepared system/domain identity.")
    if (
        not isinstance(profile_id, str)
        or not isinstance(scalar_id, str)
        or not isinstance(provider_id, str)
        or not isinstance(state_id, str)
        or not (
            isinstance(prepared_configuration, str) or callable(prepared_configuration)
        )
    ):
        raise TypeError(f"{name} PES omits explicit profile/scalar/provider identity.")
    if tuple(numbers) != tuple(prepared.atomic_numbers):
        raise ValueError(f"{name} PES atomic numbers differ from prepared inputs.")
    if tuple(float(item) for item in radii) != tuple(prepared.cavity_radii_angstrom):
        raise ValueError(f"{name} PES cavity radii differ from prepared inputs.")
    if charge != prepared.charge or multiplicity != prepared.multiplicity:
        raise ValueError(f"{name} PES charge/multiplicity domain differs from inputs.")
    if profile_id != expected_profile_id or scalar_id != expected_scalar_id:
        raise ValueError(f"{name} PES profile/scalar differs from provider contract.")
    if state_id != expected_state_id:
        raise ValueError(f"{name} PES state identity differs from provider contract.")
    if provider_id != getattr(pes, "provider_id", None):
        raise ValueError(f"{name} PES prepared provider identity is inconsistent.")
    prepared_digest = _sha256_text(
        (
            prepared_configuration()
            if callable(prepared_configuration)
            else prepared_configuration
        ),
        name=f"{name} prepared configuration",
    )
    if digest != prepared_digest or digest != expected_configuration_sha256:
        raise ValueError(f"{name} PES configuration differs from provider contract.")
    return {
        "profile_id": profile_id,
        "scalar_id": scalar_id,
        "provider_id": provider_id,
        "configuration_sha256": digest,
    }


def run_rich_harmonic_ef_raw_measurement_v2(
    prepared_inputs: PreparedHarmonicEFInputsV2,
    benzene_pes: object,
    water_pes: object,
    *,
    prepared_provider_contract: object,
) -> dict[str, object]:
    """Build only the complete I/O-free raw 141-occurrence science tree."""

    if not isinstance(prepared_inputs, PreparedHarmonicEFInputsV2):
        raise TypeError("prepared_inputs must be PreparedHarmonicEFInputsV2.")
    provider_contract = _normalized_prepared_provider_contract(
        prepared_provider_contract
    )
    force_panel = _mutable_prepared_json(prepared_inputs.force_panel_contract)
    if (
        canonical_json_sha256(force_panel)
        != prepared_inputs.force_panel_contract_sha256
    ):
        raise ValueError("prepared force-panel digest changed before measurement.")
    if (
        provider_contract["provider_configuration_sha256s"]["force_panel_contract"]
        != prepared_inputs.force_panel_contract_sha256
    ):
        raise ValueError("provider contract does not bind the prepared force panel.")
    provider_configurations = provider_contract["provider_configuration_sha256s"]
    if provider_configurations["force_panel_contract"] != (
        prepared_inputs.force_panel_contract_sha256
    ):
        raise ValueError("provider contract does not bind prepared force-panel digest.")
    benzene_identity = _prepared_pes_identity(
        benzene_pes,
        prepared_inputs.benzene,
        name="benzene",
        expected_profile_id=provider_contract["profile_id"],
        expected_scalar_id=provider_contract["scalar_id"],
        expected_state_id=provider_contract["state_id"],
        expected_configuration_sha256=provider_configurations[
            "benzene_pes_configuration"
        ],
    )
    water_identity = _prepared_pes_identity(
        water_pes,
        prepared_inputs.water,
        name="water",
        expected_profile_id=provider_contract["profile_id"],
        expected_scalar_id=provider_contract["scalar_id"],
        expected_state_id=provider_contract["state_id"],
        expected_configuration_sha256=provider_configurations[
            "water_pes_configuration"
        ],
    )
    benzene = Atoms(
        numbers=prepared_inputs.benzene.atomic_numbers,
        positions=prepared_inputs.benzene.positions_angstrom,
        info={"charge": 0, "multiplicity": 1, "mult": 1},
    )
    water = Atoms(
        numbers=prepared_inputs.water.atomic_numbers,
        positions=prepared_inputs.water.positions_angstrom,
        info={"charge": 0, "multiplicity": 1, "mult": 1},
    )
    benzene_recorder = RichHarmonicEFStateRecorderV2(benzene_pes)
    water_recorder = RichHarmonicEFStateRecorderV2(water_pes)
    benzene_system = record_benzene_system_v2(
        benzene_recorder,
        benzene,
        atom_index=prepared_inputs.benzene.predecessor_dof[0],
        axis_index=prepared_inputs.benzene.predecessor_dof[1],
        coarse_step_angstrom=HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "force_coarse_step_angstrom"
        ],
    )
    water_system = record_water_system_v2(
        water_recorder,
        water,
        coarse_step_angstrom=HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "force_coarse_step_angstrom"
        ],
        independent_step_angstrom=HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "independent_force_step_angstrom"
        ],
    )
    final_benzene_identity = _prepared_pes_identity(
        benzene_pes,
        prepared_inputs.benzene,
        name="benzene",
        expected_profile_id=provider_contract["profile_id"],
        expected_scalar_id=provider_contract["scalar_id"],
        expected_state_id=provider_contract["state_id"],
        expected_configuration_sha256=provider_configurations[
            "benzene_pes_configuration"
        ],
    )
    final_water_identity = _prepared_pes_identity(
        water_pes,
        prepared_inputs.water,
        name="water",
        expected_profile_id=provider_contract["profile_id"],
        expected_scalar_id=provider_contract["scalar_id"],
        expected_state_id=provider_contract["state_id"],
        expected_configuration_sha256=provider_configurations[
            "water_pes_configuration"
        ],
    )
    if final_benzene_identity != benzene_identity:
        raise RuntimeError("benzene PES identity changed during measurement.")
    if final_water_identity != water_identity:
        raise RuntimeError("water PES identity changed during measurement.")
    return merge_rich_harmonic_ef_recordings_v2(
        benzene_recorder.serialize(),
        water_recorder.serialize(),
        benzene_system=benzene_system,
        water_system=water_system,
        prepared_water_positions=prepared_inputs.water.positions_angstrom,
    )


def project_rich_harmonic_ef_raw_tree_to_v1_science_v2(
    raw_tree: object,
    prepared_inputs: PreparedHarmonicEFInputsV2,
    prepared_provider_contract: object,
) -> dict[str, object]:
    """Project validated rich leaves to the archive-only historical v1 shape."""

    if not isinstance(prepared_inputs, PreparedHarmonicEFInputsV2):
        raise TypeError("prepared_inputs must be PreparedHarmonicEFInputsV2.")
    provider_contract = _normalized_prepared_provider_contract(
        prepared_provider_contract
    )
    force_panel = _mutable_prepared_json(prepared_inputs.force_panel_contract)
    if (
        canonical_json_sha256(force_panel)
        != prepared_inputs.force_panel_contract_sha256
    ):
        raise ValueError("prepared force-panel digest changed before projection.")
    if (
        provider_contract["provider_configuration_sha256s"]["force_panel_contract"]
        != prepared_inputs.force_panel_contract_sha256
    ):
        raise ValueError("provider contract does not bind the prepared force panel.")
    raw = _exact_mapping(
        raw_tree, {"state_leaves", "solve_events", "systems"}, name="raw tree"
    )
    recording = validate_serialized_rich_state_recording_v2(
        {
            "state_leaves": raw["state_leaves"],
            "solve_events": raw["solve_events"],
        }
    )
    systems_raw = _exact_mapping(
        raw["systems"], {"benzene", "water"}, name="raw systems"
    )
    from maple.solvation.derivatives.scalar_finite_difference import (
        RICHARDSON_FORCE_CONTRACT,
    )
    from maple.solvation.release.admission import (
        BenzeneSystemV2,
        SolveEventV2,
        StateLeafV2,
        WaterSystemV2,
    )

    leaves = {
        digest: StateLeafV2.from_mapping(mapping)
        for digest, mapping in recording["state_leaves"].items()
    }
    events = tuple(
        SolveEventV2.from_mapping(mapping, state_leaves=leaves, expected_index=index)
        for index, mapping in enumerate(recording["solve_events"])
    )
    configurations = provider_contract["provider_configuration_sha256s"]
    benzene_system = BenzeneSystemV2.from_mapping(
        systems_raw["benzene"],
        solve_events=events,
        state_leaves=leaves,
        prepared_pes_configuration_sha256=configurations["benzene_pes_configuration"],
    )
    water_system = WaterSystemV2.from_mapping(
        systems_raw["water"],
        solve_events=events,
        state_leaves=leaves,
        prepared_water_positions=prepared_inputs.water.positions_angstrom,
        prepared_pes_configuration_sha256=configurations["water_pes_configuration"],
    )

    def leaf_at(index: int):
        return leaves[events[index].state_leaf_sha256]

    def start_dict(leaf, name: str) -> dict[str, object]:
        return leaf.as_dict()[name]

    def root_summary(start: int, end: int) -> dict[str, object]:
        selected = [leaf_at(index) for index in range(start, end)]
        cold = [start_dict(leaf, "cold_start") for leaf in selected]
        wide = [start_dict(leaf, "wide_start") for leaf in selected]
        return {
            "state_count": len(selected),
            "maximum_primal_residual_eV": max(
                float(np.linalg.norm(item["final_residual_eV"])) for item in cold
            ),
            "maximum_replay_field_difference_eV": max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(cold_item["final_native_field_eV"])
                            - np.asarray(wide_item["final_native_field_eV"])
                        )
                    )
                )
                for cold_item, wide_item in zip(cold, wide, strict=True)
            ),
            "maximum_replay_energy_difference_eV": max(
                abs(
                    cold_item["final_total_energy_eV"]
                    - wide_item["final_total_energy_eV"]
                )
                for cold_item, wide_item in zip(cold, wide, strict=True)
            ),
            "maximum_total_charge_error_e": max(
                abs(leaf.audit_coefficient_sum_charge_e - leaf.target_charge_e)
                for leaf in selected
            ),
            "maximum_cold_iterations": max(item["iterations"] for item in cold),
            "maximum_wide_iterations": max(item["iterations"] for item in wide),
            "topology_ids": sorted({leaf.topology_id for leaf in selected}),
        }

    benzene_component = benzene_system.predecessor_stencil.component
    b_indices = benzene_component.event_indices
    b_leaves = [leaf_at(index) for index in b_indices]
    derivative_h = (b_leaves[1].total_energy_eV - b_leaves[2].total_energy_eV) / (
        2.0 * benzene_component.coarse_step_angstrom
    )
    derivative_h2 = (b_leaves[3].total_energy_eV - b_leaves[4].total_energy_eV) / (
        2.0 * benzene_component.fine_step_angstrom
    )
    h4_indices = benzene_system.predecessor_stencil.h4_event_indices
    h4_leaves = [leaf_at(index) for index in h4_indices]
    derivative_h4 = (h4_leaves[0].total_energy_eV - h4_leaves[1].total_energy_eV) / (
        2.0 * benzene_system.predecessor_stencil.independent_step_angstrom
    )
    richardson = (4.0 * derivative_h2 - derivative_h) / 3.0
    local_error = abs(richardson - derivative_h2)
    h4_difference = abs(richardson - derivative_h4)
    benzene_center = leaf_at(benzene_system.reported_center_event_index)
    convergence = {
        "atom_index": benzene_component.atom_index,
        "axis_index": benzene_component.axis_index,
        "coarse_step_angstrom": benzene_component.coarse_step_angstrom,
        "fine_step_angstrom": benzene_component.fine_step_angstrom,
        "independent_step_angstrom": (
            benzene_system.predecessor_stencil.independent_step_angstrom
        ),
        "energy_derivative_h_eV_per_A": derivative_h,
        "energy_derivative_h2_eV_per_A": derivative_h2,
        "energy_derivative_h4_eV_per_A": derivative_h4,
        "richardson_energy_derivative_eV_per_A": richardson,
        "richardson_force_eV_per_A": -richardson,
        "local_error_estimate_eV_per_A": local_error,
        "h4_difference_eV_per_A": h4_difference,
        "maximum_local_error_eV_per_A": HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "force_maximum_local_error_estimate_ev_per_angstrom"
        ],
        "maximum_h4_difference_eV_per_A": HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_benzene_h4_difference_ev_per_angstrom"
        ],
        "topology_id": b_leaves[0].topology_id,
        "center_state_sha256": b_leaves[0].legacy_root_sha256,
        "displaced_state_sha256": {
            "plus_h": b_leaves[1].legacy_root_sha256,
            "minus_h": b_leaves[2].legacy_root_sha256,
            "plus_h2": b_leaves[3].legacy_root_sha256,
            "minus_h2": b_leaves[4].legacy_root_sha256,
            "plus_h4": h4_leaves[0].legacy_root_sha256,
            "minus_h4": h4_leaves[1].legacy_root_sha256,
        },
        "gate_passed": bool(
            local_error
            <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
                "force_maximum_local_error_estimate_ev_per_angstrom"
            ]
            and h4_difference
            <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
                "maximum_benzene_h4_difference_ev_per_angstrom"
            ]
        ),
    }
    benzene = {
        "compound_id": prepared_inputs.benzene.compound_id,
        "name": prepared_inputs.benzene.name,
        "atom_count": len(prepared_inputs.benzene.atomic_numbers),
        "cavity_radii_angstrom": list(prepared_inputs.benzene.cavity_radii_angstrom),
        "center_energy_eV": benzene_center.total_energy_eV,
        "center_polarization_energy_eV": benzene_center.polarization_energy_eV,
        "center_root_sha256": benzene_center.legacy_root_sha256,
        "coefficient_topology_id": benzene_center.topology_id,
        "force_convergence": convergence,
        "root_summary": root_summary(0, 8),
        "asset_sha256": {
            "mol2": prepared_inputs.benzene.mol2_sha256,
            "projection_result": prepared_inputs.benzene.projection_result_sha256,
        },
        "gate_passed": convergence["gate_passed"],
    }

    def force_record(panel, configuration: str) -> dict[str, object]:
        components = panel.components
        center = leaf_at(components[0].event_indices[0])
        forces = np.asarray(
            [item.richardson_force_eV_per_angstrom for item in components]
        ).reshape(3, 3)
        errors = np.asarray(
            [item.local_error_eV_per_angstrom for item in components]
        ).reshape(3, 3)
        displaced = tuple(
            leaf_at(index).legacy_root_sha256
            for component in components
            for index in component.event_indices[1:]
        )
        evaluation = RichardsonScalarForceEvaluation(
            contract_id=RICHARDSON_FORCE_CONTRACT,
            provider_configuration_sha256=configuration,
            central_sample=ScalarEnergySample(
                energy_eV=center.total_energy_eV,
                state_sha256=center.legacy_root_sha256,
                topology_id=center.topology_id,
            ),
            coarse_step_angstrom=components[0].coarse_step_angstrom,
            fine_step_angstrom=components[0].fine_step_angstrom,
            forces_eV_per_A=forces,
            error_estimates_eV_per_A=errors,
            displaced_state_sha256=displaced,
        )
        return _force_record(evaluation)

    water_configuration = configurations["water_pes_configuration"]
    center_force = force_record(water_system.base_panel, water_configuration)
    translated_force = force_record(water_system.translation.panel, water_configuration)
    rotated_force = force_record(water_system.rotation.panel, water_configuration)
    direction_raw = systems_raw["water"]["directional"]
    direction_indices = water_system.directional.event_indices
    direction_leaves = [leaf_at(index) for index in direction_indices]
    direction = {
        "seed": direction_raw["seed"],
        "step_angstrom": direction_raw["step_angstrom"],
        "direction": direction_raw["normalized_direction"],
        "energy_derivative_eV_per_A": (
            water_system.directional.energy_derivative_eV_per_angstrom
        ),
        "negative_force_projection_eV_per_A": (
            water_system.directional.negative_force_projection_eV_per_angstrom
        ),
        "absolute_error_eV_per_A": (
            water_system.directional.absolute_error_eV_per_angstrom
        ),
        "maximum_error_eV_per_A": HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_independent_directional_error_ev_per_angstrom"
        ],
        "plus_state_sha256": direction_leaves[0].legacy_root_sha256,
        "minus_state_sha256": direction_leaves[1].legacy_root_sha256,
        "gate_passed": bool(
            water_system.directional.absolute_error_eV_per_angstrom
            <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
                "maximum_independent_directional_error_ev_per_angstrom"
            ]
        ),
    }
    base_center = leaf_at(water_system.base_panel.components[0].event_indices[0])
    translated_center = leaf_at(
        water_system.translation.panel.components[0].event_indices[0]
    )
    rotated_center = leaf_at(water_system.rotation.panel.components[0].event_indices[0])
    translation = {
        "translation_angstrom": list(water_system.translation.translation_angstrom),
        "energy_absolute_error_eV": (water_system.translation_energy_abs_difference_eV),
        "force_relative_error": water_system.translation_force_relative_difference,
        "force_maximum_absolute_error_eV_per_A": (
            water_system.translation_force_max_abs_difference_eV_per_angstrom
        ),
        "translated_force": translated_force,
        "translated_root_sha256": translated_center.legacy_root_sha256,
    }
    translation["gate_passed"] = bool(
        translation["energy_absolute_error_eV"]
        <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS["maximum_translation_energy_error_ev"]
        and translation["force_relative_error"]
        <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_translation_force_relative_error"
        ]
        and translation["force_maximum_absolute_error_eV_per_A"]
        <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_translation_force_absolute_error_ev_per_angstrom"
        ]
    )
    rotation_raw = systems_raw["water"]["rotation"]
    rotation = {
        "seed": rotation_raw["seed"],
        "rotation_matrix": rotation_raw["rotation_matrix"],
        "energy_absolute_error_eV": water_system.rotation_energy_abs_difference_eV,
        "force_relative_error": (
            water_system.rotation_force_covariance_relative_difference
        ),
        "force_maximum_absolute_error_eV_per_A": (
            water_system.rotation_force_covariance_max_abs_difference_eV_per_angstrom
        ),
        "rotated_force": rotated_force,
        "rotated_root_sha256": rotated_center.legacy_root_sha256,
    }
    rotation["gate_passed"] = bool(
        rotation["energy_absolute_error_eV"]
        <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS["maximum_rotation_energy_error_ev"]
        and rotation["force_relative_error"]
        <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS["maximum_rotation_force_relative_error"]
        and rotation["force_maximum_absolute_error_eV_per_A"]
        <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_rotation_force_absolute_error_ev_per_angstrom"
        ]
    )
    loop_raw = systems_raw["water"]["closed_loop"]
    loop_edges = []
    for raw_edge, component in zip(
        loop_raw["edges"], water_system.closed_loop.components, strict=True
    ):
        displaced = [
            leaf_at(index).legacy_root_sha256 for index in component.event_indices[1:]
        ]
        contribution = (
            component.richardson_force_eV_per_angstrom
            * raw_edge["displacement_angstrom"]
        )
        error_bound = component.local_error_eV_per_angstrom * abs(
            raw_edge["displacement_angstrom"]
        )
        loop_edges.append(
            {
                "label": raw_edge["label"],
                "midpoint_offsets_angstrom": raw_edge["midpoint_offsets_angstrom"],
                "displacement_angstrom": raw_edge["displacement_angstrom"],
                "work_eV": contribution,
                "work_error_bound_eV": error_bound,
                "force_component": {
                    "atom_index": component.atom_index,
                    "axis_index": component.axis_index,
                    "force_eV_per_A": component.richardson_force_eV_per_angstrom,
                    "error_estimate_eV_per_A": component.local_error_eV_per_angstrom,
                    "displaced_state_sha256": displaced,
                },
            }
        )
    loop = {
        "cartesian_dofs": [
            list(dof) for dof in water_system.closed_loop.cartesian_dofs
        ],
        "half_width_angstrom": water_system.closed_loop.half_width_angstrom,
        "orientation": water_system.closed_loop.orientation,
        "edges": loop_edges,
        "closed_loop_work_eV": water_system.closed_loop.closed_loop_work_eV,
        "absolute_work_eV": abs(water_system.closed_loop.closed_loop_work_eV),
        "numerical_work_error_bound_eV": (
            water_system.closed_loop.numerical_work_error_bound_eV
        ),
        "guarded_absolute_work_eV": (water_system.closed_loop.guarded_absolute_work_eV),
        "maximum_guarded_absolute_work_eV": HARMONIC_EF_FORCE_PANEL_THRESHOLDS[
            "maximum_closed_loop_work_abs_ev"
        ],
        "gate_passed": bool(
            water_system.closed_loop.guarded_absolute_work_eV
            <= HARMONIC_EF_FORCE_PANEL_THRESHOLDS["maximum_closed_loop_work_abs_ev"]
        ),
    }
    water_root_summary = root_summary(8, 141)
    water_root_gate = bool(
        water_root_summary["maximum_primal_residual_eV"] < 1.0e-10
        and water_root_summary["maximum_replay_field_difference_eV"] <= 2.0e-9
        and water_root_summary["maximum_replay_energy_difference_eV"] <= 1.0e-10
        and water_root_summary["maximum_total_charge_error_e"] <= 1.0e-8
        and len(water_root_summary["topology_ids"]) == 1
    )
    water_gate = bool(
        center_force["finite"]
        and center_force["maximum_error_estimate_eV_per_A"] <= 2.0e-4
        and center_force["net_force_norm_eV_per_A"] <= 1.0e-3
        and direction["gate_passed"]
        and translation["gate_passed"]
        and rotation["gate_passed"]
        and loop["gate_passed"]
        and water_root_gate
    )
    water = {
        "system": "water",
        "geometry_angstrom": [
            list(row) for row in prepared_inputs.water.positions_angstrom
        ],
        "cavity_radii_angstrom": list(prepared_inputs.water.cavity_radii_angstrom),
        "center_energy_eV": base_center.total_energy_eV,
        "center_polarization_energy_eV": base_center.polarization_energy_eV,
        "center_root_sha256": base_center.legacy_root_sha256,
        "coefficient_topology_id": base_center.topology_id,
        "center_force": center_force,
        "independent_h4_directional_check": direction,
        "translation": translation,
        "rotation": rotation,
        "closed_loop": loop,
        "root_summary": water_root_summary,
        "root_gate_passed": water_root_gate,
        "gate_passed": water_gate,
    }
    maximum_local_error = max(
        local_error,
        center_force["maximum_error_estimate_eV_per_A"],
        translated_force["maximum_error_estimate_eV_per_A"],
        rotated_force["maximum_error_estimate_eV_per_A"],
        *(edge["force_component"]["error_estimate_eV_per_A"] for edge in loop_edges),
    )
    aggregate = {
        "maximum_local_richardson_error_eV_per_A": maximum_local_error,
        "benzene_h4_difference_eV_per_A": h4_difference,
        "water_independent_directional_error_eV_per_A": direction[
            "absolute_error_eV_per_A"
        ],
        "water_translation_energy_error_eV": translation["energy_absolute_error_eV"],
        "water_translation_force_relative_error": translation["force_relative_error"],
        "water_rotation_energy_error_eV": rotation["energy_absolute_error_eV"],
        "water_rotation_force_relative_error": rotation["force_relative_error"],
        "water_closed_loop_guarded_absolute_work_eV": loop["guarded_absolute_work_eV"],
        "all_execution_gates_passed": bool(
            benzene["gate_passed"] and water_gate and maximum_local_error <= 2.0e-4
        ),
    }
    science = {
        "benzene_gepol_regression": benzene,
        "water_force_symmetry_loop": water,
        "aggregate": aggregate,
    }
    validate_harmonic_ef_science_measurement(
        science,
        runtime_contract=_resolved_runtime_contract(None),
        force_panel=force_panel,
    )
    derive_harmonic_ef_replicate_gates(
        science,
        runtime_contract=_resolved_runtime_contract(None),
        force_panel=force_panel,
        clean_content_addressed_execution=True,
    )
    return json.loads(json.dumps(science, sort_keys=True, allow_nan=False))


def _validate_root_summary(value: object, *, name: str) -> None:
    summary = _exact_mapping(
        value,
        {
            "state_count",
            "maximum_primal_residual_eV",
            "maximum_replay_field_difference_eV",
            "maximum_replay_energy_difference_eV",
            "maximum_total_charge_error_e",
            "maximum_cold_iterations",
            "maximum_wide_iterations",
            "topology_ids",
        },
        name=name,
    )
    _integer(summary["state_count"], name=f"{name}.state_count", minimum=1)
    _integer(
        summary["maximum_cold_iterations"],
        name=f"{name}.maximum_cold_iterations",
        minimum=1,
    )
    _integer(
        summary["maximum_wide_iterations"],
        name=f"{name}.maximum_wide_iterations",
        minimum=1,
    )
    for key in (
        "maximum_primal_residual_eV",
        "maximum_replay_field_difference_eV",
        "maximum_replay_energy_difference_eV",
        "maximum_total_charge_error_e",
    ):
        if _finite_scalar(summary[key], name=f"{name}.{key}") < 0.0:
            raise ValueError(f"{name}.{key} must be nonnegative.")
    topology_ids = summary["topology_ids"]
    if not isinstance(topology_ids, list) or len(topology_ids) != 1:
        raise ValueError(f"{name}.topology_ids must contain exactly one digest.")
    _sha256_text(topology_ids[0], name=f"{name}.topology_ids[0]")


def _validate_force_record(value: object, *, atom_count: int, name: str) -> None:
    record = _exact_mapping(
        value,
        {
            "evaluation_sha256",
            "forces_eV_per_A",
            "error_estimates_eV_per_A",
            "maximum_error_estimate_eV_per_A",
            "net_force_norm_eV_per_A",
            "finite",
        },
        name=name,
    )
    _sha256_text(record["evaluation_sha256"], name=f"{name}.evaluation_sha256")
    forces = _finite_array(
        record["forces_eV_per_A"], (atom_count, 3), name=f"{name}.forces_eV_per_A"
    )
    errors = _finite_array(
        record["error_estimates_eV_per_A"],
        (atom_count, 3),
        name=f"{name}.error_estimates_eV_per_A",
    )
    if np.any(errors < 0.0):
        raise ValueError(f"{name}.error_estimates_eV_per_A must be nonnegative.")
    _require_close(
        record["maximum_error_estimate_eV_per_A"],
        float(np.max(errors)),
        name=f"{name}.maximum_error_estimate_eV_per_A",
    )
    _require_close(
        record["net_force_norm_eV_per_A"],
        float(np.linalg.norm(np.sum(forces, axis=0))),
        name=f"{name}.net_force_norm_eV_per_A",
    )
    if _exact_bool(record["finite"], name=f"{name}.finite") is not True:
        raise ValueError(
            f"engine-derived gates: {name}.finite contradicts its finite arrays."
        )


def _validate_component_record(value: object, *, atom_count: int, name: str) -> None:
    record = _exact_mapping(
        value,
        {
            "atom_index",
            "axis_index",
            "force_eV_per_A",
            "error_estimate_eV_per_A",
            "displaced_state_sha256",
        },
        name=name,
    )
    atom_index = _integer(record["atom_index"], name=f"{name}.atom_index")
    axis_index = _integer(record["axis_index"], name=f"{name}.axis_index")
    if atom_index >= atom_count or axis_index >= 3:
        raise ValueError(f"{name} Cartesian index is out of range.")
    _finite_scalar(record["force_eV_per_A"], name=f"{name}.force_eV_per_A")
    error = _finite_scalar(
        record["error_estimate_eV_per_A"],
        name=f"{name}.error_estimate_eV_per_A",
    )
    if error < 0.0:
        raise ValueError(f"{name}.error_estimate_eV_per_A must be nonnegative.")
    digests = record["displaced_state_sha256"]
    if not isinstance(digests, list) or len(digests) != 4:
        raise ValueError(f"{name}.displaced_state_sha256 must contain four digests.")
    for index, digest in enumerate(digests):
        _sha256_text(digest, name=f"{name}.displaced_state_sha256[{index}]")


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load {name} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain one JSON object.")
    return payload


def _validated_sha(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{name} omits a SHA256 binding.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 mismatch: {actual} != {expected}.")
    return actual


def _copy_with_positions(geometry: Atoms, positions: np.ndarray) -> Atoms:
    result = geometry.copy()
    values = np.asarray(positions, dtype=float)
    if values.shape != geometry.positions.shape or not np.all(np.isfinite(values)):
        raise ValueError("replacement positions must be finite with geometry shape.")
    result.positions = values
    return result


def _proper_rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, triangular = np.linalg.qr(matrix)
    rotation = rotation @ np.diag(np.where(np.diag(triangular) < 0.0, -1.0, 1.0))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-15, rtol=0.0):
        raise RuntimeError("deterministic rotation lost orthogonality.")
    return rotation


@dataclass(slots=True)
class _RecordingSampler:
    """Collect fail-closed root metrics without caching any stencil point."""

    pes: MACE_MDPPolarHybridSmoothHarmonicPES
    target_charge_e: float
    state_records: list[dict[str, object]]

    def __init__(
        self, pes: MACE_MDPPolarHybridSmoothHarmonicPES, *, target_charge_e: float
    ) -> None:
        self.pes = pes
        self.target_charge_e = float(target_charge_e)
        self.state_records = []

    @property
    def provider_id(self) -> str:
        return self.pes.provider_id

    def configuration_sha256(self) -> str:
        return self.pes.configuration_sha256()

    def solve(self, geometry: Atoms):
        state = self.pes.solve(geometry)
        charge_error = abs(
            float(np.sum(np.asarray(state.total_source4)[:, 0])) - self.target_charge_e
        )
        if charge_error > TOTAL_CHARGE_ATOL_E:
            raise RuntimeError("recorded harmonic root violates total charge.")
        self.state_records.append(
            {
                "root_sha256": state.root_sha256,
                "geometry_sha256": state.geometry_sha256,
                "topology_id": state.coefficient_topology_id,
                "energy_eV": state.total_energy_ev,
                "polarization_energy_eV": state.polarization_energy_ev,
                "primal_residual_eV": state.primal_residual_ev,
                "replay_field_max_abs_difference_eV": (
                    state.replay_field_max_abs_difference_ev
                ),
                "replay_energy_abs_difference_eV": (
                    state.replay_energy_abs_difference_ev
                ),
                "total_charge_absolute_error_e": charge_error,
                "cold_iterations": state.cold_iterations,
                "wide_iterations": state.wide_iterations,
            }
        )
        return state

    def sample(self, geometry: Atoms) -> ScalarEnergySample:
        state = self.solve(geometry)
        return ScalarEnergySample(
            energy_eV=state.total_energy_ev,
            state_sha256=state.root_sha256,
            topology_id=state.coefficient_topology_id,
        )

    def summary(self) -> dict[str, object]:
        if not self.state_records:
            raise RuntimeError("root recorder contains no states.")
        return {
            "state_count": len(self.state_records),
            "maximum_primal_residual_eV": max(
                float(record["primal_residual_eV"]) for record in self.state_records
            ),
            "maximum_replay_field_difference_eV": max(
                float(record["replay_field_max_abs_difference_eV"])
                for record in self.state_records
            ),
            "maximum_replay_energy_difference_eV": max(
                float(record["replay_energy_abs_difference_eV"])
                for record in self.state_records
            ),
            "maximum_total_charge_error_e": max(
                float(record["total_charge_absolute_error_e"])
                for record in self.state_records
            ),
            "maximum_cold_iterations": max(
                int(record["cold_iterations"]) for record in self.state_records
            ),
            "maximum_wide_iterations": max(
                int(record["wide_iterations"]) for record in self.state_records
            ),
            "topology_ids": sorted(
                {str(record["topology_id"]) for record in self.state_records}
            ),
        }


def _force_record(evaluation: RichardsonScalarForceEvaluation) -> dict[str, object]:
    forces = np.asarray(evaluation.forces_eV_per_A, dtype=float)
    errors = np.asarray(evaluation.error_estimates_eV_per_A, dtype=float)
    return {
        "evaluation_sha256": evaluation.evaluation_sha256,
        "forces_eV_per_A": forces.tolist(),
        "error_estimates_eV_per_A": errors.tolist(),
        "maximum_error_estimate_eV_per_A": (evaluation.maximum_error_estimate_eV_per_A),
        "net_force_norm_eV_per_A": float(np.linalg.norm(np.sum(forces, axis=0))),
        "finite": bool(np.all(np.isfinite(forces)) and np.all(np.isfinite(errors))),
    }


def _component_record(
    evaluation: RichardsonScalarForceComponentEvaluation,
) -> dict[str, object]:
    return {
        "atom_index": evaluation.atom_index,
        "axis_index": evaluation.axis_index,
        "force_eV_per_A": evaluation.force_eV_per_A,
        "error_estimate_eV_per_A": evaluation.error_estimate_eV_per_A,
        "displaced_state_sha256": list(evaluation.displaced_state_sha256),
    }


def _displaced_component_sample(
    sampler: _RecordingSampler,
    geometry: Atoms,
    *,
    atom: int,
    axis: int,
    delta: float,
) -> ScalarEnergySample:
    displaced = geometry.copy()
    displaced.positions[atom, axis] += float(delta)
    return sampler.sample(displaced)


def _component_convergence_record(
    sampler: _RecordingSampler,
    geometry: Atoms,
    *,
    atom: int,
    axis: int,
    coarse_step: float,
    h4_step: float,
    maximum_local_error: float,
    maximum_h4_difference: float,
) -> dict[str, object]:
    center = sampler.sample(geometry)
    samples: dict[str, ScalarEnergySample] = {}
    for label, delta in (
        ("plus_h", coarse_step),
        ("minus_h", -coarse_step),
        ("plus_h2", 0.5 * coarse_step),
        ("minus_h2", -0.5 * coarse_step),
        ("plus_h4", h4_step),
        ("minus_h4", -h4_step),
    ):
        sample = _displaced_component_sample(
            sampler, geometry, atom=atom, axis=axis, delta=delta
        )
        if sample.topology_id != center.topology_id:
            raise RuntimeError("component convergence stencil changed topology.")
        samples[label] = sample
    derivative_h = (samples["plus_h"].energy_eV - samples["minus_h"].energy_eV) / (
        2.0 * coarse_step
    )
    derivative_h2 = (
        samples["plus_h2"].energy_eV - samples["minus_h2"].energy_eV
    ) / coarse_step
    derivative_h4 = (samples["plus_h4"].energy_eV - samples["minus_h4"].energy_eV) / (
        2.0 * h4_step
    )
    richardson_derivative = (4.0 * derivative_h2 - derivative_h) / 3.0
    local_error = abs(richardson_derivative - derivative_h2)
    h4_difference = abs(richardson_derivative - derivative_h4)
    return {
        "atom_index": atom,
        "axis_index": axis,
        "coarse_step_angstrom": coarse_step,
        "fine_step_angstrom": 0.5 * coarse_step,
        "independent_step_angstrom": h4_step,
        "energy_derivative_h_eV_per_A": derivative_h,
        "energy_derivative_h2_eV_per_A": derivative_h2,
        "energy_derivative_h4_eV_per_A": derivative_h4,
        "richardson_energy_derivative_eV_per_A": richardson_derivative,
        "richardson_force_eV_per_A": -richardson_derivative,
        "local_error_estimate_eV_per_A": local_error,
        "h4_difference_eV_per_A": h4_difference,
        "maximum_local_error_eV_per_A": maximum_local_error,
        "maximum_h4_difference_eV_per_A": maximum_h4_difference,
        "topology_id": center.topology_id,
        "center_state_sha256": center.state_sha256,
        "displaced_state_sha256": {
            label: sample.state_sha256 for label, sample in samples.items()
        },
        "gate_passed": bool(
            local_error <= maximum_local_error
            and h4_difference <= maximum_h4_difference
        ),
    }


def _directional_record(
    sampler: _RecordingSampler,
    geometry: Atoms,
    forces: np.ndarray,
    *,
    seed: int,
    step: float,
    maximum_error: float,
) -> dict[str, object]:
    direction = np.random.default_rng(seed).normal(size=geometry.positions.shape)
    direction /= np.linalg.norm(direction)
    plus = _copy_with_positions(geometry, geometry.positions + step * direction)
    minus = _copy_with_positions(geometry, geometry.positions - step * direction)
    plus_sample = sampler.sample(plus)
    minus_sample = sampler.sample(minus)
    if plus_sample.topology_id != minus_sample.topology_id:
        raise RuntimeError("independent directional stencil changed topology.")
    energy_derivative = (plus_sample.energy_eV - minus_sample.energy_eV) / (2.0 * step)
    force_projection = -float(np.vdot(forces, direction))
    error = abs(energy_derivative - force_projection)
    return {
        "seed": seed,
        "step_angstrom": step,
        "direction": direction.tolist(),
        "energy_derivative_eV_per_A": energy_derivative,
        "negative_force_projection_eV_per_A": force_projection,
        "absolute_error_eV_per_A": error,
        "maximum_error_eV_per_A": maximum_error,
        "plus_state_sha256": plus_sample.state_sha256,
        "minus_state_sha256": minus_sample.state_sha256,
        "gate_passed": bool(error <= maximum_error),
    }


def _closed_loop_record(
    sampler: _RecordingSampler,
    backend: RichardsonScalarForce,
    geometry: Atoms,
    *,
    dofs: tuple[tuple[int, int], tuple[int, int]],
    half_width: float,
    maximum_work: float,
) -> dict[str, object]:
    (atom_x, axis_x), (atom_y, axis_y) = dofs
    edge_contract = (
        ("bottom", 0.0, -half_width, atom_x, axis_x, 2.0 * half_width),
        ("right", half_width, 0.0, atom_y, axis_y, 2.0 * half_width),
        ("top", 0.0, half_width, atom_x, axis_x, -2.0 * half_width),
        ("left", -half_width, 0.0, atom_y, axis_y, -2.0 * half_width),
    )
    edges = []
    work = 0.0
    numerical_error_bound = 0.0
    for label, x, y, atom, axis, displacement in edge_contract:
        midpoint = geometry.copy()
        midpoint.positions[atom_x, axis_x] += x
        midpoint.positions[atom_y, axis_y] += y
        component = backend.evaluate_component(
            sampler, midpoint, atom_index=atom, axis_index=axis
        )
        contribution = component.force_eV_per_A * displacement
        error_bound = component.error_estimate_eV_per_A * abs(displacement)
        work += contribution
        numerical_error_bound += error_bound
        edges.append(
            {
                "label": label,
                "midpoint_offsets_angstrom": [x, y],
                "displacement_angstrom": displacement,
                "work_eV": contribution,
                "work_error_bound_eV": error_bound,
                "force_component": _component_record(component),
            }
        )
    guarded_work = abs(work) + numerical_error_bound
    return {
        "cartesian_dofs": [list(dof) for dof in dofs],
        "half_width_angstrom": half_width,
        "orientation": "counterclockwise",
        "edges": edges,
        "closed_loop_work_eV": work,
        "absolute_work_eV": abs(work),
        "numerical_work_error_bound_eV": numerical_error_bound,
        "guarded_absolute_work_eV": guarded_work,
        "maximum_guarded_absolute_work_eV": maximum_work,
        "gate_passed": bool(guarded_work <= maximum_work),
    }


def _build_pes(
    hybrid: object, radial: object, atoms: Atoms, radii: np.ndarray, contract
):
    return MACE_MDPPolarHybridSmoothHarmonicPES(
        hybrid=hybrid,
        atomic_numbers=atoms.numbers,
        cavity_radii_angstrom=radii,
        dtype=radial.dtype,
        device=radial.device,
        transition_width_angstrom2=float(contract["transition_width_angstrom2"]),
        surface_lmax=int(contract["surface_lmax"]),
        exposure_lmax=int(contract["exposure_lmax"]),
        exposure_radial_quadrature_order=int(
            contract["exposure_radial_quadrature_order"]
        ),
        source_radial_quadrature_order=int(contract["source_radial_quadrature_order"]),
        green_radial_quadrature_order=int(contract["green_radial_quadrature_order"]),
        force_backend=RichardsonScalarForce(
            coarse_step_angstrom=float(contract["force_coarse_step_angstrom"]),
            maximum_error_eV_per_A=float(
                contract["force_maximum_local_error_estimate_ev_per_angstrom"]
            ),
        ),
    )


def _benzene_record(
    *,
    asset_root: Path,
    parent: dict[str, object],
    hybrid: object,
    radial: object,
    runtime_contract: dict[str, object],
    force_panel: dict[str, object],
) -> dict[str, object]:
    records = parent.get("records")
    if not isinstance(records, list):
        raise RuntimeError("parent preregistration omits records.")
    compound_id = str(force_panel["gepol_regression_compound_id"])
    selected = [
        record
        for record in records
        if isinstance(record, dict) and record.get("compound_id") == compound_id
    ]
    if len(selected) != 1:
        raise RuntimeError("benzene regression record is not unique.")
    record = selected[0]
    mol2 = asset_root / str(record["mol2_path"])
    _validated_sha(mol2, record.get("mol2_sha256"), name="benzene MOL2")
    atoms = MOL2Reader(str(mol2), charge=0, mult=1)
    result_path = (
        asset_root
        / PANEL_RELATIVE_ROOT
        / compound_id
        / CUTOFF_DIRECTORY
        / "result.json"
    )
    result = _load_json(result_path, name="benzene frozen projection result")
    inputs = result.get("inputs")
    parsed = inputs.get("parsed_pcm_input") if isinstance(inputs, dict) else None
    if not isinstance(parsed, dict):
        raise RuntimeError("benzene result omits frozen cavity radii.")
    radii = np.asarray(parsed.get("cavity_radii_angstrom"), dtype=float)
    if radii.shape != (len(atoms),) or np.any(radii <= 0.0):
        raise RuntimeError("benzene frozen cavity radii are invalid.")
    pes = _build_pes(hybrid, radial, atoms, radii, runtime_contract)
    sampler = _RecordingSampler(pes, target_charge_e=0.0)
    dof = force_panel["gepol_regression_cartesian_dof"]
    if not isinstance(dof, list) or len(dof) != 2:
        raise RuntimeError("benzene regression Cartesian DOF is invalid.")
    convergence = _component_convergence_record(
        sampler,
        atoms,
        atom=int(dof[0]),
        axis=int(dof[1]),
        coarse_step=float(runtime_contract["force_coarse_step_angstrom"]),
        h4_step=float(runtime_contract["independent_force_step_angstrom"]),
        maximum_local_error=float(
            runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"]
        ),
        maximum_h4_difference=float(
            force_panel["maximum_benzene_h4_difference_ev_per_angstrom"]
        ),
    )
    center = sampler.solve(atoms)
    return {
        "compound_id": compound_id,
        "name": str(record["name"]),
        "atom_count": len(atoms),
        "cavity_radii_angstrom": radii.tolist(),
        "center_energy_eV": center.total_energy_ev,
        "center_polarization_energy_eV": center.polarization_energy_ev,
        "center_root_sha256": center.root_sha256,
        "coefficient_topology_id": center.coefficient_topology_id,
        "force_convergence": convergence,
        "root_summary": sampler.summary(),
        "asset_sha256": {
            "mol2": sha256_file(mol2),
            "projection_result": sha256_file(result_path),
        },
        "gate_passed": bool(convergence["gate_passed"]),
    }


def _water_record(
    *,
    hybrid: object,
    radial: object,
    runtime_contract: dict[str, object],
    force_panel: dict[str, object],
) -> dict[str, object]:
    positions = np.asarray(force_panel["water_geometry_angstrom"], dtype=float)
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("frozen water geometry is invalid.")
    atoms = Atoms(numbers=[8, 1, 1], positions=positions)
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    radii = np.asarray(
        smd_water_coulomb_radii(atoms.get_chemical_symbols()), dtype=float
    )
    pes = _build_pes(hybrid, radial, atoms, radii, runtime_contract)
    sampler = _RecordingSampler(pes, target_charge_e=0.0)
    backend = pes.force_backend
    center = sampler.solve(atoms)
    center_sample = ScalarEnergySample(
        energy_eV=center.total_energy_ev,
        state_sha256=center.root_sha256,
        topology_id=center.coefficient_topology_id,
    )
    center_force = backend.evaluate(sampler, atoms, central_sample=center_sample)
    center_force_record = _force_record(center_force)
    forces = np.asarray(center_force.forces_eV_per_A, dtype=float)
    direction = _directional_record(
        sampler,
        atoms,
        forces,
        seed=int(force_panel["independent_direction_seed"]),
        step=float(runtime_contract["independent_force_step_angstrom"]),
        maximum_error=float(
            force_panel["maximum_independent_directional_error_ev_per_angstrom"]
        ),
    )

    translation = np.asarray(force_panel["translation_angstrom"], dtype=float)
    translated = _copy_with_positions(atoms, atoms.positions + translation)
    translated_state = sampler.solve(translated)
    translated_force = backend.evaluate(
        sampler,
        translated,
        central_sample=ScalarEnergySample(
            energy_eV=translated_state.total_energy_ev,
            state_sha256=translated_state.root_sha256,
            topology_id=translated_state.coefficient_topology_id,
        ),
    )
    translated_forces = np.asarray(translated_force.forces_eV_per_A, dtype=float)
    translation_force_delta = translated_forces - forces
    translation_force_relative = float(
        np.linalg.norm(translation_force_delta)
        / max(np.linalg.norm(forces), np.finfo(float).tiny)
    )
    translation_force_absolute = float(np.max(np.abs(translation_force_delta)))
    translation_energy_error = abs(
        translated_state.total_energy_ev - center.total_energy_ev
    )
    translation_record = {
        "translation_angstrom": translation.tolist(),
        "energy_absolute_error_eV": translation_energy_error,
        "force_relative_error": translation_force_relative,
        "force_maximum_absolute_error_eV_per_A": translation_force_absolute,
        "translated_force": _force_record(translated_force),
        "translated_root_sha256": translated_state.root_sha256,
        "gate_passed": bool(
            translation_energy_error
            <= float(force_panel["maximum_translation_energy_error_ev"])
            and translation_force_relative
            <= float(force_panel["maximum_translation_force_relative_error"])
            and translation_force_absolute
            <= float(
                force_panel["maximum_translation_force_absolute_error_ev_per_angstrom"]
            )
        ),
    }

    rotation = _proper_rotation(int(force_panel["rotation_seed"]))
    centroid = np.mean(atoms.positions, axis=0)
    rotated_positions = (atoms.positions - centroid) @ rotation.T + centroid
    rotated = _copy_with_positions(atoms, rotated_positions)
    rotated_state = sampler.solve(rotated)
    rotated_force = backend.evaluate(
        sampler,
        rotated,
        central_sample=ScalarEnergySample(
            energy_eV=rotated_state.total_energy_ev,
            state_sha256=rotated_state.root_sha256,
            topology_id=rotated_state.coefficient_topology_id,
        ),
    )
    rotated_forces = np.asarray(rotated_force.forces_eV_per_A, dtype=float)
    expected_rotated_forces = forces @ rotation.T
    rotation_force_delta = rotated_forces - expected_rotated_forces
    rotation_force_relative = float(
        np.linalg.norm(rotation_force_delta)
        / max(np.linalg.norm(expected_rotated_forces), np.finfo(float).tiny)
    )
    rotation_force_absolute = float(np.max(np.abs(rotation_force_delta)))
    rotation_energy_error = abs(rotated_state.total_energy_ev - center.total_energy_ev)
    rotation_record = {
        "seed": int(force_panel["rotation_seed"]),
        "rotation_matrix": rotation.tolist(),
        "energy_absolute_error_eV": rotation_energy_error,
        "force_relative_error": rotation_force_relative,
        "force_maximum_absolute_error_eV_per_A": rotation_force_absolute,
        "rotated_force": _force_record(rotated_force),
        "rotated_root_sha256": rotated_state.root_sha256,
        "gate_passed": bool(
            rotation_energy_error
            <= float(force_panel["maximum_rotation_energy_error_ev"])
            and rotation_force_relative
            <= float(force_panel["maximum_rotation_force_relative_error"])
            and rotation_force_absolute
            <= float(
                force_panel["maximum_rotation_force_absolute_error_ev_per_angstrom"]
            )
        ),
    }

    raw_dofs = force_panel["closed_loop_cartesian_dofs"]
    if (
        not isinstance(raw_dofs, list)
        or len(raw_dofs) != 2
        or any(not isinstance(dof, list) or len(dof) != 2 for dof in raw_dofs)
    ):
        raise RuntimeError("closed-loop Cartesian DOFs are invalid.")
    dofs = tuple((int(dof[0]), int(dof[1])) for dof in raw_dofs)
    loop = _closed_loop_record(
        sampler,
        backend,
        atoms,
        dofs=dofs,  # type: ignore[arg-type]
        half_width=float(force_panel["closed_loop_half_width_angstrom"]),
        maximum_work=float(force_panel["maximum_closed_loop_work_abs_ev"]),
    )
    root_summary = sampler.summary()
    root_gate = bool(
        float(root_summary["maximum_primal_residual_eV"]) < ROOT_TOLERANCE_EV
        and float(root_summary["maximum_replay_field_difference_eV"])
        <= ROOT_REPLAY_FIELD_ATOL_EV
        and float(root_summary["maximum_replay_energy_difference_eV"])
        <= ROOT_REPLAY_ENERGY_ATOL_EV
        and float(root_summary["maximum_total_charge_error_e"]) <= TOTAL_CHARGE_ATOL_E
        and len(root_summary["topology_ids"]) == 1
    )
    return {
        "system": "water",
        "geometry_angstrom": positions.tolist(),
        "cavity_radii_angstrom": radii.tolist(),
        "center_energy_eV": center.total_energy_ev,
        "center_polarization_energy_eV": center.polarization_energy_ev,
        "center_root_sha256": center.root_sha256,
        "coefficient_topology_id": center.coefficient_topology_id,
        "center_force": center_force_record,
        "independent_h4_directional_check": direction,
        "translation": translation_record,
        "rotation": rotation_record,
        "closed_loop": loop,
        "root_summary": root_summary,
        "root_gate_passed": root_gate,
        "gate_passed": bool(
            center_force_record["finite"]
            and float(center_force_record["maximum_error_estimate_eV_per_A"])
            <= NUMERICAL_FORCE_MAX_ERROR_EV_PER_ANGSTROM
            and float(center_force_record["net_force_norm_eV_per_A"])
            <= float(force_panel["maximum_net_force_norm_ev_per_angstrom"])
            and direction["gate_passed"]
            and translation_record["gate_passed"]
            and rotation_record["gate_passed"]
            and loop["gate_passed"]
            and root_gate
        ),
    }


def validate_harmonic_ef_science_measurement(
    science: object,
    *,
    runtime_contract: dict[str, object] | None = None,
    force_panel: dict[str, object] | None = None,
) -> None:
    """Validate the exact auditable shape of a harmonic E/F science preimage."""

    runtime_contract = _resolved_runtime_contract(runtime_contract)
    force_panel = _resolved_force_panel(force_panel)

    measurement = _exact_mapping(
        science,
        {
            "benzene_gepol_regression",
            "water_force_symmetry_loop",
            "aggregate",
        },
        name="harmonic E/F science measurement",
    )
    benzene = _exact_mapping(
        measurement["benzene_gepol_regression"],
        {
            "compound_id",
            "name",
            "atom_count",
            "cavity_radii_angstrom",
            "center_energy_eV",
            "center_polarization_energy_eV",
            "center_root_sha256",
            "coefficient_topology_id",
            "force_convergence",
            "root_summary",
            "asset_sha256",
            "gate_passed",
        },
        name="benzene_gepol_regression",
    )
    for key in ("compound_id", "name"):
        if not isinstance(benzene[key], str) or not benzene[key]:
            raise ValueError(f"benzene_gepol_regression.{key} must be non-empty text.")
    benzene_atom_count = _integer(
        benzene["atom_count"], name="benzene_gepol_regression.atom_count", minimum=1
    )
    benzene_radii = _finite_array(
        benzene["cavity_radii_angstrom"],
        (benzene_atom_count,),
        name="benzene_gepol_regression.cavity_radii_angstrom",
    )
    if np.any(benzene_radii <= 0.0):
        raise ValueError("benzene cavity radii must be strictly positive.")
    for key in ("center_energy_eV", "center_polarization_energy_eV"):
        _finite_scalar(benzene[key], name=f"benzene_gepol_regression.{key}")
    for key in ("center_root_sha256", "coefficient_topology_id"):
        _sha256_text(benzene[key], name=f"benzene_gepol_regression.{key}")
    convergence = _exact_mapping(
        benzene["force_convergence"],
        {
            "atom_index",
            "axis_index",
            "coarse_step_angstrom",
            "fine_step_angstrom",
            "independent_step_angstrom",
            "energy_derivative_h_eV_per_A",
            "energy_derivative_h2_eV_per_A",
            "energy_derivative_h4_eV_per_A",
            "richardson_energy_derivative_eV_per_A",
            "richardson_force_eV_per_A",
            "local_error_estimate_eV_per_A",
            "h4_difference_eV_per_A",
            "maximum_local_error_eV_per_A",
            "maximum_h4_difference_eV_per_A",
            "topology_id",
            "center_state_sha256",
            "displaced_state_sha256",
            "gate_passed",
        },
        name="benzene_gepol_regression.force_convergence",
    )
    atom_index = _integer(
        convergence["atom_index"],
        name="benzene_gepol_regression.force_convergence.atom_index",
    )
    axis_index = _integer(
        convergence["axis_index"],
        name="benzene_gepol_regression.force_convergence.axis_index",
    )
    if atom_index >= benzene_atom_count or axis_index >= 3:
        raise ValueError("benzene force-convergence Cartesian index is out of range.")
    for key in (
        "coarse_step_angstrom",
        "fine_step_angstrom",
        "independent_step_angstrom",
        "energy_derivative_h_eV_per_A",
        "energy_derivative_h2_eV_per_A",
        "energy_derivative_h4_eV_per_A",
        "richardson_energy_derivative_eV_per_A",
        "richardson_force_eV_per_A",
        "local_error_estimate_eV_per_A",
        "h4_difference_eV_per_A",
        "maximum_local_error_eV_per_A",
        "maximum_h4_difference_eV_per_A",
    ):
        _finite_scalar(
            convergence[key],
            name=f"benzene_gepol_regression.force_convergence.{key}",
        )
    for key in ("topology_id", "center_state_sha256"):
        _sha256_text(
            convergence[key],
            name=f"benzene_gepol_regression.force_convergence.{key}",
        )
    displaced = _exact_mapping(
        convergence["displaced_state_sha256"],
        {"plus_h", "minus_h", "plus_h2", "minus_h2", "plus_h4", "minus_h4"},
        name="benzene_gepol_regression.force_convergence.displaced_state_sha256",
    )
    for label, digest in displaced.items():
        _sha256_text(
            digest,
            name=(
                "benzene_gepol_regression.force_convergence."
                f"displaced_state_sha256.{label}"
            ),
        )
    _exact_bool(
        convergence["gate_passed"],
        name="benzene_gepol_regression.force_convergence.gate_passed",
    )
    _validate_root_summary(
        benzene["root_summary"], name="benzene_gepol_regression.root_summary"
    )
    assets = _exact_mapping(
        benzene["asset_sha256"],
        {"mol2", "projection_result"},
        name="benzene_gepol_regression.asset_sha256",
    )
    for key, digest in assets.items():
        _sha256_text(digest, name=f"benzene_gepol_regression.asset_sha256.{key}")
    _exact_bool(benzene["gate_passed"], name="benzene_gepol_regression.gate_passed")

    water = _exact_mapping(
        measurement["water_force_symmetry_loop"],
        {
            "system",
            "geometry_angstrom",
            "cavity_radii_angstrom",
            "center_energy_eV",
            "center_polarization_energy_eV",
            "center_root_sha256",
            "coefficient_topology_id",
            "center_force",
            "independent_h4_directional_check",
            "translation",
            "rotation",
            "closed_loop",
            "root_summary",
            "root_gate_passed",
            "gate_passed",
        },
        name="water_force_symmetry_loop",
    )
    if water["system"] != "water":
        raise ValueError("water_force_symmetry_loop.system must be 'water'.")
    _finite_array(water["geometry_angstrom"], (3, 3), name="water.geometry_angstrom")
    water_radii = _finite_array(
        water["cavity_radii_angstrom"],
        (3,),
        name="water.cavity_radii_angstrom",
    )
    if np.any(water_radii <= 0.0):
        raise ValueError("water cavity radii must be strictly positive.")
    for key in ("center_energy_eV", "center_polarization_energy_eV"):
        _finite_scalar(water[key], name=f"water.{key}")
    for key in ("center_root_sha256", "coefficient_topology_id"):
        _sha256_text(water[key], name=f"water.{key}")
    _validate_force_record(
        water["center_force"], atom_count=3, name="water.center_force"
    )

    direction = _exact_mapping(
        water["independent_h4_directional_check"],
        {
            "seed",
            "step_angstrom",
            "direction",
            "energy_derivative_eV_per_A",
            "negative_force_projection_eV_per_A",
            "absolute_error_eV_per_A",
            "maximum_error_eV_per_A",
            "plus_state_sha256",
            "minus_state_sha256",
            "gate_passed",
        },
        name="water.independent_h4_directional_check",
    )
    _integer(direction["seed"], name="water.direction.seed")
    _finite_array(direction["direction"], (3, 3), name="water.direction.direction")
    for key in (
        "step_angstrom",
        "energy_derivative_eV_per_A",
        "negative_force_projection_eV_per_A",
        "absolute_error_eV_per_A",
        "maximum_error_eV_per_A",
    ):
        _finite_scalar(direction[key], name=f"water.direction.{key}")
    for key in ("plus_state_sha256", "minus_state_sha256"):
        _sha256_text(direction[key], name=f"water.direction.{key}")
    _exact_bool(direction["gate_passed"], name="water.direction.gate_passed")

    translation = _exact_mapping(
        water["translation"],
        {
            "translation_angstrom",
            "energy_absolute_error_eV",
            "force_relative_error",
            "force_maximum_absolute_error_eV_per_A",
            "translated_force",
            "translated_root_sha256",
            "gate_passed",
        },
        name="water.translation",
    )
    _finite_array(
        translation["translation_angstrom"],
        (3,),
        name="water.translation.translation_angstrom",
    )
    for key in (
        "energy_absolute_error_eV",
        "force_relative_error",
        "force_maximum_absolute_error_eV_per_A",
    ):
        _finite_scalar(translation[key], name=f"water.translation.{key}")
    _validate_force_record(
        translation["translated_force"], atom_count=3, name="water.translated_force"
    )
    _sha256_text(
        translation["translated_root_sha256"], name="water.translated_root_sha256"
    )
    _exact_bool(translation["gate_passed"], name="water.translation.gate_passed")

    rotation = _exact_mapping(
        water["rotation"],
        {
            "seed",
            "rotation_matrix",
            "energy_absolute_error_eV",
            "force_relative_error",
            "force_maximum_absolute_error_eV_per_A",
            "rotated_force",
            "rotated_root_sha256",
            "gate_passed",
        },
        name="water.rotation",
    )
    _integer(rotation["seed"], name="water.rotation.seed")
    _finite_array(
        rotation["rotation_matrix"], (3, 3), name="water.rotation.rotation_matrix"
    )
    for key in (
        "energy_absolute_error_eV",
        "force_relative_error",
        "force_maximum_absolute_error_eV_per_A",
    ):
        _finite_scalar(rotation[key], name=f"water.rotation.{key}")
    _validate_force_record(
        rotation["rotated_force"], atom_count=3, name="water.rotated_force"
    )
    _sha256_text(rotation["rotated_root_sha256"], name="water.rotated_root_sha256")
    _exact_bool(rotation["gate_passed"], name="water.rotation.gate_passed")

    loop = _exact_mapping(
        water["closed_loop"],
        {
            "cartesian_dofs",
            "half_width_angstrom",
            "orientation",
            "edges",
            "closed_loop_work_eV",
            "absolute_work_eV",
            "numerical_work_error_bound_eV",
            "guarded_absolute_work_eV",
            "maximum_guarded_absolute_work_eV",
            "gate_passed",
        },
        name="water.closed_loop",
    )
    dofs = loop["cartesian_dofs"]
    if (
        not isinstance(dofs, list)
        or len(dofs) != 2
        or any(
            not isinstance(dof, list)
            or len(dof) != 2
            or any(
                isinstance(index, bool) or not isinstance(index, int) for index in dof
            )
            or not (0 <= dof[0] < 3 and 0 <= dof[1] < 3)
            for dof in dofs
        )
    ):
        raise ValueError(
            "water.closed_loop.cartesian_dofs must contain two valid DOFs."
        )
    if loop["orientation"] != "counterclockwise":
        raise ValueError("water.closed_loop.orientation must be counterclockwise.")
    for key in (
        "half_width_angstrom",
        "closed_loop_work_eV",
        "absolute_work_eV",
        "numerical_work_error_bound_eV",
        "guarded_absolute_work_eV",
        "maximum_guarded_absolute_work_eV",
    ):
        _finite_scalar(loop[key], name=f"water.closed_loop.{key}")
    edges = loop["edges"]
    labels = ("bottom", "right", "top", "left")
    if not isinstance(edges, list) or len(edges) != len(labels):
        raise ValueError("water.closed_loop.edges must contain exactly four edges.")
    for index, (edge_value, label) in enumerate(zip(edges, labels, strict=True)):
        edge = _exact_mapping(
            edge_value,
            {
                "label",
                "midpoint_offsets_angstrom",
                "displacement_angstrom",
                "work_eV",
                "work_error_bound_eV",
                "force_component",
            },
            name=f"water.closed_loop.edges[{index}]",
        )
        if edge["label"] != label:
            raise ValueError("water.closed_loop edge labels/order drifted.")
        _finite_array(
            edge["midpoint_offsets_angstrom"],
            (2,),
            name=f"water.closed_loop.edges[{index}].midpoint_offsets_angstrom",
        )
        for key in ("displacement_angstrom", "work_eV", "work_error_bound_eV"):
            _finite_scalar(edge[key], name=f"water.closed_loop.edges[{index}].{key}")
        _validate_component_record(
            edge["force_component"],
            atom_count=3,
            name=f"water.closed_loop.edges[{index}].force_component",
        )
    _exact_bool(loop["gate_passed"], name="water.closed_loop.gate_passed")
    _validate_root_summary(water["root_summary"], name="water.root_summary")
    _exact_bool(water["root_gate_passed"], name="water.root_gate_passed")
    _exact_bool(water["gate_passed"], name="water.gate_passed")

    aggregate = _exact_mapping(
        measurement["aggregate"],
        {
            "maximum_local_richardson_error_eV_per_A",
            "benzene_h4_difference_eV_per_A",
            "water_independent_directional_error_eV_per_A",
            "water_translation_energy_error_eV",
            "water_translation_force_relative_error",
            "water_rotation_energy_error_eV",
            "water_rotation_force_relative_error",
            "water_closed_loop_guarded_absolute_work_eV",
            "all_execution_gates_passed",
        },
        name="aggregate",
    )
    for key in aggregate.keys() - {"all_execution_gates_passed"}:
        _finite_scalar(aggregate[key], name=f"aggregate.{key}")
    _exact_bool(
        aggregate["all_execution_gates_passed"],
        name="aggregate.all_execution_gates_passed",
    )

    runtime_threshold_keys = {
        "force_coarse_step_angstrom",
        "force_fine_step_angstrom",
        "independent_force_step_angstrom",
        "force_maximum_local_error_estimate_ev_per_angstrom",
        "root_tolerance_ev",
        "maximum_root_iterations",
        "total_charge_tolerance_e",
        "multi_start_field_tolerance_ev",
        "multi_start_energy_tolerance_ev",
    }
    for key, expected in HARMONIC_EF_FORCE_PANEL_THRESHOLDS.items():
        contract = runtime_contract if key in runtime_threshold_keys else force_panel
        if key == "maximum_root_iterations":
            if _integer(
                contract[key], name=f"runtime_contract.{key}", minimum=1
            ) != int(expected):
                raise ValueError(
                    f"runtime_contract.{key} drifted from the frozen panel."
                )
        else:
            _require_close(contract[key], expected, name=f"frozen_contract.{key}")

    if benzene["compound_id"] != force_panel["gepol_regression_compound_id"]:
        raise ValueError("benzene compound identity drifted from the frozen panel.")
    expected_benzene_dof = force_panel["gepol_regression_cartesian_dof"]
    if expected_benzene_dof != [atom_index, axis_index]:
        raise ValueError("benzene Cartesian DOF drifted from the frozen panel.")
    coarse_step = _finite_scalar(
        convergence["coarse_step_angstrom"], name="benzene.force.coarse_step"
    )
    fine_step = _finite_scalar(
        convergence["fine_step_angstrom"], name="benzene.force.fine_step"
    )
    independent_step = _finite_scalar(
        convergence["independent_step_angstrom"],
        name="benzene.force.independent_step",
    )
    _require_close(
        coarse_step,
        _finite_scalar(
            runtime_contract["force_coarse_step_angstrom"],
            name="runtime_contract.force_coarse_step_angstrom",
        ),
        name="benzene.force.coarse_step",
    )
    _require_close(fine_step, 0.5 * coarse_step, name="benzene.force.fine_step")
    _require_close(
        independent_step,
        _finite_scalar(
            runtime_contract["independent_force_step_angstrom"],
            name="runtime_contract.independent_force_step_angstrom",
        ),
        name="benzene.force.independent_step",
    )
    derivative_h = _finite_scalar(
        convergence["energy_derivative_h_eV_per_A"], name="benzene.force.derivative_h"
    )
    derivative_h2 = _finite_scalar(
        convergence["energy_derivative_h2_eV_per_A"],
        name="benzene.force.derivative_h2",
    )
    derivative_h4 = _finite_scalar(
        convergence["energy_derivative_h4_eV_per_A"],
        name="benzene.force.derivative_h4",
    )
    richardson = (4.0 * derivative_h2 - derivative_h) / 3.0
    local_error = abs(richardson - derivative_h2)
    h4_difference = abs(richardson - derivative_h4)
    _require_close(
        convergence["richardson_energy_derivative_eV_per_A"],
        richardson,
        name="benzene.force.richardson_derivative",
    )
    _require_close(
        convergence["richardson_force_eV_per_A"],
        -richardson,
        name="benzene.force.richardson_force",
    )
    _require_close(
        convergence["local_error_estimate_eV_per_A"],
        local_error,
        name="benzene.force.local_error",
    )
    _require_close(
        convergence["h4_difference_eV_per_A"],
        h4_difference,
        name="benzene.force.h4_difference",
    )
    local_threshold = _finite_scalar(
        runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"],
        name="runtime_contract.force_maximum_local_error",
    )
    h4_threshold = _finite_scalar(
        force_panel["maximum_benzene_h4_difference_ev_per_angstrom"],
        name="force_panel.maximum_benzene_h4_difference",
    )
    _require_close(
        convergence["maximum_local_error_eV_per_A"],
        local_threshold,
        name="benzene.force.maximum_local_error",
    )
    _require_close(
        convergence["maximum_h4_difference_eV_per_A"],
        h4_threshold,
        name="benzene.force.maximum_h4_difference",
    )
    benzene_gate = local_error <= local_threshold and h4_difference <= h4_threshold
    if convergence["gate_passed"] is not benzene_gate:
        raise ValueError(
            "engine-derived gates: benzene force-convergence gate contradicts detailed values."
        )
    if benzene["gate_passed"] is not benzene_gate:
        raise ValueError(
            "engine-derived gates: benzene gate contradicts force convergence."
        )
    benzene_topology = benzene["coefficient_topology_id"]
    if not (
        convergence["topology_id"]
        == benzene_topology
        == benzene["root_summary"]["topology_ids"][0]
    ):
        raise ValueError("benzene topology bindings disagree.")
    if convergence["center_state_sha256"] != benzene["center_root_sha256"]:
        raise ValueError("benzene center-state digest bindings disagree.")

    expected_geometry = _finite_array(
        force_panel["water_geometry_angstrom"],
        (3, 3),
        name="force_panel.water_geometry_angstrom",
    )
    actual_geometry = np.asarray(water["geometry_angstrom"], dtype=float)
    _require_array_close(actual_geometry, expected_geometry, name="water.geometry")
    water_topology = water["coefficient_topology_id"]
    if water["root_summary"]["topology_ids"][0] != water_topology:
        raise ValueError("water topology bindings disagree.")

    center_forces = np.asarray(water["center_force"]["forces_eV_per_A"], dtype=float)
    direction_values = np.asarray(direction["direction"], dtype=float)
    expected_direction = np.random.default_rng(int(direction["seed"])).normal(
        size=(3, 3)
    )
    expected_direction /= np.linalg.norm(expected_direction)
    if direction["seed"] != force_panel["independent_direction_seed"]:
        raise ValueError("direction seed drifted from the frozen panel.")
    _require_array_close(
        direction_values, expected_direction, name="water.direction.direction"
    )
    _require_close(
        direction["step_angstrom"],
        _finite_scalar(
            runtime_contract["independent_force_step_angstrom"],
            name="runtime_contract.independent_force_step_angstrom",
        ),
        name="water.direction.step_angstrom",
    )
    force_projection = -float(np.vdot(center_forces, direction_values))
    _require_close(
        direction["negative_force_projection_eV_per_A"],
        force_projection,
        name="water.direction.force_projection",
    )
    directional_error = abs(
        _finite_scalar(
            direction["energy_derivative_eV_per_A"],
            name="water.direction.energy_derivative",
        )
        - force_projection
    )
    directional_threshold = _finite_scalar(
        force_panel["maximum_independent_directional_error_ev_per_angstrom"],
        name="force_panel.maximum_directional_error",
    )
    _require_close(
        direction["absolute_error_eV_per_A"],
        directional_error,
        name="water.direction.absolute_error",
    )
    _require_close(
        direction["maximum_error_eV_per_A"],
        directional_threshold,
        name="water.direction.maximum_error",
    )
    direction_gate = directional_error <= directional_threshold
    if direction["gate_passed"] is not direction_gate:
        raise ValueError(
            "engine-derived gates: direction gate contradicts detailed values."
        )

    expected_translation = _finite_array(
        force_panel["translation_angstrom"],
        (3,),
        name="force_panel.translation_angstrom",
    )
    _require_array_close(
        np.asarray(translation["translation_angstrom"], dtype=float),
        expected_translation,
        name="water.translation.translation_angstrom",
    )
    translated_forces = np.asarray(
        translation["translated_force"]["forces_eV_per_A"], dtype=float
    )
    translation_delta = translated_forces - center_forces
    translation_relative = float(
        np.linalg.norm(translation_delta)
        / max(np.linalg.norm(center_forces), np.finfo(float).tiny)
    )
    translation_absolute = float(np.max(np.abs(translation_delta)))
    _require_close(
        translation["force_relative_error"],
        translation_relative,
        name="water.translation.force_relative_error",
    )
    _require_close(
        translation["force_maximum_absolute_error_eV_per_A"],
        translation_absolute,
        name="water.translation.force_maximum_absolute_error",
    )
    translation_energy_error = _finite_scalar(
        translation["energy_absolute_error_eV"],
        name="water.translation.energy_absolute_error",
    )
    if translation_energy_error < 0.0:
        raise ValueError("water.translation.energy_absolute_error must be nonnegative.")
    translation_gate = bool(
        translation_energy_error
        <= _finite_scalar(
            force_panel["maximum_translation_energy_error_ev"],
            name="force_panel.maximum_translation_energy_error",
        )
        and translation_relative
        <= _finite_scalar(
            force_panel["maximum_translation_force_relative_error"],
            name="force_panel.maximum_translation_force_relative_error",
        )
        and translation_absolute
        <= _finite_scalar(
            force_panel["maximum_translation_force_absolute_error_ev_per_angstrom"],
            name="force_panel.maximum_translation_force_absolute_error",
        )
    )
    if translation["gate_passed"] is not translation_gate:
        raise ValueError(
            "engine-derived gates: translation gate contradicts detailed values."
        )

    if rotation["seed"] != force_panel["rotation_seed"]:
        raise ValueError("rotation seed drifted from the frozen panel.")
    expected_rotation = _proper_rotation(int(rotation["seed"]))
    rotation_matrix = np.asarray(rotation["rotation_matrix"], dtype=float)
    _require_array_close(
        rotation_matrix, expected_rotation, name="water.rotation.rotation_matrix"
    )
    rotated_forces = np.asarray(
        rotation["rotated_force"]["forces_eV_per_A"], dtype=float
    )
    expected_rotated_forces = center_forces @ rotation_matrix.T
    rotation_delta = rotated_forces - expected_rotated_forces
    rotation_relative = float(
        np.linalg.norm(rotation_delta)
        / max(np.linalg.norm(expected_rotated_forces), np.finfo(float).tiny)
    )
    rotation_absolute = float(np.max(np.abs(rotation_delta)))
    _require_close(
        rotation["force_relative_error"],
        rotation_relative,
        name="water.rotation.force_relative_error",
    )
    _require_close(
        rotation["force_maximum_absolute_error_eV_per_A"],
        rotation_absolute,
        name="water.rotation.force_maximum_absolute_error",
    )
    rotation_energy_error = _finite_scalar(
        rotation["energy_absolute_error_eV"],
        name="water.rotation.energy_absolute_error",
    )
    if rotation_energy_error < 0.0:
        raise ValueError("water.rotation.energy_absolute_error must be nonnegative.")
    rotation_gate = bool(
        rotation_energy_error
        <= _finite_scalar(
            force_panel["maximum_rotation_energy_error_ev"],
            name="force_panel.maximum_rotation_energy_error",
        )
        and rotation_relative
        <= _finite_scalar(
            force_panel["maximum_rotation_force_relative_error"],
            name="force_panel.maximum_rotation_force_relative_error",
        )
        and rotation_absolute
        <= _finite_scalar(
            force_panel["maximum_rotation_force_absolute_error_ev_per_angstrom"],
            name="force_panel.maximum_rotation_force_absolute_error",
        )
    )
    if rotation["gate_passed"] is not rotation_gate:
        raise ValueError(
            "engine-derived gates: rotation gate contradicts detailed values."
        )

    if loop["cartesian_dofs"] != force_panel["closed_loop_cartesian_dofs"]:
        raise ValueError("closed-loop DOFs drifted from the frozen panel.")
    half_width = _finite_scalar(
        force_panel["closed_loop_half_width_angstrom"],
        name="force_panel.closed_loop_half_width_angstrom",
    )
    _require_close(
        loop["half_width_angstrom"], half_width, name="water.closed_loop.half_width"
    )
    (atom_x, axis_x), (atom_y, axis_y) = loop["cartesian_dofs"]
    expected_edges = (
        ("bottom", [0.0, -half_width], atom_x, axis_x, 2.0 * half_width),
        ("right", [half_width, 0.0], atom_y, axis_y, 2.0 * half_width),
        ("top", [0.0, half_width], atom_x, axis_x, -2.0 * half_width),
        ("left", [-half_width, 0.0], atom_y, axis_y, -2.0 * half_width),
    )
    loop_work = 0.0
    loop_error_bound = 0.0
    for index, (edge, expected) in enumerate(
        zip(loop["edges"], expected_edges, strict=True)
    ):
        label, offsets, atom, axis, displacement = expected
        if edge["label"] != label:
            raise ValueError("closed-loop edge order drifted.")
        _require_array_close(
            np.asarray(edge["midpoint_offsets_angstrom"], dtype=float),
            np.asarray(offsets),
            name=f"water.closed_loop.edges[{index}].midpoint_offsets",
        )
        _require_close(
            edge["displacement_angstrom"],
            displacement,
            name=f"water.closed_loop.edges[{index}].displacement",
        )
        component = edge["force_component"]
        if component["atom_index"] != atom or component["axis_index"] != axis:
            raise ValueError("closed-loop force-component DOF drifted.")
        contribution = (
            _finite_scalar(
                component["force_eV_per_A"], name="closed-loop component force"
            )
            * displacement
        )
        error_bound = _finite_scalar(
            component["error_estimate_eV_per_A"],
            name="closed-loop component error",
        ) * abs(displacement)
        _require_close(
            edge["work_eV"], contribution, name=f"closed-loop edge {label} work"
        )
        _require_close(
            edge["work_error_bound_eV"],
            error_bound,
            name=f"closed-loop edge {label} error bound",
        )
        loop_work += contribution
        loop_error_bound += error_bound
    guarded_work = abs(loop_work) + loop_error_bound
    _require_close(loop["closed_loop_work_eV"], loop_work, name="closed-loop work")
    _require_close(
        loop["absolute_work_eV"], abs(loop_work), name="closed-loop abs work"
    )
    _require_close(
        loop["numerical_work_error_bound_eV"],
        loop_error_bound,
        name="closed-loop numerical bound",
    )
    _require_close(
        loop["guarded_absolute_work_eV"], guarded_work, name="closed-loop guarded work"
    )
    loop_threshold = _finite_scalar(
        force_panel["maximum_closed_loop_work_abs_ev"],
        name="force_panel.maximum_closed_loop_work_abs_ev",
    )
    _require_close(
        loop["maximum_guarded_absolute_work_eV"],
        loop_threshold,
        name="closed-loop maximum guarded work",
    )
    loop_gate = guarded_work <= loop_threshold
    if loop["gate_passed"] is not loop_gate:
        raise ValueError(
            "engine-derived gates: closed-loop gate contradicts detailed values."
        )

    maximum_iterations = _integer(
        runtime_contract["maximum_root_iterations"],
        name="runtime_contract.maximum_root_iterations",
        minimum=1,
    )
    root_summaries = (benzene["root_summary"], water["root_summary"])
    roots_converged = all(
        1 <= summary["maximum_cold_iterations"] <= maximum_iterations
        and 1 <= summary["maximum_wide_iterations"] <= maximum_iterations
        for summary in root_summaries
    )
    root_metrics_pass = all(
        summary["maximum_primal_residual_eV"] < runtime_contract["root_tolerance_ev"]
        and summary["maximum_replay_field_difference_eV"]
        <= runtime_contract["multi_start_field_tolerance_ev"]
        and summary["maximum_replay_energy_difference_eV"]
        <= runtime_contract["multi_start_energy_tolerance_ev"]
        and summary["maximum_total_charge_error_e"]
        <= runtime_contract["total_charge_tolerance_e"]
        for summary in root_summaries
    )
    water_root_gate = bool(root_metrics_pass and roots_converged)
    if water["root_gate_passed"] is not water_root_gate:
        raise ValueError(
            "engine-derived gates: water root gate contradicts root summaries."
        )

    center_force = water["center_force"]
    center_force_gate = bool(
        center_force["maximum_error_estimate_eV_per_A"] <= local_threshold
        and center_force["net_force_norm_eV_per_A"]
        <= force_panel["maximum_net_force_norm_ev_per_angstrom"]
    )
    water_gate = bool(
        center_force_gate
        and direction_gate
        and translation_gate
        and rotation_gate
        and loop_gate
        and water_root_gate
    )
    if water["gate_passed"] is not water_gate:
        raise ValueError(
            "engine-derived gates: water gate contradicts detailed subrecords."
        )

    force_records = (
        water["center_force"],
        translation["translated_force"],
        rotation["rotated_force"],
    )
    maximum_local_error = max(
        local_error,
        *(record["maximum_error_estimate_eV_per_A"] for record in force_records),
        *(edge["force_component"]["error_estimate_eV_per_A"] for edge in loop["edges"]),
    )
    aggregate_expected = {
        "maximum_local_richardson_error_eV_per_A": maximum_local_error,
        "benzene_h4_difference_eV_per_A": h4_difference,
        "water_independent_directional_error_eV_per_A": directional_error,
        "water_translation_energy_error_eV": translation["energy_absolute_error_eV"],
        "water_translation_force_relative_error": translation_relative,
        "water_rotation_energy_error_eV": rotation["energy_absolute_error_eV"],
        "water_rotation_force_relative_error": rotation_relative,
        "water_closed_loop_guarded_absolute_work_eV": guarded_work,
    }
    for key, expected in aggregate_expected.items():
        _require_close(aggregate[key], expected, name=f"aggregate.{key}")
    execution_gate = bool(
        benzene_gate and water_gate and maximum_local_error <= local_threshold
    )
    if aggregate["all_execution_gates_passed"] is not execution_gate:
        raise ValueError(
            "engine-derived gates: aggregate execution gate contradicts detailed records."
        )


def derive_harmonic_ef_replicate_gates(
    measurement: dict[str, object],
    *,
    runtime_contract: dict[str, object],
    force_panel: dict[str, object],
    clean_content_addressed_execution: bool,
) -> dict[str, bool]:
    """Recompute one replicate's gates from raw metrics and frozen thresholds.

    Repository cleanliness is intentionally supplied by the release envelope:
    it cannot be inferred from the protocol-neutral numerical measurement.
    Cross-process equality is an aggregate gate and is therefore excluded.
    """

    if type(clean_content_addressed_execution) is not bool:
        raise TypeError("clean_content_addressed_execution must be exactly bool.")
    runtime_contract = _resolved_runtime_contract(runtime_contract)
    force_panel = _resolved_force_panel(force_panel)
    validate_harmonic_ef_science_measurement(
        measurement,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    benzene = measurement["benzene_gepol_regression"]
    water = measurement["water_force_symmetry_loop"]
    aggregate = measurement["aggregate"]
    if not all(isinstance(value, dict) for value in (benzene, water, aggregate)):
        raise TypeError("harmonic E/F measurement records must be mappings.")
    assert isinstance(benzene, dict)
    assert isinstance(water, dict)
    assert isinstance(aggregate, dict)

    convergence = benzene["force_convergence"]
    center_force = water["center_force"]
    direction = water["independent_h4_directional_check"]
    translation = water["translation"]
    rotation = water["rotation"]
    loop = water["closed_loop"]
    if not all(
        isinstance(value, dict)
        for value in (
            convergence,
            center_force,
            direction,
            translation,
            rotation,
            loop,
        )
    ):
        raise TypeError("harmonic E/F measurement subrecords must be mappings.")
    assert isinstance(convergence, dict)
    assert isinstance(center_force, dict)
    assert isinstance(direction, dict)
    assert isinstance(translation, dict)
    assert isinstance(rotation, dict)
    assert isinstance(loop, dict)

    center_forces = np.asarray(center_force["forces_eV_per_A"], dtype=float)
    center_errors = np.asarray(center_force["error_estimates_eV_per_A"], dtype=float)
    center_finite = bool(
        center_forces.ndim == 2
        and center_forces.shape[-1:] == (3,)
        and center_errors.shape == center_forces.shape
        and np.all(np.isfinite(center_forces))
        and np.all(np.isfinite(center_errors))
    )
    local_threshold = float(
        runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"]
    )
    maximum_iterations = int(runtime_contract["maximum_root_iterations"])
    root_summaries = (
        benzene["root_summary"],
        water["root_summary"],
    )
    if not all(isinstance(summary, dict) for summary in root_summaries):
        raise TypeError("harmonic E/F root summaries must be mappings.")

    roots_converged = all(
        int(summary["state_count"]) > 0
        and 0 <= int(summary["maximum_cold_iterations"]) <= maximum_iterations
        and 0 <= int(summary["maximum_wide_iterations"]) <= maximum_iterations
        for summary in root_summaries
    )
    root_metrics_pass = all(
        float(summary["maximum_primal_residual_eV"])
        < float(runtime_contract["root_tolerance_ev"])
        and float(summary["maximum_replay_field_difference_eV"])
        <= float(runtime_contract["multi_start_field_tolerance_ev"])
        and float(summary["maximum_replay_energy_difference_eV"])
        <= float(runtime_contract["multi_start_energy_tolerance_ev"])
        and float(summary["maximum_total_charge_error_e"])
        <= float(runtime_contract["total_charge_tolerance_e"])
        and len(summary["topology_ids"]) == 1
        for summary in root_summaries
    )
    gates = {
        "clean_content_addressed_execution": clean_content_addressed_execution,
        "benzene_predecessor_failure_coordinate_passes_h_h2_h4": bool(
            float(convergence["local_error_estimate_eV_per_A"]) <= local_threshold
            and float(convergence["h4_difference_eV_per_A"])
            <= float(force_panel["maximum_benzene_h4_difference_ev_per_angstrom"])
        ),
        "water_full_cartesian_h_h2_force_is_finite": center_finite,
        "all_local_richardson_error_estimates_below_2e-4_ev_per_angstrom": bool(
            float(aggregate["maximum_local_richardson_error_eV_per_A"])
            <= local_threshold
        ),
        "independent_h4_directional_derivative_below_5e-4_ev_per_angstrom": bool(
            float(direction["absolute_error_eV_per_A"])
            <= float(
                force_panel["maximum_independent_directional_error_ev_per_angstrom"]
            )
        ),
        "translation_energy_and_net_force_gates_pass": bool(
            float(translation["energy_absolute_error_eV"])
            <= float(force_panel["maximum_translation_energy_error_ev"])
            and float(translation["force_relative_error"])
            <= float(force_panel["maximum_translation_force_relative_error"])
            and float(translation["force_maximum_absolute_error_eV_per_A"])
            <= float(
                force_panel["maximum_translation_force_absolute_error_ev_per_angstrom"]
            )
            and float(center_force["net_force_norm_eV_per_A"])
            <= float(force_panel["maximum_net_force_norm_ev_per_angstrom"])
        ),
        "rotation_energy_and_force_covariance_gates_pass": bool(
            float(rotation["energy_absolute_error_eV"])
            <= float(force_panel["maximum_rotation_energy_error_ev"])
            and float(rotation["force_relative_error"])
            <= float(force_panel["maximum_rotation_force_relative_error"])
            and float(rotation["force_maximum_absolute_error_eV_per_A"])
            <= float(
                force_panel["maximum_rotation_force_absolute_error_ev_per_angstrom"]
            )
        ),
        "closed_loop_work_plus_numerical_error_bound_gate_passes": bool(
            float(loop["guarded_absolute_work_eV"])
            <= float(force_panel["maximum_closed_loop_work_abs_ev"])
        ),
        "all_roots_converge_from_both_starts": bool(roots_converged),
        "all_final_residuals_and_total_charge_errors_pass": bool(root_metrics_pass),
    }
    if tuple(gates) != HARMONIC_EF_REPLICATE_GATE_NAMES:
        raise RuntimeError("harmonic E/F replicate gate set drifted.")
    return gates


def run_harmonic_ef_measurement(
    *,
    asset_root: Path,
    parent: dict[str, object],
    hybrid: object,
    radial: object,
    runtime_contract: dict[str, object],
    force_panel: dict[str, object],
) -> dict[str, object]:
    """Evaluate the frozen deterministic science preimage without an envelope."""

    benzene = _benzene_record(
        asset_root=asset_root,
        parent=parent,
        hybrid=hybrid,
        radial=radial,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    water = _water_record(
        hybrid=hybrid,
        radial=radial,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    maximum_local_error = max(
        float(benzene["force_convergence"]["local_error_estimate_eV_per_A"]),
        float(water["center_force"]["maximum_error_estimate_eV_per_A"]),
        float(
            water["translation"]["translated_force"]["maximum_error_estimate_eV_per_A"]
        ),
        float(water["rotation"]["rotated_force"]["maximum_error_estimate_eV_per_A"]),
        max(
            float(edge["force_component"]["error_estimate_eV_per_A"])
            for edge in water["closed_loop"]["edges"]
        ),
    )
    execution_gate = bool(
        benzene["gate_passed"]
        and water["gate_passed"]
        and maximum_local_error
        <= float(runtime_contract["force_maximum_local_error_estimate_ev_per_angstrom"])
    )
    science = {
        "benzene_gepol_regression": benzene,
        "water_force_symmetry_loop": water,
        "aggregate": {
            "maximum_local_richardson_error_eV_per_A": maximum_local_error,
            "benzene_h4_difference_eV_per_A": float(
                benzene["force_convergence"]["h4_difference_eV_per_A"]
            ),
            "water_independent_directional_error_eV_per_A": float(
                water["independent_h4_directional_check"]["absolute_error_eV_per_A"]
            ),
            "water_translation_energy_error_eV": float(
                water["translation"]["energy_absolute_error_eV"]
            ),
            "water_translation_force_relative_error": float(
                water["translation"]["force_relative_error"]
            ),
            "water_rotation_energy_error_eV": float(
                water["rotation"]["energy_absolute_error_eV"]
            ),
            "water_rotation_force_relative_error": float(
                water["rotation"]["force_relative_error"]
            ),
            "water_closed_loop_guarded_absolute_work_eV": float(
                water["closed_loop"]["guarded_absolute_work_eV"]
            ),
            "all_execution_gates_passed": execution_gate,
        },
    }
    validate_harmonic_ef_science_measurement(
        science,
        runtime_contract=runtime_contract,
        force_panel=force_panel,
    )
    return science
