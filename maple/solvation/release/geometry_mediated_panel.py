"""Preregistered AIMNet2 geometry-mediated PES-panel shard audit.

The frozen Route-2 PES asset contains twenty neutral molecules.  The local
AIMNet2 checkpoint contract exercised by this research lane declares only
H/C/N/O support, so this module explicitly restricts the panel to the first
seventeen compatible molecules and records the three S/Cl controls as excluded.

One shard contains exactly one molecule, all three frozen geometry variants,
all three frozen internal directions, and all three central-difference steps.
The summarizer recomputes every derivative from raw energies and verifies the
geometry, direction, topology, event-margin, reciprocity, replay, and stationary
continuum records.  It is an evidence helper only and never admits a public
energy, force, optimizer, Hessian, dynamics, or mutual-polarization capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256

from .geometry_mediated import (
    GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV,
    GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE,
    summarize_geometry_mediated_directional_audit,
    summarize_geometry_mediated_reciprocity_audit,
)
from .pes_panel import (
    PES_PANEL_DIRECTION_NAMES,
    PES_PANEL_DIRECTIONAL_STEPS_A,
    PES_PANEL_VARIANT_NAMES,
    PESPanelMolecule,
    load_pes_panel,
    panel_directions,
    panel_geometries,
)

AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-shard-contract-v2"
)
AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-shard-summary-v2"
)
AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-shard-artifact-v2"
)
AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-panel-contract-v2"
)
AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-panel-summary-v2"
)
AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_ARTIFACT_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-pes-panel-aggregate-v2"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_CONTRACT_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-shard-contract-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-shard-summary-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_ARTIFACT_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-shard-artifact-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_CONTRACT_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-contract-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-summary-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_ARTIFACT_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-pes-panel-aggregate-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_CONTRACT_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-shard-contract-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-shard-summary-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_ARTIFACT_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-shard-artifact-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_CONTRACT_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-panel-contract-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-panel-summary-v1"
)
AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_ARTIFACT_SCHEMA_VERSION = (
    "route2-aimnet2-frozen-charge-water-harmonic-ddpcm-smdcds-pes-panel-aggregate-v1"
)
AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS = (1, 6, 7, 8)
AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS = (
    "water",
    "methanol",
    "ethanol",
    "acetone",
    "acetonitrile",
    "benzene",
    "methane",
    "trans-butane",
    "dimethyl-ether",
    "formic-acid",
    "acetic-acid",
    "acetaldehyde",
    "acetamide",
    "ethylamine",
    "pyridine",
    "nitromethane",
    "hydrogen-peroxide",
)
AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES = {
    "thiophene": "contains sulfur outside the local H/C/N/O checkpoint contract",
    "methanethiol": "contains sulfur outside the local H/C/N/O checkpoint contract",
    "chloroform": "contains chlorine outside the local H/C/N/O checkpoint contract",
}
AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER = 1.0e12


@dataclass(frozen=True, slots=True)
class AIMNet2GeometryMediatedPESContinuumContract:
    """Immutable shard/panel identities for one continuum diagnostic."""

    continuum_kind: str
    nonpolar_kind: str
    shard_contract_version: str
    shard_summary_schema_version: str
    shard_artifact_schema_version: str
    panel_contract_version: str
    panel_summary_schema_version: str
    panel_artifact_schema_version: str
    shard_artifact_kind: str
    panel_artifact_kind: str


_PES_CONTINUUM_CONTRACTS = {
    ("harmonic-point", "none"): AIMNet2GeometryMediatedPESContinuumContract(
        continuum_kind="harmonic-point",
        nonpolar_kind="none",
        shard_contract_version=AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION,
        shard_summary_schema_version=AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_SCHEMA_VERSION,
        shard_artifact_schema_version=(
            AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION
        ),
        panel_contract_version=AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION,
        panel_summary_schema_version=AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_SCHEMA_VERSION,
        panel_artifact_schema_version=(
            AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_ARTIFACT_SCHEMA_VERSION
        ),
        shard_artifact_kind=(
            "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
            "smooth-harmonic-pes-shard"
        ),
        panel_artifact_kind=(
            "disabled-aimnet2-reconstructed-float64-geometry-mediated-"
            "smooth-harmonic-pes-panel-aggregate"
        ),
    ),
    ("harmonic-ddpcm-water", "none"): AIMNet2GeometryMediatedPESContinuumContract(
        continuum_kind="harmonic-ddpcm-water",
        nonpolar_kind="none",
        shard_contract_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_CONTRACT_VERSION
        ),
        shard_summary_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_SCHEMA_VERSION
        ),
        shard_artifact_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_ARTIFACT_SCHEMA_VERSION
        ),
        panel_contract_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_CONTRACT_VERSION
        ),
        panel_summary_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_SCHEMA_VERSION
        ),
        panel_artifact_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_ARTIFACT_SCHEMA_VERSION
        ),
        shard_artifact_kind=(
            "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
            "smooth-harmonic-ddpcm-pes-shard"
        ),
        panel_artifact_kind=(
            "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
            "smooth-harmonic-ddpcm-pes-panel-aggregate"
        ),
    ),
    (
        "harmonic-ddpcm-water",
        "pyscf-smd-cds-water",
    ): AIMNet2GeometryMediatedPESContinuumContract(
        continuum_kind="harmonic-ddpcm-water",
        nonpolar_kind="pyscf-smd-cds-water",
        shard_contract_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_CONTRACT_VERSION
        ),
        shard_summary_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_SCHEMA_VERSION
        ),
        shard_artifact_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_ARTIFACT_SCHEMA_VERSION
        ),
        panel_contract_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_CONTRACT_VERSION
        ),
        panel_summary_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_SCHEMA_VERSION
        ),
        panel_artifact_schema_version=(
            AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_ARTIFACT_SCHEMA_VERSION
        ),
        shard_artifact_kind=(
            "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
            "smooth-harmonic-ddpcm-pyscf-smdcds-pes-shard"
        ),
        panel_artifact_kind=(
            "disabled-aimnet2-reconstructed-float64-frozen-charge-water-"
            "smooth-harmonic-ddpcm-pyscf-smdcds-pes-panel-aggregate"
        ),
    ),
}


def aimnet2_geometry_mediated_pes_continuum_contract(
    continuum_kind: str,
    *,
    nonpolar_kind: str = "none",
) -> AIMNet2GeometryMediatedPESContinuumContract:
    """Return the exact immutable evidence contract for one total scalar."""

    try:
        return _PES_CONTINUUM_CONTRACTS[(continuum_kind, nonpolar_kind)]
    except KeyError as exc:
        choices = ", ".join(
            f"{continuum}+{nonpolar}"
            for continuum, nonpolar in sorted(_PES_CONTINUUM_CONTRACTS)
        )
        raise ValueError(
            "unsupported geometry-mediated PES scalar "
            f"{continuum_kind!r}+{nonpolar_kind!r}; "
            f"expected one of: {choices}."
        ) from exc


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _eligible_panel() -> tuple[PESPanelMolecule, ...]:
    panel = load_pes_panel()
    identifiers = tuple(molecule.molecule_id for molecule in panel)
    expected_all = AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS + tuple(
        AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES
    )
    if identifiers != expected_all:
        raise RuntimeError(
            "Frozen PES-panel ordering changed from the AIMNet2 shard contract."
        )
    supported = set(AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS)
    for molecule in panel[: len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)]:
        if not set(int(value) for value in molecule.atoms.numbers) <= supported:
            raise RuntimeError(
                f"Eligible molecule {molecule.molecule_id!r} left the H/C/N/O domain."
            )
    for molecule in panel[len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS) :]:
        if set(int(value) for value in molecule.atoms.numbers) <= supported:
            raise RuntimeError(
                f"Excluded molecule {molecule.molecule_id!r} no longer needs exclusion."
            )
    return panel[: len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)]


def aimnet2_geometry_mediated_pes_molecule(
    molecule_index: int,
) -> PESPanelMolecule:
    """Return one exact H/C/N/O molecule selected by stable shard index."""

    if isinstance(molecule_index, bool) or not isinstance(
        molecule_index, (int, np.integer)
    ):
        raise TypeError("molecule_index must be an integer.")
    panel = _eligible_panel()
    index = int(molecule_index)
    if not 0 <= index < len(panel):
        raise ValueError(
            f"molecule_index must be in [0, {len(panel) - 1}] for this contract."
        )
    return panel[index]


def _replay_summary(record: Mapping[str, object]) -> dict[str, object]:
    energy = _finite_float(
        record.get("energy_absolute_error_eV"),
        name="replay energy error",
        nonnegative=True,
    )
    source = _finite_float(
        record.get("source_difference_norm"),
        name="replay source error",
        nonnegative=True,
    )
    gradient = _finite_float(
        record.get("gradient_difference_norm_eV_per_A"),
        name="replay gradient error",
        nonnegative=True,
    )
    gate = (
        energy <= GEOMETRY_MEDIATED_REPLAY_ENERGY_TOLERANCE_EV
        and source <= GEOMETRY_MEDIATED_REPLAY_SOURCE_TOLERANCE
        and gradient <= GEOMETRY_MEDIATED_REPLAY_GRADIENT_TOLERANCE_EV_PER_A
    )
    if record.get("gate_passed") is not gate:
        raise ValueError("deterministic replay gate disagrees with raw errors.")
    return {
        "energy_absolute_error_eV": energy,
        "source_difference_norm": source,
        "gradient_difference_norm_eV_per_A": gradient,
        "gate_passed": gate,
    }


def _summarize_harmonic_point_stationarity(
    record: Mapping[str, object],
) -> dict[str, object]:
    absolute = _finite_float(
        record.get("absolute_residual_eV_per_e"),
        name="stationarity absolute residual",
        nonnegative=True,
    )
    relative = _finite_float(
        record.get("relative_residual"),
        name="stationarity relative residual",
        nonnegative=True,
    )
    condition = _finite_float(
        record.get("surface_condition_number"),
        name="stationarity condition number",
        nonnegative=True,
    )
    dimension = record.get("state_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
        raise TypeError("stationarity state dimension must be an integer.")
    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("stationarity state dimension must be positive.")
    thresholds = _mapping(record.get("thresholds"), name="stationarity thresholds")
    expected_thresholds = {
        "absolute_residual_eV_per_e": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E
        ),
        "relative_residual": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        ),
        "surface_condition_number": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER
        ),
    }
    if set(thresholds) != set(expected_thresholds) or any(
        _finite_float(thresholds[name], name=f"stationarity threshold {name}")
        != expected
        for name, expected in expected_thresholds.items()
    ):
        raise ValueError("stationarity thresholds changed from the shard contract.")
    gate = (
        absolute <= AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E
        and relative <= AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        and condition <= AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER
    )
    if record.get("gate_passed") is not gate:
        raise ValueError("stationarity gate disagrees with raw measurements.")
    if record.get("capability_admitted") is not False:
        raise ValueError("stationarity diagnostic must not admit a capability.")
    return {
        "absolute_residual_eV_per_e": absolute,
        "relative_residual": relative,
        "surface_condition_number": condition,
        "state_dimension": dimension,
        "gate_passed": gate,
        "capability_admitted": False,
    }


_DDPCM_RESIDUAL_UNITS = {
    "vacuum_projection_primal": "eV/e",
    "dielectric_primal": "eV/e",
    "single_layer_primal": "eV/e",
    "single_layer_adjoint": "eV/e",
    "dielectric_adjoint": "e",
    "vacuum_projection_adjoint": "e",
}
_DDPCM_CONDITION_NAMES = ("mass", "surface", "dielectric")
_DDPCM_RESPONSE_NONNEGATIVE_FIELDS = (
    "primal_response_relative_asymmetry",
    "primal_charge_tangent_relative_asymmetry",
    "energy_cotangent_relative_asymmetry",
    "energy_cotangent_charge_tangent_relative_asymmetry",
    "energy_cotangent_vs_symmetric_primal_relative_error",
    "kkt_vs_autograd_absolute_source_covector_norm",
    "kkt_vs_autograd_relative_error",
    "primal_vs_energy_cotangent_absolute_source_covector_norm",
    "primal_vs_energy_cotangent_relative_error",
    "half_coupling_absolute_error_eV",
    "threshold",
)
_DDPCM_RESPONSE_FIELDS = frozenset(
    {
        "pairing_metric_id",
        "operator_representation",
        "primal_response_used_as_provider_field",
        "primal_response_is_energy_cotangent",
        *_DDPCM_RESPONSE_NONNEGATIVE_FIELDS,
        "scalar_energy_eV",
        "half_energy_cotangent_pairing_eV",
        "gate_passed",
    }
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _summarize_ddpcm_residual(
    record: Mapping[str, object],
    *,
    name: str,
    expected_unit: str,
) -> dict[str, object]:
    expected_fields = {
        "absolute",
        "right_hand_side_norm",
        "relative",
        "scaled",
        "unit",
    }
    if set(record) != expected_fields:
        raise ValueError(f"ddPCM {name} residual schema changed.")
    absolute = _finite_float(
        record.get("absolute"), name=f"ddPCM {name} absolute residual", nonnegative=True
    )
    rhs_norm = _finite_float(
        record.get("right_hand_side_norm"),
        name=f"ddPCM {name} right-hand-side norm",
        nonnegative=True,
    )
    relative = _finite_float(
        record.get("relative"),
        name=f"ddPCM {name} relative residual",
        nonnegative=True,
    )
    scaled = _finite_float(
        record.get("scaled"),
        name=f"ddPCM {name} scaled residual",
        nonnegative=True,
    )
    expected_relative = absolute / rhs_norm if rhs_norm > np.finfo(float).tiny else 0.0
    expected_scaled = absolute / max(rhs_norm, 1.0)
    if not math.isclose(
        relative, expected_relative, rel_tol=2.0e-15, abs_tol=2.0e-15
    ) or not math.isclose(scaled, expected_scaled, rel_tol=2.0e-15, abs_tol=2.0e-15):
        raise ValueError(f"ddPCM {name} normalized residual is not reproducible.")
    if record.get("unit") != expected_unit:
        raise ValueError(f"ddPCM {name} residual unit changed.")
    return {
        "absolute": absolute,
        "right_hand_side_norm": rhs_norm,
        "relative": relative,
        "scaled": scaled,
        "unit": expected_unit,
    }


def _summarize_ddpcm_response_audit(
    record: Mapping[str, object], *, threshold: float
) -> dict[str, object]:
    if set(record) != _DDPCM_RESPONSE_FIELDS:
        raise ValueError("ddPCM energy-cotangent response-audit schema changed.")
    pairing_metric_id = record.get("pairing_metric_id")
    representation = record.get("operator_representation")
    if not isinstance(pairing_metric_id, str) or not pairing_metric_id:
        raise ValueError("ddPCM response audit requires a pairing-metric identity.")
    if not isinstance(representation, str) or not representation:
        raise ValueError("ddPCM response audit requires an operator representation.")
    if record.get("primal_response_used_as_provider_field") is not False:
        raise ValueError("ddPCM primal response must not be the provider field.")
    primal_is_cotangent = record.get("primal_response_is_energy_cotangent")
    if not isinstance(primal_is_cotangent, bool):
        raise TypeError("ddPCM primal/cotangent identity flag must be boolean.")
    numeric = {
        name: _finite_float(
            record.get(name), name=f"ddPCM response metric {name}", nonnegative=True
        )
        for name in _DDPCM_RESPONSE_NONNEGATIVE_FIELDS
    }
    scalar_energy = _finite_float(
        record.get("scalar_energy_eV"), name="ddPCM scalar energy"
    )
    half_energy = _finite_float(
        record.get("half_energy_cotangent_pairing_eV"),
        name="ddPCM half energy-cotangent pairing",
    )
    if numeric["threshold"] != threshold:
        raise ValueError("ddPCM response threshold changed from the shard contract.")
    expected_half_error = abs(scalar_energy - half_energy)
    if not math.isclose(
        numeric["half_coupling_absolute_error_eV"],
        expected_half_error,
        rel_tol=2.0e-15,
        abs_tol=2.0e-15,
    ):
        raise ValueError("ddPCM half-coupling error is not reproducible.")
    gate = (
        numeric["energy_cotangent_relative_asymmetry"] <= threshold
        and numeric["energy_cotangent_charge_tangent_relative_asymmetry"] <= threshold
        and numeric["energy_cotangent_vs_symmetric_primal_relative_error"] <= threshold
        and numeric["kkt_vs_autograd_relative_error"] <= threshold
        and expected_half_error
        <= threshold * max(abs(scalar_energy), abs(half_energy), 1.0)
    )
    if record.get("gate_passed") is not gate:
        raise ValueError("ddPCM energy-cotangent gate disagrees with raw metrics.")
    return {
        "pairing_metric_id": pairing_metric_id,
        "operator_representation": representation,
        "primal_response_used_as_provider_field": False,
        "primal_response_is_energy_cotangent": primal_is_cotangent,
        **numeric,
        "scalar_energy_eV": scalar_energy,
        "half_energy_cotangent_pairing_eV": half_energy,
        "gate_passed": gate,
    }


def _summarize_harmonic_ddpcm_stationarity(
    record: Mapping[str, object],
) -> dict[str, object]:
    if record.get("finite_dielectric_parameterization") is not True:
        raise ValueError("ddPCM stationarity requires finite dielectric equations.")
    dielectric = _finite_float(record.get("dielectric"), name="ddPCM dielectric")
    if dielectric <= 1.0:
        raise ValueError("ddPCM dielectric must be greater than one.")
    dimension = _positive_integer(
        record.get("state_dimension_per_block"),
        name="ddPCM state dimension per block",
    )
    primal_blocks = _positive_integer(
        record.get("primal_block_count"), name="ddPCM primal block count"
    )
    adjoint_blocks = _positive_integer(
        record.get("adjoint_block_count"), name="ddPCM adjoint block count"
    )
    if primal_blocks != 3 or adjoint_blocks != 3:
        raise ValueError("ddPCM stationarity requires three primal and adjoint blocks.")

    thresholds = _mapping(record.get("thresholds"), name="ddPCM thresholds")
    expected_thresholds = {
        "absolute_residual": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E
        ),
        "relative_residual": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        ),
        "max_rhs_or_one_scaled_residual": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        ),
        "condition_number": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER
        ),
        "energy_cotangent_closure": (
            AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE
        ),
    }
    if set(thresholds) != set(expected_thresholds):
        raise ValueError("ddPCM stationarity threshold schema changed.")
    normalized_thresholds = {
        name: _finite_float(
            thresholds[name], name=f"ddPCM stationarity threshold {name}"
        )
        for name in expected_thresholds
    }
    if normalized_thresholds != expected_thresholds:
        raise ValueError("ddPCM stationarity thresholds changed from the contract.")

    raw_residuals = _mapping(record.get("residuals"), name="ddPCM residuals")
    if set(raw_residuals) != set(_DDPCM_RESIDUAL_UNITS):
        raise ValueError("ddPCM primal/adjoint residual coverage changed.")
    residuals = {
        name: _summarize_ddpcm_residual(
            _mapping(raw_residuals[name], name=f"ddPCM {name} residual"),
            name=name,
            expected_unit=unit,
        )
        for name, unit in _DDPCM_RESIDUAL_UNITS.items()
    }

    raw_conditions = _mapping(
        record.get("condition_numbers"), name="ddPCM condition numbers"
    )
    if set(raw_conditions) != set(_DDPCM_CONDITION_NAMES):
        raise ValueError("ddPCM condition-number coverage changed.")
    conditions = {
        name: _finite_float(
            raw_conditions[name],
            name=f"ddPCM {name} condition number",
            nonnegative=True,
        )
        for name in _DDPCM_CONDITION_NAMES
    }
    response = _summarize_ddpcm_response_audit(
        _mapping(
            record.get("response_operator_audit"),
            name="ddPCM response-operator audit",
        ),
        threshold=normalized_thresholds["energy_cotangent_closure"],
    )
    residual_gate = all(
        residual["absolute"] <= normalized_thresholds["absolute_residual"]
        and residual["relative"] <= normalized_thresholds["relative_residual"]
        and residual["scaled"]
        <= normalized_thresholds["max_rhs_or_one_scaled_residual"]
        for residual in residuals.values()
    )
    condition_gate = all(
        value <= normalized_thresholds["condition_number"]
        for value in conditions.values()
    )
    gate = response["gate_passed"] is True and residual_gate and condition_gate
    if record.get("gate_passed") is not gate:
        raise ValueError("ddPCM stationarity gate disagrees with raw measurements.")
    if record.get("capability_admitted") is not False:
        raise ValueError("ddPCM stationarity diagnostic must not admit capability.")
    return {
        "stationarity_kind": "harmonic-ddpcm-primal-adjoint-kkt",
        "dielectric": dielectric,
        "state_dimension": dimension,
        "state_dimension_per_block": dimension,
        "primal_block_count": primal_blocks,
        "adjoint_block_count": adjoint_blocks,
        "maximum_absolute_residual": max(
            residual["absolute"] for residual in residuals.values()
        ),
        "maximum_relative_residual": max(
            residual["relative"] for residual in residuals.values()
        ),
        "maximum_scaled_residual": max(
            residual["scaled"] for residual in residuals.values()
        ),
        "surface_condition_number": conditions["surface"],
        "maximum_condition_number": max(conditions.values()),
        "residuals": residuals,
        "condition_numbers": conditions,
        "response_operator_audit": response,
        "gate_passed": gate,
        "capability_admitted": False,
    }


def summarize_aimnet2_geometry_mediated_stationarity(
    record: Mapping[str, object],
    *,
    continuum_kind: str = "harmonic-point",
) -> dict[str, object]:
    """Recompute one continuum's stationary/KKT audit from raw measurements."""

    aimnet2_geometry_mediated_pes_continuum_contract(continuum_kind)
    if continuum_kind == "harmonic-point":
        return _summarize_harmonic_point_stationarity(record)
    return _summarize_harmonic_ddpcm_stationarity(record)


def _geometry_record(
    raw: Mapping[str, object],
    *,
    molecule: PESPanelMolecule,
    variant: str,
    continuum_kind: str,
    nonpolar_kind: str,
) -> dict[str, object]:
    geometries = panel_geometries(molecule)
    expected_atoms = geometries[variant]
    if (
        raw.get("molecule_id") != molecule.molecule_id
        or raw.get("source_record_id") != molecule.source_record_id
        or raw.get("chemical_formula") != molecule.chemical_formula
        or tuple(raw.get("scope_tags", ())) != molecule.scope_tags
        or raw.get("variant") != variant
        or raw.get("geometry_sha256") != geometry_sha256(expected_atoms)
    ):
        raise ValueError("PES shard geometry identity changed from the frozen asset.")
    numbers = np.asarray(raw.get("atomic_numbers"))
    positions = np.asarray(raw.get("positions_A"), dtype=float)
    if not np.array_equal(numbers, expected_atoms.numbers) or not np.array_equal(
        positions, expected_atoms.positions
    ):
        raise ValueError("PES shard atomic numbers or positions changed.")
    if raw.get("charge") != 0 or raw.get("multiplicity") != 1:
        raise ValueError("AIMNet2 PES shards are restricted to neutral singlets.")

    center = _mapping(raw.get("center"), name="geometry center")
    energy = _finite_float(center.get("total_energy_eV"), name="center energy")
    vacuum_energy = _finite_float(
        center.get("vacuum_energy_eV"), name="center vacuum energy"
    )
    continuum_energy = _finite_float(
        center.get("continuum_energy_eV"), name="center continuum energy"
    )
    nonpolar_energy = 0.0
    if nonpolar_kind == "pyscf-smd-cds-water":
        nonpolar_energy = _finite_float(
            center.get("nonpolar_energy_eV"), name="center nonpolar energy"
        )
    if not math.isclose(
        energy,
        vacuum_energy + continuum_energy + nonpolar_energy,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("center energy ledger does not close.")
    gradient = np.asarray(center.get("total_gradient_eV_per_A"), dtype=float)
    if gradient.shape != (len(expected_atoms), 3) or not np.all(np.isfinite(gradient)):
        raise ValueError("center gradient must be finite with shape (N,3).")
    forces = np.asarray(center.get("forces_eV_per_A"), dtype=float)
    source = np.asarray(center.get("source"), dtype=float)
    reaction_field = np.asarray(center.get("reaction_field"), dtype=float)
    if (
        forces.shape != gradient.shape
        or source.shape != (len(expected_atoms), 4)
        or reaction_field.shape != source.shape
        or not np.all(np.isfinite(forces))
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(reaction_field))
    ):
        raise ValueError("center force/source/reaction-field arrays are invalid.")
    if not np.array_equal(forces, -gradient):
        raise ValueError("center force is not the exact negative scalar gradient.")
    nonpolar_gradient = None
    if nonpolar_kind == "pyscf-smd-cds-water":
        nonpolar_gradient = np.asarray(
            center.get("nonpolar_gradient_eV_per_A"), dtype=float
        )
        components = _mapping(
            center.get("gradient_components_eV_per_A"),
            name="center gradient components",
        )
        expected_component_names = {
            "intrinsic",
            "continuum_fixed_source",
            "source_response",
            "electrostatic_total",
            "nonpolar",
        }
        if set(components) != expected_component_names:
            raise ValueError("center total-SMD gradient component ledger changed.")
        component_arrays = {
            name: np.asarray(components[name], dtype=float)
            for name in expected_component_names
        }
        if any(
            values.shape != gradient.shape or not np.all(np.isfinite(values))
            for values in component_arrays.values()
        ) or (
            nonpolar_gradient.shape != gradient.shape
            or not np.all(np.isfinite(nonpolar_gradient))
        ):
            raise ValueError(
                "center total-SMD gradient components must be finite with shape (N,3)."
            )
        electrostatic_expected = (
            component_arrays["intrinsic"]
            + component_arrays["continuum_fixed_source"]
            + component_arrays["source_response"]
        )
        if (
            not np.allclose(
                component_arrays["electrostatic_total"],
                electrostatic_expected,
                rtol=0.0,
                atol=2.0e-10,
            )
            or not np.array_equal(component_arrays["nonpolar"], nonpolar_gradient)
            or not np.allclose(
                gradient,
                component_arrays["electrostatic_total"] + nonpolar_gradient,
                rtol=0.0,
                atol=2.0e-10,
            )
        ):
            raise ValueError("center total-SMD gradient ledger does not close.")
    if not np.array_equal(
        source[:, 1:], np.zeros_like(source[:, 1:])
    ) or not math.isclose(
        float(np.sum(source[:, 0])), 0.0, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("center point-l0 source violates the neutral-charge contract.")
    model_topology = _mapping(
        raw.get("center_model_topology"), name="center model topology"
    )
    continuum_topology = _mapping(
        raw.get("center_continuum_topology"), name="center continuum topology"
    )
    reciprocity = _mapping(
        raw.get("reciprocity_metric_charge_gauge"), name="reciprocity audit"
    )
    reciprocity_summary = summarize_geometry_mediated_reciprocity_audit(
        reciprocity,
        reaction_field=reaction_field,
    )
    reciprocity_gate = reciprocity_summary["gate_passed"] is True
    replay = _replay_summary(
        _mapping(raw.get("deterministic_replay"), name="deterministic replay")
    )
    stationarity = summarize_aimnet2_geometry_mediated_stationarity(
        _mapping(raw.get("stationarity"), name="stationarity audit"),
        continuum_kind=continuum_kind,
    )
    if stationarity["state_dimension"] != 4 * len(expected_atoms):
        raise ValueError("stationarity dimension changed from surface_lmax=1.")
    if continuum_topology.get("coefficient_count") != stationarity["state_dimension"]:
        raise ValueError("continuum topology and stationarity dimensions disagree.")

    raw_directions = _mapping(raw.get("directions"), name="direction records")
    if len(raw_directions) != len(PES_PANEL_DIRECTION_NAMES) or set(
        raw_directions
    ) != set(PES_PANEL_DIRECTION_NAMES):
        raise ValueError("direction record coverage changed from the frozen contract.")
    expected_directions = panel_directions(expected_atoms, molecule.molecule_id)
    audits: dict[str, object] = {}
    neighbor_margins = [
        _finite_float(
            model_topology.get("minimum_cutoff_margin_angstrom"),
            name="center neighbor cutoff margin",
            nonnegative=True,
        )
    ]
    continuum_margins = [
        _finite_float(
            continuum_topology.get("minimum_point_source_shell_margin_angstrom"),
            name="center continuum event margin",
            nonnegative=True,
        )
    ]
    sphere_tangency_margins = [
        _finite_float(
            continuum_topology.get("minimum_sphere_tangency_margin_angstrom"),
            name="center sphere-tangency margin",
            nonnegative=True,
        )
    ]
    for direction_name in PES_PANEL_DIRECTION_NAMES:
        direction_record = _mapping(
            raw_directions[direction_name], name=f"{direction_name} record"
        )
        direction = np.asarray(direction_record.get("direction"), dtype=float)
        if not np.array_equal(direction, expected_directions[direction_name]):
            raise ValueError(
                f"{direction_name} changed from the frozen panel direction."
            )
        samples = direction_record.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("directional samples must be a sequence of mappings.")
        normalized_samples = tuple(
            _mapping(sample, name=f"{direction_name} sample") for sample in samples
        )
        for sample in normalized_samples:
            for sign in ("plus", "minus"):
                sampled_model = _mapping(
                    sample.get(f"{sign}_model_topology"),
                    name=f"{direction_name} {sign} model topology",
                )
                sampled_continuum = _mapping(
                    sample.get(f"{sign}_continuum_topology"),
                    name=f"{direction_name} {sign} continuum topology",
                )
                neighbor_margins.append(
                    _finite_float(
                        sampled_model.get("minimum_cutoff_margin_angstrom"),
                        name="sampled neighbor cutoff margin",
                        nonnegative=True,
                    )
                )
                continuum_margins.append(
                    _finite_float(
                        sampled_continuum.get(
                            "minimum_point_source_shell_margin_angstrom"
                        ),
                        name="sampled continuum event margin",
                        nonnegative=True,
                    )
                )
                sphere_tangency_margins.append(
                    _finite_float(
                        sampled_continuum.get(
                            "minimum_sphere_tangency_margin_angstrom"
                        ),
                        name="sampled sphere-tangency margin",
                        nonnegative=True,
                    )
                )
        audits[direction_name] = summarize_geometry_mediated_directional_audit(
            analytic_gradient_eV_per_A=gradient,
            direction=direction,
            center_model_topology=model_topology,
            center_continuum_topology=continuum_topology,
            samples=normalized_samples,
            reciprocity_audit=reciprocity,
            expected_steps_A=PES_PANEL_DIRECTIONAL_STEPS_A,
        )

    directional_gate = all(
        bool(_mapping(audits[name], name="directional audit").get("gate_passed"))
        for name in PES_PANEL_DIRECTION_NAMES
    )
    gate = (
        replay["gate_passed"] is True
        and stationarity["gate_passed"] is True
        and reciprocity_gate
        and directional_gate
    )
    result = {
        "molecule_id": molecule.molecule_id,
        "variant": variant,
        "geometry_sha256": geometry_sha256(expected_atoms),
        "vacuum_energy_eV": vacuum_energy,
        "continuum_energy_eV": continuum_energy,
        "total_energy_eV": energy,
        "deterministic_replay": replay,
        "stationarity": stationarity,
        "reciprocity_metric_charge_gauge": reciprocity_summary,
        "reciprocity_metric_charge_gauge_gate_passed": reciprocity_gate,
        "directional_force_fd": audits,
        "minimum_neighbor_cutoff_margin_A": min(neighbor_margins),
        "minimum_continuum_event_margin_A": min(continuum_margins),
        "minimum_sphere_tangency_margin_A": min(sphere_tangency_margins),
        "gate_passed": gate,
    }
    if nonpolar_kind == "pyscf-smd-cds-water":
        result.update(
            {
                "nonpolar_kind": nonpolar_kind,
                "nonpolar_energy_eV": nonpolar_energy,
                "nonpolar_gradient_eV_per_A": nonpolar_gradient.tolist(),
            }
        )
    return result


def summarize_aimnet2_geometry_mediated_pes_shard(
    *,
    molecule_index: int,
    records: Sequence[Mapping[str, object]],
    continuum_kind: str = "harmonic-point",
    nonpolar_kind: str = "none",
) -> dict[str, object]:
    """Validate and summarize one exact H/C/N/O three-geometry shard."""

    contract = aimnet2_geometry_mediated_pes_continuum_contract(
        continuum_kind, nonpolar_kind=nonpolar_kind
    )
    molecule = aimnet2_geometry_mediated_pes_molecule(molecule_index)
    values = tuple(records)
    if len(values) != len(PES_PANEL_VARIANT_NAMES):
        raise ValueError(
            "AIMNet2 geometry-mediated PES shard requires exactly three variants."
        )
    by_variant = {str(record.get("variant")): record for record in values}
    if len(by_variant) != len(values) or tuple(by_variant) != PES_PANEL_VARIANT_NAMES:
        raise ValueError("PES shard variant coverage or order is invalid.")
    summaries = [
        _geometry_record(
            by_variant[variant],
            molecule=molecule,
            variant=variant,
            continuum_kind=continuum_kind,
            nonpolar_kind=nonpolar_kind,
        )
        for variant in PES_PANEL_VARIANT_NAMES
    ]

    directional = [
        audit
        for summary in summaries
        for audit in _mapping(
            summary["directional_force_fd"], name="directional force summary"
        ).values()
    ]
    topology_gates = [
        _mapping(audit, name="directional audit")["topology"] for audit in directional
    ]
    stationarity_summaries = [
        _mapping(summary["stationarity"], name="stationarity") for summary in summaries
    ]
    maximum_condition = max(
        float(stationarity["surface_condition_number"])
        for stationarity in stationarity_summaries
    )
    reciprocity_summaries = [
        _mapping(
            summary["reciprocity_metric_charge_gauge"],
            name="reciprocity summary",
        )
        for summary in summaries
    ]
    maximum_directional_error = max(
        float(record["absolute_error_eV_per_A"])
        for audit in directional
        for record in _mapping(audit, name="directional audit")["records"]
    )
    minimum_neighbor_margin = min(
        float(summary["minimum_neighbor_cutoff_margin_A"]) for summary in summaries
    )
    minimum_continuum_margin = min(
        float(summary["minimum_continuum_event_margin_A"]) for summary in summaries
    )
    minimum_sphere_tangency_margin = min(
        float(summary["minimum_sphere_tangency_margin_A"]) for summary in summaries
    )
    gates = {
        "all_deterministic_replays": all(
            _mapping(summary["deterministic_replay"], name="replay").get("gate_passed")
            is True
            for summary in summaries
        ),
        "all_stationarity_audits": all(
            _mapping(summary["stationarity"], name="stationarity").get("gate_passed")
            is True
            for summary in summaries
        ),
        "all_reciprocity_metric_charge_gauge_audits": all(
            summary["reciprocity_metric_charge_gauge_gate_passed"] is True
            for summary in summaries
        ),
        "all_directional_force_fd": all(
            _mapping(audit, name="directional audit").get("gate_passed") is True
            for audit in directional
        ),
        "all_stencils_same_stratum": all(
            _mapping(topology, name="topology").get("all_stencils_same_stratum") is True
            for topology in topology_gates
        ),
        "all_neighbor_cutoff_guards": all(
            _mapping(topology, name="topology").get("all_neighbor_cutoff_guards_passed")
            is True
            for topology in topology_gates
        ),
        "all_continuum_event_guards": all(
            _mapping(topology, name="topology").get("all_continuum_event_guards_passed")
            is True
            for topology in topology_gates
        ),
        "all_sphere_tangency_guards": all(
            _mapping(topology, name="topology").get("all_sphere_tangency_guards_passed")
            is True
            for topology in topology_gates
        ),
    }
    if continuum_kind == "harmonic-ddpcm-water":
        gates.update(
            {
                "all_energy_cotangent_kkt_audits": all(
                    _mapping(
                        stationarity["response_operator_audit"],
                        name="ddPCM response audit",
                    ).get("gate_passed")
                    is True
                    for stationarity in stationarity_summaries
                ),
                "all_primal_responses_excluded_from_provider_field": all(
                    _mapping(
                        stationarity["response_operator_audit"],
                        name="ddPCM response audit",
                    ).get("primal_response_used_as_provider_field")
                    is False
                    for stationarity in stationarity_summaries
                ),
            }
        )
    result = {
        "schema_version": contract.shard_summary_schema_version,
        "contract_version": contract.shard_contract_version,
        "molecule_index": int(molecule_index),
        "molecule_id": molecule.molecule_id,
        "source_record_id": molecule.source_record_id,
        "chemical_formula": molecule.chemical_formula,
        "supported_atomic_numbers": list(
            AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS
        ),
        "excluded_molecules": dict(AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES),
        "variant_count": len(summaries),
        "directional_record_count": len(directional),
        "directional_sample_count": len(directional)
        * len(PES_PANEL_DIRECTIONAL_STEPS_A),
        "maximum_surface_condition_number": maximum_condition,
        "maximum_reciprocity_absolute_error_eV": max(
            float(summary["maximum_reciprocity_absolute_error_eV"])
            for summary in reciprocity_summaries
        ),
        "maximum_reciprocity_relative_error": max(
            float(summary["maximum_reciprocity_relative_error"])
            for summary in reciprocity_summaries
        ),
        "maximum_apply_adjoint_absolute_error_eV": max(
            float(summary["maximum_apply_adjoint_absolute_error_eV"])
            for summary in reciprocity_summaries
        ),
        "maximum_apply_adjoint_relative_error": max(
            float(summary["maximum_apply_adjoint_relative_error"])
            for summary in reciprocity_summaries
        ),
        "maximum_charge_fd_absolute_error_eV_per_e": max(
            float(summary["maximum_charge_fd_absolute_error_eV_per_e"])
            for summary in reciprocity_summaries
        ),
        "maximum_charge_fd_relative_error": max(
            float(summary["maximum_charge_fd_relative_error"])
            for summary in reciprocity_summaries
        ),
        "maximum_charge_gauge_vjp_norm_eV_per_A": max(
            float(summary["charge_gauge_vjp_norm_eV_per_A"])
            for summary in reciprocity_summaries
        ),
        "maximum_directional_absolute_error_eV_per_A": maximum_directional_error,
        "minimum_neighbor_cutoff_margin_A": minimum_neighbor_margin,
        "minimum_continuum_event_margin_A": minimum_continuum_margin,
        "minimum_sphere_tangency_margin_A": minimum_sphere_tangency_margin,
        "geometry_records": summaries,
        "gates": gates,
        "diagnostic_gates_passed": all(gates.values()),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "opt_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }
    if continuum_kind == "harmonic-ddpcm-water":
        response_summaries = [
            _mapping(
                stationarity["response_operator_audit"],
                name="ddPCM response audit",
            )
            for stationarity in stationarity_summaries
        ]
        result.update(
            {
                "continuum_kind": continuum_kind,
                "maximum_stationarity_condition_number": max(
                    float(stationarity["maximum_condition_number"])
                    for stationarity in stationarity_summaries
                ),
                "maximum_stationarity_absolute_residual": max(
                    float(stationarity["maximum_absolute_residual"])
                    for stationarity in stationarity_summaries
                ),
                "maximum_stationarity_relative_residual": max(
                    float(stationarity["maximum_relative_residual"])
                    for stationarity in stationarity_summaries
                ),
                "maximum_stationarity_scaled_residual": max(
                    float(stationarity["maximum_scaled_residual"])
                    for stationarity in stationarity_summaries
                ),
                "maximum_energy_cotangent_relative_asymmetry": max(
                    float(response["energy_cotangent_relative_asymmetry"])
                    for response in response_summaries
                ),
                "maximum_kkt_vs_autograd_relative_error": max(
                    float(response["kkt_vs_autograd_relative_error"])
                    for response in response_summaries
                ),
                "maximum_half_coupling_absolute_error_eV": max(
                    float(response["half_coupling_absolute_error_eV"])
                    for response in response_summaries
                ),
                "maximum_primal_response_relative_asymmetry": max(
                    float(response["primal_response_relative_asymmetry"])
                    for response in response_summaries
                ),
                "maximum_primal_vs_energy_cotangent_relative_error": max(
                    float(response["primal_vs_energy_cotangent_relative_error"])
                    for response in response_summaries
                ),
            }
        )
    if nonpolar_kind != "none":
        result["nonpolar_kind"] = nonpolar_kind
    return result


def summarize_aimnet2_geometry_mediated_pes_panel(
    shards: Sequence[Mapping[str, object]],
    *,
    continuum_kind: str = "harmonic-point",
    nonpolar_kind: str = "none",
) -> dict[str, object]:
    """Recompute and aggregate all seventeen exact H/C/N/O shards."""

    contract = aimnet2_geometry_mediated_pes_continuum_contract(
        continuum_kind, nonpolar_kind=nonpolar_kind
    )
    values = tuple(shards)
    expected_count = len(AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS)
    if len(values) != expected_count:
        raise ValueError(
            f"AIMNet2 geometry-mediated PES panel requires exactly {expected_count} "
            "shards."
        )
    summaries: list[dict[str, object]] = []
    for expected_index, shard in enumerate(values):
        raw_index = shard.get("molecule_index")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, (int, np.integer))
            or int(raw_index) != expected_index
        ):
            raise ValueError("PES-panel shard indices changed from frozen order.")
        records = shard.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("Every PES-panel shard requires raw geometry records.")
        normalized_records = tuple(
            _mapping(record, name="PES-panel geometry record") for record in records
        )
        summaries.append(
            summarize_aimnet2_geometry_mediated_pes_shard(
                molecule_index=expected_index,
                records=normalized_records,
                continuum_kind=continuum_kind,
                nonpolar_kind=nonpolar_kind,
            )
        )

    gate_names = tuple(_mapping(summaries[0]["gates"], name="shard gates"))
    if any(
        tuple(_mapping(summary["gates"], name="shard gates")) != gate_names
        for summary in summaries[1:]
    ):
        raise RuntimeError("PES-panel shard gate schemas disagree.")
    gates = {
        name: all(
            _mapping(summary["gates"], name="shard gates").get(name) is True
            for summary in summaries
        )
        for name in gate_names
    }
    failed_molecule_ids = [
        str(summary["molecule_id"])
        for summary in summaries
        if summary["diagnostic_gates_passed"] is not True
    ]
    result = {
        "schema_version": contract.panel_summary_schema_version,
        "contract_version": contract.panel_contract_version,
        "molecule_count": len(summaries),
        "molecule_ids": [str(summary["molecule_id"]) for summary in summaries],
        "geometry_count": sum(int(summary["variant_count"]) for summary in summaries),
        "directional_record_count": sum(
            int(summary["directional_record_count"]) for summary in summaries
        ),
        "directional_sample_count": sum(
            int(summary["directional_sample_count"]) for summary in summaries
        ),
        "maximum_surface_condition_number": max(
            float(summary["maximum_surface_condition_number"]) for summary in summaries
        ),
        "maximum_reciprocity_absolute_error_eV": max(
            float(summary["maximum_reciprocity_absolute_error_eV"])
            for summary in summaries
        ),
        "maximum_reciprocity_relative_error": max(
            float(summary["maximum_reciprocity_relative_error"])
            for summary in summaries
        ),
        "maximum_apply_adjoint_absolute_error_eV": max(
            float(summary["maximum_apply_adjoint_absolute_error_eV"])
            for summary in summaries
        ),
        "maximum_apply_adjoint_relative_error": max(
            float(summary["maximum_apply_adjoint_relative_error"])
            for summary in summaries
        ),
        "maximum_charge_fd_absolute_error_eV_per_e": max(
            float(summary["maximum_charge_fd_absolute_error_eV_per_e"])
            for summary in summaries
        ),
        "maximum_charge_fd_relative_error": max(
            float(summary["maximum_charge_fd_relative_error"]) for summary in summaries
        ),
        "maximum_charge_gauge_vjp_norm_eV_per_A": max(
            float(summary["maximum_charge_gauge_vjp_norm_eV_per_A"])
            for summary in summaries
        ),
        "maximum_directional_absolute_error_eV_per_A": max(
            float(summary["maximum_directional_absolute_error_eV_per_A"])
            for summary in summaries
        ),
        "minimum_neighbor_cutoff_margin_A": min(
            float(summary["minimum_neighbor_cutoff_margin_A"]) for summary in summaries
        ),
        "minimum_continuum_event_margin_A": min(
            float(summary["minimum_continuum_event_margin_A"]) for summary in summaries
        ),
        "minimum_sphere_tangency_margin_A": min(
            float(summary["minimum_sphere_tangency_margin_A"]) for summary in summaries
        ),
        "failed_molecule_ids": failed_molecule_ids,
        "shard_summaries": summaries,
        "gates": gates,
        "diagnostic_gates_passed": not failed_molecule_ids and all(gates.values()),
        "capabilities": {tier: False for tier in ("E", "F", "H", "V", "M")},
        "opt_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }
    if continuum_kind == "harmonic-ddpcm-water":
        result.update(
            {
                "continuum_kind": continuum_kind,
                "maximum_stationarity_condition_number": max(
                    float(summary["maximum_stationarity_condition_number"])
                    for summary in summaries
                ),
                "maximum_stationarity_absolute_residual": max(
                    float(summary["maximum_stationarity_absolute_residual"])
                    for summary in summaries
                ),
                "maximum_stationarity_relative_residual": max(
                    float(summary["maximum_stationarity_relative_residual"])
                    for summary in summaries
                ),
                "maximum_stationarity_scaled_residual": max(
                    float(summary["maximum_stationarity_scaled_residual"])
                    for summary in summaries
                ),
                "maximum_energy_cotangent_relative_asymmetry": max(
                    float(summary["maximum_energy_cotangent_relative_asymmetry"])
                    for summary in summaries
                ),
                "maximum_kkt_vs_autograd_relative_error": max(
                    float(summary["maximum_kkt_vs_autograd_relative_error"])
                    for summary in summaries
                ),
                "maximum_half_coupling_absolute_error_eV": max(
                    float(summary["maximum_half_coupling_absolute_error_eV"])
                    for summary in summaries
                ),
                "maximum_primal_response_relative_asymmetry": max(
                    float(summary["maximum_primal_response_relative_asymmetry"])
                    for summary in summaries
                ),
                "maximum_primal_vs_energy_cotangent_relative_error": max(
                    float(summary["maximum_primal_vs_energy_cotangent_relative_error"])
                    for summary in summaries
                ),
            }
        )
    if nonpolar_kind != "none":
        result["nonpolar_kind"] = nonpolar_kind
    return result


__all__ = [
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_ARTIFACT_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_CONTRACT_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_PANEL_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_ARTIFACT_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_CONTRACT_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_PES_SHARD_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_ARTIFACT_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_CONTRACT_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_PANEL_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_ARTIFACT_SCHEMA_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_CONTRACT_VERSION",
    "AIMNET2_FROZEN_CHARGE_WATER_DDPCM_SMDCDS_PES_SHARD_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_EXCLUDED_MOLECULES",
    "AIMNET2_GEOMETRY_MEDIATED_PES_MOLECULE_IDS",
    "AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_ARTIFACT_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_PANEL_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_ARTIFACT_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_PES_SHARD_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_SUPPORTED_ATOMIC_NUMBERS",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_ABSOLUTE_TOLERANCE_EV_PER_E",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_MAXIMUM_CONDITION_NUMBER",
    "AIMNET2_GEOMETRY_MEDIATED_STATIONARITY_RELATIVE_TOLERANCE",
    "AIMNet2GeometryMediatedPESContinuumContract",
    "aimnet2_geometry_mediated_pes_continuum_contract",
    "aimnet2_geometry_mediated_pes_molecule",
    "summarize_aimnet2_geometry_mediated_stationarity",
    "summarize_aimnet2_geometry_mediated_pes_shard",
    "summarize_aimnet2_geometry_mediated_pes_panel",
]
