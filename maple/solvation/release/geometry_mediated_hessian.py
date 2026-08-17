"""Source-recomputable HVP audit for the weak AIMNet2/PCM scalar.

The contract freezes one water geometry, two internal directions, all three
rigid translations, and a three-step central-difference stencil.  It validates
the complete weak-scalar HVP ledger

``H_E h + (G_RR h + G_Rc J_c h)``
``+ J_c.T (G_cR h + G_cc J_c h) + D_R[J_c.T v][h]``

from raw operands.  A passing result is a local, fixed-stratum canary only.  It
does not admit Tier H, FREQ, TS, IRC, MD, a public ASE HVP, or global ``C2``
regularity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256

from .geometry_mediated import (
    geometry_mediated_trial_step_guard,
    summarize_geometry_mediated_reciprocity_audit,
)
from .geometry_mediated_panel import (
    summarize_aimnet2_geometry_mediated_stationarity,
)
from .geometry_mediated_path import (
    aimnet2_geometry_mediated_water_loop_atoms,
    aimnet2_geometry_mediated_water_loop_directions,
)

AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-hvp-water-contract-v1"
)
AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-hvp-water-summary-v1"
)
AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A = (4.0e-4, 2.0e-4, 1.0e-4)
AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES = (
    "seeded-internal",
    "orthogonal-radial-internal",
    "translation-x",
    "translation-y",
    "translation-z",
)

# These tolerances are preregistered against the source-bound float64 runtime.
# They remain diagnostic thresholds, not public capability tolerances.
AIMNET2_GEOMETRY_MEDIATED_HVP_CENTRAL_CONVERGENCE_MAX_RATIO = 0.45
AIMNET2_GEOMETRY_MEDIATED_CHARGE_JVP_FD_TOLERANCE_E_PER_A = 5.0e-8
AIMNET2_GEOMETRY_MEDIATED_CONTRACTED_CHARGE_HESSIAN_FD_TOLERANCE_EV_PER_A2 = 1.0e-6
AIMNET2_GEOMETRY_MEDIATED_TOTAL_HVP_FD_TOLERANCE_EV_PER_A2 = 5.0e-5
AIMNET2_GEOMETRY_MEDIATED_HVP_BILINEAR_TOLERANCE_EV_PER_A2 = 2.0e-8
AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_SOURCE_JVP_TOLERANCE_E_PER_A = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_HVP_TOLERANCE_EV_PER_A2 = 2.0e-8
AIMNET2_GEOMETRY_MEDIATED_HVP_COMPONENT_LEDGER_TOLERANCE_EV_PER_A2 = 2.0e-10
AIMNET2_GEOMETRY_MEDIATED_HVP_SOURCE_COTANGENT_TOLERANCE_EV_PER_E = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_HVP_CHARGE_TANGENT_TOLERANCE_E_PER_A = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_ENERGY_PARITY_TOLERANCE_EV = 1.0e-8
AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_CHARGE_PARITY_TOLERANCE_E = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_GRADIENT_PARITY_TOLERANCE_EV_PER_A = 1.0e-7
AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_VJP_PARITY_TOLERANCE_EV_PER_A = 1.0e-10

_NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
_NO_WORKFLOWS = {
    name: False
    for name in (
        "public_ase_hvp",
        "opt",
        "freq",
        "ts",
        "irc",
        "md",
    )
}


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


def _array(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.array(result, copy=True)


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def _convergence_ratios(errors: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in errors)
    tiny = np.finfo(float).tiny
    return tuple(
        current / max(previous, tiny) for previous, current in zip(values, values[1:])
    )


def aimnet2_geometry_mediated_hvp_translation_directions() -> tuple[np.ndarray, ...]:
    """Return the three normalized rigid-translation directions for water."""

    atom_count = len(aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0)))
    result = []
    for axis in range(3):
        direction = np.zeros((atom_count, 3), dtype=float)
        direction[:, axis] = 1.0 / math.sqrt(atom_count)
        direction.setflags(write=False)
        result.append(direction)
    return tuple(result)


def aimnet2_geometry_mediated_hvp_directions() -> tuple[np.ndarray, ...]:
    """Return the complete frozen direction panel in contract order."""

    return (
        *aimnet2_geometry_mediated_water_loop_directions(),
        *aimnet2_geometry_mediated_hvp_translation_directions(),
    )


@dataclass(frozen=True, slots=True)
class _Center:
    positions: np.ndarray
    source: np.ndarray
    reaction_field: np.ndarray
    total_gradient: np.ndarray
    model_topology: Mapping[str, object]
    continuum_topology: Mapping[str, object]
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class _HVP:
    label: str
    direction: np.ndarray
    source_cotangent: np.ndarray
    source_jvp: np.ndarray
    contracted_source_hessian: np.ndarray
    total_hvp: np.ndarray
    summary: dict[str, object]


def _center_record(raw: Mapping[str, object], *, continuum_kind: str) -> _Center:
    expected_atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    shape = (len(expected_atoms), 3)
    source_shape = (len(expected_atoms), 4)
    positions = _array(raw.get("positions_A"), shape=shape, name="center positions")
    numbers = np.asarray(raw.get("atomic_numbers"))
    if not np.array_equal(numbers, expected_atoms.numbers) or not np.array_equal(
        positions, expected_atoms.positions
    ):
        raise ValueError("HVP water center geometry changed from the frozen contract.")
    expected_geometry_sha = geometry_sha256(expected_atoms)
    if raw.get("geometry_sha256") != expected_geometry_sha:
        raise ValueError("HVP water center geometry SHA changed.")

    energy = _mapping(raw.get("energy"), name="center energy")
    vacuum = _finite(energy.get("vacuum_energy_eV"), name="center vacuum energy")
    continuum = _finite(
        energy.get("continuum_energy_eV"), name="center continuum energy"
    )
    total = _finite(energy.get("total_energy_eV"), name="center total energy")
    if not math.isclose(total, vacuum + continuum, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("HVP center scalar energy ledger does not close.")

    source = _array(raw.get("source"), shape=source_shape, name="center source")
    reaction = _array(
        raw.get("reaction_field"),
        shape=source_shape,
        name="center reaction field",
    )
    if not np.array_equal(source[:, 1:], np.zeros((len(expected_atoms), 3))):
        raise ValueError("HVP center source is not exactly point-l0.")
    if abs(float(np.sum(source[:, 0]))) > 1.0e-10:
        raise ValueError("HVP center source violates the neutral-water contract.")
    if not math.isclose(
        continuum,
        0.5 * float(np.vdot(source, reaction)),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError("HVP center continuum energy violates half coupling.")

    intrinsic = _array(
        raw.get("intrinsic_gradient_eV_per_A"),
        shape=shape,
        name="center intrinsic gradient",
    )
    fixed_source = _array(
        raw.get("continuum_fixed_source_gradient_eV_per_A"),
        shape=shape,
        name="center fixed-source gradient",
    )
    source_response = _array(
        raw.get("source_response_gradient_eV_per_A"),
        shape=shape,
        name="center source-response gradient",
    )
    total_gradient = _array(
        raw.get("total_gradient_eV_per_A"),
        shape=shape,
        name="center total gradient",
    )
    gradient_ledger_error = _maximum_absolute(
        intrinsic + fixed_source + source_response - total_gradient
    )
    if gradient_ledger_error > 2.0e-10:
        raise ValueError("HVP center gradient component ledger does not close.")

    model_topology = _mapping(raw.get("model_topology"), name="center model topology")
    continuum_topology = _mapping(
        raw.get("continuum_topology"), name="center continuum topology"
    )
    center_guard = geometry_mediated_trial_step_guard(
        center_positions_A=positions,
        trial_positions_A=positions,
        center_model_topology=model_topology,
        trial_model_topology=model_topology,
        center_continuum_topology=continuum_topology,
        trial_continuum_topology=continuum_topology,
    )
    stationarity = summarize_aimnet2_geometry_mediated_stationarity(
        _mapping(raw.get("stationarity"), name="center stationarity"),
        continuum_kind=continuum_kind,
    )
    reciprocity = summarize_geometry_mediated_reciprocity_audit(
        _mapping(raw.get("reciprocity"), name="center reciprocity"),
        reaction_field=reaction,
    )
    summary = {
        "geometry_sha256": expected_geometry_sha,
        "vacuum_energy_eV": vacuum,
        "continuum_energy_eV": continuum,
        "total_energy_eV": total,
        "gradient_ledger_max_absolute_error_eV_per_A": gradient_ledger_error,
        "source_total_charge_e": float(np.sum(source[:, 0])),
        "center_event_guard": center_guard,
        "stationarity": stationarity,
        "reciprocity": reciprocity,
    }
    return _Center(
        positions=positions,
        source=source,
        reaction_field=reaction,
        total_gradient=total_gradient,
        model_topology=model_topology,
        continuum_topology=continuum_topology,
        summary=summary,
    )


def _hvp_record(
    raw: Mapping[str, object],
    *,
    expected_label: str,
    expected_direction: np.ndarray,
    center: _Center,
) -> _HVP:
    if raw.get("label") != expected_label:
        raise ValueError("HVP direction labels changed from the frozen contract.")
    shape = center.positions.shape
    source_shape = center.source.shape
    direction = _array(
        raw.get("coordinate_direction"), shape=shape, name=f"{expected_label} direction"
    )
    if not np.array_equal(direction, expected_direction):
        raise ValueError(f"{expected_label} direction changed from the contract.")
    source = _array(raw.get("source"), shape=source_shape, name="HVP source")
    if not np.array_equal(source, center.source):
        raise ValueError("HVP source differs from the center first-order state.")
    source_cotangent = _array(
        raw.get("source_gradient_cotangent"),
        shape=source_shape,
        name="HVP source cotangent",
    )
    cotangent_error = _maximum_absolute(source_cotangent - center.reaction_field)
    if (
        cotangent_error
        > AIMNET2_GEOMETRY_MEDIATED_HVP_SOURCE_COTANGENT_TOLERANCE_EV_PER_E
    ):
        raise ValueError("HVP source cotangent differs from the reaction potential.")
    source_jvp = _array(
        raw.get("source_position_jvp"), shape=source_shape, name="source JVP"
    )
    if not np.array_equal(source_jvp[:, 1:], np.zeros((shape[0], 3))):
        raise ValueError("HVP source JVP is not exactly point-l0.")
    tangent_residual = abs(float(np.sum(source_jvp[:, 0])))

    component_names = (
        "intrinsic_energy_hvp_eV_per_A2",
        "continuum_joint_position_hvp_eV_per_A2",
        "continuum_source_response_pullback_eV_per_A2",
        "contracted_source_hessian_eV_per_A2",
    )
    components = {
        name: _array(raw.get(name), shape=shape, name=f"{expected_label} {name}")
        for name in component_names
    }
    continuum_source_hvp = _array(
        raw.get("continuum_joint_source_hvp"),
        shape=source_shape,
        name=f"{expected_label} continuum source HVP",
    )
    total_hvp = _array(
        raw.get("total_hvp_eV_per_A2"),
        shape=shape,
        name=f"{expected_label} total HVP",
    )
    ledger_error = _maximum_absolute(sum(components.values()) - total_hvp)
    if (
        ledger_error
        > AIMNET2_GEOMETRY_MEDIATED_HVP_COMPONENT_LEDGER_TOLERANCE_EV_PER_A2
    ):
        raise ValueError(f"{expected_label} HVP component ledger does not close.")

    parity = {
        "energy_absolute_error_eV": _finite(
            raw.get("model_standard_decomposed_energy_absolute_error_eV"),
            name="model energy parity",
            nonnegative=True,
        ),
        "charge_max_absolute_error_e": _finite(
            raw.get("model_standard_decomposed_charge_max_absolute_error_e"),
            name="model charge parity",
            nonnegative=True,
        ),
        "intrinsic_gradient_max_absolute_error_eV_per_A": _finite(
            raw.get(
                "model_standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A"
            ),
            name="model gradient parity",
            nonnegative=True,
        ),
        "charge_vjp_max_absolute_error_eV_per_A": _finite(
            raw.get("model_standard_decomposed_charge_vjp_max_absolute_error_eV_per_A"),
            name="model charge VJP parity",
            nonnegative=True,
        ),
        "charge_tangent_residual_e_per_A": _finite(
            raw.get("model_charge_tangent_residual_e_per_A"),
            name="model charge tangent residual",
            nonnegative=True,
        ),
    }
    if (
        raw.get("diagnostic_only") is not True
        or raw.get("tier_h_admitted") is not False
    ):
        raise ValueError("HVP raw record attempted to change its admission boundary.")

    summary = {
        "label": expected_label,
        "scalar_id": _text(raw.get("scalar_id"), name="scalar ID"),
        "profile_id": _text(raw.get("profile_id"), name="profile ID"),
        "scalar_fingerprint_sha256": _sha(
            raw.get("scalar_fingerprint_sha256"), name="scalar fingerprint"
        ),
        "model_second_order_behavior_sha256": _sha(
            raw.get("model_second_order_behavior_sha256"),
            name="model second-order behavior",
        ),
        "continuum_second_order_behavior_sha256": _sha(
            raw.get("continuum_second_order_behavior_sha256"),
            name="continuum second-order behavior",
        ),
        "source_cotangent_max_absolute_error_eV_per_e": cotangent_error,
        "source_tangent_residual_e_per_A": tangent_residual,
        "component_ledger_max_absolute_error_eV_per_A2": ledger_error,
        "component_norms": {
            **{name: _norm(values) for name, values in components.items()},
            "continuum_joint_source_hvp": _norm(continuum_source_hvp),
            "total_hvp_eV_per_A2": _norm(total_hvp),
        },
        "model_parity": parity,
        "diagnostic_only": True,
        "tier_h_admitted": False,
    }
    return _HVP(
        label=expected_label,
        direction=direction,
        source_cotangent=source_cotangent,
        source_jvp=source_jvp,
        contracted_source_hessian=components["contracted_source_hessian_eV_per_A2"],
        total_hvp=total_hvp,
        summary=summary,
    )


def _endpoint(
    raw: Mapping[str, object],
    *,
    expected_positions: np.ndarray,
    center: _Center,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    positions = _array(raw.get("positions_A"), shape=center.positions.shape, name=name)
    if not np.array_equal(positions, expected_positions):
        raise ValueError(f"{name} positions changed from the frozen stencil.")
    expected_atoms = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    expected_atoms.positions = expected_positions
    if raw.get("geometry_sha256") != geometry_sha256(expected_atoms):
        raise ValueError(f"{name} geometry SHA changed from the frozen stencil.")
    numbers = np.asarray(raw.get("atomic_numbers"))
    if not np.array_equal(numbers, expected_atoms.numbers):
        raise ValueError(f"{name} atomic numbers changed from water.")
    source = _array(raw.get("source"), shape=center.source.shape, name=f"{name} source")
    if not np.array_equal(source[:, 1:], np.zeros((len(source), 3))):
        raise ValueError(f"{name} source is not exactly point-l0.")
    if abs(float(np.sum(source[:, 0]))) > 1.0e-10:
        raise ValueError(f"{name} source violates fixed total charge.")
    fixed_vjp = _array(
        raw.get("fixed_source_cotangent_vjp_eV_per_A"),
        shape=center.positions.shape,
        name=f"{name} fixed-cotangent VJP",
    )
    gradient = _array(
        raw.get("total_gradient_eV_per_A"),
        shape=center.positions.shape,
        name=f"{name} total gradient",
    )
    guard = geometry_mediated_trial_step_guard(
        center_positions_A=center.positions,
        trial_positions_A=positions,
        center_model_topology=center.model_topology,
        trial_model_topology=_mapping(
            raw.get("model_topology"), name=f"{name} model topology"
        ),
        center_continuum_topology=center.continuum_topology,
        trial_continuum_topology=_mapping(
            raw.get("continuum_topology"), name=f"{name} continuum topology"
        ),
    )
    return source, fixed_vjp, gradient, guard


def summarize_aimnet2_geometry_mediated_hvp_water(
    *,
    center_record: Mapping[str, object],
    direction_records: Sequence[Mapping[str, object]],
    finite_difference_records: Sequence[Mapping[str, object]],
    continuum_kind: str = "harmonic-point",
) -> dict[str, object]:
    """Recompute the complete local HVP canary from raw tensor operands."""

    center = _center_record(
        _mapping(center_record, name="center record"),
        continuum_kind=continuum_kind,
    )
    expected_directions = aimnet2_geometry_mediated_hvp_directions()
    raw_directions = _sequence(direction_records, name="HVP direction records")
    if len(raw_directions) != len(AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES):
        raise ValueError("HVP direction panel coverage changed from the contract.")
    hvps = tuple(
        _hvp_record(
            _mapping(raw, name=f"HVP direction record {label}"),
            expected_label=label,
            expected_direction=direction,
            center=center,
        )
        for raw, label, direction in zip(
            raw_directions,
            AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES,
            expected_directions,
            strict=True,
        )
    )
    first, second, *translations = hvps
    identity_fields = (
        "scalar_id",
        "profile_id",
        "scalar_fingerprint_sha256",
        "model_second_order_behavior_sha256",
        "continuum_second_order_behavior_sha256",
    )
    for field in identity_fields:
        values = {record.summary[field] for record in hvps}
        if len(values) != 1:
            raise ValueError(f"HVP direction records disagree on {field}.")
    if any(
        not np.array_equal(record.source_cotangent, first.source_cotangent)
        for record in hvps[1:]
    ):
        raise ValueError("HVP direction records changed the center source cotangent.")

    raw_fd = _sequence(finite_difference_records, name="HVP finite differences")
    if len(raw_fd) != len(AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A):
        raise ValueError("HVP finite-difference coverage changed from the contract.")
    charge_jvp_errors: list[float] = []
    contracted_hessian_errors: list[float] = []
    total_hvp_errors: list[float] = []
    fd_summaries: list[dict[str, object]] = []
    all_segments_certified = True
    all_topologies_match = True
    for raw, expected_step in zip(
        raw_fd, AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A, strict=True
    ):
        record = _mapping(raw, name="HVP finite-difference record")
        step = _finite(record.get("step_A"), name="HVP finite-difference step")
        if step != expected_step:
            raise ValueError("HVP finite-difference steps changed from the contract.")
        plus_positions = center.positions + step * first.direction
        minus_positions = center.positions - step * first.direction
        plus_source, plus_vjp, plus_gradient, plus_guard = _endpoint(
            _mapping(record.get("plus"), name="HVP plus endpoint"),
            expected_positions=plus_positions,
            center=center,
            name=f"HVP +{step:g} A endpoint",
        )
        minus_source, minus_vjp, minus_gradient, minus_guard = _endpoint(
            _mapping(record.get("minus"), name="HVP minus endpoint"),
            expected_positions=minus_positions,
            center=center,
            name=f"HVP -{step:g} A endpoint",
        )
        charge_jvp_fd = (plus_source - minus_source) / (2.0 * step)
        contracted_hessian_fd = (plus_vjp - minus_vjp) / (2.0 * step)
        total_hvp_fd = (plus_gradient - minus_gradient) / (2.0 * step)
        charge_error = _norm(charge_jvp_fd - first.source_jvp)
        contracted_error = _norm(
            contracted_hessian_fd - first.contracted_source_hessian
        )
        total_error = _norm(total_hvp_fd - first.total_hvp)
        charge_jvp_errors.append(charge_error)
        contracted_hessian_errors.append(contracted_error)
        total_hvp_errors.append(total_error)
        segments_certified = bool(plus_guard["gate_passed"]) and bool(
            minus_guard["gate_passed"]
        )
        topologies_match = all(
            bool(guard["same_model_topology"])
            and bool(guard["same_continuum_topology"])
            for guard in (plus_guard, minus_guard)
        )
        all_segments_certified &= segments_certified
        all_topologies_match &= topologies_match
        fd_summaries.append(
            {
                "step_A": step,
                "charge_jvp_error_norm_e_per_A": charge_error,
                "contracted_charge_hessian_error_norm_eV_per_A2": contracted_error,
                "total_hvp_error_norm_eV_per_A2": total_error,
                "plus_segment_guard": plus_guard,
                "minus_segment_guard": minus_guard,
            }
        )

    charge_ratios = _convergence_ratios(charge_jvp_errors)
    contracted_ratios = _convergence_ratios(contracted_hessian_errors)
    total_ratios = _convergence_ratios(total_hvp_errors)
    convergence_gate = all(
        ratio < AIMNET2_GEOMETRY_MEDIATED_HVP_CENTRAL_CONVERGENCE_MAX_RATIO
        for ratio in (*charge_ratios, *contracted_ratios, *total_ratios)
    )
    bilinear_left = float(np.vdot(first.direction, second.total_hvp))
    bilinear_right = float(np.vdot(second.direction, first.total_hvp))
    bilinear_error = abs(bilinear_left - bilinear_right)
    translation_source_norms = tuple(
        _norm(record.source_jvp) for record in translations
    )
    translation_hvp_norms = tuple(_norm(record.total_hvp) for record in translations)

    parity_maxima = {
        name: max(float(record.summary["model_parity"][name]) for record in hvps)
        for name in first.summary["model_parity"]
    }
    maximum_source_tangent_residual = max(
        float(record.summary["source_tangent_residual_e_per_A"]) for record in hvps
    )
    maximum_ledger_error = max(
        float(record.summary["component_ledger_max_absolute_error_eV_per_A2"])
        for record in hvps
    )
    gates = {
        "center_scalar_and_gradient_ledgers": (
            center.summary["gradient_ledger_max_absolute_error_eV_per_A"] <= 2.0e-10
        ),
        "center_stationarity": bool(center.summary["stationarity"]["gate_passed"]),
        "center_reciprocity_metric_and_charge_gauge": bool(
            center.summary["reciprocity"]["gate_passed"]
        ),
        "center_event_guard": bool(center.summary["center_event_guard"]["gate_passed"]),
        "all_hvp_component_ledgers": (
            maximum_ledger_error
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_COMPONENT_LEDGER_TOLERANCE_EV_PER_A2
        ),
        "model_second_order_forward_parity": (
            parity_maxima["energy_absolute_error_eV"]
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_ENERGY_PARITY_TOLERANCE_EV
            and parity_maxima["charge_max_absolute_error_e"]
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_CHARGE_PARITY_TOLERANCE_E
            and parity_maxima["intrinsic_gradient_max_absolute_error_eV_per_A"]
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_GRADIENT_PARITY_TOLERANCE_EV_PER_A
            and parity_maxima["charge_vjp_max_absolute_error_eV_per_A"]
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_VJP_PARITY_TOLERANCE_EV_PER_A
        ),
        "fixed_total_charge_tangent": (
            maximum_source_tangent_residual
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_CHARGE_TANGENT_TOLERANCE_E_PER_A
            and parity_maxima["charge_tangent_residual_e_per_A"]
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_CHARGE_TANGENT_TOLERANCE_E_PER_A
        ),
        "all_stencils_same_stratum": all_topologies_match,
        "all_stencil_segments_event_guarded": all_segments_certified,
        "central_second_order_convergence": convergence_gate,
        "charge_jvp_finite_difference": (
            charge_jvp_errors[-1]
            <= AIMNET2_GEOMETRY_MEDIATED_CHARGE_JVP_FD_TOLERANCE_E_PER_A
        ),
        "contracted_charge_hessian_finite_difference": (
            contracted_hessian_errors[-1]
            <= AIMNET2_GEOMETRY_MEDIATED_CONTRACTED_CHARGE_HESSIAN_FD_TOLERANCE_EV_PER_A2
        ),
        "complete_hvp_total_gradient_finite_difference": (
            total_hvp_errors[-1]
            <= AIMNET2_GEOMETRY_MEDIATED_TOTAL_HVP_FD_TOLERANCE_EV_PER_A2
        ),
        "hessian_bilinear_symmetry": (
            bilinear_error <= AIMNET2_GEOMETRY_MEDIATED_HVP_BILINEAR_TOLERANCE_EV_PER_A2
        ),
        "three_translation_source_jvp_zero_modes": all(
            value <= AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_SOURCE_JVP_TOLERANCE_E_PER_A
            for value in translation_source_norms
        ),
        "three_translation_total_hvp_zero_modes": all(
            value <= AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_HVP_TOLERANCE_EV_PER_A2
            for value in translation_hvp_norms
        ),
        "diagnostic_only_admission_flags": all(
            record.summary["diagnostic_only"] is True
            and record.summary["tier_h_admitted"] is False
            for record in hvps
        ),
    }
    return {
        "schema_version": AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_SCHEMA_VERSION,
        "water_only": True,
        "fixed_graph_cavity_stratum_only": True,
        "conductor_reference_only": continuum_kind == "harmonic-point",
        "finite_dielectric_parameterization": continuum_kind != "harmonic-point",
        "center": center.summary,
        "direction_records": [record.summary for record in hvps],
        "finite_difference_records": fd_summaries,
        "finite_difference_error_norms": {
            "charge_jvp_e_per_A": charge_jvp_errors,
            "contracted_charge_hessian_eV_per_A2": contracted_hessian_errors,
            "total_hvp_eV_per_A2": total_hvp_errors,
        },
        "central_convergence_ratios": {
            "charge_jvp": list(charge_ratios),
            "contracted_charge_hessian": list(contracted_ratios),
            "total_hvp": list(total_ratios),
        },
        "bilinear_symmetry": {
            "first_dot_H_second_eV_per_A2": bilinear_left,
            "second_dot_H_first_eV_per_A2": bilinear_right,
            "absolute_error_eV_per_A2": bilinear_error,
        },
        "translation_zero_modes": {
            "source_jvp_norms_e_per_A": list(translation_source_norms),
            "total_hvp_norms_eV_per_A2": list(translation_hvp_norms),
        },
        "model_parity_maxima": parity_maxima,
        "maximum_source_tangent_residual_e_per_A": maximum_source_tangent_residual,
        "maximum_component_ledger_error_eV_per_A2": maximum_ledger_error,
        "thresholds": {
            "central_convergence_max_ratio": (
                AIMNET2_GEOMETRY_MEDIATED_HVP_CENTRAL_CONVERGENCE_MAX_RATIO
            ),
            "charge_jvp_fd_e_per_A": (
                AIMNET2_GEOMETRY_MEDIATED_CHARGE_JVP_FD_TOLERANCE_E_PER_A
            ),
            "contracted_charge_hessian_fd_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_CONTRACTED_CHARGE_HESSIAN_FD_TOLERANCE_EV_PER_A2
            ),
            "total_hvp_fd_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_TOTAL_HVP_FD_TOLERANCE_EV_PER_A2
            ),
            "bilinear_symmetry_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_HVP_BILINEAR_TOLERANCE_EV_PER_A2
            ),
            "translation_source_jvp_e_per_A": (
                AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_SOURCE_JVP_TOLERANCE_E_PER_A
            ),
            "translation_total_hvp_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_HVP_TOLERANCE_EV_PER_A2
            ),
        },
        "gates": gates,
        "diagnostic_gates_passed": all(gates.values()),
        "capabilities": dict(_NO_CAPABILITIES),
        "workflow_admission": dict(_NO_WORKFLOWS),
        "tier_h_admitted": False,
        "freq_ts_irc_admitted": False,
        "md_admitted": False,
    }


__all__ = [
    "AIMNET2_GEOMETRY_MEDIATED_CHARGE_JVP_FD_TOLERANCE_E_PER_A",
    "AIMNET2_GEOMETRY_MEDIATED_CONTRACTED_CHARGE_HESSIAN_FD_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_HVP_BILINEAR_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_HVP_CENTRAL_CONVERGENCE_MAX_RATIO",
    "AIMNET2_GEOMETRY_MEDIATED_HVP_DIRECTION_NAMES",
    "AIMNET2_GEOMETRY_MEDIATED_HVP_STEPS_A",
    "AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_HVP_WATER_SCHEMA_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_TOTAL_HVP_FD_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_HVP_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_TRANSLATION_SOURCE_JVP_TOLERANCE_E_PER_A",
    "aimnet2_geometry_mediated_hvp_directions",
    "aimnet2_geometry_mediated_hvp_translation_directions",
    "summarize_aimnet2_geometry_mediated_hvp_water",
]
