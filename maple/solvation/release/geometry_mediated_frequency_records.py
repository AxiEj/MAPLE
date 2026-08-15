"""Raw-record validation for the stationary-water frequency diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

from maple.solvation.coupling.state_equation import geometry_sha256

from .geometry_mediated import (
    geometry_mediated_trial_step_guard,
    summarize_geometry_mediated_reciprocity_audit,
)
from .geometry_mediated_panel import summarize_aimnet2_geometry_mediated_stationarity
from .geometry_mediated_path import aimnet2_geometry_mediated_water_loop_atoms


def mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


def _array(values: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return np.array(result, copy=True)


def finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _sha(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must contain exactly 64 hexadecimal digits.")
    return result


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))


def maximum_absolute(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def parse_frequency_center(
    raw: Mapping[str, object],
    *,
    positions: np.ndarray,
) -> dict[str, object]:
    """Validate the stationary scalar center and deterministic replay."""

    reference = aimnet2_geometry_mediated_water_loop_atoms((0.0, 0.0))
    atoms = reference.copy()
    atoms.positions = positions
    if raw.get("geometry_sha256") != geometry_sha256(atoms):
        raise ValueError("frequency center geometry SHA changed.")
    if not np.array_equal(np.asarray(raw.get("atomic_numbers")), atoms.numbers):
        raise ValueError("frequency center atomic numbers changed from water.")
    if not np.array_equal(
        _array(raw.get("positions_A"), shape=(3, 3), name="center positions"),
        positions,
    ):
        raise ValueError("frequency center positions differ from the root solution.")
    masses = _array(raw.get("masses_amu"), shape=(3,), name="center masses")
    if not np.array_equal(masses, atoms.get_masses()):
        raise ValueError("frequency center masses differ from ASE water masses.")

    energy = mapping(raw.get("energy"), name="frequency center energy")
    vacuum = finite(energy.get("vacuum_energy_eV"), name="vacuum energy")
    continuum = finite(energy.get("continuum_energy_eV"), name="continuum energy")
    total = finite(energy.get("total_energy_eV"), name="total energy")
    if not math.isclose(total, vacuum + continuum, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("frequency center scalar energy ledger does not close.")

    source = _array(raw.get("source"), shape=(3, 4), name="center source")
    reaction = _array(
        raw.get("reaction_field"), shape=(3, 4), name="center reaction field"
    )
    if (
        not np.array_equal(source[:, 1:], np.zeros((3, 3)))
        or abs(float(np.sum(source[:, 0]))) > 1.0e-10
    ):
        raise ValueError("frequency center source violates neutral point-l0.")
    if not math.isclose(
        continuum,
        0.5 * float(np.vdot(source, reaction)),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError("frequency center continuum energy violates half coupling.")

    component_names = (
        "intrinsic_gradient_eV_per_A",
        "continuum_fixed_source_gradient_eV_per_A",
        "source_response_gradient_eV_per_A",
    )
    components = {
        name: _array(raw.get(name), shape=(3, 3), name=f"center {name}")
        for name in component_names
    }
    gradient = _array(
        raw.get("total_gradient_eV_per_A"),
        shape=(3, 3),
        name="center total gradient",
    )
    gradient_ledger_error = maximum_absolute(sum(components.values()) - gradient)
    if gradient_ledger_error > 2.0e-10:
        raise ValueError("frequency center gradient component ledger does not close.")

    model_topology = mapping(raw.get("model_topology"), name="model topology")
    continuum_topology = mapping(
        raw.get("continuum_topology"), name="continuum topology"
    )
    event_guard = geometry_mediated_trial_step_guard(
        center_positions_A=positions,
        trial_positions_A=positions,
        center_model_topology=model_topology,
        trial_model_topology=model_topology,
        center_continuum_topology=continuum_topology,
        trial_continuum_topology=continuum_topology,
    )
    stationarity = summarize_aimnet2_geometry_mediated_stationarity(
        mapping(raw.get("stationarity"), name="center PCM stationarity")
    )
    reciprocity = summarize_geometry_mediated_reciprocity_audit(
        mapping(raw.get("reciprocity"), name="center reciprocity"),
        reaction_field=reaction,
    )

    replay = mapping(raw.get("replay"), name="center deterministic replay")
    replay_energy = finite(replay.get("total_energy_eV"), name="replay energy")
    replay_source = _array(replay.get("source"), shape=(3, 4), name="replay source")
    replay_gradient = _array(
        replay.get("total_gradient_eV_per_A"),
        shape=(3, 3),
        name="replay gradient",
    )
    replay_summary = {
        "energy_absolute_error_eV": abs(replay_energy - total),
        "source_difference_norm": norm(replay_source - source),
        "gradient_difference_norm_eV_per_A": norm(replay_gradient - gradient),
    }
    replay_summary["gate_passed"] = all(
        value <= tolerance
        for value, tolerance in zip(
            replay_summary.values(),
            (1.0e-8, 1.0e-8, 1.0e-7),
            strict=True,
        )
    )
    return {
        "atoms": atoms,
        "masses": masses,
        "source": source,
        "reaction": reaction,
        "gradient": gradient,
        "model_topology": model_topology,
        "continuum_topology": continuum_topology,
        "summary": {
            "geometry_sha256": geometry_sha256(atoms),
            "vacuum_energy_eV": vacuum,
            "continuum_energy_eV": continuum,
            "total_energy_eV": total,
            "gradient_norm_eV_per_A": norm(gradient),
            "gradient_max_abs_eV_per_A": maximum_absolute(gradient),
            "gradient_ledger_max_abs_eV_per_A": gradient_ledger_error,
            "event_guard": event_guard,
            "stationarity": stationarity,
            "reciprocity": reciprocity,
            "deterministic_replay": replay_summary,
        },
    }


def parse_dense_hvp_record(
    raw: Mapping[str, object],
    *,
    coordinate_index: int,
    center: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, object], dict[str, float]]:
    """Validate one canonical Cartesian HVP record."""

    expected_direction = np.zeros((3, 3), dtype=float)
    expected_direction.reshape(-1)[coordinate_index] = 1.0
    if int(raw.get("coordinate_index")) != coordinate_index or not np.array_equal(
        _array(raw.get("coordinate_direction"), shape=(3, 3), name="HVP direction"),
        expected_direction,
    ):
        raise ValueError("dense Cartesian HVP basis changed from the contract.")
    source = _array(raw.get("source"), shape=(3, 4), name="HVP source")
    if not np.array_equal(source, center["source"]):
        raise ValueError("dense HVP source differs from the center source.")
    cotangent = _array(
        raw.get("source_gradient_cotangent"),
        shape=(3, 4),
        name="HVP source cotangent",
    )
    cotangent_error = maximum_absolute(cotangent - center["reaction"])

    source_jvp = _array(
        raw.get("source_position_jvp"), shape=(3, 4), name="HVP source JVP"
    )
    if not np.array_equal(source_jvp[:, 1:], np.zeros((3, 3))):
        raise ValueError("dense HVP source JVP is not point-l0.")
    tangent_residual = abs(float(np.sum(source_jvp[:, 0])))
    component_names = (
        "intrinsic_energy_hvp_eV_per_A2",
        "continuum_joint_position_hvp_eV_per_A2",
        "continuum_source_response_pullback_eV_per_A2",
        "contracted_source_hessian_eV_per_A2",
    )
    components = {
        name: _array(raw.get(name), shape=(3, 3), name=f"HVP {name}")
        for name in component_names
    }
    total_hvp = _array(raw.get("total_hvp_eV_per_A2"), shape=(3, 3), name="total HVP")
    ledger_error = maximum_absolute(sum(components.values()) - total_hvp)
    parity = {
        "energy_absolute_error_eV": finite(
            raw.get("model_standard_decomposed_energy_absolute_error_eV"),
            name="energy parity",
            nonnegative=True,
        ),
        "charge_max_absolute_error_e": finite(
            raw.get("model_standard_decomposed_charge_max_absolute_error_e"),
            name="charge parity",
            nonnegative=True,
        ),
        "intrinsic_gradient_max_absolute_error_eV_per_A": finite(
            raw.get(
                "model_standard_decomposed_intrinsic_gradient_max_absolute_error_eV_per_A"
            ),
            name="gradient parity",
            nonnegative=True,
        ),
        "charge_vjp_max_absolute_error_eV_per_A": finite(
            raw.get("model_standard_decomposed_charge_vjp_max_absolute_error_eV_per_A"),
            name="charge VJP parity",
            nonnegative=True,
        ),
        "charge_tangent_residual_e_per_A": finite(
            raw.get("model_charge_tangent_residual_e_per_A"),
            name="charge tangent residual",
            nonnegative=True,
        ),
    }
    if (
        raw.get("diagnostic_only") is not True
        or raw.get("tier_h_admitted") is not False
    ):
        raise ValueError("dense HVP record attempted to change admission state.")
    identity = {
        "scalar_id": _text(raw.get("scalar_id"), name="scalar ID"),
        "profile_id": _text(raw.get("profile_id"), name="profile ID"),
        "scalar_fingerprint_sha256": _sha(
            raw.get("scalar_fingerprint_sha256"), name="scalar fingerprint"
        ),
        "model_second_order_behavior_sha256": _sha(
            raw.get("model_second_order_behavior_sha256"), name="model behavior"
        ),
        "continuum_second_order_behavior_sha256": _sha(
            raw.get("continuum_second_order_behavior_sha256"),
            name="continuum behavior",
        ),
    }
    summary = {
        "coordinate_index": coordinate_index,
        **identity,
        "source_cotangent_max_abs_error_eV_per_e": cotangent_error,
        "source_tangent_residual_e_per_A": tangent_residual,
        "component_ledger_max_abs_eV_per_A2": ledger_error,
        "total_hvp_norm_eV_per_A2": norm(total_hvp),
        "model_parity": parity,
        "diagnostic_only": True,
        "tier_h_admitted": False,
    }
    return total_hvp.reshape(-1), summary, parity


def parse_frequency_fd_endpoint(
    raw: Mapping[str, object],
    *,
    expected_positions: np.ndarray,
    center: Mapping[str, object],
    name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Validate one total-gradient endpoint in the dense Cartesian stencil."""

    positions = _array(raw.get("positions_A"), shape=(3, 3), name=f"{name} positions")
    if not np.array_equal(positions, expected_positions):
        raise ValueError(f"{name} positions changed from the Cartesian stencil.")
    atoms = center["atoms"].copy()
    atoms.positions = positions
    if raw.get("geometry_sha256") != geometry_sha256(atoms):
        raise ValueError(f"{name} geometry SHA changed.")
    gradient = _array(
        raw.get("total_gradient_eV_per_A"),
        shape=(3, 3),
        name=f"{name} gradient",
    )
    model_topology = mapping(raw.get("model_topology"), name=f"{name} model topology")
    continuum_topology = mapping(
        raw.get("continuum_topology"), name=f"{name} continuum topology"
    )
    guard = geometry_mediated_trial_step_guard(
        center_positions_A=center["atoms"].positions,
        trial_positions_A=positions,
        center_model_topology=center["model_topology"],
        trial_model_topology=model_topology,
        center_continuum_topology=center["continuum_topology"],
        trial_continuum_topology=continuum_topology,
    )
    return gradient.reshape(-1), guard


__all__ = [
    "finite",
    "mapping",
    "maximum_absolute",
    "norm",
    "parse_dense_hvp_record",
    "parse_frequency_center",
    "parse_frequency_fd_endpoint",
    "sequence",
]
