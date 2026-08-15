"""Stationary-water dense Hessian and vibrational-subspace evidence reducer.

The reducer consumes only raw scalar, HVP, and finite-difference operands.  It
reconstructs the full Cartesian Hessian, checks its symmetry and rigid modes,
compares every column with central differences of the total scalar gradient,
and diagonalizes the correctly mass-weighted vibrational block.  Passing is a
local implementation canary; all public E/F/H/V/M and FREQ/TS/IRC/MD gates
remain closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from maple.function.dispatcher.frequency.normal_modes import (
    analyze_cartesian_hessian,
    mass_weighted_basis_to_cartesian,
)
from .geometry_mediated_hessian import (
    AIMNET2_GEOMETRY_MEDIATED_HVP_CHARGE_TANGENT_TOLERANCE_E_PER_A,
    AIMNET2_GEOMETRY_MEDIATED_HVP_COMPONENT_LEDGER_TOLERANCE_EV_PER_A2,
    AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_CHARGE_PARITY_TOLERANCE_E,
    AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_ENERGY_PARITY_TOLERANCE_EV,
    AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_GRADIENT_PARITY_TOLERANCE_EV_PER_A,
    AIMNET2_GEOMETRY_MEDIATED_HVP_MODEL_VJP_PARITY_TOLERANCE_EV_PER_A,
    AIMNET2_GEOMETRY_MEDIATED_HVP_SOURCE_COTANGENT_TOLERANCE_EV_PER_E,
)
from .geometry_mediated_frequency_records import (
    finite as _finite,
    mapping as _mapping,
    maximum_absolute as _maximum_absolute,
    norm as _norm,
    parse_dense_hvp_record as _hvp_record,
    parse_frequency_center as _center_record,
    parse_frequency_fd_endpoint as _fd_endpoint,
    sequence as _sequence,
)
from .geometry_mediated_stationary import (
    summarize_aimnet2_geometry_mediated_stationary_water_search,
)

AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_CONTRACT_VERSION = (
    "route2-aimnet2-geometry-mediated-frequency-water-contract-v1"
)
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_SCHEMA_VERSION = (
    "route2-aimnet2-geometry-mediated-frequency-water-summary-v1"
)
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A = (
    8.0e-4,
    4.0e-4,
    2.0e-4,
    1.0e-4,
)
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_HESSIAN_SYMMETRY_TOLERANCE_EV_PER_A2 = 1.0e-8
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_FROBENIUS_TOLERANCE_EV_PER_A2 = 2.0e-4
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_MAX_ABS_TOLERANCE_EV_PER_A2 = 1.0e-4
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_CENTRAL_CONVERGENCE_MAX_RATIO = 0.35
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_TERMINAL_REFINEMENT_MAX_RATIO = 0.55
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_RIGID_HVP_TOLERANCE_EV_PER_A2 = 2.0e-7
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_MINIMUM_VIBRATIONAL_EIGENVALUE = 1.0e-4
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_MODE_ORTHONORMALITY_TOLERANCE = 1.0e-10
AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_EIGENPAIR_RESIDUAL_TOLERANCE = 1.0e-8

_NO_CAPABILITIES = {tier: False for tier in ("E", "F", "H", "V", "M")}
_NO_WORKFLOWS = {
    name: False for name in ("public_ase_hessian", "opt", "freq", "ts", "irc", "md")
}


def summarize_aimnet2_geometry_mediated_frequency_water(
    *,
    search_record: Mapping[str, object],
    center_record: Mapping[str, object],
    hessian_vector_records: Sequence[Mapping[str, object]],
    finite_difference_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute the stationary-water dense-Hessian diagnostic."""

    search = summarize_aimnet2_geometry_mediated_stationary_water_search(
        _mapping(search_record, name="search record")
    )
    solution_positions = np.asarray(search["result"]["positions_A"], dtype=float)
    center = _center_record(
        _mapping(center_record, name="center record"), positions=solution_positions
    )
    final_search = search["evaluations"][-1]
    search_center_energy_error = abs(
        float(final_search["energy_eV"]) - float(center["summary"]["total_energy_eV"])
    )
    search_center_gradient_error = _norm(
        np.asarray(final_search["total_gradient_eV_per_A"], dtype=float)
        - center["gradient"]
    )

    raw_hvps = _sequence(hessian_vector_records, name="dense HVP records")
    if len(raw_hvps) != 9:
        raise ValueError("dense water Hessian requires exactly nine Cartesian HVPs.")
    hessian_columns: list[np.ndarray] = []
    hvp_summaries: list[dict[str, object]] = []
    parity_records: list[dict[str, float]] = []
    for coordinate_index, raw in enumerate(raw_hvps):
        column, summary, parity = _hvp_record(
            _mapping(raw, name=f"HVP record {coordinate_index}"),
            coordinate_index=coordinate_index,
            center=center,
        )
        hessian_columns.append(column)
        hvp_summaries.append(summary)
        parity_records.append(parity)
    identity_fields = (
        "scalar_id",
        "profile_id",
        "scalar_fingerprint_sha256",
        "model_second_order_behavior_sha256",
        "continuum_second_order_behavior_sha256",
    )
    for field in identity_fields:
        if len({record[field] for record in hvp_summaries}) != 1:
            raise ValueError(f"dense HVP records disagree on {field}.")
    hessian = np.column_stack(hessian_columns)
    hessian_symmetry_error = _maximum_absolute(hessian - hessian.T)

    raw_fd = _sequence(
        finite_difference_records, name="dense Hessian finite differences"
    )
    if len(raw_fd) != len(AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A):
        raise ValueError("dense Hessian finite-difference step coverage changed.")
    fd_summaries: list[dict[str, object]] = []
    fd_frobenius_errors: list[float] = []
    fd_maximum_errors: list[float] = []
    all_fd_guards = True
    all_fd_topologies = True
    for raw_step, expected_step in zip(
        raw_fd, AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A, strict=True
    ):
        step_record = _mapping(raw_step, name="dense Hessian FD step")
        step = _finite(step_record.get("step_A"), name="dense Hessian FD step")
        if step != expected_step:
            raise ValueError("dense Hessian finite-difference steps changed.")
        axes = _sequence(step_record.get("axes"), name="dense Hessian FD axes")
        if len(axes) != 9:
            raise ValueError("dense Hessian FD step requires all nine axes.")
        columns: list[np.ndarray] = []
        guards: list[dict[str, object]] = []
        for coordinate_index, raw_axis in enumerate(axes):
            axis = _mapping(raw_axis, name="dense Hessian FD axis")
            if int(axis.get("coordinate_index")) != coordinate_index:
                raise ValueError("dense Hessian FD coordinate order changed.")
            direction = np.zeros((3, 3), dtype=float)
            direction.reshape(-1)[coordinate_index] = 1.0
            plus_gradient, plus_guard = _fd_endpoint(
                _mapping(axis.get("plus"), name="plus endpoint"),
                expected_positions=center["atoms"].positions + step * direction,
                center=center,
                name=f"+ axis {coordinate_index} at {step:g} A",
            )
            minus_gradient, minus_guard = _fd_endpoint(
                _mapping(axis.get("minus"), name="minus endpoint"),
                expected_positions=center["atoms"].positions - step * direction,
                center=center,
                name=f"- axis {coordinate_index} at {step:g} A",
            )
            columns.append((plus_gradient - minus_gradient) / (2.0 * step))
            guards.extend((plus_guard, minus_guard))
        finite_difference_hessian = np.column_stack(columns)
        difference = finite_difference_hessian - hessian
        frobenius_error = _norm(difference)
        maximum_error = _maximum_absolute(difference)
        fd_frobenius_errors.append(frobenius_error)
        fd_maximum_errors.append(maximum_error)
        all_fd_guards &= all(bool(guard["gate_passed"]) for guard in guards)
        all_fd_topologies &= all(
            bool(guard["same_model_topology"])
            and bool(guard["same_continuum_topology"])
            for guard in guards
        )
        fd_summaries.append(
            {
                "step_A": step,
                "HVP_FD_frobenius_error_eV_per_A2": frobenius_error,
                "HVP_FD_max_abs_error_eV_per_A2": maximum_error,
                "finite_difference_symmetry_max_abs_eV_per_A2": _maximum_absolute(
                    finite_difference_hessian - finite_difference_hessian.T
                ),
                "all_endpoint_guards_passed": all(
                    bool(guard["gate_passed"]) for guard in guards
                ),
            }
        )

    frobenius_ratios = [
        current / max(previous, np.finfo(float).tiny)
        for previous, current in zip(fd_frobenius_errors, fd_frobenius_errors[1:])
    ]
    maximum_ratios = [
        current / max(previous, np.finfo(float).tiny)
        for previous, current in zip(fd_maximum_errors, fd_maximum_errors[1:])
    ]

    analysis = analyze_cartesian_hessian(
        hessian,
        center["masses"],
        center["atoms"].positions,
        symmetry_tolerance_eV_per_A2=(
            AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_HESSIAN_SYMMETRY_TOLERANCE_EV_PER_A2
        ),
    )
    translations = mass_weighted_basis_to_cartesian(
        analysis.subspaces.translation_basis_mass_weighted,
        center["masses"],
        normalize_columns=True,
    )
    rotations = mass_weighted_basis_to_cartesian(
        analysis.subspaces.rotation_basis_mass_weighted,
        center["masses"],
        normalize_columns=True,
    )
    translation_residuals = np.linalg.norm(hessian @ translations, axis=0)
    rotation_residuals = np.linalg.norm(hessian @ rotations, axis=0)
    mode_orthonormality_error = _maximum_absolute(
        analysis.modes_mass_weighted.T @ analysis.modes_mass_weighted
        - np.eye(analysis.subspaces.vibrational_rank)
    )
    mode_residuals = np.linalg.norm(
        analysis.mass_weighted_hessian_eV_per_A2_amu @ analysis.modes_mass_weighted
        - analysis.modes_mass_weighted * analysis.eigenvalues_eV_per_A2_amu[None, :],
        axis=0,
    )

    parity_maxima = {
        name: max(record[name] for record in parity_records)
        for name in parity_records[0]
    }
    maximum_ledger_error = max(
        float(record["component_ledger_max_abs_eV_per_A2"]) for record in hvp_summaries
    )
    maximum_source_tangent_residual = max(
        float(record["source_tangent_residual_e_per_A"]) for record in hvp_summaries
    )
    maximum_source_cotangent_error = max(
        float(record["source_cotangent_max_abs_error_eV_per_e"])
        for record in hvp_summaries
    )
    quadratic_window_gate = all(
        ratio < AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_CENTRAL_CONVERGENCE_MAX_RATIO
        for ratio in (*frobenius_ratios[:-1], *maximum_ratios[:-1])
    )
    terminal_refinement_gate = all(
        ratio < AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_TERMINAL_REFINEMENT_MAX_RATIO
        for ratio in (frobenius_ratios[-1], maximum_ratios[-1])
    )
    gates = {
        "stationary_internal_search": bool(search["search_gates_passed"]),
        "search_center_replay": (
            search_center_energy_error <= 1.0e-8
            and search_center_gradient_error <= 1.0e-8
        ),
        "center_scalar_and_gradient_ledgers": (
            float(center["summary"]["gradient_ledger_max_abs_eV_per_A"]) <= 2.0e-10
        ),
        "center_pcm_stationarity": bool(
            center["summary"]["stationarity"]["gate_passed"]
        ),
        "center_reciprocity_metric_and_charge_gauge": bool(
            center["summary"]["reciprocity"]["gate_passed"]
        ),
        "center_deterministic_replay": bool(
            center["summary"]["deterministic_replay"]["gate_passed"]
        ),
        "center_event_guard": bool(center["summary"]["event_guard"]["gate_passed"]),
        "all_hvp_component_ledgers": (
            maximum_ledger_error
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_COMPONENT_LEDGER_TOLERANCE_EV_PER_A2
        ),
        "reaction_potential_is_hvp_source_cotangent": (
            maximum_source_cotangent_error
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_SOURCE_COTANGENT_TOLERANCE_EV_PER_E
        ),
        "fixed_total_charge_tangent": (
            maximum_source_tangent_residual
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_CHARGE_TANGENT_TOLERANCE_E_PER_A
            and parity_maxima["charge_tangent_residual_e_per_A"]
            <= AIMNET2_GEOMETRY_MEDIATED_HVP_CHARGE_TANGENT_TOLERANCE_E_PER_A
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
        "dense_hessian_symmetry": (
            hessian_symmetry_error
            <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_HESSIAN_SYMMETRY_TOLERANCE_EV_PER_A2
        ),
        "all_fd_stencils_same_stratum": all_fd_topologies,
        "all_fd_segments_event_guarded": all_fd_guards,
        "dense_hessian_central_second_order_window": quadratic_window_gate,
        "dense_hessian_terminal_refinement_no_growth": terminal_refinement_gate,
        "dense_hessian_total_gradient_finite_difference": (
            fd_frobenius_errors[-1]
            <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_FROBENIUS_TOLERANCE_EV_PER_A2
            and fd_maximum_errors[-1]
            <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_MAX_ABS_TOLERANCE_EV_PER_A2
        ),
        "three_translation_zero_modes": bool(
            np.all(
                translation_residuals
                <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_RIGID_HVP_TOLERANCE_EV_PER_A2
            )
        ),
        "three_rotation_zero_modes_at_stationary_point": bool(
            analysis.subspaces.rotation_rank == 3
            and np.all(
                rotation_residuals
                <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_RIGID_HVP_TOLERANCE_EV_PER_A2
            )
        ),
        "nonlinear_water_three_mode_vibrational_subspace": (
            not analysis.subspaces.is_linear
            and analysis.subspaces.rigid_rank == 6
            and analysis.subspaces.vibrational_rank == 3
        ),
        "positive_vibrational_hessian": bool(
            np.all(
                analysis.eigenvalues_eV_per_A2_amu
                > AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_MINIMUM_VIBRATIONAL_EIGENVALUE
            )
        ),
        "mass_weighted_mode_orthonormality": (
            mode_orthonormality_error
            <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_MODE_ORTHONORMALITY_TOLERANCE
        ),
        "mass_weighted_eigenpair_residuals": bool(
            np.all(
                mode_residuals
                <= AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_EIGENPAIR_RESIDUAL_TOLERANCE
            )
        ),
        "diagnostic_only_admission_flags": all(
            record["diagnostic_only"] is True and record["tier_h_admitted"] is False
            for record in hvp_summaries
        ),
    }
    return {
        "schema_version": AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_SCHEMA_VERSION,
        "water_only": True,
        "stationary_point_only": True,
        "fixed_graph_cavity_stratum_only": True,
        "conductor_reference_only": True,
        "finite_dielectric_parameterization": False,
        "search": search,
        "center": center["summary"],
        "search_center_replay": {
            "energy_absolute_error_eV": search_center_energy_error,
            "gradient_difference_norm_eV_per_A": search_center_gradient_error,
        },
        "hvp_records": hvp_summaries,
        "dense_hessian": {
            "shape": list(hessian.shape),
            "symmetry_max_abs_eV_per_A2": hessian_symmetry_error,
            "frobenius_norm_eV_per_A2": _norm(hessian),
        },
        "finite_difference_records": fd_summaries,
        "finite_difference_error_norms": {
            "frobenius_eV_per_A2": fd_frobenius_errors,
            "maximum_absolute_eV_per_A2": fd_maximum_errors,
        },
        "central_convergence_ratios": {
            "frobenius": frobenius_ratios,
            "maximum_absolute": maximum_ratios,
        },
        "rigid_modes": {
            "translation_HVP_norms_eV_per_A2": translation_residuals.tolist(),
            "rotation_HVP_norms_eV_per_A2": rotation_residuals.tolist(),
            "rotation_singular_values_sqrt_amu_A": (
                analysis.subspaces.rotation_singular_values_sqrt_amu_angstrom.tolist()
            ),
        },
        "vibrational_analysis": {
            "rigid_rank": analysis.subspaces.rigid_rank,
            "vibrational_rank": analysis.subspaces.vibrational_rank,
            "is_linear": analysis.subspaces.is_linear,
            "eigenvalues_eV_per_A2_amu": (analysis.eigenvalues_eV_per_A2_amu.tolist()),
            "frequencies_cm1": analysis.frequencies_cm1.tolist(),
            "mode_orthonormality_max_abs": mode_orthonormality_error,
            "eigenpair_residual_norms_eV_per_A2_amu": mode_residuals.tolist(),
        },
        "model_parity_maxima": parity_maxima,
        "maximum_component_ledger_error_eV_per_A2": maximum_ledger_error,
        "maximum_source_tangent_residual_e_per_A": (maximum_source_tangent_residual),
        "thresholds": {
            "hessian_symmetry_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_HESSIAN_SYMMETRY_TOLERANCE_EV_PER_A2
            ),
            "FD_frobenius_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_FROBENIUS_TOLERANCE_EV_PER_A2
            ),
            "FD_max_abs_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_MAX_ABS_TOLERANCE_EV_PER_A2
            ),
            "central_convergence_max_ratio": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_CENTRAL_CONVERGENCE_MAX_RATIO
            ),
            "terminal_refinement_max_ratio": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_TERMINAL_REFINEMENT_MAX_RATIO
            ),
            "rigid_HVP_eV_per_A2": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_RIGID_HVP_TOLERANCE_EV_PER_A2
            ),
            "minimum_vibrational_eigenvalue_eV_per_A2_amu": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_MINIMUM_VIBRATIONAL_EIGENVALUE
            ),
            "eigenpair_residual_eV_per_A2_amu": (
                AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_EIGENPAIR_RESIDUAL_TOLERANCE
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
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_CENTRAL_CONVERGENCE_MAX_RATIO",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_EIGENPAIR_RESIDUAL_TOLERANCE",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_FROBENIUS_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_MAX_ABS_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_FD_STEPS_A",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_HESSIAN_SYMMETRY_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_MINIMUM_VIBRATIONAL_EIGENVALUE",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_RIGID_HVP_TOLERANCE_EV_PER_A2",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_TERMINAL_REFINEMENT_MAX_RATIO",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_CONTRACT_VERSION",
    "AIMNET2_GEOMETRY_MEDIATED_FREQUENCY_WATER_SCHEMA_VERSION",
    "summarize_aimnet2_geometry_mediated_frequency_water",
]
