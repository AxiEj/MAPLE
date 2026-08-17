"""Shared synthetic raw records for geometry-mediated release-audit tests."""

from __future__ import annotations

from maple.solvation.coupling.geometry_mediated import (
    GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E,
    GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE,
    GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E,
    GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A,
    GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV,
    GEOMETRY_MEDIATED_RECIPROCITY_PROBES,
    GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE,
    GEOMETRY_MEDIATED_RECIPROCITY_SEED,
)


def synthetic_reciprocity_record() -> dict[str, object]:
    """Return a complete zero-error record under the production thresholds."""

    bilinear = []
    charge = []
    for probe in range(GEOMETRY_MEDIATED_RECIPROCITY_PROBES):
        value = 0.25 * (probe + 1)
        bilinear.append(
            {
                "probe_index": probe,
                "left_P_right_eV": value,
                "right_P_left_eV": value,
                "apply_adjoint_eV": value,
                "reciprocity_absolute_error_eV": 0.0,
                "reciprocity_relative_error": 0.0,
                "apply_adjoint_absolute_error_eV": 0.0,
                "apply_adjoint_relative_error": 0.0,
            }
        )
        for step in GEOMETRY_MEDIATED_CHARGE_FD_STEPS_E:
            charge.append(
                {
                    "probe_index": probe,
                    "step_e": step,
                    "analytic_eV_per_e": value,
                    "finite_difference_eV_per_e": value,
                    "absolute_error_eV_per_e": 0.0,
                    "relative_error": 0.0,
                }
            )
    return {
        "seed": GEOMETRY_MEDIATED_RECIPROCITY_SEED,
        "requested_probe_count": GEOMETRY_MEDIATED_RECIPROCITY_PROBES,
        "effective_probe_count": GEOMETRY_MEDIATED_RECIPROCITY_PROBES,
        "bilinear_records": bilinear,
        "charge_directional_fd_records": charge,
        "source_gradient_half_error_eV_per_source_unit": 0.0,
        "charge_gauge_vjp_norm_eV_per_A": 0.0,
        "maximum_reciprocity_absolute_error_eV": 0.0,
        "maximum_reciprocity_relative_error": 0.0,
        "maximum_apply_adjoint_absolute_error_eV": 0.0,
        "maximum_apply_adjoint_relative_error": 0.0,
        "maximum_charge_fd_absolute_error_eV_per_e": 0.0,
        "maximum_charge_fd_relative_error": 0.0,
        "thresholds": {
            "reciprocity_absolute_eV": (
                GEOMETRY_MEDIATED_RECIPROCITY_ABSOLUTE_TOLERANCE_EV
            ),
            "reciprocity_relative": (GEOMETRY_MEDIATED_RECIPROCITY_RELATIVE_TOLERANCE),
            "charge_fd_absolute_eV_per_e": (
                GEOMETRY_MEDIATED_CHARGE_FD_ABSOLUTE_TOLERANCE_EV_PER_E
            ),
            "charge_fd_relative": GEOMETRY_MEDIATED_CHARGE_FD_RELATIVE_TOLERANCE,
            "charge_gauge_vjp_norm_eV_per_A": (
                GEOMETRY_MEDIATED_GAUGE_VJP_TOLERANCE_EV_PER_A
            ),
        },
        "gate_passed": True,
    }


def synthetic_ddpcm_stationarity_record() -> dict[str, object]:
    """Return a zero-residual finite-dielectric primal/adjoint/KKT audit."""

    residual_units = {
        "vacuum_projection_primal": "eV/e",
        "dielectric_primal": "eV/e",
        "single_layer_primal": "eV/e",
        "single_layer_adjoint": "eV/e",
        "dielectric_adjoint": "e",
        "vacuum_projection_adjoint": "e",
    }
    return {
        "state_dimension_per_block": 12,
        "primal_block_count": 3,
        "adjoint_block_count": 3,
        "dielectric": 78.355,
        "finite_dielectric_parameterization": True,
        "residuals": {
            name: {
                "absolute": 0.0,
                "right_hand_side_norm": 0.0,
                "relative": 0.0,
                "scaled": 0.0,
                "unit": unit,
            }
            for name, unit in residual_units.items()
        },
        "condition_numbers": {"mass": 1.0, "surface": 1.0, "dielectric": 1.0},
        "response_operator_audit": {
            "pairing_metric_id": "synthetic-pairing",
            "operator_representation": "synthetic point-monopole block",
            "primal_response_used_as_provider_field": False,
            "primal_response_is_energy_cotangent": True,
            "primal_response_relative_asymmetry": 0.0,
            "primal_charge_tangent_relative_asymmetry": 0.0,
            "energy_cotangent_relative_asymmetry": 0.0,
            "energy_cotangent_charge_tangent_relative_asymmetry": 0.0,
            "energy_cotangent_vs_symmetric_primal_relative_error": 0.0,
            "kkt_vs_autograd_absolute_source_covector_norm": 0.0,
            "kkt_vs_autograd_relative_error": 0.0,
            "primal_vs_energy_cotangent_absolute_source_covector_norm": 0.0,
            "primal_vs_energy_cotangent_relative_error": 0.0,
            "scalar_energy_eV": 0.0,
            "half_energy_cotangent_pairing_eV": 0.0,
            "half_coupling_absolute_error_eV": 0.0,
            "threshold": 1.0e-10,
            "gate_passed": True,
        },
        "thresholds": {
            "absolute_residual": 1.0e-10,
            "relative_residual": 1.0e-10,
            "max_rhs_or_one_scaled_residual": 1.0e-10,
            "condition_number": 1.0e12,
            "energy_cotangent_closure": 1.0e-10,
        },
        "gate_passed": True,
        "capability_admitted": False,
    }


__all__ = [
    "synthetic_ddpcm_stationarity_record",
    "synthetic_reciprocity_record",
]
