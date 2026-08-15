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


__all__ = ["synthetic_reciprocity_record"]
