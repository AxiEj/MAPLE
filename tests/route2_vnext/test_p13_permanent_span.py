from __future__ import annotations

import numpy as np
import pytest
from ase.units import Hartree
from scipy.linalg import null_space

from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.point_quadrupole import (
    point_traceless_quadrupole_surface_operator,
)
from maple.solvation.coupling.spaces import MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
from maple.solvation.reference.p13_permanent_span import (
    evaluate_p13_permanent_span,
)
from maple.solvation.reference.permanent_quadrupole_probes import (
    build_permanent_p13_probe_partition,
)


def test_p13_permanent_span_recovers_an_exact_charge_constrained_source() -> None:
    pytest.importorskip("pyscf")
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
    )
    radii = np.asarray([1.52, 1.20, 1.20])
    probes = build_permanent_p13_probe_partition(
        positions,
        radii,
        clearance_angstrom=1.0,
    )
    operator = np.concatenate(
        (
            MACEPolarRadialGTOCoupling().surface_operator(
                FixedSurfaceGeometry(positions, probes.surface_points_bohr)
            )
            / Hartree,
            point_traceless_quadrupole_surface_operator(
                points_bohr=probes.surface_points_bohr,
                centers_angstrom=positions,
            ),
        ),
        axis=1,
    )
    charge = np.concatenate(
        (
            np.tile(
                MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.effective_charge_weights,
                len(positions),
            ),
            np.zeros(5 * len(positions)),
        )
    )
    tangent = null_space(charge[None, :])
    source = tangent @ np.random.default_rng(17).normal(size=tangent.shape[1])
    target = operator @ source

    result = evaluate_p13_permanent_span(
        atom_positions_angstrom=positions,
        surface_points_bohr=probes.surface_points_bohr,
        quadrature_weights=probes.quadrature_weights,
        partition_indices=probes.partition_indices,
        total_surface_mep_hartree_per_e=target,
        total_charge_e=0.0,
    )

    assert result.source_dimension == 13 * len(positions)
    assert result.reduced_dimension == 13 * len(positions) - 1
    assert result.weighted_fit_operator_rank == result.reduced_dimension
    assert result.fit_relative_mep_error < 5.0e-12
    assert result.audit_relative_mep_error < 2.0e-10
    assert result.charge_constraint_residual_e < 2.0e-11
    assert result.coefficient_label_emitted is False
    assert result.model_fit_performed is False
