from __future__ import annotations

import numpy as np
import pytest
from ase.units import Hartree
from scipy.linalg import null_space

from maple.solvation.coupling.exact_gto import (
    FixedSurfaceGeometry,
    MACEPolarRadialGTOCoupling,
)
from maple.solvation.coupling.spaces import MACE_POLAR_RADIAL_GTO_SOURCE_SPACE
from maple.solvation.reference.exterior_probe_partition import (
    build_dense_exterior_probe_partition,
)
from maple.solvation.reference.radial_gto_observable_span import (
    evaluate_radial_gto_observable_span,
)


def test_radial_gto_observable_span_recovers_an_exact_passive_source_family() -> None:
    pytest.importorskip("pyscf")
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
    )
    radii = np.asarray([1.52, 1.20, 1.20])
    probes = build_dense_exterior_probe_partition(
        positions,
        radii,
        clearance_angstrom=1.0,
    )
    fit_indices = np.flatnonzero(probes.mask("fit"))
    source_indices = fit_indices[[0, len(fit_indices) // 3, 2 * len(fit_indices) // 3, -1]]
    operator = MACEPolarRadialGTOCoupling().surface_operator(
        FixedSurfaceGeometry(positions, probes.surface_points_bohr)
    ) / Hartree
    charge = np.tile(
        MACE_POLAR_RADIAL_GTO_SOURCE_SPACE.effective_charge_weights,
        len(positions),
    )
    tangent = null_space(charge[None, :])
    generator = np.random.default_rng(20260827)
    zero_source = tangent @ generator.normal(size=tangent.shape[1])
    source_operator_reduced = operator[source_indices] @ tangent
    response_sources = tangent @ (-source_operator_reduced.T)
    zero_mep = operator @ zero_source
    response_mep = (operator @ response_sources).T

    result = evaluate_radial_gto_observable_span(
        atom_positions_angstrom=positions,
        surface_points_bohr=probes.surface_points_bohr,
        quadrature_weights=probes.quadrature_weights,
        partition_indices=probes.partition_indices,
        source_surface_indices=source_indices,
        zero_total_surface_mep_hartree_per_e=zero_mep,
        induced_surface_mep_hartree_per_e_per_source_e=response_mep,
        total_charge_e=0.0,
    )

    assert result.weighted_fit_operator_rank == result.reduced_dimension
    assert result.zero_mep_fit_relative < 2.0e-11
    assert result.zero_mep_audit_relative < 2.0e-9
    assert result.response_mep_fit_relative < 2.0e-11
    assert result.response_mep_audit_relative < 2.0e-9
    assert result.maximum_charge_constraint_residual_e < 2.0e-11
    assert result.source_response_relative_error < 2.0e-11
    assert result.source_response_reciprocity_relative < 2.0e-11
    assert max(result.source_response_symmetric_eigenvalues_hartree_per_e2) < 0.0
    assert result.response_direction_rank == 4
    assert result.positive_hessian_extension_minimum_gram_eigenvalue > 0.0
    assert result.coefficient_label_emitted is False
    assert result.model_fit_performed is False
