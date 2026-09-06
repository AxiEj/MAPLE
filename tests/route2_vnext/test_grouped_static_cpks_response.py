from __future__ import annotations

import numpy as np

from tools.route2_release.run_vqm24_grouped_static_cpks_response import (
    assemble_grouped_observables,
)


def test_grouped_cpks_assembly_recovers_cross_group_source_response() -> None:
    generator = np.random.default_rng(20260827)
    mode_count = 12
    point_count = 25
    source_indices = np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    factor = generator.normal(size=(18, mode_count))
    source_response = -(factor.T @ factor)
    induced_mep = generator.normal(size=(mode_count, point_count))
    induced_mep[:, source_indices] = source_response
    induced_dipole = generator.normal(size=(mode_count, 3))
    curvature = np.diag(source_response).copy()

    result = assemble_grouped_observables(
        induced_mep_parts=[induced_mep[index : index + 4] for index in range(0, 12, 4)],
        induced_dipole_parts=[
            induced_dipole[index : index + 4] for index in range(0, 12, 4)
        ],
        curvature_parts=[curvature[index : index + 4] for index in range(0, 12, 4)],
        source_surface_indices=source_indices,
    )

    np.testing.assert_array_equal(result["induced_mep"], induced_mep)
    np.testing.assert_array_equal(result["induced_dipole"], induced_dipole)
    np.testing.assert_array_equal(result["source_response"], source_response)
    assert float(result["reciprocity"]) < 2.0e-16
    assert float(np.max(result["symmetric_eigenvalues"])) <= 1.0e-12
    assert np.linalg.norm(source_response[:4, 4:8]) > 0.0

