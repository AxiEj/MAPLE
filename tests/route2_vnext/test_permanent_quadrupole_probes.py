from __future__ import annotations

import numpy as np
import pytest
from ase.units import Bohr

from maple.solvation.reference.permanent_quadrupole_probes import (
    PERMANENT_P13_LEBEDEV_POINTS,
    build_permanent_p13_probe_partition,
)


def test_p13_permanent_probe_partitions_exceed_the_source_dimension() -> None:
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

    assert np.count_nonzero(probes.mask("fit")) > 13 * len(positions) - 1
    assert np.count_nonzero(probes.mask("audit")) > 13 * len(positions) - 1
    assert probes.candidate_count_per_partition == (
        len(positions) * PERMANENT_P13_LEBEDEV_POINTS
    )


def test_p13_permanent_probes_translate_with_the_molecule() -> None:
    pytest.importorskip("pyscf")
    positions = np.asarray([[0.0, 0.0, 0.0], [1.4, -0.2, 0.1]])
    radii = np.asarray([1.7, 1.5])
    translation = np.asarray([0.31, -0.42, 0.27])
    reference = build_permanent_p13_probe_partition(
        positions,
        radii,
        clearance_angstrom=1.0,
    )
    shifted = build_permanent_p13_probe_partition(
        positions + translation,
        radii,
        clearance_angstrom=1.0,
    )

    np.testing.assert_allclose(
        shifted.surface_points_bohr,
        reference.surface_points_bohr + translation / Bohr,
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_array_equal(
        shifted.partition_indices,
        reference.partition_indices,
    )
