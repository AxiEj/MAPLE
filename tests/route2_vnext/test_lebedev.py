from __future__ import annotations

import numpy as np
import pytest

from maple.solvation.surfaces.lebedev import (
    LEBEDEV_GRID_CONTRACT,
    ordered_lebedev_grid,
)

EXPECTED_HASHES = {
    50: "e693359e89a11d3ab1918fbaf68f2e1de0b71aa6faa08024ef42b453fd6f59da",
    194: "5c5d366f54a23faf3e1a4611fe02d432e51804bb09f5d779283a94ed8e19bf89",
    1202: "199f8eb5d87092a47398a25f335ecf6fbe6383cd6b254e18d4641ba2fad9e7a6",
}


@pytest.mark.parametrize("point_count", (50, 194, 1202))
def test_ordered_lebedev_rule_is_immutable_normalized_and_content_addressed(
    point_count: int,
) -> None:
    grid = ordered_lebedev_grid(point_count)
    assert grid.contract == LEBEDEV_GRID_CONTRACT
    assert grid.sha256 == EXPECTED_HASHES[point_count]
    assert not grid.directions.flags.writeable
    assert not grid.weights.flags.writeable
    assert not grid.directions.flags.owndata
    assert not grid.weights.flags.owndata
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        grid.directions.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        grid.weights.setflags(write=True)
    np.testing.assert_allclose(
        np.linalg.norm(grid.directions, axis=1), 1.0, rtol=0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(grid.weights.sum(), 4.0 * np.pi, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(
        grid.weights @ grid.directions, np.zeros(3), rtol=0.0, atol=2.0e-15
    )
    second = np.einsum("n,ni,nj->ij", grid.weights, grid.directions, grid.directions)
    np.testing.assert_allclose(second, np.eye(3) * 4.0 * np.pi / 3.0, atol=2e-14)


@pytest.mark.parametrize("point_count", (50, 194, 1202))
def test_ordered_nodes_match_pyddx_0p8_single_sphere(point_count: int) -> None:
    pyddx = pytest.importorskip("pyddx")
    if pyddx.__version__ != "0.8.0":
        pytest.skip("oracle is pinned to pyddx 0.8.0")
    model = pyddx.Model(
        "pcm",
        np.zeros((3, 1)),
        np.ones(1),
        78.39,
        eta=0.1,
        shift=0.0,
        lmax=1,
        n_lebedev=point_count,
        enable_fmm=False,
        enable_force=False,
    )
    np.testing.assert_array_equal(
        np.asarray(model.cavity).T, ordered_lebedev_grid(point_count).directions
    )


def test_unsupported_grid_fails_closed() -> None:
    with pytest.raises(ValueError, match="one of"):
        ordered_lebedev_grid(302)


def test_cached_grid_cannot_be_mutated_through_a_shared_reference() -> None:
    first = ordered_lebedev_grid(50)
    second = ordered_lebedev_grid(50)
    assert first is second
    with pytest.raises(ValueError):
        first.directions[0, 0] = 123.0
    np.testing.assert_array_equal(first.directions, second.directions)
