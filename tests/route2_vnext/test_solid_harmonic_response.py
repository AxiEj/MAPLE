from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from maple.solvation.continuum.harmonic_torch_primitives import (
    _torch_real_harmonic_design,
)
from maple.solvation.continuum.solid_harmonic_response import (
    CONTRACTED_HESSIAN_TABLE_VERSION,
    DERIVATIVE_TABLE_VERSION,
    contracted_hessian_table_nbytes,
    contracted_hessian_table_sha256,
    derivative_table_sha256,
    solid_harmonic_contracted_hessian,
    solid_harmonic_jets,
)


def _oracle_values(vector: torch.Tensor, *, lmax: int, kind: str) -> torch.Tensor:
    radius = torch.linalg.vector_norm(vector)
    angular = _torch_real_harmonic_design((vector / radius)[None, :], lmax=lmax)[0]
    blocks = []
    for ell in range(lmax + 1):
        power = ell if kind == "regular" else -ell - 1
        blocks.append(angular[ell * ell : (ell + 1) ** 2] * radius**power)
    return torch.cat(blocks)


def _oracle_jets(vector: torch.Tensor, *, lmax: int, kind: str):
    function = lambda point: _oracle_values(point, lmax=lmax, kind=kind)
    values = function(vector)
    gradient = torch.func.jacfwd(function)(vector)
    hessian = torch.func.jacfwd(torch.func.jacfwd(function))(vector)
    return values, gradient, hessian


def _assert_old_oracle_hessian_close(
    actual: torch.Tensor, expected: torch.Tensor
) -> None:
    # The old angular implementation uses complex ``pow``.  At Cartesian axes
    # it leaves cancellation noise in analytically zero high-degree entries.
    # Compare nonzeros normally and independently require those structural
    # zeros to be exact in the new Cartesian recurrence.
    oracle_noise = expected.abs() < 2.0e-11
    if torch.any(oracle_noise):
        assert torch.max(actual[oracle_noise].abs()).item() < 5.0e-14
    torch.testing.assert_close(
        actual[~oracle_noise], expected[~oracle_noise], atol=2.0e-12, rtol=2.0e-12
    )


def _series_coefficients(lmax: int, *, device: torch.device | str = "cpu"):
    index = torch.arange((lmax + 1) ** 2, dtype=torch.float64, device=device)
    return torch.cos(0.37 * index) / (index + 1.0)


def _assert_contracted_matches_both_oracles(
    point: torch.Tensor, lmax: int, kind: str
) -> None:
    coefficients = _series_coefficients(lmax, device=point.device)
    actual = solid_harmonic_contracted_hessian(point, lmax, kind, coefficients)
    full = solid_harmonic_jets(point, lmax, kind, derivative_order=2)
    expected_full = torch.einsum("pab,p->ab", full.hessian, coefficients)
    expected_original = torch.func.jacfwd(
        torch.func.jacfwd(
            lambda vector: torch.dot(
                _oracle_values(vector, lmax=lmax, kind=kind), coefficients
            )
        )
    )(point)
    torch.testing.assert_close(actual.basis_values, full.values, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual.basis_gradient, full.gradient, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        actual.contracted_hessian, expected_full, atol=2.0e-12, rtol=2.0e-12
    )
    torch.testing.assert_close(
        actual.contracted_hessian,
        expected_original,
        atol=2.0e-12,
        rtol=2.0e-12,
    )


@pytest.mark.parametrize("kind", ("regular", "irregular"))
def test_low_degree_values_and_regular_closed_form(kind: str):
    vector = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)
    actual = solid_harmonic_jets(vector, 2, kind, derivative_order=2)
    expected = _oracle_jets(vector, lmax=2, kind=kind)
    torch.testing.assert_close(actual.values, expected[0], atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(actual.gradient, expected[1], atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(actual.hessian, expected[2], atol=2.0e-12, rtol=2.0e-12)

    if kind == "regular":
        factor = math.sqrt(3.0 / (4.0 * math.pi))
        closed = torch.tensor(
            [1.0 / math.sqrt(4.0 * math.pi), -0.7 * factor, 1.1 * factor, 0.4 * factor],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual.values[:4], closed, atol=2.0e-15, rtol=0.0)
        assert torch.count_nonzero(actual.gradient[0]) == 0
        assert torch.count_nonzero(actual.hessian[:4]) == 0


@pytest.mark.parametrize("kind", ("regular", "irregular"))
@pytest.mark.parametrize(
    "vector",
    (
        (0.0, 0.0, 0.7),
        (0.0, -1.3, 0.0),
        (1.9, 0.0, 0.0),
        (0.31, -0.82, 1.47),
        (-1.12, 0.63, -0.41),
    ),
)
def test_all_degrees_through_fifteen_match_autograd(kind: str, vector):
    point = torch.tensor(vector, dtype=torch.float64)
    actual = solid_harmonic_jets(point, 15, kind, derivative_order=2)
    expected = _oracle_jets(point, lmax=15, kind=kind)
    torch.testing.assert_close(actual.values, expected[0], atol=2.0e-12, rtol=2.0e-12)
    torch.testing.assert_close(actual.gradient, expected[1], atol=2.0e-12, rtol=2.0e-12)
    _assert_old_oracle_hessian_close(actual.hessian, expected[2])
    torch.testing.assert_close(
        actual.hessian, actual.hessian.transpose(-1, -2), atol=5.0e-14, rtol=0.0
    )


def test_batch_shape_derivative_orders_and_radius_scaling():
    vectors = torch.tensor(
        [[[0.7, -0.2, 0.5], [1.1, 0.4, -0.8]], [[-0.3, 1.2, 0.6], [0.9, -1.0, 0.2]]],
        dtype=torch.float64,
    )
    order0 = solid_harmonic_jets(vectors, 4, "regular", derivative_order=0)
    order1 = solid_harmonic_jets(vectors, 4, "regular", derivative_order=1)
    order2 = solid_harmonic_jets(vectors, 4, "regular", derivative_order=2)
    assert order0.values.shape == (2, 2, 25)
    assert order0.gradient is None and order0.hessian is None
    assert order1.gradient.shape == (2, 2, 25, 3) and order1.hessian is None
    torch.testing.assert_close(order0.values, order2.values)
    torch.testing.assert_close(order1.gradient, order2.gradient)

    radius = 1.73
    delta = torch.tensor([0.8, -0.5, 1.2], dtype=torch.float64)
    scaled = solid_harmonic_jets(delta / radius, 5, "irregular", derivative_order=2)
    function = lambda x: _oracle_values(x / radius, lmax=5, kind="irregular")
    torch.testing.assert_close(
        scaled.gradient / radius,
        torch.func.jacfwd(function)(delta),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        scaled.hessian / radius**2,
        torch.func.jacfwd(torch.func.jacfwd(function))(delta),
        atol=2e-12,
        rtol=2e-12,
    )


def test_near_cancellation_derivative_is_not_thresholded_to_zero():
    point = torch.tensor([1.9, 1.0e-7, 0.0], dtype=torch.float64)
    actual = solid_harmonic_jets(point, 9, "regular", derivative_order=2)
    expected = _oracle_jets(point, lmax=9, kind="regular")[2][93, 1, 1]
    value = actual.hessian[93, 1, 1]
    assert abs(value.item()) > 5.0e-12
    torch.testing.assert_close(value, expected, atol=2.0e-13, rtol=2.0e-12)


@pytest.mark.parametrize("kind", ("regular", "irregular"))
@pytest.mark.parametrize("lmax", range(16))
def test_contracted_hessian_all_degrees_matches_full_jets_and_original_basis_autograd(
    kind: str, lmax: int
):
    point = torch.tensor([0.31, -0.82, 1.47], dtype=torch.float64)
    _assert_contracted_matches_both_oracles(point, lmax, kind)


@pytest.mark.parametrize("kind", ("regular", "irregular"))
@pytest.mark.parametrize(
    "vector",
    (
        (0.0, 0.0, 0.7),
        (0.31, -0.22, 0.41),
        (3.2, -2.7, 4.1),
        (1.0e-7, -0.63, 0.41),
        (1.9, 1.0e-7, 0.0),
    ),
    ids=("axis", "near", "far", "small-nonzero", "cancellation"),
)
def test_contracted_hessian_edge_cases_match_both_oracles(kind: str, vector):
    point = torch.tensor(vector, dtype=torch.float64)
    _assert_contracted_matches_both_oracles(point, 15, kind)


def test_contracted_hessian_preserves_legitimate_small_cancellation_value():
    point = torch.tensor([1.9, 1.0e-7, 0.0], dtype=torch.float64)
    coefficients = torch.zeros(100, dtype=torch.float64)
    coefficients[93] = 1.0
    actual = solid_harmonic_contracted_hessian(
        point, 9, "regular", coefficients
    ).contracted_hessian[1, 1]
    assert abs(actual.item()) > 5.0e-12
    expected = solid_harmonic_jets(point, 9, "regular", derivative_order=2).hessian[
        93, 1, 1
    ]
    torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-12)


def test_tables_are_versioned_hash_bound_and_inputs_are_validated():
    assert DERIVATIVE_TABLE_VERSION
    assert CONTRACTED_HESSIAN_TABLE_VERSION
    hashes = {
        derivative_table_sha256(15, "regular"),
        derivative_table_sha256(15, "irregular"),
    }
    assert all(len(value) == 64 for value in hashes)
    assert len(hashes) == 2
    assert derivative_table_sha256(15, "regular") == derivative_table_sha256(
        15, "regular"
    )
    contracted_hashes = {
        contracted_hessian_table_sha256(15, "regular"),
        contracted_hessian_table_sha256(15, "irregular"),
    }
    assert all(len(value) == 64 for value in contracted_hashes)
    assert len(contracted_hashes) == 2

    vector = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="lmax"):
        solid_harmonic_jets(vector, 16, "regular")
    with pytest.raises(ValueError, match="derivative_order"):
        solid_harmonic_jets(vector, 2, "regular", derivative_order=3)
    with pytest.raises(ValueError, match="kind"):
        solid_harmonic_jets(vector, 2, "other")
    with pytest.raises(ValueError, match="float64"):
        solid_harmonic_jets(vector.float(), 2, "regular")
    with pytest.raises(ValueError, match="nonzero"):
        solid_harmonic_jets(torch.zeros(3, dtype=torch.float64), 2, "irregular")

    coefficients = torch.ones(9, dtype=torch.float64)
    with pytest.raises(TypeError, match="coefficients"):
        solid_harmonic_contracted_hessian(vector, 2, "regular", [1.0] * 9)
    with pytest.raises(ValueError, match="shape"):
        solid_harmonic_contracted_hessian(vector, 2, "regular", coefficients[:-1])
    with pytest.raises(ValueError, match="float64"):
        solid_harmonic_contracted_hessian(vector, 2, "regular", coefficients.float())
    nonfinite = coefficients.clone()
    nonfinite[4] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        solid_harmonic_contracted_hessian(vector, 2, "regular", nonfinite)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ("regular", "irregular"))
def test_cuda_matches_cpu(kind: str):
    vectors = torch.tensor(
        [[0.7, -0.2, 0.5], [1.1, 0.4, -0.8], [-0.3, 1.2, 0.6]],
        dtype=torch.float64,
    )
    cpu = solid_harmonic_jets(vectors, 15, kind, derivative_order=2)
    gpu = solid_harmonic_jets(vectors.cuda(), 15, kind, derivative_order=2)
    torch.testing.assert_close(gpu.values.cpu(), cpu.values, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(gpu.gradient.cpu(), cpu.gradient, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(gpu.hessian.cpu(), cpu.hessian, atol=2e-12, rtol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ("regular", "irregular"))
def test_contracted_hessian_cuda_matches_cpu_full_jets_and_original_basis(kind: str):
    vectors = torch.tensor(
        [
            [0.0, 0.0, 0.7],
            [0.31, -0.22, 0.41],
            [3.2, -2.7, 4.1],
            [1.0e-7, -0.63, 0.41],
            [1.9, 1.0e-7, 0.0],
        ],
        dtype=torch.float64,
    )
    coefficients = _series_coefficients(15)
    cpu = solid_harmonic_contracted_hessian(vectors, 15, kind, coefficients)
    gpu = solid_harmonic_contracted_hessian(
        vectors.cuda(), 15, kind, coefficients.cuda()
    )
    full_gpu = solid_harmonic_jets(vectors.cuda(), 15, kind, derivative_order=2)
    torch.testing.assert_close(
        gpu.basis_values.cpu(), cpu.basis_values, atol=2.0e-12, rtol=2.0e-12
    )
    torch.testing.assert_close(
        gpu.basis_gradient.cpu(), cpu.basis_gradient, atol=2.0e-12, rtol=2.0e-12
    )
    torch.testing.assert_close(
        gpu.contracted_hessian.cpu(),
        cpu.contracted_hessian,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    torch.testing.assert_close(
        gpu.contracted_hessian,
        torch.einsum("gpab,p->gab", full_gpu.hessian, coefficients.cuda()),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    point = vectors[3].cuda()
    expected_original = torch.func.jacfwd(
        torch.func.jacfwd(
            lambda vector: torch.dot(
                _oracle_values(vector, lmax=15, kind=kind), coefficients.cuda()
            )
        )
    )(point)
    torch.testing.assert_close(
        solid_harmonic_contracted_hessian(
            point, 15, kind, coefficients.cuda()
        ).contracted_hessian,
        expected_original,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    with pytest.raises(ValueError, match="same device"):
        solid_harmonic_contracted_hessian(point, 15, kind, coefficients)


def test_cached_sparse_tables_are_bytes_backed_immutable():
    from maple.solvation.continuum.solid_harmonic_response import _derivative_tables

    tables = _derivative_tables(3, "irregular")
    for rows in (
        *tables.gradient,
        *(x for row in tables.hessian for x in row),
        *(x for row in tables.hessian_transpose for x in row),
    ):
        for array in (rows.indices, rows.coefficients):
            with pytest.raises(ValueError):
                array.setflags(write=True)


@pytest.mark.parametrize("kind", ("regular", "irregular"))
def test_contracted_transpose_tables_have_reproducible_hash_and_exact_storage(kind):
    from maple.solvation.continuum.solid_harmonic_response import _derivative_tables

    tables = _derivative_tables(15, kind)
    transpose_rows = tuple(rows for axis in tables.hessian_transpose for rows in axis)
    exact_nbytes = sum(
        rows.indices.nbytes + rows.coefficients.nbytes for rows in transpose_rows
    )
    assert contracted_hessian_table_nbytes(15, kind) == exact_nbytes
    assert exact_nbytes == {"regular": 77_824, "irregular": 98_496}[kind]
    digest = contracted_hessian_table_sha256(15, kind)
    assert (
        digest
        == {
            "regular": "5c962080e3b81138db758c3651008deaa71be48901b8946f369a9f6a62fbf803",
            "irregular": "98079461b64e5bb036d1b8cc152184ad552cb14215e9286d361e3da10c9e9215",
        }[kind]
    )
    assert digest == contracted_hessian_table_sha256(15, kind)
