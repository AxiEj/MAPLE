"""Exact analytic Cartesian jets of MAPLE's real solid harmonics.

The derivative tables are generated from normalized complex-basis lowering
and raising identities, then converted once to MAPLE's signed-order real
basis.  Runtime evaluation is gather-and-sum only: it does not differentiate
coordinates with autograd and does not use scatter reductions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np

SolidHarmonicKind = Literal["regular", "irregular"]

DERIVATIVE_TABLE_VERSION = "maple-real-solid-harmonic-shifts-v2-exact"
CONTRACTED_HESSIAN_TABLE_VERSION = (
    "maple-real-solid-harmonic-contracted-hessian-transpose-v1-exact"
)
MAXIMUM_PHYSICAL_LMAX = 15
MAXIMUM_INTERNAL_LMAX = MAXIMUM_PHYSICAL_LMAX + 2


@dataclass(frozen=True)
class SolidHarmonicJets:
    """Values and requested Cartesian derivatives in signed-order layout."""

    values: Any
    gradient: Any | None
    hessian: Any | None


@dataclass(frozen=True)
class SolidHarmonicContractedHessian:
    """Basis values/gradients and one contracted scalar-series Hessian."""

    basis_values: Any
    basis_gradient: Any
    contracted_hessian: Any


@dataclass(frozen=True)
class _SparseRows:
    indices: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class _DerivativeTables:
    gradient: tuple[_SparseRows, _SparseRows, _SparseRows]
    hessian: tuple[tuple[_SparseRows, ...], ...]
    hessian_transpose: tuple[tuple[_SparseRows, ...], ...]


def _normalization(ell: int, order: int) -> float:
    return math.sqrt(
        (2 * ell + 1)
        * math.exp(math.lgamma(ell - order + 1) - math.lgamma(ell + order + 1))
        / (4.0 * math.pi)
    )


_GaussianRational = tuple[Fraction, Fraction]
_ZERO_GAUSSIAN = (Fraction(0), Fraction(0))


def _gaussian_add(
    left: _GaussianRational, right: _GaussianRational
) -> _GaussianRational:
    return left[0] + right[0], left[1] + right[1]


def _gaussian_multiply(
    left: _GaussianRational, right: _GaussianRational
) -> _GaussianRational:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _unnormalized_derivative_terms(
    ell: int, order: int, kind: SolidHarmonicKind, axis: int
) -> tuple[int, dict[int, _GaussianRational]]:
    """One exact Cartesian derivative of an unnormalized complex harmonic."""

    if kind == "regular":
        if ell == 0:
            return -1, {}
        neighbour = ell - 1
        z_coefficient = ell + order
        plus_coefficient = 1
        minus_coefficient = -(ell + order) * (ell + order - 1)
    else:
        neighbour = ell + 1
        z_coefficient = -(ell - order + 1)
        plus_coefficient = 1
        minus_coefficient = -(ell - order + 1) * (ell - order + 2)
    if axis == 0:
        candidates = (
            (order + 1, (Fraction(plus_coefficient, 2), Fraction(0))),
            (order - 1, (Fraction(minus_coefficient, 2), Fraction(0))),
        )
    elif axis == 1:
        candidates = (
            (order + 1, (Fraction(0), Fraction(-plus_coefficient, 2))),
            (order - 1, (Fraction(0), Fraction(minus_coefficient, 2))),
        )
    else:
        candidates = ((order, (Fraction(z_coefficient), Fraction(0))),)
    return neighbour, {
        target_order: coefficient
        for target_order, coefficient in candidates
        if coefficient != _ZERO_GAUSSIAN and abs(target_order) <= neighbour
    }


def _composed_unnormalized_terms(
    ell: int,
    order: int,
    kind: SolidHarmonicKind,
    axes: tuple[int, ...],
) -> tuple[int, dict[int, _GaussianRational]]:
    degree = ell
    terms = {order: (Fraction(1), Fraction(0))}
    for axis in axes:
        accumulated: dict[int, _GaussianRational] = {}
        next_degree = degree - 1 if kind == "regular" else degree + 1
        for source_order, source_coefficient in terms.items():
            target_degree, shifted = _unnormalized_derivative_terms(
                degree, source_order, kind, axis
            )
            next_degree = target_degree
            for target_order, shift_coefficient in shifted.items():
                product = _gaussian_multiply(source_coefficient, shift_coefficient)
                accumulated[target_order] = _gaussian_add(
                    accumulated.get(target_order, _ZERO_GAUSSIAN), product
                )
        degree = next_degree
        terms = {
            target_order: coefficient
            for target_order, coefficient in accumulated.items()
            if coefficient != _ZERO_GAUSSIAN
        }
        if not terms:
            break
    return degree, terms


def _normalization_ratio_squared(
    ell: int, order: int, target_ell: int, target_order: int
) -> Fraction:
    return Fraction(
        (2 * ell + 1)
        * math.factorial(ell - order)
        * math.factorial(target_ell + target_order),
        (2 * target_ell + 1)
        * math.factorial(ell + order)
        * math.factorial(target_ell - target_order),
    )


def _squarefree_integer(value: int) -> tuple[int, int]:
    outside = 1
    inside = 1
    remainder = value
    divisor = 2
    while divisor * divisor <= remainder:
        exponent = 0
        while remainder % divisor == 0:
            remainder //= divisor
            exponent += 1
        outside *= divisor ** (exponent // 2)
        if exponent % 2:
            inside *= divisor
        divisor += 1
    if remainder > 1:
        inside *= remainder
    return outside, inside


def _canonical_square_root(value: Fraction) -> tuple[Fraction, Fraction]:
    numerator_outside, numerator_inside = _squarefree_integer(value.numerator)
    denominator_outside, denominator_inside = _squarefree_integer(value.denominator)
    return (
        Fraction(numerator_outside, denominator_outside),
        Fraction(numerator_inside, denominator_inside),
    )


def _complex_outputs_for_real(
    signed_order: int,
) -> tuple[tuple[int, _GaussianRational, Fraction], ...]:
    if signed_order == 0:
        return ((0, (Fraction(1), Fraction(0)), Fraction(1)),)
    order = abs(signed_order)
    phase = (-1) ** order
    if signed_order > 0:
        return (
            (order, (Fraction(phase), Fraction(0)), Fraction(1, 2)),
            (-order, (Fraction(1), Fraction(0)), Fraction(1, 2)),
        )
    return (
        (order, (Fraction(0), Fraction(-phase)), Fraction(1, 2)),
        (-order, (Fraction(0), Fraction(1)), Fraction(1, 2)),
    )


def _real_inputs_for_complex(
    signed_order: int,
) -> tuple[tuple[int, _GaussianRational, Fraction], ...]:
    if signed_order == 0:
        return ((0, (Fraction(1), Fraction(0)), Fraction(1)),)
    order = abs(signed_order)
    if signed_order > 0:
        phase = (-1) ** order
        return (
            (order, (Fraction(phase), Fraction(0)), Fraction(1, 2)),
            (-order, (Fraction(0), Fraction(phase)), Fraction(1, 2)),
        )
    return (
        (order, (Fraction(1), Fraction(0)), Fraction(1, 2)),
        (-order, (Fraction(0), Fraction(-1)), Fraction(1, 2)),
    )


def _real_derivative_block(
    ell: int, kind: SolidHarmonicKind, axes: tuple[int, ...]
) -> tuple[int, np.ndarray]:
    target_ell = ell + len(axes) * (1 if kind == "irregular" else -1)
    if target_ell < 0:
        return target_ell, np.zeros((2 * ell + 1, 0), dtype=float)
    symbolic: dict[tuple[int, int, Fraction], _GaussianRational] = {}
    for real_output in range(-ell, ell + 1):
        for (
            complex_output,
            output_coefficient,
            output_radicand,
        ) in _complex_outputs_for_real(real_output):
            _, derivative_terms = _composed_unnormalized_terms(
                ell, complex_output, kind, axes
            )
            for complex_input, derivative_coefficient in derivative_terms.items():
                normalization_radicand = _normalization_ratio_squared(
                    ell, complex_output, target_ell, complex_input
                )
                for (
                    real_input,
                    input_coefficient,
                    input_radicand,
                ) in _real_inputs_for_complex(complex_input):
                    coefficient = _gaussian_multiply(
                        output_coefficient,
                        _gaussian_multiply(derivative_coefficient, input_coefficient),
                    )
                    radicand = output_radicand * normalization_radicand * input_radicand
                    if coefficient[0] != 0 and coefficient[1] != 0:
                        raise RuntimeError("complex radical coefficient is not axial.")
                    if coefficient[0] != 0:
                        radicand *= coefficient[0] * coefficient[0]
                        coefficient = (
                            Fraction(1 if coefficient[0] > 0 else -1),
                            Fraction(0),
                        )
                    elif coefficient[1] != 0:
                        radicand *= coefficient[1] * coefficient[1]
                        coefficient = (
                            Fraction(0),
                            Fraction(1 if coefficient[1] > 0 else -1),
                        )
                    outside, radicand = _canonical_square_root(radicand)
                    coefficient = (
                        coefficient[0] * outside,
                        coefficient[1] * outside,
                    )
                    key = (real_output + ell, real_input + target_ell, radicand)
                    symbolic[key] = _gaussian_add(
                        symbolic.get(key, _ZERO_GAUSSIAN), coefficient
                    )

    result = np.zeros((2 * ell + 1, 2 * target_ell + 1), dtype=float)
    for (row, column, radicand), coefficient in symbolic.items():
        if coefficient == _ZERO_GAUSSIAN:
            continue
        if coefficient[1] != 0:
            raise RuntimeError(
                "real-basis derivative coefficient is not real: "
                f"ell={ell}, axes={axes}, key={(row, column, radicand)}, "
                f"coefficient={coefficient}."
            )
        result[row, column] += float(coefficient[0]) * math.sqrt(float(radicand))
    return target_ell, result


def _derivative_matrices(
    output_lmax: int,
    input_lmax: int,
    kind: SolidHarmonicKind,
    axes: tuple[int, ...],
) -> np.ndarray:
    output_count = (output_lmax + 1) ** 2
    input_count = (input_lmax + 1) ** 2
    matrix = np.zeros((output_count, input_count), dtype=float)
    for ell in range(output_lmax + 1):
        neighbour, real_block = _real_derivative_block(ell, kind, axes)
        if neighbour < 0:
            continue
        output_slice = slice(ell * ell, (ell + 1) ** 2)
        input_slice = slice(neighbour * neighbour, (neighbour + 1) ** 2)
        matrix[output_slice, input_slice] = real_block
    return np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)


def _sparse_rows(matrix: np.ndarray) -> _SparseRows:
    row_entries: list[list[tuple[int, float]]] = []
    maximum_terms = 0
    for row in matrix:
        columns = np.flatnonzero(row)
        entries = sorted(
            ((int(column), float(row[column])) for column in columns),
            key=lambda entry: abs(entry[1]),
        )
        row_entries.append(entries)
        maximum_terms = max(maximum_terms, len(entries))
    indices = np.zeros((matrix.shape[0], maximum_terms), dtype=np.int64)
    coefficients = np.zeros((matrix.shape[0], maximum_terms), dtype=np.float64)
    for row, entries in enumerate(row_entries):
        for term, (column, coefficient) in enumerate(entries):
            indices[row, term] = column
            coefficients[row, term] = coefficient
    return _SparseRows(
        indices=np.frombuffer(indices.tobytes(), dtype=indices.dtype).reshape(
            indices.shape
        ),
        coefficients=np.frombuffer(
            coefficients.tobytes(), dtype=coefficients.dtype
        ).reshape(coefficients.shape),
    )


@lru_cache(maxsize=64)
def _derivative_tables(lmax: int, kind: SolidHarmonicKind) -> _DerivativeTables:
    first_input_lmax = lmax + 1 if kind == "irregular" else lmax
    physical_first = tuple(
        _derivative_matrices(lmax, first_input_lmax, kind, (axis,)) for axis in range(3)
    )
    second_input_lmax = lmax + 2 if kind == "irregular" else lmax
    hessian_dense = tuple(
        tuple(
            _derivative_matrices(
                lmax, second_input_lmax, kind, (first_axis, second_axis)
            )
            for second_axis in range(3)
        )
        for first_axis in range(3)
    )
    for first_axis in range(3):
        for second_axis in range(3):
            if not np.array_equal(
                hessian_dense[first_axis][second_axis],
                hessian_dense[second_axis][first_axis],
            ):
                raise RuntimeError(
                    "exact mixed-partial derivative tables do not commute."
                )

    return _DerivativeTables(
        gradient=tuple(  # type: ignore[arg-type]
            _sparse_rows(matrix) for matrix in physical_first
        ),
        hessian=tuple(
            tuple(_sparse_rows(matrix) for matrix in row) for row in hessian_dense
        ),
        hessian_transpose=tuple(
            tuple(_sparse_rows(matrix.T) for matrix in row) for row in hessian_dense
        ),
    )


def _validate_lmax(lmax: int) -> int:
    if isinstance(lmax, bool) or not isinstance(lmax, int):
        raise TypeError("lmax must be an integer.")
    if not 0 <= lmax <= MAXIMUM_PHYSICAL_LMAX:
        raise ValueError(f"lmax must lie in [0, {MAXIMUM_PHYSICAL_LMAX}].")
    return lmax


def _validate_kind(kind: str) -> SolidHarmonicKind:
    if kind not in ("regular", "irregular"):
        raise ValueError("kind must be 'regular' or 'irregular'.")
    return kind  # type: ignore[return-value]


def _regular_solid_values(flat: Any, lmax: int):
    """The existing real-harmonic recurrence, evaluated homogeneously."""

    torch = __import__("torch")
    x, y, z = flat.unbind(-1)
    harmonic_parts: dict[tuple[int, int, str], Any] = {}
    complex_xy = torch.complex(x, y)
    sectoral = torch.ones_like(complex_xy)
    for order in range(lmax + 1):
        if order == 0:
            cosine = torch.ones_like(z)
            sine = torch.zeros_like(z)
        else:
            # Repeated Cartesian multiplication preserves structural zeros on
            # coordinate axes; complex ``pow`` may route through polar form.
            sectoral = sectoral * complex_xy
            phase = (-1) ** order * math.prod(range(1, 2 * order, 2))
            cosine = phase * sectoral.real
            sine = phase * sectoral.imag
        cosine_sequence = {order: cosine}
        sine_sequence = {order: sine}
        if order < lmax:
            cosine_sequence[order + 1] = (2 * order + 1) * z * cosine
            sine_sequence[order + 1] = (2 * order + 1) * z * sine
        for ell in range(order + 2, lmax + 1):
            cosine_sequence[ell] = (
                (2 * ell - 1) * z * cosine_sequence[ell - 1]
                - (ell + order - 1) * (x * x + y * y + z * z) * cosine_sequence[ell - 2]
            ) / (ell - order)
            sine_sequence[ell] = (
                (2 * ell - 1) * z * sine_sequence[ell - 1]
                - (ell + order - 1) * (x * x + y * y + z * z) * sine_sequence[ell - 2]
            ) / (ell - order)
        for ell in range(order, lmax + 1):
            harmonic_parts[(ell, order, "c")] = cosine_sequence[ell]
            harmonic_parts[(ell, order, "s")] = sine_sequence[ell]

    columns = []
    for ell in range(lmax + 1):
        for signed_order in range(-ell, ell + 1):
            order = abs(signed_order)
            normalization = _normalization(ell, order)
            if signed_order < 0:
                column = (
                    math.sqrt(2.0)
                    * ((-1) ** order)
                    * normalization
                    * harmonic_parts[(ell, order, "s")]
                )
            elif signed_order == 0:
                column = normalization * harmonic_parts[(ell, 0, "c")]
            else:
                column = (
                    math.sqrt(2.0)
                    * ((-1) ** order)
                    * normalization
                    * harmonic_parts[(ell, order, "c")]
                )
            columns.append(column)
    return torch.stack(columns, dim=-1)


def _solid_values(vectors: Any, lmax: int, kind: SolidHarmonicKind):
    torch = __import__("torch")
    flat = vectors.reshape(-1, 3)
    radii = torch.linalg.vector_norm(flat, dim=-1)
    if kind == "irregular" and bool(torch.any(radii == 0).item()):
        raise ValueError("irregular solid harmonics require nonzero vectors.")
    regular = _regular_solid_values(flat, lmax)
    if kind == "regular":
        result = regular
    else:
        blocks = []
        for ell in range(lmax + 1):
            blocks.append(
                regular[:, ell * ell : (ell + 1) ** 2] / radii[:, None] ** (2 * ell + 1)
            )
        result = torch.cat(blocks, dim=-1)
    return result.reshape(vectors.shape[:-1] + ((lmax + 1) ** 2,))


def _apply_sparse(values: Any, rows: _SparseRows):
    torch = __import__("torch")
    if rows.indices.shape[1] == 0:
        return values.new_zeros(values.shape[:-1] + (rows.indices.shape[0],))
    indices = values.new_tensor(rows.indices, dtype=torch.long)
    coefficients = values.new_tensor(rows.coefficients)
    return (values[..., indices] * coefficients).sum(dim=-1)


def _validate_vectors(vectors: Any, torch: Any) -> None:
    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a torch.Tensor.")
    if vectors.dtype != torch.float64:
        raise ValueError("vectors must use torch.float64.")
    if vectors.device.type not in ("cpu", "cuda"):
        raise ValueError("vectors must be on CPU or CUDA.")
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError("vectors must have shape (..., 3).")
    if not bool(torch.all(torch.isfinite(vectors)).item()):
        raise ValueError("vectors must be finite.")


def solid_harmonic_jets(
    vectors: Any,
    lmax: int,
    kind: SolidHarmonicKind,
    derivative_order: int = 0,
) -> SolidHarmonicJets:
    """Evaluate real solid harmonics and exact analytic derivatives.

    ``vectors`` has shape ``(..., 3)`` and must be FP64 on CPU or CUDA.  The
    result uses flattened ``ell, m=-ell..ell`` ordering.  If vectors have
    already been divided by a radius, callers apply ``1/radius`` and
    ``1/radius**2`` to the returned gradient and Hessian respectively.
    """

    torch = __import__("torch")
    maximum = _validate_lmax(lmax)
    harmonic_kind = _validate_kind(kind)
    if derivative_order not in (0, 1, 2):
        raise ValueError("derivative_order must be 0, 1, or 2.")
    _validate_vectors(vectors, torch)

    value_lmax = maximum
    if derivative_order:
        value_lmax += derivative_order if harmonic_kind == "irregular" else 0
    internal_values = _solid_values(vectors, value_lmax, harmonic_kind)
    physical_count = (maximum + 1) ** 2
    values = internal_values[..., :physical_count]
    if derivative_order == 0:
        return SolidHarmonicJets(values=values, gradient=None, hessian=None)

    tables = _derivative_tables(maximum, harmonic_kind)
    gradient = torch.stack(
        tuple(_apply_sparse(internal_values, table) for table in tables.gradient),
        dim=-1,
    )
    if derivative_order == 1:
        return SolidHarmonicJets(values=values, gradient=gradient, hessian=None)
    hessian = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    _apply_sparse(internal_values, tables.hessian[a][b])
                    for b in range(3)
                ),
                dim=-1,
            )
            for a in range(3)
        ),
        dim=-2,
    )
    return SolidHarmonicJets(values=values, gradient=gradient, hessian=hessian)


def solid_harmonic_contracted_hessian(
    vectors: Any,
    lmax: int,
    kind: SolidHarmonicKind,
    coefficients: Any,
) -> SolidHarmonicContractedHessian:
    """Evaluate basis values/gradients and one exact contracted Hessian.

    ``coefficients`` is the fixed central coefficient vector of the scalar
    solid-harmonic series.  The Hessian is evaluated as
    ``Y @ (D_ab.T @ coefficients)`` for all nine Cartesian axis pairs, so no
    ``(..., basis, 3, 3)`` basis-Hessian tensor is formed.
    """

    torch = __import__("torch")
    maximum = _validate_lmax(lmax)
    harmonic_kind = _validate_kind(kind)
    _validate_vectors(vectors, torch)
    physical_count = (maximum + 1) ** 2
    if not isinstance(coefficients, torch.Tensor):
        raise TypeError("coefficients must be a torch.Tensor.")
    if coefficients.shape != (physical_count,):
        raise ValueError(f"coefficients must have shape ({physical_count},).")
    if coefficients.dtype != torch.float64:
        raise ValueError("coefficients must use torch.float64.")
    if coefficients.device != vectors.device:
        raise ValueError("coefficients and vectors must use the same device.")
    if not bool(torch.all(torch.isfinite(coefficients)).item()):
        raise ValueError("coefficients must be finite.")

    value_lmax = maximum + 2 if harmonic_kind == "irregular" else maximum
    internal_values = _solid_values(vectors, value_lmax, harmonic_kind)
    basis_values = internal_values[..., :physical_count]
    tables = _derivative_tables(maximum, harmonic_kind)
    basis_gradient = torch.stack(
        tuple(_apply_sparse(internal_values, table) for table in tables.gradient),
        dim=-1,
    )
    transformed_coefficients = torch.stack(
        tuple(
            _apply_sparse(coefficients, tables.hessian_transpose[a][b])
            for a in range(3)
            for b in range(3)
        ),
        dim=-1,
    )
    contracted_hessian = (internal_values @ transformed_coefficients).reshape(
        vectors.shape[:-1] + (3, 3)
    )
    return SolidHarmonicContractedHessian(
        basis_values=basis_values,
        basis_gradient=basis_gradient,
        contracted_hessian=contracted_hessian,
    )


def derivative_table_sha256(lmax: int, kind: SolidHarmonicKind) -> str:
    """Return a deterministic provenance hash of the analytic sparse tables."""

    maximum = _validate_lmax(lmax)
    harmonic_kind = _validate_kind(kind)
    tables = _derivative_tables(maximum, harmonic_kind)

    def encoded(rows: _SparseRows) -> dict[str, object]:
        return {
            "indices": rows.indices.tolist(),
            "coefficients_hex": [
                [float(value).hex() for value in row] for row in rows.coefficients
            ],
        }

    payload = {
        "version": DERIVATIVE_TABLE_VERSION,
        "lmax": maximum,
        "kind": harmonic_kind,
        "gradient": [encoded(rows) for rows in tables.gradient],
        "hessian": [[encoded(rows) for rows in axis] for axis in tables.hessian],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def contracted_hessian_table_sha256(lmax: int, kind: SolidHarmonicKind) -> str:
    """Return the provenance hash of the exact sparse transpose tables."""

    maximum = _validate_lmax(lmax)
    harmonic_kind = _validate_kind(kind)
    tables = _derivative_tables(maximum, harmonic_kind)

    def encoded(rows: _SparseRows) -> dict[str, object]:
        return {
            "indices": rows.indices.tolist(),
            "coefficients_hex": [
                [float(value).hex() for value in row] for row in rows.coefficients
            ],
        }

    payload = {
        "version": CONTRACTED_HESSIAN_TABLE_VERSION,
        "lmax": maximum,
        "kind": harmonic_kind,
        "hessian_transpose": [
            [encoded(rows) for rows in axis] for axis in tables.hessian_transpose
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def derivative_table_nbytes(lmax: int, kind: SolidHarmonicKind) -> int:
    """Return exact host storage for the full-jets sparse table family."""

    maximum = _validate_lmax(lmax)
    harmonic_kind = _validate_kind(kind)
    tables = _derivative_tables(maximum, harmonic_kind)
    rows = (*tables.gradient, *(item for axis in tables.hessian for item in axis))
    return sum(item.indices.nbytes + item.coefficients.nbytes for item in rows)


def contracted_hessian_table_nbytes(lmax: int, kind: SolidHarmonicKind) -> int:
    """Return exact host storage retained by the sparse transpose tables."""

    maximum = _validate_lmax(lmax)
    harmonic_kind = _validate_kind(kind)
    tables = _derivative_tables(maximum, harmonic_kind)
    rows = tuple(item for axis in tables.hessian_transpose for item in axis)
    return sum(item.indices.nbytes + item.coefficients.nbytes for item in rows)


__all__ = [
    "CONTRACTED_HESSIAN_TABLE_VERSION",
    "DERIVATIVE_TABLE_VERSION",
    "MAXIMUM_INTERNAL_LMAX",
    "MAXIMUM_PHYSICAL_LMAX",
    "SolidHarmonicContractedHessian",
    "SolidHarmonicJets",
    "contracted_hessian_table_nbytes",
    "contracted_hessian_table_sha256",
    "derivative_table_nbytes",
    "derivative_table_sha256",
    "solid_harmonic_contracted_hessian",
    "solid_harmonic_jets",
]
