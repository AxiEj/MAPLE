"""Exact coordinate-only preconditions for a three-site affine CHA segment.

This module intentionally does not evaluate R6, Born radii, CHA weighted
signs, energies, forces, or quadratures.  It characterizes only pair-distance
and electrostatic-size coordinate facts in a declared exact-real model built
from decoded finite float64 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
import struct

from .continuum_chagb_inputs import ContinuumChaTopology
from .torch_chagb import (
    EFFECTIVE_PROBE_ANGSTROM,
    SIZE_GUARD_ANGSTROM,
    SIZE_SWITCH_ANGSTROM,
)

_POINT_MARGIN_ANGSTROM = 1.0e-8
_MAX_DEPTH = 12
_MAX_NODES = 8191
_MAX_INTEGER_BITS = 8192
_PAIRS = ((0, 1), (0, 2), (1, 2))


class SegmentPreconditionValidationError(ValueError):
    """The externally pinned three-site input contract is invalid."""


class SegmentPreconditionResourceError(ValueError):
    """Exact arithmetic exceeded a frozen deterministic resource limit."""


@dataclass(frozen=True)
class SegmentPreconditionBudget:
    max_depth: int = _MAX_DEPTH
    max_nodes: int = _MAX_NODES
    max_integer_bits: int = _MAX_INTEGER_BITS

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_depth", self.max_depth, _MAX_DEPTH),
            ("max_nodes", self.max_nodes, _MAX_NODES),
            ("max_integer_bits", self.max_integer_bits, _MAX_INTEGER_BITS),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SegmentPreconditionValidationError(
                    f"{name} must be a nonnegative integer."
                )
            if value > maximum:
                raise SegmentPreconditionValidationError(
                    f"{name} exceeds the frozen v1 maximum {maximum}."
                )
        if self.max_nodes < 1 or self.max_integer_bits < 2:
            raise SegmentPreconditionValidationError(
                "At least one node and two integer bits are required."
            )


@dataclass(frozen=True)
class ExactConstant:
    name: str
    source_expression: str
    units: str
    float64_hex: str
    exact_value: Fraction
    mathematical_use: str


@dataclass(frozen=True)
class PairDistanceCharacterization:
    pair: tuple[int, int]
    coefficients_angstrom2: tuple[Fraction, Fraction, Fraction]
    minimum_angstrom2: Fraction
    minimum_at_t: Fraction
    maximum_angstrom2: Fraction
    maximum_at_t: Fraction
    collision_status: str
    distinct_center_guard_status: str
    distinct_center_guard_angstrom2: Fraction
    expanded_pair_relation: str
    expanded_radii_angstrom: tuple[Fraction, Fraction]
    guard_thresholds_angstrom2: tuple[tuple[str, Fraction], ...]
    units: str = field(init=False, default="angstrom^2")


@dataclass(frozen=True)
class BernsteinLeaf:
    lower_t: Fraction
    upper_t: Fraction
    coefficients_angstrom6: tuple[Fraction, ...]
    lower_bound_angstrom6: Fraction
    upper_bound_angstrom6: Fraction
    status: str
    depth: int


@dataclass(frozen=True)
class SizePolynomialCharacterization:
    power_coefficients_angstrom6: tuple[Fraction, ...]
    degree: int
    lower_guard_angstrom6: Fraction
    upper_guard_angstrom6: Fraction
    leaves: tuple[BernsteinLeaf, ...]
    range_status: str
    units: str = field(init=False, default="angstrom^6")


@dataclass(frozen=True)
class SegmentPreconditionResources:
    nodes_consumed: int
    maximum_depth_consumed: int
    maximum_tracked_integer_bits_observed: int
    comparison_scratch_bound_bits: int
    budget: SegmentPreconditionBudget


@dataclass(frozen=True)
class ChaSegmentPreconditionReport:
    status: str
    start_sha256: str
    trial_sha256: str
    topology_sha256: str
    pair_distances: tuple[PairDistanceCharacterization, ...]
    size_polynomial: SizePolynomialCharacterization
    constants: tuple[ExactConstant, ...]
    resources: SegmentPreconditionResources
    implemented_predicates: tuple[str, ...]
    omitted_predicates: tuple[str, ...]
    dynamic_born_sign_status: str = field(init=False, default="UNRESOLVED")
    computed_quadrature_smoothness: str = field(init=False, default="UNRESOLVED")
    optimizer_eligible: bool = field(init=False, default=False)
    public_admission: bool = field(init=False, default=False)
    segment_certificate: bool = field(init=False, default=False)
    arithmetic_scope: str = field(
        init=False,
        default=(
            "declared exact-real model of decoded float64 inputs; not IEEE/Torch "
            "reduction, determinant, or branch equivalence; max_integer_bits "
            "covers stored rational components and explicit checked arithmetic"
        ),
    )


def _fraction(value: float) -> Fraction:
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _float_hex(value: float) -> str:
    return struct.pack(">d", value).hex()


def _integer_bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


class _CheckedRationalArithmetic:
    """Small local exact-arithmetic budget tracker for this degree-six proof."""

    def __init__(self, budget: SegmentPreconditionBudget):
        self.budget = budget
        self.maximum_tracked_integer_bits_observed = 0

    def _observe_int(self, value: int) -> int:
        bits = abs(value).bit_length()
        self.maximum_tracked_integer_bits_observed = max(
            self.maximum_tracked_integer_bits_observed, bits
        )
        if bits > self.budget.max_integer_bits:
            raise SegmentPreconditionResourceError(
                "Exact integer intermediate exceeded max_integer_bits."
            )
        return value

    def observe(self, value: Fraction | int) -> Fraction:
        result = value if isinstance(value, Fraction) else Fraction(value)
        self._observe_int(result.numerator)
        self._observe_int(result.denominator)
        return result

    def _mul_int(self, left: int, right: int) -> int:
        if left and right:
            minimum_bits = abs(left).bit_length() + abs(right).bit_length() - 1
            if minimum_bits > self.budget.max_integer_bits:
                raise SegmentPreconditionResourceError(
                    "Exact integer product cannot fit max_integer_bits."
                )
        return self._observe_int(left * right)

    def neg(self, value: Fraction | int) -> Fraction:
        item = self.observe(value)
        return self.observe(Fraction(-item.numerator, item.denominator))

    def add(self, left: Fraction | int, right: Fraction | int) -> Fraction:
        first = self.observe(left)
        second = self.observe(right)
        common = math.gcd(first.denominator, second.denominator)
        first_scale = second.denominator // common
        second_scale = first.denominator // common
        first_term = self._mul_int(first.numerator, first_scale)
        second_term = self._mul_int(second.numerator, second_scale)
        numerator = self._observe_int(first_term + second_term)
        denominator = self._mul_int(first.denominator, first_scale)
        return self.observe(Fraction(numerator, denominator))

    def sub(self, left: Fraction | int, right: Fraction | int) -> Fraction:
        return self.add(left, self.neg(right))

    def mul(self, left: Fraction | int, right: Fraction | int) -> Fraction:
        first = self.observe(left)
        second = self.observe(right)
        cross_first = math.gcd(abs(first.numerator), second.denominator)
        cross_second = math.gcd(abs(second.numerator), first.denominator)
        numerator = self._mul_int(
            first.numerator // cross_first, second.numerator // cross_second
        )
        denominator = self._mul_int(
            first.denominator // cross_second, second.denominator // cross_first
        )
        return self.observe(Fraction(numerator, denominator))

    def div(self, left: Fraction | int, right: Fraction | int) -> Fraction:
        divisor = self.observe(right)
        if divisor == 0:
            raise ZeroDivisionError("exact rational division by zero")
        return self.mul(left, Fraction(divisor.denominator, divisor.numerator))

    def pow(self, value: Fraction | int, exponent: int) -> Fraction:
        if exponent < 0:
            raise ValueError("checked exact powers require a nonnegative exponent")
        result = self.observe(Fraction(1))
        base = self.observe(value)
        for _ in range(exponent):
            result = self.mul(result, base)
        return result

    def sum(self, values) -> Fraction:
        result = self.observe(Fraction(0))
        for value in values:
            result = self.add(result, value)
        return result


def _checked(arithmetic=None) -> _CheckedRationalArithmetic:
    return arithmetic or _CheckedRationalArithmetic(SegmentPreconditionBudget())


def _poly_trim(values: list[Fraction]) -> tuple[Fraction, ...]:
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _poly_add(left, right, arithmetic=None):
    exact = _checked(arithmetic)
    result = [Fraction(0)] * max(len(left), len(right))
    for index, value in enumerate(left):
        result[index] = exact.add(result[index], value)
    for index, value in enumerate(right):
        result[index] = exact.add(result[index], value)
    return _poly_trim(result)


def _poly_scale(values, factor, arithmetic=None):
    exact = _checked(arithmetic)
    return _poly_trim([exact.mul(value, factor) for value in values])


def _poly_mul(left, right, arithmetic=None):
    exact = _checked(arithmetic)
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for first, left_value in enumerate(left):
        for second, right_value in enumerate(right):
            result[first + second] = exact.add(
                result[first + second], exact.mul(left_value, right_value)
            )
    return _poly_trim(result)


def _evaluate_polynomial(coefficients, value: Fraction, arithmetic=None) -> Fraction:
    exact = _checked(arithmetic)
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = exact.add(exact.mul(result, value), coefficient)
    return result


def _compose_interval(coefficients, lower: Fraction, upper: Fraction, arithmetic=None):
    """Return coefficients of p(lower + (upper-lower)*x)."""
    exact = _checked(arithmetic)
    width = exact.sub(upper, lower)
    result = [Fraction(0)]
    power = [Fraction(1)]
    affine = [lower, width]
    for index, coefficient in enumerate(coefficients):
        result = list(_poly_add(result, _poly_scale(power, coefficient, exact), exact))
        if index + 1 < len(coefficients):
            power = list(_poly_mul(power, affine, exact))
    return tuple(result)


def _bernstein_coefficients(
    coefficients, lower: Fraction, upper: Fraction, arithmetic=None
) -> tuple[Fraction, ...]:
    """Convert a power polynomial to exact fixed-degree Bernstein form."""
    exact = _checked(arithmetic)
    degree = len(coefficients) - 1
    local = _compose_interval(coefficients, lower, upper, exact)
    local = tuple(local) + (Fraction(0),) * (degree + 1 - len(local))
    return tuple(
        exact.sum(
            exact.mul(
                local[index],
                Fraction(math.comb(k, index), math.comb(degree, index)),
            )
            for index in range(k + 1)
        )
        for k in range(degree + 1)
    )


def _coordinate_hash(values) -> str:
    detached = values.detach().contiguous().cpu()
    payload = b"torch.float64:[3,3]:cpu:" + detached.numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _validate_inputs(start, trial, topology, expected_topology_sha256, budget):
    import torch

    if not isinstance(topology, ContinuumChaTopology):
        raise SegmentPreconditionValidationError(
            "topology must be a ContinuumChaTopology."
        )
    try:
        topology.assert_current(expected_topology_sha256)
    except (TypeError, ValueError) as error:
        raise SegmentPreconditionValidationError(str(error)) from error
    if topology.atom_count != 3:
        raise SegmentPreconditionValidationError(
            "Exactly three topology sites are required."
        )
    if not isinstance(budget, SegmentPreconditionBudget):
        raise SegmentPreconditionValidationError(
            "budget must be a SegmentPreconditionBudget."
        )
    for name, value in (("start", start), ("trial", trial)):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float64
            or value.shape != (3, 3)
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())
        ):
            raise SegmentPreconditionValidationError(
                f"{name} must be a finite CPU float64 [3,3] tensor."
            )


def _quadratic_extrema(
    coefficients: tuple[Fraction, Fraction, Fraction],
    arithmetic=None,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    exact = _checked(arithmetic)
    c, b, a = coefficients
    candidates: list[tuple[Fraction, Fraction]] = [
        (Fraction(0), c),
        (Fraction(1), exact.add(exact.add(a, b), c)),
    ]
    if a > 0:
        vertex = exact.div(exact.neg(b), exact.mul(2, a))
        if 0 <= vertex <= 1:
            candidates.append(
                (vertex, _evaluate_polynomial(coefficients, vertex, exact))
            )
    minimum_t, minimum = min(candidates, key=lambda item: (item[1], item[0]))
    maximum_t, maximum = max(candidates, key=lambda item: (item[1], exact.neg(item[0])))
    return minimum, minimum_t, maximum, maximum_t


def _threshold_relation(minimum, maximum, thresholds):
    values = sorted(set(value for _, value in thresholds))
    if any(minimum < value < maximum for value in values):
        return "INTERIOR_RELATION_CHANGE"
    if any(minimum == value or maximum == value for value in values):
        return "TOUCHES_GUARD"
    threshold_map = dict(thresholds)
    inner_minus = threshold_map["inner_minus_margin"]
    inner_plus = threshold_map["inner_plus_margin"]
    outer_minus = threshold_map["outer_minus_margin"]
    outer_plus = threshold_map["outer_plus_margin"]
    if maximum < inner_minus:
        return "CONSTANT_CONTAINED"
    if minimum > outer_plus:
        return "CONSTANT_DISJOINT"
    if minimum > inner_plus and maximum < outer_minus:
        return "CONSTANT_REGULAR_INTERSECTION"
    if minimum >= inner_minus and maximum <= inner_plus:
        return "INSIDE_INNER_GUARD"
    if minimum >= outer_minus and maximum <= outer_plus:
        return "INSIDE_OUTER_GUARD"
    return "INSIDE_OVERLAPPING_GUARDS"


def _distinct_center_relation(minimum, maximum, threshold):
    if minimum < threshold < maximum:
        return "INTERIOR_GUARD_CROSSING"
    if minimum == threshold or maximum == threshold:
        return "TOUCHES_DISTINCT_CENTER_GUARD"
    if maximum < threshold:
        return "CONSTANT_INSIDE_DISTINCT_CENTER_GUARD"
    return "CONSTANT_OUTSIDE_DISTINCT_CENTER_GUARD"


def _pair_characterizations(start, trial, expanded, arithmetic=None):
    exact = _checked(arithmetic)
    result = []
    margin = exact.observe(_fraction(_POINT_MARGIN_ANGSTROM))
    distinct_threshold = exact.pow(margin, 2)
    for first, second in _PAIRS:
        initial = [
            exact.sub(start[first][axis], start[second][axis]) for axis in range(3)
        ]
        delta = [
            exact.sub(
                exact.sub(trial[first][axis], start[first][axis]),
                exact.sub(trial[second][axis], start[second][axis]),
            )
            for axis in range(3)
        ]
        coefficients = (
            exact.sum(exact.mul(value, value) for value in initial),
            exact.mul(
                2,
                exact.sum(
                    exact.mul(x, dx) for x, dx in zip(initial, delta, strict=True)
                ),
            ),
            exact.sum(exact.mul(value, value) for value in delta),
        )
        minimum, minimum_t, maximum, maximum_t = _quadratic_extrema(coefficients, exact)
        inner = abs(exact.sub(expanded[first], expanded[second]))
        outer = exact.add(expanded[first], expanded[second])
        thresholds_distance = (
            ("inner_minus_margin", max(Fraction(0), exact.sub(inner, margin))),
            ("inner_plus_margin", exact.add(inner, margin)),
            ("outer_minus_margin", max(Fraction(0), exact.sub(outer, margin))),
            ("outer_plus_margin", exact.add(outer, margin)),
        )
        thresholds = tuple(
            (name, exact.pow(value, 2)) for name, value in thresholds_distance
        )
        collision = (
            "CONSTANT_COLLISION"
            if minimum == 0 and maximum == 0
            else (
                "INTERIOR_COLLISION"
                if minimum == 0 and 0 < minimum_t < 1
                else "TOUCHES_COLLISION" if minimum == 0 else "NONCOLLIDING"
            )
        )
        result.append(
            PairDistanceCharacterization(
                pair=(first, second),
                coefficients_angstrom2=coefficients,
                minimum_angstrom2=minimum,
                minimum_at_t=minimum_t,
                maximum_angstrom2=maximum,
                maximum_at_t=maximum_t,
                collision_status=collision,
                distinct_center_guard_status=_distinct_center_relation(
                    minimum, maximum, distinct_threshold
                ),
                distinct_center_guard_angstrom2=distinct_threshold,
                expanded_pair_relation=_threshold_relation(
                    minimum, maximum, thresholds
                ),
                expanded_radii_angstrom=(expanded[first], expanded[second]),
                guard_thresholds_angstrom2=thresholds,
            )
        )
    return tuple(result)


def _determinant3(matrix, arithmetic=None):
    exact = _checked(arithmetic)

    def product(first, second, third):
        return _poly_mul(first, _poly_mul(second, third, exact), exact)

    positive = _poly_add(
        _poly_add(
            product(matrix[0][0], matrix[1][1], matrix[2][2]),
            product(matrix[0][1], matrix[1][2], matrix[2][0]),
            exact,
        ),
        product(matrix[0][2], matrix[1][0], matrix[2][1]),
        exact,
    )
    negative = _poly_add(
        _poly_add(
            product(matrix[0][2], matrix[1][1], matrix[2][0]),
            product(matrix[0][1], matrix[1][0], matrix[2][2]),
            exact,
        ),
        product(matrix[0][0], matrix[1][2], matrix[2][1]),
        exact,
    )
    return _poly_add(positive, _poly_scale(negative, -1, exact), exact)


def _size_polynomial(start, trial, radii, arithmetic=None):
    exact = _checked(arithmetic)
    weights = tuple(exact.pow(radius, 3) for radius in radii)
    mass = exact.sum(weights)
    positions = []
    for site in range(3):
        positions.append(
            tuple(
                (
                    start[site][axis],
                    exact.sub(trial[site][axis], start[site][axis]),
                )
                for axis in range(3)
            )
        )
    center = []
    for axis in range(3):
        center.append(
            tuple(
                exact.div(
                    exact.sum(
                        exact.mul(weights[site], positions[site][axis][power])
                        for site in range(3)
                    ),
                    mass,
                )
                for power in range(2)
            )
        )
    second = [[[Fraction(0)] for _ in range(3)] for _ in range(3)]
    for row in range(3):
        for column in range(3):
            value = (Fraction(0),)
            for site in range(3):
                left = tuple(
                    exact.sub(positions[site][row][power], center[row][power])
                    for power in range(2)
                )
                right = tuple(
                    exact.sub(positions[site][column][power], center[column][power])
                    for power in range(2)
                )
                value = _poly_add(
                    value,
                    _poly_scale(_poly_mul(left, right, exact), weights[site], exact),
                    exact,
                )
            second[row][column] = list(value)
    literal_two_over_five = exact.observe(_fraction(2.0 / 5.0))
    sphere = exact.mul(
        literal_two_over_five,
        exact.sum(
            exact.mul(weight, exact.pow(radius, 2))
            for weight, radius in zip(weights, radii, strict=True)
        ),
    )
    trace = _poly_add(_poly_add(second[0][0], second[1][1], exact), second[2][2], exact)
    inertia = []
    for row in range(3):
        inertia_row = []
        for column in range(3):
            if row == column:
                inertia_row.append(
                    _poly_add(
                        _poly_add(trace, (sphere,), exact),
                        _poly_scale(second[row][column], -1, exact),
                        exact,
                    )
                )
            else:
                inertia_row.append(_poly_scale(second[row][column], -1, exact))
        inertia.append(inertia_row)
    determinant = _determinant3(inertia, exact)
    factor = exact.div(exact.pow(Fraction(5, 2), 3), exact.pow(mass, 3))
    polynomial = list(_poly_scale(determinant, factor, exact))
    if len(polynomial) > 7:
        raise AssertionError("electrostatic size polynomial exceeded degree six")
    polynomial.extend([Fraction(0)] * (7 - len(polynomial)))
    for value in polynomial:
        exact.observe(value)
    return tuple(polynomial)


def _bound_size(polynomial, budget, arithmetic=None):
    exact = arithmetic or _CheckedRationalArithmetic(budget)
    if exact.budget != budget:
        raise SegmentPreconditionValidationError(
            "Shared exact arithmetic tracker and segment budget disagree."
        )
    for value in polynomial:
        exact.observe(value)
    switch = exact.observe(_fraction(SIZE_SWITCH_ANGSTROM))
    guard = exact.observe(_fraction(SIZE_GUARD_ANGSTROM))
    lower_guard = exact.pow(exact.sub(switch, guard), 6)
    upper_guard = exact.pow(exact.add(switch, guard), 6)
    stack = [(Fraction(0), Fraction(1), 0)]
    leaves = []
    nodes = 0
    max_depth = 0
    while stack:
        lower, upper, depth = stack.pop()
        nodes += 1
        coefficients = _bernstein_coefficients(polynomial, lower, upper, exact)
        bound_lower = min(coefficients)
        bound_upper = max(coefficients)
        max_depth = max(max_depth, depth)
        if bound_upper < lower_guard:
            status = "BELOW_SIZE_GUARD"
        elif bound_lower > upper_guard:
            status = "ABOVE_SIZE_GUARD"
        elif depth >= budget.max_depth or nodes + len(stack) + 2 > budget.max_nodes:
            status = "TOUCHES_OR_UNRESOLVED_SIZE_GUARD"
        else:
            midpoint = exact.div(exact.add(lower, upper), 2)
            stack.append((midpoint, upper, depth + 1))
            stack.append((lower, midpoint, depth + 1))
            continue
        leaves.append(
            BernsteinLeaf(
                lower_t=lower,
                upper_t=upper,
                coefficients_angstrom6=coefficients,
                lower_bound_angstrom6=bound_lower,
                upper_bound_angstrom6=bound_upper,
                status=status,
                depth=depth,
            )
        )
    statuses = {leaf.status for leaf in leaves}
    range_status = (
        statuses.pop() if len(statuses) == 1 else "TOUCHES_OR_UNRESOLVED_SIZE_GUARD"
    )
    return (
        SizePolynomialCharacterization(
            power_coefficients_angstrom6=polynomial,
            degree=max(
                (index for index, value in enumerate(polynomial) if value), default=0
            ),
            lower_guard_angstrom6=lower_guard,
            upper_guard_angstrom6=upper_guard,
            leaves=tuple(sorted(leaves, key=lambda item: item.lower_t)),
            range_status=range_status,
        ),
        nodes,
        max_depth,
        exact.maximum_tracked_integer_bits_observed,
    )


def _constant(name, expression, units, value, use):
    return ExactConstant(
        name, expression, units, _float_hex(value), _fraction(value), use
    )


def characterize_cha_segment_preconditions(
    start,
    trial,
    topology: ContinuumChaTopology,
    *,
    expected_topology_sha256: str,
    budget: SegmentPreconditionBudget = SegmentPreconditionBudget(),
) -> ChaSegmentPreconditionReport:
    """Characterize exact coordinate preconditions; never certify an OPT step."""
    _validate_inputs(start, trial, topology, expected_topology_sha256, budget)
    start_hash = _coordinate_hash(start)
    trial_hash = _coordinate_hash(trial)
    start_snapshot = start.detach().clone()
    trial_snapshot = trial.detach().clone()
    decoded_start = tuple(
        tuple(_fraction(float(value)) for value in row) for row in start_snapshot
    )
    decoded_trial = tuple(
        tuple(_fraction(float(value)) for value in row) for row in trial_snapshot
    )
    radii_float = tuple(float(value) for value in topology.cha_radii_angstrom)
    expanded_float = tuple(
        float(value + EFFECTIVE_PROBE_ANGSTROM) for value in radii_float
    )
    exact = _CheckedRationalArithmetic(budget)
    radii = tuple(exact.observe(_fraction(value)) for value in radii_float)
    expanded = tuple(exact.observe(_fraction(value)) for value in expanded_float)
    for value in (
        *(item for row in decoded_start for item in row),
        *(item for row in decoded_trial for item in row),
        *radii,
        *expanded,
    ):
        exact.observe(value)
    pairs = _pair_characterizations(decoded_start, decoded_trial, expanded, exact)
    polynomial = _size_polynomial(decoded_start, decoded_trial, radii, exact)
    size, nodes, depth, maximum_bits = _bound_size(polynomial, budget, exact)
    status = (
        "PRECONDITION_BOUNDS_UNRESOLVED"
        if size.range_status == "TOUCHES_OR_UNRESOLVED_SIZE_GUARD"
        else "SEGMENT_PRECONDITIONS_CHARACTERIZED"
    )
    constants = (
        *(
            _constant(
                f"cha_radius_{index}",
                "topology.cha_radii_angstrom",
                "angstrom",
                radius,
                "radius-cubed weight and exact radius powers",
            )
            for index, radius in enumerate(radii_float)
        ),
        *(
            _constant(
                f"expanded_radius_{index}",
                "actual float64(cha_radius + effective_probe), then decoded",
                "angstrom",
                radius,
                "expanded-pair relation; rounded point-code addition",
            )
            for index, radius in enumerate(expanded_float)
        ),
        _constant(
            "effective_probe",
            "float64(1.4 - 0.52)",
            "angstrom",
            EFFECTIVE_PROBE_ANGSTROM,
            "expanded-radius rounded addition operand",
        ),
        _constant(
            "point_domain_margin",
            "1.0e-8",
            "angstrom",
            _POINT_MARGIN_ANGSTROM,
            "pair guard",
        ),
        _constant(
            "size_switch", "10.0", "angstrom", SIZE_SWITCH_ANGSTROM, "size branch"
        ),
        _constant(
            "size_guard", "1.0e-8", "angstrom", SIZE_GUARD_ANGSTROM, "size branch guard"
        ),
        _constant(
            "sphere_moment_factor",
            "float64(2.0 / 5.0)",
            "dimensionless",
            2.0 / 5.0,
            "positive sphere inertia moment",
        ),
        _constant(
            "size_prefactor",
            "float64(2.5)",
            "dimensionless",
            2.5,
            "exact (2.5 / mass)^3 prefactor for size^6",
        ),
    )
    if _coordinate_hash(start) != start_hash or _coordinate_hash(trial) != trial_hash:
        raise SegmentPreconditionValidationError(
            "Coordinate input mutated during exact characterization."
        )
    return ChaSegmentPreconditionReport(
        status=status,
        start_sha256=start_hash,
        trial_sha256=trial_hash,
        topology_sha256=topology.content_sha256,
        pair_distances=pairs,
        size_polynomial=size,
        constants=constants,
        resources=SegmentPreconditionResources(
            nodes,
            depth,
            maximum_bits,
            2 * budget.max_integer_bits,
            budget,
        ),
        implemented_predicates=(
            "exact-pair-distance-quadratic-extrema",
            "expanded-pair-distance-guard-relations",
            "exact-size-sixth-power-polynomial",
            "bounded-dyadic-bernstein-size-branch",
            "positive-weight-sphere-moment-inertia-spd",
        ),
        omitted_predicates=(
            "dynamic-r6-contact-and-torus-integrals",
            "r6-contact-plus-torus-flux-j-angstrom^-3",
            "r6-unshifted-inverse-born-j^(1/3)-angstrom^-1",
            "dynamic-born-radii-and-shift",
            "weighted-cha-signs",
            "mu-branch",
            "r6-triple-collinearity-1e-9-angstrom",
            "r6-triple-height-squared-1e-18-angstrom^2",
            "r6-circle-amplitude-1e-18-angstrom^2",
            "r6-coverage-threshold-1e-9-dimensionless",
            "r6-full-circle-endpoint-tolerance-1e-12-radian",
            "r6-torus-pinch-and-tube-separation-angstrom",
            "sav-dispersion-sas-ownership-and-tangency",
            "sav-sas-pair-tangency-1e-9-angstrom",
            "integration-chart-axis-1e-12-dimensionless",
            "integration-breakpoint-deduplication-1e-12-angstrom",
            "integration-burial-offset-1e-10-angstrom",
            "integration-chart-and-adaptive-quadrature-smoothness",
            "dispersion-sigma-c1-c2",
            "scalar-force-accuracy",
            "gas-model-behavior",
            "optimizer-rollback-and-cache-semantics",
        ),
    )
