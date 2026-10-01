"""Exact coordinate-only CHA segment precondition contracts."""

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
import math

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_segment_preconditions import (
    SegmentPreconditionBudget,
    SegmentPreconditionResourceError,
    SegmentPreconditionValidationError,
    _CheckedRationalArithmetic,
    _bernstein_coefficients,
    _bound_size,
    _compose_interval,
    _evaluate_polynomial,
    _pair_characterizations,
    _size_polynomial,
    _threshold_relation,
    characterize_cha_segment_preconditions,
)

TOPOLOGY_SHA = "2d2659680d850157bf41574257bdae7d593ad166b56eea69dce28c62e54c9391"


def _topology():
    payload = {
        "schema_version": 1,
        "profile": "chagb-r6-pbsa-continuum-v1",
        "atom_ids": [1, 2, 3],
        "atom_names": ["O1", "H2", "H3"],
        "elements": ["O", "H", "H"],
        "atomic_numbers": [8, 1, 1],
        "gaff2_types": ["oh", "ho", "ho"],
        "bonds": [[0, 1, "1"], [0, 2, "1"]],
        "declared_charge_e": 0,
        "source_charges_e": [-0.784666666667, 0.392333333333, 0.392333333333],
        "effective_charges_e": [
            -0.7846666666666666,
            0.3923333333333333,
            0.3923333333333333,
        ],
        "source_charges_sha256": "c4d207dde1cfe0517072e98278a33530d1638ccefc02e40518feae00b8151859",
        "effective_charges_sha256": "2ffd972bf7afcb833e13579730e19b60f2ecdf195e562fb37b1963829fa8b721",
        "source_mol2_sha256": "fc314068cb1cae5232366d8b8a6b56e17f9916a8dc64380db54882a3cc8d8b61",
        "prepared_prmtop_sha256": "9f154d8499f142000e89384530a102ec755e97dd7e5cd3672cd356c83a8f1527",
        "parameter_source_sha256": "1d113ace073e1aa8c0f70ae9f925e87d868e3bf5407406e735eeef4965a651dc",
        "serialization_profile": "ambertools26-prmtop-charge-5e16.8-v1",
        "bondi_radii_angstrom": [1.5, 1.2, 1.2],
        "cha_radii_angstrom": [1.8800000000000001, 1.04, 1.04],
        "lj_rmin_angstrom": [
            1.819999999703896,
            0.30190000004057294,
            0.30190000004057294,
        ],
        "lj_epsilon_kcal_mol": [
            0.09300000011594518,
            0.00469999999789031,
            0.00469999999789031,
        ],
        "content_sha256": TOPOLOGY_SHA,
    }
    return ContinuumChaTopology.from_mapping(
        payload, expected_content_sha256=TOPOLOGY_SHA
    )


def _tensor(rows):
    return torch.tensor(rows, dtype=torch.float64)


def _report(start, trial, *, budget=SegmentPreconditionBudget()):
    return characterize_cha_segment_preconditions(
        _tensor(start),
        _tensor(trial),
        _topology(),
        expected_topology_sha256=TOPOLOGY_SHA,
        budget=budget,
    )


def test_pair_quadratics_include_constant_linear_and_hidden_interior_collision():
    start = [[-1, 0, 0], [0, 0, 0], [0, 2, 0]]
    trial = [[1, 0, 0], [0, 0, 0], [1, 2, 0]]
    report = _report(start, trial)
    pairs = {item.pair: item for item in report.pair_distances}

    assert pairs[(0, 1)].coefficients_angstrom2 == (
        Fraction(1),
        Fraction(-4),
        Fraction(4),
    )
    assert pairs[(0, 1)].minimum_angstrom2 == 0
    assert pairs[(0, 1)].minimum_at_t == Fraction(1, 2)
    assert pairs[(0, 1)].collision_status == "INTERIOR_COLLISION"
    assert pairs[(0, 1)].distinct_center_guard_status == "INTERIOR_GUARD_CROSSING"
    assert pairs[(1, 2)].coefficients_angstrom2 == (
        Fraction(4),
        Fraction(0),
        Fraction(1),
    )
    assert pairs[(1, 2)].maximum_angstrom2 == 5


def test_convex_pair_vertex_and_zero_length_segment_are_exact():
    start = [[0, 0, 0], [2, 1, 0], [0, 4, 0]]
    trial = [[0, 0, 0], [-2, 1, 0], [0, 4, 0]]
    report = _report(start, trial)
    pair = report.pair_distances[0]
    assert pair.coefficients_angstrom2 == (Fraction(5), Fraction(-16), Fraction(16))
    assert pair.minimum_angstrom2 == 1
    assert pair.minimum_at_t == Fraction(1, 2)

    singleton = _report(start, start)
    assert all(item.coefficients_angstrom2[2] == 0 for item in singleton.pair_distances)


def test_exact_guard_equality_and_adjacent_ulp_do_not_silently_pass():
    topology = _topology()
    expanded = float(topology.cha_radii_angstrom[0] + (1.4 - 0.52))
    other = float(topology.cha_radii_angstrom[1] + (1.4 - 0.52))
    boundary = expanded + other - 1.0e-8
    base = [[0, 0, 0], [boundary, 0, 0], [0, 8, 0]]
    equal = _report(base, base)
    assert equal.pair_distances[0].expanded_pair_relation in {
        "TOUCHES_GUARD",
        "INSIDE_OUTER_GUARD",
    }
    exact_threshold = Fraction(17, 5)
    assert (
        _threshold_relation(
            exact_threshold,
            exact_threshold,
            (("exact", exact_threshold),),
        )
        == "TOUCHES_GUARD"
    )

    lower = math.nextafter(boundary, -math.inf)
    upper = math.nextafter(boundary, math.inf)
    adjacent = _report(
        [[0, 0, 0], [lower, 0, 0], [0, 8, 0]],
        [[0, 0, 0], [upper, 0, 0], [0, 8, 0]],
    )
    assert adjacent.pair_distances[0].expanded_pair_relation in {
        "INTERIOR_RELATION_CHANGE",
        "TOUCHES_GUARD",
    }


def test_size_polynomial_degree_and_exact_bernstein_containment():
    report = _report(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [3, 0, 0], [0, 2, 0]],
    )
    polynomial = report.size_polynomial.power_coefficients_angstrom6
    assert len(polynomial) == 7
    assert report.size_polynomial.degree <= 6
    coefficients = _bernstein_coefficients(polynomial, Fraction(0), Fraction(1))
    for value in (
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 4),
        Fraction(1),
    ):
        evaluated = _evaluate_polynomial(polynomial, value)
        assert min(coefficients) <= evaluated <= max(coefficients)


def test_distinct_center_guard_is_independent_of_expanded_radius_relation():
    inside = _report(
        [[0, 0, 0], [5.0e-9, 0, 0], [0, 8, 0]],
        [[0, 0, 0], [5.0e-9, 0, 0], [0, 8, 0]],
    ).pair_distances[0]
    assert inside.collision_status == "NONCOLLIDING"
    assert (
        inside.distinct_center_guard_status == "CONSTANT_INSIDE_DISTINCT_CENTER_GUARD"
    )

    equality = _report(
        [[0, 0, 0], [1.0e-8, 0, 0], [0, 8, 0]],
        [[0, 0, 0], [1.0e-8, 0, 0], [0, 8, 0]],
    ).pair_distances[0]
    assert equality.distinct_center_guard_status == "TOUCHES_DISTINCT_CENTER_GUARD"

    crossing = _report(
        [[0, 0, 0], [5.0e-9, 0, 0], [0, 8, 0]],
        [[0, 0, 0], [2.0e-8, 0, 0], [0, 8, 0]],
    ).pair_distances[0]
    assert crossing.distinct_center_guard_status == "INTERIOR_GUARD_CROSSING"

    collision = _report(
        [[0, 0, 0], [0, 0, 0], [0, 8, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 8, 0]],
    ).pair_distances[0]
    assert collision.collision_status == "CONSTANT_COLLISION"


def test_size_guard_below_above_touching_and_interior_extremum_are_exact():
    budget = SegmentPreconditionBudget(max_depth=4, max_nodes=31)
    switch = Fraction(*float(10.0).as_integer_ratio())
    guard = Fraction(*float(1.0e-8).as_integer_ratio())
    lower = (switch - guard) ** 6
    upper = (switch + guard) ** 6
    below, *_ = _bound_size((lower - 1,) + (Fraction(0),) * 6, budget)
    above, *_ = _bound_size((upper + 1,) + (Fraction(0),) * 6, budget)
    touching, *_ = _bound_size((lower,) + (Fraction(0),) * 6, budget)
    assert below.range_status == "BELOW_SIZE_GUARD"
    assert above.range_status == "ABOVE_SIZE_GUARD"
    assert touching.range_status == "TOUCHES_OR_UNRESOLVED_SIZE_GUARD"

    # p(t)=(t-1/2)^2 has an exact interior minimum at t=1/2.
    polynomial = (Fraction(1, 4), Fraction(-1), Fraction(1)) + (Fraction(0),) * 4
    coefficients = _bernstein_coefficients(polynomial, Fraction(0), Fraction(1))
    assert _evaluate_polynomial(polynomial, Fraction(1, 2)) == 0
    assert min(coefficients) <= 0 <= max(coefficients)


def test_reversal_rigid_transform_and_permutation_preserve_exact_results():
    start = _tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    trial = _tensor([[0, 0, 0], [2, 1, 0], [0, 3, 0]])
    topology = _topology()
    original = characterize_cha_segment_preconditions(
        start, trial, topology, expected_topology_sha256=TOPOLOGY_SHA
    )
    reversed_report = characterize_cha_segment_preconditions(
        trial, start, topology, expected_topology_sha256=TOPOLOGY_SHA
    )
    transform = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=torch.float64)
    translation = _tensor([[2.0, -4.0, 0.5]])
    moved = characterize_cha_segment_preconditions(
        start @ transform.T + translation,
        trial @ transform.T + translation,
        topology,
        expected_topology_sha256=TOPOLOGY_SHA,
    )
    expected_reversed = tuple(
        sum(
            original.size_polynomial.power_coefficients_angstrom6[j] * math.comb(j, k)
            for j in range(k, 7)
        )
        * ((-1) ** k)
        for k in range(7)
    )
    assert (
        reversed_report.size_polynomial.power_coefficients_angstrom6
        == expected_reversed
    )
    assert (
        original.size_polynomial.range_status
        == reversed_report.size_polynomial.range_status
    )
    assert (
        original.size_polynomial.power_coefficients_angstrom6
        == moved.size_polynomial.power_coefficients_angstrom6
    )
    original_pair_ranges = sorted(
        (item.minimum_angstrom2, item.maximum_angstrom2)
        for item in original.pair_distances
    )
    assert original_pair_ranges == sorted(
        (item.minimum_angstrom2, item.maximum_angstrom2)
        for item in reversed_report.pair_distances
    )
    assert original_pair_ranges == sorted(
        (item.minimum_angstrom2, item.maximum_angstrom2)
        for item in moved.pair_distances
    )
    assert [item.expanded_pair_relation for item in original.pair_distances] == [
        item.expanded_pair_relation for item in reversed_report.pair_distances
    ]

    permutation = (2, 0, 1)
    # Canonical remapping is checked with distinct synthetic radii below; the
    # source-bound water topology itself intentionally has non-identical O/H radii.
    assert tuple(sorted(permutation)) == (0, 1, 2)
    exact_start = tuple(
        tuple(Fraction(int(value)) for value in row) for row in start.tolist()
    )
    exact_trial = tuple(
        tuple(Fraction(int(value)) for value in row) for row in trial.tolist()
    )
    remapped_radii = (Fraction(3, 2), Fraction(5, 4), Fraction(7, 4))
    assert _size_polynomial(
        exact_start, exact_trial, remapped_radii
    ) == _size_polynomial(
        tuple(exact_start[index] for index in permutation),
        tuple(exact_trial[index] for index in permutation),
        tuple(remapped_radii[index] for index in permutation),
    )
    original_pairs = {
        item.pair: (
            item.minimum_angstrom2,
            item.maximum_angstrom2,
            item.collision_status,
            item.distinct_center_guard_status,
            item.expanded_pair_relation,
        )
        for item in _pair_characterizations(exact_start, exact_trial, remapped_radii)
    }
    permuted_pairs = {}
    for item in _pair_characterizations(
        tuple(exact_start[index] for index in permutation),
        tuple(exact_trial[index] for index in permutation),
        tuple(remapped_radii[index] for index in permutation),
    ):
        canonical_pair = tuple(sorted(permutation[index] for index in item.pair))
        permuted_pairs[canonical_pair] = (
            item.minimum_angstrom2,
            item.maximum_angstrom2,
            item.collision_status,
            item.distinct_center_guard_status,
            item.expanded_pair_relation,
        )
    assert original_pairs == permuted_pairs


def test_fixed_nonadmission_fields_and_complete_omission_ledger():
    report = _report(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]],
    )
    assert report.dynamic_born_sign_status == "UNRESOLVED"
    assert report.computed_quadrature_smoothness == "UNRESOLVED"
    assert report.optimizer_eligible is False
    assert report.public_admission is False
    assert "weighted-cha-signs" in report.omitted_predicates
    assert "runtime-finite-difference-forces" not in report.implemented_predicates
    with pytest.raises(ValueError):
        replace(report, optimizer_eligible=True)
    serialized = asdict(report)
    assert serialized["arithmetic_scope"].startswith("declared exact-real")
    assert serialized["pair_distances"][0]["units"] == "angstrom^2"
    assert serialized["size_polynomial"]["units"] == "angstrom^6"
    assert "r6-triple-collinearity-1e-9-angstrom" in report.omitted_predicates
    assert "integration-chart-axis-1e-12-dimensionless" in report.omitted_predicates


@pytest.mark.parametrize(
    ("start", "trial"),
    [
        (
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float64),
        ),
        (
            torch.zeros((3, 3), dtype=torch.float32),
            torch.zeros((3, 3), dtype=torch.float32),
        ),
        (
            torch.full((3, 3), float("nan"), dtype=torch.float64),
            torch.zeros((3, 3), dtype=torch.float64),
        ),
    ],
)
def test_invalid_coordinates_fail_before_characterization(start, trial):
    with pytest.raises(SegmentPreconditionValidationError):
        characterize_cha_segment_preconditions(
            start, trial, _topology(), expected_topology_sha256=TOPOLOGY_SHA
        )


def test_topology_external_pin_is_checked_before_coordinate_work():
    with pytest.raises(SegmentPreconditionValidationError, match="topology hash"):
        characterize_cha_segment_preconditions(
            torch.zeros((3, 3), dtype=torch.float64),
            torch.zeros((3, 3), dtype=torch.float64),
            _topology(),
            expected_topology_sha256="0" * 64,
        )


def test_frozen_resource_limits_and_exhaustion_are_fail_closed():
    with pytest.raises(SegmentPreconditionValidationError):
        SegmentPreconditionBudget(max_depth=13)
    with pytest.raises(SegmentPreconditionValidationError):
        SegmentPreconditionBudget(max_nodes=8192)

    report = _report(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [50, 0, 0], [0, 50, 0]],
        budget=SegmentPreconditionBudget(max_depth=0, max_nodes=1),
    )
    assert report.status in {
        "SEGMENT_PRECONDITIONS_CHARACTERIZED",
        "PRECONDITION_BOUNDS_UNRESOLVED",
    }
    assert report.resources.nodes_consumed == 1

    huge = math.ldexp(1.0, 1023)
    with pytest.raises(SegmentPreconditionResourceError):
        _report(
            [[huge, 0, 0], [0, huge, 0], [0, 0, huge]],
            [[-huge, 0, 0], [0, -huge, 0], [0, 0, -huge]],
            budget=SegmentPreconditionBudget(max_integer_bits=100),
        )


def test_checked_arithmetic_tracks_real_growth_and_cross_cancels_locally():
    arithmetic = _CheckedRationalArithmetic(
        SegmentPreconditionBudget(max_integer_bits=128)
    )
    large = (1 << 120) - 3
    assert arithmetic.mul(Fraction(large, 5), Fraction(5, large)) == 1
    assert arithmetic.maximum_tracked_integer_bits_observed == 120

    report = _report(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
    )
    assert report.resources.maximum_tracked_integer_bits_observed > 984
    assert report.resources.comparison_scratch_bound_bits == 16384


def test_interval_composition_does_not_build_unused_degree_seven_power():
    arithmetic = _CheckedRationalArithmetic(
        SegmentPreconditionBudget(max_integer_bits=650)
    )
    polynomial = (Fraction(0),) * 6 + (Fraction(1),)
    composed = _compose_interval(
        polynomial, Fraction(0), Fraction(1, 2**100), arithmetic
    )
    assert composed[-1] == Fraction(1, 2**600)
    assert arithmetic.maximum_tracked_integer_bits_observed == 601


@pytest.mark.parametrize("max_nodes", [5, 7, 9, 15])
def test_pending_stack_budget_yields_partition_not_overflow(max_nodes):
    switch = Fraction(*float(10.0).as_integer_ratio())
    guard = Fraction(*float(1.0e-8).as_integer_ratio())
    touching = (switch - guard) ** 6
    bounded, nodes, _, _ = _bound_size(
        (touching,) + (Fraction(0),) * 6,
        SegmentPreconditionBudget(max_depth=12, max_nodes=max_nodes),
    )
    assert nodes <= max_nodes
    assert bounded.range_status == "TOUCHES_OR_UNRESOLVED_SIZE_GUARD"
    assert bounded.leaves[0].lower_t == 0
    assert bounded.leaves[-1].upper_t == 1
    assert all(
        left.upper_t == right.lower_t
        for left, right in zip(bounded.leaves, bounded.leaves[1:])
    )
