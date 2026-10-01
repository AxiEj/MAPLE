"""RED contracts for the versioned analytic-azimuth R6 contact root."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.torch_continuum_r6_contact_v2 import (
    MAX_CONTACT_V2_ORDER,
    NUMERICAL_PROFILE_ID,
    CapUnionStateV2,
    ContactV2DomainError,
    ContactV2FailureReason,
    _CoveredCapV2,
    _cap_flux_v2,
    _classify_owner_v2,
    _classify_two_caps_v2,
    _direct_complement_flux_v2,
    _full_sphere_flux_v2,
    _validate_node_algebra_v2,
    contact_patch_inverse_cube_v2,
)


def _cap(receiver, axis, cosine):
    return _CoveredCapV2(
        receiver_index=receiver,
        axis=torch.tensor(axis, dtype=torch.float64),
        cosine=torch.tensor(cosine, dtype=torch.float64),
    )


def test_classifier_covers_nested_disjoint_crossing_tangency_ambiguity_and_full_union():
    z = [0.0, 0.0, 1.0]
    minus_z = [0.0, 0.0, -1.0]
    x = [1.0, 0.0, 0.0]
    nested = _classify_two_caps_v2(_cap(1, z, 0.2), _cap(2, z, 0.6), 2.0)
    assert nested.state is CapUnionStateV2.CAP1_CONTAINS_CAP2_NESTED
    assert nested.true_predicates == ("P_1contains2",)

    disjoint = _classify_two_caps_v2(_cap(1, z, 0.5), _cap(2, minus_z, 0.5), 2.0)
    assert disjoint.state is CapUnionStateV2.TWO_DISJOINT_CAPS
    assert disjoint.true_predicates == ("P_disjoint",)

    with pytest.raises(ContactV2DomainError) as crossing:
        _classify_two_caps_v2(_cap(1, z, 0.5), _cap(2, x, 0.5), 2.0)
    assert crossing.value.reason is ContactV2FailureReason.BOUNDARY_CROSSING

    c45 = math.sqrt(0.5)
    with pytest.raises(ContactV2DomainError) as tangency:
        _classify_two_caps_v2(_cap(1, z, c45), _cap(2, x, c45), 2.0)
    assert tangency.value.reason is ContactV2FailureReason.TANGENCY

    with pytest.raises(ContactV2DomainError) as ambiguous:
        _classify_two_caps_v2(_cap(1, z, 0.2), _cap(2, [0.0, 0.0, 1.000001], 0.6), 2.0)
    assert ambiguous.value.reason is ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION

    positions = np.asarray([[0.0, 0.0, 0.0], [1.5, 0.1, 0.0], [-1.5, 0.1, 0.0]])
    axes = []
    cosines = []
    expanded = [1.0, 2.0, 2.0]
    for receiver in (1, 2):
        vector = positions[receiver] - positions[0]
        distance = np.linalg.norm(vector)
        axes.append(vector / distance)
        cosines.append(
            (distance**2 + expanded[0] ** 2 - expanded[receiver] ** 2)
            / (2.0 * expanded[0] * distance)
        )
    full = _classify_two_caps_v2(
        _cap(1, axes[0], cosines[0]),
        _cap(2, axes[1], cosines[1]),
        2.0,
    )
    assert full.state is CapUnionStateV2.FULL_COVERAGE_BY_TWO_LARGE_CAPS
    assert full.true_predicates == ("P_full",)


def test_full_union_counterexample_returns_exact_zero_for_covered_owner():
    probe = float(1.4 - 0.52)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.1, 0.0], [-1.5, 0.1, 0.0]],
        dtype=torch.float64,
    )
    intrinsic = torch.tensor(
        [1.0 - probe, 2.0 - probe, 2.0 - probe], dtype=torch.float64
    )
    result = contact_patch_inverse_cube_v2(positions, intrinsic, order=64)
    owner = result.diagnostics.owners[0]
    assert owner.state is CapUnionStateV2.FULL_COVERAGE_BY_TWO_LARGE_CAPS
    assert owner.exposed_solid_angle_steradian == 0.0
    assert all(
        item.representation == "exact-zero-full-coverage"
        for item in owner.receiver_accumulations[1:]
    )


def test_physical_no_cap_one_cap_disjoint_and_single_ball_containment_states():
    radii = torch.full((3,), 0.5, dtype=torch.float64)
    no_cap = contact_patch_inverse_cube_v2(
        torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=torch.float64
        ),
        radii,
        order=16,
    )
    assert all(
        owner.state is CapUnionStateV2.NO_CAP_FULL_SPHERE
        for owner in no_cap.diagnostics.owners
    )

    one_cap = contact_patch_inverse_cube_v2(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=torch.float64
        ),
        radii,
        order=16,
    )
    assert one_cap.diagnostics.owners[0].state is CapUnionStateV2.ONE_CAP

    expanded = 0.5 + float(1.4 - 0.52)
    distance = 1.2 * expanded
    disjoint = contact_patch_inverse_cube_v2(
        torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0], [-distance, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        radii,
        order=16,
    )
    assert disjoint.diagnostics.owners[0].state is CapUnionStateV2.TWO_DISJOINT_CAPS
    assert all(
        item.representation
        == "receiver-corresponding-direct-complement-minus-other-cap"
        for item in disjoint.diagnostics.owners[0].receiver_accumulations[1:]
    )

    probe = float(1.4 - 0.52)
    containment = contact_patch_inverse_cube_v2(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 6.0, 0.0]], dtype=torch.float64
        ),
        torch.tensor([1.0 - probe, 3.0 - probe, 1.0 - probe], dtype=torch.float64),
        order=16,
    )
    assert (
        containment.diagnostics.owners[0].state
        is CapUnionStateV2.FULL_COVERAGE_BY_ONE_BALL
    )
    covered_owner = containment.diagnostics.owners[0]
    assert covered_owner.covering_receiver_index == 1
    assert covered_owner.cap_receiver_indices == ()


def test_analytic_azimuth_cap_flux_matches_independent_two_dimensional_quadrature():
    positions = torch.tensor(
        [[0.2, -0.1, 0.3], [-0.5, 0.1, -0.2], [4.0, 4.0, 4.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.3, 0.7, 0.5], dtype=torch.float64)
    axis_np = np.asarray([0.1, 0.9, -0.3], dtype=np.float64)
    axis_np /= np.linalg.norm(axis_np)
    cosine = 0.27
    actual, minimum_q = _cap_flux_v2(
        positions,
        radii,
        0,
        1,
        (
            torch.tensor(axis_np, dtype=torch.float64),
            torch.tensor(cosine, dtype=torch.float64),
        ),
        64,
    )

    trial = np.asarray([1.0, 0.0, 0.0])
    basis_u = trial - axis_np * np.dot(trial, axis_np)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis_np, basis_u)
    nodes, weights = np.polynomial.legendre.leggauss(160)
    mu = 0.5 * (1.0 - cosine) * nodes + 0.5 * (1.0 + cosine)
    mu_weights = 0.5 * (1.0 - cosine) * weights
    phi = math.pi * (nodes + 1.0)
    phi_weights = math.pi * weights
    horizontal = np.sqrt(1.0 - mu**2)
    normal = mu[:, None, None] * axis_np + horizontal[:, None, None] * (
        np.cos(phi)[None, :, None] * basis_u + np.sin(phi)[None, :, None] * basis_v
    )
    displacement = (positions[0].numpy() - positions[1].numpy())[None, None, :] + radii[
        0
    ].item() * normal
    separation_squared = np.sum(displacement**2, axis=-1)
    kernel = np.sum(displacement * normal, axis=-1) / separation_squared**3
    expected = (
        radii[0].item() ** 2
        / (4.0 * math.pi)
        * np.sum(mu_weights[:, None] * phi_weights[None, :] * kernel)
    )
    assert minimum_q is not None
    assert minimum_q > 2.0**-20
    assert float(actual) == pytest.approx(expected, abs=2e-12)


def test_near_surface_delta_conditioning_fails_typed_without_clamp():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [-1.000001, 0.0, 0.0], [4.0, 4.0, 4.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64)
    with pytest.raises(ContactV2DomainError) as captured:
        _cap_flux_v2(
            positions,
            radii,
            0,
            1,
            (
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
                torch.tensor(0.0, dtype=torch.float64),
            ),
            128,
            cap_index=1,
        )
    assert captured.value.reason is ContactV2FailureReason.ILL_CONDITIONED_DELTA
    payload = dict(captured.value.raw_margins)
    assert payload["q_min"] == 2.0**-20
    assert payload["owner_index"] == 0
    assert payload["receiver_index"] == 1
    assert payload["cap_index"] == 1
    assert payload["node_index"] >= 0
    assert set(("A", "B2", "Delta", "q", "q_min")) <= payload.keys()


@pytest.mark.parametrize(
    ("reason", "A", "B2", "Delta", "q"),
    (
        (ContactV2FailureReason.NONFINITE_ALGEBRA, float("nan"), 0.0, 1.0, 1.0),
        (ContactV2FailureReason.NONPOSITIVE_DISTANCE_SCALE, -1.0, 0.0, 1.0, 1.0),
        (ContactV2FailureReason.NEGATIVE_B2, 1.0, -0.1, 1.0, 1.0),
        (ContactV2FailureReason.NONPOSITIVE_DELTA, 1.0, 0.0, 0.0, 0.0),
        (ContactV2FailureReason.ILL_CONDITIONED_DELTA, 1.0, 0.0, 1e-8, 1e-8),
    ),
)
def test_every_typed_node_failure_retains_complete_indexed_algebra_payload(
    reason, A, B2, Delta, q
):
    with pytest.raises(ContactV2DomainError) as captured:
        _validate_node_algebra_v2(
            torch.tensor([A], dtype=torch.float64),
            torch.tensor([B2], dtype=torch.float64),
            torch.tensor([Delta], dtype=torch.float64),
            torch.tensor([q], dtype=torch.float64),
            owner_index=2,
            receiver_index=1,
            cap_index=0,
        )
    assert captured.value.reason is reason
    payload = dict(captured.value.raw_margins)
    assert payload["owner_index"] == 2
    assert payload["receiver_index"] == 1
    assert payload["cap_index"] == 0
    assert payload["node_index"] == 0
    assert payload["A"] == pytest.approx(A, nan_ok=True)
    assert payload["B2"] == B2
    assert payload["Delta"] == Delta
    assert payload["q"] == q
    assert payload["q_min"] == 2.0**-20


def test_direct_complement_avoids_known_near_receiver_cancellation():
    positions = torch.tensor(
        [
            [0.010538813934279549, 0.39164232041464536, 1.8309399258250978e-16],
            [0.768508800461066, -0.19237452253773285, -4.846003055979318e-18],
            [-0.77772761439535, -0.1507877978769215, -8.236316583063313e-17],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.88, 1.04, 1.04], dtype=torch.float64)
    result = contact_patch_inverse_cube_v2(positions, radii, order=64)
    h_owner = result.diagnostics.owners[1]
    assert h_owner.state is CapUnionStateV2.CAP1_CONTAINS_CAP2_NESTED
    assert h_owner.receiver_accumulations[0].representation == "direct-complement"
    full_sphere = h_owner.receiver_accumulations[0].full_sphere_diagnostic
    assert full_sphere is not None
    assert full_sphere > 245.0
    assert h_owner.receiver_accumulations[0].minimum_q == 1.0
    assert result.diagnostics.numerical_profile_id == NUMERICAL_PROFILE_ID

    union = _classify_owner_v2(positions, radii, 1)
    cap = union.union_cap
    assert cap is not None
    full = _full_sphere_flux_v2(positions, radii, 1, 0)

    def preregistered_cap_value(axis, cosine):
        nodes, weights = np.polynomial.legendre.leggauss(1024)
        mu = 0.5 * (1.0 - cosine) * nodes + 0.5 * (1.0 + cosine)
        mu_weights = 0.5 * (1.0 - cosine) * weights
        radius = radii[1].item()
        displacement = (positions[1] - positions[0]).numpy()
        t = float(np.dot(displacement, axis))
        perpendicular = displacement - t * axis
        p2 = float(np.dot(perpendicular, perpendicular))
        d2 = float(np.dot(displacement, displacement))
        A = d2 + radius**2 + 2.0 * radius * mu * t
        B2 = 4.0 * radius**2 * (1.0 - mu**2) * p2
        delta = A**2 - B2
        I2 = 2.0 * math.pi * A / delta**1.5
        I3 = math.pi * (2.0 * A**2 + B2) / delta**2.5
        value = (
            radius / (8.0 * math.pi) * np.sum(mu_weights * (I2 + (radius**2 - d2) * I3))
        )
        return float(value), float(np.min(delta / A**2))

    covered, covered_q = preregistered_cap_value(cap.axis.numpy(), cap.cosine.item())
    direct, direct_q = preregistered_cap_value(-cap.axis.numpy(), -cap.cosine.item())
    production_direct, production_direct_q = _direct_complement_flux_v2(
        positions, radii, 1, 0, cap, 64
    )
    forbidden = full - covered
    absolute_loss = abs(float(forbidden) - direct)
    relative_loss = absolute_loss / abs(direct)
    assert float(full) == pytest.approx(245.88386030578712, abs=1e-12)
    assert covered == pytest.approx(245.8804998069856, abs=1e-12)
    assert float(forbidden) == pytest.approx(0.003360498801527001, abs=1e-13)
    assert direct == pytest.approx(0.0033604994943907166, abs=1e-15)
    assert absolute_loss == pytest.approx(6.928637156199124e-10, rel=1e-4)
    assert relative_loss == pytest.approx(2.0617878883077583e-7, rel=1e-4)
    assert covered_q == 1.0
    assert direct_q == 1.0
    assert float(production_direct) == pytest.approx(direct, abs=1e-15)
    assert production_direct_q == 1.0


def test_direct_complement_is_not_rejected_by_unused_singular_full_sphere_diagnostic():
    result = contact_patch_inverse_cube_v2(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64),
        order=32,
    )
    receiver = result.diagnostics.owners[0].receiver_accumulations[1]
    assert receiver.representation == "direct-complement"
    assert receiver.full_sphere_diagnostic is None
    assert torch.isfinite(result.inverse_cube_per_angstrom3).all()


def test_contact_scalar_is_rotation_permutation_covariant_with_live_gradient():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    radii = torch.tensor([1.88, 1.04, 1.04], dtype=torch.float64)
    base = contact_patch_inverse_cube_v2(positions, radii, order=64)
    scalar = base.inverse_cube_per_angstrom3.sum()
    gradient = torch.autograd.grad(scalar, positions)[0]
    assert torch.isfinite(gradient).all()

    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    moved_positions = (positions.detach() @ rotation.T).requires_grad_(True)
    moved = contact_patch_inverse_cube_v2(moved_positions, radii, order=64)
    moved_gradient = torch.autograd.grad(
        moved.inverse_cube_per_angstrom3.sum(), moved_positions
    )[0]
    tolerance = 4096 * torch.finfo(torch.float64).eps
    assert torch.allclose(
        moved.inverse_cube_per_angstrom3,
        base.inverse_cube_per_angstrom3.detach(),
        atol=tolerance,
        rtol=0.0,
    )
    assert torch.allclose(
        moved_gradient, gradient @ rotation.T, atol=tolerance, rtol=0.0
    )

    permutation = [0, 2, 1]
    permuted = contact_patch_inverse_cube_v2(
        positions.detach()[permutation], radii[permutation], order=64
    )
    assert torch.allclose(
        permuted.inverse_cube_per_angstrom3,
        base.inverse_cube_per_angstrom3.detach()[permutation],
        atol=tolerance,
        rtol=0.0,
    )


def test_fixed_dense_rotation_preserves_full_vector_jacobian():
    positions = torch.tensor(
        [
            [0.010538813934279549, 0.39164232041464536, 1.8309399258250978e-16],
            [0.768508800461066, -0.19237452253773285, -4.846003055979318e-18],
            [-0.77772761439535, -0.1507877978769215, -8.236316583063313e-17],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.88, 1.04, 1.04], dtype=torch.float64)
    rotation = torch.tensor(
        [[0.36, -0.48, 0.8], [0.8, 0.60, 0.0], [-0.48, 0.64, 0.60]],
        dtype=torch.float64,
    )

    def values(coordinates):
        return contact_patch_inverse_cube_v2(
            coordinates, radii, order=64
        ).inverse_cube_per_angstrom3

    base_values = values(positions)
    base_jacobian = torch.autograd.functional.jacobian(values, positions)
    rotated_positions = positions @ rotation.T
    rotated_values = values(rotated_positions)
    rotated_jacobian = torch.autograd.functional.jacobian(values, rotated_positions)
    expected_jacobian = base_jacobian @ rotation.T
    scale = max(
        1.0,
        float(base_values.abs().max()),
        float(base_jacobian.abs().max()),
        float(positions.abs().max()),
        float(radii.max()),
    )
    tolerance = 4096 * torch.finfo(torch.float64).eps * scale
    assert torch.allclose(rotated_values, base_values, atol=tolerance, rtol=0.0)
    assert torch.allclose(rotated_jacobian, expected_jacobian, atol=tolerance, rtol=0.0)


def test_h_exchange_preserves_full_jacobian_classifier_and_invariant_margins():
    positions = torch.tensor(
        [
            [0.010538813934279549, 0.39164232041464536, 1.8309399258250978e-16],
            [0.768508800461066, -0.19237452253773285, -4.846003055979318e-18],
            [-0.77772761439535, -0.1507877978769215, -8.236316583063313e-17],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.88, 1.04, 1.04], dtype=torch.float64)

    def evaluate(coordinates, local_radii):
        return contact_patch_inverse_cube_v2(
            coordinates, local_radii, order=64
        ).inverse_cube_per_angstrom3

    base = contact_patch_inverse_cube_v2(positions, radii, order=64)
    base_jacobian = torch.autograd.functional.jacobian(
        lambda coordinates: evaluate(coordinates, radii), positions
    )
    permutation = [0, 2, 1]
    exchanged_positions = positions[permutation]
    exchanged_radii = radii[permutation]
    exchanged = contact_patch_inverse_cube_v2(
        exchanged_positions, exchanged_radii, order=64
    )
    exchanged_jacobian = torch.autograd.functional.jacobian(
        lambda coordinates: evaluate(coordinates, exchanged_radii),
        exchanged_positions,
    )
    expected_jacobian = base_jacobian[permutation][:, permutation, :]
    scale = max(1.0, float(base_jacobian.abs().max()), float(radii.max()))
    tolerance = 4096 * torch.finfo(torch.float64).eps * scale
    assert torch.allclose(
        exchanged.inverse_cube_per_angstrom3,
        base.inverse_cube_per_angstrom3[permutation],
        atol=tolerance,
        rtol=0.0,
    )
    assert torch.allclose(
        exchanged_jacobian, expected_jacobian, atol=tolerance, rtol=0.0
    )
    assert [owner.state for owner in exchanged.diagnostics.owners] == [
        base.diagnostics.owners[index].state for index in permutation
    ]
    invariant_keys = (
        "g",
        "s_beta",
        "chi",
        "m_disjoint",
        "m_full_union",
        "m_large",
    )
    before = dict(base.diagnostics.owners[0].raw_margins)
    after = dict(exchanged.diagnostics.owners[0].raw_margins)
    for key in invariant_keys:
        assert after[key] == pytest.approx(before[key], abs=tolerance, rel=0.0)


@pytest.mark.parametrize("order", [True, 7, 129, 8.0])
def test_input_and_order_resource_contract_fails_closed(order):
    positions = torch.zeros((3, 3), dtype=torch.float64)
    radii = torch.ones(3, dtype=torch.float64)
    with pytest.raises((TypeError, ValueError, ContactV2DomainError)):
        contact_patch_inverse_cube_v2(positions, radii, order=order)
    assert MAX_CONTACT_V2_ORDER == 128


def test_cpu_float64_exactly_three_sites_are_mandatory():
    with pytest.raises((TypeError, ValueError)):
        contact_patch_inverse_cube_v2(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
        )
    with pytest.raises((TypeError, ValueError)):
        contact_patch_inverse_cube_v2(
            torch.zeros((3, 3), dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        )
