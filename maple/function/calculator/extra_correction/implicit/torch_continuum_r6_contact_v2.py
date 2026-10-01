"""Versioned analytic-azimuth contact root for three-site R6 derivatives.

This numerical representation preserves the R6 contact-surface functional but
does not sort global-axis latitude breakpoints.  Cap classification is
basis-free and fail-closed; accepted nonowner integrals use the preregistered
physical direct-complement representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from .torch_continuum_ses_geometry import (
    PROBE_ANGSTROM,
    R6_GEOMETRY_EVENT_MARGIN_ANGSTROM,
)
from .torch_sphere_union_geometry import legendre_rule

if TYPE_CHECKING:
    from torch import Tensor


NUMERICAL_PROFILE_ID = (
    "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement"
)
MAX_CONTACT_V2_ORDER = 128
_FP64_EPSILON = 2.220446049250313e-16
_FP_MARGIN = 4096.0 * _FP64_EPSILON
_Q_MIN = 2.0**-20


class ContactV2FailureReason(str, Enum):
    INVALID_INPUT = "invalid-input"
    TANGENCY = "tangency"
    BOUNDARY_CROSSING = "boundary-crossing"
    AMBIGUOUS_CLASSIFICATION = "ambiguous-classification"
    UNSUPPORTED_RECEIVER_CAP_MAPPING = "unsupported-receiver-cap-mapping"
    NONFINITE_ALGEBRA = "nonfinite-algebra"
    NONPOSITIVE_DISTANCE_SCALE = "nonpositive-distance-scale"
    NEGATIVE_B2 = "negative-b2"
    NONPOSITIVE_DELTA = "nonpositive-delta"
    ILL_CONDITIONED_DELTA = "ill-conditioned-delta"
    RECEIVER_ON_OWNER_SURFACE = "receiver-on-owner-surface"


class ContactV2DomainError(ValueError):
    """Typed fail-closed contact-v2 classification or conditioning error."""

    def __init__(
        self,
        reason: ContactV2FailureReason,
        message: str,
        raw_margins: tuple[tuple[str, float], ...] = (),
    ):
        super().__init__(f"R6 contact-v2 rejected ({reason.value}): {message}")
        self.reason = reason
        self.raw_margins = tuple(raw_margins)


class CapUnionStateV2(str, Enum):
    NO_CAP_FULL_SPHERE = "no-cap-full-sphere"
    ONE_CAP = "one-cap"
    FULL_COVERAGE_BY_ONE_BALL = "full-coverage-by-one-ball"
    CAP1_CONTAINS_CAP2_NESTED = "cap1-contains-cap2-nested"
    CAP2_CONTAINS_CAP1_NESTED = "cap2-contains-cap1-nested"
    TWO_DISJOINT_CAPS = "two-disjoint-caps"
    FULL_COVERAGE_BY_TWO_LARGE_CAPS = "full-coverage-by-two-large-caps"


@dataclass(frozen=True)
class _CoveredCapV2:
    receiver_index: int
    axis: Tensor
    cosine: Tensor


@dataclass(frozen=True)
class _CapUnionV2:
    state: CapUnionStateV2
    caps: tuple[_CoveredCapV2, ...]
    union_cap: _CoveredCapV2 | None
    true_predicates: tuple[str, ...]
    raw_margins: tuple[tuple[str, float], ...]
    covering_receiver_index: int | None = None


@dataclass(frozen=True)
class ReceiverAccumulationV2:
    receiver_index: int
    representation: str
    minimum_q: float | None
    full_sphere_diagnostic: float | None


@dataclass(frozen=True)
class OwnerContactV2Diagnostics:
    owner_index: int
    state: CapUnionStateV2
    cap_receiver_indices: tuple[int, ...]
    covering_receiver_index: int | None
    true_predicates: tuple[str, ...]
    exposed_solid_angle_steradian: float
    receiver_accumulations: tuple[ReceiverAccumulationV2, ...]
    raw_margins: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ContactFluxV2Diagnostics:
    numerical_profile_id: str
    quadrature_order: int
    owners: tuple[OwnerContactV2Diagnostics, ...]
    minimum_q: float | None


@dataclass(frozen=True)
class ContactFluxV2Result:
    inverse_cube_per_angstrom3: Tensor
    diagnostics: ContactFluxV2Diagnostics
    quadrature_order: int
    complete_ses: bool = False


def _float(value) -> float:
    return float(value.detach().cpu())


def _margins(**values) -> tuple[tuple[str, float], ...]:
    return tuple(
        (name, _float(value) if hasattr(value, "detach") else float(value))
        for name, value in values.items()
    )


def _guards(scale_angstrom: float) -> tuple[float, float, float, float]:
    tau = max(_FP_MARGIN, R6_GEOMETRY_EVENT_MARGIN_ANGSTROM / max(1.0, scale_angstrom))
    return tau, 2.0 * tau, 16.0 * tau, 8.0 * tau


def _raise(
    reason: ContactV2FailureReason,
    message: str,
    raw_margins: tuple[tuple[str, float], ...] = (),
):
    raise ContactV2DomainError(reason, message, raw_margins)


def _validate_cap(cap: _CoveredCapV2, *, tau_axis: float) -> None:
    import torch

    if type(cap.receiver_index) is not int:
        _raise(
            ContactV2FailureReason.INVALID_INPUT,
            "cap receiver index must be an integer",
        )
    if (
        not isinstance(cap.axis, torch.Tensor)
        or cap.axis.dtype != torch.float64
        or cap.axis.device.type != "cpu"
        or cap.axis.shape != (3,)
        or not isinstance(cap.cosine, torch.Tensor)
        or cap.cosine.dtype != torch.float64
        or cap.cosine.device != cap.axis.device
        or cap.cosine.shape != ()
        or not bool(torch.isfinite(cap.axis).all() and torch.isfinite(cap.cosine))
    ):
        _raise(
            ContactV2FailureReason.INVALID_INPUT,
            "cap descriptor must be finite CPU float64",
        )
    norm_residual = torch.abs(torch.dot(cap.axis, cap.axis) - 1.0)
    threshold_clearance = 1.0 - torch.abs(cap.cosine)
    raw = _margins(
        axis_unit_residual=norm_residual,
        threshold_clearance=threshold_clearance,
        tau_axis=tau_axis,
    )
    if bool(norm_residual > tau_axis) or bool(threshold_clearance <= tau_axis):
        _raise(
            ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION,
            "cap axis or threshold is outside its preregistered guard",
            raw,
        )


def _classify_two_caps_v2(
    cap1: _CoveredCapV2, cap2: _CoveredCapV2, scale_angstrom: float
) -> _CapUnionV2:
    """Apply the exact-one preregistered two-cap decision table."""
    import torch

    _tau, tau_axis, tau_chi, tau_angle = _guards(float(scale_angstrom))
    _validate_cap(cap1, tau_axis=tau_axis)
    _validate_cap(cap2, tau_axis=tau_axis)
    if cap1.receiver_index == cap2.receiver_index:
        _raise(
            ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION,
            "two caps cannot name the same physical receiver",
        )
    a1, a2 = cap1.axis, cap2.axis
    c1, c2 = cap1.cosine, cap2.cosine
    s1 = torch.sqrt(1.0 - c1.square())
    s2 = torch.sqrt(1.0 - c2.square())
    g = torch.dot(a1, a2)
    s_beta = torch.linalg.vector_norm(torch.linalg.cross(a1, a2))
    beta = torch.atan2(s_beta, g)
    alpha1 = torch.atan2(s1, c1)
    alpha2 = torch.atan2(s2, c2)
    chi = (1.0 - c1.square()) * (1.0 - c2.square()) - (g - c1 * c2).square()
    values = {
        "g": g,
        "s_beta": s_beta,
        "chi": chi,
        "axis2_in_cap1": g - c1,
        "axis1_in_cap2": g - c2,
        "antipode2_in_cap1": -g - c1,
        "antipode1_in_cap2": -g - c2,
        "boundary2_inside_cap1": c2 * g - s2 * s_beta - c1,
        "boundary1_inside_cap2": c1 * g - s1 * s_beta - c2,
        "m_disjoint": beta - alpha1 - alpha2,
        "m_1contains2": alpha1 - beta - alpha2,
        "m_2contains1": alpha2 - beta - alpha1,
        "m_full_union": alpha1 + alpha2 + beta - 2.0 * math.pi,
        "m_large": c1 + c2,
    }
    raw = _margins(
        **values,
        tau_axis=tau_axis,
        tau_chi=tau_chi,
        tau_angle=tau_angle,
    )
    if not all(bool(torch.isfinite(value)) for value in values.values()):
        _raise(ContactV2FailureReason.NONFINITE_ALGEBRA, "nonfinite cap invariant", raw)
    if bool(chi > tau_chi):
        _raise(
            ContactV2FailureReason.BOUNDARY_CROSSING,
            "two covered-cap boundaries cross",
            raw,
        )
    angular = (
        values["m_disjoint"],
        values["m_1contains2"],
        values["m_2contains1"],
        values["m_full_union"],
    )
    if bool(torch.abs(chi) <= tau_chi) or any(
        bool(torch.abs(value) <= tau_angle) for value in angular
    ):
        _raise(
            ContactV2FailureReason.TANGENCY,
            "cap relation lies in a tangency guard",
            raw,
        )
    if not bool(chi < -tau_chi):
        _raise(
            ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION,
            "cap discriminant has no admitted sign",
            raw,
        )
    predicates = {
        "P_1contains2": bool(values["m_1contains2"] > tau_angle)
        and bool(values["axis2_in_cap1"] > tau_axis)
        and bool(values["boundary2_inside_cap1"] > tau_axis),
        "P_2contains1": bool(values["m_2contains1"] > tau_angle)
        and bool(values["axis1_in_cap2"] > tau_axis)
        and bool(values["boundary1_inside_cap2"] > tau_axis),
        "P_disjoint": bool(values["m_disjoint"] > tau_angle)
        and bool(values["axis2_in_cap1"] < -tau_axis)
        and bool(values["axis1_in_cap2"] < -tau_axis)
        and bool(values["m_large"] > tau_axis),
        "P_full": bool(values["m_full_union"] > tau_angle)
        and bool(values["antipode2_in_cap1"] > tau_axis)
        and bool(values["antipode1_in_cap2"] > tau_axis)
        and bool(values["m_large"] < -tau_axis),
    }
    true_predicates = tuple(name for name, selected in predicates.items() if selected)
    if len(true_predicates) != 1:
        _raise(
            ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION,
            f"expected exactly one cap-union predicate, got {true_predicates}",
            raw,
        )
    selected = true_predicates[0]
    if selected == "P_1contains2":
        state = CapUnionStateV2.CAP1_CONTAINS_CAP2_NESTED
        union_cap = cap1
    elif selected == "P_2contains1":
        state = CapUnionStateV2.CAP2_CONTAINS_CAP1_NESTED
        union_cap = cap2
    elif selected == "P_disjoint":
        state = CapUnionStateV2.TWO_DISJOINT_CAPS
        union_cap = None
    else:
        state = CapUnionStateV2.FULL_COVERAGE_BY_TWO_LARGE_CAPS
        union_cap = None
    return _CapUnionV2(state, (cap1, cap2), union_cap, true_predicates, raw)


def _classify_owner_v2(positions, intrinsic, owner: int) -> _CapUnionV2:
    import torch

    expanded = intrinsic + PROBE_ANGSTROM
    distances = [
        torch.linalg.vector_norm(positions[other] - positions[owner])
        for other in range(3)
        if other != owner
    ]
    scale = max(
        1.0,
        *(_float(value) for value in expanded),
        *(_float(value) for value in distances),
    )
    caps = []
    raw: list[tuple[str, float]] = []
    for other in range(3):
        if other == owner:
            continue
        vector = positions[other] - positions[owner]
        distance = torch.linalg.vector_norm(vector)
        if bool(distance <= R6_GEOMETRY_EVENT_MARGIN_ANGSTROM):
            _raise(
                ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION,
                "coincident centers have no cap axis",
                ((f"owner_{owner}_receiver_{other}_distance", _float(distance)),),
            )
        external = distance - expanded[owner] - expanded[other]
        owner_contains = expanded[owner] - distance - expanded[other]
        other_contains = expanded[other] - distance - expanded[owner]
        inner = distance - torch.abs(expanded[owner] - expanded[other])
        pair_raw = _margins(
            **{
                f"owner_{owner}_receiver_{other}_external": external,
                f"owner_{owner}_receiver_{other}_owner_contains": owner_contains,
                f"owner_{owner}_receiver_{other}_other_contains": other_contains,
                f"owner_{owner}_receiver_{other}_inner": inner,
            }
        )
        raw.extend(pair_raw)
        if bool(other_contains > R6_GEOMETRY_EVENT_MARGIN_ANGSTROM):
            return _CapUnionV2(
                CapUnionStateV2.FULL_COVERAGE_BY_ONE_BALL,
                (),
                None,
                (),
                tuple(raw),
                other,
            )
        if bool(external > R6_GEOMETRY_EVENT_MARGIN_ANGSTROM) or bool(
            owner_contains > R6_GEOMETRY_EVENT_MARGIN_ANGSTROM
        ):
            continue
        if bool(torch.abs(external) <= R6_GEOMETRY_EVENT_MARGIN_ANGSTROM) or bool(
            torch.abs(inner) <= R6_GEOMETRY_EVENT_MARGIN_ANGSTROM
        ):
            _raise(
                ContactV2FailureReason.TANGENCY,
                "expanded balls are tangent",
                tuple(raw),
            )
        if not bool(external < -R6_GEOMETRY_EVENT_MARGIN_ANGSTROM) or not bool(
            inner > R6_GEOMETRY_EVENT_MARGIN_ANGSTROM
        ):
            _raise(
                ContactV2FailureReason.AMBIGUOUS_CLASSIFICATION,
                "expanded-ball relation has no admitted cap state",
                tuple(raw),
            )
        axis = vector / distance
        cosine = (
            distance.square() + expanded[owner].square() - expanded[other].square()
        ) / (2.0 * expanded[owner] * distance)
        cap = _CoveredCapV2(other, axis, cosine)
        _tau, tau_axis, _tau_chi, _tau_angle = _guards(scale)
        _validate_cap(cap, tau_axis=tau_axis)
        caps.append(cap)
    if not caps:
        return _CapUnionV2(CapUnionStateV2.NO_CAP_FULL_SPHERE, (), None, (), tuple(raw))
    if len(caps) == 1:
        return _CapUnionV2(CapUnionStateV2.ONE_CAP, (caps[0],), caps[0], (), tuple(raw))
    classified = _classify_two_caps_v2(caps[0], caps[1], scale)
    return _CapUnionV2(
        classified.state,
        classified.caps,
        classified.union_cap,
        classified.true_predicates,
        tuple(raw) + classified.raw_margins,
    )


def _node_failure_payload_v2(
    A,
    B2,
    delta,
    q,
    node_index: int,
    *,
    owner_index: int,
    receiver_index: int,
    cap_index: int,
):
    return (
        ("owner_index", float(owner_index)),
        ("receiver_index", float(receiver_index)),
        ("cap_index", float(cap_index)),
        ("node_index", float(node_index)),
        ("A", _float(A[node_index])),
        ("B2", _float(B2[node_index])),
        ("Delta", _float(delta[node_index])),
        ("q", _float(q[node_index])),
        ("q_min", _Q_MIN),
    )


def _validate_node_algebra_v2(
    A,
    B2,
    delta,
    q,
    *,
    owner_index: int,
    receiver_index: int,
    cap_index: int,
) -> float:
    """Validate analytic nodes and retain the complete first offending tuple."""
    import torch

    if not (
        isinstance(A, torch.Tensor)
        and isinstance(B2, torch.Tensor)
        and isinstance(delta, torch.Tensor)
        and isinstance(q, torch.Tensor)
        and A.shape == B2.shape == delta.shape == q.shape
        and A.ndim == 1
        and A.numel() > 0
    ):
        raise TypeError("contact-v2 node algebra must use nonempty equal 1-D tensors")
    checks = (
        (
            ContactV2FailureReason.NONFINITE_ALGEBRA,
            ~(
                torch.isfinite(A)
                & torch.isfinite(B2)
                & torch.isfinite(delta)
                & torch.isfinite(q)
            ),
            "nonfinite analytic-azimuth root",
        ),
        (
            ContactV2FailureReason.NONPOSITIVE_DISTANCE_SCALE,
            A <= 0.0,
            "A must be positive",
        ),
        (
            ContactV2FailureReason.NEGATIVE_B2,
            B2 < 0.0,
            "B2 must be nonnegative without clamp",
        ),
        (
            ContactV2FailureReason.NONPOSITIVE_DELTA,
            delta <= 0.0,
            "Delta must be positive",
        ),
        (
            ContactV2FailureReason.ILL_CONDITIONED_DELTA,
            q < _Q_MIN,
            "Delta/A^2 lies below 2^-20",
        ),
    )
    for reason, mask, message in checks:
        offending = torch.nonzero(mask, as_tuple=False)
        if offending.numel():
            node_index = int(offending[0, 0].item())
            _raise(
                reason,
                message,
                _node_failure_payload_v2(
                    A,
                    B2,
                    delta,
                    q,
                    node_index,
                    owner_index=owner_index,
                    receiver_index=receiver_index,
                    cap_index=cap_index,
                ),
            )
    return _float(q.min())


def _cap_flux_v2(
    positions,
    intrinsic,
    owner: int,
    receiver: int,
    cap,
    order: int,
    *,
    cap_index: int = -1,
):
    """Integrate one physical cap directly with analytic full-azimuth roots."""
    import torch

    radius = intrinsic[owner]
    displacement = positions[owner] - positions[receiver]
    d2 = torch.dot(displacement, displacement)
    axis, cosine = cap
    nodes, weights = legendre_rule(order, "cpu")
    half = 0.5 * (1.0 - cosine)
    mu = 0.5 * (1.0 + cosine) + half * nodes
    t = torch.dot(displacement, axis)
    perpendicular = displacement - t * axis
    p2 = torch.dot(perpendicular, perpendicular)
    A = d2 + radius.square() + 2.0 * radius * mu * t
    B2 = 4.0 * radius.square() * (1.0 - mu.square()) * p2
    delta = A.square() - B2
    q = delta / A.square()
    minimum_q = _validate_node_algebra_v2(
        A,
        B2,
        delta,
        q,
        owner_index=owner,
        receiver_index=receiver,
        cap_index=cap_index,
    )
    I2 = 2.0 * math.pi * A / delta.pow(1.5)
    I3 = math.pi * (2.0 * A.square() + B2) / delta.pow(2.5)
    integrand = I2 + (radius.square() - d2) * I3
    flux = radius / (8.0 * math.pi) * half * (weights * integrand).sum()
    if not bool(torch.isfinite(flux)):
        _raise(ContactV2FailureReason.NONFINITE_ALGEBRA, "cap flux is nonfinite")
    return flux, minimum_q


def _direct_complement_flux_v2(
    positions, intrinsic, owner: int, receiver: int, cap: _CoveredCapV2, order: int
):
    return _cap_flux_v2(
        positions,
        intrinsic,
        owner,
        receiver,
        (-cap.axis, -cap.cosine),
        order,
        cap_index=cap.receiver_index,
    )


def _full_sphere_flux_v2(positions, intrinsic, owner: int, receiver: int):
    import torch

    radius = intrinsic[owner]
    displacement = positions[owner] - positions[receiver]
    distance = torch.linalg.vector_norm(displacement)
    if bool(torch.abs(distance - radius) <= R6_GEOMETRY_EVENT_MARGIN_ANGSTROM):
        _raise(
            ContactV2FailureReason.RECEIVER_ON_OWNER_SURFACE,
            "full-sphere receiver lies within the surface separation guard",
            _margins(distance=distance, radius=radius),
        )
    denominator = radius.square() - torch.dot(displacement, displacement)
    flux = radius.pow(3) / denominator.pow(3)
    if not bool(torch.isfinite(flux)):
        _raise(
            ContactV2FailureReason.NONFINITE_ALGEBRA, "full-sphere flux is nonfinite"
        )
    return flux


def _full_sphere_diagnostic_v2(positions, intrinsic, owner: int, receiver: int):
    """Return a detached diagnostic without constraining a direct-complement path."""
    radius = _float(intrinsic[owner])
    displacement = positions[owner].detach() - positions[receiver].detach()
    distance_squared = _float((displacement * displacement).sum())
    distance = math.sqrt(distance_squared)
    if abs(distance - radius) <= R6_GEOMETRY_EVENT_MARGIN_ANGSTROM:
        return None
    denominator = radius * radius - distance_squared
    value = radius**3 / denominator**3
    return value if math.isfinite(value) else None


def _solid_angle_v2(union: _CapUnionV2, template):
    if union.state is CapUnionStateV2.NO_CAP_FULL_SPHERE:
        return template.new_tensor(4.0 * math.pi)
    if union.state in {
        CapUnionStateV2.FULL_COVERAGE_BY_ONE_BALL,
        CapUnionStateV2.FULL_COVERAGE_BY_TWO_LARGE_CAPS,
    }:
        return template.new_zeros(())
    if union.state in {
        CapUnionStateV2.ONE_CAP,
        CapUnionStateV2.CAP1_CONTAINS_CAP2_NESTED,
        CapUnionStateV2.CAP2_CONTAINS_CAP1_NESTED,
    }:
        assert union.union_cap is not None
        return 2.0 * math.pi * (1.0 + union.union_cap.cosine)
    return 2.0 * math.pi * sum(cap.cosine for cap in union.caps)


def _validate_inputs(positions, intrinsic, order: int) -> None:
    import torch

    if (
        not isinstance(positions, torch.Tensor)
        or not isinstance(intrinsic, torch.Tensor)
        or positions.dtype != torch.float64
        or intrinsic.dtype != torch.float64
        or positions.device.type != "cpu"
        or intrinsic.device != positions.device
        or positions.shape != (3, 3)
        or intrinsic.shape != (3,)
    ):
        raise TypeError(
            "R6 contact-v2 requires CPU float64 [3,3] positions and [3] radii"
        )
    if not bool(torch.isfinite(positions).all() and torch.isfinite(intrinsic).all()):
        raise ValueError("R6 contact-v2 inputs must be finite")
    if not bool((intrinsic > 0.0).all()):
        raise ValueError("R6 contact-v2 intrinsic radii must be positive")
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError("R6 contact-v2 order must be an integer")
    if not 8 <= order <= MAX_CONTACT_V2_ORDER:
        raise ValueError(
            f"R6 contact-v2 order must be between 8 and {MAX_CONTACT_V2_ORDER}"
        )


def contact_patch_inverse_cube_v2(
    positions_angstrom: Tensor,
    intrinsic_radii_angstrom: Tensor,
    *,
    order: int = 64,
) -> ContactFluxV2Result:
    """Return three-site contact inverse-cube flux with one live Torch graph."""
    import torch

    _validate_inputs(positions_angstrom, intrinsic_radii_angstrom, order)
    flux = torch.zeros(3, dtype=torch.float64, device=positions_angstrom.device)
    owner_diagnostics = []
    all_q = []
    for owner in range(3):
        union = _classify_owner_v2(positions_angstrom, intrinsic_radii_angstrom, owner)
        solid_angle = _solid_angle_v2(union, positions_angstrom)
        accumulations = []
        for receiver in range(3):
            if receiver == owner:
                value = solid_angle / (
                    4.0 * math.pi * intrinsic_radii_angstrom[owner].pow(3)
                )
                representation = "owner-exact-solid-angle"
                minimum_q = None
                full_diagnostic = 1.0 / _float(intrinsic_radii_angstrom[owner].pow(3))
            elif union.state in {
                CapUnionStateV2.FULL_COVERAGE_BY_ONE_BALL,
                CapUnionStateV2.FULL_COVERAGE_BY_TWO_LARGE_CAPS,
            }:
                value = positions_angstrom.new_zeros(())
                representation = "exact-zero-full-coverage"
                minimum_q = None
                full_diagnostic = None
            elif union.state is CapUnionStateV2.NO_CAP_FULL_SPHERE:
                value = _full_sphere_flux_v2(
                    positions_angstrom,
                    intrinsic_radii_angstrom,
                    owner,
                    receiver,
                )
                representation = "full-sphere-no-cap"
                minimum_q = None
                full_diagnostic = _float(value)
            elif union.state in {
                CapUnionStateV2.ONE_CAP,
                CapUnionStateV2.CAP1_CONTAINS_CAP2_NESTED,
                CapUnionStateV2.CAP2_CONTAINS_CAP1_NESTED,
            }:
                assert union.union_cap is not None
                value, minimum_q = _direct_complement_flux_v2(
                    positions_angstrom,
                    intrinsic_radii_angstrom,
                    owner,
                    receiver,
                    union.union_cap,
                    order,
                )
                representation = "direct-complement"
                full_diagnostic = _full_sphere_diagnostic_v2(
                    positions_angstrom,
                    intrinsic_radii_angstrom,
                    owner,
                    receiver,
                )
                all_q.append(minimum_q)
            else:
                matching = [cap for cap in union.caps if cap.receiver_index == receiver]
                if len(matching) != 1:
                    _raise(
                        ContactV2FailureReason.UNSUPPORTED_RECEIVER_CAP_MAPPING,
                        "disjoint nonowner must correspond to exactly one physical cap",
                        union.raw_margins,
                    )
                own_cap = matching[0]
                other_cap = next(cap for cap in union.caps if cap is not own_cap)
                direct, q_direct = _direct_complement_flux_v2(
                    positions_angstrom,
                    intrinsic_radii_angstrom,
                    owner,
                    receiver,
                    own_cap,
                    order,
                )
                other, q_other = _cap_flux_v2(
                    positions_angstrom,
                    intrinsic_radii_angstrom,
                    owner,
                    receiver,
                    (other_cap.axis, other_cap.cosine),
                    order,
                    cap_index=other_cap.receiver_index,
                )
                value = direct - other
                minimum_q = min(q_direct, q_other)
                all_q.append(minimum_q)
                representation = (
                    "receiver-corresponding-direct-complement-minus-other-cap"
                )
                full_diagnostic = None
            flux = (
                flux
                + torch.nn.functional.one_hot(torch.tensor(receiver), num_classes=3).to(
                    dtype=torch.float64
                )
                * value
            )
            accumulations.append(
                ReceiverAccumulationV2(
                    receiver,
                    representation,
                    minimum_q,
                    full_diagnostic,
                )
            )
        owner_diagnostics.append(
            OwnerContactV2Diagnostics(
                owner,
                union.state,
                tuple(cap.receiver_index for cap in union.caps),
                union.covering_receiver_index,
                union.true_predicates,
                _float(solid_angle),
                tuple(accumulations),
                union.raw_margins,
            )
        )
    if not bool(torch.isfinite(flux).all()):
        _raise(
            ContactV2FailureReason.NONFINITE_ALGEBRA,
            "assembled contact flux is nonfinite",
        )
    return ContactFluxV2Result(
        flux,
        ContactFluxV2Diagnostics(
            NUMERICAL_PROFILE_ID,
            order,
            tuple(owner_diagnostics),
            min(all_q) if all_q else None,
        ),
        order,
    )
