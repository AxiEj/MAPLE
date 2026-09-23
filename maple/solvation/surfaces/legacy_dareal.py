"""Exact differentiable legacy DAREAL union-of-spheres surface geometry.

This module is a clean Python/Torch translation of the energy-only geometry
in PySCF 2.13.1 ``pyscf/lib/solvent/mnsol.F`` (DAREAL, originally by
D. Liotard, 1992).  PySCF distributes that source under Apache-2.0; this
translation retains its branch ordering and constants but deliberately omits
the hand-written first derivatives because Torch autograd differentiates the
same scalar geometry twice.

The discrete topology is selected on the host.  Every continuous geometric
quantity used to assemble an area remains a live ``torch.float64`` tensor.
Unlike the historical Fortran, ambiguous four-sphere/tangent events are not
regularized by changing a radius: they fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping

import torch

_PI = 3.14159265358979
_EPSI = 1.0e-11
_SOURCE_SHA256 = "f57b94c0eb6d5a29f1a1441294795321a5a39af74e0bec890628fac83027250b"


class LegacyDAREALTopologyError(RuntimeError):
    """Raised when the legacy surface is not locally differentiable."""


@dataclass(frozen=True)
class LegacyDAREALConfig:
    relative_topology_tolerance: float = _EPSI
    guard_version: str = "pyscf-2.13.1-dareal-fixed-topology-v1"

    def __post_init__(self) -> None:
        if self.relative_topology_tolerance <= 0.0:
            raise ValueError("DAREAL topology tolerance must be positive.")


@dataclass(frozen=True)
class LegacyDAREALDiagnostics:
    guard_version: str
    atom_branches: tuple[str, ...]
    overlap_counts: tuple[int, ...]
    polygon_counts: tuple[int, ...]
    minimum_guard_margin: float | None
    topology_sha256: str


@dataclass(frozen=True)
class LegacyDAREALResult:
    areas_angstrom2: torch.Tensor
    solid_angles: torch.Tensor
    diagnostics: LegacyDAREALDiagnostics
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _host(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _require_margin(
    value: float, scale: float, config: LegacyDAREALConfig, message: str
) -> float:
    margin = abs(value)
    if margin <= config.relative_topology_tolerance * max(1.0, abs(scale)):
        raise LegacyDAREALTopologyError(message)
    return margin


def _cap_point(
    points: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    i: int,
    j: int,
    plus: bool,
) -> torch.Tensor:
    low, high = sorted((i, j))
    positive, negative = points[(low, high)]
    # The stored positive point uses cross(u_low, u_high).
    if i == low:
        return positive if plus else negative
    return negative if plus else positive


def _solid_angle_one(
    positions: torch.Tensor,
    radii: torch.Tensor,
    k: int,
    config: LegacyDAREALConfig,
) -> tuple[torch.Tensor, str, int, int, float, str]:
    """Translate the energy path of DAREAL for sphere ``k``."""

    two_pi = positions.new_tensor(2.0 * _PI)
    four_pi = positions.new_tensor(4.0 * _PI)
    rk = radii[k]
    rk_h = _host(rk)
    if rk_h <= 0.0:
        return positions.new_zeros(()), "zero-radius", 0, 0, float("inf"), "zero-radius"

    overlap_atoms: list[int] = []
    minimum_margin = float("inf")

    def record_margin(value: float, scale: float, message: str) -> float:
        nonlocal minimum_margin
        margin = _require_margin(value, scale, config, message)
        minimum_margin = min(minimum_margin, margin)
        return margin

    for atom in range(positions.shape[0]):
        if atom == k or _host(radii[atom]) <= 0.0:
            continue
        delta = positions[atom] - positions[k]
        distance = torch.linalg.vector_norm(delta)
        d_h = _host(distance)
        if d_h == 0.0:
            raise LegacyDAREALTopologyError("DAREAL rejects coincident sphere centres.")
        outer = rk_h + _host(radii[atom]) - d_h
        inner = d_h - abs(rk_h - _host(radii[atom]))
        minimum_margin = min(minimum_margin, abs(outer), abs(inner))
        threshold = config.relative_topology_tolerance * rk_h
        if abs(outer) <= threshold or abs(inner) <= threshold:
            raise LegacyDAREALTopologyError(
                f"DAREAL sphere {k} is on a tangent/containment boundary with sphere {atom}."
            )
        if outer < threshold:
            continue
        if inner < threshold:
            if rk_h <= _host(radii[atom]):
                return (
                    positions.new_zeros(()),
                    "contained",
                    0,
                    0,
                    minimum_margin,
                    f"contained-by:{atom}",
                )
            continue
        overlap_atoms.append(atom)

    if not overlap_atoms:
        return four_pi, "free", 0, 0, minimum_margin, "free"

    count = len(overlap_atoms)
    cap_cos: list[torch.Tensor] = []
    cap_sin: list[torch.Tensor] = []
    directions: list[torch.Tensor] = []
    for atom in overlap_atoms:
        delta = positions[atom] - positions[k]
        distance = torch.linalg.vector_norm(delta)
        cosine = 0.5 / rk * (distance + (rk * rk - radii[atom] ** 2) / distance)
        radicand = 1.0 - cosine * cosine
        radicand_h = _host(radicand)
        if radicand_h <= 0.0:
            raise LegacyDAREALTopologyError(
                "DAREAL cap half-angle is singular or tangent."
            )
        record_margin(
            radicand_h,
            1.0,
            "DAREAL cap half-angle is too near a singular or tangent boundary.",
        )
        cap_cos.append(cosine)
        cap_sin.append(torch.sqrt(radicand))
        directions.append(delta / distance)

    embedded = [False] * count
    connected = [[False] * count for _ in range(count)]
    direction_cos: dict[tuple[int, int], torch.Tensor] = {}
    for i in range(1, count):
        for j in range(i):
            if embedded[j]:
                continue
            dot = torch.dot(directions[i], directions[j])
            direction_cos[(j, i)] = dot
            tij = dot - cap_cos[i] * cap_cos[j]
            sisj = cap_sin[i] * cap_sin[j]
            cisj = cap_cos[i] * cap_sin[j]
            sicj = cap_sin[i] * cap_cos[j]
            embed_test = _host(
                tij - sisj + config.relative_topology_tolerance * torch.abs(cisj - sicj)
            )
            record_margin(
                embed_test, 1.0, "DAREAL cap-containment topology is ambiguous."
            )
            if embed_test > 0.0:
                containment_order = _host(cap_cos[j] - cap_cos[i])
                record_margin(
                    containment_order,
                    1.0,
                    "DAREAL embedded-cap selection is ambiguous.",
                )
                if containment_order > 0.0:
                    embedded[j] = True
                else:
                    embedded[i] = True
                    break
            else:
                sum_cs = sicj + cisj
                sum_h = _host(sum_cs)
                record_margin(
                    sum_h,
                    1.0,
                    "DAREAL cap-connectivity sign branch is ambiguous.",
                )
                epsij = config.relative_topology_tolerance * sum_h
                if sum_h >= 0.0:
                    cross_test = _host(tij + sisj) - epsij
                    record_margin(
                        cross_test,
                        1.0,
                        "DAREAL cap-intersection topology is ambiguous.",
                    )
                    connected[i][j] = connected[j][i] = cross_test > 0.0
                else:
                    burial_test = _host(tij + sisj) + epsij
                    record_margin(
                        burial_test,
                        1.0,
                        "DAREAL two-cap burial topology is ambiguous.",
                    )
                    if burial_test <= 0.0:
                        trace = f"overlap:{overlap_atoms};embedded:{embedded};buried-pair:{i},{j}"
                        return (
                            positions.new_zeros(()),
                            "two-cap-buried",
                            0,
                            0,
                            minimum_margin,
                            trace,
                        )
                    connected[i][j] = connected[j][i] = True

    isolated_slice = positions.new_zeros(())
    cluster: list[int] = []
    for i in range(count):
        if embedded[i]:
            continue
        if any(connected[i][j] and not embedded[j] for j in range(count)):
            cluster.append(i)
        else:
            isolated_slice = isolated_slice + (1.0 - cap_cos[i])
    isolated_slice = two_pi * isolated_slice
    if not cluster:
        trace = f"overlap:{overlap_atoms};embedded:{embedded};connected:none"
        return (
            four_pi - isolated_slice,
            "disjoint-caps",
            count,
            0,
            minimum_margin,
            trace,
        )

    boundary: dict[int, list[tuple[int, torch.Tensor, str]]] = {i: [] for i in cluster}
    points: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    free_count = 0
    for ii in range(1, len(cluster)):
        i = cluster[ii]
        for jj in range(ii):
            j = cluster[jj]
            if not connected[i][j]:
                continue
            dot = direction_cos.get((min(i, j), max(i, j)))
            if dot is None:
                dot = torch.dot(directions[i], directions[j])
            intersection_denominator = 1.0 - dot * dot
            record_margin(
                _host(intersection_denominator),
                1.0,
                "DAREAL intersection-circle denominator is singular.",
            )
            sin2inv = 1.0 / intersection_denominator
            ai = (cap_cos[i] - cap_cos[j] * dot) * sin2inv
            bi = (cap_cos[j] - cap_cos[i] * dot) * sin2inv
            ci2 = (1.0 - ai * cap_cos[i] - bi * cap_cos[j]) * sin2inv
            ci2_h = _host(ci2)
            if ci2_h <= 0.0:
                raise LegacyDAREALTopologyError(
                    "DAREAL cap intersection is tangent or singular."
                )
            record_margin(
                ci2_h,
                1.0,
                "DAREAL cap intersection is too near a tangent or singular boundary.",
            )
            cross = torch.linalg.cross(directions[i], directions[j])
            ci = torch.sqrt(ci2)
            positive = ai * directions[i] + bi * directions[j] + ci * cross
            negative = ai * directions[i] + bi * directions[j] - ci * cross
            points[(min(i, j), max(i, j))] = (
                positive if i < j else negative,
                negative if i < j else positive,
            )
            free_positive = True
            free_negative = True
            for ell in cluster:
                if ell in (i, j) or not (connected[ell][i] and connected[ell][j]):
                    continue
                high = cap_cos[ell] + config.relative_topology_tolerance * cap_sin[ell]
                low = cap_cos[ell] - config.relative_topology_tolerance * cap_sin[ell]
                for label, point in (("positive", positive), ("negative", negative)):
                    if (label == "positive" and not free_positive) or (
                        label == "negative" and not free_negative
                    ):
                        continue
                    check = torch.dot(point, directions[ell])
                    check_h, high_h, low_h = _host(check), _host(high), _host(low)
                    if check_h > high_h:
                        record_margin(
                            check_h - high_h,
                            1.0,
                            "DAREAL covered-intersection selection is ambiguous.",
                        )
                        if label == "positive":
                            free_positive = False
                        else:
                            free_negative = False
                    elif check_h >= low_h:
                        raise LegacyDAREALTopologyError(
                            "DAREAL four-sphere shared-point guard triggered; radius perturbation is forbidden."
                        )
                    else:
                        record_margin(
                            low_h - check_h,
                            1.0,
                            "DAREAL free-intersection selection is ambiguous.",
                        )
            if free_negative:
                free_count += 1
                boundary[i].append((j, negative, "negative"))
                boundary[j].append((i, negative, "negative"))
            if free_positive:
                free_count += 1
                boundary[i].append((j, positive, "positive"))
                boundary[j].append((i, positive, "positive"))

    if free_count == 0:
        trace = (
            f"overlap:{overlap_atoms};embedded:{embedded};cluster:{cluster};free:none"
        )
        return positions.new_zeros(()), "cluster-buried", 0, 0, minimum_margin, trace

    polygon_term = positions.new_zeros(())
    slice_term = isolated_slice
    ordered_neighbors: dict[int, list[int]] = {}
    for i in cluster:
        entries = boundary[i]
        if not entries:
            continue
        first_neighbor, first_point, _ = entries[0]
        orientation_value = _host(
            torch.dot(
                torch.linalg.cross(directions[i], first_point),
                directions[first_neighbor],
            )
        )
        record_margin(
            orientation_value,
            1.0,
            "DAREAL polygon orientation is ambiguous.",
        )
        orientation = orientation_value > 0.0
        angles: list[tuple[float, torch.Tensor, int]] = []
        c2 = cap_cos[i] * cap_cos[i]
        vij = torch.linalg.cross(directions[i], first_point)
        for neighbor, point, _ in entries[1:]:
            x = torch.dot(vij, point)
            y = torch.dot(point, first_point) - c2
            angle_denominator = _host(x * x + y * y)
            if angle_denominator <= 0.0:
                raise LegacyDAREALTopologyError(
                    "DAREAL boundary-angle ordering is degenerate."
                )
            record_margin(
                angle_denominator,
                1.0,
                "DAREAL atan2 boundary-angle denominator is near singular.",
            )
            angle = torch.atan2(x, y)
            angle_h = _host(angle)
            # x=0,y>0 is not merely the atan2 chart seam here: it means this
            # boundary point coincides with the selected first point.
            if _host(y) > 0.0:
                record_margin(
                    _host(x),
                    1.0,
                    "DAREAL boundary-angle chart contains a duplicate point.",
                )
            if angle_h <= 0.0:
                angle = angle + two_pi
            angles.append((_host(angle), angle, neighbor))
        angles.sort(key=lambda item: item[0])
        for left, right in zip(angles, angles[1:], strict=False):
            record_margin(
                right[0] - left[0],
                2.0 * _PI,
                "DAREAL boundary-angle sort order is ambiguous.",
            )
        neighbors = [first_neighbor, *[item[2] for item in angles]]
        if len(neighbors) % 2:
            raise LegacyDAREALTopologyError(
                "DAREAL cap boundary has an odd number of intersections."
            )
        aodd = angles[0][1]
        for idx in range(2, len(angles), 2):
            aodd = aodd + angles[idx][1] - angles[idx - 1][1]
        even = two_pi - aodd
        xcap = 1.0 - cap_cos[i]
        if orientation:
            polygon_term = polygon_term + aodd
            slice_term = slice_term + even * xcap
        else:
            polygon_term = polygon_term + even
            slice_term = slice_term + aodd * xcap
            neighbors = neighbors[1:] + neighbors[:1]
        ordered_neighbors[i] = neighbors

    # Follow and erase the directed polygon edges exactly as DAREAL does.
    available: dict[int, list[bool]] = {
        i: [True] * (len(neighbors) // 2) for i, neighbors in ordered_neighbors.items()
    }
    polygon_count = 0
    for start_cap, neighbors in ordered_neighbors.items():
        for pair_index in range(len(neighbors) // 2):
            if not available[start_cap][pair_index]:
                continue
            ia = neighbors[2 * pair_index]
            ib = start_cap
            available[start_cap][pair_index] = False
            phi_cosine = (
                torch.dot(directions[ia], directions[ib]) - cap_cos[ia] * cap_cos[ib]
            ) / (cap_sin[ia] * cap_sin[ib])
            phi_cosine_h = _host(phi_cosine)
            if not -1.0 <= phi_cosine_h <= 1.0:
                raise LegacyDAREALTopologyError(
                    "DAREAL polygon angle is outside acos domain."
                )
            record_margin(
                1.0 - abs(phi_cosine_h),
                1.0,
                "DAREAL polygon angle is too near an acos endpoint.",
            )
            last_phi = torch.acos(phi_cosine)
            polygon_term = polygon_term + last_phi
            steps = 0
            while True:
                steps += 1
                if steps > 2 * free_count + 2:
                    raise LegacyDAREALTopologyError(
                        "DAREAL polygon connectivity did not close."
                    )
                next_pair = None
                for candidate, flag in enumerate(available.get(ia, ())):
                    if flag and ordered_neighbors[ia][2 * candidate + 1] == ib:
                        next_pair = candidate
                        break
                if next_pair is None:
                    break
                old_ib = ib
                available[ia][next_pair] = False
                ib = ia
                ia = ordered_neighbors[ib][2 * next_pair]
                if ia != old_ib:
                    phi_cosine = (
                        torch.dot(directions[ia], directions[ib])
                        - cap_cos[ia] * cap_cos[ib]
                    ) / (cap_sin[ia] * cap_sin[ib])
                    phi_cosine_h = _host(phi_cosine)
                    if not -1.0 <= phi_cosine_h <= 1.0:
                        raise LegacyDAREALTopologyError(
                            "DAREAL polygon angle is outside acos domain."
                        )
                    record_margin(
                        1.0 - abs(phi_cosine_h),
                        1.0,
                        "DAREAL polygon angle is too near an acos endpoint.",
                    )
                    last_phi = torch.acos(phi_cosine)
                    polygon_term = polygon_term + last_phi
                else:
                    ib = start_cap
                    ia = neighbors[2 * pair_index]
                    # DAREAL deliberately reuses PHI when the loop closes.
                    polygon_term = polygon_term + last_phi
            polygon_count += 1

    polygon_term = polygon_term + (polygon_count - free_count) * two_pi
    # torch.remainder is the exact Fortran MOD branch here because APOLY is nonnegative.
    modulo_value = _host(torch.remainder(polygon_term, four_pi))
    record_margin(
        min(modulo_value, 4.0 * _PI - modulo_value),
        4.0 * _PI,
        "DAREAL polygon-area modulo branch is ambiguous.",
    )
    area = four_pi - slice_term - torch.remainder(polygon_term, four_pi)
    if not torch.isfinite(area):
        raise LegacyDAREALTopologyError("DAREAL produced a non-finite solid angle.")
    connected_pairs = [
        (i, j) for i in range(count) for j in range(i) if connected[i][j]
    ]
    boundary_labels = {
        i: [(neighbor, sign) for neighbor, _, sign in entries]
        for i, entries in boundary.items()
    }
    trace = json.dumps(
        {
            "overlap_atoms": overlap_atoms,
            "embedded": embedded,
            "connected_pairs": connected_pairs,
            "cluster": cluster,
            "boundary_labels": boundary_labels,
            "ordered_neighbors": ordered_neighbors,
            "polygon_count": polygon_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return area, "polygon", count, polygon_count, minimum_margin, trace


def legacy_dareal_areas_torch(
    positions_angstrom: torch.Tensor,
    radii_angstrom: torch.Tensor,
    *,
    config: LegacyDAREALConfig | None = None,
) -> LegacyDAREALResult:
    """Return exact per-sphere accessible areas for a union of spheres."""

    cfg = LegacyDAREALConfig() if config is None else config
    if not isinstance(positions_angstrom, torch.Tensor) or not isinstance(
        radii_angstrom, torch.Tensor
    ):
        raise TypeError("DAREAL positions and radii must be Torch tensors.")
    if (
        positions_angstrom.dtype != torch.float64
        or radii_angstrom.dtype != torch.float64
    ):
        raise TypeError("DAREAL requires torch.float64 positions and radii.")
    if positions_angstrom.device != radii_angstrom.device:
        raise ValueError("DAREAL positions and radii must share one device.")
    if positions_angstrom.ndim != 2 or positions_angstrom.shape[1] != 3:
        raise ValueError("DAREAL positions must have shape (n_atoms, 3).")
    if radii_angstrom.shape != (positions_angstrom.shape[0],):
        raise ValueError("DAREAL radii must have shape (n_atoms,).")
    if not bool(torch.isfinite(positions_angstrom).all()) or not bool(
        torch.isfinite(radii_angstrom).all()
    ):
        raise ValueError("DAREAL inputs must be finite.")
    if bool((radii_angstrom <= 0.0).any()):
        raise ValueError("DAREAL radii must be positive.")

    solid, branches, overlaps, polygons, margins, topology_traces = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for atom in range(positions_angstrom.shape[0]):
        value, branch, overlap, polygon, margin, topology_trace = _solid_angle_one(
            positions_angstrom, radii_angstrom, atom, cfg
        )
        solid.append(value)
        branches.append(branch)
        overlaps.append(overlap)
        polygons.append(polygon)
        margins.append(margin)
        topology_traces.append(topology_trace)
    solid_angles = torch.stack(solid)
    areas = solid_angles * radii_angstrom * radii_angstrom
    minimum_guard_margin = min(margins, default=float("inf"))
    return LegacyDAREALResult(
        areas_angstrom2=areas,
        solid_angles=solid_angles,
        diagnostics=LegacyDAREALDiagnostics(
            guard_version=cfg.guard_version,
            atom_branches=tuple(branches),
            overlap_counts=tuple(overlaps),
            polygon_counts=tuple(polygons),
            minimum_guard_margin=(
                minimum_guard_margin if math.isfinite(minimum_guard_margin) else None
            ),
            topology_sha256=hashlib.sha256(
                json.dumps(topology_traces, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        ),
        provenance={
            "algorithm": "legacy-dareal-analytic-union-spheres",
            "upstream": "PySCF 2.13.1 pyscf/lib/solvent/mnsol.F:1332-2168",
            "upstream_license": "Apache-2.0",
            "upstream_sha256": _SOURCE_SHA256,
        },
    )


__all__ = [
    "LegacyDAREALConfig",
    "LegacyDAREALDiagnostics",
    "LegacyDAREALResult",
    "LegacyDAREALTopologyError",
    "legacy_dareal_areas_torch",
]
