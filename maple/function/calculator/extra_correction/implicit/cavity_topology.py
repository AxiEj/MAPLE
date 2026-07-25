from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CavityTopologyError(ValueError):
    """A proposed outer cavity violates a fail-closed topology contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class SphereUnionReport:
    component_count: int
    components: tuple[tuple[int, ...], ...]
    fragment_component_count: int
    cross_fragment_edge_count: int
    maximum_cross_fragment_overlap_A: float
    minimum_positive_cross_fragment_overlap_A: float | None
    fragment_bridge_bottleneck_A: float


@dataclass(frozen=True)
class CommonEnvelopeReport:
    component_count: int
    minimum_containment_margin_A: float
    containment_margins_A: tuple[float, ...]


def _positions_and_radii(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_angstrom, dtype=float)
    radii = np.asarray(radii_angstrom, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or radii.shape != (positions.shape[0],)
        or positions.shape[0] == 0
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError(
            f"{label} positions must be finite (N,3) coordinates and radii "
            "must be one finite positive value per center."
        )
    return positions, radii


def _connected_components(
    positions: np.ndarray,
    radii: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(len(positions))]
    for left in range(len(positions)):
        displacement = positions[left + 1 :] - positions[left]
        distances = np.linalg.norm(displacement, axis=1)
        overlaps = radii[left] + radii[left + 1 :] - distances
        for right, overlap in enumerate(overlaps, start=left + 1):
            if float(overlap) >= 0.0:
                adjacency[left].add(right)
                adjacency[right].add(left)

    unseen = set(range(len(positions)))
    components: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        reached = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            new_neighbors = adjacency[current].intersection(unseen)
            unseen.difference_update(new_neighbors)
            reached.update(new_neighbors)
            frontier.extend(sorted(new_neighbors, reverse=True))
        components.append(tuple(sorted(reached)))
    return tuple(components)


def assess_atom_sphere_union(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    fragment_ids: np.ndarray,
) -> SphereUnionReport:
    """Measure atom-sphere connectivity without accepting a cavity.

    This diagnostic is intentionally geometric.  A warning-free PCM solve does
    not override a disconnected union or a near-tangent solute--solvent bridge.
    """

    positions, radii = _positions_and_radii(
        positions_angstrom,
        radii_angstrom,
        label="Atom-sphere cavity",
    )
    fragments = np.asarray(fragment_ids)
    if (
        fragments.ndim != 1
        or fragments.shape != (len(positions),)
        or not np.issubdtype(fragments.dtype, np.integer)
    ):
        raise ValueError("fragment_ids must be one integer label per atom.")

    fragment_labels = tuple(int(value) for value in np.unique(fragments))
    fragment_edge_weights: dict[tuple[int, int], float] = {}
    cross_overlaps: list[float] = []
    positive_cross_overlaps: list[float] = []
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            if fragments[left] == fragments[right]:
                continue
            overlap = float(
                radii[left]
                + radii[right]
                - np.linalg.norm(positions[left] - positions[right])
            )
            cross_overlaps.append(overlap)
            pair = tuple(sorted((int(fragments[left]), int(fragments[right]))))
            fragment_edge_weights[pair] = max(
                overlap,
                fragment_edge_weights.get(pair, float("-inf")),
            )
            if overlap >= 0.0:
                positive_cross_overlaps.append(overlap)

    fragment_parent = {label: label for label in fragment_labels}

    def find(label: int) -> int:
        while fragment_parent[label] != label:
            fragment_parent[label] = fragment_parent[fragment_parent[label]]
            label = fragment_parent[label]
        return label

    selected_weights: list[float] = []
    for (left, right), weight in sorted(
        fragment_edge_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if weight < 0.0:
            continue
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        fragment_parent[right_root] = left_root
        selected_weights.append(float(weight))

    fragment_components = len({find(label) for label in fragment_labels})
    if len(fragment_labels) == 1:
        bridge_bottleneck = float("inf")
    elif fragment_components == 1:
        bridge_bottleneck = min(selected_weights)
    else:
        bridge_bottleneck = float("-inf")

    components = _connected_components(positions, radii)
    return SphereUnionReport(
        component_count=len(components),
        components=components,
        fragment_component_count=fragment_components,
        cross_fragment_edge_count=len(positive_cross_overlaps),
        maximum_cross_fragment_overlap_A=(
            max(cross_overlaps) if cross_overlaps else float("-inf")
        ),
        minimum_positive_cross_fragment_overlap_A=(
            min(positive_cross_overlaps)
            if positive_cross_overlaps
            else None
        ),
        fragment_bridge_bottleneck_A=bridge_bottleneck,
    )


def validate_atom_sphere_supermolecule(
    positions_angstrom: np.ndarray,
    radii_angstrom: np.ndarray,
    fragment_ids: np.ndarray,
    *,
    minimum_bridge_overlap_A: float,
) -> SphereUnionReport:
    """Require one robust atom-sphere component for the supermolecule."""

    threshold = float(minimum_bridge_overlap_A)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("minimum_bridge_overlap_A must be finite and positive.")
    report = assess_atom_sphere_union(
        positions_angstrom,
        radii_angstrom,
        fragment_ids,
    )
    if report.component_count != 1:
        raise CavityTopologyError(
            "SMD_CAVITY_TOPOLOGY_DISCONTINUITY",
            "the atom-sphere union has "
            f"{report.component_count} connected components.",
        )
    if report.fragment_bridge_bottleneck_A < threshold:
        raise CavityTopologyError(
            "SMD_CAVITY_BRIDGE_MARGIN_TOO_SMALL",
            "the maximum-spanning fragment bridge bottleneck is "
            f"{report.fragment_bridge_bottleneck_A:.6f} A, below the frozen "
            f"{threshold:.6f} A margin.",
        )
    return report


def validate_common_envelope(
    envelope_centers_angstrom: np.ndarray,
    envelope_radii_angstrom: np.ndarray,
    *,
    protected_centers_angstrom: np.ndarray,
    protected_radii_angstrom: np.ndarray,
    minimum_containment_margin_A: float,
) -> CommonEnvelopeReport:
    """Conservatively require protected spheres inside one fixed envelope.

    A protected sphere passes only when it is wholly contained in at least one
    envelope sphere.  This sufficient (not necessary) test is deliberately
    conservative and precedes native cavity-mesh validation.
    """

    envelope_centers, envelope_radii = _positions_and_radii(
        envelope_centers_angstrom,
        envelope_radii_angstrom,
        label="Common envelope",
    )
    protected_centers, protected_radii = _positions_and_radii(
        protected_centers_angstrom,
        protected_radii_angstrom,
        label="Protected",
    )
    threshold = float(minimum_containment_margin_A)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "minimum_containment_margin_A must be finite and non-negative."
        )

    components = _connected_components(envelope_centers, envelope_radii)
    if len(components) != 1:
        raise CavityTopologyError(
            "SMD_CAVITY_TOPOLOGY_DISCONTINUITY",
            "the common envelope has "
            f"{len(components)} connected components.",
        )

    distances = np.linalg.norm(
        protected_centers[:, None, :] - envelope_centers[None, :, :],
        axis=2,
    )
    margins = np.max(
        envelope_radii[None, :]
        - distances
        - protected_radii[:, None],
        axis=1,
    )
    minimum_margin = float(np.min(margins))
    if minimum_margin < threshold:
        failing_index = int(np.argmin(margins))
        raise CavityTopologyError(
            "SMD_COMMON_CAVITY_COVERAGE",
            f"protected sphere {failing_index} has containment margin "
            f"{minimum_margin:.6f} A, below the frozen "
            f"{threshold:.6f} A margin.",
        )
    return CommonEnvelopeReport(
        component_count=1,
        minimum_containment_margin_A=minimum_margin,
        containment_margins_A=tuple(float(value) for value in margins),
    )
