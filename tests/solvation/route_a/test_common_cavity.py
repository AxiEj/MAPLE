from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.cavity_topology import (
    CavityTopologyError,
    assess_atom_sphere_union,
    validate_atom_sphere_supermolecule,
    validate_common_envelope,
)
from maple.function.calculator.extra_correction.implicit.supermolecule_pcm import (
    SupermoleculePCMRule,
)
from maple.function.dispatcher.solvfe.protocol import RouteAProtocol

from .conftest import DOCS_DIR, PROJECT_ROOT


RADII_A = np.asarray(
    [1.70, 1.85, 1.85, 1.85, *([1.20] * 6), 1.52, 1.20, 1.20],
    dtype=float,
)
FRAGMENT_IDS = np.asarray([0] * 10 + [1] * 3, dtype=int)

# Real P-endpoint frames from the acetone+n=1 enhanced OMOL pilot.  The first
# has only a near-tangent atom-sphere bridge; the second has two disconnected
# atom-sphere components even though the legacy PCM evaluation emitted no
# warning.  Keeping both coordinates here makes the topology regression
# independent of the untracked pilot archive.
NEAR_TANGENT_FRAME_A = np.asarray(
    [
        [0.0178297286, 0.0746796173, 1.4825107185],
        [-0.0408738827, -0.0244221700, 0.2866460205],
        [0.0279811938, 1.2385445853, -0.6068966161],
        [-0.0997195569, -1.2868215736, -0.5359231323],
        [0.2647717856, 2.1562294433, -0.1566283046],
        [0.2821667448, -2.1199527591, 0.0717726555],
        [-0.8447536249, 1.2126147002, -1.1966520341],
        [0.8602028827, 1.1178787070, -1.3543806031],
        [0.5761663308, -1.1838254226, -1.3497825415],
        [-1.0563664172, -1.4414665383, -0.8908954159],
        [2.5550868957, -0.0976038704, -3.0232946355],
        [1.7488283147, -0.5805931941, -3.3481285002],
        [3.0675308729, 0.1148476915, -3.7861234484],
    ],
    dtype=float,
)
DISCONNECTED_FRAME_A = np.asarray(
    [
        [0.0234620367, 0.0695097901, 1.4794041917],
        [-0.0449773157, -0.0066232989, 0.2796915682],
        [0.0367192578, 1.2457947018, -0.6113235798],
        [-0.0937092102, -1.2992291695, -0.5138485532],
        [0.1820652543, 2.1758474828, -0.1112223931],
        [0.3383888199, -2.1235308146, 0.0380839929],
        [-0.8608083965, 1.2227645768, -1.2239784522],
        [0.8049539390, 1.0851348645, -1.3862374165],
        [0.5444527450, -1.1663935715, -1.3690626124],
        [-1.1102687709, -1.4617709127, -0.8564188405],
        [2.8649157734, 0.5569303545, -3.1776318529],
        [3.4106725148, 0.2985015561, -3.9447936647],
        [2.5875656407, 1.4140664044, -3.4480596205],
    ],
    dtype=float,
)


def test_real_near_tangent_frame_is_connected_but_fails_bridge_margin():
    report = assess_atom_sphere_union(
        NEAR_TANGENT_FRAME_A,
        RADII_A,
        FRAGMENT_IDS,
    )

    assert report.component_count == 1
    assert report.cross_fragment_edge_count == 2
    assert report.maximum_cross_fragment_overlap_A == pytest.approx(
        0.0488010630,
        abs=1.0e-9,
    )
    with pytest.raises(
        CavityTopologyError,
        match="SMD_CAVITY_BRIDGE_MARGIN_TOO_SMALL",
    ):
        validate_atom_sphere_supermolecule(
            NEAR_TANGENT_FRAME_A,
            RADII_A,
            FRAGMENT_IDS,
            minimum_bridge_overlap_A=0.10,
        )


def test_real_disconnected_frame_fails_even_if_pcm_was_warning_free():
    report = assess_atom_sphere_union(
        DISCONNECTED_FRAME_A,
        RADII_A,
        FRAGMENT_IDS,
    )

    assert report.component_count == 2
    assert report.cross_fragment_edge_count == 0
    assert report.maximum_cross_fragment_overlap_A < 0.0
    with pytest.raises(
        CavityTopologyError,
        match="SMD_CAVITY_TOPOLOGY_DISCONTINUITY",
    ):
        validate_atom_sphere_supermolecule(
            DISCONNECTED_FRAME_A,
            RADII_A,
            FRAGMENT_IDS,
            minimum_bridge_overlap_A=0.10,
        )


def test_common_envelope_requires_full_protected_sphere_containment():
    envelope_centers = np.asarray([[0.0, 0.0, 0.0]])
    envelope_radii = np.asarray([5.0])

    report = validate_common_envelope(
        envelope_centers,
        envelope_radii,
        protected_centers_angstrom=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        protected_radii_angstrom=np.asarray([1.7, 1.2]),
        minimum_containment_margin_A=0.5,
    )
    assert report.minimum_containment_margin_A == pytest.approx(0.8)

    with pytest.raises(
        CavityTopologyError,
        match="SMD_COMMON_CAVITY_COVERAGE",
    ):
        validate_common_envelope(
            envelope_centers,
            envelope_radii,
            protected_centers_angstrom=np.asarray([[4.0, 0.0, 0.0]]),
            protected_radii_angstrom=np.asarray([1.2]),
            minimum_containment_margin_A=0.5,
        )


def test_fragment_bridge_gate_uses_weakest_required_connection():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.49, 0.0, 0.0],
        ]
    )
    radii = np.ones(3)
    fragments = np.asarray([0, 1, 2])
    report = assess_atom_sphere_union(positions, radii, fragments)

    assert report.component_count == 1
    assert report.fragment_component_count == 1
    assert report.maximum_cross_fragment_overlap_A == pytest.approx(0.5)
    assert report.fragment_bridge_bottleneck_A == pytest.approx(0.01)
    with pytest.raises(
        CavityTopologyError,
        match="SMD_CAVITY_BRIDGE_MARGIN_TOO_SMALL",
    ):
        validate_atom_sphere_supermolecule(
            positions,
            radii,
            fragments,
            minimum_bridge_overlap_A=0.10,
        )


@pytest.mark.parametrize("positions", [NEAR_TANGENT_FRAME_A, DISCONNECTED_FRAME_A])
def test_protocol_v2_scaled_supermolecule_rule_repairs_real_pilot_topology(
    positions,
):
    protocol = RouteAProtocol.load(
        DOCS_DIR / "protocol-v2.json",
        project_root=PROJECT_ROOT,
    )
    atoms = Atoms(
        numbers=[8, 6, 6, 6, *([1] * 6), 8, 1, 1],
        positions=positions,
    )
    atoms.info["mol2"] = {
        "atom_types": ["o", "c", "c3", "c3", *(["h1"] * 6), "ow", "hw", "hw"]
    }
    rule = SupermoleculePCMRule.from_protocol(protocol)
    report = rule.preflight(atoms)

    assert rule.radius_scale == 1.2
    assert report.component_count == 1
    assert report.fragment_component_count == 1
    assert report.fragment_bridge_bottleneck_A >= 0.10


def test_protocol_v2_scaled_supermolecule_rule_requires_atom_types():
    protocol = RouteAProtocol.load(
        DOCS_DIR / "protocol-v2.json",
        project_root=PROJECT_ROOT,
    )
    rule = SupermoleculePCMRule.from_protocol(protocol)

    with pytest.raises(ValueError, match="MOL2 atom type"):
        rule.radii_angstrom(Atoms("OH2"))
