"""Pure complete-scalar assembly contracts for continuum CHA research."""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb import (
    ChaBranchFailureReason,
    ContinuumChaBranchError,
    ContinuumChaDomainError,
    continuum_cha_scalar,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_domain import (
    DomainCertificationFailure,
    DomainFailureReason,
    certify_three_site_domain,
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _charge_hash(charges):
    return hashlib.sha256(
        (json.dumps(charges, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _topology(permutation=(0, 1, 2), *, charges_override=None):
    names = ["O", "H1", "H2"]
    elements = ["O", "H", "H"]
    atomic_numbers = [8, 1, 1]
    types = ["oh", "ho", "ho"]
    charges = [-0.8, 0.4, 0.4] if charges_override is None else charges_override
    bondi = [1.5, 1.5, 1.5]
    cha = [1.88, 1.88, 1.88]
    rmin = [1.7683, 1.2, 1.2]
    epsilon = [0.152, 0.02, 0.02]
    inverse = {old: new for new, old in enumerate(permutation)}
    bonds = sorted(
        (
            min(inverse[first], inverse[second]),
            max(inverse[first], inverse[second]),
            kind,
        )
        for first, second, kind in ((0, 1, "1"), (0, 2, "1"))
    )

    def permute(values):
        return [values[index] for index in permutation]

    charges = permute(charges)
    payload = {
        "schema_version": 1,
        "profile": "chagb-r6-pbsa-continuum-v1",
        "atom_ids": [1, 2, 3],
        "atom_names": permute(names),
        "elements": permute(elements),
        "atomic_numbers": permute(atomic_numbers),
        "gaff2_types": permute(types),
        "bonds": bonds,
        "declared_charge_e": 0,
        "source_charges_e": charges,
        "effective_charges_e": charges,
        "source_charges_sha256": _charge_hash(charges),
        "effective_charges_sha256": _charge_hash(charges),
        "source_mol2_sha256": "0" * 64,
        "prepared_prmtop_sha256": "1" * 64,
        "parameter_source_sha256": "2" * 64,
        "serialization_profile": "direct-fixed-charge-v1",
        "bondi_radii_angstrom": permute(bondi),
        "cha_radii_angstrom": permute(cha),
        "lj_rmin_angstrom": permute(rmin),
        "lj_epsilon_kcal_mol": permute(epsilon),
    }
    payload["content_sha256"] = _hash(payload)
    return ContinuumChaTopology.from_mapping(
        payload, expected_content_sha256=payload["content_sha256"]
    )


def _source_bound_water_topology():
    """Exact prepared-water topology identity frozen by preparation receipt."""
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
        "content_sha256": "2d2659680d850157bf41574257bdae7d593ad166b56eea69dce28c62e54c9391",
    }
    return ContinuumChaTopology.from_mapping(
        payload, expected_content_sha256=payload["content_sha256"]
    )


def test_source_bound_water_uses_cha_radii_for_r6_ses_not_bondi_radii():
    topology = _source_bound_water_topology()
    positions = torch.tensor(
        [[0.011, 0.404, 0.0], [0.777, -0.223, 0.0], [-0.788, -0.181, 0.0]],
        dtype=torch.float64,
    )
    fixed = topology.tensors(
        expected_content_sha256=topology.content_sha256, device="cpu"
    )
    wrong_domain = certify_three_site_domain(positions, fixed.bondi_radii_angstrom)
    assert isinstance(wrong_domain, DomainCertificationFailure)
    assert wrong_domain.reason is DomainFailureReason.EXPOSED_TRIPLE_PROBE

    result = continuum_cha_scalar(
        positions,
        topology,
        expected_topology_sha256=topology.content_sha256,
        order=48,
    )
    assert result.point_domain.active_pairs == ((0, 1), (0, 2))
    assert result.point_domain.fully_occluded_pairs == ((1, 2),)
    assert result.radius_provenance.r6_ses.startswith("cha_radii_angstrom")
    assert "unused" in result.radius_provenance.bondi
    assert result.radius_provenance.cavity.startswith("lj_rmin_angstrom")
    assert result.radius_provenance.dispersion.startswith("lj_rmin_angstrom")
    assert float(result.polar_kcal_mol) == pytest.approx(-9.6967765, abs=1.0e-6)


def test_complete_scalar_keeps_one_live_graph_and_exact_component_identity():
    topology = _topology()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = continuum_cha_scalar(
        positions,
        topology,
        expected_topology_sha256=topology.content_sha256,
        order=8,
    )

    components = (
        result.polar_kcal_mol,
        result.cavity_kcal_mol,
        result.dispersion_kcal_mol,
    )
    assert result.total_kcal_mol.requires_grad
    assert all(component.requires_grad for component in components)
    assert torch.equal(
        result.total_kcal_mol,
        result.polar_kcal_mol + result.cavity_kcal_mol + result.dispersion_kcal_mol,
    )
    for component in components:
        gradient = torch.autograd.grad(component, positions, retain_graph=True)[0]
        assert gradient.shape == positions.shape
        assert torch.isfinite(gradient).all()
    force = -torch.autograd.grad(result.total_kcal_mol, positions)[0]
    assert force.shape == positions.shape
    assert torch.isfinite(force).all()
    assert result.quadrature_identity.phi_order == 8
    assert result.quadrature_identity.nonpolar_atol == 1.0e-11
    assert result.quadrature_identity.nonpolar_rtol == 1.0e-11
    assert result.resources.site_count == 3
    assert result.resources.dispersion_ad_site_cap == 6
    assert result.point_domain.active_pairs == ((0, 1),)


def test_complete_scalar_is_rigid_transform_and_permutation_invariant():
    topology = _topology()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=torch.float64,
    )
    reference = continuum_cha_scalar(
        positions,
        topology,
        expected_topology_sha256=topology.content_sha256,
        order=32,
    )

    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    moved = positions @ rotation.T + torch.tensor([1.2, -0.7, 2.3], dtype=torch.float64)
    transformed = continuum_cha_scalar(
        moved,
        topology,
        expected_topology_sha256=topology.content_sha256,
        order=32,
    )

    permutation = (2, 0, 1)
    permuted_topology = _topology(permutation)
    permuted = continuum_cha_scalar(
        positions[list(permutation)],
        permuted_topology,
        expected_topology_sha256=permuted_topology.content_sha256,
        order=32,
    )
    for field in (
        "polar_kcal_mol",
        "cavity_kcal_mol",
        "dispersion_kcal_mol",
        "total_kcal_mol",
    ):
        expected = getattr(reference, field)
        assert torch.allclose(
            getattr(transformed, field), expected, atol=2e-9, rtol=0.0
        )
        assert torch.allclose(getattr(permuted, field), expected, atol=2e-9, rtol=0.0)


def test_scalar_rejects_bad_hash_and_uncertified_geometry_without_partial_result():
    topology = _topology()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="external identity"):
        continuum_cha_scalar(
            positions,
            topology,
            expected_topology_sha256="f" * 64,
            order=8,
        )
    with pytest.raises(ContinuumChaDomainError) as captured:
        continuum_cha_scalar(
            positions,
            topology,
            expected_topology_sha256=topology.content_sha256,
            order=8,
        )
    assert captured.value.failure.raw_margins


@pytest.mark.parametrize("order", [True, 7, 8.0])
def test_order_and_inputs_fail_closed_before_quadrature_work(order):
    topology = _topology()
    positions = torch.zeros((3, 3), dtype=torch.float64)
    with pytest.raises((TypeError, ValueError)):
        continuum_cha_scalar(
            positions,
            topology,
            expected_topology_sha256=topology.content_sha256,
            order=order,
        )


@pytest.mark.parametrize("order", [129, 10**9])
def test_order_resource_cap_rejects_before_r6_quadrature(monkeypatch, order):
    topology = _topology()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=torch.float64,
    )

    def quadrature_must_not_start(*args, **kwargs):
        raise AssertionError("R6 quadrature started before the order resource guard")

    monkeypatch.setattr(
        "maple.function.calculator.extra_correction.implicit."
        "torch_continuum_chagb._r6_inverse_born",
        quadrature_must_not_start,
    )
    with pytest.raises(ValueError, match="at most 128"):
        continuum_cha_scalar(
            positions,
            topology,
            expected_topology_sha256=topology.content_sha256,
            order=order,
        )


@pytest.mark.parametrize("requires_grad", [False, True])
def test_weighted_sign_guard_is_structured_with_or_without_coordinate_ad(requires_grad):
    # Frozen for this geometry/order so site 0's weighted charge is zero while
    # its source charge remains nonzero and the total molecular charge is zero.
    charges = [0.0008324998734147369, -1.0008324998734148, 1.0]
    topology = _topology(charges_override=charges)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    with pytest.raises(ContinuumChaBranchError) as captured:
        continuum_cha_scalar(
            positions,
            topology,
            expected_topology_sha256=topology.content_sha256,
            order=48,
        )
    assert captured.value.reason is ChaBranchFailureReason.WEIGHTED_SIGN_ZERO
    assert dict(captured.value.raw_margins)["minimum_relevant_sign_margin_e"] <= 1e-10


@pytest.mark.parametrize("requires_grad", [False, True])
def test_size_switch_guard_is_structured_with_or_without_coordinate_ad(requires_grad):
    topology = _topology()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 23.01645274087538, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    with pytest.raises(ContinuumChaBranchError) as captured:
        continuum_cha_scalar(
            positions,
            topology,
            expected_topology_sha256=topology.content_sha256,
            order=48,
        )
    assert captured.value.reason is ChaBranchFailureReason.SIZE_SWITCH
    assert dict(captured.value.raw_margins)["size_switch_margin_angstrom"] <= 1e-8
