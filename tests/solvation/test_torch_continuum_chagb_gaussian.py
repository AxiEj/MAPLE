"""Complete-scalar contracts for experimental Gaussian-sign continuum CHA."""

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb import (
    ContinuumChaDomainError,
)
from maple.function.calculator.extra_correction.implicit.torch_continuum_chagb_gaussian import (
    continuum_gaussian_cha_scalar,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _charge_sha(values):
    return hashlib.sha256(
        (json.dumps(values, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def topology():
    charges = [-0.8, 0.4, 0.4]
    payload = {
        "schema_version": 1,
        "profile": "chagb-r6-pbsa-continuum-v1",
        "atom_ids": [1, 2, 3],
        "atom_names": ["O", "H1", "H2"],
        "elements": ["O", "H", "H"],
        "atomic_numbers": [8, 1, 1],
        "gaff2_types": ["oh", "ho", "ho"],
        "bonds": [[0, 1, "1"], [0, 2, "1"]],
        "declared_charge_e": 0,
        "source_charges_e": charges,
        "effective_charges_e": charges,
        "source_charges_sha256": _charge_sha(charges),
        "effective_charges_sha256": _charge_sha(charges),
        "source_mol2_sha256": "0" * 64,
        "prepared_prmtop_sha256": "1" * 64,
        "parameter_source_sha256": "2" * 64,
        "serialization_profile": "direct-fixed-charge-v1",
        "bondi_radii_angstrom": [1.5, 1.2, 1.2],
        "cha_radii_angstrom": [1.88, 1.04, 1.04],
        "lj_rmin_angstrom": [1.7683, 1.2, 1.2],
        "lj_epsilon_kcal_mol": [0.152, 0.02, 0.02],
    }
    payload["content_sha256"] = _sha(payload)
    return ContinuumChaTopology.from_mapping(
        payload, expected_content_sha256=payload["content_sha256"]
    )


def positions(requires_grad=False):
    return torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def test_full_scalar_has_live_polar_cavity_dispersion_graph_and_diagnostics():
    topo = topology()
    xyz = positions(requires_grad=True)
    result = continuum_gaussian_cha_scalar(
        xyz, topo, expected_topology_sha256=topo.content_sha256, sigma_e=0.01, order=8
    )
    assert torch.equal(
        result.total_kcal_mol,
        result.polar_kcal_mol + result.cavity_kcal_mol + result.dispersion_kcal_mol,
    )
    assert result.total_kcal_mol.requires_grad
    assert result.cha.smoothed_signs.shape == (3,)
    assert result.cha.charge_sigma_e == 0.01
    assert result.cha.weighted_signs_over_sigma.shape == (3,)
    assert result.cha.gaussian_erfc_tails.shape == (3,)
    assert result.cha.gaussian_erfcx_tails.shape == (3,)
    force = -torch.autograd.grad(result.total_kcal_mol, xyz)[0]
    assert torch.isfinite(force).all()
    assert result.point_domain.active_pairs == ((0, 1),)
    assert result.quadrature_identity.r6_order == 8


def test_default_order_is_64_and_sigma_changes_only_new_polar_identity():
    topo = topology()
    narrow = continuum_gaussian_cha_scalar(
        positions(), topo, expected_topology_sha256=topo.content_sha256, sigma_e=0.001
    )
    wide = continuum_gaussian_cha_scalar(
        positions(), topo, expected_topology_sha256=topo.content_sha256, sigma_e=0.01
    )
    assert narrow.quadrature_identity.r6_order == 64
    assert torch.equal(narrow.cavity_kcal_mol, wide.cavity_kcal_mol)
    assert torch.equal(narrow.dispersion_kcal_mol, wide.dispersion_kcal_mol)
    assert narrow.radius_provenance == wide.radius_provenance


def test_bad_topology_hash_and_existing_r6_geometry_guards_fail_closed():
    topo = topology()
    with pytest.raises(ValueError, match="external identity"):
        continuum_gaussian_cha_scalar(
            positions(), topo, expected_topology_sha256="f" * 64, sigma_e=0.01, order=8
        )
    bad = torch.tensor(
        [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(ContinuumChaDomainError):
        continuum_gaussian_cha_scalar(
            bad,
            topo,
            expected_topology_sha256=topo.content_sha256,
            sigma_e=0.01,
            order=8,
        )
