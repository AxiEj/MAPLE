"""Closed Gaussian-CHA R6 numerical-profile integration contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import numpy as np
import pytest
from ase import Atoms

torch = pytest.importorskip("torch")

from maple.function.calculator.extra_correction.implicit.continuum_chagb_inputs import (
    ContinuumChaTopology,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_correction import (
    GaussianChaCorrection,
)
from maple.function.calculator.extra_correction.implicit.gaussian_cha_profiles import (
    GAUSSIAN_CHA_R6_V1_PROFILE_ID,
    GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    GaussianChaNumericalProfile,
    resolve_gaussian_cha_profile,
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


def _topology():
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


def _positions():
    return torch.tensor(
        [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=torch.float64,
    )


def _atoms():
    atoms = Atoms("OHH", positions=_positions().numpy())
    atoms.new_array(
        "_maple_implicit_atom_identity", np.asarray([1, 2, 3], dtype=np.int64)
    )
    return atoms


def test_profile_registry_is_exact_closed_and_unknown_fails_before_backend_import():
    v1 = resolve_gaussian_cha_profile(GAUSSIAN_CHA_R6_V1_PROFILE_ID)
    v2 = resolve_gaussian_cha_profile(GAUSSIAN_CHA_R6_V2_PROFILE_ID)
    assert v1.profile_id == "gaussian-cha-r6-v1"
    assert v2.profile_id == (
        "gaussian-cha-r6-derivative-v2-numerical-profile-20261001.2-direct-complement"
    )
    assert v1 != v2
    with pytest.raises(ValueError, match="Unknown Gaussian-CHA numerical profile"):
        resolve_gaussian_cha_profile("gaussian-cha-r6-v3-unknown")
    with pytest.raises(ValueError, match="Unknown Gaussian-CHA numerical profile"):
        GaussianChaNumericalProfile("gaussian-cha-r6-v3-unknown")


def test_default_scalar_is_fieldwise_identical_to_explicit_v1_profile():
    topology = _topology()
    implicit = continuum_gaussian_cha_scalar(
        _positions(),
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=8,
    )
    explicit = continuum_gaussian_cha_scalar(
        _positions(),
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=8,
        numerical_profile_id=GAUSSIAN_CHA_R6_V1_PROFILE_ID,
    )
    for field in (
        "polar_kcal_mol",
        "cavity_kcal_mol",
        "dispersion_kcal_mol",
        "total_kcal_mol",
    ):
        assert torch.equal(getattr(implicit, field), getattr(explicit, field))
    assert torch.equal(
        implicit.cha.born_radii_angstrom, explicit.cha.born_radii_angstrom
    )
    assert implicit.numerical_profile_id == explicit.numerical_profile_id
    assert implicit.r6_backend_diagnostics == explicit.r6_backend_diagnostics


def test_correction_profile_is_immutable_forwarded_and_provenance_bound():
    topology = _topology()
    atoms = _atoms()
    correction = GaussianChaCorrection(
        atoms,
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=8,
        numerical_profile_id=GAUSSIAN_CHA_R6_V1_PROFILE_ID,
    )
    result = correction.evaluate(atoms, need_forces=True)
    assert correction.numerical_profile_id == GAUSSIAN_CHA_R6_V1_PROFILE_ID
    assert result.provenance["numerical_profile_id"] == GAUSSIAN_CHA_R6_V1_PROFILE_ID
    assert result.provenance["r6_backend_diagnostics"]["profile_id"] == (
        GAUSSIAN_CHA_R6_V1_PROFILE_ID
    )
    with pytest.raises((AttributeError, TypeError)):
        setattr(
            cast(Any, correction), "numerical_profile_id", GAUSSIAN_CHA_R6_V2_PROFILE_ID
        )


def test_explicit_v2_profile_uses_only_the_new_closed_backend():
    topology = _topology()
    result = continuum_gaussian_cha_scalar(
        _positions(),
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=8,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )
    assert result.numerical_profile_id == GAUSSIAN_CHA_R6_V2_PROFILE_ID
    assert result.r6_backend_diagnostics.profile_id == GAUSSIAN_CHA_R6_V2_PROFILE_ID
    assert result.r6_backend_diagnostics.backend_function == (
        "torch_continuum_r6_derivative_v2._r6_inverse_born_v2"
    )
    assert torch.isfinite(result.total_kcal_mol)
    atoms = _atoms()
    correction = GaussianChaCorrection(
        atoms,
        topology,
        expected_topology_sha256=topology.content_sha256,
        sigma_e=0.01,
        order=8,
        numerical_profile_id=GAUSSIAN_CHA_R6_V2_PROFILE_ID,
    )
    corrected = correction.evaluate(atoms, need_forces=True)
    assert corrected.provenance["numerical_profile_id"] == GAUSSIAN_CHA_R6_V2_PROFILE_ID
    assert corrected.provenance["r6_backend_diagnostics"] == {
        "profile_id": GAUSSIAN_CHA_R6_V2_PROFILE_ID,
        "backend_function": "torch_continuum_r6_derivative_v2._r6_inverse_born_v2",
        "order": 8,
    }
