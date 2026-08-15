from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from maple.solvation.api.profiles import MACE_POLAR_RADIAL_GTO_COUPLING_ID
from maple.solvation.coupling.variational_adapters import (
    ScalarFirstElectronicResponseAdapter,
)
from maple.solvation.models.base import ModelDomain, ModelProvenance
from maple.solvation.models.mace_polar_variational import (
    MACE_POLAR_VARIATIONAL_DUALITY_MAP,
)
from maple.solvation.models.passive_polarization import (
    PASSIVE_QUADRATIC_CONSTRUCTION_ID,
    PASSIVE_QUADRATIC_FIELD_ENERGY_PROVIDER_ID,
    PassiveHeadTensors,
    PassiveQuadraticFieldEnergy,
)

ROOT = Path(__file__).resolve().parents[2]


def _hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class SyntheticAuditedPassiveHead:
    """Contract oracle only; this is not chemical or admission evidence."""

    __slots__ = (
        "_factor_scale",
        "_permanent_scale",
        "coupling_id",
        "duality_map_sha256",
        "model_profile_id",
        "provenance",
        "provenance_sha256",
        "provider_id",
    )

    def __init__(self, *, audited: bool = True) -> None:
        self.provider_id = "test.route2.synthetic-trained-passive-head.v1"
        self.model_profile_id = "route2-model-synthetic-passive-head-test-v1"
        self.coupling_id = MACE_POLAR_RADIAL_GTO_COUPLING_ID
        self.duality_map_sha256 = (
            MACE_POLAR_VARIATIONAL_DUALITY_MAP.configuration_sha256()
        )
        self._factor_scale = 0.17
        self._permanent_scale = 0.023
        self.provenance = ModelProvenance(
            provider_id=self.provider_id,
            model_profile_id=self.model_profile_id,
            model_family="synthetic-passive-head-contract-oracle",
            checkpoint_sha256="1" * 64,
            upstream_version="test-only",
            upstream_commit="test-only",
            inference_code_sha256="2" * 64,
            dtype="float64",
            device="cpu",
            domain=ModelDomain((1, 6), (0, 0), (1,)),
            field_convention=(
                MACE_POLAR_VARIATIONAL_DUALITY_MAP.field_space.field_convention
            ),
            coordinate_frame_policy="laboratory Cartesian Angstrom",
            optimizer_parameter_groups_audited=audited,
            optimizer_audit_evidence_sha256="3" * 64 if audited else None,
        )
        self.provenance_sha256 = self.provenance.sha256

    def parameter_state_sha256(self) -> str:
        return _hash(
            {
                "factor_scale": self._factor_scale,
                "permanent_scale": self._permanent_scale,
            }
        )

    def configuration_sha256(self) -> str:
        return _hash(
            {
                "provider_id": self.provider_id,
                "model_profile_id": self.model_profile_id,
                "coupling_id": self.coupling_id,
                "duality_map_sha256": self.duality_map_sha256,
                "provenance_sha256": self.provenance_sha256,
                "parameter_state_sha256": self.parameter_state_sha256(),
                "construction": "synthetic-test-coefficients-only",
            }
        )

    def coefficients_torch(self, geometry, *, reduced_dimension: int):
        torch = importlib.import_module("torch")
        dtype = geometry.positions.dtype
        device = geometry.positions.device
        coordinate_scale = 1.0 + 0.01 * torch.sum(geometry.positions**2)
        permanent = (
            self._permanent_scale
            * coordinate_scale
            * torch.linspace(
                -1.0,
                1.0,
                reduced_dimension,
                dtype=dtype,
                device=device,
            )
        )
        diagonal = (
            self._factor_scale
            * coordinate_scale
            * (
                1.0
                + torch.arange(reduced_dimension, dtype=dtype, device=device)
                / (3.0 * reduced_dimension)
            )
        )
        factor = torch.diag(diagonal)
        vacuum = 0.04 * torch.sum(geometry.positions**2)
        return PassiveHeadTensors(vacuum, permanent, factor)


def _atoms() -> Atoms:
    return Atoms(
        "HC",
        positions=np.asarray([[0.1, -0.2, 0.3], [1.2, 0.4, -0.1]]),
        info={"charge": 0, "mult": 1},
    )


def _functional(*, audited: bool = True):
    torch = pytest.importorskip("torch")
    return PassiveQuadraticFieldEnergy(
        SyntheticAuditedPassiveHead(audited=audited),
        duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
        dtype=torch.float64,
        device="cpu",
    )


def test_passive_head_contract_imports_without_optional_model_runtimes() -> None:
    script = r"""
import sys
for name in ("torch", "mace", "aimnet2calc", "pyscf"):
    sys.modules[name] = None
from maple.solvation.models.passive_polarization import (
    PASSIVE_QUADRATIC_CONSTRUCTION_ID,
    PassiveHeadTensors,
    PassiveQuadraticFieldEnergy,
)
assert PASSIVE_QUADRATIC_CONSTRUCTION_ID.endswith("-v1")
assert PassiveHeadTensors.__name__ == "PassiveHeadTensors"
assert PassiveQuadraticFieldEnergy.__name__ == "PassiveQuadraticFieldEnergy"
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_untrained_or_unaudited_head_is_rejected_before_scalar_construction() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="optimizer parameter-group audit"):
        PassiveQuadraticFieldEnergy(
            SyntheticAuditedPassiveHead(audited=False),
            duality_map=MACE_POLAR_VARIATIONAL_DUALITY_MAP,
            dtype=torch.float64,
            device="cpu",
        )

    negative_sign = replace(MACE_POLAR_VARIATIONAL_DUALITY_MAP, conjugacy_sign=-1)
    with pytest.raises(ValueError, match="positive energy-dual"):
        PassiveQuadraticFieldEnergy(
            SyntheticAuditedPassiveHead(),
            duality_map=negative_sign,
            dtype=torch.float64,
            device="cpu",
        )


def test_quadratic_scalar_structurally_generates_source_and_passive_response() -> None:
    functional = _functional()
    atoms = _atoms()
    coordinates = functional.duality_map.coordinates(
        atom_count=len(atoms), total_charge=0.0
    )
    reduced = np.linspace(-0.02, 0.03, coordinates.reduced_dimension)
    direction = np.linspace(0.04, -0.01, coordinates.reduced_dimension)

    directional = functional.energy_directional_derivative(
        atoms, reduced, direction, total_charge=0.0
    )
    step = 2.0e-5
    finite_difference = (
        functional.energy_eV(atoms, reduced + step * direction, total_charge=0.0)
        - functional.energy_eV(atoms, reduced - step * direction, total_charge=0.0)
    ) / (2.0 * step)
    assert directional == pytest.approx(finite_difference, abs=2.0e-11)

    dense_hessian = np.column_stack(
        [
            functional.field_hvp(
                atoms,
                reduced,
                np.eye(coordinates.reduced_dimension)[index],
                total_charge=0.0,
            )
            for index in range(coordinates.reduced_dimension)
        ]
    )
    geometry = functional._geometry(atoms, requires_grad=False)
    factor = functional.trained_head.coefficients_torch(
        geometry, reduced_dimension=coordinates.reduced_dimension
    ).response_factor
    expected_hessian = -np.asarray(factor.detach().cpu()).T @ np.asarray(
        factor.detach().cpu()
    )
    np.testing.assert_allclose(dense_hessian, expected_hessian, atol=3.0e-16)
    np.testing.assert_allclose(dense_hessian, dense_hessian.T, atol=3.0e-16)
    assert np.max(np.linalg.eigvalsh(dense_hessian)) < 1.0e-14
    assert np.min(np.linalg.eigvalsh(-dense_hessian)) > 0.0

    source_direction = functional.source_jvp(
        atoms, reduced, direction, total_charge=0.0
    )
    rng = np.random.default_rng(20260815)
    source_cotangent = rng.normal(size=(len(atoms), 8))
    reduced_vjp = functional.source_vjp(
        atoms,
        reduced,
        source_cotangent,
        total_charge=0.0,
    )
    assert np.vdot(source_cotangent, source_direction) == pytest.approx(
        np.vdot(reduced_vjp, direction), abs=2.0e-14
    )

    certificate = functional.pointwise_passivity_certificate(atoms, total_charge=0.0)
    assert certificate.provider_id == PASSIVE_QUADRATIC_FIELD_ENERGY_PROVIDER_ID
    assert certificate.structural_passivity is True
    assert certificate.strict_passivity_at_this_point is True
    assert certificate.response_factor_rank == coordinates.reduced_dimension
    assert certificate.susceptibility_min_eigenvalue_lower_bound > 0.0
    assert certificate.global_domain_passivity_admitted is False
    assert certificate.coupled_root_stability_admitted is False
    assert certificate.public_capability_admitted is False
    assert functional.capabilities.enabled_tiers == ()
    assert functional.variational_functional_admitted is False


def test_zero_field_source_is_the_head_permanent_covector_on_fixed_charge_chart() -> (
    None
):
    functional = _functional()
    atoms = _atoms()
    coordinates = functional.duality_map.coordinates(
        atom_count=len(atoms), total_charge=0.0
    )
    zero = np.zeros(coordinates.reduced_dimension)
    geometry = functional._geometry(atoms, requires_grad=False)
    coefficients = functional.trained_head.coefficients_torch(
        geometry, reduced_dimension=coordinates.reduced_dimension
    )
    expected = functional.duality_map.source_from_reduced_covector(
        np.asarray(coefficients.permanent_reduced_covector.detach().cpu()),
        atom_count=len(atoms),
        total_charge=0.0,
    )
    actual = functional.source_from_energy(atoms, zero, total_charge=0.0)
    np.testing.assert_allclose(actual, expected, atol=2.0e-16, rtol=0.0)
    assert functional.source_space.total_charge(actual, atom_count=len(atoms)) == (
        pytest.approx(0.0, abs=2.0e-15)
    )


def test_coordinate_derivatives_come_from_the_same_passive_scalar_graph() -> None:
    functional = _functional()
    atoms = _atoms()
    coordinates = functional.duality_map.coordinates(
        atom_count=len(atoms), total_charge=0.0
    )
    reduced = np.linspace(-0.015, 0.021, coordinates.reduced_dimension)
    rng = np.random.default_rng(91)
    position_direction = rng.normal(size=(len(atoms), 3))
    position_direction /= np.linalg.norm(position_direction)
    source_cotangent = rng.normal(size=(len(atoms), 8))
    fixed_gradient = functional.fixed_field_coordinate_gradient(
        atoms, reduced, total_charge=0.0
    )
    mixed = functional.mixed_coordinate_field_vjp(
        atoms,
        reduced,
        source_cotangent,
        total_charge=0.0,
    )
    step = 2.0e-5
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions += step * position_direction
    minus.positions -= step * position_direction
    energy_fd = (
        functional.energy_eV(plus, reduced, total_charge=0.0)
        - functional.energy_eV(minus, reduced, total_charge=0.0)
    ) / (2.0 * step)
    source_fd = np.vdot(
        source_cotangent,
        (
            functional.source_from_energy(plus, reduced, total_charge=0.0)
            - functional.source_from_energy(minus, reduced, total_charge=0.0)
        )
        / (2.0 * step),
    )
    assert np.vdot(fixed_gradient, position_direction) == pytest.approx(
        energy_fd, abs=2.0e-10
    )
    assert np.vdot(mixed, position_direction) == pytest.approx(source_fd, abs=2.0e-10)


def test_generic_scalar_adapter_accepts_the_head_without_model_name_branching() -> None:
    functional = _functional()
    adapter = ScalarFirstElectronicResponseAdapter(functional)
    assert adapter.functional is functional
    assert adapter.model_profile_id == functional.model_profile_id
    assert adapter.coupling_id == MACE_POLAR_RADIAL_GTO_COUPLING_ID
    assert adapter.source_space == functional.source_space
    assert adapter.field_space == functional.field_space
    assert len(adapter.configuration_sha256()) == 64


def test_parameter_drift_and_derivative_override_attempts_fail_closed() -> None:
    functional = _functional()
    head = functional.trained_head
    object.__setattr__(head, "_factor_scale", 0.29)
    with pytest.raises(RuntimeError, match="configuration drifted"):
        functional.configuration_sha256()

    with pytest.raises(TypeError, match="is final"):

        class BadOverride(PassiveQuadraticFieldEnergy):
            def _energy_torch(self, geometry, reduced_field):
                return reduced_field.sum()


def test_contract_ids_are_stable_and_not_an_admitted_scientific_profile() -> None:
    assert PASSIVE_QUADRATIC_CONSTRUCTION_ID == (
        "route2-scalar-first-passive-quadratic-reduced-field-v1"
    )
    assert PASSIVE_QUADRATIC_FIELD_ENERGY_PROVIDER_ID == (
        "maple.route2.models.passive-quadratic-field-energy.v1"
    )
