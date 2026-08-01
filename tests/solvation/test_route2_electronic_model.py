from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.route2_electronic_model import (
    ATOMIC_L1_SOURCE_SPACE,
    COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL,
    FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    Route2ElectronicModelCapabilities,
    Route2ElectronicModelDescriptor,
    resolve_route2_electronic_model,
    validate_route2_electronic_model_capabilities,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
    Route2EngineSettings,
)
from maple.function.route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
)
from maple.function.route2_model_contracts import (
    ROUTE2_MACE_POLAR_MODEL_FAMILY,
    validate_route2_input_model_family,
)


class _ZeroReactionField:
    atom_count = 2
    runtime_provenance = {"provider": "synthetic-zero-field"}

    @staticmethod
    def apply_scf(density):
        return np.zeros_like(np.asarray(density, dtype=float))

    @staticmethod
    def scf_polarization_energy_hartree(_density):
        return 0.0


class _AlternativeChargeFieldMLIPAdapter:
    """A non-MACE adapter with a deliberately unrelated native API."""

    descriptor = Route2ElectronicModelDescriptor(
        adapter_name="alternative-charge-field-mlip-adapter-v1",
        model_family="alternative-charge-field-mlip",
        field_evaluator="alternative-local-field-v1",
        source_space=ATOMIC_L1_SOURCE_SPACE,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"local-jet"}),
        ),
        energy_semantics=FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        profile_binding="alternative-route2-profile-v1",
    )

    def __init__(self, source):
        self.source = np.asarray(source, dtype=float)
        self.cache_identity = id(self)
        self.drives = []

    def cached_state(self, _atoms, *, require_forces=False):
        assert require_forces is False
        return SimpleNamespace(
            energy_ev=-1.0,
            density_coefficients=self.source.copy(),
            dipole_e_angstrom=np.zeros(3),
            fixed_field_forces_ev_per_angstrom=None,
        )

    def evaluate_state(self, _atoms, drive, *, compute_forces=False):
        assert compute_forces is False
        self.drives.append(drive)
        return self.cached_state(_atoms), {"native_api": "external-field-charges"}

    def preprojected_field_projector(self):
        raise NotImplementedError

    def linearize_source_response(self, _atoms, _drive):
        raise NotImplementedError

    def field_conditioned_energy_field_gradient(self, _atoms, _drive):
        raise NotImplementedError

    def source_position_vjp(self, _atoms, _drive, *, source_cotangent):
        del source_cotangent
        raise NotImplementedError


def _settings() -> Route2EngineSettings:
    return Route2EngineSettings(
        continuum_label="synthetic-zero-continuum",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=2,
        adjoint_relative_tolerance=1.0e-10,
        adjoint_absolute_tolerance=1.0e-13,
        adjoint_max_iterations=20,
        energy_identity_tolerance_ev=1.0e-12,
        force_state_energy_tolerance_ev=1.0e-12,
        neutral_density_tolerance=1.0e-12,
        scf_require_two_energy_samples=False,
    )


def test_engine_accepts_an_explicit_non_mace_charge_field_adapter():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    source = np.asarray(
        [[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]],
        dtype=float,
    )
    adapter = _AlternativeChargeFieldMLIPAdapter(source)
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _ZeroReactionField(),
        cds_evaluator=lambda _atoms: SimpleNamespace(energy_hartree=0.0),
        settings=_settings(),
    )

    gas_state = engine.gas_state(adapter, atoms, need_forces=False)
    coupled = engine.solve_coupled_state(
        atoms,
        adapter,
        gas_state,
        provider_cache_signature=("alternative-mlip",),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )

    np.testing.assert_allclose(coupled.density_coefficients, source)
    assert coupled.electronic_model_identity == adapter.cache_identity
    assert adapter.drives[-1].projector == "local-jet"


def test_engine_builds_a_frozen_source_state_without_evaluating_a_field_state():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    source = np.asarray(
        [[-0.2, 0.1, -0.3, 0.4], [0.2, -0.5, 0.6, -0.7]],
        dtype=float,
    )
    adapter = _AlternativeChargeFieldMLIPAdapter(source)
    adapter.descriptor = replace(
        adapter.descriptor,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"exact-gto-v1"})
        ),
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _ZeroReactionField(),
        cds_evaluator=lambda _atoms: SimpleNamespace(energy_hartree=0.25),
        settings=_settings(),
    )

    gas_state = engine.gas_state(adapter, atoms, need_forces=False)
    coupled = engine.solve_frozen_source_state(
        atoms,
        adapter,
        gas_state,
        provider_cache_signature=("alternative-frozen-source",),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )

    np.testing.assert_allclose(coupled.density_coefficients, source)
    np.testing.assert_allclose(coupled.response_density_coefficients, source)
    np.testing.assert_allclose(coupled.reaction_field_values_ev, 0.0)
    assert coupled.response_mode == "frozen"
    assert coupled.fixed_point_applicable is False
    assert coupled.history == ()
    assert coupled.scf_convergence == {
        "reason": "frozen-source-no-fixed-point-v1",
        "fixed_point_applicable": False,
        "source_label": "gas-electronic-source",
    }
    assert coupled.solvent_state is gas_state
    assert coupled.cds_result.energy_hartree == pytest.approx(0.25)
    assert adapter.drives == []


def test_frozen_source_capability_gate_does_not_require_a_field_projector():
    adapter = _AlternativeChargeFieldMLIPAdapter(np.zeros((2, 4)))
    adapter.descriptor = replace(
        adapter.descriptor,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"exact-gto-v1"})
        ),
    )

    validate_route2_electronic_model_capabilities(
        adapter,
        expected_model_family="alternative-charge-field-mlip",
        expected_source_space=ATOMIC_L1_SOURCE_SPACE.name,
        expected_profile_binding="alternative-route2-profile-v1",
        expected_field_evaluator="alternative-local-field-v1",
        expected_energy_semantics=FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        reaction_field_projector="local-jet",
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        response_mode="frozen",
        need_forces=False,
    )


def test_resolver_rejects_accidental_polar_state_duck_typing():
    accidental = SimpleNamespace(polar_state=lambda *_args, **_kwargs: None)
    with pytest.raises(TypeError, match="validated descriptor"):
        resolve_route2_electronic_model(accidental)


def test_input_model_selection_uses_the_same_family_identity_as_profiles():
    assert (
        validate_route2_input_model_family(
            "macepol-m",
            expected_model_family=ROUTE2_MACE_POLAR_MODEL_FAMILY,
        )
        == ROUTE2_MACE_POLAR_MODEL_FAMILY
    )
    with pytest.raises(ValueError, match="not registered for Route 2"):
        validate_route2_input_model_family(
            "unregistered-field-mlip",
            expected_model_family=ROUTE2_MACE_POLAR_MODEL_FAMILY,
        )


def test_profile_validation_is_model_family_and_capability_specific():
    adapter = _AlternativeChargeFieldMLIPAdapter(np.zeros((2, 4)))
    common = {
        "expected_source_space": ATOMIC_L1_SOURCE_SPACE.name,
        "expected_profile_binding": "alternative-route2-profile-v1",
        "expected_field_evaluator": "alternative-local-field-v1",
        "expected_energy_semantics": FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        "reaction_field_projector": "local-jet",
        "electrostatic_energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
    }

    validate_route2_electronic_model_capabilities(
        adapter,
        expected_model_family="alternative-charge-field-mlip",
        need_forces=False,
        **common,
    )
    with pytest.raises(TypeError, match="profile/model mismatch"):
        validate_route2_electronic_model_capabilities(
            adapter,
            expected_model_family="mace-polar-1",
            need_forces=False,
            **common,
        )
    with pytest.raises(TypeError, match="source-response-jvp-vjp"):
        validate_route2_electronic_model_capabilities(
            adapter,
            expected_model_family="alternative-charge-field-mlip",
            need_forces=True,
            **common,
        )


def test_legacy_ledger_requires_the_extra_energy_derivative_capability():
    adapter = _AlternativeChargeFieldMLIPAdapter(np.zeros((2, 4)))
    with pytest.raises(TypeError, match="field-conditioned-energy-field-gradient"):
        validate_route2_electronic_model_capabilities(
            adapter,
            expected_model_family="alternative-charge-field-mlip",
            expected_source_space=ATOMIC_L1_SOURCE_SPACE.name,
            expected_profile_binding="alternative-route2-profile-v1",
            expected_field_evaluator="alternative-local-field-v1",
            expected_energy_semantics=FIELD_CONDITIONED_OPERATIONAL_ENERGY,
            reaction_field_projector="local-jet",
            electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
            need_forces=True,
        )


def test_projector_capabilities_do_not_overclaim_exact_gto_forces():
    adapter = _AlternativeChargeFieldMLIPAdapter(np.zeros((2, 4)))
    adapter.descriptor = replace(
        adapter.descriptor,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"local-jet", "exact-gto-v1"}),
            response_projectors=frozenset({"local-jet", "exact-gto-v1"}),
            position_vjp_projectors=frozenset({"local-jet"}),
            fixed_field_force_projectors=frozenset({"local-jet"}),
            energy_gradient_projectors=frozenset({"local-jet"}),
        ),
    )
    common = {
        "expected_model_family": "alternative-charge-field-mlip",
        "expected_source_space": ATOMIC_L1_SOURCE_SPACE.name,
        "expected_profile_binding": "alternative-route2-profile-v1",
        "expected_field_evaluator": "alternative-local-field-v1",
        "expected_energy_semantics": FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        "reaction_field_projector": "exact-gto-v1",
        "electrostatic_energy_ledger": PCM_HALF_COUPLING_ONLY_V1,
    }

    validate_route2_electronic_model_capabilities(
        adapter,
        need_forces=False,
        **common,
    )
    with pytest.raises(TypeError, match="source-position-vjp"):
        validate_route2_electronic_model_capabilities(
            adapter,
            need_forces=True,
            **common,
        )


def test_projector_derivative_capabilities_must_be_state_capabilities():
    with pytest.raises(ValueError, match="subset of state_projectors"):
        Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"local-jet"}),
            response_projectors=frozenset({"exact-gto-v1"}),
        )


def test_descriptor_provenance_serializes_capabilities_deterministically():
    descriptor = _AlternativeChargeFieldMLIPAdapter.descriptor

    provenance = descriptor.as_provenance()

    json.dumps(provenance, allow_nan=False)
    assert provenance["model_family"] == "alternative-charge-field-mlip"
    assert provenance["source_space"] == ATOMIC_L1_SOURCE_SPACE.name
    assert provenance["capabilities"] == {
        "state_projectors": ["local-jet"],
        "gas_forces": False,
        "response_projectors": [],
        "position_vjp_projectors": [],
        "fixed_field_force_projectors": [],
        "energy_gradient_projectors": [],
    }


def test_direct_pcm_force_does_not_require_field_conditioned_force_terms():
    adapter = _AlternativeChargeFieldMLIPAdapter(np.zeros((2, 4)))
    adapter.descriptor = replace(
        adapter.descriptor,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"local-jet"}),
            gas_forces=True,
            response_projectors=frozenset({"local-jet"}),
            position_vjp_projectors=frozenset({"local-jet"}),
        ),
    )
    common = {
        "expected_model_family": "alternative-charge-field-mlip",
        "expected_source_space": ATOMIC_L1_SOURCE_SPACE.name,
        "expected_profile_binding": "alternative-route2-profile-v1",
        "expected_field_evaluator": "alternative-local-field-v1",
        "expected_energy_semantics": FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        "reaction_field_projector": "local-jet",
        "need_forces": True,
    }

    validate_route2_electronic_model_capabilities(
        adapter,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        **common,
    )
    with pytest.raises(TypeError, match="fixed-field-forces"):
        validate_route2_electronic_model_capabilities(
            adapter,
            electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
            **common,
        )


def test_descriptor_takes_an_immutable_deep_provenance_snapshot():
    mutable = {"checkpoint": {"sha256": "abc"}, "tags": ["frozen"]}
    descriptor = replace(
        _AlternativeChargeFieldMLIPAdapter.descriptor,
        provenance=mutable,
    )
    mutable["checkpoint"]["sha256"] = "changed"
    mutable["tags"].append("changed")

    assert descriptor.as_provenance()["provenance"] == {
        "checkpoint": {"sha256": "abc"},
        "tags": ["frozen"],
    }
    with pytest.raises(TypeError):
        descriptor.provenance["checkpoint"]["sha256"] = "changed"


def test_fixed_point_engine_rejects_a_variational_kkt_model_semantics():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    adapter = _AlternativeChargeFieldMLIPAdapter(np.zeros((2, 4)))
    adapter.descriptor = replace(
        adapter.descriptor,
        energy_semantics=COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL,
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _ZeroReactionField(),
        cds_evaluator=lambda _atoms: SimpleNamespace(energy_hartree=0.0),
        settings=_settings(),
    )

    gas_state = engine.gas_state(adapter, atoms, need_forces=False)
    with pytest.raises(TypeError, match="separate KKT engine"):
        engine.solve_coupled_state(
            atoms,
            adapter,
            gas_state,
            provider_cache_signature=("wrong-engine",),
        )
