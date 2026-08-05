from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from maple.function.calculator.extra_correction.implicit.electrostatic_pairing import (
    MACE_POLAR_L1_PAIRING,
)
from maple.function.calculator.extra_correction.implicit.route2_electronic_model import (
    ATOMIC_L1_SOURCE_SPACE,
    FIELD_CONDITIONED_OPERATIONAL_ENERGY,
    Route2ElectronicModelCapabilities,
    Route2ElectronicModelDescriptor,
    resolve_route2_electronic_model,
)
from maple.function.calculator.extra_correction.implicit.route2_field_state import (
    LocalReactionField,
)
from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
    Route2EngineSettings,
)
from maple.function.calculator.extra_correction.implicit.route2_plugin import (
    ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
    ATOMIC_L1_PLUGIN_SOURCE_SPACE,
    ArtifactDispositionRecord,
    ArtifactDispositionRegistry,
    LegacyAtomicL1BridgeSpec,
    ContinuumCoupling,
    DEFAULT_ROUTE2_PLUGIN_REGISTRY,
    MACE_P0_ARTIFACT_ID,
    MACEPolarPlugin,
    PluginCapabilityDeclaration,
    PluginContractError,
    PluginElectronicState,
    PluginProfile,
    PluginProvenance,
    Route2PluginRegistry,
    adapt_atomic_l1_plugin_to_legacy_engine,
    admit_plugin,
    born_point_charge_sign_canary,
    load_mace_p0_certificate,
    select_uniform_field_convention,
)
from maple.function.route2_energy_ledger import PCM_HALF_COUPLING_ONLY_V1
from maple.function.route2_model_contracts import (
    ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL,
    ROUTE2_MACE_POLAR_MODEL_FAMILY,
    ROUTE2_MACE_POLAR_PROFILE_BINDING,
)


def _operational_provenance(**updates):
    values = {
        "checkpoint_sha256": "synthetic-checkpoint-sha",
        "training_code_sha": "synthetic-training-sha",
        "inference_code_sha": "synthetic-inference-sha",
        "representation": ATOMIC_L1_PLUGIN_SOURCE_SPACE.representation,
        "supported_atomic_numbers": frozenset({1, 8}),
        "total_charge_domain": (0.0, 0.0),
        "field_convention": "potential-gradient",
        "potential_unit": ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.potential_unit,
        "gradient_unit": ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.gradient_unit,
        "energy_semantics": FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        "optimizer_coverage": "not-applicable",
        "extra": {"nested": {"tags": ["synthetic"]}},
    }
    values.update(updates)
    return PluginProvenance(**values)


class _SyntheticResponsePlugin:
    plugin_id = "synthetic-response-mlip-v1"
    source_space = ATOMIC_L1_PLUGIN_SOURCE_SPACE
    field_dual_space = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE
    provenance = _operational_provenance()
    capabilities = PluginCapabilityDeclaration(field_response=True)

    def cached_source_state(self, atoms, *, require_forces=False):
        assert require_forces is False
        return PluginElectronicState(
            energy_ev=-1.0,
            source=np.zeros((len(atoms), 4)),
            dipole_e_angstrom=np.zeros(3),
        )

    def evaluate_source_state(self, atoms, field, *, compute_forces=False):
        assert compute_forces is False
        source = np.zeros((len(atoms), 4))
        if field is not None:
            source[:, 1:] = 0.1 * field.potential_gradient_ev_per_angstrom
        return (
            PluginElectronicState(
                energy_ev=-1.0,
                source=source,
                dipole_e_angstrom=np.sum(source[:, 1:], axis=0),
            ),
            {"provider": self.plugin_id},
        )

    def linearize_source_response(self, atoms, field):
        del atoms, field
        return SimpleNamespace(jvp=lambda direction: 0.1 * np.asarray(direction))


class _SyntheticVariationalPlugin(_SyntheticResponsePlugin):
    plugin_id = "synthetic-variational-mlip-v1"
    provenance = _operational_provenance(
        energy_semantics=ROUTE2_COMMON_VARIATIONAL_ELECTRONIC_FUNCTIONAL
    )
    capabilities = PluginCapabilityDeclaration(
        field_response=True,
        fixed_field_forces=True,
        source_position_vjp=True,
        feature_vjp=True,
        variational_functional=True,
    )

    def fixed_field_forces(self, atoms, field):
        del field
        return np.zeros((len(atoms), 3))

    def source_position_vjp(self, atoms, field, *, source_cotangent):
        del field, source_cotangent
        return np.zeros((len(atoms), 3))

    def feature_vjp(self, atoms, field, *, feature_cotangent):
        del atoms, field
        return np.asarray(feature_cotangent, dtype=float)

    def electronic_functional(self, atoms, source):
        del atoms
        return 0.5 * float(np.vdot(source, source))

    def electronic_source_gradient(self, atoms, source):
        del atoms
        return np.asarray(source, dtype=float)

    def electronic_source_hvp(self, atoms, source, direction):
        del atoms, source
        return np.asarray(direction, dtype=float)


class _FrozenOnlyPlugin:
    plugin_id = "synthetic-frozen-source-v1"
    source_space = ATOMIC_L1_PLUGIN_SOURCE_SPACE
    field_dual_space = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE
    provenance = _operational_provenance(field_convention="unattested")
    capabilities = PluginCapabilityDeclaration()

    def cached_source_state(self, atoms, *, require_forces=False):
        del require_forces
        return PluginElectronicState(
            energy_ev=0.0,
            source=np.zeros((len(atoms), 4)),
            dipole_e_angstrom=np.zeros(3),
        )

    def evaluate_source_state(self, atoms, field, *, compute_forces=False):
        assert field is None
        assert compute_forces is False
        return self.cached_source_state(atoms), {"mode": "frozen"}


class _ZeroReactionField:
    atom_count = 3
    runtime_provenance = {"provider": "synthetic-zero-continuum"}

    @staticmethod
    def apply_scf(source):
        return np.zeros_like(np.asarray(source, dtype=float))

    @staticmethod
    def scf_polarization_energy_hartree(source):
        del source
        return 0.0


def _engine_settings():
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


def _coupling(*, adjoint=True):
    matrix = np.arange(1.0, 25.0).reshape(3, 8) / 17.0

    def source_to_surface(source):
        return matrix @ np.asarray(source).reshape(-1)

    def surface_to_dual(surface):
        raw_density_order = (matrix.T @ np.asarray(surface)).reshape(2, 4)
        field = MACE_POLAR_L1_PAIRING.density_to_field_order(raw_density_order)
        if not adjoint:
            field = field.copy()
            field[0, 1] += 0.25
        return field

    return ContinuumCoupling(
        name="synthetic-B/Bstar-v1",
        source_space=ATOMIC_L1_PLUGIN_SOURCE_SPACE,
        field_dual_space=ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
        source_to_surface=source_to_surface,
        surface_to_field_dual=surface_to_dual,
        surface_pairing=lambda left, right: float(np.vdot(left, right)),
        surface_unit="synthetic-surface-coordinate",
        total_charge_constraint="sum atomic monopoles equals declared charge",
        gauge="continuum-zero-at-infinity",
    )


def _adjoint_probe():
    source = np.arange(8.0).reshape(2, 4) / 11.0
    surface = np.asarray([0.7, -0.3, 0.2])
    return source, surface, 2


_DERIVATIVE_TEST_PROFILE = PluginProfile(
    name="synthetic-derivative-admission-v1",
    allowed_routes=frozenset({"response-only", "operational-force", "variational"}),
)


def test_local_reaction_field_is_an_immutable_potential_gradient_jet():
    raw = np.asarray([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    field = LocalReactionField.from_nodewise_jet(raw)
    raw[:] = 0.0

    np.testing.assert_allclose(
        field.as_nodewise_jet(),
        [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]],
    )
    assert field.potential_ev.flags.writeable is False
    assert field.potential_gradient_ev_per_angstrom.flags.writeable is False


def test_physical_electric_field_conversion_is_debug_neutral_and_unledgered_only():
    potential = np.zeros(2)
    electric = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

    field = LocalReactionField.from_physical_electric_field_for_debug(
        potential,
        electric,
        total_charge_e=0.0,
        debug_unledgered=True,
    )
    np.testing.assert_allclose(field.potential_gradient_ev_per_angstrom, -electric)
    with pytest.raises(ValueError, match="unledgered debug"):
        LocalReactionField.from_physical_electric_field_for_debug(
            potential,
            electric,
            total_charge_e=0.0,
            debug_unledgered=False,
        )
    with pytest.raises(ValueError, match="neutral systems"):
        LocalReactionField.from_physical_electric_field_for_debug(
            potential,
            electric,
            total_charge_e=1.0,
            debug_unledgered=True,
        )


def test_p0_born_sign_and_field_convention_selection_are_fail_closed():
    positive = born_point_charge_sign_canary(charge_e=1.0)
    negative = born_point_charge_sign_canary(charge_e=-1.0)

    assert positive["passed"] is True
    assert negative["passed"] is True
    assert positive["reaction_potential_hartree_per_e"] < 0.0
    assert negative["reaction_potential_hartree_per_e"] > 0.0
    assert positive["half_coupling_energy_hartree"] < 0.0
    assert negative["half_coupling_energy_hartree"] < 0.0

    assert (
        select_uniform_field_convention(
            positive_mapping_errors=[1.0e-13, 2.0e-13],
            negative_mapping_errors=[1.0e-4, 2.0e-4],
            absolute_tolerance=1.0e-10,
            minimum_discrimination_ratio=100.0,
        )
        == "potential-gradient"
    )
    assert (
        select_uniform_field_convention(
            positive_mapping_errors=[1.0e-4, 2.0e-4],
            negative_mapping_errors=[1.0e-13, 2.0e-13],
            absolute_tolerance=1.0e-10,
            minimum_discrimination_ratio=100.0,
        )
        == "physical-electric-field"
    )
    with pytest.raises(PluginContractError, match="did not select a unique"):
        select_uniform_field_convention(
            positive_mapping_errors=[1.0e-13],
            negative_mapping_errors=[2.0e-13],
            absolute_tolerance=1.0e-10,
            minimum_discrimination_ratio=100.0,
        )


def test_atomic_l1_plugin_pairing_reuses_the_canonical_component_permutation():
    source = np.asarray([[0.0, 1.0, 2.0, 3.0]])
    field = np.asarray([[0.0, 10.0, 20.0, 30.0]])

    value = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.pair(
        source,
        field,
        source_space=ATOMIC_L1_PLUGIN_SOURCE_SPACE,
        atom_count=1,
    )

    assert value == pytest.approx(MACE_POLAR_L1_PAIRING.pair(source, field))
    assert value == pytest.approx(110.0)


def test_continuum_coupling_verifies_B_and_Bstar_in_declared_pairings():
    source, surface, atom_count = _adjoint_probe()

    evidence = _coupling().verify_adjoint(source, surface, atom_count=atom_count)
    bad_evidence = _coupling(adjoint=False).verify_adjoint(
        source, surface, atom_count=atom_count
    )

    assert evidence.passed is True
    assert evidence.absolute_error <= evidence.tolerance
    assert bad_evidence.passed is False


def test_non_mace_response_plugin_is_admitted_with_domain_and_provenance_checks():
    plugin = _SyntheticResponsePlugin()

    admission = admit_plugin(
        plugin,
        coupling=_coupling(),
        adjoint_probe=_adjoint_probe(),
        atomic_numbers=np.asarray([8, 1, 1]),
        total_charge_e=0.0,
    )

    assert admission.route == "response-only"
    assert admission.execution_mode == "scf"
    assert admission.coupling_evidence is not None
    assert admission.coupling_evidence.passed is True
    assert admission.diagnostics_only is False


def test_bad_adjoint_is_diagnostic_only_for_response_and_blocked_for_variational():
    response = admit_plugin(
        _SyntheticResponsePlugin(),
        coupling=_coupling(adjoint=False),
        adjoint_probe=_adjoint_probe(),
    )
    assert response.diagnostics_only is True

    with pytest.raises(PluginContractError, match=r"verified B/B\* adjoint"):
        admit_plugin(
            _SyntheticVariationalPlugin(),
            profile=_DERIVATIVE_TEST_PROFILE,
            route="variational",
            coupling=_coupling(adjoint=False),
            adjoint_probe=_adjoint_probe(),
        )


def test_variational_plugin_requires_and_passes_the_complete_derivative_layer():
    admission = admit_plugin(
        _SyntheticVariationalPlugin(),
        profile=_DERIVATIVE_TEST_PROFILE,
        route="variational",
        coupling=_coupling(),
        adjoint_probe=_adjoint_probe(),
    )

    assert admission.route == "variational"
    assert admission.diagnostics_only is False


def test_capability_downgrades_fail_closed_without_fake_methods():
    with pytest.raises(PluginContractError, match="does not admit route"):
        admit_plugin(_SyntheticVariationalPlugin(), route="variational")

    with pytest.raises(PluginContractError, match="fixed-field forces"):
        admit_plugin(
            _SyntheticResponsePlugin(),
            profile=_DERIVATIVE_TEST_PROFILE,
            route="operational-force",
        )

    accidental = _SyntheticResponsePlugin()
    accidental.capabilities = PluginCapabilityDeclaration(
        field_response=True,
        fixed_field_forces=True,
        source_position_vjp=True,
        feature_vjp=True,
    )
    with pytest.raises(PluginContractError, match=r"fixed_field_forces\(\)"):
        admit_plugin(
            accidental,
            profile=_DERIVATIVE_TEST_PROFILE,
            route="operational-force",
        )


def test_frozen_source_provider_needs_no_field_response_or_audited_field_sign():
    admission = admit_plugin(_FrozenOnlyPlugin(), execution_mode="frozen")
    assert admission.execution_mode == "frozen"
    with pytest.raises(PluginContractError, match="declared source response"):
        admit_plugin(_FrozenOnlyPlugin(), execution_mode="scf")


def test_field_conditioned_path_rejects_unattested_field_convention():
    plugin = _SyntheticResponsePlugin()
    plugin.provenance = replace(plugin.provenance, field_convention="unattested")

    with pytest.raises(PluginContractError, match="field convention is audited"):
        admit_plugin(plugin)


def test_element_charge_units_and_representation_are_admission_invariants():
    plugin = _SyntheticResponsePlugin()
    with pytest.raises(PluginContractError, match="excludes atomic numbers: 6"):
        admit_plugin(plugin, atomic_numbers=np.asarray([6]), total_charge_e=0.0)
    with pytest.raises(PluginContractError, match="excludes total charge"):
        admit_plugin(plugin, atomic_numbers=np.asarray([1]), total_charge_e=1.0)

    plugin.provenance = replace(plugin.provenance, potential_unit="hartree/e")
    with pytest.raises(PluginContractError, match="potential-unit mismatch"):
        admit_plugin(plugin)


def test_provenance_is_deeply_immutable_and_json_serializable():
    mutable = {"nested": {"tags": ["synthetic"]}}
    provenance = _operational_provenance(extra=mutable)
    mutable["nested"]["tags"].append("mutated")

    payload = provenance.as_provenance()
    json.dumps(payload, allow_nan=False)
    assert payload["extra"] == {"nested": {"tags": ["synthetic"]}}
    with pytest.raises(TypeError):
        provenance.extra["nested"]["new"] = "not allowed"


def test_registry_requires_explicit_factories_and_rejects_identity_substitution():
    registry = Route2PluginRegistry()
    registry.register(
        _SyntheticResponsePlugin.plugin_id,
        lambda: _SyntheticResponsePlugin(),
    )

    assert registry.create(_SyntheticResponsePlugin.plugin_id).plugin_id == (
        _SyntheticResponsePlugin.plugin_id
    )
    with pytest.raises(PluginContractError, match="No Route-2 plug-in factory"):
        registry.create("object-with-similar-methods")

    registry.register("wrong-id", lambda: _SyntheticResponsePlugin())
    with pytest.raises(PluginContractError, match="returned"):
        registry.create("wrong-id")


def test_explicit_atomic_l1_bridge_reuses_existing_scf_and_pcm_engine():
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
    )
    plugin = _SyntheticResponsePlugin()
    with pytest.raises(TypeError, match="validated descriptor"):
        resolve_route2_electronic_model(plugin)

    adapter = adapt_atomic_l1_plugin_to_legacy_engine(
        plugin,
        spec=LegacyAtomicL1BridgeSpec(
            model_family=plugin.plugin_id,
            field_evaluator=plugin.field_dual_space.name,
            profile_binding="synthetic-water-response-profile-v1",
        ),
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _ZeroReactionField(),
        cds_evaluator=lambda _atoms: SimpleNamespace(energy_hartree=0.0),
        settings=_engine_settings(),
    )

    gas_state = engine.gas_state(adapter, atoms, need_forces=False)
    coupled = engine.solve_coupled_state(
        atoms,
        adapter,
        gas_state,
        provider_cache_signature=("synthetic-plugin-bridge",),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )

    np.testing.assert_allclose(coupled.density_coefficients, 0.0)
    assert coupled.electronic_model_identity == id(plugin)
    assert coupled.response_mode == "scf"


def test_mace_plugin_is_registered_but_p0_field_and_training_audits_remain_unattested():
    descriptor = Route2ElectronicModelDescriptor(
        adapter_name="synthetic-mace-polar-adapter",
        model_family=ROUTE2_MACE_POLAR_MODEL_FAMILY,
        field_evaluator="synthetic-mace-field",
        source_space=ATOMIC_L1_SOURCE_SPACE,
        capabilities=Route2ElectronicModelCapabilities(
            state_projectors=frozenset({"local-jet"}),
            response_projectors=frozenset({"local-jet"}),
        ),
        energy_semantics=FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        profile_binding=ROUTE2_MACE_POLAR_PROFILE_BINDING,
        provenance={"checkpoint": {"sha256": "mace-checkpoint-sha"}},
    )
    calculator = SimpleNamespace(
        route2_electronic_model_descriptor=descriptor,
        atomic_numbers=[1, 8],
    )

    plugin = DEFAULT_ROUTE2_PLUGIN_REGISTRY.create("mace-polar-1", calculator)

    assert isinstance(plugin, MACEPolarPlugin)
    assert plugin.provenance.checkpoint_sha256 == "mace-checkpoint-sha"
    assert plugin.provenance.training_code_sha == "unattestable"
    assert plugin.provenance.inference_code_sha == "unattestable"
    assert plugin.provenance.optimizer_coverage == "unattestable"
    assert plugin.provenance.field_convention == "unattested"
    assert resolve_route2_electronic_model(plugin).descriptor is descriptor
    with pytest.raises(PluginContractError, match="field convention is audited"):
        admit_plugin(plugin)


def test_artifact_dispositions_are_separate_from_historical_artifact_json():
    registry = ArtifactDispositionRegistry(
        (
            ArtifactDispositionRecord(
                artifact_id="sha256:legacy-field-audit",
                artifact_path="docs/implicit-solvation/benchmarks/legacy.json",
                artifact_sha256="a" * 64,
                disposition="legacy-unresolved",
                rationale="field sign predates the P0 convention audit",
            ),
        )
    )

    assert registry.disposition_for("sha256:legacy-field-audit").disposition == (
        "legacy-unresolved"
    )
    assert registry.disposition_for("sha256:unknown") is None
    with pytest.raises(ValueError, match="duplicate"):
        ArtifactDispositionRegistry(registry.records + registry.records)


def test_tracked_p0_disposition_registry_binds_historical_bytes_without_rewriting():
    root = Path(__file__).resolve().parents[2]
    registry_path = (
        root
        / "docs/implicit-solvation/benchmarks/route2-mace-p0-artifact-dispositions-v1.json"
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = ArtifactDispositionRegistry.from_mapping(payload)

    assert len(registry.records) == 3
    for record in registry.records:
        artifact = root / record.artifact_path
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == record.artifact_sha256
    current = registry.disposition_for(MACE_P0_ARTIFACT_ID)
    assert current is not None
    assert current.disposition == "accepted-current"
    legacy = [
        record
        for record in registry.records
        if record.artifact_id != MACE_P0_ARTIFACT_ID
    ]
    assert all(record.disposition == "legacy-unresolved" for record in legacy)


def test_tracked_p0_certificate_is_hash_bound_and_response_only():
    root = Path(__file__).resolve().parents[2]
    certificate = load_mace_p0_certificate(root)

    assert certificate.artifact_id == MACE_P0_ARTIFACT_ID
    assert certificate.field_convention == "potential-gradient"
    assert certificate.audited_atomic_numbers == frozenset({1, 8})
    assert certificate.total_charge_domain_e == (0.0, 0.0)
    assert certificate.optimizer_coverage == "unattestable"
    assert set(certificate.blocked_routes) == {
        "operational-force",
        "variational",
    }

    artifact = json.loads(
        (root / certificate.artifact_path).read_text(encoding="utf-8")
    )
    assert artifact["decision"]["field_interface_certificate_passed"] is True
    assert artifact["decision"]["admitted_route"] == "response-only"
    assert artifact["gates"]["autograd_energy_dipole_conjugacy"]["passed"] is False
    assert artifact["gates"]["source_response"]["passed"] is False
