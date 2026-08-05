from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Hartree

from maple.function.calculator.extra_correction.implicit.route2_engine import (
    Route2ContinuumEngine,
    Route2EngineSettings,
    Route2SCFConvergenceError,
)
from maple.function.calculator.extra_correction.implicit.route2_field_state import (
    LocalReactionField,
)
from maple.function.calculator.extra_correction.implicit.route2_plugin import (
    ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE,
    ATOMIC_L1_PLUGIN_SOURCE_SPACE,
    P_MINUS_1_WATER_CONTINUUM_SPEC,
    P_MINUS_1_WATER_CONTINUUM_SPEC_ID,
    RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION,
    ContinuumSolveEvidence,
    EnergySourceConjugacyEvidence,
    LegacyAtomicL1BridgeSpec,
    PluginCapabilityDeclaration,
    PluginElectronicState,
    PluginProvenance,
    adapt_atomic_l1_plugin_to_legacy_engine,
    audit_plugin_energy_source_conjugacy,
    build_operational_energy_audit,
    classify_continuum_solve_evidence,
)
from maple.function.route2_energy_ledger import (
    LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
    PCM_HALF_COUPLING_ONLY_V1,
)
from maple.function.route2_model_contracts import (
    ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY,
)


def _provenance() -> PluginProvenance:
    return PluginProvenance(
        checkpoint_sha256="synthetic-checkpoint",
        training_code_sha="synthetic-training-code",
        inference_code_sha="synthetic-inference-code",
        representation=ATOMIC_L1_PLUGIN_SOURCE_SPACE.representation,
        supported_atomic_numbers=frozenset({1, 8}),
        total_charge_domain=(0.0, 0.0),
        field_convention="potential-gradient",
        potential_unit=ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.potential_unit,
        gradient_unit=ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE.gradient_unit,
        energy_semantics=ROUTE2_FIELD_CONDITIONED_OPERATIONAL_ENERGY,
        optimizer_coverage="not-applicable",
    )


class _ScheduledSourcePlugin:
    plugin_id = "synthetic-p1-source-plugin-v1"
    source_space = ATOMIC_L1_PLUGIN_SOURCE_SPACE
    field_dual_space = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE
    provenance = _provenance()
    capabilities = PluginCapabilityDeclaration(field_response=True)

    def __init__(self, gas_source: np.ndarray, responses: list[np.ndarray]):
        self.gas_source = np.asarray(gas_source, dtype=float)
        self.responses = [np.asarray(value, dtype=float) for value in responses]
        self.calls = 0

    def cached_source_state(self, atoms, *, require_forces=False):
        assert require_forces is False
        return PluginElectronicState(
            energy_ev=-2.0,
            source=self.gas_source,
            dipole_e_angstrom=np.zeros(3),
        )

    def evaluate_source_state(self, atoms, field, *, compute_forces=False):
        assert isinstance(field, LocalReactionField)
        assert compute_forces is False
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return (
            PluginElectronicState(
                energy_ev=-1.0,
                source=response,
                dipole_e_angstrom=np.zeros(3),
            ),
            {"provider": self.plugin_id},
        )

    def linearize_source_response(self, atoms, field):
        del atoms, field
        return SimpleNamespace(jvp=lambda direction: np.zeros_like(direction))


class _ZeroReactionField:
    atom_count = 3

    def __init__(self, continuum_evidence: ContinuumSolveEvidence | None = None):
        self.runtime_provenance = {
            "backend": "synthetic-ddpcm",
            "solver_tolerance": 1.0e-10,
        }
        if continuum_evidence is not None:
            self.route2_continuum_solve_evidence = continuum_evidence

    @staticmethod
    def apply_scf(source):
        return np.zeros_like(np.asarray(source, dtype=float))

    @staticmethod
    def scf_polarization_energy_hartree(source):
        del source
        return 0.0


@dataclass
class _CDS:
    energy_hartree: float = 0.02


def _settings(**updates) -> Route2EngineSettings:
    settings = Route2EngineSettings(
        continuum_label="synthetic ddPCM",
        scf_mixing=1.0,
        scf_density_tolerance=1.0e-12,
        scf_dipole_tolerance_e_angstrom=1.0e-12,
        scf_energy_tolerance_ev=1.0e-12,
        scf_max_iterations=2,
        adjoint_relative_tolerance=1.0e-10,
        adjoint_absolute_tolerance=1.0e-13,
        adjoint_max_iterations=20,
        energy_identity_tolerance_ev=1.0e-12,
        force_state_energy_tolerance_ev=1.0e-12,
        neutral_density_tolerance=1.0e-8,
        scf_raw_response_charge_tolerance_e=1.0e-12,
        scf_total_charge_residual_tolerance_e=1.0e-12,
        scf_molecular_dipole_tolerance_e_angstrom=1.0e-12,
        scf_require_two_energy_samples=True,
    )
    return replace(settings, **updates)


def _solve_zero_state(
    continuum_evidence: ContinuumSolveEvidence | None = None,
):
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]],
    )
    source = np.zeros((3, 4))
    plugin = _ScheduledSourcePlugin(source, [source, source])
    adapter = adapt_atomic_l1_plugin_to_legacy_engine(
        plugin,
        spec=LegacyAtomicL1BridgeSpec(
            model_family=plugin.plugin_id,
            field_evaluator=plugin.field_dual_space.name,
            profile_binding="synthetic-p1-water-v1",
        ),
    )
    reaction_field = _ZeroReactionField(continuum_evidence)
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: reaction_field,
        cds_evaluator=lambda _atoms: _CDS(),
        settings=_settings(),
    )
    gas_state = engine.gas_state(adapter, atoms, need_forces=False)
    coupled = engine.solve_coupled_state(
        atoms,
        adapter,
        gas_state,
        provider_cache_signature=("synthetic-p1-water",),
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
    )
    conjugacy = EnergySourceConjugacyEvidence(
        status="failed",
        method="synthetic finite-field falsification",
        finite_difference_step=1.0e-4,
        finite_difference_derivative_ev=1.0,
        source_pairing_derivative_ev=0.0,
        absolute_error_ev=1.0,
        relative_error=1.0,
        absolute_tolerance_ev=1.0e-7,
        relative_tolerance=1.0e-5,
    )
    return gas_state, coupled, conjugacy


def test_pminus1_water_continuum_profile_is_numeric_frozen_and_ho_only():
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.1], [0.0, 0.76, -0.48], [0.0, -0.76, -0.48]],
    )
    spec = P_MINUS_1_WATER_CONTINUUM_SPEC

    assert spec.identity == P_MINUS_1_WATER_CONTINUUM_SPEC_ID
    np.testing.assert_allclose(spec.radii_angstrom(atoms.numbers), [1.52, 1.2, 1.2])
    assert len(spec.cavity_sha256(atoms)) == 64
    assert spec.as_provenance()["include_cds"] is False
    settings = spec.engine_settings()
    assert settings.scf_require_two_energy_samples is True
    assert settings.scf_finite_resolution_policy is None
    assert settings.effective_raw_response_charge_tolerance_e == 2.0e-12
    assert settings.effective_molecular_dipole_tolerance_e_angstrom == 2.0e-12
    with pytest.raises(ValueError, match="excludes atomic numbers: 6"):
        spec.radii_angstrom([1, 6, 8])


def test_p1_ledger_records_every_scalar_but_tolerance_only_inner_solve_blocks():
    gas_state, coupled, conjugacy = _solve_zero_state()

    evidence = classify_continuum_solve_evidence(coupled.reaction_field)
    assert evidence.status == "tolerance-only"
    assert evidence.residual_gate_passed is False
    audit = build_operational_energy_audit(
        gas_state,
        coupled,
        electrostatic_energy_ledger=PCM_HALF_COUPLING_ONLY_V1,
        energy_source_conjugacy=conjugacy,
    )

    assert audit.p1_complete is False
    assert audit.blockers == ("inner-continuum-algebraic-residual-not-certified",)
    assert audit.energy_interpretation == (RESPONSE_CONDITIONED_OPERATIONAL_PREDICTION)
    assert audit.model_zero_field_energy_ev == -2.0
    assert audit.model_fixed_field_energy_ev == -1.0
    assert audit.model_energy_change_ev == 1.0
    assert audit.source_field_pairing_ev == 0.0
    assert audit.pcm_half_coupling_ev == 0.0
    assert audit.cds_energy_ev == pytest.approx(0.02 * Hartree)
    assert audit.final_scalar_ev == pytest.approx(0.02 * Hartree)
    assert audit.outer_scf.passed is True
    assert audit.outer_scf.metrics["energy_delta_ev"] == 0.0
    json.dumps(audit.as_dict(), allow_nan=False)


def test_residual_certified_synthetic_backend_can_complete_the_narrow_p1_gate():
    certified = ContinuumSolveEvidence(
        backend="synthetic-ddpcm",
        status="residual-certified",
        requested_tolerance=1.0e-10,
        actual_residual=2.0e-12,
        residual_definition="L2 algebraic residual of A sigma minus b",
        solve_completed_without_exception=True,
    )
    gas_state, coupled, conjugacy = _solve_zero_state(certified)

    audit = build_operational_energy_audit(
        gas_state,
        coupled,
        electrostatic_energy_ledger=LEGACY_MACE_FIELD_ENERGY_PLUS_PCM_V1,
        energy_source_conjugacy=conjugacy,
    )

    assert audit.p1_complete is True
    assert audit.blockers == ()
    assert audit.final_scalar_ev == pytest.approx(1.0 + 0.02 * Hartree)
    assert audit.energy_ledger_hartree.leaf_components_hartree[
        "solute_polarization"
    ] == pytest.approx(1.0 / Hartree)


class _LinearEnergySourcePlugin:
    plugin_id = "synthetic-linear-conjugate-plugin-v1"
    source_space = ATOMIC_L1_PLUGIN_SOURCE_SPACE
    field_dual_space = ATOMIC_L1_PLUGIN_FIELD_DUAL_SPACE
    provenance = _provenance()
    capabilities = PluginCapabilityDeclaration(field_response=True)

    def __init__(self, source: np.ndarray, *, conjugate: bool):
        self.source = np.asarray(source, dtype=float)
        self.conjugate = conjugate

    def cached_source_state(self, atoms, *, require_forces=False):
        del require_forces
        return self._state(np.zeros((len(atoms), 4)))

    def _state(self, field: np.ndarray) -> PluginElectronicState:
        energy = (
            self.field_dual_space.pair(
                self.source,
                field,
                source_space=self.source_space,
                atom_count=self.source.shape[0],
            )
            if self.conjugate
            else 0.0
        )
        return PluginElectronicState(
            energy_ev=energy,
            source=self.source,
            dipole_e_angstrom=np.zeros(3),
        )

    def evaluate_source_state(self, atoms, field, *, compute_forces=False):
        del atoms, compute_forces
        assert isinstance(field, LocalReactionField)
        return self._state(field.as_nodewise_jet()), {}

    def linearize_source_response(self, atoms, field):
        del atoms, field
        return SimpleNamespace(jvp=lambda direction: np.zeros_like(direction))


@pytest.mark.parametrize(
    ("conjugate", "expected"), [(True, "passed"), (False, "failed")]
)
def test_finite_field_energy_source_conjugacy_is_measured_not_assumed(
    conjugate,
    expected,
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    source = np.asarray([[-0.2, 0.1, 0.0, 0.0], [0.2, -0.1, 0.0, 0.0]])
    plugin = _LinearEnergySourcePlugin(source, conjugate=conjugate)
    base = LocalReactionField.from_nodewise_jet(np.zeros((2, 4)))
    direction = np.asarray([[0.3, -0.2, 0.1, 0.4], [-0.1, 0.5, -0.3, 0.2]])

    evidence = audit_plugin_energy_source_conjugacy(
        plugin,
        atoms,
        base,
        direction,
        finite_difference_step=1.0e-5,
        absolute_tolerance_ev=1.0e-9,
        relative_tolerance=1.0e-8,
    )

    assert evidence.status == expected
    assert evidence.finite_difference_derivative_ev == pytest.approx(
        evidence.source_pairing_derivative_ev if conjugate else 0.0,
        abs=1.0e-10,
    )


def test_molecular_dipole_and_raw_charge_are_independent_outer_scf_gates():
    far_atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    zero = np.zeros((2, 4))
    charge_transfer = np.asarray([[5.0e-13, 0.0, 0.0, 0.0], [-5.0e-13, 0.0, 0.0, 0.0]])
    plugin = _ScheduledSourcePlugin(zero, [charge_transfer])
    adapter = adapt_atomic_l1_plugin_to_legacy_engine(
        plugin,
        spec=LegacyAtomicL1BridgeSpec(
            model_family=plugin.plugin_id,
            field_evaluator=plugin.field_dual_space.name,
            profile_binding="synthetic-aggregate-residual-v1",
        ),
    )
    engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _ZeroReactionField(),
        cds_evaluator=lambda _atoms: _CDS(0.0),
        settings=_settings(
            scf_max_iterations=1,
            scf_require_two_energy_samples=False,
            scf_molecular_dipole_tolerance_e_angstrom=1.0e-12,
        ),
    )
    gas_state = engine.gas_state(adapter, far_atoms, need_forces=False)

    with pytest.raises(Route2SCFConvergenceError) as caught:
        engine.solve_coupled_state(
            far_atoms,
            adapter,
            gas_state,
            provider_cache_signature=("aggregate-dipole-gate",),
        )
    record = caught.value.history[-1]
    assert record["monopole_residual_e"] <= 1.0e-12
    assert record["monopole_residual_rms_e"] <= 1.0e-12
    assert record["molecular_dipole_residual_l2_e_angstrom"] == pytest.approx(5.0e-12)
    assert "molecular dipole residual=" in str(caught.value)

    raw_charge = np.asarray([[5.0e-11, 0.0, 0.0, 0.0], [5.0e-11, 0.0, 0.0, 0.0]])
    noisy_plugin = _ScheduledSourcePlugin(zero, [raw_charge])
    noisy_adapter = adapt_atomic_l1_plugin_to_legacy_engine(
        noisy_plugin,
        spec=LegacyAtomicL1BridgeSpec(
            model_family=noisy_plugin.plugin_id,
            field_evaluator=noisy_plugin.field_dual_space.name,
            profile_binding="synthetic-raw-charge-residual-v1",
        ),
    )
    charge_engine = Route2ContinuumEngine(
        reaction_field_factory=lambda _atoms: _ZeroReactionField(),
        cds_evaluator=lambda _atoms: _CDS(0.0),
        settings=_settings(
            scf_max_iterations=1,
            scf_require_two_energy_samples=False,
            scf_raw_response_charge_tolerance_e=1.0e-12,
        ),
    )
    gas_state = charge_engine.gas_state(noisy_adapter, far_atoms, need_forces=False)
    with pytest.raises(Route2SCFConvergenceError) as caught:
        charge_engine.solve_coupled_state(
            far_atoms,
            noisy_adapter,
            gas_state,
            provider_cache_signature=("raw-charge-gate",),
        )
    record = caught.value.history[-1]
    assert record["density_residual_e"] == pytest.approx(0.0)
    assert abs(record["raw_response_charge_delta_e"]) == pytest.approx(1.0e-10)
    assert record["projected_total_charge_residual_e"] == pytest.approx(0.0)
