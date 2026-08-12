from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import ast
from pathlib import Path

import numpy as np
import pytest

from maple.solvation.coupling.spaces import (
    ATOMIC_L1_FIELD_DUAL_SPACE,
    ATOMIC_L1_SOURCE_SPACE,
)
from maple.solvation.models import (
    ElectronicSourceModel,
    ElectronicSourceState,
    ElectronicResponseEquationAdapter,
    FieldEnergyState,
    FieldResponsiveModel,
    ModelCapabilityDeclaration,
    ModelDomain,
    ModelProvenance,
    SoluteModelBundle,
    VacuumEnergyModel,
    VacuumState,
    VacuumScalarEquationAdapter,
    VariationalFieldModel,
    VariationalIdentityDeclaration,
    array_sha256,
    model_input_sha256,
    validate_field_energy_evaluation,
    validate_response_linearization,
    validate_source_evaluation,
    validate_vacuum_evaluation,
    validate_variational_declaration,
)


class _Atoms:
    def __init__(self, count: int = 2, shift: float = 0.0, info=None):
        self.count = count
        self.shift = shift
        self.info = {} if info is None else dict(info)

    def __len__(self) -> int:
        return self.count

    def get_atomic_numbers(self):
        return np.ones(self.count, dtype=int)

    def get_positions(self):
        return (
            np.arange(3 * self.count, dtype=float).reshape(self.count, 3) + self.shift
        )

    def get_cell(self):
        return np.zeros((3, 3))

    def get_pbc(self):
        return np.zeros(3, dtype=bool)


DOMAIN = ModelDomain((1, 6, 8), (-1, 1), (1, 2, 3))


def _provenance(
    *,
    provider_id: str = "fake-model-v1",
    model_profile_id: str = "route2-model-v1",
    family: str = "fake-family",
    checkpoint: str = "1" * 64,
    inference: str = "2" * 64,
    audited: bool = False,
) -> ModelProvenance:
    return ModelProvenance(
        provider_id=provider_id,
        model_profile_id=model_profile_id,
        model_family=family,
        checkpoint_sha256=checkpoint,
        upstream_version="1.2.3",
        upstream_commit="abcdef0",
        inference_code_sha256=inference,
        dtype="float64",
        device="cpu",
        domain=DOMAIN,
        field_convention=ATOMIC_L1_FIELD_DUAL_SPACE.field_convention,
        coordinate_frame_policy="laboratory Cartesian Angstrom",
        optimizer_parameter_groups_audited=audited,
        optimizer_audit_evidence_sha256="3" * 64 if audited else None,
    )


class FakeVacuum:
    def __init__(self, provenance: ModelProvenance | None = None):
        self.provenance = provenance or _provenance()
        self.provider_id = self.provenance.provider_id
        self.model_profile_id = self.provenance.model_profile_id
        self.provenance_sha256 = self.provenance.sha256
        self.dtype = self.provenance.dtype
        self.device = self.provenance.device
        self.domain = self.provenance.domain
        self.coordinate_frame_policy = self.provenance.coordinate_frame_policy
        self.configuration_token = "a" * 64

    def configuration_sha256(self):
        return self.configuration_token

    def evaluate_vacuum(self, atoms, *, need_forces: bool):
        return VacuumState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            len(atoms),
            -1.25,
            need_forces,
            np.zeros((len(atoms), 3)) if need_forces else None,
        )


class FakeSource:
    source_space = ATOMIC_L1_SOURCE_SPACE
    field_space = ATOMIC_L1_FIELD_DUAL_SPACE

    def __init__(self, provenance: ModelProvenance | None = None):
        self.provenance = provenance or _provenance()
        self.provider_id = self.provenance.provider_id
        self.model_profile_id = self.provenance.model_profile_id
        self.provenance_sha256 = self.provenance.sha256
        self.dtype = self.provenance.dtype
        self.device = self.provenance.device
        self.domain = self.provenance.domain
        self.field_convention = self.provenance.field_convention
        self.coordinate_frame_policy = self.provenance.coordinate_frame_policy
        self.configuration_token = "a" * 64

    def configuration_sha256(self):
        return self.configuration_token

    def evaluate_source(self, atoms, field, *, need_fixed_field_forces: bool):
        source = self.field_space.pairing_metric.field_to_source_dual(field)
        return ElectronicSourceState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            array_sha256(field, name="field"),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            len(atoms),
            source,
            need_fixed_field_forces,
            np.zeros((len(atoms), 3)) if need_fixed_field_forces else None,
        )


class FakeResponse(FakeSource):
    def source_jvp(self, atoms, field, field_direction):
        return self.field_space.pairing_metric.field_to_source_dual(field_direction)

    def source_vjp(self, atoms, field, source_cotangent):
        return self.field_space.pairing_metric.source_to_field_dual(source_cotangent)

    def source_position_vjp(self, atoms, field, source_cotangent):
        return np.zeros((len(atoms), 3))


class FakeVariational(FakeResponse):
    def __init__(self, provenance=None, evidence_changes=None):
        super().__init__(provenance)
        self.evidence_changes = evidence_changes or {}

    def field_energy(self, atoms, field, *, need_fixed_field_forces: bool):
        return FieldEnergyState(
            self.provider_id,
            self.provenance_sha256,
            model_input_sha256(atoms),
            array_sha256(field, name="field"),
            self.source_space.metadata_hash(),
            self.field_space.metadata_hash(),
            len(atoms),
            0.5 * float(np.vdot(field, field)),
            need_fixed_field_forces,
            np.zeros((len(atoms), 3)) if need_fixed_field_forces else None,
        )

    def field_energy_source_identity_evidence(self):
        values = dict(
            provider_id=self.provider_id,
            provenance_sha256=self.provenance_sha256,
            evidence_sha256="4" * 64,
            field_energy_source_finite_difference=True,
            source_jvp_vjp_transpose=True,
            q_self_adjoint_susceptibility=True,
            passive_stable_spectrum=True,
            locally_invertible_unique_root=True,
            field_sign_gauge_origin=True,
            finite_field_domain=True,
            same_checkpoint_and_inference=True,
            no_untrained_response_submodule=True,
            full_coordinate_derivative=True,
        )
        values.update(self.evidence_changes)
        return VariationalIdentityDeclaration(**values)


def test_protocol_layers_are_runtime_structural_and_do_not_promote_vacuum_force_models():
    vacuum = FakeVacuum()
    source = FakeSource()
    response = FakeResponse()
    variational = FakeVariational(_provenance(audited=True))
    assert isinstance(vacuum, VacuumEnergyModel)
    assert not isinstance(vacuum, ElectronicSourceModel)
    assert isinstance(source, ElectronicSourceModel)
    assert not isinstance(source, FieldResponsiveModel)
    assert isinstance(response, FieldResponsiveModel)
    assert not isinstance(response, VariationalFieldModel)
    assert isinstance(variational, VariationalFieldModel)


def test_provenance_is_complete_content_addressed_and_optimizer_audit_is_bound():
    provenance = _provenance(audited=True)
    assert len(provenance.sha256) == 64
    assert provenance.metadata()["domain"] == DOMAIN.metadata()
    with pytest.raises(ValueError, match="declared together"):
        replace(provenance, optimizer_parameter_groups_audited=False)
    with pytest.raises(FrozenInstanceError):
        provenance.dtype = "float32"
    missing_metadata = FakeVacuum()
    del missing_metadata.dtype
    with pytest.raises(TypeError, match="must declare dtype"):
        validate_vacuum_evaluation(missing_metadata, _Atoms(), need_forces=False)


def test_vacuum_and_source_states_are_immutable_finite_and_provider_bound():
    atoms = _Atoms()
    vacuum = FakeVacuum()
    source = FakeSource()
    field = np.arange(8.0).reshape(2, 4)
    vacuum_state = validate_vacuum_evaluation(vacuum, atoms, need_forces=True)
    source_state = validate_source_evaluation(
        source, atoms, field, need_fixed_field_forces=True
    )
    assert vacuum_state.forces_eV_per_A.shape == (2, 3)
    assert source_state.source.shape == (2, 4)
    with pytest.raises(ValueError):
        source_state.source[0, 0] = 9.0
    bad = FakeSource()
    bad.provenance_sha256 = "f" * 64
    with pytest.raises(ValueError, match="immutable provenance"):
        validate_source_evaluation(bad, atoms, field, need_fixed_field_forces=False)


def test_states_reject_same_size_wrong_geometry_stale_field_space_and_request():
    atoms = _Atoms()
    field = np.arange(8.0).reshape(2, 4)

    class WrongGeometryVacuum(FakeVacuum):
        def evaluate_vacuum(self, atoms, *, need_forces):
            state = super().evaluate_vacuum(atoms, need_forces=need_forces)
            return replace(
                state, model_input_sha256=model_input_sha256(_Atoms(2, shift=1.0))
            )

    with pytest.raises(ValueError, match="model-input SHA256"):
        validate_vacuum_evaluation(WrongGeometryVacuum(), atoms, need_forces=False)

    class StaleSource(FakeSource):
        changed_field = "input_field_sha256"

        def evaluate_source(self, atoms, field, *, need_fixed_field_forces):
            state = super().evaluate_source(
                atoms, field, need_fixed_field_forces=need_fixed_field_forces
            )
            return replace(state, **{self.changed_field: "f" * 64})

    for attribute in (
        "input_field_sha256",
        "source_space_sha256",
        "field_space_sha256",
    ):
        model = StaleSource()
        model.changed_field = attribute
        with pytest.raises(ValueError, match=attribute):
            validate_source_evaluation(
                model, atoms, field, need_fixed_field_forces=False
            )

    class WrongRequest(FakeSource):
        def evaluate_source(self, atoms, field, *, need_fixed_field_forces):
            state = super().evaluate_source(
                atoms, field, need_fixed_field_forces=need_fixed_field_forces
            )
            return replace(state, need_fixed_field_forces=not need_fixed_field_forces)

    with pytest.raises(ValueError, match="need_fixed_field_forces"):
        validate_source_evaluation(
            WrongRequest(), atoms, field, need_fixed_field_forces=False
        )

    class StaleFieldEnergy(FakeVariational):
        def field_energy(self, atoms, field, *, need_fixed_field_forces):
            state = super().field_energy(
                atoms, field, need_fixed_field_forces=need_fixed_field_forces
            )
            return replace(state, input_field_sha256="e" * 64)

    with pytest.raises(ValueError, match="input_field_sha256"):
        validate_field_energy_evaluation(
            StaleFieldEnergy(_provenance(audited=True)),
            atoms,
            field,
            need_fixed_field_forces=False,
        )


@pytest.mark.parametrize(
    "reference_info,current_info",
    (
        ({"charge": 0, "multiplicity": 1}, {"total_charge": 1, "spin": 1}),
        ({"charge": 0, "multiplicity": 1}, {"charge": 0, "spin": 1.0}),
    ),
)
def test_model_state_cache_identity_rejects_same_geometry_different_charge_or_spin(
    reference_info, current_info
):
    reference = _Atoms(info=reference_info)
    current = _Atoms(info=current_info)
    assert np.array_equal(reference.get_positions(), current.get_positions())
    assert model_input_sha256(reference) != model_input_sha256(current)

    class StaleVacuum(FakeVacuum):
        def evaluate_vacuum(self, atoms, *, need_forces):
            state = super().evaluate_vacuum(atoms, need_forces=need_forces)
            return replace(state, model_input_sha256=model_input_sha256(reference))

    with pytest.raises(ValueError, match="model-input SHA256"):
        validate_vacuum_evaluation(StaleVacuum(), current, need_forces=False)

    class StaleSource(FakeSource):
        def evaluate_source(self, atoms, field, *, need_fixed_field_forces):
            state = super().evaluate_source(
                atoms, field, need_fixed_field_forces=need_fixed_field_forces
            )
            return replace(state, model_input_sha256=model_input_sha256(reference))

    with pytest.raises(ValueError, match="model-input SHA256"):
        validate_source_evaluation(
            StaleSource(),
            current,
            np.zeros((2, 4)),
            need_fixed_field_forces=False,
        )


def test_model_input_alias_policy_is_explicit_and_conflicts_fail_closed():
    assert model_input_sha256(_Atoms()) == model_input_sha256(
        _Atoms(info={"charge": 0, "multiplicity": 1})
    )
    assert model_input_sha256(_Atoms(info={"charge": 1})) == model_input_sha256(
        _Atoms(info={"total_charge": 1})
    )
    assert model_input_sha256(_Atoms(info={"mult": 3})) == model_input_sha256(
        _Atoms(info={"spin": 1.0, "multiplicity": 3})
    )
    with pytest.raises(ValueError, match="conflicting atoms.info aliases"):
        model_input_sha256(_Atoms(info={"charge": 0, "total_charge": 1}))
    with pytest.raises(ValueError, match="conflicting atoms.info multiplicity"):
        model_input_sha256(_Atoms(info={"spin": 0.0, "multiplicity": 3}))
    with pytest.raises(ValueError, match="conflicting atoms.info aliases"):
        model_input_sha256(_Atoms(info={"mult": 1, "multiplicity": 3}))
    with pytest.raises(ValueError, match="spin.*non-negative"):
        model_input_sha256(_Atoms(info={"spin": -0.5}))
    with pytest.raises(ValueError, match="half-integer"):
        model_input_sha256(_Atoms(info={"spin": 0.25}))


def test_model_input_accepts_maple_reader_spin_quantum_number_contract():
    singlet_from_reader = _Atoms(info={"charge": 0, "mult": 1, "spin": 0.0})
    triplet_from_reader = _Atoms(info={"charge": 0, "mult": 3, "spin": 1.0})
    assert model_input_sha256(singlet_from_reader) == model_input_sha256(
        _Atoms(info={"charge": 0, "multiplicity": 1})
    )
    assert model_input_sha256(triplet_from_reader) == model_input_sha256(
        _Atoms(info={"charge": 0, "multiplicity": 3})
    )
    with pytest.raises(ValueError, match=r"multiplicity must equal 2\*spin\+1"):
        model_input_sha256(_Atoms(info={"mult": 3, "spin": 0.0}))


def test_every_model_entry_enforces_atomic_charge_and_spin_domain():
    class UnsupportedElementAtoms(_Atoms):
        def get_atomic_numbers(self):
            return np.asarray([1, 7], dtype=int)

    vacuum = FakeVacuum()
    response = FakeResponse()
    field = np.zeros((2, 4))
    with pytest.raises(ValueError, match="atomic numbers outside model domain"):
        validate_vacuum_evaluation(vacuum, UnsupportedElementAtoms(), need_forces=False)
    with pytest.raises(ValueError, match="total charge 2 is outside model domain"):
        validate_source_evaluation(
            response,
            _Atoms(info={"charge": 2}),
            field,
            need_fixed_field_forces=False,
        )
    with pytest.raises(ValueError, match="spin multiplicity 5 is outside model domain"):
        validate_response_linearization(
            response,
            _Atoms(info={"mult": 5}),
            field,
            field_direction=np.ones((2, 4)),
            source_cotangent=np.ones((2, 4)),
        )


def test_requested_forces_shape_and_finiteness_fail_closed():
    class MissingForces(FakeVacuum):
        def evaluate_vacuum(self, atoms, *, need_forces):
            return VacuumState(
                self.provider_id,
                self.provenance_sha256,
                model_input_sha256(atoms),
                len(atoms),
                0.0,
                need_forces,
            )

    with pytest.raises(ValueError, match="requested vacuum forces"):
        validate_vacuum_evaluation(MissingForces(), _Atoms(), need_forces=True)
    with pytest.raises(ValueError, match="finite with shape"):
        VacuumState("x", "1" * 64, "2" * 64, 2, 0.0, True, np.zeros((1, 3)))
    with pytest.raises(ValueError, match="finite"):
        ElectronicSourceState(
            "x",
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            2,
            np.full((2, 4), np.nan),
            False,
        )


@pytest.mark.parametrize("bad_energy", (True, 1, np.float64(1.0), np.array(1.0)))
def test_energy_is_strict_python_float_without_scalar_aliases(bad_energy):
    with pytest.raises(TypeError, match="Python float"):
        VacuumState("x", "1" * 64, "2" * 64, 1, bad_energy, False)
    with pytest.raises(TypeError, match="Python float"):
        FieldEnergyState(
            "x",
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            1,
            bad_energy,
            False,
        )


def test_response_jvp_vjp_position_vjp_contract_and_transpose_identity():
    model = FakeResponse()
    atoms = _Atoms()
    field = np.arange(8.0).reshape(2, 4) / 10.0
    direction = np.arange(8.0, 16.0).reshape(2, 4) / 13.0
    cotangent = np.arange(-4.0, 4.0).reshape(2, 4) / 7.0
    jvp, vjp, position_vjp = validate_response_linearization(
        model,
        atoms,
        field,
        field_direction=direction,
        source_cotangent=cotangent,
    )
    assert jvp.shape == vjp.shape == (2, 4)
    assert position_vjp.shape == (2, 3)

    class BadAdjoint(FakeResponse):
        def source_vjp(self, atoms, field, source_cotangent):
            return np.zeros_like(source_cotangent)

    with pytest.raises(ValueError, match="transpose contract failed"):
        validate_response_linearization(
            BadAdjoint(),
            atoms,
            field,
            field_direction=direction,
            source_cotangent=cotangent,
        )


def test_bundle_capabilities_default_false_and_runtime_methods_must_be_callable():
    vacuum = FakeVacuum()
    vacuum_only = SoluteModelBundle(vacuum, None, "route2-model-v1")
    assert vacuum_only.mutual_polarization_capable is False
    assert vacuum_only.field_response_linearization_capable is False
    assert vacuum_only.variational_functional_admitted is False
    with pytest.raises(TypeError, match="evaluate_source must be callable"):
        SoluteModelBundle(vacuum, vacuum, "route2-model-v1")

    broken_vacuum = FakeVacuum()
    broken_vacuum.evaluate_vacuum = None
    with pytest.raises(TypeError, match="evaluate_vacuum must be callable"):
        SoluteModelBundle(broken_vacuum, None, "route2-model-v1")

    noncallable = FakeResponse()
    noncallable.source_vjp = None
    with pytest.raises(TypeError, match="source_vjp must be callable"):
        SoluteModelBundle(
            vacuum,
            noncallable,
            "route2-model-v1",
            ModelCapabilityDeclaration(True, True),
        )

    declared = SoluteModelBundle(
        vacuum,
        FakeResponse(),
        "route2-model-v1",
        ModelCapabilityDeclaration(True, True),
    )
    assert declared.mutual_polarization_capable
    assert declared.field_response_linearization_capable


def test_mixed_bundles_are_closed_without_registered_profile_and_release_artifact():
    vacuum = FakeVacuum()

    mixed_response = FakeResponse(
        _provenance(
            provider_id="other-response-v1", family="other-family", checkpoint="5" * 64
        )
    )
    with pytest.raises(ValueError, match="fail-closed"):
        SoluteModelBundle(vacuum, mixed_response, "route2-model-v1")
    with pytest.raises(ValueError, match="fail-closed"):
        SoluteModelBundle(
            vacuum,
            mixed_response,
            "route2-model-v1",
            composite_validation_evidence_sha256="6" * 64,
        )


@pytest.mark.parametrize(
    "change",
    (
        {"dtype": "float32"},
        {"device": "cuda:0"},
        {"domain": ModelDomain((1, 6), (-1, 1), (1, 2, 3))},
        {"upstream_commit": "different-upstream-commit"},
    ),
)
def test_bundle_rejects_same_checkpoint_with_incompatible_runtime_or_domain(change):
    response_provenance = replace(_provenance(), **change)
    with pytest.raises(ValueError, match="mismatched fields"):
        SoluteModelBundle(
            FakeVacuum(), FakeResponse(response_provenance), "route2-model-v1"
        )


def test_bundle_rejects_response_space_or_field_convention_mismatch():
    model = FakeResponse()
    model.field_convention = "model-native negative field"
    with pytest.raises(ValueError, match="immutable provenance"):
        SoluteModelBundle(FakeVacuum(), model, "route2-model-v1")


def test_bundle_requires_vacuum_and_response_model_profile_identity():
    wrong_vacuum = FakeVacuum(_provenance(model_profile_id="route2-other-v1"))
    with pytest.raises(ValueError, match="vacuum model_profile_id"):
        SoluteModelBundle(wrong_vacuum, None, "route2-model-v1")

    wrong_response = FakeResponse(_provenance(model_profile_id="route2-other-v1"))
    with pytest.raises(ValueError, match="response model_profile_id"):
        SoluteModelBundle(FakeVacuum(), wrong_response, "route2-model-v1")


def test_generic_model_to_equation_adapters_close_signature_and_shape_boundary():
    atoms = _Atoms()
    model = FakeResponse()
    response = ElectronicResponseEquationAdapter(
        model,
        "maple.route2.coupling.synthetic-test.v1",
    )
    vacuum = VacuumScalarEquationAdapter(model=FakeVacuum(model.provenance))
    field = np.arange(8.0).reshape(2, 4) / 10.0
    direction = np.arange(8.0, 16.0).reshape(2, 4) / 13.0
    cotangent = np.arange(-4.0, 4.0).reshape(2, 4) / 7.0
    np.testing.assert_allclose(
        response.evaluate_source(atoms, field),
        model.field_space.pairing_metric.field_to_source_dual(field),
    )
    jvp = response.field_jvp(atoms, field, direction)
    vjp = response.field_vjp(atoms, field, cotangent)
    assert float(np.vdot(jvp, cotangent)) == pytest.approx(
        float(np.vdot(direction, vjp)), abs=1e-14
    )
    assert response.coordinate_vjp(atoms, field, cotangent).shape == (6,)
    assert vacuum.evaluate_energy(atoms) == pytest.approx(-1.25)
    np.testing.assert_array_equal(vacuum.coordinate_gradient(atoms), np.zeros(6))
    assert len(response.configuration_sha256()) == 64
    assert len(vacuum.configuration_sha256()) == 64

    model.configuration_token = "b" * 64
    with pytest.raises(ValueError, match="configuration drifted"):
        response.configuration_sha256()


def test_variational_tier_defaults_false_and_requires_every_gate_and_optimizer_audit():
    ordinary = SoluteModelBundle(FakeVacuum(), FakeResponse(), "route2-model-v1")
    assert ordinary.variational_functional_admitted is False

    unaudited = FakeVariational()
    with pytest.raises(ValueError, match="optimizer parameter-group"):
        validate_variational_declaration(unaudited)
    assert (
        SoluteModelBundle(
            FakeVacuum(), unaudited, "route2-model-v1"
        ).variational_functional_admitted
        is False
    )

    incomplete = FakeVariational(
        _provenance(audited=True), {"finite_field_domain": False}
    )
    with pytest.raises(ValueError, match="finite_field_domain"):
        validate_variational_declaration(incomplete)

    declared = FakeVariational(_provenance(audited=True))
    energy_state = validate_field_energy_evaluation(
        declared, _Atoms(), np.zeros((2, 4)), need_fixed_field_forces=True
    )
    assert energy_state.fixed_field_forces_eV_per_A.shape == (2, 3)
    assert validate_variational_declaration(declared).complete
    bundle = SoluteModelBundle(
        FakeVacuum(declared.provenance), declared, "route2-model-v1"
    )
    assert bundle.variational_functional_admitted is False
    with pytest.raises(ValueError, match="fail-closed"):
        ModelCapabilityDeclaration(True, True, True)


def test_model_contract_module_does_not_import_optional_model_runtimes():
    root = Path(__file__).parents[2] / "maple" / "solvation" / "models"
    imported_roots: set[str] = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
    assert not ({"ase", "torch", "mace", "aimnet", "pyscf"} & imported_roots)
