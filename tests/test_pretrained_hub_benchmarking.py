from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import MappingProxyType

import pytest

import maple.function.benchmarking.pretrained_hub as pretrained_hub_module
from maple.function.benchmarking import (
    AccuracyAdapterArtifactReceipt,
    AccuracyAdapterRegistration,
    BenchmarkExperimentalReference,
    BenchmarkIdentity,
    BenchmarkRecordInput,
    BenchmarkResultStore,
    BenchmarkQuantity,
    END_TO_END_RUNTIME_SCOPE_COMPONENTS,
    FORMAL_FUNCTIONAL_GROUP_TAXONOMY_FINGERPRINT,
    FORMAL_FUNCTIONAL_GROUP_TAXONOMY_ID,
    FunctionalGroupAssignment,
    FunctionalGroupTaxonomyManifest,
    LeakageStatus,
    MINIMUM_ACCURACY_FUNCTIONAL_GROUPS,
    MINIMUM_ACCURACY_RECORDS,
    MolecularInputReceipt,
    ModelTrainingEvidence,
    SolventComposition,
    ValidationLedger,
    ValidationPanel,
    ValidationStage,
    audit_training_overlap,
    derive_functional_group_assignment,
    evaluate_matched_qm_performance,
    load_functional_group_taxonomy,
    paired_comparison,
    run_accuracy_panel,
    summarize_runtime_smoke,
    summarize_predictions,
)

ACCURACY_ADAPTER_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ACCURACY_ADAPTER_FIXTURE_DIR))
import pretrained_hub_accuracy_adapter_fixture as accuracy_adapter_fixture

_LabelStealingAccuracyAdapter = accuracy_adapter_fixture.LabelStealingAccuracyAdapter
_FilesystemScanningAccuracyAdapter = (
    accuracy_adapter_fixture.FilesystemScanningAccuracyAdapter
)
_HelperDelegatingAccuracyAdapter = (
    accuracy_adapter_fixture.HelperDelegatingAccuracyAdapter
)
_MissingPrecisionAccuracyAdapter = (
    accuracy_adapter_fixture.MissingPrecisionAccuracyAdapter
)
_StackInspectingAccuracyAdapter = (
    accuracy_adapter_fixture.StackInspectingAccuracyAdapter
)
_UnitTestAccuracyAdapter = accuracy_adapter_fixture.UnitTestAccuracyAdapter

ROOT = Path(__file__).resolve().parents[1]
ACCURACY_INPUTS = ROOT / "tests/data/pretrained_hub_accuracy_inputs"
FORMAL_TAXONOMY_PATH = (
    ROOT / "docs/pretrained-solvation-hub/functional-group-taxonomy-v1.json"
)
FORMAL_TAXONOMY_SHA256 = (
    "46e073d0e253d7534b7f516f1cc2096ba0abfea84ddc170c95fc86634c917b36"
)
FUNCTIONAL_GROUP_EXAMPLES = (
    ("alcohol", "CCO"),
    ("aldehyde", "CC=O"),
    ("amide", "CC(=O)N"),
    ("amine", "CCN"),
    ("carboxylic_acid", "CC(=O)O"),
    ("ester", "CC(=O)OC"),
    ("ether", "COC"),
    ("ketone", "CC(=O)C"),
    ("nitrile", "CC#N"),
    ("nitro", "C[N+](=O)[O-]"),
)
EXPERIMENTAL_REFERENCE_PATH = ACCURACY_INPUTS / "experimental-references.json"
EXPERIMENTAL_REFERENCE_OFFSET_PATH = (
    ACCURACY_INPUTS / "experimental-references-offset1.json"
)
EXPERIMENTAL_REFERENCE_SHA256 = hashlib.sha256(
    EXPERIMENTAL_REFERENCE_PATH.read_bytes()
).hexdigest()
UNIT_TEST_CHECKPOINT_PATH = ACCURACY_INPUTS / "unit-test-adapter-checkpoint.json"
ACCURACY_ADAPTER_FIXTURE_PATH = Path(accuracy_adapter_fixture.__file__).resolve()


def _adapter_id(adapter):
    if isinstance(adapter, _LabelStealingAccuracyAdapter):
        return "unit-test-label-stealing-adapter"
    if isinstance(adapter, _FilesystemScanningAccuracyAdapter):
        return "unit-test-filesystem-scanning-adapter"
    if isinstance(adapter, _StackInspectingAccuracyAdapter):
        return "unit-test-stack-inspecting-adapter"
    if isinstance(adapter, _HelperDelegatingAccuracyAdapter):
        return "unit-test-helper-delegating-adapter"
    if adapter.nonfinite_record is not None:
        return f"unit-test-registered-nonfinite-{adapter.nonfinite_record}"
    return f"unit-test-registered-offset-{adapter.prediction_offset}"


def _adapter_registration(adapter):
    adapter_id = _adapter_id(adapter)
    return AccuracyAdapterRegistration.from_components(
        adapter_id=adapter_id,
        adapter=adapter,
        implementation_artifacts=(
            AccuracyAdapterArtifactReceipt.from_file(
                role="adapter_code",
                path=ACCURACY_ADAPTER_FIXTURE_PATH,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="featurizer_code",
                path=Path(pretrained_hub_module.__file__),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="checkpoint",
                path=UNIT_TEST_CHECKPOINT_PATH,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="dependency_lock",
                path=ROOT / "pyproject.toml",
            ),
        ),
        configuration={
            "prediction_offset": getattr(adapter, "prediction_offset", None),
            "nonfinite_record": getattr(adapter, "nonfinite_record", None),
            "experimental_reference_path": getattr(
                adapter,
                "experimental_reference_path",
                None,
            ),
        },
    )


def test_concrete_accuracy_adapter_must_explicitly_declare_precision_policy():
    with pytest.raises(
        ValueError,
        match="explicitly declare.*accuracy_precision_policy",
    ):
        AccuracyAdapterRegistration.from_components(
            adapter_id="unit-test-missing-precision",
            adapter=_MissingPrecisionAccuracyAdapter(),
            implementation_artifacts=(
                AccuracyAdapterArtifactReceipt.from_file(
                    role="adapter_code",
                    path=ACCURACY_ADAPTER_FIXTURE_PATH,
                ),
                AccuracyAdapterArtifactReceipt.from_file(
                    role="featurizer_code",
                    path=Path(pretrained_hub_module.__file__),
                ),
                AccuracyAdapterArtifactReceipt.from_file(
                    role="checkpoint",
                    path=UNIT_TEST_CHECKPOINT_PATH,
                ),
                AccuracyAdapterArtifactReceipt.from_file(
                    role="dependency_lock",
                    path=ROOT / "pyproject.toml",
                ),
            ),
            configuration={},
        )


UNIT_TEST_REGISTRATIONS = {
    adapter.prediction_offset: _adapter_registration(adapter)
    for adapter in (
        _UnitTestAccuracyAdapter(0.5),
        _UnitTestAccuracyAdapter(1.0),
    )
}
NONFINITE_TEST_REGISTRATION = _adapter_registration(
    _UnitTestAccuracyAdapter(
        1.0,
        nonfinite_record="record_1",
    )
)
LABEL_STEALING_TEST_REGISTRATION = _adapter_registration(
    _LabelStealingAccuracyAdapter()
)
FILESYSTEM_SCANNING_TEST_REGISTRATION = _adapter_registration(
    _FilesystemScanningAccuracyAdapter(str(EXPERIMENTAL_REFERENCE_PATH))
)
STACK_INSPECTING_TEST_REGISTRATION = _adapter_registration(
    _StackInspectingAccuracyAdapter()
)
HELPER_DELEGATING_TEST_REGISTRATION = _adapter_registration(
    _HelperDelegatingAccuracyAdapter()
)
pretrained_hub_module._FORMAL_ACCURACY_ADAPTERS = MappingProxyType(
    {
        registration.adapter_id: registration
        for registration in (
            *UNIT_TEST_REGISTRATIONS.values(),
            NONFINITE_TEST_REGISTRATION,
            LABEL_STEALING_TEST_REGISTRATION,
            FILESYSTEM_SCANNING_TEST_REGISTRATION,
            STACK_INSPECTING_TEST_REGISTRATION,
            HELPER_DELEGATING_TEST_REGISTRATION,
        )
    }
)


def _taxonomy():
    return load_functional_group_taxonomy(
        FORMAL_TAXONOMY_PATH,
        expected_sha256=FORMAL_TAXONOMY_SHA256,
    )


def _accuracy_record_input(index, smiles):
    record_id = f"record_{index}"
    source_by_smiles = {
        value: ACCURACY_INPUTS / f"record_{source_index}.smi"
        for source_index, (_group, value) in enumerate(FUNCTIONAL_GROUP_EXAMPLES)
    }
    source_by_smiles.update(
        {
            "C": ACCURACY_INPUTS / "methane.smi",
            "CC": ACCURACY_INPUTS / "ethane.smi",
            "CO": ACCURACY_INPUTS / "methanol.smi",
        }
    )
    receipt = MolecularInputReceipt.from_file(
        source_by_smiles[smiles],
        molecular_input_format="smiles",
    )
    return BenchmarkRecordInput.from_receipt(
        record_id=record_id,
        receipt=receipt,
    )


def _identity(**overrides):
    values = {
        "dataset": "FreeSolv",
        "dataset_version": "0.52",
        "record_ids": ("mobley_1", "mobley_2"),
        "temperature_kelvin": 298.15,
        "standard_state": "1M",
        "protonation_policy": "dataset",
        "tautomer_policy": "dataset",
        "conformer_policy": "dataset-3d",
        "geometry_protocol": "fixed-input",
        "solvent_protocol": "water",
        "potential": "example-pes-v1",
        "solvation_backend": "example-solvent-v1",
        "cavity_model": "provider-native",
        "sampling_protocol": "NVT molecular-dynamics trajectory, 40 retained snapshots",
        "estimator": "BAR",
        "experimental_provenance": "FreeSolv database value",
        "target_quantity": BenchmarkQuantity.ABSOLUTE_SOLVATION_FREE_ENERGY,
    }
    values.update(overrides)
    return BenchmarkIdentity(**values)


def _accuracy_references(path=EXPERIMENTAL_REFERENCE_PATH):
    return tuple(
        BenchmarkExperimentalReference.from_json_file(
            path,
            record_id=f"record_{index}",
            source_record_locator=f"$.record_{index}",
        )
        for index in range(10)
    )


def _records(predictions=(0.0, 2.0)):
    return [
        {
            "record_id": "mobley_1",
            "experimental_kcal_mol": 1.0,
            "predicted_kcal_mol": predictions[0],
        },
        {
            "record_id": "mobley_2",
            "experimental_kcal_mol": 3.0,
            "predicted_kcal_mol": predictions[1],
        },
    ]


def _accuracy_identity(**overrides):
    record_ids = tuple(f"record_{index}" for index in range(10))
    taxonomy = _taxonomy()
    registration = UNIT_TEST_REGISTRATIONS[1.0]
    values = {
        "record_ids": record_ids,
        "record_inputs": tuple(
            _accuracy_record_input(index, FUNCTIONAL_GROUP_EXAMPLES[index][1])
            for index, _record_id in enumerate(record_ids)
        ),
        "record_experimental_references": _accuracy_references(),
        "accuracy_adapter_id": registration.adapter_id,
        "accuracy_adapter_fingerprint": registration.compute_fingerprint(),
        "functional_group_taxonomy": taxonomy,
        "record_functional_groups": tuple(
            derive_functional_group_assignment(
                record_id=record_id,
                structure_identifier=f"SMILES:{FUNCTIONAL_GROUP_EXAMPLES[index][1]}",
                taxonomy=taxonomy,
                assignment_evidence=(
                    "predeclared unit-test SMARTS match before predictions"
                ),
            )
            for index, record_id in enumerate(record_ids)
        ),
    }
    values.update(overrides)
    return _identity(**values)


def _accuracy_run(
    *,
    identity=None,
    prediction_offset=1.0,
    bootstrap_samples=0,
    seed=0,
):
    if identity is None:
        registration = UNIT_TEST_REGISTRATIONS[prediction_offset]
        identity = _accuracy_identity(
            accuracy_adapter_id=registration.adapter_id,
            accuracy_adapter_fingerprint=registration.compute_fingerprint(),
        )
    return run_accuracy_panel(
        identity,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _accuracy_mixture_compositions(water_fraction):
    return tuple(
        SolventComposition(
            f"record_{index}",
            ("water", "methanol"),
            (water_fraction, 1.0 - water_fraction),
        )
        for index in range(10)
    )


def test_identity_is_stable_and_changes_with_any_protocol_dimension():
    identity = _identity()
    assert len(identity.fingerprint) == 64
    assert identity.fingerprint == _identity().fingerprint
    assert identity.fingerprint != _identity(cavity_model="different").fingerprint
    assert identity.fingerprint != _identity(potential="alternative").fingerprint
    assert (
        identity.panel_fingerprint
        == _identity(potential="alternative").panel_fingerprint
    )
    assert (
        identity.panel_fingerprint
        != _identity(solvent_protocol="methanol").panel_fingerprint
    )
    assert (
        identity.matched_task_fingerprint
        == _identity(
            potential="alternative",
            solvation_backend="alternative-backend",
            cavity_model="alternative-cavity",
        ).matched_task_fingerprint
    )
    assert (
        identity.matched_task_fingerprint
        != _identity(
            sampling_protocol=(
                "NVT molecular-dynamics trajectory, 20 retained snapshots"
            ),
        ).matched_task_fingerprint
    )
    assert (
        identity.matched_task_fingerprint
        != _identity(estimator="thermodynamic-integration").matched_task_fingerprint
    )


def test_identity_rejects_duplicate_or_implicit_records():
    with pytest.raises(ValueError, match="duplicate"):
        _identity(record_ids=("same", "same"))
    with pytest.raises(ValueError, match="non-empty"):
        _identity(estimator="")


def test_absolute_solvation_identity_rejects_single_point_energy_differences():
    with pytest.raises(ValueError, match="thermodynamic estimator"):
        _identity(estimator="energy-difference")
    with pytest.raises(ValueError, match="sampling protocol"):
        _identity(sampling_protocol="single-point")
    with pytest.raises(ValueError, match="sampling protocol"):
        _identity(sampling_protocol="fixed-input geometry")
    with pytest.raises(ValueError, match="standard state"):
        _identity(standard_state="solution")

    mechanics = _identity(
        target_quantity=BenchmarkQuantity.GEOMETRY_LEVEL_SOLUTION_PMF,
        sampling_protocol="single-point",
        estimator="energy-difference",
    )
    assert mechanics.target_quantity is BenchmarkQuantity.GEOMETRY_LEVEL_SOLUTION_PMF


def test_record_solvent_compositions_are_normalized_and_order_invariant():
    left = _identity(
        solvent_protocol="binary liquid mixture; record-resolved components and mole fractions",
        record_solvent_compositions=(
            SolventComposition("mobley_1", ("water", "methanol"), (0.25, 0.75)),
            SolventComposition("mobley_2", ("water", "methanol"), (0.5, 0.5)),
        ),
    )
    right = _identity(
        solvent_protocol="binary liquid mixture; record-resolved components and mole fractions",
        record_solvent_compositions=(
            SolventComposition("mobley_1", ("methanol", "water"), (0.75, 0.25)),
            SolventComposition("mobley_2", ("methanol", "water"), (0.5, 0.5)),
        ),
    )

    assert left.fingerprint == right.fingerprint
    assert left.panel_fingerprint == right.panel_fingerprint
    assert left.record_solvent_compositions[0].components == ("methanol", "water")
    assert left.record_solvent_compositions[0].mole_fractions == (0.75, 0.25)


def test_record_solvent_compositions_reject_nonphysical_or_incomplete_identity():
    with pytest.raises(ValueError, match="sum to one"):
        SolventComposition("mobley_1", ("water", "methanol"), (0.2, 0.7))
    with pytest.raises(ValueError, match="strictly positive"):
        SolventComposition("mobley_1", ("water", "methanol"), (0.0, 1.0))
    with pytest.raises(ValueError, match="duplicate"):
        SolventComposition("mobley_1", ("water", "water"), (0.5, 0.5))

    with pytest.raises(ValueError, match="exactly one composition"):
        _identity(
            record_solvent_compositions=(
                SolventComposition("mobley_1", ("water",), (1.0,)),
            )
        )


def test_accuracy_identity_binds_record_level_functional_groups():
    identity = _accuracy_identity()

    assert MINIMUM_ACCURACY_FUNCTIONAL_GROUPS == 10
    assert MINIMUM_ACCURACY_RECORDS == 10
    assert identity.functional_group_taxonomy.taxonomy_id == (
        FORMAL_FUNCTIONAL_GROUP_TAXONOMY_ID
    )
    assert (
        identity.functional_group_taxonomy.fingerprint
        == FORMAL_FUNCTIONAL_GROUP_TAXONOMY_FINGERPRINT
    )
    assert len(identity.record_functional_groups) == 10
    assert len(identity.record_inputs) == 10
    assert identity.record_inputs[0].solute_structure_identifier == "SMILES:CCO"
    assert identity.record_functional_groups[0].primary_group == "alcohol"
    assert identity.record_functional_groups[0].groups == ("alcohol",)

    changed_inputs = list(identity.record_inputs)
    changed_inputs[0] = BenchmarkRecordInput.from_receipt(
        record_id=changed_inputs[0].record_id,
        receipt=MolecularInputReceipt.from_file(
            ACCURACY_INPUTS / "record_0_alt.smi",
            molecular_input_format="smiles",
        ),
    )
    changed_identity = _accuracy_identity(record_inputs=tuple(changed_inputs))
    assert changed_identity.fingerprint != identity.fingerprint
    assert changed_identity.panel_fingerprint != identity.panel_fingerprint

    with pytest.raises(ValueError, match="exactly one functional-group"):
        _identity(
            functional_group_taxonomy=_taxonomy(),
            record_functional_groups=(
                FunctionalGroupAssignment(
                    record_id="mobley_1",
                    structure_identifier="SMILES:CO",
                    primary_group="alcohol",
                    matched_groups=("alcohol",),
                    assignment_evidence="predeclared unit-test SMARTS match",
                ),
            ),
        )


def test_accuracy_identity_requires_hash_bound_record_inputs():
    identity = _accuracy_identity()

    with pytest.raises(ValueError, match="exact, hash-bound"):
        _accuracy_identity(
            record_inputs=(),
            record_functional_groups=identity.record_functional_groups,
        )
    with pytest.raises(ValueError, match="frozen BenchmarkExperimentalReference"):
        _accuracy_identity(record_experimental_references=())
    with pytest.raises(ValueError, match="registered accuracy adapter"):
        _accuracy_identity(
            accuracy_adapter_id=None,
            accuracy_adapter_fingerprint=None,
        )


def test_accuracy_identity_rejects_unfrozen_or_unknown_group_labels():
    assignment = FunctionalGroupAssignment(
        record_id="record_0",
        structure_identifier="SMILES:CC",
        primary_group="alkane",
        matched_groups=("alkane",),
        assignment_evidence="predeclared chemistry-class label",
    )
    with pytest.raises(ValueError, match="frozen FunctionalGroupTaxonomyManifest"):
        _identity(
            functional_group_taxonomy="ten chemistry classes",
            record_functional_groups=(assignment,),
        )
    with pytest.raises(ValueError, match="absent from the frozen taxonomy"):
        _identity(
            record_ids=("record_0",),
            record_inputs=(
                BenchmarkRecordInput.from_receipt(
                    record_id="record_0",
                    receipt=MolecularInputReceipt.from_file(
                        ACCURACY_INPUTS / "ethane.smi",
                        molecular_input_format="smiles",
                    ),
                ),
            ),
            record_experimental_references=_accuracy_references()[:1],
            accuracy_adapter_id=UNIT_TEST_REGISTRATIONS[1.0].adapter_id,
            accuracy_adapter_fingerprint=(
                UNIT_TEST_REGISTRATIONS[1.0].compute_fingerprint()
            ),
            functional_group_taxonomy=_taxonomy(),
            record_functional_groups=(assignment,),
        )

    official = _taxonomy()
    custom = FunctionalGroupTaxonomyManifest(
        taxonomy_id="caller-defined-taxonomy",
        version=official.version,
        source_citation=official.source_citation,
        assignment_policy=official.assignment_policy,
        definitions=official.definitions,
        primary_precedence=official.primary_precedence,
    )
    alcohol = derive_functional_group_assignment(
        record_id="record_0",
        structure_identifier="SMILES:CO",
        taxonomy=official,
        assignment_evidence="official SMARTS match",
    )
    with pytest.raises(ValueError, match="only the frozen MAPLE"):
        _identity(
            record_ids=("record_0",),
            record_inputs=(
                BenchmarkRecordInput.from_receipt(
                    record_id="record_0",
                    receipt=MolecularInputReceipt.from_file(
                        ACCURACY_INPUTS / "methanol.smi",
                        molecular_input_format="smiles",
                    ),
                ),
            ),
            record_experimental_references=_accuracy_references()[:1],
            accuracy_adapter_id=UNIT_TEST_REGISTRATIONS[1.0].adapter_id,
            accuracy_adapter_fingerprint=(
                UNIT_TEST_REGISTRATIONS[1.0].compute_fingerprint()
            ),
            functional_group_taxonomy=custom,
            record_functional_groups=(alcohol,),
        )


def test_taxonomy_loader_requires_the_exact_artifact_hash(tmp_path):
    payload = _taxonomy().canonical_payload()
    path = tmp_path / "functional-groups.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_functional_group_taxonomy(
        path,
        expected_sha256=expected_sha256,
    )
    assert loaded.fingerprint == _taxonomy().fingerprint
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_functional_group_taxonomy(path, expected_sha256="0" * 64)


def test_paired_comparison_rejects_different_mixture_compositions():
    common = {
        "solvent_protocol": "binary liquid mixture; record-resolved components and mole fractions",
    }
    left = _accuracy_identity(
        **common,
        record_solvent_compositions=_accuracy_mixture_compositions(0.25),
    )
    right = _accuracy_identity(
        **common,
        potential="alternative",
        record_solvent_compositions=_accuracy_mixture_compositions(0.5),
    )

    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            left,
            right,
        )


def test_leakage_audit_is_conservative():
    identity = _identity()
    overlap = audit_training_overlap(
        identity,
        ModelTrainingEvidence(
            known_training_record_ids=("mobley_2",),
            evidence_source="training manifest",
        ),
    )
    assert overlap.status is LeakageStatus.KNOWN_OVERLAP
    assert overlap.overlapping_record_ids == ("mobley_2",)

    unknown = audit_training_overlap(identity, ModelTrainingEvidence())
    assert unknown.status is LeakageStatus.OVERLAP_UNKNOWN

    holdout = audit_training_overlap(
        identity,
        ModelTrainingEvidence(
            explicitly_excluded_datasets=("FreeSolv",),
            record_accounting_complete=True,
            evidence_source="signed split manifest",
        ),
    )
    assert holdout.status is LeakageStatus.STRICT_HOLDOUT


def test_named_training_dataset_without_record_manifest_is_not_known_overlap():
    audit = audit_training_overlap(
        _identity(),
        ModelTrainingEvidence(training_datasets=("FreeSolv",)),
    )
    assert audit.status is LeakageStatus.OVERLAP_UNKNOWN


def test_validation_ledger_seals_confirmation_and_independent_extrapolation():
    with pytest.raises(ValueError, match="before development"):
        ValidationLedger().add(
            ValidationPanel(
                "model-a",
                ValidationStage.CONFIRMATION,
                _identity(record_ids=("confirm",)),
                ModelTrainingEvidence(),
                selection_allowed=False,
            )
        )

    ledger = ValidationLedger()
    ledger.add(
        ValidationPanel(
            "model-a",
            ValidationStage.DEVELOPMENT,
            _identity(record_ids=("development",)),
            ModelTrainingEvidence(),
            selection_allowed=True,
        )
    )
    ledger.add(
        ValidationPanel(
            "model-a",
            ValidationStage.CONFIRMATION,
            _identity(record_ids=("confirmation",)),
            ModelTrainingEvidence(),
            selection_allowed=False,
        )
    )

    with pytest.raises(ValueError, match="strict training holdout"):
        ledger.add(
            ValidationPanel(
                "model-a",
                ValidationStage.INDEPENDENT_EXTRAPOLATION,
                _identity(record_ids=("final",)),
                ModelTrainingEvidence(),
                selection_allowed=False,
                independent_experimental_source="independent literature source",
            )
        )


def test_validation_ledger_accepts_only_a_disjoint_strict_final_holdout():
    ledger = ValidationLedger(
        (
            ValidationPanel(
                "model-a",
                ValidationStage.DEVELOPMENT,
                _identity(record_ids=("development",)),
                ModelTrainingEvidence(),
                selection_allowed=True,
            ),
            ValidationPanel(
                "model-a",
                ValidationStage.CONFIRMATION,
                _identity(record_ids=("confirmation",)),
                ModelTrainingEvidence(),
                selection_allowed=False,
            ),
        )
    )
    final = ValidationPanel(
        "model-a",
        ValidationStage.INDEPENDENT_EXTRAPOLATION,
        _identity(record_ids=("final",)),
        ModelTrainingEvidence(
            explicitly_excluded_datasets=("FreeSolv",),
            record_accounting_complete=True,
            evidence_source="public training manifest",
        ),
        selection_allowed=False,
        independent_experimental_source="independent experimental literature source",
    )
    ledger.add(final)
    assert ledger.panels[-1] is final

    other = ValidationLedger(
        (
            ValidationPanel(
                "model-a",
                ValidationStage.DEVELOPMENT,
                _identity(record_ids=("development",)),
                ModelTrainingEvidence(),
                selection_allowed=True,
            ),
            ValidationPanel(
                "model-a",
                ValidationStage.CONFIRMATION,
                _identity(record_ids=("confirmation",)),
                ModelTrainingEvidence(),
                selection_allowed=False,
            ),
        )
    )
    with pytest.raises(ValueError, match="must not reuse record IDs"):
        other.add(
            ValidationPanel(
                "model-a",
                ValidationStage.INDEPENDENT_EXTRAPOLATION,
                _identity(record_ids=("confirmation",)),
                ModelTrainingEvidence(
                    explicitly_excluded_datasets=("FreeSolv",),
                    record_accounting_complete=True,
                ),
                selection_allowed=False,
                independent_experimental_source="independent experimental literature source",
            )
        )


def test_metrics_include_coverage_correlations_and_seeded_bootstrap():
    summary = _accuracy_run(
        bootstrap_samples=200,
        seed=7,
    )
    assert summary["coverage"] == 1.0
    assert summary["metrics"]["mae_kcal_mol"] == pytest.approx(1.0)
    assert summary["metrics"]["rmse_kcal_mol"] == pytest.approx(1.0)
    assert summary["metrics"]["maximum_absolute_error_kcal_mol"] == pytest.approx(1.0)
    assert summary["metrics"]["spearman_rho"] == pytest.approx(1.0)
    assert summary["metrics"]["kendall_tau_b"] == pytest.approx(1.0)
    assert summary["mae_bootstrap_95_ci_kcal_mol"] == pytest.approx([1.0, 1.0])
    assert summary["identity_fingerprint"] == _accuracy_identity().fingerprint
    assert summary["target_quantity"] == "absolute_solvation_free_energy"
    assert summary["functional_group_coverage"]["observed_count"] == 10
    assert summary["functional_group_coverage"]["passes"] is True
    assert len(summary["per_record_errors"]) == 10
    assert summary["per_record_errors"][0]["experimental_provenance"] == (
        "literature-source-0"
    )
    assert summary["per_record_errors"][0]["experimental_unit"] == "kcal/mol"
    assert (
        summary["per_record_errors"][0]["experimental_source_artifact_sha256"]
        == EXPERIMENTAL_REFERENCE_SHA256
    )
    assert summary["per_record_errors"][0]["primary_functional_group"] == "alcohol"
    assert (
        summary["per_record_errors"][0]["functional_group_structure_identifier"]
        == "SMILES:CCO"
    )
    assert (
        summary["per_record_errors"][0]["solute_structure_identifier"] == "SMILES:CCO"
    )
    assert len(summary["per_record_errors"][0]["molecular_input_sha256"]) == 64
    assert (
        summary["per_record_errors"][0]["molecular_input_locator"]
        == f"file://{(ACCURACY_INPUTS / 'record_0.smi').resolve()}"
    )
    assert summary["per_record_errors"][0]["absolute_error_kcal_mol"] == 1.0
    assert summary["accuracy_evaluation_performed"] is True
    assert summary["accuracy_precision_policy"] == "float64"


def test_accuracy_metrics_reject_nonfinite_predictions_instead_of_shrinking_panel():
    identity = _accuracy_identity(
        accuracy_adapter_id=NONFINITE_TEST_REGISTRATION.adapter_id,
        accuracy_adapter_fingerprint=(
            NONFINITE_TEST_REGISTRATION.compute_fingerprint()
        ),
    )
    with pytest.raises(ValueError, match="finite prediction for every record"):
        _accuracy_run(identity=identity)


def test_runtime_smoke_counts_failures_without_computing_accuracy_metrics():
    records = _records()
    records[1]["predicted_kcal_mol"] = None
    summary = summarize_runtime_smoke(_identity(), records)
    assert summary["evaluated_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["coverage"] == 0.5
    assert summary["metrics"] is None
    assert summary["accuracy_evaluation_performed"] is False
    assert summary["accuracy_metric_reporting_allowed"] is False


def test_accuracy_metrics_reject_fewer_than_ten_distinct_functional_groups():
    taxonomy = _taxonomy()
    assignments = tuple(
        derive_functional_group_assignment(
            record_id=f"record_{index}",
            structure_identifier="SMILES:CCO",
            taxonomy=taxonomy,
            assignment_evidence="predeclared unit-test SMARTS match",
        )
        for index in range(10)
    )
    identity = _accuracy_identity(
        record_inputs=tuple(
            _accuracy_record_input(index, "CCO") for index in range(10)
        ),
        record_functional_groups=assignments,
    )

    with pytest.raises(ValueError, match="at least 10 distinct functional groups"):
        _accuracy_run(identity=identity)


def test_secondary_matches_cannot_inflate_primary_functional_group_coverage():
    taxonomy = _taxonomy()
    assignments = tuple(
        derive_functional_group_assignment(
            record_id=f"record_{index}",
            structure_identifier="SMILES:CC(=O)OC",
            taxonomy=taxonomy,
            assignment_evidence="predeclared unit-test SMARTS matches",
        )
        for index in range(10)
    )
    identity = _accuracy_identity(
        record_inputs=tuple(
            _accuracy_record_input(index, "CC(=O)OC") for index in range(10)
        ),
        record_functional_groups=assignments,
    )

    with pytest.raises(ValueError, match="at least 10 distinct functional groups"):
        _accuracy_run(identity=identity)


def test_same_methane_cannot_be_declared_as_ten_functional_groups():
    assignments = tuple(
        FunctionalGroupAssignment(
            record_id=f"record_{index}",
            structure_identifier="SMILES:C",
            primary_group=group_id,
            matched_groups=(group_id,),
            assignment_evidence="adversarial caller declaration",
        )
        for index, (group_id, _smiles) in enumerate(FUNCTIONAL_GROUP_EXAMPLES)
    )

    with pytest.raises(ValueError, match="no structural match"):
        _accuracy_identity(
            record_inputs=tuple(
                _accuracy_record_input(index, "C") for index in range(10)
            ),
            record_functional_groups=assignments,
        )


def test_actual_methane_prediction_inputs_cannot_borrow_ten_valid_assignments():
    def methane_only_predictor(*, record_id, molecular_input):
        del record_id, molecular_input
        return float(len((ACCURACY_INPUTS / "methane.smi").read_bytes()))

    with pytest.raises(TypeError):
        run_accuracy_panel(
            _accuracy_identity(),
            methane_only_predictor,
        )


def test_accuracy_metrics_reject_free_prediction_tables():
    records = [{"record_id": "record_0", "predicted_kcal_mol": 0.0}]
    with pytest.raises(TypeError, match="Direct prediction-table summarization"):
        summarize_predictions(records)


def test_accuracy_runner_fails_closed_without_reviewed_registered_adapter():
    identity = _accuracy_identity(
        accuracy_adapter_id="unregistered-model-adapter",
        accuracy_adapter_fingerprint="0" * 64,
    )

    with pytest.raises(RuntimeError, match="No reviewed formal accuracy adapter"):
        run_accuracy_panel(identity)

    drifted = _accuracy_identity(
        accuracy_adapter_id=UNIT_TEST_REGISTRATIONS[1.0].adapter_id,
        accuracy_adapter_fingerprint="0" * 64,
    )
    with pytest.raises(ValueError, match="implementation fingerprint"):
        run_accuracy_panel(drifted)


def test_accuracy_adapter_receipts_require_real_artifacts(tmp_path):
    with pytest.raises(TypeError, match="cannot be constructed from metadata"):
        AccuracyAdapterArtifactReceipt()
    with pytest.raises(FileNotFoundError):
        AccuracyAdapterArtifactReceipt.from_file(
            role="checkpoint",
            path=tmp_path / "missing-checkpoint.json",
        )


def test_accuracy_runner_rejects_live_predict_code_drift(monkeypatch):
    registration = UNIT_TEST_REGISTRATIONS[1.0]
    identity = _accuracy_identity(
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
    )

    def forged_predict(self, *, context):
        del self
        context.molecular_input.read()
        return float(context.record_id.removeprefix("record_"))

    monkeypatch.setattr(_UnitTestAccuracyAdapter, "predict", forged_predict)
    with pytest.raises(ValueError, match="implementation fingerprint"):
        run_accuracy_panel(identity)


def test_accuracy_runner_rejects_adapter_artifact_drift(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(UNIT_TEST_CHECKPOINT_PATH.read_bytes())
    adapter = _UnitTestAccuracyAdapter(1.0)
    registration = AccuracyAdapterRegistration.from_components(
        adapter_id="unit-test-artifact-drift",
        adapter=adapter,
        implementation_artifacts=(
            AccuracyAdapterArtifactReceipt.from_file(
                role="adapter_code",
                path=ACCURACY_ADAPTER_FIXTURE_PATH,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="featurizer_code",
                path=Path(pretrained_hub_module.__file__),
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="checkpoint",
                path=checkpoint,
            ),
            AccuracyAdapterArtifactReceipt.from_file(
                role="dependency_lock",
                path=ROOT / "pyproject.toml",
            ),
        ),
        configuration={"prediction_offset": 1.0},
    )
    monkeypatch.setattr(
        pretrained_hub_module,
        "_FORMAL_ACCURACY_ADAPTERS",
        MappingProxyType({registration.adapter_id: registration}),
    )
    identity = _accuracy_identity(
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
    )
    checkpoint.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="artifact bytes changed"):
        run_accuracy_panel(identity)


def test_accuracy_adapter_receives_no_experimental_labels():
    registration = LABEL_STEALING_TEST_REGISTRATION
    identity = _accuracy_identity(
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
    )

    with pytest.raises(AttributeError, match="record_experimental_references"):
        run_accuracy_panel(identity)


def test_accuracy_adapter_cannot_read_labels_from_parent_call_stack():
    registration = STACK_INSPECTING_TEST_REGISTRATION
    identity = _accuracy_identity(
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
    )

    summary = run_accuracy_panel(identity, bootstrap_samples=0)

    assert summary["metrics"]["mae_kcal_mol"] == 1.0
    assert summary["metrics"]["maximum_absolute_error_kcal_mol"] == 1.0
    assert summary["predictions_executed_in_label_free_spawned_interpreter"] is True


def test_accuracy_adapter_cannot_read_experimental_files_from_sandbox():
    registration = FILESYSTEM_SCANNING_TEST_REGISTRATION
    identity = _accuracy_identity(
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
    )

    summary = run_accuracy_panel(identity, bootstrap_samples=0)

    assert summary["metrics"]["mae_kcal_mol"] == 1.0
    assert summary["metrics"]["maximum_absolute_error_kcal_mol"] == 1.0
    assert summary["predictions_executed_in_filesystem_sandbox"] is True


def test_label_free_worker_job_contains_bytes_but_no_input_source_path():
    identity = _accuracy_identity()
    record_input = identity.record_inputs[0]

    job = pretrained_hub_module._label_free_prediction_job(
        identity,
        record_id=record_input.record_id,
        record_input=record_input,
    )
    serialized = json.dumps(job, sort_keys=True)

    assert job["molecular_input"]["payload_base64"]
    assert "source_path" not in serialized
    assert "archive_member" not in serialized
    assert "molecular_input_locator" not in serialized


def test_isolated_context_receives_only_registered_implementation_artifact_paths():
    identity = _accuracy_identity()
    record_input = identity.record_inputs[0]
    job = pretrained_hub_module._label_free_prediction_job(
        identity,
        record_id=record_input.record_id,
        record_input=record_input,
    )
    artifact_paths = {
        "adapter_code": "/registered/adapter.py",
        "checkpoint": "/registered/model.pt",
    }

    context = pretrained_hub_module._prediction_context_from_isolated_job(
        job,
        implementation_artifact_paths=artifact_paths,
    )

    assert dict(context.implementation_artifact_paths) == artifact_paths
    assert "implementation_artifact_paths" not in json.dumps(job, sort_keys=True)


def test_accuracy_panel_fails_closed_without_filesystem_sandbox(monkeypatch):
    identity = _accuracy_identity()
    monkeypatch.setattr(
        pretrained_hub_module.shutil,
        "which",
        lambda _name: None,
    )

    with pytest.raises(RuntimeError, match="requires bubblewrap"):
        run_accuracy_panel(identity, bootstrap_samples=0)


def test_formal_accuracy_worker_has_a_runner_owned_whole_panel_timeout(monkeypatch):
    identity = _accuracy_identity()
    registration = UNIT_TEST_REGISTRATIONS[1.0]
    jobs = tuple(
        pretrained_hub_module._label_free_prediction_job(
            identity,
            record_id=record_input.record_id,
            record_input=record_input,
        )
        for record_input in identity.record_inputs
    )
    monkeypatch.setattr(
        pretrained_hub_module,
        "_accuracy_sandbox_command",
        lambda **_kwargs: [
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
    )
    monkeypatch.setattr(
        pretrained_hub_module,
        "_accuracy_worker_timeout_seconds",
        lambda *_args, **_kwargs: 0.05,
    )

    with pytest.raises(RuntimeError, match="whole-panel .* timeout"):
        pretrained_hub_module._run_isolated_accuracy_predictions(
            identity=identity,
            registration=registration,
            jobs=jobs,
        )


def test_formal_accuracy_worker_bounds_aggregate_output_without_limiting_scratch(
    monkeypatch,
    tmp_path,
):
    identity = _accuracy_identity()
    registration = UNIT_TEST_REGISTRATIONS[1.0]
    jobs = tuple(
        pretrained_hub_module._label_free_prediction_job(
            identity,
            record_id=record_input.record_id,
            record_input=record_input,
        )
        for record_input in identity.record_inputs
    )
    output_limit = 1024
    monkeypatch.setattr(
        pretrained_hub_module,
        "FORMAL_ACCURACY_WORKER_MAX_OUTPUT_BYTES",
        output_limit,
    )
    scratch_path = tmp_path / "scratch-artifact.bin"
    completed = pretrained_hub_module._run_bounded_accuracy_worker(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path;"
                f"Path({str(scratch_path)!r}).write_bytes(b'x'*{output_limit * 4});"
                "print('ok')"
            ),
        ],
        serialized_request="{}",
        timeout_seconds=5.0,
        env={
            "HOME": str(tmp_path),
            "PATH": str(Path(sys.executable).parent),
            "TMPDIR": str(tmp_path),
        },
    )
    assert completed.returncode == 0
    assert completed.stdout == "ok\n"
    assert scratch_path.stat().st_size == output_limit * 4

    monkeypatch.setattr(
        pretrained_hub_module,
        "_accuracy_sandbox_command",
        lambda **_kwargs: [
            sys.executable,
            "-c",
            (
                "import sys;"
                f"sys.stdout.write('x'*{output_limit});"
                "sys.stderr.write('y')"
            ),
        ],
    )
    with pytest.raises(RuntimeError, match="aggregate stdout/stderr output limit"):
        pretrained_hub_module._run_isolated_accuracy_predictions(
            identity=identity,
            registration=registration,
            jobs=jobs,
        )


def test_whole_panel_timeout_covers_every_configured_record_budget():
    class RegistrationWithLongRecordTimeout:
        configuration_json = '{"timeout_seconds":180.0}'

    assert (
        pretrained_hub_module._accuracy_worker_timeout_seconds(
            RegistrationWithLongRecordTimeout(),
            job_count=1,
        )
        == 240.0
    )
    assert (
        pretrained_hub_module._accuracy_worker_timeout_seconds(
            RegistrationWithLongRecordTimeout(),
            job_count=10,
        )
        == 1860.0
    )


def test_timeout_kills_descendants_after_the_direct_worker_exits(tmp_path):
    marker = tmp_path / "surviving-descendant.txt"
    child_code = (
        "import time;"
        "from pathlib import Path;"
        "time.sleep(0.4);"
        f"Path({str(marker)!r}).write_text('survived', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess,sys;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}])"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        pretrained_hub_module._run_bounded_accuracy_worker(
            [sys.executable, "-c", parent_code],
            serialized_request="{}",
            timeout_seconds=0.05,
            env={
                "HOME": str(tmp_path),
                "PATH": str(Path(sys.executable).parent),
                "TMPDIR": str(tmp_path),
            },
        )

    time.sleep(0.5)
    assert not marker.exists()


def _matched_qm_performance_kwargs():
    return {
        "maximum_absolute_error_limit_kcal_mol": 1.5,
        "candidate_full_task_seconds": (2.0, 2.1, 1.9),
        "qm_full_task_seconds": (4.0, 4.2, 3.8),
        "candidate_cold_start_seconds": (2.8,),
        "qm_cold_start_seconds": (4.8,),
        "candidate_warm_state_seconds": (1.0, 1.1, 0.9),
        "qm_warm_state_seconds": (2.0, 2.1, 1.9),
        "candidate_scope_components": END_TO_END_RUNTIME_SCOPE_COMPONENTS,
        "qm_scope_components": END_TO_END_RUNTIME_SCOPE_COMPONENTS,
        "accepted_candidate_precision_policy": "float64",
        "timed_candidate_precision_policy": "float64",
        "accepted_qm_precision_policy": "float64",
        "timed_qm_precision_policy": "float64",
        "hardware_fingerprint": "unit-test-host-cpu-and-gpu-v1",
    }


def _matched_qm_accuracy_identities(
    *,
    candidate_overrides=None,
    qm_overrides=None,
):
    candidate_registration = UNIT_TEST_REGISTRATIONS[0.5]
    candidate_values = {
        "accuracy_adapter_id": candidate_registration.adapter_id,
        "accuracy_adapter_fingerprint": candidate_registration.compute_fingerprint(),
    }
    candidate_values.update(candidate_overrides or {})
    return (
        _accuracy_identity(**candidate_values),
        _accuracy_identity(**(qm_overrides or {})),
    )


def test_matched_qm_performance_requires_accuracy_then_strict_full_task_speedup():
    candidate, qm = _matched_qm_accuracy_identities()

    result = evaluate_matched_qm_performance(
        candidate,
        qm,
        **_matched_qm_performance_kwargs(),
    )

    assert result["accuracy_precondition_passed"] is True
    assert result["full_task"]["speedup_qm_over_candidate"] == pytest.approx(2.0)
    assert result["full_task"]["gate_passed"] is True
    assert result["performance_policy_gate_passed"] is True
    assert result["performance_admission_passed"] is False
    assert result["admission_status"] == ("blocked_pending_trusted_timing_receipts")
    assert "trusted timing receipts" in result["blocking_reason"]
    assert result["cold_start_diagnostic"]["speedup_qm_over_candidate"] == (
        pytest.approx(4.8 / 2.8)
    )

    tied = _matched_qm_performance_kwargs()
    tied["qm_full_task_seconds"] = tied["candidate_full_task_seconds"]
    result = evaluate_matched_qm_performance(candidate, qm, **tied)
    assert result["full_task"]["speedup_qm_over_candidate"] == 1.0
    assert result["performance_policy_gate_passed"] is False
    assert result["performance_admission_passed"] is False
    assert result["admission_status"] == "rejected_not_faster_than_qm"
    assert "not faster than QM" in result["rejection_reason"]

    slower = _matched_qm_performance_kwargs()
    slower["candidate_full_task_seconds"] = (5.0, 5.1, 4.9)
    result = evaluate_matched_qm_performance(candidate, qm, **slower)
    assert result["full_task"]["speedup_qm_over_candidate"] == pytest.approx(0.8)
    assert result["performance_policy_gate_passed"] is False
    assert result["admission_status"] == "rejected_not_faster_than_qm"


def test_matched_qm_performance_rejects_failed_accuracy_or_precision_reduction():
    candidate, qm = _matched_qm_accuracy_identities()
    forged_aggregate = _accuracy_run(identity=candidate)
    forged_aggregate["metrics"]["maximum_absolute_error_kcal_mol"] = 1.5

    with pytest.raises(TypeError, match="trusted formal accuracy"):
        evaluate_matched_qm_performance(
            forged_aggregate,
            qm,
            **_matched_qm_performance_kwargs(),
        )

    with pytest.raises(ValueError, match="accuracy gate failed"):
        evaluate_matched_qm_performance(
            candidate,
            qm,
            maximum_absolute_error_limit_kcal_mol=0.5,
            **{
                key: value
                for key, value in _matched_qm_performance_kwargs().items()
                if key != "maximum_absolute_error_limit_kcal_mol"
            },
        )

    reduced_precision = _matched_qm_performance_kwargs()
    reduced_precision["timed_candidate_precision_policy"] = "float32"
    with pytest.raises(ValueError, match="reduced-precision speedups are forbidden"):
        evaluate_matched_qm_performance(
            candidate,
            qm,
            **reduced_precision,
        )

    reduced_precision = _matched_qm_performance_kwargs()
    reduced_precision["accepted_candidate_precision_policy"] = "float16"
    reduced_precision["timed_candidate_precision_policy"] = "float16"
    with pytest.raises(ValueError, match="explicit float32 or float64"):
        evaluate_matched_qm_performance(
            candidate,
            qm,
            **reduced_precision,
        )


def test_matched_qm_performance_requires_the_same_precision_for_both_methods():
    candidate, qm = _matched_qm_accuracy_identities()
    mismatched_precision = _matched_qm_performance_kwargs()
    mismatched_precision.update(
        {
            "accepted_candidate_precision_policy": "float32",
            "timed_candidate_precision_policy": "float32",
            "accepted_qm_precision_policy": "float64",
            "timed_qm_precision_policy": "float64",
        }
    )

    with pytest.raises(ValueError, match="same precision policy"):
        evaluate_matched_qm_performance(
            candidate,
            qm,
            **mismatched_precision,
        )


def test_matched_qm_performance_allows_omitted_diagnostic_timings():
    candidate, qm = _matched_qm_accuracy_identities()
    full_task_only = _matched_qm_performance_kwargs()
    for field in (
        "candidate_cold_start_seconds",
        "qm_cold_start_seconds",
        "candidate_warm_state_seconds",
        "qm_warm_state_seconds",
    ):
        full_task_only.pop(field)

    result = evaluate_matched_qm_performance(candidate, qm, **full_task_only)

    assert result["full_task"]["gate_passed"] is True
    assert result["cold_start_diagnostic"] is None
    assert result["warm_state_diagnostic"] is None


def test_matched_qm_performance_rejects_consistent_ten_group_relabeling():
    candidate = _accuracy_run(prediction_offset=0.5)
    qm = _accuracy_run(prediction_offset=1.0)
    forged_candidate = json.loads(json.dumps(candidate))
    forged_qm = json.loads(json.dumps(qm))
    forged_groups = [f"claimed-group-{index}" for index in range(10)]
    for summary in (forged_candidate, forged_qm):
        for row, forged_group in zip(summary["per_record_errors"], forged_groups):
            row["primary_functional_group"] = forged_group
        summary["functional_group_coverage"]["observed_count"] = 10
        summary["functional_group_coverage"][
            "observed_functional_groups"
        ] = forged_groups

    with pytest.raises(TypeError, match="trusted formal accuracy"):
        evaluate_matched_qm_performance(
            forged_candidate,
            forged_qm,
            **_matched_qm_performance_kwargs(),
        )


def test_matched_qm_performance_rejects_forged_task_fingerprint_for_other_geometry():
    candidate_identity, qm_identity = _matched_qm_accuracy_identities(
        candidate_overrides={"geometry_protocol": "optimized-candidate"},
        qm_overrides={"geometry_protocol": "fixed-qm-input"},
    )
    candidate = _accuracy_run(identity=candidate_identity)
    qm = _accuracy_run(identity=qm_identity)
    assert candidate["matched_task_fingerprint"] != qm["matched_task_fingerprint"]
    forged_candidate = json.loads(json.dumps(candidate))
    forged_candidate["matched_task_fingerprint"] = qm["matched_task_fingerprint"]

    with pytest.raises(TypeError, match="trusted formal accuracy"):
        evaluate_matched_qm_performance(
            forged_candidate,
            qm_identity,
            **_matched_qm_performance_kwargs(),
        )
    with pytest.raises(ValueError, match="same trusted benchmark task"):
        evaluate_matched_qm_performance(
            candidate_identity,
            qm_identity,
            **_matched_qm_performance_kwargs(),
        )


def test_matched_qm_performance_recomputes_errors_and_group_coverage():
    candidate = _accuracy_run(prediction_offset=0.5)
    forged_errors = json.loads(json.dumps(candidate))
    for row in forged_errors["per_record_errors"]:
        row["predicted_kcal_mol"] = 999.0
        row["signed_error_kcal_mol"] = 999.0
        row["absolute_error_kcal_mol"] = 999.0

    with pytest.raises(ValueError, match="per-record accuracy evidence"):
        pretrained_hub_module._accuracy_signature_for_performance(
            forged_errors,
            maximum_absolute_error_limit_kcal_mol=1.5,
            label="candidate",
        )

    forged_groups = json.loads(json.dumps(candidate))
    for row in forged_groups["per_record_errors"]:
        row["primary_functional_group"] = "alcohol"
    forged_groups["functional_group_coverage"]["observed_count"] = 10
    forged_groups["functional_group_coverage"]["observed_functional_groups"] = [
        f"claimed-group-{index}" for index in range(10)
    ]

    with pytest.raises(ValueError, match="missing or duplicate rows"):
        pretrained_hub_module._accuracy_signature_for_performance(
            forged_groups,
            maximum_absolute_error_limit_kcal_mol=1.5,
            label="candidate",
        )


def test_parent_helper_monkeypatch_cannot_change_fresh_worker_predictions(
    monkeypatch,
):
    registration = HELPER_DELEGATING_TEST_REGISTRATION
    identity = _accuracy_identity(
        accuracy_adapter_id=registration.adapter_id,
        accuracy_adapter_fingerprint=registration.compute_fingerprint(),
    )
    frozen_fingerprint = registration.compute_fingerprint()

    def forged_helper(self, record_id):
        del self
        return float(record_id.removeprefix("record_"))

    monkeypatch.setattr(
        _HelperDelegatingAccuracyAdapter,
        "_impl",
        forged_helper,
    )
    assert registration.compute_fingerprint() == frozen_fingerprint

    summary = run_accuracy_panel(identity, bootstrap_samples=0)

    assert summary["metrics"]["mae_kcal_mol"] == 1.0
    assert summary["metrics"]["maximum_absolute_error_kcal_mol"] == 1.0


def test_molecular_input_receipt_rejects_missing_or_changed_source(tmp_path):
    with pytest.raises(TypeError, match="cannot be constructed from metadata"):
        MolecularInputReceipt()

    missing = tmp_path / "missing.smi"
    with pytest.raises(FileNotFoundError):
        MolecularInputReceipt.from_file(
            missing,
            molecular_input_format="smiles",
        )

    source = tmp_path / "input.smi"
    source.write_text("CCO\n", encoding="utf-8")
    receipt = MolecularInputReceipt.from_file(
        source,
        molecular_input_format="smiles",
    )
    source.write_text("C\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source bytes changed"):
        receipt.read_verified_payload()


def test_molecular_input_receipt_reads_exact_zip_member(tmp_path):
    archive_path = tmp_path / "inputs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("records/alcohol.smi", b"CCO\n")
        archive.writestr("records/methane.smi", b"C\n")

    receipt = MolecularInputReceipt.from_file(
        archive_path,
        archive_member="records/alcohol.smi",
        molecular_input_format="smiles",
    )

    assert receipt.read_verified_payload() == b"CCO\n"
    assert receipt.solute_structure_identifier == "SMILES:CCO"
    assert receipt.molecular_input_locator.endswith("!/records/alcohol.smi")


def test_molecular_input_receipt_accepts_only_same_graph_sdf_conformers(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    def mol_block(smiles, seed):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = seed
        assert AllChem.EmbedMolecule(molecule, parameters) == 0
        return Chem.MolToMolBlock(molecule)

    same_graph = tmp_path / "ethanol-conformers.sdf"
    same_graph.write_text(
        mol_block("CCO", 11) + "\n$$$$\n" + mol_block("CCO", 12) + "\n$$$$\n",
        encoding="utf-8",
    )

    receipt = MolecularInputReceipt.from_file(
        same_graph,
        molecular_input_format="sdf_conformers",
    )
    assert receipt.solute_structure_identifier == "SMILES:CCO"

    unspecified_stereocenters = tmp_path / "mannitol-conformers.sdf"
    unspecified_stereocenters.write_text(
        mol_block("OCC(O)C(O)C(O)C(O)CO", 31)
        + "\n$$$$\n"
        + mol_block("OCC(O)C(O)C(O)C(O)CO", 32)
        + "\n$$$$\n",
        encoding="utf-8",
    )
    receipt = MolecularInputReceipt.from_file(
        unspecified_stereocenters,
        molecular_input_format="sdf_conformers",
    )
    assert receipt.solute_structure_identifier == "SMILES:OCC(O)C(O)C(O)C(O)CO"

    same_stereoisomer = tmp_path / "same-stereoisomer-conformers.sdf"
    same_stereoisomer.write_text(
        mol_block("F[C@H](Cl)[C@H](Br)I", 41)
        + "\n$$$$\n"
        + mol_block("F[C@H](Cl)[C@H](Br)I", 42)
        + "\n$$$$\n",
        encoding="utf-8",
    )
    MolecularInputReceipt.from_file(
        same_stereoisomer,
        molecular_input_format="sdf_conformers",
    )

    mixed_stereoisomers = tmp_path / "mixed-stereoisomer-conformers.sdf"
    mixed_stereoisomers.write_text(
        mol_block("F[C@H](Cl)[C@H](Br)I", 51)
        + "\n$$$$\n"
        + mol_block("F[C@H](Cl)[C@@H](Br)I", 52)
        + "\n$$$$\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distinct stereoisomers cannot be averaged"):
        MolecularInputReceipt.from_file(
            mixed_stereoisomers,
            molecular_input_format="sdf_conformers",
        )

    with pytest.raises(ValueError, match="exactly one molecule record"):
        MolecularInputReceipt.from_file(
            same_graph,
            molecular_input_format="sdf",
        )

    mixed_graphs = tmp_path / "mixed-conformers.sdf"
    mixed_graphs.write_text(
        mol_block("CCO", 21) + "\n$$$$\n" + mol_block("CCN", 22) + "\n$$$$\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same canonical molecular graph"):
        MolecularInputReceipt.from_file(
            mixed_graphs,
            molecular_input_format="sdf_conformers",
        )


def test_caller_cannot_choose_a_secondary_match_as_the_primary_family():
    taxonomy = _taxonomy()
    derived = derive_functional_group_assignment(
        record_id="record_5",
        structure_identifier="SMILES:CC(=O)OC",
        taxonomy=taxonomy,
        assignment_evidence="predeclared unit-test SMARTS matches",
    )
    assert derived.primary_group == "ester"
    assert "ether" in derived.matched_groups
    forged = FunctionalGroupAssignment(
        record_id=derived.record_id,
        structure_identifier=derived.structure_identifier,
        primary_group="ether",
        matched_groups=derived.matched_groups,
        assignment_evidence=derived.assignment_evidence,
    )
    assignments = list(_accuracy_identity().record_functional_groups)
    assignments[5] = forged

    with pytest.raises(ValueError, match="does not follow the frozen precedence"):
        _accuracy_identity(record_functional_groups=tuple(assignments))


def test_experimental_reference_requires_real_json_source(tmp_path):
    with pytest.raises(TypeError, match="cannot be constructed from metadata"):
        BenchmarkExperimentalReference()
    with pytest.raises(FileNotFoundError):
        BenchmarkExperimentalReference.from_json_file(
            tmp_path / "missing.json",
            record_id="record_0",
        )
    with pytest.raises(ValueError, match="top-level JSON entry"):
        BenchmarkExperimentalReference.from_json_file(
            EXPERIMENTAL_REFERENCE_PATH,
            record_id="record_0",
            source_record_locator="$.record_1",
        )


def test_experimental_reference_source_is_reverified_before_prediction(tmp_path):
    source = tmp_path / "experimental-references.json"
    source.write_bytes(EXPERIMENTAL_REFERENCE_PATH.read_bytes())
    references = _accuracy_references(source)
    identity = _accuracy_identity(record_experimental_references=references)
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source bytes changed"):
        run_accuracy_panel(identity)


def test_experimental_reference_paths_do_not_change_panel_identity(tmp_path):
    copied_source = tmp_path / "same-references.json"
    copied_source.write_bytes(EXPERIMENTAL_REFERENCE_PATH.read_bytes())

    original = _accuracy_identity()
    copied = _accuracy_identity(
        record_experimental_references=_accuracy_references(copied_source)
    )

    assert original.fingerprint == copied.fingerprint
    assert original.panel_fingerprint == copied.panel_fingerprint


@pytest.mark.parametrize("invalid_reference", [None, "not-a-number", float("nan")])
def test_metrics_reject_invalid_experimental_references(
    invalid_reference,
    tmp_path,
):
    path = tmp_path / "invalid-reference.json"
    path.write_text(
        json.dumps(
            {
                "record_0": {
                    "value": invalid_reference,
                    "unit": "kcal/mol",
                    "provenance": "literature-source-0",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((TypeError, ValueError), match="value"):
        BenchmarkExperimentalReference.from_json_file(
            path,
            record_id="record_0",
            source_record_locator="$.record_0",
        )


def test_metrics_reject_records_that_do_not_match_the_declared_panel():
    with pytest.raises(ValueError, match="exactly one frozen reference"):
        _accuracy_identity(record_experimental_references=_accuracy_references()[:1])


def test_accuracy_metrics_require_record_level_experimental_provenance(tmp_path):
    path = tmp_path / "missing-provenance.json"
    path.write_text(
        json.dumps(
            {
                "record_0": {
                    "value": 0.0,
                    "unit": "kcal/mol",
                    "provenance": "",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance"):
        BenchmarkExperimentalReference.from_json_file(
            path,
            record_id="record_0",
            source_record_locator="$.record_0",
        )


def test_experimental_label_changes_are_frozen_into_identity():
    original = _accuracy_identity()
    changed = _accuracy_identity(
        record_experimental_references=_accuracy_references(
            EXPERIMENTAL_REFERENCE_OFFSET_PATH
        )
    )

    assert original.fingerprint != changed.fingerprint
    assert _accuracy_run(identity=original)["metrics"]["mae_kcal_mol"] == 1.0
    assert _accuracy_run(identity=changed)["metrics"]["mae_kcal_mol"] == 0.0


def test_paired_comparison_requires_panel_identity_match():
    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            _accuracy_identity(),
            _accuracy_identity(dataset="MNSol"),
        )


def test_paired_comparison_requires_complete_successful_record_matrix():
    with pytest.raises(TypeError, match="BenchmarkIdentity"):
        paired_comparison(
            _accuracy_identity(),
            [{"record_id": "record_0"}],
        )


def test_paired_comparison_reports_wins_and_delta():
    registration_a = UNIT_TEST_REGISTRATIONS[0.5]
    identity_a = _accuracy_identity(
        accuracy_adapter_id=registration_a.adapter_id,
        accuracy_adapter_fingerprint=registration_a.compute_fingerprint(),
    )
    identity_b = _accuracy_identity(potential="alternative")
    comparison = paired_comparison(
        identity_a,
        identity_b,
    )
    assert comparison.record_count == 10
    assert comparison.functional_group_count == 10
    assert comparison.wins_a == 10
    assert comparison.mean_delta_absolute_error_kcal_mol == pytest.approx(-0.5)
    assert comparison.metrics_a["maximum_absolute_error_kcal_mol"] == 0.5
    assert comparison.metrics_b["maximum_absolute_error_kcal_mol"] == 1.0
    assert len(comparison.per_record_errors_a) == 10
    assert comparison.per_record_errors_a[0]["experimental_provenance"] == (
        "literature-source-0"
    )
    assert comparison.benchmark_fingerprint == _accuracy_identity().panel_fingerprint
    assert comparison.panel_fingerprint == _accuracy_identity().panel_fingerprint
    assert comparison.run_fingerprint_a == identity_a.fingerprint
    assert comparison.run_fingerprint_b == identity_b.fingerprint


def test_paired_comparison_rejects_solvent_protocol_or_backend_mismatch():
    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            _accuracy_identity(),
            _accuracy_identity(solvation_backend="different-backend"),
        )
    with pytest.raises(ValueError, match="scientific panel identities"):
        paired_comparison(
            _accuracy_identity(),
            _accuracy_identity(solvent_protocol="octanol"),
        )


def test_result_store_is_path_safe_and_requires_full_artifact_set(tmp_path):
    with pytest.raises(ValueError, match="path-safe"):
        BenchmarkResultStore(tmp_path, "../escape")

    store = BenchmarkResultStore(tmp_path, "model-a")
    with pytest.raises(RuntimeError, match="incomplete"):
        store.assert_complete()
    with pytest.raises(ValueError, match="path-safe"):
        store.write_json("../escape.json", {"unsafe": True})
    with pytest.raises(ValueError, match="path-safe"):
        store.write_csv(
            "nested/escape.csv",
            [],
            fieldnames=("record_id",),
        )

    store.write_json_yaml("model_card.yaml", {"model_id": "model-a"})
    store.write_json_yaml("protocol.yaml", {"fingerprint": _identity().fingerprint})
    store.write_environment_lock("python==3.11")
    store.write_csv(
        "per_record.csv",
        _records(),
        fieldnames=("record_id", "experimental_kcal_mol", "predicted_kcal_mol"),
    )
    store.write_csv(
        "failures.csv",
        [],
        fieldnames=("record_id", "reason"),
    )
    store.write_json("diagnostics.json", {"status": "ok"})
    store.write_csv(
        "uncertainty.csv",
        [],
        fieldnames=("record_id", "uncertainty_kcal_mol"),
    )
    store.write_json("leakage_report.json", {"status": "overlap_unknown"})
    store.write_json("speed.json", {"energy_evaluations_per_second": 1.0})
    store.write_json("summary.json", {"mae_kcal_mol": 1.0})
    store.assert_complete()

    model_card = json.loads(
        (store.directory / "model_card.yaml").read_text(encoding="utf-8")
    )
    assert model_card["model_id"] == "model-a"
