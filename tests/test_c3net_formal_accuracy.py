from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import maple.function.benchmarking.c3net_accuracy as c3net_accuracy
import maple.function.benchmarking.pretrained_hub as pretrained_hub
from maple.function.benchmarking import (
    AccuracyPredictionContext,
    BenchmarkQuantity,
    VerifiedMolecularInput,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sdf_payload(smiles: str = "CCO") -> bytes:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 19
    assert AllChem.EmbedMolecule(molecule, parameters) == 0
    return (Chem.MolToMolBlock(molecule) + "\n$$$$\n").encode("utf-8")


def _context(
    payload: bytes,
    artifact_paths: dict[str, str],
    *,
    target_quantity: BenchmarkQuantity = BenchmarkQuantity.PROPERTY_PREDICTION,
    solvent_components: tuple[str, ...] = ("dmf",),
    solvent_mole_fractions: tuple[float, ...] = (1.0,),
) -> AccuracyPredictionContext:
    molecular_input = VerifiedMolecularInput._from_verified_payload(
        payload,
        solute_structure_identifier="SMILES:CCO",
        molecular_input_sha256=hashlib.sha256(payload).hexdigest(),
        molecular_input_format="sdf_conformers",
    )
    return AccuracyPredictionContext(
        record_id="record-1",
        molecular_input=molecular_input,
        temperature_kelvin=298.15,
        standard_state="1 mol/L gas and 1 mol/L liquid",
        protonation_policy="neutral",
        tautomer_policy="as supplied",
        conformer_policy="one deterministic conformer",
        geometry_protocol="ETKDGv3",
        solvent_protocol="pure solvent",
        solvent_components=solvent_components,
        solvent_mole_fractions=solvent_mole_fractions,
        potential="C3Net checkpoint-1",
        solvation_backend="C3Net property head",
        cavity_model="not applicable",
        sampling_protocol="not applicable",
        estimator="arithmetic mean",
        target_quantity=target_quantity,
        implementation_artifact_paths=MappingProxyType(artifact_paths),
    )


def _artifact_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    paths = {
        "adapter_code": tmp_path / "adapter.py",
        "featurizer_code": tmp_path / "source.bundle",
        "checkpoint": tmp_path / "checkpoint.pt",
        "embedding": tmp_path / "embedding.pt",
        "property_adapter_code": tmp_path / "c3net_property.py",
        "runner_code": tmp_path / "pretrained_hub.py",
        "dependency_lock": tmp_path / "environment.yaml",
        "runtime_lock": tmp_path / "runtime-lock.json",
    }
    for role, path in paths.items():
        path.write_bytes(f"{role}-bytes".encode("utf-8"))
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_SOURCE_BUNDLE_SHA256",
        _sha256(paths["featurizer_code"]),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_CHECKPOINT_SHA256",
        _sha256(paths["checkpoint"]),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_EMBEDDING_SHA256",
        _sha256(paths["embedding"]),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_DEPENDENCY_LOCK_SHA256",
        _sha256(paths["dependency_lock"]),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_FORMAL_RUNTIME_LOCK_SHA256",
        _sha256(paths["runtime_lock"]),
    )
    return {role: str(path) for role, path in paths.items()}


def test_formal_adapter_uses_only_registered_artifacts_and_pure_solvent(
    tmp_path, monkeypatch
):
    artifact_paths = _artifact_paths(tmp_path, monkeypatch)
    seen: dict[str, object] = {}

    def materialize(_bundle_path, destination):
        checkpoint = destination / "prediction/model/checkpoint-1.pth.tar"
        embedding = destination / "module/input/embedding.pt"
        checkpoint.parent.mkdir(parents=True)
        embedding.parent.mkdir(parents=True)
        checkpoint.write_bytes(Path(artifact_paths["checkpoint"]).read_bytes())
        embedding.write_bytes(Path(artifact_paths["embedding"]).read_bytes())
        return destination

    class FakePropertyAdapter:
        def __init__(
            self,
            source_root,
            *,
            timeout_seconds,
            runtime_manifest_path,
        ):
            seen["source_root"] = source_root
            seen["timeout_seconds"] = timeout_seconds
            seen["runtime_manifest_path"] = runtime_manifest_path

        def predict_conformers(self, sdf_path, solvent):
            seen["payload"] = Path(sdf_path).read_bytes()
            seen["solvent"] = solvent
            return SimpleNamespace(predicted_solvation_free_energy_kcal_mol=-4.25)

    monkeypatch.setattr(
        c3net_accuracy,
        "_materialize_source_checkout",
        materialize,
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NetPropertyAdapter",
        FakePropertyAdapter,
    )
    payload = _sdf_payload()
    context = _context(payload, artifact_paths)
    code_fingerprint_before = pretrained_hub._adapter_runtime_implementation_payload(
        c3net_accuracy.C3NetFormalAccuracyAdapter()
    )["predict_code_sha256"]

    prediction = c3net_accuracy.C3NetFormalAccuracyAdapter().predict(context=context)
    code_fingerprint_after = pretrained_hub._adapter_runtime_implementation_payload(
        c3net_accuracy.C3NetFormalAccuracyAdapter()
    )["predict_code_sha256"]

    assert prediction == -4.25
    assert code_fingerprint_after == code_fingerprint_before
    assert seen["payload"] == payload
    assert seen["solvent"] == "dimethylformamide"
    assert seen["runtime_manifest_path"] == Path(artifact_paths["runtime_lock"])
    assert context.molecular_input.was_consumed is True


def test_formal_adapter_rejects_wrong_quantity_or_mixture_before_consumption(
    tmp_path, monkeypatch
):
    artifact_paths = _artifact_paths(tmp_path, monkeypatch)
    payload = _sdf_payload()
    adapter = c3net_accuracy.C3NetFormalAccuracyAdapter()

    wrong_quantity = _context(
        payload,
        artifact_paths,
        target_quantity=BenchmarkQuantity.ABSOLUTE_SOLVATION_FREE_ENERGY,
    )
    with pytest.raises(ValueError, match="property_prediction"):
        adapter.predict(context=wrong_quantity)
    assert wrong_quantity.molecular_input.was_consumed is False

    mixture = _context(
        payload,
        artifact_paths,
        solvent_components=("water", "methanol"),
        solvent_mole_fractions=(0.5, 0.5),
    )
    with pytest.raises(ValueError, match="exactly one pure solvent"):
        adapter.predict(context=mixture)
    assert mixture.molecular_input.was_consumed is False


def test_registration_fingerprint_is_independent_of_artifact_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        c3net_accuracy.C3NetPropertyAdapter,
        "__init__",
        lambda self, source_root, timeout_seconds, runtime_manifest_path: None,
    )
    monkeypatch.setattr(
        c3net_accuracy.C3NetPropertyAdapter,
        "assert_source_identity",
        lambda self: None,
    )

    def make_source(parent: Path) -> tuple[Path, Path]:
        root = parent / "source"
        (root / "prediction/model").mkdir(parents=True)
        (root / "module/input").mkdir(parents=True)
        (root / "prediction/model/checkpoint-1.pth.tar").write_bytes(b"checkpoint")
        (root / "module/input/embedding.pt").write_bytes(b"embedding")
        (root / "environment.yaml").write_bytes(b"dependencies")
        bundle = parent / "source.bundle"
        bundle.write_bytes(b"bundle")
        return root, bundle

    left_root, left_bundle = make_source(tmp_path / "left")
    right_root, right_bundle = make_source(tmp_path / "right")
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_SOURCE_BUNDLE_SHA256",
        _sha256(left_bundle),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_CHECKPOINT_SHA256",
        _sha256(left_root / "prediction/model/checkpoint-1.pth.tar"),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_EMBEDDING_SHA256",
        _sha256(left_root / "module/input/embedding.pt"),
    )
    monkeypatch.setattr(
        c3net_accuracy,
        "C3NET_DEPENDENCY_LOCK_SHA256",
        _sha256(left_root / "environment.yaml"),
    )

    left = c3net_accuracy.build_c3net_formal_accuracy_registration(
        source_root=left_root,
        source_bundle=left_bundle,
    )
    right = c3net_accuracy.build_c3net_formal_accuracy_registration(
        source_root=right_root,
        source_bundle=right_bundle,
    )

    assert left.compute_fingerprint() == right.compute_fingerprint()
    assert all(
        str(tmp_path) not in str(value) for value in left.adapter.__dict__.values()
    )


def test_partial_environment_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("MAPLE_C3NET_FORMAL_SOURCE_ROOT", "/configured/source")
    monkeypatch.delenv("MAPLE_C3NET_FORMAL_SOURCE_BUNDLE", raising=False)

    with pytest.raises(RuntimeError, match="requires both"):
        pretrained_hub._configured_formal_accuracy_adapters()
