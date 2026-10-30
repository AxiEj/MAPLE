from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maple.function.calculator.extra_correction.implicit.charges import ChargeResult
from maple.function.calculator.extra_correction.implicit.correction import (
    ImplicitSolvationCorrection,
)
from maple.function.calculator.extra_correction.implicit import (
    private_torch_chagb as private_adapter,
)
from maple.function.calculator.extra_correction.implicit.private_torch_chagb import (
    KCAL_PER_MOL_PER_HARTREE,
    PrivateTorchChaGB,
)
from maple.function.calculator.extra_correction.implicit.private_chagb_preparation import (
    prepare_private_chagb_frozen_source,
)
from maple.function.read.filereader.mol2_reader import MOL2Reader

PROFILE = "torch-fp32-sp-v1"
PROFILE_SHA = "1" * 64
_REAL_LOAD_RELEASE_CONTRACT = private_adapter._load_release_contract


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expected_release():
    release = {
        "schema_version": 1,
        "release_id": "private-cha-torch-fp32-release-v1",
        "api_version": 4,
        "distribution": {
            "name": "maple-cha-exact-private",
            "version": "0.4.0",
            "package": "maple_cha_exact_private",
        },
        "maple_distribution": {"name": "maple"},
        "arithmetic_profile": PROFILE,
        "model_profile_sha256": PROFILE_SHA,
        "runtime": private_adapter._runtime_versions(),
        "runtime_contract": {
            "supported_device_types": ["cpu", "cuda"],
            "default_device": "cpu",
            "floating_dtype": "float32",
            "supports_forces": False,
            "failure_fallback": False,
        },
        "private_source_sha256": {"api.py": "5" * 64},
        "private_provenance_sha256": {
            "share/doc/maple-cha-exact-private/NOTICE.md": "a" * 64,
            "share/doc/maple-cha-exact-private/API.md": "b" * 64,
            "share/doc/maple-cha-exact-private/BASELINE.json": "c" * 64,
            "share/doc/maple-cha-exact-private/SOURCE_MAP.json": "d" * 64,
            "share/licenses/maple-cha-exact-private/PRIVATE-LICENSE.txt": "e" * 64,
        },
        "maple_public_source": {
            "maple_public_polar": {
                "module": "maple.function.calculator.extra_correction.implicit.torch_chagb",
                "relative_path": "maple/function/calculator/extra_correction/implicit/torch_chagb.py",
                "sha256": "6" * 64,
            },
            "maple_public_cavity": {
                "module": "maple.function.calculator.extra_correction.implicit.torch_pbsa_exact_cavity",
                "relative_path": "maple/function/calculator/extra_correction/implicit/torch_pbsa_exact_cavity.py",
                "sha256": "7" * 64,
            },
            "maple_public_dense_budget": {
                "module": "maple.function.calculator.extra_correction.implicit.torch_dense_budget",
                "relative_path": "maple/function/calculator/extra_correction/implicit/torch_dense_budget.py",
                "sha256": "8" * 64,
            },
        },
    }
    release["approved_source_aggregate_sha256"] = _sha256(
        private_adapter._canonical_json(
            private_adapter._release_aggregate_payload(release)
        )
    )
    release["content_sha256"] = _sha256(private_adapter._canonical_json(release))
    return release


class _UnitRuntimeBinding:
    def require_unchanged(self):
        return None


@pytest.fixture(autouse=True)
def expected_release(monkeypatch):
    release = _expected_release()

    def bind(engine, expected):
        identity = engine.engine_identity()
        private_adapter._validate_engine_contract(engine, identity, expected)
        return dict(identity), _UnitRuntimeBinding()

    monkeypatch.setattr(
        private_adapter, "_load_release_contract", lambda: (release, None)
    )
    monkeypatch.setattr(private_adapter, "_bind_installed_runtime", bind)
    return release


def _mol2(tmp_path: Path):
    path = tmp_path / "water.mol2"
    path.write_text(
        """@<TRIPOS>MOLECULE
water
3 2 1 0 0
SMALL
USER_CHARGES
@<TRIPOS>ATOM
1 O1 0.0000000 0.0000000 0.0000000 ow 1 MOL -0.8340000
2 H1 0.9572000 0.0000000 0.0000000 hw 1 MOL 0.4170000
3 H2 -0.2399872 0.9272970 0.0000000 hw 1 MOL 0.4170000
@<TRIPOS>BOND
1 1 2 1
2 1 3 1
@<TRIPOS>SUBSTRUCTURE
1 MOL 1
""",
        encoding="utf-8",
    )
    return MOL2Reader(str(path), charge=0, mult=1, validate_charge=True), path


def _topology(tmp_path: Path, atoms, mol2_path: Path):
    charges = atoms.get_initial_charges().tolist()
    mapping = {
        "content_sha256": "2" * 64,
        "source_mol2_sha256": _sha256(mol2_path.read_bytes()),
        "source_charges_e": charges,
        "source_charges_sha256": "3" * 64,
        "source_charge_manifest_sha256": "f" * 64,
        "effective_charges_e": [-0.8339999914, 0.4169999957, 0.4169999957],
        "effective_charges_sha256": "4" * 64,
        "prepared_prmtop_sha256": "e" * 64,
        "atom_ids": [1, 2, 3],
        "atomic_numbers": [8, 1, 1],
        "atom_names": ["O1", "H1", "H2"],
        "gaff2_types": ["ow", "hw", "hw"],
        "profile_sha256": PROFILE_SHA,
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
    return path, mapping, _sha256(path.read_bytes())


class FakePrivateEngine:
    API_VERSION = 4
    ARITHMETIC_PROFILE = PROFILE
    MODEL_PROFILE_SHA256 = PROFILE_SHA
    ExecutionPolicy = SimpleNamespace

    def __init__(self):
        self.calls = []

    def engine_identity(self):
        return {
            "distribution_name": "maple-cha-exact-private",
            "expected_distribution_version": "0.4.0",
            "installed_distribution_version": "0.4.0",
            "installed_distribution_metadata_origin": "/private/site-packages",
            "api_version": 4,
            "arithmetic_profile": PROFILE,
            "model_profile_sha256": PROFILE_SHA,
            "module_origin": "/private/site-packages/maple_cha_exact_private/__init__.py",
            "private_source_sha256": {"api.py": "5" * 64},
            "installed_provenance_sha256": {
                "share/doc/maple-cha-exact-private/NOTICE.md": {
                    "origin": "/private/share/doc/maple-cha-exact-private/NOTICE.md",
                    "sha256": "a" * 64,
                },
                "share/doc/maple-cha-exact-private/API.md": {
                    "origin": "/private/share/doc/maple-cha-exact-private/API.md",
                    "sha256": "b" * 64,
                },
                "share/doc/maple-cha-exact-private/BASELINE.json": {
                    "origin": "/private/share/doc/maple-cha-exact-private/BASELINE.json",
                    "sha256": "c" * 64,
                },
                "share/doc/maple-cha-exact-private/SOURCE_MAP.json": {
                    "origin": "/private/share/doc/maple-cha-exact-private/SOURCE_MAP.json",
                    "sha256": "d" * 64,
                },
                "share/licenses/maple-cha-exact-private/PRIVATE-LICENSE.txt": {
                    "origin": "/private/share/licenses/maple-cha-exact-private/PRIVATE-LICENSE.txt",
                    "sha256": "e" * 64,
                },
            },
            "maple_public_source": {
                "maple_public_polar": {
                    "origin": "/maple/torch_chagb.py",
                    "sha256": "6" * 64,
                },
                "maple_public_cavity": {
                    "origin": "/maple/torch_pbsa_exact_cavity.py",
                    "sha256": "7" * 64,
                },
                "maple_public_dense_budget": {
                    "origin": "/maple/torch_dense_budget.py",
                    "sha256": "8" * 64,
                },
            },
        }

    def load_topology(
        self, mapping, *, expected_content_sha256, device, execution_policy
    ):
        self.calls.append(("load", expected_content_sha256))
        assert isinstance(device, torch.device)
        assert device.type in {"cpu", "cuda"}
        if str(device) == "cuda:99":
            raise RuntimeError("CUDA device index 99 is unavailable")
        self.device = device
        assert execution_policy is not None
        assert expected_content_sha256 == mapping["content_sha256"]
        effective = np.asarray(mapping["effective_charges_e"], dtype=np.float32)
        return SimpleNamespace(
            **{
                **mapping,
                "content_sha256": "9" * 64,
                "compute_tensor_sha256": (
                    ("effective_charges_e", _sha256(effective.tobytes())),
                ),
                "maximum_float32_cast_delta": float(
                    np.max(
                        np.abs(
                            effective.astype(np.float64)
                            - np.asarray(
                                mapping["effective_charges_e"], dtype=np.float64
                            )
                        )
                    )
                ),
            }
        )

    def bind_coordinates(
        self, topology, live_positions, *, device, execution_policy
    ):
        positions = np.asarray(live_positions)
        assert np.issubdtype(positions.dtype, np.floating)
        tokens = tuple(f"{float(value):12.7f}" for value in positions.reshape(-1))
        digest = _sha256("".join(tokens).encode("ascii"))
        compute = SimpleNamespace(
            dtype=torch.float32,
            device=torch.device(device),
            requires_grad=False,
        )
        self.calls.append(("bind", digest))
        return SimpleNamespace(
            topology_sha256=topology.content_sha256,
            content_sha256=digest,
            coordinate_tokens_f12_7=tokens,
            coordinate_bytes_sha256=digest,
            positions_f32=compute,
        )

    def evaluate_exact_sp(self, topology, coordinates, *, execution_policy):
        self.calls.append(("evaluate", coordinates.content_sha256))
        return SimpleNamespace(
            polar_kcal_mol=-2.0,
            cavity_kcal_mol=1.25,
            dispersion_kcal_mol=-0.5,
            total_kcal_mol=-1.25,
            topology_sha256=topology.content_sha256,
            coordinate_sha256=coordinates.content_sha256,
            profile_sha256=PROFILE_SHA,
            electrostatic_size_angstrom=float(np.float32(9.5)),
            electrostatic_size_switch_angstrom=float(np.float32(10.0)),
            selected_size_branch="below-switch",
            device=self.device,
            platform=self.device.type.upper(),
            runtime="torch",
            dtype="torch.float32",
            workspace_bytes=2**30,
            peak_allocated_bytes=(1024 if self.device.type == "cuda" else None),
            peak_reserved_bytes=(2048 if self.device.type == "cuda" else None),
            readback_records=(),
            readback_bytes=0,
            source_sha256=(("exact_sp_chain.py", "a" * 64),),
            content_sha256="d" * 64,
            supports_forces=False,
        )


_INVALID_MOLECULAR_IDENTITIES = (
    ({"mult": 1}, "explicit 'charge multiplicity'"),
    ({"charge": 0}, "explicit 'charge multiplicity'"),
    ({"charge": None, "mult": 1}, "must be integers"),
    ({"charge": "invalid", "mult": 1}, "must be integers"),
    ({"charge": 1, "mult": 1}, "neutral closed-shell"),
    ({"charge": 0, "mult": 2}, "neutral closed-shell"),
)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, mapping, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    charge_result = ChargeResult(
        charges=atoms.get_initial_charges().copy(),
        method="mol2-fixed",
        mode="fixed",
        provenance={"source": "mol2", "path": str(mol2_path)},
    )
    instance = PrivateTorchChaGB(
        atoms,
        charge_result,
        prepared_topology=topology_path,
        prepared_topology_sha256=file_sha,
        device="cuda:0",
        dtype="float32",
    )
    return instance, atoms, engine, topology_path, mapping


def test_private_provider_defaults_to_cpu(tmp_path, monkeypatch):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    instance = PrivateTorchChaGB(
        atoms,
        ChargeResult(atoms.get_initial_charges(), "mol2-fixed"),
        prepared_topology=topology_path,
        prepared_topology_sha256=file_sha,
    )
    assert instance.provenance["device"] == "cpu"


def test_private_provider_rejects_resolved_mps_before_release_or_engine_io(monkeypatch):
    monkeypatch.setattr(
        private_adapter,
        "_load_release_contract",
        lambda: pytest.fail("release manifest read"),
    )
    with pytest.raises(ValueError, match="CPU and CUDA"):
        PrivateTorchChaGB._initialize_loaded(
            object.__new__(PrivateTorchChaGB),
            prepared_topology="unused.json",
            prepared_topology_sha256="a" * 64,
            device=torch.device("mps"),
            dtype="float32",
        )


def test_private_provider_rejects_non_fp32_before_release_or_engine_io(monkeypatch):
    monkeypatch.setattr(
        private_adapter,
        "_load_release_contract",
        lambda: pytest.fail("release manifest read"),
    )
    with pytest.raises(ValueError, match="dtype=float32"):
        PrivateTorchChaGB._initialize_loaded(
            object.__new__(PrivateTorchChaGB),
            prepared_topology="unused.json",
            prepared_topology_sha256="a" * 64,
            device="cuda:0",
            dtype="float64",
        )


def test_private_provider_propagates_invalid_cuda_index_without_fallback(
    tmp_path, monkeypatch
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    with pytest.raises(ValueError, match="index 99 is unavailable"):
        PrivateTorchChaGB(
            atoms,
            ChargeResult(atoms.get_initial_charges(), "mol2-fixed"),
            prepared_topology=topology_path,
            prepared_topology_sha256=file_sha,
            device="cuda:99",
        )
    assert engine.calls == []


def test_private_provider_returns_fp32_closed_existing_result(provider):
    instance, atoms, engine, _, mapping = provider
    result = instance.evaluate(atoms)
    expected = np.float32(-1.25) / np.float32(KCAL_PER_MOL_PER_HARTREE)
    assert result.energy_hartree == float(expected)
    conversion = np.float32(KCAL_PER_MOL_PER_HARTREE)
    cavity_h = np.float32(np.float32(1.25) / conversion)
    dispersion_h = np.float32(np.float32(-0.5) / conversion)
    nonpolar_h = np.float32(cavity_h + dispersion_h)
    reconstructed_h = np.float32(np.float32(np.float32(-2.0) / conversion) + nonpolar_h)
    assert np.float32(result.components_hartree["nonpolar"]).tobytes() == (
        nonpolar_h.tobytes()
    )
    assert (
        np.float32(result.provenance["component_reconstructed_total_hartree"]).tobytes()
        == reconstructed_h.tobytes()
    )
    assert result.provenance["serialized_kcal_mol"] == {
        "polar": -2.0,
        "cavity": 1.25,
        "dispersion": -0.5,
        "total": -1.25,
    }
    assert (
        result.provenance["unit_conversion_rounding_delta_hartree"]
        <= result.provenance["unit_conversion_rounding_bound_hartree"]
    )
    assert result.forces_hartree_per_angstrom is None
    assert instance.supported_properties == {"energy"}
    assert result.provenance["arithmetic_profile"] == PROFILE
    assert result.provenance["topology"]["content_sha256"] == "9" * 64
    assert (
        result.provenance["topology"]["source_content_sha256"]
        == mapping["content_sha256"]
    )
    assert len(result.provenance["topology"]["effective_charges_fp32_sha256"]) == 64
    assert result.provenance["topology"]["effective_charges_fp32_max_cast_delta_e"] > 0
    assert result.provenance["electrostatic_size_angstrom"] == 9.5
    assert result.provenance["electrostatic_size_switch_angstrom"] == 10.0
    assert result.provenance["selected_size_branch"] == "below-switch"
    assert result.provenance["coordinates"]["raw_positions_dtype"] == "<f8"
    assert result.provenance["device"] == "cuda:0"
    assert (
        result.provenance["coordinates"]["f12_7_max_quantization_delta_angstrom"] >= 0.0
    )
    assert (
        result.provenance["coordinates"][
            "f12_7_decimal_to_returned_float32_max_cast_delta_angstrom"
        ]
        >= 0.0
    )
    assert (
        result.provenance["coordinates"]["raw_to_returned_float32_max_delta_angstrom"]
        >= 0.0
    )
    assert (
        result.provenance["coordinates"][
            "direct_raw_to_float32_diagnostic_max_cast_delta_angstrom"
        ]
        >= 0.0
    )
    assert [call[0] for call in engine.calls] == ["load", "bind", "evaluate"]


def test_source_and_effective_charge_identities_remain_separate(provider):
    instance, _, _, _, mapping = provider
    assert mapping["source_charges_e"] != mapping["effective_charges_e"]
    assert instance.provenance["topology"]["source_charges_sha256"] == "3" * 64
    assert instance.provenance["topology"]["effective_charges_sha256"] == "4" * 64


def test_live_changed_coordinates_are_rebound_and_change_hash(provider):
    instance, atoms, _, _, _ = provider
    first = instance.evaluate(atoms)
    moved = atoms.copy()
    moved.positions[1, 0] += 0.001
    second = instance.evaluate(moved)
    assert first.provenance["coordinates"]["coordinate_sha256"] != (
        second.provenance["coordinates"]["coordinate_sha256"]
    )


def test_force_request_fails_before_binding(provider):
    instance, atoms, engine, _, _ = provider
    with pytest.raises(NotImplementedError, match="energy-only"):
        instance.evaluate(atoms, need_forces=True)
    assert [call[0] for call in engine.calls] == ["load"]


def test_stale_source_mol2_and_topology_fail_closed(provider):
    instance, atoms, engine, topology_path, _ = provider
    Path(atoms.info["mol2"]["path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="source MOL2.*changed"):
        instance.evaluate(atoms)
    assert [call[0] for call in engine.calls] == ["load"]

    atoms, mol2_path = _mol2(topology_path.parent)
    topology_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="prepared topology.*changed"):
        instance.evaluate(atoms)


def test_bound_atom_or_charge_mutation_fails(provider):
    instance, atoms, engine, _, _ = provider
    reordered = atoms[[1, 0, 2]]
    with pytest.raises(ValueError, match="atom identity|atom order"):
        instance.evaluate(reordered)
    changed_charge = atoms.copy()
    changed_charge.set_initial_charges([-0.8, 0.4, 0.4])
    with pytest.raises(ValueError, match="charge"):
        instance.evaluate(changed_charge)
    assert [call[0] for call in engine.calls] == ["load"]


def test_equal_type_equal_charge_atom_id_swap_fails_before_engine(provider):
    instance, atoms, engine, _, _ = provider
    swapped = atoms.copy()
    atom_ids = swapped.arrays["_maple_mol2_atom_id"].copy()
    atom_ids[[1, 2]] = atom_ids[[2, 1]]
    swapped.set_array("_maple_mol2_atom_id", atom_ids)
    with pytest.raises(ValueError, match="atom-ID array|atom order"):
        instance.evaluate(swapped)
    assert [call[0] for call in engine.calls] == ["load"]


def test_source_mol2_metadata_path_retarget_fails_before_engine(provider, tmp_path):
    instance, atoms, engine, _, _ = provider
    replacement = tmp_path / "replacement.mol2"
    replacement.write_bytes(Path(atoms.info["mol2"]["path"]).read_bytes())
    retargeted = atoms.copy()
    retargeted.info["mol2"] = dict(retargeted.info["mol2"])
    retargeted.info["mol2"]["path"] = str(replacement)
    with pytest.raises(ValueError, match="metadata path changed"):
        instance.evaluate(retargeted)
    assert [call[0] for call in engine.calls] == ["load"]


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("API_VERSION", 1, "API_VERSION=4"),
        ("ARITHMETIC_PROFILE", "cpu-fp64", "arithmetic profile"),
        ("MODEL_PROFILE_SHA256", "0" * 64, "model profile"),
    ],
)
def test_wrong_private_engine_identity_fails_before_topology_load(
    tmp_path, monkeypatch, attribute, value, message
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    setattr(engine, attribute, value)
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    with pytest.raises(RuntimeError, match=message):
        PrivateTorchChaGB(
            atoms,
            ChargeResult(atoms.get_initial_charges(), "mol2-fixed"),
            prepared_topology=topology_path,
            prepared_topology_sha256=file_sha,
        )
    assert engine.calls == []


def test_missing_private_package_is_actionable_and_preserves_cause(
    tmp_path, monkeypatch
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    original_import = importlib.import_module

    def missing_private(name, package=None):
        if name == "maple_cha_exact_private":
            raise ModuleNotFoundError("private engine absent")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_private)
    with pytest.raises(RuntimeError, match="separately installed") as error:
        PrivateTorchChaGB(
            atoms,
            ChargeResult(atoms.get_initial_charges(), "mol2-fixed"),
            prepared_topology=topology_path,
            prepared_topology_sha256=file_sha,
        )
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_non_fp32_engine_result_is_rejected(provider):
    instance, atoms, engine, _, _ = provider
    original = engine.evaluate_exact_sp

    def wrong_dtype(topology, coordinates, *, execution_policy):
        result = original(topology, coordinates, execution_policy=execution_policy)
        result.polar_kcal_mol = np.float64(result.polar_kcal_mol)
        return result

    engine.evaluate_exact_sp = wrong_dtype
    with pytest.raises(RuntimeError, match="Python-float serialization"):
        instance.evaluate(atoms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_bytes", None),
        ("workspace_bytes", True),
        ("workspace_bytes", "1024"),
        ("workspace_bytes", -1),
        ("peak_allocated_bytes", None),
        ("peak_allocated_bytes", False),
        ("peak_allocated_bytes", "1024"),
        ("peak_allocated_bytes", -1),
        ("peak_reserved_bytes", None),
        ("peak_reserved_bytes", True),
        ("peak_reserved_bytes", "2048"),
        ("peak_reserved_bytes", -1),
    ],
)
def test_cuda_result_rejects_malformed_runtime_telemetry(provider, field, value):
    instance, atoms, engine, _, _ = provider
    original = engine.evaluate_exact_sp

    def malformed(topology, coordinates, *, execution_policy):
        result = original(topology, coordinates, execution_policy=execution_policy)
        setattr(result, field, value)
        return result

    engine.evaluate_exact_sp = malformed
    with pytest.raises(RuntimeError, match="Private CHA"):
        instance.evaluate(atoms)


@pytest.mark.parametrize("field", ["peak_allocated_bytes", "peak_reserved_bytes"])
def test_cpu_result_rejects_nonnull_cuda_peak_telemetry(tmp_path, monkeypatch, field):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(sys.modules, "maple_cha_exact_private", engine)
    instance = PrivateTorchChaGB(
        atoms,
        ChargeResult(atoms.get_initial_charges(), "mol2-fixed"),
        prepared_topology=topology_path,
        prepared_topology_sha256=file_sha,
        device=torch.device("cpu"),
    )
    original = engine.evaluate_exact_sp

    def malformed(topology, coordinates, *, execution_policy):
        result = original(topology, coordinates, execution_policy=execution_policy)
        setattr(result, field, 0)
        return result

    engine.evaluate_exact_sp = malformed
    with pytest.raises(RuntimeError, match="null CUDA peak-memory"):
        instance.evaluate(atoms)


def test_engine_self_attestation_cannot_replace_independent_release_pin(
    expected_release,
):
    engine = FakePrivateEngine()
    engine.MODEL_PROFILE_SHA256 = "0" * 64
    original_identity = engine.engine_identity

    def self_consistent_identity():
        identity = original_identity()
        identity["model_profile_sha256"] = engine.MODEL_PROFILE_SHA256
        return identity

    engine.engine_identity = self_consistent_identity
    with pytest.raises(RuntimeError, match="independent release pin"):
        private_adapter._validate_engine_contract(
            engine, engine.engine_identity(), expected_release
        )


def test_engine_self_reported_source_drift_is_rejected_by_release_pin(
    expected_release,
):
    engine = FakePrivateEngine()
    identity = engine.engine_identity()
    identity["private_source_sha256"] = {"api.py": "9" * 64}
    with pytest.raises(RuntimeError, match="independent release pin"):
        private_adapter._validate_engine_contract(engine, identity, expected_release)


def test_engine_self_reported_notice_drift_is_rejected_by_release_pin(
    expected_release,
):
    engine = FakePrivateEngine()
    identity = engine.engine_identity()
    identity["installed_provenance_sha256"][
        "share/doc/maple-cha-exact-private/NOTICE.md"
    ]["sha256"] = ("0" * 64)
    with pytest.raises(RuntimeError, match="provenance"):
        private_adapter._validate_engine_contract(engine, identity, expected_release)


def test_release_contract_requires_independent_file_content_and_source_seals(
    expected_release,
):
    payload = private_adapter._canonical_json(expected_release)
    digest = _sha256(payload)
    assert (
        private_adapter._validate_release_contract(
            expected_release, file_sha256=digest, expected_file_sha256=digest
        )["release_id"]
        == "private-cha-torch-fp32-release-v1"
    )
    with pytest.raises(RuntimeError, match="file hash"):
        private_adapter._validate_release_contract(
            expected_release,
            file_sha256=digest,
            expected_file_sha256="0" * 64,
        )


def test_runtime_release_pin_matches_actual_packaged_manifest_and_seals():
    payload = private_adapter._RELEASE_DATA_PATH.read_bytes()
    actual_file_sha256 = _sha256(payload)
    if private_adapter._PINNED_RELEASE_FILE_SHA256 is None:
        with pytest.raises(RuntimeError, match="not sealed"):
            _REAL_LOAD_RELEASE_CONTRACT()
        return
    assert actual_file_sha256 == private_adapter._PINNED_RELEASE_FILE_SHA256
    release, bound = _REAL_LOAD_RELEASE_CONTRACT()
    assert bound.sha256 == actual_file_sha256
    without_content = dict(release)
    recorded_content = without_content.pop("content_sha256")
    assert _sha256(private_adapter._canonical_json(without_content)) == recorded_content
    assert (
        _sha256(
            private_adapter._canonical_json(
                private_adapter._release_aggregate_payload(release)
            )
        )
        == release["approved_source_aggregate_sha256"]
    )


def test_runtime_release_pin_none_remains_fail_closed(monkeypatch):
    monkeypatch.setattr(private_adapter, "_PINNED_RELEASE_FILE_SHA256", None)
    with pytest.raises(RuntimeError, match="not sealed"):
        _REAL_LOAD_RELEASE_CONTRACT()


def test_runtime_prefix_membership_accepts_dot_omx_venv_but_rejects_checkout(
    tmp_path,
):
    prefix = tmp_path / ".omx" / "venv"
    installed = prefix / "lib" / "python" / "site-packages" / "module.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("# installed\n", encoding="utf-8")
    private_adapter._require_within(installed, prefix, label="installed module")
    checkout = tmp_path / "checkout" / "module.py"
    checkout.parent.mkdir()
    checkout.write_text("# checkout\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime prefix"):
        private_adapter._require_within(checkout, prefix, label="checkout module")


def test_editable_distribution_is_not_accepted_as_installed_wheel():
    distribution = SimpleNamespace(
        read_text=lambda name: json.dumps({"dir_info": {"editable": True}})
    )
    with pytest.raises(RuntimeError, match="non-editable installed distribution"):
        private_adapter._require_noneditable_distribution(
            distribution, label="private fixture"
        )


def test_bound_source_inventory_detects_added_python_entry_without_rehash(
    tmp_path,
):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "api.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    bound = private_adapter.CheckpointFileIdentity.bind(source)
    binding = private_adapter._RuntimeSourceBinding(
        files=(bound,),
        directories={str(package): private_adapter._source_directory_snapshot(package)},
        modules={
            name: (module, private_adapter._module_token(module))
            for name, module in tuple(sys.modules.items())
            if module is not None
            and (
                name == private_adapter._ENGINE_MODULE
                or name.startswith(private_adapter._ENGINE_MODULE + ".")
            )
        },
    )
    binding.require_unchanged()
    (package / "extra.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory changed"):
        binding.require_unchanged()


def test_oversized_topology_fails_on_fstat_before_any_read(tmp_path, monkeypatch):
    path = tmp_path / "oversized.json"
    with path.open("wb") as handle:
        handle.truncate(private_adapter._MAX_PREPARED_TOPOLOGY_BYTES + 1)
    original_open = open

    class NoRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, *args, **kwargs):
            pytest.fail("oversized topology bytes were read")

    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: NoRead(original_open(*args, **kwargs)),
    )
    with pytest.raises(ValueError, match="resource limit"):
        private_adapter._read_bounded_stable_file(
            path,
            max_bytes=private_adapter._MAX_PREPARED_TOPOLOGY_BYTES,
            label="prepared_topology JSON",
        )


def test_loaded_evaluate_performs_no_file_open_or_engine_rehash(provider, monkeypatch):
    instance, atoms, engine, _, _ = provider
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: pytest.fail("file opened")
    )
    monkeypatch.setattr(
        Path, "read_bytes", lambda *args, **kwargs: pytest.fail("file read")
    )
    engine.engine_identity = lambda: pytest.fail("engine source rehashed")
    first = instance.evaluate(atoms)
    moved = atoms.copy()
    moved.positions[1, 0] += 0.001
    second = instance.evaluate(moved)
    assert first.provenance["coordinates"]["coordinate_sha256"] != (
        second.provenance["coordinates"]["coordinate_sha256"]
    )


def test_correction_wires_one_existing_composition_boundary(tmp_path, monkeypatch):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "implicit": "water",
            "method": "gb",
            "model": "chagb",
            "profile": "chagb-bondi-pbsa-inp2",
            "nonpolar": "cavity-dispersion",
            "dtype": "float32",
            "prepared_topology": str(topology_path),
            "prepared_topology_sha256": file_sha,
            "experimental": True,
        },
        output=tmp_path / "maple.out",
        model_device=torch.device("cuda:0"),
        task_context="sp",
    )
    result = correction.evaluate(atoms)
    assert np.isfinite(result.energy_hartree)
    assert correction.solvation_options["provider"] == "private-torch-cha"
    assert correction.model_device == "cuda:0"
    assert result.provenance["device"] == "cuda:0"
    assert [call[0] for call in engine.calls].count("evaluate") == 1


def test_correction_accepts_cpu_without_device_fallback(tmp_path, monkeypatch):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    output = tmp_path / "cpu-rejected.out"
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "mol2", "mode": "fixed", "geometry": "keep"},
        {
            "implicit": "water",
            "method": "gb",
            "model": "chagb",
            "profile": "chagb-bondi-pbsa-inp2",
            "nonpolar": "cavity-dispersion",
            "dtype": "float32",
            "prepared_topology": str(topology_path),
            "prepared_topology_sha256": file_sha,
            "experimental": True,
        },
        output=output,
        model_device=torch.device("cpu"),
        task_context="sp",
    )
    result = correction.evaluate(atoms)
    assert result.provenance["device"] == "cpu"
    assert engine.device == torch.device("cpu")


def test_prepared_source_selects_frozen_topology_vector_with_one_load(
    tmp_path, monkeypatch
):
    atoms, mol2_path = _mol2(tmp_path)
    embedded = atoms.get_initial_charges().copy()
    topology_path, mapping, _ = _topology(tmp_path, atoms, mol2_path)
    mapping["source_charges_e"] = [-0.8, 0.4, 0.4]
    topology_path.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
    file_sha = _sha256(topology_path.read_bytes())
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": "prepared", "mode": "fixed", "geometry": "keep"},
        {
            "implicit": "water",
            "method": "gb",
            "model": "chagb",
            "profile": "chagb-bondi-pbsa-inp2",
            "nonpolar": "cavity-dispersion",
            "dtype": "float32",
            "prepared_topology": str(topology_path),
            "prepared_topology_sha256": file_sha,
            "experimental": True,
        },
        output=tmp_path / "prepared.out",
        task_context="sp",
    )
    assert not np.array_equal(embedded, correction.charge_result.charges)
    assert np.array_equal(correction.charge_result.charges, [-0.8, 0.4, 0.4])
    assert np.array_equal(atoms.get_initial_charges(), [-0.8, 0.4, 0.4])
    provenance = correction.charge_result.provenance
    assert provenance["source"] == "prepared-topology-frozen-source-charges"
    assert provenance["generation_performed"] is False
    assert provenance["normalization_performed"] is False
    assert provenance["incoming_caller_initial_charges_role"] == (
        "metadata-only-not-selected"
    )
    assert [call[0] for call in engine.calls] == ["load"]
    correction.evaluate(atoms)
    assert [call[0] for call in engine.calls] == ["load", "bind", "evaluate"]


def test_real_frozen_methane_mismatch_selects_topology_source_without_energy_read(
    monkeypatch,
):
    worktree = Path(__file__).resolve().parents[2]
    original_root = (
        worktree if (worktree / ".omx/benchmarks").is_dir() else worktree.parents[2]
    )
    topology_path = (
        original_root
        / ".omx/benchmarks/route1-torch-amber-exact-sp/exact_inputs_v1"
        / "mobley_9055303/topology.json"
    )
    mol2_path = (
        original_root
        / ".omx/benchmarks/neutral-water-freesolv-route1-20260723/dataset"
        / "mol2files_gaff/mobley_9055303.mol2"
    )
    mapping = json.loads(topology_path.read_text(encoding="utf-8"))
    atoms = MOL2Reader(str(mol2_path), charge=0, mult=1, validate_charge=True)
    embedded = atoms.get_initial_charges().copy()
    source = np.asarray(mapping["source_charges_e"], dtype=np.float64)
    assert not np.array_equal(embedded, source)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    provider, result = prepare_private_chagb_frozen_source(
        atoms,
        prepared_topology=topology_path,
        prepared_topology_sha256=_sha256(topology_path.read_bytes()),
    )
    assert isinstance(provider, PrivateTorchChaGB)
    assert not hasattr(private_adapter, "PrivateChaPreparedState")
    assert not hasattr(provider, "export_prepared_state")
    assert not hasattr(provider, "from_prepared_state")
    assert provider._source_charges_sha256 == mapping["source_charges_sha256"]
    assert np.array_equal(result.charges, source)
    assert result.provenance["generation_performed"] is False
    assert result.provenance["normalization_performed"] is False
    assert [call[0] for call in engine.calls] == ["load"]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"mult": 1}, "explicit 'charge multiplicity'"),
        ({"charge": 0}, "explicit 'charge multiplicity'"),
        ({"charge": 1, "mult": 1}, "neutral closed-shell"),
        ({"charge": 0, "mult": 2}, "neutral closed-shell"),
    ],
)
def test_prepared_source_requires_neutral_closed_shell_before_topology_load(
    tmp_path, monkeypatch, metadata, message
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, _ = _topology(tmp_path, atoms, mol2_path)
    atoms.info.pop("charge", None)
    atoms.info.pop("mult", None)
    atoms.info.update(metadata)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)

    with pytest.raises(ValueError, match=message):
        prepare_private_chagb_frozen_source(
            atoms,
            prepared_topology=topology_path,
            prepared_topology_sha256=_sha256(topology_path.read_bytes()),
        )

    assert engine.calls == []


@pytest.mark.parametrize(
    ("metadata", "message"),
    _INVALID_MOLECULAR_IDENTITIES,
)
def test_direct_provider_requires_domain_identity_before_topology_load(
    tmp_path, monkeypatch, metadata, message
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, _ = _topology(tmp_path, atoms, mol2_path)
    atoms.info.pop("charge", None)
    atoms.info.pop("mult", None)
    atoms.info.update(metadata)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)

    error = None
    try:
        PrivateTorchChaGB(
            atoms,
            ChargeResult(atoms.get_initial_charges(), "mol2-fixed"),
            prepared_topology=topology_path,
            prepared_topology_sha256=_sha256(topology_path.read_bytes()),
        )
    except ValueError as exc:
        error = str(exc)

    assert (error is not None and message in error, engine.calls) == (True, [])


@pytest.mark.parametrize(
    ("metadata", "message"),
    _INVALID_MOLECULAR_IDENTITIES,
)
def test_direct_provider_revalidates_live_domain_before_engine(
    provider, metadata, message
):
    instance, atoms, engine, _, _ = provider
    atoms.info.pop("charge", None)
    atoms.info.pop("mult", None)
    atoms.info.update(metadata)

    error = None
    try:
        instance.evaluate(atoms)
    except ValueError as exc:
        error = str(exc)

    engine_calls = [call[0] for call in engine.calls]
    assert (
        error is not None and message in error,
        engine_calls.count("bind"),
        engine_calls.count("evaluate"),
    ) == (True, 0, 0)


@pytest.mark.parametrize(
    ("metadata", "message"),
    _INVALID_MOLECULAR_IDENTITIES,
)
@pytest.mark.parametrize("source", ["mol2", "prepared"])
def test_composition_revalidates_live_domain_before_provider(
    tmp_path, monkeypatch, source, metadata, message
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, file_sha = _topology(tmp_path, atoms, mol2_path)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    correction = ImplicitSolvationCorrection(
        atoms,
        {"source": source, "mode": "fixed", "geometry": "keep"},
        {
            "implicit": "water",
            "method": "gb",
            "model": "chagb",
            "profile": "chagb-bondi-pbsa-inp2",
            "nonpolar": "cavity-dispersion",
            "dtype": "float32",
            "prepared_topology": str(topology_path),
            "prepared_topology_sha256": file_sha,
            "experimental": True,
        },
        output=tmp_path / f"live-identity-{source}.out",
        task_context="sp",
    )
    atoms.info.pop("charge", None)
    atoms.info.pop("mult", None)
    atoms.info.update(metadata)

    error = None
    try:
        correction.evaluate(atoms)
    except ValueError as exc:
        error = str(exc)

    engine_calls = [call[0] for call in engine.calls]
    assert (
        error is not None and message in error,
        engine_calls.count("bind"),
        engine_calls.count("evaluate"),
    ) == (True, 0, 0)


def test_prepared_source_labels_preexisting_charges_as_caller_state(
    tmp_path, monkeypatch
):
    atoms, mol2_path = _mol2(tmp_path)
    topology_path, _, _ = _topology(tmp_path, atoms, mol2_path)
    incoming = np.array([-0.6, 0.3, 0.3])
    atoms.set_initial_charges(incoming)
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)

    _, result = prepare_private_chagb_frozen_source(
        atoms,
        prepared_topology=topology_path,
        prepared_topology_sha256=_sha256(topology_path.read_bytes()),
    )

    provenance = result.provenance
    assert provenance["incoming_caller_initial_charges_role"] == (
        "metadata-only-not-selected"
    )
    assert provenance["incoming_caller_initial_charges_binary64_sha256"] == (
        _sha256(np.ascontiguousarray(incoming).tobytes())
    )
    assert not any("embedded" in key for key in provenance)


def test_prepared_source_identity_failure_does_not_mutate_atoms_charges(
    tmp_path, monkeypatch
):
    atoms, mol2_path = _mol2(tmp_path)
    original = atoms.get_initial_charges().copy()
    topology_path, mapping, _ = _topology(tmp_path, atoms, mol2_path)
    mapping["atom_ids"] = [1, 3, 2]
    mapping["source_charges_e"] = [-0.8, 0.4, 0.4]
    topology_path.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    with pytest.raises(ValueError, match="atom IDs"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "prepared", "mode": "fixed", "geometry": "keep"},
            {
                "implicit": "water",
                "method": "gb",
                "model": "chagb",
                "prepared_topology": str(topology_path),
                "prepared_topology_sha256": _sha256(topology_path.read_bytes()),
                "experimental": True,
            },
            output=tmp_path / "failure.out",
            task_context="sp",
        )
    assert np.array_equal(atoms.get_initial_charges(), original)
    assert [call[0] for call in engine.calls] == ["load"]


def test_mol2_source_does_not_fallback_to_prepared_source_vector(tmp_path, monkeypatch):
    atoms, mol2_path = _mol2(tmp_path)
    original_info = json.loads(json.dumps(atoms.info))
    original_arrays = {key: value.copy() for key, value in atoms.arrays.items()}
    original_positions = atoms.get_positions().copy()
    original_charges = atoms.get_initial_charges().copy()
    topology_path, mapping, _ = _topology(tmp_path, atoms, mol2_path)
    mapping["source_charges_e"] = [-0.8, 0.4, 0.4]
    topology_path.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
    engine = FakePrivateEngine()
    monkeypatch.setitem(__import__("sys").modules, "maple_cha_exact_private", engine)
    with pytest.raises(ValueError, match="exactly match.*source charges"):
        ImplicitSolvationCorrection(
            atoms,
            {"source": "mol2", "mode": "fixed", "geometry": "keep"},
            {
                "implicit": "water",
                "method": "gb",
                "model": "chagb",
                "prepared_topology": str(topology_path),
                "prepared_topology_sha256": _sha256(topology_path.read_bytes()),
                "experimental": True,
            },
            output=tmp_path / "strict-mol2.out",
            task_context="sp",
        )
    assert atoms.info == original_info
    assert set(atoms.arrays) == set(original_arrays)
    for key, value in original_arrays.items():
        assert np.array_equal(atoms.arrays[key], value)
    assert np.array_equal(atoms.get_positions(), original_positions)
    assert np.array_equal(atoms.get_initial_charges(), original_charges)
