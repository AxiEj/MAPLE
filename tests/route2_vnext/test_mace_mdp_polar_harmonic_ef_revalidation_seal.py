from __future__ import annotations

import errno
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from maple.solvation.release.admission import ComputationSealV2
from maple.solvation.release.evidence import canonical_json_sha256
import tools.route2_release.seal_mace_mdp_polar_harmonic_ef_revalidation as sealer

REAL_LOADED_MODULE_CHECK = sealer._assert_loaded_repo_modules_bound
REAL_FRESH_PROCESS_CHECK = sealer._require_fresh_execution_process

REAL_PREREG = Path(__file__).parents[2] / (
    "docs/route2/preregistrations/"
    "mace-mdp-polar-hybrid-harmonic-ef-revalidation-v2.json"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _runtime() -> dict[str, object]:
    return {
        "python": "3.11.13",
        "implementation": "CPython",
        "platform": "Linux-test",
        "machine": "x86_64",
        "cpu_model": "Test CPU",
        "packages": {
            "maple": None,
            "ase": "3.26.0",
            "numpy": "2.1.0",
            "scipy": "1.15.0",
            "torch": "2.7.0",
            "mace-torch": "0.3.14",
            "graph-longrange": "0.1.0",
            "pyscf": None,
            "pyddx": "0.6.0",
        },
        "numpy": {"version": "2.1.0", "show_config": "blas=openblas"},
        "torch": {
            "version": "2.7.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
            "cuda_available": True,
            "devices": ["Test GPU"],
            "default_dtype": "torch.float64",
            "threads": 1,
            "interop_threads": 1,
            "driver_version": "580.95.05",
            "device_uuids": ["GPU-11111111-1111-4111-8111-111111111111"],
            "device_capabilities": ["8.9"],
            "device_multiprocessor_counts": [128],
            "deterministic_algorithms_enabled": True,
            "deterministic_debug_mode": 2,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_LAUNCH_BLOCKING": None,
            "PYTHONHASHSEED": "0",
        },
        "execution_device": "cuda",
        "execution_dtype": "float64",
    }


def test_cavity_manifest_binds_canonical_h1_input_and_separates_legacy_projection() -> (
    None
):
    legacy_projection_sha256 = "a" * 64
    mol2_sha256 = "b" * 64
    manifest = sealer._cavity_system_manifest(
        system_id="benzene",
        atomic_numbers=(6, 6),
        chemical_symbols=("C", "C"),
        positions_angstrom=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
        radii_angstrom=(1.85, 1.85),
        radii_provider_id="smd-water-coulomb-radii-v1",
        source_asset={"role": "benzene_mol2", "sha256": mol2_sha256},
        legacy_projection_artifact_sha256=legacy_projection_sha256,
        legacy_projection_radii_angstrom=(1.849999998754871,) * 2,
        legacy_projection_maximum_absolute_difference_angstrom=1.25e-9,
    )

    assert manifest["input_sha256"] == sealer._canonical_hash(
        manifest["canonical_input"]
    )
    assert manifest["input_sha256"] != legacy_projection_sha256
    assert manifest["canonical_input"]["source_asset"] == {
        "role": "benzene_mol2",
        "sha256": mol2_sha256,
    }
    assert manifest["legacy_projection_artifact_sha256"] == legacy_projection_sha256
    assert (
        manifest["legacy_projection_role"] == "provenance-only-not-the-h1-cavity-input"
    )


def test_cavity_implementation_ledger_includes_canonical_radii_provider() -> None:
    source_ledger = {
        "maple/solvation/continuum/harmonic.py": "1" * 64,
        sealer.SMD_CDS_SOURCE_ID: "2" * 64,
        "maple/unrelated.py": "3" * 64,
    }
    selected = sealer._relevant_source_ledger(
        source_ledger, sealer.CAVITY_IMPLEMENTATION_SOURCE_PREFIXES
    )
    assert selected == {
        "maple/solvation/continuum/harmonic.py": "1" * 64,
        sealer.SMD_CDS_SOURCE_ID: "2" * 64,
    }


def test_seal_canonical_hash_matches_shared_strict_json_domain() -> None:
    value = {"unicode": "Å", "nested": [1, 2.5, {"ok": True}]}
    assert sealer._canonical_hash(value) == canonical_json_sha256(value)
    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            sealer._canonical_hash({"value": nonfinite})


def test_seal_provider_factory_uses_same_captured_bytes_lane_as_runner() -> None:
    source = inspect.getsource(sealer._provider_configuration_factory)
    assert "checkpoint_bytes=mace_mdp_checkpoint_bytes" in source
    assert "checkpoint_bytes=mace_polar_checkpoint_bytes" in source
    assert "checkpoint_path=" not in source


class Fixture:
    def __init__(self, root: Path, external: Path) -> None:
        self.root = root
        self.external = external
        self.prereg = root / "docs/prereg-v2.json"
        self.runner = root / sealer.RUNNER_ID
        self.aggregator = root / sealer.AGGREGATOR_ID
        self.mdp = external / "mdp.model"
        self.polar = external / "polar.model"
        self.panel = root / "docs/panel.json"
        self.v1_prereg = root / "docs/v1-prereg.json"
        self.asset_root = external / "assets"
        self.mol2 = self.asset_root / "inputs/benzene.mol2"
        self.projection = (
            self.asset_root
            / sealer.PANEL_RELATIVE_ROOT
            / "mobley_3053621"
            / sealer.CUTOFF_DIRECTORY
            / "result.json"
        )
        self.output = external / "seal.json"

    def build_args(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.prereg,
            self.runner,
            self.aggregator,
            self.mdp,
            self.polar,
            self.panel,
            self.asset_root,
        )


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    root, external = tmp_path / "repo", tmp_path / "external"
    (root / "docs").mkdir(parents=True)
    (root / "tools/route2_release").mkdir(parents=True)
    for relative in (
        sealer.SEAL_GENERATOR_ID,
        sealer.RUNNER_ID,
        sealer.AGGREGATOR_ID,
        "maple/solvation/release/admission.py",
        "maple/solvation/release/evidence.py",
        "maple/solvation/api/capabilities.py",
        "maple/solvation/continuum/harmonic_exposure.py",
        "maple/solvation/experimental/mace_mdp_polar_harmonic.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            Path(sealer.__file__).read_bytes()
            if relative == sealer.SEAL_GENERATOR_ID
            else f"# synthetic {relative}\n".encode()
        )
    external.mkdir()
    asset_root = external / "assets"
    (asset_root / "inputs").mkdir(parents=True)
    projection = (
        asset_root
        / sealer.PANEL_RELATIVE_ROOT
        / "mobley_3053621"
        / sealer.CUTOFF_DIRECTORY
        / "result.json"
    )
    projection.parent.mkdir(parents=True)
    mdp_bytes, polar_bytes = b"mdp", b"polar"
    mol2_bytes = b"synthetic captured benzene mol2\n"
    v1_bytes = b'{"artifact_id":"synthetic-v1"}\n'
    panel_payload = {
        "records": [
            {
                "compound_id": "mobley_3053621",
                "mol2_path": "inputs/benzene.mol2",
                "mol2_sha256": _sha(mol2_bytes),
            }
        ]
    }
    panel_bytes = (json.dumps(panel_payload, sort_keys=True) + "\n").encode()
    mdp, polar = external / "mdp.model", external / "polar.model"
    panel = root / "docs/panel.json"
    mdp.write_bytes(mdp_bytes)
    polar.write_bytes(polar_bytes)
    panel.write_bytes(panel_bytes)
    (asset_root / "inputs/benzene.mol2").write_bytes(mol2_bytes)
    projection.write_text(
        json.dumps(
            {
                "inputs": {
                    "parsed_pcm_input": {
                        "cavity_radii_angstrom": [1.85] * 6 + [1.20] * 6
                    }
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    v1_prereg = root / "docs/v1-prereg.json"
    v1_prereg.write_bytes(v1_bytes)
    payload = json.loads(REAL_PREREG.read_text(encoding="utf-8"))
    real_root = Path(__file__).parents[2]
    audit_binding = payload["verified_pro_schema_amendment"]["verified_pro_audit"]
    audit_relative = audit_binding["relative_path"]
    audit = json.loads((real_root / audit_relative).read_text(encoding="utf-8"))
    review_relatives = [
        audit_relative,
        audit["prompt"]["relative_path"],
        audit["response"]["relative_path"],
        audit["mode_verification"]["relative_path"],
    ]
    for relative in review_relatives:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((real_root / relative).read_bytes())
    payload["parent_bindings"]["v1_parent_panel"].update(
        {
            "relative_path": "docs/panel.json",
            "sha256": _sha(panel_bytes),
            "record_order": ["mobley_3053621"],
        }
    )
    payload["parent_bindings"]["v1_preregistration"].update(
        {"relative_path": "docs/v1-prereg.json", "sha256": _sha(v1_bytes)}
    )
    payload["inherited_v1_contract"]["frozen_runtime_contract"].update(
        {
            "mace_mdp_checkpoint_sha256": _sha(mdp_bytes),
            "mace_polar_checkpoint_sha256": _sha(polar_bytes),
        }
    )
    prereg = root / "docs/prereg-v2.json"
    prereg.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", root), check=True)
    subprocess.run(
        ("git", "-C", root, "config", "user.email", "test@example.test"), check=True
    )
    subprocess.run(("git", "-C", root, "config", "user.name", "Seal Test"), check=True)
    subprocess.run(("git", "-C", root, "add", "."), check=True)
    subprocess.run(("git", "-C", root, "commit", "-qm", "frozen"), check=True)
    monkeypatch.setattr(sealer, "_running_repository_root", lambda: root.resolve())
    monkeypatch.setattr(sealer, "_require_fresh_execution_process", lambda: None)
    monkeypatch.setattr(sealer, "_assert_loaded_repo_modules_bound", lambda *args: None)
    monkeypatch.setattr(sealer, "_configure_and_capture_runtime", _runtime)

    def provider_configuration(*args: object) -> dict[str, str]:
        force_panel = args[6]
        assert isinstance(force_panel, dict)
        return {
            "hybrid_configuration": "1" * 64,
            "hybrid_provenance": "2" * 64,
            "continuum_configuration": "3" * 64,
            "continuum_provenance": "4" * 64,
            "cavity_configuration": "5" * 64,
            "prepared_input_manifest": "6" * 64,
            "force_panel_contract": sealer._canonical_hash(force_panel),
            "benzene_pes_configuration": "7" * 64,
            "water_pes_configuration": "8" * 64,
        }

    monkeypatch.setattr(
        sealer,
        "_provider_configuration_factory",
        provider_configuration,
    )
    return Fixture(root, external)


def test_builds_exact_typed_seal_and_binds_every_tracked_python(
    fixture: Fixture,
) -> None:
    before = subprocess.check_output(
        ("git", "-C", fixture.root, "status", "--porcelain")
    )
    payload = sealer.build_computation_seal(*fixture.build_args())
    seal = ComputationSealV2.from_mapping(payload)
    tracked_python = {
        value
        for value in subprocess.check_output(
            ("git", "-C", fixture.root, "ls-tree", "-r", "--name-only", "HEAD"),
            text=True,
        ).splitlines()
        if value.endswith(".py")
    }
    assert set(dict(seal.computation_source_sha256s)) == tracked_python
    assert seal.runner_id == sealer.RUNNER_ID
    assert seal.aggregator_id == sealer.AGGREGATOR_ID
    assert set(dict(seal.asset_sha256s)) == {
        "benzene_mol2",
        "benzene_projection_result",
        "parent_panel",
        "v1_preregistration",
        "verified_pro_audit",
        "verified_pro_prompt",
        "verified_pro_response",
        "verified_pro_mode_verification",
    }
    provider_hashes = dict(seal.provider_configuration_sha256s)
    assert set(provider_hashes) == {
        "hybrid_configuration",
        "hybrid_provenance",
        "continuum_configuration",
        "continuum_provenance",
        "cavity_configuration",
        "prepared_input_manifest",
        "force_panel_contract",
        "benzene_pes_configuration",
        "water_pes_configuration",
    }
    preregistration = json.loads(fixture.prereg.read_text(encoding="utf-8"))
    assert provider_hashes["force_panel_contract"] == sealer._canonical_hash(
        preregistration["inherited_v1_contract"]["force_panel"]
    )
    assert (
        subprocess.check_output(("git", "-C", fixture.root, "status", "--porcelain"))
        == before
    )


def test_verified_pro_artifacts_are_root_relative_not_cwd_relative(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(fixture.external)
    seal = ComputationSealV2.from_mapping(
        sealer.build_computation_seal(*fixture.build_args())
    )
    assert dict(seal.asset_sha256s)["verified_pro_response"] == (
        "252323ec9b93fbefd6c77e9181addb5e2269ca73031497217f5c39d60ae6fbb2"
    )


@pytest.mark.parametrize("path", ("../outside.json", "/tmp/outside.json"))
def test_verified_pro_review_paths_reject_escape_before_capture(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def unexpected_capture(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("capture must not run")

    monkeypatch.setattr(sealer, "_committed_capture", unexpected_capture)
    with pytest.raises(sealer.SealInputError, match="canonical repository-relative"):
        sealer._reviewed_relative(path, role="verified-Pro audit")
    assert called is False


def _prepared_manifest() -> dict[str, object]:
    force_panel = json.loads(REAL_PREREG.read_text(encoding="utf-8"))[
        "inherited_v1_contract"
    ]["force_panel"]
    return sealer._prepared_input_manifest(
        compound_id="mobley_3053621",
        name="benzene",
        benzene_atomic_numbers=(6,) * 12,
        benzene_positions_angstrom=tuple(
            (float(index), 0.0, 0.0) for index in range(12)
        ),
        benzene_radii_angstrom=(1.7,) * 12,
        benzene_mol2_sha256="a" * 64,
        benzene_projection_sha256="b" * 64,
        predecessor_dof=(0, 0),
        water_positions_angstrom=(
            (0.0, 0.0, 0.0),
            (0.9572, 0.0, 0.0),
            (-0.239, 0.9266, 0.0),
        ),
        water_radii_angstrom=(1.5, 1.2, 1.2),
        v1_preregistration_sha256="c" * 64,
        parent_panel_sha256="d" * 64,
        force_panel_contract=force_panel,
        force_panel_contract_sha256=sealer._canonical_hash(force_panel),
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("benzene", "atomic_numbers", 0), 7),
        (("benzene", "positions_angstrom", 0, 0), 99.0),
        (("benzene", "cavity_radii_angstrom", 0), 99.0),
        (("water", "positions_angstrom", 1, 0), 99.0),
        (("water", "cavity_radii_angstrom", 0), 99.0),
        (("force_panel_contract", "water_geometry_angstrom", 1, 0), 99.0),
        (
            ("force_panel_contract", "maximum_closed_loop_work_abs_ev"),
            99.0,
        ),
        (("force_panel_contract_sha256",), "f" * 64),
    ),
)
def test_prepared_manifest_digest_changes_for_scientific_or_contract_tamper(
    path: tuple[object, ...], replacement: object
) -> None:
    baseline = _prepared_manifest()
    tampered = json.loads(json.dumps(baseline))
    cursor = tampered
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    assert sealer._canonical_hash(tampered) != sealer._canonical_hash(baseline)


def test_secure_external_publication_is_no_replace(fixture: Fixture) -> None:
    payload = sealer.seal_revalidation(*fixture.build_args(), fixture.output)
    assert json.loads(fixture.output.read_text()) == payload
    original = fixture.output.read_bytes()
    with pytest.raises(sealer.SealInputError, match="already exists"):
        sealer.seal_revalidation(*fixture.build_args(), fixture.output)
    assert fixture.output.read_bytes() == original


@pytest.mark.parametrize("role", ["runner", "aggregator", "panel"])
def test_rejects_dirty_or_mismatched_committed_inputs(
    fixture: Fixture, role: str
) -> None:
    getattr(fixture, role).write_text("changed", encoding="utf-8")
    with pytest.raises(sealer.SealInputError, match="clean Git worktree"):
        sealer.build_computation_seal(*fixture.build_args())


def test_rejects_cross_root_execution(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sealer, "_running_repository_root", lambda: fixture.external)
    with pytest.raises(sealer.SealInputError, match="does not originate"):
        sealer.build_computation_seal(*fixture.build_args())


def test_rejects_checkpoint_and_parent_asset_mismatch(fixture: Fixture) -> None:
    fixture.mdp.write_bytes(b"wrong")
    with pytest.raises(sealer.SealInputError, match="checkpoint bytes"):
        sealer.build_computation_seal(*fixture.build_args())


def test_rejects_provider_reported_checkpoint_mismatch(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mismatch(*args: object) -> dict[str, str]:
        raise sealer.SealInputError(
            "MACE-POLAR provider reports a different checkpoint"
        )

    monkeypatch.setattr(sealer, "_provider_configuration_factory", mismatch)
    with pytest.raises(sealer.SealInputError, match="provider reports"):
        sealer.build_computation_seal(*fixture.build_args())


def test_rejects_checkpoint_path_swap_during_provider_construction(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = {
        "hybrid_configuration": "1" * 64,
        "hybrid_provenance": "2" * 64,
        "continuum_configuration": "3" * 64,
        "continuum_provenance": "4" * 64,
        "cavity_configuration": "5" * 64,
        "prepared_input_manifest": "6" * 64,
        "force_panel_contract": "7" * 64,
        "benzene_pes_configuration": "8" * 64,
        "water_pes_configuration": "9" * 64,
    }

    def swap(*args: object) -> dict[str, str]:
        replacement = fixture.external / "replacement.model"
        replacement.write_bytes(fixture.mdp.read_bytes())
        os.replace(replacement, fixture.mdp)
        return result

    monkeypatch.setattr(sealer, "_provider_configuration_factory", swap)
    with pytest.raises(sealer.SealInputError, match="checkpoint changed"):
        sealer.build_computation_seal(*fixture.build_args())


def test_rejects_input_alias(fixture: Fixture) -> None:
    fixture.polar.unlink()
    os.link(fixture.mdp, fixture.polar)
    with pytest.raises(sealer.SealInputError, match="hard-linked"):
        sealer.build_computation_seal(*fixture.build_args())


def test_rejects_output_race_without_overwrite(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def race() -> None:
        fixture.output.write_text("racer", encoding="utf-8")

    monkeypatch.setattr(sealer, "_before_publish", race)
    with pytest.raises(sealer.SealInputError, match="appeared"):
        sealer.seal_revalidation(*fixture.build_args(), fixture.output)
    assert fixture.output.read_text(encoding="utf-8") == "racer"


def test_rejects_output_parent_rename_and_symlink_retarget(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = fixture.output.parent
    moved = parent.with_name("external-moved")

    def retarget_parent() -> None:
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)

    monkeypatch.setattr(sealer, "_before_publish", retarget_parent)
    with pytest.raises(sealer.SealInputError, match="symbolic|changed"):
        sealer.seal_revalidation(*fixture.build_args(), fixture.output)
    assert not (moved / fixture.output.name).exists()


@pytest.mark.parametrize("target", ["runner", "mdp", "mol2"])
def test_prepublish_stability_closes_build_to_link_race(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    def mutate() -> None:
        getattr(fixture, target).write_text("mutated before link", encoding="utf-8")

    monkeypatch.setattr(sealer, "_before_publish", mutate)
    with pytest.raises(sealer.SealInputError, match="repository changed|changed while"):
        sealer.seal_revalidation(*fixture.build_args(), fixture.output)
    assert not fixture.output.exists()


def test_forced_procfs_fallback_publishes_verified_inode(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = sealer._call_linkat
    calls: list[int] = []

    def force_fallback(
        oldfd: int, old: bytes, newfd: int, new: bytes, flags: int
    ) -> int | None:
        calls.append(flags)
        if len(calls) == 1:
            return errno.EPERM
        return original(oldfd, old, newfd, new, flags)

    monkeypatch.setattr(sealer, "_call_linkat", force_fallback)
    payload = sealer.seal_revalidation(*fixture.build_args(), fixture.output)
    assert json.loads(fixture.output.read_text()) == payload
    assert calls == [0x1000, 0x400]


def test_fresh_process_rejects_preloaded_scientific_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "torch.synthetic_preload", ModuleType("torch.synthetic_preload")
    )
    with pytest.raises(sealer.SealInputError, match="preloaded"):
        sealer._require_fresh_execution_process()


def test_build_invokes_fresh_process_gate(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sealer, "_require_fresh_execution_process", REAL_FRESH_PROCESS_CHECK
    )
    with pytest.raises(sealer.SealInputError, match="preloaded"):
        sealer.build_computation_seal(*fixture.build_args())


def test_loaded_repo_module_binding_rejects_external_module(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_module = fixture.external / "external_provider.py"
    external_module.write_text("VALUE = 1\n", encoding="utf-8")
    module = ModuleType("maple.synthetic_external_provider")
    module.__file__ = str(external_module)
    module.__spec__ = importlib.util.spec_from_file_location(
        module.__name__, external_module
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(sealer.SealInputError, match="inside the repository"):
        REAL_LOADED_MODULE_CHECK(fixture.root, "0" * 40, {})


def test_namespace_package_binding_accepts_only_exact_source_bearing_repo_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    package = root / "maple/synthetic_namespace"
    package.mkdir(parents=True)
    child = package / "child.py"
    child.write_text("VALUE = 1\n", encoding="utf-8")
    module = ModuleType("maple.synthetic_namespace")
    module.__path__ = [str(package)]
    module.__spec__ = SimpleNamespace(origin=None)
    sealer._assert_namespace_package_bound(
        root,
        module.__name__,
        module,
        {"maple/synthetic_namespace/child.py": hashlib.sha256(child.read_bytes()).hexdigest()},
    )

    external = tmp_path / "external"
    external.mkdir()
    module.__path__ = [str(package), str(external)]
    with pytest.raises(sealer.SealInputError, match="escapes"):
        sealer._assert_namespace_package_bound(
            root,
            module.__name__,
            module,
            {"maple/synthetic_namespace/child.py": "a" * 64},
        )


def test_namespace_package_binding_rejects_missing_tracked_children(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    package = root / "maple/empty_namespace"
    package.mkdir(parents=True)
    module = ModuleType("maple.empty_namespace")
    module.__path__ = [str(package)]
    module.__spec__ = SimpleNamespace(origin=None)
    with pytest.raises(sealer.SealInputError, match="tracked Python children"):
        sealer._assert_namespace_package_bound(root, module.__name__, module, {})


@pytest.mark.parametrize("external_cache", [False, True])
def test_loaded_repo_module_binding_rejects_existing_bytecode_cache(
    fixture: Fixture, external_cache: bool
) -> None:
    source = fixture.root / "maple/synthetic_cached.py"
    cache_root = fixture.external if external_cache else fixture.root / "maple"
    cache = cache_root / "__pycache__/synthetic_cached.cpython-311.pyc"
    source.parent.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    cache.write_bytes(b"unsealed-bytecode")
    module = ModuleType("maple.synthetic_cached")
    module.__file__ = str(source)
    module.__cached__ = str(cache)

    with pytest.raises(sealer.SealInputError, match="bytecode is not admissible"):
        sealer._assert_no_unsealed_bytecode(module, module_name=module.__name__)


def test_fresh_process_rejects_external_pycache_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/tmp/external-python-cache")
    with pytest.raises(sealer.SealInputError, match="PYTHONPYCACHEPREFIX"):
        sealer._require_fresh_execution_process()


def test_real_prepared_preregistration_has_current_exact_contract_shape() -> None:
    capture = sealer._capture(REAL_PREREG, role="real preregistration")
    payload = sealer._read_json(capture, role="real preregistration")
    science, panel_sha, inherited_sha = sealer._validate_preregistration(payload)
    assert science == payload["prospective_h1_scientific_settings"]
    assert len(panel_sha) == len(inherited_sha) == 64


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("OMP_NUM_THREADS", "2", "OMP_NUM_THREADS"),
        ("CUDA_VISIBLE_DEVICES", "1", "CUDA_VISIBLE_DEVICES"),
        ("CUBLAS_WORKSPACE_CONFIG", ":bad", "CUBLAS_WORKSPACE_CONFIG"),
        ("PYTHONHASHSEED", "random", "PYTHONHASHSEED"),
    ],
)
def test_runtime_environment_fails_before_torch_import(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str, message: str
) -> None:
    for name, expected in sealer.EXPECTED_ENVIRONMENT.items():
        monkeypatch.setenv(name, expected)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv(key, value)
    with pytest.raises(sealer.SealInputError, match=message):
        sealer._configure_and_capture_runtime()


def test_gpu_identity_matches_nvidia_smi_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uuid = "GPU-11111111-1111-4111-8111-111111111111"
    monkeypatch.setattr(
        sealer.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, f"{uuid}, 580.95.05, Test GPU, 8.9\n", ""
        ),
    )
    cuda = SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(
            uuid=uuid, multi_processor_count=128
        ),
        get_device_name=lambda index: "Test GPU",
        get_device_capability=lambda index: (8, 9),
    )
    identity = sealer._selected_gpu_identity(SimpleNamespace(cuda=cuda))
    assert identity == {
        "driver_version": "580.95.05",
        "device_uuid": uuid,
        "device_capability": "8.9",
        "device_multiprocessor_count": 128,
    }


def test_gpu_identity_rejects_nvidia_smi_torch_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uuid = "GPU-11111111-1111-4111-8111-111111111111"
    monkeypatch.setattr(
        sealer.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, f"{uuid}, 580.95.05, Other GPU, 8.9\n", ""
        ),
    )
    cuda = SimpleNamespace(
        get_device_properties=lambda index: SimpleNamespace(
            uuid=uuid, multi_processor_count=128
        ),
        get_device_name=lambda index: "Test GPU",
        get_device_capability=lambda index: (8, 9),
    )
    with pytest.raises(sealer.SealInputError, match="differs from Torch"):
        sealer._selected_gpu_identity(SimpleNamespace(cuda=cuda))


def test_protocol_science_tamper_is_rejected_by_exact_seal_contract(
    fixture: Fixture,
) -> None:
    payload = json.loads(fixture.prereg.read_text())
    payload["prospective_h1_scientific_settings"]["force_stencil"][
        "fine_step_angstrom"
    ] = 0.001
    fixture.prereg.write_text(json.dumps(payload, sort_keys=True) + "\n")
    subprocess.run(("git", "-C", fixture.root, "add", "."), check=True)
    subprocess.run(("git", "-C", fixture.root, "commit", "-qm", "tamper"), check=True)
    with pytest.raises(ValueError, match="steps|scientific_settings"):
        sealer.build_computation_seal(*fixture.build_args())
