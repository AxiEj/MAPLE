from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import py_compile
import subprocess

import pytest

import tools.route2_release.run_mace_mdp_polar_harmonic_ef_revalidation as runner


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "input.json").write_text('{"value":1}\n', encoding="utf-8")
    subprocess.run(("git", "init", "-q", root), check=True)
    subprocess.run(
        ("git", "-C", root, "config", "user.email", "test@example.test"), check=True
    )
    subprocess.run(
        ("git", "-C", root, "config", "user.name", "Runner Test"), check=True
    )
    subprocess.run(("git", "-C", root, "add", "."), check=True)
    subprocess.run(("git", "-C", root, "commit", "-qm", "fixture"), check=True)
    return root


def _stability(root: Path):
    clean, head, tree = runner.clean_repository_identity(root)
    captures = runner.capture_single_read_inputs({"input": root / "input.json"})
    return captures, runner.RunnerStability(tuple(captures.items()), clean, head, tree)


def test_fresh_process_guard_and_runtime_delegation(monkeypatch) -> None:
    with pytest.raises(runner.RunnerInputError, match="preloaded"):
        runner.require_fresh_execution_process()

    monkeypatch.setattr(runner.sys, "modules", {})
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    monkeypatch.setattr(runner.sys, "pycache_prefix", None)
    runner.require_fresh_execution_process()

    class Security:
        class SealInputError(ValueError):
            pass

        @staticmethod
        def _configure_and_capture_runtime():
            return {"execution_device": "cuda", "execution_dtype": "float64"}

    monkeypatch.setattr(runner, "_security_module", lambda: Security)
    assert runner.configure_and_capture_runtime() == {
        "execution_device": "cuda",
        "execution_dtype": "float64",
    }


def test_runner_root_anchor_and_module_binding_hook(
    monkeypatch, tmp_path: Path
) -> None:
    current_root = Path(runner.__file__).resolve().parents[2]
    runner.anchor_running_repository(current_root)
    with pytest.raises(runner.RunnerInputError, match="runner root differs"):
        runner.anchor_running_repository(tmp_path)

    class ExternalSpec:
        origin = str(tmp_path / "external" / "maple" / "__init__.py")

    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: ExternalSpec())
    with pytest.raises(runner.RunnerInputError, match="outside repo_root"):
        runner.anchor_running_repository(current_root)

    calls = []

    class Security:
        class SealInputError(ValueError):
            pass

        @staticmethod
        def _assert_loaded_repo_modules_bound(root, head, ledger):
            calls.append((root, head, ledger))

    monkeypatch.setattr(runner, "_security_module", lambda: Security)
    stability = runner.RunnerStability(
        (),
        current_root,
        "b" * 40,
        "c" * 40,
        (("runner.py", "d" * 64),),
    )
    stability.assert_modules_bound()
    stability.assert_modules_bound()
    assert len(calls) == 2
    assert calls[0][2] == {"runner.py": "d" * 64}


def test_direct_script_help_and_anchor_need_no_ambient_pythonpath() -> None:
    root = Path(runner.__file__).resolve().parents[2]
    script = Path(runner.__file__).resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    help_result = subprocess.run(
        (runner.sys.executable, str(script), "--help"),
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr

    code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(script)!r})
spec = importlib.util.spec_from_file_location('direct_h1_runner', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
root = pathlib.Path({str(root)!r})
sys.path[:] = [item for item in sys.path if pathlib.Path(item or '.').resolve() != root]
module.anchor_running_repository(root)
print(sys.path[0])
"""
    anchor_result = subprocess.run(
        (runner.sys.executable, "-c", code),
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert anchor_result.returncode == 0, anchor_result.stderr
    assert Path(anchor_result.stdout.strip()).resolve() == root


def test_security_loader_ignores_external_fake_tools_package(
    monkeypatch, tmp_path: Path
) -> None:
    fake = tmp_path / "fake"
    package = fake / "tools" / "route2_release"
    package.mkdir(parents=True)
    (fake / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "seal_mace_mdp_polar_harmonic_ef_revalidation.py").write_text(
        "EXTERNAL_FAKE = True\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(fake))
    monkeypatch.setattr(runner, "_SECURITY_MODULE", None)

    security = runner._security_module()

    assert not hasattr(security, "EXTERNAL_FAKE")
    assert Path(security.__file__).resolve() == (
        Path(runner.__file__).resolve().parent
        / "seal_mace_mdp_polar_harmonic_ef_revalidation.py"
    )


def test_security_loader_ignores_timestamp_valid_malicious_pyc(
    monkeypatch, tmp_path: Path
) -> None:
    tools = tmp_path / "tools" / "route2_release"
    tools.mkdir(parents=True)
    fake_runner = tools / "run_mace_mdp_polar_harmonic_ef_revalidation.py"
    fake_runner.write_text("# runner\n", encoding="utf-8")
    sealer = tools / "seal_mace_mdp_polar_harmonic_ef_revalidation.py"
    malicious = 'LOADED = "cached"\n'
    benign = 'LOADED = "source"\n'
    assert len(malicious) == len(benign)
    sealer.write_text(malicious, encoding="utf-8")
    value = sealer.stat()
    pyc = Path(py_compile.compile(str(sealer), doraise=True))
    assert pyc.exists()
    sealer.write_text(benign, encoding="utf-8")
    os.utime(sealer, ns=(value.st_atime_ns, value.st_mtime_ns))
    monkeypatch.setattr(runner, "__file__", str(fake_runner))
    monkeypatch.setattr(runner, "_SECURITY_MODULE", None)

    security = runner._security_module()

    assert security.LOADED == "source"
    assert security.__cached__ is None
    assert (
        security._captured_source_sha256 == hashlib.sha256(benign.encode()).hexdigest()
    )


def test_process_identity_allows_only_one_replicate_per_interpreter(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "_CLAIMED_PROCESS_IDENTITIES", set())
    runner.claim_fresh_process_identity(
        label="a",
        process_uuid="00000000-0000-0000-0000-000000000001",
        process_started_at_utc="2026-08-23T00:00:00+00:00",
    )
    with pytest.raises(runner.RunnerInputError, match="only one replicate"):
        runner.claim_fresh_process_identity(
            label="b",
            process_uuid="00000000-0000-0000-0000-000000000002",
            process_started_at_utc="2026-08-23T00:01:00+00:00",
        )


def test_single_read_capture_json_stability_and_alias_rejection(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    captures, stability = _stability(root)
    assert runner.read_captured_json(captures["input"], role="input") == {"value": 1}
    stability.assert_stable()

    (root / "input.json").write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(runner.RunnerInputError, match="changed|clean Git"):
        stability.assert_stable()

    original = tmp_path / "original"
    original.write_text("x", encoding="utf-8")
    hardlink = tmp_path / "hardlink"
    os.link(original, hardlink)
    with pytest.raises(runner.RunnerInputError, match="hard-linked"):
        runner.capture_single_read_inputs({"hardlink": hardlink})
    symlink = tmp_path / "symlink"
    symlink.symlink_to(original)
    with pytest.raises(runner.RunnerInputError, match="symbolic"):
        runner.capture_single_read_inputs({"symlink": symlink})


def test_captured_json_rejects_duplicates_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    capture = runner.capture_single_read_inputs({"duplicate": duplicate})["duplicate"]
    with pytest.raises(runner.RunnerInputError, match="duplicate"):
        runner.read_captured_json(capture, role="duplicate")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
    capture = runner.capture_single_read_inputs({"nonfinite": nonfinite})["nonfinite"]
    with pytest.raises(runner.RunnerInputError, match="non-finite"):
        runner.read_captured_json(capture, role="nonfinite")


def test_terminal_selection_uses_only_injected_shared_constructors() -> None:
    calls = []

    def candidate():
        calls.append("candidate")
        return {"kind": "candidate"}

    def failure(stage, error):
        calls.append((stage, error))
        return {"kind": "failure", "stage": stage}

    kind, payload = runner.select_terminal_artifact(
        terminal_active=True,
        candidate_factory=candidate,
        failure_factory=failure,
    )
    assert kind == "failure"
    assert payload == {"kind": "failure", "stage": "rich-v2-terminal"}
    assert calls == ["candidate", ("rich-v2-terminal", None)]

    kind, payload = runner.select_terminal_artifact(
        terminal_active=False,
        candidate_factory=candidate,
        failure_factory=failure,
    )
    assert kind == "success"
    assert payload == {"kind": "candidate"}

    def broken():
        raise RuntimeError("candidate failed")

    kind, payload = runner.select_terminal_artifact(
        terminal_active=False,
        candidate_factory=broken,
        failure_factory=failure,
    )
    assert kind == "failure"
    assert payload["stage"] == "candidate-validation"

    with pytest.raises(runner.RunnerInputError, match="distinct paths"):
        runner.orchestrate_terminal_publication(
            repo_root=Path("."),
            success_path="same.json",
            failure_path="same.json",
            terminal_active=True,
            candidate_factory=candidate,
            failure_factory=failure,
            stability=runner.RunnerStability(()),
        )


def test_exact_shared_candidate_and_failure_api_wiring(monkeypatch) -> None:
    class Seal:
        artifact_id = "seal-id"
        content_sha256 = "a" * 64
        git_head = "b" * 40
        git_tree = "c" * 40
        source_ledger_sha256 = "d" * 64
        asset_ledger_sha256 = "e" * 64
        runtime_fingerprint_sha256 = "f" * 64
        claim_boundary_id = "claim"
        non_admissions = ("one", "two")

    def canonical(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    class Typed:
        def __init__(self, payload):
            self.payload = payload

        @classmethod
        def from_mapping(cls, payload, *, seal):
            assert seal is Seal
            return cls(payload)

        def as_dict(self):
            return self.payload

    class Admission:
        REPLICATE_ADMISSION_RECORD_V2_SCHEMA = "replicate-v2"
        RICH_MEASUREMENT_SCHEMA_ID = "measurement-v2"
        EXECUTION_FAILURE_V2_SCHEMA = "failure-v2"
        ReplicateAdmissionRecordV2 = Typed
        ExecutionFailureV2 = Typed
        canonical_json_sha256 = staticmethod(canonical)

        @staticmethod
        def build_rich_harmonic_ef_measurement_candidate_v2(**kwargs):
            assert kwargs["seal"] is Seal
            return {"raw": kwargs["recording"]}, {"gate": True}

    monkeypatch.setattr(runner, "_admission_module", lambda: Admission)
    candidate = runner.build_typed_replicate_candidate(
        seal=Seal,
        label="a",
        process_uuid="00000000-0000-0000-0000-000000000001",
        process_started_at_utc="2026-08-23T00:00:00+00:00",
        prepared_inputs={"input": True},
        prepared_providers={"provider": True},
        recording={"events": []},
    )
    assert candidate["measurement"] == {"raw": {"events": []}}
    assert candidate["gate_results"] == {"gate": True}
    assert candidate["artifact_sha256"] == canonical(
        {key: value for key, value in candidate.items() if key != "artifact_sha256"}
    )

    failure = runner.build_typed_execution_failure(
        seal=Seal,
        label="b",
        process_uuid="00000000-0000-0000-0000-000000000002",
        process_started_at_utc="2026-08-23T00:01:00+00:00",
        stage="rich-v2-terminal",
        error=None,
        available_partial_evidence={"events": 141},
    )
    assert failure["schema_id"] == "failure-v2"
    assert failure["label"] == "b"
    assert failure["exception_message_truncated"] is False
    assert failure["exception_message_sha256"] == canonical(
        failure["exception_message"]
    )
    assert failure["capabilities"] == {
        "E": False,
        "F": False,
        "H": False,
        "V": False,
        "M": False,
    }


def test_execute_revalidation_switches_from_typed_failure_to_success(
    monkeypatch,
) -> None:
    class Seal:
        pass

    stability = runner.RunnerStability(())
    monkeypatch.setattr(runner, "_CLAIMED_PROCESS_IDENTITIES", set())
    monkeypatch.setattr(
        runner,
        "prepare_sealed_replay_inputs",
        lambda **kwargs: (Seal(), {}, stability, {"runtime": True}),
    )
    science = lambda **kwargs: (
        {"prepared": True},
        {"providers": True},
        {"solve_events": [1]},
    )
    failure_arguments = []
    monkeypatch.setattr(
        runner,
        "build_typed_execution_failure",
        lambda **kwargs: (
            failure_arguments.append(kwargs)
            or {"kind": "typed-failure", "stage": kwargs["stage"]}
        ),
    )

    def select_only(**kwargs):
        kind, payload = runner.select_terminal_artifact(
            terminal_active=kwargs["terminal_active"],
            candidate_factory=kwargs["candidate_factory"],
            failure_factory=kwargs["failure_factory"],
        )
        return kind, Path(f"/{kind}.json"), payload

    monkeypatch.setattr(runner, "orchestrate_terminal_publication", select_only)
    monkeypatch.setattr(
        runner,
        "build_typed_replicate_candidate",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("shared terminal")),
    )
    kind, _, payload = runner.execute_revalidation(
        repo_root="/repo",
        paths={},
        label="a",
        process_uuid="00000000-0000-0000-0000-000000000001",
        process_started_at_utc="2026-08-23T00:00:00+00:00",
        success_path="success.json",
        failure_path="failure.json",
        science_factory=science,
    )
    assert kind == "failure"
    assert payload["kind"] == "typed-failure"
    partial = failure_arguments[0]["available_partial_evidence"]
    assert "recording" not in partial
    assert "recording_summary" not in partial
    assert partial["solve_event_count"] == 1

    monkeypatch.setattr(runner, "_CLAIMED_PROCESS_IDENTITIES", set())
    monkeypatch.setattr(
        runner,
        "build_typed_replicate_candidate",
        lambda **kwargs: {"kind": "typed-success"},
    )
    kind, _, payload = runner.execute_revalidation(
        repo_root="/repo",
        paths={},
        label="b",
        process_uuid="00000000-0000-0000-0000-000000000002",
        process_started_at_utc="2026-08-23T00:01:00+00:00",
        success_path="success.json",
        failure_path="failure.json",
        science_factory=science,
    )
    assert kind == "success"
    assert payload["kind"] == "typed-success"

    monkeypatch.setattr(runner, "_CLAIMED_PROCESS_IDENTITIES", set())

    def failed_science(**kwargs):
        raise OSError("memfd unavailable")

    kind, _, payload = runner.execute_revalidation(
        repo_root="/repo",
        paths={},
        label="a",
        process_uuid="00000000-0000-0000-0000-000000000003",
        process_started_at_utc="2026-08-23T00:02:00+00:00",
        success_path="success.json",
        failure_path="failure.json",
        science_factory=failed_science,
    )
    assert kind == "failure"
    assert payload["stage"] == "science-execution"


def test_execute_rejects_output_alias_before_preparation(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(
        runner,
        "prepare_sealed_replay_inputs",
        lambda **kwargs: called.append(kwargs),
    )
    with pytest.raises(runner.RunnerInputError, match="distinct paths"):
        runner.execute_revalidation(
            repo_root="/repo",
            paths={},
            label="a",
            process_uuid="00000000-0000-0000-0000-000000000001",
            process_started_at_utc="2026-08-23T00:00:00+00:00",
            success_path="same.json",
            failure_path="same.json",
        )
    assert called == []


def test_runtime_configuration_precedes_admission_import() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    function = source[source.index("def prepare_sealed_replay_inputs") :]
    assert function.index("runtime = configure_and_capture_runtime()") < function.index(
        "admission = _admission_module()"
    )
    candidate = source[source.index("def execute_revalidation") :]
    assert candidate.index("science_factory(") < candidate.index(
        "stability.assert_modules_bound()"
    )
    assert 'checkpoint_bytes=captures["mace_mdp_checkpoint"].data' in source
    assert 'polar_kwargs["checkpoint_bytes"]' in source
    assert 'polar_kwargs["checkpoint_path"]' not in source
    assert "source_ledger.get(security_relative)" in source
    assert "security._captured_source_sha256" in source


def test_failure_message_is_truncated_but_full_digest_is_retained(monkeypatch) -> None:
    full_message = "x" * 5000

    class Seal:
        artifact_id = "seal"
        content_sha256 = "a" * 64
        claim_boundary_id = "claim"
        non_admissions = ("one",)

    def canonical(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    class Typed:
        def __init__(self, payload):
            self.payload = payload

        @classmethod
        def from_mapping(cls, payload, *, seal):
            return cls(payload)

        def as_dict(self):
            return self.payload

    class Admission:
        EXECUTION_FAILURE_V2_SCHEMA = "failure-v2"
        ExecutionFailureV2 = Typed
        canonical_json_sha256 = staticmethod(canonical)

    monkeypatch.setattr(runner, "_admission_module", lambda: Admission)
    failure = runner.build_typed_execution_failure(
        seal=Seal,
        label="a",
        process_uuid="00000000-0000-0000-0000-000000000001",
        process_started_at_utc="2026-08-23T00:00:00+00:00",
        stage="science-execution",
        error=RuntimeError(full_message),
        available_partial_evidence={"bounded": True},
    )
    assert len(failure["exception_message"]) == 4096
    assert failure["exception_message_truncated"] is True
    assert failure["exception_message_sha256"] == canonical(full_message)


def test_atomic_terminal_publication_switch_no_replace_and_no_repo_mutation(
    tmp_path: Path,
) -> None:
    root = _git_repo(tmp_path)
    _, stability = _stability(root)
    external = tmp_path / "external"
    external.mkdir()
    success = external / "success.json"
    failure = external / "failure.json"
    before = subprocess.check_output(("git", "-C", root, "status", "--porcelain"))

    def candidate():
        return {"kind": "candidate"}

    def failure_factory(stage, error):
        return {
            "kind": "failure",
            "stage": stage,
            "error": None if error is None else str(error),
        }

    try:
        kind, published, payload = runner.orchestrate_terminal_publication(
            repo_root=root,
            success_path=success,
            failure_path=failure,
            terminal_active=True,
            candidate_factory=candidate,
            failure_factory=failure_factory,
            stability=stability,
        )
    except runner.RunnerInputError as error:
        if "O_TMPFILE is unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert kind == "failure"
    assert published == failure
    assert json.loads(failure.read_text()) == payload
    assert not success.exists()
    assert (
        subprocess.check_output(("git", "-C", root, "status", "--porcelain")) == before
    )

    _, success_stability = _stability(root)
    kind, published, _ = runner.orchestrate_terminal_publication(
        repo_root=root,
        success_path=success,
        failure_path=external / "unused-failure.json",
        terminal_active=False,
        candidate_factory=candidate,
        failure_factory=failure_factory,
        stability=success_stability,
    )
    assert kind == "success"
    assert published == success
    with pytest.raises(runner.RunnerInputError, match="already exists"):
        runner.publish_external_artifact(
            repo_root=root,
            output_path=success,
            payload={"again": True},
            stability=success_stability,
        )


def test_publication_rejects_inside_repo_symlink_and_commit_race(
    tmp_path: Path, monkeypatch
) -> None:
    root = _git_repo(tmp_path)
    _, stability = _stability(root)
    with pytest.raises(runner.RunnerInputError, match="external"):
        runner.publish_external_artifact(
            repo_root=root,
            output_path=root / "inside.json",
            payload={"x": 1},
            stability=stability,
        )

    external = tmp_path / "external"
    external.mkdir()
    target = external / "target.json"
    link = external / "link.json"
    link.symlink_to(target)
    with pytest.raises(runner.RunnerInputError, match="symbolic"):
        runner.publish_external_artifact(
            repo_root=root,
            output_path=link,
            payload={"x": 1},
            stability=stability,
        )

    security = runner._security_module()
    race = external / "race.json"
    monkeypatch.setattr(
        security,
        "_before_publish",
        lambda: race.write_text("racer", encoding="utf-8"),
    )
    with pytest.raises(runner.RunnerInputError, match="appeared"):
        runner.publish_external_artifact(
            repo_root=root,
            output_path=race,
            payload={"x": 1},
            stability=stability,
        )
    assert race.read_text(encoding="utf-8") == "racer"

    retarget_parent = tmp_path / "runner-parent"
    retarget_parent.mkdir()
    moved_parent = tmp_path / "runner-parent-moved"
    retarget_output = retarget_parent / "artifact.json"

    def retarget() -> None:
        retarget_parent.rename(moved_parent)
        retarget_parent.symlink_to(moved_parent, target_is_directory=True)

    monkeypatch.setattr(security, "_before_publish", retarget)
    with pytest.raises(runner.RunnerInputError, match="symbolic|changed"):
        runner.publish_external_artifact(
            repo_root=root,
            output_path=retarget_output,
            payload={"x": 1},
            stability=stability,
        )
    assert not (moved_parent / "artifact.json").exists()


def test_cli_wires_exact_paths_and_reports_selected_artifact(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(runner, "require_fresh_execution_process", lambda: None)
    observed = {}

    def execute(**kwargs):
        observed.update(kwargs)
        return (
            "failure",
            tmp_path / "failure.json",
            {"artifact_id": "failure-id", "artifact_sha256": "a" * 64},
        )

    monkeypatch.setattr(runner, "execute_revalidation", execute)
    common = tmp_path / "input"
    arguments = [
        "--repo-root",
        str(tmp_path),
        "--seal",
        str(common),
        "--preregistration",
        str(common),
        "--v1-preregistration",
        str(common),
        "--parent-panel",
        str(common),
        "--benzene-mol2",
        str(common),
        "--benzene-projection",
        str(common),
        "--mace-mdp-checkpoint",
        str(common),
        "--mace-polar-checkpoint",
        str(common),
        "--label",
        "a",
        "--success-output",
        str(tmp_path / "success.json"),
        "--failure-output",
        str(tmp_path / "failure.json"),
    ]
    assert runner.main(arguments) == 2
    assert observed["label"] == "a"
    assert set(observed["paths"]) == {
        "seal",
        "preregistration",
        "v1_preregistration",
        "parent_panel",
        "benzene_mol2",
        "benzene_projection",
        "mace_mdp_checkpoint",
        "mace_polar_checkpoint",
        "runner",
    }
    output = capsys.readouterr().out
    assert "ROUTE2_HARMONIC_EF_REVALIDATION=" in output
    assert '"kind": "failure"' in output


@pytest.mark.parametrize(
    "message", ("nvidia-smi failed", "O_TMPFILE failed", "memfd failed")
)
def test_cli_converts_environment_oserrors_to_controlled_exit_two(
    message, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        runner,
        "require_fresh_execution_process",
        lambda: (_ for _ in ()).throw(OSError(message)),
    )
    assert runner.main([]) == 2
    assert message in capsys.readouterr().err
